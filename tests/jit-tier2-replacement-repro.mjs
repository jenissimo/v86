#!/usr/bin/env node
// Retired-instruction Tier-2 and its bounded replacement policy must both run.
//
// The image visits 260 distinct 4 KiB pages. Every page contains a long local loop,
// enough to cross the ordinary JIT threshold and then the much lower test-only Tier-2
// retired threshold. TIER2_MAX_PAGES=1 keeps admissions page-granular. The active set
// can hold 256 pages, so a successful run must evict at least four rather than permanently
// refusing every page that appeared after the boot-time set filled.

const { V86 } = await import("../build/libv86.mjs");

const BASE = 0x100000;
const ENTRY_OFF = 0x20;
const PAGES = 260;
const ITER_PER_PAGE = 900000;
const MEM_SIZE = 32 * 1024 * 1024;
const TIMEOUT_MS = 120000;

function build_image()
{
    // One extra non-hot trailer page gives cycle_internal a safe point to drain the
    // final hot page's pending promotion before HLT ends the run.
    const buf = new Uint8Array((PAGES + 1) * 0x1000 + 16);
    const dv = new DataView(buf.buffer);
    const MAGIC = 0x1BADB002, FLAGS = 0x10000;
    dv.setUint32(0x00, MAGIC, true);
    dv.setUint32(0x04, FLAGS, true);
    dv.setUint32(0x08, (-(MAGIC + FLAGS)) >>> 0, true);
    dv.setUint32(0x0c, BASE, true);
    dv.setUint32(0x10, BASE, true);
    dv.setUint32(0x14, BASE + buf.length, true);
    dv.setUint32(0x18, BASE + buf.length + 0x4000, true);
    dv.setUint32(0x1c, BASE + ENTRY_OFF, true);

    const emit = (at, ...bytes) => { for(const b of bytes) buf[at++] = b & 0xff; return at; };
    const put32 = (at, value) => { dv.setUint32(at, value >>> 0, true); return at + 4; };
    for(let page = 0; page < PAGES; page++)
    {
        let o = page === 0 ? ENTRY_OFF : page * 0x1000;
        if(page === 0)
        {
            o = emit(o, 0xBC); o = put32(o, BASE + buf.length + 0x3000); // mov esp,stack
            o = emit(o, 0x31, 0xC0);                                   // xor eax,eax
        }
        o = emit(o, 0xB9); o = put32(o, ITER_PER_PAGE);                 // mov ecx,N
        const loop = o;
        o = emit(o, 0x49);                                              // dec ecx
        o = emit(o, 0x0F, 0x85);                                        // jnz rel32
        o = put32(o, loop - (o + 4));
        o = emit(o, 0x05); o = put32(o, page + 1);                      // add eax,page+1
        o = emit(o, 0xE9);                                             // jmp next/trailer page
        const next = (page + 1) * 0x1000;
        o = put32(o, next - (o + 4));
    }
    let o = PAGES * 0x1000;
    o = emit(o, 0xB9); o = put32(o, 150000);                            // below JIT threshold
    const loop = o;
    o = emit(o, 0x49, 0x0F, 0x85); o = put32(o, loop - (o + 4));
    emit(o, 0xF4, 0xEB, 0xFE);                                         // hlt; jmp $
    return buf;
}

function run()
{
    return new Promise(resolve => {
        const emulator = new V86({ autostart: false, memory_size: MEM_SIZE, disable_jit: 0, log_level: 0 });
        let timer, done = false;
        const finish = status => {
            if(done) return;
            done = true;
            clearTimeout(timer);
            try { emulator.stop(); } catch {}
            const cpu = emulator.v86.cpu;
            const w = cpu.wm.exports;
            const result = {
                status,
                eax: cpu.reg32[0] >>> 0,
                pages: w.jit_get_tier2_page_count?.() ?? 0,
                promotions: w.jit_get_tier2_promotions?.() ?? 0,
                evictions: w.jit_get_tier2_evictions?.() ?? 0,
                blocked: w.jit_get_tier2_blocked_by_cap?.() ?? 0,
                blockedDistinct: w.jit_get_tier2_blocked_distinct?.() ?? 0,
                afterDisablePages: -1,
                afterDisableCache: -1,
            };
            if(status === "halt") {
                cpu.set_jit_config(15, 0);
                result.afterDisablePages = w.jit_get_tier2_page_count?.() ?? -1;
                result.afterDisableCache = w.jit_get_cache_size?.() ?? -1;
            }
            resolve(result);
        };
        emulator.bus.register("cpu-event-halt", () => finish("halt"));
        emulator.add_listener("emulator-loaded", () => {
            const cpu = emulator.v86.cpu;
            cpu.reboot_internal();
            cpu.reset_memory();
            cpu.set_jit_config(1, 1);       // baseline module max pages
            // Loop safety exits near 100K retired instructions, so this produces one
            // promotion credit per activation. Late pages must survive eight distinct
            // probation deferrals rather than arriving with one saturated credit.
            cpu.set_jit_config(15, 100000); // retired instruction threshold
            cpu.set_jit_config(17, 1);      // tier-2 module max pages
            cpu.jit_clear_cache?.();
            cpu.load_multiboot(build_image().buffer);
            timer = setTimeout(() => finish("HANG"), TIMEOUT_MS);
            emulator.run();
        });
    });
}

const r = await run();
console.log("jit-tier2-replacement " + JSON.stringify(r));
const fail = msg => { console.error("FAIL: " + msg); process.exit(1); };
const expected = PAGES * (PAGES + 1) / 2;
if(r.status !== "halt" || r.eax !== expected) fail(`program did not finish exactly (eax=${r.eax}, expected=${expected})`);
if(r.promotions < PAGES) fail(`only ${r.promotions}/${PAGES} pages promoted`);
if(r.pages !== 256) fail(`active tier-2 set is ${r.pages}, expected 256`);
if(r.evictions < PAGES - 256) fail(`only ${r.evictions} evictions, expected at least ${PAGES - 256}`);
if(r.blocked < (PAGES - 256) * 7) fail(`probation was bypassed (blocked=${r.blocked}, expected >= ${(PAGES - 256) * 7})`);
if(r.blockedDistinct !== 0) fail(`admitted pages remained marked probationary (${r.blockedDistinct})`);
if(r.afterDisablePages !== 0 || r.afterDisableCache !== 0)
    fail(`Tier-2 OFF did not clear state (pages=${r.afterDisablePages}, cache=${r.afterDisableCache})`);
console.log(`  ok — promotions=${r.promotions} pages=${r.pages} evictions=${r.evictions} blocked=${r.blocked} offPages=${r.afterDisablePages}`);
process.exit(0);
