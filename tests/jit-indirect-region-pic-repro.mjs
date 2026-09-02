#!/usr/bin/env node
// Profile-guided indirect-region PIC repro.
//
// Two identity-mapped code pages execute an indirect CALL/RET loop:
//   page0: call edx -> page1; increment counter; stop-flag check; loop
//   page1: ret -> page0
//
// MAX_PAGES=1 first forces separate modules while trace2 records both dynamic
// targets. The test then keeps those histograms, enables indirect regions with a
// two-page budget, clears the JIT cache, and lets the hot pair recompile into one
// module. The module-local PIC must serve both edges without calling
// jit_find_cache_entry_in_page on the steady-state path.

const { V86 } = await import("../build/libv86.mjs");

const BASE = 0x100000;
const ENTRY_OFF = 0x20;
const PAGE1_OFF = 0x1000;
const STOP = BASE + 0x2000;
const COUNTER = STOP + 4;
const MEM_SIZE = 16 * 1024 * 1024;
const PROFILE_MS = 300;
const REGION_WARM_MS = 400;
const MEASURE_MS = 300;
const TIMEOUT_MS = 8000;

function build_image()
{
    const buf = new Uint8Array(PAGE1_OFF + 16);
    const dv = new DataView(buf.buffer);
    const MAGIC = 0x1BADB002, FLAGS = 0x10000;
    dv.setUint32(0x00, MAGIC, true);
    dv.setUint32(0x04, FLAGS, true);
    dv.setUint32(0x08, (-(MAGIC + FLAGS)) >>> 0, true);
    dv.setUint32(0x0c, BASE, true);
    dv.setUint32(0x10, BASE, true);
    dv.setUint32(0x14, BASE + PAGE1_OFF + 16, true);
    dv.setUint32(0x18, BASE + 0x5000, true);
    dv.setUint32(0x1c, BASE + ENTRY_OFF, true);

    let o = ENTRY_OFF;
    const labels = {}, patches = [];
    const label = name => { labels[name] = o; };
    const emit = (...bytes) => { for(const b of bytes) buf[o++] = b & 0xff; };
    const u32 = value => { dv.setUint32(o, value >>> 0, true); o += 4; };
    const rel32 = name => {
        patches.push({ at: o, end: o + 4, to: name });
        emit(0, 0, 0, 0);
    };

    emit(0xBC); u32(BASE + 0x4800);       // mov esp, stack
    emit(0xBA); u32(BASE + PAGE1_OFF);   // mov edx, callee
    label("loop");
    emit(0xFF, 0xD2);                    // call edx
    emit(0xFF, 0x05); u32(COUNTER);      // inc dword [COUNTER]
    emit(0x83, 0x3D); u32(STOP); emit(0);// cmp dword [STOP], 0
    emit(0x0F, 0x84); rel32("loop");    // je loop
    emit(0xF4, 0xEB, 0xFE);             // hlt; jmp $

    o = PAGE1_OFF;
    label("callee");
    emit(0xC3);                          // ret

    for(const p of patches) dv.setInt32(p.at, labels[p.to] - p.end, true);
    return buf;
}

function run()
{
    return new Promise(resolve => {
        const emulator = new V86({
            autostart: false,
            memory_size: MEM_SIZE,
            disable_jit: 0,
            log_level: 0,
        });
        const timers = [];
        let done = false;
        let phase = "boot";
        let profilePairs = 0;
        let counterStart = 0;
        let counterEnd = 0;
        let abseipDispatch = 0;
        let maxRegionPages = 0;

        const later = (fn, ms) => {
            const timer = setTimeout(fn, ms);
            timers.push(timer);
            return timer;
        };
        const read32 = (cpu, addr) =>
            new DataView(cpu.mem8.buffer, cpu.mem8.byteOffset).getUint32(addr, true);
        const write32 = (cpu, addr, value) =>
            new DataView(cpu.mem8.buffer, cpu.mem8.byteOffset).setUint32(addr, value >>> 0, true);
        const finish = status => {
            if(done) return;
            done = true;
            for(const timer of timers) clearTimeout(timer);
            try { emulator.stop(); } catch(e) {}
            const cpu = emulator.v86.cpu;
            resolve({
                status,
                phase,
                profilePairs,
                iterations: counterEnd - counterStart >>> 0,
                abseipDispatch,
                maxRegionPages,
                regions: cpu.get_jit_config ? cpu.get_jit_config(6) >>> 0 : 0,
                finalCounter: read32(cpu, COUNTER),
            });
        };

        emulator.bus.register("cpu-event-halt", () => finish("halt"));
        emulator.add_listener("emulator-loaded", () => {
            const cpu = emulator.v86.cpu;
            const w = cpu.wm.exports;
            cpu.reboot_internal();
            cpu.reset_memory();
            cpu.set_jit_config(1, 1);  // baseline: force the two pages apart
            cpu.set_jit_config(12, 1); // preserve the production dynamic-chain fallback
            cpu.set_jit_config(13, 0); // isolate this repro from static RET speculation
            cpu.set_jit_config(15, 0); // production Tier2 OFF
            w.set_dispatch_stats?.(1);
            w.profiler_init?.();
            w.trace2_watch_page?.(BASE);
            w.trace2_watch_page?.(BASE + PAGE1_OFF);
            cpu.jit_clear_cache?.();
            cpu.load_multiboot(build_image().buffer);
            phase = "profile";
            emulator.run();

            later(() => {
                profilePairs = w.trace2_indirect_snapshot?.() >>> 0;
                w.trace2_unwatch_all?.(); // keep the target histograms
                cpu.set_jit_config(6, 1);
                cpu.set_jit_config(7, 1);
                cpu.set_jit_config(8, 2);
                cpu.jit_clear_cache?.();
                phase = "region-warm";

                later(() => {
                    maxRegionPages = w.jit_debug_max_region_pages?.() >>> 0;
                    w.profiler_init?.();
                    counterStart = read32(cpu, COUNTER);
                    phase = "measure";

                    later(() => {
                        counterEnd = read32(cpu, COUNTER);
                        abseipDispatch = Number(w.profiler_dispatch_stat_get?.(10) ?? 0);
                        phase = "stop";
                        write32(cpu, STOP, 1);
                    }, MEASURE_MS);
                }, REGION_WARM_MS);
            }, PROFILE_MS);

            later(() => finish("TIMEOUT"), TIMEOUT_MS);
        });
    });
}

const r = await run();
console.log("jit-indirect-region-pic " + JSON.stringify(r));

const fail = message => { console.error("FAIL: " + message); process.exit(1); };
if(r.status !== "halt") fail(`guest did not halt cleanly (status=${r.status}, phase=${r.phase})`);
if(r.profilePairs < 2) fail(`trace2 recorded only ${r.profilePairs} indirect target pair(s)`);
if(!r.regions) fail("JIT_INDIRECT_REGIONS did not stay enabled");
if(r.maxRegionPages < 2) fail(`region formation never joined both pages (max=${r.maxRegionPages})`);
if(r.iterations < 1000) fail(`guest made too little measured progress (${r.iterations} iterations)`);
// The fixed loop has two AbsoluteEip edges per iteration. Without the inline PIC,
// ABSEIP_DISPATCH is approximately 2*iterations; allow a tiny amount of asynchronous
// compile/warm-up residue while still making the discriminator unambiguous.
if(r.abseipDispatch * 1000 >= r.iterations)
{
    fail(`module-local PIC did not cover steady-state indirects: ` +
         `abseip=${r.abseipDispatch}, iterations=${r.iterations}`);
}

console.log(`  ok — profilePairs=${r.profilePairs} regionPages=${r.maxRegionPages} ` +
            `iterations=${r.iterations} abseip=${r.abseipDispatch}`);
process.exit(0);
