#!/usr/bin/env node
// Read micro-TLB (set_jit_config 29/30, default OFF) must never cache what
// safe_read*_slow_jit returned: for a page-crossing or MMIO read that answer is a one-shot
// alias into jit_paging_scratch_buffer. Caching it makes a LATER same-page read serve
// uninitialised scratch — and, for MMIO, perform no device access at all.
//
// The guest loops three reads of the SAME page in one basic block: one crossing into the next
// page (slow helper), then two that do not. The crossing read must not poison the pair behind
// it. Instrumented: the census counters prove the lever was actually engaged, so a green run
// cannot mean "the cache never ran".
//
//   node tests/jit-read-tlb-cache-repro.mjs   [V86_WASM_PATH=<candidate v86.wasm>]

const { V86 } = await import("../build/libv86.mjs");

const BASE = 0x100000, ENTRY = BASE + 0x40, LOOP = BASE + 0x1000;
const PD_ADDR = 0x108000, PT0_ADDR = 0x109000;
const DP = 0x200000;               // data page under test
const CROSS = DP + 0xFFE;          // 4 bytes spanning DP and DP + 0x1000
const SAME = DP + 0x10;            // filled into the micro-TLB by the read before it
const SAME2 = DP + 0x20;           // served FROM the micro-TLB
const RES = 0x1F0000;
const CROSS_VALUE = 0x11223344, SAME_VALUE = 0xAABBCCDD, SAME2_VALUE = 0x55667788;
const ITER = 100000;
const MEM_SIZE = 16 * 1024 * 1024, TIMEOUT_MS = 30000;

function build_image()
{
    const size = 0x1100;
    const buf = new Uint8Array(size);
    const dv = new DataView(buf.buffer);
    const MAGIC = 0x1BADB002, FLAGS = 0x10000;
    dv.setUint32(0x00, MAGIC, true);
    dv.setUint32(0x04, FLAGS, true);
    dv.setUint32(0x08, (-(MAGIC + FLAGS)) >>> 0, true);
    dv.setUint32(0x0c, BASE, true);
    dv.setUint32(0x10, BASE, true);
    dv.setUint32(0x14, BASE + size, true);
    dv.setUint32(0x18, BASE + 0x4000, true);
    dv.setUint32(0x1c, ENTRY, true);

    let o = ENTRY - BASE;
    const emit = (...bytes) => { for(const b of bytes) buf[o++] = b & 0xFF; };
    const u32 = value => { dv.setUint32(o, value >>> 0, true); o += 4; };

    emit(0xFC);                                    // cld
    emit(0xBF); u32(PT0_ADDR);                     // mov edi,PT0
    emit(0xB8); u32(0x00000003);                   // mov eax,0|P|RW
    emit(0xB9); u32(0x400);                        // mov ecx,1024
    const fill = o;
    emit(0xAB);                                    // stosd
    emit(0x05); u32(0x1000);                       // add eax,0x1000
    emit(0xE2, (fill - (o + 2)) & 0xFF);           // loop fill
    emit(0xBF); u32(PD_ADDR);                      // mov edi,PD
    emit(0x31, 0xC0);                              // xor eax,eax
    emit(0xB9); u32(0x400);                        // mov ecx,1024
    emit(0xF3, 0xAB);                              // rep stosd
    emit(0xC7, 0x05); u32(PD_ADDR); u32((PT0_ADDR | 3) >>> 0);
    emit(0xB8); u32(PD_ADDR);                      // mov eax,PD
    emit(0x0F, 0x22, 0xD8);                        // mov cr3,eax
    emit(0x0F, 0x20, 0xC0);                        // mov eax,cr0
    emit(0x0D); u32(0x80000000);                   // or eax,0x80000000
    emit(0x0F, 0x22, 0xC0);                        // mov cr0,eax
    emit(0xBC); u32(0x190000);                     // mov esp,...
    emit(0xE9); dv.setInt32(o, LOOP - (BASE + o + 4), true); o += 4; // jmp LOOP

    // The read trio lives alone on its own code page — that page index is what
    // set_jit_config(30) arms, so nothing else in the image can be affected. It must also be
    // ONE basic block: the micro-TLB locals are created (invalid) per block, so a hit is only
    // observable from a third read behind the fill.
    o = LOOP - BASE;
    emit(0xB9); u32(ITER);                         // mov ecx,ITER
    const top = o;
    emit(0x8B, 0x05); u32(CROSS);                  // mov eax,[CROSS]   -> slow helper
    emit(0x8B, 0x1D); u32(SAME);                   // mov ebx,[SAME]    -> must NOT hit a cached alias
    emit(0x8B, 0x35); u32(SAME2);                  // mov esi,[SAME2]   -> micro-TLB hit
    emit(0x49);                                    // dec ecx
    emit(0x0F, 0x85); dv.setInt32(o, top - (o + 4), true); o += 4;   // jnz top
    emit(0x89, 0x05); u32(RES);                    // mov [RES],eax
    emit(0x89, 0x1D); u32(RES + 4);                // mov [RES+4],ebx
    emit(0x89, 0x35); u32(RES + 8);                // mov [RES+8],esi
    emit(0xF4, 0xEB, 0xFE);                        // hlt; jmp $
    return buf;
}

const fail = message => { throw new Error(message); };

function run()
{
    return new Promise((resolve, reject) => {
        const emulator = new V86({
            autostart: false, memory_size: MEM_SIZE, disable_jit: 0, log_level: 0,
            wasm_path: process.env.V86_WASM_PATH || undefined,
        });
        let timer;
        emulator.bus.register("cpu-event-halt", () => {
            clearTimeout(timer);
            try {
                const cpu = emulator.v86.cpu;
                const ex = cpu.wm.exports;
                const dv = new DataView(cpu.mem8.buffer, cpu.mem8.byteOffset, cpu.mem8.byteLength);
                resolve({
                    paging: (cpu.cr[0] >>> 0) & 0x80000000 ? 1 : 0,
                    cross: dv.getUint32(RES, true),
                    same: dv.getUint32(RES + 4, true),
                    same2: dv.getUint32(RES + 8, true),
                    cache_hits: ex.profiler_dispatch_stat_get(23),
                    cache_fills: ex.profiler_dispatch_stat_get(24),
                });
            }
            catch(error) { reject(error); }
            finally { try { emulator.stop(); } catch {} }
        });
        emulator.add_listener("emulator-loaded", () => {
            try {
                const cpu = emulator.v86.cpu;
                const ex = cpu.wm.exports;
                cpu.reboot_internal();
                cpu.reset_memory();
                const dv = new DataView(cpu.mem8.buffer, cpu.mem8.byteOffset, cpu.mem8.byteLength);
                dv.setUint32(SAME, SAME_VALUE, true);
                dv.setUint32(SAME2, SAME2_VALUE, true);
                dv.setUint32(CROSS, CROSS_VALUE, true);

                ex.set_jit_config(30, LOOP >>> 12);
                ex.set_jit_config(29, 2);   // 2 = enabled + census
                cpu.jit_clear_cache?.();
                cpu.load_multiboot(build_image().buffer);
                timer = setTimeout(() => reject(new Error("guest did not halt")), TIMEOUT_MS);
                emulator.run();
            }
            catch(error) { reject(error); }
        });
    });
}

const r = await run();
console.log("jit-read-tlb-cache " + JSON.stringify(r));
if(!r.paging) fail("guest did not enable paging");
if(r.cache_fills === 0) fail("read micro-TLB never filled — the lever was not engaged");
if(r.cache_hits === 0) fail("read micro-TLB never hit — the test proves nothing");
if(r.cross !== CROSS_VALUE) fail(`page-crossing read returned 0x${r.cross.toString(16)}`);
if(r.same !== SAME_VALUE) fail(`same-page read after a slow-helper read returned 0x${r.same.toString(16)} (scratch-buffer alias was cached)`);
if(r.same2 !== SAME2_VALUE) fail(`micro-TLB-served read returned 0x${r.same2.toString(16)}`);
console.log("  ok — the slow helper's scratch alias never reaches the read micro-TLB");
