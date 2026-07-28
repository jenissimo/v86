#!/usr/bin/env node
// A ZEROED dispatch-slab cell must be a MISS, never a live dispatch.
//
// Why this exists: DISPATCH_SLABS cells used to store the entry-block index raw, with
// u16::MAX as "no entry here". Zero was therefore BOTH "block 0 of this module" and the
// value that any memory corruption (or an unpublished cell) leaves behind — so a single
// stray zero turned into a dispatch INTO a live module at the wrong entry point, with
// whatever registers the current call site happened to hold. That is silent wrong-code
// execution, observed in House of 1000 Doors / Blade of Darkness as an AV with several
// registers sharing one garbage value. Cells now hold `state + 1`, making 0 a structural
// miss (interpret + recompile: slower, never wrong).
//
// The test injects the corruption directly — it zeroes the slab cells of a page that has
// a live compiled module while the guest keeps running — and then asserts the guest still
// computes the right answer. On the old encoding the guest re-enters at block 0 and the
// checksum diverges (or it faults/hangs); on the new one the zeroed cells simply miss.
//
// VERIFIED RED against the raw encoding (dispatch_state_lookup returning the cell as-is
// and dispatch_meta_set filling u16::MAX): the run diverges from the interpreter.
//
//   node tests/jit-slab-zero-sentinel.mjs

const { V86 } = await import("../build/libv86.mjs");

const BASE = 0x100000;
const CHK = BASE + 0x8000;
const ENTRY_OFF = 0x20;
const OUTER = 3000000;
const MEM_SIZE = 16 * 1024 * 1024;
const TIMEOUT_MS = 30000;

// Two-page workload: an outer loop on page 0 calling a small routine on page 1 that
// folds a counter into a checksum. Several entry points per page, so a zeroed slab has
// something to dispatch wrongly INTO.
function buildImage() {
    const buf = new Uint8Array(0x9000);
    const dv = new DataView(buf.buffer);
    const MAGIC = 0x1BADB002, FLAGS = 0x10000;
    dv.setUint32(0x00, MAGIC, true);
    dv.setUint32(0x04, FLAGS, true);
    dv.setUint32(0x08, (-(MAGIC + FLAGS)) >>> 0, true);
    dv.setUint32(0x0c, BASE, true);
    dv.setUint32(0x10, BASE, true);
    dv.setUint32(0x14, BASE + buf.length, true);
    dv.setUint32(0x18, BASE + buf.length, true);
    dv.setUint32(0x1c, BASE + ENTRY_OFF, true);

    let o = ENTRY_OFF;
    const emit = (...b) => { for (const x of b) buf[o++] = x & 0xff; };
    const u32 = (v) => { dv.setUint32(o, v >>> 0, true); o += 4; };

    emit(0xBC); u32(BASE + 0x7F00);          // mov esp, stack
    emit(0xBF); u32(OUTER);                  // mov edi, OUTER
    emit(0x31, 0xD2);                        // xor edx, edx
    const outer = o;
    emit(0xE8); u32(0x1000 - (o + 4));       // call f1  (page 1, entry A)
    emit(0xE8); u32(0x1040 - (o + 4));       // call f2  (page 1, entry B)
    emit(0x31, 0x15); u32(CHK);              // xor [chk], edx
    emit(0x83, 0xFA, 0x00);                  // cmp edx, 0 (define flags)
    emit(0x4F);                              // dec edi
    emit(0x0F, 0x85); u32((outer - (o + 4)) >>> 0);  // jnz outer
    emit(0xF4); emit(0xEB, 0xFE);            // hlt; jmp $

    // Page 1 holds TWO independent entry points doing DIFFERENT arithmetic, each
    // reached only by a cross-page CALL � so each call is a real dispatch, and
    // entering B's address at A's block is observable in the checksum. (MAX_PAGES=1
    // keeps page 1 in its own module, so the calls cannot be inlined into the caller.)
    o = 0x1000;                              // f1 � entry A (block 0 of the module)
    emit(0x42);                              // inc edx
    emit(0xD1, 0xC2);                        // rol edx, 1
    emit(0xC3);                              // ret
    o = 0x1040;                              // f2 � entry B (a later block)
    emit(0x83, 0xF2, 0x5A);                  // xor edx, 0x5A
    emit(0x83, 0xC2, 0x07);                  // add edx, 7
    emit(0xC3);                              // ret
    return buf;
}

function run({ corrupt }) {
    return new Promise((resolve) => {
        const emulator = new V86({ autostart: false, memory_size: MEM_SIZE,
                                   disable_jit: corrupt === null ? 1 : 0, log_level: 0 });
        let halted = false, timer, poker = null, zeroed = 0;
        const finish = (status) => {
            clearTimeout(timer);
            if (poker) clearInterval(poker);
            try { emulator.stop(); } catch (e) {}
            const cpu = emulator.v86.cpu;
            const chk = new DataView(cpu.mem8.buffer, cpu.mem8.byteOffset).getUint32(CHK, true);
            resolve({ status, chk, edx: cpu.reg32[2] >>> 0, zeroed });
        };
        emulator.bus.register("cpu-event-halt", () => { halted = true; finish("halt"); });
        emulator.add_listener("emulator-loaded", () => {
            const cpu = emulator.v86.cpu;
            cpu.reboot_internal();
            cpu.reset_memory();
            if (corrupt !== null) {
                cpu.set_jit_config(1, 1);   // MAX_PAGES=1: page 1 is its own module, so calls dispatch
                cpu.set_jit_config(12, 1);
                cpu.set_jit_config(13, 1);
                cpu.jit_clear_cache?.();
            }
            cpu.load_multiboot(buildImage().buffer);
            timer = setTimeout(() => { if (!halted) finish("HANG"); }, TIMEOUT_MS);

            if (corrupt) {
                const ex = cpu.wm.exports;
                if (!ex.jit_get_dispatch_slabs_ptr || !ex.jit_debug_meta_lo) {
                    console.error("FAIL: build lacks the slab readback exports"); process.exit(1);
                }
                const poolBase = ex.jit_get_dispatch_slabs_ptr() >>> 0;
                // Emulate the stomp with ONE targeted cell: entry B's. Zeroing the whole
                // slab would be indistinguishable from "everything dispatches to block 0",
                // which is CORRECT for entry A and would prove nothing. Under the old raw
                // encoding this single zero makes a call to B enter A's block; under the
                // fixed encoding it is a miss.
                poker = setInterval(() => {
                    const meta = ex.jit_debug_meta_lo((BASE + 0x1000) >>> 12) >>> 0;
                    if (meta === 0) return;
                    const slab = meta & 0xFFFF;
                    const cells = new Uint16Array(cpu.wasm_memory.buffer, poolBase + slab * 0x2000, 0x1000);
                    if (cells[0x040] !== 0) { cells[0x040] = 0; zeroed++; }
                }, 1);
            }
            emulator.run();
        });
    });
}

const interp = await run({ corrupt: null });
const clean = await run({ corrupt: false });
const stomped = await run({ corrupt: true });

const sig = (r) => `${r.status} chk=${r.chk.toString(16)} edx=${r.edx.toString(16)}`;
console.log("interp  " + sig(interp));
console.log("jit     " + sig(clean));
console.log("stomped " + sig(stomped) + ` (zeroed ${stomped.zeroed} cells)`);

if (sig(clean) !== sig(interp)) {
    console.error("FAIL: plain JIT run already disagrees with the interpreter"); process.exit(1);
}
if (stomped.zeroed === 0) {
    console.error("FAIL: no published cells were ever zeroed — the injection missed, so the\n" +
                  "      assertion below proves nothing (did the page get compiled at all?)");
    process.exit(1);
}
if (sig(stomped) !== sig(interp)) {
    console.error("FAIL: zeroed dispatch-slab cells changed guest behaviour\n" +
                  `      expected ${sig(interp)}\n      got      ${sig(stomped)}\n` +
                  "      A zero cell must be a dispatch MISS (cells store state+1), not a live\n" +
                  "      dispatch into the module's block 0 — see dispatch_state_lookup.");
    process.exit(1);
}
console.log("  ok — zeroed slab cells behaved as misses; guest result unchanged");
process.exit(0);
