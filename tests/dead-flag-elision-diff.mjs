#!/usr/bin/env node
// Differential test for dead-flag elision (jit config idx 5).
//
// Oracle:  idx5 = 0 — every flag-writing instruction publishes its flags.
// Suspect: idx5 = 1 — a write is skipped when the analysis proves nothing can read it first.
//
// Elision must be INVISIBLE. The two builds are handed the same image and the same register
// state, and the guest-visible flags at the module's exit must be bit-identical.
//
// What this catches that a "does the loop still produce the right answer" test cannot: the
// module can LEAVE before the instruction that was supposed to overwrite the flags. The
// loop-safety exit (idx 2) sits at a loop head and branches out of the module on the retired
// counter, so a cross-block proof that steps over that edge is proving deadness past a point
// where the guest can already be looking at the flags. We observe that exit by calling the
// compiled module directly and checking that it returned inside the live loop.
//
// The `pad` sweep moves the overwriting instruction relative to the block boundary, because the
// divergence only appears for the placements that put the loop head between the write and its
// overwriter.
//
//   node vendor/v86/tests/dead-flag-elision-diff.mjs

import { fileURLToPath } from "node:url";
const { V86 } = await import("../build/libv86.mjs");
const { SHIPPING_JIT } = await import("../../../tools/jit-config/shipping.mjs");

const BASE = 0x100000, ENTRY = 0x40;
const PADS = 12;
const TIMEOUT_MS = 20000;

/**
 * A multiboot image whose loop writes flags, crosses a block boundary, and overwrites them.
 *
 * `pad` NOPs sit between the two, so the sweep walks the overwriter across the loop head.
 */
function image(pad) {
    const b = new Uint8Array(4096), d = new DataView(b.buffer);
    [0x1badb002, 0x10000, (-0x1badb002 - 0x10000) >>> 0, BASE, BASE, BASE + 4096, BASE + 8192, BASE + ENTRY]
        .forEach((v, i) => d.setUint32(i * 4, v, true));
    let o = ENTRY;
    const e = (...a) => { for (const x of a) b[o++] = x; };
    const u = (x) => { d.setUint32(o, x, true); o += 4; };
    e(0xb9); u(1000000);                    // mov ecx, 1000000
    e(0xbb); u(1);                          // mov ebx, 1
    e(0x31, 0xc0);                          // xor eax, eax        — writes flags
    const A = o;
    e(0x83, 0xf9, 0);                       // cmp ecx, 0
    e(0x74, 0); const endpatch = o - 1;     // je end
    for (let i = 0; i < pad; i++) e(0x90);  // nops
    e(0x83, 0xc0, 1);                       // add eax, 1          — writes flags
    const B = o;
    e(0x83, 0xc0, 2);                       // add eax, 2          — overwrites them
    e(0x4b);                                // dec ebx
    e(0x75, (B - o - 2) & 255);             // jnz B
    e(0xbb); u(1);                          // mov ebx, 1
    e(0x49);                                // dec ecx
    e(0xeb, (A - o - 2) & 255);             // jmp A
    b[endpatch] = (o - endpatch - 1) & 255;
    e(0xf4);                                // hlt
    return { b, A: BASE + A, B: BASE + B };
}

/** Compile the image with `flag`, then run the compiled module to its budget exit. */
function run(flag, pad) {
    return new Promise((resolve, reject) => {
        const em = new V86({
            autostart: false, memory_size: 16 << 20,
            wasm_path: process.env.V86_WASM_PATH || fileURLToPath(new URL("../build/v86.wasm", import.meta.url)),
            log_level: 0,
        });
        const timer = setTimeout(() => reject(new Error("timeout")), TIMEOUT_MS);
        em.add_listener("emulator-loaded", () => {
            try {
                const c = em.v86.cpu, w = c.wm.exports;
                c.reboot_internal();
                c.reset_memory();
                const im = image(pad);
                c.load_multiboot(im.b.buffer);
                for (const [i, v] of SHIPPING_JIT) w.set_jit_config(i, v);
                w.set_jit_config(5, flag);
                w.set_relaxed_fpu(1);
                globalThis.__wasmDump = { out: [] };
                c.test_hook_did_finalize_wasm = () => {
                    try {
                        // Instantiate the module the JIT just produced and enter it with a state
                        // that takes the loop-safety exit: what the guest sees THERE is the
                        // question, and waiting for the loop to end would never ask it.
                        const rec = globalThis.__wasmDump.out[0];
                        const ins = new WebAssembly.Instance(
                            new WebAssembly.Module(rec.bytes), { e: c.jit_imports });
                        c.reg32[0] = 0; c.reg32[1] = 1000000; c.reg32[3] = 1;
                        ins.exports.f(0);
                        const exitEip = c.instruction_pointer[0] >>> 0;
                        const remaining = c.reg32[1] >>> 0;
                        if ((exitEip !== im.A && exitEip !== im.B)
                            || remaining === 0 || remaining >= 1000000
                            || (c.reg32[0] >>> 0) === 0) {
                            throw new Error(`expected budget exit inside live loop: eip=0x${exitEip.toString(16)}, ecx=${remaining}`);
                        }
                        clearTimeout(timer);
                        em.stop();
                        resolve({
                            flag, pad,
                            eip: c.instruction_pointer[0] >>> 0,
                            eax: c.reg32[0] >>> 0,
                            ecx: c.reg32[1] >>> 0,
                            eflags: w.get_eflags(),
                        });
                    } catch (e) { clearTimeout(timer); reject(e); }
                };
                em.run();
            } catch (e) { clearTimeout(timer); reject(e); }
        });
    });
}

const bad = [];
for (let pad = 0; pad < PADS; pad++) {
    const off = await run(0, pad);
    const on = await run(1, pad);
    const same = off.eflags === on.eflags && off.eax === on.eax
        && off.ecx === on.ecx && off.eip === on.eip;
    console.log(`  ${same ? "PASS" : "FAIL"}  pad ${String(pad).padStart(2)}`
        + `  eflags 0x${off.eflags.toString(16)} vs 0x${on.eflags.toString(16)}`
        + `  eip 0x${off.eip.toString(16)}/0x${on.eip.toString(16)}`);
    if (!same) bad.push(pad);
}
console.log(`\n=== ${bad.length === 0 ? "PASS" : "FAIL"} — ${PADS - bad.length}/${PADS} placements agree`
    + `${bad.length ? `; elision changes guest state at pad ${bad.join(", ")}` : ""} ===`);
process.exit(bad.length === 0 ? 0 : 1);
