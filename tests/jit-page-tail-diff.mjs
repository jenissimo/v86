#!/usr/bin/env node
// Differential oracle for page-tail compilation (set_jit_page_tails).
//
// Stock v86 never compiles an instruction that starts in the last 16 bytes of a page. With
// page tails on, the JIT compiles every instruction that lies wholly inside its page (one
// ending exactly at the page end included) and leaves a page-straddling instruction to the
// interpreter. What can go wrong, and what each section here is built to catch:
//
//   - decoding bytes of the NEXT PHYSICAL page: the virtual page after the tails is remapped
//     to a non-adjacent frame, and the physically adjacent frame holds decoy bytes. A block
//     that baked a straddler's immediate from the wrong frame computes a different checksum.
//   - an instruction ending exactly at the page end: its "eip after" is the next page; the
//     stock `page | low12` formula names the START of the same page.
//   - a CALL at the page end (return address on the next page) and a leaf function whose
//     entry sits in a tail (a dispatcher entry point in the last 16 bytes).
//   - SMC: a guest store into a compiled tail instruction and into the second half of a
//     straddler, and a host (JS) write + jit_dirty_cache — the writeGuestCode path.
//
// Arms: interpreter, JIT with tails off, JIT with tails on (production knob set). Final
// registers + checksum must match the interpreter in both JIT arms.
//
// Built-in negative control (--nc-noinval): the host patch skips jit_dirty_cache. The tails-ON
// arm must then DIVERGE (the patched tail instruction was compiled and is now stale) while the
// tails-OFF arm must NOT (it interprets that instruction, so there is nothing stale). That pair
// is the proof the tail code actually ran compiled, not a claim.
//
//   node tests/jit-page-tail-diff.mjs [seedStart] [seedCount] [--engine <v86.wasm>] [--nc-noinval]

import path from "node:path";
import url from "node:url";

const __dirname = path.dirname(url.fileURLToPath(import.meta.url));
// Identity mode compares two ENGINES, so it must not depend on how fast each engine runs.
// v86 slices execution by wall clock (main_loop vs TIME_PER_FRAME) and publishes compiled
// modules asynchronously at slice boundaries, so a faster or slower build compiles different
// module sets from the same guest. A call-counting clock makes every slice a fixed number of
// do_many_cycles rounds, long enough that each compile finishes inside the slice it began in.
if (process.argv.includes("--identity")) {
    const step = Number(process.env.VCLOCK_STEP || 0.2);
    globalThis.__vclock = 0;   // reset per run: every arm starts from the same clock
    performance.now = () => (globalThis.__vclock += step);
    // ...and compile synchronously: v86 publishes a JIT module in a promise continuation, so an
    // asynchronous compile lands after however many slices the HOST needed to finish it.
    // A pre-resolved promise lands at the first microtask checkpoint, i.e. right after the
    // slice that asked for it, whatever the engine's speed.
    const instantiate = WebAssembly.instantiate;
    WebAssembly.instantiate = (src, imports) => {
        if (src instanceof WebAssembly.Module) return instantiate(src, imports);
        if (src.length > (1 << 20)) return instantiate(src, imports);   // the engine itself
        const module = new WebAssembly.Module(src);
        return Promise.resolve({ module, instance: new WebAssembly.Instance(module, imports) });
    };
}
const { V86 } = await import("../build/libv86.mjs");

const args = process.argv.slice(2);
const flag = n => args.includes(n);
const opt = n => { const i = args.indexOf(n); return i >= 0 ? args[i + 1] : undefined; };
const positional = args.filter((a, i) => !a.startsWith("--") && !(i > 0 && ["--engine", "--identity"].includes(args[i - 1])));
const SEED_START = parseInt(positional[0] ?? "1", 10);
const SEED_COUNT = parseInt(positional[1] ?? "8", 10);
const ENGINE = opt("--engine") ? path.resolve(opt("--engine")) : path.join(__dirname, "../build/v86.wasm");
const NC_NOINVAL = flag("--nc-noinval");
// --identity <baseline.wasm>: the tails-OFF arm must emit byte-identical modules to an engine
// built without the page-tail switch (the switch's own default-off contract).
const IDENTITY = opt("--identity") ? path.resolve(opt("--identity")) : null;

const BASE = 0x100000;
const IMG_PAGES = 2 + 2 * 16 + 2;
const IMG_SIZE = IMG_PAGES * 0x1000;
const ENTRY_OFF = 0x40;
const DATA = BASE + 0x1000;            // page 1: data
const CHK = DATA + 0x00;
const CTR = DATA + 0x04;
const PHASE = DATA + 0x08;
const STACK_TOP = BASE + 0x1F00;
const G0 = 2;                          // first gauntlet page index
const NG = 16;                         // gauntlets; each owns pages G0+2k (tails) and G0+2k+1
const REMAP_K = 3;                     // gauntlet whose second page is remapped
const ALT_FRAME = 0x600000;            // physical frame backing that page
const PD = 0x300000, PT0 = 0x301000, PT1 = 0x302000;
const OUTER = 80000;
const PATCH_GUEST_AT = 25000, PATCH_HOST_AT = 55000;
const DONE_PORT = 0x9999, HOST_PORT = 0x9990, FENCE_PORT = 0x9991;
const MEM_SIZE = 32 * 1024 * 1024;
const TIMEOUT_MS = 120_000;

function mulberry32(a) {
    return () => {
        a |= 0; a = a + 0x6D2B79F5 | 0;
        let t = Math.imul(a ^ a >>> 15, 1 | a);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
}

// Straight-line instructions: length, encoder, and where their patchable imm32 is (or -1).
// Flags are only ever consumed right after a CMP that defines them all, so IMUL's undefined
// flags cannot make the arms legitimately differ.
const OPS = [
    { n: 5, e: i => [0x05, ...d32(i)], imm: 1 },                                   // add eax, id
    { n: 6, e: i => [0x81, 0xF1, ...d32(i)], imm: 2 },                             // xor ecx, id
    { n: 6, e: i => [0x69, 0xD2, ...d32(i | 1)], imm: 2 },                         // imul edx, edx, id
    { n: 7, e: i => [0x8D, 0x9C, 0xB3, ...d32(i)], imm: 3 },                       // lea ebx, [ebx+esi*4+d]
    { n: 4, e: i => [0x66, 0x05, i & 0xFF, i >> 8 & 0xFF], imm: -1 },              // add ax, iw
    { n: 3, e: i => [0xC1, 0xC0, 1 + (i & 15)], imm: -1 },                         // rol eax, ib
    { n: 1, e: () => [0x46], imm: -1 },                                            // inc esi
    { n: 2, e: () => [0x01, 0xC7], imm: -1 },                                      // add edi, eax
    { n: 2, e: () => [0x31, 0xD9], imm: -1 },                                      // xor ecx, ebx
    { n: 3, e: () => [0x0F, 0xAF, 0xC2], imm: -1 },                                // imul eax, edx
    { n: 2, e: () => [0xF7, 0xD3], imm: -1 },                                      // not ebx
    { n: 6, e: i => [0x81, 0xC6, ...d32(i)], imm: 2 },                             // add esi, id
    { n: 9, e: i => [0x3E, 0x3E, 0x3E, 0x81, 0xC3, ...d32(i)], imm: 5 },           // ds*3 add ebx, id
    { n: 11, e: i => [0x3E, 0x3E, 0x3E, 0x3E, 0x3E, 0x69, 0xC0, ...d32(i | 1)], imm: 7 },
    { n: 15, e: i => [0x3E, 0x3E, 0x3E, 0x3E, 0x3E, 0x3E, 0x3E, 0x3E, 0x3E, 0x69, 0xC9, ...d32(i | 1)], imm: 11 },
    { n: 7, e: i => [0x8D, 0x84, 0x30, ...d32(i)], imm: 3 },                       // lea eax, [eax+esi+d]
];
function d32(v) { v >>>= 0; return [v & 255, v >> 8 & 255, v >> 16 & 255, v >>> 24]; }

function buildImage(seed) {
    const rnd = mulberry32(seed);
    const R = n => Math.floor(rnd() * n);
    const img = new Uint8Array(IMG_SIZE).fill(0xCC);
    const dv = new DataView(img.buffer);
    const MAGIC = 0x1BADB002, FLAGS = 0x10000;
    dv.setUint32(0x00, MAGIC, true);
    dv.setUint32(0x04, FLAGS, true);
    dv.setUint32(0x08, (-(MAGIC + FLAGS)) >>> 0, true);
    dv.setUint32(0x0c, BASE, true);
    dv.setUint32(0x10, BASE, true);
    dv.setUint32(0x14, BASE + IMG_SIZE, true);
    dv.setUint32(0x18, BASE + IMG_SIZE, true);
    dv.setUint32(0x1c, BASE + ENTRY_OFF, true);

    let o = 0;
    const fix = [];                     // { at, to, rel } rel32 / rel8 fixups against labels
    const labels = {};
    const emit = (...b) => { for (const x of b) img[o++] = x & 0xFF; };
    const rel32 = to => { fix.push({ at: o, to, size: 4 }); o += 4; };
    const rel8 = to => { fix.push({ at: o, to, size: 1 }); o += 1; };
    const label = n => { labels[n] = BASE + o; };

    // ── gauntlets: page A = G0+2k holds the entry at its start and a tail run that crosses
    // into page B = A+1, where it continues and ends in RET.
    const patchTail = [];  // guest-patchable imm32 wholly inside a tail (last 16 bytes)
    const patchCross = []; // imm32 bytes on the far side of a straddler
    const leafs = [];
    for (let k = 0; k < NG; k++) {
        const pa = (G0 + 2 * k) * 0x1000;
        // a leaf whose ENTRY sits in the tail of page B: called from another page, so the
        // dispatcher enters it at a last-16-byte address
        const leafOff = pa + 0x2000 - 1 - 6 - R(9);       // "add ecx, id; ret" fits in page B
        o = leafOff; label(`leaf${k}`);
        emit(0x81, 0xC1, ...d32((rnd() * 2 ** 32) >>> 0)); emit(0xC3);
        leafs.push(BASE + leafOff);

        o = pa; label(`g${k}`);
        for (let i = 0; i < 3; i++) { const op = OPS[R(OPS.length)]; emit(...op.e((rnd() * 2 ** 32) >>> 0)); }
        emit(0xE9); rel32(`t${k}`);                       // jmp into the tail region
        // tail run starts between 44 and 20 bytes before the page end
        o = pa + 0x1000 - 20 - R(24);
        label(`t${k}`);
        let calls = 0;
        while (o < pa + 0x1000 + 24) {
            const r = R(10);
            if (r === 0 && o + 7 < pa + 0x1000 + 24) {
                // cmp eax, id ; jcc rel8 over the next 2-byte op (a conditional block end in the tail)
                emit(0x3D, ...d32((rnd() * 2 ** 32) >>> 0));
                emit(0x70 + [2, 3, 4, 5, 8, 9][R(6)]); emit(2);
                emit(0x01, 0xC7);
            }
            else if (r === 1 && calls < 2) {
                calls++;
                emit(0xE8); rel32(`leaf${(k + 1 + R(NG - 1)) % NG}`);  // call a tail leaf
            }
            else {
                const op = OPS[R(OPS.length)];
                const at = o;
                emit(...op.e((rnd() * 2 ** 32) >>> 0));
                if (op.imm >= 0) {
                    const imm = at + op.imm;
                    const end = pa + 0x1000;
                    if (at >= end - 16 && imm + 4 <= end) patchTail.push(BASE + imm);
                    else if (at < end && imm + 4 > end) patchCross.push(BASE + imm);
                }
            }
        }
        emit(0xC3);                                       // ret, in page B
    }

    // ── driver on page 0
    o = ENTRY_OFF;
    emit(0xBC, ...d32(STACK_TOP));                         // mov esp
    // identity paging for 0..8 MiB, then remap page B of gauntlet REMAP_K to ALT_FRAME
    emit(0xB8, ...d32(PT0 | 3)); emit(0xA3, ...d32(PD));   // mov [PD], eax
    emit(0xB8, ...d32(PT1 | 3)); emit(0xA3, ...d32(PD + 4));
    emit(0xB8, ...d32(3)); emit(0xB9, ...d32(2048)); emit(0xBA, ...d32(PT0));
    label("pt"); emit(0x89, 0x02); emit(0x05, ...d32(0x1000)); emit(0x83, 0xC2, 4); emit(0x49);
    emit(0x75); rel8("pt");
    const remapVa = BASE + (G0 + 2 * REMAP_K + 1) * 0x1000;
    emit(0xC7, 0x05, ...d32(PT0 + (remapVa >>> 12) * 4), ...d32(ALT_FRAME | 3));
    emit(0xB8, ...d32(PD)); emit(0x0F, 0x22, 0xD8);        // cr3
    emit(0x0F, 0x20, 0xC0); emit(0x0D, ...d32(0x80000000)); emit(0x0F, 0x22, 0xC0); // PG
    for (const r of [0, 1, 2, 3, 6, 7]) emit(0xB8 + r, ...d32((0x1234567 * (r + 3)) >>> 0));
    emit(0xC7, 0x05, ...d32(CTR), ...d32(OUTER));
    label("outer");
    for (let k = 0; k < NG; k++) {
        emit(0xE8); rel32(`g${k}`);
        for (const r of [0, 1, 2, 3, 6, 7]) emit(0x31, 0x05 | r << 3, ...d32(CHK)); // xor [CHK], r
        emit(0xD1, 0x05, ...d32(CHK));                     // rol dword [CHK], 1
    }
    // phase triggers
    emit(0x81, 0x3D, ...d32(CTR), ...d32(OUTER - PATCH_GUEST_AT)); // cmp [CTR], id
    emit(0x75); rel8("noguest");
    for (const a of [...patchTail.slice(0, 4), ...patchCross.slice(0, 4)])
        emit(0xC7, 0x05, ...d32(a), ...d32((rnd() * 2 ** 32) >>> 0 | 1)); // guest SMC
    // Leave the module after patching (OUT is a block boundary): v86 keeps running a module
    // that stored into its own page set until it exits, a pre-existing hole this oracle is not
    // about (see jit-hot-edge-diff.mjs, SMC_NO_FENCE).
    if (!process.env.SMC_NO_FENCE) { emit(0x66, 0xBA, FENCE_PORT & 0xFF, FENCE_PORT >> 8); emit(0xEE); }
    label("noguest");
    emit(0x81, 0x3D, ...d32(CTR), ...d32(OUTER - PATCH_HOST_AT));
    emit(0x75); rel8("nohost");
    emit(0x66, 0xBA, HOST_PORT & 0xFF, HOST_PORT >> 8); emit(0xEE);   // out dx, al -> host patch
    label("nohost");
    emit(0xFF, 0x0D, ...d32(CTR));                         // dec [CTR]
    emit(0x0F, 0x85); rel32("outer");
    emit(0x66, 0xBA, DONE_PORT & 0xFF, DONE_PORT >> 8); emit(0xEE);
    emit(0xF4); emit(0xEB, 0xFE);

    for (const f of fix) {
        const t = labels[f.to]; if (t === undefined) throw new Error("label " + f.to);
        const d = t - (BASE + f.at + f.size);
        if (f.size === 1) { if (d < -128 || d > 127) throw new Error("rel8 " + f.to + " " + d); img[f.at] = d & 0xFF; }
        else dv.setInt32(f.at, d, true);
    }
    // host patch set: tail immediates the guest did NOT patch (so it is the host write that matters)
    const hostPatch = patchTail.slice(4).map(a => ({ a, v: (rnd() * 2 ** 32) >>> 0 | 1 }));
    return { img, patchTail, patchCross, hostPatch, leafs, remapVa };
}

// A broken engine can crash the guest out of the emulator's timer loop; that is a divergence
// to report, not a reason for the oracle itself to die.
let currentFinish = null;
process.on("uncaughtException", e => {
    if (currentFinish) currentFinish("THREW " + String(e.message ?? e).slice(0, 60));
    else { console.error(e); process.exit(2); }
});

function run(build, cfg) {
    return new Promise(resolve => {
        const emulator = new V86({ autostart: false, memory_size: MEM_SIZE, disable_jit: cfg.jit ? 0 : 1,
                                   log_level: 0, wasm_path: cfg.engine ?? ENGINE });
        if (cfg.dump) globalThis.__wasmDump = { out: [] };
        if (globalThis.__vclock !== undefined) globalThis.__vclock = 0;
        let done = false, timer, cpu, w, hostPatches = 0, hostPagesCompiled = 0;
        const finish = status => {
            if (done) return; done = true;
            currentFinish = null;
            clearTimeout(timer);
            try { emulator.stop(); } catch { }
            const regs = Array.from({ length: 8 }, (_, i) => cpu.reg32[i] >>> 0);
            const b = emulator.read_memory(CHK, 4);
            const chk = (b[0] | b[1] << 8 | b[2] << 16 | b[3] << 24) >>> 0;
            // tail-entry evidence: slab cells of the leaf entries (state+1, 0 = no entry)
            let leafEntries = 0;
            if (cfg.jit && w.jit_debug_slab_cell) {
                for (const a of build.leafs) {
                    const c = w.jit_debug_slab_cell(a >>> 12, a & 0xFFF) >>> 0;
                    if (c !== 0 && c !== 0xFFFFFFFF) leafEntries++;
                }
            }
            const modules = cfg.dump ? globalThis.__wasmDump.out : null;
            if (cfg.dump) delete globalThis.__wasmDump;
            resolve({ status, regs, chk, hostPatches, hostPagesCompiled, leafEntries, modules,
                      tails: w?.get_jit_page_tails ? w.get_jit_page_tails() >>> 0 : -1 });
        };
        emulator.add_listener("emulator-loaded", () => {
            cpu = emulator.v86.cpu; w = cpu.wm.exports;
            cpu.reboot_internal(); cpu.reset_memory();
            if (process.env.TRACE_COMPILE) globalThis.__jitPublicationCapture = { begin: (start, idx, slot, bytes) => {
                console.log(`  compile start=${start.toString(16)} len=${bytes.length} insn=${cpu.instruction_counter[0] >>> 0}`);
                return null; } };
            if (cfg.jit) {
                // KNOBS="4=1,15=20000" widens the production knob set (chaining, tier-2, ...)
                const knobs = new Map([[5, 1], [11, 1], [12, 1], [13, 1], [22, 1]]);
                // Identity compares two ENGINE BUILDS, and with a multi-page budget which pages a
                // module admits under its cap follows HashSet order, which std on wasm32 seeds
                // from a heap address -- a static-layout change alone reorders it (a padding-only
                // build reproduces that). One page per module takes the walk order out.
                if (IDENTITY) knobs.set(1, 1);
                for (const kv of (process.env.KNOBS || "").split(",").filter(Boolean)) {
                    const [i, v] = kv.split("=").map(Number); knobs.set(i, v);
                }
                for (const [i, v] of knobs) cpu.set_jit_config(i, v);
                if (cfg.tails) {
                    if (!w.set_jit_page_tails) { resolve({ status: "NO-SWITCH" }); return; }
                    w.set_jit_page_tails(1);
                }
                else if (w.set_jit_page_tails) w.set_jit_page_tails(0);
                cpu.jit_clear_cache?.();
            }
            cpu.load_multiboot(build.img.buffer);
            // The remapped page: real bytes at ALT_FRAME, decoys in the physically adjacent frame.
            const real = emulator.read_memory(build.remapVa, 0x1000);
            emulator.write_memory(real, ALT_FRAME);
            emulator.write_memory(real.map(x => x ^ 0x5A), build.remapVa);
            cpu.io.register_write(DONE_PORT, cpu, () => finish("done"));
            cpu.io.register_write(FENCE_PORT, cpu, () => {});
            cpu.io.register_write(HOST_PORT, cpu, () => {
                // writeGuestCode in shape: write the bytes, invalidate in the same turn
                for (const { a } of build.hostPatch)
                    if (w.jit_aot_page_table_index && (w.jit_aot_page_table_index(a) >>> 0) !== 0xFFFF) hostPagesCompiled++;
                for (const { a, v } of build.hostPatch) {
                    // through mem8 like a JS HLE write: write_memory would invalidate by itself
                    cpu.mem8.set(d32(v), a);
                    if (!cfg.noInval) w.jit_dirty_cache(a, a + 4);
                }
                hostPatches++;
            });
            timer = setTimeout(() => finish("TIMEOUT"), TIMEOUT_MS);
            currentFinish = finish;
            emulator.run();
        });
    });
}

if (IDENTITY) {
    let bad = 0;
    for (let seed = SEED_START; seed < SEED_START + SEED_COUNT; seed++) {
        const build = buildImage(seed);
        const a = await run(build, { jit: true, tails: false, dump: true, engine: IDENTITY });
        // --identity-nc: compare against the tails-ON arm instead; it MUST come out DIFFERENT.
        const b = await run(build, { jit: true, tails: flag("--identity-nc"), dump: true });
        const hex = m => Buffer.from(m.bytes).toString("base64");
        // Two things legitimately differ between the runs. (1) Compilation is asynchronous, so
        // WHICH wasm-table slot a module gets, and the order modules finish in, follows host
        // timing: the slot is baked as `i32.const <slot>` and as "@t<slot>" in the name, so both
        // are normalised away and modules are matched as a multiset. (2) Two engine builds link
        // their statics at different addresses, baked as LEB128 immediates: derive old->new
        // pairs from where matched modules differ, require one consistent bijection, apply it to
        // every occurrence, and require equality. Anything else stays a diff.
        const leb = v => { const o = []; do { let x = v & 0x7f; v = Math.floor(v / 128); if (v) x |= 0x80; o.push(x); } while (v); return o; };
        const rd = (buf, p) => { let v = 0, sh = 0; for (;;) { const c = buf[p++]; v += (c & 0x7f) * 2 ** sh; sh += 7; if (!(c & 0x80)) return [v, p]; } };
        const replaceAll = (bytes, f, t) => {
            const out = [];
            for (let k = 0; k < bytes.length;) {
                if (k + f.length <= bytes.length && f.every((q, n) => bytes[k + n] === q)) { out.push(...t); k += f.length; }
                else out.push(bytes[k++]);
            }
            return Uint8Array.from(out);
        };
        const normalize = m => {
            let x = replaceAll(m.bytes, [0x41, ...leb(m.table_index)], [0x41, 0x00]);
            x = replaceAll(x, [...Buffer.from("@t" + m.table_index)], [...Buffer.from("@t#")]);
            return x;
        };
        const A = a.modules.map(m => ({ start: m.start, x: normalize(m) }));
        const B = b.modules.map(m => ({ start: m.start, x: normalize(m) }));
        const map = new Map();
        const tryMatch = (x, y) => {
            if (x.length !== y.length) return null;
            const local = new Map();
            for (let j = 0; j < x.length;) {
                if (x[j] === y[j]) { j++; continue; }
                let st = j; while (st > 0 && (x[st - 1] & 0x80)) st--;
                const [va, ea] = rd(x, st), [vb, eb] = rd(y, st);
                if (ea !== eb || va < 0x10000) return null;
                const want = local.get(va) ?? map.get(va);
                if (want !== undefined && want !== vb) return null;
                local.set(va, vb); j = ea;
            }
            return local;
        };
        const used = new Set();
        let same = A.length === B.length;
        for (const m of A) {
            if (!same) break;
            let hit = -1;
            for (let i = 0; i < B.length && hit < 0; i++) {
                if (used.has(i) || B[i].start !== m.start) continue;
                const local = tryMatch(m.x, B[i].x);
                if (local) { for (const [f, t] of local) map.set(f, t); hit = i; }
            }
            if (hit < 0) same = false; else used.add(hit);
        }
        if (new Set(map.values()).size !== map.size) same = false;
        if (map.size) console.log(`  relocated statics: ${[...map].map(([f, t]) => f.toString(16) + "->" + t.toString(16)).join(" ")}`);
        const bytes = ms => ms.reduce((n, m) => n + m.len, 0);
        console.log(`seed ${seed}: baseline ${a.modules.length} modules/${bytes(a.modules)} B, current ${b.modules.length}/${bytes(b.modules)} B — ${same ? "IDENTICAL" : "DIFFERENT"}`);
        if (process.env.LIST_MODULES) {
            const l = r => r.modules.map(m => m.start.toString(16) + ":" + m.len).sort().join(" ");
            console.log("  A " + l(a)); console.log("  B " + l(b));
        }
        if (!same && process.env.DUMP_DIFF) { const fs = await import("node:fs"); fs.writeFileSync(process.env.DUMP_DIFF + "a.json", JSON.stringify(a.modules.map(hex))); fs.writeFileSync(process.env.DUMP_DIFF + "b.json", JSON.stringify(b.modules.map(hex))); }
        if (!same || a.modules.length === 0) bad++;
    }
    if (bad) { console.error(`identity: ${bad} seed(s) differ or compiled nothing`); process.exit(1); }
    console.log("identity: tails-OFF modules are byte-identical to the baseline engine");
    process.exit(0);
}

const sig = r => `${r.status} chk=${r.chk.toString(16)} regs=${r.regs.map(x => x.toString(16)).join(",")}`;
let failures = 0;
const fail = m => { failures++; console.error("FAIL " + m); };
console.log(`engine ${ENGINE}`);
for (let seed = SEED_START; seed < SEED_START + SEED_COUNT; seed++) {
    const build = buildImage(seed);
    if (build.patchTail.length < 6 || build.patchCross.length < 2) {
        console.log(`seed ${seed}: skipped (patch sites tail=${build.patchTail.length} cross=${build.patchCross.length})`);
        continue;
    }
    const noInval = NC_NOINVAL;
    const interp = await run(build, { jit: false });
    const off = await run(build, { jit: true, tails: false, noInval });
    const on = await run(build, { jit: true, tails: true, noInval });
    if (on.status === "NO-SWITCH") { console.log(`seed ${seed}: engine has no set_jit_page_tails — tails arm skipped`); }
    const note = ` (hostPatches=${interp.hostPatches}/${off.hostPatches}/${on.hostPatches}` +
                 ` leafEntries off=${off.leafEntries} on=${on.leafEntries}/${build.leafs.length}` +
                 ` hostPagesCompiled off=${off.hostPagesCompiled} on=${on.hostPagesCompiled}/${build.hostPatch.length}` +
                 ` sites tail=${build.patchTail.length} cross=${build.patchCross.length})`;
    if (interp.status !== "done") { fail(`seed ${seed}: interpreter did not finish (${interp.status})`); continue; }
    const okOff = sig(off) === sig(interp), okOn = on.status === "NO-SWITCH" || sig(on) === sig(interp);
    if (!NC_NOINVAL) {
        if (okOff && okOn) console.log(`seed ${seed}: ok${note}`);
        else {
            fail(`seed ${seed}: DIVERGENCE${note}`);
            console.error(`  interp ${sig(interp)}\n  off    ${sig(off)}\n  on     ${sig(on)}`);
        }
        if (on.status !== "NO-SWITCH") {
            if (off.leafEntries !== 0) fail(`seed ${seed}: tails OFF registered ${off.leafEntries} tail entries`);
            if (on.leafEntries === 0) fail(`seed ${seed}: tails ON compiled no tail entry — the arm did not exercise the feature`);
        }
    }
    else {
        // negative control: stale compiled tail code must be visible ON, and only ON
        const detail = `off ${okOff ? "matches" : "DIVERGES"}, on ${okOn ? "matches" : "DIVERGES"}`;
        if (okOff && !okOn) console.log(`seed ${seed}: negative control ok (${detail})${note}`);
        else fail(`seed ${seed}: negative control inconclusive (${detail})${note}`);
    }
}
if (failures) { console.error(`${failures} failure(s)`); process.exit(1); }
console.log(NC_NOINVAL ? "negative control: stale tail code detected in every seed" : "all seeds consistent");
process.exit(0);
