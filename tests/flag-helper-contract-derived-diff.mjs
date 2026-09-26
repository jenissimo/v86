// Flag-helper contracts (set_flag_helper_contract) must not change architectural results.
//
// One guest loop interleaves flag-producing ALU ops with helper calls the derived table calls
// flag-neutral (x87 arithmetic, FSQRT, SSE), a flag WRITER (FCOMI), a flag READER (FCMOVB) and
// consumers after each helper (SETcc, ADC, Jcc, PUSHFD). Every arm must end with the
// interpreter's checksum. The contract arm must also have actually narrowed a sync, and the
// helpers under test must appear in the emitted modules, or the pass means nothing.
//
// Usage: node tests/flag-helper-contract-derived-diff.mjs [--expect-fail] [--counting]
//   V86_WASM_PATH=<artifact>   (default build/v86.wasm)
//   --expect-fail   negative control: exit 0 only if some contract arm DIVERGES.
//   --baseline <wasm>  also require contract OFF (idx21 on and off) to emit modules byte-identical
//                   to an engine built without contracts (tlb_data's link address relocated).
import { fileURLToPath } from "node:url";
import { V86 } from "../build/libv86.mjs";
import { SHIPPING_JIT } from "../../../tools/jit-config/shipping.mjs";
import { findTlbDataBase } from "../../../tools/aot/lib/tlb-base.mjs";

const wasm = process.env.V86_WASM_PATH || fileURLToPath(new URL("../build/v86.wasm", import.meta.url));
const EXPECT_FAIL = process.argv.includes("--expect-fail");
const baselineArg = process.argv.indexOf("--baseline");
const BASELINE = baselineArg > 0 ? process.argv[baselineArg + 1] : null;
// Stores go to SCRATCH, a page of its own: a store into the code page invalidates the JIT.
const N = 100000, BASE = 0x100000, DATA = BASE + 0x800, SCRATCH = 0x180000, OUT = SCRATCH + 0x100;

function image() {
    const b = new Uint8Array(4096), d = new DataView(b.buffer);
    [0x1badb002, 0x10000, (-0x1badb002 - 0x10000) >>> 0, BASE, BASE, BASE + 4096, BASE + 4096, BASE + 0x40]
        .forEach((v, i) => d.setUint32(i * 4, v, true));
    d.setFloat64(DATA - BASE + 0, 2.5, true);
    d.setFloat64(DATA - BASE + 8, 1.25, true);
    [1.5, -3.25, 0.5, 7.0].forEach((v, i) => d.setFloat32(DATA - BASE + 16 + i * 4, v, true));
    [2.0, 0.75].forEach((v, i) => d.setFloat64(DATA - BASE + 32 + i * 8, v, true));
    let p = 0x40;
    const e = (...x) => { b.set(x, p); p += x.length; };
    const u = x => { d.setUint32(p, x >>> 0, true); p += 4; };
    const fldA = () => { e(0xdd, 0x05); u(DATA); };
    const fldB = () => { e(0xdd, 0x05); u(DATA + 8); };
    const mix = reg => { e(0x6b, 0xff, 0x1f); e(0x01, 0xc7 | (reg << 3)); };   // edi = edi*31 + reg

    e(0xbc); u(0x200000);                        // mov esp
    e(0x31, 0xff); e(0x31, 0xf6);                // edi = checksum, esi = iterations
    e(0xbb); u(0x12345678);                      // ebx = LCG state
    e(0x0f, 0x20, 0xe0); e(0x0d); u(0x600); e(0x0f, 0x22, 0xe0);   // CR4.OSFXSR|OSXMMEXCPT
    e(0xdb, 0xe3);                               // fninit
    const loop = p;
    e(0x69, 0xdb); u(1103515245); e(0x81, 0xc3); u(12345);

    // A: ADD flags dirty -> FADD/FMUL (neutral) -> SETC/SETO/ADC read the ADD flags.
    e(0x89, 0xd8); e(0x89, 0xd9); e(0xc1, 0xc1, 0x07); e(0x01, 0xc8);
    fldA(); fldB(); e(0xd8, 0xc1); e(0xd8, 0xc9);
    e(0x0f, 0x92, 0xc2); e(0x0f, 0x90, 0xc6); e(0x11, 0xc7);
    e(0x0f, 0xb7, 0xd2); mix(2);
    e(0xdd, 0x1d); u(SCRATCH); e(0xdd, 0xd8);

    // B: SUB flags dirty -> MULPD/PUNPCKLWD/CVTPS2PD (neutral) -> PUSHFD reads them.
    e(0x89, 0xd8); e(0x2d); u(0x40000000);
    e(0x66, 0x0f, 0x10, 0x05); u(DATA + 16); e(0x66, 0x0f, 0x10, 0x0d); u(DATA + 32);
    e(0x66, 0x0f, 0x59, 0xc1); e(0x66, 0x0f, 0x61, 0xc1); e(0x0f, 0x5a, 0xc8);
    e(0x9c, 0x5a); e(0x81, 0xe2); u(0x8d5); mix(2);
    e(0x66, 0x0f, 0x7e, 0xc0); mix(0);

    // C: CMP flags dirty -> FCOMI (WRITER) -> SETB/SETZ/SETP/SETO must see FCOMI's flags.
    e(0x89, 0xd8); e(0x25); u(0xff); e(0x3d); u(0x80);
    fldA(); fldB(); e(0xdb, 0xf1);
    e(0x0f, 0x92, 0xc2); e(0x0f, 0x94, 0xc6); e(0x0f, 0x9a, 0xc1); e(0x0f, 0x90, 0xc5);
    e(0xdd, 0xd8, 0xdd, 0xd8);
    e(0x0f, 0xb7, 0xd2); mix(2); e(0x0f, 0xb7, 0xc9); mix(1);

    // D: SHL sets CF (dirty) -> FCMOVB (READER) must see it.
    e(0x89, 0xd8); e(0xd1, 0xe0);
    fldA(); fldB(); e(0xda, 0xc1);
    e(0xdd, 0x1d); u(SCRATCH); e(0xdd, 0xd8);
    e(0x8b, 0x05); u(SCRATCH + 4); mix(0);

    // E: TEST sets ZF (dirty) -> FSQRT (neutral D9 /7) -> JZ.
    e(0x89, 0xd8); e(0xa9); u(0x10000);
    fldA(); e(0xd9, 0xfa); e(0xdd, 0x1d); u(SCRATCH);
    e(0x74, 0x06); e(0x81, 0xc7); u(0x1111);

    // F: ADD sets CF (dirty) -> FSQRT -> ADC.
    e(0x89, 0xd8); e(0x01, 0xc0);
    fldA(); e(0xd9, 0xfa); e(0xdd, 0xd8);
    e(0x83, 0xd7, 0x00);

    e(0x46); e(0x81, 0xfe); u(N); e(0x0f, 0x82); u(loop - (p + 4));
    e(0x89, 0x3d); u(OUT); e(0xf4);
    return b;
}

const EXPECTED_HELPERS = {
    0: ["fpu_fadd", "fpu_fmul", "fpu_fcomi", "instr_DA_0_reg", "instr16_D9_7_reg", "instr_660F61", "instr_0F5A"],
    1: ["instr_DA_0_reg", "instr16_D9_7_reg", "instr_660F61", "instr_0F5A"],
};
const STAT_NAMES = ["calls", "contracted", "spillWords", "reloadWords", "spillElided", "reloadElided",
    "execCalls", "execSpillWords", "execReloadWords"];

async function run({ relaxed, jit, locals, contract, counting, engine = wasm }) {
    const em = new V86({ autostart: false, memory_size: 16 << 20, wasm_path: engine, log_level: 0 });
    await new Promise(r => em.add_listener("emulator-loaded", r));
    const c = em.v86.cpu, w = c.wm.exports;
    const legacy = engine !== wasm;   // a pre-contract baseline engine has none of the new exports
    if (!legacy && typeof w.set_flag_helper_contract !== "function") throw Error("engine exports no set_flag_helper_contract");
    c.reboot_internal(); c.reset_memory(); c.load_multiboot(image().buffer);
    for (const [i, v] of SHIPPING_JIT) w.set_jit_config(i, v);
    w.set_jit_config(0, jit ? 0 : 1);
    w.set_relaxed_fpu(relaxed);
    w.set_jit_config(21, locals);
    if (!legacy) {
        w.set_flag_helper_contract(contract);
        w.set_flag_sync_counting(counting ? 1 : 0);
        if (w.get_flag_helper_contract() !== contract) throw Error("contract mode did not read back");
    }
    globalThis.__wasmDump = { out: [] };
    await new Promise((resolve, reject) => {
        const t = setTimeout(() => { em.stop(); reject(Error("timeout")); }, 60000);
        em.bus.register("cpu-event-halt", () => { clearTimeout(t); em.stop(); resolve(); });
        em.run();
    });
    const modules = globalThis.__wasmDump.out;
    const imports = new Set();
    for (const m of modules) for (const x of WebAssembly.Module.imports(new WebAssembly.Module(m.bytes))) imports.add(x.name);
    const stats = legacy ? {} : Object.fromEntries(STAT_NAMES.map((n, i) => [n, w.flag_sync_stat_get(i)]));
    const result = {
        checksum: c.read32s(OUT) >>> 0, iterations: c.reg32[6] >>> 0, modules: modules.length,
        imports, stats, mutated: legacy ? 0 : w.get_flag_helper_contract_mutated(),
        bytes: modules.map(m => m.bytes),
        tlbBase: findTlbDataBase(c.wasm_memory, c.mem8.byteOffset, [BASE >>> 12, SCRATCH >>> 12, 0x1ff], 2).base,
    };
    em.destroy();
    return result;
}

let failures = 0, contractDivergences = 0;
const report = [];
for (const relaxed of [0, 1]) {
    const ref = await run({ relaxed, jit: false, locals: 0, contract: 0 });
    if (ref.iterations !== N) throw Error(`interpreter did ${ref.iterations} iterations`);
    const arms = [
        { name: "idx21=0", locals: 0, contract: 0 },
        { name: "idx21=0 contract=1", locals: 0, contract: 1 },
        { name: "idx21=1 contract=0", locals: 1, contract: 0 },
        { name: "idx21=1 contract=1", locals: 1, contract: 1 },
        { name: "idx21=1 contract=1 counting", locals: 1, contract: 1, counting: true },
        { name: "idx21=1 contract=0 counting", locals: 1, contract: 0, counting: true },
    ];
    for (const arm of arms) {
        const r = await run({ relaxed, jit: true, ...arm });
        const ok = r.checksum === ref.checksum && r.iterations === N;
        const missing = EXPECTED_HELPERS[relaxed].filter(h => !r.imports.has(h));
        const line = { relaxed, arm: arm.name, checksum: r.checksum.toString(16), ref: ref.checksum.toString(16),
            ok, modules: r.modules, missingHelpers: missing, mutatedTable: r.mutated, stats: r.stats };
        report.push(line);
        console.log(JSON.stringify(line));
        if (!r.modules) { failures++; console.log("  FAIL: nothing was JIT-compiled"); }
        if (missing.length) { failures++; console.log(`  FAIL: helpers under test absent from emitted code: ${missing}`); }
        if (arm.locals && arm.contract && r.stats.contracted === 0) { failures++; console.log("  FAIL: contract never applied"); }
        if (!ok) {
            if (arm.contract) contractDivergences++;
            else failures++;
            console.log(`  ${EXPECT_FAIL && arm.contract ? "DIVERGED (expected)" : "FAIL"}: checksum ${r.checksum.toString(16)} != interpreter ${ref.checksum.toString(16)}`);
        }
    }
}

if (BASELINE) {
    const leb = (v) => { const o = []; do { let b = v & 0x7f; v >>>= 7; if (v) b |= 0x80; o.push(b); } while (v); return o; };
    const relocate = (bytes, from, to) => {
        if (from === to) return bytes;
        const f = leb(from), t = leb(to), out = Uint8Array.from(bytes);
        if (f.length !== t.length) return null;
        for (let i = 0; i + f.length <= out.length; i++) if (f.every((b, k) => out[i + k] === b)) { out.set(t, i); i += f.length - 1; }
        return out;
    };
    const b64 = (x) => Buffer.from(x).toString("base64");
    for (const relaxed of [0, 1]) for (const locals of [0, 1]) {
        const base = await run({ relaxed, jit: true, locals, contract: 0, engine: BASELINE });
        const cur = await run({ relaxed, jit: true, locals, contract: 0 });
        const same = base.bytes.length > 0 && base.bytes.length === cur.bytes.length
            && base.bytes.every((m, i) => { const r = relocate(m, base.tlbBase, cur.tlbBase); return r && b64(r) === b64(cur.bytes[i]); });
        console.log(`identity relaxed=${relaxed} idx21=${locals} contract=0: baseline ${base.bytes.length} module(s), `
            + `tlb 0x${base.tlbBase.toString(16)} -> 0x${cur.tlbBase.toString(16)} — ${same ? "IDENTICAL" : "DIFFERENT"}`);
        if (!same) { failures++; console.log("  FAIL: contract OFF does not reproduce the pre-contract engine's modules"); }
    }
}

if (EXPECT_FAIL) {
    if (failures) { console.log(`NEGATIVE CONTROL INVALID: ${failures} non-contract failure(s)`); process.exit(1); }
    if (!contractDivergences) { console.log("NEGATIVE CONTROL FAILED: a wrong contract produced no divergence"); process.exit(1); }
    console.log(`NEGATIVE CONTROL OK: ${contractDivergences} contract arm(s) diverged from the interpreter`);
    process.exit(0);
}
if (failures || contractDivergences) { console.log(`FAIL: ${failures + contractDivergences} problem(s)`); process.exit(1); }
console.log(`PASS: ${report.length} arms match the interpreter (relaxed 0/1 x idx21 x contract x counting)`);
