#!/usr/bin/env node
// The JIT must actually PUBLISH modules — not merely generate them.
//
// The gap this closes: every other oracle here asserts a RESULT (ecx == 0, no faults, matching
// FPU bits). All of those pass just as happily when the JIT is completely dead, because the
// interpreter computes the same answers — only slower. So a build in which every generated
// module fails to instantiate looked green across the whole suite while the real emulator ran
// ~7x slower.
//
// The failure is silent by construction: WebAssembly.instantiate rejects asynchronously,
// nothing on the Rust side observes it, codegen_finalize_finished never runs, and JitState's
// single in-flight `compiling` slot is therefore never released — so
// jit_increase_hotness_and_maybe_compile returns early forever after the FIRST bad module.
// One unresolvable import (e.g. a #[no_mangle] lost to an inserted function) is enough.
//
// cpu.js exposes both halves of the discriminator:
//   globalThis.__jitCompileStats.count — codegen_finalize entries, i.e. a module was BUILT
//                                        (test_hook_did_generate_wasm is DEBUG-only, so it
//                                        never fires in the shipped build)
//   test_hook_did_finalize_wasm        — that module INSTANTIATED and was installed
// generated > 0 with finalized == 0 is exactly the dead-JIT signature.
//
// The workload is a hot cross-page CALL/RET loop with MAX_PAGES=1, so the callee lands in its
// own module and every RET returns into a DIFFERENT one — which is what drives the RET dynamic
// chaining path (jit_find_cache_entry_for_dynamic_chaining). That matters: the plain
// direct-jump loop in jit-block-chaining-repro never emits that helper, so it cannot see a
// break in it.

const { V86 } = await import("../build/libv86.mjs");

const BASE = 0x100000;
const ENTRY_OFF = 0x20;
const PAGE1_OFF = 0x1000;
const ITER = 400000;
const MEM_SIZE = 16 * 1024 * 1024;
const TIMEOUT_MS = 20000;

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
    dv.setUint32(0x18, BASE + 0x4000, true);
    dv.setUint32(0x1c, BASE + ENTRY_OFF, true);

    let o = ENTRY_OFF;
    const labels = {}, patches = [];
    const label = n => { labels[n] = o; };
    const emit = (...bytes) => { for(const b of bytes) buf[o++] = b & 0xff; };
    const u32 = value => { dv.setUint32(o, value >>> 0, true); o += 4; };
    const rel32 = n => { patches.push({ at: o, sz: 4, end: o + 4, to: n }); emit(0, 0, 0, 0); };

    emit(0xBC); u32(BASE + 0x3000);        // mov esp, stack
    emit(0xB9); u32(ITER);                 // mov ecx, ITER
    label("loop");
    emit(0xE8); rel32("callee");           // call page1:callee  (cross-page → other module)
    emit(0x49);                            // dec ecx
    emit(0x0F, 0x85); rel32("loop");       // jnz loop
    emit(0xF4); emit(0xEB, 0xFE);          // hlt; jmp $

    o = PAGE1_OFF;
    label("callee");
    emit(0xC3);                            // ret  → returns into page0's module

    for(const p of patches)
    {
        const d = labels[p.to] - p.end;
        dv.setInt32(p.at, d, true);
    }
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
        let halted = false, timer, finalized = 0;
        globalThis.__jitCompileStats = { count: 0, bytes: 0 };
        const finish = status => {
            clearTimeout(timer);
            try { emulator.stop(); } catch(e) {}
            const cpu = emulator.v86.cpu;
            const dget = cpu.wm.exports["profiler_dispatch_stat_get"];
            let published = 0;
            for(let i = 1; i < 900; i++) if(cpu.wm.wasm_table.get(i + 1024)) published++;
            resolve({
                status,
                ecx: cpu.reg32[1] >>> 0,
                generated: globalThis.__jitCompileStats.count | 0, finalized, published,
                retChaining: cpu.get_jit_config ? cpu.get_jit_config(12) >>> 0 : 0,
                retChainHit: dget ? dget(11) : 0,
                chainEntries: cpu.wm.exports["jit_get_tier2_chain_entries"]?.() ?? 0,
                directEntries: cpu.wm.exports["jit_get_tier2_direct_entries"]?.() ?? 0,
            });
        };

        emulator.bus.register("cpu-event-halt", () => { halted = true; finish("halt"); });
        emulator.add_listener("emulator-loaded", () => {
            const cpu = emulator.v86.cpu;
            cpu.reboot_internal();
            cpu.reset_memory();
            cpu.set_jit_config(1, 1);   // MAX_PAGES=1 — callee gets its own module
            cpu.set_jit_config(12, 1);  // RET dynamic chaining ON (the path under test)
            cpu.wm.exports["set_dispatch_stats"]?.(1);
            cpu.wm.exports["profiler_init"]?.();
            cpu.jit_clear_cache?.();
            cpu.test_hook_did_finalize_wasm = () => { finalized++; };
            cpu.load_multiboot(build_image().buffer);
            timer = setTimeout(() => { if(!halted) finish("HANG"); }, TIMEOUT_MS);
            emulator.run();
        });
    });
}

const r = await run();
console.log("jit-alive " + JSON.stringify(r));

const fail = msg => { console.error("FAIL: " + msg); process.exit(1); };

if(r.status !== "halt" || r.ecx !== 0) fail("loop did not halt cleanly with ecx=0");
if(r.generated === 0) fail("the JIT generated no modules at all — workload never got hot");
if(r.finalized === 0)
{
    fail("modules were GENERATED but none INSTANTIATED (generated=" + r.generated + ", finalized=0).\n" +
         "      Every generated module is failing to link, which also wedges JitState's single\n" +
         "      in-flight compile slot, so the JIT is dead for the rest of the run.\n" +
         "      Run `bun tools/validate-jit-exports.ts` — the usual cause is a JIT helper that\n" +
         "      lost its #[no_mangle] and is no longer reachable through cpu.jit_imports.");
}
if(r.published === 0) fail("nothing was installed in the wasm table despite finalized=" + r.finalized);
if(r.retChaining && r.retChainHit <= 0) fail("RET chaining enabled but RET_CHAIN_HIT stayed zero — the chaining path never ran");
if(r.retChaining && r.chainEntries <= 0) fail("RET chaining ran but chained entries were not accounted (jit_get_tier2_chain_entries == 0)");

console.log(`  ok — generated=${r.generated} finalized=${r.finalized} published=${r.published} ` +
            `retChainHit=${r.retChainHit} chainEntries=${r.chainEntries} directEntries=${r.directEntries}`);
process.exit(0);
