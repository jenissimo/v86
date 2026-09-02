#!/usr/bin/env node
// Cross-page direct-jump loop for block chaining.
//
// MAX_PAGES=1 forces page0 and page1 into separate JIT modules:
//   page0: dec ecx; jz done; jmp page1
//   page1: jmp page0
//
// With tail-call support, JIT_BLOCK_CHAINING should produce CHAINED_EDGE > 0
// while still halting with ecx == 0.

const { V86 } = await import("../build/libv86.mjs");

const BASE = 0x100000;
const ENTRY_OFF = 0x20;
const PAGE1_OFF = 0x1000;
const ITER = 800000;
const MEM_SIZE = 16 * 1024 * 1024;
const TIMEOUT_MS = 12000;

// Exercise the exported resolver directly after the guest has populated both pages.
// This makes the guards deterministic instead of hoping that a scheduler tick happens
// at a particular edge. Every probe passes a distinct retired count, so their sum also
// detects either lost or double accounting on hit and fallback paths.
function probeDirectChainInvariants(cpu)
{
    const w = cpu.wm.exports;
    const page = (BASE + PAGE1_OFF) >>> 12;
    const target = BASE + PAGE1_OFF;
    const metaLo = w.jit_debug_meta_lo(page) >>> 0;
    const stateFlags = w.jit_debug_meta_hi(page) >>> 0;
    const tableIndex = metaLo >>> 16;
    const slab = metaLo & 0xFFFF;
    if(!metaLo || !slab) throw new Error("direct-chain target page was not published");

    const hp = w.get_hypercall_page_ptr() >>> 0;
    const data = new DataView(cpu.wasm_memory.buffer);
    const cell = new Uint16Array(
        cpu.wasm_memory.buffer,
        (w.jit_get_dispatch_slabs_ptr() >>> 0) + (slab * 0x1000) * 2,
        1,
    );
    const saved = {
        eip: cpu.instruction_pointer[0],
        inHlt: cpu.in_hlt[0],
        enabled: data.getUint32(hp + 0x008, true),
        limit: data.getUint32(hp + 0x000, true),
        cell: cell[0],
        wrongMode: cpu.get_jit_config(24) >>> 0,
    };
    const stat = i => w.profiler_dispatch_stat_get(i) >>> 0;
    const retiredBefore = w.jit_get_tier2_retired_total();
    const budgetBefore = stat(6);
    const missBefore = stat(7);
    const wrongBefore = w.jit_get_wrong_entry_chain() >>> 0;
    const retiredInputs = [7, 11, 13, 17, 19, 23, 29];

    try {
        data.setUint32(hp + 0x008, 1, true);
        cpu.instruction_pointer[0] = target;
        cpu.in_hlt[0] = 0;

        data.setUint32(hp + 0x000, 0, true);
        const urgentExit = w.jit_find_cache_entry_for_chaining(stateFlags, tableIndex, retiredInputs[0]);

        data.setUint32(hp + 0x000, 0xFFFF_FFFF, true);
        cpu.in_hlt[0] = 1;
        const hltExit = w.jit_find_cache_entry_for_chaining(stateFlags, tableIndex, retiredInputs[1]);

        cpu.in_hlt[0] = 0;
        const hit = w.jit_find_cache_entry_for_chaining(stateFlags, tableIndex, retiredInputs[2]);

        cpu.instruction_pointer[0] = target + 1;
        const noEntryMiss = w.jit_find_cache_entry_for_chaining(stateFlags, tableIndex, retiredInputs[3]);

        cpu.instruction_pointer[0] = target;
        cpu.set_jit_config(24, 2);
        const verifiedHit = w.jit_find_cache_entry_for_chaining(
            stateFlags, tableIndex, retiredInputs[4]);
        cell[0] = saved.cell === 0xFFFE ? 1 : 0xFFFE;
        const wrongEntryRefused = w.jit_find_cache_entry_for_chaining(
            stateFlags, tableIndex, retiredInputs[5]);
        cell[0] = saved.cell;
        const restoredHit = w.jit_find_cache_entry_for_chaining(
            stateFlags, tableIndex, retiredInputs[6]);

        return {
            urgentExit,
            hltExit,
            hit,
            noEntryMiss,
            verifiedHit,
            wrongEntryRefused,
            restoredHit,
            budgetExitDelta: (stat(6) - budgetBefore) >>> 0,
            missDelta: (stat(7) - missBefore) >>> 0,
            wrongEntryDelta: (w.jit_get_wrong_entry_chain() - wrongBefore) >>> 0,
            retiredDelta: w.jit_get_tier2_retired_total() - retiredBefore,
            retiredExpected: retiredInputs.reduce((a, b) => a + b, 0),
        };
    }
    finally {
        cell[0] = saved.cell;
        cpu.instruction_pointer[0] = saved.eip;
        cpu.in_hlt[0] = saved.inHlt;
        data.setUint32(hp + 0x008, saved.enabled, true);
        data.setUint32(hp + 0x000, saved.limit, true);
        cpu.set_jit_config(24, saved.wrongMode);
    }
}

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
    const rel8 = n => { patches.push({ at: o, sz: 1, end: o + 1, to: n }); emit(0); };
    const rel32 = n => { patches.push({ at: o, sz: 4, end: o + 4, to: n }); emit(0, 0, 0, 0); };

    emit(0xB9); u32(ITER);                 // mov ecx, ITER
    label("page0");
    emit(0x49);                            // dec ecx
    emit(0x74); rel8("done");             // jz done
    emit(0xE9); rel32("page1");           // jmp page1
    label("done");
    emit(0xF4); emit(0xEB, 0xFE);          // hlt; jmp $

    o = PAGE1_OFF;
    label("page1");
    emit(0xE9); rel32("page0");           // jmp page0

    for(const p of patches)
    {
        const d = labels[p.to] - p.end;
        if(p.sz === 1) buf[p.at] = d & 0xff;
        else dv.setInt32(p.at, d, true);
    }
    return buf;
}

function run(chaining)
{
    return new Promise(resolve => {
        const emulator = new V86({
            autostart: false,
            memory_size: MEM_SIZE,
            disable_jit: 0,
            log_level: 0,
        });
        let halted = false, timer, defaultChaining = 0xFFFF_FFFF;
        const finish = status => {
            clearTimeout(timer);
            try { emulator.stop(); } catch(e) {}
            const cpu = emulator.v86.cpu;
            const dget = cpu.wm.exports["profiler_dispatch_stat_get"];
            const naturalWrongEntryChain = cpu.wm.exports["jit_get_wrong_entry_chain"]?.() >>> 0;
            const naturalRetired = cpu.wm.exports["jit_get_tier2_retired_total"]?.() ?? 0;
            const instructionCounter = cpu.instruction_counter[0] >>> 0;
            const probe = chaining ? probeDirectChainInvariants(cpu) : null;
            resolve({
                status,
                ecx: cpu.reg32[1] >>> 0,
                chaining: cpu.get_jit_config ? cpu.get_jit_config(4) >>> 0 : 0,
                reentry: dget ? dget(1) : 0,
                chainableFallback: dget ? dget(2) : 0,
                chainedEdge: dget ? dget(5) : 0,
                budgetExit: dget ? dget(6) : 0,
                miss: dget ? dget(7) : 0,
                instructionCounter,
                naturalRetired,
                naturalWrongEntryChain,
                chainEntries: cpu.wm.exports["jit_get_tier2_chain_entries"]?.() ?? 0,
                defaultChaining,
                probe,
            });
        };

        emulator.bus.register("cpu-event-halt", () => {
            halted = true;
            finish("halt");
        });
        emulator.add_listener("emulator-loaded", () => {
            const cpu = emulator.v86.cpu;
            cpu.reboot_internal();
            cpu.reset_memory();
            defaultChaining = cpu.get_jit_config(4) >>> 0;
            cpu.set_jit_config(1, 1); // MAX_PAGES=1, force cross-page module exits
            cpu.set_jit_config(4, chaining ? 1 : 0);
            cpu.set_jit_config(15, 0xFFFF_FFFF); // emit accounting, never promote this fixture
            cpu.set_jit_config(24, 0); // production inline path; probe covers verifier/refusal
            cpu.wm.exports["set_dispatch_stats"]?.(1);
            cpu.wm.exports["profiler_init"]?.();
            cpu.jit_clear_cache?.();
            cpu.load_multiboot(build_image().buffer);
            timer = setTimeout(() => { if(!halted) finish("HANG"); }, TIMEOUT_MS);
            emulator.run();
        });
    });
}

const off = await run(false);
const on = await run(true);
console.log("jit-block-chaining " + JSON.stringify({ off, on }));

if(off.status !== "halt" || off.ecx !== 0 || on.status !== "halt" || on.ecx !== 0)
{
    console.error("FAIL: OFF/ON loop did not halt cleanly with ecx=0");
    process.exit(1);
}

if(off.defaultChaining !== 0 || on.defaultChaining !== 0 ||
   off.chaining !== 0 || on.chaining !== 1)
{
    console.error("FAIL: idx 4 did not report the requested OFF/ON state");
    process.exit(1);
}

if(off.chainEntries !== 0 || on.chainEntries <= 0)
{
    console.error("FAIL: direct chained-entry accounting is inconsistent");
    process.exit(1);
}

if(off.chainedEdge !== 0 || on.chainedEdge <= 0)
{
    console.error("FAIL: idx 4 kill switch/effectiveness counters are inconsistent");
    process.exit(1);
}

if(on.reentry >= off.reentry)
{
    console.error("FAIL: chaining did not reduce module re-entry");
    process.exit(1);
}

if(off.naturalRetired <= 0 || on.naturalRetired <= 0 ||
   off.naturalRetired > off.instructionCounter || on.naturalRetired > on.instructionCounter)
{
    console.error("FAIL: retired accounting is empty or exceeds architectural instructions");
    process.exit(1);
}

if(off.naturalWrongEntryChain !== 0 || on.naturalWrongEntryChain !== 0)
{
    console.error("FAIL: normal OFF/ON execution triggered the wrong-entry detector");
    process.exit(1);
}

const p = on.probe;
if(!p || p.urgentExit !== -1 || p.hltExit !== -1 || p.hit < 0 || p.verifiedHit < 0 ||
   p.noEntryMiss !== -1 || p.wrongEntryRefused !== -1 || p.restoredHit < 0 ||
   p.budgetExitDelta !== 2 || p.missDelta !== 2 || p.wrongEntryDelta !== 1 ||
   p.retiredDelta !== p.retiredExpected)
{
    console.error("FAIL: direct-chain guard/fallback/accounting invariant failed: " + JSON.stringify(p));
    process.exit(1);
}

process.exit(0);
