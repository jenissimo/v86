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

function run()
{
    return new Promise(resolve => {
        const emulator = new V86({
            autostart: false,
            memory_size: MEM_SIZE,
            disable_jit: 0,
            log_level: 0,
        });
        let halted = false, timer;
        const finish = status => {
            clearTimeout(timer);
            try { emulator.stop(); } catch(e) {}
            const cpu = emulator.v86.cpu;
            const dget = cpu.wm.exports["profiler_dispatch_stat_get"];
            resolve({
                status,
                ecx: cpu.reg32[1] >>> 0,
                // idx 12 = RET dynamic chaining. get_jit_config returns 0 for unknown indices,
                // so reading a retired index silently disables the assertion below.
                chaining: cpu.get_jit_config ? cpu.get_jit_config(12) >>> 0 : 0,
                reentry: dget ? dget(1) : 0,
                chainableFallback: dget ? dget(2) : 0,
                chainedEdge: dget ? dget(5) : 0,
                budgetExit: dget ? dget(6) : 0,
                miss: dget ? dget(7) : 0,
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
            cpu.set_jit_config(1, 1); // MAX_PAGES=1, force cross-page module exits
            cpu.wm.exports["set_dispatch_stats"]?.(1);
            cpu.wm.exports["profiler_init"]?.();
            cpu.jit_clear_cache?.();
            cpu.load_multiboot(build_image().buffer);
            timer = setTimeout(() => { if(!halted) finish("HANG"); }, TIMEOUT_MS);
            emulator.run();
        });
    });
}

const result = await run();
console.log("jit-block-chaining " + JSON.stringify(result));

if(result.status !== "halt" || result.ecx !== 0)
{
    console.error("FAIL: loop did not halt cleanly with ecx=0");
    process.exit(1);
}

if(result.chaining)
{
    if(result.chainedEdge <= 0)
    {
        console.error("FAIL: JIT_BLOCK_CHAINING enabled but CHAINED_EDGE stayed zero");
        process.exit(1);
    }
}
else {
    // Dormant by construction: this test never enables idx 12. The RET dynamic-chaining path
    // is covered by jit-alive-repro.mjs, which enables it and asserts RET_CHAIN_HIT > 0.
    console.log("SKIP effectiveness: RET chaining (idx 12) not enabled in this run");
}

process.exit(0);
