#!/usr/bin/env node
// Corrupt exactly one generated JIT module, then prove the next hot page still publishes.

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
    const rel32 = n => { patches.push({ at: o, end: o + 4, to: n }); emit(0, 0, 0, 0); };

    emit(0xBC); u32(BASE + 0x3000);
    emit(0xB9); u32(ITER);
    label("loop");
    emit(0xE8); rel32("callee");
    emit(0x49);
    emit(0x0F, 0x85); rel32("loop");
    emit(0xF4); emit(0xEB, 0xFE);

    o = PAGE1_OFF;
    label("callee");
    emit(0xC3);

    for(const p of patches)
    {
        dv.setInt32(p.at, labels[p.to] - p.end, true);
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
        let halted = false, timer, finalized = 0, corruptions = 0;
        let get_failure_count, get_failure_info, get_compiling, exports_ready = false;
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
                generated: globalThis.__jitCompileStats.count | 0,
                finalized,
                published,
                corruptions,
                exports_ready,
                failures: get_failure_count ? get_failure_count() >>> 0 : -1,
                last_failure_kind: get_failure_info ? get_failure_info(2) >>> 0 : -1,
                last_recovery_status: get_failure_info ? get_failure_info(3) >>> 0 : -1,
                compiling: get_compiling ? get_compiling() >>> 0 : -1,
                ret_chain_hit: dget ? dget(11) >>> 0 : 0,
            });
        };

        emulator.bus.register("cpu-event-halt", () => { halted = true; finish("halt"); });
        emulator.add_listener("emulator-loaded", () => {
            const cpu = emulator.v86.cpu;
            get_failure_count = cpu.wm.exports["codegen_get_finalize_failure_count"];
            get_failure_info = cpu.wm.exports["codegen_get_finalize_failure_info"];
            get_compiling = cpu.wm.exports["codegen_is_compiling"];
            exports_ready = typeof get_failure_count === "function" &&
                typeof get_failure_info === "function" && typeof get_compiling === "function";
            if(!exports_ready)
            {
                finish("STALE_WASM_EXPORTS");
                return;
            }

            cpu.reboot_internal();
            cpu.reset_memory();
            cpu.set_jit_config(1, 1);
            cpu.set_jit_config(12, 1);
            cpu.wm.exports["set_dispatch_stats"]?.(1);
            cpu.wm.exports["profiler_init"]?.();
            cpu.jit_clear_cache?.();
            // The CPU hook receives an isolated instantiation copy; corrupt exactly one
            // candidate without poisoning the reusable Rust wasm-builder buffer.
            let inject_one_invalid_module = true;
            cpu.test_hook_did_generate_wasm = code => {
                if(inject_one_invalid_module)
                {
                    inject_one_invalid_module = false;
                    corruptions++;
                    code[0] ^= 0xff;
                }
            };
            cpu.test_hook_did_finalize_wasm = () => { finalized++; };
            cpu.load_multiboot(build_image().buffer);
            timer = setTimeout(() => { if(!halted) finish("HANG"); }, TIMEOUT_MS);
            emulator.run();
        });
    });
}

const r = await run();
console.log("jit-recovery " + JSON.stringify(r));

const fail = msg => { console.error("FAIL: " + msg); process.exit(1); };

if(!r.exports_ready) fail("stale wasm: missing codegen failure-recovery diagnostic exports");
if(r.status !== "halt" || r.ecx !== 0) fail("loop did not halt cleanly with ecx=0");
if(r.corruptions !== 1) fail("expected exactly one corrupted generated module, got " + r.corruptions);
if(r.failures !== 1 || r.last_failure_kind !== 1 || r.last_recovery_status !== 0)
{
    fail("expected one recovered CompileError, got failures=" + r.failures +
         " kind=" + r.last_failure_kind + " recovery=" + r.last_recovery_status);
}
if(r.compiling !== 0) fail("JIT compiling slot remained occupied after the rejected module");
if(r.generated <= 1) fail("recovery did not generate a later module (generated=" + r.generated + ")");
if(r.finalized <= 0 || r.published <= 0)
{
    fail("no later module finalized/published (finalized=" + r.finalized +
         ", published=" + r.published + ")");
}
if(r.ret_chain_hit <= 0) fail("JIT execution evidence stayed zero (ret_chain_hit=0)");

console.log(`  ok — generated=${r.generated} finalized=${r.finalized} published=${r.published} ` +
            `failures=${r.failures} ret_chain_hit=${r.ret_chain_hit}`);
