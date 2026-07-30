#!/usr/bin/env node
// A reset can re-seed the unified-clock page just after set_tsc(0) sampled its old base.
// The raw source then steps briefly backwards. RDTSC must advance from zero, not expose the
// wrapping subtraction as a near-u64::MAX timestamp.

const { V86 } = await import("../build/libv86.mjs");

const BASE = 0x100000;
const ENTRY = BASE + 0x40;
const OUT_LO = BASE + 0x3000;
const OUT_HI = OUT_LO + 4;

function buildImage()
{
    const buf = new Uint8Array(0x4000);
    const dv = new DataView(buf.buffer);
    const MAGIC = 0x1BADB002;
    const FLAGS = 0x10000;
    dv.setUint32(0x00, MAGIC, true);
    dv.setUint32(0x04, FLAGS, true);
    dv.setUint32(0x08, (-(MAGIC + FLAGS)) >>> 0, true);
    dv.setUint32(0x0c, BASE, true);
    dv.setUint32(0x10, BASE, true);
    dv.setUint32(0x14, BASE + buf.length, true);
    dv.setUint32(0x18, BASE + buf.length, true);
    dv.setUint32(0x1c, ENTRY, true);

    let o = ENTRY - BASE;
    const emit = (...bytes) => { for(const b of bytes) buf[o++] = b; };
    const imm32 = value => { dv.setUint32(o, value >>> 0, true); o += 4; };
    emit(0xBC); imm32(0x200000);            // mov esp, 0x200000
    emit(0x0F, 0x31);                       // rdtsc
    emit(0xA3); imm32(OUT_LO);              // mov [OUT_LO], eax
    emit(0x89, 0x15); imm32(OUT_HI);        // mov [OUT_HI], edx
    emit(0xF4);                             // hlt
    emit(0xEB, 0xFE);
    return buf;
}

const result = await new Promise(resolve => {
    const emulator = new V86({
        autostart: false,
        memory_size: 16 * 1024 * 1024,
        disable_jit: 0,
        log_level: 0,
    });
    let timer;
    const finish = status => {
        clearTimeout(timer);
        const cpu = emulator.v86.cpu;
        const mem = cpu.mem8;
        const lo = mem[OUT_LO] | mem[OUT_LO + 1] << 8 |
            mem[OUT_LO + 2] << 16 | mem[OUT_LO + 3] << 24;
        const hi = mem[OUT_HI] | mem[OUT_HI + 1] << 8 |
            mem[OUT_HI + 2] << 16 | mem[OUT_HI + 3] << 24;
        try { emulator.stop(); } catch {}
        resolve({ status, lo: lo >>> 0, hi: hi >>> 0 });
    };

    emulator.bus.register("cpu-event-halt", () => finish("halt"));
    emulator.add_listener("emulator-loaded", () => {
        const cpu = emulator.v86.cpu;
        cpu.reboot_internal();
        cpu.reset_memory();
        cpu.load_multiboot(buildImage().buffer);

        const hp = cpu.wm.exports.get_hypercall_page_ptr() >>> 0;
        const page = new DataView(cpu.wasm_memory.buffer);
        const writeClock = micros => {
            page.setUint32(hp + 0x000, 100_003, true); // cycle_limit
            page.setUint32(hp + 0x008, 1, true); // hc_enabled
            page.setUint32(hp + 0x014, micros >>> 0, true);
            page.setUint32(hp + 0x018, Math.floor(micros / 0x100000000), true);
            page.setUint32(hp + 0x02c, cpu.instruction_counter[0] >>> 0, true);
            page.setUint32(hp + 0x030, 100, true);
        };

        writeClock(100_000_000);
        cpu.set_tsc(0, 0);
        writeClock(99_000_000); // one-second backward re-seed after set_tsc

        timer = setTimeout(() => finish("HANG"), 5000);
        emulator.run();
    });
});

const ok = result.status === "halt" && result.hi === 0 && result.lo > 0 && result.lo < 0x100000;
console.log(`${ok ? "PASS" : "FAIL"} tsc-backstep status=${result.status} value=0x${result.hi.toString(16).padStart(8, "0")}${result.lo.toString(16).padStart(8, "0")}`);
process.exit(ok ? 0 : 1);
