#!/usr/bin/env node
// Raw/bulk-memory boundary contract; runs only against a freshly built v86 artifact.
const { V86 } = await import("../build/libv86.mjs");
const RAM = 16 * 1024 * 1024;
const assert = (ok, msg) => { if (!ok) throw new Error(msg); };
const emu = new V86({ autostart: false, memory_size: RAM, disable_jit: 1, log_level: 0 });
await new Promise(resolve => emu.add_listener("emulator-loaded", resolve));
const cpu = emu.v86.cpu, w = cpu.wm.exports, mem = cpu.mem8;
for (const n of [
    "zero_memory", "is_memory_zeroed", "memory_get_oob_writes", "memory_get_oob_info",
    "memory_raw_read8", "memory_raw_read32", "memory_raw_write8", "memory_raw_write32", "memory_raw_memcpy",
]) assert(typeof w[n] === "function", `missing ${n}`);
const before = w.memory_get_oob_writes() >>> 0;

// Valid direct reads and writes, including a naturally unaligned i32 load/store at RAM's end.
mem.set([0x12, 0x34, 0x56, 0x78], 0x100);
assert((w.memory_raw_read8(0x100) >>> 0) === 0x12, "valid raw read8 failed");
assert((w.memory_raw_read32(0x100) >>> 0) === 0x78563412, "valid raw read32 failed");
w.memory_raw_write8(RAM - 1, 0xA5);
assert(mem[RAM - 1] === 0xA5, "valid end-of-RAM raw write8 failed");
w.memory_raw_write32(RAM - 4, 0x11223344);
assert(mem[RAM - 4] === 0x44 && mem[RAM - 1] === 0x11, "valid end-of-RAM raw write32 failed");

// Reads fail closed (zero) and writes preserve their sentinel for crossing-end and u32-wrap.
// Completing this test proves the rejected calls did not host-trap.
assert((w.memory_raw_read32(RAM - 1) >>> 0) === 0, "crossing-end raw read32 was accepted");
assert((w.memory_raw_read32(0xFFFF_FFFE) >>> 0) === 0, "u32-wrap raw read32 was accepted");
mem[RAM - 1] = 0xA5;
w.memory_raw_write32(RAM - 1, 0x11223344);
w.memory_raw_write32(0xFFFF_FFFE, 0x55667788);
assert(mem[RAM - 1] === 0xA5, "OOB raw write changed end sentinel");

// Valid bulk copy and ptr::copy (memmove) overlap semantics in both directions.
mem.set([1, 2, 3, 4, 5, 6], 0x200);
w.memory_raw_memcpy(0x200, 0x240, 6);
assert([...mem.subarray(0x240, 0x246)].join(",") === "1,2,3,4,5,6", "valid raw memcpy failed");
mem.set([1, 2, 3, 4, 5, 6], 0x280);
w.memory_raw_memcpy(0x280, 0x282, 4);
assert([...mem.subarray(0x280, 0x286)].join(",") === "1,2,1,2,3,4", "forward-overlap memcpy was not memmove");
mem.set([1, 2, 3, 4, 5, 6], 0x2A0);
w.memory_raw_memcpy(0x2A2, 0x2A0, 4);
assert([...mem.subarray(0x2A0, 0x2A6)].join(",") === "3,4,5,6,5,6", "backward-overlap memcpy was not memmove");

// Neither invalid side of bulk copy is dereferenced. A zero count permits exactly RAM-end, but
// does not waive address validation for an out-of-range endpoint.
mem[0x300] = 0xA1; mem[RAM - 1] = 0xB2;
w.memory_raw_memcpy(RAM - 2, 0x300, 4);
w.memory_raw_memcpy(0x200, RAM - 2, 4);
assert(mem[0x300] === 0xA1 && mem[RAM - 1] === 0xB2, "OOB memcpy changed a sentinel");
w.memory_raw_memcpy(RAM, RAM, 0);
w.memory_raw_memcpy(RAM + 1, RAM, 0);
w.memory_raw_memcpy(RAM, 0xFFFF_FFFE, 0);

// Existing exported bulk-zero path follows the same fail-closed convention.
mem[RAM - 8] = 0xA5; mem[RAM - 1] = 0x5A;
w.zero_memory(RAM - 8, 8);
assert(mem[RAM - 8] === 0 && mem[RAM - 1] === 0, "valid end-of-RAM raw zero failed");
mem[RAM - 1] = 0xA5;
w.zero_memory(RAM - 1, 2);
w.zero_memory(0xFFFF_FFF8, 16);
assert(mem[RAM - 1] === 0xA5, "OOB raw write changed sentinel");
assert((w.memory_get_oob_writes() >>> 0) >= before + 8, "OOB diagnostics did not increment for every raw/bulk refusal");
assert((w.memory_get_oob_info(0) >>> 0) === 0xFFFF_FFF8 && (w.memory_get_oob_info(1) >>> 0) === 16,
    "OOB diagnostics did not retain the final raw bulk request");
assert((w.is_memory_zeroed(RAM, 0) !== 0), "exact-end zero length invalid");
assert((w.is_memory_zeroed(RAM + 1, 0) === 0) && (w.is_memory_zeroed(RAM, 8) === 0)
    && (w.is_memory_zeroed(0xFFFF_FFF8, 8) === 0), "invalid raw zero checks accepted");
emu.stop();
console.log("PASS memory-oob-contract");
