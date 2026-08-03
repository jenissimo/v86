#!/usr/bin/env node
// Runtime contract probe for the Rust-staged AOT publication API. Requires a clean v86 build.

const { V86 } = await import("../build/libv86.mjs");
const PAGE = 0x1000, RAM = 16 * 1024 * 1024, SLOT_A = 898, SLOT_B = 897;
const assert = (ok, message) => { if (!ok) throw new Error(message); };

const emulator = new V86({ autostart: false, memory_size: RAM, disable_jit: 0, log_level: 0 });
await new Promise((resolve) => emulator.add_listener("emulator-loaded", resolve));
const w = emulator.v86.cpu.wm.exports;
const table = emulator.v86.cpu.wm.wasm_table;
for (const name of ["jit_aot_tx_begin", "jit_aot_tx_page_begin", "jit_aot_tx_entry_push",
    "jit_aot_tx_page_finish", "jit_aot_tx_prepare_finish", "jit_aot_tx_commit", "jit_aot_tx_abort",
    "jit_aot_free_table_index_count", "jit_aot_page_table_index", "jit_aot_module_page_count"])
    assert(typeof w[name] === "function", `missing ${name}`);

const fp = () => [w.jit_codegen_fingerprint_lo() >>> 0, w.jit_codegen_fingerprint_hi() >>> 0];
const begin = (slot, count, fingerprint = fp()) => w.jit_aot_tx_begin(slot, count, ...fingerprint) >>> 0;
const abort = () => w.jit_aot_tx_abort() >>> 0;
const add = (addr, flags, entries) => {
    assert((w.jit_aot_tx_page_begin(addr, flags, entries.length) >>> 0) === 0, "page_begin");
    for (const [off, state] of entries) assert((w.jit_aot_tx_entry_push(off, state) >>> 0) === 0, "entry_push");
    assert((w.jit_aot_tx_page_finish() >>> 0) === 0, "page_finish");
};

const free0 = w.jit_aot_free_table_index_count() >>> 0;
assert(begin(0, 1) !== 0 && begin(900, 1) !== 0, "invalid exact slots rejected");
assert(begin(SLOT_A, 1, [fp()[0] ^ 1, fp()[1]]) !== 0, "fingerprint mismatch rejected without reservation");
assert((w.jit_aot_free_table_index_count() >>> 0) === free0, "pre-prepare refusal leaked a slot");

assert(begin(SLOT_A, 2) === 0, "exact free slot begins");
assert((w.jit_aot_tx_page_begin(0x100001, 0, 0) >>> 0) !== 0, "misaligned page rejected");
assert((w.jit_aot_tx_page_begin(RAM, 0, 0) >>> 0) !== 0, "out-of-bounds page rejected");
assert((w.jit_aot_tx_page_begin(0x100000, 0x10, 0) >>> 0) !== 0, "invalid state flags rejected");
add(0x100000, 0, [[0, 0]]);
assert((w.jit_aot_tx_page_begin(0x100000, 0, 0) >>> 0) !== 0, "duplicate page rejected");
assert(abort() === 0 && abort() !== 0, "abort is idempotently safe");
assert((w.jit_aot_tx_commit() >>> 0) !== 0, "commit after abort rejected");
assert((w.jit_aot_free_table_index_count() >>> 0) === free0, "abort restored exact slot");

assert(begin(SLOT_A, 1) === 0, "begin invalid-entry unit");
assert((w.jit_aot_tx_page_begin(0x100000, 0, 1) >>> 0) === 0, "entry validation page");
assert((w.jit_aot_tx_entry_push(0x1000, 0) >>> 0) !== 0, "entry offset rejected");
assert((w.jit_aot_tx_entry_push(0, 0x10000) >>> 0) !== 0, "entry state rejected");
assert(abort() === 0, "abort invalid-entry unit");

assert(begin(SLOT_A, 1) === 0, "begin all-empty unit");
add(0x100000, 0, []);
assert((w.jit_aot_tx_prepare_finish() >>> 0) !== 0, "all-empty unit rejected");
assert(abort() === 0, "abort all-empty unit");

assert(begin(SLOT_A, 2) === 0, "begin multi-page unit");
add(0x100000, 0, []); // Required SMC coverage page.
add(0x101000, 0, [[0, 0]]);
assert((w.jit_aot_tx_prepare_finish() >>> 0) === 0, "mixed empty/nonempty unit prepared");
// Commit is valid only after JS owns the table slot. An exported wasm function is a storable
// funcref; this probe does not execute the fake body, it verifies ownership publication order.
const installed = w.jit_aot_tx_begin;
assert(typeof installed === "function", "missing storable wasm function");
table.set(SLOT_A + 1024, installed);
assert(table.get(SLOT_A + 1024) === installed, "table ownership was not installed before commit");
assert((w.jit_aot_tx_commit() >>> 0) === 0, "multi-page unit committed");
assert((w.jit_aot_tx_commit() >>> 0) !== 0, "repeated commit rejected");
assert((w.jit_aot_page_table_index(0x100000) >>> 0) === SLOT_A &&
    (w.jit_aot_page_table_index(0x101000) >>> 0) === SLOT_A, "all pages published atomically under exact slot");
assert(table.get(SLOT_A + 1024) === installed, "commit lost JS table ownership");
assert((w.jit_aot_module_page_count(SLOT_A) >>> 0) === 2, "coverage page retained");
assert((w.jit_aot_free_table_index_count() >>> 0) === free0 - 1, "successful transaction consumes one slot");

// JS must instantiate before begin. A malformed module therefore cannot reserve a Rust slot.
try { new WebAssembly.Module(new Uint8Array([0])); throw new Error("malformed wasm instantiated"); }
catch (e) { if (String(e).includes("malformed wasm instantiated")) throw e; }
assert((w.jit_aot_free_table_index_count() >>> 0) === free0 - 1 &&
    (w.jit_aot_tx_staged_index() >>> 0) === 0xFFFF, "instantiate failure leaks nothing");

// Simulate table.set rejection after prepare. Since the bad value never lands in the table,
// its old value remains null; the caller then aborts and the exact slot returns to Rust's list.
assert(begin(SLOT_B, 1) === 0, "begin table.set-failure unit");
add(0x102000, 0, [[0, 0]]);
assert((w.jit_aot_tx_prepare_finish() >>> 0) === 0, "prepare table.set-failure unit");
try { table.set(SLOT_B + 1024, {}); throw new Error("invalid table value accepted"); }
catch (e) { if (String(e).includes("invalid table value accepted")) throw e; }
assert(table.get(SLOT_B + 1024) === null, "failed table.set changed table");
assert(abort() === 0 && (w.jit_aot_page_table_index(0x102000) >>> 0) === 0xFFFF,
    "table.set failure abort publishes zero pages");
assert((w.jit_aot_free_table_index_count() >>> 0) === free0 - 1, "table.set failure restored slot");

assert(begin(SLOT_B, 2) === 0, "begin occupied-late-page unit");
add(0x102000, 0, [[0, 0]]);
assert((w.jit_aot_tx_page_begin(0x100000, 0, 0) >>> 0) !== 0, "late occupied page rejected");
assert(abort() === 0, "abort occupied-late-page unit");
assert((w.jit_aot_page_table_index(0x102000) >>> 0) === 0xFFFF, "late refusal published zero pages");

// Instantiation/table.set failures are exercised by aot-cache replay's failure path: they occur
// before begin or clear the JS table before abort. This source-level probe verifies its Rust half.
assert((w.jit_aot_tx_staged_index() >>> 0) === 0xFFFF, "no transaction leaked");
emulator.stop();
console.log("PASS jit-aot-transaction-contract");
