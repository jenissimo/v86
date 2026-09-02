// Post-build tripwire: assert that #[no_mangle] symbols from workspace crates
// (crates/d3d9-webgpu) survived into the final cdylib's wasm exports. rustc keeps
// no_mangle rlib symbols through LTO today, but that is a toolchain behavior, not a
// contract — if it ever regresses (or wasm-opt GC is added), this fails the build
// instead of manifesting as "Missing import: <name>" at worker startup.
// Usage: node tools/check-wasm-exports.mjs <path-to-v86.wasm>
import { readFileSync } from 'node:fs';

const REQUIRED = [
    // crates/d3d9-webgpu (arena.rs)
    'get_d3d9_arena_ptr',
    'get_d3d9_arena_layout_ptr',
    'get_d3d9_arena_abi_version',
    'd3d9_reset_frame',
    'd3d9_record_draw',
    'd3d9_record_draw_indexed',
    'd3d9_record_wbuf_indexed_run',
    'd3d9_record_draw_up',
    'd3d9_record_draw_indexed_up',
    'd3d9_block_capture',
    'd3d9_block_apply',
];

const wasmPath = process.argv[2];
if (!wasmPath) {
    console.error('usage: node tools/check-wasm-exports.mjs <v86.wasm>');
    process.exit(2);
}

const mod = new WebAssembly.Module(readFileSync(wasmPath));
const names = new Set(WebAssembly.Module.exports(mod).map(e => e.name));
const missing = REQUIRED.filter(n => !names.has(n));

if (missing.length) {
    console.error(`FAIL: ${wasmPath} is missing required exports: ${missing.join(', ')}`);
    console.error('Likely cause: LTO/linker GC dropped #[no_mangle] symbols from a workspace crate.');
    console.error('Fix: add thin #[no_mangle] shims in src/rust/lib.rs that call into the crate.');
    process.exit(1);
}
console.log(`  export check OK (${REQUIRED.length} required symbols present)`);
