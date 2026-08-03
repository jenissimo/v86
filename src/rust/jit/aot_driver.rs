// AOT offline driver — SPIKE (feature `aot-driver` only; absent from the shipped engine).
//
// The whole point: reuse jit_find_basic_blocks / control_flow / jit_generate_module
// UNCHANGED, so instruction semantics, flag protocol, memory shapes and the exit contract
// are the engine's own. This module is jit_analyze_and_generate minus publication:
// no free-list pop, no ctx.pages mutation, no tlb_set_has_code, no codegen_finalize.
//
// Exports are prefixed aot_drv_ so they cannot collide with the shipped jit_aot_* surface.
//
// No wasm build links it, so `make check-aot-driver` is the only thing keeping it compiling
// against the jit.rs internals it reaches into.

use super::*;
use std::ptr::{addr_of, addr_of_mut};

static mut ENTRY_OFFSETS: Vec<u16> = Vec::new();
static mut OUT_ENTRIES: Vec<(u32, u16)> = Vec::new();
static mut OUT_BYTES: Vec<u8> = Vec::new();
static mut BUILDER: Option<WasmBuilder> = None;
static mut LAST_BLOCK_COUNT: u32 = 0;
static mut LAST_PAGE_COUNT: u32 = 0;

#[no_mangle]
pub fn aot_drv_reset() {
    unsafe {
        (*addr_of_mut!(ENTRY_OFFSETS)).clear();
        (*addr_of_mut!(OUT_ENTRIES)).clear();
        (*addr_of_mut!(OUT_BYTES)).clear();
    }
}

#[no_mangle]
pub fn aot_drv_push_entry(offset: u32) {
    unsafe { (*addr_of_mut!(ENTRY_OFFSETS)).push((offset & 0xFFF) as u16) }
}

/// Seed the entry-point set from the engine's own `ctx.entry_points` for a physical page —
/// the same set `jit_analyze_and_generate` would have analysed. Returns how many THIS call
/// appended (the queue accumulates until `aot_drv_reset`).
/// (S0 in the design: the job's entry list is the engine's, never guessed.)
#[no_mangle]
pub fn aot_drv_seed_from_ctx(phys_page_addr: u32) -> u32 {
    let ctx = get_jit_state();
    let page = Page::page_of(phys_page_addr);
    match ctx.entry_points.get(&page) {
        None => 0,
        Some((_, entries)) => unsafe {
            let queue = &mut *addr_of_mut!(ENTRY_OFFSETS);
            for e in entries.iter() {
                queue.push(*e);
            }
            entries.len() as u32
        },
    }
}

/// Seed `ctx.entry_points` for a page. REQUIRED, not optional: `follow_jump` re-derives each
/// page's entry set from `ctx.entry_points` and blacklists a page that is absent from it (or
/// whose entries are all already published) — so an offline compile of an unpublished page
/// needs this state reconstructed, not merely an entry-point argument (design S1).
#[no_mangle]
pub fn aot_drv_add_ctx_entry(phys_page_addr: u32, offset: u32) {
    let mut ctx = get_jit_state();
    let page = Page::page_of(phys_page_addr);
    ctx.entry_points
        .entry(page)
        .or_insert_with(|| (0, HashSet::new()))
        .1
        .insert((offset & 0xFFF) as u16);
}

/// Is this page currently owned by a published module? A driver compile of an owned page is
/// refused by the analyzer (empty block set), so the host must know before it calls.
#[no_mangle]
pub fn aot_drv_page_owned(phys_page_addr: u32) -> u32 {
    let ctx = get_jit_state();
    if ctx.pages.contains_key(&Page::page_of(phys_page_addr)) { 1 } else { 0 }
}

/// Compile one page (or page set, if the analyzer follows edges off-page) offline.
/// Takes both addresses for the same reason `jit_analyze_and_generate` does: the analyzer
/// works in VIRTUAL addresses (the queued offsets rebase onto `virt_entry`'s page), while
/// `ctx.pages` / `ctx.entry_points` are keyed by PHYSICAL page. Passing one for the other
/// happens to work under an identity map and silently misses otherwise.
/// Returns the emitted module length in bytes, or 0 on refusal.
#[no_mangle]
pub fn aot_drv_compile(
    virt_entry: i32,
    phys_entry: u32,
    cs_offset: u32,
    state_flags: u32,
    table_index: u32,
) -> u32 {
    let mut ctx = get_jit_state();
    let ctx = &mut *ctx;
    let state_flags = CachedStateFlags::of_u32(state_flags);

    let cpu = CpuContext { eip: 0, prefixes: 0, cs_offset, state_flags };

    let virt_page = Page::page_of(virt_entry as u32);
    let entry_points: HashSet<i32> = unsafe {
        (*addr_of!(ENTRY_OFFSETS))
            .iter()
            .map(|e| virt_page.to_address() as i32 | *e as i32)
            .collect()
    };
    if entry_points.is_empty() {
        return 0;
    }

    // The analyzer's tail does `for i in 0..basic_blocks.len() - 1` (jit.rs:2671) — a usize
    // underflow if the analysis yields nothing, which the JIT can never observe because it only
    // ever calls with a heated, unpublished page. The offline driver CAN, so refuse first.
    // Same predicate as aot_drv_page_owned, on the same physical key.
    if ctx.pages.contains_key(&Page::page_of(phys_entry)) {
        return 0;
    }

    let basic_blocks = jit_find_basic_blocks(ctx, entry_points, cpu.clone());
    if basic_blocks.is_empty() {
        return 0;
    }

    let mut pages = HashSet::new();
    for b in basic_blocks.iter() {
        pages.insert(Page::page_of(b.addr));
    }

    let graph = control_flow::make_graph(&basic_blocks);
    let mut structure = control_flow::loopify(&graph);
    control_flow::blockify(&mut structure, &graph);

    let basic_block_by_addr: HashMap<u32, BasicBlock> =
        basic_blocks.into_iter().map(|b| (b.addr, b)).collect();

    unsafe {
        LAST_BLOCK_COUNT = basic_block_by_addr.len() as u32;
        LAST_PAGE_COUNT = pages.len() as u32;
    }


    // Own builder instance: the driver must not disturb the live JIT's builder state.
    let builder = unsafe {
        let builder = &mut *addr_of_mut!(BUILDER);
        if builder.is_none() {
            *builder = Some(WasmBuilder::new());
        }
        builder.as_mut().unwrap()
    };

    let entries = jit_generate_module(
        structure,
        &basic_block_by_addr,
        cpu,
        builder,
        WasmTableIndex(table_index as u16),
        state_flags,
    );

    unsafe {
        *addr_of_mut!(OUT_ENTRIES) = entries;
        let len = builder.get_output_len() as usize;
        let ptr = builder.get_output_ptr();
        let out = &mut *addr_of_mut!(OUT_BYTES);
        *out = std::slice::from_raw_parts(ptr, len).to_vec();
        out.len() as u32
    }
}

#[no_mangle]
pub fn aot_drv_output_ptr() -> u32 { unsafe { (*addr_of!(OUT_BYTES)).as_ptr() as u32 } }

#[no_mangle]
pub fn aot_drv_output_len() -> u32 { unsafe { (*addr_of!(OUT_BYTES)).len() as u32 } }

#[no_mangle]
pub fn aot_drv_entry_count() -> u32 { unsafe { (*addr_of!(OUT_ENTRIES)).len() as u32 } }

#[no_mangle]
pub fn aot_drv_entry_addr(i: u32) -> u32 { unsafe { (&*addr_of!(OUT_ENTRIES))[i as usize].0 } }

#[no_mangle]
pub fn aot_drv_entry_state(i: u32) -> u32 {
    unsafe { (&*addr_of!(OUT_ENTRIES))[i as usize].1 as u32 }
}

#[no_mangle]
pub fn aot_drv_block_count() -> u32 { unsafe { LAST_BLOCK_COUNT } }

#[no_mangle]
pub fn aot_drv_page_count() -> u32 { unsafe { LAST_PAGE_COUNT } }
