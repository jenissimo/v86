// AOT offline driver — SPIKE (feature `aot-driver` only; absent from the shipped engine).
//
// The whole point: reuse jit_find_basic_blocks / control_flow / jit_generate_module
// UNCHANGED, so instruction semantics, flag protocol, memory shapes and the exit contract
// are the engine's own. This module is jit_analyze_and_generate minus publication:
// no free-list pop, no ctx.pages mutation, no tlb_set_has_code, no codegen_finalize.
//
// Exports are prefixed aot_drv_ so they cannot collide with the shipped jit_aot_* surface.

use super::*;

static mut ENTRY_OFFSETS: Vec<u16> = Vec::new();
static mut OUT_ENTRIES: Vec<(u32, u16)> = Vec::new();
static mut OUT_BYTES: Vec<u8> = Vec::new();
static mut BUILDER: Option<WasmBuilder> = None;
static mut LAST_BLOCK_COUNT: u32 = 0;
static mut LAST_PAGE_COUNT: u32 = 0;

#[no_mangle]
pub fn aot_drv_reset() {
    unsafe {
        ENTRY_OFFSETS.clear();
        OUT_ENTRIES.clear();
        OUT_BYTES.clear();
    }
}

#[no_mangle]
pub fn aot_drv_push_entry(offset: u32) {
    unsafe { ENTRY_OFFSETS.push((offset & 0xFFF) as u16) }
}

/// Seed the entry-point set from the engine's own `ctx.entry_points` for a physical page —
/// the same set `jit_analyze_and_generate` would have analysed. Returns how many were seeded.
/// (S0 in the design: the job's entry list is the engine's, never guessed.)
#[no_mangle]
pub fn aot_drv_seed_from_ctx(phys_addr: u32) -> u32 {
    let ctx = get_jit_state();
    let page = Page::page_of(phys_addr);
    match ctx.entry_points.get(&page) {
        None => 0,
        Some((_, entries)) => unsafe {
            for e in entries.iter() {
                ENTRY_OFFSETS.push(*e);
            }
            ENTRY_OFFSETS.len() as u32
        },
    }
}

/// Seed `ctx.entry_points` for a page. REQUIRED, not optional: `follow_jump` re-derives each
/// page's entry set from `ctx.entry_points` and blacklists a page that is absent from it (or
/// whose entries are all already published) — so an offline compile of an unpublished page
/// needs this state reconstructed, not merely an entry-point argument (design S1).
#[no_mangle]
pub fn aot_drv_add_ctx_entry(phys_addr: u32, offset: u32) {
    let mut ctx = get_jit_state();
    let page = Page::page_of(phys_addr);
    ctx.entry_points
        .entry(page)
        .or_insert_with(|| (0, HashSet::new()))
        .1
        .insert((offset & 0xFFF) as u16);
}

/// Is this page currently owned by a published module? A driver compile of an owned page is
/// refused by the analyzer (empty block set), so the host must know before it calls.
#[no_mangle]
pub fn aot_drv_page_owned(phys_addr: u32) -> u32 {
    let ctx = get_jit_state();
    if ctx.pages.contains_key(&Page::page_of(phys_addr)) { 1 } else { 0 }
}

/// Compile one page (or page set, if the analyzer follows edges off-page) offline.
/// Returns the emitted module length in bytes, or 0 on refusal.
#[no_mangle]
pub fn aot_drv_compile(virt_entry: i32, cs_offset: u32, state_flags: u32, table_index: u32) -> u32 {
    let mut ctx = get_jit_state();
    let ctx = &mut *ctx;
    let state_flags = CachedStateFlags::of_u32(state_flags);

    let cpu = CpuContext { eip: 0, prefixes: 0, cs_offset, state_flags };

    let virt_page = Page::page_of(virt_entry as u32);
    let entry_points: HashSet<i32> = unsafe {
        ENTRY_OFFSETS
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
    if ctx.pages.contains_key(&Page::page_of(virt_entry as u32)) {
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

    let fastmem_generation = fastmem_compile_generation(state_flags);

    // Own builder instance: the driver must not disturb the live JIT's builder state.
    let builder = unsafe {
        if BUILDER.is_none() {
            BUILDER = Some(WasmBuilder::new());
        }
        BUILDER.as_mut().unwrap()
    };

    let entries = jit_generate_module(
        structure,
        &basic_block_by_addr,
        cpu,
        builder,
        WasmTableIndex(table_index as u16),
        state_flags,
        fastmem_generation,
    );

    unsafe {
        OUT_ENTRIES = entries;
        let len = builder.get_output_len() as usize;
        let ptr = builder.get_output_ptr();
        OUT_BYTES = std::slice::from_raw_parts(ptr, len).to_vec();
        OUT_BYTES.len() as u32
    }
}

#[no_mangle]
pub fn aot_drv_output_ptr() -> u32 { unsafe { OUT_BYTES.as_ptr() as u32 } }

#[no_mangle]
pub fn aot_drv_output_len() -> u32 { unsafe { OUT_BYTES.len() as u32 } }

#[no_mangle]
pub fn aot_drv_entry_count() -> u32 { unsafe { OUT_ENTRIES.len() as u32 } }

#[no_mangle]
pub fn aot_drv_entry_addr(i: u32) -> u32 { unsafe { OUT_ENTRIES[i as usize].0 } }

#[no_mangle]
pub fn aot_drv_entry_state(i: u32) -> u32 { unsafe { OUT_ENTRIES[i as usize].1 as u32 } }

#[no_mangle]
pub fn aot_drv_block_count() -> u32 { unsafe { LAST_BLOCK_COUNT } }

#[no_mangle]
pub fn aot_drv_page_count() -> u32 { unsafe { LAST_PAGE_COUNT } }
