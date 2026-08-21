use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::iter::FromIterator;
use std::mem::{self, MaybeUninit};
use std::ops::{Deref, DerefMut};
use std::sync::{Mutex, MutexGuard};

use crate::analysis;
use crate::analysis::AnalysisType;
use crate::codegen;
use crate::control_flow;
use crate::control_flow::WasmStructure;
use crate::cpu::cpu;
use crate::cpu::global_pointers;
use crate::cpu::hypercall;
use crate::cpu::memory;
use crate::cpu_context::CpuContext;
use crate::jit_instructions;
use crate::opstats;
use crate::page::Page;
use crate::profiler;
use crate::profiler::stat;
use crate::state_flags::CachedStateFlags;
use crate::trace_profiler;
use crate::wasmgen::wasm_builder::{Label, WasmBuilder, WasmLocal, WasmLocalI64};

#[derive(Copy, Clone, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct WasmTableIndex(u16);
impl WasmTableIndex {
    pub fn to_u16(self) -> u16 { self.0 }
}

mod unsafe_jit {
    use super::{CachedStateFlags, WasmTableIndex};

    #[link(wasm_import_module = "env")]
    extern "C" {
        pub fn codegen_finalize(
            wasm_table_index: WasmTableIndex,
            phys_addr: u32,
            state_flags: CachedStateFlags,
            ptr: u32,
            len: u32,
        );
        pub fn jit_clear_func(wasm_table_index: WasmTableIndex);
    }
}

fn codegen_finalize(
    wasm_table_index: WasmTableIndex,
    phys_addr: u32,
    state_flags: CachedStateFlags,
    ptr: u32,
    len: u32,
) {
    unsafe { unsafe_jit::codegen_finalize(wasm_table_index, phys_addr, state_flags, ptr, len) }
}

pub fn jit_clear_func(wasm_table_index: WasmTableIndex) {
    unsafe { unsafe_jit::jit_clear_func(wasm_table_index) }
}

static mut JIT_DISABLED: bool = false;

// Maximum number of pages per wasm module. Necessary for the following reasons:
// - There is an upper limit on the size of a single function in wasm (currently ~7MB in all browsers)
//   See https://github.com/WebAssembly/design/issues/1138
// - v8 poorly handles large br_table elements and OOMs on modules much smaller than the above limit
//   See https://bugs.chromium.org/p/v8/issues/detail?id=9697 and https://bugs.chromium.org/p/v8/issues/detail?id=9141
//   Will hopefully be fixed in the near future by generating direct control flow
static mut MAX_PAGES: u32 = 3;

static mut JIT_USE_LOOP_SAFETY: bool = true;
// RET/AbsoluteEip dynamic chaining: when the in-module AbsoluteEip
// re-dispatch misses, attempt a cross-module tail-call at the runtime eip instead of
// exiting to main_loop. Gated at COMPILE time — toggle via
// set_jit_config(12) and clear the JIT cache.
static mut JIT_RET_CHAINING: bool = false;
// Count CHAINED module entries toward tier-2 hotness (idx 20, default ON). Pure accounting:
// it changes no emitted code, so both arms run identical modules and switching needs no
// cache clear — which is what separates idx 12's CODE effect from its COUNTER effect.
static mut JIT_CHAIN_TIER2_ACCOUNTING: bool = true;
// RET-target speculation (superblock lite): annotate the RET of a
// small module-local leaf with its call sites' return addresses and emit inline
// eip-compare + direct dispatcher re-entry, skipping the jit_find_cache_entry_in_page
// helper on the hot return path. Same-page CALL discovery already splices the callee's
// blocks into the caller's module (follow_jump), so no new SMC surface: the callee's
// page is already in the module's page set. set_jit_config(13); budget idx 14 caps the
// callee's total instruction count (leaf qualification).
static mut JIT_RET_SPECULATION: bool = false;
static mut JIT_RET_SPEC_MAX_INSTR: u32 = 24;
const RET_SPEC_MAX_CANDIDATES: usize = 4;

// B1b: direct-mapped memo in front of the dynamic-chaining tlb_code walk
// (measured 1.5% self + part of the 7% indirect-jump bucket, NFSU in-race).
// Entries are (virt eip, state_flags, packed target, epoch); packed < 0 = empty.
// The cache holds MODULE-LIFETIME data (a packed wasm-table slot + dispatcher state),
// so every event that could invalidate a dispatch target the stock per-dispatch
// re-validation would have caught MUST bump RET_CACHE_EPOCH (O(1) invalidate-all;
// entries stamped with an older epoch miss on probe). Bump sites:
//   - free_wasm_table_index — the ONLY place a wasm-table slot is nulled
//     (jit_clear_func). NOT free_wasm_module: codegen_finalize_finished's
//     module-overwrite path (INVALIDATE_MODULE_UNUSED_AFTER_OVERWRITE) frees the
//     replaced module's index WITHOUT going through free_wasm_module — missing this
//     bump causes a null-function crash (Mechanism 0).
//   - clear_tlb_code, when it actually drops a Code entry — the stock resolver
//     re-derived liveness from tlb_code on every dispatch; after an eviction/remap
//     a memo hit would dispatch a stale-but-live module the resolver would have
//     rejected (Mechanism 1: wrong-code execution, not a trap).
// Fastmem-tracked units (generation != 0) are never cached — their per-dispatch
// generation check cannot be memoized. No per-thread state: entries are eip-keyed,
// and the budget/in_hlt guard still runs before every probe.
const RET_CACHE_SIZE: usize = 512;
static mut RET_CACHE: [(u32, u32, i32, u64); RET_CACHE_SIZE] = [(0, 0, -1, 0); RET_CACHE_SIZE];
// Starts at 1 so zero-initialized entries can never match before their first fill.
static mut RET_CACHE_EPOCH: u64 = 1;

pub fn ret_cache_invalidate_all() {
    unsafe {
        RET_CACHE_EPOCH += 1;
    }
}

// Count of double-frees the release-safe guard in free_wasm_table_index absorbed.
// Nonzero means a free-discipline bug survives somewhere — investigate, don't shrug.
static mut WASM_TABLE_INDEX_DOUBLE_FREE_SKIPPED: u32 = 0;

#[no_mangle]
pub fn jit_get_double_free_skipped() -> u32 { unsafe { WASM_TABLE_INDEX_DOUBLE_FREE_SKIPPED } }

// ── Wrong-entry dispatch detector (diagnostic; House/BoD garbage-register hunt) ──
// Every dispatch resolved through DISPATCH_META is re-verified against ctx.pages —
// the authoritative publication record. A mismatch is a STALE/WRONG dispatch (the
// silent wrong-entry class: the guest enters a valid-looking block with another call
// site's registers). Mismatches are counted, their details latched, and the dispatch
// REFUSED (interpreter/main-loop takes over) — so if this class is the crash, the
// detector build both proves it (counter > 0) and survives.
static mut WRONG_ENTRY_DIRECT: u32 = 0;
static mut WRONG_ENTRY_CHAIN: u32 = 0;
static mut RET_MEMO_MISMATCH: u32 = 0;
// Stale memo hits split by whether the cached slot still names a LIVE module
// (primary or hidden in ctx.pages). live = benign overwrite staleness (same guest
// bytes, older module); dead/recycled = genuine wrong-code dispatch.
static mut RET_MEMO_STALE_LIVE: u32 = 0;
static mut RET_MEMO_STALE_DEAD: u32 = 0;
// Wrong-entry detector mode (set_jit_config idx 24):
// 0 = OFF (no verification — production default; the verify costs a lock + entry-list
//     find per module entry), 1 = passive (verify + count, dispatch anyway),
// 2 = refuse (verify + count, mismatches fall back to the interpreter/verified path).
static mut WRONG_ENTRY_REFUSE: u32 = 0;
// Ring of the first N mismatches, 4 u32 each:
// [eip, phys, dispatched idx<<16|state, expected idx<<16|state] for wrong-entry records;
// [eip, cached packed, fresh packed, 2 (tag)] for memo-mismatch records.
const WRONG_ENTRY_RING_CAP: usize = 32;
static mut WRONG_ENTRY_RING: [[u32; 4]; WRONG_ENTRY_RING_CAP] = [[0; 4]; WRONG_ENTRY_RING_CAP];
static mut WRONG_ENTRY_RING_LEN: u32 = 0;
// eip, phys, dispatched idx<<16|state, expected idx<<16|state
static mut WRONG_ENTRY_LAST: [u32; 4] = [0; 4];

fn wrong_entry_ring_push(rec: [u32; 4]) {
    unsafe {
        let len = WRONG_ENTRY_RING_LEN as usize;
        if len < WRONG_ENTRY_RING_CAP {
            WRONG_ENTRY_RING[len] = rec;
            WRONG_ENTRY_RING_LEN += 1;
        }
    }
}

/// Read ring record field: i = record index, f = field 0..3. 0 when out of range.
#[no_mangle]
pub fn jit_get_wrong_entry_ring(i: u32, f: u32) -> u32 {
    unsafe {
        if (i as usize) < WRONG_ENTRY_RING_LEN as usize && f < 4 {
            WRONG_ENTRY_RING[i as usize][f as usize]
        }
        else {
            0
        }
    }
}
#[no_mangle]
pub fn jit_get_wrong_entry_ring_len() -> u32 { unsafe { WRONG_ENTRY_RING_LEN } }

/// Static layout probes: where the JIT's big statics live in linear memory, to check
/// adjacency against externally-writable buffers (D3D9_ARENA & co).
#[no_mangle]
pub fn jit_get_dispatch_slabs_ptr() -> u32 {
    unsafe { std::ptr::addr_of!(DISPATCH_SLABS) as u32 }
}
#[no_mangle]
pub fn jit_get_dispatch_slabs_len() -> u32 { (DISPATCH_SLAB_COUNT * 0x1000 * 2) as u32 }
#[no_mangle]
pub fn jit_get_dispatch_meta_ptr() -> u32 {
    unsafe { std::ptr::addr_of!(DISPATCH_META) as u32 }
}

/// Raw meta word for a virt page (diagnostic readback), split into two u32 halves.
#[no_mangle]
pub fn jit_debug_meta_lo(vpage: u32) -> u32 { dispatch_meta_get(vpage) as u32 }
#[no_mangle]
pub fn jit_debug_meta_hi(vpage: u32) -> u32 { (dispatch_meta_get(vpage) >> 32) as u32 }
/// Raw slab cell for a virt page's slab at byte-offset `off` (diagnostic readback).
#[no_mangle]
pub fn jit_debug_slab_cell(vpage: u32, off: u32) -> u32 {
    let meta = dispatch_meta_get(vpage);
    if meta == 0 {
        return 0xFFFF_FFFF;
    }
    let slab = (meta as u16) as usize;
    if slab == 0 || slab >= DISPATCH_SLAB_COUNT || off >= 0x1000 {
        return 0xFFFF_FFFF;
    }
    unsafe { DISPATCH_SLABS[slab * 0x1000 + off as usize] as u32 }
}

#[no_mangle]
pub fn jit_get_wrong_entry_direct() -> u32 { unsafe { WRONG_ENTRY_DIRECT } }
#[no_mangle]
pub fn jit_get_wrong_entry_chain() -> u32 { unsafe { WRONG_ENTRY_CHAIN } }
#[no_mangle]
pub fn jit_get_ret_memo_mismatch() -> u32 { unsafe { RET_MEMO_MISMATCH } }
#[no_mangle]
pub fn jit_get_wrong_entry_info(i: u32) -> u32 {
    unsafe { (&*std::ptr::addr_of!(WRONG_ENTRY_LAST)).get(i as usize).copied().unwrap_or(0) }
}
#[no_mangle]
pub fn jit_get_ret_memo_stale_live() -> u32 { unsafe { RET_MEMO_STALE_LIVE } }
#[no_mangle]
pub fn jit_get_ret_memo_stale_dead() -> u32 { unsafe { RET_MEMO_STALE_DEAD } }

// Free-time meta audit: DISPATCH_META entries still referencing a slot at the moment
// it is freed, split by why the TLB sweep missed them. Any nonzero count means a virt
// page keeps dispatching into the freed (soon-recycled) slot.
static mut STALE_META_AT_FREE: u32 = 0;
static mut STALE_META_TLB_LIVE: u32 = 0; // tlb_data != 0 yet sweep missed it (?!)
static mut STALE_META_TLB_DEAD: u32 = 0; // tlb_data == 0 — meta survived an eviction path

#[no_mangle]
pub fn jit_get_stale_meta_at_free() -> u32 { unsafe { STALE_META_AT_FREE } }
#[no_mangle]
pub fn jit_get_stale_meta_tlb_live() -> u32 { unsafe { STALE_META_TLB_LIVE } }
#[no_mangle]
pub fn jit_get_stale_meta_tlb_dead() -> u32 { unsafe { STALE_META_TLB_DEAD } }

fn wrong_entry_refuse() -> bool { unsafe { WRONG_ENTRY_REFUSE >= 2 } }

pub fn wrong_entry_verify_enabled() -> bool { unsafe { WRONG_ENTRY_REFUSE != 0 } }

#[no_mangle]
pub fn jit_wrong_entry_refuse_enabled() -> bool { wrong_entry_refuse() }

/// Verify a meta-resolved dispatch target against ctx.pages. Returns true when they
/// agree (dispatch may proceed). `virt` is only for the latched diagnostic record.
pub fn jit_verify_dispatch_entry(
    phys_addr: u32,
    state_flags: CachedStateFlags,
    wasm_table_index: u16,
    initial_state: u16,
    virt: u32,
    chain: bool,
) -> bool {
    let entry = jit_find_cache_entry(phys_addr, state_flags);
    if entry.wasm_table_index.to_u16() == wasm_table_index && entry.initial_state == initial_state
    {
        return true;
    }
    unsafe {
        if chain {
            WRONG_ENTRY_CHAIN += 1;
        }
        else {
            WRONG_ENTRY_DIRECT += 1;
        }
        WRONG_ENTRY_LAST = [
            virt,
            phys_addr,
            (wasm_table_index as u32) << 16 | initial_state as u32,
            (entry.wasm_table_index.to_u16() as u32) << 16 | entry.initial_state as u32,
        ];
        wrong_entry_ring_push(WRONG_ENTRY_LAST);
        // Capture the two disagreeing sources: the published SLAB for the virt page
        // (tag 4) and ctx.pages' entry list for the phys page (tag 5) — first pairs
        // of each, packed offset<<16|state.
        let meta = dispatch_meta_get(virt >> 12);
        if meta != 0 {
            let slab = (meta as u16) as usize;
            let mut pairs = [0u32; 2];
            let mut n = 0;
            for off in 0..0x1000usize {
                let st = unsafe { DISPATCH_SLABS[slab * 0x1000 + off] };
                // Cells hold state + 1, so 0 — not u16::MAX — is the miss sentinel
                // (see dispatch_state_lookup), and the reported state must be decoded
                // back so it is comparable with the tag-5 record below.
                if st != 0 {
                    if n < 2 {
                        pairs[n] = (off as u32) << 16 | (st - 1) as u32;
                    }
                    n += 1;
                }
            }
            wrong_entry_ring_push([virt & !0xFFF | (n as u32 & 0xFFF), pairs[0], pairs[1], 4]);
            // Is this slab SHARED? Scan every meta for the same slab id — sharing means
            // a slab id was handed out twice (double free / free-stack corruption) and
            // pages overwrite each other's dispatch tables.
            let mut share = 0u32;
            let mut others = [0u32; 2];
            for p in 0..(1usize << 20) {
                let m = unsafe { DISPATCH_META[p] };
                if m != 0 && (m as u16) as usize == slab {
                    if (p as u32) != virt >> 12 && (share as usize) < 2 {
                        others[share as usize] = (p as u32) << 12;
                    }
                    share += 1;
                }
            }
            wrong_entry_ring_push([slab as u32 | share << 16, others[0], others[1], 6]);
        }
        {
            let ctx = get_jit_state();
            if let Some(info) = ctx.pages.get(&Page::page_of(phys_addr)) {
                let l = &info.entry_points;
                let pk = |i: usize| {
                    l.get(i)
                        .map_or(0, |&(o, s)| (o as u32) << 16 | s as u32)
                };
                wrong_entry_ring_push([
                    phys_addr & !0xFFF | (l.len() as u32 & 0xFFF),
                    pk(0),
                    pk(1),
                    5,
                ]);
            }
        }
    }
    dbg_log!(
        "WRONG-ENTRY {} eip={:x} phys={:x} dispatched={}:{} expected={}:{}",
        if chain { "chain" } else { "direct" },
        virt,
        phys_addr,
        wasm_table_index,
        initial_state,
        entry.wasm_table_index.to_u16(),
        entry.initial_state,
    );
    false
}

// B3 hotness tiering: a module whose RE-ENTRY count (bumped per cycle_internal entry —
// the cheapest per-module execution proxy that needs no codegen) crosses the threshold
// gets its pages marked tier-2 and is freed; the ordinary hotness path recompiles it,
// and jit_find_basic_blocks sees the tier-2 marking and compiles with expanded budgets
// (more pages per module + a deeper RET-speculation window). Cold code never pays for
// the expensive compilation. Threshold 0 disables (set_jit_config idx 15); the page-set
// cap bounds runaway promotion (compile-storm guard — once full, no new promotions).
static mut JIT_TIER2_THRESHOLD: u32 = 300_000;
static mut JIT_TIER2_RET_SPEC_MAX_INSTR: u32 = 96;
// Runtime-tunable (set_jit_config idx 17) so the tier-2 module-size budget can be
// A/B'd in-race without a rebuild; 8 was never tuned. Raising it grows only PROMOTED
// modules (cold code keeps the global MAX_PAGES), so the V8 large-function OOM risk
// that forbids raising the global cap doesn't apply at moderate values.
static mut TIER2_MAX_PAGES: u32 = 8;
const TIER2_PAGE_SET_CAP: usize = 256;
static mut MODULE_EXEC_COUNTS: [u32; 0x10000] = [0; 0x10000];

// Per-module entry count for the SLOT'S CURRENT LIFE (reset when the index is freed),
// unlike MODULE_EXEC_COUNTS which tier-2 zeroes on promotion. It weights a per-slot V8
// tier sample (dbg.jitTierStats) by execution, turning "how many modules are in baseline"
// into "what share of EXECUTION never reaches the optimizing tier". Reset-on-free is what
// lets the sampler detect slot recycling: a negative delta means a different module.
static mut MODULE_ENTRY_TOTALS: [u32; 0x10000] = [0; 0x10000];

#[no_mangle]
pub fn jit_get_module_entry_total(wasm_table_index: u32) -> u32 {
    unsafe { MODULE_ENTRY_TOTALS[(wasm_table_index & 0xFFFF) as usize] }
}

// Tier-2 observability (read via dbg.tier2Stats()): without these there is no way to
// tell "promotions landed" apart from "promotions starved by the page-set cap" — the
// exact ambiguity that made the in-race B3 A/B unreadable (threshold changes showed
// zero FPS delta because the cap, not the threshold, was the limiter candidate).
static mut TIER2_PROMOTIONS: u32 = 0;
static mut TIER2_BLOCKED_BY_CAP: u32 = 0;

#[no_mangle]
pub fn jit_get_tier2_page_count() -> u32 {
    get_jit_state().tier2_pages.len() as u32
}
#[no_mangle]
pub fn jit_get_tier2_promotions() -> u32 {
    unsafe { TIER2_PROMOTIONS }
}
#[no_mangle]
pub fn jit_get_tier2_blocked_by_cap() -> u32 {
    unsafe { TIER2_BLOCKED_BY_CAP }
}
/// Distinct pages the cap refused (see `tier2_blocked_pages`). Read against the refusal
/// count: distinct << count means a few hot modules retry, distinct ~ count means the
/// hot set genuinely outgrew TIER2_PAGE_SET_CAP.
#[no_mangle]
pub fn jit_get_tier2_blocked_distinct() -> u32 {
    get_jit_state().tier2_blocked_pages.len() as u32
}
/// i-th tier-2 page address (page<<12), 0 when i >= count. Iteration order is the
/// HashSet's (arbitrary but stable between mutations) — callers use this to feed
/// trace2_watch_page with known-hot pages (tier-2 membership == crossed the re-entry
/// threshold), closing the "which pages should Tier-2R watch" loop without an EIP
/// sampler (which only sees idle/yield EIPs — JS timers can't fire mid-cycle-slice).
#[no_mangle]
pub fn jit_get_tier2_page_at(i: u32) -> u32 {
    let ctx = get_jit_state();
    match ctx.tier2_pages.iter().nth(i as usize) {
        Some(p) => p.to_address(),
        None => 0,
    }
}

/// Bump the entry counters for one module entry. Returns true when THIS entry crossed the
/// tier-2 threshold (and consumes the crossing by resetting the promotion counter).
///
/// Split out of `jit_tier2_note_execution` so a caller that cannot safely promote — a
/// chained edge, which runs inside a live generated frame — can still do the ACCOUNTING,
/// which is all a chained edge owes. Touches nothing but two plain static arrays: no
/// JitState lock, no allocation, no map walk. That matters because the chain path is the
/// hottest dispatch path in the JIT.
#[inline]
fn tier2_count_entry(wasm_table_index: u16) -> bool {
    unsafe {
        let t = &mut (*std::ptr::addr_of_mut!(MODULE_ENTRY_TOTALS))[wasm_table_index as usize];
        *t = t.wrapping_add(1);
    }
    let threshold = unsafe { JIT_TIER2_THRESHOLD };
    if threshold == 0 {
        return false;
    }
    let count = unsafe {
        let c = &mut (*std::ptr::addr_of_mut!(MODULE_EXEC_COUNTS))[wasm_table_index as usize];
        *c += 1;
        *c
    };
    if count < threshold {
        return false;
    }
    unsafe { MODULE_EXEC_COUNTS[wasm_table_index as usize] = 0 };
    true
}

/// Promote a module's pages to tier-2 and free the module so hotness recompiles it with the
/// tier-2 budget. Returns true when the module was actually freed.
///
/// CONTRACT: no generated frame for `wasm_table_index` (or any module it shares pages with)
/// may be live — this nulls table slots, clears dispatch meta and drops the TLB HAS_CODE
/// bit. Only two call sites satisfy that: `jit_tier2_note_execution` (cycle_internal,
/// BETWEEN module entries) and `jit_tier2_drain_pending` (same, before dispatch).
fn tier2_promote(wasm_table_index: u16) -> bool {
    let mut ctx = get_jit_state();
    let index = WasmTableIndex(wasm_table_index);
    let pages: Vec<Page> = ctx
        .pages
        .iter()
        .filter(|(_, info)| info.wasm_table_index == index)
        .map(|(p, _)| *p)
        .collect();
    if pages.is_empty() {
        return false;
    }
    // Already fully tier-2? Nothing to gain from another free/recompile churn.
    if pages.iter().all(|p| ctx.tier2_pages.contains(p)) {
        return false;
    }
    if ctx.tier2_pages.len() + pages.len() > TIER2_PAGE_SET_CAP {
        unsafe { TIER2_BLOCKED_BY_CAP += 1 };
        for p in &pages {
            if !ctx.tier2_pages.contains(p) {
                ctx.tier2_blocked_pages.insert(*p);
            }
        }
        return false;
    }
    for p in &pages {
        ctx.tier2_pages.insert(*p);
        // A page that got in is no longer starved — otherwise the set reports a lifetime
        // tally instead of what is currently being refused, and only ever grows.
        ctx.tier2_blocked_pages.remove(p);
    }
    unsafe { TIER2_PROMOTIONS += 1 };
    free_wasm_module_tree(&mut ctx, index);
    true
}

/// Called from cycle_internal on every compiled-module entry. Returns true when the
/// module was just promoted to tier-2 AND freed — the caller must not dispatch into it
/// (run interpreted this slice; hotness recompiles it with the tier-2 budget).
#[no_mangle]
pub fn jit_tier2_note_execution(wasm_table_index: u16) -> bool {
    unsafe { TIER2_DIRECT_ENTRIES += 1 };
    tier2_count_entry(wasm_table_index) && tier2_promote(wasm_table_index)
}

// Entry-event census, split by the path the entry arrived on. RET chaining (idx 12) changes
// both the emitted CODE and the hotness COUNTER; the share chain/(chain+direct) is what makes
// the two separable.
static mut TIER2_CHAIN_ENTRIES: u64 = 0;
static mut TIER2_DIRECT_ENTRIES: u64 = 0;

#[no_mangle]
pub fn jit_get_tier2_chain_entries() -> f64 { unsafe { TIER2_CHAIN_ENTRIES as f64 } }
#[no_mangle]
pub fn jit_get_tier2_direct_entries() -> f64 { unsafe { TIER2_DIRECT_ENTRIES as f64 } }

// Promotions owed to CHAINED entries. A chained edge is counted where it happens (see
// chain_note_execution) but promoted here, because the counting site runs inside a live
// generated frame and tier2_promote's contract forbids that.
//
// Bounded and lossy on purpose: overflow only delays a promotion to the module's next
// threshold crossing, so the cap costs accuracy, never correctness — and a static array
// keeps the enqueue allocation-free.
const TIER2_PENDING_CAP: usize = 64;
static mut TIER2_PENDING: [u16; TIER2_PENDING_CAP] = [0; TIER2_PENDING_CAP];
static mut TIER2_PENDING_LEN: u32 = 0;
static mut TIER2_PENDING_DROPPED: u32 = 0;

/// Queued chain promotions that were dropped because the queue was full. Nonzero means
/// some promotions slipped to a later crossing — visible rather than mysterious.
#[no_mangle]
pub fn jit_get_tier2_pending_dropped() -> u32 { unsafe { TIER2_PENDING_DROPPED } }

/// Drop every queued promotion naming `idx`. Called when the slot is freed: the queue holds
/// slot numbers, not module identities, so a stale entry would promote the slot's next owner.
fn tier2_pending_drop(idx: u16) {
    unsafe {
        let n = TIER2_PENDING_LEN as usize;
        let mut w = 0usize;
        for r in 0..n {
            if TIER2_PENDING[r] != idx {
                TIER2_PENDING[w] = TIER2_PENDING[r];
                w += 1;
            }
        }
        TIER2_PENDING_LEN = w as u32;
    }
}

/// Apply the promotions queued by chained entries.
///
/// MUST be called only from cycle_internal, i.e. between module entries: it is the same
/// safe point `jit_tier2_note_execution` already promotes at. Near-free when idle (one
/// static load and a branch), which is why it can sit on the per-block path.
#[no_mangle]
pub fn jit_tier2_drain_pending() {
    let n = unsafe { TIER2_PENDING_LEN } as usize;
    if n == 0 {
        return;
    }
    unsafe { TIER2_PENDING_LEN = 0 };
    for i in 0..n {
        tier2_promote(unsafe { TIER2_PENDING[i] });
    }
}

static mut JIT_DEAD_FLAG_ELISION: bool = false;
static mut JIT_X87_LOCALS: bool = false;
static mut JIT_PUSH_RUN_COALESCING: bool = false;
// Fastmem WRITES behind a per-page writability map (idx 19).
// Off by default until the in-game gate passes.
static mut JIT_FASTMEM_WRITES: bool = false;
// Lazy-flag tuple in wasm
// LOCALS instead of linear-memory globals 96-120 (idx 21, default OFF). Removes
// per-ALU-op flag stores AND their TurboFan aliasing barriers. Correctness
// contract: locals are authoritative between spills; the builder-level call_fn
// funnel spills/reloads around every non-whitelisted helper call (covers arith
// flag-protocol helpers AND OUT/hypercall context saves), and the
// module epilogues spill at every exit.
static mut JIT_FLAG_LOCALS: bool = false;

// Wasm branch hints (idx 22, bitmask of wasm_builder::HINT_GROUP_*, default 0 = OFF).
// Emits a "metadata.code.branch_hint" custom section marking the slow-path side of
// memory/TLB/x87 guards as unlikely. Pure layout advice: no observable semantics, and a
// malformed section only loses the hints (V8 decodes it with an inner decoder). Read by
// the optimizing tier only — Liftoff ignores hints, so any win is scaled by the module's
// share of tier-up'd time. Bit 1 = memory guards, bit 2 = x87 relaxed cache.
static mut JIT_BRANCH_HINTS: u32 = 0;
// Robustness self-test knob (idx 23): shift every emitted hint offset by N bytes so the
// hints deliberately point at non-branch instructions. Modules must still compile and
// produce identical results — that is the property being asserted.
static mut JIT_BRANCH_HINT_OFFSET_FUZZ: u32 = 0;

// Tier-2R region recompiler: grow page groups across
// indirect edges using trace_profiler target histograms, and make hot indirect
// targets dispatcher entries so AbsoluteEip re-dispatches stay intra-module.
// Off by default; requires collected trace2 data to have any effect.
static mut JIT_INDIRECT_REGIONS: bool = false;

// Region-growth safety: virtual-address range that indirect-region growth must NEVER pull
// targets from — the thunk/callback/spin bucket. Guest code indirect-calls thunk
// stubs constantly (GetProcAddress'd exports), so profiled indirect targets point
// into stub pages; compiling those into a guest superblock traps `unreachable`
// at CALLBACK_STUB+0x477f0. Set by JS via
// jit_set_region_exclusion (scheduler arms it with [THUNK_CODE_BASE, ROM_BASE)).
// hi == 0 → no exclusion (feature off, e.g. older TS).
static mut REGION_EXCLUDE_LO: u32 = 0;
static mut REGION_EXCLUDE_HI: u32 = 0;

#[no_mangle]
pub fn jit_set_region_exclusion(lo: u32, hi: u32) {
    unsafe {
        REGION_EXCLUDE_LO = lo;
        REGION_EXCLUDE_HI = hi;
    }
}

fn region_target_excluded(target: u32) -> bool {
    unsafe { REGION_EXCLUDE_HI != 0 && target >= REGION_EXCLUDE_LO && target < REGION_EXCLUDE_HI }
}
static mut JIT_INDIRECT_REGION_MIN_SHARE: u32 = 5; // percent of per-site hits
const JIT_INDIRECT_REGION_MAX_TARGETS: usize = 16;
// Page budget for region growth ACROSS INDIRECT EDGES only — kept separate from
// the global MAX_PAGES (which caps normal direct-jump BFS at 3). Raising the
// global cap instead bloats EVERY module via long direct-call chains and OOMs
// V8 on NFSU (large generated functions / br_tables — v86's own warning). Here
// only a dispatcher block that hits recorded hot targets grows, and only up to
// this many pages, prioritising the hottest targets first.
static mut JIT_INDIRECT_REGION_MAX_PAGES: u32 = 8;

pub static mut MAX_EXTRA_BASIC_BLOCKS: u32 = 250;

// Block-chaining dispatch characterisation toggle.
// When enabled, the JIT emits the always-on dispatch-characterisation counters (BLOCK_EXECUTION
// and MODULE_EXIT_*) and the runtime increments MODULE_REENTRY / MODULE_EXIT_INDIRECT. The
// codegen-emitted counters are gated at COMPILE time, so enable this BEFORE the workload compiles
// its hot modules (set_dispatch_stats(1) at boot, then clear the JIT cache) and read the result
// via profiler_dispatch_stat_get. OFF by default — zero cost on the production path.
pub static mut DISPATCH_STATS: bool = false;
pub fn dispatch_stats_enabled() -> bool { unsafe { DISPATCH_STATS } }
fn ret_chaining_enabled() -> bool { unsafe { JIT_RET_CHAINING } }
fn ret_speculation_enabled() -> bool { unsafe { JIT_RET_SPECULATION } }
fn dead_flag_elision_enabled() -> bool { unsafe { JIT_DEAD_FLAG_ELISION } }


// ── Fastmem READ map ───────────────────────────────────────────────────────
// One byte per 4 KB virtual page. The JIT loads this byte for every speculative
// read, so a mapping/protection change is a local data update, never a global
// JIT invalidation. Byte 1 means exactly: present, identity-mapped RAM, outside
// low memory and the guard band. All other values force the normal TLB path.

// ── Fastmem WRITE map ─────────────────────────────────────────────────────────────
// One byte per 4 KB VA page across the full 4 GB space (1 MB static). The JIT store
// fast path (codegen::gen_fastmem_write_map) accepts a page IFF its byte == 1, so the
// byte is a bitfield where every restriction independently vetoes the fast path:
//   bit0  BASE_WRITABLE  committed, RW, plain identity-mapped RAM   — owner: TS choke points
//   bit1  HAS_CODE       page holds compiled code (SMC net)         — owner: rust tlb_set_has_code
//   bit2  WRITE_WATCH    debug write-watch armed on this page       — owner: rust dbg_set_write_watch
// Unlike read speculation this carries NO stale window and NO generation guard: the map
// is DATA, read per store, updated synchronously at the same choke points that keep the
// TLB honest, so compiled code never goes stale. A stale-writable byte on an
// RO/CoW/code page would be silent memory corruption, so the ONLY safe
// failure direction is leaving a byte != 1 (slow path, byte-precise). Init all zeros =
// conservative = correct; TS marks RW ranges as regions register/commit during boot.
pub const FASTMEM_WRITE_MAP_LEN: usize = 1 << 20; // 4 GB / 4 KB pages, one byte each
static mut FASTMEM_WRITE_MAP: [u8; FASTMEM_WRITE_MAP_LEN] = [0; FASTMEM_WRITE_MAP_LEN];
const FASTMEM_WRITE_BASE_WRITABLE: u8 = 1 << 0;
const FASTMEM_WRITE_HAS_CODE: u8 = 1 << 1;
const FASTMEM_WRITE_WATCH: u8 = 1 << 2;
static mut FASTMEM_SPECULATED_STORES_COMPILED: u32 = 0;
// Highest VA page for which TS has ever set bit0 — bounds the audit/count scans so they
// don't walk the whole 1 MB map every call (the populated prefix is tiny in practice).
static mut FASTMEM_WRITE_MAP_MAX_PAGE: u32 = 0;
// Hard "never fast-writable" page band [lo, hi) — TS points this at THUNK_CODE (which
// holds immutable RX stubs but has RW PTEs under the identity map, so a kind-blind PTE
// path could otherwise set bit0 on it). bit0 SET is refused inside this band regardless
// of caller. hi == 0 ⇒ no exclusion (feature off / older TS).
static mut FASTMEM_WRITE_EXCLUDE_LO_PAGE: u32 = 0;
static mut FASTMEM_WRITE_EXCLUDE_HI_PAGE: u32 = 0;

// ── DOD dispatch metadata ─────────────────────────────────────────────────────────
// Replaces the Box<Code> layout behind the old `cpu::tlb_code` pointer array. The
// hot resolvers (jit_find_cache_entry* — called on EVERY guest ret/indirect jump,
// hit or miss) previously walked THREE dependent loads, two through heap pointers:
//   tlb_code[page] → *Box<Code> → .state_table[offset]
// a cache-miss-bound pointer chase. The SoA replacement derives every address
// from `page` alone — the loads are INDEPENDENT and issue in parallel:
//   DISPATCH_META[page]                        ; packed word, dense 8 MB array
//   DISPATCH_SLABS[slab*0x1000 + (addr&0xFFF)] ; dense u16 pool, no chase
//
// meta packing: state_flags(u32) << 32 | wasm_table_index(u16) << 16 | slab(u16).
// meta == 0 ⇒ page has no compiled code. Slab index 0 is RESERVED-invalid so the
// zero word stays an unambiguous sentinel; usable slabs are 1..DISPATCH_SLAB_COUNT.
//
// Maintenance funnels are exactly the old tlb_code writers: set_tlb_code (compile/
// TLB-fill) and cpu::clear_tlb_code (eviction/invlpg/dirty) — no new choke points.
pub const DISPATCH_SLAB_COUNT: usize = 4096; // 4096 × 8 KB = 32 MB pool
static mut DISPATCH_META: [u64; 1 << 20] = [0; 1 << 20];
static mut DISPATCH_SLABS: [u16; DISPATCH_SLAB_COUNT * 0x1000] =
    [0; DISPATCH_SLAB_COUNT * 0x1000];
// Free stack of slab indices; filled 1..DISPATCH_SLAB_COUNT by rust_init.
static mut DISPATCH_SLAB_FREE: [u16; DISPATCH_SLAB_COUNT] = [0; DISPATCH_SLAB_COUNT];
static mut DISPATCH_SLAB_FREE_TOP: usize = 0;
static mut DISPATCH_SLAB_HIGH_WATER: u32 = 0;
static mut DISPATCH_SLAB_OVERFLOWS: u32 = 0;

pub fn dispatch_meta_init() {
    unsafe {
        // Stack of free slabs, slab 0 excluded (reserved sentinel).
        for i in 1..DISPATCH_SLAB_COUNT {
            DISPATCH_SLAB_FREE[i - 1] = i as u16;
        }
        DISPATCH_SLAB_FREE_TOP = DISPATCH_SLAB_COUNT - 1;
    }
}

#[inline]
pub fn dispatch_meta_get(page: u32) -> u64 { unsafe { DISPATCH_META[page as usize & 0xFFFFF] } }

#[inline]
pub fn dispatch_meta_state_flags(meta: u64) -> u32 { (meta >> 32) as u32 }

#[inline]
pub fn dispatch_meta_table_index(meta: u64) -> u16 { (meta >> 16) as u16 }

/// Slab cells store `initial_state + 1`, so the "no entry here" sentinel is ZERO —
/// the value uninitialized or externally-zeroed memory already has. With the previous
/// encoding (0 = a perfectly valid entry-block index, u16::MAX = miss) any agent that
/// zeroed a slab cell turned it into a LIVE dispatch to the module's block 0: the guest
/// re-enters a real compiled module at the wrong entry point, carrying whatever
/// registers the current call site had — silent wrong-code execution, observed as an
/// AV with several registers sharing one garbage value. Making the sentinel structural
/// downgrades that whole class to a dispatch MISS (interpret + recompile: slower, correct).
#[inline]
pub fn dispatch_state_lookup(meta: u64, virt_address: u32) -> u16 {
    unsafe {
        let slab = (meta as u16) as usize;
        dbg_assert!(slab != 0 && slab < DISPATCH_SLAB_COUNT);
        let cell = DISPATCH_SLABS[slab * 0x1000 + (virt_address as usize & 0xFFF)];
        if cell == 0 { u16::MAX } else { cell - 1 }
    }
}

/// Publish (or refresh) a page's dispatch entries. Reuses the page's existing slab
/// when present. On pool exhaustion the page simply stays unpublished (meta 0) —
/// resolvers miss, the interpreter runs the code, correctness is unaffected; the
/// loud counter makes the condition visible in stats long before it can matter
/// (pool = 4095 pages-with-code, typical live set is a few hundred).
pub fn dispatch_meta_set(
    virt_page: Page,
    wasm_table_index: WasmTableIndex,
    entries: &Vec<(u16, u16)>,
    state_flags: CachedStateFlags,
) {
    unsafe {
        let page = virt_page.to_u32() as usize & 0xFFFFF;
        let existing = DISPATCH_META[page];
        let slab = if existing != 0 {
            (existing as u16) as usize
        }
        else {
            if DISPATCH_SLAB_FREE_TOP == 0 {
                DISPATCH_SLAB_OVERFLOWS = DISPATCH_SLAB_OVERFLOWS.saturating_add(1);
                dbg_log!("dispatch: slab pool exhausted, page {:x} unpublished", page);
                return;
            }
            DISPATCH_SLAB_FREE_TOP -= 1;
            let s = DISPATCH_SLAB_FREE[DISPATCH_SLAB_FREE_TOP] as usize;
            let in_use = (DISPATCH_SLAB_COUNT - 1 - DISPATCH_SLAB_FREE_TOP) as u32;
            if in_use > DISPATCH_SLAB_HIGH_WATER {
                DISPATCH_SLAB_HIGH_WATER = in_use;
            }
            s
        };
        dbg_assert!(slab != 0 && slab < DISPATCH_SLAB_COUNT);

        // 0 = miss (see dispatch_state_lookup): cells hold state + 1.
        let table = &mut DISPATCH_SLABS[slab * 0x1000..slab * 0x1000 + 0x1000];
        table.fill(0);
        for &(addr, state) in entries {
            dbg_assert!(state != u16::MAX);
            if state == u16::MAX {
                continue; // cannot be represented as state+1; publishing it as a miss is safe
            }
            table[addr as usize] = state + 1;
        }

        DISPATCH_META[page] = (state_flags.to_u32() as u64) << 32
            | (wasm_table_index.to_u16() as u64) << 16
            | slab as u64;
        // Live-cell census, so a later rescan can tell "this slab was overwritten by
        // something outside the JIT" from "this slab legitimately has few entries".
        // With 0 = miss, a zero cell is no longer self-evidently damage.
        DISPATCH_SLAB_LIVE[slab] = entries.len() as u16;
    }
}

// Entries published per slab (see dispatch_meta_set). Compared against a live recount
// by `jit_slab_audit`.
static mut DISPATCH_SLAB_LIVE: [u16; DISPATCH_SLAB_COUNT] = [0; DISPATCH_SLAB_COUNT];

/// Slabs whose live-cell count no longer matches what was published — i.e. cells were
/// zeroed (or written) by something other than the JIT. Returns the number of damaged
/// slabs; `jit_slab_audit_last` names the most recent one.
#[no_mangle]
pub fn jit_slab_audit() -> u32 {
    unsafe {
        let mut damaged = 0;
        for s in 1..DISPATCH_SLAB_COUNT {
            let published = DISPATCH_SLAB_LIVE[s];
            if published == 0 {
                continue;
            }
            let mut live = 0u32;
            for i in 0..0x1000 {
                if DISPATCH_SLABS[s * 0x1000 + i] != 0 {
                    live += 1;
                }
            }
            if live != published as u32 {
                damaged += 1;
                SLAB_AUDIT_LAST = [s as u32, published as u32, live];
            }
        }
        damaged
    }
}

static mut SLAB_AUDIT_LAST: [u32; 3] = [0; 3];

#[no_mangle]
pub fn jit_slab_audit_last(i: u32) -> u32 {
    unsafe { (&*std::ptr::addr_of!(SLAB_AUDIT_LAST)).get(i as usize).copied().unwrap_or(0) }
}

/// Unpublish a page. Returns true if the page actually had an entry (callers use
/// this to bump the B1b ret-memo epoch only on real evictions, as before).
pub fn dispatch_meta_clear(page: u32) -> bool {
    unsafe {
        let page = page as usize & 0xFFFFF;
        let meta = DISPATCH_META[page];
        if meta == 0 {
            return false;
        }
        let slab = (meta as u16) as usize;
        dbg_assert!(slab != 0 && slab < DISPATCH_SLAB_COUNT);
        dbg_assert!(DISPATCH_SLAB_FREE_TOP < DISPATCH_SLAB_COUNT);
        DISPATCH_SLAB_FREE[DISPATCH_SLAB_FREE_TOP] = slab as u16;
        DISPATCH_SLAB_FREE_TOP += 1;
        DISPATCH_SLAB_LIVE[slab] = 0; // no longer published — exclude from the audit
        DISPATCH_META[page] = 0;
        true
    }
}

// Dispatch-slab occupancy counters exported for the TS stats verb.
#[no_mangle]
pub fn dispatch_slab_high_water() -> u32 { unsafe { DISPATCH_SLAB_HIGH_WATER } }
#[no_mangle]
pub fn dispatch_slab_overflows() -> u32 { unsafe { DISPATCH_SLAB_OVERFLOWS } }

// Compile-site counters; runtime hit/fill split is not instrumented.
static mut X87_LOCAL_CACHE_LOAD_SITES_COMPILED: u32 = 0;
static mut X87_LOCAL_CACHE_STORES_COMPILED: u32 = 0;
static mut X87_LOCAL_CACHE_INVALIDATES_COMPILED: u32 = 0;

static mut PUSH_RUN_SITES_COMPILED: u32 = 0;
static mut PUSH_RUN_REUSE_BRANCHES_COMPILED: u32 = 0;

// Mirrors emulator-config.ts; dbg.fastmemReads asserts equality.
pub const FASTMEM_LOW_MEM_END: u32 = 0x0010_0000;
pub const FASTMEM_GUARD_BASE: u32 = 0x2300_0000;
pub const FASTMEM_GUARD_SIZE: u32 = 0x0100_0000;

#[no_mangle]
pub fn fastmem_get_low_mem_end() -> u32 { FASTMEM_LOW_MEM_END }
#[no_mangle]
pub fn fastmem_get_guard_base() -> u32 { FASTMEM_GUARD_BASE }
#[no_mangle]
pub fn fastmem_get_guard_size() -> u32 { FASTMEM_GUARD_SIZE }

#[inline]
pub fn x87_locals_enabled() -> bool { unsafe { JIT_X87_LOCALS } }

#[inline]
pub fn push_run_coalescing_enabled() -> bool { unsafe { JIT_PUSH_RUN_COALESCING } }

#[inline]
pub fn flag_locals_enabled() -> bool { unsafe { JIT_FLAG_LOCALS } }

#[inline]
pub fn branch_hint_mask() -> u32 { unsafe { JIT_BRANCH_HINTS } }

// Compile-time gate for the store fast path. Same regime as fastmem reads (32-bit
// protected mode + paging): the map's identity-map store `mem8 + addr` is only valid
// where VA == PA, which BottleShip guarantees under paging. Any page NOT identity-RW
// simply never has bit0 set by TS, so even if the shape is emitted the fast path is
// never taken there — this gate only avoids emitting dead shape in other regimes.
pub fn fastmem_writes_compile_enabled(state_flags: CachedStateFlags) -> bool {
    unsafe {
        JIT_FASTMEM_WRITES
            && state_flags.is_32()
            && *global_pointers::protected_mode
            && (*global_pointers::cr & cpu::CR0_PG) != 0
    }
}

// Wasm-memory address of the write map, baked as a load base in the store fast path.
#[inline]
pub fn fastmem_write_map_base() -> u32 {
    unsafe { &FASTMEM_WRITE_MAP[0] as *const u8 as u32 }
}

#[inline]
pub fn fastmem_note_speculated_store_compiled() {
    unsafe {
        FASTMEM_SPECULATED_STORES_COMPILED = FASTMEM_SPECULATED_STORES_COMPILED.saturating_add(1);
    }
}

// ── Write-map maintenance ─────────────────────────────────────────────────────────
// TS owns bit0 only; rust owns bit1/bit2. The worker is single-threaded, so the split
// ownership is a plain (non-atomic) read-modify-write with no race.

/// bit0 (BASE_WRITABLE) over [start_page, start_page+page_count). Owner: TS choke
/// points (region register / commit / decommit / protect). Only bit0 is touched.
///
/// Setting bit0 is authoritatively clamped HERE to the SAME identity-RAM envelope the
/// read fast path trusts — [LOW_MEM_END, min(GUARD_BASE, ram)) ∪ [GUARD_END, ram), in
/// pages — so no TS caller can ever fast-enable a low-mem/MMIO page, the guard red zone,
/// or an unbacked (> ram) page, whatever range it passes. Clearing is unconditional
/// (slow path is always the safe direction).
#[no_mangle]
pub fn fastmem_write_map_set_base(start_page: u32, page_count: u32, writable: u32) {
    unsafe {
        let start = (start_page as usize).min(FASTMEM_WRITE_MAP_LEN);
        let end = (start_page as usize)
            .saturating_add(page_count as usize)
            .min(FASTMEM_WRITE_MAP_LEN);
        if writable == 0 {
            for p in start..end {
                FASTMEM_WRITE_MAP[p] &= !FASTMEM_WRITE_BASE_WRITABLE;
            }
            return;
        }
        let ram = *global_pointers::memory_size;
        let lo1 = (FASTMEM_LOW_MEM_END >> 12) as usize;
        let hi1 = (FASTMEM_GUARD_BASE.min(ram) >> 12) as usize;
        let lo2 = (FASTMEM_GUARD_BASE.wrapping_add(FASTMEM_GUARD_SIZE) >> 12) as usize;
        let hi2 = (ram >> 12) as usize;
        let excl_lo = FASTMEM_WRITE_EXCLUDE_LO_PAGE as usize;
        let excl_hi = FASTMEM_WRITE_EXCLUDE_HI_PAGE as usize;
        for p in start..end {
            let in_envelope = (p >= lo1 && p < hi1) || (p >= lo2 && p < hi2);
            let excluded = excl_hi != 0 && p >= excl_lo && p < excl_hi;
            if in_envelope && !excluded {
                FASTMEM_WRITE_MAP[p] |= FASTMEM_WRITE_BASE_WRITABLE;
                if (p as u32) > FASTMEM_WRITE_MAP_MAX_PAGE {
                    FASTMEM_WRITE_MAP_MAX_PAGE = p as u32;
                }
            }
        }
    }
}

/// TS points this at the THUNK_CODE band at boot so no PTE-level (kind-blind) SET can
/// ever fast-enable the immutable RX stubs. Also clears bit0 across the band defensively.
#[no_mangle]
pub fn fastmem_write_map_set_exclude(lo_page: u32, hi_page: u32) {
    unsafe {
        FASTMEM_WRITE_EXCLUDE_LO_PAGE = lo_page;
        FASTMEM_WRITE_EXCLUDE_HI_PAGE = hi_page;
        if hi_page > lo_page {
            let lo = (lo_page as usize).min(FASTMEM_WRITE_MAP_LEN);
            let hi = (hi_page as usize).min(FASTMEM_WRITE_MAP_LEN);
            for p in lo..hi {
                FASTMEM_WRITE_MAP[p] &= !FASTMEM_WRITE_BASE_WRITABLE;
            }
        }
    }
}

/// Wipe the whole map to zero (conservative). Called by TS at v86 (re)init before it
/// re-marks the RW regions, in case the wasm instance (and thus this static) persisted.
#[no_mangle]
pub fn fastmem_write_map_reset() {
    unsafe {
        core::ptr::write_bytes(&raw mut FASTMEM_WRITE_MAP as *mut u8, 0, FASTMEM_WRITE_MAP_LEN);
        FASTMEM_WRITE_MAP_MAX_PAGE = 0;
        FASTMEM_SPECULATED_STORES_COMPILED = 0;
    }
}

#[inline]
pub fn fastmem_write_map_set_code(page: u32) {
    unsafe {
        if (page as usize) < FASTMEM_WRITE_MAP_LEN {
            FASTMEM_WRITE_MAP[page as usize] |= FASTMEM_WRITE_HAS_CODE;
        }
    }
}

#[inline]
pub fn fastmem_write_map_clear_code(page: u32) {
    unsafe {
        if (page as usize) < FASTMEM_WRITE_MAP_LEN {
            FASTMEM_WRITE_MAP[page as usize] &= !FASTMEM_WRITE_HAS_CODE;
        }
    }
}

#[inline]
pub fn fastmem_write_map_set_watch(page: u32) {
    unsafe {
        if (page as usize) < FASTMEM_WRITE_MAP_LEN {
            FASTMEM_WRITE_MAP[page as usize] |= FASTMEM_WRITE_WATCH;
        }
    }
}

#[inline]
pub fn fastmem_write_map_clear_watch(page: u32) {
    unsafe {
        if (page as usize) < FASTMEM_WRITE_MAP_LEN {
            FASTMEM_WRITE_MAP[page as usize] &= !FASTMEM_WRITE_WATCH;
        }
    }
}

/// Raw map byte for one page (audit verb).
#[no_mangle]
pub fn fastmem_write_map_get(page: u32) -> u32 {
    unsafe {
        if (page as usize) < FASTMEM_WRITE_MAP_LEN {
            FASTMEM_WRITE_MAP[page as usize] as u32
        }
        else {
            0
        }
    }
}

/// Count pages within the populated prefix. mask == 0 → count acceptance (byte == 1);
/// otherwise count pages where (byte & mask) != 0. Bounded by the max marked page.
#[no_mangle]
pub fn fastmem_write_map_count(mask: u32) -> u32 {
    unsafe {
        let hi = (FASTMEM_WRITE_MAP_MAX_PAGE as usize + 1).min(FASTMEM_WRITE_MAP_LEN);
        let mut n = 0u32;
        for p in 0..hi {
            let b = FASTMEM_WRITE_MAP[p] as u32;
            let hit = if mask == 0 { b == 1 } else { (b & mask) != 0 };
            if hit {
                n = n.saturating_add(1);
            }
        }
        n
    }
}

#[no_mangle]
pub fn fastmem_get_speculated_stores_compiled() -> u32 {
    unsafe { FASTMEM_SPECULATED_STORES_COMPILED }
}

/// Highest VA page ever marked base-writable — upper bound for the audit scan.
#[no_mangle]
pub fn fastmem_write_map_max_page() -> u32 {
    unsafe { FASTMEM_WRITE_MAP_MAX_PAGE }
}

#[inline]
pub fn x87_locals_note_cache_load_site_compiled() {
    unsafe {
        X87_LOCAL_CACHE_LOAD_SITES_COMPILED =
            X87_LOCAL_CACHE_LOAD_SITES_COMPILED.saturating_add(1);
    }
}

#[inline]
pub fn x87_locals_note_cache_store_compiled() {
    unsafe {
        X87_LOCAL_CACHE_STORES_COMPILED =
            X87_LOCAL_CACHE_STORES_COMPILED.saturating_add(1);
    }
}

#[inline]
pub fn x87_locals_note_cache_invalidate_compiled() {
    unsafe {
        X87_LOCAL_CACHE_INVALIDATES_COMPILED =
            X87_LOCAL_CACHE_INVALIDATES_COMPILED.saturating_add(1);
    }
}

#[inline]
pub fn push_run_note_site_compiled() {
    unsafe {
        PUSH_RUN_SITES_COMPILED = PUSH_RUN_SITES_COMPILED.saturating_add(1);
    }
}

#[inline]
pub fn push_run_note_reuse_branch_compiled() {
    unsafe {
        PUSH_RUN_REUSE_BRANCHES_COMPILED =
            PUSH_RUN_REUSE_BRANCHES_COMPILED.saturating_add(1);
    }
}

#[no_mangle]
pub fn x87_locals_get_cache_load_sites_compiled() -> u32 {
    unsafe { X87_LOCAL_CACHE_LOAD_SITES_COMPILED }
}

#[no_mangle]
pub fn x87_locals_get_cache_stores_compiled() -> u32 { unsafe { X87_LOCAL_CACHE_STORES_COMPILED } }

#[no_mangle]
pub fn x87_locals_get_cache_invalidates_compiled() -> u32 {
    unsafe { X87_LOCAL_CACHE_INVALIDATES_COMPILED }
}

#[no_mangle]
pub fn push_run_get_sites_compiled() -> u32 { unsafe { PUSH_RUN_SITES_COMPILED } }

#[no_mangle]
pub fn push_run_get_reuse_branches_compiled() -> u32 {
    unsafe { PUSH_RUN_REUSE_BRANCHES_COMPILED }
}

#[no_mangle]
pub fn set_dispatch_stats(enabled: u32) { unsafe { DISPATCH_STATS = enabled != 0; } }

#[no_mangle]
pub fn get_dispatch_stats() -> u32 { unsafe { DISPATCH_STATS as u32 } }

pub const JIT_THRESHOLD: u32 = 200 * 1000;

// less branches will generate if-else, more will generate brtable
pub const BRTABLE_CUTOFF: usize = 10;

// needs to be synced to const.js
pub const WASM_TABLE_SIZE: u32 = 900;

// Light the invariant checks up in debug builds — the corruption class
// (silent ExitProcess via #PF on garbage state) needs the free/publish
// discipline asserted loudly, not assumed.
pub const CHECK_JIT_STATE_INVARIANTS: bool = cfg!(debug_assertions);

const MAX_INSTRUCTION_LENGTH: u32 = 16;

static JIT_STATE: Mutex<MaybeUninit<JitState>> = Mutex::new(MaybeUninit::uninit());
fn get_jit_state() -> JitStateRef { JitStateRef(JIT_STATE.try_lock().unwrap()) }

struct JitStateRef(MutexGuard<'static, MaybeUninit<JitState>>);

impl Deref for JitStateRef {
    type Target = JitState;
    fn deref(&self) -> &Self::Target { unsafe { self.0.assume_init_ref() } }
}
impl DerefMut for JitStateRef {
    fn deref_mut(&mut self) -> &mut Self::Target { unsafe { self.0.assume_init_mut() } }
}

#[no_mangle]
pub fn rust_init() {
    dispatch_meta_init();

    let _ = JIT_STATE
        .try_lock()
        .unwrap()
        .write(JitState::create_and_initialise());

    crate::d3d9_glue::init();

    use std::panic;

    panic::set_hook(Box::new(|panic_info| {
        console_log!("{}", panic_info.to_string());
    }));
}

struct PageInfo {
    wasm_table_index: WasmTableIndex,
    hidden_wasm_table_indices: Vec<WasmTableIndex>,
    entry_points: Vec<(u16, u16)>,
    state_flags: CachedStateFlags,
}

enum CompilingPageState {
    Compiling { pages: HashMap<Page, PageInfo> },
    CompilingWritten,
}

struct JitState {
    wasm_builder: WasmBuilder,

    // as an alternative to HashSet, we could use a bitmap of 4096 bits here
    // (faster, but uses much more memory)
    // or a compressed bitmap (likely faster)
    // or HashSet<u32> rather than nested
    entry_points: HashMap<Page, (u32, HashSet<u16>)>,
    pages: HashMap<Page, PageInfo>,
    wasm_table_index_free_list: Vec<WasmTableIndex>,
    compiling: Option<(WasmTableIndex, CompilingPageState)>,
    // Rust owns this reservation until commit or abort. JS must clear its corresponding
    // table entry before calling abort; Rust never infers ownership from that table.
    aot_staged: Option<AotTransaction>,
    // B3 hotness tiering: pages promoted to tier-2 (jit_tier2_note_execution) — modules
    // whose entries land on these pages compile with the expanded tier-2 budgets.
    // Survives jit_clear_cache (the pages are still the hot ones); dies with the wasm
    // instance (per game load).
    tier2_pages: HashSet<Page>,
    // Pages a promotion wanted and the cap refused. The refusal COUNT cannot distinguish
    // "one hot module retrying forever" from "hundreds of distinct pages starved", and those
    // two want opposite fixes — a bigger cap versus a replacement policy.
    tier2_blocked_pages: HashSet<Page>,
    #[cfg(debug_assertions)]
    wasm_table_index_to_page: HashMap<WasmTableIndex, HashSet<Page>>,
}

fn check_jit_state_invariants(ctx: &mut JitState) {
    if !CHECK_JIT_STATE_INVARIANTS {
        return;
    }

    match &ctx.compiling {
        Some((_, CompilingPageState::Compiling { pages })) => {
            dbg_assert!(pages.keys().all(|page| ctx.entry_points.contains_key(page)));
        },
        _ => {},
    }

    let free: HashSet<WasmTableIndex> =
        HashSet::from_iter(ctx.wasm_table_index_free_list.iter().copied());
    let used = HashSet::from_iter(ctx.pages.values().map(|info| info.wasm_table_index));
    let compiling = HashSet::from_iter(ctx.compiling.as_ref().map(|&(index, _)| index));
    let staged = HashSet::from_iter(ctx.aot_staged.as_ref().map(|tx| tx.wasm_table_index));
    dbg_assert!(free.intersection(&used).next().is_none());
    dbg_assert!(used.intersection(&compiling).next().is_none());
    dbg_assert!(free.intersection(&staged).next().is_none());
    dbg_assert!(used.intersection(&staged).next().is_none());
    dbg_assert!(compiling.intersection(&staged).next().is_none());
    dbg_assert!(free.len() + used.len() + compiling.len() + staged.len() == (WASM_TABLE_SIZE - 1) as usize);

    let hidden: HashSet<WasmTableIndex> = ctx
        .pages
        .values()
        .flat_map(|info| info.hidden_wasm_table_indices.iter().copied())
        .collect();
    dbg_assert!(free.intersection(&hidden).next().is_none());
    dbg_assert!(hidden.is_subset(&used));
    dbg_assert!(hidden.intersection(&staged).next().is_none());

    #[cfg(debug_assertions)]
    for (wasm_table_index, pages) in &ctx.wasm_table_index_to_page {
        for page in pages {
            match ctx.pages.get(page) {
                Some(info) => dbg_assert!(
                    info.wasm_table_index == *wasm_table_index
                        || info.hidden_wasm_table_indices.contains(wasm_table_index)
                ),
                None => dbg_assert!(false),
            }
        }
    }

    match &ctx.compiling {
        Some((_, CompilingPageState::Compiling { pages })) => {
            dbg_assert!(pages.keys().all(|page| ctx.entry_points.contains_key(page)));
        },
        _ => {},
    }

    for i in 0..unsafe { cpu::valid_tlb_entries_count } {
        let page = unsafe { cpu::valid_tlb_entries[i as usize] };
        let entry = unsafe { cpu::tlb_data[page as usize] };
        if 0 != entry {
            let tlb_physical_page = Page::of_u32(
                (entry as u32 >> 12 ^ page as u32) - (unsafe { memory::mem8 } as u32 >> 12),
            );
            let meta = dispatch_meta_get(page as u32);
            let w = if meta != 0 {
                Some(WasmTableIndex(dispatch_meta_table_index(meta)))
            }
            else {
                None
            };
            let tlb_has_code = entry & cpu::TLB_HAS_CODE == cpu::TLB_HAS_CODE;
            let infos = ctx.pages.get(&tlb_physical_page);
            let entry_points = ctx.entry_points.get(&tlb_physical_page);
            dbg_assert!(tlb_has_code || !w.is_some());
            dbg_assert!(tlb_has_code || !infos.is_some());
            dbg_assert!(tlb_has_code || !entry_points.is_some());
            //dbg_assert!((w.is_some() || page.is_some() || entry_points.is_some()) == tlb_has_code); // XXX: check this
        }
    }
}

impl JitState {
    pub fn create_and_initialise() -> JitState {
        // don't assign 0 (XXX: Check)
        let wasm_table_indices = (1..=(WASM_TABLE_SIZE - 1) as u16).map(|x| WasmTableIndex(x));

        JitState {
            wasm_builder: WasmBuilder::new(),

            entry_points: HashMap::new(),
            pages: HashMap::new(),

            wasm_table_index_free_list: Vec::from_iter(wasm_table_indices),
            compiling: None,
            aot_staged: None,
            tier2_pages: HashSet::new(),
            tier2_blocked_pages: HashSet::new(),

            #[cfg(debug_assertions)]
            wasm_table_index_to_page: HashMap::new(),
        }
    }
}

#[derive(PartialEq, Eq)]
pub enum BasicBlockType {
    Normal {
        next_block_addr: Option<u32>,
        jump_offset: i32,
        jump_offset_is_32: bool,
    },
    ConditionalJump {
        next_block_addr: Option<u32>,
        next_block_branch_taken_addr: Option<u32>,
        condition: u8,
        jump_offset: i32,
        jump_offset_is_32: bool,
    },
    // Set eip to an absolute value (ret, jmp r/m, call r/m)
    AbsoluteEip,
    Exit,
}

pub struct BasicBlock {
    pub addr: u32,
    pub virt_addr: i32,
    pub last_instruction_addr: u32,
    pub end_addr: u32,
    pub is_entry_block: bool,
    pub ty: BasicBlockType,
    pub has_sti: bool,
    pub number_of_instructions: u32,
    /// RET-target speculation (superblock lite): for an AbsoluteEip
    /// block that is a genuine RET of a small leaf function called from within this
    /// module, the (virt, phys) return addresses of its module-local call sites. The
    /// emitter turns each into `if eip == virt { target_block = <dispatcher idx>;
    /// br main_loop }` ahead of the jit_find_cache_entry_in_page helper call, so a
    /// leaf's return stays intra-module without the dispatch helper. Filled by the
    /// post-pass in jit_find_basic_blocks when JIT_RET_SPECULATION is on; empty
    /// otherwise. The compare guards correctness — a stale/wrong candidate simply
    /// falls through to the existing dispatch.
    pub ret_speculation: Vec<(i32, u32)>,
}

#[derive(Copy, Clone, PartialEq)]
pub struct CachedCode {
    pub wasm_table_index: WasmTableIndex,
    pub initial_state: u16,
}

impl CachedCode {
    pub const NONE: CachedCode = CachedCode {
        wasm_table_index: WasmTableIndex(0),
        initial_state: 0,
    };
}

#[derive(PartialEq)]
pub enum InstructionOperandDest {
    WasmLocal(WasmLocal),
    Other,
}
#[derive(PartialEq)]
pub enum InstructionOperand {
    WasmLocal(WasmLocal),
    Immediate(i32),
    Other,
}
impl InstructionOperand {
    pub fn is_zero(&self) -> bool {
        match self {
            InstructionOperand::Immediate(0) => true,
            _ => false,
        }
    }
}
impl Into<InstructionOperand> for InstructionOperandDest {
    fn into(self: InstructionOperandDest) -> InstructionOperand {
        match self {
            InstructionOperandDest::WasmLocal(l) => InstructionOperand::WasmLocal(l),
            InstructionOperandDest::Other => InstructionOperand::Other,
        }
    }
}
pub enum Instruction {
    Cmp {
        dest: InstructionOperandDest,
        source: InstructionOperand,
        opsize: i32,
    },
    Sub {
        dest: InstructionOperandDest,
        source: InstructionOperand,
        opsize: i32,
        is_dec: bool,
    },
    Add {
        dest: InstructionOperandDest,
        source: InstructionOperand,
        opsize: i32,
        is_inc: bool,
    },
    AdcSbb {
        dest: InstructionOperandDest,
        #[allow(dead_code)]
        source: InstructionOperand,
        opsize: i32,
    },
    NonZeroShift {
        dest: InstructionOperandDest,
        opsize: i32,
    },
    Bitwise {
        dest: InstructionOperandDest,
        opsize: i32,
    },
    Other,
}

pub struct JitContext<'a> {
    pub cpu: &'a mut CpuContext,
    pub builder: &'a mut WasmBuilder,
    pub register_locals: &'a mut Vec<WasmLocal>,
    pub start_of_current_instruction: u32,
    pub exit_with_fault_label: Label,
    pub exit_label: Label,
    pub current_instruction: Instruction,
    pub previous_instruction: Instruction,
    pub fpu_simd_dirty_marked: bool,
    pub elide_current_flags: bool,
    pub instruction_counter: WasmLocal,
    /// Emit the per-page-map store fast path for this unit.
    pub fastmem_writes: bool,
    pub x87_local_cache: [Option<X87LocalCacheSlot>; 8],
    pub push32_write_cache: Option<Push32WriteCache>,
    /// Set true by any x87 relaxed wrapper that leaves the block-scoped st-local
    /// cache coherent (it either updated the touched slot or invalidated all
    /// slots). Reset to false before each instruction; if an x87 opcode
    /// (D8–DF) is compiled through a raw-helper path that did NOT set this, the
    /// emission loop invalidates the cache so a later relaxed op re-reads memory
    /// and re-checks the tag. Without this, helper-dispatched FPU ops (FISTP m64
    /// aka _ftol, FSQRT/FSIN/FCOS, FINCSTP/FDECSTP, FLD m80, FIADD-family, …)
    /// silently mutate the FPU stack / shift TOP behind stale cached values.
    pub x87_cache_kept: bool,
}

pub struct X87LocalCacheSlot {
    pub bits: WasmLocalI64,
    pub valid: WasmLocal,
}

pub struct Push32WriteCache {
    pub page: WasmLocal,
    pub entry: WasmLocal,
    pub valid: WasmLocal,
}

impl<'a> JitContext<'a> {
    pub fn reg(&self, i: u32) -> WasmLocal {
        match self.register_locals.get(i as usize) {
            Some(x) => x.unsafe_clone(),
            None => {
                dbg_assert!(false);
                unsafe { std::hint::unreachable_unchecked() }
            },
        }
    }
}

pub const JIT_INSTR_BLOCK_BOUNDARY_FLAG: u32 = 1 << 0;

pub fn is_near_end_of_page(address: u32) -> bool {
    address & 0xFFF >= 0x1000 - MAX_INSTRUCTION_LENGTH
}

// Classification of one x86 instruction for the dead-flag liveness walk. Safe by construction:
// the walk only ELIDES when it can prove flags dead, so anything not provably an `Overwrite` or a
// provably-clean `NeutralNoFault` is `Stop` — we never need to enumerate flag *readers* precisely.
#[derive(Copy, Clone, PartialEq)]
enum FlagClass {
    // Fully overwrites every lazy-tracked flag (CF/PF/AF/ZF/SF/OF) before reading any.
    // non_faulting = the instruction cannot fault before the overwrite (register-only form).
    Overwrite { non_faulting: bool },
    // Provably touches NO flags AND cannot fault (register-only mov/lea/movzx/movsx/nop) — safe to
    // skip over while walking forward to the next flag-overwriter.
    NeutralNoFault,
    // Everything else: reads a flag (Jcc/ADC/SBB/SETcc/CMOVcc/PUSHF/LAHF/…), modifies flags
    // partially (INC/DEC/shift/rotate/SAHF/POPF), can fault (any memory operand), or is
    // control-flow/unrecognized. Conservatively stops the walk WITHOUT eliding.
    Stop,
}

fn read_jit_u8(addr: u32) -> u8 { memory::read8(addr) as u8 }

fn skip_instruction_prefixes(mut addr: u32) -> u32 {
    loop {
        match read_jit_u8(addr) {
            0x26 | 0x2E | 0x36 | 0x3E | 0x64 | 0x65 | 0x66 | 0x67 | 0xF0 | 0xF2 | 0xF3 => {
                addr += 1;
            },
            _ => return addr,
        }
    }
}

fn decode_jit_opcode(addr: u32) -> (u32, u32) {
    let opcode_addr = skip_instruction_prefixes(addr);
    let opcode = read_jit_u8(opcode_addr);
    if opcode == 0x0F {
        (0x100 | read_jit_u8(opcode_addr + 1) as u32, opcode_addr + 2)
    }
    else {
        (opcode as u32, opcode_addr + 1)
    }
}

fn group_alu_is_full_overwrite(group: u8) -> bool {
    matches!(group, 0 | 1 | 4 | 5 | 6 | 7)
}

fn classify_flag_class(addr: u32) -> FlagClass {
    let (opcode, operand_addr) = decode_jit_opcode(addr);
    // Read the byte after the opcode; only meaningful for opcodes that actually have a ModRM.
    // For no-ModRM opcodes (imm/reg-encoded/NOP) the branches below don't consult it.
    let modrm = read_jit_u8(operand_addr);
    let reg_only = modrm & 0xC0 == 0xC0;

    match opcode {
        // --- Full flag overwriters: ADD/OR/AND/SUB/XOR/CMP r/m<->reg, TEST r/m,reg ---
        // Register-only forms can't fault before overwriting flags; memory forms can.
        0x00..=0x03 | 0x08..=0x0B | 0x20..=0x23 | 0x28..=0x2B | 0x30..=0x33
        | 0x38..=0x3B | 0x84 | 0x85 => FlagClass::Overwrite { non_faulting: reg_only },

        // Accumulator-immediate ALU/TEST (no ModRM, no memory) — always non-faulting.
        0x04 | 0x05 | 0x0C | 0x0D | 0x24 | 0x25 | 0x2C | 0x2D | 0x34 | 0x35
        | 0x3C | 0x3D | 0xA8 | 0xA9 => FlagClass::Overwrite { non_faulting: true },

        // Group 1 ALU immediates: /0 ADD /1 OR /4 AND /5 SUB /6 XOR /7 CMP overwrite;
        // /2 ADC /3 SBB read CF → Stop.
        0x80 | 0x81 | 0x82 | 0x83 => {
            if group_alu_is_full_overwrite((modrm >> 3) & 7) {
                FlagClass::Overwrite { non_faulting: reg_only }
            }
            else {
                FlagClass::Stop
            }
        },

        // Group 3 /0 TEST imm overwrites; /2 NOT doesn't touch flags but other /n
        // (NEG/MUL/IMUL/DIV/IDIV) set them in special ways → Stop for everything but /0.
        0xF6 | 0xF7 => {
            if (modrm >> 3) & 7 == 0 {
                FlagClass::Overwrite { non_faulting: reg_only }
            }
            else {
                FlagClass::Stop
            }
        },

        // --- Flag-neutral, non-faulting (register-only forms only; memory forms can #PF) ---
        0x88 | 0x89 | 0x8A | 0x8B => if reg_only { FlagClass::NeutralNoFault } else { FlagClass::Stop }, // MOV r/m<->reg
        0x8D => FlagClass::NeutralNoFault,                                                               // LEA (no deref, no flags)
        0xB0..=0xBF => FlagClass::NeutralNoFault,                                                        // MOV reg, imm
        0xC6 | 0xC7 => if reg_only { FlagClass::NeutralNoFault } else { FlagClass::Stop },               // MOV r/m, imm
        0x90 => FlagClass::NeutralNoFault,                                                               // NOP
        0x1B6 | 0x1B7 | 0x1BE | 0x1BF => if reg_only { FlagClass::NeutralNoFault } else { FlagClass::Stop }, // MOVZX/MOVSX

        _ => FlagClass::Stop,
    }
}

fn instruction_end(cpu: &CpuContext, addr: u32) -> u32 {
    let mut step_cpu = cpu.clone();
    step_cpu.eip = addr;
    analysis::analyze_step(&mut step_cpu);
    step_cpu.eip
}

// How far to look ahead for the next flag-overwriter before giving up (bounded so compile time
// stays predictable; pointer-chase prologues rarely have long flag-neutral runs).
const FLAG_LIVENESS_WALK_LIMIT: u32 = 8;

// Continues the flag-liveness walk from `addr_in` (inside `block`, which ends at
// `block.end_addr`) onward. `origin_addr` is the address of the original flag-overwriting
// instruction we're trying to elide — fixed for the whole (possibly cross-block) walk, used only
// as a wraparound sanity guard, same as in the original single-block version.
//
// When the walk runs off the end of `block` still undecided, it does NOT stop there: the
// compiled module already knows this block's successor edge(s) exactly (BasicBlockType's
// next_block_addr / next_block_branch_taken_addr are resolved at compile time from the real CFG,
// not profiled/speculative), so we keep walking into the sole Normal successor. Only Normal
// reaches this point in practice: the block-discovery loop's only way to end a block WITHOUT a
// real control-flow instruction is the artificial merge-split for Normal (jit.rs, discovery loop
// — `basic_blocks.contains_key(&current_address)` cuts the block at a plain non-branching
// instruction because another path already made that address a block entry). ConditionalJump has
// no equivalent: it is only ever created from an actually-decoded Jcc (AnalysisType::Jump
// {condition: Some(_)}), so `block.last_instruction_addr` is always that Jcc — and since Jcc
// necessarily reads flags, classify_flag_class's Stop for it fires during the walk above, before
// `addr` can reach `block.end_addr`. So the ConditionalJump arm below is unreachable by
// construction, not just untested — see its comment.
fn flags_dead_from_addr(
    cpu: &CpuContext,
    origin_addr: u32,
    addr_in: u32,
    block: &BasicBlock,
    basic_blocks: &HashMap<u32, BasicBlock>,
    steps: &mut u32,
) -> bool {
    let mut addr = addr_in;
    while addr > origin_addr && addr < block.end_addr && *steps < FLAG_LIVENESS_WALK_LIMIT {
        match classify_flag_class(addr) {
            // A faulting overwriter could #PF before overwriting, and the fault frame would need the
            // (now elided) architectural flags — so only a non-faulting overwriter proves dead.
            FlagClass::Overwrite { non_faulting: true } => {
                profiler::stat_increment_always(stat::DEAD_FLAG_ELIDED);
                return true;
            },
            FlagClass::Overwrite { non_faulting: false } => return false,
            FlagClass::NeutralNoFault => {
                addr = instruction_end(cpu, addr);
                *steps += 1;
            },
            FlagClass::Stop => return false,
        }
    }

    if addr != block.end_addr || *steps >= FLAG_LIVENESS_WALK_LIMIT {
        // Either resolved to false already (Stop / faulting overwriter) or ran out of budget.
        return false;
    }

    match &block.ty {
        BasicBlockType::Normal { next_block_addr: Some(next), .. } => match basic_blocks.get(next) {
            Some(next_block) => {
                flags_dead_from_addr(cpu, origin_addr, *next, next_block, basic_blocks, steps)
            },
            // Successor isn't part of this compiled module (e.g. not yet discovered) — can't
            // prove anything about it, so don't elide.
            None => false,
        },
        BasicBlockType::ConditionalJump { .. } => {
            // Unreachable by construction (see the walk's doc comment above): a
            // ConditionalJump block always ends on a real Jcc, and Jcc always reads flags, so
            // the while-loop above always resolves via Stop before addr can reach
            // block.end_addr. Canary, not a load-bearing check — debug_assert! is compiled out
            // in release, zero cost. If block-discovery ever changes so a ConditionalJump block
            // can end elsewhere, this fires instead of silently mis-proving flags dead across a
            // branch we never actually evaluated.
            dbg_assert!(
                false,
                "flags_dead_from_addr: ConditionalJump reached block.end_addr — should be \
                 unreachable, its terminating Jcc always resolves via Stop first"
            );
            false
        },
        // AbsoluteEip / Exit (no statically known successor) or a partially-unresolved
        // Normal edge (e.g. target crosses into an unmapped page): stop here, same as the
        // original single-block behavior.
        _ => false,
    }
}

fn should_elide_current_flags(
    cpu: &CpuContext,
    current_addr: u32,
    block: &BasicBlock,
    basic_blocks: &HashMap<u32, BasicBlock>,
) -> bool {
    if !dead_flag_elision_enabled() {
        return false;
    }
    // The current instruction must itself fully overwrite the flags — only then is there a flag
    // computation to skip (the elision-aware emitters fire on these opcodes).
    if !matches!(classify_flag_class(current_addr), FlagClass::Overwrite { .. }) {
        return false;
    }
    profiler::stat_increment_always(stat::DEAD_FLAG_ELISION_CANDIDATE);

    // Walk forward, skipping flag-neutral non-faulting instructions, until the flags are proven
    // dead (a non-faulting full overwriter reached first) or possibly-live (a reader / partial /
    // faulting / control-flow instruction, a dead end at module scope, or the step limit).
    let addr = instruction_end(cpu, current_addr);
    let mut steps = 0;
    flags_dead_from_addr(cpu, current_addr, addr, block, basic_blocks, &mut steps)
}

pub fn jit_find_cache_entry(phys_address: u32, state_flags: CachedStateFlags) -> CachedCode {
    // TODO: dedup with jit_find_cache_entry_in_page?
    // NOTE: This is currently only used for invariant/missed-entry-point checking
    let ctx = get_jit_state();

    match ctx.pages.get(&Page::page_of(phys_address)) {
        Some(PageInfo {
            wasm_table_index,
            state_flags: s,
            entry_points,
            hidden_wasm_table_indices: _,
        }) => {
            if *s == state_flags {
                let page_offset = phys_address as u16 & 0xFFF;
                if let Some(&(_, initial_state)) =
                    entry_points.iter().find(|(p, _)| p == &page_offset)
                {
                    return CachedCode {
                        wasm_table_index: *wasm_table_index,
                        initial_state,
                    };
                }
            }
        },
        None => {},
    }

    return CachedCode::NONE;
}

#[no_mangle]
pub fn jit_find_cache_entry_in_page(
    virt_address: u32,
    wasm_table_index: WasmTableIndex,
    state_flags: u32,
) -> i32 {
    // TODO: generate code for this
    profiler::stat_increment(stat::INDIRECT_JUMP);
    if dispatch_stats_enabled() {
        profiler::stat_increment_always(stat::ABSEIP_DISPATCH);
    }

    let state_flags = CachedStateFlags::of_u32(state_flags);

    // DOD SoA lookup (no pointer chase; stale-generation units self-deopt on entry).
    let meta = dispatch_meta_get(virt_address >> 12);
    if meta != 0
        && dispatch_meta_state_flags(meta) == state_flags.to_u32()
        && dispatch_meta_table_index(meta) == wasm_table_index.to_u16()
    {
        let unit_state = dispatch_state_lookup(meta, virt_address);
        if unit_state != u16::MAX {
            return unit_state.into();
        }
    }

    profiler::stat_increment(stat::INDIRECT_JUMP_NO_ENTRY);

    // Block-chaining: an indirect jmp/call (AbsoluteEip) whose target is not in this
    // module → real exit to main_loop. eip was computed at runtime, so not statically chainable.
    if dispatch_stats_enabled() {
        profiler::stat_increment_always(stat::MODULE_EXIT_INDIRECT);
    }

    return -1;
}

/// Account a CHAINED module entry.
///
/// A chained edge jumps module→module without passing through `cycle_internal`, the only
/// place `jit_tier2_note_execution` runs — so without this a large share of module entries is
/// invisible to hotness accounting and to `MODULE_ENTRY_TOTALS`.
///
/// Counted here, but only counted: promotion is QUEUED for `jit_tier2_drain_pending`, because
/// this runs inside a live generated frame and promotion frees the module tree, which
/// `tier2_promote`'s contract forbids. Deferring costs at most one cycle slice against a
/// threshold of hundreds of thousands of entries and leaves the edge itself untouched — the
/// target module is still valid and installed.
#[inline]
unsafe fn chain_note_execution(packed: i32) {
    let idx = (packed >> 16) - cpu::WASM_TABLE_OFFSET as i32;
    if idx < 0 || idx > u16::MAX as i32 {
        return;
    }
    TIER2_CHAIN_ENTRIES += 1;
    if !JIT_CHAIN_TIER2_ACCOUNTING {
        return;
    }
    let idx = idx as u16;
    if !tier2_count_entry(idx) {
        return;
    }
    let len = TIER2_PENDING_LEN as usize;
    if len >= TIER2_PENDING_CAP {
        TIER2_PENDING_DROPPED += 1;
        return;
    }
    // A module can cross the threshold twice before a drain; promoting it twice would walk
    // (and free) an index that is no longer its own.
    if TIER2_PENDING[..len].contains(&idx) {
        return;
    }
    TIER2_PENDING[len] = idx;
    TIER2_PENDING_LEN += 1;
}

/// RET/AbsoluteEip dynamic chaining: budget-guarded tlb_code lookup at the runtime eip,
/// returning the packed (table_slot << 16 | unit_state) target convention, with its own
/// RET_CHAIN_HIT/RET_CHAIN_MISS stats.
#[no_mangle]
pub unsafe fn jit_find_cache_entry_for_dynamic_chaining(state_flags: u32) -> i32 {
    // same quantum as do_many_cycles_native (limit==0 urgent exit and in_hlt still bail) —
    // this is what keeps the async-park/spin-loop invariant: an urgent
    // exit request zeroes the budget, so we never chain past it.
    let limit = hypercall::read_cycle_limit();
    let elapsed = (*global_pointers::instruction_counter)
        .wrapping_sub(cpu::jit_cycle_start_instruction_counter);

    if limit == 0 || elapsed >= limit || *global_pointers::in_hlt {
        if dispatch_stats_enabled() {
            profiler::stat_increment_always(stat::RET_CHAIN_MISS);
        }
        return -1;
    }

    let virt_address = *global_pointers::instruction_pointer as u32;

    // B1b: direct-mapped memo probe. An entry is valid only if its epoch is current —
    // any table-slot free or code-TLB eviction since the fill bumps the epoch and
    // invalidates everything (see RET_CACHE).
    let cache_idx = (virt_address >> 2) as usize & (RET_CACHE_SIZE - 1);
    let cached = RET_CACHE[cache_idx];
    if cached.0 == virt_address
        && cached.1 == state_flags
        && cached.2 >= 0
        && cached.3 == RET_CACHE_EPOCH
    {
        // Wrong-entry detector (idx 24; OFF by default): a memo hit bypasses
        // DISPATCH_META entirely, so verify the cached packed target against a fresh
        // meta resolution; overwrite staleness (live older module) is counted as
        // stale-live and is benign — same guest bytes.
        if !wrong_entry_verify_enabled() {
            if dispatch_stats_enabled() {
                profiler::stat_increment_always(stat::RET_CHAIN_HIT);
            }
            chain_note_execution(cached.2);
            return cached.2;
        }
        let meta = dispatch_meta_get(virt_address >> 12);
        let fresh = if meta != 0
            && dispatch_meta_state_flags(meta) == state_flags
        {
            let st = dispatch_state_lookup(meta, virt_address);
            if st != u16::MAX {
                (dispatch_meta_table_index(meta) as i32 + cpu::WASM_TABLE_OFFSET as i32) << 16
                    | st as i32
            }
            else {
                -1
            }
        }
        else {
            -1
        };
        if fresh == cached.2 {
            if dispatch_stats_enabled() {
                profiler::stat_increment_always(stat::RET_CHAIN_HIT);
            }
            chain_note_execution(cached.2);
            return cached.2;
        }
        RET_MEMO_MISMATCH += 1;
        // Classify the stale target: a LIVE module (primary or hidden) is benign
        // overwrite-staleness — same guest bytes, older module. A dead/recycled slot
        // is genuine wrong-code dispatch.
        let stale_idx = ((cached.2 >> 16) - cpu::WASM_TABLE_OFFSET as i32) as u16;
        let target = WasmTableIndex(stale_idx);
        let live = {
            let ctx = get_jit_state();
            ctx.pages.values().any(|info| {
                info.wasm_table_index == target
                    || info.hidden_wasm_table_indices.contains(&target)
            })
        };
        if live {
            RET_MEMO_STALE_LIVE += 1;
        }
        else {
            RET_MEMO_STALE_DEAD += 1;
        }
        if RET_MEMO_MISMATCH <= 8 {
            wrong_entry_ring_push([virt_address, cached.2 as u32, fresh as u32, 2]);
        }
        dbg_log!(
            "RET-MEMO-MISMATCH eip={:x} cached={:x} fresh={:x} staleTargetLive={}",
            virt_address,
            cached.2,
            fresh,
            live
        );
        if !wrong_entry_refuse() {
            // Passive: preserve stock behavior exactly — dispatch the stale target.
            if dispatch_stats_enabled() {
                profiler::stat_increment_always(stat::RET_CHAIN_HIT);
            }
            chain_note_execution(cached.2);
            return cached.2;
        }
        RET_CACHE[cache_idx].2 = -1;
    }

    let raw_state_flags = state_flags;
    let state_flags = CachedStateFlags::of_u32(state_flags);

    // DOD SoA lookup (no pointer chase). Generation staleness is handled by the
    // unit's own prologue guard (self-deopt on entry), and a deopt/free bumps
    // RET_CACHE_EPOCH via free_wasm_table_index — so unlike the old Box walk, the
    // memo may now cache EVERY unit: there is no lookup-time generation check left
    // for it to skip.
    let meta = dispatch_meta_get(virt_address >> 12);
    if meta != 0 && dispatch_meta_state_flags(meta) == state_flags.to_u32() {
        let unit_state = dispatch_state_lookup(meta, virt_address);
        if unit_state != u16::MAX {
            // Wrong-entry detector (idx 24; OFF by default): re-verify the meta-resolved
            // target against ctx.pages before tail-calling into it.
            let verified = if !wrong_entry_verify_enabled() {
                true
            }
            else {
                match cpu::translate_address_read_no_side_effects(virt_address as i32) {
                    Ok(phys) => jit_verify_dispatch_entry(
                        phys,
                        state_flags,
                        dispatch_meta_table_index(meta),
                        unit_state,
                        virt_address,
                        true,
                    ),
                    Err(()) => false,
                }
            };
            if verified || !wrong_entry_refuse() {
                if dispatch_stats_enabled() {
                    profiler::stat_increment_always(stat::RET_CHAIN_HIT);
                }

                let table_slot =
                    dispatch_meta_table_index(meta) as i32 + cpu::WASM_TABLE_OFFSET as i32;
                let packed = table_slot << 16 | unit_state as i32;
                RET_CACHE[cache_idx] = (virt_address, raw_state_flags, packed, RET_CACHE_EPOCH);
                chain_note_execution(packed);
                return packed;
            }
        }
    }

    if dispatch_stats_enabled() {
        profiler::stat_increment_always(stat::RET_CHAIN_MISS);
    }
    -1
}

fn jit_find_basic_blocks(
    ctx: &mut JitState,
    entry_points: HashSet<i32>,
    cpu: CpuContext,
) -> Vec<BasicBlock> {
    fn follow_jump(
        virt_target: i32,
        ctx: &mut JitState,
        pages: &mut HashSet<Page>,
        page_blacklist: &mut HashSet<Page>,
        max_pages: u32,
        marked_as_entry: &mut HashSet<i32>,
        to_visit_stack: &mut Vec<i32>,
    ) -> Option<u32> {
        if is_near_end_of_page(virt_target as u32) {
            return None;
        }
        let phys_target = match cpu::translate_address_read_no_side_effects(virt_target) {
            Err(()) => {
                dbg_log!("Not analysing {:x} (page not mapped)", virt_target);
                return None;
            },
            Ok(t) => t,
        };

        let phys_page = Page::page_of(phys_target);

        // Never GROW a module INTO the thunk/callback/spin bucket (REGION_EXCLUDE_*):
        // stub pages are full of OUT traps and must stay standalone modules. This must
        // gate EVERY growth edge, not just profiled indirect targets — direct IAT-style
        // CALLs into stubs reached CALLBACK_STUB pages once the tier-2 page budget grew
        // past the default (fatal 0x3003).
        // `!pages.is_empty()` is load-bearing: the INITIAL entry points are seeded
        // through follow_jump with an empty page set — gating those blocks stub-page
        // modules from compiling at all (their dispatch then hits `unreachable` at the
        // stub EIP on the first execution, deterministic at boot).
        if !pages.is_empty()
            && region_target_excluded(virt_target as u32)
            && !pages.contains(&phys_page)
        {
            return None;
        }

        // `>=` (not `==`): a proper ceiling. Equivalent to the original for the
        // single-cap default path (growth is monotonic), but required so the cap
        // holds when region formation seeds the page set above the base cap.
        if !pages.contains(&phys_page) && pages.len() as u32 >= max_pages
            || page_blacklist.contains(&phys_page)
        {
            return None;
        }

        if !pages.contains(&phys_page) {
            // page seen for the first time, handle entry points
            if let Some((hotness, entry_points)) = ctx.entry_points.get_mut(&phys_page) {
                let existing_entry_points = match ctx.pages.get(&phys_page) {
                    Some(PageInfo { entry_points, .. }) => {
                        HashSet::from_iter(entry_points.iter().map(|x| x.0))
                    },
                    None => HashSet::new(),
                };

                if entry_points
                    .iter()
                    .all(|entry_point| existing_entry_points.contains(entry_point))
                {
                    page_blacklist.insert(phys_page);
                    return None;
                }

                // XXX: Remove this paragraph
                //let old_length = entry_points.len();
                //entry_points.extend(existing_entry_points);
                //dbg_assert!(
                //    entry_points.union(&existing_entry_points).count() == entry_points.len()
                //);

                *hotness = 0;

                for &addr_low in entry_points.iter() {
                    let addr = virt_target & !0xFFF | addr_low as i32;
                    to_visit_stack.push(addr);
                    marked_as_entry.insert(addr);
                }
            }
            else {
                // no entry points: ignore this page?
                page_blacklist.insert(phys_page);
                return None;
            }

            pages.insert(phys_page);
            dbg_assert!(pages.len() as u32 <= max_pages);
        }

        to_visit_stack.push(virt_target);
        Some(phys_target)
    }

    let mut to_visit_stack: Vec<i32> = Vec::new();
    let mut marked_as_entry: HashSet<i32> = HashSet::new();
    let mut basic_blocks: BTreeMap<u32, BasicBlock> = BTreeMap::new();
    let mut pages: HashSet<Page> = HashSet::new();
    let mut page_blacklist = HashSet::new();

    // B3 hotness tiering: a compilation whose entry lands on a tier-2-promoted page
    // gets the expanded budgets (more pages per module + deeper RET-speculation).
    let tier2 = ctx.tier2_pages.len() > 0
        && entry_points.iter().any(|&virt| {
            match cpu::translate_address_read_no_side_effects(virt) {
                Ok(phys) => ctx.tier2_pages.contains(&Page::page_of(phys)),
                Err(()) => false,
            }
        });

    // 16-bit doesn't work correctly, most likely due to instruction pointer wrap-around
    // When indirect regions are on, use the (larger) region page budget as the
    // compilation-wide cap so dispatchers can absorb hot targets. Non-dispatcher
    // modules rarely reach it (hot direct-jump chains are short); it stays far
    // below the global-MAX_PAGES=48 setting that OOM'd V8.
    let max_pages = if cpu.state_flags.is_32() {
        let base = if unsafe { JIT_INDIRECT_REGIONS } {
            unsafe { MAX_PAGES.max(JIT_INDIRECT_REGION_MAX_PAGES) }
        } else {
            unsafe { MAX_PAGES }
        };
        if tier2 { base.max(unsafe { TIER2_MAX_PAGES }) } else { base }
    } else {
        1
    };

    for virt_addr in entry_points {
        let ok = follow_jump(
            virt_addr,
            ctx,
            &mut pages,
            &mut page_blacklist,
            max_pages,
            &mut marked_as_entry,
            &mut to_visit_stack,
        );
        dbg_assert!(ok.is_some());
        dbg_assert!(marked_as_entry.contains(&virt_addr));
    }

    while let Some(to_visit) = to_visit_stack.pop() {
        let phys_addr = match cpu::translate_address_read_no_side_effects(to_visit) {
            Err(()) => {
                dbg_log!("Not analysing {:x} (page not mapped)", to_visit);
                continue;
            },
            Ok(phys_addr) => phys_addr,
        };

        if basic_blocks.contains_key(&phys_addr) {
            continue;
        }

        if is_near_end_of_page(phys_addr) {
            // Empty basic block, don't insert
            profiler::stat_increment(stat::COMPILE_CUT_OFF_AT_END_OF_PAGE);
            continue;
        }

        let mut current_address = phys_addr;
        let mut current_block = BasicBlock {
            addr: current_address,
            virt_addr: to_visit,
            last_instruction_addr: 0,
            end_addr: 0,
            ty: BasicBlockType::Exit,
            is_entry_block: false,
            has_sti: false,
            number_of_instructions: 0,
            ret_speculation: Vec::new(),
        };
        loop {
            let addr_before_instruction = current_address;
            let mut cpu = &mut CpuContext {
                eip: current_address,
                ..cpu
            };
            let analysis = analysis::analyze_step(&mut cpu);
            let has_next_instruction = !analysis.no_next_instruction;
            current_address = cpu.eip;

            dbg_assert!(Page::page_of(current_address) == Page::page_of(addr_before_instruction));
            let current_virt_addr = to_visit & !0xFFF | current_address as i32 & 0xFFF;

            if analysis.ty == AnalysisType::STI && is_near_end_of_page(current_address) {
                // cut off before the STI so that it is handled by interpreted mode
                profiler::stat_increment(stat::COMPILE_CUT_OFF_AT_END_OF_PAGE);
                break;
            }

            current_block.number_of_instructions += 1;
            current_block.last_instruction_addr = addr_before_instruction;
            current_block.end_addr = current_address;

            match analysis.ty {
                AnalysisType::Normal | AnalysisType::STI => {
                    dbg_assert!(has_next_instruction);
                    dbg_assert!(!analysis.absolute_jump);

                    if current_block.has_sti {
                        // Convert next instruction after STI (i.e., the current instruction) into block boundary

                        marked_as_entry.insert(current_virt_addr);
                        to_visit_stack.push(current_virt_addr);

                        break;
                    }

                    if analysis.ty == AnalysisType::STI {
                        dbg_assert!(
                            !is_near_end_of_page(current_address),
                            "should be handled above"
                        );

                        current_block.has_sti = true;
                    }
                    else {
                        // Only split non-STI blocks (one instruction needs to run after STI before
                        // handle_irqs may be called)

                        if basic_blocks.contains_key(&current_address) {
                            dbg_assert!(!is_near_end_of_page(current_address));
                            current_block.ty = BasicBlockType::Normal {
                                next_block_addr: Some(current_address),
                                jump_offset: 0,
                                jump_offset_is_32: true,
                            };
                            break;
                        }
                    }
                },
                AnalysisType::Jump {
                    offset,
                    is_32,
                    condition: Some(condition),
                } => {
                    dbg_assert!(!analysis.absolute_jump);
                    // conditional jump: continue at next and continue at jump target

                    let jump_target = if is_32 {
                        current_virt_addr + offset
                    }
                    else {
                        cpu.cs_offset as i32
                            + (current_virt_addr - cpu.cs_offset as i32 + offset & 0xFFFF)
                    };

                    dbg_assert!(has_next_instruction);
                    to_visit_stack.push(current_virt_addr);

                    let next_block_addr = if is_near_end_of_page(current_address) {
                        None
                    }
                    else {
                        Some(current_address)
                    };

                    current_block.ty = BasicBlockType::ConditionalJump {
                        next_block_addr,
                        next_block_branch_taken_addr: follow_jump(
                            jump_target,
                            ctx,
                            &mut pages,
                            &mut page_blacklist,
                            max_pages,
                            &mut marked_as_entry,
                            &mut to_visit_stack,
                        ),
                        condition,
                        jump_offset: offset,
                        jump_offset_is_32: is_32,
                    };

                    break;
                },
                AnalysisType::Jump {
                    offset,
                    is_32,
                    condition: None,
                } => {
                    dbg_assert!(!analysis.absolute_jump);
                    // non-conditional jump: continue at jump target

                    let jump_target = if is_32 {
                        current_virt_addr + offset
                    }
                    else {
                        cpu.cs_offset as i32
                            + (current_virt_addr - cpu.cs_offset as i32 + offset & 0xFFFF)
                    };

                    if has_next_instruction {
                        // Execution will eventually come back to the next instruction (CALL)
                        marked_as_entry.insert(current_virt_addr);
                        to_visit_stack.push(current_virt_addr);
                    }

                    current_block.ty = BasicBlockType::Normal {
                        next_block_addr: follow_jump(
                            jump_target,
                            ctx,
                            &mut pages,
                            &mut page_blacklist,
                            max_pages,
                            &mut marked_as_entry,
                            &mut to_visit_stack,
                        ),
                        jump_offset: offset,
                        jump_offset_is_32: is_32,
                    };

                    break;
                },
                AnalysisType::BlockBoundary => {
                    // a block boundary but not a jump, get out

                    if has_next_instruction {
                        // block boundary, but execution will eventually come back
                        // to the next instruction. Create a new basic block
                        // starting at the next instruction and register it as an
                        // entry point
                        marked_as_entry.insert(current_virt_addr);
                        to_visit_stack.push(current_virt_addr);
                    }

                    if analysis.absolute_jump {
                        current_block.ty = BasicBlockType::AbsoluteEip;

                        // Tier-2R: grow the region across this indirect
                        // edge using profiled targets. Targets that join the
                        // region are marked as entries so the runtime
                        // jit_find_cache_entry_in_page re-dispatch stays
                        // intra-module instead of exiting to main_loop.
                        if unsafe { JIT_INDIRECT_REGIONS } {
                            let targets = trace_profiler::hot_indirect_targets(
                                addr_before_instruction,
                                JIT_INDIRECT_REGION_MAX_TARGETS,
                                unsafe { JIT_INDIRECT_REGION_MIN_SHARE } as u64,
                            );
                            // max_pages already carries the region budget (set at
                            // the top of jit_find_basic_blocks when regions are on).
                            // Targets are hottest-first, so the cap keeps the
                            // most-executed cases when it binds.
                            for target in targets {
                                if region_target_excluded(target) {
                                    // Thunk/callback/spin bucket — stub pages full of
                                    // OUT traps must never join a guest superblock.
                                    continue;
                                }
                                // Profiled eips are RAW runtime values: they can be
                                // stale (recorded before the page was overwritten) or
                                // land mid-instruction relative to the CURRENT bytes.
                                // Seeding such an offset as a dispatcher entry makes
                                // the runtime dispatch ENTER a misdecoded block —
                                // wrong ModRM/base → stores through garbage/NULL
                                // pointers at random guest sites. Direct edges
                                // only ever mark interpreter-registered boundaries;
                                // hold profiled targets to the same standard: the
                                // exact offset must be a registered entry point of
                                // its page (hot indirect targets are, via the
                                // module-exit hotness path). Conservative: an
                                // unregistered target just doesn't grow the region.
                                let registered = cpu::translate_address_read_no_side_effects(
                                    target as i32,
                                )
                                .ok()
                                .map_or(false, |phys| {
                                    ctx.entry_points
                                        .get(&Page::page_of(phys))
                                        .map_or(false, |(_, eps)| {
                                            eps.contains(&(phys as u16 & 0xFFF))
                                        })
                                });
                                if !registered {
                                    continue;
                                }
                                if follow_jump(
                                    target as i32,
                                    ctx,
                                    &mut pages,
                                    &mut page_blacklist,
                                    max_pages,
                                    &mut marked_as_entry,
                                    &mut to_visit_stack,
                                )
                                .is_some()
                                {
                                    marked_as_entry.insert(target as i32);
                                }
                            }
                        }
                    }

                    break;
                },
            }

            if is_near_end_of_page(current_address) {
                profiler::stat_increment(stat::COMPILE_CUT_OFF_AT_END_OF_PAGE);
                break;
            }
        }

        if current_block.number_of_instructions == 0 {
            // Empty basic block, don't insert (only happens when STI is found near end of page)
            continue;
        }

        let previous_block = basic_blocks
            .range(..current_block.addr)
            .next_back()
            .filter(|(_, previous_block)| !previous_block.has_sti)
            .map(|(_, previous_block)| previous_block);

        if let Some(previous_block) = previous_block {
            if current_block.addr < previous_block.end_addr {
                // If this block overlaps with the previous block, re-analyze the previous block
                to_visit_stack.push(previous_block.virt_addr);

                let addr = previous_block.addr;
                let old_block = basic_blocks.remove(&addr);
                dbg_assert!(old_block.is_some());

                // Note that this does not ensure the invariant that two consecutive blocks don't
                // overlay. For that, we also need to check the following block.
            }
        }

        dbg_assert!(current_block.addr < current_block.end_addr);
        dbg_assert!(current_block.addr <= current_block.last_instruction_addr);
        dbg_assert!(current_block.last_instruction_addr < current_block.end_addr);

        basic_blocks.insert(current_block.addr, current_block);
    }

    dbg_assert!(pages.len() as u32 <= max_pages);

    for block in basic_blocks.values_mut() {
        if marked_as_entry.contains(&block.virt_addr) {
            block.is_entry_block = true;
        }
    }

    // delete edges pointing to blocks that were dropped (currently only due to STI near the end of a page)
    let known_addresses: HashSet<u32> = basic_blocks.keys().copied().collect();
    for block in basic_blocks.values_mut() {
        match &mut block.ty {
            BasicBlockType::Normal {
                next_block_addr, ..
            } => {
                if next_block_addr.map_or(false, |a| !known_addresses.contains(&a)) {
                    *next_block_addr = None;
                }
            },
            BasicBlockType::ConditionalJump {
                next_block_addr,
                next_block_branch_taken_addr,
                ..
            } => {
                if next_block_addr.map_or(false, |a| !known_addresses.contains(&a)) {
                    *next_block_addr = None;
                }
                if next_block_branch_taken_addr.map_or(false, |a| !known_addresses.contains(&a)) {
                    *next_block_branch_taken_addr = None;
                }
            },
            BasicBlockType::Exit | BasicBlockType::AbsoluteEip => {},
        }
    }

    // RET-target speculation post-pass. For every module-local
    // call site (a Normal block whose jump target isn't its fall-through AND whose
    // fall-through was registered as an entry — the CALL discovery shape at the
    // Jump{condition:None} arm above), walk the callee's blocks within a bounded
    // instruction budget. If every path ends in a genuine RET (opcode C3/C2 — an
    // AbsoluteEip block can also be jmp/call r/m, which must NOT be speculated as a
    // return) with no nested calls/STI/module-exits, annotate those RET blocks with
    // the call site's return address. Wrong or stale candidates are harmless: the
    // emitter guards each with an eip compare and falls through to the normal
    // dispatch. A page dirtied under this module frees the whole module (multi-page
    // sweep in free_wasm_module), annotations included — no new SMC surface.
    if ret_speculation_enabled() {
        let fall_through_virt =
            |b: &BasicBlock| b.virt_addr & !0xFFF | b.end_addr as i32 & 0xFFF;

        let mut call_sites: Vec<(u32, i32, u32)> = Vec::new();
        for block in basic_blocks.values() {
            if let BasicBlockType::Normal { next_block_addr: Some(target), .. } = block.ty {
                if target != block.end_addr
                    && marked_as_entry.contains(&fall_through_virt(block))
                    && basic_blocks.contains_key(&block.end_addr)
                {
                    call_sites.push((target, fall_through_virt(block), block.end_addr));
                }
            }
        }

        let mut annotations: Vec<(u32, (i32, u32))> = Vec::new();
        for &(callee, ret_virt, ret_phys) in &call_sites {
            let mut visited: HashSet<u32> = HashSet::new();
            let mut stack = vec![callee];
            let mut instr_budget = unsafe {
                if tier2 { JIT_TIER2_RET_SPEC_MAX_INSTR } else { JIT_RET_SPEC_MAX_INSTR }
            };
            let mut rets: Vec<u32> = Vec::new();
            let mut ok = true;
            while let Some(addr) = stack.pop() {
                if !visited.insert(addr) {
                    continue;
                }
                let b = match basic_blocks.get(&addr) {
                    Some(b) => b,
                    None => {
                        ok = false;
                        break;
                    },
                };
                if b.has_sti {
                    ok = false;
                    break;
                }
                match instr_budget.checked_sub(b.number_of_instructions) {
                    Some(rest) => instr_budget = rest,
                    None => {
                        ok = false;
                        break;
                    },
                }
                match &b.ty {
                    BasicBlockType::AbsoluteEip => {
                        let opcode = memory::read8(b.last_instruction_addr) as u8;
                        if opcode == 0xC3 || opcode == 0xC2 {
                            rets.push(b.addr);
                        }
                        else {
                            ok = false;
                            break;
                        }
                    },
                    BasicBlockType::Normal { next_block_addr: Some(t), .. } => {
                        // A nested call returns INTO the callee, not to our site —
                        // its RETs must not be annotated with our return address.
                        if *t != b.end_addr && marked_as_entry.contains(&fall_through_virt(b)) {
                            ok = false;
                            break;
                        }
                        stack.push(*t);
                    },
                    BasicBlockType::Normal { next_block_addr: None, .. }
                    | BasicBlockType::Exit => {
                        ok = false;
                        break;
                    },
                    BasicBlockType::ConditionalJump {
                        next_block_addr,
                        next_block_branch_taken_addr,
                        ..
                    } => {
                        match (next_block_addr, next_block_branch_taken_addr) {
                            (Some(n), Some(t)) => {
                                stack.push(*n);
                                stack.push(*t);
                            },
                            _ => {
                                ok = false;
                                break;
                            },
                        }
                    },
                }
            }
            if ok {
                for ret_addr in rets {
                    annotations.push((ret_addr, (ret_virt, ret_phys)));
                }
            }
        }

        for (ret_addr, cand) in annotations {
            if let Some(b) = basic_blocks.get_mut(&ret_addr) {
                if b.ret_speculation.len() < RET_SPEC_MAX_CANDIDATES
                    && !b.ret_speculation.contains(&cand)
                {
                    b.ret_speculation.push(cand);
                }
            }
        }
    }

    let basic_blocks: Vec<BasicBlock> = basic_blocks.into_iter().map(|(_, block)| block).collect();

    for i in 0..basic_blocks.len() - 1 {
        let next_block_addr = basic_blocks[i + 1].addr;
        let next_block_end_addr = basic_blocks[i + 1].end_addr;
        let next_block_is_entry = basic_blocks[i + 1].is_entry_block;
        let block = &basic_blocks[i];
        dbg_assert!(block.addr < next_block_addr);
        if next_block_addr < block.end_addr {
            dbg_log!(
                "Overlapping first=[from={:x} to={:x} is_entry={}] second=[from={:x} to={:x} is_entry={}]",
                block.addr,
                block.end_addr,
                block.is_entry_block as u8,
                next_block_addr,
                next_block_end_addr,
                next_block_is_entry as u8
            );
        }
    }

    basic_blocks
}

#[no_mangle]
#[cfg(debug_assertions)]
pub fn jit_force_generate_unsafe(virt_addr: i32) {
    dbg_assert!(
        !is_near_end_of_page(virt_addr as u32),
        "cannot force compile near end of page"
    );
    jit_increase_hotness_and_maybe_compile(
        virt_addr,
        cpu::translate_address_read(virt_addr).unwrap(),
        cpu::get_seg_cs() as u32,
        cpu::get_state_flags(),
        JIT_THRESHOLD,
    );
    dbg_assert!(get_jit_state().compiling.is_some());
}

#[inline(never)]
fn jit_analyze_and_generate(
    ctx: &mut JitState,
    virt_entry_point: i32,
    phys_entry_point: u32,
    cs_offset: u32,
    state_flags: CachedStateFlags,
) {
    let page = Page::page_of(phys_entry_point);

    dbg_assert!(ctx.compiling.is_none());

    let (_, entry_points) = match ctx.entry_points.get(&page) {
        None => return,
        Some(entry_points) => entry_points,
    };

    let existing_entry_points = match ctx.pages.get(&page) {
        Some(PageInfo { entry_points, .. }) => HashSet::from_iter(entry_points.iter().map(|x| x.0)),
        None => HashSet::new(),
    };

    if entry_points
        .iter()
        .all(|entry_point| existing_entry_points.contains(entry_point))
    {
        profiler::stat_increment(stat::COMPILE_SKIPPED_NO_NEW_ENTRY_POINTS);
        return;
    }

    // XXX: check and remove
    //let old_length = entry_points.len();
    //entry_points.extend(existing_entry_points);
    //dbg_log!(
    //    "{} + {} = {}",
    //    entry_points.len(),
    //    existing_entry_points.len(),
    //    entry_points.union(&existing_entry_points).count()
    //);
    //dbg_assert!(entry_points.union(&existing_entry_points).count() == entry_points.len());

    profiler::stat_increment(stat::COMPILE);

    let cpu = CpuContext {
        eip: 0,
        prefixes: 0,
        cs_offset,
        state_flags,
    };

    dbg_assert!(
        cpu::translate_address_read_no_side_effects(virt_entry_point).unwrap() == phys_entry_point
    );
    let virt_page = Page::page_of(virt_entry_point as u32);
    let entry_points: HashSet<i32> = entry_points
        .iter()
        .map(|e| virt_page.to_address() as i32 | *e as i32)
        .collect();
    let basic_blocks = jit_find_basic_blocks(ctx, entry_points, cpu.clone());

    let mut pages = HashSet::new();

    for b in basic_blocks.iter() {
        // Remove this assertion once page-crossing jit is enabled
        dbg_assert!(Page::page_of(b.addr) == Page::page_of(b.end_addr));
        pages.insert(Page::page_of(b.addr));
    }

    let print = false;

    for b in basic_blocks.iter() {
        if !print {
            break;
        }
        let last_instruction_opcode = memory::read32s(b.last_instruction_addr);
        let op = opstats::decode(last_instruction_opcode as u32);
        dbg_log!(
            "BB: 0x{:x} {}{:02x} {} {}",
            b.addr,
            if op.is_0f { "0f" } else { "" },
            op.opcode,
            if b.is_entry_block { "entry" } else { "noentry" },
            match &b.ty {
                BasicBlockType::ConditionalJump {
                    next_block_addr: Some(next_block_addr),
                    next_block_branch_taken_addr: Some(next_block_branch_taken_addr),
                    ..
                } => format!(
                    "0x{:x} 0x{:x}",
                    next_block_addr, next_block_branch_taken_addr
                ),
                BasicBlockType::ConditionalJump {
                    next_block_addr: None,
                    next_block_branch_taken_addr: Some(next_block_branch_taken_addr),
                    ..
                } => format!("0x{:x}", next_block_branch_taken_addr),
                BasicBlockType::ConditionalJump {
                    next_block_addr: Some(next_block_addr),
                    next_block_branch_taken_addr: None,
                    ..
                } => format!("0x{:x}", next_block_addr),
                BasicBlockType::ConditionalJump {
                    next_block_addr: None,
                    next_block_branch_taken_addr: None,
                    ..
                } => format!(""),
                BasicBlockType::Normal {
                    next_block_addr: Some(next_block_addr),
                    ..
                } => format!("0x{:x}", next_block_addr),
                BasicBlockType::Normal {
                    next_block_addr: None,
                    ..
                } => format!(""),
                BasicBlockType::Exit => format!(""),
                BasicBlockType::AbsoluteEip => format!(""),
            }
        );
    }

    let graph = control_flow::make_graph(&basic_blocks);
    let mut structure = control_flow::loopify(&graph);

    if print {
        dbg_log!("before blockify:");
        for group in &structure {
            dbg_log!("=> Group");
            group.print(0);
        }
    }

    control_flow::blockify(&mut structure, &graph);

    if cfg!(debug_assertions) {
        control_flow::assert_invariants(&structure);
    }

    if print {
        dbg_log!("after blockify:");
        for group in &structure {
            dbg_log!("=> Group");
            group.print(0);
        }
    }

    if ctx.wasm_table_index_free_list.is_empty() {
        dbg_log!("wasm_table_index_free_list empty, clearing cache");

        // When no free slots are available, delete all cached modules. We could increase the
        // size of the table, but this way the initial size acts as an upper bound for the
        // number of wasm modules that we generate, which we want anyway to avoid getting our
        // tab killed by browsers due to memory constraints.
        jit_clear_cache(ctx);

        profiler::stat_increment(stat::INVALIDATE_ALL_MODULES_NO_FREE_WASM_INDICES);

        dbg_log!(
            "after jit_clear_cache: {} free",
            ctx.wasm_table_index_free_list.len(),
        );

        // This assertion can fail if all entries are pending (not possible unless
        // WASM_TABLE_SIZE is set very low)
        dbg_assert!(!ctx.wasm_table_index_free_list.is_empty());
    }

    // allocate an index in the wasm table
    let wasm_table_index = ctx
        .wasm_table_index_free_list
        .pop()
        .expect("allocate wasm table index");
    dbg_assert!(wasm_table_index != WasmTableIndex(0));

    dbg_assert!(!pages.is_empty());
    // The effective cap can exceed the global MAX_PAGES when indirect regions or
    // tier-2 budgets are active (see max_pages in jit_find_basic_blocks) — assert
    // against the widest configured budget, not the base knob.
    dbg_assert!(
        pages.len()
            <= unsafe {
                MAX_PAGES
                    .max(JIT_INDIRECT_REGION_MAX_PAGES)
                    .max(TIER2_MAX_PAGES)
            } as usize
    );

    let basic_block_by_addr: HashMap<u32, BasicBlock> =
        basic_blocks.into_iter().map(|b| (b.addr, b)).collect();

    let entries = jit_generate_module(
        structure,
        &basic_block_by_addr,
        cpu,
        &mut ctx.wasm_builder,
        wasm_table_index,
        state_flags,
    );
    dbg_assert!(!entries.is_empty());

    let mut page_info = HashMap::new();
    for &(addr, state) in &entries {
        let code = page_info
            .entry(Page::page_of(addr))
            .or_insert_with(|| PageInfo {
                wasm_table_index,
                state_flags,
                entry_points: Vec::new(),
                hidden_wasm_table_indices: Vec::new(),
            });
        code.entry_points.push((addr as u16 & 0xFFF, state));
    }
    // Invalidation completeness: EVERY page the module compiled code from must be
    // findable by jit_dirty_page, or a write to it leaves a STALE module running old
    // code. page_info above only covers pages that materialized a dispatcher entry;
    // entry blocks can be dropped by overlap elimination or near-end-of-page cutoffs
    // while non-entry blocks from the page remain. Register the rest with an empty
    // entry list — set_tlb_code leaves their state_table all-miss (u16::MAX), so the
    // only effect is that free_wasm_module's page sweep covers them.
    for &p in &pages {
        page_info.entry(p).or_insert_with(|| PageInfo {
            wasm_table_index,
            state_flags,
            entry_points: Vec::new(),
            hidden_wasm_table_indices: Vec::new(),
        });
    }

    profiler::stat_increment_by(
        stat::COMPILE_WASM_TOTAL_BYTES,
        ctx.wasm_builder.get_output_len() as u64,
    );
    profiler::stat_increment_by(stat::COMPILE_PAGE, pages.len() as u64);

    for &p in &pages {
        ctx.entry_points
            .entry(p)
            .or_insert_with(|| (0, HashSet::new()));
    }

    cpu::tlb_set_has_code_multiple(&pages, true);

    dbg_assert!(ctx.compiling.is_none());
    ctx.compiling = Some((
        wasm_table_index,
        CompilingPageState::Compiling { pages: page_info },
    ));

    let phys_addr = page.to_address();

    // will call codegen_finalize_finished asynchronously when finished
    codegen_finalize(
        wasm_table_index,
        phys_addr,
        state_flags,
        ctx.wasm_builder.get_output_ptr() as u32,
        ctx.wasm_builder.get_output_len(),
    );

    check_jit_state_invariants(ctx);
}

// [wasm_table_index, phys_addr, error_kind, recovery_status]
static mut CODEGEN_FINALIZE_FAILURE_LAST: [u32; 4] = [0; 4];
static mut CODEGEN_FINALIZE_FAILURE_COUNT: u32 = 0;

#[no_mangle]
pub fn codegen_get_finalize_failure_count() -> u32 {
    unsafe { CODEGEN_FINALIZE_FAILURE_COUNT }
}

const AOT_TX_MAX_PAGES: usize = 256;
const AOT_TX_MAX_ENTRIES_PER_PAGE: usize = 4096;
const AOT_TX_MAX_ENTRIES: usize = 65_536;

struct AotStagedPage {
    page: Page,
    info: PageInfo,
    expected_entries: usize,
}

struct AotTransaction {
    wasm_table_index: WasmTableIndex,
    expected_pages: usize,
    pages: Vec<AotStagedPage>,
    building: Option<AotStagedPage>,
    entry_total: usize,
    ready: bool,
    #[cfg(debug_assertions)]
    debug_pages: HashSet<Page>,
}

#[no_mangle]
pub fn codegen_get_finalize_failure_info(i: u32) -> u32 {
    unsafe {
        (&*std::ptr::addr_of!(CODEGEN_FINALIZE_FAILURE_LAST))
            .get(i as usize)
            .copied()
            .unwrap_or(0)
    }
}

#[no_mangle]
pub fn codegen_is_compiling() -> u32 {
    let ctx = get_jit_state();
    ctx.compiling.is_some() as u32
}

#[no_mangle]
pub fn codegen_finalize_failed(
    wasm_table_index: WasmTableIndex,
    phys_addr: u32,
    error_kind: u32,
) -> u32 {
    let mut ctx = get_jit_state();
    let recovery_status = if ctx.compiling.is_none() {
        1
    }
    else if ctx.compiling.as_ref().unwrap().0 != wasm_table_index {
        2
    }
    else {
        let (pending_wasm_table_index, _) = ctx.compiling.take().unwrap();
        dbg_assert!(pending_wasm_table_index == wasm_table_index);
        free_wasm_table_index(&mut ctx, pending_wasm_table_index);
        check_jit_state_invariants(&mut ctx);
        0
    };

    unsafe {
        CODEGEN_FINALIZE_FAILURE_COUNT = CODEGEN_FINALIZE_FAILURE_COUNT.wrapping_add(1);
        CODEGEN_FINALIZE_FAILURE_LAST = [
            wasm_table_index.to_u16() as u32,
            phys_addr,
            error_kind,
            recovery_status,
        ];
    }

    recovery_status
}

#[no_mangle]
pub fn codegen_finalize_finished(
    wasm_table_index: WasmTableIndex,
    phys_addr: u32,
    state_flags: CachedStateFlags,
) {
    let mut ctx = get_jit_state();

    dbg_assert!(wasm_table_index != WasmTableIndex(0));

    dbg_log!(
        "Finished compiling for page at {:x}",
        Page::page_of(phys_addr).to_address()
    );

    let pages = match mem::replace(&mut ctx.compiling, None) {
        None => {
            dbg_assert!(false);
            return;
        },
        Some((in_progress_wasm_table_index, CompilingPageState::CompilingWritten)) => {
            dbg_assert!(wasm_table_index == in_progress_wasm_table_index);

            profiler::stat_increment(stat::INVALIDATE_MODULE_WRITTEN_WHILE_COMPILED);
            free_wasm_table_index(&mut ctx, wasm_table_index);
            check_jit_state_invariants(&mut ctx);
            return;
        },
        Some((in_progress_wasm_table_index, CompilingPageState::Compiling { pages })) => {
            dbg_assert!(wasm_table_index == in_progress_wasm_table_index);
            dbg_assert!(!pages.is_empty());
            pages
        },
    };

    for i in 0..unsafe { cpu::valid_tlb_entries_count } {
        let page = unsafe { cpu::valid_tlb_entries[i as usize] };
        let entry = unsafe { cpu::tlb_data[page as usize] };
        if 0 != entry {
            let tlb_physical_page = Page::of_u32(
                (entry as u32 >> 12 ^ page as u32) - (unsafe { memory::mem8 } as u32 >> 12),
            );
            if let Some(info) = pages.get(&tlb_physical_page) {
                set_tlb_code(
                    Page::of_u32(page as u32),
                    wasm_table_index,
                    &info.entry_points,
                    state_flags,
                );
            }
        }
    }

    #[cfg(debug_assertions)]
    if CHECK_JIT_STATE_INVARIANTS {
        ctx.wasm_table_index_to_page
            .insert(wasm_table_index, pages.keys().copied().collect());
    }

    let mut check_for_unused_wasm_table_index = HashSet::new();

    for (page, mut info) in pages {
        if let Some(old_entry) = ctx.pages.remove(&page) {
            info.hidden_wasm_table_indices
                .extend(old_entry.hidden_wasm_table_indices);
            info.hidden_wasm_table_indices
                .push(old_entry.wasm_table_index);
            check_for_unused_wasm_table_index.insert(old_entry.wasm_table_index);
        }
        ctx.pages.insert(page, info);
    }

    let unused: Vec<&WasmTableIndex> = check_for_unused_wasm_table_index
        .iter()
        .filter(|&&i| ctx.pages.values().all(|page| page.wasm_table_index != i))
        .collect();

    for &index in unused {
        for p in ctx.pages.values_mut() {
            p.hidden_wasm_table_indices.retain(|&w| w != index);
        }

        dbg_log!("unused after overwrite {}", index.to_u16());
        profiler::stat_increment(stat::INVALIDATE_MODULE_UNUSED_AFTER_OVERWRITE);
        free_wasm_table_index(&mut ctx, index);
    }

    check_jit_state_invariants(&mut ctx);
}

// ─── AOT units (MS-A stage 1) ────────────────────────────────────────────────
//
// An AOT unit PRETENDS TO BE A JIT MODULE: it is published into exactly the structures
// codegen_finalize_finished publishes into, so dispatch, tier-2, SMC invalidation
// (jit_dirty_page via the TLB HAS_CODE bit) and free_wasm_table_index all treat it like
// any other module. Only the SOURCE OF THE MODULE BYTES differs — which is the whole
// design principle of the track (ms-aot-design.md §4.1).
//
// The caller (JS) owns instantiation: it compiles the bytes with the SAME `jit_imports`
// object the live path uses, writes the exported `f` into wasm_table[index + 1024], and
// only then registers here. Content binding (§5.1 — the sha of the page AFTER relocation,
// IAT patching and hle-lib patches) is verified caller-side against guest memory, which
// keeps the Rust surface at what the RFC asked for: registration, nothing else.
//
// The staged builder never reads caller memory and never truncates caller-provided values.
// A successful `prepare_finish` means all validation and recoverable allocation are complete;
// `commit` can therefore move the complete unit into ctx.pages without a fallible step.
const AOT_TX_OK: u32 = 0;
const AOT_TX_BAD_STATE: u32 = 1;
const AOT_TX_BAD_INDEX: u32 = 2;
const AOT_TX_SLOT_UNAVAILABLE: u32 = 3;
const AOT_TX_FINGERPRINT_MISMATCH: u32 = 4;
const AOT_TX_BAD_PAGE_COUNT: u32 = 5;
const AOT_TX_BAD_PAGE: u32 = 6;
const AOT_TX_PAGE_OWNED: u32 = 7;
const AOT_TX_BAD_FLAGS: u32 = 8;
const AOT_TX_BAD_ENTRY: u32 = 9;
const AOT_TX_NO_ENTRIES: u32 = 10;
const AOT_TX_CAPACITY: u32 = 11;

#[inline]
fn aot_tx_release(ctx: &mut JitState, tx: AotTransaction) {
    let index = tx.wasm_table_index;
    drop(tx);
    free_wasm_table_index(ctx, index);
}

#[no_mangle]
pub fn jit_aot_tx_begin(wasm_table_index: u32, page_count: u32, fp_lo: u32, fp_hi: u32) -> u32 {
    let mut ctx = get_jit_state();
    if ctx.aot_staged.is_some() { return AOT_TX_BAD_STATE; }
    if wasm_table_index == 0 || wasm_table_index >= WASM_TABLE_SIZE as u32 { return AOT_TX_BAD_INDEX; }
    if page_count == 0 || page_count as usize > AOT_TX_MAX_PAGES { return AOT_TX_BAD_PAGE_COUNT; }
    let fingerprint = jit_codegen_fingerprint();
    if fp_lo != fingerprint as u32 || fp_hi != (fingerprint >> 32) as u32 {
        return AOT_TX_FINGERPRINT_MISMATCH;
    }
    let index = WasmTableIndex(wasm_table_index as u16);
    if ctx.compiling.as_ref().map_or(false, |(i, _)| *i == index)
        || ctx.pages.values().any(|info| info.wasm_table_index == index || info.hidden_wasm_table_indices.contains(&index))
    {
        return AOT_TX_SLOT_UNAVAILABLE;
    }
    let free_at = match ctx.wasm_table_index_free_list.iter().position(|&i| i == index) {
        Some(position) => position,
        None => return AOT_TX_SLOT_UNAVAILABLE,
    };
    let count = page_count as usize;
    let mut pages = Vec::new();
    #[cfg(debug_assertions)]
    let mut debug_pages = HashSet::new();
    #[allow(unused_mut)]
    let mut reserve_failed = pages.try_reserve_exact(count).is_err()
        || ctx.pages.try_reserve(count).is_err()
        || ctx.tier2_pages.try_reserve(count).is_err()
        || ctx.tier2_blocked_pages.try_reserve(count).is_err();
    #[cfg(debug_assertions)]
    {
        reserve_failed = reserve_failed
            || ctx.wasm_table_index_to_page.try_reserve(1).is_err()
            || debug_pages.try_reserve(count).is_err();
    }
    if reserve_failed {
        return AOT_TX_CAPACITY;
    }
    ctx.wasm_table_index_free_list.swap_remove(free_at);
    ctx.aot_staged = Some(AotTransaction {
        wasm_table_index: index,
        expected_pages: count,
        pages,
        building: None,
        entry_total: 0,
        ready: false,
        #[cfg(debug_assertions)]
        debug_pages,
    });
    check_jit_state_invariants(&mut ctx);
    AOT_TX_OK
}

#[no_mangle]
pub fn jit_aot_tx_page_begin(phys_addr: u32, state_flags: u32, entry_count: u32) -> u32 {
    let mut ctx = get_jit_state();
    let ram = unsafe { *global_pointers::memory_size };
    if phys_addr & 0xFFF != 0 || phys_addr > ram || ram - phys_addr < 0x1000 { return AOT_TX_BAD_PAGE; }
    if state_flags & !0x0F != 0 { return AOT_TX_BAD_FLAGS; }
    let page = Page::page_of(phys_addr);
    if ctx.pages.contains_key(&page) { return AOT_TX_PAGE_OWNED; }
    let tx = match ctx.aot_staged.as_mut() { Some(tx) if !tx.ready => tx, _ => return AOT_TX_BAD_STATE };
    if tx.building.is_some() || tx.pages.len() >= tx.expected_pages { return AOT_TX_BAD_STATE; }
    let count = entry_count as usize;
    if count > AOT_TX_MAX_ENTRIES_PER_PAGE || tx.entry_total.saturating_add(count) > AOT_TX_MAX_ENTRIES {
        return AOT_TX_BAD_ENTRY;
    }
    if tx.pages.iter().any(|p| p.page == page) { return AOT_TX_PAGE_OWNED; }
    let mut entries = Vec::new();
    if entries.try_reserve_exact(count).is_err() { return AOT_TX_CAPACITY; }
    tx.building = Some(AotStagedPage {
        page,
        info: PageInfo {
            wasm_table_index: tx.wasm_table_index,
            hidden_wasm_table_indices: vec![],
            entry_points: entries,
            state_flags: CachedStateFlags::of_u32(state_flags),
        },
        expected_entries: count,
    });
    AOT_TX_OK
}

#[no_mangle]
pub fn jit_aot_tx_entry_push(page_offset: u32, initial_state: u32) -> u32 {
    let mut ctx = get_jit_state();
    let tx = match ctx.aot_staged.as_mut() { Some(tx) if !tx.ready => tx, _ => return AOT_TX_BAD_STATE };
    let page = match tx.building.as_mut() { Some(page) => page, None => return AOT_TX_BAD_STATE };
    if page_offset > 0xFFF || initial_state > u16::MAX as u32 || page.info.entry_points.len() >= page.expected_entries {
        return AOT_TX_BAD_ENTRY;
    }
    page.info.entry_points.push((page_offset as u16, initial_state as u16));
    AOT_TX_OK
}

#[no_mangle]
pub fn jit_aot_tx_page_finish() -> u32 {
    let mut ctx = get_jit_state();
    let tx = match ctx.aot_staged.as_mut() { Some(tx) if !tx.ready => tx, _ => return AOT_TX_BAD_STATE };
    let page = match tx.building.take() { Some(page) => page, None => return AOT_TX_BAD_STATE };
    if page.info.entry_points.len() != page.expected_entries {
        tx.building = Some(page);
        return AOT_TX_BAD_ENTRY;
    }
    tx.entry_total += page.info.entry_points.len();
    #[cfg(debug_assertions)]
    { tx.debug_pages.insert(page.page); }
    tx.pages.push(page);
    AOT_TX_OK
}

#[no_mangle]
pub fn jit_aot_tx_prepare_finish() -> u32 {
    let mut ctx = get_jit_state();
    let tx = match ctx.aot_staged.as_mut() { Some(tx) if !tx.ready => tx, _ => return AOT_TX_BAD_STATE };
    if tx.building.is_some() || tx.pages.len() != tx.expected_pages { return AOT_TX_BAD_STATE; }
    if tx.entry_total == 0 { return AOT_TX_NO_ENTRIES; }
    tx.ready = true;
    AOT_TX_OK
}

#[no_mangle]
pub fn jit_aot_tx_abort() -> u32 {
    let mut ctx = get_jit_state();
    match ctx.aot_staged.take() {
        Some(tx) => {
            aot_tx_release(&mut ctx, tx);
            check_jit_state_invariants(&mut ctx);
            AOT_TX_OK
        },
        None => AOT_TX_BAD_STATE,
    }
}

#[no_mangle]
pub fn jit_aot_tx_commit() -> u32 {
    let mut ctx = get_jit_state();
    if !ctx.aot_staged.as_ref().map_or(false, |tx| tx.ready) { return AOT_TX_BAD_STATE; }
    let tx = ctx.aot_staged.take().unwrap();
    #[cfg(debug_assertions)]
    let index = tx.wasm_table_index;
    let page_count = tx.pages.len();
    #[cfg(debug_assertions)]
    let debug_pages = tx.debug_pages;
    for staged in tx.pages {
        let page = staged.page;
        ctx.pages.insert(page, staged.info);
        if ctx.tier2_pages.len() < TIER2_PAGE_SET_CAP {
            ctx.tier2_pages.insert(page);
            ctx.tier2_blocked_pages.remove(&page);
        }
        else {
            unsafe { AOT_TIER2_BLOCKED_BY_CAP += 1 };
            ctx.tier2_blocked_pages.insert(page);
        }
    }
    #[cfg(debug_assertions)]
    { ctx.wasm_table_index_to_page.insert(index, debug_pages); }
    unsafe { AOT_REGISTERED = AOT_REGISTERED.wrapping_add(page_count as u32); }
    check_jit_state_invariants(&mut ctx);
    AOT_TX_OK
}

#[no_mangle]
pub fn jit_aot_tx_staged_index() -> u32 {
    get_jit_state().aot_staged.as_ref().map_or(0xFFFF, |tx| tx.wasm_table_index.to_u16() as u32)
}

#[no_mangle]
pub fn jit_aot_free_table_index_count() -> u32 { get_jit_state().wasm_table_index_free_list.len() as u32 }

/// Retired compatibility export. Exact-slot reservation belongs exclusively to
/// `jit_aot_tx_begin`; this refuses without changing ownership.
#[no_mangle]
pub fn jit_aot_alloc_table_index() -> u32 {
    // Retired: exact-slot ownership belongs exclusively to jit_aot_tx_begin.
    0xFFFF
}

/// Retired compatibility export. `jit_aot_tx_abort` is the only release path.
#[no_mangle]
pub fn jit_aot_free_table_index(wasm_table_index: u32) {
    let _ = wasm_table_index;
    // Retired: abort is the only way to release a transaction reservation.
}

/// Retired compatibility export. Per-page publication is refused to preserve atomic units.
#[no_mangle]
pub fn jit_register_aot_module(wasm_table_index: u32, phys_addr: u32, state_flags: u32) -> u32 {
    let _ = (wasm_table_index, phys_addr, state_flags);
    // Retired: per-page publication would violate unit atomicity.
    AOT_TX_BAD_STATE
}

static mut AOT_TIER2_BLOCKED_BY_CAP: u32 = 0;

/// Registrations that could not mark their page tier-2 because the page-set cap was full.
/// Nonzero means some units will still be evicted at their first promotion.
#[no_mangle]
pub fn jit_aot_tier2_blocked_count() -> u32 { unsafe { AOT_TIER2_BLOCKED_BY_CAP } }

/// Drop every TLB entry so the next access to a registered page rebuilds it — and
/// `update_tlb_code` then stamps dispatch meta FROM `ctx.pages`, i.e. correct by
/// construction.
///
/// Both obvious alternatives are wrong. Stamping eagerly at registration (what the compile
/// path does) hands a virtual page a module compiled for a different one, since an AOT unit is
/// published long after the mapping it was built under. Not stamping at all is worse in a
/// quieter way: a page already in the TLB never gets meta, so dispatch never enters the unit
/// while `ctx.pages` still names it the page's owner — the JIT will not compile it either, and
/// the guest runs that code INTERPRETED.
///
/// Call once after a batch of registrations, not per unit.
#[no_mangle]
pub fn jit_aot_flush_tlb() { unsafe { cpu::full_clear_tlb() } }

static mut AOT_REGISTERED: u32 = 0;

#[no_mangle]
pub fn jit_aot_registered_count() -> u32 { unsafe { AOT_REGISTERED } }

// Readback of a live page's publication record — this is what lets a RECORDING pass
// capture what a replay pass must reproduce (entry points are computed in Rust and are
// otherwise invisible to the JS side that owns the module bytes).
#[no_mangle]
pub fn jit_aot_page_table_index(phys_addr: u32) -> u32 {
    match get_jit_state().pages.get(&Page::page_of(phys_addr)) {
        Some(info) => info.wasm_table_index.to_u16() as u32,
        None => 0xFFFF,
    }
}

#[no_mangle]
pub fn jit_aot_page_state_flags(phys_addr: u32) -> u32 {
    match get_jit_state().pages.get(&Page::page_of(phys_addr)) {
        Some(info) => info.state_flags.to_u32(),
        None => 0xFFFF_FFFF,
    }
}

// A compiled module usually spans SEVERAL pages (MAX_PAGES), and every one of them is
// published with its own entry-point list pointing at the same table index. Capturing only
// the entry page would leave the rest to be recompiled separately — two modules covering
// overlapping code. These two let the caller enumerate a module's full page set.
#[no_mangle]
pub fn jit_aot_module_page_count(wasm_table_index: u32) -> u32 {
    let idx = WasmTableIndex(wasm_table_index as u16);
    get_jit_state()
        .pages
        .iter()
        .filter(|(_, info)| info.wasm_table_index == idx)
        .count() as u32
}

/// n-th page address of the module in `wasm_table_index`; 0xFFFF_FFFF when out of range.
#[no_mangle]
pub fn jit_aot_module_page_at(wasm_table_index: u32, n: u32) -> u32 {
    let idx = WasmTableIndex(wasm_table_index as u16);
    match get_jit_state()
        .pages
        .iter()
        .filter(|(_, info)| info.wasm_table_index == idx)
        .nth(n as usize)
    {
        Some((page, _)) => page.to_address(),
        None => 0xFFFF_FFFF,
    }
}

#[no_mangle]
pub fn jit_aot_page_entry_count(phys_addr: u32) -> u32 {
    match get_jit_state().pages.get(&Page::page_of(phys_addr)) {
        Some(info) => info.entry_points.len() as u32,
        None => 0,
    }
}

/// i-th entry point as (page_offset << 16) | initial_state; 0xFFFF_FFFF when out of range.
#[no_mangle]
pub fn jit_aot_page_entry_at(phys_addr: u32, i: u32) -> u32 {
    match get_jit_state().pages.get(&Page::page_of(phys_addr)) {
        Some(info) => match info.entry_points.get(i as usize) {
            Some(&(off, state)) => ((off as u32) << 16) | state as u32,
            None => 0xFFFF_FFFF,
        },
        None => 0xFFFF_FFFF,
    }
}

pub fn update_tlb_code(virt_page: Page, phys_page: Page) {
    let ctx = get_jit_state();

    match ctx.pages.get(&phys_page) {
        Some(PageInfo {
            wasm_table_index,
            entry_points,
            state_flags,
            hidden_wasm_table_indices: _,
        }) => set_tlb_code(virt_page, *wasm_table_index, entry_points, *state_flags),
        None => cpu::clear_tlb_code(phys_page.to_u32() as i32),
    };
}

// Publish a page's dispatch entries into the DOD SoA (see DISPATCH_META above).
// The per-unit fastmem generation is intentionally NOT stored: the unit's own
// prologue guard self-deopts a stale unit on entry (see the SoA header comment).
pub fn set_tlb_code(
    virt_page: Page,
    wasm_table_index: WasmTableIndex,
    entries: &Vec<(u16, u16)>,
    state_flags: CachedStateFlags,
) {
    dispatch_meta_set(virt_page, wasm_table_index, entries, state_flags);
}

// Statically-chainable direct-jump exit: record the exit as chainable and branch to the
// module exit label (main_loop re-dispatch). (Static block-chaining was removed; the live
// cross-module chaining mechanism is the dynamic RET path, set_jit_config idx 12.)
fn gen_chain_or_exit_to_known_successor(
    ctx: &mut JitContext,
    _state_flags: CachedStateFlags,
    _last_instruction_addr: u32,
) {
    codegen::gen_dispatch_stat_increment(ctx.builder, stat::MODULE_EXIT_CHAINABLE);
    ctx.builder.br(ctx.exit_label);
}

fn jit_generate_module(
    structure: Vec<WasmStructure>,
    basic_blocks: &HashMap<u32, BasicBlock>,
    mut cpu: CpuContext,
    builder: &mut WasmBuilder,
    wasm_table_index: WasmTableIndex,
    state_flags: CachedStateFlags,
) -> Vec<(u32, u16)> {
    builder.reset();
    builder.branch_hint_mask = branch_hint_mask();
    builder.branch_hint_offset_fuzz = unsafe { JIT_BRANCH_HINT_OFFSET_FUZZ };

    let mut register_locals = (0..8)
        .map(|i| {
            builder.load_fixed_i32(global_pointers::get_reg32_offset(i));
            builder.set_new_local()
        })
        .collect();

    builder.const_i32(0);
    let instruction_counter = builder.set_new_local();

    // Flag-locals (idx 21): lazy-flag tuple lives in wasm locals for the whole
    // module — initialized from the memory globals here, spilled back at every
    // exit epilogue and around every non-whitelisted helper call (builder funnel).
    if flag_locals_enabled() {
        let addrs: [u32; 5] = [
            global_pointers::last_op1 as u32,
            global_pointers::last_result as u32,
            global_pointers::last_op_size as u32,
            global_pointers::flags_changed as u32,
            global_pointers::flags as u32,
        ];
        let mut locals = [(0u8, 0u32); 5];
        for (i, &addr) in addrs.iter().enumerate() {
            builder.load_fixed_i32(addr);
            let l = builder.set_new_local();
            locals[i] = (l.idx(), addr);
            std::mem::forget(l); // whole-module lifetime; freed via free_flag_locals
        }
        builder.flag_locals = Some(locals);
    }

    let exit_label = builder.block_void();
    let exit_with_fault_label = builder.block_void();
    let main_loop_label = builder.loop_void();
    if unsafe { JIT_USE_LOOP_SAFETY } {
        builder.get_local(&instruction_counter);
        builder.const_i32(cpu::LOOP_COUNTER);
        builder.geu_i32();
        if cfg!(feature = "profiler") {
            builder.if_void();
            codegen::gen_debug_track_jit_exit(builder, 0);
            builder.br(exit_label);
            builder.block_end();
        }
        else {
            builder.br_if(exit_label);
        }
    }
    let brtable_default = builder.block_void();

    let ctx = &mut JitContext {
        cpu: &mut cpu,
        builder,
        register_locals: &mut register_locals,
        start_of_current_instruction: 0,
        exit_with_fault_label,
        exit_label,
        current_instruction: Instruction::Other,
        previous_instruction: Instruction::Other,
        fpu_simd_dirty_marked: false,
        elide_current_flags: false,
        instruction_counter,
        fastmem_writes: fastmem_writes_compile_enabled(state_flags),
        x87_local_cache: std::array::from_fn(|_| None),
        push32_write_cache: None,
        x87_cache_kept: false,
    };

    let entry_blocks = {
        let mut nodes = &structure;
        let result;
        loop {
            match &nodes[0] {
                WasmStructure::Dispatcher(e) => {
                    result = e.clone();
                    break;
                },
                WasmStructure::Loop { .. } => {
                    dbg_assert!(false);
                },
                WasmStructure::BasicBlock(_) => {
                    dbg_assert!(false);
                },
                // Note: We could use these blocks as entry points, which will yield
                // more entries for free, but it requires adding those to the dispatcher
                // It's to be investigated if this yields a performance improvement
                // See also the comment at the bottom of this function when creating entry
                // points
                WasmStructure::Block(children) => {
                    nodes = children;
                },
            }
        }
        result
    };

    let mut index_for_addr = HashMap::new();
    for (i, &addr) in entry_blocks.iter().enumerate() {
        dbg_assert!(i < 0x10000);
        index_for_addr.insert(addr, i as u16);
    }
    for b in basic_blocks.values() {
        if !index_for_addr.contains_key(&b.addr) {
            let i = index_for_addr.len();
            dbg_assert!(i < 0x10000);
            index_for_addr.insert(b.addr, i as u16);
        }
    }

    let mut label_for_addr: HashMap<u32, (Label, Option<u16>)> = HashMap::new();

    enum Work {
        WasmStructure(WasmStructure),
        BlockEnd {
            label: Label,
            targets: Vec<u32>,
            olds: HashMap<u32, (Label, Option<u16>)>,
        },
        LoopEnd {
            label: Label,
            entries: Vec<u32>,
            olds: HashMap<u32, (Label, Option<u16>)>,
        },
    }
    let mut work: VecDeque<Work> = structure
        .into_iter()
        .map(|x| Work::WasmStructure(x))
        .collect();

    while let Some(block) = work.pop_front() {
        let next_addr: Option<Vec<u32>> = work.iter().find_map(|x| match x {
            Work::WasmStructure(l) => Some(l.head().collect()),
            _ => None,
        });
        let target_block = &ctx.builder.arg_local_initial_state.unsafe_clone();

        match block {
            Work::WasmStructure(WasmStructure::BasicBlock(addr)) => {
                let block = basic_blocks.get(&addr).unwrap();
                jit_generate_basic_block(ctx, block, basic_blocks);

                if block.has_sti {
                    match block.ty {
                        BasicBlockType::ConditionalJump {
                            condition,
                            jump_offset,
                            jump_offset_is_32,
                            ..
                        } => {
                            codegen::gen_set_eip_low_bits(
                                ctx.builder,
                                block.end_addr as i32 & 0xFFF,
                            );
                            codegen::gen_condition_fn(ctx, condition);
                            ctx.builder.if_void();
                            if jump_offset_is_32 {
                                codegen::gen_relative_jump(ctx.builder, jump_offset);
                            }
                            else {
                                codegen::gen_jmp_rel16(ctx.builder, jump_offset as u16);
                            }
                            ctx.builder.block_end();
                        },
                        BasicBlockType::Normal {
                            jump_offset,
                            jump_offset_is_32,
                            ..
                        } => {
                            if jump_offset_is_32 {
                                codegen::gen_set_eip_low_bits_and_jump_rel32(
                                    ctx.builder,
                                    block.end_addr as i32 & 0xFFF,
                                    jump_offset,
                                );
                            }
                            else {
                                codegen::gen_set_eip_low_bits(
                                    ctx.builder,
                                    block.end_addr as i32 & 0xFFF,
                                );
                                codegen::gen_jmp_rel16(ctx.builder, jump_offset as u16);
                            }
                        },
                        BasicBlockType::Exit => {},
                        BasicBlockType::AbsoluteEip => {},
                    };
                    codegen::gen_debug_track_jit_exit(ctx.builder, block.last_instruction_addr);
                    // STI forces a module exit (one instruction must run before handle_irqs).
                    // Not a chainable dispatch — count as dynamic.
                    codegen::gen_dispatch_stat_increment(ctx.builder, stat::MODULE_EXIT_DYNAMIC);
                    codegen::gen_move_registers_from_locals_to_memory(ctx);
                    codegen::gen_fn0_const(ctx.builder, "handle_irqs");
                    codegen::gen_update_instruction_counter(ctx);
                    ctx.builder.return_();
                    continue;
                }

                match &block.ty {
                    BasicBlockType::Exit => {
                        // Exit this function
                        codegen::gen_debug_track_jit_exit(ctx.builder, block.last_instruction_addr);
                        codegen::gen_profiler_stat_increment(ctx.builder, stat::DIRECT_EXIT);
                        // Terminating instruction set eip at runtime (ret/int/iret/far jmp)
                        codegen::gen_dispatch_stat_increment(ctx.builder, stat::MODULE_EXIT_DYNAMIC);
                        ctx.builder.br(ctx.exit_label);
                    },
                    BasicBlockType::AbsoluteEip => {
                        // Indirect-target histogram for watched pages.
                        // Records (terminal instruction addr, runtime eip) via an import
                        // into the generated module; emitted only when the page is watched.
                        if trace_profiler::is_page_watched(Page::page_of(block.addr)) {
                            ctx.builder.const_i32(block.last_instruction_addr as i32);
                            codegen::gen_get_eip(ctx.builder);
                            ctx.builder.call_fn2("trace2_record_indirect");
                        }
                        // RET-target speculation: a leaf's RET
                        // whose module-local call sites are known compares the runtime
                        // eip against each return address and re-enters the module
                        // dispatcher directly, skipping the helper call below. Return
                        // addresses were marked_as_entry at their CALLs, so they are
                        // top-dispatcher entries (index < entry_blocks.len()); anything
                        // else is skipped defensively.
                        for &(cand_virt, cand_phys) in &block.ret_speculation {
                            if let Some(&idx) = index_for_addr.get(&cand_phys) {
                                if (idx as usize) < entry_blocks.len() {
                                    codegen::gen_get_eip(ctx.builder);
                                    ctx.builder.const_i32(cand_virt);
                                    ctx.builder.eq_i32();
                                    ctx.builder.if_void();
                                    ctx.builder.const_i32(idx.into());
                                    ctx.builder.set_local(target_block);
                                    ctx.builder.br(main_loop_label);
                                    ctx.builder.block_end();
                                }
                            }
                        }

                        // Check if we can stay in this module, if not exit
                        codegen::gen_get_eip(ctx.builder);
                        ctx.builder.const_i32(wasm_table_index.to_u16() as i32);
                        ctx.builder.const_i32(state_flags.to_u32() as i32);
                        ctx.builder.call_fn3_ret("jit_find_cache_entry_in_page");
                        ctx.builder.tee_local(target_block);
                        ctx.builder.const_i32(0);
                        ctx.builder.ge_i32();
                        // The branch stays conditional by design: the miss path below is no
                        // longer a plain exit — it attempts a cross-module tail-call (RET
                        // dynamic chaining), which must run between the in-page miss and the
                        // module exit. Folding the miss into the dispatcher br_table (the old
                        // idea here) would lose that attempt for the price of one predictable
                        // branch on the hit path.
                        ctx.builder.br_if(main_loop_label);

                        // RET/indirect dynamic chaining: the in-module
                        // re-dispatch missed, but the runtime eip may hit ANOTHER compiled
                        // module — tail-call straight into it instead of round-tripping
                        // through main_loop. Same flush + packed-slot convention as
                        // gen_chain_or_exit_to_known_successor; on a chain miss we fall
                        // through to the plain module exit (the register flush is idempotent
                        // and the instruction counter was zeroed after flushing, so the
                        // epilogue's second flush/add is harmless).
                        if ret_chaining_enabled() {
                            codegen::gen_move_registers_from_locals_to_memory(ctx);
                            codegen::gen_update_instruction_counter(ctx);
                            ctx.builder.const_i32(0);
                            ctx.builder.set_local(&ctx.instruction_counter);

                            ctx.builder.const_i32(state_flags.to_u32() as i32);
                            ctx.builder
                                .call_fn1_ret("jit_find_cache_entry_for_dynamic_chaining");
                            let packed_target = ctx.builder.tee_new_local();

                            ctx.builder.get_local(&packed_target);
                            ctx.builder.const_i32(0);
                            ctx.builder.ge_i32();
                            ctx.builder.if_void();
                            ctx.builder.get_local(&packed_target);
                            ctx.builder.const_i32(0xFFFF);
                            ctx.builder.and_i32();
                            ctx.builder.get_local(&packed_target);
                            ctx.builder.const_i32(16);
                            ctx.builder.shr_u_i32();
                            ctx.builder.return_call_indirect_fn1();
                            ctx.builder.block_end();
                            ctx.builder.free_local(packed_target);
                        }

                        codegen::gen_debug_track_jit_exit(ctx.builder, block.last_instruction_addr);
                        ctx.builder.br(ctx.exit_label);
                    },
                    &BasicBlockType::Normal {
                        next_block_addr: None,
                        jump_offset,
                        jump_offset_is_32,
                    } => {
                        if jump_offset_is_32 {
                            codegen::gen_set_eip_low_bits_and_jump_rel32(
                                ctx.builder,
                                block.end_addr as i32 & 0xFFF,
                                jump_offset,
                            );
                        }
                        else {
                            codegen::gen_set_eip_low_bits(
                                ctx.builder,
                                block.end_addr as i32 & 0xFFF,
                            );
                            codegen::gen_jmp_rel16(ctx.builder, jump_offset as u16);
                        }

                        codegen::gen_profiler_stat_increment(ctx.builder, stat::DIRECT_EXIT);
                        // Direct unconditional JMP whose target is outside this module —
                        // successor eip is a compile-time constant (jump_offset). Chainable.
                        gen_chain_or_exit_to_known_successor(
                            ctx,
                            state_flags,
                            block.last_instruction_addr,
                        );
                    },
                    &BasicBlockType::Normal {
                        next_block_addr: Some(next_block_addr),
                        jump_offset,
                        jump_offset_is_32,
                    } => {
                        // Unconditional jump to next basic block
                        // - All instructions that don't change eip
                        // - Unconditional jumps

                        if Page::page_of(next_block_addr) != Page::page_of(block.addr) {
                            if jump_offset_is_32 {
                                codegen::gen_set_eip_low_bits_and_jump_rel32(
                                    ctx.builder,
                                    block.end_addr as i32 & 0xFFF,
                                    jump_offset,
                                );
                            }
                            else {
                                codegen::gen_set_eip_low_bits(
                                    ctx.builder,
                                    block.end_addr as i32 & 0xFFF,
                                );
                                codegen::gen_jmp_rel16(ctx.builder, jump_offset as u16);
                            }

                            codegen::gen_profiler_stat_increment(
                                ctx.builder,
                                stat::NORMAL_PAGE_CHANGE,
                            );

                            codegen::gen_page_switch_check(
                                ctx,
                                next_block_addr,
                                block.last_instruction_addr,
                            );

                            #[cfg(debug_assertions)]
                            codegen::gen_fn2_const(
                                ctx.builder,
                                "check_page_switch",
                                block.addr,
                                next_block_addr,
                            );
                        }

                        if next_addr
                            .as_ref()
                            .map_or(false, |n| n.contains(&next_block_addr))
                        {
                            // Blocks are consecutive
                            if next_addr.unwrap().len() > 1 {
                                let target_index = *index_for_addr.get(&next_block_addr).unwrap();
                                if cfg!(feature = "profiler") {
                                    ctx.builder.const_i32(target_index.into());
                                    ctx.builder.call_fn1("debug_set_dispatcher_target");
                                }
                                ctx.builder.const_i32(target_index.into());
                                ctx.builder.set_local(target_block);
                                codegen::gen_profiler_stat_increment(
                                    ctx.builder,
                                    stat::NORMAL_FALLTHRU_WITH_TARGET_BLOCK,
                                );
                            }
                            else {
                                codegen::gen_profiler_stat_increment(
                                    ctx.builder,
                                    stat::NORMAL_FALLTHRU,
                                );
                            }
                        }
                        else {
                            let &(br, target_index) = label_for_addr.get(&next_block_addr).unwrap();
                            if let Some(target_index) = target_index {
                                if cfg!(feature = "profiler") {
                                    ctx.builder.const_i32(target_index.into());
                                    ctx.builder.call_fn1("debug_set_dispatcher_target");
                                }
                                ctx.builder.const_i32(target_index.into());
                                ctx.builder.set_local(target_block);
                                codegen::gen_profiler_stat_increment(
                                    ctx.builder,
                                    stat::NORMAL_BRANCH_WITH_TARGET_BLOCK,
                                );
                            }
                            else {
                                codegen::gen_profiler_stat_increment(
                                    ctx.builder,
                                    stat::NORMAL_BRANCH,
                                );
                            }
                            ctx.builder.br(br);
                        }
                    },
                    &BasicBlockType::ConditionalJump {
                        next_block_addr,
                        next_block_branch_taken_addr,
                        condition,
                        jump_offset,
                        jump_offset_is_32,
                    } => {
                        // Conditional jump to next basic block
                        // - jnz, jc, loop, jcxz, etc.

                        // Generate:
                        // (1) condition()
                        // (2) br_if()
                        // (3) br()
                        // Except:
                        // If we need to update eip in case (2), it's replaced by if { update_eip(); br() }
                        // If case (3) can fall through to the next basic block, the branch is eliminated
                        // Dispatcher target writes can be generated in either case
                        // Condition may be inverted if it helps generate a fallthrough instead of the second branch

                        codegen::gen_profiler_stat_increment(ctx.builder, stat::CONDITIONAL_JUMP);

                        #[derive(PartialEq)]
                        enum Case {
                            BranchTaken,
                            BranchNotTaken,
                        }

                        let mut handle_case = |case: Case, is_first| {
                            // first case generates condition and *has* to branch away,
                            // second case branches unconditionally or falls through

                            if is_first {
                                if case == Case::BranchNotTaken {
                                    codegen::gen_condition_fn_negated(ctx, condition);
                                }
                                else {
                                    codegen::gen_condition_fn(ctx, condition);
                                }
                            }

                            let next_block_addr = if case == Case::BranchTaken {
                                next_block_branch_taken_addr
                            }
                            else {
                                next_block_addr
                            };

                            if let Some(next_block_addr) = next_block_addr {
                                if Page::page_of(next_block_addr) != Page::page_of(block.addr) {
                                    dbg_assert!(case == Case::BranchTaken); // currently not possible in other case
                                    if is_first {
                                        ctx.builder.if_i32();
                                    }
                                    if jump_offset_is_32 {
                                        codegen::gen_set_eip_low_bits_and_jump_rel32(
                                            ctx.builder,
                                            block.end_addr as i32 & 0xFFF,
                                            jump_offset,
                                        );
                                    }
                                    else {
                                        codegen::gen_set_eip_low_bits(
                                            ctx.builder,
                                            block.end_addr as i32 & 0xFFF,
                                        );
                                        codegen::gen_jmp_rel16(ctx.builder, jump_offset as u16);
                                    }

                                    codegen::gen_profiler_stat_increment(
                                        ctx.builder,
                                        stat::CONDITIONAL_JUMP_PAGE_CHANGE,
                                    );
                                    codegen::gen_page_switch_check(
                                        ctx,
                                        next_block_addr,
                                        block.last_instruction_addr,
                                    );

                                    #[cfg(debug_assertions)]
                                    codegen::gen_fn2_const(
                                        ctx.builder,
                                        "check_page_switch",
                                        block.addr,
                                        next_block_addr,
                                    );

                                    if is_first {
                                        ctx.builder.const_i32(1);
                                        ctx.builder.else_();
                                        ctx.builder.const_i32(0);
                                        ctx.builder.block_end();
                                    }
                                }

                                if next_addr
                                    .as_ref()
                                    .map_or(false, |n| n.contains(&next_block_addr))
                                {
                                    // blocks are consecutive

                                    // fallthrough, has to be second
                                    dbg_assert!(!is_first);

                                    if next_addr.as_ref().unwrap().len() > 1 {
                                        let target_index =
                                            *index_for_addr.get(&next_block_addr).unwrap();
                                        if cfg!(feature = "profiler") {
                                            ctx.builder.const_i32(target_index.into());
                                            ctx.builder.call_fn1("debug_set_dispatcher_target");
                                        }
                                        ctx.builder.const_i32(target_index.into());
                                        ctx.builder.set_local(target_block);
                                        codegen::gen_profiler_stat_increment(
                                            ctx.builder,
                                            stat::CONDITIONAL_JUMP_FALLTHRU_WITH_TARGET_BLOCK,
                                        );
                                    }
                                    else {
                                        codegen::gen_profiler_stat_increment(
                                            ctx.builder,
                                            stat::CONDITIONAL_JUMP_FALLTHRU,
                                        );
                                    }
                                }
                                else {
                                    let &(br, target_index) =
                                        label_for_addr.get(&next_block_addr).unwrap();
                                    if let Some(target_index) = target_index {
                                        if cfg!(feature = "profiler") {
                                            // Note: Currently called unconditionally, even if the
                                            // br_if below doesn't branch
                                            ctx.builder.const_i32(target_index.into());
                                            ctx.builder.call_fn1("debug_set_dispatcher_target");
                                        }
                                        ctx.builder.const_i32(target_index.into());
                                        ctx.builder.set_local(target_block);
                                    }

                                    if is_first {
                                        if cfg!(feature = "profiler") {
                                            ctx.builder.if_void();
                                            codegen::gen_profiler_stat_increment(
                                                ctx.builder,
                                                if target_index.is_some() {
                                                    stat::CONDITIONAL_JUMP_BRANCH_WITH_TARGET_BLOCK
                                                }
                                                else {
                                                    stat::CONDITIONAL_JUMP_BRANCH
                                                },
                                            );
                                            ctx.builder.br(br);
                                            ctx.builder.block_end();
                                        }
                                        else {
                                            ctx.builder.br_if(br);
                                        }
                                    }
                                    else {
                                        codegen::gen_profiler_stat_increment(
                                            ctx.builder,
                                            if target_index.is_some() {
                                                stat::CONDITIONAL_JUMP_BRANCH_WITH_TARGET_BLOCK
                                            }
                                            else {
                                                stat::CONDITIONAL_JUMP_BRANCH
                                            },
                                        );
                                        ctx.builder.br(br);
                                    }
                                }
                            }
                            else {
                                // target is outside of this module, update eip and exit
                                if is_first {
                                    ctx.builder.if_void();
                                }

                                if case == Case::BranchTaken {
                                    if jump_offset_is_32 {
                                        codegen::gen_set_eip_low_bits_and_jump_rel32(
                                            ctx.builder,
                                            block.end_addr as i32 & 0xFFF,
                                            jump_offset,
                                        );
                                    }
                                    else {
                                        codegen::gen_set_eip_low_bits(
                                            ctx.builder,
                                            block.end_addr as i32 & 0xFFF,
                                        );
                                        codegen::gen_jmp_rel16(ctx.builder, jump_offset as u16);
                                    }
                                }
                                else {
                                    codegen::gen_set_eip_low_bits(
                                        ctx.builder,
                                        block.end_addr as i32 & 0xFFF,
                                    );
                                }

                                codegen::gen_profiler_stat_increment(
                                    ctx.builder,
                                    stat::CONDITIONAL_JUMP_EXIT,
                                );
                                // Conditional JMP leaving the module — successor eip is a
                                // compile-time constant (taken=jump_offset, not-taken=end_addr). Chainable.
                                gen_chain_or_exit_to_known_successor(
                                    ctx,
                                    state_flags,
                                    block.last_instruction_addr,
                                );

                                if is_first {
                                    ctx.builder.block_end();
                                }
                            }
                        };

                        let branch_taken_is_fallthrough = next_block_branch_taken_addr
                            .map_or(false, |addr| {
                                next_addr.as_ref().map_or(false, |n| n.contains(&addr))
                            });
                        let branch_not_taken_is_fallthrough = next_block_addr
                            .map_or(false, |addr| {
                                next_addr.as_ref().map_or(false, |n| n.contains(&addr))
                            });

                        if branch_not_taken_is_fallthrough && branch_taken_is_fallthrough {
                            let next_block_addr = next_block_addr.unwrap();
                            let next_block_branch_taken_addr =
                                next_block_branch_taken_addr.unwrap();

                            dbg_log!(
                                "Conditional control flow: fallthrough in both cases, page_switch={} next_is_multi={}",
                                Page::page_of(next_block_branch_taken_addr)
                                    != Page::page_of(block.addr),
                                next_addr.as_ref().unwrap().len() > 1,
                            );

                            dbg_assert!(
                                Page::page_of(next_block_addr) == Page::page_of(block.addr)
                            ); // currently not possible

                            if Page::page_of(next_block_branch_taken_addr)
                                != Page::page_of(block.addr)
                            {
                                codegen::gen_condition_fn(ctx, condition);
                                ctx.builder.if_void();

                                if jump_offset_is_32 {
                                    codegen::gen_set_eip_low_bits_and_jump_rel32(
                                        ctx.builder,
                                        block.end_addr as i32 & 0xFFF,
                                        jump_offset,
                                    );
                                }
                                else {
                                    codegen::gen_set_eip_low_bits(
                                        ctx.builder,
                                        block.end_addr as i32 & 0xFFF,
                                    );
                                    codegen::gen_jmp_rel16(ctx.builder, jump_offset as u16);
                                }

                                codegen::gen_profiler_stat_increment(
                                    ctx.builder,
                                    stat::CONDITIONAL_JUMP_PAGE_CHANGE,
                                );
                                codegen::gen_page_switch_check(
                                    ctx,
                                    next_block_branch_taken_addr,
                                    block.last_instruction_addr,
                                );

                                #[cfg(debug_assertions)]
                                codegen::gen_fn2_const(
                                    ctx.builder,
                                    "check_page_switch",
                                    block.addr,
                                    next_block_branch_taken_addr,
                                );

                                dbg_assert!(next_addr.unwrap().len() > 1);

                                let target_index_taken =
                                    *index_for_addr.get(&next_block_branch_taken_addr).unwrap();
                                let target_index_not_taken =
                                    *index_for_addr.get(&next_block_addr).unwrap();

                                ctx.builder.const_i32(target_index_taken.into());
                                ctx.builder.set_local(target_block);

                                ctx.builder.else_();
                                ctx.builder.const_i32(target_index_not_taken.into());
                                ctx.builder.set_local(target_block);

                                ctx.builder.block_end();
                            }
                            else if next_addr.unwrap().len() > 1 {
                                let target_index_taken =
                                    *index_for_addr.get(&next_block_branch_taken_addr).unwrap();
                                let target_index_not_taken =
                                    *index_for_addr.get(&next_block_addr).unwrap();

                                codegen::gen_condition_fn(ctx, condition);
                                ctx.builder.if_i32();
                                ctx.builder.const_i32(target_index_taken.into());
                                ctx.builder.else_();
                                ctx.builder.const_i32(target_index_not_taken.into());
                                ctx.builder.block_end();
                                ctx.builder.set_local(target_block);
                            }
                        }
                        else if branch_taken_is_fallthrough {
                            handle_case(Case::BranchNotTaken, true);
                            handle_case(Case::BranchTaken, false);
                        }
                        else {
                            handle_case(Case::BranchTaken, true);
                            handle_case(Case::BranchNotTaken, false);
                        }
                    },
                }
            },
            Work::WasmStructure(WasmStructure::Dispatcher(entries)) => {
                profiler::stat_increment(stat::COMPILE_DISPATCHER);

                if cfg!(feature = "profiler") {
                    ctx.builder.get_local(target_block);
                    ctx.builder.const_i32(index_for_addr.len() as i32);
                    ctx.builder.call_fn2("check_dispatcher_target");
                }

                if entries.len() > BRTABLE_CUTOFF {
                    // generate a brtable
                    codegen::gen_profiler_stat_increment(ctx.builder, stat::DISPATCHER_LARGE);
                    let mut cases = Vec::new();
                    for &addr in &entries {
                        let &(label, target_index) = label_for_addr.get(&addr).unwrap();
                        let &index = index_for_addr.get(&addr).unwrap();
                        dbg_assert!(target_index.is_none() || target_index == Some(index));
                        while index as usize >= cases.len() {
                            cases.push(brtable_default);
                        }
                        cases[index as usize] = label;
                    }
                    ctx.builder.get_local(target_block);
                    ctx.builder.brtable(brtable_default, &mut cases.iter());
                }
                else {
                    // generate a if target == block.addr then br block.label ...
                    codegen::gen_profiler_stat_increment(ctx.builder, stat::DISPATCHER_SMALL);
                    let nexts: HashSet<u32> = next_addr
                        .as_ref()
                        .map_or(HashSet::new(), |nexts| nexts.iter().copied().collect());
                    for &addr in &entries {
                        if nexts.contains(&addr) {
                            continue;
                        }
                        let index = *index_for_addr.get(&addr).unwrap();
                        let &(label, _) = label_for_addr.get(&addr).unwrap();
                        ctx.builder.get_local(target_block);
                        ctx.builder.const_i32(index.into());
                        ctx.builder.eq_i32();
                        ctx.builder.br_if(label);
                    }
                }
            },
            Work::WasmStructure(WasmStructure::Loop(children)) => {
                profiler::stat_increment(stat::COMPILE_WASM_LOOP);

                let entries: Vec<u32> = children[0].head().collect();
                let label = ctx.builder.loop_void();
                codegen::gen_profiler_stat_increment(ctx.builder, stat::LOOP);

                if entries.len() == 1 {
                    let addr = entries[0];
                    codegen::gen_set_eip_low_bits(ctx.builder, addr as i32 & 0xFFF);
                    profiler::stat_increment(stat::COMPILE_WITH_LOOP_SAFETY);
                    codegen::gen_profiler_stat_increment(ctx.builder, stat::LOOP_SAFETY);
                    if unsafe { JIT_USE_LOOP_SAFETY } {
                        ctx.builder.get_local(&ctx.instruction_counter);
                        ctx.builder.const_i32(cpu::LOOP_COUNTER);
                        ctx.builder.geu_i32();
                        if cfg!(feature = "profiler") {
                            ctx.builder.if_void();
                            codegen::gen_debug_track_jit_exit(ctx.builder, addr);
                            ctx.builder.br(exit_label);
                            ctx.builder.block_end();
                        }
                        else {
                            ctx.builder.br_if(exit_label);
                        }
                    }
                }

                let mut olds = HashMap::new();
                for &target in entries.iter() {
                    let index = if entries.len() == 1 {
                        None
                    }
                    else {
                        Some(*index_for_addr.get(&target).unwrap())
                    };
                    let old = label_for_addr.insert(target, (label, index));
                    if let Some(old) = old {
                        olds.insert(target, old);
                    }
                }

                work.push_front(Work::LoopEnd {
                    label,
                    entries,
                    olds,
                });
                for c in children.into_iter().rev() {
                    work.push_front(Work::WasmStructure(c));
                }
            },
            Work::LoopEnd {
                label,
                entries,
                olds,
            } => {
                for target in entries {
                    let old = label_for_addr.remove(&target);
                    dbg_assert!(old.map(|(l, _)| l) == Some(label));
                }
                for (target, old) in olds {
                    let old = label_for_addr.insert(target, old);
                    dbg_assert!(old.is_none());
                }

                ctx.builder.block_end();
            },
            Work::WasmStructure(WasmStructure::Block(children)) => {
                profiler::stat_increment(stat::COMPILE_WASM_BLOCK);

                let targets = next_addr.clone().unwrap();
                let label = ctx.builder.block_void();
                let mut olds = HashMap::new();
                for &target in targets.iter() {
                    let index = if targets.len() == 1 {
                        None
                    }
                    else {
                        Some(*index_for_addr.get(&target).unwrap())
                    };
                    let old = label_for_addr.insert(target, (label, index));
                    if let Some(old) = old {
                        olds.insert(target, old);
                    }
                }

                work.push_front(Work::BlockEnd {
                    label,
                    targets,
                    olds,
                });
                for c in children.into_iter().rev() {
                    work.push_front(Work::WasmStructure(c));
                }
            },
            Work::BlockEnd {
                label,
                targets,
                olds,
            } => {
                for target in targets {
                    let old = label_for_addr.remove(&target);
                    dbg_assert!(old.map(|(l, _)| l) == Some(label));
                }
                for (target, old) in olds {
                    let old = label_for_addr.insert(target, old);
                    dbg_assert!(old.is_none());
                }

                ctx.builder.block_end();
            },
        }
    }

    dbg_assert!(label_for_addr.is_empty());

    {
        ctx.builder.block_end(); // default case for the brtable
        // A dispatch index with no matching case is a STALE dispatch — a recycled
        // wasm_table_index whose old tlb_code state_table wasn't swept yet, or a
        // racing invalidation mid-slice. Every path that can land here (module
        // entry, in-page re-dispatch, ret-speculation) has already materialized
        // the runtime instruction_pointer, so the recoverable move is a clean
        // module exit: the interpreter re-resolves the eip through the normal
        // cache path (worst case: recompile). `unreachable` turned this race
        // into a fatal wasm trap. Debug builds still flag it via check_dispatcher_target.
        ctx.builder.br(ctx.exit_label);
    }
    {
        ctx.builder.block_end(); // main loop
    }
    {
        // exit-with-fault case
        ctx.builder.block_end();
        codegen::gen_move_registers_from_locals_to_memory(ctx);
        codegen::gen_fn0_const(ctx.builder, "trigger_fault_end_jit");
        codegen::gen_update_instruction_counter(ctx);
        ctx.builder.return_();
    }
    {
        // exit
        ctx.builder.block_end();
        codegen::gen_move_registers_from_locals_to_memory(ctx);
        codegen::gen_update_instruction_counter(ctx);
    }

    for local in ctx.register_locals.drain(..) {
        ctx.builder.free_local(local);
    }
    ctx.builder
        .free_local(ctx.instruction_counter.unsafe_clone());
    ctx.builder.free_flag_locals();

    ctx.builder.finish();

    let entries = Vec::from_iter(entry_blocks.iter().map(|addr| {
        let block = basic_blocks.get(&addr).unwrap();
        let index = *index_for_addr.get(&addr).unwrap();

        profiler::stat_increment(stat::COMPILE_ENTRY_POINT);

        dbg_assert!(block.addr < block.end_addr);
        // Note: We also insert blocks that weren't originally marked as entries here
        //       This doesn't have any downside, besides making the hash table slightly larger

        (block.addr, index)
    }));

    for b in basic_blocks.values() {
        if b.is_entry_block {
            dbg_assert!(entries.iter().find(|(addr, _)| *addr == b.addr).is_some());
        }
    }

    return entries;
}

/// True if the instruction at `eip` is an x87 escape, after prefixes.
fn opcode_is_x87(eip: u32) -> bool {
    let mut addr = eip;
    for _ in 0..4 {
        match read_jit_u8(addr) {
            0x26 | 0x2E | 0x36 | 0x3E | 0x64 | 0x65 | 0x66 | 0x67 | 0xF0 | 0xF2 | 0xF3 => {
                addr = addr.wrapping_add(1);
            },
            op => return (0xD8..=0xDF).contains(&op),
        }
    }
    false
}

/// True if the instruction at `eip` MAY be an MMX op (0F-escape into the MMX opcode
/// ranges, incl. EMMS 0F 77), after prefixes. MMX registers alias fpu_st storage
/// (get_reg_mmx_offset), so these mutate st memory behind the x87 local cache exactly
/// like raw x87 helpers do — the emission loop must invalidate live slots after them.
/// Deliberately conservative: prefixed SSE forms of the same opcodes (66/F2/F3) match
/// too, costing only a spurious runtime invalidate when x87 slots happen to be live.
fn opcode_is_mmx(eip: u32) -> bool {
    let mut addr = eip;
    for _ in 0..4 {
        match read_jit_u8(addr) {
            0x26 | 0x2E | 0x36 | 0x3E | 0x64 | 0x65 | 0x66 | 0x67 | 0xF0 | 0xF2 | 0xF3 => {
                addr = addr.wrapping_add(1);
            },
            0x0F => {
                let op = read_jit_u8(addr.wrapping_add(1));
                return (0x60..=0x77).contains(&op)
                    || op == 0x7E
                    || op == 0x7F
                    || (0xD1..=0xFE).contains(&op);
            },
            _ => return false,
        }
    }
    false
}

fn jit_generate_basic_block(
    ctx: &mut JitContext,
    block: &BasicBlock,
    basic_blocks: &HashMap<u32, BasicBlock>,
) {
    let needs_eip_updated = match block.ty {
        BasicBlockType::Exit => true,
        _ => false,
    };

    profiler::stat_increment(stat::COMPILE_BASIC_BLOCK);

    let start_addr = block.addr;
    let last_instruction_addr = block.last_instruction_addr;
    let stop_addr = block.end_addr;

    // First iteration of do-while assumes the caller confirms this condition
    dbg_assert!(!is_near_end_of_page(start_addr));

    if cfg!(feature = "profiler") {
        ctx.builder.const_i32(start_addr as i32);
        ctx.builder.call_fn1("enter_basic_block");
    }

    ctx.builder.get_local(&ctx.instruction_counter);
    ctx.builder.const_i32(block.number_of_instructions as i32);
    ctx.builder.add_i32();
    ctx.builder.set_local(&ctx.instruction_counter);

    // Block-chaining: count every basic-block execution. INTRA_MODULE_EDGE is derived as
    // BLOCK_EXECUTION - MODULE_REENTRY in the readout (each module run executes one entry block via
    // dispatch; every further block it runs was reached by an in-module edge).
    codegen::gen_dispatch_stat_increment(ctx.builder, stat::BLOCK_EXECUTION);

    // Tier-2 trace-compiler: per-block exec
    // counter + CFG registration for watched pages. Emits one fixed-address u64 increment;
    // nothing is emitted for unwatched pages (zero cost when profiling is off).
    if trace_profiler::is_enabled() && trace_profiler::is_page_watched(Page::page_of(block.addr)) {
        let (kind, condition, succ_fallthrough, succ_taken) = match block.ty {
            BasicBlockType::Normal { next_block_addr, .. } => {
                (trace_profiler::KIND_NORMAL, 0u8, next_block_addr.unwrap_or(0), 0)
            },
            BasicBlockType::ConditionalJump {
                next_block_addr,
                next_block_branch_taken_addr,
                condition,
                ..
            } => (
                trace_profiler::KIND_CONDITIONAL,
                condition,
                next_block_addr.unwrap_or(0),
                next_block_branch_taken_addr.unwrap_or(0),
            ),
            BasicBlockType::AbsoluteEip => (trace_profiler::KIND_ABSOLUTE_EIP, 0, 0, 0),
            BasicBlockType::Exit => (trace_profiler::KIND_EXIT, 0, 0, 0),
        };
        if let Some(counter_addr) = trace_profiler::register_block(trace_profiler::BlockRecord {
            addr: block.addr,
            last_instruction_addr: block.last_instruction_addr,
            end_addr: block.end_addr,
            kind,
            condition,
            succ_fallthrough,
            succ_taken,
            number_of_instructions: block.number_of_instructions,
            is_entry_block: block.is_entry_block,
            slot: u32::MAX,
        }) {
            ctx.builder.increment_fixed_i64(counter_addr, 1);
        }
    }

    ctx.cpu.eip = start_addr;
    ctx.current_instruction = Instruction::Other;
    ctx.previous_instruction = Instruction::Other;
    ctx.fpu_simd_dirty_marked = false;
    ctx.elide_current_flags = false;

    loop {
        let mut instruction = 0;
        if cfg!(feature = "profiler") {
            instruction = memory::read32s(ctx.cpu.eip) as u32;
            opstats::gen_opstats(ctx.builder, instruction);
            opstats::record_opstat_compiled(instruction);
        }

        if ctx.cpu.eip == last_instruction_addr {
            // Before the last instruction:
            // - Set eip to *after* the instruction
            // - Set previous_eip to *before* the instruction
            if needs_eip_updated {
                codegen::gen_set_previous_eip_offset_from_eip_with_low_bits(
                    ctx.builder,
                    last_instruction_addr as i32 & 0xFFF,
                );
                codegen::gen_set_eip_low_bits(ctx.builder, stop_addr as i32 & 0xFFF);
            }
        }

        let wasm_length_before = ctx.builder.instruction_body_length();

        ctx.start_of_current_instruction = ctx.cpu.eip;
        let start_eip = ctx.cpu.eip;
        ctx.elide_current_flags =
            should_elide_current_flags(&*ctx.cpu, start_eip, block, basic_blocks);
        // Relaxed x87 wrappers set this when they keep the st cache coherent.
        ctx.x87_cache_kept = false;
        let mut instruction_flags = 0;
        jit_instructions::jit_instruction(ctx, &mut instruction_flags);
        let end_eip = ctx.cpu.eip;

        // Raw x87 helpers mutate TOP/st memory behind the local cache; MMX ops
        // (incl. EMMS) alias the same fpu_st storage and must invalidate too.
        if !ctx.x87_cache_kept
            && ctx.x87_local_cache.iter().any(|s| s.is_some())
            && (opcode_is_x87(start_eip) || opcode_is_mmx(start_eip))
        {
            codegen::gen_x87_local_cache_invalidate_all_runtime(ctx);
        }

        let instruction_length = end_eip - start_eip;
        let was_block_boundary = instruction_flags & JIT_INSTR_BLOCK_BOUNDARY_FLAG != 0;

        let wasm_length = ctx.builder.instruction_body_length() - wasm_length_before;
        opstats::record_opstat_size_wasm(instruction, wasm_length as u64);

        dbg_assert!((end_eip == stop_addr) == (start_eip == last_instruction_addr));
        dbg_assert!(instruction_length < MAX_INSTRUCTION_LENGTH);

        let end_addr = ctx.cpu.eip;

        if end_addr == stop_addr {
            // no page was crossed
            dbg_assert!(Page::page_of(end_addr) == Page::page_of(start_addr));
            codegen::gen_x87_local_cache_free_all(ctx);
            codegen::gen_push32_write_cache_free(ctx);
            break;
        }

        if was_block_boundary || is_near_end_of_page(end_addr) || end_addr > stop_addr {
            dbg_log!(
                "Overlapping basic blocks start={:x} expected_end={:x} end={:x} was_block_boundary={} near_end_of_page={}",
                start_addr,
                stop_addr,
                end_addr,
                was_block_boundary,
                is_near_end_of_page(end_addr)
            );
            dbg_assert!(false);
            codegen::gen_x87_local_cache_free_all(ctx);
            codegen::gen_push32_write_cache_free(ctx);
            break;
        }

        ctx.previous_instruction = mem::replace(&mut ctx.current_instruction, Instruction::Other);
    }
}

pub fn jit_increase_hotness_and_maybe_compile(
    virt_address: i32,
    phys_address: u32,
    cs_offset: u32,
    state_flags: CachedStateFlags,
    heat: u32,
) {
    if unsafe { JIT_DISABLED } {
        return;
    }

    let mut ctx = get_jit_state();
    let is_compiling = ctx.compiling.is_some();
    let page = Page::page_of(phys_address);
    let (hotness, entry_points) = ctx.entry_points.entry(page).or_insert_with(|| {
        cpu::tlb_set_has_code(page, true);
        profiler::stat_increment(stat::RUN_INTERPRETED_NEW_PAGE);
        (0, HashSet::new())
    });

    if !is_near_end_of_page(phys_address) {
        entry_points.insert(phys_address as u16 & 0xFFF);
    }

    *hotness += heat;
    if *hotness >= JIT_THRESHOLD {
        if is_compiling {
            return;
        }
        // only try generating if we're in the correct address space
        if cpu::translate_address_read_no_side_effects(virt_address) == Ok(phys_address) {
            *hotness = 0;
            jit_analyze_and_generate(&mut ctx, virt_address, phys_address, cs_offset, state_flags)
        }
        else {
            profiler::stat_increment(stat::COMPILE_WRONG_ADDRESS_SPACE);
        }
    }
}

fn free_wasm_table_index(ctx: &mut JitState, wasm_table_index: WasmTableIndex) {
    if CHECK_JIT_STATE_INVARIANTS {
        dbg_assert!(!ctx.wasm_table_index_free_list.contains(&wasm_table_index));

        match &ctx.compiling {
            Some((wasm_table_index_compiling, _)) => {
                dbg_assert!(
                    *wasm_table_index_compiling != wasm_table_index,
                    "Attempt to free wasm table index that is currently being compiled"
                );
            },
            _ => {},
        }

        dbg_assert!(
            ctx.aot_staged
                .as_ref()
                .map_or(true, |tx| tx.wasm_table_index != wasm_table_index),
            "Attempt to free wasm table index that is AOT-staged"
        );

        dbg_assert!(!ctx
            .pages
            .values()
            .any(|info| info.wasm_table_index == wasm_table_index));

        dbg_assert!(!ctx
            .pages
            .values()
            .any(|info| info.hidden_wasm_table_indices.contains(&wasm_table_index)));

        for i in 0..unsafe { cpu::valid_tlb_entries_count } {
            let page = unsafe { cpu::valid_tlb_entries[i as usize] };
            let meta = dispatch_meta_get(page as u32);
            dbg_assert!(
                meta == 0 || dispatch_meta_table_index(meta) != wasm_table_index.to_u16()
            );
        }
    }

    // Release-safe double-free guard (see
    // free_wasm_module_forest): a second push would hand the SAME table slot to
    // two future modules — silent cross-module dispatch corruption. The debug
    // assert above screams first; release skips loudly and keeps the state sane.
    if ctx.wasm_table_index_free_list.contains(&wasm_table_index) {
        unsafe { WASM_TABLE_INDEX_DOUBLE_FREE_SKIPPED += 1 };
        dbg_log!(
            "BUG: double-free of wasm table index {} skipped",
            wasm_table_index.to_u16()
        );
        return;
    }

    #[cfg(debug_assertions)]
    ctx.wasm_table_index_to_page.remove(&wasm_table_index);

    // Diagnostic audit (idx 24; OFF by default): any DISPATCH_META entry still naming
    // this slot right now is a stale dispatch source the TLB sweeps missed — the next
    // module recycling the slot inherits those virt pages' dispatches.
    if wrong_entry_verify_enabled()
    {
    unsafe {
        let idx = wasm_table_index.to_u16();
        for page in 0..(1usize << 20) {
            let meta = DISPATCH_META[page];
            if meta != 0 && dispatch_meta_table_index(meta) == idx {
                STALE_META_AT_FREE += 1;
                if cpu::tlb_data[page] != 0 {
                    STALE_META_TLB_LIVE += 1;
                }
                else {
                    STALE_META_TLB_DEAD += 1;
                }
                wrong_entry_ring_push([
                    (page as u32) << 12,
                    idx as u32,
                    cpu::tlb_data[page] as u32,
                    3, // tag: stale-meta-at-free record
                ]);
            }
        }
    }
    }

    ctx.wasm_table_index_free_list.push(wasm_table_index);

    // This is the ONLY place a table slot is nulled — invalidate the B1b ret-target
    // memo HERE, not in free_wasm_module: codegen_finalize_finished's module-overwrite
    // path frees replaced indices without going through free_wasm_module (that gap was
    // the null-function crash of the first landing — see the RET_CACHE comment). Also
    // reset the tier-2 execution counter for the recycled index (B3).
    ret_cache_invalidate_all();
    unsafe { MODULE_EXEC_COUNTS[wasm_table_index.to_u16() as usize] = 0 };
    unsafe { MODULE_ENTRY_TOTALS[wasm_table_index.to_u16() as usize] = 0 };
    // A queued promotion names a SLOT, and the slot is about to be recycled. Left in the
    // queue it would promote — and free — whichever freshly compiled module lands here next.
    tier2_pending_drop(wasm_table_index.to_u16());

    // It is not strictly necessary to clear the function, but it will fail more predictably if we
    // accidentally use the function and may garbage collect unused modules earlier
    jit_clear_func(wasm_table_index);
}

fn free_wasm_module(ctx: &mut JitState, wasm_table_index: WasmTableIndex) -> Vec<WasmTableIndex> {
    // B1b memo invalidation lives in free_wasm_table_index (reached below), the one
    // true funnel — this function is NOT on every free path (module-overwrite frees
    // bypass it).
    for i in 0..unsafe { cpu::valid_tlb_entries_count } {
        let page = unsafe { cpu::valid_tlb_entries[i as usize] };
        let entry = unsafe { cpu::tlb_data[page as usize] };
        if 0 != entry {
            let tlb_physical_page = Page::of_u32(
                (entry as u32 >> 12 ^ page as u32) - (unsafe { memory::mem8 } as u32 >> 12),
            );
            let meta = dispatch_meta_get(page as u32);
            if meta != 0 && dispatch_meta_table_index(meta) == wasm_table_index.to_u16() {
                dispatch_meta_clear(page as u32);
                if !ctx.entry_points.contains_key(&tlb_physical_page) {
                    // XXX
                    unsafe { cpu::tlb_data[page as usize] &= !cpu::TLB_HAS_CODE };
                }
            }
        }
    }

    let mut hidden_to_free = Vec::new();
    ctx.pages.retain(
        |_,
         PageInfo {
             wasm_table_index: w,
             hidden_wasm_table_indices,
             ..
         }| {
            if *w == wasm_table_index {
                hidden_to_free.extend(hidden_wasm_table_indices.iter().copied());
                false
            }
            else {
                true
            }
        },
    );

    for info in ctx.pages.values_mut() {
        info.hidden_wasm_table_indices
            .retain(|&w| w != wasm_table_index)
    }

    free_wasm_table_index(ctx, wasm_table_index);
    hidden_to_free
}

fn free_wasm_module_tree(ctx: &mut JitState, root: WasmTableIndex) {
    free_wasm_module_forest(ctx, vec![root]);
}

/// Free a SET of module roots under ONE `seen` guard. The roots of a dirtied
/// page (primary + its captured hidden list) can reach each other: a sibling
/// page sharing the primary carries the same hidden index in its own list, so
/// the primary's tree walk already frees it. Walking each root with a fresh
/// `seen` set would double-free such shared indices —
/// the free list would then hold the index TWICE, two later modules would be handed the
/// SAME wasm table slot, and dispatch_meta of the first would point into the
/// second: cross-module dispatch corruption = the silent-ExitProcess class
/// (garbage-register #PF / wild EIP into data / stale brtable traps).
fn free_wasm_module_forest(ctx: &mut JitState, roots: Vec<WasmTableIndex>) {
    // Hidden entries cannot be promoted; free them with the removed primary.
    let mut seen = HashSet::new();
    let mut stack = roots;
    while let Some(index) = stack.pop() {
        if !seen.insert(index) {
            continue;
        }
        stack.extend(free_wasm_module(ctx, index));
    }
}

/// Register a write in this page: Delete all present code
fn jit_dirty_page_ctx(ctx: &mut JitState, page: Page) {
    let mut did_have_code = false;

    if let Some(PageInfo {
        wasm_table_index,
        hidden_wasm_table_indices,
        state_flags: _,
        entry_points: _,
    }) = ctx.pages.remove(&page)
    {
        profiler::stat_increment(stat::INVALIDATE_PAGE_HAD_CODE);
        did_have_code = true;

        // ONE forest walk for primary + hidden: the captured hidden list and the
        // primary's tree overlap (see free_wasm_module_forest) — separate walks
        // double-free the shared indices.
        let mut roots = hidden_wasm_table_indices;
        roots.push(wasm_table_index);
        free_wasm_module_forest(ctx, roots);
    }

    match ctx.entry_points.remove(&page) {
        None => {},
        Some(_) => {
            profiler::stat_increment(stat::INVALIDATE_PAGE_HAD_ENTRY_POINTS);
            did_have_code = true;

            match &ctx.compiling {
                Some((index, CompilingPageState::Compiling { pages })) => {
                    if pages.contains_key(&page) {
                        ctx.compiling = Some((*index, CompilingPageState::CompilingWritten));
                    }
                },
                _ => {},
            }
        },
    }

    match &ctx.compiling {
        Some((_, CompilingPageState::Compiling { pages })) => {
            dbg_assert!(!pages.contains_key(&page));
        },
        _ => {},
    }

    check_jit_state_invariants(ctx);

    dbg_assert!(!jit_page_has_code_ctx(ctx, page));

    if did_have_code {
        cpu::tlb_set_has_code(page, false);
    }

    if !did_have_code {
        profiler::stat_increment(stat::DIRTY_PAGE_DID_NOT_HAVE_CODE);
    }
}

#[no_mangle]
pub fn jit_dirty_cache(start_addr: u32, end_addr: u32) {
    dbg_assert!(start_addr < end_addr);

    let start_page = Page::page_of(start_addr);
    let end_page = Page::page_of(end_addr - 1);

    for page in start_page.to_u32()..end_page.to_u32() + 1 {
        jit_dirty_page_ctx(&mut get_jit_state(), Page::page_of(page << 12));
    }
}

#[no_mangle]
pub fn jit_dirty_page(page: Page) { jit_dirty_page_ctx(&mut get_jit_state(), page) }

/// dirty pages in the range of start_addr and end_addr, which must span at most two pages
pub fn jit_dirty_cache_small(start_addr: u32, end_addr: u32) {
    dbg_assert!(start_addr < end_addr);

    let start_page = Page::page_of(start_addr);
    let end_page = Page::page_of(end_addr - 1);

    let mut ctx = get_jit_state();
    jit_dirty_page_ctx(&mut ctx, start_page);

    // Note: This can't happen when paging is enabled, as writes across
    //       boundaries are split up on two pages
    if start_page != end_page {
        dbg_assert!(start_page.to_u32() + 1 == end_page.to_u32());
        jit_dirty_page_ctx(&mut ctx, end_page);
    }
}

#[no_mangle]
pub fn jit_clear_cache_js() { jit_clear_cache(&mut get_jit_state()) }

fn jit_clear_cache(ctx: &mut JitState) {
    let mut pages_with_code = HashSet::new();

    for &p in ctx.entry_points.keys() {
        pages_with_code.insert(p);
    }
    for &p in ctx.pages.keys() {
        pages_with_code.insert(p);
    }

    for page in pages_with_code {
        jit_dirty_page_ctx(ctx, page);
    }
}

pub fn jit_page_has_code(page: Page) -> bool { jit_page_has_code_ctx(&mut get_jit_state(), page) }

fn jit_page_has_code_ctx(ctx: &mut JitState, page: Page) -> bool {
    ctx.pages.contains_key(&page) || ctx.entry_points.contains_key(&page)
}

#[no_mangle]
pub fn jit_get_wasm_table_index_free_list_count() -> u32 {
    if cfg!(feature = "profiler") {
        get_jit_state().wasm_table_index_free_list.len() as u32
    }
    else {
        0
    }
}
#[no_mangle]
pub fn jit_get_cache_size() -> u32 {
    if cfg!(feature = "profiler") {
        get_jit_state()
            .pages
            .values()
            .map(|p| p.entry_points.len() as u32)
            .sum()
    }
    else {
        0
    }
}

// Ungated JIT-table diagnostics for slot pressure / hidden-index pile-up.
#[no_mangle]
pub fn jit_debug_free_slots() -> u32 {
    get_jit_state().wasm_table_index_free_list.len() as u32
}
#[no_mangle]
pub fn jit_debug_module_count() -> u32 {
    let ctx = get_jit_state();
    let mut set = HashSet::new();
    for info in ctx.pages.values() {
        set.insert(info.wasm_table_index);
    }
    set.len() as u32
}
#[no_mangle]
pub fn jit_debug_page_count() -> u32 {
    get_jit_state().pages.len() as u32
}
#[no_mangle]
pub fn jit_debug_hidden_count() -> u32 {
    get_jit_state()
        .pages
        .values()
        .map(|p| p.hidden_wasm_table_indices.len() as u32)
        .sum()
}
#[no_mangle]
pub fn jit_debug_max_region_pages() -> u32 {
    let ctx = get_jit_state();
    let mut per_index: HashMap<WasmTableIndex, u32> = HashMap::new();
    for info in ctx.pages.values() {
        *per_index.entry(info.wasm_table_index).or_insert(0) += 1;
    }
    per_index.values().copied().max().unwrap_or(0)
}

#[cfg(feature = "profiler")]
pub fn check_missed_entry_points(phys_address: u32, state_flags: CachedStateFlags) {
    let ctx = get_jit_state();

    if let Some(infos) = ctx.pages.get(&Page::page_of(phys_address)) {
        if infos.state_flags != state_flags {
            return;
        }

        #[allow(static_mut_refs)]
        let last_jump_type = unsafe { cpu::debug_last_jump.name() };
        #[allow(static_mut_refs)]
        let last_jump_addr = unsafe { cpu::debug_last_jump.phys_address() }.unwrap_or(0);
        let last_jump_opcode =
            if last_jump_addr != 0 { memory::read32s(last_jump_addr) } else { 0 };

        let opcode = memory::read32s(phys_address);
        dbg_log!(
            "Compiled exists, but no entry point, \
                 phys_addr={:x} opcode={:02x} {:02x} {:02x} {:02x}. \
                 Last jump at {:x} ({}) opcode={:02x} {:02x} {:02x} {:02x}",
            phys_address,
            opcode & 0xFF,
            opcode >> 8 & 0xFF,
            opcode >> 16 & 0xFF,
            opcode >> 16 & 0xFF,
            last_jump_addr,
            last_jump_type,
            last_jump_opcode & 0xFF,
            last_jump_opcode >> 8 & 0xFF,
            last_jump_opcode >> 16 & 0xFF,
            last_jump_opcode >> 16 & 0xFF,
        );
    }
}

#[no_mangle]
#[cfg(feature = "profiler")]
pub fn debug_set_dispatcher_target(_target_index: i32) {
    //dbg_log!("About to call dispatcher target_index={}", target_index);
}

#[no_mangle]
#[cfg(feature = "profiler")]
pub fn check_dispatcher_target(target_index: i32, max: i32) {
    //dbg_log!("Dispatcher called target={}", target_index);
    dbg_assert!(target_index >= 0);
    dbg_assert!(target_index < max);
}

#[no_mangle]
#[cfg(feature = "profiler")]
pub fn enter_basic_block(phys_eip: u32) {
    let eip =
        unsafe { cpu::translate_address_read(*global_pointers::instruction_pointer).unwrap() };
    if Page::page_of(eip) != Page::page_of(phys_eip) {
        dbg_log!(
            "enter basic block failed block=0x{:x} actual eip=0x{:x}",
            phys_eip,
            eip
        );
        panic!();
    }
}

pub const JIT_CONFIG_ABI_VERSION: u32 = 1;
const JIT_CONFIG_SUPPORTED_MASK: u32 = 0x01FB_FDEF;
const JIT_CONFIG_UNSUPPORTED: u32 = u32::MAX;

#[inline]
fn jit_config_is_supported(index: u32) -> bool {
    index < u32::BITS && JIT_CONFIG_SUPPORTED_MASK & (1 << index) != 0
}

#[no_mangle]
pub fn jit_config_abi_version() -> u32 { JIT_CONFIG_ABI_VERSION }

#[no_mangle]
pub fn jit_config_supported_mask() -> u32 { JIT_CONFIG_SUPPORTED_MASK }

// FNV-1a over the exact inputs that affect emitted JIT wasm. Field order is ABI-stable:
// configuration indices 1-3, 5-8, 10-14, 16-17, 19, and 21-23; relaxed-FPU mode and its
// hit/fallback counters; DISPATCH_STATS; and the fixed fastmem layout constants.
// Policy/accounting/diagnostic indices 0, 15, 20, and 24 deliberately do not participate.
fn jit_codegen_fingerprint() -> u64 {
    let mut hash = 0xCBF2_9CE4_8422_2325u64;
    let mut add = |value: u32| {
        hash ^= value as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    };
    unsafe {
        add(MAX_PAGES);
        add(JIT_USE_LOOP_SAFETY as u32);
        add(MAX_EXTRA_BASIC_BLOCKS);
        add(JIT_DEAD_FLAG_ELISION as u32);
        add(JIT_INDIRECT_REGIONS as u32);
        add(JIT_INDIRECT_REGION_MIN_SHARE);
        add(JIT_INDIRECT_REGION_MAX_PAGES);
        add(JIT_X87_LOCALS as u32);
        add(JIT_PUSH_RUN_COALESCING as u32);
        add(JIT_RET_CHAINING as u32);
        add(JIT_RET_SPECULATION as u32);
        add(JIT_RET_SPEC_MAX_INSTR);
        add(JIT_TIER2_RET_SPEC_MAX_INSTR);
        add(TIER2_MAX_PAGES);
        add(JIT_FASTMEM_WRITES as u32);
        add(JIT_FLAG_LOCALS as u32);
        add(JIT_BRANCH_HINTS);
        add(JIT_BRANCH_HINT_OFFSET_FUZZ);
        add(DISPATCH_STATS as u32);
    }
    add(crate::softfloat::get_relaxed_fpu());
    add(crate::softfloat::get_fpu_relaxed_stats());
    add(FASTMEM_LOW_MEM_END);
    add(FASTMEM_GUARD_BASE);
    add(FASTMEM_GUARD_SIZE);
    hash
}

#[no_mangle]
pub fn jit_codegen_fingerprint_lo() -> u32 { jit_codegen_fingerprint() as u32 }

#[no_mangle]
pub fn jit_codegen_fingerprint_hi() -> u32 { (jit_codegen_fingerprint() >> 32) as u32 }

#[no_mangle]
pub unsafe fn set_jit_config(index: u32, value: u32) -> u32 {
    if !jit_config_is_supported(index) {
        return JIT_CONFIG_UNSUPPORTED;
    }
    match index {
        0 => JIT_DISABLED = value != 0,
        1 => MAX_PAGES = value,
        2 => JIT_USE_LOOP_SAFETY = value != 0,
        3 => MAX_EXTRA_BASIC_BLOCKS = value,
        5 => JIT_DEAD_FLAG_ELISION = value != 0,
        6 => JIT_INDIRECT_REGIONS = value != 0,
        7 => JIT_INDIRECT_REGION_MIN_SHARE = value,
        8 => JIT_INDIRECT_REGION_MAX_PAGES = value,
        10 => JIT_X87_LOCALS = value != 0,
        11 => JIT_PUSH_RUN_COALESCING = value != 0,
        12 => JIT_RET_CHAINING = value != 0,
        13 => JIT_RET_SPECULATION = value != 0,
        14 => JIT_RET_SPEC_MAX_INSTR = value,
        15 => JIT_TIER2_THRESHOLD = value,
        16 => JIT_TIER2_RET_SPEC_MAX_INSTR = value,
        17 => TIER2_MAX_PAGES = value,
        19 => JIT_FASTMEM_WRITES = value != 0,
        20 => JIT_CHAIN_TIER2_ACCOUNTING = value != 0,
        21 => JIT_FLAG_LOCALS = value != 0,
        22 => JIT_BRANCH_HINTS = value,
        23 => JIT_BRANCH_HINT_OFFSET_FUZZ = value,
        24 => WRONG_ENTRY_REFUSE = value,
        _ => unreachable!(),
    }
    0
}

#[no_mangle]
pub unsafe fn get_jit_config(index: u32) -> u32 {
    if !jit_config_is_supported(index) {
        return JIT_CONFIG_UNSUPPORTED;
    }
    match index {
        0 => JIT_DISABLED as u32,
        1 => MAX_PAGES as u32,
        2 => JIT_USE_LOOP_SAFETY as u32,
        3 => MAX_EXTRA_BASIC_BLOCKS as u32,
        5 => JIT_DEAD_FLAG_ELISION as u32,
        6 => JIT_INDIRECT_REGIONS as u32,
        7 => JIT_INDIRECT_REGION_MIN_SHARE,
        8 => JIT_INDIRECT_REGION_MAX_PAGES,
        10 => JIT_X87_LOCALS as u32,
        11 => JIT_PUSH_RUN_COALESCING as u32,
        12 => JIT_RET_CHAINING as u32,
        13 => JIT_RET_SPECULATION as u32,
        14 => JIT_RET_SPEC_MAX_INSTR,
        15 => JIT_TIER2_THRESHOLD,
        16 => JIT_TIER2_RET_SPEC_MAX_INSTR,
        17 => TIER2_MAX_PAGES,
        19 => JIT_FASTMEM_WRITES as u32,
        20 => JIT_CHAIN_TIER2_ACCOUNTING as u32,
        21 => JIT_FLAG_LOCALS as u32,
        22 => JIT_BRANCH_HINTS,
        23 => JIT_BRANCH_HINT_OFFSET_FUZZ,
        24 => WRONG_ENTRY_REFUSE,
        _ => unreachable!(),
    }
}

// ──────────────────────────────────────────────────────────────────────────
// JIT cache snapshot for diagnostics (BottleShip dumpHotJitBlocks).
//
// Pattern: JS calls `jit_snapshot_cache()` to take a point-in-time snapshot
// of the current JitState.pages map, sorted by physical page address for
// stable output. Entry fields are then read one at a time via the three
// accessor functions. Kept in a mutable static mirroring the existing
// JIT_DISABLED / MAX_PAGES pattern.
// ──────────────────────────────────────────────────────────────────────────

// (wasm_table_index, phys_page_addr, entry_points_count)
static mut JIT_CACHE_SNAPSHOT: Option<Vec<(u16, u32, u16)>> = None;

#[no_mangle]
pub unsafe fn jit_snapshot_cache() -> u32 {
    let ctx = get_jit_state();
    let mut snapshot: Vec<(u16, u32, u16)> = ctx
        .pages
        .iter()
        .map(|(page, info)| {
            (
                info.wasm_table_index.to_u16(),
                page.to_address(),
                info.entry_points.len() as u16,
            )
        })
        .collect();
    // Sort by physical page address so repeated snapshots give stable indexing.
    snapshot.sort_by_key(|&(_, addr, _)| addr);
    let len = snapshot.len() as u32;
    JIT_CACHE_SNAPSHOT = Some(snapshot);
    len
}

#[no_mangle]
pub unsafe fn jit_snapshot_get_wasm_idx(i: u32) -> u32 {
    if let Some(ref snap) = JIT_CACHE_SNAPSHOT {
        if let Some(&(idx, _, _)) = snap.get(i as usize) {
            return idx as u32;
        }
    }
    0
}

#[no_mangle]
pub unsafe fn jit_snapshot_get_phys_addr(i: u32) -> u32 {
    if let Some(ref snap) = JIT_CACHE_SNAPSHOT {
        if let Some(&(_, addr, _)) = snap.get(i as usize) {
            return addr;
        }
    }
    0
}

#[no_mangle]
pub unsafe fn jit_snapshot_get_entry_count(i: u32) -> u32 {
    if let Some(ref snap) = JIT_CACHE_SNAPSHOT {
        if let Some(&(_, _, count)) = snap.get(i as usize) {
            return count as u32;
        }
    }
    0
}

// AOT offline driver (spike): feature-gated, absent from the shipped engine.
#[cfg(feature = "aot-driver")]
mod aot_driver;
