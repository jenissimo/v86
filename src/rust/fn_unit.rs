#![allow(static_mut_refs)]

// Function units: the baseline oracle for a function tier. Default OFF.
//
// A unit is one census row made executable: an M1 row (no unit refusal, clean static window,
// a known return form) compiled from its entry by the page JIT's own emitter, with the block
// walk held to the row's code pages and runs. It has exactly one entry, the function start,
// and keeps canonical CPU state at every boundary: it leaves through the emitter's ordinary
// exits with the exact EIP. A RET that lands on the return address pushed at entry, with the
// recorded `ret imm16` applied to ESP, is MATCHED; every other exit is EXITED and the
// dispatcher continues at that EIP.
//
// Units live in a dispatcher-owned table keyed by the virtual entry. Each entry has a
// generation. A write to any page of a unit (the existing jit_dirty_page path, reached through
// the TLB has-code bit the unit sets on its pages) bumps the generation, frees the table slot
// and drops the unit. A compile records the generation it started under; a write during the
// asynchronous compile makes the finished module stale and it is freed instead of published.
// The dispatcher enters a unit only at the current generation and the unit's state flags.
//
// Off, the dispatcher pays one load and one untaken branch; page modules, the codegen
// fingerprint and the census are unchanged. The mutation switch exists only for the negative
// controls in tests/jit-fn-unit.mjs.
//
// Chained calls (B, separate switch, default off). With the switch on every unit is emitted as
// a module of two functions. `g` takes the guest state as values, (eax..edi, the native lazy-flag
// tuple flags/flags_changed/last_op_size/last_op1/last_result, budget), and returns
// (outcome, eax..edi, the same tuple, budget); it sits in a second dispatcher-owned table slot.
// `f` is the dispatcher entry: it loads the state, calls `g` and stores the result on MATCHED.
// A direct CALL in a unit whose target is a published unit becomes, after a depth, budget,
// slice, callee-token and TLB-mapping check, the committed CALL (push, EIP = target) and a
// call_indirect of the callee's `g`. Any failed check leaves the unit at the CALL EIP before
// anything commits. The push can write a code page and drop the callee: the callee token is
// checked again after it, and a moved token leaves with the CALL committed (pushed stack, EIP
// at the target, the CALL retired once). The budget is the live remainder: the dispatcher
// starts `f` with the smaller of the loop budget and what is left of the slice, and every
// chained call honours the slice deadline and an urgent exit request (cycle limit 0) like the
// page chain. A unit that leaves before its first instruction retired (a chained CALL at its
// entry that fell back) does not end the dispatch: it continues on the ordinary path, so the
// unit is not entered again at the same EIP without progress. After MATCHED the caller's
// EIP regains its pre-call value, so the block epilogue computes the continuation once.
// Sites are admitted only for a plain E8 rel32 whose decoded target is the recorded one, and
// units only for a 32-bit stack segment. The callee's RET checks ESP against its entry ESP
// and the popped value against the expected continuation before it pops; on a mismatch it
// leaves at the RET EIP.
// EXITED from a callee propagates without writing caller locals: the callee left canonical
// state. After MATCHED the caller compares its own token; a moved token stores the returned
// state at the continuation and leaves. Units promote no stack memory, so there is nothing to
// publish around a call. Mode 1 is the mechanism control (state stored and reloaded at every
// call boundary); mode 2 keeps GPRs and flags private across the call.

use crate::jit::{FixedHasher, WasmTableIndex};
use crate::page::Page;
use crate::state_flags::CachedStateFlags;
use std::collections::HashMap;

pub const SCHEMA_VERSION: u32 = 2;

pub const S_ENTRIES: usize = 0;
pub const S_MATCHED: usize = 1;
pub const S_EXITED: usize = 2;
pub const S_COMPILES: usize = 3;
pub const S_PUBLISHED: usize = 4;
pub const S_WRITTEN_DURING_COMPILE: usize = 5;
pub const S_COMPILE_FAILED: usize = 6;
pub const S_INVALIDATED: usize = 7;
pub const S_REFUSE_NOT_M1: usize = 8;
pub const S_REFUSE_RET_FORM: usize = 9;
pub const S_REFUSE_SHAPE: usize = 10;
pub const S_REFUSE_STATE: usize = 11;
pub const S_REFUSE_PHYS: usize = 12;
pub const S_REFUSE_STALE_ROW: usize = 13;
pub const S_DEFERRED_BUSY: usize = 14;
pub const S_SKIP_STACK: usize = 15;
pub const S_SKIP_GENERATION: usize = 16;
pub const S_SKIP_MAPPING: usize = 17;
pub const S_SKIP_STATE: usize = 18;
pub const S_INSNS: usize = 19;
pub const S_CHAIN_SITES: usize = 20;
pub const S_CHAIN_UNCHAINED: usize = 21;
pub const S_CHAIN_DROPPED: usize = 22;
pub const S_CHAIN_STALE_PUBLISH: usize = 23;
pub const S_MAX_RETIRED: usize = 24;
pub const S_NO_PROGRESS: usize = 25;
pub const S_CHAIN_TARGET_MISMATCH: usize = 26;
pub const STAT_COUNT: usize = 27;

pub const C_CALLS: usize = 0;
pub const C_FB_DEPTH: usize = 1;
pub const C_FB_BUDGET: usize = 2;
pub const C_FB_TOKEN: usize = 3;
pub const C_FB_MAPPING: usize = 4;
pub const C_CALLEE_EXITED: usize = 5;
pub const C_CALLER_TOKEN: usize = 6;
pub const C_RET_MATCHED: usize = 7;
pub const C_RET_MISMATCH: usize = 8;
pub const C_TOKEN_AFTER_PUSH: usize = 9;
pub const C_COUNT: usize = 10;

pub const CHAIN_MAX_DEPTH: u32 = 16;

pub const M_NO_INVALIDATION: u32 = 1 << 0;
pub const M_NO_GENERATION_CHECK: u32 = 1 << 1;
pub const M_RESUME_AFTER_EXIT: u32 = 1 << 2;
pub const M_NO_ACTIVE_WRITE_EXIT: u32 = 1 << 3;
pub const M_CHAIN_NO_TOKEN: u32 = 1 << 4;
pub const M_CHAIN_NO_RET_CHECK: u32 = 1 << 5;
pub const M_CHAIN_NO_CALLER_TOKEN: u32 = 1 << 6;
pub const M_CHAIN_NO_DEPTH: u32 = 1 << 7;
pub const M_CHAIN_STALE_FLAGS: u32 = 1 << 8;
pub const M_CHAIN_NO_DEPENDANTS: u32 = 1 << 9;
pub const M_CHAIN_BUDGET_NOT_CARRIED: u32 = 1 << 10;
pub const M_CHAIN_NO_TOKEN_AFTER_PUSH: u32 = 1 << 11;
pub const M_CHAIN_EIP_ABSOLUTE: u32 = 1 << 12;
pub const M_NO_PROGRESS_GUARD: u32 = 1 << 13;
pub const M_CHAIN_FIXED_BUDGET: u32 = 1 << 14;
pub const M_ADMIT_SS16: u32 = 1 << 15;

static mut ENABLED: bool = false;
static mut CHAIN_MODE: u32 = 0;
static mut AND_ESP: bool = false;
static mut CHAIN_COUNTING: bool = false;
static mut NEXT_TOKEN: u32 = 1;
pub static mut CHAIN_DEPTH: u32 = 0;
pub static mut CHAIN_RET_EXPECT: [u32; CHAIN_MAX_DEPTH as usize + 1] = [0; CHAIN_MAX_DEPTH as usize + 1];
pub static mut CHAIN_TOKENS: [u32; crate::jit::WASM_TABLE_SIZE as usize] = [0; crate::jit::WASM_TABLE_SIZE as usize];
pub static mut CHAIN_COUNTERS: [u32; C_COUNT] = [0; C_COUNT];
pub static mut CHAIN_ENTRY_BUDGET: u32 = 0;
static mut MUTATION: u32 = 0;
static mut THRESHOLD: u32 = 16;
static mut STATS: [f64; STAT_COUNT] = [0.0; STAT_COUNT];

pub struct Unit {
    pub phys: u32,
    pub index: WasmTableIndex,
    pub state: u16,
    pub generation: u32,
    pub state_flags: CachedStateFlags,
    pub pages: Vec<Page>,
    pub ret_imm: u32,
    pub chain: Option<WasmTableIndex>,
    pub token: u32,
    pub links: Vec<u32>,
}

pub struct Pending {
    pub virt: u32,
    pub phys: u32,
    pub state: u16,
    pub generation: u32,
    pub state_flags: CachedStateFlags,
    pub pages: Vec<Page>,
    pub ret_imm: u32,
    pub chain: Option<WasmTableIndex>,
    pub token: u32,
    pub callees: Vec<(u32, u16, u32)>,
    pub waiting_on: Vec<u32>,
    pub links: Vec<u32>,
}

/// One direct CALL of a unit being compiled whose target is a published unit (or the unit
/// itself): the callee's chain slot, its token slot and token, and its entry page.
#[derive(Clone, Copy)]
pub struct ChainSite {
    pub site_virt: u32,
    pub target: u32,
    pub callee_phys: u32,
    pub chain_slot: u16,
    pub token_slot: u16,
    pub token: u32,
}

pub struct Units {
    pub units: HashMap<u32, Unit, FixedHasher>,
    pub generations: HashMap<u32, u32, FixedHasher>,
    pub page_entries: HashMap<Page, Vec<u32>, FixedHasher>,
    pub refused: HashMap<u32, u32, FixedHasher>,
    pub pending: Option<Pending>,
    pub dependants: HashMap<u32, Vec<u32>, FixedHasher>,
    pub waiting: HashMap<u32, Vec<u32>, FixedHasher>,
}

impl Units {
    pub fn new() -> Units {
        Units {
            units: HashMap::default(),
            generations: HashMap::default(),
            page_entries: HashMap::default(),
            refused: HashMap::default(),
            pending: None,
            dependants: HashMap::default(),
            waiting: HashMap::default(),
        }
    }

    pub fn generation(&self, virt: u32) -> u32 { *self.generations.get(&virt).unwrap_or(&0) }

    /// The active-code-write owner asks whether the slot running a store owns `page`.
    pub fn slot_has_page(&self, index: u16, page: Page) -> bool {
        !mutation(M_NO_ACTIVE_WRITE_EXIT) && !self.units.is_empty() && self.units.values().any(|u| u.index.to_u16() == index && u.pages.contains(&page))
    }

    /// Removes a published unit and every unit compiled against it. Returns the slots to free.
    pub fn drop_unit(&mut self, virt: u32, freed: &mut Vec<WasmTableIndex>) {
        let mut work = vec![virt];
        while let Some(v) = work.pop() {
            if let Some(u) = self.units.remove(&v) {
                set_token(u.index, 0);
                for p in &u.pages {
                    if let Some(list) = self.page_entries.get_mut(p) {
                        list.retain(|&x| x != v);
                    }
                }
                freed.push(u.index);
                if let Some(c) = u.chain {
                    freed.push(c);
                }
                if v != virt {
                    stat_add(S_CHAIN_DROPPED, 1.0);
                }
            }
            if let Some(deps) = self.dependants.remove(&v) {
                if !mutation(M_CHAIN_NO_DEPENDANTS) {
                    work.extend(deps.into_iter().filter(|&d| d != v));
                }
            }
        }
    }

    pub fn track_pages(&mut self, virt: u32, pages: &[Page]) {
        for &p in pages {
            let v = self.page_entries.entry(p).or_insert_with(Vec::new);
            if !v.contains(&virt) {
                v.push(virt);
            }
        }
    }

    /// A write reached `page`. Returns the table slots to free.
    pub fn note_page_dirty(&mut self, page: Page) -> Vec<WasmTableIndex> {
        let mut freed = Vec::new();
        if mutation(M_NO_INVALIDATION) {
            return freed;
        }
        let entries = match self.page_entries.remove(&page) {
            Some(v) => v,
            None => return freed,
        };
        for virt in entries {
            let g = self.generations.entry(virt).or_insert(0);
            *g = g.wrapping_add(1);
            self.refused.remove(&virt);
            if self.units.contains_key(&virt) {
                stat_add(S_INVALIDATED, 1.0);
            }
            self.drop_unit(virt, &mut freed);
        }
        freed
    }

    pub fn take_all(&mut self) -> Vec<WasmTableIndex> {
        let mut freed = Vec::new();
        for (_, u) in self.units.drain() {
            set_token(u.index, 0);
            freed.push(u.index);
            if let Some(c) = u.chain {
                freed.push(c);
            }
        }
        self.dependants = HashMap::default();
        self.waiting = HashMap::default();
        for g in self.generations.values_mut() {
            *g = g.wrapping_add(1);
        }
        if let Some(p) = &self.pending {
            let g = self.generations.entry(p.virt).or_insert(0);
            *g = g.wrapping_add(1);
        }
        self.page_entries = HashMap::default();
        self.refused = HashMap::default();
        freed
    }
}

#[inline]
pub fn enabled() -> bool { unsafe { ENABLED } }

#[inline]
pub fn chain_mode() -> u32 { unsafe { CHAIN_MODE } }

pub fn and_esp() -> bool { unsafe { AND_ESP } }

pub fn chain_counting() -> bool { unsafe { CHAIN_COUNTING } }

pub fn next_token() -> u32 {
    unsafe {
        let t = NEXT_TOKEN;
        NEXT_TOKEN = if NEXT_TOKEN == u32::MAX { 1 } else { NEXT_TOKEN + 1 };
        t
    }
}

pub fn set_token(index: WasmTableIndex, token: u32) {
    unsafe { CHAIN_TOKENS[index.to_u16() as usize] = token }
}

pub fn token_of(slot: u16) -> u32 { unsafe { CHAIN_TOKENS[slot as usize] } }

pub fn token_addr(slot: u16) -> u32 { unsafe { std::ptr::addr_of!(CHAIN_TOKENS[slot as usize]) as u32 } }

pub fn depth_addr() -> u32 { std::ptr::addr_of!(CHAIN_DEPTH) as u32 }

pub fn entry_budget_addr() -> u32 { std::ptr::addr_of!(CHAIN_ENTRY_BUDGET) as u32 }

pub fn set_entry_budget(n: u32) { unsafe { CHAIN_ENTRY_BUDGET = n } }

pub fn ret_expect_addr() -> u32 { std::ptr::addr_of!(CHAIN_RET_EXPECT) as u32 }

pub fn counter_addr(i: usize) -> u32 { unsafe { std::ptr::addr_of!(CHAIN_COUNTERS[i]) as u32 } }

pub fn reset_depth() { unsafe { CHAIN_DEPTH = 0 } }

pub fn set_chain_mode_raw(mode: u32) { unsafe { CHAIN_MODE = mode } }

pub fn set_and_esp_raw(on: bool) { unsafe { AND_ESP = on } }

#[no_mangle]
pub fn fn_unit_set_chain_counting(on: u32) { unsafe { CHAIN_COUNTING = on != 0 } }

#[no_mangle]
pub fn fn_unit_chain_counter(i: u32) -> u32 {
    unsafe { if (i as usize) < C_COUNT { CHAIN_COUNTERS[i as usize] } else { 0 } }
}

#[no_mangle]
pub fn fn_unit_chain_counter_count() -> u32 { C_COUNT as u32 }

#[no_mangle]
pub fn fn_unit_chain_depth() -> u32 { unsafe { CHAIN_DEPTH } }

pub fn mutation(bit: u32) -> bool { unsafe { MUTATION & bit != 0 } }

pub fn threshold() -> u32 { unsafe { THRESHOLD } }

pub fn stat_add(i: usize, n: f64) { unsafe { STATS[i] += n } }

pub fn stat_max(i: usize, n: f64) { unsafe { STATS[i] = STATS[i].max(n) } }

pub fn set_enabled(on: bool) { unsafe { ENABLED = on } }

#[no_mangle]
pub fn fn_unit_version() -> u32 { SCHEMA_VERSION }

#[no_mangle]
pub fn fn_unit_set_threshold(n: u32) { unsafe { THRESHOLD = n.max(1) } }

#[no_mangle]
pub fn fn_unit_get_threshold() -> u32 { threshold() }

#[no_mangle]
pub fn fn_unit_set_mutation(bits: u32) { unsafe { MUTATION = bits } }

#[no_mangle]
pub fn fn_unit_get_mutation() -> u32 { unsafe { MUTATION } }

#[no_mangle]
pub fn fn_unit_stat(i: u32) -> f64 {
    unsafe { if (i as usize) < STAT_COUNT { STATS[i as usize] } else { 0.0 } }
}

#[no_mangle]
pub fn fn_unit_stat_count() -> u32 { STAT_COUNT as u32 }

pub fn reset_stats() {
    unsafe {
        STATS = [0.0; STAT_COUNT];
        CHAIN_COUNTERS = [0; C_COUNT];
    }
}
