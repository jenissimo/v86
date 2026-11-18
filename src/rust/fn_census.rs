#![allow(static_mut_refs)]

// Function census: admission data for a future function tier, nothing else. Default OFF.
//
// While on, every 32-bit direct CALL compiled by the page JIT calls `fn_census_note_call`
// with its target and the callee's entry ESP. The first call to a target decodes it once
// (recursive descent over the analyser, the page budget counted over every decoded byte)
// and records: unit admission or the refusal reasons, stack-window shape (ESP/EBP tracked
// from the entry), slot-promotion and P4 flags, code pages and page-crossing instructions.
// Every recorded entry probes the window over RESIDENT TLB entries only: valid, writable,
// permitted, RAM, without code, and physical == linear. The probe never walks page tables.
//
// The call hook changes generated code, so the switch is folded into the codegen
// fingerprint and callers clear the JIT cache through their existing owner. Off, codegen
// is byte-identical and the TLB writer, the exception path and the page invalidation path
// each take one untaken branch. Rows live in JitState, the window generation and stats next
// to the TLB they observe; CPU reset and state restore bump the census generation so a
// reader can refuse a window across them. Every map uses the fixed hasher: a RandomState
// created here would reseed the HashSets the page walk creates after it, and change which
// pages a stock module admits even with the census off.
//
// Schema 2 adds the call sites of every candidate (direct target or indirect, and the
// callee's return form). A direct call moves the caller's ESP model by the callee's
// `ret imm16`; an indirect call or a callee with no or mixed returns leaves ESP unknown.
//
// The window generation is 64-bit and saturates: a proof taken under an older generation
// never compares equal again, and an exhausted generation refuses new proofs. The mutation
// switch exists only for the negative controls in tests/jit-fn-census.mjs.

use crate::analysis::{self, AnalysisType};
use crate::cpu::cpu;
use crate::cpu::global_pointers;
use crate::cpu::memory;
use crate::cpu_context::{self, CpuContext};
use crate::page::Page;
use crate::state_flags::CachedStateFlags;
use crate::jit::FixedHasher;
use std::collections::{BTreeMap, HashMap, HashSet};

pub const SCHEMA_VERSION: u32 = 2;

pub const ROW_WORDS: usize = 40;
pub const MAX_ROWS: usize = 8192;
pub const MAX_RANGES: usize = 65536;
const MAX_PROOFS: usize = 16;
const RECENT_WINDOW_PAGES: usize = 64;

pub const R_ENTRY_VIRT: usize = 0;
pub const R_ENTRY_PHYS: usize = 1;
pub const R_ENTRIES: usize = 2;
pub const R_UNIT_REFUSE: usize = 3;
pub const R_WINDOW_STATIC: usize = 4;
pub const R_PROMOTION: usize = 5;
pub const R_P4_REFUSE: usize = 6;
pub const R_INSNS: usize = 7;
pub const R_BLOCKS: usize = 8;
pub const R_BYTES: usize = 9;
pub const R_CODE_PAGES: usize = 10;
pub const R_PAGE_CROSSING: usize = 11;
pub const R_FRAME_LO: usize = 12;
pub const R_FRAME_HI: usize = 13;
pub const R_WINDOW_ADMITTED: usize = 14;
pub const R_WINDOW_REFUSED: usize = 15;
pub const R_WINDOW_REFUSE_SEEN: usize = 16;
pub const R_WINDOW_MAX_PAGES: usize = 17;
pub const R_SMC_EVENTS: usize = 18;
pub const R_FRAME_ACCESSES: usize = 19;
pub const R_OTHER_ACCESSES: usize = 20;
pub const R_X87_OPS: usize = 21;
pub const R_FIRST_REFUSE_EIP: usize = 22;
pub const R_FIRST_REFUSE_OP: usize = 23;
pub const R_RANGE_START: usize = 24;
pub const R_RANGE_COUNT: usize = 25;
pub const R_FLAGS: usize = 26;
pub const R_OUT_CALLS: usize = 27;
pub const R_ANALYSES: usize = 28;
pub const R_CODE_PAGE0: usize = 29;
pub const R_FIRST_WINDOW_EIP: usize = 32;
pub const R_FIRST_WINDOW_OP: usize = 33;
pub const R_CALL_START: usize = 34;
pub const R_CALL_COUNT: usize = 35;
pub const R_RET_SUMMARY: usize = 36;

pub const MAX_CALLS: usize = 131072;
pub const CALL_WORDS: usize = 4;
pub const CALL_INDIRECT: u32 = 1;
pub const CALL_CALLEE_RET_KNOWN: u32 = 2;
pub const CALL_CALLEE_RET_MIXED: u32 = 4;
pub const RET_KNOWN: u32 = 0x1_0000;
pub const RET_MIXED: u32 = 0x2_0000;

pub const ROW_FLAG_STALE: u32 = 1;
pub const ROW_FLAG_ANALYSED: u32 = 2;

pub const U_GUEST_CALL: u32 = 1 << 0;
pub const U_GUEST_CALL_INDIRECT: u32 = 1 << 1;
pub const U_INDIRECT_JUMP: u32 = 1 << 2;
pub const U_STATE_INSN: u32 = 1 << 3;
pub const U_HELPER_REENTRY: u32 = 1 << 4;
pub const U_PAGE_BUDGET: u32 = 1 << 5;
pub const U_INSN_CAP: u32 = 1 << 6;
pub const U_BLOCK_CAP: u32 = 1 << 7;
pub const U_BYTE_CAP: u32 = 1 << 8;
pub const U_CODE_NOT_MAPPED: u32 = 1 << 9;
pub const U_PAGE_CROSS_SPLIT: u32 = 1 << 10;
pub const U_ADDR16: u32 = 1 << 11;

pub const W_ESP_UNTRACKED: u32 = 1 << 0;
pub const W_ESP_ALIGN: u32 = 1 << 1;
pub const W_ESP_MERGE: u32 = 1 << 2;
pub const W_ESP_UNBALANCED: u32 = 1 << 3;
pub const W_OUT_CALL: u32 = 1 << 4;
pub const W_EXTENT_PAGES: u32 = 1 << 5;
pub const W_CALLEE_ESP_UNKNOWN: u32 = 1 << 6;

pub const PR_SLOT_ESCAPES: u32 = 1 << 0;
pub const PR_UNKNOWN_ALIAS: u32 = 1 << 1;
pub const PR_NO_WINDOW: u32 = 1 << 2;

pub const P4_FLDCW: u32 = 1 << 0;
pub const P4_DEPTH_MERGE: u32 = 1 << 1;
pub const P4_DEPTH_UNKNOWN: u32 = 1 << 2;
pub const P4_STRICT_FPU: u32 = 1 << 3;
pub const P4_FPU_ENV: u32 = 1 << 4;
pub const P4_NO_X87: u32 = 1 << 5;

pub const WP_NOT_RESIDENT: u32 = 1 << 0;
pub const WP_READONLY: u32 = 1 << 1;
pub const WP_NO_USER: u32 = 1 << 2;
pub const WP_MAPPED_RANGE: u32 = 1 << 3;
pub const WP_HAS_CODE: u32 = 1 << 4;
pub const WP_NOT_IDENTITY: u32 = 1 << 5;
pub const WP_WRAP: u32 = 1 << 6;
pub const WP_SS_NOT_FLAT: u32 = 1 << 7;
pub const WP_GENERATION_EXHAUSTED: u32 = 1 << 8;
pub const WP_EXTENT_PAGES: u32 = 1 << 9;
pub const WP_NO_STATIC_WINDOW: u32 = 1 << 10;

pub const S_CALLS: usize = 0;
pub const S_CALLS_TABLE_FULL: usize = 1;
pub const S_ANALYSES: usize = 2;
pub const S_REANALYSES: usize = 3;
pub const S_TLB_REVOCATIONS: usize = 4;
pub const S_REVOCATIONS_ON_WINDOW_PAGES: usize = 5;
pub const S_SMC_DIRTIES: usize = 6;
pub const S_SMC_ON_CANDIDATES: usize = 7;
pub const S_RANGE_OVERFLOW: usize = 8;
pub const S_CALLS_NOT_FLAT32: usize = 9;
pub const S_EXCEPTIONS: usize = 10;
pub const S_WINDOW_PROBES: usize = 11;
pub const S_WINDOW_ADMITTED: usize = 12;
pub const S_CALL_OVERFLOW: usize = 13;
const STAT_COUNT: usize = 14;

pub const M_SKIP_IDENTITY: u32 = 1 << 0;
pub const M_FILL_TLB: u32 = 1 << 1;
pub const M_NO_REVOCATION: u32 = 1 << 2;
pub const M_GENERATION_WRAPS: u32 = 1 << 3;
pub const M_JOIN_ESP_ONLY: u32 = 1 << 5;
pub const M_KEEP_OBSOLETE_RANGES: u32 = 1 << 6;
pub const M_CALLS_ESP_NEUTRAL: u32 = 1 << 4;
pub const M_NO_CALLEE_DEPS: u32 = 1 << 7;

const GENERATION_EXHAUSTED: u64 = u64::MAX;

static mut ENABLED: bool = false;
static mut MUTATION: u32 = 0;
static mut CENSUS_GENERATION: u32 = 0;
static mut WINDOW_GENERATION: u64 = 1;
static mut STATS: [f64; STAT_COUNT] = [0.0; STAT_COUNT];
static mut RECENT_PAGES: [u32; RECENT_WINDOW_PAGES] = [0; RECENT_WINDOW_PAGES];
static mut RECENT_NEXT: usize = 0;
static mut PROOFS: [(u64, u32, u32); MAX_PROOFS] = [(0, 0, 0); MAX_PROOFS];
static mut LAST_EXCEPTION_COUNTER: u32 = 0;
static mut LAST_EXCEPTION_VECTOR: u32 = u32::MAX;
static mut CAP_PAGES: u32 = 3;
static mut CAP_INSNS: u32 = 2048;
static mut CAP_BLOCKS: u32 = 512;
static mut CAP_BYTES: u32 = 12288;
static mut CAP_WINDOW_PAGES: u32 = 3;

pub struct Census {
    rows: Vec<u32>,
    index: HashMap<u32, u32, FixedHasher>,
    ranges: Vec<(u32, u32, u32)>,
    owners: BTreeMap<u32, (u32, u32)>,
    page_rows: HashMap<Page, Vec<u32>, FixedHasher>,
    calls: Vec<u32>,
    callee_ret: HashMap<u32, (u32, Vec<Page>), FixedHasher>,
    row_deps: HashMap<u32, Vec<Page>, FixedHasher>,
}

impl Census {
    pub fn new() -> Census {
        Census {
            rows: Vec::new(),
            index: HashMap::default(),
            ranges: Vec::new(),
            owners: BTreeMap::new(),
            page_rows: HashMap::default(),
            calls: Vec::new(),
            callee_ret: HashMap::default(),
            row_deps: HashMap::default(),
        }
    }
    fn clear(&mut self) {
        self.rows = Vec::new();
        self.index = HashMap::default();
        self.ranges = Vec::new();
        self.owners = BTreeMap::new();
        self.page_rows = HashMap::default();
        self.calls = Vec::new();
        self.callee_ret = HashMap::default();
        self.row_deps = HashMap::default();
    }
    fn row(&mut self, i: u32) -> &mut [u32] {
        let s = i as usize * ROW_WORDS;
        &mut self.rows[s..s + ROW_WORDS]
    }
}

#[inline]
pub fn enabled() -> bool { unsafe { ENABLED } }

fn stat_add(i: usize, n: f64) { unsafe { STATS[i] += n } }

#[no_mangle]
pub fn fn_census_version() -> u32 { SCHEMA_VERSION }

#[no_mangle]
pub fn fn_census_set_enabled(on: u32) {
    unsafe {
        ENABLED = on != 0;
        CENSUS_GENERATION = CENSUS_GENERATION.wrapping_add(1);
    }
}

#[no_mangle]
pub fn fn_census_enabled() -> u32 { enabled() as u32 }

#[no_mangle]
pub fn fn_census_generation() -> u32 { unsafe { CENSUS_GENERATION } }

#[no_mangle]
pub fn fn_census_reset() {
    crate::jit::with_fn_census(|c| c.clear());
    unsafe {
        STATS = [0.0; STAT_COUNT];
        RECENT_PAGES = [0; RECENT_WINDOW_PAGES];
        RECENT_NEXT = 0;
        PROOFS = [(0, 0, 0); MAX_PROOFS];
        LAST_EXCEPTION_COUNTER = 0;
        LAST_EXCEPTION_VECTOR = u32::MAX;
        CENSUS_GENERATION = CENSUS_GENERATION.wrapping_add(1);
    }
}

#[no_mangle]
pub fn fn_census_note_restore() {
    unsafe {
        CENSUS_GENERATION = CENSUS_GENERATION.wrapping_add(1);
    }
}

pub fn note_cpu_reset() { fn_census_note_restore() }

#[no_mangle]
pub fn fn_census_set_caps(pages: u32, insns: u32, blocks: u32, bytes: u32, window_pages: u32) {
    unsafe {
        CAP_PAGES = pages.max(1).min(4);
        CAP_INSNS = insns.max(1);
        CAP_BLOCKS = blocks.max(1);
        CAP_BYTES = bytes.max(1);
        CAP_WINDOW_PAGES = window_pages.max(1).min(16);
    }
}

#[no_mangle]
pub fn fn_census_get_cap(i: u32) -> u32 {
    unsafe {
        match i {
            0 => CAP_PAGES,
            1 => CAP_INSNS,
            2 => CAP_BLOCKS,
            3 => CAP_BYTES,
            4 => CAP_WINDOW_PAGES,
            _ => 0,
        }
    }
}

#[no_mangle]
pub fn fn_census_set_mutation(bits: u32) { unsafe { MUTATION = bits } }

#[no_mangle]
pub fn fn_census_get_mutation() -> u32 { unsafe { MUTATION } }

#[no_mangle]
pub fn fn_census_stat(i: u32) -> f64 { unsafe { if (i as usize) < STAT_COUNT { STATS[i as usize] } else { 0.0 } } }

#[no_mangle]
pub fn fn_census_stat_count() -> u32 { STAT_COUNT as u32 }

#[no_mangle]
pub fn fn_census_row_words() -> u32 { ROW_WORDS as u32 }

#[no_mangle]
pub fn fn_census_max_rows() -> u32 { MAX_ROWS as u32 }

#[no_mangle]
pub fn fn_census_row_count() -> u32 { crate::jit::with_fn_census(|c| (c.rows.len() / ROW_WORDS) as u32) }

#[no_mangle]
pub fn fn_census_rows_ptr() -> u32 { crate::jit::with_fn_census(|c| c.rows.as_ptr() as u32) }

#[no_mangle]
pub fn fn_census_call_count() -> u32 { crate::jit::with_fn_census(|c| (c.calls.len() / CALL_WORDS) as u32) }

#[no_mangle]
pub fn fn_census_calls_ptr() -> u32 { crate::jit::with_fn_census(|c| c.calls.as_ptr() as u32) }

#[no_mangle]
pub fn fn_census_range_count() -> u32 { crate::jit::with_fn_census(|c| c.ranges.len() as u32) }

#[no_mangle]
pub fn fn_census_range(i: u32, end: u32) -> u32 {
    crate::jit::with_fn_census(|c| match c.ranges.get(i as usize) {
        Some(&(s, e, _)) => if end != 0 { e } else { s },
        None => 0,
    })
}

#[no_mangle]
pub fn fn_census_owner_of(phys: u32) -> u32 {
    crate::jit::with_fn_census(|c| match c.owners.range(..=phys).next_back() {
        Some((_, &(end, row))) if phys < end => row + 1,
        _ => 0,
    })
}

#[no_mangle]
pub fn fn_census_window_generation_lo() -> u32 { unsafe { WINDOW_GENERATION as u32 } }

#[no_mangle]
pub fn fn_census_window_generation_hi() -> u32 { unsafe { (WINDOW_GENERATION >> 32) as u32 } }

#[no_mangle]
pub fn fn_census_test_set_window_generation(lo: u32, hi: u32) {
    unsafe { WINDOW_GENERATION = (hi as u64) << 32 | lo as u64 }
}

#[no_mangle]
pub fn fn_census_last_exception_counter() -> u32 { unsafe { LAST_EXCEPTION_COUNTER } }

#[no_mangle]
pub fn fn_census_last_exception_vector() -> u32 { unsafe { LAST_EXCEPTION_VECTOR } }

#[no_mangle]
pub fn fn_census_debug_tlb_entry(page: u32) -> u32 { unsafe { cpu::tlb_data[(page & 0xFFFFF) as usize] as u32 } }

#[cold]
#[inline(never)]
pub fn note_tlb_write(page: i32, new: i32) {
    let old = unsafe { cpu::tlb_data[page as usize] };
    unsafe {
        if old == 0 || old == new || MUTATION & M_NO_REVOCATION != 0 {
            return;
        }
        if WINDOW_GENERATION < GENERATION_EXHAUSTED {
            WINDOW_GENERATION += 1;
        }
        STATS[S_TLB_REVOCATIONS] += 1.0;
        let p = page as u32;
        if RECENT_PAGES.iter().any(|&r| r == p + 1) {
            STATS[S_REVOCATIONS_ON_WINDOW_PAGES] += 1.0;
        }
    }
}

#[cold]
#[inline(never)]
pub fn note_exception(vector: i32) {
    unsafe {
        LAST_EXCEPTION_COUNTER = *global_pointers::instruction_counter;
        LAST_EXCEPTION_VECTOR = vector as u32;
        STATS[S_EXCEPTIONS] += 1.0;
    }
}

fn remember_window_page(page: u32) {
    unsafe {
        if RECENT_PAGES.iter().any(|&r| r == page + 1) {
            return;
        }
        RECENT_PAGES[RECENT_NEXT] = page + 1;
        RECENT_NEXT = (RECENT_NEXT + 1) % RECENT_WINDOW_PAGES;
    }
}

pub fn window_probe(lo: u32, hi: u32) -> u32 {
    unsafe {
        let mut bits = 0;
        let sf = *global_pointers::state_flags;
        if !sf.has_flat_segmentation() || !sf.ssize_32() || !sf.is_32() {
            bits |= WP_SS_NOT_FLAT;
        }
        if WINDOW_GENERATION == GENERATION_EXHAUSTED {
            bits |= WP_GENERATION_EXHAUSTED;
        }
        if hi <= lo {
            return bits | WP_WRAP;
        }
        let first = lo >> 12;
        let last = (hi - 1) >> 12;
        if last - first + 1 > CAP_WINDOW_PAGES {
            return bits | WP_EXTENT_PAGES;
        }
        let user = *global_pointers::cpl == 3;
        for page in first..=last {
            let addr = (page << 12) as i32;
            let mut entry = cpu::tlb_data[page as usize];
            if MUTATION & M_FILL_TLB != 0 && entry & cpu::TLB_VALID == 0 {
                if let Some(phys) = cpu::translate_address_no_fault(addr, true) {
                    let host = phys.wrapping_add(memory::mem8 as u32);
                    entry = ((host ^ addr as u32) & !0xFFF) as i32 | cpu::TLB_VALID;
                }
            }
            if entry & cpu::TLB_VALID == 0 {
                bits |= WP_NOT_RESIDENT;
                continue;
            }
            if entry & cpu::TLB_READONLY != 0 {
                bits |= WP_READONLY;
            }
            if user && entry & cpu::TLB_NO_USER != 0 {
                bits |= WP_NO_USER;
            }
            if entry & cpu::TLB_IN_MAPPED_RANGE != 0 {
                bits |= WP_MAPPED_RANGE;
                continue;
            }
            if entry & cpu::TLB_HAS_CODE != 0 {
                bits |= WP_HAS_CODE;
            }
            if MUTATION & M_SKIP_IDENTITY == 0 {
                let host = (entry & !0xFFF ^ addr) as u32;
                let phys = host.wrapping_sub(memory::mem8 as u32);
                if phys >> 12 != page {
                    bits |= WP_NOT_IDENTITY;
                }
            }
        }
        bits
    }
}

#[no_mangle]
pub fn fn_census_window_probe(lo: u32, hi: u32) -> u32 { window_probe(lo, hi) }

#[no_mangle]
pub fn fn_census_proof_admit(lo: u32, hi: u32) -> u32 {
    if window_probe(lo, hi) != 0 {
        return 0;
    }
    unsafe {
        for (i, p) in PROOFS.iter_mut().enumerate() {
            if p.1 == p.2 {
                *p = (WINDOW_GENERATION, lo, hi);
                return i as u32 + 1;
            }
        }
    }
    0
}

#[no_mangle]
pub fn fn_census_proof_valid(id: u32) -> u32 {
    unsafe {
        let i = id as usize;
        if i == 0 || i > MAX_PROOFS {
            return 0;
        }
        let p = PROOFS[i - 1];
        if p.1 == p.2 {
            return 0;
        }
        let current = if MUTATION & M_GENERATION_WRAPS != 0 {
            WINDOW_GENERATION & 1
        }
        else {
            WINDOW_GENERATION
        };
        let held = if MUTATION & M_GENERATION_WRAPS != 0 { p.0 & 1 } else { p.0 };
        (current != GENERATION_EXHAUSTED && current == held) as u32
    }
}

#[no_mangle]
pub fn fn_census_proof_release(id: u32) {
    unsafe {
        let i = id as usize;
        if i >= 1 && i <= MAX_PROOFS {
            PROOFS[i - 1] = (0, 0, 0);
        }
    }
}

#[no_mangle]
pub fn fn_census_note_call(target: u32, esp: u32) {
    if !enabled() {
        return;
    }
    stat_add(S_CALLS, 1.0);
    let sf = unsafe { *global_pointers::state_flags };
    if !sf.is_32() || !sf.has_flat_segmentation() {
        stat_add(S_CALLS_NOT_FLAT32, 1.0);
        return;
    }
    crate::jit::with_fn_census(|c| {
        let row = match c.index.get(&target) {
            Some(&r) => r,
            None => {
                let n = (c.rows.len() / ROW_WORDS) as u32;
                if n as usize >= MAX_ROWS {
                    stat_add(S_CALLS_TABLE_FULL, 1.0);
                    return;
                }
                c.rows.extend_from_slice(&[0u32; ROW_WORDS]);
                c.row(n)[R_ENTRY_VIRT] = target;
                c.index.insert(target, n);
                n
            },
        };
        let flags = c.row(row)[R_FLAGS];
        if flags & ROW_FLAG_ANALYSED == 0 || flags & ROW_FLAG_STALE != 0 {
            if flags & ROW_FLAG_STALE != 0 {
                stat_add(S_REANALYSES, 1.0);
            }
            analyse_into(c, row, target, sf);
        }
        let r = c.row(row);
        r[R_ENTRIES] = r[R_ENTRIES].saturating_add(1);
        let lo_off = r[R_FRAME_LO] as i32;
        let hi_off = r[R_FRAME_HI] as i32;
        let wstatic = r[R_WINDOW_STATIC];
        let bits = if wstatic != 0 {
            WP_NO_STATIC_WINDOW
        }
        else {
            let lo = (esp as i64 + lo_off as i64) as i64;
            let hi = (esp as i64 + hi_off as i64) as i64;
            if lo < 0 || hi > 0x1_0000_0000 {
                WP_WRAP
            }
            else {
                stat_add(S_WINDOW_PROBES, 1.0);
                window_probe(lo as u32, hi as u32)
            }
        };
        if bits == 0 {
            stat_add(S_WINDOW_ADMITTED, 1.0);
            r[R_WINDOW_ADMITTED] = r[R_WINDOW_ADMITTED].saturating_add(1);
            let lo = (esp as i64 + lo_off as i64) as u32;
            let hi = (esp as i64 + hi_off as i64) as u32;
            let pages = ((hi - 1) >> 12) - (lo >> 12) + 1;
            if pages > r[R_WINDOW_MAX_PAGES] {
                r[R_WINDOW_MAX_PAGES] = pages;
            }
            for p in (lo >> 12)..=((hi - 1) >> 12) {
                remember_window_page(p);
            }
        }
        else {
            r[R_WINDOW_REFUSED] = r[R_WINDOW_REFUSED].saturating_add(1);
            r[R_WINDOW_REFUSE_SEEN] |= bits;
        }
    });
}

#[cold]
#[inline(never)]
pub fn note_page_dirty(c: &mut Census, page: Page) {
    stat_add(S_SMC_DIRTIES, 1.0);
    if !c.callee_ret.is_empty() {
        c.callee_ret = HashMap::default();
    }
    let rows = match c.page_rows.get(&page) {
        Some(rows) => rows.clone(),
        None => return,
    };
    stat_add(S_SMC_ON_CANDIDATES, 1.0);
    for row in rows {
        let r = c.row(row);
        r[R_SMC_EVENTS] = r[R_SMC_EVENTS].saturating_add(1);
        r[R_FLAGS] |= ROW_FLAG_STALE;
    }
}

fn frame_differs(a: &FlowState, b: &FlowState) -> bool {
    if unsafe { MUTATION } & M_JOIN_ESP_ONLY != 0 {
        return a.esp != b.esp;
    }
    a.esp != b.esp || a.ebp != b.ebp || a.saved_ebp != b.saved_ebp
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct FlowState {
    esp: Option<i32>,
    ebp: Option<i32>,
    x87: Option<i32>,
    saved_ebp: Option<(i32, i32)>,
}

struct Mem {
    base: i32,
    index: i32,
    disp: i32,
    size: i32,
}

struct Insn {
    op: u32,
    opsize16: bool,
    addr16: bool,
    modrm: Option<u8>,
    mem: Option<Mem>,
    imm_at: u32,
}

const NONE: i32 = -1;

fn has_modrm_1byte(op: u32) -> bool {
    matches!(op,
        0x00..=0x03 | 0x08..=0x0B | 0x10..=0x13 | 0x18..=0x1B | 0x20..=0x23 | 0x28..=0x2B
        | 0x30..=0x33 | 0x38..=0x3B | 0x62 | 0x63 | 0x69 | 0x6B | 0x80..=0x8F
        | 0xC0 | 0xC1 | 0xC4..=0xC7 | 0xD0..=0xD3 | 0xD8..=0xDF | 0xF6 | 0xF7 | 0xFE | 0xFF)
}

fn has_modrm_0f(op: u32) -> bool {
    !matches!(op,
        0x05..=0x0B | 0x0E | 0x30..=0x37 | 0x77 | 0x80..=0x8F | 0xA0..=0xA2 | 0xA8..=0xAA | 0xC8..=0xCF)
}

fn read8(phys: u32) -> u32 { memory::read8(phys) as u32 }

fn mem_size(op: u32, reg: u32) -> i32 {
    match op {
        0xDD | 0xDC => if op == 0xDD && (reg == 4 || reg == 6) { 108 } else { 8 },
        0xDB => if reg == 5 || reg == 7 { 10 } else { 4 },
        0xDF => if reg == 5 || reg == 7 { 8 } else if reg == 4 || reg == 6 { 10 } else { 2 },
        0xD9 => if reg == 4 || reg == 6 { 28 } else if reg == 5 || reg == 7 { 2 } else { 4 },
        0x1AE => if reg == 0 || reg == 1 { 512 } else { 4 },
        x if x >= 0x100 => 16,
        _ => 4,
    }
}

fn decode(phys: u32, len: u32) -> Insn {
    let mut p = phys;
    let end = phys + len;
    let mut opsize16 = false;
    let mut addr16 = false;
    loop {
        let b = read8(p);
        match b {
            0x66 => opsize16 = true,
            0x67 => addr16 = true,
            0x26 | 0x2E | 0x36 | 0x3E | 0x64 | 0x65 | 0xF0 | 0xF2 | 0xF3 => {},
            _ => break,
        }
        p += 1;
        if p >= end {
            break;
        }
    }
    let mut op = read8(p);
    p += 1;
    let mut modrm_present = has_modrm_1byte(op);
    if op == 0x0F {
        let b = read8(p);
        p += 1;
        if b == 0x38 || b == 0x3A {
            p += 1;
            op = 0x200 | b;
            modrm_present = true;
        }
        else {
            op = 0x100 | b;
            modrm_present = has_modrm_0f(b);
        }
    }
    let mut insn = Insn { op, opsize16, addr16, modrm: None, mem: None, imm_at: p - phys };
    if !modrm_present || p >= end {
        return insn;
    }
    let modrm = read8(p);
    p += 1;
    insn.modrm = Some(modrm as u8);
    let md = modrm >> 6;
    let rm = modrm & 7;
    let reg = (modrm >> 3) & 7;
    if md != 3 && !addr16 {
        let mut base = rm as i32;
        let mut index = NONE;
        if rm == 4 {
            let sib = read8(p);
            p += 1;
            base = (sib & 7) as i32;
            let idx = ((sib >> 3) & 7) as i32;
            if idx != 4 {
                index = idx;
            }
            if base == 5 && md == 0 {
                base = NONE;
            }
        }
        else if rm == 5 && md == 0 {
            base = NONE;
        }
        let disp = match md {
            0 => if base == NONE { let d = memory::read32s(p) ; p += 4; d } else { 0 },
            1 => { let d = read8(p) as u8 as i8 as i32; p += 1; d },
            _ => { let d = memory::read32s(p); p += 4; d },
        };
        insn.mem = Some(Mem { base, index, disp, size: mem_size(op, reg) });
    }
    else if md != 3 {
        insn.mem = Some(Mem { base: NONE, index: NONE, disp: 0, size: 4 });
    }
    insn.imm_at = p - phys;
    insn
}

fn writes_reg_field(op: u32) -> bool {
    matches!(op,
        0x02 | 0x03 | 0x0A | 0x0B | 0x12 | 0x13 | 0x1A | 0x1B | 0x22 | 0x23 | 0x2A | 0x2B
        | 0x32 | 0x33 | 0x8A | 0x8B | 0x8D | 0x69 | 0x6B | 0x86 | 0x87 | 0xC4 | 0xC5
        | 0x1AF | 0x1B6 | 0x1B7 | 0x1BE | 0x1BF | 0x140..=0x14F | 0x1BC | 0x1BD
        | 0x102 | 0x103 | 0x1B2 | 0x1B4 | 0x1B5 | 0x1C0 | 0x1C1)
}

fn writes_rm_field(op: u32, reg: u32) -> bool {
    match op {
        0x00 | 0x01 | 0x08 | 0x09 | 0x10 | 0x11 | 0x18 | 0x19 | 0x20 | 0x21 | 0x28 | 0x29
        | 0x30 | 0x31 | 0x86 | 0x87 | 0x88 | 0x89 | 0x8F | 0xC6 | 0xC7 | 0xC0 | 0xC1 | 0xD0..=0xD3
        | 0x1A4 | 0x1A5 | 0x1AC | 0x1AD | 0x1AB | 0x1B3 | 0x1BB | 0x1B0 | 0x1B1 | 0x1C0
        | 0x1C1 | 0x190..=0x19F => true,
        0x80 | 0x81 | 0x83 => reg != 7,
        0xF6 | 0xF7 => reg == 2 || reg == 3,
        0xFE | 0xFF => reg == 0 || reg == 1,
        0x1BA => reg >= 5,
        _ => false,
    }
}

fn reads_reg_field_as_value(op: u32) -> bool {
    matches!(op,
        0x00 | 0x01 | 0x08 | 0x09 | 0x10 | 0x11 | 0x18 | 0x19 | 0x20 | 0x21 | 0x28 | 0x29
        | 0x30 | 0x31 | 0x86 | 0x87 | 0x88 | 0x89 | 0x1A4 | 0x1A5 | 0x1AC | 0x1AD | 0x1B0 | 0x1B1
        | 0x1C0 | 0x1C1)
}

fn reads_rm_as_value(op: u32) -> bool {
    matches!(op,
        0x02 | 0x03 | 0x0A | 0x0B | 0x12 | 0x13 | 0x1A | 0x1B | 0x22 | 0x23 | 0x2A | 0x2B
        | 0x32 | 0x33 | 0x8A | 0x8B | 0x69 | 0x6B | 0x86 | 0x87 | 0x1AF | 0x140..=0x14F)
}

fn x87_effect(op: u32, modrm: u8) -> Result<i32, ()> {
    let md = modrm >> 6;
    let reg = ((modrm >> 3) & 7) as u32;
    let low = (modrm & 7) as u32;
    let mem = md != 3;
    Ok(match op {
        0xD8 => if reg == 3 { -1 } else { 0 },
        0xDC => if reg == 3 { -1 } else { 0 },
        0xD9 => if mem {
            match reg { 0 => 1, 3 => -1, _ => 0 }
        }
        else {
            match reg {
                0 => 1,
                1 => 0,
                2 => 0,
                3 => -1,
                4 => 0,
                5 => if low <= 6 { 1 } else { 0 },
                6 => match low { 0 | 2 | 4 => if low == 0 { 0 } else { 1 }, 1 | 3 => -1, 6 | 7 => return Err(()), _ => 0 },
                _ => match low { 1 => -1, 3 => 1, _ => 0 },
            }
        },
        0xDA => if mem { if reg == 3 { -1 } else { 0 } } else if reg == 5 && low == 1 { -2 } else { 0 },
        0xDB => if mem {
            match reg { 0 | 5 => 1, 1 | 3 | 7 => -1, _ => 0 }
        }
        else if reg == 4 && low == 3 {
            return Err(());
        }
        else {
            0
        },
        0xDD => if mem {
            match reg { 0 => 1, 1 | 3 => -1, _ => 0 }
        }
        else {
            match reg { 3 | 5 => -1, _ => 0 }
        },
        0xDE => if mem { if reg == 3 { -1 } else { 0 } } else if reg == 3 && low == 1 { -2 } else { -1 },
        0xDF => if mem {
            match reg { 0 | 5 | 4 => 1, 1 | 3 | 7 | 6 => -1, _ => 0 }
        }
        else {
            match reg { 5 | 6 => -1, _ => 0 }
        },
        _ => 0,
    })
}

fn is_finit(op: u32, modrm: u8) -> bool { op == 0xDB && modrm == 0xE3 }

struct Ctx {
    unit: u32,
    window: u32,
    promo: u32,
    p4: u32,
    insns: u32,
    blocks: u32,
    bytes: u32,
    crossing: u32,
    frame_lo: i32,
    frame_hi: i32,
    frame_accesses: u32,
    other_accesses: u32,
    x87_ops: u32,
    out_calls: u32,
    ret_summary: u32,
    calls: Vec<[u32; CALL_WORDS]>,
    dep_pages: Vec<Page>,
    first_refuse_eip: u32,
    first_refuse_op: u32,
    first_window_eip: u32,
    first_window_op: u32,
    pages: Vec<Page>,
    runs: Vec<(u32, u32)>,
}

impl Ctx {
    fn refuse(&mut self, bit: u32, eip: u32, op: u32, modrm: Option<u8>) {
        if self.unit == 0 {
            self.first_refuse_eip = eip;
            self.first_refuse_op = op | modrm.map_or(0, |m| ((m as u32 >> 3) & 7) << 12 | 0x8000);
        }
        self.unit |= bit;
    }
    fn page(&mut self, page: Page) -> bool {
        if !self.pages.contains(&page) {
            self.pages.push(page);
        }
        self.pages.len() as u32 <= unsafe { CAP_PAGES }
    }
}

fn analyse(entry: u32, sf: CachedStateFlags, mut callees: Option<&mut HashMap<u32, (u32, Vec<Page>), FixedHasher>>) -> (Ctx, u32) {
    let mut c = Ctx {
        unit: 0,
        window: 0,
        promo: 0,
        p4: 0,
        insns: 0,
        blocks: 0,
        bytes: 0,
        crossing: 0,
        frame_lo: 0,
        frame_hi: 4,
        frame_accesses: 0,
        other_accesses: 0,
        x87_ops: 0,
        out_calls: 0,
        ret_summary: 0,
        calls: Vec::new(),
        dep_pages: Vec::new(),
        first_refuse_eip: 0,
        first_refuse_op: 0,
        first_window_eip: 0,
        first_window_op: 0,
        pages: Vec::new(),
        runs: Vec::new(),
    };
    let cs_offset = cpu::get_seg_cs() as u32;
    let entry_phys = match cpu::translate_address_read_no_side_effects(entry as i32) {
        Ok(p) => p,
        Err(()) => {
            c.refuse(U_CODE_NOT_MAPPED, entry, 0, None);
            return (c, 0);
        },
    };
    let mut seen: HashMap<u32, FlowState, FixedHasher> = HashMap::default();
    let mut work: Vec<(u32, FlowState)> = vec![(entry, FlowState { esp: Some(0), ebp: None, x87: Some(0), saved_ebp: None })];
    let mut block_starts: HashSet<u32, FixedHasher> = HashSet::default();
    let mut ebp_is_frame_anywhere = false;
    let relaxed = crate::softfloat::get_relaxed_fpu() != 0;

    'paths: while let Some((start, mut st)) = work.pop() {
        if let Some(prev) = seen.get(&start) {
            if frame_differs(prev, &st) {
                if c.window == 0 {
                    c.first_window_eip = start;
                }
                c.window |= W_ESP_MERGE;
            }
            if prev.x87 != st.x87 {
                c.p4 |= P4_DEPTH_MERGE;
            }
            continue;
        }
        if block_starts.insert(start) {
            c.blocks += 1;
            if c.blocks > unsafe { CAP_BLOCKS } {
                c.refuse(U_BLOCK_CAP, start, 0, None);
                break;
            }
        }
        let mut virt = start;
        let mut run: Option<(u32, u32)> = None;
        loop {
            if let Some(prev) = seen.get(&virt) {
                if frame_differs(prev, &st) {
                    if c.window == 0 {
                        c.first_window_eip = virt;
                    }
                    c.window |= W_ESP_MERGE;
                }
                if prev.x87 != st.x87 {
                    c.p4 |= P4_DEPTH_MERGE;
                }
                break;
            }
            seen.insert(virt, st);
            let phys = match cpu::translate_address_read_no_side_effects(virt as i32) {
                Ok(p) => p,
                Err(()) => {
                    c.refuse(U_CODE_NOT_MAPPED, virt, 0, None);
                    break;
                },
            };
            match run {
                Some((s, e)) if e != phys => {
                    c.runs.push((s, e));
                    run = Some((phys, phys));
                },
                None => run = Some((phys, phys)),
                _ => {},
            }
            if !c.page(Page::page_of(phys)) {
                c.refuse(U_PAGE_BUDGET, virt, 0, None);
                if let Some(r) = run.take() {
                    c.runs.push(r);
                }
                break 'paths;
            }
            let mut ctx = CpuContext { eip: phys, prefixes: 0, cs_offset, state_flags: sf };
            let limit = (phys & !0xFFF).wrapping_add(0x1000);
            let (mut a, fits) = cpu_context::decode_within(limit, || analysis::analyze_step(&mut ctx));
            if !fits {
                let next_virt = (virt & !0xFFF).wrapping_add(0x1000);
                match cpu::translate_address_read_no_side_effects(next_virt as i32) {
                    Ok(np) if np == limit => {
                        c.crossing += 1;
                        if !c.page(Page::page_of(np)) {
                            c.refuse(U_PAGE_BUDGET, virt, 0, None);
                            break 'paths;
                        }
                        ctx = CpuContext { eip: phys, prefixes: 0, cs_offset, state_flags: sf };
                        a = analysis::analyze_step(&mut ctx);
                    },
                    _ => {
                        c.refuse(U_PAGE_CROSS_SPLIT, virt, 0, None);
                        break;
                    },
                }
            }
            let len = ctx.eip - phys;
            if let Some((s, _)) = run {
                run = Some((s, phys + len));
            }
            c.insns += 1;
            c.bytes += len;
            if c.insns > unsafe { CAP_INSNS } {
                c.refuse(U_INSN_CAP, virt, 0, None);
                break 'paths;
            }
            if c.bytes > unsafe { CAP_BYTES } {
                c.refuse(U_BYTE_CAP, virt, 0, None);
                break 'paths;
            }
            let ins = decode(phys, len);
            let next = virt.wrapping_add(len);
            let op = ins.op;
            let reg = ins.modrm.map_or(0, |m| ((m >> 3) & 7) as u32);
            let md = ins.modrm.map_or(0, |m| m >> 6);
            let rm = ins.modrm.map_or(0, |m| (m & 7) as u32);
            if ins.addr16 {
                c.refuse(U_ADDR16, virt, op, ins.modrm);
            }
            classify_unit(&mut c, &ins, virt, reg);
            if (0xD8..=0xDF).contains(&op) {
                c.x87_ops += 1;
                let m = ins.modrm.unwrap_or(0);
                if op == 0xD9 && md != 3 && reg == 5 {
                    c.p4 |= P4_FLDCW;
                }
                if (op == 0xD9 && md != 3 && (reg == 4 || reg == 6)) || (op == 0xDD && md != 3 && (reg == 4 || reg == 6)) {
                    c.p4 |= P4_FPU_ENV;
                }
                if is_finit(op, m) {
                    st.x87 = Some(0);
                }
                else {
                    match (x87_effect(op, m), st.x87) {
                        (Ok(d), Some(x)) => st.x87 = Some(x + d),
                        (Err(()), _) => {
                            c.p4 |= P4_DEPTH_UNKNOWN;
                            st.x87 = None;
                        },
                        (Ok(_), None) => {},
                    }
                }
            }
            if op == 0x1AE && md != 3 && (reg <= 1) {
                c.p4 |= P4_FPU_ENV;
            }
            track_frame(&mut c, &mut st, &ins, phys, reg, md, rm, &mut ebp_is_frame_anywhere);
            let stop = match a.ty {
                AnalysisType::Jump { offset, is_32, condition } => {
                    let target = if is_32 { next.wrapping_add(offset as u32) } else { (next.wrapping_sub(cs_offset).wrapping_add(offset as u32) & 0xFFFF).wrapping_add(cs_offset) };
                    if op == 0xE8 {
                        c.out_calls += 1;
                        c.window |= W_OUT_CALL;
                        let summary = match callees.as_deref_mut() {
                            Some(cache) => {
                                if !cache.contains_key(&target) {
                                    let callee = analyse(target, sf, None).0;
                                    cache.insert(target, (callee.ret_summary, callee.pages));
                                }
                                let (v, pages) = &cache[&target];
                                if unsafe { MUTATION } & M_NO_CALLEE_DEPS == 0 {
                                    for p in pages {
                                        if !c.dep_pages.contains(p) {
                                            c.dep_pages.push(*p);
                                        }
                                    }
                                }
                                *v
                            },
                            None => 0,
                        };
                        let mut flags = 0;
                        if summary & RET_MIXED != 0 {
                            flags |= CALL_CALLEE_RET_MIXED;
                        }
                        else if summary & RET_KNOWN != 0 {
                            flags |= CALL_CALLEE_RET_KNOWN;
                        }
                        c.calls.push([virt, target, flags, summary & 0xFFFF]);
                        if unsafe { MUTATION } & M_CALLS_ESP_NEUTRAL == 0 {
                            if flags & CALL_CALLEE_RET_KNOWN != 0 {
                                st.esp = st.esp.map(|e| e + (summary & 0xFFFF) as i32);
                            }
                            else {
                                st.esp = None;
                                c.window |= W_CALLEE_ESP_UNKNOWN;
                            }
                        }
                        false
                    }
                    else if condition.is_some() {
                        work.push((target, st));
                        false
                    }
                    else {
                        work.push((target, st));
                        true
                    }
                },
                _ => a.no_next_instruction,
            };
            if op == 0xFF && reg == 2 {
                c.calls.push([virt, 0, CALL_INDIRECT, 0]);
                if unsafe { MUTATION } & M_CALLS_ESP_NEUTRAL == 0 {
                    st.esp = None;
                    c.window |= W_CALLEE_ESP_UNKNOWN;
                }
            }
            if op == 0xC3 || op == 0xC2 {
                let imm = if op == 0xC2 { memory::read16(phys + ins.imm_at) as u32 } else { 0 };
                c.ret_summary = if c.ret_summary == 0 {
                    RET_KNOWN | imm
                }
                else if c.ret_summary & RET_MIXED != 0 || c.ret_summary & 0xFFFF != imm {
                    RET_MIXED
                }
                else {
                    c.ret_summary
                };
                if st.esp.map_or(false, |e| e != 0) {
                    c.window |= W_ESP_UNBALANCED;
                }
            }
            if c.window != 0 && c.first_window_eip == 0 {
                c.first_window_eip = virt;
                c.first_window_op = op | ins.modrm.map_or(0, |m| ((m as u32 >> 3) & 7) << 12 | 0x8000);
            }
            if stop || a.ty == AnalysisType::STI {
                break;
            }
            virt = next;
        }
        if let Some(r) = run.take() {
            c.runs.push(r);
        }
    }
    if c.x87_ops == 0 {
        c.p4 |= P4_NO_X87;
    }
    else if !relaxed {
        c.p4 |= P4_STRICT_FPU;
    }
    let extent = (c.frame_hi - c.frame_lo) as u32;
    if extent > unsafe { CAP_WINDOW_PAGES } * 0x1000 {
        c.window |= W_EXTENT_PAGES;
    }
    if c.window != 0 {
        c.promo |= PR_NO_WINDOW;
    }
    if c.other_accesses != 0 {
        c.promo |= PR_UNKNOWN_ALIAS;
    }
    c.runs.retain(|r| r.1 > r.0);
    (c, entry_phys)
}

fn classify_unit(c: &mut Ctx, ins: &Insn, virt: u32, reg: u32) {
    let op = ins.op;
    let m = ins.modrm;
    let state = match op {
        0x07 | 0x17 | 0x1F | 0x8E | 0x9A | 0x9D | 0xC4 | 0xC5 | 0xCA | 0xCB | 0xCC | 0xCD
        | 0xCE | 0xCF | 0xF1 | 0xF4 | 0xFA | 0xFB | 0xEA => true,
        0x100 | 0x101 | 0x106 | 0x108 | 0x109 | 0x120..=0x123 | 0x130 | 0x132 | 0x134
        | 0x135 | 0x105 | 0x107 | 0x1A1 | 0x1A9 | 0x1B2 | 0x1B4 | 0x1B5 => true,
        0xFF => reg == 3 || reg == 5,
        _ => false,
    };
    if state {
        c.refuse(U_STATE_INSN, virt, op, m);
    }
    if matches!(op, 0x6C..=0x6F | 0xE4..=0xE7 | 0xEC..=0xEF) {
        c.refuse(U_HELPER_REENTRY, virt, op, m);
    }
    if op == 0xE8 {
        c.refuse(U_GUEST_CALL, virt, op, m);
    }
    if op == 0xFF && reg == 2 {
        c.refuse(U_GUEST_CALL_INDIRECT, virt, op, m);
    }
    if op == 0xFF && reg == 4 {
        c.refuse(U_INDIRECT_JUMP, virt, op, m);
    }
}

fn track_frame(
    c: &mut Ctx,
    st: &mut FlowState,
    ins: &Insn,
    phys: u32,
    reg: u32,
    md: u8,
    rm: u32,
    ebp_frame_seen: &mut bool,
) {
    let op = ins.op;
    const ESP: u32 = 4;
    const EBP: u32 = 5;
    let imm8 = || memory::read8(phys + ins.imm_at) as u8 as i8 as i32;
    let imm32 = || memory::read32s(phys + ins.imm_at);
    if let Some(mem) = &ins.mem {
        let off = if mem.index != NONE {
            None
        }
        else if mem.base == ESP as i32 {
            st.esp.map(|e| e + mem.disp)
        }
        else if mem.base == EBP as i32 {
            st.ebp.map(|e| e + mem.disp)
        }
        else {
            None
        };
        let lea = op == 0x8D;
        match off {
            Some(o) if !lea => {
                c.frame_accesses += 1;
                c.frame_lo = c.frame_lo.min(o);
                c.frame_hi = c.frame_hi.max(o + mem.size);
            },
            Some(_) => c.promo |= PR_SLOT_ESCAPES,
            None if !lea => c.other_accesses += 1,
            None => {},
        }
    }
    let push = |st: &mut FlowState, c: &mut Ctx, n: i32| {
        if let Some(e) = st.esp {
            let e = e - n;
            st.esp = Some(e);
            c.frame_lo = c.frame_lo.min(e);
        }
    };
    match op {
        0x50..=0x57 | 0x68 | 0x6A | 0x06 | 0x0E | 0x16 | 0x1E | 0x9C | 0x1A0 | 0x1A8 => {
            if op == 0x54 || (op == 0x55 && st.ebp.is_some()) {
                c.promo |= PR_SLOT_ESCAPES;
            }
            push(st, c, if ins.opsize16 { 2 } else { 4 });
            if op == 0x55 && !ins.opsize16 {
                if let (Some(e), Some(b)) = (st.esp, st.ebp) {
                    st.saved_ebp = Some((e, b));
                }
            }
            return;
        },
        0x60 => {
            c.promo |= PR_SLOT_ESCAPES;
            push(st, c, 32);
            return;
        },
        0xFF if reg == 6 => {
            push(st, c, 4);
            return;
        },
        0x58..=0x5F => {
            let r = op - 0x58;
            if r == ESP {
                st.esp = None;
                c.window |= W_ESP_UNTRACKED;
                return;
            }
            if let Some(e) = st.esp {
                if r == EBP {
                    st.ebp = match st.saved_ebp {
                        Some((at, v)) if at == e => Some(v),
                        _ => None,
                    };
                    st.saved_ebp = None;
                }
                st.esp = Some(e + 4);
            }
            return;
        },
        0x8F if reg == 0 => {
            if let Some(e) = st.esp {
                st.esp = Some(e + 4);
            }
            return;
        },
        0x61 => {
            st.ebp = None;
            if let Some(e) = st.esp {
                st.esp = Some(e + 32);
            }
            return;
        },
        0xC9 => {
            match st.ebp {
                Some(b) => st.esp = Some(b + 4),
                None => {
                    st.esp = None;
                    c.window |= W_ESP_UNTRACKED;
                },
            }
            st.ebp = None;
            return;
        },
        0xC8 => {
            c.window |= W_ESP_UNTRACKED;
            st.esp = None;
            return;
        },
        0xC2 | 0xC3 => return,
        0xE8 => return,
        _ => {},
    }
    if let Some(_) = ins.modrm {
        if md == 3 {
            if op == 0x8B && reg == EBP && rm == ESP || op == 0x89 && rm == EBP && reg == ESP {
                st.ebp = st.esp;
                *ebp_frame_seen = true;
                return;
            }
            if op == 0x8B && reg == ESP && rm == EBP || op == 0x89 && rm == ESP && reg == EBP {
                match st.ebp {
                    Some(b) => st.esp = Some(b),
                    None => {
                        st.esp = None;
                        c.window |= W_ESP_UNTRACKED;
                    },
                }
                return;
            }
            if (op == 0x83 || op == 0x81) && rm == ESP && (reg == 0 || reg == 5) && !ins.opsize16 {
                let v = if op == 0x83 { imm8() } else { imm32() };
                if let Some(e) = st.esp {
                    let e = if reg == 0 { e + v } else { e - v };
                    st.esp = Some(e);
                    c.frame_lo = c.frame_lo.min(e);
                }
                return;
            }
            if (op == 0x83 || op == 0x81) && rm == ESP && reg == 4 {
                st.esp = None;
                c.window |= W_ESP_ALIGN;
                return;
            }
            if writes_rm_field(op, reg) && rm == ESP {
                st.esp = None;
                c.window |= W_ESP_UNTRACKED;
            }
            if writes_rm_field(op, reg) && rm == EBP {
                st.ebp = None;
            }
            if reads_rm_as_value(op) && (rm == ESP || rm == EBP && st.ebp.is_some()) {
                c.promo |= PR_SLOT_ESCAPES;
            }
        }
        if op == 0x8D && reg == ESP {
            if let Some(mem) = &ins.mem {
                if mem.index == NONE && mem.base == EBP as i32 {
                    if let Some(b) = st.ebp {
                        st.esp = Some(b + mem.disp);
                        c.frame_lo = c.frame_lo.min(b + mem.disp);
                        return;
                    }
                }
                if mem.index == NONE && mem.base == ESP as i32 {
                    if let Some(e) = st.esp {
                        st.esp = Some(e + mem.disp);
                        c.frame_lo = c.frame_lo.min(e + mem.disp);
                        return;
                    }
                }
            }
        }
        if writes_reg_field(op) && reg == ESP {
            st.esp = None;
            c.window |= W_ESP_UNTRACKED;
        }
        if writes_reg_field(op) && reg == EBP {
            st.ebp = None;
        }
        if reads_reg_field_as_value(op) && (reg == ESP || reg == EBP && st.ebp.is_some()) {
            c.promo |= PR_SLOT_ESCAPES;
        }
    }
    else {
        match op {
            0xBC | 0x44 | 0x4C | 0x94 => {
                st.esp = None;
                c.window |= W_ESP_UNTRACKED;
            },
            0xBD | 0x45 | 0x4D | 0x95 => st.ebp = None,
            0x1CC | 0x1CD => {
                if op == 0x1CC {
                    st.esp = None;
                    c.window |= W_ESP_UNTRACKED;
                }
                else {
                    st.ebp = None;
                }
            },
            _ => {},
        }
    }
}

fn analyse_into(census: &mut Census, row: u32, entry: u32, sf: CachedStateFlags) {
    stat_add(S_ANALYSES, 1.0);
    let old_pages: Vec<Page> = {
        let r = census.row(row);
        (0..3).filter(|&i| r[R_CODE_PAGE0 + i] != 0).map(|i| Page::page_of(r[R_CODE_PAGE0 + i] << 12)).collect()
    };
    let old_deps = census.row_deps.remove(&row).unwrap_or_default();
    for p in old_pages.into_iter().chain(old_deps.into_iter()) {
        if let Some(v) = census.page_rows.get_mut(&p) {
            v.retain(|&x| x != row);
        }
    }
    let (old_start, old_count) = {
        let r = census.row(row);
        (r[R_RANGE_START] as usize, r[R_RANGE_COUNT] as usize)
    };
    let old_count = if unsafe { MUTATION } & M_KEEP_OBSOLETE_RANGES != 0 { 0 } else { old_count };
    for i in old_start..(old_start + old_count).min(census.ranges.len()) {
        let (s, e, owner) = census.ranges[i];
        census.ranges[i] = (0, 0, u32::MAX);
        if owner != row || e <= s {
            continue;
        }
        if census.owners.get(&s) == Some(&(e, row)) {
            census.owners.remove(&s);
            if let Some(&(s2, e2, r2)) = census.ranges.iter().find(|&&(s2, e2, r2)| s2 == s && e2 > s2 && r2 != row && r2 != u32::MAX) {
                census.owners.insert(s2, (e2, r2));
            }
        }
    }
    let (c, entry_phys) = analyse(entry, sf, Some(&mut census.callee_ret));
    let mut range_start = census.ranges.len() as u32;
    let mut range_count = 0;
    for &(s, e) in &c.runs {
        if census.ranges.len() >= MAX_RANGES {
            stat_add(S_RANGE_OVERFLOW, 1.0);
            break;
        }
        census.ranges.push((s, e, row));
        census.owners.entry(s).or_insert((e, row));
        range_count += 1;
    }
    if range_count == 0 {
        range_start = 0;
    }
    for &p in &c.pages {
        census.page_rows.entry(p).or_insert_with(Vec::new).push(row);
    }
    let deps: Vec<Page> = c.dep_pages.iter().copied().filter(|p| !c.pages.contains(p)).collect();
    if !deps.is_empty() {
        for &p in &deps {
            census.page_rows.entry(p).or_insert_with(Vec::new).push(row);
        }
        census.row_deps.insert(row, deps);
    }
    let call_start = (census.calls.len() / CALL_WORDS) as u32;
    let mut call_count = 0;
    for rec in &c.calls {
        if census.calls.len() / CALL_WORDS >= MAX_CALLS {
            stat_add(S_CALL_OVERFLOW, 1.0);
            break;
        }
        census.calls.extend_from_slice(rec);
        call_count += 1;
    }
    let analyses = census.row(row)[R_ANALYSES];
    let r = census.row(row);
    r[R_ENTRY_PHYS] = entry_phys;
    r[R_UNIT_REFUSE] = c.unit;
    r[R_WINDOW_STATIC] = c.window;
    r[R_PROMOTION] = c.promo;
    r[R_P4_REFUSE] = c.p4;
    r[R_INSNS] = c.insns;
    r[R_BLOCKS] = c.blocks;
    r[R_BYTES] = c.bytes;
    r[R_CODE_PAGES] = c.pages.len() as u32;
    r[R_PAGE_CROSSING] = c.crossing;
    r[R_FRAME_LO] = c.frame_lo as u32;
    r[R_FRAME_HI] = c.frame_hi as u32;
    r[R_FRAME_ACCESSES] = c.frame_accesses;
    r[R_OTHER_ACCESSES] = c.other_accesses;
    r[R_X87_OPS] = c.x87_ops;
    r[R_FIRST_REFUSE_EIP] = c.first_refuse_eip;
    r[R_FIRST_REFUSE_OP] = c.first_refuse_op;
    r[R_FIRST_WINDOW_EIP] = c.first_window_eip;
    r[R_FIRST_WINDOW_OP] = c.first_window_op;
    r[R_RANGE_START] = range_start;
    r[R_RANGE_COUNT] = range_count;
    r[R_OUT_CALLS] = c.out_calls;
    r[R_RET_SUMMARY] = c.ret_summary;
    r[R_CALL_START] = call_start;
    r[R_CALL_COUNT] = call_count;
    r[R_ANALYSES] = analyses + 1;
    r[R_FLAGS] = (r[R_FLAGS] & !ROW_FLAG_STALE) | ROW_FLAG_ANALYSED;
    for i in 0..3 {
        r[R_CODE_PAGE0 + i] = c.pages.get(i).map_or(0, |p| p.to_address() >> 12);
    }
}

pub struct UnitCandidate {
    pub phys: u32,
    pub entries: u32,
    pub unit: u32,
    pub window: u32,
    pub ret_summary: u32,
    pub analyses: u32,
    pub current: bool,
    pub pages: Vec<Page>,
    pub runs: Vec<(u32, u32)>,
}

pub fn unit_candidate(c: &mut Census, virt: u32) -> Option<UnitCandidate> {
    let row = *c.index.get(&virt)?;
    let r = c.row(row);
    let flags = r[R_FLAGS];
    let pages = (0..3)
        .filter(|&i| r[R_CODE_PAGE0 + i] != 0)
        .map(|i| Page::page_of(r[R_CODE_PAGE0 + i] << 12))
        .collect::<Vec<Page>>();
    let (start, count) = (r[R_RANGE_START] as usize, r[R_RANGE_COUNT] as usize);
    let mut cand = UnitCandidate {
        phys: r[R_ENTRY_PHYS],
        entries: r[R_ENTRIES],
        unit: r[R_UNIT_REFUSE],
        window: r[R_WINDOW_STATIC],
        ret_summary: r[R_RET_SUMMARY],
        analyses: r[R_ANALYSES],
        current: flags & ROW_FLAG_ANALYSED != 0 && flags & ROW_FLAG_STALE == 0,
        pages,
        runs: Vec::new(),
    };
    for i in start..(start + count).min(c.ranges.len()) {
        let (s, e, owner) = c.ranges[i];
        if owner == row && e > s {
            cand.runs.push((s, e));
        }
    }
    Some(cand)
}

/// The call sites recorded for the row at `virt`: (site, target, flags, callee ret imm16).
pub fn unit_calls(c: &mut Census, virt: u32) -> Vec<[u32; CALL_WORDS]> {
    let row = match c.index.get(&virt) {
        Some(&r) => r,
        None => return Vec::new(),
    };
    let r = c.row(row);
    let (start, count) = (r[R_CALL_START] as usize, r[R_CALL_COUNT] as usize);
    let mut out = Vec::with_capacity(count);
    for i in start..(start + count) {
        let s = i * CALL_WORDS;
        if s + CALL_WORDS > c.calls.len() {
            break;
        }
        let mut rec = [0u32; CALL_WORDS];
        rec.copy_from_slice(&c.calls[s..s + CALL_WORDS]);
        out.push(rec);
    }
    out
}
