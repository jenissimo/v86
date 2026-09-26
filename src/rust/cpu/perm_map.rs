//! Permission bitmap: one byte per 4 KiB page (docs/performance/sota-roadmap/03).
//!
//! The read fast path in `gen_safe_read` does a TLB-entry lookup — shift, scale by 4, load
//! a 32-bit entry out of a 4 MiB table, mask, compare, then XOR-remap the address. Under
//! BottleShip's identity map the TRANSLATION is the identity, so all of that work exists to
//! answer one question: may this page be read? A byte per page answers it in a 1 MiB table
//! with no remap.
//!
//! Measured before writing any of it (RESULTS-2026-09-02.md): removing the check entirely
//! saves ~1.02 ns per non-stack access, which is 6.87% of a live window — four times the
//! ceiling of the stack-fastmem item that the roadmap listed first. This is the lever that
//! measurement chose.
//!
//! The map is a MIRROR of `tlb_data`, and the only thing that keeps a mirror honest is
//! having one writer: `cpu::set_tlb_entry` is the sole place `tlb_data` is assigned, and it
//! derives the byte in the same statement. `tools/validate-tlb-mirror.mjs` fails the build
//! if a second writer appears — a desynced permission byte is a missing page fault or a
//! silent read of the wrong page, with nothing to notice it.
//!
//! `PERM_IDENTITY` is the load-bearing bit. Nothing here assumes identity mapping: a page
//! whose physical base is not its linear base simply does not get the bit, and the
//! generated code falls through to the ordinary TLB path for it.

use crate::cpu::memory;

/// 4 GiB of linear space, one byte per 4 KiB page.
pub const PERM_MAP_PAGES: usize = 1 << 20;

#[allow(non_upper_case_globals)]
pub static mut perm_map: [u8; PERM_MAP_PAGES] = [0; PERM_MAP_PAGES];

/// A valid translation exists and the page is real memory (not MMIO).
pub const PERM_READABLE: u8 = 1 << 0;
/// ... and it is not read-only.
pub const PERM_WRITABLE: u8 = 1 << 1;
/// ... and it is reachable at CPL3.
pub const PERM_USER: u8 = 1 << 2;
/// The page's physical base equals its linear base, so no remap is needed.
pub const PERM_IDENTITY: u8 = 1 << 3;
/// Mirrors TLB_HAS_CODE. Reads do not care; a future write path must.
pub const PERM_HAS_CODE: u8 = 1 << 4;
/// READABLE & IDENTITY, precomputed so the replacement read path tests ONE bit. Positive
/// sense on purpose: the map starts zeroed, and zero must mean "take the slow path".
pub const PERM_FAST_READ_CPL0: u8 = 1 << 5;
/// READABLE & IDENTITY & USER.
pub const PERM_FAST_READ_CPL3: u8 = 1 << 6;

/// What a read at CPL3 requires. CPL0 drops the USER bit from the mask, exactly as
/// `gen_safe_read` drops TLB_NO_USER from its own mask.
pub const PERM_READ_MASK_CPL3: u8 = PERM_READABLE | PERM_IDENTITY | PERM_USER;
pub const PERM_READ_MASK_CPL0: u8 = PERM_READABLE | PERM_IDENTITY;

/// Derive the byte for `page` from the TLB entry that was just stored for it.
///
/// The entry is `(physical_base + mem8) ^ (page << 12) | info_bits`, so XORing the page
/// back out recovers `physical_base + mem8`; the page is identity-mapped exactly when that
/// equals `mem8 + (page << 12)`.
pub fn perm_byte_of(page: i32, entry: i32) -> u8 {
    use crate::cpu::cpu::{TLB_HAS_CODE, TLB_IN_MAPPED_RANGE, TLB_NO_USER, TLB_READONLY, TLB_VALID};
    if entry == 0 {
        return 0;
    }
    let mut b = 0u8;
    if entry & TLB_HAS_CODE != 0 {
        b |= PERM_HAS_CODE;
    }
    if entry & TLB_VALID == 0 || entry & TLB_IN_MAPPED_RANGE != 0 {
        // No translation, or MMIO — the slow path owns both.
        return b;
    }
    let base = ((entry as u32) & !0xFFF) ^ ((page as u32) << 12);
    let identity = base == (unsafe { memory::mem8 } as u32).wrapping_add((page as u32) << 12);
    b |= PERM_READABLE;
    if identity {
        b |= PERM_IDENTITY;
    }
    if entry & TLB_READONLY == 0 {
        b |= PERM_WRITABLE;
    }
    if entry & TLB_NO_USER == 0 {
        b |= PERM_USER;
    }
    if b & PERM_READ_MASK_CPL0 == PERM_READ_MASK_CPL0 {
        b |= PERM_FAST_READ_CPL0;
    }
    if b & PERM_READ_MASK_CPL3 == PERM_READ_MASK_CPL3 {
        b |= PERM_FAST_READ_CPL3;
    }
    b
}

/// Address of the map, for the constant the codegen bakes into a load — and for the
/// differential, which corrupts a byte on purpose to prove the mirror check can see it.
#[no_mangle]
pub fn perm_map_base() -> u32 {
    #[allow(static_mut_refs)]
    unsafe { perm_map.as_ptr() as u32 }
}

// ---------------------------------------------------------------------------
// The feature switch
// ---------------------------------------------------------------------------
//
// Default OFF, so the shipped read path is byte-identical to what it was: nothing below is
// emitted unless a mode is selected, which is what makes the OFF arm a real baseline rather
// than a differently-compiled approximation of one.
//
// Modes are mutually exclusive by construction:
//   0  off — the TLB chain alone.
//   1  probe — the byte is tested IN FRONT of the unchanged TLB chain (measured 0.9626).
//   2  replace — the byte is the ONLY inline check; the TLB is consulted only by the slow
//      helper (codegen.rs gen_perm_read). Exact because the byte is written in the same
//      statement as the TLB entry, never cached in a local, and zero means slow.
//   3  ABLATION, UNSOUND, measurement only — mode 2 with the slow arm's helper call removed:
//      a miss reads mem8 + addr raw (no translation, no #PF, no page-crossing split) and
//      bumps PERM_ABLATION_MISSES. It asks one question: does the CALL EDGE in the cold arm
//      cost what deleting the whole guard once measured, although mode 2's fewer ops did not?
//      The setter refuses it unless arm_perm_map_unsound_ablation(1) was called first.

pub const PERM_READS_OFF: u8 = 0;
pub const PERM_READS_PROBE: u8 = 1;
pub const PERM_READS_REPLACE: u8 = 2;
pub const PERM_READS_ABLATE_NO_CALL: u8 = 3;

static mut PERM_ABLATION_ARMED: bool = false;
/// Reads that took mode 3's call-free slow arm, i.e. would have been translated (or faulted)
/// by the real one. Nonzero means the arm's guest-visible behaviour may differ from mode 2.
#[allow(non_upper_case_globals)]
pub static mut PERM_ABLATION_MISSES: u64 = 0;
pub fn perm_ablation_misses_addr() -> u32 { unsafe { (&raw mut PERM_ABLATION_MISSES) as u32 } }

/// Must precede set_perm_map_reads(3). Disarming also drops an active mode 3 to OFF.
#[no_mangle]
pub fn arm_perm_map_unsound_ablation(on: u32) {
    unsafe {
        PERM_ABLATION_ARMED = on != 0;
        if !PERM_ABLATION_ARMED && PERM_MAP_READS == PERM_READS_ABLATE_NO_CALL {
            PERM_MAP_READS = PERM_READS_OFF;
        }
    }
}
#[no_mangle]
pub fn perm_ablation_misses() -> f64 { unsafe { PERM_ABLATION_MISSES as f64 } }

#[allow(non_upper_case_globals)]
pub static mut PERM_MAP_READS: u8 = PERM_READS_OFF;

pub fn perm_map_reads_enabled() -> bool { unsafe { PERM_MAP_READS == PERM_READS_PROBE } }
/// Modes 2 and 3 share the replacement shape; 3 differs only in its slow arm.
pub fn perm_map_reads_replace() -> bool {
    unsafe { PERM_MAP_READS == PERM_READS_REPLACE || PERM_MAP_READS == PERM_READS_ABLATE_NO_CALL }
}
pub fn perm_map_reads_ablate_no_call() -> bool { unsafe { PERM_MAP_READS == PERM_READS_ABLATE_NO_CALL } }

/// Unknown modes select OFF rather than the nearest valid one: an arm that asked for a mode
/// this build does not have must read back as the baseline, not as a different experiment.
#[no_mangle]
pub fn set_perm_map_reads(mode: u32) {
    unsafe {
        PERM_MAP_READS = if mode <= PERM_READS_REPLACE as u32 {
            mode as u8
        }
        else if mode == PERM_READS_ABLATE_NO_CALL as u32 && PERM_ABLATION_ARMED {
            PERM_READS_ABLATE_NO_CALL
        }
        else {
            PERM_READS_OFF
        }
    }
}

#[no_mangle]
pub fn get_perm_map_reads() -> u32 { unsafe { PERM_MAP_READS as u32 } }

/// Rebuild the whole map from `tlb_data`. The map is maintained incrementally by
/// `set_tlb_entry`; this exists for the differential test, which needs to assert that the
/// incremental result equals the recomputed one — a mirror nobody ever checks against its
/// source is the failure mode this file is most exposed to.
#[no_mangle]
pub fn perm_map_rebuild_and_diff() -> u32 {
    use crate::cpu::cpu::tlb_data;
    let mut mismatches = 0u32;
    for page in 0..PERM_MAP_PAGES {
        let entry = unsafe { tlb_data[page] };
        let want = perm_byte_of(page as i32, entry);
        if unsafe { perm_map[page] } != want {
            mismatches += 1;
        }
    }
    mismatches
}

#[no_mangle]
pub fn perm_map_byte(page: u32) -> u32 {
    if (page as usize) < PERM_MAP_PAGES { unsafe { perm_map[page as usize] as u32 } } else { 0 }
}
