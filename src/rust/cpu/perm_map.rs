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
// Default OFF, so the shipped read path is byte-identical to what it was: the wrapper is
// only emitted when this is on, which is what makes the OFF arm a real baseline rather
// than a differently-compiled approximation of one.

#[allow(non_upper_case_globals)]
pub static mut PERM_MAP_READS: bool = false;

pub fn perm_map_reads_enabled() -> bool { unsafe { PERM_MAP_READS } }

#[no_mangle]
pub fn set_perm_map_reads(enabled: u32) { unsafe { PERM_MAP_READS = enabled != 0 } }

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
