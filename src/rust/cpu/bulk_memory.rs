//! Bulk copy/fill/compare/search over guest RAM for the CRT memory hypercalls.
//!
//! The scalar handlers walk guest memory one element at a time through the safe
//! accessors, which is correct everywhere but costs a translation per element.
//! These kernels do the whole span with one native copy/fill or a SIMD scan, and
//! are reachable only when a probe proves the span is plain resident RAM.
//!
//! Nothing here may fault, so the probe never populates a TLB entry: a miss means
//! "not eligible" and the caller keeps the scalar path. That is also why a miss
//! must be cheap and must never have written anything.

use std::ptr::{addr_of, copy, write_bytes};

use crate::cpu::{cpu, global_pointers, memory};
use cpu::{TLB_HAS_CODE, TLB_IN_MAPPED_RANGE, TLB_NO_USER, TLB_READONLY, TLB_VALID};

#[cfg(target_feature = "simd128")]
use std::arch::wasm32::*;

/// Host kill switch: lets a guest run entirely on the scalar handlers without a rebuild.
static mut ENABLED: bool = true;

/// copy, fill, compare, move, find, rejected. Every bail-out increments `REJECTED`, so
/// hits plus rejects account for every call the dispatcher handed us; a hit count alone
/// cannot show that the fast path did the same work as the slow one.
static mut STATS: [u32; 6] = [0; 6];

const STAT_COPY: usize = 0;
const STAT_FILL: usize = 1;
const STAT_COMPARE: usize = 2;
const STAT_MOVE: usize = 3;
const STAT_FIND: usize = 4;
const STAT_REJECTED: usize = 5;

/// Bumped when the kernels change shape. The TS side refuses to register the hypercall
/// ids unless this answers, so a stale v86.wasm keeps the JS fallbacks instead of
/// dispatching into a handler that is not there.
#[no_mangle]
pub extern "C" fn get_bulk_memory_abi() -> u32 { 1 }

#[no_mangle]
pub unsafe extern "C" fn set_bulk_memory_enabled(enabled: u32) { ENABLED = enabled != 0; }

#[no_mangle]
pub unsafe extern "C" fn get_bulk_memory_stats_ptr() -> u32 { addr_of!(STATS) as u32 }

#[inline]
unsafe fn count(index: usize) { STATS[index] = STATS[index].wrapping_add(1); }

/// True when every page of the span is resident, identity-mapped plain RAM that the
/// current privilege level may touch in the requested direction.
///
/// Reads `tlb_data` without populating it: a probe must neither fault nor set
/// accessed/dirty bits, and a translation walk here would do both. The accept mask is
/// strictly narrower than `translate_address`, and two rejections carry the safety
/// argument. `TLB_IN_MAPPED_RANGE` keeps us off device memory. `TLB_HAS_CODE` keeps a
/// bulk write away from any page the JIT has compiled, because a native store is
/// invisible to v86's guest-store-driven invalidation.
///
/// Do not re-body this from `perm_map`: that mirror carries no write-watch bit, so a
/// perm_map-derived write guard would silently blind an armed watch.
pub(super) unsafe fn resident_span(start: u32, len: u32, write: bool) -> bool {
    if !ENABLED || len == 0 || start < 0x10_0000 {
        return false;
    }
    let Some(end) = start.checked_add(len - 1) else { return false; };
    if end >= *global_pointers::memory_size {
        return false;
    }
    if write && cpu::DBG_WRITE_WATCH != 0 {
        return false;
    }

    let reject = TLB_IN_MAPPED_RANGE
        | if *global_pointers::cpl == 3 { TLB_NO_USER } else { 0 }
        | if write { TLB_READONLY | TLB_HAS_CODE } else { 0 };

    for page in (start >> 12)..=(end >> 12) {
        let entry = cpu::tlb_data[page as usize];
        if entry & (TLB_VALID | reject) != TLB_VALID {
            return false;
        }
        // Decode the entry back to its physical base the way translate_address does.
        // These kernels index `mem8` linearly, so anything but identity is not ours.
        let physical = ((entry & !0xfff) as u32 ^ (page << 12)).wrapping_sub(memory::mem8 as u32);
        if physical != page << 12 {
            return false;
        }
    }
    true
}

/// `overlapping` is the memmove/memcpy distinction. A C `memcpy` whose spans overlap is
/// undefined, and the scalar handler resolves it by copying forward; answering with a
/// backward-safe move instead would quietly change what a guest observes, so overlap is
/// refused on the memcpy path rather than silently repaired.
pub unsafe fn try_copy(dst: u32, src: u32, len: u32, overlapping: bool) -> bool {
    if !resident_span(src, len, false) || !resident_span(dst, len, true) {
        count(STAT_REJECTED);
        return false;
    }
    if !overlapping && dst < src + len && src < dst + len {
        count(STAT_REJECTED);
        return false;
    }
    copy(memory::mem8.add(src as usize), memory::mem8.add(dst as usize), len as usize);
    count(if overlapping { STAT_MOVE } else { STAT_COPY });
    true
}

pub unsafe fn try_fill(dst: u32, value: u8, len: u32) -> bool {
    if !resident_span(dst, len, true) {
        count(STAT_REJECTED);
        return false;
    }
    write_bytes(memory::mem8.add(dst as usize), value, len as usize);
    count(STAT_FILL);
    true
}

/// Validates one page chunk at a time so an early mismatch never probes a later page: a
/// `memcmp` that differs in its first byte must not care whether the tail is mapped.
pub unsafe fn try_compare(a: u32, b: u32, len: u32) -> Option<i32> {
    let mut offset = 0u32;
    while offset < len {
        let Some(left) = a.checked_add(offset) else { count(STAT_REJECTED); return None; };
        let Some(right) = b.checked_add(offset) else { count(STAT_REJECTED); return None; };
        let chunk = (len - offset).min(4096 - (left & 4095)).min(4096 - (right & 4095));
        if !resident_span(left, chunk, false) || !resident_span(right, chunk, false) {
            count(STAT_REJECTED);
            return None;
        }
        let p = memory::mem8.add(left as usize);
        let q = memory::mem8.add(right as usize);
        let mut i = 0usize;
        #[cfg(target_feature = "simd128")]
        while i + 16 <= chunk as usize {
            let mask = i8x16_bitmask(i8x16_ne(v128_load(p.add(i).cast()), v128_load(q.add(i).cast())));
            if mask != 0 {
                i += mask.trailing_zeros() as usize;
                count(STAT_COMPARE);
                return Some(*p.add(i) as i32 - *q.add(i) as i32);
            }
            i += 16;
        }
        while i < chunk as usize {
            let diff = *p.add(i) as i32 - *q.add(i) as i32;
            if diff != 0 {
                count(STAT_COMPARE);
                return Some(diff);
            }
            i += 1;
        }
        offset += chunk;
    }
    count(STAT_COMPARE);
    Some(0)
}

/// Returns the guest address of the first `byte`, or 0 for absent — the same NULL the
/// caller writes into EAX for a `memchr` miss.
pub unsafe fn try_find(src: u32, byte: u8, len: u32) -> Option<u32> {
    if !ENABLED {
        return None;
    }
    let mut offset = 0u32;
    while offset < len {
        let Some(start) = src.checked_add(offset) else { count(STAT_REJECTED); return None; };
        let chunk = (len - offset).min(4096 - (start & 4095));
        if !resident_span(start, chunk, false) {
            count(STAT_REJECTED);
            return None;
        }
        let p = memory::mem8.add(start as usize);
        let mut i = 0usize;
        #[cfg(target_feature = "simd128")]
        while i + 16 <= chunk as usize {
            let mask = i8x16_bitmask(i8x16_eq(v128_load(p.add(i).cast()), u8x16_splat(byte)));
            if mask != 0 {
                count(STAT_FIND);
                return Some(start + i as u32 + mask.trailing_zeros());
            }
            i += 16;
        }
        while i < chunk as usize {
            if *p.add(i) == byte {
                count(STAT_FIND);
                return Some(start + i as u32);
            }
            i += 1;
        }
        offset += chunk;
    }
    count(STAT_FIND);
    Some(0)
}
