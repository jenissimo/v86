//! SIMD bodies for the repeated string instructions, over one already-translated page chunk.
//!
//! The caller owns everything that makes a REP restartable: fault delivery, the direction
//! flag, the ECX/ESI/EDI updates derived from the completed count, the per-page re-entry,
//! and the final `cmp8/16/32` that sets EFLAGS. These functions only answer "how many
//! elements completed, and which pair did the comparison stop on", so the lazy-flag source
//! stays whatever the existing instruction makes it.
//!
//! `Movs` and byte-sized `Stos` already run as bulk memcpy/memset in string.rs and are
//! deliberately untouched, which also makes them the controls when measuring this.

use std::ptr::{addr_of, read_unaligned, write_bytes, write_unaligned};

use std::arch::wasm32::*;

use crate::cpu::{cpu, global_pointers, memory};

/// Host kill switch: restores the per-element loops without a rebuild.
static mut ENABLED: bool = true;

/// cmps chunks, scas chunks, stosw chunks, stosd chunks, rejected.
///
/// These count BODY invocations, not entries from the aligned gate: `unaligned_memory`
/// reaches the same functions by its own route and lands in the same counters. Read them
/// against the unaligned ledger to attribute a chunk to a route.
static mut STATS: [u32; 5] = [0; 5];

const STAT_CMPS: usize = 0;
const STAT_SCAS: usize = 1;
const STAT_STOSW: usize = 2;
const STAT_STOSD: usize = 3;
const STAT_REJECTED: usize = 4;

#[no_mangle]
pub extern "C" fn get_rep_memory_abi() -> u32 { 1 }

#[no_mangle]
pub unsafe extern "C" fn set_rep_memory_enabled(enabled: u32) { ENABLED = enabled != 0; }

/// The unaligned entry reaches these bodies by its own route, so switching the kernels
/// off has to reach it too; one flag, not two that can disagree.
#[inline]
pub(super) unsafe fn is_enabled() -> bool { ENABLED }

#[no_mangle]
pub unsafe extern "C" fn get_rep_memory_stats_ptr() -> u32 { addr_of!(STATS) as u32 }

#[inline]
unsafe fn record(index: usize) { STATS[index] = STATS[index].wrapping_add(1); }

/// Lowest address of the chunk, or None if it is not a plain single-page span.
///
/// These are PHYSICAL addresses that already passed ordinary CPU translation, so unlike
/// the HLE bulk kernels remapped RAM is fine here and permissions are already settled.
/// What must still hold is that the vector load stays inside the one page the caller
/// validated: never widen a chunk, never touch the following page.
#[inline]
unsafe fn span<const SIZE: u32>(start: u32, count: u32, backwards: bool) -> Option<u32> {
    let bytes = count.checked_mul(SIZE)?;
    let low = if backwards { start.checked_sub(bytes.checked_sub(SIZE)?)? } else { start };
    let end = low.checked_add(bytes.checked_sub(1)?)?;
    if low >> 12 != end >> 12
        || end >= *global_pointers::memory_size
        || memory::in_mapped_range(low)
        || memory::in_mapped_range(end)
    {
        return None;
    }
    Some(low)
}

#[inline]
unsafe fn load<const SIZE: u32>(start: u32, index: u32, backwards: bool) -> i32 {
    let addr = if backwards { start - index * SIZE } else { start + index * SIZE };
    let p = memory::mem8.add(addr as usize);
    match SIZE {
        1 => *p as i32,
        2 => read_unaligned(p.cast::<u16>()) as i32,
        _ => read_unaligned(p.cast::<i32>()),
    }
}

#[inline]
fn splat<const SIZE: u32>(value: i32) -> v128 {
    match SIZE {
        1 => i8x16_splat(value as i8),
        2 => i16x8_splat(value as i16),
        _ => i32x4_splat(value),
    }
}

#[inline]
fn equal_mask<const SIZE: u32>(a: v128, b: v128) -> u32 {
    match SIZE {
        1 => i8x16_bitmask(i8x16_eq(a, b)) as u32,
        2 => i16x8_bitmask(i16x8_eq(a, b)) as u32,
        _ => i32x4_bitmask(i32x4_eq(a, b)) as u32,
    }
}

/// CMPS (`SCAN == false`) and SCAS (`SCAN == true`).
///
/// Returns (completed elements, left operand, right operand) for the element the scan
/// stopped on, matching what the per-element loop would have handed to `cmp*`. Returning
/// operands rather than flags is what keeps EFLAGS a function of the existing instruction.
#[inline(never)]
pub unsafe fn compare<const SIZE: u32, const SCAN: bool>(
    src: u32,
    dst: u32,
    count: u32,
    backwards: bool,
    while_equal: bool,
    value: i32,
) -> Option<(u32, i32, i32)> {
    let lanes = 16 / SIZE;
    if !ENABLED || count < lanes {
        return None;
    }
    if span::<SIZE>(dst, count, backwards).is_none()
        || (!SCAN && span::<SIZE>(src, count, backwards).is_none())
    {
        record(STAT_REJECTED);
        return None;
    }
    let mask = match SIZE { 1 => 0xff, 2 => 0xffff, _ => -1 };
    let constant = value & mask;
    let left = if SCAN { constant } else { load::<SIZE>(src, 0, backwards) };
    let right = load::<SIZE>(dst, 0, backwards);
    record(if SCAN { STAT_SCAS } else { STAT_CMPS });
    // Settle the first element scalar-side: a REP that stops immediately is common and
    // would otherwise pay a vector load and a mask extraction to learn the same thing.
    if (left == right) != while_equal {
        return Some((1, left, right));
    }

    let mut i = 1;
    let all = (1u32 << lanes) - 1;
    while i + lanes <= count {
        // Backwards runs descend, so the vector covering elements i..i+lanes starts at the
        // HIGHEST of them; index by the far end and read the lane order back reversed.
        let index = if backwards { i + lanes - 1 } else { i };
        let delta = index * SIZE;
        let b = memory::mem8.add(if backwards { dst - delta } else { dst + delta } as usize);
        let a = if SCAN {
            splat::<SIZE>(constant)
        }
        else {
            let p = memory::mem8.add(if backwards { src - delta } else { src + delta } as usize);
            v128_load(p.cast())
        };
        let eq = equal_mask::<SIZE>(a, v128_load(b.cast()));
        let stop = if while_equal { !eq & all } else { eq };
        if stop != 0 {
            let lane = if backwards {
                lanes - 1 - (31 - stop.leading_zeros())
            }
            else {
                stop.trailing_zeros()
            };
            i += lane;
            let a = if SCAN { constant } else { load::<SIZE>(src, i, backwards) };
            return Some((i + 1, a, load::<SIZE>(dst, i, backwards)));
        }
        i += lanes;
    }
    while i < count {
        let a = if SCAN { constant } else { load::<SIZE>(src, i, backwards) };
        let b = load::<SIZE>(dst, i, backwards);
        i += 1;
        if (a == b) != while_equal {
            return Some((i, a, b));
        }
    }
    Some((
        count,
        if SCAN { constant } else { load::<SIZE>(src, count - 1, backwards) },
        load::<SIZE>(dst, count - 1, backwards),
    ))
}

/// Word and dword STOS. Byte STOS already goes through `memset_no_mmap_or_dirty_check`.
///
/// The caller has already dirtied the destination page, so no invalidation is owed here.
/// An armed write watch keeps the scalar stores: it observes each element individually
/// and a single bulk store would skip every one of those callbacks.
#[inline(never)]
pub unsafe fn fill<const SIZE: u32>(dst: u32, count: u32, backwards: bool, value: i32) -> bool {
    if !ENABLED || count < 64 / SIZE {
        return false;
    }
    if cpu::DBG_WRITE_WATCH != 0 {
        record(STAT_REJECTED);
        return false;
    }
    let Some(low) = span::<SIZE>(dst, count, backwards) else {
        record(STAT_REJECTED);
        return false;
    };
    let p = memory::mem8.add(low as usize);
    let bytes = count * SIZE;
    // A value whose bytes are all equal is a plain memset regardless of element size.
    let repeated = (value as u8 as u32).wrapping_mul(0x01010101);
    if (SIZE == 2 && value as u16 == repeated as u16) || (SIZE == 4 && value as u32 == repeated) {
        write_bytes(p, value as u8, bytes as usize);
    }
    else {
        let v = splat::<SIZE>(value);
        let mut offset = 0;
        while offset + 16 <= bytes {
            v128_store(p.add(offset as usize).cast(), v);
            offset += 16;
        }
        while offset < bytes {
            if SIZE == 2 {
                write_unaligned(p.add(offset as usize).cast::<u16>(), value as u16);
            }
            else {
                write_unaligned(p.add(offset as usize).cast::<i32>(), value);
            }
            offset += SIZE;
        }
    }
    record(if SIZE == 2 { STAT_STOSW } else { STAT_STOSD });
    true
}
