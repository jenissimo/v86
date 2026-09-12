//! SIMD bodies for the CRT string leaves, over guest RAM proven resident page by page.
//!
//! The scalar handlers translate once per character. These scan a validated page chunk at
//! a time, so the cost of proving a span safe is paid per page rather than per byte.
//!
//! Semantics are the scalar handlers', not the C standard's, wherever the two differ: the
//! case fold is ASCII-only because ours is, and the scan stops at the same 0x100001-unit
//! cap. Declining (`None`) is always safe — the caller keeps its own loop.

use std::ptr::{addr_of, copy_nonoverlapping, read_unaligned};

use crate::cpu::{bulk_memory::resident_span, memory};

#[cfg(target_feature = "simd128")]
use std::arch::wasm32::*;

/// Units scanned before giving up, matching the scalar handlers' `i > 0x100000` break.
const LIMIT: u32 = 0x100001;

/// Host kill switch: restores the per-character loops without a rebuild.
static mut ENABLED: bool = true;

/// strlen, wcslen, strcmp, stricmp, wcsicmp, strchr, strrchr, wcschr, strcpy, wcscpy, declined.
static mut STATS: [u32; 11] = [0; 11];

const STAT_DECLINED: usize = 10;

#[no_mangle]
pub extern "C" fn get_string_memory_abi() -> u32 { 1 }

#[no_mangle]
pub unsafe extern "C" fn set_string_memory_enabled(enabled: u32) { ENABLED = enabled != 0; }

#[no_mangle]
pub unsafe extern "C" fn get_string_memory_stats_ptr() -> u32 { addr_of!(STATS) as u32 }

/// Counts the answer and the decline on the same call, so hits plus declines account for
/// every request and neither number can drift from the other.
#[inline]
unsafe fn record<T>(value: Option<T>, hit: usize) -> Option<T> {
    let index = if value.is_some() { hit } else { STAT_DECLINED };
    STATS[index] = STATS[index].wrapping_add(1);
    value
}

#[inline]
fn step<const WIDE: bool>() -> u32 { if WIDE { 2 } else { 1 } }

#[inline]
unsafe fn unit<const WIDE: bool>(p: *const u8) -> u32 {
    if WIDE { u16::from_le(read_unaligned(p.cast())) as u32 } else { *p as u32 }
}

/// ASCII-only, matching the scalar handlers. A real CRT folds by codepage; that divergence
/// predates this file and must not be narrowed here alone, or the two paths disagree.
#[inline]
fn fold(value: u32) -> u32 {
    if (b'A' as u32..=b'Z' as u32).contains(&value) { value + 32 } else { value }
}

/// Units addressable from `p` without leaving its page.
///
/// Never zero: an odd-addressed wchar straddles the page end, and yielding one unit lets
/// `resident_span` validate exactly its two bytes across both pages rather than declining.
#[inline]
fn page_units<const WIDE: bool>(p: u32) -> u32 {
    ((4096 - (p & 4095)) / step::<WIDE>()).max(1)
}

#[inline]
unsafe fn checked_ptr(start: u32, bytes: u32, write: bool) -> Option<*mut u8> {
    if resident_span(start, bytes, write) { Some(memory::mem8.add(start as usize)) } else { None }
}

#[cfg(target_feature = "simd128")]
#[inline]
fn equal_mask<const WIDE: bool>(a: v128, b: v128) -> u32 {
    if WIDE { i16x8_bitmask(i16x8_eq(a, b)) as u32 } else { i8x16_bitmask(i8x16_eq(a, b)) as u32 }
}

#[cfg(target_feature = "simd128")]
#[inline]
fn splat<const WIDE: bool>(value: u32) -> v128 {
    if WIDE { u16x8_splat(value as u16) } else { u8x16_splat(value as u8) }
}

#[cfg(target_feature = "simd128")]
#[inline]
fn fold_vector<const WIDE: bool>(v: v128) -> v128 {
    let upper = if WIDE {
        v128_and(u16x8_ge(v, u16x8_splat(65)), u16x8_le(v, u16x8_splat(90)))
    }
    else {
        v128_and(u8x16_ge(v, u8x16_splat(65)), u8x16_le(v, u8x16_splat(90)))
    };
    v128_or(v, v128_and(upper, splat::<WIDE>(32)))
}

/// Address of `target`, or of the terminator when scanning for NUL; 0 when absent.
///
/// A NUL masks every later match in its vector: the string ends there, so a byte beyond it
/// is not part of the string even though the vector load already read it.
unsafe fn scan<const WIDE: bool, const REVERSE: bool>(src: u32, target: u32) -> Option<u32> {
    if !ENABLED {
        return None;
    }
    let stride = step::<WIDE>();
    let mut offset = 0u32;
    let mut last = 0;
    while offset < LIMIT {
        let start = src.checked_add(offset.checked_mul(stride)?)?;
        let count = page_units::<WIDE>(start).min(LIMIT - offset);
        let p = checked_ptr(start, count * stride, false)?;
        let mut i = 0u32;
        // An empty string or a hit on the first unit is common enough to be worth settling
        // before building any mask.
        let first = unit::<WIDE>(p);
        if first == target {
            last = start;
            if !REVERSE {
                return Some(last);
            }
        }
        if first == 0 {
            return Some(last);
        }
        i += 1;
        #[cfg(target_feature = "simd128")]
        while (count - i) * stride >= 16 {
            let v = v128_load(p.add((i * stride) as usize).cast());
            let zeros = equal_mask::<WIDE>(v, splat::<WIDE>(0));
            let mut matches = equal_mask::<WIDE>(v, splat::<WIDE>(target));
            if REVERSE {
                if zeros != 0 {
                    matches &= (1u32 << (zeros.trailing_zeros() + 1)) - 1;
                }
                if matches != 0 {
                    last = start + (i + 31 - matches.leading_zeros()) * stride;
                }
                if zeros != 0 {
                    return Some(last);
                }
            }
            else {
                let stop = matches | zeros;
                if stop != 0 {
                    let lane = stop.trailing_zeros();
                    return Some(if matches & (1 << lane) != 0 {
                        start + (i + lane) * stride
                    }
                    else {
                        0
                    });
                }
            }
            i += 16 / stride;
        }
        while i < count {
            let c = unit::<WIDE>(p.add((i * stride) as usize));
            if c == target {
                last = start + i * stride;
                if !REVERSE {
                    return Some(last);
                }
            }
            if c == 0 {
                return Some(last);
            }
            i += 1;
        }
        offset += count;
    }
    // Past the cap: decline and let the scalar handler apply its own give-up behaviour.
    None
}

pub unsafe fn try_length<const WIDE: bool>(src: u32) -> Option<u32> {
    record(
        scan::<WIDE, false>(src, 0).map(|end| (end - src) / step::<WIDE>()),
        if WIDE { 1 } else { 0 },
    )
}

pub unsafe fn try_find<const WIDE: bool, const REVERSE: bool>(src: u32, target: u32) -> Option<u32> {
    record(
        scan::<WIDE, REVERSE>(src, target),
        if WIDE { 7 } else if REVERSE { 6 } else { 5 },
    )
}

unsafe fn compare<const WIDE: bool, const FOLD: bool>(a: u32, b: u32) -> Option<i32> {
    if !ENABLED {
        return None;
    }
    let stride = step::<WIDE>();
    let mut offset = 0u32;
    while offset < LIMIT {
        let left = a.checked_add(offset.checked_mul(stride)?)?;
        let right = b.checked_add(offset.checked_mul(stride)?)?;
        // Two operands, two page edges: the chunk is the shorter remainder.
        let count = page_units::<WIDE>(left).min(page_units::<WIDE>(right)).min(LIMIT - offset);
        let p = checked_ptr(left, count * stride, false)?;
        let q = checked_ptr(right, count * stride, false)?;
        let mut i = 0u32;
        let first = unit::<WIDE>(p);
        let second = unit::<WIDE>(q);
        let diff = if FOLD {
            fold(first) as i32 - fold(second) as i32
        }
        else {
            first as i32 - second as i32
        };
        if diff != 0 || first == 0 {
            return Some(diff);
        }
        i += 1;
        #[cfg(target_feature = "simd128")]
        while (count - i) * stride >= 16 {
            let x = v128_load(p.add((i * stride) as usize).cast());
            let y = v128_load(q.add((i * stride) as usize).cast());
            let zeros = equal_mask::<WIDE>(x, splat::<WIDE>(0));
            let eq = if FOLD {
                equal_mask::<WIDE>(fold_vector::<WIDE>(x), fold_vector::<WIDE>(y))
            }
            else {
                equal_mask::<WIDE>(x, y)
            };
            let stop = ((!eq) & if WIDE { 0xff } else { 0xffff }) | zeros;
            if stop != 0 {
                let at = ((i + stop.trailing_zeros()) * stride) as usize;
                let x = unit::<WIDE>(p.add(at));
                let y = unit::<WIDE>(q.add(at));
                return Some(if FOLD {
                    fold(x) as i32 - fold(y) as i32
                }
                else {
                    x as i32 - y as i32
                });
            }
            i += 16 / stride;
        }
        while i < count {
            let x = unit::<WIDE>(p.add((i * stride) as usize));
            let y = unit::<WIDE>(q.add((i * stride) as usize));
            let diff = if FOLD {
                fold(x) as i32 - fold(y) as i32
            }
            else {
                x as i32 - y as i32
            };
            if diff != 0 || x == 0 {
                return Some(diff);
            }
            i += 1;
        }
        offset += count;
    }
    None
}

pub unsafe fn try_compare<const WIDE: bool, const FOLD: bool>(a: u32, b: u32) -> Option<i32> {
    record(compare::<WIDE, FOLD>(a, b), if WIDE { 4 } else if FOLD { 3 } else { 2 })
}

unsafe fn copy_string<const WIDE: bool>(dst: u32, src: u32) -> Option<()> {
    let end = scan::<WIDE, false>(src, 0)?;
    let bytes = (end - src).checked_add(step::<WIDE>())?;
    let dst_end = dst.checked_add(bytes)?;
    let src_end = src.checked_add(bytes)?;
    // Overlapping strcpy is undefined in C and resolved by the scalar handler copying
    // forward; a bulk copy would answer differently, so decline instead of redefining it.
    if dst < src_end && src < dst_end {
        return None;
    }
    let source = checked_ptr(src, bytes, false)?;
    // The write probe rejects a page the JIT has compiled, which is what keeps a native
    // store from going unseen by v86's guest-store-driven invalidation.
    let destination = checked_ptr(dst, bytes, true)?;
    copy_nonoverlapping(source, destination, bytes as usize);
    Some(())
}

pub unsafe fn try_copy<const WIDE: bool>(dst: u32, src: u32) -> bool {
    record(copy_string::<WIDE>(dst, src), if WIDE { 9 } else { 8 }).is_some()
}
