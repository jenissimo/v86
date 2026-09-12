//! Reaches the REP vector bodies when the operands are not naturally aligned.
//!
//! `string_instruction` only enters its fast path for `Movs` or when both operands are
//! size-aligned, so an unaligned CMPS/SCAS/STOS runs the fully scalar loop with a safe
//! accessor per element. Alignment is not something the vector bodies need, only
//! something that gate demands, so this entry does the translation itself and hands the
//! same page chunk to the same kernels.
//!
//! It owns nothing architectural: it reports how many elements completed and the pair the
//! comparison stopped on, and the caller commits registers and decides re-entry.

use std::ptr::addr_of;

use crate::cpu::{cpu, memory, rep_memory};
use crate::jit;
use crate::page::Page;
use crate::paging::OrPageFault;

/// Host kill switch, independent of the aligned path's.
static mut ENABLED: bool = true;

/// cmpsw, cmpsd, scasw, scasd, stosw, stosd, scalar bridge elements, rejected.
static mut STATS: [u32; 8] = [0; 8];

const STAT_BRIDGE: usize = 6;
const STAT_REJECTED: usize = 7;

#[no_mangle]
pub extern "C" fn get_unaligned_rep_abi() -> u32 { 1 }

#[no_mangle]
pub unsafe extern "C" fn set_unaligned_rep_enabled(enabled: u32) { ENABLED = enabled != 0; }

#[no_mangle]
pub unsafe extern "C" fn get_unaligned_rep_stats_ptr() -> u32 { addr_of!(STATS) as u32 }

#[inline]
unsafe fn record(index: usize) { STATS[index] = STATS[index].wrapping_add(1); }

/// Elements addressable from `address` without leaving its page, in the walk direction.
///
/// An element straddling the page end belongs to neither page's chunk: every access reads
/// or writes SIZE bytes upward from its own address, backwards runs included, so a start
/// within SIZE of the page end yields nothing and the caller bridges it scalar-side.
#[inline]
fn chunk<const SIZE: u32>(address: u32, backwards: bool) -> u32 {
    let offset = address & 4095;
    if offset > 4096 - SIZE {
        0
    }
    else if backwards {
        offset / SIZE + 1
    }
    else {
        (4096 - offset) / SIZE
    }
}

#[inline]
unsafe fn read<const SIZE: u32>(address: i32) -> OrPageFault<i32> {
    if SIZE == 2 { cpu::safe_read16(address) } else { cpu::safe_read32s(address) }
}

/// KIND: 0 = CMPS, 1 = SCAS, 2 = STOS. Word and dword only; byte operands are always aligned.
///
/// Returns None to decline, leaving the caller's own paths in charge. A page fault
/// propagates unchanged, so fault ordering stays the architectural source-then-destination.
#[inline(never)]
pub unsafe fn run<const SIZE: u32, const KIND: u32>(
    src: i32,
    dst: i32,
    count: u32,
    backwards: bool,
    while_equal: bool,
    value: i32,
) -> OrPageFault<Option<(u32, i32, i32)>> {
    if !ENABLED || !rep_memory::is_enabled() || count < 64 / SIZE {
        return Ok(None);
    }
    if KIND == 2 && cpu::DBG_WRITE_WATCH != 0 {
        record(STAT_REJECTED);
        return Ok(None);
    }

    let n = count.min(chunk::<SIZE>(dst as u32, backwards));
    let n = if KIND == 0 { n.min(chunk::<SIZE>(src as u32, backwards)) } else { n };
    if n < if KIND == 2 { 64 / SIZE } else { 16 / SIZE } {
        // One element bridges the page edge through the ordinary accessors, which carry
        // the source-before-destination fault order, the writer's two-page preflight and
        // its JIT invalidation. Doing it here rather than declining keeps the run moving.
        let left = if KIND == 0 {
            read::<SIZE>(src)?
        }
        else if SIZE == 2 {
            value & 0xffff
        }
        else {
            value
        };
        let right = if KIND == 2 {
            if SIZE == 2 { cpu::safe_write16(dst, value)?; } else { cpu::safe_write32(dst, value)?; }
            0
        }
        else {
            read::<SIZE>(dst)?
        };
        record(STAT_BRIDGE);
        return Ok(Some((1, left, right)));
    }

    // Translate source first: a CMPS that faults must fault on the source.
    let phys_src = if KIND == 0 {
        let p = cpu::translate_address_read(src)?;
        if memory::in_mapped_range(p) {
            record(STAT_REJECTED);
            return Ok(None);
        }
        p
    }
    else {
        0
    };
    let (phys_dst, skip_dirty) = if KIND == 2 {
        cpu::translate_address_write_and_can_skip_dirty(dst)?
    }
    else {
        (cpu::translate_address_read(dst)?, true)
    };
    if memory::in_mapped_range(phys_dst) {
        record(STAT_REJECTED);
        return Ok(None);
    }

    let result = if KIND == 2 {
        // The aligned path dirties before its loop; this route has to do it itself.
        if !skip_dirty {
            jit::jit_dirty_page(Page::page_of(phys_dst));
        }
        if rep_memory::fill::<SIZE>(phys_dst, n, backwards, value) { Some((n, 0, 0)) } else { None }
    }
    else if KIND == 0 {
        rep_memory::compare::<SIZE, false>(phys_src, phys_dst, n, backwards, while_equal, value)
    }
    else {
        rep_memory::compare::<SIZE, true>(0, phys_dst, n, backwards, while_equal, value)
    };
    record(if result.is_some() {
        KIND as usize * 2 + usize::from(SIZE == 4)
    }
    else {
        STAT_REJECTED
    });
    Ok(result)
}
