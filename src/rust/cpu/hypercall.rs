//! Hypercall page: shared data between WASM and JS for fast thunk dispatch
//! and preemption control.
//!
//! Layout of HYPERCALL_PAGE (81920 bytes):
//!   0x000: cycle_limit          u32 — writable replacement for LOOP_COUNTER
//!   0x004: (reserved)           u32
//!   0x008: hc_enabled           u32 — hypercall master switch
//!   0x00C: hc_call_count        u32 — total WASM-handled calls
//!   0x010: hc_tick_count        u32 — GetTickCount value (written by JS)
//!   0x014: hc_perf_counter_lo   u32 — QPC low word
//!   0x018: hc_perf_counter_hi   u32 — QPC high word
//!   0x01C: hc_perf_freq_lo      u32 — QPF low word (constant)
//!   0x020: hc_perf_freq_hi      u32 — QPF high word (constant)
//!   0x024: hc_last_error        u32 — current thread's LastError
//!   0x028: hc_teb_base          u32 — current thread's TEB guest address
//!   0x02C: hc_insn_at_time_update u32 — instruction_counter snapshot for QPC interpolation
//!   0x030: hc_mips_estimate     u32 — instructions per microsecond
//!   0x034: hc_current_thread_id u32 — current thread ID (for CS ownership)
//!   0x080: hc_cursor_x          i32 — mouse X position (written by JS)
//!   0x084: hc_cursor_y          i32 — mouse Y position (written by JS)
//!   0x088: hc_window_x          i32 — main window X offset (written by JS)
//!   0x08C: hc_window_y          i32 — main window Y offset (written by JS)
//!   0x090: hc_msg_queue_flag    u32 — 1 if message queue non-empty (written by JS)
//!   0x094: hc_peek_starvation_counter u32 — consecutive WASM-handled PeekMessage calls
//!   0x098: hc_peek_starvation_limit   u32 — max consecutive before JS fallthrough
//!   0x09C: hc_has_runnable_peers      u32 — 1 if other threads are READY/RUNNING (written by JS)
//!   0x0A0: hc_sleep_starvation_counter u32 — consecutive WASM-handled Sleep(0) calls with peers
//!   0x0A4: hc_sleep_starvation_limit   u32 — max consecutive before JS fallthrough
//!   0x0B0: hc_rand_seed         u32 — RNG seed for handle_rand
//!   0x100: hc_dispatch_table    [u8; 4096] — dispatch_table[functionId] = handler_id
//!   0x1100: hc_fls_allocated    [u8; 129] — FLS slot allocation bitmap (written by JS, slot 0 unused)
//!   0x1184: hc_fls_values       [u32; 129] — FLS slot values (written by JS)
//!   0x1400: hc_slab_base        u32 — heap arena slab start (0=disabled, written by JS)
//!   0x1404: hc_slab_end         u32 — heap arena slab end (exclusive)
//!   0x1408: hc_slab_bump        u32 — current bump pointer
//!   0x140C: hc_slab_generation  u32 — incremented on process reset
//!   0x1410: hc_slab_alloc_count u32 — stats: allocations from slab
//!   0x1414: hc_slab_free_count  u32 — stats: frees returned to slab
//!   0x1418: hc_slab_fallback_count u32 — stats: fallbacks to JS
//!   0x1420: hc_slab_freelist    [u32; 9] — per-bin free list heads (bins: 16..4096)
//!   0x1444: hc_slab_ctl_ptr     u32 — GUEST address of the slab control block, or 0.
//!          When nonzero the slab control fields (base/end/bump/gen/counts/freelist) live in
//!          GUEST RAM at this address (same relative layout as the 0x1400 fields rebased to 0),
//!          NOT in this page. Required because the inline x86 stubs can only address guest RAM
//!          (this page is a WASM static below guest RAM, unreachable from guest code).
//!          The 0x1400.. page fields are then vestigial (legacy mode).
//!   0x1448: hc_event_table      [u8; 2048] — mirrored kernel event state (see EVT_* flags)
//!   0x1C48: hc_event_starvation_counter u32 — consecutive WASM-handled SetEvent calls
//!   0x1C4C: hc_event_starvation_limit   u32 — max consecutive before JS fallthrough
//!   0x1C50: hc_mutex_mirror_ptr         u32 — guest addr of mutex mirror table (2048×u32)
//!     Mutex mirror word: MUX_VALID | MUX_HAS_WAITERS | MUX_ABANDONED | owner:16 | rec:8
//!   0x1C54: hc_eagl_token_cfg_ptr       u32 — guest addr of the EAGL token-dispatch
//!     config block for handler 132 (0 = disabled); layout in handle_eagl_token_dispatch
//!   0x2000: hc_handler_calls     [u32; 256] — per-handler_id count of calls SERVED in WASM
//!   0x2400: hc_handler_fallbacks [u32; 256] — per-handler_id count of calls that returned
//!     false and fell through to the JS thunk. calls+fallbacks = times the dispatch table
//!     routed that id; the ratio is what says whether a fast path is actually fast.
//!     Both saturate at u32::MAX rather than wrapping, so a reading can never look small
//!     because it lapped.
//!   0x2800: hc_thread_suspend    [u32; 64] — 32 slots of (thread handle, suspend count),
//!     republished by the JS scheduler whenever a count changes. Read-only here: it exists so
//!     handle_resume_thread can answer the NO-OP resume (count already 0) without a JS round
//!     trip. A handle absent from the table, or one with a nonzero count, falls through.
//!   0x4000: hc_dispatch_table_ext [u8; 61440] — ids 4096..65535. Kept separate so all
//!     established ABI offsets remain stable while synthetic HLE images can use the same
//!     WASM handlers as ordinary import stubs.

use std::ptr::{addr_of, addr_of_mut};

use crate::cpu::memory;
use crate::cpu::cpu::{
    read_reg32, safe_read32s, safe_write32, write_reg32, EAX, EDX, ESP,
};
use crate::cpu::hypercall_rtti::{rt_dynamic_cast, RtDynamicCastResult};
use crate::cpu::fpu::{fpu_get_st0, fpu_get_sti, fpu_pop, fpu_push, fpu_write_st};
use crate::cpu::global_pointers::{fpu_stack_ptr, instruction_counter};
use crate::softfloat::F80;

/// Dedicated page for hypercall shared data + preemption control.
/// Lives in WASM data section, not in CPU state area.
#[no_mangle]
pub static mut HYPERCALL_PAGE: [u8; 81920] = [0u8; 81920];

// Offset constants
const OFF_CYCLE_LIMIT: usize = 0x000;
const OFF_HC_ENABLED: usize = 0x008;
const OFF_HC_CALL_COUNT: usize = 0x00C;
const OFF_HC_TICK_COUNT: usize = 0x010;
const OFF_HC_PERF_COUNTER_LO: usize = 0x014;
const OFF_HC_PERF_COUNTER_HI: usize = 0x018;
const OFF_HC_PERF_FREQ_LO: usize = 0x01C;
const OFF_HC_PERF_FREQ_HI: usize = 0x020;
const OFF_HC_LAST_ERROR: usize = 0x024;
const OFF_HC_TEB_BASE: usize = 0x028;
const OFF_HC_INSN_AT_TIME_UPDATE: usize = 0x02C;
const OFF_HC_MIPS_ESTIMATE: usize = 0x030;
const OFF_HC_CURRENT_THREAD_ID: usize = 0x034;
// (pending wake buffer removed — CS wake now uses LockSemaphore events)
const OFF_HC_CURSOR_X: usize = 0x080;
const OFF_HC_CURSOR_Y: usize = 0x084;
const OFF_HC_WINDOW_X: usize = 0x088;
const OFF_HC_WINDOW_Y: usize = 0x08C;
const OFF_HC_MSG_QUEUE_FLAG: usize = 0x090;
const OFF_HC_PEEK_STARVATION_COUNTER: usize = 0x094;
const OFF_HC_PEEK_STARVATION_LIMIT: usize = 0x098;
const OFF_HC_HAS_RUNNABLE_PEERS: usize = 0x09C;
const OFF_HC_SLEEP_STARVATION_COUNTER: usize = 0x0A0;
const OFF_HC_SLEEP_STARVATION_LIMIT: usize = 0x0A4;
const OFF_HC_RAND_SEED: usize = 0x0B0;
/// Mouse-capture owner (GetCapture). Unlike the thread-id / cursor slots, 0 is a
/// LEGITIMATE answer ("no window holds capture") and is the common case, so this
/// slot carries no validity sentinel — JS publishes it unconditionally at init,
/// on every capture mutation and on buffer-change resync, which makes the page
/// authoritative at all times.
const OFF_HC_CAPTURE_HWND: usize = 0x0B4;
const OFF_HC_DISPATCH_TABLE: usize = 0x100;
const OFF_HC_DISPATCH_TABLE_EXT: usize = 0x4000;
const HC_DISPATCH_LIMIT: i32 = 0x10000;
const OFF_HC_FLS_ALLOCATED: usize = 0x1100;
const OFF_HC_FLS_VALUES: usize = 0x1184;
const HC_FLS_SLOT_COUNT: usize = 129;

// Arena slab control block (HeapAlloc/HeapFree fast path). The live control block lives in
// GUEST RAM at hc_slab_ctl_ptr and is accessed via SLAB_REL_* below; the legacy 0x1400-based
// page layout is documented in the header block above and mirrored on the JS side.
const OFF_HC_SLAB_CTL_PTR: usize = 0x1444; // guest addr of slab control block (0 = legacy page)
const OFF_HC_EVENT_TABLE: usize = 0x1448;

// Relative offsets WITHIN the slab control block (page fields rebased to 0). Used when the
// control block lives in guest RAM (hc_slab_ctl_ptr != 0).
const SLAB_REL_BASE: u32 = 0x00;
const SLAB_REL_END: u32 = 0x04;
const SLAB_REL_BUMP: u32 = 0x08;
const SLAB_REL_ALLOC_COUNT: u32 = 0x10;
const SLAB_REL_FREE_COUNT: u32 = 0x14;
const SLAB_REL_FALLBACK_COUNT: u32 = 0x18;
const SLAB_REL_FREELIST: u32 = 0x20; // 9 × u32

/// Read a u32 slab control field from the guest-RAM control block at `ctl + rel`.
#[inline]
unsafe fn slab_rd(ctl: u32, rel: u32) -> u32 {
    memory::read32_no_mmap_check(ctl.wrapping_add(rel)) as u32
}

/// Write a u32 slab control field into the guest-RAM control block at `ctl + rel`.
#[inline]
unsafe fn slab_wr(ctl: u32, rel: u32, value: u32) {
    memory::write32_no_mmap_or_dirty_check(ctl.wrapping_add(rel), value as i32);
}
const OFF_HC_EVENT_STARVATION_COUNTER: usize = 0x1C48;
const OFF_HC_EVENT_STARVATION_LIMIT: usize = 0x1C4C;
const OFF_HC_MUTEX_MIRROR_PTR: usize = 0x1C50;
/// Guest address of the EAGL token-dispatch config block (0 = handler 132
/// disabled). Written by JS (hle-lib libs/eagl) once the d3d9 WBUF ring +
/// setter shadow tables exist. Layout: see handle_eagl_token_dispatch.
pub(crate) const OFF_HC_EAGL_TOKEN_CFG_PTR: usize = 0x1C54;

/// Per-handler_id accounting. Indexed by the dispatch-table byte, so all 256 ids
/// (WinAPI/CRT tiers 1..=127 AND the engine inner-loop band 128..=255) are covered —
/// a guard miss in the inner-loop band is exactly as interesting as a CS contention.
const OFF_HC_HANDLER_CALLS: usize = 0x2000;
const OFF_HC_HANDLER_FALLBACKS: usize = 0x2400;
const OFF_HC_THREAD_SUSPEND: usize = 0x2800;
const HC_THREAD_SUSPEND_SLOTS: usize = 32;
const HC_HANDLER_SLOTS: usize = 256;
// The tables are the last thing in the page; if a future field pushes them past the end
// the writes would silently corrupt whatever static follows. Fail the build instead.
const _: () = assert!(OFF_HC_HANDLER_FALLBACKS + HC_HANDLER_SLOTS * 4 <= 16384);

/// Saturating +1 on a u32 counter in the page. Saturating, not wrapping: a wrapped
/// counter reads as a plausible small number, which is the failure mode where an
/// instrument lies instead of admitting it ran out of range.
#[inline(always)]
unsafe fn bump_counter(offset: usize) {
    let p = hp_mut().add(offset) as *mut u32;
    *p = (*p).saturating_add(1);
}

const KERNEL_HANDLE_BASE: u32 = 0x30000;
const EVENT_TABLE_SLOTS: u32 = 2048;
const EVT_VALID: u8 = 0x01;
const EVT_SIGNALED: u8 = 0x02;
const EVT_MANUAL: u8 = 0x04;
const EVT_HAS_WAITERS: u8 = 0x08;
const EVT_PENDING_WAKE: u8 = 0x10;

const MUX_VALID: u32 = 0x8000_0000;
const MUX_HAS_WAITERS: u32 = 0x4000_0000;
const MUX_ABANDONED: u32 = 0x2000_0000;
const MUX_OWNER_MASK: u32 = 0x0000_FFFF;
const MUX_REC_MASK: u32 = 0x00FF_0000;
const MUX_REC_SHIFT: u32 = 16;
const MUX_REC_MAX: u32 = 0xFF;
const WAIT_TIMEOUT: i32 = 0x102;

const SLAB_MAGIC: u32 = 0x534C4100; // "SLA\0" (BUSY) — low nibble reserved for bin index
// FREE marker ("SLF\0"). A block on the per-bin free list carries this in its header so a
// double-free is rejected (the BUSY-only validate fails) and getSlabSizeForPtr won't report a
// free-listed block as a live sized allocation. MUST mirror the inline stubs + TS SLAB_MAGIC_FREE.
const SLAB_MAGIC_FREE: u32 = 0x534C4600;
// Free-list link offset below the user pointer: [user-16..user-4) is the block's header
// zone, [user-4] the magic. MUST mirror the inline stubs (heap-slab-stubs.ts,
// crt-slab-stubs.ts) and the TS walkers in kernel32/memory.ts.
const SLAB_LINK: u32 = 8;
// Large-bin FIFO. NT5's front-end LIFO lookaside only covers allocations whose index
// (size >> 3) is < 128, i.e. sizes < 1024 bytes (heap.h HEAP_MAXIMUM_FREELISTS=128,
// HEAP_GRANULARITY=8). Blocks >= 1024 bypass the lookaside and land on the coalescing
// back-end, which does NOT hand the just-freed block back to the very next same-size
// allocation. Our slab was strict LIFO for every bin, so a just-freed 2 KiB object was
// re-handed and (on HEAP_ZERO_MEMORY) zeroed immediately — turning a benign app
// use-after-free (a freed widget still linked in a parent list, read once more before
// the parent forgets it — Worms Armageddon's results screen) into a NULL-vtable crash.
// So bins >= SLAB_LARGE_BIN_MIN use a FIFO free list (append at tail, pop at head): a
// freed large block is reused only after every other free block of its size, giving its
// bytes the same grace the real back-end does. Small bins stay LIFO, matching the
// lookaside. FREELIST_TAIL is per-bin and only meaningful for the large bins.
const SLAB_REL_FREELIST_TAIL: u32 = 0x48; // 9 × u32 — per-bin free-list tail (large bins)
const SLAB_LARGE_BIN_MIN: u32 = 6; // bins 6,7,8 = 1024/2048/4096 bytes
const HEAP_ZERO_MEMORY_FLAG: u32 = 0x08;
const BIN_SIZES: [u32; 9] = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096];

/// Helper: raw pointer to HYPERCALL_PAGE (avoids static_mut_refs warnings).
#[inline(always)]
pub(crate) unsafe fn hp_ptr() -> *const u8 {
    addr_of!(HYPERCALL_PAGE).cast::<u8>()
}

/// Helper: mutable raw pointer to HYPERCALL_PAGE.
#[inline(always)]
unsafe fn hp_mut() -> *mut u8 {
    addr_of_mut!(HYPERCALL_PAGE).cast::<u8>()
}

/// Returns the WASM-linear pointer to HYPERCALL_PAGE for JS to create typed views.
#[no_mangle]
pub fn get_hypercall_page_ptr() -> u32 {
    unsafe { hp_ptr() as u32 }
}

/// Interpolated guest-virtual time in MICROSECONDS — the exact same base + interpolation
/// the QPC hypercall (handle_qpc) serves to the guest: JS-written perf counter snapshot
/// plus instructions-retired-since-snapshot / mips. Used by read_tsc so RDTSC and
/// QPC/GetTickCount are derived from ONE clock (constant ratio 2^32 ticks : 1e6 µs) —
/// guest cross-clock calibration (UE1 GSecondsPerCycle et al.) then measures the true
/// ratio no matter how far virtual time lags wall during boot.
/// Returns None until the unified clock is live (page disabled or time fields unset).
#[inline(always)]
pub unsafe fn virtual_time_us() -> Option<u64> {
    let page = hp_ptr();
    if *(page.add(OFF_HC_ENABLED) as *const u32) == 0 {
        return None;
    }
    let mips_est = *(page.add(OFF_HC_MIPS_ESTIMATE) as *const u32);
    if mips_est == 0 {
        return None;
    }
    let base_lo = *(page.add(OFF_HC_PERF_COUNTER_LO) as *const u32);
    let base_hi = *(page.add(OFF_HC_PERF_COUNTER_HI) as *const u32);
    let insn_at_update = *(page.add(OFF_HC_INSN_AT_TIME_UPDATE) as *const u32);
    let delta_insn = (*instruction_counter).wrapping_sub(insn_at_update);
    Some((base_lo as u64 | (base_hi as u64) << 32) + (delta_insn / mips_est) as u64)
}

/// Read the cycle limit from shared page. Used by do_many_cycles_native().
#[inline(always)]
pub unsafe fn read_cycle_limit() -> u32 {
    let val = *(hp_ptr().add(OFF_CYCLE_LIMIT) as *const u32);
    if val == 0 {
        if *(hp_ptr().add(OFF_HC_ENABLED) as *const u32) == 0 {
            // Default: match original LOOP_COUNTER when JS hasn't initialized yet.
            100_003
        }
        else {
            // JS uses 0 as an urgent-exit request for async parks / scheduler wakeups.
            0
        }
    } else {
        val
    }
}

/// Async-park spin-loop address (virtual EIP of the JMP $ parking slot). Set once by
/// JS at boot via set_park_eip(); 0 = unknown (check disabled). The cycle loop exits
/// the slice when EIP lands EXACTLY here — the address is a parking slot, not code,
/// so honestly executing it burns the slice budget for nothing (see
/// do_many_cycles_native). Strict equality: SEH stubs at +2/+4/+0x200 must keep running.
static mut PARK_EIP: u32 = 0;

/// JS boot hook: tell the cycle loop where the async-park spin loop lives.
#[no_mangle]
pub unsafe fn set_park_eip(addr: u32) {
    PARK_EIP = addr;
}

/// True when `eip` is parked on the spin-loop base (exact match).
#[inline(always)]
pub unsafe fn eip_at_park(eip: u32) -> bool {
    PARK_EIP != 0 && eip == PARK_EIP
}

/// Try to dispatch a hypercall for the given function ID.
/// Returns true if handled in WASM, false to fall through to JS.
pub unsafe fn try_dispatch(function_id: i32) -> bool {
    let page = hp_ptr();

    // Check enabled flag
    if *(page.add(OFF_HC_ENABLED) as *const u32) == 0 {
        return false;
    }

    if function_id <= 0 || function_id >= HC_DISPATCH_LIMIT {
        return false;
    }

    let fid = function_id as usize;
    let entry = if fid < 4096 {
        OFF_HC_DISPATCH_TABLE + fid
    } else {
        OFF_HC_DISPATCH_TABLE_EXT + fid - 4096
    };
    let handler_id = *page.add(entry);
    if handler_id == 0 {
        return false;
    }

    // Guest accesses made by a handler are attributed to the hypercall, not left in the
    // UNATTRIBUTED bucket that names JIT-called instruction helpers.
    let ga_prev = crate::cpu::cpu::ga_enter(crate::cpu::cpu::GA_HYPERCALL);
    let handled = match handler_id {
        1 => handle_get_tick_count(),
        2 => handle_get_tick_count(), // GetTickCount64 → same 32-bit value
        3 => handle_qpc(),
        4 => handle_qpf(),
        5 => handle_get_last_error(),
        6 => handle_set_last_error(),
        7 => handle_interlocked_inc(),
        8 => handle_interlocked_dec(),
        9 => handle_interlocked_xchg(),
        10 => handle_interlocked_cmp_xchg(),
        11 => handle_enter_critical_section(),
        12 => handle_leave_critical_section(),
        13 => handle_is_iconic(),
        14 => handle_screen_to_client(),
        15 => handle_get_cursor_pos(),
        16 => handle_peek_message(),
        // Math/FPU hypercalls (Tier 2)
        17 => handle_ftol(),
        18 => handle_ci_sin(),
        19 => handle_ci_cos(),
        20 => handle_ci_tan(),
        21 => handle_ci_sqrt(),
        22 => handle_ci_log(),
        23 => handle_ci_exp(),
        24 => handle_ci_acos(),
        25 => handle_ci_asin(),
        26 => handle_ci_log10(),
        27 => handle_ci_atan2(),
        28 => handle_ci_fmod(),
        29 => handle_ci_pow(),
        30 => handle_cdecl_sin(),
        31 => handle_cdecl_cos(),
        32 => handle_cdecl_tan(),
        33 => handle_cdecl_sqrt(),
        34 => handle_cdecl_log(),
        35 => handle_cdecl_exp(),
        36 => handle_cdecl_acos(),
        37 => handle_cdecl_asin(),
        38 => handle_cdecl_log10(),
        39 => handle_cdecl_atan(),
        40 => handle_cdecl_fabs(),
        41 => handle_cdecl_atan2(),
        42 => handle_cdecl_fmod(),
        43 => handle_cdecl_pow(),
        44 => handle_cdecl_ceil(),
        45 => handle_cdecl_floor(),
        // String/memory hypercalls (Tier 3)
        51 => handle_wcslen(),
        52 => handle_wcscpy(),
        53 => handle_wcscat(),
        54 => handle_wcsicmp(),
        55 => handle_wcschr(),
        56 => handle_memcpy(),
        57 => handle_memset(),
        58 => handle_strlen(),
        59 => handle_strcmp(),
        60 => handle_strcpy(),
        61 => handle_stricmp(),
        62 => handle_memcmp(),
        // Scheduler hypercalls (Tier 4)
        63 => handle_sleep(),
        64 => handle_tls_get_value(),
        65 => handle_rand(),
        66 => handle_wcsstr(),
        67 => handle_wcsnicmp(),
        68 => handle_wcsncpy(),
        69 => handle_fls_get_value(),
        // Heap arena hypercalls (Tier 5)
        70 => handle_heap_alloc(),
        71 => handle_heap_free(),
        72 => handle_set_event(),
        // Tier 1/3 additions: current-thread-id (pure page read) + narrow ANSI string leaves.
        73 => handle_get_current_thread_id(),
        74 => handle_strnicmp(),
        75 => handle_strstr(),
        76 => handle_atoi(),
        77 => handle_rt_dynamic_cast(),
        // Page-probe pointer validation (IsBadReadPtr/IsBadWritePtr share one probe).
        78 => handle_is_bad_ptr(),
        79 => handle_is_bad_ptr(),
        // Uncontended mutex fast paths (see mutex mirror table @ OFF_HC_MUTEX_MIRROR_PTR).
        80 => handle_release_mutex(),
        81 => handle_wait_for_single_object(),
        82 => handle_get_capture(),
        83 => handle_resume_thread(),

        // ── Handler-id band 128..=255: Guarded Inner-Loop HLE engine kernels,
        //    kept in a distinct range from the
        //    conventional WinAPI/CRT tiers (1..=127 above, which are generic
        //    Windows/CRT semantics) so the category is obvious from the
        //    dispatch byte alone. Engine-specific handlers do NOT live in this
        //    file — the whole band delegates to the engine module(s); a second
        //    engine graduates this into a dedicated band router. ──
        128..=255 => {
            let prev = crate::cpu::cpu::ga_enter(crate::cpu::cpu::GA_EAGL);
            let r = super::hypercall_eagl::dispatch_inner_loop(handler_id);
            crate::cpu::cpu::ga_leave(prev);
            r
        },
        _ => false,
    };
    crate::cpu::cpu::ga_leave(ga_prev);

    // Per-handler accounting. handler_id is a u8 and the tables are 256 slots, so the
    // index is in range by construction — no bound to silently truncate against.
    let slot = handler_id as usize * 4;
    if handled {
        bump_counter(OFF_HC_HANDLER_CALLS + slot);
        let count_ptr = hp_mut().add(OFF_HC_CALL_COUNT) as *mut u32;
        *count_ptr = (*count_ptr).wrapping_add(1);
    }
    else {
        bump_counter(OFF_HC_HANDLER_FALLBACKS + slot);
    }
    handled
}

// ---------------------------------------------------------------------------
// Handler implementations
// ---------------------------------------------------------------------------

/// GetTickCount / GetTickCount64 — interpolated tick from shared page
///
/// Without interpolation, all GetTickCount calls within a single tick
/// (up to 16ms in single-thread mode) return the SAME value, causing
/// spin-waits to burn through entire quantums and timing code to see
/// 0ms deltas. We interpolate using the instruction counter delta and
/// MIPS estimate, same approach as QPC.
unsafe fn handle_get_tick_count() -> bool {
    let page = hp_ptr();
    let base_tick = *(page.add(OFF_HC_TICK_COUNT) as *const u32);
    let insn_at_update = *(page.add(OFF_HC_INSN_AT_TIME_UPDATE) as *const u32);
    let mips_est = *(page.add(OFF_HC_MIPS_ESTIMATE) as *const u32);

    // Interpolate: delta_ms = instructions_since_update / (insns_per_us * 1000)
    // mips_est = instructions/microsecond, so mips_est * 1000 = instructions/millisecond
    let delta_ms = if mips_est > 0 {
        let delta_insn = (*instruction_counter).wrapping_sub(insn_at_update);
        delta_insn / (mips_est * 1000)
    } else {
        0
    };

    write_reg32(EAX, base_tick.wrapping_add(delta_ms) as i32);
    true
}

/// QueryPerformanceCounter — interpolates between JS updates using instruction_counter
unsafe fn handle_qpc() -> bool {
    let esp = read_reg32(ESP);
    let ptr = match safe_read32s(esp + 4) {
        Ok(v) => v as u32,
        Err(_) => return false,
    };
    if ptr == 0 {
        write_reg32(EAX, 0);
        return true;
    }

    let page = hp_ptr();
    let base_lo = *(page.add(OFF_HC_PERF_COUNTER_LO) as *const u32);
    let base_hi = *(page.add(OFF_HC_PERF_COUNTER_HI) as *const u32);
    let insn_at_update = *(page.add(OFF_HC_INSN_AT_TIME_UPDATE) as *const u32);
    let mips_est = *(page.add(OFF_HC_MIPS_ESTIMATE) as *const u32);

    // Interpolate: delta_micros ≈ (insn_now - insn_at_update) / mips_estimate
    let delta_insn = (*instruction_counter).wrapping_sub(insn_at_update);
    let delta_micros = if mips_est > 0 {
        delta_insn / mips_est
    } else {
        0
    };

    let qpc = (base_lo as u64 | (base_hi as u64) << 32) + delta_micros as u64;

    if safe_write32(ptr as i32, qpc as i32).is_err() {
        return false;
    }
    if safe_write32((ptr + 4) as i32, (qpc >> 32) as i32).is_err() {
        return false;
    }

    write_reg32(EAX, 1);
    true
}

/// QueryPerformanceFrequency — read constant from shared page
unsafe fn handle_qpf() -> bool {
    let esp = read_reg32(ESP);
    let ptr = match safe_read32s(esp + 4) {
        Ok(v) => v as u32,
        Err(_) => return false,
    };
    if ptr == 0 {
        write_reg32(EAX, 0);
        return true;
    }

    let page = hp_ptr();
    let freq_lo = *(page.add(OFF_HC_PERF_FREQ_LO) as *const u32);
    let freq_hi = *(page.add(OFF_HC_PERF_FREQ_HI) as *const u32);

    if safe_write32(ptr as i32, freq_lo as i32).is_err() {
        return false;
    }
    if safe_write32((ptr + 4) as i32, freq_hi as i32).is_err() {
        return false;
    }

    write_reg32(EAX, 1);
    true
}

/// GetLastError — read from shared page
unsafe fn handle_get_last_error() -> bool {
    let err = *(hp_ptr().add(OFF_HC_LAST_ERROR) as *const u32);
    write_reg32(EAX, err as i32);
    true
}

/// GetCurrentThreadId — stdcall(0). Returns the current thread ID from the shared page,
/// which JS republishes on every context switch (syncThreadData → OFF_HC_CURRENT_THREAD_ID).
/// Games flood this from the CRT's per-thread-data lookup (_getptd) inside malloc/strtok/etc.;
/// serving it here removes the JS round-trip entirely. tid==0 means JS hasn't populated thread
/// info yet → fall through so the JS thunk answers (and seeds the page on the next switch).
unsafe fn handle_get_current_thread_id() -> bool {
    let tid = *(hp_ptr().add(OFF_HC_CURRENT_THREAD_ID) as *const u32);
    if tid == 0 { return false; }
    write_reg32(EAX, tid as i32);
    true
}

/// ResumeThread(hThread) — stdcall(1). Serves ONLY the case where the target is not
/// suspended: Win32 returns the PREVIOUS count, which is then 0, and nothing changes — no
/// state to flip, no run queue to touch, no switch to request. That case is the whole storm:
/// Discworld Noir's main loop calls ResumeThread once per iteration (~270k/s, ~11k per frame)
/// against a worker that is usually already running, and each one crossed into JS for a
/// no-op. A real resume (count > 0), an unknown handle, or a full slot table all return false
/// so the JS scheduler answers — every state change stays in the one implementation that owns
/// the thread state machine.
unsafe fn handle_resume_thread() -> bool {
    let esp = read_reg32(ESP);
    let handle = match safe_read32s(esp + 4) {
        Ok(v) => v as u32,
        Err(_) => return false,
    };
    if handle == 0 {
        return false;
    }
    let table = hp_ptr().add(OFF_HC_THREAD_SUSPEND) as *const u32;
    for i in 0..HC_THREAD_SUSPEND_SLOTS {
        let slot_handle = *table.add(i * 2);
        if slot_handle == 0 {
            break; // slots are packed from the front
        }
        if slot_handle != handle {
            continue;
        }
        if *table.add(i * 2 + 1) != 0 {
            return false; // a real resume — the JS scheduler owns the transition
        }
        write_reg32(EAX, 0); // previous suspend count
        return true;
    }
    false
}

/// msvcrt/crtdll rand() — MSVCRT LCG, fully in WASM (no JS round-trip). UE1 games flood
/// rand() hundreds of times per frame; routing it here removes that thunk-dispatch cost.
/// Seed lives in the shared page (OFF_HC_RAND_SEED); srand() syncs JS→page via updateRandSeed,
/// and since every rand() goes through this handler the JS-side seed is never read, so the
/// sequence stays consistent. Returns (seed >> 16) & 0x7fff in EAX, matching the JS impl.
unsafe fn handle_rand() -> bool {
    let p = hp_ptr().add(OFF_HC_RAND_SEED) as *mut u32;
    let seed = (*p).wrapping_mul(214013).wrapping_add(2531011);
    *p = seed;
    write_reg32(EAX, ((seed >> 16) & 0x7fff) as i32);
    true
}

/// SetLastError — write to shared page AND guest TEB
unsafe fn handle_set_last_error() -> bool {
    let esp = read_reg32(ESP);
    let error_code = match safe_read32s(esp + 4) {
        Ok(v) => v as u32,
        Err(_) => return false,
    };

    // Update shared area
    *(hp_mut().add(OFF_HC_LAST_ERROR) as *mut u32) = error_code;

    // Also write to guest TEB at offset 0x34 if TEB address is known
    let teb_base = *(hp_ptr().add(OFF_HC_TEB_BASE) as *const u32);
    if teb_base != 0 {
        let _ = safe_write32((teb_base + 0x34) as i32, error_code as i32);
    }

    write_reg32(EAX, 0); // SetLastError returns void, EAX convention
    true
}

/// InterlockedIncrement — atomic in single-threaded WASM
unsafe fn handle_interlocked_inc() -> bool {
    let esp = read_reg32(ESP);
    let ptr = match safe_read32s(esp + 4) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let old = match safe_read32s(ptr) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let new_val = old.wrapping_add(1);
    if safe_write32(ptr, new_val).is_err() {
        return false;
    }
    write_reg32(EAX, new_val);
    true
}

/// InterlockedDecrement — atomic in single-threaded WASM
unsafe fn handle_interlocked_dec() -> bool {
    let esp = read_reg32(ESP);
    let ptr = match safe_read32s(esp + 4) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let old = match safe_read32s(ptr) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let new_val = old.wrapping_sub(1);
    if safe_write32(ptr, new_val).is_err() {
        return false;
    }
    write_reg32(EAX, new_val);
    true
}

/// InterlockedExchange(ptr, value) → old value
unsafe fn handle_interlocked_xchg() -> bool {
    let esp = read_reg32(ESP);
    let ptr = match safe_read32s(esp + 4) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let new_value = match safe_read32s(esp + 8) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let old = match safe_read32s(ptr) {
        Ok(v) => v,
        Err(_) => return false,
    };
    if safe_write32(ptr, new_value).is_err() {
        return false;
    }
    write_reg32(EAX, old);
    true
}

/// InterlockedCompareExchange(ptr, exchange, comparand) → old value
unsafe fn handle_interlocked_cmp_xchg() -> bool {
    let esp = read_reg32(ESP);
    let ptr = match safe_read32s(esp + 4) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let exchange = match safe_read32s(esp + 8) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let comparand = match safe_read32s(esp + 12) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let current = match safe_read32s(ptr) {
        Ok(v) => v,
        Err(_) => return false,
    };
    if current == comparand {
        if safe_write32(ptr, exchange).is_err() {
            return false;
        }
    }
    write_reg32(EAX, current);
    true
}

/// EnterCriticalSection — uncontended fast path only.
/// CRITICAL_SECTION layout (24 bytes):
///   +0:  DebugInfo        (ptr, ignored)
///   +4:  LockCount        (i32)
///   +8:  RecursionCount   (i32)
///   +12: OwningThread     (u32, thread ID)
///   +16: LockSemaphore    (handle, ignored)
///   +20: SpinCount        (u32, ignored)
///
/// Returns false (→ JS fallback) if contended.
unsafe fn handle_enter_critical_section() -> bool {
    let esp = read_reg32(ESP);
    let ptr = match safe_read32s(esp + 4) {
        Ok(v) => v,
        Err(_) => return false,
    };
    if ptr == 0 { return false; }

    let current_thread = *(hp_ptr().add(OFF_HC_CURRENT_THREAD_ID) as *const u32);
    if current_thread == 0 { return false; } // no thread info → JS fallback

    // Read OwningThread at offset 12
    let owner = match safe_read32s(ptr + 12) {
        Ok(v) => v as u32,
        Err(_) => return false,
    };

    if owner == 0 {
        // FREE — acquire it
        if safe_write32(ptr + 4, 0).is_err() { return false; }         // LockCount = 0
        if safe_write32(ptr + 8, 1).is_err() { return false; }         // RecursionCount = 1
        if safe_write32(ptr + 12, current_thread as i32).is_err() { return false; } // OwningThread
        write_reg32(EAX, 0);
        return true;
    }

    if owner == current_thread {
        // RECURSIVE — increment RecursionCount
        let rec = match safe_read32s(ptr + 8) {
            Ok(v) => v,
            Err(_) => return false,
        };
        if safe_write32(ptr + 8, rec + 1).is_err() { return false; }
        write_reg32(EAX, 0);
        return true;
    }

    // CONTENDED — fall through to JS (needs scheduler wait)
    false
}

/// LeaveCriticalSection — handles both recursive and full release.
/// If LockSemaphore (CS+16) != 0, returns false so JS handles release + SetEvent.
/// If LockSemaphore == 0 (no waiters ever), releases entirely in WASM (zero JS overhead).
///
/// CRITICAL: Must verify OwningThread matches current thread. Without this check,
/// a non-owner thread calling LeaveCS (game bug or race) silently releases the CS,
/// allowing another thread to acquire it — corrupting shared state and causing
/// downstream crashes (ESP corruption → infinite #PF).
unsafe fn handle_leave_critical_section() -> bool {
    let esp = read_reg32(ESP);
    let ptr = match safe_read32s(esp + 4) {
        Ok(v) => v,
        Err(_) => return false,
    };
    if ptr == 0 { return false; }

    // Verify ownership: only the owning thread may release.
    // Non-owner releases fall to JS for proper error reporting (FATAL_GUARD 0x3009).
    let current_thread = *(hp_ptr().add(OFF_HC_CURRENT_THREAD_ID) as *const u32);
    let owner = match safe_read32s(ptr + 12) {
        Ok(v) => v as u32,
        Err(_) => return false,
    };
    if current_thread == 0 || (owner != 0 && owner != current_thread) {
        return false; // Non-owner or unknown thread → JS slow path handles it
    }

    // Read RecursionCount at offset 8
    let rec = match safe_read32s(ptr + 8) {
        Ok(v) => v,
        Err(_) => return false,
    };

    if rec > 1 {
        // Still recursive — just decrement, no wake needed
        if safe_write32(ptr + 8, rec - 1).is_err() { return false; }
        write_reg32(EAX, 0);
        return true;
    }

    // Check LockSemaphore at offset 16 BEFORE releasing.
    // If event handle exists, waiters may be present — fall to JS for SetEvent.
    let lock_sem = match safe_read32s(ptr + 16) {
        Ok(v) => v as u32,
        Err(_) => return false,
    };
    if lock_sem != 0 {
        // JS handles: release CS fields + SetEvent(lockSem) + ownership transfer
        return false;
    }

    // No LockSemaphore — no waiter has ever contended. Release fully in WASM.
    if safe_write32(ptr + 4, -1).is_err() { return false; }  // LockCount = -1 (free)
    if safe_write32(ptr + 8, 0).is_err() { return false; }   // RecursionCount = 0
    if safe_write32(ptr + 12, 0).is_err() { return false; }  // OwningThread = 0

    write_reg32(EAX, 0);
    true
}

/// GetCapture — pure read of the capture owner mirrored at OFF_HC_CAPTURE_HWND.
/// Takes no arguments and touches no scheduler state, so it never falls through:
/// 0 (no capture) is the correct answer, not an "unpublished" sentinel.
unsafe fn handle_get_capture() -> bool {
    let hwnd = *(hp_ptr().add(OFF_HC_CAPTURE_HWND) as *const u32);
    write_reg32(EAX, hwnd as i32);
    true
}

/// IsIconic — always returns FALSE (we never minimize windows)
unsafe fn handle_is_iconic() -> bool {
    // Argument: HWND at [ESP+4], ignored — always return FALSE
    write_reg32(EAX, 0);
    true
}

/// ScreenToClient — subtract window offset from POINT
unsafe fn handle_screen_to_client() -> bool {
    let esp = read_reg32(ESP);
    // Args: HWND at [ESP+4] (ignored), lpPoint at [ESP+8]
    let lp_point = match safe_read32s(esp + 8) {
        Ok(v) => v,
        Err(_) => return false,
    };
    if lp_point == 0 { return false; }

    let page = hp_ptr();
    let win_x = *(page.add(OFF_HC_WINDOW_X) as *const i32);
    let win_y = *(page.add(OFF_HC_WINDOW_Y) as *const i32);

    let sx = match safe_read32s(lp_point) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let sy = match safe_read32s(lp_point + 4) {
        Ok(v) => v,
        Err(_) => return false,
    };

    if safe_write32(lp_point, sx - win_x).is_err() { return false; }
    if safe_write32(lp_point + 4, sy - win_y).is_err() { return false; }

    write_reg32(EAX, 1);
    true
}

/// PeekMessage — empty-queue fast path.
/// Returns FALSE (0) immediately when the queue is empty and starvation
/// counter hasn't reached the limit. Falls through to JS otherwise so
/// that onThunkComplete() and the scheduler still get to run periodically.
unsafe fn handle_peek_message() -> bool {
    let page = hp_ptr();
    let counter_ptr = hp_mut().add(OFF_HC_PEEK_STARVATION_COUNTER) as *mut u32;
    let counter = *counter_ptr;
    let limit = *(page.add(OFF_HC_PEEK_STARVATION_LIMIT) as *const u32);

    // Uninitialized or starvation limit hit → fall through to JS
    if limit == 0 || counter >= limit {
        *counter_ptr = 0;
        return false;
    }

    // Queue has messages → fall through to JS for full processing
    if *(page.add(OFF_HC_MSG_QUEUE_FLAG) as *const u32) != 0 {
        *counter_ptr = 0;
        return false;
    }

    // Queue empty → return FALSE immediately, no JS transition
    *counter_ptr = counter + 1;
    write_reg32(EAX, 0);
    true
}

/// GetCursorPos — write cached cursor position to POINT
unsafe fn handle_get_cursor_pos() -> bool {
    let esp = read_reg32(ESP);
    let lp_point = match safe_read32s(esp + 4) {
        Ok(v) => v,
        Err(_) => return false,
    };
    if lp_point == 0 { return false; }

    let page = hp_ptr();
    let cx = *(page.add(OFF_HC_CURSOR_X) as *const i32);
    let cy = *(page.add(OFF_HC_CURSOR_Y) as *const i32);

    if safe_write32(lp_point, cx).is_err() { return false; }
    if safe_write32(lp_point + 4, cy).is_err() { return false; }

    write_reg32(EAX, 1);
    true
}

// ---------------------------------------------------------------------------
// Tier 2: Math/FPU hypercall handlers
// ---------------------------------------------------------------------------

/// Helper: set ST(0) in-place without push/pop
#[inline(always)]
unsafe fn fpu_set_st0(value: F80) {
    fpu_write_st(*fpu_stack_ptr as i32, value);
}

/// Helper: read an f64 from the x86 stack at the given address
#[inline(always)]
unsafe fn read_stack_f64(addr: i32) -> f64 {
    let lo = match safe_read32s(addr) {
        Ok(v) => v as u32,
        Err(_) => return 0.0,
    };
    let hi = match safe_read32s(addr + 4) {
        Ok(v) => v as u32,
        Err(_) => return 0.0,
    };
    f64::from_bits((hi as u64) << 32 | lo as u64)
}

/// _ftol / __ftol: convert ST(0) to a signed __int64 (truncate toward zero), pop the stack,
/// return the FULL 64-bit result in EDX:EAX. The caller uses EAX alone for `(long)` or EDX:EAX
/// for `__int64`. The old code returned only a 32-bit value CLAMPED to INT32_MAX/MIN — that
/// broke every 64-bit user, notably the UE1 (Harry Potter) launcher's RDTSC timebase:
/// `now = _ftol(GSecondsPerCycle * TSC * 2^32)` exceeds INT32_MAX within ~0.5s of uptime, so the
/// clamp pinned `now` to 0x7fffffff every frame -> per-frame DeltaTime = (now-last)/2^32 = 0 ->
/// frozen splash countdown / racing storybook / audio underrun (all one defect). Must be int64.
unsafe fn handle_ftol() -> bool {
    let st0 = fpu_get_st0();
    let val = f64::from_bits(st0.to_f64());
    fpu_pop();
    let v = val as i64; // Rust float->int `as` truncates toward zero and saturates on overflow
    write_reg32(EAX, v as i32);          // low 32
    write_reg32(EDX, (v >> 32) as i32);  // high 32
    true
}

// --- _CI* intrinsics: read ST(0)/ST(1), operate, write result back ---

/// _CIsin: ST(0) = sin(ST(0))
unsafe fn handle_ci_sin() -> bool {
    let x = f64::from_bits(fpu_get_st0().to_f64());
    fpu_set_st0(F80::of_f64(x.sin().to_bits()));
    true
}

/// _CIcos: ST(0) = cos(ST(0))
unsafe fn handle_ci_cos() -> bool {
    let x = f64::from_bits(fpu_get_st0().to_f64());
    fpu_set_st0(F80::of_f64(x.cos().to_bits()));
    true
}

/// _CItan: ST(0) = tan(ST(0))
unsafe fn handle_ci_tan() -> bool {
    let x = f64::from_bits(fpu_get_st0().to_f64());
    fpu_set_st0(F80::of_f64(x.tan().to_bits()));
    true
}

/// _CIsqrt: ST(0) = sqrt(ST(0))
unsafe fn handle_ci_sqrt() -> bool {
    let x = f64::from_bits(fpu_get_st0().to_f64());
    fpu_set_st0(F80::of_f64(x.sqrt().to_bits()));
    true
}

/// _CIlog: ST(0) = ln(ST(0))
unsafe fn handle_ci_log() -> bool {
    let x = f64::from_bits(fpu_get_st0().to_f64());
    fpu_set_st0(F80::of_f64(x.ln().to_bits()));
    true
}

/// _CIexp: ST(0) = exp(ST(0))
unsafe fn handle_ci_exp() -> bool {
    let x = f64::from_bits(fpu_get_st0().to_f64());
    fpu_set_st0(F80::of_f64(x.exp().to_bits()));
    true
}

/// _CIacos: ST(0) = acos(ST(0))
unsafe fn handle_ci_acos() -> bool {
    let x = f64::from_bits(fpu_get_st0().to_f64());
    fpu_set_st0(F80::of_f64(x.acos().to_bits()));
    true
}

/// _CIasin: ST(0) = asin(ST(0))
unsafe fn handle_ci_asin() -> bool {
    let x = f64::from_bits(fpu_get_st0().to_f64());
    fpu_set_st0(F80::of_f64(x.asin().to_bits()));
    true
}

/// _CIlog10: ST(0) = log10(ST(0))
unsafe fn handle_ci_log10() -> bool {
    let x = f64::from_bits(fpu_get_st0().to_f64());
    fpu_set_st0(F80::of_f64(x.log10().to_bits()));
    true
}

/// _CIatan2: atan2(y=ST(1), x=ST(0)), pop ST(0), write result to new ST(0)
unsafe fn handle_ci_atan2() -> bool {
    let x = f64::from_bits(fpu_get_st0().to_f64());
    let y = f64::from_bits(fpu_get_sti(1).to_f64());
    let result = y.atan2(x);
    fpu_pop();
    fpu_set_st0(F80::of_f64(result.to_bits()));
    true
}

/// _CIfmod: fmod(y=ST(1), x=ST(0)), pop ST(0), write result to new ST(0)
unsafe fn handle_ci_fmod() -> bool {
    let x = f64::from_bits(fpu_get_st0().to_f64());
    let y = f64::from_bits(fpu_get_sti(1).to_f64());
    let result = if x != 0.0 { y % x } else { 0.0 };
    fpu_pop();
    fpu_set_st0(F80::of_f64(result.to_bits()));
    true
}

/// _CIpow: pow(base=ST(1), exp=ST(0)), pop ST(0), write result to new ST(0)
unsafe fn handle_ci_pow() -> bool {
    let exponent = f64::from_bits(fpu_get_st0().to_f64());
    let base = f64::from_bits(fpu_get_sti(1).to_f64());
    let result = base.powf(exponent);
    fpu_pop();
    fpu_set_st0(F80::of_f64(result.to_bits()));
    true
}

// --- cdecl math functions: read double arg from stack, push result to FPU ---

/// cdecl sin(double x): [ESP+4..+11] = x, result pushed to FPU ST(0)
unsafe fn handle_cdecl_sin() -> bool {
    let x = read_stack_f64(read_reg32(ESP) + 4);
    fpu_push(F80::of_f64(x.sin().to_bits()));
    true
}

unsafe fn handle_cdecl_cos() -> bool {
    let x = read_stack_f64(read_reg32(ESP) + 4);
    fpu_push(F80::of_f64(x.cos().to_bits()));
    true
}

unsafe fn handle_cdecl_tan() -> bool {
    let x = read_stack_f64(read_reg32(ESP) + 4);
    fpu_push(F80::of_f64(x.tan().to_bits()));
    true
}

unsafe fn handle_cdecl_sqrt() -> bool {
    let x = read_stack_f64(read_reg32(ESP) + 4);
    fpu_push(F80::of_f64(x.sqrt().to_bits()));
    true
}

unsafe fn handle_cdecl_log() -> bool {
    let x = read_stack_f64(read_reg32(ESP) + 4);
    fpu_push(F80::of_f64(x.ln().to_bits()));
    true
}

unsafe fn handle_cdecl_exp() -> bool {
    let x = read_stack_f64(read_reg32(ESP) + 4);
    fpu_push(F80::of_f64(x.exp().to_bits()));
    true
}

unsafe fn handle_cdecl_acos() -> bool {
    let x = read_stack_f64(read_reg32(ESP) + 4);
    fpu_push(F80::of_f64(x.acos().to_bits()));
    true
}

unsafe fn handle_cdecl_asin() -> bool {
    let x = read_stack_f64(read_reg32(ESP) + 4);
    fpu_push(F80::of_f64(x.asin().to_bits()));
    true
}

unsafe fn handle_cdecl_log10() -> bool {
    let x = read_stack_f64(read_reg32(ESP) + 4);
    fpu_push(F80::of_f64(x.log10().to_bits()));
    true
}

unsafe fn handle_cdecl_atan() -> bool {
    let x = read_stack_f64(read_reg32(ESP) + 4);
    fpu_push(F80::of_f64(x.atan().to_bits()));
    true
}

unsafe fn handle_cdecl_fabs() -> bool {
    let x = read_stack_f64(read_reg32(ESP) + 4);
    fpu_push(F80::of_f64(x.abs().to_bits()));
    true
}

/// cdecl atan2(double y, double x): y at [ESP+4], x at [ESP+12]
unsafe fn handle_cdecl_atan2() -> bool {
    let esp = read_reg32(ESP);
    let y = read_stack_f64(esp + 4);
    let x = read_stack_f64(esp + 12);
    fpu_push(F80::of_f64(y.atan2(x).to_bits()));
    true
}

/// cdecl fmod(double x, double y): x at [ESP+4], y at [ESP+12]
unsafe fn handle_cdecl_fmod() -> bool {
    let esp = read_reg32(ESP);
    let x = read_stack_f64(esp + 4);
    let y = read_stack_f64(esp + 12);
    let result = if y != 0.0 { x % y } else { 0.0 };
    fpu_push(F80::of_f64(result.to_bits()));
    true
}

/// cdecl pow(double x, double y): x at [ESP+4], y at [ESP+12]
unsafe fn handle_cdecl_pow() -> bool {
    let esp = read_reg32(ESP);
    let x = read_stack_f64(esp + 4);
    let y = read_stack_f64(esp + 12);
    fpu_push(F80::of_f64(x.powf(y).to_bits()));
    true
}

/// cdecl ceil(double x): result pushed to FPU ST(0), EAX = truncated int
unsafe fn handle_cdecl_ceil() -> bool {
    let x = read_stack_f64(read_reg32(ESP) + 4);
    let result = x.ceil();
    fpu_push(F80::of_f64(result.to_bits()));
    write_reg32(EAX, result as i32);
    true
}

/// cdecl floor(double x): result pushed to FPU ST(0), EAX = truncated int
unsafe fn handle_cdecl_floor() -> bool {
    let x = read_stack_f64(read_reg32(ESP) + 4);
    let result = x.floor();
    fpu_push(F80::of_f64(result.to_bits()));
    write_reg32(EAX, result as i32);
    true
}

// ---------------------------------------------------------------------------
// Tier 3: String/memory hypercall handlers
// ---------------------------------------------------------------------------

/// Helper: safe_read16 wrapper (local, to avoid import issues)
unsafe fn hc_safe_read16(addr: i32) -> Result<i32, ()> {
    crate::cpu::cpu::safe_read16(addr).map_err(|_| ())
}

/// Helper: safe_read8 wrapper
pub(crate) unsafe fn hc_safe_read8(addr: i32) -> Result<i32, ()> {
    crate::cpu::cpu::safe_read8(addr).map_err(|_| ())
}

/// Helper: safe_write16 wrapper
unsafe fn hc_safe_write16(addr: i32, value: i32) -> Result<(), ()> {
    crate::cpu::cpu::safe_write16(addr, value).map_err(|_| ())
}

/// Helper: safe_write8 wrapper
unsafe fn hc_safe_write8(addr: i32, value: i32) -> Result<(), ()> {
    crate::cpu::cpu::safe_write8(addr, value).map_err(|_| ())
}

/// wcslen(wchar_t* str): count of wchar_t until NUL
unsafe fn handle_wcslen() -> bool {
    let esp = read_reg32(ESP);
    let str_ptr = match safe_read32s(esp + 4) {
        Ok(v) => v,
        Err(_) => return false,
    };
    if str_ptr == 0 {
        write_reg32(EAX, 0);
        return true;
    }
    let mut len: u32 = 0;
    loop {
        let ch = match hc_safe_read16(str_ptr + len as i32 * 2) {
            Ok(v) => v,
            Err(_) => break,
        };
        if ch == 0 { break; }
        len += 1;
        if len > 0x100000 { break; }
    }
    write_reg32(EAX, len as i32);
    true
}

/// wcscpy(wchar_t* dst, wchar_t* src): copy wide string, return dst
unsafe fn handle_wcscpy() -> bool {
    let esp = read_reg32(ESP);
    let dst = match safe_read32s(esp + 4) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let src = match safe_read32s(esp + 8) {
        Ok(v) => v,
        Err(_) => return false,
    };
    if dst == 0 || src == 0 {
        write_reg32(EAX, dst);
        return true;
    }
    let mut i: u32 = 0;
    loop {
        let ch = match hc_safe_read16(src + i as i32 * 2) {
            Ok(v) => v,
            Err(_) => return false,
        };
        if hc_safe_write16(dst + i as i32 * 2, ch).is_err() { return false; }
        if ch == 0 { break; }
        i += 1;
        if i > 0x100000 { break; }
    }
    write_reg32(EAX, dst);
    true
}

/// wcscat(wchar_t* dst, wchar_t* src): append src to dst, return dst
unsafe fn handle_wcscat() -> bool {
    let esp = read_reg32(ESP);
    let dst = match safe_read32s(esp + 4) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let src = match safe_read32s(esp + 8) {
        Ok(v) => v,
        Err(_) => return false,
    };
    if dst == 0 || src == 0 {
        write_reg32(EAX, dst);
        return true;
    }
    // Find end of dst
    let mut dst_len: u32 = 0;
    loop {
        let ch = match hc_safe_read16(dst + dst_len as i32 * 2) {
            Ok(v) => v,
            Err(_) => return false,
        };
        if ch == 0 { break; }
        dst_len += 1;
        if dst_len > 0x100000 { break; }
    }
    // Copy src
    let mut i: u32 = 0;
    loop {
        let ch = match hc_safe_read16(src + i as i32 * 2) {
            Ok(v) => v,
            Err(_) => return false,
        };
        if hc_safe_write16(dst + (dst_len + i) as i32 * 2, ch).is_err() { return false; }
        if ch == 0 { break; }
        i += 1;
        if i > 0x100000 { break; }
    }
    write_reg32(EAX, dst);
    true
}

/// _wcsicmp(wchar_t* s1, wchar_t* s2): case-insensitive wide string compare
unsafe fn handle_wcsicmp() -> bool {
    let esp = read_reg32(ESP);
    let s1 = match safe_read32s(esp + 4) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let s2 = match safe_read32s(esp + 8) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let mut i: u32 = 0;
    let result = loop {
        let c1 = match hc_safe_read16(s1 + i as i32 * 2) {
            Ok(v) => v as u32,
            Err(_) => return false,
        };
        let c2 = match hc_safe_read16(s2 + i as i32 * 2) {
            Ok(v) => v as u32,
            Err(_) => return false,
        };
        // ASCII tolower
        let l1 = if c1 >= 0x41 && c1 <= 0x5A { c1 + 0x20 } else { c1 };
        let l2 = if c2 >= 0x41 && c2 <= 0x5A { c2 + 0x20 } else { c2 };
        if l1 != l2 { break (l1 as i32) - (l2 as i32); }
        if c1 == 0 { break 0i32; }
        i += 1;
        if i > 0x100000 { break 0i32; }
    };
    write_reg32(EAX, result);
    true
}

/// wcschr(wchar_t* str, wchar_t ch): find first occurrence
unsafe fn handle_wcschr() -> bool {
    let esp = read_reg32(ESP);
    let str_ptr = match safe_read32s(esp + 4) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let target = match safe_read32s(esp + 8) {
        Ok(v) => v & 0xFFFF,
        Err(_) => return false,
    };
    if str_ptr == 0 {
        write_reg32(EAX, 0);
        return true;
    }
    let mut i: u32 = 0;
    loop {
        let ch = match hc_safe_read16(str_ptr + i as i32 * 2) {
            Ok(v) => v,
            Err(_) => return false,
        };
        if ch == target {
            write_reg32(EAX, str_ptr + i as i32 * 2);
            return true;
        }
        if ch == 0 { break; }
        i += 1;
        if i > 0x100000 { break; }
    }
    write_reg32(EAX, 0);
    true
}

/// wcsstr(wchar_t* haystack, wchar_t* needle): find substring, return ptr or NULL.
/// Naive O(n*m) search, matching MSVCRT. Hot during game asset/string loading.
unsafe fn handle_wcsstr() -> bool {
    let esp = read_reg32(ESP);
    let haystack = match safe_read32s(esp + 4) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let needle = match safe_read32s(esp + 8) {
        Ok(v) => v,
        Err(_) => return false,
    };
    if haystack == 0 || needle == 0 {
        write_reg32(EAX, 0);
        return true;
    }
    // Empty needle → return haystack (C standard).
    let first = match hc_safe_read16(needle) {
        Ok(v) => v,
        Err(_) => return false,
    };
    if first == 0 {
        write_reg32(EAX, haystack);
        return true;
    }
    let mut i: u32 = 0;
    loop {
        let hc = match hc_safe_read16(haystack + i as i32 * 2) {
            Ok(v) => v,
            Err(_) => return false,
        };
        if hc == 0 { break; }
        if hc == first {
            let mut j: u32 = 1;
            let matched = loop {
                let nc = match hc_safe_read16(needle + j as i32 * 2) {
                    Ok(v) => v,
                    Err(_) => return false,
                };
                if nc == 0 { break true; }
                let h2 = match hc_safe_read16(haystack + (i + j) as i32 * 2) {
                    Ok(v) => v,
                    Err(_) => return false,
                };
                if h2 != nc { break false; }
                j += 1;
                if j > 0x100000 { break false; }
            };
            if matched {
                write_reg32(EAX, haystack + i as i32 * 2);
                return true;
            }
        }
        i += 1;
        if i > 0x100000 { break; }
    }
    write_reg32(EAX, 0);
    true
}

/// _wcsnicmp(wchar_t* s1, wchar_t* s2, size_t count): case-insensitive compare up to count.
/// MUST mirror the JS fastPathWcsnicmp (msvcrt.ts) byte-for-byte: in this codebase's
/// convention count==0 means "compare the whole string" (limit 0x10000), NOT "0 chars =
/// equal". Returning 0 for count==0 made every such compare report equal → wrong config/
/// name branch → HP camera-follow regression. Keep the count==0 semantics identical to JS.
unsafe fn handle_wcsnicmp() -> bool {
    let esp = read_reg32(ESP);
    let s1 = match safe_read32s(esp + 4) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let s2 = match safe_read32s(esp + 8) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let count = match safe_read32s(esp + 12) {
        Ok(v) => v as u32,
        Err(_) => return false,
    };
    let limit: u32 = if count > 0 { count } else { 0x10000 };
    let mut i: u32 = 0;
    let result = loop {
        if i >= limit { break 0i32; }
        let c1 = match hc_safe_read16(s1 + i as i32 * 2) {
            Ok(v) => v as u32,
            Err(_) => return false,
        };
        let c2 = match hc_safe_read16(s2 + i as i32 * 2) {
            Ok(v) => v as u32,
            Err(_) => return false,
        };
        // ASCII tolower
        let l1 = if c1 >= 0x41 && c1 <= 0x5A { c1 + 0x20 } else { c1 };
        let l2 = if c2 >= 0x41 && c2 <= 0x5A { c2 + 0x20 } else { c2 };
        if l1 != l2 { break (l1 as i32) - (l2 as i32); }
        if c1 == 0 { break 0i32; }
        i += 1;
    };
    write_reg32(EAX, result);
    true
}

/// wcsncpy(wchar_t* dst, wchar_t* src, size_t count): copy up to count wchars, null-pad
/// remainder if src is shorter. No implicit terminator when src >= count (C standard).
/// Mirrors JS wcsncpy (msvcrt.ts): only dst==0 short-circuits. Do NOT special-case src==0
/// — JS doesn't, and early-returning there left dst holding stale data (divergence). When
/// src==0 the read below faults → hc_safe_read16 Err → JS fallback handles it like JS.
unsafe fn handle_wcsncpy() -> bool {
    let esp = read_reg32(ESP);
    let dst = match safe_read32s(esp + 4) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let src = match safe_read32s(esp + 8) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let count = match safe_read32s(esp + 12) {
        Ok(v) => v as u32,
        Err(_) => return false,
    };
    if dst == 0 {
        write_reg32(EAX, dst);
        return true;
    }
    // Defer pathologically large copies to JS rather than spin in WASM.
    if count > 0x100000 { return false; }
    let mut i: u32 = 0;
    let mut hit_null = false;
    while i < count {
        let ch = if hit_null {
            0
        } else {
            match hc_safe_read16(src + i as i32 * 2) {
                Ok(v) => v,
                Err(_) => return false,
            }
        };
        if hc_safe_write16(dst + i as i32 * 2, ch).is_err() { return false; }
        if ch == 0 { hit_null = true; }
        i += 1;
    }
    write_reg32(EAX, dst);
    true
}

/// memcpy(void* dst, void* src, size_t n): copy n bytes
unsafe fn handle_memcpy() -> bool {
    let esp = read_reg32(ESP);
    let dst = match safe_read32s(esp + 4) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let src = match safe_read32s(esp + 8) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let len = match safe_read32s(esp + 12) {
        Ok(v) => v as u32,
        Err(_) => return false,
    };
    if dst == 0 || src == 0 || len == 0 {
        write_reg32(EAX, dst);
        return true;
    }
    // Copy 4 bytes at a time, then remaining bytes
    let mut i: u32 = 0;
    while i + 4 <= len {
        let val = match safe_read32s(src + i as i32) {
            Ok(v) => v,
            Err(_) => return false,
        };
        if safe_write32(dst + i as i32, val).is_err() { return false; }
        i += 4;
    }
    while i < len {
        let byte = match hc_safe_read8(src + i as i32) {
            Ok(v) => v,
            Err(_) => return false,
        };
        if hc_safe_write8(dst + i as i32, byte).is_err() { return false; }
        i += 1;
    }
    write_reg32(EAX, dst);
    true
}

/// memset(void* dst, int c, size_t n): fill n bytes with c
unsafe fn handle_memset() -> bool {
    let esp = read_reg32(ESP);
    let dst = match safe_read32s(esp + 4) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let c = match safe_read32s(esp + 8) {
        Ok(v) => v & 0xFF,
        Err(_) => return false,
    };
    let len = match safe_read32s(esp + 12) {
        Ok(v) => v as u32,
        Err(_) => return false,
    };
    if dst == 0 || len == 0 {
        write_reg32(EAX, dst);
        return true;
    }
    // Fill 4 bytes at a time
    let fill32 = c | (c << 8) | (c << 16) | (c << 24);
    let mut i: u32 = 0;
    while i + 4 <= len {
        if safe_write32(dst + i as i32, fill32).is_err() { return false; }
        i += 4;
    }
    while i < len {
        if hc_safe_write8(dst + i as i32, c).is_err() { return false; }
        i += 1;
    }
    write_reg32(EAX, dst);
    true
}

/// strlen(char* str): count bytes until NUL
unsafe fn handle_strlen() -> bool {
    let esp = read_reg32(ESP);
    let str_ptr = match safe_read32s(esp + 4) {
        Ok(v) => v,
        Err(_) => return false,
    };
    if str_ptr == 0 {
        write_reg32(EAX, 0);
        return true;
    }
    let mut len: u32 = 0;
    loop {
        let ch = match hc_safe_read8(str_ptr + len as i32) {
            Ok(v) => v,
            Err(_) => break,
        };
        if ch == 0 { break; }
        len += 1;
        if len > 0x100000 { break; }
    }
    write_reg32(EAX, len as i32);
    true
}

/// strcmp(char* s1, char* s2): byte-by-byte comparison
unsafe fn handle_strcmp() -> bool {
    let esp = read_reg32(ESP);
    let s1 = match safe_read32s(esp + 4) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let s2 = match safe_read32s(esp + 8) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let mut i: u32 = 0;
    let result = loop {
        let c1 = match hc_safe_read8(s1 + i as i32) {
            Ok(v) => v as u32,
            Err(_) => return false,
        };
        let c2 = match hc_safe_read8(s2 + i as i32) {
            Ok(v) => v as u32,
            Err(_) => return false,
        };
        if c1 != c2 { break (c1 as i32) - (c2 as i32); }
        if c1 == 0 { break 0i32; }
        i += 1;
        if i > 0x100000 { break 0i32; }
    };
    write_reg32(EAX, result);
    true
}

/// strcpy(char* dst, char* src): copy ANSI string, return dst
unsafe fn handle_strcpy() -> bool {
    let esp = read_reg32(ESP);
    let dst = match safe_read32s(esp + 4) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let src = match safe_read32s(esp + 8) {
        Ok(v) => v,
        Err(_) => return false,
    };
    if dst == 0 || src == 0 {
        write_reg32(EAX, dst);
        return true;
    }
    let mut i: u32 = 0;
    loop {
        let ch = match hc_safe_read8(src + i as i32) {
            Ok(v) => v,
            Err(_) => return false,
        };
        if hc_safe_write8(dst + i as i32, ch).is_err() { return false; }
        if ch == 0 { break; }
        i += 1;
        if i > 0x100000 { break; }
    }
    write_reg32(EAX, dst);
    true
}

/// _stricmp(char* s1, char* s2): case-insensitive ANSI string compare
unsafe fn handle_stricmp() -> bool {
    let esp = read_reg32(ESP);
    let s1 = match safe_read32s(esp + 4) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let s2 = match safe_read32s(esp + 8) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let mut i: u32 = 0;
    let result = loop {
        let c1 = match hc_safe_read8(s1 + i as i32) {
            Ok(v) => v as u32,
            Err(_) => return false,
        };
        let c2 = match hc_safe_read8(s2 + i as i32) {
            Ok(v) => v as u32,
            Err(_) => return false,
        };
        let l1 = if c1 >= 0x41 && c1 <= 0x5A { c1 + 0x20 } else { c1 };
        let l2 = if c2 >= 0x41 && c2 <= 0x5A { c2 + 0x20 } else { c2 };
        if l1 != l2 { break (l1 as i32) - (l2 as i32); }
        if c1 == 0 { break 0i32; }
        i += 1;
        if i > 0x100000 { break 0i32; }
    };
    write_reg32(EAX, result);
    true
}

/// memcmp(void* s1, void* s2, size_t n): compare n bytes
unsafe fn handle_memcmp() -> bool {
    let esp = read_reg32(ESP);
    let s1 = match safe_read32s(esp + 4) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let s2 = match safe_read32s(esp + 8) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let len = match safe_read32s(esp + 12) {
        Ok(v) => v as u32,
        Err(_) => return false,
    };
    let mut i: u32 = 0;
    while i < len {
        let c1 = match hc_safe_read8(s1 + i as i32) {
            Ok(v) => v as u32,
            Err(_) => return false,
        };
        let c2 = match hc_safe_read8(s2 + i as i32) {
            Ok(v) => v as u32,
            Err(_) => return false,
        };
        if c1 != c2 {
            write_reg32(EAX, (c1 as i32) - (c2 as i32));
            return true;
        }
        i += 1;
    }
    write_reg32(EAX, 0);
    true
}

/// _strnicmp(char* s1, char* s2, size_t count): case-insensitive ANSI compare up to count.
/// Mirrors the NARROW JS strnicmp (msvcrt.ts): count==0 → 0 ("compare zero chars → equal").
/// This is the OPPOSITE of the wide _wcsnicmp fast-path convention (count==0 → whole string,
/// see handle_wcsnicmp) — keep the two distinct or config/name branches flip. Same tolower
/// difference convention as handle_stricmp, just bounded by count.
unsafe fn handle_strnicmp() -> bool {
    let esp = read_reg32(ESP);
    let s1 = match safe_read32s(esp + 4) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let s2 = match safe_read32s(esp + 8) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let count = match safe_read32s(esp + 12) {
        Ok(v) => v as u32,
        Err(_) => return false,
    };
    if count == 0 {
        write_reg32(EAX, 0);
        return true;
    }
    let mut i: u32 = 0;
    let result = loop {
        if i >= count { break 0i32; }
        let c1 = match hc_safe_read8(s1 + i as i32) {
            Ok(v) => v as u32,
            Err(_) => return false,
        };
        let c2 = match hc_safe_read8(s2 + i as i32) {
            Ok(v) => v as u32,
            Err(_) => return false,
        };
        let l1 = if c1 >= 0x41 && c1 <= 0x5A { c1 + 0x20 } else { c1 };
        let l2 = if c2 >= 0x41 && c2 <= 0x5A { c2 + 0x20 } else { c2 };
        if l1 != l2 { break (l1 as i32) - (l2 as i32); }
        if c1 == 0 { break 0i32; }
        i += 1;
    };
    write_reg32(EAX, result);
    true
}

/// strstr(char* haystack, char* needle): substring search, return ptr into haystack or NULL.
/// Byte-wise analogue of handle_wcsstr; mirrors the JS strstr (crt-string.ts): null haystack
/// OR null needle → 0, empty needle → haystack. Naive O(n*m), matching MSVCRT / the JS impl.
unsafe fn handle_strstr() -> bool {
    let esp = read_reg32(ESP);
    let haystack = match safe_read32s(esp + 4) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let needle = match safe_read32s(esp + 8) {
        Ok(v) => v,
        Err(_) => return false,
    };
    if haystack == 0 || needle == 0 {
        write_reg32(EAX, 0);
        return true;
    }
    // Empty needle → return haystack (matches JS `sub.length === 0`).
    let first = match hc_safe_read8(needle) {
        Ok(v) => v,
        Err(_) => return false,
    };
    if first == 0 {
        write_reg32(EAX, haystack);
        return true;
    }
    let mut i: u32 = 0;
    loop {
        let hc = match hc_safe_read8(haystack + i as i32) {
            Ok(v) => v,
            Err(_) => return false,
        };
        if hc == 0 { break; }
        if hc == first {
            let mut j: u32 = 1;
            let matched = loop {
                let nc = match hc_safe_read8(needle + j as i32) {
                    Ok(v) => v,
                    Err(_) => return false,
                };
                if nc == 0 { break true; }
                let h2 = match hc_safe_read8(haystack + (i + j) as i32) {
                    Ok(v) => v,
                    Err(_) => return false,
                };
                if h2 != nc { break false; }
                j += 1;
                if j > 0x100000 { break false; }
            };
            if matched {
                write_reg32(EAX, haystack + i as i32);
                return true;
            }
        }
        i += 1;
        if i > 0x100000 { break; }
    }
    write_reg32(EAX, 0);
    true
}

/// atoi/atol(char* str): parse a leading optional-signed decimal integer (C semantics).
/// Mirrors the JS atoi (msvcrt.ts): readCString.trim() then parseInt(...,10), result truncated
/// to i32 (`parsed | 0`). Skips ASCII whitespace, one optional +/- sign, then decimal digits;
/// stops at the first non-digit. Digit runs > 15 are deferred to JS: an exact i64 accumulator
/// past 2^53 would diverge from JS's double-precision parseInt, so we let JS own those (rare).
unsafe fn handle_atoi() -> bool {
    let esp = read_reg32(ESP);
    let mut p = match safe_read32s(esp + 4) {
        Ok(v) => v,
        Err(_) => return false,
    };
    if p == 0 {
        write_reg32(EAX, 0);
        return true;
    }
    // Skip leading ASCII whitespace: ' ' (0x20) and \t\n\v\f\r (0x09..=0x0D).
    loop {
        let c = match hc_safe_read8(p) {
            Ok(v) => v as u32,
            Err(_) => return false,
        };
        if c == 0x20 || (c >= 0x09 && c <= 0x0D) { p += 1; } else { break; }
    }
    // Optional sign.
    let mut neg = false;
    let c = match hc_safe_read8(p) {
        Ok(v) => v as u32,
        Err(_) => return false,
    };
    if c == '+' as u32 || c == '-' as u32 {
        neg = c == '-' as u32;
        p += 1;
    }
    // Decimal digits.
    let mut acc: i64 = 0;
    let mut digits: u32 = 0;
    loop {
        let d = match hc_safe_read8(p) {
            Ok(v) => v as u32,
            Err(_) => return false,
        };
        if d < '0' as u32 || d > '9' as u32 { break; }
        digits += 1;
        if digits > 15 { return false; } // stays exact in f64 → matches JS parseInt|0; longer → JS
        acc = acc * 10 + (d - '0' as u32) as i64;
        p += 1;
    }
    let val = if neg { -acc } else { acc };
    write_reg32(EAX, val as i32);
    true
}

/// IsBadReadPtr(lp, ucb) / IsBadWritePtr(lp, ucb) — stdcall(2), BOOL: 0 = accessible.
/// Faithful mechanism: like real Windows, PROBE the pages (one safe read per 4KB page)
/// instead of consulting an allocator-side list. Decommitted pages (MEM_DECOMMIT clears
/// the PTE Present bit — page-table-manager.ts) fail the probe. Any probe fault falls
/// through to the JS handler (return false), which distinguishes decommit from CoW /
/// other #PF causes — WASM only answers the hot all-pages-present case. The JS impl
/// checks read-access only for both variants (write perms are not modelled there), so
/// one probe routine serves both entry points; JS stays the source of truth.
unsafe fn handle_is_bad_ptr() -> bool {
    let esp = read_reg32(ESP);
    let lp = match safe_read32s(esp + 4) {
        Ok(v) => v as u32,
        Err(_) => return false,
    };
    let ucb = match safe_read32s(esp + 8) {
        Ok(v) => v as u32,
        Err(_) => return false,
    };
    // NULL with zero size is valid per Windows docs; NULL with size is bad.
    if ucb == 0 {
        write_reg32(EAX, 0);
        return true;
    }
    if lp == 0 {
        write_reg32(EAX, 1);
        return true;
    }
    // Out of guest RAM bounds (matches the JS `lp + ucb > mem.length` check).
    let mem_size = *crate::cpu::global_pointers::memory_size as u64;
    if lp as u64 + ucb as u64 > mem_size {
        write_reg32(EAX, 1);
        return true;
    }
    // Probe the first byte of every touched 4KB page. All present → accessible.
    let first_page = lp >> 12;
    let last_page = (lp + ucb - 1) >> 12;
    let mut page = first_page;
    while page <= last_page {
        let probe_addr = if page == first_page { lp } else { page << 12 };
        if hc_safe_read8(probe_addr as i32).is_err() {
            // Fault: decommit vs CoW vs other — let JS decide.
            return false;
        }
        page += 1;
    }
    write_reg32(EAX, 0);
    true
}

// ---------------------------------------------------------------------------
// Tier 4: Scheduler hypercall handlers
// ---------------------------------------------------------------------------

/// SetEvent(hEvent) — stdcall(1).
/// Fast path: valid mirrored event with no waiters → set signaled in shared table.
/// Falls through to JS when waiters are present, handle is out of range, or
/// periodic starvation limit is hit (scheduler/onThunkComplete hooks).
unsafe fn handle_set_event() -> bool {
    let esp = read_reg32(ESP);
    let handle = match safe_read32s(esp + 4) {
        Ok(v) => v as u32,
        Err(_) => return false,
    };

    if handle < KERNEL_HANDLE_BASE {
        return false;
    }
    let slot = (handle - KERNEL_HANDLE_BASE) >> 2;
    if slot >= EVENT_TABLE_SLOTS {
        return false;
    }

    let table_ptr = hp_mut().add(OFF_HC_EVENT_TABLE + slot as usize);
    let flags = *table_ptr;
    if flags & EVT_VALID == 0 {
        return false;
    }
    if flags & EVT_HAS_WAITERS != 0 {
        return false;
    }

    let counter_ptr = hp_mut().add(OFF_HC_EVENT_STARVATION_COUNTER) as *mut u32;
    let counter = *counter_ptr;
    let limit = *(hp_ptr().add(OFF_HC_EVENT_STARVATION_LIMIT) as *const u32);
    if limit > 0 && counter >= limit {
        *counter_ptr = 0;
        return false;
    }
    *counter_ptr = counter + 1;

    let mut new_flags = flags | EVT_SIGNALED;
    if flags & EVT_MANUAL != 0 {
        new_flags |= EVT_PENDING_WAKE;
    }
    *table_ptr = new_flags;

    write_reg32(EAX, 1);
    true
}

#[inline]
unsafe fn kernel_handle_slot(handle: u32) -> Option<u32> {
    if handle < KERNEL_HANDLE_BASE {
        return None;
    }
    let slot = (handle - KERNEL_HANDLE_BASE) >> 2;
    if slot >= EVENT_TABLE_SLOTS {
        return None;
    }
    Some(slot)
}

#[inline]
unsafe fn mutex_mirror_base() -> u32 {
    *(hp_ptr().add(OFF_HC_MUTEX_MIRROR_PTR) as *const u32)
}

#[inline]
unsafe fn read_mutex_mirror(slot: u32) -> u32 {
    let base = mutex_mirror_base();
    if base == 0 {
        return 0;
    }
    memory::read32_no_mmap_check(base + slot * 4) as u32
}

#[inline]
unsafe fn write_mutex_mirror(slot: u32, value: u32) {
    let base = mutex_mirror_base();
    if base == 0 {
        return;
    }
    memory::write32_no_mmap_or_dirty_check(base + slot * 4, value as i32);
}

/// ReleaseMutex(hMutex) — uncontended, no waiters. Mirrors sync-objects release.
unsafe fn handle_release_mutex() -> bool {
    let esp = read_reg32(ESP);
    let handle = match safe_read32s(esp + 4) {
        Ok(v) => v as u32,
        Err(_) => return false,
    };
    let slot = match kernel_handle_slot(handle) {
        Some(s) => s,
        None => return false,
    };

    let current = *(hp_ptr().add(OFF_HC_CURRENT_THREAD_ID) as *const u32);
    if current == 0 || current > MUX_OWNER_MASK {
        return false;
    }

    let mux = read_mutex_mirror(slot);
    if mux & MUX_VALID == 0 {
        return false;
    }
    if mux & MUX_HAS_WAITERS != 0 {
        return false;
    }
    if mux & MUX_ABANDONED != 0 {
        return false;
    }

    let owner = mux & MUX_OWNER_MASK;
    if owner != current {
        return false;
    }

    let rec = (mux & MUX_REC_MASK) >> MUX_REC_SHIFT;
    if rec > 1 {
        write_mutex_mirror(slot, (mux & !MUX_REC_MASK) | ((rec - 1) << MUX_REC_SHIFT));
    } else {
        write_mutex_mirror(slot, mux & !(MUX_OWNER_MASK | MUX_REC_MASK));
    }
    write_reg32(EAX, 1);
    true
}

/// WaitForSingleObject(h, ms) — immediate acquire for free/recursive mutex or signaled auto-reset event.
unsafe fn handle_wait_for_single_object() -> bool {
    let esp = read_reg32(ESP);
    let handle = match safe_read32s(esp + 4) {
        Ok(v) => v as u32,
        Err(_) => return false,
    };
    let timeout = match safe_read32s(esp + 8) {
        Ok(v) => v as u32,
        Err(_) => return false,
    };
    let slot = match kernel_handle_slot(handle) {
        Some(s) => s,
        None => return false,
    };

    let current = *(hp_ptr().add(OFF_HC_CURRENT_THREAD_ID) as *const u32);
    if current == 0 || current > MUX_OWNER_MASK {
        return false;
    }

    let evt = *hp_ptr().add(OFF_HC_EVENT_TABLE + slot as usize);
    if evt & EVT_VALID != 0 {
        if evt & EVT_MANUAL != 0 || evt & EVT_PENDING_WAKE != 0 {
            return false;
        }
        if evt & EVT_SIGNALED != 0 {
            *hp_mut().add(OFF_HC_EVENT_TABLE + slot as usize) = evt & !EVT_SIGNALED;
            write_reg32(EAX, 0);
            return true;
        }
        if timeout == 0 {
            write_reg32(EAX, WAIT_TIMEOUT);
            return true;
        }
        return false;
    }

    let mux = read_mutex_mirror(slot);
    if mux & MUX_VALID == 0 {
        return false;
    }
    if mux & MUX_HAS_WAITERS != 0 {
        return false;
    }
    if mux & MUX_ABANDONED != 0 {
        return false;
    }

    let owner = mux & MUX_OWNER_MASK;
    if owner == 0 || owner == current {
        let rec = (mux & MUX_REC_MASK) >> MUX_REC_SHIFT;
        if owner == current && rec >= MUX_REC_MAX {
            return false;
        }
        let new_owner = if owner == 0 { current } else { owner };
        let new_rec = if owner == 0 { 1 } else { rec + 1 };
        if new_rec > MUX_REC_MAX {
            return false;
        }
        let new_mux = MUX_VALID | new_owner | (new_rec << MUX_REC_SHIFT);
        write_mutex_mirror(slot, new_mux);
        write_reg32(EAX, 0);
        return true;
    }

    if timeout == 0 {
        write_reg32(EAX, WAIT_TIMEOUT);
        return true;
    }
    false
}

/// Sleep(dwMilliseconds) — stdcall(1).
/// Fast path: Sleep(0) with no other runnable threads is a pure no-op.
/// When peers exist, uses a starvation counter: only every Nth call falls
/// through to JS for actual context switch. This matches real Windows behavior
/// where Sleep(0) yields the remainder of the time quantum — often a near no-op
/// if the thread was just scheduled.
unsafe fn handle_sleep() -> bool {
    let esp = read_reg32(ESP);
    let ms = match safe_read32s(esp + 4) {
        Ok(v) => v as u32,
        Err(_) => return false,
    };

    if ms == 0 {
        let page = hp_ptr();
        let has_peers = *(page.add(OFF_HC_HAS_RUNNABLE_PEERS) as *const u32);
        if has_peers == 0 {
            // No-op: no peer thread to yield to. Return void (EAX=0 by convention).
            write_reg32(EAX, 0);
            return true;
        }

        // Peers exist — use starvation counter to avoid excessive JS transitions.
        let counter_ptr = hp_mut().add(OFF_HC_SLEEP_STARVATION_COUNTER) as *mut u32;
        let counter = *counter_ptr;
        let limit = *(page.add(OFF_HC_SLEEP_STARVATION_LIMIT) as *const u32);

        if limit > 0 && counter < limit {
            // Not yet starved — no-op in WASM, peers can wait a few more µs
            *counter_ptr = counter + 1;
            write_reg32(EAX, 0);
            return true;
        }

        // Starvation limit reached or unset → reset counter and fall through to JS
        *counter_ptr = 0;
    }

    // Non-zero sleep or starvation limit hit → fall through to JS for context switch
    false
}

/// TlsGetValue(dwTlsIndex) — stdcall(1).
/// Reads TLS slot from guest TEB memory: TEB+0x2C → TLS array ptr, then array[index*4].
/// Sets LastError to 0 on success (Windows behavior).
/// Falls through to JS for out-of-range indices or memory faults.
unsafe fn handle_tls_get_value() -> bool {
    let esp = read_reg32(ESP);
    let index = match safe_read32s(esp + 4) {
        Ok(v) => v as u32,
        Err(_) => return false,
    };

    // Only handle standard TLS slots (0..63)
    if index >= 64 {
        return false;
    }

    // Read TEB base from hypercall page
    let teb_base = *(hp_ptr().add(OFF_HC_TEB_BASE) as *const u32);
    if teb_base == 0 {
        return false;
    }

    // TEB+0x2C = ThreadLocalStoragePointer → address of TLS array
    // Use direct memory read — TEB is in identity-mapped HEAP, TLB never has these pages
    // (all guest TEB access is HLE via FS:), so safe_read32s always faults → JS fallback.
    let tls_array_ptr = memory::read32_no_mmap_check(teb_base + 0x2C) as u32;
    if tls_array_ptr == 0 {
        return false;
    }

    // Read TLS slot value (direct memory — same identity-mapped HEAP region)
    let value = memory::read32_no_mmap_check(tls_array_ptr + index * 4);

    // SetLastError(0) — TlsGetValue clears last error on success
    *(hp_mut().add(OFF_HC_LAST_ERROR) as *mut u32) = 0;
    memory::write32_no_mmap_or_dirty_check(teb_base + 0x34, 0);

    write_reg32(EAX, value);
    true
}

/// FlsGetValue(dwFlsIndex) — stdcall(1).
/// Reads FLS slot value from HYPERCALL_PAGE shadow written by JS.
/// Falls through to JS for out-of-range slots or for slots that are not allocated.
unsafe fn handle_fls_get_value() -> bool {
    let esp = read_reg32(ESP);
    let index = match safe_read32s(esp + 4) {
        Ok(v) => v as u32,
        Err(_) => return false,
    };

    if index as usize >= HC_FLS_SLOT_COUNT {
        return false;
    }

    let allocated = *(hp_ptr().add(OFF_HC_FLS_ALLOCATED + index as usize) as *const u8);
    if allocated == 0 {
        return false;
    }

    let value = *(hp_ptr().add(OFF_HC_FLS_VALUES + index as usize * 4) as *const u32);
    write_reg32(EAX, value as i32);
    true
}

// ---------------------------------------------------------------------------
// Heap arena: WASM-resident slab allocator for HeapAlloc/HeapFree
// ---------------------------------------------------------------------------

/// Map requested byte count to a size-class bin index (0–8).
/// Returns None for 0 bytes or >4KB (JS fallback).
fn size_to_bin(bytes: u32) -> Option<u32> {
    if bytes == 0 { return None; }
    if bytes <= 16 { return Some(0); }
    if bytes <= 32 { return Some(1); }
    if bytes <= 64 { return Some(2); }
    if bytes <= 128 { return Some(3); }
    if bytes <= 256 { return Some(4); }
    if bytes <= 512 { return Some(5); }
    if bytes <= 1024 { return Some(6); }
    if bytes <= 2048 { return Some(7); }
    if bytes <= 4096 { return Some(8); }
    None
}

/// Zero a block of guest memory using 4-byte writes (identity-mapped).
unsafe fn zero_block(addr: u32, size: u32) {
    let mut off = 0u32;
    while off < size {
        memory::write32_no_mmap_or_dirty_check(addr + off, 0);
        off += 4;
    }
}

/// Times a guest-corrupted slab structure (free-list next pointer or control block)
/// was caught before it could drive a raw write outside the slab. The slab free list
/// is threaded THROUGH guest heap memory, so a guest use-after-free store turns a
/// next pointer into game data; unvalidated, that pointer feeds
/// `write32_no_mmap_or_dirty_check` — whose `mem8.offset(addr as isize)` goes
/// NEGATIVE for addr >= 0x8000_0000 and lands in the wasm module's own statics
/// (observed: 4 KiB HEAP_ZERO_MEMORY zero_blocks stomping jit::DISPATCH_SLABS, where
/// a zeroed cell reads as dispatch state 0 → wrong-block entry → garbage-register AV).
static mut HC_SLAB_POISONED: u32 = 0;

#[no_mangle]
pub fn hc_slab_poisoned_count() -> u32 { unsafe { HC_SLAB_POISONED } }

/// Validate the slab bounds read from the (guest-writable) control block. Anchored on
/// the one thing the guest cannot fake: real guest memory size. Returns None when the
/// control block is corrupt — callers must fall back to the JS path.
#[inline]
unsafe fn slab_bounds(ctl: u32) -> Option<(u32, u32)> {
    let base = slab_rd(ctl, SLAB_REL_BASE);
    let end = slab_rd(ctl, SLAB_REL_END);
    // base == 0 is "slab not initialized yet", the ordinary pre-init state — NOT
    // corruption. Counting it would drown the poison signal in boot noise.
    if base == 0 {
        return Some((0, 0));
    }
    let mem_size = *crate::cpu::global_pointers::memory_size;
    if base < 0x1000 || base >= end || end > mem_size {
        return None;
    }
    Some((base, end))
}

/// A block pointer from guest-writable state (free-list next, bump cursor) is only
/// usable for RAW writes if the whole block lies inside the slab.
#[inline]
fn slab_block_ok(ptr: u32, size_class: u32, base: u32, end: u32) -> bool {
    ptr >= base + 16 && ptr.wrapping_add(size_class) <= end && ptr % 4 == 0
}

/// HeapAlloc(hHeap, dwFlags, dwBytes) — stdcall, 3 args.
/// Fast path: slab bump allocation or free-list pop for blocks ≤4KB.
/// Returns false to fall through to JS for large/zero-size/slab-exhausted cases.
unsafe fn handle_heap_alloc() -> bool {
    let page = hp_ptr();
    // The slab control block lives in GUEST RAM at hc_slab_ctl_ptr (the inline x86 stubs can
    // only address guest RAM; this page is unreachable from guest code). ctl==0 ⇒ slab off.
    let ctl = *(page.add(OFF_HC_SLAB_CTL_PTR) as *const u32);
    if ctl == 0 { return false; }
    // Bounds come from a guest-RAM control block — validate before ANY raw write
    // derives from them (see HC_SLAB_POISONED).
    let (slab_base, slab_end_checked) = match slab_bounds(ctl) {
        Some(b) => b,
        None => {
            HC_SLAB_POISONED = HC_SLAB_POISONED.wrapping_add(1);
            return false;
        }
    };
    if slab_base == 0 { return false; } // slab not initialized

    let esp = read_reg32(ESP);
    // HeapAlloc(hHeap, dwFlags, dwBytes) at ESP+4, +8, +12
    let dw_flags = match safe_read32s(esp + 8) { Ok(v) => v as u32, Err(_) => return false };
    let dw_bytes = match safe_read32s(esp + 12) { Ok(v) => v as u32, Err(_) => return false };

    if dw_bytes == 0 { return false; } // let JS set ERROR_NOT_ENOUGH_MEMORY

    let bin = match size_to_bin(dw_bytes) {
        Some(b) => b,
        None => {
            // >4KB — count fallback, let JS handle
            slab_wr(ctl, SLAB_REL_FALLBACK_COUNT, slab_rd(ctl, SLAB_REL_FALLBACK_COUNT).wrapping_add(1));
            return false;
        }
    };
    let size_class = BIN_SIZES[bin as usize];
    let zero = (dw_flags & HEAP_ZERO_MEMORY_FLAG) != 0;

    // Try free list first
    let fl_rel = SLAB_REL_FREELIST + bin * 4;
    let head = slab_rd(ctl, fl_rel);
    if head != 0 {
        // The next pointer lives in FREED GUEST MEMORY — a guest use-after-free store
        // replaces it with game data. Unvalidated it would feed the raw header write and
        // zero_block below, and a value >= 0x8000_0000 wraps below mem8 into the wasm
        // statics. A poisoned head means the whole bin list is untrustworthy: drop it
        // (real Windows would AV/corrupt the APP's heap here, never the machine) and
        // let JS serve the allocation.
        if !slab_block_ok(head, size_class, slab_base, slab_end_checked) {
            HC_SLAB_POISONED = HC_SLAB_POISONED.wrapping_add(1);
            slab_wr(ctl, fl_rel, 0);
            if bin >= SLAB_LARGE_BIN_MIN {
                slab_wr(ctl, SLAB_REL_FREELIST_TAIL + bin * 4, 0);
            }
            return false;
        }
        // Pop from free list: the link lives in the block's header zone (see SLAB_LINK).
        let next = memory::read32_no_mmap_check(head - SLAB_LINK) as u32;
        slab_wr(ctl, fl_rel, next);
        // FIFO large bins keep a tail pointer; when the list drains to empty the tail must
        // follow, or the next free would append after a stale (now live) block.
        if bin >= SLAB_LARGE_BIN_MIN && next == 0 {
            slab_wr(ctl, SLAB_REL_FREELIST_TAIL + bin * 4, 0);
        }
        // Restore the BUSY marker (free() flipped it to FREE) — mirrors the inline stub's
        // `MOV byte [EAX-3],'A'`. Without this a popped block stays FREE-marked while live,
        // so getSlabSizeForPtr (BUSY-only) reports it as not-live and a later inline free
        // sends it down .slow → the free is lost / the block is double-owned.
        memory::write32_no_mmap_or_dirty_check(head - 4, (SLAB_MAGIC | bin) as i32);
        if zero { zero_block(head, size_class); }
        write_reg32(EAX, head as i32);
        slab_wr(ctl, SLAB_REL_ALLOC_COUNT, slab_rd(ctl, SLAB_REL_ALLOC_COUNT).wrapping_add(1));
        return true;
    }

    // Bump allocate: [bump..bump+16) = header zone, [bump+16..bump+16+size_class) = user data
    let bump = slab_rd(ctl, SLAB_REL_BUMP);
    let slab_end = slab_end_checked;
    // The bump cursor is guest-RAM state too — hold it to the same standard as a
    // free-list pointer before the raw header write / zero_block below.
    if bump < slab_base || bump > slab_end {
        HC_SLAB_POISONED = HC_SLAB_POISONED.wrapping_add(1);
        return false;
    }
    let user_ptr = bump + 16;
    let new_bump = user_ptr + size_class;
    if new_bump > slab_end {
        slab_wr(ctl, SLAB_REL_FALLBACK_COUNT, slab_rd(ctl, SLAB_REL_FALLBACK_COUNT).wrapping_add(1));
        return false; // slab exhausted → JS allocates + can refill slab
    }

    // Write block header at user_ptr - 4
    memory::write32_no_mmap_or_dirty_check(user_ptr - 4, (SLAB_MAGIC | bin) as i32);
    slab_wr(ctl, SLAB_REL_BUMP, new_bump);

    if zero { zero_block(user_ptr, size_class); }

    write_reg32(EAX, user_ptr as i32);
    slab_wr(ctl, SLAB_REL_ALLOC_COUNT, slab_rd(ctl, SLAB_REL_ALLOC_COUNT).wrapping_add(1));
    true
}

/// HeapFree(hHeap, dwFlags, lpMem) — stdcall, 3 args.
/// Fast path: push freed block onto per-bin free list if it's within slab range.
unsafe fn handle_heap_free() -> bool {
    let page = hp_ptr();
    let ctl = *(page.add(OFF_HC_SLAB_CTL_PTR) as *const u32);
    if ctl == 0 { return false; }
    // Same guest-writable-bounds discipline as handle_heap_alloc.
    let (slab_base, slab_end) = match slab_bounds(ctl) {
        Some(b) => b,
        None => {
            HC_SLAB_POISONED = HC_SLAB_POISONED.wrapping_add(1);
            return false;
        }
    };
    if slab_base == 0 { return false; }

    let esp = read_reg32(ESP);
    let lp_mem = match safe_read32s(esp + 12) { Ok(v) => v as u32, Err(_) => return false };

    if lp_mem == 0 {
        write_reg32(EAX, 1); // HeapFree(NULL) = TRUE per Windows spec
        return true;
    }

    // Must be within slab with room for header
    if lp_mem < slab_base + 16 || lp_mem >= slab_end { return false; }

    // Validate slab header at [lp_mem - 4]
    let header = memory::read32_no_mmap_check(lp_mem - 4) as u32;
    if (header & 0xFFFFFF00) != SLAB_MAGIC { return false; }
    let bin = header & 0x0F;
    if bin > 8 { return false; }

    // Mark the block FREE before linking it in — mirrors the inline stub's `MOV byte [EAX-3],'F'`.
    // The BUSY-only validate above already rejected an already-FREE block, so this closes the
    // double-free hole: a second free now fails the validate and goes .slow (JS no-op) instead
    // of being pushed twice → free-list cycle → the same block handed to two owners.
    memory::write32_no_mmap_or_dirty_check(lp_mem - 4, (SLAB_MAGIC_FREE | bin) as i32);

    // Push to free list. The link goes in the header zone, NOT the user data: a real
    // heap leaves a freed block's contents intact (LFH), and titles that read a freed
    // object's vtable before the parent forgets it depend on that.
    let fl_rel = SLAB_REL_FREELIST + bin * 4;
    if bin >= SLAB_LARGE_BIN_MIN {
        // FIFO tail-append (see SLAB_LARGE_BIN_MIN): the freed block becomes the new tail
        // with no successor; the old tail (if any) links forward to it.
        let tl_rel = SLAB_REL_FREELIST_TAIL + bin * 4;
        let tail = slab_rd(ctl, tl_rel);
        memory::write32_no_mmap_or_dirty_check(lp_mem - SLAB_LINK, 0);
        if tail == 0 {
            slab_wr(ctl, fl_rel, lp_mem);
        } else {
            memory::write32_no_mmap_or_dirty_check(tail - SLAB_LINK, lp_mem as i32);
        }
        slab_wr(ctl, tl_rel, lp_mem);
    } else {
        let old_head = slab_rd(ctl, fl_rel);
        memory::write32_no_mmap_or_dirty_check(lp_mem - SLAB_LINK, old_head as i32);
        slab_wr(ctl, fl_rel, lp_mem);
    }

    slab_wr(ctl, SLAB_REL_FREE_COUNT, slab_rd(ctl, SLAB_REL_FREE_COUNT).wrapping_add(1));
    write_reg32(EAX, 1); // TRUE
    true
}

/// __RTDynamicCast(void* inptr, int vfDelta, void* srcType, void* targetType, int isReference)
/// — cdecl, 5 args. Primary RTTI cast path; falls through to JS only for bad_cast (reference
/// cast failure on a non-null pointer, which must throw via terminateProcess).
unsafe fn handle_rt_dynamic_cast() -> bool {
    let esp = read_reg32(ESP);
    let inptr = match safe_read32s(esp + 4) {
        Ok(v) => v as u32,
        Err(_) => return false,
    };
    let vf_delta = match safe_read32s(esp + 8) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let src_type = match safe_read32s(esp + 12) {
        Ok(v) => v as u32,
        Err(_) => return false,
    };
    let target_type = match safe_read32s(esp + 16) {
        Ok(v) => v as u32,
        Err(_) => return false,
    };
    let is_reference = match safe_read32s(esp + 20) {
        Ok(v) => v,
        Err(_) => return false,
    };

    match rt_dynamic_cast(inptr, vf_delta, src_type, target_type, is_reference) {
        RtDynamicCastResult::Success(addr) => {
            write_reg32(EAX, addr as i32);
            true
        }
        RtDynamicCastResult::FailNull => {
            write_reg32(EAX, 0);
            true
        }
        RtDynamicCastResult::FailBadCast => false,
    }
}

