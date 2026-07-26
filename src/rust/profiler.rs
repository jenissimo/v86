#[allow(non_camel_case_types)]
pub enum stat {
    COMPILE,
    COMPILE_SKIPPED_NO_NEW_ENTRY_POINTS,
    COMPILE_WRONG_ADDRESS_SPACE,
    COMPILE_CUT_OFF_AT_END_OF_PAGE,
    COMPILE_WITH_LOOP_SAFETY,
    COMPILE_PAGE,
    COMPILE_BASIC_BLOCK,
    COMPILE_DUPLICATED_BASIC_BLOCK,
    COMPILE_WASM_BLOCK,
    COMPILE_WASM_LOOP,
    COMPILE_DISPATCHER,
    COMPILE_ENTRY_POINT,
    COMPILE_WASM_TOTAL_BYTES,

    RUN_INTERPRETED,
    RUN_INTERPRETED_NEW_PAGE,
    RUN_INTERPRETED_PAGE_HAS_CODE,
    RUN_INTERPRETED_PAGE_HAS_ENTRY_AFTER_PAGE_WALK,
    RUN_INTERPRETED_NEAR_END_OF_PAGE,
    RUN_INTERPRETED_DIFFERENT_STATE,
    RUN_INTERPRETED_DIFFERENT_STATE_CPL3,
    RUN_INTERPRETED_DIFFERENT_STATE_FLAT,
    RUN_INTERPRETED_DIFFERENT_STATE_IS32,
    RUN_INTERPRETED_DIFFERENT_STATE_SS32,
    RUN_INTERPRETED_MISSED_COMPILED_ENTRY_RUN_INTERPRETED,
    RUN_INTERPRETED_STEPS,

    RUN_FROM_CACHE,
    RUN_FROM_CACHE_STEPS,

    DIRECT_EXIT,
    INDIRECT_JUMP,
    INDIRECT_JUMP_NO_ENTRY,
    NORMAL_PAGE_CHANGE,
    NORMAL_FALLTHRU,
    NORMAL_FALLTHRU_WITH_TARGET_BLOCK,
    NORMAL_BRANCH,
    NORMAL_BRANCH_WITH_TARGET_BLOCK,
    CONDITIONAL_JUMP,
    CONDITIONAL_JUMP_PAGE_CHANGE,
    CONDITIONAL_JUMP_EXIT,
    CONDITIONAL_JUMP_FALLTHRU,
    CONDITIONAL_JUMP_FALLTHRU_WITH_TARGET_BLOCK,
    CONDITIONAL_JUMP_BRANCH,
    CONDITIONAL_JUMP_BRANCH_WITH_TARGET_BLOCK,
    DISPATCHER_SMALL,
    DISPATCHER_LARGE,
    LOOP,

    LOOP_SAFETY,

    CONDITION_OPTIMISED,
    CONDITION_UNOPTIMISED,
    CONDITION_UNOPTIMISED_PF,
    CONDITION_UNOPTIMISED_UNHANDLED_L,
    CONDITION_UNOPTIMISED_UNHANDLED_LE,

    FAILED_PAGE_CHANGE,

    SAFE_READ_FAST,
    SAFE_READ_SLOW_PAGE_CROSSED,
    SAFE_READ_SLOW_NOT_VALID,
    SAFE_READ_SLOW_NOT_USER,
    SAFE_READ_SLOW_IN_MAPPED_RANGE,

    SAFE_WRITE_FAST,
    SAFE_WRITE_SLOW_PAGE_CROSSED,
    SAFE_WRITE_SLOW_NOT_VALID,
    SAFE_WRITE_SLOW_NOT_USER,
    SAFE_WRITE_SLOW_IN_MAPPED_RANGE,
    SAFE_WRITE_SLOW_READ_ONLY,
    SAFE_WRITE_SLOW_HAS_CODE,

    SAFE_READ_WRITE_FAST,
    SAFE_READ_WRITE_SLOW_PAGE_CROSSED,
    SAFE_READ_WRITE_SLOW_NOT_VALID,
    SAFE_READ_WRITE_SLOW_NOT_USER,
    SAFE_READ_WRITE_SLOW_IN_MAPPED_RANGE,
    SAFE_READ_WRITE_SLOW_READ_ONLY,
    SAFE_READ_WRITE_SLOW_HAS_CODE,

    PAGE_FAULT,
    TLB_MISS,

    MAIN_LOOP,
    MAIN_LOOP_IDLE,
    DO_MANY_CYCLES,
    CYCLE_INTERNAL,

    INVALIDATE_ALL_MODULES_NO_FREE_WASM_INDICES,
    INVALIDATE_MODULE_WRITTEN_WHILE_COMPILED,
    INVALIDATE_MODULE_UNUSED_AFTER_OVERWRITE,
    INVALIDATE_MODULE_DIRTY_PAGE,

    INVALIDATE_PAGE_HAD_CODE,
    INVALIDATE_PAGE_HAD_ENTRY_POINTS,
    DIRTY_PAGE_DID_NOT_HAVE_CODE,

    RUN_FROM_CACHE_EXIT_SAME_PAGE,
    RUN_FROM_CACHE_EXIT_NEAR_END_OF_PAGE,
    RUN_FROM_CACHE_EXIT_DIFFERENT_PAGE,

    CLEAR_TLB,
    FULL_CLEAR_TLB,
    TLB_FULL,
    TLB_GLOBAL_FULL,

    MODRM_SIMPLE_REG,
    MODRM_SIMPLE_REG_WITH_OFFSET,
    MODRM_SIMPLE_CONST_OFFSET,
    MODRM_COMPLEX,

    SEG_OFFSET_OPTIMISED,
    SEG_OFFSET_NOT_OPTIMISED,
    SEG_OFFSET_NOT_OPTIMISED_ES,
    SEG_OFFSET_NOT_OPTIMISED_FS,
    SEG_OFFSET_NOT_OPTIMISED_GS,
    SEG_OFFSET_NOT_OPTIMISED_NOT_FLAT,

    FPU_RELAXED_HIT,
    FPU_RELAXED_FALLBACK,

    // Block-chaining dispatch characterisation.
    // These are ALWAYS-ON counters (incremented via increment_fixed_i64 in compiled code and
    // stat_increment_always at runtime), gated only by the jit::DISPATCH_STATS toggle so they
    // work WITHOUT the `profiler` feature. Read them through profiler_dispatch_stat_get below
    // (NOT profiler_stat_get, which returns 0 unless the profiler feature is compiled in).
    //
    //   BLOCK_EXECUTION       — every compiled basic-block execution (entry + intra-module).
    //   MODULE_REENTRY        — every return to main_loop after a compiled module ran.
    //   MODULE_EXIT_CHAINABLE — module exit whose successor eip is a compile-time constant
    //                           (direct JMP / conditional JMP leaving the module). Phase-1
    //                           tail-call chaining can target exactly these.
    //   MODULE_EXIT_DYNAMIC   — module exit whose eip is computed at runtime by the block's
    //                           terminating instruction (ret / int / iret / far jmp, sti).
    //   MODULE_EXIT_INDIRECT  — indirect jmp/call (AbsoluteEip) whose target is not in this
    //                           module. Not statically chainable.
    //
    // Derived in the readout: INTRA_MODULE_EDGE = BLOCK_EXECUTION - MODULE_REENTRY, and the
    // headline number CHAINABLE_FRACTION = MODULE_EXIT_CHAINABLE / MODULE_REENTRY.
    BLOCK_EXECUTION,
    MODULE_REENTRY,
    MODULE_EXIT_CHAINABLE,
    MODULE_EXIT_DYNAMIC,
    MODULE_EXIT_INDIRECT,
    MODULE_CHAINED_EDGE,
    MODULE_CHAIN_BUDGET_EXIT,
    MODULE_CHAIN_MISS,

    // Dead-flag elision (always-on, read via profiler_dispatch_stat_get 8/9). Compile-time counts:
    //   DEAD_FLAG_ELISION_CANDIDATE — instructions that fully overwrite flags (walk entered).
    //   DEAD_FLAG_ELIDED            — of those, instructions whose flag writes were elided
    //                                (flags proven dead via the intra-block non-faulting walk).
    DEAD_FLAG_ELISION_CANDIDATE,
    DEAD_FLAG_ELIDED,

    // RET/AbsoluteEip dynamic chaining (always-on, read via profiler_dispatch_stat_get 10-12,
    // gated by jit::DISPATCH_STATS like the block-chaining counters above).
    //   ABSEIP_DISPATCH — every jit_find_cache_entry_in_page call (the in-module AbsoluteEip
    //                     re-dispatch). The vision doc's gate-1 rate ("RET-dispatch pressure"):
    //                     if this is low in-race, RET chaining has a small ceiling.
    //   RET_CHAIN_HIT   — AbsoluteEip exit chained via tail-call (JIT_RET_CHAINING).
    //   RET_CHAIN_MISS  — AbsoluteEip exit that fell back to main_loop after a chain attempt.
    ABSEIP_DISPATCH,
    RET_CHAIN_HIT,
    RET_CHAIN_MISS,

    // Shape-flag RUNTIME census (read via profiler_dispatch_stat_get 13-17, emitted only
    // while jit::DISPATCH_STATS was on at compile time). The compile-site counters next to
    // these flags answer "was the shape emitted"; only these answer "does it pay off",
    // because both caches are bets that a later access finds the slot still valid.
    //   X87_CACHE_HIT/FILL  — relaxed-x87 ST read served from the local vs re-read from
    //                         fpu_st memory. FILL is the bet lost.
    //   X87_CACHE_INVALIDATE— an executed invalidate-all (any x87/MMX op the wrappers do
    //                         not keep coherent, incl. every TOP change).
    //   PUSH_RUN_HIT/FILL   — push32 store that reused the previous push's TLB entry vs
    //                         one that had to do the lookup and refill the cache.
    X87_CACHE_HIT,
    X87_CACHE_FILL,
    X87_CACHE_INVALIDATE,
    PUSH_RUN_HIT,
    PUSH_RUN_FILL,
}

#[allow(non_upper_case_globals)]
pub static mut stat_array: [u64; 500] = [0; 500];

pub fn stat_increment(stat: stat) { stat_increment_by(stat, 1); }

pub fn stat_increment_by(stat: stat, by: u64) {
    if cfg!(feature = "profiler") {
        unsafe { stat_array[stat as usize] += by }
    }
}

pub fn stat_increment_always(stat: stat) { unsafe { stat_array[stat as usize] += 1 } }

#[no_mangle]
pub fn profiler_init() {
    unsafe {
        #[allow(static_mut_refs)]
        for x in stat_array.iter_mut() {
            *x = 0
        }
    }
}

#[no_mangle]
pub fn profiler_stat_get(stat: stat) -> f64 {
    if cfg!(feature = "profiler") {
        unsafe { stat_array[stat as usize] as f64 }
    }
    else {
        0.0
    }
}

#[no_mangle]
pub fn profiler_is_enabled() -> bool { cfg!(feature = "profiler") }

#[no_mangle]
pub fn profiler_fpu_relaxed_hit_get() -> f64 {
    unsafe { stat_array[stat::FPU_RELAXED_HIT as usize] as f64 }
}

#[no_mangle]
pub fn profiler_fpu_relaxed_fallback_get() -> f64 {
    unsafe { stat_array[stat::FPU_RELAXED_FALLBACK as usize] as f64 }
}

// Block-chaining readout. Reads the dispatch-characterisation counters directly out of
// stat_array regardless of the `profiler` feature (unlike profiler_stat_get). Index order:
// 0=BLOCK_EXECUTION 1=MODULE_REENTRY 2=MODULE_EXIT_CHAINABLE 3=MODULE_EXIT_DYNAMIC
// 4=MODULE_EXIT_INDIRECT 5=MODULE_CHAINED_EDGE 6=MODULE_CHAIN_BUDGET_EXIT 7=MODULE_CHAIN_MISS.
#[no_mangle]
pub fn profiler_dispatch_stat_get(index: u32) -> f64 {
    let stat = match index {
        0 => stat::BLOCK_EXECUTION,
        1 => stat::MODULE_REENTRY,
        2 => stat::MODULE_EXIT_CHAINABLE,
        3 => stat::MODULE_EXIT_DYNAMIC,
        4 => stat::MODULE_EXIT_INDIRECT,
        5 => stat::MODULE_CHAINED_EDGE,
        6 => stat::MODULE_CHAIN_BUDGET_EXIT,
        7 => stat::MODULE_CHAIN_MISS,
        8 => stat::DEAD_FLAG_ELISION_CANDIDATE,
        9 => stat::DEAD_FLAG_ELIDED,
        10 => stat::ABSEIP_DISPATCH,
        11 => stat::RET_CHAIN_HIT,
        12 => stat::RET_CHAIN_MISS,
        13 => stat::X87_CACHE_HIT,
        14 => stat::X87_CACHE_FILL,
        15 => stat::X87_CACHE_INVALIDATE,
        16 => stat::PUSH_RUN_HIT,
        17 => stat::PUSH_RUN_FILL,
        _ => return 0.0,
    };
    unsafe { stat_array[stat as usize] as f64 }
}
