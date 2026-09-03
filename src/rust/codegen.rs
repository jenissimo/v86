use crate::cpu::cpu::{
    tlb_data, FLAGS_ALL, FLAG_CARRY, FLAG_OVERFLOW, FLAG_PARITY, FLAG_SIGN, FLAG_ZERO, OPSIZE_16,
    OPSIZE_32, OPSIZE_8, TLB_GLOBAL, TLB_HAS_CODE, TLB_NO_USER, TLB_READONLY, TLB_VALID,
};
use crate::cpu::global_pointers;
use crate::cpu::memory;
use crate::jit::{Instruction, InstructionOperand, InstructionOperandDest, JitContext};
use crate::modrm;
use crate::modrm::ModrmByte;
use crate::opstats;
use crate::profiler;
use crate::regs;
use crate::wasmgen::wasm_builder::{
    WasmBuilder, WasmLocal, WasmLocalI64, HINT_GROUP_MEM, HINT_GROUP_X87, HINT_LIKELY,
    HINT_UNLIKELY,
};

pub fn gen_add_cs_offset(ctx: &mut JitContext) {
    if !ctx.cpu.has_flat_segmentation() {
        ctx.builder
            .load_fixed_i32(global_pointers::get_seg_offset(regs::CS));
        ctx.builder.add_i32();
    }
}

pub fn gen_get_eip(builder: &mut WasmBuilder) {
    builder.load_fixed_i32(global_pointers::instruction_pointer as u32);
}

pub fn gen_mark_fpu_simd_dirty(builder: &mut WasmBuilder) {
    builder.const_i32(global_pointers::fpu_simd_dirty as i32);
    builder.const_i32(1);
    builder.store_u8(0);
}

pub fn gen_mark_fpu_simd_dirty_once(ctx: &mut JitContext) {
    if ctx.fpu_simd_dirty_marked {
        return;
    }
    gen_mark_fpu_simd_dirty(ctx.builder);
    ctx.fpu_simd_dirty_marked = true;
}

pub fn gen_set_eip_to_after_current_instruction(ctx: &mut JitContext) {
    ctx.builder
        .const_i32(global_pointers::instruction_pointer as i32);
    gen_get_eip(ctx.builder);
    ctx.builder.const_i32(!0xFFF);
    ctx.builder.and_i32();
    ctx.builder.const_i32(ctx.cpu.eip as i32 & 0xFFF);
    ctx.builder.or_i32();
    ctx.builder.store_aligned_i32(0);
}

pub fn gen_set_previous_eip_offset_from_eip_with_low_bits(
    builder: &mut WasmBuilder,
    low_bits: i32,
) {
    // previous_ip = instruction_pointer & ~0xFFF | low_bits;
    dbg_assert!(low_bits & !0xFFF == 0);
    builder.const_i32(global_pointers::previous_ip as i32);
    gen_get_eip(builder);
    builder.const_i32(!0xFFF);
    builder.and_i32();
    builder.const_i32(low_bits);
    builder.or_i32();
    builder.store_aligned_i32(0);
}

pub fn gen_set_eip_low_bits(builder: &mut WasmBuilder, low_bits: i32) {
    // instruction_pointer = instruction_pointer & ~0xFFF | low_bits;
    dbg_assert!(low_bits & !0xFFF == 0);
    builder.const_i32(global_pointers::instruction_pointer as i32);
    gen_get_eip(builder);
    builder.const_i32(!0xFFF);
    builder.and_i32();
    builder.const_i32(low_bits);
    builder.or_i32();
    builder.store_aligned_i32(0);
}

pub fn gen_set_eip_low_bits_and_jump_rel32(builder: &mut WasmBuilder, low_bits: i32, n: i32) {
    // instruction_pointer = (instruction_pointer & ~0xFFF | low_bits) + n;
    dbg_assert!(low_bits & !0xFFF == 0);
    builder.const_i32(global_pointers::instruction_pointer as i32);
    gen_get_eip(builder);
    builder.const_i32(!0xFFF);
    builder.and_i32();
    builder.const_i32(low_bits);
    builder.or_i32();
    if n != 0 {
        builder.const_i32(n);
        builder.add_i32();
    }
    builder.store_aligned_i32(0);
}

pub fn gen_relative_jump(builder: &mut WasmBuilder, n: i32) {
    // add n to instruction_pointer
    if n != 0 {
        builder.const_i32(global_pointers::instruction_pointer as i32);
        gen_get_eip(builder);
        builder.const_i32(n);
        builder.add_i32();
        builder.store_aligned_i32(0);
    }
}

pub fn gen_page_switch_check(
    ctx: &mut JitContext,
    next_block_addr: u32,
    last_instruction_addr: u32,
) {
    // After switching a page while in jitted code, check if the page mapping still holds

    gen_get_eip(ctx.builder);
    let address_local = ctx.builder.set_new_local();
    gen_get_phys_eip_plus_mem(ctx, &address_local);
    ctx.builder.free_local(address_local);

    ctx.builder
        .const_i32(next_block_addr as i32 + unsafe { memory::mem8 } as i32);
    ctx.builder.ne_i32();

    if cfg!(debug_assertions) {
        ctx.builder.if_void();
        gen_profiler_stat_increment(ctx.builder, profiler::stat::FAILED_PAGE_CHANGE);
        gen_debug_track_jit_exit(ctx.builder, last_instruction_addr);
        ctx.builder.br(ctx.exit_label);
        ctx.builder.block_end();
    }
    else {
        // The page mapping surviving the switch is the overwhelmingly common case;
        // a changed mapping means we leave the module entirely.
        ctx.builder.br_if_hinted(ctx.exit_label, HINT_GROUP_MEM, HINT_UNLIKELY);
    }
}

pub fn gen_update_instruction_counter(ctx: &mut JitContext) {
    ctx.builder
        .const_i32(global_pointers::instruction_counter as i32);
    ctx.builder
        .load_fixed_i32(global_pointers::instruction_counter as u32);
    ctx.builder.get_local(&ctx.instruction_counter);
    ctx.builder.add_i32();
    ctx.builder.store_aligned_i32(0);
}

pub fn gen_get_reg8(ctx: &mut JitContext, r: u32) {
    match r {
        regs::AL | regs::CL | regs::DL | regs::BL => {
            ctx.builder.get_local(&ctx.register_locals[r as usize]);
            ctx.builder.const_i32(0xFF);
            ctx.builder.and_i32();
        },
        regs::AH | regs::CH | regs::DH | regs::BH => {
            ctx.builder
                .get_local(&ctx.register_locals[(r - 4) as usize]);
            ctx.builder.const_i32(8);
            ctx.builder.shr_u_i32();
            ctx.builder.const_i32(0xFF);
            ctx.builder.and_i32();
        },
        _ => assert!(false),
    }
}

/// Return a new local referencing one of the 8 bit registers or a direct reference to one of the
/// register locals. Higher bits might be garbage (suitable for gen_cmp8 etc.). Must be freed with
/// gen_free_reg8_or_alias.
pub fn gen_get_reg8_or_alias_to_reg32(ctx: &mut JitContext, r: u32) -> WasmLocal {
    match r {
        regs::AL | regs::CL | regs::DL | regs::BL => ctx.register_locals[r as usize].unsafe_clone(),
        regs::AH | regs::CH | regs::DH | regs::BH => {
            ctx.builder
                .get_local(&ctx.register_locals[(r - 4) as usize]);
            ctx.builder.const_i32(8);
            ctx.builder.shr_u_i32();
            ctx.builder.set_new_local()
        },
        _ => panic!(),
    }
}

pub fn gen_free_reg8_or_alias(ctx: &mut JitContext, r: u32, local: WasmLocal) {
    match r {
        regs::AL | regs::CL | regs::DL | regs::BL => {},
        regs::AH | regs::CH | regs::DH | regs::BH => ctx.builder.free_local(local),
        _ => panic!(),
    }
}

pub fn gen_get_reg16(ctx: &mut JitContext, r: u32) {
    ctx.builder.get_local(&ctx.register_locals[r as usize]);
    ctx.builder.const_i32(0xFFFF);
    ctx.builder.and_i32();
}

pub fn gen_get_reg32(ctx: &mut JitContext, r: u32) {
    ctx.builder.get_local(&ctx.register_locals[r as usize]);
}

pub fn gen_set_reg8(ctx: &mut JitContext, r: u32) {
    match r {
        regs::AL | regs::CL | regs::DL | regs::BL => {
            // reg32[r] = stack_value & 0xFF | reg32[r] & ~0xFF
            ctx.builder.const_i32(0xFF);
            ctx.builder.and_i32();

            ctx.builder.get_local(&ctx.register_locals[r as usize]);
            ctx.builder.const_i32(!0xFF);
            ctx.builder.and_i32();

            ctx.builder.or_i32();
            ctx.builder.set_local(&ctx.register_locals[r as usize]);
        },
        regs::AH | regs::CH | regs::DH | regs::BH => {
            // reg32[r] = stack_value << 8 & 0xFF00 | reg32[r] & ~0xFF00
            ctx.builder.const_i32(8);
            ctx.builder.shl_i32();
            ctx.builder.const_i32(0xFF00);
            ctx.builder.and_i32();

            ctx.builder
                .get_local(&ctx.register_locals[(r - 4) as usize]);
            ctx.builder.const_i32(!0xFF00);
            ctx.builder.and_i32();

            ctx.builder.or_i32();
            ctx.builder
                .set_local(&ctx.register_locals[(r - 4) as usize]);
        },
        _ => assert!(false),
    }
}

pub fn gen_set_reg8_unmasked(ctx: &mut JitContext, r: u32) {
    if cfg!(debug_assertions) {
        let val = ctx.builder.set_new_local();
        ctx.builder.get_local(&val);
        ctx.builder.const_i32(!0xFF);
        ctx.builder.and_i32();
        ctx.builder.if_void();
        ctx.builder.unreachable();
        ctx.builder.block_end();
        ctx.builder.get_local(&val);
        ctx.builder.free_local(val);
    }

    match r {
        regs::AL | regs::CL | regs::DL | regs::BL => {
            // reg32[r] = stack_value | reg32[r] & ~0xFF
            ctx.builder.get_local(&ctx.register_locals[r as usize]);
            ctx.builder.const_i32(!0xFF);
            ctx.builder.and_i32();

            ctx.builder.or_i32();
            ctx.builder.set_local(&ctx.register_locals[r as usize]);
        },
        regs::AH | regs::CH | regs::DH | regs::BH => {
            // reg32[r] = stack_value << 8 | reg32[r] & ~0xFF00
            ctx.builder.const_i32(8);
            ctx.builder.shl_i32();
            ctx.builder.const_i32(0xFF00);
            ctx.builder.and_i32();

            ctx.builder
                .get_local(&ctx.register_locals[(r - 4) as usize]);
            ctx.builder.const_i32(!0xFF00);
            ctx.builder.and_i32();

            ctx.builder.or_i32();
            ctx.builder
                .set_local(&ctx.register_locals[(r - 4) as usize]);
        },
        _ => assert!(false),
    }
}

pub fn gen_set_reg16(ctx: &mut JitContext, r: u32) {
    gen_set_reg16_local(ctx.builder, &ctx.register_locals[r as usize]);
}

pub fn gen_set_reg16_unmasked(ctx: &mut JitContext, r: u32) {
    if cfg!(debug_assertions) {
        let val = ctx.builder.set_new_local();
        ctx.builder.get_local(&val);
        ctx.builder.const_i32(!0xFFFF);
        ctx.builder.and_i32();
        ctx.builder.if_void();
        ctx.builder.unreachable();
        ctx.builder.block_end();
        ctx.builder.get_local(&val);
        ctx.builder.free_local(val);
    }

    ctx.builder.get_local(&ctx.reg(r));
    ctx.builder.const_i32(!0xFFFF);
    ctx.builder.and_i32();
    ctx.builder.or_i32();
    ctx.builder.set_local(&ctx.reg(r));
}

pub fn gen_set_reg16_local(builder: &mut WasmBuilder, local: &WasmLocal) {
    // reg32[r] = v & 0xFFFF | reg32[r] & ~0xFFFF
    builder.const_i32(0xFFFF);
    builder.and_i32();
    builder.get_local(local);
    builder.const_i32(!0xFFFF);
    builder.and_i32();
    builder.or_i32();
    builder.set_local(local);
}

pub fn gen_set_reg32(ctx: &mut JitContext, r: u32) {
    ctx.builder.set_local(&ctx.register_locals[r as usize]);
}

pub fn decr_exc_asize(ctx: &mut JitContext) {
    gen_get_reg32(ctx, regs::ECX);
    ctx.builder.const_i32(1);
    ctx.builder.sub_i32();
    if ctx.cpu.asize_32() {
        gen_set_reg32(ctx, regs::ECX);
    }
    else {
        gen_set_reg16(ctx, regs::CX);
    }
}

pub fn gen_read_reg_xmm128_into_scratch(ctx: &mut JitContext, r: u32) {
    ctx.builder
        .const_i32(global_pointers::sse_scratch_register as i32);
    let dest = global_pointers::get_reg_xmm_offset(r);
    ctx.builder.const_i32(dest as i32);
    ctx.builder.load_aligned_i64(0);
    ctx.builder.store_aligned_i64(0);

    ctx.builder
        .const_i32(global_pointers::sse_scratch_register as i32 + 8);
    let dest = global_pointers::get_reg_xmm_offset(r) + 8;
    ctx.builder.const_i32(dest as i32);
    ctx.builder.load_aligned_i64(0);
    ctx.builder.store_aligned_i64(0);
}

pub fn gen_get_sreg(ctx: &mut JitContext, r: u32) {
    ctx.builder
        .load_fixed_u16(global_pointers::get_sreg_offset(r))
}

pub fn gen_get_ss_offset(ctx: &mut JitContext) {
    ctx.builder
        .load_fixed_i32(global_pointers::get_seg_offset(regs::SS));
}

pub fn gen_get_flags(builder: &mut WasmBuilder) {
    if builder.flag_local_get(4) {
        return; // FLAG_LOCAL_FLAGS (const defined below)
    }
    builder.load_fixed_i32(global_pointers::flags as u32);
}
fn gen_get_flags_changed(builder: &mut WasmBuilder) {
    if builder.flag_local_get(3) {
        return; // FLAG_LOCAL_FLAGS_CHANGED
    }
    builder.load_fixed_i32(global_pointers::flags_changed as u32);
}
fn gen_get_last_result(builder: &mut WasmBuilder, previous_instruction: &Instruction) {
    match previous_instruction {
        Instruction::Add {
            dest: InstructionOperandDest::WasmLocal(l),
            opsize: OPSIZE_32,
            ..
        }
        | Instruction::AdcSbb {
            dest: InstructionOperandDest::WasmLocal(l),
            opsize: OPSIZE_32,
            ..
        }
        | Instruction::Sub {
            dest: InstructionOperandDest::WasmLocal(l),
            opsize: OPSIZE_32,
            ..
        }
        | Instruction::Bitwise {
            dest: InstructionOperandDest::WasmLocal(l),
            opsize: OPSIZE_32,
        }
        | Instruction::NonZeroShift {
            dest: InstructionOperandDest::WasmLocal(l),
            opsize: OPSIZE_32,
        } => builder.get_local(&l),
        Instruction::Cmp {
            dest: InstructionOperandDest::WasmLocal(l),
            source,
            opsize: OPSIZE_32,
        } => {
            if source.is_zero() {
                builder.get_local(&l)
            }
            else {
                gen_load_last_result(builder)
            }
        },
        _ => gen_load_last_result(builder),
    }
}
fn gen_load_last_result(builder: &mut WasmBuilder) {
    if builder.flag_local_get(1) {
        return; // FLAG_LOCAL_LAST_RESULT
    }
    builder.load_fixed_i32(global_pointers::last_result as u32);
}
fn gen_get_last_op_size(builder: &mut WasmBuilder) {
    if builder.flag_local_get(2) {
        return; // FLAG_LOCAL_LAST_OP_SIZE
    }
    builder.load_fixed_i32(global_pointers::last_op_size as u32);
}
fn gen_get_last_op1(builder: &mut WasmBuilder, previous_instruction: &Instruction) {
    match previous_instruction {
        Instruction::Cmp {
            dest: InstructionOperandDest::WasmLocal(l),
            source: _,
            opsize: OPSIZE_32,
        } => builder.get_local(&l),
        _ => {
            if builder.flag_local_get(0) {
                return; // FLAG_LOCAL_LAST_OP1
            }
            builder.load_fixed_i32(global_pointers::last_op1 as u32)
        },
    }
}

pub fn gen_readable_or_pagefault(ctx: &mut JitContext, address_local: &WasmLocal, size: i32) {
    ctx.builder.get_local(address_local);
    ctx.builder.const_i32(size);
    ctx.builder
        .const_i32(ctx.start_of_current_instruction as i32 & 0xFFF);
    ctx.builder.call_fn3_ret("readable_or_pagefault_jit");
    if cfg!(feature = "profiler") {
        ctx.builder.if_void();
        gen_debug_track_jit_exit(ctx.builder, ctx.start_of_current_instruction);
        ctx.builder.br(ctx.exit_with_fault_label);
        ctx.builder.block_end();
    }
    else {
        ctx.builder
            .br_if_hinted(ctx.exit_with_fault_label, HINT_GROUP_MEM, HINT_UNLIKELY);
    }
}

pub fn gen_writable_or_pagefault(ctx: &mut JitContext, address_local: &WasmLocal, size: i32) {
    ctx.builder.get_local(address_local);
    ctx.builder.const_i32(size);
    ctx.builder
        .const_i32(ctx.start_of_current_instruction as i32 & 0xFFF);
    ctx.builder.call_fn3_ret("writable_or_pagefault_jit");
    if cfg!(feature = "profiler") {
        ctx.builder.if_void();
        gen_debug_track_jit_exit(ctx.builder, ctx.start_of_current_instruction);
        ctx.builder.br(ctx.exit_with_fault_label);
        ctx.builder.block_end();
    }
    else {
        ctx.builder
            .br_if_hinted(ctx.exit_with_fault_label, HINT_GROUP_MEM, HINT_UNLIKELY);
    }
}

/// sign-extend a byte value on the stack and leave it on the stack
pub fn sign_extend_i8(builder: &mut WasmBuilder) {
    builder.const_i32(24);
    builder.shl_i32();
    builder.const_i32(24);
    builder.shr_s_i32();
}

/// sign-extend a two byte value on the stack and leave it on the stack
pub fn sign_extend_i16(builder: &mut WasmBuilder) {
    builder.const_i32(16);
    builder.shl_i32();
    builder.const_i32(16);
    builder.shr_s_i32();
}

pub fn gen_fn0_const(builder: &mut WasmBuilder, name: &str) { builder.call_fn0(name) }
pub fn gen_fn1_const(builder: &mut WasmBuilder, name: &str, arg0: u32) {
    builder.const_i32(arg0 as i32);
    builder.call_fn1(name);
}
pub fn gen_fn2_const(builder: &mut WasmBuilder, name: &str, arg0: u32, arg1: u32) {
    builder.const_i32(arg0 as i32);
    builder.const_i32(arg1 as i32);
    builder.call_fn2(name);
}

// helper functions for gen/generate_jit.js
pub fn gen_modrm_fn0(builder: &mut WasmBuilder, name: &str) {
    // generates: fn( _ )
    builder.call_fn1(name);
}
pub fn gen_modrm_fn1(builder: &mut WasmBuilder, name: &str, arg0: u32) {
    // generates: fn( _, arg0 )
    builder.const_i32(arg0 as i32);
    builder.call_fn2(name);
}

pub fn gen_modrm_resolve(ctx: &mut JitContext, modrm_byte: ModrmByte) {
    modrm::gen(ctx, modrm_byte, 0)
}
pub fn gen_modrm_resolve_with_local(
    ctx: &mut JitContext,
    modrm_byte: ModrmByte,
    gen: &dyn Fn(&mut JitContext, &WasmLocal),
) {
    if let Some(r) = modrm::get_as_reg_index_if_possible(ctx, &modrm_byte) {
        gen(ctx, &ctx.reg(r));
    }
    else {
        gen_modrm_resolve(ctx, modrm_byte);
        let address = ctx.builder.set_new_local();
        gen(ctx, &address);
        ctx.builder.free_local(address);
    }
}
pub fn gen_modrm_resolve_with_esp_offset(
    ctx: &mut JitContext,
    modrm_byte: ModrmByte,
    esp_offset: i32,
) {
    modrm::gen(ctx, modrm_byte, esp_offset)
}

pub fn gen_set_reg8_r(ctx: &mut JitContext, dest: u32, src: u32) {
    // generates: reg8[r_dest] = reg8[r_src]
    if src != dest {
        gen_get_reg8(ctx, src);
        gen_set_reg8_unmasked(ctx, dest);
    }
}
pub fn gen_set_reg16_r(ctx: &mut JitContext, dest: u32, src: u32) {
    // generates: reg16[r_dest] = reg16[r_src]
    if src != dest {
        gen_get_reg16(ctx, src);
        gen_set_reg16_unmasked(ctx, dest);
    }
}
pub fn gen_set_reg32_r(ctx: &mut JitContext, dest: u32, src: u32) {
    // generates: reg32[r_dest] = reg32[r_src]
    if src != dest {
        gen_get_reg32(ctx, src);
        gen_set_reg32(ctx, dest);
    }
}

// ---------------------------------------------------------------------------
// Stack fastmem CEILING EXPERIMENT (docs/performance/sota-roadmap/02)
// ---------------------------------------------------------------------------
//
// Item 02 proposes replacing the TLB lookup on ESP/EBP-relative accesses with ONE guard per
// compiled unit. Before writing that guard it prescribes measuring the ceiling: what the
// same accesses cost with NO check at all. The difference between this and the guarded
// version is the budget the guard has to fit inside; if the ceiling is small, the item
// closes without the guard ever being written.
//
// This is that measurement and NOTHING else. It is UNSOUND by construction: it drops the
// present/writable/has-code checks, so a decommitted stack page is read as garbage instead
// of faulting, and a store into a guarded page corrupts memory silently. It exists to be
// switched on for a synthetic fixture (tools/guestbench) and switched off again.
//
// Two conditions make the raw form equivalent to the guarded one on the paths it takes:
// BottleShip identity-maps guest linear to physical, and the operand must carry no segment
// base (checked per operand). The wasm memory offset of guest RAM is baked in at compile
// time, which a shipping version would have to invalidate on memory growth — one more
// reason this is a measurement rather than a feature.
// Modes: 0 = off, 1 = the roadmap-02 class only (ESP/EBP base, constant displacement),
// 2 = EVERY flat 32-bit read, which is roadmap 03's ceiling — what a permission check
// costs on the accesses stack fastmem would NOT cover. Two modes rather than two flags so
// the arms are mutually exclusive by construction.
#[allow(non_upper_case_globals)]
pub static mut STACK_RAW_UNSAFE: u32 = 0;

pub fn stack_raw_unsafe_mode() -> u32 { unsafe { STACK_RAW_UNSAFE } }

#[no_mangle]
pub fn set_stack_raw_unsafe(mode: u32) { unsafe { STACK_RAW_UNSAFE = mode } }

#[no_mangle]
pub fn get_stack_raw_unsafe() -> u32 { unsafe { STACK_RAW_UNSAFE } }

/// The guest-RAM base inside the wasm linear memory, or None before memory is allocated.
fn raw_guest_base() -> Option<u32> {
    let base = unsafe { crate::cpu::memory::mem8 } as u32;
    if base == 0 { None } else { Some(base) }
}

fn stack_raw_applies(ctx: &mut JitContext, modrm_byte: &ModrmByte) -> Option<u32> {
    let mode = stack_raw_unsafe_mode();
    if mode == 0 { return None; }
    if mode == 1 && !modrm_byte.is_stack_const() { return None; }
    if !modrm::stack_const_is_flat(ctx, modrm_byte) { return None; }
    raw_guest_base()
}

pub fn gen_modrm_resolve_safe_read8(ctx: &mut JitContext, modrm_byte: ModrmByte) {
    gen_modrm_resolve_with_local(ctx, modrm_byte, &|ctx, addr| gen_safe_read8(ctx, addr));
}
pub fn gen_modrm_resolve_safe_read16(ctx: &mut JitContext, modrm_byte: ModrmByte) {
    gen_modrm_resolve_with_local(ctx, modrm_byte, &|ctx, addr| gen_safe_read16(ctx, addr));
}
pub fn gen_modrm_resolve_safe_read32(ctx: &mut JitContext, modrm_byte: ModrmByte) {
    if let Some(base) = stack_raw_applies(ctx, &modrm_byte) {
        gen_modrm_resolve(ctx, modrm_byte);
        ctx.builder.load_unaligned_i32(base);
        return;
    }
    gen_modrm_resolve_with_local(ctx, modrm_byte, &|ctx, addr| gen_safe_read32(ctx, addr));
}
pub fn gen_modrm_resolve_safe_read64(ctx: &mut JitContext, modrm_byte: ModrmByte) {
    gen_modrm_resolve_with_local(ctx, modrm_byte, &|ctx, addr| gen_safe_read64(ctx, addr));
}
pub fn gen_modrm_resolve_safe_read128(
    ctx: &mut JitContext,
    modrm_byte: ModrmByte,
    where_to_write: u32,
) {
    gen_modrm_resolve_with_local(ctx, modrm_byte, &|ctx, addr| {
        gen_safe_read128(ctx, addr, where_to_write)
    });
}

pub fn gen_safe_read8(ctx: &mut JitContext, address_local: &WasmLocal) {
    gen_safe_read(ctx, BitSize::BYTE, address_local, None);
}
pub fn gen_safe_read16(ctx: &mut JitContext, address_local: &WasmLocal) {
    gen_safe_read(ctx, BitSize::WORD, address_local, None);
}
pub fn gen_safe_read32(ctx: &mut JitContext, address_local: &WasmLocal) {
    gen_safe_read(ctx, BitSize::DWORD, address_local, None);
}
pub fn gen_safe_read64(ctx: &mut JitContext, address_local: &WasmLocal) {
    gen_safe_read(ctx, BitSize::QWORD, &address_local, None);
}
pub fn gen_safe_read128(ctx: &mut JitContext, address_local: &WasmLocal, where_to_write: u32) {
    gen_safe_read(ctx, BitSize::DQWORD, &address_local, Some(where_to_write));
}

// only used internally for gen_safe_write
enum GenSafeWriteValue<'a> {
    I32(&'a WasmLocal),
    I64(&'a WasmLocalI64),
    TwoI64s(&'a WasmLocalI64, &'a WasmLocalI64),
}

enum GenSafeReadWriteValue {
    I32(WasmLocal),
    I64(WasmLocalI64),
}

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum BitSize {
    BYTE,
    WORD,
    DWORD,
    QWORD,
    DQWORD,
}
impl BitSize {
    pub fn bytes(&self) -> u32 {
        match self {
            BitSize::BYTE => 1,
            BitSize::WORD => 2,
            BitSize::DWORD => 4,
            BitSize::QWORD => 8,
            BitSize::DQWORD => 16,
        }
    }
}

pub fn gen_safe_write8(ctx: &mut JitContext, address_local: &WasmLocal, value_local: &WasmLocal) {
    gen_safe_write(
        ctx,
        BitSize::BYTE,
        address_local,
        GenSafeWriteValue::I32(value_local),
    )
}
pub fn gen_safe_write16(ctx: &mut JitContext, address_local: &WasmLocal, value_local: &WasmLocal) {
    gen_safe_write(
        ctx,
        BitSize::WORD,
        address_local,
        GenSafeWriteValue::I32(value_local),
    )
}
pub fn gen_safe_write32(ctx: &mut JitContext, address_local: &WasmLocal, value_local: &WasmLocal) {
    gen_safe_write(
        ctx,
        BitSize::DWORD,
        address_local,
        GenSafeWriteValue::I32(value_local),
    )
}
pub fn gen_safe_write64(
    ctx: &mut JitContext,
    address_local: &WasmLocal,
    value_local: &WasmLocalI64,
) {
    gen_safe_write(
        ctx,
        BitSize::QWORD,
        address_local,
        GenSafeWriteValue::I64(value_local),
    )
}

pub fn gen_safe_write128(
    ctx: &mut JitContext,
    address_local: &WasmLocal,
    value_local_low: &WasmLocalI64,
    value_local_high: &WasmLocalI64,
) {
    gen_safe_write(
        ctx,
        BitSize::DQWORD,
        address_local,
        GenSafeWriteValue::TwoI64s(value_local_low, value_local_high),
    )
}

/// Release the unit-scoped permission-bitmap scratch, alongside the other per-unit locals.
pub fn gen_perm_map_off_free(ctx: &mut JitContext) {
    if let Some(off) = ctx.perm_map_off.take() {
        ctx.builder.free_local(off);
    }
}

pub fn gen_read_tlb_cache_free(ctx: &mut JitContext) {
    if let Some(cache) = ctx.read_tlb_cache.take() {
        ctx.builder.free_local(cache.page);
        ctx.builder.free_local(cache.entry);
        ctx.builder.free_local(cache.valid);
    }
}

fn gen_read_tlb_cache_slot(ctx: &mut JitContext) -> Option<(WasmLocal, WasmLocal, WasmLocal)> {
    if !crate::jit::read_tlb_cache_enabled_for(ctx.start_of_current_instruction) {
        return None;
    }
    if ctx.read_tlb_cache.is_none() {
        ctx.builder.const_i32(0);
        let page = ctx.builder.set_new_local();
        ctx.builder.const_i32(0);
        let entry = ctx.builder.set_new_local();
        ctx.builder.const_i32(0);
        let valid = ctx.builder.set_new_local();
        ctx.read_tlb_cache = Some(crate::jit::ReadTlbCache { page, entry, valid });
    }
    let cache = ctx.read_tlb_cache.as_ref().unwrap();
    Some((
        cache.page.unsafe_clone(),
        cache.entry.unsafe_clone(),
        cache.valid.unsafe_clone(),
    ))
}

/** A generated guest write may enter a slow translation helper, which is allowed to
 * clear/refill the architectural TLB. Drop the read-local before every write. */
fn gen_read_tlb_cache_invalidate(ctx: &mut JitContext) {
    if let Some(cache) = ctx.read_tlb_cache.as_ref() {
        ctx.builder.const_i32(0);
        ctx.builder.set_local(&cache.valid);
    }
}

fn gen_read_tlb_cache_stat(ctx: &mut JitContext, stat: profiler::stat) {
    if !crate::jit::read_tlb_cache_census_enabled() {
        return;
    }
    let addr = unsafe { &raw mut profiler::stat_array[stat as usize] } as u32;
    ctx.builder.increment_fixed_i64(addr, 1);
}

/// Permission-bitmap read probe (docs/performance/sota-roadmap/03).
///
/// Opens a block and, when the page's byte says "readable, real memory, identity-mapped,
/// reachable at this CPL", writes `mem8 + addr` into `off` and branches out — skipping the
/// TLB entry load, the scale-by-4, and the XOR remap. Returns the label to close and the
/// local the ordinary path must also write, or None when the probe is not emitted.
///
/// Everything the TLB check enforces is enforced here: the byte carries VALID, not-MMIO,
/// read-only and CPL3 reachability, and the page-crossing test is the same one. What the
/// byte adds is IDENTITY — without it there is no remap to skip, so such a page simply
/// falls through to the ordinary path.
fn gen_perm_map_read_probe(
    ctx: &mut JitContext,
    bits: BitSize,
    address_local: &WasmLocal,
) -> Option<(crate::wasmgen::wasm_builder::Label, WasmLocal)> {
    use crate::cpu::perm_map;
    if !perm_map::perm_map_reads_enabled() {
        return None;
    }
    let mem8 = unsafe { crate::cpu::memory::mem8 } as u32;
    if mem8 == 0 {
        return None; // guest memory not allocated yet; nothing to bake
    }
    let mask = if ctx.cpu.cpl3() { perm_map::PERM_READ_MASK_CPL3 } else { perm_map::PERM_READ_MASK_CPL0 };

    // One local per compiled unit, not one per access: a loop body with ten memory operands
    // otherwise pays ten locals and ten initialisers, and that cost is what turned an
    // eight-percent win on a scattered walk into a four-percent LOSS on a dense one.
    if ctx.perm_map_off.is_none() {
        ctx.builder.const_i32(0);
        ctx.perm_map_off = Some(ctx.builder.set_new_local());
    }
    let off = ctx.perm_map_off.as_ref().unwrap().unsafe_clone();
    let done = ctx.builder.block_void();

    ctx.builder.get_local(address_local);
    ctx.builder.const_i32(12);
    ctx.builder.shr_u_i32();
    ctx.builder.load_u8(perm_map::perm_map_base());
    ctx.builder.const_i32(mask as i32);
    ctx.builder.and_i32();
    ctx.builder.const_i32(mask as i32);
    ctx.builder.eq_i32();

    if bits != BitSize::BYTE {
        // The same page-crossing test the TLB path makes: a multi-byte read straddling two
        // pages is two permission questions, and the byte only answers one of them.
        ctx.builder.get_local(address_local);
        ctx.builder.const_i32(0xFFF);
        ctx.builder.and_i32();
        ctx.builder.const_i32(0x1000 - bits.bytes() as i32);
        ctx.builder.le_i32();
        ctx.builder.and_i32();
    }

    ctx.builder.if_void();
    gen_dispatch_stat_increment(ctx.builder, profiler::stat::PERM_MAP_READ_HIT);
    ctx.builder.get_local(address_local);
    ctx.builder.const_i32(mem8 as i32);
    ctx.builder.add_i32();
    ctx.builder.set_local(&off);
    ctx.builder.br(done);
    ctx.builder.block_end();

    gen_dispatch_stat_increment(ctx.builder, profiler::stat::PERM_MAP_READ_MISS);
    Some((done, off))
}

fn gen_safe_read(
    ctx: &mut JitContext,
    bits: BitSize,
    address_local: &WasmLocal,
    where_to_write: Option<u32>,
) {
    // Execute a virtual memory read. All slow paths (memory-mapped IO, tlb miss, page fault and
    // read across page boundary are handled in safe_read_jit_slow
    //   entry <- tlb_data[addr >> 12 << 2]
    //   if entry & MASK == TLB_VALID && (addr & 0xFFF) <= 0x1000 - bytes: goto fast
    //   entry <- safe_read_jit_slow(addr, instruction_pointer)
    //   if page_fault: goto exit-with-pagefault
    //   fast: mem[(entry & ~0xFFF) ^ addr]

    // The read-cache locals are created, and zero-initialised, ONCE per unit; that initialiser
    // must be emitted BEFORE the perm-map probe's block. On a probe hit control branches past
    // everything below, so a recycled local left non-zero would make the NEXT read in this unit
    // take the cache path with a stale page — a wrong read no counter can see.
    let read_cache = gen_read_tlb_cache_slot(ctx);
    let perm_probe = gen_perm_map_read_probe(ctx, bits, address_local);
    ctx.builder.const_i32(0);
    let entry_local = ctx.builder.set_new_local();
    let mut cache_page_now: Option<WasmLocal> = None;
    let entry_ready = if let Some((cache_page, cache_entry, cache_valid)) = read_cache.as_ref() {
        ctx.builder.get_local(address_local);
        ctx.builder.const_i32(12);
        ctx.builder.shr_u_i32();
        let page = ctx.builder.set_new_local();
        let ready = ctx.builder.block_void();
        ctx.builder.get_local(cache_valid);
        ctx.builder.get_local(cache_page);
        ctx.builder.get_local(&page);
        ctx.builder.eq_i32();
        ctx.builder.and_i32();
        if bits != BitSize::BYTE {
            ctx.builder.get_local(address_local);
            ctx.builder.const_i32(0xFFF);
            ctx.builder.and_i32();
            ctx.builder.const_i32(0x1000 - bits.bytes() as i32);
            ctx.builder.le_i32();
            ctx.builder.and_i32();
        }
        ctx.builder.if_void();
        gen_read_tlb_cache_stat(ctx, profiler::stat::READ_TLB_CACHE_HIT);
        ctx.builder.get_local(cache_entry);
        ctx.builder.set_local(&entry_local);
        ctx.builder.br(ready);
        ctx.builder.block_end();
        cache_page_now = Some(page);
        Some(ready)
    }
    else {
        None
    };

    let cont = ctx.builder.block_void();
    ctx.builder.get_local(&address_local);

    ctx.builder.const_i32(12);
    ctx.builder.shr_u_i32();
    ctx.builder.const_i32(2);
    ctx.builder.shl_i32();

    ctx.builder
        .load_aligned_i32(unsafe { &tlb_data[0] as *const i32 as u32 });
    ctx.builder.tee_local(&entry_local);

    ctx.builder.const_i32(
        (0xFFF
            & !TLB_READONLY
            & !TLB_GLOBAL
            & !TLB_HAS_CODE
            & !(if ctx.cpu.cpl3() { 0 } else { TLB_NO_USER })) as i32,
    );
    ctx.builder.and_i32();

    ctx.builder.const_i32(TLB_VALID as i32);
    ctx.builder.eq_i32();

    if bits != BitSize::BYTE {
        ctx.builder.get_local(&address_local);
        ctx.builder.const_i32(0xFFF);
        ctx.builder.and_i32();
        ctx.builder.const_i32(0x1000 - bits.bytes() as i32);
        ctx.builder.le_i32();

        ctx.builder.and_i32();
    }

    // TLB hit is the fast path; falling through means calling safe_read*_slow_jit.
    if let (Some((cache_page, cache_entry, cache_valid)), Some(page)) =
        (read_cache.as_ref(), cache_page_now.as_ref())
    {
        // Fill ONLY here. safe_read*_slow_jit can answer with a one-shot alias into
        // jit_paging_scratch_buffer (page-crossing or MMIO); caching that would serve a later
        // same-page read stale scratch bytes and skip the device access entirely.
        ctx.builder.if_void_hinted(HINT_GROUP_MEM, HINT_LIKELY);
        gen_read_tlb_cache_stat(ctx, profiler::stat::READ_TLB_CACHE_FILL);
        ctx.builder.get_local(page);
        ctx.builder.set_local(cache_page);
        ctx.builder.get_local(&entry_local);
        ctx.builder.set_local(cache_entry);
        ctx.builder.const_i32(1);
        ctx.builder.set_local(cache_valid);
        ctx.builder.br(cont);
        ctx.builder.block_end();
    }
    else {
        ctx.builder.br_if_hinted(cont, HINT_GROUP_MEM, HINT_LIKELY);
    }

    if cfg!(feature = "profiler") {
        ctx.builder.get_local(&address_local);
        ctx.builder.get_local(&entry_local);
        ctx.builder.call_fn2("report_safe_read_jit_slow");
    }

    ctx.builder.get_local(&address_local);
    ctx.builder
        .const_i32(ctx.start_of_current_instruction as i32 & 0xFFF);
    match bits {
        BitSize::BYTE => {
            ctx.builder.call_fn2_ret("safe_read8_slow_jit");
        },
        BitSize::WORD => {
            ctx.builder.call_fn2_ret("safe_read16_slow_jit");
        },
        BitSize::DWORD => {
            ctx.builder.call_fn2_ret("safe_read32s_slow_jit");
        },
        BitSize::QWORD => {
            ctx.builder.call_fn2_ret("safe_read64s_slow_jit");
        },
        BitSize::DQWORD => {
            ctx.builder.call_fn2_ret("safe_read128s_slow_jit");
        },
    }
    ctx.builder.tee_local(&entry_local);
    ctx.builder.const_i32(1);
    ctx.builder.and_i32();

    if cfg!(feature = "profiler") {
        ctx.builder.if_void();
        gen_debug_track_jit_exit(ctx.builder, ctx.start_of_current_instruction);
        ctx.builder.block_end();

        ctx.builder.get_local(&entry_local);
        ctx.builder.const_i32(1);
        ctx.builder.and_i32();
    }

    // Page fault out of a guest memory access: cold by construction.
    ctx.builder
        .br_if_hinted(ctx.exit_with_fault_label, HINT_GROUP_MEM, HINT_UNLIKELY);

    ctx.builder.block_end();

    if entry_ready.is_some() {
        ctx.builder.block_end();
        ctx.builder.free_local(cache_page_now.take().unwrap());
    }

    gen_profiler_stat_increment(ctx.builder, profiler::stat::SAFE_READ_FAST); // XXX: Both fast and slow

    ctx.builder.get_local(&entry_local);
    ctx.builder.const_i32(!0xFFF);
    ctx.builder.and_i32();
    ctx.builder.get_local(&address_local);
    ctx.builder.xor_i32();

    // Both paths converge on ONE wasm offset, so the load below is emitted once and the
    // arms cannot drift into loading different widths or from different bases.
    if let Some((done, off)) = perm_probe.as_ref() {
        ctx.builder.set_local(off);
        ctx.builder.block_end();
        let _ = done;
        ctx.builder.get_local(off);
    }

    // where_to_write is only used by dqword
    dbg_assert!((where_to_write != None) == (bits == BitSize::DQWORD));

    match bits {
        BitSize::BYTE => {
            ctx.builder.load_u8(0);
        },
        BitSize::WORD => {
            ctx.builder.load_unaligned_u16(0);
        },
        BitSize::DWORD => {
            ctx.builder.load_unaligned_i32(0);
        },
        BitSize::QWORD => {
            ctx.builder.load_unaligned_i64(0);
        },
        BitSize::DQWORD => {
            let where_to_write = where_to_write.unwrap();
            let virt_address_local = ctx.builder.set_new_local();
            ctx.builder.const_i32(0);
            ctx.builder.get_local(&virt_address_local);
            ctx.builder.load_unaligned_i64(0);
            ctx.builder.store_unaligned_i64(where_to_write);

            ctx.builder.const_i32(0);
            ctx.builder.get_local(&virt_address_local);
            ctx.builder.load_unaligned_i64(8);
            ctx.builder.store_unaligned_i64(where_to_write + 8);

            ctx.builder.free_local(virt_address_local);
        },
    }

    ctx.builder.free_local(entry_local);
    // `off` belongs to the unit (ctx.perm_map_off) and is released with it, not here.
    drop(perm_probe);
}

// Split-range shape of the fastmem read fast path. DEAD: reads go through the inlined TLB
// probe in gen_safe_read. Idx 18 is NOT in JIT_CONFIG_SUPPORTED_MASK, so set_jit_config(18)
// answers UNSUPPORTED — this is not a knob anything can turn.
// Same acceptance set as the legacy shape — [LOW_MEM_END, min(GUARD_BASE, ram) - bytes]
// ∪ [GUARD_END, ram - bytes] — decomposed into two early-exit range tests so the hot
// case (below-guard HEAP/image data) costs one sub+cmp+br_if and a direct load:
// ~10 wasm ops instead of the legacy ~25 (4-compare and/or chain + if/else + local).
// The value flows on the stack via a result-typed block; no address local at all
// (except DQWORD, which needs the host address twice).
#[allow(dead_code)]
fn gen_fastmem_read_split(
    _ctx: &mut JitContext,
    _bits: BitSize,
    _address_local: &WasmLocal,
    _where_to_write: Option<u32>,
) {
    unreachable!("read fastmem was removed");
/*
    let bytes = bits.bytes() as u32;
    let ram_size = unsafe { *global_pointers::memory_size };
    const LOW_MEM_END: u32 = crate::jit::FASTMEM_LOW_MEM_END;
    const GUARD_BASE: u32 = crate::jit::FASTMEM_GUARD_BASE;
    const GUARD_SIZE: u32 = crate::jit::FASTMEM_GUARD_SIZE;
    const GUARD_END: u32 = GUARD_BASE + GUARD_SIZE;

    // Range 1: [LOW_MEM_END, r1_top]; min() keeps small-RAM configs correct.
    let r1_top = GUARD_BASE.min(ram_size) - bytes;
    dbg_assert!(r1_top >= LOW_MEM_END); // caller checked max_addr >= LOW_MEM_END
    // Range 2: [GUARD_END, ram - bytes], only when RAM extends past the red zone.
    let r2 = if ram_size >= GUARD_END + bytes {
        Some(ram_size - bytes - GUARD_END)
    }
    else {
        None
    };

    let host = ctx.builder.block_i32(); // yields the wasm-memory offset to load from

    let try_next = ctx.builder.block_void();
    ctx.builder.get_local(address_local);
    ctx.builder.const_i32(LOW_MEM_END as i32);
    ctx.builder.sub_i32();
    ctx.builder.const_i32((r1_top - LOW_MEM_END) as i32);
    ctx.builder.gtu_i32();
    // Range-1 miss (below-guard HEAP/image data is the hot case).
    ctx.builder.br_if_hinted(try_next, HINT_GROUP_MEM, HINT_UNLIKELY);
    ctx.builder.const_i32(unsafe { memory::mem8 } as i32);
    ctx.builder.get_local(address_local);
    ctx.builder.add_i32();
    ctx.builder.br(host);
    ctx.builder.block_end();

    if let Some(r2_span) = r2 {
        let slow = ctx.builder.block_void();
        ctx.builder.get_local(address_local);
        ctx.builder.const_i32(GUARD_END as i32);
        ctx.builder.sub_i32();
        ctx.builder.const_i32(r2_span as i32);
        ctx.builder.gtu_i32();
        // Range-2 miss ⇒ the universal slow helper.
        ctx.builder.br_if_hinted(slow, HINT_GROUP_MEM, HINT_UNLIKELY);
        ctx.builder.const_i32(unsafe { memory::mem8 } as i32);
        ctx.builder.get_local(address_local);
        ctx.builder.add_i32();
        ctx.builder.br(host);
        ctx.builder.block_end();
    }

    // Slow path: universal helper (MMIO, guard faults, decommit, out-of-range).
    if cfg!(feature = "profiler") {
        ctx.builder.get_local(address_local);
        ctx.builder.const_i32(0);
        ctx.builder.call_fn2("report_safe_read_jit_slow");
    }

    ctx.builder.get_local(address_local);
    ctx.builder
        .const_i32(ctx.start_of_current_instruction as i32 & 0xFFF);
    match bits {
        BitSize::BYTE => {
            ctx.builder.call_fn2_ret("safe_read8_slow_jit");
        },
        BitSize::WORD => {
            ctx.builder.call_fn2_ret("safe_read16_slow_jit");
        },
        BitSize::DWORD => {
            ctx.builder.call_fn2_ret("safe_read32s_slow_jit");
        },
        BitSize::QWORD => {
            ctx.builder.call_fn2_ret("safe_read64s_slow_jit");
        },
        BitSize::DQWORD => {
            ctx.builder.call_fn2_ret("safe_read128s_slow_jit");
        },
    }
    let entry_local = ctx.builder.tee_new_local();
    ctx.builder.const_i32(1);
    ctx.builder.and_i32();

    if cfg!(feature = "profiler") {
        ctx.builder.if_void();
        gen_debug_track_jit_exit(ctx.builder, ctx.start_of_current_instruction);
        ctx.builder.block_end();

        ctx.builder.get_local(&entry_local);
        ctx.builder.const_i32(1);
        ctx.builder.and_i32();
    }

    // Page fault out of a guest memory access: cold by construction.
    ctx.builder
        .br_if_hinted(ctx.exit_with_fault_label, HINT_GROUP_MEM, HINT_UNLIKELY);

    // TLB entries fold mem8 in: (entry & ~0xFFF) ^ addr is a wasm-memory offset.
    ctx.builder.get_local(&entry_local);
    ctx.builder.const_i32(!0xFFF);
    ctx.builder.and_i32();
    ctx.builder.get_local(address_local);
    ctx.builder.xor_i32();
    ctx.builder.free_local(entry_local);

    ctx.builder.block_end(); // host

    gen_profiler_stat_increment(ctx.builder, profiler::stat::SAFE_READ_FAST);

    dbg_assert!((where_to_write != None) == (bits == BitSize::DQWORD));

    match bits {
        BitSize::BYTE => {
            ctx.builder.load_u8(0);
        },
        BitSize::WORD => {
            ctx.builder.load_unaligned_u16(0);
        },
        BitSize::DWORD => {
            ctx.builder.load_unaligned_i32(0);
        },
        BitSize::QWORD => {
            ctx.builder.load_unaligned_i64(0);
        },
        BitSize::DQWORD => {
            let where_to_write = where_to_write.unwrap();
            let virt_address_local = ctx.builder.set_new_local();

            ctx.builder.const_i32(0);
            ctx.builder.get_local(&virt_address_local);
            ctx.builder.load_unaligned_i64(0);
            ctx.builder.store_unaligned_i64(where_to_write);

            ctx.builder.const_i32(0);
            ctx.builder.get_local(&virt_address_local);
            ctx.builder.load_unaligned_i64(8);
            ctx.builder.store_unaligned_i64(where_to_write + 8);

            ctx.builder.free_local(virt_address_local);
        },
    }
*/
}

pub fn gen_get_phys_eip_plus_mem(ctx: &mut JitContext, address_local: &WasmLocal) {
    // Similar to gen_safe_read, but return the physical eip + memory::mem rather than reading from memory
    // In functions that need to use this value we need to fix it by substracting memory::mem
    // this is done in order to remove one instruction from the fast path of memory accesses (no need to add
    // memory::mem anymore ).
    // We need to account for this in gen_page_switch_check and we compare with next_block_addr + memory::mem8
    // We cannot the same while processing an AbsoluteEip flow control change so there we need to fix the value
    // by subscracting memory::mem. Overall, since AbsoluteEip is encountered less often than memory accesses so
    // this ends up improving perf.
    // Does not (need to) handle mapped memory
    // XXX: Currently does not use ctx.start_of_current_instruction, but rather assumes that eip is
    //      already correct (pointing at the current instruction)

    let cont = ctx.builder.block_void();
    ctx.builder.get_local(&address_local);

    ctx.builder.const_i32(12);
    ctx.builder.shr_u_i32();
    ctx.builder.const_i32(2);
    ctx.builder.shl_i32();

    ctx.builder
        .load_aligned_i32(unsafe { &tlb_data[0] as *const i32 as u32 });
    let entry_local = ctx.builder.tee_new_local();

    ctx.builder.const_i32(
        (0xFFF
            & !TLB_READONLY
            & !TLB_GLOBAL
            & !TLB_HAS_CODE
            & !(if ctx.cpu.cpl3() { 0 } else { TLB_NO_USER })) as i32,
    );
    ctx.builder.and_i32();

    ctx.builder.const_i32(TLB_VALID as i32);
    ctx.builder.eq_i32();

    // TLB hit; the fall-through calls get_phys_eip_slow_jit.
    ctx.builder.br_if_hinted(cont, HINT_GROUP_MEM, HINT_LIKELY);

    if cfg!(feature = "profiler") {
        ctx.builder.get_local(&address_local);
        ctx.builder.get_local(&entry_local);
        ctx.builder.call_fn2("report_safe_read_jit_slow");
    }

    ctx.builder.get_local(&address_local);
    ctx.builder.call_fn1_ret("get_phys_eip_slow_jit");

    ctx.builder.tee_local(&entry_local);
    ctx.builder.const_i32(1);
    ctx.builder.and_i32();

    if cfg!(feature = "profiler") {
        ctx.builder.if_void();
        gen_debug_track_jit_exit(ctx.builder, ctx.start_of_current_instruction); // XXX
        ctx.builder.block_end();

        ctx.builder.get_local(&entry_local);
        ctx.builder.const_i32(1);
        ctx.builder.and_i32();
    }

    // Page fault out of a guest memory access: cold by construction.
    ctx.builder
        .br_if_hinted(ctx.exit_with_fault_label, HINT_GROUP_MEM, HINT_UNLIKELY);

    ctx.builder.block_end();

    gen_profiler_stat_increment(ctx.builder, profiler::stat::SAFE_READ_FAST); // XXX: Both fast and slow

    ctx.builder.get_local(&entry_local);
    ctx.builder.const_i32(!0xFFF);
    ctx.builder.and_i32();
    ctx.builder.get_local(&address_local);
    ctx.builder.xor_i32();

    ctx.builder.free_local(entry_local);
}

// Store fast path gated by the per-page write map (idx 19).
// Structure mirrors gen_fastmem_read_split: a result-typed `host` block yields the
// wasm-memory offset to store to. Fast path (map byte == 1 → base-writable, no
// compiled code, no watch, and the access does not cross a page) yields `mem8 + addr`
// (identity map, VA == PA). Slow path calls the universal byte-precise helper, which
// itself validates + performs MMIO / page-cross writes and returns a TLB-style entry
// so the trailing inline store lands in real RAM (or a scratch page for the cases the
// helper already handled). No generation guard — the map is DATA read per store, kept
// honest synchronously at the same choke points as the TLB.
fn gen_fastmem_write_map(
    ctx: &mut JitContext,
    bits: BitSize,
    address_local: &WasmLocal,
    value_local: GenSafeWriteValue,
) {
    let bytes = bits.bytes() as i32;
    let map_base = crate::jit::fastmem_write_map_base();

    crate::jit::fastmem_note_speculated_store_compiled();

    let host = ctx.builder.block_i32(); // yields the wasm-memory offset to store to

    let slow = ctx.builder.block_void();
    // Reject unless map[addr >> 12] == 1 AND (multi-byte) the access stays in-page.
    ctx.builder.get_local(address_local);
    ctx.builder.const_i32(12);
    ctx.builder.shr_u_i32();
    ctx.builder.load_u8(map_base);
    ctx.builder.const_i32(1);
    ctx.builder.ne_i32();
    if bits != BitSize::BYTE {
        ctx.builder.get_local(address_local);
        ctx.builder.const_i32(0xFFF);
        ctx.builder.and_i32();
        ctx.builder.const_i32(0x1000 - bytes);
        ctx.builder.gt_i32();
        ctx.builder.or_i32();
    }
    ctx.builder.br_if(slow);
    // fast: identity map, mem8 + addr
    ctx.builder.const_i32(unsafe { memory::mem8 } as i32);
    ctx.builder.get_local(address_local);
    ctx.builder.add_i32();
    ctx.builder.br(host);
    ctx.builder.block_end(); // slow

    // Slow path: universal helper (fault, MMIO, page-cross, code page, out-of-range).
    if cfg!(feature = "profiler") {
        ctx.builder.get_local(address_local);
        ctx.builder.const_i32(0);
        ctx.builder.call_fn2("report_safe_write_jit_slow");
    }

    ctx.builder.get_local(address_local);
    match value_local {
        GenSafeWriteValue::I32(local) => ctx.builder.get_local(local),
        GenSafeWriteValue::I64(local) => ctx.builder.get_local_i64(local),
        GenSafeWriteValue::TwoI64s(local1, local2) => {
            ctx.builder.get_local_i64(local1);
            ctx.builder.get_local_i64(local2)
        },
    }
    ctx.builder
        .const_i32(ctx.start_of_current_instruction as i32 & 0xFFF);
    match bits {
        BitSize::BYTE => ctx.builder.call_fn3_ret("safe_write8_slow_jit"),
        BitSize::WORD => ctx.builder.call_fn3_ret("safe_write16_slow_jit"),
        BitSize::DWORD => ctx.builder.call_fn3_ret("safe_write32_slow_jit"),
        BitSize::QWORD => ctx.builder.call_fn3_i32_i64_i32_ret("safe_write64_slow_jit"),
        BitSize::DQWORD => ctx
            .builder
            .call_fn4_i32_i64_i64_i32_ret("safe_write128_slow_jit"),
    }
    let entry_local = ctx.builder.tee_new_local();
    ctx.builder.const_i32(1);
    ctx.builder.and_i32();

    if cfg!(feature = "profiler") {
        ctx.builder.if_void();
        gen_debug_track_jit_exit(ctx.builder, ctx.start_of_current_instruction);
        ctx.builder.block_end();

        ctx.builder.get_local(&entry_local);
        ctx.builder.const_i32(1);
        ctx.builder.and_i32();
    }

    // Page fault out of a guest memory access: cold by construction.
    ctx.builder
        .br_if_hinted(ctx.exit_with_fault_label, HINT_GROUP_MEM, HINT_UNLIKELY);

    // Helper returned a TLB-style entry: (entry & ~0xFFF) ^ addr is the store offset.
    ctx.builder.get_local(&entry_local);
    ctx.builder.const_i32(!0xFFF);
    ctx.builder.and_i32();
    ctx.builder.get_local(address_local);
    ctx.builder.xor_i32();
    ctx.builder.free_local(entry_local);

    ctx.builder.block_end(); // host — store offset now on stack

    gen_profiler_stat_increment(ctx.builder, profiler::stat::SAFE_WRITE_FAST);

    // Store the value at the yielded offset (stack top is the offset).
    match value_local {
        GenSafeWriteValue::I32(local) => {
            ctx.builder.get_local(local);
            match bits {
                BitSize::BYTE => ctx.builder.store_u8(0),
                BitSize::WORD => ctx.builder.store_unaligned_u16(0),
                BitSize::DWORD => ctx.builder.store_unaligned_i32(0),
                _ => dbg_assert!(false),
            }
        },
        GenSafeWriteValue::I64(local) => {
            ctx.builder.get_local_i64(local);
            ctx.builder.store_unaligned_i64(0);
        },
        GenSafeWriteValue::TwoI64s(local1, local2) => {
            let store_addr = ctx.builder.set_new_local();
            ctx.builder.get_local(&store_addr);
            ctx.builder.get_local_i64(local1);
            ctx.builder.store_unaligned_i64(0);
            ctx.builder.get_local(&store_addr);
            ctx.builder.get_local_i64(local2);
            ctx.builder.store_unaligned_i64(8);
            ctx.builder.free_local(store_addr);
        },
    }
}

fn gen_safe_write(
    ctx: &mut JitContext,
    bits: BitSize,
    address_local: &WasmLocal,
    value_local: GenSafeWriteValue,
) {
    // When enabled for this unit, route through the per-page write
    // map instead of the inline TLB fast path. Flag off = byte-identical to below.
    if ctx.fastmem_writes {
        // The experimental write-map helper owns a different slow-path shape. Keep the
        // conservative invalidation there; the ordinary TLB path below narrows it to the
        // actual slow edge.
        gen_read_tlb_cache_invalidate(ctx);
        gen_fastmem_write_map(ctx, bits, address_local, value_local);
        return;
    }

    // Execute a virtual memory write. All slow paths (memory-mapped IO, tlb miss, page fault,
    // write across page boundary and page containing jitted code are handled in safe_write_jit_slow

    //   entry <- tlb_data[addr >> 12 << 2]
    //   if entry & MASK == TLB_VALID && (addr & 0xFFF) <= 0x1000 - bytes: goto fast
    //   entry <- safe_write_jit_slow(addr, value, instruction_pointer)
    //   if page_fault: goto exit-with-pagefault
    //   fast: mem[(entry & ~0xFFF) ^ addr] <- value

    let cont = ctx.builder.block_void();
    ctx.builder.get_local(&address_local);

    ctx.builder.const_i32(12);
    ctx.builder.shr_u_i32();
    ctx.builder.const_i32(2);
    ctx.builder.shl_i32();

    ctx.builder
        .load_aligned_i32(unsafe { &tlb_data[0] as *const i32 as u32 });
    let entry_local = ctx.builder.tee_new_local();

    ctx.builder
        .const_i32((0xFFF & !TLB_GLOBAL & !(if ctx.cpu.cpl3() { 0 } else { TLB_NO_USER })) as i32);
    ctx.builder.and_i32();

    ctx.builder.const_i32(TLB_VALID as i32);
    ctx.builder.eq_i32();

    if bits != BitSize::BYTE {
        ctx.builder.get_local(&address_local);
        ctx.builder.const_i32(0xFFF);
        ctx.builder.and_i32();
        ctx.builder.const_i32(0x1000 - bits.bytes() as i32);
        ctx.builder.le_i32();

        ctx.builder.and_i32();
    }

    // TLB hit; the fall-through calls safe_write*_slow_jit.
    ctx.builder.br_if_hinted(cont, HINT_GROUP_MEM, HINT_LIKELY);

    if cfg!(feature = "profiler") {
        ctx.builder.get_local(&address_local);
        ctx.builder.get_local(&entry_local);
        ctx.builder.call_fn2("report_safe_write_jit_slow");
    }

    // A TLB-hit store cannot refill or clear the architectural translation cache. Only the
    // helper edge can, so preserve the block-local read entry across ordinary guest writes.
    gen_read_tlb_cache_invalidate(ctx);
    ctx.builder.get_local(&address_local);
    match value_local {
        GenSafeWriteValue::I32(local) => ctx.builder.get_local(local),
        GenSafeWriteValue::I64(local) => ctx.builder.get_local_i64(local),
        GenSafeWriteValue::TwoI64s(local1, local2) => {
            ctx.builder.get_local_i64(local1);
            ctx.builder.get_local_i64(local2)
        },
    }
    ctx.builder
        .const_i32(ctx.start_of_current_instruction as i32 & 0xFFF);
    match bits {
        BitSize::BYTE => {
            ctx.builder.call_fn3_ret("safe_write8_slow_jit");
        },
        BitSize::WORD => {
            ctx.builder.call_fn3_ret("safe_write16_slow_jit");
        },
        BitSize::DWORD => {
            ctx.builder.call_fn3_ret("safe_write32_slow_jit");
        },
        BitSize::QWORD => {
            ctx.builder
                .call_fn3_i32_i64_i32_ret("safe_write64_slow_jit");
        },
        BitSize::DQWORD => {
            ctx.builder
                .call_fn4_i32_i64_i64_i32_ret("safe_write128_slow_jit");
        },
    }
    ctx.builder.tee_local(&entry_local);
    ctx.builder.const_i32(1);
    ctx.builder.and_i32();

    if cfg!(feature = "profiler") {
        ctx.builder.if_void();
        gen_debug_track_jit_exit(ctx.builder, ctx.start_of_current_instruction);
        ctx.builder.block_end();

        ctx.builder.get_local(&entry_local);
        ctx.builder.const_i32(1);
        ctx.builder.and_i32();
    }

    // Page fault out of a guest memory access: cold by construction.
    ctx.builder
        .br_if_hinted(ctx.exit_with_fault_label, HINT_GROUP_MEM, HINT_UNLIKELY);

    ctx.builder.block_end();

    gen_profiler_stat_increment(ctx.builder, profiler::stat::SAFE_WRITE_FAST); // XXX: Both fast and slow

    ctx.builder.get_local(&entry_local);
    ctx.builder.const_i32(!0xFFF);
    ctx.builder.and_i32();
    ctx.builder.get_local(&address_local);
    ctx.builder.xor_i32();

    match value_local {
        GenSafeWriteValue::I32(local) => ctx.builder.get_local(local),
        GenSafeWriteValue::I64(local) => ctx.builder.get_local_i64(local),
        GenSafeWriteValue::TwoI64s(local1, local2) => {
            assert!(bits == BitSize::DQWORD);

            let virt_address_local = ctx.builder.tee_new_local();
            ctx.builder.get_local_i64(local1);
            ctx.builder.store_unaligned_i64(0);

            ctx.builder.get_local(&virt_address_local);
            ctx.builder.get_local_i64(local2);
            ctx.builder.store_unaligned_i64(8);
            ctx.builder.free_local(virt_address_local);
        },
    }
    match bits {
        BitSize::BYTE => {
            ctx.builder.store_u8(0);
        },
        BitSize::WORD => {
            ctx.builder.store_unaligned_u16(0);
        },
        BitSize::DWORD => {
            ctx.builder.store_unaligned_i32(0);
        },
        BitSize::QWORD => {
            ctx.builder.store_unaligned_i64(0);
        },
        BitSize::DQWORD => {}, // handled above
    }

    ctx.builder.free_local(entry_local);
}

pub fn gen_push32_write_cache_free(ctx: &mut JitContext) {
    if let Some(cache) = ctx.push32_write_cache.take() {
        ctx.builder.free_local(cache.page);
        ctx.builder.free_local(cache.entry);
        ctx.builder.free_local(cache.valid);
    }
}

fn gen_push32_write_cache_slot(ctx: &mut JitContext) -> Option<(WasmLocal, WasmLocal, WasmLocal)> {
    if !crate::jit::push_run_coalescing_enabled() {
        return None;
    }
    if ctx.push32_write_cache.is_none() {
        ctx.builder.const_i32(0);
        let page = ctx.builder.set_new_local();
        ctx.builder.const_i32(0);
        let entry = ctx.builder.set_new_local();
        ctx.builder.const_i32(0);
        let valid = ctx.builder.set_new_local();
        ctx.push32_write_cache = Some(crate::jit::Push32WriteCache { page, entry, valid });
    }

    let cache = ctx.push32_write_cache.as_ref().unwrap();
    Some((
        cache.page.unsafe_clone(),
        cache.entry.unsafe_clone(),
        cache.valid.unsafe_clone(),
    ))
}

fn gen_write32_with_entry(
    ctx: &mut JitContext,
    address_local: &WasmLocal,
    entry_local: &WasmLocal,
    value_local: &WasmLocal,
) {
    ctx.builder.get_local(entry_local);
    ctx.builder.const_i32(!0xFFF);
    ctx.builder.and_i32();
    ctx.builder.get_local(address_local);
    ctx.builder.xor_i32();
    ctx.builder.get_local(value_local);
    ctx.builder.store_unaligned_i32(0);
}

fn gen_push32_coalesced_write(
    ctx: &mut JitContext,
    address_local: &WasmLocal,
    value_local: &WasmLocal,
) -> bool {
    let Some((cache_page, cache_entry, cache_valid)) = gen_push32_write_cache_slot(ctx) else {
        return false;
    };

    crate::jit::push_run_note_site_compiled();

    ctx.builder.get_local(address_local);
    ctx.builder.const_i32(12);
    ctx.builder.shr_u_i32();
    let page_local = ctx.builder.set_new_local();

    let done = ctx.builder.block_void();

    ctx.builder.get_local(&cache_valid);
    ctx.builder.get_local(&cache_page);
    ctx.builder.get_local(&page_local);
    ctx.builder.eq_i32();
    ctx.builder.and_i32();
    ctx.builder.get_local(address_local);
    ctx.builder.const_i32(0xFFF);
    ctx.builder.and_i32();
    ctx.builder.const_i32(0x1000 - 4);
    ctx.builder.le_i32();
    ctx.builder.and_i32();
    ctx.builder.if_void();
    crate::jit::push_run_note_reuse_branch_compiled();
    gen_dispatch_stat_increment(ctx.builder, profiler::stat::PUSH_RUN_HIT);
    gen_write32_with_entry(ctx, address_local, &cache_entry, value_local);
    ctx.builder.br(done);
    ctx.builder.block_end();

    let fast = ctx.builder.block_void();
    ctx.builder.get_local(address_local);
    ctx.builder.const_i32(12);
    ctx.builder.shr_u_i32();
    ctx.builder.const_i32(2);
    ctx.builder.shl_i32();
    ctx.builder
        .load_aligned_i32(unsafe { &tlb_data[0] as *const i32 as u32 });
    let entry_local = ctx.builder.tee_new_local();

    ctx.builder
        .const_i32((0xFFF & !TLB_GLOBAL & !(if ctx.cpu.cpl3() { 0 } else { TLB_NO_USER })) as i32);
    ctx.builder.and_i32();
    ctx.builder.const_i32(TLB_VALID as i32);
    ctx.builder.eq_i32();

    ctx.builder.get_local(address_local);
    ctx.builder.const_i32(0xFFF);
    ctx.builder.and_i32();
    ctx.builder.const_i32(0x1000 - 4);
    ctx.builder.le_i32();
    ctx.builder.and_i32();
    ctx.builder.br_if(fast);

    if cfg!(feature = "profiler") {
        ctx.builder.get_local(address_local);
        ctx.builder.get_local(&entry_local);
        ctx.builder.call_fn2("report_safe_write_jit_slow");
    }

    gen_read_tlb_cache_invalidate(ctx);
    ctx.builder.get_local(address_local);
    ctx.builder.get_local(value_local);
    ctx.builder
        .const_i32(ctx.start_of_current_instruction as i32 & 0xFFF);
    ctx.builder.call_fn3_ret("safe_write32_slow_jit");
    ctx.builder.tee_local(&entry_local);
    ctx.builder.const_i32(1);
    ctx.builder.and_i32();

    if cfg!(feature = "profiler") {
        ctx.builder.if_void();
        gen_debug_track_jit_exit(ctx.builder, ctx.start_of_current_instruction);
        ctx.builder.block_end();

        ctx.builder.get_local(&entry_local);
        ctx.builder.const_i32(1);
        ctx.builder.and_i32();
    }

    // Page fault out of a guest memory access: cold by construction.
    ctx.builder
        .br_if_hinted(ctx.exit_with_fault_label, HINT_GROUP_MEM, HINT_UNLIKELY);
    gen_profiler_stat_increment(ctx.builder, profiler::stat::SAFE_WRITE_FAST);
    gen_write32_with_entry(ctx, address_local, &entry_local, value_local);
    ctx.builder.br(done);

    ctx.builder.block_end();

    gen_dispatch_stat_increment(ctx.builder, profiler::stat::PUSH_RUN_FILL);
    ctx.builder.get_local(&page_local);
    ctx.builder.set_local(&cache_page);
    ctx.builder.get_local(&entry_local);
    ctx.builder.set_local(&cache_entry);
    ctx.builder.const_i32(1);
    ctx.builder.set_local(&cache_valid);
    gen_profiler_stat_increment(ctx.builder, profiler::stat::SAFE_WRITE_FAST);
    gen_write32_with_entry(ctx, address_local, &entry_local, value_local);

    ctx.builder.block_end();
    ctx.builder.free_local(entry_local);
    ctx.builder.free_local(page_local);
    true
}

pub fn gen_safe_read_write(
    ctx: &mut JitContext,
    bits: BitSize,
    address_local: &WasmLocal,
    f: &dyn Fn(&mut JitContext),
) {
    // Execute a virtual memory read+write. All slow paths (memory-mapped IO, tlb miss, page fault,
    // write across page boundary and page containing jitted code are handled in
    // safe_read_write_jit_slow

    //   entry <- tlb_data[addr >> 12 << 2]
    //   can_use_fast_path <- entry & MASK == TLB_VALID && (addr & 0xFFF) <= 0x1000 - bytes
    //   if can_use_fast_path: goto fast
    //   entry <- safe_read_write_jit_slow(addr, instruction_pointer)
    //   if page_fault: goto exit-with-pagefault
    //   fast: value <- f(mem[(entry & ~0xFFF) ^ addr])
    //   if !can_use_fast_path { safe_write_jit_slow(addr, value, instruction_pointer) }
    //   mem[(entry & ~0xFFF) ^ addr] <- value

    let cont = ctx.builder.block_void();
    ctx.builder.get_local(address_local);

    ctx.builder.const_i32(12);
    ctx.builder.shr_u_i32();
    ctx.builder.const_i32(2);
    ctx.builder.shl_i32();

    ctx.builder
        .load_aligned_i32(unsafe { &tlb_data[0] as *const i32 as u32 });
    let entry_local = ctx.builder.tee_new_local();

    ctx.builder
        .const_i32((0xFFF & !TLB_GLOBAL & !(if ctx.cpu.cpl3() { 0 } else { TLB_NO_USER })) as i32);
    ctx.builder.and_i32();

    ctx.builder.const_i32(TLB_VALID as i32);
    ctx.builder.eq_i32();

    if bits != BitSize::BYTE {
        ctx.builder.get_local(&address_local);
        ctx.builder.const_i32(0xFFF);
        ctx.builder.and_i32();
        ctx.builder.const_i32(0x1000 - bits.bytes() as i32);
        ctx.builder.le_i32();
        ctx.builder.and_i32();
    }

    let can_use_fast_path_local = ctx.builder.tee_new_local();

    // TLB hit; the fall-through calls safe_read_write*_slow_jit.
    ctx.builder.br_if_hinted(cont, HINT_GROUP_MEM, HINT_LIKELY);

    if cfg!(feature = "profiler") {
        ctx.builder.get_local(&address_local);
        ctx.builder.get_local(&entry_local);
        ctx.builder.call_fn2("report_safe_read_write_jit_slow");
    }

    // The fast read-modify-write path uses the already validated entry and cannot mutate the
    // architectural TLB. Invalidate only when the translation helper is actually entered.
    gen_read_tlb_cache_invalidate(ctx);
    ctx.builder.get_local(&address_local);
    ctx.builder
        .const_i32(ctx.start_of_current_instruction as i32 & 0xFFF);

    match bits {
        BitSize::BYTE => {
            ctx.builder.call_fn2_ret("safe_read_write8_slow_jit");
        },
        BitSize::WORD => {
            ctx.builder.call_fn2_ret("safe_read_write16_slow_jit");
        },
        BitSize::DWORD => {
            ctx.builder.call_fn2_ret("safe_read_write32s_slow_jit");
        },
        BitSize::QWORD => {
            ctx.builder.call_fn2_ret("safe_read_write64_slow_jit");
        },
        BitSize::DQWORD => {
            dbg_assert!(false);
        },
    }
    ctx.builder.tee_local(&entry_local);
    ctx.builder.const_i32(1);
    ctx.builder.and_i32();

    if cfg!(feature = "profiler") {
        ctx.builder.if_void();
        gen_debug_track_jit_exit(ctx.builder, ctx.start_of_current_instruction);
        ctx.builder.block_end();

        ctx.builder.get_local(&entry_local);
        ctx.builder.const_i32(1);
        ctx.builder.and_i32();
    }

    // Page fault out of a guest memory access: cold by construction.
    ctx.builder
        .br_if_hinted(ctx.exit_with_fault_label, HINT_GROUP_MEM, HINT_UNLIKELY);

    ctx.builder.block_end();

    gen_profiler_stat_increment(ctx.builder, profiler::stat::SAFE_READ_WRITE_FAST); // XXX: Also slow

    ctx.builder.get_local(&entry_local);
    ctx.builder.const_i32(!0xFFF);
    ctx.builder.and_i32();
    ctx.builder.get_local(&address_local);
    ctx.builder.xor_i32();

    ctx.builder.free_local(entry_local);
    let phys_addr_local = ctx.builder.tee_new_local();

    match bits {
        BitSize::BYTE => {
            ctx.builder.load_u8(0);
        },
        BitSize::WORD => {
            ctx.builder.load_unaligned_u16(0);
        },
        BitSize::DWORD => {
            ctx.builder.load_unaligned_i32(0);
        },
        BitSize::QWORD => {
            ctx.builder.load_unaligned_i64(0);
        },
        BitSize::DQWORD => assert!(false), // not used
    }

    // value is now on stack

    f(ctx);

    // TODO: Could get rid of this local by returning one from f
    let value_local = if bits == BitSize::QWORD {
        GenSafeReadWriteValue::I64(ctx.builder.set_new_local_i64())
    }
    else {
        GenSafeReadWriteValue::I32(ctx.builder.set_new_local())
    };

    ctx.builder.get_local(&can_use_fast_path_local);

    ctx.builder.eqz_i32();
    ctx.builder.if_void();
    {
        ctx.builder.get_local(&address_local);

        match &value_local {
            GenSafeReadWriteValue::I32(l) => ctx.builder.get_local(l),
            GenSafeReadWriteValue::I64(l) => ctx.builder.get_local_i64(l),
        }

        ctx.builder
            .const_i32(ctx.start_of_current_instruction as i32 & 0xFFF);

        match bits {
            BitSize::BYTE => {
                ctx.builder.call_fn3_ret("safe_write8_slow_jit");
            },
            BitSize::WORD => {
                ctx.builder.call_fn3_ret("safe_write16_slow_jit");
            },
            BitSize::DWORD => {
                ctx.builder.call_fn3_ret("safe_write32_slow_jit");
            },
            BitSize::QWORD => {
                ctx.builder
                    .call_fn3_i32_i64_i32_ret("safe_write64_slow_jit");
            },
            BitSize::DQWORD => {
                dbg_assert!(false);
            },
        }

        if cfg!(debug_assertions) {
            ctx.builder.const_i32(1);
            ctx.builder.and_i32();

            ctx.builder.if_void();
            {
                // handled above
                ctx.builder.const_i32(match bits {
                    BitSize::BYTE => 8,
                    BitSize::WORD => 16,
                    BitSize::DWORD => 32,
                    BitSize::QWORD => 64,
                    _ => {
                        dbg_assert!(false);
                        0
                    },
                });
                ctx.builder.get_local(&address_local);
                ctx.builder.call_fn2("bug_gen_safe_read_write_page_fault");
            }
            ctx.builder.block_end();
        }
        else {
            ctx.builder.drop_();
        }
    }
    ctx.builder.block_end();

    ctx.builder.get_local(&phys_addr_local);
    match &value_local {
        GenSafeReadWriteValue::I32(l) => ctx.builder.get_local(l),
        GenSafeReadWriteValue::I64(l) => ctx.builder.get_local_i64(l),
    }

    match bits {
        BitSize::BYTE => {
            ctx.builder.store_u8(0);
        },
        BitSize::WORD => {
            ctx.builder.store_unaligned_u16(0);
        },
        BitSize::DWORD => {
            ctx.builder.store_unaligned_i32(0);
        },
        BitSize::QWORD => {
            ctx.builder.store_unaligned_i64(0);
        },
        BitSize::DQWORD => {
            dbg_assert!(false);
        },
    }

    match value_local {
        GenSafeReadWriteValue::I32(l) => ctx.builder.free_local(l),
        GenSafeReadWriteValue::I64(l) => ctx.builder.free_local_i64(l),
    }
    ctx.builder.free_local(can_use_fast_path_local);
    ctx.builder.free_local(phys_addr_local);
}

#[cfg(debug_assertions)]
#[no_mangle]
pub fn bug_gen_safe_read_write_page_fault(bits: i32, addr: u32) {
    dbg_log!("bug: gen_safe_read_write_page_fault {} {:x}", bits, addr);
    dbg_assert!(false);
}

pub fn gen_jmp_rel16(builder: &mut WasmBuilder, rel16: u16) {
    let cs_offset_addr = global_pointers::get_seg_offset(regs::CS);
    builder.load_fixed_i32(cs_offset_addr);
    let local = builder.set_new_local();

    // generate:
    // *instruction_pointer = cs_offset + ((*instruction_pointer - cs_offset + rel16) & 0xFFFF);
    {
        builder.const_i32(global_pointers::instruction_pointer as i32);

        gen_get_eip(builder);
        builder.get_local(&local);
        builder.sub_i32();

        builder.const_i32(rel16 as i32);
        builder.add_i32();

        builder.const_i32(0xFFFF);
        builder.and_i32();

        builder.get_local(&local);
        builder.add_i32();

        builder.store_aligned_i32(0);
    }
    builder.free_local(local);
}

pub fn gen_pop16_ss16(ctx: &mut JitContext) {
    // sp = segment_offsets[SS] + reg16[SP] (or just reg16[SP] if has_flat_segmentation)
    gen_get_reg16(ctx, regs::SP);

    if !ctx.cpu.has_flat_segmentation() {
        gen_get_ss_offset(ctx);
        ctx.builder.add_i32();
    }

    // result = safe_read16(sp)
    let address_local = ctx.builder.set_new_local();
    gen_safe_read16(ctx, &address_local);
    ctx.builder.free_local(address_local);

    // reg16[SP] += 2;
    gen_get_reg16(ctx, regs::SP);
    ctx.builder.const_i32(2);
    ctx.builder.add_i32();
    gen_set_reg16(ctx, regs::SP);

    // return value is already on stack
}

pub fn gen_pop16_ss32(ctx: &mut JitContext) {
    // esp = segment_offsets[SS] + reg32[ESP] (or just reg32[ESP] if has_flat_segmentation)
    gen_get_reg32(ctx, regs::ESP);

    if !ctx.cpu.has_flat_segmentation() {
        gen_get_ss_offset(ctx);
        ctx.builder.add_i32();
    }

    // result = safe_read16(esp)
    let address_local = ctx.builder.set_new_local();
    gen_safe_read16(ctx, &address_local);
    ctx.builder.free_local(address_local);

    // reg32[ESP] += 2;
    gen_get_reg32(ctx, regs::ESP);
    ctx.builder.const_i32(2);
    ctx.builder.add_i32();
    gen_set_reg32(ctx, regs::ESP);

    // return value is already on stack
}

pub fn gen_pop16(ctx: &mut JitContext) {
    if ctx.cpu.ssize_32() {
        gen_pop16_ss32(ctx);
    }
    else {
        gen_pop16_ss16(ctx);
    }
}

pub fn gen_pop32s_ss16(ctx: &mut JitContext) {
    // sp = reg16[SP]
    gen_get_reg16(ctx, regs::SP);

    // result = safe_read32s(segment_offsets[SS] + sp) (or just sp if has_flat_segmentation)
    if !ctx.cpu.has_flat_segmentation() {
        gen_get_ss_offset(ctx);
        ctx.builder.add_i32();
    }

    let address_local = ctx.builder.set_new_local();
    gen_safe_read32(ctx, &address_local);
    ctx.builder.free_local(address_local);

    // reg16[SP] = sp + 4;
    gen_get_reg16(ctx, regs::SP);
    ctx.builder.const_i32(4);
    ctx.builder.add_i32();
    gen_set_reg16(ctx, regs::SP);

    // return value is already on stack
}

pub fn gen_pop32s_ss32(ctx: &mut JitContext) {
    if !ctx.cpu.has_flat_segmentation() {
        gen_get_reg32(ctx, regs::ESP);
        gen_get_ss_offset(ctx);
        ctx.builder.add_i32();
        let address_local = ctx.builder.set_new_local();
        gen_safe_read32(ctx, &address_local);
        ctx.builder.free_local(address_local);
    }
    else {
        let reg = ctx.register_locals[regs::ESP as usize].unsafe_clone();
        gen_safe_read32(ctx, &reg);
    }

    gen_get_reg32(ctx, regs::ESP);
    ctx.builder.const_i32(4);
    ctx.builder.add_i32();
    gen_set_reg32(ctx, regs::ESP);

    // return value is already on stack
}

pub fn gen_pop32s(ctx: &mut JitContext) {
    if ctx.cpu.ssize_32() {
        gen_pop32s_ss32(ctx);
    }
    else {
        gen_pop32s_ss16(ctx);
    }
}

pub fn gen_adjust_stack_reg(ctx: &mut JitContext, offset: u32) {
    if ctx.cpu.ssize_32() {
        gen_get_reg32(ctx, regs::ESP);
        ctx.builder.const_i32(offset as i32);
        ctx.builder.add_i32();
        gen_set_reg32(ctx, regs::ESP);
    }
    else {
        gen_get_reg16(ctx, regs::SP);
        ctx.builder.const_i32(offset as i32);
        ctx.builder.add_i32();
        gen_set_reg16(ctx, regs::SP);
    }
}

pub fn gen_leave(ctx: &mut JitContext, os32: bool) {
    // [e]bp = safe_read{16,32}([e]bp)

    if ctx.cpu.ssize_32() {
        gen_get_reg32(ctx, regs::EBP);
    }
    else {
        gen_get_reg16(ctx, regs::BP);
    }

    let old_vbp = ctx.builder.tee_new_local();

    if !ctx.cpu.has_flat_segmentation() {
        gen_get_ss_offset(ctx);
        ctx.builder.add_i32();
    }
    if os32 {
        let address_local = ctx.builder.set_new_local();
        gen_safe_read32(ctx, &address_local);
        ctx.builder.free_local(address_local);
        gen_set_reg32(ctx, regs::EBP);
    }
    else {
        let address_local = ctx.builder.set_new_local();
        gen_safe_read16(ctx, &address_local);
        ctx.builder.free_local(address_local);
        gen_set_reg16(ctx, regs::BP);
    }

    // [e]sp = [e]bp + (os32 ? 4 : 2)

    if ctx.cpu.ssize_32() {
        ctx.builder.get_local(&old_vbp);
        ctx.builder.const_i32(if os32 { 4 } else { 2 });
        ctx.builder.add_i32();
        gen_set_reg32(ctx, regs::ESP);
    }
    else {
        ctx.builder.get_local(&old_vbp);
        ctx.builder.const_i32(if os32 { 4 } else { 2 });
        ctx.builder.add_i32();
        gen_set_reg16(ctx, regs::SP);
    }

    ctx.builder.free_local(old_vbp);
}

pub fn gen_task_switch_test(ctx: &mut JitContext) {
    // generate if(cr[0] & (CR0_EM | CR0_TS)) { task_switch_test_jit(); goto exit_with_fault; }
    let cr0_offset = global_pointers::get_creg_offset(0);

    dbg_assert!(regs::CR0_EM | regs::CR0_TS <= 0xFF);
    ctx.builder.load_fixed_u8(cr0_offset);
    ctx.builder.const_i32((regs::CR0_EM | regs::CR0_TS) as i32);
    ctx.builder.and_i32();

    ctx.builder.if_void();
    {
        gen_debug_track_jit_exit(ctx.builder, ctx.start_of_current_instruction);
        gen_fn1_const(
            ctx.builder,
            "task_switch_test_jit",
            ctx.start_of_current_instruction & 0xFFF,
        );
        ctx.builder.br(ctx.exit_with_fault_label);
    }
    ctx.builder.block_end();
}

pub fn gen_task_switch_test_mmx(ctx: &mut JitContext) {
    // generate if(cr[0] & (CR0_EM | CR0_TS)) { task_switch_test_mmx_jit(); goto exit_with_fault; }
    let cr0_offset = global_pointers::get_creg_offset(0);

    dbg_assert!(regs::CR0_EM | regs::CR0_TS <= 0xFF);
    ctx.builder.load_fixed_u8(cr0_offset);
    ctx.builder.const_i32((regs::CR0_EM | regs::CR0_TS) as i32);
    ctx.builder.and_i32();

    ctx.builder.if_void();
    {
        gen_debug_track_jit_exit(ctx.builder, ctx.start_of_current_instruction);
        gen_fn1_const(
            ctx.builder,
            "task_switch_test_mmx_jit",
            ctx.start_of_current_instruction & 0xFFF,
        );
        ctx.builder.br(ctx.exit_with_fault_label);
    }
    ctx.builder.block_end();
}

pub fn gen_push16(ctx: &mut JitContext, value_local: &WasmLocal) {
    if ctx.cpu.ssize_32() {
        gen_get_reg32(ctx, regs::ESP);
    }
    else {
        gen_get_reg16(ctx, regs::SP);
    };

    ctx.builder.const_i32(2);
    ctx.builder.sub_i32();

    let reg_updated_local = if !ctx.cpu.ssize_32() || !ctx.cpu.has_flat_segmentation() {
        let reg_updated_local = ctx.builder.tee_new_local();
        if !ctx.cpu.ssize_32() {
            ctx.builder.const_i32(0xFFFF);
            ctx.builder.and_i32();
        }

        if !ctx.cpu.has_flat_segmentation() {
            gen_get_ss_offset(ctx);
            ctx.builder.add_i32();
        }

        let sp_local = ctx.builder.set_new_local();
        gen_safe_write16(ctx, &sp_local, &value_local);
        ctx.builder.free_local(sp_local);

        ctx.builder.get_local(&reg_updated_local);
        reg_updated_local
    }
    else {
        // short path: The address written to is equal to ESP/SP minus two
        let reg_updated_local = ctx.builder.tee_new_local();
        gen_safe_write16(ctx, &reg_updated_local, &value_local);
        reg_updated_local
    };

    if ctx.cpu.ssize_32() {
        gen_set_reg32(ctx, regs::ESP);
    }
    else {
        gen_set_reg16(ctx, regs::SP);
    };
    ctx.builder.free_local(reg_updated_local);
}

pub fn gen_push32(ctx: &mut JitContext, value_local: &WasmLocal) {
    if ctx.cpu.ssize_32() {
        gen_get_reg32(ctx, regs::ESP);
    }
    else {
        gen_get_reg16(ctx, regs::SP);
    };

    ctx.builder.const_i32(4);
    ctx.builder.sub_i32();

    let new_sp_local = if !ctx.cpu.ssize_32() || !ctx.cpu.has_flat_segmentation() {
        let new_sp_local = ctx.builder.tee_new_local();
        if !ctx.cpu.ssize_32() {
            ctx.builder.const_i32(0xFFFF);
            ctx.builder.and_i32();
        }

        if !ctx.cpu.has_flat_segmentation() {
            gen_get_ss_offset(ctx);
            ctx.builder.add_i32();
        }

        let sp_local = ctx.builder.set_new_local();

        gen_safe_write32(ctx, &sp_local, &value_local);
        ctx.builder.free_local(sp_local);

        ctx.builder.get_local(&new_sp_local);
        new_sp_local
    }
    else {
        // short path: The address written to is equal to ESP/SP minus four
        let new_sp_local = ctx.builder.tee_new_local();
        if !gen_push32_coalesced_write(ctx, &new_sp_local, &value_local) {
            gen_safe_write32(ctx, &new_sp_local, &value_local);
        }
        new_sp_local
    };

    if ctx.cpu.ssize_32() {
        gen_set_reg32(ctx, regs::ESP);
    }
    else {
        gen_set_reg16(ctx, regs::SP);
    };
    ctx.builder.free_local(new_sp_local);
}

pub fn gen_push32_sreg(ctx: &mut JitContext, reg: u32) {
    gen_get_sreg(ctx, reg);
    let value_local = ctx.builder.set_new_local();

    if ctx.cpu.ssize_32() {
        gen_get_reg32(ctx, regs::ESP);
    }
    else {
        gen_get_reg16(ctx, regs::SP);
    };

    ctx.builder.const_i32(4);
    ctx.builder.sub_i32();

    let new_sp_local = if !ctx.cpu.ssize_32() || !ctx.cpu.has_flat_segmentation() {
        let new_sp_local = ctx.builder.tee_new_local();
        if !ctx.cpu.ssize_32() {
            ctx.builder.const_i32(0xFFFF);
            ctx.builder.and_i32();
        }

        if !ctx.cpu.has_flat_segmentation() {
            gen_get_ss_offset(ctx);
            ctx.builder.add_i32();
        }

        let sp_local = ctx.builder.set_new_local();

        gen_safe_write16(ctx, &sp_local, &value_local);
        ctx.builder.free_local(sp_local);

        ctx.builder.get_local(&new_sp_local);
        new_sp_local
    }
    else {
        // short path: The address written to is equal to ESP/SP minus four
        let new_sp_local = ctx.builder.tee_new_local();
        gen_safe_write16(ctx, &new_sp_local, &value_local);
        new_sp_local
    };

    if ctx.cpu.ssize_32() {
        gen_set_reg32(ctx, regs::ESP);
    }
    else {
        gen_set_reg16(ctx, regs::SP);
    };
    ctx.builder.free_local(new_sp_local);
    ctx.builder.free_local(value_local);
}

pub fn gen_get_real_eip(ctx: &mut JitContext) {
    gen_get_eip(ctx.builder);
    ctx.builder.const_i32(!0xFFF);
    ctx.builder.and_i32();
    ctx.builder.const_i32(ctx.cpu.eip as i32 & 0xFFF);
    ctx.builder.or_i32();
    if !ctx.cpu.has_flat_segmentation() {
        ctx.builder
            .load_fixed_i32(global_pointers::get_seg_offset(regs::CS));
        ctx.builder.sub_i32();
    }
}

// Flag-local slot order (must match jit_generate_module's registration):
pub const FLAG_LOCAL_LAST_OP1: usize = 0;
pub const FLAG_LOCAL_LAST_RESULT: usize = 1;
pub const FLAG_LOCAL_LAST_OP_SIZE: usize = 2;
pub const FLAG_LOCAL_FLAGS_CHANGED: usize = 3;
pub const FLAG_LOCAL_FLAGS: usize = 4;

pub fn gen_set_last_op1(builder: &mut WasmBuilder, source: &WasmLocal) {
    if builder.flag_locals.is_some() {
        builder.get_local(&source);
        builder.flag_local_set(FLAG_LOCAL_LAST_OP1);
        return;
    }
    builder.const_i32(global_pointers::last_op1 as i32);
    builder.get_local(&source);
    builder.store_aligned_i32(0);
}

pub fn gen_set_last_result(builder: &mut WasmBuilder, source: &WasmLocal) {
    if builder.flag_locals.is_some() {
        builder.get_local(&source);
        builder.flag_local_set(FLAG_LOCAL_LAST_RESULT);
        return;
    }
    builder.const_i32(global_pointers::last_result as i32);
    builder.get_local(&source);
    builder.store_aligned_i32(0);
}

pub fn gen_clear_flags_changed_bits(builder: &mut WasmBuilder, bits_to_clear: i32) {
    if builder.flag_locals.is_some() {
        gen_get_flags_changed(builder);
        builder.const_i32(!bits_to_clear);
        builder.and_i32();
        builder.flag_local_set(FLAG_LOCAL_FLAGS_CHANGED);
        return;
    }
    builder.const_i32(global_pointers::flags_changed as i32);
    gen_get_flags_changed(builder);
    builder.const_i32(!bits_to_clear);
    builder.and_i32();
    builder.store_aligned_i32(0);
}

pub fn gen_set_last_op_size_and_flags_changed(
    builder: &mut WasmBuilder,
    last_op_size: i32,
    flags_changed: i32,
) {
    dbg_assert!(last_op_size == OPSIZE_8 || last_op_size == OPSIZE_16 || last_op_size == OPSIZE_32);
    if builder.flag_locals.is_some() {
        builder.const_i32(last_op_size);
        builder.flag_local_set(FLAG_LOCAL_LAST_OP_SIZE);
        builder.const_i32(flags_changed);
        builder.flag_local_set(FLAG_LOCAL_FLAGS_CHANGED);
        return;
    }
    dbg_assert!(global_pointers::last_op_size as i32 % 8 == 0);
    dbg_assert!(global_pointers::last_op_size as i32 + 4 == global_pointers::flags_changed as i32);
    builder.const_i32(global_pointers::last_op_size as i32);
    builder.const_i64(last_op_size as u32 as i64 | (flags_changed as u32 as i64) << 32);
    builder.store_aligned_i64(0);
}

pub fn gen_set_flags_bits(builder: &mut WasmBuilder, bits_to_set: i32) {
    if builder.flag_locals.is_some() {
        gen_get_flags(builder);
        builder.const_i32(bits_to_set);
        builder.or_i32();
        builder.flag_local_set(FLAG_LOCAL_FLAGS);
        return;
    }
    builder.const_i32(global_pointers::flags as i32);
    gen_get_flags(builder);
    builder.const_i32(bits_to_set);
    builder.or_i32();
    builder.store_aligned_i32(0);
}

pub fn gen_clear_flags_bits(builder: &mut WasmBuilder, bits_to_clear: i32) {
    if builder.flag_locals.is_some() {
        gen_get_flags(builder);
        builder.const_i32(!bits_to_clear);
        builder.and_i32();
        builder.flag_local_set(FLAG_LOCAL_FLAGS);
        return;
    }
    builder.const_i32(global_pointers::flags as i32);
    gen_get_flags(builder);
    builder.const_i32(!bits_to_clear);
    builder.and_i32();
    builder.store_aligned_i32(0);
}

#[derive(PartialEq)]
pub enum ConditionNegate {
    True,
    False,
}

pub fn gen_getzf(ctx: &mut JitContext, negate: ConditionNegate) {
    match &ctx.previous_instruction {
        Instruction::Cmp {
            dest: InstructionOperandDest::WasmLocal(dest),
            source: InstructionOperand::WasmLocal(source),
            opsize: OPSIZE_32,
        } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            ctx.builder.get_local(dest);
            ctx.builder.get_local(source);
            if negate == ConditionNegate::False {
                ctx.builder.eq_i32();
            }
            else {
                ctx.builder.ne_i32();
            }
        },
        Instruction::Cmp {
            dest: InstructionOperandDest::WasmLocal(dest),
            source: InstructionOperand::Immediate(0),
            opsize: OPSIZE_32,
        } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            ctx.builder.get_local(dest);
            if negate == ConditionNegate::False {
                ctx.builder.eqz_i32();
            }
        },
        Instruction::Cmp {
            dest: InstructionOperandDest::WasmLocal(dest),
            source: InstructionOperand::Immediate(i),
            opsize: OPSIZE_32,
        } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            ctx.builder.get_local(dest);
            ctx.builder.const_i32(*i);
            if negate == ConditionNegate::False {
                ctx.builder.eq_i32();
            }
            else {
                ctx.builder.ne_i32();
            }
        },
        Instruction::Cmp { .. }
        | Instruction::Sub { .. }
        | Instruction::Add { .. }
        | Instruction::AdcSbb { .. }
        | Instruction::NonZeroShift { .. } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            gen_get_last_result(ctx.builder, &ctx.previous_instruction);
            if negate == ConditionNegate::False {
                ctx.builder.eqz_i32();
            }
        },
        Instruction::Bitwise { opsize, .. } => {
            let &opsize = opsize;
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            // Note: Necessary because test{8,16} don't mask either last_result or any of their operands
            // TODO: Use local instead of last_result for 8-bit/16-bit
            if opsize == OPSIZE_32 {
                gen_get_last_result(ctx.builder, &ctx.previous_instruction);
            }
            else if opsize == OPSIZE_16 {
                ctx.builder
                    .flag_load_u16(FLAG_LOCAL_LAST_RESULT, global_pointers::last_result as u32);
            }
            else if opsize == OPSIZE_8 {
                ctx.builder
                    .flag_load_u8(FLAG_LOCAL_LAST_RESULT, global_pointers::last_result as u32);
            }
            if negate == ConditionNegate::False {
                ctx.builder.eqz_i32();
            }
        },
        &Instruction::Other => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_UNOPTIMISED);
            gen_get_flags_changed(ctx.builder);
            ctx.builder.const_i32(FLAG_ZERO);
            ctx.builder.and_i32();
            ctx.builder.if_i32();

            gen_get_last_result(ctx.builder, &ctx.previous_instruction);
            let last_result = ctx.builder.tee_new_local();
            ctx.builder.const_i32(-1);
            ctx.builder.xor_i32();
            ctx.builder.get_local(&last_result);
            ctx.builder.free_local(last_result);
            ctx.builder.const_i32(1);
            ctx.builder.sub_i32();
            ctx.builder.and_i32();
            gen_get_last_op_size(ctx.builder);
            ctx.builder.shr_u_i32();
            ctx.builder.const_i32(1);
            ctx.builder.and_i32();

            ctx.builder.else_();
            gen_get_flags(ctx.builder);
            ctx.builder.const_i32(FLAG_ZERO);
            ctx.builder.and_i32();
            ctx.builder.block_end();

            if negate == ConditionNegate::True {
                ctx.builder.eqz_i32();
            }
        },
    }
}

pub fn gen_getcf(ctx: &mut JitContext, negate: ConditionNegate) {
    match &ctx.previous_instruction {
        Instruction::Cmp { source, opsize, .. }
        | Instruction::Sub {
            source,
            opsize,
            is_dec: false,
            ..
        } => {
            // Note: x < y and x < x - y can be used interchangeably (see getcf)
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
            match (opsize, source) {
                (&OPSIZE_32, InstructionOperand::WasmLocal(l)) => ctx.builder.get_local(l),
                (_, &InstructionOperand::Immediate(i)) => ctx.builder.const_i32(i),
                _ => gen_get_last_result(ctx.builder, &ctx.previous_instruction),
            }
            if negate == ConditionNegate::True {
                ctx.builder.geu_i32();
            }
            else {
                ctx.builder.ltu_i32();
            }
        },
        Instruction::Add {
            source,
            opsize,
            is_inc: false,
            ..
        } => {
            gen_get_last_result(ctx.builder, &ctx.previous_instruction);
            match (opsize, source) {
                (&OPSIZE_32, InstructionOperand::WasmLocal(l)) => ctx.builder.get_local(l),
                (_, &InstructionOperand::Immediate(i)) => ctx.builder.const_i32(i),
                _ => gen_get_last_op1(ctx.builder, &ctx.previous_instruction),
            }
            if negate == ConditionNegate::True {
                ctx.builder.geu_i32();
            }
            else {
                ctx.builder.ltu_i32();
            }
        },
        Instruction::Add { is_inc: true, .. } | Instruction::Sub { is_dec: true, .. } => {
            gen_get_flags(ctx.builder);
            ctx.builder.const_i32(FLAG_CARRY);
            ctx.builder.and_i32();
            if negate == ConditionNegate::True {
                ctx.builder.eqz_i32();
            }
        },
        Instruction::Bitwise { .. } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            ctx.builder
                .const_i32(if negate == ConditionNegate::True { 1 } else { 0 });
        },
        Instruction::NonZeroShift { .. } | Instruction::AdcSbb { .. } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            gen_get_flags(ctx.builder);
            ctx.builder.const_i32(FLAG_CARRY);
            ctx.builder.and_i32();
            if negate == ConditionNegate::True {
                ctx.builder.eqz_i32();
            }
        },
        &Instruction::Other => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_UNOPTIMISED);

            gen_get_flags_changed(ctx.builder);
            let flags_changed = ctx.builder.tee_new_local();
            ctx.builder.const_i32(FLAG_CARRY);
            ctx.builder.and_i32();
            ctx.builder.if_i32();

            ctx.builder.get_local(&flags_changed);
            ctx.builder.const_i32(31);
            ctx.builder.shr_s_i32();
            ctx.builder.free_local(flags_changed);
            let sub_mask = ctx.builder.set_new_local();

            gen_get_last_result(ctx.builder, &ctx.previous_instruction);
            ctx.builder.get_local(&sub_mask);
            ctx.builder.xor_i32();

            gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
            ctx.builder.get_local(&sub_mask);
            ctx.builder.xor_i32();

            ctx.builder.ltu_i32();

            ctx.builder.else_();
            gen_get_flags(ctx.builder);
            ctx.builder.const_i32(FLAG_CARRY);
            ctx.builder.and_i32();
            ctx.builder.block_end();

            ctx.builder.free_local(sub_mask);

            if negate == ConditionNegate::True {
                ctx.builder.eqz_i32();
            }
        },
    }
}

pub fn gen_getsf(ctx: &mut JitContext, negate: ConditionNegate) {
    match &ctx.previous_instruction {
        Instruction::Cmp { opsize, .. }
        | Instruction::Sub { opsize, .. }
        | Instruction::Add { opsize, .. }
        | Instruction::AdcSbb { opsize, .. }
        | Instruction::Bitwise { opsize, .. }
        | Instruction::NonZeroShift { opsize, .. } => {
            let &opsize = opsize;
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            gen_get_last_result(ctx.builder, &ctx.previous_instruction);
            if opsize == OPSIZE_32 {
                ctx.builder.const_i32(0);
                if negate == ConditionNegate::True {
                    ctx.builder.ge_i32();
                }
                else {
                    ctx.builder.lt_i32();
                }
            }
            else {
                // TODO: use register (see get_last_result)
                ctx.builder
                    .const_i32(if opsize == OPSIZE_16 { 0x8000 } else { 0x80 });
                ctx.builder.and_i32();
                if negate == ConditionNegate::True {
                    ctx.builder.eqz_i32();
                }
            }
        },
        &Instruction::Other => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_UNOPTIMISED);
            gen_get_flags_changed(ctx.builder);
            ctx.builder.const_i32(FLAG_SIGN);
            ctx.builder.and_i32();
            ctx.builder.if_i32();
            {
                gen_get_last_result(ctx.builder, &ctx.previous_instruction);
                gen_get_last_op_size(ctx.builder);
                ctx.builder.shr_u_i32();
                ctx.builder.const_i32(1);
                ctx.builder.and_i32();
            }
            ctx.builder.else_();
            {
                gen_get_flags(ctx.builder);
                ctx.builder.const_i32(FLAG_SIGN);
                ctx.builder.and_i32();
            }
            ctx.builder.block_end();
            if negate == ConditionNegate::True {
                ctx.builder.eqz_i32();
            }
        },
    }
}

pub fn gen_getof(ctx: &mut JitContext) {
    match &ctx.previous_instruction {
        Instruction::Cmp { opsize, .. } | Instruction::Sub { opsize, .. } => {
            // TODO: a better formula might be possible
            let &opsize = opsize;
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
            gen_get_last_result(ctx.builder, &ctx.previous_instruction);
            ctx.builder.xor_i32();

            gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
            gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
            gen_get_last_result(ctx.builder, &ctx.previous_instruction);
            ctx.builder.sub_i32();
            ctx.builder.xor_i32();
            ctx.builder.and_i32();

            ctx.builder.const_i32(if opsize == OPSIZE_32 {
                0x8000_0000u32 as i32
            }
            else if opsize == OPSIZE_16 {
                0x8000
            }
            else {
                0x80
            });
            ctx.builder.and_i32();
        },
        Instruction::Add { opsize, .. } => {
            // TODO: a better formula might be possible
            let &opsize = opsize;
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
            gen_get_last_result(ctx.builder, &ctx.previous_instruction);
            ctx.builder.xor_i32();

            gen_get_last_result(ctx.builder, &ctx.previous_instruction);
            gen_get_last_result(ctx.builder, &ctx.previous_instruction);
            gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
            ctx.builder.sub_i32();
            ctx.builder.xor_i32();
            ctx.builder.and_i32();

            ctx.builder.const_i32(if opsize == OPSIZE_32 {
                0x8000_0000u32 as i32
            }
            else if opsize == OPSIZE_16 {
                0x8000
            }
            else {
                0x80
            });
            ctx.builder.and_i32();
        },
        Instruction::Bitwise { .. } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            ctx.builder.const_i32(0);
        },
        Instruction::NonZeroShift { .. } | Instruction::AdcSbb { .. } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            gen_get_flags(ctx.builder);
            ctx.builder.const_i32(FLAG_OVERFLOW);
            ctx.builder.and_i32();
        },
        &Instruction::Other => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_UNOPTIMISED);
            gen_get_flags_changed(ctx.builder);
            let flags_changed = ctx.builder.tee_new_local();
            ctx.builder.const_i32(FLAG_OVERFLOW);
            ctx.builder.and_i32();
            ctx.builder.if_i32();
            {
                gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
                let last_op1 = ctx.builder.tee_new_local();
                gen_get_last_result(ctx.builder, &ctx.previous_instruction);
                let last_result = ctx.builder.tee_new_local();
                ctx.builder.xor_i32();

                ctx.builder.get_local(&last_result);
                ctx.builder.get_local(&last_op1);
                ctx.builder.sub_i32();
                gen_get_flags_changed(ctx.builder);
                ctx.builder.const_i32(31);
                ctx.builder.shr_u_i32();
                ctx.builder.sub_i32();

                ctx.builder.get_local(&last_result);
                ctx.builder.xor_i32();

                ctx.builder.and_i32();

                gen_get_last_op_size(ctx.builder);
                ctx.builder.shr_u_i32();
                ctx.builder.const_i32(1);
                ctx.builder.and_i32();

                ctx.builder.free_local(last_op1);
                ctx.builder.free_local(last_result);
            }
            ctx.builder.else_();
            {
                gen_get_flags(ctx.builder);
                ctx.builder.const_i32(FLAG_OVERFLOW);
                ctx.builder.and_i32();
            }
            ctx.builder.block_end();
            ctx.builder.free_local(flags_changed);
        },
    }
}

pub fn gen_test_be(ctx: &mut JitContext, negate: ConditionNegate) {
    match &ctx.previous_instruction {
        Instruction::Cmp {
            dest,
            source,
            opsize,
        } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            match dest {
                InstructionOperandDest::WasmLocal(l) => {
                    ctx.builder.get_local(l);
                    if *opsize == OPSIZE_8 || *opsize == OPSIZE_16 {
                        ctx.builder
                            .const_i32(if *opsize == OPSIZE_8 { 0xFF } else { 0xFFFF });
                        ctx.builder.and_i32();
                    }
                },
                InstructionOperandDest::Other => {
                    gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
                },
            }
            match source {
                InstructionOperand::WasmLocal(l) => {
                    ctx.builder.get_local(l);
                    if *opsize == OPSIZE_8 || *opsize == OPSIZE_16 {
                        ctx.builder
                            .const_i32(if *opsize == OPSIZE_8 { 0xFF } else { 0xFFFF });
                        ctx.builder.and_i32();
                    }
                },
                InstructionOperand::Other => {
                    gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
                    gen_get_last_result(ctx.builder, &ctx.previous_instruction);
                    ctx.builder.sub_i32();
                    if *opsize == OPSIZE_8 || *opsize == OPSIZE_16 {
                        ctx.builder
                            .const_i32(if *opsize == OPSIZE_8 { 0xFF } else { 0xFFFF });
                        ctx.builder.and_i32();
                    }
                },
                &InstructionOperand::Immediate(i) => {
                    dbg_assert!(*opsize != OPSIZE_8 || i >= 0 && i < 0x100);
                    dbg_assert!(*opsize != OPSIZE_16 || i >= 0 && i < 0x10000);
                    ctx.builder.const_i32(i);
                },
            }

            if negate == ConditionNegate::True {
                ctx.builder.gtu_i32();
            }
            else {
                ctx.builder.leu_i32();
            }
        },
        Instruction::Sub {
            opsize,
            source,
            is_dec: false,
            ..
        } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);

            gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
            match (opsize, source) {
                (&OPSIZE_32, InstructionOperand::WasmLocal(l)) => ctx.builder.get_local(l),
                (_, &InstructionOperand::Immediate(i)) => ctx.builder.const_i32(i),
                _ => {
                    gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
                    gen_get_last_result(ctx.builder, &ctx.previous_instruction);
                    ctx.builder.sub_i32();
                    if *opsize == OPSIZE_8 || *opsize == OPSIZE_16 {
                        ctx.builder
                            .const_i32(if *opsize == OPSIZE_8 { 0xFF } else { 0xFFFF });
                        ctx.builder.and_i32();
                    }
                },
            }

            if negate == ConditionNegate::True {
                ctx.builder.gtu_i32();
            }
            else {
                ctx.builder.leu_i32();
            }
        },
        &Instruction::Bitwise { .. } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            gen_getzf(ctx, negate);
        },
        &Instruction::Add { .. } | &Instruction::Sub { is_dec: true, .. } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            // not the best code generation, but reasonable for this fairly uncommon case
            gen_getcf(ctx, ConditionNegate::False);
            gen_getzf(ctx, ConditionNegate::False);
            ctx.builder.or_i32();
            if negate == ConditionNegate::True {
                ctx.builder.eqz_i32();
            }
        },
        Instruction::Other | Instruction::NonZeroShift { .. } | Instruction::AdcSbb { .. } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_UNOPTIMISED);
            gen_getcf(ctx, ConditionNegate::False);
            gen_getzf(ctx, ConditionNegate::False);
            ctx.builder.or_i32();
            if negate == ConditionNegate::True {
                ctx.builder.eqz_i32();
            }
        },
    }
}

pub fn gen_test_l(ctx: &mut JitContext, negate: ConditionNegate) {
    match &ctx.previous_instruction {
        Instruction::Cmp {
            dest,
            source,
            opsize,
        } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            match dest {
                InstructionOperandDest::WasmLocal(l) => {
                    ctx.builder.get_local(l);
                    if *opsize == OPSIZE_8 || *opsize == OPSIZE_16 {
                        ctx.builder
                            .const_i32(if *opsize == OPSIZE_8 { 24 } else { 16 });
                        ctx.builder.shl_i32();
                    }
                },
                InstructionOperandDest::Other => {
                    gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
                    if *opsize == OPSIZE_8 || *opsize == OPSIZE_16 {
                        ctx.builder
                            .const_i32(if *opsize == OPSIZE_8 { 24 } else { 16 });
                        ctx.builder.shl_i32();
                    }
                },
            }
            match source {
                InstructionOperand::WasmLocal(l) => {
                    ctx.builder.get_local(l);
                    if *opsize == OPSIZE_8 || *opsize == OPSIZE_16 {
                        ctx.builder
                            .const_i32(if *opsize == OPSIZE_8 { 24 } else { 16 });
                        ctx.builder.shl_i32();
                    }
                },
                InstructionOperand::Other => {
                    gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
                    gen_get_last_result(ctx.builder, &ctx.previous_instruction);
                    ctx.builder.sub_i32();
                    if *opsize == OPSIZE_8 || *opsize == OPSIZE_16 {
                        ctx.builder
                            .const_i32(if *opsize == OPSIZE_8 { 24 } else { 16 });
                        ctx.builder.shl_i32();
                    }
                },
                &InstructionOperand::Immediate(i) => {
                    ctx.builder.const_i32(i);
                    if *opsize == OPSIZE_8 || *opsize == OPSIZE_16 {
                        ctx.builder
                            .const_i32(if *opsize == OPSIZE_8 { 24 } else { 16 });
                        ctx.builder.shl_i32();
                    }
                },
            }
            if negate == ConditionNegate::True {
                ctx.builder.ge_i32();
            }
            else {
                ctx.builder.lt_i32();
            }
        },
        Instruction::Sub { opsize, source, .. } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
            if *opsize == OPSIZE_8 || *opsize == OPSIZE_16 {
                ctx.builder
                    .const_i32(if *opsize == OPSIZE_8 { 24 } else { 16 });
                ctx.builder.shl_i32();
            }
            match (opsize, source) {
                (&OPSIZE_32, InstructionOperand::WasmLocal(l)) => ctx.builder.get_local(l),
                (_, &InstructionOperand::Immediate(i)) => ctx.builder.const_i32(
                    i << if *opsize == OPSIZE_32 {
                        0
                    }
                    else if *opsize == OPSIZE_16 {
                        16
                    }
                    else {
                        24
                    },
                ),
                _ => {
                    gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
                    gen_get_last_result(ctx.builder, &ctx.previous_instruction);
                    ctx.builder.sub_i32();
                    if *opsize == OPSIZE_8 || *opsize == OPSIZE_16 {
                        ctx.builder
                            .const_i32(if *opsize == OPSIZE_8 { 24 } else { 16 });
                        ctx.builder.shl_i32();
                    }
                },
            }
            if negate == ConditionNegate::True {
                ctx.builder.ge_i32();
            }
            else {
                ctx.builder.lt_i32();
            }
        },
        &Instruction::Bitwise { .. } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            gen_getsf(ctx, negate);
        },
        &Instruction::Other
        | Instruction::Add { .. }
        | Instruction::NonZeroShift { .. }
        | Instruction::AdcSbb { .. } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_UNOPTIMISED);
            if let Instruction::Add { .. } = ctx.previous_instruction {
                gen_profiler_stat_increment(
                    ctx.builder,
                    profiler::stat::CONDITION_UNOPTIMISED_UNHANDLED_L,
                );
            }
            gen_getsf(ctx, ConditionNegate::False);
            ctx.builder.eqz_i32();
            gen_getof(ctx);
            ctx.builder.eqz_i32();
            ctx.builder.xor_i32();
            if negate == ConditionNegate::True {
                ctx.builder.eqz_i32();
            }
        },
    }
}

pub fn gen_test_le(ctx: &mut JitContext, negate: ConditionNegate) {
    match &ctx.previous_instruction {
        Instruction::Cmp {
            dest,
            source,
            opsize,
        } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            match dest {
                InstructionOperandDest::WasmLocal(l) => {
                    ctx.builder.get_local(l);
                    if *opsize == OPSIZE_8 || *opsize == OPSIZE_16 {
                        ctx.builder
                            .const_i32(if *opsize == OPSIZE_8 { 24 } else { 16 });
                        ctx.builder.shl_i32();
                    }
                },
                InstructionOperandDest::Other => {
                    gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
                    if *opsize == OPSIZE_8 || *opsize == OPSIZE_16 {
                        ctx.builder
                            .const_i32(if *opsize == OPSIZE_8 { 24 } else { 16 });
                        ctx.builder.shl_i32();
                    }
                },
            }
            match source {
                InstructionOperand::WasmLocal(l) => {
                    ctx.builder.get_local(l);
                    if *opsize == OPSIZE_8 || *opsize == OPSIZE_16 {
                        ctx.builder
                            .const_i32(if *opsize == OPSIZE_8 { 24 } else { 16 });
                        ctx.builder.shl_i32();
                    }
                },
                InstructionOperand::Other => {
                    gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
                    gen_get_last_result(ctx.builder, &ctx.previous_instruction);
                    ctx.builder.sub_i32();
                    if *opsize == OPSIZE_8 || *opsize == OPSIZE_16 {
                        ctx.builder
                            .const_i32(if *opsize == OPSIZE_8 { 24 } else { 16 });
                        ctx.builder.shl_i32();
                    }
                },
                &InstructionOperand::Immediate(i) => {
                    ctx.builder.const_i32(i);
                    if *opsize == OPSIZE_8 || *opsize == OPSIZE_16 {
                        ctx.builder
                            .const_i32(if *opsize == OPSIZE_8 { 24 } else { 16 });
                        ctx.builder.shl_i32();
                    }
                },
            }
            if negate == ConditionNegate::True {
                ctx.builder.gt_i32();
            }
            else {
                ctx.builder.le_i32();
            }
        },
        Instruction::Sub { opsize, source, .. } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
            if *opsize == OPSIZE_8 || *opsize == OPSIZE_16 {
                ctx.builder
                    .const_i32(if *opsize == OPSIZE_8 { 24 } else { 16 });
                ctx.builder.shl_i32();
            }
            match (opsize, source) {
                (&OPSIZE_32, InstructionOperand::WasmLocal(l)) => ctx.builder.get_local(l),
                (_, &InstructionOperand::Immediate(i)) => ctx.builder.const_i32(
                    i << if *opsize == OPSIZE_32 {
                        0
                    }
                    else if *opsize == OPSIZE_16 {
                        16
                    }
                    else {
                        24
                    },
                ),
                _ => {
                    gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
                    gen_get_last_result(ctx.builder, &ctx.previous_instruction);
                    ctx.builder.sub_i32();
                    if *opsize == OPSIZE_8 || *opsize == OPSIZE_16 {
                        ctx.builder
                            .const_i32(if *opsize == OPSIZE_8 { 24 } else { 16 });
                        ctx.builder.shl_i32();
                    }
                },
            }
            if negate == ConditionNegate::True {
                ctx.builder.gt_i32();
            }
            else {
                ctx.builder.le_i32();
            }
        },
        &Instruction::Bitwise { .. } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            // TODO: Could probably be improved (<= 0)
            gen_test_l(ctx, ConditionNegate::False);
            gen_getzf(ctx, ConditionNegate::False);
            ctx.builder.or_i32();
            if negate == ConditionNegate::True {
                ctx.builder.eqz_i32();
            }
        },
        Instruction::Other
        | Instruction::Add { .. }
        | Instruction::NonZeroShift { .. }
        | Instruction::AdcSbb { .. } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_UNOPTIMISED);
            if let Instruction::Add { .. } = ctx.previous_instruction {
                gen_profiler_stat_increment(
                    ctx.builder,
                    profiler::stat::CONDITION_UNOPTIMISED_UNHANDLED_LE,
                );
            }
            gen_test_l(ctx, ConditionNegate::False);
            gen_getzf(ctx, ConditionNegate::False);
            ctx.builder.or_i32();
            if negate == ConditionNegate::True {
                ctx.builder.eqz_i32();
            }
        },
    }
}

pub fn gen_test_loopnz(ctx: &mut JitContext, is_asize_32: bool) {
    gen_test_loop(ctx, is_asize_32);
    ctx.builder.eqz_i32();
    gen_getzf(ctx, ConditionNegate::False);
    ctx.builder.or_i32();
    ctx.builder.eqz_i32();
}
pub fn gen_test_loopz(ctx: &mut JitContext, is_asize_32: bool) {
    gen_test_loop(ctx, is_asize_32);
    ctx.builder.eqz_i32();
    gen_getzf(ctx, ConditionNegate::False);
    ctx.builder.eqz_i32();
    ctx.builder.or_i32();
    ctx.builder.eqz_i32();
}
pub fn gen_test_loop(ctx: &mut JitContext, is_asize_32: bool) {
    if is_asize_32 {
        gen_get_reg32(ctx, regs::ECX);
    }
    else {
        gen_get_reg16(ctx, regs::CX);
    }
}
pub fn gen_test_jcxz(ctx: &mut JitContext, is_asize_32: bool) {
    if is_asize_32 {
        gen_get_reg32(ctx, regs::ECX);
    }
    else {
        gen_get_reg16(ctx, regs::CX);
    }
    ctx.builder.eqz_i32();
}

// Every caller reproduces fpu_get_sti (cpu/fpu.rs) exactly: the strict-mode branches, the
// runtime fallbacks of the relaxed fast paths, and FLD ST(i). So the empty-slot contract
// (INDEFINITE_NAN + stack fault, not stale register bytes) holds here too — inlined rather
// than routed through the helper, which would put a call on FLD ST(i).
pub fn gen_fpu_get_sti(ctx: &mut JitContext, i: u32) {
    if crate::softfloat::is_fpu_relaxed() {
        // Relaxed mode: directly read from fpu_st array, skipping scratch register round-trip.
        // addr = fpu_st + ((fpu_stack_ptr + i) & 7) * 16
        ctx.builder
            .load_fixed_u8(global_pointers::fpu_stack_ptr as u32);
        ctx.builder.const_i32(i as i32);
        ctx.builder.add_i32();
        ctx.builder.const_i32(7);
        ctx.builder.and_i32();
        let index = ctx.builder.tee_new_local();
        ctx.builder.const_i32(16);
        ctx.builder.mul_i32();
        ctx.builder.const_i32(global_pointers::fpu_st as i32);
        ctx.builder.add_i32();
        let addr_local = ctx.builder.set_new_local();

        ctx.builder
            .load_fixed_u8(global_pointers::fpu_stack_empty as u32);
        ctx.builder.get_local(&index);
        ctx.builder.shr_u_i32();
        ctx.builder.const_i32(1);
        ctx.builder.and_i32();
        let empty = ctx.builder.set_new_local();

        // fpu_stack_fault, inline. Emitted before the value so nothing of ours is on the
        // wasm stack across the block.
        ctx.builder.get_local(&empty);
        ctx.builder.if_void();
        gen_mark_fpu_simd_dirty(ctx.builder);
        ctx.builder
            .const_i32(global_pointers::fpu_status_word as i32);
        ctx.builder
            .load_fixed_u16(global_pointers::fpu_status_word as u32);
        ctx.builder.const_i32(!FPU_C1);
        ctx.builder.and_i32();
        ctx.builder.const_i32(FPU_EX_SF | FPU_EX_I);
        ctx.builder.or_i32();
        ctx.builder.store_aligned_u16(0);
        ctx.builder.block_end();

        ctx.builder.const_i64(F80_INDEFINITE_NAN_MANTISSA);
        ctx.builder.get_local(&addr_local);
        ctx.builder.load_unaligned_i64(0); // mantissa
        ctx.builder.get_local(&empty);
        ctx.builder.select();
        ctx.builder.const_i32(F80_INDEFINITE_NAN_SIGN_EXPONENT);
        ctx.builder.get_local(&addr_local);
        ctx.builder.load_unaligned_u16(8); // sign_exponent
        ctx.builder.get_local(&empty);
        ctx.builder.select();

        ctx.builder.free_local(empty);
        ctx.builder.free_local(addr_local);
        ctx.builder.free_local(index);
    }
    else {
        ctx.builder
            .const_i32(global_pointers::sse_scratch_register as i32);
        ctx.builder.const_i32(i as i32);
        ctx.builder.call_fn2("fpu_get_sti_jit");
        ctx.builder
            .load_fixed_i64(global_pointers::sse_scratch_register as u32);
        ctx.builder
            .load_fixed_u16(global_pointers::sse_scratch_register as u32 + 8);
    }
}

pub fn gen_fpu_load_m32(ctx: &mut JitContext, modrm_byte: ModrmByte) {
    if crate::softfloat::is_fpu_relaxed() {
        // Relaxed mode: inline WASM type conversion ops, no scratch register.
        // F80{mantissa=f64bits(f32bits), sign_exponent=RELAXED_TAG}
        gen_modrm_resolve_safe_read32(ctx, modrm_byte); // i32 raw f32 bits
        ctx.builder.reinterpret_i32_as_f32();           // f32
        ctx.builder.promote_f32_to_f64();               // f64
        ctx.builder.reinterpret_f64_as_i64();           // i64 mantissa
        ctx.builder.const_i32(0x7FFE);                  // RELAXED_TAG sign_exponent
    }
    else {
        ctx.builder
            .const_i32(global_pointers::sse_scratch_register as i32);
        gen_modrm_resolve_safe_read32(ctx, modrm_byte);
        ctx.builder.call_fn2("f32_to_f80_jit");
        ctx.builder
            .load_fixed_i64(global_pointers::sse_scratch_register as u32);
        ctx.builder
            .load_fixed_u16(global_pointers::sse_scratch_register as u32 + 8);
    }
}

pub fn gen_fpu_load_m64(ctx: &mut JitContext, modrm_byte: ModrmByte) {
    if crate::softfloat::is_fpu_relaxed() {
        // Relaxed mode: raw f64 bits from memory are already the mantissa.
        // F80{mantissa=raw_bits, sign_exponent=RELAXED_TAG}
        gen_modrm_resolve_safe_read64(ctx, modrm_byte); // i64 raw f64 bits = mantissa
        ctx.builder.const_i32(0x7FFE);                  // RELAXED_TAG sign_exponent
    }
    else {
        ctx.builder
            .const_i32(global_pointers::sse_scratch_register as i32);
        gen_modrm_resolve_safe_read64(ctx, modrm_byte);
        ctx.builder.call_fn2_i32_i64("f64_to_f80_jit");
        ctx.builder
            .load_fixed_i64(global_pointers::sse_scratch_register as u32);
        ctx.builder
            .load_fixed_u16(global_pointers::sse_scratch_register as u32 + 8);
    }
}

// Load an int memory operand as a real 80-bit value (mantissa i64, sign_exponent u16)
// for the fpu_* helpers. Distinct from the relaxed (f64-bits, RELAXED_TAG) form below.
pub fn gen_fpu_load_i16_f80(ctx: &mut JitContext, modrm_byte: ModrmByte) {
    ctx.builder
        .const_i32(global_pointers::sse_scratch_register as i32);
    gen_modrm_resolve_safe_read16(ctx, modrm_byte);
    sign_extend_i16(ctx.builder);
    ctx.builder.call_fn2("i32_to_f80_jit");
    ctx.builder
        .load_fixed_i64(global_pointers::sse_scratch_register as u32);
    ctx.builder
        .load_fixed_u16(global_pointers::sse_scratch_register as u32 + 8);
}
pub fn gen_fpu_load_i32_f80(ctx: &mut JitContext, modrm_byte: ModrmByte) {
    ctx.builder
        .const_i32(global_pointers::sse_scratch_register as i32);
    gen_modrm_resolve_safe_read32(ctx, modrm_byte);
    ctx.builder.call_fn2("i32_to_f80_jit");
    ctx.builder
        .load_fixed_i64(global_pointers::sse_scratch_register as u32);
    ctx.builder
        .load_fixed_u16(global_pointers::sse_scratch_register as u32 + 8);
}
pub fn gen_fpu_load_i64_f80(ctx: &mut JitContext, modrm_byte: ModrmByte) {
    ctx.builder
        .const_i32(global_pointers::sse_scratch_register as i32);
    gen_modrm_resolve_safe_read64(ctx, modrm_byte);
    ctx.builder.call_fn2_i32_i64("i64_to_f80_jit");
    ctx.builder
        .load_fixed_i64(global_pointers::sse_scratch_register as u32);
    ctx.builder
        .load_fixed_u16(global_pointers::sse_scratch_register as u32 + 8);
}

// Relaxed mode: leave (f64-bits i64, RELAXED_TAG i32) for push_loaded / relaxed FICOM.
// Strict mode: fall through to the f80 loaders above.
pub fn gen_fpu_load_i16(ctx: &mut JitContext, modrm_byte: ModrmByte) {
    if crate::softfloat::is_fpu_relaxed() {
        gen_modrm_resolve_safe_read16(ctx, modrm_byte);
        sign_extend_i16(ctx.builder);
        ctx.builder.convert_i32_to_f64();
        ctx.builder.reinterpret_f64_as_i64();
        ctx.builder.const_i32(FPU_RELAXED_TAG);
        return;
    }
    gen_fpu_load_i16_f80(ctx, modrm_byte);
}
pub fn gen_fpu_load_i32(ctx: &mut JitContext, modrm_byte: ModrmByte) {
    if crate::softfloat::is_fpu_relaxed() {
        gen_modrm_resolve_safe_read32(ctx, modrm_byte);
        ctx.builder.convert_i32_to_f64();
        ctx.builder.reinterpret_f64_as_i64();
        ctx.builder.const_i32(FPU_RELAXED_TAG);
        return;
    }
    gen_fpu_load_i32_f80(ctx, modrm_byte);
}
/// `fild m64` has NO relaxed form: f64 carries 53 mantissa bits and an i64 needs 64, so
/// `convert_i64_to_f64` rounds away the low bits of any |value| >= 2^53. That is not a
/// precision nicety — `fild qword`/`fistp qword` is a 90s CRT block-COPY idiom (Carmageddon
/// 2's memcpy moves 8 bytes per pair), so the rounding lands in copied DATA: one corrupted
/// 16-bit word per 8 bytes. Load the true 80-bit value in both modes, matching the
/// interpreter's fpu_load_i64. The 16/32-bit loaders above stay relaxed — they fit in f64
/// exactly.
pub fn gen_fpu_load_i64(ctx: &mut JitContext, modrm_byte: ModrmByte) {
    gen_fpu_load_i64_f80(ctx, modrm_byte);
}

// ── Relaxed FPU inline fast path (BottleShip plan B) ───────────────────────
// Tag-check ST entries; fall back to fpu_* helpers on non-RELAXED_TAG values.
// Skips stack-fault / status-word accumulation (same as relaxed softfloat).

const FPU_RELAXED_TAG: i32 = 0x7FFE;
const FPU_C0: i32 = 0x100;
const FPU_C1: i32 = 0x200;
const FPU_C2: i32 = 0x400;
const FPU_C3: i32 = 0x4000;
// includes C1: x87 FCOM clears C0/C1/C2/C3 (see cpu/fpu.rs FPU_RESULT_FLAGS)
const FPU_RESULT_FLAGS: i32 = FPU_C0 | FPU_C1 | FPU_C2 | FPU_C3;
const FPU_EX_I: i32 = 1 << 0;
const FPU_EX_SF: i32 = 1 << 6;
const F80_INDEFINITE_NAN_MANTISSA: i64 = 0xC000000000000000u64 as i64;
const F80_INDEFINITE_NAN_SIGN_EXPONENT: i32 = 0x7FFF;

#[derive(Copy, Clone)]
pub enum FpuFastBinOp {
    Add,
    Mul,
    Sub,
    SubR,
    Div,
    DivR,
}

struct FpuBitsLocal {
    local: WasmLocalI64,
}

fn x87_local_cache_enabled() -> bool {
    crate::jit::x87_locals_enabled() && crate::softfloat::is_fpu_relaxed()
}

fn gen_x87_local_slot(ctx: &mut JitContext, i: u32) -> Option<(WasmLocalI64, WasmLocal)> {
    if !x87_local_cache_enabled() {
        return None;
    }

    // This wrapper keeps the st-cache coherent for this instruction.
    ctx.x87_cache_kept = true;

    let idx = i as usize;
    if ctx.x87_local_cache[idx].is_none() {
        ctx.builder.const_i64(0);
        let bits = ctx.builder.set_new_local_i64();
        ctx.builder.const_i32(0);
        let valid = ctx.builder.set_new_local();
        ctx.x87_local_cache[idx] = Some(crate::jit::X87LocalCacheSlot { bits, valid });
    }

    let slot = ctx.x87_local_cache[idx].as_ref().unwrap();
    Some((slot.bits.unsafe_clone(), slot.valid.unsafe_clone()))
}

pub fn gen_x87_local_cache_invalidate_all_runtime(ctx: &mut JitContext) {
    // Invalidate at runtime; keep locals allocated for later refill.
    ctx.x87_cache_kept = true;
    let valids: Vec<WasmLocal> = ctx
        .x87_local_cache
        .iter()
        .filter_map(|slot| slot.as_ref().map(|slot| slot.valid.unsafe_clone()))
        .collect();
    if valids.is_empty() {
        return;
    }

    crate::jit::x87_locals_note_cache_invalidate_compiled();
    gen_dispatch_stat_increment(ctx.builder, profiler::stat::X87_CACHE_INVALIDATE);
    for valid in valids {
        ctx.builder.const_i32(0);
        ctx.builder.set_local(&valid);
    }
}

pub fn gen_x87_local_cache_free_all(ctx: &mut JitContext) {
    for slot in ctx.x87_local_cache.iter_mut() {
        if let Some(slot) = slot.take() {
            ctx.builder.free_local_i64(slot.bits);
            ctx.builder.free_local(slot.valid);
        }
    }
}

fn gen_fpu_relaxed_st_ok(ctx: &mut JitContext, i: u32, addr: &WasmLocal) {
    if let Some((_bits, valid)) = gen_x87_local_slot(ctx, i) {
        ctx.builder.get_local(&valid);
        // UNLIKELY, not LIKELY: the slot cache is invalidated almost as often as it is read
        // (x87LocalStats invalidatePerUse), so the hit side is the minority. An inverted hint is
        // worse than none — it makes the optimizing tier lay out the cold side as fallthrough.
        // Sole member of HINT_GROUP_X87 (bit1), which is not in the shipped mask.
        ctx.builder.if_i32_hinted(HINT_GROUP_X87, HINT_UNLIKELY);
        ctx.builder.const_i32(1);
        ctx.builder.else_();
        gen_fpu_relaxed_tag_ok(ctx, addr);
        ctx.builder.block_end();
    }
    else {
        gen_fpu_relaxed_tag_ok(ctx, addr);
    }
}

fn gen_fpu_load_relaxed_st_bits(
    ctx: &mut JitContext,
    i: u32,
    addr: &WasmLocal,
) -> FpuBitsLocal {
    if let Some((bits_cache, valid)) = gen_x87_local_slot(ctx, i) {
        crate::jit::x87_locals_note_cache_load_site_compiled();
        ctx.builder.get_local(&valid);
        ctx.builder.if_i64();
        gen_dispatch_stat_increment(ctx.builder, profiler::stat::X87_CACHE_HIT);
        ctx.builder.get_local_i64(&bits_cache);
        ctx.builder.else_();
        gen_dispatch_stat_increment(ctx.builder, profiler::stat::X87_CACHE_FILL);
        ctx.builder.const_i32(1);
        ctx.builder.set_local(&valid);
        ctx.builder.get_local(addr);
        ctx.builder.load_unaligned_i64(0);
        ctx.builder.tee_local_i64(&bits_cache);
        ctx.builder.block_end();
        FpuBitsLocal { local: ctx.builder.set_new_local_i64() }
    }
    else {
        FpuBitsLocal {
            local: gen_fpu_load_relaxed_f64_bits(ctx, addr),
        }
    }
}

fn gen_free_fpu_bits(ctx: &mut JitContext, bits: FpuBitsLocal) {
    ctx.builder.free_local_i64(bits.local);
}

fn gen_fpu_load_relaxed_st_f64(ctx: &mut JitContext, i: u32, addr: &WasmLocal) {
    let bits = gen_fpu_load_relaxed_st_bits(ctx, i, addr);
    ctx.builder.get_local_i64(&bits.local);
    ctx.builder.reinterpret_i64_as_f64();
    gen_free_fpu_bits(ctx, bits);
}

fn gen_fpu_store_relaxed_f64_st(ctx: &mut JitContext, i: u32, addr: &WasmLocal) {
    gen_mark_fpu_simd_dirty_once(ctx);
    ctx.builder.reinterpret_f64_as_i64();
    let bits = ctx.builder.set_new_local_i64();
    ctx.builder.get_local(addr);
    ctx.builder.get_local_i64(&bits);
    ctx.builder.store_unaligned_i64(0);
    ctx.builder.get_local(addr);
    ctx.builder.const_i32(FPU_RELAXED_TAG);
    ctx.builder.store_unaligned_u16(8);

    if let Some((bits_cache, valid)) = gen_x87_local_slot(ctx, i) {
        crate::jit::x87_locals_note_cache_store_compiled();
        ctx.builder.get_local_i64(&bits);
        ctx.builder.set_local_i64(&bits_cache);
        ctx.builder.const_i32(1);
        ctx.builder.set_local(&valid);
    }

    ctx.builder.free_local_i64(bits);
}

fn gen_fpu_st_addr(ctx: &mut JitContext, i: u32) -> WasmLocal {
    ctx.builder
        .load_fixed_u8(global_pointers::fpu_stack_ptr as u32);
    ctx.builder.const_i32(i as i32);
    ctx.builder.add_i32();
    ctx.builder.const_i32(7);
    ctx.builder.and_i32();
    ctx.builder.const_i32(16);
    ctx.builder.mul_i32();
    ctx.builder.const_i32(global_pointers::fpu_st as i32);
    ctx.builder.add_i32();
    ctx.builder.set_new_local()
}

fn gen_fpu_relaxed_tag_ok(ctx: &mut JitContext, addr: &WasmLocal) {
    ctx.builder.get_local(addr);
    ctx.builder.load_unaligned_u16(8);
    ctx.builder.const_i32(FPU_RELAXED_TAG);
    ctx.builder.eq_i32();
}

fn gen_fpu_relaxed_stat_increment(builder: &mut WasmBuilder, stat: profiler::stat) {
    let addr = unsafe { &raw mut profiler::stat_array[stat as usize] } as u32;
    builder.increment_fixed_i64(addr, 1);
}

// Emit an always-on dispatch-characterisation counter increment into the
// compiled block (gated at compile time by jit::DISPATCH_STATS so it's free unless measuring).
// See profiler::stat. Call sites pass MODULE_EXIT_* / BLOCK_EXECUTION.
pub fn gen_dispatch_stat_increment(builder: &mut WasmBuilder, stat: profiler::stat) {
    if !crate::jit::dispatch_stats_enabled() {
        return;
    }
    let addr = unsafe { &raw mut profiler::stat_array[stat as usize] } as u32;
    builder.increment_fixed_i64(addr, 1);
}

fn gen_fpu_relaxed_record_hit(ctx: &mut JitContext) {
    if !crate::softfloat::is_fpu_relaxed_stats() {
        return;
    }
    gen_fpu_relaxed_stat_increment(ctx.builder, profiler::stat::FPU_RELAXED_HIT);
}

fn gen_fpu_relaxed_record_fallback(ctx: &mut JitContext) {
    if !crate::softfloat::is_fpu_relaxed_stats() {
        return;
    }
    gen_fpu_relaxed_stat_increment(ctx.builder, profiler::stat::FPU_RELAXED_FALLBACK);
}

fn gen_fpu_load_relaxed_f64_bits(ctx: &mut JitContext, addr: &WasmLocal) -> WasmLocalI64 {
    ctx.builder.get_local(addr);
    ctx.builder.load_unaligned_i64(0);
    ctx.builder.set_new_local_i64()
}

fn gen_fpu_copy_raw(ctx: &mut JitContext, src_addr: &WasmLocal, dst_addr: &WasmLocal) {
    gen_mark_fpu_simd_dirty_once(ctx);
    ctx.builder.get_local(src_addr);
    ctx.builder.load_unaligned_i64(0);
    let lo = ctx.builder.set_new_local_i64();
    ctx.builder.get_local(src_addr);
    ctx.builder.load_unaligned_i64(8);
    let hi = ctx.builder.set_new_local_i64();

    ctx.builder.get_local(dst_addr);
    ctx.builder.get_local_i64(&lo);
    ctx.builder.store_unaligned_i64(0);
    ctx.builder.get_local(dst_addr);
    ctx.builder.get_local_i64(&hi);
    ctx.builder.store_unaligned_i64(8);

    ctx.builder.free_local_i64(hi);
    ctx.builder.free_local_i64(lo);
}

fn gen_fpu_relaxed_pop_n(ctx: &mut JitContext, count: u32) {
    for _ in 0..count {
        gen_fpu_relaxed_pop(ctx);
    }
}

fn gen_fpu_load_m32_as_f64_bits(ctx: &mut JitContext, modrm_byte: ModrmByte) -> WasmLocalI64 {
    gen_fpu_load_m32_as_f64(ctx, modrm_byte);
    ctx.builder.reinterpret_f64_as_i64();
    ctx.builder.set_new_local_i64()
}

fn gen_fpu_load_m64_as_f64_bits(ctx: &mut JitContext, modrm_byte: ModrmByte) -> WasmLocalI64 {
    gen_fpu_load_m64_as_f64(ctx, modrm_byte);
    ctx.builder.reinterpret_f64_as_i64();
    ctx.builder.set_new_local_i64()
}

fn gen_fpu_load_i16_as_f64(ctx: &mut JitContext, modrm_byte: ModrmByte) {
    gen_modrm_resolve_safe_read16(ctx, modrm_byte);
    sign_extend_i16(ctx.builder);
    ctx.builder.convert_i32_to_f64();
}

fn gen_fpu_load_i32_as_f64(ctx: &mut JitContext, modrm_byte: ModrmByte) {
    gen_modrm_resolve_safe_read32(ctx, modrm_byte);
    ctx.builder.convert_i32_to_f64();
}

fn gen_fpu_load_i16_as_f64_bits(ctx: &mut JitContext, modrm_byte: ModrmByte) -> WasmLocalI64 {
    gen_fpu_load_i16_as_f64(ctx, modrm_byte);
    ctx.builder.reinterpret_f64_as_i64();
    ctx.builder.set_new_local_i64()
}

fn gen_fpu_load_i32_as_f64_bits(ctx: &mut JitContext, modrm_byte: ModrmByte) -> WasmLocalI64 {
    gen_fpu_load_i32_as_f64(ctx, modrm_byte);
    ctx.builder.reinterpret_f64_as_i64();
    ctx.builder.set_new_local_i64()
}

fn gen_f64_is_nan(ctx: &mut JitContext, bits: &WasmLocalI64) {
    ctx.builder.get_local_i64(bits);
    ctx.builder.reinterpret_i64_as_f64();
    ctx.builder.get_local_i64(bits);
    ctx.builder.reinterpret_i64_as_f64();
    ctx.builder.ne_f64();
}

fn gen_fpu_compare_flags(
    ctx: &mut JitContext,
    x_bits: &WasmLocalI64,
    y_bits: &WasmLocalI64,
    unordered_flags: i32,
    less_flags: i32,
    equal_flags: i32,
) -> WasmLocal {
    gen_f64_is_nan(ctx, x_bits);
    gen_f64_is_nan(ctx, y_bits);
    ctx.builder.or_i32();
    ctx.builder.if_i32();
    ctx.builder.const_i32(unordered_flags);
    ctx.builder.else_();
    ctx.builder.get_local_i64(x_bits);
    ctx.builder.reinterpret_i64_as_f64();
    ctx.builder.get_local_i64(y_bits);
    ctx.builder.reinterpret_i64_as_f64();
    ctx.builder.lt_f64();
    ctx.builder.if_i32();
    ctx.builder.const_i32(less_flags);
    ctx.builder.else_();
    ctx.builder.get_local_i64(x_bits);
    ctx.builder.reinterpret_i64_as_f64();
    ctx.builder.get_local_i64(y_bits);
    ctx.builder.reinterpret_i64_as_f64();
    ctx.builder.eq_f64();
    ctx.builder.if_i32();
    ctx.builder.const_i32(equal_flags);
    ctx.builder.else_();
    ctx.builder.const_i32(0);
    ctx.builder.block_end();
    ctx.builder.block_end();
    ctx.builder.block_end();
    ctx.builder.set_new_local()
}

fn gen_fpu_write_status_compare_flags(ctx: &mut JitContext, compare_flags: &WasmLocal) {
    gen_mark_fpu_simd_dirty_once(ctx);
    ctx.builder
        .const_i32(global_pointers::fpu_status_word as i32);
    ctx.builder
        .load_fixed_u16(global_pointers::fpu_status_word as u32);
    ctx.builder.const_i32(!FPU_RESULT_FLAGS);
    ctx.builder.and_i32();
    ctx.builder.get_local(compare_flags);
    ctx.builder.or_i32();
    ctx.builder.store_aligned_u16(0);
}

fn gen_fpu_write_eflags_compare_flags(ctx: &mut JitContext, compare_flags: &WasmLocal) {
    ctx.builder.const_i32(global_pointers::flags_changed as i32);
    ctx.builder.const_i32(0);
    ctx.builder.flag_store_i32(FLAG_LOCAL_FLAGS_CHANGED);

    ctx.builder.const_i32(global_pointers::flags as i32);
    ctx.builder
        .flag_load_i32(FLAG_LOCAL_FLAGS, global_pointers::flags as u32);
    ctx.builder.const_i32(!FLAGS_ALL);
    ctx.builder.and_i32();
    ctx.builder.get_local(compare_flags);
    ctx.builder.or_i32();
    ctx.builder.flag_store_i32(FLAG_LOCAL_FLAGS);
}

fn gen_fpu_compare_status_from_bits(
    ctx: &mut JitContext,
    x_bits: &WasmLocalI64,
    y_bits: &WasmLocalI64,
) {
    let flags = gen_fpu_compare_flags(ctx, x_bits, y_bits, FPU_C0 | FPU_C2 | FPU_C3, FPU_C0, FPU_C3);
    gen_fpu_write_status_compare_flags(ctx, &flags);
    ctx.builder.free_local(flags);
}

fn gen_fpu_compare_eflags_from_bits(
    ctx: &mut JitContext,
    x_bits: &WasmLocalI64,
    y_bits: &WasmLocalI64,
) {
    let flags = gen_fpu_compare_flags(
        ctx,
        x_bits,
        y_bits,
        FLAG_CARRY | FLAG_PARITY | FLAG_ZERO,
        FLAG_CARRY,
        FLAG_ZERO,
    );
    gen_fpu_write_eflags_compare_flags(ctx, &flags);
    ctx.builder.free_local(flags);
}

fn gen_fpu_round_f64_bits_to_i32(ctx: &mut JitContext, bits: &WasmLocalI64, truncate: bool) {
    if truncate {
        ctx.builder.get_local_i64(bits);
        ctx.builder.reinterpret_i64_as_f64();
        ctx.builder.trunc_f64();
        ctx.builder.reinterpret_f64_as_i64();
    }
    else {
        ctx.builder.load_fixed_u16(global_pointers::fpu_control_word as u32);
        ctx.builder.const_i32(10);
        ctx.builder.shr_u_i32();
        ctx.builder.const_i32(3);
        ctx.builder.and_i32();
        let rc = ctx.builder.set_new_local();

        ctx.builder.get_local(&rc);
        ctx.builder.const_i32(1);
        ctx.builder.eq_i32();
        ctx.builder.if_i64();
        ctx.builder.get_local_i64(bits);
        ctx.builder.reinterpret_i64_as_f64();
        ctx.builder.floor_f64();
        ctx.builder.reinterpret_f64_as_i64();
        ctx.builder.else_();
        ctx.builder.get_local(&rc);
        ctx.builder.const_i32(2);
        ctx.builder.eq_i32();
        ctx.builder.if_i64();
        ctx.builder.get_local_i64(bits);
        ctx.builder.reinterpret_i64_as_f64();
        ctx.builder.ceil_f64();
        ctx.builder.reinterpret_f64_as_i64();
        ctx.builder.else_();
        ctx.builder.get_local(&rc);
        ctx.builder.const_i32(3);
        ctx.builder.eq_i32();
        ctx.builder.if_i64();
        ctx.builder.get_local_i64(bits);
        ctx.builder.reinterpret_i64_as_f64();
        ctx.builder.trunc_f64();
        ctx.builder.reinterpret_f64_as_i64();
        ctx.builder.else_();
        ctx.builder.get_local_i64(bits);
        ctx.builder.reinterpret_i64_as_f64();
        ctx.builder.nearest_f64();
        ctx.builder.reinterpret_f64_as_i64();
        ctx.builder.block_end();
        ctx.builder.block_end();
        ctx.builder.block_end();

        ctx.builder.free_local(rc);
    }

    let rounded_bits = ctx.builder.set_new_local_i64();
    gen_f64_is_nan(ctx, &rounded_bits);
    ctx.builder.get_local_i64(&rounded_bits);
    ctx.builder.reinterpret_i64_as_f64();
    ctx.builder.const_f64(2147483648.0);
    ctx.builder.ge_f64();
    ctx.builder.or_i32();
    ctx.builder.get_local_i64(&rounded_bits);
    ctx.builder.reinterpret_i64_as_f64();
    ctx.builder.const_f64(-2147483648.0);
    ctx.builder.lt_f64();
    ctx.builder.or_i32();
    ctx.builder.if_i32();
    ctx.builder.const_i32(i32::MIN);
    ctx.builder.else_();
    ctx.builder.get_local_i64(&rounded_bits);
    ctx.builder.reinterpret_i64_as_f64();
    ctx.builder.trunc_s_f64_to_i32();
    ctx.builder.block_end();
    ctx.builder.free_local_i64(rounded_bits);
}

fn gen_fpu_clamp_i32_to_i16(ctx: &mut JitContext, value: &WasmLocal) {
    ctx.builder.get_local(value);
    ctx.builder.const_i32(-0x8000);
    ctx.builder.lt_i32();
    ctx.builder.get_local(value);
    ctx.builder.const_i32(0x7FFF);
    ctx.builder.gt_i32();
    ctx.builder.or_i32();
    ctx.builder.if_i32();
    ctx.builder.const_i32(-0x8000);
    ctx.builder.else_();
    ctx.builder.get_local(value);
    ctx.builder.block_end();
}

/// x87 PC=00 rounds every arithmetic result to a 24-bit significand. PC is a RUNTIME
/// value — the CRT saves and restores the control word around every transcendental —
/// so it cannot be baked into the compiled block or ignored (a title that compares a
/// computed value against one it stored as f32 then never sees them equal; GTA III's
/// CPathFind de-duplicates car path links exactly that way, and overruns the array into
/// its own map-object table).
///
/// The optional local is populated at runtime on every block activation and survives
/// only across consecutive relaxed arithmetic instructions. Any other instruction is
/// a fence, including every control-word writer/restorer. The old PC gate did not cost what the
/// ROUNDING costs — two conversions — it cost the F80 HELPER CALL it fell back to,
/// which is the whole expense the relaxed path exists to avoid.
fn gen_fpu_pc_cache_prepare(ctx: &mut JitContext) {
    if !crate::jit::x87_pc_local_enabled() || ctx.fpu_pc_cache.is_some() {
        return;
    }

    // This initialization must dominate the relaxed tag fast/slow branch. A run may
    // execute the helper branch for its first instruction (for example an F80 ST(i))
    // and the inline branch for the next one. Initializing inside the first inline
    // branch leaves the reused Wasm local stale on that runtime path.
    ctx.builder
        .load_fixed_u16(global_pointers::fpu_control_word as u32);
    ctx.builder.const_i32(0x300);
    ctx.builder.and_i32();
    ctx.builder.eqz_i32();
    ctx.fpu_pc_cache = Some(ctx.builder.set_new_local());
}

fn gen_fpu_round_f64_by_precision_control(ctx: &mut JitContext) {
    ctx.builder.reinterpret_f64_as_i64();
    let raw = ctx.builder.tee_new_local_i64();
    ctx.builder.reinterpret_i64_as_f64();
    ctx.builder.demote_f64_to_f32();
    ctx.builder.promote_f32_to_f64();
    ctx.builder.get_local_i64(&raw);
    ctx.builder.reinterpret_i64_as_f64();
    // select() keeps the FIRST value when the condition is non-zero: PC field == 00.
    if crate::jit::x87_pc_local_enabled() {
        ctx.fpu_pc_cache_kept = true;
        // Every relaxed binop prepares this before emitting its tag branch, so all
        // runtime paths reaching a lexical run observe an initialized predicate.
        ctx.builder.get_local(ctx.fpu_pc_cache.as_ref().unwrap());
    }
    else {
        ctx.builder
            .load_fixed_u16(global_pointers::fpu_control_word as u32);
        ctx.builder.const_i32(0x300);
        ctx.builder.and_i32();
        ctx.builder.eqz_i32();
    }
    ctx.builder.select();
    ctx.builder.free_local_i64(raw);
}

pub fn gen_fpu_pc_cache_free(ctx: &mut JitContext) {
    if let Some(pc) = ctx.fpu_pc_cache.take() {
        ctx.builder.free_local(pc);
    }
}

fn gen_fpu_apply_f64_binop(ctx: &mut JitContext, op: FpuFastBinOp) {
    match op {
        FpuFastBinOp::Add => ctx.builder.add_f64(),
        FpuFastBinOp::Mul => ctx.builder.mul_f64(),
        FpuFastBinOp::Sub => ctx.builder.sub_f64(),
        FpuFastBinOp::SubR => ctx.builder.sub_f64(),
        FpuFastBinOp::Div => ctx.builder.div_f64(),
        FpuFastBinOp::DivR => ctx.builder.div_f64(),
    }
    // The inline path and the F80 fallback are two implementations of one instruction:
    // whatever softfloat's apply_precision does to the helper result must happen here too.
    gen_fpu_round_f64_by_precision_control(ctx);
}

fn gen_fpu_load_m32_as_f64(ctx: &mut JitContext, modrm_byte: ModrmByte) {
    gen_modrm_resolve_safe_read32(ctx, modrm_byte);
    ctx.builder.reinterpret_i32_as_f32();
    ctx.builder.promote_f32_to_f64();
}

fn gen_fpu_load_m64_as_f64(ctx: &mut JitContext, modrm_byte: ModrmByte) {
    gen_modrm_resolve_safe_read64(ctx, modrm_byte);
    ctx.builder.reinterpret_i64_as_f64();
}

/// Memory operand type for the relaxed-FPU fast paths. The m32/i32/i16/m64 binop and
/// fcom families are identical apart from which typed load helper they call; this
/// selects it. (m32/m64 float and i16/i32 int operands all convert to f64 exactly, so
/// the only semantic difference vs the F80 path is the binop/compare rounding.)
#[derive(Clone, Copy)]
enum FpuMemSrc { M32, I32, I16, M64 }

// F80-producing slow load for the relaxed BINOP slow path (feeds call_fn3_i32_i64_i32).
fn gen_fpu_load_mem_binop_slow(ctx: &mut JitContext, src: FpuMemSrc, modrm_byte: ModrmByte) {
    match src {
        FpuMemSrc::M32 => gen_fpu_load_m32(ctx, modrm_byte),
        FpuMemSrc::I32 => gen_fpu_load_i32_f80(ctx, modrm_byte),
        FpuMemSrc::I16 => gen_fpu_load_i16_f80(ctx, modrm_byte),
        FpuMemSrc::M64 => gen_fpu_load_m64(ctx, modrm_byte),
    }
}

// Plain slow load for the relaxed FCOM slow path (feeds call_fn2_i64_i32).
fn gen_fpu_load_mem_fcom_slow(ctx: &mut JitContext, src: FpuMemSrc, modrm_byte: ModrmByte) {
    match src {
        FpuMemSrc::M32 => gen_fpu_load_m32(ctx, modrm_byte),
        FpuMemSrc::I32 => gen_fpu_load_i32(ctx, modrm_byte),
        FpuMemSrc::I16 => gen_fpu_load_i16(ctx, modrm_byte),
        FpuMemSrc::M64 => gen_fpu_load_m64(ctx, modrm_byte),
    }
}

// Exact-f64 fast load of the memory operand (pushed on the wasm stack).
fn gen_fpu_load_mem_as_f64(ctx: &mut JitContext, src: FpuMemSrc, modrm_byte: ModrmByte) {
    match src {
        FpuMemSrc::M32 => gen_fpu_load_m32_as_f64(ctx, modrm_byte),
        FpuMemSrc::I32 => gen_fpu_load_i32_as_f64(ctx, modrm_byte),
        FpuMemSrc::I16 => gen_fpu_load_i16_as_f64(ctx, modrm_byte),
        FpuMemSrc::M64 => gen_fpu_load_m64_as_f64(ctx, modrm_byte),
    }
}

// Exact-f64 fast load of the memory operand, returned as an i64-bits local.
fn gen_fpu_load_mem_as_f64_bits(ctx: &mut JitContext, src: FpuMemSrc, modrm_byte: ModrmByte) -> WasmLocalI64 {
    match src {
        FpuMemSrc::M32 => gen_fpu_load_m32_as_f64_bits(ctx, modrm_byte),
        FpuMemSrc::I32 => gen_fpu_load_i32_as_f64_bits(ctx, modrm_byte),
        FpuMemSrc::I16 => gen_fpu_load_i16_as_f64_bits(ctx, modrm_byte),
        FpuMemSrc::M64 => gen_fpu_load_m64_as_f64_bits(ctx, modrm_byte),
    }
}

fn gen_fpu_relaxed_load_binop_operands_mem(
    ctx: &mut JitContext,
    op: FpuFastBinOp,
    st0_addr: &WasmLocal,
    src: FpuMemSrc,
    modrm_byte: ModrmByte,
) {
    let reversed = matches!(op, FpuFastBinOp::SubR | FpuFastBinOp::DivR);
    if reversed {
        gen_fpu_load_mem_as_f64(ctx, src, modrm_byte);
        gen_fpu_load_relaxed_st_f64(ctx, 0, st0_addr);
    }
    else {
        gen_fpu_load_relaxed_st_f64(ctx, 0, st0_addr);
        gen_fpu_load_mem_as_f64(ctx, src, modrm_byte);
    }
}

fn gen_fpu_relaxed_load_binop_operands_sti(
    ctx: &mut JitContext,
    op: FpuFastBinOp,
    st0_addr: &WasmLocal,
    op_sti: u32,
    op_addr: &WasmLocal,
) {
    let reversed = matches!(op, FpuFastBinOp::SubR | FpuFastBinOp::DivR);
    if reversed {
        gen_fpu_load_relaxed_st_f64(ctx, op_sti, op_addr);
        gen_fpu_load_relaxed_st_f64(ctx, 0, st0_addr);
    }
    else {
        gen_fpu_load_relaxed_st_f64(ctx, 0, st0_addr);
        gen_fpu_load_relaxed_st_f64(ctx, op_sti, op_addr);
    }
}

// Relaxed-FPU binop with a memory operand (fadd/fmul/fsub/fsubr/fdiv/fdivr and their
// integer forms fiadd/fimul/… DA /r m32int, DE /r m16int). The int operand converts to
// f64 EXACTLY (both fit), so the only semantic difference vs the F80 helper is the binop
// rounding — identical to the float-operand relaxed path. Without the int fast paths,
// integer-operand loops (int PCM × float gain in audio converters) would hit the F80
// helper on every sample even in relaxed mode.
fn gen_fpu_relaxed_binop_mem(
    ctx: &mut JitContext,
    modrm_byte: ModrmByte,
    target_sti: u32,
    op: FpuFastBinOp,
    slow_helper: &str,
    src: FpuMemSrc,
) {
    if !crate::softfloat::is_fpu_relaxed() {
        ctx.builder.const_i32(target_sti as i32);
        gen_fpu_load_mem_binop_slow(ctx, src, modrm_byte);
        ctx.builder.call_fn3_i32_i64_i32(slow_helper);
        return;
    }
    gen_fpu_pc_cache_prepare(ctx);
    let modrm_slow = modrm_byte.clone();
    let target_addr = gen_fpu_st_addr(ctx, target_sti);
    // D8/DA/DE memory arithmetic overwhelmingly writes ST0. Target and source are
    // then the same physical stack slot; compute TOP->address once.
    let st0_aliases_target = target_sti == 0;
    let st0_addr = if st0_aliases_target {
        target_addr.unsafe_clone()
    }
    else {
        gen_fpu_st_addr(ctx, 0)
    };
    gen_fpu_relaxed_st_ok(ctx, 0, &st0_addr);
    ctx.builder.eqz_i32();
    ctx.builder.if_void();
    gen_fpu_relaxed_record_fallback(ctx);
    ctx.builder.const_i32(target_sti as i32);
    gen_fpu_load_mem_binop_slow(ctx, src, modrm_slow);
    ctx.builder.call_fn3_i32_i64_i32(slow_helper);
    gen_x87_local_cache_invalidate_all_runtime(ctx);
    ctx.builder.else_();
    gen_fpu_relaxed_record_hit(ctx);
    gen_fpu_relaxed_load_binop_operands_mem(ctx, op, &st0_addr, src, modrm_byte);
    gen_fpu_apply_f64_binop(ctx, op);
    gen_fpu_store_relaxed_f64_st(ctx, target_sti, &target_addr);
    ctx.builder.block_end();
    if !st0_aliases_target {
        ctx.builder.free_local(st0_addr);
    }
    ctx.builder.free_local(target_addr);
}

pub fn gen_fpu_relaxed_binop_m32(ctx: &mut JitContext, modrm_byte: ModrmByte, target_sti: u32, op: FpuFastBinOp, slow_helper: &str) {
    gen_fpu_relaxed_binop_mem(ctx, modrm_byte, target_sti, op, slow_helper, FpuMemSrc::M32)
}
pub fn gen_fpu_relaxed_binop_i32(ctx: &mut JitContext, modrm_byte: ModrmByte, target_sti: u32, op: FpuFastBinOp, slow_helper: &str) {
    gen_fpu_relaxed_binop_mem(ctx, modrm_byte, target_sti, op, slow_helper, FpuMemSrc::I32)
}
pub fn gen_fpu_relaxed_binop_i16(ctx: &mut JitContext, modrm_byte: ModrmByte, target_sti: u32, op: FpuFastBinOp, slow_helper: &str) {
    gen_fpu_relaxed_binop_mem(ctx, modrm_byte, target_sti, op, slow_helper, FpuMemSrc::I16)
}
pub fn gen_fpu_relaxed_binop_m64(ctx: &mut JitContext, modrm_byte: ModrmByte, target_sti: u32, op: FpuFastBinOp, slow_helper: &str) {
    gen_fpu_relaxed_binop_mem(ctx, modrm_byte, target_sti, op, slow_helper, FpuMemSrc::M64)
}

pub fn gen_fpu_relaxed_binop_sti(
    ctx: &mut JitContext,
    sti: u32,
    target_sti: u32,
    op: FpuFastBinOp,
    slow_helper: &str,
) {
    if !crate::softfloat::is_fpu_relaxed() {
        ctx.builder.const_i32(target_sti as i32);
        gen_fpu_get_sti(ctx, sti);
        ctx.builder.call_fn3_i32_i64_i32(slow_helper);
        return;
    }
    gen_fpu_pc_cache_prepare(ctx);
    let target_addr = gen_fpu_st_addr(ctx, target_sti);
    let st0_aliases_target = target_sti == 0;
    let st0_addr = if st0_aliases_target {
        target_addr.unsafe_clone()
    }
    else {
        gen_fpu_st_addr(ctx, 0)
    };
    let op_aliases_target = sti == target_sti;
    let op_aliases_st0 = !op_aliases_target && sti == 0;
    let op_addr = if op_aliases_target {
        target_addr.unsafe_clone()
    }
    else if op_aliases_st0 {
        st0_addr.unsafe_clone()
    }
    else {
        gen_fpu_st_addr(ctx, sti)
    };
    gen_fpu_relaxed_st_ok(ctx, 0, &st0_addr);
    gen_fpu_relaxed_st_ok(ctx, sti, &op_addr);
    ctx.builder.and_i32();
    ctx.builder.eqz_i32();
    ctx.builder.if_void();
    gen_fpu_relaxed_record_fallback(ctx);
    ctx.builder.const_i32(target_sti as i32);
    gen_fpu_get_sti(ctx, sti);
    ctx.builder.call_fn3_i32_i64_i32(slow_helper);
    gen_x87_local_cache_invalidate_all_runtime(ctx);
    ctx.builder.else_();
    gen_fpu_relaxed_record_hit(ctx);
    gen_fpu_relaxed_load_binop_operands_sti(ctx, op, &st0_addr, sti, &op_addr);
    gen_fpu_apply_f64_binop(ctx, op);
    gen_fpu_store_relaxed_f64_st(ctx, target_sti, &target_addr);
    ctx.builder.block_end();
    if !op_aliases_target && !op_aliases_st0 {
        ctx.builder.free_local(op_addr);
    }
    if !st0_aliases_target {
        ctx.builder.free_local(st0_addr);
    }
    ctx.builder.free_local(target_addr);
}

pub fn gen_fpu_relaxed_pop(ctx: &mut JitContext) {
    gen_mark_fpu_simd_dirty_once(ctx);
    if !crate::softfloat::is_fpu_relaxed() {
        ctx.builder.call_fn0("fpu_pop");
        return;
    }
    gen_x87_local_cache_invalidate_all_runtime(ctx);
    ctx.builder.load_fixed_u8(global_pointers::fpu_stack_ptr as u32);
    let ptr_local = ctx.builder.set_new_local();
    ctx.builder.const_i32(1);
    ctx.builder.get_local(&ptr_local);
    ctx.builder.shl_i32();
    ctx.builder.load_fixed_u8(global_pointers::fpu_stack_empty as u32);
    ctx.builder.or_i32();
    let new_empty = ctx.builder.set_new_local();
    ctx.builder.const_i32(global_pointers::fpu_stack_empty as i32);
    ctx.builder.get_local(&new_empty);
    ctx.builder.store_u8(0);

    ctx.builder.load_fixed_u8(global_pointers::fpu_stack_ptr as u32);
    ctx.builder.const_i32(1);
    ctx.builder.add_i32();
    ctx.builder.const_i32(7);
    ctx.builder.and_i32();
    let new_ptr = ctx.builder.set_new_local();
    ctx.builder.const_i32(global_pointers::fpu_stack_ptr as i32);
    ctx.builder.get_local(&new_ptr);
    ctx.builder.store_u8(0);
    ctx.builder.free_local(ptr_local);
    ctx.builder.free_local(new_ptr);
    ctx.builder.free_local(new_empty);
}

pub fn gen_fpu_relaxed_push_loaded(ctx: &mut JitContext) {
    gen_mark_fpu_simd_dirty_once(ctx);
    // stack: i64 mantissa, i32 tag
    if !crate::softfloat::is_fpu_relaxed() {
        ctx.builder.call_fn2_i64_i32("fpu_push");
        return;
    }
    let tag = ctx.builder.set_new_local();
    let mantissa = ctx.builder.set_new_local_i64();
    gen_x87_local_cache_invalidate_all_runtime(ctx);

    ctx.builder.load_fixed_u8(global_pointers::fpu_stack_ptr as u32);
    ctx.builder.const_i32(1);
    ctx.builder.sub_i32();
    ctx.builder.const_i32(7);
    ctx.builder.and_i32();
    let new_ptr = ctx.builder.set_new_local();

    // Pushing onto an OCCUPIED slot is a stack overflow (C1 + SF|IE, INDEFINITE_NAN, the
    // live register kept) — rare, so take the helper. Nothing is mutated on this path yet:
    // fpu_push does its own decrement from the unmodified fpu_stack_ptr.
    ctx.builder
        .load_fixed_u8(global_pointers::fpu_stack_empty as u32);
    ctx.builder.get_local(&new_ptr);
    ctx.builder.shr_u_i32();
    ctx.builder.const_i32(1);
    ctx.builder.and_i32();
    ctx.builder.eqz_i32();
    ctx.builder.if_void();
    ctx.builder.get_local_i64(&mantissa);
    ctx.builder.get_local(&tag);
    ctx.builder.call_fn2_i64_i32("fpu_push");
    ctx.builder.else_();

    ctx.builder.const_i32(global_pointers::fpu_stack_ptr as i32);
    ctx.builder.get_local(&new_ptr);
    ctx.builder.store_u8(0);

    ctx.builder.const_i32(1);
    ctx.builder.get_local(&new_ptr);
    ctx.builder.shl_i32();
    ctx.builder.const_i32(-1);
    ctx.builder.xor_i32();
    ctx.builder.load_fixed_u8(global_pointers::fpu_stack_empty as u32);
    ctx.builder.and_i32();
    let new_empty = ctx.builder.set_new_local();
    ctx.builder.const_i32(global_pointers::fpu_stack_empty as i32);
    ctx.builder.get_local(&new_empty);
    ctx.builder.store_u8(0);

    // fpu_push clears C1 on a successful push; C1 is a result flag the guest reads back
    // with FNSTSW, not fault accumulation.
    ctx.builder
        .const_i32(global_pointers::fpu_status_word as i32);
    ctx.builder
        .load_fixed_u16(global_pointers::fpu_status_word as u32);
    ctx.builder.const_i32(!FPU_C1);
    ctx.builder.and_i32();
    ctx.builder.store_aligned_u16(0);

    ctx.builder.get_local(&new_ptr);
    ctx.builder.const_i32(16);
    ctx.builder.mul_i32();
    ctx.builder.const_i32(global_pointers::fpu_st as i32);
    ctx.builder.add_i32();
    let st_addr = ctx.builder.set_new_local();

    ctx.builder.get_local(&st_addr);
    ctx.builder.get_local_i64(&mantissa);
    ctx.builder.store_unaligned_i64(0);
    ctx.builder.get_local(&st_addr);
    ctx.builder.get_local(&tag);
    ctx.builder.store_unaligned_u16(8);
    ctx.builder.free_local(st_addr);
    ctx.builder.free_local(new_empty);

    ctx.builder.block_end();

    ctx.builder.free_local(new_ptr);
    ctx.builder.free_local(tag);
    ctx.builder.free_local_i64(mantissa);
}

pub fn gen_fpu_relaxed_fxch(ctx: &mut JitContext, i: u32) {
    gen_mark_fpu_simd_dirty_once(ctx);
    if !crate::softfloat::is_fpu_relaxed() {
        ctx.builder.const_i32(i as i32);
        ctx.builder.call_fn1("fpu_fxch");
        return;
    }
    gen_x87_local_cache_invalidate_all_runtime(ctx);

    let st0_addr = gen_fpu_st_addr(ctx, 0);
    let sti_addr = gen_fpu_st_addr(ctx, i);

    ctx.builder.get_local(&st0_addr);
    ctx.builder.load_unaligned_i64(0);
    let st0_lo = ctx.builder.set_new_local_i64();
    ctx.builder.get_local(&st0_addr);
    ctx.builder.load_unaligned_i64(8);
    let st0_hi = ctx.builder.set_new_local_i64();

    ctx.builder.get_local(&sti_addr);
    ctx.builder.load_unaligned_i64(0);
    let sti_lo = ctx.builder.set_new_local_i64();
    ctx.builder.get_local(&sti_addr);
    ctx.builder.load_unaligned_i64(8);
    let sti_hi = ctx.builder.set_new_local_i64();

    ctx.builder.get_local(&st0_addr);
    ctx.builder.get_local_i64(&sti_lo);
    ctx.builder.store_unaligned_i64(0);
    ctx.builder.get_local(&st0_addr);
    ctx.builder.get_local_i64(&sti_hi);
    ctx.builder.store_unaligned_i64(8);

    ctx.builder.get_local(&sti_addr);
    ctx.builder.get_local_i64(&st0_lo);
    ctx.builder.store_unaligned_i64(0);
    ctx.builder.get_local(&sti_addr);
    ctx.builder.get_local_i64(&st0_hi);
    ctx.builder.store_unaligned_i64(8);

    ctx.builder.free_local_i64(sti_hi);
    ctx.builder.free_local_i64(sti_lo);
    ctx.builder.free_local_i64(st0_hi);
    ctx.builder.free_local_i64(st0_lo);
    ctx.builder.free_local(sti_addr);
    ctx.builder.free_local(st0_addr);
}

// FST/FSTP ST(i) marks the destination non-empty in the tag word (cpu/fpu.rs fpu_fst); the
// raw 16-byte copy alone leaves ST(i) tagged empty. Emitted before the pop, so fpu_stack_ptr
// still selects the source, as in the helper. Callers guarantee a live ST(0).
fn gen_fpu_clear_stack_empty_sti(ctx: &mut JitContext, i: u32) {
    ctx.builder.const_i32(global_pointers::fpu_stack_empty as i32);
    ctx.builder.load_fixed_u8(global_pointers::fpu_stack_empty as u32);
    ctx.builder.const_i32(1);
    ctx.builder.load_fixed_u8(global_pointers::fpu_stack_ptr as u32);
    ctx.builder.const_i32(i as i32);
    ctx.builder.add_i32();
    ctx.builder.const_i32(7);
    ctx.builder.and_i32();
    ctx.builder.shl_i32();
    ctx.builder.const_i32(-1);
    ctx.builder.xor_i32();
    ctx.builder.and_i32();
    ctx.builder.store_u8(0);
}

pub fn gen_fpu_relaxed_fst(ctx: &mut JitContext, i: u32, pop: bool) {
    let helper = if pop { "fpu_fstp" } else { "fpu_fst" };
    if !crate::softfloat::is_fpu_relaxed() {
        gen_fn1_const(ctx.builder, helper, i);
        return;
    }
    gen_mark_fpu_simd_dirty_once(ctx);

    // The helper reads the source through fpu_get_st0, so an EMPTY ST(0) is a stack fault
    // (SF|IE, INDEFINITE_NAN into the destination) — rare, and the raw copy below would
    // instead resurrect the freed slot's stale bytes.
    ctx.builder
        .load_fixed_u8(global_pointers::fpu_stack_empty as u32);
    ctx.builder
        .load_fixed_u8(global_pointers::fpu_stack_ptr as u32);
    ctx.builder.shr_u_i32();
    ctx.builder.const_i32(1);
    ctx.builder.and_i32();
    ctx.builder.if_void();
    gen_fn1_const(ctx.builder, helper, i);
    gen_x87_local_cache_invalidate_all_runtime(ctx);
    ctx.builder.else_();
    let st0_addr = gen_fpu_st_addr(ctx, 0);
    let sti_addr = gen_fpu_st_addr(ctx, i);
    gen_fpu_copy_raw(ctx, &st0_addr, &sti_addr);
    gen_fpu_clear_stack_empty_sti(ctx, i);
    gen_x87_local_cache_invalidate_all_runtime(ctx);
    if pop {
        gen_fpu_relaxed_pop(ctx);
    }
    ctx.builder.free_local(sti_addr);
    ctx.builder.free_local(st0_addr);
    ctx.builder.block_end();
}

pub fn gen_fpu_relaxed_fchs_or_fabs(ctx: &mut JitContext, r: u32) {
    if !crate::softfloat::is_fpu_relaxed() {
        ctx.builder.const_i32(r as i32);
        ctx.builder.call_fn1("instr16_D9_4_reg");
        return;
    }

    let st0_addr = gen_fpu_st_addr(ctx, 0);
    gen_fpu_relaxed_st_ok(ctx, 0, &st0_addr);
    ctx.builder.eqz_i32();
    ctx.builder.if_void();
    gen_fpu_relaxed_record_fallback(ctx);
    ctx.builder.const_i32(r as i32);
    ctx.builder.call_fn1("instr16_D9_4_reg");
    gen_x87_local_cache_invalidate_all_runtime(ctx);
    ctx.builder.else_();
    gen_fpu_relaxed_record_hit(ctx);
    gen_fpu_load_relaxed_st_f64(ctx, 0, &st0_addr);
    if r == 0 {
        ctx.builder.neg_f64();
    }
    else {
        ctx.builder.abs_f64();
    }
    gen_fpu_store_relaxed_f64_st(ctx, 0, &st0_addr);
    ctx.builder.block_end();
    ctx.builder.free_local(st0_addr);
}

pub fn gen_fpu_relaxed_push_const_f64(ctx: &mut JitContext, value: f64, slow_r: u32) {
    if !crate::softfloat::is_fpu_relaxed() {
        gen_fn1_const(ctx.builder, "instr16_D9_5_reg", slow_r);
        return;
    }

    ctx.builder.const_f64(value);
    ctx.builder.reinterpret_f64_as_i64();
    ctx.builder.const_i32(FPU_RELAXED_TAG);
    gen_fpu_relaxed_push_loaded(ctx);
}

pub fn gen_fpu_relaxed_store_m32(ctx: &mut JitContext, modrm_byte: ModrmByte, pop: bool) {
    if !crate::softfloat::is_fpu_relaxed() {
        gen_modrm_resolve(ctx, modrm_byte);
        let address_local = ctx.builder.set_new_local();
        gen_fpu_get_sti(ctx, 0);
        ctx.builder.call_fn2_i64_i32_ret("f80_to_f32");
        let value_local = ctx.builder.set_new_local();
        gen_safe_write32(ctx, &address_local, &value_local);
        ctx.builder.free_local(address_local);
        ctx.builder.free_local(value_local);
        if pop {
            ctx.builder.call_fn0("fpu_pop");
        }
        return;
    }

    gen_modrm_resolve(ctx, modrm_byte);
    let address_local = ctx.builder.set_new_local();
    let st0_addr = gen_fpu_st_addr(ctx, 0);
    gen_fpu_relaxed_st_ok(ctx, 0, &st0_addr);
    ctx.builder.eqz_i32();
    ctx.builder.if_void();
    gen_fpu_relaxed_record_fallback(ctx);
    gen_fpu_get_sti(ctx, 0);
    ctx.builder.call_fn2_i64_i32_ret("f80_to_f32");
    let slow_value = ctx.builder.set_new_local();
    gen_safe_write32(ctx, &address_local, &slow_value);
    ctx.builder.free_local(slow_value);
    gen_x87_local_cache_invalidate_all_runtime(ctx);
    ctx.builder.else_();
    gen_fpu_relaxed_record_hit(ctx);
    gen_fpu_load_relaxed_st_f64(ctx, 0, &st0_addr);
    ctx.builder.demote_f64_to_f32();
    ctx.builder.reinterpret_f32_as_i32();
    let fast_value = ctx.builder.set_new_local();
    gen_safe_write32(ctx, &address_local, &fast_value);
    ctx.builder.free_local(fast_value);
    ctx.builder.block_end();
    if pop {
        gen_fpu_relaxed_pop(ctx);
    }
    ctx.builder.free_local(st0_addr);
    ctx.builder.free_local(address_local);
}

pub fn gen_fpu_relaxed_store_m64(ctx: &mut JitContext, modrm_byte: ModrmByte, pop: bool) {
    if !crate::softfloat::is_fpu_relaxed() {
        gen_modrm_resolve(ctx, modrm_byte);
        let address_local = ctx.builder.set_new_local();
        gen_fpu_get_sti(ctx, 0);
        ctx.builder.call_fn2_i64_i32_ret_i64("f80_to_f64");
        let value_local = ctx.builder.set_new_local_i64();
        gen_safe_write64(ctx, &address_local, &value_local);
        ctx.builder.free_local(address_local);
        ctx.builder.free_local_i64(value_local);
        if pop {
            ctx.builder.call_fn0("fpu_pop");
        }
        return;
    }

    gen_modrm_resolve(ctx, modrm_byte);
    let address_local = ctx.builder.set_new_local();
    let st0_addr = gen_fpu_st_addr(ctx, 0);
    gen_fpu_relaxed_st_ok(ctx, 0, &st0_addr);
    ctx.builder.eqz_i32();
    ctx.builder.if_void();
    gen_fpu_relaxed_record_fallback(ctx);
    gen_fpu_get_sti(ctx, 0);
    ctx.builder.call_fn2_i64_i32_ret_i64("f80_to_f64");
    let slow_value = ctx.builder.set_new_local_i64();
    gen_safe_write64(ctx, &address_local, &slow_value);
    ctx.builder.free_local_i64(slow_value);
    gen_x87_local_cache_invalidate_all_runtime(ctx);
    ctx.builder.else_();
    gen_fpu_relaxed_record_hit(ctx);
    let fast_value = gen_fpu_load_relaxed_st_bits(ctx, 0, &st0_addr);
    gen_safe_write64(ctx, &address_local, &fast_value.local);
    gen_free_fpu_bits(ctx, fast_value);
    ctx.builder.block_end();
    if pop {
        gen_fpu_relaxed_pop(ctx);
    }
    ctx.builder.free_local(st0_addr);
    ctx.builder.free_local(address_local);
}

pub fn gen_fpu_relaxed_fist_m32(
    ctx: &mut JitContext,
    modrm_byte: ModrmByte,
    pop: bool,
    truncate: bool,
) {
    let helper = if truncate { "fpu_truncate_to_i32" } else { "fpu_convert_to_i32" };
    if !crate::softfloat::is_fpu_relaxed() {
        gen_modrm_resolve(ctx, modrm_byte);
        let address_local = ctx.builder.set_new_local();
        gen_fpu_get_sti(ctx, 0);
        ctx.builder.call_fn2_i64_i32_ret(helper);
        let value_local = ctx.builder.set_new_local();
        gen_safe_write32(ctx, &address_local, &value_local);
        ctx.builder.free_local(address_local);
        ctx.builder.free_local(value_local);
        if pop {
            ctx.builder.call_fn0("fpu_pop");
        }
        return;
    }

    gen_modrm_resolve(ctx, modrm_byte);
    let address_local = ctx.builder.set_new_local();
    let st0_addr = gen_fpu_st_addr(ctx, 0);
    gen_fpu_relaxed_st_ok(ctx, 0, &st0_addr);
    ctx.builder.eqz_i32();
    ctx.builder.if_void();
    gen_fpu_relaxed_record_fallback(ctx);
    gen_fpu_get_sti(ctx, 0);
    ctx.builder.call_fn2_i64_i32_ret(helper);
    let slow_value = ctx.builder.set_new_local();
    gen_safe_write32(ctx, &address_local, &slow_value);
    ctx.builder.free_local(slow_value);
    gen_x87_local_cache_invalidate_all_runtime(ctx);
    ctx.builder.else_();
    gen_fpu_relaxed_record_hit(ctx);
    let bits = gen_fpu_load_relaxed_st_bits(ctx, 0, &st0_addr);
    gen_fpu_round_f64_bits_to_i32(ctx, &bits.local, truncate);
    let fast_value = ctx.builder.set_new_local();
    gen_safe_write32(ctx, &address_local, &fast_value);
    ctx.builder.free_local(fast_value);
    gen_free_fpu_bits(ctx, bits);
    ctx.builder.block_end();
    if pop {
        gen_fpu_relaxed_pop(ctx);
    }
    ctx.builder.free_local(st0_addr);
    ctx.builder.free_local(address_local);
}

pub fn gen_fpu_relaxed_fist_m16(
    ctx: &mut JitContext,
    modrm_byte: ModrmByte,
    pop: bool,
    truncate: bool,
) {
    let helper = if truncate { "fpu_truncate_to_i16" } else { "fpu_convert_to_i16" };
    if !crate::softfloat::is_fpu_relaxed() {
        gen_modrm_resolve(ctx, modrm_byte);
        let address_local = ctx.builder.set_new_local();
        gen_fpu_get_sti(ctx, 0);
        ctx.builder.call_fn2_i64_i32_ret(helper);
        let value_local = ctx.builder.set_new_local();
        gen_safe_write16(ctx, &address_local, &value_local);
        ctx.builder.free_local(address_local);
        ctx.builder.free_local(value_local);
        if pop {
            ctx.builder.call_fn0("fpu_pop");
        }
        return;
    }

    gen_modrm_resolve(ctx, modrm_byte);
    let address_local = ctx.builder.set_new_local();
    let st0_addr = gen_fpu_st_addr(ctx, 0);
    gen_fpu_relaxed_st_ok(ctx, 0, &st0_addr);
    ctx.builder.eqz_i32();
    ctx.builder.if_void();
    gen_fpu_relaxed_record_fallback(ctx);
    gen_fpu_get_sti(ctx, 0);
    ctx.builder.call_fn2_i64_i32_ret(helper);
    let slow_value = ctx.builder.set_new_local();
    gen_safe_write16(ctx, &address_local, &slow_value);
    ctx.builder.free_local(slow_value);
    gen_x87_local_cache_invalidate_all_runtime(ctx);
    ctx.builder.else_();
    gen_fpu_relaxed_record_hit(ctx);
    let bits = gen_fpu_load_relaxed_st_bits(ctx, 0, &st0_addr);
    gen_fpu_round_f64_bits_to_i32(ctx, &bits.local, truncate);
    let rounded = ctx.builder.set_new_local();
    gen_fpu_clamp_i32_to_i16(ctx, &rounded);
    let fast_value = ctx.builder.set_new_local();
    gen_safe_write16(ctx, &address_local, &fast_value);
    ctx.builder.free_local(fast_value);
    ctx.builder.free_local(rounded);
    gen_free_fpu_bits(ctx, bits);
    ctx.builder.block_end();
    if pop {
        gen_fpu_relaxed_pop(ctx);
    }
    ctx.builder.free_local(st0_addr);
    ctx.builder.free_local(address_local);
}

pub fn gen_fpu_relaxed_fcom_sti(
    ctx: &mut JitContext,
    i: u32,
    pop_count: u32,
    slow_helper: &str,
) {
    if !crate::softfloat::is_fpu_relaxed() {
        gen_fpu_get_sti(ctx, i);
        ctx.builder.call_fn2_i64_i32(slow_helper);
        let helper_pops = if slow_helper == "fpu_fcomp" { 1 } else { 0 };
        if pop_count > helper_pops {
            for _ in 0..(pop_count - helper_pops) {
                gen_fn0_const(ctx.builder, "fpu_pop");
            }
        }
        return;
    }

    let st0_addr = gen_fpu_st_addr(ctx, 0);
    let sti_addr = gen_fpu_st_addr(ctx, i);
    gen_fpu_relaxed_st_ok(ctx, 0, &st0_addr);
    gen_fpu_relaxed_st_ok(ctx, i, &sti_addr);
    ctx.builder.and_i32();
    ctx.builder.eqz_i32();
    ctx.builder.if_void();
    gen_fpu_relaxed_record_fallback(ctx);
    gen_fpu_get_sti(ctx, i);
    ctx.builder.call_fn2_i64_i32(slow_helper);
    let helper_pops = if slow_helper == "fpu_fcomp" { 1 } else { 0 };
    if pop_count > helper_pops {
        for _ in 0..(pop_count - helper_pops) {
            gen_fn0_const(ctx.builder, "fpu_pop");
        }
    }
    gen_x87_local_cache_invalidate_all_runtime(ctx);
    ctx.builder.else_();
    gen_fpu_relaxed_record_hit(ctx);
    let x_bits = gen_fpu_load_relaxed_st_bits(ctx, 0, &st0_addr);
    let y_bits = gen_fpu_load_relaxed_st_bits(ctx, i, &sti_addr);
    gen_fpu_compare_status_from_bits(ctx, &x_bits.local, &y_bits.local);
    gen_free_fpu_bits(ctx, y_bits);
    gen_free_fpu_bits(ctx, x_bits);
    gen_fpu_relaxed_pop_n(ctx, pop_count);
    ctx.builder.block_end();
    ctx.builder.free_local(sti_addr);
    ctx.builder.free_local(st0_addr);
}

// Relaxed-FPU compare (fcom/fcomp/ficom/…) against a memory operand.
fn gen_fpu_relaxed_fcom_mem(
    ctx: &mut JitContext,
    modrm_byte: ModrmByte,
    pop_count: u32,
    slow_helper: &str,
    src: FpuMemSrc,
) {
    if !crate::softfloat::is_fpu_relaxed() {
        gen_fpu_load_mem_fcom_slow(ctx, src, modrm_byte);
        ctx.builder.call_fn2_i64_i32(slow_helper);
        return;
    }
    let modrm_slow = modrm_byte.clone();
    let st0_addr = gen_fpu_st_addr(ctx, 0);
    gen_fpu_relaxed_st_ok(ctx, 0, &st0_addr);
    ctx.builder.eqz_i32();
    ctx.builder.if_void();
    gen_fpu_relaxed_record_fallback(ctx);
    gen_fpu_load_mem_fcom_slow(ctx, src, modrm_slow);
    ctx.builder.call_fn2_i64_i32(slow_helper);
    gen_x87_local_cache_invalidate_all_runtime(ctx);
    ctx.builder.else_();
    gen_fpu_relaxed_record_hit(ctx);
    let x_bits = gen_fpu_load_relaxed_st_bits(ctx, 0, &st0_addr);
    let y_bits = gen_fpu_load_mem_as_f64_bits(ctx, src, modrm_byte);
    gen_fpu_compare_status_from_bits(ctx, &x_bits.local, &y_bits);
    ctx.builder.free_local_i64(y_bits);
    gen_free_fpu_bits(ctx, x_bits);
    gen_fpu_relaxed_pop_n(ctx, pop_count);
    ctx.builder.block_end();
    ctx.builder.free_local(st0_addr);
}

pub fn gen_fpu_relaxed_fcom_m32(ctx: &mut JitContext, modrm_byte: ModrmByte, pop_count: u32, slow_helper: &str) {
    gen_fpu_relaxed_fcom_mem(ctx, modrm_byte, pop_count, slow_helper, FpuMemSrc::M32)
}
pub fn gen_fpu_relaxed_fcom_m64(ctx: &mut JitContext, modrm_byte: ModrmByte, pop_count: u32, slow_helper: &str) {
    gen_fpu_relaxed_fcom_mem(ctx, modrm_byte, pop_count, slow_helper, FpuMemSrc::M64)
}
pub fn gen_fpu_relaxed_fcom_i16(ctx: &mut JitContext, modrm_byte: ModrmByte, pop_count: u32, slow_helper: &str) {
    gen_fpu_relaxed_fcom_mem(ctx, modrm_byte, pop_count, slow_helper, FpuMemSrc::I16)
}
pub fn gen_fpu_relaxed_fcom_i32(ctx: &mut JitContext, modrm_byte: ModrmByte, pop_count: u32, slow_helper: &str) {
    gen_fpu_relaxed_fcom_mem(ctx, modrm_byte, pop_count, slow_helper, FpuMemSrc::I32)
}

pub fn gen_fpu_relaxed_fucom_sti(
    ctx: &mut JitContext,
    i: u32,
    pop_count: u32,
    slow_helper: &str,
) {
    if !crate::softfloat::is_fpu_relaxed() {
        gen_fn1_const(ctx.builder, slow_helper, i);
        return;
    }

    let st0_addr = gen_fpu_st_addr(ctx, 0);
    let sti_addr = gen_fpu_st_addr(ctx, i);
    gen_fpu_relaxed_st_ok(ctx, 0, &st0_addr);
    gen_fpu_relaxed_st_ok(ctx, i, &sti_addr);
    ctx.builder.and_i32();
    ctx.builder.eqz_i32();
    ctx.builder.if_void();
    gen_fpu_relaxed_record_fallback(ctx);
    gen_fn1_const(ctx.builder, slow_helper, i);
    gen_x87_local_cache_invalidate_all_runtime(ctx);
    ctx.builder.else_();
    gen_fpu_relaxed_record_hit(ctx);
    let x_bits = gen_fpu_load_relaxed_st_bits(ctx, 0, &st0_addr);
    let y_bits = gen_fpu_load_relaxed_st_bits(ctx, i, &sti_addr);
    gen_fpu_compare_status_from_bits(ctx, &x_bits.local, &y_bits.local);
    gen_free_fpu_bits(ctx, y_bits);
    gen_free_fpu_bits(ctx, x_bits);
    gen_fpu_relaxed_pop_n(ctx, pop_count);
    ctx.builder.block_end();
    ctx.builder.free_local(sti_addr);
    ctx.builder.free_local(st0_addr);
}

pub fn gen_fpu_relaxed_fucompp(ctx: &mut JitContext) {
    if !crate::softfloat::is_fpu_relaxed() {
        ctx.builder.call_fn0("fpu_fucompp");
        return;
    }

    let st0_addr = gen_fpu_st_addr(ctx, 0);
    let st1_addr = gen_fpu_st_addr(ctx, 1);
    gen_fpu_relaxed_st_ok(ctx, 0, &st0_addr);
    gen_fpu_relaxed_st_ok(ctx, 1, &st1_addr);
    ctx.builder.and_i32();
    ctx.builder.eqz_i32();
    ctx.builder.if_void();
    gen_fpu_relaxed_record_fallback(ctx);
    ctx.builder.call_fn0("fpu_fucompp");
    gen_x87_local_cache_invalidate_all_runtime(ctx);
    ctx.builder.else_();
    gen_fpu_relaxed_record_hit(ctx);
    let x_bits = gen_fpu_load_relaxed_st_bits(ctx, 0, &st0_addr);
    let y_bits = gen_fpu_load_relaxed_st_bits(ctx, 1, &st1_addr);
    gen_fpu_compare_status_from_bits(ctx, &x_bits.local, &y_bits.local);
    gen_free_fpu_bits(ctx, y_bits);
    gen_free_fpu_bits(ctx, x_bits);
    gen_fpu_relaxed_pop_n(ctx, 2);
    ctx.builder.block_end();
    ctx.builder.free_local(st1_addr);
    ctx.builder.free_local(st0_addr);
}

pub fn gen_fpu_relaxed_fcomi(ctx: &mut JitContext, i: u32, pop: bool, slow_helper: &str) {
    if !crate::softfloat::is_fpu_relaxed() {
        gen_fn1_const(ctx.builder, slow_helper, i);
        return;
    }

    let st0_addr = gen_fpu_st_addr(ctx, 0);
    let sti_addr = gen_fpu_st_addr(ctx, i);
    gen_fpu_relaxed_st_ok(ctx, 0, &st0_addr);
    gen_fpu_relaxed_st_ok(ctx, i, &sti_addr);
    ctx.builder.and_i32();
    ctx.builder.eqz_i32();
    ctx.builder.if_void();
    gen_fpu_relaxed_record_fallback(ctx);
    gen_fn1_const(ctx.builder, slow_helper, i);
    gen_x87_local_cache_invalidate_all_runtime(ctx);
    ctx.builder.else_();
    gen_fpu_relaxed_record_hit(ctx);
    let x_bits = gen_fpu_load_relaxed_st_bits(ctx, 0, &st0_addr);
    let y_bits = gen_fpu_load_relaxed_st_bits(ctx, i, &sti_addr);
    gen_fpu_compare_eflags_from_bits(ctx, &x_bits.local, &y_bits.local);
    gen_free_fpu_bits(ctx, y_bits);
    gen_free_fpu_bits(ctx, x_bits);
    if pop {
        gen_fpu_relaxed_pop(ctx);
    }
    ctx.builder.block_end();
    ctx.builder.free_local(sti_addr);
    ctx.builder.free_local(st0_addr);
}

pub fn gen_trigger_de(ctx: &mut JitContext) {
    gen_fn1_const(
        ctx.builder,
        "trigger_de_jit",
        ctx.start_of_current_instruction & 0xFFF,
    );
    gen_debug_track_jit_exit(ctx.builder, ctx.start_of_current_instruction);
    ctx.builder.br(ctx.exit_with_fault_label);
}

pub fn gen_trigger_ud(ctx: &mut JitContext) {
    gen_fn1_const(
        ctx.builder,
        "trigger_ud_jit",
        ctx.start_of_current_instruction & 0xFFF,
    );
    gen_debug_track_jit_exit(ctx.builder, ctx.start_of_current_instruction);
    ctx.builder.br(ctx.exit_with_fault_label);
}

pub fn gen_trigger_gp(ctx: &mut JitContext, error_code: u32) {
    gen_fn2_const(
        ctx.builder,
        "trigger_gp_jit",
        error_code,
        ctx.start_of_current_instruction & 0xFFF,
    );
    gen_debug_track_jit_exit(ctx.builder, ctx.start_of_current_instruction);
    ctx.builder.br(ctx.exit_with_fault_label);
}

pub fn gen_condition_fn_negated(ctx: &mut JitContext, condition: u8) {
    gen_condition_fn(ctx, condition ^ 1)
}

pub fn gen_condition_fn(ctx: &mut JitContext, condition: u8) {
    if condition & 0xF0 == 0x00 || condition & 0xF0 == 0x70 || condition & 0xF0 == 0x80 {
        match condition & 0xF {
            0x0 => {
                gen_getof(ctx);
            },
            0x1 => {
                gen_getof(ctx);
                ctx.builder.eqz_i32();
            },
            0x2 => {
                gen_getcf(ctx, ConditionNegate::False);
            },
            0x3 => {
                gen_getcf(ctx, ConditionNegate::True);
            },
            0x4 => {
                gen_getzf(ctx, ConditionNegate::False);
            },
            0x5 => {
                gen_getzf(ctx, ConditionNegate::True);
            },
            0x6 => {
                gen_test_be(ctx, ConditionNegate::False);
            },
            0x7 => {
                gen_test_be(ctx, ConditionNegate::True);
            },
            0x8 => {
                gen_getsf(ctx, ConditionNegate::False);
            },
            0x9 => {
                gen_getsf(ctx, ConditionNegate::True);
            },
            0xA => {
                gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_UNOPTIMISED);
                gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_UNOPTIMISED_PF);
                ctx.builder.call_fn0_ret("test_p");
            },
            0xB => {
                gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_UNOPTIMISED);
                gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_UNOPTIMISED_PF);
                ctx.builder.call_fn0_ret("test_np");
            },
            0xC => {
                gen_test_l(ctx, ConditionNegate::False);
            },
            0xD => {
                gen_test_l(ctx, ConditionNegate::True);
            },
            0xE => {
                gen_test_le(ctx, ConditionNegate::False);
            },
            0xF => {
                gen_test_le(ctx, ConditionNegate::True);
            },
            _ => {
                dbg_assert!(false);
            },
        }
    }
    else {
        // loop, loopnz, loopz, jcxz
        dbg_assert!(condition & !0x3 == 0xE0);
        if condition == 0xE0 {
            gen_test_loopnz(ctx, ctx.cpu.asize_32());
        }
        else if condition == 0xE1 {
            gen_test_loopz(ctx, ctx.cpu.asize_32());
        }
        else if condition == 0xE2 {
            gen_test_loop(ctx, ctx.cpu.asize_32());
        }
        else if condition == 0xE3 {
            gen_test_jcxz(ctx, ctx.cpu.asize_32());
        }
    }
}

pub fn gen_move_registers_from_locals_to_memory(ctx: &mut JitContext) {
    if opstats::opstats_enabled() {
        let instruction = memory::read32s(ctx.start_of_current_instruction) as u32;
        opstats::gen_opstat_unguarded_register(ctx.builder, instruction);
    }

    for i in 0..8 {
        ctx.builder
            .const_i32(global_pointers::get_reg32_offset(i as u32) as i32);
        ctx.builder.get_local(&ctx.register_locals[i]);
        ctx.builder.store_aligned_i32(0);
    }
    // Flag locals share the registers' spill discipline: wherever guest register
    // state must be visible in memory (block-boundary helpers, module exits, the
    // OUT/hypercall context-save class), the flag tuple must be too.
    ctx.builder.emit_flag_spill();
}
pub fn gen_move_registers_from_memory_to_locals(ctx: &mut JitContext) {
    if opstats::opstats_enabled() {
        let instruction = memory::read32s(ctx.start_of_current_instruction) as u32;
        opstats::gen_opstat_unguarded_register(ctx.builder, instruction);
    }

    for i in 0..8 {
        ctx.builder
            .const_i32(global_pointers::get_reg32_offset(i as u32) as i32);
        ctx.builder.load_aligned_i32(0);
        ctx.builder.set_local(&ctx.register_locals[i]);
    }
    // Mirror of the spill above: the helper may have modified flag state in memory.
    ctx.builder.emit_flag_reload();
}

pub fn gen_profiler_stat_increment(builder: &mut WasmBuilder, stat: profiler::stat) {
    if !cfg!(feature = "profiler") {
        return;
    }
    let addr = unsafe { &raw mut profiler::stat_array[stat as usize] } as u32;
    builder.increment_fixed_i64(addr, 1)
}

pub fn gen_debug_track_jit_exit(builder: &mut WasmBuilder, address: u32) {
    if cfg!(feature = "profiler") {
        gen_fn1_const(builder, "track_jit_exit", address);
    }
}
