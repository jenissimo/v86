use crate::wasmgen::wasm_builder::WasmBuilder;

// The census used to exist only in a `profiler` build, whose shipping counterpart
// exported a stub returning zeros — a readout indistinguishable from "this game runs no
// x87". It is now a RUNTIME switch instead: production emits nothing (the flag is off, so
// no increment is generated), and a census run flips the flag and clears the JIT cache so
// hot code recompiles with the counters in it. Cost when off is the static buffers below.
const SIZE: usize = 8192;

#[allow(non_upper_case_globals)]
pub static mut OPSTATS_ENABLED: bool = false;

pub fn opstats_enabled() -> bool { unsafe { OPSTATS_ENABLED } }

/// Turn census emission on/off. Only blocks compiled while this is on carry the
/// increments, so the caller must clear the JIT cache and let the workload warm up.
#[no_mangle]
pub fn set_opstats(enabled: u32) { unsafe { OPSTATS_ENABLED = enabled != 0 } }

#[no_mangle]
pub fn get_opstats() -> u32 { unsafe { OPSTATS_ENABLED as u32 } }

#[no_mangle]
pub fn opstats_reset() {
    unsafe {
        #[allow(static_mut_refs)]
        for b in [
            &mut opstats_buffer, &mut opstats_compiled_buffer, &mut opstats_jit_exit_buffer,
            &mut opstats_unguarded_register_buffer, &mut opstats_wasm_size,
        ] {
            for x in b.iter_mut() { *x = 0 }
        }
        #[allow(static_mut_refs)]
        for x in opstats_addr_buffer.iter_mut() { *x = 0 }
        #[allow(static_mut_refs)]
        for x in opstats_simd_buffer.iter_mut() { *x = 0 }
    }
}

#[allow(non_upper_case_globals)]
pub static mut opstats_buffer: [u64; SIZE] = [0; SIZE];
#[allow(non_upper_case_globals)]
pub static mut opstats_compiled_buffer: [u64; SIZE] = [0; SIZE];
#[allow(non_upper_case_globals)]
pub static mut opstats_jit_exit_buffer: [u64; SIZE] = [0; SIZE];
#[allow(non_upper_case_globals)]
pub static mut opstats_unguarded_register_buffer: [u64; SIZE] = [0; SIZE];
#[allow(non_upper_case_globals)]
pub static mut opstats_wasm_size: [u64; SIZE] = [0; SIZE];

pub struct Instruction {
    pub prefixes: Vec<u8>,
    pub opcode: u8,
    pub fixed_g: u8,
    pub is_mem: bool,
    pub is_0f: bool,
    /// The ModRM byte, when the opcode has one, and the SIB byte when ModRM asks for one.
    /// `is_mem` alone cannot tell `[esp+8]` from `[eax+ebx*4]`, and that difference is the
    /// whole question behind stack fastmem (roadmap 02) and the permission bitmap (03).
    pub modrm: u8,
    pub has_modrm: bool,
    pub sib: u8,
    pub has_sib: bool,
    /// A 0x67 prefix was present, so the ModRM encoding is the 16-bit one. Classified as
    /// its own bucket rather than mis-modelled as a 32-bit form.
    pub addr16: bool,
}

pub fn decode(instruction: u32) -> Instruction { decode64(instruction as u64) }

/// The same prefix walk over up to eight bytes. A prefixed 0F opcode pushes its SIB byte
/// past the fourth byte (`66 0F 6F 44 24 10`), so a u32 cannot see the operand form.
pub fn decode64(bytes: u64) -> Instruction {
    let mut instruction = bytes;
    let mut is_0f = false;
    let mut prefixes = vec![];
    let mut final_opcode = 0;

    for _ in 0..7 {
        let opcode = (instruction & 0xFF) as u8;
        instruction >>= 8;

        // TODO:
        // - If the instruction uses 4 or more prefixes, only the prefixes will be counted

        if is_0f {
            final_opcode = opcode;
            break;
        }
        else {
            if opcode == 0x0F {
                is_0f = true;
            }
            else if opcode == 0x26
                || opcode == 0x2E
                || opcode == 0x36
                || opcode == 0x3E
                || opcode == 0x64
                || opcode == 0x65
                || opcode == 0x66
                || opcode == 0x67
                || opcode == 0xF0
                || opcode == 0xF2
                || opcode == 0xF3
            {
                prefixes.push(opcode);
            }
            else {
                final_opcode = opcode;
                break;
            }
        }
    }

    let has_modrm_byte = if is_0f {
        match final_opcode {
            0x0 | 0x1 | 0x2 | 0x3 | 0x10 | 0x11 | 0x12 | 0x13 | 0x14 | 0x15 | 0x16 | 0x17
            | 0x18 | 0x19 | 0x20 | 0x21 | 0x22 | 0x23 | 0x28 | 0x29 | 0x40 | 0x41 | 0x42 | 0x43
            | 0x44 | 0x45 | 0x46 | 0x47 | 0x48 | 0x49 | 0x50 | 0x51 | 0x52 | 0x53 | 0x54 | 0x55
            | 0x56 | 0x57 | 0x58 | 0x59 | 0x60 | 0x61 | 0x62 | 0x63 | 0x64 | 0x65 | 0x66 | 0x67
            | 0x68 | 0x69 | 0x70 | 0x71 | 0x72 | 0x73 | 0x74 | 0x75 | 0x76 | 0x90 | 0x91 | 0x92
            | 0x93 | 0x94 | 0x95 | 0x96 | 0x97 | 0x98 | 0x99 | 0x1c | 0x1d | 0x1e | 0x1f | 0x2a
            | 0x2b | 0x2c | 0x2d | 0x2e | 0x2f | 0x4a | 0x4b | 0x4c | 0x4d | 0x4e | 0x4f | 0x5a
            | 0x5b | 0x5c | 0x5d | 0x5e | 0x5f | 0x6a | 0x6b | 0x6c | 0x6d | 0x6e | 0x6f | 0x7e
            | 0x7f | 0x9a | 0x9b | 0x9c | 0x9d | 0x9e | 0x9f | 0xa3 | 0xa4 | 0xa5 | 0xab | 0xac
            | 0xad | 0xae | 0xaf | 0xb0 | 0xb1 | 0xb2 | 0xb3 | 0xb4 | 0xb5 | 0xb6 | 0xb7 | 0xb8
            | 0xba | 0xbb | 0xbc | 0xbd | 0xbe | 0xbf | 0xc0 | 0xc1 | 0xc2 | 0xc3 | 0xc4 | 0xc5
            | 0xc6 | 0xc7 | 0xd1 | 0xd2 | 0xd3 | 0xd4 | 0xd5 | 0xd6 | 0xd7 | 0xd8 | 0xd9 | 0xda
            | 0xdb | 0xdc | 0xdd | 0xde | 0xdf | 0xe0 | 0xe1 | 0xe2 | 0xe3 | 0xe4 | 0xe5 | 0xe6
            | 0xe7 | 0xe8 | 0xe9 | 0xea | 0xeb | 0xec | 0xed | 0xee | 0xef | 0xf1 | 0xf2 | 0xf3
            | 0xf4 | 0xf5 | 0xf6 | 0xf7 | 0xf8 | 0xf9 | 0xfa | 0xfb | 0xfc | 0xfd | 0xfe => true,
            _ => false,
        }
    }
    else {
        match final_opcode {
            0x0 | 0x1 | 0x2 | 0x3 | 0x8 | 0x9 | 0x10 | 0x11 | 0x12 | 0x13 | 0x18 | 0x19 | 0x20
            | 0x21 | 0x22 | 0x23 | 0x28 | 0x29 | 0x30 | 0x31 | 0x32 | 0x33 | 0x38 | 0x39 | 0x62
            | 0x63 | 0x69 | 0x80 | 0x81 | 0x82 | 0x83 | 0x84 | 0x85 | 0x86 | 0x87 | 0x88 | 0x89
            | 0xa | 0xb | 0x1a | 0x1b | 0x2a | 0x2b | 0x3a | 0x3b | 0x6b | 0x8a | 0x8b | 0x8c
            | 0x8d | 0x8e | 0x8f | 0xc0 | 0xc1 | 0xc4 | 0xc5 | 0xc6 | 0xc7 | 0xd0 | 0xd1 | 0xd2
            | 0xd3 | 0xd8 | 0xd9 | 0xda | 0xdb | 0xdc | 0xdd | 0xde | 0xdf | 0xf6 | 0xf7 | 0xfe
            | 0xff => true,
            _ => false,
        }
    };

    let has_fixed_g = if is_0f {
        final_opcode == 0x71
            || final_opcode == 0x72
            || final_opcode == 0x73
            || final_opcode == 0xAE
            || final_opcode == 0xBA
            || final_opcode == 0xC7
    }
    else {
        final_opcode >= 0x80 && final_opcode < 0x84
            || final_opcode >= 0xC0 && final_opcode < 0xC2
            || final_opcode >= 0xD0 && final_opcode < 0xD4
            || final_opcode >= 0xD8 && final_opcode < 0xE0
            || final_opcode >= 0xF6 && final_opcode < 0xF8
            || final_opcode == 0xFE
            || final_opcode == 0xFF
    };

    let mut is_mem = false;
    let mut fixed_g = 0;

    let modrm = (instruction & 0xFF) as u8;
    if has_fixed_g {
        dbg_assert!(has_modrm_byte);
        fixed_g = modrm >> 3 & 7;
        is_mem = modrm < 0xC0
    }
    if has_modrm_byte {
        is_mem = modrm < 0xC0
    }
    let addr16 = prefixes.contains(&0x67);
    // A SIB byte follows ModRM only in 32-bit addressing, for a memory operand whose
    // rm field is 4. In 16-bit addressing that encoding means [si], not "see SIB".
    let has_sib = has_modrm_byte && !addr16 && modrm < 0xC0 && (modrm & 7) == 4;
    let sib = if has_sib { ((instruction >> 8) & 0xFF) as u8 } else { 0 };

    Instruction {
        prefixes,
        opcode: final_opcode,
        is_mem,
        fixed_g,
        is_0f,
        modrm,
        has_modrm: has_modrm_byte,
        sib,
        has_sib,
        addr16,
    }
}


// ---------------------------------------------------------------------------
// Addressing-form census (roadmap 02/03)
// ---------------------------------------------------------------------------
//
// `is_mem` says an instruction touches memory; it cannot say through WHAT. The share of
// accesses that are ESP/EBP-based with a constant displacement is the ceiling on stack
// fastmem, and the remainder is what a permission bitmap would have to serve. Guessing
// that split from a proxy demo is what the last campaign did.
//
// This side does bit extraction ONLY. What counts as "the stack class", "absolute" or
// "base+index" is a question about x86 semantics, and it is answered once — in
// TypeScript, where it has a unit test over a table of encodings
// (tools/tests/guest-opcode-census.test.ts). Two implementations of a judgement drift;
// one implementation plus a mechanical feed does not.
//
// Only instructions with an explicit ModRM memory operand are counted here, so the sum of
// this buffer must equal the `is_mem` half of the opcode census — an independent
// cross-check between two separately produced numbers, not an identity.

pub const ADDRKEY_COUNT: usize = 256;

#[allow(non_upper_case_globals)]
pub static mut opstats_addr_buffer: [u64; ADDRKEY_COUNT] = [0; ADDRKEY_COUNT];

#[no_mangle]
pub fn get_opstats_addr(index: u32) -> f64 {
    if (index as usize) < ADDRKEY_COUNT { unsafe { opstats_addr_buffer[index as usize] as f64 } } else { 0.0 }
}

/// bit 7 addr16 | bit 6 has_sib | bits 5-4 mod | bits 3-1 base (SIB base, else rm) |
/// bit 0 index present (SIB index != 4).
pub fn addr_key(i: &Instruction) -> u32 {
    let md = (i.modrm >> 6 & 3) as u32;
    let base = if i.has_sib { (i.sib & 7) as u32 } else { (i.modrm & 7) as u32 };
    let index_present = if i.has_sib && (i.sib >> 3 & 7) != 4 { 1 } else { 0 };
    (i.addr16 as u32) << 7 | (i.has_sib as u32) << 6 | md << 4 | base << 1 | index_present
}

pub fn gen_addr_stat(builder: &mut WasmBuilder, i: &Instruction) {
    // is_mem is only ever set from a ModRM byte, so the second test is an invariant, not
    // a case: without a ModRM the key below would be extracted from whatever byte follows.
    if !i.is_mem || !i.has_modrm { return; }
    builder.increment_fixed_i64(
        unsafe { &mut opstats_addr_buffer[addr_key(i) as usize] as *mut _ } as u32,
        1,
    );
}

// The 0F census key drops prefixes, and for SIMD the prefix IS the family: 0F 58 is
// ADDPS, 66 0F 58 ADDPD, F3 0F 58 ADDSS, F2 0F 58 ADDSD, and the unprefixed integer forms
// are MMX rather than SSE. Reporting all of those as one row would answer "how much SSE2"
// with a number that also contains MMX. One extra buffer, keyed by the mandatory prefix,
// keeps the families apart; the naming of the families stays in TypeScript.

pub const SIMDKEY_COUNT: usize = 1024;

#[allow(non_upper_case_globals)]
pub static mut opstats_simd_buffer: [u64; SIMDKEY_COUNT] = [0; SIMDKEY_COUNT];

#[no_mangle]
pub fn get_opstats_simd(index: u32) -> f64 {
    if (index as usize) < SIMDKEY_COUNT { unsafe { opstats_simd_buffer[index as usize] as f64 } } else { 0.0 }
}

/// 0 = none (MMX / packed-single), 1 = 0x66, 2 = 0xF3, 3 = 0xF2. The LAST such prefix
/// wins, which is what the hardware does.
fn mandatory_prefix(i: &Instruction) -> u32 {
    let mut p = 0;
    for &b in i.prefixes.iter() {
        match b {
            0x66 => p = 1,
            0xF3 => p = 2,
            0xF2 => p = 3,
            _ => {},
        }
    }
    p
}

pub fn gen_simd_stat(builder: &mut WasmBuilder, i: &Instruction) {
    if !i.is_0f { return; }
    let key = mandatory_prefix(i) << 8 | i.opcode as u32;
    builder.increment_fixed_i64(
        unsafe { &mut opstats_simd_buffer[key as usize] as *mut _ } as u32,
        1,
    );
}

pub fn gen_opstats(builder: &mut WasmBuilder, opcode: u64) {
    if !opstats_enabled() {
        return;
    }

    let instruction = decode64(opcode);
    gen_addr_stat(builder, &instruction);
    gen_simd_stat(builder, &instruction);

    for prefix in instruction.prefixes {
        let index = (prefix as u32) << 4;
        builder.increment_fixed_i64(
            unsafe { &mut opstats_buffer[index as usize] as *mut _ } as u32,
            1,
        );
    }

    let index = (instruction.is_0f as u32) << 12
        | (instruction.opcode as u32) << 4
        | (instruction.is_mem as u32) << 3
        | instruction.fixed_g as u32;

    builder.increment_fixed_i64(
        unsafe { &mut opstats_buffer[index as usize] as *mut _ } as u32,
        1,
    );
}

pub fn record_opstat_compiled(opcode: u64) {
    if !opstats_enabled() {
        return;
    }

    let instruction = decode64(opcode);

    for prefix in instruction.prefixes {
        let index = (prefix as u32) << 4;
        unsafe { opstats_compiled_buffer[index as usize] += 1 }
    }

    let index = (instruction.is_0f as u32) << 12
        | (instruction.opcode as u32) << 4
        | (instruction.is_mem as u32) << 3
        | instruction.fixed_g as u32;

    unsafe { opstats_compiled_buffer[index as usize] += 1 }
}

pub fn record_opstat_jit_exit(opcode: u32) {
    if !opstats_enabled() {
        return;
    }

    let instruction = decode(opcode);

    for prefix in instruction.prefixes {
        let index = (prefix as u32) << 4;
        unsafe { opstats_jit_exit_buffer[index as usize] += 1 }
    }

    let index = (instruction.is_0f as u32) << 12
        | (instruction.opcode as u32) << 4
        | (instruction.is_mem as u32) << 3
        | instruction.fixed_g as u32;

    unsafe { opstats_jit_exit_buffer[index as usize] += 1 }
}

pub fn gen_opstat_unguarded_register(builder: &mut WasmBuilder, opcode: u32) {
    if !opstats_enabled() {
        return;
    }

    let instruction = decode(opcode);

    for prefix in instruction.prefixes {
        let index = (prefix as u32) << 4;
        builder.increment_fixed_i64(
            unsafe { &mut opstats_unguarded_register_buffer[index as usize] as *mut _ } as u32,
            1,
        );
    }

    let index = (instruction.is_0f as u32) << 12
        | (instruction.opcode as u32) << 4
        | (instruction.is_mem as u32) << 3
        | instruction.fixed_g as u32;

    builder.increment_fixed_i64(
        unsafe { &mut opstats_unguarded_register_buffer[index as usize] as *mut _ } as u32,
        1,
    );
}

pub fn record_opstat_size_wasm(opcode: u64, size: u64) {
    if !opstats_enabled() {
        return;
    }

    let instruction = decode64(opcode);

    for prefix in instruction.prefixes {
        let index = (prefix as u32) << 4;
        unsafe { opstats_wasm_size[index as usize] += size }
    }

    let index = (instruction.is_0f as u32) << 12
        | (instruction.opcode as u32) << 4
        | (instruction.is_mem as u32) << 3
        | instruction.fixed_g as u32;

    unsafe { opstats_wasm_size[index as usize] += size }
}
