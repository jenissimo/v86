use std::collections::HashMap;
use std::mem::transmute;

use crate::leb::{
    write_fixed_leb16_at_idx, write_fixed_leb32_at_idx, write_leb_i32, write_leb_i64, write_leb_u32,
};
use crate::wasmgen::wasm_opcodes as op;

pub trait SafeToU8 {
    fn safe_to_u8(self) -> u8;
}
impl SafeToU8 for usize {
    fn safe_to_u8(self) -> u8 {
        dbg_assert!(self <= ::std::u8::MAX as usize);
        self as u8
    }
}

pub trait SafeToU16 {
    fn safe_to_u16(self) -> u16;
}
impl SafeToU16 for usize {
    fn safe_to_u16(self) -> u16 {
        dbg_assert!(self <= ::std::u16::MAX as usize);
        self as u16
    }
}

#[derive(PartialEq)]
#[allow(non_camel_case_types)]
enum FunctionType {
    FN0,
    FN1,
    FN2,
    FN3,

    FN0_RET,
    FN0_RET_I64,
    FN1_RET,
    FN2_RET,

    FN1_RET_I64,
    FN1_F32_RET,
    FN1_F64_RET,

    FN2_I32_I64,
    FN2_I64_I32,
    FN2_I64_I32_RET,
    FN2_I64_I32_RET_I64,
    FN2_F32_I32,

    FN3_RET,

    FN3_I64_I32_I32,
    FN3_I32_I64_I32,
    FN3_I32_I64_I32_RET,
    FN4_I32_I64_I64_I32_RET,
    // When adding at the end, update LAST below
}

impl FunctionType {
    pub fn of_u8(x: u8) -> FunctionType {
        dbg_assert!(x <= FunctionType::LAST as u8);
        unsafe { transmute(x) }
    }
    pub fn to_u8(self: FunctionType) -> u8 { self as u8 }
    pub const LAST: FunctionType = FunctionType::FN4_I32_I64_I64_I32_RET;
}

pub const WASM_MODULE_ARGUMENT_COUNT: u8 = 1;

// Wasm branch hinting (finished proposal, core 3.0): a custom section named
// "metadata.code.branch_hint" annotates `if`/`br_if` instructions with the expected
// value of their condition. Consumed by V8's optimizing tier only (Liftoff ignores it);
// a malformed section can never invalidate the module — the decoder drops the hints.
pub const BRANCH_HINT_SECTION_NAME: &str = "metadata.code.branch_hint";
/// The standard wasm `name` custom section (BinaryFormatAnnotations.md).
pub const NAME_SECTION_NAME: &str = "name";
/// Condition is expected to be false (the hinted branch is NOT taken).
pub const HINT_UNLIKELY: u8 = 0;
/// Condition is expected to be true (the hinted branch IS taken).
pub const HINT_LIKELY: u8 = 1;

// Hint groups — bits of jit config idx 22, so a hint whose real distribution turns out
// to be less lopsided than assumed can be ablated without a rebuild.
/// Memory guards: TLB hit/miss, fastmem range tests, #PF exits, page-switch check.
pub const HINT_GROUP_MEM: u32 = 1;
/// x87 relaxed-mode local-cache validity (churns more than the memory guards).
pub const HINT_GROUP_X87: u32 = 2;

pub struct WasmBuilder {
    output: Vec<u8>,
    instruction_body: Vec<u8>,

    idx_import_table_size: usize, // for rewriting once finished
    idx_import_count: usize,      // for rewriting once finished
    idx_import_entries: usize,    // for searching the imports

    import_table_size: usize, // the current import table size (to avoid reading 2 byte leb)
    import_count: u16,        // same as above
    function_import_count: u16,
    import_indirect_function_table: bool,

    initial_static_size: usize, // size of module after initialization, rest is drained on reset

    // label for referencing block/if/loop constructs directly via branch instructions
    next_label: Label,
    label_stack: Vec<Label>,
    label_to_depth: HashMap<Label, usize>,

    free_locals_i32: Vec<WasmLocal>,
    free_locals_i64: Vec<WasmLocalI64>,
    local_count: u8,
    pub arg_local_initial_state: WasmLocal,

    // Flag-tuple in locals (jit config idx 21). Registered by
    // jit_generate_module when enabled: (local_idx, linear-memory address) for each
    // of the five lazy-flag globals. While Some, the locals are the authoritative
    // flag state and call_fn() spills them to memory before — and reloads them
    // after — every call whose target is not in the proven-flag-pure whitelist
    // (unknown helper = spilled = safe; covers arith flag-protocol helpers AND
    // OUT/hypercall thunks whose context switches read cpu.get_eflags()).
    pub flag_locals: Option<[(u8, u32); 5]>,
    /// Which flag locals may differ from their memory globals at this point in the emission.
    ///
    /// A word nothing has written since the last sync is already in memory, so the spill before a
    /// call has nothing to write for it. That is the whole cost of this feature in code that never
    /// touches flags — x87 runs, where every instruction is a helper call and the tuple is
    /// untouched between them — and it is why the feature measured as a loss there.
    ///
    /// The tracking is LINEAR, and codegen here sees one path at a time, so it is pessimised to
    /// "all dirty" at every control-flow boundary: a label can be reached from a path that left a
    /// word in its local. A set bit costs a store that may be redundant; a clear bit is a claim
    /// that memory is already right on EVERY path reaching here.
    flag_dirty: u8,

    // Branch hints (jit config idx 22). Offsets are recorded relative to the start of
    // instruction_body and rebased onto the locals declaration in finish(); this is only
    // sound because instruction_body is append-only (no insert/splice/truncate anywhere).
    // Any future pass that rewrites the body must recompute or clear this vector — a stale
    // offset does not corrupt the module, it silently loses the hint.
    branch_hints: Vec<(u32, u8)>,
    /// Bitmask of enabled HINT_GROUP_*; 0 disables emission entirely.
    pub branch_hint_mask: u32,
    // Robustness self-test (jit config idx 23): shift every emitted offset by N so the
    // hints deliberately miss their instruction. The module must still compile and run.
    pub branch_hint_offset_fuzz: u32,

    /// Name for this module's single function body, emitted as a wasm `name` section.
    /// Empty = no section (the default; the JIT fills it only when its naming knob is on).
    /// Owned buffer reused across modules so naming costs no allocation per compile.
    pub function_name: Vec<u8>,
}

// Helpers proven not to touch the lazy-flag globals. Everything else
// gets the spill/reload pair — including every generated instr_* helper.
fn flag_spill_whitelisted(name: &str) -> bool {
    name.starts_with("safe_read")
        || name.starts_with("safe_write")
        || name.starts_with("report_")
        || name.starts_with("jit_find_cache_entry")
        || name == "coverage_log"
}

#[derive(Eq, PartialEq)]
pub struct WasmLocal(u8);
impl WasmLocal {
    pub fn idx(&self) -> u8 { self.0 }
    /// Unsafe: Can result in multiple free's. Should only be used for locals that are used during
    /// the whole module (for example, registers)
    pub fn unsafe_clone(&self) -> WasmLocal { WasmLocal(self.0) }
}

pub struct WasmLocalI64(u8);
impl WasmLocalI64 {
    pub fn idx(&self) -> u8 { self.0 }
    /// Unsafe: ownership stays with the cache entry.
    pub fn unsafe_clone(&self) -> WasmLocalI64 { WasmLocalI64(self.0) }
}

#[derive(Copy, Clone, Eq, Hash, PartialEq)]
pub struct Label(u32);
impl Label {
    const ZERO: Label = Label(0);
    fn next(&self) -> Label { Label(self.0.wrapping_add(1)) }
}

impl WasmBuilder {
    pub fn new() -> Self {
        let mut b = WasmBuilder {
            output: Vec::with_capacity(256),
            instruction_body: Vec::with_capacity(256),

            idx_import_table_size: 0,
            idx_import_count: 0,
            idx_import_entries: 0,

            import_table_size: 2,
            import_count: 0,
            function_import_count: 0,
            import_indirect_function_table: false,

            initial_static_size: 0,

            label_to_depth: HashMap::new(),
            label_stack: Vec::new(),
            next_label: Label::ZERO,

            free_locals_i32: Vec::with_capacity(8),
            free_locals_i64: Vec::with_capacity(8),
            local_count: 0,
            arg_local_initial_state: WasmLocal(0),
            flag_locals: None,
            flag_dirty: 0,

            branch_hints: Vec::new(),
            branch_hint_mask: 0,
            branch_hint_offset_fuzz: 0,
            function_name: Vec::new(),
        };
        b.init();
        b
    }

    fn init(&mut self) {
        self.output.extend("\0asm".as_bytes());

        // wasm version in leb128, 4 bytes
        self.output.push(op::WASM_VERSION);
        self.output.push(0);
        self.output.push(0);
        self.output.push(0);

        self.write_type_section();
        self.write_import_section_preamble();

        // store state of current pointers etc. so we can reset them later
        self.initial_static_size = self.output.len();
    }

    pub fn reset(&mut self) {
        self.output.drain(self.initial_static_size..);
        self.set_import_table_size(2);
        self.set_import_count(0);
        self.function_import_count = 0;
        self.import_indirect_function_table = false;
        self.instruction_body.clear();
        self.free_locals_i32.clear();
        self.free_locals_i64.clear();
        self.local_count = 0;
        self.flag_locals = None;
        self.flag_dirty = 0;
        self.branch_hints.clear();
        self.function_name.clear();

        dbg_assert!(self.label_to_depth.is_empty());
        dbg_assert!(self.label_stack.is_empty());
        self.next_label = Label::ZERO;
    }

    pub fn finish(&mut self) -> usize {
        dbg_assert!(self.label_to_depth.is_empty());
        dbg_assert!(self.label_stack.is_empty());

        if self.import_indirect_function_table {
            self.write_indirect_function_table_import();
        }
        self.write_memory_import();
        self.write_function_section();
        self.write_export_section();

        dbg_assert!(
            self.local_count as usize == self.free_locals_i32.len() + self.free_locals_i64.len(),
            "All locals should have been freed"
        );

        let free_locals_i32 = &self.free_locals_i32;
        let free_locals_i64 = &self.free_locals_i64;

        let locals = (0..self.local_count).map(|i| {
            let local_index = WASM_MODULE_ARGUMENT_COUNT + i;
            if free_locals_i64.iter().any(|v| v.idx() == local_index) {
                op::TYPE_I64
            }
            else {
                dbg_assert!(free_locals_i32.iter().any(|v| v.idx() == local_index));
                op::TYPE_I32
            }
        });
        let mut groups = vec![];
        for local_type in locals {
            if let Some(last) = groups.last_mut() {
                let (last_type, last_count) = *last;
                if last_type == local_type {
                    *last = (local_type, last_count + 1);
                    continue;
                }
            }
            groups.push((local_type, 1));
        }
        dbg_assert!(groups.len() < 128);

        // Branch hints must precede the code section. Offsets are relative to the first byte
        // of the locals declaration (V8: locals_offset_ is taken before DecodeLocals), which
        // is exactly the group count byte written below — hence the +locals_decl_size rebase.
        self.write_branch_hint_section(1 + 2 * groups.len() as u32);

        // write code section preamble
        self.output.push(op::SC_CODE);

        let idx_code_section_size = self.output.len(); // we will write to this location later
        self.output.push(0);
        self.output.push(0); // write temp val for now using 4 bytes
        self.output.push(0);
        self.output.push(0);

        self.output.push(1); // number of function bodies: just 1

        // same as above but for body size of the function
        let idx_fn_body_size = self.output.len();
        self.output.push(0);
        self.output.push(0);
        self.output.push(0);
        self.output.push(0);

        self.output.push(groups.len().safe_to_u8());
        for (local_type, count) in groups {
            dbg_assert!(count < 128);
            self.output.push(count);
            self.output.push(local_type);
        }

        self.output.append(&mut self.instruction_body);

        self.output.push(op::OP_END);

        // write the actual sizes to the pointer locations stored above. We subtract 4 from the actual
        // value because the ptr itself points to four bytes
        let fn_body_size = (self.output.len() - idx_fn_body_size - 4) as u32;
        write_fixed_leb32_at_idx(&mut self.output, idx_fn_body_size, fn_body_size);

        let code_section_size = (self.output.len() - idx_code_section_size - 4) as u32;
        write_fixed_leb32_at_idx(&mut self.output, idx_code_section_size, code_section_size);

        // Convention (and what V8 expects to find): the name section comes last.
        self.write_name_section();

        self.output.len()
    }

    /// Emit the wasm `name` custom section naming this module's single function body.
    ///
    /// Chrome's CPU sampler renders a wasm frame as `wasm-function[N] @ wasm://wasm/<hash>`,
    /// where N is the index WITHIN the module — a different namespace from v86's global wasm
    /// TABLE index, which is what `bottleship.hotblocks` reports. The two cannot be joined
    /// (measured: 0 of 6660 samples resolved, and 28 table slots were observed under more
    /// than one code offset, so slots are recycled and a numeric coincidence would mean
    /// nothing). A name section closes the gap at the source: Chrome reads it and the frame
    /// self-attributes, which is the only time-proportional guest attribution available —
    /// the embedded EIP sampler fires at yield points and ranks where the guest PARKS.
    ///
    /// Granularity is the MODULE, not the basic block: a JIT module contains exactly one
    /// function body (a br_table over its blocks), and the name section can only name
    /// functions. The name carries the module's primary entry address, which is
    /// authoritative, plus its table index as a cross-check.
    fn write_name_section(&mut self) {
        if self.function_name.is_empty() {
            return;
        }

        // Name subsection 1 = function names: count, then (funcidx, name) pairs. The single
        // defined function follows the imports, so its index is function_import_count.
        let mut names: Vec<u8> = Vec::with_capacity(8 + self.function_name.len());
        write_leb_u32(&mut names, 1); // one named function
        write_leb_u32(&mut names, self.function_import_count as u32);
        write_leb_u32(&mut names, self.function_name.len() as u32);
        names.extend(&self.function_name);

        let mut payload: Vec<u8> = Vec::with_capacity(6 + names.len());
        payload.push(1); // subsection id: function names
        write_leb_u32(&mut payload, names.len() as u32);
        payload.extend(&names);

        let mut section: Vec<u8> =
            Vec::with_capacity(1 + NAME_SECTION_NAME.len() + payload.len());
        write_leb_u32(&mut section, NAME_SECTION_NAME.len() as u32);
        section.extend(NAME_SECTION_NAME.as_bytes());
        section.extend(&payload);

        self.output.push(0); // custom section id
        write_leb_u32(&mut self.output, section.len() as u32);
        self.output.extend(&section);
    }

    /// Emit the "metadata.code.branch_hint" custom section for the single function body.
    /// `locals_decl_size` rebases body-relative offsets onto the spec's zero point.
    fn write_branch_hint_section(&mut self, locals_decl_size: u32) {
        if self.branch_hints.is_empty() {
            return;
        }

        let mut payload: Vec<u8> = Vec::with_capacity(2 + 3 * self.branch_hints.len());
        write_leb_u32(&mut payload, 1); // one function annotated
        write_leb_u32(&mut payload, self.function_import_count as u32);
        write_leb_u32(&mut payload, self.branch_hints.len() as u32);

        let fuzz = self.branch_hint_offset_fuzz;
        let mut last_offset: Option<u32> = None;
        for &(offset, hint) in &self.branch_hints {
            dbg_assert!(hint <= 1);
            dbg_assert!(last_offset.map_or(true, |p| offset > p)); // spec: strictly increasing
            last_offset = Some(offset);
            write_leb_u32(&mut payload, offset + locals_decl_size + fuzz);
            write_leb_u32(&mut payload, 1); // hint payload size, always 1
            payload.push(hint);
        }

        let mut section: Vec<u8> = Vec::with_capacity(1 + BRANCH_HINT_SECTION_NAME.len() + payload.len());
        write_leb_u32(&mut section, BRANCH_HINT_SECTION_NAME.len() as u32);
        section.extend(BRANCH_HINT_SECTION_NAME.as_bytes());
        section.extend(&payload);

        self.output.push(0); // custom section id
        write_leb_u32(&mut self.output, section.len() as u32);
        self.output.extend(&section);
    }

    pub fn write_type_section(&mut self) {
        self.output.push(op::SC_TYPE);

        let idx_section_size = self.output.len();
        self.output.push(0);
        self.output.push(0);

        let nr_of_function_types = FunctionType::to_u8(FunctionType::LAST) + 1;
        dbg_assert!(nr_of_function_types < 128);
        self.output.push(nr_of_function_types);

        for i in 0..(nr_of_function_types) {
            match FunctionType::of_u8(i) {
                FunctionType::FN0 => {
                    self.output.push(op::TYPE_FUNC);
                    self.output.push(0); // no args
                    self.output.push(0); // no return val
                },
                FunctionType::FN1 => {
                    self.output.push(op::TYPE_FUNC);
                    self.output.push(1);
                    self.output.push(op::TYPE_I32);
                    self.output.push(0);
                },
                FunctionType::FN2 => {
                    self.output.push(op::TYPE_FUNC);
                    self.output.push(2);
                    self.output.push(op::TYPE_I32);
                    self.output.push(op::TYPE_I32);
                    self.output.push(0);
                },
                FunctionType::FN3 => {
                    self.output.push(op::TYPE_FUNC);
                    self.output.push(3);
                    self.output.push(op::TYPE_I32);
                    self.output.push(op::TYPE_I32);
                    self.output.push(op::TYPE_I32);
                    self.output.push(0);
                },
                FunctionType::FN0_RET => {
                    self.output.push(op::TYPE_FUNC);
                    self.output.push(0);
                    self.output.push(1);
                    self.output.push(op::TYPE_I32);
                },
                FunctionType::FN0_RET_I64 => {
                    self.output.push(op::TYPE_FUNC);
                    self.output.push(0);
                    self.output.push(1);
                    self.output.push(op::TYPE_I64);
                },
                FunctionType::FN1_RET => {
                    self.output.push(op::TYPE_FUNC);
                    self.output.push(1);
                    self.output.push(op::TYPE_I32);
                    self.output.push(1);
                    self.output.push(op::TYPE_I32);
                },
                FunctionType::FN2_RET => {
                    self.output.push(op::TYPE_FUNC);
                    self.output.push(2);
                    self.output.push(op::TYPE_I32);
                    self.output.push(op::TYPE_I32);
                    self.output.push(1);
                    self.output.push(op::TYPE_I32);
                },
                FunctionType::FN1_RET_I64 => {
                    self.output.push(op::TYPE_FUNC);
                    self.output.push(1);
                    self.output.push(op::TYPE_I32);
                    self.output.push(1);
                    self.output.push(op::TYPE_I64);
                },
                FunctionType::FN1_F32_RET => {
                    self.output.push(op::TYPE_FUNC);
                    self.output.push(1);
                    self.output.push(op::TYPE_F32);
                    self.output.push(1);
                    self.output.push(op::TYPE_I32);
                },
                FunctionType::FN1_F64_RET => {
                    self.output.push(op::TYPE_FUNC);
                    self.output.push(1);
                    self.output.push(op::TYPE_F64);
                    self.output.push(1);
                    self.output.push(op::TYPE_I32);
                },
                FunctionType::FN2_I32_I64 => {
                    self.output.push(op::TYPE_FUNC);
                    self.output.push(2);
                    self.output.push(op::TYPE_I32);
                    self.output.push(op::TYPE_I64);
                    self.output.push(0);
                },
                FunctionType::FN2_I64_I32 => {
                    self.output.push(op::TYPE_FUNC);
                    self.output.push(2);
                    self.output.push(op::TYPE_I64);
                    self.output.push(op::TYPE_I32);
                    self.output.push(0);
                },
                FunctionType::FN2_I64_I32_RET => {
                    self.output.push(op::TYPE_FUNC);
                    self.output.push(2);
                    self.output.push(op::TYPE_I64);
                    self.output.push(op::TYPE_I32);
                    self.output.push(1);
                    self.output.push(op::TYPE_I32);
                },
                FunctionType::FN2_I64_I32_RET_I64 => {
                    self.output.push(op::TYPE_FUNC);
                    self.output.push(2);
                    self.output.push(op::TYPE_I64);
                    self.output.push(op::TYPE_I32);
                    self.output.push(1);
                    self.output.push(op::TYPE_I64);
                },
                FunctionType::FN2_F32_I32 => {
                    self.output.push(op::TYPE_FUNC);
                    self.output.push(2);
                    self.output.push(op::TYPE_F32);
                    self.output.push(op::TYPE_I32);
                    self.output.push(0);
                },
                FunctionType::FN3_RET => {
                    self.output.push(op::TYPE_FUNC);
                    self.output.push(3);
                    self.output.push(op::TYPE_I32);
                    self.output.push(op::TYPE_I32);
                    self.output.push(op::TYPE_I32);
                    self.output.push(1);
                    self.output.push(op::TYPE_I32);
                },
                FunctionType::FN3_I64_I32_I32 => {
                    self.output.push(op::TYPE_FUNC);
                    self.output.push(3);
                    self.output.push(op::TYPE_I64);
                    self.output.push(op::TYPE_I32);
                    self.output.push(op::TYPE_I32);
                    self.output.push(0);
                },
                FunctionType::FN3_I32_I64_I32 => {
                    self.output.push(op::TYPE_FUNC);
                    self.output.push(3);
                    self.output.push(op::TYPE_I32);
                    self.output.push(op::TYPE_I64);
                    self.output.push(op::TYPE_I32);
                    self.output.push(0);
                },
                FunctionType::FN3_I32_I64_I32_RET => {
                    self.output.push(op::TYPE_FUNC);
                    self.output.push(3);
                    self.output.push(op::TYPE_I32);
                    self.output.push(op::TYPE_I64);
                    self.output.push(op::TYPE_I32);
                    self.output.push(1);
                    self.output.push(op::TYPE_I32);
                },
                FunctionType::FN4_I32_I64_I64_I32_RET => {
                    self.output.push(op::TYPE_FUNC);
                    self.output.push(4);
                    self.output.push(op::TYPE_I32);
                    self.output.push(op::TYPE_I64);
                    self.output.push(op::TYPE_I64);
                    self.output.push(op::TYPE_I32);
                    self.output.push(1);
                    self.output.push(op::TYPE_I32);
                },
            }
        }

        let new_len = self.output.len();
        let size = (new_len - 2) - idx_section_size;
        write_fixed_leb16_at_idx(&mut self.output, idx_section_size, size.safe_to_u16());
    }

    /// Goes over the import block to find index of an import entry by function name
    pub fn get_import_index(&self, fn_name: &str) -> Option<u16> {
        let mut offset = self.idx_import_entries;
        for i in 0..self.import_count {
            offset += 1; // skip length of module name
            offset += 1; // skip module name itself
            let len = self.output[offset] as usize;
            offset += 1;
            let name = self
                .output
                .get(offset..(offset + len))
                .expect("get function name");
            if name == fn_name.as_bytes() {
                return Some(i);
            }
            offset += len; // skip the string
            offset += 1; // skip import kind
            offset += 1; // skip type index
        }
        None
    }

    pub fn set_import_count(&mut self, count: u16) {
        dbg_assert!(count < 0x4000);
        self.import_count = count;
        let idx_import_count = self.idx_import_count;
        write_fixed_leb16_at_idx(&mut self.output, idx_import_count, count);
    }

    pub fn set_import_table_size(&mut self, size: usize) {
        dbg_assert!(size < 0x4000);
        self.import_table_size = size;
        let idx_import_table_size = self.idx_import_table_size;
        write_fixed_leb16_at_idx(&mut self.output, idx_import_table_size, size.safe_to_u16());
    }

    pub fn write_import_section_preamble(&mut self) {
        self.output.push(op::SC_IMPORT);

        self.idx_import_table_size = self.output.len();
        self.output.push(1 | 0b10000000);
        self.output.push(2); // 2 in 2 byte leb

        self.idx_import_count = self.output.len();
        self.output.push(1 | 0b10000000);
        self.output.push(0); // 0 in 2 byte leb

        // here after starts the actual list of imports
        self.idx_import_entries = self.output.len();
    }

    pub fn write_memory_import(&mut self) {
        self.output.push(1);
        self.output.push('e' as u8);
        self.output.push(1);
        self.output.push('m' as u8);

        self.output.push(op::EXT_MEMORY);

        self.output.push(0); // memory flag, 0 for no maximum memory limit present
        write_leb_u32(&mut self.output, 64); // initial memory length of 64 pages, takes 1 bytes in leb128

        let new_import_count = self.import_count + 1;
        self.set_import_count(new_import_count);

        let new_table_size = self.import_table_size + 7;
        self.set_import_table_size(new_table_size);
    }

    fn write_import_entry(&mut self, fn_name: &str, type_index: FunctionType) -> u16 {
        let function_index = self.function_import_count;

        self.output.push(1); // length of module name
        self.output.push('e' as u8); // module name
        self.output.push(fn_name.len().safe_to_u8());
        self.output.extend(fn_name.as_bytes());
        self.output.push(op::EXT_FUNCTION);
        self.output.push(type_index.to_u8());

        let new_import_count = self.import_count + 1;
        self.set_import_count(new_import_count);
        self.function_import_count += 1;

        let new_table_size = self.import_table_size + 1 + 1 + 1 + fn_name.len() + 1 + 1;
        self.set_import_table_size(new_table_size);

        function_index
    }

    pub fn write_indirect_function_table_import(&mut self) {
        let fn_name = "__indirect_function_table";

        self.output.push(1); // length of module name
        self.output.push('e' as u8); // module name
        self.output.push(fn_name.len().safe_to_u8());
        self.output.extend(fn_name.as_bytes());
        self.output.push(op::EXT_TABLE);
        self.output.push(op::TYPE_ANYFUNC);
        self.output.push(0); // limits: min only
        write_leb_u32(&mut self.output, 0);

        let new_import_count = self.import_count + 1;
        self.set_import_count(new_import_count);

        let new_table_size = self.import_table_size + 1 + 1 + 1 + fn_name.len() + 1 + 1 + 1 + 1;
        self.set_import_table_size(new_table_size);
    }

    pub fn write_function_section(&mut self) {
        self.output.push(op::SC_FUNCTION);
        self.output.push(2); // length of this section
        self.output.push(1); // count of signature indices
        self.output.push(FunctionType::FN1.to_u8());
    }

    pub fn write_export_section(&mut self) {
        self.output.push(op::SC_EXPORT);
        self.output.push(1 + 1 + 1 + 1 + 2); // size of this section
        self.output.push(1); // count of table: just one function exported

        self.output.push(1); // length of exported function name
        self.output.push('f' as u8); // function name
        self.output.push(op::EXT_FUNCTION);

        // index of the exported function: function index space only counts imported
        // functions, not imported memories/tables.
        let next_op_idx = self.output.len();
        self.output.push(0);
        self.output.push(0); // add 2 bytes for writing 16 byte val
        write_fixed_leb16_at_idx(&mut self.output, next_op_idx, self.function_import_count);
    }

    fn get_fn_idx(&mut self, fn_name: &str, type_index: FunctionType) -> u16 {
        match self.get_import_index(fn_name) {
            Some(idx) => idx,
            None => {
                let idx = self.write_import_entry(fn_name, type_index);
                idx
            },
        }
    }

    pub fn get_output_ptr(&self) -> *const u8 { self.output.as_ptr() }
    pub fn get_output_len(&self) -> u32 { self.output.len() as u32 }

    fn open_block(&mut self) -> Label {
        let label = self.next_label;
        self.next_label = self.next_label.next();
        self.label_to_depth
            .insert(label, self.label_stack.len() + 1);
        self.label_stack.push(label);
        label
    }
    fn close_block(&mut self) {
        let label = self.label_stack.pop().unwrap();
        let old_depth = self.label_to_depth.remove(&label).unwrap();
        dbg_assert!(self.label_to_depth.len() + 1 == old_depth);
    }

    #[must_use = "local allocated but not used"]
    fn alloc_local(&mut self) -> WasmLocal {
        match self.free_locals_i32.pop() {
            Some(local) => local,
            None => {
                let new_idx = self.local_count + WASM_MODULE_ARGUMENT_COUNT;
                self.local_count = self.local_count.checked_add(1).unwrap();
                WasmLocal(new_idx)
            },
        }
    }
    pub fn free_local(&mut self, local: WasmLocal) {
        dbg_assert!(
            (WASM_MODULE_ARGUMENT_COUNT..self.local_count + WASM_MODULE_ARGUMENT_COUNT)
                .contains(&local.0)
        );
        self.free_locals_i32.push(local)
    }

    #[must_use = "local allocated but not used"]
    pub fn set_new_local(&mut self) -> WasmLocal {
        let local = self.alloc_local();
        self.instruction_body.push(op::OP_SETLOCAL);
        self.instruction_body.push(local.idx());
        local
    }
    #[must_use = "local allocated but not used"]
    pub fn tee_new_local(&mut self) -> WasmLocal {
        let local = self.alloc_local();
        self.instruction_body.push(op::OP_TEELOCAL);
        self.instruction_body.push(local.idx());
        local
    }
    pub fn set_local(&mut self, local: &WasmLocal) {
        self.instruction_body.push(op::OP_SETLOCAL);
        self.instruction_body.push(local.idx());
    }
    pub fn tee_local(&mut self, local: &WasmLocal) {
        self.instruction_body.push(op::OP_TEELOCAL);
        self.instruction_body.push(local.idx());
    }
    pub fn get_local(&mut self, local: &WasmLocal) {
        self.instruction_body.push(op::OP_GETLOCAL);
        self.instruction_body.push(local.idx());
    }

    #[must_use = "local allocated but not used"]
    fn alloc_local_i64(&mut self) -> WasmLocalI64 {
        match self.free_locals_i64.pop() {
            Some(local) => local,
            None => {
                let new_idx = self.local_count + WASM_MODULE_ARGUMENT_COUNT;
                self.local_count += 1;
                WasmLocalI64(new_idx)
            },
        }
    }
    pub fn free_local_i64(&mut self, local: WasmLocalI64) {
        dbg_assert!(
            (WASM_MODULE_ARGUMENT_COUNT..self.local_count + WASM_MODULE_ARGUMENT_COUNT)
                .contains(&local.0)
        );
        self.free_locals_i64.push(local)
    }
    #[must_use = "local allocated but not used"]
    pub fn set_new_local_i64(&mut self) -> WasmLocalI64 {
        let local = self.alloc_local_i64();
        self.instruction_body.push(op::OP_SETLOCAL);
        self.instruction_body.push(local.idx());
        local
    }
    #[must_use = "local allocated but not used"]
    pub fn tee_new_local_i64(&mut self) -> WasmLocalI64 {
        let local = self.alloc_local_i64();
        self.instruction_body.push(op::OP_TEELOCAL);
        self.instruction_body.push(local.idx());
        local
    }
    pub fn get_local_i64(&mut self, local: &WasmLocalI64) {
        self.instruction_body.push(op::OP_GETLOCAL);
        self.instruction_body.push(local.idx());
    }
    pub fn set_local_i64(&mut self, local: &WasmLocalI64) {
        self.instruction_body.push(op::OP_SETLOCAL);
        self.instruction_body.push(local.idx());
    }
    pub fn tee_local_i64(&mut self, local: &WasmLocalI64) {
        self.instruction_body.push(op::OP_TEELOCAL);
        self.instruction_body.push(local.idx());
    }

    pub fn const_i32(&mut self, v: i32) {
        self.instruction_body.push(op::OP_I32CONST);
        write_leb_i32(&mut self.instruction_body, v);
    }
    pub fn const_i64(&mut self, v: i64) {
        self.instruction_body.push(op::OP_I64CONST);
        write_leb_i64(&mut self.instruction_body, v);
    }
    pub fn const_f64(&mut self, v: f64) {
        self.instruction_body.push(op::OP_F64CONST);
        self.instruction_body.extend(v.to_le_bytes());
    }

    pub fn load_fixed_u8(&mut self, addr: u32) {
        self.const_i32(addr as i32);
        self.load_u8(0);
    }
    pub fn load_fixed_u16(&mut self, addr: u32) {
        // doesn't cause a failure in the generated code, but it will be much slower
        dbg_assert!((addr & 1) == 0);

        self.const_i32(addr as i32);
        self.instruction_body.push(op::OP_I32LOAD16U);
        self.instruction_body.push(op::MEM_ALIGN16);
        self.instruction_body.push(0); // immediate offset
    }
    pub fn load_fixed_i32(&mut self, addr: u32) {
        // doesn't cause a failure in the generated code, but it will be much slower
        dbg_assert!((addr & 3) == 0);

        self.const_i32(addr as i32);
        self.load_aligned_i32(0);
    }
    pub fn load_fixed_i64(&mut self, addr: u32) {
        // doesn't cause a failure in the generated code, but it will be much slower
        dbg_assert!((addr & 7) == 0);

        self.const_i32(addr as i32);
        self.load_aligned_i64(0);
    }

    pub fn load_u8(&mut self, byte_offset: u32) {
        self.instruction_body.push(op::OP_I32LOAD8U);
        self.instruction_body.push(op::MEM_NO_ALIGN);
        write_leb_u32(&mut self.instruction_body, byte_offset);
    }

    pub fn load_unaligned_i64(&mut self, byte_offset: u32) {
        self.instruction_body.push(op::OP_I64LOAD);
        self.instruction_body.push(op::MEM_NO_ALIGN);
        write_leb_u32(&mut self.instruction_body, byte_offset);
    }

    pub fn load_unaligned_i32(&mut self, byte_offset: u32) {
        self.instruction_body.push(op::OP_I32LOAD);
        self.instruction_body.push(op::MEM_NO_ALIGN);
        write_leb_u32(&mut self.instruction_body, byte_offset);
    }

    pub fn load_unaligned_u16(&mut self, byte_offset: u32) {
        self.instruction_body.push(op::OP_I32LOAD16U);
        self.instruction_body.push(op::MEM_NO_ALIGN);
        write_leb_u32(&mut self.instruction_body, byte_offset);
    }

    pub fn load_aligned_f64(&mut self, byte_offset: u32) {
        self.instruction_body.push(op::OP_F64LOAD);
        self.instruction_body.push(op::MEM_ALIGN64);
        write_leb_u32(&mut self.instruction_body, byte_offset);
    }

    pub fn load_aligned_i64(&mut self, byte_offset: u32) {
        self.instruction_body.push(op::OP_I64LOAD);
        self.instruction_body.push(op::MEM_ALIGN64);
        write_leb_u32(&mut self.instruction_body, byte_offset);
    }

    pub fn load_aligned_f32(&mut self, byte_offset: u32) {
        self.instruction_body.push(op::OP_F32LOAD);
        self.instruction_body.push(op::MEM_ALIGN32);
        write_leb_u32(&mut self.instruction_body, byte_offset);
    }

    pub fn load_aligned_i32(&mut self, byte_offset: u32) {
        self.instruction_body.push(op::OP_I32LOAD);
        self.instruction_body.push(op::MEM_ALIGN32);
        write_leb_u32(&mut self.instruction_body, byte_offset);
    }

    pub fn load_aligned_u16(&mut self, byte_offset: u32) {
        self.instruction_body.push(op::OP_I32LOAD16U);
        self.instruction_body.push(op::MEM_ALIGN16);
        write_leb_u32(&mut self.instruction_body, byte_offset);
    }

    pub fn store_u8(&mut self, byte_offset: u32) {
        self.instruction_body.push(op::OP_I32STORE8);
        self.instruction_body.push(op::MEM_NO_ALIGN);
        write_leb_u32(&mut self.instruction_body, byte_offset);
    }

    pub fn store_aligned_u16(&mut self, byte_offset: u32) {
        self.instruction_body.push(op::OP_I32STORE16);
        self.instruction_body.push(op::MEM_ALIGN16);
        write_leb_u32(&mut self.instruction_body, byte_offset);
    }

    pub fn store_aligned_i32(&mut self, byte_offset: u32) {
        self.instruction_body.push(op::OP_I32STORE);
        self.instruction_body.push(op::MEM_ALIGN32);
        write_leb_u32(&mut self.instruction_body, byte_offset);
    }

    pub fn store_aligned_i64(&mut self, byte_offset: u32) {
        self.instruction_body.push(op::OP_I64STORE);
        self.instruction_body.push(op::MEM_ALIGN64);
        write_leb_u32(&mut self.instruction_body, byte_offset);
    }

    pub fn store_unaligned_u16(&mut self, byte_offset: u32) {
        self.instruction_body.push(op::OP_I32STORE16);
        self.instruction_body.push(op::MEM_NO_ALIGN);
        write_leb_u32(&mut self.instruction_body, byte_offset);
    }

    pub fn store_unaligned_i32(&mut self, byte_offset: u32) {
        self.instruction_body.push(op::OP_I32STORE);
        self.instruction_body.push(op::MEM_NO_ALIGN);
        write_leb_u32(&mut self.instruction_body, byte_offset);
    }

    pub fn store_unaligned_i64(&mut self, byte_offset: u32) {
        self.instruction_body.push(op::OP_I64STORE);
        self.instruction_body.push(op::MEM_NO_ALIGN);
        write_leb_u32(&mut self.instruction_body, byte_offset);
    }

    // ─── v128 / WASM SIMD ───────────────────────────────────────────────
    // All SIMD ops are 2-byte: 0xfd prefix + LEB128 sub-opcode.
    // Memory ops use memarg (align + offset) the same way scalar ops do.
    pub fn load_aligned_v128(&mut self, byte_offset: u32) {
        self.instruction_body.push(op::OP_SIMD_PREFIX);
        self.instruction_body.push(op::SIMD_V128_LOAD);
        self.instruction_body.push(op::MEM_ALIGN128);
        write_leb_u32(&mut self.instruction_body, byte_offset);
    }
    pub fn load_unaligned_v128(&mut self, byte_offset: u32) {
        self.instruction_body.push(op::OP_SIMD_PREFIX);
        self.instruction_body.push(op::SIMD_V128_LOAD);
        self.instruction_body.push(op::MEM_NO_ALIGN);
        write_leb_u32(&mut self.instruction_body, byte_offset);
    }
    pub fn store_aligned_v128(&mut self, byte_offset: u32) {
        self.instruction_body.push(op::OP_SIMD_PREFIX);
        self.instruction_body.push(op::SIMD_V128_STORE);
        self.instruction_body.push(op::MEM_ALIGN128);
        write_leb_u32(&mut self.instruction_body, byte_offset);
    }
    pub fn store_unaligned_v128(&mut self, byte_offset: u32) {
        self.instruction_body.push(op::OP_SIMD_PREFIX);
        self.instruction_body.push(op::SIMD_V128_STORE);
        self.instruction_body.push(op::MEM_NO_ALIGN);
        write_leb_u32(&mut self.instruction_body, byte_offset);
    }
    pub fn and_v128(&mut self) {
        self.instruction_body.push(op::OP_SIMD_PREFIX);
        self.instruction_body.push(op::SIMD_V128_AND);
    }
    pub fn or_v128(&mut self) {
        self.instruction_body.push(op::OP_SIMD_PREFIX);
        self.instruction_body.push(op::SIMD_V128_OR);
    }
    pub fn xor_v128(&mut self) {
        self.instruction_body.push(op::OP_SIMD_PREFIX);
        self.instruction_body.push(op::SIMD_V128_XOR);
    }
    // i64x2 shifts — shift count pushed as i32 before op.
    pub fn shl_i64x2(&mut self) {
        self.instruction_body.push(op::OP_SIMD_PREFIX);
        self.instruction_body.push(op::SIMD_I64X2_SHL_LEB0);
        self.instruction_body.push(op::SIMD_I64X2_SHL_LEB1);
    }
    pub fn shr_s_i64x2(&mut self) {
        self.instruction_body.push(op::OP_SIMD_PREFIX);
        self.instruction_body.push(op::SIMD_I64X2_SHR_S_LEB0);
        self.instruction_body.push(op::SIMD_I64X2_SHR_S_LEB1);
    }
    pub fn shr_u_i64x2(&mut self) {
        self.instruction_body.push(op::OP_SIMD_PREFIX);
        self.instruction_body.push(op::SIMD_I64X2_SHR_U_LEB0);
        self.instruction_body.push(op::SIMD_I64X2_SHR_U_LEB1);
    }
    pub fn add_i64x2(&mut self) {
        self.instruction_body.push(op::OP_SIMD_PREFIX);
        self.instruction_body.push(op::SIMD_I64X2_ADD_LEB0);
        self.instruction_body.push(op::SIMD_I64X2_ADD_LEB1);
    }
    pub fn sub_i64x2(&mut self) {
        self.instruction_body.push(op::OP_SIMD_PREFIX);
        self.instruction_body.push(op::SIMD_I64X2_SUB_LEB0);
        self.instruction_body.push(op::SIMD_I64X2_SUB_LEB1);
    }

    /// Emit `0xfd` + the LEB128 encoding of a SIMD sub-opcode. Sub-opcodes are a
    /// u32 immediate, so anything >= 0x80 is two bytes; encoding here keeps the
    /// call sites naming one raw value from the spec table instead of a
    /// hand-split byte pair.
    fn simd_op(&mut self, sub_opcode: u8) {
        self.instruction_body.push(op::OP_SIMD_PREFIX);
        write_leb_u32(&mut self.instruction_body, sub_opcode as u32);
    }

    pub fn andnot_v128(&mut self) { self.simd_op(op::SIMD_V128_ANDNOT); }
    pub fn add_sat_s_i16x8(&mut self) { self.simd_op(op::SIMD_I16X8_ADD_SAT_S); }
    pub fn mul_i16x8(&mut self) { self.simd_op(op::SIMD_I16X8_MUL); }
    pub fn sub_sat_u_i8x16(&mut self) { self.simd_op(op::SIMD_I8X16_SUB_SAT_U); }

    /// `i8x16.shuffle` — the 16 lane indices are an immediate, not stack
    /// operands. Index 0..15 selects from the first vector, 16..31 the second.
    pub fn shuffle_i8x16(&mut self, lanes: &[u8; 16]) {
        self.simd_op(op::SIMD_I8X16_SHUFFLE);
        for &l in lanes {
            dbg_assert!(l < 32);
            self.instruction_body.push(l);
        }
    }

    // ─── Scalar float arithmetic (for SSE2 scalar ops) ──────────────────
    // MULSD/ADDSD/SUBSD/DIVSD write ONLY the low 64 bits of the XMM — the
    // upper 64 bits must be preserved. In WASM the cleanest way is to do
    // the op as SCALAR f64 (not SIMD) and store back just the low 8 bytes
    // via f64.store, leaving the upper 8 bytes physically untouched.
    pub fn add_f64(&mut self) { self.instruction_body.push(op::OP_F64ADD); }
    pub fn sub_f64(&mut self) { self.instruction_body.push(op::OP_F64SUB); }
    pub fn mul_f64(&mut self) { self.instruction_body.push(op::OP_F64MUL); }
    pub fn div_f64(&mut self) { self.instruction_body.push(op::OP_F64DIV); }
    pub fn abs_f64(&mut self) { self.instruction_body.push(op::OP_F64ABS); }
    pub fn neg_f64(&mut self) { self.instruction_body.push(op::OP_F64NEG); }
    pub fn ceil_f64(&mut self) { self.instruction_body.push(op::OP_F64CEIL); }
    pub fn floor_f64(&mut self) { self.instruction_body.push(op::OP_F64FLOOR); }
    pub fn trunc_f64(&mut self) { self.instruction_body.push(op::OP_F64TRUNC); }
    pub fn nearest_f64(&mut self) { self.instruction_body.push(op::OP_F64NEAREST); }
    pub fn add_f32(&mut self) { self.instruction_body.push(op::OP_F32ADD); }
    pub fn sub_f32(&mut self) { self.instruction_body.push(op::OP_F32SUB); }
    pub fn mul_f32(&mut self) { self.instruction_body.push(op::OP_F32MUL); }
    pub fn div_f32(&mut self) { self.instruction_body.push(op::OP_F32DIV); }
    pub fn store_aligned_f64(&mut self, byte_offset: u32) {
        self.instruction_body.push(op::OP_F64STORE);
        self.instruction_body.push(op::MEM_ALIGN64);
        write_leb_u32(&mut self.instruction_body, byte_offset);
    }
    pub fn store_aligned_f32(&mut self, byte_offset: u32) {
        self.instruction_body.push(op::OP_F32STORE);
        self.instruction_body.push(op::MEM_ALIGN32);
        write_leb_u32(&mut self.instruction_body, byte_offset);
    }

    pub fn increment_fixed_i64(&mut self, byte_offset: u32, n: i64) {
        self.const_i32(byte_offset as i32);
        self.load_fixed_i64(byte_offset);
        self.const_i64(n);
        self.add_i64();
        self.store_aligned_i64(0);
    }

    pub fn add_i32(&mut self) { self.instruction_body.push(op::OP_I32ADD); }
    pub fn add_i64(&mut self) { self.instruction_body.push(op::OP_I64ADD); }
    pub fn sub_i32(&mut self) { self.instruction_body.push(op::OP_I32SUB); }
    pub fn and_i32(&mut self) { self.instruction_body.push(op::OP_I32AND); }
    pub fn or_i32(&mut self) { self.instruction_body.push(op::OP_I32OR); }
    pub fn or_i64(&mut self) { self.instruction_body.push(op::OP_I64OR); }
    pub fn xor_i32(&mut self) { self.instruction_body.push(op::OP_I32XOR); }
    pub fn mul_i32(&mut self) { self.instruction_body.push(op::OP_I32MUL); }
    pub fn mul_i64(&mut self) { self.instruction_body.push(op::OP_I64MUL); }
    pub fn div_i64(&mut self) { self.instruction_body.push(op::OP_I64DIVU); }
    pub fn rem_i64(&mut self) { self.instruction_body.push(op::OP_I64REMU); }

    pub fn rotl_i32(&mut self) { self.instruction_body.push(op::OP_I32ROTL); }

    pub fn shl_i32(&mut self) { self.instruction_body.push(op::OP_I32SHL); }
    pub fn shl_i64(&mut self) { self.instruction_body.push(op::OP_I64SHL); }
    pub fn shr_u_i32(&mut self) { self.instruction_body.push(op::OP_I32SHRU); }
    pub fn shr_u_i64(&mut self) { self.instruction_body.push(op::OP_I64SHRU); }
    pub fn shr_s_i32(&mut self) { self.instruction_body.push(op::OP_I32SHRS); }

    pub fn eq_i32(&mut self) { self.instruction_body.push(op::OP_I32EQ); }
    pub fn eq_i64(&mut self) { self.instruction_body.push(op::OP_I64EQ); }
    pub fn ne_i32(&mut self) { self.instruction_body.push(op::OP_I32NE); }
    pub fn ne_i64(&mut self) { self.instruction_body.push(op::OP_I64NE); }

    pub fn le_i32(&mut self) { self.instruction_body.push(op::OP_I32LES); }
    pub fn lt_i32(&mut self) { self.instruction_body.push(op::OP_I32LTS); }
    pub fn ge_i32(&mut self) { self.instruction_body.push(op::OP_I32GES); }
    pub fn gt_i32(&mut self) { self.instruction_body.push(op::OP_I32GTS); }

    pub fn gtu_i32(&mut self) { self.instruction_body.push(op::OP_I32GTU); }
    pub fn geu_i32(&mut self) { self.instruction_body.push(op::OP_I32GEU); }
    pub fn ltu_i32(&mut self) { self.instruction_body.push(op::OP_I32LTU); }
    pub fn leu_i32(&mut self) { self.instruction_body.push(op::OP_I32LEU); }

    pub fn gtu_i64(&mut self) { self.instruction_body.push(op::OP_I64GTU); }

    pub fn reinterpret_i32_as_f32(&mut self) {
        self.instruction_body.push(op::OP_F32REINTERPRETI32);
    }
    pub fn reinterpret_f32_as_i32(&mut self) {
        self.instruction_body.push(op::OP_I32REINTERPRETF32);
    }
    pub fn reinterpret_i64_as_f64(&mut self) {
        self.instruction_body.push(op::OP_F64REINTERPRETI64);
    }
    pub fn reinterpret_f64_as_i64(&mut self) {
        self.instruction_body.push(op::OP_I64REINTERPRETF64);
    }
    pub fn promote_f32_to_f64(&mut self) { self.instruction_body.push(op::OP_F64PROMOTEF32); }
    pub fn demote_f64_to_f32(&mut self) { self.instruction_body.push(op::OP_F32DEMOTEF64); }
    pub fn convert_i32_to_f64(&mut self) { self.instruction_body.push(op::OP_F64CONVERTSI32); }
    pub fn convert_i64_to_f64(&mut self) { self.instruction_body.push(op::OP_F64CONVERTSI64); }
    pub fn trunc_s_f64_to_i32(&mut self) { self.instruction_body.push(op::OP_I32TRUNCSF64); }
    pub fn extend_unsigned_i32_to_i64(&mut self) {
        self.instruction_body.push(op::OP_I64EXTENDUI32);
    }
    pub fn extend_signed_i32_to_i64(&mut self) { self.instruction_body.push(op::OP_I64EXTENDSI32); }
    pub fn wrap_i64_to_i32(&mut self) { self.instruction_body.push(op::OP_I32WRAPI64); }

    pub fn eqz_i32(&mut self) { self.instruction_body.push(op::OP_I32EQZ); }
    pub fn eq_f64(&mut self) { self.instruction_body.push(op::OP_F64EQ); }
    pub fn ne_f64(&mut self) { self.instruction_body.push(op::OP_F64NE); }
    pub fn lt_f64(&mut self) { self.instruction_body.push(op::OP_F64LT); }
    pub fn ge_f64(&mut self) { self.instruction_body.push(op::OP_F64GE); }

    pub fn select(&mut self) { self.instruction_body.push(op::OP_SELECT); }

    pub fn if_i32(&mut self) {
        self.flag_boundary();
        self.open_block();
        self.instruction_body.push(op::OP_IF);
        self.instruction_body.push(op::TYPE_I32);
    }
    #[allow(dead_code)]
    pub fn if_i64(&mut self) {
        self.flag_boundary();
        self.open_block();
        self.instruction_body.push(op::OP_IF);
        self.instruction_body.push(op::TYPE_I64);
    }
    pub fn block_i32(&mut self) -> Label {
        self.flag_boundary();
        self.instruction_body.push(op::OP_BLOCK);
        self.instruction_body.push(op::TYPE_I32);
        self.open_block()
    }

    pub fn if_void(&mut self) {
        self.flag_boundary();
        self.open_block();
        self.instruction_body.push(op::OP_IF);
        self.instruction_body.push(op::TYPE_VOID_BLOCK);
    }

    pub fn else_(&mut self) {
        self.flag_boundary();
        dbg_assert!(!self.label_stack.is_empty());
        self.instruction_body.push(op::OP_ELSE);
    }

    pub fn loop_void(&mut self) -> Label {
        self.flag_boundary();
        self.instruction_body.push(op::OP_LOOP);
        self.instruction_body.push(op::TYPE_VOID_BLOCK);
        self.open_block()
    }

    pub fn block_void(&mut self) -> Label {
        self.flag_boundary();
        self.instruction_body.push(op::OP_BLOCK);
        self.instruction_body.push(op::TYPE_VOID_BLOCK);
        self.open_block()
    }

    pub fn block_end(&mut self) {
        self.flag_boundary();
        self.close_block();
        self.instruction_body.push(op::OP_END);
    }

    pub fn return_(&mut self) {
        self.flag_boundary();
        self.instruction_body.push(op::OP_RETURN);
    }

    #[allow(dead_code)]
    pub fn drop_(&mut self) { self.instruction_body.push(op::OP_DROP); }

    pub fn brtable(
        &mut self,
        default_case: Label,
        cases: &mut dyn std::iter::ExactSizeIterator<Item = &Label>,
    ) {
        self.flag_boundary();
        self.instruction_body.push(op::OP_BRTABLE);
        write_leb_u32(&mut self.instruction_body, cases.len() as u32);
        for case in cases {
            self.write_label(*case);
        }
        self.write_label(default_case);
    }

    pub fn br(&mut self, label: Label) {
        self.flag_boundary();
        self.instruction_body.push(op::OP_BR);
        self.write_label(label);
    }
    pub fn br_if(&mut self, label: Label) {
        self.flag_boundary();
        self.instruction_body.push(op::OP_BRIF);
        self.write_label(label);
    }

    /// Record a hint for the instruction that is about to be pushed. Must be called
    /// immediately before the opcode byte: the spec addresses the `if`/`br_if` opcode itself.
    fn note_branch_hint(&mut self, group: u32, hint: u8) {
        if self.branch_hint_mask & group != 0 {
            self.branch_hints
                .push((self.instruction_body.len() as u32, hint));
        }
    }

    pub fn br_if_hinted(&mut self, label: Label, group: u32, hint: u8) {
        self.note_branch_hint(group, hint);
        self.br_if(label);
    }
    pub fn if_void_hinted(&mut self, group: u32, hint: u8) {
        // open_block() only touches label bookkeeping, not instruction_body, so the
        // recorded offset still lands on OP_IF.
        self.note_branch_hint(group, hint);
        self.if_void();
    }
    pub fn if_i32_hinted(&mut self, group: u32, hint: u8) {
        self.note_branch_hint(group, hint);
        self.if_i32();
    }
    #[allow(dead_code)]
    pub fn if_i64_hinted(&mut self, group: u32, hint: u8) {
        self.note_branch_hint(group, hint);
        self.if_i64();
    }

    fn write_label(&mut self, label: Label) {
        let depth = *self.label_to_depth.get(&label).unwrap();
        dbg_assert!(depth <= self.label_stack.len());
        write_leb_u32(
            &mut self.instruction_body,
            (self.label_stack.len() - depth) as u32,
        );
    }

    fn call_fn(&mut self, name: &str, function: FunctionType) {
        // Flag-locals funnel: materialize the lazy-flag tuple to its
        // memory globals before any helper that may read them, and re-read after
        // any helper that may have written them. Emitting stores/loads here is
        // stack-safe even with the call's arguments already pushed: each store
        // pops exactly the address/value pair it pushes.
        let spill = self.flag_locals.is_some() && !flag_spill_whitelisted(name);
        if spill {
            self.emit_flag_spill();
        }
        let i = self.get_fn_idx(name, function);
        self.instruction_body.push(op::OP_CALL);
        write_leb_u32(&mut self.instruction_body, i as u32);
        if spill {
            self.emit_flag_reload();
        }
    }

    /// Locals → memory globals (call sites + module epilogues), for the words that can differ.
    pub fn emit_flag_spill(&mut self) {
        if let Some(locals) = self.flag_locals {
            for (slot, (idx, addr)) in locals.into_iter().enumerate() {
                if self.flag_dirty & (1 << slot) == 0 {
                    continue;
                }
                self.const_i32(addr as i32);
                self.instruction_body.push(op::OP_GETLOCAL);
                self.instruction_body.push(idx);
                self.store_aligned_i32(0);
            }
            self.flag_dirty = 0;
        }
    }

    /// Memory globals → locals (after helpers that write flags, e.g. update_eflags).
    pub fn emit_flag_reload(&mut self) {
        if let Some(locals) = self.flag_locals {
            for (idx, addr) in locals {
                self.load_fixed_i32(addr);
                self.instruction_body.push(op::OP_SETLOCAL);
                self.instruction_body.push(idx);
            }
            self.flag_dirty = 0;
        }
    }

    /// A point another path can reach. Nothing is known about what it left in the locals, so every
    /// word is assumed to need storing again.
    fn flag_boundary(&mut self) {
        if self.flag_locals.is_some() {
            self.flag_dirty = 0x1f;
        }
    }

    /// Flag-local slot accessors (tuple order fixed at registration:
    /// 0=last_op1 1=last_result 2=last_op_size 3=flags_changed 4=flags).
    /// Return false when flag locals are not registered — caller falls back to
    /// the linear-memory global. Value semantics identical.
    pub fn flag_local_get(&mut self, slot: usize) -> bool {
        match self.flag_locals {
            Some(locals) => {
                self.get_local_raw(locals[slot].0);
                true
            },
            None => false,
        }
    }
    pub fn flag_local_set(&mut self, slot: usize) -> bool {
        match self.flag_locals {
            Some(locals) => {
                self.set_local_raw(locals[slot].0);
                self.flag_dirty |= 1 << slot;
                true
            },
            None => false,
        }
    }

    /// Drop-in replacement for `store_aligned_i32(0)` at flag-global write sites
    /// whose address const is already emitted: stack is [addr, value]. In locals
    /// mode SETLOCAL pops the value, DROP pops the now-dead address — same stack
    /// effect as the store, so call sites swap a single line.
    pub fn flag_store_i32(&mut self, slot: usize) {
        match self.flag_locals {
            Some(locals) => {
                self.set_local_raw(locals[slot].0);
                self.drop_();
                self.flag_dirty |= 1 << slot;
            },
            None => self.store_aligned_i32(0),
        }
    }

    /// Drop-in replacements for load_fixed_{u8,u16,i32}(flag global) read sites.
    /// `fallback_addr` = the global's linear-memory address (used when locals off).
    pub fn flag_load_u8(&mut self, slot: usize, fallback_addr: u32) {
        match self.flag_locals {
            Some(locals) => {
                self.get_local_raw(locals[slot].0);
                self.const_i32(0xFF);
                self.and_i32();
            },
            None => self.load_fixed_u8(fallback_addr),
        }
    }
    pub fn flag_load_u16(&mut self, slot: usize, fallback_addr: u32) {
        match self.flag_locals {
            Some(locals) => {
                self.get_local_raw(locals[slot].0);
                self.const_i32(0xFFFF);
                self.and_i32();
            },
            None => self.load_fixed_u16(fallback_addr),
        }
    }
    pub fn flag_load_i32(&mut self, slot: usize, fallback_addr: u32) {
        match self.flag_locals {
            Some(locals) => self.get_local_raw(locals[slot].0),
            None => self.load_fixed_i32(fallback_addr),
        }
    }

    /// Unregister + return the flag locals to the free pool (module epilogue,
    /// after the final spill — satisfies the all-locals-freed finish invariant).
    pub fn free_flag_locals(&mut self) {
        if let Some(locals) = self.flag_locals.take() {
            for (idx, _) in locals {
                self.free_local(WasmLocal(idx));
            }
        }
    }

    /// Raw local access for the registered flag locals (indices held as u8, not
    /// WasmLocal, because they live for the whole module like register locals).
    pub fn get_local_raw(&mut self, idx: u8) {
        self.instruction_body.push(op::OP_GETLOCAL);
        self.instruction_body.push(idx);
    }
    pub fn set_local_raw(&mut self, idx: u8) {
        self.instruction_body.push(op::OP_SETLOCAL);
        self.instruction_body.push(idx);
    }

    pub fn return_call_indirect_fn1(&mut self) {
        self.import_indirect_function_table = true;
        self.instruction_body.push(op::OP_RETURN_CALL_INDIRECT);
        self.instruction_body.push(FunctionType::FN1.to_u8());
        self.instruction_body.push(0); // table index
    }

    pub fn call_fn0(&mut self, name: &str) { self.call_fn(name, FunctionType::FN0) }
    pub fn call_fn0_ret(&mut self, name: &str) { self.call_fn(name, FunctionType::FN0_RET) }
    pub fn call_fn0_ret_i64(&mut self, name: &str) { self.call_fn(name, FunctionType::FN0_RET_I64) }
    pub fn call_fn1(&mut self, name: &str) { self.call_fn(name, FunctionType::FN1) }
    pub fn call_fn1_ret(&mut self, name: &str) { self.call_fn(name, FunctionType::FN1_RET) }
    pub fn call_fn1_ret_i64(&mut self, name: &str) { self.call_fn(name, FunctionType::FN1_RET_I64) }
    pub fn call_fn1_f32_ret(&mut self, name: &str) { self.call_fn(name, FunctionType::FN1_F32_RET) }
    pub fn call_fn1_f64_ret(&mut self, name: &str) { self.call_fn(name, FunctionType::FN1_F64_RET) }
    pub fn call_fn2(&mut self, name: &str) { self.call_fn(name, FunctionType::FN2) }
    pub fn call_fn2_i32_i64(&mut self, name: &str) { self.call_fn(name, FunctionType::FN2_I32_I64) }
    pub fn call_fn2_i64_i32(&mut self, name: &str) { self.call_fn(name, FunctionType::FN2_I64_I32) }
    pub fn call_fn2_i64_i32_ret(&mut self, name: &str) {
        self.call_fn(name, FunctionType::FN2_I64_I32_RET)
    }
    pub fn call_fn2_i64_i32_ret_i64(&mut self, name: &str) {
        self.call_fn(name, FunctionType::FN2_I64_I32_RET_I64)
    }
    pub fn call_fn2_f32_i32(&mut self, name: &str) { self.call_fn(name, FunctionType::FN2_F32_I32) }
    pub fn call_fn2_ret(&mut self, name: &str) { self.call_fn(name, FunctionType::FN2_RET) }
    pub fn call_fn3(&mut self, name: &str) { self.call_fn(name, FunctionType::FN3) }
    pub fn call_fn3_ret(&mut self, name: &str) { self.call_fn(name, FunctionType::FN3_RET) }
    pub fn call_fn3_i64_i32_i32(&mut self, name: &str) {
        self.call_fn(name, FunctionType::FN3_I64_I32_I32)
    }
    pub fn call_fn3_i32_i64_i32(&mut self, name: &str) {
        self.call_fn(name, FunctionType::FN3_I32_I64_I32)
    }
    pub fn call_fn3_i32_i64_i32_ret(&mut self, name: &str) {
        self.call_fn(name, FunctionType::FN3_I32_I64_I32_RET)
    }
    pub fn call_fn4_i32_i64_i64_i32_ret(&mut self, name: &str) {
        self.call_fn(name, FunctionType::FN4_I32_I64_I64_I32_RET)
    }

    pub fn unreachable(&mut self) { self.instruction_body.push(op::OP_UNREACHABLE) }

    pub fn instruction_body_length(&self) -> u32 { self.instruction_body.len() as u32 }
}

#[cfg(test)]
mod tests {
    use super::{FunctionType, WasmBuilder, WASM_MODULE_ARGUMENT_COUNT};
    use std::fs::File;
    use std::io::Write;

    #[test]
    fn import_table_management() {
        let mut w = WasmBuilder::new();

        assert_eq!(0, w.get_fn_idx("foo", FunctionType::FN0));
        assert_eq!(1, w.get_fn_idx("bar", FunctionType::FN1));
        assert_eq!(0, w.get_fn_idx("foo", FunctionType::FN0));
        assert_eq!(2, w.get_fn_idx("baz", FunctionType::FN2));
    }

    #[test]
    fn builder_test() {
        let mut m = WasmBuilder::new();

        m.call_fn("foo", FunctionType::FN0);
        m.call_fn("bar", FunctionType::FN0);

        let local0 = m.alloc_local(); // for ensuring that reset clears previous locals
        m.free_local(local0);

        m.finish();
        m.reset();

        m.const_i32(2);

        m.call_fn("baz", FunctionType::FN1_RET);
        m.call_fn("foo", FunctionType::FN1);

        m.const_i32(10);
        let local1 = m.alloc_local();
        m.tee_local(&local1); // local1 = 10

        m.const_i32(20);
        m.add_i32();
        let local2 = m.alloc_local();
        m.tee_local(&local2); // local2 = 30

        m.free_local(local1);

        let local3 = m.alloc_local();
        assert_eq!(local3.idx(), WASM_MODULE_ARGUMENT_COUNT);

        m.free_local(local2);
        m.free_local(local3);

        m.const_i32(30);
        m.ne_i32();
        m.if_void();
        m.unreachable();
        m.block_end();

        m.finish();

        let op_ptr = m.get_output_ptr();
        let op_len = m.get_output_len();
        dbg_log!("op_ptr: {:?}, op_len: {:?}", op_ptr, op_len);

        let mut f = File::create("build/dummy_output.wasm").expect("creating dummy_output.wasm");
        f.write_all(&m.output).expect("write dummy_output.wasm");
    }
}
