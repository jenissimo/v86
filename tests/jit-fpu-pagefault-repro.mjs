#!/usr/bin/env node
// #PF DURING custom-jitted FPU/BCD ops: fnstenv/fldenv (32-bit), fld tbyte, fbld,
// fbstp, plus a page-crossing fnstenv. These go through the pre-validated
// gen_{readable,writable}_or_pagefault path in the JIT (a4ec212b port): the address
// is checked BEFORE the de-faulted helper runs, and a #PF must exit the compiled
// block through exit_with_fault_label with all register locals spilled.
//
// Guest: protected mode, own GDT/IDT, identity-paged 0..4MB. Each case unmaps
// FAULT_PAGE, runs a hot loop (gets JIT-compiled) executing the op against a
// mapped address, and on the LAST iteration points it at the unmapped page.
// The #PF handler (vector 14) verifies:
//   - live EBX == the per-case marker set right before the faulting op
//     (validates the exit-with-fault local spill),
//   - CR2 page == FAULT_PAGE,
// then maps the page and iret's; the instruction re-executes and must complete
// with the correct memory/FPU result (validated in-guest).
//
// Results land in registers at hlt:
//   esi = fault count (must be == number of cases)
//   edi = marker_errors | cr2_errors<<8 | value_errors<<16   (must be 0)
//
//   node tests/jit-fpu-pagefault-repro.mjs [path-to-libv86.mjs]

const LIB = process.argv[2] || "../build/libv86.mjs";
const { V86 } = await import(LIB.startsWith(".") || LIB.startsWith("/") || /^[A-Za-z]:/.test(LIB) ? LIB : "./" + LIB);

const BASE = 0x100000, ENTRY_OFF = 0x20;
const PD_ADDR = 0x108000, PT0_ADDR = 0x109000;
const FAULT_PAGE = 0x200000;
const PTE_ADDR = PT0_ADDR + (FAULT_PAGE >> 12) * 4;
const VALID = 0x210000;            // always-mapped scratch operand page
const DATA = 0x220000;             // result/variable page (always mapped)
const D_FAULTS = DATA + 0, D_EXPECT_EBX = DATA + 4, D_MARKER_ERR = DATA + 8;
const D_CR2_ERR = DATA + 12, D_FILD_SRC = DATA + 16, D_CW_TMP = DATA + 20;
const D_VALUE_ERR = DATA + 24, D_INT_TMP = DATA + 28;
const WARM = 60000;
const MEM_SIZE = 16 * 1024 * 1024, TIMEOUT_MS = 30000;

// static tables inside the image (first page)
const GDT_OFF = 0xE00, GDTR_OFF = 0xE20, IDT_OFF = 0xE30, IDTR_OFF = 0xEB8;
const PF_HANDLER_OFF = 0xF00;      // #PF handler, fixed offset

const N_CASES = 7;

function build_image()
{
    const buf = new Uint8Array(0x1000);
    const dv = new DataView(buf.buffer);
    const MAGIC = 0x1BADB002, FLAGS = 0x10000;
    dv.setUint32(0x00, MAGIC, true);
    dv.setUint32(0x04, FLAGS, true);
    dv.setUint32(0x08, (-(MAGIC + FLAGS)) >>> 0, true);
    dv.setUint32(0x0c, BASE, true);
    dv.setUint32(0x10, BASE, true);
    dv.setUint32(0x14, BASE + 0x1000, true);
    dv.setUint32(0x18, BASE + 0x4000, true);
    dv.setUint32(0x1c, BASE + ENTRY_OFF, true);

    let o = ENTRY_OFF;
    const emit = (...b) => { for(const x of b) buf[o++] = x & 0xff; };
    const u32 = (v) => { dv.setUint32(o, v >>> 0, true); o += 4; };
    const u16 = (v) => { dv.setUint16(o, v & 0xffff, true); o += 2; };

    // ---- static GDT: null, code 0x08, data 0x10 (flat 4GB) ----
    buf.set([0,0,0,0,0,0,0,0,
             0xFF,0xFF,0,0,0,0x9A,0xCF,0,
             0xFF,0xFF,0,0,0,0x92,0xCF,0], GDT_OFF);
    dv.setUint16(GDTR_OFF, 23, true);
    dv.setUint32(GDTR_OFF + 2, BASE + GDT_OFF, true);
    // ---- static IDT: entry 14 -> PF_HANDLER, selector 0x08, 32-bit intr gate ----
    const H = BASE + PF_HANDLER_OFF;
    const e14 = IDT_OFF + 14 * 8;
    dv.setUint16(e14, H & 0xFFFF, true);
    dv.setUint16(e14 + 2, 0x08, true);
    buf[e14 + 4] = 0; buf[e14 + 5] = 0x8E;
    dv.setUint16(e14 + 6, H >>> 16, true);
    dv.setUint16(IDTR_OFF, 16 * 8 - 1, true);
    dv.setUint32(IDTR_OFF + 2, BASE + IDT_OFF, true);

    // ---- entry: GDT/segments/IDT ----
    emit(0x0F, 0x01, 0x15); u32(BASE + GDTR_OFF);      // lgdt [gdtr]
    emit(0xEA); u32(0); u16(0x08);                     // ljmp 0x08:next (target patched)
    dv.setUint32(o - 6, BASE + o, true);
    emit(0x66, 0xB8, 0x10, 0x00);                      // mov ax, 0x10
    emit(0x8E, 0xD8, 0x8E, 0xC0, 0x8E, 0xD0);          // mov ds/es/ss, ax
    emit(0xBC); u32(0x300000);                         // mov esp, 0x300000
    emit(0x0F, 0x01, 0x1D); u32(BASE + IDTR_OFF);      // lidt [idtr]

    // ---- paging: identity map 0..4MB, one PT, CR0.PG on ----
    emit(0xFC);                                        // cld
    emit(0xBF); u32(PT0_ADDR);                         // mov edi, PT0
    emit(0xB8); u32(0x00000003);                       // mov eax, P|RW
    emit(0xB9); u32(0x400);                            // mov ecx, 1024
    const fill = o;
    emit(0xAB);                                        // stosd
    emit(0x05); u32(0x1000);                           // add eax, 0x1000
    emit(0xE2, (fill - (o + 2)) & 0xff);               // loop fill
    emit(0xBF); u32(PD_ADDR);                          // mov edi, PD
    emit(0x31, 0xC0);                                  // xor eax, eax
    emit(0xB9); u32(0x400);                            // mov ecx, 1024
    emit(0xF3, 0xAB);                                  // rep stosd
    emit(0xC7, 0x05); u32(PD_ADDR); u32(PT0_ADDR | 3); // PD[0] = PT0|3
    emit(0xB8); u32(PD_ADDR);                          // mov eax, PD
    emit(0x0F, 0x22, 0xD8);                            // mov cr3, eax
    emit(0x0F, 0x20, 0xC0);                            // mov eax, cr0
    emit(0x0D); u32(0x80000000);                       // or eax, PG
    emit(0x0F, 0x22, 0xC0);                            // mov cr0, eax

    // ---- FPU + variables ----
    emit(0xDB, 0xE3);                                  // fninit
    emit(0xC7, 0x05); u32(D_FILD_SRC); u32(1234);      // fild source
    emit(0xC7, 0x05); u32(D_FAULTS); u32(0);
    emit(0xC7, 0x05); u32(D_MARKER_ERR); u32(0);
    emit(0xC7, 0x05); u32(D_CR2_ERR); u32(0);
    emit(0xC7, 0x05); u32(D_VALUE_ERR); u32(0);
    // zero the VALID operand page head (warm-loop target: benign env/BCD)
    for(let i = 0; i < 32; i += 4) { emit(0xC7, 0x05); u32(VALID + i); u32(0); }

    // helpers -------------------------------------------------------------
    const unmap = () => {
        emit(0xC7, 0x05); u32(PTE_ADDR); u32(0);       // PTE = not present
        emit(0x0F, 0x01, 0x3D); u32(FAULT_PAGE);       // invlpg
    };
    const value_err_if_ne = () => {                    // flags -> value_errors
        emit(0x74, 0x06);                              // je +6
        emit(0xFF, 0x05); u32(D_VALUE_ERR);            // inc dword [value_errors]
    };
    // one hot-loop case; core(addrRegIsEdi) emits the op using [edx] (or [edi])
    const run_case = (marker, fault_addr, core, use_edi) => {
        emit(0xC7, 0x05); u32(D_EXPECT_EBX); u32(marker);
        emit(0xB9); u32(WARM);                         // mov ecx, WARM
        const loop = o;
        emit(use_edi ? 0xBF : 0xBA); u32(VALID);       // mov edx/edi, VALID
        emit(0x83, 0xF9, 0x01);                        // cmp ecx, 1
        emit(0x75, 0x05);                              // jne +5
        emit(use_edi ? 0xBF : 0xBA); u32(fault_addr);  // mov edx/edi, fault_addr
        emit(0xBB); u32(marker);                       // mov ebx, marker
        core();
        emit(0x49);                                    // dec ecx
        emit(0x75, (loop - (o + 2)) & 0xff);           // jnz loop
    };

    // ---- case 1: fnstenv m32 (write, 28 bytes) ----
    unmap();
    run_case(0xC0DE0010, FAULT_PAGE, () => emit(0xD9, 0x32));       // fnstenv [edx]
    emit(0x0F, 0xB7, 0x05); u32(FAULT_PAGE);           // movzx eax, word [FAULT_PAGE]
    emit(0x3D); u32(0x037F);                           // cmp eax, 0x37F
    value_err_if_ne();

    // ---- case 2: fldenv m32 (read; loads the env case 1 stored) ----
    unmap();
    run_case(0xC0DE0020, FAULT_PAGE, () => emit(0xD9, 0x22));       // fldenv [edx]
    emit(0xD9, 0x3D); u32(D_CW_TMP);                   // fnstcw [tmp]
    emit(0x0F, 0xB7, 0x05); u32(D_CW_TMP);
    emit(0x3D); u32(0x037F);
    value_err_if_ne();

    // ---- case 3: fld tbyte (read, 10 bytes) ----
    emit(0xD9, 0xE8);                                  // fld1
    emit(0xDB, 0x3D); u32(FAULT_PAGE);                 // fstp tbyte [FAULT_PAGE] (while mapped)
    unmap();
    run_case(0xC0DE0030, FAULT_PAGE, () => emit(0xDB, 0x2A, 0xDD, 0xD8)); // fld tbyte [edx]; fstp st0
    emit(0xDB, 0x2D); u32(FAULT_PAGE);                 // fld tbyte [FAULT_PAGE]
    emit(0xDB, 0x1D); u32(D_INT_TMP);                  // fistp dword [tmp]
    emit(0x81, 0x3D); u32(D_INT_TMP); u32(1);          // cmp dword [tmp], 1
    value_err_if_ne();

    // ---- case 4: fbld m80 (read, packed BCD "25") ----
    emit(0xC7, 0x05); u32(FAULT_PAGE); u32(0x25);      // while mapped
    emit(0xC7, 0x05); u32(FAULT_PAGE + 4); u32(0);
    emit(0xC7, 0x05); u32(FAULT_PAGE + 8); u32(0);
    unmap();
    run_case(0xC0DE0040, FAULT_PAGE, () => emit(0xDF, 0x22, 0xDD, 0xD8)); // fbld [edx]; fstp st0
    emit(0xDF, 0x25); u32(FAULT_PAGE);                 // fbld [FAULT_PAGE]
    emit(0xDB, 0x1D); u32(D_INT_TMP);                  // fistp dword [tmp]
    emit(0x81, 0x3D); u32(D_INT_TMP); u32(25);
    value_err_if_ne();

    // ---- case 5: fbstp m80 (write) ----
    unmap();
    run_case(0xC0DE0050, FAULT_PAGE, () => {
        emit(0xDB, 0x05); u32(D_FILD_SRC);             // fild dword [1234]
        emit(0xDF, 0x32);                              // fbstp [edx]
    });
    emit(0x81, 0x3D); u32(FAULT_PAGE); u32(0x1234);    // packed BCD 1234
    value_err_if_ne();

    // ---- case 6: maskmovq (write via EDI/DS, mask all-set) ----
    emit(0xC7, 0x05); u32(FAULT_PAGE); u32(0);         // while mapped
    emit(0x0F, 0x74, 0xC0);                            // pcmpeqb mm0, mm0 (all FF)
    unmap();
    run_case(0xC0DE0060, FAULT_PAGE, () => emit(0x0F, 0xF7, 0xC0), true); // maskmovq mm0, mm0
    emit(0x0F, 0x77);                                  // emms
    emit(0x81, 0x3D); u32(FAULT_PAGE); u32(0xFFFFFFFF);
    value_err_if_ne();

    // ---- case 7: page-crossing fnstenv at FAULT_PAGE-8 (2nd page unmapped) ----
    unmap();
    run_case(0xC0DE0070, FAULT_PAGE - 8, () => emit(0xD9, 0x32));
    emit(0x0F, 0xB7, 0x05); u32(FAULT_PAGE - 8);
    emit(0x3D); u32(0x037F);
    value_err_if_ne();

    // ---- results -> registers; halt ----
    emit(0x8B, 0x35); u32(D_FAULTS);                   // mov esi, [faults]
    emit(0x8B, 0x3D); u32(D_MARKER_ERR);               // mov edi, [marker_errors]
    emit(0xA1); u32(D_CR2_ERR);                        // mov eax, [cr2_errors]
    emit(0xC1, 0xE0, 0x08);                            // shl eax, 8
    emit(0x09, 0xC7);                                  // or edi, eax
    emit(0xA1); u32(D_VALUE_ERR);
    emit(0xC1, 0xE0, 0x10);                            // shl eax, 16
    emit(0x09, 0xC7);                                  // or edi, eax
    emit(0xF4, 0xEB, 0xFE);                            // hlt; jmp $

    if(o > GDT_OFF) throw new Error("code overlaps tables: " + o.toString(16));

    // ---- #PF handler at fixed offset ----
    o = PF_HANDLER_OFF;
    emit(0x50);                                        // push eax
    emit(0xFF, 0x05); u32(D_FAULTS);                   // inc dword [faults]
    emit(0xA1); u32(D_EXPECT_EBX);                     // mov eax, [expected_ebx]
    emit(0x39, 0xD8);                                  // cmp eax, ebx
    emit(0x74, 0x06);                                  // je +6
    emit(0xFF, 0x05); u32(D_MARKER_ERR);
    emit(0x0F, 0x20, 0xD0);                            // mov eax, cr2
    emit(0x25); u32(0xFFFFF000);                       // and eax, ~0xFFF
    emit(0x3D); u32(FAULT_PAGE);                       // cmp eax, FAULT_PAGE
    emit(0x74, 0x06);                                  // je +6
    emit(0xFF, 0x05); u32(D_CR2_ERR);
    emit(0xC7, 0x05); u32(PTE_ADDR); u32(FAULT_PAGE | 3); // map the page
    emit(0x0F, 0x01, 0x3D); u32(FAULT_PAGE);           // invlpg
    emit(0x58);                                        // pop eax
    emit(0x83, 0xC4, 0x04);                            // add esp, 4 (error code)
    emit(0xCF);                                        // iretd

    return buf;
}

function run({ jit })
{
    return new Promise((resolve) => {
        const buf = build_image();
        const emulator = new V86({ autostart: false, memory_size: MEM_SIZE,
                                   disable_jit: jit ? 0 : 1, log_level: 0 });
        let halted = false, timer;
        const finish = (status) => {
            clearTimeout(timer);
            try { emulator.stop(); } catch(e) {}
            const cpu = emulator.v86.cpu;
            resolve({ status, faults: cpu.reg32[6] >>> 0, errors: cpu.reg32[7] >>> 0 });
        };
        emulator.bus.register("cpu-event-halt", () => { halted = true; finish("halt"); });
        emulator.add_listener("emulator-loaded", () => {
            const cpu = emulator.v86.cpu;
            cpu.reboot_internal(); cpu.reset_memory();
            cpu.load_multiboot(buf.buffer);
            timer = setTimeout(() => { if(!halted) finish("HANG"); }, TIMEOUT_MS);
            emulator.run();
        });
    });
}

const ok = (r) => r.status === "halt" && r.faults === N_CASES && r.errors === 0;
const show = (l, r) => console.log(l.padEnd(14) + JSON.stringify(r) +
    (ok(r) ? "  <- ok" : "  <- FAIL (want faults=" + N_CASES + " errors=0)"));

console.log("=== #PF during jitted fnstenv/fldenv/fld80/fbld/fbstp (+page-cross), lib=%s ===", LIB);
const ri = await run({ jit: false });
show("interp", ri);
const rj = await run({ jit: true });
show("JIT", rj);
process.exit(ok(ri) && ok(rj) ? 0 : 1);
