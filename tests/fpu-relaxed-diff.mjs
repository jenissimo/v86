#!/usr/bin/env node
// Differential test for the relaxed-FPU inline JIT fast path.
//
// Oracle:  interpreter + relaxed(1)  — helper-based semantics (known-good, shipped for weeks)
// Suspect: JIT         + relaxed(1)  — new inline codegen (gen_fpu_relaxed_*)
// The same multiboot image runs in both; final results land in eax/ebx/ecx/edx
// and must match BIT-EXACT (same f64 arithmetic on both paths).
//
// Variants isolate instruction families so a divergence pinpoints the buggy
// codegen path without rebuild-bisecting:
//   push   - FLD m32/m64/st(i) inline pushes (+helper FILD/FSTP)
//   d8mem  - D8 m32 binops (FADD/FMUL/FSUB/FSUBR/FDIV/FDIVR m32)
//   dcmem  - DC m64 binops
//   reg    - D8 (st0,sti) and DC (sti,st0) register binops
//   pfx    - DE P-forms (binop + inline pop)
//   full   - mixed chain incl. FXCH (helper) interplay
//
//   node tests/fpu-relaxed-diff.mjs
const { V86 } = await import("../build/libv86.mjs");
// Resolved from THIS FILE, not the working directory: the loader's default is relative to the
// cwd, so the suite only ran from inside vendor/v86 and its own npm script could not start it.
const WASM_PATH = process.env.V86_WASM_PATH || new URL("../build/v86.wasm", import.meta.url)
    .pathname.replace(/^\/([A-Za-z]:)/, "$1");

const BASE = 0x100000, ENTRY_OFF = 0x40;
const DATA = BASE + 0x3000;
const N = 400000;                 // > JIT_THRESHOLD(200k) so the loop compiles naturally
const MEM_SIZE = 16 * 1024 * 1024;
const TIMEOUT_MS = 30000;

// data layout (absolute guest addresses)
const COUNTER = DATA + 0;         // i32, increments per iteration
const C32     = DATA + 8;         // f32 1.5
const C64     = DATA + 16;        // f64 0.75
const C64B    = DATA + 24;        // f64 1.25
const ACC     = DATA + 32;        // f64 result slot
const LAST    = DATA + 40;        // f32 scratch slot
const C80     = DATA + 48;        // f80 1.5 (untagged when FLD'd)
const CW_NEAR = DATA + 64;
const CW_CEIL = DATA + 66;
const CW_TRUNC = DATA + 68;
const CW_FLOOR = DATA + 70;        // RC=01 (round down)
const CW_SINGLE = DATA + 72;       // PC=00: round arithmetic to a 24-bit significand
const ROUND_POS = DATA + 80;      // f64 2.5
const ROUND_NEG = DATA + 112;     // f64 -2.5
const OUT0    = DATA + 96;
const OUT1    = DATA + 100;
const OUT2    = DATA + 104;
const OUT3    = DATA + 108;
const I16V    = DATA + 120;       // i16 7
const I64V    = DATA + 128;       // i64 5
const C32_TINY = DATA + 136;      // f32 2^-30: lost when added to 1 at PC=00
const C64_ONE = DATA + 144;       // f64 1.0

function build_image(bodyName)
{
    const IMG_SIZE = 0x4000;
    const buf = new Uint8Array(IMG_SIZE);
    const dv  = new DataView(buf.buffer);
    const MAGIC = 0x1BADB002, FLAGS = 0x10000;
    dv.setUint32(0x00, MAGIC, true);
    dv.setUint32(0x04, FLAGS, true);
    dv.setUint32(0x08, (-(MAGIC + FLAGS)) >>> 0, true);
    dv.setUint32(0x0c, BASE, true);                 // header_addr
    dv.setUint32(0x10, BASE, true);                 // load_addr
    dv.setUint32(0x14, BASE + IMG_SIZE, true);      // load_end_addr
    dv.setUint32(0x18, BASE + IMG_SIZE, true);      // bss_end_addr
    dv.setUint32(0x1c, BASE + ENTRY_OFF, true);     // entry_addr

    // initialized data
    dv.setUint32(COUNTER - BASE, 1, true);          // counter starts at 1 (avoid div-by-0)
    dv.setFloat32(C32 - BASE, 1.5, true);
    dv.setFloat64(C64 - BASE, 0.75, true);
    dv.setFloat64(C64B - BASE, 1.25, true);
    dv.setFloat64(ACC - BASE, 0.0, true);
    dv.setFloat32(LAST - BASE, 0.0, true);
    // f80 1.5: mantissa 0xC000000000000000, sign_exponent 0x3FFF
    dv.setUint32(C80 - BASE, 0, true);
    dv.setUint32(C80 - BASE + 4, 0xC0000000, true);
    dv.setUint16(C80 - BASE + 8, 0x3FFF, true);
    dv.setUint16(CW_NEAR - BASE, 0x037F, true);
    dv.setUint16(CW_CEIL - BASE, 0x0B7F, true);
    dv.setUint16(CW_TRUNC - BASE, 0x0F7F, true);
    dv.setUint16(CW_FLOOR - BASE, 0x077F, true);
    dv.setUint16(CW_SINGLE - BASE, 0x007F, true);
    dv.setFloat64(ROUND_POS - BASE, 2.5, true);
    dv.setFloat64(ROUND_NEG - BASE, -2.5, true);
    dv.setInt16(I16V - BASE, 7, true);
    dv.setBigInt64(I64V - BASE, 5n, true);
    dv.setFloat32(C32_TINY - BASE, 2 ** -30, true);
    dv.setFloat64(C64_ONE - BASE, 1.0, true);

    let o = ENTRY_OFF;
    const labels = {}, patches = [];
    const label = (n) => { labels[n] = o; };
    const emit = (...b) => { for(const x of b) buf[o++] = x & 0xff; };
    const imm32 = (v) => { dv.setUint32(o, v >>> 0, true); o += 4; };
    const rel32 = (n) => { patches.push({ at:o, to:n, end:o+4 }); emit(0,0,0,0); };

    // mem-operand emitters (mod=00 r/m=101 disp32)
    const fild32  = (a) => { emit(0xDB, 0x05); imm32(a); };
    const fld32   = (a) => { emit(0xD9, 0x05); imm32(a); };
    const fld64   = (a) => { emit(0xDD, 0x05); imm32(a); };
    const fldsti  = (i) => emit(0xD9, 0xC0 + i);
    const fstp32  = (a) => { emit(0xD9, 0x1D); imm32(a); };
    const fstp64  = (a) => { emit(0xDD, 0x1D); imm32(a); };
    const fist32  = (a) => { emit(0xDB, 0x15); imm32(a); };
    const fldcw   = (a) => { emit(0xD9, 0x2D); imm32(a); };
    const d8mem   = (op, a) => { emit(0xD8, op); imm32(a); };   // op: 05 add, 0D mul, 25 sub, 2D subr, 35 div, 3D divr
    const dcmem   = (op, a) => { emit(0xDC, op); imm32(a); };
    const fxch1   = () => emit(0xD9, 0xC9);
    const fcom1   = () => emit(0xD8, 0xD1);
    const fnstsw_ax = () => emit(0xDF, 0xE0);
    const fld1    = () => emit(0xD9, 0xE8);
    const fldpi   = () => emit(0xD9, 0xEB);
    const fldz    = () => emit(0xD9, 0xEE);
    const fchs    = () => emit(0xD9, 0xE0);
    const fabs    = () => emit(0xD9, 0xE1);
    const faddp1  = () => emit(0xDE, 0xC1);
    const fcomi1  = () => emit(0xDB, 0xF1);
    const movEaxMem = (a) => { emit(0xA1); imm32(a); };
    const movMemEax = (a) => { emit(0xA3); imm32(a); };
    // --- additional emitters for the previously-untested new relaxed families ---
    const fcom_m32  = (a) => { emit(0xD8, 0x15); imm32(a); };   // FCOM  m32
    const fcomp_m32 = (a) => { emit(0xD8, 0x1D); imm32(a); };   // FCOMP m32
    const fcom_m64  = (a) => { emit(0xDC, 0x15); imm32(a); };   // FCOM  m64
    const fcomp_m64 = (a) => { emit(0xDC, 0x1D); imm32(a); };   // FCOMP m64
    const fcom_i32  = (a) => { emit(0xDA, 0x15); imm32(a); };   // FICOM  m32int
    const fcom_i16  = (a) => { emit(0xDE, 0x15); imm32(a); };   // FICOM  m16int
    const fst_sti   = (i) => emit(0xDD, 0xD0 + i);             // FST  ST(i)
    const fstp_sti  = (i) => emit(0xDD, 0xD8 + i);             // FSTP ST(i)
    const fild16    = (a) => { emit(0xDF, 0x05); imm32(a); };   // FILD m16int
    const fild64    = (a) => { emit(0xDF, 0x2D); imm32(a); };   // FILD m64int
    const fist16    = (a) => { emit(0xDF, 0x15); imm32(a); };   // FIST m16int
    const fistp16   = (a) => { emit(0xDF, 0x1D); imm32(a); };   // FISTP m16int
    const fldl2t = () => emit(0xD9, 0xE9);
    const fldl2e = () => emit(0xD9, 0xEA);
    const fldlg2 = () => emit(0xD9, 0xEC);
    const fldln2 = () => emit(0xD9, 0xED);
    const fucomi1  = () => emit(0xDB, 0xE9);                    // FUCOMI  ST,ST(1)
    const fucomip1 = () => emit(0xDF, 0xE9);                    // FUCOMIP ST,ST(1)
    const fcomip1  = () => emit(0xDF, 0xF1);                    // FCOMIP  ST,ST(1)

    const bodies = {
        // inline pushes: FLD m32, FLD m64, FLD st(i); pops via helper FSTP
        push() {
            fild32(COUNTER);              // st0=n                      (helper push)
            fld32(C32);                   // st0=1.5  st1=n             (inline push m32)
            fld64(C64);                   // st0=.75  st1=1.5 st2=n     (inline push m64)
            fldsti(2);                    // st0=n    ...               (inline push sti)
            fstp64(ACC);                  // acc=n
            fstp32(LAST);                 // .75
            fstp32(LAST);                 // 1.5
            fstp32(LAST);                 // n
        },
        d8mem() {
            fild32(COUNTER);
            d8mem(0x05, C32);             // fadd  st0 = n+1.5
            d8mem(0x0D, C32);             // fmul  *1.5
            d8mem(0x2D, C32);             // fsubr 1.5-st0
            d8mem(0x35, C32);             // fdiv  /1.5
            d8mem(0x3D, C32);             // fdivr 1.5/st0
            d8mem(0x25, C32);             // fsub  st0-1.5
            fstp64(ACC);
        },
        dcmem() {
            fild32(COUNTER);
            dcmem(0x05, C64);             // fadd m64
            dcmem(0x0D, C64B);            // fmul m64
            dcmem(0x2D, C64);             // fsubr m64
            dcmem(0x35, C64B);            // fdiv m64
            dcmem(0x3D, C64);             // fdivr m64
            dcmem(0x25, C64B);            // fsub m64
            fstp64(ACC);
        },
        reg() {
            fild32(COUNTER);              // st0=n
            fld32(C32);                   // st0=1.5 st1=n
            emit(0xD8, 0xC1);             // fadd  st0,st1
            emit(0xD8, 0xC9);             // fmul  st0,st1
            emit(0xD8, 0xE1);             // fsub  st0,st1
            emit(0xD8, 0xE9);             // fsubr st0,st1
            emit(0xD8, 0xF1);             // fdiv  st0,st1
            emit(0xD8, 0xF9);             // fdivr st0,st1
            emit(0xDC, 0xC1);             // fadd  st1,st0
            emit(0xDC, 0xE9);             // fsub  st1,st0
            emit(0xDC, 0xF9);             // fdiv  st1,st0
            fstp32(LAST);
            fstp64(ACC);
        },
        pfx() {
            fild32(COUNTER);              // st0=n
            fld32(C32);  emit(0xDE, 0xC1);   // faddp  st1,st0
            fld32(C32);  emit(0xDE, 0xC9);   // fmulp  st1,st0
            fld32(C32);  emit(0xDE, 0xE9);   // fsubp  st1,st0
            fld32(C32);  emit(0xDE, 0xE1);   // fsubrp st1,st0
            fld32(C32);  emit(0xDE, 0xF9);   // fdivp  st1,st0
            fld32(C32);  emit(0xDE, 0xF1);   // fdivrp st1,st0
            fstp64(ACC);
        },
        // reg+disp8 / SIB addressing — exercises the other gen_modrm_resolve paths
        // (TLB fast/slow if-else nesting inside the new inline branches)
        addr() {
            // ebx = DATA set in prologue below via patch hook (see emitAddrSetup)
            fild32(COUNTER);
            emit(0xD8, 0x43, C32 - DATA);    // fadd dword [ebx+8]
            emit(0xD8, 0x4B, C32 - DATA);    // fmul dword [ebx+8]
            emit(0xDC, 0x63, C64 - DATA);    // fsub qword [ebx+16]
            emit(0xDC, 0x73, C64B - DATA);   // fdiv qword [ebx+24]
            emit(0xD9, 0x44, 0x23, C32 - DATA); // fld dword [ebx+ahem SIB: base=ebx]
            emit(0xDE, 0xC1);                // faddp
            fstp64(ACC);
        },
        // untagged F80 interplay: FLD m80 pushes a GENUINE 80-bit value (no RELAXED_TAG)
        // → inline binops must take their SLOW (helper) branch inside compiled code
        m80() {
            fild32(COUNTER);                 // tagged
            emit(0xDB, 0x2D); imm32(C80);    // fld tbyte [c80] — untagged 1.5
            emit(0xD8, 0xC1);                // fadd st0,st1   (one side untagged → slow)
            d8mem(0x0D, C32);                // fmul m32 on untagged-result chain
            emit(0xDE, 0xC1);                // faddp st1,st0  (slow branch + inline pop)
            dcmem(0x2D, C64);                // fsubr m64
            fstp64(ACC);
        },
        // minimal slow-branch repros
        m80_sti() {                          // sti binop slow branch only
            emit(0xDB, 0x2D); imm32(C80);    // fld tbyte — untagged
            emit(0xD8, 0xC0);                // fadd st0,st0 (slow: untagged)
            fstp64(ACC);
        },
        m80_mem() {                          // m32 binop slow branch only
            emit(0xDB, 0x2D); imm32(C80);    // fld tbyte — untagged
            d8mem(0x05, C32);                // fadd m32 (slow: untagged st0)
            fstp64(ACC);
        },
        m80_pop() {                          // slow binop + inline pop
            emit(0xDB, 0x2D); imm32(C80);
            emit(0xDB, 0x2D); imm32(C80);
            emit(0xDE, 0xC1);                // faddp st1,st0 (slow + inline pop)
            fstp64(ACC);
        },
        m80_nopop() {                        // slow binop r=1, helper pops only
            emit(0xDB, 0x2D); imm32(C80);
            emit(0xDB, 0x2D); imm32(C80);
            emit(0xDC, 0xC1);                // fadd st1,st0 (slow, no pop)
            fstp64(ACC);                     // helper pop
            fstp64(ACC);                     // helper pop (overwrites — fine)
        },
        // PC-local dominance: the first arithmetic instruction takes its F80 slow
        // branch but leaves the relaxed ST0 untouched; the immediately following D8
        // takes the fast branch. The live PC predicate must be initialized on both paths.
        pc_local_mixed() {
            // Alternate two predecessor paths into one compiled arithmetic block. Odd
            // iterations initialize its PC local to true through the fast branch;
            // even iterations change PC to double precision and take the first op's
            // F80 slow branch. The latter must overwrite, rather than reuse, that one.
            emit(0xF7, 0x05); imm32(COUNTER); imm32(1); // test dword [counter],1
            emit(0x0F, 0x84); rel32("pc_single");
            fldcw(CW_SINGLE);
            fld64(C64);                      // relaxed ST1 seed
            emit(0xE9); rel32("pc_join");
            label("pc_single");
            fldcw(CW_NEAR);
            emit(0xDB, 0x2D); imm32(C80);    // F80 ST1 seed
            label("pc_join");
            fld64(C64_ONE);                  // ST0: relaxed 1.0
            emit(0xDC, 0xC1);                // slow on even iterations; ST0 unchanged
            d8mem(0x05, C32_TINY);            // even: must retain the 2^-30 increment
            fstp64(ACC);
            fstp64(OUT0);                    // drain the F80 result
            fldcw(CW_NEAR);
        },
        tag_pop() {                          // tagged values through the SAME DE faddp
            fld32(C32);
            fld32(C32);
            emit(0xDE, 0xC1);                // faddp (fast + inline pop)
            fstp64(ACC);
        },
        // diagnostic: m80_pop body, but bail out at the FIRST NaN in ACC and
        // dump FPU env. Exit regs: eax=iteration, ebx=status word(TOP in 11-13),
        // ecx=tag word, edx=counter.
        diag() {
            emit(0xDB, 0x2D); imm32(C80);
            emit(0xDB, 0x2D); imm32(C80);
            emit(0xDE, 0xC1);                // faddp st1,st0 (slow + inline pop)
            fstp64(ACC);
            emit(0xA1); imm32(ACC + 4);      // mov eax, [acc hi]
            emit(0x3D); imm32(0x7ff80000);   // cmp eax, NaN-hi
            emit(0x0F, 0x84); rel32("nan_hit"); // je nan_hit
        },
        full() {
            fild32(COUNTER);              // st0=n
            fld32(C32);                   // push 1.5
            emit(0xD8, 0xC1);             // fadd st0,st1
            fld64(C64);                   // push .75
            emit(0xDE, 0xC9);             // fmulp st1,st0
            d8mem(0x2D, C32);             // fsubr m32
            dcmem(0x35, C64B);            // fdiv m64
            fxch1();                      // helper interplay
            emit(0xD8, 0xE1);             // fsub st0,st1
            dcmem(0x05, ACC);             // fadd qword [acc]  — accumulate across iterations
            fstp64(ACC);
            fstp32(LAST);
        },
        fxch_sticky() {
            fild32(COUNTER);              // tagged int -> relaxed f64
            fld32(C32);                   // tagged m32
            fxch1();                      // must preserve tags
            emit(0xD8, 0xC1);             // fadd st0,st1 should stay fast
            fstp64(ACC);
            fstp32(LAST);
        },
        fcom_sticky() {
            fild32(COUNTER);
            fld32(C32);
            fcom1();                      // compare must not poison either slot
            emit(0xD8, 0xC9);             // fmul st0,st1 should stay fast
            fstp64(ACC);
            fstp32(LAST);
            fnstsw_ax();
            movMemEax(LAST);
        },
        consts_sign() {
            fld1();
            fchs();
            fabs();
            fldpi();
            faddp1();                     // st0 = pi + abs(-1)
            fstp64(ACC);
        },
        fist_round() {
            fld64(ROUND_POS);
            fldcw(CW_NEAR);
            fist32(OUT0);
            movEaxMem(OUT0);
            movMemEax(ACC);
            fldcw(CW_CEIL);
            fist32(OUT1);
            movEaxMem(OUT1);
            movMemEax(ACC + 4);
            fldcw(CW_TRUNC);
            fist32(OUT2);
            movEaxMem(OUT2);
            movMemEax(LAST);
            fstp32(OUT3);
            fldcw(CW_NEAR);
        },
        fcomi_flags() {
            fldz();
            fld1();
            fcomi1();                     // st0=1 > st1=0, CF=ZF=PF=0
            emit(0x31, 0xC0);             // xor eax,eax
            emit(0x0F, 0x97, 0xC0);       // seta al
            movMemEax(LAST);
            fstp32(OUT0);
            fstp32(OUT1);
        },
        // FIST of a negative value across the RC=01 (floor/down) mode the positive
        // fist_round case never exercises. floor(-2.5)=-3 distinguishes it from
        // nearest/ceil(-2.5)=-2, so a wrong RC-decode in the codegen diverges here.
        fist_round_neg() {
            fld64(ROUND_NEG);             // st0 = -2.5
            fldcw(CW_FLOOR);
            fist32(OUT0);                 // floor(-2.5) = -3
            movEaxMem(OUT0);
            movMemEax(ACC);
            fldcw(CW_NEAR);
            fist32(OUT1);                 // nearest-even(-2.5) = -2
            movEaxMem(OUT1);
            movMemEax(ACC + 4);
            fldcw(CW_CEIL);
            fist32(OUT2);                 // ceil(-2.5) = -2
            movEaxMem(OUT2);
            movMemEax(LAST);
            fstp32(OUT3);                 // drop st0
            fldcw(CW_NEAR);
        },
        // FCOM against MEMORY operands (m32/m64/m16int/m32int) — none of which the
        // other variants exercise (they only do FCOM ST(i)). Each: push, compare, drain.
        fcom_mem() {
            fild32(COUNTER); fcomp_m64(C64); fnstsw_ax(); movMemEax(ACC + 4);   // FCOMP m64 (pops)
            fild32(COUNTER); fcom_i16(I16V); fnstsw_ax(); movMemEax(LAST); fstp32(LAST); // FICOM m16
            fild32(COUNTER); fcom_i32(COUNTER); fnstsw_ax(); movMemEax(ACC); fstp32(LAST); // FICOM m32
            fild32(COUNTER); fcomp_m32(C32); fnstsw_ax();                       // FCOMP m32 (pops)
        },
        // FST / FSTP to a REGISTER (DD /2, DD /3 reg) — raw 16-byte slot copy path.
        fst_reg() {
            fild32(COUNTER);
            fld32(C32);                   // [1.5, n]
            fst_sti(1);                   // FST ST(1): [1.5, 1.5]
            fstp_sti(1);                  // FSTP ST(1): [1.5]
            fstp64(ACC);                  // ACC = 1.5, empty
        },
        // FILD m16int / m64int widths (only m32 was covered).
        fild_widths() {
            fild16(I16V);                 // FILD m16
            fild64(I64V);                 // FILD m64
            emit(0xDE, 0xC1);             // faddp -> i16 + i64
            fstp64(ACC);
        },
        // FIST / FISTP m16int + the 16-bit clamp path.
        fist16_round() {
            fld64(ROUND_POS);             // 2.5
            fldcw(CW_NEAR);
            fist16(OUT0);                 // nearest-even(2.5) = 2
            movEaxMem(OUT0);
            movMemEax(ACC);
            fistp16(OUT1);                // nearest(2.5)=2, pop
            movEaxMem(OUT1);
            movMemEax(ACC + 4);
            fldcw(CW_NEAR);
        },
        // remaining FLD constants (l2t/l2e/lg2/ln2 — only 1/pi/z were covered).
        consts_all() {
            fldl2t(); fldl2e(); faddp1();
            fldlg2(); faddp1();
            fldln2(); faddp1();
            fstp64(ACC);
        },
        // FUCOMI / FCOMIP / FUCOMIP eflags forms (only FCOMI was covered).
        fcomi_more() {
            fldz(); fld1(); fucomi1();    // FUCOMI ST,ST(1)
            emit(0x9C); emit(0x58); movMemEax(ACC);   // pushfd; pop eax
            fstp32(LAST); fstp32(LAST);
            fldz(); fld1(); fcomip1();    // FCOMIP (pops 1)
            emit(0x9C); emit(0x58); movMemEax(ACC + 4);
            fstp32(LAST);
            fldz(); fld1(); fucomip1();   // FUCOMIP (pops 1)
            emit(0x9C); emit(0x58); movMemEax(LAST);
            fstp32(LAST);
        },
        // Compiled-VC6-T&L-shaped mix: FILD/FLD/FMUL/FXCH/FADDP/FCOM-mem/FNSTSW/
        // FDIV-mem/FST-reg/FSTP interleaved (closer to real engine code than the
        // isolated patterns above; this is where the poison/structural bug should bite).
        tl_mix() {
            fild32(COUNTER);              // [n]
            fld32(C32);                   // [1.5, n]
            fld64(C64);                   // [.75, 1.5, n]
            emit(0xD8, 0xC9);             // fmul st0,st1 -> [1.125, 1.5, n]
            fxch1();                      // -> [1.5, 1.125, n]
            emit(0xDE, 0xC1);             // faddp st1,st0 -> [2.625, n]
            fcom_m64(C64);                // compare vs .75 (no pop)
            fnstsw_ax(); movMemEax(LAST);
            d8mem(0x35, C32);             // fdiv m32 -> [1.75, n]
            fst_sti(1);                   // FST ST(1) -> [1.75, 1.75]
            fstp64(ACC);                  // -> [1.75]
            fstp64(OUT0);                 // -> empty
        },
        // FIADD/FIMUL/FISUB/FIDIV m32int (DA group): integer operand reaches the f80
        // helper, so it must be loaded as f80 — relaxed mode used to emit the relaxed
        // (f64-bits, RELAXED_TAG) form here, misread as a wild operand. Regression guard.
        fiarith32() {
            fild32(COUNTER);              // st0 = n (relaxed-tagged)
            emit(0xDA, 0x05); imm32(COUNTER);  // FIADD m32 -> 2n
            emit(0xDA, 0x0D); imm32(COUNTER);  // FIMUL m32 -> 2n*n
            emit(0xDA, 0x25); imm32(COUNTER);  // FISUB m32
            emit(0xDA, 0x35); imm32(COUNTER);  // FIDIV m32 (counter >= 1, never 0)
            emit(0xDA, 0x3D); imm32(COUNTER);  // FIDIVR m32
            fstp64(ACC);
        },
        // FNSTSW AX with a non-trivial TOP (stack_ptr) value. The inline codegen for
        // DF /4 rebuilds the status word as *sw & !(7<<11) | (stack_ptr&7)<<11; this pins
        // TOP=5 (three pushes from a fninit'd stack) so a botched shift/mask in the TOP
        // field diverges from the helper oracle. eax (zeroed first) captures the raw SW.
        fnstsw_top() {
            fld1();                        // TOP=7
            fld1();                        // TOP=6
            fld64(C64);                    // TOP=5, st0=0.75
            emit(0x31, 0xC0);              // xor eax,eax (zero high bits)
            fnstsw_ax();                   // AX = status word, TOP field = 5
            movMemEax(ACC);                // ACC = raw SW -> eax/ebx compared
            fstp64(LAST);                  // drain 0.75  -> TOP=6
            fstp32(LAST);                  // drain       -> TOP=7
            fstp32(LAST);                  // drain       -> TOP=0 (empty)
        },
        // FIADD/FIMUL/FISUB/FIDIV m16int (DE group) — same f80-format requirement.
        fiarith16() {
            fild32(COUNTER);
            emit(0xDE, 0x05); imm32(I16V);     // FIADD m16 (7)
            emit(0xDE, 0x0D); imm32(I16V);     // FIMUL m16
            emit(0xDE, 0x25); imm32(I16V);     // FISUB m16
            emit(0xDE, 0x35); imm32(I16V);     // FIDIV m16
            emit(0xDE, 0x2D); imm32(I16V);     // FISUBR m16
            fstp64(ACC);
        },
        // FST ST(i) also clears the destination's stack-empty bit; a raw register copy
        // leaves the tag word claiming ST(i) is empty.
        fst_tagword() {
            fld1();                            // TOP=7, st0=1.0
            fst_sti(3);                        // -> phys (7+3)&7 = 2, currently empty
            emit(0xD9, 0x35); imm32(DATA + 0x200);  // fnstenv [env]
            emit(0xA1); imm32(DATA + 0x208);   // mov eax, [tag word]
            movMemEax(ACC);
            emit(0xDD, 0xC3);                  // ffree st(3) -> release phys 2
            fstp32(LAST);                      // drain st0
        },
        // FLD ST(i) on an empty slot yields INDEFINITE_NAN + a stack fault, not stale bytes.
        fld_empty() {
            emit(0xDB, 0xE2);                  // fnclex (measure this iteration's flags)
            fld1();                            // TOP=7
            fldsti(3);                         // ST(3) = phys 2, empty
            fstp64(ACC);                       // ACC = what the empty slot produced
            fstp32(LAST);                      // drain the 1.0
            emit(0x31, 0xC0);                  // xor eax,eax
            fnstsw_ax();
            movMemEax(LAST);                   // LAST = status word (SF|IE, C1 clear)
        },
        // A 9th push onto a full stack is an OVERFLOW: fpu_push sets C1 + SF|IE and writes
        // INDEFINITE_NAN, keeping the live register — it does not silently store the value.
        push_overflow() {
            emit(0xDB, 0xE3);                  // fninit (empty stack, SW=0, TOP=0)
            for(let i = 0; i < 8; i++) fld64(C64B);   // fill the stack with 1.25 -> TOP=0
            fld64(C64);                        // 9th push of 0.75 -> overflow
            emit(0x31, 0xC0);                  // xor eax,eax
            fnstsw_ax();
            movMemEax(LAST);                   // LAST = status word
            fstp64(ACC);                       // ACC = ST(0) after the 9th push
            emit(0xDB, 0xE3);                  // fninit (drain for the next iteration)
        },
        // FST ST(i) reads ST(0) through fpu_get_st0: an EMPTY source raises SF|IE and stores
        // INDEFINITE_NAN, so a raw slot copy would resurrect the freed slot's stale bytes.
        fst_empty_src() {
            emit(0xDB, 0xE3);                  // fninit
            fld1();                            // TOP=7
            fld64(C64);                        // TOP=6, phys6 = 0.75
            emit(0xDD, 0xC0);                  // ffree st(0) -> phys6 empty, stale bytes = 0.75
            emit(0xDB, 0xE2);                  // fnclex (measure this iteration's flags)
            fst_sti(1);                        // FST ST(1) with an empty source
            emit(0x31, 0xC0);                  // xor eax,eax
            fnstsw_ax();
            movMemEax(LAST);                   // LAST = status word
            emit(0xD9, 0xF7);                  // fincstp -> ST(0) is now the destination
            fstp64(ACC);                       // ACC = what FST wrote into ST(1)
            emit(0xDB, 0xE3);                  // fninit
        },
        // An ARITHMETIC binop reads its operands the same way FST does, and the inline relaxed
        // path has its own operand check: the f64 tag in the slot's exponent word. FFREE leaves
        // that word — and the value — untouched, so the tag still says "an f64 lives here" while
        // the register is empty. The helper path raises SF|IE and computes on INDEFINITE_NAN;
        // an inline path that only trusts the tag computes on the freed 0.75 and reports neither.
        fadd_empty_src() {
            emit(0xDB, 0xE3);                  // fninit
            fld1();                            // TOP=7, phys7 = 1.0
            fld64(C64);                        // TOP=6, phys6 = 0.75
            emit(0xDD, 0xC0);                  // ffree st(0) -> phys6 empty, stale bytes = 0.75
            emit(0xDB, 0xE2);                  // fnclex (measure this iteration's flags)
            emit(0xDC, 0xC1);                  // FADD ST(1), ST(0) with an empty ST(0)
            emit(0x31, 0xC0);                  // xor eax,eax
            fnstsw_ax();
            movMemEax(LAST);                   // LAST = status word
            emit(0xD9, 0xF7);                  // fincstp -> ST(0) is now the destination
            fstp64(ACC);                       // ACC = what FADD left in ST(1)
            emit(0xDB, 0xE3);                  // fninit
        },
        // A successful push clears C1 (fpu_push); FXAM sets it first so a skipped clear shows.
        push_c1() {
            fld64(C64);                        // 0.75
            fchs();                            // -0.75
            emit(0xD9, 0xE5);                  // fxam -> C1 = sign = 1
            fld1();                            // push must clear C1
            emit(0x31, 0xC0);                  // xor eax,eax
            fnstsw_ax();
            movMemEax(ACC);
            fstp32(LAST);
            fstp32(LAST);
        },
    };

    emit(0xBC); imm32(0x200000);          // mov esp, 0x200000
    emit(0xBB); imm32(DATA);              // mov ebx, DATA (for reg-based addressing)
    emit(0xDB, 0xE3);                     // fninit
    emit(0x31, 0xF6);                     // xor esi, esi
    label("outer");
    emit(0x81, 0xFE); imm32(N);           // cmp esi, N
    emit(0x0F, 0x8D); rel32("done");      // jge done
    bodies[bodyName]();
    emit(0xFF, 0x05); imm32(COUNTER);     // inc dword [counter]
    emit(0x46);                           // inc esi
    emit(0xE9); rel32("outer");           // jmp outer
    label("done");
    emit(0xA1); imm32(ACC);               // mov eax, [acc lo]
    emit(0x8B, 0x1D); imm32(ACC + 4);     // mov ebx, [acc hi]
    emit(0x8B, 0x0D); imm32(LAST);        // mov ecx, [last]
    emit(0x8B, 0x15); imm32(COUNTER);     // mov edx, [counter]
    emit(0xF4);                           // hlt
    label("nan_hit");
    emit(0xD9, 0x35); imm32(DATA + 0x200); // fnstenv [env]
    emit(0x89, 0xF0);                      // mov eax, esi (iteration)
    emit(0x8B, 0x1D); imm32(DATA + 0x204); // mov ebx, [status word]
    emit(0x8B, 0x0D); imm32(DATA + 0x208); // mov ecx, [tag word]
    emit(0x8B, 0x15); imm32(COUNTER);      // mov edx, [counter]
    emit(0xF4);                            // hlt
    emit(0xEB, 0xFE);                      // jmp $

    for(const p of patches)
        dv.setInt32(p.at, labels[p.to] - p.end, true);
    return buf;
}

function run(bodyName, { jit, relaxed, x87Locals = false })
{
    return new Promise((resolve) => {
        const img = build_image(bodyName);
        const emulator = new V86({ autostart:false, memory_size:MEM_SIZE,
                                   wasm_path: WASM_PATH,
                                   disable_jit: jit ? 0 : 1, log_level:0 });
        let halted = false, timer;
        const finish = (status) => {
            clearTimeout(timer);
            try { emulator.stop(); } catch(e) {}
            const cpu = emulator.v86.cpu;
            const ex = cpu.wm?.exports ?? {};
            const hit = typeof ex.profiler_fpu_relaxed_hit_get === "function" ? Number(ex.profiler_fpu_relaxed_hit_get()) : 0;
            const fallback = typeof ex.profiler_fpu_relaxed_fallback_get === "function" ? Number(ex.profiler_fpu_relaxed_fallback_get()) : 0;
            resolve({ status,
                      eax: cpu.reg32[0] >>> 0, ecx: cpu.reg32[1] >>> 0,
                      edx: cpu.reg32[2] >>> 0, ebx: cpu.reg32[3] >>> 0,
                      hit, fallback,
                      eip: cpu.instruction_pointer[0] >>> 0 });
        };
        emulator.bus.register("cpu-event-halt", () => { halted = true; finish("halt"); });
        emulator.add_listener("emulator-loaded", () => {
            const cpu = emulator.v86.cpu;
            const setRelaxed = cpu.wm?.exports?.set_relaxed_fpu;
            if(!setRelaxed) { console.error("FATAL: set_relaxed_fpu export not found"); process.exit(2); }
            setRelaxed(relaxed ? 1 : 0);
            cpu.wm?.exports?.set_fpu_relaxed_stats?.(1);   // counters are gated; enable for the diff harness
            cpu.wm?.exports?.profiler_init?.();
            cpu.reboot_internal(); cpu.reset_memory();
            cpu.load_multiboot(img.buffer);
            setRelaxed(relaxed ? 1 : 0);   // re-assert in case reset touched it
            cpu.wm?.exports?.set_fpu_relaxed_stats?.(1);
            // Phase 2: JIT_X87_LOCALS = config idx 10. Engages the block-scoped st
            // read-through local cache. The central invalidation in jit.rs must keep
            // it coherent across helper-path x87 ops (fld m80 / fild / fsqrt / etc.).
            cpu.wm?.exports?.set_jit_config?.(10, (jit && relaxed && x87Locals) ? 1 : 0);
            cpu.wm?.exports?.set_jit_config?.(21, process.env.V86_FLAG_LOCALS === '1' ? 1 : 0);
            cpu.wm?.exports?.profiler_init?.();
            timer = setTimeout(() => { if(!halted) finish("HANG"); }, TIMEOUT_MS);
            emulator.run();
        });
    });
}

const fmt = (r) => `${r.status} acc=${r.ebx.toString(16).padStart(8,"0")}:${r.eax.toString(16).padStart(8,"0")} last=${r.ecx.toString(16).padStart(8,"0")} n=${r.edx} hit=${r.hit} fallback=${r.fallback}`;

const variants = ["push", "d8mem", "dcmem", "reg", "pfx", "addr", "m80", "m80_sti", "m80_mem", "m80_pop", "m80_nopop", "pc_local_mixed", "tag_pop", "full", "fxch_sticky", "fcom_sticky", "consts_sign", "fist_round", "fist_round_neg", "fcomi_flags",
    // newly-covered families (previously untested, where Bug #2 is hypothesised to live)
    "fcom_mem", "fst_reg", "fild_widths", "fist16_round", "consts_all", "fcomi_more", "tl_mix",
    "fnstsw_top", "fiarith32", "fiarith16",
    // architectural state the inline paths used to drop: tag word, empty-slot reads, C1
    "fst_tagword", "fld_empty", "push_c1",
    // stack faults the inline paths used to swallow: push overflow, FST from an empty ST(0)
    "push_overflow", "fst_empty_src", "fadd_empty_src"]
    // optional argv filter: `node tests/fpu-relaxed-diff.mjs fld_empty push_c1`
    .filter(v => process.argv.length <= 2 || process.argv.slice(2).includes(v));
// FCOM/FXCH between two relaxed values must not fall back to the helper (that re-poisons
// the slot). Only meaningful in relaxed(1).
const stickyVariants = ["fxch_sticky", "fcom_sticky"];
let anyDiverged = false;
// Run both modes: relaxed(1) = inline fast path, relaxed(0) = helper branch. A non-"halt"
// status in either mode is a failure (a structural codegen bug hangs), not just a mismatch.
for(const relaxed of [true, false]) {
    console.log(`\n===== relaxed(${relaxed ? 1 : 0}) =====`);
    for(const v of variants) {
        const oracle  = await run(v, { jit:false, relaxed });
        const suspect = await run(v, { jit:true,  relaxed });
        const bothHalted = oracle.status === "halt" && suspect.status === "halt";
        const same = bothHalted
                  && oracle.eax === suspect.eax && oracle.ebx === suspect.ebx
                  && oracle.ecx === suspect.ecx && oracle.edx === suspect.edx;
        const stickyCountersOk = !relaxed || !stickyVariants.includes(v)
                  || (suspect.hit > 0 && suspect.fallback === 0);
        if(!same || !stickyCountersOk) anyDiverged = true;
        console.log(`${v.padEnd(14)} interp: ${fmt(oracle)}`);
        console.log(`${"".padEnd(14)} jit   : ${fmt(suspect)}  ${same && stickyCountersOk ? "OK" : "<<< DIVERGED"}`);
        if(!bothHalted) {
            console.log(`${"".padEnd(14)}       non-halt (hang/crash): interp=${oracle.status}@${oracle.eip.toString(16)} jit=${suspect.status}@${suspect.eip.toString(16)}`);
        }
        if(relaxed && stickyVariants.includes(v) && !stickyCountersOk) {
            console.log(`${"".padEnd(14)}       expected ${v} to stay fully relaxed (hit>0 fallback=0)`);
        }
    }
}
// Phase 2 pass: JIT + relaxed + x87 locals vs the interpreter oracle. This is the
// stale-cache regression guard — the mixed kernels (tl_mix, m80_*, fiarith,
// fist_round, fcom_mem) interleave helper-path FPU ops that mutate the stack /
// shift TOP behind the local cache; without central invalidation they diverge.
console.log(`\n===== x87-locals(1) [jit+relaxed] =====`);
for(const v of variants) {
    const oracle  = await run(v, { jit:false, relaxed:true });
    const suspect = await run(v, { jit:true,  relaxed:true, x87Locals:true });
    const bothHalted = oracle.status === "halt" && suspect.status === "halt";
    const same = bothHalted
              && oracle.eax === suspect.eax && oracle.ebx === suspect.ebx
              && oracle.ecx === suspect.ecx && oracle.edx === suspect.edx;
    if(!same) anyDiverged = true;
    console.log(`${v.padEnd(14)} interp: ${fmt(oracle)}`);
    console.log(`${"".padEnd(14)} x87loc: ${fmt(suspect)}  ${same ? "OK" : "<<< DIVERGED"}`);
    if(!bothHalted) {
        console.log(`${"".padEnd(14)}       non-halt (hang/crash): interp=${oracle.status}@${oracle.eip.toString(16)} jit=${suspect.status}@${suspect.eip.toString(16)}`);
    }
}
console.log(anyDiverged ? "\nVERDICT: inline fast path DIVERGES from helpers" : "\nVERDICT: all variants match");
process.exit(anyDiverged ? 1 : 0);
