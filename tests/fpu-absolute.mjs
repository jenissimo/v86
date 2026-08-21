#!/usr/bin/env node
// Absolute x87 semantics checks, against values a real 387 produces.
//
// fpu-relaxed-diff.mjs compares the JIT against the interpreter, so it is blind to bugs
// that live in the shared Rust helpers — both sides are wrong identically. These two were:
// FXTRACT dropping the significand's sign, and FLD m80 misreading an image whose exponent
// field equals RELAXED_TAG (2^16383, e.g. LDBL_MAX). Run in all four jit x relaxed modes.
//
//   node tests/fpu-absolute.mjs
const { V86 } = await import("../build/libv86.mjs");

const BASE = 0x100000, ENTRY_OFF = 0x40;
const DATA = BASE + 0x3000;
const NEG8 = DATA + 0;      // f64 -8.0
const LDBLMAX = DATA + 16;  // f80 mantissa=FFFF.., sign_exponent=0x7FFE
const OUT_A = DATA + 32;    // significand
const OUT_B = DATA + 40;    // exponent
const OUT_C = DATA + 48;    // m80 -> f64

function build_image()
{
    const IMG_SIZE = 0x4000;
    const buf = new Uint8Array(IMG_SIZE);
    const dv = new DataView(buf.buffer);
    const MAGIC = 0x1BADB002, FLAGS = 0x10000;
    dv.setUint32(0x00, MAGIC, true);
    dv.setUint32(0x04, FLAGS, true);
    dv.setUint32(0x08, (-(MAGIC + FLAGS)) >>> 0, true);
    dv.setUint32(0x0c, BASE, true);
    dv.setUint32(0x10, BASE, true);
    dv.setUint32(0x14, BASE + IMG_SIZE, true);
    dv.setUint32(0x18, BASE + IMG_SIZE, true);
    dv.setUint32(0x1c, BASE + ENTRY_OFF, true);

    dv.setFloat64(NEG8 - BASE, -8.0, true);
    dv.setUint32(LDBLMAX - BASE, 0xFFFFFFFF, true);
    dv.setUint32(LDBLMAX - BASE + 4, 0xFFFFFFFF, true);
    dv.setUint16(LDBLMAX - BASE + 8, 0x7FFE, true);

    let o = ENTRY_OFF;
    const emit = (...b) => { for(const x of b) buf[o++] = x & 0xff; };
    const imm32 = (v) => { dv.setUint32(o, v >>> 0, true); o += 4; };

    emit(0xBC); imm32(0x200000);            // mov esp, 0x200000
    emit(0xDB, 0xE3);                       // fninit
    emit(0xDD, 0x05); imm32(NEG8);          // fld  qword [-8.0]
    emit(0xD9, 0xF4);                       // fxtract
    emit(0xDD, 0x1D); imm32(OUT_A);         // fstp qword [significand]
    emit(0xDD, 0x1D); imm32(OUT_B);         // fstp qword [exponent]
    emit(0xDB, 0x2D); imm32(LDBLMAX);       // fld  tbyte [LDBL_MAX]
    emit(0xDD, 0x1D); imm32(OUT_C);         // fstp qword
    emit(0xA1); imm32(OUT_A);               // mov eax, [sig lo]
    emit(0x8B, 0x1D); imm32(OUT_A + 4);     // mov ebx, [sig hi]
    emit(0x8B, 0x0D); imm32(OUT_B + 4);     // mov ecx, [exp hi]
    emit(0x8B, 0x15); imm32(OUT_C + 4);     // mov edx, [m80 hi]
    emit(0xF4);                             // hlt
    emit(0xEB, 0xFE);
    return buf;
}

// A STRICT-mode FLD m64 / FSTP m64 of an f64 subnormal must round-trip bit-exactly: the
// pair is of_f64_strict -> to_f64_strict, so a biasing error in either shows up here.
// Relaxed mode never reaches of_f64_strict on this path (of_f64 short-circuits), which is
// why only the strict rows can catch it.
const SUBNORM = DATA + 64;      // largest f64 subnormal, 0x000FFFFFFFFFFFFF
const SUBNORM_OUT = DATA + 72;
const SUBNORM_LO = 0xFFFFFFFF, SUBNORM_HI = 0x000FFFFF;

function build_subnormal_image()
{
    const IMG_SIZE = 0x4000;
    const buf = new Uint8Array(IMG_SIZE);
    const dv = new DataView(buf.buffer);
    const MAGIC = 0x1BADB002, FLAGS = 0x10000;
    dv.setUint32(0x00, MAGIC, true);
    dv.setUint32(0x04, FLAGS, true);
    dv.setUint32(0x08, (-(MAGIC + FLAGS)) >>> 0, true);
    dv.setUint32(0x0c, BASE, true);
    dv.setUint32(0x10, BASE, true);
    dv.setUint32(0x14, BASE + IMG_SIZE, true);
    dv.setUint32(0x18, BASE + IMG_SIZE, true);
    dv.setUint32(0x1c, BASE + ENTRY_OFF, true);

    dv.setUint32(SUBNORM - BASE, SUBNORM_LO, true);
    dv.setUint32(SUBNORM - BASE + 4, SUBNORM_HI, true);

    let o = ENTRY_OFF;
    const emit = (...b) => { for(const x of b) buf[o++] = x & 0xff; };
    const imm32 = (v) => { dv.setUint32(o, v >>> 0, true); o += 4; };

    emit(0xBC); imm32(0x200000);            // mov esp, 0x200000
    emit(0xDB, 0xE3);                       // fninit
    emit(0xDD, 0x05); imm32(SUBNORM);       // fld  qword [subnormal]
    emit(0xDD, 0x1D); imm32(SUBNORM_OUT);   // fstp qword
    emit(0xA1); imm32(SUBNORM_OUT);         // mov eax, [lo]
    emit(0x8B, 0x1D); imm32(SUBNORM_OUT+4); // mov ebx, [hi]
    emit(0xF4);                             // hlt
    emit(0xEB, 0xFE);
    return buf;
}

function run_subnormal({ jit, relaxed })
{
    return new Promise((resolve) => {
        const img = build_subnormal_image();
        const emulator = new V86({ autostart:false, memory_size:16*1024*1024,
                                   disable_jit: jit ? 0 : 1, log_level:0 });
        let timer;
        const finish = (status) => {
            clearTimeout(timer);
            const cpu = emulator.v86.cpu;
            const r = { status, eax: cpu.reg32[0]>>>0, ebx: cpu.reg32[3]>>>0 };
            try { emulator.stop(); } catch(e) {}
            resolve(r);
        };
        emulator.bus.register("cpu-event-halt", () => finish("halt"));
        emulator.add_listener("emulator-loaded", () => {
            const cpu = emulator.v86.cpu;
            const setRelaxed = cpu.wm.exports.set_relaxed_fpu;
            setRelaxed(relaxed ? 1 : 0);
            cpu.reboot_internal(); cpu.reset_memory();
            cpu.load_multiboot(img.buffer);
            setRelaxed(relaxed ? 1 : 0);
            timer = setTimeout(() => finish("HANG"), 15000);
            emulator.run();
        });
    });
}

// Mid-run mode switch: FLD m80 in STRICT mode leaves a genuine 2^16383-scale value in the
// register (its exponent field is RELAXED_TAG); the OUT toggles relaxed on, and the FSTP
// must still see +Inf rather than the mantissa reinterpreted as raw f64 bits. Single pass,
// so this runs in the interpreter — it characterises set_relaxed_fpu, not the codegen.
const TOGGLE_PORT = 0x8888;

function build_toggle_image()
{
    const IMG_SIZE = 0x4000;
    const buf = new Uint8Array(IMG_SIZE);
    const dv = new DataView(buf.buffer);
    const MAGIC = 0x1BADB002, FLAGS = 0x10000;
    dv.setUint32(0x00, MAGIC, true);
    dv.setUint32(0x04, FLAGS, true);
    dv.setUint32(0x08, (-(MAGIC + FLAGS)) >>> 0, true);
    dv.setUint32(0x0c, BASE, true);
    dv.setUint32(0x10, BASE, true);
    dv.setUint32(0x14, BASE + IMG_SIZE, true);
    dv.setUint32(0x18, BASE + IMG_SIZE, true);
    dv.setUint32(0x1c, BASE + ENTRY_OFF, true);

    dv.setUint32(LDBLMAX - BASE, 0xFFFFFFFF, true);
    dv.setUint32(LDBLMAX - BASE + 4, 0xFFFFFFFF, true);
    dv.setUint16(LDBLMAX - BASE + 8, 0x7FFE, true);

    let o = ENTRY_OFF;
    const emit = (...b) => { for(const x of b) buf[o++] = x & 0xff; };
    const imm32 = (v) => { dv.setUint32(o, v >>> 0, true); o += 4; };

    emit(0xBC); imm32(0x200000);            // mov esp, 0x200000
    emit(0xDB, 0xE3);                       // fninit
    emit(0xDB, 0x2D); imm32(LDBLMAX);       // fld  tbyte [LDBL_MAX]
    emit(0xBA); imm32(TOGGLE_PORT);         // mov edx, TOGGLE_PORT
    emit(0xEE);                             // out dx, al  -> host flips relaxed on
    emit(0xDD, 0x1D); imm32(OUT_C);         // fstp qword
    emit(0xA1); imm32(OUT_C);               // mov eax, [lo]
    emit(0x8B, 0x15); imm32(OUT_C + 4);     // mov edx, [hi]
    emit(0xF4);                             // hlt
    emit(0xEB, 0xFE);
    return buf;
}

function run_toggle()
{
    return new Promise((resolve) => {
        const img = build_toggle_image();
        const emulator = new V86({ autostart:false, memory_size:16*1024*1024,
                                   disable_jit: 1, log_level:0 });
        let timer;
        const finish = (status) => {
            clearTimeout(timer);
            try { emulator.stop(); } catch(e) {}
            const cpu = emulator.v86.cpu;
            resolve({ status, eax: cpu.reg32[0]>>>0, edx: cpu.reg32[2]>>>0 });
        };
        emulator.bus.register("cpu-event-halt", () => finish("halt"));
        emulator.add_listener("emulator-loaded", () => {
            const cpu = emulator.v86.cpu;
            const setRelaxed = cpu.wm.exports.set_relaxed_fpu;
            setRelaxed(0);
            cpu.reboot_internal(); cpu.reset_memory();
            cpu.load_multiboot(img.buffer);
            setRelaxed(0);
            cpu.io.register_write(TOGGLE_PORT, cpu, () => setRelaxed(1));
            timer = setTimeout(() => finish("HANG"), 15000);
            emulator.run();
        });
    });
}

function run({ jit, relaxed })
{
    return new Promise((resolve) => {
        const img = build_image();
        const emulator = new V86({ autostart:false, memory_size:16*1024*1024,
                                   disable_jit: jit ? 0 : 1, log_level:0 });
        let timer;
        const finish = (status) => {
            clearTimeout(timer);
            try { emulator.stop(); } catch(e) {}
            const cpu = emulator.v86.cpu;
            resolve({ status, eax: cpu.reg32[0]>>>0, ecx: cpu.reg32[1]>>>0,
                      edx: cpu.reg32[2]>>>0, ebx: cpu.reg32[3]>>>0 });
        };
        emulator.bus.register("cpu-event-halt", () => finish("halt"));
        emulator.add_listener("emulator-loaded", () => {
            const cpu = emulator.v86.cpu;
            const setRelaxed = cpu.wm?.exports?.set_relaxed_fpu;
            setRelaxed(relaxed ? 1 : 0);
            cpu.reboot_internal(); cpu.reset_memory();
            cpu.load_multiboot(img.buffer);
            setRelaxed(relaxed ? 1 : 0);
            timer = setTimeout(() => finish("HANG"), 15000);
            emulator.run();
        });
    });
}

const hex = (v) => v.toString(16).padStart(8, "0");
let fail = false;
for(const relaxed of [true, false])
for(const jit of [false, true]) {
    const r = await run({ jit, relaxed });
    // FXTRACT(-8.0): significand -1.0 (BFF0000000000000), exponent 3.0 (4008000000000000)
    // FLD tbyte LDBL_MAX -> +Inf when narrowed to f64 (7FF0000000000000)
    const okSig = r.ebx === 0xBFF00000 && r.eax === 0;
    const okExp = r.ecx === 0x40080000;
    const okM80 = r.edx === 0x7FF00000;
    if(!(okSig && okExp && okM80)) fail = true;
    console.log(`relaxed=${relaxed?1:0} jit=${jit?1:0} ${r.status}`,
                `significand=${hex(r.ebx)}:${hex(r.eax)} ${okSig?"OK":"<<< BAD (want bff00000:00000000)"}`,
                `exponent=${hex(r.ecx)} ${okExp?"OK":"<<< BAD (want 40080000)"}`,
                `m80=${hex(r.edx)} ${okM80?"OK":"<<< BAD (want 7ff00000)"}`);
}
{
    const r = await run_toggle();
    const okToggle = r.edx === 0x7FF00000 && r.eax === 0;
    if(!okToggle) fail = true;
    console.log(`strict->relaxed toggle ${r.status}`,
                `m80=${hex(r.edx)}:${hex(r.eax)} ${okToggle?"OK":"<<< BAD (want 7ff00000:00000000)"}`);
}
for(const relaxed of [false, true])
for(const jit of [false, true]) {
    const r = await run_subnormal({ jit, relaxed });
    const ok = r.ebx === SUBNORM_HI && r.eax === SUBNORM_LO;
    if(!ok) fail = true;
    console.log(`subnormal round-trip relaxed=${relaxed?1:0} jit=${jit?1:0} ${r.status}`,
                `out=${hex(r.ebx)}:${hex(r.eax)}`,
                ok ? "OK" : `<<< BAD (want ${hex(SUBNORM_HI)}:${hex(SUBNORM_LO)})`);
}
// ─── Control word written BEHIND set_control_word ──────────────────────────────
//
// The host swaps the whole x87 state on every guest-thread switch by writing the
// 134-byte snapshot straight into wasm memory (src/worker/core/fpu-helper.ts
// fpuRestore) — fpu_control_word included, and no set_control_word in sight. CPU
// reset and any state-region restore do the same. So RC/PC must be read from the
// live control word at use, never from a value derived when FLDCW last ran: a
// derived copy belongs to whichever thread executed FLDCW last, while the JIT's
// inline FIST decodes the control word itself (codegen gen_fpu_round_f64_bits_to_i32)
// — two implementations of one instruction, disagreeing. FRNDINT has no inline path
// at all, which is what makes it the discriminator here.
//
// The OUT is the context switch: the host writes a new control word into wasm
// memory mid-run, exactly as fpuRestore does, without touching the guest.
const CW_PORT = 0x8889;
const CW_ADDR = 1036;               // fpu_control_word (cpu/global_pointers.rs)
const CW_BOOT = DATA + 128;         // 0x037F: RC=near, PC=extended
const VAL27 = DATA + 136;           // f64 2.7
const ONE = DATA + 144;             // f64 1.0
const THREE = DATA + 152;           // f64 3.0
const OUT_RND = DATA + 160;         // f64 FRNDINT result
const OUT_FIST = DATA + 168;        // i32 FISTP result
const OUT_CW = DATA + 172;          // u16 FNSTCW readback (setup self-check)
const OUT_DIV = DATA + 176;         // f64 division result

function build_cw_image(kind)
{
    const IMG_SIZE = 0x4000;
    const buf = new Uint8Array(IMG_SIZE);
    const dv = new DataView(buf.buffer);
    const MAGIC = 0x1BADB002, FLAGS = 0x10000;
    dv.setUint32(0x00, MAGIC, true);
    dv.setUint32(0x04, FLAGS, true);
    dv.setUint32(0x08, (-(MAGIC + FLAGS)) >>> 0, true);
    dv.setUint32(0x0c, BASE, true);
    dv.setUint32(0x10, BASE, true);
    dv.setUint32(0x14, BASE + IMG_SIZE, true);
    dv.setUint32(0x18, BASE + IMG_SIZE, true);
    dv.setUint32(0x1c, BASE + ENTRY_OFF, true);

    dv.setUint16(CW_BOOT - BASE, 0x037F, true);
    dv.setFloat64(VAL27 - BASE, 2.7, true);
    dv.setFloat64(ONE - BASE, 1.0, true);
    dv.setFloat64(THREE - BASE, 3.0, true);

    let o = ENTRY_OFF;
    const emit = (...b) => { for(const x of b) buf[o++] = x & 0xff; };
    const imm32 = (v) => { dv.setUint32(o, v >>> 0, true); o += 4; };

    emit(0xBC); imm32(0x200000);            // mov  esp, 0x200000
    emit(0xDB, 0xE3);                       // fninit
    // The one FLDCW the guest ever executes. Its only job is to make a cached
    // RC/PC hold "near / extended" — without it there is nothing stale to catch.
    emit(0xD9, 0x2D); imm32(CW_BOOT);       // fldcw  [0x037F]
    emit(0xBA); imm32(CW_PORT);             // mov  edx, CW_PORT
    emit(0xEE);                             // out  dx, al   -> host rewrites the CW
    emit(0xD9, 0x3D); imm32(OUT_CW);        // fnstcw [OUT_CW]   (did the write land?)

    if(kind === "rc") {
        emit(0xDD, 0x05); imm32(VAL27);     // fld   qword [2.7]
        emit(0xD9, 0xFC);                   // frndint            (helper only)
        emit(0xDD, 0x1D); imm32(OUT_RND);   // fstp  qword
        emit(0xDD, 0x05); imm32(VAL27);     // fld   qword [2.7]
        emit(0xDB, 0x1D); imm32(OUT_FIST);  // fistp dword        (inline under relaxed+JIT)
        emit(0xA1); imm32(OUT_RND + 4);     // mov  eax, [rnd hi]
        emit(0x8B, 0x1D); imm32(OUT_FIST);  // mov  ebx, [fist]
    }
    else {
        emit(0xDD, 0x05); imm32(ONE);       // fld  qword [1.0]
        emit(0xDC, 0x35); imm32(THREE);     // fdiv qword [3.0]
        emit(0xDD, 0x1D); imm32(OUT_DIV);   // fstp qword
        emit(0xA1); imm32(OUT_DIV + 4);     // mov  eax, [div hi]
        emit(0x8B, 0x1D); imm32(OUT_DIV);   // mov  ebx, [div lo]
    }
    emit(0x0F, 0xB7, 0x0D); imm32(OUT_CW);  // movzx ecx, word [OUT_CW]
    emit(0xF4);                             // hlt
    emit(0xEB, 0xFE);
    return buf;
}

function run_cw(kind, cw, { jit, relaxed })
{
    return new Promise((resolve) => {
        const img = build_cw_image(kind);
        const emulator = new V86({ autostart:false, memory_size:16*1024*1024,
                                   disable_jit: jit ? 0 : 1, log_level:0 });
        let timer;
        const finish = (status) => {
            clearTimeout(timer);
            try { emulator.stop(); } catch(e) {}
            const cpu = emulator.v86.cpu;
            resolve({ status, eax: cpu.reg32[0]>>>0, ebx: cpu.reg32[3]>>>0,
                      ecx: cpu.reg32[1]>>>0 });
        };
        emulator.bus.register("cpu-event-halt", () => finish("halt"));
        emulator.add_listener("emulator-loaded", () => {
            const cpu = emulator.v86.cpu;
            const setRelaxed = cpu.wm.exports.set_relaxed_fpu;
            setRelaxed(relaxed ? 1 : 0);
            cpu.reboot_internal(); cpu.reset_memory();
            cpu.load_multiboot(img.buffer);
            setRelaxed(relaxed ? 1 : 0);
            cpu.io.register_write(CW_PORT, cpu, () => {
                new DataView(cpu.wasm_memory.buffer).setUint16(CW_ADDR, cw, true);
            });
            timer = setTimeout(() => finish("HANG"), 15000);
            emulator.run();
        });
    });
}

const f64hi = (v) => { const d = new DataView(new ArrayBuffer(8)); d.setFloat64(0, v); return d.getUint32(0); };
const f64lo = (v) => { const d = new DataView(new ArrayBuffer(8)); d.setFloat64(0, v); return d.getUint32(4); };

// RC=11 (truncate) arrives behind the guest's back: FRNDINT(2.7) and FISTP(2.7) must
// both yield 2, not the 3 a stale RC=near produces.
for(const relaxed of [false, true])
for(const jit of [false, true]) {
    const r = await run_cw("rc", 0x0F7F, { jit, relaxed });
    const okCw = r.ecx === 0x0F7F;                  // setup self-check
    const okRnd = r.eax === f64hi(2.0);
    const okFist = r.ebx === 2;
    if(!(okCw && okRnd && okFist)) fail = true;
    console.log(`live CW rc=trunc relaxed=${relaxed?1:0} jit=${jit?1:0} ${r.status}`,
                `cw=${r.ecx.toString(16)} ${okCw?"OK":"<<< SETUP INVALID (want 0f7f)"}`,
                `frndint=${hex(r.eax)} ${okRnd?"OK":`<<< BAD (want ${hex(f64hi(2.0))})`}`,
                `fistp=${r.ebx} ${okFist?"OK":"<<< BAD (want 2)"}`);
}

// PC=00 (24-bit single) arrives the same way. Strict mode must round 1/3 to f32;
// relaxed mode ignores PC by contract (see softfloat apply_precision) and keeps f64.
for(const relaxed of [false, true])
for(const jit of [false, true]) {
    const r = await run_cw("pc", 0x007F, { jit, relaxed });
    const want = relaxed ? 1/3 : Math.fround(1/3);
    const okCw = r.ecx === 0x007F;
    const okDiv = r.eax === f64hi(want) && r.ebx === f64lo(want);
    if(!(okCw && okDiv)) fail = true;
    console.log(`live CW pc=single relaxed=${relaxed?1:0} jit=${jit?1:0} ${r.status}`,
                `cw=${r.ecx.toString(16)} ${okCw?"OK":"<<< SETUP INVALID (want 007f)"}`,
                `1/3=${hex(r.eax)}:${hex(r.ebx)}`,
                okDiv ? "OK" : `<<< BAD (want ${hex(f64hi(want))}:${hex(f64lo(want))})`);
}

console.log(fail ? "VERDICT: FAIL" : "VERDICT: all OK");
process.exit(fail ? 1 : 0);
