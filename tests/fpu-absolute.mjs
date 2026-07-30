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
console.log(fail ? "VERDICT: FAIL" : "VERDICT: all OK");
process.exit(fail ? 1 : 0);
