#!/usr/bin/env node
// Absolute SSE3 + CPU-identity checks, run against the BUILT wasm in both the interpreter and
// the JIT. Reading the .rs proves nothing here: the three SSE3 opcodes we advertise via
// CPUID.1:ECX[0] used to decode to unimplemented_sse() -> #UD with no log line in a release
// build, so the only trustworthy evidence is executing them. The SSE block runs in a loop so
// the JIT pass really compiles it rather than interpreting a single iteration.
//
//   node tests/sse3-absolute.mjs
const { V86 } = await import("../build/libv86.mjs");

const BASE = 0x100000, ENTRY_OFF = 0x40;
const DATA = BASE + 0x3000;

// inputs
const A_PS   = DATA + 0x00;   // f32[4] destination for addsubps
const B_PS   = DATA + 0x10;   // f32[4] source
const C_PD   = DATA + 0x20;   // f64[2] destination for addsubpd
const D_PD   = DATA + 0x30;   // f64[2] source
const UNALIGNED = DATA + 0x41; // deliberately not 16-byte aligned — lddqu's whole reason to exist
// outputs
const OUT_PS = DATA + 0x100;
const OUT_PD = DATA + 0x110;
const OUT_DQ = DATA + 0x120;
const OUT_CPUID = DATA + 0x130; // leaf1.ecx, leaf 0x40000000.ebx, leaf 0x80000000.eax
const OUT_BRAND = DATA + 0x140; // 48 bytes

const LOOP_ITERATIONS = 0x4000;

const UNALIGNED_BYTES = [
    0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
    0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF,
];

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

    for(let i = 0; i < 4; i++) dv.setFloat32(A_PS - BASE + i * 4, [10, 20, 30, 40][i], true);
    for(let i = 0; i < 4; i++) dv.setFloat32(B_PS - BASE + i * 4, [1, 2, 3, 4][i], true);
    dv.setFloat64(C_PD - BASE + 0, 100, true);
    dv.setFloat64(C_PD - BASE + 8, 200, true);
    dv.setFloat64(D_PD - BASE + 0, 1, true);
    dv.setFloat64(D_PD - BASE + 8, 2, true);
    for(let i = 0; i < 16; i++) buf[UNALIGNED - BASE + i] = UNALIGNED_BYTES[i];

    let o = ENTRY_OFF;
    const emit = (...b) => { for(const x of b) buf[o++] = x & 0xff; };
    const imm32 = (v) => { dv.setUint32(o, v >>> 0, true); o += 4; };

    emit(0xBC); imm32(0x200000);                    // mov esp, 0x200000
    emit(0x0F, 0x20, 0xE0);                         // mov eax, cr4
    emit(0x0D); imm32(0x600);                       // or  eax, OSFXSR|OSXMMEXCPT
    emit(0x0F, 0x22, 0xE0);                         // mov cr4, eax

    emit(0xB9); imm32(LOOP_ITERATIONS);             // mov ecx, N
    const loop_start = o;
    emit(0x0F, 0x10, 0x05); imm32(A_PS);            // movups xmm0, [A]
    emit(0x0F, 0x10, 0x0D); imm32(B_PS);            // movups xmm1, [B]
    emit(0xF2, 0x0F, 0xD0, 0xC1);                   // addsubps xmm0, xmm1     (register form)
    emit(0x0F, 0x11, 0x05); imm32(OUT_PS);          // movups [OUT_PS], xmm0

    emit(0x0F, 0x10, 0x15); imm32(C_PD);            // movups xmm2, [C]
    emit(0x66, 0x0F, 0xD0, 0x15); imm32(D_PD);      // addsubpd xmm2, [D]      (memory form)
    emit(0x0F, 0x11, 0x15); imm32(OUT_PD);          // movups [OUT_PD], xmm2

    emit(0xF2, 0x0F, 0xF0, 0x25); imm32(UNALIGNED); // lddqu xmm4, [unaligned]
    emit(0x0F, 0x11, 0x25); imm32(OUT_DQ);          // movups [OUT_DQ], xmm4

    emit(0x49);                                     // dec ecx
    emit(0x0F, 0x85); dv.setInt32(o, loop_start - (o + 4), true); o += 4; // jnz loop_start

    emit(0xB8); imm32(1);                           // mov eax, 1
    emit(0x31, 0xC9);                               // xor ecx, ecx
    emit(0x0F, 0xA2);                               // cpuid
    emit(0x89, 0x0D); imm32(OUT_CPUID + 0);         // mov [OUT_CPUID+0], ecx

    emit(0xB8); imm32(0x40000000);                  // mov eax, 0x40000000
    emit(0x0F, 0xA2);                               // cpuid
    emit(0x89, 0x1D); imm32(OUT_CPUID + 4);         // mov [OUT_CPUID+4], ebx

    emit(0xB8); imm32(0x80000000);                  // mov eax, 0x80000000
    emit(0x0F, 0xA2);                               // cpuid
    emit(0xA3); imm32(OUT_CPUID + 8);               // mov [OUT_CPUID+8], eax

    for(let leaf = 0; leaf < 3; leaf++) {
        const dst = OUT_BRAND + leaf * 16;
        emit(0xB8); imm32(0x80000002 + leaf);       // mov eax, 0x8000000{2,3,4}
        emit(0x0F, 0xA2);                           // cpuid
        emit(0xA3); imm32(dst + 0);                 // mov [dst+0],  eax
        emit(0x89, 0x1D); imm32(dst + 4);           // mov [dst+4],  ebx
        emit(0x89, 0x0D); imm32(dst + 8);           // mov [dst+8],  ecx
        emit(0x89, 0x15); imm32(dst + 12);          // mov [dst+12], edx
    }

    emit(0xF4);                                     // hlt
    emit(0xEB, 0xFE);
    return buf;
}

function run({ jit })
{
    return new Promise((resolve) => {
        const img = build_image();
        const emulator = new V86({ autostart: false, memory_size: 16 * 1024 * 1024,
                                   disable_jit: jit ? 0 : 1, log_level: 0 });
        let timer;
        const finish = (status) => {
            clearTimeout(timer);
            const cpu = emulator.v86.cpu;
            const mem = cpu.mem8;
            const dv = new DataView(mem.buffer, mem.byteOffset, mem.byteLength);
            const r = {
                status,
                ps: [0, 1, 2, 3].map(i => dv.getFloat32(OUT_PS + i * 4, true)),
                pd: [0, 1].map(i => dv.getFloat64(OUT_PD + i * 8, true)),
                dq: Array.from(mem.subarray(OUT_DQ, OUT_DQ + 16)),
                leaf1_ecx: dv.getUint32(OUT_CPUID + 0, true),
                hv_ebx: dv.getUint32(OUT_CPUID + 4, true),
                ext_max: dv.getUint32(OUT_CPUID + 8, true),
                brand: Buffer.from(mem.subarray(OUT_BRAND, OUT_BRAND + 48))
                    .toString("latin1").replace(/\0.*$/, ""),
            };
            try { emulator.stop(); } catch(e) {}
            resolve(r);
        };
        emulator.bus.register("cpu-event-halt", () => finish("halt"));
        emulator.add_listener("emulator-loaded", () => {
            const cpu = emulator.v86.cpu;
            cpu.reboot_internal(); cpu.reset_memory();
            cpu.load_multiboot(img.buffer);
            timer = setTimeout(() => finish("HANG"), 30000);
            emulator.run();
        });
    });
}

let failures = 0;
const check = (label, actual, expected) => {
    const a = JSON.stringify(actual), e = JSON.stringify(expected);
    const ok = a === e;
    if(!ok) failures++;
    console.log(`${ok ? "  ok  " : "  FAIL"} ${label}: ${a}${ok ? "" : ` (expected ${e})`}`);
};

for(const jit of [false, true]) {
    const r = await run({ jit });
    console.log(`--- ${jit ? "jit" : "interpreter"} (status=${r.status})`);
    check("status", r.status, "halt");
    // A #UD would have aborted before any store, so a correct value here is proof the opcode
    // decoded, executed, and produced SSE3 semantics (odd lanes add, even lanes subtract).
    check("addsubps", r.ps, [9, 22, 27, 44]);
    check("addsubpd", r.pd, [99, 202]);
    check("lddqu", r.dq, UNALIGNED_BYTES);
    check("cpuid.1:ecx SSE3 bit", (r.leaf1_ecx & 1) !== 0, true);
    check("cpuid.1:ecx hypervisor bit", (r.leaf1_ecx >>> 31) & 1, 0);
    check("cpuid.0x40000000:ebx", r.hv_ebx, 0);
    check("cpuid.0x80000000:eax", r.ext_max, 0x80000004);
    check("brand string", r.brand, "Intel(R) Pentium(R) III CPU");
}

console.log(failures === 0 ? "\nALL PASS" : `\n${failures} FAILURE(S)`);
process.exit(failures === 0 ? 0 : 1);
