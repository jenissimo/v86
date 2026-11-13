#!/usr/bin/env node
// Differential for the WBUF intrinsic switches set_jit_wbuf_tail_jmp (the `jmp [IAT]` tail of
// an import thunk) and set_jit_wbuf_full_range (the indirect-CALL guard over the whole registry).
//
// The guest stubs write the ring themselves, the way the BottleShip trampoline does: funcId,
// then the arguments, head advanced by the stride, an overflow path when the ring is full,
// EAX = 0 / EDX = 0xB077 under PUSHFD/POPFD and RET N. The guest runs with paging on and
// goes through these cases, each in a hot loop:
//   A  call thunk; thunk: jmp [IAT]                    scalar stub, two stdcall arguments
//   Z  the same into a no-arguments stub (kind 2)
//   T  MSVC-style tail call: push ebp; ...; pop ebp; jmp [IAT]
//   U  jmp [IAT] into an unregistered stub inside the registered span (probe miss)
//   P  call [IAT] into a registered stub that is not a hot slot (the CALL guard)
//   F  a ring that is full after eight entries                         (must decline)
//   H  the arguments run into a page that is not present: the stub raises the #PF itself,
//      the handler maps the page and the stub retries                       (must decline)
//   S  a 16-bit stack whose SP wraps under the arguments                    (must decline)
//   C  a one-entry ring inside the page of the compiled caller, emptied every 8192 calls and
//      full in between; the entry's last argument patches the immediate of a
//      `mov eax, imm32` on that page, which the guest calls afterwards     (must decline)
//
// Oracles: the interpreter and the JIT with both switches off. Every arm must leave the same
// rings, heads, overflow counter, #PF records, per-case ESP/EFLAGS/EAX/EDX and final registers.
// Negative controls, both switches on: a registration with the wrong funcId must change the
// ring, one without the stdcall cleanup must change ESP.
//
//   node tests/jit-wbuf-intrinsic-tail-jmp.mjs   [V86_WASM_PATH=<candidate v86.wasm>]

import crypto from "node:crypto";

const { V86 } = await import(process.env.LIB || "../build/libv86.mjs");

const BASE = 0x100000;
const ENTRY = BASE + 0x40;
const IMG_SIZE = 0x8000;
const THUNKS = BASE + 0x2000;
const STUBS = BASE + 0x3000;
const IAT = BASE + 0x4000;
const GDT = BASE + 0x4800;
const IDT = BASE + 0x4C00;
const PD_ADDR = 0x108000, PT0_ADDR = 0x109000;

const STATE = 0x180000;
const CTRL_MAIN = STATE, CTRL_FULL = STATE + 0x10, CTRL_CODE = STATE + 0x20;
const OVERFLOW = STATE + 0x30, PF_COUNT = STATE + 0x34, PF_CR2 = STATE + 0x38, PF_EIP = STATE + 0x3C;
const SAVE_ESP = STATE + 0x40;
const OUT = STATE + 0x100;
const STATE_END = STATE + 0x300;
const STACK = 0x190000;
const HOLE = 0x1A1000;
const STACK16_BASE = 0x50000, STACK16_ESP = 0x10004;
const DATA_MAIN = 0x200000, CAP_MAIN = 0x1C0000;
const DATA_FULL = 0x3E0000, CAP_FULL = 64;
const GETVAL = BASE + 0xF00;
const DATA_CODE = GETVAL + 1 - 28, CAP_CODE = 32;

const ID_A = 0x44, ID_Z = 0x55, ID_F = 0x66, ID_C = 0x77;
const ITER = 30000, ITER_F = 2000, ITER_C = 50000, ITER_H = 1500, ITER_S = 20000;
const MEM_SIZE = 16 * 1024 * 1024;
const TIMEOUT_MS = 60000;

const STUB = {};

function build_image()
{
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
    dv.setUint32(0x1c, ENTRY, true);

    let o = 0;
    const labels = {}, patches = [];
    const at = addr => { o = addr - BASE; };
    const label = name => { labels[name] = o; };
    const emit = (...bytes) => { for(const b of bytes) buf[o++] = b & 0xFF; };
    const u32 = value => { dv.setUint32(o, value >>> 0, true); o += 4; };
    const rel32 = name => { patches.push({ at: o, end: o + 4, name, size: 4 }); o += 4; };
    const rel8 = name => { patches.push({ at: o, end: o + 1, name, size: 1 }); o += 1; };

    // A stub that writes the ring like the BottleShip trampoline.
    const stub = (name, func_id, nargs, ctrl, data, cap, stdcall) => {
        label(name);
        STUB[name] = BASE + o;
        const stride = nargs === 0 ? 8 : (nargs + 1) * 4;
        emit(0x9C, 0x53, 0x51);                          // pushfd; push ebx; push ecx
        emit(0x8B, 0x1D); u32(ctrl);                     // mov ebx,[ctrl]
        emit(0x8D, 0x8B); u32(stride);                   // lea ecx,[ebx+stride]
        emit(0x81, 0xF9); u32(cap);                      // cmp ecx,cap
        emit(0x77); rel8(name + "_full");                // ja full
        emit(0xC7, 0x83); u32(data); u32(func_id);       // mov dword [ebx+data],func_id
        if(nargs === 0) {
            emit(0xC7, 0x83); u32(data + 4); u32(0);     // mov dword [ebx+data+4],0
        }
        for(let i = 0; i < nargs; i++) {
            emit(0x8B, 0x44, 0x24, 16 + 4 * i);          // mov eax,[esp+16+4i]
            emit(0x89, 0x83); u32(data + 4 + 4 * i);     // mov [ebx+data+4+4i],eax
        }
        emit(0x89, 0x0D); u32(ctrl);                     // mov [ctrl],ecx
        emit(0xEB); rel8(name + "_done");                // jmp done
        label(name + "_full");
        emit(0xFF, 0x05); u32(OVERFLOW);                 // inc dword [OVERFLOW]
        label(name + "_done");
        emit(0x59, 0x5B);                                // pop ecx; pop ebx
        emit(0xB8); u32(0);                              // mov eax,0
        emit(0xBA); u32(0xB077);                         // mov edx,0xB077
        emit(0x9D);                                      // popfd
        if(stdcall && nargs) emit(0xC2, nargs * 4, 0);   // ret n
        else emit(0xC3);                                 // ret
    };

    // Records ESP, EFLAGS, EAX, EDX of the case into OUT[slot..slot+3].
    const record = slot => {
        emit(0x89, 0x25); u32(OUT + slot * 4);           // mov [out],esp
        emit(0x89, 0x05); u32(OUT + slot * 4 + 8);       // mov [out+8],eax
        emit(0x89, 0x15); u32(OUT + slot * 4 + 12);      // mov [out+12],edx
        emit(0x9C, 0x58);                                // pushfd; pop eax
        emit(0xA3); u32(OUT + slot * 4 + 4);             // mov [out+4],eax
    };
    const loop_head = (name, count) => {
        emit(0xBE); u32(count);                          // mov esi,count
        label(name);
    };
    const loop_tail = name => {
        emit(0xBC); u32(STACK);                          // mov esp,STACK  (contains a wrong cleanup)
        emit(0x4E);                                      // dec esi
        emit(0x0F, 0x85); rel32(name);                   // jnz name
    };
    const set_flags = () => {
        emit(0x31, 0xC0);                                // xor eax,eax  (ZF=1, PF=1)
        emit(0xF9);                                      // stc
        emit(0xFD);                                      // std
    };

    at(ENTRY);
    emit(0xFA);                                          // cli
    emit(0x0F, 0x01, 0x15); u32(GDT + 0x40);             // lgdt [gdtr]
    emit(0xEA); u32(BASE + 0x200); emit(0x08, 0x00);     // jmp 0x08:reload
    at(BASE + 0x200);
    emit(0x66, 0xB8, 0x10, 0x00);                        // mov ax,0x10
    emit(0x8E, 0xD8, 0x8E, 0xC0, 0x8E, 0xD0);            // mov ds,ax; mov es,ax; mov ss,ax
    emit(0xBC); u32(STACK);                              // mov esp,STACK
    emit(0x0F, 0x01, 0x1D); u32(IDT + 0x100);            // lidt [idtr]
    emit(0xFC);                                          // cld
    emit(0xBF); u32(PT0_ADDR);                           // mov edi,PT0
    emit(0xB8); u32(3);                                  // mov eax,0|P|RW
    emit(0xB9); u32(0x400);                              // mov ecx,1024
    const fill = o;
    emit(0xAB);                                          // stosd
    emit(0x05); u32(0x1000);                             // add eax,0x1000
    emit(0xE2, (fill - (o + 2)) & 0xFF);                 // loop fill
    emit(0xBF); u32(PD_ADDR);                            // mov edi,PD
    emit(0x31, 0xC0);                                    // xor eax,eax
    emit(0xB9); u32(0x400);                              // mov ecx,1024
    emit(0xF3, 0xAB);                                    // rep stosd
    emit(0xC7, 0x05); u32(PD_ADDR); u32(PT0_ADDR | 3);
    emit(0xB8); u32(PD_ADDR);                            // mov eax,PD
    emit(0x0F, 0x22, 0xD8);                              // mov cr3,eax
    emit(0x0F, 0x20, 0xC0);                              // mov eax,cr0
    emit(0x0D); u32(0x80000000);                         // or eax,PG
    emit(0x0F, 0x22, 0xC0);                              // mov cr0,eax

    // A, which also warms thunk_c up while its ring is full, so the thunk page is compiled
    // with both entries before C starts
    emit(0xC7, 0x05); u32(CTRL_CODE); u32(CAP_CODE);     // mov dword [CTRL_CODE],CAP_CODE
    loop_head("la", ITER);
    emit(0x56);                                          // push esi
    emit(0x68); u32(0xA0000000);                         // push imm
    set_flags();
    emit(0xE8); rel32("thunk_a");                        // call thunk_a
    record(0);
    for(let i = 0; i < 7; i++) { emit(0x68); u32(0xD0000000 + i); }
    emit(0xE8); rel32("thunk_c");                        // call thunk_c
    emit(0xFC);
    loop_tail("la");
    // Z
    loop_head("lz", ITER);
    set_flags();
    emit(0xE8); rel32("thunk_z");                        // call thunk_z
    record(4);
    emit(0xFC);
    loop_tail("lz");
    // T
    loop_head("lt", ITER);
    emit(0x56);                                          // push esi
    emit(0x68); u32(0x7A000000);                         // push imm
    emit(0xE8); rel32("tailfn");                         // call tailfn
    record(8);
    emit(0xFC);
    loop_tail("lt");
    // U
    loop_head("lu", ITER);
    emit(0x56);                                          // push esi
    emit(0xE8); rel32("thunk_u");                        // call thunk_u
    record(12);
    loop_tail("lu");
    // P
    loop_head("lp", ITER);
    emit(0x56);                                          // push esi
    emit(0x68); u32(0xCA000000);                         // push imm
    set_flags();
    emit(0xFF, 0x15); u32(IAT);                          // call [IAT_A]
    record(16);
    emit(0xFC);
    loop_tail("lp");
    // F
    loop_head("lf", ITER_F);
    emit(0x56);                                          // push esi
    emit(0xE8); rel32("thunk_f");                        // call thunk_f
    record(20);
    loop_tail("lf");
    // H
    emit(0xC7, 0x05); u32(HOLE); u32(0x40404040);        // arguments in the hole page
    emit(0xC7, 0x05); u32(HOLE + 4); u32(0x41414141);
    loop_head("lh", ITER_H);
    emit(0xC7, 0x05); u32(PT0_ADDR + (HOLE >>> 12) * 4); u32(0); // unmap
    emit(0x0F, 0x01, 0x3D); u32(HOLE);                   // invlpg [HOLE]
    emit(0x89, 0x25); u32(SAVE_ESP);                     // mov [SAVE_ESP],esp
    emit(0xBC); u32(HOLE);                               // mov esp,HOLE
    set_flags();
    emit(0xE8); rel32("thunk_a");                        // call thunk_a
    record(28);
    emit(0xFC);
    emit(0x8B, 0x25); u32(SAVE_ESP);                     // mov esp,[SAVE_ESP]
    loop_tail("lh");
    // S
    loop_head("ls", ITER_S);
    emit(0x89, 0x25); u32(SAVE_ESP);                     // mov [SAVE_ESP],esp
    emit(0x66, 0xB8, 0x18, 0x00);                        // mov ax,0x18
    emit(0x8E, 0xD0);                                    // mov ss,ax
    emit(0xBC); u32(STACK16_ESP);                        // mov esp,0x10004: SP wraps, ESP[31:16] stays
    emit(0xE8); rel32("spin");                           // call spin
    emit(0x56);                                          // push esi
    emit(0x68); u32(0x16000000);                         // push imm
    emit(0xE8); rel32("thunk_a");                        // call thunk_a
    emit(0x89, 0x25); u32(OUT + 32 * 4);                 // mov [out],esp
    emit(0x66, 0xB8, 0x10, 0x00);                        // mov ax,0x10
    emit(0x8E, 0xD0);                                    // mov ss,ax
    emit(0x8B, 0x25); u32(SAVE_ESP);                     // mov esp,[SAVE_ESP]
    loop_tail("ls");

    // C
    loop_head("lc", ITER_C);
    emit(0x89, 0xF0);                                    // mov eax,esi
    emit(0x25); u32(8191);                               // and eax,8191
    emit(0x75, 10);                                      // jnz +10
    emit(0xC7, 0x05); u32(CTRL_CODE); u32(0);            // mov dword [CTRL_CODE],0
    emit(0x56);                                          // push esi        (a6 -> imm32 of getval)
    emit(0x68); u32(0xB8000000);                         // push imm        (a5 -> its opcode byte)
    for(let i = 4; i >= 0; i--) { emit(0x68); u32(0xC0000000 + i); }
    emit(0xE8); rel32("thunk_c");                        // call thunk_c
    emit(0xE8); rel32("getval");                         // call getval
    emit(0xA3); u32(OUT + 24 * 4);                       // mov [out],eax
    emit(0x89, 0x25); u32(OUT + 25 * 4);                 // mov [out],esp
    loop_tail("lc");

    emit(0xF4, 0xEB, 0xFE);                              // hlt; jmp $

    // #PF handler: record, map the hole page, retry.
    label("pf");
    const pf = BASE + o;
    emit(0x50);                                          // push eax
    emit(0x0F, 0x20, 0xD0);                              // mov eax,cr2
    emit(0xA3); u32(PF_CR2);
    emit(0x8B, 0x44, 0x24, 0x08);                        // mov eax,[esp+8]  (faulting EIP)
    emit(0xA3); u32(PF_EIP);
    emit(0xFF, 0x05); u32(PF_COUNT);
    emit(0xC7, 0x05); u32(PT0_ADDR + (HOLE >>> 12) * 4); u32(HOLE | 3);
    emit(0x0F, 0x01, 0x3D); u32(HOLE);                   // invlpg [HOLE]
    emit(0x58);                                          // pop eax
    emit(0x83, 0xC4, 0x04);                              // add esp,4 (error code)
    emit(0xCF);                                          // iret

    // MSVC-style function ending in a tail jump through the IAT.
    label("tailfn");
    emit(0x55, 0x89, 0xE5);                              // push ebp; mov ebp,esp
    emit(0x8B, 0x45, 0x08);                              // mov eax,[ebp+8]
    emit(0x01, 0xC0);                                    // add eax,eax
    emit(0xA3); u32(OUT + 36 * 4);
    emit(0x5D);                                          // pop ebp
    set_flags();
    emit(0xFF, 0x25); u32(IAT);                          // jmp [IAT_A]

    at(THUNKS);
    label("thunk_a"); emit(0xFF, 0x25); u32(IAT); emit(0x8B, 0xC0);
    at(THUNKS + 0x10);
    label("thunk_z"); emit(0xFF, 0x25); u32(IAT + 4); emit(0x8B, 0xC0);
    at(THUNKS + 0x20);
    label("thunk_u"); emit(0xFF, 0x25); u32(IAT + 8); emit(0x8B, 0xC0);
    at(THUNKS + 0x30);
    label("thunk_f"); emit(0xFF, 0x25); u32(IAT + 12); emit(0x8B, 0xC0);

    // Heats the thunk page under the 16-bit stack, so it is compiled under those flags too.
    at(THUNKS + 0x80);
    label("spin");
    emit(0xB9); u32(20);                                 // mov ecx,20
    emit(0x49, 0x75, 0xFD);                              // dec ecx; jnz $-1
    emit(0xC3);                                          // ret
    at(STUBS);
    stub("stub_a", ID_A, 2, CTRL_MAIN, DATA_MAIN, CAP_MAIN, true);
    at(STUBS + 0x100);
    label("stub_u");
    STUB.stub_u = BASE + o;
    emit(0xB8); u32(0x0BAD0000);                         // mov eax,imm
    emit(0xC2, 0x04, 0x00);                              // ret 4
    at(STUBS + 0x200);
    stub("stub_f", ID_F, 1, CTRL_FULL, DATA_FULL, CAP_FULL, true);
    at(STUBS + 0x300);
    stub("stub_c", ID_C, 7, CTRL_CODE, DATA_CODE, CAP_CODE, true);
    at(STUBS + 0x400);
    stub("stub_z", ID_Z, 0, CTRL_MAIN, DATA_MAIN, CAP_MAIN, true);

    at(THUNKS + 0x40);
    label("thunk_c"); emit(0xFF, 0x25); u32(IAT + 16); emit(0x8B, 0xC0);
    at(GETVAL);
    label("getval"); emit(0xB8); u32(0); emit(0xC3);     // mov eax,imm32; ret

    dv.setUint32(IAT - BASE, STUB.stub_a, true);
    dv.setUint32(IAT + 4 - BASE, STUB.stub_z, true);
    dv.setUint32(IAT + 8 - BASE, STUB.stub_u, true);
    dv.setUint32(IAT + 12 - BASE, STUB.stub_f, true);
    dv.setUint32(IAT + 16 - BASE, STUB.stub_c, true);

    const gdt = [
        [0, 0],
        [0x0000FFFF, 0x00CF9A00],
        [0x0000FFFF, 0x00CF9200],
        [0x0000FFFF | (STACK16_BASE & 0xFFFF) << 16, (STACK16_BASE >>> 16 & 0xFF) | 0x00009200 | (STACK16_BASE >>> 24) << 24],
    ];
    gdt.forEach(([lo, hi], i) => { dv.setUint32(GDT - BASE + i * 8, lo >>> 0, true); dv.setUint32(GDT - BASE + i * 8 + 4, hi >>> 0, true); });
    dv.setUint16(GDT - BASE + 0x40, gdt.length * 8 - 1, true);
    dv.setUint32(GDT - BASE + 0x42, GDT, true);
    dv.setUint32(IDT - BASE + 14 * 8, (0x0008 << 16 | pf & 0xFFFF) >>> 0, true);
    dv.setUint32(IDT - BASE + 14 * 8 + 4, (pf & 0xFFFF0000 | 0x8E00) >>> 0, true);
    dv.setUint16(IDT - BASE + 0x100, 15 * 8 - 1, true);
    dv.setUint32(IDT - BASE + 0x102, IDT, true);

    for(const p of patches) {
        const rel = labels[p.name] - p.end;
        if(p.size === 1) {
            if(rel < -128 || rel > 127) throw new Error("rel8 out of range: " + p.name);
            dv.setInt8(p.at, rel);
        }
        else dv.setInt32(p.at, rel, true);
    }
    return buf;
}

const fail = message => { throw new Error(message); };

function register(ex, { wrong_id = false, no_cleanup = false } = {})
{
    const ok = [
        ex.jit_wbuf_intrinsic_register(STUB.stub_a, wrong_id ? ID_A + 1 : ID_A, 0, 2, no_cleanup ? 0 : 1, CTRL_MAIN, DATA_MAIN, CAP_MAIN),
        ex.jit_wbuf_intrinsic_register(STUB.stub_f, ID_F, 0, 1, 1, CTRL_FULL, DATA_FULL, CAP_FULL),
        ex.jit_wbuf_intrinsic_register(STUB.stub_c, ID_C, 0, 7, 1, CTRL_CODE, DATA_CODE, CAP_CODE),
        ex.jit_wbuf_intrinsic_register(STUB.stub_z, ID_Z, 2, 0, 1, CTRL_MAIN, DATA_MAIN, CAP_MAIN),
    ];
    if(ok.some(x => !x)) fail("registration failed: " + ok.join(","));
    if(ex.jit_wbuf_intrinsic_register(STUB.stub_z, ID_Z, 2, 1, 1, CTRL_MAIN, DATA_MAIN, CAP_MAIN))
        fail("no-args kind accepted an argument count");
}

function run({ jit, tail = 0, full = 0, negative = {} })
{
    return new Promise((resolve, reject) => {
        const emulator = new V86({ autostart: false, memory_size: MEM_SIZE, disable_jit: jit ? 0 : 1, log_level: 0,
                                   wasm_path: process.env.V86_WASM_PATH || undefined });
        let timer;
        const finish = status => {
            clearTimeout(timer);
            try {
                const cpu = emulator.v86.cpu;
                const ex = cpu.wm.exports;
                const mem = cpu.mem8;
                const dv = new DataView(mem.buffer, mem.byteOffset, mem.byteLength);
                const head = dv.getUint32(CTRL_MAIN, true);
                const h = crypto.createHash("sha256");
                h.update(mem.subarray(STATE, STATE_END));
                h.update(mem.subarray(DATA_MAIN, DATA_MAIN + Math.min(head + 64, CAP_MAIN)));
                h.update(mem.subarray(DATA_FULL, DATA_FULL + 2 * CAP_FULL));
                h.update(mem.subarray(GETVAL - 0x40, GETVAL + 0x40));
                h.update(mem.subarray(STACK16_BASE, STACK16_BASE + 0x40));
                h.update(mem.subarray(STACK16_BASE + 0xFFC0, STACK16_BASE + 0x10000));
                for(const r of [0, 1, 2, 3, 4, 5, 6, 7]) h.update(String(cpu.reg32[r] >>> 0) + ",");
                resolve({
                    status,
                    digest: h.digest("hex").slice(0, 16),
                    state: mem.slice(STATE, STATE_END),
                    head,
                    overflow: dv.getUint32(OVERFLOW, true),
                    pf: dv.getUint32(PF_COUNT, true),
                    hits: ex.jit_wbuf_intrinsic_get_hits() >>> 0,
                    fallbacks: ex.jit_wbuf_intrinsic_get_fallbacks() >>> 0,
                    misses: typeof ex.jit_wbuf_intrinsic_get_probe_misses === "function" ? ex.jit_wbuf_intrinsic_get_probe_misses() >>> 0 : -1,
                });
            }
            catch(error) { reject(error); }
            finally { try { emulator.stop(); } catch {} }
        };
        emulator.bus.register("cpu-event-halt", () => finish("halt"));
        emulator.add_listener("emulator-loaded", () => {
            try {
                const cpu = emulator.v86.cpu;
                const ex = cpu.wm.exports;
                if(typeof ex.set_jit_wbuf_tail_jmp !== "function" || typeof ex.set_jit_wbuf_full_range !== "function")
                    fail("this build has no set_jit_wbuf_tail_jmp / set_jit_wbuf_full_range");
                cpu.reboot_internal();
                cpu.reset_memory();
                const image = build_image();
                ex.jit_wbuf_intrinsic_clear_registry();
                ex.jit_wbuf_intrinsic_set_enabled(1);
                register(ex, negative);
                ex.set_jit_wbuf_tail_jmp(tail);
                ex.set_jit_wbuf_full_range(full);
                ex.jit_wbuf_intrinsic_reset_stats();
                cpu.jit_clear_cache?.();
                cpu.load_multiboot(image.buffer);
                timer = setTimeout(() => finish("HANG"), TIMEOUT_MS);
                emulator.run();
            }
            catch(error) { reject(error); }
        });
    });
}

process.on("uncaughtException", error => {
    console.error(`jit-wbuf-intrinsic-tail-jmp: FAIL: ${error}`);
    process.exit(1);
});

const first_diff = (a, b) => {
    for(let i = 0; i < a.length; i += 4) {
        const x = a[i] | a[i + 1] << 8 | a[i + 2] << 16 | a[i + 3] << 24;
        const y = b[i] | b[i + 1] << 8 | b[i + 2] << 16 | b[i + 3] << 24;
        if(x !== y) return `0x${(STATE + i).toString(16)}: ${(x >>> 0).toString(16)} vs ${(y >>> 0).toString(16)}`;
    }
    return "state words equal";
};

const arms = [
    ["interpreter", { jit: false }],
    ["jit off", { jit: true }],
    ["tail jmp", { jit: true, tail: 1 }],
    ["full range", { jit: true, full: 1 }],
    ["both", { jit: true, tail: 1, full: 1 }],
];
const res = {};
for(const [name, cfg] of arms) {
    res[name] = await run(cfg);
    const r = res[name];
    console.log(`  ${name.padEnd(12)} ${r.status} ${r.digest} head=${r.head} overflow=${r.overflow} pf=${r.pf} hits=${r.hits} fallbacks=${r.fallbacks} probe_misses=${r.misses}`);
}
const oracle = res.interpreter;
if(oracle.status !== "halt") fail("interpreter did not halt");
if(oracle.pf !== ITER_H) fail(`oracle took ${oracle.pf} #PF, expected ${ITER_H}`);
if(oracle.overflow === 0) fail("oracle never filled the ring");
for(const [name] of arms) {
    const r = res[name];
    if(r.status !== "halt" || r.digest !== oracle.digest)
        fail(`${name} differs from the interpreter: ${r.status}, ${first_diff(r.state, oracle.state)}`);
}
if(res["jit off"].hits !== 0) fail("switches off, yet the intrinsic hit");
if(res["tail jmp"].hits === 0) fail("tail jmp on never hit");
if(res["full range"].hits === 0) fail("full range on never hit a non-hot CALL target");
const both = res.both;
if(both.fallbacks === 0) fail("no decline was exercised");
if(both.misses === 0) fail("the unregistered stub inside the span was not counted as a probe miss");

// Counter contract, through the helper exports directly: a registered target that declines
// counts as a fallback, an unregistered target inside the span as a probe miss, with and
// without an armed debugger breakpoint (which declines registered targets only).
function counters()
{
    return new Promise((resolve, reject) => {
        const emulator = new V86({ autostart: false, memory_size: MEM_SIZE, disable_jit: 0, log_level: 0,
                                   wasm_path: process.env.V86_WASM_PATH || undefined });
        emulator.add_listener("emulator-loaded", () => {
            try {
                const cpu = emulator.v86.cpu;
                const ex = cpu.wm.exports;
                cpu.reboot_internal();
                cpu.reset_memory();
                const dv = new DataView(cpu.mem8.buffer, cpu.mem8.byteOffset, cpu.mem8.byteLength);
                const REG = 0x101000, UNREG = 0x101010, FULL = 0x101020, ESP = 0x190000 - 16;
                ex.jit_wbuf_intrinsic_clear_registry();
                ex.jit_wbuf_intrinsic_set_enabled(1);
                if(!ex.jit_wbuf_intrinsic_register(REG, 0x11, 0, 1, 1, CTRL_MAIN, DATA_MAIN, CAP_MAIN)) fail("register");
                if(!ex.jit_wbuf_intrinsic_register(FULL, 0x12, 0, 1, 1, CTRL_FULL, DATA_FULL, CAP_FULL)) fail("register");
                dv.setUint32(CTRL_FULL, CAP_FULL, true);
                dv.setUint32(ESP, 0x1234, true);
                dv.setUint32(ESP + 4, 0x5678, true);
                const read = () => [ex.jit_wbuf_intrinsic_get_hits() >>> 0, ex.jit_wbuf_intrinsic_get_fallbacks() >>> 0,
                                    ex.jit_wbuf_intrinsic_get_probe_misses() >>> 0];
                const step = (fn, target, expectRet, expectDelta, label) => {
                    const before = read();
                    const head = dv.getUint32(CTRL_MAIN, true);
                    const ret = fn === "tail" ? ex.jit_wbuf_intrinsic_execute_tail(target, ESP) | 0
                                              : ex.jit_wbuf_intrinsic_execute(target, ESP + 4) | 0;
                    const after = read();
                    const delta = after.map((v, i) => v - before[i]);
                    if(ret !== expectRet || delta.join() !== expectDelta.join())
                        fail(`counter contract, ${label} (${fn}): returned ${ret}, hits/fallbacks/misses delta ${delta}`);
                    if(expectRet < 0 && dv.getUint32(CTRL_MAIN, true) !== head) fail(`counter contract, ${label}: declined call moved the ring`);
                };
                for(const fn of ["call", "tail"]) {
                    ex.dbg_clear(); ex.dbg_enable(0);
                    step(fn, REG, 4, [1, 0, 0], "registered, no debugger");
                    step(fn, UNREG, -2, [0, 0, 1], "unregistered, no debugger");
                    step(fn, FULL, -1, [0, 1, 0], "registered, ring full");
                    ex.dbg_enable(1); ex.dbg_add_bp(0x7000);
                    step(fn, REG, -1, [0, 1, 0], "registered, breakpoint armed");
                    step(fn, UNREG, -2, [0, 0, 1], "unregistered, breakpoint armed");
                    ex.dbg_clear(); ex.dbg_enable(0);
                }
                resolve();
            }
            catch(error) { reject(error); }
            finally { try { emulator.destroy?.(); } catch {} }
        });
    });
}
await counters();
console.log("  counter contract: fallbacks = registered and declined, probe misses = unregistered, with and without a breakpoint");

const neg = [
    ["wrong funcId", { wrong_id: true }],
    ["no cleanup", { no_cleanup: true }],
];
for(const [name, negative] of neg) {
    const r = await run({ jit: true, tail: 1, full: 1, negative });
    console.log(`  ${("neg " + name).padEnd(16)} ${r.status} ${r.digest} hits=${r.hits}`);
    if(r.status === "halt" && r.digest === oracle.digest) fail(`negative control "${name}" did not diverge`);
}
console.log("jit-wbuf-intrinsic-tail-jmp: ok");
process.exit(0);
