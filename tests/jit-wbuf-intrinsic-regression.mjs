#!/usr/bin/env node
// WBUF dynarec intrinsic correctness contract:
//   - process reset clears every runtime-owned descriptor/window/counter;
//   - VS, PS, and barrier direct slots retain independent descriptors;
//   - the generated indirect-CALL hit matches the canonical trampoline ABI,
//     especially its PUSHFD/POPFD preservation of incoming EFLAGS.

const { V86 } = await import("../build/libv86.mjs");

const BASE = 0x100000;
const ENTRY = BASE + 0x40;
const TARGET = BASE + 0x1000;
const CTRL = 0x180000;
const STACK = 0x190000;
const DATA = 0x200000;
const FLAGS_BEFORE = 0x170000;
const FLAGS_AFTER = FLAGS_BEFORE + 4;
const RESULT_EAX = FLAGS_BEFORE + 8;
const RESULT_EDX = FLAGS_BEFORE + 12;
const ITER = 400000;
const CAPACITY = 4 * 1024 * 1024;
const MEM_SIZE = 16 * 1024 * 1024;
const TIMEOUT_MS = 20000;

function build_image()
{
    const size = 0x1100;
    const buf = new Uint8Array(size);
    const dv = new DataView(buf.buffer);
    const MAGIC = 0x1BADB002, FLAGS = 0x10000;
    dv.setUint32(0x00, MAGIC, true);
    dv.setUint32(0x04, FLAGS, true);
    dv.setUint32(0x08, (-(MAGIC + FLAGS)) >>> 0, true);
    dv.setUint32(0x0c, BASE, true);
    dv.setUint32(0x10, BASE, true);
    dv.setUint32(0x14, BASE + size, true);
    dv.setUint32(0x18, BASE + 0x4000, true);
    dv.setUint32(0x1c, ENTRY, true);

    let o = ENTRY - BASE;
    const labels = {}, patches = [];
    const label = name => { labels[name] = o; };
    const emit = (...bytes) => { for(const b of bytes) buf[o++] = b & 0xFF; };
    const u32 = value => { dv.setUint32(o, value >>> 0, true); o += 4; };
    const rel32 = name => { patches.push({ at:o, end:o + 4, name }); emit(0, 0, 0, 0); };

    emit(0xBC); u32(STACK);                 // mov esp,STACK
    emit(0xB9); u32(ITER);                  // mov ecx,ITER
    emit(0xBF); u32(TARGET);                // mov edi,TARGET
    label("loop");
    emit(0x68); u32(0x12345678);            // push argument
    emit(0xF9);                             // stc: pin a non-zero incoming flag
    emit(0x9C, 0x5B);                       // pushfd; pop ebx
    emit(0x89, 0x1D); u32(FLAGS_BEFORE);    // mov [before],ebx
    emit(0xFF, 0xD7);                       // call edi (intrinsic candidate)
    emit(0xA3); u32(RESULT_EAX);             // mov [result_eax],eax
    emit(0x89, 0x15); u32(RESULT_EDX);       // mov [result_edx],edx
    emit(0x9C, 0x58);                       // pushfd; pop eax
    emit(0xA3); u32(FLAGS_AFTER);            // mov [after],eax
    emit(0x49);                             // dec ecx
    emit(0x0F, 0x85); rel32("loop");        // jnz loop
    emit(0xF4, 0xEB, 0xFE);                 // hlt; jmp $

    o = TARGET - BASE;
    emit(0x9C);                             // pushfd
    emit(0xB8); u32(0);                     // mov eax,0
    emit(0xBA); u32(0xB077);                // mov edx,0xB077
    emit(0x9D);                             // popfd
    emit(0xC2, 0x04, 0x00);                 // ret 4

    for(const p of patches) dv.setInt32(p.at, labels[p.name] - p.end, true);
    return buf;
}

const fail = message => { throw new Error(message); };

function run()
{
    return new Promise((resolve, reject) => {
        const emulator = new V86({ autostart:false, memory_size:MEM_SIZE, disable_jit:0, log_level:0,
                                   wasm_path: process.env.V86_WASM_PATH || undefined });
        let timer;
        emulator.bus.register("cpu-event-halt", () => {
            clearTimeout(timer);
            try {
                const cpu = emulator.v86.cpu;
                const dv = new DataView(cpu.mem8.buffer, cpu.mem8.byteOffset, cpu.mem8.byteLength);
                resolve({
                    before: dv.getUint32(FLAGS_BEFORE, true),
                    after: dv.getUint32(FLAGS_AFTER, true),
                    eax: dv.getUint32(RESULT_EAX, true),
                    edx: dv.getUint32(RESULT_EDX, true),
                    head: dv.getUint32(CTRL, true),
                    hits: cpu.wm.exports.jit_wbuf_intrinsic_get_hits() >>> 0,
                    fallbacks: cpu.wm.exports.jit_wbuf_intrinsic_get_fallbacks() >>> 0,
                });
            }
            catch(error) { reject(error); }
            finally { try { emulator.stop(); } catch {} }
        });
        emulator.add_listener("emulator-loaded", () => {
            try {
                const cpu = emulator.v86.cpu;
                const ex = cpu.wm.exports;
                cpu.reboot_internal();
                cpu.reset_memory();
                const dv = new DataView(cpu.mem8.buffer, cpu.mem8.byteOffset, cpu.mem8.byteLength);

                // Each direct slot must retain its own snapshot. Updating the backing table
                // afterwards lets execution distinguish a retained hot descriptor (old id)
                // from an accidental eviction followed by an exact-table lookup (new id).
                ex.jit_wbuf_intrinsic_clear_registry();
                ex.jit_wbuf_intrinsic_set_enabled(1);
                for(let slot = 0; slot < 3; slot++) {
                    const target = TARGET + 0x100 + slot * 0x10;
                    if(!ex.jit_wbuf_intrinsic_register(target, 0xA0 + slot, 0, 1, 1, CTRL, DATA, CAPACITY))
                        fail(`registration for hot slot ${slot} failed`);
                    if(!ex.jit_wbuf_intrinsic_mark_hot(slot, target))
                        fail(`mark_hot(${slot}) failed`);
                    ex.jit_wbuf_intrinsic_register(target, 0xB0 + slot, 0, 1, 1, CTRL, DATA, CAPACITY);
                }
                dv.setUint32(STACK, 0xCAFEBABE, true);
                for(let slot = 0; slot < 3; slot++) {
                    dv.setUint32(CTRL, 0, true);
                    const target = TARGET + 0x100 + slot * 0x10;
                    const cleanup = ex.jit_wbuf_intrinsic_execute(target, STACK) | 0;
                    if(cleanup !== 4 || dv.getUint32(DATA, true) !== 0xA0 + slot)
                        fail(`hot slot ${slot} was evicted or aliased`);
                }

                // CR0.PG on with a CR3 that maps nothing: every range the call would touch
                // fails the walk, so it must decline before the ring or the guest stack moves.
                // (tests/jit-wbuf-intrinsic-paging.mjs covers the mapped cases.)
                const cr0 = cpu.cr[0] | 0;
                dv.setUint32(CTRL, 0x20, true);
                dv.setUint32(DATA + 0x20, 0xDEADBEEF, true);
                cpu.cr[0] = cr0 | 0x80000000;
                const pagingResult = ex.jit_wbuf_intrinsic_execute(TARGET + 0x100, STACK) | 0;
                cpu.cr[0] = cr0;
                if(pagingResult !== -1 || dv.getUint32(CTRL, true) !== 0x20
                    || dv.getUint32(DATA + 0x20, true) !== 0xDEADBEEF)
                    fail("paging decline had a guest-memory side effect");

                ex.jit_wbuf_intrinsic_clear_registry();
                if((ex.jit_wbuf_intrinsic_get_enabled() >>> 0) !== 0
                    || (ex.jit_wbuf_intrinsic_get_registered() >>> 0) !== 0
                    || (ex.jit_wbuf_intrinsic_get_min_target() >>> 0) !== 0xFFFFFFFF
                    || (ex.jit_wbuf_intrinsic_get_max_target() >>> 0) !== 0
                    || (ex.jit_wbuf_intrinsic_get_hits() >>> 0) !== 0
                    || (ex.jit_wbuf_intrinsic_get_fallbacks() >>> 0) !== 0)
                    fail("clear_registry left process-owned state behind");

                ex.jit_wbuf_intrinsic_set_enabled(1);
                if(!ex.jit_wbuf_intrinsic_register(TARGET, 0x44, 0, 1, 1, CTRL, DATA, CAPACITY))
                    fail("CALL target registration failed");
                if(!ex.jit_wbuf_intrinsic_mark_hot(0, TARGET)) fail("CALL target hot mark failed");
                dv.setUint32(CTRL, 0, true);
                cpu.jit_clear_cache?.();
                cpu.load_multiboot(build_image().buffer);
                timer = setTimeout(() => reject(new Error("guest did not halt")), TIMEOUT_MS);
                emulator.run();
            }
            catch(error) { reject(error); }
        });
    });
}

const result = await run();
console.log("jit-wbuf-intrinsic " + JSON.stringify(result));
if(result.hits === 0) fail("generated indirect CALL never hit the intrinsic");
if(result.before !== result.after) fail(`EFLAGS changed: 0x${result.before.toString(16)} -> 0x${result.after.toString(16)}`);
if(result.eax !== 0 || result.edx !== 0xB077) fail("trampoline register ABI mismatch");
if(result.head === 0) fail("intrinsic hit did not append to the ring");
console.log("  ok — clear/reset, three direct slots, and EFLAGS ABI match");
