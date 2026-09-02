#!/usr/bin/env node
// WBUF dynarec intrinsic under x86 PAGING — the state every BottleShip guest actually runs in.
//   (a) the generated indirect CALL still hits while CR0.PG is set and CR3 maps identity;
//   (b) a shader-constant block whose SECOND page is not present is DECLINED with zero
//       guest-visible side effects, so the guest's own CALL raises the real #PF.
// (b) is the assertion that separates a paging-aware check from both a blanket refusal
// (which never hits at all) and a bounds-only check (which copies the absent page's
// physical bytes and advances the ring).
//
//   node tests/jit-wbuf-intrinsic-paging.mjs   [V86_WASM_PATH=<candidate v86.wasm>]

const { V86 } = await import("../build/libv86.mjs");

const BASE = 0x100000;
const ENTRY = BASE + 0x40;
const TARGET = BASE + 0x1000;      // canonical trampoline, registered as a scalar WBUF stub
const CONST_TARGET = TARGET + 0x100;
const PD_ADDR = 0x108000, PT0_ADDR = 0x109000;
const CTRL = 0x180000;
const STACK = 0x190000;
const DATA = 0x200000, CAPACITY = 0x100000;
const ARGS = 0x310000;             // synthetic pre-CALL frame for the JS-driven calls
const CONST_OK = 0x320FF0;         // spans 0x320/0x321 — both present
const CONST_BAD = 0x330FF0;        // spans 0x330 (present) and HOLE_PAGE
const HOLE_PAGE = 0x331000;
const VEC4 = 16, PAYLOAD = VEC4 * 16, STRIDE = 16 + PAYLOAD;
const ITER = 100000;
const MEM_SIZE = 16 * 1024 * 1024;
const TIMEOUT_MS = 30000;

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

    // Identity-map 0..4MB with 4KB pages, then punch out exactly one page.
    emit(0xFC);                                    // cld
    emit(0xBF); u32(PT0_ADDR);                     // mov edi,PT0
    emit(0xB8); u32(0x00000003);                   // mov eax,0|P|RW
    emit(0xB9); u32(0x400);                        // mov ecx,1024
    const fill = o;
    emit(0xAB);                                    // stosd
    emit(0x05); u32(0x1000);                       // add eax,0x1000
    emit(0xE2, (fill - (o + 2)) & 0xFF);           // loop fill
    emit(0xC7, 0x05); u32(PT0_ADDR + (HOLE_PAGE >>> 12) * 4); u32(0); // PTE[hole] <- not present
    emit(0xBF); u32(PD_ADDR);                      // mov edi,PD
    emit(0x31, 0xC0);                              // xor eax,eax
    emit(0xB9); u32(0x400);                        // mov ecx,1024
    emit(0xF3, 0xAB);                              // rep stosd
    emit(0xC7, 0x05); u32(PD_ADDR); u32((PT0_ADDR | 3) >>> 0);
    emit(0xB8); u32(PD_ADDR);                      // mov eax,PD
    emit(0x0F, 0x22, 0xD8);                        // mov cr3,eax
    emit(0x0F, 0x20, 0xC0);                        // mov eax,cr0
    emit(0x0D); u32(0x80000000);                   // or eax,0x80000000
    emit(0x0F, 0x22, 0xC0);                        // mov cr0,eax

    emit(0xBC); u32(STACK);                        // mov esp,STACK
    emit(0xB9); u32(ITER);                         // mov ecx,ITER
    emit(0xBF); u32(TARGET);                       // mov edi,TARGET
    label("loop");
    emit(0x68); u32(0x12345678);                   // push arg
    emit(0xFF, 0xD7);                              // call edi
    emit(0x49);                                    // dec ecx
    emit(0x0F, 0x85); rel32("loop");               // jnz loop
    emit(0xF4, 0xEB, 0xFE);                        // hlt; jmp $

    o = TARGET - BASE;
    emit(0x9C);                                    // pushfd
    emit(0xB8); u32(0);                            // mov eax,0
    emit(0xBA); u32(0xB077);                       // mov edx,0xB077
    emit(0x9D);                                    // popfd
    emit(0xC2, 0x04, 0x00);                        // ret 4

    for(const p of patches) dv.setInt32(p.at, labels[p.name] - p.end, true);
    return buf;
}

const fail = message => { throw new Error(message); };

function run()
{
    return new Promise((resolve, reject) => {
        const emulator = new V86({
            autostart: false, memory_size: MEM_SIZE, disable_jit: 0, log_level: 0,
            wasm_path: process.env.V86_WASM_PATH || undefined,
        });
        let timer;
        emulator.bus.register("cpu-event-halt", () => {
            clearTimeout(timer);
            try { resolve(after_halt(emulator)); }
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

                ex.jit_wbuf_intrinsic_clear_registry();
                ex.jit_wbuf_intrinsic_set_enabled(1);
                if(!ex.jit_wbuf_intrinsic_register(TARGET, 0x44, 0, 1, 1, CTRL, DATA, CAPACITY))
                    fail("scalar registration failed");
                if(!ex.jit_wbuf_intrinsic_mark_hot(0, TARGET)) fail("scalar mark_hot failed");
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

// Phase 2 runs with the guest's own paging state still live: CR0.PG set, CR3 pointing at the
// identity map the guest built, CPL 0. Driving the helper directly is what lets the ring be
// inspected byte-exactly around a decline.
function after_halt(emulator)
{
    const cpu = emulator.v86.cpu;
    const ex = cpu.wm.exports;
    const dv = new DataView(cpu.mem8.buffer, cpu.mem8.byteOffset, cpu.mem8.byteLength);
    const result = {
        paging: (cpu.cr[0] >>> 0) & 0x80000000 ? 1 : 0,
        hits_under_paging: ex.jit_wbuf_intrinsic_get_hits() >>> 0,
        fallbacks_under_paging: ex.jit_wbuf_intrinsic_get_fallbacks() >>> 0,
        ring_head_after_guest: dv.getUint32(CTRL, true),
    };

    if(!ex.jit_wbuf_intrinsic_register(CONST_TARGET, 0x55, 1, 4, 1, CTRL, DATA, CAPACITY))
        fail("shader-constant registration failed");
    if(!ex.jit_wbuf_intrinsic_mark_hot(1, CONST_TARGET)) fail("shader-constant mark_hot failed");

    // Both blocks exist PHYSICALLY; only CONST_BAD's second page is absent from the page table.
    for(let i = 0; i < PAYLOAD; i += 4) {
        dv.setUint32(CONST_OK + i, 0xC0DE0000 + i, true);
        dv.setUint32(CONST_BAD + i, 0xBAD00000 + i, true);
    }
    const frame = data_ptr => {
        dv.setUint32(ARGS + 0, 0x11112222, true);
        dv.setUint32(ARGS + 4, 7, true);
        dv.setUint32(ARGS + 8, data_ptr, true);
        dv.setUint32(ARGS + 12, VEC4, true);
    };

    // (b1) positive control: a two-page block with both pages present must still be encoded.
    const HEAD_OK = 0x1000;
    dv.setUint32(CTRL, HEAD_OK, true);
    frame(CONST_OK);
    result.ok_cleanup = ex.jit_wbuf_intrinsic_execute(CONST_TARGET, ARGS) | 0;
    result.ok_head = dv.getUint32(CTRL, true);
    result.ok_first_word = dv.getUint32(DATA + HEAD_OK + 16, true);
    result.ok_last_word = dv.getUint32(DATA + HEAD_OK + 16 + PAYLOAD - 4, true);

    // (b2) the decline: identical call, block's second page not present.
    const HEAD_BAD = 0x2000;
    dv.setUint32(CTRL, HEAD_BAD, true);
    const witness = [];
    for(let i = 0; i < STRIDE; i += 4) {
        dv.setUint32(DATA + HEAD_BAD + i, 0x5A5A0000 + i, true);
        witness.push(0x5A5A0000 + i);
    }
    const fallbacks_before = ex.jit_wbuf_intrinsic_get_fallbacks() >>> 0;
    frame(CONST_BAD);
    result.bad_cleanup = ex.jit_wbuf_intrinsic_execute(CONST_TARGET, ARGS) | 0;
    result.bad_head = dv.getUint32(CTRL, true);
    result.bad_fallback_delta = (ex.jit_wbuf_intrinsic_get_fallbacks() >>> 0) - fallbacks_before;
    result.bad_ring_intact = witness.every(
        (word, i) => dv.getUint32(DATA + HEAD_BAD + i * 4, true) === word >>> 0);

    // (b3) the same ring slot still accepts a fully mapped block — the decline was the page,
    // not a wedged descriptor.
    frame(CONST_OK);
    result.recover_cleanup = ex.jit_wbuf_intrinsic_execute(CONST_TARGET, ARGS) | 0;
    result.recover_head = dv.getUint32(CTRL, true);

    // (b4) a mapped block that ALIASES the ring slot this call writes: the copy would read
    // bytes the same call is still producing, so it must decline rather than encode garbage.
    const HEAD_ALIAS = 0x3000;
    dv.setUint32(CTRL, HEAD_ALIAS, true);
    dv.setUint32(DATA + HEAD_ALIAS, 0x7E577E57, true);
    frame(DATA + HEAD_ALIAS + 8);
    result.alias_cleanup = ex.jit_wbuf_intrinsic_execute(CONST_TARGET, ARGS) | 0;
    result.alias_head = dv.getUint32(CTRL, true);
    result.alias_first_word = dv.getUint32(DATA + HEAD_ALIAS, true);
    return result;
}

const r = await run();
console.log("jit-wbuf-intrinsic-paging " + JSON.stringify(r));
if(!r.paging) fail("guest did not enable paging");
if(r.hits_under_paging === 0) fail("(a) generated indirect CALL never hit the intrinsic under paging");
if(r.ring_head_after_guest === 0) fail("(a) guest hits did not append to the ring");
if(r.ok_cleanup !== 16 || r.ok_head !== 0x1000 + STRIDE)
    fail("(b1) a two-page mapped constant block was refused");
if(r.ok_first_word !== 0xC0DE0000 || r.ok_last_word !== (0xC0DE0000 + PAYLOAD - 4))
    fail("(b1) constant payload was not copied verbatim");
if(r.bad_cleanup !== -1) fail("(b2) a block crossing into a NOT-PRESENT page was executed");
if(r.bad_head !== 0x2000) fail("(b2) declined call advanced the ring head");
if(!r.bad_ring_intact) fail("(b2) declined call wrote ring bytes");
if(r.bad_fallback_delta !== 1) fail("(b2) declined call was not counted as a fallback");
if(r.recover_cleanup !== 16 || r.recover_head !== 0x2000 + STRIDE)
    fail("(b3) intrinsic did not recover after the decline");
if(r.alias_cleanup !== -1 || r.alias_head !== 0x3000 || r.alias_first_word !== 0x7E577E57)
    fail("(b4) a constant block aliasing its own ring slot was executed");
console.log("  ok — hits under paging, and a non-present constant page declines with no side effect");
