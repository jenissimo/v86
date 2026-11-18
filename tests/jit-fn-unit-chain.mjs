#!/usr/bin/env node
// Function units, chained calls (fn_unit.rs, chain switch, default off). A direct CALL inside a
// unit whose target is a published unit becomes a call of the callee's chain function with the
// guest GPRs and the native lazy-flag tuple as values; every failed pre-call check leaves the
// unit at the CALL before it commits, and a RET that does not land where the chain expects
// leaves at the RET.
//
// Fixture: the census fixture's protected-mode setup. The functions live on pages the caller's
// modules may not absorb (region exclusion), so the page JIT reaches them only through the
// dispatcher. Phase 1: `outer` (one page) calls `mid` (another page), which calls `sum`, a leaf
// reading the caller's CF, a leaf setting flags the caller reads, and `pv` (its own page);
// `outer` also calls a function that moves its own return address by two bytes, one that
// rewrites an immediate in `outer`'s continuation every 8192nd call, and an `and esp` frame.
// Between phases the main loop patches `pv` (callee invalidated, dependants dropped and
// recompiled). Phase 3 recurses 40 deep (self calls past the depth cap). Phase 4 runs long
// chained loops that must leave by the shared loop budget, and `lc`, whose own loop runs on
// what its callee left of the budget. Phase 5 (`pt`) chains a CALL that ends on a page
// boundary. Phase 11 enters `ent`, whose first instruction is a chained CALL, after a CR3
// reload, so the call falls back with nothing retired. Phase 13 requests an urgent exit from
// an MMIO read inside the chained unit `uc`. Phase 14 makes `kc`'s CALL push its return address
// over the immediate of the callee `kv` (a write to kv's code page after the token check).
// A second image takes a #PF two chained calls deep; a third runs on a 16-bit stack segment.
//
// Every asserting phase is gated: the guest repeats a fixed round until the host reports the
// units and chained links the phase needs as published, so the coverage does not depend on
// how fast compiled modules are instantiated. The interpreter runs every gate exactly twice;
// extra rounds must retire exactly its second round. Expected values are computed here,
// independently; every arm is also compared with the interpreter (registers, memory, retired
// instruction count at every checkpoint once the extra rounds are subtracted).
//
// Arms: interpreter, off (census only), units (chain off), canonical (mode 1), private
// (mode 2), private with `and esp` frames, private with page tails, private under a 20000
// instruction slice, offAfterOn. The mutations apply to the private, tails, deadline, fault
// and 16-bit-stack arms and each must make this test fail, by an assertion, a trap or a hang
// (run with --mutation=N to see it).
//
//   node tests/jit-fn-unit-chain.mjs [--mutation=bits]

const { V86 } = await import("../build/libv86.mjs");

const MUTATION = +(process.argv.find(a => a.startsWith("--mutation="))?.split("=")[1] ?? 0);

const BASE = 0x100000, ENTRY_OFF = 0x20, IMAGE = 0xC000;
const PD = 0x200000, PT0 = 0x201000, PT1 = 0x202000, PT2 = 0x203000;
const STACK = 0x380000, BUF = 0x400000, MARK = 0x401010, GATE_FLAG = 0x401020, UNMAPPED = 0xA00000, MMIO = 0x2000000, STACK16 = 0xF000;
const GDT_OFF = 0xE00, GDTR_OFF = 0xE20, IDT_OFF = 0xE30, IDTR_OFF = 0xEB8;
const FN_LO = 0x2000, FN_HI = 0xA000, PF_OFF = 0xB000, KV_OFF = 0xA000 - 21, KV_IMM0 = 0x5A5A0000, K_SPECIALS = 8;
const OUTER_OFF = 0x2000, LEAF_OFF = 0x3000, MID_OFF = 0x4000, PV_OFF = 0x5000, REC_OFF = 0x6000, TAIL_OFF = 0x7000;
const CHECK_PORT = 0x5F0, GATE_PORT = 0x5E0;
const ITER = 40000, ITER2 = 40000, REC_N = 40, REC_ITER = 3000, LONG_K = 20, LONG_N = 3000, LONG_CALLS = 40;
const PT_ITER = 4000, P_ITER = 2000, UC_ITER = 10000, SS16_ITER = 400000;
const LC_SPIN = 30000, LC_CALLS = 4;
const FAULT_ITER = 400000, LOOP_COUNTER = 100003, DEADLINE = 20000, GATE_MS = 60000;
const MEM_SIZE = 32 * 1024 * 1024, TIMEOUT_MS = 240000;
const S = { ENTRIES: 0, MATCHED: 1, EXITED: 2, COMPILES: 3, PUBLISHED: 4, WRITTEN: 5, FAILED: 6, INVALIDATED: 7,
    NOT_M1: 8, RET_FORM: 9, SHAPE: 10, STATE: 11, INSNS: 19, SITES: 20, UNCHAINED: 21, DROPPED: 22, STALE_PUBLISH: 23,
    MAX_RETIRED: 24, NO_PROGRESS: 25, TARGET_MISMATCH: 26 };
const C = { CALLS: 0, FB_DEPTH: 1, FB_BUDGET: 2, FB_TOKEN: 3, FB_MAPPING: 4, CALLEE_EXITED: 5, CALLER_TOKEN: 6,
    RET_MATCHED: 7, RET_MISMATCH: 8, TOKEN_AFTER_PUSH: 9 };

function header(dv, entry, end)
{
    const MAGIC = 0x1BADB002, FLAGS = 0x10000;
    [MAGIC, FLAGS, (-(MAGIC + FLAGS)) >>> 0, BASE, BASE, end, BASE + 0x10000, entry]
        .forEach((v, i) => dv.setUint32(i * 4, v, true));
}

function asm(buf, start)
{
    const dv = new DataView(buf.buffer);
    const a = { o: start };
    a.emit = (...b) => { for(const x of b) buf[a.o++] = x & 0xff; };
    a.u32 = v => { dv.setUint32(a.o, v >>> 0, true); a.o += 4; };
    a.u16 = v => { dv.setUint16(a.o, v & 0xffff, true); a.o += 2; };
    a.here = () => BASE + a.o;
    a.call = target => { a.emit(0xE8); a.u32(target - (BASE + a.o + 4)); };
    a.jnz = target => { a.emit(0x0F, 0x85); a.u32(target - (BASE + a.o + 4)); };
    a.out = v => { a.emit(0xBA); a.u32(CHECK_PORT); a.emit(0xB0, v, 0xEE); };
    // Warm-up gate: runs `block` (edi/esi kept) until the host leaves 0 in GATE_FLAG. Every arm
    // runs it at least twice, the interpreter exactly twice; its second round is the reference
    // for every further round an arm needs until its units are published.
    a.gate = (id, block) => {
        const top = a.here();
        a.emit(0xBA); a.u32(GATE_PORT + id); a.emit(0xEE);        // mov edx, port; out dx, al: the host sets GATE_FLAG
        a.emit(0x80, 0x3D); a.u32(GATE_FLAG); a.emit(0x00);         // cmp byte [GATE_FLAG], 0
        a.emit(0x0F, 0x84); const jz = a.o; a.u32(0);               // je done
        a.emit(0x57, 0x56);                                         // push edi; push esi
        block();
        a.emit(0x5E, 0x5F);                                         // pop esi; pop edi
        a.emit(0xE9); a.u32(top - (BASE + a.o + 4));                // jmp top
        dv.setUint32(jz, BASE + a.o - (BASE + jz + 4), true);
    };
    return a;
}

function prologue(buf, a)
{
    const dv = new DataView(buf.buffer);
    buf.set([0,0,0,0,0,0,0,0, 0xFF,0xFF,0,0,0,0x9A,0xCF,0, 0xFF,0xFF,0,0,0,0x92,0xCF,0, 0xFF,0xFF,0,0,0,0x92,0x8F,0], GDT_OFF);
    dv.setUint16(GDTR_OFF, 31, true);
    dv.setUint32(GDTR_OFF + 2, BASE + GDT_OFF, true);
    const H = BASE + PF_OFF, e14 = IDT_OFF + 14 * 8;
    dv.setUint16(e14, H & 0xFFFF, true);
    dv.setUint16(e14 + 2, 0x08, true);
    buf[e14 + 5] = 0x8E;
    dv.setUint16(e14 + 6, H >>> 16, true);
    dv.setUint16(IDTR_OFF, 16 * 8 - 1, true);
    dv.setUint32(IDTR_OFF + 2, BASE + IDT_OFF, true);
    a.emit(0x0F, 0x01, 0x15); a.u32(BASE + GDTR_OFF);
    a.emit(0xEA); a.u32(0); a.u16(0x08);
    dv.setUint32(a.o - 6, BASE + a.o, true);
    a.emit(0x66, 0xB8, 0x10, 0x00, 0x8E, 0xD8, 0x8E, 0xC0, 0x8E, 0xD0);
    a.emit(0xBC); a.u32(STACK);
    a.emit(0x0F, 0x01, 0x1D); a.u32(BASE + IDTR_OFF);
    a.emit(0xB8); a.u32(PD);
    a.emit(0x0F, 0x22, 0xD8);
    a.emit(0x0F, 0x20, 0xC0);
    a.emit(0x0D); a.u32(0x80000000);
    a.emit(0x0F, 0x22, 0xC0);
    a.emit(0xDB, 0xE3);
}

function page_tables(mem)
{
    const dv = new DataView(mem.buffer, mem.byteOffset);
    dv.setUint32(PD + 0, PT0 | 3, true);
    dv.setUint32(PD + 4, PT1 | 3, true);
    dv.setUint32(PD + (MMIO >>> 22) * 4, PT2 | 3, true);
    for(let i = 0; i < 1024; i++)
    {
        dv.setUint32(PT0 + i * 4, (i << 12) | 3, true);
        dv.setUint32(PT1 + i * 4, ((i + 1024) << 12) | 3, true);
        dv.setUint32(PT2 + i * 4, (MMIO + (i << 12)) | 3, true);
    }
}

function pf_handler(buf)
{
    const pf = asm(buf, PF_OFF);
    pf.emit(0x8B, 0x44, 0x24, 0x04);                // mov eax, [esp+4]: faulting eip
    pf.emit(0xA3); pf.u32(MARK);                    // mov [MARK], eax
    pf.emit(0xF4, 0xEB, 0xFE);
}

function build_main_image()
{
    const buf = new Uint8Array(IMAGE);
    header(new DataView(buf.buffer), BASE + ENTRY_OFF, BASE + IMAGE);
    pf_handler(buf);
    const F = {};

    const l = asm(buf, LEAF_OFF);
    F.sum = l.here();
    l.emit(0x55, 0x8B, 0xEC, 0x8B, 0x4D, 0x08);     // push ebp; mov ebp, esp; mov ecx, [ebp+8]
    l.emit(0x31, 0xC0);                             // xor eax, eax
    const sl = l.here();
    l.emit(0x01, 0xC8, 0xC1, 0xC0, 0x03, 0x49);     // l: add eax, ecx; rol eax, 3; dec ecx
    l.emit(0x75, (sl - (l.here() + 2)) & 0xFF);     // jnz l
    l.emit(0x5D, 0xC2, 0x04, 0x00);                 // pop ebp; ret 4
    l.o = LEAF_OFF + 0x100;
    F.cfadd = l.here();
    l.emit(0x15); l.u32(0x10); l.emit(0xC3);        // adc eax, 0x10; ret: reads the caller's CF
    l.o = LEAF_OFF + 0x140;
    F.setcf = l.here();
    l.emit(0x39, 0xD8, 0xC3);                       // cmp eax, ebx; ret: the caller reads CF
    l.o = LEAF_OFF + 0x180;
    F.retmod = l.here();
    l.emit(0x5A, 0x83, 0xC2, 0x02, 0x52, 0xC3);     // pop edx; add edx, 2; push edx; ret: returns 2 bytes late
    l.o = LEAF_OFF + 0x1C0;
    F.andesp = l.here();
    l.emit(0x55, 0x8B, 0xEC, 0x83, 0xE4, 0xF0);     // push ebp; mov ebp, esp; and esp, -16
    l.emit(0x83, 0xEC, 0x10, 0x89, 0x04, 0x24);     // sub esp, 16; mov [esp], eax
    l.emit(0x03, 0x04, 0x24, 0x8B, 0xE5, 0x5D, 0xC3); // add eax, [esp]; mov esp, ebp; pop ebp; ret

    const p = asm(buf, PV_OFF);
    F.pv = p.here();
    p.emit(0xBA); p.u32(0x11); p.emit(0xC3);        // mov edx, imm; ret

    const m = asm(buf, MID_OFF);
    F.mid = m.here();
    m.emit(0x55, 0x8B, 0xEC, 0x53);                 // push ebp; mov ebp, esp; push ebx
    m.emit(0xFF, 0x75, 0x08); m.call(F.sum);        // push dword [ebp+8]; call sum
    m.emit(0x89, 0xC3);                             // mov ebx, eax
    m.emit(0x81, 0xFB); m.u32(0x40000000);          // cmp ebx, 0x40000000
    m.call(F.cfadd);                                // eax += 0x10 + CF
    m.emit(0xBB); m.u32(0x7FFFFFFF);                // mov ebx, 0x7fffffff
    m.call(F.setcf);                                // flags of cmp eax, ebx
    m.emit(0x0F, 0x92, 0xC3, 0x0F, 0xB6, 0xDB, 0x01, 0xD8); // setb bl; movzx ebx, bl; add eax, ebx
    m.call(F.pv); m.emit(0x01, 0xD0);               // call pv; add eax, edx
    m.emit(0x5B, 0x5D, 0xC2, 0x04, 0x00);           // pop ebx; pop ebp; ret 4

    const o = asm(buf, OUTER_OFF);
    F.outer = o.here();
    o.emit(0x55, 0x8B, 0xEC, 0x51);                 // push ebp; mov ebp, esp; push ecx
    o.emit(0x8B, 0x45, 0x08, 0x83, 0xE0, 0x0F, 0x83, 0xC0, 0x05, 0x50); // mov eax, [ebp+8]; and eax, 15; add eax, 5; push eax
    o.call(F.mid); o.emit(0x01, 0xC7);              // call mid; add edi, eax
    o.emit(0x8B, 0x4D, 0x08);                       // mov ecx, [ebp+8]
    F.wrCall = o.here();
    o.o += 5;
    F.patch = o.here() + 1;
    o.emit(0xB8); o.u32(0x1234);                    // mov eax, imm: rewritten by wr
    o.emit(0x01, 0xC7);                             // add edi, eax
    o.call(F.andesp); o.emit(0x01, 0xC6);           // call andesp; add esi, eax
    o.call(F.retmod); o.emit(0x47, 0x47);           // call retmod; inc edi; inc edi (skipped)
    o.emit(0x59, 0x5D, 0xC2, 0x04, 0x00);           // pop ecx; pop ebp; ret 4

    l.o = LEAF_OFF + 0x200;
    F.wr = l.here();
    l.emit(0xF7, 0xC1); l.u32(0x1FFF);              // test ecx, 0x1fff
    l.emit(0x75, 0x06);                             // jnz done
    l.emit(0x89, 0x0D); l.u32(F.patch);             // mov [patch], ecx: the callee rewrites its caller
    l.emit(0xC3);                                   // done: ret
    const wc = asm(buf, F.wrCall - BASE);
    wc.call(F.wr);

    const r = asm(buf, REC_OFF);
    F.rec = r.here();
    r.emit(0x8B, 0x44, 0x24, 0x04, 0x85, 0xC0);     // mov eax, [esp+4]; test eax, eax
    r.emit(0x74, 0x0E, 0x48, 0x50);                 // jz base; dec eax; push eax
    r.call(F.rec);                                  // call rec
    r.emit(0x03, 0x44, 0x24, 0x04, 0xC2, 0x04, 0x00); // add eax, [esp+4]; ret 4
    r.emit(0xC2, 0x04, 0x00);                       // base: ret 4
    r.o = REC_OFF + 0x100;
    F.lng = r.here();
    r.emit(0x55, 0x8B, 0xEC, 0x53, 0x31, 0xDB, 0x8B, 0x4D, 0x08); // push ebp; mov ebp, esp; push ebx; xor ebx, ebx; mov ecx, [ebp+8]
    const ll = r.here();
    r.emit(0x51, 0x68); r.u32(LONG_N); r.call(F.sum); // l: push ecx; push LONG_N; call sum
    r.emit(0x01, 0xC3, 0x59, 0x49); r.jnz(ll);      // add ebx, eax; pop ecx; dec ecx; jnz l
    r.emit(0x89, 0xD8, 0x5B, 0x5D, 0xC2, 0x04, 0x00); // mov eax, ebx; pop ebx; pop ebp; ret 4
    r.o = REC_OFF + 0x200;
    F.lc = r.here();
    r.emit(0x68); r.u32(LONG_N); r.call(F.sum);    // push LONG_N; call sum: the callee uses up part of the budget
    r.emit(0xB9); r.u32(LC_SPIN);                   // mov ecx, LC_SPIN
    const ls = r.here();
    r.emit(0x49, 0x75, (ls - (r.here() + 3)) & 0xFF, 0xC3); // l: dec ecx; jnz l; ret: the caller's own loop runs on the budget left

    l.o = LEAF_OFF + 0x240;
    F.ptl = l.here();
    l.emit(0x8D, 0x41, 0x03, 0xC3);                 // lea eax, [ecx+3]; ret
    const t = asm(buf, TAIL_OFF);
    t.o = TAIL_OFF + 0x40;
    F.ent = t.here();
    t.call(F.ptl); t.emit(0x40, 0xC3);              // call ptl (the unit's first instruction); inc eax; ret
    t.o = TAIL_OFF + 0x80;
    F.uc = t.here();
    t.emit(0x55, 0x8B, 0xEC, 0xA1); t.u32(MMIO);    // push ebp; mov ebp, esp; mov eax, [MMIO]: the host may request an urgent exit here
    t.emit(0x01, 0xC7, 0x6A, 0x0A); t.call(F.sum);  // add edi, eax; push 10; call sum
    t.emit(0x01, 0xC7, 0x5D, 0xC3);                 // add edi, eax; pop ebp; ret
    t.o = TAIL_OFF + 0x1000 - 8;
    F.pt = t.here();
    t.emit(0x55, 0x8B, 0xEC); t.call(F.ptl);        // push ebp; mov ebp, esp; call ptl: ends on the page boundary
    if(t.o !== TAIL_OFF + 0x1000) throw new Error("pt layout");
    F.ptCont = t.here();
    t.emit(0x01, 0xC7, 0x5D, 0xC3);                 // next page: add edi, eax; pop ebp; ret
    // kv sits in the last bytes of its page. A special call enters kc with ESP at the end of that
    // page: kc's `sub esp, 16` leaves kv's RET alone and its CALL pushes the return address over
    // kv's immediate, which writes kv's code page and drops kv after kc checked kv's token.
    const kv = asm(buf, KV_OFF);
    F.kv = kv.here();
    kv.emit(0xB8); kv.u32(KV_IMM0); kv.emit(0xC3);  // mov eax, imm (the pushed return address); ret
    t.o = TAIL_OFF + 0xC0;
    F.kc = t.here();
    t.emit(0x83, 0xEC, 0x10, 0xF6, 0xC1, 0x01);     // sub esp, 16; test cl, 1
    t.emit(0x74, 0x07);                             // jz site2
    t.call(F.kv); F.kcCont1 = t.here();             // site1
    t.emit(0xEB, 0x05);                             // jmp done
    t.call(F.kv); F.kcCont2 = t.here();             // site2
    t.emit(0x83, 0xC4, 0x10, 0xC3);                 // done: add esp, 16; ret

    const a = asm(buf, ENTRY_OFF);
    prologue(buf, a);
    a.emit(0x31, 0xFF, 0x31, 0xF6);                 // xor edi, edi; xor esi, esi
    const loop = n => {
        a.emit(0xB9); a.u32(n);
        const top = a.here();
        a.emit(0x51, 0x51); a.call(F.outer);        // push ecx; push ecx; call outer
        a.emit(0x59, 0x49); a.jnz(top);             // pop ecx; dec ecx; jnz
    };
    const calls = (n, fn, after = []) => {
        a.emit(0xB9); a.u32(n);
        const top = a.here();
        a.emit(0x51); a.call(fn); a.emit(...after);  // push ecx; call fn
        a.emit(0x59, 0x49); a.jnz(top);             // pop ecx; dec ecx; jnz
    };
    loop(ITER);
    a.gate(1, () => loop(64));
    a.out(1);
    a.emit(0xC7, 0x05); a.u32(F.pv + 1); a.u32(0x77); // mov dword [pv+1], 0x77
    a.gate(2, () => loop(64));
    a.out(6);
    loop(ITER2);
    a.out(2);
    a.gate(3, () => {
        a.emit(0xB9); a.u32(4);
        const top = a.here();
        a.emit(0x51, 0x6A, REC_N); a.call(F.rec);   // push ecx; push REC_N; call rec
        a.emit(0x59, 0x49); a.jnz(top);
    });
    a.out(7);
    a.emit(0xB9); a.u32(REC_ITER);
    const rt = a.here();
    a.emit(0x51, 0x6A, REC_N); a.call(F.rec);       // push ecx; push REC_N; call rec
    a.emit(0x01, 0xC7, 0x59, 0x49); a.jnz(rt);      // add edi, eax; pop ecx; dec ecx; jnz
    a.out(3);
    a.gate(4, () => { a.emit(0x6A, 0x02); a.call(F.lng); a.call(F.lc); });
    a.out(8);
    a.emit(0xB9); a.u32(LONG_CALLS);
    const lt = a.here();
    a.emit(0x51, 0x6A, LONG_K); a.call(F.lng);      // push ecx; push LONG_K; call lng
    a.emit(0x01, 0xC7, 0x59, 0x49); a.jnz(lt);      // add edi, eax; pop ecx; dec ecx; jnz
    for(let i = 0; i < LC_CALLS; i++) { a.call(F.lc); a.emit(0x01, 0xC7); } // call lc; add edi, eax
    a.out(4);
    a.gate(5, () => calls(64, F.pt));
    a.out(9);
    calls(PT_ITER, F.pt);                           // the continuation of pt's chained CALL is the next page
    a.out(5);
    a.gate(6, () => calls(64, F.ent));
    a.out(10);
    a.emit(0xB9); a.u32(P_ITER);
    const pt = a.here();
    a.emit(0x51, 0x0F, 0x20, 0xD8, 0x0F, 0x22, 0xD8); // push ecx; mov eax, cr3; mov cr3, eax: ent's call target is not in the TLB
    a.call(F.ent);
    a.emit(0x01, 0xC7, 0x59, 0x49); a.jnz(pt);      // add edi, eax; pop ecx; dec ecx; jnz
    a.out(11);
    a.gate(7, () => calls(16, F.uc));
    a.out(12);
    calls(UC_ITER, F.uc);
    a.out(13);
    for(let j = 1; j <= K_SPECIALS; j++)
    {
        a.gate(7 + j, () => calls(16, F.kc));
        a.emit(0x89, 0xE3, 0xBC); a.u32(BASE + KV_OFF + 25); // mov ebx, esp; mov esp, end of kv's page + 4
        a.emit(0xB9); a.u32(j); a.call(F.kc);       // mov ecx, j; call kc
        a.emit(0x89, 0xDC, 0x01, 0xC7);             // mov esp, ebx; add edi, eax
    }
    a.out(14);
    a.emit(0xF4);
    return { buf, F };
}

function build_fault_image()
{
    const buf = new Uint8Array(IMAGE);
    header(new DataView(buf.buffer), BASE + ENTRY_OFF, BASE + IMAGE);
    pf_handler(buf);
    const F = {};
    const f = asm(buf, LEAF_OFF);
    F.rd = f.here();
    f.emit(0x8B, 0x44, 0x24, 0x04);                 // mov eax, [esp+4]
    F.load = f.here();
    f.emit(0x8B, 0x00, 0xC2, 0x04, 0x00);           // mov eax, [eax]; ret 4
    const o = asm(buf, MID_OFF);
    F.fo = o.here();
    o.emit(0x55, 0x8B, 0xEC, 0xFF, 0x75, 0x08);     // push ebp; mov ebp, esp; push dword [ebp+8]
    o.call(F.rd); o.emit(0x40, 0x5D, 0xC2, 0x04, 0x00); // call rd; inc eax; pop ebp; ret 4
    const a = asm(buf, ENTRY_OFF);
    prologue(buf, a);
    a.emit(0x31, 0xFF);                             // xor edi, edi
    a.emit(0xB9); a.u32(FAULT_ITER);
    const top = a.here();
    a.emit(0x51, 0x68); a.u32(BUF); a.call(F.fo);  // push ecx; push BUF; call fo
    a.emit(0x01, 0xC7, 0x59, 0x49); a.jnz(top);    // add edi, eax; pop ecx; dec ecx; jnz
    a.gate(1, () => {
        a.emit(0xB9); a.u32(64);
        const t = a.here();
        a.emit(0x51, 0x68); a.u32(BUF); a.call(F.fo);
        a.emit(0x59, 0x49); a.jnz(t);
    });
    a.emit(0x68); a.u32(UNMAPPED); a.call(F.fo);   // push UNMAPPED; call fo: #PF two calls deep
    a.emit(0xC7, 0x05); a.u32(MARK); a.u32(0xBAD); // must not run
    a.emit(0xF4);
    return { buf, F };
}

// A 16-bit stack segment with base 0 and flat code and data: is_32 and flat segmentation hold,
// SS32 does not, so no function may become a unit.
function build_ss16_image()
{
    const buf = new Uint8Array(IMAGE);
    header(new DataView(buf.buffer), BASE + ENTRY_OFF, BASE + IMAGE);
    pf_handler(buf);
    const F = {};
    const f = asm(buf, LEAF_OFF);
    F.f16 = f.here();
    f.emit(0x8B, 0x44, 0x24, 0x04, 0x01, 0xC0, 0xC2, 0x04, 0x00); // mov eax, [esp+4]; add eax, eax; ret 4
    const g = asm(buf, MID_OFF);
    F.g16 = g.here();
    g.emit(0xFF, 0x74, 0x24, 0x04); g.call(F.f16);  // push dword [esp+4]; call f16
    g.emit(0x40, 0xC2, 0x04, 0x00);                 // inc eax; ret 4
    const a = asm(buf, ENTRY_OFF);
    prologue(buf, a);
    a.emit(0x31, 0xFF, 0x66, 0xB8, 0x18, 0x00, 0x8E, 0xD0); // xor edi, edi; mov ax, 0x18; mov ss, ax
    a.emit(0xBC); a.u32(STACK16);                   // mov esp, STACK16
    a.gate(1, () => {
        a.emit(0xB9); a.u32(8);
        const t = a.here();
        a.emit(0x51); a.call(F.g16); a.emit(0x49); a.jnz(t);
    });
    a.emit(0xB9); a.u32(SS16_ITER);
    const top = a.here();
    a.emit(0x51); a.call(F.g16);                    // push ecx; call g16
    a.emit(0x01, 0xC7, 0x49); a.jnz(top);           // add edi, eax; dec ecx; jnz
    a.emit(0xF4);
    return { buf, F };
}

const rol = (v, n) => ((v << n) | (v >>> (32 - n))) >>> 0;
function sum(n) { let a = 0; for(let c = n; c > 0; c--) { a = (a + c) >>> 0; a = rol(a, 3); } return a; }
function expected_main(F)
{
    let edi = 0, esi = 0, patch = 0x1234;
    for(const [n, pv] of [[ITER, 0x11], [ITER2, 0x77]])
        for(let c = n; c > 0; c--)
        {
            const s = sum((c & 15) + 5);
            let eax = (s + 0x10 + (s < 0x40000000 ? 1 : 0)) >>> 0;
            eax = (eax + (eax < 0x7FFFFFFF ? 1 : 0)) >>> 0;
            eax = (eax + pv) >>> 0;
            edi = (edi + eax) >>> 0;
            if((c & 0x1FFF) === 0) patch = c;
            edi = (edi + patch) >>> 0;
            esi = (esi + 2 * patch) >>> 0;
        }
    edi = (edi + REC_ITER * (REC_N * (REC_N + 1) / 2)) >>> 0;
    const lng = Math.imul(LONG_K, sum(LONG_N)) >>> 0;
    for(let i = 0; i < LONG_CALLS; i++) edi = (edi + lng) >>> 0;
    edi = (edi + Math.imul(LC_CALLS, sum(LONG_N))) >>> 0;
    for(let c = PT_ITER; c > 0; c--) edi = (edi + c + 3) >>> 0;
    for(let c = P_ITER; c > 0; c--) edi = (edi + c + 4) >>> 0;
    edi = (edi + Math.imul(UC_ITER, sum(10))) >>> 0;
    for(let j = 1; j <= K_SPECIALS; j++) edi = (edi + (j & 1 ? F.kcCont1 : F.kcCont2)) >>> 0;
    return { edi, esi };
}

function run(opts)
{
    if(process.env.UDBG) console.error("RUN", opts.name);
    return new Promise((resolve, reject) => {
        const { build, units = false, chain = 0, andEsp = false, toggle = false, mutation = 0, jit = true, onCheck,
            gates: gateConds = null, tails = false, deadline = 0 } = opts;
        const emulator = new V86({ autostart: false, memory_size: MEM_SIZE, disable_jit: jit ? 0 : 1, log_level: 0 });
        const img = build();
        const modules = [];
        const checks = {};
        const gates = {};
        const urgent = {};
        let timer, fp, freeInitial = 0;
        const finish = status => {
            clearTimeout(timer);
            try { emulator.stop(); } catch(e) {}
            delete globalThis.__jitPublicationCapture;
            const cpu = emulator.v86.cpu, w = cpu.wm.exports;
            const dv = new DataView(cpu.mem8.buffer, cpu.mem8.byteOffset);
            const out = {
                status, img, modules, fp, checks, gates, urgent,
                stats: Array.from({ length: w.fn_unit_stat_count() }, (_, i) => w.fn_unit_stat(i)),
                counters: Array.from({ length: w.fn_unit_chain_counter_count() }, (_, i) => w.fn_unit_chain_counter(i) >>> 0),
                depth: w.fn_unit_chain_depth() >>> 0,
                count: w.fn_unit_count() >>> 0,
                counter: new Uint32Array(cpu.wasm_memory.buffer, 664, 1)[0],
                hook: w.fn_census_last_exception_counter() >>> 0,
                edi: cpu.reg32[7] >>> 0, esi: cpu.reg32[6] >>> 0, esp: cpu.reg32[4] >>> 0,
                mark: dv.getUint32(MARK, true),
                fnBytes: Array.from(cpu.mem8.subarray(BASE + FN_LO, BASE + FN_HI)),
            };
            if(process.env.UDBG)
            {
                const rw = w.fn_census_row_words(), n = w.fn_census_row_count();
                const rows = new Uint32Array(cpu.wasm_memory.buffer, w.fn_census_rows_ptr() >>> 0, rw * n);
                const want = new Set(Object.values(img.F));
                const dump = {};
                for(let i = 0; i < n; i++)
                    if(want.has(rows[i * rw])) dump[hex(rows[i * rw])] = { entries: rows[i * rw + 2], unit: hex(rows[i * rw + 3]), window: hex(rows[i * rw + 4]), flags: hex(rows[i * rw + 26]), pages: rows[i * rw + 10] };
                console.error("ROWS", opts.name, JSON.stringify(dump), "STATS", JSON.stringify(out.stats), "CENSUS", JSON.stringify(Array.from({ length: w.fn_census_stat_count() }, (_, i) => w.fn_census_stat(i))));
            }
            const unitsBefore = out.count;
            w.fn_unit_set_enabled(0);
            out.countAfterOff = w.fn_unit_count() >>> 0;
            w.fn_unit_set_chain(0);
            w.fn_unit_set_and_esp(0);
            w.fn_unit_set_chain_counting(0);
            w.fn_unit_set_mutation(0);
            w.fn_unit_reset();
            w.fn_census_set_enabled(0);
            w.fn_census_reset();
            setTimeout(() => {
                cpu.jit_clear_cache();
                out.leak = { units: unitsBefore, free: w.jit_aot_free_table_index_count() >>> 0, initial: freeInitial,
                    pending: w.fn_unit_pending() >>> 0 };
                resolve(out);
            }, 100);
        };
        emulator.bus.register("cpu-event-halt", () => finish("halt"));
        emulator.add_listener("emulator-loaded", () => {
            try {
                const cpu = emulator.v86.cpu, w = cpu.wm.exports;
                cpu.reboot_internal();
                cpu.reset_memory();
                const fp0 = [w.jit_codegen_fingerprint_lo() >>> 0, w.jit_codegen_fingerprint_hi() >>> 0];
                const defaults = { units: w.fn_unit_enabled() >>> 0, chain: w.fn_unit_get_chain() >>> 0, andEsp: w.fn_unit_get_and_esp() >>> 0 };
                w.fn_unit_reset();
                w.fn_unit_set_mutation(mutation);
                w.fn_unit_set_threshold(4);
                w.fn_unit_set_chain_counting(1);
                w.fn_census_set_enabled(1);
                const fp1 = [w.jit_codegen_fingerprint_lo() >>> 0, w.jit_codegen_fingerprint_hi() >>> 0];
                if(units || toggle) w.fn_unit_set_enabled(1);
                if(chain || toggle) w.fn_unit_set_chain(toggle ? 2 : chain);
                if(andEsp || toggle) w.fn_unit_set_and_esp(1);
                if(toggle)
                {
                    w.fn_unit_set_and_esp(0);
                    w.fn_unit_set_chain(0);
                    w.fn_unit_set_enabled(0);
                }
                fp = { before: fp0, census: fp1, after: [w.jit_codegen_fingerprint_lo() >>> 0, w.jit_codegen_fingerprint_hi() >>> 0], defaults };
                w.jit_set_region_exclusion(BASE + FN_LO, BASE + FN_HI);
                globalThis.__jitPublicationCapture = {
                    begin(start, tableIndex, slot, code) {
                        let h = 0x811C9DC5;
                        for(const b of code) h = Math.imul(h ^ b, 0x01000193) >>> 0;
                        modules.push({ start, tableIndex, hash: h, length: code.length });
                        return null;
                    },
                };
                cpu.jit_clear_cache?.();
                freeInitial = w.jit_aot_free_table_index_count() >>> 0;
                cpu.load_multiboot(img.buf.buffer);
                page_tables(cpu.mem8);
                if(tails) w.set_jit_page_tails(1);
                const hp = new Uint32Array(cpu.wasm_memory.buffer, w.get_hypercall_page_ptr() >>> 0, 4);
                const limit0 = hp[0], enabled0 = hp[2];
                if(deadline) hp[0] = deadline;
                const counterNow = () => new Uint32Array(cpu.wasm_memory.buffer, 664, 1)[0];
                const snap = () => ({
                    stats: Array.from({ length: w.fn_unit_stat_count() }, (_, i) => w.fn_unit_stat(i)),
                    counters: Array.from({ length: w.fn_unit_chain_counter_count() }, (_, i) => w.fn_unit_chain_counter(i) >>> 0),
                });
                for(let id = 1; id <= 7 + K_SPECIALS; id++)
                {
                    cpu.io.register_write(GATE_PORT + id, cpu, () => {
                        const g = gates[id] ??= { reads: [], t0: Date.now() };
                        g.reads.push(counterNow());
                        const more = g.reads.length <= 2 ? 1
                            : !gateConds?.[id] || gateConds[id](w) ? 0
                            : Date.now() - g.t0 > GATE_MS ? (g.timeout = true, 0) : 1;
                        cpu.mem8[GATE_FLAG] = more;
                    });
                }
                cpu.io.mmap_register(MMIO, 0x20000, () => 0, () => {}, () => {
                    if(urgent.armed)
                    {
                        urgent.armed = false;
                        urgent.before = snap();
                        hp[0] = 0;
                        hp[2] = 1;
                        setTimeout(() => {
                            urgent.after = snap();
                            hp[2] = enabled0;
                            hp[0] = deadline || limit0;
                        }, 0);
                    }
                    return 0;
                }, () => {});
                cpu.io.register_write(CHECK_PORT, cpu, v => {
                    if(v === 12) urgent.armed = true;
                    checks[v] = {
                        stats: Array.from({ length: w.fn_unit_stat_count() }, (_, i) => w.fn_unit_stat(i)),
                        counters: Array.from({ length: w.fn_unit_chain_counter_count() }, (_, i) => w.fn_unit_chain_counter(i) >>> 0),
                        counter: new Uint32Array(cpu.wasm_memory.buffer, 664, 1)[0],
                        extra: onCheck ? onCheck(v, cpu, w) : null,
                    };
                });
                timer = setTimeout(() => finish("HANG"), TIMEOUT_MS);
                emulator.run();
            }
            catch(e) { reject(e); }
        });
    });
}

const failures = [];
const fail = msg => failures.push(msg);
const hex = v => "0x" + (v >>> 0).toString(16);
const F = build_main_image().F;
const FF0 = build_fault_image().F;
const F16 = build_ss16_image().F;
const slotsAt = (v, cpu, w) => Object.fromEntries(["outer", "mid", "sum", "cfadd", "setcf", "pv", "retmod", "wr", "andesp", "rec", "lng", "pt", "ptl", "ent", "uc"]
    .map(k => [k, { slot: w.fn_unit_slot(F[k]) >>> 0, chain: w.fn_unit_chain_slot(F[k]) >>> 0 }]));

// Gate conditions: the units and chained links an asserting phase needs, published before it starts.
const chainUnit = (w, f) => (w.fn_unit_chain_slot(f) >>> 0) < 0xFFFF;
const linked = (w, a, b) => (w.fn_unit_chain_linked(a, b) >>> 0) === 1;
const ready = (units, links) => w => units.every(k => chainUnit(w, F[k])) && links.every(([a, b]) => linked(w, F[a], F[b]));
const mainUnits = ["outer", "mid", "sum", "cfadd", "setcf", "pv", "retmod", "wr"];
const mainLinks = [["outer", "mid"], ["outer", "wr"], ["outer", "retmod"], ["mid", "sum"], ["mid", "cfadd"], ["mid", "setcf"], ["mid", "pv"]];
const GATES = {
    ...Object.fromEntries(Array.from({ length: K_SPECIALS }, (_, j) => [8 + j, ready(["kc", "kv"], [["kc", "kv"]])])),
    1: ready(mainUnits, mainLinks),
    2: ready(mainUnits, mainLinks),
    3: ready(["rec"], [["rec", "rec"]]),
    4: ready(["lng", "sum", "lc"], [["lng", "sum"], ["lc", "sum"]]),
    6: ready(["ent", "ptl"], [["ent", "ptl"]]),
    7: ready(["uc", "sum"], [["uc", "sum"]]),
};
const GATES_ANDESP = { ...GATES, 1: ready([...mainUnits, "andesp"], [...mainLinks, ["outer", "andesp"]]), 2: ready([...mainUnits, "andesp"], [...mainLinks, ["outer", "andesp"]]) };
const GATES_TAILS = { ...GATES, 5: ready(["pt", "ptl"], [["pt", "ptl"]]) };
const GATES_FAULT = { 1: w => chainUnit(w, FF0.fo) && chainUnit(w, FF0.rd) && linked(w, FF0.fo, FF0.rd) };
const pageModule = (w, virt) => {
    const meta = new Uint32Array(w.memory.buffer, w.jit_get_dispatch_meta_ptr() >>> 0, 2 * 0x100000);
    return (meta[2 * (virt >>> 12)] | meta[2 * (virt >>> 12) + 1]) !== 0;
};
const GATES_FAULT_OFF = { 1: w => pageModule(w, FF0.load) };
const GATES_SS16 = { 1: w => w.fn_unit_stat(S.STATE) + w.fn_unit_stat(S.PUBLISHED) > 0 };

const interp = await run({ name: "interp", build: build_main_image, jit: false });
const off = await run({ name: "off", build: build_main_image });
const units = await run({ name: "units", build: build_main_image, units: true });
const canonical = await run({ name: "canonical", build: build_main_image, units: true, chain: 1, onCheck: slotsAt, gates: GATES });
const priv = await run({ name: "private", build: build_main_image, units: true, chain: 2, mutation: MUTATION, onCheck: slotsAt, gates: GATES });
const andEsp = await run({ name: "andEsp", build: build_main_image, units: true, chain: 2, andEsp: true, onCheck: slotsAt, gates: GATES_ANDESP });
const tails = await run({ name: "tails", build: build_main_image, units: true, chain: 2, tails: true, mutation: MUTATION, onCheck: slotsAt, gates: GATES_TAILS });
const deadline = await run({ name: "deadline", build: build_main_image, units: true, chain: 2, deadline: DEADLINE, mutation: MUTATION, onCheck: slotsAt, gates: GATES });
const offAfterOn = await run({ name: "offAfterOn", build: build_main_image, toggle: true });
const E = expected_main(F);
const arms = [["interpreter", interp], ["off", off], ["units", units], ["canonical", canonical], ["private", priv], ["andEsp", andEsp],
    ["tails", tails], ["deadline", deadline], ["offAfterOn", offAfterOn]];
const chainArms = [["canonical", canonical], ["private", priv], ["andEsp", andEsp], ["tails", tails], ["deadline", deadline]];

// Retired counts. A gate round is a fixed instruction sequence; the interpreter runs one round per
// gate, an arm may need more until its units are published. Every round must retire exactly the
// interpreter's round, and the counts outside the gates must match the interpreter's exactly.
const MAIN_ORDER = ["g1", 1, "g2", 6, 2, "g3", 7, 3, "g4", 8, 4, "g5", 9, 5, "g6", 10, 11, "g7", 12, 13,
    ...Array.from({ length: K_SPECIALS }, (_, j) => "g" + (8 + j)), 14];
const round = (ref, id, i = 2) => ref.gates[id].reads[i] - ref.gates[id].reads[i - 1];
const extra = (r, ref, id) => (r.gates[id]?.reads.length ?? 0) > 3 ? ((r.gates[id].reads.length - 3) * round(ref, id)) >>> 0 : 0;
const adjBefore = (r, ref, order, item) => {
    let a = 0;
    for(const x of order)
    {
        if(x === item) break;
        if(typeof x === "string") a = (a + extra(r, ref, +x.slice(1))) >>> 0;
    }
    return a;
};
const adjTotal = (r, ref, order) => adjBefore(r, ref, order, undefined);
function check_gates(name, r, ref, order)
{
    for(const x of order)
    {
        if(typeof x !== "string") continue;
        const id = +x.slice(1), g = r.gates[id], rg = ref.gates[id];
        if(!g || g.reads.length < 3) { fail(`${name}: gate ${id} not reached`); continue; }
        if(g.timeout) fail(`${name}: gate ${id}: units not published after ${GATE_MS} ms (${g.reads.length - 1} rounds)`);
        if(((g.reads[0] - adjBefore(r, ref, order, x)) >>> 0) !== rg.reads[0]) fail(`${name}: retired at gate ${id} ${g.reads[0]} != interpreter ${rg.reads[0]} (+ earlier extra rounds)`);
        for(let i = 1; i < g.reads.length; i++)
            if(((g.reads[i] - g.reads[i - 1]) >>> 0) !== round(ref, id, Math.min(i, 2))) { fail(`${name}: gate ${id} round ${i} retired ${g.reads[i] - g.reads[i - 1]} != ${round(ref, id, Math.min(i, 2))}`); break; }
    }
}

if(interp.gates[1]?.reads.length !== 3) fail("interpreter: gate 1 did not run exactly two rounds");
for(const [name, r] of arms)
{
    if(r.status !== "halt") fail(`${name}: guest did not reach hlt (${r.status})`);
    if(r.fp.defaults.units || r.fp.defaults.chain || r.fp.defaults.andEsp) fail(`${name}: units, chain and and-esp must default OFF (${JSON.stringify(r.fp.defaults)})`);
    if(r.edi !== E.edi) fail(`${name}: checksum ${hex(r.edi)} != expected ${hex(E.edi)}`);
    if(r.esi !== E.esi) fail(`${name}: and-esp sum ${hex(r.esi)} != expected ${hex(E.esi)}`);
    if(r.esp !== STACK) fail(`${name}: esp ${hex(r.esp)} != ${hex(STACK)} at hlt`);
    if(r.mark !== 0) fail(`${name}: a fault reached the handler (${hex(r.mark)})`);
    check_gates(name, r, interp, MAIN_ORDER);
    if(((r.counter - adjTotal(r, interp, MAIN_ORDER)) >>> 0) !== interp.counter) fail(`${name}: retired ${r.counter} - gate rounds ${adjTotal(r, interp, MAIN_ORDER)} != interpreter ${interp.counter}`);
    for(const k of MAIN_ORDER.filter(x => typeof x === "number"))
        if(((r.checks[k]?.counter - adjBefore(r, interp, MAIN_ORDER, k)) >>> 0) !== interp.checks[k]?.counter) fail(`${name}: retired at check ${k} ${r.checks[k]?.counter} != interpreter ${interp.checks[k]?.counter} (+ gate rounds)`);
    if(r.depth !== 0) fail(`${name}: chain depth ${r.depth} left behind`);
}
const fpEq = (x, y) => x[0] === y[0] && x[1] === y[1];
for(const [name, r] of arms.slice(1))
    if(!fpEq(r.fp.after, r.fp.census)) fail(`${name}: a unit switch moved the codegen fingerprint`);
for(const [name, r] of [["off", off], ["offAfterOn", offAfterOn], ["interpreter", interp]])
{
    if(r.stats.some(v => v !== 0)) fail(`${name}: unit stats moved while off: ${r.stats}`);
    if(r.counters.some(v => v !== 0)) fail(`${name}: chain counters moved while off: ${r.counters}`);
    if(r.count !== 0) fail(`${name}: units published while off`);
}
// Which pages get compiled when, and with which entry points, depends on instantiation timing in
// two independent runs; every page compiled in both must have at least one byte-identical module.
{
    const byStart = r => {
        const m = new Map();
        for(const x of r.modules.filter(x => x.start >= BASE && x.start < BASE + IMAGE))
            m.set(x.start, (m.get(x.start) ?? new Set()).add(x.hash + ":" + x.length));
        return m;
    };
    const a = byStart(off), b = byStart(offAfterOn);
    let common = 0;
    for(const [start, hs] of a)
    {
        const o = b.get(start);
        if(!o) continue;
        common++;
        if(![...hs].some(h => o.has(h))) fail(`page module at ${hex(start)} differs after enabling and disabling units, chain and and-esp`);
    }
    if(common === 0) fail("off and offAfterOn compiled no common page module");
}
if(units.counters.some(v => v !== 0)) fail(`units: chain counters moved with the chain off: ${units.counters}`);
if(units.stats[S.SITES] !== 0) fail("units: chain sites compiled with the chain off");

const phase = (r, k, from) => r.counters.map((v, i) => (r.checks[k]?.counters[i] ?? 0) - (from ? r.checks[from]?.counters[i] ?? 0 : 0));
const phaseStat = (r, i, k, from) => (r.checks[k]?.stats[i] ?? 0) - (r.checks[from]?.stats[i] ?? 0);
for(const [name, r] of chainArms)
{
    const st = r.stats, c = r.counters;
    const p2 = phase(r, 2, 6), p3 = phase(r, 3, 7), p4 = phase(r, 4, 8), p5 = phase(r, 5, 9), p11 = phase(r, 11, 10), p13 = phase(r, 13, 12);
    if(st[S.FAILED] !== 0) fail(`${name}: a unit compile failed`);
    if(st[S.SITES] < 6) fail(`${name}: ${st[S.SITES]} chain sites compiled`);
    if(st[S.TARGET_MISMATCH] !== 0) fail(`${name}: ${st[S.TARGET_MISMATCH]} recorded call sites did not decode as CALL rel32 to the recorded target`);
    if(p2[C.CALLS] < ITER2 * 3) fail(`${name}: phase 2 made ${p2[C.CALLS]} chained calls after pv changed, want >= ${ITER2 * 3} (dependants recompiled)`);
    if(p2[C.FB_TOKEN] > 64) fail(`${name}: phase 2 fell back ${p2[C.FB_TOKEN]} times on a stale callee token`);
    if(st[S.DROPPED] < 2) fail(`${name}: ${st[S.DROPPED]} dependant units dropped; pv's change must drop mid and outer`);
    if(p2[C.CALLER_TOKEN] < 1) fail(`${name}: ${p2[C.CALLER_TOKEN]} caller-token exits in phase 2; wr rewrites outer every 8192 calls`);
    if(p2[C.RET_MISMATCH] < ITER2 / 2) fail(`${name}: ${p2[C.RET_MISMATCH]} RET mismatches in phase 2; retmod returns 2 bytes late on every call`);
    if(p2[C.RET_MATCHED] < ITER2 * 4) fail(`${name}: ${p2[C.RET_MATCHED]} matched RETs in phase 2`);
    if(p3[C.FB_DEPTH] < REC_ITER / 4) fail(`${name}: phase 3 hit the depth cap ${p3[C.FB_DEPTH]} times, want >= ${REC_ITER / 4}`);
    if(p3[C.CALLS] < REC_ITER * 10) fail(`${name}: phase 3 made ${p3[C.CALLS]} chained self calls`);
    if(p4[C.CALLS] < LONG_K) fail(`${name}: phase 4 made ${p4[C.CALLS]} chained calls`);
    const maxRetired = r === deadline ? DEADLINE + 512 : 2 * LOOP_COUNTER + 4 * LONG_N + 64;
    if(st[S.MAX_RETIRED] > maxRetired) fail(`${name}: one unit entry retired ${st[S.MAX_RETIRED]} instructions, want <= ${maxRetired}; the chain must share the loop budget and the slice`);
    if(st[S.MATCHED] + st[S.EXITED] !== st[S.ENTRIES]) fail(`${name}: matched + exited != entries`);
    if(r.countAfterOff !== 0) fail(`${name}: ${r.countAfterOff} units left after disabling`);
    const s = r.checks[6]?.extra ?? {};
    for(const k of ["outer", "mid", "sum", "cfadd", "setcf", "pv", "retmod", "wr"])
        if(!(s[k]?.slot < 0xFFFF && s[k]?.chain < 0xFFFF)) fail(`${name}: no chain unit for ${k} before phase 2 (${JSON.stringify(s[k])})`);
    // A chained CALL at a unit's first instruction that falls back (ent's target is not in the
    // TLB after the CR3 reload) leaves with nothing retired; the dispatch must continue on the
    // ordinary path instead of entering the unit again at the same EIP.
    if(p11[C.FB_MAPPING] < P_ITER / 2) fail(`${name}: phase 11: ${p11[C.FB_MAPPING]} mapping fallbacks at ent's first instruction, want >= ${P_ITER / 2}`);
    if(phaseStat(r, S.NO_PROGRESS, 11, 10) < P_ITER / 2) fail(`${name}: phase 11: ${phaseStat(r, S.NO_PROGRESS, 11, 10)} unit exits without progress continued on the ordinary path, want >= ${P_ITER / 2}`);
    // Urgent exit requested from inside a chained unit (uc's MMIO read): no chained call commits
    // until the host runs again, and the CALL that follows the read falls back.
    const u = r.urgent;
    if(!u.before || !u.after) fail(`${name}: the urgent-exit request did not run (${JSON.stringify(u)})`);
    else
    {
        if(u.after.counters[C.CALLS] !== u.before.counters[C.CALLS]) fail(`${name}: ${u.after.counters[C.CALLS] - u.before.counters[C.CALLS]} chained calls committed after the urgent-exit request`);
        if(u.after.counters[C.FB_BUDGET] <= u.before.counters[C.FB_BUDGET]) fail(`${name}: the chained call after the urgent-exit request did not fall back`);
    }
    if(c[C.TOKEN_AFTER_PUSH] < K_SPECIALS / 2) fail(`${name}: ${c[C.TOKEN_AFTER_PUSH]} committed exits after the CALL's push dropped kv, want >= ${K_SPECIALS / 2}`);
    if(p13[C.CALLS] < UC_ITER / 4) fail(`${name}: phase 13 made ${p13[C.CALLS]} chained calls; the chain must resume after the urgent exit`);
}
{
    const s = tails.checks[9]?.extra?.pt;
    if(!(s?.chain < 0xFFFF)) fail(`tails: pt is not a chain unit (${JSON.stringify(s)})`);
    const p5 = phase(tails, 5, 9);
    if(p5[C.CALLS] < PT_ITER) fail(`tails: phase 5 made ${p5[C.CALLS]} chained calls from the page tail, want >= ${PT_ITER}`);
    if(p5[C.RET_MATCHED] < PT_ITER) fail(`tails: phase 5 matched ${p5[C.RET_MATCHED]} RETs, want >= ${PT_ITER}`);
    if(deadline.counters[C.FB_BUDGET] < 1) fail("deadline: no chained call fell back at the slice deadline");
}
{
    const a = andEsp.checks[6]?.extra?.andesp, b = priv.checks[6]?.extra?.andesp;
    if(!(a?.slot < 0xFFFF)) fail(`andEsp: the and-esp frame was not admitted (${JSON.stringify(a)})`);
    if(b?.slot !== 0xFFFFFFFF) fail(`private: the and-esp frame was admitted without the option (${JSON.stringify(b)})`);
}
for(const [name, r] of chainArms.concat([["units", units]]))
    if(r.fnBytes.some((b, i) => b !== off.fnBytes[i])) fail(`${name}: function bytes differ from the page-JIT arm`);

const FAULT_ORDER = ["g1"];
const faultInterp = await run({ name: "faultInterp", build: build_fault_image, jit: false });
const faultOff = await run({ name: "faultOff", build: build_fault_image, gates: GATES_FAULT_OFF });
const faultOn = await run({ name: "faultOn", build: build_fault_image, units: true, chain: 2, mutation: MUTATION, gates: GATES_FAULT });
const FF = faultOn.img.F;
const fadj = r => adjTotal(r, faultInterp, FAULT_ORDER);
for(const [name, r] of [["faultInterp", faultInterp], ["faultOff", faultOff], ["faultOn", faultOn]])
{
    if(r.status !== "halt") fail(`${name}: no halt (${r.status})`);
    if(r.mark !== FF.load) fail(`${name}: handler saw eip ${hex(r.mark)}, want the load at ${hex(FF.load)}`);
    check_gates(name, r, faultInterp, FAULT_ORDER);
    if(((r.hook - fadj(r)) >>> 0) !== faultInterp.hook) fail(`${name}: count at the exception hook ${r.hook} != interpreter ${faultInterp.hook} (+ gate rounds)`);
}
if(((faultOn.counter - fadj(faultOn)) - faultInterp.counter) !== ((faultOff.counter - fadj(faultOff)) - faultInterp.counter)) fail(`faultOn: after the fault the chain counts ${faultOn.counter - fadj(faultOn) - faultInterp.counter} more than the interpreter, the page JIT ${faultOff.counter - fadj(faultOff) - faultInterp.counter}`);
if(faultOn.counters[C.CALLEE_EXITED] < 1) fail("faultOn: the fault in the chained callee did not propagate EXITED to its caller");

const ss16Interp = await run({ name: "ss16Interp", build: build_ss16_image, jit: false });
const ss16 = await run({ name: "ss16", build: build_ss16_image, units: true, chain: 2, mutation: MUTATION, gates: GATES_SS16 });
{
    const want = (() => { let e = 0; for(let c = SS16_ITER; c > 0; c--) e = (e + 2 * c + 1) >>> 0; return e; })();
    for(const [name, r] of [["ss16Interp", ss16Interp], ["ss16", ss16]])
    {
        if(r.status !== "halt") fail(`${name}: no halt (${r.status})`);
        if(r.edi !== want) fail(`${name}: checksum ${hex(r.edi)} != expected ${hex(want)}`);
        check_gates(name, r, ss16Interp, ["g1"]);
        if(((r.counter - adjTotal(r, ss16Interp, ["g1"])) >>> 0) !== ss16Interp.counter) fail(`${name}: retired ${r.counter} != interpreter ${ss16Interp.counter} (+ gate rounds)`);
    }
    if(ss16.stats[S.PUBLISHED] !== 0) fail(`ss16: ${ss16.stats[S.PUBLISHED]} units published on a 16-bit stack segment`);
    if(ss16.stats[S.STATE] < 1) fail("ss16: the 16-bit stack segment was never refused (the test did not reach admission)");
}

for(const [name, r] of chainArms.concat([["units", units], ["faultOn", faultOn], ["off", off], ["ss16", ss16]]))
    if(r.leak.free !== r.leak.initial || r.leak.pending !== 0) fail(`${name}: table slots ${r.leak.free} free after clear, ${r.leak.initial} before the run (pending ${hex(r.leak.pending)})`);

const sum4 = r => ({ entries: r.stats[S.ENTRIES], matched: r.stats[S.MATCHED], exited: r.stats[S.EXITED], published: r.stats[S.PUBLISHED],
    sites: r.stats[S.SITES], unchained: r.stats[S.UNCHAINED], dropped: r.stats[S.DROPPED], stalePublish: r.stats[S.STALE_PUBLISH],
    maxRetired: r.stats[S.MAX_RETIRED], noProgress: r.stats[S.NO_PROGRESS], targetMismatch: r.stats[S.TARGET_MISMATCH],
    gateRounds: Object.fromEntries(Object.entries(r.gates).map(([k, g]) => [k, g.reads.length - 1])),
    counters: Object.fromEntries(Object.entries(C).map(([k, i]) => [k, r.counters[i]])) });
const report = {
    mutation: MUTATION, retired: interp.counter,
    units: sum4(units), canonical: sum4(canonical), private: sum4(priv), andEsp: sum4(andEsp), tails: sum4(tails), deadline: sum4(deadline),
    fault: { hook: faultOn.hook, interpreterHook: faultInterp.hook, after: faultOn.counter, interpreterAfter: faultInterp.counter, pageJitAfter: faultOff.counter, ...sum4(faultOn) },
    ss16: { published: ss16.stats[S.PUBLISHED], refusedState: ss16.stats[S.STATE] },
    leaks: { private: priv.leak, fault: faultOn.leak },
};
if(failures.length)
{
    console.error("jit-fn-unit-chain " + JSON.stringify(report));
    for(const f of failures) console.error("FAIL: " + f);
    process.exit(1);
}
console.log("jit-fn-unit-chain " + JSON.stringify(report));
process.exit(0);
