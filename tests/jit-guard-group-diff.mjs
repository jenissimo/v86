#!/usr/bin/env node
// Differential oracle for guard groups (set_jit_guard_groups).
//
// With groups on, an access whose page an earlier access of the same (segment, base, index,
// scale) key already validated runs without its own TLB check (a member), as long as the
// anchor's widened check passed; otherwise it takes its normal guarded path. What can go wrong,
// and what each part of the workload is built to catch:
//
//   - a group that outlives a base write: a linked-list walk over nodes scattered across pages
//     (several remapped to non-adjacent frames, decoys in the identity frames), reloading the
//     base from the node itself;
//   - offset tracking: stack frames (push/pop/call/ret/leave, esp- and ebp-relative locals) whose
//     frames straddle a page boundary into a remapped stack page; pointer-bump and indexed loops;
//   - every access kind: R, W, RMW, x87 m64/m32/m32int operands, an FS-based segment key whose
//     base is not page-aligned, absolute operands;
//   - kills: a PTE rewrite + INVLPG between two accesses of one key, also inside a loop so the
//     group must restart every iteration; a guest PTE clear + INVLPG; a HOST decommit (PTE
//     present bit cleared + TLB flush inside an OUT) followed by a group that crosses into the
//     decommitted page — the member there must fault into the guest #PF handler;
//   - code: a group anchored by a READ of a page that holds compiled code, whose member WRITES
//     patch that code (the write must still reach jit_dirty_page), and a host write +
//     jit_dirty_cache (the writeGuestCode path).
//
// Arms: interpreter, JIT groups off, JIT groups on, JIT groups on with runtime counters. Final
// registers, the guest checksum, the #PF count and a hash of every data frame must match the
// interpreter, and the ON arms must have compiled members and executed them on the fast path.
//
// Negative controls (each must DIVERGE in the ON arm):
//   --nc-skip-ok        members take the fast path without testing ok
//   --nc-no-base-kill   a base-register write neither ends nor shifts a group
//   --nc-no-barrier     barriers (kill-all instructions, non-transparent helper calls) are ignored.
//                       NOT expected to diverge: every kill-all instruction is also a JIT block
//                       boundary, and the block after it is a dispatcher entry that starts empty,
//                       so the barrier is the second of two independent kills.
//
//   node tests/jit-guard-group-diff.mjs [seedStart] [seedCount] [--engine <v86.wasm>] [--nc-...]

import path from "node:path";
import url from "node:url";

const __dirname = path.dirname(url.fileURLToPath(import.meta.url));
const { V86 } = await import("../build/libv86.mjs");

const args = process.argv.slice(2);
const flag = n => args.includes(n);
const opt = n => { const i = args.indexOf(n); return i >= 0 ? args[i + 1] : undefined; };
const positional = args.filter((a, i) => !a.startsWith("--") && !(i > 0 && ["--engine", "--cases"].includes(args[i - 1])));
const SEED_START = parseInt(positional[0] ?? "1", 10);
const SEED_COUNT = parseInt(positional[1] ?? "6", 10);
const ENGINE = opt("--engine") ? path.resolve(opt("--engine")) : path.join(__dirname, "../build/v86.wasm");
const NC = flag("--nc-skip-ok") ? 1 : flag("--nc-no-base-kill") ? 2 : flag("--nc-no-barrier") ? 8 : 0;
// --cases a,b,...: only these parts of the workload (a negative control is only conclusive
// when it diverges on the case it is about)
const ALL_CASES = ["struct", "stack", "array", "x87", "fs", "remap", "gclear", "decommit", "smc"];
const CASES = new Set(opt("--cases") ? opt("--cases").split(",") : ALL_CASES);

const BASE = 0x100000;
const ENTRY_OFF = 0x40;
const SMCP = BASE + 0x9000;           // f at +0x100: mov eax, imm32; ret
const GDT = BASE + 0xA000, GDTR = GDT + 0x40, IDTR = GDT + 0x48, IDT = BASE + 0xA800;
const VARS = BASE + 0xB000;
const CHK = VARS, CTR = VARS + 4, PFC = VARS + 8, HEAD = VARS + 12, LOOPC = VARS + 16;
const DATA = BASE + 0x10000, DATA_PAGES = 16;
const REMAP_DATA = [3, 7, 11];        // data pages served from ALT frames
const ARR = DATA + 0x2F80;            // array crossing into page 3 (remapped)
const ARR2 = DATA + 0x6FA0;
const DBL = DATA + 0x9100;
const FSBASE = DATA + 0x5FF0;
const STACK_LO = BASE + 0x32000, STACK_TOP = BASE + 0x33000 + 0x14;
const VP1 = BASE + 0x38000, VP2 = BASE + 0x3A000, VP3 = BASE + 0x3B000;   // VP2-0x1000 stays mapped
const F1 = 0x700000, F2 = 0x701000;   // VP1 toggles between these frames
const ALT = 0x600000;                 // ALT + k*0x1000: backing frames of remapped pages
const IMG_SIZE = 0x3C000;
const PD = 0x300000, PT0 = 0x301000, PT1 = 0x302000;
const pte = va => PT0 + (va >>> 12) * 4;
const OUTER = 24000;
const DONE_PORT = 0x9999, HOST_PATCH_PORT = 0x9990, FENCE_PORT = 0x9991, DECOMMIT_PORT = 0x9992;
const MEM_SIZE = 32 * 1024 * 1024;
const TIMEOUT_MS = 240_000;

function mulberry32(a) {
    return () => {
        a |= 0; a = a + 0x6D2B79F5 | 0;
        let t = Math.imul(a ^ a >>> 15, 1 | a);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
}
const d32 = v => { v >>>= 0; return [v & 255, v >> 8 & 255, v >> 16 & 255, v >>> 24]; };
const d16 = v => [v & 255, v >> 8 & 255];

// ── a tiny x86-32 assembler: enough forms for the workload ─────────────────────────────────
const EAX = 0, ECX = 1, EDX = 2, EBX = 3, ESP = 4, EBP = 5, ESI = 6, EDI = 7;
const M = (base, disp = 0, index = null, scale = 0, seg = null) => ({ base, disp, index, scale, seg });
const ABS = (disp, seg = null) => ({ base: null, disp, index: null, scale: 0, seg });
function modrm(reg, m) {
    const pre = m.seg === "fs" ? [0x64] : [];
    if (m.base === null && m.index === null) return [pre, [0x05 | reg << 3, ...d32(m.disp)]];
    if (m.index !== null || m.base === ESP) {
        const idx = m.index ?? 4, ss = m.index !== null ? m.scale : 0;
        if (m.base === null) return [pre, [0x04 | reg << 3, ss << 6 | idx << 3 | 5, ...d32(m.disp)]];
        return [pre, [0x84 | reg << 3, ss << 6 | idx << 3 | m.base, ...d32(m.disp)]];
    }
    return [pre, [0x80 | reg << 3 | m.base, ...d32(m.disp)]];
}
const op = (opc, reg, m, tail = []) => { const [pre, mr] = modrm(reg, m); return [...pre, ...opc, ...mr, ...tail]; };

class Asm {
    constructor(img) { this.img = img; this.o = 0; this.labels = {}; this.fix = []; }
    at(off) { this.o = off; return this; }
    label(n) { this.labels[n] = BASE + this.o; return this; }
    b(...bytes) { for (const x of bytes.flat(3)) this.img[this.o++] = x & 0xFF; return this; }
    rel32(n) { this.fix.push({ at: this.o, to: n }); this.o += 4; return this; }
    // instructions
    movRI(r, i) { return this.b(0xB8 + r, d32(i)); }
    movRM(r, m) { return this.b(op([0x8B], r, m)); }
    movMR(m, r) { return this.b(op([0x89], r, m)); }
    movMI(m, i) { return this.b(op([0xC7], 0, m, d32(i))); }
    movRR(d, s) { return this.b(0x89, 0xC0 | s << 3 | d); }
    addRM(r, m) { return this.b(op([0x03], r, m)); }
    addMR(m, r) { return this.b(op([0x01], r, m)); }
    subRM(r, m) { return this.b(op([0x2B], r, m)); }
    xorRM(r, m) { return this.b(op([0x33], r, m)); }
    xorMR(m, r) { return this.b(op([0x31], r, m)); }
    xorMI(m, i) { return this.b(op([0x81], 6, m, d32(i))); }
    andMI(m, i) { return this.b(op([0x81], 4, m, d32(i))); }
    orMI(m, i) { return this.b(op([0x81], 1, m, d32(i))); }
    addRI(r, i) { return this.b(0x81, 0xC0 | r, d32(i)); }
    subRI(r, i) { return this.b(0x81, 0xE8 | r, d32(i)); }
    addRI8(r, i) { return this.b(0x83, 0xC0 | r, i & 0xFF); }
    subRI8(r, i) { return this.b(0x83, 0xE8 | r, i & 0xFF); }
    cmpRI(r, i) { return this.b(0x81, 0xF8 | r, d32(i)); }
    cmpMI(m, i) { return this.b(op([0x81], 7, m, d32(i))); }
    testMI(m, i) { return this.b(op([0xF7], 0, m, d32(i))); }
    incM(m) { return this.b(op([0xFF], 0, m)); }
    decM(m) { return this.b(op([0xFF], 1, m)); }
    incR(r) { return this.b(0x40 + r); }
    decR(r) { return this.b(0x48 + r); }
    leaRM(r, m) { return this.b(op([0x8D], r, m)); }
    xchgRM(r, m) { return this.b(op([0x87], r, m)); }
    movzxB(r, m) { return this.b(op([0x0F, 0xB6], r, m)); }
    movB_MR(m, r) { return this.b(op([0x88], r, m)); }
    imulRM(r, m) { return this.b(op([0x0F, 0xAF], r, m)); }
    rolM1(m) { return this.b(op([0xD1], 0, m)); }
    pushR(r) { return this.b(0x50 + r); }
    popR(r) { return this.b(0x58 + r); }
    pushM(m) { return this.b(op([0xFF], 6, m)); }
    pushI(i) { return this.b(0x68, d32(i)); }
    call(n) { return this.b(0xE8).rel32(n); }
    jmp(n) { return this.b(0xE9).rel32(n); }
    jcc(cc, n) { return this.b(0x0F, 0x80 + cc).rel32(n); }
    ret() { return this.b(0xC3); }
    leave() { return this.b(0xC9); }
    fldM64(m) { return this.b(op([0xDD], 0, m)); }
    faddM64(m) { return this.b(op([0xDC], 0, m)); }
    fmulM64(m) { return this.b(op([0xDC], 1, m)); }
    fstM64(m) { return this.b(op([0xDD], 2, m)); }
    fstpM64(m) { return this.b(op([0xDD], 3, m)); }
    fildM32(m) { return this.b(op([0xDB], 0, m)); }
    fistpM32(m) { return this.b(op([0xDB], 3, m)); }
    fiaddM32(m) { return this.b(op([0xDA], 0, m)); }
    fldM32(m) { return this.b(op([0xD9], 0, m)); }
    fstpM32(m) { return this.b(op([0xD9], 3, m)); }
    invlpg(m) { return this.b(op([0x0F, 0x01], 7, m)); }
    out(port) { return this.b(0x66, 0xBA, d16(port), 0xEE); }
    resolve() {
        const dv = new DataView(this.img.buffer);
        for (const f of this.fix) {
            const t = this.labels[f.to]; if (t === undefined) throw new Error("label " + f.to);
            dv.setInt32(f.at, t - (BASE + f.at + 4), true);
        }
    }
}
const JL = 0xC, JNZ = 0x5, JZ = 0x4;

function buildImage(seed) {
    const rnd = mulberry32(seed);
    const R = n => Math.floor(rnd() * n);
    const rnd32 = () => (rnd() * 2 ** 32) >>> 0;
    const img = new Uint8Array(IMG_SIZE);
    for (let i = 0; i < IMG_SIZE; i++) img[i] = R(256);          // random data everywhere
    const dv = new DataView(img.buffer);
    const w32 = (va, v) => dv.setUint32(va - BASE, v >>> 0, true);
    const MAGIC = 0x1BADB002, FLAGS = 0x10000;
    dv.setUint32(0x00, MAGIC, true);
    dv.setUint32(0x04, FLAGS, true);
    dv.setUint32(0x08, (-(MAGIC + FLAGS)) >>> 0, true);
    dv.setUint32(0x0c, BASE, true);
    dv.setUint32(0x10, BASE, true);
    dv.setUint32(0x14, BASE + IMG_SIZE, true);
    dv.setUint32(0x18, BASE + IMG_SIZE, true);
    dv.setUint32(0x1c, BASE + ENTRY_OFF, true);

    // GDT: null, code 0x08, data 0x10, fs 0x18 (base FSBASE)
    const gdt = [0, 0, 0xFFFF, 0x00CF9A00, 0xFFFF, 0x00CF9200,
                 (0xFFFF | (FSBASE & 0xFFFF) << 16) >>> 0, ((FSBASE >>> 16 & 0xFF) | 0x92 << 8 | 0xCF << 16 | (FSBASE >>> 24) << 24) >>> 0];
    gdt.forEach((v, i) => w32(GDT + 4 * i, v));
    dv.setUint16(GDTR - BASE, 31, true); w32(GDTR + 2, GDT);
    dv.setUint16(IDTR - BASE, 256 * 8 - 1, true); w32(IDTR + 2, IDT);
    for (let i = 0; i < 256 * 8; i++) img[IDT - BASE + i] = 0;
    for (let i = 0; i < 0x40; i++) img[VARS - BASE + i] = 0;

    // circular list of 24-byte nodes scattered over the data pages; some straddle a page end
    const N = 24, nodes = [];
    const used = new Set();
    while (nodes.length < N) {
        const pg = R(DATA_PAGES);
        if (pg === 9) continue;                                    // DBL page
        const off = R(3) === 0 ? 0x1000 - 4 - 4 * R(5) : 0x40 + 4 * R(0x3C0);
        const a = DATA + pg * 0x1000 + off;
        if (a + 24 > DATA + DATA_PAGES * 0x1000 || used.has(a >> 5)) continue;
        if ((a >= ARR && a < ARR + 0x200) || (a >= ARR2 && a < ARR2 + 0x100) || (a >= FSBASE && a < FSBASE + 0x40)) continue;
        used.add(a >> 5); nodes.push(a);
    }
    nodes.forEach((a, k) => { w32(a, nodes[(k + 1) % N]); for (let f = 4; f < 24; f += 4) w32(a + f, rnd32()); });
    w32(HEAD, nodes[0]);
    // exact x87 inputs: small integers as doubles / floats
    dv.setFloat64(DBL - BASE, 3 + R(50), true);
    dv.setFloat64(DBL + 8 - BASE, 5 + R(50), true);
    dv.setFloat64(DBL + 16 - BASE, 2, true);
    dv.setFloat32(DBL + 52 - BASE, 7 + R(9), true);

    const a = new Asm(img);
    // ── f (SMC target) and its data words, on its own page
    a.at(SMCP - BASE + 0x100).label("f").movRI(EAX, rnd32()).ret();
    // ── #PF handler: set the faulting page present again, count, retry
    a.at(0x6000).label("pf");
    a.pushR(EAX).pushR(EBX);
    a.b(0x0F, 0x20, 0xD0);                                          // mov eax, cr2
    a.b(0xC1, 0xE8, 12).b(0xC1, 0xE0, 2).addRI(EAX, PT0);           // eax = &pte
    a.orMI(M(EAX, 0), 1);
    a.b(0x0F, 0x20, 0xD0).invlpg(M(EAX, 0));
    a.incM(ABS(PFC));
    a.popR(EBX).popR(EAX).addRI8(ESP, 4).b(0xCF);                  // drop error code; iret

    // ── routines
    a.at(0x1000).label("r_struct");
    a.movRM(ESI, ABS(HEAD)).movMI(ABS(LOOPC), 2 * N);
    a.label("rs_loop");
    a.movRM(EAX, M(ESI, 4)).addRM(EBX, M(ESI, 8)).xorRM(ECX, M(ESI, 12));
    a.addMR(M(ESI, 16), EAX).incM(M(ESI, 20));
    a.movRM(EDX, M(ESI, 16)).addMR(ABS(CHK), EDX);
    a.movRM(ESI, M(ESI, 0));                                        // base reload
    a.addRM(EBX, M(ESI, 8)).xorRM(EDX, M(ESI, 4));                  // same block, new node
    a.decM(ABS(LOOPC)).jcc(JNZ, "rs_loop");
    a.ret();

    a.label("r_stack");
    a.pushR(EAX).pushR(EBX).pushR(ECX).call("F1").addRI8(ESP, 12).addRM(EDX, M(ESP, -4)).xorRM(EDX, M(ESP, -8)).ret();
    a.label("F1");
    a.pushR(EBP).movRR(EBP, ESP).subRI8(ESP, 40);
    a.movMR(M(EBP, -4), EAX).movMR(M(EBP, -8), ECX).movMR(M(ESP, 8), EBX);
    a.movRM(EAX, M(EBP, 8)).addRM(EAX, M(EBP, 12)).xorRM(EAX, M(EBP, 16));
    a.pushR(EAX).pushR(ESI).call("F2").addRI8(ESP, 8);
    a.addRM(EAX, M(EBP, -4)).addRM(EAX, M(ESP, 8));
    a.movMR(M(EBP, -12), EAX).incM(M(EBP, -12)).movRM(EAX, M(EBP, -12));
    a.leaRM(ESP, M(ESP, -8)).movMR(M(ESP, 0), ECX).leaRM(ESP, M(ESP, 8));
    a.leave().ret();
    a.label("F2");
    a.movRM(EAX, M(ESP, 4)).addRM(EAX, M(ESP, 8)).pushR(EBX).movRM(EBX, M(ESP, 8)).xorRM(EAX, M(ESP, 12));
    a.popR(EBX).b(0xC2, 0, 0);                                      // ret 0

    a.label("r_array");
    a.movRI(ESI, ARR).movRI(ECX, 0);
    a.label("ra1");
    a.movRM(EAX, M(ESI, 0, ECX, 2)).addMR(M(ESI, 0x100, ECX, 2), EAX).incR(ECX).cmpRI(ECX, 64).jcc(JL, "ra1");
    a.movRI(EDI, ARR2).movRI(EDX, 16);
    a.label("ra2");
    a.movRM(EAX, M(EDI, 0)).addRM(EAX, M(EDI, 4)).movMR(M(EDI, 8), EAX).addRI8(EDI, 12).decR(EDX).jcc(JNZ, "ra2");
    a.movzxB(EBX, M(ESI, 3)).movB_MR(M(ESI, 0x41), EBX).imulRM(EBX, M(ESI, 8));
    a.xchgRM(ECX, M(ESI, 12));
    a.addMR(ABS(CHK), EAX).ret();

    a.label("r_x87");
    a.movRI(ESI, DBL);
    a.movRM(EAX, ABS(CTR)).b(0x25, d32(0xFF)).movMR(M(ESI, 40), EAX);  // and eax, 0xFF
    a.fldM64(M(ESI, 0)).faddM64(M(ESI, 8)).fmulM64(M(ESI, 16)).fstM64(M(ESI, 24)).fstpM64(M(ESI, 32));
    a.fildM32(M(ESI, 40)).fiaddM32(M(ESI, 40)).fistpM32(M(ESI, 48));
    a.fldM32(M(ESI, 52)).fstpM32(M(ESI, 56));
    a.addRM(EBX, M(ESI, 48)).xorRM(EBX, M(ESI, 28)).addRM(EBX, M(ESI, 56)).ret();

    a.label("r_fs");
    a.movRM(EAX, ABS(0x08, "fs")).addRM(EAX, ABS(0x0C, "fs")).movMR(ABS(0x14, "fs"), EAX);
    a.movRI(EDI, 4).addRM(EAX, M(EDI, 0x10, null, 0, "fs")).movMR(M(EDI, 0x18, null, 0, "fs"), EAX);
    a.addMR(ABS(CHK), EAX).ret();

    // remap: a PTE rewrite + INVLPG between two accesses of one key, and inside a loop
    a.label("r_remap");
    a.movRM(EAX, ABS(VP1)).addRM(EBX, ABS(VP1 + 4));                // group on VP1 (frame F1 or F2)
    a.xorMI(ABS(pte(VP1)), (F1 ^ F2) >>> 0).invlpg(ABS(VP1));        // remap between two accesses
    a.addRM(ECX, ABS(VP1 + 8)).xorRM(EDX, ABS(VP1 + 12)).movMR(ABS(VP1 + 16), ECX);
    a.movRI(ESI, VP1).movRI(EDI, 5);
    a.label("rv1");
    a.movRM(EAX, M(ESI, 0)).addRM(EBX, M(ESI, 4));
    a.xorMI(ABS(pte(VP1)), (F1 ^ F2) >>> 0).invlpg(M(ESI, 0));
    a.addRM(ECX, M(ESI, 8)).addMR(M(ESI, 20), EBX).decR(EDI).jcc(JNZ, "rv1");
    a.ret();
    // gclear: reads of a page whose PTE the guest clears (+ INVLPG) now and then
    a.label("r_gclear");
    a.movRI(ESI, VP3).addRM(EDX, M(ESI, 0x10)).addRM(EDX, M(ESI, 0x14));
    a.movRM(EAX, ABS(CTR)).b(0x25, d32(255)).cmpRI(EAX, 7).jcc(JNZ, "no_gc");
    a.andMI(ABS(pte(VP3)), ~1).invlpg(ABS(VP3)).label("no_gc");
    a.addRM(EDX, M(ESI, 0x18)).xorRM(EDX, M(ESI, 0x1C)).ret();
    // decommit: a group crossing into a page the HOST decommits now and then
    a.label("r_decommit");
    a.movRM(EAX, ABS(CTR)).b(0x25, d32(511)).cmpRI(EAX, 3).jcc(JNZ, "no_dc").out(DECOMMIT_PORT).label("no_dc");
    a.movRI(ESI, VP2 - 8);                                          // crosses into VP2
    a.movRM(EAX, M(ESI, 0)).addRM(EAX, M(ESI, 4)).addRM(EAX, M(ESI, 8)).addRM(EAX, M(ESI, 12));
    a.addMR(ABS(CHK), EAX).ret();

    // ── driver
    a.at(ENTRY_OFF).label("entry");
    a.movRI(ESP, STACK_TOP);
    a.b(0x0F, 0x01, 0x15, d32(GDTR));                               // lgdt
    a.b(0xEA).rel32("reload_abs").b(d16(0x08));
    a.fix[a.fix.length - 1].absolute = true;
    a.label("reload");
    a.b(0x66, 0xB8, d16(0x10)).b(0x8E, 0xD8).b(0x8E, 0xC0).b(0x8E, 0xD0).b(0x8E, 0xE8);
    a.b(0x66, 0xB8, d16(0x18)).b(0x8E, 0xE0);
    // IDT gate 14 -> pf
    a.movRI(EAX, 0).movRI(EDX, 0);
    a.b(0xB8).rel32("pf_abs"); a.fix[a.fix.length - 1].absolute = true;
    a.b(0x66, 0x89, 0x05, d32(IDT + 14 * 8));                       // offset low
    a.b(0xC1, 0xE8, 16).b(0x66, 0x89, 0x05, d32(IDT + 14 * 8 + 6)); // offset high
    a.b(0x66, 0xC7, 0x05, d32(IDT + 14 * 8 + 2), d16(0x08));        // selector
    a.b(0xC6, 0x05, d32(IDT + 14 * 8 + 5), 0x8E);                   // type
    a.b(0x0F, 0x01, 0x1D, d32(IDTR));                               // lidt
    // identity paging 0..8 MiB, then the remaps
    a.movRI(EAX, PT0 | 3).movMR(ABS(PD), EAX);
    a.movRI(EAX, PT1 | 3).movMR(ABS(PD + 4), EAX);
    a.movRI(EAX, 3).movRI(ECX, 2048).movRI(EDX, PT0);
    a.label("pt").movMR(M(EDX, 0), EAX).addRI(EAX, 0x1000).addRI8(EDX, 4).decR(ECX).jcc(JNZ, "pt");
    REMAP_DATA.forEach((pg, k) => a.movMI(ABS(pte(DATA + pg * 0x1000)), (ALT + k * 0x1000) | 3));
    a.movMI(ABS(pte(STACK_LO)), (ALT + 3 * 0x1000) | 3);
    a.movMI(ABS(pte(VP1)), F1 | 3);
    a.movRI(EAX, PD).b(0x0F, 0x22, 0xD8);
    a.b(0x0F, 0x20, 0xC0).b(0x0D, d32(0x80000000)).b(0x0F, 0x22, 0xC0);
    a.b(0xDB, 0xE3);                                                // fninit
    for (const r of [EAX, ECX, EDX, EBX, ESI, EDI]) a.movRI(r, rnd32());
    a.movMI(ABS(CTR), OUTER);
    a.label("outer");
    for (const c of ["struct", "stack", "array", "x87", "fs", "remap", "gclear", "decommit"])
        if (CASES.has(c)) a.call("r_" + c);
    // periodic: SMC through a group, host patch, host decommit, guest PTE clear
    // SMC inline in the hot loop, so the (rarely taken) block is compiled with it: a READ anchor
    // on f's page (it holds compiled code), a member WRITE patching f, a member read
    if (CASES.has("smc")) {
        a.movRM(EAX, ABS(CTR)).b(0x25, d32(2047)).cmpRI(EAX, 100).jcc(JNZ, "no_smc");
        a.movRI(EDI, SMCP).movRM(EAX, M(EDI, 0x10));
        a.movRM(ECX, ABS(CTR)).movMR(M(EDI, 0x101), ECX);
        a.addRM(EAX, M(EDI, 0x20)).addMR(ABS(CHK), EAX);
        a.out(FENCE_PORT).label("no_smc");
    }
    a.movRM(EAX, ABS(CTR)).b(0x25, d32(4095)).cmpRI(EAX, 2000).jcc(JNZ, "no_hp").out(HOST_PATCH_PORT).label("no_hp");
    a.movRI(EDI, 32).label("fcall").call("f").addMR(ABS(CHK), EAX).decR(EDI).jcc(JNZ, "fcall");
    for (const r of [ECX, EDX, EBX, ESI]) a.xorMR(ABS(CHK), r);
    a.rolM1(ABS(CHK));
    a.decM(ABS(CTR)).jcc(JNZ, "outer");
    a.out(DONE_PORT).b(0xF4).b(0xEB, 0xFE);

    // absolute fixups (far jmp target, handler address)
    a.labels.reload_abs = null; a.labels.pf_abs = null;
    for (const f of a.fix) {
        if (!f.absolute) continue;
        const t = f.to === "reload_abs" ? a.labels.reload : a.labels.pf;
        dv.setUint32(f.at, t, true);
    }
    a.fix = a.fix.filter(f => !f.absolute);
    a.resolve();
    return { img, nodes };
}

function fnv(bytes, h = 0x811C9DC5) {
    for (let i = 0; i < bytes.length; i++) h = Math.imul(h ^ bytes[i], 0x01000193);
    return h >>> 0;
}

let currentFinish = null;
process.on("uncaughtException", e => {
    if (currentFinish) currentFinish("THREW " + String(e.message ?? e).slice(0, 80));
    else { console.error(e); process.exit(2); }
});

function run(build, cfg) {
    return new Promise(resolve => {
        const emulator = new V86({ autostart: false, memory_size: MEM_SIZE, disable_jit: cfg.jit ? 0 : 1,
                                   log_level: 0, wasm_path: cfg.engine ?? ENGINE });
        let done = false, timer, cpu, w;
        const finish = status => {
            if (done) return; done = true;
            currentFinish = null;
            clearTimeout(timer);
            try { emulator.stop(); } catch { }
            const regs = Array.from({ length: 8 }, (_, i) => cpu.reg32[i] >>> 0);
            const rd = (a, n) => emulator.read_memory(a, n);
            const u32 = a => { const b = rd(a, 4); return (b[0] | b[1] << 8 | b[2] << 16 | b[3] << 24) >>> 0; };
            let h = fnv(rd(DATA, DATA_PAGES * 0x1000));
            h = fnv(rd(ALT, 4 * 0x1000), h);
            h = fnv(rd(F1, 0x2000), h);
            h = fnv(rd(STACK_LO, 0x2000), h);
            h = fnv(rd(VP2 - 0x1000, 0x3000), h);
            h = fnv(rd(SMCP, 0x1000), h);
            if (process.env.GG_TRACE && cfg.groups && w.gg_trace_len) {
                const n = w.gg_trace_len(), p = w.gg_trace_ptr();
                console.log(Buffer.from(new Uint8Array(w.memory?.buffer ?? cpu.wasm_memory.buffer, p, n)).toString());
            }
            const stat = i => w?.get_jit_guard_group_stat ? w.get_jit_guard_group_stat(i) : -1;
            const rt = i => w?.get_jit_guard_group_rt ? w.get_jit_guard_group_rt(i) : -1;
            resolve({ status, regs, chk: u32(CHK), pf: u32(PFC), mem: h,
                      planned: stat(0), withRoles: stat(1), abandoned: stat(2), anchors: stat(3), members: stat(4),
                      fast: rt(0) + rt(1) + rt(2), slow: rt(3) + rt(4) + rt(5),
                      accesses: rt(8) + rt(9) + rt(10) });
        };
        emulator.add_listener("emulator-loaded", () => {
            cpu = emulator.v86.cpu; w = cpu.wm.exports;
            cpu.reboot_internal(); cpu.reset_memory();
            if (cfg.selftest && w.gg_plan_selftest) {
                // planner self-test over synthetic streams, and proof that it can fail
                const ok = w.gg_plan_selftest() >>> 0;
                w.set_jit_guard_groups_debug(2);
                const mutated = w.gg_plan_selftest() >>> 0;
                w.set_jit_guard_groups_debug(0);
                console.log(`planner self-test: failures=0x${ok.toString(16)}; with the no-base-kill mutation 0x${mutated.toString(16)}`);
                if (ok !== 0) fail("planner self-test failed: 0x" + ok.toString(16));
                if (mutated === 0) fail("planner self-test cannot fail: the no-base-kill mutation passed it");
            }
            if (cfg.jit) {
                const knobs = new Map([[5, 1], [11, 1], [12, 1], [13, 1], [22, 1]]);
                for (const kv of (process.env.KNOBS || "").split(",").filter(Boolean)) {
                    const [i, v] = kv.split("=").map(Number); knobs.set(i, v);
                }
                for (const [i, v] of knobs) cpu.set_jit_config(i, v);
                if (w.set_jit_page_tails) w.set_jit_page_tails(1);
                if (cfg.groups && !w.set_jit_guard_groups) { resolve({ status: "NO-SWITCH" }); return; }
                if (w.set_jit_guard_groups) {
                    w.set_jit_guard_groups(cfg.groups ? 1 : 0);
                    w.set_jit_guard_groups_debug(cfg.debug ?? 0);
                    if (process.env.GG_EXTENT) w.set_jit_guard_groups_params(+process.env.GG_EXTENT, 24);
                    w.jit_guard_group_stats_reset();
                    if (process.env.GG_TRACE && cfg.groups) w.set_jit_guard_groups_trace(1);
                }
                cpu.jit_clear_cache?.();
            }
            cpu.load_multiboot(build.img.buffer);
            // remapped pages: the real bytes in their ALT frames, decoys in the identity frames
            [...REMAP_DATA.map(pg => DATA + pg * 0x1000), STACK_LO].forEach((va, k) => {
                const real = emulator.read_memory(va, 0x1000);
                emulator.write_memory(real, ALT + k * 0x1000);
                emulator.write_memory(real.map(x => x ^ 0x5A), va);
            });
            const f1 = emulator.read_memory(VP1, 0x1000);
            emulator.write_memory(f1, F1);
            emulator.write_memory(f1.map(x => x ^ 0xA5), F2);
            cpu.io.register_write(DONE_PORT, cpu, () => finish("done"));
            cpu.io.register_write(FENCE_PORT, cpu, () => {});
            cpu.io.register_write(HOST_PATCH_PORT, cpu, () => {
                const v = (0xC0DE0000 | (cpu.reg32[0] & 0xFFFF)) >>> 0;
                cpu.mem8.set(d32(v), SMCP + 0x101);                     // like a JS HLE write
                w.jit_dirty_cache(SMCP + 0x101, SMCP + 0x105);
            });
            cpu.io.register_write(DECOMMIT_PORT, cpu, () => {
                const p = pte(VP2);
                const cur = emulator.read_memory(p, 4);
                cpu.mem8.set([cur[0] & ~1, cur[1], cur[2], cur[3]], p);  // present bit off
                w.jit_aot_flush_tlb();
            });
            timer = setTimeout(() => finish("TIMEOUT"), TIMEOUT_MS);
            currentFinish = finish;
            emulator.run();
        });
    });
}

let failures = 0;
const fail = m => { failures++; console.error("FAIL " + m); };
const sig = r => `${r.status} chk=${r.chk?.toString(16)} pf=${r.pf} mem=${r.mem?.toString(16)} regs=${r.regs?.map(x => x.toString(16)).join(",")}`;
console.log(`engine ${ENGINE}${NC ? `  NEGATIVE CONTROL debug=${NC}` : ""}  cases ${[...CASES].join(",")}`);

for (let seed = SEED_START; seed < SEED_START + SEED_COUNT; seed++) {
    const build = buildImage(seed);
    const interp = await run(build, { jit: false, selftest: seed === SEED_START });
    if (interp.status !== "done") { fail(`seed ${seed}: interpreter did not finish (${interp.status})`); continue; }
    if (NC) {
        const on = await run(build, { jit: true, groups: true, debug: NC });
        const same = sig(on) === sig(interp);
        const note = ` (members ${on.members}, planned ${on.planned}, abandoned ${on.abandoned})`;
        if (!same) console.log(`seed ${seed}: negative control diverges, as it must${note}`);
        else fail(`seed ${seed}: negative control did NOT diverge${note}`);
        continue;
    }
    const off = await run(build, { jit: true, groups: false });
    const on = await run(build, { jit: true, groups: true });
    const cnt = await run(build, { jit: true, groups: true, debug: 4 });
    if (on.status === "NO-SWITCH") { console.log(`seed ${seed}: engine has no set_jit_guard_groups`); continue; }
    const note = ` (pf=${interp.pf} modules planned=${on.planned} withRoles=${on.withRoles} abandoned=${on.abandoned}` +
                 ` anchors=${on.anchors} members=${on.members} memberFast=${cnt.fast} memberSlow=${cnt.slow}` +
                 ` guardedAccesses=${cnt.accesses})`;
    const ok = [off, on, cnt].every(r => sig(r) === sig(interp));
    if (ok) console.log(`seed ${seed}: ok${note}`);
    else {
        fail(`seed ${seed}: DIVERGENCE${note}`);
        console.error(`  interp ${sig(interp)}\n  off    ${sig(off)}\n  on     ${sig(on)}\n  count  ${sig(cnt)}`);
    }
    if (off.members > 0) fail(`seed ${seed}: groups OFF compiled ${off.members} members`);
    if (!(on.members > 0)) fail(`seed ${seed}: groups ON compiled no member — the arm did not exercise the feature`);
    if (!(cnt.fast > 0) || !(cnt.slow > 0)) fail(`seed ${seed}: members ran fast ${cnt.fast} / slow ${cnt.slow} times — both arms must be exercised`);
    if (!(interp.pf > 0) && (CASES.has("gclear") || CASES.has("decommit"))) fail(`seed ${seed}: no #PF was taken — the decommit cases did not fire`);
}
if (failures) { console.error(`${failures} failure(s)`); process.exit(1); }
console.log(NC ? "negative control: divergence detected in every seed" : "all seeds consistent");
process.exit(0);
