#!/usr/bin/env node
// Differential oracle for the TRANSPARENT-helper contract of guard groups.
//
// A group member reuses the anchor's validated frame across every helper call that
// analysis::gg_call_is_transparent accepts; the safe_read*/safe_write* slow paths are on that
// list. Those slow paths are where a guarded access ends up when its page is not in the TLB,
// crosses a page, faults, or is device memory, so each of those ways out is exercised inside a
// live group, followed by members that must observe what the slow path caused:
//
//   mmio   a 32-bit store and a load to a JS-handled MMIO page between the anchor and its
//          members; the store handler "DMAs" into the group's frame (members read live memory);
//   pf     a #PF raised by the slow path inside a group (read, write, RMW, page-crossing read,
//          fnstenv through writable_or_pagefault_jit); the guest #PF handler REMAPS the group's
//          page before the faulting instruction is retried;
//   irq    LAPIC writes (TPR, a self-IPI ICR) inside a group with IF=0 — mmap_write32 calls
//          handle_irqs, which must not deliver — then POPF takes the IRQ at its module exit and
//          the handler remaps the group's page.
//
// Arms: interpreter, JIT groups off, JIT groups on, JIT groups on with runtime counters. Final
// registers, counters (#PF, IRQ, MMIO accesses) and a hash of every data frame must match.
//
// Negative control (must DIVERGE in the ON arm, and must NOT diverge with groups off):
//   --nc-mmio-remap   the MMIO store handler also rewrites the group page's PTE and flushes the
//                     TLB — a transparent helper that changes a translation, which is exactly
//                     what the contract forbids. Run with --cases mmio.
//
// Characterization (opt-in, not a guard-group property):
//   --cases inline    a self-IPI written to the LAPIC with IF=1 inside a group: handle_irqs then
//                     delivers the interrupt INSIDE safe_write32_slow_jit, with the module's
//                     registers still in locals. Asserted: if groups-off matches the
//                     interpreter, groups-on must too. Reported: whether groups-off does.
//
//   node tests/jit-guard-group-transparent-diff.mjs [seedStart] [seedCount] [--engine <v86.wasm>]
//        [--cases mmio,pf,irq] [--nc-mmio-remap]

import path from "node:path";
import url from "node:url";

const __dirname = path.dirname(url.fileURLToPath(import.meta.url));
const { V86 } = await import("../build/libv86.mjs");

const args = process.argv.slice(2);
const flag = n => args.includes(n);
const opt = n => { const i = args.indexOf(n); return i >= 0 ? args[i + 1] : undefined; };
const positional = args.filter((a, i) => !a.startsWith("--") && !(i > 0 && ["--engine", "--cases"].includes(args[i - 1])));
const SEED_START = parseInt(positional[0] ?? "1", 10);
const SEED_COUNT = parseInt(positional[1] ?? "3", 10);
const ENGINE = opt("--engine") ? path.resolve(opt("--engine")) : path.join(__dirname, "../build/v86.wasm");
const NC_REMAP = flag("--nc-mmio-remap");
const DEFAULT_CASES = ["mmio", "pf", "irq"];
const CASES = new Set(opt("--cases") ? opt("--cases").split(",") : DEFAULT_CASES);
const INLINE = CASES.has("inline");

const BASE = 0x100000;
const ENTRY_OFF = 0x40;
const GDT = BASE + 0xA000, GDTR = GDT + 0x40, IDTR = GDT + 0x48, IDT = BASE + 0xA800;
const VARS = BASE + 0xB000;
const CHK = VARS, CTR = VARS + 4, PFC = VARS + 8, IRQC = VARS + 12, IRQC2 = VARS + 16;
const VPG = BASE + 0x20000;           // mmio case group page
const VPF = BASE + 0x22000;           // pf case group page (the #PF handler toggles its frame)
const VPI = BASE + 0x24000;           // irq case group page (the IRQ handler toggles its frame)
const VPN = BASE + 0x28000;           // inline case group page (identity)
const QP = BASE + 0x26000;            // the page the pf cases fault on; QP - 0x1000 stays mapped
const FENV = QP + 0x100;              // fnstenv target
const STACK_TOP = BASE + 0x33000;
const IMG_SIZE = 0x34000;
const FRAMES = [[VPG, 0x700000, 0x701000], [VPF, 0x702000, 0x703000], [VPI, 0x704000, 0x705000]];
const FR = va => FRAMES.find(f => f[0] === va);
const PD = 0x300000, PT0 = 0x301000, PT_MMIO = 0x303000, PT_APIC = 0x304000;
const pte = va => PT0 + (va >>> 12) * 4;   // PT0 and PT1 are contiguous: valid below 8 MiB
const MMIO = 0xD0000000, APIC = 0xFEE00000;
const OUTER = INLINE ? 6000 : 16000;
const REPS = { mmio: 8, irq: 4 };
const DONE_PORT = 0x9999;
const MEM_SIZE = 32 * 1024 * 1024;
const TIMEOUT_MS = INLINE ? 60_000 : 240_000;

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

// ── a tiny x86-32 assembler (the subset of tests/jit-guard-group-diff.mjs this workload needs)
const EAX = 0, ECX = 1, EDX = 2, EBX = 3, ESP = 4, EBP = 5, ESI = 6, EDI = 7;
const M = (base, disp = 0) => ({ base, disp });
const ABS = disp => ({ base: null, disp });
function modrm(reg, m) {
    if (m.base === null) return [0x05 | reg << 3, ...d32(m.disp)];
    if (m.base === ESP) return [0x84 | reg << 3, 0x24, ...d32(m.disp)];
    return [0x80 | reg << 3 | m.base, ...d32(m.disp)];
}
const op = (opc, reg, m, tail = []) => [...opc, ...modrm(reg, m), ...tail];

class Asm {
    constructor(img) { this.img = img; this.o = 0; this.labels = {}; this.fix = []; }
    at(off) { this.o = off; return this; }
    label(n) { this.labels[n] = BASE + this.o; return this; }
    b(...bytes) { for (const x of bytes.flat(3)) this.img[this.o++] = x & 0xFF; return this; }
    rel32(n) { this.fix.push({ at: this.o, to: n }); this.o += 4; return this; }
    abs32(n) { this.fix.push({ at: this.o, to: n, absolute: true }); this.o += 4; return this; }
    movRI(r, i) { return this.b(0xB8 + r, d32(i)); }
    movRM(r, m) { return this.b(op([0x8B], r, m)); }
    movMR(m, r) { return this.b(op([0x89], r, m)); }
    movMI(m, i) { return this.b(op([0xC7], 0, m, d32(i))); }
    addRM(r, m) { return this.b(op([0x03], r, m)); }
    addMR(m, r) { return this.b(op([0x01], r, m)); }
    addRR(d, s) { return this.b(0x01, 0xC0 | s << 3 | d); }
    xorRM(r, m) { return this.b(op([0x33], r, m)); }
    xorMR(m, r) { return this.b(op([0x31], r, m)); }
    xorMI(m, i) { return this.b(op([0x81], 6, m, d32(i))); }
    andMI(m, i) { return this.b(op([0x81], 4, m, d32(i))); }
    orMI(m, i) { return this.b(op([0x81], 1, m, d32(i))); }
    addRI(r, i) { return this.b(0x81, 0xC0 | r, d32(i)); }
    addRI8(r, i) { return this.b(0x83, 0xC0 | r, i & 0xFF); }
    andRI(r, i) { return this.b(0x81, 0xE0 | r, d32(i)); }
    cmpRI(r, i) { return this.b(0x81, 0xF8 | r, d32(i)); }
    incM(m) { return this.b(op([0xFF], 0, m)); }
    decM(m) { return this.b(op([0xFF], 1, m)); }
    decR(r) { return this.b(0x48 + r); }
    pushR(r) { return this.b(0x50 + r); }
    popR(r) { return this.b(0x58 + r); }
    call(n) { return this.b(0xE8).rel32(n); }
    jcc(cc, n) { return this.b(0x0F, 0x80 + cc).rel32(n); }
    ret() { return this.b(0xC3); }
    iret() { return this.b(0xCF); }
    pushfd() { return this.b(0x9C); }
    popfd() { return this.b(0x9D); }
    cli() { return this.b(0xFA); }
    setIF() { return this.pushfd().orMI(M(ESP, 0), 0x200).popfd(); }  // IF 0->1 via POPF
    fnstenv(m) { return this.b(op([0xD9], 6, m)); }
    invlpg(m) { return this.b(op([0x0F, 0x01], 7, m)); }
    outB(port, v) { return this.b(0xB0, v, 0x66, 0xBA, d16(port), 0xEE); }  // mov al, v; mov dx, port; out dx, al
    resolve() {
        const dv = new DataView(this.img.buffer);
        for (const f of this.fix) {
            const t = this.labels[f.to]; if (t === undefined) throw new Error("label " + f.to);
            dv.setUint32(f.at, f.absolute ? t : t - (BASE + f.at + 4), true);
        }
    }
}
const JNZ = 0x5;

function buildImage(seed) {
    const rnd = mulberry32(seed);
    const R = n => Math.floor(rnd() * n);
    const rnd32 = () => (rnd() * 2 ** 32) >>> 0;
    const img = new Uint8Array(IMG_SIZE);
    for (let i = 0; i < IMG_SIZE; i++) img[i] = R(256);
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

    [0, 0, 0xFFFF, 0x00CF9A00, 0xFFFF, 0x00CF9200].forEach((v, i) => w32(GDT + 4 * i, v));
    dv.setUint16(GDTR - BASE, 23, true); w32(GDTR + 2, GDT);
    dv.setUint16(IDTR - BASE, 256 * 8 - 1, true); w32(IDTR + 2, IDT);
    for (let i = 0; i < 256 * 8; i++) img[IDT - BASE + i] = 0;
    for (let i = 0; i < 0x40; i++) img[VARS - BASE + i] = 0;

    const a = new Asm(img);
    const [, F1, F2] = FR(VPF), [, I1, I2] = FR(VPI);

    // ── #PF: make the faulting page present, REMAP the pf group page, count, retry
    a.at(0x6000).label("pf");
    a.pushR(EAX);
    a.b(0x0F, 0x20, 0xD0);                                          // mov eax, cr2
    a.b(0xC1, 0xE8, 12).b(0xC1, 0xE0, 2).addRI(EAX, PT0);           // eax = &pte(cr2)
    a.orMI(M(EAX, 0), 1);
    a.b(0x0F, 0x20, 0xD0).invlpg(M(EAX, 0));
    a.xorMI(ABS(pte(VPF)), (F1 ^ F2) >>> 0).invlpg(ABS(VPF));
    a.incM(ABS(PFC));
    a.popR(EAX).addRI8(ESP, 4).iret();                              // drop the error code
    // ── vector 0x41: REMAP the irq group page, EOI
    a.at(0x6100).label("irq41");
    a.xorMI(ABS(pte(VPI)), (I1 ^ I2) >>> 0).invlpg(ABS(VPI));
    a.incM(ABS(IRQC)).movMI(ABS(APIC + 0xB0), 0).iret();
    // ── vector 0x42 (inline characterization): count, EOI
    a.at(0x6200).label("irq42");
    a.incM(ABS(IRQC2)).movMI(ABS(APIC + 0xB0), 0).iret();

    a.at(0x1000);
    // mmio: anchor, MMIO store (handler writes the group frame), members, MMIO load, members
    a.label("r_mmio");
    a.movRI(ESI, VPG).movRI(EDI, MMIO);
    a.movRM(EAX, M(ESI, 0)).addRM(EAX, M(ESI, 4));
    a.movMR(M(EDI, 0x10), EAX);                                     // safe_write32_slow_jit -> JS
    a.addRM(EAX, M(ESI, 0x40)).xorRM(EAX, M(ESI, 8));
    a.movRM(ECX, M(EDI, 0x20));                                     // safe_read32s_slow_jit -> JS
    a.addRR(EAX, ECX).addRM(EAX, M(ESI, 12)).movMR(M(ESI, 0x80), EAX);
    a.addMR(ABS(CHK), EAX).ret();

    // pf: anchor, (maybe) a #PF from the slow path, members that must see the handler's remap
    const pfCase = (name, k, fault) => {
        a.label(name);
        a.movRM(EAX, ABS(CTR)).andRI(EAX, 15).cmpRI(EAX, k).jcc(JNZ, name + "_go");
        a.andMI(ABS(pte(QP)), ~1).invlpg(ABS(QP));
        a.label(name + "_go");
        a.movRI(ESI, VPF).movRI(EDI, QP);
        a.movRM(EAX, M(ESI, 0)).addRM(EAX, M(ESI, 4));
        fault();
        a.addRM(EAX, M(ESI, 8)).xorRM(EAX, M(ESI, 12)).movMR(M(ESI, 0x10 + 4 * k), EAX);
        a.addMR(ABS(CHK), EAX).ret();
    };
    pfCase("r_pf_read", 1, () => a.addRM(EAX, M(EDI, 8)));
    pfCase("r_pf_write", 3, () => a.movMR(M(EDI, 0x20), EAX));
    pfCase("r_pf_rmw", 5, () => a.addMR(M(EDI, 0x24), EAX));
    pfCase("r_pf_cross", 7, () => a.addRM(EAX, M(EDI, -2)));        // crosses from QP-0x1000
    pfCase("r_pf_fenv", 9, () => a.fnstenv(M(EDI, FENV - QP)));      // writable_or_pagefault_jit

    // irq: LAPIC writes inside a group with IF=0, then POPF takes the IRQ (handler remaps)
    a.label("r_irq");
    a.movRI(ESI, VPI).movRI(EDI, APIC);
    a.movRM(EAX, M(ESI, 0));
    a.movMI(M(EDI, 0x80), 0);                                       // TPR: handle_irqs, IF=0
    a.addRM(EAX, M(ESI, 4));
    a.movMI(M(EDI, 0x300), 0x40041);                                // ICR: self IPI 0x41, pending
    a.addRM(EAX, M(ESI, 8)).movMR(M(ESI, 0x80), EAX);
    a.setIF().cli();                                                // taken at POPF's exit
    a.addRM(EAX, M(ESI, 12)).xorRM(EAX, M(ESI, 16)).movMR(M(ESI, 0x84), EAX);
    a.addMR(ABS(CHK), EAX).ret();

    // inline: a self-IPI written with IF=1 inside a group (delivered inside the slow path)
    a.label("r_inline");
    a.movRI(ESI, VPN).movRI(EDI, APIC).setIF().movRI(ECX, 8);
    a.label("ri_loop");
    a.movRM(EAX, M(ESI, 0));
    a.movMI(M(EDI, 0x300), 0x40042);
    a.addRM(EAX, M(ESI, 4)).addMR(ABS(CHK), EAX);
    a.decR(ECX).jcc(JNZ, "ri_loop");
    a.cli().ret();

    // ── driver
    a.at(ENTRY_OFF).label("entry");
    a.movRI(ESP, STACK_TOP);
    a.b(0x0F, 0x01, 0x15, d32(GDTR));                               // lgdt
    a.b(0xEA).abs32("reload").b(d16(0x08));
    a.label("reload");
    a.b(0x66, 0xB8, d16(0x10)).b(0x8E, 0xD8).b(0x8E, 0xC0).b(0x8E, 0xD0).b(0x8E, 0xE0).b(0x8E, 0xE8);
    a.b(0x0F, 0x01, 0x1D, d32(IDTR));                               // lidt
    a.outB(0x21, 0xFF).outB(0xA1, 0xFF);                            // mask the PICs
    a.movRI(EAX, PD).b(0x0F, 0x22, 0xD8);                           // cr3 (tables written by the host)
    a.b(0x0F, 0x20, 0xC0).b(0x0D, d32(0x80000000)).b(0x0F, 0x22, 0xC0);
    a.b(0xDB, 0xE3);                                                // fninit
    for (const r of [EAX, ECX, EDX, EBX, EBP]) a.movRI(r, rnd32());
    a.movMI(ABS(CTR), OUTER);
    a.label("outer");
    // repeated so that each routine's page passes the JIT hotness threshold early in the run
    if (CASES.has("mmio")) for (let i = 0; i < REPS.mmio; i++) a.call("r_mmio");
    if (CASES.has("pf")) for (const n of ["read", "write", "rmw", "cross", "fenv"]) a.call("r_pf_" + n);
    if (CASES.has("irq")) for (let i = 0; i < REPS.irq; i++) a.call("r_irq");
    if (CASES.has("inline")) a.call("r_inline");
    for (const r of [EDX, EBX, EBP]) a.xorMR(ABS(CHK), r);
    a.b(op([0xD1], 0, ABS(CHK)));                                   // rol dword [CHK], 1
    a.decM(ABS(CTR)).jcc(JNZ, "outer");
    a.b(0x66, 0xBA, d16(DONE_PORT), 0xEE).b(0xF4).b(0xEB, 0xFE);
    a.resolve();

    const gate = (vec, target) => {
        const o = IDT - BASE + vec * 8;
        dv.setUint16(o, target & 0xFFFF, true); dv.setUint16(o + 2, 0x08, true);
        img[o + 4] = 0; img[o + 5] = 0x8E; dv.setUint16(o + 6, target >>> 16, true);
    };
    gate(14, a.labels.pf); gate(0x41, a.labels.irq41); gate(0x42, a.labels.irq42);
    return { img, frameSeed: seed * 7919 + 1 };
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
                                   log_level: 0, wasm_path: cfg.engine ?? ENGINE, acpi: true });
        let done = false, timer, cpu, w;
        const mm = { w: 0, r: 0, remaps: 0 };
        const finish = status => {
            if (done) return; done = true;
            currentFinish = null;
            clearTimeout(timer);
            try { emulator.stop(); } catch { }
            const regs = Array.from({ length: 8 }, (_, i) => cpu.reg32[i] >>> 0);
            const rd = (a, n) => emulator.read_memory(a, n);
            const u32 = a => { const b = rd(a, 4); return (b[0] | b[1] << 8 | b[2] << 16 | b[3] << 24) >>> 0; };
            let h = fnv(rd(0x700000, 0x6000));
            h = fnv(rd(QP - 0x1000, 0x2000), h);
            h = fnv(rd(VPN, 0x1000), h);
            if (process.env.GG_TRACE && cfg.groups && w.gg_trace_len) {
                const n = w.gg_trace_len(), p = w.gg_trace_ptr();
                console.log(Buffer.from(new Uint8Array(cpu.wasm_memory.buffer, p, n)).toString());
            }
            const stat = i => w?.get_jit_guard_group_stat ? w.get_jit_guard_group_stat(i) : -1;
            const rt = i => w?.get_jit_guard_group_rt ? w.get_jit_guard_group_rt(i) : -1;
            resolve({ status, regs, chk: u32(CHK), pf: u32(PFC), irq: u32(IRQC), irq2: u32(IRQC2), mem: h,
                      mmio: `${mm.w}/${mm.r}`, remaps: mm.remaps,
                      members: stat(4), fast: rt(0) + rt(1) + rt(2), slow: rt(3) + rt(4) + rt(5) });
        };
        emulator.add_listener("emulator-loaded", () => {
            cpu = emulator.v86.cpu; w = cpu.wm.exports;
            cpu.reboot_internal(); cpu.reset_memory();
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
                    w.jit_guard_group_stats_reset();
                    if (process.env.GG_TRACE && cfg.groups) w.set_jit_guard_groups_trace(1);
                }
                cpu.jit_clear_cache?.();
            }
            cpu.load_multiboot(build.img.buffer);
            // paging: identity 0..8 MiB with the three group pages on their first frame, one
            // MMIO page and the LAPIC page; frame pairs hold different bytes
            const wr32 = (pa, v) => cpu.mem8.set(d32(v), pa);
            wr32(PD, PT0 | 3); wr32(PD + 4, (PT0 + 0x1000) | 3);
            wr32(PD + (MMIO >>> 22) * 4, PT_MMIO | 3); wr32(PD + (APIC >>> 22) * 4, PT_APIC | 3);
            for (let i = 0; i < 2048; i++) wr32(PT0 + 4 * i, (i << 12) | 3);
            wr32(PT_MMIO + ((MMIO >>> 12) & 0x3FF) * 4, MMIO | 3);
            wr32(PT_APIC + ((APIC >>> 12) & 0x3FF) * 4, APIC | 3);
            const frnd = mulberry32(build.frameSeed);
            for (const [va, f1, f2] of FRAMES) {
                wr32(pte(va), f1 | 3);
                const bytes = Uint8Array.from({ length: 0x1000 }, () => Math.floor(frnd() * 256));
                emulator.write_memory(bytes, f1);
                emulator.write_memory(bytes.map(x => x ^ 0xA5), f2);
            }
            const [, G1, G2] = FR(VPG);
            const rd32 = pa => { const b = emulator.read_memory(pa, 4); return (b[0] | b[1] << 8 | b[2] << 16 | b[3] << 24) >>> 0; };
            cpu.io.mmap_register(MMIO, 0x20000,
                () => 0,
                () => { mm.w++; },
                addr => { mm.r++; return (Math.imul(mm.r, 0x85EBCA6B) ^ addr) | 0; },
                (addr, value) => {
                    mm.w++;
                    if (NC_REMAP && mm.w % 5 === 0) {
                        // NEGATIVE CONTROL: a transparent helper that changes a translation
                        wr32(pte(VPG), (rd32(pte(VPG)) ^ (G1 ^ G2)) >>> 0);
                        w.jit_aot_flush_tlb();
                        mm.remaps++;
                    }
                    const frame = (rd32(pte(VPG)) & 0xFFFFF000) >>> 0;
                    wr32(frame + 0x40, (value ^ Math.imul(mm.w, 0x9E3779B1)) >>> 0);
                });
            cpu.io.register_write(DONE_PORT, cpu, () => finish("done"));
            timer = setTimeout(() => finish("TIMEOUT"), TIMEOUT_MS);
            currentFinish = finish;
            emulator.run();
        });
    });
}

let failures = 0;
const fail = m => { failures++; console.error("FAIL " + m); };
const sig = r => `${r.status} chk=${r.chk?.toString(16)} pf=${r.pf} irq=${r.irq}/${r.irq2} mmio=${r.mmio}` +
                 ` mem=${r.mem?.toString(16)} regs=${r.regs?.map(x => x.toString(16)).join(",")}`;
console.log(`engine ${ENGINE}${NC_REMAP ? "  NEGATIVE CONTROL mmio-remap" : ""}  cases ${[...CASES].join(",")}`);

for (let seed = SEED_START; seed < SEED_START + SEED_COUNT; seed++) {
    const build = buildImage(seed);
    const interp = await run(build, { jit: false });
    if (interp.status !== "done") { fail(`seed ${seed}: interpreter did not finish (${interp.status})`); continue; }
    const off = await run(build, { jit: true, groups: false });
    const on = await run(build, { jit: true, groups: true });
    if (on.status === "NO-SWITCH") { console.log(`seed ${seed}: engine has no set_jit_guard_groups`); continue; }
    if (INLINE) {
        const offOk = sig(off) === sig(interp), onOk = sig(on) === sig(interp);
        console.log(`seed ${seed}: inline IRQ delivery — interpreter irq2=${interp.irq2};` +
                    ` groups OFF ${offOk ? "matches" : "DIVERGES"}, groups ON ${onOk ? "matches" : "DIVERGES"}`);
        if (!offOk) console.log(`  interp ${sig(interp)}\n  off    ${sig(off)}\n  on     ${sig(on)}`);
        if (interp.irq2 !== OUTER * 8) fail(`seed ${seed}: interpreter took ${interp.irq2} of ${OUTER * 8} IPIs`);
        if (offOk && !onOk) fail(`seed ${seed}: groups ON diverges where groups OFF does not`);
        continue;
    }
    if (NC_REMAP) {
        const cnt = await run(build, { jit: true, groups: true, debug: 4 });
        const note = ` (host remaps ${interp.remaps}, members ${on.members}, memberFast ${cnt.fast} memberSlow ${cnt.slow})`;
        if (process.env.GG_VERBOSE) console.log(`  interp ${sig(interp)}\n  off    ${sig(off)}\n  on     ${sig(on)}\n  count  ${sig(cnt)}`);
        if (!(interp.remaps > 0)) fail(`seed ${seed}: the handler never remapped${note}`);
        if (sig(off) !== sig(interp)) fail(`seed ${seed}: groups OFF diverges too — the control is not about groups${note}`);
        if (sig(on) !== sig(interp)) console.log(`seed ${seed}: negative control diverges, as it must${note}`);
        else fail(`seed ${seed}: negative control did NOT diverge${note}`);
        continue;
    }
    const cnt = await run(build, { jit: true, groups: true, debug: 4 });
    const note = ` (pf=${interp.pf} irq=${interp.irq} mmio=${interp.mmio} members=${on.members}` +
                 ` memberFast=${cnt.fast} memberSlow=${cnt.slow})`;
    const ok = [off, on, cnt].every(r => sig(r) === sig(interp));
    if (ok) console.log(`seed ${seed}: ok${note}`);
    else {
        fail(`seed ${seed}: DIVERGENCE${note}`);
        console.error(`  interp ${sig(interp)}\n  off    ${sig(off)}\n  on     ${sig(on)}\n  count  ${sig(cnt)}`);
    }
    if (off.members > 0) fail(`seed ${seed}: groups OFF compiled ${off.members} members`);
    if (!(on.members > 0)) fail(`seed ${seed}: groups ON compiled no member`);
    if (!(cnt.fast > 0)) fail(`seed ${seed}: no member ran on the fast path`);
    if (CASES.has("pf") && !(interp.pf > 0)) fail(`seed ${seed}: no #PF was taken`);
    if (CASES.has("irq") && interp.irq !== OUTER * REPS.irq) fail(`seed ${seed}: ${interp.irq} of ${OUTER * REPS.irq} IRQs taken`);
}
if (failures) { console.error(`${failures} failure(s)`); process.exit(1); }
console.log(INLINE ? "inline characterization done" : NC_REMAP ? "negative control: divergence detected in every seed" : "all seeds consistent");
process.exit(0);
