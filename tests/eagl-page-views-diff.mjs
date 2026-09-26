#!/usr/bin/env node
// EAGL page views (src/rust/cpu/hypercall_eagl.rs, eagl_page_views_set): differential of the
// inner-loop handlers 128..=132 with the views OFF vs ON, under x86 paging, over synthetic
// descriptor trees / token lists / shader records placed at random — straddling page
// boundaries, on aliased pages (linear page -> a different physical page), on read-only and
// not-present pages — so every read, narrow read and ring write takes both the view path and
// the exact fallback.
//
// Per case both arms start from the same byte-identical state and are compared on: handled
// flag, all eight GPRs, EIP, EFLAGS, CR2 and a hash of every physical region a handler or a
// #PF delivery can touch (page tables with their A/D bits, stack, arena, alias backing, image).
// Variants: cold TLB, warm TLB (a priming dispatch), a remap between two dispatches (PTE
// rewritten + TLB flush — a view that outlived its invocation reads the old page), and a
// nearly-full software TLB so the handler's own page walk flushes it mid-invocation.
// A third arm runs ON with the view oracle (every view read re-read through the exact
// accessor, every view write's target re-translated): it must also agree, report 0
// mismatches, and report view hits (a differential that never took the view path proves
// nothing).
//
//   node tests/eagl-page-views-diff.mjs [cases] [--seed N] [--expect-diverge]
//   V86_WASM_PATH=<engine.wasm> selects the build. --expect-diverge inverts the verdict: the
//   run passes only if a divergence is found (negative-control mutants, V86_EAGL_PV_MUTANT).

import { V86 } from "../build/libv86.mjs";
import { createHash } from "node:crypto";

const argv = process.argv.slice(2);
const CASES = Number(argv.find(a => /^\d+$/.test(a)) ?? 800);
const seedIdx = argv.indexOf("--seed");
const SEED = seedIdx >= 0 ? Number(argv[seedIdx + 1]) : 0x5eed;
const EXPECT_DIVERGE = argv.includes("--expect-diverge");
const WASM = process.env.V86_WASM_PATH || undefined;

const MEM = 64 << 20;
const BASE = 0x100000;
const PD = 0x108000, PT0 = 0x110000;         // 16 page tables: identity 0..64 MB
const STACK_TOP = 0x300000;
const DUMMY = 0x2e0000;                       // always-valid scratch (cfg-cache reset), not hashed
const ARENA = 0x1000000, ARENA_PAGES = 256;   // linear window whose PTEs are randomized
const BACK = 0x2000000;                        // alias backing (physical)
const TOUCH = 0x1200000;                       // pages touched to fill the software TLB
const TLB_MAX = 10000;
const HANDLER_PF = BASE + 0x800;
const TOUCH_ENTRY = BASE + 0x600;
const REGIONS = [[BASE, 0x3000], [PD, 0x18000], [0x2f0000, 0x10000], [ARENA, ARENA_PAGES << 12], [BACK, ARENA_PAGES << 12]];

// ---- deterministic PRNG -------------------------------------------------------------------
let rs = SEED >>> 0;
const rnd = () => { rs ^= rs << 13; rs >>>= 0; rs ^= rs >>> 17; rs ^= rs << 5; rs >>>= 0; return rs; };
const ri = n => rnd() % n;
const chance = p => rnd() / 4294967296 < p;
const pick = xs => xs[ri(xs.length)];

// ---- guest image: GDT, IDT (every vector -> hlt), paging on, hlt; a TLB-touch routine ------
function image() {
    const b = new Uint8Array(0x2000), d = new DataView(b.buffer);
    [0x1badb002, 0x10000, (-0x1badb002 - 0x10000) >>> 0, BASE, BASE, BASE + 0x2000, BASE + 0x2000, BASE + 0x40]
        .forEach((v, i) => d.setUint32(i * 4, v, true));
    // null, code 0x08, data 0x10, 32-bit TSS 0x18 (so a #PF raised at CPL 3 has an esp0)
    b.set([0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 0, 0, 0, 0x9a, 0xcf, 0, 255, 255, 0, 0, 0, 0x92, 0xcf, 0,
        0x67, 0, 0x00, 0x0d, 0x10, 0x89, 0x00, 0x00], 0xc00);
    d.setUint16(0xc20, 31, true); d.setUint32(0xc22, BASE + 0xc00, true);
    d.setUint32(0xd04, STACK_TOP - 0x80, true); d.setUint32(0xd08, 0x10, true);
    for (let v = 0; v < 256; v++) {
        const g = 0x1000 + v * 8;
        d.setUint16(g, HANDLER_PF & 0xffff, true); d.setUint16(g + 2, 8, true);
        b[g + 5] = 0x8e; d.setUint16(g + 6, HANDLER_PF >>> 16, true);
    }
    d.setUint16(0xcb8, 0x7ff, true); d.setUint32(0xcba, BASE + 0x1000, true);
    let p = 0x40;
    const e = (...x) => { b.set(x, p); p += x.length; }, u = x => { d.setUint32(p, x >>> 0, true); p += 4; };
    e(0x0f, 0x01, 0x15); u(BASE + 0xc20); e(0xea); u(BASE + p + 6); e(8, 0);
    e(0x66, 0xb8, 0x10, 0, 0x8e, 0xd8, 0x8e, 0xc0, 0x8e, 0xd0, 0x8e, 0xe0, 0x8e, 0xe8); e(0xbc); u(STACK_TOP);
    e(0x0f, 0x01, 0x1d); u(BASE + 0xcb8);
    e(0x66, 0xb8, 0x18, 0x00, 0x0f, 0x00, 0xd8);      // mov ax,0x18 ; ltr ax
    e(0xb8); u(PD); e(0x0f, 0x22, 0xd8); e(0x0f, 0x20, 0xc0); e(0x0d); u(0x80000000); e(0x0f, 0x22, 0xc0);
    e(0xf4);
    p = TOUCH_ENTRY - BASE;                          // ecx = page count
    e(0xbb); u(TOUCH);
    const loop = p; e(0x8b, 0x03); e(0x81, 0xc3); u(0x1000); e(0x49); e(0x75, (loop - (p + 2)) & 0xff);
    e(0xf4);
    p = HANDLER_PF - BASE; e(0xf4, 0xeb, 0xfd);
    return b;
}

// ---- emulator ----------------------------------------------------------------------------
const em = new V86({ autostart: false, memory_size: MEM, disable_jit: 0, log_level: 0, wasm_path: WASM });
await new Promise(r => em.add_listener("emulator-loaded", r));
const c = em.v86.cpu, ex = c.wm.exports;
for (const f of ["eagl_page_views_set", "eagl_test_dispatch", "eagl_page_views_stat", "eagl_page_views_mutant"])
    if (typeof ex[f] !== "function") throw new Error(`engine lacks ${f} — not a page-views build`);
let onHalt = null;
em.bus.register("cpu-event-halt", () => { const f = onHalt; onHalt = null; f?.(); });
function runGuest() {
    return new Promise((res, rej) => {
        const t = setTimeout(() => { em.stop(); rej(new Error("guest did not halt")); }, 60000);
        onHalt = () => { clearTimeout(t); em.stop(); setTimeout(res, 20); };
        em.run();
    });
}
c.reboot_internal(); c.reset_memory(); c.load_multiboot(image().buffer);
for (let i = 0; i < 16; i++) c.write32(PD + i * 4, (PT0 + i * 0x1000) | 7);
for (let i = 16; i < 1024; i++) c.write32(PD + i * 4, 0);
for (let i = 0; i < 16 * 1024; i++) c.write32(PT0 + i * 4, (i << 12) | 7);
await runGuest();
if (!(c.cr[0] & 0x80000000)) throw new Error("paging not enabled");

if (ex.eagl_page_views_selftest) {
    const st = ex.eagl_page_views_selftest() >>> 0;
    if (st !== 0 && ex.eagl_page_views_mutant() === 0) throw new Error(`eagl_page_views_selftest failed: mask 0x${st.toString(16)}`);
}
const mem = () => c.mem8;
const cpuBlock = () => new Uint8Array(c.reg32.buffer, 0, 2048);
function snapshot() { return { cpu: cpuBlock().slice(), regions: REGIONS.map(([a, n]) => mem().slice(a, a + n)) }; }
function restore(s) {
    cpuBlock().set(s.cpu);
    REGIONS.forEach(([a], i) => mem().set(s.regions[i], a));
}
function stateHash() {
    const h = createHash("sha1");
    for (const [a, n] of REGIONS) h.update(mem().subarray(a, a + n));
    return h.digest("hex");
}
const base = snapshot();

// ---- per-case linear memory model ---------------------------------------------------------
let pte;            // per arena page: {mode, phys}
const pteAddr = lin => PT0 + (lin >>> 12) * 4;
function setPages() {
    const pAlias = pick([0, 0.1, 0.25]), pHole = pick([0, 0, 0.02, 0.06]), pRO = pick([0, 0.03]), pSup = pick([0, 0, 0.05]);
    pte = [];
    for (let i = 0; i < ARENA_PAGES; i++) {
        const lin = ARENA + (i << 12);
        let m = { mode: "id", phys: lin, flags: 7 };
        const r = rnd() / 4294967296;
        if (r < pHole) m = { mode: "hole", phys: 0, flags: 0 };
        else if (r < pHole + pAlias) m = { mode: "alias", phys: BACK + (ri(ARENA_PAGES) << 12), flags: 7 };
        else if (r < pHole + pAlias + pRO) m = { mode: "ro", phys: lin, flags: 5 };
        else if (r < pHole + pAlias + pRO + pSup) m = { mode: "sup", phys: lin, flags: 3 };
        pte.push(m);
        c.write32(pteAddr(lin), m.mode === "hole" ? 0 : (m.phys | m.flags));
    }
}
function phys(lin) {
    if (lin >= ARENA && lin < ARENA + (ARENA_PAGES << 12)) {
        const m = pte[(lin - ARENA) >>> 12];
        return m.mode === "hole" ? -1 : m.phys + (lin & 0xfff);
    }
    return lin;
}
function w8(lin, v) { const p = phys(lin >>> 0); if (p >= 0) mem()[p] = v & 0xff; }
function w16(lin, v) { w8(lin, v); w8(lin + 1, v >>> 8); }
function w32(lin, v) { w16(lin, v); w16(lin + 2, v >>> 16); }
function r32(lin) { let v = 0; for (let i = 3; i >= 0; i--) { const p = phys(lin + i); v = (v << 8) | (p >= 0 ? mem()[p] : 0); } return v >>> 0; }
/** Random bytes straight into a physical range (identity pages AND the alias backing). */
function randomPhys(a, n) {
    const m = mem();
    const u = (m.byteOffset + a) % 4 === 0 ? new Uint32Array(m.buffer, m.byteOffset + a, n >>> 2) : null;
    if (u) for (let i = 0; i < u.length; i++) u[i] = rnd();
    else for (let i = 0; i < n; i++) m[a + i] = rnd() & 0xff;
}

let used;
/** A struct of `size` bytes somewhere in the arena; ~30% straddle a page boundary. */
function alloc(size, align = 4) {
    for (let tries = 0; tries < 40; tries++) {
        const page = ri(ARENA_PAGES - 2);
        let off;
        if (chance(0.3) && size > 1) off = 0x1000 - 1 - ri(Math.min(size - 1, 0xfff));
        else off = ri(Math.max(1, 0x1000 - Math.min(size, 0xfff)));
        if (align > 1 && !chance(0.1)) off &= ~(align - 1);
        const a = ARENA + (page << 12) + off;
        if (a + size > ARENA + ((ARENA_PAGES - 1) << 12)) continue;
        if (used.some(([s, e]) => a < e && s < a + size)) continue;
        used.push([a, a + size]);
        return a;
    }
    return ARENA + (ri(ARENA_PAGES - 4) << 12);
}
const f32 = () => pick([0, 0x3f800000, 0xbf800000, 0x7fc00000, 0x7f800000, 0x4f000000, 0xdf000000,
    0x42f60000, 0xc2f60000, 0x3eaaaaab, rnd(), rnd()]) >>> 0;
function fill(lin, n, gen = f32) { for (let i = 0; i < n; i += 4) w32(lin + i, gen()); }

// ---- builders: each returns the handler id and sets the CPU registers ---------------------
function stackArgs(vals) {
    const esp = STACK_TOP - 0x400 + (ri(64) << 2);
    c.reg32[4] = esp;
    vals.forEach((v, i) => c.write32(esp + 4 + i * 4, v >>> 0));
}
function build128() {
    const desc = alloc(0x1c);
    const rows = chance(0.05) ? 17 : 1 + ri(5), cols = 1 + ri(5), count = ri(20);
    w32(desc, pick([1, 2, 3, 1, 2, 3, 0, 4])); w32(desc + 0x14, rows); w32(desc + 0x18, cols);
    const src = alloc(Math.max(1, count) * 0x40); fill(src, Math.max(1, count) * 0x40);
    const dst = alloc(count * rows * cols * 4 + 4);
    stackArgs([desc, dst, src, count]);
    return 128;
}
function buildApply() {
    const cells = [alloc(4), alloc(4), alloc(4), alloc(4)];
    // A sequential descriptor array: one top-level descriptor, a container (cls 5) or a leaf.
    const leaves = [];
    const top = chance(0.4) ? 5 : pick([0, 1, 2, 3, 3, 4]);
    const kids = top === 5 ? 1 + ri(3) : 0;
    const bytes = top === 5 ? 0x18 + kids * 0x1c : 0x1c;
    const d = alloc(bytes);
    const leaf = at => {
        w32(at, pick([1, 2, 3, 1, 2, 3, 0])); w32(at + 4, pick([0, 1, 2, 3, 3, 9]));
        w32(at + 0x10, ri(4)); w32(at + 0x14, 1 + ri(6)); w32(at + 0x18, 1 + ri(6));
        leaves.push(at);
    };
    if (top === 5) {
        w32(d + 4, 5); w32(d + 0x10, ri(3)); w32(d + 0x14, kids);
        for (let k = 0; k < kids; k++) leaf(d + 0x18 + k * 0x1c);
    } else { leaf(d); w32(d + 4, top); }
    const src = alloc(0x900); fill(src, 0x900, () => (chance(0.5) ? f32() : rnd()) >>> 0);
    const dst = alloc(0x900);
    w32(cells[0], d); w32(cells[1], src); w32(cells[2], dst); w32(cells[3], ri(48));
    stackArgs(cells);
    return 129 + ri(3);
}
const FID = { srs: 0x101, samp: 0x102, tss: 0x103, fvf: 0x104, svs: 0x105, sps: 0x106, vscf: 0x107, pscf: 0x108, tex: 0x109 };
const VT = { srs: 0xe4, tss: 0x10c, samp: 0x114, tex: 0x104, fvf: 0x164, svs: 0x170, vscf: 0x178, sps: 0x1ac, pscf: 0x1b4 };
const TOKENS = () => [0x01000000 | ri(300), 0x01000000 | ri(40), 0x02000000 | ri(30), 0x08000000 | ri(20),
    0x06000008, 0x06000002, 0x06000102, 0x06000005, 0x06000405, 0x06000000, 0x06000001, 0x06000001,
    0x05000000, 0x03000000, 0x06000003, 0x0b000000, 0x06000009];
function build132() {
    const ntok = 16, np = 6, nrec = 4;
    const cfg = alloc(0x58);
    const table = alloc(ntok * 0x1c);
    const toks = []; for (let t = 0; t < ntok; t++) { toks.push(pick(TOKENS())); w32(table + t * 0x1c, toks[t]); }
    const cap = pick([256, 512, 1024, 2048]);
    const ringCtrl = alloc(8), ring = alloc(cap);
    w32(ringCtrl, chance(0.1) ? cap - ri(64) * 4 : ri(16) * 4);
    const ownerG = alloc(4);
    const srsShadow = chance(0.8) ? alloc(256 * 4) : 0, sampShadow = chance(0.8) ? alloc(256 * 4) : 0;
    const srsSkip = chance(0.8) ? alloc(4) : 0, sampSkip = chance(0.8) ? alloc(4) : 0;
    if (srsShadow) fill(srsShadow, 1024, () => ri(4)); if (sampShadow) fill(sampShadow, 1024, () => ri(4));
    // device -> vtable -> stubs "B8 <funcId>"
    const dev = alloc(4), vt = alloc(0x1d0);
    w32(dev, vt); w32(ownerG, chance(0.85) ? dev : dev + 4);
    for (const k of Object.keys(VT)) {
        const stub = alloc(5, 1);
        w8(stub, chance(0.03) ? 0xe9 : 0xb8); w32(stub + 1, chance(0.03) ? FID[k] + 1 : FID[k]);
        w32(vt + VT[k], stub);
    }
    const fields = [[0, pick([2, 2, 2, 3, 1])], [4, table], [8, ringCtrl], [0xc, ring], [0x10, cap], [0x14, chance(0.9) ? ownerG : 0],
        [0x18, FID.srs], [0x1c, srsShadow], [0x20, srsSkip], [0x24, FID.samp], [0x28, sampShadow], [0x2c, sampSkip],
        [0x30, FID.tss], [0x34, 1], [0x38, 0x1000 + ri(1000)], [0x3c, FID.fvf], [0x40, FID.svs], [0x44, FID.sps],
        [0x48, FID.vscf], [0x4c, FID.pscf], [0x50, chance(0.9) ? FID.tex : 0]];
    for (const [o, v] of fields) w32(cfg + o, v);
    // ctx + handle tables + records
    const ctx = alloc(0x90), pageTbl = alloc(np * 4), recBase = alloc(nrec * 0x1c);
    w32(ctx + 8, dev); w32(ctx + 0x84, pick([0, 0, 0, 0, 1, 2])); w32(ctx + 0x8c, pageTbl); w32(ctx + 0x24, recBase);
    w32(ctx + 0x2c, pick([0, 4])); const X = alloc(0xc); w32(ctx + 0xc, X); w32(X + 8, pick([0, 4]));
    const node = (depth) => {
        const n = alloc(0xac);
        fill(n, 0xac, () => ri(8));
        const tok = ri(ntok);
        w32(n, chance(0.08) ? -1 : tok); w32(n + 4, ri(8)); w32(n + 0xc, ri(np)); w32(n + 0x14, pick([0, 4, 8]));
        const cnt = chance(0.03) ? 300 : ri(6); const src = alloc(Math.max(1, cnt) * 16); fill(src, Math.max(1, cnt) * 16);
        w32(n + 0x4c, src); w32(n + 0xa8, cnt);
        const cls = toks[tok] >>> 24, en = toks[tok] & 0xffffff;
        let value = ri(4);
        if (cls === 1 && srsShadow && en < 256 && chance(0.4)) value = r32(srsShadow + en * 4);
        w32(n + 0x68, value);
        if (chance(0.08)) { const alias = alloc(0x70); w32(alias, tok); w32(alias + 0x68, ri(4)); w32(n + 0x64, alias); }
        return n;
    };
    for (let i = 0; i < np; i++) {
        const r = alloc(0x44);
        const aBase = alloc(0x20); fill(aBase, 0x20, () => ri(nrec));
        let t = 0;
        if (chance(0.3)) { t = alloc(0xc); const tbl = alloc(16); fill(tbl, 16, () => ri(nrec)); w32(t + 8, tbl); fill(aBase, 0x20, () => ri(4)); }
        w32(r + 0x38, t); w32(r + 0x28, aBase);
        const m = chance(0.2) ? 0 : 1 + ri(4); const sub = alloc(Math.max(1, m) * 0xac);
        w32(r + 0x3c, m); w32(r + 0x40, sub);
        for (let k = 0; k < m; k++) {
            const sn = sub + k * 0xac; fill(sn, 0xac, () => ri(8));
            let tk = ri(ntok); for (let g = 0; g < 4 && (toks[tk] >>> 24) === 6 && chance(0.7); g++) tk = ri(ntok);
            w32(sn, tk); w32(sn + 4, ri(8)); w32(sn + 0xc, ri(np)); w32(sn + 0x14, pick([0, 4, 8]));
            w32(sn + 0x68, ri(4)); const cnt = ri(4); const src = alloc(Math.max(1, cnt) * 16); fill(src, Math.max(1, cnt) * 16);
            w32(sn + 0x4c, src); w32(sn + 0xa8, cnt);
        }
        w32(pageTbl + i * 4, r);
    }
    for (let i = 0; i < nrec; i++) {
        const rec = recBase + i * 0x1c;
        w32(rec + 0xc, chance(0.9) ? 1 : 0); w32(rec + 4, 0x7000 + i);
        const total = ri(6), E = 0x20;
        const ct = chance(0.15) ? 0 : alloc(E + 8 + total * 20);
        if (ct) {
            w32(ct + 0xc, total); w32(ct + 0x10, E);
            for (let k = 0; k < total; k++) {
                const ep = ct + E + 6 + k * 20;
                w16(ep - 2, pick([2, 2, 2, 3, 3, 0, 7])); w16(ep, ri(8)); w16(ep + 2, ri(4));
            }
        }
        w32(rec + 0x14, ct);
        const hdrPtr = alloc(0x48), hdr = alloc(8 + 16 * 12); fill(hdr, 8 + 16 * 12);
        w32(hdrPtr + 0x44, hdr);
        const subArr = alloc(8 * 4); w32(hdrPtr + 0x30, subArr);
        for (let k = 0; k < 8; k++) {
            if (chance(0.3)) { w32(subArr + k * 4, 0); continue; }
            const s = alloc(8), Y = alloc(8); w32(s + 4, Y); w32(Y + 4, ri(np)); w32(subArr + k * 4, s);
        }
        w32(rec + 0x18, hdrPtr);
    }
    const n = node(0);
    c.reg32[1] = ctx;
    stackArgs([n, chance(0.2) ? -1 : ri(8)]);
    const t0 = r32(n), top = t0 === 0xffffffff ? r32(r32(n + 0x64)) : t0;
    return { id: 132, cfg, tag: "132:" + ((toks[top] ?? 0) >>> 0).toString(16) };
}

function setCfgPtr(v) { new DataView(c.reg32.buffer).setUint32(ex.get_hypercall_page_ptr() + 0x1c54, v >>> 0, true); }
let dummyGen = 1;
/** Replace the handler's (ptr, generation)-keyed cfg cache with a throwaway entry, so each arm
 *  refreshes from the case's cfg exactly like the other did. */
function resetCfgCache() {
    const saved = cpuBlock().slice();
    c.write32(DUMMY, 2); c.write32(DUMMY + 0x38, dummyGen++);
    for (let o = 4; o < 0x54; o += 4) if (o !== 0x38) c.write32(DUMMY + o, 0);
    c.write32(DUMMY + 0x100, 0); c.write32(DUMMY + 0x104, 0);
    setCfgPtr(DUMMY); c.reg32[4] = DUMMY + 0xfc; c.reg32[1] = 0;
    ex.eagl_test_dispatch(132);                      // node == 0 -> declines right after the refresh
    cpuBlock().set(saved);
}

async function fillTlb(slack) {
    ex.full_clear_tlb();
    const saved = cpuBlock().slice();
    c.reg32[1] = TLB_MAX - 12 - slack; c.instruction_pointer[0] = TOUCH_ENTRY; c.last_virt_eip[0] = -1; c.in_hlt[0] = 0;
    await runGuest();
    cpuBlock().set(saved);
}

function outcome() {
    return { eip: c.instruction_pointer[0] >>> 0, cr2: c.cr[2] >>> 0, flags: c.flags[0] >>> 0, regs: Array.from(c.reg32, x => x >>> 0), hash: stateHash() };
}

async function runArm(cs, arm, variant, cfgPtr, handler, remap) {
    restore(cs);
    ex.eagl_read_cursor_set(1);
    ex.eagl_read_cursor_set_policy(variant.policy);
    ex.eagl_page_views_set(arm.views);
    ex.eagl_page_views_set_verify(arm.verify ? 1 : 0);
    if (ex.eagl_read_cursor_set_verify) ex.eagl_read_cursor_set_verify(arm.verify ? 1 : 0);
    if (cfgPtr) resetCfgCache();
    const regs0 = cpuBlock().slice();
    if (cfgPtr) setCfgPtr(cfgPtr);
    ex.full_clear_tlb();
    if (variant.kind === "tlbfull") { await fillTlb(variant.slack); cpuBlock().set(regs0); }
    const out = [];
    if (variant.kind === "cpl3") c.cpl[0] = 3;
    if (variant.kind === "warm" || variant.kind === "remap" || variant.kind === "cplswitch") {
        out.push(ex.eagl_test_dispatch(handler), outcome().hash);
        if (variant.kind === "remap") {
            // Move a page the first dispatch used to a different physical page, then flush.
            c.write32(pteAddr(remap.lin), remap.phys | 7);
            ex.full_clear_tlb();
        }
        restoreRegs(regs0);
        if (variant.kind === "cplswitch") c.cpl[0] = 3;
    }
    try { out.push(ex.eagl_test_dispatch(handler)); } catch (e) { out.push("threw: " + String(e).slice(0, 80)); }
    const o = { ret: out, cpl: c.cpl[0], ...outcome() };
    c.cpl[0] = 0;
    return o;
}
function restoreRegs(regs0) { cpuBlock().set(regs0); }

// ---- main loop ----------------------------------------------------------------------------
const mutant = ex.eagl_page_views_mutant();
const tally = { cases: 0, handled: 0, declined: 0, faulted: 0, byHandler: {}, variants: {} };
let viewHits = 0, viewFills = 0, oracleChecked = 0, oracleMismatch = 0, divergences = 0;
const firstDiv = [];
for (let n = 0; n < CASES; n++) {
    restore(base);
    ex.full_clear_tlb();
    used = [[DUMMY, DUMMY + 0x200]];
    setPages();
    randomPhys(ARENA, ARENA_PAGES << 12);
    randomPhys(BACK, ARENA_PAGES << 12);
    c.reg32.fill(0); c.reg32[4] = STACK_TOP - 0x400;
    const kind = pick(["128", "apply", "132", "132", "132"]);
    let handler, cfgPtr = 0, tag = kind;
    if (kind === "128") handler = build128();
    else if (kind === "apply") handler = buildApply();
    else { const b = build132(); handler = b.id; cfgPtr = b.cfg; tag = b.tag; }
    const cs = snapshot();
    const variant = { kind: pick(["cold", "cold", "warm", "remap", "remap", "tlbfull", "cplswitch", "cplswitch", "cpl3"]), policy: pick([1, 1, 1, 0]), slack: ri(24) };
    const remap = { lin: ARENA + (ri(ARENA_PAGES) << 12), phys: BACK + (ri(ARENA_PAGES) << 12) };
    // Remap a page the handler really uses: the page of one of its structures.
    if (used.length > 1) { const [s] = used[1 + ri(used.length - 1)]; remap.lin = s & ~0xfff; }
    const off = await runArm(cs, { views: 0 }, variant, cfgPtr, handler, remap);
    const on = await runArm(cs, { views: 1 }, variant, cfgPtr, handler, remap);
    ex.eagl_page_views_reset_stats();
    const orc = await runArm(cs, { views: 1, verify: true }, variant, cfgPtr, handler, remap);
    viewHits += ex.eagl_page_views_stat(0); viewFills += ex.eagl_page_views_stat(1);
    oracleChecked += ex.eagl_page_views_stat(2); oracleMismatch += ex.eagl_page_views_stat(3);
    if (ex.eagl_read_cursor_mismatch) { oracleMismatch += ex.eagl_read_cursor_mismatch() >>> 0; ex.eagl_read_cursor_reset_stats(); }
    const A = JSON.stringify(off), B = JSON.stringify(on), C = JSON.stringify(orc);
    tally.cases++;
    const tv = tally.variants[variant.kind] ??= { n: 0, handled: 0, faulted: 0, threw: 0 };
    tv.n++;
    if (typeof off.ret.at(-1) === "string") tv.threw++; else if (off.eip === HANDLER_PF) tv.faulted++; else if (off.ret.at(-1)) tv.handled++;
    const h = tally.byHandler[handler === 132 ? tag : handler] ??= { n: 0, handled: 0 };
    h.n++;
    const last = off.ret.at(-1);
    if (off.eip === HANDLER_PF) tally.faulted++; else if (last) { tally.handled++; h.handled++; } else tally.declined++;
    if (A !== B || A !== C) {
        divergences++;
        if (firstDiv.length < 3) firstDiv.push({ case: n, handler, variant, off, on: B === A ? "same" : on, oracle: C === A ? "same" : orc });
    }
}

const report = { engineMutant: mutant, seed: SEED, ...tally, viewHits, viewFills, oracleChecked, oracleMismatch, divergences };
console.log(JSON.stringify(report));
if (firstDiv.length) console.log(JSON.stringify(firstDiv, null, 1).slice(0, 4000));
em.destroy?.();
if (EXPECT_DIVERGE) {
    if (divergences === 0 && oracleMismatch === 0) { console.log("FAIL: negative control did not diverge"); process.exit(1); }
    console.log("PASS (negative control diverged as required)");
    process.exit(0);
}
const ok = divergences === 0 && oracleMismatch === 0 && viewHits > 0 && oracleChecked > 0 && tally.handled > CASES / 5 && tally.faulted > 0;
console.log(ok ? "PASS" : "FAIL");
process.exit(ok ? 0 : 1);
