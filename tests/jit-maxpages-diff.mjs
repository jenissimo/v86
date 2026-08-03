#!/usr/bin/env node
// Differential oracle for the multi-page-module miscompilation class.
//
// Evidence (House of 1000 Doors, Blade of Darkness): guests die with garbage registers
// when JIT modules span MORE pages than the default budget (MAX_PAGES=3 survives,
// MAX_PAGES/TIER2_MAX_PAGES=8 dies), with guest-code invalidation ruled out. This test
// reproduces the CLASS headlessly: a seeded random control-flow graph (calls/rets,
// conditional branches, memory-counter loops with cross-page back edges, flag
// consumers) is scattered across N code pages, run to completion under (a) the
// interpreter, (b) JIT with MAX_PAGES=3, (c) JIT with MAX_PAGES=8 — all with the
// production BottleShip knob set (dead-flag elision, RET chaining, RET speculation,
// branch hints). Any divergence in final registers or the per-iteration checksum is a
// MISCOMPILE, not a perf property: a dead JIT still passes (the interpreter computes
// the same answers), which is exactly the discriminator the House/BoD class needs.
//
//   node tests/jit-maxpages-diff.mjs [seedStart] [seedCount]
//
// Exits non-zero on the first divergent seed, printing the seed and per-config state.

const { V86 } = await import("../build/libv86.mjs");

const BASE = 0x100000;
const N_PAGES = 24;                      // code pages — deliberately MORE than any module
                                         // budget, so modules must tile and overlap
const DATA_OFF = N_PAGES * 0x1000;       // one data page after the code
const CHK_OFF = DATA_OFF + 0x40;         // checksum dword
const CTR_OFF = DATA_OFF + 0x80;         // loop counters (dwords)
const FPTR_OFF = DATA_OFF + 0x180;       // 8-slot function-pointer table
const SCRATCH_OFF = DATA_OFF + 0x200;    // 64 dwords of scratch
const STACK = BASE + DATA_OFF + 0xF00;
const ENTRY_OFF = 0x20;
const OUTER = 30000;
const MEM_SIZE = 16 * 1024 * 1024;
const TIMEOUT_MS = 30000;

// ── deterministic PRNG ────────────────────────────────────────────────────────
function mulberry32(a) {
    return function() {
        a |= 0; a = a + 0x6D2B79F5 | 0;
        let t = Math.imul(a ^ a >>> 15, 1 | a);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
}

// ── tiny x86-32 assembler ─────────────────────────────────────────────────────
// regs: 0=eax 1=ecx 2=edx 3=ebx 5=ebp 6=esi 7=edi  (esp=4 reserved)
function makeAsm(buf, dv) {
    const labels = {}, patches = [];
    let o = 0;
    const a = {
        get o() { return o; },
        set o(v) { o = v; },
        labels,
        label: n => { labels[n] = o; },
        emit: (...b) => { for (const x of b) buf[o++] = x & 0xff; },
        u32: v => { dv.setUint32(o, v >>> 0, true); o += 4; },
        rel32: n => { patches.push({ at: o, end: o + 4, to: n }); o += 4; },
        abs32: n => { patches.push({ at: o, end: o + 4, to: n, abs: true }); o += 4; },
        movRegImm: (r, imm) => { a.emit(0xB8 + r); a.u32(imm); },
        aluRegImm: (op, r, imm) => { a.emit(0x81, 0xC0 | op << 3 | r); a.u32(imm); }, // op:0=add 1=or 4=and 5=sub 6=xor 7=cmp
        aluRegReg: (opc, rDst, rSrc) => { a.emit(opc, 0xC0 | rSrc << 3 | rDst); },    // 01 add,09 or,11 adc,19 sbb,21 and,29 sub,31 xor,39 cmp
        imulRegReg: (rDst, rSrc) => { a.emit(0x0F, 0xAF, 0xC0 | rDst << 3 | rSrc); },
        incReg: r => a.emit(0x40 + r),
        decReg: r => a.emit(0x48 + r),
        rolReg1: r => a.emit(0xD1, 0xC0 | r),
        pushReg: r => a.emit(0x50 + r),
        popReg: r => a.emit(0x58 + r),
        pushfd: () => a.emit(0x9C),
        movRegMem: (r, addr) => { a.emit(0x8B, 0x05 | r << 3); a.u32(addr); },
        movMemReg: (addr, r) => { a.emit(0x89, 0x05 | r << 3); a.u32(addr); },
        xorMemReg: (addr, r) => { a.emit(0x31, 0x05 | r << 3); a.u32(addr); },
        movMemImm: (addr, imm) => { a.emit(0xC7, 0x05); a.u32(addr); a.u32(imm); },
        // mov dword [addr], <address of label>  (abs32 patched at finalize)
        movMemLabel: (addr, n) => {
            a.emit(0xC7, 0x05); a.u32(addr);
            a.abs32(n);
        },
        decMem: addr => { a.emit(0xFF, 0x0D); a.u32(addr); },
        callMem: addr => { a.emit(0xFF, 0x15); a.u32(addr); },  // call [addr]
        jmp: n => { a.emit(0xE9); a.rel32(n); },
        jcc: (cc, n) => { a.emit(0x0F, 0x80 + cc); a.rel32(n); },  // cc: 4=z 5=nz 2=b 3=nb 8=s 9=ns
        call: n => { a.emit(0xE8); a.rel32(n); },
        ret: () => a.emit(0xC3),
        hlt: () => { a.emit(0xF4); a.emit(0xEB, 0xFE); },
        finalize: (imageBase) => {
            for (const p of patches) {
                const t = labels[p.to];
                if (t === undefined) throw new Error("undefined label " + p.to);
                if (p.abs) dv.setUint32(p.at, (imageBase + t) >>> 0, true);
                else dv.setInt32(p.at, t - p.end, true);
            }
        },
    };
    return a;
}

// ── random program ────────────────────────────────────────────────────────────
// fn_0 .. fn_{F-1}; fn_i may call fn_j only for j > i (no recursion). Each fn is a
// forward DAG of blocks plus 0-2 non-overlapping LOOP REGIONS [i..j]: a memory
// counter (re)initialized in block 0, decremented at the latch (block j), with a
// backward jnz to block i. Blocks are pinned to random pages, so back edges and
// call/ret edges cross pages constantly; call fall-throughs inside loop bodies make
// the SCCs multi-entry (dispatcher entries mid-loop), which is what stresses
// loopify/blockify and the entry-point machinery at wide page budgets.
function buildImage(seed) {
    const rnd = mulberry32(seed);
    const R = n => Math.floor(rnd() * n);
    const buf = new Uint8Array(DATA_OFF + 0x1000);
    const dv = new DataView(buf.buffer);

    // multiboot header
    const MAGIC = 0x1BADB002, FLAGS = 0x10000;
    dv.setUint32(0x00, MAGIC, true);
    dv.setUint32(0x04, FLAGS, true);
    dv.setUint32(0x08, (-(MAGIC + FLAGS)) >>> 0, true);
    dv.setUint32(0x0c, BASE, true);
    dv.setUint32(0x10, BASE, true);
    dv.setUint32(0x14, BASE + buf.length, true);
    dv.setUint32(0x18, BASE + buf.length, true);
    dv.setUint32(0x1c, BASE + ENTRY_OFF, true);

    const asm = makeAsm(buf, dv);
    const GPR = [0, 1, 2, 3, 5, 6]; // eax ecx edx ebx ebp esi (edi = outer counter)
    const CHK = BASE + CHK_OFF;
    const SCR = BASE + SCRATCH_OFF;

    const F = 10 + R(4);             // functions; the last two are indirect-call leaves
    const LEAF0 = F - 2;
    let ctrSlot = 0;
    const plans = [];
    for (let f = 0; f < F; f++) {
        const nb = 8 + R(12);        // blocks per fn
        let callSites = 0;           // bound call fan-out: exponential depth kills runtime
        const blocks = [];
        for (let b = 0; b < nb; b++) {
            const stmts = [];
            const ns = 1 + R(6);
            for (let s = 0; s < ns; s++) {
                const r1 = GPR[R(GPR.length)], r2 = GPR[R(GPR.length)];
                switch (R(12)) {
                    case 0: stmts.push(["movImm", r1, R(0x10000)]); break;
                    case 1: stmts.push(["aluImm", [0, 1, 4, 5, 6][R(5)], r1, R(0x100000)]); break;
                    case 2: stmts.push(["aluReg", [0x01, 0x09, 0x21, 0x29, 0x31][R(5)], r1, r2]); break;
                    case 3: stmts.push(["adcSbb", R(2) ? 0x11 : 0x19, r1, r2]); break; // reads CF
                    case 4: stmts.push(["imul", r1, r2]); break;
                    case 5: stmts.push(["incdec", r1, R(2)]); break;
                    case 6: stmts.push(["load", r1, SCR + 4 * R(64)]); break;
                    case 7: stmts.push(["store", SCR + 4 * R(64), r1]); break;
                    case 8: stmts.push(["rol", r1]); break;
                    case 9: stmts.push(["flagsFold", r1]); break;      // pushfd/pop/and/xor->chk
                    case 10:
                        // indirect call through the rewritable table — only from
                        // fns strictly below the leaves, so the call graph stays acyclic
                        if (f < LEAF0 && callSites < 2) { stmts.push(["callInd", R(8)]); callSites++; }
                        break;
                    case 11: {
                        // register-indirect, DATA-DEPENDENT address: the fastmem shapes
                        // guard computed addresses, absolute loads never exercise them
                        const rB = [0, 1, 2, 3, 6][R(5)];
                        stmts.push(["memInd", rB, r1 === rB ? (rB + 1) % 4 : r1, R(2), R(0x1C) * 4]);
                        break;
                    }
                    case 12:
                        // OUT — a block boundary mid-flow (the thunk shape): forces a
                        // module exit/re-entry and marks the fall-through as an entry
                        stmts.push(["outp"]);
                        break;
                }
            }
            blocks.push({ stmts, page: R(N_PAGES), term: null, loopInits: [], latch: null });
        }
        // loop regions: non-overlapping [i..j], i<j, counter init in block 0
        const nLoops = R(3);
        let lo = 1;
        for (let l = 0; l < nLoops && lo + 1 < nb; l++) {
            const i = lo + R(Math.max(1, nb - lo - 1));
            const j = i + 1 + R(Math.max(1, nb - i - 1));
            if (j >= nb) break;
            const ctr = BASE + CTR_OFF + 4 * ctrSlot++;
            blocks[0].loopInits.push([ctr, 2 + R(4)]);
            blocks[j].latch = [ctr, i];
            lo = j + 1;
        }
        for (let b = 0; b < nb; b++) {
            const last = b === nb - 1;
            const t = last ? 0 : R(12);
            if (last) blocks[b].term = ["ret"];
            else if (t < 4) blocks[b].term = ["jmp", b + 1];
            else if (t < 7) blocks[b].term = ["jcc", [4, 5, 2, 3, 8, 9][R(6)], b + 1 + R(nb - b - 1), b + 1,
                                              GPR[R(GPR.length)], R(0x10000)];
            else if (t < 9)
                // phase gate on the outer counter (edi counts DOWN): the taken path is
                // COLD for the first part of the run and becomes hot later — new entry
                // points appear mid-run, forcing recompiles that overwrite live modules.
                blocks[b].term = ["phase", OUTER >> (1 + R(2)), b + 1 + R(nb - b - 1), b + 1];
            else if (f + 1 < F && callSites < 2) {
                // depth discipline: shallow fns may call the next few, deep fns only the
                // leaves — keeps total invocations per iteration linear, not exponential
                const target = f < 3 ? f + 1 + R(Math.min(3, F - f - 1)) : LEAF0 + R(2);
                blocks[b].term = ["callThen", Math.max(target, f + 1), b + 1];
                callSites++;
            }
            else blocks[b].term = ["jmp", b + 1];
        }
        plans.push(blocks);
    }

    const emitted = [];
    const nameOf = (f, b) => `f${f}b${b}`;

    // driver (page 0, fixed at ENTRY_OFF)
    const FPTR = BASE + FPTR_OFF;
    const leafName = k => nameOf(LEAF0 + (k & 1), 0);
    asm.o = ENTRY_OFF;
    asm.movRegImm(4, STACK);          // mov esp, STACK
    // ── identity paging (matches the production regime: CR0.PG=1 unlocks fastmem
    // reads and the TLB/dispatch-meta path; unpaged guests never exercise either).
    // PD at 0x90000, 8 PTs at 0x91000.. mapping 0..32MB identity, all P|RW.
    {
        const PD = 0x90000, PT = 0x91000, NPT = 8;
        asm.movRegImm(0, PT | 3);         // eax = first PT entry for PD
        asm.movRegImm(1, NPT);            // ecx
        asm.movRegImm(2, PD);             // edx
        asm.label("pd_loop");
        asm.emit(0x89, 0x02);             // mov [edx], eax
        asm.aluRegImm(0, 0, 0x1000);      // add eax, 0x1000
        asm.aluRegImm(0, 2, 4);           // add edx, 4
        asm.decReg(1);
        asm.jcc(5, "pd_loop");            // jnz
        asm.movRegImm(0, 3);              // eax = 0|P|RW
        asm.movRegImm(1, NPT * 1024);     // ecx = PTE count
        asm.movRegImm(2, PT);             // edx = PT base
        asm.label("pt_loop");
        asm.emit(0x89, 0x02);             // mov [edx], eax
        asm.aluRegImm(0, 0, 0x1000);      // add eax, 0x1000 (next frame | flags)
        asm.aluRegImm(0, 2, 4);           // add edx, 4
        asm.decReg(1);
        asm.jcc(5, "pt_loop");            // jnz
        asm.movRegImm(0, PD);
        asm.emit(0x0F, 0x22, 0xD8);       // mov cr3, eax
        asm.emit(0x0F, 0x20, 0xC0);       // mov eax, cr0
        asm.aluRegImm(1, 0, 0x80000000);  // or eax, PG
        asm.emit(0x0F, 0x22, 0xC0);       // mov cr0, eax
    }
    for (const r of GPR) asm.movRegImm(r, 0x11111111 * (r + 1) >>> 0);
    for (let k = 0; k < 8; k++) asm.movMemLabel(FPTR + 4 * k, leafName(R(2)));
    asm.movRegImm(7, OUTER);          // edi = outer counter
    asm.label("outer");
    // phase rewrites of the indirect-call table (one-shot at edi == K)
    for (let k = 0; k < 3; k++) {
        asm.aluRegImm(7, 7, (OUTER >> (1 + k)) + 1); // cmp edi, K
        asm.jcc(5, `noswap${k}`);                     // jnz
        asm.movMemLabel(FPTR + 4 * R(8), leafName(R(2)));
        asm.movMemLabel(FPTR + 4 * R(8), leafName(R(2)));
        asm.label(`noswap${k}`);
    }
    asm.call(nameOf(0, 0));
    for (const r of GPR) asm.xorMemReg(CHK, r);
    asm.movRegMem(0, CHK); asm.rolReg1(0); asm.movMemReg(CHK, 0);
    asm.decReg(7);
    asm.jcc(5, "outer_more");         // jnz
    asm.hlt();
    asm.label("outer_more");
    asm.jmp("outer");

    const cursors = new Array(N_PAGES).fill(0);
    cursors[0] = asm.o;
    const PAGE_LIMIT = 0x1000 - 0x30; // stay clear of near-end-of-page
    const place = (page, size) => {
        let p = page;
        for (let tries = 0; tries < N_PAGES; tries++, p = (p + 1) % N_PAGES) {
            if (cursors[p] + size <= PAGE_LIMIT) return { p, at: p * 0x1000 + cursors[p] };
        }
        throw new Error("out of code space");
    };

    const genBlock = (f, b) => (a) => {
        const blk = plans[f][b];
        a.label(nameOf(f, b));
        for (const [ctr, n] of blk.loopInits) a.movMemImm(ctr, n);
        for (const st of blk.stmts) {
            switch (st[0]) {
                case "movImm": a.movRegImm(st[1], st[2]); break;
                case "aluImm": a.aluRegImm(st[1], st[2], st[3]); break;
                case "aluReg": a.aluRegReg(st[1], st[2], st[3]); break;
                case "adcSbb": a.aluRegReg(st[1], st[2], st[3]); break;
                case "imul": a.imulRegReg(st[1], st[2]); break;
                case "incdec": st[2] ? a.incReg(st[1]) : a.decReg(st[1]); break;
                case "load": a.movRegMem(st[1], st[2]); break;
                case "store": a.movMemReg(st[1], st[2]); break;
                case "rol": a.rolReg1(st[1]); break;
                case "flagsFold":
                    // Define ALL folded flags first (cmp defines CF/PF/AF/ZF/SF/OF):
                    // after imul SF/ZF/AF/PF are architecturally UNDEFINED, and after
                    // and/or/xor AF is — interpreter and JIT may legitimately differ.
                    a.aluRegImm(7, st[1], 0x5A5A);
                    a.pushfd(); a.popReg(st[1]);
                    a.aluRegImm(4, st[1], 0x8D5);
                    a.xorMemReg(CHK, st[1]);
                    break;
                case "callInd":
                    a.callMem(BASE + FPTR_OFF + 4 * st[1]);
                    break;
                case "memInd": {
                    // rB = SCR + (rV & 0xFC)  (data-dependent, stays inside scratch);
                    // then load/store [rB + disp8]
                    const [, rB, rV, isStore, disp] = st;
                    a.movRegImm(rB, SCR);
                    a.pushReg(rV);
                    a.aluRegImm(4, rV, 0xFC);            // and rV, 0xFC
                    a.aluRegReg(0x01, rB, rV);           // add rB, rV
                    a.popReg(rV);
                    if (isStore) a.emit(0x89, 0x40 | rV << 3 | rB, disp & 0x7F); // mov [rB+d8], rV
                    else a.emit(0x8B, 0x40 | rV << 3 | rB, disp & 0x7F);         // mov rV, [rB+d8]
                    break;
                }
            }
        }
        if (blk.latch) {
            const [ctr, head] = blk.latch;
            a.decMem(ctr);
            a.jcc(5, nameOf(f, head));    // jnz backward — the loop edge
        }
        const t = blk.term;
        switch (t[0]) {
            case "ret": a.ret(); break;
            case "jmp": a.jmp(nameOf(f, Math.min(t[1], plans[f].length - 1))); break;
            case "jcc":
                // Definer before the consumer: without it the branch could read
                // flags left undefined by imul (SF/ZF) — legitimate JIT/interp skew.
                a.aluRegImm(7, t[4], t[5]);
                a.jcc(t[1], nameOf(f, Math.min(t[2], plans[f].length - 1)));
                a.jmp(nameOf(f, Math.min(t[3], plans[f].length - 1)));
                break;
            case "callThen":
                a.call(nameOf(t[1], 0));
                a.jmp(nameOf(f, Math.min(t[2], plans[f].length - 1)));
                break;
            case "phase":
                a.aluRegImm(7, 7, t[1]);                              // cmp edi, K
                a.jcc(2, nameOf(f, Math.min(t[2], plans[f].length - 1))); // jb (late phase)
                a.jmp(nameOf(f, Math.min(t[3], plans[f].length - 1)));
                break;
        }
    };

    for (let f = 0; f < F; f++)
        for (let b = 0; b < plans[f].length; b++)
            emitted.push({ f, b, gen: genBlock(f, b) });

    // measure sizes with a throwaway assembler
    for (const e of emitted) {
        const sb = new Uint8Array(0x800);
        const sa = makeAsm(sb, new DataView(sb.buffer));
        sa.rel32 = () => { sa.o += 4; };
        sa.label = () => {};
        e.gen(sa);
        e.size = sa.o;
    }
    for (const e of emitted) {
        const pin = plans[e.f][e.b].page;
        const { p, at } = place(pin, e.size);
        cursors[p] = (at & 0xFFF) + e.size;
        asm.o = at;
        e.gen(asm);
        if (asm.o !== at + e.size) throw new Error("size mismatch");
    }

    asm.finalize(BASE);
    return buf;
}

// ── run one config ────────────────────────────────────────────────────────────
function run(image, cfg) {
    return new Promise(resolve => {
        const emulator = new V86({ autostart: false, memory_size: MEM_SIZE,
                                   disable_jit: cfg.jit ? 0 : 1, log_level: 0 });
        let halted = false, timer;
        globalThis.__jitCompileStats = { count: 0, bytes: 0 };
        const finish = status => {
            clearTimeout(timer);
            try { emulator.stop(); } catch (e) {}
            const cpu = emulator.v86.cpu;
            const regs = Array.from({ length: 8 }, (_, i) => cpu.reg32[i] >>> 0);
            const chk = new DataView(cpu.mem8.buffer, cpu.mem8.byteOffset)
                .getUint32(BASE + CHK_OFF, true);
            // widest live module span, in pages (are we actually stressing the class?)
            let maxSpan = 0;
            const w = cpu.wm?.exports;
            if (cfg.jit && w?.jit_aot_page_table_index && w?.jit_aot_module_page_count) {
                const seen = new Set();
                for (let p = 0; p < N_PAGES; p++) {
                    const idx = w.jit_aot_page_table_index(BASE + p * 0x1000) >>> 0;
                    if (idx === 0xFFFF || seen.has(idx)) continue;
                    seen.add(idx);
                    maxSpan = Math.max(maxSpan, w.jit_aot_module_page_count(idx) >>> 0);
                }
            }
            const paged = ((cpu.cr?.[0] ?? 0) & 0x80000000) !== 0;
            resolve({ status, regs, chk, generated: globalThis.__jitCompileStats.count | 0, maxSpan, paged });
        };
        emulator.bus.register("cpu-event-halt", () => { halted = true; finish("halt"); });
        emulator.add_listener("emulator-loaded", () => {
            const cpu = emulator.v86.cpu;
            cpu.reboot_internal();
            cpu.reset_memory();
            if (cfg.jit && cpu.set_jit_config) {
                cpu.set_jit_config(1, cfg.maxPages);   // MAX_PAGES
                cpu.set_jit_config(5, 1);              // dead-flag elision (prod default)
                cpu.set_jit_config(10, 1);             // explicit x87-locals ablation
                cpu.set_jit_config(11, 1);             // push-run coalescing (prod default)
                cpu.set_jit_config(12, 1);             // RET dynamic chaining (prod default)
                cpu.set_jit_config(13, 1);             // RET speculation (prod default)
                cpu.set_jit_config(22, 1);             // branch hints bit0 (prod default)
                cpu.set_jit_config(15, cfg.tier2 ?? 0);// tier-2 threshold
                cpu.jit_clear_cache?.();
            }
            cpu.load_multiboot(image.buffer);
            timer = setTimeout(() => { if (!halted) finish("HANG"); }, TIMEOUT_MS);
            emulator.run();
        });
    });
}

// ── main ──────────────────────────────────────────────────────────────────────
const seedStart = parseInt(process.argv[2] ?? "1", 10);
const seedCount = parseInt(process.argv[3] ?? "20", 10);

let failures = 0;
for (let seed = seedStart; seed < seedStart + seedCount; seed++) {
    let image;
    try { image = buildImage(seed); }
    catch (e) { console.log(`seed ${seed}: skipped (${e.message})`); continue; }

    const interp = await run(image, { jit: false });
    const jit3 = await run(image, { jit: true, maxPages: 3 });
    const jit8 = await run(image, { jit: true, maxPages: 8 });
    // mirrors the production default arm: MAX_PAGES=3 but tier-2 promotion ON with the
    // default TIER2_MAX_PAGES=8 budget; low threshold so promotions actually happen here
    const jitT2 = await run(image, { jit: true, maxPages: 3, tier2: 20000 });

    const sig = r => `${r.status} chk=${(r.chk >>> 0).toString(16)} regs=${r.regs.map(x => x.toString(16)).join(",")}`;
    const ok3 = sig(interp) === sig(jit3);
    const ok8 = sig(interp) === sig(jit8);
    const okT2 = sig(interp) === sig(jitT2);
    const note = ` (gen3=${jit3.generated} gen8=${jit8.generated} genT2=${jitT2.generated}` +
                 ` span3=${jit3.maxSpan} span8=${jit8.maxSpan} spanT2=${jitT2.maxSpan}` +
                 ` paged=${jit8.paged ? 1 : 0})`;
    if (ok3 && ok8 && okT2) {
        console.log(`seed ${seed}: ok${note}`);
    }
    else {
        failures++;
        console.error(`seed ${seed}: DIVERGENCE${note}`);
        console.error(`  interp: ${sig(interp)}`);
        console.error(`  jit3:   ${sig(jit3)}   ${ok3 ? "(match)" : "(MISMATCH)"}`);
        console.error(`  jit8:   ${sig(jit8)}   ${ok8 ? "(match)" : "(MISMATCH)"}`);
        console.error(`  jitT2:  ${sig(jitT2)}   ${okT2 ? "(match)" : "(MISMATCH)"}`);
    }
}
if (failures) { console.error(`FAIL: ${failures} divergent seed(s)`); process.exit(1); }
console.log("all seeds consistent");
process.exit(0);
