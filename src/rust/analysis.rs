#![allow(non_snake_case)]

use crate::cpu_context::CpuContext;
use std::collections::{BTreeMap, HashMap, HashSet};
use crate::gen;
use crate::modrm;
use crate::prefix::{PREFIX_66, PREFIX_67, PREFIX_F2, PREFIX_F3, PREFIX_MASK_SEGMENT};
use crate::regs::{CS, DS, ES, FS, GS, SS};

#[derive(PartialEq, Eq)]
pub enum AnalysisType {
    Normal,
    BlockBoundary,
    Jump {
        offset: i32,
        is_32: bool,
        condition: Option<u8>,
    },
    STI,
}

pub struct Analysis {
    pub no_next_instruction: bool,
    pub absolute_jump: bool,
    pub ty: AnalysisType,
}

pub fn analyze_step(mut cpu: &mut CpuContext) -> Analysis {
    let mut analysis = Analysis {
        no_next_instruction: false,
        absolute_jump: false,
        ty: AnalysisType::Normal,
    };
    cpu.prefixes = 0;
    let opcode = cpu.read_imm8() as u32 | (cpu.osize_32() as u32) << 8;
    gen::analyzer::analyzer(opcode, &mut cpu, &mut analysis);
    analysis
}

pub fn analyze_step_handle_prefix(cpu: &mut CpuContext, analysis: &mut Analysis) {
    gen::analyzer::analyzer(
        cpu.read_imm8() as u32 | (cpu.osize_32() as u32) << 8,
        cpu,
        analysis,
    )
}
pub fn analyze_step_handle_segment_prefix(
    segment: u32,
    cpu: &mut CpuContext,
    analysis: &mut Analysis,
) {
    dbg_assert!(segment <= 5);
    cpu.prefixes = cpu.prefixes & !PREFIX_MASK_SEGMENT | (segment as u8 + 1);
    analyze_step_handle_prefix(cpu, analysis)
}

pub fn instr16_0F_analyze(cpu: &mut CpuContext, analysis: &mut Analysis) {
    gen::analyzer0f::analyzer(cpu.read_imm8() as u32, cpu, analysis)
}
pub fn instr32_0F_analyze(cpu: &mut CpuContext, analysis: &mut Analysis) {
    gen::analyzer0f::analyzer(cpu.read_imm8() as u32 | 0x100, cpu, analysis)
}
pub fn instr_26_analyze(cpu: &mut CpuContext, analysis: &mut Analysis) {
    analyze_step_handle_segment_prefix(ES, cpu, analysis)
}
pub fn instr_2E_analyze(cpu: &mut CpuContext, analysis: &mut Analysis) {
    analyze_step_handle_segment_prefix(CS, cpu, analysis)
}
pub fn instr_36_analyze(cpu: &mut CpuContext, analysis: &mut Analysis) {
    analyze_step_handle_segment_prefix(SS, cpu, analysis)
}
pub fn instr_3E_analyze(cpu: &mut CpuContext, analysis: &mut Analysis) {
    analyze_step_handle_segment_prefix(DS, cpu, analysis)
}
pub fn instr_64_analyze(cpu: &mut CpuContext, analysis: &mut Analysis) {
    analyze_step_handle_segment_prefix(FS, cpu, analysis)
}
pub fn instr_65_analyze(cpu: &mut CpuContext, analysis: &mut Analysis) {
    analyze_step_handle_segment_prefix(GS, cpu, analysis)
}
pub fn instr_66_analyze(cpu: &mut CpuContext, analysis: &mut Analysis) {
    cpu.prefixes |= PREFIX_66;
    analyze_step_handle_prefix(cpu, analysis)
}
pub fn instr_67_analyze(cpu: &mut CpuContext, analysis: &mut Analysis) {
    cpu.prefixes |= PREFIX_67;
    analyze_step_handle_prefix(cpu, analysis)
}
pub fn instr_F0_analyze(cpu: &mut CpuContext, analysis: &mut Analysis) {
    // lock: Ignored
    analyze_step_handle_prefix(cpu, analysis)
}
pub fn instr_F2_analyze(cpu: &mut CpuContext, analysis: &mut Analysis) {
    cpu.prefixes |= PREFIX_F2;
    analyze_step_handle_prefix(cpu, analysis)
}
pub fn instr_F3_analyze(cpu: &mut CpuContext, analysis: &mut Analysis) {
    cpu.prefixes |= PREFIX_F3;
    analyze_step_handle_prefix(cpu, analysis)
}

pub fn modrm_analyze(ctx: &mut CpuContext, modrm_byte: u8) { modrm::skip(ctx, modrm_byte); }

// ---------------------------------------------------------------------------------------------
// Guard groups (jit.rs set_jit_guard_groups): which guarded memory accesses of a module may run
// WITHOUT their own TLB check because an earlier access (the anchor) already validated the page.
//
// The facts are not decoded here from guest bytes: they are recorded from the module's own
// codegen in a dry-run pass, so they describe the code that is actually emitted — every guarded
// access (codegen::gen_safe_*), every write of a register local (the builder sees each one) and
// every helper call not known to leave translations alone. The emitting pass records the same
// stream again, and its module is kept only if the two streams are identical.
//
// Group key = (segment class, base, index, scale); a flat absolute operand keys on its page. A
// key becomes available at an anchor and stays available until a write to its index, a
// non-constant write to its base (a declared constant add shifts the key's offset instead), or a
// barrier. Across the module's CFG it is must-availability: a block the dispatcher can enter
// starts empty, and a join keeps a key only when every incoming path carries it at one offset.
// ---------------------------------------------------------------------------------------------

pub const GG_NOREG: u8 = 0xFF;
/// Segment class of an operand whose linear address is its effective address (flat DS/SS/CS
/// or the zero-segment prefix); any other class is `segment register + 1`.
pub const GG_SEG_FLAT: u8 = 0;

pub const GG_R: u8 = 0;
pub const GG_W: u8 = 1;
pub const GG_RMW: u8 = 2;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct GgDesc {
    pub seg: u8,
    pub base: u8,
    pub index: u8,
    pub scale: u8,
    pub disp: i32,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GgEvent {
    Block(u32),
    /// `top`: emitted at the instruction's own nesting level, so it runs whenever the
    /// instruction completes. Only such an access may anchor a group.
    Access { at: u32, kind: u8, width: u8, desc: Option<GgDesc>, top: bool },
    /// `delta`: new value = old value + constant, declared by the emitter at the write and only
    /// honoured at the instruction's top level.
    RegWrite { reg: u8, delta: Option<i32> },
    Barrier,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GgRole {
    None,
    /// Validates [addr + lo_rel, addr + lo_rel + len) as one page (writable and code-free when
    /// `write`) and publishes the TLB entry in `slot`.
    Anchor { slot: u8, lo_rel: i32, len: u32, write: bool },
    Member { slot: u8 },
}

pub struct GgBlock {
    pub addr: u32,
    pub succs: Vec<u32>,
    pub entry: bool,
}

pub struct GgPlan {
    pub roles: Vec<GgRole>,
    /// The recording pass's Access events by ordinal: the emitting pass must meet the same
    /// access at the same ordinal, or the plan is abandoned.
    pub accesses: Vec<GgEvent>,
    pub log: Vec<GgEvent>,
    pub slots: u8,
    pub anchors: u32,
    pub members: u32,
    pub dropped_extent: u32,
    pub dropped_slots: u32,
    pub converged: bool,
    /// Why no plan was made: 2 block set mismatch, 3 availability did not settle, 4 anchor
    /// sets did not settle.
    pub bail: u8,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
struct GgKey {
    seg: u8,
    base: u8,
    index: u8,
    scale: u8,
    page: u32,
}

fn gg_is_flat_abs(d: &GgDesc) -> bool {
    d.base == GG_NOREG && d.index == GG_NOREG && d.seg == GG_SEG_FLAT
}

fn gg_key(d: &GgDesc) -> GgKey {
    GgKey {
        seg: d.seg,
        base: d.base,
        index: d.index,
        scale: if d.index == GG_NOREG { 0 } else { d.scale },
        page: if gg_is_flat_abs(d) { (d.disp as u32) >> 12 } else { 0 },
    }
}

type GgState = HashMap<GgKey, (i32, Vec<u32>)>;

struct GgRec {
    member: bool,
    anchors: Vec<u32>,
    key: GgKey,
    off: i32,
    width: u8,
    kind: u8,
}

/// Negative control (UNSOUND): a base-register write neither ends nor shifts a group.
pub static mut GG_DEBUG_NO_BASE_KILL: bool = false;
/// Negative control (UNSOUND): barriers do not end groups.
pub static mut GG_DEBUG_NO_BARRIER: bool = false;

fn gg_transfer(
    events: &[(GgEvent, u32)],
    mut st: GgState,
    mut record: Option<&mut HashMap<u32, GgRec>>,
) -> GgState {
    let no_base_kill = unsafe { GG_DEBUG_NO_BASE_KILL };
    for &(ev, ord) in events {
        match ev {
            GgEvent::Block(_) => {},
            GgEvent::Barrier => {
                if !unsafe { GG_DEBUG_NO_BARRIER } {
                    st.clear()
                }
            },
            GgEvent::RegWrite { reg, delta } => {
                st.retain(|k, v| {
                    if k.index == reg {
                        return false;
                    }
                    if k.base == reg && !no_base_kill {
                        match delta {
                            Some(c) => v.0 = v.0.wrapping_add(c),
                            None => return false,
                        }
                    }
                    true
                });
            },
            GgEvent::Access { kind, width, desc, top, .. } => {
                let d = match desc {
                    Some(d) => d,
                    None => continue,
                };
                if gg_is_flat_abs(&d) && (d.disp as u32 & 0xFFF) + width as u32 > 0x1000 {
                    continue; // statically page-crossing: no group can hold it
                }
                let k = gg_key(&d);
                if let Some((dl, an)) = st.get(&k) {
                    if let Some(r) = record.as_deref_mut() {
                        r.insert(ord, GgRec {
                            member: true,
                            anchors: an.clone(),
                            key: k,
                            off: d.disp.wrapping_add(*dl),
                            width,
                            kind,
                        });
                    }
                }
                else if top {
                    st.insert(k, (0, vec![ord]));
                    if let Some(r) = record.as_deref_mut() {
                        r.insert(ord, GgRec {
                            member: false,
                            anchors: vec![ord],
                            key: k,
                            off: d.disp,
                            width,
                            kind,
                        });
                    }
                }
            },
        }
    }
    st
}

fn gg_meet(states: &[&GgState]) -> GgState {
    let mut out = GgState::new();
    if states.is_empty() {
        return out;
    }
    'keys: for (k, (dl, an)) in states[0].iter() {
        let mut anchors = an.clone();
        for s in &states[1..] {
            match s.get(k) {
                Some((dl2, an2)) if dl2 == dl => anchors.extend_from_slice(an2),
                _ => continue 'keys,
            }
        }
        anchors.sort_unstable();
        anchors.dedup();
        out.insert(*k, (*dl, anchors));
    }
    out
}

/// Plans one module from the recording pass. `max_extent` bounds the range an anchor validates
/// (a wide range fails its in-page test more often than grouping saves); `max_slots` bounds the
/// number of keys that get locals.
pub fn gg_plan(log: Vec<GgEvent>, blocks: &[GgBlock], max_extent: u32, max_slots: u8) -> GgPlan {
    // Events per block occurrence. A block the structure emits more than once (each copy is
    // real code) is planned as an entry block: every copy starts empty and publishes nothing to
    // its successors. Events before the first block belong to the module prologue, which every
    // entry passes through before any block, so they cannot carry a group into one.
    let mut per_block: HashMap<u32, Vec<Vec<(GgEvent, u32)>>> = HashMap::new();
    let mut accesses = Vec::new();
    let mut cur: Option<u32> = None;
    for &ev in &log {
        match ev {
            GgEvent::Block(a) => {
                per_block.entry(a).or_default().push(Vec::new());
                cur = Some(a);
            },
            _ => {
                let ord = if let GgEvent::Access { .. } = ev {
                    accesses.push(ev);
                    (accesses.len() - 1) as u32
                }
                else {
                    u32::MAX
                };
                if let Some(a) = cur {
                    per_block.get_mut(&a).unwrap().last_mut().unwrap().push((ev, ord));
                }
            },
        }
    }
    let dup: HashSet<u32> = per_block.iter().filter(|(_, v)| v.len() > 1).map(|(a, _)| *a).collect();
    let mut plan = GgPlan {
        roles: vec![GgRole::None; accesses.len()],
        accesses,
        log: Vec::new(),
        slots: 0,
        anchors: 0,
        members: 0,
        dropped_extent: 0,
        dropped_slots: 0,
        converged: false,
        bail: 0,
    };
    if blocks.len() != per_block.len() || blocks.iter().any(|b| !per_block.contains_key(&b.addr)) {
        plan.bail = 2;
        plan.log = log;
        return plan;
    }

    let mut addrs: Vec<u32> = blocks.iter().map(|b| b.addr).collect();
    addrs.sort_unstable();
    let by_addr: HashMap<u32, &GgBlock> = blocks.iter().map(|b| (b.addr, b)).collect();
    let mut preds: HashMap<u32, Vec<u32>> = HashMap::new();
    for b in blocks {
        for &s in &b.succs {
            if by_addr.contains_key(&s) {
                preds.entry(s).or_default().push(b.addr);
            }
        }
    }
    let strip = |st: GgState| -> GgState { st.into_iter().map(|(k, (d, _))| (k, (d, Vec::new()))).collect() };
    let meet2 = |a: &GgState, b: &GgState| -> GgState {
        a.iter()
            .filter(|(k, (d, _))| b.get(*k).map_or(false, |(d2, _)| d2 == d))
            .map(|(k, (d, _))| (*k, (*d, Vec::new())))
            .collect()
    };

    // Phase 1, availability (key + offset), must. A block the dispatcher can enter, or with no
    // predecessor, starts empty; otherwise predecessors without a state yet are skipped. Once a
    // block has an in-state it only ever shrinks (meet with the previous one): re-anchoring
    // resets a key's offset to 0, which is not monotone, and a loop can otherwise flip a key
    // between "member at offset d" and "anchor at 0" forever. At the fixpoint every in-state
    // is covered by all its predecessors' out-states, which is the soundness condition.
    let mut ins: HashMap<u32, GgState> = HashMap::new();
    let mut out: HashMap<u32, GgState> = HashMap::new();
    for _ in 0..1000 {
        let mut changed = false;
        for &a in &addrs {
            if dup.contains(&a) {
                if !out.contains_key(&a) {
                    ins.insert(a, GgState::new());
                    out.insert(a, GgState::new());
                    changed = true;
                }
                continue;
            }
            let computed = match preds.get(&a) {
                Some(ps) if !by_addr[&a].entry && !ps.is_empty() => {
                    let known: Vec<&GgState> = ps.iter().filter_map(|p| out.get(p)).collect();
                    if known.is_empty() {
                        continue; // not reachable yet in this sweep
                    }
                    strip(gg_meet(&known))
                },
                _ => GgState::new(),
            };
            let new_in = match ins.get(&a) {
                Some(old) => meet2(old, &computed),
                None => computed,
            };
            let o = strip(gg_transfer(&per_block[&a][0], new_in.clone(), None));
            if ins.get(&a) != Some(&new_in) {
                ins.insert(a, new_in);
                changed = true;
            }
            if out.get(&a) != Some(&o) {
                out.insert(a, o);
                changed = true;
            }
        }
        if !changed {
            plan.converged = true;
            break;
        }
    }
    if !plan.converged {
        plan.bail = 3;
        plan.log = log;
        return plan;
    }

    // Phase 2, which anchors can reach each member (may): availability is fixed now, so every
    // access is an anchor or a member in every sweep and the anchor sets only grow.
    let with_anchors = |a: u32, out3: &HashMap<u32, GgState>| -> GgState {
        let mut st = ins.get(&a).cloned().unwrap_or_default();
        if let Some(ps) = preds.get(&a) {
            if !by_addr[&a].entry && !dup.contains(&a) {
                for (k, v) in st.iter_mut() {
                    for p in ps {
                        if let Some((d2, an)) = out3.get(p).and_then(|o| o.get(k)) {
                            if *d2 == v.0 {
                                v.1.extend_from_slice(an);
                            }
                        }
                    }
                    v.1.sort_unstable();
                    v.1.dedup();
                }
            }
        }
        st
    };
    let mut out3: HashMap<u32, GgState> = HashMap::new();
    let mut settled = false;
    for _ in 0..1000 {
        let mut changed = false;
        for &a in &addrs {
            let o = if dup.contains(&a) { GgState::new() } else { gg_transfer(&per_block[&a][0], with_anchors(a, &out3), None) };
            if out3.get(&a) != Some(&o) {
                out3.insert(a, o);
                changed = true;
            }
        }
        if !changed {
            settled = true;
            break;
        }
    }
    if !settled {
        plan.converged = false;
        plan.bail = 4;
        plan.log = log;
        return plan;
    }

    let mut recs: HashMap<u32, GgRec> = HashMap::new();
    for &a in &addrs {
        for occurrence in &per_block[&a] {
            gg_transfer(occurrence, with_anchors(a, &out3), Some(&mut recs));
        }
    }
    // A member with no anchor would read a slot nobody wrote: never plan one.
    let anchor_ords: HashSet<u32> = recs.iter().filter(|(_, r)| !r.member).map(|(o, _)| *o).collect();
    recs.retain(|_, r| !r.member || (!r.anchors.is_empty() && r.anchors.iter().all(|a| anchor_ords.contains(a))));

    // Extent and write-need per anchor, over itself and every member that names it.
    struct Ext {
        lo: i64,
        hi: i64,
        write: bool,
        members: u32,
        key: GgKey,
    }
    let mut ext: BTreeMap<u32, Ext> = BTreeMap::new();
    for (&ord, r) in recs.iter().filter(|(_, r)| !r.member) {
        ext.insert(ord, Ext { lo: 0, hi: r.width as i64, write: r.kind != GG_R, members: 0, key: r.key });
    }
    for r in recs.values().filter(|r| r.member) {
        for an in &r.anchors {
            let rel = r.off.wrapping_sub(recs[an].off) as i64;
            let e = ext.get_mut(an).unwrap();
            e.lo = e.lo.min(rel);
            e.hi = e.hi.max(rel + r.width as i64);
            e.write |= r.kind != GG_R;
            e.members += 1;
        }
    }
    let mut kept: HashSet<u32> = HashSet::new();
    for (&ord, e) in &ext {
        if e.members == 0 {
            continue;
        }
        if e.hi - e.lo > max_extent.min(0x1000) as i64 {
            plan.dropped_extent += 1;
            continue;
        }
        kept.insert(ord);
    }
    let mut slot_of: BTreeMap<GgKey, u8> = BTreeMap::new();
    for (&ord, e) in &ext {
        if !kept.contains(&ord) || slot_of.contains_key(&e.key) {
            continue;
        }
        if (slot_of.len() as u8) < max_slots {
            let s = slot_of.len() as u8;
            slot_of.insert(e.key, s);
        }
        else {
            plan.dropped_slots += 1;
        }
    }
    kept.retain(|a| slot_of.contains_key(&ext[a].key));
    // A member keeps its role only if EVERY anchor that can reach it is kept: its slot holds
    // whatever the most recent anchor on the executed path published.
    let mut used: HashSet<u32> = HashSet::new();
    for (&ord, r) in recs.iter().filter(|(_, r)| r.member) {
        if r.anchors.iter().all(|a| kept.contains(a)) {
            plan.roles[ord as usize] = GgRole::Member { slot: slot_of[&r.key] };
            plan.members += 1;
            used.extend(r.anchors.iter().copied());
        }
    }
    for &ord in kept.iter().filter(|a| used.contains(a)) {
        let e = &ext[&ord];
        plan.roles[ord as usize] = GgRole::Anchor {
            slot: slot_of[&e.key],
            lo_rel: e.lo as i32,
            len: (e.hi - e.lo) as u32,
            write: e.write,
        };
        plan.anchors += 1;
    }
    plan.slots = slot_of.len() as u8;
    plan.log = log;
    plan
}

/// Planner self-test over synthetic streams (tests/jit-guard-group-diff.mjs runs it). Returns a
/// bitmask of the failed cases; 0 = all pass. Case 7 is the loop that re-anchors a key whose
/// offset changes on the back edge: an optimistic iteration without descent oscillates on it.
#[no_mangle]
pub fn gg_plan_selftest() -> u32 {
    const ESI: u8 = 6;
    const ESP: u8 = 4;
    let acc = |base: u8, disp: i32| GgEvent::Access {
        at: 0,
        kind: GG_R,
        width: 4,
        desc: Some(GgDesc { seg: GG_SEG_FLAT, base, index: GG_NOREG, scale: 0, disp }),
        top: true,
    };
    let reg = |reg: u8, delta: Option<i32>| GgEvent::RegWrite { reg, delta };
    let blk = |addr: u32, succs: Vec<u32>, entry: bool| GgBlock { addr, succs, entry };
    let members = |p: &GgPlan| p.roles.iter().filter(|r| matches!(r, GgRole::Member { .. })).count() as u32;
    let mut fail = 0u32;
    let mut check = |bit: u32, got: u32, want: u32| {
        if got != want {
            fail |= 1 << bit;
        }
    };
    // 0: three reads through esi -> 1 anchor + 2 members
    let p = gg_plan(vec![GgEvent::Block(0), acc(ESI, 0), acc(ESI, 4), acc(ESI, 8)], &[blk(0, vec![], true)], 1024, 8);
    check(0, members(&p), 2);
    // 1: base reload between: [esi+4] member, then esi written, [esi+8] anchor with no member
    let p = gg_plan(
        vec![GgEvent::Block(0), acc(ESI, 0), acc(ESI, 4), reg(ESI, None), acc(ESI, 8)],
        &[blk(0, vec![], true)],
        1024,
        8,
    );
    check(1, members(&p), 1);
    // 2: cross-block through a non-entry successor
    let p = gg_plan(
        vec![GgEvent::Block(0), acc(ESI, 0), GgEvent::Block(1), acc(ESI, 8)],
        &[blk(0, vec![1], true), blk(1, vec![], false)],
        1024,
        8,
    );
    check(2, members(&p), 1);
    // 3: the successor is an entry block: it starts empty
    let p = gg_plan(
        vec![GgEvent::Block(0), acc(ESI, 0), GgEvent::Block(1), acc(ESI, 8)],
        &[blk(0, vec![1], true), blk(1, vec![], true)],
        1024,
        8,
    );
    check(3, members(&p), 0);
    // 4: offset tracking: push (esp -4) between two stack reads
    let p = gg_plan(
        vec![GgEvent::Block(0), acc(ESP, 4), reg(ESP, Some(-4)), acc(ESP, 8)],
        &[blk(0, vec![], true)],
        1024,
        8,
    );
    check(4, members(&p), 1);
    // 5: a barrier kills
    let p = gg_plan(
        vec![GgEvent::Block(0), acc(ESI, 0), GgEvent::Barrier, acc(ESI, 4)],
        &[blk(0, vec![], true)],
        1024,
        8,
    );
    check(5, members(&p), 0);
    // 6: join with different offsets: [esp+8] after the join must anchor
    let p = gg_plan(
        vec![
            GgEvent::Block(0),
            acc(ESP, 0),
            GgEvent::Block(1),
            reg(ESP, Some(-4)),
            GgEvent::Block(2),
            acc(ESP, 8),
        ],
        &[blk(0, vec![1, 2], true), blk(1, vec![2], false), blk(2, vec![], false)],
        1024,
        8,
    );
    check(6, members(&p), 0);
    // 7: loop 1 -> 1 with "add esi, 4; [esi]" in the body and [esi] anchored before it: the
    // body access must anchor every iteration, and the planner must settle
    let p = gg_plan(
        vec![GgEvent::Block(0), acc(ESI, 0), GgEvent::Block(1), reg(ESI, Some(4)), acc(ESI, 0)],
        &[blk(0, vec![1], true), blk(1, vec![1], false)],
        1024,
        8,
    );
    check(7, p.converged as u32, 1);
    check(8, members(&p), 0);
    // 9: loop that leaves esi alone: the body access is a member of the pre-loop anchor
    let p = gg_plan(
        vec![GgEvent::Block(0), acc(ESI, 0), GgEvent::Block(1), acc(ESI, 4)],
        &[blk(0, vec![1], true), blk(1, vec![1], false)],
        1024,
        8,
    );
    check(9, members(&p), 1);
    // 10: an index write kills the key it indexes
    let idx = GgEvent::Access {
        at: 0,
        kind: GG_R,
        width: 4,
        desc: Some(GgDesc { seg: GG_SEG_FLAT, base: ESI, index: 1, scale: 2, disp: 0 }),
        top: true,
    };
    let p = gg_plan(vec![GgEvent::Block(0), idx, reg(1, Some(1)), idx], &[blk(0, vec![], true)], 1024, 8);
    check(10, members(&p), 0);
    // 11: extent beyond the cap drops the group
    let p = gg_plan(vec![GgEvent::Block(0), acc(ESI, 0), acc(ESI, 2000)], &[blk(0, vec![], true)], 1024, 8);
    check(11, members(&p), 0);
    // 12: a non-top access can be a member but never an anchor
    let low = GgEvent::Access {
        at: 0,
        kind: GG_R,
        width: 4,
        desc: Some(GgDesc { seg: GG_SEG_FLAT, base: ESI, index: GG_NOREG, scale: 0, disp: 0 }),
        top: false,
    };
    let p = gg_plan(vec![GgEvent::Block(0), low, acc(ESI, 4)], &[blk(0, vec![], true)], 1024, 8);
    check(12, members(&p), 0);
    // 13: a block emitted twice: each copy is planned on its own, from empty
    let p = gg_plan(
        vec![GgEvent::Block(0), acc(ESI, 0), acc(ESI, 4), GgEvent::Block(0), acc(ESI, 0), acc(ESI, 4)],
        &[blk(0, vec![], false)],
        1024,
        8,
    );
    check(13, members(&p), 2);
    // 14: a copy publishes nothing to its successor
    let p = gg_plan(
        vec![GgEvent::Block(0), acc(ESI, 0), GgEvent::Block(0), acc(ESI, 0), GgEvent::Block(1), acc(ESI, 4)],
        &[blk(0, vec![1], false), blk(1, vec![], false)],
        1024,
        8,
    );
    check(14, members(&p), 0);
    fail
}

/// The kill-all class of tools/guard-group-census.py, decided from the guest's code bytes at
/// `addr` (prefixes skipped): instructions that may change a translation, a segment base or
/// the privilege state, whether or not codegen implements them with a helper call.
pub fn gg_insn_is_kill_all(addr: u32, read8: &dyn Fn(u32) -> u8) -> bool {
    let mut a = addr;
    let mut op = read8(a);
    for _ in 0..14 {
        match op {
            0x26 | 0x2E | 0x36 | 0x3E | 0x64 | 0x65 | 0x66 | 0x67 | 0xF0 | 0xF2 | 0xF3 => {
                a = a.wrapping_add(1);
                op = read8(a);
            },
            _ => break,
        }
    }
    match op {
        // pop/mov sreg, les/lds, far call/jmp, retf, int3/int/into/iret, in/out/ins/outs, hlt
        0x07 | 0x17 | 0x1F | 0x8E | 0xC4 | 0xC5 | 0x9A | 0xEA | 0xCA | 0xCB | 0xCC | 0xCD
        | 0xCE | 0xCF | 0x6C..=0x6F | 0xE4..=0xE7 | 0xEC..=0xEF | 0xF4 => true,
        // call/jmp far through memory
        0xFF => matches!(read8(a.wrapping_add(1)) >> 3 & 7, 3 | 5),
        // descriptor tables, invlpg, lmsw, clts, invd/wbinvd, ud2, mov cr/dr, wrmsr,
        // sysenter/sysexit, pop fs/gs, lss/lfs/lgs
        0x0F => matches!(
            read8(a.wrapping_add(1)),
            0x00 | 0x01 | 0x06 | 0x08 | 0x09 | 0x0B | 0x20..=0x23 | 0x30 | 0x34 | 0x35 | 0xA1
                | 0xA9 | 0xB2 | 0xB4 | 0xB5
        ),
        _ => false,
    }
}

/// Helper calls that leave every translation as it was: memory slow paths (they may fill or
/// flush TLB ENTRIES — the same translation cached again, never a different one), fault
/// triggers (the module exits right after), and computation over CPU state. Anything else is
/// a barrier.
pub fn gg_call_is_transparent(name: &str) -> bool {
    if name.starts_with("safe_read")
        || name.starts_with("safe_write")
        || name.starts_with("report_")
        || name.starts_with("trigger_")
        || name.starts_with("task_switch_test")
        || name.starts_with("fpu_")
        || name.starts_with("sse_convert")
        || name.starts_with("div")
        || name.starts_with("idiv")
    {
        return true;
    }
    match name {
        "readable_or_pagefault_jit" | "writable_or_pagefault_jit" | "get_phys_eip_slow_jit"
        | "test_p" | "test_np" | "get_eflags" | "f80_to_f64" | "f80_to_f32" | "f64_to_f80_jit"
        | "f32_to_f80_jit" | "i32_to_f80_jit" | "i64_to_f80_jit" | "set_control_word"
        | "transition_fpu_to_mmx" | "mul16" | "imul16" | "imul_reg16" | "bsf32" | "bsr32"
        | "popcnt" | "rol32" | "ror32" | "rcl32" | "rcr32" | "adc8" | "sbb8" | "neg16"
        | "xadd16" | "cmpxchg16" | "maskmovq" | "maskmovdqu" | "read_tsc" => true,
        _ => gg_generic_helper_is_vector(name),
    }
}

/// Generic `instr*_<opcode>` helpers of the MMX/SSE opcode space (0F 10-17, 28-2F, 50-7F,
/// C2-C6, D0-FF, optionally 66/F2/F3-prefixed): register/memory data movement and arithmetic.
fn gg_generic_helper_is_vector(name: &str) -> bool {
    let rest = match name.strip_prefix("instr") {
        Some(r) => r,
        None => return false,
    };
    let rest = rest.strip_prefix("16").or_else(|| rest.strip_prefix("32")).unwrap_or(rest);
    let rest = match rest.strip_prefix('_') {
        Some(r) => r,
        None => return false,
    };
    let hex: &str = &rest[..rest.find(|c: char| !c.is_ascii_hexdigit()).unwrap_or(rest.len())];
    let hex = hex
        .strip_prefix("66")
        .or_else(|| hex.strip_prefix("F2"))
        .or_else(|| hex.strip_prefix("F3"))
        .unwrap_or(hex);
    match hex.strip_prefix("0F") {
        Some(o) if o.len() == 2 => match u8::from_str_radix(o, 16) {
            Ok(op) => matches!(op, 0x10..=0x17 | 0x28..=0x2F | 0x50..=0x7F | 0xC2..=0xC6 | 0xD0..=0xFF),
            Err(_) => false,
        },
        _ => false,
    }
}
