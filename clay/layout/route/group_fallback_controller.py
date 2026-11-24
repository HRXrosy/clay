"""Run the declared group-route pass and causally retry eligible local exhaustion."""
from dataclasses import dataclass
from clay.layout.route.engine_gridless import (
    GroupRouteBudgetStop,
    GroupRouteLandingUnavailable,
)


@dataclass(frozen=True)
class LocalExhaustion(Exception):
    group: str
    net: str
    # Complete native diagnostic, including all hard rejection provenance.
    edge_fail: tuple
    details: object
    original: Exception
    samples: tuple = ()

    @property
    def committed_owners(self):
        failure = self.edge_fail
        if (type(failure) is not tuple or len(failure) != 6
            or failure[0] != 'exhausted'
            or any(type(x) is not int or x < 0 for x in failure[1:5])
            or type(failure[5]) is not tuple):
            return frozenset()
        owners = set()
        for row in failure[5]:
            if (type(row) is not tuple or len(row) != 3
                or type(row[0]) is not str or type(row[1]) is not str
                or type(row[2]) is not int or row[2] <= 0):
                return frozenset()
            if row[0].startswith('committed_'):
                owners.add(row[1])
        return frozenset(owners)


@dataclass(frozen=True)
class LocalInconclusive(Exception):
    """One interface did not close; the symbolic endpoint domain is incomplete."""
    group: str
    net: str
    edge_fail: tuple
    details: object
    original: Exception
    samples: tuple = ()
    co_starved: tuple = ()

    @property
    def committed_owners(self):
        """Identify recorded blockers as trial hints, never as exhaustion proof."""
        value = self.edge_fail
        edge_incomplete_kinds = {
            'candidate_source_union_unknown',
            'disconnected',
            'static_disconnected',
            'planar_shared_endpoint_proof_failed',
        }
        is_wrapped_edge_fail = (
            type(value) is tuple and len(value) == 3
            and value[0] == 'symbolic_domain_incomplete'
            and type(value[1]) is tuple and len(value[1]) >= 1
            and value[1][0] in edge_incomplete_kinds
        )
        if (
            (type(value) is tuple and len(value) >= 1 and value[0] in edge_incomplete_kinds)
            or is_wrapped_edge_fail
        ):
            found = set()
            if type(self.samples) is tuple:
                for raw in self.samples:
                    if (type(raw) is not tuple or any(type(p) is not tuple or len(p) != 2
                            or type(p[0]) is not str for p in raw)):
                        continue
                    item = dict(raw)
                    cat = item.get('category')
                    owner = item.get('owner')
                    if (type(cat) is str and cat.startswith('committed_')
                            and type(owner) is str and owner and owner != self.net
                            and owner != '__clay_group_causal_reservation__'):
                        found.add(owner)
            if found:
                return frozenset(found)
            inner = value[1] if is_wrapped_edge_fail else value
            summary = inner[6] if len(inner) > 6 and type(inner[6]) is tuple else ()
            return frozenset(
                row[1]
                for row in summary
                if type(row) is tuple and len(row) >= 2
                and type(row[0]) is str and row[0].startswith('committed_')
                and type(row[1]) is str and row[1] and row[1] != self.net
                and row[1] != '__clay_group_causal_reservation__'
            )
        if (type(value) is not tuple or len(value) != 3 or value[0] != 'symbolic_domain_incomplete'
                or type(value[1]) is not tuple or len(value[1]) != 6
                or value[1][0] != 'exhausted'
                or any(type(x) is not int or x < 0 for x in value[1][1:5])
                or type(value[1][5]) is not tuple or type(value[2]) is not tuple):
            return frozenset()
        def owners(rows):
            if any(type(row) is not tuple or len(row) != 3 or type(row[0]) is not str
                   or type(row[1]) is not str or type(row[2]) is not int or row[2] <= 0
                   for row in rows):
                return frozenset()
            return frozenset(row[1] for row in rows if row[0].startswith('committed_'))
        return owners(value[1][5]) & owners(value[2])


@dataclass(frozen=True)
class _Frame:
    index: int
    port: object
    snapshot: object
    selected: object
    key: object


class _PhysicalRejection(Exception):
    pass


class FullPrefixFeedbackRejected(RuntimeError):
    """The actual group prefix failed feedback and no tried single change passed."""

    def __init__(self, attempts, first_feedback=None):
        self.attempts = attempts
        self.first_feedback = first_feedback
        super().__init__('complete group prefix failed physical feedback; replays forcing one interface and regenerating subsequent interfaces did not pass')

    def flow_error_data(self):
        return {'reason':'GROUP_PREFIX_PHYSICAL_REJECTED', 'attempts':self.attempts,
                'first_feedback':self.first_feedback, 'search_scope':'one forced interface with subsequent interface regeneration',
                'physical_admission':False, 'infeasibility':'UNKNOWN'}


class _BranchLandingUnavailable(Exception):
    def __init__(self, original):
        self.original = original
        super().__init__(str(original))


@dataclass(frozen=True)
class _Continue:
    index: int
    frames: tuple
    guards: tuple


@dataclass(frozen=True)
class _Complete:
    frames: tuple


def allocate(ports, adapter, *, full_prefix_feedback=False):
    """Run the original greedy pass and, only on typed exhaustion, replay.

    Adapter methods: snapshot/restore; checkpoint(port, phase); candidates;
    default_candidate; candidate_key; snapshot_key; alternatives; route; owns_committed;
    physical_nonworsening(snapshot).  That feedback must also reject any
    electrical short using the production short and deck-connectivity checks:
    merged-layer spacing alone can hide a new short. Snapshot must include ALL transactional
    fields, endpoint publication and scratch, but never monotonic counters.
    route raises LocalExhaustion only for the native exact exhausted shape.
    A LocalInconclusive can try another interface in the same restored world;
    exhausting those trials preserves the original unknown result.
    all budget, watchdog, cancellation, invalid input and observer exceptions
    propagate.  On a trial, feedback compares the complete emitted geometry at
    the replayed prefix against the restored historical pre-choice world.
    """
    if type(full_prefix_feedback) is not bool:
        raise TypeError('full-prefix feedback flag must be a bool')
    ports = list(ports)
    groups = [port.group for port in ports]
    if any(groups[i] != groups[i - 1] and groups[i] in groups[:i]
           for i in range(1, len(groups))):
        raise ValueError('group-route ports must be contiguous by group')
    group_spans = {}
    for idx, port in enumerate(ports):
        if port.group not in group_spans:
            group_spans[port.group] = [idx, idx + 1]
        else:
            group_spans[port.group][1] = idx + 1
    initial = adapter.snapshot()
    save_history = getattr(adapter, 'save_history', None)
    restore_history = getattr(adapter, 'restore_history', None)
    apply_negotiated_history = getattr(adapter, 'apply_negotiated_history', None)
    supports_negotiation = bool(
        getattr(adapter, 'supports_group_negotiation', lambda: False)()
    )
    recoverable_budget = getattr(adapter, 'recoverable_edge_budget_failure', None)
    base_history = None if save_history is None else save_history()
    promoted_victims = {}
    walk_prefix_sink = None
    neg_trial_sink = None
    active_neg_group_history = None
    visited = set()
    first_failure = None
    pending_recovery_budgets = []

    def reset_history():
        if restore_history is not None:
            restore_history(base_history)

    def _is_recoverable_edge_budget(exc):
        if type(exc) is not GroupRouteBudgetStop:
            return False
        diag = exc.flow_error_data()
        return diag.get('reason') in {'EDGE_EXPANSION_BUDGET', 'RESOURCE_BUDGET_EXCEEDED'}

    def remember_recovery_budget(exc):
        if isinstance(exc, (LocalInconclusive, LocalExhaustion)):
            exc = exc.original
        if type(exc) is GroupRouteBudgetStop:
            if not _is_recoverable_edge_budget(exc):
                raise exc
            pending_recovery_budgets.append(exc)

    def negotiate_group_recovery(failure, failed_world, frames, causal_frames, guards,
                                 group_start_hint=None):
        if not supports_negotiation:
            return None
        engine_obj = getattr(adapter, 'engine', None)
        prior_neg_legalize = (
            getattr(engine_obj, '_negotiation_port_legalize', False)
            if engine_obj is not None else False
        )
        if engine_obj is not None:
            engine_obj._negotiation_port_legalize = True
        try:
            return _negotiate_group_recovery_inner(
                failure, failed_world, frames, causal_frames, guards, group_start_hint
            )
        finally:
            reset_history()
            if engine_obj is not None:
                if prior_neg_legalize:
                    engine_obj._negotiation_port_legalize = True
                else:
                    engine_obj.__dict__.pop('_negotiation_port_legalize', None)

    def _negotiate_group_recovery_inner(failure, failed_world, frames, causal_frames,
                                        guards, group_start_hint=None):
        nonlocal walk_prefix_sink, neg_trial_sink, active_neg_group_history
        g_start, g_end = group_spans[failure.group]
        prepare_watch = getattr(adapter, 'prepare_escape_starvation_watch', None)
        prepare_neg = getattr(adapter, 'prepare_group_negotiation', None)
        clean_snap = frames[g_start].snapshot if len(frames) > g_start else group_start_hint
        if prepare_watch is not None:
            prepare_watch(failure.group, tuple(ports[g_start:g_end]), clean_snap)
        promoted = promoted_victims.setdefault(failure.group, set())
        vp_best = None
        vp_solo_blocks = set()
        vp_ever_blockers = {}
        exact_witness = getattr(adapter, 'owns_committed_witness', None)
        if failure.net not in promoted and len(frames) > g_start:
            promoted.add(failure.net)
            saved_slice = list(ports[g_start:g_end])
            by_net = {p.net: p for p in saved_slice}
            victim_port = by_net.get(failure.net)
            if victim_port is not None and saved_slice[0].net != victim_port.net:
                start_snapshot = frames[g_start].snapshot
                cascade_prefixes = {(): (start_snapshot, frames[:g_start])}
                for k in range(1, min(len(saved_slice), len(frames) - g_start)):
                    if ([frames[g_start + j].port.net for j in range(k)]
                            == [saved_slice[j].net for j in range(k)]):
                        cascade_prefixes[tuple(p.net for p in saved_slice[:k])] = (
                            frames[g_start + k].snapshot,
                            frames[:g_start + k],
                        )
                victim_rel = next(
                    (i for i, p in enumerate(saved_slice) if p.net == victim_port.net), 0
                )
                rel_causal = sorted({
                    f.index - g_start
                    for f in causal_frames
                    if 0 <= f.index - g_start < victim_rel
                })
                cand_slots = []
                if len(rel_causal) >= 2:
                    cand_slots.append(min(rel_causal))
                    if max(rel_causal) not in cand_slots:
                        cand_slots.append(max(rel_causal))
                elif len(rel_causal) == 1:
                    cand_slots.append(rel_causal[0])
                if 0 not in cand_slots:
                    cand_slots.append(0)
                orig_index = {p.net: i for i, p in enumerate(saved_slice)}

                def _build_vp_order(prom_ports, slots_map):
                    f_nets = {p.net for p in prom_ports}
                    buckets = {}
                    for p in prom_ports:
                        s_pos = max(0, min(len(saved_slice), int(slots_map[p.net])))
                        buckets.setdefault(s_pos, []).append(p)
                    ordered = []
                    for s_idx, orig_p in enumerate(saved_slice):
                        if s_idx in buckets:
                            ordered.extend(buckets[s_idx])
                        if orig_p.net not in f_nets:
                            ordered.append(orig_p)
                    if len(saved_slice) in buckets:
                        ordered.extend(buckets[len(saved_slice)])
                    return ordered

                def _rej_owner_set(exc_obj):
                    if not isinstance(exc_obj, (LocalInconclusive, LocalExhaustion)):
                        return set()
                    owners_set = set(exc_obj.committed_owners)
                    if exact_witness is not None:
                        for cand_p in saved_slice:
                            if cand_p.net != exc_obj.net and exact_witness(cand_p, exc_obj):
                                owners_set.add(cand_p.net)
                    return {o for o in owners_set if o in orig_index and o != exc_obj.net}

                slot_idx = 0
                insert_pos = cand_slots[0]
                prev_sink = walk_prefix_sink
                walk_prefix_sink = (g_start, g_end, cascade_prefixes)
                promoted_ports = [victim_port]
                slot_by_net = {victim_port.net: insert_pos}
                max_cascade = max(1, len(saved_slice) - 1)
                seen_vp_orders = {tuple(p.net for p in saved_slice)}
                solo_blocks = set()
                try:
                    for _cascade_step in range(len(cand_slots) * (max_cascade + 2)):
                        front_nets = {p.net for p in promoted_ports}
                        target_order = _build_vp_order(promoted_ports, slot_by_net)
                        seen_vp_orders.add(tuple(p.net for p in target_order))
                        ports[g_start:g_end] = target_order
                        resume_k = 0
                        for k in range(len(target_order) - 1, 0, -1):
                            if tuple(p.net for p in target_order[:k]) in cascade_prefixes:
                                resume_k = k
                                break
                        resume_snap, resume_frames = cascade_prefixes[
                            tuple(p.net for p in target_order[:resume_k])
                        ]
                        adapter.restore(resume_snap)
                        reset_history()
                        adapter.checkpoint(ports[g_start + resume_k], 'group_fallback_victim_promotion')
                        try:
                            return walk(
                                g_start + resume_k,
                                resume_frames,
                                (*(g for g in guards if g[0] < g_start),
                                 (g_start, start_snapshot, ports[g_start])),
                            )
                        except (_BranchLandingUnavailable, LocalInconclusive,
                                LocalExhaustion, _PhysicalRejection,
                                GroupRouteBudgetStop) as rejected:
                            if (type(rejected) is GroupRouteBudgetStop
                                    and not _is_recoverable_edge_budget(rejected)):
                                raise
                            remember_recovery_budget(rejected)
                            vp_world = adapter.snapshot()
                            vp_k = 0
                            for k in range(len(target_order), 0, -1):
                                if tuple(p.net for p in target_order[:k]) in cascade_prefixes:
                                    vp_k = k
                                    break
                            vp_prog = g_start + vp_k
                            victim_already_closed = (
                                vp_prog >= len(frames)
                                and any(p.net == victim_port.net for p in target_order[:vp_k])
                                and isinstance(rejected, (LocalInconclusive, LocalExhaustion))
                                and orig_index.get(rejected.net, -1) > victim_rel
                            )
                            vp_closed_victim_cleanly = (
                                vp_prog == len(frames) and victim_already_closed
                            )
                            if ((vp_prog > len(frames) or vp_closed_victim_cleanly)
                                    and (vp_best is None or vp_prog > vp_best[0])):
                                vp_frames = list(
                                    cascade_prefixes[tuple(p.net for p in target_order[:vp_k])][1]
                                )
                                vp_rej = rejected
                                if (type(vp_rej) is GroupRouteBudgetStop
                                        and recoverable_budget is not None
                                        and vp_prog < g_end):
                                    conv = recoverable_budget(
                                        target_order[vp_k], vp_rej, tuple(vp_frames)
                                    )
                                    if conv is not None:
                                        vp_rej = conv
                                if (isinstance(vp_rej, (LocalInconclusive, LocalExhaustion))
                                        and vp_rej.group == failure.group):
                                    vp_owners = set(vp_rej.committed_owners)
                                    for co_vp in getattr(vp_rej, 'co_starved', ()):
                                        if (isinstance(co_vp, (LocalInconclusive, LocalExhaustion))
                                                and co_vp.group == failure.group):
                                            vp_owners.update(co_vp.committed_owners)
                                    if exact_witness is not None:
                                        for p in target_order:
                                            if p.net != vp_rej.net and exact_witness(p, vp_rej):
                                                vp_owners.add(p.net)
                                            for co_vp in getattr(vp_rej, 'co_starved', ()):
                                                if (isinstance(co_vp, (LocalInconclusive, LocalExhaustion))
                                                        and co_vp.group == failure.group
                                                        and p.net != co_vp.net
                                                        and exact_witness(p, co_vp)):
                                                    vp_owners.add(p.net)
                                    vp_causal = [
                                        vp_frames[i] for i in range(g_start, len(vp_frames))
                                        if vp_frames[i].port.net in vp_owners
                                    ]
                                    if vp_causal:
                                        vp_pre = {}
                                        for j in range(vp_k + 1):
                                            pfx = tuple(p.net for p in target_order[:j])
                                            if pfx in cascade_prefixes:
                                                vp_pre[g_start + j] = cascade_prefixes[pfx]
                                        vp_best = (
                                            vp_prog, list(target_order), vp_pre,
                                            vp_frames, vp_world, vp_rej, vp_causal,
                                        )
                            record = getattr(adapter, 'record_outcome', None)
                            if record is not None:
                                record(victim_port, frames[g_start].key, rejected)
                            starved_batch = [rejected, *getattr(rejected, 'co_starved', ())]
                            first_starved_net = None
                            if type(rejected) is _BranchLandingUnavailable:
                                first_starved_net = getattr(rejected.original, 'net', None)
                            elif type(rejected) is GroupRouteBudgetStop:
                                first_starved_net = rejected.flow_error_data().get('net')
                            elif isinstance(rejected, (LocalInconclusive, LocalExhaustion)):
                                first_starved_net = rejected.net

                            if (first_starved_net is not None
                                    and isinstance(rejected, (LocalInconclusive, LocalExhaustion))):
                                direct_owners = _rej_owner_set(rejected)
                                if first_starved_net in by_net and direct_owners:
                                    vp_ever_blockers.setdefault(first_starved_net, set()).update(
                                        direct_owners
                                    )
                                if len(direct_owners) == 1:
                                    sole_owner = next(iter(direct_owners))
                                    solo_blocks.add((sole_owner, first_starved_net))
                                    vp_solo_blocks.add((sole_owner, first_starved_net))
                                    if ((first_starved_net, sole_owner) in solo_blocks
                                            and victim_port.net not in (sole_owner, first_starved_net)):
                                        break
                            for co_st in getattr(rejected, 'co_starved', ()):
                                if (isinstance(co_st, (LocalInconclusive, LocalExhaustion))
                                        and getattr(co_st, 'net', None) in by_net):
                                    co_dir_owners = _rej_owner_set(co_st)
                                    if co_dir_owners:
                                        vp_ever_blockers.setdefault(co_st.net, set()).update(
                                            co_dir_owners
                                        )
                                    if len(co_dir_owners) == 1:
                                        co_sole = next(iter(co_dir_owners))
                                        solo_blocks.add((co_sole, co_st.net))
                                        vp_solo_blocks.add((co_sole, co_st.net))

                            if first_starved_net in front_nets:
                                moved_existing = False
                                if (first_starved_net != victim_port.net
                                        and isinstance(rejected, (LocalInconclusive, LocalExhaustion))):
                                    rej_owners = _rej_owner_set(rejected)
                                    if victim_port.net not in rej_owners:
                                        cur_slot = slot_by_net[first_starved_net]
                                        earlier_orig = min(
                                            (
                                                orig_index[o]
                                                for o in rej_owners
                                                if o not in front_nets and orig_index[o] < cur_slot
                                            ),
                                            default=None,
                                        )
                                        helper_idx = next(
                                            (
                                                idx_p
                                                for idx_p, hp in enumerate(promoted_ports)
                                                if hp.net in rej_owners and hp.net != victim_port.net
                                            ),
                                            None,
                                        )
                                        cand_slots_map = dict(slot_by_net)
                                        cand_prom = [
                                            hp for hp in promoted_ports if hp.net != first_starved_net
                                        ]
                                        if helper_idx is not None:
                                            target_h = promoted_ports[helper_idx].net
                                            new_slot = cand_slots_map[target_h]
                                            if earlier_orig is not None and earlier_orig < new_slot:
                                                new_slot = earlier_orig
                                            cand_slots_map[first_starved_net] = new_slot
                                            ins_i = next(
                                                i_c
                                                for i_c, hp in enumerate(cand_prom)
                                                if hp.net == target_h
                                            )
                                            cand_prom.insert(ins_i, by_net[first_starved_net])
                                        elif earlier_orig is not None:
                                            cand_slots_map[first_starved_net] = earlier_orig
                                            cand_prom.append(by_net[first_starved_net])
                                        next_order = _build_vp_order(cand_prom, cand_slots_map)
                                        if tuple(p.net for p in next_order) not in seen_vp_orders:
                                            promoted_ports = cand_prom
                                            slot_by_net = cand_slots_map
                                            moved_existing = True
                                if moved_existing:
                                    continue
                                if slot_idx + 1 < len(cand_slots):
                                    slot_idx += 1
                                    insert_pos = cand_slots[slot_idx]
                                    promoted_ports = [victim_port]
                                    slot_by_net = {victim_port.net: insert_pos}
                                    continue
                                break

                            cand_prom = list(promoted_ports)
                            cand_slots_map = dict(slot_by_net)
                            cur_front = set(front_nets)
                            added_any = False
                            for st_exc in starved_batch:
                                if type(st_exc) is _BranchLandingUnavailable:
                                    st_net = getattr(st_exc.original, 'net', None)
                                elif type(st_exc) is GroupRouteBudgetStop:
                                    st_net = st_exc.flow_error_data().get('net')
                                elif isinstance(st_exc, (LocalInconclusive, LocalExhaustion)):
                                    st_net = st_exc.net
                                else:
                                    st_net = None
                                if (st_net not in by_net
                                        or st_net in cur_front
                                        or len(cand_prom) - 1 >= max_cascade):
                                    continue
                                rej_owners = _rej_owner_set(st_exc)
                                if (victim_already_closed
                                        and vp_best is not None
                                        and isinstance(st_exc, (LocalInconclusive, LocalExhaustion))
                                        and orig_index.get(st_net, -1) > victim_rel
                                        and rej_owners
                                        and not (rej_owners & cur_front)):
                                    continue
                                helper_idx = next(
                                    (
                                        idx_p
                                        for idx_p, hp in enumerate(cand_prom)
                                        if hp.net in rej_owners and hp.net != victim_port.net
                                    ),
                                    None,
                                )
                                earlier_prefix_pos = next(
                                    (
                                        i
                                        for i, p in enumerate(saved_slice[:insert_pos])
                                        if p.net in rej_owners and p.net not in cur_front
                                    ),
                                    None,
                                )
                                if helper_idx is not None and victim_port.net not in rej_owners:
                                    req_slot = cand_slots_map[cand_prom[helper_idx].net]
                                    if earlier_prefix_pos is not None and earlier_prefix_pos < req_slot:
                                        req_slot = earlier_prefix_pos
                                    cand_slots_map[st_net] = req_slot
                                    cand_prom.insert(helper_idx, by_net[st_net])
                                else:
                                    req_slot = (
                                        earlier_prefix_pos
                                        if earlier_prefix_pos is not None
                                        else insert_pos
                                    )
                                    cand_slots_map[st_net] = req_slot
                                    cand_prom.append(by_net[st_net])
                                cur_front.add(st_net)
                                added_any = True
                            if added_any:
                                next_order = _build_vp_order(cand_prom, cand_slots_map)
                                if tuple(p.net for p in next_order) not in seen_vp_orders:
                                    promoted_ports = cand_prom
                                    slot_by_net = cand_slots_map
                                    continue
                            if not (victim_already_closed and vp_best is not None) and slot_idx + 1 < len(cand_slots):
                                slot_idx += 1
                                insert_pos = cand_slots[slot_idx]
                                promoted_ports = [victim_port]
                                slot_by_net = {victim_port.net: insert_pos}
                                continue
                            break
                    ports[g_start:g_end] = saved_slice
                except BaseException:
                    ports[g_start:g_end] = saved_slice
                    raise
                finally:
                    walk_prefix_sink = prev_sink

        if not causal_frames and vp_best is None:
            reset_history()
            return None
        deposit = getattr(adapter, 'deposit_causal_congestion', None)
        if deposit is None or apply_negotiated_history is None:
            reset_history()
            return None
        if prepare_neg is not None:
            prepare_neg(failure.group, tuple(ports[g_start:g_end]), clean_snap)
        group_history = {}
        saved_neg_slice = list(ports[g_start:g_end])
        neg_by_net = {p.net: p for p in saved_neg_slice}
        start_snap = frames[g_start].snapshot
        group_ports = tuple(saved_neg_slice)
        seeded_from_vp = False
        if vp_best is not None:
            (vp_prog, vp_order, vp_pre, vp_frames,
             vp_world, vp_rej, vp_causal) = vp_best
            vp_victim = neg_by_net.get(vp_rej.net, vp_causal[0].port)
            adapter.restore(vp_world)
            if deposit(group_ports, vp_victim, vp_rej, group_history, 1):
                for co_rej in getattr(vp_rej, 'co_starved', ()):
                    co_victim = neg_by_net.get(co_rej.net)
                    if co_victim is not None:
                        deposit(group_ports, co_victim, co_rej, group_history, 1)
                seeded_from_vp = True
                ports[g_start:g_end] = list(vp_order)
                best_order = list(vp_order)
                best_pre = dict(vp_pre)
                best_pre.setdefault(g_start, (start_snap, tuple(frames[:g_start])))
                best_frames = list(vp_frames)
                best_progress = vp_prog
                next_run_idx = min(f.index for f in vp_causal)
                frontier_blockers_by_net = {
                    vp_rej.net: sorted({f.index for f in vp_causal if g_start <= f.index < g_end})
                }
        if not seeded_from_vp:
            if not causal_frames:
                reset_history()
                return None
            victim_obj = next(
                (p for p in group_ports if p.net == failure.net),
                causal_frames[0].port,
            )
            adapter.restore(failed_world)
            if not deposit(group_ports, victim_obj, failure, group_history, 1):
                reset_history()
                return None
            best_order = list(saved_neg_slice)
            best_pre = {
                i: (frames[i].snapshot, tuple(frames[:i]))
                for i in range(g_start, len(frames))
            }
            best_pre.setdefault(g_start, (start_snap, tuple(frames[:g_start])))
            best_frames = list(frames)
            best_progress = len(frames)
            next_run_idx = min(f.index for f in causal_frames)
            frontier_blockers_by_net = {
                failure.net: sorted({f.index for f in causal_frames if g_start <= f.index < g_end})
            }

        best_history = {}
        best_frontier_history = dict(group_history)
        current_pre = dict(best_pre)
        seen_orders = {
            tuple(p.net for p in saved_neg_slice),
            tuple(p.net for p in best_order),
        }
        fail_counts_by_order = {}
        pos_fail_counts_by_order = {}
        pair_solo_fails = {}
        ever_blockers_by_net = {}

        def _committed_frame_nets_for(frames_seq, rej_obj):
            if not isinstance(rej_obj, (LocalInconclusive, LocalExhaustion)):
                return set()
            r_owners = set(rej_obj.committed_owners)
            if exact_witness is not None:
                for p in group_ports:
                    if p.net != rej_obj.net and exact_witness(p, rej_obj):
                        r_owners.add(p.net)
            frame_nets = {
                frames_seq[i].port.net
                for i in range(g_start, len(frames_seq))
            }
            return {o for o in (r_owners & frame_nets) if o != rej_obj.net}

        init_causal_nets = {f.port.net for f in causal_frames if f.port.net != failure.net}
        if init_causal_nets:
            ever_blockers_by_net.setdefault(failure.net, set()).update(init_causal_nets)
            if len(init_causal_nets) == 1:
                solo_init = next(iter(init_causal_nets))
                pair_solo_fails[(solo_init, failure.net)] = (
                    pair_solo_fails.get((solo_init, failure.net), 0) + 1
                )
        for co_init in getattr(failure, 'co_starved', ()):
            if (getattr(co_init, 'group', None) != failure.group
                    or getattr(co_init, 'net', None) not in neg_by_net):
                continue
            co_init_owners = _committed_frame_nets_for(frames, co_init)
            if co_init_owners:
                ever_blockers_by_net.setdefault(co_init.net, set()).update(
                    co_init_owners
                )
            if len(co_init_owners) == 1:
                solo_o = next(iter(co_init_owners))
                pair_solo_fails[(solo_o, co_init.net)] = (
                    pair_solo_fails.get((solo_o, co_init.net), 0) + 1
                )
        for vp_victim_net, vp_owners_set in vp_ever_blockers.items():
            ever_blockers_by_net.setdefault(vp_victim_net, set()).update(vp_owners_set)
        for vp_pair in vp_solo_blocks:
            pair_solo_fails[vp_pair] = max(1, pair_solo_fails.get(vp_pair, 0))

        if seeded_from_vp and vp_best is not None:
            vp_order_key = tuple(p.net for p in best_order)
            init_order_fails = fail_counts_by_order.setdefault(vp_order_key, {})
            init_pos_fails = pos_fail_counts_by_order.setdefault(vp_order_key, {})
            vp_fail_idx = min(max(g_start, vp_best[0]), g_end - 1)
            for seed_rej in (vp_best[5], *getattr(vp_best[5], 'co_starved', ())):
                if (getattr(seed_rej, 'group', None) != failure.group
                        or getattr(seed_rej, 'net', None) not in neg_by_net):
                    continue
                init_order_fails[seed_rej.net] = (
                    init_order_fails.get(seed_rej.net, 0) + 1
                )
                init_pos_fails[(vp_fail_idx, seed_rej.net)] = (
                    init_pos_fails.get((vp_fail_idx, seed_rej.net), 0) + 1
                )
                seed_owners = _committed_frame_nets_for(vp_best[3], seed_rej)
                if seed_rej is vp_best[5]:
                    seed_owners |= {
                        f.port.net for f in vp_best[6] if f.port.net != seed_rej.net
                    }
                if seed_owners:
                    ever_blockers_by_net.setdefault(seed_rej.net, set()).update(
                        seed_owners
                    )
                    seed_blk_idxs = [
                        i for i in range(g_start, len(vp_best[3]))
                        if vp_best[3][i].port.net in seed_owners
                    ]
                    if seed_blk_idxs:
                        frontier_blockers_by_net.setdefault(
                            seed_rej.net, seed_blk_idxs
                        )
                if len(seed_owners) == 1:
                    solo_o = next(iter(seed_owners))
                    pair_solo_fails[(solo_o, seed_rej.net)] = (
                        pair_solo_fails.get((solo_o, seed_rej.net), 0) + 1
                    )

        def _is_mutually_blocked(net_a, net_b):
            return (
                net_b in ever_blockers_by_net.get(net_a, set())
                and net_a in ever_blockers_by_net.get(net_b, set())
            )

        def _try_promote_in_slice(
            base_slice,
            base_pre,
            target_net,
            blocker_idxs,
            allow_zero_fallback=True,
            prefer_max_blocker=False,
        ):
            cur_pos = next((i for i, p in enumerate(base_slice) if p.net == target_net), None)
            if cur_pos is None or cur_pos <= 0:
                return None
            t_port = base_slice[cur_pos]
            rel_blockers = [idx - g_start for idx in blocker_idxs if 0 <= idx - g_start < cur_pos]
            slots = []
            if rel_blockers:
                lo_b, hi_b = min(rel_blockers), max(rel_blockers)
                if prefer_max_blocker and hi_b != lo_b:
                    slots.extend((hi_b, lo_b))
                else:
                    slots.append(lo_b)
                    if hi_b != lo_b:
                        slots.append(hi_b)
            if not rel_blockers and cur_pos - 1 not in slots:
                slots.append(cur_pos - 1)
            if allow_zero_fallback and 0 not in slots:
                slots.append(0)
            for slot in slots:
                if not (0 <= slot < cur_pos):
                    continue
                if any(
                    pair_solo_fails.get((target_net, p.net), 0) >= 1
                    for p in base_slice[slot:cur_pos]
                ):
                    continue
                cand_slice = (
                    base_slice[:slot]
                    + [t_port]
                    + base_slice[slot:cur_pos]
                    + base_slice[cur_pos + 1:]
                )
                cand_key = tuple(p.net for p in cand_slice)
                if cand_key in seen_orders:
                    continue
                seen_orders.add(cand_key)
                resume_at = g_start + slot
                while resume_at > g_start and resume_at not in base_pre:
                    resume_at -= 1
                kept_pre = {k: v for k, v in base_pre.items() if k <= resume_at}
                return cand_slice, kept_pre, resume_at
            return None

        max_neg_rounds = max(14, 3 * len(saved_neg_slice))
        try:
            for neg_round in range(1, max_neg_rounds + 1):
                run_idx = max(g_start, next_run_idx)
                while run_idx > g_start and run_idx not in current_pre:
                    run_idx -= 1
                run_snap, run_frames = current_pre.get(
                    run_idx, (start_snap, tuple(frames[:g_start]))
                )
                run_port = ports[run_idx]
                adapter.restore(run_snap)
                apply_negotiated_history(base_history, group_history)
                adapter.checkpoint(run_port, 'group_fallback_negotiated_congestion')
                trial_pre = {k: v for k, v in current_pre.items() if k <= run_idx}
                trial_pre[run_idx] = (run_snap, run_frames)
                prev_neg_hist = active_neg_group_history
                prev_trial_sink = neg_trial_sink
                active_neg_group_history = group_history
                neg_trial_sink = (g_start, g_end, trial_pre)
                try:
                    outcome = walk(
                        run_idx,
                        run_frames,
                        (*(g for g in guards if g[0] < g_start),
                         (g_start, start_snap, ports[g_start])),
                    )
                    record_round = getattr(adapter, 'record_negotiation_round', None)
                    if record_round is not None:
                        record_round({
                            'group': failure.group,
                            'round': neg_round + 1,
                            'status': 'closed',
                            'progress': g_end - g_start,
                            'slice_size': g_end - g_start,
                            'order': list(p.net for p in ports[g_start:g_end]),
                            'history_cells': len(group_history),
                        })
                    reset_history()
                    return outcome
                except (LocalInconclusive, LocalExhaustion, _PhysicalRejection,
                        _BranchLandingUnavailable, GroupRouteBudgetStop) as raw_rejected:
                    remember_recovery_budget(raw_rejected)
                    max_pre_idx = max(trial_pre.keys()) if trial_pre else run_idx
                    trial_frames = list(trial_pre[max_pre_idx][1]) if max_pre_idx in trial_pre else list(run_frames)
                    if type(raw_rejected) is GroupRouteBudgetStop:
                        if not _is_recoverable_edge_budget(raw_rejected):
                            reset_history()
                            raise
                        fail_port_idx = min(max(g_start, len(trial_frames)), g_end - 1)
                        converted_rej = (
                            recoverable_budget(
                                ports[fail_port_idx], raw_rejected, tuple(trial_frames)
                            )
                            if recoverable_budget is not None
                            else None
                        )
                        rejected = converted_rej if converted_rej is not None else raw_rejected
                    else:
                        rejected = raw_rejected

                    order_key = tuple(p.net for p in ports[g_start:g_end])
                    best_key = tuple(p.net for p in best_order)
                    if type(rejected) is _PhysicalRejection:
                        reset_history()
                        if order_key != best_key:
                            ports[g_start:g_end] = list(best_order)
                            current_pre = dict(best_pre)
                            frontier_port = best_order[min(best_progress - g_start, len(best_order) - 1)]
                            fb_idxs = frontier_blockers_by_net.get(frontier_port.net, [])
                            promoted_step = _try_promote_in_slice(
                                best_order, best_pre, frontier_port.net, fb_idxs
                            )
                            if promoted_step is not None:
                                ports[g_start:g_end], current_pre, next_run_idx = promoted_step
                                continue
                        break

                    if type(rejected) is _BranchLandingUnavailable:
                        rej_group = getattr(rejected.original, 'group', None)
                        rej_net = getattr(rejected.original, 'net', None)
                    elif type(rejected) is GroupRouteBudgetStop:
                        diag = rejected.flow_error_data()
                        rej_group = diag.get('group')
                        rej_net = diag.get('net')
                    else:
                        rej_group = rejected.group
                        rej_net = rejected.net
                    if rej_group != failure.group or rej_net not in neg_by_net:
                        reset_history()
                        break

                    rej_idx = min(max(g_start, len(trial_frames)), g_end - 1)
                    owners = (
                        set(rejected.committed_owners)
                        if isinstance(rejected, (LocalInconclusive, LocalExhaustion))
                        else set()
                    )
                    if (exact_witness is not None
                            and isinstance(rejected, (LocalInconclusive, LocalExhaustion))):
                        for p in group_ports:
                            if p.net != rej_net and exact_witness(p, rejected):
                                owners.add(p.net)
                    blocker_indices = [
                        i for i in range(g_start, min(rej_idx, len(trial_frames)))
                        if trial_frames[i].port.net in owners
                    ]
                    if blocker_indices:
                        frontier_blockers_by_net[rej_net] = list(blocker_indices)

                    victim_obj = neg_by_net[rej_net]
                    trial_history = dict(group_history)
                    deposit(group_ports, victim_obj, rejected, group_history, neg_round + 1)
                    for co_rej in getattr(rejected, 'co_starved', ()):
                        co_victim = neg_by_net.get(co_rej.net)
                        if co_victim is not None:
                            deposit(group_ports, co_victim, co_rej, group_history, neg_round + 1)
                    reset_history()

                    record = getattr(adapter, 'record_outcome', None)
                    if record is not None and order_key == tuple(p.net for p in saved_neg_slice):
                        rec_idx = min(run_idx, len(frames) - 1)
                        if rec_idx >= g_start:
                            record(frames[rec_idx].port, frames[rec_idx].key, raw_rejected)

                    if rej_idx > best_progress:
                        best_progress = rej_idx
                        best_order = list(ports[g_start:g_end])
                        best_pre = dict(trial_pre)
                        best_frames = list(trial_frames)
                        best_key = tuple(p.net for p in best_order)
                        best_history = trial_history
                        best_frontier_history = dict(group_history)
                    elif order_key == best_key and rej_idx == best_progress:
                        best_frontier_history = dict(group_history)

                    order_fails = fail_counts_by_order.setdefault(order_key, {})
                    net_fails = order_fails.get(rej_net, 0) + 1
                    order_fails[rej_net] = net_fails
                    order_total_fails = sum(order_fails.values())
                    order_pos_fails = pos_fail_counts_by_order.setdefault(order_key, {})
                    pos_fails = order_pos_fails.get((rej_idx, rej_net), 0) + 1
                    order_pos_fails[(rej_idx, rej_net)] = pos_fails
                    raw_det = getattr(rejected, 'details', ()) or ()
                    is_starved_watch = False
                    if isinstance(raw_det, dict):
                        is_starved_watch = bool(raw_det.get('sealed_pocket_escape_starved', False))
                    elif isinstance(raw_det, (tuple, list)):
                        is_starved_watch = any(
                            isinstance(kv, tuple)
                            and len(kv) == 2
                            and kv[0] == 'sealed_pocket_escape_starved'
                            and bool(kv[1])
                            for kv in raw_det
                        )
                    if owners:
                        ever_blockers_by_net.setdefault(rej_net, set()).update(
                            o for o in owners if o != rej_net
                        )
                    if len(owners) == 1:
                        solo_blk = next(iter(owners))
                        if solo_blk != rej_net:
                            pair_solo_fails[(solo_blk, rej_net)] = (
                                pair_solo_fails.get((solo_blk, rej_net), 0) + 1
                            )
                    for co_rej in getattr(rejected, 'co_starved', ()):
                        if (getattr(co_rej, 'group', None) != failure.group
                                or getattr(co_rej, 'net', None) not in neg_by_net):
                            continue
                        order_fails[co_rej.net] = order_fails.get(co_rej.net, 0) + 1
                        co_owners = _committed_frame_nets_for(
                            trial_frames[:rej_idx], co_rej
                        )
                        if co_owners:
                            ever_blockers_by_net.setdefault(co_rej.net, set()).update(
                                co_owners
                            )
                            co_blk_idxs = [
                                i for i in range(g_start, min(rej_idx, len(trial_frames)))
                                if trial_frames[i].port.net in co_owners
                            ]
                            if co_blk_idxs:
                                frontier_blockers_by_net[co_rej.net] = co_blk_idxs
                        if len(co_owners) == 1:
                            co_solo = next(iter(co_owners))
                            pair_solo_fails[(co_solo, co_rej.net)] = (
                                pair_solo_fails.get((co_solo, co_rej.net), 0) + 1
                            )
                    order_total_fails = sum(order_fails.values())
                    record_round = getattr(adapter, 'record_negotiation_round', None)
                    if record_round is not None:
                        ef_raw = getattr(rejected, 'edge_fail', None)
                        record_round({
                            'group': failure.group,
                            'round': neg_round + 1,
                            'status': 'rejected',
                            'failed_net': rej_net,
                            'reason': type(rejected).__name__,
                            'edge_fail': ef_raw[:3] if type(ef_raw) is tuple else ef_raw,
                            'progress': rej_idx - g_start,
                            'best_progress': best_progress - g_start,
                            'slice_size': g_end - g_start,
                            'owners': sorted(owners),
                            'order': list(order_key),
                            'history_cells': len(group_history),
                        })

                    # Same-order regression: congestion added for a downstream
                    # frontier broke a prefix port that had closed under the
                    # best history.  Discard that speculative history and give
                    # the frontier port an earlier slot instead of letting the
                    # deposits keep compounding onto the closed prefix.
                    if order_key == best_key and rej_idx < best_progress:
                        frontier_idx = min(best_progress - g_start, len(best_order) - 1)
                        frontier_net = best_order[frontier_idx].net
                        promoted_step = _try_promote_in_slice(
                            best_order, best_pre, frontier_net,
                            frontier_blockers_by_net.get(frontier_net, []),
                        )
                        if promoted_step is not None:
                            group_history.clear()
                            group_history.update(best_history)
                            ports[g_start:g_end], current_pre, next_run_idx = promoted_step
                            continue

                    # Anti-regression: if a speculative reordering stalled earlier than
                    # best_progress even after congestion feedback, try the next
                    # untried promotion from best_order if one remains.
                    if (order_key != best_key
                            and rej_idx < best_progress
                            and (order_total_fails >= 5 or not blocker_indices)):
                        frontier_idx = min(best_progress - g_start, len(best_order) - 1)
                        frontier_net = best_order[frontier_idx].net
                        fb_idxs = frontier_blockers_by_net.get(frontier_net, [])
                        promoted_step = _try_promote_in_slice(
                            best_order, best_pre, frontier_net, fb_idxs
                        )
                        if promoted_step is not None:
                            group_history.clear()
                            group_history.update(best_frontier_history)
                            ports[g_start:g_end], current_pre, next_run_idx = promoted_step
                            continue
                        best_net_fails = fail_counts_by_order.get(best_key, {}).get(frontier_net, 0)
                        if fb_idxs and best_net_fails <= 8 and group_history != best_frontier_history:
                            group_history.clear()
                            group_history.update(best_frontier_history)
                            ports[g_start:g_end] = list(best_order)
                            current_pre = dict(best_pre)
                            next_run_idx = min(fb_idxs)
                            continue

                    # Coupled pocket pair promotion: when a downstream port has
                    # mutual blocking history with an earlier committed
                    # blocker whose prefix remained unstarved until an
                    # intermediate port committed, place the coupled port
                    # directly after that partner before intermediate ports.
                    if (
                        blocker_indices
                        and is_starved_watch
                        and rej_idx > g_start + 1
                        and (rej_idx - 1) in blocker_indices
                        and min(blocker_indices) < rej_idx - 1
                    ):
                        coupled_partner_idxs = [
                            b_idx
                            for b_idx in blocker_indices
                            if b_idx < rej_idx - 1
                            and _is_mutually_blocked(ports[b_idx].net, rej_net)
                        ]
                        if coupled_partner_idxs:
                            promoted_coupled = _try_promote_in_slice(
                                list(ports[g_start:g_end]),
                                trial_pre,
                                rej_net,
                                [max(coupled_partner_idxs) + 1],
                                allow_zero_fallback=False,
                            )
                            if promoted_coupled is not None:
                                ports[g_start:g_end], current_pre, next_run_idx = (
                                    promoted_coupled
                                )
                                continue

                    # Incremental starvation trigger: when the starvation watch
                    # passed at rej_idx - 1 and tripped right after ports[rej_idx - 1]
                    # committed, keep the proven non-starving prefix frames[:rej_idx - 1]
                    # and rerun the newly committed blocker at rej_idx - 1 first.
                    if (
                        blocker_indices
                        and is_starved_watch
                        and rej_idx > g_start + 1
                        and (rej_idx - 1) in blocker_indices
                        and min(blocker_indices) < rej_idx - 1
                        and pos_fails <= 1
                        and group_history != trial_history
                    ):
                        current_pre = dict(trial_pre)
                        next_run_idx = rej_idx - 1
                        continue

                    # Congestion-first on initial failures of this net in this order;
                    # on repeated failure, let pre-blocker promotion try an earlier slot first.
                    if blocker_indices and net_fails <= 1 and group_history != trial_history:
                        current_pre = dict(trial_pre)
                        next_run_idx = min(blocker_indices)
                        continue

                    # Adjacent mutual-starvation block promotion: when two ports
                    # have mutual blocking history and still starve behind
                    # an earlier prefix slice even after congestion feedback,
                    # promote the 2-port coupled block ahead of their combined
                    # historical prefix blockers.
                    if (
                        rej_idx > g_start + 1
                        and blocker_indices == [rej_idx - 1]
                        and _is_mutually_blocked(ports[rej_idx - 1].net, rej_net)
                    ):
                        adj_net = ports[rej_idx - 1].net
                        cur_slice_blk = list(ports[g_start:g_end])
                        adj_rel = rej_idx - 1 - g_start
                        rej_rel = next(
                            (i_s for i_s, p_s in enumerate(cur_slice_blk) if p_s.net == rej_net),
                            None,
                        )
                        if rej_rel is not None and adj_rel < rej_rel:
                            pair_hist_blks = (
                                ever_blockers_by_net.get(adj_net, set())
                                | ever_blockers_by_net.get(rej_net, set())
                            ) - {adj_net, rej_net}
                            rel_hist_idxs = [
                                i_s for i_s in range(adj_rel)
                                if cur_slice_blk[i_s].net in pair_hist_blks
                            ]
                            cand_block_slots = []
                            if rel_hist_idxs:
                                cand_block_slots.append(min(rel_hist_idxs))
                            if 0 not in cand_block_slots:
                                cand_block_slots.append(0)
                            rest_without_pair = [
                                p_s for p_s in cur_slice_blk
                                if p_s.net not in (adj_net, rej_net)
                            ]
                            adj_port_obj = cur_slice_blk[adj_rel]
                            rej_port_obj = cur_slice_blk[rej_rel]
                            promoted_pair = None
                            for b_slot in cand_block_slots:
                                if not (0 <= b_slot < adj_rel):
                                    continue
                                for pair_seq in (
                                    [adj_port_obj, rej_port_obj],
                                    [rej_port_obj, adj_port_obj],
                                ):
                                    cand_pair_slice = (
                                        rest_without_pair[:b_slot]
                                        + pair_seq
                                        + rest_without_pair[b_slot:]
                                    )
                                    cand_pair_key = tuple(p_s.net for p_s in cand_pair_slice)
                                    if cand_pair_key in seen_orders:
                                        continue
                                    seen_orders.add(cand_pair_key)
                                    res_at = g_start + b_slot
                                    while res_at > g_start and res_at not in trial_pre:
                                        res_at -= 1
                                    kept_pair_pre = {
                                        k: v for k, v in trial_pre.items() if k <= res_at
                                    }
                                    promoted_pair = (cand_pair_slice, kept_pair_pre, res_at)
                                    break
                                if promoted_pair is not None:
                                    break
                            if promoted_pair is not None:
                                ports[g_start:g_end], current_pre, next_run_idx = promoted_pair
                                continue

                    # Targeted pre-blocker promotion in current slice
                    cur_slice = list(ports[g_start:g_end])
                    hist_rej_blks = ever_blockers_by_net.get(rej_net, set())
                    eff_blocker_indices = sorted(
                        set(blocker_indices)
                        | {
                            g_start + idx_s
                            for idx_s, p_s in enumerate(cur_slice)
                            if p_s.net in hist_rej_blks
                        }
                    )
                    prefer_trigger_slot = bool(
                        is_starved_watch
                        and blocker_indices
                        and (rej_idx - 1) in blocker_indices
                        and min(blocker_indices) < rej_idx - 1
                        and (rej_idx - 1 - min(blocker_indices)) >= 4
                    )
                    promoted_step = _try_promote_in_slice(
                        cur_slice,
                        trial_pre,
                        rej_net,
                        eff_blocker_indices,
                        allow_zero_fallback=True,
                        prefer_max_blocker=prefer_trigger_slot,
                    )
                    if promoted_step is None and owners:
                        for blk_net in sorted(owners):
                            if pair_solo_fails.get((rej_net, blk_net), 0) >= 1:
                                hist_blks = ever_blockers_by_net.get(blk_net, set())
                                blk_anc_idxs = [
                                    g_start + idx_s
                                    for idx_s, p_s in enumerate(cur_slice)
                                    if p_s.net in hist_blks
                                ]
                                promoted_step = _try_promote_in_slice(
                                    cur_slice, trial_pre, blk_net, blk_anc_idxs
                                )
                                if promoted_step is not None:
                                    break
                    if promoted_step is not None:
                        ports[g_start:g_end], current_pre, next_run_idx = promoted_step
                        continue

                    if blocker_indices and group_history != trial_history and net_fails <= 3:
                        current_pre = dict(trial_pre)
                        next_run_idx = min(blocker_indices)
                        continue
                    break
                finally:
                    active_neg_group_history = prev_neg_hist
                    neg_trial_sink = prev_trial_sink
            ports[g_start:g_end] = saved_neg_slice
        except BaseException:
            ports[g_start:g_end] = saved_neg_slice
            raise
        reset_history()
        return None

    def walk(index, frames, guards, forced=None, prepared=None, avoidance=None):
        nonlocal first_failure
        while index < len(ports):
            port = ports[index]
            adapter.checkpoint(port, 'group_fallback_port')
            snapshot = adapter.snapshot() if prepared is None else prepared[0]
            if (walk_prefix_sink is not None
                    and avoidance is None and forced is None):
                sink_start, sink_end, sink_map = walk_prefix_sink
                if sink_start <= index < sink_end and len(frames) == index:
                    key_prefix = tuple(f.port.net for f in frames[sink_start:index])
                    if key_prefix not in sink_map:
                        sink_map[key_prefix] = (snapshot, frames)
            if (neg_trial_sink is not None
                    and avoidance is None and forced is None):
                ts_start, ts_end, ts_pre = neg_trial_sink
                if ts_start <= index < ts_end and len(frames) == index:
                    ts_pre[index] = (snapshot, frames)
            starved_fn = getattr(adapter, 'first_starved_remaining_port', None)
            if (starved_fn is not None
                    and guards
                    and avoidance is None
                    and forced is None
                    and any(origin.group == port.group and guard_index < index
                            for guard_index, _snapshot, origin in guards)):
                g_end_here = group_spans[port.group][1]
                starved = starved_fn(ports[index:g_end_here], frames)
                if starved is not None:
                    raise starved
            try:
                rows = adapter.candidates(port) if prepared is None else prepared[1]
                prepared = None
            except GroupRouteLandingUnavailable as error:
                if (type(error) is GroupRouteLandingUnavailable
                    and type(error.group) is str and type(error.net) is str
                    and error.group == port.group and error.net == port.net
                    and any(origin.group == port.group and guard_index < index
                            for guard_index, _snapshot, origin in guards)
                    and type(error.via_redundancy) is int and error.via_redundancy > 0
                    and error.via_redundancy == adapter.via_redundancy()):
                    # This landing domain was empty only in a speculative world;
                    # retain the typed producer error as this branch's evidence.
                    raise _BranchLandingUnavailable(error) from error
                raise
            adapter.checkpoint(port, 'group_fallback_candidate_domain_complete')
            was_unforced = (forced is None and avoidance is None)
            selected = (adapter.default_candidate(port, rows) if forced is None
                        else forced)
            if forced is not None:
                forced_key = adapter.candidate_key(port, forced)
                matches = [row for row in rows if adapter.candidate_key(port, row) == forced_key]
                if len(matches) != 1:
                    raise ValueError('regenerated candidate domain changed before replay')
                selected = matches[0]
            forced = None
            frame = _Frame(index, port, snapshot, selected,
                           adapter.candidate_key(port, selected))

            def _try_negotiated_alternates(initial_failure):
                nonlocal selected, frame
                alt_fn = getattr(adapter, 'negotiated_alternate_candidates', None)
                if (not guards or active_neg_group_history is None or not was_unforced
                        or alt_fn is None or len(rows) <= 1):
                    return False
                for alt_row in alt_fn(port, rows, selected, limit=2):
                    adapter.restore(snapshot)
                    if apply_negotiated_history is not None:
                        apply_negotiated_history(base_history, active_neg_group_history)
                    adapter.checkpoint(port, 'group_fallback_candidate')
                    alt_key = adapter.candidate_key(port, alt_row)
                    alt_frame = _Frame(index, port, snapshot, alt_row, alt_key)
                    try:
                        adapter.route(port, alt_row)
                        selected = alt_row
                        frame = alt_frame
                        return True
                    except GroupRouteBudgetStop as alt_budget:
                        alt_conv = (
                            None if recoverable_budget is None
                            else recoverable_budget(port, alt_budget, frames)
                        )
                        if alt_conv is None:
                            raise
                        remember_recovery_budget(alt_budget)
                        record = getattr(adapter, 'record_outcome', None)
                        if record is not None:
                            record(port, alt_key, alt_conv)
                    except (LocalInconclusive, LocalExhaustion) as alt_rej:
                        remember_recovery_budget(alt_rej)
                        record = getattr(adapter, 'record_outcome', None)
                        if record is not None:
                            record(port, alt_key, alt_rej)
                return False

            try:
                if avoidance is None:
                    adapter.route(port, selected)
                else:
                    adapter.route_avoiding(port, selected, avoidance)
                    avoidance = None
            except GroupRouteBudgetStop as budget_exc:
                converted = (
                    None
                    if recoverable_budget is None
                    else recoverable_budget(port, budget_exc, frames)
                )
                if converted is None:
                    raise
                if not _try_negotiated_alternates(converted):
                    if guards:
                        raise converted
                    return recover_current(converted, frames, frame)
            except LocalInconclusive as failure:
                if failure.group != port.group or failure.net != port.net:
                    raise ValueError('local inconclusive result does not bind current port')
                if not _try_negotiated_alternates(failure):
                    if guards:
                        raise failure
                    return recover_current(failure, frames, frame)
            except LocalExhaustion as failure:
                if failure.group != port.group or failure.net != port.net:
                    raise ValueError('local exhaustion does not bind current port')
                if first_failure is None:
                    first_failure = failure
                if guards:
                    # A trial failure returns to its owning fair branch loop.
                    # It must not recursively exhaust one prior owner first.
                    raise failure
                return recover(failure, frames, guards)
            frames = (*frames, frame)
            index += 1
            if index == len(ports) or ports[index].group != port.group:
                pending = []
                checked = False
                for guard_index, prior, origin in guards:
                    if origin.group != port.group:
                        pending.append((guard_index, prior, origin))
                        continue
                    adapter.checkpoint(origin, 'group_fallback_physical_feedback')
                    accepted = adapter.physical_nonworsening(prior)
                    adapter.checkpoint(origin, 'group_fallback_physical_feedback_complete')
                    if not accepted:
                        raise _PhysicalRejection()
                    checked = True
                reset_history()
                guards = tuple(pending)
                if checked:
                    # Exit every nested retry scope before entering a new group.
                    return _Continue(index, frames, guards)
        reset_history()
        return _Complete(frames) if full_prefix_feedback else adapter.finish()

    def recover_current(failure, frames, frame):
        # A failed finite search is not a proof against the current interface.
        # Explore other producer candidates and retain that unknown if none
        # produces a physically accepted group. Never rewind the watchdog.
        pending_recovery_budgets.clear()
        remember_recovery_budget(failure)
        failed_world = adapter.snapshot()
        exact = getattr(adapter, 'owns_committed_witness', None)
        earlier = [] if exact is None else [prior for prior in reversed(frames)
            if prior.port.group == failure.group and prior.port.net in failure.committed_owners
            and exact(prior.port, failure)]
        ran_early_negotiation = False
        if supports_negotiation and len(earlier) >= 2:
            ran_early_negotiation = True
            negotiated = negotiate_group_recovery(
                failure, failed_world, frames, earlier, (), group_start_hint=frame.snapshot
            )
            if negotiated is not None:
                return negotiated
            adapter.restore(failed_world)
            reset_history()

        def check_speculative_budget(error, port, prefix):
            underlying = (error.original if isinstance(error, (LocalInconclusive, LocalExhaustion))
                          else error)
            if type(underlying) is not GroupRouteBudgetStop:
                return
            diag = underlying.flow_error_data()
            if (port is None or not _is_recoverable_edge_budget(underlying)
                    or diag.get('group') != port.group or diag.get('net') != port.net
                    or (isinstance(error, (LocalInconclusive, LocalExhaustion))
                        and (error.group != port.group or error.net != port.net))
                    or recoverable_budget is None
                    or recoverable_budget(port, underlying, prefix) is None):
                raise underlying

        completed_reservation = getattr(adapter, 'completed_route_reservation', None)
        if completed_reservation is not None and getattr(adapter, 'route_avoiding', None) is not None:
            # Route the victim in the owner's pre-choice Book to discover its
            # actual geometry, then discard that probe and replay the original
            # order with the geometry temporarily reserved for the owner.
            for target in earlier:
                adapter.restore(target.snapshot)
                reset_history()
                adapter.checkpoint(frame.port, 'group_fallback_victim_reservation_domain')
                try:
                    rows = adapter.candidates(frame.port)
                    adapter.checkpoint(frame.port, 'group_fallback_victim_reservation_route')
                    selected = adapter.default_candidate(frame.port, rows)
                    adapter.route(frame.port, selected)
                    reservation = completed_reservation(frame.port)
                except GroupRouteLandingUnavailable as error:
                    if not (type(error) is GroupRouteLandingUnavailable
                            and error.group == frame.port.group and error.net == frame.port.net
                            and type(error.via_redundancy) is int and error.via_redundancy > 0
                            and error.via_redundancy == adapter.via_redundancy()):
                        raise
                    continue
                except (LocalInconclusive, LocalExhaustion) as rejected:
                    check_speculative_budget(rejected, frame.port, frames[:target.index])
                    if rejected.group != frame.port.group or rejected.net != frame.port.net:
                        raise ValueError('local result does not bind reservation victim')
                    remember_recovery_budget(rejected)
                    continue
                except GroupRouteBudgetStop as budget_exc:
                    check_speculative_budget(budget_exc, frame.port, frames[:target.index])
                    remember_recovery_budget(budget_exc)
                    continue
                if not reservation:
                    continue
                adapter.restore(target.snapshot)
                reset_history()
                adapter.checkpoint(target.port, 'group_fallback_victim_reservation_replay')
                try:
                    return walk(target.index, frames[:target.index],
                                ((target.index, target.snapshot, target.port),),
                                target.selected, avoidance=reservation)
                except (LocalInconclusive, LocalExhaustion, _PhysicalRejection,
                        _BranchLandingUnavailable) as rejected:
                    if isinstance(rejected, (LocalInconclusive, LocalExhaustion)):
                        matches = [i for i in range(target.index, len(ports))
                                   if ports[i].group == rejected.group
                                   and ports[i].net == rejected.net]
                        if len(matches) == 1:
                            idx = matches[0]
                            check_speculative_budget(rejected, ports[idx], frames[:idx])
                        else:
                            check_speculative_budget(rejected, None, ())
                    remember_recovery_budget(rejected)
                    record = getattr(adapter, 'record_outcome', None)
                    if record is not None:
                        record(target.port, target.key, rejected)
                except GroupRouteBudgetStop as budget_exc:
                    # walk already asked the adapter whether this stop was
                    # recoverable; a remaining budget stop must propagate.
                    raise
            adapter.restore(failed_world)
            reset_history()
        victim_streams = []
        if completed_reservation is not None and getattr(adapter, 'route_avoiding', None) is not None:
            for target in earlier:
                adapter.restore(target.snapshot)
                reset_history()
                adapter.checkpoint(frame.port, 'group_fallback_victim_alternative_domain')
                try:
                    victim_rows = adapter.candidates(frame.port)
                    victim_default = adapter.default_candidate(frame.port, victim_rows)
                except GroupRouteLandingUnavailable as error:
                    if not (type(error) is GroupRouteLandingUnavailable
                            and error.group == frame.port.group and error.net == frame.port.net
                            and type(error.via_redundancy) is int and error.via_redundancy > 0
                            and error.via_redundancy == adapter.via_redundancy()):
                        raise
                    continue
                adapter.checkpoint(frame.port, 'group_fallback_victim_alternative_order')
                alternatives = iter(adapter.alternatives(frame.port, victim_rows, victim_default))
                adapter.checkpoint(frame.port, 'group_fallback_victim_alternative_order_complete')
                victim_streams.append([target, alternatives,
                                       {adapter.candidate_key(frame.port, victim_default)}])

        def try_victim_alternative(stream):
            target, alternatives, seen = stream
            adapter.restore(target.snapshot)
            reset_history()
            while True:
                adapter.checkpoint(frame.port, 'group_fallback_victim_alternative_candidate')
                try:
                    selected = next(alternatives)
                except StopIteration:
                    return False, None
                key = adapter.candidate_key(frame.port, selected)
                if key not in seen:
                    seen.add(key)
                    break
            try:
                adapter.checkpoint(frame.port, 'group_fallback_victim_alternative_route')
                # Rows are bound to the Book that produced them. Recreate the
                # victim domain in this restored world and use its unique row,
                # rather than carrying a prior object across transactions.
                fresh_rows = adapter.candidates(frame.port)
                matches = []
                for index, row in enumerate(fresh_rows):
                    if index % 64 == 0:
                        adapter.checkpoint(frame.port, 'group_fallback_victim_alternative_rebind')
                    if adapter.candidate_key(frame.port, row) == key:
                        matches.append(row)
                if len(matches) != 1:
                    raise ValueError('victim alternative differs in its restored Book')
                selected = matches[0]
                adapter.route(frame.port, selected)
                reservation = completed_reservation(frame.port)
            except GroupRouteLandingUnavailable as error:
                if not (type(error) is GroupRouteLandingUnavailable
                        and error.group == frame.port.group and error.net == frame.port.net
                        and type(error.via_redundancy) is int and error.via_redundancy > 0
                        and error.via_redundancy == adapter.via_redundancy()):
                    raise
                return True, None
            except (LocalInconclusive, LocalExhaustion) as rejected:
                check_speculative_budget(rejected, frame.port, frames[:target.index])
                if rejected.group != frame.port.group or rejected.net != frame.port.net:
                    raise ValueError('local result does not bind reservation victim')
                remember_recovery_budget(rejected)
                return True, None
            except GroupRouteBudgetStop as budget_exc:
                check_speculative_budget(budget_exc, frame.port, frames[:target.index])
                remember_recovery_budget(budget_exc)
                return True, None
            if not reservation:
                return True, None
            adapter.restore(target.snapshot)
            reset_history()
            adapter.checkpoint(target.port, 'group_fallback_victim_alternative_replay')
            try:
                result = walk(target.index, frames[:target.index],
                              ((target.index, target.snapshot, target.port),),
                              target.selected, avoidance=reservation)
                return True, (result,)
            except (LocalInconclusive, LocalExhaustion, _PhysicalRejection,
                    _BranchLandingUnavailable) as rejected:
                if isinstance(rejected, (LocalInconclusive, LocalExhaustion)):
                    matches = [i for i in range(target.index, len(ports))
                               if ports[i].group == rejected.group and ports[i].net == rejected.net]
                    if len(matches) == 1:
                        idx = matches[0]
                        check_speculative_budget(rejected, ports[idx], frames[:idx])
                    else:
                        check_speculative_budget(rejected, None, ())
                remember_recovery_budget(rejected)
                record = getattr(adapter, 'record_outcome', None)
                if record is not None:
                    record(target.port, target.key, rejected)
            return True, None

        # One round before expensive negotiation; keep every remaining entry
        # for the ordinary fair loop below, subject to the same live budget.
        remaining_victims = []
        for victim_stream in victim_streams:
            more, result = try_victim_alternative(victim_stream)
            if result is not None:
                return result[0]
            if more:
                remaining_victims.append(victim_stream)
        victim_streams = remaining_victims
        adapter.restore(failed_world)
        reset_history()
        reserve = getattr(adapter, 'causal_reservations', None)
        if reserve is not None:
            # Samples belong to the current failed world, before any restore.
            trials = [(target, reserve(target.port, failure)) for target in earlier]
            for target, reservation in trials:
                if not reservation:
                    continue
                reservation_rounds = 0
                while reservation and reservation_rounds < 3:
                    reservation_rounds += 1
                    adapter.restore(target.snapshot)
                    reset_history()
                    adapter.checkpoint(target.port, 'group_fallback_causal_reservation')
                    report = getattr(adapter, 'record_causal_reservation', None)
                    if report is not None:
                        report(target.port, target.key, reservation)
                    try:
                        return walk(target.index, frames[:target.index],
                                    ((target.index, target.snapshot, target.port),),
                                    target.selected, avoidance=reservation)
                    except (LocalInconclusive, LocalExhaustion, _PhysicalRejection,
                            _BranchLandingUnavailable) as rejected:
                        remember_recovery_budget(rejected)
                        record = getattr(adapter, 'record_outcome', None)
                        if record is not None:
                            record(target.port, target.key, rejected)
                        if (not isinstance(rejected, LocalInconclusive)
                                or rejected.group != failure.group or rejected.net != failure.net
                                or target.port.net not in rejected.committed_owners):
                            break
                        # Read new witnesses while this actual trial Book is live.
                        addition = reserve(target.port, rejected)
                        if not addition or addition[0] != reservation[0]:
                            break
                        combined = tuple(sorted(set(reservation[1]) | set(addition[1])))
                        if combined == reservation[1]:
                            break
                        reservation = (reservation[0], combined)
                    except GroupRouteBudgetStop as budget_exc:
                        if not _is_recoverable_edge_budget(budget_exc):
                            raise
                        remember_recovery_budget(budget_exc)
                        record = getattr(adapter, 'record_outcome', None)
                        if record is not None:
                            record(target.port, target.key, budget_exc)
                        break
        adapter.restore(failed_world)
        if not ran_early_negotiation:
            negotiated = negotiate_group_recovery(
                failure, failed_world, frames, earlier, (), group_start_hint=frame.snapshot
            )
            if negotiated is not None:
                return negotiated
            adapter.restore(failed_world)
        seen_candidates = set()
        streams = []
        for target in (*earlier, frame):
            adapter.restore(target.snapshot)
            adapter.checkpoint(target.port, 'group_fallback_candidate_domain')
            rows = adapter.candidates(target.port)
            adapter.checkpoint(target.port, 'group_fallback_candidate_domain_complete')
            matches = [row for row in rows if adapter.candidate_key(target.port, row) == target.key]
            if len(matches) != 1:
                raise ValueError('original group candidate differs in its restored Book')
            seen_candidates.add((target.index, adapter.snapshot_key(target.snapshot), target.key))
            streams.append([target, rows, iter(adapter.alternatives(target.port, rows, matches[0]))])
        while streams or victim_streams:
            remaining = []
            for target, rows, alternatives in streams:
                candidate = None
                while True:
                    try:
                        cand = next(alternatives)
                    except StopIteration:
                        break
                    adapter.checkpoint(target.port, 'group_fallback_candidate')
                    identity = (target.index, adapter.snapshot_key(target.snapshot),
                                adapter.candidate_key(target.port, cand))
                    if identity not in seen_candidates:
                        seen_candidates.add(identity)
                        candidate = cand
                        break
                if candidate is None:
                    continue
                adapter.restore(target.snapshot)
                remaining.append([target, rows, alternatives])
                key = adapter.candidate_key(target.port, candidate)
                report = getattr(adapter, 'record_attempt', None)
                if report is not None:
                    report(target.port, key, candidate)
                try:
                    return walk(target.index, frames[:target.index],
                                ((target.index, target.snapshot, target.port),),
                                candidate, (target.snapshot, rows))
                except (LocalInconclusive, LocalExhaustion, _PhysicalRejection,
                        _BranchLandingUnavailable) as rejected:
                    remember_recovery_budget(rejected)
                    record = getattr(adapter, 'record_outcome', None)
                    if record is not None:
                        record(target.port, key, rejected)
                except GroupRouteBudgetStop as budget_exc:
                    if not _is_recoverable_edge_budget(budget_exc):
                        raise
                    remember_recovery_budget(budget_exc)
                    record = getattr(adapter, 'record_outcome', None)
                    if record is not None:
                        record(target.port, key, budget_exc)
            streams = remaining
            victim_remaining = []
            for victim_stream in victim_streams:
                more, result = try_victim_alternative(victim_stream)
                if result is not None:
                    return result[0]
                if more:
                    victim_remaining.append(victim_stream)
            victim_streams = victim_remaining
        if pending_recovery_budgets:
            raise pending_recovery_budgets[0]
        raise failure

    def recover(failure, frames, guards):
        pending_recovery_budgets.clear()
        remember_recovery_budget(failure)
        owners = failure.committed_owners
        if not owners:
            raise failure
        failed_world = adapter.snapshot()
        eligible = [(position, frames[position])
                    for position in range(len(frames) - 1, -1, -1)
                    if frames[position].port.group == failure.group
                    and frames[position].port.net in owners
                    and adapter.owns_committed(frames[position].port, failure)]
        negotiated = negotiate_group_recovery(
            failure, failed_world, frames, [frame for _pos, frame in eligible], guards
        )
        if negotiated is not None:
            return negotiated
        streams = []
        for position, frame in eligible:
            adapter.restore(frame.snapshot)
            adapter.checkpoint(frame.port, 'group_fallback_candidate_domain')
            regenerated = adapter.candidates(frame.port)
            adapter.checkpoint(frame.port, 'group_fallback_candidate_domain_complete')
            snapshot_digest = adapter.snapshot_key(frame.snapshot)
            streams.append([position, frame, snapshot_digest,
                            iter(adapter.alternatives(frame.port, regenerated, frame.selected))])
        # Give each causal earlier port one candidate per round. This is in
        # addition to source/portal fairness inside the adapter's iterator.
        while streams:
            remaining = []
            for position, frame, snapshot_digest, candidates in streams:
                adapter.checkpoint(frame.port, 'group_fallback_candidate')
                try:
                    candidate = next(candidates)
                except StopIteration:
                    continue
                remaining.append([position, frame, snapshot_digest, candidates])
                key = adapter.candidate_key(frame.port, candidate)
                prefix = tuple((f.index, f.key) for f in frames[:position])
                attempt = (prefix, frame.index, snapshot_digest, key)
                if attempt in visited:
                    continue
                visited.add(attempt)
                adapter.restore(frame.snapshot)
                report = getattr(adapter, 'record_attempt', None)
                if report is not None:
                    report(frame.port, key, candidate)
                try:
                    return walk(frame.index, frames[:position],
                                (*(guard for guard in guards if guard[0] < frame.index),
                                 (frame.index, frame.snapshot, frame.port)), candidate)
                except (LocalExhaustion, LocalInconclusive, _PhysicalRejection, _BranchLandingUnavailable) as rejected:
                    remember_recovery_budget(rejected)
                    # Preserve full branch provenance when a capture sink exists.
                    # Observer errors, budget and cancellation propagate normally.
                    record = getattr(adapter, 'record_outcome', None)
                    if record is not None:
                        record(frame.port, key, rejected)
                    continue
                except GroupRouteBudgetStop as budget_exc:
                    if not _is_recoverable_edge_budget(budget_exc):
                        raise
                    remember_recovery_budget(budget_exc)
                    record = getattr(adapter, 'record_outcome', None)
                    if record is not None:
                        record(frame.port, key, budget_exc)
                    continue
            streams = remaining
        if pending_recovery_budgets:
            raise pending_recovery_budgets[0]
        raise failure

    def complete(outcome):
        nonlocal first_failure
        while isinstance(outcome, _Continue):
            first_failure = None
            outcome = walk(outcome.index, outcome.frames, outcome.guards)
        return outcome

    def full_check():
        if ports:
            adapter.checkpoint(ports[-1], 'group_full_prefix_physical_feedback')
        accepted = adapter.physical_nonworsening(initial)
        if type(accepted) is not bool:
            raise ValueError('complete group physical feedback must return an exact bool')
        if ports:
            adapter.checkpoint(ports[-1], 'group_full_prefix_physical_feedback_complete')
        return accepted

    def replay_physical_failure(frames, first_feedback):
        nonlocal first_failure
        streams = [[frame, {frame.key}, True] for frame in reversed(frames)]
        attempts = 0
        prefix_budgets = []
        def remember_prefix_budget(error):
            if isinstance(error, (LocalInconclusive, LocalExhaustion)):
                error = error.original
            if type(error) is GroupRouteBudgetStop:
                if not _is_recoverable_edge_budget(error):
                    raise error
                prefix_budgets.append(error)
        while any(active for _frame, _tried, active in streams):
            for stream in streams:
                frame, tried, active = stream
                if not active:
                    continue
                adapter.restore(frame.snapshot)
                adapter.checkpoint(frame.port, 'group_full_prefix_candidate_domain')
                rows = adapter.candidates(frame.port)
                current = [row for row in rows if adapter.candidate_key(frame.port, row) == frame.key]
                if len(current) != 1:
                    raise ValueError('original group candidate differs in its restored Book')
                alternative = None
                for row in adapter.alternatives(frame.port, rows, current[0]):
                    key = adapter.candidate_key(frame.port, row)
                    if key not in tried:
                        alternative = row
                        tried.add(key)
                        break
                if alternative is None:
                    stream[2] = False
                    continue
                attempts += 1
                adapter.checkpoint(frame.port, 'group_full_prefix_candidate')
                first_failure = None
                try:
                    outcome = complete(walk(frame.index, frames[:frame.index], (),
                                            alternative, (frame.snapshot, rows)))
                except (LocalExhaustion, LocalInconclusive) as error:
                    # Only this speculative prefix exhausted; try the next one.
                    remember_prefix_budget(error)
                    continue
                except GroupRouteBudgetStop as error:
                    if not _is_recoverable_edge_budget(error):
                        raise
                    remember_prefix_budget(error)
                    continue
                except GroupRouteLandingUnavailable as error:
                    if (type(error) is GroupRouteLandingUnavailable
                            and type(error.via_redundancy) is int
                            and error.via_redundancy == adapter.via_redundancy()
                            and any(port.group == error.group and port.net == error.net
                                    for port in ports[frame.index:])):
                        continue
                    raise
                if not isinstance(outcome, _Complete):
                    raise ValueError('complete group replay did not reach the final prefix')
                if full_check():
                    return adapter.finish()
        if supports_negotiation:
            engine_obj = getattr(adapter, 'engine', None)
            if engine_obj is not None and not getattr(engine_obj, '_negotiation_port_legalize', False):
                adapter.restore(initial)
                reset_history()
                engine_obj._negotiation_port_legalize = True
                try:
                    first_failure = None
                    outcome = complete(walk(0, (), ()))
                    if isinstance(outcome, _Complete) and full_check():
                        return adapter.finish()
                except (LocalExhaustion, LocalInconclusive) as error:
                    remember_prefix_budget(error)
                except GroupRouteLandingUnavailable:
                    pass
                except GroupRouteBudgetStop as error:
                    if not _is_recoverable_edge_budget(error):
                        raise
                    remember_prefix_budget(error)
                finally:
                    engine_obj.__dict__.pop('_negotiation_port_legalize', None)
        if prefix_budgets:
            raise prefix_budgets[0]
        raise FullPrefixFeedbackRejected(attempts, first_feedback)

    try:
        outcome = walk(0, (), ())
        outcome = complete(outcome)
        if not full_prefix_feedback:
            return outcome
        if not isinstance(outcome, _Complete):
            raise ValueError('complete group pass did not reach the final prefix')
        if not ports or full_check():
            return adapter.finish()
        get_failure = getattr(adapter, 'last_physical_failure', None)
        first_feedback = None if get_failure is None else get_failure()
        return replay_physical_failure(outcome.frames, first_feedback)
    except LocalInconclusive as failure:
        reset_history()
        try:
            adapter.restore(initial)
        except BaseException as ending:
            from clay.flow.control_stop import record_unwind_failure
            record_unwind_failure(failure.original, ending)
        raise failure.original
    except LocalExhaustion:
        reset_history()
        original = (first_failure.original if first_failure is not None
                    else RuntimeError('unbound local exhaustion'))
        if full_prefix_feedback:
            try:
                adapter.restore(initial)
            except BaseException as ending:
                from clay.flow.control_stop import record_unwind_failure
                record_unwind_failure(original, ending)
        else:
            adapter.restore(initial)
        raise original
    except BaseException as original:
        reset_history()
        if (full_prefix_feedback or
                getattr(getattr(adapter, 'engine', None), '_post_selection_observation', None) is not None):
            try:
                adapter.restore(initial)
            except BaseException as ending:
                from clay.layout.route.group_endpoint_observation import record_observation_unwind
                record_observation_unwind(original, ending)
            raise
        adapter.restore(initial)
        raise
