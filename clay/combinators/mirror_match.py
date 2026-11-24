"""
Mirror match combinator — bias mirror coupling (v0.0.89 + v0.1.12 split).

`mirror_match(stage_a, stage_b)` declares that two stages' tail current
sources share a bias mirror. The lowering ties the two stages' bias
rails together so one stage's bias mirror absorbs the other's
diode-connected reference.

The v0.0.90 / v0.0.91 mirror-current invariant runtime (the
`MirrorMismatchViolation` dataclass + `check_mirror_match_invariant`
+ `inject_mirror_match_measures` helpers) lives here too. The
closed-loop ngspice driver for end-to-end mirror checks lives in
`mirror_match_runtime.py`.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ..ir import Cell, Device, Measure, Spec, Testbench
from ..stage_interface import ComposedStage
from ._common import StageRef, TemplateBuilder, _resolve


# ─── Mirror match ───────────────────────────────────────────────────

def mirror_match(
    stage_a: StageRef,
    stage_b: StageRef,
) -> ComposedStage:
    """`mirror_match(a, b)` — operating-point coupling declaration.

    Declares that `stage_a` and `stage_b` share a bias mirror — the
    sizing pass should enforce that the two stages' tail current
    sources track each other (same gate node + matched W·M).

    Type-check rules (this slice):
      * Both stages must have at least one bias_port OR both must
        carry an internal bias chain (in this slice we trust the
        declaration; future combinator runtime will verify by walking
        the lowered netlist).

    The lowering stage records the pairing in `meta['pairs']` so a
    follow-up sizing pass can wire the constraint. The external
    surface of the result is `stage_a`'s — `mirror_match` is a
    declarative modifier, not a structural transform.
    """
    a = _resolve(stage_a)
    b = _resolve(stage_b)
    return ComposedStage(
        kind="mirror",
        interface=a.interface,
        parts=(a, b),
        meta={"pairs": (a.template_name, b.template_name)},
    )


# ─── Mirror match lowering (v0.0.89) ───────────────────────────────


def _lower_mirror_match(
    first: ComposedStage,
    second: ComposedStage,
    meta: Dict[str, Any],
) -> TemplateBuilder:
    """Build the mirror-match builder.

    Two stages declared in `mirror_match(a, b)` share a bias mirror —
    i.e. their tail current sources reference the same gate node and
    track each other across PVT. The lowering enforces this by tying
    the two stages' bias rails (`pbias` and / or `nbias`) together so
    one stage's bias mirror absorbs the other's diode-connected current
    reference.

    Strategy: build BOTH stages with shared cell names, then walk the
    union of their device lists; for every device on stage B whose
    pins reference `pbias` / `nbias`, redirect to stage A's
    corresponding rail. Stage B's own diode + Iref devices for the
    shared rail are then dropped (stage A already provides them).

    The shared bias name lands on `meta['shared_bias']` so downstream
    tooling can see which rail was tied. If neither stage has a
    recognisable bias rail to share, the lowering degrades to stage
    A's own builder (the v0.0.88 fallback path).

    Mirror current enforcement (v0.0.90):
      In addition to structural bias-rail sharing, the lowering now
      identifies every pair of mirror devices (devices on stage A
      and their `b_`-prefixed counterparts on stage B that both gate
      from a shared bias rail). The pair list lands on
      `meta['mirror_pairs']` AND on `cell.__dict__['_mirror_pairs']`
      so:
        * `check_mirror_match_invariant` can compare measured I_mirror
          on each pair and surface a `MirrorMismatchViolation` when
          the ratio sits outside ±tolerance.
        * `inject_mirror_match_measures` can append the per-device
          current probes (`i(M<name>)`) to a Testbench so ngspice
          actually reports the mirror currents in the verify result's
          `measures` map.
      The lowering also stashes the `.print op @m<dev>[id]` directive
      strings on `cell.__dict__['_mirror_print_directives']` for any
      downstream tooling that wants to round-trip the literal probe
      lines into a generated deck.
    """
    from ._lower import lower_to_template
    first_builder = lower_to_template(first)
    second_builder = lower_to_template(second)

    def _builder(cell_name: str, spec: Spec, **kwargs: Any) -> Cell:
        # The mirror-match composed surface is stage A's surface — the
        # second stage is structurally subordinate (shared bias rail).
        # We build stage A normally and stage B as a sub-cell whose
        # bias devices get folded into stage A's bias chain.
        first_cell = first_builder(f"{cell_name}_a", spec, **kwargs)
        second_cell = second_builder(f"{cell_name}_b", spec, **kwargs)

        # Find the common bias rails between the two cells. We look for
        # any net named `pbias` or `nbias` that appears on a device pin
        # in BOTH cells — that's the canonical mirror anchor in every
        # first-cohort template.
        first_nets = _collect_internal_nets(first_cell)
        second_nets = _collect_internal_nets(second_cell)
        shared_rails: List[str] = sorted(
            {n for n in ("pbias", "nbias")
             if n in first_nets and n in second_nets}
        )

        # If the two stages share NO recognised bias rail, the mirror
        # constraint is purely declarative and the composed cell is
        # just stage A (matching the v0.0.88 lowering for unknown
        # topology pairs).
        if not shared_rails:
            return first_cell

        meta["shared_bias"] = list(shared_rails)

        # Devices to add: every device on stage B that doesn't redundantly
        # generate the shared rail. Specifically, drop:
        #   * Bias diode devices that source/sink the shared rail (their
        #     gate equals their drain equals a shared rail).
        #   * Iref current sources whose terminal is a shared rail.
        # The retained stage-B devices get prefix `b_` on their instance
        # names so they don't collide with stage A's instances.
        kept_b_devices: List[Device] = []
        for d in second_cell.devices:
            if _device_supplies_shared_rail(d, shared_rails):
                continue
            kept_b_devices.append(d)

        # Now: every retained stage-B device that pins to a shared rail
        # net keeps that name (it now refers to stage A's shared rail).
        # Internal stage-B nets (non-shared, non-supply, non-port) get a
        # `b_` prefix to avoid collisions with stage A's internal nets.
        external_b_set = set(second_cell.ports)
        supply_set = set(p for p in second_cell.ports if p in ("vdd", "vss"))
        # Sized as "external bias rails to keep" — namely those nets
        # that ALREADY appear on stage A's surface (the merge target).
        keep_as_is: set[str] = set(shared_rails) | supply_set | external_b_set

        def _rename_b(net: str) -> str:
            if net in keep_as_is:
                return net
            return f"b_{net}"

        renamed_b: List[Device] = []
        for d in kept_b_devices:
            new_pins = {
                pin_name: _rename_b(net) for pin_name, net in d.pins.items()
            }
            renamed_b.append(Device(
                instance_name=f"b_{d.instance_name}",
                device_class=d.device_class,
                pdk_variant=d.pdk_variant,
                pins=new_pins,
                sizing=dict(d.sizing),
            ))

        merged = list(first_cell.devices) + renamed_b

        # ─── Mirror current enforcement (v0.0.90): identify pairs ────
        #
        # A mirror pair is a stage-A device + its stage-B `b_`-prefixed
        # counterpart where BOTH gate from one of the shared bias rails
        # AND the drain is something other than the rail itself (i.e.
        # they're a real mirror sourcing/sinking current to a node, not
        # a degenerate gate-drain-shorted diode chain).
        mirror_pairs = _identify_mirror_pairs(
            first_cell.devices, renamed_b, shared_rails,
        )
        meta["mirror_pairs"] = list(mirror_pairs)

        out_cell = Cell(
            name=cell_name,
            ports=list(first_cell.ports),
            spec=spec,
            devices=merged,
            pdk_id=first_cell.pdk_id,
        )
        # Stash mirror-pair metadata on the cell __dict__ so:
        #   * `check_mirror_match_invariant` can read it when only the
        #     cell (not the composed) is in scope.
        #   * `inject_mirror_match_measures` can append per-device
        #     current probes to a testbench built around this cell.
        out_cell.__dict__["_mirror_pairs"] = list(mirror_pairs)
        out_cell.__dict__["_mirror_print_directives"] = (
            _format_mirror_print_directives(mirror_pairs)
        )
        return out_cell

    _builder.__name__ = (
        f"mirror_match__{first.template_name}__{second.template_name}"
    )
    return _builder


# ─── Mirror match invariant + utilities (v0.0.90) ──────────────────


@dataclass
class MirrorMismatchViolation:
    """One mirror-current mismatch outcome from `check_mirror_match_invariant`.

    Mirror match enforces that every (stage_a_device, stage_b_device)
    pair on a shared bias rail carries currents whose ratio sits within
    a tolerance band around 1.0 (mismatch percentage). When the measured
    ratio exceeds the band, this violation is emitted with structured
    fields an LLM consumer can read directly.

    Categories:
      * `mismatch`        — both currents resolved, ratio outside the
                            tolerance band.
      * `measure_missing` — one or both per-device currents absent from
                            `op_result`; the runtime cannot compare.
                            The `actual_ratio` / `relative_error` fields
                            land as `None` in this case.
      * `skipped`         — the composed stage has no mirror pairs at
                            all (e.g. `mirror_match` degenerated on an
                            unsupported topology pair); the runtime
                            returns None directly so callers do not see
                            this category.

    Every MirrorMismatchViolation is JSON-serializable via `to_json_dict`
    — the schema mirrors `clay.invariants.InvariantViolation` so a
    consumer that handles both can use one rendering path.
    """
    pair: Tuple[str, str]
    target_ratio: float
    actual_ratio: Optional[float]
    relative_error: Optional[float]
    tolerance: float
    category: str
    message: str
    suggested_adjustment: str = ""
    # The per-device currents the comparison consumed. `None` for
    # `measure_missing` outcomes. Surfaced verbatim so an LLM can also
    # see the raw numbers, not just the ratio.
    current_a: Optional[float] = None
    current_b: Optional[float] = None
    # Free-form `{key: value}` for tooling that wants extra context
    # (e.g. which measure key was consulted, the M-ratio asymmetry, etc.).
    extras: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """True iff this violation is an informational `skipped` entry.

        Mirror match never emits `skipped` outcomes through the public
        check function (the function returns None instead), so this
        property is False for every violation a caller will see in
        practice. It exists to keep parity with the
        `InvariantViolation` surface so downstream rendering / filter
        code can treat both violation types uniformly.
        """
        return self.category == "skipped"

    def to_json_dict(self) -> dict:
        """Return a plain Python dict suitable for `json.dumps`.

        Tuple fields land as 2-element lists (JSON has no tuple type).
        The `extras` map is included verbatim.
        """
        out = asdict(self)
        # Tuples don't survive JSON cleanly — normalise to a list so
        # `json.dumps(v.to_json_dict())` produces a stable wire form.
        out["pair"] = list(out["pair"])
        return out


def _identify_mirror_pairs(
    first_devices: List[Device],
    renamed_b_devices: List[Device],
    shared_rails: List[str],
) -> List[Tuple[str, str]]:
    """Find every (stage_a_device, stage_b_device) pair on a shared rail.

    A "mirror device" gates from one of the shared bias rails AND
    drains to a different net (otherwise it's a diode-connected
    reference, not a mirror). The pair lookup uses the `b_`-prefix
    convention the lowering emits for stage B device renames.

    Returns a list of `(name_a, name_b)` tuples in deterministic order
    (sorted by `name_a`). Stage A devices without a matching stage B
    counterpart are skipped silently — they may be valid mirrors that
    were dropped on the B side as bias-supply devices.
    """
    rails = set(shared_rails)

    def _is_mirror(d: Device) -> bool:
        if d.device_class not in ("nmos", "pmos"):
            return False
        g = d.pins.get("g")
        drain = d.pins.get("d")
        if g is None or drain is None:
            return False
        # Gate on a shared rail AND drain NOT on the rail itself —
        # the canonical mirror shape.
        if g in rails and drain not in rails:
            return True
        return False

    a_mirrors: Dict[str, Device] = {
        d.instance_name: d for d in first_devices if _is_mirror(d)
    }
    b_by_name: Dict[str, Device] = {
        d.instance_name: d for d in renamed_b_devices
    }

    pairs: List[Tuple[str, str]] = []
    for a_name in sorted(a_mirrors):
        b_name = f"b_{a_name}"
        if b_name not in b_by_name:
            continue
        b_dev = b_by_name[b_name]
        if not _is_mirror(b_dev):
            # The B counterpart was dropped or rewired into something
            # other than a mirror; skip the pair.
            continue
        pairs.append((a_name, b_name))
    return pairs


def _format_mirror_print_directives(
    pairs: List[Tuple[str, str]],
) -> List[str]:
    """Return `.print op @m<dev>[id]` directive strings for every device
    in `pairs` (both sides). Used by tooling that wants to surface the
    literal probe lines (for grep-friendly debug logs or deck round-trip).

    Each line follows the same format the `_ir_bridge` functional-role
    OP injection uses, so the produced strings drop into an existing
    .control block without surface changes.
    """
    out: List[str] = []
    seen: set[str] = set()
    for (a_name, b_name) in pairs:
        for dev in (a_name, b_name):
            if dev in seen:
                continue
            seen.add(dev)
            out.append(f".print op @m{dev}[id]")
    return out


def _mirror_pairs_of(composed_or_cell: Any) -> List[Tuple[str, str]]:
    """Resolve mirror-pair metadata from a ComposedStage or a Cell.

    Accepts:
      * `ComposedStage` — reads `composed.meta['mirror_pairs']`.
      * `Cell`          — reads `cell.__dict__['_mirror_pairs']`.

    Returns an empty list when no metadata is present (e.g. the cell
    was not built via a mirror_match lowering, or the lowering
    degraded to stage A's own builder because no shared rail was
    available).
    """
    if isinstance(composed_or_cell, ComposedStage):
        raw = composed_or_cell.meta.get("mirror_pairs", [])
    elif isinstance(composed_or_cell, Cell):
        raw = composed_or_cell.__dict__.get("_mirror_pairs", [])
    else:
        return []
    out: List[Tuple[str, str]] = []
    for item in raw:
        if isinstance(item, tuple) and len(item) == 2:
            out.append((str(item[0]), str(item[1])))
        elif isinstance(item, list) and len(item) == 2:
            out.append((str(item[0]), str(item[1])))
    return out


# Resolution order for finding a device current in op_result. The keys
# below mirror the conventions used by `_ir_bridge._op_quantities_for_role`
# and the OP-quantity log parser in `clay.invariants`, so a measure dict
# produced by the existing testbench machinery resolves on the FIRST
# attempt for either spelling.
def _resolve_device_current(
    name: str, op_result: Mapping[str, float],
) -> Tuple[Optional[float], str]:
    """Return `(current, source_key)` for the named device, or
    `(None, "")` when no resolvable key is present.

    Resolution order (first finite value wins):
      1. `i_<name>`        — the canonical key
         `_op_quantities_for_role` emits for tail_current_source roles.
      2. `id_<name>`       — alternate emitted by the same machinery.
      3. `i(<name>)`       — direct `Measure("i_<name>", "op", "i(<name>)")`
                             surface form a caller might attach.
      4. `i_M<name>` /     — fallback spellings that surface when the
         `id_M<name>`        device naming carried an instance-letter
                             prefix in the raw probe.
      5. Lowercase variants — ngspice often lowercases identifiers in
                             print-op output; the canonical key may have
                             been re-lowercased through the merge layer.
    """
    candidates: List[str] = [
        f"i_{name}", f"id_{name}",
        f"i({name})",
        f"i_m{name}", f"id_m{name}",
        f"i_{name.lower()}", f"id_{name.lower()}",
    ]
    for key in candidates:
        if key in op_result:
            val = op_result[key]
            if isinstance(val, (int, float)) and math.isfinite(float(val)):
                return float(val), key
    return None, ""


def check_mirror_match_invariant(
    composed_stage: Any,
    op_result: Mapping[str, float],
    tolerance: float = 0.10,
) -> Optional[MirrorMismatchViolation]:
    """Compare measured mirror currents on every declared mirror pair.

    Mirror match declares that two stages' shared bias rail forces
    pairs of mirror devices to carry currents in a target ratio
    (default 1.0). The runtime resolves each device's current from
    `op_result`, computes the actual ratio on every pair, and returns
    a `MirrorMismatchViolation` for the FIRST pair whose ratio sits
    outside `1.0 ± tolerance`. Returns `None` when:
      * the composed stage has no mirror pairs (declarative-only
        `mirror_match` with no shared rail), or
      * every pair's ratio matches the target within tolerance.

    The returned violation carries:
      * `pair`                — `(name_a, name_b)`.
      * `target_ratio`        — `1.0`.
      * `actual_ratio`        — `|i_b / i_a|`. `None` for measure_missing.
      * `relative_error`      — `(actual_ratio - target_ratio) / target_ratio`.
      * `tolerance`           — the supplied band.
      * `category`            — `'mismatch'` or `'measure_missing'`.
      * `message`             — one-line human-readable description.
      * `suggested_adjustment`— text hint for the LLM (which device's W
                                to change, in which direction).
      * `current_a`, `current_b` — raw per-device currents.
      * `extras`              — `{actual_source_a, actual_source_b}`.

    Args:
      composed_stage: a `ComposedStage` produced by `mirror_match(a, b)`
        (the runtime reads `composed.meta['mirror_pairs']`). Also accepts
        a `Cell` whose `__dict__['_mirror_pairs']` carries the same
        metadata (the path the lowering takes when the cell is the only
        in-scope handle).
      op_result: `{measure_name: value}` map of OP-analysis measures
        produced by the testbench. The runtime resolves the per-device
        currents via the same key conventions as the v0.0.87 functional-
        role injection layer (`i_<name>`, `id_<name>`, `i(<name>)`).
      tolerance: relative band around the target ratio. The pair's
        ratio must satisfy `|ratio - 1.0| <= tolerance` to pass.
        Default 0.10 (±10%) — tighter than the v0.0.86 functional-role
        default of ±20% because mirror-current matching is the textbook
        bias-design guarantee, not a behavioral soft target.

    Returns:
      First `MirrorMismatchViolation` encountered, or `None` when every
      pair is in tolerance OR the composed stage has no pairs.
    """
    if tolerance <= 0:
        raise ValueError(
            f"tolerance must be positive, got {tolerance!r}"
        )

    pairs = _mirror_pairs_of(composed_stage)
    if not pairs:
        return None

    op = dict(op_result) if op_result is not None else {}

    for (name_a, name_b) in pairs:
        i_a, src_a = _resolve_device_current(name_a, op)
        i_b, src_b = _resolve_device_current(name_b, op)

        # If either current is missing, surface a measure_missing
        # violation pointing at exactly which device probe wasn't
        # produced. We return at the first such pair — the LLM should
        # fix its testbench (or wire `inject_mirror_match_measures`)
        # before any meaningful comparison runs.
        if i_a is None or i_b is None:
            missing = [
                n for n, v in ((name_a, i_a), (name_b, i_b)) if v is None
            ]
            hint = (
                f"add `.print op @m<dev>[id]` directives or call "
                f"`clay.stage_combinators.inject_mirror_match_measures"
                f"(testbench)` so ngspice reports per-device currents "
                f"for: {missing}"
            )
            return MirrorMismatchViolation(
                pair=(name_a, name_b),
                target_ratio=1.0,
                actual_ratio=None,
                relative_error=None,
                tolerance=tolerance,
                category="measure_missing",
                message=(
                    f"mirror_match pair ({name_a!r}, {name_b!r}): "
                    f"missing per-device current measures for "
                    f"{missing}. {hint}"
                ),
                suggested_adjustment=hint,
                current_a=i_a,
                current_b=i_b,
                extras={
                    "missing": missing,
                    "actual_source_a": src_a,
                    "actual_source_b": src_b,
                },
            )

        # Compare on absolute magnitudes — current direction conventions
        # vary across testbenches (some return signed values per the
        # ngspice device-pin polarity convention). Mirror matching is
        # about magnitude.
        abs_a, abs_b = abs(i_a), abs(i_b)
        if abs_a < 1e-30:
            # Pathological — stage A current is zero. Surface as a
            # mismatch with an explicit infinity-shaped ratio so the
            # LLM sees the failure mode clearly.
            return MirrorMismatchViolation(
                pair=(name_a, name_b),
                target_ratio=1.0,
                actual_ratio=float("inf") if abs_b > 0 else 1.0,
                relative_error=float("inf") if abs_b > 0 else 0.0,
                tolerance=tolerance,
                category="mismatch",
                message=(
                    f"mirror_match pair ({name_a!r}, {name_b!r}): "
                    f"stage-A current is ~zero ({i_a:g} A); mirror "
                    f"ratio is undefined. Check the bias chain."
                ),
                suggested_adjustment=(
                    "stage A's mirror device is off — verify the shared "
                    "bias rail settles to a sensible Vgs (Iref present, "
                    "diode device sized to carry the reference)."
                ),
                current_a=i_a,
                current_b=i_b,
                extras={
                    "actual_source_a": src_a,
                    "actual_source_b": src_b,
                },
            )

        ratio = abs_b / abs_a
        rel_err = (ratio - 1.0) / 1.0
        if abs(rel_err) <= tolerance:
            # Within band — skip silently, keep scanning.
            continue

        direction = "exceeds" if rel_err > 0 else "underperforms"
        suggested = _suggest_mirror_adjustment(name_a, name_b, ratio)
        return MirrorMismatchViolation(
            pair=(name_a, name_b),
            target_ratio=1.0,
            actual_ratio=ratio,
            relative_error=rel_err,
            tolerance=tolerance,
            category="mismatch",
            message=(
                f"mirror_match pair ({name_a!r}, {name_b!r}): "
                f"current ratio I_{name_b}/I_{name_a}={ratio:.4g} "
                f"{direction} target 1.0 by {abs(rel_err)*100:.1f}% "
                f"(|i_a|={abs_a:.3e} A, |i_b|={abs_b:.3e} A, "
                f"tolerance=±{tolerance*100:g}%)"
            ),
            suggested_adjustment=suggested,
            current_a=i_a,
            current_b=i_b,
            extras={
                "actual_source_a": src_a,
                "actual_source_b": src_b,
            },
        )

    # Every pair within tolerance.
    return None


def _suggest_mirror_adjustment(
    name_a: str, name_b: str, actual_ratio: float,
) -> str:
    """LLM-targeted hint for fixing a mirror current mismatch."""
    if actual_ratio > 1.0:
        # i_b exceeds i_a — shrink stage B's mirror device or grow A's.
        return (
            f"stage B's mirror current exceeds stage A by "
            f"{(actual_ratio - 1.0) * 100:.1f}%. Options: reduce "
            f"{name_b!r}'s W (or M multiplier) so it matches "
            f"{name_a!r}, or verify both devices share the same gate "
            f"net (the lowering ties stage B's bias rail to stage A — "
            f"a stray b_pbias / b_nbias suggests the rename was "
            f"incomplete)."
        )
    return (
        f"stage A's mirror current exceeds stage B by "
        f"{(1.0 - actual_ratio) * 100:.1f}%. Options: increase "
        f"{name_b!r}'s W (or M multiplier) so it matches "
        f"{name_a!r}, or check whether stage B's mirror device "
        f"shares the same bias rail (a missed rename leaves it on "
        f"a floating internal net)."
    )


def inject_mirror_match_measures(testbench: Testbench) -> Testbench:
    """Append per-device current probes to `testbench.measures` for every
    mirror pair declared on `testbench.cell.__dict__['_mirror_pairs']`.

    The added measures take the form `Measure("i_<dev>", "op",
    "i(<dev>)")` which the ngspice driver translates to a `.meas op`
    directive returning the device's branch current. After the verify
    run, the resulting measures dict contains `i_<dev>` entries the
    `check_mirror_match_invariant` runtime resolves on its first
    attempt.

    Returns the SAME testbench (mutated in place) so the helper drops
    into a fluent chain: `verify(inject_mirror_match_measures(tb))`.

    No-op when:
      * The cell has no `_mirror_pairs` metadata.
      * A measure with the same name is already in `tb.measures`
        (idempotent — calling twice produces no duplicates).
    """
    pairs = _mirror_pairs_of(testbench.cell)
    if not pairs:
        return testbench

    existing_names: set[str] = {m.name for m in testbench.measures}
    devices_seen: set[str] = set()
    for (a, b) in pairs:
        for dev in (a, b):
            if dev in devices_seen:
                continue
            devices_seen.add(dev)
            meas_name = f"i_{dev}"
            if meas_name in existing_names:
                continue
            testbench.measures.append(Measure(
                name=meas_name,
                analysis_ref="op",
                expr=f"i({dev})",
            ))
            existing_names.add(meas_name)
    return testbench


# ─── Utility helpers shared between lowering + invariant ────────────


def _collect_internal_nets(cell: Cell) -> set[str]:
    """Return every net name referenced by any pin of `cell.devices`."""
    nets: set[str] = set()
    for d in cell.devices:
        for net in d.pins.values():
            nets.add(net)
    return nets


def _device_supplies_shared_rail(
    d: Device, shared_rails: List[str],
) -> bool:
    """True if `d` is a bias-supply device for one of the shared rails.

    Three patterns count as "supplies the rail":
      * NMOS/PMOS diode-connected to the rail: `g == d` and both are
        the rail name.
      * A `cur_src` whose `p` or `n` terminal is the rail.
      * An NMOS/PMOS mirror whose `g` is the rail and whose `d` is also
        the rail (i.e. shorted-gate diode chain — uncommon but possible).
    """
    rails = set(shared_rails)
    cls = d.device_class
    if cls in ("nmos", "pmos"):
        g = d.pins.get("g")
        drain = d.pins.get("d")
        if g is not None and g == drain and g in rails:
            return True
        if g in rails and drain in rails:
            return True
    if cls == "cur_src":
        p = d.pins.get("p")
        n = d.pins.get("n")
        if p in rails or n in rails:
            return True
    return False
