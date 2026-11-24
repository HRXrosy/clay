"""Unit tests for the EDALLM -> clean-room core desugaring.

These assert byte-stable emit for fixed fixtures (golden strings) plus the
structural invariants the clean-room kernel's C1 check enforces (net
completeness, role projection, constraint routing). They do NOT invoke the
Lean kernel — that end-to-end check is run by the elaboration harness.
"""

from __future__ import annotations

from clay.ir import Cell, Device, Spec, SpecField, CellInstance
from clay.elaborate.to_core import (
    emit_cell,
    emit_leaf,
    _role_kw,
    _sanitize_ident,
    _esc_pred,
    resource_feasibility_constraint,
    DeferredRecord,
)


def _mk_inverter() -> Cell:
    """Minimal 2-device fixture: a CMOS inverter (pmos pull-up, nmos pull-down)."""
    return Cell(
        name="Inverter",
        ports=["a", "y", "vdd", "vss"],
        spec=Spec(fields=[SpecField("gain_dc", "db", ">=", 20.0)]),
        devices=[
            Device("MP", "pmos", None, {"g": "a", "d": "y", "s": "vdd", "b": "vdd"}, {}),
            Device("MN", "nmos", None, {"g": "a", "d": "y", "s": "vss", "b": "vss"}, {}),
        ],
    )


GOLDEN_INVERTER = """# desugared core for cell 'Inverter' (clay.elaborate.to_core)
node interface {
    in a,
    in y,
    supply vdd,
    supply vss
  } composition {
    nets { a, y, vdd, vss }
    inst leaf interface { in g, out d, supply s, supply b } model pmos bind { g = a, d = y, s = vdd, b = vdd },
    inst leaf interface { in g, out d, supply s, supply b } model nmos bind { g = a, d = y, s = vss, b = vss }
  }
constraints {
    constraint oracle electrical gain_dc "gain_dc >= 20"
  }
"""


def test_inverter_golden() -> None:
    """Emit is byte-stable for the inverter fixture."""
    assert emit_cell(_mk_inverter()) == GOLDEN_INVERTER


def test_nets_complete() -> None:
    """nets{} lists every bind-RHS net AND every interface port (C1 obligation)."""
    text = emit_cell(_mk_inverter())
    nets_line = [l for l in text.splitlines() if "nets {" in l][0]
    declared = {t.strip() for t in nets_line.split("{", 1)[1].rstrip(" }").split(",")}
    # every port present
    for p in ("a", "y", "vdd", "vss"):
        assert p in declared
    # every bind net present (all nets here are also ports)
    assert declared == {"a", "y", "vdd", "vss"}


def test_role_projection_lossy_to_five() -> None:
    """7 EDALLM roles collapse to 5 kernel keywords."""
    assert _role_kw("vdd", "power_p") == "supply"
    assert _role_kw("vss", "power_n") == "supply"
    assert _role_kw("gnd", "ground") == "supply"
    assert _role_kw("inp", "input") == "in"
    assert _role_kw("outp", "output") == "out"
    assert _role_kw("d", "inout") == "in"
    assert _role_kw("clk", "clock") == "clock"
    assert _role_kw("ib", "bias") == "bias"


def test_role_heuristic_when_unannotated() -> None:
    assert _role_kw("vdd", None) == "supply"
    assert _role_kw("vss", None) == "supply"
    assert _role_kw("clk", None) == "clock"
    assert _role_kw("ibias", None) == "bias"
    assert _role_kw("inp", None) == "in"
    assert _role_kw("out", None) == "out"
    assert _role_kw("mystery", None) == "in"


def test_pure_leaf_emit() -> None:
    cell = Cell(
        name="folded_cascode_ota",
        ports=["vinp", "vinn", "vout", "ibias", "vdd", "vss"],
        spec=Spec(fields=[]),
        devices=[],
    )
    text = emit_leaf(cell)
    assert text.startswith("leaf interface {")
    assert 'model "folded_cascode_ota"' in text or "model folded_cascode_ota" in text
    assert "bias ibias" in text
    assert "supply vdd" in text


def test_spec_op_and_value_in_pred() -> None:
    cell = Cell(
        name="X",
        ports=["a"],
        spec=Spec(fields=[
            SpecField("gbw", "freq", ">=", 1.0e7),
            SpecField("offset", "volt", "<=", 1e-3),
        ]),
        devices=[Device("M", "nmos", None, {"g": "a", "d": "a", "s": "a", "b": "a"}, {})],
    )
    text = emit_cell(cell)
    assert 'constraint oracle electrical gbw "gbw >= 10000000.0"' in text
    assert 'constraint oracle electrical offset "offset <= 0.001"' in text


def test_resource_feasibility_emits_one_c3_line() -> None:
    """gbw/cl/vdd/power in the Spec -> exactly one oracle/resource/feasibility line,
    pred = whitespace-joined key=val in stable order (the clay.c3.adjudicate --spec
    format). The original >=/<= op is dropped (the parser only reads key=value)."""
    cell = Cell(
        name="ResCell",
        ports=["a"],
        spec=Spec(fields=[
            SpecField("gain_dc", "db", ">=", 40.0),
            SpecField("gbw", "freq", ">=", 1.0e7),
            SpecField("cl", "cap", ">=", 1e-12),
            SpecField("vdd", "volt", ">=", 1.8),
            SpecField("power", "pwr", "<=", 1e-3),
        ]),
        devices=[Device("M", "nmos", None, {"g": "a", "d": "a", "s": "a", "b": "a"}, {})],
    )
    lines = resource_feasibility_constraint(cell)
    assert lines == [
        'constraint oracle resource feasibility "gbw=10000000.0 cl=1e-12 vdd=1.8 power=0.001"'
    ]
    # and it lands in the full emit exactly once
    text = emit_cell(cell)
    assert text.count("constraint oracle resource feasibility") == 1
    assert 'constraint oracle resource feasibility "gbw=10000000.0 cl=1e-12 vdd=1.8 power=0.001"' in text


def test_resource_feasibility_absent_when_no_resource_field() -> None:
    """A spec with only gain_dc emits no resource feasibility line (C3 never fires)."""
    cell = Cell(
        name="NoRes",
        ports=["a"],
        spec=Spec(fields=[SpecField("gain_dc", "db", ">=", 20.0)]),
        devices=[Device("M", "nmos", None, {"g": "a", "d": "a", "s": "a", "b": "a"}, {})],
    )
    assert resource_feasibility_constraint(cell) == []
    assert "resource feasibility" not in emit_cell(cell)


def test_resource_feasibility_cl_exact_match_no_false_positive() -> None:
    """`cl` requires an exact field name; `clock`/`recall` must NOT match it."""
    cell = Cell(
        name="ClockCell",
        ports=["a"],
        spec=Spec(fields=[
            SpecField("clock_freq", "freq", ">=", 1e8),
            SpecField("recall", "db", ">=", 1.0),
        ]),
        devices=[Device("M", "nmos", None, {"g": "a", "d": "a", "s": "a", "b": "a"}, {})],
    )
    # no bare `cl`, no gbw/vdd/power -> no resource line
    assert resource_feasibility_constraint(cell) == []


def test_sizing_routes_structural_vs_electrical() -> None:
    cell = _mk_inverter()
    text = emit_cell(cell, sizing_kwargs={"gm_id_diff": 15, "N_tail": 2})
    assert 'constraint oracle electrical sizing_gm_id_diff "gm_id_diff = 15 [intent]"' in text
    assert 'constraint core structural device_multiplier "N_tail = 2 [intent]"' in text


def test_annotate_geometric() -> None:
    from clay.layout.annotations import Matching, Symmetric, MatchingKind, SymmetryAxis

    cell = _mk_inverter()
    cell.__dict__["_layout_annotations"] = {
        "MP": [Matching("MN", MatchingKind.RANDOM_VT),
               Symmetric(["MP", "MN"], SymmetryAxis.VERTICAL)],
    }
    text = emit_cell(cell)
    assert 'constraint representation geometric matching "matching_pair(MP,MN,random_vt)"' in text
    assert 'constraint representation geometric symmetry "symmetric([MP,MN],vertical)"' in text


def test_no_orchestration_leaks() -> None:
    """The emitter NEVER puts flow/quality/policy/verify content into the core."""
    cell = _mk_inverter()
    cell.__dict__["flow"] = {"pdk": "gf180mcu", "steps": ["compile", "sign"]}
    cell.__dict__["quality"] = {"kpi": "drc_clean"}
    deferred = DeferredRecord()
    text = emit_cell(cell, deferred=deferred)
    assert "flow" not in text
    assert "quality" not in text
    assert "pdk" not in text
    # but recorded as deferred
    assert any("flow_decl" in n for n in deferred.notes)
    assert any("quality_decl" in n for n in deferred.notes)


def test_hierarchical_inline() -> None:
    """A CellInstance inlines the resolved child as a nested node."""
    child = _mk_inverter()
    parent = Cell(
        name="Buffer",
        ports=["bin", "bout", "vdd", "vss"],
        spec=Spec(fields=[]),
        devices=[],
    )
    parent.__dict__["instances"] = [
        CellInstance(name="inv0", cell_path="local::Inverter",
                     port_map={"a": "bin", "y": "bout", "vdd": "vdd", "vss": "vss"}),
    ]

    def resolve(path: str) -> Cell:
        assert path == "local::Inverter"
        return child

    text = emit_cell(parent, resolve=resolve)
    assert text.startswith("# desugared core for cell 'Buffer'")
    # nested child node is inlined inside an inst ... bind
    assert "inst node interface {" in text
    assert "bind { a = bin, y = bout, vdd = vdd, vss = vss }" in text
    # parent nets include outer nets referenced by the child bind
    nets_line = [l for l in text.splitlines() if "nets {" in l][0]
    declared = {t.strip() for t in nets_line.split("{", 1)[1].rstrip(" }").split(",")}
    assert {"bin", "bout", "vdd", "vss"} <= declared


def test_sanitize_and_escape() -> None:
    assert _sanitize_ident("net-1") == "net_1"
    assert _sanitize_ident("3v3") == "n_3v3"
    assert _sanitize_ident("a.b.c") == "a.b.c"  # dots are legal kernel idents
    assert '"' not in _esc_pred('gain >= "80"')
    assert "\n" not in _esc_pred("line1\nline2")
