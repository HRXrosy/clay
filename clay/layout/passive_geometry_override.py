"""Exact passive override dispatch; preserve the legacy LDO contract separately."""


def validate_passive_geometry_override(label, rec, geometry, *, source_cell=None, tech=None):
    from clay.layout.device_cell import DeviceGeom
    from clay.layout.source_mim_materialization import SourceMimMaterialization, replay_source_mim_materialization

    if type(geometry) is SourceMimMaterialization:
        if label not in ("<attached>", geometry.source_member[0]):
            raise ValueError("source MIM override logical owner differs")
        return replay_source_mim_materialization(geometry, source_cell=source_cell, record=rec, tech=tech)

    if type(geometry) is not DeviceGeom:
        raise TypeError(f"passive geometry override {label!r} must be exact DeviceGeom")
    kind = str(rec.get("kind", "res"))
    orient = str(rec.get("orient", "R0"))
    multiplicity = rec.get("M", 1)
    pins = rec.get("pins")
    fixed = geometry.__dict__.get("fixed_pcell")
    accesses = geometry.__dict__.get("terminal_accesses")
    layers = geometry.__dict__.get("terminal_layers")
    if (
        geometry.kind != kind
        or geometry.orient != orient
        or type(multiplicity) is not int
        or multiplicity <= 0
        or type(pins) is not dict
        or set(geometry.terminals) != {"p", "n"}
        or {role: terminal.net for role, terminal in geometry.terminals.items()}
        != {str(role): str(net) for role, net in pins.items()}
        or type(accesses) is not dict
        or set(accesses) != {"p", "n"}
        or type(layers) is not dict
        or set(layers) != {"p", "n"}
        or type(fixed) is not dict
        or fixed.get("unit_occurrences") != multiplicity
        or fixed.get("position_authority") != "excluded"
        or fixed.get("supported_scope") != "bound_ldo_exact_parameter_key_only"
    ):
        raise ValueError(
            f"passive geometry override {label!r} differs from its source contract"
        )
    if kind == "cap" and geometry.finger_count != multiplicity:
        raise ValueError(
            f"passive geometry override {label!r} cap multiplicity differs"
        )
    from clay.layout.bank_interconnect import replay_bank_interconnect_plan
    plan = rec.get("bank_interconnect_plan")
    plan_data = replay_bank_interconnect_plan(plan).canonical_data() if plan is not None else None
    proof = geometry.__dict__.get("bank_interconnect")
    observed_plan = proof.get("plan") if type(proof) is dict else None
    if fixed.get("bank_interconnect_plan") != plan_data or observed_plan != plan_data:
        raise ValueError(f"passive geometry override {label!r} bank interconnect differs from source")
    return geometry
