"""Netlist half of the consistent-intent extractor.

Parses a reference SPICE / auCdl netlist into ExtractedDevice / ExtractedPassive
and DERIVES topology intent (differential pairs, current-mirror / matched pairs)
mechanically from shared-net + equal-size rules -- never by literal device names.

Then ``correspond_to_gds`` clusters the reference GDS device fingers and maps each
named netlist device to its reference-layout cluster, reporting honest per-device
confidence (exact / inferred / ambiguous).

Public API:
    extract_netlist_intent(netlist_path, cell_name) -> ExtractedIntent
    correspond_to_gds(intent, gds_path) -> dict   # name -> {bbox, confidence, ...}
"""

from __future__ import annotations

import re
from collections import defaultdict

from .types import ExtractedDevice, ExtractedIntent, ExtractedPassive

# --------------------------------------------------------------------------- #
# Unit normalization
# --------------------------------------------------------------------------- #

_SI_SUFFIX = {
    "t": 1e12,
    "g": 1e9,
    "meg": 1e6,
    "x": 1e6,
    "k": 1e3,
    "m": 1e-3,
    "u": 1e-6,
    "n": 1e-9,
    "p": 1e-12,
    "f": 1e-15,
    "a": 1e-18,
}


def _parse_si(token: str) -> float:
    """Parse a SPICE numeric token with optional SI suffix -> float (base units).

    e.g. '21.0u' -> 2.1e-5, '1.5233K' -> 1523.3, '3p' -> 3e-12, '1.65p' -> 1.65e-12.
    A trailing 'meg' takes priority over 'm'. Bare numbers pass through.
    """
    s = token.strip().lower()
    # strip a trailing simple unit word like 'ohm'/'f'/'farad' rarely present
    m = re.match(r"^([+-]?[0-9]*\.?[0-9]+(?:e[+-]?[0-9]+)?)\s*([a-z]*)$", s)
    if not m:
        return float(s)
    val = float(m.group(1))
    suf = m.group(2)
    if not suf:
        return val
    if suf.startswith("meg"):
        return val * 1e6
    # take the first character as the SI multiplier (handles 'k', 'u', 'pf'->p)
    mult = _SI_SUFFIX.get(suf[0])
    if mult is None:
        return val
    return val * mult


def _to_um(metres_value: float) -> float:
    """Convert a length already in base metres to micrometres."""
    return metres_value * 1e6


def _parse_length_um(token: str) -> float:
    """Parse a MOS l=/w= value to micrometres.

    Netlist header declares ``*.SCALE METER`` so bare numbers are metres, but the
    actual device lines here use explicit 'u' suffixes (l=2u w=21.0u). We honour
    the suffix when present; a bare number is treated as metres (SCALE METER) and
    converted to um.
    """
    s = token.strip().lower()
    m = re.match(r"^([+-]?[0-9]*\.?[0-9]+(?:e[+-]?[0-9]+)?)\s*([a-z]*)$", s)
    if not m:
        return float(s) * 1e6
    val = float(m.group(1))
    suf = m.group(2)
    if not suf:
        # bare -> metres per *.SCALE METER -> um
        return _to_um(val)
    mult = _SI_SUFFIX.get(suf[0], 1.0)
    return _to_um(val * mult)


# --------------------------------------------------------------------------- #
# Netlist parsing
# --------------------------------------------------------------------------- #

# Model -> kind. The reference uses 'PD' (p-device, 3.3V pmos) and 'ND' (nmos).
# Rule: model/prefix starting with 'p' is pmos, 'n' is nmos. Kept general.
def _model_kind(model: str) -> str | None:
    m = model.strip().lower()
    if m.startswith("p"):
        return "pmos"
    if m.startswith("n"):
        return "nmos"
    return None


# Model -> gate-oxide type (voltage domain). external_pdk's THICK-gate-oxide 3.3V core
# devices are ``nch3`` / ``pch3`` (TOX 6.8nm vs the 1.8V core's 4.08nm); the signed
# auCdl reference names them ``ND`` / ``PD``, and the codebase's 3.3V device-cell
# convention is ``nmos3v`` / ``pmos3v``. Any of those markers -> 'thick'. The plain
# core devices (``nch`` / ``pch`` / bare ``nmos`` / ``pmos``) are the THIN-oxide 1.8V
# domain -- the SAFE DEFAULT: an unmarked model is treated as thin so a 1.8V design
# is byte-unchanged and never spuriously marked thick.
#
# This DROPPED FACT is the deepest defect the OD2 cure closes: ``_model_kind``
# maps both PD->'pmos' and pch->'pmos', discarding whether the device is 3.3V
# thick-ox. Carrying it here lets the device-cell emit OD2 for a thick device and
# the LVS deck split the gate on OD2 (LVGT/HVGT) so a 3.3V design is not silently
# built + extracted as 1.8V.
_THICK_OX_MARKERS = ("3v", "33", "_3")


def model_oxide(model: str) -> str:
    """Return the gate-oxide domain for a SPICE MOS model token: 'thick' (3.3V) or
    'thin' (1.8V). General marker rule (no per-circuit device-name list):

      * exact ``pd`` / ``nd`` (the auCdl thick-ox reference tokens) -> 'thick';
      * a trailing ``3`` on a core token (``nch3`` / ``pch3``) -> 'thick';
      * a ``3v`` / ``33`` / ``_3`` marker anywhere (``nmos3v`` / ``nch_33``) -> 'thick';
      * everything else (``nch`` / ``pch`` / ``nmos`` / ``pmos``) -> 'thin' (the safe
        default; an unmarked model is the 1.8V core).
    """
    m = str(model or "").strip().lower()
    if not m:
        return "thin"
    if m in ("pd", "nd"):
        return "thick"
    # a trailing '3' distinguishes the 3.3V core (nch3/pch3) from the 1.8V core.
    if m.endswith("3") and not m.endswith("33"):
        return "thick"
    if any(mk in m for mk in _THICK_OX_MARKERS):
        return "thick"
    return "thin"


def _logical_lines(text: str) -> list[str]:
    """Join SPICE '+' continuation lines and drop comments / control lines."""
    raw = text.replace("\r\n", "\n").split("\n")
    out: list[str] = []
    for line in raw:
        if line.startswith("+"):
            if out:
                out[-1] = out[-1] + " " + line[1:].strip()
            continue
        out.append(line)
    return out


def _parse_kv(tokens: list[str]) -> tuple[list[str], dict]:
    """Split a token list into positional tokens and key=value params."""
    pos: list[str] = []
    kv: dict = {}
    for t in tokens:
        if "=" in t:
            k, v = t.split("=", 1)
            kv[k.strip().lower()] = v.strip()
        else:
            pos.append(t)
    return pos, kv


def extract_netlist_intent(netlist_path: str, cell_name: str) -> ExtractedIntent:
    """Parse the reference netlist -> ExtractedIntent (devices/passives + topology).

    Geometric fields are left empty for the geometric sibling. Topology intent
    (diff_pairs / mirror_pairs / matched_pairs) is derived mechanically.
    """
    with open(netlist_path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()

    intent = ExtractedIntent(cell=cell_name)

    for line in _logical_lines(text):
        s = line.strip()
        if not s:
            continue
        if s.startswith("*") or s.startswith(".") or s.startswith("$"):
            # comments, dot-cards (.SUBCKT/.PARAM/.ENDS), stray $ lines
            continue
        tokens = s.split()
        if not tokens:
            continue
        head = tokens[0]
        prefix = head[0].upper()

        if prefix == "M":
            _parse_mos(head, tokens, intent)
        elif prefix in ("R", "C"):
            _parse_passive(head, prefix, tokens, intent)
        # other element types (subckt instances etc.) not present here

    _derive_topology(intent)
    return intent


def _parse_mos(name: str, tokens: list[str], intent: ExtractedIntent) -> None:
    """MMxx d g s b MODEL l=.. w=.. m=.. -> ExtractedDevice."""
    body = tokens[1:]
    pos, kv = _parse_kv(body)
    # positional layout for an auCdl MOS: drain gate source bulk MODEL [params...]
    if len(pos) < 5:
        return  # malformed; skip rather than fake
    drain, gate, source, bulk = pos[0], pos[1], pos[2], pos[3]
    model = pos[4]
    kind = _model_kind(model)
    if kind is None:
        return
    L = _parse_length_um(kv["l"]) if "l" in kv else 0.0
    W = _parse_length_um(kv["w"]) if "w" in kv else 0.0
    M = int(round(_parse_si(kv["m"]))) if "m" in kv else 1
    dev = ExtractedDevice(
        name=name,
        kind=kind,
        W=round(W, 4),
        L=round(L, 4),
        M=M,
        nets={"drain": drain, "gate": gate, "source": source, "body": bulk},
    )
    # PRESERVE the gate-oxide domain the model token carries ('thick' 3.3V vs 'thin'
    # 1.8V). ``kind`` stays 'nmos'/'pmos' (every existing consumer is byte-unchanged);
    # the oxide is carried as an attribute a downstream builder reads via
    # ``getattr(dev, 'oxide', 'thin')`` -> build_device(oxide=...). The default 'thin'
    # keeps any device/path that never sets it 1.8V (byte-identical). It is set here
    # rather than as an ExtractedDevice field so the shared types contract (owned
    # elsewhere) is untouched; the attribute is additive and read-defaulted.
    dev.oxide = model_oxide(model)
    intent.devices.append(dev)


def _parse_passive(name: str, prefix: str, tokens: list[str], intent: ExtractedIntent) -> None:
    """RRxx n1 n2 value [m=N] [$.MODEL=XX] -> ExtractedPassive (value in base SI).

    ``value`` stays PER-UNIT; ``m=`` becomes the parallel multiplicity ``M``
    (default 1) and ``$.MODEL=`` is captured verbatim -- the signed reference
    netlist writes ``CC1 net25 VOUT 830.76f m=2 $.MODEL=M5`` (two parallel
    CTM5-level MIM units)."""
    body = tokens[1:]
    pos, kv = _parse_kv(body)
    if len(pos) < 3:
        return
    n1, n2 = pos[0], pos[1]
    value = _parse_si(pos[2])
    kind = "res" if prefix == "R" else "cap"
    m = int(round(_parse_si(kv["m"]))) if "m" in kv else 1
    model = str(kv.get("$.model", "")).strip()
    intent.passives.append(
        ExtractedPassive(name=name, kind=kind, value=value, nets={"p": n1, "n": n2},
                         M=max(1, m), model=model)
    )


# --------------------------------------------------------------------------- #
# Mechanical topology derivation (general rules, no literal device names)
# --------------------------------------------------------------------------- #

def _size_key(d: ExtractedDevice) -> tuple:
    """Equal-size key: same kind / W / L (M may differ for mirror ratios)."""
    return (d.kind, round(d.W, 3), round(d.L, 3))


# Power-rail vocabulary for the DIFF-PAIR rule only: a pair whose shared source
# is a SUPPLY is common-source/dummy structure, not a differential pair (the
# shared node must be a TAIL). Matched case-insensitively with/without the
# auCdl trailing '!'. Kept local & minimal -- mirrors and matched groups are
# NOT power-filtered (a rail-sourced mirror is real).
_DIFF_POWER_BASES = frozenset({"vdd", "vss", "gnd", "vcc", "vee", "avdd", "avss"})


def _is_power_node(net: str) -> bool:
    low = str(net or "").strip().lower()
    base = low[:-1] if low.endswith("!") else low
    return base in _DIFF_POWER_BASES


def _derive_topology(intent: ExtractedIntent) -> None:
    devs = intent.devices

    # --- differential pairs ---------------------------------------------- #
    # Two MOS of same kind/W/L whose SOURCES tie to the same TAIL node and
    # whose gates are DISTINCT INPUT nets. Precision rules (real-world netlists
    # with dummies, e.g. the signed test4):
    #   * the shared source must NOT be a power rail (rail-sourced equal pairs
    #     are common-source devices / all-tied dummies, not a diff pair);
    #   * a member whose gate ties to its OWN source (cap/diode-connected, the
    #     test4 MM29/MM30 form) is not a diff input;
    #   * degenerate both-diode (gate==drain) pairs stay excluded.
    diff_pairs: list[tuple[str, str]] = []
    seen_pair: set[frozenset] = set()
    for i in range(len(devs)):
        for j in range(i + 1, len(devs)):
            a, b = devs[i], devs[j]
            if _size_key(a) != _size_key(b):
                continue
            if a.nets["source"] != b.nets["source"]:
                continue
            if _is_power_node(a.nets["source"]):
                continue  # shared SUPPLY, not a tail
            if a.nets["gate"] == b.nets["gate"]:
                continue
            if (a.nets["gate"] == a.nets["source"]
                    or b.nets["gate"] == b.nets["source"]):
                continue  # self-gated member: not a differential input
            # exclude degenerate diode-connected (gate==drain) cases.
            if a.nets["gate"] == a.nets["drain"] and b.nets["gate"] == b.nets["drain"]:
                continue
            key = frozenset((a.name, b.name))
            if key in seen_pair:
                continue
            seen_pair.add(key)
            diff_pairs.append((a.name, b.name))
    intent.diff_pairs = diff_pairs

    # --- current mirrors / matched pairs --------------------------------- #
    # Two MOS same kind/W/L sharing a GATE node = mirror (matched). When one of
    # them is diode-connected (gate==drain) it is the reference leg.
    mirror_pairs: list[tuple[str, str]] = []
    matched_pairs: list[tuple[str, str]] = []
    seen_mir: set[frozenset] = set()

    # group by shared gate node
    by_gate: dict[str, list[ExtractedDevice]] = defaultdict(list)
    for d in devs:
        by_gate[d.nets["gate"]].append(d)

    for gate, group in by_gate.items():
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if (a.kind, round(a.L, 3)) != (b.kind, round(b.L, 3)):
                    continue
                # A self-gated member (gate == source: Vgs=0, a fully-off edge
                # DUMMY like MNDUM_LOAD/MNDUM_OUT tied to a rail) is NOT a
                # current-mirror leg -- a real mirror leg gates on a bias node
                # distinct from its source. Two such dummies sharing the rail as
                # their gate node would otherwise be spuriously paired as a
                # "mirror" and forced mirror-symmetric in placement (which, once a
                # cluster is folded wider, makes the shared-axis CP INFEASIBLE).
                # Same exclusion the diff-pair inference already applies above.
                if (a.nets["gate"] == a.nets["source"]
                        or b.nets["gate"] == b.nets["source"]):
                    continue
                # mirror: same gate, same kind, same L (W/M may scale the ratio).
                key = frozenset((a.name, b.name))
                if key in seen_mir:
                    continue
                seen_mir.add(key)
                mirror_pairs.append((a.name, b.name))
                # matched (truly equal-size) subset
                if _size_key(a) == _size_key(b):
                    matched_pairs.append((a.name, b.name))

    intent.mirror_pairs = mirror_pairs
    # matched_pairs = equal-size devices that are mirror-coupled OR diff-paired.
    for p in diff_pairs:
        if frozenset(p) not in {frozenset(m) for m in matched_pairs}:
            matched_pairs.append(p)
    intent.matched_pairs = matched_pairs

    # equal_size_groups: convenience grouping by size key (>=2 members).
    groups: dict[tuple, list[str]] = defaultdict(list)
    for d in devs:
        groups[_size_key(d)].append(d.name)
    intent.equal_size_groups = [sorted(v) for v in groups.values() if len(v) >= 2]


# --------------------------------------------------------------------------- #
# GDS correspondence
# --------------------------------------------------------------------------- #

def _cluster_gds_devices(gds_path: str) -> tuple[list[dict], dict]:
    """Cluster reference-GDS device fingers into device clusters.

    Returns (clusters, labels) where each cluster is a dict with:
        kind, x0,y0,x1,y1, n_fingers, finger_h, xc, yc, band
    and labels maps anchor net names -> (x,y) for any spatially placed pin text.
    """
    import klayout.db as db  # lazy import; klayout.db available in env

    ly = db.Layout()
    ly.read(gds_path)
    top = ly.top_cell()
    dbu = ly.dbu

    def kind_of(cellname: str) -> str | None:
        if cellname.startswith("pmos3v"):
            return "pmos"
        if cellname.startswith("nmos3v"):
            return "nmos"
        if cellname.startswith("mimcap"):
            return "cap"
        if cellname.startswith("poly_res"):
            return "res"
        return None

    fingers: list[dict] = []
    for inst in top.each_inst():
        k = kind_of(inst.cell.name)
        if not k:
            continue
        bb = inst.bbox()
        fingers.append(
            {
                "kind": k,
                "x0": bb.left * dbu,
                "y0": bb.bottom * dbu,
                "x1": bb.right * dbu,
                "y1": bb.top * dbu,
                "xc": (bb.left + bb.right) / 2.0 * dbu,
            }
        )

    # spatially-placed net labels (anchors). Pin layer 41/0 here is placed at the
    # origin (unusable); cap-terminal labels (BOT 44/0, TOP 45/0) are real.
    labels: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for li in ly.layer_indexes():
        info = ly.get_info(li)
        si = top.begin_shapes_rec(li)
        while not si.at_end():
            sh = si.shape()
            if sh.is_text():
                s = sh.text.string
                if not s.startswith("&"):
                    tr = si.trans()
                    labels[s].append((tr.disp.x * dbu, tr.disp.y * dbu))
            si.next()

    # Cluster fingers: group by (kind, banded y0, banded y1), then split on x gaps.
    def band_key(f: dict) -> tuple:
        return (f["kind"], round(f["y0"], 1), round(f["y1"], 1))

    banded: dict[tuple, list[dict]] = defaultdict(list)
    for f in fingers:
        banded[band_key(f)].append(f)

    clusters: list[dict] = []
    for (k, y0, y1), group in banded.items():
        group.sort(key=lambda r: r["xc"])
        run: list[dict] = [group[0]]
        runs: list[list[dict]] = []
        for f in group[1:]:
            if f["x0"] - run[-1]["x1"] > 1.0:  # >1um x-gap -> new cluster
                runs.append(run)
                run = [f]
            else:
                run.append(f)
        runs.append(run)
        for r in runs:
            x0c = min(f["x0"] for f in r)
            x1c = max(f["x1"] for f in r)
            y0c = min(f["y0"] for f in r)
            y1c = max(f["y1"] for f in r)
            clusters.append(
                {
                    "kind": k,
                    "x0": round(x0c, 3),
                    "y0": round(y0c, 3),
                    "x1": round(x1c, 3),
                    "y1": round(y1c, 3),
                    "n_fingers": len(r),
                    "finger_h": round(y1 - y0, 3),
                    "xc": round((x0c + x1c) / 2.0, 3),
                    "yc": round((y0c + y1c) / 2.0, 3),
                    "band": round((y0 + y1) / 2.0, 1),
                }
            )

    return clusters, dict(labels)


def correspond_to_gds(intent: ExtractedIntent, gds_path: str) -> dict:
    """Map each netlist device/passive -> its reference-GDS cluster bbox.

    Fills ``intent.device_bboxes`` (name -> (x0,y0,x1,y1)). Returns a per-device
    report: name -> {'bbox', 'confidence', 'kind', 'n_fingers', 'finger_h',
    'reason'}. Confidence is one of 'exact' / 'inferred' / 'band_only';
    'band_only' marks a device that has no dedicated finger bank (fingers are
    interdigitated/merged in the RF layout) and so gets a defensible band-level
    position from kind+role rather than a faked exact coordinate.

    Idempotent: re-running rebuilds ``intent.device_bboxes`` from scratch.
    """
    clusters, labels = _cluster_gds_devices(gds_path)
    report: dict = {}

    # Self-contained / idempotent: clear any bboxes from a prior call so the
    # group/residual passes below see a clean slate (fixtures reuse one intent).
    intent.device_bboxes.clear()

    used: set[int] = set()  # cluster indices already consumed

    # ---- passives first (cleanest signal) ------------------------------- #
    caps = [d for d in intent.passives if d.kind == "cap"]
    cap_clusters = [(i, c) for i, c in enumerate(clusters) if c["kind"] == "cap"]
    # the single mimcap-array cluster holds n_fingers==len(caps) terminals; the
    # TOP/BOT labels anchor the array. Each netlist cap maps to one mim sub-box.
    if cap_clusters:
        ci, cc = cap_clusters[0]
        used.add(ci)
        # split the mim array along x into len(caps) equal sub-bboxes ordered L->R
        # (the layout shows two abutted mim tiles for the two caps).
        n = max(len(caps), 1)
        span = (cc["x1"] - cc["x0"]) / n
        for idx, cap in enumerate(sorted(caps, key=lambda d: d.value, reverse=True)):
            bx0 = round(cc["x0"] + idx * span, 3)
            bx1 = round(cc["x0"] + (idx + 1) * span, 3)
            bbox = (bx0, cc["y0"], bx1, cc["y1"])
            intent.device_bboxes[cap.name] = bbox
            report[cap.name] = {
                "bbox": bbox,
                "confidence": "inferred",
                "kind": "cap",
                "n_fingers": cc["n_fingers"],
                "finger_h": cc["finger_h"],
                "reason": "mimcap array anchored by TOP/BOT labels; split L->R by value",
            }

    res_devs = [d for d in intent.passives if d.kind == "res"]
    res_clusters = [(i, c) for i, c in enumerate(clusters) if c["kind"] == "res"]
    # 2 resistors, 2 poly_res clusters: map by x order (deterministic, unambiguous count).
    if len(res_devs) == len(res_clusters) and res_clusters:
        rc_sorted = sorted(res_clusters, key=lambda t: t[1]["xc"])
        # order netlist resistors by value for a stable, reproducible assignment.
        rd_sorted = sorted(res_devs, key=lambda d: d.value)
        for (rd), (ci, cc) in zip(rd_sorted, rc_sorted):
            used.add(ci)
            bbox = (cc["x0"], cc["y0"], cc["x1"], cc["y1"])
            intent.device_bboxes[rd.name] = bbox
            report[rd.name] = {
                "bbox": bbox,
                "confidence": "inferred",
                "kind": "res",
                "n_fingers": cc["n_fingers"],
                "finger_h": cc["finger_h"],
                "reason": "poly_res cluster matched by x-order (count parity 2==2)",
            }

    # ---- MOS devices: group-by-(kind, finger_height) + distribute-M -------- #
    # This RF layout folds/interdigitates fingers, so a logical M=1 device does
    # NOT get its own cluster: several W-equal devices share a finger bank whose
    # finger HEIGHT (per-finger drawn W) is the stable cue, while total finger
    # COUNT is the sum of the members' M (folding splits each M into fingers).
    # General rule (no MM-name special-casing):
    #   1. group devices by (kind, W); group clusters by (kind, finger_h).
    #   2. match device-W-groups to cluster-fh-groups by best mutual rank
    #      (finger-conservation: sum(M) ~ sum(n_fingers); ties by W<->fh).
    #   3. inside a matched group, distribute the devices over the group's
    #      clusters: split each bank's finger-count proportionally to M, carve
    #      x-sub-regions, and order twins by their mirror/diff pairing.
    #   4. residual devices with no group get a BAND-LEVEL position from
    #      kind+role (diode-reference vs mirror-leg) -- 'band_only', not faked.
    mos_devs = [d for d in intent.devices]
    dev_by_name = {d.name: d for d in mos_devs}

    # twin partner map (mirror or diff) for symmetric ordering / role inference.
    twin_of: dict[str, str] = {}
    for a, b in list(intent.mirror_pairs) + list(intent.diff_pairs):
        twin_of.setdefault(a, b)
        twin_of.setdefault(b, a)

    def is_diode(d: ExtractedDevice) -> bool:
        return d.nets["gate"] == d.nets["drain"]

    for kind in ("nmos", "pmos"):
        kdevs = [d for d in mos_devs if d.kind == kind]
        kclusters = [
            (i, c)
            for i, c in enumerate(clusters)
            if c["kind"] == kind and i not in used
        ]
        if not kdevs:
            continue

        # --- group devices by W, clusters by finger height ------------------ #
        dev_groups: dict[float, list[ExtractedDevice]] = defaultdict(list)
        for d in kdevs:
            dev_groups[round(d.W, 1)].append(d)
        clu_groups: dict[float, list[tuple[int, dict]]] = defaultdict(list)
        for i, c in kclusters:
            # bucket 10.8 / 10.9 etc. together (per-finger heights vary <1um)
            clu_groups[round(c["finger_h"])].append((i, c))

        # --- match W-groups to fh-groups (greedy, finger-conservation first) - #
        # candidate score: prefer groups whose total fingers ~ total M, then
        # whose representative finger_h is closest to W (both robust to folding).
        dev_keys = sorted(dev_groups, reverse=True)        # largest W first
        clu_keys = sorted(clu_groups, reverse=True)        # largest fh first
        pair_scores = []
        for wk in dev_keys:
            sm = sum(d.M for d in dev_groups[wk])
            for fk in clu_keys:
                nf = sum(c["n_fingers"] for _, c in clu_groups[fk])
                # finger-conservation error (how close sumM is to sumNf, scaled)
                cons = abs(nf - sm) / max(sm, 1)
                # height proximity (W vs finger height); RF folding keeps fh<=W
                prox = abs(fk - wk) / max(wk, 1.0)
                pair_scores.append((cons + 0.5 * prox, wk, fk))
        pair_scores.sort(key=lambda t: t[0])

        wk_taken: set[float] = set()
        fk_taken: set[float] = set()
        group_match: dict[float, float] = {}   # W-key -> fh-key
        for _sc, wk, fk in pair_scores:
            if wk in wk_taken or fk in fk_taken:
                continue
            wk_taken.add(wk)
            fk_taken.add(fk)
            group_match[wk] = fk

        # --- distribute each matched W-group over its fh-group clusters ------ #
        for wk, devs in dev_groups.items():
            fk = group_match.get(wk)
            if fk is None:
                continue  # handled by the band-level residual pass below
            group_clusters = [c for _, c in clu_groups[fk]]
            ci_list = [i for i, _ in clu_groups[fk]]
            for i in ci_list:
                used.add(i)
            # order clusters and devices for a stable, symmetric assignment.
            group_clusters_sorted = sorted(
                group_clusters, key=lambda c: (c["band"], c["xc"])
            )
            # devices: diode-reference leg first (it anchors the mirror), then by
            # descending M (larger device -> larger / earlier bank), then name.
            devs_sorted = sorted(
                devs, key=lambda d: (not is_diode(d), -d.M, d.name)
            )

            n_dev = len(devs_sorted)
            n_clu = len(group_clusters_sorted)

            if n_dev <= n_clu:
                # enough clusters: assign one (or the first n_dev) cluster each.
                # round-robin so twins land on adjacent banks (mirror symmetry).
                for k, d in enumerate(devs_sorted):
                    c = group_clusters_sorted[k % n_clu]
                    bbox = (c["x0"], c["y0"], c["x1"], c["y1"])
                    intent.device_bboxes[d.name] = bbox
                    conf = "inferred"
                    reason = (
                        f"W-group W={wk} -> fh-group fh~={fk}: {n_dev} dev / "
                        f"{n_clu} bank(s); device #{k} -> bank band={c['band']}"
                    )
                    report[d.name] = {
                        "bbox": bbox,
                        "confidence": conf,
                        "kind": d.kind,
                        "n_fingers": c["n_fingers"],
                        "finger_h": c["finger_h"],
                        "reason": reason,
                    }
            else:
                # more devices than banks: attribute finger STRIPS to devices by
                # matching strip count to each device's M. The GDS holds the real
                # single-finger instances; a device with M fingers must own M of
                # them. We do this in two steps:
                #   1. BIN-PACK whole devices into banks by finger CAPACITY (each
                #      bank holds n_fingers fingers). Walk devices LARGEST-M first
                #      and give each the bank whose REMAINING capacity most TIGHTLY
                #      fits its M (smallest remaining >= M); fall back to the bank
                #      with the most room only when none can still hold it. Tight-
                #      fit matching sends an M=16 device to the 30-finger bank and
                #      an M=4 device to a 6-finger bank -- a sound finger-
                #      conservation assignment that keeps a device's fingers
                #      CONTIGUOUS inside ONE bank (not smeared across non-adjacent
                #      banks) and does NOT cram the mirror's diode reference plus
                #      both scaled legs into a single shared band.
                #   2. inside each bank, carve contiguous x-strips proportional to
                #      the M (== finger count) of the devices assigned there.
                # The slice width is therefore MONOTONIC in M (a larger-M device
                # gets a proportionally wider strip), unlike the old concatenated
                # width-line mapping that handed an M=16 device a NARROWER slice
                # than an M=8 one because it mixed banks of different finger
                # density onto one coordinate axis.
                banks = sorted(
                    group_clusters_sorted, key=lambda c: (-c["n_fingers"], c["xc"])
                )
                remaining = {id(c): c["n_fingers"] for c in banks}
                assigned: dict[int, list] = {id(c): [] for c in banks}
                # largest device first so it claims the bank sized to it; ties by
                # name for determinism (diode-vs-leg role no longer drives the bank
                # choice -- the finger-count match does).
                for d in sorted(devs_sorted, key=lambda d: (-d.M, d.name)):
                    fit = [c for c in banks if remaining[id(c)] >= d.M]
                    if fit:
                        # tightest fit: smallest remaining capacity that holds M
                        # (ties by xc for a stable, left-biased choice).
                        host = min(fit, key=lambda c: (remaining[id(c)], c["xc"]))
                    else:
                        # none can hold it: best-effort to the bank with most room
                        # (the plausibility guard below still flags any genuine
                        # under-capture rather than silently passing it).
                        host = max(banks, key=lambda c: (remaining[id(c)], -c["xc"]))
                    assigned[id(host)].append(d)
                    remaining[id(host)] -= d.M
                for c in banks:
                    bank_devs = assigned[id(c)]
                    if not bank_devs:
                        continue
                    # carve left->right by M; order by descending M then name so
                    # the carve is deterministic and the widest device leads.
                    bank_devs_sorted = sorted(
                        bank_devs, key=lambda d: (-d.M, d.name)
                    )
                    bank_total_m = sum(d.M for d in bank_devs_sorted) or len(
                        bank_devs_sorted
                    )
                    x0b, y0b, x1b, y1b = c["x0"], c["y0"], c["x1"], c["y1"]
                    span = x1b - x0b
                    cursor = 0.0  # cumulative M consumed WITHIN this bank
                    n_bank = len(bank_devs_sorted)
                    for d in bank_devs_sorted:
                        x_lo = x0b + span * (cursor / bank_total_m)
                        cursor += d.M
                        x_hi = x0b + span * (cursor / bank_total_m)
                        bbox = (round(x_lo, 3), y0b, round(x_hi, 3), y1b)
                        intent.device_bboxes[d.name] = bbox
                        # honest per-device finger estimate: this device's share of
                        # the bank's fingers (M-proportional, integer-rounded).
                        dev_fingers = int(round(c["n_fingers"] * d.M / bank_total_m))
                        report[d.name] = {
                            "bbox": bbox,
                            "confidence": "inferred",
                            "kind": d.kind,
                            "n_fingers": dev_fingers,
                            "finger_h": c["finger_h"],
                            "reason": (
                                f"W-group W={wk} -> fh-group fh~={fk}: {n_dev} dev / "
                                f"{n_clu} bank(s); finger-strip carve: M={d.M} of "
                                f"{bank_total_m} fingers in bank band={c['band']} "
                                f"({n_bank} dev sharing this bank)"
                            ),
                        }

    # --- band-level residual: any device still without a bbox -------------- #
    # Assign a defensible TIER position from kind + role rather than None.
    # Reference frame: nmos sit low (small y), pmos high; the diode-reference
    # leg of a mirror anchors its band, mirror legs offset around it. This is a
    # band-only position (correct tier, honest about sub-position uncertainty).
    placed_y = [
        (intent.device_bboxes[d.name][1] + intent.device_bboxes[d.name][3]) / 2.0
        for d in mos_devs
        if d.name in intent.device_bboxes
    ]
    for d in mos_devs:
        if d.name in intent.device_bboxes:
            continue
        # band y: median of same-kind placed devices, else a kind-default tier.
        same = [
            (intent.device_bboxes[o.name][1] + intent.device_bboxes[o.name][3]) / 2.0
            for o in mos_devs
            if o.kind == d.kind and o.name in intent.device_bboxes
        ]
        if same:
            yb = sorted(same)[len(same) // 2]
        elif placed_y:
            yb = sorted(placed_y)[len(placed_y) // 2]
        else:
            yb = 10.0 if d.kind == "nmos" else 60.0
        # x: nudge mirror twins apart so they don't collapse onto one point.
        partner = twin_of.get(d.name)
        px = None
        if partner and partner in intent.device_bboxes:
            pb = intent.device_bboxes[partner]
            px = (pb[0] + pb[2]) / 2.0
        x_center = px if px is not None else 5.0
        # half a finger-height as a nominal device half-extent.
        hx = 2.0
        hy = 2.0
        bbox = (
            round(x_center - hx, 3),
            round(yb - hy, 3),
            round(x_center + hx, 3),
            round(yb + hy, 3),
        )
        intent.device_bboxes[d.name] = bbox
        report[d.name] = {
            "bbox": bbox,
            "confidence": "band_only",
            "kind": d.kind,
            "n_fingers": None,
            "finger_h": None,
            "reason": (
                "no dedicated finger bank (fingers interdigitated/merged in the RF "
                f"layout); band-level position from kind={d.kind}+role "
                f"({'diode-ref' if is_diode(d) else 'mirror-leg'})"
            ),
        }

    # ---- plausibility guard: every MOS bbox must hold its channel ---------- #
    # The RF folding heuristics above can hand a device a bbox that is
    # PHYSICALLY too small to contain its drawn channel. Two observed cases on
    # the external_pdk reference:
    #   * a large folded device (e.g. M=16) carved an M-proportional x-slice of a
    #     shared finger bank whose x-span was not proportional to total M, so the
    #     slice area fell below the device gate area.
    #   * a lone small device (e.g. an M=1 diode leg / MOScap) that won no
    #     finger-bank group fell into the 4x4 band-only residual stub.
    # An undersized corresponded bbox means the device was MIS-ATTRIBUTED in the
    # interdigitated RF layout -- its true extent was not recovered. We FLAG this
    # honestly (bbox_source='implausible_unresolved') and DELIBERATELY DO NOT
    # fabricate a replacement bbox: injecting the single-row physical extent at
    # the (equally unreliable) corresponded centroid yields a wide footprint that
    # OVERLAPS real neighbours -- geometry the silicon does not contain. Verified
    # on this reference: the raw correspondence is overlap-free except the one
    # documented MM15/MM16 interdigitation, whereas the injected extents fabricate
    # MM20/MM21 and CR1/MM16 overlaps. The physical extent is recorded as ADVISORY
    # metadata only (advisory_physical_extent_um); the bbox is left as corresponded.
    for d in mos_devs:
        report.setdefault(d.name, {})
        report[d.name].setdefault("bbox_source", "corresponded")
        bb = intent.device_bboxes.get(d.name)
        if bb is None:
            continue
        gate_area = d.W * d.L * d.M
        bbox_area = abs((bb[2] - bb[0]) * (bb[3] - bb[1]))
        if bbox_area + 1e-9 >= gate_area:
            continue
        report[d.name]["bbox_source"] = "implausible_unresolved"
        phys = _physical_extent_um(d)
        if phys is None:
            # cannot even build a physical cell: flag, never silently report an
            # undersized bbox as correct.
            report[d.name]["plausibility"] = (
                f"corresponded area {bbox_area:.2f} um2 < gate area "
                f"{gate_area:.2f} um2 and no physical cell available"
            )
            continue
        pw, ph = phys
        report[d.name]["advisory_physical_extent_um"] = (round(pw, 3), round(ph, 3))
        report[d.name]["plausibility"] = (
            f"corresponded area {bbox_area:.2f} um2 < gate area {gate_area:.2f} "
            f"um2; correspondence unreliable (interdigitated/MOScap in the RF "
            f"layout). Physical single-row extent would be {pw:.2f}x{ph:.2f} um but "
            f"is NOT injected -- its width/centroid would fabricate neighbour "
            f"overlaps absent from the silicon; bbox left as corresponded + flagged."
        )

    # Disambiguate diff/mirror twins by left/right position. For each diff/mirror
    # pair whose two devices map to clusters, order them L->R for a stable twin id.
    _refine_twin_positions(intent, report)

    return report


def _physical_extent_um(d: ExtractedDevice) -> tuple[float, float] | None:
    """Real (width_um, height_um) of device ``d`` from the PDK cell builder.

    Builds the m=M MOS via ``clay.layout.device_cell.build_device`` on the
    external_pdk tech and returns its drawn extent in micrometres. Returns None if
    the builder is unavailable so the caller can flag the device honestly rather
    than fabricate a size. Geometry is in integer nanometres -> /1000 to um.
    """
    try:
        from clay.layout.device_cell import build_device
        from clay.layout.tech import LayoutTech

        tech = LayoutTech.resolve("gf180mcu")
        geom = build_device(tech, d.kind, d.W * 1e-6, d.L * 1e-6, nf=max(d.M, 1))
        x0, y0, x1, y1 = geom.extent()
    except Exception:
        return None
    return ((x1 - x0) / 1000.0, (y1 - y0) / 1000.0)


def _frac_to_x(banks: list[dict], widths: list[float], total_w: float, frac: float) -> float:
    """Map a fraction [0,1] of total bank width onto the concatenated x-line."""
    target = max(0.0, min(1.0, frac)) * total_w
    acc = 0.0
    for c, w in zip(banks, widths):
        if target <= acc + w or c is banks[-1]:
            local = target - acc
            return c["x0"] + local
        acc += w
    return banks[-1]["x1"]


def _bank_containing(banks: list[dict], x: float) -> dict:
    """Return the bank whose x-extent contains x (nearest by xc as fallback)."""
    for c in banks:
        if c["x0"] <= x <= c["x1"]:
            return c
    return min(banks, key=lambda c: abs(c["xc"] - x))


def clusters_enum(clusters: list[dict]):
    for i, c in enumerate(clusters):
        yield i, c


def _refine_twin_positions(intent: ExtractedIntent, report: dict) -> None:
    """For each diff/mirror pair, if both twins are matched, tag left/right by xc.

    This resolves the symmetric-twin ambiguity (e.g. the two M=4 input devices)
    using reference-layout x-position, exactly as a placer would re-derive it.
    """
    def xc(name: str) -> float | None:
        bb = intent.device_bboxes.get(name)
        if bb is None:
            return None
        return (bb[0] + bb[2]) / 2.0

    for a, b in list(intent.diff_pairs) + list(intent.mirror_pairs):
        xa, xb = xc(a), xc(b)
        if xa is None or xb is None:
            continue
        left, right = (a, b) if xa <= xb else (b, a)
        for nm, side in ((left, "left"), (right, "right")):
            if nm in report:
                report[nm]["twin_side"] = side
