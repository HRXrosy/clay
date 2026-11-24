"""Native FEOL domains, separate from emitted routing conductor inventory.

The rectangles below are component bounding boxes of actual source polygons or
explicit foundry Boolean regions. They are conservative placement operands, not
invented GDS artwork. Exact integer displacement unions preserve their cavities
between components without expanding an entire solver master.
"""
from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
import hashlib
from pathlib import Path
import re


NATIVE_FEOL_PREDICATE_ID = "clay.native-feol-local-domain-separation.v1"


@dataclass(frozen=True, slots=True)
class NativeFeolPairIR:
    left_stamp: str
    right_stamp: str
    left_master: str
    right_master: str
    left_kind: str
    right_kind: str
    left_layers: tuple[tuple[int, int], ...]
    right_layers: tuple[tuple[int, int], ...]
    left_rectangles_nm: tuple[tuple[int, int, int, int], ...]
    right_rectangles_nm: tuple[tuple[int, int, int, int], ...]
    gap_nm: int
    rule_id: str
    rule_source: str
    rule_sha256: str

    def __post_init__(self):
        from .edge_rail_reservation import _box, _integer, _layer, _require, _sha, _text
        for name in ("left_stamp", "right_stamp", "left_master", "right_master",
                     "left_kind", "right_kind", "rule_id", "rule_source"):
            _text(getattr(self, name), "native FEOL " + name)
        _require(self.left_stamp < self.right_stamp and self.left_master != self.right_master,
                 "native FEOL requires distinct canonical stamps and original masters")
        _sha(self.rule_sha256, "native FEOL rule source")
        _integer(self.gap_nm, "native FEOL clearance")
        for side in ("left", "right"):
            layers = getattr(self, side + "_layers")
            _require(type(layers) is tuple and bool(layers)
                     and layers == tuple(sorted(set(layers))), "native FEOL layer inventory is malformed")
            for layer in layers:
                _layer(layer)
            boxes = getattr(self, side + "_rectangles_nm")
            _require(type(boxes) is tuple and 0 < len(boxes) <= 65536
                     and boxes == tuple(sorted(set(boxes))), "native FEOL rectangles must be canonical")
            for box in boxes:
                _box(box, "native FEOL component bbox")


@lru_cache(maxsize=8)
def _foundry_rules(path, mtime_ns, size):
    """Parse the actual local rule expressions, retaining their immutable hash."""
    data = Path(path).read_bytes()
    source = data.decode("utf-8")

    def nm(pattern):
        matches = re.findall(pattern, source, re.MULTILINE)
        if len(matches) != 1:
            raise ValueError("Native FEOL foundry rule expression is missing or ambiguous: " + pattern)
        value = Decimal(matches[0]) * 1000
        if value != value.to_integral_value() or value <= 0:
            raise ValueError("Native FEOL foundry clearance is not positive integer nm")
        return int(value)

    variables = dict(SB3="sb3_l1", SB4="sb4_l1", SB5a="sb5a_l1",
                     SB5b="sb5b_l1", SB15a="sb15a_l1", NP7="np7_l1", LRES7="lres7_l1")
    values = {key: nm(r"^" + var + r"\s*=.*?,\s*([0-9.]+)\.um")
              for key, var in variables.items()}
    values["SB2"] = nm(r"^sb2_l1\s*=.*?\.space\(([0-9.]+)\.um")
    values["NW33"] = nm(r"^nw2b_l1\s*=\s*nwell\.isolated\(([0-9.]+)\.um, euclidian\).*not_interacting\(dualgate\)")
    values["NW5"] = nm(r"^nw2b_l1\s*=\s*nwell\.isolated\(([0-9.]+)\.um, euclidian\).*overlapping\(dualgate\)")
    # This zero-overlap operand protects the exact recognition expression,
    # rather than assigning an invented numeric spacing to RES_MK.
    if "mk_lres9 = res_mk.edges.not(poly2.and(nplus).and(sab).edges).inside_part(poly2)" not in source:
        raise ValueError("Native FEOL resistor recognition expression changed")
    return values, hashlib.sha256(data).hexdigest()


def foundry_rules(tech):
    if getattr(tech, "pdk_id", None) != "gf180mcu":
        return None
    selected = tech.klayout_decks().get("drc")
    if selected is None:
        raise FileNotFoundError("Native FEOL requires the DRC deck from the selected GF180 PDK install")
    candidate = Path(selected)
    stat = candidate.stat()
    rules, digest = _foundry_rules(str(candidate.resolve()), stat.st_mtime_ns, stat.st_size)
    return rules, str(candidate.resolve()), digest


def _regions(stamp, position, delta, keys):
    import klayout.db as kdb
    from clay.layout.placed_geometry import placed_point_nm
    result = {}
    for name, key in keys.items():
        region = kdb.Region()
        if key is not None:
            for ring in stamp.polys_by_layer.get(key, ()):
                points = [placed_point_nm(float(x), float(y), position) for x, y in ring]
                polygon = kdb.Polygon([kdb.Point(x + delta[0], y + delta[1]) for x, y in points])
                if polygon.area() <= 0:
                    raise ValueError("Native FEOL source polygon has no physical area")
                region.insert(polygon)
        result[name] = region.merged()
    return result


def _boxes(region):
    return tuple(sorted(set((box.left, box.bottom, box.right, box.top)
                            for polygon in region.each() for box in (polygon.bbox(),))))


def _sky130_lvt_normal_row_envelope(*, bundle, cell, stamps, geometry, placement_um,
                                    row_kinds, row_program, row_materialization, tech,
                                    first, second, pair, magic, magic_text, mr_text,
                                    source_output_extension=False, source_output_replay=None):
    """Prove a conservative source-bound separation for two N row stamps.

    The complete stamp envelopes are separated by the greatest SKY130 LVT
    spacing applicable to these ordinary masks.  Unlike per-layer clearance,
    this also keeps foreign poly, taps, contacts and the LVT selector apart.
    An unproved pair receives no exception and keeps the caller's fallback.
    """
    if (row_kinds is None or row_program is None or row_materialization is None
            or getattr(tech, 'pdk_id', None) != 'sky130'):
        return None
    entries = {entry.group: entry for entry in row_program.entries}
    names = (first.stamp_id, second.stamp_id)
    if any(name not in entries or row_kinds.get(name) != 'nmos' for name in names):
        return None
    devices = {device.instance_name: device for device in getattr(cell, 'devices', ())}
    if len(devices) != len(getattr(cell, 'devices', ())):
        raise ValueError('native FEOL source device identity is duplicated')
    from clay.pdk.sky130_native_geometry import native_geometry_spec
    models = []
    for name in names:
        owners = entries[name].consumed_devices
        if not owners or any(owner not in devices for owner in owners):
            raise ValueError('native FEOL same-polarity row source ownership differs')
        models.append({native_geometry_spec(devices[owner])['model'] for owner in owners})
    if {frozenset(models[0]), frozenset(models[1])} != {
            frozenset(('sky130_fd_pr__nfet_01v8',)),
            frozenset(('sky130_fd_pr__nfet_01v8_lvt',))}:
        return None
    allowed = {(65, 20), (65, 44), (66, 20), (66, 44), (67, 20),
               (67, 44), (68, 20), (93, 44), (94, 20), (95, 20), (125, 44)}
    if source_output_extension:
        from .native_feol_output_profile import EXTRA, admits
        replayed=(source_output_replay(first,second) if source_output_replay is not None else
                  admits(bundle=bundle,cell=cell,stamps=stamps,geometry=geometry,
                         placement_um=placement_um,row_program=row_program,
                         row_materialization=row_materialization,tech=tech,
                         first=first,second=second,magic_text=magic_text))
        if replayed!=source_output_extension:
            raise ValueError('native FEOL output extension proof differs')
        allowed |= EXTRA
    if any(set(geometry[name].polys_by_layer) - allowed for name in names):
        return None
    # Verify the complete row program and native source partition again at
    # this trust boundary, including on final post-merge replay.
    proven = sky130_native_row_kinds(bundle=bundle, cell=cell, stamps=stamps,
        geometry=geometry, row_program=row_program,
        row_materialization=row_materialization, tech=tech)
    if any(proven.get(name) != 'nmos' for name in names):
        raise ValueError('native FEOL same-polarity row proof differs')
    magic_checks = (
        r'(?m)^\s*allndifflv\s+\*ndif,\*nsd,\*ndiode,ndiffres,nfet,npass,npd,scnfet,scnfetlvt,nfetlvt,nsonos\s*$',
        r'(?m)^\s*allpdifflv\s+\*pdif,\*psd,\*pdiode,pdiffres,pfet,ppu,scpfet,scpfethvt,pfetlvt,pfetmvt,pfethvt\s*$',
        r'(?m)^\s*alldifflv\s+allndifflv,allpdifflv\s*$',
        r'spacing\s+pfetlvt,nfetlvt,pdiodelvt,ndiodelvt\s*\\\s*allfetsnolvt\s+360\s+touching_illegal\s*\\\s*"Min\. spacing between FET and LVTN < %d \(lvtn\.3a\)"',
        r'edge4way\s+allfetsnolvt\s+allactivenonfet\s+415\s*\\\s*~\(pfetlvt,nfetlvt,pdiodelvt,ndiodelvt\)/a\s+allfetsnolvt\s+415\s*\\\s*"Min\. spacing between FET and LVTN in S/D direction < %d \(lvtn\.3b\)"',
        r'spacing\s+alldifflv,var,varhvt,corenvar,corepvar,fomfill\s*\\\s*alldifflv,var,varhvt,corenvar,corepvar,fomfill\s+270\s+touching_ok',
        r'spacing\s+allpoly,polyfill\s+allpoly,polyfill\s+210\s+touching_ok',
        r'spacing\s+allpolynonfet,polyfill\s*\\\s*\*ndiff,\*mvndiff,\*ndiode,\*nndiode,ndiffres,\*ndiodelvt,\*pdiff,\*mvpdiff,\*pdiode,pdiffres,\*pdiodelvt,\*pdiodehvt\s*\\\s*75\s+corner_ok\s+allfets',
        r'spacing\s+allpolynonfet,polyfill\s+\*psd\s+55\s+corner_ok\s+corepvar',
        r'edge4way\s+\*psd\s+\*ndiff\s+300\s+~\(nfet,npass,npd,scnfet,scnfetlvt,nfetlvt,nsonos\)/a\s+\*psd\s+300',
        r'spacing\s+allndiffcont\s+allndiffcont\s+170\s+touching_ok',
        r'spacing\s+allpdiffcont\s+allndiffcont\s+170\s+touching_illegal',
        r'spacing\s+pc\s+alldiff\s+190\s+touching_illegal',
        r'spacing\s+pc\s+allpdifflv,allpdiffmv\s+235\s+touching_illegal',
        r'spacing\s+\*locali,rli\s+\*locali,rli,\*obsli\s+170\s+touching_ok',
        r'spacing\s+mcon/m1\s+mcon/m1,obsmcon/m1\s+190\s+touching_ok',
        r'widespacing\s+allm1\s+3005\s+allm1,\*obsm1,m1fill\s+280\s+touching_ok',
        r'cifspacing\s+m1_large_halo\s+m1_large_halo\s+280\s+touching_ok',
    )
    if not all(re.search(pattern, magic_text) for pattern in magic_checks):
        raise ValueError('native FEOL same-polarity SKY130 deck authority differs')
    mapping_checks = (
        r'(?m)^diff\s*=\s*input\(65,\s*20\)', r'(?m)^tap\s*=\s*polygons\(65,\s*44\)',
        r'(?m)^poly\s*=\s*polygons\(66,\s*20\)', r'(?m)^licon\s*=\s*polygons\(66,\s*44\)',
        r'(?m)^li_wildcard\s*=\s*"67/20"\s*$', r'(?m)^mcon_wildcard\s*=\s*"67/44"\s*$',
        r'(?m)^m1_wildcard\s*=\s*"68/20"\s*$', r'(?m)^nsdm_wildcard\s*=\s*"93/44"\s*$',
        r'(?m)^psdm_wildcard\s*=\s*"94/20"\s*$', r'(?m)^npc\s*=\s*polygons\(95,\s*20\)',
        r'(?m)^nsdm\s*=\s*polygons\(nsdm_wildcard\)', r'(?m)^psdm\s*=\s*polygons\(psdm_wildcard\)',
        r'(?m)^li\s*=\s*polygons\(li_wildcard\)', r'(?m)^mcon\s*=\s*polygons\(mcon_wildcard\)',
        r'(?m)^m1\s*=\s*polygons\(m1_wildcard\)',
        r'(?m)^lvtn\s*=\s*polygons\(125,\s*44\)',
        r'(?m)^\s*nsdm\s*\.not\(areaid_ce\)\s*\.space\(0\.38,\s*euclidian\)\s*\.output\("nsdm\.1"',
        r'(?m)^\s*psdm\s*\.not\(areaid_ce\)\s*\.space\(0\.38,\s*euclidian\)\s*\.output\("psdm\.1"',
        r'npc\.space\(0\.27,\s*euclidian\)\.output\(\s*"npc\.2"',
        r'lvtn\.space\(0\.38,\s*euclidian\)\.output\(',
        r'poly\s*\.not\(areaid_ce\)\s*\.space\(0\.21,\s*euclidian\)\s*\.output\("poly\.2"',
        r'mcon\.space\(0\.19,\s*euclidian\)\.output\(\s*"ct\.2"',
        r'huge_m1\.separation\(non_huge_m1,\s*0\.28,\s*euclidian\)',
    )
    if not all(re.search(pattern, mr_text) for pattern in mapping_checks):
        raise ValueError('native FEOL same-polarity layer/deck mapping differs')
    # Both stamp digests bind all actual native mask polygons.  Full bboxes
    # are conservative, so 415 nm separates all of those polygons, including
    # different layers, without inventing or trimming any PDK artwork.
    from .edge_rail_geometry import _stamp_geometry
    for stamp in (first, second):
        actual_box, digest = _stamp_geometry(geometry[stamp.stamp_id], placement_um[stamp.stamp_id])
        if digest != stamp.geometry_sha256:
            raise ValueError('native FEOL source stamp changed')
        if stamp.bbox_nm[:2] != (0, 0) or (
                stamp.bbox_nm[2] - stamp.bbox_nm[0], stamp.bbox_nm[3] - stamp.bbox_nm[1]) != (
                actual_box[2] - actual_box[0], actual_box[3] - actual_box[1]):
            raise ValueError('native FEOL same-polarity source stamp bounds differ')
    left, right = sorted((first, second), key=lambda stamp: stamp.stamp_id)
    rule_id='sky130:lvtn.3b-conservative-envelope'
    row = NativeFeolPairIR(left.stamp_id, right.stamp_id, left.master_id, right.master_id,
        'nmos-envelope', 'nmos-envelope', tuple(sorted(geometry[left.stamp_id].polys_by_layer)),
        tuple(sorted(geometry[right.stamp_id].polys_by_layer)), (left.bbox_nm,), (right.bbox_nm,),
        415, rule_id, str(magic),
        hashlib.sha256(magic.read_bytes()).hexdigest())
    common = set(geometry[first.stamp_id].polys_by_layer) & set(geometry[second.stamp_id].polys_by_layer)
    return row, {(pair[0], pair[1], layer): 415 for layer in common}


def compile_sky130_native_feol_pairs(*, bundle, cell, stamps, geometry, placement_um,
                                    eligible_master_pairs, row_kinds=None,
                                    eligible_stamp_pairs=None, row_program=None,
                                    row_materialization=None, tech=None):
    """Bind distinct native P/N stamps to the copied, validated PDK decks.

    The same-layer allowances apply only to the listed original master pairs;
    every other collision layer and master pair retains its prior clearance.
    These are conservative component bboxes, not a substitute for PDK DRC.
    """
    bundle.require_current(cell)
    magic = Path(bundle.output_dir) / 'sky130A.tech'
    mr = Path(bundle.output_dir) / 'sky130A_mr.drc'
    for path in (magic, mr):
        if not path.is_file() or bundle.common_artifacts.get(path.name) != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError('native FEOL requires source-bound copied SKY130 decks')
    magic_text, mr_text = magic.read_text(), mr.read_text()
    if not re.search(r'spacing\s+\*ndiff,\*ndiode,nfet\s+allnwell\s+340\s+touching_illegal\s*\\\s*"N-Diffusion spacing to N-well < %d \(diff/tap\.9\)"', magic_text):
        raise ValueError('native FEOL SKY130 diff/tap.9 authority differs')
    checks = (
        r'licon_peri\.not\(prec_resistor\)\.space\(0\.17,\s*euclidian\)',
        r'li_outside_or_touching_areaidce\.space\(0\.17,\s*euclidian\)',
        r'not_sram_nsdm\s*\.not\(areaid_ce\)\s*\.space\(0\.38,\s*euclidian\)\s*\.output\("nsdm\.1"',
        r'\.not\(areaid_ce\)\s*\.space\(0\.38,\s*euclidian\)\s*\.output\("psdm\.1"',
        r'\.space\(0\.27,\s*euclidian\)\s*\.output\(\s*"npc\.2"',
    )
    if not all(re.search(pattern, mr_text) for pattern in checks):
        raise ValueError('native FEOL SKY130 same-layer deck authority differs')
    if not (re.search(r'(?m)^nwell_wildcard\s*=\s*"64/20"\s*$', mr_text) and
            re.search(r'(?m)^nwell\s*=\s*polygons\(nwell_wildcard\)', mr_text) and
            re.search(r'(?m)^lvtn\s*=\s*polygons\(125,\s*44\)', mr_text) and
            re.search(r'lvtn\s*\.interacting\(nwell\)\s*\.not_inside\(nwell\)\s*\.edges\s*\.and\(nwell\)\s*\.output\("MR_lvtn\.OVL\.2"', mr_text)):
        raise ValueError('native FEOL SKY130 LVT selector deck authority differs')
    from .edge_rail_geometry import _stamp_geometry
    by_master = {}
    by_stamp = {}
    for stamp in stamps:
        if stamp.stamp_id in by_stamp:
            raise ValueError('native FEOL source stamp identity is duplicated')
        by_stamp[stamp.stamp_id] = stamp
        by_master.setdefault(stamp.master_id, []).append(stamp)
    keys = {'diff': (65, 20), 'nsdm': (93, 44), 'nwell': (64, 20), 'lvtn': (125, 44)}
    digest = hashlib.sha256(magic.read_bytes()).hexdigest()
    rows, clearances = [], {}
    candidates = []
    if eligible_stamp_pairs is None:
        for pair in sorted(set(eligible_master_pairs)):
            if len(pair) != 2 or pair[0] >= pair[1]:
                raise ValueError('native FEOL eligible master pair is malformed')
            sides = [by_master.get(master, ()) for master in pair]
            if any(len(side) != 1 for side in sides):
                continue
            candidates.append((pair, sides[0][0], sides[1][0]))
    else:
        if eligible_master_pairs or type(eligible_stamp_pairs) is not tuple:
            raise ValueError('native FEOL stamp selection must be an exclusive exact tuple')
        if eligible_stamp_pairs != tuple(sorted(set(eligible_stamp_pairs))):
            raise ValueError('native FEOL stamp pairs must be canonical and unique')
        for stamps_pair in eligible_stamp_pairs:
            if (type(stamps_pair) is not tuple or len(stamps_pair) != 2
                    or any(type(name) is not str or not name for name in stamps_pair)
                    or stamps_pair[0] >= stamps_pair[1]
                    or any(name not in by_stamp for name in stamps_pair)):
                raise ValueError('native FEOL eligible stamp pair is malformed or absent')
            first, second = (by_stamp[name] for name in stamps_pair)
            if first.master_id == second.master_id:
                raise ValueError('native FEOL stamp pair must have distinct masters')
            pair = tuple(sorted((first.master_id, second.master_id)))
            candidates.append((pair, first, second))
    # One output-stamp proof is independent of its P/N partner. Reuse it only
    # inside this exact synchronous compilation, then independently replay it
    # before returning. A later post-merge compilation starts a fresh cache.
    output_proofs = {}
    def output_extension(first, second):
        from .native_feol_output_profile import EXTRA, admits
        ordinary={(64,20),(65,20),(65,44),(66,20),(66,44),(67,20),(67,44),
                  (68,20),(93,44),(94,20),(95,20),(125,44)}
        extra={s.stamp_id:frozenset(geometry[s.stamp_id].polys_by_layer)-ordinary
               for s in (first,second)}
        hits=[name for name, values in extra.items() if values]
        if len(hits)!=1 or extra[hits[0]]!=EXTRA:
            return False
        group=hits[0]
        if group in output_proofs:
            proof, _first, _second=output_proofs[group]
            _box, actual_digest=_stamp_geometry(geometry[group],placement_um[group])
            if actual_digest!=proof.placed_geometry_sha256:
                raise ValueError('native FEOL cached output stamp changed')
            return proof
        proof=admits(bundle=bundle,cell=cell,stamps=stamps,geometry=geometry,
            placement_um=placement_um,row_program=row_program,row_materialization=row_materialization,
            tech=tech,first=first,second=second,magic_text=magic_text)
        if proof:output_proofs[group]=(proof,first,second)
        return proof
    for pair, first, second in candidates:
        kinds_by_stamp = {name: value.kind for name, value in bundle.geometries.items()}
        if row_kinds is not None:
            if type(row_kinds) is not dict or any(kind not in ('nmos', 'pmos') for kind in row_kinds.values()):
                raise ValueError('native FEOL row kinds are malformed')
            kinds_by_stamp.update(row_kinds)
        if any(stamp.stamp_id not in kinds_by_stamp for stamp in (first, second)):
            continue
        kinds = [kinds_by_stamp[stamp.stamp_id] for stamp in (first, second)]
        is_nn = kinds == ['nmos', 'nmos']
        if set(kinds) != {'nmos', 'pmos'} and not is_nn:
            continue
        # The cited periphery rules do not cover arbitrary SRAM/precision-resistor
        # markers. Limit this lowering to ordinary native MOS artwork; leave the
        # pre-existing fallback in force when a stamp has any other mask.
        allowed = {(64, 20), (65, 20), (65, 44), (66, 20), (66, 44),
                   (67, 20), (67, 44), (68, 20), (93, 44), (94, 20), (95, 20)}
        allowed.add((125, 44))
        source_output_extension=False
        if any(
                set(geometry[stamp.stamp_id].polys_by_layer) - allowed for stamp in (first,second)):
            from .native_feol_output_profile import EXTRA
            source_output_extension=output_extension(first,second)
            if source_output_extension:allowed |= EXTRA
        if any(set(geometry[stamp.stamp_id].polys_by_layer) - allowed
               for stamp in (first, second)):
            continue
        source_devices = {device.instance_name: device for device in getattr(cell, 'devices', ())}
        row_owners = {entry.group: entry.consumed_devices for entry in getattr(row_program, 'entries', ())}
        for stamp, kind in zip((first, second), kinds):
            owners = row_owners.get(stamp.stamp_id, (stamp.stamp_id,))
            has_selector = bool(geometry[stamp.stamp_id].polys_by_layer.get((125, 44)))
            is_lvt = bool(owners) and all(name in source_devices and
                    source_devices[name].source_model == 'sky130_fd_pr__nfet_01v8_lvt'
                    for name in owners)
            if (has_selector and (kind != 'nmos' or not is_lvt)) or (is_lvt and not has_selector):
                raise ValueError('native FEOL LVT selector lacks source model ownership')
        inventories = []
        for stamp in (first, second):
            actual, shape_sha = _stamp_geometry(geometry[stamp.stamp_id], placement_um[stamp.stamp_id])
            if shape_sha != stamp.geometry_sha256:
                raise ValueError('native FEOL source stamp changed')
            delta = (stamp.bbox_nm[0] - actual[0], stamp.bbox_nm[1] - actual[1])
            inventories.append(_regions(geometry[stamp.stamp_id], placement_um[stamp.stamp_id], delta, keys))
        if is_nn:
            # Whole-master allowances are safe only for two exact source row
            # owners, never for a stamp inside a larger composite master.
            if eligible_stamp_pairs is not None:
                continue
            evidence = _sky130_lvt_normal_row_envelope(bundle=bundle, cell=cell,
                stamps=stamps, geometry=geometry, placement_um=placement_um,
                row_kinds=row_kinds, row_program=row_program,
                row_materialization=row_materialization, tech=tech,
                first=first, second=second, pair=pair, magic=magic,
                magic_text=magic_text, mr_text=mr_text,
                source_output_extension=source_output_extension,
                source_output_replay=output_extension)
            if evidence is not None:
                row, pair_clearances = evidence
                rows.append(row)
                clearances.update(pair_clearances)
            continue
        nside = kinds.index('nmos')
        pside = 1 - nside
        ndiff = (inventories[nside]['diff'] & inventories[nside]['nsdm']) - inventories[nside]['nwell']
        nwell = inventories[pside]['nwell']
        if ndiff.is_empty() or nwell.is_empty():
            raise ValueError('native FEOL source lacks actual N diffusion or P well')
        sides_rect = [None, None]
        sides_rect[nside], sides_rect[pside] = _boxes(ndiff.merged()), _boxes(nwell.merged())
        sides_kind = ['ndiff' if i == nside else 'nwell' for i in range(2)]
        sides_layers = [tuple(sorted(((64, 20), (65, 20), (93, 44)))) if i == nside else ((64, 20),) for i in range(2)]
        # NativeFeolPairIR orders stamps, while eligible pairs order masters.
        ordered = sorted(zip((first, second), sides_kind, sides_layers, sides_rect), key=lambda item: item[0].stamp_id)
        a, b = ordered
        rows.append(NativeFeolPairIR(a[0].stamp_id, b[0].stamp_id, a[0].master_id, b[0].master_id,
            a[1], b[1], a[2], b[2], a[3], b[3], 340, 'sky130:diff/tap.9', str(magic), digest))
        lvtn = inventories[nside]['lvtn']
        if not lvtn.is_empty():
            sides_rect[nside], sides_rect[pside] = _boxes(lvtn), _boxes(nwell)
            sides_kind = ['lvtn' if i == nside else 'nwell' for i in range(2)]
            sides_layers = [((125, 44),) if i == nside else ((64, 20),) for i in range(2)]
            a, b = sorted(zip((first, second), sides_kind, sides_layers, sides_rect), key=lambda item: item[0].stamp_id)
            rows.append(NativeFeolPairIR(a[0].stamp_id, b[0].stamp_id, a[0].master_id, b[0].master_id,
                a[1], b[1], a[2], b[2], a[3], b[3], 0, 'sky130:MR_lvtn.OVL.2', str(mr),
                hashlib.sha256(mr.read_bytes()).hexdigest()))
        if eligible_stamp_pairs is None:
            # A stamp-local proof must never relax all layers of a composite
            # master containing unrelated stamps.
            for layer, value in (((66, 44), 170), ((67, 20), 170), ((93, 44), 380),
                                 ((94, 20), 380), ((95, 20), 270)):
                clearances[(pair[0], pair[1], layer)] = value
    for proof, first, second in output_proofs.values():
        from .native_feol_output_profile import admits
        replayed=admits(bundle=bundle,cell=cell,stamps=stamps,geometry=geometry,
            placement_um=placement_um,row_program=row_program,row_materialization=row_materialization,
            tech=tech,first=first,second=second,magic_text=magic_text)
        if replayed!=proof:
            raise ValueError('native FEOL source output proof changed during pair compilation')
    return tuple(rows), clearances


def sky130_native_row_kinds(*, bundle, cell, stamps, geometry,
                            row_program, row_materialization, tech):
    """Prove each row stamp's native source ownership and physical polarity."""
    if bundle is None or row_program is None or row_materialization is None:
        return {}
    bundle.require_current(cell)
    from clay.layout.place.rail_row_replay_scope import require_rail_row_materialization
    require_rail_row_materialization(row_materialization, program=row_program, tech_id=tech.pdk_id)
    source_members = {member.device: member for member in row_program.members}
    stamp_by_name = {stamp.stamp_id: stamp for stamp in stamps}
    if len(stamp_by_name) != len(stamps):
        raise ValueError('native FEOL row stamp identity is duplicated')
    row_kinds = {}
    for entry in row_program.entries:
        if entry.group not in geometry or entry.group not in stamp_by_name:
            raise ValueError('native FEOL row is missing its materialized stamp')
        if entry.group in bundle.geometries or entry.group in row_kinds:
            raise ValueError('native FEOL row stamp identity is ambiguous')
        kinds = {source_members[name].kind for name in entry.consumed_devices}
        if len(kinds) != 1 or any(name not in bundle.geometries or
                bundle.geometries[name].kind != source_members[name].kind
                for name in entry.consumed_devices):
            raise ValueError('native FEOL row source owner/kind differs')
        row_kinds[entry.group] = kinds.pop()
    return row_kinds


def compile_sky130_row_feol_pairs(*, bundle, cell, stamps, geometry, placement_um,
                                  row_program, row_materialization, tech):
    """Bind cross-well clearance to source-owned, materialized native row stamps."""
    row_kinds=sky130_native_row_kinds(bundle=bundle,cell=cell,stamps=stamps,
        geometry=geometry,row_program=row_program,row_materialization=row_materialization,tech=tech)
    if not row_kinds:
        return ()
    stamp_by_name = {stamp.stamp_id: stamp for stamp in stamps}
    if len(stamp_by_name) != len(stamps):
        raise ValueError('native FEOL row stamp identity is duplicated')
    eligible = []
    names = sorted(row_kinds)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            if row_kinds[left] == row_kinds[right]:
                continue
            if stamp_by_name[left].master_id != stamp_by_name[right].master_id:
                eligible.append((left,right))
    rows, _ = compile_sky130_native_feol_pairs(bundle=bundle, cell=cell, stamps=stamps,
        geometry=geometry, placement_um=placement_um, eligible_master_pairs=(),
        eligible_stamp_pairs=tuple(sorted(eligible)), row_kinds=row_kinds, row_program=row_program,
        row_materialization=row_materialization,tech=tech)
    if {(row.left_stamp, row.right_stamp) for row in rows} != set(eligible):
        raise ValueError('native FEOL row pair cannot be represented by source-bound artwork')
    return tuple(sorted(rows, key=lambda row: (row.left_stamp, row.right_stamp, row.rule_id,
                                             row.left_kind, row.right_kind)))


def compile_sky130_group_row_spacing(*, intent, bundle, cell, stamps, geometry, placement_um,
                                     row_program, row_materialization, tech):
    """Prove the full four-stamp boundary of two explicitly separated native groups."""
    if getattr(tech, 'pdk_id', None) != 'sky130' or row_program is None or row_materialization is None:
        return ()
    separations = tuple(getattr(intent, 'group_separations', ()) or ())
    if not any(row.axis == 'vertical' for row in separations):
        return ()
    entries = {row.group: row for row in row_program.entries}
    by_name = {row.stamp_id: row for row in stamps}
    devices = {row.instance_name: row for row in cell.devices}
    groups = {row.group: set(row.devices) for row in getattr(intent, 'place_groups', ())}
    allowed = {(64,20),(65,20),(65,44),(66,20),(66,44),(67,20),(67,44),
               (68,20),(93,44),(94,20),(95,20)}
    candidates = []
    for separation in separations:
        if separation.axis != 'vertical':
            continue
        group_sets = (groups.get(separation.before), groups.get(separation.after))
        if any(value is None or len(value) != 4 for value in group_sets) or group_sets[0] & group_sets[1]:
            continue
        sides = []
        for members in group_sets:
            selected = [entry for entry in entries.values() if set(entry.consumed_devices) <= members
                        and set(entry.consumed_devices) & members]
            if (len(selected) != 2 or set().union(*(set(e.consumed_devices) for e in selected)) != members
                    or sum(len(e.consumed_devices) for e in selected) != len(members)
                    or any(entry.group not in by_name or entry.group not in geometry for entry in selected)
                    or any(name not in devices for name in members)):
                break
            models = [{devices[name].source_model for name in entry.consumed_devices} for entry in selected]
            if {frozenset(value) for value in models} != {
                    frozenset(('sky130_fd_pr__nfet_01v8',)), frozenset(('sky130_fd_pr__pfet_01v8',))}:
                break
            names = {entry.group for entry in selected}
            masters = {by_name[name].master_id for name in names}
            if (len(masters) != 1 or {s.stamp_id for s in stamps if s.master_id in masters} != names
                    or any(set(geometry[name].polys_by_layer) - allowed for name in names)):
                break
            sides.append(tuple(sorted(names)))
        if len(sides) == 2 and by_name[sides[0][0]].master_id != by_name[sides[1][0]].master_id:
            candidates.append(tuple(sides))
    if not candidates:
        return ()
    kinds = sky130_native_row_kinds(bundle=bundle, cell=cell, stamps=stamps,
        geometry=geometry, row_program=row_program, row_materialization=row_materialization, tech=tech)
    if any({kinds.get(name) for name in side} != {'nmos', 'pmos'}
           for candidate in candidates for side in candidate):
        raise ValueError('native group source row polarity differs')
    magic = Path(bundle.output_dir) / 'sky130A.tech'
    mr = Path(bundle.output_dir) / 'sky130A_mr.drc'
    for path in (magic, mr):
        if bundle.common_artifacts.get(path.name) != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError('native group spacing PDK deck differs')
    magic_text, mr_text = magic.read_text(), mr.read_text()
    checks = (
        r'spacing\s+allpoly,polyfill\s+allpoly,polyfill\s+210\s+touching_ok',
        r'spacing\s+alldifflv,var,varhvt,corenvar,corepvar,fomfill\s*\\\s*alldifflv,var,varhvt,corenvar,corepvar,fomfill\s+270\s+touching_ok',
        r'edge4way\s+\*psd\s+\*ndiff\s+300',
        r'spacing\s+\*locali,rli\s+\*locali,rli,\*obsli\s+170\s+touching_ok',
        r'spacing\s+mcon/m1\s+mcon/m1,obsmcon/m1\s+190\s+touching_ok',
        r'widespacing\s+allm1\s+3005\s+allm1,\*obsm1,m1fill\s+280\s+touching_ok',
    )
    mappings = (
        r'(?m)^diff\s*=\s*input\(65,\s*20\)', r'(?m)^tap\s*=\s*polygons\(65,\s*44\)',
        r'(?m)^poly\s*=\s*polygons\(66,\s*20\)', r'(?m)^licon\s*=\s*polygons\(66,\s*44\)',
        r'(?m)^li_wildcard\s*=\s*"67/20"\s*$', r'(?m)^mcon_wildcard\s*=\s*"67/44"\s*$',
        r'(?m)^m1_wildcard\s*=\s*"68/20"\s*$', r'(?m)^nwell_wildcard\s*=\s*"64/20"\s*$',
        r'(?m)^nsdm_wildcard\s*=\s*"93/44"\s*$', r'(?m)^psdm_wildcard\s*=\s*"94/20"\s*$',
        r'(?m)^npc\s*=\s*polygons\(95,\s*20\)',
        r'difftap\s*=\s*diff\.or\(tap\)', r'difftap\.space\(0\.27,\s*euclidian\)',
        r'nsdm\s*\.not\(areaid_ce\)\s*\.space\(0\.38,\s*euclidian\)',
        r'psdm\s*\.not\(areaid_ce\)\s*\.space\(0\.38,\s*euclidian\)',
        r'npc\.space\(0\.27,\s*euclidian\)',
        r'nwell\.space\(1\.27,\s*euclidian\)',
        r'huge_m1\.separation\(non_huge_m1,\s*0\.28,\s*euclidian\)',
    )
    if not all(re.search(p, magic_text) for p in checks) or not all(re.search(p, mr_text) for p in mappings):
        raise ValueError('native group spacing SKY130 deck authority differs')
    rules = {(64,20):1270, (93,44):380, (94,20):380}
    digest = hashlib.sha256(magic.read_bytes()).hexdigest()
    from .edge_rail_geometry import _stamp_geometry
    output = []
    for sides in candidates:
        for first_name in sides[0]:
            for second_name in sides[1]:
                first, second = sorted((by_name[first_name], by_name[second_name]), key=lambda s:s.stamp_id)
                inventories = []
                for stamp in (first, second):
                    actual, identity = _stamp_geometry(geometry[stamp.stamp_id], placement_um[stamp.stamp_id])
                    if identity != stamp.geometry_sha256:
                        raise ValueError('native group source stamp changed')
                    delta = (stamp.bbox_nm[0]-actual[0], stamp.bbox_nm[1]-actual[1])
                    keys = {str(key):key for key in geometry[stamp.stamp_id].polys_by_layer}
                    inventories.append(_regions(geometry[stamp.stamp_id], placement_um[stamp.stamp_id], delta, keys))
                common = set(geometry[first.stamp_id].polys_by_layer) & set(geometry[second.stamp_id].polys_by_layer)
                output.append(NativeFeolPairIR(first.stamp_id,second.stamp_id,first.master_id,second.master_id,
                    'group-envelope','group-envelope',tuple(sorted(geometry[first.stamp_id].polys_by_layer)),
                    tuple(sorted(geometry[second.stamp_id].polys_by_layer)),(first.bbox_nm,),(second.bbox_nm,),
                    300,'sky130:group-row-envelope',str(magic),digest))
                for layer in sorted(common & set(rules)):
                    left, right = _boxes(inventories[0][str(layer)]), _boxes(inventories[1][str(layer)])
                    if left and right:
                        output.append(NativeFeolPairIR(first.stamp_id,second.stamp_id,first.master_id,second.master_id,
                            'group-layer','group-layer',(layer,),(layer,),left,right,rules[layer],
                            'sky130:group-row-layer-'+str(layer[0])+'-'+str(layer[1]),str(magic),digest))
    return tuple(sorted(output, key=lambda row:(row.left_stamp,row.right_stamp,row.rule_id,row.left_kind,row.right_kind)))


def sky130_group_spacing_master_pairs(ir):
    """Return only masters with the complete four-stamp and layer proof."""
    if ir is None:
        return ()
    pairs = {}
    for row in ir.native_feol_pairs:
        if row.rule_id == 'sky130:group-row-envelope':
            key = tuple(sorted((row.left_master,row.right_master)))
            pairs.setdefault(key,[]).append(row)
    admitted = []
    rules = {(64,20):1270, (93,44):380, (94,20):380}
    for key, envelopes in pairs.items():
        stamp_pairs = {(row.left_stamp,row.right_stamp) for row in envelopes}
        if (len(envelopes) != 4 or len(stamp_pairs) != 4
                or len({name for pair in stamp_pairs for name in pair}) != 4
                or any(row.gap_nm != 300 for row in envelopes)):
            continue
        relevant = [row for row in ir.native_feol_pairs
                    if tuple(sorted((row.left_master,row.right_master))) == key]
        complete = True
        for envelope in envelopes:
            for layer in set(envelope.left_layers) & set(envelope.right_layers) & set(rules):
                expected = 'sky130:group-row-layer-'+str(layer[0])+'-'+str(layer[1])
                matching = [row for row in relevant if row.left_stamp == envelope.left_stamp
                            and row.right_stamp == envelope.right_stamp and row.rule_id == expected]
                if (len(matching) != 1 or matching[0].gap_nm != rules[layer]
                        or matching[0].left_layers != (layer,) or matching[0].right_layers != (layer,)
                        or matching[0].rule_sha256 != envelope.rule_sha256):
                    complete = False
        cross = {(row.left_stamp,row.right_stamp) for row in relevant
                 if row.rule_id == 'sky130:diff/tap.9' and row.gap_nm == 340}
        if complete and len(cross & stamp_pairs) == 2:
            admitted.append(key)
    return tuple(sorted(admitted))


def compile_native_feol_pairs(*, tech, stamps, geometry, placement_um):
    """Bind foreign-macro recognition exclusions to original source stamps.

    Local primitive builders remain responsible for their own fixed interiors.
    Separate solver masters do not acquire permission to merge body domains
    just because an unrelated routing net happens to have the same spelling.
    """
    authority = foundry_rules(tech)
    if authority is None:
        return ()
    from .edge_rail_geometry import _stamp_geometry
    rules, source, digest = authority
    keys = {kind: tech.build_layer(kind) for kind in
            ("nwell", "poly", "contact", "nplus", "pplus", "diff")}
    keys.update({kind: tech.marker_key(kind) for kind in ("sab", "res_mk", "dualgate")})
    if any(keys[kind] is None for kind in ("nwell", "poly", "contact", "nplus", "pplus", "sab", "res_mk")):
        raise ValueError("Native FEOL requires complete foundry layer identities")
    inventories = {}
    for stamp in stamps:
        name = stamp.stamp_id
        box, shape_sha = _stamp_geometry(geometry[name], placement_um[name])
        if shape_sha != stamp.geometry_sha256:
            raise ValueError("Native FEOL source stamp hash changed")
        delta = (stamp.bbox_nm[0] - box[0], stamp.bbox_nm[1] - box[1])
        regions = _regions(geometry[name], placement_um[name], delta, keys)
        derived = dict(regions)
        derived["field_poly"] = regions["poly"] - regions["diff"]
        derived["gate"] = regions["poly"] & regions["diff"]
        derived["uns_poly"] = regions["poly"] & regions["sab"]
        res_poly = (regions["poly"] & regions["nplus"]).interacting(regions["sab"]).interacting(regions["res_mk"])
        derived["res_contact"] = regions["contact"].inside(res_poly)
        dependencies = {kind: (kind,) for kind in keys}
        dependencies.update(field_poly=("poly", "diff"), gate=("poly", "diff"),
                            uns_poly=("poly", "sab"),
                            res_contact=("poly", "nplus", "sab", "res_mk", "contact"))
        inventory = {kind: (_boxes(region), tuple(sorted({keys[key] for key in dependencies[kind]
                                                         if keys[key] is not None})))
                     for kind, region in derived.items() if not region.is_empty()}
        inventories[name] = (stamp.master_id, inventory)

    contracts = (
        ("nwell", "nwell", "NW33", "NW.2b_3.3V_:independent-body-domain"),
        ("sab", "sab", "SB2", "SB.2:independent-recognition-domain"),
        ("sab", "diff", "SB3", "SB.3"),
        ("sab", "contact", "SB4", "SB.4"),
        ("sab", "res_contact", "LRES7", "LRES.7"),
        ("sab", "field_poly", "SB5a", "SB.5a"),
        ("sab", "gate", "SB5b", "SB.5b"),
        ("nplus", "uns_poly", "NP7", "NP.7"),
        ("pplus", "uns_poly", "SB15a", "SB.15a/PP.9:foreign-implant"),
        ("res_mk", "poly", None, "LRES.9a:foreign-recognition-no-overlap"),
    )
    rows = []
    names = sorted(inventories)
    for i, left in enumerate(names):
        left_master, left_inv = inventories[left]
        for right in names[i + 1:]:
            right_master, right_inv = inventories[right]
            if left_master == right_master:
                continue
            for a_kind, b_kind, value, rule_id in contracts:
                for left_kind, right_kind in ((a_kind, b_kind),) if a_kind == b_kind else ((a_kind, b_kind), (b_kind, a_kind)):
                    if left_kind not in left_inv or right_kind not in right_inv:
                        continue
                    gap = 0 if value is None else rules[value]
                    active_rule = rule_id
                    if value == "NW33" and ("dualgate" in left_inv or "dualgate" in right_inv):
                        gap, active_rule = rules["NW5"], "NW.2b_5V_:independent-body-domain"
                    left_boxes, left_layers = left_inv[left_kind]
                    right_boxes, right_layers = right_inv[right_kind]
                    rows.append(NativeFeolPairIR(
                        left, right, left_master, right_master, left_kind, right_kind,
                        left_layers, right_layers, left_boxes, right_boxes,
                        gap, active_rule, source, digest))
    return tuple(sorted(rows, key=lambda row: (row.left_stamp, row.right_stamp, row.rule_id,
                                             row.left_kind, row.right_kind)))


def displacement_cover(row):
    from .constraints_spatial import _pair_collision_forbidden_box_cover
    return _pair_collision_forbidden_box_cover(
        tuple((None, box) for box in row.left_rectangles_nm),
        tuple((None, box) for box in row.right_rectangles_nm), required_nm=row.gap_nm)[0]


def master_displacement_covers(rows):
    """Union all rule exclusions per master pair, preserving every raw operand.

    Nested device/ring N-wells and several recognition rules often forbid the
    same translations. Their geometric union is identical to their conjunction
    of outside-domain clauses; no original proof row is dropped from the IR.
    """
    from clay.layout.spatial.rectangle_union import rectangle_union_cover
    groups = {}
    for row in rows:
        pair = tuple(sorted((row.left_master, row.right_master)))
        boxes = displacement_cover(row)
        if pair[0] != row.left_master:
            # Integer points of [a,b) negate to [-b+1,-a+1), not [-b,-a).
            boxes = tuple((-x1 + 1, -y1 + 1, -x0 + 1, -y0 + 1)
                          for x0, y0, x1, y1 in boxes)
        groups.setdefault(pair, []).extend(boxes)
    return tuple((left, right, rectangle_union_cover(boxes))
                 for (left, right), boxes in sorted(groups.items()))


def native_feol_rows_satisfied(rows, positions):
    """Independent direct atom-pair oracle, never consuming displacement union."""
    for row in rows:
        lx, ly = positions[row.left_master]
        rx, ry = positions[row.right_master]
        dx, dy = rx - lx, ry - ly
        gap = row.gap_nm
        for a0, b0, a1, b1 in row.left_rectangles_nm:
            for c0, d0, c1, d1 in row.right_rectangles_nm:
                if not (a1 + gap <= dx + c0 or dx + c1 + gap <= a0
                        or b1 + gap <= dy + d0 or dy + d1 + gap <= b0):
                    return False
    return True
