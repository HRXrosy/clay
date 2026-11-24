"""Complete native SKY130 DRC, LVS, RC extraction, and source net binding.

Each invocation owns a new directory and an explicitly pinned container image.
DRC failures remain failures even when LVS and the RC source binding succeed.
The source-bound PexReport is a measurement view; native simulation bytes are
never rewritten, filtered, or replaced by schematic devices.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import time
import uuid
import xml.etree.ElementTree as ET

from clay.frontend.native_spice_import import _read_subcircuits, import_native_spice
from clay.layout.pex.base import PexCap, PexReport, PexRes
from clay.layout.pex.magic_runner import build_extract_tcl
from clay.pdk.native_spice import native_spice_model_contract
from .native_rc import (
    NativeEvidenceError, _require, bind_native_rc, native_nets,
    parse_magic_ext, parse_native_extraction,
)

_SCHEMA = "clay.native-sky130-verification/v2"
_PDK_ID = "sky130"
_DECKS = (
    ("feol", "sky130A_mr.drc", ("feol=true", "beol=false", "offgrid=true")),
    ("beol", "sky130A_mr.drc", ("feol=false", "beol=true", "offgrid=false")),
    ("density", "met_min_ca_density.lydrc", ()),
    ("pincheck", "pin_label_purposes_overlapping_drawing.rb.drc", ("threads=4", "flat_mode=true")),
    ("zeroarea", "zeroarea.rb.drc", ()),
)
_DRC_STEPS = tuple(row[0] for row in _DECKS) + ("magic_drc",)
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_NETGEN_RUNTIME = ("/foss/tools/netgen/lib/netgen/tcl/netgen.tcl",
                   "/foss/tools/netgen/lib/netgen/tcl/tclnetgen.so")


def _identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _source_inventory(text, top_cell):
    # The existing native importer validates every original model, terminal and
    # parameter token. Exact source-net spelling comes from its parsed flat input,
    # not from a new name encoder or an inferred hierarchy-prefix conversion.
    import_native_spice(text, top_cell=top_cell, pdk_id=_PDK_ID)
    definitions = _read_subcircuits(text)
    _require(set(definitions) == {top_cell.casefold()}, "REFERENCE_MUST_BE_COMPLETE_FLAT_NATIVE_SPICE")
    top = definitions[top_cell.casefold()]
    models = {row.model for row in top.instances}
    contracts = {model: native_spice_model_contract(_PDK_ID, model).canonical_data() for model in sorted(models)}
    nets = set(top.ports) | {node for row in top.instances for node in row.nodes}
    _require(len({n.casefold() for n in nets}) == len(nets), "SOURCE_NET_CASE_COLLISION")
    return {"ports": list(top.ports), "nets": sorted(nets), "models": contracts,
            "logical_device_count": len(top.instances)}


def _runner_sources():
    from clay.frontend import native_spice_import
    from clay.layout.pex import joint_extraction, magic_parser, magic_runner
    from clay.pdk import native_spice, sky130
    from . import native_rc
    modules = (native_spice_import, joint_extraction, magic_parser, magic_runner,
               native_spice, sky130, native_rc)
    return {Path(path).as_posix(): _identity(path)
            for path in [__file__, *(m.__file__ for m in modules)]}


def _native_command(output, label, argv, *, image, pdk_root, container_name):
    pdk = pdk_root.rstrip("/") + "/sky130A"
    return ["docker", "run", "--rm", "--network", "none", "--name", container_name,
               "-v", f"{output.as_posix()}:/work", "-v", f"{(output / 'inputs').as_posix()}:/inputs:ro",
               "-w", f"/work/{label}", "-e", "PDK=sky130A", "-e", f"PDK_ROOT={pdk_root}",
               "-e", f"PDKPATH={pdk}", "-e", "STD_CELL_LIBRARY=sky130_fd_sc_hd",
               "--entrypoint", "bash", image, "-lc", "exec " + shlex.join(argv)]


def _native_commands(top, pdk_root):
    pdk = pdk_root.rstrip("/") + "/sky130A"
    commands = {
        "toolchain": ["python3", "/inputs/toolchain.py"],
        "magic_drc": ["magic", "-dnull", "-noconsole", "-rcfile", f"{pdk}/libs.tech/magic/sky130A.magicrc", "/inputs/drc.tcl"],
        "lvs_extract": ["magic", "-dnull", "-noconsole", "-rcfile", f"{pdk}/libs.tech/magic/sky130A.magicrc", "/inputs/lvs.tcl"],
        "pex": ["magic", "-dnull", "-noconsole", "-rcfile", f"{pdk}/libs.tech/magic/sky130A.magicrc", "/inputs/rc.tcl"],
        "lvs": ["netgen", "-batch", "source", "/inputs/netgen.tcl"],
    }
    for label, deck, options in _DECKS:
        argv = ["klayout", "-b", "-r", f"{pdk}/libs.tech/klayout/drc/{deck}"]
        for option in ("input=/inputs/design.gds", f"top_cell={top}", f"report=/work/{label}/report.xml", *options):
            argv.extend(["-rd", option])
        commands[label] = argv
    return commands


def _run_step(output, label, argv, *, image, pdk_root, timeout_seconds):
    folder = output / label
    folder.mkdir(exist_ok=False)
    container_name = f"clay-native-{label}-{uuid.uuid4().hex[:12]}"
    command = _native_command(output, label, argv, image=image, pdk_root=pdk_root,
                              container_name=container_name)
    record = {"label": label, "command": command, "container_name": container_name,
              "image": image, "started_utc": _now(), "timeout_seconds": timeout_seconds}
    _write_json(folder / "command.json", record)
    start = time.monotonic()
    with (folder / "stdout.log").open("wb") as stdout, (folder / "stderr.log").open("wb") as stderr:
        try:
            proc = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=timeout_seconds)
            record.update(exit_code=proc.returncode, timed_out=False)
        except subprocess.TimeoutExpired:
            # Only this invocation's uniquely named container is stopped.
            cleanup = subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, timeout=30)
            (folder / "timeout-cleanup.stdout.log").write_bytes(cleanup.stdout)
            (folder / "timeout-cleanup.stderr.log").write_bytes(cleanup.stderr)
            record.update(exit_code=None, timed_out=True, error="native tool timed out")
        except OSError as exc:
            record.update(exit_code=None, timed_out=False, error=str(exc))
    record.update(finished_utc=_now(), wall_seconds=time.monotonic() - start)
    _write_json(folder / "execution.json", record)
    return record


def _runtime_expectations(expected, pdk_root):
    if expected is None:
        return {}
    _require(type(expected) is dict, "INVALID_RUNTIME_EXPECTATIONS")
    allowed = (PurePosixPath(pdk_root), PurePosixPath("/foss/tools/netgen"))
    for path, digest in expected.items():
        _require(type(path) is str and PurePosixPath(path).is_absolute()
                 and str(PurePosixPath(path)) == path and ".." not in PurePosixPath(path).parts
                 and any(root in PurePosixPath(path).parents for root in allowed),
                 "RUNTIME_PATH_OUTSIDE_AUTHORITY_ROOTS", path)
        _require(type(digest) is str and bool(re.fullmatch(r"[0-9a-f]{64}", digest)),
                 "INVALID_RUNTIME_EXPECTED_HASH", path)
    return dict(sorted(expected.items()))


def _toolchain_script(pdk_root, source, expected_runtime_files):
    pdk = pdk_root.rstrip("/") + "/sky130A"
    required = [f"libs.tech/magic/sky130A.{suffix}" for suffix in ("magicrc", "tech", "tcl")]
    required += ["libs.tech/netgen/sky130A_setup.tcl"]
    required += ["libs.tech/klayout/drc/" + deck for _, deck, _ in _DECKS]
    required += ["libs.tech/klayout/tech/sky130A." + suffix for suffix in ("map", "lyt", "lyp")]
    # This runs in the same pinned image as the tools. Model headers are checked
    # against the PDK typed contract, and their actual installed bytes are bound.
    return "\n".join([
        "import hashlib,json,pathlib,re,subprocess",
        f"root=pathlib.Path({pdk!r})", f"required={required!r}", f"contracts={source['models']!r}",
        f"expected={expected_runtime_files!r}", f"runtime_required={list(_NETGEN_RUNTIME)!r}",
        f"allowed_roots=[pathlib.Path({pdk_root!r}).resolve(),pathlib.Path('/foss/tools/netgen').resolve()]",
        "assert all((root/p).is_file() for p in required), 'required native PDK deck missing'",
        "paths=set()",
        "for sub in ('libs.tech/magic','libs.tech/netgen','libs.tech/klayout'):",
        " paths.update(p for p in (root/sub).rglob('*') if p.is_file())",
        "for name in sorted(set(runtime_required)|set(expected)):",
        " path=pathlib.Path(name)",
        " assert path.is_file(), ('NATIVE_RUNTIME_FILE_MISSING',name)",
        " assert any(path.resolve().is_relative_to(base) for base in allowed_roots), ('RUNTIME_PATH_OUTSIDE_AUTHORITY_ROOTS',name)",
        " paths.add(path)",
        "models={}",
        "for model,contract in contracts.items():",
        " definitions=[]",
        " for path in sorted((root/'libs.ref/sky130_fd_pr/spice').glob(model+'*.spice')):",
        "  for number,line in enumerate(path.read_text(errors='strict').splitlines(),1):",
        "   words=line.split()",
        "   if len(words)>1 and words[0].lower()=='.subckt' and words[1].lower()==model.lower():",
        "    tail=words[2:]; stop=next((i for i,w in enumerate(tail) if '=' in w or w.lower()=='params:'),len(tail))",
        "    ports=tail[:stop]",
        "    assert [p.lower() for p in ports]==[p[0].lower() for p in contract['terminals']], ('model port contract differs',model,path)",
        "    definitions.append({'path':str(path),'line':number,'ports':ports,'header':line})",
        "    paths.add(path)",
        " assert definitions, ('native model definition missing',model)",
        " models[model]=definitions",
        "def identity(p): return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}",
        "versions={}",
        "for name,cmd in {'magic':['magic','--version'],'klayout':['klayout','-v'],'netgen':['netgen','-batch','help']}.items():",
        " p=subprocess.run(cmd,capture_output=True,text=True,check=True); versions[name]={'command':cmd,'stdout':p.stdout,'stderr':p.stderr}",
        "result={'pdk_root':str(root),'required':required,'runtime_required':runtime_required,'expected_runtime_files':expected,'files':[identity(p) for p in sorted(paths)],'models':models,'versions':versions}",
        "pathlib.Path('/work/toolchain.json').write_text(json.dumps(result,indent=2)+'\\n')",
        "print('CLAY_NATIVE_TOOLCHAIN_COMPLETE')", "",
    ])


def _flatten(top):
    flat = top + "_clay_native_flat"
    return (f"flatten {flat}\nload {flat}\ncellname delete {top}\n"
            f"cellname rename {flat} {top}\nselect top cell\n")


def _input_scripts(top, pdk_root, source, expected_runtime_files):
    """One producer for executable input bytes and their later read-only replay."""
    drc = f"""crashbackups stop
gds read /inputs/design.gds
load {top}
select top cell
drc euclidean on
drc style drc(full)
drc check
set findings [drc listall why]
set fout [open /work/magic_drc/violations.txt w]
set oscale [cif scale out]
set count 0
foreach {{category boxes}} $findings {{
  puts $fout $category
  foreach box $boxes {{
    puts $fout [format " %.9gum %.9gum %.9gum %.9gum" [expr {{$oscale*[lindex $box 0]}}] [expr {{$oscale*[lindex $box 1]}}] [expr {{$oscale*[lindex $box 2]}}] [expr {{$oscale*[lindex $box 3]}}]]
    incr count
  }}
}}
puts $fout "COUNT: $count"
close $fout
puts "CLAY_NATIVE_DRC_COMPLETE"
quit -noprompt
"""
    lvs = (f"crashbackups stop\ndrc off\ngds read /inputs/design.gds\nload {top}\nselect top cell\n"
           + _flatten(top) + "extract path /work/lvs_extract\nextract no capacitance\n"
           "extract no coupling\nextract no resistance\nextract no length\nextract all\next2spice lvs\n"
           f"ext2spice -p /work/lvs_extract -o /work/lvs_extract/{top}.ext.spc\n"
           "puts \"CLAY_NATIVE_LVS_EXTRACTION_COMPLETE\"\nquit -noprompt\n")
    rc = build_extract_tcl(top, with_resistance=True, gds_filename="inputs/design.gds",
                           output_filename="pex/native.spice")
    rc = "crashbackups stop\ndrc off\n" + rc.replace("select top cell\n", "select top cell\n" + _flatten(top), 1)
    rc = rc.replace("quit -noprompt\n", "puts \"CLAY_NATIVE_RC_EXTRACTION_COMPLETE\"\nquit -noprompt\n")
    return {"toolchain.py": _toolchain_script(pdk_root, source, expected_runtime_files),
            "drc.tcl": drc, "lvs.tcl": lvs, "rc.tcl": rc}


def _write_inputs(output, top, pdk_root, source, expected_runtime_files):
    for name, text in _input_scripts(top, pdk_root, source, expected_runtime_files).items():
        (output / "inputs" / name).write_text(text, encoding="utf-8")


def _step_ok(output, executions, label, marker=None):
    record = executions.get(label, {})
    if record.get("exit_code") != 0 or record.get("timed_out"):
        return False
    text = "\n".join((output / label / f"{stream}.log").read_text(encoding="utf-8", errors="replace")
                     for stream in ("stdout", "stderr"))
    if re.search(r"Ports .*?electrically shorted", text, re.I):
        return False
    return marker is None or marker in text


def _validate_toolchain(output, source, pdk_root, expected_runtime_files):
    data = json.loads((output / "toolchain.json").read_text(encoding="utf-8"))
    pdk = pdk_root.rstrip("/") + "/sky130A"
    expected = {f"libs.tech/magic/sky130A.{suffix}" for suffix in ("magicrc", "tech", "tcl")}
    expected |= {"libs.tech/netgen/sky130A_setup.tcl"}
    expected |= {"libs.tech/klayout/drc/" + deck for _, deck, _ in _DECKS}
    expected |= {"libs.tech/klayout/tech/sky130A." + suffix for suffix in ("map", "lyt", "lyp")}
    files = {row["path"]: row for row in data["files"]}
    _require(data["pdk_root"] == pdk and set(data["required"]) == expected
             and {pdk + "/" + name for name in expected} <= files.keys()
             and len(files) == len(data["files"]), "NATIVE_TOOLCHAIN_DECK_INVENTORY_INCOMPLETE")
    _require(set(data["runtime_required"]) == set(_NETGEN_RUNTIME)
             and set(_NETGEN_RUNTIME) <= files.keys(), "NATIVE_TOOLCHAIN_BINARY_INVENTORY_INCOMPLETE")
    _require(data["expected_runtime_files"] == expected_runtime_files,
             "NATIVE_RUNTIME_EXPECTATIONS_DIFFER")
    _require(all(path in files and files[path]["sha256"] == digest
                 for path, digest in expected_runtime_files.items()), "NATIVE_RUNTIME_FILE_HASH_MISMATCH")
    _require(all(type(row["bytes"]) is int and row["bytes"] > 0
                 and re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) for row in files.values()),
             "NATIVE_TOOLCHAIN_FILE_IDENTITY_INVALID")
    _require(set(data["models"]) == set(source["models"]), "NATIVE_MODEL_HEADER_INVENTORY_INCOMPLETE")
    for model, contract in source["models"].items():
        definitions = data["models"][model]
        _require(bool(definitions) and all(row["path"] in files
                 and [n.casefold() for n in row["ports"]] == [p[0].casefold() for p in contract["terminals"]]
                 and type(row["line"]) is int and row["line"] > 0 and row["header"].strip()
                 for row in definitions), "NATIVE_MODEL_HEADER_CONTRACT_DIFFERS", model)
    _require(set(data["versions"]) == {"magic", "klayout", "netgen"}
             and all(row["command"] and (row["stdout"].strip() or row["stderr"].strip())
                     for row in data["versions"].values()), "NATIVE_TOOL_VERSION_EVIDENCE_INCOMPLETE")
    return data


def _xml_report(path):
    root = ET.parse(path).getroot()
    _require(root.tag == "report-database" and root.find("items") is not None
             and root.find("categories") is not None, "INCOMPLETE_KLAYOUT_REPORT", str(path))
    items = [{"category": row.findtext("category"), "cell": row.findtext("cell"),
              "multiplicity": row.findtext("multiplicity"),
              "values": [v.text for v in row.findall("./values/value")]}
             for row in root.findall("./items/item")]
    return {"item_count": len(items), "items": items,
            "categories": [{"name": c.findtext("name"), "description": c.findtext("description")}
                           for c in root.findall("./categories/category")]}


def _drc_reports(output, executions):
    reports = {}
    for label, _, _ in _DECKS:
        try:
            row = _xml_report(output / label / "report.xml")
            row.update(complete=_step_ok(output, executions, label), passed=False)
            row["passed"] = row["complete"] and row["item_count"] == 0
        except (OSError, ET.ParseError, NativeEvidenceError) as exc:
            row = {"complete": False, "passed": False, "error": str(exc)}
        reports[label] = row
    try:
        text = (output / "magic_drc/violations.txt").read_text(encoding="utf-8")
        counts = re.findall(r"^COUNT: (\d+)\s*$", text, re.M)
        items, category = [], None
        for line in text.splitlines():
            if line.startswith("COUNT:") or not line.strip():
                continue
            if line.startswith(" "):
                coords = [float(x) for x in re.findall(r"([-+0-9.eE]+)um", line)]
                _require(category is not None and len(coords) == 4 and all(math.isfinite(x) for x in coords),
                         "MALFORMED_MAGIC_DRC_ITEM")
                items.append({"category": category, "coordinates_um": coords})
            else:
                category = line
        complete = (len(counts) == 1 and int(counts[0]) == len(items)
                    and _step_ok(output, executions, "magic_drc", "CLAY_NATIVE_DRC_COMPLETE"))
        reports["magic_drc"] = {"complete": complete, "passed": complete and not items,
                                "item_count": len(items), "items": items}
    except (OSError, ValueError) as exc:
        reports["magic_drc"] = {"complete": False, "passed": False, "error": str(exc)}
    return {"passed": all(row["passed"] for row in reports.values()), "reports": reports,
            "scope": "Complete installed Magic full DRC and KLayout FEOL/BEOL/offgrid/density/pincheck/zeroarea; no exclusions or no-new-violation waiver."}


def _netgen_tcl(top, pdk_root, raw):
    nets = sorted(native_nets(raw))
    _require(all(re.fullmatch(r"[A-Za-z0-9_./:#$-]+", n) for n in nets), "UNSUPPORTED_EXTRACTED_IDENTIFIER")
    setup = pdk_root.rstrip("/") + "/sky130A/libs.tech/netgen/sky130A_setup.tcl"
    text = (f"lvs {{/work/lvs_extract/{top}.ext.spc {top}}} {{/inputs/reference.spice {top}}} "
            f"{{{setup}}} /work/lvs/report.out -json\n"
            "set unique [verify unique]\nputs \"CLAY_NATIVE_UNIQUE=$unique\"\n"
            "if {$unique != 1} {error \"Complete native LVS is not uniquely equivalent\"}\n"
            "set fout [open /work/lvs/net-map.tsv w]\n")
    for node in nets:
        text += "puts $fout [join [list {" + node + "} [matching node {" + node + "}]] \"\\t\"]\n"
    return text + "close $fout\nputs \"CLAY_NATIVE_NETMAP_COMPLETE\"\n"


def _read_netmap(output, raw, source, executions):
    _require(_step_ok(output, executions, "lvs", "CLAY_NATIVE_NETMAP_COMPLETE"), "NATIVE_LVS_NOT_COMPLETED")
    text = (output / "lvs/report.out").read_text(encoding="utf-8")
    _require("Circuits match uniquely." in text and not re.search(
        r"(?im)^(?:Property errors|Netlists do not match|Circuits do not match|The following cells had property errors)", text),
        "NATIVE_LVS_NOT_PASSED")
    report = json.loads((output / "lvs/report.json").read_text(encoding="utf-8"))
    _require(type(report) is list and bool(report), "NATIVE_LVS_JSON_MISSING")
    top_rows = [row for row in report if row.get("name") == [raw["top_cell"], raw["top_cell"]]]
    _require(len(top_rows) == 1, "NATIVE_LVS_TOP_SUMMARY_MISSING")
    summary = top_rows[0]
    _require(summary.get("nets") == [len(native_nets(raw)), len(source["nets"])]
             and summary.get("badnets") == [] and summary.get("badelements") == []
             and not summary.get("properties"), "NATIVE_LVS_SUMMARY_INCOMPLETE")
    devices, pins = summary.get("devices"), summary.get("pins")
    _require(type(devices) is list and len(devices) == 2
             and dict(devices[0]) == dict(devices[1])
             and {name.casefold() for name, count in devices[0] if type(count) is int and count > 0}
                 == {model.casefold() for model in source["models"]}
             and type(pins) is list and len(pins) == 2
             and [p.casefold() for p in pins[0]] == [p.casefold() for p in pins[1]]
             and set(p.casefold() for p in pins[0]) == set(raw["ports"]),
             "NATIVE_LVS_DEVICE_OR_PORT_DENOMINATOR_INCOMPLETE")
    source_names = {n.casefold(): n for n in source["nets"]}
    mapping = {}
    for line in (output / "lvs/net-map.tsv").read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        _require(len(fields) == 2 and fields[0].casefold() not in mapping, "DUPLICATE_OR_MALFORMED_NET_MAPPING")
        physical, emitted = fields[0].casefold(), fields[1].casefold()
        _require(emitted in source_names, "NETGEN_NET_NOT_IN_EXACT_SOURCE", fields[1])
        mapping[physical] = source_names[emitted]
    _require(set(mapping) == native_nets(raw) and set(mapping.values()) == set(source["nets"])
             and len(set(mapping.values())) == len(mapping), "INCOMPLETE_SOURCE_NET_MAPPING")
    _require(set(raw["ports"]) == {p.casefold() for p in source["ports"]}
             and all(mapping[p].casefold() == p for p in raw["ports"]), "SOURCE_PORT_ANCHORS_DIFFER")
    return mapping


def _replay_source_binding(output, top, source, executions):
    _require(_step_ok(output, executions, "lvs_extract", "CLAY_NATIVE_LVS_EXTRACTION_COMPLETE"), "NATIVE_LVS_EXTRACTION_NOT_COMPLETED")
    _require(_step_ok(output, executions, "pex", "CLAY_NATIVE_RC_EXTRACTION_COMPLETE"), "NATIVE_RC_EXTRACTION_NOT_COMPLETED")
    raw = parse_native_extraction((output / f"lvs_extract/{top}.ext.spc").read_text(), top_cell=top, pdk_id=_PDK_ID)
    raw["top_cell"] = top
    rc = parse_native_extraction((output / "pex/native.spice").read_text(), top_cell=top, pdk_id=_PDK_ID)
    mapping = _read_netmap(output, raw, source, executions)
    proof = bind_native_rc(raw=raw, rc=rc,
        raw_ext=parse_magic_ext((output / f"lvs_extract/{top}.ext").read_text()),
        rc_ext=parse_magic_ext((output / f"pex/{top}.ext").read_text()),
        res_ext=parse_magic_ext((output / f"pex/{top}.res.ext").read_text()),
        physical_to_source=mapping)
    return proof, rc


def run_native_verification(gds, reference_spice, top_cell, output, *, image,
                            pdk_root="/foss/pdks", timeout_seconds=1200.0, max_workers=3,
                            expected_runtime_files=None) -> dict:
    """Run complete native checks in a fresh output directory.

    ``reference_spice`` must be the complete flat native source. ``pdk_root`` is
    the container directory containing ``sky130A``. ``image`` is mandatory and
    must be a digest. A returned overall FAIL can still carry valid LVS/PEX
    measurements; inspect the explicit stage flags rather than an exit code.
    ``expected_runtime_files`` binds additional installed model/deck dependencies
    by absolute container path and SHA256; all are checked before native jobs.
    """
    _require(type(top_cell) is str and bool(_NAME.fullmatch(top_cell)), "INVALID_TOP_CELL")
    _require(type(image) is str and bool(re.fullmatch(r"(?:[^\s]+@)?sha256:[0-9a-f]{64}", image)), "PINNED_IMAGE_REQUIRED")
    _require(type(pdk_root) is str and pdk_root.startswith("/")
             and ".." not in PurePosixPath(pdk_root).parts
             and re.fullmatch(r"/[A-Za-z0-9_./-]+", pdk_root), "INVALID_CONTAINER_PDK_ROOT")
    _require(type(max_workers) is int and max_workers > 0 and math.isfinite(timeout_seconds) and timeout_seconds > 0,
             "INVALID_EXECUTION_LIMIT")
    expected_runtime_files = _runtime_expectations(expected_runtime_files, pdk_root)
    gds, reference, output = Path(gds).resolve(), Path(reference_spice).resolve(), Path(output).resolve()
    _require(gds.is_file() and reference.is_file(), "MISSING_NATIVE_INPUT")
    source = _source_inventory(reference.read_text(encoding="utf-8"), top_cell)
    initial = {"gds": _identity(gds), "reference_spice": _identity(reference)}
    producers = _runner_sources()
    output.mkdir(parents=True, exist_ok=False)
    (output / "inputs").mkdir()
    shutil.copyfile(gds, output / "inputs/design.gds")
    shutil.copyfile(reference, output / "inputs/reference.spice")
    _write_inputs(output, top_cell, pdk_root, source, expected_runtime_files)
    _write_json(output / "input-binding.json", {"schema": _SCHEMA, "inputs": initial,
                "producers": producers, "source_inventory": source, "top_cell": top_cell,
                "expected_runtime_files": expected_runtime_files,
                "image": image, "pdk_root": pdk_root, "created_utc": _now()})
    executions, errors = {}, []
    result = {"schema": _SCHEMA, "output": str(output), "top_cell": top_cell, "image": image,
              "pdk_root": pdk_root, "inputs": initial, "source_inventory": source,
              "expected_runtime_files": expected_runtime_files,
              "drc": {"passed": False}, "lvs": {"passed": False}, "pex": {"passed": False},
              "source_binding": {"passed": False}, "errors": errors, "executions": executions,
              "manifest": str(output / "manifest.json")}
    config = dict(image=image, pdk_root=pdk_root, timeout_seconds=float(timeout_seconds))
    commands = _native_commands(top_cell, pdk_root)
    try:
        executions["toolchain"] = _run_step(output, "toolchain", commands["toolchain"], **config)
        _require(_step_ok(output, executions, "toolchain", "CLAY_NATIVE_TOOLCHAIN_COMPLETE")
                 and (output / "toolchain.json").is_file(), "NATIVE_TOOLCHAIN_INCOMPLETE")
        _validate_toolchain(output, source, pdk_root, expected_runtime_files)
        jobs = {label: argv for label, argv in commands.items() if label not in {"toolchain", "lvs"}}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            pending = {pool.submit(_run_step, output, label, argv, **config): label for label, argv in jobs.items()}
            for future in as_completed(pending):
                executions[pending[future]] = future.result()
        result["drc"] = _drc_reports(output, executions)
        _require(_step_ok(output, executions, "lvs_extract", "CLAY_NATIVE_LVS_EXTRACTION_COMPLETE"), "NATIVE_LVS_EXTRACTION_NOT_COMPLETED")
        raw = parse_native_extraction((output / f"lvs_extract/{top_cell}.ext.spc").read_text(), top_cell=top_cell, pdk_id=_PDK_ID)
        raw["top_cell"] = top_cell
        _require(not raw["resistors"] and not raw["capacitors"], "LVS_EXTRACTION_HAS_PARASITICS")
        (output / "inputs/netgen.tcl").write_text(_netgen_tcl(top_cell, pdk_root, raw), encoding="utf-8")
        executions["lvs"] = _run_step(output, "lvs", commands["lvs"], **config)
        mapping = _read_netmap(output, raw, source, executions)
        result["lvs"] = {"passed": True, "physical_to_source_net": mapping,
                         "raw_device_count": len(raw["devices"]), "net_count": len(mapping),
                         "report": _identity(output / "lvs/report.out"), "mapping": _identity(output / "lvs/net-map.tsv")}
        proof, _ = _replay_source_binding(output, top_cell, source, executions)
        _write_json(output / "source-binding.json", proof)
        result["source_binding"] = {"passed": True, "evidence": _identity(output / "source-binding.json"),
                                    "source_net_count": proof["source_net_count"],
                                    "distributed_node_count": proof["distributed_node_count"]}
        result["pex"] = {"passed": True, "native_spice": _identity(output / "pex/native.spice"),
                          "device_count": proof["device_count"], "resistor_count": proof["resistor_count"],
                          "capacitor_count": proof["capacitor_count"], "negative_capacitor_count": proof["negative_capacitor_count"],
                          "junction_parameters_equal_to_nonRC_LVS": proof["junction_parameters_equal_to_nonRC_LVS"],
                          "junction_parameter_differences": proof["junction_parameter_differences"],
                          "scope": "Untouched full native RC; positive-R contraction is validation/measurement only."}
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError) as exc:
        errors.append({"type": type(exc).__name__, "message": str(exc)})
    result["inputs_unchanged"] = (
        all(Path(row["path"]).is_file() and _identity(row["path"]) == row for row in [*initial.values(), *producers.values()])
        and (output / "inputs/design.gds").is_file() and (output / "inputs/reference.spice").is_file()
        and _identity(output / "inputs/design.gds")["sha256"] == initial["gds"]["sha256"]
        and _identity(output / "inputs/reference.spice")["sha256"] == initial["reference_spice"]["sha256"])
    if not result["inputs_unchanged"]:
        errors.append({"type": "NativeEvidenceError", "message": "INPUT_OR_PRODUCER_BYTE_DRIFT"})
    result["passed"] = not errors and result["inputs_unchanged"] and all(result[k]["passed"] for k in ("drc", "lvs", "pex", "source_binding"))
    result["finished_utc"] = _now()
    # Physical-device coordinate keys can be tuples. Return the exact JSON
    # representation that the sealed replay API accepts, including diagnostics.
    result = json.loads(json.dumps(result))
    _write_json(output / "summary.json", result)
    manifest = {"schema": _SCHEMA, "inputs": initial, "producers": producers,
                "files": {p.relative_to(output).as_posix(): _identity(p)
                          for p in sorted(output.rglob("*")) if p.is_file() and p != output / "manifest.json"}}
    _write_json(output / "manifest.json", manifest)
    return result


def _validate_native_envelope(result_or_output):
    """Read the sealed receipt and bind every producer/input before stage replay."""
    output = Path(result_or_output["output"] if isinstance(result_or_output, dict) else result_or_output).resolve()
    result = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    if isinstance(result_or_output, dict):
        _require(result == result_or_output, "RESULT_DIFFERS_FROM_SEALED_SUMMARY")
    manifest_bytes = (output / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    manifest_identity = {"path": str(output / "manifest.json"), "bytes": len(manifest_bytes),
                         "sha256": hashlib.sha256(manifest_bytes).hexdigest()}
    binding = json.loads((output / "input-binding.json").read_text(encoding="utf-8"))
    _require(result.get("schema") == manifest.get("schema") == binding.get("schema") == _SCHEMA,
             "WRONG_NATIVE_VERIFICATION_SCHEMA")
    _require(Path(result["output"]).resolve() == output
             and Path(result["manifest"]).resolve() == output / "manifest.json", "NATIVE_OUTPUT_IDENTITY_DIFFERS")
    expected_files = {p.relative_to(output).as_posix() for p in output.rglob("*")
                      if p.is_file() and p != output / "manifest.json"}
    _require(set(manifest["files"]) == expected_files and "summary.json" in expected_files, "INCOMPLETE_NATIVE_MANIFEST")
    for relative, row in manifest["files"].items():
        path = (output / relative).resolve()
        _require(path.is_relative_to(output) and path.is_file() and _identity(path) == row, "NATIVE_ARTIFACT_BYTE_DRIFT", relative)
    _require(result["inputs"] == manifest["inputs"] == binding["inputs"]
             and set(result["inputs"]) == {"gds", "reference_spice"}, "NATIVE_INPUT_BINDING_DIFFERS")
    _require(manifest["producers"] == binding["producers"] == _runner_sources(),
             "NATIVE_PRODUCER_INVENTORY_OR_BYTE_DRIFT")
    for row in [*manifest["inputs"].values(), *manifest["producers"].values()]:
        _require(Path(row["path"]).is_file() and _identity(row["path"]) == row, "NATIVE_INPUT_OR_PRODUCER_BYTE_DRIFT", row["path"])
    _require(result["inputs_unchanged"] is True, "NATIVE_INPUT_OR_PRODUCER_BYTE_DRIFT")
    for name, relative in (("gds", "inputs/design.gds"), ("reference_spice", "inputs/reference.spice")):
        copy = _identity(output / relative)
        _require(all(copy[key] == result["inputs"][name][key] for key in ("sha256", "bytes")),
                 "NATIVE_INPUT_COPY_DIFFERS", relative)
    for key in ("top_cell", "image", "pdk_root", "source_inventory", "expected_runtime_files"):
        _require(result[key] == binding[key], "NATIVE_SOURCE_EXECUTION_BINDING_DIFFERS", key)
    _require(type(result["top_cell"]) is str and bool(_NAME.fullmatch(result["top_cell"])), "INVALID_TOP_CELL")
    _require(type(result["image"]) is str and bool(re.fullmatch(r"(?:[^\s]+@)?sha256:[0-9a-f]{64}", result["image"])),
             "PINNED_IMAGE_REQUIRED")
    pdk_root = result["pdk_root"]
    _require(type(pdk_root) is str and pdk_root.startswith("/")
             and ".." not in PurePosixPath(pdk_root).parts and re.fullmatch(r"/[A-Za-z0-9_./-]+", pdk_root),
             "INVALID_CONTAINER_PDK_ROOT")
    _require(_runtime_expectations(result["expected_runtime_files"], pdk_root) == result["expected_runtime_files"],
             "NATIVE_RUNTIME_EXPECTATIONS_DIFFER")
    source = _source_inventory((output / "inputs/reference.spice").read_text(encoding="utf-8"), result["top_cell"])
    _require(source == result["source_inventory"], "SOURCE_INVENTORY_DRIFT")
    for name, text in _input_scripts(result["top_cell"], pdk_root, source, result["expected_runtime_files"]).items():
        _require((output / "inputs" / name).read_text(encoding="utf-8") == text,
                 "NATIVE_EXECUTABLE_INPUT_DIFFERS", name)
    commands = _native_commands(result["top_cell"], pdk_root)
    executions = result["executions"]
    _require(type(executions) is dict and set(executions) <= set(commands), "NATIVE_EXECUTION_INVENTORY_DIFFERS")
    for label, record in executions.items():
        observed = json.loads((output / label / "execution.json").read_text(encoding="utf-8"))
        command = json.loads((output / label / "command.json").read_text(encoding="utf-8"))
        _require(type(command) is dict and set(command) == {
                    "label", "command", "container_name", "image", "started_utc", "timeout_seconds"}
                 and observed == record and record["label"] == label and record["image"] == result["image"]
                 and all(record[key] == value for key, value in command.items()), "NATIVE_EXECUTION_RECORD_DIFFERS", label)
        _require(all((output / label / f"{stream}.log").is_file() for stream in ("stdout", "stderr")),
                 "NATIVE_EXECUTION_LOG_MISSING", label)
        expected_command = _native_command(output, label, commands[label], image=result["image"],
                                           pdk_root=pdk_root, container_name=record["container_name"])
        _require(record["command"] == expected_command, "NATIVE_EXECUTION_COMMAND_DIFFERS", label)
    _require(_step_ok(output, executions, "toolchain", "CLAY_NATIVE_TOOLCHAIN_COMPLETE"), "NATIVE_TOOLCHAIN_INCOMPLETE")
    _validate_toolchain(output, source, result["pdk_root"], result["expected_runtime_files"])
    _require(type(result["errors"]) is list and all(type(row) is dict for row in result["errors"]),
             "NATIVE_ERROR_INVENTORY_INVALID")
    _require(all(type(result[key].get("passed")) is bool for key in ("drc", "lvs", "pex", "source_binding")),
             "NATIVE_STAGE_STATUS_INVALID")
    expected_passed = not result["errors"] and all(result[key]["passed"] for key in ("drc", "lvs", "pex", "source_binding"))
    _require(result["passed"] is expected_passed, "NATIVE_OVERALL_STATUS_DIFFERS")
    return output, result, manifest, manifest_identity, source


def _replay_native_lvs_result(output, result, source):
    top, executions = result["top_cell"], result["executions"]
    _require(_step_ok(output, executions, "lvs_extract", "CLAY_NATIVE_LVS_EXTRACTION_COMPLETE"),
             "NATIVE_LVS_EXTRACTION_NOT_COMPLETED")
    raw = parse_native_extraction((output / f"lvs_extract/{top}.ext.spc").read_text(), top_cell=top, pdk_id=_PDK_ID)
    raw["top_cell"] = top
    _require(not raw["resistors"] and not raw["capacitors"], "LVS_EXTRACTION_HAS_PARASITICS")
    _require((output / "inputs/netgen.tcl").read_text(encoding="utf-8") == _netgen_tcl(top, result["pdk_root"], raw),
             "NATIVE_EXECUTABLE_INPUT_DIFFERS", "netgen.tcl")
    mapping = _read_netmap(output, raw, source, executions)
    replayed = {"passed": True, "physical_to_source_net": mapping,
                "raw_device_count": len(raw["devices"]), "net_count": len(mapping),
                "report": _identity(output / "lvs/report.out"), "mapping": _identity(output / "lvs/net-map.tsv")}
    _require(replayed == result["lvs"], "NATIVE_LVS_REPLAY_DIFFERS")


def _validate_native_evidence(result_or_output, required_stages):
    _require(type(required_stages) in (tuple, list) and len(set(required_stages)) == len(required_stages)
             and set(required_stages) <= {"drc", "lvs", "pex", "source_binding"}, "INVALID_NATIVE_REQUIRED_STAGES")
    output, result, manifest, manifest_identity, source = _validate_native_envelope(result_or_output)
    stages = set(required_stages)
    if "drc" in stages:
        replayed = _drc_reports(output, result["executions"])
        _require(all(row["complete"] for row in replayed["reports"].values()), "NATIVE_DRC_REPORTS_INCOMPLETE")
        _require(replayed == result["drc"], "NATIVE_DRC_REPLAY_DIFFERS")
    proof = rc = None
    if stages & {"lvs", "pex", "source_binding"}:
        _replay_native_lvs_result(output, result, source)
    if stages & {"pex", "source_binding"}:
        # Keep the complete physical-terminal/net binding used by the original
        # measurement view. A DRC violation or missing DRC report cannot be
        # silently promoted, but does not erase independently valid RC data.
        _require(not result["errors"] and all(result[key]["passed"] for key in ("lvs", "pex", "source_binding")),
                 "NATIVE_SOURCE_BINDING_NOT_PASSED")
        proof, rc = _replay_source_binding(output, result["top_cell"], source, result["executions"])
        saved = json.loads((output / "source-binding.json").read_text(encoding="utf-8"))
        _require(json.loads(json.dumps(proof)) == saved, "SOURCE_BINDING_REPLAY_DIFFERS")
        source_binding = {"passed": True, "evidence": _identity(output / "source-binding.json"),
                          "source_net_count": proof["source_net_count"],
                          "distributed_node_count": proof["distributed_node_count"]}
        pex = {"passed": True, "native_spice": _identity(output / "pex/native.spice"),
               **{key: proof[key] for key in ("device_count", "resistor_count", "capacitor_count", "negative_capacitor_count",
                                              "junction_parameters_equal_to_nonRC_LVS", "junction_parameter_differences")},
               "scope": "Untouched full native RC; positive-R contraction is validation/measurement only."}
        _require(source_binding == result["source_binding"]
                 and json.loads(json.dumps(pex)) == result["pex"], "NATIVE_PEX_REPLAY_DIFFERS")
    # Stage replay reads multiple files. Close with the same byte identities so
    # a change during those reads cannot acquire a current result.
    for row in [manifest_identity, *manifest["files"].values(), *manifest["inputs"].values(), *manifest["producers"].values()]:
        _require(Path(row["path"]).is_file() and _identity(row["path"]) == row, "NATIVE_EVIDENCE_CHANGED_DURING_REPLAY")
    return result, proof, rc


def validate_native_evidence(result_or_output, *, required_stages=("drc", "lvs", "pex", "source_binding")) -> dict:
    """Return a sealed result only after replaying the requested real evidence.

    Every call checks the complete artifact/input/producer/runtime envelope.
    ``('drc',)`` recomputes all six configured DRC reports: complete violations
    return ``drc.passed=False``; missing or malformed reports raise. LVS reads
    Netgen's text, JSON, and complete net correspondence independently of DRC.
    A failed/incomplete LVS raises. PEX/source-binding retains full signed RC
    and same-GDS physical-terminal replay. Unrelated stage failures therefore
    cannot grant PASS or prevent reporting an independently complete DRC/LVS
    conclusion. This function performs no external tool execution.
    """
    try:
        return _validate_native_evidence(result_or_output, required_stages)[0]
    except NativeEvidenceError:
        raise
    except (OSError, ValueError, KeyError, TypeError, ET.ParseError) as exc:
        raise NativeEvidenceError(f"NATIVE_EVIDENCE_INVALID: {type(exc).__name__}: {exc}") from exc


def source_bound_pex_report(result_or_output) -> PexReport:
    """Replay complete bound evidence and return signed source-net parasitics.

    A DRC failure does not erase a valid electrical measurement, but this view
    never grants physical signoff. Missing maps, raw bytes, logs or producer
    freshness reject before any PexReport is returned.
    """
    try:
        result, proof, rc = _validate_native_evidence(result_or_output, ("lvs", "pex", "source_binding"))
    except NativeEvidenceError:
        raise
    except (OSError, ValueError, KeyError, TypeError, ET.ParseError) as exc:
        raise NativeEvidenceError(f"NATIVE_EVIDENCE_INVALID: {type(exc).__name__}: {exc}") from exc
    mapping = proof["native_rc_node_to_source_net"]
    caps = tuple(PexCap(mapping[c["nodes"][0]], mapping[c["nodes"][1]], c["value"] * 1e18)
                 for c in rc["capacitors"] if mapping[c["nodes"][0]] != mapping[c["nodes"][1]])
    resistors = tuple(PexRes(mapping[r["nodes"][0]], mapping[r["nodes"][1]], r["value"]) for r in rc["resistors"])
    return PexReport(cell_name=result["top_cell"], caps=caps, resistors=resistors, backend="magic",
                     connectivity_source="full-native-LVS-and-same-GDS-RC-physical-terminal-binding",
                     constants_source=f"{result['image']}:{result['pdk_root']}/sky130A",
                     calibrated=True, field_solver=False,
                     note="Source-net measurement view only: signed external C, internal C excluded; original native RC simulation bytes unchanged.")


__all__ = ["run_native_verification", "validate_native_evidence", "source_bound_pex_report", "NativeEvidenceError"]
