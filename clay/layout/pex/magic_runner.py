"""Run Magic inside a Docker container to produce ext2spice output.

This host has no native magic; availability is probed via the Docker image. When
unavailable, MagicPexBackend gates out -- exactly like ``pex.klayout_available()``
and ``ngspice_available()`` elsewhere in the repo.

The container invocation mirrors the verified smoke-test command (see
``tests/fixtures/pex/README.md``): a bind-mount of the GDS directory as ``/work``
and ``magic -dnull -noconsole -rcfile <pdk.magicrc> /work/<tcl>``. The mount path
is passed in forward-slash Windows form so Docker Desktop accepts it.
"""
from __future__ import annotations

import shutil
import subprocess
import re
import uuid
from pathlib import Path
from typing import Optional

MAGIC_DOCKER_IMAGE = "hpretl/iic-osic-tools:latest"

# Runnable container-internal Magic technologies.  GF180/SKY130 ship complete
# technology definitions inside the Docker image.
_RCFILE = {
    "gf180mcu": "/foss/pdks/gf180mcuD/libs.tech/magic/gf180mcuD.magicrc",
    "sky130": "/foss/pdks/sky130A/libs.tech/magic/sky130A.magicrc",
}

# Registered host-side constants overlays.  They intentionally are not in
# ``_RCFILE``: without a complete Magic technology/layer definition, loading an
# overlay alone cannot read or extract the commercial GDS honestly.
_HOST_RCFILE: dict[str, str] = {}

_TCL_NAME = "_clay_pex_extract.tcl"


class MagicTechnologyMismatchError(RuntimeError):
    """The selected installed technology cannot interpret this native artwork."""


class MagicPortTopologyError(RuntimeError):
    """Magic merged named ports; its output is not a valid topology witness."""


def expected_mim_option_for_pdk(pdk: str) -> Optional[str]:
    """Return the MIM stack option expected by the installed Magic technology."""
    if str(pdk).strip().casefold() == "gf180mcu":
        return "B"
    return None


def inspect_gf180_mim_stack(gds_path: Path, cell_name: str) -> dict[str, object]:
    """Inspect a GF180 GDS top cell for Option-A vs 5LM Option-B MIM stack shapes."""
    import klayout.db as kdb

    layout = kdb.Layout()
    layout.read(str(gds_path))
    cell = layout.cell(cell_name)
    if cell is None:
        raise MagicTechnologyMismatchError(
            f"Magic input has no declared top cell {cell_name!r}"
        )

    def region(layer):
        index = layout.find_layer(*layer)
        return (
            kdb.Region(cell.begin_shapes_rec(index))
            if index is not None
            else kdb.Region()
        )

    marker = region((75, 0))
    dbu2 = layout.dbu ** 2
    if marker.is_empty():
        return {
            "has_mim": False,
            "detected_option": None,
            "magic_compatible": True,
            "missing_m4_area_um2": 0.0,
            "missing_capdef_area_um2": 0.0,
            "forbidden_via3_count": 0,
        }
    m2 = region((36, 0))
    m4 = region((46, 0))
    capdef = region((117, 5))
    v2 = region((38, 0))
    v3 = region((40, 0))
    missing_bottom = marker - m4
    missing_recognition = marker - capdef
    option_a_m2 = not (marker & m2).is_empty() or not (marker & v2).is_empty()
    # MIMTM.10: no Via3 (40/0) may touch M4 interacting with FuseTop (75/0).
    forbidden_via3 = v3.interacting(m4.interacting(marker))
    option_b_ready = (
        missing_bottom.is_empty()
        and missing_recognition.is_empty()
        and forbidden_via3.is_empty()
    )
    if option_b_ready and not option_a_m2:
        detected = "B"
    elif option_a_m2 and not (marker & m4).is_empty():
        detected = "mixed"
    elif option_a_m2:
        detected = "A"
    elif option_b_ready:
        detected = "B"
    else:
        detected = "incomplete"
    return {
        "has_mim": True,
        "detected_option": detected,
        "magic_compatible": bool(option_b_ready),
        "missing_m4_area_um2": missing_bottom.area() * dbu2,
        "missing_capdef_area_um2": missing_recognition.area() * dbu2,
        "forbidden_via3_count": int(forbidden_via3.count()),
    }


def validate_gds_technology(gds_path: Path, cell_name: str, pdk: str) -> None:
    """Reject a proven GF180 MIM-stack mismatch before native extraction.

    The installed gf180mcuD rcfile resolves a technology whose CAPM (75/0)
    device is on metal4 (46/0), with CAPDEF (117/5) recognition and metal5
    access (5LM Option-B). Its cifinput section explicitly derives
    ``mimarea CAPDEF and MET4``. A lower-stack Option-A MIM (M2/V2/M3) or a
    bottom plate contacted by Via3 is therefore inadmissible in Magic and fails
    closed before extraction.
    """
    if pdk != "gf180mcu":
        return
    info = inspect_gf180_mim_stack(gds_path, cell_name)
    if not info["has_mim"]:
        return
    missing_m4 = float(info["missing_m4_area_um2"])
    missing_capdef = float(info["missing_capdef_area_um2"])
    bad_v3 = int(info["forbidden_via3_count"])
    if missing_m4 > 0.0 or missing_capdef > 0.0 or bad_v3 > 0:
        raise MagicTechnologyMismatchError(
            "GF180_MIM_STACK_MISMATCH: the selected gf180mcuD Magic technology "
            "requires 5LM Option-B CAPM(75/0) on M4(46/0), recognized by CAPDEF(117/5), "
            "with V4(41/0)/M5(81/0) access and no V3(40/0) on bottom-plate M4 "
            f"(detected_option={info['detected_option']!r}). "
            f"Uncovered marker area: M4={missing_m4:.9g} um^2, "
            f"CAPDEF={missing_capdef:.9g} um^2, forbidden_via3={bad_v3}. "
            "Emit GF180 Option-B MIM (mim_option='B') before running Magic R+C extraction; "
            "extraction was not launched."
        )


def rcfile_for_pdk(pdk: str) -> str:
    try:
        return _RCFILE[pdk]
    except KeyError:
        if pdk in _HOST_RCFILE:
            raise ValueError(
                f"Magic constants overlay for pdk={pdk!r} is registered but "
                "no complete runnable Magic technology is installed"
            )
        raise ValueError(f"no Magic rcfile mapping for pdk={pdk!r}")


def pdk_supported(pdk: str) -> bool:
    return str(pdk) in _RCFILE


def build_extract_tcl(
    cell_name: str,
    with_resistance: bool = False,
    gds_filename: Optional[str] = None,
    output_filename: Optional[str] = None,
) -> str:
    """Build the Magic batch tcl that drives ext2spice.

    ``gds_filename`` is the GDS file to ``gds read`` (defaults to
    ``<cell_name>.gds`` for backward-compat); ``cell_name`` is the TOP CELL to
    ``load`` (which may differ from the file's stem -- e.g. a comparator GDS named
    ``..._coclean_v183.gds`` whose top cell is ``gf180_dense_router_comparator_v183``).

    Default (``with_resistance=False``) emits the cap-only flow and is kept
    byte-identical to the original golden-fixture capture when the file is
    ``<cell_name>.gds`` (see ``tests/fixtures/pex/extract.tcl``).

    When ``with_resistance=True`` an OPT-IN distributed-resistance pass is woven
    in: after ``extract all`` Magic runs ``extresist all`` and ``ext2spice
    extresist on`` is set before the final ``ext2spice -o``, so the output
    ``.spice`` carries ``R<n>`` lines (wire / contact / diffusion resistance)
    in addition to the caps. Verified against the gf180mcuD ``sample_nmos_3p3``
    cell, which yields 3 ``R`` lines.
    """
    res_lines = "extresist all\next2spice extresist on\n" if with_resistance else ""
    gf = gds_filename or f"{cell_name}.gds"
    output = output_filename or f"{cell_name}.pex.spice"
    return (
        f"gds read /work/{gf}\n"
        f"load {cell_name}\n"
        "select top cell\n"
        "extract all\n"
        "ext2spice lvs\n"
        "ext2spice cthresh 0\n"
        "ext2spice rthresh 0\n"
        f"{res_lines}"
        f"ext2spice -o /work/{output}\n"
        "quit -noprompt\n"
    )


def _docker_cli() -> Optional[str]:
    return shutil.which("docker")


def _docker_image_present() -> bool:
    docker = _docker_cli()
    if not docker:
        return False
    try:
        out = subprocess.run(
            [docker, "images", "-q", MAGIC_DOCKER_IMAGE],
            capture_output=True, text=True, timeout=30,
        )
        return out.returncode == 0 and bool(out.stdout.strip())
    except Exception:
        return False


def magic_available() -> bool:
    return _docker_image_present()


def run_ext2spice(
    gds_path: str,
    cell_name: str,
    pdk: str = "gf180mcu",
    with_resistance: bool = False,
) -> str:
    """Run Magic in a container; return the ext2spice output text.

    Bind-mounts the GDS directory as ``/work`` and writes ``<cell>.pex.spice``
    there. Raises RuntimeError if Magic is unavailable or the run fails.

    ``with_resistance`` (OPT-IN, default False) adds Magic's ``extresist`` pass
    so the output carries ``R`` lines; the default cap-only flow is unchanged.
    """
    if not magic_available():
        raise RuntimeError(
            "Magic unavailable (no Docker image); use GeometricPexBackend"
        )
    docker = _docker_cli()
    gds = Path(gds_path).resolve()
    workdir = gds.parent
    mount = str(workdir).replace("\\", "/")  # Docker Desktop accepts C:/...
    rcfile = rcfile_for_pdk(pdk)
    validate_gds_technology(gds, cell_name, pdk)
    # An unchanged output from an earlier run is never evidence for this one.
    # Leave that artifact intact on failure and publish only a fresh result.
    output_name = f"_clay_pex_{uuid.uuid4().hex}.spice"
    out_file = workdir / output_name
    (workdir / _TCL_NAME).write_text(
        build_extract_tcl(cell_name, with_resistance=with_resistance,
                          gds_filename=gds.name, output_filename=output_name)
    )
    magic_cmd = f"magic -dnull -noconsole -rcfile {rcfile} /work/{_TCL_NAME}"
    proc = subprocess.run(
        [docker, "run", "--rm", "-v", f"{mount}:/work", "-w", "/work",
         "--entrypoint", "bash", MAGIC_DOCKER_IMAGE, "-lc", magic_cmd],
        capture_output=True, text=True, timeout=600,
    )
    (workdir / "_clay_pex_extract.stdout.log").write_text(proc.stdout, encoding="utf-8")
    (workdir / "_clay_pex_extract.stderr.log").write_text(proc.stderr, encoding="utf-8")
    port_shorts = re.findall(
        r'\bPorts\s+"([^"]+)"\s+and\s+"([^"]+)"\s+are\s+electrically\s+shorted\b',
        proc.stdout + "\n" + proc.stderr, flags=re.IGNORECASE,
    )
    if port_shorts:
        raise MagicPortTopologyError(
            f"Magic extraction merged named ports {port_shorts!r} "
            f"under {rcfile} (rc={proc.returncode}); no PEX result was published. "
            "Check the extraction technology and native stack against independent LVS; "
            "this warning alone does not establish a short in the source GDS.\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    if proc.returncode != 0 or not out_file.exists():
        raise RuntimeError(
            f"Magic ext2spice failed (rc={proc.returncode}):\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    result = out_file.read_text()
    if not result.strip():
        raise RuntimeError("Magic ext2spice produced an empty fresh output; no PEX result was published")
    (workdir / f"{cell_name}.pex.spice").write_text(result)
    return result
