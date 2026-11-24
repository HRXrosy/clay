"""
ngspice OP-quantity log parser + merge helper (v0.1.14 R91 B split).

Extracted verbatim from `clay/invariants.py` lines 1509-1738.

Reads `.print op @m<dev>[<q>]` log lines and folds them into a
canonical OP-result dict the per-keyword check functions key off.
"""

from __future__ import annotations

import math
import re
from typing import Mapping, Optional


# ─── v0.0.87: ngspice OP-quantity log parser ───────────────────────
#
# v0.0.87's testbench-side injection (`clay/_ir_bridge.py`) emits
# `.print op @m<dev>[<q>]` directives so ngspice surfaces per-device
# small-signal parameters in the batch log. Two surface forms can land
# in the log depending on the ngspice version + analysis kind:
#
#   1. The `.print op` directive itself produces ngspice-formatted
#      lines like `@m1[gm] = 4.32e-03` (with the bracket form intact).
#   2. The companion `let + echo` pair lands a deterministic
#      `gm_M1=4.32e-03` line that the existing `_MEAS_LINE_RE` in
#      `_ir_bridge.py::_parse_measure_log` already reads.
#
# `parse_op_quantity_lines(log_text)` reads form (1) — the raw
# `@m<dev>[<q>] = <value>` lines from `.print op` output — and converts
# them into the canonical `gm_<name>` / `id_<name>` / `vds_<name>` /
# `vgs_<name>` / `vov_<name>` keys the per-keyword check functions
# already know how to resolve. The bracket form is parsed
# case-insensitively (ngspice lowercases device names in print output)
# and the device name is normalized to match the role.name as
# declared in the .clay source.
#
# `merge_op_quantities(op_result, log_text, *, roles=None)` builds an
# augmented mapping: the original `op_result` keys are preserved
# verbatim, and any newly-resolved `gm_<name>` / `id_<name>` keys are
# added when they're absent or NaN in the original. Pass a non-None
# `roles` list to restrict the merge to declared functional-role
# devices — otherwise every `@m<x>[q]` line in the log gets folded in
# (useful for tests that don't carry roles but want to assert
# resolution coverage).


# Regex for raw `.print op @m<dev>[<q>] = <value>` log lines.
#
# Surface forms ngspice emits:
#
#   Top-level (literal v0.0.87 form):  `@m1[gm] = 4.32e-03`
#                                      `@mMdp[gm] = 1.85e-04`
#   Subckt-instanced (hierarchical):   `@m.xdut.xmdp[gm] = 1.5e-04`
#                                      `@m.xdut.mdp[id] = 5.0e-05`
#
# The parser strips hierarchical path components (everything between
# `@m.` and the trailing identifier) but PRESERVES the leading device
# prefix when present — for the top-level form `@m<Name>` we read
# `<Name>` directly without re-prepending `m`. ngspice typically
# lowercases identifiers in print-op output, so a downstream alias
# step in `merge_op_quantities` reconciles the parsed name with the
# original mixed-case `role.name` from the .clay declaration.
_OP_QUANTITY_LINE_RE = re.compile(
    r"@m(?:"
    # Hierarchical form: `@m.path.path.<dev>[<q>]`
    r"\.(?:[a-zA-Z0-9_]+\.)*(?P<dev_hier>[a-zA-Z0-9_]+)"
    r"|"
    # Top-level form: `@m<dev>[<q>]` (note: leading `m` is already
    # consumed by the `@m` literal at the start of the regex, so
    # the captured group is just the rest of the device identifier).
    r"(?P<dev_top>[a-zA-Z0-9_]+)"
    r")"
    r"\s*\[(?P<qty>\w+)\]"
    r"\s*=\s*(?P<val>[-+]?[\d.]+(?:[eE][-+]?\d+)?)",
    re.MULTILINE,
)


def parse_op_quantity_lines(log_text: str) -> dict[str, float]:
    """Extract per-device OP quantities from a raw ngspice log.

    Reads lines of the form ``@m<dev>[<q>] = <value>`` (the output of
    ``.print op @m<dev>[<q>]``) and converts them into the canonical
    invariant-runtime key set:

      * ``@mMdp[gm] = 1.5e-4``     → ``{"gm_Mdp": 1.5e-4}``
      * ``@m.xdut.mtail[id] = 5e-5`` → ``{"id_mtail": 5e-5}``
      * ``@m1[gm] = 4.32e-3``       → ``{"gm_m1": 4.32e-3}``

    Hierarchical paths (e.g. `m.xdut.<dev>`) drop the path segments
    and use only the trailing device identifier. Top-level forms
    (`@m<dev>`) preserve the literal `<dev>` segment that follows the
    `@m` prefix — so a `@m1` line lands under `gm_m1` (the leading
    `m` is kept because for top-level MOSFETs `m1` is the canonical
    SPICE device name). Mixed-case `@mMdp` lines land under `gm_Mdp`
    (verbatim what the brief's literal `.print op @m<DeviceName>[gm]`
    deck line probes for).

    Returns an empty dict when no `@m...` lines match.
    """
    out: dict[str, float] = {}
    for m in _OP_QUANTITY_LINE_RE.finditer(log_text):
        # The regex branches between hierarchical (`dev_hier`) and
        # top-level (`dev_top`) forms — exactly one branch matches
        # per line. For hierarchical the trailing identifier IS the
        # device name. For top-level the `@m` literal was already
        # consumed, so we re-prepend it to land the canonical `m<x>`
        # device name (top-level MOSFETs are named like `m1`, `mdp`).
        dev_hier = m.group("dev_hier")
        dev_top = m.group("dev_top")
        if dev_hier is not None:
            dev = dev_hier
        else:
            dev = "m" + dev_top
        qty = m.group("qty").lower()
        try:
            val = float(m.group("val"))
        except ValueError:
            continue
        if not math.isfinite(val):
            continue
        key = f"{qty}_{dev}"
        # First match wins — `.print op` blocks typically land before
        # any sibling echo line, and we want the canonical ngspice
        # value rather than a downstream rewrite.
        out.setdefault(key, val)
    return out


def _device_name_matches_role(dev_name: str, role_name: str) -> bool:
    """Return True iff a parsed ngspice device name maps to a declared
    role name.

    Three accepted matching forms (all case-insensitive):

      * Direct match: `dev_name.lower() == role_name.lower()`.
      * Subckt-instance prefix: `dev_name == "x" + role_name` (the
        Rust spice_export emits MOS devices as `xMdp ...` because the
        PDK model is itself a subckt; ngspice surfaces them as
        `m.xdut.xmdp` in print-op output).
      * Lowercase canonical: `dev_name.lower() == ("x" +
        role_name).lower()` (some ngspice builds re-lowercase the
        subckt instance segment).
    """
    if not dev_name or not role_name:
        return False
    rn_lc = role_name.lower()
    dn_lc = dev_name.lower()
    if dn_lc == rn_lc:
        return True
    # Strip a leading `x` (subckt-instance prefix from Rust spice_export).
    if dn_lc.startswith("x") and dn_lc[1:] == rn_lc:
        return True
    return False


def merge_op_quantities(
    op_result: Mapping[str, float],
    log_text: str,
    *,
    roles: Optional[list] = None,
) -> dict[str, float]:
    """Augment `op_result` with `gm_<name>` / `id_<name>` keys parsed
    from a raw ngspice log.

    Args:
      op_result: existing OP-analysis measure dict (e.g. the return of
        `_parse_measure_log`). Keys already present + finite are kept
        verbatim; missing or NaN keys may be filled by the merge.
      log_text: raw ngspice batch-mode log text containing
        `@m<dev>[<q>] = <value>` lines from `.print op` directives.
      roles: optional list of `ParsedFunctionalRole` declarations to
        narrow the merge. When provided, only keys whose device name
        matches a declared role name (direct, subckt-prefix `x`-stripped,
        or case-insensitive) are folded in — and each match also gets
        a CANONICAL alias key (`gm_<role.name>`) so the invariant
        runtime's `gm_<role.name>` resolution path lands a hit.
        When `None`, every parsed `@m<x>[q]` line lands in the output
        verbatim.

    Returns:
      A new dict (does not mutate `op_result`) carrying the union of
      the original keys + any newly-resolved per-device OP quantities.
      Existing finite values in `op_result` are preserved; only NaN /
      absent keys get overwritten.
    """
    merged: dict[str, float] = dict(op_result)
    extracted = parse_op_quantity_lines(log_text)

    # Without a role filter, fold every parsed entry verbatim — useful
    # for ad-hoc audits / debug tooling that wants the raw OP dump.
    if roles is None:
        for key, val in extracted.items():
            existing = merged.get(key)
            if existing is None or (
                isinstance(existing, float) and math.isnan(existing)
            ):
                merged[key] = val
        return merged

    # With a role filter, do BOTH a verbatim merge of matching keys
    # AND emit a canonical alias matching the role's declared name.
    # The invariant runtime keys off `gm_<role.name>` (verbatim case),
    # so the alias is what actually lands the resolution.
    for key, val in extracted.items():
        qty, _, dev_name = key.partition("_")
        for r in roles:
            rn = getattr(r, "name", "")
            if not _device_name_matches_role(dev_name, rn):
                continue
            # Verbatim parsed key (e.g. `gm_xmdp`).
            prev = merged.get(key)
            if prev is None or (
                isinstance(prev, float) and math.isnan(prev)
            ):
                merged[key] = val
            # Canonical alias matching the declared role name
            # (e.g. `gm_Mdp`). This is the key the invariant runtime
            # actually reads — landing it here is what moves the
            # demo cell's v0.0.86 `measure_missing` to `skipped` or
            # `mismatch` in v0.0.87.
            alias = f"{qty}_{rn}"
            if alias != key:
                prev_alias = merged.get(alias)
                if prev_alias is None or (
                    isinstance(prev_alias, float)
                    and math.isnan(prev_alias)
                ):
                    merged[alias] = val
            # Also emit `i_<name>` for tail-current-source convenience:
            # the runtime's `_check_tail_current_source` resolution
            # attempts `i_<name>` first. Surface it here so a single
            # `id_` log line can resolve both keys when the role is
            # tail_current_source.
            if qty == "id" and getattr(r, "kind", "") == "tail_current_source":
                i_alias = f"i_{rn}"
                prev_i = merged.get(i_alias)
                if prev_i is None or (
                    isinstance(prev_i, float) and math.isnan(prev_i)
                ):
                    merged[i_alias] = val
    return merged
