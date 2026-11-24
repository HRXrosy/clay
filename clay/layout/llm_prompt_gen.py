"""Few-shot LLM prompt generator for Clay layout cell template factories.

Given a natural-language description of a desired analog cell, this module
builds a structured few-shot prompt that an LLM can consume to emit a single
factory call (e.g. ``build_diff_pair(name='dp', w_um=4.0, l_um=0.5,
finger_count=4)``) which the Clay layout pipeline can then instantiate.

The module is intentionally self-contained: it does not import or modify any
of the cell-template factories themselves; it only references their names
and parameter conventions in string form. This keeps the prompt generator
decoupled from the live factory APIs and avoids cyclic imports.

Pure Python; no Rust mirror. Does not modify ``clay.layout.__init__`` or any
existing module — fully additive.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FewShotExample:
    """One in-context example for the LLM prompt.

    Attributes:
        description: Natural-language single-sentence description of the
            desired cell (the "user request" side of the example).
        factory_call: The exact Python factory invocation that realizes the
            description, as a single-line string starting with ``build_``.
            For example: ``"build_diff_pair(name='dp', w_um=4.0, l_um=0.5,
            finger_count=4)"``.
    """

    description: str
    factory_call: str


@dataclass
class FewShotPromptBundle:
    """Bundle of system prompt + examples + user request ready for an LLM."""

    system_prompt: str
    examples: List[FewShotExample] = field(default_factory=list)
    user_request: str = ""


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT = """\
You are an analog-layout synthesis assistant for the Clay HDL.

Clay exposes 8 cell template factories that build annotated analog cells
ready for placement and routing. When given a natural-language description
of a cell, your task is to emit exactly ONE line of Python: a call to the
single most appropriate factory below, with explicit keyword arguments.

The 8 factories are:
  1. build_diff_pair(name, w_um, l_um, finger_count, ...)
       — 2-MOS NMOS differential pair (M1, M2) with matching annotations.
  2. build_current_mirror(name, w_um, l_um, finger_count_ref,
        finger_count_out, ratio)
       — 2-MOS current mirror; output:input ratio is realized as a finger
         multiplier (finger_count_out = finger_count_ref * ratio).
  3. build_cascoded_current_mirror(name, w_um_low, l_um_low, w_um_cas,
        l_um_cas, finger_count_low, finger_count_cas)
       — 4-MOS wide-swing cascoded current mirror.
  4. build_5t_ota(name, input_w_um, input_l_um, input_finger_count, ...)
       — Five-transistor OTA built on top of a diff-pair + mirror load.
  5. build_folded_cascode(name, input_w_um, input_l_um, input_finger_count,
        ...) — Folded-cascode amplifier (single-ended output).
  6. build_telescopic_cascode(name, input_w_um, input_l_um,
        input_finger_count, ...) — Telescopic-cascode amplifier.
  7. build_two_stage_miller(name, input_w_um, input_l_um,
        input_finger_count, ...) — Two-stage Miller-compensated op-amp.
  8. build_bandgap_reference(name, ota_input_w_um, ota_input_l_um,
        ota_input_finger_count, ...) — Bandgap voltage reference with an
        OTA-driven feedback loop.

Parameter conventions:
  - w_um, l_um: device geometry in micrometres (floats, > 0).
  - finger_count / *_finger_count: positive integer (>= 1).
  - For amplifiers (5T OTA, folded/telescopic cascode, two-stage Miller,
    bandgap), the user-tunable sizing parameters are the *input pair*
    geometry only; the rest of the cell uses factory defaults.
  - For build_current_mirror, the ratio is the mirror ratio (1:N) and
    finger_count_out should equal finger_count_ref * ratio.
  - Always pass ``name`` as a short, snake_case identifier.

Output format:
  - Emit a SINGLE LINE containing one factory call and nothing else.
  - The line MUST start with ``build_``.
  - Use only keyword arguments (no positional args beyond ``name`` if
    you choose to pass it positionally — but keyword is preferred).
  - No surrounding prose, no Markdown fences, no comments.
"""


def _system_prompt() -> str:
    return _SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Few-shot examples
# ---------------------------------------------------------------------------


def build_few_shot_examples() -> List[FewShotExample]:
    """Return hand-curated few-shot examples covering all 8 templates.

    The list is intentionally ordered so the first 8 entries each cover a
    distinct factory; entries 9+ are additional flavor variants used to
    pad larger ``n_examples`` requests.
    """
    return [
        FewShotExample(
            description=(
                "A simple NMOS differential pair with W=4 um, L=0.5 um and "
                "4 fingers per device."
            ),
            factory_call=(
                "build_diff_pair(name='dp', w_um=4.0, l_um=0.5, "
                "finger_count=4)"
            ),
        ),
        FewShotExample(
            description=(
                "A 1:2 NMOS current mirror with W=2 um, L=1 um, reference "
                "device with 2 fingers and output device with 4 fingers."
            ),
            factory_call=(
                "build_current_mirror(name='cm', w_um=2.0, l_um=1.0, "
                "finger_count_ref=2, finger_count_out=4, ratio=2)"
            ),
        ),
        FewShotExample(
            description=(
                "A wide-swing cascoded current mirror, all four devices "
                "sized at W=4 um, L=0.5 um with 4 fingers."
            ),
            factory_call=(
                "build_cascoded_current_mirror(name='cas_cm', "
                "w_um_low=4.0, l_um_low=0.5, w_um_cas=4.0, l_um_cas=0.5, "
                "finger_count_low=4, finger_count_cas=4)"
            ),
        ),
        FewShotExample(
            description=(
                "A five-transistor OTA with input-pair geometry W=8 um, "
                "L=0.5 um, 8 fingers."
            ),
            factory_call=(
                "build_5t_ota(name='ota5t', input_w_um=8.0, input_l_um=0.5, "
                "input_finger_count=8)"
            ),
        ),
        FewShotExample(
            description=(
                "A folded-cascode amplifier with input pair sized W=4 um, "
                "L=1 um, 4 fingers."
            ),
            factory_call=(
                "build_folded_cascode(name='fc', input_w_um=4.0, "
                "input_l_um=1.0, input_finger_count=4)"
            ),
        ),
        FewShotExample(
            description=(
                "A telescopic-cascode amplifier with input pair W=2 um, "
                "L=0.5 um, 2 fingers."
            ),
            factory_call=(
                "build_telescopic_cascode(name='tc', input_w_um=2.0, "
                "input_l_um=0.5, input_finger_count=2)"
            ),
        ),
        FewShotExample(
            description=(
                "A two-stage Miller-compensated op-amp with input pair "
                "W=4 um, L=0.5 um and 4 fingers."
            ),
            factory_call=(
                "build_two_stage_miller(name='miller', input_w_um=4.0, "
                "input_l_um=0.5, input_finger_count=4)"
            ),
        ),
        FewShotExample(
            description=(
                "A bandgap reference whose internal OTA has an input pair "
                "of W=2 um, L=1 um and 4 fingers."
            ),
            factory_call=(
                "build_bandgap_reference(name='bgr', ota_input_w_um=2.0, "
                "ota_input_l_um=1.0, ota_input_finger_count=4)"
            ),
        ),
        # --- additional flavor variants for larger few-shot windows ---
        FewShotExample(
            description=(
                "A large differential pair, W=16 um, L=0.5 um with 16 "
                "fingers for high transconductance."
            ),
            factory_call=(
                "build_diff_pair(name='dp_large', w_um=16.0, l_um=0.5, "
                "finger_count=16)"
            ),
        ),
        FewShotExample(
            description=(
                "A 1:4 current mirror with W=2 um, L=2 um, reference "
                "1 finger and output 4 fingers."
            ),
            factory_call=(
                "build_current_mirror(name='cm_1to4', w_um=2.0, l_um=2.0, "
                "finger_count_ref=1, finger_count_out=4, ratio=4)"
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Bundle builder
# ---------------------------------------------------------------------------


def build_prompt(user_request: str, n_examples: int = 5) -> FewShotPromptBundle:
    """Assemble a complete few-shot prompt for an LLM.

    Args:
        user_request: Natural-language description of the cell the LLM
            should generate. Passed through verbatim into the bundle.
        n_examples: Number of in-context examples to include. Clamped to
            ``[0, len(build_few_shot_examples())]``; negative values are
            treated as 0 and values larger than the example pool yield
            the full pool.

    Returns:
        :class:`FewShotPromptBundle` with system prompt, the requested
        examples, and the user request.
    """
    all_examples = build_few_shot_examples()
    if n_examples < 0:
        n = 0
    elif n_examples > len(all_examples):
        n = len(all_examples)
    else:
        n = n_examples
    return FewShotPromptBundle(
        system_prompt=_system_prompt(),
        examples=all_examples[:n],
        user_request=user_request,
    )


# ---------------------------------------------------------------------------
# Plain-text rendering
# ---------------------------------------------------------------------------


def format_prompt_text(bundle: FewShotPromptBundle) -> str:
    """Render the bundle as a plain-text prompt for an LLM API.

    The format is:

        <system_prompt>

        Example 1:
        Description: <description>
        Factory call: <factory_call>

        Example 2:
        ...

        User request:
        <user_request>

        Factory call:
    """
    parts: List[str] = []
    parts.append(bundle.system_prompt.rstrip())
    parts.append("")  # blank line

    for idx, ex in enumerate(bundle.examples, start=1):
        parts.append(f"Example {idx}:")
        parts.append(f"Description: {ex.description}")
        parts.append(f"Factory call: {ex.factory_call}")
        parts.append("")  # blank line between examples

    parts.append("User request:")
    parts.append(bundle.user_request)
    parts.append("")
    parts.append("Factory call:")
    return "\n".join(parts)
