"""ASCII grid helpers for layout visualization.

Utilities for aligning multiple ASCII art blocks side-by-side or
vertically, padding to a common bounding box, adding borders, and
labeling. Pure stdlib; no dependencies on other clay modules.

All functions treat ASCII art as a newline-separated string. Rows
do not need equal length; widths are normalized internally.
"""

from __future__ import annotations

from typing import List, Tuple


def measure(art: str) -> Tuple[int, int]:
    """Return (max_row_width, num_rows). Empty string = (0, 0)."""
    if art == "":
        return (0, 0)
    rows = art.split("\n")
    max_w = max((len(r) for r in rows), default=0)
    return (max_w, len(rows))


def pad_to_size(art: str, width: int, height: int, fill_char: str = " ") -> str:
    """Pad ASCII art (newline-separated rows) to exact (width, height).

    Extends each row with ``fill_char`` to ``width`` columns and adds
    blank rows at the bottom until ``height`` rows are present.
    Truncates rows / columns if the input is larger than the target.
    Returns the newline-joined result.
    """
    if len(fill_char) != 1:
        raise ValueError("fill_char must be exactly one character")
    if width < 0 or height < 0:
        raise ValueError("width and height must be non-negative")

    if height == 0:
        return ""

    rows = art.split("\n") if art != "" else []
    # Truncate rows beyond height
    rows = rows[:height]
    # Adjust each row width
    adjusted: List[str] = []
    for r in rows:
        if len(r) >= width:
            adjusted.append(r[:width])
        else:
            adjusted.append(r + fill_char * (width - len(r)))
    # Add blank rows to reach height
    while len(adjusted) < height:
        adjusted.append(fill_char * width)
    return "\n".join(adjusted)


def align_horizontal(
    art_blocks: List[str], spacing: int = 2, fill_char: str = " "
) -> str:
    """Place blocks side-by-side.

    Each block is padded to the max height of all blocks (using its own
    width). Between blocks ``spacing`` columns of ``fill_char`` are
    inserted. Returns the joined ASCII art as a single newline-separated
    string.
    """
    if len(fill_char) != 1:
        raise ValueError("fill_char must be exactly one character")
    if spacing < 0:
        raise ValueError("spacing must be non-negative")
    if not art_blocks:
        return ""

    measurements = [measure(b) for b in art_blocks]
    max_h = max((h for _, h in measurements), default=0)
    if max_h == 0:
        return ""

    # Pad each block to (its_own_width, max_h)
    padded_blocks: List[List[str]] = []
    for block, (w, _h) in zip(art_blocks, measurements):
        padded = pad_to_size(block, w, max_h, fill_char=fill_char)
        padded_blocks.append(padded.split("\n"))

    gap = fill_char * spacing
    out_rows: List[str] = []
    for row_idx in range(max_h):
        pieces = [pb[row_idx] for pb in padded_blocks]
        out_rows.append(gap.join(pieces))
    return "\n".join(out_rows)


def align_vertical(
    art_blocks: List[str], spacing: int = 1, fill_char: str = " "
) -> str:
    """Stack blocks vertically.

    Each block is padded to the max width of all blocks (using its own
    height). Between blocks ``spacing`` rows of ``fill_char`` (full
    width) are inserted.
    """
    if len(fill_char) != 1:
        raise ValueError("fill_char must be exactly one character")
    if spacing < 0:
        raise ValueError("spacing must be non-negative")
    if not art_blocks:
        return ""

    measurements = [measure(b) for b in art_blocks]
    max_w = max((w for w, _ in measurements), default=0)
    if max_w == 0:
        return ""

    padded_blocks: List[str] = []
    for block, (_w, h) in zip(art_blocks, measurements):
        padded = pad_to_size(block, max_w, h, fill_char=fill_char)
        padded_blocks.append(padded)

    gap_row = fill_char * max_w
    sep = "\n" + "\n".join([gap_row] * spacing) if spacing > 0 else "\n"
    # Build manually to keep separator semantics clean
    if spacing == 0:
        return "\n".join(padded_blocks)
    out_parts: List[str] = []
    for i, pb in enumerate(padded_blocks):
        if i > 0:
            out_parts.append("\n".join([gap_row] * spacing))
        out_parts.append(pb)
    return "\n".join(out_parts)


def add_border(
    art: str, *, hchar: str = "-", vchar: str = "|", corner: str = "+"
) -> str:
    """Surround art with single-char border on all 4 sides."""
    for name, ch in (("hchar", hchar), ("vchar", vchar), ("corner", corner)):
        if len(ch) != 1:
            raise ValueError(f"{name} must be exactly one character")

    w, h = measure(art)
    # Normalize art to a rectangle of (w, h)
    if h == 0:
        # Empty: produce a 2x2 corner-only border? Use empty interior.
        top = corner + corner
        bot = corner + corner
        return top + "\n" + bot
    body = pad_to_size(art, w, h, fill_char=" ").split("\n")
    top = corner + (hchar * w) + corner
    bottom = corner + (hchar * w) + corner
    middle = [vchar + row + vchar for row in body]
    return "\n".join([top, *middle, bottom])


def label_block(art: str, label: str) -> str:
    """Add a label line above the block, centered within the block width."""
    w, h = measure(art)
    label_str = label
    # If label is wider than the block, widen the block to fit label.
    target_w = max(w, len(label_str))
    if target_w == 0:
        # Both empty
        return label_str

    # Pad existing art to target_w
    if h == 0:
        body = ""
    else:
        body = pad_to_size(art, target_w, h, fill_char=" ")

    # Center label within target_w
    pad_total = target_w - len(label_str)
    left = pad_total // 2
    right = pad_total - left
    centered_label = (" " * left) + label_str + (" " * right)

    if body == "":
        return centered_label
    return centered_label + "\n" + body
