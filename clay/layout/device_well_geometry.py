"""Shared PMOS active-enclosure geometry used by native array builders."""


def shared_nwell_bbox(comp_bboxes, rules):
    """Keep the builders' centered X and downward Y minimum-width extension."""
    nx0 = min(box[0] for box in comp_bboxes) - rules.nwell_pcomp_overlap
    nx1 = max(box[2] for box in comp_bboxes) + rules.nwell_pcomp_overlap
    ny0 = min(box[1] for box in comp_bboxes) - rules.nwell_pcomp_overlap
    ny1 = max(box[3] for box in comp_bboxes) + rules.nwell_pcomp_overlap
    if nx1 - nx0 < rules.nwell_min_width:
        mid = (nx0 + nx1) // 2
        nx0 = mid - rules.nwell_min_width // 2
        nx1 = mid + rules.nwell_min_width // 2
    if ny1 - ny0 < rules.nwell_min_width:
        ny0 = ny1 - rules.nwell_min_width
    return nx0, ny0, nx1, ny1
