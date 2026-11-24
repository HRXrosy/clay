from __future__ import annotations


def core_preview_payload(preview) -> dict:
    return {
        "status": preview.status,
        "reason": preview.reason,
        "source_obligation_ids": list(preview.source_obligation_ids),
        "row_count": preview.row_count,
        "certificate_present": preview.certificate is not None,
    }


def attach_core_preview_payload(payload: dict, preview=None) -> dict:
    attached = dict(payload)
    if preview is not None:
        attached["core_certificate_preview"] = core_preview_payload(preview)
    return attached
