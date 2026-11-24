"""Backward-compatible re-export of canonical OverlayAdapter (no silent subclass shadowing)."""
from clay.layout.route.group_fallback_adapter_base import OverlayAdapter
from clay.layout.route.group_fallback_order import ordered_alternatives

BaseOverlayAdapter = OverlayAdapter

__all__ = ['BaseOverlayAdapter', 'OverlayAdapter', 'ordered_alternatives']
