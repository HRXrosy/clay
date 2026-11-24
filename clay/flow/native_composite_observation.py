"""Explicit terminal capture of a source-owned native composite producer.

The ordinary layout producer and capture contracts remain authoritative. This
request stops immediately after capture and conveys no placement or signoff
verdict. Capture consumers must revalidate the retained receipt and files.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class NativeCompositeObservation:
    """Request one capture before layout policy resolution."""

