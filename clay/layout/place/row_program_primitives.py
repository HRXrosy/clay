"""Shared row-program failure type and integer dimension boundary."""

MAX_DIMENSION_NM = 1_000_000_000


class RowProgramError(ValueError):
    """The normalized RowProgram contract is malformed or unbounded."""
