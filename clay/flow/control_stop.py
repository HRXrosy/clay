"""Keep failed boundary checks attached to a consumed control-flow stop."""


class CheckedControlStop(BaseException):
    """A deliberate stop can resume only after every unwound boundary passed."""

    def __init__(self, *args):
        super().__init__(*args)
        self._unwind_failures = ()

    @property
    def unwind_failures(self):
        return self._unwind_failures

    def require_clean_unwind(self):
        if self._unwind_failures:
            raise ValueError('control stop has failed boundary checks') from self._unwind_failures[0]


def record_unwind_failure(original, ending):
    """Retain the primary exception and a typed failure for control consumers."""
    if not isinstance(original, BaseException) or not isinstance(ending, BaseException):
        raise TypeError('unwind failure requires actual exceptions')
    if isinstance(original, CheckedControlStop):
        original._unwind_failures += (ending,)
    try:
        detail = str(ending)
    except BaseException:
        detail = '<unprintable>'
    note = f'Ending boundary check also failed: {type(ending).__name__}: {detail}'
    try:
        original.add_note(note)
    except BaseException:
        # A custom exception may reject add_note; try the native implementation
        # so note-based terminal consumers still see the failed boundary.
        try:
            BaseException.add_note(original, note)
        except BaseException:
            # Exception annotations are best effort and must not replace the
            # primary. CheckedControlStop already retains the typed failure.
            pass
