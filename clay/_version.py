"""Single source of truth for Clay's version.

This MUST stay in sync with the top of CHANGELOG.md and the latest git tag (the
release facts), and with Cargo workspace version. The release gate checks all
three agree (see tools/check_version_consistency_v184). README references this
value; do not hard-code the version elsewhere -- import it from here.
"""

__version__ = "0.4.45"
