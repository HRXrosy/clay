# Contributing to Clay (Open-Source Edition)

Thank you for contributing to Clay!

## License (GNU AGPLv3)

All contributions submitted to Clay are licensed under the **[GNU Affero General Public License v3.0 (AGPL-3.0-only)](LICENSE)**. By submitting a pull request, you confirm that you have the right to license your contribution under AGPL-3.0.

## Open-PDK-Only Policy

1. **No Proprietary Foundry Data**: Never commit commercial PDK rules, layer maps, SPICE model cards, DRC/LVS decks, or NDA-restricted measurements. Only open-source PDKs (such as **SKY130** and **GF180MCU**) are accepted.
2. **C++20 Core Purity**: All files in `cpp/clay_pnr/include/` and core `cpp/clay_pnr/src/` (except `py_*` / `*_py.cpp` bindings) must remain 100% free of `<Python.h>` and build with zero warnings under `/W4 /permissive- /WX` (MSVC) or `-Wall -Wextra -Wpedantic -Werror` (GCC/Clang).
3. **Verification Before PR**: Run `python scripts/verify_oss_package.py` to confirm NDA hygiene, pure C++20 unit tests, standalone CLI synthesis, and Python parity tests all pass.
