# Contributing to Clay (Open-Source Edition)

Thank you for contributing to Clay! External contributions are welcome via **Fork & Pull Request** (or direct branch pushes for invited collaborators).

## How to Contribute (Fork & Pull Request)

1. **Fork** [`bzr2915/clay`](https://github.com/bzr2915/clay) to your GitHub account and clone your fork:
   ```bash
   git clone https://github.com/<your-username>/clay.git
   cd clay
   git checkout -b feat/my-change
   ```
2. **Build & test locally** (Python 3.12+ and C++20 compiler required):
   ```bash
   pip install -e ".[dev]"
   python scripts/verify_oss_package.py
   ```
3. **Push to your fork and open a Pull Request** against `bzr2915/clay:main`. GitHub Actions (`Clay OSS CI`) will automatically check NDA hygiene, C++20 `-Werror` build, all 51 pure C++20 native unit tests, standalone CLI synthesis, and the full pytest suite.

> Need direct branch push access as a regular collaborator? Open an Issue or contact `zhaori_bi@fudan.edu.cn` with your GitHub handle.

## License (GNU AGPLv3)

All contributions submitted to Clay are licensed under the **[GNU Affero General Public License v3.0 (AGPL-3.0-only)](LICENSE)**. By submitting a pull request, you confirm that you have the right to license your contribution under AGPL-3.0.

## Open-PDK-Only Policy

1. **No Proprietary Foundry Data**: Never commit commercial PDK rules, layer maps, SPICE model cards, DRC/LVS decks, or NDA-restricted measurements. Only open-source PDKs (such as **SKY130** and **GF180MCU**) are accepted.
2. **C++20 Core Purity**: All files in `cpp/clay_pnr/include/` and core `cpp/clay_pnr/src/` (except `py_*` / `*_py.cpp` bindings) must remain 100% free of `<Python.h>` and build with zero warnings under `/W4 /permissive- /WX` (MSVC) or `-Wall -Wextra -Wpedantic -Werror` (GCC/Clang).
3. **Verification Before PR**: Run `python scripts/verify_oss_package.py` to confirm NDA hygiene, pure C++20 unit tests, standalone CLI synthesis, and Python parity tests all pass.
