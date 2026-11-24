# Clay — Open-Source Analog Intent Compiler & C++20 PnR Engine (SKY130 / GF180MCU)

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)
[![ASPDAC '27](https://img.shields.io/badge/ASPDAC_'27-10.1145%2F3840404.3860941-informational.svg)](https://doi.org/10.1145/3840404.3860941)
[![C++20](https://img.shields.io/badge/Engine-C%2B%2B20-00599C.svg)](cpp/clay_pnr/)
[![Open PDKs](https://img.shields.io/badge/Open_PDKs-SKY130_%7C_GF180MCU-2ea44f.svg)](benchmarks/)

**Clay (`v0.4.45` Open-Source Edition)** is the official open-source release accompanying our ASP-DAC '27 paper *"CLAY: A Programming Model for Intent-Driven Analog Circuit Design Automation"*. Clay compiles declarative analog circuit & physical layout intent (`.clay`) into **DRC/LVS-clean, post-layout-verified GDSII** targeting open-source silicon nodes:

- **GlobalFoundries GF180MCU** (180 nm 3.3V/6V CMOS, 5LM/6LM, MIM caps, precision poly resistors, vertical PNP BJTs)
- **SkyWater SKY130** (130 nm 1.8V CMOS, `sky130A`, LVT/HVT/varactor/MIM devices)

> **Open-PDK-Only Hygiene**: This distribution contains **zero proprietary foundry rules, decks, layer maps, or model files**. All PDK adapters, BSIM `gm/Id` lookup tables (`data/pdk_luts/`), benchmarks (`benchmarks/`), and C++20 PCell/GDSII kernels exclusively target open-source **SKY130** and **GF180MCU**.

---

## Architecture Overview

```text
  SPICE Netlist / .clay Design
         │
         ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │ 1. Frontend & Intent Synthesis   (clay/frontend/, clay/parser_*)    │
  │    - Automatic role & craft inference (`clay onboard`)              │
  │    - Dual parsers: Python/Lark & zero-Python C++20 recursive-descent│
  └──────────────────────────────┬──────────────────────────────────────┘
                                 ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │ 2. Pure C++20 Main Engine        (cpp/clay_pnr/ — 33k+ LOC, /WX)    │
  │    - Exact Manhattan PolygonSet booleans, morphology & contours     │
  │    - Native binary GDSII v6.0 reader/writer & hierarchical flatten  │
  │    - Rule-driven GF180MCU & SKY130 MOS PCell & shared-OD rows       │
  │    - 1D/2D Compactor, 2D Envelope Placer, Banded Macro Placer       │
  │    - PathFinder (negotiated congestion) & Gridless Detailed Router  │
  │    - Standalone zero-Python binary: `clay_engine_cli`               │
  └──────────────────────────────┬──────────────────────────────────────┘
                                 ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │ 3. 8-Step Public Flow & Signoff  (clay/flow/, tools/, docker/)      │
  │    Compile -> Layout -> DRC -> LVS -> PEX -> PostSim -> Intent ->   │
  │    Sign (KLayout + Docker `hpretl/iic-osic-tools`: Magic/Netgen/    │
  │    ngspice)                                                         │
  └─────────────────────────────────────────────────────────────────────┘
```

---

## Quickstart

### 1. Install Python Dependencies & Build the C++20 Main Engine

Requires **Python 3.12+** and a **C++20 compiler** (MSVC 2022 on Windows, GCC 11+ or Clang 14+ on Linux/macOS).

```bash
pip install -e ".[dev]"

# Compile all 4 native C++20 extensions (_engine_cpp, _place_cpp, _pathfinder_cpp, _gridless_cpp),
# standalone zero-Python `clay_engine_cli`, and run 51 pure C++20 unit tests:
python cpp/clay_pnr/build.py --target all --werror --test
```

### 2. Run Standalone Zero-Python Synthesis (Works Immediately Without External PDK Downloads)

```bash
# Self-test the C++20 engine (~2 ms)
./cpp/clay_pnr/build/modular/bin/clay_engine_cli selftest

# Synthesize a real 20-device Folded-Cascode OTA directly to GDSII v6.0 in <10 ms:
./cpp/clay_pnr/build/modular/bin/clay_engine_cli synth \
    benchmarks/gf180mcu/folded_cascode_ota/source/design.clay \
    -o fc_ota_gf180.gds

# Retarget the Two-Stage OTA directly to SKY130:
./cpp/clay_pnr/build/modular/bin/clay_engine_cli synth \
    benchmarks/gf180mcu/two_stage_ota/source/design.clay \
    --pdk sky130 -o two_stage_ota_sky130.gds
```

### 3. Connect Open PDKs (SKY130 & GF180MCU)

Clay reads open PDK roots from `clay.toml` (`[tool.clay]`) or environment variables (`CLAY_GF180MCU_PATH`, `CLAY_PDK_PATH`, `PDK_ROOT`):

```bash
# Detect existing open_pdks / volare / IIC-OSIC installs or bootstrap SKY130 & GF180MCU:
python scripts/setup_open_pdks.py --detect --write-config

# Or automatically download upstream Apache-2.0 GF180MCU / SKY130 PDKs:
python scripts/setup_open_pdks.py --install-gf180 --install-sky130
```

See [`docker/README.md`](docker/README.md) to run full Magic DRC / Netgen LVS / PEX / `ngspice` post-layout signoff using the open-source `hpretl/iic-osic-tools` Docker image (which bundles SKY130A and GF180MCUD pre-installed).

### 4. Validate Benchmarks & Run the Physical Flow

```bash
# Validate any open-PDK benchmark package
python -m clay case validate benchmarks/gf180mcu/two_stage_ota
python -m clay case validate benchmarks/sky130/five_t_ota

# Run full 8-step physical synthesis & signoff (with open PDK + KLayout/Docker)
python -m clay flow run --case benchmarks/gf180mcu/two_stage_ota --force
```

### 5. Run the Full Self-Verification Suite

```bash
# Zero-tolerance NDA audit + C++20 /WX build & 51 native tests + CLI synth + 140 pytest checks
python scripts/verify_oss_package.py
```

---

## Included Open-PDK Benchmarks (`benchmarks/`)

| PDK Node | Benchmark | Topology / Class | Path |
|---|---|---|---|
| **GF180MCU** | `two_stage_ota` | Two-Stage Miller OTA (shared-OD, dense contacts, 8/8 Docker signoff) | [`benchmarks/gf180mcu/two_stage_ota`](benchmarks/gf180mcu/two_stage_ota) |
| **GF180MCU** | `folded_cascode_ota` | Folded-Cascode OTA (20 devices, 8/8 Docker signoff) | [`benchmarks/gf180mcu/folded_cascode_ota`](benchmarks/gf180mcu/folded_cascode_ota) |
| **GF180MCU** | `ldo_regulator` | Low-Dropout Regulator (32-finger pass-FET + 3D MIM-over-resistor, 8/8 signoff) | [`benchmarks/gf180mcu/ldo_regulator`](benchmarks/gf180mcu/ldo_regulator) |
| **GF180MCU** | `bandgap_reference` | BJT Bandgap Reference (vertical PNP + poly resistors, 8/8 signoff) | [`benchmarks/gf180mcu/bandgap_reference`](benchmarks/gf180mcu/bandgap_reference) |
| **GF180MCU** | `telescopic_ota` | Telescopic Cascode OTA | [`benchmarks/gf180mcu/telescopic_ota`](benchmarks/gf180mcu/telescopic_ota) |
| **GF180MCU** | `cs_dac_4bit` | 4-bit Current-Steering DAC | [`benchmarks/gf180mcu/cs_dac_4bit`](benchmarks/gf180mcu/cs_dac_4bit) |
| **GF180MCU** | `five_t_ota` | 5-Transistor OTA (with L5-signoff GDS + Magic PEX + ngspice KPIs) | [`benchmarks/gf180mcu/five_t_ota`](benchmarks/gf180mcu/five_t_ota) |
| **SKY130** | `quickstart_amp` | Quickstart Miller Amplifier | [`benchmarks/sky130/quickstart_amp`](benchmarks/sky130/quickstart_amp) |
| **SKY130** | `five_t_ota` | 5-Transistor OTA | [`benchmarks/sky130/five_t_ota`](benchmarks/sky130/five_t_ota) |
| **SKY130** | `folded_cascode_ota` | Folded-Cascode OTA | [`benchmarks/sky130/folded_cascode_ota`](benchmarks/sky130/folded_cascode_ota) |
| **SKY130** | `telescopic_cascode_ota` | Telescopic Cascode OTA | [`benchmarks/sky130/telescopic_cascode_ota`](benchmarks/sky130/telescopic_cascode_ota) |
| **SKY130** | `fully_differential_ota` | Fully Differential OTA | [`benchmarks/sky130/fully_differential_ota`](benchmarks/sky130/fully_differential_ota) |
| **SKY130** | `ldo_regulator` | AnalogGym LDO Regulator | [`benchmarks/sky130/ldo_regulator`](benchmarks/sky130/ldo_regulator) |
| **SKY130** | `bandgap_reference` | AnalogGym Bandgap Reference | [`benchmarks/sky130/bandgap_reference`](benchmarks/sky130/bandgap_reference) |
| **SKY130** | `lc_tank_vco` | LC-Tank VCO with Accumulation-Mode Varactors | [`benchmarks/sky130/lc_tank_vco`](benchmarks/sky130/lc_tank_vco) |
| **SKY130** | `bias_tree` | Hierarchical Bias Network | [`benchmarks/sky130/bias_tree`](benchmarks/sky130/bias_tree) |

---

## Citation

If you use Clay in your research or open-source tooling, please cite our ASP-DAC '27 paper:

```bibtex
@inproceedings{bi2027clay,
  author    = {Zhaori Bi},
  title     = {CLAY: A Programming Model for Intent-Driven Analog Circuit Design Automation},
  booktitle = {Proceedings of the 32nd Asia and South Pacific Design Automation Conference (ASPDAC '27)},
  year      = {2027},
  month     = jan,
  location  = {Tokyo, Japan},
  publisher = {ACM},
  doi       = {10.1145/3840404.3860941},
  isbn      = {979-8-4007-2923-2/2027/01}
}
```

---

## License (Strong Copyleft — GNU AGPLv3)

Copyright © 2026 Zhaori Bi, Fudan University.

Clay is free software licensed under the **[GNU Affero General Public License v3.0 (AGPL-3.0-only)](LICENSE)**.

- **Strong Copyleft**: Any redistribution, modification, or derivative work based on Clay must be licensed under **GNU AGPL-3.0** with complete corresponding source code made available.
- **Network / SaaS Clause (AGPLv3 §13)**: If you run a modified version of Clay as a remote network service, cloud EDA backend, or AI agent API accessible to external users, you must prominently offer those users the opportunity to receive the complete corresponding source code of your modified version under AGPL-3.0.

External upstream open-source PDKs (downloaded separately, not vendored in this repository):
- **SkyWater SKY130 PDK** (`google/skywater-pdk`, Apache-2.0)
- **GlobalFoundries GF180MCU PDK** (`google/gf180mcu-pdk`, Apache-2.0)
