# Clay PDK adapters + unified install layout

Clay keeps **two things separate** for PDKs:

1. **In-tree, committable** (under `clay/pdk/`):
   - `gf180mcu.py` (and future `<pdk_id>.py`) — small adapter that resolves the PDK install path from the environment and wires Clay's drivers/router to that process.
   - `gf180mcu_layout_intent.yaml` (and future `<pdk_id>_layout_intent.yaml`) — process-specific layout gotchas + design intent (see `clay/layout/layout_knowledge.yaml` for the agnostic principles).
   - **No PDK content** ever lives here. These are ~50–100 line glue files, license-clean.

2. **Out-of-tree, gitignored** (under `<repo>/pdks/`):
   - Each PDK's actual install (models, GDS cells, KLayout decks, OPC files, etc.). Often multi-GB.
   - `/pdks/` is in `.gitignore` so PDK content **cannot enter git history**, regardless of whether the PDK is open-source (gf180mcu, sky130) or proprietary (open_pdk, open_pdk, GlobalFoundries non-OS, ...). Legal/IP safe by construction.

## The unified `pdks/` layout

```
EDALLM/
├── clay/pdk/                  # committed: adapters + intent yaml
│   ├── gf180mcu.py
│   ├── gf180mcu_layout_intent.yaml
│   ├── sky130.py              (add when needed)
│   ├── sky130_layout_intent.yaml
│   └── <new_pdk>.py / .yaml
└── pdks/                      # GITIGNORED: the real PDK installs
    ├── gf180mcu/              # the AnalogGym gf180mcu_pdk tree
    ├── sky130/                # the AnalogGym sky130_pdk tree
    └── <new_pdk>/             # any proprietary PDK you drop in
```

## Resolving a PDK at runtime

Every Clay adapter looks at **environment variables first**, then a small list of
in-tree fallbacks. Set one env per PDK you use:

```bash
# gf180mcu
export CLAY_PDK_ID=gf180mcu
export CLAY_GF180MCU_PATH=<repo>/pdks/gf180mcu      # absolute path

# A proprietary PDK (example: open_pdk)
export CLAY_PDK_ID=open_pdk
export CLAY_TSMC65_PATH=<repo>/pdks/open_pdk
# or generic
export CLAY_PDK_PATH=<repo>/pdks/open_pdk
```

`circuits/_harness.py` and `clay/pdk/<pdk_id>.py` will pick the env-resolved
install up automatically. No `.cir` / `.sp` / Python source needs to know the
absolute install path.

## Adding a new PDK

1. Drop the install at `<repo>/pdks/<pdk_id>/`. Confirm it's ignored:
   `git check-ignore pdks/<pdk_id>` → should print the path.
2. Add `clay/pdk/<pdk_id>.py` following `gf180mcu.py`:
   - Env vars (`CLAY_<PDK_ID>_PATH`).
   - `_resolve_pdk_root()`.
   - Corner-file helper, model file names, scale convention.
3. Add `clay/pdk/<pdk_id>_layout_intent.yaml` (start empty; capture gotchas as
   you hit them — `m=` scaling, bin ceilings, model name aliasing, scale
   convention, well-tap quirks, etc.). The layout-diagnostics tool reads it
   automatically via `clay.layout.layout_diagnostics.load_pdk_layout_intent`.
4. The diagnostics + reproducibility tooling needs no changes; it's already
   PDK-id-parameterised.

## Why gitignored even for open-source PDKs

The gf180mcu and sky130 PDKs ARE open-source, but they're:
- 2.1 GB (gf180mcu) / 53 MB (sky130) — too big to track meaningfully in git.
- Already distributed via their upstream repos / volare / IIC-OSIC. Tracking a
  copy here would just rot.

The Clay design treats every PDK as an external dependency the user installs +
points at via env. Open vs proprietary is then purely a redistribution policy
question — Clay's code paths are identical.
