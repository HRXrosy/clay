# Clay Open-PDK Benchmark Suite (GF180MCU & SKY130)

All circuits in this directory are pure open-PDK `.clay` designs targeting **GF180MCU** or **SKY130**.

## Quick Commands

```bash
# Parse & inspect any benchmark with the zero-Python C++20 CLI
./cpp/clay_pnr/build/modular/bin/clay_engine_cli parse benchmarks/gf180mcu/folded_cascode_ota/source/design.clay

# Standalone zero-Python C++20 synthesis to GDSII (GF180MCU or SKY130)
./cpp/clay_pnr/build/modular/bin/clay_engine_cli synth benchmarks/gf180mcu/two_stage_ota/source/design.clay -o out_gf180.gds
./cpp/clay_pnr/build/modular/bin/clay_engine_cli synth benchmarks/gf180mcu/two_stage_ota/source/design.clay --pdk sky130 -o out_sky130.gds

# Full 8-step public flow (requires open PDK + KLayout / Docker signoff tools)
python -m clay flow check --case benchmarks/gf180mcu/two_stage_ota
python -m clay flow run   --case benchmarks/gf180mcu/two_stage_ota --force
```
