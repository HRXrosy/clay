# Open-Source Signoff Environment (Magic + Netgen + KLayout + ngspice + SKY130A + GF180MCUD)

Clay integrates directly with the open-source [`hpretl/iic-osic-tools`](https://github.com/iic-jku/IIC-OSIC-TOOLS) Docker image, which bundles:

- **Magic VLSI** (DRC + parasitic extraction `ext2spice`)
- **Netgen** (LVS netlist equivalence checking)
- **KLayout** (DRC / LVS / GDSII inspection)
- **ngspice** (pre-layout & post-layout SPICE simulation)
- **Open PDKs**: `sky130A` (`/foss/pdks/sky130A`) and `gf180mcuD` (`/foss/pdks/gf180mcuD`)

## Pull the Image

```bash
docker pull hpretl/iic-osic-tools:latest
```

## Run Full Open-Source Signoff on a Clay Benchmark

```bash
python -m clay flow run --case benchmarks/gf180mcu/two_stage_ota --force
```

Or invoke the standalone Docker Magic/Netgen/ngspice signoff helper directly:

```bash
python scripts/run_docker_magic_signoff.py --help
```
