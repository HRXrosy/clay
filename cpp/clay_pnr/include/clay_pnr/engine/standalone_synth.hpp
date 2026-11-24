#pragma once

// Pure C++20 Zero-Python Standalone .clay -> Physical Synthesis -> Binary GDSII Engine (Phase 4A).
// Wires:
//   parse_clay_source -> MosPCell / Passive Macro Generators -> solve_banded_area
//   -> Manhattan Multi-Layer Trunk-and-Branch Router -> LayoutDB -> GDSII v6.0
// Zero <Python.h>, zero KLayout, zero OR-Tools, zero SciPy.

#include "clay_pnr/core/geometry.hpp"
#include "clay_pnr/gds/gds_stream.hpp"
#include "clay_pnr/ir/clay_ir.hpp"
#include "clay_pnr/layout/layout_db.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace clay_pnr::engine {

struct SynthOptions {
    std::string pdk_override;       // optional: "gf180mcu" or "sky130"; empty -> use file.pdk (fallback gf180mcu)
    int64_t block_gap_nm = 2000;    // inter-block clearance in nm
    int64_t wire_width_nm = 300;    // default M2/M3 wire width in nm
    int64_t track_pitch_nm = 600;   // Manhattan channel routing track pitch in nm
    bool emit_routes = true;        // emit inter-device M2/M3 Manhattan connectivity + pin labels
    int32_t max_place_evals = 1500;
};

struct PlacedMacroInfo {
    std::string name;
    std::string kind;
    int32_t band = 0;
    core::Box64 bbox_nm{};
};

struct StandaloneSynthResult {
    bool ok = false;
    std::string error;
    std::string cell_name;
    std::string pdk_id;
    size_t device_count = 0;
    size_t net_count = 0;
    size_t routed_net_count = 0;
    size_t polygon_count = 0;
    core::Box64 die_bbox_nm{};
    int64_t grid_nm = 5;
    std::vector<PlacedMacroInfo> macros;
    gds::GdsLibrary gds_lib;
};

StandaloneSynthResult synthesize_clay_ir(
    const ir::ClayFile& file,
    const SynthOptions& opts = {});

StandaloneSynthResult synthesize_clay_source(
    const std::string& source,
    const SynthOptions& opts = {});

StandaloneSynthResult synthesize_clay_file_to_gds(
    const std::string& clay_path,
    const std::string& out_gds_path,
    const SynthOptions& opts = {});

}  // namespace clay_pnr::engine

