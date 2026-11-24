#pragma once

// Pure C++20 Rule-Driven Parameterized MOS & Shared-Diffusion Row PCell Engine.
// Strictly free of <Python.h> and KLayout.
//
// Generates exact DRC-compliant multi-finger NMOS/PMOS and shared-OD matched-row
// mask artwork on the PDK manufacturing grid, together with terminal access rectangles
// and local bounding boxes, ready for direct LayoutDB / GDSII emission.

#include "clay_pnr/core/geometry.hpp"
#include "clay_pnr/layout/layout_db.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace clay_pnr::pcell {

using clay_pnr::core::Box64;
using clay_pnr::layout::LayerKey;
using clay_pnr::layout::LayoutCell;

struct MosTechRules {
    std::string pdk_id = "gf180mcu";
    int64_t grid_nm = 5;

    // Mask layers (GDS layer, datatype)
    LayerKey active{22, 0};   // COMP / diff
    LayerKey poly{30, 0};     // Poly2 / poly
    LayerKey contact{33, 0};  // Contact / licon
    LayerKey metal1{34, 0};   // Metal1 / li1
    LayerKey nplus{32, 0};    // Nplus / nsdm
    LayerKey pplus{31, 0};    // Pplus / psdm
    LayerKey nwell{21, 0};    // Nwell

    // Exact integer-nm design rules (derived from PDK tables)
    int64_t poly_ext_nm = 220;          // poly overhang past active top/bottom
    int64_t poly_to_cont_nm = 180;      // gate poly edge to S/D contact edge
    int64_t cont_size_nm = 220;         // contact square width/height
    int64_t cont_space_nm = 280;        // contact-to-contact spacing
    int64_t active_enc_cont_nm = 100;   // active enclosure around contact
    int64_t m1_enc_cont_nm = 60;        // M1 enclosure around contact
    int64_t m1_min_w_nm = 230;          // M1 minimum strap width
    int64_t implant_enc_active_nm = 230;// N+/P+ enclosure around active
    int64_t nwell_enc_active_nm = 430;  // NWELL enclosure around PMOS active
    int64_t gate_strap_space_nm = 280;  // space between active top and poly gate strap
};

// Built-in canonical PDK rule presets (GF180MCU & SKY130).
MosTechRules gf180mcu_mos_rules() noexcept;
MosTechRules sky130_mos_rules() noexcept;

struct MosSpec {
    std::string name = "M0";
    bool is_pmos = false;
    int64_t w_nm = 1000;      // per-finger channel width (nm)
    int64_t l_nm = 280;       // channel length (nm)
    int32_t fingers = 2;      // active gate fingers (>= 1)
    int32_t dummies_left = 0; // optional left dummy gate fingers
    int32_t dummies_right = 0;// optional right dummy gate fingers
};

struct MosTerminals {
    std::vector<Box64> source_rects;
    std::vector<Box64> drain_rects;
    std::vector<Box64> gate_rects;
    Box64 bbox{};
};

struct MosPCellResult {
    LayoutCell cell;
    MosTerminals terminals;
    Box64 active_bbox{};
    int64_t gate_pitch_nm = 0;
};

// Generates a single parameterized multi-finger MOS cell centered at (0, 0) on grid.
MosPCellResult generate_mos_pcell(const MosSpec& spec, const MosTechRules& rules);

struct SharedRowMember {
    std::string name;
    int32_t fingers = 2;
};

struct SharedRowResult {
    LayoutCell cell;
    Box64 bbox{};
    Box64 active_bbox{};
    std::map<std::string, MosTerminals> member_terminals;
    int64_t gate_pitch_nm = 0;
};

// Generates a shared-diffusion (continuous active island) interdigitated row for matched
// same-polarity/same-(W,L) transistors (e.g. ABBA common-centroid / interdigitated mirror)
// with optional end dummies. Zero Python, zero KLayout!
SharedRowResult generate_shared_diffusion_row(
    const std::string& row_name,
    bool is_pmos,
    int64_t w_nm,
    int64_t l_nm,
    const std::vector<std::string>& finger_owners,  // length = total active fingers, e.g. {"M1","M2","M2","M1"}
    int32_t dummies_left,
    int32_t dummies_right,
    const MosTechRules& rules);

}  // namespace clay_pnr::pcell

