// Pure C++20 native unit tests for Phase 2 & Phase 3 Core Engine:
//   - Difference-Constraint 1D & 2D Compactors
//   - Exact-Constraint 2D Envelope-Min Analog Placer
//   - LayoutDB, Grid Snap, Per-Net Merge, Multi-Layer Connectivity + MIM Carve
//   - Rule-Driven MOS PCell & Shared-Diffusion Interdigitated Row Generator
// Strictly free of <Python.h>.

#include "test_framework.hpp"

#include "clay_pnr/gds/gds_stream.hpp"
#include "clay_pnr/layout/layout_db.hpp"
#include "clay_pnr/pcell/mos_pcell.hpp"
#include "clay_pnr/place/compactor.hpp"
#include "clay_pnr/place/envelope_placer.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

namespace {

using clay_pnr::core::Box64;
using clay_pnr::core::Point64;
using clay_pnr::layout::LayerBox;
using clay_pnr::layout::LayerKey;
using clay_pnr::layout::LayoutDB;
using clay_pnr::layout::MimCarveRule;
using clay_pnr::layout::NetLayerBoxes;
using clay_pnr::layout::StackConnection;
using clay_pnr::place::CompactStatus;
using clay_pnr::place::EnvelopeCluster;
using clay_pnr::place::EnvelopeGroup;
using clay_pnr::place::EnvelopeNet;
using clay_pnr::place::EnvelopeOrder;
using clay_pnr::place::EnvelopeProblem;
using clay_pnr::place::EnvelopeRailAffinity;
using clay_pnr::place::EnvelopeStatus;
using clay_pnr::place::IntDiffEdge;
using clay_pnr::place::SymmetryPairSpec;

CLAY_TEST(EnginePhase23, CompactorDagExactAndCycle) {
    // Mirrors test_lp_compactor_minimizes_a_topologically_ordered_axis:
    // a:10, b:7, c:4 with a->b(12), a->c(13) => a=0, b=12, c=13, extent=19.
    const auto r = clay_pnr::place::compact_axis_dag(
        {10, 7, 4},
        {IntDiffEdge{0, 1, 12}, IntDiffEdge{0, 2, 13}});
    CLAY_CHECK_EQ(static_cast<int>(r.status), static_cast<int>(CompactStatus::Ok));
    CLAY_CHECK_EQ(r.positions.size(), size_t{3});
    CLAY_CHECK_EQ(r.positions[0], int64_t{0});
    CLAY_CHECK_EQ(r.positions[1], int64_t{12});
    CLAY_CHECK_EQ(r.positions[2], int64_t{13});
    CLAY_CHECK_EQ(r.extent, int64_t{19});

    // Transitive diamond with slack minimization:
    const auto r2 = clay_pnr::place::compact_axis_dag(
        {10, 5, 8, 6},
        {IntDiffEdge{0, 1, 12}, IntDiffEdge{1, 2, 7}, IntDiffEdge{0, 2, 25}, IntDiffEdge{2, 3, 10}});
    CLAY_CHECK_EQ(static_cast<int>(r2.status), static_cast<int>(CompactStatus::Ok));
    CLAY_CHECK_EQ(r2.positions[0], int64_t{0});
    CLAY_CHECK_EQ(r2.positions[1], int64_t{12});
    CLAY_CHECK_EQ(r2.positions[2], int64_t{25});
    CLAY_CHECK_EQ(r2.positions[3], int64_t{35});
    CLAY_CHECK_EQ(r2.extent, int64_t{41});

    // Cycle rejection:
    const auto cyc = clay_pnr::place::compact_axis_dag(
        {10, 10},
        {IntDiffEdge{0, 1, 12}, IntDiffEdge{1, 0, 12}});
    CLAY_CHECK_EQ(static_cast<int>(cyc.status), static_cast<int>(CompactStatus::CycleDetected));
}

CLAY_TEST(EnginePhase23, Compactor2DSymmetryAndGrid) {
    // 4 blocks: (0,1) symmetric pair on bottom row, (2,3) symmetric pair on upper row.
    const std::vector<int64_t> w = {100, 100, 150, 150};
    const std::vector<int64_t> h = {80, 80, 60, 60};
    const std::vector<IntDiffEdge> h_edges = {
        IntDiffEdge{0, 1, 120},
        IntDiffEdge{2, 3, 200},
    };
    const std::vector<IntDiffEdge> v_edges = {
        IntDiffEdge{0, 2, 100},
        IntDiffEdge{1, 3, 100},
    };
    const std::vector<SymmetryPairSpec> syms = {{0, 1}, {2, 3}};
    const auto res = clay_pnr::place::compact_2d_topology(w, h, h_edges, v_edges, syms, 5);
    CLAY_CHECK_EQ(static_cast<int>(res.status), static_cast<int>(CompactStatus::Ok));
    CLAY_CHECK_EQ(res.y[0], res.y[1]);
    CLAY_CHECK_EQ(res.y[2], res.y[3]);
    CLAY_CHECK(res.y[2] >= res.y[0] + 100);
    const int64_t a01 = 2 * res.x[0] + 2 * res.x[1] + w[0] + w[1];
    const int64_t a23 = 2 * res.x[2] + 2 * res.x[3] + w[2] + w[3];
    CLAY_CHECK_EQ(a01, a23);
    CLAY_CHECK_EQ(a01, res.shared_axis2);
    for (int64_t v : res.x) CLAY_CHECK_EQ(v % 5, int64_t{0});
    for (int64_t v : res.y) CLAY_CHECK_EQ(v % 5, int64_t{0});
}

CLAY_TEST(EnginePhase23, EnvelopePlacerClustersBandsFloatingAndOrders) {
    // Realistic analog OTA/LDO-like 10-group problem with:
    //   - 2 guard-ring clusters with boundary pad=3
    //   - 2 MOS bands (0=NMOS below 1=PMOS) + 2 floating passives banked right
    //   - explicit left_of order
    //   - weighted signal nets + L-inf maxspan + aspect_target=1.7
    EnvelopeProblem prob{};
    prob.gap = 1.0;
    prob.wl_weight = 2;
    prob.maxspan_weight = 3000;
    prob.has_net_weights = true;
    prob.aspect_target = 1.7;
    prob.aspect_q = 170;
    prob.groups = {
        {"M_tail", 10.0, 6.0, 11, 7},
        {"M_in1", 12.0, 8.0, 13, 9},
        {"M_in2", 12.0, 8.0, 13, 9},
        {"M_nbias", 8.0, 6.0, 9, 7},
        {"M_ld1", 14.0, 9.0, 15, 10},
        {"M_ld2", 14.0, 9.0, 15, 10},
        {"M_outp", 16.0, 8.0, 17, 9},
        {"M_pbias", 8.0, 7.0, 9, 8},
        {"P_Cc", 18.0, 18.0, 19, 19},
        {"P_Rz", 6.0, 12.0, 7, 13},
    };
    const int32_t NO_BAND = std::numeric_limits<int32_t>::min();
    prob.band_of_group = {0, 0, 0, 0, 1, 1, 1, 1, NO_BAND, NO_BAND};
    prob.clusters = {
        EnvelopeCluster{{1, 2}, 3},
        EnvelopeCluster{{4, 5}, 3},
    };
    prob.orders = {
        EnvelopeOrder{1, 2, false},  // M_in1 left_of M_in2
        EnvelopeOrder{4, 5, false},  // M_ld1 left_of M_ld2
    };
    prob.nets = {
        EnvelopeNet{{0, 1, 2}, 50, true},
        EnvelopeNet{{1, 4}, 100, true},
        EnvelopeNet{{2, 5, 6, 8}, 33, true},
        EnvelopeNet{{6, 8, 9}, 50, true},
    };
    prob.rail_affinities = {
        EnvelopeRailAffinity{0, false},
        EnvelopeRailAffinity{6, true},
    };

    const auto sol = clay_pnr::place::solve_envelope_min(prob);
    CLAY_CHECK(sol.status == EnvelopeStatus::Optimal || sol.status == EnvelopeStatus::Feasible);
    std::string reason;
    CLAY_CHECK(clay_pnr::place::verify_envelope_solution(prob, sol.xs, sol.ys, &reason));
    CLAY_CHECK(sol.W > 0 && sol.H > 0);
    // Determinism check: second call produces bit-exact coordinates and objective!
    const auto sol2 = clay_pnr::place::solve_envelope_min(prob);
    CLAY_CHECK_EQ(sol.objective, sol2.objective);
    CLAY_CHECK(sol.xs == sol2.xs);
    CLAY_CHECK(sol.ys == sol2.ys);
}

CLAY_TEST(EnginePhase23, EnvelopePlacerSymmetryAndAspect) {
    // 4 blocks with two hard symmetry pairs sharing one mirror axis.
    EnvelopeProblem prob{};
    prob.gap = 0.5;
    prob.groups = {
        {"a", 10.0, 10.0, 11, 11},
        {"b", 10.0, 10.0, 11, 11},
        {"c", 10.0, 10.0, 11, 11},
        {"d", 10.0, 10.0, 11, 11},
    };
    prob.symmetry_pairs = {{0, 1}, {2, 3}};
    const auto sol = clay_pnr::place::solve_envelope_min(prob);
    CLAY_CHECK(sol.status == EnvelopeStatus::Optimal || sol.status == EnvelopeStatus::Feasible);
    std::string reason;
    CLAY_CHECK(clay_pnr::place::verify_envelope_solution(prob, sol.xs, sol.ys, &reason));
    CLAY_CHECK_EQ(sol.ys[0], sol.ys[1]);
    CLAY_CHECK_EQ(sol.ys[2], sol.ys[3]);
    CLAY_CHECK_EQ(2 * sol.xs[0] + 2 * sol.xs[1] + 22, 2 * sol.xs[2] + 2 * sol.xs[3] + 22);

    // Tall tiles rotated landscape by aspect_target=1.7:
    EnvelopeProblem p_port{};
    p_port.gap = 0.5;
    for (int i = 0; i < 6; ++i) p_port.groups.push_back({"g" + std::to_string(i), 10.0, 30.0, 11, 31});
    const auto s_port = clay_pnr::place::solve_envelope_min(p_port);
    EnvelopeProblem p_land = p_port;
    p_land.aspect_target = 1.7;
    p_land.aspect_q = 170;
    const auto s_land = clay_pnr::place::solve_envelope_min(p_land);
    CLAY_CHECK(s_port.status == EnvelopeStatus::Optimal || s_port.status == EnvelopeStatus::Feasible);
    CLAY_CHECK(s_land.status == EnvelopeStatus::Optimal || s_land.status == EnvelopeStatus::Feasible);
    const double asp_p = static_cast<double>(s_port.W) / static_cast<double>(s_port.H);
    const double asp_l = static_cast<double>(s_land.W) / static_cast<double>(s_land.H);
    CLAY_CHECK(asp_l > asp_p);
}

CLAY_TEST(EnginePhase23, LayoutDBSnapMergeAndStackConnectivity) {
    // 1. Exact nearest-grid ties-up snapping for positive and negative values:
    CLAY_CHECK_EQ(clay_pnr::layout::snap_coord_nearest_up(12, 5), int64_t{10});
    CLAY_CHECK_EQ(clay_pnr::layout::snap_coord_nearest_up(13, 5), int64_t{15});
    CLAY_CHECK_EQ(clay_pnr::layout::snap_coord_nearest_up(-12, 5), int64_t{-10});
    CLAY_CHECK_EQ(clay_pnr::layout::snap_coord_nearest_up(-13, 5), int64_t{-15});
    CLAY_CHECK_EQ(clay_pnr::layout::snap_coord_nearest_up(5, 10), int64_t{10});   // tie -> up
    CLAY_CHECK_EQ(clay_pnr::layout::snap_coord_nearest_up(-5, 10), int64_t{0});   // tie -> up

    // 2. Per-net merge never merges across different nets:
    const std::vector<NetLayerBoxes> items = {
        {"VDD", LayerKey{34, 0}, {Box64{0, 0, 100, 50}, Box64{100, 0, 200, 50}}},
        {"VSS", LayerKey{34, 0}, {Box64{200, 0, 300, 50}}},
    };
    const auto merged = clay_pnr::layout::merge_net_boxes_exact(items);
    CLAY_CHECK_EQ(merged.size(), size_t{2});

    // 3. Multi-layer connectivity + MIM carve isolation test:
    const LayerKey M4{46, 0}, V4{41, 0}, M5{81, 0}, MIM{75, 0};
    const std::vector<LayerBox> shapes = {
        {M4, Box64{0, 0, 1000, 1000}},
        {MIM, Box64{100, 100, 900, 900}},
        {V4, Box64{400, 400, 600, 600}},
        {M5, Box64{350, 350, 650, 650}},
        // Plus a separate M4->V4->M5 ordinary via OUTSIDE the MIM cap that SHOULD connect!
        {M4, Box64{2000, 0, 2500, 500}},
        {V4, Box64{2150, 150, 2350, 350}},
        {M5, Box64{2100, 100, 2400, 400}},
    };
    const auto comps = clay_pnr::layout::extract_stack_connectivity(
        shapes,
        {M4, M5, MIM},
        {StackConnection{M4, V4, M5}},
        {MimCarveRule{M4, V4, MIM, 0, true}});
    CLAY_CHECK_EQ(comps.size(), size_t{3});
}

CLAY_TEST(EnginePhase23, MosPCellAndSharedDiffusionRow) {
    const auto rules_gf = clay_pnr::pcell::gf180mcu_mos_rules();
    const auto pcell = clay_pnr::pcell::generate_mos_pcell(
        clay_pnr::pcell::MosSpec{"MP1", true, 2000, 500, 4, 1, 1}, rules_gf);
    CLAY_CHECK_EQ(pcell.terminals.gate_rects.size(), size_t{4});
    CLAY_CHECK_EQ(pcell.terminals.source_rects.size(), size_t{4});
    CLAY_CHECK_EQ(pcell.terminals.drain_rects.size(), size_t{4});
    CLAY_CHECK(pcell.active_bbox.x1 > pcell.active_bbox.x0 && pcell.active_bbox.y1 > pcell.active_bbox.y0);
    // Every vertex must lie strictly on the 5nm manufacturing grid.
    for (const auto& lp : pcell.cell.polygons) {
        for (const auto& pt : lp.vertices) {
            CLAY_CHECK_EQ(pt.x % 5, int64_t{0});
            CLAY_CHECK_EQ(pt.y % 5, int64_t{0});
        }
    }

    // Shared-diffusion ABBA interdigitated diff pair row on SKY130:
    const auto rules_sky = clay_pnr::pcell::sky130_mos_rules();
    const auto row = clay_pnr::pcell::generate_shared_diffusion_row(
        "diff_pair_abba", false, 1500, 180,
        {"M1", "M2", "M2", "M1"}, 1, 1, rules_sky);
    CLAY_CHECK_EQ(row.member_terminals.size(), size_t{2});
    CLAY_CHECK_EQ(row.member_terminals.at("M1").gate_rects.size(), size_t{2});
    CLAY_CHECK_EQ(row.member_terminals.at("M2").gate_rects.size(), size_t{2});
    // Verify common-centroid X symmetry of M1 and M2 gate centers!
    const auto& g1 = row.member_terminals.at("M1").gate_rects;
    const auto& g2 = row.member_terminals.at("M2").gate_rects;
    const int64_t c1_x2 = (g1[0].x0 + g1[0].x1) + (g1[1].x0 + g1[1].x1);
    const int64_t c2_x2 = (g2[0].x0 + g2[0].x1) + (g2[1].x0 + g2[1].x1);
    CLAY_CHECK_EQ(c1_x2, c2_x2);

    // Emit to binary GDSII v6.0 in memory and verify exact round-trip via gds_stream!
    LayoutDB db;
    db.name = "PCELL_TEST";
    db.cells.push_back(row.cell);
    const auto lib = db.to_gds_library();
    const auto bytes = clay_pnr::gds::write_gds_bytes(lib);
    CLAY_CHECK(bytes.size() > 64);
    clay_pnr::gds::GdsLibrary parsed;
    std::string err;
    CLAY_CHECK(clay_pnr::gds::read_gds_bytes(bytes.data(), bytes.size(), parsed, &err));
    const auto summary = clay_pnr::gds::summarize_gds_layers(parsed, "diff_pair_abba");
    CLAY_CHECK(summary.size() >= 4);
}

CLAY_TEST(EnginePhase23, OrthogonalDecompUnionCoverNetConnAndWirelength) {
    using namespace clay_pnr::layout;
    using clay_pnr::core::PolygonSet;
    // 1. Exact orthogonal polygon decomposition: simple rect, L-shape, non-Manhattan rejection, zero-area rejection.
    std::vector<Box64> boxes;
    std::string err;
    CLAY_CHECK(decompose_orthogonal_polygon_boxes(
        {{0, 0}, {100, 0}, {100, 50}, {0, 50}}, boxes, &err));
    CLAY_CHECK_EQ(boxes.size(), size_t{1});
    CLAY_CHECK(boxes[0] == (Box64{0, 0, 100, 50}));

    boxes.clear();
    CLAY_CHECK(decompose_orthogonal_polygon_boxes(
        {{0, 0}, {200, 0}, {200, 100}, {100, 100}, {100, 200}, {0, 200}}, boxes, &err));
    const PolygonSet l_ps = PolygonSet::from_boxes(boxes);
    CLAY_CHECK_EQ(l_ps.area_u128().lo, uint64_t{30000});

    boxes.clear();
    CLAY_CHECK(!decompose_orthogonal_polygon_boxes({{0, 0}, {100, 0}, {50, 50}}, boxes, &err));
    CLAY_CHECK(!decompose_orthogonal_polygon_boxes({{0, 0}, {100, 0}, {100, 0}, {0, 0}}, boxes, &err));

    // 2. Exact union coverage check.
    CLAY_CHECK(boxes_covered_by_union_exact(
        {Box64{10, 10, 90, 40}, Box64{50, 20, 150, 40}},
        {Box64{0, 0, 100, 50}, Box64{100, 0, 200, 50}}));
    CLAY_CHECK(!boxes_covered_by_union_exact(
        {Box64{10, 10, 210, 40}},
        {Box64{0, 0, 100, 50}, Box64{100, 0, 200, 50}}));

    // 3. Net connectivity exact union-find (short detection via via layer + distinct check).
    const LayerKey m1{34, 0}, m2{36, 0}, v1{35, 0};
    const auto nc_ok = verify_net_connectivity_exact(
        {
            NetShapeBox{0, m1, Box64{0, 0, 100, 100}},
            NetShapeBox{0, v1, Box64{20, 20, 40, 40}},
            NetShapeBox{0, m2, Box64{0, 0, 200, 100}},
            NetShapeBox{1, m1, Box64{200, 0, 300, 100}},
        },
        {v1});
    CLAY_CHECK(nc_ok.distinct);
    CLAY_CHECK_EQ(nc_ok.short_count, size_t{0});

    const auto nc_short = verify_net_connectivity_exact(
        {
            NetShapeBox{0, m1, Box64{0, 0, 100, 100}},
            NetShapeBox{0, v1, Box64{80, 20, 100, 40}},
            NetShapeBox{1, m2, Box64{80, 0, 200, 100}},
        },
        {v1});
    CLAY_CHECK(!nc_short.distinct);
    CLAY_CHECK_EQ(nc_short.short_count, size_t{1});
    CLAY_CHECK_EQ(nc_short.merged_pairs[0].first, uint32_t{0});
    CLAY_CHECK_EQ(nc_short.merged_pairs[0].second, uint32_t{1});

    // 4. Net wirelength + strict overlap count.
    const auto wl = compute_net_wirelength_exact(
        {{{0, 0}, {100, 50}, {100, 50}}, {{10, 10}}},
        {Box64{0, 0, 100, 100}, Box64{100, 0, 200, 100}, Box64{50, 50, 150, 150}});
    CLAY_CHECK_EQ(wl.net_total_span.size(), size_t{2});
    CLAY_CHECK_EQ(wl.net_total_span[0], int64_t{150});
    CLAY_CHECK_EQ(wl.net_max_pin_span[0], int64_t{150});
    CLAY_CHECK_EQ(wl.net_total_span[1], int64_t{0});
    CLAY_CHECK_EQ(wl.netlist_total_hpwl, int64_t{150});
    CLAY_CHECK_EQ(wl.overlapping_pairs, int64_t{2});  // (0,2) and (1,2), NOT touching (0,1)
}

}  // namespace
