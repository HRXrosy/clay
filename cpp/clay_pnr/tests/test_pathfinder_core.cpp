// Pure C++20 native unit tests for PathFinderGridCore and RectClipperCore.
// Built and executed completely standalone without any CPython dependency.

#include "test_framework.hpp"

#include "clay_pnr/core/geometry.hpp"
#include "clay_pnr/core/status.hpp"
#include "clay_pnr/route/pathfinder_core.hpp"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

using namespace clay_pnr::core;
using namespace clay_pnr::route;

namespace {

PathFinderGridCore make_standard_3layer_grid(int32_t nx, int32_t ny, int64_t pitch_nm = 560) {
    PathFinderGridCore g;
    const bool ok = g.init(
        nx, ny, 3,
        0, 0, pitch_nm,
        /*pref_h=*/{1, 0, 1},
        /*min_w=*/{230, 280, 280},
        /*min_s=*/{230, 280, 280},
        /*v_cut=*/{260, 260},
        /*v_eb=*/{60, 60},
        /*v_ea=*/{60, 60},
        /*v_sp=*/{260, 260}
    );
    CLAY_REQUIRE(ok);
    return g;
}

}  // namespace

CLAY_TEST(PathFinderCore, GridConstructionIndexingAndRounding) {
    PathFinderGridCore bad;
    // Reject non-positive dims, too many layers, mismatched rule lengths, and overflow.
    CLAY_CHECK(!bad.init(0, 10, 2, 0, 0, 500, {1, 0}, {200, 200}, {200, 200}, {200}, {50}, {50}, {200}));
    CLAY_CHECK(!bad.init(10, 10, 31, 0, 0, 500, std::vector<uint8_t>(31, 1),
                         std::vector<int32_t>(31, 200), std::vector<int32_t>(31, 200),
                         std::vector<int32_t>(30, 200), std::vector<int32_t>(30, 50),
                         std::vector<int32_t>(30, 50), std::vector<int32_t>(30, 200)));
    CLAY_CHECK(!bad.init(10, 10, 2, 0, 0, 500, {1}, {200, 200}, {200, 200}, {200}, {50}, {50}, {200}));

    PathFinderGridCore g = make_standard_3layer_grid(12, 8, 500);
    CLAY_CHECK_EQ(g.nxy, 96);
    CLAY_CHECK_EQ(g.ntotal, 288);
    CLAY_CHECK_EQ(g.nvia_total, 192);

    const int32_t idx = g.cell_idx(7, 5, 2);
    int32_t ix = -1, iy = -1, gi = -1;
    g.unpack_idx(idx, ix, iy, gi);
    CLAY_CHECK_EQ(ix, 7);
    CLAY_CHECK_EQ(iy, 5);
    CLAY_CHECK_EQ(gi, 2);

    // Verify banker's rounding (py_round_i64) on um->nm conversion inside add_obs_rect.
    g.add_obs_rect(0, 0.0025, 0.0035, 1.0005, 1.0015, 1);
    CLAY_REQUIRE(g.obs_by_gi[0].size() == 1U);
    CLAY_CHECK_EQ(g.obs_by_gi[0][0].bx0_nm, 2);     // round(2.5) == 2
    CLAY_CHECK_EQ(g.obs_by_gi[0][0].by0_nm, 4);     // round(3.5) == 4
    CLAY_CHECK_EQ(g.obs_by_gi[0][0].bx1_nm, 1000);  // round(1000.5) == 1000
    CLAY_CHECK_EQ(g.obs_by_gi[0][0].by1_nm, 1002);  // round(1001.5) == 1002
}

CLAY_TEST(PathFinderCore, ObstacleEnvelopeAndMimRasterization) {
    PathFinderGridCore g = make_standard_3layer_grid(16, 16, 500);
    // Floorplan envelope restricting cells strictly inside [500, 7000] x [500, 7000].
    g.set_envelope(500, 500, 7000, 7000);
    // Foreign metal obstacle at (2.0um..3.0um, 2.0um..3.0um) on layer 0 owned by net 1.
    g.add_obs_rect(0, 2.0, 2.0, 3.0, 3.0, /*owner_id=*/1);
    // MIM clearance rect on layer 1 owned by net 0 (covering [4000, 5000] with clr=400nm).
    g.add_mim_clear_rect(1, 4000, 4000, 5000, 5000, 400, /*owner_id=*/0);
    // MIM forbidden cut rect on via layer 0 covering [1500, 2000] x [5000, 5500].
    g.add_mim_forbidden_cut_rect(0, 1500, 5000, 2000, 5500);
    // Physical via cut obstacle on via layer 0 at (5.0um..5.26um, 2.0um..2.26um).
    g.add_via_obs_rect(0, 5.0, 2.0, 5.26, 2.26, /*owner_id=*/1, /*is_cap_plate_via=*/false);

    const uint32_t all_layers = 0x7U;
    // Net 0 has NO pin inside its own MIM rect -> MIM clearance IS active for net 0.
    g.prepare_net(0, /*route_w_nm=*/280, /*own_width_floor=*/280, all_layers,
                  {g.cell_idx(2, 2, 0)}, {}, /*trunk_gi=*/0, true, true);
    // Envelope blocks border ix=0 (x=0 < 500 + 140) and ix=15 (x=7500 > 7000 - 140).
    CLAY_CHECK(g.is_blocked_cell(0, 0, 5, 0));
    CLAY_CHECK(g.is_blocked_cell(0, 15, 5, 0));
    // Foreign obstacle on layer 0 blocks (5, 5, 0) (x=2500nm, y=2500nm).
    CLAY_CHECK(g.is_blocked_cell(0, 5, 5, 0));
    // MIM clearance on layer 1 blocks (9, 9, 1) (x=4500nm, y=4500nm).
    CLAY_CHECK(g.is_blocked_cell(0, 9, 9, 1));
    // MIM forbidden cut blocks via 0 at (3, 10) (cx=1500, cy=5000) and (4, 11) (cx=2000, cy=5500).
    const auto via_inf = g.get_via_infeasible_cells(0, 0);
    CLAY_CHECK(std::find(via_inf.begin(), via_inf.end(), std::pair<int32_t, int32_t>{3, 10}) != via_inf.end());
    CLAY_CHECK(std::find(via_inf.begin(), via_inf.end(), std::pair<int32_t, int32_t>{10, 4}) != via_inf.end());

    // Now re-prepare net 0 WITH a pin inside its own MIM rect on layer 1 -> MIM exempted!
    std::vector<std::vector<std::pair<int64_t, int64_t>>> pin_xy(3);
    pin_xy[1].emplace_back(4500, 4500);
    g.prepare_net(0, 280, 280, all_layers, {g.cell_idx(9, 9, 1)}, pin_xy, 0, true, true);
    CLAY_CHECK(!g.is_blocked_cell(0, 9, 9, 1));

    // Net 1 owns the layer-0 obstacle -> (5, 5, 0) is NOT blocked for net 1.
    g.prepare_net(1, 280, 280, all_layers, {g.cell_idx(5, 5, 0)}, {}, 0, true, true);
    CLAY_CHECK(!g.is_blocked_cell(1, 5, 5, 0));
    // Net 1 does NOT own the MIM rect -> (9, 9, 1) IS blocked for net 1.
    CLAY_CHECK(g.is_blocked_cell(1, 9, 9, 1));
}

CLAY_TEST(PathFinderCore, UndirectedAStarViaTransitionsTieBreakingAndTrappedPin) {
    PathFinderGridCore g = make_standard_3layer_grid(12, 10, 560);
    // Block an entire vertical column x=5 on layer 0 so net 0 must climb via V0 to layer 1.
    std::vector<int32_t> wall;
    for (int32_t y = 0; y < 10; ++y) {
        wall.push_back(g.cell_idx(5, y, 0));
    }
    g.prepare_net(0, 280, 280, 0x3U, {g.cell_idx(2, 5, 0), g.cell_idx(8, 5, 0)}, {}, 0, true, true);
    g.set_net_blocked_cells(0, wall);

    const int32_t src = g.cell_idx(2, 5, 0);
    const int32_t dst = g.cell_idx(8, 5, 0);
    std::vector<int32_t> path;
    const int rc = g.astar_undirected(
        /*net_id=*/0, /*check_committed=*/false, {src}, dst, /*allowed_mask=*/0x3U,
        /*use_pf_cost=*/true, /*pres_fac=*/0.5, /*hist_fac=*/1.0, /*trunk_gi=*/0,
        /*use_via_cost=*/false, false, 0, false, 0, 0.0, 0,
        false, 0, 0, 0, 0, /*max_expansions=*/100000,
        CheckpointHook{}, kDefaultCheckpointInterval, path
    );
    CLAY_CHECK_EQ(rc, 1);
    CLAY_REQUIRE(!path.empty());
    CLAY_CHECK_EQ(path.front(), src);
    CLAY_CHECK_EQ(path.back(), dst);

    bool climbed_to_l1 = false;
    for (size_t i = 0; i < path.size(); ++i) {
        int32_t ix = 0, iy = 0, gi = 0;
        g.unpack_idx(path[i], ix, iy, gi);
        if (gi == 1) climbed_to_l1 = true;
        if (i > 0 && i + 1 < path.size()) {
            CLAY_CHECK(!g.is_blocked_cell(0, ix, iy, gi));
        }
    }
    CLAY_CHECK(climbed_to_l1);

    // Heap tie-breaking verification: (f, -g, g, ix, iy, gi).
    BinaryMinHeap<HeapNode, HeapNodeLess> h;
    h.push(HeapNode{10.0, -4.0, 4.0, 3, 2, 0, 1});
    h.push(HeapNode{10.0, -6.0, 6.0, 5, 5, 1, 2});  // larger g -> smaller -g -> wins!
    h.push(HeapNode{10.0, -6.0, 6.0, 4, 9, 2, 3});  // same f,-g,g, smaller ix -> wins!
    CLAY_CHECK_EQ(h.pop().idx, 3);
    CLAY_CHECK_EQ(h.pop().idx, 2);
    CLAY_CHECK_EQ(h.pop().idx, 1);

    // Trapped pin pocket BFS: enclose (2, 2, 0) and block via-up at (2, 2, 0).
    std::vector<int32_t> cage;
    for (int32_t dx = -1; dx <= 1; ++dx) {
        for (int32_t dy = -1; dy <= 1; ++dy) {
            cage.push_back(g.cell_idx(2 + dx, 2 + dy, 0));
        }
    }
    g.set_net_blocked_cells(1, cage);
    g.set_net_via_blocked_cells(1, {0 * g.nxy + 2 * g.nx + 2});
    CLAY_CHECK(g.pin_is_trapped(1, 2, 2, 0, 0x3U));
    // Unblocking the via lets the pin escape to layer 1 -> no longer trapped.
    g.set_net_via_blocked_cells(1, {});
    CLAY_CHECK(!g.pin_is_trapped(1, 2, 2, 0, 0x3U));
}

CLAY_TEST(PathFinderCore, DirectedCombAStarWrongDirRunPenaltyAndViaReset) {
    // Layer 0 strictly prefers Horizontal with strong weight; Layer 1 prefers Vertical.
    // Moving vertically by 6 steps on Layer 0 incurs wrong-dir run > kDirRunFree (2),
    // which triggers extra kDirRunPenalty (2.5 * weight per step) and forces climbing to Layer 1!
    PathFinderGridCore g = make_standard_3layer_grid(12, 14, 560);
    const int32_t src = g.cell_idx(5, 1, 0);
    const int32_t dst = g.cell_idx(5, 11, 0);  // delta_y = 10 wrong-dir steps on L0
    g.prepare_net(0, 280, 280, 0x3U, {src, dst}, {}, 0, true, false);

    std::vector<DirPrefEntry> dir_pref(3);
    dir_pref[0] = DirPrefEntry{/*active=*/true, /*horiz=*/true, /*weight=*/3.0};
    dir_pref[1] = DirPrefEntry{/*active=*/true, /*horiz=*/false, /*weight=*/3.0};

    // Compare undirected (L0 wrong-dir penalty only 0.5/step -> 15.0 < 2 vias + 10 = 16.0 -> stays on L0)
    // vs directed comb contract (L0 wrong-dir run > 2 -> huge penalty -> climbs to L1!).
    std::vector<int32_t> path_undir;
    CLAY_CHECK_EQ(
        g.astar_undirected(0, false, {src}, dst, 0x3U, false, 0.0, 0.0, 0, false,
                           false, 0, false, 0, 0.0, 0, false, 0, 0, 0, 0, 100000,
                           CheckpointHook{}, kDefaultCheckpointInterval, path_undir),
        1
    );
    bool undir_used_l1 = false;
    for (int32_t c : path_undir) {
        int32_t ix = 0, iy = 0, gi = 0;
        g.unpack_idx(c, ix, iy, gi);
        if (gi == 1) undir_used_l1 = true;
    }
    CLAY_CHECK(!undir_used_l1);

    std::vector<int32_t> path_dir;
    CLAY_CHECK_EQ(
        g.astar_directed(0, false, {src}, dst, 0x3U, dir_pref, false, 0, false,
                         0.0, 0.0, 0, false, false, 0, false, 0, 0.0, 0,
                         false, 0, 0, 0, 0, 100000,
                         CheckpointHook{}, kDefaultCheckpointInterval, path_dir),
        1
    );
    bool dir_used_l1 = false;
    for (int32_t c : path_dir) {
        int32_t ix = 0, iy = 0, gi = 0;
        g.unpack_idx(c, ix, iy, gi);
        if (gi == 1) dir_used_l1 = true;
    }
    CLAY_CHECK(dir_used_l1);
    CLAY_CHECK_EQ(path_dir.front(), src);
    CLAY_CHECK_EQ(path_dir.back(), dst);

    // Multi-pin route_one_net with dir_pref exercises update_dir_runs tree memory.
    std::vector<int32_t> wire_cells;
    const std::vector<int32_t> pins = {
        g.cell_idx(2, 2, 0), g.cell_idx(9, 2, 0), g.cell_idx(9, 10, 0)
    };
    CLAY_CHECK_EQ(
        g.route_one_net(0, false, true, pins, 0x3U, {}, 0.5, 1.0, 0, false,
                        true, dir_pref, false, {}, 0.0, true, 100000,
                        CheckpointHook{}, kDefaultCheckpointInterval, wire_cells),
        1
    );
    for (int32_t p : pins) {
        CLAY_CHECK(std::find(wire_cells.begin(), wire_cells.end(), p) != wire_cells.end());
    }
}

CLAY_TEST(PathFinderCore, RoundOverusePadContentionAndHistoryUpdates) {
    // Configure via pads larger than wire width so exact pad keepout stencils are non-empty.
    PathFinderGridCore g;
    CLAY_REQUIRE(g.init(
        12, 12, 2,
        0, 0, 500,
        {1, 0},
        {230, 230},
        {230, 230},
        {260},
        {120},  // pad_half = 130 + 120 = 250 > wire_half = 115
        {120},
        {260}
    ));

    const int32_t p0a = g.cell_idx(2, 5, 0);
    const int32_t p0b = g.cell_idx(8, 5, 0);
    const int32_t p1a = g.cell_idx(5, 2, 0);
    const int32_t p1b = g.cell_idx(5, 8, 0);
    g.prepare_net(0, 230, 230, 0x3U, {p0a, p0b}, {}, 0, true, false);
    g.prepare_net(1, 230, 230, 0x3U, {p1a, p1b}, {}, 0, true, false);

    // Round 1: both nets overlap at (5, 5, 0), AND net 0 drops a via at (5, 4, 0)->(5, 4, 1)
    // whose exact pad keepout touches net 1's metal at (5, 3, 0).
    g.begin_round({0, 1});
    CLAY_CHECK_EQ(g.unrouted_pin_count[static_cast<size_t>(p0a)], 1);
    g.mark_net_starting(0);
    CLAY_CHECK_EQ(g.unrouted_pin_count[static_cast<size_t>(p0a)], 0);

    const int32_t shared_cell = g.cell_idx(5, 5, 0);
    const int32_t via_low = g.cell_idx(5, 4, 0);
    const int32_t via_hi = g.cell_idx(5, 4, 1);
    const int32_t adj_cell = g.cell_idx(5, 3, 0);

    g.commit_net_route(0, {p0a, shared_cell, via_low, via_hi, p0b}, {}, true);
    g.mark_net_starting(1);
    g.commit_net_route(1, {p1a, adj_cell, shared_cell, p1b}, {}, true);

    const FinishRoundResult r1 = g.finish_round(/*hist_fac=*/1.5);
    CLAY_CHECK(!r1.converged);
    CLAY_REQUIRE(r1.overused.size() == 1U);
    CLAY_CHECK_EQ(r1.overused[0], shared_cell);
    CLAY_CHECK_NEAR(g.history[static_cast<size_t>(shared_cell)], 1.5, 1e-12);
    // Adjacent cell (5, 3, 0) owned by net 1 is inside net 0's via pad keepout at (5, 4, 0).
    CLAY_CHECK(!r1.pad_contended.empty());
    bool found_adj = false;
    for (const auto& row : r1.pad_contended) {
        if (row.cell_idx == adj_cell) {
            found_adj = true;
            CLAY_CHECK_EQ(row.metal_owner_id, 1);
            CLAY_REQUIRE(row.foreign_owners.size() == 1U);
            CLAY_CHECK_EQ(row.foreign_owners[0], 0);
        }
    }
    CLAY_CHECK(found_adj);
    CLAY_CHECK(g.via_pad_history[static_cast<size_t>(via_low)] > 0.0);

    // Round 2: disjoint routes -> converged == true and history must NOT increase.
    const double prev_hist = g.history[static_cast<size_t>(shared_cell)];
    g.begin_round({0, 1});
    g.commit_net_route(0, {g.cell_idx(1, 1, 0), g.cell_idx(2, 1, 0)}, {}, true);
    g.commit_net_route(1, {g.cell_idx(9, 9, 0), g.cell_idx(10, 9, 0)}, {}, true);
    const FinishRoundResult r2 = g.finish_round(/*hist_fac=*/2.0);
    CLAY_CHECK(r2.converged);
    CLAY_CHECK(r2.overused.empty());
    CLAY_CHECK(r2.pad_contended.empty());
    CLAY_CHECK_NEAR(g.history[static_cast<size_t>(shared_cell)], prev_hist, 1e-12);
}

CLAY_TEST(PathFinderCore, MazeCommittedHaloAndNetFootprint) {
    PathFinderGridCore g = make_standard_3layer_grid(12, 12, 500);
    const int32_t c0 = g.cell_idx(4, 4, 0);
    const int32_t c1 = g.cell_idx(5, 4, 0);
    const int32_t c1_up = g.cell_idx(5, 4, 1);
    g.prepare_net(0, 400, 230, 0x3U, {c0, c1_up}, {}, 0, true, false);

    g.commit_maze_halo(
        /*net_id=*/0, {c0, c1, c1_up},
        /*wire_w_nm=*/400, /*extra_sp_nm=*/0, /*max_other_w_nm=*/230,
        /*use_taper=*/true, /*via_partner_half_nm=*/115,
        /*overwrite_wires=*/true, /*include_via_halo=*/true
    );
    CLAY_CHECK(g.is_committed_foreign(/*net_id=*/1, 4, 4, 0));
    CLAY_CHECK(!g.is_committed_foreign(/*net_id=*/0, 4, 4, 0));
    // Neighbor (4, 5, 0) must be covered by the committed halo of net 0.
    CLAY_CHECK(g.is_committed_foreign(/*net_id=*/1, 4, 5, 0));
    CLAY_CHECK(g.is_blocked_cell(/*net_id=*/1, 4, 5, 0, /*check_committed=*/true));
    CLAY_CHECK(!g.is_blocked_cell(/*net_id=*/1, 4, 5, 0, /*check_committed=*/false));

    const auto frects = g.foreign_committed_rects(/*net_id=*/1);
    CLAY_CHECK(!frects[0].empty());
    CLAY_CHECK(!frects[1].empty());

    const auto fp = g.net_footprint(0, {c0, c1, c1_up}, 400, true);
    CLAY_CHECK(std::find(fp.begin(), fp.end(), c0) != fp.end());
    CLAY_CHECK(std::find(fp.begin(), fp.end(), c1_up) != fp.end());

    g.reset_committed();
    CLAY_CHECK(!g.is_committed_foreign(1, 4, 4, 0));
}

CLAY_TEST(PathFinderCore, MidSearchCancellationViaCheckpointHook) {
    // Large grid with target walled off so A* must run many expansions and poll every 4096.
    PathFinderGridCore g = make_standard_3layer_grid(120, 120, 560);
    const int32_t src = g.cell_idx(5, 5, 0);
    const int32_t dst = g.cell_idx(110, 110, 0);
    g.prepare_net(0, 280, 280, 0x3U, {src, dst}, {}, 0, true, false);

    // Wall off dst completely on layers 0 and 1.
    std::vector<int32_t> wall;
    for (int32_t gi = 0; gi < 2; ++gi) {
        for (int32_t dx = -2; dx <= 2; ++dx) {
            for (int32_t dy = -2; dy <= 2; ++dy) {
                if (std::abs(dx) == 2 || std::abs(dy) == 2) {
                    wall.push_back(g.cell_idx(110 + dx, 110 + dy, gi));
                }
            }
        }
    }
    g.set_net_blocked_cells(0, wall);

    struct StopCounter {
        int calls = 0;
        int stop_on_call = 2;
    } counter;

    auto hook_fn = [](void* raw) -> bool {
        auto* sc = static_cast<StopCounter*>(raw);
        ++sc->calls;
        return sc->calls < sc->stop_on_call;
    };
    const CheckpointHook hook{hook_fn, &counter};

    // 1. Undirected A* aborts on the 2nd 4096-expansion checkpoint (8192 expansions).
    std::vector<int32_t> path;
    const int rc_u = g.astar_undirected(
        0, false, {src}, dst, 0x3U, true, 0.5, 1.0, 0, false,
        false, 0, false, 0, 0.0, 0, false, 0, 0, 0, 0,
        /*max_expansions=*/2000000, hook, kDefaultCheckpointInterval, path
    );
    CLAY_CHECK_EQ(rc_u, -1);
    CLAY_CHECK_EQ(counter.calls, 2);
    CLAY_CHECK(path.empty());

    // 2. Directed A* also aborts cleanly via the exact same CheckpointHook.
    counter.calls = 0;
    counter.stop_on_call = 3;
    std::vector<DirPrefEntry> dir_pref(3);
    dir_pref[0] = DirPrefEntry{true, true, 1.5};
    dir_pref[1] = DirPrefEntry{true, false, 1.5};
    const int rc_d = g.astar_directed(
        0, false, {src}, dst, 0x3U, dir_pref, false, 0, true,
        0.5, 1.0, 0, false, false, 0, false, 0, 0.0, 0,
        false, 0, 0, 0, 0, /*max_expansions=*/2000000, hook, kDefaultCheckpointInterval, path
    );
    CLAY_CHECK_EQ(rc_d, -1);
    CLAY_CHECK_EQ(counter.calls, 3);

    // 3. Grid remains completely reusable after mid-search abort!
    const int32_t near_dst = g.cell_idx(12, 5, 0);
    const int rc_ok = g.astar_undirected(
        0, false, {src}, near_dst, 0x3U, true, 0.5, 1.0, 0, false,
        false, 0, false, 0, 0.0, 0, false, 0, 0, 0, 0,
        /*max_expansions=*/100000, CheckpointHook{}, kDefaultCheckpointInterval, path
    );
    CLAY_CHECK_EQ(rc_ok, 1);
    CLAY_CHECK_EQ(path.front(), src);
    CLAY_CHECK_EQ(path.back(), near_dst);
}

CLAY_TEST(PathFinderCore, RectClipperSubtractAndSpineRetry) {
    RectClipperCore clipper;
    // Negative-coordinate hole subtraction
    clipper.add_layer_holes(34, 0, {
        Box64{-200, -100, 100, 100},
        Box64{250, -50, 400, 50},
    });
    const auto& sub = clipper.subtract_rects(34, 0, -300, -200, 500, 200);
    CLAY_CHECK(!sub.empty());
    // Every returned fragment must be valid and strictly disjoint from all holes.
    for (const auto& r : sub) {
        CLAY_CHECK(r.x0 < r.x1 && r.y0 < r.y1);
        CLAY_CHECK(r.x1 <= -200 || r.x0 >= 100 || r.y1 <= -100 || r.y0 >= 100);
        CLAY_CHECK(r.x1 <= 250 || r.x0 >= 400 || r.y1 <= -50 || r.y0 >= 50);
    }

    // Sub-min-width filter + spine retry on a widened wire clipped along top edge.
    RectClipperCore c2;
    // Horizontal wire [0, 1000] x [0, 600] (full_w=600, min_w=230), hole notches top leaving
    // a 100nm sliver (< 230nm) near y=[500..600] on x=[300..700]? Better: hole at [300, 100, 700, 600]
    // leaves bottom sliver [300, 0, 700, 100] (dropped: 100 < 230), while top is blocked;
    // try hole [300, 450, 700, 600] -> top sliver none, full height under hole is 450 >= 230.
    // Hole [300, 500, 700, 550] leaves a 50nm sliver [300, 550, 700, 600] (< 230 -> dropped!),
    // triggering spine retry along center [0, 185, 1000, 415] (untouched by hole!).
    c2.add_layer_holes(34, 0, {Box64{300, 500, 700, 550}});
    const auto& clipped = c2.clip_emit(34, 0, 0, 0, 1000, 600, 230);
    CLAY_CHECK(!clipped.empty());
    for (const auto& r : clipped) {
        CLAY_CHECK(std::min(r.x1 - r.x0, r.y1 - r.y0) >= 230);
    }
    // Verify center spine [0, 185, 1000, 415] was appended by spine retry!
    CLAY_CHECK(std::find(clipped.begin(), clipped.end(), Box64{0, 185, 1000, 415}) != clipped.end());
}
