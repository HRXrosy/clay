// Pure C++20 native unit tests for Clay's Gridless routing core kernels.
// Strictly free of CPython headers.

#include "test_framework.hpp"

#include "clay_pnr/route/gridless_core.hpp"
#include "clay_pnr/route/gridless_search.hpp"
#include "clay_pnr/route/gridless_spatial.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <unordered_map>
#include <vector>

using namespace clay_pnr::core;
using namespace clay_pnr::route;

namespace {

int64_t brute_2d_potential(const std::vector<Pt2D>& pts, int64_t rate, int64_t x, int64_t y) {
    int64_t best = std::numeric_limits<int64_t>::max();
    for (const auto& p : pts) {
        int64_t dx = x - p.x;
        if (dx < 0) dx = -dx;
        int64_t dy = y - p.y;
        if (dy < 0) dy = -dy;
        const int64_t cand = p.c + (dx + dy) * rate;
        if (cand < best) best = cand;
    }
    return best;
}

uint64_t lcg_next(uint64_t& state) noexcept {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return state;
}

int64_t lcg_range(uint64_t& state, int64_t lo, int64_t hi) noexcept {
    const uint64_t span = static_cast<uint64_t>(hi - lo) + 1ULL;
    return lo + static_cast<int64_t>(lcg_next(state) % span);
}

}  // namespace

CLAY_TEST(GridlessCore, Potential2DSinglePointAndNegativeCoords) {
    Potential2DCore pot;
    pot.init({Pt2D{-350, 120, 4000}}, 51);
    CLAY_CHECK_EQ(pot.kd_root, -1);

    const int64_t xs[] = {-1000, -351, -350, -349, 0, 500};
    const int64_t ys[] = {-500, 119, 120, 121, 900};
    for (int64_t x : xs) {
        for (int64_t y : ys) {
            const int64_t dx = std::llabs(x - (-350));
            const int64_t dy = std::llabs(y - 120);
            CLAY_CHECK_EQ(pot.eval(x, y), 4000 + (dx + dy) * 51);
        }
    }
}

CLAY_TEST(GridlessCore, Potential2DMultiPointParetoAndSmallInterior) {
    // Exercise 1D Pareto elimination of dominated points and small-N linear interior (<= 48 pts).
    std::vector<Pt1D> p1d = {
        {0, 1000}, {0, 500},    // duplicate coordinate, keep 500
        {10, 2000},             // dominated by (0, 500) since 500 + 10*100 = 1500 < 2000
        {20, 800},              // Pareto optimal
        {25, 9000},             // dominated by both sides
        {40, 300},              // Pareto optimal
    };
    const auto reduced = reduce_1d_pareto_cpp(p1d, 100);
    CLAY_CHECK_EQ(reduced.size(), static_cast<size_t>(3));
    CLAY_CHECK_EQ(reduced[0].p, 0);
    CLAY_CHECK_EQ(reduced[0].v, 500);
    CLAY_CHECK_EQ(reduced[1].p, 20);
    CLAY_CHECK_EQ(reduced[1].v, 800);
    CLAY_CHECK_EQ(reduced[2].p, 40);
    CLAY_CHECK_EQ(reduced[2].v, 300);

    std::vector<Pt2D> pts;
    uint64_t rng = 0x2026112301ULL;
    for (int i = 0; i < 24; ++i) {
        pts.push_back(Pt2D{
            lcg_range(rng, -5000, 5000),
            lcg_range(rng, -4000, 4000),
            lcg_range(rng, 0, 200000),
        });
    }
    const int64_t rate = 73;
    Potential2DCore pot;
    pot.init(pts, rate);
    CLAY_CHECK_EQ(pot.kd_root, -1);

    const std::vector<int64_t> probe_xs = {
        -20000, pot.gx0 - 1, pot.gx0, -2500, 0, 2500, pot.gx1, pot.gx1 + 1, 20000
    };
    const std::vector<int64_t> probe_ys = {
        -20000, pot.gy0 - 1, pot.gy0, -2000, 0, 2000, pot.gy1, pot.gy1 + 1, 20000
    };
    for (int64_t x : probe_xs) {
        for (int64_t y : probe_ys) {
            CLAY_CHECK_EQ(pot.eval(x, y), brute_2d_potential(pts, rate, x, y));
        }
    }
}

CLAY_TEST(GridlessCore, Potential2DKDTreeLargeAndNegativeCoords) {
    // > 48 points triggers balanced 2D KD-tree construction & branch-and-bound interior search.
    std::vector<Pt2D> pts;
    uint64_t rng = 0x2026112302ULL;
    for (int i = 0; i < 140; ++i) {
        pts.push_back(Pt2D{
            lcg_range(rng, -15000, 15000),
            lcg_range(rng, -12000, 12000),
            lcg_range(rng, 0, 400000),
        });
    }
    // Include explicit negative duplicate coordinate with smaller cost.
    pts.push_back(Pt2D{-7777, -3333, 500});
    pts.push_back(Pt2D{-7777, -3333, 50});

    const int64_t rate = 125;
    Potential2DCore pot;
    pot.init(pts, rate);
    CLAY_CHECK(pot.kd_root >= 0);
    CLAY_CHECK_EQ(pot.kd_nodes.size(), pts.size());

    // Exact point query on negative duplicate
    CLAY_CHECK_EQ(pot.eval(-7777, -3333), brute_2d_potential(pts, rate, -7777, -3333));

    // Random interior, strip, and exterior queries
    for (int i = 0; i < 300; ++i) {
        const int64_t qx = lcg_range(rng, -22000, 22000);
        const int64_t qy = lcg_range(rng, -18000, 18000);
        CLAY_CHECK_EQ(pot.eval(qx, qy), brute_2d_potential(pts, rate, qx, qy));
    }
}

CLAY_TEST(GridlessCore, LayerHminMultiLayerHopsAndViaCosts) {
    Potential2DCore pot0;
    pot0.init({Pt2D{1000, 1000, 0}}, 50);

    Potential2DCore pot2;
    pot2.init({Pt2D{5000, 5000, 1000}}, 10);

    LayerHminCore hmin;
    hmin.n_gi = 4;
    hmin.via_cost = 40000;
    hmin.max_cache = 8;
    hmin.target_boxes = {Box64{900, 900, 1100, 1100}};
    hmin.specs = {
        LayerHminSpec{0, 50, 0, &pot0},
        LayerHminSpec{2, 10, 80000, &pot2},
    };

    // At (1000, 1000):
    //   layer 0: min(dist=0*50+0, pot0=0) = 0
    //   layer 2: min(dist=0*10+80000, pot2=8000*10+1000=81000) = 80000
    //   layer hops from 0: layer 1 -> 40000, layer 2 -> 80000, layer 3 -> 120000
    CLAY_CHECK_EQ(hmin.eval(1000, 1000, 0), 0);
    CLAY_CHECK_EQ(hmin.eval(1000, 1000, 1), 40000);
    CLAY_CHECK_EQ(hmin.eval(1000, 1000, 2), 80000);
    CLAY_CHECK_EQ(hmin.eval(1000, 1000, 3), 120000);

    // At (5000, 5000):
    //   dist to box [900,1100]^2 is 3900 + 3900 = 7800
    //   layer 0: min(7800*50 = 390000, pot0=8000*50=400000) = 390000
    //   layer 2: min(7800*10+80000 = 158000, pot2=1000) = 1000
    //   backward & forward via sweeps from layer 2:
    //     layer 3 = 1000 + 40000 = 41000
    //     layer 1 = 1000 + 40000 = 41000
    //     layer 0 = min(390000, 41000 + 40000) = 81000
    CLAY_CHECK_EQ(hmin.eval(5000, 5000, 2), 1000);
    CLAY_CHECK_EQ(hmin.eval(5000, 5000, 3), 41000);
    CLAY_CHECK_EQ(hmin.eval(5000, 5000, 1), 41000);
    CLAY_CHECK_EQ(hmin.eval(5000, 5000, 0), 81000);

    // Cached repeat matches identically
    CLAY_CHECK_EQ(hmin.cache.size(), static_cast<size_t>(2));
    CLAY_CHECK_EQ(hmin.eval(5000, 5000, 0), 81000);
    CLAY_CHECK_EQ(hmin.eval(1000, 1000, 3), 120000);
}

CLAY_TEST(GridlessCore, ExactDistanceTransform1D2D3DAndValidation) {
    // 1D exact transform with negative & non-uniform X coords
    {
        const std::vector<int64_t> xs = {-100, -20, 0, 50, 200};
        const std::vector<int64_t> ys = {0};
        const std::vector<int64_t> layers = {0};
        const std::vector<DistanceTransformGoal> goals = {
            {-20, 0, 0, 500},
            {200, 0, 0, 100},
        };
        std::vector<int64_t> pot;
        const auto st = exact_distance_transform_core(xs, ys, layers, goals, 10, 40000, pot);
        CLAY_CHECK_EQ(static_cast<int>(st), static_cast<int>(DistanceTransformStatus::Ok));
        CLAY_CHECK_EQ(pot.size(), xs.size());
        for (size_t ix = 0; ix < xs.size(); ++ix) {
            const int64_t d0 = 500 + std::llabs(xs[ix] - (-20)) * 10;
            const int64_t d1 = 100 + std::llabs(xs[ix] - 200) * 10;
            CLAY_CHECK_EQ(pot[ix], std::min(d0, d1));
        }
    }

    // 2D exact transform
    {
        const std::vector<int64_t> xs = {-50, 10, 40, 120};
        const std::vector<int64_t> ys = {-30, 0, 75};
        const std::vector<int64_t> layers = {1};
        const std::vector<DistanceTransformGoal> goals = {
            {-50, 75, 1, 200},
            {120, -30, 1, 50},
        };
        std::vector<int64_t> pot;
        const auto st = exact_distance_transform_core(xs, ys, layers, goals, 5, 40000, pot);
        CLAY_CHECK_EQ(static_cast<int>(st), static_cast<int>(DistanceTransformStatus::Ok));
        CLAY_CHECK_EQ(pot.size(), xs.size() * ys.size());
        for (size_t iy = 0; iy < ys.size(); ++iy) {
            for (size_t ix = 0; ix < xs.size(); ++ix) {
                int64_t expected = std::numeric_limits<int64_t>::max();
                for (const auto& g : goals) {
                    const int64_t cand = g.cost + (std::llabs(xs[ix] - g.x) + std::llabs(ys[iy] - g.y)) * 5;
                    if (cand < expected) expected = cand;
                }
                CLAY_CHECK_EQ(pot[iy * xs.size() + ix], expected);
            }
        }
    }

    // 3D exact transform with non-contiguous layers [0, 1, 2, 4]
    {
        const std::vector<int64_t> xs = {-500, -100, 0, 250, 900};
        const std::vector<int64_t> ys = {-300, 50, 400, 1000};
        const std::vector<int64_t> layers = {0, 1, 2, 4};
        const std::vector<DistanceTransformGoal> goals = {
            {-500, -300, 0, 1200},
            {250, 400, 2, 300},
            {900, 1000, 4, 0},
            {250, 400, 2, 100},  // duplicate goal coordinate with lower cost
        };
        const int64_t step_cost = 51;
        const int64_t via_cost = 40000;
        std::vector<int64_t> pot;
        const auto st = exact_distance_transform_core(xs, ys, layers, goals, step_cost, via_cost, pot);
        CLAY_CHECK_EQ(static_cast<int>(st), static_cast<int>(DistanceTransformStatus::Ok));
        const size_t nx = xs.size(), ny = ys.size(), nl = layers.size();
        const size_t plane = nx * ny;
        CLAY_CHECK_EQ(pot.size(), plane * nl);
        for (size_t il = 0; il < nl; ++il) {
            for (size_t iy = 0; iy < ny; ++iy) {
                for (size_t ix = 0; ix < nx; ++ix) {
                    int64_t expected = std::numeric_limits<int64_t>::max();
                    for (const auto& g : goals) {
                        const int64_t cand = g.cost
                            + (std::llabs(xs[ix] - g.x) + std::llabs(ys[iy] - g.y)) * step_cost
                            + std::llabs(layers[il] - g.layer) * via_cost;
                        if (cand < expected) expected = cand;
                    }
                    CLAY_CHECK_EQ(pot[il * plane + iy * nx + ix], expected);
                }
            }
        }
    }

    // Error handling: non-monotonic axis, missing goal, empty, overflow
    {
        std::vector<int64_t> pot;
        CLAY_CHECK_EQ(
            static_cast<int>(exact_distance_transform_core({10, 10}, {0}, {0}, {{10, 0, 0, 0}}, 1, 100, pot)),
            static_cast<int>(DistanceTransformStatus::NonMonotonicAxis));
        CLAY_CHECK_EQ(
            static_cast<int>(exact_distance_transform_core({0, 10}, {0}, {0}, {{5, 0, 0, 0}}, 1, 100, pot)),
            static_cast<int>(DistanceTransformStatus::GoalNotFound));
        CLAY_CHECK_EQ(
            static_cast<int>(exact_distance_transform_core({0, 10}, {0}, {0}, {}, 1, 100, pot)),
            static_cast<int>(DistanceTransformStatus::Empty));
        CLAY_CHECK_EQ(
            static_cast<int>(exact_distance_transform_core(
                {0, 1000000000LL}, {0}, {0}, {{0, 0, 0, 0}}, 5000000000LL, 100, pot)),
            static_cast<int>(DistanceTransformStatus::Overflow));
    }
}

CLAY_TEST(GridlessCore, HistoryGridIncrementQuerySaturationDecayAndSparse) {
    HistoryGridCore hg;
    const int64_t rc = 1000;

    // 1. Increment with negative coords, repeated accumulation, saturation cap, and extra layer >= 64
    hg.increment(0, -2, -1, 500, 1200);
    hg.increment(0, -2, -1, 500, 1200);
    hg.increment(0, -2, -1, 500, 1200);  // saturates at 1200!
    hg.increment(0, 0, 0, 300, 1200);
    hg.increment(0, 2, 1, 800, 1200);
    hg.increment(70, 1, 1, 450, 1000);   // extra_layers path (rgi >= FAST_LAYER_LIMIT)

    // Query before compile (sparse path)
    const int64_t q_all_sparse = hg.eval_units(0, Box64{-3 * rc, -2 * rc, 4 * rc, 3 * rc}, rc);
    CLAY_CHECK_EQ(q_all_sparse, 1200 + 300 + 800);
    CLAY_CHECK_EQ(hg.eval_units(0, Box64{-2 * rc, -1 * rc, -1 * rc, 0}, rc), 1200);
    CLAY_CHECK_EQ(hg.eval_units(0, Box64{0, 0, 0, 0}, rc), 0);  // degenerate box
    CLAY_CHECK_EQ(hg.eval_units(70, Box64{rc, rc, 2 * rc, 2 * rc}, rc), 450);

    // 2. Compile to O(1) dense 2D SAT and verify bit-identical queries
    hg.compile();
    CLAY_CHECK(hg.get_layer(0) != nullptr && hg.get_layer(0)->dense);
    CLAY_CHECK(hg.get_layer(70) != nullptr && hg.get_layer(70)->dense);
    CLAY_CHECK_EQ(hg.eval_units(0, Box64{-3 * rc, -2 * rc, 4 * rc, 3 * rc}, rc), 2300);
    CLAY_CHECK_EQ(hg.eval_units(0, Box64{-2 * rc, -1 * rc, -1 * rc, 0}, rc), 1200);
    CLAY_CHECK_EQ(hg.eval_units(0, Box64{0, 0, rc, rc}, rc), 300);
    CLAY_CHECK_EQ(hg.eval_units(0, Box64{50 * rc, 50 * rc, 60 * rc, 60 * rc}, rc), 0);

    // 3. eval_pending_lb exact floor scaling
    bool ovf = false;
    const int64_t lb = hg.eval_pending_lb(0, 0, 0, 2500, 0, 500, rc, 2000, 1000, ovf);
    CLAY_CHECK(!ovf);
    // Box [-500, 3000] x [-500, 500] covers cells cx in [-1..2], cy in [-1..0]: contains (0,0)=300
    CLAY_CHECK_EQ(lb, 300 * 2000 / 1000);

    // 4. Decay (0.5x with round-half-to-even) preserves dense SAT coherence
    hg.decay(0.5);
    CLAY_CHECK(hg.get_layer(0) != nullptr && hg.get_layer(0)->dense);
    CLAY_CHECK_EQ(hg.eval_units(0, Box64{-2 * rc, -1 * rc, -1 * rc, 0}, rc), 600);
    CLAY_CHECK_EQ(hg.eval_units(0, Box64{0, 0, rc, rc}, rc), 150);
    CLAY_CHECK_EQ(hg.eval_units(0, Box64{2 * rc, 1 * rc, 3 * rc, 2 * rc}, rc), 400);
    CLAY_CHECK_EQ(hg.eval_units(70, Box64{rc, rc, 2 * rc, 2 * rc}, rc), 225);

    // 5. Increment on a dense layer unpacks seamlessly, saturates, and re-compiles
    hg.increment(0, 0, 0, 1000, 1000);
    CLAY_CHECK_EQ(hg.eval_units(0, Box64{0, 0, rc, rc}, rc), 1000);
    hg.compile();
    CLAY_CHECK(hg.get_layer(0)->dense);
    CLAY_CHECK_EQ(hg.eval_units(0, Box64{-3 * rc, -2 * rc, 4 * rc, 3 * rc}, rc), 600 + 1000 + 400);

    // 6. Decay to zero clears everything
    hg.decay(0.0);
    CLAY_CHECK(!hg.has_any);
    CLAY_CHECK_EQ(hg.eval_units(0, Box64{-3 * rc, -2 * rc, 4 * rc, 3 * rc}, rc), 0);

    // 7. Sparse fallback (> 4096 cell span) via build()
    std::unordered_map<int64_t, std::vector<HistoryRawCell>> sparse_in;
    sparse_in[0] = {
        HistoryRawCell{0, 0, 100},
        HistoryRawCell{10000, 10000, 250},
        HistoryRawCell{-5000, 200, 350},
    };
    hg.build(sparse_in);
    CLAY_CHECK(hg.get_layer(0) != nullptr && !hg.get_layer(0)->dense);
    CLAY_CHECK_EQ(hg.eval_units(0, Box64{-10000 * rc, -10000 * rc, 20000 * rc, 20000 * rc}, rc), 700);
    CLAY_CHECK_EQ(hg.eval_units(0, Box64{9999 * rc, 9999 * rc, 10002 * rc, 10002 * rc}, rc), 250);
}

CLAY_TEST(GridlessCore, ThroatEventsAndRegionEventGeometry) {
    CLAY_CHECK_EQ(ceil_sqrt_py_exact(-5), 1);
    CLAY_CHECK_EQ(ceil_sqrt_py_exact(0), 0);
    CLAY_CHECK_EQ(ceil_sqrt_py_exact(25), 5);
    CLAY_CHECK_EQ(ceil_sqrt_py_exact(26), 6);

    const auto te = compute_throat_events_core({100, 300}, {50, 250}, 0, 0, 600, 400, 150, 200, 5);
    CLAY_CHECK(!te.xs.empty());
    CLAY_CHECK(!te.ys.empty());
    CLAY_CHECK(std::is_sorted(te.xs.begin(), te.xs.end()));
    CLAY_CHECK(std::is_sorted(te.ys.begin(), te.ys.end()));
    CLAY_CHECK(std::is_sorted(te.thresh.begin(), te.thresh.end()));
    for (int64_t x : te.xs) CLAY_CHECK_EQ(x % 5, 0);
    for (int64_t y : te.ys) CLAY_CHECK_EQ(y % 5, 0);

    // Region portal & target path projection
    SmallPath paths[2];
    int np = 0;
    const auto perr = compute_portal_paths_core(
        Box64{0, 0, 10, 10}, Box64{10, 2, 20, 12}, true, 1, 10, 2, 10, 3, 5, paths, np);
    CLAY_CHECK_EQ(static_cast<int>(perr), static_cast<int>(RegionGeomError::Ok));
    CLAY_CHECK_EQ(np, 1);
    CLAY_CHECK_EQ(paths[0].len, 3);
    CLAY_CHECK_EQ(paths[0].pts[2].first, 10);
    CLAY_CHECK_EQ(paths[0].pts[2].second, 5);

    const auto terr = compute_target_paths_core(Box64{0, 0, 10, 10}, Box64{5, 5, 12, 12}, 2, 3, paths, np);
    CLAY_CHECK_EQ(static_cast<int>(terr), static_cast<int>(RegionGeomError::Ok));
    CLAY_CHECK_EQ(np, 2);

    int64_t ox = 0, oy = 0;
    CLAY_CHECK(settled_axis_prefix_core(50, 0, 650, 0, {0, 100, 300, 700}, {0}, ox, oy));
    CLAY_CHECK_EQ(ox, 300);
    CLAY_CHECK_EQ(oy, 0);
}

CLAY_TEST(GridlessCore, PureSpatialIndexAndEdgeContextPreflightAStar) {
    // Pure C++20 end-to-end Gridless routing test with ZERO <Python.h>:
    // Two-layer (M1=0, M2=1) obstacle avoidance + via detour + AxisPriceCache.
    PureSpatialIndex obs_m1(2000);
    PureSpatialIndex obs_m2(2000);
    // Foreign vertical wall on M1 blocking direct horizontal path at x=1000.
    obs_m1.add_entry(Box64{900, -500, 1100, 2500}, 2u, 101ULL, 2u);
    CLAY_CHECK(obs_m1.has_foreign_intersecting(Box64{850, 0, 950, 100}, 1u));
    CLAY_CHECK(!obs_m1.has_foreign_intersecting(Box64{850, 0, 950, 100}, 2u));

    PureEdgeContext ctx;
    ctx.net_tag = 1u;
    ctx.n_layers = 2;
    ctx.grid = 5;
    ctx.rc = 200;
    ctx.pres = 100;
    ctx.mul_scale = 1000;
    ctx.cost_factor_scale = 1000;
    ctx.turn_cost = 500;
    ctx.via_cost = 40000;
    ctx.step_cost = 100;
    ctx.off_pref_cost = 50;
    ctx.has_env = true;
    ctx.env_box = Box64{-2000, -2000, 4000, 4000};

    ctx.layers.resize(2);
    for (int gi = 0; gi < 2; ++gi) {
        auto& L = ctx.layers[static_cast<size_t>(gi)];
        L.hw = 100;
        L.min_hw = 100;
        L.space = 200;
        L.base_rate = 100;
        L.pref_h = (gi == 0);
        L.obs_idx = (gi == 0) ? &obs_m1 : &obs_m2;
        L.net_obs_id = 1u;
    }
    ctx.vias.resize(1);
    ctx.vias[0].bh0 = 150;
    ctx.vias[0].ah0 = 150;
    ctx.vias[0].ext0 = 100;
    ctx.vias[0].vsp = 200;

    // Direct M1 segment across wall x=0 -> x=2000 must be rejected as REJ_FIXED_METAL.
    int64_t pres = 0;
    CLAY_CHECK_EQ(ctx.check_wire_legal(0, 0, 1000, 2000, 1000, pres), 0);
    CLAY_CHECK(ctx.pending_rej_count >= 1);
    // M2 bridge across same coordinates is legal.
    CLAY_CHECK_EQ(ctx.check_wire_legal(1, 0, 1000, 2000, 1000, pres), 1);

    // Run pure C++20 Hard Connectivity Preflight + Cartesian A* hopping M1 -> M2 -> M1.
    const std::vector<int64_t> xs = {0, 500, 1500, 2000};
    const std::vector<int64_t> ys = {1000};
    const std::vector<int32_t> layers = {0, 1};
    // node_id = lp * (nx * ny) + yp * nx + xp: start=(0,0,0)->0, goal=(3,0,0)->3
    const auto pre = run_hard_connectivity_preflight_pure(ctx, xs, ys, layers, {0}, {3}, 10000, true);
    CLAY_CHECK_EQ(pre.status, 1);

    const std::vector<AStarEndpointSpec> starts = {AStarEndpointSpec{0, 0, 0, false, 0, 0, -1, 0, 0}};
    const std::vector<AStarEndpointSpec> goals = {AStarEndpointSpec{3, 0, 0, false, 0, 0, -1, 0, 0}};
    const auto astar = run_cartesian_astar_pure(
        ctx, xs, ys, layers, starts, goals,
        5000, 0, 50000, 10000, true, true, false, 0, false);
    CLAY_CHECK_EQ(astar.status, 0);
    CLAY_CHECK(astar.path.size() >= 4u);
    CLAY_CHECK_EQ(astar.path.front().x, 0);
    CLAY_CHECK_EQ(astar.path.front().gi, 0);
    CLAY_CHECK_EQ(astar.path.back().x, 2000);
    CLAY_CHECK_EQ(astar.path.back().gi, 0);
    bool visited_m2 = false;
    for (const auto& pn : astar.path) {
        if (pn.gi == 1) visited_m2 = true;
    }
    CLAY_CHECK(visited_m2);

    // Pure C++20 AxisPriceCache check on M2.
    PureAxisPriceCache acache;
    acache.xs = xs;
    acache.ys = ys;
    int64_t steps_ch = 0, sp_steps = 0, pc_hits = 0, out_c = 0;
    bool bstop = false;
    CLAY_CHECK_EQ(run_axis_price_ctx_pure(acache, ctx, 1, 0, 1000, 2000, 1000,
                                          100, steps_ch, sp_steps, pc_hits, 1000, out_c, bstop),
                  1);
    CLAY_CHECK(!bstop);
    CLAY_CHECK(out_c > 0);
    int64_t out_c2 = 0;
    CLAY_CHECK_EQ(run_axis_price_ctx_pure(acache, ctx, 1, 0, 1000, 2000, 1000,
                                          100, steps_ch, sp_steps, pc_hits, 1000, out_c2, bstop),
                  1);
    CLAY_CHECK_EQ(out_c, out_c2);
    CLAY_CHECK_EQ(pc_hits, 1);
}

CLAY_TEST(GridlessCore, PureViaTargetProjectorAndAdvanceRegionStep) {
    // Pure C++20 Phase 1B test: ViaTargetProjector projection/cache/heap/saturation
    // + advance_region_step_pure without any Python dependency.
    Potential2DCore pot0, pot1;
    pot0.init({Pt2D{1000, 1000, 0}}, 100);
    pot1.init({Pt2D{1000, 1000, 0}}, 100);
    LayerHminCore hmin;
    hmin.n_gi = 2;
    hmin.via_cost = 40000;
    hmin.max_cache = 64;
    hmin.specs = {LayerHminSpec{0, 100, 0, &pot0}, LayerHminSpec{1, 100, 0, &pot1}};

    PureSpatialIndex other_regions(100);
    other_regions.add_entry(Box64{5, 4, 15, 12}, 7u, 7ULL, 7u);

    PureViaTargetProjector proj;
    proj.hmin_core = &hmin;
    proj.grid = 100;
    proj.turn_cost = 500;
    proj.via_cost = 40000;
    proj.rect_bucket_nm = 200;
    proj.neg_bucket_dedup = true;
    proj.via_succ_dir_1 = 1;
    proj.via_succ_dir_2 = 2;
    proj.goal_by_layer[0].emplace(1000, 1000);

    struct SpatialLookupCtx {
        PureViaTargetProjector* proj;
        const PureSpatialIndex* idx;
    } lctx{&proj, &other_regions};
    auto lookup_fn = [](void* raw, int64_t ex, int64_t ey) -> const std::vector<int32_t>* {
        auto* c = static_cast<SpatialLookupCtx*>(raw);
        return c->proj->lookup_owner_ids_spatial(c->idx, ex, ey);
    };

    const std::vector<Box64> clipped = {Box64{6, 5, 12, 9}, Box64{2, 2, 4, 4}};
    bool u_used = false;
    const int mask = proj.enqueue_via_targets_pure(
        0, 1, 3, 2, 2, 0, 0, 20, 20, &clipped,
        true, lookup_fn, &lctx,
        0u, 1000, 1, 100, 1000,
        false, 0, false, 0,
        2, false, u_used);
    CLAY_CHECK(mask >= 0);
    CLAY_CHECK((mask & 1) == 1);  // (2,2) lies inside Box64{2,2,4,4} -> has_cross=true
    CLAY_CHECK(u_used);
    CLAY_CHECK(proj.has_cached_target(0, 1, 3, 2, 2, true));

    int64_t top_f = 0;
    CLAY_CHECK(proj.frontier_top_f(top_f));
    CLAY_CHECK(top_f > 0);

    RegionFrontierEntry popped{};
    CLAY_CHECK(proj.pop_frontier_raw(popped));
    CLAY_CHECK_EQ(popped.r_dest, -1 - 2);
    bool sat_pruned = false;
    CLAY_CHECK(proj.check_frontier_entry_open(popped, 1, 2, sat_pruned));
    CLAY_CHECK(!sat_pruned);

    // Saturate owner 7 at that bucket and verify pruning triggers.
    const auto& rec = popped.cache_entry->recs[popped.rec_idx];
    proj.inc_rect_entry(1, 7, rec.other_ulast, rec.ebx, rec.eby);
    proj.inc_rect_entry(1, 7, rec.other_ulast, rec.ebx, rec.eby);
    CLAY_CHECK(!proj.check_frontier_entry_open(popped, 1, 2, sat_pruned));
    CLAY_CHECK(sat_pruned);

    // Pure C++20 advance_region_step_pure test (legal horizontal move + settled prefix + turn).
    PureEdgeContext ctx;
    ctx.n_layers = 1;
    ctx.grid = 100;
    ctx.rc = 200;
    ctx.turn_cost = 500;
    ctx.layers.resize(1);
    ctx.layers[0].hw = 50;
    ctx.layers[0].min_hw = 50;
    ctx.layers[0].space = 100;
    ctx.layers[0].base_rate = 100;
    ctx.layers[0].pref_h = true;

    PureAxisPriceCache acache;
    acache.xs = {0, 200, 500, 1000};
    acache.ys = {0, 200, 500, 1000};

    RegionStepState u{};
    u.ux = 0; u.uy = 0; u.ul = 0; u.ulast = 2; u.rect = 1;
    u.run_x = 0; u.run_y = 0; u.run_dir = 0;
    int64_t sp_steps = 0, pc_hits = 0;
    const auto step = advance_region_step_pure(
        ctx, acache, u, 6, 0, 0, 9, 100,
        0, 1000, 0, 1000, 1000,
        sp_steps, pc_hits, 256);
    CLAY_CHECK_EQ(step.status, 1);
    CLAY_CHECK_EQ(step.vx, 600);
    CLAY_CHECK_EQ(step.vy, 0);
    CLAY_CHECK_EQ(step.vdir, 1);
    CLAY_CHECK_EQ(step.out_rect, 9);
    CLAY_CHECK_EQ(step.start_x, 500);  // settled prefix snapped at axis x=500
    CLAY_CHECK_EQ(step.signed_dir, 1);
    CLAY_CHECK(step.step_cost >= 500);  // includes turn_cost=500 + static cost
    CLAY_CHECK(!step.budget_stopped);
}

CLAY_TEST(GridlessCore, PureSpatialScansViaFilterAndFastLegality) {
    // Pure C++20 verification of all lifted PureSpatialIndex scans, batch_filter_via_sites_pure,
    // batch_via_congestion_pure, check_wire_legal_fast_pure, check_via_legal_fast_pure, and box_congestion_fast_pure.
    PureSpatialIndex obs(1000);
    obs.add_entry(Box64{0, 0, 200, 200}, 1u, 1ULL, 101ULL);     // own net
    obs.add_entry(Box64{500, 0, 700, 200}, 2u, 2ULL, 202ULL);   // foreign net

    const auto obs_hits = obs.scan_obstacle_conflicts_pure(Box64{350, 0, 450, 200}, 100, 1u);
    CLAY_CHECK_EQ(obs_hits.size(), 1u);
    const auto mc_hits = obs.scan_metal_conflicts_pure(Box64{350, 0, 450, 200}, 100, 1u, UINT32_MAX);
    CLAY_CHECK_EQ(mc_hits.size(), 1u);
    CLAY_CHECK(!mc_hits[0].is_touch);
    const auto cm_res = obs.scan_committed_metal_pure(Box64{450, 0, 550, 200}, 100, 100, 50, 1u);
    CLAY_CHECK_EQ(cm_res.hits.size(), 1u);
    CLAY_CHECK(cm_res.hits[0].is_touch);
    CLAY_CHECK(cm_res.present_total > 0);
    CLAY_CHECK_EQ(obs.scan_same_net_band_count_pure(Box64{250, 0, 350, 200}, 100, 1u, true), 1);
    CLAY_CHECK_EQ(obs.scan_same_net_band_rects_pure(Box64{250, 0, 350, 200}, 100, 1u).size(), 1u);

    PureSpatialIndex cuts(1000);
    cuts.add_entry(Box64{100, 100, 200, 200}, 2u, 10ULL, 202ULL);
    const auto cc = cuts.scan_cut_conflicts_pure(Box64{250, 100, 350, 200}, 100, 1u);
    CLAY_CHECK_EQ(cc.spacing_violations.size(), 1u);
    const auto fc = cuts.scan_fixed_cut_conflicts_pure(Box64{250, 100, 350, 200}, 100);
    CLAY_CHECK_EQ(fc.size(), 1u);

    // Batch via site filter: site (600, 100) conflicts with foreign obs; site (1000, 100) is clean.
    const std::vector<std::pair<int64_t, int64_t>> sites = {{600, 100}, {1000, 100}};
    ViaSiteFilterRequest vreq{};
    vreq.use_explicit_sites = true;
    vreq.explicit_sites = &sites;
    vreq.lo = 0;
    vreq.bh = 50;
    vreq.ah = 50;
    vreq.ext = 30;
    vreq.sp_lo = 100;
    vreq.sp_hi = 100;
    vreq.vsp = 100;
    vreq.grid = 5;
    vreq.net_owner_tag = 101ULL;
    vreq.obs_lo = &obs;
    vreq.net_obs_lo = 1u;
    const auto vres = batch_filter_via_sites_pure(vreq);
    CLAY_CHECK_EQ(vres.candidate_count, 2);
    CLAY_CHECK_EQ(vres.legal_sites.size(), 1u);
    CLAY_CHECK_EQ(vres.legal_sites[0].first, 1000);
    CLAY_CHECK(!vres.agg_rejs.empty());

    // Fast wire legality + box congestion + batch via congestion
    WireLegalFastRequest wreq{};
    wreq.wb = Box64{900, 0, 1100, 200};
    wreq.sp = 100;
    wreq.rc = 100;
    wreq.pres = 50;
    wreq.obs_idx = &obs;
    wreq.own_obs = 1u;
    int64_t pres_tot = 0;
    CLAY_CHECK(check_wire_legal_fast_pure(wreq, pres_tot));
    wreq.wb = Box64{450, 0, 550, 200};
    CLAY_CHECK(!check_wire_legal_fast_pure(wreq, pres_tot));

    ViaLegalFastRequest vlreq{};
    vlreq.x = 1000; vlreq.y = 100;
    vlreq.bh = 50; vlreq.ah = 50; vlreq.ext = 30;
    vlreq.sp_lo = 100; vlreq.sp_hi = 100; vlreq.vsp = 100;
    vlreq.grid = 5; vlreq.rc = 100; vlreq.pres = 50;
    vlreq.obs_lo = &obs; vlreq.own_obs_lo = 1u;
    int64_t plo = 0, phi = 0, pcut = 0;
    CLAY_CHECK_EQ(check_via_legal_fast_pure(vlreq, plo, phi, pcut), 1);

    BatchViaCongestionRequest bcreq{};
    bcreq.sites = &sites;
    bcreq.bh = 50; bcreq.ah = 50; bcreq.ext = 30;
    bcreq.sp_lo = 100; bcreq.sp_hi = 100; bcreq.vsp = 100;
    bcreq.rc = 100; bcreq.pres = 50; bcreq.mul_scale = 1000; bcreq.cost_scale = 1000;
    bcreq.metal_lo = &obs; bcreq.net_metal_lo = 1u;
    const auto bcong = batch_via_congestion_pure(bcreq);
    CLAY_CHECK_EQ(bcong.size(), 2u);
    CLAY_CHECK(bcong[0] > 0);
    CLAY_CHECK_EQ(bcong[1], 0);
}
