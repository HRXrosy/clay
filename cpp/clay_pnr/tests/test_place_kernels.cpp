// Pure C++20 native unit tests for Clay placement kernels (sweep_union, bvh_index, seqpair).
// Strictly free of CPython headers.

#include "test_framework.hpp"

#include "clay_pnr/core/geometry.hpp"
#include "clay_pnr/core/status.hpp"
#include "clay_pnr/place/bvh_index.hpp"
#include "clay_pnr/place/seqpair.hpp"
#include "clay_pnr/place/sweep_union.hpp"

#include <algorithm>
#include <cstdint>
#include <set>
#include <utility>
#include <vector>

using namespace clay_pnr::core;
using namespace clay_pnr::place;

namespace {

std::set<std::pair<int64_t, int64_t>> raster_grid_points(const std::vector<Box64>& boxes) {
    std::set<std::pair<int64_t, int64_t>> pts;
    for (const Box64& b : boxes) {
        for (int64_t x = b.x0; x < b.x1; ++x) {
            for (int64_t y = b.y0; y < b.y1; ++y) {
                pts.emplace(x, y);
            }
        }
    }
    return pts;
}

uint64_t total_area_small(const std::vector<Box64>& boxes) {
    uint64_t sum = 0;
    for (const Box64& b : boxes) {
        sum += static_cast<uint64_t>(b.x1 - b.x0) * static_cast<uint64_t>(b.y1 - b.y0);
    }
    return sum;
}

bool pairwise_disjoint(const std::vector<Box64>& boxes) {
    for (size_t i = 0; i < boxes.size(); ++i) {
        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (box_clearance_linf(boxes[i], boxes[j]) < 0) {
                return false;
            }
        }
    }
    return true;
}

}  // namespace

// ============================================================================
// 1. sweep_union tests
// ============================================================================

CLAY_TEST(PlaceSweepUnion, EmptySingleAndDisjointBoxes) {
    std::vector<Box64> out;
    CLAY_CHECK(rectangle_union_cover({}, out));
    CLAY_CHECK(out.empty());

    const std::vector<Box64> single = {Box64{-10, 5, 20, 35}};
    CLAY_CHECK(rectangle_union_cover(single, out));
    CLAY_CHECK_EQ(out.size(), 1u);
    CLAY_CHECK(out[0] == single[0]);

    // Disjoint boxes including negative and large 64-bit coordinates.
    const int64_t big = (1LL << 50);
    const std::vector<Box64> disjoint = {
        Box64{big + 20, -big, big + 30, -big + 10},
        Box64{-100, -50, -80, -30},
        Box64{0, 0, 10, 10},
        Box64{10, 10, 20, 20},  // corner-touching only
    };
    CLAY_CHECK(rectangle_union_cover(disjoint, out));
    CLAY_CHECK_EQ(out.size(), 4u);
    CLAY_CHECK(pairwise_disjoint(out));
    std::vector<Box64> expected = disjoint;
    std::sort(expected.begin(), expected.end());
    CLAY_CHECK(out == expected);
}

CLAY_TEST(PlaceSweepUnion, OverlappingRingWithHoleAndDuplicates) {
    // Ring with interior hole + touching ear + duplicate boxes.
    const std::vector<Box64> ring = {
        Box64{-3, -3, 3, -2},
        Box64{-3, 2, 3, 3},
        Box64{-3, -2, -2, 2},
        Box64{2, -2, 3, 2},
        Box64{3, -1, 5, 1},
        Box64{-3, -3, 3, -2},  // duplicate
    };
    std::vector<Box64> out;
    CLAY_CHECK(rectangle_union_cover(ring, out));
    CLAY_CHECK(pairwise_disjoint(out));
    CLAY_CHECK(std::is_sorted(out.begin(), out.end()));
    const auto pts_in = raster_grid_points(ring);
    const auto pts_out = raster_grid_points(out);
    CLAY_CHECK(pts_in == pts_out);
    CLAY_CHECK_EQ(total_area_small(out), static_cast<uint64_t>(pts_in.size()));

    // Overlapping horizontal/vertical bars merging cleanly into smaller partition.
    const std::vector<Box64> bars = {
        Box64{0, 0, 10, 4},
        Box64{5, 0, 15, 4},
        Box64{15, 0, 20, 4},
    };
    CLAY_CHECK(rectangle_union_cover(bars, out));
    CLAY_CHECK_EQ(out.size(), 1u);
    CLAY_CHECK(out[0] == (Box64{0, 0, 20, 4}));
}

CLAY_TEST(PlaceSweepUnion, ExactForbiddenUnionContracts) {
    // Dense overlapping grid with a hole shrinks strictly while preserving exact coverage.
    std::vector<Box64> dense;
    for (int64_t x = 0; x < 12; ++x) {
        for (int64_t y = 0; y < 10; ++y) {
            if (x > 3 && x < 8 && y > 2 && y < 7) continue;
            dense.push_back(Box64{2 * x, 2 * y, 2 * x + 5, 2 * y + 5});
        }
    }
    std::vector<Box64> out;
    CLAY_CHECK(exact_forbidden_union(dense, out) == ExactUnionStatus::Ok);
    CLAY_CHECK(out.size() < dense.size());
    CLAY_CHECK(pairwise_disjoint(out));
    const auto pts_in = raster_grid_points(dense);
    const auto pts_out = raster_grid_points(out);
    CLAY_CHECK(pts_in == pts_out);
    CLAY_CHECK_EQ(total_area_small(out), static_cast<uint64_t>(pts_in.size()));

    // Crossing pair that does not shrink (2 -> 3 pieces) falls back to sorted unique inputs.
    const std::vector<Box64> crossing = {Box64{-1, -2, 0, 1}, Box64{-2, -1, 1, 0}};
    CLAY_CHECK(exact_forbidden_union(crossing, out) == ExactUnionStatus::Ok);
    CLAY_CHECK_EQ(out.size(), 2u);
    CLAY_CHECK(out[0] == (Box64{-2, -1, 1, 0}));
    CLAY_CHECK(out[1] == (Box64{-1, -2, 0, 1}));

    // Span exceeding kI32Max preserves canonical deduplicated boxes.
    const std::vector<Box64> wide_span = {
        Box64{-2000000000LL, -1, 1, 2},
        Box64{0, -1, 2000000000LL, 2},
    };
    CLAY_CHECK(exact_forbidden_union(wide_span, out) == ExactUnionStatus::Ok);
    CLAY_CHECK(out == wide_span);
}

CLAY_TEST(PlaceSweepUnion, ForbiddenBoxCoverLayeredAndWitnessPruning) {
    // Disjoint layers produce no forbidden boxes unless wildcard (layer 0) is used.
    const std::vector<AtomRec> left_disjoint = {AtomRec{1, Box64{0, 0, 4, 4}}};
    const std::vector<AtomRec> right_disjoint = {AtomRec{2, Box64{0, 0, 4, 4}}};
    const auto r_empty = forbidden_box_cover(left_disjoint, right_disjoint, 2);
    CLAY_CHECK(r_empty.status == ForbiddenStatus::Ok);
    CLAY_CHECK(r_empty.cover.empty());
    CLAY_CHECK_EQ(r_empty.raw_count, 0ULL);
    CLAY_CHECK_EQ(r_empty.unique_count, 0ULL);

    // Wildcard left matches all right layers; exact Minkowski difference for required_nm=1.
    const std::vector<AtomRec> left_wc = {AtomRec{0, Box64{0, 0, 2, 2}}};
    const std::vector<AtomRec> right_wc = {
        AtomRec{1, Box64{0, 0, 2, 2}},
        AtomRec{2, Box64{1, 0, 3, 2}},
    };
    const auto r_wc = forbidden_box_cover(left_wc, right_wc, 1);
    CLAY_CHECK(r_wc.status == ForbiddenStatus::Ok);
    CLAY_CHECK_EQ(r_wc.raw_count, 2ULL);
    CLAY_CHECK_EQ(r_wc.unique_count, 2ULL);
    // Box 1: {-2, -2, 3, 3}, Box 2: {-3, -2, 2, 3} -> union {-3, -2, 3, 3}
    CLAY_CHECK_EQ(r_wc.cover.size(), 1u);
    CLAY_CHECK(r_wc.cover[0] == (Box64{-3, -2, 3, 3}));

    // Multi-batch (> 2048) + 8-witness containment pruning: 1 huge outer box + 3000 contained boxes.
    const Box64 outer{-500, -500, 15000, 15000};
    std::vector<AtomRec> left_w;
    left_w.reserve(3001);
    left_w.push_back(AtomRec{0, outer});
    for (int64_t i = 0; i < 3000; ++i) {
        left_w.push_back(AtomRec{0, Box64{i, i, i + 2, i + 2}});
    }
    const std::vector<AtomRec> right_w = {AtomRec{0, Box64{0, 0, 1, 1}}};
    const auto r_w = forbidden_box_cover(left_w, right_w, 0);
    CLAY_CHECK(r_w.status == ForbiddenStatus::Ok);
    CLAY_CHECK_EQ(r_w.raw_count, 3001ULL);
    CLAY_CHECK_EQ(r_w.unique_count, 3001ULL);
    CLAY_CHECK_EQ(r_w.cover.size(), 1u);
    CLAY_CHECK(r_w.cover[0] == outer);
}

CLAY_TEST(PlaceSweepUnion, CheckpointHookCancellation) {
    struct AbortState {
        int calls = 0;
        int limit = 5;
    };
    auto hook_fn = [](void* raw) -> bool {
        auto* st = static_cast<AbortState*>(raw);
        ++st->calls;
        return st->calls < st->limit;
    };

    std::vector<Box64> many;
    many.reserve(2000);
    for (int64_t i = 0; i < 2000; ++i) {
        many.push_back(Box64{0, i * 2, 10, i * 2 + 1});
    }

    AbortState st1{0, 6};
    std::vector<Box64> out;
    const bool ok = rectangle_union_cover(many, out, CheckpointHook{hook_fn, &st1});
    CLAY_CHECK(!ok);
    CLAY_CHECK_EQ(st1.calls, 6);

    std::vector<AtomRec> atoms;
    atoms.reserve(300);
    for (int64_t i = 0; i < 300; ++i) {
        atoms.push_back(AtomRec{0, Box64{i * 10, 0, i * 10 + 3, 5}});
    }
    AbortState st2{0, 10};
    const auto fr = forbidden_box_cover(atoms, atoms, 1, CheckpointHook{hook_fn, &st2});
    CLAY_CHECK(fr.status == ForbiddenStatus::Cancelled);
    CLAY_CHECK_EQ(st2.calls, 10);
}

// ============================================================================
// 2. bvh_index tests
// ============================================================================

CLAY_TEST(PlaceBvhIndex, ConflictsAndCancellation) {
    BvhIndex empty_idx;
    CLAY_CHECK(empty_idx.empty());
    CLAY_CHECK(empty_idx.query_conflict_one(Box64{0, 0, 10, 10}, 5) == ConflictResult::NoConflict);

    // Build > 32 boxes to exercise multi-level BVH splits across X and Y.
    std::vector<Box64> indexed;
    for (int64_t gx = 0; gx < 6; ++gx) {
        for (int64_t gy = 0; gy < 6; ++gy) {
            indexed.push_back(Box64{gx * 100, gy * 100, gx * 100 + 20, gy * 100 + 20});
        }
    }
    BvhIndex idx;
    CLAY_CHECK(idx.build(indexed));
    CLAY_CHECK_EQ(idx.size(), 36u);
    CLAY_CHECK(!idx.empty());

    // Box sitting at gap == 15 from (0,0,20,20): clearance 15 >= 15 (clear), < 16 (conflict).
    const Box64 q{35, 0, 50, 20};
    CLAY_CHECK(idx.query_conflict_one(q, 15) == ConflictResult::NoConflict);
    CLAY_CHECK(idx.query_conflict_one(q, 16) == ConflictResult::Conflict);
    CLAY_CHECK(idx.query_conflict_one(q, 0) == ConflictResult::NoConflict);

    // Overlapping query conflicts at gap == 0, clear at gap == -10 if overlap depth <= 10.
    const Box64 ov{15, 0, 25, 20};  // clearance = -5
    CLAY_CHECK(idx.query_conflict_one(ov, 0) == ConflictResult::Conflict);
    CLAY_CHECK(idx.query_conflict_one(ov, -5) == ConflictResult::NoConflict);
    CLAY_CHECK(idx.query_conflict_one(ov, -4) == ConflictResult::Conflict);

    // Build & query cooperative cancellation.
    struct AbortState {
        int calls = 0;
        int limit = 3;
    };
    auto hook_fn = [](void* raw) -> bool {
        auto* st = static_cast<AbortState*>(raw);
        ++st->calls;
        return st->calls < st->limit;
    };

    std::vector<Box64> big_set;
    big_set.reserve(500);
    for (int64_t i = 0; i < 500; ++i) {
        big_set.push_back(Box64{i * 4, 0, i * 4 + 1, 1});
    }
    AbortState st_build{0, 4};
    BvhIndex aborted_idx;
    CLAY_CHECK(!aborted_idx.build(big_set, CheckpointHook{hook_fn, &st_build}));
    CLAY_CHECK_EQ(st_build.calls, 4);

    BvhIndex full_idx;
    CLAY_CHECK(full_idx.build(big_set));
    AbortState st_query{0, 2};
    CheckpointPoller poller(CheckpointHook{hook_fn, &st_query}, 3ULL);
    std::vector<Box64> queries;
    for (int64_t i = 0; i < 50; ++i) {
        queries.push_back(Box64{i * 4 + 2, 0, i * 4 + 3, 1});
    }
    CLAY_CHECK(full_idx.query_conflict_any(queries, 1, poller) == ConflictResult::Cancelled);
    CLAY_CHECK_EQ(st_query.calls, 2);
}

CLAY_TEST(PlaceBvhIndex, ConflictWitnessTieBreakingBy128BitArea) {
    // 1. Tie-breaking by exact 128-bit area when clearance is identical (overflows 64-bit area!).
    const int64_t kHuge = (1LL << 35);  // 3.4e10; square ~1.18e21 > UINT64_MAX (~1.84e19)
    const Box64 q{0, 0, 100, 100};
    const Box64 small_box{110, 0, 120, 10};                         // clearance = 10, area = 100
    const Box64 huge_a{110, 0, 110 + kHuge, kHuge};                 // clearance = 10, area = kHuge^2
    const Box64 huge_b{110, 0, 110 + kHuge + 1, kHuge};             // clearance = 10, area = (kHuge+1)*kHuge
    const Box64 closer_tiny{105, 0, 106, 1};                        // clearance = 5, area = 1

    BvhIndex idx_tie;
    CLAY_CHECK(idx_tie.build({small_box, huge_a, huge_b}));
    ConflictWitness w{};
    CLAY_CHECK(!idx_tie.find_conflict_witness(q, 10, w));
    CLAY_CHECK(idx_tie.find_conflict_witness(q, 11, w));
    CLAY_CHECK(w.box == huge_b);
    CLAY_CHECK_EQ(w.clearance_nm, 10);
    CLAY_CHECK(w.area == box_area_u128(huge_b));

    // 2. Strictly smaller clearance beats larger area.
    BvhIndex idx_closer;
    CLAY_CHECK(idx_closer.build({huge_b, closer_tiny}));
    CLAY_CHECK(idx_closer.find_conflict_witness(q, 20, w));
    CLAY_CHECK(w.box == closer_tiny);
    CLAY_CHECK_EQ(w.clearance_nm, 5);

    // 3. Equal clearance AND equal area breaks tie by canonical Box64 order.
    const Box64 eq1{-20, 0, -10, 10};  // clearance = 10, area = 100, x0 = -20
    const Box64 eq2{110, 0, 120, 10};  // clearance = 10, area = 100, x0 = 110
    BvhIndex idx_eq;
    CLAY_CHECK(idx_eq.build({eq2, eq1}));
    CLAY_CHECK(idx_eq.find_conflict_witness(q, 15, w));
    CLAY_CHECK(w.box == eq1);
}

CLAY_TEST(PlaceBvhIndex, IntersectingWithAndWithoutTouching) {
    const std::vector<Box64> boxes = {
        Box64{0, 0, 4, 4},
        Box64{0, 0, 4, 4},    // duplicate retained
        Box64{4, 0, 8, 4},    // edge-touching q
        Box64{4, 4, 8, 8},    // corner-touching q
        Box64{10, 10, 12, 12},
    };
    BvhIndex idx;
    CLAY_CHECK(idx.build(boxes));

    const Box64 q{2, 2, 4, 4};
    std::vector<Box64> strict_hits;
    idx.query_intersecting(q, false, strict_hits);
    std::sort(strict_hits.begin(), strict_hits.end());
    const std::vector<Box64> expected_strict = {Box64{0, 0, 4, 4}, Box64{0, 0, 4, 4}};
    CLAY_CHECK(strict_hits == expected_strict);

    std::vector<Box64> touch_hits;
    idx.query_intersecting(q, true, touch_hits);
    std::sort(touch_hits.begin(), touch_hits.end());
    const std::vector<Box64> expected_touch = {
        Box64{0, 0, 4, 4},
        Box64{0, 0, 4, 4},
        Box64{4, 0, 8, 4},
        Box64{4, 4, 8, 8},
    };
    CLAY_CHECK(touch_hits == expected_touch);
}

// ============================================================================
// 3. seqpair tests
// ============================================================================

CLAY_TEST(PlaceSeqPair, LongestPathDagCycleAndMissingWeight) {
    // Diamond DAG: 0->1 (12.5), 1->2 (7.25), 0->2 (15.0) => pos = {0.0, 12.5, 19.75}
    const std::vector<DagEdge> dag = {
        DagEdge{0, 1, 12.5, true, 0},
        DagEdge{1, 2, 7.25, true, 1},
        DagEdge{0, 2, 15.0, true, 2},
    };
    const auto ok = longest_path_positions(3, dag);
    CLAY_CHECK(ok.status == LongestPathStatus::Ok);
    CLAY_CHECK_EQ(ok.positions.size(), 3u);
    CLAY_CHECK_NEAR(ok.positions[0], 0.0, 1e-12);
    CLAY_CHECK_NEAR(ok.positions[1], 12.5, 1e-12);
    CLAY_CHECK_NEAR(ok.positions[2], 19.75, 1e-12);

    // Acyclic prefix + 3-cycle: S(0)->A(1)->B(2)->C(3)->A(1) => resolved 1/4.
    const std::vector<DagEdge> cyc = {
        DagEdge{0, 1, 1.0, true, 0},
        DagEdge{1, 2, 2.0, true, 1},
        DagEdge{2, 3, 3.0, true, 2},
        DagEdge{3, 1, 4.0, true, 3},
    };
    const auto cyc_res = longest_path_positions(4, cyc);
    CLAY_CHECK(cyc_res.status == LongestPathStatus::CycleDetected);
    CLAY_CHECK_EQ(cyc_res.resolved_count, 1u);

    // Missing weight on visited edge reports exact raw_index.
    const std::vector<DagEdge> missing = {
        DagEdge{0, 1, 1.0, true, 0},
        DagEdge{1, 2, 0.0, false, 7},
    };
    const auto miss_res = longest_path_positions(3, missing);
    CLAY_CHECK(miss_res.status == LongestPathStatus::MissingWeight);
    CLAY_CHECK_EQ(miss_res.missing_edge_index, 7u);
}

CLAY_TEST(PlaceSeqPair, PackBandSpHorizontalVerticalAndRelations) {
    const std::vector<double> bw = {10.0, 20.0, 30.0};
    const std::vector<double> bh = {5.0, 8.0, 6.0};
    const double gap = 2.0;

    // Horizontal row: gp = {0,1,2}, gm = {0,1,2} -> W = 10+2+20+2+30 = 64, H = max(5,8,6) = 8
    const auto horiz = pack_band_sp({0, 1, 2}, {0, 1, 2}, bw, bh, gap);
    CLAY_CHECK_NEAR(horiz.width, 64.0, 1e-12);
    CLAY_CHECK_NEAR(horiz.height, 8.0, 1e-12);
    const auto h_rels = extract_pair_relations(horiz.pos_plus, horiz.pos_minus);
    CLAY_CHECK_EQ(h_rels.size(), 3u);
    for (const auto& entry : h_rels) {
        CLAY_CHECK(entry.rel == PairRel::ILeftJ);
    }

    // Vertical stack: gp = {2,1,0}, gm = {0,1,2} -> W = max(10,20,30) = 30, H = 5+2+8+2+6 = 23
    const auto vert = pack_band_sp({2, 1, 0}, {0, 1, 2}, bw, bh, gap);
    CLAY_CHECK_NEAR(vert.width, 30.0, 1e-12);
    CLAY_CHECK_NEAR(vert.height, 23.0, 1e-12);
    const auto v_rels = extract_pair_relations(vert.pos_plus, vert.pos_minus);
    CLAY_CHECK_EQ(v_rels.size(), 3u);
    for (const auto& entry : v_rels) {
        CLAY_CHECK(entry.rel == PairRel::IBelowJ);
    }
}

CLAY_TEST(PlaceSeqPair, SearchBlockLayoutExhaustiveAndHillClimb) {
    // n < 2 returns empty relations.
    const auto tiny = search_block_layout({10.0}, {10.0}, 1.0);
    CLAY_CHECK(extract_pair_relations(tiny.pos_plus, tiny.pos_minus).empty());

    // n = 4 (exhaustive search): 4 identical 10x10 squares with gap=2.
    // Vertical stack: 10 x 46 (aspect 4.6, out_of_band=1).
    // Exhaustive search finds an in-band packing (34 x 22, aspect 1.545 in [1.4, 1.9], out_of_band=0)
    // that strictly beats the vertical stack under sp_cost_better.
    const std::vector<double> bw4 = {10.0, 10.0, 10.0, 10.0};
    const std::vector<double> bh4 = {10.0, 10.0, 10.0, 10.0};
    const auto res4 = search_block_layout(bw4, bh4, 2.0);
    CLAY_CHECK_EQ(res4.n, 4);
    const auto rels4 = extract_pair_relations(res4.pos_plus, res4.pos_minus);
    CLAY_CHECK_EQ(rels4.size(), 6u);

    // Reconstruct gp/gm from pos_plus/pos_minus and check cost strictly beats vertical stack.
    std::vector<int> gp4(4), gm4(4);
    for (int i = 0; i < 4; ++i) {
        gp4[static_cast<size_t>(res4.pos_plus[static_cast<size_t>(i)])] = i;
        gm4[static_cast<size_t>(res4.pos_minus[static_cast<size_t>(i)])] = i;
    }
    const auto seed4 = pack_band_sp({3, 2, 1, 0}, {0, 1, 2, 3}, bw4, bh4, 2.0);
    const auto packed4 = pack_band_sp(gp4, gm4, bw4, bh4, 2.0);
    const SpCost c_seed4 = cost_from_dims(seed4.width, seed4.height);
    const SpCost c_best4 = cost_from_dims(packed4.width, packed4.height);
    CLAY_CHECK_EQ(c_seed4.out_of_band, 1);
    CLAY_CHECK_EQ(c_best4.out_of_band, 0);
    CLAY_CHECK(sp_cost_better(c_best4, c_seed4));

    // n = 8 (hill-climb path): heterogeneous blocks improve over naive vertical stack.
    const std::vector<double> bw8 = {12.0, 18.0, 10.0, 22.0, 14.0, 16.0, 11.0, 20.0};
    const std::vector<double> bh8 = {8.0, 9.0, 7.0, 10.0, 6.0, 8.0, 9.0, 7.0};
    const auto res8 = search_block_layout(bw8, bh8, 2.0);
    CLAY_CHECK_EQ(res8.n, 8);
    const auto rels8 = extract_pair_relations(res8.pos_plus, res8.pos_minus);
    CLAY_CHECK_EQ(rels8.size(), 28u);

    std::vector<int> gp8(8), gm8(8);
    for (int i = 0; i < 8; ++i) {
        gp8[static_cast<size_t>(res8.pos_plus[static_cast<size_t>(i)])] = i;
        gm8[static_cast<size_t>(res8.pos_minus[static_cast<size_t>(i)])] = i;
    }
    const auto seed8 = pack_band_sp({7, 6, 5, 4, 3, 2, 1, 0}, {0, 1, 2, 3, 4, 5, 6, 7}, bw8, bh8, 2.0);
    const auto best8 = pack_band_sp(gp8, gm8, bw8, bh8, 2.0);
    const SpCost c_seed = cost_from_dims(seed8.width, seed8.height);
    const SpCost c_best = cost_from_dims(best8.width, best8.height);
    CLAY_CHECK(sp_cost_better(c_best, c_seed) ||
               (c_best.out_of_band == c_seed.out_of_band && c_best.area <= c_seed.area));
}

CLAY_TEST(PlaceSeqPair, SearchBandDevLayoutExhaustiveAndHillClimb) {
    // 3 bands (exhaustive): [2 devs, 1 dev, 2 devs].
    const std::vector<uint32_t> off3 = {0, 2, 3, 5};
    const std::vector<double> hw3 = {1000.0, 1200.0, 1500.0, 900.0, 1100.0};
    const std::vector<double> hh3 = {1500.0, 1500.0, 2000.0, 1200.0, 1400.0};
    const auto res3 = search_band_dev_layout(off3, hw3, hh3, 500.0);
    CLAY_CHECK_EQ(res3.n, 3);
    CLAY_CHECK_EQ(extract_pair_relations(res3.pos_plus, res3.pos_minus).size(), 3u);

    // 7 bands (hill-climb): 1..2 devices per band.
    const std::vector<uint32_t> off7 = {0, 2, 3, 5, 6, 8, 9, 11};
    const std::vector<double> hw7 = {
        800.0, 900.0, 1200.0, 700.0, 1100.0, 1500.0,
        600.0, 1000.0, 1300.0, 950.0, 850.0
    };
    const std::vector<double> hh7 = {
        1000.0, 1100.0, 1300.0, 900.0, 1200.0, 1400.0,
        800.0, 1000.0, 1500.0, 1100.0, 950.0
    };
    const auto res7 = search_band_dev_layout(off7, hw7, hh7, 400.0);
    CLAY_CHECK_EQ(res7.n, 7);
    const auto rels7 = extract_pair_relations(res7.pos_plus, res7.pos_minus);
    CLAY_CHECK_EQ(rels7.size(), 21u);

    // Determinism check: identical inputs produce identical pos_plus/pos_minus.
    const auto res7_repeat = search_band_dev_layout(off7, hw7, hh7, 400.0);
    CLAY_CHECK(res7.pos_plus == res7_repeat.pos_plus);
    CLAY_CHECK(res7.pos_minus == res7_repeat.pos_minus);
}
