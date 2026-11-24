#include "test_framework.hpp"

#include "clay_pnr/core/polygon_set.hpp"
#include "clay_pnr/core/symbol_table.hpp"
#include "clay_pnr/gds/gds_stream.hpp"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

using namespace clay_pnr::core;
using namespace clay_pnr::gds;

CLAY_TEST(CorePolygonSet, CanonicalUnionAndBoolOps) {
    // Two overlapping 100x100 squares offset by (50, 50) -> L/stepped union of area 17500.
    const PolygonSet a = PolygonSet::from_box(Box64{0, 0, 100, 100});
    const PolygonSet b = PolygonSet::from_box(Box64{50, 50, 150, 150});

    const PolygonSet u = PolygonSet::op_or(a, b);
    const PolygonSet inter = PolygonSet::op_and(a, b);
    const PolygonSet diff = PolygonSet::op_not(a, b);
    const PolygonSet sym = PolygonSet::op_xor(a, b);

    CLAY_CHECK_EQ(u.area_u128().lo, 17500ULL);
    CLAY_CHECK_EQ(inter.area_u128().lo, 2500ULL);
    CLAY_CHECK_EQ(diff.area_u128().lo, 7500ULL);
    CLAY_CHECK_EQ(sym.area_u128().lo, 15000ULL);

    // Set algebraic identities: (A - B) U (A & B) == A, (A ^ B) U (A & B) == (A U B).
    CLAY_CHECK(PolygonSet::op_or(diff, inter) == a);
    CLAY_CHECK(PolygonSet::op_or(sym, inter) == u);
    CLAY_CHECK(PolygonSet::op_not(u, a) == PolygonSet::op_not(b, a));
}

CLAY_TEST(CorePolygonSet, NonConvexErosionAndHoles) {
    // Square frame [0, 100]^2 with interior hole [30, 70]^2 (wall thickness 30).
    const PolygonSet outer = PolygonSet::from_box(Box64{0, 0, 100, 100});
    const PolygonSet hole = PolygonSet::from_box(Box64{30, 30, 70, 70});
    const PolygonSet frame = PolygonSet::op_not(outer, hole);
    CLAY_CHECK_EQ(frame.area_u128().lo, 10000ULL - 1600ULL);

    // Erode by 10 -> outer shrinks to [10, 90]^2, hole expands to [20, 80]^2 -> wall thickness 10.
    const PolygonSet eroded = frame.sized(-10, -10);
    const PolygonSet expected = PolygonSet::op_not(
        PolygonSet::from_box(Box64{10, 10, 90, 90}),
        PolygonSet::from_box(Box64{20, 20, 80, 80}));
    CLAY_CHECK(eroded == expected);
    CLAY_CHECK_EQ(eroded.area_u128().lo, 6400ULL - 3600ULL);

    // Erode by 15 -> vanishes completely.
    CLAY_CHECK(frame.sized(-15, -15).empty());

    // Grow by 15 -> closes the 40x40 hole (becomes [0-15, 100+15]^2 with [45, 55]^2 hole remaining).
    const PolygonSet grown = frame.sized(15, 15);
    const PolygonSet expected_grown = PolygonSet::op_not(
        PolygonSet::from_box(Box64{-15, -15, 115, 115}),
        PolygonSet::from_box(Box64{45, 45, 55, 55}));
    CLAY_CHECK(grown == expected_grown);

    // Grow by 20 -> hole closes completely into solid [-20, 120]^2.
    CLAY_CHECK(frame.sized(20, 20) == PolygonSet::from_box(Box64{-20, -20, 120, 120}));
}

CLAY_TEST(CorePolygonSet, ConnectedComponentsAndContours) {
    // Two squares kissing at corner (10, 10) + one disconnected square at [50, 60]^2.
    const PolygonSet ps = PolygonSet::from_boxes({
        Box64{0, 0, 10, 10},
        Box64{10, 10, 20, 20},
        Box64{50, 50, 60, 60},
    });
    const auto comps_corner = ps.connected_components(true);
    const auto comps_edge = ps.connected_components(false);
    CLAY_CHECK_EQ(comps_corner.size(), 2u);
    CLAY_CHECK_EQ(comps_edge.size(), 3u);

    // Contours of frame with hole: 1 CCW outer loop + 1 CW hole loop.
    const PolygonSet frame = PolygonSet::op_not(
        PolygonSet::from_box(Box64{0, 0, 100, 100}),
        PolygonSet::from_box(Box64{25, 25, 75, 75}));
    const auto contours = frame.to_contours();
    CLAY_CHECK_EQ(contours.size(), 2u);
    CLAY_CHECK(!contours[0].is_hole);
    CLAY_CHECK(contours[1].is_hole);
    CLAY_CHECK_EQ(PolygonSet::contour_signed_area2(contours[0].points), 20000LL);
    CLAY_CHECK_EQ(PolygonSet::contour_signed_area2(contours[1].points), -5000LL);
}

CLAY_TEST(CoreSymbolTableAndBudget, DeterministicInterningAndLimits) {
    SymbolTable st;
    CLAY_CHECK_EQ(st.intern(""), kInvalidSymbol);
    const uint32_t vdd = st.intern("VDD");
    const uint32_t vss = st.intern("VSS");
    CLAY_CHECK_EQ(vdd, 1u);
    CLAY_CHECK_EQ(vss, 2u);
    CLAY_CHECK_EQ(st.intern("VDD"), vdd);
    CLAY_CHECK_EQ(st.find("VSS"), vss);
    CLAY_CHECK_EQ(st.find("MISSING"), kInvalidSymbol);
    CLAY_CHECK(st.resolve(vdd) == "VDD");

    SessionBudget budget(0.0, 5);
    CLAY_CHECK(budget.ok());
    CheckpointPoller poller(budget.as_checkpoint_hook(), 0ULL);
    int steps = 0;
    for (int i = 0; i < 10; ++i) {
        if (budget.tick(1) != BudgetStatus::Ok) break;
        if (!poller.call_now()) break;
        ++steps;
    }
    CLAY_CHECK_EQ(steps, 4);
    CLAY_CHECK_EQ(static_cast<int>(budget.status()), static_cast<int>(BudgetStatus::TicksExceeded));
}

CLAY_TEST(GdsStream, RealEncodingAndHierarchicalRoundTrip) {
    // Verify exact GDSII Excess-64 real encoding/decoding across standard DBU/user scales.
    for (const double v : {0.0, 1.0, -1.0, 1e-3, 1e-6, 1e-9, 5e-9, 0.001, 2.5, -123.456}) {
        const auto enc = double_to_gds_real(v);
        const double dec = gds_real_to_double(enc.data());
        CLAY_CHECK_NEAR(dec, v, std::fabs(v) * 1e-14 + 1e-18);
    }
    // Verify exact byte pattern for 1.0 (0x41, 0x10, 0, 0, 0, 0, 0, 0).
    const auto one = double_to_gds_real(1.0);
    CLAY_CHECK_EQ(static_cast<int>(one[0]), 0x41);
    CLAY_CHECK_EQ(static_cast<int>(one[1]), 0x10);

    // Build hierarchical library with SREF (90 deg rotation) + AREF (2x3 array) + PATH + TEXT.
    GdsLibrary lib;
    lib.name = "UNIT_TEST_LIB";
    auto& leaf = lib.add_cell("LEAF");
    leaf.add_box(34, 0, Box64{0, 0, 100, 50});  // area 5000

    auto& top = lib.add_cell("TOP");
    top.add_box(34, 0, Box64{0, 0, 200, 200});  // area 40000
    top.srefs.push_back(GdsSref{"LEAF", 500, 0, 0, 1.0, 90.0});  // rotated -> [450, 500] x [0, 100], area 5000
    top.arefs.push_back(GdsAref{"LEAF", 2, 3, {1000, 0}, {1400, 0}, {1000, 600}, 0, 1.0, 0.0});  // 6x5000 = 30000
    top.texts.push_back(GdsText{34, 10, 100, 100, "VDD"});

    const auto bytes = write_gds_bytes(lib);
    CLAY_CHECK(bytes.size() > 100u);

    GdsLibrary parsed;
    std::string err;
    CLAY_CHECK(read_gds_bytes(bytes.data(), bytes.size(), parsed, &err));
    CLAY_CHECK_EQ(parsed.name, std::string("UNIT_TEST_LIB"));
    CLAY_CHECK_EQ(parsed.cells.size(), 2u);
    CLAY_CHECK_NEAR(parsed.user_unit, 1e-3, 1e-15);
    CLAY_CHECK_NEAR(parsed.dbu_meters, 1e-9, 1e-21);

    const auto summary = summarize_gds_layers(parsed, "TOP");
    CLAY_CHECK_EQ(summary.size(), 1u);
    CLAY_CHECK_EQ(summary[0].layer, 34);
    CLAY_CHECK_EQ(summary[0].datatype, 0);
    // 40000 + 5000 + 30000 = 75000 (all disjoint!)
    CLAY_CHECK_EQ(summary[0].union_area_dbu2.lo, 75000ULL);
}

