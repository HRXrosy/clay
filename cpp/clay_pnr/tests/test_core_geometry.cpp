#include "test_framework.hpp"

#include "clay_pnr/core/geometry.hpp"
#include "clay_pnr/core/hash.hpp"
#include "clay_pnr/core/status.hpp"

#include <cstdint>
#include <vector>

using namespace clay_pnr::core;

CLAY_TEST(CoreGeometry, BoxNormalizeAndClearanceLinf) {
    const Box64 b = normalize_box(20, 40, 10, 5);
    CLAY_CHECK_EQ(b.x0, 10);
    CLAY_CHECK_EQ(b.y0, 5);
    CLAY_CHECK_EQ(b.x1, 20);
    CLAY_CHECK_EQ(b.y1, 40);

    const Box64 a{0, 0, 100, 100};
    const Box64 c{130, 20, 200, 80};
    CLAY_CHECK_EQ(box_clearance_linf(a, c), 30);
    CLAY_CHECK(rects_clear_linf(a, c, 30));
    CLAY_CHECK(!rects_clear_linf(a, c, 31));
    CLAY_CHECK(rects_clear_linf(a, a, 0));
    CLAY_CHECK((closed_rects_touch(Box64{0, 0, 10, 10}, Box64{10, 10, 20, 20})));
    CLAY_CHECK((!closed_rects_touch(Box64{0, 0, 10, 10}, Box64{11, 0, 20, 10})));
}

CLAY_TEST(CoreGeometry, Exact128BitEuclideanClearanceAndArea) {
    // 3-4-5 triangle exact boundary at standard scale.
    const Box64 a{0, 0, 10, 10};
    const Box64 b{13, 14, 20, 20};  // dx=3, dy=4, dist=5
    CLAY_CHECK(rects_clear_euclid_exact(a, b, 5));
    CLAY_CHECK(!rects_clear_euclid_exact(a, b, 6));

    // Exact 128-bit precision above 2^31 (where double/long double loses precision!).
    const int64_t kScale = (1LL << 33);
    const int64_t dx = 3LL * kScale;
    const int64_t dy = 4LL * kScale;
    const int64_t sp_exact = 5LL * kScale;
    CLAY_CHECK(!euclid_sq_lt_128(static_cast<uint64_t>(dx), static_cast<uint64_t>(dy),
                                 static_cast<uint64_t>(sp_exact)));
    CLAY_CHECK(euclid_sq_lt_128(static_cast<uint64_t>(dx), static_cast<uint64_t>(dy),
                                static_cast<uint64_t>(sp_exact + 1)));

    const Box64 big_a{0, 0, 10, 10};
    const Box64 big_b{10 + dx, 10 + dy, 20 + dx, 20 + dy};
    CLAY_CHECK(rects_clear_euclid_exact(big_a, big_b, sp_exact));
    CLAY_CHECK(!rects_clear_euclid_exact(big_a, big_b, sp_exact + 1));

    // Overflow-free 128-bit box area.
    const Box64 huge{-kCoordLimit60, -kCoordLimit60, kCoordLimit60, kCoordLimit60};
    const U128 area = box_area_u128(huge);
    CLAY_CHECK((area > U128{0, 0}));
    const Box64 slightly_smaller{-kCoordLimit60, -kCoordLimit60, kCoordLimit60 - 1, kCoordLimit60};
    CLAY_CHECK(area > box_area_u128(slightly_smaller));
}

CLAY_TEST(CoreGeometry, FloorCeilDivAndBankersRounding) {
    CLAY_CHECK_EQ(floor_div_i64(7, 3), 2);
    CLAY_CHECK_EQ(floor_div_i64(-7, 3), -3);
    CLAY_CHECK_EQ(floor_div_i64(-6, 3), -2);
    CLAY_CHECK_EQ(floor_div_i64(7, -3), -3);
    CLAY_CHECK_EQ(floor_div_i64(-7, -3), 2);
    CLAY_CHECK_EQ(ceil_div_i64(7, 3), 3);
    CLAY_CHECK_EQ(ceil_div_i64(-7, 3), -2);
    CLAY_CHECK_EQ(floor_div_pos(-1, 6000), -1);
    CLAY_CHECK_EQ(floor_div_pos(-6000, 6000), -1);
    CLAY_CHECK_EQ(floor_div_pos(-6001, 6000), -2);
    CLAY_CHECK_EQ(py_round_i64(2.5), 2);
    CLAY_CHECK_EQ(py_round_i64(3.5), 4);
    CLAY_CHECK_EQ(py_round_i64(-2.5), -2);
    CLAY_CHECK_EQ(py_round_i64(-3.5), -4);
}

CLAY_TEST(CoreGeometry, GapCoverAndThroat) {
    const Box64 gap{0, 0, 100, 100};
    const std::vector<Box64> partial = {{0, 0, 50, 100}, {50, 0, 100, 99}};
    CLAY_CHECK(!box_fully_covered_by_rects(gap, partial));

    const std::vector<Box64> full = {{0, 0, 60, 100}, {50, 0, 100, 50}, {50, 40, 100, 100}};
    CLAY_CHECK(box_fully_covered_by_rects(gap, full));

    // Owned landing throat checks.
    const Box64 atom{0, 0, 200, 200};
    const Box64 pad_ok{100, 0, 300, 200};     // overlap 100x200 >= 150
    const Box64 pad_small{180, 180, 300, 300}; // overlap 20x20, hypot ~28.28 < 50
    CLAY_CHECK(owned_landing_has_min_throat(atom, pad_ok, 150));
    CLAY_CHECK(!owned_landing_has_min_throat(atom, pad_small, 50));
    CLAY_CHECK(owned_landing_has_min_throat(atom, pad_small, 28));
}

CLAY_TEST(CoreStatus, CheckpointHookAndPoller) {
    struct Counter {
        int calls = 0;
        int stop_at = 3;
    } counter;

    auto cb = [](void* raw) -> bool {
        auto* c = static_cast<Counter*>(raw);
        ++c->calls;
        return c->calls < c->stop_at;
    };

    CheckpointPoller poller(CheckpointHook{cb, &counter}, 3ULL);  // every 4 steps
    int completed = 0;
    for (int i = 0; i < 20; ++i) {
        if (!poller.poll_step()) break;
        ++completed;
    }
    CLAY_CHECK_EQ(counter.calls, 3);
    CLAY_CHECK_EQ(completed, 11);  // steps 4 (ok), 8 (ok), 12 (stop!) -> 11 iterations finished
}

CLAY_TEST(CoreHash, DeterministicBucketAndClaimHashing) {
    BucketKeyHash bh;
    const size_t h1 = bh(BucketKey{12, -34});
    const size_t h2 = bh(BucketKey{12, -34});
    const size_t h3 = bh(BucketKey{-34, 12});
    CLAY_CHECK_EQ(h1, h2);
    CLAY_CHECK(h1 != h3);

    ClaimKeyHash ch;
    CLAY_CHECK_EQ(ch(ClaimKey{7u, 0x12345678ULL}), ch(ClaimKey{7u, 0x12345678ULL}));
    CLAY_CHECK(ch(ClaimKey{7u, 0x12345678ULL}) != ch(ClaimKey{8u, 0x12345678ULL}));
}

