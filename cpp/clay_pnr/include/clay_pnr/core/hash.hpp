#pragma once

// Pure C++20 hashing & spatial bucket key primitives for Clay PnR.
// Strictly free of <Python.h>.

#include <cstddef>
#include <cstdint>

namespace clay_pnr::core {

inline constexpr uint64_t mix64(uint64_t k) noexcept {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return k;
}

inline constexpr uint64_t hash_combine64(uint64_t seed, uint64_t value) noexcept {
    return mix64(seed ^ (mix64(value) + 0x9e3779b97f4a7c15ULL));
}

struct BucketKey {
    int64_t bx = 0;
    int64_t by = 0;

    constexpr bool operator==(const BucketKey& o) const noexcept {
        return bx == o.bx && by == o.by;
    }
};

struct BucketKeyHash {
    size_t operator()(const BucketKey& k) const noexcept {
        const uint64_t h = mix64(static_cast<uint64_t>(k.bx) ^
                                 (mix64(static_cast<uint64_t>(k.by)) + 0x9e3779b97f4a7c15ULL));
        return static_cast<size_t>(h);
    }
};

struct ClaimKey {
    uint32_t owner_id = 0;
    uintptr_t rect_id = 0;

    constexpr bool operator==(const ClaimKey& o) const noexcept {
        return owner_id == o.owner_id && rect_id == o.rect_id;
    }
};

struct ClaimKeyHash {
    size_t operator()(const ClaimKey& k) const noexcept {
        const uint64_t h = mix64(static_cast<uint64_t>(k.rect_id) ^
                                 (static_cast<uint64_t>(k.owner_id) * 0x9e3779b97f4a7c15ULL));
        return static_cast<size_t>(h);
    }
};

}  // namespace clay_pnr::core

