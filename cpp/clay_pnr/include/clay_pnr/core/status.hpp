#pragma once

// Pure C++20 cooperative cancellation & status primitives for Clay PnR.
// Strictly free of <Python.h>; CPython callbacks bridge in via CheckpointHook.

#include <cstddef>
#include <cstdint>

namespace clay_pnr::core {

enum class IntStatus : uint8_t {
    Ok = 0,
    NotExactInt = 1,
    Overflow = 2,
};

// Zero-overhead type-erased cancellation / checkpoint callback.
// Returns true to continue execution, false to abort immediately (e.g., when a
// Python ExecutionStopped exception or cooperative deadline has been raised).
struct CheckpointHook {
    using Fn = bool (*)(void*);

    Fn fn = nullptr;
    void* ctx = nullptr;

    constexpr bool enabled() const noexcept { return fn != nullptr; }

    inline bool call_now() const {
        return fn ? fn(ctx) : true;
    }
};

// Periodic checkpoint poller parameterized by a power-of-two mask (default 255 -> every 256 steps).
// Exact match for Python's `if checkpoint and (idx & mask) == 0: checkpoint()`.
struct CheckpointPoller {
    CheckpointHook hook{};
    uint64_t step = 0;
    uint64_t mask = 255ULL;

    constexpr CheckpointPoller() noexcept = default;
    constexpr explicit CheckpointPoller(CheckpointHook h, uint64_t poll_mask = 255ULL) noexcept
        : hook(h), step(0), mask(poll_mask) {}

    constexpr bool enabled() const noexcept { return hook.enabled(); }

    inline bool call_now() const {
        return hook.call_now();
    }

    inline bool poll_index(size_t idx) const {
        if (!hook.enabled() || (static_cast<uint64_t>(idx) & mask) != 0ULL) {
            return true;
        }
        return hook.call_now();
    }

    inline bool poll_step() {
        if (!hook.enabled()) return true;
        ++step;
        if ((step & mask) != 0ULL) return true;
        return hook.call_now();
    }
};

}  // namespace clay_pnr::core

