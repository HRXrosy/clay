#pragma once

// Pure C++20 deterministic string interning (SymbolTable) and engine-owned
// execution budget / cancellation controller (SessionBudget).
// Strictly free of <Python.h>.

#include "clay_pnr/core/status.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace clay_pnr::core {

constexpr uint32_t kInvalidSymbol = 0u;

// Transparent C++20 string hash/equality for zero-allocation string_view lookup.
struct StringViewHash {
    using is_transparent = void;

    size_t operator()(std::string_view sv) const noexcept {
        // FNV-1a 64-bit
        uint64_t h = 14695981039346656037ULL;
        for (unsigned char c : sv) {
            h ^= static_cast<uint64_t>(c);
            h *= 1099511628211ULL;
        }
        return static_cast<size_t>(h);
    }

    size_t operator()(const std::string& s) const noexcept {
        return (*this)(std::string_view{s});
    }
};

struct StringViewEq {
    using is_transparent = void;

    bool operator()(std::string_view a, std::string_view b) const noexcept {
        return a == b;
    }
};

// Deterministic 1-based string interner (0 == kInvalidSymbol / empty).
class SymbolTable {
public:
    SymbolTable();

    // Intern a string (empty string maps to kInvalidSymbol == 0).
    uint32_t intern(std::string_view name);

    // Zero-allocation lookup; returns kInvalidSymbol if absent.
    uint32_t find(std::string_view name) const noexcept;

    // Resolve symbol ID back to string_view (empty for kInvalidSymbol or out of range).
    std::string_view resolve(uint32_t id) const noexcept;

    // Number of interned non-empty symbols.
    size_t size() const noexcept { return id_to_str_.size() - 1; }

    void clear();

private:
    std::vector<std::string> id_to_str_;
    std::unordered_map<std::string, uint32_t, StringViewHash, StringViewEq> str_to_id_;
};

enum class BudgetStatus : uint8_t {
    Ok = 0,
    WallExceeded = 1,
    TicksExceeded = 2,
    Cancelled = 3,
};

// Engine-native cooperative budget & cancellation tracker usable standalone in C++20
// AND bridgeable to any existing CheckpointHook / CheckpointPoller kernel.
class SessionBudget {
public:
    explicit SessionBudget(double wall_seconds = 0.0,
                           uint64_t max_ticks = 0,
                           CheckpointHook external_hook = {}) noexcept;

    void reset_clock() noexcept;
    void cancel() noexcept { cancelled_ = true; status_ = BudgetStatus::Cancelled; }

    // Advance deterministic tick counter by `delta` and verify wall/tick/external limits.
    BudgetStatus tick(uint64_t delta = 1) noexcept;

    // Immediate wall + external hook check without advancing ticks.
    BudgetStatus check_now() noexcept;

    BudgetStatus status() const noexcept { return status_; }
    bool ok() const noexcept { return status_ == BudgetStatus::Ok; }
    uint64_t ticks() const noexcept { return ticks_; }
    double elapsed_seconds() const noexcept;

    // Export as a zero-overhead CheckpointHook for legacy/existing C++20 kernels.
    CheckpointHook as_checkpoint_hook() noexcept;

private:
    static bool hook_trampoline(void* raw) noexcept;

    std::chrono::steady_clock::time_point start_tp_{};
    double wall_limit_s_ = 0.0;
    uint64_t max_ticks_ = 0;
    uint64_t ticks_ = 0;
    CheckpointHook external_hook_{};
    bool cancelled_ = false;
    BudgetStatus status_ = BudgetStatus::Ok;
};

}  // namespace clay_pnr::core

