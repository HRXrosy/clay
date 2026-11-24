// Pure C++20 SymbolTable & SessionBudget implementation.
// Strictly free of <Python.h>.

#include "clay_pnr/core/symbol_table.hpp"

namespace clay_pnr::core {

SymbolTable::SymbolTable() {
    id_to_str_.emplace_back("");  // id 0 == kInvalidSymbol
}

uint32_t SymbolTable::intern(std::string_view name) {
    if (name.empty()) {
        return kInvalidSymbol;
    }
    const auto it = str_to_id_.find(name);
    if (it != str_to_id_.end()) {
        return it->second;
    }
    const uint32_t id = static_cast<uint32_t>(id_to_str_.size());
    std::string owned(name);
    id_to_str_.push_back(owned);
    str_to_id_.emplace(std::move(owned), id);
    return id;
}

uint32_t SymbolTable::find(std::string_view name) const noexcept {
    if (name.empty()) {
        return kInvalidSymbol;
    }
    const auto it = str_to_id_.find(name);
    return (it == str_to_id_.end()) ? kInvalidSymbol : it->second;
}

std::string_view SymbolTable::resolve(uint32_t id) const noexcept {
    if (id == kInvalidSymbol || id >= id_to_str_.size()) {
        return {};
    }
    return id_to_str_[id];
}

void SymbolTable::clear() {
    id_to_str_.clear();
    id_to_str_.emplace_back("");
    str_to_id_.clear();
}

SessionBudget::SessionBudget(double wall_seconds,
                             uint64_t max_ticks,
                             CheckpointHook external_hook) noexcept
    : start_tp_(std::chrono::steady_clock::now()),
      wall_limit_s_(wall_seconds),
      max_ticks_(max_ticks),
      ticks_(0),
      external_hook_(external_hook),
      cancelled_(false),
      status_(BudgetStatus::Ok) {}

void SessionBudget::reset_clock() noexcept {
    start_tp_ = std::chrono::steady_clock::now();
    ticks_ = 0;
    cancelled_ = false;
    status_ = BudgetStatus::Ok;
}

double SessionBudget::elapsed_seconds() const noexcept {
    const auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now - start_tp_).count();
}

BudgetStatus SessionBudget::check_now() noexcept {
    if (status_ != BudgetStatus::Ok) {
        return status_;
    }
    if (cancelled_) {
        status_ = BudgetStatus::Cancelled;
        return status_;
    }
    if (max_ticks_ > 0 && ticks_ >= max_ticks_) {
        status_ = BudgetStatus::TicksExceeded;
        return status_;
    }
    if (wall_limit_s_ > 0.0 && elapsed_seconds() >= wall_limit_s_) {
        status_ = BudgetStatus::WallExceeded;
        return status_;
    }
    if (external_hook_.enabled() && !external_hook_.call_now()) {
        status_ = BudgetStatus::Cancelled;
        return status_;
    }
    return BudgetStatus::Ok;
}

BudgetStatus SessionBudget::tick(uint64_t delta) noexcept {
    ticks_ += delta;
    if (status_ != BudgetStatus::Ok) {
        return status_;
    }
    if (max_ticks_ > 0 && ticks_ >= max_ticks_) {
        status_ = BudgetStatus::TicksExceeded;
        return status_;
    }
    return check_now();
}

bool SessionBudget::hook_trampoline(void* raw) noexcept {
    auto* self = static_cast<SessionBudget*>(raw);
    return self->check_now() == BudgetStatus::Ok;
}

CheckpointHook SessionBudget::as_checkpoint_hook() noexcept {
    return CheckpointHook{&SessionBudget::hook_trampoline, this};
}

}  // namespace clay_pnr::core

