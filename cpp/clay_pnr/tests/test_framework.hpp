#pragma once

// Minimal zero-dependency C++20 native unit-test harness for Clay PnR core libraries.
// Designed for sub-second native compilation & execution with zero <Python.h> dependency.

#include <cmath>
#include <cstdio>
#include <exception>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace clay_pnr::test {

struct TestCase {
    std::string suite;
    std::string name;
    std::function<void()> fn;
};

struct TestRegistry {
    std::vector<TestCase> cases;

    static TestRegistry& instance() {
        static TestRegistry reg;
        return reg;
    }

    int register_case(const char* suite, const char* name, std::function<void()> fn) {
        cases.push_back(TestCase{suite, name, std::move(fn)});
        return static_cast<int>(cases.size());
    }
};

struct AssertionFailure : public std::runtime_error {
    explicit AssertionFailure(const std::string& msg) : std::runtime_error(msg) {}
};

inline int run_all_tests(const char* filter = nullptr) {
    auto& reg = TestRegistry::instance();
    int total = 0;
    int failed = 0;
    for (const auto& tc : reg.cases) {
        const std::string full = tc.suite + "." + tc.name;
        if (filter && filter[0] != '\0' && full.find(filter) == std::string::npos) {
            continue;
        }
        ++total;
        try {
            tc.fn();
            std::printf("[  PASS  ] %s\n", full.c_str());
        } catch (const std::exception& ex) {
            ++failed;
            std::printf("[  FAIL  ] %s: %s\n", full.c_str(), ex.what());
        } catch (...) {
            ++failed;
            std::printf("[  FAIL  ] %s: unknown exception\n", full.c_str());
        }
    }
    std::printf("[SUMMARY ] %d/%d passed (%d failed)\n", total - failed, total, failed);
    return failed == 0 ? 0 : 1;
}

template <typename A, typename B>
inline void check_eq_impl(const A& a, const B& b, const char* expr_a, const char* expr_b,
                          const char* file, int line) {
    if (!(a == b)) {
        std::ostringstream oss;
        oss << file << ":" << line << ": expected " << expr_a << " == " << expr_b
            << " (got " << a << " vs " << b << ")";
        throw AssertionFailure(oss.str());
    }
}

inline void check_near_impl(double a, double b, double eps, const char* expr_a, const char* expr_b,
                            const char* file, int line) {
    if (std::fabs(a - b) > eps) {
        std::ostringstream oss;
        oss << file << ":" << line << ": expected |" << expr_a << " - " << expr_b
            << "| <= " << eps << " (got " << a << " vs " << b << ")";
        throw AssertionFailure(oss.str());
    }
}

}  // namespace clay_pnr::test

#define CLAY_TEST(suite, name)                                                                 \
    static void clay_test_fn_##suite##_##name();                                               \
    static const int clay_test_reg_##suite##_##name =                                          \
        ::clay_pnr::test::TestRegistry::instance().register_case(                              \
            #suite, #name, &clay_test_fn_##suite##_##name);                                    \
    static void clay_test_fn_##suite##_##name()

#define CLAY_CHECK(...)                                                                        \
    do {                                                                                       \
        if (!(__VA_ARGS__)) {                                                                  \
            std::ostringstream clay_oss_;                                                      \
            clay_oss_ << __FILE__ << ":" << __LINE__ << ": check failed: " #__VA_ARGS__;       \
            throw ::clay_pnr::test::AssertionFailure(clay_oss_.str());                         \
        }                                                                                      \
    } while (false)

#define CLAY_REQUIRE(...) CLAY_CHECK(__VA_ARGS__)

#define CLAY_CHECK_EQ(a, b)                                                                    \
    ::clay_pnr::test::check_eq_impl((a), (b), #a, #b, __FILE__, __LINE__)

#define CLAY_CHECK_NEAR(a, b, eps)                                                             \
    ::clay_pnr::test::check_near_impl((a), (b), (eps), #a, #b, __FILE__, __LINE__)

