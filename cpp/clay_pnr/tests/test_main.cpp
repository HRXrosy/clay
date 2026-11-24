#include "test_framework.hpp"

int main(int argc, char** argv) {
    const char* filter = (argc > 1) ? argv[1] : nullptr;
    return clay_pnr::test::run_all_tests(filter);
}

