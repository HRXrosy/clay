cmake_minimum_required(VERSION 3.20)
project(clay_pnr LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(CLAY_PNR_BUILD_PYTHON "Build CPython extension modules into clay/_native" ON)
option(CLAY_PNR_BUILD_TESTS "Build pure C++20 native unit tests (no Python dependency)" ON)
option(CLAY_PNR_WERROR "Treat compiler warnings as errors" OFF)

set(CLAY_PNR_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(CLAY_NATIVE_OUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../clay/_native")

function(clay_apply_cxx_flags target_name)
    if(MSVC)
        target_compile_options(${target_name} PRIVATE
            /O2 /Oi /Ot /arch:AVX2 /EHsc /DNDEBUG /W4 /permissive-
        )
        if(CLAY_PNR_WERROR)
            target_compile_options(${target_name} PRIVATE /WX)
        endif()
    else()
        target_compile_options(${target_name} PRIVATE
            -O3 -march=native -DNDEBUG -Wall -Wextra -Wpedantic
        )
        if(CLAY_PNR_WERROR)
            target_compile_options(${target_name} PRIVATE -Werror)
        endif()
    endif()
endfunction()

# ---------------------------------------------------------------------------
# 1. Pure C++20 Core Library (strictly free of <Python.h>)
# ---------------------------------------------------------------------------
add_library(clay_pnr_core STATIC
    src/core/polygon_set.cpp
    src/core/symbol_table.cpp
    src/gds/gds_stream.cpp
    src/layout/layout_db.cpp
    src/pcell/mos_pcell.cpp
    src/ir/clay_parser.cpp
    src/engine/standalone_synth.cpp
    src/place/sweep_union.cpp
    src/place/bvh_index.cpp
    src/place/seqpair.cpp
    src/place/compactor.cpp
    src/place/envelope_placer.cpp
    src/place/banded_placer.cpp
    src/route/pathfinder_core.cpp
    src/route/gridless_core.cpp
    src/route/gridless_spatial.cpp
    src/route/gridless_search.cpp
)
target_include_directories(clay_pnr_core PUBLIC "${CLAY_PNR_INCLUDE_DIR}")
clay_apply_cxx_flags(clay_pnr_core)

# Standalone Pure C++20 Engine CLI (zero Python dependency)
add_executable(clay_engine_cli
    src/cli/clay_engine_main.cpp
)
target_link_libraries(clay_engine_cli PRIVATE clay_pnr_core)
clay_apply_cxx_flags(clay_engine_cli)

# ---------------------------------------------------------------------------
# 2. Pure C++20 Native Unit Tests (zero Python include/link dependency)
# ---------------------------------------------------------------------------
if(CLAY_PNR_BUILD_TESTS)
    include(CTest)
    enable_testing()
    add_executable(clay_pnr_cpp_tests
        tests/test_main.cpp
        tests/test_core_geometry.cpp
        tests/test_core_polygon_gds.cpp
        tests/test_place_kernels.cpp
        tests/test_pathfinder_core.cpp
        tests/test_gridless_core.cpp
        tests/test_engine_phase23.cpp
        tests/test_ir_and_synth.cpp
    )
    target_link_libraries(clay_pnr_cpp_tests PRIVATE clay_pnr_core)
    clay_apply_cxx_flags(clay_pnr_cpp_tests)
    add_test(NAME clay_pnr_cpp_tests COMMAND clay_pnr_cpp_tests)
endif()

# ---------------------------------------------------------------------------
# 3. Thin CPython Extension Modules -> clay/_native/
# ---------------------------------------------------------------------------
if(CLAY_PNR_BUILD_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

    function(add_clay_py_ext target_name)
        Python3_add_library(${target_name} MODULE ${ARGN})
        target_link_libraries(${target_name} PRIVATE clay_pnr_core)
        target_include_directories(${target_name} PRIVATE "${CLAY_PNR_INCLUDE_DIR}")
        clay_apply_cxx_flags(${target_name})
        if(NOT MSVC)
            target_compile_options(${target_name} PRIVATE -Wno-cast-function-type)
        endif()
        set_target_properties(${target_name} PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${CLAY_NATIVE_OUT_DIR}"
            RUNTIME_OUTPUT_DIRECTORY "${CLAY_NATIVE_OUT_DIR}"
        )
        foreach(cfg DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
            set_target_properties(${target_name} PROPERTIES
                LIBRARY_OUTPUT_DIRECTORY_${cfg} "${CLAY_NATIVE_OUT_DIR}"
                RUNTIME_OUTPUT_DIRECTORY_${cfg} "${CLAY_NATIVE_OUT_DIR}"
            )
        endforeach()
    endfunction()

    add_clay_py_ext(_engine_cpp
        src/engine/py_engine_module.cpp
    )
    add_clay_py_ext(_place_cpp
        src/place/py_place_module.cpp
    )
    add_clay_py_ext(_pathfinder_cpp
        src/route/py_pathfinder_module.cpp
    )
    add_clay_py_ext(_gridless_cpp
        src/route/gridless_spatial_py.cpp
        src/route/gridless_search_py.cpp
        src/route/py_gridless_module.cpp
    )
endif()
