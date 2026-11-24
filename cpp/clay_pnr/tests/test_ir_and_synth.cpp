// Pure C++20 native unit tests for Phase 3B (BandedPlacer) and
// Phase 4A (.clay DSL parser + standalone zero-Python GDSII synthesis).
// Strictly free of <Python.h>.

#include "test_framework.hpp"

#include "clay_pnr/engine/standalone_synth.hpp"
#include "clay_pnr/gds/gds_stream.hpp"
#include "clay_pnr/ir/clay_ir.hpp"
#include "clay_pnr/place/banded_placer.hpp"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

using namespace clay_pnr::core;
using namespace clay_pnr::engine;
using namespace clay_pnr::gds;
using namespace clay_pnr::ir;
using namespace clay_pnr::place;

CLAY_TEST(EnginePhase34, BandedPlacerBandsOrdersKeepoutsAndAspect) {
    BandedProblem prob{};
    prob.gap_nm = 500;
    prob.grid_nm = 5;
    prob.w_cap_nm = 40000;
    prob.h_bound_nm = 40000;
    prob.enforce_band_order = true;
    prob.objective_area = true;
    prob.dense_2d = true;
    prob.aspect_obligation = true;
    prob.aspect_lower = 500000;    // W/H >= 0.5
    prob.aspect_upper = 2500000;   // W/H <= 2.5
    prob.aspect_scale = 1000000;
    prob.max_evals = 1200;

    prob.blocks = {
        BandedBlock{"MN1", 6000, 4000, 0, false, true, false, false, false, 1000, false},
        BandedBlock{"MN2", 5000, 4000, 0, false, false, false, false, false, 1000, false},
        BandedBlock{"R1",  7000, 2500, 1, false, false, false, false, false, 1000, false},
        BandedBlock{"C1",  6000, 6000, 1, false, false, false, false, false, 1000, false},
        BandedBlock{"MP1", 8000, 4500, 2, false, false, false, true, false, 1000, false},
        BandedBlock{"MP2", 7000, 4500, 2, false, false, true, false, true, 1000, false},
        BandedBlock{"LAT", 3500, 9000, 0, true, false, false, false, false, 1000, false},
    };
    prob.order_pairs = {{0, 1}, {4, 5}};
    prob.pair_clearances = {{2, 3, 1200}};

    const auto sol = solve_banded_area(prob);
    CLAY_CHECK(sol.status == BandedStatus::Optimal || sol.status == BandedStatus::Feasible);
    std::string err;
    CLAY_CHECK(verify_banded_solution(prob, sol.xs, sol.ys, &err));
    CLAY_CHECK_EQ(sol.xs[0], 0);  // left_edge anchor
    CLAY_CHECK_EQ(sol.xs[5] + prob.blocks[5].w, sol.W);  // right_edge anchor
    CLAY_CHECK(sol.xs[0] + prob.blocks[0].w + 500 <= sol.xs[1]);
    CLAY_CHECK(sol.xs[4] + prob.blocks[4].w + 500 <= sol.xs[5]);
    CLAY_CHECK(sol.ys[0] >= 1000);
    // Band stacking: band 0 below band 1 below band 2
    CLAY_CHECK(sol.ys[0] + prob.blocks[0].h + 500 <= sol.ys[2]);
    CLAY_CHECK(sol.ys[3] + prob.blocks[3].h + 500 <= sol.ys[4]);

    // Determinism check
    const auto sol2 = solve_banded_area(prob);
    CLAY_CHECK_EQ(sol.W, sol2.W);
    CLAY_CHECK_EQ(sol.H, sol2.H);
    CLAY_CHECK_EQ(sol.xs.size(), sol2.xs.size());
    for (size_t i = 0; i < sol.xs.size(); ++i) {
        CLAY_CHECK_EQ(sol.xs[i], sol2.xs[i]);
        CLAY_CHECK_EQ(sol.ys[i], sol2.ys[i]);
    }
}

CLAY_TEST(EnginePhase34, ClayParserUnitsGrammarAndRealSyntax) {
    // SI unit parity with clay/parser_modules/_constants.py
    double mult = 0.0;
    CLAY_CHECK(resolve_si_unit("u", mult) && std::fabs(mult - 1e-6) < 1e-18);
    CLAY_CHECK(resolve_si_unit("n", mult) && std::fabs(mult - 1e-9) < 1e-21);
    CLAY_CHECK(resolve_si_unit("p", mult) && std::fabs(mult - 1e-12) < 1e-24);
    CLAY_CHECK(resolve_si_unit("f", mult) && std::fabs(mult - 1e-15) < 1e-27);
    CLAY_CHECK(resolve_si_unit("k", mult) && std::fabs(mult - 1e3) < 1e-9);
    CLAY_CHECK(resolve_si_unit("m", mult) && std::fabs(mult - 1.0) < 1e-12);
    CLAY_CHECK(resolve_si_unit("V/us", mult) && std::fabs(mult - 1e6) < 1e-6);
    CLAY_CHECK(resolve_si_unit("mA/V", mult) && std::fabs(mult - 1e-3) < 1e-15);
    CLAY_CHECK(!resolve_si_unit("V/us/K", mult));
    CLAY_CHECK(!resolve_si_unit("nonsense", mult));

    const std::string src = R"(
// Sample real-style .clay design
package test.ota;
pdk "gf180mcu";

cell MiniOta(vip: input, vin: input, vout: output, vdd: power_p, vss: power_n) {
  topology {
    device MP1 : pmos {
      drain = d1; gate = vip; source = tail; body = vdd;
      W = 8u; L = 0.5u; M = 2;
    }
    device MP2 : pmos {
      drain = d2; gate = vin; source = tail; body = vdd;
      W = 8u; L = 0.5u; M = 2;
    }
    device MN1 : nmos {
      drain = d1; gate = d1; source = vss; body = vss;
      W = 4u; L = 1u; M = 2;
    }
    device MN2 : nmos {
      drain = d2; gate = d1; source = vss; body = vss;
      W = 4u; L = 1u; M = 2;
    }
    res RZ {
      p = d2; n = ncc;
      value = 1.5k;
      model = ppolyf_u;
    }
    cap CC {
      p = ncc; n = vout;
      value = 800f;
      model = mim;
      M = 2;
    }
    functional_group diff_pair {
      role = input_pair,
      devices = (MP1, MP2),
      intent = "differential pair"
    }
  }
  sizing {
    supply_v = 3.3;
    derived_w = 2u * 3 + 1u;
  }
  spec {
    gain_dc : real >= 60 dB;
    slew : real >= 5 V/us with_margin(10%);
  }
  corners { tt; ss; ff; }
  temps { -40C; 27; 125C; }
}

layout_intent MiniOta {
  @objective(area, minimize);
  @band(top=vdd, bottom=vss, center=passives);
  @order(MP1, MP2, left_of);
  @place_group(in_grp, devices=(MP1, MP2));
  @shield(vip -> vout, vss);
}

annotate MiniOta {
  MP1: @matching(MP2);
  @cell: @symmetric(vertical);
}

flow MiniOta {
  pdk "gf180mcu";
  corners(tt, ss, ff);
  steps(sim, layout, drc, lvs);
}
)";

    const ParseResult pr = parse_clay_source(src);
    CLAY_CHECK(pr.ok);
    CLAY_CHECK_EQ(pr.file.package_name, std::string("test.ota"));
    CLAY_CHECK_EQ(pr.file.pdk, std::string("gf180mcu"));
    CLAY_CHECK_EQ(pr.file.cells.size(), static_cast<size_t>(1));
    const auto& cell = pr.file.cells[0];
    CLAY_CHECK_EQ(cell.name, std::string("MiniOta"));
    CLAY_CHECK_EQ(cell.ports.size(), static_cast<size_t>(5));
    CLAY_CHECK_EQ(cell.devices.size(), static_cast<size_t>(6));
    CLAY_CHECK_EQ(cell.devices[0].pin_or("d"), std::string("d1"));
    CLAY_CHECK_EQ(cell.devices[0].pin_or("g"), std::string("vip"));
    CLAY_CHECK(std::fabs(cell.devices[0].param_or("W") - 8e-6) < 1e-12);
    CLAY_CHECK(std::fabs(cell.devices[4].param_or("value") - 1500.0) < 1e-9);
    CLAY_CHECK_EQ(cell.devices[4].str_param_or("model"), std::string("ppolyf_u"));
    CLAY_CHECK(std::fabs(cell.devices[5].param_or("value") - 800e-15) < 1e-24);
    CLAY_CHECK_EQ(cell.functional_groups.size(), static_cast<size_t>(1));
    CLAY_CHECK_EQ(cell.functional_groups[0].devices.size(), static_cast<size_t>(2));
    CLAY_CHECK_EQ(cell.corners.size(), static_cast<size_t>(3));
    CLAY_CHECK_EQ(cell.temps.size(), static_cast<size_t>(3));
    CLAY_CHECK(std::fabs(cell.temps[0] - (-40.0)) < 1e-9);
    CLAY_CHECK_EQ(cell.sizing.size(), static_cast<size_t>(2));
    CLAY_CHECK(std::fabs(cell.sizing[1].second - 7e-6) < 1e-12);

    const auto* li = pr.file.intent_for_cell("MiniOta");
    CLAY_CHECK(li != nullptr);
    CLAY_CHECK_EQ(li->entries.size(), static_cast<size_t>(5));
    CLAY_CHECK_EQ(li->entries[4].kind, std::string("shield"));
    CLAY_CHECK_EQ(li->entries[4].positional.size(), static_cast<size_t>(3));

    const auto* ab = pr.file.annotations_for_cell("MiniOta");
    CLAY_CHECK(ab != nullptr);
    CLAY_CHECK_EQ(ab->entries.size(), static_cast<size_t>(2));
    CLAY_CHECK_EQ(ab->entries[1].target, std::string("__cell__"));

    CLAY_CHECK_EQ(pr.file.flows.size(), static_cast<size_t>(1));
    CLAY_CHECK_EQ(pr.file.flows[0].steps.size(), static_cast<size_t>(4));

    // Phase 5 IR spans, imports, cell aliases, and named top-level declarations
    CLAY_CHECK(cell.span.line >= 1 && cell.span.col >= 1);
    CLAY_CHECK(cell.devices[0].span.line > cell.span.line);
    CLAY_CHECK(li->span.line > cell.span.line);
    CLAY_CHECK(li->entries[0].span.line > li->span.line);

    const std::string ext_src = R"(
package test.ext;
import "lib/base.clay" as base;
import common.blocks;
pdk "gf180mcu";
cell Top = base.Core;
policy RouteFast { router = "gridless"; }
quality StrictQ { drc = 0; }
verify CheckLvs { tool = "netgen"; }
fn helper(x) { return x; }
)";
    const ParseResult pr2 = parse_clay_source(ext_src);
    CLAY_CHECK(pr2.ok);
    CLAY_CHECK_EQ(pr2.file.import_decls.size(), static_cast<size_t>(2));
    CLAY_CHECK_EQ(pr2.file.import_decls[0].path, std::string("lib/base.clay"));
    CLAY_CHECK_EQ(pr2.file.import_decls[0].alias, std::string("base"));
    CLAY_CHECK(pr2.file.import_decls[0].span.line >= 1);
    CLAY_CHECK_EQ(pr2.file.cell_aliases.size(), static_cast<size_t>(1));
    CLAY_CHECK_EQ(pr2.file.cell_aliases[0].name, std::string("Top"));
    CLAY_CHECK_EQ(pr2.file.cell_aliases[0].target, std::string("base.Core"));
    CLAY_CHECK_EQ(pr2.file.policy_names.size(), static_cast<size_t>(1));
    CLAY_CHECK_EQ(pr2.file.policy_names[0], std::string("RouteFast"));
    CLAY_CHECK_EQ(pr2.file.quality_names.size(), static_cast<size_t>(1));
    CLAY_CHECK_EQ(pr2.file.quality_names[0], std::string("StrictQ"));
    CLAY_CHECK_EQ(pr2.file.verify_names.size(), static_cast<size_t>(1));
    CLAY_CHECK_EQ(pr2.file.verify_names[0], std::string("CheckLvs"));
    CLAY_CHECK_EQ(pr2.file.fn_names.size(), static_cast<size_t>(1));
    CLAY_CHECK_EQ(pr2.file.fn_names[0], std::string("helper"));
}

CLAY_TEST(EnginePhase34, StandaloneZeroPythonSynthGF180AndSky130) {
    const std::string src = R"(
package demo.diffamp;
pdk "gf180mcu";

cell DiffAmp(vip: input, vin: input, vout: output, vdd: power_p, vss: power_n) {
  topology {
    device M1 : nmos { drain = d1; gate = vip; source = tail; body = vss; W = 4u; L = 0.5u; M = 2; }
    device M2 : nmos { drain = vout; gate = vin; source = tail; body = vss; W = 4u; L = 0.5u; M = 2; }
    device M3 : pmos { drain = d1; gate = d1; source = vdd; body = vdd; W = 6u; L = 0.5u; M = 2; }
    device M4 : pmos { drain = vout; gate = d1; source = vdd; body = vdd; W = 6u; L = 0.5u; M = 2; }
    device M5 : nmos { drain = tail; gate = vbias; source = vss; body = vss; W = 8u; L = 1u; M = 2; }
    res R1 { p = vout; n = mid; value = 2k; }
    cap C1 { p = mid; n = vss; value = 500f; }
  }
}
layout_intent DiffAmp {
  @order(M1, M2, left_of);
  @order(M3, M4, left_of);
}
)";

    for (const std::string& pdk_id : {std::string("gf180mcu"), std::string("sky130")}) {
        SynthOptions opts{};
        opts.pdk_override = pdk_id;
        const auto res = synthesize_clay_source(src, opts);
        CLAY_CHECK(res.ok);
        CLAY_CHECK_EQ(res.cell_name, std::string("DiffAmp"));
        CLAY_CHECK_EQ(res.device_count, static_cast<size_t>(7));
        CLAY_CHECK(res.routed_net_count >= static_cast<size_t>(4));
        CLAY_CHECK(res.polygon_count > static_cast<size_t>(20));
        CLAY_CHECK(res.die_bbox_nm.width() > 0 && res.die_bbox_nm.height() > 0);

        // Verify zero macro overlaps
        for (size_t i = 0; i < res.macros.size(); ++i) {
            for (size_t j = i + 1; j < res.macros.size(); ++j) {
                const Box64& a = res.macros[i].bbox_nm;
                const Box64& b = res.macros[j].bbox_nm;
                const bool disjoint = (a.x1 <= b.x0) || (b.x1 <= a.x0) ||
                                      (a.y1 <= b.y0) || (b.y1 <= a.y0);
                CLAY_CHECK(disjoint);
            }
        }

        // Verify binary GDSII round-trip + 0 off-grid vertices
        const auto bytes = write_gds_bytes(res.gds_lib);
        CLAY_CHECK(bytes.size() > 256u);
        GdsLibrary reloaded;
        std::string gerr;
        CLAY_CHECK(read_gds_bytes(bytes.data(), bytes.size(), reloaded, &gerr));
        const auto sums = summarize_gds_layers(reloaded, "DiffAmp");
        CLAY_CHECK(sums.size() >= static_cast<size_t>(6));
        for (const auto& cell : reloaded.cells) {
            for (const auto& poly : cell.boundaries) {
                for (const auto& pt : poly.xy) {
                    CLAY_CHECK_EQ(static_cast<int64_t>(pt.x) % res.grid_nm, 0);
                    CLAY_CHECK_EQ(static_cast<int64_t>(pt.y) % res.grid_nm, 0);
                }
            }
        }
    }
}
