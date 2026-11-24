"""Clay Phase 1 layout-as-co-variable subsystem (pure Python).

Standalone Python implementation; a parallel Rust crate at
``crates/clay_layout/`` is being developed but is not wired into Python
yet. This package owns the canonical Phase 1 semantics until the Rust
bridge lands.
"""

from .mismatch import diff_pair_mismatch_mv, sigma_delta_vth_mv
from .pdk_dims import DeviceType, Gf180mcu, PdkDims, Sky130
from .sizing_hook import DefaultMismatchCostHook, LayoutCostHook
from .symbolic_layout import SymbolicLayout

__all__ = [
    "SymbolicLayout",
    "PdkDims",
    "Sky130",
    "Gf180mcu",
    "DeviceType",
    "sigma_delta_vth_mv",
    "diff_pair_mismatch_mv",
    "LayoutCostHook",
    "DefaultMismatchCostHook",
]

from .annotations import (
    LayoutAnnotation, Matching, GuardRing, WellIsolated,
    Symmetric, Interdigitated, Adjacent, Bulk, Tap, Orient, DenseContacts, NativeBodyTap, Fold, ResistorSegments, EqualSize, Fix, AnnotatedCell,
    WellType, MatchingKind, SymmetryAxis, InterdigitationPattern, Orientation,
)
from .spatial_validate import (
    validate, SpatialTypeError,
    MatchingFingerCountMismatch, SymmetricOddDeviceCount,
    WellTypeConflict, UnknownDeviceReference,
    InterdigitatedZeroFingerPairs, AdjacentNonPositiveSeparation,
)

# P2 reconciler: frozen LayoutPlan + builder (consumed by P3 placer / P4 router).
from .layout_plan import (
    LayoutPlan,
    layout_plan_from_cell_def,
    net_terminals_from_cell_def,
)

# Phase 3a backend (pure-Python mirror of crates/clay_layout Rust backend).
from .backend import (
    Point, Polygon, LayerPolygon, NamedCell, ParameterizedLibrary,
    emit_ascii_gds, parse_ascii_gds, AsciiGdsParseError,
    lower_mos_finger_geometry, lower_annotated_cell_to_parameterized,
    LayerMap, LogicalLayer, GdsLayer,
    Sky130LayerMap, Gf180mcuLayerMap,
    SKY130_FIN_PITCH_NM, GF180_FIN_PITCH_NM,
    DeviceSizing,
)
# Phase 5 DRC-aware lowering (uses max(fin_pitch, min_poly_space) for spacing).
from .backend import (
    lower_mos_finger_geometry_drc_aware,
    lower_annotated_cell_drc_aware,
)

# Phase 5.1 PDK min poly-poly DRC space lookup helpers.
from .backend import (
    SKY130_MIN_POLY_SPACE_NM,
    GF180_MIN_POLY_SPACE_NM,
    min_poly_space_nm_for_pdk_layers,
)
# Phase 5.2 PDK nwell-diff enclosure lookup helpers.
from .backend import (
    SKY130_NWELL_DIFF_ENCLOSURE_NM,
    GF180_NWELL_DIFF_ENCLOSURE_NM,
    nwell_diff_enclosure_nm_for_pdk_layers,
)
# Phase 5.3 DRC-complete wrapper (auto-lookup + enclosure post-processing).
from .backend import lower_annotated_cell_drc_complete

from .binary_gds_subprocess import (
    emit_binary_gds_via_subprocess,
    BinaryGdsSubprocessError,
    cargo_binary_available,
)

from .drc import (
    DrcRule, DrcRuleKind, DrcReport, DrcViolation,
    SKY130_MIN_DRC_RULES, GF180MCU_MIN_DRC_RULES,
    run_drc, KLAYOUT_AVAILABLE,
)

# Phase 4 cosizing (joint W x L x finger_count Pareto sweep).
from .cosizing import CosizingPoint, joint_sweep, pareto_optimal, select_optimal

# Phase 6 cell template library.
from .cell_templates import (
    build_diff_pair,
    build_current_mirror,
    build_cascoded_current_mirror,
    build_5t_ota,
    build_folded_cascode,
    build_telescopic_cascode,
    build_two_stage_miller,
    build_bandgap_reference,
)

# Phase 10 M3 routing primitives + via23.
from .routing_m3 import (
    route_m3_segment,
    route_m3_manhattan,
    route_m3_multibend,
    insert_via23,
    route_m2_to_m3,
    metal3_layer_for_pdk_layers,
    via23_layer_for_pdk_layers,
)

# Phase 4.x continuous relaxation of finger_count for gradient sizing.
from .continuous_relax import (
    sigma_delta_vth_continuous,
    grad_sigma_wrt_n_finger,
    grad_sigma_wrt_w,
    grad_sigma_wrt_l,
    ContinuousSizingPoint,
    gradient_descent_minimize_sigma,
    round_to_integer_finger,
)

# Phase 11 netlist back-annotation.
from .netlist_annotate import (
    NetlistDeviceParams,
    cosizing_point_to_netlist_params,
    format_spice_param_string,
    back_annotate_pareto_choice,
    batch_back_annotate,
    format_spice_device_lines,
)

# Phase 12 LVS topology match.
from .lvs import (
    LvsDeviceExpectation,
    LvsLayerCountMismatch,
    LvsResult,
    check_topology_match,
)

# Phase 13 mixed-cell composer.
from .composer import PlacedSubcell, compose_cells

# Phase 4.y Adam optimizer.
from .continuous_relax_adam import AdamSizingResult, adam_minimize_sigma

# Phase 8d routing congestion estimator.
from .congestion import ChannelOccupancy, CongestionReport, estimate_congestion

# Phase 14 cell template benchmarking suite.
from .template_benchmark import (
    TemplateBenchmark,
    BenchmarkReport,
    benchmark_template,
    benchmark_all_templates,
)

# Phase 15 ASCII layout art renderer.
from .ascii_render import render_ascii

# Phase 16 SPICE netlist parser stub.
from .spice_parse import (
    ParsedDevice,
    parse_spice_device_line,
    parse_spice_netlist,
    to_device_sizing,
)

# Phase 17 layout JSON serde round-trip.
from .json_serde import (
    serialize_symbolic_layout, deserialize_symbolic_layout,
    serialize_annotated_cell, deserialize_annotated_cell,
    serialize_parameterized_library, deserialize_parameterized_library,
    serialize_device_sizing, deserialize_device_sizing,
    to_json, from_json,
)

# Phase 22 Monte Carlo mismatch sampler.
from .monte_carlo import (
    MonteCarloMismatchResult,
    sample_delta_vth_mv,
    sample_diff_pair_offset,
    estimate_3sigma_yield,
)

# Phase 23 cell template ASCII visualization.
from .template_visualize import visualize_template, visualize_all_templates

# Phase 24 SVG export.
from .svg_export import (
    LayerStyle,
    DEFAULT_SKY130_STYLES,
    render_svg,
    write_svg_to_file,
)

# Phase 26 template DRC verification harness.
from .template_drc import (
    TemplateDrcResult,
    TemplateDrcReport,
    drc_check_template,
    drc_check_all_templates,
)

# Phase 27 power grid template.
from .power_grid import (
    PowerGridConfig,
    generate_power_grid,
    count_rails,
    estimate_grid_coverage_pct,
)

# Phase 7.5 edge coupling parasitic.
from .edge_coupling import (
    EdgeCouplingParams,
    EdgeCouplingReport,
    DEFAULT_EDGE_PARAMS,
    estimate_edge_coupling,
)

# Phase 28 gm/Id LUT device characterization.
from .gm_id_lut import (
    GmIdEntry,
    SKY130_NMOS_LUT_L05,
    lookup_by_gm_id,
    interpolate_id_per_w,
    size_w_for_target_id,
)

# Phase 30 ParameterizedLibrary diff utility.
from .lib_diff import CellDiff, LibraryDiff, diff_libraries, summarize_diff

# Phase 32 benchmark report renderers.
from .benchmark_report import render_markdown_table, render_csv, render_summary

# Phase 34 batch GDS generator.
from .batch_gds import (
    BatchGdsResult,
    generate_all_template_gds,
    generate_template_gds,
)

# Phase 35 hierarchical cell reference.
from .hierarchical import (
    CellInstance,
    HierarchicalLibrary,
    flatten_hierarchical,
    detect_cycles,
    add_instance,
)

# Phase 36 combined layout cost predictor.
from .cost_predictor import (
    CostWeights,
    LayoutCostComponents,
    LayoutCostScore,
    compute_layout_cost,
    compare_layouts,
    rank_layouts,
)

# Phase 39 router escape pattern library.
from .escape_patterns import (
    EscapePattern,
    diff_pair_input_escape,
    current_mirror_input_escape,
    power_grid_tap_escape,
)

# Phase 40 layout statistics dashboard.
from .stats_dashboard import (
    LayoutStats,
    compute_layout_stats,
    render_stats_text,
    render_stats_markdown,
)

# Phase 41 layout assertion language.
from .assertions import (
    AssertionError_,
    AssertionReport,
    assert_device_count,
    assert_min_finger_count,
    assert_annotation_present,
    assert_matching_pair,
    run_assertions,
)

# Phase 42 cell template parametric sweep.
from .template_sweep import SweepResult, sweep_template, filter_clean, summarize_sweep

# Phase 43 lowering cache.
from .lowering_cache import LoweringCache

# Phase 44 cell template comparison.
from .template_compare import (
    ComparisonRow,
    TemplateComparison,
    compare_templates,
    render_comparison_text,
)

# Phase 45 GDS metadata reader.
from .gds_metadata import GdsMetadata, read_gds_metadata

# Phase 48 layout regression baseline.
from .regression_baseline import (
    RegressionResult,
    save_baseline,
    load_baseline,
    compare_to_baseline,
    regression_check_all,
)

# Phase 50 LLM dataset generator.
from .dataset_gen import (
    DatasetSample,
    generate_diff_pair_samples,
    generate_current_mirror_samples,
    generate_all_template_samples,
    write_dataset_jsonl,
)

# Phase 52 cell template introspection.
from .template_introspect import (
    TemplateParam,
    TemplateSignature,
    introspect_template,
    list_all_templates,
    list_required_params,
    list_optional_params,
    get_template_docstring,
)

# Phase 53 cell template stress test harness.
from .stress_test import (
    StressTestResult,
    run_stress_test_for,
    run_stress_test_all,
    summarize_stress,
)

# Phase 54 JSON schema validator.
from .json_schema_validate import (
    SchemaValidationError,
    SchemaValidationResult,
    validate_symbolic_layout_schema,
    validate_annotated_cell_schema,
    validate_parameterized_library_schema,
    validate_device_sizing_schema,
)

# Phase 56 ParameterizedLibrary merger.
from .lib_merge import MergeConflict, merge_libraries, find_collisions

# Phase 58 grid snapper.
from .grid_snap import (
    SnapStats,
    snap_library_to_grid,
    snap_point_to_grid,
    check_polygon_on_grid,
)

# Phase 49 spec doc generator.
from .spec_doc_gen import (
    generate_template_spec_doc,
    generate_all_template_docs,
    write_template_docs,
)

# Phase 55 GDS topology check.
from .topology_check import TopologyIssue, TopologyReport, check_topology

# Phase 57 dimension cache.
from .dimension_cache import DimensionCache

# Phase 59 PDK cross-validator.
from .pdk_cross_validate import (
    PdkValidationResult,
    CrossPdkReport,
    validate_on_sky130,
    validate_on_gf180mcu,
    cross_pdk_validate,
)

# Phase 60 cell port definitions.
from .cell_ports import CellPort, CellPortSet, get_ports_for, list_ports

# Phase 64 A* router path finder.
from .a_star_router import Obstacle, RouteResult, find_path

# Phase 65 cell area estimator.
from .area_estimator import (
    CellAreaEstimate,
    estimate_cell_area,
    estimate_template_area,
    compare_template_areas,
)

# Phase 67 HTML report.
from .html_report import render_html_report, write_html_report, render_minimal_report

# Phase 68 closed-loop sizing optimizer.
from .sizing_optimizer import (
    SizingConstraints,
    SizingOptimumResult,
    optimize_sizing,
)

# Phase 70 PDK migration tool.
from .pdk_migration import MigrationResult, migrate_sizing

# Phase 73 cell snapshot diff.
from .snapshot_diff import (
    DeviceDiff,
    AnnotationDiff,
    CellSnapshotDiff,
    diff_cells,
    summarize_diff as summarize_snapshot_diff,
)

# Phase 74 LLM few-shot prompt generator.
from .llm_prompt_gen import (
    FewShotExample,
    FewShotPromptBundle,
    build_few_shot_examples,
    build_prompt,
    format_prompt_text,
)

# Phase 18 LEF library exporter.
from .lef_export import (
    LefExportConfig,
    cell_bounding_box_um,
    export_cell_to_lef,
    export_library_to_lef,
    write_library_to_lef,
)

# Phase 47 ParameterizedLibrary canonical hash.
from .library_hash import (
    LibraryHashResult,
    canonical_json,
    library_hash,
    compare_hashes,
)

# Phase 51 sizing interpolation.
from .sizing_interpolate import (
    InterpolatedPoint,
    interpolate_sizing,
    interpolate_symbolic_layout,
)

# Phase 61 matching-pair metric.
from .matching_metric import (
    MatchingPairSignature,
    MatchingMetricReport,
    extract_matching_pairs,
    compute_l1_distance,
)

# Phase 19 routing-layer pitch ruleset.
from .layer_pitch import (
    LayerPitchRule,
    SKY130_LAYER_PITCH,
    GF180MCU_LAYER_PITCH,
    get_layer_pitch_rule,
    list_layers_for_pdk,
    compute_track_count,
)

# Phase 25 DRC margin computation.
from .drc_margin import LayerMargin, DrcMarginReport, compute_drc_margin

# Phase 46 layout coverage report.
from .coverage import (
    CoverageReport,
    compute_coverage,
    render_coverage_text,
)

# Phase 72 cell template extension / clone / merge.
from .template_extend import (
    DeviceAddition,
    clone_cell,
    extend_cell,
    merge_devices,
)

# Phase 20 cell template variant catalog.
from .variant_catalog import (
    VariantSpec,
    VariantCatalog,
    build_variant_label,
    generate_variant_catalog,
    render_catalog_csv,
)

# Phase 29 routing congestion histogram.
from .congestion_hist import (
    HistogramBin,
    CongestionHistogram,
    compute_congestion_histogram,
    render_histogram_text,
)

# Phase 62 template equivalence checker.
from .template_equivalence import (
    EquivalenceResult,
    signature_of_device,
    signature_of_cell,
    annotation_signature_multiset,
    cells_equivalent,
)

# Phase 75 human-readable text dump.
from .text_dump import TextDumpConfig, dump_library_to_text, dump_cell_to_text

# Phase 21 per-cell statistics CSV.
from .stats_csv import (
    CellStats,
    compute_cell_stats,
    compute_library_stats,
    render_stats_csv,
    write_stats_csv,
)

# Phase 31 routed wire JSON serde.
from .wire_serde import (
    serialize_wire_segment,
    deserialize_wire_segment,
    serialize_wire_list,
    deserialize_wire_list,
    write_wires_json,
    read_wires_json,
)

# Phase 33 cell template version metadata.
from .template_version import (
    TemplateVersion,
    compute_cell_content_hash,
    stamp_cell,
    parse_semver,
    is_compatible,
)

# Phase 38 ASCII art alignment helpers.
from .ascii_align import (
    measure,
    pad_to_size,
    align_horizontal,
    align_vertical,
    add_border,
    label_block,
)

# Phase 8a M1 routing primitive.
from .routing import (
    WireSegment,
    route_m1_segment,
    route_m1_manhattan,
)

# Phase 8b multi-bend M1 routing + via12 + M1-to-M2 cross-layer.
from .routing_advanced import (
    route_m1_multibend,
    insert_via12,
    route_m1_to_m2,
    via12_layer_for_pdk_layers,
    metal2_layer_for_pdk_layers,
)

# Phase 8c M2 routing primitives (mirror of Phase 8a/8b for M2).
from .routing_m2 import (
    route_m2_segment,
    route_m2_manhattan,
    route_m2_multibend,
)

# Phase 9 net-aware parasitic (per-net partition of Phase 7 parasitic).
from .parasitic_net_aware import (
    NetAssignment,
    NetParasiticEntry,
    NetParasiticReport,
    extract_parasitic_net_aware,
)

# Phase 7.3 parasitic-aware sizing cost hook.
from .parasitic_sizing_hook import ParasiticAwareMismatchCostHook

# Phase 7.4 gm rebalance helpers (sqrt scaling when finger_count changes).
from .gm_rebalance import (
    GmRebalanceMode,
    GmRebalanceResult,
    rebalance_for_finger_change,
    rebalance_strict,
    rebalance_sqrt,
)

# Phase 7 parasitic-aware extraction stub.
from .parasitic import (
    LayerParasiticParams,
    LayerParasiticEntry,
    CellParasiticReport,
    LibraryParasiticReport,
    SKY130_PARASITIC,
    GF180MCU_PARASITIC,
    parasitic_params_for_pdk_layers,
    polygon_area_um2,
    polygon_perimeter_um,
    extract_parasitic,
)

# Phase 7.2 parasitic-aware Pareto sweep.
from .cosizing_parasitic import (
    CosizingPointWithParasitic,
    joint_sweep_with_parasitic,
    pareto_optimal_3obj,
    select_optimal_3obj,
)
