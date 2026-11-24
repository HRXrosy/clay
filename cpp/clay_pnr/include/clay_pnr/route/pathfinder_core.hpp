#pragma once

// Pure C++20 3D grid PathFinder & Maze routing core and axis-aligned rectangle clipper.
// Strictly standalone C++20 with zero CPython runtime dependencies.
//
// Core design:
//   * Contiguous flat 3D arrays indexed by idx = gi * (nx * ny) + iy * nx + ix
//   * Per-net compact uint64_t bitsets for blocked[] and via_blocked[]
//   * O(1) generation-stamped A* reset (zero memset across 2M+ cells)
//   * Exact binary min-heap tie-breaking:
//       - undirected: (f, -g, g, ix, iy, gi)
//       - directed:   (f, -g, g, ix, iy, gi, run)
//   * Periodic cooperative CheckpointHook polling every 4096 expansions by default
//   * Batch obstacle rasterization (incl. MIM clearance & via-infeasible cells),
//     via-pad keepout bookkeeping, overuse/pad-contention round finalization,
//     and Maze committed-halo support.

#include "clay_pnr/core/geometry.hpp"
#include "clay_pnr/core/status.hpp"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace clay_pnr::route {

constexpr double kWrongDirPenalty = 0.5;
constexpr double kViaPenalty = 3.0;
constexpr double kTrunkBelowPenalty = 0.3;
constexpr int kDirRunFree = 2;
constexpr double kDirRunPenalty = 2.5;
constexpr int kDirRunCap = kDirRunFree + 1;     // 3
constexpr int kDirRunStates = kDirRunCap + 1;   // 4
constexpr double kNmPerUm = 1000.0;
constexpr int64_t kDefaultCheckpointInterval = 4096;

inline void bitset_set(std::vector<uint64_t>& bits, int32_t idx) noexcept {
    bits[static_cast<size_t>(idx) >> 6] |= (1ULL << (static_cast<uint32_t>(idx) & 63U));
}

inline bool bitset_test(const std::vector<uint64_t>& bits, int32_t idx) noexcept {
    if (bits.empty()) return false;
    return ((bits[static_cast<size_t>(idx) >> 6] >> (static_cast<uint32_t>(idx) & 63U)) & 1ULL) != 0ULL;
}

struct RectUm {
    double x0 = 0.0;
    double y0 = 0.0;
    double x1 = 0.0;
    double y1 = 0.0;
};

struct ObsRect {
    int32_t gi = 0;
    double x0_um = 0.0;
    double y0_um = 0.0;
    double x1_um = 0.0;
    double y1_um = 0.0;
    int64_t bx0_nm = 0;
    int64_t by0_nm = 0;
    int64_t bx1_nm = 0;
    int64_t by1_nm = 0;
    int32_t owner_id = -1;  // -1 = unowned, -2 = foreign net not in net list, >= 0 = net_id
};

struct ViaObsRect {
    int32_t gl = 0;
    double x0_um = 0.0;
    double y0_um = 0.0;
    double x1_um = 0.0;
    double y1_um = 0.0;
    int64_t bx0_nm = 0;
    int64_t by0_nm = 0;
    int64_t bx1_nm = 0;
    int64_t by1_nm = 0;
    int32_t owner_id = -1;
    bool is_cap_plate_via = false;
};

struct MimClearRect {
    int32_t gi = 0;
    int64_t bx0_nm = 0;
    int64_t by0_nm = 0;
    int64_t bx1_nm = 0;
    int64_t by1_nm = 0;
    int64_t clr_nm = 0;
    int32_t owner_id = -1;
};

struct MimForbiddenCutRect {
    int32_t gl = 0;
    int64_t bx0_nm = 0;
    int64_t by0_nm = 0;
    int64_t bx1_nm = 0;
    int64_t by1_nm = 0;
};

struct StencilOff {
    int32_t dx = 0;
    int32_t dy = 0;
    int32_t g = 0;
};

struct Off2D {
    int32_t dx = 0;
    int32_t dy = 0;
};

struct HeapNode {
    double f = 0.0;
    double neg_g = 0.0;
    double g = 0.0;
    int32_t ix = 0;
    int32_t iy = 0;
    int32_t gi = 0;
    int32_t idx = 0;
};

// Strict weak ordering matching tuple comparison (f, -g, g, (ix, iy, gi)).
inline bool heap_node_less(const HeapNode& a, const HeapNode& b) noexcept {
    if (a.f != b.f) return a.f < b.f;
    if (a.neg_g != b.neg_g) return a.neg_g < b.neg_g;
    if (a.g != b.g) return a.g < b.g;
    if (a.ix != b.ix) return a.ix < b.ix;
    if (a.iy != b.iy) return a.iy < b.iy;
    return a.gi < b.gi;
}

struct DirHeapNode {
    double f = 0.0;
    double neg_g = 0.0;
    double g = 0.0;
    int32_t ix = 0;
    int32_t iy = 0;
    int32_t gi = 0;
    int32_t run = 0;
    int32_t state_idx = 0;
};

// Strict weak ordering matching tuple comparison (f, -g, g, (ix, iy, gi), run).
inline bool dir_heap_node_less(const DirHeapNode& a, const DirHeapNode& b) noexcept {
    if (a.f != b.f) return a.f < b.f;
    if (a.neg_g != b.neg_g) return a.neg_g < b.neg_g;
    if (a.g != b.g) return a.g < b.g;
    if (a.ix != b.ix) return a.ix < b.ix;
    if (a.iy != b.iy) return a.iy < b.iy;
    if (a.gi != b.gi) return a.gi < b.gi;
    return a.run < b.run;
}

struct HeapNodeLess {
    bool operator()(const HeapNode& a, const HeapNode& b) const noexcept {
        return heap_node_less(a, b);
    }
};

struct DirHeapNodeLess {
    bool operator()(const DirHeapNode& a, const DirHeapNode& b) const noexcept {
        return dir_heap_node_less(a, b);
    }
};

template <typename Node, typename LessFn>
class BinaryMinHeap {
public:
    void clear() noexcept { data_.clear(); }
    bool empty() const noexcept { return data_.empty(); }
    size_t size() const noexcept { return data_.size(); }
    void reserve(size_t n) { data_.reserve(n); }

    void push(const Node& node) {
        size_t pos = data_.size();
        data_.push_back(node);
        while (pos > 0) {
            const size_t parent = (pos - 1) >> 1;
            if (!less_(data_[pos], data_[parent])) break;
            std::swap(data_[pos], data_[parent]);
            pos = parent;
        }
    }

    Node pop() {
        Node top = data_[0];
        const size_t last = data_.size() - 1;
        if (last > 0) {
            data_[0] = data_[last];
            data_.pop_back();
            const size_t n = data_.size();
            size_t pos = 0;
            while (true) {
                const size_t left = (pos << 1) + 1;
                if (left >= n) break;
                size_t best = left;
                const size_t right = left + 1;
                if (right < n && less_(data_[right], data_[left])) {
                    best = right;
                }
                if (!less_(data_[best], data_[pos])) break;
                std::swap(data_[pos], data_[best]);
                pos = best;
            }
        } else {
            data_.pop_back();
        }
        return top;
    }

private:
    std::vector<Node> data_;
    LessFn less_{};
};

struct NetData {
    int32_t route_w_nm = 0;
    int32_t own_width_floor = 0;
    int32_t trunk_gi = 0;
    uint32_t allowed_mask = 0;
    std::vector<uint64_t> blocked_bits;      // size: (ntotal + 63) / 64
    std::vector<uint64_t> via_blocked_bits;  // size: (nvia_total + 63) / 64, empty if none
    std::vector<uint64_t> neck_bits;         // size: (nxy + 63) / 64
    std::vector<int32_t> unique_pin_cells;
    // Per via-layer stencils for via_cost (exact_diagonal = true)
    std::vector<std::vector<StencilOff>> stencil_trunk;
    std::vector<std::vector<StencilOff>> stencil_neck;
    // Per (gl, side 0/1) keepout offsets for pad_keepout (exact_diagonal=false) and exact_pad_keepout (true)
    // Indexed by [gl * 2 + side], separate for trunk vs neck
    std::vector<std::vector<Off2D>> pk_soft_trunk;
    std::vector<std::vector<Off2D>> pk_soft_neck;
    std::vector<std::vector<Off2D>> pk_exact_trunk;
    std::vector<std::vector<Off2D>> pk_exact_neck;
};

struct ExactPadKeepoutEntry {
    int32_t cell_idx = 0;
    int32_t owner_net_id = 0;
    int32_t vcell_idx = 0;  // gl * nxy + iy * nx + ix
};

struct PadContendedRow {
    int32_t cell_idx = 0;
    int32_t metal_owner_id = -1;
    std::vector<int32_t> foreign_owners;
    std::vector<int32_t> foreign_vcells;
};

struct FinishRoundResult {
    bool converged = true;
    std::vector<int32_t> overused;
    std::vector<PadContendedRow> pad_contended;
};

struct DirPrefEntry {
    bool active = false;
    bool horiz = true;
    double weight = 0.0;
};

class PathFinderGridCore {
public:
    int32_t nx = 0;
    int32_t ny = 0;
    int32_t n_layers = 0;
    int32_t nxy = 0;
    int32_t ntotal = 0;
    int32_t nvia_total = 0;
    int64_t x0_nm = 0;
    int64_t y0_nm = 0;
    int64_t pitch_nm = 1;

    std::vector<uint8_t> pref_horiz;
    std::vector<int32_t> min_width_nm;
    std::vector<int32_t> min_space_nm;
    std::vector<int32_t> via_cut_size_nm;
    std::vector<int32_t> via_enc_below_nm;
    std::vector<int32_t> via_enc_above_nm;
    std::vector<int32_t> via_cut_space_nm;

    // Static obstacles (uploaded once per route() call)
    std::vector<std::vector<ObsRect>> obs_by_gi;
    std::vector<std::vector<ViaObsRect>> via_obs_by_gl;
    std::vector<std::vector<MimClearRect>> mim_clear_by_gi;
    std::vector<std::vector<MimForbiddenCutRect>> mim_forbidden_by_gl;
    bool has_envelope = false;
    int64_t env_x0 = 0;
    int64_t env_y0 = 0;
    int64_t env_x1 = 0;
    int64_t env_y1 = 0;

    // Over-active soft cost
    std::vector<uint64_t> over_active_bits;
    int32_t over_active_gi_floor = 0;
    double over_active_penalty = 0.0;

    // Per-net prepared data
    std::vector<NetData> nets;

    // Shared PathFinder round/history arrays (flat, persistent across rounds)
    std::vector<int32_t> occ;
    std::vector<double> history;
    std::vector<int32_t> pad_keepout_count;
    std::vector<double> via_pad_history;
    std::vector<int32_t> reserved_owner;
    std::vector<int32_t> unrouted_pin_count;
    std::vector<int32_t> metal_owner;
    std::vector<int32_t> committed_owner;  // for MazeRouteEngine

    // Sparse change lists for O(active) round reset
    std::vector<int32_t> touched_occ;
    std::vector<int32_t> touched_pad_keepout;
    std::vector<int32_t> touched_reserved;
    std::vector<int32_t> touched_unrouted_pins;
    std::vector<int32_t> touched_metal_owner;
    std::vector<int32_t> touched_committed;

    // Per-round exact pad keepout entries (compact flat log, processed at finish_round)
    std::vector<ExactPadKeepoutEntry> round_exact_pk;

    // Generation-stamped scratch arrays (zero memset)
    std::vector<uint32_t> astar_stamp;
    std::vector<double> astar_g;
    std::vector<int32_t> astar_parent;
    uint32_t astar_gen = 2;

    // Directed A* scratch arrays (lazily allocated on first directed search)
    std::vector<uint32_t> dir_stamp;
    std::vector<double> dir_g;
    std::vector<int32_t> dir_parent;
    uint32_t dir_gen = 2;

    // Tree / wire / mirror / dedup / dir_runs stamps
    std::vector<uint32_t> tree_stamp;
    uint32_t tree_gen = 1;
    std::vector<uint32_t> wire_stamp;
    uint32_t wire_gen = 1;
    std::vector<uint32_t> mirror_stamp;
    uint32_t mirror_gen = 1;
    std::vector<uint32_t> dedup_stamp;
    uint32_t dedup_gen = 1;
    std::vector<uint32_t> self_pk_stamp;
    uint32_t self_pk_gen = 1;
    std::vector<int32_t> dir_runs_arr;
    std::vector<uint32_t> dir_runs_stamp;
    uint32_t dir_runs_gen = 1;
    std::vector<uint32_t> bfs_stamp;
    uint32_t bfs_gen = 1;

    // Reusable heaps (avoid malloc across searches)
    BinaryMinHeap<HeapNode, HeapNodeLess> heap;
    BinaryMinHeap<DirHeapNode, DirHeapNodeLess> dir_heap;

    bool init(
        int32_t nx_, int32_t ny_, int32_t n_layers_,
        int64_t x0_, int64_t y0_, int64_t pitch_,
        const std::vector<uint8_t>& pref_h,
        const std::vector<int32_t>& min_w,
        const std::vector<int32_t>& min_s,
        const std::vector<int32_t>& v_cut,
        const std::vector<int32_t>& v_eb,
        const std::vector<int32_t>& v_ea,
        const std::vector<int32_t>& v_sp
    );

    inline int32_t cell_idx(int32_t ix, int32_t iy, int32_t gi) const noexcept {
        return gi * nxy + iy * nx + ix;
    }

    inline void unpack_idx(int32_t idx, int32_t& ix, int32_t& iy, int32_t& gi) const noexcept {
        gi = idx / nxy;
        const int32_t rem = idx - gi * nxy;
        iy = rem / nx;
        ix = rem - iy * nx;
    }

    inline void unpack_xy(int32_t idx, int32_t& ix, int32_t& iy) const noexcept {
        const int32_t gi = idx / nxy;
        const int32_t rem = idx - gi * nxy;
        iy = rem / nx;
        ix = rem - iy * nx;
    }

    inline bool in_bounds(int32_t ix, int32_t iy) const noexcept {
        return ix >= 0 && ix < nx && iy >= 0 && iy < ny;
    }

    uint32_t next_astar_gen();
    uint32_t next_dir_gen();
    uint32_t next_gen(std::vector<uint32_t>& arr, uint32_t& gen);

    static std::vector<Off2D> compute_via_keepout_offsets(
        int64_t pitch,
        int64_t pad_half,
        int64_t wire_half,
        int64_t ms,
        bool exact_diagonal
    );

    int64_t via_pad_extra_halo_nm(int32_t gi, int32_t wire_width_nm) const noexcept;

    void raster_rect_into_bits(
        std::vector<uint64_t>& bits,
        int32_t layer_offset,
        double x0, double y0, double x1, double y1,
        int64_t halo_nm
    ) const noexcept;

    void mark_envelope_blocked_bits(
        std::vector<uint64_t>& bits,
        int32_t layer_offset,
        int64_t edge_half_nm
    ) const noexcept;

    void set_over_active(
        const std::vector<RectUm>& rects_um,
        int32_t gi_floor,
        double penalty
    );

    void clear_static_obstacles() noexcept;
    void set_envelope(int64_t ex0, int64_t ey0, int64_t ex1, int64_t ey1) noexcept;
    void add_obs_rect(int32_t gi, double x0, double y0, double x1, double y1, int32_t owner_id);
    void add_via_obs_rect(
        int32_t gl, double x0, double y0, double x1, double y1,
        int32_t owner_id, bool is_cap_plate_via
    );
    void add_mim_clear_rect(
        int32_t gi, int64_t bx0, int64_t by0, int64_t bx1, int64_t by1,
        int64_t clr_nm, int32_t owner_id
    );
    void add_mim_forbidden_cut_rect(
        int32_t gl, int64_t bx0, int64_t by0, int64_t bx1, int64_t by1
    );

    void prepare_net(
        int32_t net_id,
        int32_t route_w_nm,
        int32_t own_width_floor,
        uint32_t allowed_mask,
        const std::vector<int32_t>& pin_cell_idxs,
        const std::vector<std::vector<std::pair<int64_t, int64_t>>>& pin_xy_by_gi,
        int32_t trunk_gi,
        bool add_unowned_via_pad_extra = true,
        bool build_via_blocked = true
    );

    void set_net_trunk(int32_t net_id, int32_t trunk_gi) noexcept;
    void set_net_blocked_cells(int32_t net_id, const std::vector<int32_t>& idxs);
    void set_net_via_blocked_cells(int32_t net_id, const std::vector<int32_t>& vidxs);

    bool is_blocked_cell(
        int32_t net_id, int32_t ix, int32_t iy, int32_t gi, bool check_committed = false
    ) const noexcept;
    bool is_committed_foreign(int32_t net_id, int32_t ix, int32_t iy, int32_t gi) const noexcept;

    std::vector<std::vector<core::Box64>> foreign_committed_rects(int32_t net_id) const;
    std::vector<std::pair<int32_t, int32_t>> get_blocked_cells(int32_t net_id, int32_t gi) const;
    std::vector<std::pair<int32_t, int32_t>> get_via_infeasible_cells(int32_t net_id, int32_t gl) const;

    bool pin_is_trapped(
        int32_t net_id, int32_t ix, int32_t iy, int32_t gi, uint32_t allowed_mask
    );

    void reset_history() noexcept;
    void reset_committed() noexcept;
    void begin_round(const std::vector<int32_t>& active_net_ids);
    void mark_net_starting(int32_t net_id) noexcept;

    void commit_net_route(
        int32_t net_id,
        const std::vector<int32_t>& wire_cells,
        const std::vector<int32_t>& extra_seed_cells,
        bool ok
    );

    bool try_reserve_mirror_cell(int32_t net_b_id, int32_t idx);
    FinishRoundResult finish_round(double hist_fac);

    void commit_maze_cells(
        int32_t net_id, const std::vector<int32_t>& idxs, bool overwrite = true
    );
    void commit_maze_halo(
        int32_t net_id,
        const std::vector<int32_t>& wires,
        int32_t wire_w_nm,
        int32_t extra_sp_nm,
        int32_t max_other_w_nm,
        bool use_taper,
        int32_t via_partner_half_nm,
        bool overwrite_wires = true,
        bool include_via_halo = true
    );

    void clear_occupancy() noexcept;
    void set_occupancy_cell(int32_t idx, int32_t cnt);
    void clear_history_values() noexcept;
    void clear_via_pad_history_values() noexcept;
    void clear_pad_keepout_counts() noexcept;
    void set_pad_keepout_count_cell(int32_t idx, int32_t cnt);

    std::vector<int32_t> net_footprint(
        int32_t net_id,
        const std::vector<int32_t>& wires,
        int32_t route_w_nm,
        bool use_neck = true
    );

    double compute_via_cost_fast(
        const NetData& nd,
        int32_t ix, int32_t iy, int32_t gl,
        double pres_fac, double hist_fac
    ) const noexcept;

    double compute_cell_cost_fast(
        int32_t idx, int32_t xy, int32_t gi,
        double pres_fac, double hist_fac,
        int32_t trunk_gi,
        bool has_self_pk, uint32_t spk_gen,
        bool has_mirror, uint32_t mgen, double mirror_pen,
        bool use_via_cost, const NetData* nd,
        int32_t ix, int32_t iy,
        uint32_t wgen
    ) const noexcept;

    // Core undirected A* search.
    // Returns: 1 = path found (written to out_path), 0 = no path, -1 = checkpoint aborted.
    int astar_undirected(
        int32_t net_id,
        bool check_committed,
        const std::vector<int32_t>& sources,
        int32_t target_idx,
        uint32_t allowed_mask,
        bool use_pf_cost,
        double pres_fac,
        double hist_fac,
        int32_t trunk_gi,
        bool use_via_cost,
        bool has_self_pk,
        uint32_t spk_gen,
        bool has_mirror,
        uint32_t mgen,
        double mirror_pen,
        uint32_t wgen,
        bool has_window,
        int32_t win_x0, int32_t win_y0, int32_t win_x1, int32_t win_y1,
        int64_t max_expansions,
        core::CheckpointHook checkpoint,
        int64_t checkpoint_interval,
        std::vector<int32_t>& out_path
    );

    // Core directed A* search (comb contract with run state in [0..3]).
    // Returns: 1 = path found (written to out_path), 0 = no path, -1 = checkpoint aborted.
    int astar_directed(
        int32_t net_id,
        bool check_committed,
        const std::vector<int32_t>& sources,
        int32_t target_idx,
        uint32_t allowed_mask,
        const std::vector<DirPrefEntry>& dir_pref,
        bool has_dir_runs,
        uint32_t dr_gen,
        bool use_pf_cost,
        double pres_fac,
        double hist_fac,
        int32_t trunk_gi,
        bool use_via_cost,
        bool has_self_pk,
        uint32_t spk_gen,
        bool has_mirror,
        uint32_t mgen,
        double mirror_pen,
        uint32_t wgen,
        bool has_window,
        int32_t win_x0, int32_t win_y0, int32_t win_x1, int32_t win_y1,
        int64_t max_expansions,
        core::CheckpointHook checkpoint,
        int64_t checkpoint_interval,
        std::vector<int32_t>& out_path
    );

    void update_dir_runs(
        uint32_t tgen,
        uint32_t dr_gen,
        const std::vector<int32_t>& new_cells,
        const std::vector<DirPrefEntry>& dir_pref
    );

    // Complete multi-source soft routing for one net (sink-by-sink + 4-level window ladder).
    // Returns: 1 = connected, 0 = failed, -1 = checkpoint aborted.
    int route_one_net(
        int32_t net_id,
        bool check_committed,
        bool use_pf_cost,
        const std::vector<int32_t>& pin_cells,
        uint32_t allowed_mask,
        const std::vector<int32_t>& seed_cells,
        double pres_fac,
        double hist_fac,
        int32_t trunk_gi,
        bool use_via_cost,
        bool has_dir_pref,
        const std::vector<DirPrefEntry>& dir_pref,
        bool has_mirror,
        const std::vector<int32_t>& mirror_cells,
        double mirror_pen,
        bool use_window_ladder,
        int64_t max_expansions,
        core::CheckpointHook checkpoint,
        int64_t checkpoint_interval,
        std::vector<int32_t>& out_wire_cells
    );
};

// ===================== RectClipperCore =====================
// Bit-exact C++20 implementation of via_model.subtract_rects + engine_maze._emit
// (including the WIDEN-vs-CLIP sub-min-width filter and min-width spine retry).

struct LayerHoles {
    int64_t layer = 0;
    int64_t datatype = 0;
    int64_t min_x = 0;
    int64_t min_y = 0;
    int64_t max_x = 0;
    int64_t max_y = 0;
    std::vector<core::Box64> holes;
};

void subtract_rects_rec(
    int64_t rx0, int64_t ry0, int64_t rx1, int64_t ry1,
    const std::vector<core::Box64>& holes,
    size_t start_idx,
    std::vector<core::Box64>& out
);

class RectClipperCore {
public:
    std::vector<LayerHoles> layers;
    std::vector<core::Box64> scratch_raw;
    std::vector<core::Box64> scratch_keep;
    std::vector<core::Box64> scratch_keep2;
    std::vector<core::Box64> scratch_alt;

    void clear() noexcept;
    void add_layer_holes(int64_t layer, int64_t datatype, std::vector<core::Box64> holes);
    const LayerHoles* find_layer(int64_t layer, int64_t datatype) const noexcept;

    bool run_fragments(
        int64_t rx0, int64_t ry0, int64_t rx1, int64_t ry1,
        const LayerHoles& lh, int64_t floor_w,
        std::vector<core::Box64>& keep_out
    );

    const std::vector<core::Box64>& clip_emit(
        int64_t layer, int64_t datatype,
        int64_t x0, int64_t y0, int64_t x1, int64_t y1,
        int64_t min_w
    );

    const std::vector<core::Box64>& subtract_rects(
        int64_t layer, int64_t datatype,
        int64_t x0, int64_t y0, int64_t x1, int64_t y1
    );
};

}  // namespace clay_pnr::route
