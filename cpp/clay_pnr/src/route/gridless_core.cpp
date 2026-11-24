// Pure C++20 implementation of Clay's Gridless routing algorithmic kernels.
// Strictly free of CPython headers.

#include "clay_pnr/route/gridless_core.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <unordered_map>
#include <utility>
#include <vector>

namespace clay_pnr::route {

// ---------------------------------------------------------------------------
// 1. Native2DPotential core (1D Pareto reduction & 2D KD-tree)
// ---------------------------------------------------------------------------

std::vector<Pt1D> reduce_1d_pareto_cpp(std::vector<Pt1D>& pts, int64_t rate) {
    if (pts.empty()) return {};
    std::sort(pts.begin(), pts.end(), [](const Pt1D& a, const Pt1D& b) {
        if (a.p != b.p) return a.p < b.p;
        return a.v < b.v;
    });
    // Dedup identical p keeping min v
    std::vector<Pt1D> uniq;
    uniq.reserve(pts.size());
    for (const auto& pt : pts) {
        if (uniq.empty() || uniq.back().p != pt.p) {
            uniq.push_back(pt);
        }
    }
    std::vector<Pt1D> kept_left;
    kept_left.reserve(uniq.size());
    bool has_l = false;
    int64_t best_l = 0;
    for (const auto& pt : uniq) {
        if (!has_l || pt.v < best_l + pt.p * rate) {
            kept_left.push_back(pt);
            const int64_t val = pt.v - pt.p * rate;
            if (!has_l || val < best_l) {
                best_l = val;
                has_l = true;
            }
        }
    }
    std::vector<Pt1D> kept;
    kept.reserve(kept_left.size());
    bool has_r = false;
    int64_t best_r = 0;
    for (size_t i = kept_left.size(); i > 0; --i) {
        const auto& pt = kept_left[i - 1];
        if (!has_r || pt.v < best_r - pt.p * rate) {
            kept.push_back(pt);
            const int64_t val = pt.v + pt.p * rate;
            if (!has_r || val < best_r) {
                best_r = val;
                has_r = true;
            }
        }
    }
    std::reverse(kept.begin(), kept.end());
    return kept;
}

int32_t Potential2DCore::build_kd(std::vector<Pt2D>& items, size_t lo, size_t hi) {
    if (lo >= hi) return -1;
    int64_t x0 = items[lo].x, x1 = x0, y0 = items[lo].y, y1 = y0;
    int64_t min_c = items[lo].c;
    for (size_t i = lo + 1; i < hi; ++i) {
        if (items[i].x < x0) x0 = items[i].x;
        else if (items[i].x > x1) x1 = items[i].x;
        if (items[i].y < y0) y0 = items[i].y;
        else if (items[i].y > y1) y1 = items[i].y;
        if (items[i].c < min_c) min_c = items[i].c;
    }
    const bool split_x = (x1 - x0) >= (y1 - y0);
    const size_t mid = lo + (hi - lo) / 2;
    std::nth_element(items.begin() + static_cast<ptrdiff_t>(lo),
                     items.begin() + static_cast<ptrdiff_t>(mid),
                     items.begin() + static_cast<ptrdiff_t>(hi),
                     [split_x](const Pt2D& a, const Pt2D& b) {
                         return split_x ? (a.x < b.x) : (a.y < b.y);
                     });
    const Pt2D pivot = items[mid];
    const int32_t idx = static_cast<int32_t>(kd_nodes.size());
    kd_nodes.push_back(KDNode2D{pivot.x, pivot.y, pivot.c, x0, y0, x1, y1, min_c, -1, -1});
    const int32_t l = build_kd(items, lo, mid);
    const int32_t r = build_kd(items, mid + 1, hi);
    kd_nodes[static_cast<size_t>(idx)].left = l;
    kd_nodes[static_cast<size_t>(idx)].right = r;
    return idx;
}

void Potential2DCore::init(std::vector<Pt2D> pts, int64_t r) {
    rate = r;
    raw_pts = std::move(pts);
    south_p.clear();
    north_p.clear();
    west_p.clear();
    east_p.clear();
    kd_nodes.clear();
    kd_root = -1;
    if (raw_pts.empty()) {
        gx0 = gx1 = gy0 = gy1 = 0;
        sw_k = se_k = nw_k = ne_k = 0;
        return;
    }
    gx0 = gx1 = raw_pts[0].x;
    gy0 = gy1 = raw_pts[0].y;
    bool has_k = false;
    std::vector<Pt1D> s_pts, n_pts, w_pts, e_pts;
    s_pts.reserve(raw_pts.size());
    n_pts.reserve(raw_pts.size());
    w_pts.reserve(raw_pts.size());
    e_pts.reserve(raw_pts.size());
    for (const auto& p : raw_pts) {
        if (p.x < gx0) gx0 = p.x;
        else if (p.x > gx1) gx1 = p.x;
        if (p.y < gy0) gy0 = p.y;
        else if (p.y > gy1) gy1 = p.y;
        const int64_t gx_r = p.x * rate;
        const int64_t gy_r = p.y * rate;
        const int64_t c_sw = p.c + gx_r + gy_r;
        const int64_t c_se = p.c - gx_r + gy_r;
        const int64_t c_nw = p.c + gx_r - gy_r;
        const int64_t c_ne = p.c - gx_r - gy_r;
        if (!has_k) {
            sw_k = c_sw;
            se_k = c_se;
            nw_k = c_nw;
            ne_k = c_ne;
            has_k = true;
        } else {
            if (c_sw < sw_k) sw_k = c_sw;
            if (c_se < se_k) se_k = c_se;
            if (c_nw < nw_k) nw_k = c_nw;
            if (c_ne < ne_k) ne_k = c_ne;
        }
        s_pts.push_back(Pt1D{p.x, p.c + gy_r});
        n_pts.push_back(Pt1D{p.x, p.c - gy_r});
        w_pts.push_back(Pt1D{p.y, p.c + gx_r});
        e_pts.push_back(Pt1D{p.y, p.c - gx_r});
    }
    south_p = reduce_1d_pareto_cpp(s_pts, rate);
    north_p = reduce_1d_pareto_cpp(n_pts, rate);
    west_p = reduce_1d_pareto_cpp(w_pts, rate);
    east_p = reduce_1d_pareto_cpp(e_pts, rate);

    if (raw_pts.size() > 48) {
        std::vector<Pt2D> copy = raw_pts;
        kd_nodes.reserve(copy.size());
        kd_root = build_kd(copy, 0, copy.size());
    }
}

// ---------------------------------------------------------------------------
// 2. NativeLayerHmin core
// ---------------------------------------------------------------------------

int64_t LayerHminCore::eval(int64_t x, int64_t y, int32_t gi) {
    const std::pair<int64_t, int64_t> key{x, y};
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second[static_cast<size_t>(gi)];
    }
    int64_t dist = std::numeric_limits<int64_t>::max();
    const bool has_targets = !target_boxes.empty();
    if (target_boxes.size() == 1) {
        const Box64& tb = target_boxes[0];
        const int64_t dx = (x < tb.x0) ? (tb.x0 - x) : ((x > tb.x1) ? (x - tb.x1) : 0);
        const int64_t dy = (y < tb.y0) ? (tb.y0 - y) : ((y > tb.y1) ? (y - tb.y1) : 0);
        dist = dx + dy;
    } else if (has_targets) {
        for (const auto& tb : target_boxes) {
            const int64_t dx = (x < tb.x0) ? (tb.x0 - x) : ((x > tb.x1) ? (x - tb.x1) : 0);
            const int64_t dy = (y < tb.y0) ? (tb.y0 - y) : ((y > tb.y1) ? (y - tb.y1) : 0);
            const int64_t d = dx + dy;
            if (d < dist) dist = d;
        }
    }
    constexpr int64_t INF_H = (1LL << 60);
    std::vector<int64_t> b(static_cast<size_t>(n_gi), INF_H);
    for (const auto& s : specs) {
        int64_t best = INF_H;
        if (has_targets) {
            best = dist * s.rate + s.via_tail;
        }
        if (s.pot != nullptr) {
            const int64_t idx_val = s.pot->eval(x, y);
            if (idx_val < best) best = idx_val;
        }
        b[static_cast<size_t>(s.layer)] = best;
    }
    for (int32_t g = 1; g < n_gi; ++g) {
        const int64_t cand = b[static_cast<size_t>(g - 1)] + via_cost;
        if (cand < b[static_cast<size_t>(g)]) b[static_cast<size_t>(g)] = cand;
    }
    for (int32_t g = n_gi - 2; g >= 0; --g) {
        const int64_t cand = b[static_cast<size_t>(g + 1)] + via_cost;
        if (cand < b[static_cast<size_t>(g)]) b[static_cast<size_t>(g)] = cand;
    }
    const int64_t ans = b[static_cast<size_t>(gi)];
    if (cache.size() < max_cache) {
        cache.emplace(key, std::move(b));
    }
    return ans;
}

// ---------------------------------------------------------------------------
// 3. Exact 3D separable L1 min-plus distance transform
// ---------------------------------------------------------------------------

DistanceTransformStatus exact_distance_transform_from_seeds(
    const std::vector<int64_t>& xs,
    const std::vector<int64_t>& ys,
    const std::vector<int64_t>& layers,
    const std::vector<DistanceTransformSeed>& seeds,
    int64_t max_term,
    int64_t step_cost,
    int64_t via_cost,
    std::vector<int64_t>& out_pot) {
    out_pot.clear();
    const size_t nx = xs.size();
    const size_t ny = ys.size();
    const size_t nl = layers.size();
    if (nx == 0 || ny == 0 || nl == 0 || seeds.empty()) {
        return DistanceTransformStatus::Empty;
    }

    const int64_t span_x = xs.back() - xs.front();
    const int64_t span_y = ys.back() - ys.front();
    const int64_t span_l = layers.back() - layers.front();
    const double approx_ub = static_cast<double>(max_term)
        + static_cast<double>(span_x) * static_cast<double>(step_cost)
        + static_cast<double>(span_y) * static_cast<double>(step_cost)
        + static_cast<double>(span_l) * static_cast<double>(via_cost);
    if (approx_ub > 4.0e18 || approx_ub < -4.0e18) {
        return DistanceTransformStatus::Overflow;
    }

    const size_t plane = nx * ny;
    const size_t total = plane * nl;
    const int64_t upper_bound =
        max_term + span_x * step_cost + span_y * step_cost + span_l * via_cost + 1;

    try {
        out_pot.assign(total, upper_bound);
    } catch (const std::bad_alloc&) {
        out_pot.clear();
        return DistanceTransformStatus::BadAlloc;
    }

    for (const auto& s : seeds) {
        if (s.cost < out_pot[s.flat_idx]) out_pot[s.flat_idx] = s.cost;
    }

    std::vector<int64_t> dx_cost(nx > 1 ? nx - 1 : 0);
    for (size_t x = 1; x < nx; ++x) dx_cost[x - 1] = (xs[x] - xs[x - 1]) * step_cost;
    std::vector<int64_t> dy_cost(ny > 1 ? ny - 1 : 0);
    for (size_t y = 1; y < ny; ++y) dy_cost[y - 1] = (ys[y] - ys[y - 1]) * step_cost;
    std::vector<int64_t> dl_cost(nl > 1 ? nl - 1 : 0);
    for (size_t l = 1; l < nl; ++l) dl_cost[l - 1] = (layers[l] - layers[l - 1]) * via_cost;

    int64_t* data = out_pot.data();

    // Pass 1: X forward & backward
    if (nx > 1) {
        for (size_t loff = 0; loff < total; loff += plane) {
            for (size_t yoff = 0; yoff < plane; yoff += nx) {
                int64_t* row = data + loff + yoff;
                for (size_t x = 1; x < nx; ++x) {
                    const int64_t cand = row[x - 1] + dx_cost[x - 1];
                    if (cand < row[x]) row[x] = cand;
                }
                for (size_t x = nx - 1; x > 0; --x) {
                    const int64_t cand = row[x] + dx_cost[x - 1];
                    if (cand < row[x - 1]) row[x - 1] = cand;
                }
            }
        }
    }

    // Pass 2: Y forward & backward (row-major order for cache friendliness & vectorization)
    if (ny > 1) {
        for (size_t loff = 0; loff < total; loff += plane) {
            int64_t* base = data + loff;
            for (size_t y = 1; y < ny; ++y) {
                const int64_t dy = dy_cost[y - 1];
                const int64_t* prev_row = base + (y - 1) * nx;
                int64_t* cur_row = base + y * nx;
                for (size_t x = 0; x < nx; ++x) {
                    const int64_t cand = prev_row[x] + dy;
                    if (cand < cur_row[x]) cur_row[x] = cand;
                }
            }
            for (size_t y = ny - 1; y > 0; --y) {
                const int64_t dy = dy_cost[y - 1];
                const int64_t* next_row = base + y * nx;
                int64_t* cur_row = base + (y - 1) * nx;
                for (size_t x = 0; x < nx; ++x) {
                    const int64_t cand = next_row[x] + dy;
                    if (cand < cur_row[x]) cur_row[x] = cand;
                }
            }
        }
    }

    // Pass 3: Layer forward & backward (contiguous plane sweeps)
    if (nl > 1) {
        for (size_t l = 1; l < nl; ++l) {
            const int64_t dl = dl_cost[l - 1];
            const int64_t* prev_p = data + (l - 1) * plane;
            int64_t* cur_p = data + l * plane;
            for (size_t idx = 0; idx < plane; ++idx) {
                const int64_t cand = prev_p[idx] + dl;
                if (cand < cur_p[idx]) cur_p[idx] = cand;
            }
        }
        for (size_t l = nl - 1; l > 0; --l) {
            const int64_t dl = dl_cost[l - 1];
            const int64_t* next_p = data + l * plane;
            int64_t* cur_p = data + (l - 1) * plane;
            for (size_t idx = 0; idx < plane; ++idx) {
                const int64_t cand = next_p[idx] + dl;
                if (cand < cur_p[idx]) cur_p[idx] = cand;
            }
        }
    }

    return DistanceTransformStatus::Ok;
}

DistanceTransformStatus exact_distance_transform_core(
    const std::vector<int64_t>& xs,
    const std::vector<int64_t>& ys,
    const std::vector<int64_t>& layers,
    const std::vector<DistanceTransformGoal>& goals,
    int64_t step_cost,
    int64_t via_cost,
    std::vector<int64_t>& out_pot) {
    out_pot.clear();
    const size_t nx = xs.size();
    const size_t ny = ys.size();
    const size_t nl = layers.size();
    if (nx == 0 || ny == 0 || nl == 0) {
        return DistanceTransformStatus::Empty;
    }

    auto validate_axis = [](const std::vector<int64_t>& v,
                            std::unordered_map<int64_t, size_t>& idx) -> bool {
        idx.reserve(v.size() * 2);
        for (size_t i = 0; i < v.size(); ++i) {
            if (i > 0 && v[i] <= v[i - 1]) {
                return false;
            }
            idx.emplace(v[i], i);
        }
        return true;
    };

    std::unordered_map<int64_t, size_t> x_idx, y_idx, l_idx;
    if (!validate_axis(xs, x_idx) || !validate_axis(ys, y_idx) || !validate_axis(layers, l_idx)) {
        return DistanceTransformStatus::NonMonotonicAxis;
    }

    if (goals.empty()) {
        return DistanceTransformStatus::Empty;
    }

    const size_t plane = nx * ny;
    std::vector<DistanceTransformSeed> seeds;
    seeds.reserve(goals.size());
    bool has_max = false;
    int64_t max_term = 0;
    for (const auto& g : goals) {
        auto ix = x_idx.find(g.x);
        auto iy = y_idx.find(g.y);
        auto il = l_idx.find(g.layer);
        if (ix == x_idx.end() || iy == y_idx.end() || il == l_idx.end()) {
            return DistanceTransformStatus::GoalNotFound;
        }
        if (!has_max || g.cost > max_term) {
            max_term = g.cost;
            has_max = true;
        }
        seeds.push_back(DistanceTransformSeed{
            il->second * plane + iy->second * nx + ix->second, g.cost});
    }

    return exact_distance_transform_from_seeds(
        xs, ys, layers, seeds, max_term, step_cost, via_cost, out_pot);
}

// ---------------------------------------------------------------------------
// 4. Throat events computation
// ---------------------------------------------------------------------------

void compute_axis_events(
    const std::vector<int64_t>& base_vals,
    int64_t dom_lo, int64_t dom_hi,
    int64_t atom_lo, int64_t atom_hi,
    int64_t bh, int64_t grid,
    std::vector<int64_t>& out) {
    std::vector<int64_t> seeds;
    seeds.reserve(base_vals.size() + 4);
    seeds.push_back(atom_lo - bh);
    seeds.push_back(atom_lo + bh);
    seeds.push_back(atom_hi - bh);
    seeds.push_back(atom_hi + bh);
    seeds.insert(seeds.end(), base_vals.begin(), base_vals.end());
    std::sort(seeds.begin(), seeds.end());
    seeds.erase(std::unique(seeds.begin(), seeds.end()), seeds.end());

    out.clear();
    out.reserve(seeds.size() * 4);
    int64_t buf[2];
    int cnt = 0;
    for (int64_t s : seeds) {
        for (int64_t d = -1; d <= 1; ++d) {
            const int64_t v = s + d;
            if (dom_lo <= v && v <= dom_hi) {
                grid_proj_preserve_sides(v, dom_lo, dom_hi, grid, buf, cnt);
                for (int i = 0; i < cnt; ++i) out.push_back(buf[i]);
            }
        }
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
}

ThroatEventsResult compute_throat_events_core(
    const std::vector<int64_t>& base_xs,
    const std::vector<int64_t>& base_ys,
    int64_t x0, int64_t y0, int64_t x1, int64_t y1,
    int64_t bh, int64_t min_width, int64_t grid) {
    if (x0 > x1) std::swap(x0, x1);
    if (y0 > y1) std::swap(y0, y1);
    if (grid < 1) grid = 1;

    const int64_t cx0 = x0 - bh;
    const int64_t cy0 = y0 - bh;
    const int64_t cx1 = x1 + bh;
    const int64_t cy1 = y1 + bh;

    ThroatEventsResult res;
    compute_axis_events(base_xs, cx0, cx1, x0, x1, bh, grid, res.xs);
    compute_axis_events(base_ys, cy0, cy1, y0, y1, bh, grid, res.ys);

    const int64_t width_sq = min_width * min_width;
    const int64_t max_req_x = std::min(x1 - x0, 2 * bh);
    const int64_t max_req_y = std::min(y1 - y0, 2 * bh);

    res.thresh.reserve((res.xs.size() + res.ys.size()) * 12);
    int64_t gbuf1[2], gbuf2[2];
    int c1 = 0, c2 = 0;

    for (int64_t y : res.ys) {
        const int64_t oy = std::max<int64_t>(0, std::min(y1, y + bh) - std::max(y0, y - bh));
        const int64_t req_x = ceil_sqrt_py_exact(width_sq - oy * oy);
        if (req_x <= max_req_x) {
            grid_proj_preserve_sides(y, cy0, cy1, grid, gbuf2, c2);
            if (c2 == 0) continue;
            const int64_t bounds[2] = {x0 - bh + req_x, x1 + bh - req_x};
            for (int bi = 0; bi < 2; ++bi) {
                for (int64_t d = -1; d <= 1; ++d) {
                    grid_proj_preserve_sides(bounds[bi] + d, cx0, cx1, grid, gbuf1, c1);
                    for (int i = 0; i < c1; ++i) {
                        for (int j = 0; j < c2; ++j) {
                            res.thresh.emplace_back(gbuf1[i], gbuf2[j]);
                        }
                    }
                }
            }
        }
    }
    for (int64_t x : res.xs) {
        const int64_t ox = std::max<int64_t>(0, std::min(x1, x + bh) - std::max(x0, x - bh));
        const int64_t req_y = ceil_sqrt_py_exact(width_sq - ox * ox);
        if (req_y <= max_req_y) {
            grid_proj_preserve_sides(x, cx0, cx1, grid, gbuf1, c1);
            if (c1 == 0) continue;
            const int64_t bounds[2] = {y0 - bh + req_y, y1 + bh - req_y};
            for (int bi = 0; bi < 2; ++bi) {
                for (int64_t d = -1; d <= 1; ++d) {
                    grid_proj_preserve_sides(bounds[bi] + d, cy0, cy1, grid, gbuf2, c2);
                    for (int i = 0; i < c1; ++i) {
                        for (int j = 0; j < c2; ++j) {
                            res.thresh.emplace_back(gbuf1[i], gbuf2[j]);
                        }
                    }
                }
            }
        }
    }
    std::sort(res.thresh.begin(), res.thresh.end());
    res.thresh.erase(std::unique(res.thresh.begin(), res.thresh.end()), res.thresh.end());
    return res;
}

// ---------------------------------------------------------------------------
// 5. HistoryLayer & HistoryGridCore
// ---------------------------------------------------------------------------

void HistoryLayer::clear() noexcept {
    dense = false;
    min_cx = max_cx = min_cy = max_cy = 0;
    width = height = stride = 0;
    sat.clear();
    sparse.clear();
}

void HistoryLayer::build_from_cells(const std::vector<HistoryRawCell>& cells) {
    clear();
    if (cells.empty()) return;

    int64_t min_x = cells[0].cx, max_x = cells[0].cx;
    int64_t min_y = cells[0].cy, max_y = cells[0].cy;
    uint64_t abs_sum = 0;
    bool sum_safe = true;
    for (const auto& c : cells) {
        if (c.cx < min_x) min_x = c.cx;
        else if (c.cx > max_x) max_x = c.cx;
        if (c.cy < min_y) min_y = c.cy;
        else if (c.cy > max_y) max_y = c.cy;
        const uint64_t av = (c.val < 0)
            ? (0ULL - static_cast<uint64_t>(c.val))
            : static_cast<uint64_t>(c.val);
        if (abs_sum > (static_cast<uint64_t>(INT64_MAX) / 4ULL) - av) {
            sum_safe = false;
        } else {
            abs_sum += av;
        }
    }
    min_cx = min_x;
    max_cx = max_x;
    min_cy = min_y;
    max_cy = max_y;

    const uint64_t diff_x = static_cast<uint64_t>(max_x) - static_cast<uint64_t>(min_x);
    const uint64_t diff_y = static_cast<uint64_t>(max_y) - static_cast<uint64_t>(min_y);
    if (sum_safe && diff_x < MAX_DENSE_DIM && diff_y < MAX_DENSE_DIM &&
        (diff_x + 1ULL) * (diff_y + 1ULL) <= MAX_DENSE_CELLS) {
        dense = true;
        width = static_cast<int64_t>(diff_x + 1ULL);
        height = static_cast<int64_t>(diff_y + 1ULL);
        stride = width + 1;
        const size_t st = static_cast<size_t>(stride);
        sat.assign(static_cast<size_t>(height + 1) * st, 0);
        for (const auto& c : cells) {
            const size_t x = static_cast<size_t>(c.cx - min_x + 1);
            const size_t y = static_cast<size_t>(c.cy - min_y + 1);
            sat[y * st + x] += c.val;
        }
        for (size_t y = 1; y <= static_cast<size_t>(height); ++y) {
            int64_t row_acc = 0;
            int64_t* cur = sat.data() + y * st;
            const int64_t* prev = sat.data() + (y - 1) * st;
            for (size_t x = 1; x <= static_cast<size_t>(width); ++x) {
                row_acc += cur[x];
                cur[x] = row_acc + prev[x];
            }
        }
    } else {
        dense = false;
        sparse.reserve(cells.size() * 2 + 1);
        for (const auto& c : cells) {
            sparse[{c.cx, c.cy}] += c.val;
        }
    }
}

void HistoryLayer::unpack_dense_to_sparse() {
    if (!dense) return;
    sparse.clear();
    const size_t st = static_cast<size_t>(stride);
    bool first = true;
    for (int64_t y = 1; y <= height; ++y) {
        const size_t yu = static_cast<size_t>(y);
        const int64_t* cur = sat.data() + yu * st;
        const int64_t* prev = sat.data() + (yu - 1) * st;
        for (int64_t x = 1; x <= width; ++x) {
            const size_t xu = static_cast<size_t>(x);
            const int64_t v = cur[xu] - prev[xu] - cur[xu - 1] + prev[xu - 1];
            if (v != 0) {
                const int64_t cx = min_cx + x - 1;
                const int64_t cy = min_cy + y - 1;
                sparse.emplace(std::make_pair(cx, cy), v);
                if (first) {
                    min_cx = max_cx = cx;
                    min_cy = max_cy = cy;
                    first = false;
                } else {
                    if (cx < min_cx) min_cx = cx;
                    if (cx > max_cx) max_cx = cx;
                    if (cy < min_cy) min_cy = cy;
                    if (cy > max_cy) max_cy = cy;
                }
            }
        }
    }
    sat.clear();
    dense = false;
    width = height = stride = 0;
    if (first) {
        min_cx = max_cx = min_cy = max_cy = 0;
    }
}

void HistoryLayer::increment(int64_t cx, int64_t cy, int64_t delta, int64_t saturation_cap) {
    if (delta == 0) return;
    if (dense) unpack_dense_to_sparse();
    const int64_t cap = (saturation_cap < 0) ? 0 : saturation_cap;
    const bool was_empty = sparse.empty();
    auto it = sparse.find({cx, cy});
    const int64_t cur = (it != sparse.end()) ? it->second : 0;
    int64_t next = 0;
    if (delta > 0) {
        if (cur >= cap || (cap - cur) <= delta) {
            next = cap;
        } else {
            next = cur + delta;
        }
    } else {
        const int64_t neg_cap = -cap;
        if (cur <= neg_cap || (neg_cap - cur) >= delta) {
            next = neg_cap;
        } else {
            next = cur + delta;
        }
    }
    if (next == 0) {
        if (it != sparse.end()) sparse.erase(it);
        if (sparse.empty()) {
            min_cx = max_cx = min_cy = max_cy = 0;
        }
        return;
    }
    if (it != sparse.end()) {
        it->second = next;
    } else {
        sparse.emplace(std::make_pair(cx, cy), next);
    }
    if (was_empty) {
        min_cx = max_cx = cx;
        min_cy = max_cy = cy;
    } else {
        if (cx < min_cx) min_cx = cx;
        if (cx > max_cx) max_cx = cx;
        if (cy < min_cy) min_cy = cy;
        if (cy > max_cy) max_cy = cy;
    }
}

void HistoryLayer::decay(double factor) {
    if (empty()) return;
    if (factor <= 0.0) {
        clear();
        return;
    }
    const bool was_dense = dense;
    if (dense) unpack_dense_to_sparse();
    std::vector<HistoryRawCell> kept;
    kept.reserve(sparse.size());
    for (const auto& kv : sparse) {
        const double scaled = static_cast<double>(kv.second) * factor;
        const int64_t nv = round_half_even_i64(scaled);
        if (nv != 0) {
            kept.push_back(HistoryRawCell{kv.first.first, kv.first.second, nv});
        }
    }
    if (kept.empty()) {
        clear();
        return;
    }
    if (was_dense) {
        build_from_cells(kept);
    } else {
        sparse.clear();
        sparse.reserve(kept.size() * 2 + 1);
        min_cx = max_cx = kept[0].cx;
        min_cy = max_cy = kept[0].cy;
        for (const auto& c : kept) {
            sparse.emplace(std::make_pair(c.cx, c.cy), c.val);
            if (c.cx < min_cx) min_cx = c.cx;
            if (c.cx > max_cx) max_cx = c.cx;
            if (c.cy < min_cy) min_cy = c.cy;
            if (c.cy > max_cy) max_cy = c.cy;
        }
    }
}

void HistoryLayer::compile() {
    if (dense || sparse.empty()) return;
    std::vector<HistoryRawCell> cells;
    cells.reserve(sparse.size());
    for (const auto& kv : sparse) {
        if (kv.second != 0) {
            cells.push_back(HistoryRawCell{kv.first.first, kv.first.second, kv.second});
        }
    }
    build_from_cells(cells);
}

void HistoryGridCore::clear() noexcept {
    fast_layers.clear();
    extra_layers.clear();
    has_any = false;
}

void HistoryGridCore::build(
    const std::unordered_map<int64_t, std::vector<HistoryRawCell>>& by_layer) {
    clear();
    if (by_layer.empty()) return;

    int64_t max_fast_rgi = -1;
    for (const auto& kv : by_layer) {
        if (!kv.second.empty() && kv.first >= 0 && kv.first < FAST_LAYER_LIMIT &&
            kv.first > max_fast_rgi) {
            max_fast_rgi = kv.first;
        }
    }
    if (max_fast_rgi >= 0) {
        fast_layers.resize(static_cast<size_t>(max_fast_rgi + 1));
    }

    for (const auto& kv : by_layer) {
        if (kv.second.empty()) continue;
        const int64_t rgi = kv.first;
        HistoryLayer& hl = (rgi >= 0 && rgi < static_cast<int64_t>(fast_layers.size()))
            ? fast_layers[static_cast<size_t>(rgi)]
            : extra_layers[rgi];
        hl.build_from_cells(kv.second);
        if (!hl.empty()) has_any = true;
    }
}

void HistoryGridCore::increment(int64_t rgi, int64_t cx, int64_t cy, int64_t delta,
                                int64_t saturation_cap) {
    if (delta == 0) return;
    HistoryLayer* hl = nullptr;
    if (rgi >= 0 && rgi < FAST_LAYER_LIMIT) {
        if (static_cast<size_t>(rgi) >= fast_layers.size()) {
            fast_layers.resize(static_cast<size_t>(rgi + 1));
        }
        hl = &fast_layers[static_cast<size_t>(rgi)];
    } else {
        hl = &extra_layers[rgi];
    }
    hl->increment(cx, cy, delta, saturation_cap);
    if (!hl->empty()) {
        has_any = true;
    } else {
        bool any = false;
        for (const auto& fl : fast_layers) {
            if (!fl.empty()) { any = true; break; }
        }
        if (!any) {
            for (const auto& kv : extra_layers) {
                if (!kv.second.empty()) { any = true; break; }
            }
        }
        has_any = any;
    }
}

void HistoryGridCore::decay(double factor) {
    if (!has_any) return;
    bool any = false;
    for (auto& fl : fast_layers) {
        fl.decay(factor);
        if (!fl.empty()) any = true;
    }
    for (auto it = extra_layers.begin(); it != extra_layers.end(); ) {
        it->second.decay(factor);
        if (it->second.empty()) {
            it = extra_layers.erase(it);
        } else {
            any = true;
            ++it;
        }
    }
    has_any = any;
}

void HistoryGridCore::compile() {
    if (!has_any) return;
    for (auto& fl : fast_layers) fl.compile();
    for (auto& kv : extra_layers) kv.second.compile();
}

// ---------------------------------------------------------------------------
// 6. CartesianGraphCache
// ---------------------------------------------------------------------------

void CartesianGraphCache::reset_for_graph(
    const std::vector<int64_t>& in_xs,
    const std::vector<int64_t>& in_ys,
    const std::vector<int32_t>& in_layers,
    int32_t max_gi,
    int64_t max_expansions,
    bool within_cap) {
    if (initialized && xs == in_xs && ys == in_ys && ordered_layers == in_layers &&
        graph_within_potential_cap == within_cap) {
        return;
    }
    xs = in_xs;
    ys = in_ys;
    ordered_layers = in_layers;
    nx = static_cast<int64_t>(xs.size());
    ny = static_cast<int64_t>(ys.size());
    nl = static_cast<int64_t>(ordered_layers.size());
    layer_index.assign(static_cast<size_t>(std::max<int32_t>(0, max_gi) + 1), -1);
    for (int32_t i = 0; i < static_cast<int32_t>(nl); ++i) {
        const int32_t g = ordered_layers[static_cast<size_t>(i)];
        if (g >= 0 && g < static_cast<int32_t>(layer_index.size())) {
            layer_index[static_cast<size_t>(g)] = i;
        }
    }
    via_pair_index.assign(static_cast<size_t>(std::max<int32_t>(0, max_gi) + 1), -1);
    int32_t n_pairs = 0;
    for (size_t i = 0; i + 1 < ordered_layers.size(); ++i) {
        if (ordered_layers[i + 1] == ordered_layers[i] + 1) {
            const int32_t lo = ordered_layers[i];
            if (lo >= 0 && lo < static_cast<int32_t>(via_pair_index.size())) {
                via_pair_index[static_cast<size_t>(lo)] = n_pairs++;
            }
        }
    }
    plane = nx * ny;
    potential_node_count = nl * plane;
    horizontal_edges = nl * ny * (nx > 0 ? nx - 1 : 0);
    vertical_edges = nl * (ny > 0 ? ny - 1 : 0) * nx;
    via_edges = static_cast<int64_t>(n_pairs) * plane;
    packed_edge_count = horizontal_edges + vertical_edges + via_edges;
    packed_slot_count = std::min(packed_edge_count, std::max<int64_t>(0, max_expansions));
    graph_within_potential_cap = within_cap;
    static_geometry_work = 0;
    packed_enabled = false;
    packed_disabled = false;
    packed_keys.clear();
    packed_costs.clear();
    packed_present.clear();
    preflight_tested_edges.clear();
    via_model_map.clear();
    if (via_edges <= 4000000) {
        via_model_use_flat = true;
        via_model_flat.assign(static_cast<size_t>(via_edges), 0);
    } else {
        via_model_use_flat = false;
        via_model_flat.clear();
    }
    initialized = true;
}

bool CartesianGraphCache::enable_packed(bool force) {
    if (packed_enabled || packed_disabled || !graph_within_potential_cap || packed_slot_count <= 0) {
        return packed_enabled;
    }
    if (!force && static_geometry_work <= packed_slot_count) {
        return false;
    }
    try {
        packed_keys.assign(static_cast<size_t>(packed_slot_count), PACKED_UNKNOWN);
        packed_costs.assign(static_cast<size_t>(packed_slot_count), 0);
        packed_present.assign(static_cast<size_t>(packed_slot_count) * 3, PACKED_PRESENT_UNKNOWN);
    } catch (const std::bad_alloc&) {
        packed_keys.clear();
        packed_costs.clear();
        packed_present.clear();
        packed_disabled = true;
        preflight_tested_edges.clear();
        return false;
    }
    preflight_tested_edges.clear();
    packed_enabled = true;
    return true;
}

// ---------------------------------------------------------------------------
// 7. Region-Event Geometry pure path projectors
// ---------------------------------------------------------------------------

RegionGeomError compute_portal_paths_core(
    const Box64& src_rect, const Box64& tgt_rect,
    bool axis_is_x, int64_t direction, int64_t boundary, int64_t lo, int64_t hi,
    int64_t px, int64_t py,
    SmallPath out_paths[2], int& out_n) noexcept {
    if (!(src_rect.x0 <= px && px < src_rect.x1 && src_rect.y0 <= py && py < src_rect.y1)) {
        out_n = 0;
        return RegionGeomError::EntryOutsideSource;
    }
    int64_t near_x = 0, near_y = 0, far_x = 0, far_y = 0;
    if (axis_is_x) {
        const int64_t tangent = std::min(std::max(py, lo), hi - 1);
        near_x = boundary - ((direction == 1) ? 1 : 0);
        near_y = tangent;
        far_x = boundary - ((direction == 1) ? 0 : 1);
        far_y = tangent;
    } else {
        const int64_t tangent = std::min(std::max(px, lo), hi - 1);
        near_x = tangent;
        near_y = boundary - ((direction == 1) ? 1 : 0);
        far_x = tangent;
        far_y = boundary - ((direction == 1) ? 0 : 1);
    }
    if (!(src_rect.x0 <= near_x && near_x < src_rect.x1 &&
          src_rect.y0 <= near_y && near_y < src_rect.y1 &&
          tgt_rect.x0 <= far_x && far_x < tgt_rect.x1 &&
          tgt_rect.y0 <= far_y && far_y < tgt_rect.y1)) {
        out_n = 0;
        return RegionGeomError::PortalMismatch;
    }
    SmallPath p0, p1;
    p0.push_dedup(px, py);
    p0.push_dedup(near_x, py);
    p0.push_dedup(near_x, near_y);
    p0.push_dedup(far_x, far_y);
    p1.push_dedup(px, py);
    p1.push_dedup(px, near_y);
    p1.push_dedup(near_x, near_y);
    p1.push_dedup(far_x, far_y);
    if (p0 == p1) {
        out_paths[0] = p0;
        out_n = 1;
    } else if (p0 < p1) {
        out_paths[0] = p0;
        out_paths[1] = p1;
        out_n = 2;
    } else {
        out_paths[0] = p1;
        out_paths[1] = p0;
        out_n = 2;
    }
    return RegionGeomError::Ok;
}

RegionGeomError compute_target_paths_core(
    const Box64& src_rect, const Box64& tgt_rect,
    int64_t px, int64_t py,
    SmallPath out_paths[2], int& out_n) noexcept {
    if (!(src_rect.x0 <= px && px < src_rect.x1 && src_rect.y0 <= py && py < src_rect.y1)) {
        out_n = 0;
        return RegionGeomError::EntryOutsideSource;
    }
    const int64_t lo_x = std::max(src_rect.x0, tgt_rect.x0);
    const int64_t lo_y = std::max(src_rect.y0, tgt_rect.y0);
    const int64_t hi_x = std::min(src_rect.x1, tgt_rect.x1);
    const int64_t hi_y = std::min(src_rect.y1, tgt_rect.y1);
    if (lo_x >= hi_x || lo_y >= hi_y) {
        out_n = 0;
        return RegionGeomError::Ok;
    }
    const int64_t ex = std::min(std::max(px, lo_x), hi_x - 1);
    const int64_t ey = std::min(std::max(py, lo_y), hi_y - 1);
    SmallPath p0, p1;
    p0.push_dedup(px, py);
    p0.push_dedup(ex, py);
    p0.push_dedup(ex, ey);
    p1.push_dedup(px, py);
    p1.push_dedup(px, ey);
    p1.push_dedup(ex, ey);
    if (p0 == p1) {
        out_paths[0] = p0;
        out_n = 1;
    } else if (p0 < p1) {
        out_paths[0] = p0;
        out_paths[1] = p1;
        out_n = 2;
    } else {
        out_paths[0] = p1;
        out_paths[1] = p0;
        out_n = 2;
    }
    return RegionGeomError::Ok;
}

}  // namespace clay_pnr::route
