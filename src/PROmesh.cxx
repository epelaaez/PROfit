#include "PROmesh.h"
#include "PROlog.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_set>

namespace PROfit {
namespace PROmesh {

namespace {

// Pack (i, j) integer coordinates into a 64-bit key for the chi²/bestfit maps.
inline uint64_t make_key(int i, int j) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(i)) << 32) |
            static_cast<uint64_t>(static_cast<uint32_t>(j));
}

// A cell at refinement level `level` whose bottom-left corner is (i_bl, j_bl)
// in finest-integer coords. Step size between corners on each axis = 1 << (max_levels - level).
struct Cell {
    int i_bl;
    int j_bl;
    int level;
    int step;       ///< 2^(max_levels - level), the integer-coord distance between corners.
    bool refined;   ///< True if this cell has been replaced by 2×2 children. Skipped during marching squares.
};

// Map integer coord → physical coord using transformed bounds (caller has
// already log10'd if appropriate).
inline float i_to_x(int i, int finest_n_x, float x_lo, float x_hi) {
    return x_lo + (static_cast<float>(i) / static_cast<float>(finest_n_x)) * (x_hi - x_lo);
}
inline float j_to_y(int j, int finest_n_y, float y_lo, float y_hi) {
    return y_lo + (static_cast<float>(j) / static_cast<float>(finest_n_y)) * (y_hi - y_lo);
}

// Per-cell straddle test. A cell straddles ANY target c_k if
//   min(corners) < c_k + δ  AND  max(corners) > c_k − δ
// (chi² values here are raw, NOT yet offset by global min — but the targets
// passed in are Δχ² targets, so we compare against (chi_min_global + target).)
bool straddles_any(float cmin, float cmax,
                    const std::vector<float> &targets, float delta) {
    for (float t : targets) {
        if (cmin < t + delta && cmax > t - delta) return true;
    }
    return false;
}

// Linear-interpolated crossing position along a cell edge between corners
// (c_a, t_a) and (c_b, t_b), at chi² level c_target. Returns the t value
// where the linear interp hits c_target. If the edge doesn't cross (the
// caller should have already checked), returns midpoint as a safe default.
float interp_crossing(float c_a, float c_b, float t_a, float t_b, float c_target) {
    const float denom = c_b - c_a;
    if (std::fabs(denom) < 1e-12f) return 0.5f * (t_a + t_b);
    return t_a + (c_target - c_a) / denom * (t_b - t_a);
}

}  // anonymous namespace


AMRResult run_amr(EvalFn eval,
                  float x_lo, float x_hi,
                  float y_lo, float y_hi,
                  const AMROptions &opts)
{
    AMRResult result;

    if (opts.initial_nx < 1 || opts.initial_ny < 1) {
        log<LOG_ERROR>(L"%1% || PROmesh: initial_nx and initial_ny must be ≥ 1.") % __func__;
        return result;
    }
    const int max_levels = std::max(0, std::min(opts.max_levels, 7));
    const int finest_step = 1 << max_levels;                 // step size (in fine-integer coords) between same-level corners at level 0
    const int finest_nx   = opts.initial_nx * finest_step;   // total finest-coord span on x
    const int finest_ny   = opts.initial_ny * finest_step;
    const int nthreads    = std::max(1, opts.nthreads);

    // ----- Shared state -----
    // All shared mesh state lives under a single mutex `state_mu`. 
    std::unordered_map<uint64_t, float>           chi2_map;
    std::unordered_map<uint64_t, Eigen::VectorXf> bestfit_map;
    std::vector<Cell>                             cells;
    std::deque<std::unique_ptr<std::atomic<int>>> corner_counts;
    // Map from (i, j) → list of cell indices that include (i, j) as a corner.
    std::unordered_map<uint64_t, std::vector<int>> corner_to_cells;
    // Work queue: pending evaluation requests. Producer-consumer protocol with
    // state_cv: workers wait on state_cv until the queue is non-empty or all
    // work is done.
    std::deque<EvalRequest>      work;
    std::mutex                   state_mu;
    std::condition_variable      state_cv;
    std::atomic<size_t>          in_flight{0};
    std::atomic<bool>            shutdown{false};
    std::atomic<int>             total_fits{0};
    int                          fits_by_level_local[8] = {0,0,0,0,0,0,0,0};
    std::mutex                   fits_by_level_mu;

    // ----- Helpers (closures over shared state) -----

    // Insert a cell, register its corners in corner_to_cells, and initialise its
    // pending-corner count. Caller holds state_mu.
    auto register_cell = [&](Cell c) -> int {
        const int idx = static_cast<int>(cells.size());
        cells.push_back(c);
        corner_counts.push_back(std::make_unique<std::atomic<int>>(0));

        const int i0 = c.i_bl, j0 = c.j_bl, st = c.step;
        const uint64_t corner_keys[4] = {
            make_key(i0,      j0),
            make_key(i0 + st, j0),
            make_key(i0,      j0 + st),
            make_key(i0 + st, j0 + st),
        };
        // Caller holds state_mu, so chi2_map / corner_to_cells access is safe
        // without re-acquiring it. Re-acquiring here would deadlock (mutex is
        // non-recursive). Crucially, doing the chi2_map check + corner_to_cells
        // push under the SAME state_mu region the worker uses for chi2_map
        // insert + corner_to_cells lookup is what closes the B2 race.
        int pending = 0;
        for (uint64_t k : corner_keys) {
            if (chi2_map.find(k) == chi2_map.end()) ++pending;
        }
        corner_counts[idx]->store(pending, std::memory_order_relaxed);
        for (uint64_t k : corner_keys) {
            corner_to_cells[k].push_back(idx);
        }
        return idx;
    };

    // Push a new evaluation request into the work queue if (i, j) is not yet in
    // chi2_map AND not already pending in the work queue. Single state_mu
    // critical section: chi2_map check, work-queue dedup, seed assembly, and
    // queue push happen atomically. Notify is done outside the lock.
    auto enqueue_eval = [&](int i, int j,
                            const uint64_t *neighbour_keys, int n_neighbours) {
        bool pushed = false;
        {
            std::lock_guard<std::mutex> lk(state_mu);
            const uint64_t key = make_key(i, j);
            if (chi2_map.find(key) != chi2_map.end()) return;  // already evaluated
            // Avoid double-queueing: linear scan of pending work. OK for small
            // queues (<a few thousand); for larger runs swap to an
            // unordered_set<uint64_t> tracked alongside `work`.
            for (const auto &r : work) {
                if (r.key == key) return;
            }
            // Build warm-start seeds from the cell-local corners that are
            // already in bestfit_map. Edge midpoints get 2 seeds (the endpoint
            // corners); cell centers get 4. For initial-level points (no
            // neighbours), fall back to opts.initial_seed_points (e.g. the
            // caller's freq_seed_points from a prior global fit).
            std::vector<Eigen::VectorXf> seeds;
            for (int k = 0; k < n_neighbours; ++k) {
                auto it = bestfit_map.find(neighbour_keys[k]);
                if (it != bestfit_map.end()) seeds.push_back(it->second);
            }
            if (seeds.empty()) seeds = opts.initial_seed_points;

            EvalRequest req;
            req.key    = key;
            req.x_phys = i_to_x(i, finest_nx, x_lo, x_hi);
            req.y_phys = j_to_y(j, finest_ny, y_lo, y_hi);
            req.seeds  = std::move(seeds);
            work.push_back(std::move(req));
            in_flight.fetch_add(1, std::memory_order_relaxed);
            pushed = true;
        }
        if (pushed) state_cv.notify_one();
    };

    // 2:1 balance: when subdividing a cell, ensure no neighbour leaf cell is
    // more than one level coarser. If so, queue refinements for the offending
    // neighbours. This iterates implicitly through the work queue (the queued
    // refinements re-run the balance check when they themselves classify).
    // For simplicity (matches the "rare; usually 0–1 extra cells" comment in
    // the plan): we do a single pass on the immediate neighbours of the just-
    // subdivided cell.
    auto enforce_balance_around = [&](int i_bl, int j_bl, int step, int level) {
        if (!opts.balance_2to1) return;
        // The four neighbour cells of the parent (each at the same level) sit at
        // (i_bl ± step, j_bl) and (i_bl, j_bl ± step). If a neighbour leaf is
        // coarser (level - k for k >= 2), it has a larger step and its corners
        // span this side. We detect coarse neighbours by scanning the existing
        // cell list. For typical AMR runs this is rare and the cell list is
        // small enough that the linear scan is cheap.
        const int neighbour_offsets[4][2] = {
            { +step,  0    },
            { -step,  0    },
            {  0,    +step },
            {  0,    -step },
        };
        std::vector<int> to_refine;
        {
            std::lock_guard<std::mutex> lk(state_mu);
            for (int n = 0; n < 4; ++n) {
                const int ni = i_bl + neighbour_offsets[n][0];
                const int nj = j_bl + neighbour_offsets[n][1];
                if (ni < 0 || nj < 0 || ni >= finest_nx || nj >= finest_ny) continue;
                // Find a leaf cell whose bounding box contains (ni, nj) as its bottom-left
                // and is at coarser level.
                for (int c = 0; c < (int)cells.size(); ++c) {
                    if (cells[c].refined) continue;
                    if (cells[c].level >= level) continue;       // not coarser than parent
                    // After subdividing parent (level L) → children (level L+1),
                    // 2:1 balance forbids any leaf neighbour at level < L (i.e.
                    // L-1 or coarser): difference between child (L+1) and
                    // neighbour (≤ L-1) would be ≥ 2.
                    // Spatial overlap: does this coarse cell touch our just-refined cell?
                    const int cs = cells[c].step;
                    if (ni >= cells[c].i_bl && ni <= cells[c].i_bl + cs &&
                        nj >= cells[c].j_bl && nj <= cells[c].j_bl + cs) {
                        to_refine.push_back(c);
                    }
                }
            }
        }
        for (int c : to_refine) {
            // Schedule subdivision: mark refined and queue child evaluations.
            // We re-use the classify-and-subdivide path. To avoid recursion mutex
            // issues, we'll just enqueue corner-evals for the children; the
            // children's evaluations will themselves trigger classification.
            Cell parent;
            int pi, pj, plev, pst;
            {
                std::lock_guard<std::mutex> lk(state_mu);
                if (cells[c].refined) continue;
                cells[c].refined = true;
                parent = cells[c];
                pi = parent.i_bl; pj = parent.j_bl;
                plev = parent.level; pst = parent.step;
            }
            const int half = pst / 2;
            // Add the four child cells.
            int child_indices[4];
            {
                std::lock_guard<std::mutex> lk(state_mu);
                child_indices[0] = register_cell({pi,         pj,         plev + 1, half, false});
                child_indices[1] = register_cell({pi + half,  pj,         plev + 1, half, false});
                child_indices[2] = register_cell({pi,         pj + half,  plev + 1, half, false});
                child_indices[3] = register_cell({pi + half,  pj + half,  plev + 1, half, false});
            }
            // Queue the new corner evaluations (4 edge midpoints + 1 center).
            // Existing parent corners are already in chi2_map.
            const uint64_t parent_corners[4] = {
                make_key(pi,         pj),
                make_key(pi + pst,   pj),
                make_key(pi,         pj + pst),
                make_key(pi + pst,   pj + pst),
            };
            // 4 edge midpoints (2 seeds each: the two endpoint corners).
            {
                const uint64_t s[2] = { parent_corners[0], parent_corners[1] };
                enqueue_eval(pi + half, pj,         s, 2);
            }
            {
                const uint64_t s[2] = { parent_corners[2], parent_corners[3] };
                enqueue_eval(pi + half, pj + pst,   s, 2);
            }
            {
                const uint64_t s[2] = { parent_corners[0], parent_corners[2] };
                enqueue_eval(pi,        pj + half,  s, 2);
            }
            {
                const uint64_t s[2] = { parent_corners[1], parent_corners[3] };
                enqueue_eval(pi + pst,  pj + half,  s, 2);
            }
            // Center: 4 corners as seeds.
            enqueue_eval(pi + half, pj + half, parent_corners, 4);
            // Track per-level fit count.
            {
                std::lock_guard<std::mutex> lk(fits_by_level_mu);
                if (plev + 1 < 8) fits_by_level_local[plev + 1] += 5;
            }
            (void)child_indices;
        }
    };

    // Classify a cell when its 4 corners are all evaluated; if straddling, mark
    // it refined and generate children. Returns true if the cell was refined.
    auto classify_and_maybe_subdivide = [&](int cell_idx) -> bool {
        Cell c;
        {
            std::lock_guard<std::mutex> lk(state_mu);
            if (cell_idx < 0 || cell_idx >= (int)cells.size()) return false;
            if (cells[cell_idx].refined) return false;
            c = cells[cell_idx];
        }
        if (c.level >= max_levels) return false;  // can't refine further

        const uint64_t corner_keys[4] = {
            make_key(c.i_bl,           c.j_bl),
            make_key(c.i_bl + c.step,  c.j_bl),
            make_key(c.i_bl,           c.j_bl + c.step),
            make_key(c.i_bl + c.step,  c.j_bl + c.step),
        };
        float cmin = std::numeric_limits<float>::infinity();
        float cmax = -std::numeric_limits<float>::infinity();
        {
            std::lock_guard<std::mutex> lk(state_mu);
            for (uint64_t k : corner_keys) {
                auto it = chi2_map.find(k);
                if (it == chi2_map.end()) return false;  // not all corners ready
                cmin = std::min(cmin, it->second);
                cmax = std::max(cmax, it->second);
            }
        }
        // Targets are Δχ² values; offset by current global min seen so far.
        const float global_min = result.min_chi2;  // result is captured by ref later; safe here in single-threaded init? Actually we need atomic.
        // Workaround: track running min via atomic.
        // (See the running_min atomic below.)
        (void)global_min;
        // The straddle test compares raw χ² values to (running_min + target).
        // To keep the hot path lock-free we read the running min from an atomic
        // that's updated alongside state_mu; here we read a snapshot.
        std::vector<float> shifted_targets;
        shifted_targets.reserve(opts.contour_levels.size());
        // Re-acquire state_mu briefly to read running min via maps; we use the
        // smaller of the local cmin and the just-read pool min as a robust
        // global-min estimate.
        float gmin;
        {
            std::lock_guard<std::mutex> lk(state_mu);
            gmin = std::numeric_limits<float>::infinity();
            for (const auto &kv : chi2_map) gmin = std::min(gmin, kv.second);
        }
        for (float t : opts.contour_levels) shifted_targets.push_back(gmin + t);
        if (!straddles_any(cmin, cmax, shifted_targets, opts.delta_widen)) return false;

        // Subdivide: mark refined, register 4 children, enqueue 5 new corner evals.
        Cell parent;
        {
            std::lock_guard<std::mutex> lk(state_mu);
            if (cells[cell_idx].refined) return false;
            cells[cell_idx].refined = true;
            parent = cells[cell_idx];
        }
        const int pi = parent.i_bl, pj = parent.j_bl, pst = parent.step;
        const int half = pst / 2;
        {
            std::lock_guard<std::mutex> lk(state_mu);
            register_cell({pi,         pj,         parent.level + 1, half, false});
            register_cell({pi + half,  pj,         parent.level + 1, half, false});
            register_cell({pi,         pj + half,  parent.level + 1, half, false});
            register_cell({pi + half,  pj + half,  parent.level + 1, half, false});
        }
        const uint64_t parent_corners[4] = {
            make_key(pi,         pj),
            make_key(pi + pst,   pj),
            make_key(pi,         pj + pst),
            make_key(pi + pst,   pj + pst),
        };
        {
            const uint64_t s[2] = { parent_corners[0], parent_corners[1] };
            enqueue_eval(pi + half, pj,         s, 2);
        }
        {
            const uint64_t s[2] = { parent_corners[2], parent_corners[3] };
            enqueue_eval(pi + half, pj + pst,   s, 2);
        }
        {
            const uint64_t s[2] = { parent_corners[0], parent_corners[2] };
            enqueue_eval(pi,        pj + half,  s, 2);
        }
        {
            const uint64_t s[2] = { parent_corners[1], parent_corners[3] };
            enqueue_eval(pi + pst,  pj + half,  s, 2);
        }
        enqueue_eval(pi + half, pj + half, parent_corners, 4);
        {
            std::lock_guard<std::mutex> lk(fits_by_level_mu);
            if (parent.level + 1 < 8) fits_by_level_local[parent.level + 1] += 5;
        }

        // 2:1 balance pass on neighbours.
        enforce_balance_around(pi, pj, pst, parent.level);
        return true;
    };

    // ----- Phase A: Level-0 grid -----
    // Register all level-0 cells and queue their corner evaluations.
    // Initial-grid corners use opts.initial_seed_points as seeds (no neighbour context yet).
    {
        std::lock_guard<std::mutex> lk(state_mu);
        for (int ix = 0; ix < opts.initial_nx; ++ix) {
            for (int iy = 0; iy < opts.initial_ny; ++iy) {
                const int i_bl = ix * finest_step;
                const int j_bl = iy * finest_step;
                register_cell({i_bl, j_bl, 0, finest_step, false});
            }
        }
    }
    // Enqueue corner evaluations for the initial grid (deduped via chi2_map miss).
    for (int ix = 0; ix <= opts.initial_nx; ++ix) {
        for (int iy = 0; iy <= opts.initial_ny; ++iy) {
            const int i = ix * finest_step;
            const int j = iy * finest_step;
            // No neighbours yet; seeds come from opts.initial_seed_points via build_seeds_for.
            enqueue_eval(i, j, nullptr, 0);
        }
    }
    {
        std::lock_guard<std::mutex> lk(fits_by_level_mu);
        fits_by_level_local[0] = (opts.initial_nx + 1) * (opts.initial_ny + 1);
    }

    // ----- Phase B: Worker pool -----
    auto worker_loop = [&]() {
        while (true) {
            EvalRequest req;
            {
                std::unique_lock<std::mutex> lk(state_mu);
                state_cv.wait(lk, [&]{
                    return !work.empty() || in_flight.load(std::memory_order_acquire) == 0 || shutdown.load();
                });
                if (shutdown.load()) return;
                if (work.empty()) {
                    if (in_flight.load(std::memory_order_acquire) == 0) return;
                    continue;
                }
                req = std::move(work.front());
                work.pop_front();
            }

            EvalResult r = eval(req);
            total_fits.fetch_add(1, std::memory_order_relaxed);

            std::vector<int> affected_cells;
            {
                std::lock_guard<std::mutex> lk(state_mu);
                chi2_map[req.key]    = r.chi2;
                bestfit_map[req.key] = std::move(r.best_fit);
                auto it = corner_to_cells.find(req.key);
                if (it != corner_to_cells.end()) affected_cells = it->second;
            }
            for (int cidx : affected_cells) {
                const int prev = corner_counts[cidx]->fetch_sub(1, std::memory_order_acq_rel);
                if (prev <= 1) {
                    // This corner was the last one missing → cell is classifiable.
                    classify_and_maybe_subdivide(cidx);
                }
            }

            // Done with this request. Decrement in_flight and wake any sleepers.
            in_flight.fetch_sub(1, std::memory_order_acq_rel);
            state_cv.notify_all();
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(nthreads);
    for (int t = 0; t < nthreads; ++t) workers.emplace_back(worker_loop);
    for (auto &th : workers) th.join();

    // ----- Phase B+: post-pass reclassification with final gmin -----
    // During the main pipelined run, cells are classified the moment their last
    // corner lands — using whatever gmin is visible at that instant. If gmin
    // later drops (a deeper basin appears in a region evaluated afterwards),
    // earlier cells whose corners straddled the *old* (gmin + Δ) target band
    // may not straddle the *new* one, OR — more importantly for the visible
    // bug — cells that did NOT straddle the old target may now straddle the
    // new target after gmin shifted. Those cells were never refined, leaving
    // gaps where the contour exits a fully-refined region into a coarser one.
    //
    // Fix: re-run classify_and_maybe_subdivide on every still-leaf cell after
    // the main loop converges, with the final gmin (classify_and_maybe_subdivide
    // re-reads chi2_map for gmin on every call). Any cell that *now* straddles
    // schedules child evaluations; a fresh worker pool drains them. Cascading
    // refinements (children of children) are handled because the worker pool
    // drives classify_and_maybe_subdivide on each evaluation completion just
    // like in the main phase. Iterate until a sweep produces no new refinements.
    for (int reclass_pass = 0; reclass_pass < max_levels + 1; ++reclass_pass) {
        std::vector<int> leaves_to_recheck;
        {
            std::lock_guard<std::mutex> lk(state_mu);
            leaves_to_recheck.reserve(cells.size());
            for (int c = 0; c < (int)cells.size(); ++c) {
                if (!cells[c].refined && cells[c].level < max_levels) {
                    leaves_to_recheck.push_back(c);
                }
            }
        }
        int newly_refined = 0;
        for (int cidx : leaves_to_recheck) {
            if (classify_and_maybe_subdivide(cidx)) ++newly_refined;
        }
        if (newly_refined == 0) break;
        log<LOG_INFO>(L"%1% || PROmesh reclass pass %2%: %3% leaves newly refined; draining worker pool.")
            % __func__ % reclass_pass % newly_refined;
        std::vector<std::thread> post_workers;
        post_workers.reserve(nthreads);
        for (int t = 0; t < nthreads; ++t) post_workers.emplace_back(worker_loop);
        for (auto &th : post_workers) th.join();
    }

    // ----- Phase C: extract results -----
    result.chi2_map    = std::move(chi2_map);
    result.bestfit_map = std::move(bestfit_map);
    result.total_fits  = total_fits.load();
    {
        std::lock_guard<std::mutex> lk(fits_by_level_mu);
        for (int i = 0; i < 8; ++i) result.fits_by_level[i] = fits_by_level_local[i];
    }
    if (!result.chi2_map.empty()) {
        float gmin = std::numeric_limits<float>::infinity();
        for (const auto &kv : result.chi2_map) gmin = std::min(gmin, kv.second);
        result.min_chi2 = gmin;
    }
    // Expose the leaf cells + bounds for AMR mesh visualization. Caller
    // (PROsurf::PlotAMRMesh) draws each leaf as a TBox sized by its step in
    // physical coords, which gives the classic "cells shrinking around the
    // contour" picture.
    result.finest_nx = finest_nx;
    result.finest_ny = finest_ny;
    result.x_lo = x_lo; result.x_hi = x_hi;
    result.y_lo = y_lo; result.y_hi = y_hi;
    result.leaves.reserve(cells.size());
    for (const auto &c : cells) {
        if (c.refined) continue;
        MeshCell mc;
        mc.i_bl = c.i_bl;
        mc.j_bl = c.j_bl;
        mc.step = c.step;
        mc.level = c.level;
        result.leaves.push_back(mc);
    }

    // ----- Phase D: marching squares -----
    result.polylines.assign(opts.contour_levels.size(), {});
    for (size_t lvl = 0; lvl < opts.contour_levels.size(); ++lvl) {
        const float c_target = result.min_chi2 + opts.contour_levels[lvl];
        for (const auto &cell : cells) {
            if (cell.refined) continue;
            const uint64_t k00 = make_key(cell.i_bl,            cell.j_bl);
            const uint64_t k10 = make_key(cell.i_bl + cell.step, cell.j_bl);
            const uint64_t k01 = make_key(cell.i_bl,             cell.j_bl + cell.step);
            const uint64_t k11 = make_key(cell.i_bl + cell.step, cell.j_bl + cell.step);
            auto it00 = result.chi2_map.find(k00);
            auto it10 = result.chi2_map.find(k10);
            auto it01 = result.chi2_map.find(k01);
            auto it11 = result.chi2_map.find(k11);
            if (it00 == result.chi2_map.end() || it10 == result.chi2_map.end() ||
                it01 == result.chi2_map.end() || it11 == result.chi2_map.end()) continue;
            const float c00 = it00->second, c10 = it10->second, c01 = it01->second, c11 = it11->second;
            // 4-bit case index for marching squares: bit i set iff corner i is above target.
            int code = 0;
            if (c00 > c_target) code |= 1;
            if (c10 > c_target) code |= 2;
            if (c11 > c_target) code |= 4;
            if (c01 > c_target) code |= 8;
            if (code == 0 || code == 15) continue;  // no contour through this cell

            // Edge crossing positions in physical coords.
            // edge 0 = bottom (00 → 10), edge 1 = right (10 → 11),
            // edge 2 = top (01 → 11), edge 3 = left (00 → 01).
            const float x0 = i_to_x(cell.i_bl,            finest_nx, x_lo, x_hi);
            const float x1 = i_to_x(cell.i_bl + cell.step, finest_nx, x_lo, x_hi);
            const float y0 = j_to_y(cell.j_bl,             finest_ny, y_lo, y_hi);
            const float y1 = j_to_y(cell.j_bl + cell.step, finest_ny, y_lo, y_hi);
            std::pair<float,float> p_bottom{ interp_crossing(c00, c10, x0, x1, c_target), y0 };
            std::pair<float,float> p_right { x1,                                          interp_crossing(c10, c11, y0, y1, c_target) };
            std::pair<float,float> p_top   { interp_crossing(c01, c11, x0, x1, c_target), y1 };
            std::pair<float,float> p_left  { x0,                                          interp_crossing(c00, c01, y0, y1, c_target) };

            auto add_seg = [&](const std::pair<float,float> &a, const std::pair<float,float> &b) {
                ContourSegment s;
                s.p0 = a; s.p1 = b;
                result.polylines[lvl].push_back(s);
            };

            // Standard 16-case lookup table. Ambiguous cases (5, 10) split with
            // the simple "saddle resolution" rule using the cell's mean corner
            // value; for χ² scans this is rarely triggered and either choice is
            // acceptable for plotting.
            switch (code) {
                case 1:  case 14: add_seg(p_left,   p_bottom); break;
                case 2:  case 13: add_seg(p_bottom, p_right);  break;
                case 3:  case 12: add_seg(p_left,   p_right);  break;
                case 4:  case 11: add_seg(p_top,    p_right);  break;
                case 6:  case 9:  add_seg(p_bottom, p_top);    break;
                case 7:  case 8:  add_seg(p_left,   p_top);    break;
                case 5: {
                    const float mean = 0.25f * (c00 + c10 + c01 + c11);
                    if (mean > c_target) {
                        add_seg(p_left,   p_top);
                        add_seg(p_bottom, p_right);
                    } else {
                        add_seg(p_left,   p_bottom);
                        add_seg(p_top,    p_right);
                    }
                    break;
                }
                case 10: {
                    const float mean = 0.25f * (c00 + c10 + c01 + c11);
                    if (mean > c_target) {
                        add_seg(p_left,   p_bottom);
                        add_seg(p_top,    p_right);
                    } else {
                        add_seg(p_left,   p_top);
                        add_seg(p_bottom, p_right);
                    }
                    break;
                }
                default: break;
            }
        }
    }

    // ----- Phase E: optional bilinear-reconstructed dense matrix -----
    if (opts.produce_dense && opts.dense_nx > 0 && opts.dense_ny > 0 && !cells.empty()) {
        result.reconstructed_dense = Eigen::MatrixXf::Zero(opts.dense_ny, opts.dense_nx);
        for (int dy = 0; dy < opts.dense_ny; ++dy) {
            for (int dx = 0; dx < opts.dense_nx; ++dx) {
                // Physical position of the center of dense cell (dx, dy).
                const float xfrac = (dx + 0.5f) / static_cast<float>(opts.dense_nx);
                const float yfrac = (dy + 0.5f) / static_cast<float>(opts.dense_ny);
                const int i_q = static_cast<int>(std::floor(xfrac * finest_nx));
                const int j_q = static_cast<int>(std::floor(yfrac * finest_ny));

                // Find the smallest leaf cell containing (i_q, j_q).
                const Cell *containing = nullptr;
                int best_level = -1;
                for (const auto &c : cells) {
                    if (c.refined) continue;
                    if (i_q >= c.i_bl && i_q <= c.i_bl + c.step &&
                        j_q >= c.j_bl && j_q <= c.j_bl + c.step) {
                        if (c.level > best_level) { containing = &c; best_level = c.level; }
                    }
                }
                if (!containing) continue;
                // Bilinear interp from cell corners.
                const uint64_t k00 = make_key(containing->i_bl,                     containing->j_bl);
                const uint64_t k10 = make_key(containing->i_bl + containing->step,  containing->j_bl);
                const uint64_t k01 = make_key(containing->i_bl,                     containing->j_bl + containing->step);
                const uint64_t k11 = make_key(containing->i_bl + containing->step,  containing->j_bl + containing->step);
                auto it00 = result.chi2_map.find(k00);
                auto it10 = result.chi2_map.find(k10);
                auto it01 = result.chi2_map.find(k01);
                auto it11 = result.chi2_map.find(k11);
                if (it00 == result.chi2_map.end() || it10 == result.chi2_map.end() ||
                    it01 == result.chi2_map.end() || it11 == result.chi2_map.end()) continue;
                const float xfrac_local = (i_q - containing->i_bl) / static_cast<float>(containing->step);
                const float yfrac_local = (j_q - containing->j_bl) / static_cast<float>(containing->step);
                const float c00 = it00->second, c10 = it10->second, c01 = it01->second, c11 = it11->second;
                const float c_top    = c01 * (1.0f - xfrac_local) + c11 * xfrac_local;
                const float c_bottom = c00 * (1.0f - xfrac_local) + c10 * xfrac_local;
                const float c        = c_bottom * (1.0f - yfrac_local) + c_top * yfrac_local;
                result.reconstructed_dense(dy, dx) = c - result.min_chi2;
            }
        }
    }

    log<LOG_INFO>(L"%1% || PROmesh: %2% total fits, levels [%3%, %4%, %5%, %6%], min_chi2=%7%")
        % __func__ % result.total_fits
        % result.fits_by_level[0] % result.fits_by_level[1]
        % result.fits_by_level[2] % result.fits_by_level[3]
        % result.min_chi2;

    return result;
}

}  // namespace PROmesh
}  // namespace PROfit
