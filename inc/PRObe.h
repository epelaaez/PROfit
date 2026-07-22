/**
 * @file PRObe.h
 * @brief Adaptive importance sampler for 1D chi-squared crossings, plus the shared
 *        scan-point store / fit-executor abstractions used by all PROfile scans.
 * @author PROfit Collaboration
 *
 * @details Two layers live here:
 *
 * **Shared scan infrastructure** (namespace PROfit): `ScanPoint`,
 * `ScanPointStore` and `ScanFitContext`. A ScanPointStore is the single home of
 * every completed scan-point fit of one parameter — PROfile backs it with its
 * cross-thread seed_bank, so warm-starting, deduplication, PRObe's surrogate
 * inputs, and cross-chunk knowledge all read from ONE store instead of the two
 * parallel systems that existed before (PRObe's task-local SeedPool vs
 * PROfile's seed_bank). `ScanFitContext::fitAt` is the single "fit the profile
 * at fixed θ" primitive shared by the legacy 18-point scan and PRObe: it pins
 * the scanned axis via setBounds, warm-starts from the caller's global seeds
 * plus the store's nearest-by-θ point, guards exceptions and non-finite chi²
 * (a failed fit is logged and NOT stored), deposits accepted fits into the
 * store, and fires the optional per-fit progress callback.
 *
 * **PRObe** (namespace PROfit::PRObe): `chi2_crossing_1d` locates the
 * boundaries where Δχ² = target along a single parameter direction without
 * dense uniform scanning, running entirely through a ScanFitContext.
 *
 * **Nuisance path** (`may_have_spikes = false`). For Gaussian-like parameters
 * whose chi² is well-approximated by a parabola near the minimum:
 *   1. Phase 1 — anchor fits at `bf_value + sigma_init * anchor_sigmas`
 *                (7 offsets by default), plus one anchor per global seed whose
 *                `param_idx` value lies inside the scan range.
 *   2. Phase 2 — quadratic surrogate fit through the nearest 5 store points
 *                around the store minimum. Requires at least 4 points: a
 *                3-point quadratic interpolates exactly, so its residual test
 *                is vacuous.
 *   3. Phase 3 — if the surrogate is accepted (residual ≤ `quadratic_residual_max`
 *                and no store point lies above the surrogate by more than
 *                `spike_chi2_threshold`), predict the Δχ² = target crossings
 *                analytically and run two confirmation fits there.
 *   4. Phase 4 — fallback bracket + bisection on either side of the minimum if
 *                the surrogate is rejected (or a confirmation misses by more
 *                than `boundary_tol_chi2`).
 *   Typical cost: ~11–14 fits when the surrogate accepts; up to `max_fits`
 *   when the bisection fallback fires.
 *
 * **Physics path** (`may_have_spikes = true`). For non-Gaussian parameters
 * (asymmetric basins, sharp minima, multi-modal structure):
 *   1. Sample `coarse_n` evenly-spaced grid points across [plb, pub]
 *      (default 10), plus one anchor at `bf_value` and one per in-range
 *      global seed. No surrogate, no spike test.
 *   2. Find the store minimum and walk neighbours on each side until a pair
 *      straddles `min_chi² + target_dchi2`. Run **one** linear-interpolated
 *      refine fit at the predicted crossing.
 *   Predictable cost: `coarse_n + 1 + |seeds_in_range| + ≤ 2` fits per
 *   scan/chunk.
 *
 * Because the store is shared across chunks (and threads), a chunked physics
 * scan sees the other chunks' completed fits: the Δχ² target references the
 * parameter-global minimum, duplicate points at chunk boundaries are deduped
 * at the source, and warm starts cross chunk edges. Spline scans are a single
 * task, so their store view is filled only by themselves — deterministic
 * regardless of thread count.
 *
 * The per-fit minimisation runs through `PROfitter::Fit()` (full Latin + PSO +
 * multi-LBFGS pipeline). A leaner `PROfitter::FitScan()` exists for scan-mode
 * callers but is currently unused — disabled while a fit-quality regression in
 * that path is investigated (see ScanFitContext::fitAt).
 */
#ifndef PROBE_H
#define PROBE_H

#include "PROmetric.h"
#include "PROfitter.h"

#include <Eigen/Eigen>

#include <cmath>
#include <cstdint>
#include <functional>
#include <vector>

namespace PROfit {

    /// One completed scan-point fit: the scanned parameter's pinned value, the
    /// profiled chi² there, and the full best-fit parameter vector.
    struct ScanPoint {
        float value = 0;            ///< Scanned parameter value (fit-space units).
        float chi2 = 0;             ///< Absolute profiled chi² at that value.
        Eigen::VectorXf best_fit;   ///< Full (model + spline) best-fit vector.
    };

    /// Substitute a finite default for an infinite bound (scan sampling needs a
    /// finite range; the fit itself may keep the infinite bound on other axes).
    inline float finite_or(float v, float def) { return std::isinf(v) ? def : v; }

    /**
     * @brief Interface to the per-parameter store of completed scan-point fits.
     * @details Implementations must be safe for concurrent access when tasks of
     * the same parameter can run on different threads (PROfile backs this with
     * its mutex-guarded cross-thread seed_bank). All methods refer to a single
     * scanned parameter — the binding happens at construction of the concrete
     * view.
     */
    struct ScanPointStore {
        virtual ~ScanPointStore() = default;
        /// Copy of all completed points of this parameter (any order).
        virtual std::vector<ScanPoint> snapshot() const = 0;
        /// Best-fit vector of the point nearest in value; empty if store is empty.
        virtual Eigen::VectorXf nearest_bf(float value) const = 0;
        /// True if a completed point lies within tol of value.
        virtual bool has_close(float value, float tol) const = 0;
        /// Deposit a completed fit.
        virtual void add(const ScanPoint &pt) = 0;
    };

    /**
     * @brief The single "profile fit at fixed θ" primitive shared by the legacy
     * 18-point scan and PRObe.
     * @details fitAt pins the scanned axis (tlb[param_idx] = tub[param_idx] = θ
     * via setBounds — the sole pinning mechanism; the metrics' legacy
     * fixSpline state is write-only and no longer set), assembles the seed
     * list as global_seeds + the store's nearest-by-θ best fit, runs
     * PROfitter::Fit with RNG seed `base_seed + call_count` (per-fit, so a
     * scan is reproducible and no two fits of a task share a seed), guards
     * exceptions and non-finite chi² (logged; `ok=false`; nothing stored),
     * deposits accepted fits into the store, and fires on_fit.
     */
    struct ScanFitContext {
        PROmetric &metric;                    ///< Thread-local metric clone.
        Eigen::VectorXf full_lb;              ///< Bounds for the whole vector; scanned axis overridden per fit.
        Eigen::VectorXf full_ub;
        size_t param_idx;                     ///< Scanned parameter (full-vector index).
        const PROfitterConfig &fitconfig;
        uint32_t base_seed;                   ///< Per-fit RNG seed = base_seed + call_count.
        const std::vector<Eigen::VectorXf> &global_seeds; ///< Harmonic/global seeds, passed to every fit.
        ScanPointStore &store;                ///< Shared per-parameter store (see ScanPointStore).
        std::function<void()> on_fit;         ///< Optional per-fit progress callback (may be empty).
        int call_count = 0;                   ///< Fits attempted so far (advances the RNG sequence).

        struct Outcome {
            bool ok = false;   ///< False on caught exception or non-finite chi².
            ScanPoint pt;      ///< Valid only when ok.
        };
        Outcome fitAt(float value);
    };

namespace PRObe {

    /// Configuration knobs for chi2_crossing_1d.
    struct CrossingOpts {
        float target_dchi2          = 1.0f;     ///< Δχ² level to find crossings at (1 → 1σ; 4 → 2σ).
        bool  may_have_spikes       = false;    ///< If true, use the coarse-grid physics path instead of the surrogate.
        int   coarse_n              = 10;       ///< Physics-path grid point count (uniform across [lb, ub]).
        std::vector<float> anchor_sigmas = {0.0f, -0.4f, 0.4f, -0.8f, 0.8f, -1.45f, 1.45f}; ///< Nuisance-path anchor offsets in σ (7 by default → ~11-14 fits for legible TGraphs).
        float sigma_init            = 1.0f;     ///< Caller-provided initial σ scale for the parameter.
        float boundary_tol_chi2     = 0.10f;    ///< Bisection stop tolerance on |χ² - target|.
        int   max_fits              = 25;       ///< Hard cap on total fits per scan.
        int   max_bisect_iter       = 6;        ///< Per-side bisection iteration cap.
        int   max_bisect_extends    = 5;        ///< Per-side outward-extension cap while bracketing (a second, independent limit from max_bisect_iter).
        float spike_chi2_threshold  = 0.5f;     ///< Store point lying this far above the surrogate flags a spike.
        float quadratic_residual_max = 0.15f;   ///< Max allowed surrogate residual (chi² units) for acceptance (meaningful only with ≥4 points; enforced).
    };

    /// Output bundle from chi2_crossing_1d: the fits THIS call performed
    /// (task-local — chunk merging must not duplicate other chunks' points),
    /// parallel arrays sorted ascending by theta.
    struct CrossingResult {
        std::vector<float> theta;                   ///< Fitted parameter values (sorted ascending).
        std::vector<float> chi2;                    ///< Absolute χ² at each theta.
        std::vector<Eigen::VectorXf> best_fits;     ///< Full best-fit vectors at each theta.
        bool  used_surrogate = false;               ///< True if the Phase-3 surrogate produced the crossings.
        int   n_fits = 0;                           ///< Fits attempted by this call (including failed ones).
    };

    /**
     * @brief Locate Δχ² = target crossings along one parameter axis.
     * @details Branches on `opts.may_have_spikes` between the surrogate-driven
     * nuisance path and the grid + linear-refine physics path (see the
     * file-level header). All fitting, seeding, deduplication, and progress
     * reporting run through @p ctx; algorithmic decisions (minimum location,
     * surrogate inputs, straddle tests) consult the shared store, so
     * concurrent chunks of the same parameter cooperate.
     *
     * @param ctx       Shared fit executor bound to the scanned parameter. Its
     *                  full_lb/full_ub define the scan range on the scanned
     *                  axis (±inf sampled as ±3).
     * @param bf_value  Anchor centre: the parameter's global best-fit value, or
     *                  the chunk midpoint when the BF lies outside a chunk.
     * @param opts      Tunables — see CrossingOpts.
     * @return          The fits performed by THIS call, sorted by theta.
     */
    CrossingResult chi2_crossing_1d(
        ScanFitContext &ctx,
        float bf_value,
        const CrossingOpts &opts);

}
}

#endif
