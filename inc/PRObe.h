/**
 * @file PRObe.h
 * @brief Adaptive importance sampler for 1D chi-squared crossings (PROfile / PROsurf primitive).
 * @author PROfit Collaboration
 *
 * @details PRObe locates the boundaries where Δχ² = target along a single parameter
 * direction without dense uniform scanning. `chi2_crossing_1d` selects between two
 * algorithm paths based on the chi² shape expected for the parameter, gated by
 * `CrossingOpts::may_have_spikes`. Both paths share an `add_fit` primitive that
 * dedups by parameter value, enforces `max_fits`, fires the optional `on_fit`
 * progress callback, and records each completed fit in a per-scan SeedPool of
 * (theta, best_fit, chi²) tuples.
 *
 * **Nuisance path** (`may_have_spikes = false`). For Gaussian-like parameters
 * whose chi² is well-approximated by a parabola near the minimum:
 *   1. Phase 1 — anchor fits at `bf_value + sigma_init * anchor_sigmas`, plus
 *                one anchor per global seed whose `param_idx` value lies inside
 *                the scan range.
 *   2. Phase 2 — quadratic surrogate fit through the nearest 5 pool points
 *                around the in-pool minimum.
 *   3. Phase 3 — if the surrogate is accepted (residual ≤ `quadratic_residual_max`
 *                and no pool point lies above the surrogate by more than
 *                `spike_chi2_threshold`), predict the Δχ² = target crossings
 *                analytically and run two confirmation fits there. Confirmations
 *                landing within `boundary_tol_chi2` of the target are taken as
 *                the band edges.
 *   4. Phase 4 — fallback bracket + bisection on either side of the minimum if
 *                the surrogate is rejected (or if a confirmation misses by more
 *                than `boundary_tol_chi2`). Bisection extends outward until a
 *                pool point sits above target, then halves the bracket up to
 *                `max_bisect_iter` times per side.
 *   Typical cost: ~11–14 fits when the surrogate accepts; up to ~25 when the
 *   bisection fallback fires (hard-capped by `max_fits`).
 *
 * **Physics path** (`may_have_spikes = true`). For non-Gaussian parameters
 * (asymmetric basins, sharp minima, multi-modal structure) where the surrogate
 * above rejects too often and the bisection fallback dominates wall time:
 *   1. Sample `coarse_n` evenly-spaced grid points across [plb, pub], plus one
 *      anchor at `bf_value` and one per in-range global seed. No `anchor_sigmas`,
 *      no surrogate, no spike test.
 *   2. Find the in-pool minimum and walk neighbours on each side until a pair
 *      straddles `min_chi² + target_dchi2`. Run **one** linear-interpolated
 *      refine fit at the predicted crossing.
 *   Predictable cost: `coarse_n + 1 + |seeds_in_range| + ≤ 2 ≈ 9–13 fits`.
 *
 * Each new fit is warm-started with the caller's `global_seeds` plus the
 * closest-by-theta SeedPool entry once the pool is non-empty. This preserves
 * convergence quality across the scan even when fits arrive in non-monotonic
 * order (anchors, refines, bisection midpoints).
 *
 * The per-fit minimisation runs through `PROfitter::Fit()` (full Latin + PSO +
 * multi-LBFGS pipeline). A leaner `PROfitter::FitScan()` exists for scan-mode
 * callers but is currently unused — disabled while a fit-quality regression in
 * that path is investigated.
 */
#ifndef PROBE_H
#define PROBE_H

#include "PROmetric.h"
#include "PROfitter.h"

#include <Eigen/Eigen>

#include <cstdint>
#include <functional>
#include <vector>

namespace PROfit {
namespace PRObe {

    /// Configuration knobs for chi2_crossing_1d.
    struct CrossingOpts {
        float target_dchi2          = 1.0f;     ///< Δχ² level to find crossings at (1 → 1σ; 4 → 2σ).
        bool  may_have_spikes       = false;    ///< If true, run Phase-0 coarse exploration before anchors.
        int   coarse_n              = 7;        ///< Phase-0 point count (uniform across [lb, ub]).
        std::vector<float> anchor_sigmas = {0.0f, -0.4f, 0.4f, -0.8f, 0.8f, -1.3f, 1.3f, -2.0f, 2.0f}; ///< Anchor offsets in σ. Default gives ~11 fits in smooth mode for legible TGraphs.
        float sigma_init            = 1.0f;     ///< Caller-provided initial σ scale for the parameter.
        float boundary_tol_chi2     = 0.10f;    ///< Bisection stop tolerance on |χ² - target|. Loosened from 0.05 to skip secondary refinement when surrogate confirmation already lands within 0.1 chi² — saves ~75 fits per run with no visible band-quality impact for ±1σ.
        int   max_fits              = 25;       ///< Hard cap on total fits per scan.
        int   max_bisect_iter       = 6;        ///< Per-side bisection iteration cap.
        float spike_chi2_threshold  = 0.5f;     ///< Pool point lying this far above the surrogate flags a spike.
        float quadratic_residual_max = 0.15f;   ///< Max allowed surrogate residual (chi² units) for acceptance.
        /// Optional callback invoked once after each successful fit. Lets the
        /// caller drive a per-fit progress bar live during a long
        /// chi2_crossing_1d scan (without it the bar only updates after the
        /// whole scan returns, giving a misleading "stalled" appearance).
        /// Must be thread-safe if multiple chi2_crossing_1d calls share state.
        std::function<void()> on_fit = nullptr;
    };

    /// Output bundle from chi2_crossing_1d. Pool is sorted by theta on return.
    struct CrossingResult {
        std::vector<float> theta;                   ///< All fitted parameter values (sorted ascending).
        std::vector<float> chi2;                    ///< Absolute χ² at each theta.
        std::vector<Eigen::VectorXf> best_fits;     ///< Full best-fit vectors at each theta.
        float minX  = 0.0f;                         ///< θ at the in-pool minimum χ².
        float leftX = 0.0f;                         ///< Lower Δχ² = target crossing (linear interp).
        float rightX = 0.0f;                        ///< Upper Δχ² = target crossing (linear interp).
        bool  used_surrogate = false;               ///< True if Phase-3 surrogate path produced the crossings.
        int   n_fits = 0;                           ///< Total full minimisations performed.
    };

    /**
     * @brief Locate Δχ² = target crossings along one parameter axis.
     * @details Branches on `opts.may_have_spikes` between the surrogate-driven
     * nuisance path and the grid + linear-refine physics path. See the
     * file-level header for the algorithm walk-through and which CrossingOpts
     * knobs apply to which path.
     *
     * @param metric        Metric clone owned by the caller. PRObe mutates its
     *                      `setBounds` (per-fit tlb/tub) and `fixSpline` state
     *                      across the scan.
     * @param param_idx     Index of the scanned parameter in the full
     *                      (model + spline) parameter vector.
     * @param full_lb       Lower bounds for the entire param vector. Copied into
     *                      every per-fit local bounds vector with
     *                      `tlb[param_idx]` = current θ.
     * @param full_ub       Upper bounds (same shape).
     * @param bf_value      Anchor centre. For an unchunked scan this is the
     *                      parameter's value at the global best fit. For a
     *                      chunked physics scan the dispatcher uses the chunk
     *                      midpoint when the global BF lies outside the chunk.
     * @param global_seeds  Caller's seed_points (typically
     *                      `PROfitter::freq_seed_points`). Seeds whose
     *                      `param_idx` value falls inside the scan range each
     *                      become an additional anchor fit; the full list is
     *                      also passed verbatim to every per-fit `Fit()` for
     *                      LBFGS warm-starting (no dedup at this level).
     * @param fitconfig     PROfitter configuration (Latin / PSO / LBFGS knobs).
     * @param base_seed     Base RNG seed. Each internal fit uses
     *                      `base_seed + n_call` so a single
     *                      `chi2_crossing_1d` invocation is reproducible run
     *                      to run.
     * @param opts          Tunables — see CrossingOpts.
     * @return              CrossingResult with the pool sorted by θ, plus the
     *                      linear-interpolated leftX / minX / rightX band.
     */
    CrossingResult chi2_crossing_1d(
        PROmetric &metric,
        size_t param_idx,
        const Eigen::VectorXf &full_lb,
        const Eigen::VectorXf &full_ub,
        float bf_value,
        const std::vector<Eigen::VectorXf> &global_seeds,
        const PROfitterConfig &fitconfig,
        uint32_t base_seed,
        const CrossingOpts &opts);

}
}

#endif
