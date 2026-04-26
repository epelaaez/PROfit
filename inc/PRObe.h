/**
 * @file PRObe.h
 * @brief Adaptive importance sampler for 1D chi-squared crossings (PROfile / PROsurf primitive).
 * @author PROfit Collaboration
 *
 * @details PRObe locates the boundaries where Δχ² = target along a single parameter
 * direction without dense uniform scanning. The default 1σ application solves
 * Δχ² = 1 with ~7 fits in the smooth case and falls back to bracket-and-bisect
 * sampling for non-parabolic / spiky chi² shapes.
 *
 * Algorithm (chi2_crossing_1d):
 *   Phase 0 — coarse exploration (spiky mode only): N uniform points across [lb, ub]
 *             so isolated spikes / secondary basins are surfaced.
 *   Phase 1 — anchor fits at bf_value + sigma_init * {opts.anchor_sigmas}.
 *   Phase 2 — quadratic surrogate through the nearest few pool points.
 *   Phase 3 — if the surrogate is adequate, run two confirmation fits at the
 *             analytic Δχ² = target crossings.
 *   Phase 4 — fallback bracket + bisection on either side of the minimum if the
 *             surrogate is rejected (residuals too large, downward parabola, or
 *             a spike was detected).
 *
 * A SeedPool of (theta, best_fit, chi2) records every completed fit; each new
 * fit is warm-started with the global seeds plus the closest-by-theta and
 * lowest-chi2 best-fits in the pool. This preserves the warm-start property
 * that the existing 18-uniform scan relies on, even when sample order is
 * non-monotonic.
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
     * @param metric        Metric clone owned by caller; PRObe mutates its bounds and fixSpline state.
     * @param param_idx     Index of the scanned parameter inside the full param vector.
     * @param full_lb       Lower bounds for the entire param vector at call time (will be cloned per fit).
     * @param full_ub       Upper bounds (same shape).
     * @param bf_value      The parameter's value at the global best fit (anchor centre).
     * @param global_seeds  Seed_points from the caller (e.g. fitter.freq_seed_points).
     * @param fitconfig     PROfitter configuration (Latin / PSO / LBFGS settings).
     * @param base_seed     Base RNG seed; each internal fit increments it for reproducibility.
     * @param opts          Tunables.
     * @return              CrossingResult with sorted theta/chi2/best_fits and crossing locations.
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
