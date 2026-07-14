#include "PRObe.h"
#include "PROlog.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>

namespace PROfit {

// ---------------------------------------------------------------------------
// ScanFitContext — the single shared "profile fit at fixed θ" primitive.
// ---------------------------------------------------------------------------
ScanFitContext::Outcome ScanFitContext::fitAt(float value) {
    Outcome out;
    Eigen::VectorXf tlb = full_lb;
    Eigen::VectorXf tub = full_ub;
    tlb((int)param_idx) = value;
    tub((int)param_idx) = value;

    // Seed with the caller's global/harmonic seeds plus the store's
    // nearest-by-θ completed fit (the strongest warm start; each extra seed is
    // one more LBFGS pass, so we add exactly one).
    std::vector<Eigen::VectorXf> seeds = global_seeds;
    Eigen::VectorXf near_bf = store.nearest_bf(value);
    if(near_bf.size() > 0) seeds.push_back(std::move(near_bf));

    // Per-fit RNG seed: advances even on failure so the sequence stays
    // reproducible regardless of which fits succeed.
    const uint32_t fit_seed = base_seed + (uint32_t)call_count;
    ++call_count;

    try {
        metric.setBounds(tlb, tub);
        PROfitter fitter(tub, tlb, fitconfig, fit_seed);
        // NOTE: PROfitter::FitScan (leaner scan-mode pipeline) is deliberately
        // unused while a fit-quality regression in it is investigated; every
        // scan sub-fit runs the full Fit() pipeline.
        const float chi = seeds.empty() ? fitter.Fit(metric)
                                        : fitter.Fit(metric, seeds);
        if(!std::isfinite(chi)) {
            log<LOG_WARNING>(L"%1% || Scan fit at param %2% = %3% returned non-finite chi2 (%4%); point skipped.")
                % __func__ % param_idx % value % chi;
        } else {
            out.ok = true;
            out.pt.value = value;
            out.pt.chi2 = chi;
            out.pt.best_fit = fitter.best_fit;
            store.add(out.pt);
        }
    } catch(const std::exception &e) {
        log<LOG_WARNING>(L"%1% || Scan fit at param %2% = %3% threw ('%4%'); point skipped.")
            % __func__ % param_idx % value % e.what();
    }

    if(on_fit) on_fit();
    return out;
}

namespace PRObe {

namespace {

/// Quadratic surrogate χ²(θ) ≈ a + b·(θ-θ₀) + c·(θ-θ₀)² fit by least squares.
struct QuadFit {
    float a = 0, b = 0, c = 0; ///< Coefficients about anchor θ₀.
    float anchor = 0;          ///< θ₀.
    float max_residual = 0;    ///< Max |fit - data| across the points used.
    int   n_points = 0;        ///< Points actually used in the fit.
    bool  ok = false;          ///< Upward-opening (c>0), finite, ≥4 points (a 3-point quadratic interpolates exactly, making the residual test vacuous).
};

QuadFit fit_quadratic(const std::vector<ScanPoint> &pts, float theta_anchor, size_t max_pts) {
    QuadFit q;
    q.anchor = theta_anchor;
    if (pts.size() < 4) return q;

    // Pick up to max_pts points nearest the anchor in θ.
    std::vector<size_t> idxs(pts.size());
    std::iota(idxs.begin(), idxs.end(), 0);
    std::sort(idxs.begin(), idxs.end(), [&](size_t a, size_t b) {
        return std::fabs(pts[a].value - theta_anchor) <
               std::fabs(pts[b].value - theta_anchor);
    });
    if (idxs.size() > max_pts) idxs.resize(max_pts);
    if (idxs.size() < 4) return q;

    Eigen::MatrixXf A((int)idxs.size(), 3);
    Eigen::VectorXf y((int)idxs.size());
    for (size_t i = 0; i < idxs.size(); ++i) {
        float dt = pts[idxs[i]].value - theta_anchor;
        A((int)i, 0) = 1.0f;
        A((int)i, 1) = dt;
        A((int)i, 2) = dt * dt;
        y((int)i)    = pts[idxs[i]].chi2;
    }
    Eigen::VectorXf coef = A.colPivHouseholderQr().solve(y);
    q.a = coef(0); q.b = coef(1); q.c = coef(2);
    Eigen::VectorXf res = (A * coef - y).cwiseAbs();
    q.max_residual = res.maxCoeff();
    q.n_points = (int)idxs.size();
    q.ok = (q.c > 0.0f) && std::isfinite(q.a) && std::isfinite(q.b) && std::isfinite(q.c)
        && q.n_points >= 4;
    return q;
}

/// Spike test: any point lying threshold-above the surrogate flags a spike.
/// Sub-surrogate dips do not flag (a basin found below the quadratic prediction
/// is interesting but not what spike detection is for).
bool detect_spike(const std::vector<ScanPoint> &pts, const QuadFit &q, float threshold) {
    if (!q.ok) return false;
    for (const auto &p : pts) {
        float dt = p.value - q.anchor;
        float pred = q.a + q.b * dt + q.c * dt * dt;
        if (p.chi2 - pred > threshold) return true;
    }
    return false;
}

float clamp_to(float x, float lo, float hi) {
    return std::max(lo, std::min(hi, x));
}

/// Index of the minimum-chi2 point; pts must be non-empty (chi2 are finite by
/// construction — ScanFitContext::fitAt never stores non-finite values).
size_t argmin_chi2(const std::vector<ScanPoint> &pts) {
    size_t best = 0;
    for (size_t i = 1; i < pts.size(); ++i)
        if (pts[i].chi2 < pts[best].chi2) best = i;
    return best;
}

/// Chi2 of the point nearest in value; pts must be non-empty.
float chi2_near(const std::vector<ScanPoint> &pts, float v) {
    size_t best = 0;
    float bestd = std::fabs(pts[0].value - v);
    for (size_t i = 1; i < pts.size(); ++i) {
        float d = std::fabs(pts[i].value - v);
        if (d < bestd) { bestd = d; best = i; }
    }
    return pts[best].chi2;
}

} // anonymous namespace

CrossingResult chi2_crossing_1d(
    ScanFitContext &ctx,
    float bf_value,
    const CrossingOpts &opts)
{
    CrossingResult result;
    const size_t param_idx = ctx.param_idx;

    const float plb_raw = ctx.full_lb((int)param_idx);
    const float pub_raw = ctx.full_ub((int)param_idx);
    if (!(pub_raw > plb_raw)) {
        log<LOG_WARNING>(L"%1% || PRObe: param %2% has lb >= ub (%3% >= %4%); nothing to scan.")
            % __func__ % param_idx % plb_raw % pub_raw;
        return result;
    }
    // Effective sampling bounds: substitute ±3 for ±inf so coarse sampling and
    // bisection step sizes are finite. Other parameters keep their original
    // (possibly infinite) bounds in the fit; the scanned axis is pinned to a
    // single value per fit inside ScanFitContext::fitAt.
    const float plb = finite_or(plb_raw, -3.0f);
    const float pub = finite_or(pub_raw,  3.0f);
    if (std::isinf(plb_raw) || std::isinf(pub_raw)) {
        log<LOG_DEBUG>(L"%1% || PRObe param %2%: substituting finite sampling bounds [%3%, %4%] for raw [%5%, %6%]")
            % __func__ % param_idx % plb % pub % plb_raw % pub_raw;
    }
    const float duplicate_tol = std::max(1e-4f, 1e-3f * (pub - plb));

    // Fits performed by THIS call (task-local) — the emitted result. The shared
    // store additionally holds other chunks'/tasks' points; algorithmic
    // decisions below consult the store, the emit uses only these.
    std::vector<ScanPoint> my_fits;
    const int start_count = ctx.call_count;

    // Attempt one fit: clamp into the sampling range, dedup against the SHARED
    // store (so chunk-boundary points fitted by a neighbouring chunk are not
    // re-fit and never duplicated in the merged graph), respect max_fits.
    auto add_fit = [&](float theta) -> bool {
        theta = clamp_to(theta, plb, pub);
        if (ctx.store.has_close(theta, duplicate_tol)) return false;
        if (ctx.call_count - start_count >= opts.max_fits) return false;
        auto out = ctx.fitAt(theta);
        if (!out.ok) return false;
        my_fits.push_back(std::move(out.pt));
        return true;
    };

    // Snapshot of everything known about this parameter (ours + other chunks').
    auto known = [&]() { return ctx.store.snapshot(); };

    // ---- Physics branch (may_have_spikes=true): sample-then-linear-refine ----
    // Physics chi² is non-Gaussian (asymmetric basins, sharp minima, flat tails),
    // so the quadratic surrogate path used for nuisances rejects often and falls
    // back to bracket+bisect, which blows up to max_fits. Instead: sample a small
    // evenly-spaced grid covering [plb, pub], augment with the BF and any global
    // seed values that fall inside the chunk, then for each side of the minimum
    // do **one** linear-interp refinement at the predicted Δχ²=target crossing.
    if (opts.may_have_spikes) {
        const int n_grid = std::max(3, opts.coarse_n);
        for (int i = 0; i < n_grid; ++i) {
            const float frac = (n_grid > 1) ? (float)i / (float)(n_grid - 1) : 0.5f;
            add_fit(plb + (pub - plb) * frac);
        }
        // Anchor at the BF value (the global BF in-chunk, else the chunk midpoint).
        add_fit(bf_value);
        // Seeds whose scanned-param value falls in this chunk — seeds secondary
        // basins. Out-of-chunk seeds are dropped here (the full global_seeds
        // list still warm-starts every fit inside fitAt).
        for (const auto &gs : ctx.global_seeds) {
            if ((long)gs.size() > (long)param_idx) {
                const float v = gs((int)param_idx);
                if (v >= plb && v <= pub) add_fit(v);
            }
        }

        std::vector<ScanPoint> pts = known();
        if (pts.empty()) {
            log<LOG_ERROR>(L"%1% || PRObe-phys: no valid fits for param %2%") % __func__ % param_idx;
            return result;
        }
        // Sort the known points by theta to walk neighbours. Includes other
        // chunks' points, so the target references the parameter-GLOBAL minimum
        // and a straddle across a chunk edge is still seen.
        std::sort(pts.begin(), pts.end(),
                  [](const ScanPoint &a, const ScanPoint &b) { return a.value < b.value; });
        size_t imin = 0;
        for (size_t i = 1; i < pts.size(); ++i)
            if (pts[i].chi2 < pts[imin].chi2) imin = i;
        const float min_chi2_phys = pts[imin].chi2;
        const float target_chi2   = min_chi2_phys + opts.target_dchi2;

        // Refine: walk outward on each side of the min; the first adjacent pair
        // that straddles target gets one linear-interp fit. If no straddle is
        // found, the band edge falls at the chunk boundary — the neighbouring
        // chunk picks up the actual crossing.
        auto refine_side = [&](bool right_side) {
            const ssize_t step = right_side ? +1 : -1;
            for (ssize_t i = (ssize_t)imin;
                 (step > 0 ? (i + 1 < (ssize_t)pts.size()) : (i > 0));
                 i += step)
            {
                const ssize_t j = i + step;
                const float c_inner = pts[(size_t)i].chi2;
                const float c_outer = pts[(size_t)j].chi2;
                if ((c_inner - target_chi2) * (c_outer - target_chi2) <= 0.0f) {
                    const float t_inner = pts[(size_t)i].value;
                    const float t_outer = pts[(size_t)j].value;
                    const float denom = c_outer - c_inner;
                    if (std::fabs(denom) > 1e-9f) {
                        const float frac = (target_chi2 - c_inner) / denom;
                        add_fit(t_inner + frac * (t_outer - t_inner));
                    }
                    return;
                }
            }
        };
        refine_side(false); // left of min
        refine_side(true);  // right of min

        result.used_surrogate = false;
        log<LOG_INFO>(L"%1% || PRObe-phys param %2%: %3% fits this chunk, known min chi2=%4%")
            % __func__ % param_idx % (ctx.call_count - start_count) % min_chi2_phys;
    }
    // ---- Nuisance branch: anchors → surrogate → confirm/bisect ----
    else {
        // Phase 1: anchors around the prior best-fit value (plus in-range seeds).
        for (float s : opts.anchor_sigmas) {
            add_fit(bf_value + s * opts.sigma_init);
        }
        for (const auto &gs : ctx.global_seeds) {
            if ((long)gs.size() > (long)param_idx) {
                const float v = gs((int)param_idx);
                if (v >= plb && v <= pub) add_fit(v);
            }
        }

        std::vector<ScanPoint> pts = known();
        if (pts.empty()) {
            log<LOG_ERROR>(L"%1% || PRObe: no valid anchor fits for param %2%") % __func__ % param_idx;
            return result;
        }

        const size_t imin = argmin_chi2(pts);
        const float min_theta = pts[imin].value;
        const float min_chi2  = pts[imin].chi2;

        // Phase 2: quadratic surrogate around the known minimum (≥4 points enforced).
        QuadFit q = fit_quadratic(pts, min_theta, 5);
        bool spike_found = detect_spike(pts, q, opts.spike_chi2_threshold);
        bool use_surrogate = q.ok
                           && q.max_residual <= opts.quadratic_residual_max
                           && !spike_found;

        const float target_chi2 = min_chi2 + opts.target_dchi2;

        auto try_crossing_via_surrogate = [&](float &predicted_left, float &predicted_right) {
            // Solve a + b·Δθ + c·Δθ² = target_chi2 → quadratic in Δθ.
            float A = q.c, B = q.b, C = q.a - target_chi2;
            if (A <= 0.0f) return false;
            float disc = B * B - 4.0f * A * C;
            if (disc < 0.0f) return false;
            float root = std::sqrt(disc);
            float r1 = (-B - root) / (2.0f * A) + q.anchor;
            float r2 = (-B + root) / (2.0f * A) + q.anchor;
            predicted_left  = std::min(r1, r2);
            predicted_right = std::max(r1, r2);
            return std::isfinite(predicted_left) && std::isfinite(predicted_right);
        };

        // Bisection on one side of min_theta, hunting the Δχ² = target crossing.
        auto bisect_side = [&](bool right_side) {
            float lo_th = min_theta;
            // Initial outward step: prefer an existing point above target on this
            // side, otherwise step out by sigma_init (or 10% of range).
            float step  = (right_side ? 1.0f : -1.0f) *
                          std::max(opts.sigma_init, 0.1f * (pub - plb));
            float hi_th = clamp_to(min_theta + step, plb, pub);
            float c_hi  = std::numeric_limits<float>::infinity();

            for (const auto &p : known()) {
                bool same_side = right_side ? (p.value > min_theta) : (p.value < min_theta);
                if (same_side && p.chi2 >= target_chi2 &&
                    std::fabs(p.value - min_theta) < std::fabs(hi_th - min_theta)) {
                    hi_th = p.value;
                    c_hi  = p.chi2;
                }
            }

            // Extend outward until we find a point above target or hit the bound.
            int extends = 0;
            while (c_hi < target_chi2 && extends < opts.max_bisect_extends
                   && ctx.call_count - start_count < opts.max_fits) {
                if (!add_fit(hi_th)) break; // duplicate or capped
                c_hi = chi2_near(known(), hi_th);
                if (c_hi >= target_chi2) break;
                step *= 1.6f;
                float new_hi = clamp_to(hi_th + step, plb, pub);
                if (std::fabs(new_hi - hi_th) < duplicate_tol) {
                    // Hit parameter bound. Record and bail.
                    add_fit(new_hi);
                    return;
                }
                hi_th = new_hi;
                ++extends;
            }

            if (c_hi < target_chi2) return; // crossing not bracketed within bounds

            // Bisect (lo_th, hi_th).
            for (int it = 0; it < opts.max_bisect_iter
                             && ctx.call_count - start_count < opts.max_fits; ++it) {
                if (std::fabs(hi_th - lo_th) < duplicate_tol) break;
                float mid = 0.5f * (lo_th + hi_th);
                if (!add_fit(mid)) break;
                float c_mid = chi2_near(known(), mid);
                if (std::fabs(c_mid - target_chi2) < opts.boundary_tol_chi2) break;
                if (c_mid < target_chi2) lo_th = mid;
                else                     { hi_th = mid; c_hi = c_mid; }
            }
        };

        if (use_surrogate) {
            float pl, pr;
            if (try_crossing_via_surrogate(pl, pr)) {
                // Phase 3: confirmation fits at predicted crossings, refine if off.
                add_fit(pl);
                add_fit(pr);
                if (std::fabs(chi2_near(known(), pl) - target_chi2) > opts.boundary_tol_chi2)
                    bisect_side(false);
                if (std::fabs(chi2_near(known(), pr) - target_chi2) > opts.boundary_tol_chi2)
                    bisect_side(true);
                result.used_surrogate = true;
            } else {
                bisect_side(false);
                bisect_side(true);
            }
        } else {
            bisect_side(false);
            bisect_side(true);
        }

        log<LOG_INFO>(L"%1% || PRObe param %2%: %3% fits, surrogate=%4%, known min chi2=%5%")
            % __func__ % param_idx % (ctx.call_count - start_count)
            % (int)result.used_surrogate % min_chi2;
    }

    // Emit THIS call's fits, sorted ascending by theta (parallel arrays).
    std::sort(my_fits.begin(), my_fits.end(),
              [](const ScanPoint &a, const ScanPoint &b) { return a.value < b.value; });
    result.theta.reserve(my_fits.size());
    result.chi2.reserve(my_fits.size());
    result.best_fits.reserve(my_fits.size());
    for (auto &p : my_fits) {
        result.theta.push_back(p.value);
        result.chi2.push_back(p.chi2);
        result.best_fits.push_back(std::move(p.best_fit));
    }
    result.n_fits = ctx.call_count - start_count;

    return result;
}

} // namespace PRObe
} // namespace PROfit
