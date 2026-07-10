#include "PRObe.h"
#include "PROlog.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>

namespace PROfit {
namespace PRObe {

namespace {

struct SeedPool {
    std::vector<float> theta;
    std::vector<Eigen::VectorXf> bf;
    std::vector<float> chi2;
};

void pool_add(SeedPool &p, float t, Eigen::VectorXf bf, float c2) {
    p.theta.push_back(t);
    p.bf.push_back(std::move(bf));
    p.chi2.push_back(c2);
}

bool pool_has_close(const SeedPool &p, float t, float tol) {
    for (float pt : p.theta) if (std::fabs(pt - t) < tol) return true;
    return false;
}

size_t pool_argmin_chi2(const SeedPool &p) {
    return (size_t)std::distance(p.chi2.begin(),
                                  std::min_element(p.chi2.begin(), p.chi2.end()));
}

size_t pool_nearest(const SeedPool &p, float t) {
    size_t best = 0;
    float bestd = std::fabs(p.theta[0] - t);
    for (size_t i = 1; i < p.theta.size(); ++i) {
        float d = std::fabs(p.theta[i] - t);
        if (d < bestd) { bestd = d; best = i; }
    }
    return best;
}

float pool_chi2_near(const SeedPool &p, float t) {
    return p.chi2[pool_nearest(p, t)];
}

std::vector<Eigen::VectorXf> warm_seeds_for(
    const SeedPool &p, float t,
    const std::vector<Eigen::VectorXf> &globals)
{
    // Each seed_point passed to PROfitter::Fit triggers a full LBFGS
    // minimization (see PROfitter.cxx:268-318), so seed-list size is a
    // direct linear cost. We add only the closest-by-θ pool entry — that's
    // the strongest warm-start. Adding the global minimum would cost an
    // extra LBFGS for marginal correctness benefit (the global seeds and
    // the closest-θ seed already cover the relevant basins).
    std::vector<Eigen::VectorXf> ws = globals;
    if (!p.theta.empty()) {
        size_t inear = pool_nearest(p, t);
        ws.push_back(p.bf[inear]);
    }
    return ws;
}

/// Quadratic surrogate χ²(θ) ≈ a + b·(θ-θ₀) + c·(θ-θ₀)² fit by least squares.
struct QuadFit {
    float a = 0, b = 0, c = 0; ///< Coefficients about anchor θ₀.
    float anchor = 0;          ///< θ₀.
    float max_residual = 0;    ///< Max |fit - data| across the points used.
    bool  ok = false;          ///< Upward-opening (c>0), finite, ≥3 points.
};

QuadFit fit_quadratic(const SeedPool &p, float theta_anchor, size_t max_pts) {
    QuadFit q;
    q.anchor = theta_anchor;
    if (p.theta.size() < 3) return q;

    // Pick up to max_pts pool points nearest the anchor in θ.
    std::vector<size_t> idxs(p.theta.size());
    std::iota(idxs.begin(), idxs.end(), 0);
    std::sort(idxs.begin(), idxs.end(), [&](size_t a, size_t b) {
        return std::fabs(p.theta[a] - theta_anchor) <
               std::fabs(p.theta[b] - theta_anchor);
    });
    if (idxs.size() > max_pts) idxs.resize(max_pts);
    if (idxs.size() < 3) return q;

    Eigen::MatrixXf A((int)idxs.size(), 3);
    Eigen::VectorXf y((int)idxs.size());
    for (size_t i = 0; i < idxs.size(); ++i) {
        float dt = p.theta[idxs[i]] - theta_anchor;
        A((int)i, 0) = 1.0f;
        A((int)i, 1) = dt;
        A((int)i, 2) = dt * dt;
        y((int)i)    = p.chi2[idxs[i]];
    }
    Eigen::VectorXf coef = A.colPivHouseholderQr().solve(y);
    q.a = coef(0); q.b = coef(1); q.c = coef(2);
    Eigen::VectorXf res = (A * coef - y).cwiseAbs();
    q.max_residual = res.maxCoeff();
    q.ok = (q.c > 0.0f) && std::isfinite(q.a) && std::isfinite(q.b) && std::isfinite(q.c);
    return q;
}

/// Spike test: any pool point lying threshold-above the surrogate flags a spike.
/// Sub-surrogate dips do not flag (a basin found below the quadratic prediction
/// is interesting but not what spike detection is for).
bool detect_spike(const SeedPool &p, const QuadFit &q, float threshold) {
    if (!q.ok) return false;
    for (size_t i = 0; i < p.theta.size(); ++i) {
        float dt = p.theta[i] - q.anchor;
        float pred = q.a + q.b * dt + q.c * dt * dt;
        if (p.chi2[i] - pred > threshold) return true;
    }
    return false;
}

float clamp_to(float x, float lo, float hi) {
    return std::max(lo, std::min(hi, x));
}

/// Functor encapsulating "do one full minimisation at fixed θ" with bookkeeping.
struct FitOnce {
    PROmetric &metric;
    Eigen::VectorXf full_lb;
    Eigen::VectorXf full_ub;
    size_t param_idx;
    const PROfitterConfig &fitconfig;
    uint32_t base_seed;
    int call_count = 0;

    std::pair<float, Eigen::VectorXf> operator()(
        float theta,
        const std::vector<Eigen::VectorXf> &seeds)
    {
        Eigen::VectorXf tlb = full_lb;
        Eigen::VectorXf tub = full_ub;
        tlb((int)param_idx) = theta;
        tub((int)param_idx) = theta;
        metric.setBounds(tlb, tub);
        metric.fixSpline((int)param_idx, theta);
        PROfitter fitter(tub, tlb, fitconfig, base_seed + (uint32_t)call_count);
        ++call_count;
        // TEMPORARY: FitScan disabled while debugging fit-quality issues.
        // Using the full Fit() (with Latin + PSO + multi-LBFGS) on every PRObe
        // sub-fit so we can confirm everything else (per-bar tracker, on_fit
        // callback, physics path, chunked dispatch, dedup, etc.) is clean.
        // To re-enable scan-mode: swap fitter.Fit -> fitter.FitScan below.
        float chi = seeds.empty() ? fitter.Fit(metric)
                                  : fitter.Fit(metric, seeds);
        return {chi, fitter.best_fit};
    }
};

} // anonymous namespace

CrossingResult chi2_crossing_1d(
    PROmetric &metric,
    size_t param_idx,
    const Eigen::VectorXf &full_lb,
    const Eigen::VectorXf &full_ub,
    float bf_value,
    const std::vector<Eigen::VectorXf> &global_seeds,
    const PROfitterConfig &fitconfig,
    uint32_t base_seed,
    const CrossingOpts &opts)
{
    CrossingResult result;
    SeedPool pool;
    FitOnce fit_at{metric, full_lb, full_ub, param_idx, fitconfig, base_seed, 0};

    const float plb_raw = full_lb((int)param_idx);
    const float pub_raw = full_ub((int)param_idx);
    if (!(pub_raw > plb_raw)) {
        log<LOG_WARNING>(L"%1% || PRObe: param %2% has lb >= ub (%3% >= %4%); nothing to scan.")
            % __func__ % param_idx % plb_raw % pub_raw;
        return result;
    }
    // Effective sampling bounds: substitute ±3 for ±inf so coarse sampling and
    // bisection step sizes are finite. The metric/fit still see the original
    // (possibly infinite) bounds for *other* parameters — only the scanned
    // parameter is sampled inside [plb, pub], and at each sample it's clamped
    // to a single value via tlb[idx] = tub[idx] = theta inside FitOnce.
    const float plb = std::isinf(plb_raw) ? -3.0f : plb_raw;
    const float pub = std::isinf(pub_raw) ?  3.0f : pub_raw;
    if (std::isinf(plb_raw) || std::isinf(pub_raw)) {
        log<LOG_DEBUG>(L"%1% || PRObe param %2%: substituting finite sampling bounds [%3%, %4%] for raw [%5%, %6%]")
            % __func__ % param_idx % plb % pub % plb_raw % pub_raw;
    }
    const float duplicate_tol = std::max(1e-4f, 1e-3f * (pub - plb));

    auto add_fit = [&](float theta) -> bool {
        theta = clamp_to(theta, plb, pub);
        if (pool_has_close(pool, theta, duplicate_tol)) return false;
        if (fit_at.call_count >= opts.max_fits) return false;
        auto pr = fit_at(theta, warm_seeds_for(pool, theta, global_seeds));
        pool_add(pool, theta, pr.second, pr.first);
        // Live progress: notify caller after each accepted fit so a progress
        // bar updates during the scan, not only on return.
        if (opts.on_fit) opts.on_fit();
        return true;
    };

    // ---- Physics branch (may_have_spikes=true): sample-then-linear-refine ----
    // Physics chi² is non-Gaussian (asymmetric basins, sharp minima, flat tails),
    // so the quadratic surrogate path used for nuisances rejects often and falls
    // back to bracket+bisect, which blows up to max_fits. Instead: sample a small
    // evenly-spaced grid covering [plb, pub], augment with the BF and any global
    // seed values that fall inside the chunk, then for each side of the minimum
    // do **one** linear-interp refinement at the predicted Δχ²=target crossing.
    // No surrogate, no spike detection, no bisection. Predictable cost ≈
    // coarse_n + 1 + |seeds_in_range| + 2 ≈ 8-12 fits per scan/chunk.
    if (opts.may_have_spikes) {
        const int n_grid = std::max(3, opts.coarse_n);
        for (int i = 0; i < n_grid; ++i) {
            const float frac = (n_grid > 1) ? (float)i / (float)(n_grid - 1) : 0.5f;
            add_fit(plb + (pub - plb) * frac);
        }
        // Anchor at the BF value (typically the chunk midpoint or the global BF in-chunk).
        add_fit(bf_value);
        // Seeds whose scanned-param value falls in this chunk — seeds secondary basins.
        // Out-of-chunk seeds are dropped here (the full global_seeds list is still
        // used by warm_seeds_for to start the LBFGS fits).
        for (const auto &gs : global_seeds) {
            if ((long)gs.size() > (long)param_idx) {
                const float v = gs((int)param_idx);
                if (v >= plb && v <= pub) add_fit(v);
            }
        }

        if (pool.theta.empty()) {
            log<LOG_ERROR>(L"%1% || PRObe-phys: no valid anchor fits for param %2%") % __func__ % param_idx;
            return result;
        }

        // Sort pool indices by theta to walk neighbours.
        std::vector<size_t> order(pool.theta.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](size_t a, size_t b) { return pool.theta[a] < pool.theta[b]; });

        // Locate the minimum in the sorted order.
        size_t imin = 0;
        for (size_t i = 1; i < order.size(); ++i) {
            if (pool.chi2[order[i]] < pool.chi2[order[imin]]) imin = i;
        }
        const float min_chi2_phys  = pool.chi2[order[imin]];
        const float target_chi2    = min_chi2_phys + opts.target_dchi2;

        // Refine: walk outward on each side of the min; the first adjacent pair
        // that straddles target gets one linear-interp fit. If no straddle is
        // found in the chunk, the band edge falls at the chunk boundary — the
        // chunked dispatcher's neighbour chunk will pick up the actual crossing.
        auto refine_side = [&](bool right_side) {
            const ssize_t step = right_side ? +1 : -1;
            for (ssize_t i = (ssize_t)imin;
                 (step > 0 ? (i + 1 < (ssize_t)order.size()) : (i > 0));
                 i += step)
            {
                const ssize_t j = i + step;
                const float c_inner = pool.chi2[order[(size_t)i]];
                const float c_outer = pool.chi2[order[(size_t)j]];
                if ((c_inner - target_chi2) * (c_outer - target_chi2) <= 0.0f) {
                    const float t_inner = pool.theta[order[(size_t)i]];
                    const float t_outer = pool.theta[order[(size_t)j]];
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

        // Build sorted output identical in shape to the surrogate path.
        std::vector<size_t> final_order(pool.theta.size());
        std::iota(final_order.begin(), final_order.end(), 0);
        std::sort(final_order.begin(), final_order.end(),
                  [&](size_t a, size_t b) { return pool.theta[a] < pool.theta[b]; });
        result.theta.reserve(final_order.size());
        result.chi2.reserve(final_order.size());
        result.best_fits.reserve(final_order.size());
        for (size_t i : final_order) {
            result.theta.push_back(pool.theta[i]);
            result.chi2.push_back(pool.chi2[i]);
            result.best_fits.push_back(pool.bf[i]);
        }

        // Recompute min/leftX/rightX after refinement (refines may have shifted them).
        size_t k = 0;
        float best = result.chi2[0];
        for (size_t i = 1; i < result.chi2.size(); ++i)
            if (result.chi2[i] < best) { best = result.chi2[i]; k = i; }
        result.minX  = result.theta[k];
        result.leftX = result.theta.front();
        result.rightX = result.theta.back();
        const float tgt = best + opts.target_dchi2;
        for (size_t i = 1; i <= k && i < result.theta.size(); ++i) {
            const float c0 = result.chi2[i-1], c1 = result.chi2[i];
            if ((c0 - tgt) * (c1 - tgt) <= 0.0f) {
                const float t0 = result.theta[i-1], t1 = result.theta[i];
                const float d = c1 - c0;
                const float frac = std::fabs(d) > 0.0f ? (tgt - c0) / d : 0.0f;
                result.leftX = t0 + frac * (t1 - t0);
            }
        }
        for (size_t i = k + 1; i < result.theta.size(); ++i) {
            const float c0 = result.chi2[i-1], c1 = result.chi2[i];
            if ((c0 - tgt) * (c1 - tgt) <= 0.0f) {
                const float t0 = result.theta[i-1], t1 = result.theta[i];
                const float d = c1 - c0;
                const float frac = std::fabs(d) > 0.0f ? (tgt - c0) / d : 0.0f;
                result.rightX = t0 + frac * (t1 - t0);
                break;
            }
        }
        result.used_surrogate = false;
        result.n_fits = fit_at.call_count;

        log<LOG_INFO>(L"%1% || PRObe-phys param %2%: %3% fits, min@%4% chi2=%5%, band [%6%, %7%]")
            % __func__ % param_idx % result.n_fits
            % result.minX % min_chi2_phys % result.leftX % result.rightX;

        return result;
    }
    // ---- end physics branch ----

    // Phase 1: anchors around the prior best-fit value (also seeds when they exist).
    for (float s : opts.anchor_sigmas) {
        add_fit(bf_value + s * opts.sigma_init);
    }
    // If multiple global seeds exist, also anchor at each one's parameter value
    // — this seeds secondary basins for multi-modal physics. Only seeds whose
    // scanned-param value falls inside this scan's [plb, pub] range contribute
    // an anchor; out-of-range seeds belong to other chunks (when chunked) or
    // beyond the substituted finite bounds, and adding them here would just
    // stack a wasted anchor at the chunk boundary via clamp_to. The full
    // global_seeds list is still passed to the fitter for warm-starting the
    // non-scanned coordinates.
    for (const auto &gs : global_seeds) {
        if ((long)gs.size() > (long)param_idx) {
            const float v = gs((int)param_idx);
            if (v >= plb && v <= pub) add_fit(v);
        }
    }

    if (pool.theta.empty()) {
        log<LOG_ERROR>(L"%1% || PRObe: no valid anchor fits for param %2%") % __func__ % param_idx;
        return result;
    }

    size_t imin = pool_argmin_chi2(pool);
    float min_theta = pool.theta[imin];
    float min_chi2  = pool.chi2[imin];

    // Phase 2: quadratic surrogate around the in-pool minimum.
    QuadFit q = fit_quadratic(pool, min_theta, 5);
    bool spike_found = detect_spike(pool, q, opts.spike_chi2_threshold);
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
        // Initial outward step: prefer an existing pool point above target on this side,
        // otherwise step out by sigma_init (or 10% of the parameter range, whichever larger).
        float step  = (right_side ? 1.0f : -1.0f) *
                      std::max(opts.sigma_init, 0.1f * (pub - plb));
        float hi_th = clamp_to(min_theta + step, plb, pub);
        float c_hi  = std::numeric_limits<float>::infinity();

        for (size_t i = 0; i < pool.theta.size(); ++i) {
            float th = pool.theta[i];
            bool same_side = right_side ? (th > min_theta) : (th < min_theta);
            if (same_side && pool.chi2[i] >= target_chi2 &&
                std::fabs(th - min_theta) < std::fabs(hi_th - min_theta)) {
                hi_th = th;
                c_hi  = pool.chi2[i];
            }
        }

        // Extend outward until we find a point above target or hit the bound.
        int extends = 0;
        while (c_hi < target_chi2 && extends < 5 && fit_at.call_count < opts.max_fits) {
            if (!add_fit(hi_th)) break; // duplicate or capped
            c_hi = pool_chi2_near(pool, hi_th);
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
        for (int it = 0; it < opts.max_bisect_iter && fit_at.call_count < opts.max_fits; ++it) {
            if (std::fabs(hi_th - lo_th) < duplicate_tol) break;
            float mid = 0.5f * (lo_th + hi_th);
            if (!add_fit(mid)) break;
            float c_mid = pool_chi2_near(pool, mid);
            if (std::fabs(c_mid - target_chi2) < opts.boundary_tol_chi2) break;
            if (c_mid < target_chi2) lo_th = mid;
            else                     { hi_th = mid; c_hi = c_mid; }
        }
    };

    if (use_surrogate) {
        float pl, pr;
        if (try_crossing_via_surrogate(pl, pr)) {
            // Phase 3: confirmation fits at predicted crossings, refine if too far off.
            add_fit(pl);
            add_fit(pr);
            if (std::fabs(pool_chi2_near(pool, pl) - target_chi2) > opts.boundary_tol_chi2)
                bisect_side(false);
            if (std::fabs(pool_chi2_near(pool, pr) - target_chi2) > opts.boundary_tol_chi2)
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

    // Sort pool by theta and emit.
    std::vector<size_t> order(pool.theta.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](size_t a, size_t b) { return pool.theta[a] < pool.theta[b]; });
    result.theta.reserve(order.size());
    result.chi2.reserve(order.size());
    result.best_fits.reserve(order.size());
    for (size_t i : order) {
        result.theta.push_back(pool.theta[i]);
        result.chi2.push_back(pool.chi2[i]);
        result.best_fits.push_back(pool.bf[i]);
    }

    // Refresh min after possible new pool entries.
    {
        size_t k = 0;
        float best = result.chi2[0];
        for (size_t i = 1; i < result.chi2.size(); ++i)
            if (result.chi2[i] < best) { best = result.chi2[i]; k = i; }
        result.minX = result.theta[k];

        // Linear-interpolated crossings either side of result.minX.
        result.leftX  = result.theta.front();
        result.rightX = result.theta.back();
        for (size_t i = 1; i <= k && i < result.theta.size(); ++i) {
            float c0 = result.chi2[i-1], c1 = result.chi2[i];
            if ((c0 - target_chi2) * (c1 - target_chi2) <= 0.0f) {
                float t0 = result.theta[i-1], t1 = result.theta[i];
                float denom = (c1 - c0);
                float frac = std::fabs(denom) > 0.0f ? (target_chi2 - c0) / denom : 0.0f;
                result.leftX = t0 + frac * (t1 - t0);
            }
        }
        for (size_t i = k + 1; i < result.theta.size(); ++i) {
            float c0 = result.chi2[i-1], c1 = result.chi2[i];
            if ((c0 - target_chi2) * (c1 - target_chi2) <= 0.0f) {
                float t0 = result.theta[i-1], t1 = result.theta[i];
                float denom = (c1 - c0);
                float frac = std::fabs(denom) > 0.0f ? (target_chi2 - c0) / denom : 0.0f;
                result.rightX = t0 + frac * (t1 - t0);
                break;
            }
        }
    }
    result.n_fits = fit_at.call_count;

    log<LOG_INFO>(L"%1% || PRObe param %2%: %3% fits, surrogate=%4%, min@%5% chi2=%6%, band [%7%, %8%]")
        % __func__ % param_idx % result.n_fits % (int)result.used_surrogate
        % result.minX % min_chi2 % result.leftX % result.rightX;

    return result;
}

} // namespace PRObe
} // namespace PROfit
