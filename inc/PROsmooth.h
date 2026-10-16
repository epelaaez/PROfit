#ifndef PROSMOOTH_H
#define PROSMOOTH_H

// Kernel smoothing of binned variation/CV ratios for smooth= (header-only, Eigen-only so it is
// testable without ROOT). Each bin gets a Gaussian-weighted straight-line fit to its neighbours
// (weight = kernel x CV effective count); a line rather than a mean keeps trends and edges.

#include <Eigen/Eigen>
#include <cmath>
#include <cstddef>
#include <vector>

namespace PROfit {
namespace PROsmooth {

    /// Smooth n bins spaced by stride. Bins with w <= 0 are left as-is and ignored. y_out must not alias y_in.
    inline void SmoothLine(const float *y_in, float *y_out, const float *w,
                           size_t n, size_t stride, float sigma) {
        const long radius = (long)std::ceil(3.0f * sigma);
        const double inv2s2 = 0.5 / ((double)sigma * sigma);
        for (long i = 0; i < (long)n; ++i) {
            const size_t oi = (size_t)i * stride;
            if (!(w[oi] > 0.0f)) { y_out[oi] = y_in[oi]; continue; }

            double S0 = 0, S1 = 0, S2 = 0, T0 = 0, T1 = 0;
            const long jlo = std::max(0L, i - radius), jhi = std::min((long)n - 1, i + radius);
            for (long j = jlo; j <= jhi; ++j) {
                const size_t oj = (size_t)j * stride;
                if (!(w[oj] > 0.0f) || !std::isfinite(y_in[oj])) continue;
                const double d = (double)(j - i);
                const double k = std::exp(-d * d * inv2s2) * (double)w[oj];
                S0 += k;  S1 += k * d;  S2 += k * d * d;
                T0 += k * y_in[oj];  T1 += k * d * y_in[oj];
            }
            if (S0 <= 0.0) { y_out[oi] = y_in[oi]; continue; }

            // Line fit at d = 0, or the weighted mean if the line is ill-determined.
            const double det = S0 * S2 - S1 * S1;
            y_out[oi] = (det > 1e-9 * S0 * S2 && S2 > 0.0)
                ? (float)((S2 * T0 - S1 * T1) / det)
                : (float)(T0 / S0);
        }
    }

    /// Smooth one subchannel block in place (row-major, last dim fastest), one axis at a time.
    /// preserve_rate keeps sum(cv * (y - 1)) fixed. Ratios >= 0 are never made negative.
    inline void SmoothBlock(Eigen::Ref<Eigen::VectorXf> y,
                            const Eigen::Ref<const Eigen::VectorXf> &w,
                            const Eigen::Ref<const Eigen::VectorXf> &cv,
                            const std::vector<size_t> &dims, float sigma,
                            bool preserve_rate = true) {
        if (!(sigma > 0.0f) || y.size() == 0 || dims.empty()) return;
        size_t total = 1;
        for (size_t nd : dims) total *= nd;
        if (total != (size_t)y.size()) return;

        double rate_before = 0.0, cv_sum = 0.0;
        for (Eigen::Index b = 0; b < y.size(); ++b)
            if (w(b) > 0.0f && std::isfinite(y(b))) { rate_before += cv(b) * (y(b) - 1.0); cv_sum += cv(b); }

        const Eigen::VectorXf y0 = y;
        Eigen::VectorXf tmp(y.size());
        for (size_t d = 0; d < dims.size(); ++d) {
            const size_t n = dims[d];
            if (n < 2) continue;
            size_t stride = 1;
            for (size_t k = d + 1; k < dims.size(); ++k) stride *= dims[k];
            for (size_t base = 0; base < total; ++base) {
                if ((base / stride) % n != 0) continue;
                SmoothLine(y.data() + base, tmp.data() + base, w.data() + base, n, stride, sigma);
            }
            y = tmp;
        }

        // Restore the rate; pin bins pushed below zero at 0 and redo it over the rest.
        std::vector<char> pinned(y.size(), 0);
        for (Eigen::Index iter = 0; iter <= y.size(); ++iter) {
            if (preserve_rate && cv_sum > 0.0) {
                double rate_after = 0.0, free_cv = 0.0;
                for (Eigen::Index b = 0; b < y.size(); ++b)
                    if (w(b) > 0.0f && std::isfinite(y(b))) {
                        rate_after += cv(b) * (y(b) - 1.0);
                        if (!pinned[b]) free_cv += cv(b);
                    }
                if (free_cv > 0.0) {
                    const float shift = (float)((rate_before - rate_after) / free_cv);
                    for (Eigen::Index b = 0; b < y.size(); ++b)
                        if (w(b) > 0.0f && std::isfinite(y(b)) && !pinned[b]) y(b) += shift;
                }
            }
            bool newly_pinned = false;
            for (Eigen::Index b = 0; b < y.size(); ++b)
                if (!pinned[b] && y(b) < 0.0f && !(y0(b) < 0.0f)) { y(b) = 0.0f; pinned[b] = 1; newly_pinned = true; }
            if (!newly_pinned) break;
        }
    }

    /// First bin that smoothing made NaN/inf or newly negative, else -1 (a backstop).
    inline Eigen::Index FirstInvalidSmoothed(const Eigen::Ref<const Eigen::VectorXf> &raw,
                                             const Eigen::Ref<const Eigen::VectorXf> &smoothed) {
        for (Eigen::Index b = 0; b < smoothed.size() && b < raw.size(); ++b)
            if (!std::isfinite(smoothed(b)) || (smoothed(b) < 0.0f && !(raw(b) < 0.0f))) return b;
        return -1;
    }

}  // namespace PROsmooth
}  // namespace PROfit

#endif
