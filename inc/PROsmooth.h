#ifndef PROSMOOTH_H
#define PROSMOOTH_H

/**
 * @file PROsmooth.h
 * @brief Weighted local-linear kernel smoothing of binned ratios (used by smooth= in the XML).
 * @details Header-only and Eigen-only so it can be tested without ROOT. PROsyst calls
 * SmoothBlock once per (subchannel, knob) on the variation/CV ratio before the spline
 * coefficients are computed, so smoothing never crosses a subchannel or channel boundary.
 *
 * The estimate at bin i is a weighted straight-line fit to the bins within 3 sigma of i,
 * evaluated at i, with weight K(j - i) * w_j: a Gaussian kernel in bin-index units times the
 * CV effective event count of bin j. Fitting a line instead of taking a weighted mean keeps
 * linear trends and avoids the usual pull towards the interior at the edges of the range.
 * N-D binnings are smoothed one axis at a time.
 */

#include <Eigen/Eigen>
#include <cmath>
#include <cstddef>
#include <vector>

namespace PROfit {
namespace PROsmooth {

    /**
     * @brief Smooth one strided 1D line of bins with the weighted local-linear estimator.
     * @param y_in   Input values (read at y_in[k*stride], k < n).
     * @param y_out  Output values (written at the same offsets). Must not alias y_in.
     * @param w      Per-bin weights (same offsets); bins with w <= 0 keep their input value
     *               and contribute nothing to their neighbours.
     * @param n      Number of bins along the line.
     * @param stride Distance between consecutive bins of the line in the flat arrays.
     * @param sigma  Gaussian kernel width in bins (> 0).
     */
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

            // Local-linear value at d = 0; fall back to the weighted mean when the
            // line is ill-determined (e.g. only one usable bin in the window).
            const double det = S0 * S2 - S1 * S1;
            y_out[oi] = (det > 1e-9 * S0 * S2 && S2 > 0.0)
                ? (float)((S2 * T0 - S1 * T1) / det)
                : (float)(T0 / S0);
        }
    }

    /**
     * @brief In place: smooth one subchannel block of a ratio spectrum.
     * @param y      Ratio values of the block (flat, row-major: last dimension fastest,
     *               matching PROconfig::Binning::Bin).
     * @param w      Smoothing weights for the block (CV effective event counts).
     * @param cv     CV bin contents for the block, used only for rate preservation.
     * @param dims   Number of bins along each dimension; their product must equal y.size().
     * @param sigma  Gaussian kernel width in bins; <= 0 leaves y unchanged.
     * @param preserve_rate If true, add a constant to all populated bins so that the
     *               CV-weighted rate change sum(cv * (y - 1)) of the block is unchanged.
     */
    inline void SmoothBlock(Eigen::Ref<Eigen::VectorXf> y,
                            const Eigen::Ref<const Eigen::VectorXf> &w,
                            const Eigen::Ref<const Eigen::VectorXf> &cv,
                            const std::vector<size_t> &dims, float sigma,
                            bool preserve_rate = true) {
        if (!(sigma > 0.0f) || y.size() == 0 || dims.empty()) return;
        size_t total = 1;
        for (size_t nd : dims) total *= nd;
        if (total != (size_t)y.size()) return;  // layout mismatch: caller logs, we do nothing

        // CV-weighted rate change before smoothing, over the bins that can change.
        double rate_before = 0.0, cv_sum = 0.0;
        for (Eigen::Index b = 0; b < y.size(); ++b)
            if (w(b) > 0.0f && std::isfinite(y(b))) { rate_before += cv(b) * (y(b) - 1.0); cv_sum += cv(b); }

        Eigen::VectorXf tmp(y.size());
        for (size_t d = 0; d < dims.size(); ++d) {
            const size_t n = dims[d];
            if (n < 2) continue;
            size_t stride = 1;
            for (size_t k = d + 1; k < dims.size(); ++k) stride *= dims[k];
            // A line starts at every flat index whose coordinate along d is zero.
            for (size_t base = 0; base < total; ++base) {
                if ((base / stride) % n != 0) continue;
                SmoothLine(y.data() + base, tmp.data() + base, w.data() + base, n, stride, sigma);
            }
            y = tmp;
        }

        if (preserve_rate && cv_sum > 0.0) {
            double rate_after = 0.0;
            for (Eigen::Index b = 0; b < y.size(); ++b)
                if (w(b) > 0.0f && std::isfinite(y(b))) rate_after += cv(b) * (y(b) - 1.0);
            const float shift = (float)((rate_before - rate_after) / cv_sum);
            for (Eigen::Index b = 0; b < y.size(); ++b)
                if (w(b) > 0.0f && std::isfinite(y(b))) y(b) += shift;
        }
    }

    /**
     * @brief First bin where smoothing broke the ratio: NaN/inf, or negative where @p raw was not
     * (e.g. the local-linear fit extrapolating below zero at the edge of the range). -1 if none.
     */
    inline Eigen::Index FirstInvalidSmoothed(const Eigen::Ref<const Eigen::VectorXf> &raw,
                                             const Eigen::Ref<const Eigen::VectorXf> &smoothed) {
        for (Eigen::Index b = 0; b < smoothed.size() && b < raw.size(); ++b)
            if (!std::isfinite(smoothed(b)) || (smoothed(b) < 0.0f && !(raw(b) < 0.0f))) return b;
        return -1;
    }

}  // namespace PROsmooth
}  // namespace PROfit

#endif
