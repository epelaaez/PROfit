#ifndef PROSMOOTH_H
#define PROSMOOTH_H

// Kernel smoothing of binned variation/CV ratios for smooth= (header-only, Eigen-only so it is
// testable without ROOT). Each bin gets a Gaussian-weighted straight-line fit to its neighbours
// (weight = kernel x CV effective count); a line rather than a mean keeps trends and edges.

#include <Eigen/Eigen>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace PROfit {
namespace PROsmooth {

    /// Weighted least-squares fit of y = intercept + slope * x, built up one point at a time.
    struct WeightedLineFit {
        double sum_w = 0.0, sum_wx = 0.0, sum_wxx = 0.0, sum_wy = 0.0, sum_wxy = 0.0;

        void Add(double x, double y, double weight) {
            sum_w   += weight;
            sum_wx  += weight * x;
            sum_wxx += weight * x * x;
            sum_wy  += weight * y;
            sum_wxy += weight * x * y;
        }

        bool Empty() const { return sum_w <= 0.0; }

        /// The fitted line at x = 0, or the weighted mean of y if the line is ill-determined.
        double ValueAtZero() const {
            const double determinant = sum_w * sum_wxx - sum_wx * sum_wx;
            const bool line_is_determined = determinant > 1e-9 * sum_w * sum_wxx && sum_wxx > 0.0;
            if (!line_is_determined) return sum_wy / sum_w;
            return (sum_wxx * sum_wy - sum_wx * sum_wxy) / determinant;
        }
    };

    /// Smooth one row of num_bins bins; bin i of the row is at ratio_in[i * spacing].
    /// Bins with no CV events (weight <= 0) are left as-is and ignored. Output must not alias input.
    inline void SmoothLine(const float *ratio_in, float *ratio_out, const float *weight,
                           size_t num_bins, size_t spacing, float sigma) {
        const long window = (long)std::ceil(3.0f * sigma);   // beyond 3 sigma the kernel is negligible
        const double inv_two_sigma_sq = 0.5 / ((double)sigma * sigma);
        const long last_bin = (long)num_bins - 1;

        for (long centre = 0; centre <= last_bin; ++centre) {
            const size_t centre_pos = (size_t)centre * spacing;
            if (!(weight[centre_pos] > 0.0f)) {
                ratio_out[centre_pos] = ratio_in[centre_pos];
                continue;
            }

            WeightedLineFit fit;
            const long first_neighbour = std::max(0L, centre - window);
            const long last_neighbour = std::min(last_bin, centre + window);
            for (long neighbour = first_neighbour; neighbour <= last_neighbour; ++neighbour) {
                const size_t pos = (size_t)neighbour * spacing;
                const float ratio = ratio_in[pos];
                if (!(weight[pos] > 0.0f) || !std::isfinite(ratio)) continue;

                const double distance = (double)(neighbour - centre);
                const double kernel = std::exp(-distance * distance * inv_two_sigma_sq);
                fit.Add(distance, ratio, kernel * (double)weight[pos]);
            }

            ratio_out[centre_pos] = fit.Empty() ? ratio_in[centre_pos] : (float)fit.ValueAtZero();
        }
    }

    /// Smooth every row along one axis of a flattened (row-major, last axis fastest) block.
    inline void SmoothAlongAxis(const Eigen::Ref<const Eigen::VectorXf> &ratio_in, Eigen::VectorXf &ratio_out,
                                const Eigen::Ref<const Eigen::VectorXf> &weight,
                                const std::vector<size_t> &dims, size_t axis, float sigma) {
        size_t bins_before_axis = 1, spacing = 1;   // spacing = distance between neighbours along axis
        for (size_t d = 0; d < axis; ++d) bins_before_axis *= dims[d];
        for (size_t d = axis + 1; d < dims.size(); ++d) spacing *= dims[d];
        const size_t bins_along_axis = dims[axis];

        for (size_t outer = 0; outer < bins_before_axis; ++outer) {
            for (size_t inner = 0; inner < spacing; ++inner) {
                const size_t row_start = outer * bins_along_axis * spacing + inner;
                SmoothLine(ratio_in.data() + row_start, ratio_out.data() + row_start,
                           weight.data() + row_start, bins_along_axis, spacing, sigma);
            }
        }
    }

    /// Smooth one subchannel block in place, one axis at a time. preserve_rate keeps the
    /// CV-weighted rate change sum(cv * (ratio - 1)) fixed. Ratios >= 0 are never made negative.
    inline void SmoothBlock(Eigen::Ref<Eigen::VectorXf> ratio,
                            const Eigen::Ref<const Eigen::VectorXf> &weight,
                            const Eigen::Ref<const Eigen::VectorXf> &cv,
                            const std::vector<size_t> &dims, float sigma,
                            bool preserve_rate = true) {
        const Eigen::Index num_bins = ratio.size();
        if (!(sigma > 0.0f) || num_bins == 0 || dims.empty()) return;
        size_t expected_bins = 1;
        for (size_t bins_along_axis : dims) expected_bins *= bins_along_axis;
        if (expected_bins != (size_t)num_bins) return;

        auto usable = [&](Eigen::Index bin) { return weight(bin) > 0.0f && std::isfinite(ratio(bin)); };

        double target_rate = 0.0, total_cv = 0.0;
        for (Eigen::Index bin = 0; bin < num_bins; ++bin) {
            if (!usable(bin)) continue;
            target_rate += cv(bin) * (ratio(bin) - 1.0);
            total_cv += cv(bin);
        }

        const Eigen::VectorXf unsmoothed = ratio;
        Eigen::VectorXf smoothed(num_bins);
        for (size_t axis = 0; axis < dims.size(); ++axis) {
            if (dims[axis] < 2) continue;
            SmoothAlongAxis(ratio, smoothed, weight, dims, axis, sigma);
            ratio = smoothed;
        }

        // Shift the free bins so the rate matches again. A bin pushed below zero is pinned at 0
        // and the shift redone over the rest; each pass pins a bin or stops.
        std::vector<bool> pinned(num_bins, false);
        for (Eigen::Index pass = 0; pass <= num_bins; ++pass) {
            if (preserve_rate && total_cv > 0.0) {
                double current_rate = 0.0, free_cv = 0.0;
                for (Eigen::Index bin = 0; bin < num_bins; ++bin) {
                    if (!usable(bin)) continue;
                    current_rate += cv(bin) * (ratio(bin) - 1.0);
                    if (!pinned[bin]) free_cv += cv(bin);
                }
                if (free_cv > 0.0) {
                    const float shift = (float)((target_rate - current_rate) / free_cv);
                    for (Eigen::Index bin = 0; bin < num_bins; ++bin)
                        if (usable(bin) && !pinned[bin]) ratio(bin) += shift;
                }
            }

            bool pinned_any = false;
            for (Eigen::Index bin = 0; bin < num_bins; ++bin) {
                const bool made_negative = ratio(bin) < 0.0f && !(unsmoothed(bin) < 0.0f);
                if (pinned[bin] || !made_negative) continue;
                ratio(bin) = 0.0f;
                pinned[bin] = true;
                pinned_any = true;
            }
            if (!pinned_any) break;
        }
    }

    /// First bin that smoothing made NaN/inf or newly negative, else -1 (a backstop).
    inline Eigen::Index FirstInvalidSmoothed(const Eigen::Ref<const Eigen::VectorXf> &unsmoothed,
                                             const Eigen::Ref<const Eigen::VectorXf> &smoothed) {
        const Eigen::Index num_bins = std::min(unsmoothed.size(), smoothed.size());
        for (Eigen::Index bin = 0; bin < num_bins; ++bin) {
            const bool not_finite = !std::isfinite(smoothed(bin));
            const bool made_negative = smoothed(bin) < 0.0f && !(unsmoothed(bin) < 0.0f);
            if (not_finite || made_negative) return bin;
        }
        return -1;
    }

}  // namespace PROsmooth
}  // namespace PROfit

#endif
