/**
 * @file PRObench.h
 * @brief Internal scaling / timing benchmarks for PROfit hot paths.
 * @author PROfit Collaboration
 *
 * @details Drives a fixed catalogue of micro-benchmarks covering the three
 * dominant hot paths exercised across a typical analysis:
 *   - FillSpectra evaluation (default N calls)
 *   - PROmetric operator() evaluation (default N/10 calls)
 *   - PROfitter::Fit global fits (default N/100 calls)
 *
 * For each path, three variation modes are timed:
 *   (vary_all) physics + nuisance both randomised per call
 *   (vary_phys) physics randomised, nuisance held at central value
 *   (vary_nuis) nuisance randomised, physics held at central value
 *
 * The benchmark uses the live PROmetric instance constructed by the main
 * PROfit chain (PROchi / PROCNP / PROpoisson — chosen by the existing
 * `--chi2` style argument). It does not build its own metric.
 *
 * Results are emitted via the standard PROlog `log` facility on a single
 * greppable line per test, e.g.
 *
 *   [SCALETEST] tag=fillspectra_vary_all N=1000 nbins=42 nphys=4 nnuis=87 total_us=12345 per_call_us=12.345
 *
 * A bash wrapper that varies binning/systematics/etc. across runs can
 * `grep "\[SCALETEST\]"` and parse the columns to plot scaling.
 */
#ifndef PROBENCH_H
#define PROBENCH_H

#include "PROconfig.h"
#include "PROmetric.h"
#include "PROpeller.h"
#include "PROfitter.h"

#include <Eigen/Eigen>

#include <cstdint>
#include <string>
#include <vector>

namespace PROfit {
namespace PRObench {

    /**
     * @brief Bitmask selector for which benchmarks to run.
     * @details Letters match the original spec (a–i). Combine with bitwise OR.
     */
    enum BenchTest : unsigned int {
        Bench_None             = 0,
        Bench_FillSpectra_All  = 1u << 0,  ///< (a) FillSpectra, vary phys + nuis.
        Bench_FillSpectra_Phys = 1u << 1,  ///< (b) FillSpectra, vary phys only.
        Bench_FillSpectra_Nuis = 1u << 2,  ///< (c) FillSpectra, vary nuis only.
        Bench_Metric_All       = 1u << 3,  ///< (d) PROmetric(), vary phys + nuis.
        Bench_Metric_Phys      = 1u << 4,  ///< (e) PROmetric(), vary phys only.
        Bench_Metric_Nuis      = 1u << 5,  ///< (f) PROmetric(), vary nuis only.
        Bench_Fit_All          = 1u << 6,  ///< (g) PROfitter::Fit(), vary phys + nuis seed.
        Bench_Fit_Phys         = 1u << 7,  ///< (h) PROfitter::Fit(), vary phys seed only.
        Bench_Fit_Nuis         = 1u << 8,  ///< (i) PROfitter::Fit(), vary nuis seed only.
        Bench_FillSpectra_Group = Bench_FillSpectra_All | Bench_FillSpectra_Phys | Bench_FillSpectra_Nuis,
        Bench_Metric_Group      = Bench_Metric_All       | Bench_Metric_Phys       | Bench_Metric_Nuis,
        Bench_Fit_Group         = Bench_Fit_All          | Bench_Fit_Phys          | Bench_Fit_Nuis,
        Bench_All               = Bench_FillSpectra_Group | Bench_Metric_Group | Bench_Fit_Group,
    };

    /**
     * @brief Configuration knobs for run_scale_test.
     */
    struct BenchOptions {
        int      N        = 1000;          ///< Base call count: FillSpectra=N, Metric=N/10, Fit=N/100. All clamped to ≥1.
        unsigned tests    = Bench_All;     ///< Bitmask of benchmarks to run.
        uint32_t rng_seed = 0xBEEFCAFE;    ///< Fixed seed for reproducibility.
        bool     binned   = true;          ///< FillSpectra binned vs event-by-event.
    };

    /**
     * @brief One row of timing output.
     */
    struct BenchResult {
        std::string tag;          ///< Short greppable label, e.g. "fillspectra_vary_all".
        int         n_calls = 0;
        int         nbins   = 0;
        int         nphys   = 0;
        int         nnuis   = 0;
        double      total_us    = 0.0;
        double      per_call_us = 0.0;
    };

    /**
     * @brief Run the requested scaling benchmarks and emit greppable LOG lines.
     * @details Uses the caller-provided PROmetric instance as-is — same
     * concrete class (PROchi / PROCNP / PROpoisson) and same configuration
     * the rest of the PROfit chain is using. The bench mutates internal
     * metric state (last_param, last_value, fs_cache, call_count) so the
     * caller should not rely on those after run_scale_test returns.
     * @param config     Analysis configuration (used for nbins, i_prime).
     * @param prop       MC event store (needed for FillSpectra free function).
     * @param metric     Live PROmetric — its GetSysts(), GetModel(), and
     *                   operator() drive the (a–c) and (d–i) benchmarks.
     * @param fitconfig  PROfitter configuration for the (g–i) fit benchmarks.
     * @param opts       Knobs; see BenchOptions.
     * @return Per-test BenchResult records (also logged at LOG_INFO).
     */
    std::vector<BenchResult> run_scale_test(
        const PROconfig &config,
        const PROpeller &prop,
        PROmetric &metric,
        const PROfitterConfig &fitconfig,
        const BenchOptions &opts);

}  // namespace PRObench
}  // namespace PROfit

#endif
