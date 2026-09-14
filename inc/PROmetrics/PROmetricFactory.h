/**
 * @file PROmetricFactory.h
 * @brief Single construction point for the chi^2 metrics selected by name.
 */
#ifndef PROMETRICFACTORY_H
#define PROMETRICFACTORY_H

#include "PROmetric.h"

#include <memory>
#include <string>
#include <vector>

namespace PROfit {

    /**
     * @brief Build the chi^2 metric selected by @p chi2_kind.
     * @details THE one place a metric is constructed from its name. Accepts the canonical
     * names (neyman / pearson / CNP / poisson) and the legacy aliases (PROchi / PROCNP /
     * Poisson) via PROmetric::canonicalizeMetricName. Every pseudo-experiment path (fc,
     * the brazil band, every fc-adaptive stage) must build its per-universe metric here so
     * that the test-statistic distribution is computed with exactly the options of the data
     * fit — in particular @p shape_only, which the former per-site constructor ladders
     * silently dropped (they defaulted it to false).
     * @param chi2_kind            Metric name (canonical or legacy alias).
     * @param config/prop/systs/model/data  The usual metric inputs (non-owning).
     * @param strat                EventByEvent or BinnedChi2.
     * @param shape_only           Shape-only mode (per-channel prediction rescale + shape-projected systematics).
     * @param physics_param_fixed  Optional fixed physics parameters.
     * @return The metric, or nullptr (after a LOG_ERROR) for an unknown name.
     */
    std::unique_ptr<PROmetric> MakeMetric(const std::string &chi2_kind,
                                          const PROconfig &config, const PROpeller &prop,
                                          const PROsyst *systs, const PROmodel &model,
                                          const PROdata &data, PROmetric::EvalStrategy strat,
                                          bool shape_only,
                                          std::vector<float> physics_param_fixed = std::vector<float>());
}

#endif
