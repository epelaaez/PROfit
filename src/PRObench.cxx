#include "PRObench.h"

#include "PROlog.h"
#include "PROmetric.h"
#include "PROmodel.h"
#include "PROsyst.h"
#include "PROpeller.h"
#include "PROcess.h"
#include "PROspec.h"

#include <Eigen/Eigen>

#include <algorithm>
#include <chrono>
#include <random>
#include <string>
#include <vector>

namespace PROfit {
namespace PRObench {

namespace {

// Build a random parameter vector. `vary_phys` and `vary_nuis` independently
// control which slice is randomised; the other slice is held at the
// model/spline central value (CV = model.default_val for physics, 0 for
// splines).  Physics is sampled uniform within the model's [lb, ub] range
// and nuisance is sampled N(0, 1) which is the natural pull range for splines.
Eigen::VectorXf draw_params(const PROmodel &model, const PROsyst &syst,
                            std::mt19937 &rng,
                            bool vary_phys, bool vary_nuis)
{
    const size_t nphys = model.nparams;
    const size_t nnuis = syst.GetNSplines();
    Eigen::VectorXf p = Eigen::VectorXf::Zero(nphys + nnuis);

    for (size_t i = 0; i < nphys; ++i) {
        const float lb = model.lb(i);
        const float ub = model.ub(i);
        if (vary_phys && ub > lb) {
            std::uniform_real_distribution<float> u(lb, ub);
            p(i) = u(rng);
        } else {
            p(i) = model.default_val(i);
        }
    }
    if (vary_nuis) {
        std::normal_distribution<float> g(0.0f, 1.0f);
        for (size_t i = 0; i < nnuis; ++i) p(nphys + i) = g(rng);
    }
    return p;
}

// Pre-generate `n` parameter vectors so the timed loop only measures the hot
// path, not RNG/allocation noise.
std::vector<Eigen::VectorXf> draw_param_set(const PROmodel &model, const PROsyst &syst,
                                            int n, uint32_t seed,
                                            bool vary_phys, bool vary_nuis)
{
    std::mt19937 rng(seed);
    std::vector<Eigen::VectorXf> out;
    out.reserve(n);
    for (int k = 0; k < n; ++k) out.push_back(draw_params(model, syst, rng, vary_phys, vary_nuis));
    return out;
}

// One greppable LOG line. Bash wrappers should grep for the literal
// "[SCALETEST]" prefix.
void emit_result(const BenchResult &r) {
    log<LOG_INFO>(L"[SCALETEST] tag=%1% N=%2% nbins=%3% nphys=%4% nnuis=%5% total_us=%6% per_call_us=%7%")
        % r.tag.c_str() % r.n_calls % r.nbins % r.nphys % r.nnuis
        % static_cast<long long>(r.total_us)
        % r.per_call_us;
}

BenchResult time_fillspectra(const std::string &tag,
                             const PROconfig &config,
                             const PROpeller &prop,
                             const PROsyst &syst,
                             const PROmodel &model,
                             const std::vector<Eigen::VectorXf> &params,
                             bool binned)
{
    BenchResult r;
    r.tag = tag;
    r.n_calls = static_cast<int>(params.size());
    r.nphys = static_cast<int>(model.nparams);
    r.nnuis = static_cast<int>(syst.GetNSplines());
    r.nbins = static_cast<int>(config.m_num_variable_bins_total[config.i_prime]);

    // Touch the result so the optimiser cannot fold the loop away.
    double sink = 0.0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (const auto &p : params) {
        PROspec s = FillSpectra(config, prop, syst, model, p, binned, config.i_prime);
        sink += static_cast<double>(s.Spec().sum());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    r.total_us    = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    r.per_call_us = (r.n_calls > 0) ? r.total_us / r.n_calls : 0.0;

    log<LOG_DEBUG>(L"%1% || (sink=%2% — anti-DCE only)") % __func__ % sink;
    emit_result(r);
    return r;
}

BenchResult time_metric(const std::string &tag,
                        const PROconfig &config,
                        PROmetric &metric,
                        const std::vector<Eigen::VectorXf> &params)
{
    const PROmodel &model = metric.GetModel();
    const PROsyst  &syst  = metric.GetSysts();

    BenchResult r;
    r.tag = tag;
    r.n_calls = static_cast<int>(params.size());
    r.nphys = static_cast<int>(model.nparams);
    r.nnuis = static_cast<int>(syst.GetNSplines());
    r.nbins = static_cast<int>(config.m_num_variable_bins_total[config.i_prime]);

    Eigen::VectorXf grad = Eigen::VectorXf::Zero(metric.nParams());

    double sink = 0.0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (const auto &p : params) {
        // No gradient: caller-controlled flag avoids finite-difference cost.
        sink += static_cast<double>(metric(p, grad, false));
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    r.total_us    = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    r.per_call_us = (r.n_calls > 0) ? r.total_us / r.n_calls : 0.0;

    log<LOG_DEBUG>(L"%1% || (sink=%2% — anti-DCE only)") % __func__ % sink;
    emit_result(r);
    return r;
}

BenchResult time_fit(const std::string &tag,
                     const PROconfig &config,
                     PROmetric &metric,
                     const std::vector<Eigen::VectorXf> &seed_params,
                     const PROfitterConfig &fitconfig,
                     uint32_t base_seed)
{
    const PROmodel &model = metric.GetModel();
    const PROsyst  &syst  = metric.GetSysts();

    BenchResult r;
    r.tag = tag;
    r.n_calls = static_cast<int>(seed_params.size());
    r.nphys = static_cast<int>(model.nparams);
    r.nnuis = static_cast<int>(syst.GetNSplines());
    r.nbins = static_cast<int>(config.m_num_variable_bins_total[config.i_prime]);

    double sink = 0.0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < r.n_calls; ++k) {
        PROfitter fitter(metric.UpperBound(), metric.LowerBound(), fitconfig,
                         base_seed + static_cast<uint32_t>(k));
        sink += static_cast<double>(fitter.Fit(metric, seed_params[k]));
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    r.total_us    = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    r.per_call_us = (r.n_calls > 0) ? r.total_us / r.n_calls : 0.0;

    log<LOG_DEBUG>(L"%1% || (sink=%2% — anti-DCE only)") % __func__ % sink;
    emit_result(r);
    return r;
}

}  // namespace

std::vector<BenchResult> run_scale_test(
    const PROconfig &config,
    const PROpeller &prop,
    PROmetric &metric,
    const PROfitterConfig &fitconfig,
    const BenchOptions &opts)
{
    const PROmodel &model = metric.GetModel();
    const PROsyst  &syst  = metric.GetSysts();

    std::vector<BenchResult> results;

    const int N_fill   = std::max(1, opts.N);
    const int N_metric = std::max(1, opts.N / 10);
    const int N_fit    = std::max(1, opts.N / 100);

    log<LOG_INFO>(L"%1% || ===== PRObench scale-test starting =====") % __func__;
    log<LOG_INFO>(L"%1% || N_fillspectra=%2%  N_metric=%3%  N_fit=%4%")
        % __func__ % N_fill % N_metric % N_fit;
    log<LOG_INFO>(L"%1% || nphys=%2%  nnuis=%3%  nbins=%4%")
        % __func__ % model.nparams % syst.GetNSplines() % config.m_num_variable_bins_total[config.i_prime];

    // Distinct sub-seeds so the three variation modes draw distinct sequences
    // while keeping the whole bench reproducible from opts.rng_seed.
    const uint32_t s_all  = opts.rng_seed ^ 0x1u;
    const uint32_t s_phys = opts.rng_seed ^ 0x2u;
    const uint32_t s_nuis = opts.rng_seed ^ 0x4u;

    // ---- (a–c) FillSpectra ----
    if (opts.tests & Bench_FillSpectra_All) {
        auto p = draw_param_set(model, syst, N_fill, s_all,  /*vary_phys*/true,  /*vary_nuis*/true);
        results.push_back(time_fillspectra("fillspectra_vary_all",  config, prop, syst, model, p, opts.binned));
    }
    if (opts.tests & Bench_FillSpectra_Phys) {
        auto p = draw_param_set(model, syst, N_fill, s_phys, true,  false);
        results.push_back(time_fillspectra("fillspectra_vary_phys", config, prop, syst, model, p, opts.binned));
    }
    if (opts.tests & Bench_FillSpectra_Nuis) {
        auto p = draw_param_set(model, syst, N_fill, s_nuis, false, true);
        results.push_back(time_fillspectra("fillspectra_vary_nuis", config, prop, syst, model, p, opts.binned));
    }

    // ---- (d–f) PROmetric() ----
    if (opts.tests & Bench_Metric_All) {
        auto p = draw_param_set(model, syst, N_metric, s_all,  true,  true);
        results.push_back(time_metric("metric_vary_all",  config, metric, p));
    }
    if (opts.tests & Bench_Metric_Phys) {
        auto p = draw_param_set(model, syst, N_metric, s_phys, true,  false);
        results.push_back(time_metric("metric_vary_phys", config, metric, p));
    }
    if (opts.tests & Bench_Metric_Nuis) {
        auto p = draw_param_set(model, syst, N_metric, s_nuis, false, true);
        results.push_back(time_metric("metric_vary_nuis", config, metric, p));
    }

    // ---- (g–i) PROfitter::Fit ----
    if (opts.tests & Bench_Fit_All) {
        auto p = draw_param_set(model, syst, N_fit, s_all,  true,  true);
        results.push_back(time_fit("fit_vary_all",  config, metric, p, fitconfig, s_all));
    }
    if (opts.tests & Bench_Fit_Phys) {
        auto p = draw_param_set(model, syst, N_fit, s_phys, true,  false);
        results.push_back(time_fit("fit_vary_phys", config, metric, p, fitconfig, s_phys));
    }
    if (opts.tests & Bench_Fit_Nuis) {
        auto p = draw_param_set(model, syst, N_fit, s_nuis, false, true);
        results.push_back(time_fit("fit_vary_nuis", config, metric, p, fitconfig, s_nuis));
    }

    log<LOG_INFO>(L"%1% || ===== PRObench scale-test complete: %2% tests =====")
        % __func__ % results.size();
    return results;
}

}  // namespace PRObench
}  // namespace PROfit
