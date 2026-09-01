#include "PROmetrics/PROchi_pearson.h"
#include "PROcess.h"
#include "PROdata.h"
#include "PROlog.h"
#include "PROmetric.h"
#include "PROtocall.h"

#include <Eigen/Eigen>

#include <algorithm>

using namespace PROfit;

// Floor on the Pearson mu: keeps the stat diagonal (and hence M) nonsingular in
// bins where the prediction AND the systematic covariance are both empty.
static constexpr float kMinPearsonMu = 1e-6f;

PROchi_pearson::PROchi_pearson(const std::string tag, const PROconfig &conin, const PROpeller &pin, const PROsyst *systin, const PROmodel &modelin, const PROdata &datain, EvalStrategy strat, bool shape_only, std::vector<float> physics_param_fixed) : PROmetric(), model_tag(tag), config(conin), peller(pin), syst(systin), model(modelin), data(datain), strat(strat), shape_only(shape_only), physics_param_fixed(physics_param_fixed), correlated_systematics(false) {
    last_value = 0.0; last_param = Eigen::VectorXf::Zero(model.nparams+syst->GetNSplines());
    fixed_index = -999;

    // Snapshot the config's fit-region mask (if any). Unlike PROchi, Pearson keeps
    // zero-data bins (their variance is the predicted mu > 0), so the mask is the
    // ONLY exclusion mechanism here: the chi2 below is reduced to pearson_active_idx.
    snapshotActiveBins(conin);
    if(hasActiveBinMask())
        for(Eigen::Index i = 0; i < datain.Spec().size(); ++i)
            if(binActive(i)) pearson_active_idx.push_back(i);

    // Build the correlation matrix between priors if configured to
    if (conin.m_mcgen_correlations.size()) {
        correlated_systematics = true;
        prior_covariance = Eigen::MatrixXf::Identity(syst->GetNSplines(), syst->GetNSplines());
        for (auto const &t: conin.m_mcgen_correlations) {
            auto itA = std::find(systin->spline_names.begin(), systin->spline_names.end(), std::get<0>(t));
            if (itA == systin->spline_names.end()) {
                log<LOG_WARNING>(L"%1% || Systematic correlation %2% not in list. Skipping.") % __func__ % std::get<0>(t).c_str();
                continue;
            }

            auto itB = std::find(systin->spline_names.begin(), systin->spline_names.end(), std::get<1>(t));
            if (itB == systin->spline_names.end()) {
                log<LOG_WARNING>(L"%1% || Systematic correlation %2% not in list. Skipping.") % __func__ % std::get<1>(t).c_str();
                continue;
            }

            int iA = std::distance(systin->spline_names.begin(), itA);
            int iB = std::distance(systin->spline_names.begin(), itB);

            // set correlations
            prior_covariance(iA, iB) = std::get<2>(t);
            prior_covariance(iB, iA) = std::get<2>(t);
        }
        prior_covariance = systin->spline_priors.asDiagonal() * prior_covariance * systin->spline_priors.asDiagonal();
        for(size_t i = 0; i < systin->spline_prior_types.size(); ++i) {
            if (systin->spline_prior_types[i] == SplinePriorType::Uniform) {
                prior_covariance.row(i).setZero();
                prior_covariance.col(i).setZero();
                prior_covariance(i, i) = 1.0f;
            }
        }
        prior_covariance_inv = prior_covariance.inverse();
    }
}

float PROchi_pearson::Pull(const Eigen::VectorXf &systs) {
    // No correlations: sum of squares
    Eigen::VectorXf centered = systs - syst->spline_centers;
    for(size_t i = 0; i < syst->spline_prior_types.size(); ++i) {
        if(syst->spline_prior_types[i] == SplinePriorType::Uniform) centered(i) = 0.0f;
    }
    if (!correlated_systematics) {
        return (centered.array().square() / syst->spline_priors.array().square()).sum();
    }
    return centered.dot(prior_covariance_inv * centered);
}

void PROchi_pearson::fixSpline(int fix, float valin){
    fixed_index=fix;
    fixed_val=valin;
    return;
}
float PROchi_pearson::operator()(const Eigen::VectorXf &param, Eigen::VectorXf &gradient){
    return PROchi_pearson::operator()(param, gradient, true);
}


float PROchi_pearson::operator()(const Eigen::VectorXf &param, Eigen::VectorXf &gradient, bool rungradient){
    call_count++;

    // Get Spectra from FillSpectra
    Eigen::VectorXf subvector1 = param.segment(0, model.nparams);
    log<LOG_DEBUG>(L"%1% || Created physics subvector with size %2%") % __func__ % subvector1.size();
     if(model.model_constraint){
        if(!model.model_constraint(subvector1)){
            // Keep value and gradient consistent for the minimizer.
            if(rungradient) gradient.setZero();
            return 1e10;
        }
    }

    Eigen::VectorXf subvector2 = param.segment(model.nparams, syst->GetNSplines());

    // strat != EventByEvent (not == BinnedChi2): matches the FD gradient
    // closures so BinnedGrad uses one consistent spectrum model and keeps the
    // fill cache valid.
    PROspec result = FillSpectra(config, peller, *syst, model, param, fs_cache, strat != EventByEvent, config.i_prime);

    Eigen::VectorXf collapsed_pred = CollapseMatrix(config, result.Spec());
    Eigen::VectorXf normdata = shape_only
        ? data.Normalize(config,result)
        : data.Spec();
    // Pearson diagonal: the predicted count mu at the CURRENT full parameter
    // point (physics + spline shifts) — the textbook Pearson statistic. Kept as
    // a VECTOR; already computed above for delta, so this is free. Floored at
    // kMinPearsonMu so M stays nonsingular where the systematic covariance is
    // also empty. Zero-data bins are kept (variance = mu).
    Eigen::VectorXf stat_diag = collapsed_pred.cwiseMax(kMinPearsonMu);

    // Collapsed systematic covariance without materializing the full-binning
    // dense diag(s)*F*diag(s) (see CollapsedScaledCovariance).
    Eigen::MatrixXf M = CollapsedScaledCovariance(config, syst->fractional_covariance, result.Spec());
    M.diagonal() += stat_diag;

    Eigen::VectorXf delta = collapsed_pred - normdata;

    // Fit-region mask: reduce to the active bins (Gaussian marginalization over the
    // rest). Explicit temporaries — assigning an indexed view of M back to M aliases.
    const Eigen::Map<const Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1>>
        aidx(pearson_active_idx.data(), (Eigen::Index)pearson_active_idx.size());
    if(hasActiveBinMask()) {
        Eigen::MatrixXf Mred = M(aidx, aidx);
        Eigen::VectorXf dred = delta(aidx);
        M = std::move(Mred);
        delta = std::move(dred);
    }

    float pull = Pull(subvector2);
    float dmsq_penalty = 0;
    float covar_portion = delta.dot(M.llt().solve(delta));
    float value = covar_portion + dmsq_penalty + pull;

    if(std::isnan(value)) {
        log<LOG_WARNING>(L"%1% || WARNING: Pearson chi2 is NaN. This is very bad.\n"
                L"covar_portion: %2%\npull: %3%\ndelta: %4%\n"
                L"mc spec: %5%\ndata spec: %6%")
            % __func__ % covar_portion % pull % delta % CollapseMatrix(config, result.Spec())
            % data.Spec();
                throw std::runtime_error("Pearson chi2 is nan.");
    }


    if(rungradient){
        // ----- Gradient mode dispatch (see PROmetric::GradientMode) -----
        // Identical structure to PROCNP but with the Pearson stat covariance:
        // M_stat = diag(mu) where mu is the collapsed CURRENT prediction, so it
        // depends on ALL parameters (physics AND splines). The Full modes rebuild
        // M_stat from the perturbed spectrum for every FD step (simpler than
        // CNP's physics-only rebuild); the Linearised modes intentionally freeze
        // M (including M_stat) at the base point — this is the Gauss-Newton
        // approximation, dropping the (M⁻¹δ)^T (dM/dθ) (M⁻¹δ) term that's
        // second-order in δ.
        GradientMode mode = gradient_mode;
        // Analytic gradient is implemented in PROchi only so far; the Pearson stat
        // covariance adds a diag(dμ/dθ) term that is not yet wired up. Use the
        // FD fallback mode (Gauss-Newton linearised, exact at the minimum).
        if (mode == GradientAnalytic) {
            static std::atomic<bool> warned_analytic{false};
            if(!warned_analytic.exchange(true))
                log<LOG_WARNING>(L"%1% || Analytic gradient not implemented for PROchi_pearson; falling back to %2%.") % __func__ % gradientModeName(GradientFallback);
            mode = GradientFallback;
        }
        const bool linearised = (mode == GradientCentralLin) || (mode == GradientOneSidedLin);
        const bool one_sided  = (mode == GradientOneSidedFull) || (mode == GradientOneSidedLin);
        const size_t nsyst = syst->GetNSplines();

        Eigen::VectorXf Minv_delta_b;
        Eigen::VectorXf pull_grad_nuis;
        if (linearised) {
            Minv_delta_b = M.llt().solve(delta);
            Eigen::VectorXf centered = subvector2 - syst->spline_centers;
            // Match Pull(): uniform-prior splines contribute NO pull — mask them
            // here too, or the gradient carries a phantom Gaussian pull the value
            // doesn't have.
            for(size_t i = 0; i < syst->spline_prior_types.size(); ++i)
                if(syst->spline_prior_types[i] == SplinePriorType::Uniform) centered(i) = 0.0f;
            if (!correlated_systematics) {
                pull_grad_nuis = 2.0f * centered.array() /
                                 (syst->spline_priors.array() * syst->spline_priors.array());
            } else {
                pull_grad_nuis = 2.0f * (prior_covariance_inv * centered);
                // dPull/dθ_i is exactly 0 for a uniform spline; zero it in case
                // Σ⁻¹ carries off-diagonal terms in those rows.
                for(size_t i = 0; i < syst->spline_prior_types.size(); ++i)
                    if(syst->spline_prior_types[i] == SplinePriorType::Uniform) pull_grad_nuis(i) = 0.0f;
            }
        }

        // Helper: build M (full collapsed, no idx slicing) at a perturbed
        // spectrum. The Pearson stat diagonal is ALWAYS rebuilt from the
        // perturbed prediction — every parameter (physics or spline) shifts mu.
        auto rebuild_gM_at = [&](const PROspec &rl) -> Eigen::MatrixXf {
            Eigen::MatrixXf gM = CollapsedScaledCovariance(config, syst->fractional_covariance, rl.Spec());
            gM.diagonal() += CollapseMatrix(config, rl.Spec()).cwiseMax(kMinPearsonMu);
            if(hasActiveBinMask()) {
                Eigen::MatrixXf gMred = gM(aidx, aidx);
                gM = std::move(gMred);
            }
            return gM;
        };

        // compute_delta_at: reduced delta at arbitrary param (uses base normdata).
        auto compute_delta_at = [&](const Eigen::VectorXf &param_at,
                                    Eigen::VectorXf &delta_out) -> bool {
            if(model.model_constraint &&
               !model.model_constraint(param_at.segment(0, model.nparams))) return false;
            PROspec rl = FillSpectra(config, peller, *syst, model, param_at, fs_cache,
                                     strat != EventByEvent, config.i_prime);
            delta_out = CollapseMatrix(config, rl.Spec()) - normdata;
            if(hasActiveBinMask()) {
                Eigen::VectorXf dred = delta_out(aidx);
                delta_out = std::move(dred);
            }
            return true;
        };

        // compute_chi2_at: full chi² at arbitrary param. Used by Full modes.
        auto compute_chi2_at = [&](const Eigen::VectorXf &param_at,
                                   float &chi2_out) -> bool {
            if(model.model_constraint &&
               !model.model_constraint(param_at.segment(0, model.nparams))) return false;
            PROspec rl = FillSpectra(config, peller, *syst, model, param_at, fs_cache,
                                     strat != EventByEvent, config.i_prime);
            Eigen::MatrixXf gM_lo = rebuild_gM_at(rl);
            Eigen::VectorXf dl    = CollapseMatrix(config, rl.Spec()) - normdata;
            if(hasActiveBinMask()) {
                Eigen::VectorXf dlred = dl(aidx);
                dl = std::move(dlred);
            }
            Eigen::VectorXf nuis  = param_at.segment(model.nparams, nsyst);
            chi2_out = dl.dot(gM_lo.llt().solve(dl)) + Pull(nuis);
            return true;
        };

        // One reusable work vector: perturb component i in place and restore,
        // instead of two full parameter-vector copies per FD parameter.
        Eigen::VectorXf param_work = param;

        for (size_t i = 0; i < model.nparams + nsyst; i++) {

            if(is_fixed.size() > 0 && is_fixed.at(i)) {
                gradient(i) = 0.0f;
                continue;
            }

            float h = (i < model.nparams) ? 1e-3f : 1e-4f;

            const float boundary_tol = 2.0f * std::numeric_limits<float>::epsilon();
            const bool at_lower = std::fabs(param(i) - lb(i)) < boundary_tol;
            const bool at_upper = std::fabs(ub(i) - param(i)) < boundary_tol;

            if (at_lower && at_upper) {
                gradient(i) = 0.0f;
                continue;
            }

            const bool boundary_step = (at_lower || at_upper);
            const int  sign          = boundary_step ? (at_lower ? 1 : -1) : 1;
            const bool use_central   = !boundary_step && !one_sided;

            if (linearised) {
                Eigen::VectorXf delta_plus, delta_minus;
                param_work(i) = param(i) + sign * h;
                bool ok_plus  = compute_delta_at(param_work,  delta_plus);
                bool ok_minus = true;
                if (use_central) {
                    param_work(i) = param(i) - sign * h;
                    ok_minus = compute_delta_at(param_work, delta_minus);
                }
                param_work(i) = param(i);

                Eigen::VectorXf ddelta_dtheta;
                if (use_central) {
                    if (!ok_plus && !ok_minus) { gradient(i) = 0.0f; continue; }
                    if (!ok_plus)  { gradient(i) = +1e10f; continue; }
                    if (!ok_minus) { gradient(i) = -1e10f; continue; }
                    ddelta_dtheta = (delta_plus - delta_minus) / (2.0f * h);
                } else {
                    if (!ok_plus) {
                        gradient(i) = sign * 1e10f;
                        if (boundary_step && sign * gradient(i) > 0) gradient(i) = 0.0f;
                        continue;
                    }
                    ddelta_dtheta = (sign * (delta_plus - delta)) / h;
                }

                float grad_i = 2.0f * Minv_delta_b.dot(ddelta_dtheta);
                if (i >= model.nparams) {
                    grad_i += pull_grad_nuis(i - model.nparams);
                }
                gradient(i) = grad_i;
            } else {
                if (use_central) {
                    float chi2_plus = 1e10f, chi2_minus = 1e10f;
                    param_work(i) = param(i) + sign * h;
                    compute_chi2_at(param_work,  chi2_plus);
                    param_work(i) = param(i) - sign * h;
                    compute_chi2_at(param_work, chi2_minus);
                    param_work(i) = param(i);
                    gradient(i) = (chi2_plus - chi2_minus) / (2.0f * h);
                } else {
                    float chi2_one = 0.0f;
                    param_work(i) = param(i) + sign * h;
                    const bool ok_one = compute_chi2_at(param_work, chi2_one);
                    param_work(i) = param(i);
                    if (!ok_one) {
                        gradient(i) = sign * 1e10f;
                        if (!std::isfinite(gradient(i))) gradient(i) = 0.0f;
                        continue;
                    }
                    gradient(i) = sign * (chi2_one - value) / h;
                }
            }

            if (boundary_step && sign * gradient(i) > 0) {
                gradient(i) = 0.0f;
            }
            if (!std::isfinite(gradient(i))) gradient(i) = 0.0f;
        }
    }

    //Update last param
    last_param = param;
    last_value = value;

    return value;}

float PROchi_pearson::getSingleChannelChi(size_t global_channel_index, const PROspec &cv, size_t var_index, const Eigen::MatrixXf &projection) {

    size_t nbin = config.m_channel_variable_bins[config.GetLocalChannelIndexFromGlobalChannelIndex(global_channel_index)][var_index].NBins();
    size_t startBin = config.GetCollapsedGlobalVariableBinStart(global_channel_index,var_index);

    // Restrict to this channel's active bins (mask is snapshotted for i_prime only).
    const bool masked = (var_index == (size_t)config.i_prime) && hasActiveBinMask();
    std::vector<Eigen::Index> local_idx;
    for(size_t b = 0; b < nbin; ++b)
        if(!masked || binActive((Eigen::Index)(startBin + b)))
            local_idx.push_back((Eigen::Index)(startBin + b));
    if(local_idx.empty()) return 0.0f;
    const Eigen::Map<const Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1>>
        idx(local_idx.data(), (Eigen::Index)local_idx.size());

    Eigen::VectorXf normdata = shape_only
        ? data.Normalize(config,cv)
        : data.Spec();
    Eigen::VectorXf collapsed_cv = CollapseMatrix(config, cv.Spec());
    Eigen::MatrixXf collapsed_stat_covariance = Eigen::MatrixXf::Zero(data.Spec().size(), data.Spec().size());
    for(long i = 0; i < data.Spec().size(); ++i)
        collapsed_stat_covariance(i,i) = std::max(collapsed_cv(i), kMinPearsonMu);

    Eigen::MatrixXf M;
    if(syst->GetNCovar()){
        Eigen::MatrixXf collapsed_full_covariance = CollapsedScaledCovariance(config, syst->fractional_covariance, cv.Spec());
        M = collapsed_full_covariance(idx, idx) + collapsed_stat_covariance(idx, idx);
    } else {
        M = collapsed_stat_covariance(idx, idx);
    }

    Eigen::VectorXf delta = (collapsed_cv - normdata)(idx);
    if(projection.size()) {
        Eigen::MatrixXf active_projection(projection.rows(), idx.size());
        for(Eigen::Index col = 0; col < idx.size(); ++col) {
            active_projection.col(col) = projection.col(idx(col) - (Eigen::Index)startBin);
        }
        M = active_projection * M * active_projection.transpose();
        delta = active_projection * delta;
    }
    float covar_portion = delta.dot(M.llt().solve(delta));
    float value = covar_portion;

    return value;
}

void PROchi_pearson::print(const Eigen::VectorXf &param){

     log<LOG_INFO>(L"%1% || input param: %2% ") % __func__ % param;

    // Get Spectra from FillSpectra
    Eigen::VectorXf subvector1 = param.segment(0, model.nparams);
    Eigen::VectorXf subvector2 = param.segment(model.nparams, syst->GetNSplines());

    PROspec result = FillSpectra(config, peller, *syst, model, param, strat == BinnedChi2, config.i_prime);
    log<LOG_INFO>(L"%1% || Result Spectra: ") % __func__ ;
    result.Print();

    // Pearson stat covariance: diag of the collapsed CURRENT prediction.
    Eigen::VectorXf collapsed_pred = CollapseMatrix(config, result.Spec());
    Eigen::MatrixXf collapsed_stat_covariance = collapsed_pred.cwiseMax(kMinPearsonMu).asDiagonal();

    log<LOG_INFO>(L"%1% || Pearson (prediction) stat Covariance is : \n %2%") % __func__ % collapsed_stat_covariance;
    Eigen::MatrixXf full_covariance = result.Spec().asDiagonal() * (syst->fractional_covariance) * result.Spec().asDiagonal();

    Eigen::MatrixXf collapsed_full_covariance = CollapseMatrix(config, full_covariance);
    Eigen::MatrixXf M = collapsed_stat_covariance + collapsed_full_covariance;

    // Calculate Chi^2  value
    Eigen::VectorXf delta  = collapsed_pred - data.Spec();
    log<LOG_INFO>(L"%1% || DataSpectra Spectra: ") % __func__ ;
    data.Print();

    float pull = Pull(subvector2);
    float dmsq_penalty = 0;
    float covar_portion = delta.dot(M.llt().solve(delta));
    float value = covar_portion + dmsq_penalty + pull;


    log<LOG_INFO>(L"%1% || Result chi^2/value : %2% ") % __func__ % value;
    if(std::isnan(value)) {
        log<LOG_WARNING>(L"%1% || WARNING: Pearson chi2 is NaN. This is very bad.\n"
                L"covar_portion: %2%\npull: %3%\ndelta: %4%\n"
                L"mc spec: %5%\ndata spec: %6%")
            % __func__ % covar_portion % pull % delta % CollapseMatrix(config, result.Spec())
            % data.Spec();
    }


    Eigen::VectorXf gradient = Eigen::VectorXf::Constant(param.size(),0);
    float dval = 1e-4;
    for (size_t i = 0; i < model.nparams+syst->GetNSplines(); i++) {
        Eigen::VectorXf tmpParams = param;
        int sgn = ((param(i) - last_param(i)) > 0) - ((param(i) - last_param(i)) < 0);
        if(!sgn) sgn = 1;
        tmpParams(i) = param(i) + sgn * dval;

        Eigen::VectorXf subvector1 = tmpParams.segment(0, model.nparams);
        Eigen::VectorXf subvector2 = tmpParams.segment(model.nparams, syst->GetNSplines());

        // If this new gradient evaluation point violates unitarity, set the gradient to a large value
        if(model.model_constraint){
            if(!model.model_constraint(subvector1)){
                gradient(i) = sgn * 1e10;
                continue;
            }
        }

        PROspec result = FillSpectra(config, peller, *syst, model, tmpParams, strat != EventByEvent, config.i_prime);

        // The Pearson stat diagonal depends on the full prediction, so it is
        // rebuilt for EVERY perturbed parameter (physics and splines alike).
        Eigen::VectorXf new_collapsed_pred = CollapseMatrix(config, result.Spec());
        Eigen::MatrixXf new_collapsed_stat_covariance = new_collapsed_pred.cwiseMax(kMinPearsonMu).asDiagonal();

        Eigen::MatrixXf full_covariance = result.Spec().asDiagonal() * (syst->fractional_covariance) * result.Spec().asDiagonal();

        Eigen::MatrixXf collapsed_full_covariance = CollapseMatrix(config, full_covariance);
        Eigen::MatrixXf gM = new_collapsed_stat_covariance + collapsed_full_covariance;

        // Calculate Chi^2  value
        Eigen::VectorXf delta  = new_collapsed_pred - data.Spec();

        float pull = Pull(subvector2);
        float value_grad = delta.dot(gM.llt().solve(delta)) + pull;

        gradient(i) = (value_grad-value)/(tmpParams(i) - param(i));
    }
    log<LOG_INFO>(L"%1% || Result grad is : %2% ") % __func__ % gradient;

    log<LOG_INFO>(L"%1% || value %2%, last_value %3%, pull") % __func__ % value  % last_value % pull;
    log<LOG_INFO>(L"%1% || FINISHED ITERATION got vals: %2% %3%") % __func__ % value % last_value ;

    //Update last param
    last_param = param;
    last_value = value;


    return;
}
