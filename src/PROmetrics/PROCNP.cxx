#include "PROmetrics/PROCNP.h"
#include "PROcess.h"
#include "PROdata.h"
#include "PROlog.h"
#include "PROmetric.h"
#include "PROtocall.h"

#include <Eigen/Eigen>

#include <algorithm>

using namespace PROfit;


PROCNP::PROCNP(const std::string tag, const PROconfig &conin, const PROpeller &pin, const PROsyst *systin, const PROmodel &modelin, const PROdata &datain, EvalStrategy strat, bool shape_only, std::vector<float> physics_param_fixed) : PROmetric(), model_tag(tag), config(conin), peller(pin), syst(systin), model(modelin), data(datain), strat(strat), shape_only(shape_only), physics_param_fixed(physics_param_fixed), correlated_systematics(false) {
    last_value = 0.0; last_param = Eigen::VectorXf::Zero(model.nparams+syst->GetNSplines());
    fixed_index = -999;
    gradient_mode = GradientOneSidedFull; ///< Default for PROCNP: one-sided forward FD on full chi² (~2× faster).

    // Snapshot the config's fit-region mask (if any). Unlike PROchi, CNP keeps
    // zero-data bins (mu/2 substitution), so the mask is the ONLY exclusion
    // mechanism here: the chi2 below is reduced to cnp_active_idx.
    snapshotActiveBins(conin);
    if(hasActiveBinMask())
        for(Eigen::Index i = 0; i < datain.Spec().size(); ++i)
            if(binActive(i)) cnp_active_idx.push_back(i);

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
        prior_covariance_inv = prior_covariance.inverse();
    }
}

float PROCNP::Pull(const Eigen::VectorXf &systs) {
    // No correlations: sum of squares
    Eigen::VectorXf centered = systs - syst->spline_centers;
    if (!correlated_systematics) {
        return (centered.array().square() / syst->spline_priors.array().square()).sum();
    }
    return centered.dot(prior_covariance_inv * centered);
}

Eigen::VectorXf PROCNP::cachedNoshiftCollapsedCV(const Eigen::VectorXf &phys, Eigen::Index param_size) {
    if(cnp_cv_cache_valid && cnp_cached_phys.size() == phys.size() && cnp_cached_phys == phys)
        return cnp_cached_collapsed_cv;
    Eigen::VectorXf noshiftvec = Eigen::VectorXf::Zero(param_size);
    noshiftvec.head(model.nparams) = phys;
    cnp_cached_collapsed_cv = CollapseMatrix(config, FillSpectra(config, peller, *syst, model, noshiftvec, strat != EventByEvent).Spec());
    cnp_cached_phys = phys;
    cnp_cv_cache_valid = true;
    return cnp_cached_collapsed_cv;
}

void PROCNP::fixSpline(int fix, float valin){
    fixed_index=fix;
    fixed_val=valin;
    return;
}
float PROCNP::operator()(const Eigen::VectorXf &param, Eigen::VectorXf &gradient){
    return PROCNP::operator()(param, gradient, true);
}


float PROCNP::operator()(const Eigen::VectorXf &param, Eigen::VectorXf &gradient, bool rungradient){
    call_count++;

       //size_t nparams = model->nparams+syst->GetNSplines();
    //size_t nsyst = syst->GetNSplines();

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
    //log<LOG_DEBUG>(L"%1% || Created spline subvector with size %2%") % __func__ % subvector2.size();

    // strat != EventByEvent (not == BinnedChi2): matches the FD gradient
    // closures so BinnedGrad uses one consistent spectrum model and keeps the
    // fill cache valid.
    PROspec result = FillSpectra(config, peller, *syst, model, param, fs_cache, strat != EventByEvent, config.i_prime);


    Eigen::VectorXf collapsed_cv = cachedNoshiftCollapsedCV(subvector1, param.size());
    Eigen::VectorXf normdata = shape_only
        ? data.Normalize(config,result)
        : data.Spec();
    // CNP diagonal 3/(1/n + 2/mu) with mu from the physics-only CV, kept as a
    // VECTOR (the old code allocated and zero-filled a dense N x N per call
    // just to hold a diagonal). Guard mu <= 0: it would zero the diagonal and
    // make M singular wherever the systematic covariance is also empty. The
    // zero-data test uses normdata (consistent with the gradient rebuild path).
    constexpr float kMinCNPMu = 1e-6f;
    Eigen::VectorXf stat_diag(data.Spec().size());
    for(long i = 0; i < data.Spec().size(); ++i) {
        const float mu = std::max(collapsed_cv(i), kMinCNPMu);
        stat_diag(i) = normdata(i) == 0 ? mu/2 :
            3 / (1.0 / normdata(i) + 2.0 / mu);
    }

    // Collapsed systematic covariance without materializing the full-binning
    // dense diag(s)*F*diag(s) (see CollapsedScaledCovariance).
    Eigen::MatrixXf M = CollapsedScaledCovariance(config, syst->fractional_covariance, result.Spec());
    M.diagonal() += stat_diag;

    Eigen::VectorXf delta = CollapseMatrix(config, result.Spec()) - normdata;

    // Fit-region mask: reduce to the active bins (Gaussian marginalization over the
    // rest). Explicit temporaries — assigning an indexed view of M back to M aliases.
    const Eigen::Map<const Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1>>
        aidx(cnp_active_idx.data(), (Eigen::Index)cnp_active_idx.size());
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
        log<LOG_WARNING>(L"%1% || WARNING: CNP chi2 is NaN. This is very bad.\n"
                L"covar_portion: %2%\npull: %3%\ndelta: %4%\n"
                L"mc spec: %5%\ndata spec: %6%")
            % __func__ % covar_portion % pull % delta % CollapseMatrix(config, result.Spec())
            % data.Spec();
                throw std::runtime_error("CNP chi2 is nan.");
    }


    if(rungradient){
        // ----- Gradient mode dispatch (see PROmetric::GradientMode) -----
        // Identical structure to PROchi but with PROCNP's CNP-style stat
        // covariance: M_stat depends on collapsed_cv (which depends on physics
        // params) AND on normdata, so for physics-FD steps M_stat must be
        // rebuilt with the perturbed CV. The Full modes do this; the
        // Linearised modes intentionally freeze M (including M_stat) at the
        // base point — this is the Gauss-Newton approximation, dropping the
        // (M⁻¹δ)^T (dM/dθ) (M⁻¹δ) term that's second-order in δ.
        const GradientMode mode = gradient_mode;
        const bool linearised = (mode == GradientCentralLin) || (mode == GradientOneSidedLin);
        const bool one_sided  = (mode == GradientOneSidedFull) || (mode == GradientOneSidedLin);
        const size_t nsyst = syst->GetNSplines();

        Eigen::VectorXf Minv_delta_b;
        Eigen::VectorXf pull_grad_nuis;
        if (linearised) {
            Minv_delta_b = M.llt().solve(delta);
            const Eigen::VectorXf centered = subvector2 - syst->spline_centers;
            if (!correlated_systematics) {
                pull_grad_nuis = 2.0f * centered.array() /
                                 (syst->spline_priors.array() * syst->spline_priors.array());
            } else {
                pull_grad_nuis = 2.0f * (prior_covariance_inv * centered);
            }
        }

        // Helper: build M (full collapsed, no idx slicing) at perturbed param.
        // For physics-FD the stat covariance is rebuilt from collapsed_cv at
        // the perturbed physics; for nuisance-FD the cached base stat cov is
        // reused (collapsed_cv depends on physics only).
        auto rebuild_gM_at = [&](const Eigen::VectorXf &param_at,
                                 size_t i_perturbed,
                                 const PROspec &rl) -> Eigen::MatrixXf {
            Eigen::VectorXf new_stat = stat_diag;
            if (i_perturbed < model.nparams) {
                Eigen::VectorXf cv_p = cachedNoshiftCollapsedCV(
                    param_at.segment(0, model.nparams), param.size());
                for (long j = 0; j < data.Spec().size(); ++j) {
                    const float mu = std::max(cv_p(j), kMinCNPMu);
                    new_stat(j) = normdata(j) == 0
                        ? mu / 2
                        : 3 / (1.0 / normdata(j) + 2.0 / mu);
                }
            }
            Eigen::MatrixXf gM = CollapsedScaledCovariance(config, syst->fractional_covariance, rl.Spec());
            gM.diagonal() += new_stat;
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
                                   size_t i_perturbed, float &chi2_out) -> bool {
            if(model.model_constraint &&
               !model.model_constraint(param_at.segment(0, model.nparams))) return false;
            PROspec rl = FillSpectra(config, peller, *syst, model, param_at, fs_cache,
                                     strat != EventByEvent, config.i_prime);
            Eigen::MatrixXf gM_lo = rebuild_gM_at(param_at, i_perturbed, rl);
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
                    compute_chi2_at(param_work,  i, chi2_plus);
                    param_work(i) = param(i) - sign * h;
                    compute_chi2_at(param_work, i, chi2_minus);
                    param_work(i) = param(i);
                    gradient(i) = (chi2_plus - chi2_minus) / (2.0f * h);
                } else {
                    float chi2_one = 0.0f;
                    param_work(i) = param(i) + sign * h;
                    const bool ok_one = compute_chi2_at(param_work, i, chi2_one);
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

    //log<LOG_DEBUG>(L"%1% || value %2%, last_value %3%, pull") % __func__ % value  % last_value % pull;
    //if(rungradient){
    //    log<LOG_DEBUG>(L"%1% || FINISHED ITERATION got vals: %2% %3%") % __func__ % value % last_value ;
    //    log<LOG_DEBUG>(L"%1% || FINISHED ITERATION par %2% ") % __func__ % param ;
    //    log<LOG_DEBUG>(L"%1% || FINISHED ITERATION grad %2% ") % __func__ % gradient ;
    //}


    //Update last param
    last_param = param;
    last_value = value;

    return value;}

float PROCNP::getSingleChannelChi(size_t global_channel_index, const PROspec &cv, size_t var_index, const Eigen::MatrixXf &projection) {

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
    Eigen::MatrixXf collapsed_stat_covariance = Eigen::MatrixXf::Zero(data.Spec().size(), data.Spec().size());
    Eigen::VectorXf collapsed_cv = CollapseMatrix(config, cv.Spec());
    for(long i = 0; i < data.Spec().size(); ++i)
        collapsed_stat_covariance(i,i) = data.Spec()(i) == 0 ? collapsed_cv(i)/2 :
            3 / (1.0 / normdata(i) + 2.0 / collapsed_cv(i));

    Eigen::MatrixXf M;
    if(syst->GetNCovar()){
        Eigen::MatrixXf collapsed_full_covariance = CollapsedScaledCovariance(config, syst->fractional_covariance, cv.Spec());
        M = collapsed_full_covariance(idx, idx) + collapsed_stat_covariance(idx, idx);
    } else {
        M = collapsed_stat_covariance(idx, idx);
    }

    Eigen::VectorXf delta = (CollapseMatrix(config, cv.Spec()) - normdata)(idx);
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

void PROCNP::print(const Eigen::VectorXf &param){

     log<LOG_INFO>(L"%1% || input param: %2% ") % __func__ % param;



    // Get Spectra from FillSpectra
    Eigen::VectorXf subvector1 = param.segment(0, model.nparams);
    Eigen::VectorXf noshiftvec = Eigen::VectorXf::Zero(param.size());
    noshiftvec.head(model.nparams) = subvector1;
    Eigen::VectorXf subvector2 = param.segment(model.nparams, syst->GetNSplines());

    PROspec result = FillSpectra(config, peller, *syst, model, param, strat == BinnedChi2);
    log<LOG_INFO>(L"%1% || Result Spectra: ") % __func__ ;
    result.Print();

    PROspec cv = FillSpectra(config, peller, *syst, model, noshiftvec, strat != EventByEvent);
    log<LOG_INFO>(L"%1% || CV is : \n ") % __func__ ;
    cv.Print();
    Eigen::MatrixXf collapsed_data_stat_covariance = data.Spec().array().matrix().asDiagonal();
    Eigen::MatrixXf mc_stat_covariance = cv.Spec().array().matrix().asDiagonal();
    Eigen::MatrixXf collapsed_mc_stat_covariance = CollapseMatrix(config, mc_stat_covariance);

    log<LOG_INFO>(L"%1% || data stat Covariance is : \n %2%") % __func__ % collapsed_data_stat_covariance;
    log<LOG_INFO>(L"%1% || mc stat Covariance is : \n %2%") % __func__ % collapsed_mc_stat_covariance;

    Eigen::MatrixXf collapsed_stat_covariance = 3 * (collapsed_data_stat_covariance.inverse() + 2 * collapsed_mc_stat_covariance.inverse()).inverse();
    for(int i = 0; i < collapsed_stat_covariance.cols(); ++i) {
        // If data bin is 0, this will be nan
        if(std::isnan(collapsed_stat_covariance(i,i)))
            collapsed_stat_covariance(i,i) = mc_stat_covariance(i)/2;
    }

    log<LOG_INFO>(L"%1% || Collapsed stat Covariance is : \n %2%") % __func__ % collapsed_stat_covariance;
    Eigen::MatrixXf full_covariance = result.Spec().asDiagonal() * (syst->fractional_covariance) * result.Spec().asDiagonal();

    Eigen::MatrixXf collapsed_full_covariance = CollapseMatrix(config, full_covariance);
    Eigen::MatrixXf M = collapsed_stat_covariance + collapsed_full_covariance;

    // Calculate Chi^2  value
    Eigen::VectorXf delta  = CollapseMatrix(config,result.Spec()) - data.Spec();
    log<LOG_INFO>(L"%1% || DataSpectra Spectra: ") % __func__ ;
    data.Print();

    float pull = Pull(subvector2);
    float dmsq_penalty = 0;
    float covar_portion = delta.dot(M.llt().solve(delta));
    float value = covar_portion + dmsq_penalty + pull;


    log<LOG_INFO>(L"%1% || Result chi^2/value : %2% ") % __func__ % value;
    if(std::isnan(value)) {
        log<LOG_WARNING>(L"%1% || WARNING: CNP chi2 is NaN. This is very bad.\n"
                L"covar_portion: %2%\npull: %3%\ndelta: %4%\n"
                L"mc spec: %5%\ndata spec: %6%")
            % __func__ % covar_portion % pull % delta % CollapseMatrix(config, result.Spec())
            % data.Spec();
    }


    Eigen::VectorXf gradient = Eigen::VectorXf::Constant(param.size(),0);
    float dval = 1e-4;
    for (size_t i = 0; i < model.nparams+syst->GetNSplines(); i++) {
        //Eigen::VectorXf tmpParams = last_param;
        Eigen::VectorXf tmpParams = param;
        int sgn = ((param(i) - last_param(i)) > 0) - ((param(i) - last_param(i)) < 0);
        if(!sgn) sgn = 1;
        //if(fitparams.size() != 0 && i == 1 && param(i) < -4 + dval) sgn = 1;
        //else if(fitparams.size() != 0 && i == 1 && param(i) > 0 - dval) sgn = -1;
        tmpParams(i) = /*param(i) != last_param(i) ? param(i) :*/ param(i) + sgn * dval;

        Eigen::VectorXf subvector1 = tmpParams.segment(0, model.nparams);
        //log<LOG_DEBUG>(L"%1% || Created physics subvector with size %2%") % __func__ % subvector1.size();
        Eigen::VectorXf subvector2 = tmpParams.segment(model.nparams, syst->GetNSplines());
        //log<LOG_DEBUG>(L"%1% || Created spline subvector with size %2%") % __func__ % subvector2.size();

        // If this new gradient evaluation point violates unitarity, set the gradient to a large value
        if(model.model_constraint){
            if(!model.model_constraint(subvector1)){
                //log<LOG_ERROR>(L"%1% || WARNING In PROCNP: Gradient evaluation point violates unitarity. Setting gradient to large value.") % __func__;
                gradient(i) = sgn * 1e10;
                continue;
            }
        }

        PROspec result = FillSpectra(config, peller, *syst, model, tmpParams, strat != EventByEvent);

        Eigen::MatrixXf new_collapsed_stat_covariance = collapsed_stat_covariance;
        if(i < model.nparams) {
            PROspec cv = FillSpectra(config, peller, *syst, model, subvector1, strat != EventByEvent);
            Eigen::MatrixXf collapsed_data_stat_covariance = data.Spec().array().matrix().asDiagonal();
            Eigen::MatrixXf mc_stat_covariance = cv.Spec().array().matrix().asDiagonal();
            Eigen::MatrixXf collapsed_mc_stat_covariance = CollapseMatrix(config, mc_stat_covariance);
            new_collapsed_stat_covariance = 3 * (collapsed_data_stat_covariance.inverse() + 2 * collapsed_mc_stat_covariance.inverse()).inverse();
            for(int i = 0; i < new_collapsed_stat_covariance.cols(); ++i) {
                // If data bin is 0, this will be nan
                if(std::isnan(new_collapsed_stat_covariance(i,i)))
                    new_collapsed_stat_covariance(i,i) = mc_stat_covariance(i)/2;
            }
        }

        Eigen::MatrixXf full_covariance = result.Spec().asDiagonal() * (syst->fractional_covariance) * result.Spec().asDiagonal();

        Eigen::MatrixXf collapsed_full_covariance = CollapseMatrix(config, full_covariance);
        Eigen::MatrixXf gM = collapsed_stat_covariance + collapsed_full_covariance;

        // Calculate Chi^2  value
        Eigen::VectorXf delta  = CollapseMatrix(config,result.Spec()) - data.Spec();

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


