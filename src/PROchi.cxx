#include "PROchi.h"
#include "PROcess.h"
#include "PROdata.h"
#include "PROlog.h"
#include "PROtocall.h"
#include <Eigen/Eigen>
#include <cmath>
using namespace PROfit;


PROchi::PROchi(const std::string tag, const PROconfig &conin, const PROpeller &pin, const PROsyst *systin, const PROmodel &modelin, const PROdata &datain, EvalStrategy strat, bool shape_only, std::vector<float> physics_param_fixed) : PROmetric(), model_tag(tag), config(conin), peller(pin), syst(systin), model(modelin), data(datain), strat(strat), shape_only(shape_only), physics_param_fixed(physics_param_fixed), correlated_systematics(false) {
    last_value = 0.0; last_param = Eigen::VectorXf::Zero(model.nparams+syst->GetNSplines()); 
    fixed_index = -999;

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
    }
    
    // GP: What do you do if the MC has 0 events in a bin?
    //     Proposed solution (hack?) here. Set the error to 1. This will return
    //     the correct answer if there are no data events in the bin. It is a bit
    //     iffier if there are data events in the bin, we may want to implement some
    //     error handling there.
    collapsed_stat_covariance = data.Spec().array().cwiseMax(1).matrix().asDiagonal();
}

float PROchi::Pull(const Eigen::VectorXf &systs) {
    // No correlations: sum of squares
    Eigen::VectorXf centered = systs - syst->spline_centers;
    if (!correlated_systematics) {
        return (centered.array().square() / syst->spline_priors.array().square()).sum();
    }

    // Otherwise dot onto covariance
    return centered.dot(prior_covariance.inverse() * centered);
}

void PROchi::fixSpline(int fix, float valin){
    fixed_index=fix;
    fixed_val=valin;
    return;
}
float PROchi::operator()(const Eigen::VectorXf &param, Eigen::VectorXf &gradient){
    return PROchi::operator()(param, gradient, true);
}

float PROchi::operator()(const Eigen::VectorXf &param, Eigen::VectorXf &gradient, bool rungradient){
    size_t nparams = nParams();
    size_t nsyst = syst->GetNSplines();
    //log<LOG_DEBUG>(L"%1% || nparams is %2%, nsyst is %3% ") % __func__ % nparams % nsyst;    

    Eigen::VectorXf subvector1 = param.segment(0, model.nparams);
    //log<LOG_DEBUG>(L"%1% || Created physics subvector with size %2%") % __func__ % subvector1.size();
    if(model.model_constraint){
        if(!model.model_constraint(subvector1)){
            return 1e10;
        }
    }

    // Get Spectra from FillSpectra
    Eigen::VectorXf subvector2 = param.segment(nparams - nsyst, nsyst);
    
    PROspec result = FillSpectra(config, peller, *syst, model, param, strat == BinnedChi2, config.i_prime);

    Eigen::MatrixXf inverted_collapsed_full_covariance(config.m_num_variable_bins_total_collapsed[config.i_prime],config.m_num_variable_bins_total_collapsed[config.i_prime]);

    Eigen::MatrixXf diag = result.Spec().array().matrix().asDiagonal(); 
    Eigen::MatrixXf full_covariance = diag*(syst->fractional_covariance)*diag;
   
    Eigen::VectorXf normdata = shape_only 
        ? data.Normalize(config,result)
        : data.Spec();

    Eigen::MatrixXf collapsed_full_covariance = CollapseMatrix(config, full_covariance); 
    collapsed_stat_covariance = ( normdata).matrix().asDiagonal();
    inverted_collapsed_full_covariance = (collapsed_stat_covariance+ collapsed_full_covariance).inverse();

    Eigen::VectorXf delta  = CollapseMatrix(config,result.Spec()) - (shape_only ? data.Normalize(config,result) : data.Spec());
    float pull = Pull(subvector2);
    float covar_portion = (delta.transpose())*inverted_collapsed_full_covariance*(delta);
    float value = covar_portion + pull;

    if(std::isnan(value) || value!=value) {
        log<LOG_ERROR>(L"%1% || ERROR: CNP chi2 is NaN (%2%). This is very bad.\n"
                L"covar_portion: %3%\npull: %4%\ndelta: %5%\n"
                L"mc spec: %6%\ndata spec: %7%")
            % __func__ % value % covar_portion % pull % delta % CollapseMatrix(config, result.Spec())
            % data.Spec();
        throw std::runtime_error("NANs in Chi().");

    }

    if(rungradient){
        for (size_t i = 0; i < model.nparams+syst->GetNSplines(); i++) {

            if(is_fixed.size()>0){
                if(is_fixed.at(i)) {
                    gradient(i) = 0.0f;
                    continue;  
                }
            }

            float h;
            h = 1e-4f;
            if(i < model.nparams) {
                h = 1e-3f;
            }

            float boundary_tol = 2.0f*std::numeric_limits<float>::epsilon();
            bool at_lower = fabs(param(i) - lb(i)) < boundary_tol;
            bool at_upper = fabs(ub(i) - param(i)) < boundary_tol;
        
            if(at_lower && at_upper){//shouldnt happen, if ix fixed working
                    gradient(i) = 0.0f;
                    continue;
            }

            int sign = (at_lower? 1 : (at_upper ? -1 : -999) );


            if(at_lower || at_upper) {
                Eigen::VectorXf param_plus = param;
                param_plus(i) = param(i) + sign*h;

                float chi2_oneside;
                // Calculate chi2_plus or chi2_minus, depending on boundary
                PROspec result = FillSpectra(config, peller, *syst, model, param_plus, strat != EventByEvent,config.i_prime);

                Eigen::MatrixXf diag = result.Spec().array().matrix().asDiagonal();
                Eigen::MatrixXf full_covariance = diag*(syst->fractional_covariance)*diag;
                Eigen::MatrixXf collapsed_full_covariance = CollapseMatrix(config, full_covariance);
                Eigen::MatrixXf inverted_collapsed_full_covariance = (collapsed_stat_covariance + collapsed_full_covariance).inverse();

                Eigen::VectorXf delta = CollapseMatrix(config,result.Spec()) - normdata;
                Eigen::VectorXf subvector2 = param_plus.segment(model.nparams, syst->GetNSplines());
                float pull = Pull(subvector2);
                chi2_oneside = (delta.transpose())*inverted_collapsed_full_covariance*(delta) + pull;

                gradient(i) = sign*(chi2_oneside - value) / h;

                // If gradient suggests going further out of bounds, set to zero
                if(sign*gradient(i) > 0) {
                    gradient(i) = 0.0f;//-sign*h/2.0f;  // Minimum is at boundary, really (small bounce)
                }
            }else{

                // Central difference method
                Eigen::VectorXf param_plus = param;
                Eigen::VectorXf param_minus = param;
                param_plus(i) = param(i) + h;
                param_minus(i) = param(i) - h;

                // Calculate chi2 at both points
                float chi2_plus, chi2_minus;

                // Plus point
                {
                    PROspec result = FillSpectra(config, peller, *syst, model, param_plus, strat != EventByEvent,config.i_prime);

                    Eigen::MatrixXf diag = result.Spec().array().matrix().asDiagonal();
                    Eigen::MatrixXf full_covariance = diag*(syst->fractional_covariance)*diag;
                    Eigen::MatrixXf collapsed_full_covariance = CollapseMatrix(config, full_covariance);
                    Eigen::MatrixXf inverted_collapsed_full_covariance = (collapsed_stat_covariance + collapsed_full_covariance).inverse();

                    Eigen::VectorXf delta = CollapseMatrix(config,result.Spec()) - normdata;
                    Eigen::VectorXf subvector2 = param_plus.segment(model.nparams, syst->GetNSplines());
                    float pull = Pull(subvector2);
                    chi2_plus = (delta.transpose())*inverted_collapsed_full_covariance*(delta) + pull;
                }

                // Minus point
                {
                    PROspec result = FillSpectra(config, peller, *syst, model, param_minus, strat != EventByEvent,config.i_prime);


                    Eigen::MatrixXf full_covariance = diag*(syst->fractional_covariance)*diag;
                    Eigen::MatrixXf collapsed_full_covariance = CollapseMatrix(config, full_covariance);
                    Eigen::MatrixXf inverted_collapsed_full_covariance = (collapsed_stat_covariance + collapsed_full_covariance).inverse();

                    Eigen::VectorXf delta = CollapseMatrix(config,result.Spec()) - normdata;
                    Eigen::VectorXf subvector2 = param_minus.segment(model.nparams, syst->GetNSplines());
                    float pull = Pull(subvector2);
                    chi2_minus = (delta.transpose())*inverted_collapsed_full_covariance*(delta) + pull;
                }

                // Central difference formula
                gradient(i) = (chi2_plus - chi2_minus) / (2.0f * h);
                //log<LOG_DEBUG>(L"%1% || On grad %2% result %3% with chi2_plus %4% chi2_minus %5% and h %6%") % __func__ % i % gradient(i) % chi2_plus % chi2_minus % h;

            }
            // Sanity check
            if(!std::isfinite(gradient(i))) {
                gradient(i) = 0.0f;
            }

        }
    }



    //Update last param
    last_param = param;
    last_value = value;

    return value;
}

float PROchi::getSingleChannelChi(size_t global_channel_index, const PROspec & cv, size_t var_index) {

    size_t nbin = config.m_channel_variable_bins[config.GetLocalChannelIndexFromGlobalChannelIndex(global_channel_index)][var_index].NBins();
    size_t startBin = config.GetCollapsedGlobalVariableBinStart(global_channel_index, var_index);


    if(shape_only)
        collapsed_stat_covariance = (data.Normalize(config,cv)).matrix().asDiagonal();

    Eigen::MatrixXf inverted_collapsed_full_covariance(nbin,nbin);
    //only calculate a syst covariance if we have any covariance parameters as defined in the xml
    if(syst->GetNCovar()){
        // Calculate Full Syst Covariance matrix
        Eigen::MatrixXf diag =  cv.Spec().array().matrix().asDiagonal(); 
        Eigen::MatrixXf full_covariance =  diag*(syst->fractional_covariance)*diag;

        // Collapse Covariance and Spectra 
        Eigen::MatrixXf collapsed_full_covariance =  CollapseMatrix(config,full_covariance);
        Eigen::MatrixXf sub_collapsed_full_covariance =  collapsed_full_covariance.block(startBin,startBin,nbin,nbin);
        Eigen::MatrixXf sub_collapsed_stat_covariance =  collapsed_stat_covariance.block(startBin,startBin,nbin,nbin);

        // Invert Collaped Matrix Matrix 
        inverted_collapsed_full_covariance = (sub_collapsed_full_covariance+sub_collapsed_stat_covariance).inverse();
    } else {
        Eigen::MatrixXf sub_collapsed_stat_covariance =  collapsed_stat_covariance.block(startBin,startBin,nbin,nbin);
        inverted_collapsed_full_covariance = (sub_collapsed_stat_covariance).inverse();
    }

    Eigen::VectorXf delta  = (CollapseMatrix(config, cv.Spec()) - (shape_only ? data.Normalize(config,cv) : data.Spec())).segment(startBin, nbin);
    //float pull = Pull(subvector2);
    float covar_portion = (delta.transpose())*inverted_collapsed_full_covariance*(delta);
    float value = covar_portion;//pull;

    return value;
}

void PROchi::print(const Eigen::VectorXf &param){


return;
}

