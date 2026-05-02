#include "PROpoisson.h"
#include "PROcess.h"
#include "PROdata.h"
#include "PROlog.h"
#include "PROtocall.h"
using namespace PROfit;


PROpoisson::PROpoisson(const std::string tag, const PROconfig &conin, const PROpeller &pin, const PROsyst *systin, const PROmodel &modelin, const PROdata &datain, EvalStrategy strat, bool shape_only, std::vector<float> physics_param_fixed) : PROmetric(), model_tag(tag), config(conin), peller(pin), syst(systin), model(modelin), data(datain), strat(strat), shape_only(shape_only), physics_param_fixed(physics_param_fixed), correlated_systematics(false) {
    last_value = 0.0; last_param = Eigen::VectorXf::Zero(model.nparams+syst->GetNSplines()); 
    fixed_index = -999;

    if(syst->GetNCovar()) {
        log<LOG_WARNING>(L"%1% || Warning: Using a systematics object with covariance systematics with"
                          " Poisson log likelihood ratio. This is not supported and the covariance"
                          " systematics will be ignored.") % __func__;
    }

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

float PROpoisson::Pull(const Eigen::VectorXf &systs) {
    // No correlations: sum of squares
    Eigen::VectorXf centered = systs - syst->spline_centers;
    if (!correlated_systematics) {
        return (centered.array().square() / syst->spline_priors.array().square()).sum();
    }

    return centered.dot(prior_covariance_inv * centered);
}

void PROpoisson::fixSpline(int fix, float valin){
    fixed_index=fix;
    fixed_val=valin;
    return;
}
float PROpoisson::operator()(const Eigen::VectorXf &param, Eigen::VectorXf &gradient){
    return PROpoisson::operator()(param, gradient, true);
}


float PROpoisson::operator()(const Eigen::VectorXf &param, Eigen::VectorXf &gradient, bool rungradient){
    size_t nparams = model.nparams+syst->GetNSplines();
    size_t nsyst = syst->GetNSplines();
    log<LOG_DEBUG>(L"%1% || nparams is %2%, nsyst is %3% ") % __func__ % nparams % nsyst;    

    // Get Spectra from FillSpectra
    Eigen::VectorXf subvector1 = param.segment(0, nparams - nsyst);
    if(model.model_constraint){
        if(!model.model_constraint(subvector1)){
            return 1e10;
        }
    }
    Eigen::VectorXf subvector2 = param.segment(nparams - nsyst, nsyst);
    
    PROspec result = FillSpectra(config, peller, *syst, model, param, fs_cache, strat == BinnedChi2);

    const Eigen::VectorXf vdata = shape_only
        ? data.Normalize(config,result)
        : data.Spec();
    const Eigen::VectorXf vmc = CollapseMatrix(config, result.Spec());
    float poisson = 2 * (vmc.array() - vdata.array() + vdata.array() * (vdata.array() / vmc.array()).log()).sum();
    float pull = Pull(subvector2);
    float value = poisson + pull;

    if(rungradient){
        float dval = 1e-4;
        for (size_t i = 0; i < nparams; i++) {
            //if(i == fixed_index) gradient(i) = 0;
            //Eigen::VectorXd tmpParams = last_param;
            Eigen::VectorXf tmpParams = param;
            int sgn = ((param(i) - last_param(i)) > 0) - ((param(i) - last_param(i)) < 0);
            if(!sgn) sgn = 1;
            //if(fitparams.size() != 0 && i == 1 && param(i) < -4 + dval) sgn = 1;
            //if(fitparams.size() != 0 && i == 1 && param(i) > 0 - dval) sgn = -1;
            tmpParams(i) = /*param(i) != last_param(i) ? param(i) :*/ param(i) + sgn * dval;
            
            Eigen::VectorXf subvector1 = tmpParams.segment(0, nparams - nsyst);
            Eigen::VectorXf subvector2 = tmpParams.segment(nparams - nsyst, nsyst);

            // If this new gradient evaluation point violates unitarity, set the gradient to a large value
            if(model.model_constraint){
                if(!model.model_constraint(subvector1)){
                    //log<LOG_ERROR>(L"%1% || WARNING In PROpoisson: Gradient evaluation point violates unitarity. Setting gradient to large value.") % __func__;
                    gradient(i) = sgn * 1e10;
                    continue;
                }
            }
            PROspec result = FillSpectra(config, peller, *syst, model, tmpParams, fs_cache, strat != EventByEvent);

            const Eigen::VectorXf vdata = shape_only 
                ? data.Normalize(config,result)
                : data.Spec();
            const Eigen::VectorXf vmc = CollapseMatrix(config, result.Spec());
            float poisson = 2 * (vmc.array() - vdata.array() + vdata.array() * (vdata.array() / vmc.array()).log()).sum();
            float pull = Pull(subvector2);
            float value_grad = poisson + pull;

            gradient(i) = (value_grad-value)/(tmpParams(i) - param(i));
        }
    }
    //std::cout<<"Grad: "<<gradient<<std::endl;

    //log<LOG_DEBUG>(L"%1% || value %2%, last_value %3%, pull") % __func__ % value  % last_value % pull;
    log<LOG_DEBUG>(L"%1% || FINISHED ITERATION got vals: %2% %3%") % __func__ % value % last_value ;

    //Update last param
    last_param = param;
    last_value = value;

    return value;
}

float PROpoisson::getSingleChannelChi(size_t global_channel_index, const PROspec &cv, size_t var_index) {

    size_t nbin = config.m_channel_variable_bins[global_channel_index][var_index].NBins();
    size_t startBin = config.GetCollapsedGlobalVariableBinStart(global_channel_index,var_index);


    //const Eigen::VectorXf &vdata = data.Spec().segment(startBin, nbin);
    const Eigen::VectorXf vdata = (shape_only 
        ? data.Normalize(config,cv)
        : data.Spec()).segment(startBin, nbin);
    const Eigen::VectorXf vmc = CollapseMatrix(config, cv.Spec()).segment(startBin, nbin);
    float poisson = 2 * (vmc.array() - vdata.array() + vdata.array() * (vdata.array() / vmc.array()).log()).sum();
    //float pull = Pull(subvector2);
    float value = poisson; //+ pull

    return value;
}

void PROpoisson::print([[maybe_unused]] const Eigen::VectorXf &param){
    return;
}
