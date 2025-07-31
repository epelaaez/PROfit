#include "PROCNP.h"
#include "PROcess.h"
#include "PROdata.h"
#include "PROlog.h"
#include "PROmetric.h"
#include "PROtocall.h"

#include <Eigen/Eigen>

using namespace PROfit;


PROCNP::PROCNP(const std::string tag, const PROconfig &conin, const PROpeller &pin, const PROsyst *systin, const PROmodel &modelin, const PROdata &datain, EvalStrategy strat, std::vector<float> physics_param_fixed) : PROmetric(), model_tag(tag), config(conin), peller(pin), syst(systin), model(modelin), data(datain), strat(strat), physics_param_fixed(physics_param_fixed), correlated_systematics(false) {
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
}

float PROCNP::Pull(const Eigen::VectorXf &systs) {
    // No correlations: sum of squares
    Eigen::VectorXf centered = systs - syst->spline_centers;
    if (!correlated_systematics) {
        return (centered.array().square() / syst->spline_priors.array().square()).sum();
    }
    return centered.dot(prior_covariance.inverse() * centered);
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

    //size_t nparams = model->nparams+syst->GetNSplines();
    //size_t nsyst = syst->GetNSplines();

    // Get Spectra from FillSpectra
    Eigen::VectorXf subvector1 = param.segment(0, model.nparams);
    //log<LOG_DEBUG>(L"%1% || Created physics subvector with size %2%") % __func__ % subvector1.size();
    Eigen::VectorXf subvector2 = param.segment(model.nparams, syst->GetNSplines());
    //log<LOG_DEBUG>(L"%1% || Created spline subvector with size %2%") % __func__ % subvector2.size();

    PROspec result = FillSpectra(config, peller, *syst, model, param, strat == BinnedChi2);


    Eigen::MatrixXf inverted_collapsed_full_covariance(config.m_num_variable_bins_total_collapsed[config.i_prime],config.m_num_variable_bins_total_collapsed[config.i_prime]);
    PROspec cv = FillSpectra(config, peller, *syst, model, subvector1, strat != EventByEvent);
    Eigen::MatrixXf collapsed_data_stat_covariance = data.Spec().array().matrix().asDiagonal();
    Eigen::MatrixXf mc_stat_covariance = cv.Spec().array().matrix().asDiagonal();
    Eigen::MatrixXf collapsed_mc_stat_covariance = CollapseMatrix(config, mc_stat_covariance);

    Eigen::MatrixXf collapsed_stat_covariance = 3 * (collapsed_data_stat_covariance.inverse() + 2 * collapsed_mc_stat_covariance.inverse()).inverse();
    for(int i = 0; i < collapsed_stat_covariance.cols(); ++i) {
        // If data bin is 0, this will be nan
        if(std::isnan(collapsed_stat_covariance(i,i)))
            collapsed_stat_covariance(i,i) = mc_stat_covariance(i,i)/2;
    }


    Eigen::MatrixXf diag = result.Spec().array().matrix().asDiagonal(); 
    Eigen::MatrixXf full_covariance = diag*(syst->fractional_covariance)*diag;

    Eigen::MatrixXf collapsed_full_covariance = CollapseMatrix(config, full_covariance); 
    inverted_collapsed_full_covariance = (collapsed_stat_covariance+ collapsed_full_covariance).inverse();

    // Calculate Chi^2  value
    Eigen::VectorXf delta  = CollapseMatrix(config,result.Spec()) - data.Spec(); 

    float pull = Pull(subvector2);
    float dmsq_penalty = 0;
    float covar_portion = (delta.transpose())*inverted_collapsed_full_covariance*(delta);
    float value = covar_portion + dmsq_penalty + pull;

    if(std::isnan(value)) {
        log<LOG_WARNING>(L"%1% || WARNING: CNP chi2 is NaN. This is very bad.\n"
                L"covar_portion: %2%\npull: %3%\ndelta: %4%\n"
                L"mc spec: %5%\ndata spec: %6%")
            % __func__ % covar_portion % pull % delta % CollapseMatrix(config, result.Spec())
            % data.Spec();
        abort();
    }


    if(rungradient){
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
            PROspec result = FillSpectra(config, peller, *syst, model, tmpParams, strat != EventByEvent);
            // Calcuate Full Covariance matrix
            Eigen::MatrixXf inverted_collapsed_full_covariance(config.m_num_variable_bins_total_collapsed[config.i_prime],config.m_num_variable_bins_total_collapsed[config.i_prime]);

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

            Eigen::MatrixXf diag = result.Spec().array().matrix().asDiagonal(); 
            Eigen::MatrixXf full_covariance = diag*(syst->fractional_covariance)*diag;

            Eigen::MatrixXf collapsed_full_covariance = CollapseMatrix(config, full_covariance); 
            inverted_collapsed_full_covariance = (collapsed_stat_covariance+ collapsed_full_covariance).inverse();

            // Calculate Chi^2  value
            Eigen::VectorXf delta  = CollapseMatrix(config,result.Spec()) - data.Spec(); 

            float pull = Pull(subvector2);
            float value_grad = (delta.transpose())*inverted_collapsed_full_covariance*(delta) + pull;

            gradient(i) = (value_grad-value)/(tmpParams(i) - param(i));
        }
    }
    //std::cout<<"Grad: "<<gradient<<std::endl;

    //log<LOG_DEBUG>(L"%1% || value %2%, last_value %3%, pull") % __func__ % value  % last_value % pull;
    //log<LOG_DEBUG>(L"%1% || FINISHED ITERATION got vals: %2% %3%") % __func__ % value % last_value ;

    //Update last param
    last_param = param;
    last_value = value;

    return value;
}

float PROCNP::getSingleChannelChi(size_t global_channel_index,size_t var_index) {
    PROspec cv = FillCVSpectra(config, peller,strat == BinnedChi2);

    size_t nbin =  config.m_channel_variable_num_bins[config.GetLocalChannelIndexFromGlobalChannelIndex(global_channel_index)][var_index];
    size_t startBin = config.GetCollapsedGlobalVariableBinStart(global_channel_index,var_index);


    Eigen::MatrixXf inverted_collapsed_full_covariance(nbin,nbin);


    Eigen::MatrixXf collapsed_data_stat_covariance = (data.Spec().array().matrix().asDiagonal());
    collapsed_data_stat_covariance = collapsed_data_stat_covariance.block(startBin,startBin,nbin,nbin);
    Eigen::MatrixXf mc_stat_covariance = cv.Spec().array().matrix().asDiagonal();
    Eigen::MatrixXf collapsed_mc_stat_covariance = CollapseMatrix(config, mc_stat_covariance).block(startBin,startBin,nbin,nbin);
    Eigen::MatrixXf sub_collapsed_stat_covariance = 3 * (collapsed_data_stat_covariance.inverse() + 2 * collapsed_mc_stat_covariance.inverse()).inverse();

    //only calculate a syst covariance if we have any covariance parameters as defined in the xml
    if(syst->GetNCovar()){
        // Calculate Full Syst Covariance matrix
        Eigen::MatrixXf diag =  cv.Spec().array().matrix().asDiagonal(); 
        Eigen::MatrixXf full_covariance =  diag*(syst->fractional_covariance)*diag;

        // Collapse Covariance and Spectra 
        Eigen::MatrixXf collapsed_full_covariance =  CollapseMatrix(config,full_covariance);
        Eigen::MatrixXf sub_collapsed_full_covariance =  collapsed_full_covariance.block(startBin,startBin,nbin,nbin);

        // Invert Collaped Matrix Matrix 
        inverted_collapsed_full_covariance = (sub_collapsed_full_covariance+sub_collapsed_stat_covariance).inverse();
    } else {
        inverted_collapsed_full_covariance = (sub_collapsed_stat_covariance).inverse();
    }

    Eigen::VectorXf delta  = (CollapseMatrix(config, cv.Spec()) - data.Spec()).segment(startBin,nbin);
    //float pull = Pull(subvector2);
    float covar_portion = (delta.transpose())*inverted_collapsed_full_covariance*(delta);
    float value = covar_portion;//pull;

    return value;
}

void PROCNP::print(const Eigen::VectorXf &param){

     log<LOG_INFO>(L"%1% || input param: %2% ") % __func__ % param;



    // Get Spectra from FillSpectra
    Eigen::VectorXf subvector1 = param.segment(0, model.nparams);
    Eigen::VectorXf subvector2 = param.segment(model.nparams, syst->GetNSplines());

    PROspec result = FillSpectra(config, peller, *syst, model, param, strat == BinnedChi2);
    log<LOG_INFO>(L"%1% || Result Spectra: ") % __func__ ;
    result.Print();

    Eigen::MatrixXf inverted_collapsed_full_covariance(config.m_num_variable_bins_total_collapsed[config.i_prime],config.m_num_variable_bins_total_collapsed[config.i_prime]);
    PROspec cv = FillSpectra(config, peller, *syst, model, subvector1, strat != EventByEvent);
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
    Eigen::MatrixXf diag = result.Spec().array().matrix().asDiagonal(); 
    Eigen::MatrixXf full_covariance = diag*(syst->fractional_covariance)*diag;

    Eigen::MatrixXf collapsed_full_covariance = CollapseMatrix(config, full_covariance); 
    inverted_collapsed_full_covariance = (collapsed_stat_covariance+ collapsed_full_covariance).inverse();

    // Calculate Chi^2  value
    Eigen::VectorXf delta  = CollapseMatrix(config,result.Spec()) - data.Spec(); 
    log<LOG_INFO>(L"%1% || DataSpectra Spectra: ") % __func__ ;
    data.Print();

    float pull = Pull(subvector2);
    float dmsq_penalty = 0;
    float covar_portion = (delta.transpose())*inverted_collapsed_full_covariance*(delta);
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
        PROspec result = FillSpectra(config, peller, *syst, model, tmpParams, strat != EventByEvent);
        // Calcuate Full Covariance matrix
        Eigen::MatrixXf inverted_collapsed_full_covariance(config.m_num_variable_bins_total_collapsed[config.i_prime],config.m_num_variable_bins_total_collapsed[config.i_prime]);

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

        Eigen::MatrixXf diag = result.Spec().array().matrix().asDiagonal(); 
        Eigen::MatrixXf full_covariance = diag*(syst->fractional_covariance)*diag;

        Eigen::MatrixXf collapsed_full_covariance = CollapseMatrix(config, full_covariance); 
        inverted_collapsed_full_covariance = (collapsed_stat_covariance+ collapsed_full_covariance).inverse();

        // Calculate Chi^2  value
        Eigen::VectorXf delta  = CollapseMatrix(config,result.Spec()) - data.Spec(); 

        float pull = Pull(subvector2);
        float value_grad = (delta.transpose())*inverted_collapsed_full_covariance*(delta) + pull;

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

