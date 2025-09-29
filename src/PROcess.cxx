#include "PROcess.h"
#include "PROlog.h"
#include "PROspec.h"
#include "PROsyst.h"
#include "PROtocall.h"
#include "TH2D.h"

#include <Eigen/Eigen>

#include <algorithm>
#include <random>


namespace PROfit {

    PROspec FillSpectra(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst, const PROmodel &inmodel, const std::map<std::string, float> &inparams, bool binned, size_t var_index) {
        // default parameters
        Eigen::VectorXf params = Eigen::VectorXf::Zero(inmodel.nparams + insyst.GetNSplines());

        // default pulls are all 0. Set the default model parameters
        for (size_t ind = 0; ind < inmodel.nparams; ind++) params[ind] = inmodel.default_val[ind];

        // set parameters configured by user
        for (auto const &pair: inparams) {
            auto it1 = std::find(insyst.spline_names.begin(), insyst.spline_names.end(), pair.first);
            auto it2 = std::find(inmodel.param_names.begin(), inmodel.param_names.end(), pair.first);
            if (it1 != insyst.spline_names.end()) {
                int ind = std::distance(insyst.spline_names.begin(), it1);
                params[ind + inmodel.nparams] = pair.second;
            }
            else if (it2 != inmodel.param_names.end()) {
                int ind = std::distance(inmodel.param_names.begin(), it2);
                params[ind] = pair.second;
            }
            else {
                log<LOG_WARNING>(L"%1% | unable to find parameters %2% . Skipping.") % __func__ % pair.first.c_str();
            }
        }


        return FillSpectra(inconfig, inprop, insyst, inmodel, params, binned, var_index);
    }


    PROspec FillSpectra(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst, const PROmodel &inmodel, const Eigen::VectorXf &params, bool binned, size_t var_index){
        PROspec myspectrum(inconfig.m_num_variable_bins_total[var_index]);
        Eigen::VectorXf phys   = params.segment(0, inmodel.nparams);
        Eigen::VectorXf shifts = params.segment(inmodel.nparams, params.size() - inmodel.nparams);

        if(binned) {
            Eigen::VectorXf systw = Eigen::VectorXf::Constant(inconfig.m_num_variable_bins_total[var_index], 1);
            for(int i = 0; i < shifts.size(); ++i) {
                size_t binning = insyst.spline_binnings[i];


                //const Eigen::MatrixXf &hist = inprop.variable_hist_storage(binning,var_index);
                //direct access with inprop.variable_hist_storage(binning,var_index,j,k)
                for(size_t k = 0; k < inconfig.m_num_variable_bins_total[var_index]; ++k) {
                    if(binning == var_index){
                        systw(k) *= insyst.GetSplineShift(i, shifts(i), k);
                    }
                    else {
                        float val = 0, unweighted = 0;
                        for(long int j = 0; j < inconfig.m_num_variable_bins_total[binning]; ++j) {
                            float binsystw = insyst.GetSplineShift(i, shifts(i), j);
                            val += binsystw * inprop.variable_hist_storage(binning,var_index,j, k);
                            unweighted += inprop.variable_hist_storage(binning,var_index,j, k);
                        }
                        if(unweighted > 0) systw(k) *= val/unweighted;
                    }
                }
            }

            for(long int i = 0; i < inconfig.m_num_variable_bins_total[inmodel.ivar]; ++i) {
                float le = inprop.variable_midbin[inmodel.ivar][i];

                for(size_t j = 0; j < inmodel.model_functions.size(); ++j) {
                    float oscw = inmodel.model_functions[j](phys, le);
                    if (inmodel.hists[var_index][j].data() == nullptr) {
                        log<LOG_ERROR>(L"Null matrix at var_index=%1%, j=%2%") % var_index % j;
                        continue;
                    }

                    for(size_t k = 0; k < myspectrum.GetNbins(); ++k) {
                        myspectrum.Fill(k, systw(k) * oscw * inmodel.hists[var_index][j](i, k));
                    }
                }
            }
        } else {
            for(size_t i = 0; i<inprop.NEvent(); ++i){
                float oscw  =  inmodel.model_functions[inprop.model_rule[i]](phys, inprop.VariableValue(inmodel.ivar, i));
                float add_w = inprop.added_weights[i]; 
                const int reco_bin = inprop.VariableBinIndex(var_index, i);

                float systw = 1;
                for(int j = 0; j < shifts.size(); ++j) {
                    int binning = insyst.spline_binnings[j];
                    const int spline_bin = inprop.VariableBinIndex(binning, i);
                    systw *= insyst.GetSplineShift(j, shifts[j], spline_bin);
                }
                float finalw = oscw * systw * add_w;
                myspectrum.Fill(reco_bin, finalw);

            }
        }
        return myspectrum;
    }




    //*********************************************************************************************

    PROspec FillWeightedSpectrumFromHist(const PROconfig &inconfig, const PROpeller &inprop, std::vector<TH2D*> inweighthists, const PROmodel &inmodel, const Eigen::VectorXf &params, bool binned){
        PROspec myspectrum(inconfig.m_num_variable_bins_total[inconfig.i_prime]);
        Eigen::VectorXf phys   = params.segment(0, inmodel.nparams);
        Eigen::VectorXf shifts = params.segment(inmodel.nparams, params.size() - inmodel.nparams);

        int i_osc_tmp =1;
        if (binned) {
            for(long int i = 0; i < inprop.variable_hist_storage(i_osc_tmp,inconfig.i_prime).rows(); ++i) {
                float le = inprop.variable_midbin[i_osc_tmp][i];
                float hist_w = 1.0 ;

                //Figure out what subchannel the event is in
                size_t subchan = inconfig.GetSubchannelIndexFromVariableGlobalBin(inprop.VariableBinIndex(i_osc_tmp, i), i_osc_tmp);
                std::string name = inconfig.m_fullnames[subchan];

                //Put name for ICARUS study here. How to handle more generically?
                if (name == "nu_ICARUS_numu_numucc") {
                    int ipmom = 2;
                    int ipcosth = 3;

                    float pmom = 2;//static_cast<float>(inprop.pmom[i]);
                    float pcosth = 3;// static_cast<float>(inprop.pcosth[i]);
                    for (size_t j = 0; j<inweighthists.size(); ++j){
                        TH2D h = *inweighthists[j];
                        int bin = h.FindBin(pmom,pcosth);
                        hist_w *= h.GetBinContent(bin);
                    }
                }

                for(size_t j = 0; j < inmodel.model_functions.size(); ++j) {
                    float oscw = inmodel.model_functions[j](phys, le);
                    for(size_t k = 0; k < myspectrum.GetNbins(); ++k) {
                        myspectrum.Fill(k, hist_w * oscw * inmodel.hists[inconfig.i_prime][j](i, k));
                    }
                }
            }
        } else {
            for(size_t i = 0; i<inprop.NEvent(); ++i){

                float oscw  = phys.size() != 0 ? 
                    inmodel.model_functions[inprop.model_rule[i]](phys, inprop.VariableValue(inmodel.ivar, i)) :
                    1;	
                float add_w = inprop.added_weights[i];
                float hist_w = 1.0 ;

                //Figure out what subchannel the event is in
                size_t subchan = inconfig.GetSubchannelIndexFromVariableGlobalBin(inprop.VariableBinIndex(i_osc_tmp, i), i_osc_tmp);
                std::string name = inconfig.m_fullnames[subchan];

                //Put name for ICARUS study here. How to handle more generically?
                if (name == "nu_ICARUS_numu_numucc") {
                    float pmom = 2;//static_cast<float>(inprop.pmom[i]);
                    float pcosth = 2;//static_cast<float>(inprop.pcosth[i]);

                    for (size_t j = 0; j<inweighthists.size(); ++j){
                        TH2D h = *inweighthists[j];
                        int bin = h.FindBin(pmom,pcosth);
                        hist_w *= h.GetBinContent(bin);
                    }
                }

                float finalw = oscw * add_w * hist_w;
                myspectrum.Fill(inprop.VariableBinIndex(inconfig.i_prime, i), finalw);
            }
        }
        return myspectrum;
    }

    PROspec FillSystRandomThrow(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst, const PROmodel &model, const PROspec &cvspec, const Eigen::VectorXf &cvparams, uint32_t seed, int var_index) {
        int nbins = inconfig.m_num_variable_bins_total[var_index],
        nbins_collapsed = inconfig.m_num_variable_bins_total_collapsed[var_index];
        Eigen::VectorXf spec = Eigen::VectorXf::Constant(nbins, 0);

        Eigen::VectorXf params = cvparams;

        // TODO: We should think about centralizing rng in a thread-safe/thread-aware way
        static std::mt19937 rng{seed};
        std::vector<std::normal_distribution<float>> d_spline;
        for(size_t i = 0; i < insyst.GetNSplines(); ++i)
            d_spline.emplace_back(insyst.spline_centers(i), insyst.spline_priors(i));
        std::normal_distribution<float> d_cov;
        std::vector<float> throws;
        //Eigen::VectorXf throwC = Eigen::VectorXf::Constant(inconfig.m_num_variable_bins_total[inconfig.i_prime], 0);
        Eigen::VectorXf throwC = Eigen::VectorXf::Constant(nbins_collapsed, 0);
        for(size_t i = 0; i < insyst.GetNSplines(); i++) {
            throws.push_back(d_spline[i](rng));
        }
        for(int i = 0; i < nbins_collapsed; i++)
            throwC(i) = d_cov(rng);

        //get actual splines 
        for(size_t i=0;i<throws.size();i++){
            params(i+model.nparams) = throws.at(i);
        }

        bool binned = true;//dont want to faf around with event by event here lets be honst
        if (binned){
          spec = FillSpectra(inconfig, inprop, insyst, model, params, binned, var_index).Spec();

        }else{//currently never run
            for(size_t i = 0; i<inprop.NEvent(); ++i){
                float add_w = inprop.added_weights[i]; 
                float systw = 1;
                for(size_t j = 0; j < throws.size(); ++j) {
                    int binning = insyst.spline_binnings[j];
                    const int spline_bin = inprop.VariableBinIndex(binning, i);
                    systw *= insyst.GetSplineShift(j, throws[j], spline_bin);
                }
                if(inprop.VariableBinIndex(var_index, i) >= 0) {
                    float finalw = systw * add_w;
                    spec(inprop.VariableBinIndex(var_index, i)) += finalw;
                }
            }
        }

        if(insyst.GetNCovar() == 0) {
            Eigen::VectorXf final_spec = CollapseMatrix(inconfig, spec, var_index);
            return PROspec(final_spec, final_spec.array().sqrt());
        }

        Eigen::MatrixXf decomp_cov = insyst.DecomposeFractionalCovariance(inconfig, cvspec.Spec());
        Eigen::VectorXf collapsed_spec = CollapseMatrix(inconfig, spec, var_index);
        Eigen::VectorXf final_spec = collapsed_spec + decomp_cov * throwC;

        //std::vector<float> stdVec(final_spec.data(), final_spec.data() + final_spec.size());
        //log<LOG_INFO>(L"%1% | final_spec is %2% ") % __func__ % stdVec;


        return PROspec(final_spec, final_spec.array().sqrt());
    }

    PROspec FillSplineRandomThrow(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst,  const PROmodel &model,  const Eigen::VectorXf &cvparams, int spline, uint32_t seed, int other_index) {
        int nbins =  inconfig.m_num_variable_bins_total[other_index];
        Eigen::VectorXf spec = Eigen::VectorXf::Constant(nbins, 0);

        // TODO: We should think about centralizing rng in a thread-safe/thread-aware way
        static std::mt19937 rng{seed};
        std::normal_distribution<float> d(insyst.spline_centers(spline), insyst.spline_priors(spline));
        float spline_throw = d(rng);
        int binning = insyst.spline_binnings[spline];
        Eigen::VectorXf params = cvparams;
        params(spline+model.nparams) = spline_throw;

        bool binned = true;//dont want to faf around with event by event here lets be honst
        if (binned){
            spec = FillSpectra(inconfig, inprop, insyst, model, params, binned, other_index).Spec();
        } 

        return PROspec(spec, spec.array().sqrt());
    }
};
