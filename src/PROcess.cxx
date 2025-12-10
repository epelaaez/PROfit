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
            //log<LOG_INFO>(L"%1% || Starting systw calculation %2%") % __func__ % var_index;
            //auto start_systw = std::chrono::high_resolution_clock::now();

            const size_t nbins_var = inconfig.m_num_variable_bins_total[var_index];
            Eigen::VectorXf systw = Eigen::VectorXf::Constant(nbins_var, 1);
            
            for(int i = 0; i < shifts.size(); ++i) {
                size_t binning = insyst.spline_binnings[i];

                if(binning == var_index) {
                    // Case 1: Same binning - direct multiplication
                    for(size_t k = 0; k < nbins_var; ++k) {
                        systw(k) *= insyst.GetSplineShift(i, shifts(i), k);
                    }
                } else {
                    // Case 2: Different binning - use matrix-vector multiplication
                    const size_t nbins_binning = inconfig.m_num_variable_bins_total[binning];
                    
                    // Get all spline shifts for this systematic
                    Eigen::VectorXf spline_shifts(nbins_binning);
                    for(size_t j = 0; j < nbins_binning; ++j) {
                        spline_shifts(j) = insyst.GetSplineShift(i, shifts(i), j);
                    }
                    
                    // Get the histogram matrix
                    const auto& hist = inprop.variable_hist_storage(binning, var_index);
                    
                    // Compute weighted and unweighted sums using matrix operations
                    // weighted_sum[k] = sum_j(spline_shifts[j] * hist(j, k))
                    // unweighted_sum[k] = sum_j(hist(j, k))
                    Eigen::VectorXf weighted_sum = hist.transpose() * spline_shifts;
                    Eigen::VectorXf unweighted_sum = hist.colwise().sum().transpose();
                    
                    // Apply the ratio where unweighted > 0
                    for(size_t k = 0; k < nbins_var; ++k) {
                        if(unweighted_sum(k) > 0) {
                            systw(k) *= weighted_sum(k) / unweighted_sum(k);
                        }
                    }
                }
            }

            //auto end_systw = std::chrono::high_resolution_clock::now();
            //std::chrono::duration<double> duration_systw = end_systw - start_systw;
            //log<LOG_INFO>(L"%1% || systw calculation took %2% ms") % __func__ % (duration_systw.count() * 1000.);

            //log<LOG_INFO>(L"%1% || Starting le_arr building %2%") % __func__ % var_index;
            //auto start_le = std::chrono::high_resolution_clock::now();
            
            std::vector<float> le_arr;
            for(long int i = 0; i < inconfig.m_num_variable_bins_total[inmodel.ivar]; ++i) {
                le_arr.push_back(inprop.variable_midbin[inmodel.ivar][i]);
            }

            //auto end_le = std::chrono::high_resolution_clock::now();
            //std::chrono::duration<double> duration_le = end_le - start_le;
            //log<LOG_INFO>(L"%1% || le_arr building took %2% seconds") % __func__ % duration_le.count();

            //log<LOG_INFO>(L"%1% || Starting prob calculation %2%") % __func__ % var_index;
            //auto start = std::chrono::high_resolution_clock::now();

            auto probs = inmodel.get_probs(phys, le_arr);

            //auto end = std::chrono::high_resolution_clock::now();
            //std::chrono::duration<double> duration = end - start;
            //log<LOG_INFO>(L"%1% || L/E loop took %2% seconds") % __func__ % duration.count();

            //log<LOG_INFO>(L"%1% || Starting null matrix check  %2%") % __func__ % var_index;
            //auto start2 = std::chrono::high_resolution_clock::now();

            for(size_t j = 0; j < inmodel.prob_types.size(); ++j) {
                if (inmodel.hists[var_index][j].data() == nullptr) {
                    log<LOG_ERROR>(L"Null matrix at var_index=%1%, j=%2%") % var_index % j;
                    continue;
                }
            }

            //auto end2 = std::chrono::high_resolution_clock::now();
            //std::chrono::duration<double> duration2 = end2 - start2;
            //log<LOG_INFO>(L"%1% || null matrix check took %2% seconds") % __func__ % duration2.count();

            //log<LOG_INFO>(L"%1% || Starting filling %2%") % __func__ % var_index;
            //auto start3 = std::chrono::high_resolution_clock::now();

            // Use matrix-vector multiplication instead of triple nested loop
            // Original: spec[k] = Σᵢⱼ systw[k] * probs(i,j) * hists[j](i,k)
            // Rewritten: spec = systw ⊙ Σⱼ (hists[j]ᵀ × probs.col(j))
            Eigen::VectorXf result = Eigen::VectorXf::Zero(myspectrum.GetNbins());
            
            for(size_t j = 0; j < inmodel.prob_types.size(); ++j) {
                // hists[var_index][j] has shape (le_arr.size(), nbins)
                // probs.col(j) has shape (le_arr.size(),)
                // Result: (nbins, le_arr.size()) × (le_arr.size(),) = (nbins,)
                result.noalias() += inmodel.hists[var_index][j].transpose() * probs.col(j);
            }
            
            // Apply systematic weights and create spectrum
            Eigen::VectorXf final_spec = systw.cwiseProduct(result);
            Eigen::VectorXf final_error = final_spec.array().abs().sqrt();
            myspectrum = PROspec(final_spec, final_error);

            //auto end3 = std::chrono::high_resolution_clock::now();
            //std::chrono::duration<double> duration3 = end3 - start3;
            //log<LOG_INFO>(L"%1% || filling took %2% seconds") % __func__ % duration3.count();

        } else {

            std::vector<float> le_arr;
            for(size_t i = 0; i < inprop.NEvent(); ++i) {
                le_arr.push_back(inprop.VariableValue(inmodel.ivar, i));
            }

            auto probs = inmodel.get_probs(phys, le_arr);

            for(size_t i = 0; i < inprop.NEvent(); ++i) {
                float oscw = probs(i, inprop.model_rule[i]);
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
            std::vector<float> hist_w_arr;
            std::vector<float> le_arr;
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

                hist_w_arr.push_back(hist_w);
                le_arr.push_back(le);
            }

            auto probs = inmodel.get_probs(phys, le_arr);

            // Use matrix-vector multiplication instead of triple nested loop
            // Convert hist_w_arr to Eigen vector for element-wise operations
            Eigen::Map<const Eigen::VectorXf> hist_w_vec(hist_w_arr.data(), hist_w_arr.size());
            
            Eigen::VectorXf result = Eigen::VectorXf::Zero(myspectrum.GetNbins());
            
            for(size_t j = 0; j < inmodel.prob_types.size(); ++j) {
                // Weight probs by hist_w, then do matrix-vector multiply
                Eigen::VectorXf weighted_probs = hist_w_vec.cwiseProduct(probs.col(j));
                result.noalias() += inmodel.hists[inconfig.i_prime][j].transpose() * weighted_probs;
            }
            
            Eigen::VectorXf final_error = result.array().abs().sqrt();
            myspectrum = PROspec(result, final_error);

        } else {
            std::vector<float> le_arr;
            std::vector<float> hist_w_arr;
            std::vector<float> add_w_arr;
            for(size_t i = 0; i < inprop.NEvent(); ++i) {
                le_arr.push_back(inprop.VariableValue(inmodel.ivar, i));
                add_w_arr.push_back(inprop.added_weights[i]);

                float hist_w = 1.0;

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

                hist_w_arr.push_back(hist_w);
            }

            auto probs = inmodel.get_probs(phys, le_arr);

            for(size_t i = 0; i < inprop.NEvent(); ++i) {
                float oscw = phys.size() != 0 ? probs(i, inprop.model_rule[i]) : 1.0f;

                float finalw = oscw * add_w_arr[i] * hist_w_arr[i];
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
