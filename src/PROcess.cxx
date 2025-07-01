#include "PROcess.h"
#include "PROlog.h"
#include "PROspec.h"
#include "PROsyst.h"
#include "PROtocall.h"
#include "TH2D.h"

#include <Eigen/Eigen>

#include <algorithm>
#include <random>
#include <vector>

namespace PROfit {
    PROspec FillCVSpectra(const PROconfig &inconfig, const PROpeller &inprop, bool binned, size_t var_index){
        PROspec myspectrum(inconfig.m_num_variable_bins_total[var_index]);
//        log<LOG_ERROR>(L"%1% , %2% || GARP ") % __func__ % __LINE__  ;
        if(binned) {
            for(int i = 0; i < inprop.variable_hist_storage(inconfig.i_osc, var_index).rows(); ++i) {
                for(size_t k = 0; k < myspectrum.GetNbins(); ++k) {
                   if(myspectrum.GetNbins()!=inprop.variable_hist_storage(inconfig.i_osc,var_index).cols()){
                        log<LOG_ERROR>(L"%1% , %2% || fail: %3% is not (rows,cols) : (%4%,%5%). should also be %6% ") % __func__ % __LINE__ % myspectrum.GetNbins() % inprop.variable_hist_storage(inconfig.i_osc,var_index).rows() %inprop.variable_hist_storage(inconfig.i_osc,var_index).cols() % inconfig.m_num_variable_bins_total[var_index];
                        exit(EXIT_FAILURE);
                   }
             //      log<LOG_ERROR>(L"%1% , %2% || GARP i %3% k %4% bounds %5%,%6% ") % __func__ % __LINE__ % i % k % inprop.variable_hist_storage(inconfig.i_osc,var_index).rows() % inprop.variable_hist_storage(inconfig.i_osc,var_index).cols();
                    myspectrum.Fill(k, inprop.variable_hist_storage(inconfig.i_osc,var_index)(i, k));
                }
            }
        } else {
            for(size_t i = 0; i<inprop.variable_bin_indices.size(); ++i){
                float add_w = inprop.added_weights[i]; 
                myspectrum.Fill(inprop.variable_bin_indices[i][var_index], add_w);
            }
        }
        return myspectrum;
    }


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
                int binning = insyst.spline_binnings[i];
                const Eigen::MatrixXf &hist = inprop.variable_hist_storage(binning,var_index);
                for(size_t k = 0; k < inconfig.m_num_variable_bins_total[var_index]; ++k) {
                    if(binning == var_index){
                        //log<LOG_ERROR>(L"%1% , %2% || GARP binning %3% %4% %5%") % __func__ % __LINE__  % i % shifts(i) % k;
                        systw(k) *= insyst.GetSplineShift(i, shifts(i), k);
                    }
                    else {
                        float val = 0, unweighted = 0;
                        for(long int j = 0; j < hist.rows(); ++j) {
                      //log<LOG_ERROR>(L"%1%, %2% || GARP hist stor size (%3%,%4%) and were at (%5%,%6%) of var index %7% binning %8% (which should be (%9%,%10%) size)") % __func__ % __LINE__ % hist.rows() % hist.cols() %j % k % var_index % binning % inconfig.m_num_variable_bins_total[var_index] % inconfig.m_num_variable_bins_total[binning];
                            float binsystw = insyst.GetSplineShift(i, shifts(i), j);
                            val += binsystw * hist(j, k);
                            unweighted += hist(j,k);
                        }
                        if(unweighted > 0) systw(k) *= val/unweighted;
                    }
                }
            }
            //log<LOG_ERROR>(L"%1%, %2% || GARP hist stor size (%3%,%4%)") % __func__ % __LINE__ % inprop.variable_hist_storage(inconfig.i_osc, var_index).rows() % inprop.variable_hist_storage(inconfig.i_osc, var_index).cols();
            for(long int i = 0; i < inprop.variable_hist_storage(inconfig.i_osc, var_index).rows(); ++i) {
                float le = inprop.histLE[i];
                for(size_t j = 0; j < inmodel.model_functions.size(); ++j) {
                    float oscw = inmodel.model_functions[j](phys, le);
        if (inmodel.hists[var_index][j].data() == nullptr) {
            log<LOG_ERROR>(L"Null matrix at var_index=%1%, j=%2%") % var_index % j;
                continue;
    }

                    for(size_t k = 0; k < myspectrum.GetNbins(); ++k) {
                        //log<LOG_ERROR>(L"%1% , %2% || GARP Spec: i(bin):%3% j(model fun):%4% k(specbin):%5% . size for (i,k): hists(%6%,%7%) ") % __func__ % __LINE__  % i % j % k % inmodel.hists[var_index][j].rows() % inmodel.hists[var_index][j].cols() ;
myspectrum.Fill(k, systw(k) * oscw * inmodel.hists[var_index][j](i, k));
                    }
                }
            }
        } else {
            for(size_t i = 0; i<inprop.trueLE.size(); ++i){
                float oscw  =  inmodel.model_functions[inprop.model_rule[i]](phys, inprop.trueLE[i]);
                float add_w = inprop.added_weights[i]; 
                const int reco_bin = inprop.variable_bin_indices[i][var_index];

                float systw = 1;
                for(int j = 0; j < shifts.size(); ++j) {
                    int binning = insyst.spline_binnings[j];
                    const int spline_bin = inprop.variable_bin_indices[i][binning];
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
            for(long int i = 0; i < inprop.hist.rows(); ++i) {
                float le = inprop.histLE[i];
                float hist_w = 1.0 ;

                //Figure out what subchannel the event is in
                size_t subchan = inconfig.GetSubchannelIndexFromVariableGlobalBin(inprop.variable_bin_indices[i][i_osc_tmp],i_osc_tmp);
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
        }
        else {
            for(size_t i = 0; i<inprop.trueLE.size(); ++i){

                float oscw  = phys.size() != 0 ? 
                    inmodel.model_functions[inprop.model_rule[i]](phys, inprop.trueLE[i]) :
                    1;	
                float add_w = inprop.added_weights[i];
                float hist_w = 1.0 ;

                //Figure out what subchannel the event is in
                size_t subchan = inconfig.GetSubchannelIndexFromVariableGlobalBin(inprop.variable_bin_indices[i][i_osc_tmp],i_osc_tmp);
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
                myspectrum.Fill(inprop.variable_bin_indices[i][inconfig.i_prime], finalw);
            }
        }
        return myspectrum;
    }

    PROspec FillSystRandomThrow(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst, uint32_t seed, int other_index) {
        int nbins = inconfig.m_num_variable_bins_total[other_index],
            nbins_collapsed = inconfig.m_num_variable_bins_total_collapsed[other_index];
        Eigen::VectorXf spec = Eigen::VectorXf::Constant(nbins, 0);
        Eigen::VectorXf cvspec = Eigen::VectorXf::Constant(nbins, 0);


        // TODO: We should think about centralizing rng in a thread-safe/thread-aware way
        static std::mt19937 rng{seed};
        std::normal_distribution<float> d;
        std::vector<float> throws;
        //Eigen::VectorXf throwC = Eigen::VectorXf::Constant(inconfig.m_num_variable_bins_total[inconfig.i_prime], 0);
        Eigen::VectorXf throwC = Eigen::VectorXf::Constant(nbins_collapsed, 0);
        for(size_t i = 0; i < insyst.GetNSplines(); i++)
            throws.push_back(d(rng));
        for(int i = 0; i < nbins_collapsed; i++)
            throwC(i) = d(rng);


        if(other_index < 0) {
            Eigen::VectorXf systw = Eigen::VectorXf::Constant(nbins, 1);
            for(size_t i = 0; i < throws.size(); ++i) {
                int binning = insyst.spline_binnings[i];
                const Eigen::MatrixXf &hist =  inprop.variable_hists[binning];
                for(int k = 0; k < nbins; ++k) {
                    if(binning == -1) systw(k) *= insyst.GetSplineShift(i, throws[i], k);
                    else {
                        float val = 0, unweighted = 0;
                        for(long int j = 0; j < hist.rows(); ++j) {
                            float binsystw = insyst.GetSplineShift(i, throws[i], j);
                            val += binsystw * hist(j, k);
                            unweighted += hist(j,k);
                        }
                        if(unweighted > 0) systw(k) *= val/unweighted;
                    }
                }
            }
            for(long int i = 0; i < inprop.hist.rows(); ++i) {
                for(int k = 0; k < nbins; ++k) {
                    spec(k) += systw(k) * inprop.hist(i, k);
                    cvspec(k) += inprop.hist(i, k);
                }
            }
        } else {
            for(size_t i = 0; i<inprop.trueLE.size(); ++i){
                float add_w = inprop.added_weights[i]; 
                float systw = 1;
                for(size_t j = 0; j < throws.size(); ++j) {
                    int binning = insyst.spline_binnings[j];
                    const int spline_bin =  inprop.variable_bin_indices[i][binning];
                    systw *= insyst.GetSplineShift(j, throws[j], spline_bin);
                }
                if(inprop.variable_bin_indices[i][other_index] >= 0) {
                    float finalw = systw * add_w;
                    spec(inprop.variable_bin_indices[i][other_index]) += finalw;
                    cvspec(inprop.variable_bin_indices[i][other_index]) += add_w;
                }
            }
        }

        if(insyst.GetNCovar() == 0) {
            Eigen::VectorXf final_spec = CollapseMatrix(inconfig, spec, other_index);
            return PROspec(final_spec, final_spec.array().sqrt());
        }

        Eigen::MatrixXf decomp_cov = insyst.DecomposeFractionalCovariance(inconfig, cvspec);
        Eigen::VectorXf collapsed_spec = CollapseMatrix(inconfig, spec, other_index);
        Eigen::VectorXf final_spec = collapsed_spec + decomp_cov * throwC;

        //std::vector<float> stdVec(final_spec.data(), final_spec.data() + final_spec.size());
        //log<LOG_INFO>(L"%1% | final_spec is %2% ") % __func__ % stdVec;


        return PROspec(final_spec, final_spec.array().sqrt());
    }

    PROspec FillSplineRandomThrow(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst, int spline, uint32_t seed, int other_index) {
        int nbins =  inconfig.m_num_variable_bins_total[other_index];
        Eigen::VectorXf spec = Eigen::VectorXf::Constant(nbins, 0);

        // TODO: We should think about centralizing rng in a thread-safe/thread-aware way
        static std::mt19937 rng{seed};
        std::normal_distribution<float> d;
        float spline_throw = d(rng);
        int binning = insyst.spline_binnings[spline];

        if(other_index == inconfig.i_prime) {
            const Eigen::MatrixXf &hist = inprop.variable_hists[binning];
            for(long int i = 0; i < hist.rows(); ++i) {
                float systw = 1.0;
                if(binning != -1) systw = insyst.GetSplineShift(spline, spline_throw, i);
                for(int k = 0; k < nbins; ++k) {
                    float binsystw = systw;
                    if(binning == -1) binsystw *= insyst.GetSplineShift(spline, spline_throw, k);
                    spec(k) += binsystw * hist(i, k);
                }
            }
        } else {
            for(size_t i = 0; i<inprop.trueLE.size(); ++i){
                float add_w = inprop.added_weights[i]; 
                const int spline_bin = inprop.variable_bin_indices[i][binning];
                float systw = insyst.GetSplineShift(spline, spline_throw, spline_bin);
                float finalw = systw * add_w;
                if(inprop.variable_bin_indices[i][other_index] >= 0) {
                    spec(inprop.variable_bin_indices[i][other_index]) += finalw;
                }
            }
        }

        return PROspec(spec, spec.array().sqrt());
    }
};
