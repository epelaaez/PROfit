#ifndef PROPELLER_H_
#define PROPELLER_H_

#include "PROconfig.h"
#include "PROserial.h"
#include <Eigen/Eigen>
// STANDARD
#include <Eigen/src/Core/Matrix.h>
#include <vector>
#include <chrono>
namespace PROfit{
    class PROhistStorage {
        private:
            size_t n_vars = 0;
            std::vector<Eigen::MatrixXf> data;

            // Only stores when i <= j
            size_t compute_index(size_t i, size_t j) const {
                return (i * n_vars) - (i * (i - 1)) / 2 + (j - i);
            }

            friend class boost::serialization::access;

            template<class Archive>
                void serialize(Archive& ar, const unsigned int version) {
                    (void)version;
                    ar & n_vars;

                    if (Archive::is_loading::value) {
                        if (n_vars > 0) {
                            data.resize(n_vars * (n_vars + 1) / 2);
                        } else {
                            data.clear();
                        }
                    }

                    if (n_vars > 0) {
                        for (auto& mat : data) {
                            ar & mat;
                        }
                    }
                }
        public:
            PROhistStorage() {}  
            PROhistStorage(size_t n) {init(n);}

            void init(size_t n) { n_vars = n;data.resize(n * (n + 1) / 2);}


            //Grab whole matrix
            Eigen::Ref<const Eigen::MatrixXf> operator()(size_t i, size_t j) const {
                if (i <= j) {
                    return data[compute_index(i, j)]; 
                } else {
                    return data[compute_index(j, i)].transpose();
                }
            }

            //element
            float operator()(size_t rowvar, size_t colvar, size_t l, size_t m) const {
                if (rowvar <= colvar) {
                    return data[compute_index(rowvar, colvar)](l, m);
                } else {
                    return data[compute_index(colvar, rowvar)](m, l); 
                }
            }


            // Direct access for setting (must use i <= j)
            Eigen::MatrixXf& set(size_t i, size_t j) {
                if (i > j){
                    log<LOG_ERROR>(L"%1% || If your seeing this, something went wrong. dont access PROhistStorage out of order.") % __func__;
                    exit(EXIT_FAILURE);
                }
                return data[compute_index(i, j)];
            }


            size_t size() const { return n_vars; }
    };

    /*Class: The PROpeller, which moves the analysis forward. A class to keep all MC events for oscllation event-by-event.
    */
    class PROpeller {

        private:
            friend class boost::serialization::access;

            // Serialization function for boost that will allow for save state of propeller
            template <class Archive>
                void serialize(Archive& ar, [[maybe_unused]] const unsigned int version) {
                    ar & added_weights;
                    ar & model_rule;
                    ar & variable_mc_stat_err;
                    ar & variable_bin_indices;
                    ar & variable_hist_storage;
                    ar & variable_midbin;
                    ar & variable_values;
                    ar & hash;
                }

        public:

            //Empty Constructor
            PROpeller(){
                variable_values.clear();
                added_weights.clear();
                model_rule.clear();
                hash = -1;
            };

            /*Function: Primary Constructor from raw std::vectors of MC values */ 
            PROpeller( std::vector<std::vector<float>> &invars, std::vector<float> &inadded_weights,  std::vector<int> &inmodel_rule) :  added_weights(inadded_weights),  model_rule(inmodel_rule), variable_values(invars){
                //for(size_t i = 0; i < bin_indices.size(); ++i)
            };

            /* the Core MC is saved in these vectors.*/

            std::vector<float> added_weights;
            std::vector<int>   model_rule;
            std::vector<std::vector<int>> variable_bin_indices;
            std::vector<std::vector<float>> variable_values;
            std::vector<Eigen::VectorXf> variable_mc_stat_err;
            std::vector<Eigen::VectorXf> variable_midbin;
            PROhistStorage variable_hist_storage;

            uint32_t           hash;


            // boost serialize save to file
            void save(const std::string& filename) const {
                auto start = std::chrono::high_resolution_clock::now();
                std::ofstream ofs(filename, std::ios::binary);
                boost::archive::binary_oarchive oa(ofs);
                oa << *this;
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = end - start;
                log<LOG_INFO>(L"%1% || Serialization save of PROpeller into file  %2% took %3% seconds") % __func__ % filename.c_str() % elapsed.count();
            }

            // Load from file
            void load(const std::string& filename) {
                auto start = std::chrono::high_resolution_clock::now();
                std::ifstream ifs(filename,std::ios::binary);
                boost::archive::binary_iarchive ia(ifs);
                ia >> *this;
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = end - start;
                log<LOG_INFO>(L"%1% || Serialization load of PROpeller from file  %2% took %3% seconds") % __func__ % filename.c_str() %elapsed.count();
            }

            // Scale detector weights for POT studies
            void scale(const PROconfig &inconfig, std::map<std::string, float> scaling_map){
                for (const auto& [detector, value] : scaling_map) {

                    if (value <= 0.0f) {
                        log<LOG_ERROR>(L"%1% || Scale factor %2% for '%3%' is invalid. Must be > 0.")
                            % __func__ % value % detector.c_str();
                        exit(EXIT_FAILURE);
                    }

                    log<LOG_INFO>(L"%1% || Wildcard '%2%' with scaling factor %3% matches:")
                        % __func__ % detector.c_str() % value;

                    std::vector<std::string> scalenames;
                    for (const auto& name : inconfig.m_fullnames) {
                        if (name.find(detector) != std::string::npos) {
                            scalenames.push_back(name);
                        }
                    }

                    log<LOG_INFO>(L"%1% || %2% . ") % __func__  % scalenames;


                    std::vector<std::vector<int>> scaleotherbins;
                    for(size_t io =0; io<inconfig.m_num_variables; io++){
                        std::vector<int> tmpbins;
                        for(auto &name: scalenames){
                            size_t is = inconfig.GetSubchannelIndex(name);     
                            size_t ic = inconfig.GetLocalChannelIndexFromGlobalSubchannelIndex(is); 
                            size_t start = inconfig.GetGlobalVariableBinStart(is,io); 
                            for(size_t b = 0; b < inconfig.m_channel_variable_num_bins[io][ic] ; b++){
                                tmpbins.push_back((int)(start+b));
                            }
                        }
                        scaleotherbins.push_back(tmpbins);

                    }


                    //Scale the binned bits first
                    for(size_t io =0; io<inconfig.m_num_variables; io++){
                        for(size_t jo =io; jo<inconfig.m_num_variables; jo++){
                            log<LOG_INFO>(L"%1% || and scales other bins  %2% .") % __func__  %  scaleotherbins[io];
                            for (int o : scaleotherbins[io]) {
                                for (int j : scaleotherbins[jo]) {
                                    variable_hist_storage.set(io,jo)(o, j) *= value;
                                }
                            }
                        }
                    }
                    //And then the unbinned weights
                    for (size_t i = 0; i < added_weights.size(); ++i) {
                        for(size_t io =0; io<inconfig.m_num_variables; io++){
                            if(io>0)break;//Hmm, we scale of the reco and not other bins. That seems fine, but might want to rethink
                            int bin = variable_bin_indices[i][io];
                            if (std::find(scaleotherbins[io].begin(), scaleotherbins[io].end(), bin) != scaleotherbins[io].end()) {
                                added_weights[i] *= value;
                            }
                        }
                    }

                    log<LOG_INFO>(L"%1% || Applied %2% scaling for '%3%'")
                        % __func__ % value % detector.c_str();

                }

            }

    };

}
#endif
