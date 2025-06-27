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

            Eigen::MatrixXf operator()(size_t i, size_t j) const {
                if (i <= j) {
                    return data[compute_index(i, j)];
                } else {
                    return data[compute_index(j, i)].transpose();
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
                    ar & variableMCStatErr;
                    ar & variable_bin_indices;
                    ar & variable_hist_storage;
                    ar & hist;
                    ar & histLE;
                    ar & mcStatErr;
                    ar & trueLE;
                    ar & variable_hists;
                    ar & hash;
                }

        public:

            //Empty Constructor
            PROpeller(){
                trueLE.clear();
                added_weights.clear();
                model_rule.clear();
                hash = -1;
            };

            /*Function: Primary Constructor from raw std::vectors of MC values */ 
            PROpeller( std::vector<float> &intruth, std::vector<float> &inadded_weights,  std::vector<int> &inmodel_rule) : trueLE(intruth), added_weights(inadded_weights),  model_rule(inmodel_rule) {
                //size_t nevents = trueLE.size();
                //hist = Eigen::MatrixXf::Constant(config.m_num_variable_bins_total[config.i_osc], config.m_num_variable_bins_total[config.i_prime], 0);
                //for(size_t i = 0; i < bin_indices.size(); ++i)
                //    hist(true_bin_indices[i], bin_indices[i]) += added_weights[i];
                //hash = config.hash;
            };

            /* the Core MC is saved in these vectors.*/

            std::vector<float> added_weights;
            std::vector<int>   model_rule;
            std::vector<std::vector<int>> variable_bin_indices;
            std::vector<Eigen::VectorXf> variableMCStatErr;
            PROhistStorage variable_hist_storage;


            Eigen::MatrixXf    hist;
            Eigen::VectorXf    histLE;
            Eigen::VectorXf    mcStatErr;

            std::vector<float> trueLE;
            std::vector<Eigen::MatrixXf> variable_hists;

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


    };

}
#endif
