#ifndef PROMETRIC_H
#define PROMETRIC_H

#include "PROsyst.h"
#include "PROmodel.h"

#include <Eigen/Eigen>

namespace PROfit {

    class PROmetric {
        public:
            enum EvalStrategy {
                EventByEvent,
                BinnedGrad,
                BinnedChi2
            };

            std::vector<bool> is_fixed;
            Eigen::VectorXf  lb;
            Eigen::VectorXf  ub;


            virtual void override_systs(const PROsyst &new_syst) = 0;
            virtual float operator()(const Eigen::VectorXf &param, Eigen::VectorXf &gradient) = 0;
            virtual float operator()(const Eigen::VectorXf &param, Eigen::VectorXf &gradient, bool nograd) = 0;
            virtual void reset() = 0;
            virtual PROmetric *Clone() const = 0;
            virtual const PROmodel &GetModel() const = 0;
            virtual const PROsyst  &GetSysts() const = 0;
            virtual float getSingleChannelChi(size_t channel_index, size_t var_index) = 0;
            PROmetric() = default;
            virtual ~PROmetric() {}
            virtual void fixSpline(int,float)  = 0;
            virtual float Pull(const Eigen::VectorXf &systs) = 0;
            virtual void print(const Eigen::VectorXf &param) = 0;
            size_t nParams() const {return GetModel().nparams + GetSysts().GetNSplines();}

            PROmetric(const PROmetric&) {}
            PROmetric& operator=(const PROmetric&) { return *this; }


            Eigen::VectorXf LowerBound() const {
                size_t nphys = GetModel().nparams;
                size_t nparams = nParams();
                Eigen::VectorXf lb = Eigen::VectorXf::Constant(nparams, -3.0);
                for (size_t i = 0; i < nphys; ++i) {
                    lb(i) = GetModel().lb(i);
                }
                for(size_t i = nphys; i < nparams; ++i) {
                    lb(i) = GetSysts().spline_lo[i-nphys];
                }
                return lb;
            }

            Eigen::VectorXf UpperBound() const {
                size_t nphys = GetModel().nparams;
                size_t nparams = nParams();
                Eigen::VectorXf ub = Eigen::VectorXf::Constant(nparams, 3.0);
                for (size_t i = 0; i < nphys; ++i) {
                    ub(i) = GetModel().ub(i);
                }
                for(size_t i = nphys; i < nparams; ++i) {
                    ub(i) = GetSysts().spline_hi[i-nphys];
                }
                return ub;
            }

            size_t getCallCount() const { return call_count; }
            void resetCallCount() { call_count = 0; }


            void setBounds(const Eigen::VectorXf& lbin, const Eigen::VectorXf& ubin) {
                lb = lbin;
                ub = ubin;
                is_fixed.resize(lbin.size());
                for(int i = 0; i < lbin.size(); ++i) {
                    is_fixed[i] = (std::abs(ubin(i) - lbin(i)) < 1e-10);
                }
            }

            void freeParams() {
                is_fixed.clear();
            }

        protected:
            mutable std::atomic<size_t> call_count{0};

    };

};

#endif

