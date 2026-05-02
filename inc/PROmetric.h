/**
 * @file PROmetric.h
 * @brief Abstract base class defining the chi-squared metric interface for PROfit optimisers.
 * @author PROfit Collaboration
 *
 * @details PROmetric is the pure-virtual interface that connects the physics chi-squared
 * calculation to the PROfitter multi-start optimiser.  Concrete implementations
 * (PROchi, PROCNP, PROpoisson) compute different chi-squared statistics while sharing
 * the common bounds, fixed-parameter, and call-counting infrastructure defined here.
 */
#ifndef PROMETRIC_H
#define PROMETRIC_H

#include "PROsyst.h"
#include "PROmodel.h"

#include <Eigen/Eigen>

namespace PROfit {

    /**
     * @brief Abstract base class for PROfit chi-squared metrics passed to the optimiser.
     * @details Defines the interface required by PROfitter: parameter bounds, fixed-parameter
     * masking, call counting, and the functor operator() that returns chi-squared and gradient.
     * All concrete metrics (PROchi, PROCNP, PROpoisson) derive from this class.
     */
    class PROmetric {
        public:
            /**
             * @brief Strategy for evaluating the chi-squared (and its gradient).
             */
            enum EvalStrategy {
                EventByEvent, ///< Evaluate event-by-event (slowest but most accurate for oscillation weights).
                BinnedGrad,   ///< Use pre-binned histograms with gradient calculation.
                BinnedChi2    ///< Use pre-binned histograms, chi-squared only (no analytic gradient).
            };

            std::vector<bool> is_fixed; ///< Per-parameter flags: true if the parameter is held fixed during fitting.
            Eigen::VectorXf  lb;        ///< Lower bounds for all parameters.
            Eigen::VectorXf  ub;        ///< Upper bounds for all parameters.


            /** @brief Replace the internal systematic object pointer with @p new_syst. */
            virtual void override_systs(const PROsyst &new_syst) = 0;
            /**
             * @brief Evaluate the chi-squared and its gradient.
             * @param param     Current parameter vector.
             * @param gradient  Output gradient vector (same size as @p param); filled on return.
             * @return Chi-squared value.
             */
            virtual float operator()(const Eigen::VectorXf &param, Eigen::VectorXf &gradient) = 0;
            /**
             * @brief Evaluate the chi-squared, optionally skipping gradient computation.
             * @param param       Current parameter vector.
             * @param gradient    Output gradient vector; only filled when @p rungradient is true.
             * @param rungradient If true, compute the gradient; if false, skip it for speed.
             * @return Chi-squared value.
             */
            virtual float operator()(const Eigen::VectorXf &param, Eigen::VectorXf &gradient, bool rungradient) = 0;
            /** @brief Reset any cached state (e.g. last parameter vector and value). */
            virtual void reset() = 0;
            /** @brief Return a heap-allocated deep copy of this PROmetric. */
            virtual PROmetric *Clone() const = 0;
            /** @brief Return a const reference to the physics model used by this metric. */
            virtual const PROmodel &GetModel() const = 0;
            /** @brief Return a const reference to the systematic object used by this metric. */
            virtual const PROsyst  &GetSysts() const = 0;
            /**
             * @brief Compute the chi-squared contribution from a single channel.
             * @param channel_index  Global channel index.
             * @param cv             Central-value (predicted) spectrum.
             * @param var_index      Variable index.
             * @return Chi-squared for that channel.
             */
            virtual float getSingleChannelChi(size_t channel_index, const PROspec& cv, size_t var_index) = 0;
            PROmetric() = default;
            virtual ~PROmetric() {}
            /**
             * @brief Fix a spline nuisance parameter at a specific value.
             * @param idx   0-based spline index.
             * @param val   Value to fix the spline at.
             */
            virtual void fixSpline(int,float)  = 0;
            /**
             * @brief Compute the Gaussian pull penalty for the given nuisance parameter vector.
             * @param systs  Spline nuisance parameter values.
             * @return Scalar pull penalty (chi2 contribution from priors).
             */
            virtual float Pull(const Eigen::VectorXf &systs) = 0;
            /**
             * @brief Print a human-readable summary of the metric evaluation at @p param.
             * @param param  Parameter vector to evaluate at.
             */
            virtual void print(const Eigen::VectorXf &param) = 0;
            /**
             * @brief Return the total number of parameters (physics + spline nuisance).
             * @return nparams from the model plus the number of spline systematics.
             */
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

            /** @brief Return the total number of times operator() has been called since last reset. */
            size_t getCallCount() const { return call_count; }
            /** @brief Reset the call counter to zero. */
            void resetCallCount() { call_count = 0; }


            /**
             * @brief Set parameter bounds and mark zero-range parameters as fixed.
             * @param lbin  Lower bounds vector.
             * @param ubin  Upper bounds vector.
             */
            void setBounds(const Eigen::VectorXf& lbin, const Eigen::VectorXf& ubin) {
                lb = lbin;
                ub = ubin;
                is_fixed.resize(lbin.size());
                for(int i = 0; i < lbin.size(); ++i) {
                    is_fixed[i] = (std::abs(ubin(i) - lbin(i)) < 1e-10);
                }
            }

            /** @brief Clear the is_fixed mask so that all parameters are free to be optimised. */
            void freeParams() {
                is_fixed.clear();
            }

        protected:
            mutable std::atomic<size_t> call_count{0}; ///< Thread-safe counter of operator() invocations.

    };

};

#endif

