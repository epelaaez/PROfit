/**
 * @file PROchi.h
 * @brief Covariance-matrix chi-squared metric for PROfit oscillation fitting.
 * @author PROfit Collaboration
 *
 * @details Defines PROchi, which implements the standard covariance-matrix chi-squared:
 *   chi2 = (data - pred)^T M^{-1} (data - pred) + pull_penalty
 * where M is the combined statistical + systematic covariance matrix, evaluated at
 * the current systematic nuisance parameters.  Inherits from PROmetric and is compatible
 * with the PROfitter optimiser.
 */
#ifndef PROCHI_H_
#define PROCHI_H_

// STANDARD
#include <string>
#include <vector>

#include <Eigen/Eigen>

// OUR INCLUDES
#include "PROconfig.h"
#include "PROdata.h"
#include "PROsyst.h"
#include "PROpeller.h"
#include "PROmodel.h"
#include "PROmetric.h"

namespace PROfit{

    /**
     * @brief Covariance-matrix (Pearson/Gaussian) chi-squared metric.
     * @details Gathers the MC store (PROpeller), systematic object (PROsyst), and oscillation
     * model (PROmodel) into a single callable object whose operator() returns the chi-squared
     * value and gradient for use by PROfitter.  All heavy objects are stored as (const) references
     * or pointers to objects owned by the calling executable.
     * @note The oscillation model PROsc and related names may appear in older comments; this class
     * currently uses a generic PROmodel interface.
     * @todo Add capability to define the chi-squared function externally.
     * @todo Improve analytic gradient calculation.
     */
    class PROchi : public PROmetric
    {
        private:

        public:
            std::string model_tag; ///< String tag identifying the oscillation model in use.

            const PROconfig &config;  ///< Analysis configuration (non-owning reference).
            const PROpeller &peller;  ///< MC event store (non-owning reference).
            const PROsyst *syst;      ///< Systematic object (non-owning pointer; may be swapped via override_systs).
            const PROmodel &model;    ///< Physics oscillation model (non-owning reference).
            const PROdata data;       ///< Observed data spectrum (owned copy, collapsed to channel level).
            EvalStrategy strat;       ///< Evaluation strategy (EventByEvent, BinnedGrad, or BinnedChi2).
            bool shape_only;          ///< If true, the chi-squared is computed on area-normalised spectra.
            std::vector<float> physics_param_fixed; ///< Values to hold fixed physics parameters at; empty = none fixed.
            int fixed_index;  ///< Index of the parameter fixed during a profile scan (-1 = none).
            float fixed_val;  ///< Value at which the fixed parameter is held.

            Eigen::VectorXf last_param; ///< Parameter vector from the most recent evaluation (for finite-difference gradient).
            float last_value;           ///< Chi-squared from the most recent evaluation.

            bool correlated_systematics;         ///< If true, use the correlated (off-diagonal) covariance for pull terms.
            Eigen::MatrixXf prior_covariance;    ///< Prior covariance matrix for nuisance parameters (used when correlated).
            Eigen::MatrixXf collapsed_stat_covariance; ///< Statistical covariance in the collapsed bin space.


            /*Function: Constructor bringing all objects together*/
            PROchi(const std::string tag, const PROconfig &conin, const PROpeller &pin, const PROsyst *systin, const PROmodel &modelin, const PROdata &datain, EvalStrategy strat = EventByEvent, bool shape_only = false, std::vector<float> physics_param_fixed = std::vector<float>());


            /*Function: operator() is what is passed to minimizer.*/
            virtual float operator()(const Eigen::VectorXf &param, Eigen::VectorXf &gradient);
            virtual float operator()(const Eigen::VectorXf &param, Eigen::VectorXf &gradient, bool rungradient);

            /** @brief Reset cached state and clear any fixed-parameter list. */
            virtual void reset() {
                physics_param_fixed.clear();
                last_value = 0;
                last_param = Eigen::VectorXf::Constant(last_param.size(), 0);
            }

            /** @brief Return a heap-allocated copy of this PROchi. */
            virtual PROmetric *Clone() const {
                 return new PROchi(model_tag, config, peller, syst, model, data, strat, shape_only, physics_param_fixed);

            }

            /** @brief Return a const reference to the oscillation model. */
            virtual const PROmodel &GetModel() const {
                return model;
            }

            /** @brief Return a const reference to the systematic object. */
            virtual const PROsyst &GetSysts() const {
                return *syst;
            }

            /** @brief Replace the internal systematic pointer with @p new_syst. */
            virtual void override_systs(const PROsyst &new_syst) {
                syst = &new_syst;
            }

            /**
             * @brief Compute the Gaussian pull penalty for the spline nuisance parameters.
             * @param systs  Spline nuisance parameter values.
             * @return Scalar chi-squared penalty from Gaussian priors.
             */
            virtual float Pull(const Eigen::VectorXf &systs);

            /**
             * @brief Fix a specific spline nuisance parameter at a given value.
             * @param fix    0-based spline index.
             * @param valin  Value to fix the spline at.
             */
            void fixSpline(int fix, float valin);

            /**
             * @brief Compute the chi-squared contribution from a single analysis channel.
             * @param global_channel_index  Global channel index.
             * @param cv                    Predicted (CV) spectrum.
             * @param var_index             Variable index.
             * @return Chi-squared for that channel.
             */
            float getSingleChannelChi(size_t global_channel_index, const PROspec &cv, size_t var_index);

            /**
             * @brief Print a breakdown of the chi-squared contributions at @p param.
             * @param param  Full parameter vector (physics + splines).
             */
            void print(const Eigen::VectorXf &param);
    };
}
#endif
