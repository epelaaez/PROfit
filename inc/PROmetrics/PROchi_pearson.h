/**
 * @file PROchi_pearson.h
 * @brief Pearson chi-squared metric for PROfit (canonical CLI name: "pearson").
 * @author PROfit Collaboration
 *
 * @details Defines PROchi_pearson, the textbook Pearson chi-squared: the statistical
 * covariance is diag(prediction), rebuilt from the CURRENT full (physics + spline-shifted)
 * prediction at every evaluation.  Contrast with PROchi (canonical "neyman", stat
 * covariance = diag(data), zero-data bins dropped) and PROCNP (the 3/(1/n+2/mu) mix).
 * Zero-data bins are KEPT (their variance is mu > 0); the fit-region (active-bins)
 * mask is the only exclusion mechanism.  No analytic gradient yet — GradientAnalytic
 * falls back to central-lin with a one-time warning.  Inherits from PROmetric.
 */
#ifndef PROCHI_PEARSON_H_
#define PROCHI_PEARSON_H_

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
#include "PROcess.h"
#include "PROmetric.h"

namespace PROfit{

    /**
     * @brief Pearson chi-squared metric: stat covariance = diag(prediction).
     * @details The diagonal statistical error of each bin is the predicted count mu at the
     * current parameter point (floored at a tiny minimum so M stays nonsingular where the
     * systematic covariance is also empty).  Because mu depends on ALL parameters (physics
     * and splines both shift the prediction), the Full FD gradient modes rebuild the stat
     * diagonal for every perturbation; the Linearised modes freeze M at the base point
     * (Gauss-Newton) as usual.  Systematic nuisance parameters are treated identically
     * to PROchi.
     * @note Like PROCNP, owns copies of the config, propeller, and data objects rather
     * than holding references, to facilitate thread-safe FC pseudo-experiments.
     */
    class PROchi_pearson : public PROmetric
    {
        private:
            std::string model_tag;  ///< String tag identifying the oscillation model in use.

            const PROconfig config; ///< Analysis configuration (owned copy).
            const PROpeller peller; ///< MC event store (owned copy).
            const PROsyst *syst;    ///< Systematic object (non-owning pointer).
            const PROmodel &model;  ///< Physics model (non-owning reference, as in PROchi; an owned copy would slice derived models to the PROmodel base, losing their get_probs override).
            const PROdata data;     ///< Observed data spectrum (owned copy).
            EvalStrategy strat;     ///< Evaluation strategy.
            bool shape_only;        ///< If true, compute chi-squared on area-normalised spectra.
            std::vector<float> physics_param_fixed; ///< Fixed-physics-parameter values (empty = none fixed).
            int fixed_index;  ///< Index of the parameter fixed during a scan (-1 = none).
            float fixed_val;  ///< Value at which the fixed parameter is held.

            Eigen::VectorXf last_param; ///< Parameter vector from the most recent evaluation.
            float last_value;           ///< Chi-squared from the most recent evaluation.

            bool correlated_systematics;      ///< If true, use correlated (off-diagonal) pull covariance.
            Eigen::MatrixXf prior_covariance; ///< Prior covariance matrix for nuisance parameters.
            Eigen::MatrixXf prior_covariance_inv; ///< Inverse of prior_covariance, computed once in the ctor.

            std::vector<Eigen::Index> pearson_active_idx; ///< Active collapsed bins from the fit-region mask; empty when no mask is set.

            FillSpectraCache fs_cache;             ///< Per-thread split-half cache for FillSpectra.

        public:

            /*Function: Constructor bringing all objects together*/
            PROchi_pearson(const std::string tag, const PROconfig &conin, const PROpeller &pin, const PROsyst *systin, const PROmodel &modelin, const PROdata &datain, EvalStrategy strat = EventByEvent, bool shape_only = false, std::vector<float> physics_param_fixed = std::vector<float>());

            /*Function: operator() is what is passed to minimizer.*/
            virtual float operator()(const Eigen::VectorXf &param, Eigen::VectorXf &gradient);
            virtual float operator()(const Eigen::VectorXf &param, Eigen::VectorXf &gradient, bool rungradient);

            /** @brief Return a heap-allocated copy of this PROchi_pearson. */
            PROmetric *Clone() const {
                return new PROchi_pearson(model_tag, config, peller, syst, model, data, strat, shape_only, physics_param_fixed);
            }

            /** @brief Return a const reference to the oscillation model. */
            virtual const PROmodel &GetModel() const {
                return model;
            }

            /** @brief Return a const reference to the systematic object. */
            virtual const PROsyst &GetSysts() const {
                return *syst;
            }

            /** @brief Reset cached state and clear any fixed-parameter list. */
            void reset() {
                physics_param_fixed.clear();
                last_value = 0;
                last_param = Eigen::VectorXf::Constant(last_param.size(), 0);
                fs_cache.invalidate();
            }

            /** @brief Replace the internal systematic pointer with @p new_syst. */
            void override_systs(const PROsyst &new_syst) {
                syst = &new_syst;
                fs_cache.invalidate();
            }

            /**
             * @brief Compute the Gaussian pull penalty for the spline nuisance parameters.
             * @param systs  Spline nuisance parameter values.
             * @return Scalar chi-squared penalty from Gaussian priors.
             */
            virtual float Pull(const Eigen::VectorXf &systs);

            /**
             * @brief Compute the Pearson chi-squared contribution from a single analysis channel.
             * @param global_channel_index  Global channel index.
             * @param cv                    Predicted spectrum.
             * @param var_index             Variable index.
             * @return Pearson chi-squared for that channel.
             */
            float getSingleChannelChi(size_t global_channel_index, const PROspec &cv, size_t var_index, const Eigen::MatrixXf &projection = Eigen::MatrixXf());

            /**
             * @brief Fix a spline nuisance parameter at a given value.
             * @param fix    0-based spline index.
             * @param valin  Value to fix the spline at.
             */
            void fixSpline(int fix, float valin);

            /**
             * @brief Print a breakdown of the Pearson chi-squared contributions at @p param.
             * @param param  Full parameter vector (physics + splines).
             */
            void print(const Eigen::VectorXf &param);
    };


}
#endif
