/**
 * @file PROchi_CNP.h
 * @brief Combined Neyman-Pearson (CNP) chi-squared metric for PROfit (canonical CLI name: "CNP").
 * @author PROfit Collaboration
 *
 * @details Defines PROCNP, which implements the Combined Neyman-Pearson chi-squared statistic
 * that uses a mixed covariance matrix combining Pearson (MC-predicted) and Neyman (data-driven)
 * error terms.  This is the recommended chi-squared for neutrino counting experiments where
 * data statistical uncertainty follows Poisson statistics but the predicted spectrum has
 * finite Monte Carlo statistics.  Inherits from PROmetric.
 */
#ifndef PROCNP_H_
#define PROCNP_H_

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
     * @brief Combined Neyman-Pearson chi-squared metric.
     * @details Implements the CNP statistic in which the diagonal error for each bin is
     * (1/3)(1/n_data + 2/n_pred), approximating the hybrid Poisson regime where neither pure
     * Pearson nor pure Neyman chi-squared is optimal.  Systematic nuisance parameters are
     * treated identically to PROchi.
     * @note Unlike PROchi, PROCNP owns copies of the config, propeller, model, and data
     * objects rather than holding references, to facilitate thread-safe FC pseudo-experiments.
     */
    class PROCNP : public PROmetric
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

            // Cache for CollapseMatrix(FillSpectra(noshiftvec_for_phys)). The noshiftvec
            // CV depends only on the physics parameters (splines are zeroed out). Cache
            // is invalidated by reset() and override_systs().
            Eigen::VectorXf cnp_cached_phys;       ///< Last physics subvector used to fill cnp_cached_collapsed_cv.
            Eigen::VectorXf cnp_cached_collapsed_cv; ///< Cached collapsed noshift CV spectrum.
            bool cnp_cv_cache_valid = false;       ///< True iff cnp_cached_collapsed_cv matches cnp_cached_phys.
            std::vector<Eigen::Index> cnp_active_idx; ///< Active collapsed bins from the fit-region mask; empty when no mask is set.

            /// Returns CollapseMatrix(FillSpectra(noshiftvec built from `phys`)). Hits the
            /// cache when `phys` matches the last cached call; otherwise recomputes.
            Eigen::VectorXf cachedNoshiftCollapsedCV(const Eigen::VectorXf &phys, Eigen::Index param_size);

            FillSpectraCache fs_cache;             ///< Per-thread split-half cache for FillSpectra.

        public:

            /*Function: Constructor bringing all objects together*/
            PROCNP(const std::string tag, const PROconfig &conin, const PROpeller &pin, const PROsyst *systin, const PROmodel &modelin, const PROdata &datain, EvalStrategy strat = EventByEvent, bool shape_only = false, std::vector<float> physics_param_fixed = std::vector<float>());

            /*Function: operator() is what is passed to minimizer.*/
            virtual float operator()(const Eigen::VectorXf &param, Eigen::VectorXf &gradient);
            virtual float operator()(const Eigen::VectorXf &param, Eigen::VectorXf &gradient, bool rungradient);

            /** @brief Return a heap-allocated copy of this PROCNP. */
            PROmetric *Clone() const {
                return new PROCNP(model_tag, config, peller, syst, model, data, strat, shape_only, physics_param_fixed);
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
                cnp_cv_cache_valid = false;
                fs_cache.invalidate();
            }

            /** @brief Replace the internal systematic pointer with @p new_syst. */
            void override_systs(const PROsyst &new_syst) {
                syst = &new_syst;
                cnp_cv_cache_valid = false;
                fs_cache.invalidate();
            }

            /**
             * @brief Compute the Gaussian pull penalty for the spline nuisance parameters.
             * @param systs  Spline nuisance parameter values.
             * @return Scalar chi-squared penalty from Gaussian priors.
             */
            virtual float Pull(const Eigen::VectorXf &systs);

            /**
             * @brief Compute the CNP chi-squared contribution from a single analysis channel.
             * @param global_channel_index  Global channel index.
             * @param cv                    Predicted spectrum.
             * @param var_index             Variable index.
             * @return CNP chi-squared for that channel.
             */
            float getSingleChannelChi(size_t global_channel_index, const PROspec &cv, size_t var_index, const Eigen::MatrixXf &projection = Eigen::MatrixXf());

            /**
             * @brief Fix a spline nuisance parameter at a given value.
             * @param fix    0-based spline index.
             * @param valin  Value to fix the spline at.
             */
            void fixSpline(int fix, float valin);

            /**
             * @brief Print a breakdown of the CNP chi-squared contributions at @p param.
             * @param param  Full parameter vector (physics + splines).
             */
            void print(const Eigen::VectorXf &param);
    };


}
#endif
