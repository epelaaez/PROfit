/**
 * @file PROmodel2flav.h
 * @brief Two-flavour-like 3+1 single-channel oscillation models (short-baseline approximation).
 * @author PROfit Collaboration
 * @internal These models are constructed only via
 * get_model_from_string (src/PROmodel.cxx); prefer the factory over direct construction.
 *
 * @details Declares:
 *   - PROnumudisTEST — two-variable (L, E) validation version of the numudis model.
 * The single-variable models formerly here (PROnumudis, PROnueapp, PROnuedis) are now
 * sine-kernel recipes evaluated by PROsineModel (PROmodels/PROmodelSine.h) under the
 * factory tags SBL_2flav_numudis / SBL_2flav_nueapp / SBL_2flav_nuedis (legacy
 * aliases numudis / nueapp / nuedis).
 */
#ifndef PROMODEL2FLAV_H
#define PROMODEL2FLAV_H

#include "PROmodel.h"

namespace PROfit {

/**
 * @brief Two-variable test version of the numudis model operating on separate L and E variables.
 * @details Takes separate "L" and "E" variables from parameter_map and builds H_combined on the
 * 2D (L x E) physics grid.  get_probs() computes L/E internally, so the physics is identical
 * to the numudis recipe model.  The two models should produce identical spectra and can be used
 * to validate the multi-variable code path against the standard single L/E variable approach.
 */
class PROnumudisTEST : public PROmodel {
public:
    /**
     * @brief Construct the two-variable PROnumudisTEST model.
     * @param prop          MC event store; used to build H_combined on the (L, E) grid.
     * @param parameter_map Map from physics variable name to variable index in PROpeller.
     *                      Must contain both "L" and "E".
     */
    PROnumudisTEST(const PROpeller &prop, const std::map<std::string,int> &parameter_map);

    /**
     * @brief Compute the 3+1 nu_mu survival probability (identical physics to the numudis recipe).
     * @param dmsq          Mass splitting Delta m^2 in eV^2 (may be in log10 space; converted internally).
     * @param sinsq2thmumu  sin^2(2 theta_mumu) (may be in log10 space; converted internally).
     * @param le            L/E ratio in km/GeV, computed internally from var_arrs[0]=L / var_arrs[1]=E.
     * @return Survival probability in [0, 1].
     */
    float Pmumu(float dmsq, float sinsq2thmumu, float le) const;

    /**
     * @brief Compute oscillation probabilities on the 2D (L×E) grid.
     * @details var_arrs[0] = L values, var_arrs[1] = E values, each of length n_L * n_E (flat row-major order).
     *          L/E is computed internally for each grid point, so the result is physically identical to PROnumudis::get_probs.
     * @param phys     Physics parameter vector: (log10(dmsq), log10(sinsq2thmm)).
     * @param var_arrs 2-element vector: {L array [km], E array [GeV]}, each of size n_phys_bins.
     * @return Matrix of shape (n_phys_bins, 2): column 0 = 1 (no-osc), column 1 = P(nu_mu -> nu_mu).
     */
    Eigen::MatrixXf get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const override;
    /** @brief Closed-form derivatives of get_probs (see PROmodel::get_probs_grad). */
    std::vector<Eigen::MatrixXf> get_probs_grad(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const override;

    /** @brief Closed-form derivatives above, so the analytic gradient is exact here (see PROmodel::has_analytic_gradient). */
    bool has_analytic_gradient() const override { return true; }
};

}

#endif
