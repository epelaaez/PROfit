/**
 * @file PROmodelLBL.h
 * @brief Standard three-flavour long-baseline oscillation model (NuFastLBL wrapper).
 * @author PROfit Collaboration
 * @internal Private header — PROLBL is constructed only via get_model_from_string
 * (src/PROmodel.cxx); do not include from public headers.
 */
#ifndef PROMODELLBL_H
#define PROMODELLBL_H

#include "PROmodel.h"

namespace PROfit {

/**
 * @brief Standard three-flavour long-baseline oscillation model including matter effects.
 * @details Wraps the NuFastLBL library to compute the full 3x3 oscillation probability matrix in matter,
 * assuming a baseline of 1300 km (DUNE), Earth matter density 3 g/cc, and electron fraction Ye = 0.5.
 * All nine active-flavour transitions (nu_e, nu_mu, nu_tau → nu_e, nu_mu, nu_tau) are provided.
 * Parameters: dmsq_21, dmsq_31 [eV^2, linear], sin^2(theta_12), sin^2(theta_13), sin^2(theta_23) [linear],
 * delta_CP [rad, linear].  None of the six parameters are stored in log10 space.
 */
class PROLBL : public PROmodel {
public:
    static constexpr float rho_earth = 3; ///< Earth average matter density in g/cc used for MSW potential.
    static constexpr float Ye_earth = 0.5; ///< Electron fraction of the Earth matter.

    /**
     * @brief Construct the PROLBL model.
     * @param prop          MC event store; used to build H_combined.
     * @param parameter_map Map from physics variable name (e.g., "L/E") to variable index in PROpeller.
     *                      Must contain the key "L/E".
     */
    PROLBL(const PROpeller &prop, const std::map<std::string,int> &parameter_map);

    /// @brief nu_e → nu_e survival probability via NuFastLBL. @param params Physics vector (6 params). @param le L/E [km/GeV]. @return P(nu_e→nu_e).
    float Pee(const Eigen::VectorXf &params, float le);
    /// @brief nu_e → nu_mu appearance probability via NuFastLBL. @param params Physics vector (6 params). @param le L/E [km/GeV]. @return P(nu_e→nu_mu).
    float Pemu(const Eigen::VectorXf &params, float le);
    /// @brief nu_e → nu_tau appearance probability via NuFastLBL. @param params Physics vector. @param le L/E [km/GeV]. @return P(nu_e→nu_tau).
    float Petau(const Eigen::VectorXf &params, float le);
    /// @brief nu_mu → nu_e appearance probability via NuFastLBL. @param params Physics vector. @param le L/E [km/GeV]. @return P(nu_mu→nu_e).
    float Pmue(const Eigen::VectorXf &params, float le);
    /// @brief nu_mu → nu_mu survival probability via NuFastLBL. @param params Physics vector. @param le L/E [km/GeV]. @return P(nu_mu→nu_mu).
    float Pmumu(const Eigen::VectorXf &params, float le);
    /// @brief nu_mu → nu_tau appearance probability via NuFastLBL. @param params Physics vector. @param le L/E [km/GeV]. @return P(nu_mu→nu_tau).
    float Pmutau(const Eigen::VectorXf &params, float le);
    /// @brief nu_tau → nu_e appearance probability via NuFastLBL. @param params Physics vector. @param le L/E [km/GeV]. @return P(nu_tau→nu_e).
    float Ptaue(const Eigen::VectorXf &params, float le);
    /// @brief nu_tau → nu_mu appearance probability via NuFastLBL. @param params Physics vector. @param le L/E [km/GeV]. @return P(nu_tau→nu_mu).
    float Ptaumu(const Eigen::VectorXf &params, float le);
    /// @brief nu_tau → nu_tau survival probability via NuFastLBL. @param params Physics vector. @param le L/E [km/GeV]. @return P(nu_tau→nu_tau).
    float Ptautau(const Eigen::VectorXf &params, float le);

    Eigen::MatrixXf get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const override;
};

}

#endif
