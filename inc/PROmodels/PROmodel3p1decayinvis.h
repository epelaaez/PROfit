/**
 * @file PROmodel3p1decayinvis.h
 * @brief 3+1 sterile-neutrino model with invisible decay of the heavy mass state.
 * @author PROfit Collaboration
 * @internal This model is constructed only via
 * get_model_from_string (src/PROmodel.cxx); prefer the factory over direct construction.
 *
 * @details Declares PRO3p1_decay_invis: the standard 3+1 short-baseline picture with the
 * fourth mass eigenstate decaying into invisible (non-interacting) particles.  For the
 * visible-decay counterparts, where the daughters remain detectable and migrate to lower
 * truth energies, see inc/PROmodels/PROmodel3p1decayvis.h.
 */
#ifndef PROMODEL3P1DECAYINVIS_H
#define PROMODEL3P1DECAYINVIS_H

#include "PROmodel.h"

namespace PROfit {

/**
 * @brief 3+1 sterile-neutrino model with invisible decay of the heavy mass eigenstate.
 * @details Extends the standard 3+1 picture by allowing the fourth mass eigenstate to decay into
 * invisible (non-interacting) particles with coupling strength g^2.  The oscillation probability
 * is modified by an exponential damping factor exp(-g^2 * Delta / (8 pi)) where
 * Delta = 1.267 * dmsq * L/E.  See arxiv:2204.00612 (IceCube) and PRD 110, 075002 for derivation.
 * Parameters: dmsq [log10], |U_e4|^2 [log10], |U_mu4|^2 [log10], g^2 [linear, >= 0].
 * A combined unitarity + positivity constraint is enforced via model_constraint.
 */
class PRO3p1_decay_invis : public PROmodel {
public:
    /**
     * @brief Construct the PRO3p1_decay_invis model.
     * @param prop          MC event store; used to build H_combined.
     * @param parameter_map Map from physics variable name (e.g., "L/E") to variable index in PROpeller.
     *                      Must contain the key "L/E".
     */
    PRO3p1_decay_invis(const PROpeller &prop, const std::map<std::string,int> &parameter_map);

    int UnitarityConstraint(const Eigen::VectorXf &v);

    float Pmue(float dmsq, float Ue4sq, float Um4sq, float g2, float le) const;

    float Pmumu(float dmsq, float Ue4sq, float Um4sq, float g2, float le) const;

    float Pee(float dmsq, float Ue4sq, float Um4sq, float g2, float le) const;

    Eigen::MatrixXf get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const override;
};

}

#endif
