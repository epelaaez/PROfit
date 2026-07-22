/**
 * @file PROmodel3p1decayvis.h
 * @brief 3+1 sterile-neutrino models with visible decay of the heavy mass state.
 * @author PROfit Collaboration
 * @internal These models are constructed only via
 * get_model_from_string (src/PROmodel.cxx); prefer the factory over direct construction.
 *
 * @details Declares the 3+1 + visible-decay family, implementing the two models of
 * https://journals.aps.org/prd/abstract/10.1103/PhysRevD.110.075002 :
 *   - PRO3p1_decay_vis_model1 — nu4 -> nu phi with coupling g_phi to the sterile state;
 *     the daughter neutrino is a superposition of active flavours (mixing-weighted).
 *   - PRO3p1_decay_vis_model2 — decay driven by a coupling g_e in the flavour basis;
 *     only the nue-sector channels receive a decay contribution.
 *
 * Unlike the other model families, these models are 2D in (truth L, truth E) and compute
 * event COUNTS rather than per-bin probabilities: the visible decay migrates events from a
 * parent truth-E bin to lower daughter truth-E bins at the same L, so the prediction at a
 * bin depends non-locally on the truth-level MC population N_truth of other bins.
 * FillSpectra detects this via uses_get_counts() and converts counts back to effective
 * probabilities (counts / N_truth) before the H_combined GEMV.
 */
#ifndef PROMODEL3P1DECAYVIS_H
#define PROMODEL3P1DECAYVIS_H

#include "PROmodel.h"

#include <array>

namespace PROfit {

/**
 * @brief 3+1 model with visible decay nu4 -> nu phi (model 1, sterile-coupled scalar).
 * @details Parameters: (dmsq, |U_e4|^2, |U_mu4|^2 [log10 space], g_phi [linear]).
 * Probability-type components (prob_types / model rules):
 *   - 0       — no-oscillation-no-decay,
 *   - 1, 2, 3 — same-bin oscillation contribution for P_mumu / P_mue / P_ee,
 *   - 4, 5, 6 — visible-decay daughter contribution for the same channels, populated at
 *               lower truth-E bins of the paired decay subchannels.
 * The XML must pair every osc subchannel (rule 1/2/3) with a decay subchannel (rule 4/5/6)
 * using identical truth binning; the flat-grid offset between each pair is auto-detected
 * from N_truth at construction (decay_bin_offsets).
 */
class PRO3p1_decay_vis_model1 : public PROmodel {
public:
    /// Truth-level MC populations at each flat-physics bin and model rule.  Only the
    /// visible-decay models need this: their migration sum reads truth counts at bins
    /// other than the destination (see get_counts).  Non-decay models don't build it.
    Eigen::MatrixXf N_truth;

    /// Per-pair flat-grid index offsets mapping a parent bin in an osc subchannel
    /// (rule 1/2/3) to the equivalent bin in its paired decay subchannel (rule
    /// 4/5/6).  Indexed by `osc_rule - 1`: `decay_bin_offsets[0]` is the rule-1 -> 4
    /// shift, `[1]` is rule-2 -> 5, `[2]` is rule-3 -> 6.  Each pair is allowed its
    /// own shift, so the pair-ordering within the XML can differ between channels;
    /// each individual pair must still be self-consistent across detectors (verified
    /// at construction).  Computed from N_truth — no manual XML annotation needed.
    std::array<long int, 3> decay_bin_offsets = {0, 0, 0};

    /**
     * @brief Construct the model 1 visible-decay model.
     * @param prop          MC event store; used to build H_combined and N_truth.
     * @param parameter_map Map from physics variable name to variable index in PROpeller.
     *                      Must contain the keys "L" and "E".
     */
    PRO3p1_decay_vis_model1(const PROpeller &prop, const std::map<std::string,int> &parameter_map);

    const Eigen::MatrixXf& get_N_truth() const override { return N_truth; }

    /**
     * @brief Enforce |U_e4|^2 + |U_mu4|^2 < 1 and g_phi >= 0.
     * @param v  Physics parameter vector in the fitter's internal space.
     * @return 1 if the point is physically allowed, 0 otherwise.
     */
    int UnitarityConstraint(const Eigen::VectorXf &v);

    bool uses_get_counts() const override { return true; }

    /**
     * @brief Effective per-bin probabilities: get_counts(phys, ., N_truth) / N_truth.
     * @details The counts formulation is the authoritative one for this model (down-migration
     * makes the per-bin ratio depend on other bins' populations); note that effective
     * probabilities greater than one are possible with large down-smearing.
     */
    Eigen::MatrixXf get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const override;

    /**
     * @brief Compute oscillated + decay-migrated event counts on the flat (L, E) grid.
     * @param phys          Physics parameters (dmsq, Ue4^2, Um4^2 in log10; g_phi linear).
     * @param var_arrs      {L array, E array}, each of length n_phys_bins (flat grid midpoints).
     * @param N_truth_vals  Truth-level population matrix to weight by; see PROmodel::get_counts.
     * @return Matrix (n_phys_bins, 7); see the class description for the column layout.
     */
    Eigen::MatrixXf get_counts(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs,
                               const Eigen::MatrixXf &N_truth_vals) const override;
};

/**
 * @brief 3+1 model with visible decay in the flavour basis (model 2, g_e coupling).
 * @details Parameters: (dmsq, |U_e4|^2, |U_mu4|^2 [log10 space], g_e [linear]).
 * Probability-type components: 0 = no-osc, 1/2/3 = P_mumu / P_mue / P_ee.  Decay
 * daughters land in the SAME subchannel at lower truth-E bins (columns 2 and 3 only;
 * numu disappearance sees no decay contribution), so no paired decay subchannels and
 * no bin offsets are needed.
 */
class PRO3p1_decay_vis_model2 : public PROmodel {
public:
    /// Truth-level MC populations; see PRO3p1_decay_vis_model1::N_truth for details.
    Eigen::MatrixXf N_truth;

    /**
     * @brief Construct the model 2 visible-decay model.
     * @param prop          MC event store; used to build H_combined and N_truth.
     * @param parameter_map Map from physics variable name to variable index in PROpeller.
     *                      Must contain the keys "L" and "E".
     */
    PRO3p1_decay_vis_model2(const PROpeller &prop, const std::map<std::string,int> &parameter_map);

    const Eigen::MatrixXf& get_N_truth() const override { return N_truth; }

    /**
     * @brief Enforce |U_e4|^2 + |U_mu4|^2 < 1 and g_e >= 0.
     * @param v  Physics parameter vector in the fitter's internal space.
     * @return 1 if the point is physically allowed, 0 otherwise.
     */
    int UnitarityConstraint(const Eigen::VectorXf &v);

    bool uses_get_counts() const override { return true; }

    /// Effective per-bin probabilities; see PRO3p1_decay_vis_model1::get_probs.
    Eigen::MatrixXf get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const override;

    /**
     * @brief Compute oscillated + decay-migrated event counts on the flat (L, E) grid.
     * @param phys          Physics parameters (dmsq, Ue4^2, Um4^2 in log10; g_e linear).
     * @param var_arrs      {L array, E array}, each of length n_phys_bins (flat grid midpoints).
     * @param N_truth_vals  Truth-level population matrix to weight by; see PROmodel::get_counts.
     * @return Matrix (n_phys_bins, 4); see the class description for the column layout.
     */
    Eigen::MatrixXf get_counts(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs,
                               const Eigen::MatrixXf &N_truth_vals) const override;
};

}

#endif
