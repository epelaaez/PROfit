/**
 * @file PROmodel.h
 * @brief Physics oscillation model interface and concrete implementations for PROfit.
 * @author PROfit Collaboration
 *
 * @details Defines the abstract PROmodel base class together with the following
 * concrete models:
 *   - NullModel      — no oscillation; all events receive probability 1.
 *   - PROnumudis     — 3+1 sterile-neutrino nu_mu disappearance in the short-baseline
 *                      approximation, parameterised by (Delta m^2, sin^2 2theta_mumu).
 *   - PROnumudisTEST — two-variable (L, E) version of PROnumudis for validation.
 *
 * Each model exposes a get_probs() virtual method that returns oscillation
 * probabilities on the physics grid, and pre-builds H_combined matrices for
 * fast GEMV-based spectrum filling in FillSpectra.
 */
#ifndef PROMODEL_H
#define PROMODEL_H

#include "PROconfig.h"
#include "PROpeller.h"
#include "NuFastLBL.h"

#include <Eigen/Eigen>

#include <array>
#include <cstdlib>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

namespace PROfit {

/**
 * @brief Abstract base class representing a physics model for neutrino oscillation probability.
 * @details A PROmodel encapsulates:
 *   - the set of physics parameters (names, bounds, defaults),
 *   - the mapping from analysis variables to a flat physics grid,
 *   - per-event histogram matrices H_combined used for fast spectrum filling, and
 *   - the virtual get_probs() interface for computing oscillation probabilities.
 *
 * Derived classes implement specific oscillation hypotheses.  After construction,
 * build_hists_and_combined() must be called to pre-build the internal matrices from
 * the MC event store (PROpeller).
 */
class PROmodel {
public:
    size_t nparams; ///< Number of physics parameters for this model.
    /// Indices of the physics (grid) variables used by this model within the PROpeller variable list.
    /// For a 1-variable model this is e.g. {L/E_index}; for 2-variable {L_index, E_index}.
    /// n_phys_bins = product of their bin counts.
    std::vector<int> ivars;
    long int n_phys_bins = 0; ///< Total number of flat physics-grid points (product of ivar bin counts).
    std::vector<std::string> param_names;        ///< Internal parameter names used in the fitter.
    std::vector<std::string> pretty_param_names; ///< LaTeX-formatted parameter names for plots.
    std::vector<std::string> pretty_param_units; ///< Unit strings for plots (e.g., "eV^{2}").
    Eigen::VectorXf lb;          ///< Lower bounds for physics parameters in the fitter's internal space.
    Eigen::VectorXf ub;          ///< Upper bounds for physics parameters in the fitter's internal space.
    Eigen::VectorXf default_val; ///< Default (starting) values for physics parameters.
    /// Per-probability-type functions: model_functions[m](phys, x) returns the oscillation weight
    /// for physics parameters @p phys and kinematic variable value @p x.
    std::vector<std::function<float(const Eigen::VectorXf&, float)>> model_functions;
    std::function<int(const Eigen::VectorXf&)> model_constraint; ///< Optional parameter constraint function.
    /// Pre-binned histograms: hists[v][m] has shape (n_reco_v, n_phys_bins).
    /// v = reco variable index, m = probability-type index.  Transposed for cache efficiency.
    std::vector<std::vector<Eigen::MatrixXf>> hists;
    /// Combined histograms H_combined[v] = horizontal concatenation of hists[v][0..J-1],
    /// shape (n_reco_v, n_phys_bins * J).  Enables a single GEMV in FillSpectra.
    std::vector<Eigen::MatrixXf> H_combined;
    /// Per-ivar bin counts; phys_grid_sizes[k] = number of bins for ivars[k].
    std::vector<size_t> phys_grid_sizes;

    std::vector<size_t> prob_types; ///< Probability-type indices, matching model_functions indices.

    std::vector<bool> is_log10; ///< True for each parameter stored in log10 space; false for linear.

    /// Trivial models (e.g., NullModel) have no physics dependence: probabilities are identically 1.
    /// When true, FillSpectra skips var_arrs / get_probs / GEMV and reads the spectrum directly from
    /// H_combined. Also implies an empty `ivars` so events are binned per reco variable independently
    /// (no cross-variable validity coupling through a placeholder physics-grid variable).
    bool is_trivial = false;

    /**
     * @brief Build hists and H_combined from PROpeller event data.
     * @details Must be called after ivars and model_functions are set.  Iterates over all
     * events in @p prop, distributes them onto the flat physics grid, and constructs the
     * concatenated H_combined matrices used by FillSpectra.
     * @param prop                 The MC event store.
     * @param filter_by_model_rule If true (default), each event is placed in the histogram
     *                             matrix corresponding to its model_rule; if false, all events
     *                             go into component 0 (appropriate for NullModel).
     */
    void build_hists_and_combined(const PROpeller &prop, bool filter_by_model_rule = true) {
        // Compute flat physics grid size and per-ivar bin counts
        std::vector<size_t> ivar_sizes(ivars.size());
        n_phys_bins = 1;
        for(size_t k = 0; k < ivars.size(); ++k) {
            ivar_sizes[k] = prop.variable_midbin[ivars[k]].size();
            n_phys_bins *= (long int)ivar_sizes[k];
        }

        size_t nvar = prop.variable_mc_stat_err.size();
        size_t J    = model_functions.empty() ? prob_types.size() : model_functions.size();
        hists.resize(nvar);
        H_combined.resize(nvar);

        for(size_t v = 0; v < nvar; ++v) {
            size_t n_reco_v = prop.variable_mc_stat_err[v].size();
            hists[v].clear();
            for(size_t m = 0; m < J; ++m) {
                hists[v].emplace_back(Eigen::MatrixXf::Zero(n_reco_v, n_phys_bins));
                Eigen::MatrixXf &h = hists[v].back();
                for(size_t i = 0; i < prop.NEvent(); ++i) {
                    if(filter_by_model_rule && prop.model_rule[i] != (int)m) continue;
                    // Compute row-major flat index over ivars
                    long int flat_phys = 0;
                    bool valid = true;
                    for(size_t k = 0; k < ivars.size(); ++k) {
                        int tbin = prop.VariableBinIndex(ivars[k], i);
                        if(tbin < 0) { valid = false; break; }
                        flat_phys = flat_phys * (long int)ivar_sizes[k] + tbin;
                    }
                    if(!valid) continue;
                    int rbin = prop.VariableBinIndex(v, i);
                    if(rbin < 0) continue;
                    h(rbin, flat_phys) += prop.added_weights[i];
                }
            }
            // Build H_combined[v] = [hists[v][0] | hists[v][1] | ... | hists[v][J-1]]
            // shape: (n_reco_v, n_phys_bins * J)
            H_combined[v].resize(n_reco_v, n_phys_bins * J);
            for(size_t m = 0; m < J; ++m)
                H_combined[v].block(0, m * n_phys_bins, n_reco_v, n_phys_bins) = hists[v][m];
        }

        // Store per-ivar bin counts for downstream use (e.g. decay redistribution).
        phys_grid_sizes.assign(ivar_sizes.begin(), ivar_sizes.end());
    }

    /**
     * @brief Compute the truth-level MC population matrix N_truth(flat, j) = total added_weight
     *   of model_rule-j events at physics bin flat, regardless of reco.  Shape (n_phys_bins, J).
     * @details Only visible-decay models need this (their migration sum reads truth-level
     *   counts at bins other than the destination), so it is not built by default. Decay
     *   model constructors call this explicitly and store the result on their own N_truth
     *   member. Must be called after ivars, prob_types/model_functions, and
     *   build_hists_and_combined have been set up.
     */
    Eigen::MatrixXf compute_N_truth(const PROpeller &prop, bool filter_by_model_rule = true) const {
        size_t J = model_functions.empty() ? prob_types.size() : model_functions.size();
        std::vector<size_t> ivar_sizes(ivars.size());
        for(size_t k = 0; k < ivars.size(); ++k)
            ivar_sizes[k] = prop.variable_midbin[ivars[k]].size();

        Eigen::MatrixXf N = Eigen::MatrixXf::Zero(n_phys_bins, J);
        for(size_t i = 0; i < prop.NEvent(); ++i) {
            int j = filter_by_model_rule ? prop.model_rule[i] : 0;
            if(j < 0 || (size_t)j >= J) continue;
            long int flat_phys = 0;
            bool valid = true;
            for(size_t k = 0; k < ivars.size(); ++k) {
                int tbin = prop.VariableBinIndex(ivars[k], i);
                if(tbin < 0) { valid = false; break; }
                flat_phys = flat_phys * (long int)ivar_sizes[k] + tbin;
            }
            if(!valid) continue;
            N(flat_phys, j) += prop.added_weights[i];
        }
        return N;
    }

    /**
     * @brief Compute oscillation probabilities for all physics-grid points and all probability types.
     * @details Default implementation evaluates each model_function independently for every
     * grid point.  Derived classes may override for a vectorised, faster computation.
     * @param phys      Physics parameter vector in the fitter's internal space (log10 where applicable).
     * @param var_arrs  var_arrs[k] contains the value of ivars[k] for each flat grid point
     *                  (length = n_phys_bins).
     * @return Matrix of shape (n_phys_bins, n_prob_types) where each element is the oscillation
     *         weight for that grid point and probability type.
     */
    virtual Eigen::MatrixXf get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const {
        const auto &le_arr = var_arrs[0];
        Eigen::MatrixXf probs(le_arr.size(), prob_types.size());
        for(size_t i = 0; i < le_arr.size(); ++i) {
            for(size_t j = 0; j < prob_types.size(); ++j) {
                probs(i, j) = model_functions[j](phys, le_arr[i]);
            }
        }
        return probs;
    }

    /**
     * @brief Compute oscillated event counts for all physics-grid points and probability types.
     * @details For models with non-local effects (e.g. visible decay energy redistribution),
     *   this is the authoritative output; FillSpectra converts counts back to effective
     *   probabilities via counts / N_truth before the H_combined multiplication.
     * @param phys          Physics parameter vector in the fitter's internal space.
     * @param var_arrs      var_arrs[k] contains the value of ivars[k] for each flat grid point.
     * @param N_truth_vals  Truth-level event counts matrix of shape (n_phys_bins, J) to use
     *                      in the calculation. Callers pass either the model's own N_truth
     *                      member (for the baseline spectrum) or a pre-reweighted copy
     *                      (for pre-migration flux systematics: multiply each row by the
     *                      per-truth-E flux weight). Keeping this as an explicit argument —
     *                      rather than always reading this->N_truth — lets FillSpectra apply
     *                      a flux reweight at the parent energy before migration, without
     *                      threading the weight through every override in the class hierarchy.
     * @return Matrix of shape (n_phys_bins, J) of oscillated event counts.
     * Only called when uses_get_counts() returns true (the visible-decay models).
     */
    virtual Eigen::MatrixXf get_counts(const Eigen::VectorXf &, const std::vector<std::vector<float>> &,
                                       const Eigen::MatrixXf & /*N_truth_vals*/) const {
        return {};
    }

    /** @brief Whether FillSpectra should use get_counts() (with an explicit N_truth argument)
     *  rather than get_probs() to compute the spectrum. Returns true only for visible-decay
     *  models, where the flat-physics-grid N_truth enters non-locally via the migration sum. */
    virtual bool uses_get_counts() const { return false; }

    /** @brief Returns the truth-level event count matrix N_truth(flat, j), required by decay
     *  models for the migration sum and its counts -> probs normalization.
     *  Default: empty (non-decay models do not store N_truth). Decay models override to
     *  return their own stored matrix. Only meaningful when uses_get_counts() is true. */
    virtual const Eigen::MatrixXf& get_N_truth() const {
        static const Eigen::MatrixXf empty;
        return empty;
    }

    std::unordered_map<std::string, size_t> param_name_to_index; ///< Fast lookup: parameter name -> index in param_names.

    /**
     * @brief Populate param_name_to_index from the current param_names vector.
     * @details Must be called after param_names is finalised in the derived constructor.
     */
    inline void build_param_index() {
        param_name_to_index.clear();
        for(size_t i = 0; i < param_names.size(); ++i){
            param_name_to_index[param_names[i]] = i;
        }
    }

    /**
     * @brief Convert a named parameter from log10 to linear space if required.
     * @param param_name  The parameter name as listed in param_names.
     * @param value       The parameter value in the fitter's internal space.
     * @return The value in linear space: 10^value if is_log10[i] is true, otherwise value unchanged.
     */
    inline float maybe_convert_log(const std::string &param_name, float value) const {
        auto it = param_name_to_index.find(param_name);
        if(it == param_name_to_index.end()){
            log<LOG_ERROR>(L"%1% || Parameter name '%2%' not found in this model. Terminating.") % __func__ % param_name.c_str();
            exit(EXIT_FAILURE);
        }
        size_t idx = it->second;
        return is_log10[idx] ? std::pow(10.0f, value) : value;
    }

    virtual ~PROmodel(){}

};

/**
 * @brief Trivial "no oscillation" model — all events receive oscillation probability 1.
 * @details Used as a central-value baseline and for systematic-only fits where oscillation
 * is not being tested.  All events are placed into a single histogram component regardless
 * of their model_rule.
 */
class NullModel : public PROmodel {
public:
    /**
     * @brief Construct the NullModel from an MC event store.
     * @param prop  The PROpeller containing MC events; used only to build hists.
     */
    NullModel(const PROpeller &prop) {
        nparams = 0;
        // Empty ivars: no placeholder physics-grid variable. Each reco variable is binned
        // independently, so events out-of-range in one variable don't get dropped from another.
        ivars = {};
        is_trivial = true;
        model_functions.push_back([](const Eigen::VectorXf &, float){ return 1.0f; });
        prob_types = {0};

        build_hists_and_combined(prop, /*filter_by_model_rule=*/false);
        is_log10.clear();
    }
};

/**
 * @brief 3+1 sterile-neutrino nu_mu disappearance model in the short-baseline approximation.
 * @details Parameterises the two-flavour-like nu_mu survival probability as:
 *   P(nu_mu -> nu_mu) = 1 - sin^2(2*theta_mumu) * sin^2(1.267 * Delta m^2 * L/E)
 * where Delta m^2 is in eV^2 and L/E is in km/GeV.  Both parameters are stored in log10 space.
 * The model uses a single physics variable (L/E) identified by the "L/E" entry in parameter_map.
 */
class PROnumudis : public PROmodel {
public:
    /**
     * @brief Construct the PROnumudis model.
     * @param prop          MC event store; used to build H_combined.
     * @param parameter_map Map from physics variable name (e.g., "L/E") to variable index in PROpeller.
     *                      Must contain the key "L/E".
     */
    PROnumudis(const PROpeller &prop,const std::map<std::string,int> &parameter_map) {
        // model_functions is the non-unified version, these are optional
        // these get combined into one get_probs function in the constructor, but we can override this for faster computation
        model_functions.push_back([this]([[maybe_unused]] const Eigen::VectorXf &v, float) {(void)this; return 1.0;});
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pmumu(v(0),v(1),le);});
        prob_types = {0, 1};

        if(parameter_map.find("L/E") == parameter_map.end()) {
            log<LOG_ERROR>(L"%1%, %2% || Missing expected parameter: 'L/E'.Make sure its in your model section of XML. ") % __func__ % __LINE__;
            throw std::runtime_error("Missing parameter: L/E");
        }
        ivars = {parameter_map.at("L/E")};

        build_hists_and_combined(prop);
        nparams = 2;
        param_names = {"dmsq", "sinsq2thmm"}; 
        pretty_param_names = {"#Deltam^{2}", "sin^{2}2#theta_{#mu#mu}"}; 
        pretty_param_units = {"eV^{2}", ""}; 
        is_log10 = {true, true};
        build_param_index();
        lb = Eigen::VectorXf(2);
        ub = Eigen::VectorXf(2);
        default_val = Eigen::VectorXf(2);
        lb << -2, -std::numeric_limits<float>::infinity();
        ub << 2, 0;
        default_val << -2, -10;

    };

    /**
     * @brief Compute the 3+1 nu_mu survival probability in the short-baseline approximation.
     * @details P(nu_mu -> nu_mu) = 1 - sin^2(2*theta_mumu) * sin^2(1.267 * Delta m^2 * L/E).
     * @param dmsq          Mass splitting Delta m^2 in eV^2 (may be in log10 space; converted internally).
     * @param sinsq2thmumu  sin^2(2 theta_mumu) (may be in log10 space; converted internally).
     * @param le            L/E ratio in km/GeV.
     * @return Survival probability in [0, 1].
     */
    float Pmumu(float dmsq, float sinsq2thmumu, float le) const{
        dmsq = maybe_convert_log("dmsq", dmsq);
        sinsq2thmumu = maybe_convert_log("sinsq2thmm", sinsq2thmumu);

        if(sinsq2thmumu > 1) {
            //log<LOG_ERROR>(L"%1% || sinsq2thmumu is %2% which is greater than 1. Setting to 1.")     % __func__ % sinsq2thmumu;
            sinsq2thmumu = 1;
        }
        if(sinsq2thmumu < 0) {
            log<LOG_ERROR>(L"%1% || sinsq2thmumu is %2% which is less than 0. Setting to 0.")
                % __func__ % sinsq2thmumu;
            sinsq2thmumu = 0;
        }

        float sinterm = std::sin(1.266932679f*dmsq*(le));
        float prob    = 1.0f - (sinsq2thmumu*sinterm*sinterm);

        if(prob<0.0 || prob >1.0 ){;//|| std::isnan(prob)){
            log<LOG_ERROR>(L"%1% || Your probability %2% is outside the bounds of math."
                           L"dmsq = %3%, sinsq2thmumu = %4%, L/E = %5%")
                % __func__ % prob % dmsq % sinsq2thmumu % le;
            log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
            exit(EXIT_FAILURE);
        }

        return prob;
    }

    Eigen::MatrixXf get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const override {
        const auto &le_arr = var_arrs[0];
        //log<LOG_ERROR>(L"%1% || Using unified, optimized get_probs function for model") % __func__;
        // Precompute physics parameters once
        float dmsq = maybe_convert_log("dmsq", phys(0));
        float sinsq2thmumu = maybe_convert_log("sinsq2thmm", phys(1));

        float freq = 1.266932679f * dmsq;

        if(sinsq2thmumu > 1) sinsq2thmumu = 1;
        if(sinsq2thmumu < 0) sinsq2thmumu = 0;

        Eigen::MatrixXf probs(le_arr.size(), model_functions.size());

        for(size_t i = 0; i < le_arr.size(); ++i) {

            // no oscillation
            probs(i, 0) = 1.0f;

            // P_mumu
            float sinterm = std::sin(freq * le_arr[i]);
            probs(i, 1) = 1.0f - (sinsq2thmumu * sinterm * sinterm);
        }

        return probs;
    }
};

/**
 * @brief Two-variable test version of PROnumudis operating on separate L and E variables.
 * @details Takes separate "L" and "E" variables from parameter_map and builds H_combined on the
 * 2D (L x E) physics grid.  get_probs() computes L/E internally, so the physics is identical
 * to PROnumudis.  The two models should produce identical spectra and can be used to validate
 * the multi-variable code path against the standard single L/E variable approach.
 */
class PROnumudisTEST : public PROmodel {
public:
    /**
     * @brief Construct the two-variable PROnumudisTEST model.
     * @param prop          MC event store; used to build H_combined on the (L, E) grid.
     * @param parameter_map Map from physics variable name to variable index in PROpeller.
     *                      Must contain both "L" and "E".
     */
    PROnumudisTEST(const PROpeller &prop, const std::map<std::string,int> &parameter_map) {
        prob_types = {0, 1};
        model_functions.push_back([this]([[maybe_unused]] const Eigen::VectorXf &v, float) {(void)this; return 1.0f;});
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pmumu(v(0),v(1),le);});

        if(parameter_map.find("L") == parameter_map.end()) {
            log<LOG_ERROR>(L"%1%, %2% || PROnumudisTEST: Missing expected parameter: 'L'.") % __func__ % __LINE__;
            throw std::runtime_error("Missing parameter: L");
        }
        if(parameter_map.find("E") == parameter_map.end()) {
            log<LOG_ERROR>(L"%1%, %2% || PROnumudisTEST: Missing expected parameter: 'E'.") % __func__ % __LINE__;
            throw std::runtime_error("Missing parameter: E");
        }
        // ivars[0] = L variable index, ivars[1] = E variable index.
        // build_hists_and_combined will make the flat grid L x E.
        ivars = {parameter_map.at("L"), parameter_map.at("E")};

        build_hists_and_combined(prop);

        nparams = 2;
        param_names = {"dmsq", "sinsq2thmm"};
        pretty_param_names = {"#Deltam^{2}", "sin^{2}2#theta_{#mu#mu}"};
        pretty_param_units = {"eV^{2}", ""};
        is_log10 = {true, true};
        build_param_index();
        lb = Eigen::VectorXf(2);
        ub = Eigen::VectorXf(2);
        default_val = Eigen::VectorXf(2);
        lb << -2, -std::numeric_limits<float>::infinity();
        ub << 2, 0;
        default_val << -2, -10;
    };

    /**
     * @brief Compute the 3+1 nu_mu survival probability (identical physics to PROnumudis::Pmumu).
     * @param dmsq          Mass splitting Delta m^2 in eV^2 (may be in log10 space; converted internally).
     * @param sinsq2thmumu  sin^2(2 theta_mumu) (may be in log10 space; converted internally).
     * @param le            L/E ratio in km/GeV, computed internally from var_arrs[0]=L / var_arrs[1]=E.
     * @return Survival probability in [0, 1].
     */
    float Pmumu(float dmsq, float sinsq2thmumu, float le) const {
        dmsq         = maybe_convert_log("dmsq",       dmsq);
        sinsq2thmumu = maybe_convert_log("sinsq2thmm", sinsq2thmumu);
        if(sinsq2thmumu > 1) sinsq2thmumu = 1;
        if(sinsq2thmumu < 0) sinsq2thmumu = 0;
        float sinterm = std::sin(1.266932679f * dmsq * le);
        float prob    = 1.0f - (sinsq2thmumu * sinterm * sinterm);
        if(prob < 0.0f || prob > 1.0f) {
            log<LOG_ERROR>(L"%1% || Probability %2% outside [0,1]. dmsq=%3%, sinsq2thmumu=%4%, L/E=%5%")
                % __func__ % prob % dmsq % sinsq2thmumu % le;
            exit(EXIT_FAILURE);
        }
        return prob;
    }

    /**
     * @brief Compute oscillation probabilities on the 2D (L×E) grid.
     * @details var_arrs[0] = L values, var_arrs[1] = E values, each of length n_L * n_E (flat row-major order).
     *          L/E is computed internally for each grid point, so the result is physically identical to PROnumudis::get_probs.
     * @param phys     Physics parameter vector: (log10(dmsq), log10(sinsq2thmm)).
     * @param var_arrs 2-element vector: {L array [km], E array [GeV]}, each of size n_phys_bins.
     * @return Matrix of shape (n_phys_bins, 2): column 0 = 1 (no-osc), column 1 = P(nu_mu -> nu_mu).
     */
    Eigen::MatrixXf get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const override {
        float dmsq         = maybe_convert_log("dmsq",       phys(0));
        float sinsq2thmumu = maybe_convert_log("sinsq2thmm", phys(1));
        if(sinsq2thmumu > 1) sinsq2thmumu = 1;
        if(sinsq2thmumu < 0) sinsq2thmumu = 0;

        float freq = 1.266932679f * dmsq;
        const size_t n_flat = var_arrs[0].size(); // = n_L * n_E
        Eigen::MatrixXf probs(n_flat, 2);

        for(size_t i = 0; i < n_flat; ++i) {
            float L = var_arrs[0][i];
            float E = var_arrs[1][i];
            // Guard against zero energy — same convention as L/E variable (out-of-range events
            // get bin index -1 in PROpeller so they never enter H; but be safe here too).
            float le = (E > 0.0f) ? L / E : 0.0f;
            probs(i, 0) = 1.0f;
            float sinterm = std::sin(freq * le);
            probs(i, 1) = 1.0f - (sinsq2thmumu * sinterm * sinterm);
        }
        return probs;
    }
};

/**
 * @brief 3+1 sterile-neutrino nu_mu → nu_e appearance model in the short-baseline approximation.
 * @details Parameterises the two-flavour-like nu_e appearance probability as:
 *   P(nu_mu -> nu_e) = sin^2(2*theta_mue) * sin^2(1.267 * Delta m^2 * L/E)
 * where Delta m^2 is in eV^2 and L/E is in km/GeV.  Both parameters are stored in log10 space.
 * The model uses a single physics variable (L/E) identified by the "L/E" entry in parameter_map.
 */
class PROnueapp : public PROmodel {
public:
    /**
     * @brief Construct the PROnueapp model.
     * @param prop          MC event store; used to build H_combined.
     * @param parameter_map Map from physics variable name (e.g., "L/E") to variable index in PROpeller.
     *                      Must contain the key "L/E".
     */
    PROnueapp(const PROpeller &prop,const std::map<std::string,int> &parameter_map) {
        model_functions.push_back([this]([[maybe_unused]] const Eigen::VectorXf &v, float) {(void)this; return 1.0;});
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pmue(v(0),v(1),le);});
        prob_types = {0, 1};
        if(parameter_map.find("L/E") == parameter_map.end()) {
            log<LOG_ERROR>(L"%1%, %2% || Missing expected parameter: 'L/E'. Make sure its in your model section of XML.") % __func__ % __LINE__;
            throw std::runtime_error("Missing parameter: L/E");
        }
        ivars = {parameter_map.at("L/E")};

        build_hists_and_combined(prop);
         nparams = 2;
        param_names = {"dmsq", "sinsq2thme"}; 
        pretty_param_names = {"#Deltam^{2}", "sin^{2}2#theta_{#mue}"}; 
        pretty_param_units = {"eV^{2}", ""}; 
        is_log10 = {true, true};
        build_param_index();
        lb = Eigen::VectorXf(2);
        ub = Eigen::VectorXf(2);
        default_val = Eigen::VectorXf(2);
        lb << -2, -10; //-std::numeric_limits<float>::infinity();
        ub << 2, 0;
        //default_val << -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity();
        default_val << -2, -10; //std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity();
    
        log<LOG_INFO>(L"%1% || setting up a model nueapp, with  %2% params.")     % __func__ % nparams;
        for(size_t i=0; i< nparams;i++){
            log<LOG_INFO>(L"%1% || Param %2% is %3% with lower bound/upper bound of %4%/%5% and default %6%")     % __func__ % i % param_names[i].c_str() % lb[i] % ub[i] % default_val[i];
        }

    };

    /**
     * @brief Compute the 3+1 nu_mu → nu_e appearance probability in the short-baseline approximation.
     * @details P(nu_mu -> nu_e) = sin^2(2*theta_mue) * sin^2(1.267 * Delta m^2 * L/E).
     * @param dmsq         Mass splitting Delta m^2 in eV^2 (may be in log10 space; converted internally).
     * @param sinsq2thmue  sin^2(2 theta_mue) (may be in log10 space; converted internally).
     * @param le           L/E ratio in km/GeV.
     * @return Appearance probability in [0, 1].
     */
    float Pmue(float dmsq, float sinsq2thmue, float le) const{
        dmsq = maybe_convert_log("dmsq", dmsq);
        sinsq2thmue = maybe_convert_log("sinsq2thme", sinsq2thmue);

        if(sinsq2thmue > 1) {
            //log<LOG_ERROR>(L"%1% || sinsq2thmue is %2% which is greater than 1. Setting to 1.")  % __func__ % sinsq2thmue;
            sinsq2thmue = 1;
        }
        if(sinsq2thmue < 0) {
            log<LOG_ERROR>(L"%1% || sinsq2thmue is %2% which is less than 0. Setting to 0.")
                % __func__ % sinsq2thmue;
            sinsq2thmue = 0;
        }

        float sinterm = std::sin(1.266932679f*dmsq*(le));
        float prob    = sinsq2thmue*sinterm*sinterm;

        if(prob<0.0 || prob >1.0){
            log<LOG_ERROR>(L"%1% || Your probability %2% is outside the bounds of math."
                           L"dmsq = %3%, sinsq2thmue = %4%, L/E = %5%")
                % __func__ % prob % dmsq % sinsq2thmue % le;
            log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
            exit(EXIT_FAILURE);
        }

        return prob;
    }

    Eigen::MatrixXf get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const override {
        const auto &le_arr = var_arrs[0];
        //log<LOG_ERROR>(L"%1% || Using unified, optimized get_probs function for model") % __func__;
        // Precompute physics parameters once
        float dmsq = maybe_convert_log("dmsq", phys(0));
        float sinsq2thmue = maybe_convert_log("sinsq2thme", phys(1));

        float freq = 1.266932679f * dmsq;

        if(sinsq2thmue > 1) sinsq2thmue = 1;
        if(sinsq2thmue < 0) sinsq2thmue = 0;

        Eigen::MatrixXf probs(le_arr.size(), model_functions.size());

        for(size_t i = 0; i < le_arr.size(); ++i) {

            // no oscillation
            probs(i, 0) = 1.0f;

            // P_mumu
            float sinterm = std::sin(freq * le_arr[i]);
            probs(i, 1) = (sinsq2thmue * sinterm * sinterm);
        }

        return probs;
    }


};

/**
 * @brief 3+1 sterile-neutrino nu_e disappearance model in the short-baseline approximation.
 * @details Parameterises the two-flavour-like nu_e survival probability as:
 *   P(nu_e -> nu_e) = 1 - sin^2(2*theta_ee) * sin^2(1.267 * Delta m^2 * L/E)
 * where Delta m^2 is in eV^2 and L/E is in km/GeV.  Both parameters are stored in log10 space.
 * The model uses a single physics variable (L/E) identified by the "L/E" entry in parameter_map.
 */
class PROnuedis : public PROmodel {
public:
    /**
     * @brief Construct the PROnuedis model.
     * @param prop          MC event store; used to build H_combined.
     * @param parameter_map Map from physics variable name (e.g., "L/E") to variable index in PROpeller.
     *                      Must contain the key "L/E".
     */
    PROnuedis(const PROpeller &prop,const std::map<std::string,int> &parameter_map) {
        model_functions.push_back([this]([[maybe_unused]] const Eigen::VectorXf &v, float) {(void)this; return 1.0;});
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pee(v(0),v(1),le);});
        prob_types = {0, 1};

        if(parameter_map.find("L/E") == parameter_map.end()) {
            log<LOG_ERROR>(L"%1%, %2% || Missing expected parameter: 'L/E'.Make sure its in your model section of XML. ") % __func__ % __LINE__;
            throw std::runtime_error("Missing parameter: L/E");
        }
        ivars = {parameter_map.at("L/E")};

        build_hists_and_combined(prop);
        nparams = 2;
        param_names = {"dmsq", "sinsq2thee"}; 
        pretty_param_names = {"#Deltam^{2}", "sin^{2}2#theta_{ee}"}; 
        pretty_param_units = {"eV^{2}", ""}; 
        is_log10 = {true, true};
        build_param_index();
        lb = Eigen::VectorXf(2);
        ub = Eigen::VectorXf(2);
        default_val = Eigen::VectorXf(2);
        lb << -2, -std::numeric_limits<float>::infinity();
        ub << 2, 0;
        default_val << -2, -10;

    };

    /**
     * @brief Compute the 3+1 nu_e survival probability in the short-baseline approximation.
     * @details P(nu_e -> nu_e) = 1 - sin^2(2*theta_ee) * sin^2(1.267 * Delta m^2 * L/E).
     * @param dmsq        Mass splitting Delta m^2 in eV^2 (may be in log10 space; converted internally).
     * @param sinsq2thee  sin^2(2 theta_ee) (may be in log10 space; converted internally).
     * @param le          L/E ratio in km/GeV.
     * @return Survival probability in [0, 1].
     */
    float Pee(float dmsq, float sinsq2thee, float le) const{
        dmsq = maybe_convert_log("dmsq", dmsq);
        sinsq2thee = maybe_convert_log("sinsq2thee", sinsq2thee);

        if(sinsq2thee > 1) {
            //log<LOG_ERROR>(L"%1% || sinsq2thee is %2% which is greater than 1. Setting to 1.")     % __func__ % sinsq2thee;
            sinsq2thee = 1;
        }
        if(sinsq2thee < 0) {
            log<LOG_ERROR>(L"%1% || sinsq2thee is %2% which is less than 0. Setting to 0.")
                % __func__ % sinsq2thee;
            sinsq2thee = 0;
        }

        float sinterm = std::sin(1.266932679f*dmsq*(le));
        float prob    = 1.0f - (sinsq2thee*sinterm*sinterm);

        if(prob<0.0 || prob >1.0 ){;//|| std::isnan(prob)){
            log<LOG_ERROR>(L"%1% || Your probability %2% is outside the bounds of math."
                           L"dmsq = %3%, sinsq2thee = %4%, L/E = %5%")
                % __func__ % prob % dmsq % sinsq2thee % le;
            log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
            exit(EXIT_FAILURE);
        }

        return prob;
    }

    Eigen::MatrixXf get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const override {
        const auto &le_arr = var_arrs[0];
        //log<LOG_ERROR>(L"%1% || Using unified, optimized get_probs function for model") % __func__;
        // Precompute physics parameters once
        float dmsq = maybe_convert_log("dmsq", phys(0));
        float sinsq2thee = maybe_convert_log("sinsq2thee", phys(1));

        float freq = 1.266932679f * dmsq;

        if(sinsq2thee > 1) sinsq2thee = 1;
        if(sinsq2thee < 0) sinsq2thee = 0;

        Eigen::MatrixXf probs(le_arr.size(), model_functions.size());

        for(size_t i = 0; i < le_arr.size(); ++i) {

            // no oscillation
            probs(i, 0) = 1.0f;

            // P_mumu
            float sinterm = std::sin(freq * le_arr[i]);
            probs(i, 1) = 1.0f-(sinsq2thee * sinterm * sinterm);
        }

        return probs;
    }


};


/**
 * @brief Full 3+1 sterile-neutrino model parameterised by |U_e4|^2 and |U_mu4|^2.
 * @details Provides all three oscillation channels in the short-baseline approximation:
 *   - P(nu_mu -> nu_mu) = 1 - 4*|U_mu4|^2*(1 - |U_mu4|^2) * sin^2(1.267 * Dm^2 * L/E)
 *   - P(nu_mu -> nu_e)  = 4*|U_e4|^2*|U_mu4|^2          * sin^2(1.267 * Dm^2 * L/E)
 *   - P(nu_e  -> nu_e)  = 1 - 4*|U_e4|^2*(1 - |U_e4|^2) * sin^2(1.267 * Dm^2 * L/E)
 * A unitarity constraint |U_e4|^2 + |U_mu4|^2 < 1 is enforced via model_constraint.
 * All three parameters (dmsq, Ue4^2, Um4^2) are stored in log10 space.
 */
class PRO3p1 : public PROmodel {
public:
    /**
     * @brief Construct the PRO3p1 model.
     * @param prop          MC event store; used to build H_combined.
     * @param parameter_map Map from physics variable name (e.g., "L/E") to variable index in PROpeller.
     *                      Must contain the key "L/E".
     */
    PRO3p1(const PROpeller &prop, const std::map<std::string,int> &parameter_map) {

        model_functions.push_back([this]([[maybe_unused]] const Eigen::VectorXf &v, float) {(void)this; return 1.0; });
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pmumu(v(0),v(1),v(2),le); });
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pmue(v(0),v(1),v(2),le); });
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pee(v(0),v(1),v(2),le); });
        prob_types = {0, 1, 2, 3};
        if(parameter_map.find("L/E") == parameter_map.end()) {
            log<LOG_ERROR>(L"%1%, %2% || Missing expected parameter: 'L/E'. Make sure its in your model section of XML.") % __func__ % __LINE__;
            throw std::runtime_error("Missing parameter: L/E");
        }
        ivars = {parameter_map.at("L/E")};

        //constraints
        model_constraint = [this](const Eigen::VectorXf &v){return this->UnitarityConstraint(v);};


        build_hists_and_combined(prop);


        nparams = 3;
        param_names = {"dmsq", "Ue4^2", "Um4^2"}; 
        pretty_param_names = {"#Deltam^{2}", "|U_{e4}|^{2}", "|U_{#mu4}|^{2}"}; 
        pretty_param_units = {"eV^{2}", "",""}; 
        is_log10 = {true, true, true};
        build_param_index();
        lb = Eigen::VectorXf(3);
        ub = Eigen::VectorXf(3);
        default_val = Eigen::VectorXf(3);
        lb << -2, -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity();
        ub << 2, -1e-4, -1e-4;
        //default_val << -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity();
        default_val << -2, -8, -8;
    };

    /**
     * @brief Enforce the unitarity constraint |U_e4|^2 + |U_mu4|^2 < 1.
     * @param v  Physics parameter vector in log10 space.
     * @return 1 if the point is physically allowed, 0 otherwise.
     */
    int UnitarityConstraint(const Eigen::VectorXf &v){
        const float Ue4sq = maybe_convert_log("Ue4^2", v(param_name_to_index.at("Ue4^2")));
        const float Um4sq = maybe_convert_log("Um4^2", v(param_name_to_index.at("Um4^2")));
        return   ((Ue4sq+Um4sq)<1 ? 1 : 0);
    }

    /**
     * @brief Compute the 3+1 nu_mu → nu_e appearance probability.
     * @param dmsq   Mass splitting Delta m^2 in eV^2 (log10 space; converted internally).
     * @param Ue4sq  |U_e4|^2 (log10 space; converted internally).
     * @param Um4sq  |U_mu4|^2 (log10 space; converted internally).
     * @param le     L/E ratio in km/GeV.
     * @return Appearance probability in [0, 1].
     */
    float Pmue(float dmsq, float Ue4sq, float Um4sq, float le) const{
        dmsq =  maybe_convert_log("dmsq", dmsq);
        Ue4sq = maybe_convert_log("Ue4^2", Ue4sq);
        Um4sq = maybe_convert_log("Um4^2", Um4sq);

        if(Ue4sq > 1) {
            log<LOG_ERROR>(L"%1% || Ue4sq is %2% which is greater than 1. Setting to 1.") 
                % __func__ % Ue4sq;
            Ue4sq = 1;
        }
        if(Ue4sq < 0) {
            log<LOG_ERROR>(L"%1% || Ue4sq is %2% which is less than 0. Setting to 0.")
                % __func__ % Ue4sq;
            Ue4sq = 0;
        }
        if(Um4sq > 1) {
            log<LOG_ERROR>(L"%1% || Um4sq is %2% which is greater than 1. Setting to 1.") 
                % __func__ % Um4sq;
            Um4sq = 1;
        }
        if(Um4sq < 0) {
            log<LOG_ERROR>(L"%1% || Um4sq is %2% which is less than 0. Setting to 0.")
                % __func__ % Um4sq;
            Um4sq = 0;
        }

        float sinterm = std::sin(1.266932679f*dmsq*(le));
        float prob    = 4.0f*Ue4sq*Um4sq*sinterm*sinterm;

        if(prob<0.0 || prob >1.0){
            log<LOG_ERROR>(L"%1% || Your probability %2% is outside the bounds of math."
                           L"dmsq = %3%, Ue4sq = %4%, Um4sq = %5%, L/E = %6%")
                % __func__ % prob % dmsq % Ue4sq % Um4sq % le;
            log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
            exit(EXIT_FAILURE);
        }

        return prob;
    }

    /**
     * @brief Compute the 3+1 nu_mu survival probability.
     * @param dmsq   Mass splitting Delta m^2 in eV^2 (log10 space; converted internally).
     * @param Ue4sq  |U_e4|^2 — unused in this channel.
     * @param Um4sq  |U_mu4|^2 (log10 space; converted internally).
     * @param le     L/E ratio in km/GeV.
     * @return Survival probability in [0, 1].
     */
    float Pmumu(float dmsq, [[maybe_unused]]float Ue4sq, float Um4sq, float le) const{
        dmsq = maybe_convert_log("dmsq", dmsq);
        Um4sq = maybe_convert_log("Um4^2", Um4sq);

        if(Um4sq > 1) {
            log<LOG_ERROR>(L"%1% || Um4sq is %2% which is greater than 1. Setting to 1.")
                % __func__ % Um4sq;
            Um4sq = 1;
        }
        if(Um4sq < 0) {
            log<LOG_ERROR>(L"%1% || Um4sq is %2% which is less than 0. Setting to 0.")
                % __func__ % Um4sq;
            Um4sq = 0;
        }

        float sinterm = std::sin(1.266932679f*dmsq*(le));
        float prob    = 1.0f - 4.0f*Um4sq*(1.0f-Um4sq)*sinterm*sinterm;

        if(prob<0.0 || prob >1.0){
            log<LOG_ERROR>(L"%1% || Your probability %2% is outside the bounds of math. dmsq = %3%, Um4sq = %4%, L/E = %5%") % __func__ % prob % dmsq % Um4sq % le;
            log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
            exit(EXIT_FAILURE);
        }

        return prob;
    }

    /**
     * @brief Compute the 3+1 nu_e survival probability.
     * @param dmsq   Mass splitting Delta m^2 in eV^2 (log10 space; converted internally).
     * @param Ue4sq  |U_e4|^2 (log10 space; converted internally).
     * @param Um4sq  |U_mu4|^2 — unused in this channel.
     * @param le     L/E ratio in km/GeV.
     * @return Survival probability in [0, 1].
     */
    float Pee(float dmsq, float Ue4sq, [[maybe_unused]]float Um4sq, float le) const{
        dmsq = maybe_convert_log("dmsq", dmsq);
        Ue4sq = maybe_convert_log("Ue4^2", Ue4sq);

        if(Ue4sq > 1) {
            log<LOG_ERROR>(L"%1% || Ue4sq is %2% which is greater than 1. Setting to 1.")
                % __func__ % Ue4sq;
            Ue4sq = 1;
        }
        if(Ue4sq < 0) {
            log<LOG_ERROR>(L"%1% || Ue4sq is %2% which is less than 0. Setting to 0.")
                % __func__ % Ue4sq;
            Ue4sq = 0;
        }

        float sinterm = std::sin(1.266932679f*dmsq*(le));
        float prob    = 1.0f - 4.0f*Ue4sq*(1.0f-Ue4sq)*sinterm*sinterm;

        if(prob<0.0 || prob >1.0){
            log<LOG_ERROR>(L"%1% || Your probability %2% is outside the bounds of math. dmsq = %3%, Ue4sq = %4%, L/E = %5%") % __func__ % prob % dmsq % Ue4sq % le;
            log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
            exit(EXIT_FAILURE);
        }

        return prob;
    }

    /**
     * @brief Compute all oscillation probabilities for the PRO3p1 model at each L/E grid point.
     * @param phys     Physics vector: (log10(dmsq), log10(Ue4sq), log10(Um4sq)).
     * @param var_arrs 1-element vector containing the L/E array [km/GeV] of length n_phys_bins.
     * @return Matrix (n_phys_bins, 4): columns = {1, P_mumu, P_mue, P_ee}.
     */
    Eigen::MatrixXf get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const override {
        const auto &le_arr = var_arrs[0];
        //log<LOG_ERROR>(L"%1% || Using unified, optimized get_probs function for model") % __func__;

        // Precompute physics parameters once
        float dmsq = maybe_convert_log("dmsq", phys(0));
        float Ue4sq = maybe_convert_log("Ue4^2", phys(1));
        float Um4sq = maybe_convert_log("Um4^2", phys(2));

        Eigen::MatrixXf probs(le_arr.size(), model_functions.size());

        for(size_t i = 0; i < le_arr.size(); ++i) {

            float sinterm = std::sin(1.266932679f*dmsq*(le_arr[i]));

            // no oscillation
            probs(i, 0) = 1.0f;


            // P_mumu
            probs(i, 1) = 1.0f - 4.0f*Um4sq*(1.0f-Um4sq)*sinterm*sinterm;

            // P_mue
            probs(i, 2) = 4.0f*Ue4sq*Um4sq*sinterm*sinterm;

            // P_ee
            probs(i, 3) = 1.0f - 4.0f*Ue4sq*(1.0f-Ue4sq)*sinterm*sinterm;

        }

        return probs;
    }
};


/**
 * @brief 3+1 sterile-neutrino model parameterised by mixing angles sin^2(2*theta_14) and sin^2(theta_24).
 * @details An alternative to PRO3p1 that uses the angle parameterisation instead of |U|^2 elements directly.
 * Provides nu_mu disappearance, nu_mu → nu_e appearance, and nu_e disappearance channels.
 * All parameters (dmsq, sinsq2th14, sinsqth24) are stored in log10 space.
 */
class PRO3p1_angles : public PROmodel {
public:
    /**
     * @brief Construct the PRO3p1_angles model.
     * @param prop          MC event store; used to build H_combined.
     * @param parameter_map Map from physics variable name (e.g., "L/E") to variable index in PROpeller.
     *                      Must contain the key "L/E".
     */
    PRO3p1_angles(const PROpeller &prop, const std::map<std::string,int> &parameter_map) {

        model_functions.push_back([this]([[maybe_unused]] const Eigen::VectorXf &v, float) {(void)this; return 1.0; });
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pmumu(v(0),v(1),v(2),le); });
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pmue(v(0),v(1),v(2),le); });
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pee(v(0),v(1),v(2),le); });
        prob_types = {0, 1, 2, 3};
        if(parameter_map.find("L/E") == parameter_map.end()) {
            log<LOG_ERROR>(L"%1%, %2% || Missing expected parameter: 'L/E'. Make sure its in your model section of XML.") % __func__ % __LINE__;
            throw std::runtime_error("Missing parameter: L/E");
        }
        ivars = {parameter_map.at("L/E")};

        //constraints
        model_constraint = [this](const Eigen::VectorXf &v){return this->UnitarityConstraint(v);};


        build_hists_and_combined(prop);


        nparams = 3;
        param_names = {"dmsq", "sinsq2th14", "sinsqth24"}; 
        pretty_param_names = {"#Deltam^{2}", "sin^{2}2#theta_{14}", "sin^{2}#theta_{24}"}; 
        pretty_param_units = {"eV^{2}", "",""}; 
        is_log10 = {true, true, true};
        build_param_index();
        lb = Eigen::VectorXf(3);
        ub = Eigen::VectorXf(3);
        default_val = Eigen::VectorXf(3);
        lb << -2, -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity();
        ub << 2, -1e-4, -1e-4;
        //default_val << -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity();
        default_val << -2, -8, -8;
    };

    int UnitarityConstraint(const Eigen::VectorXf &){
        return   1;
    }

    float Pmue(float dmsq, float sinsq2th14, float sinsqth24, float le) const{
        dmsq =  maybe_convert_log("dmsq", dmsq);
        float s214 = maybe_convert_log("sinsq2th14", sinsq2th14);
        float s24 = maybe_convert_log("sinsqth24", sinsqth24);

        float sinterm = std::sin(1.266932679f*dmsq*(le));
        float prob    = s214*s24*sinterm*sinterm;

        if(prob<0.0 || prob >1.0){
            log<LOG_ERROR>(L"%1% || Your probability %2% is outside the bounds of math."
                           L"dmsq = %3%, s214 = %4%, s24 = %5%, L/E = %6%")
                % __func__ % prob % dmsq % s214 % s24 % le;
            log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
            exit(EXIT_FAILURE);
        }

        return prob;
    }

    float Pmumu(float dmsq, float sinsq2th14, float sinsqth24, float le) const{
        dmsq =  maybe_convert_log("dmsq", dmsq);
        float s214 = maybe_convert_log("sinsq2th14", sinsq2th14);
        float s24 = maybe_convert_log("sinsqth24", sinsqth24);
        float c14 = (1.0+sqrt(1.0-s214))/2.0f;

        float sinterm = std::sin(1.266932679f*dmsq*(le));
        float prob    = 1.0f - (c14*s24*(1.0f-c14*s24))*sinterm*sinterm;


        if(prob<0.0 || prob >1.0){
            log<LOG_ERROR>(L"%1% || Your probability %2% is outside the bounds of math."
                           L"dmsq = %3%, s214 = %4%, s24 = %5%, L/E = %6%")
                % __func__ % prob % dmsq % s214 % s24 % le;
            log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
            exit(EXIT_FAILURE);
        }
        return prob;
    }

    float Pee(float dmsq, float sinsq2th14, [[maybe_unused]]float sinsqth24, float le) const{
        dmsq = maybe_convert_log("dmsq", dmsq);
        float s214 = maybe_convert_log("sinsq2th14", sinsq2th14);

        float sinterm = std::sin(1.266932679f*dmsq*(le));
        float prob    = 1.0f - s214*sinterm*sinterm;

        if(prob<0.0 || prob >1.0){
            log<LOG_ERROR>(L"%1% || Your probability %2% is outside the bounds of math."
                           L"dmsq = %3%, s214 = %4%,  L/E = %5%")
                % __func__ % prob % dmsq % s214   % le;
            log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
            exit(EXIT_FAILURE);
        }
        return prob;
    }

    Eigen::MatrixXf get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const override {
        const auto &le_arr = var_arrs[0];
        //log<LOG_ERROR>(L"%1% || Using unified, optimized get_probs function for model") % __func__;

        // Precompute physics parameters once
        float dmsq = maybe_convert_log("dmsq", phys(0));
        float s214 = maybe_convert_log("sinsq2th14", phys(1));
        float s24 = maybe_convert_log("sinsqth24", phys(2));
        float c14 = (1.0+sqrt(1.0-s214))/2.0f;

        Eigen::MatrixXf probs(le_arr.size(), model_functions.size());

        for(size_t i = 0; i < le_arr.size(); ++i) {
            
            float sinterm = std::sin(1.266932679f*dmsq*(le_arr[i]));

            // no oscillation
            probs(i, 0) = 1.0f;

            // P_mumu
            probs(i, 1) = 1.0f - c14*s24*(1.0f-c14*s24)*sinterm*sinterm;

            // P_mue
            probs(i, 2) = s214*s24*sinterm*sinterm;

            // P_ee
            probs(i, 3) = 1.0f - s214*sinterm*sinterm;

        }

        return probs;
    }
};

/**
 * @brief 3+1 model variant 3A: parameterised by sin^2(2*theta_ee) and sin^2(theta_24).
 * @details This variant expresses all three channels — nu_mu disappearance, nu_mu → nu_e appearance,
 * and nu_e disappearance — in terms of the nue-sector mixing angle sin^2(2*theta_ee) = 4*|U_e4|^2*(1-|U_e4|^2)
 * and the muon-sector angle sin^2(theta_24).  Parameters (dmsq, sinsq2thee, sinsqth24) are in log10 space.
 */
class PRO3p1_3A : public PROmodel {
public:
    /**
     * @brief Construct the PRO3p1_3A model.
     * @param prop          MC event store; used to build H_combined.
     * @param parameter_map Map from physics variable name (e.g., "L/E") to variable index in PROpeller.
     *                      Must contain the key "L/E".
     */
    PRO3p1_3A(const PROpeller &prop, const std::map<std::string,int> &parameter_map) {

        model_functions.push_back([this]([[maybe_unused]] const Eigen::VectorXf &v, float) {(void)this; return 1.0; });
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pmumu(v(0),v(1),v(2),le); });
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pmue(v(0),v(1),v(2),le); });
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pee(v(0),v(1),v(2),le); });
        prob_types = {0, 1, 2, 3};
        if(parameter_map.find("L/E") == parameter_map.end()) {
            log<LOG_ERROR>(L"%1%, %2% || Missing expected parameter: 'L/E'. Make sure its in your model section of XML.") % __func__ % __LINE__;
            throw std::runtime_error("Missing parameter: L/E");
        }
        ivars = {parameter_map.at("L/E")};

        //constraints
        model_constraint = [](const Eigen::VectorXf &){return 1;};


         build_hists_and_combined(prop);


        nparams = 3;
        param_names = {"dmsq", "sinsq2thee", "sinsqth24"}; 
        pretty_param_names = {"#Deltam^{2}", "sin^{2}2#theta_{ee}", "sin^{2}#theta_{24}"}; 
        pretty_param_units = {"eV^{2}", "",""};
        is_log10 = {true, true, true};
        build_param_index();
        lb = Eigen::VectorXf(3);
        ub = Eigen::VectorXf(3);
        default_val = Eigen::VectorXf(3);
        lb << -2, -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity();
        ub << 2, -1e-3, -1e-3;
        //default_val << -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity();
        default_val << -2, -8, -8;
    };

    int UnitarityConstraint(const Eigen::VectorXf &){
        return   1;
    }

    float Pmue(float dmsq, float sinsq2thee, [[maybe_unused]]float sinsqth24, float le) const{
        dmsq = maybe_convert_log("dmsq",dmsq);
        float sinsq2thmue = maybe_convert_log("sinsqth24",sinsqth24)*maybe_convert_log("sinsq2thee", sinsq2thee);
        
        if(sinsq2thmue > 1) {
            log<LOG_ERROR>(L"%1% || sinsq2thmue is %2% which is greater than 1. Setting to 1.")
                % __func__ % sinsq2thmue;
            sinsq2thmue = 1;
        }
        if(sinsq2thmue < 0) {
            log<LOG_ERROR>(L"%1% || sinsq2thmue is %2% which is less than 0. Setting to 0.")
                % __func__ % sinsq2thmue;
            sinsq2thmue = 0;
        }

        float sinterm = std::sin(1.266932679f*dmsq*(le));
        float prob    = sinsq2thmue*sinterm*sinterm;

        if(prob<0.0 || prob >1.0){
            log<LOG_ERROR>(L"%1% || Your probability %2% is outside the bounds of math."
                           L"dmsq = %3%, sinsq2thmue = %4%, L/E = %5%")
                % __func__ % prob % dmsq % sinsq2thmue % le;
            log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
            exit(EXIT_FAILURE);
        }

        return prob;
    }

    float Pmumu(float dmsq, float sinsq2thee, [[maybe_unused]]float sinsqth24, float le) const{
        float Um4sq = maybe_convert_log("sinsqth24",sinsqth24)/2.0*(1.0+sqrt(1- maybe_convert_log("sinsq2thee", sinsq2thee)));
        dmsq =maybe_convert_log("dmsq",dmsq);
        

        float sinterm = std::sin(1.266932679f*dmsq*(le));
        float prob    = 1.0f - 4.0f*Um4sq*(1.0f-Um4sq)*sinterm*sinterm;

        if(prob<0.0 || prob >1.0){
            log<LOG_ERROR>(L"%1% || Your probability %2% is outside the bounds of math. dmsq = %3%, Um4sq = %4%, sinsq2thee = %5%, sinsqth24 = %6%, L/E = %7%") % __func__ % prob % dmsq % Um4sq % sinsq2thee % sinsqth24 % le;
            log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
            exit(EXIT_FAILURE);
        }

        return prob;
    }

    float Pee(float dmsq, float sinsq2thee, [[maybe_unused]]float sinsqth24, float le) const{

        dmsq =maybe_convert_log("dmsq",dmsq);
        sinsq2thee =maybe_convert_log("sinsq2thee",sinsq2thee);

        float sinterm = std::sin(1.266932679f*dmsq*(le));
        float prob    = 1.0f - sinsq2thee*sinterm*sinterm;

        if(prob<0.0 || prob >1.0){
            log<LOG_ERROR>(L"%1% || Your probability %2% is outside the bounds of math. dmsq = %3%,sinsq2thee = %4%, sinsqth24 = %5%, L/E = %6%") % __func__ % prob % dmsq % sinsq2thee % sinsqth24 % le;
            log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
            exit(EXIT_FAILURE);
        }

        return prob;
    }

    Eigen::MatrixXf get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const override {
        const auto &le_arr = var_arrs[0];
        //log<LOG_ERROR>(L"%1% || Using unified, optimized get_probs function for model") % __func__;

        // Precompute physics parameters once
        float dmsq = maybe_convert_log("dmsq", phys(0));
        float sinsq2thee = maybe_convert_log("sinsq2thee", phys(1));
        float sinsqth24 = maybe_convert_log("sinsqth24", phys(2));
        
        float Um4sq = sinsqth24/2.0*(1.0+sqrt(1.0f- sinsq2thee));
        float sinsq2thmue = sinsqth24*sinsq2thee;

        Eigen::MatrixXf probs(le_arr.size(), model_functions.size());

        for(size_t i = 0; i < le_arr.size(); ++i) {
            
            float sinterm = std::sin(1.266932679f*dmsq*(le_arr[i]));

            // no oscillation
            probs(i, 0) = 1.0f;

            // P_mumu
            probs(i, 1) =  1.0f - 4.0f*Um4sq*(1.0f-Um4sq)*sinterm*sinterm;

            // P_mue
            probs(i, 2) = sinsq2thmue*sinterm*sinterm;

            // P_ee
            probs(i, 3) = 1.0f - sinsq2thee*sinterm*sinterm;

        }

        return probs;
    }

};

/**
 * @brief 3+1 model variant 3B: parameterised by sin^2(2*theta_mumu) and an asymmetry ratio sB.
 * @details Variant B uses the nu_mu disappearance amplitude directly (sin^2(2*theta_mumu)) together
 * with a ratio parameter sB that controls how much of the nu_mu mixing leaks into the nu_e sector.
 * The nu_e sector mixing is derived as Ue4^2 = sB * (1 - Um4^2) where
 * Um4^2 = (1 - sqrt(1 - sin^2(2*theta_mumu))) / 2.
 * A unitarity constraint is enforced via model_constraint.
 * Parameters (dmsq, sinsq2thmumu, sB) are all in log10 space.
 */
class PRO3p1_3B : public PROmodel {
public:
    /**
     * @brief Construct the PRO3p1_3B model.
     * @param prop          MC event store; used to build H_combined.
     * @param parameter_map Map from physics variable name (e.g., "L/E") to variable index in PROpeller.
     *                      Must contain the key "L/E".
     */
    PRO3p1_3B(const PROpeller &prop,
              const std::map<std::string,int> &parameter_map) {

        // -----------------------------------------
        // 1) Model functions:
        //    0: constant (unosc / NC-like)
        //    1: Pmumu
        //    2: Pmue
        //    3: Pee
        // -----------------------------------------
        model_functions.push_back(
            [this]([[maybe_unused]] const Eigen::VectorXf &v, float) {
                (void)this;
                return 1.0f;
            }
        );
        model_functions.push_back(
            [this](const Eigen::VectorXf &v, float le) {
                return this->Pmumu(v(0), v(1), v(2), le);
            }
        );
        model_functions.push_back(
            [this](const Eigen::VectorXf &v, float le) {
                return this->Pmue(v(0), v(1), v(2), le);
            }
        );
        model_functions.push_back(
            [this](const Eigen::VectorXf &v, float le) {
                return this->Pee(v(0), v(1), v(2), le);
            }
        );
        prob_types = {0, 1, 2, 3};

        // -----------------------------------------
        // 2) L/E variable index
        // -----------------------------------------
        if (parameter_map.find("L/E") == parameter_map.end()) {
            log<LOG_ERROR>(
                L"%1%, %2% || Missing expected parameter: 'L/E'. "
                L"Make sure it's in your model section of the XML."
            ) % __func__ % __LINE__;
            throw std::runtime_error("Missing parameter: L/E");
        }
        ivars = {parameter_map.at("L/E")};

        // -----------------------------------------
        // 3) Build histograms for each model component
        // -----------------------------------------
        build_hists_and_combined(prop);

        // -----------------------------------------
        // 4) Parameters and bounds
        // v(0) = dmsq_log      = log10(Δm²_41)
        // v(1) = s2mumu_log    = log10(sin²2θ_μμ)
        // v(2) = sB  (can be thought of as sinsqth24prime = log10(sin²θ_24′))
        // -----------------------------------------
        nparams = 3;
        param_names        = {"dmsq", "sinsq2thmumu", "sB"};
        pretty_param_names = {"#Deltam^{2}", "sin^{2}2#theta_{#mu#mu}", "sB"};
        pretty_param_units = {"eV^{2}", "",""};
        is_log10         = {true, true, true};
        build_param_index();

        lb          = Eigen::VectorXf(3);
        ub          = Eigen::VectorXf(3);
        default_val = Eigen::VectorXf(3);
        lb << -2.0f, -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity();
        ub <<  2.0f, -1e-3f, -1e-3 ;
        // Some reasonable defaults
        default_val << -2.0f, -8.0f, -8.0f;

        // -----------------------------------------
        // 5) Unitarity / physicality constraint
        // -----------------------------------------
        model_constraint = [this](const Eigen::VectorXf &v) {
            return this->UnitarityConstraint(v);
        };
    }

   

    // ---------------------------------------------
    // Unitarity / physicality constraint:
    // 0 ≤ sB ≤ 1 and Ue4² + Uμ4² < 1
    // ---------------------------------------------
    int UnitarityConstraint(const Eigen::VectorXf &v) {
        float sinsq2thmumu = std::pow(10.0f, v(1));  // sin²2θμμ
        float sB = std::pow(10.0f, v(2));                   // ratio parameter

        float rad = 1.0f - sinsq2thmumu;
        float Um4sq = (1.0f - std::sqrt(rad)) / 2.0f;
        float Ue4sq = sB * (1.0f - Um4sq);     // from definition of sB

        return Um4sq + Ue4sq < 0.999 ? 1 :0;  // allowed
    }

    // ---------------------------------------------
    // νμ → νμ disappearance
    // ---------------------------------------------
    float Pmumu(float dmsq, float sinsq2thmumu, [[maybe_unused]] float sinsqth24prime, float le) const {
        dmsq   = std::pow(10.0f, dmsq);
        sinsq2thmumu = std::pow(10.0f, sinsq2thmumu);


        float sinterm = std::sin(1.266932679f * dmsq * le);
        float prob    = 1.0f - sinsq2thmumu * sinterm * sinterm;

        if (prob < 0.0f || prob > 1.0f) {
            log<LOG_ERROR>(
                L"%1% || Pmumu %2% outside [0,1]. "
                L"dmsq = %3% L/E = %5%"
            ) % __func__ % prob % dmsq % le;
            log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
            exit(EXIT_FAILURE);
        }

        return prob;
    }

    // ---------------------------------------------
    // νμ → νe appearance
    // ---------------------------------------------
    float Pmue(float dmsq, float sinsq2thmumu, float sB, float le) const {
        dmsq   = std::pow(10.0f, dmsq);
        sinsq2thmumu = std::pow(10.0f, sinsq2thmumu);

        float sinterm = std::sin(1.266932679f * dmsq * le);
        float prob    = sB*sinsq2thmumu * sinterm * sinterm;

        if (prob < 0.0f || prob > 1.0f) {
            log<LOG_ERROR>(
                L"%1% || Pmue %2% outside [0,1]. "
                L"dmsq = %3%, sinsq2thmuu = %4%, sB = %5%, L/E = %6%"
            ) % __func__ % prob % dmsq % sinsq2thmumu % sB % le;
            log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
            exit(EXIT_FAILURE);
        }

        return prob;
    }

    // ---------------------------------------------
    // νe → νe disappearance
    // ---------------------------------------------
    float Pee(float dmsq, float sinsq2thmumu, float sB, float le) const {
        dmsq   = std::pow(10.0f, dmsq);
        sinsq2thmumu = std::pow(10.0f, sinsq2thmumu);

        float rad = 1.0f - sinsq2thmumu;
        float Um4sq = (1.0f - std::sqrt(rad)) / 2.0f;
        float Ue4sq = sB * (1.0f - Um4sq);  


        float sinterm = std::sin(1.266932679f * dmsq * le);
        float prob    = 1.0f - 4.0f * Ue4sq * (1.0f - Ue4sq) * sinterm * sinterm;

        if (prob < 0.0f || prob > 1.0f) {
            log<LOG_ERROR>(
                L"%1% || Pee %2% outside [0,1]. "
                L"dmsq = %3%, Ue4sq = %4%, L/E = %5%"
            ) % __func__ % prob % dmsq % Ue4sq % le;
            log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
            exit(EXIT_FAILURE);
        }

        return prob;
    }
    
    Eigen::MatrixXf get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const override {
        const auto &le_arr = var_arrs[0];
        //log<LOG_ERROR>(L"%1% || Using unified, optimized get_probs function for model") % __func__;

        // Precompute physics parameters once
        float dmsq = maybe_convert_log("dmsq", phys(0));
        float sinsq2thmumu = maybe_convert_log("sinsq2thmumu", phys(1));
        float sB = maybe_convert_log("sB", phys(2));
        float Ue4sq = (sB/2.0)*(1.0+sqrt(1.0f-sinsq2thmumu));

        Eigen::MatrixXf probs(le_arr.size(), model_functions.size());

        for(size_t i = 0; i < le_arr.size(); ++i) {
            
            float sinterm = std::sin(1.266932679f*dmsq*(le_arr[i]));

            // no oscillation
            probs(i, 0) = 1.0f;

            // P_mumu
            probs(i, 1) =  1.0f - sinsq2thmumu * sinterm * sinterm;

            // P_mue
            probs(i, 2) =  sB*sinsq2thmumu* sinterm * sinterm;


            // P_ee
            probs(i, 3) = 1.0f-4.0f*(1-Ue4sq)*Ue4sq *sinterm*sinterm;

        }

        return probs;
    }


};


/**
 * @brief 3+1 model variant 3C: parameterised by sin^2(2*theta_mue) and an asymmetry angle xi.
 * @details Variant C uses the appearance amplitude sin^2(2*theta_mue) directly together with a
 * hyperbolic asymmetry parameter xi that sets the relative size of the nu_e and nu_mu mixing elements:
 *   |U_mu4|^2 = exp(-xi) * sqrt(sin^2(2*theta_mue)) / 2
 *   |U_e4|^2  = exp(+xi) * sqrt(sin^2(2*theta_mue)) / 2
 * A unitarity constraint sqrt(sin^2(2*theta_mue)) * cosh(xi) < 1 is enforced via model_constraint.
 * dmsq and sinsq2thmue are in log10 space; xi is linear.
 */
class PRO3p1_3C : public PROmodel {
public:
    /**
     * @brief Construct the PRO3p1_3C model.
     * @param prop          MC event store; used to build H_combined.
     * @param parameter_map Map from physics variable name (e.g., "L/E") to variable index in PROpeller.
     *                      Must contain the key "L/E".
     */
    PRO3p1_3C(const PROpeller &prop, const std::map<std::string,int> &parameter_map) {

        model_functions.push_back([this]([[maybe_unused]] const Eigen::VectorXf &v, float) {(void)this; return 1.0; });
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pmumu(v(0),v(1),v(2),le); });
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pmue(v(0),v(1),v(2),le); });
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pee(v(0),v(1),v(2),le); });
        prob_types = {0, 1, 2, 3};
        if(parameter_map.find("L/E") == parameter_map.end()) {
            log<LOG_ERROR>(L"%1%, %2% || Missing expected parameter: 'L/E'. Make sure its in your model section of XML.") % __func__ % __LINE__;
            throw std::runtime_error("Missing parameter: L/E");
        }
        ivars = {parameter_map.at("L/E")};

        //constraints
        model_constraint = [this](const Eigen::VectorXf &v){return this->UnitarityConstraint(v);};


        build_hists_and_combined(prop);


        nparams = 3;
        param_names = {"dmsq", "sinsq2thmue", "xi"}; 
        pretty_param_names = {"#Deltam^{2}", "sin^{2}2#theta_{#mue}", "#xi"}; 
        pretty_param_units = {"eV^{2}", "", ""};
        is_log10 = {true, true, false};
        build_param_index();
        lb = Eigen::VectorXf(3);
        ub = Eigen::VectorXf(3);
        default_val = Eigen::VectorXf(3);
        lb << -2, -std::numeric_limits<float>::infinity(), -10;
        ub << 2, -1e-3, 10;
        //default_val << -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity();
        default_val << -2, -8, 0;
    };

    int UnitarityConstraint(const Eigen::VectorXf &v){
        const float sinsq2thmue = maybe_convert_log("sinsq2thmue", v(param_name_to_index.at("sinsq2thmue")));
        const float xi = maybe_convert_log("xi", v(param_name_to_index.at("xi")));
        return   (std::sqrt(sinsq2thmue)*std::cosh(xi)<0.999 ? 1 : 0);      
    }

    float Pmue(float dmsq, float sinsq2thmue, [[maybe_unused]]float xi, float le) const{
        dmsq = maybe_convert_log("dmsq", dmsq);
        sinsq2thmue = maybe_convert_log("sinsq2thmue", sinsq2thmue);
        
        if(sinsq2thmue > 1) {
            log<LOG_ERROR>(L"%1% || sinsq2thmue is %2% which is greater than 1. Setting to 1.")
                % __func__ % sinsq2thmue;
            sinsq2thmue = 1;
        }
        if(sinsq2thmue < 0) {
            log<LOG_ERROR>(L"%1% || sinsq2thmue is %2% which is less than 0. Setting to 0.")
                % __func__ % sinsq2thmue;
            sinsq2thmue = 0;
        }

        float sinterm = std::sin(1.266932679f*dmsq*(le));
        float prob    = sinsq2thmue*sinterm*sinterm;

        if(prob<0.0 || prob >1.0){
            log<LOG_ERROR>(L"%1% || Your probability %2% is outside the bounds of math."
                           L"dmsq = %3%, sinsq2thmue = %4%, L/E = %5%")
                % __func__ % prob % dmsq % sinsq2thmue % le;
            log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
            exit(EXIT_FAILURE);
        }

        return prob;
    }

    float Pmumu(float dmsq, float sinsq2thmue, float xi, float le) const{
        dmsq = maybe_convert_log("dmsq", dmsq);
        sinsq2thmue = maybe_convert_log("sinsq2thmue", sinsq2thmue);

        float Um4sq=(std::exp(-xi) * std::sqrt(sinsq2thmue)) / 2.0;

        float sinterm = std::sin(1.266932679f*dmsq*(le));
        float prob    = 1.0f - 4.0f*Um4sq*(1.0f-Um4sq)*sinterm*sinterm;

        if(prob<0.0 || prob >1.0){
            log<LOG_ERROR>(L"%1% || Your probability %2% is outside the bounds of math. dmsq = %3%, Um4sq = %4%, sinsq2thmue = %5%, xi = %6%, L/E = %7%") % __func__ % prob % dmsq % Um4sq % sinsq2thmue % xi % le;
            log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
            exit(EXIT_FAILURE);
        }

        return prob;
    }

    float Pee(float dmsq, float sinsq2thmue, float xi, float le) const{
        dmsq = maybe_convert_log("dmsq", dmsq);
        sinsq2thmue = maybe_convert_log("sinsq2thmue", sinsq2thmue);

        float Ue4sq=(std::exp(xi) * std::sqrt(sinsq2thmue)) / 2.0;

        float sinterm = std::sin(1.266932679f*dmsq*(le));
        float prob    = 1.0f - 4.0f*Ue4sq*(1.0f-Ue4sq)*sinterm*sinterm;

        if(prob<0.0 || prob >1.0){
            log<LOG_ERROR>(L"%1% || Your probability %2% is outside the bounds of math. dmsq = %3%, Ue4sq = %4%, sinsq2thmue = %5%, xi = %6%, L/E = %7%") % __func__ % prob % dmsq % Ue4sq % sinsq2thmue % xi % le;
            log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
            exit(EXIT_FAILURE);
        }

        return prob;
    }
    
    Eigen::MatrixXf get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const override {
        const auto &le_arr = var_arrs[0];
        //log<LOG_ERROR>(L"%1% || Using unified, optimized get_probs function for model") % __func__;

        // Precompute physics parameters once
        float dmsq = maybe_convert_log("dmsq", phys(0));
        float sinsq2thmue = maybe_convert_log("sinsq2thmue", phys(1));
        float xi = maybe_convert_log("xi", phys(2));

        float sqrtsin = std::sqrt(sinsq2thmue);
        float Um4sq=(std::exp(-xi) *sqrtsin ) / 2.0;
        float Ue4sq=(std::exp(xi) *sqrtsin) / 2.0;


        Eigen::MatrixXf probs(le_arr.size(), model_functions.size());

        for(size_t i = 0; i < le_arr.size(); ++i) {
            
            float sinterm = std::sin(1.266932679f*dmsq*(le_arr[i]));

            // no oscillation
            probs(i, 0) = 1.0f;

            // P_mumu
            probs(i, 1) =  1.0f - 4.0f*(1-Um4sq)*Um4sq * sinterm * sinterm;

            // P_mue
            probs(i, 2) =  sinsq2thmue* sinterm * sinterm;


            // P_ee
            probs(i, 3) = 1.0f-4.0f*(1-Ue4sq)*Ue4sq *sinterm*sinterm;

        }

        return probs;
    }


};


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
    PRO3p1_decay_invis(const PROpeller &prop, const std::map<std::string,int> &parameter_map) {
        // 3+1+decay to invisible particles, example from IceCube: https://arxiv.org/pdf/2204.00612
        // (invisible means no active or sterile-oscillating-to-active neutrinos after the decay)

        // model_functions is the non-unified version, these are optional
        // these get combined into one get_probs function in the constructor, but we can override this for faster computation
        model_functions.push_back([this]([[maybe_unused]] const Eigen::VectorXf &v, float) {(void)this; return 1.0; });
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pmumu(v(0),v(1),v(2),v(3),le); });
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pmue(v(0),v(1),v(2),v(3),le); });
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pee(v(0),v(1),v(2),v(3),le); });
        prob_types = {0, 1, 2, 3};
        if(parameter_map.find("L/E") == parameter_map.end()) {
            log<LOG_ERROR>(L"%1%, %2% || Missing expected parameter: 'L/E'. Make sure its in your model section of XML.") % __func__ % __LINE__;
            throw std::runtime_error("Missing parameter: L/E");
        }
        ivars = {parameter_map.at("L/E")};

        model_constraint = [this](const Eigen::VectorXf &v){return this->UnitarityConstraint(v);};

        build_hists_and_combined(prop);

        nparams = 4;
        param_names = {"dmsq", "Ue4^2", "Um4^2", "g2"}; 
        pretty_param_names = {"#Deltam^{2}", "|U_{e4}|^{2}", "|U_{#mu4}|^{2}", "g^{2}"}; 
        pretty_param_units = {"eV^{2}", "", "", ""}; 
        is_log10 = {true, true, true, false};
        build_param_index();
        lb = Eigen::VectorXf(4);
        ub = Eigen::VectorXf(4);
        default_val = Eigen::VectorXf(4);
        lb << -2, -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), 0;
        ub << 2, -1e-4, -1e-4, 10;
        default_val << -2, -8, -8, 0;
    };

    int UnitarityConstraint(const Eigen::VectorXf &v){
        // ensures positive g2 in addition to the usual unitarity constraints
        const float Ue4sq = maybe_convert_log("Ue4^2", v(param_name_to_index.at("Ue4^2")));
        const float Um4sq = maybe_convert_log("Um4^2", v(param_name_to_index.at("Um4^2")));
        const float g2 = maybe_convert_log("g2", v(param_name_to_index.at("g2")));
        return   ((Ue4sq+Um4sq)<1 && g2>0 ? 1 : 0);      
    }

    // Equations from Jesse Mendez, slide 5 bottom https://microboone-docdb.fnal.gov/cgi-bin/sso/RetrieveFile?docid=45475&filename=2025-10-31-mendez-sterile-deacy.pdf&version=1
    //
    // Derivation from references:
    // See equation 10 here, written in terms of L_osc and L_dec: https://journals.aps.org/prd/pdf/10.1103/PhysRevD.110.075002
    //     This is the only term in equation 9 if we set the visible decay term to zero (P_dec == 0)
    // Using Delta = 1/4 1/(hbar c) * (m / 1 eV)^2 * ((L / 1 km) / (E / 1 GeV)) = 1/4 m^2 L/E (natural units) = 1.266932679 m^2 L / E (km and GeV units)
    // L_osc = 2 pi E / m^2 (just after equation 10)
    // Simplifying this term: pi L / L_osc = pi L / (2 pi E / m^2) = 1/2 m^2 L/E = 2 Delta
    // From equation 1 of https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.129.151801: tau = 16 pi / (g^2 m)
    // L_dec = (relativistic gamma factor) * c * tau = E/m * tau = E/m * 16 pi / (g^2 m) = 16 pi E / (g^2 m^2)
    // Simplifying this term: L / (2 L_dec) = g^2 m^2 L / (32 pi E) = g^2 / 8 pi * (1/4 m^2 L/E) = g^2 / 8 pi * Delta
    // Final formula: P_ab = delta_ab - 2 delta_ab |U_a4 U_b4| [1 - exp(-g^2 / 8 pi * Delta) cos(2 Delta)]
    //                       + |U_a4 U_b4|^2 [1 - 2 exp(-g^2 / 8 pi * Delta) cos(2 Delta) + exp(-g^2 / 4 pi * Delta)]
    //
    // Another reference for 3+1+invisible decay: Equation 15 of https://journals.aps.org/prd/pdf/10.1103/PhysRevD.97.055017
    //     P_aa = cos^4(theta) + 1/2 exp_term cos(2 Delta) sin^2(2 theta) + exp_term^2 sin^4(theta)
    //          = (1 - sin^2(theta))^2 + 1/2 exp_term cos(2 Delta) (4 cos^2(theta) sin^2(theta)) + exp_term^2 sin^4(theta)
    //          = (1 - 2 sin^2(theta) + sin^4(theta)) + 1/2 exp_term cos(2 Delta) (4 sin^2(theta) - 4 sin^4(theta)) + exp_term^2 sin^4(theta)
    //          = 1 - 2 sin^2(theta) + sin^4(theta) + 2 exp_term cos(2 Delta) sin^2(theta) - 2 exp_term cos(2 Delta) sin^4(theta) + exp_term^2 sin^4(theta)
    //          = 1 - 2 sin^2(theta) [1 - exp_term cos(2 Delta)] + sin^4(theta) [1 - 2 exp_term cos(2 Delta) + exp_term^2]
    // Taking our full equation, looking at just the disappearance case, and substituting |U_a4|^2 = sin^2(theta):
    //     P_aa = 1 - 2 |U_a4|^2 [1 - exp_term cos(2 Delta)] + |U_a4|^4 [1 - 2 exp_term cos(2 Delta) + exp_term^2]
    //          = 1 - 2 sin^2(theta) [1 - exp_term cos(2 Delta)] + sin^4(theta) [1 - 2 exp_term cos(2 Delta) + exp_term^2]
    // This exactly matches the equation above, confirming that the references are consistent.

    float Pmue(float dmsq, float Ue4sq, float Um4sq, float g2, float le) const{
        dmsq = maybe_convert_log("dmsq", dmsq);
        Ue4sq = maybe_convert_log("Ue4^2", Ue4sq);
        Um4sq = maybe_convert_log("Um4^2", Um4sq);
        g2 = maybe_convert_log("g2", g2);

        if (Ue4sq > 1 || Ue4sq < 0 || Um4sq > 1 || Um4sq < 0 || g2 < 0) {
            log<LOG_ERROR>(L"%1% || Parameter(s) out of bounds. Setting to limits. Values: Ue4sq=%2%, Um4sq=%3%, g2=%4%, dmsq=%5%, le=%6%")
                % __func__ % Ue4sq % Um4sq % g2 % dmsq % le;
            if (Ue4sq > 1) Ue4sq = 1;
            if (Ue4sq < 0) Ue4sq = 0;
            if (Um4sq > 1) Um4sq = 1;
            if (Um4sq < 0) Um4sq = 0;
            if (g2 < 0) g2 = 0;
            exit(EXIT_FAILURE);
        }

        float delta = 1.266932679f*dmsq*le;
        float costerm = std::cos(2.0f*delta);
        float expterm = std::exp(-g2*delta/(8.0f*3.14159f));
        float prob    = Ue4sq*Um4sq*(1.0f-2.0f*expterm*costerm + expterm*expterm);
        //exit(0);

        // numerical precision issues can cause small negative probabilities
        if(-1e-6f < prob && prob<0.0f){
            prob = 0.0f;
        }

        if(prob<0.0 || prob >1.0 || std::isnan(prob)){
            log<LOG_ERROR>(L"%1% || Your probability %2% is outside the 0-1 range. dmsq = %3%, Ue4sq = %4%, Um4sq = %5%, g2 = %6%, L/E = %7%") % __func__ % prob % dmsq % Ue4sq % Um4sq % g2 % le;
            log<LOG_ERROR>(L"delta = %1%, costerm = %2%, expterm = %3%") % delta % costerm % expterm;
            log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
            exit(EXIT_FAILURE);
        }

        return prob;
    }

    float Pmumu(float dmsq, [[maybe_unused]]float Ue4sq, float Um4sq, float g2, float le) const{
        dmsq = maybe_convert_log("dmsq", dmsq);
        Um4sq = maybe_convert_log("Um4^2", Um4sq);
        g2 = maybe_convert_log("g2", g2);

        if (Um4sq > 1 || Um4sq < 0 || g2 < 0) {
            log<LOG_ERROR>(L"%1% || Parameter(s) out of bounds. Setting to limits. Values: Um4sq=%2%, g2=%3%, dmsq=%4%, le=%5%")
                % __func__ % Um4sq % g2 % dmsq % le;
            if (Um4sq > 1) Um4sq = 1;
            if (Um4sq < 0) Um4sq = 0;
            if (g2 < 0) g2 = 0;
            exit(EXIT_FAILURE);
        }

        float delta = 1.266932679f*dmsq*le;
        float costerm = std::cos(2.0f*delta);
        float expterm = std::exp(-g2*delta/(8.0f*3.14159f));
        float prob    = 1.0f - 2.0f*Um4sq*(1.0f-expterm*costerm) + Um4sq*Um4sq*(1.0f-2.0f*expterm*costerm + expterm*expterm);

        // numerical precision issues can cause small negative probabilities
        if(-1e-6f < prob && prob<0.0f){
            prob = 0.0f;
        }

        if(prob<0.0 || prob >1.0 || std::isnan(prob)){
            log<LOG_ERROR>(L"%1% || Your probability %2% is the 0-1 range. dmsq = %3%, Um4sq = %4%, g2 = %5%, L/E = %6%") % __func__ % prob % dmsq % Um4sq % g2 % le;
            log<LOG_ERROR>(L"delta = %1%, costerm = %2%, expterm = %3%") % delta % costerm % expterm;
            log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
            exit(EXIT_FAILURE);
        }

        return prob;
    }

    float Pee(float dmsq, float Ue4sq, [[maybe_unused]]float Um4sq, float g2, float le) const{
        dmsq = maybe_convert_log("dmsq", dmsq);
        Ue4sq = maybe_convert_log("Ue4^2", Ue4sq);
        g2 = maybe_convert_log("g2", g2);

        if (Ue4sq > 1 || Ue4sq < 0 || g2 < 0) {
            log<LOG_ERROR>(L"%1% || Parameter(s) out of bounds. Setting to limits. Values: Ue4sq=%2%, g2=%3%, dmsq=%4%, le=%5%")
                % __func__ % Ue4sq % g2 % dmsq % le;
            if (Ue4sq > 1) Ue4sq = 1;
            if (Ue4sq < 0) Ue4sq = 0;
            if (g2 < 0) g2 = 0;
            exit(EXIT_FAILURE);
        }

        float delta = 1.266932679f*dmsq*le;
        float costerm = std::cos(2.0f*delta);
        float expterm = std::exp(-g2*delta/(8.0f*3.14159f));
        float prob    = 1.0f - 2.0f*Ue4sq*(1.0f-expterm*costerm) + Ue4sq*Ue4sq*(1.0f-2.0f*expterm*costerm + expterm*expterm);

        // numerical precision issues can cause small negative probabilities
        if(-1e-6f < prob && prob<0.0f){
            prob = 0.0f;
        }

        if(prob<0.0 || prob >1.0 || std::isnan(prob)){
            log<LOG_ERROR>(L"%1% || Your probability %2% is outside the 0-1 range. dmsq = %3%, Ue4sq = %4%, g2 = %5%, L/E = %6%") % __func__ % prob % dmsq % Ue4sq % g2 % le;
            log<LOG_ERROR>(L"delta = %1%, costerm = %2%, expterm = %3%") % delta % costerm % expterm;
            log<LOG_ERROR>(L"term1 = %1%, term2 = %2%, term3 = %3%") % 1.0f % (-2.0f*Ue4sq*(1.0f-expterm*costerm)) % (Ue4sq*Ue4sq*(1.0f-2.0f*expterm*costerm + expterm*expterm));
            log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
            exit(EXIT_FAILURE);
        }

        return prob;
    }

    Eigen::MatrixXf get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const override {
        const auto &le_arr = var_arrs[0];
        //log<LOG_ERROR>(L"%1% || Using unified, optimized get_probs function for model") % __func__;

        // Precompute physics parameters once
        float dmsq = maybe_convert_log("dmsq", phys(0));
        float Ue4sq = maybe_convert_log("Ue4^2", phys(1));
        float Um4sq = maybe_convert_log("Um4^2", phys(2));
        float g2 = maybe_convert_log("g2", phys(3));

        //log<LOG_ERROR>(L"%1% || dmsq = %2%, Ue4sq = %3%, Um4sq = %4%, g2 = %5%") % __func__ % dmsq % Ue4sq % Um4sq % g2;

        float freq = 1.266932679f * dmsq;

        Eigen::MatrixXf probs(le_arr.size(), model_functions.size());

        for(size_t i = 0; i < le_arr.size(); ++i) {
            
            // no oscillation
            probs(i, 0) = 1.0f;

            float delta = freq*le_arr[i];
            float costerm = std::cos(2.0f*delta);
            float expterm = std::exp(-g2*delta/(8.0f*3.14159f));
            float cos_mult_exp_term = costerm*expterm;
            float osc_term =(1.0f-2.0f*cos_mult_exp_term + expterm*expterm);

            // P_mumu
            probs(i, 1) = 1.0f - 2.0f*Um4sq*(1.0f-cos_mult_exp_term) + Um4sq*Um4sq*osc_term;

            // P_mue
            probs(i, 2) = Ue4sq*Um4sq*osc_term;

            // P_ee
            probs(i, 3) = 1.0f - 2.0f*Ue4sq*(1.0f-cos_mult_exp_term) + Ue4sq*Ue4sq*osc_term;

        }

        return probs;
    }
    
};


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

        PRO3p1_decay_vis_model1(const PROpeller &prop, const std::map<std::string,int> &parameter_map) {
            // 3+1+decay to lower energy neutrinos
            // model 1 from https://journals.aps.org/prd/abstract/10.1103/PhysRevD.110.075002

            if(parameter_map.find("L") == parameter_map.end()) {
                log<LOG_ERROR>(L"%1%, %2% || Missing expected parameter: 'L'. Make sure it's in the model section of XML.") % __func__ % __LINE__;
                throw std::runtime_error("Missing parameter: L");
            }
            if(parameter_map.find("E") == parameter_map.end()) {
                log<LOG_ERROR>(L"%1%, %2% || Missing expected parameter: 'E'. Make sure it's in the model section of XML.") % __func__ % __LINE__;
                throw std::runtime_error("Missing parameter: E");
            }
            ivars = {parameter_map.at("L"), parameter_map.at("E")};

            // Rules 1/2/3 hold the same-bin (parent-bin) survival contribution for
            // P_mumu / P_mue / P_ee.  Rules 4/5/6 hold the visible-decay contribution
            // arriving at lower-E daughter bins from those same flavor channels.
            prob_types = {0, 1, 2, 3, 4, 5, 6};

            model_constraint = [this](const Eigen::VectorXf &v){return this->UnitarityConstraint(v);};

            build_hists_and_combined(prop); // fills hists, H_combined, phys_grid_sizes
            N_truth = compute_N_truth(prop); // save N_truth, so it can be modified by flux-weighting in PROcess.cxx

            // Detect per-pair flat-bin offsets between paired osc and decay subchannels from
            // N_truth.  For each rule pair (osc r in {1,2,3}, decay r+3), the rule-(r+3)
            // carriers live at a fixed shift from rule-r in the flat grid.  Each pair is
            // allowed its own shift (so e.g. nue's osc-then-decay pair-order can differ
            // from numu's); within a pair, the shift must be self-consistent across
            // detectors — verified by checking that every non-zero rule-osc bin has a
            // matching non-zero rule-decay bin at the detected shift.
            auto first_nonzero_row = [](const Eigen::MatrixXf &M, int col) -> long int {
                for(long int i = 0; i < M.rows(); ++i) if(M(i, col) > 0.0f) return i;
                return -1;
            };
            for(int osc_r = 1; osc_r <= 3; ++osc_r) {
                const int decay_r = osc_r + 3;
                long int o = first_nonzero_row(N_truth, osc_r);
                long int d = first_nonzero_row(N_truth, decay_r);
                if(o < 0 || d < 0) {
                    log<LOG_ERROR>(L"%1% || PRO3p1_decay_vis_model1: rule %2% or %3% has no MC carriers in N_truth — "
                                   L"the XML must duplicate every rule-1/2/3 branch with a paired rule-4/5/6 branch.")
                        % __func__ % osc_r % decay_r;
                    exit(EXIT_FAILURE);
                }
                const long int shift = d - o;
                // Verify the shift is self-consistent: every non-zero rule-osc bin must map
                // to a non-zero rule-decay bin at the same offset.  Catches XML layouts that
                // pair osc/decay subchannels inconsistently across detectors.
                for(long int i = 0; i < N_truth.rows(); ++i) {
                    if(N_truth(i, osc_r) <= 0.0f) continue;
                    const long int j = i + shift;
                    if(j < 0 || j >= N_truth.rows() || N_truth(j, decay_r) <= 0.0f) {
                        log<LOG_ERROR>(L"%1% || PRO3p1_decay_vis_model1: rule %2%->%3% offset %4% is not "
                                       L"self-consistent (bin %5% has rule-%2% events but bin %6% has no "
                                       L"rule-%3% events). Each osc subchannel must be paired immediately "
                                       L"with its decay partner using identical truth binning, in every detector.")
                            % __func__ % osc_r % decay_r % shift % i % j;
                        exit(EXIT_FAILURE);
                    }
                }
                decay_bin_offsets[osc_r - 1] = shift;
            }

            nparams = 4;
            param_names = {"dmsq", "Ue4^2", "Um4^2", "g_phi"};
            pretty_param_names = {"#Deltam^{2}", "|U_{e4}|^{2}", "|U_{#mu4}|^{2}", "g_{#phi}"};
            pretty_param_units = {"eV^{2}", "", "", ""};
            is_log10 = {true, true, true, false};
            build_param_index();
            lb = Eigen::VectorXf(4);
            ub = Eigen::VectorXf(4);
            default_val = Eigen::VectorXf(4);
            lb << -2, -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), 0;
            ub << 2, -1e-4, -1e-4, 10;
            default_val << -2, -8, -8, 0;
        };

        const Eigen::MatrixXf& get_N_truth() const override { return N_truth; }

        int UnitarityConstraint(const Eigen::VectorXf &v){
            const float Ue4sq = maybe_convert_log("Ue4^2", v(param_name_to_index.at("Ue4^2")));
            const float Um4sq = maybe_convert_log("Um4^2", v(param_name_to_index.at("Um4^2")));
            const float g_phi = maybe_convert_log("g_phi", v(param_name_to_index.at("g_phi")));
            return ((Ue4sq+Um4sq)<1 && g_phi>=0 ? 1 : 0);
        }

        bool uses_get_counts() const override { return true; }

        Eigen::MatrixXf get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const override {
            // For this model, we calculate get_counts, which is more intuitive with energy smearing
            // (it's not exactly a probability for a single neutrino event when it depends on the behavior
            // of higher energy events), but it can be re-framed as a probability for consistency.
            // Note that this means probabilities greater than one are possible with large down-smearing!
            auto counts = get_counts(phys, var_arrs, N_truth);
            auto N_arr = N_truth.array();
            return (N_arr > 0.0f).select(counts.array() / N_arr, 0.0f);
        }

        // get_counts reads truth-level event counts from N_truth_vals (passed explicitly by the
        // caller) rather than this->N_truth.  Callers pass either this->N_truth for the baseline
        // spectrum, or a pre-reweighted copy (rows scaled by the per-truth-E flux weight) when
        // evaluating pre-migration flux systematics.
        Eigen::MatrixXf get_counts(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs,
                                   const Eigen::MatrixXf &N_truth_vals) const override {
            float dmsq  = maybe_convert_log("dmsq",  phys(param_name_to_index.at("dmsq")));
            float Ue4sq = maybe_convert_log("Ue4^2", phys(param_name_to_index.at("Ue4^2")));
            float Um4sq = maybe_convert_log("Um4^2", phys(param_name_to_index.at("Um4^2")));
            float g_phi = maybe_convert_log("g_phi", phys(param_name_to_index.at("g_phi")));
            float Us4sq = 1.0f - Ue4sq - Um4sq; // assuming no tau mixing (small effect on NC events)
            float sum_active = Ue4sq + Um4sq;

            float freq = 1.266932679f * dmsq;
            const auto &L_arr = var_arrs[0];
            const auto &E_arr = var_arrs[1];
            size_t n_flat = L_arr.size();
            size_t n_E    = phys_grid_sizes[1];

            float E_bin_width = E_arr[1] - E_arr[0];
            if (E_bin_width <= 0.0f) {
                log<LOG_ERROR>(L"%1% || expected positive truth-E bin width but got %2%") % __func__ % E_bin_width;
                exit(EXIT_FAILURE);
            }
            const float dE_tol = 1e-4f * E_bin_width;
            for(size_t i = 2; i < n_E; ++i) {
                float diff = E_arr[i] - E_arr[i-1];
                if(diff > 0.0f && std::abs(diff - E_bin_width) > dE_tol) {
                    log<LOG_ERROR>(L"%1% || 3+1+decay requires uniform truth energy binning, "
                        L"found bin width of %2% at bin %3% when expecting %4%.") % __func__ % diff % i % E_bin_width;
                    exit(EXIT_FAILURE);
                }
            }

            // alpha_phi = g_phi^2/(4 pi)
            // Gamma_nu4 = |Us4|^2 (1 - |Us4|^2) alpha_phi/4 m_4^2/E_4
            //           = |Us4|^2 (1 - |Us4|^2) g_phi^2/(16 pi) m_4^2/E_4
            // 1 / L_dec = Gamma_nu4
            // expterm = exp(-L/(2 L_dec)) = exp(-L/2 Gamma_nu4)
            //         = exp(-L/2 |Us4|^2 (1 - |Us4|^2) g_phi^2/(16 pi) m_4^2/E_4)
            //         = exp(-[m_4^2 L/4E] |Us4|^2 (1 - |Us4|^2) g_phi^2/(8 pi))
            //         = exp(-delta |Us4|^2 * sum_active g_phi^2/(8 pi))
            //         = exp(-delta * exp_prefactor)
            const float exp_prefactor = Us4sq * sum_active * g_phi * g_phi / (8.0f * 3.14159f);

            // Mixing-only prefactors for the decay-migration term — see
            // https://github.com/kjkellyphys/muB_oscillation/blob/main/OscTools/sterile_tools.py#L261
            // In https://journals.aps.org/prd/abstract/10.1103/PhysRevD.110.075002, just after
            // equation 11, I believe they actually give the formula for the flavor projection
            // squared, not the flavor projection.
            const float mix_mumu = Um4sq * Um4sq * Us4sq / sum_active;
            const float mix_mue  = Um4sq * Ue4sq * Us4sq / sum_active;
            const float mix_ee   = Ue4sq * Ue4sq * Us4sq / sum_active;

            // Column 0 holds the no-oscillation-no-decay prediction.
            // Columns 1..3 hold the same-bin (parent-bin) oscillation contribution.
            // Columns 4..6 hold the visible-decay daughter contribution for the
            // P_mumu / P_mue / P_ee channels respectively, populated at lower-E bins.
            Eigen::MatrixXf counts = Eigen::MatrixXf::Zero(n_flat, 7);

            // looping over flattened (L, E) bins — flat_par is the parent neutrino index
            for(size_t flat_par = 0; flat_par < n_flat; ++flat_par) {
                float E_par = E_arr[flat_par];
                float L_par = L_arr[flat_par];
                float delta = freq * L_par / E_par;
                float expterm = std::exp(-delta * exp_prefactor);
                float cos_mult_exp = std::cos(2.0f * delta) * expterm;
                float osc_term = 1.0f - 2.0f*cos_mult_exp + expterm*expterm;

                // Oscillation contribution at flat_par (parent = daughter for non-decayed events).
                // no-osc
                counts(flat_par, 0) += N_truth_vals(flat_par, 0);
                // P_mu_mu osc
                counts(flat_par, 1) += (1.0f - 2.0f*Um4sq*(1.0f - cos_mult_exp) + Um4sq*Um4sq*osc_term) * N_truth_vals(flat_par, 1);
                // P_mu_e osc
                counts(flat_par, 2) += Ue4sq * Um4sq * osc_term * N_truth_vals(flat_par, 2);
                // P_e_e osc
                counts(flat_par, 3) += (1.0f - 2.0f*Ue4sq*(1.0f - cos_mult_exp) + Ue4sq*Ue4sq*osc_term) * N_truth_vals(flat_par, 3);

                // Decay redistribution: this parent decays into daughters at lower truth-E
                // within the same truth-L bin.  Everything that does not depend on the
                // daughter's energy is collected into kernel_base_*; multiplying by
                // s_dec(E_dst, E_par) inside the inner loop gives the contribution to each
                // daughter bin.
                float decay_frac = 1.0f - expterm * expterm;             // (1 - exp(-L/L_dec(E_par)))
                float kernel_base_mumu = mix_mumu * E_bin_width * decay_frac * N_truth_vals(flat_par, 1);
                float kernel_base_mue  = mix_mue  * E_bin_width * decay_frac * N_truth_vals(flat_par, 2);
                float kernel_base_ee   = mix_ee   * E_bin_width * decay_frac * N_truth_vals(flat_par, 3);

                size_t l_idx     = flat_par / n_E;
                size_t e_par_idx = flat_par % n_E;
                // looping over lower-E daughter bins at the same L.  flat_dst lives in the
                // OSC subchannel's bin range (same as the parent); we shift it by the
                // per-pair offset when writing to cols 4/5/6 so each decay daughter lands at
                // its paired decay subchannel's bin, where the rule-4/5/6 carrier events
                // actually live.  Out-of-range writes (from cross-subchannel iterations) are
                // filtered: at those positions, N_truth for the decay rule is zero and
                // probs = counts/N_truth clamps to zero.
                for(size_t e_dst_idx = 0; e_dst_idx < e_par_idx; ++e_dst_idx) {
                    size_t flat_dst = l_idx * n_E + e_dst_idx;
                    float E_dst = E_arr[flat_dst];
                    float s_dec = 2.0f * E_dst / (E_par * E_par);        // dP/dE_d, 1/GeV
                    const long int dst_mumu = (long int)flat_dst + decay_bin_offsets[0];
                    const long int dst_mue  = (long int)flat_dst + decay_bin_offsets[1];
                    const long int dst_ee   = (long int)flat_dst + decay_bin_offsets[2];
                    if(dst_mumu >= 0 && dst_mumu < (long int)n_flat) counts(dst_mumu, 4) += kernel_base_mumu * s_dec;
                    if(dst_mue  >= 0 && dst_mue  < (long int)n_flat) counts(dst_mue,  5) += kernel_base_mue  * s_dec;
                    if(dst_ee   >= 0 && dst_ee   < (long int)n_flat) counts(dst_ee,   6) += kernel_base_ee   * s_dec;
                }
            }
            return counts;
        }
    };


class PRO3p1_decay_vis_model2 : public PROmodel {
    public:
        /// Truth-level MC populations; see PRO3p1_decay_vis_model1::N_truth for details.
        Eigen::MatrixXf N_truth;

        PRO3p1_decay_vis_model2(const PROpeller &prop, const std::map<std::string,int> &parameter_map) {
            // 3+1+decay to lower energy neutrinos
            // model 2 from https://journals.aps.org/prd/abstract/10.1103/PhysRevD.110.075002

            if(parameter_map.find("L") == parameter_map.end()) {
                log<LOG_ERROR>(L"%1%, %2% || Missing expected parameter: 'L'. Make sure it's in the model section of XML.") % __func__ % __LINE__;
                throw std::runtime_error("Missing parameter: L");
            }
            if(parameter_map.find("E") == parameter_map.end()) {
                log<LOG_ERROR>(L"%1%, %2% || Missing expected parameter: 'E'. Make sure it's in the model section of XML.") % __func__ % __LINE__;
                throw std::runtime_error("Missing parameter: E");
            }
            ivars = {parameter_map.at("L"), parameter_map.at("E")};

            prob_types = {0, 1, 2, 3};

            model_constraint = [this](const Eigen::VectorXf &v){return this->UnitarityConstraint(v);};

            build_hists_and_combined(prop); // fills hists, H_combined, phys_grid_sizes
            N_truth = compute_N_truth(prop); // save N_truth, so it can be modified by flux-weighting in PROcess.cxx

            nparams = 4;
            param_names = {"dmsq", "Ue4^2", "Um4^2", "g_e"};
            pretty_param_names = {"#Deltam^{2}", "|U_{e4}|^{2}", "|U_{#mu4}|^{2}", "g_{e}"};
            pretty_param_units = {"eV^{2}", "", "", ""};
            is_log10 = {true, true, true, false};
            build_param_index();
            lb = Eigen::VectorXf(4);
            ub = Eigen::VectorXf(4);
            default_val = Eigen::VectorXf(4);
            lb << -2, -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), 0;
            ub << 2, -1e-4, -1e-4, 10;
            default_val << -2, -8, -8, 0;
        };

        const Eigen::MatrixXf& get_N_truth() const override { return N_truth; }

        int UnitarityConstraint(const Eigen::VectorXf &v){
            const float Ue4sq = maybe_convert_log("Ue4^2", v(param_name_to_index.at("Ue4^2")));
            const float Um4sq = maybe_convert_log("Um4^2", v(param_name_to_index.at("Um4^2")));
            const float g_e = maybe_convert_log("g_e", v(param_name_to_index.at("g_e")));
            return ((Ue4sq+Um4sq)<1 && g_e>=0 ? 1 : 0);
        }

        bool uses_get_counts() const override { return true; }

        Eigen::MatrixXf get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const override {
            auto counts = get_counts(phys, var_arrs, N_truth);
            auto N_arr = N_truth.array();
            return (N_arr > 0.0f).select(counts.array() / N_arr, 0.0f);
        }

        Eigen::MatrixXf get_counts(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs,
                                   const Eigen::MatrixXf &N_truth_vals) const override {
            float dmsq  = maybe_convert_log("dmsq",  phys(param_name_to_index.at("dmsq")));
            float Ue4sq = maybe_convert_log("Ue4^2", phys(param_name_to_index.at("Ue4^2")));
            float Um4sq = maybe_convert_log("Um4^2", phys(param_name_to_index.at("Um4^2")));
            float g_e   = maybe_convert_log("g_e",   phys(param_name_to_index.at("g_e")));

            float freq = 1.266932679f * dmsq;
            const auto &L_arr = var_arrs[0];
            const auto &E_arr = var_arrs[1];
            size_t n_flat = L_arr.size();
            size_t n_E    = phys_grid_sizes[1];

            float E_bin_width = E_arr[1] - E_arr[0];
            if (E_bin_width <= 0.0f) {
                log<LOG_ERROR>(L"%1% || expected positive truth-E bin width but got %2%") % __func__ % E_bin_width;
                exit(EXIT_FAILURE);
            }
            const float dE_tol = 1e-4f * E_bin_width;
            for(size_t i = 2; i < n_E; ++i) {
                float diff = E_arr[i] - E_arr[i-1];
                if(diff > 0.0f && std::abs(diff - E_bin_width) > dE_tol) {
                    log<LOG_ERROR>(L"%1% || 3+1+decay requires uniform truth energy binning, "
                        L"found bin width of %2% at bin %3% when expecting %4%.") % __func__ % diff % i % E_bin_width;
                    exit(EXIT_FAILURE);
                }
            }

            // alpha_e = g_e^2/(4 pi)
            // Gamma_nu4 = alpha_e / 4 m_4^2/E_4
            //           = g_e^2/(16 pi) m_4^2/E_4
            // 1 / L_dec = Gamma_nu4
            // expterm = exp(-L/(2 L_dec)) = exp(-L/2 Gamma_nu4)
            //         = exp(-L/2 g_e^2/(16 pi) m_4^2/E_4)
            //         = exp(-[m_4^2 L/4E] g_e^2/(8 pi))
            //         = exp(-delta g_e^2/(8 pi))
            const float exp_prefactor = g_e * g_e / (8.0f * 3.14159f);

            // Zero-initialize counts: each cell accumulates (a) the oscillation contribution
            // when flat_par equals the cell, and (b) decay-migration from higher-E parents
            // at the same L.
            Eigen::MatrixXf counts = Eigen::MatrixXf::Zero(n_flat, 4);

            // looping over flattened (L, E) bins — flat_par is the parent index
            for(size_t flat_par = 0; flat_par < n_flat; ++flat_par) {
                float E_par = E_arr[flat_par];
                float L_par = L_arr[flat_par];
                float delta = freq * L_par / E_par;
                float expterm = std::exp(-delta * exp_prefactor);
                float cos_mult_exp = std::cos(2.0f * delta) * expterm;
                float osc_term = 1.0f - 2.0f*cos_mult_exp + expterm*expterm;

                // Oscillation contribution at flat_par (same as model1, g_e drives the damping)
                counts(flat_par, 0) += N_truth_vals(flat_par, 0);
                // P_mu_mu osc
                counts(flat_par, 1) += (1.0f - 2.0f*Um4sq*(1.0f - cos_mult_exp) + Um4sq*Um4sq*osc_term) * N_truth_vals(flat_par, 1);
                // P_mu_e osc
                counts(flat_par, 2) += Ue4sq * Um4sq * osc_term * N_truth_vals(flat_par, 2);
                // P_e_e osc
                counts(flat_par, 3) += (1.0f - 2.0f*Ue4sq*(1.0f - cos_mult_exp) + Ue4sq*Ue4sq*osc_term) * N_truth_vals(flat_par, 3);

                // Decay redistribution: in model2 (g_e coupling) only fullosc and nue are affected;
                // numu disappearance (j=1) sees no decay contribution.  The decay damping
                // (1 - exp(-L/L_dec)) is evaluated at the PARENT energy E_par.  Daughter-energy
                // dependence enters only through s_dec(E_dst, E_par).
                float decay_frac = 1.0f - expterm * expterm;             // (1 - exp(-L/L_dec(E_par)))
                float kernel_base_mue = Um4sq * E_bin_width * decay_frac * N_truth_vals(flat_par, 2);
                float kernel_base_ee  = Ue4sq * E_bin_width * decay_frac * N_truth_vals(flat_par, 3);

                size_t l_idx     = flat_par / n_E;
                size_t e_par_idx = flat_par % n_E;
                // looping over lower-E daughter bins at the same L
                for(size_t e_dst_idx = 0; e_dst_idx < e_par_idx; ++e_dst_idx) {
                    size_t flat_dst = l_idx * n_E + e_dst_idx;
                    float E_dst = E_arr[flat_dst];
                    float s_dec = 2.0f * E_dst / (E_par * E_par);
                    counts(flat_dst, 2) += kernel_base_mue * s_dec;
                    counts(flat_dst, 3) += kernel_base_ee  * s_dec;
                }
            }
            return counts;
        }
    };

/**
 * @brief 3+2 sterile-neutrino model with two independent heavy mass eigenstates.
 * @details Provides nu_mu disappearance, nu_mu → nu_e appearance, and nu_e disappearance channels
 * driven by two mass splittings Delta m^2_41 and Delta m^2_51 and four mixing elements
 * |U_e4|^2, |U_mu4|^2, |U_e5|^2, |U_mu5|^2 plus an inter-sterile CP phase phi_54.
 * Seven parameters total: dmsq41, dmsq51 [log10], Ue4sq, Um4sq, Ue5sq, Um5sq [log10], phi54 [linear, rad].
 * Unitarity constraints |U_ea|^2 sum < 1 and |U_mua|^2 sum < 1 are enforced via model_constraint.
 */
class PRO3p2 : public PROmodel {
public:
    /**
     * @brief Construct the PRO3p2 model.
     * @param prop          MC event store; used to build H_combined.
     * @param parameter_map Map from physics variable name (e.g., "L/E") to variable index in PROpeller.
     *                      Must contain the key "L/E".
     */
    PRO3p2(const PROpeller &prop, const std::map<std::string,int> &parameter_map) {

        // model functions: 0 = null, 1 = numu->numu, 2 = numu->nue, 3 = nue->nue, 4 = antinumu->antinue
        model_functions.push_back(
            [this]([[maybe_unused]] const Eigen::VectorXf &v, float) { (void)this; return 1.0f; });
        model_functions.push_back(
            [this](const Eigen::VectorXf &v, float le) { return this->Pmumu(v, le); });
        model_functions.push_back(
            [this](const Eigen::VectorXf &v, float le) { return this->Pmue(v, le); });
        model_functions.push_back(
            [this](const Eigen::VectorXf &v, float le) { return this->Pee(v, le); });
        model_functions.push_back(
            [this](const Eigen::VectorXf &v, float le) { return this->Pmue_anti(v, le); });
        prob_types = {0, 1, 2, 3, 4};

        if(parameter_map.find("L/E") == parameter_map.end()) {
            log<LOG_ERROR>(L"%1%, %2% || Missing expected parameter: 'L/E'. Make sure its in your model section of XML.")
                % __func__ % __LINE__;
            throw std::runtime_error("Missing parameter: L/E");
        }
        ivars = {parameter_map.at("L/E")};

        // Unitarity constraints for e and mu rows
        model_constraint = [this](const Eigen::VectorXf &v){ return this->UnitarityConstraint(v); };

        // build histograms as in other models
        build_hists_and_combined(prop);

        // parameters: dmsq41, dmsq51, log10(Ue4^2), log10(Um4^2), log10(Ue5^2), log10(Um5^2), phi54
        nparams = 7;
        param_names = {"dmsq41", "dmsq51", "Ue4sq", "Um4sq", "Ue5sq", "Um5sq", "phi54"};
        pretty_param_names = {
            "#Delta m^{2}_{41}", "#Delta m^{2}_{51}",
            "|U_{e4}|^{2}",      "|U_{#mu4}|^{2}",
            "|U_{e5}|^{2}",      "|U_{#mu5}|^{2}",
            "#phi_{54}"
        };
        pretty_param_units = {"eV^{2}", "eV^{2}", "", "", "", "", "rad"};
        is_log10 = {true, true, true, true, true, true, false};

        lb = Eigen::VectorXf(nparams);
        ub = Eigen::VectorXf(nparams);
        default_val = Eigen::VectorXf(nparams);

        // log10(Δm^2) between 10^-2 and 10^2, mixings between ~10^-8 and 1, phi in [-pi,pi]
        lb << -2, -2,
              -5, -5,
              -5, -5,
               0.0f;
        ub <<  2,  2,
              -0.01,  -0.01,
              -0.01,  -0.01,
              6.28318f;

        // some reasonable defaults
        default_val << -1, 0, -4, -4, -4, -4, 0.0f;
    }

    // Enforce |Ue4|^2 + |Ue5|^2 <= 1 and |Um4|^2 + |Um5|^2 <= 1
    int UnitarityConstraint(const Eigen::VectorXf &v) {
        float Ue4sq = std::pow(10.0f, v(2));
        float Um4sq = std::pow(10.0f, v(3));
        float Ue5sq = std::pow(10.0f, v(4));
        float Um5sq = std::pow(10.0f, v(5));

        if(Ue4sq + Ue5sq >= 1.0f) return 0;
        if(Um4sq + Um5sq >= 1.0f) return 0;
        // more careful with unitarity since tau and steriles are not specified and checking the max value of probability
        if(4*Um4sq*Ue4sq + 4*Um5sq*Ue5sq + 8*std::sqrt(Um4sq*Ue4sq*Um5sq*Ue5sq) >= 1.0f) return 0;

        if(1-4*(1-Ue4sq- Ue5sq)*(Ue4sq + Ue5sq) - 4*Ue4sq*Ue5sq >= 1.0f) return 0;

        if(1-4*(1-Um4sq- Um5sq)*(Um4sq + Um5sq) - 4*Um4sq*Um5sq >= 1.0f) return 0;


        return 1;
    }

    // Convenience to unpack parameters

    // ---------- Appearance: νμ → νe (α=μ, β=e) ----------
    float Pmue(const Eigen::VectorXf &v, float le) const {

    // Convert log10 parameters to physical values
    float dm41  = std::pow(10.0f, v(0));
    float dm51  = std::pow(10.0f, v(1));
    float Ue4sq = std::pow(10.0f, v(2));
    float Um4sq = std::pow(10.0f, v(3));
    float Ue5sq = std::pow(10.0f, v(4));
    float Um5sq = std::pow(10.0f, v(5));
    float phi54 = v(6);

    // Oscillation phases
    float x41 = 1.266932679f * dm41 * le;
    float x51 = 1.266932679f * dm51 * le;
    float x54 = 1.266932679f * (dm51 - dm41) * le;

    // Standard 3+2 appearance terms
    float term1 = 4.0f * Um4sq * Ue4sq * std::sin(x41) * std::sin(x41);
    float term2 = 4.0f * Um5sq * Ue5sq * std::sin(x51) * std::sin(x51);
    float term3 = 8.0f * std::sqrt(Um4sq * Ue4sq * Um5sq * Ue5sq)
                        * std::sin(x41) * std::sin(x51)
                        * std::cos(x54 - phi54);

    float prob = term1 + term2 + term3;

    const float eps = 1e-6f;

    if (prob < 0.0f && prob > -eps) prob = 0.0f;

    if (prob > 1.0f && prob < 1.0f + eps) prob = 1.0f;

    if(prob < 0.0f || prob > 1.0f || std::isnan(prob)) {
        log<LOG_ERROR>(L"%1% || Bad Pmue = %2%  (le=%3%)"
            L"\ndm41=%4% dm51=%5%  Ue4sq=%6% Um4sq=%7%  Ue5sq=%8% Um5sq=%9%  phi54=%10% term1=%11% term2=%12% term3=%13%"
             ) % __func__ % prob % le
             % dm41 % dm51 % Ue4sq % Um4sq % Ue5sq % Um5sq % phi54% term1 % term2 % term3;
            exit(EXIT_FAILURE);
        }
        return prob;
    }


    // ---------- Appearance: anti-νμ → anti-νe (CP-conjugate, phi54 flipped) ----------
    float Pmue_anti(const Eigen::VectorXf &v, float le) const {

    float dm41  = std::pow(10.0f, v(0));
    float dm51  = std::pow(10.0f, v(1));
    float Ue4sq = std::pow(10.0f, v(2));
    float Um4sq = std::pow(10.0f, v(3));
    float Ue5sq = std::pow(10.0f, v(4));
    float Um5sq = std::pow(10.0f, v(5));
    float phi54 = v(6);

    float x41 = 1.266932679f * dm41 * le;
    float x51 = 1.266932679f * dm51 * le;
    float x54 = 1.266932679f * (dm51 - dm41) * le;

    // Standard 3+2 appearance terms (anti-nu: +phi54 instead of -phi54)
    float term1 = 4.0f * Um4sq * Ue4sq * std::sin(x41) * std::sin(x41);
    float term2 = 4.0f * Um5sq * Ue5sq * std::sin(x51) * std::sin(x51);
    float term3 = 8.0f * std::sqrt(Um4sq * Ue4sq * Um5sq * Ue5sq)
                        * std::sin(x41) * std::sin(x51)
                        * std::cos(x54 + phi54);

    float prob = term1 + term2 + term3;

    const float eps = 1e-6f;

    if (prob < 0.0f && prob > -eps) prob = 0.0f;

    if (prob > 1.0f && prob < 1.0f + eps) prob = 1.0f;

    if(prob < 0.0f || prob > 1.0f || std::isnan(prob)) {
        log<LOG_ERROR>(L"%1% || Bad Pmue_anti = %2%  (le=%3%)"
            L"\ndm41=%4% dm51=%5%  Ue4sq=%6% Um4sq=%7%  Ue5sq=%8% Um5sq=%9%  phi54=%10% term1=%11% term2=%12% term3=%13%"
             ) % __func__ % prob % le
             % dm41 % dm51 % Ue4sq % Um4sq % Ue5sq % Um5sq % phi54 % term1 % term2 % term3;
            exit(EXIT_FAILURE);
        }
        return prob;
    }

    // ---------- Disappearance: νμ → νμ (α=μ) ----------
   float Pmumu(const Eigen::VectorXf &v, float le) const {

    float dm41  = std::pow(10.0f, v(0));
    float dm51  = std::pow(10.0f, v(1));
    float Ue4sq = std::pow(10.0f, v(2));
    float Um4sq = std::pow(10.0f, v(3));
    float Ue5sq = std::pow(10.0f, v(4));
    float Um5sq = std::pow(10.0f, v(5));
    float phi54 = v(6);

    float x41 = 1.266932679f * dm41 * le;
    float x51 = 1.266932679f * dm51 * le;
    float x54 = 1.266932679f * (dm51 - dm41) * le;

    float one_minus = 1.0f - Um4sq - Um5sq;

    float s41 = std::sin(x41);
    float s51 = std::sin(x51);
    float s54 = std::sin(x54);

// Individual components
    float term1 = 1.0f;

    float term2 = -4.0f * one_minus *
              (Um4sq * s41 * s41 +
               Um5sq * s51 * s51);

    float term3 = -4.0f * Um4sq * Um5sq *
              (s54 * s54);

// Full probability
    float prob = term1 + term2 + term3;

    
    const float eps = 1e-6f;

    if (prob < 0.0f && prob > -eps) prob = 0.0f;

    if (prob > 1.0f && prob < 1.0f + eps) prob = 1.0f;

    if(prob < 0.0f || prob > 1.0f || std::isnan(prob)) {
        log<LOG_ERROR>(L"%1% || Bad Pmumu = %2%  (le=%3%)"
            L"\ndm41=%4% dm51=%5%  Ue4sq=%6% Um4sq=%7%  Ue5sq=%8% Um5sq=%9%  phi54=%10% term1=%11% term2=%12% term3=%13%"
        ) % __func__ % prob % le
          % dm41 % dm51 % Ue4sq % Um4sq % Ue5sq % Um5sq % phi54 % term1 % term2 % term3;
        exit(EXIT_FAILURE);
    }
    return prob;
    }


    // ---------- Disappearance: νe → νe (α=e) ----------
   float Pee(const Eigen::VectorXf &v, float le) const {

    float dm41  = std::pow(10.0f, v(0));
    float dm51  = std::pow(10.0f, v(1));
    float Ue4sq = std::pow(10.0f, v(2));
    float Um4sq = std::pow(10.0f, v(3));
    float Ue5sq = std::pow(10.0f, v(4));
    float Um5sq = std::pow(10.0f, v(5));
    float phi54 = v(6);

    float x41 = 1.266932679f * dm41 * le;
    float x51 = 1.266932679f * dm51 * le;
    float x54 = 1.266932679f * (dm51 - dm41) * le;

    float one_minus = 1.0f - Ue4sq - Ue5sq;

    float s41 = std::sin(x41);
    float s51 = std::sin(x51);
    float s54 = std::sin(x54);

    float term1 = 1.0f;
    float term2 = -4.0f * one_minus * (Ue4sq * s41 * s41 + Ue5sq * s51 * s51);
    float term3 = -4.0f * Ue4sq * Ue5sq * s54 * s54;

    float prob = term1 + term2 + term3;

    const float eps = 1e-6f;

    if (prob < 0.0f && prob > -eps) prob = 0.0f;

    if (prob > 1.0f && prob < 1.0f + eps) prob = 1.0f;

    if(prob < 0.0f || prob > 1.0f || std::isnan(prob)) {
        log<LOG_ERROR>(L"%1% || Bad Pee = %2%  (le=%3%)"
            L"\ndm41=%4% dm51=%5%  Ue4sq=%6% Um4sq=%7%  Ue5sq=%8% Um5sq=%9%  phi54=%10% term1=%11% term2=%12% term3=%13%"
        ) % __func__ % prob % le
          % dm41 % dm51 % Ue4sq % Um4sq % Ue5sq % Um5sq % phi54 % term1 % term2 % term3;
        exit(EXIT_FAILURE);
    }
    return prob;
    }

};

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
    PROLBL(const PROpeller &prop, const std::map<std::string,int> &parameter_map) {

        model_functions.push_back([this](const Eigen::VectorXf &v, float) {(void)this;(void)v; return 1.0; });
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pee(v,le); });
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pemu(v,le); });
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Petau(v,le); });
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pmue(v,le); });
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pmumu(v,le); });
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pmutau(v,le); });
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Ptaue(v,le); });
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Ptaumu(v,le); });
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Ptautau(v,le); });

        prob_types = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

        if(parameter_map.find("L/E") == parameter_map.end()) {
            log<LOG_ERROR>(L"%1%, %2% || Missing expected parameter: 'L/E'.Make sure its in your model section of XML. ") % __func__ % __LINE__;
            throw std::runtime_error("Missing parameter: L/E");
        }

        ivars = {parameter_map.at("L/E")};

        build_hists_and_combined(prop);


        nparams = 6;
        param_names = {"dmsq_21", "dmsq_31", "sinsqt12", "sinsqt13", "sinsqt23", "delta_CP"}; 
        pretty_param_names = {"#Delta m^{2}_{21}", "#Delta m^{2}_{31}", "sin^{2}#theta_{12}",
            "sin^2#theta_{13}", "sin^{2}#theta_{23}", "delta_{CP}"}; 
        pretty_param_units = {"eV^{2}", "eV^{2}", "", "", "", "rad"}; 
        is_log10 = {false, false, false, false, false, false};
        build_param_index();
        lb = Eigen::VectorXf(6);
        ub = Eigen::VectorXf(6);
        lb << 6e-5f, -3e-3f, 0.2f, 0.01f, 0.3f, -M_PI;
        ub << 9e-5f, 3e-3f, 0.4f, 0.04f, 0.7f, M_PI;
        default_val = Eigen::VectorXf(6);
        // Defaults set to midpoint of [lb, ub] for each parameter so the fitter starts in-range.
        default_val << 7.5e-5f, 1e-3f, 0.3f, 0.025f, 0.5f, 0.0f;
    }

    /// @brief nu_e → nu_e survival probability via NuFastLBL. @param params Physics vector (6 params). @param le L/E [km/GeV]. @return P(nu_e→nu_e).
    float Pee(const Eigen::VectorXf &params, float le) {
        float probs_returned[3][3];
        NuFastLBL::Probability_Matter_LBL(params(2), params(3), params(4), params(5),
                  params(0), params(1), 1300, le, rho_earth, Ye_earth, 0, &probs_returned);
        return probs_returned[0][0];
    }

    /// @brief nu_e → nu_mu appearance probability via NuFastLBL. @param params Physics vector (6 params). @param le L/E [km/GeV]. @return P(nu_e→nu_mu).
    float Pemu(const Eigen::VectorXf &params, float le) {
        float probs_returned[3][3];
        NuFastLBL::Probability_Matter_LBL(params(2), params(3), params(4), params(5), 
                  params(0), params(1), 1300, le, rho_earth, Ye_earth, 0, &probs_returned);
        return probs_returned[0][1];
    }
    
    /// @brief nu_e → nu_tau appearance probability via NuFastLBL. @param params Physics vector. @param le L/E [km/GeV]. @return P(nu_e→nu_tau).
    float Petau(const Eigen::VectorXf &params, float le) {
        float probs_returned[3][3];
        NuFastLBL::Probability_Matter_LBL(params(2), params(3), params(4), params(5),
                  params(0), params(1), 1300, le, rho_earth, Ye_earth, 0, &probs_returned);
        return probs_returned[0][2];
    }

    /// @brief nu_mu → nu_e appearance probability via NuFastLBL. @param params Physics vector. @param le L/E [km/GeV]. @return P(nu_mu→nu_e).
    float Pmue(const Eigen::VectorXf &params, float le) {
        float probs_returned[3][3];
        NuFastLBL::Probability_Matter_LBL(params(2), params(3), params(4), params(5),
                  params(0), params(1), 1300, le, rho_earth, Ye_earth, 0, &probs_returned);
        return probs_returned[1][0];
    }

    /// @brief nu_mu → nu_mu survival probability via NuFastLBL. @param params Physics vector. @param le L/E [km/GeV]. @return P(nu_mu→nu_mu).
    float Pmumu(const Eigen::VectorXf &params, float le) {
        float probs_returned[3][3];
        NuFastLBL::Probability_Matter_LBL(params(2), params(3), params(4), params(5),
                  params(0), params(1), 1300, le, rho_earth, Ye_earth, 0, &probs_returned);
        return probs_returned[1][1];
    }

    /// @brief nu_mu → nu_tau appearance probability via NuFastLBL. @param params Physics vector. @param le L/E [km/GeV]. @return P(nu_mu→nu_tau).
    float Pmutau(const Eigen::VectorXf &params, float le) {
        float probs_returned[3][3];
        NuFastLBL::Probability_Matter_LBL(params(2), params(3), params(4), params(5),
                  params(0), params(1), 1300, le, rho_earth, Ye_earth, 0, &probs_returned);
        return probs_returned[1][2];
    }

    /// @brief nu_tau → nu_e appearance probability via NuFastLBL. @param params Physics vector. @param le L/E [km/GeV]. @return P(nu_tau→nu_e).
    float Ptaue(const Eigen::VectorXf &params, float le) {
        float probs_returned[3][3];
        NuFastLBL::Probability_Matter_LBL(params(2), params(3), params(4), params(5),
                  params(0), params(1), 1300, le, rho_earth, Ye_earth, 0, &probs_returned);
        return probs_returned[2][0];
    }

    /// @brief nu_tau → nu_mu appearance probability via NuFastLBL. @param params Physics vector. @param le L/E [km/GeV]. @return P(nu_tau→nu_mu).
    float Ptaumu(const Eigen::VectorXf &params, float le) {
        float probs_returned[3][3];
        NuFastLBL::Probability_Matter_LBL(params(2), params(3), params(4), params(5),
                  params(0), params(1), 1300, le, rho_earth, Ye_earth, 0, &probs_returned);
        return probs_returned[2][1];
    }

    /// @brief nu_tau → nu_tau survival probability via NuFastLBL. @param params Physics vector. @param le L/E [km/GeV]. @return P(nu_tau→nu_tau).
    float Ptautau(const Eigen::VectorXf &params, float le) {
        float probs_returned[3][3];
        NuFastLBL::Probability_Matter_LBL(params(2), params(3), params(4), params(5),
                  params(0), params(1), 1300, le, rho_earth, Ye_earth, 0, &probs_returned);
        return probs_returned[2][2];
    }

    Eigen::MatrixXf get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const override {
        const auto &le_arr = var_arrs[0];
        // Eigen matrices are column major by default so we want this layout to get a 
        // contiguous probs array from each column, then transpose before returning.
        Eigen::MatrixXf probs(model_functions.size(), le_arr.size());
        probs.row(0).setConstant(1);
        for(size_t i = 0; i < le_arr.size(); ++i) {
            NuFastLBL::Probability_Matter_LBL(phys(2), phys(3), phys(4), phys(5), 
                      phys(0), phys(1), 1300, le_arr[i], rho_earth, Ye_earth, 0,
                      (float(*)[3][3])((float*)probs.col(i).data()+1));
        }
        return probs.transpose();
    }
};

/**
 * @brief Factory function: construct a PROmodel subclass by name.
 * @details Reads `config.m_model_tag` to select the appropriate model and passes
 * `config.m_model_parameter_map` for variable-index lookup.
 * Supported names: "nullmodel", "numudis", "numudisTEST", "nueapp", "nuedis",
 * "3+1", "3+1_angles", "3+1_3A", "3+1_3B", "3+1_3C", "3+1_decay_invis", "3+1_decay_vis_model1", "3+1_decay_vis_model2", "3+2", "LBL".
 * Terminates with LOG_ERROR if the name is unrecognised.
 * @param config  Parsed configuration; provides the model tag and parameter map.
 * @param prop    MC event store used to build H_combined histograms.
 * @return        Owning pointer to the constructed PROmodel.
 */
static inline
std::unique_ptr<PROmodel> get_model_from_string(const PROconfig& config, const PROpeller &prop) {
    std::string name = config.m_model_tag;
    
    if(name == "nullmodel") {
        return std::unique_ptr<PROmodel>(new NullModel(prop));
    } else if(name == "numudis") {
        return std::unique_ptr<PROmodel>(new PROnumudis(prop,config.m_model_parameter_map));
    } else if(name == "numudisTEST") {
        return std::unique_ptr<PROmodel>(new PROnumudisTEST(prop,config.m_model_parameter_map));
    } else if(name == "nueapp") {
        return std::unique_ptr<PROmodel>(new PROnueapp(prop,config.m_model_parameter_map));
    } else if(name == "nuedis") {
        return std::unique_ptr<PROmodel>(new PROnuedis(prop,config.m_model_parameter_map));
    } else if(name == "3+1") {
        return std::unique_ptr<PROmodel>(new PRO3p1(prop,config.m_model_parameter_map));
    } else if(name == "3+1_angles") {
        return std::unique_ptr<PROmodel>(new PRO3p1_angles(prop,config.m_model_parameter_map));
    } else if(name == "3+1_3A") {
        return std::unique_ptr<PROmodel>(new PRO3p1_3A(prop,config.m_model_parameter_map));
    } else if(name == "3+1_3B") {
        return std::unique_ptr<PROmodel>(new PRO3p1_3B(prop,config.m_model_parameter_map));
    } else if(name == "3+1_3C") {
        return std::unique_ptr<PROmodel>(new PRO3p1_3C(prop,config.m_model_parameter_map));
    } else if(name == "3+1_decay_invis") {
        return std::unique_ptr<PROmodel>(new PRO3p1_decay_invis(prop,config.m_model_parameter_map));
    } else if(name == "3+1_decay_vis_model1") {
        return std::unique_ptr<PROmodel>(new PRO3p1_decay_vis_model1(prop,config.m_model_parameter_map));
    } else if(name == "3+1_decay_vis_model2") {
        return std::unique_ptr<PROmodel>(new PRO3p1_decay_vis_model2(prop,config.m_model_parameter_map));
    } else if(name == "3+2") {
        return std::unique_ptr<PROmodel>(new PRO3p2(prop, config.m_model_parameter_map));
    } else if(name == "LBL") {
        return std::unique_ptr<PROmodel>(new PROLBL(prop, config.m_model_parameter_map));
    }
    log<LOG_ERROR>(L"%1% || Unrecognized model name %2%. Try numudis, nueapp, nuedis, 3+1, 3+1_angles, 3+1_3(A,B,C), 3+1_decay_invis, 3+1_decay_vis, 3+2. for now. Terminating.") % __func__ % name.c_str();
    exit(EXIT_FAILURE);
}

}

#endif

