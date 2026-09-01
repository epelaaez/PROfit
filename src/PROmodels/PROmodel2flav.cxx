/**
 * @file PROmodel2flav.cxx
 * @brief Implementation of the two-variable (L, E) validation model. The single-variable
 * two-flavour models (numudis / nueapp / nuedis) are sine-kernel recipes in
 * PROmodels/PROmodelSine.cxx.
 * @author PROfit Collaboration
 */
#include "PROmodels/PROmodel2flav.h"

namespace PROfit {

// ------------------------------------------------------------------
// PROnumudisTEST
// ------------------------------------------------------------------

PROnumudisTEST::PROnumudisTEST(const PROpeller &prop, const std::map<std::string,int> &parameter_map) {
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
}

float PROnumudisTEST::Pmumu(float dmsq, float sinsq2thmumu, float le) const {
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

Eigen::MatrixXf PROnumudisTEST::get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const {
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

std::vector<Eigen::MatrixXf> PROnumudisTEST::get_probs_grad(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const {
    // Same physics and derivatives as the numudis recipe's get_probs_grad (P_mumu = 1 − s·sin²x),
    // evaluated on the flat L × E grid with L/E formed per grid point.
    float dmsq         = maybe_convert_log("dmsq",       phys(0));
    float sinsq2thmumu = maybe_convert_log("sinsq2thmm", phys(1));
    constexpr float LN10 = 2.302585093f;
    float ddm = is_log10[0] ? LN10 * dmsq : 1.0f;
    float dss = is_log10[1] ? LN10 * sinsq2thmumu : 1.0f;
    if(sinsq2thmumu > 1) { sinsq2thmumu = 1; dss = 0; }
    if(sinsq2thmumu < 0) { sinsq2thmumu = 0; dss = 0; }

    float freq = 1.266932679f * dmsq;
    const size_t n_flat = var_arrs[0].size();
    std::vector<Eigen::MatrixXf> grads(2, Eigen::MatrixXf::Zero(n_flat, 2));
    for(size_t i = 0; i < n_flat; ++i) {
        float L = var_arrs[0][i];
        float E = var_arrs[1][i];
        float le = (E > 0.0f) ? L / E : 0.0f;
        float x = freq * le;
        float sinterm = std::sin(x);
        grads[0](i, 1) = -sinsq2thmumu * std::sin(2.0f*x) * 1.266932679f * le * ddm;
        grads[1](i, 1) = -sinterm * sinterm * dss;
    }
    return grads;
}

}
