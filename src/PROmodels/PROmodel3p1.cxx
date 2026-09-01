/**
 * @file PROmodel3p1.cxx
 * @brief Implementation of the 3+1 invisible-decay model. The sine-kernel 3+1 variants
 * (3+1, 3+1_angles, 3+1_3A/3B/3C and their _NC extensions) are recipes in
 * PROmodels/PROmodelSine.cxx.
 * @author PROfit Collaboration
 */
#include "PROmodels/PROmodel3p1.h"

namespace PROfit {

// ------------------------------------------------------------------
// PRO3p1_decay_invis
// ------------------------------------------------------------------

PRO3p1_decay_invis::PRO3p1_decay_invis(const PROpeller &prop, const std::map<std::string,int> &parameter_map) {
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
}

int PRO3p1_decay_invis::UnitarityConstraint(const Eigen::VectorXf &v){
    // ensures positive g2 in addition to the usual unitarity constraints
    const float Ue4sq = maybe_convert_log("Ue4^2", v(param_name_to_index.at("Ue4^2")));
    const float Um4sq = maybe_convert_log("Um4^2", v(param_name_to_index.at("Um4^2")));
    const float g2 = maybe_convert_log("g2", v(param_name_to_index.at("g2")));
    return   ((Ue4sq+Um4sq)<1 && g2>=0 ? 1 : 0);
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

float PRO3p1_decay_invis::Pmue(float dmsq, float Ue4sq, float Um4sq, float g2, float le) const{
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

float PRO3p1_decay_invis::Pmumu(float dmsq, [[maybe_unused]]float Ue4sq, float Um4sq, float g2, float le) const{
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

float PRO3p1_decay_invis::Pee(float dmsq, float Ue4sq, [[maybe_unused]]float Um4sq, float g2, float le) const{
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

Eigen::MatrixXf PRO3p1_decay_invis::get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const {
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

std::vector<Eigen::MatrixXf> PRO3p1_decay_invis::get_probs_grad(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const {
    // Parameters: Δm², Ue = |Ue4|², Um = |Um4|², g² (linear). With
    //   δ = k · Δm² · L/E,   c = cos 2δ,   e = exp(−g² δ / 8π),   osc = 1 − 2 e c + e²
    // the probabilities (see the derivation above Pmue) are
    //   P_aa  = 1 − 2 Ua (1 − e c) + Ua² · osc        (a = μ for col 1, e for col 3)
    //   P_mue = Ue · Um · osc
    // Everything except the mixings depends on θ only through δ and g²:
    //   ∂e/∂δ = −(g²/8π) e,   ∂c/∂δ = −2 sin 2δ,   ∂(ec)/∂δ = e' c + e c',   ∂osc/∂δ = −2 ∂(ec)/∂δ + 2 e e'
    //   ∂e/∂g² = −(δ/8π) e,   ∂(ec)/∂g² = c ∂e/∂g²,                          ∂osc/∂g² = −2 ∂(ec)/∂g² + 2 e ∂e/∂g²
    //   ∂δ/∂Δm² = k · L/E
    // and for the mixings  ∂P_aa/∂Ua = −2 (1 − e c) + 2 Ua · osc,  ∂P_mue/∂Ue = Um · osc,  ∂P_mue/∂Um = Ue · osc.
    const auto &le_arr = var_arrs[0];
    float dmsq  = maybe_convert_log("dmsq", phys(0));
    float Ue4sq = maybe_convert_log("Ue4^2", phys(1));
    float Um4sq = maybe_convert_log("Um4^2", phys(2));
    float g2    = maybe_convert_log("g2", phys(3));

    constexpr float LN10 = 2.302585093f;
    float ddm = is_log10[0] ? LN10 * dmsq  : 1.0f;
    float dUe = is_log10[1] ? LN10 * Ue4sq : 1.0f;
    float dUm = is_log10[2] ? LN10 * Um4sq : 1.0f;
    float dg2 = is_log10[3] ? LN10 * g2    : 1.0f;

    constexpr float k = 1.266932679f;
    constexpr float inv8pi = 1.0f / (8.0f * 3.14159f);   // same constant as get_probs
    std::vector<Eigen::MatrixXf> grads(4, Eigen::MatrixXf::Zero(le_arr.size(), model_functions.size()));
    for(size_t i = 0; i < le_arr.size(); ++i) {
        float delta   = k * dmsq * le_arr[i];
        float costerm = std::cos(2.0f*delta);
        float expterm = std::exp(-g2 * delta * inv8pi);
        float ec      = expterm * costerm;
        float osc     = 1.0f - 2.0f*ec + expterm*expterm;

        // Derivatives of ec and osc wrt delta and g2.
        float de_ddelta = -g2 * inv8pi * expterm;
        float dc_ddelta = -2.0f * std::sin(2.0f*delta);
        float dec_ddelta  = de_ddelta * costerm + expterm * dc_ddelta;
        float dosc_ddelta = -2.0f * dec_ddelta + 2.0f * expterm * de_ddelta;
        float de_dg2   = -delta * inv8pi * expterm;
        float dec_dg2  = de_dg2 * costerm;
        float dosc_dg2 = -2.0f * dec_dg2 + 2.0f * expterm * de_dg2;
        float ddelta_ddm = k * le_arr[i] * ddm;

        // col 1: P_mumu = 1 - 2 Um (1 - ec) + Um^2 osc
        grads[0](i, 1) = (2.0f*Um4sq*dec_ddelta + Um4sq*Um4sq*dosc_ddelta) * ddelta_ddm;
        grads[2](i, 1) = (-2.0f*(1.0f - ec) + 2.0f*Um4sq*osc) * dUm;
        grads[3](i, 1) = (2.0f*Um4sq*dec_dg2 + Um4sq*Um4sq*dosc_dg2) * dg2;
        // col 2: P_mue = Ue Um osc
        grads[0](i, 2) = Ue4sq * Um4sq * dosc_ddelta * ddelta_ddm;
        grads[1](i, 2) = Um4sq * osc * dUe;
        grads[2](i, 2) = Ue4sq * osc * dUm;
        grads[3](i, 2) = Ue4sq * Um4sq * dosc_dg2 * dg2;
        // col 3: P_ee = 1 - 2 Ue (1 - ec) + Ue^2 osc
        grads[0](i, 3) = (2.0f*Ue4sq*dec_ddelta + Ue4sq*Ue4sq*dosc_ddelta) * ddelta_ddm;
        grads[1](i, 3) = (-2.0f*(1.0f - ec) + 2.0f*Ue4sq*osc) * dUe;
        grads[3](i, 3) = (2.0f*Ue4sq*dec_dg2 + Ue4sq*Ue4sq*dosc_dg2) * dg2;
    }
    return grads;
}

}
