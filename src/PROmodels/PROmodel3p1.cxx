/**
 * @file PROmodel3p1.cxx
 * @brief Implementation of the 3+1 sterile-neutrino model family.
 * @author PROfit Collaboration
 */
#include "PROmodels/PROmodel3p1.h"

namespace PROfit {

// ------------------------------------------------------------------
// PRO3p1
// ------------------------------------------------------------------

PRO3p1::PRO3p1(const PROpeller &prop, const std::map<std::string,int> &parameter_map) {

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
}

int PRO3p1::UnitarityConstraint(const Eigen::VectorXf &v){
    const float Ue4sq = maybe_convert_log("Ue4^2", v(param_name_to_index.at("Ue4^2")));
    const float Um4sq = maybe_convert_log("Um4^2", v(param_name_to_index.at("Um4^2")));
    return   ((Ue4sq+Um4sq)<1 ? 1 : 0);
}

float PRO3p1::Pmue(float dmsq, float Ue4sq, float Um4sq, float le) const{
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

float PRO3p1::Pmumu(float dmsq, [[maybe_unused]]float Ue4sq, float Um4sq, float le) const{
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

float PRO3p1::Pee(float dmsq, float Ue4sq, [[maybe_unused]]float Um4sq, float le) const{
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

Eigen::MatrixXf PRO3p1::get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const {
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

std::vector<Eigen::MatrixXf> PRO3p1::get_probs_grad(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const {
    // Model (x = k · Δm² · L/E, Ue = |Ue4|², Um = |Um4|²; columns 1..3):
    //   P_mumu = 1 − 4 Um (1 − Um) sin²x
    //   P_mue  =     4 Ue Um       sin²x
    //   P_ee   = 1 − 4 Ue (1 − Ue) sin²x
    // Derivatives w.r.t. the physical parameters (then × chain factors ddm, dUe, dUm):
    //   ∂/∂Δm²: each probability's sin²x coefficient × d(sin²x)/dΔm² = sin(2x)·k·L/E
    //   ∂P_mumu/∂Um = −4 (1 − 2 Um) sin²x          (d[Um(1−Um)]/dUm = 1 − 2Um)
    //   ∂P_mue/∂Ue  =  4 Um sin²x,   ∂P_mue/∂Um = 4 Ue sin²x
    //   ∂P_ee/∂Ue   = −4 (1 − 2 Ue) sin²x
    const auto &le_arr = var_arrs[0];
    float dmsq  = maybe_convert_log("dmsq",  phys(0));
    float Ue4sq = maybe_convert_log("Ue4^2", phys(1));
    float Um4sq = maybe_convert_log("Um4^2", phys(2));

    // Chain factors d(linear)/d(internal); all three params are log10 by default.
    constexpr float LN10 = 2.302585093f;
    float ddm = is_log10[0] ? LN10 * dmsq  : 1.0f;
    float dUe = is_log10[1] ? LN10 * Ue4sq : 1.0f;
    float dUm = is_log10[2] ? LN10 * Um4sq : 1.0f;

    constexpr float k = 1.266932679f;
    std::vector<Eigen::MatrixXf> grads(3, Eigen::MatrixXf::Zero(le_arr.size(), model_functions.size()));
    for(size_t i = 0; i < le_arr.size(); ++i) {
        float x = k * dmsq * le_arr[i];
        float sinterm = std::sin(x);
        float s2 = sinterm * sinterm;
        float dsin2_ddm = std::sin(2.0f*x) * k * le_arr[i] * ddm;  // d(sin^2 x)/d(internal dmsq)

        // col 1: P_mumu = 1 - 4 Um(1-Um) sin^2
        grads[0](i, 1) = -4.0f * Um4sq * (1.0f - Um4sq) * dsin2_ddm;   // ∂/∂θ(Δm²)
        grads[2](i, 1) = -4.0f * (1.0f - 2.0f*Um4sq) * s2 * dUm;       // ∂/∂θ(Um)
        // col 2: P_mue = 4 Ue Um sin^2
        grads[0](i, 2) =  4.0f * Ue4sq * Um4sq * dsin2_ddm;            // ∂/∂θ(Δm²)
        grads[1](i, 2) =  4.0f * Um4sq * s2 * dUe;                     // ∂/∂θ(Ue)
        grads[2](i, 2) =  4.0f * Ue4sq * s2 * dUm;                     // ∂/∂θ(Um)
        // col 3: P_ee = 1 - 4 Ue(1-Ue) sin^2
        grads[0](i, 3) = -4.0f * Ue4sq * (1.0f - Ue4sq) * dsin2_ddm;   // ∂/∂θ(Δm²)
        grads[1](i, 3) = -4.0f * (1.0f - 2.0f*Ue4sq) * s2 * dUe;       // ∂/∂θ(Ue)
    }
    return grads;
}

// ------------------------------------------------------------------
// PRO3p1_angles
// ------------------------------------------------------------------

PRO3p1_angles::PRO3p1_angles(const PROpeller &prop, const std::map<std::string,int> &parameter_map) {

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
}

int PRO3p1_angles::UnitarityConstraint(const Eigen::VectorXf &){
    return   1;
}

float PRO3p1_angles::Pmue(float dmsq, float sinsq2th14, float sinsqth24, float le) const{
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

float PRO3p1_angles::Pmumu(float dmsq, float sinsq2th14, float sinsqth24, float le) const{
    dmsq =  maybe_convert_log("dmsq", dmsq);
    float s214 = maybe_convert_log("sinsq2th14", sinsq2th14);
    float s24 = maybe_convert_log("sinsqth24", sinsqth24);
    float c14 = (1.0+sqrt(1.0-s214))/2.0f;

    float sinterm = std::sin(1.266932679f*dmsq*(le));
    float prob    = 1.0f - 4.0f*(c14*s24*(1.0f-c14*s24))*sinterm*sinterm;


    if(prob<0.0 || prob >1.0){
        log<LOG_ERROR>(L"%1% || Your probability %2% is outside the bounds of math."
                       L"dmsq = %3%, s214 = %4%, s24 = %5%, L/E = %6%")
            % __func__ % prob % dmsq % s214 % s24 % le;
        log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
        exit(EXIT_FAILURE);
    }
    return prob;
}

float PRO3p1_angles::Pee(float dmsq, float sinsq2th14, [[maybe_unused]]float sinsqth24, float le) const{
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

Eigen::MatrixXf PRO3p1_angles::get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const {
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
        probs(i, 1) = 1.0f - 4.0f*c14*s24*(1.0f-c14*s24)*sinterm*sinterm;

        // P_mue
        probs(i, 2) = s214*s24*sinterm*sinterm;

        // P_ee
        probs(i, 3) = 1.0f - s214*sinterm*sinterm;

    }

    return probs;
}

std::vector<Eigen::MatrixXf> PRO3p1_angles::get_probs_grad(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const {
    // Parameters: Δm², s214 = sin²2θ₁₄, s24 = sin²θ₂₄. With
    //   c14 = cos²θ₁₄ = (1 + √(1 − s214)) / 2     ⇒  dc14/ds214 = −1 / (4 √(1 − s214))
    //   A   = c14 · s24  (= |Um4|²)                ⇒  ∂A/∂s214 = s24 · dc14/ds214,  ∂A/∂s24 = c14
    // the probabilities are (x = k · Δm² · L/E)
    //   P_mumu = 1 − 4 A (1 − A) sin²x   ⇒  ∂P_mumu/∂A = −4 (1 − 2A) sin²x, chained through ∂A/∂s214, ∂A/∂s24
    //   (the factor 4 is explicit here because sin²2θ_μμ = 4A(1−A) is not itself a parameter)
    //   P_mue  = s214 · s24 · sin²x
    //   P_ee   = 1 − s214 · sin²x
    // Δm² enters only through sin²x: d(sin²x)/dΔm² = sin(2x) · k · L/E.
    const auto &le_arr = var_arrs[0];
    float dmsq = maybe_convert_log("dmsq", phys(0));
    float s214 = maybe_convert_log("sinsq2th14", phys(1));
    float s24  = maybe_convert_log("sinsqth24", phys(2));
    float root = std::sqrt(std::max(0.0f, 1.0f - s214));
    float c14  = (1.0f + root) / 2.0f;

    // Chain factors d(linear)/d(internal) for log10 parameters.
    constexpr float LN10 = 2.302585093f;
    float ddm   = is_log10[0] ? LN10 * dmsq : 1.0f;
    float ds214 = is_log10[1] ? LN10 * s214 : 1.0f;
    float ds24  = is_log10[2] ? LN10 * s24  : 1.0f;
    // A = c14 * s24 (the |Um4|^2 of Pmumu); dc14/ds214 = -1/(4 root).
    float A = c14 * s24;
    float dA_ds214 = (root > 0.0f ? -s24 / (4.0f * root) : 0.0f) * ds214;
    float dA_ds24  = c14 * ds24;

    constexpr float k = 1.266932679f;
    std::vector<Eigen::MatrixXf> grads(3, Eigen::MatrixXf::Zero(le_arr.size(), model_functions.size()));
    for(size_t i = 0; i < le_arr.size(); ++i) {
        float x = k * dmsq * le_arr[i];
        float sinterm = std::sin(x);
        float s2 = sinterm * sinterm;
        float dsin2_ddm = std::sin(2.0f*x) * k * le_arr[i] * ddm;

        // col 1: P_mumu = 1 - 4 A(1-A) sin^2
        grads[0](i, 1) = -4.0f * A * (1.0f - A) * dsin2_ddm;
        grads[1](i, 1) = -4.0f * (1.0f - 2.0f*A) * s2 * dA_ds214;
        grads[2](i, 1) = -4.0f * (1.0f - 2.0f*A) * s2 * dA_ds24;
        // col 2: P_mue = s214 s24 sin^2
        grads[0](i, 2) = s214 * s24 * dsin2_ddm;
        grads[1](i, 2) = s24 * s2 * ds214;
        grads[2](i, 2) = s214 * s2 * ds24;
        // col 3: P_ee = 1 - s214 sin^2
        grads[0](i, 3) = -s214 * dsin2_ddm;
        grads[1](i, 3) = -s2 * ds214;
    }
    return grads;
}

// ------------------------------------------------------------------
// PRO3p1_3A
// ------------------------------------------------------------------

PRO3p1_3A::PRO3p1_3A(const PROpeller &prop, const std::map<std::string,int> &parameter_map) {

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
}

int PRO3p1_3A::UnitarityConstraint(const Eigen::VectorXf &){
    return   1;
}

float PRO3p1_3A::Pmue(float dmsq, float sinsq2thee, [[maybe_unused]]float sinsqth24, float le) const{
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

float PRO3p1_3A::Pmumu(float dmsq, float sinsq2thee, [[maybe_unused]]float sinsqth24, float le) const{
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

float PRO3p1_3A::Pee(float dmsq, float sinsq2thee, [[maybe_unused]]float sinsqth24, float le) const{

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

Eigen::MatrixXf PRO3p1_3A::get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const {
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

std::vector<Eigen::MatrixXf> PRO3p1_3A::get_probs_grad(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const {
    // Parameters: Δm², see = sin²2θ_ee, s24 = sin²θ₂₄. With
    //   Um = |Um4|² = s24 (1 + √(1 − see)) / 2   ⇒  ∂Um/∂see = −s24 / (4 √(1 − see)),  ∂Um/∂s24 = (1 + √(1 − see)) / 2
    // the probabilities are (x = k · Δm² · L/E)
    //   P_mumu = 1 − 4 Um (1 − Um) sin²x   ⇒  ∂P_mumu/∂Um = −4 (1 − 2 Um) sin²x, chained through ∂Um/∂see, ∂Um/∂s24
    //   P_mue  = s24 · see · sin²x
    //   P_ee   = 1 − see · sin²x
    const auto &le_arr = var_arrs[0];
    float dmsq = maybe_convert_log("dmsq", phys(0));
    float see  = maybe_convert_log("sinsq2thee", phys(1));
    float s24  = maybe_convert_log("sinsqth24", phys(2));
    float root = std::sqrt(std::max(0.0f, 1.0f - see));
    float Um4sq = s24 / 2.0f * (1.0f + root);
    float smue  = s24 * see;

    constexpr float LN10 = 2.302585093f;
    float ddm  = is_log10[0] ? LN10 * dmsq : 1.0f;
    float dsee = is_log10[1] ? LN10 * see  : 1.0f;
    float ds24 = is_log10[2] ? LN10 * s24  : 1.0f;
    float dUm_dsee = (root > 0.0f ? -s24 / (4.0f * root) : 0.0f) * dsee;
    float dUm_ds24 = (1.0f + root) / 2.0f * ds24;

    constexpr float k = 1.266932679f;
    std::vector<Eigen::MatrixXf> grads(3, Eigen::MatrixXf::Zero(le_arr.size(), model_functions.size()));
    for(size_t i = 0; i < le_arr.size(); ++i) {
        float x = k * dmsq * le_arr[i];
        float sinterm = std::sin(x);
        float s2 = sinterm * sinterm;
        float dsin2_ddm = std::sin(2.0f*x) * k * le_arr[i] * ddm;

        // col 1: P_mumu = 1 - 4 Um(1-Um) sin^2
        grads[0](i, 1) = -4.0f * Um4sq * (1.0f - Um4sq) * dsin2_ddm;
        grads[1](i, 1) = -4.0f * (1.0f - 2.0f*Um4sq) * s2 * dUm_dsee;
        grads[2](i, 1) = -4.0f * (1.0f - 2.0f*Um4sq) * s2 * dUm_ds24;
        // col 2: P_mue = s24 see sin^2
        grads[0](i, 2) = smue * dsin2_ddm;
        grads[1](i, 2) = s24 * s2 * dsee;
        grads[2](i, 2) = see * s2 * ds24;
        // col 3: P_ee = 1 - see sin^2
        grads[0](i, 3) = -see * dsin2_ddm;
        grads[1](i, 3) = -s2 * dsee;
    }
    return grads;
}

// ------------------------------------------------------------------
// PRO3p1_3B
// ------------------------------------------------------------------

PRO3p1_3B::PRO3p1_3B(const PROpeller &prop,
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
int PRO3p1_3B::UnitarityConstraint(const Eigen::VectorXf &v) {
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
float PRO3p1_3B::Pmumu(float dmsq, float sinsq2thmumu, [[maybe_unused]] float sinsqth24prime, float le) const {
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
float PRO3p1_3B::Pmue(float dmsq, float sinsq2thmumu, float sB, float le) const {
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
float PRO3p1_3B::Pee(float dmsq, float sinsq2thmumu, float sB, float le) const {
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

Eigen::MatrixXf PRO3p1_3B::get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const {
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

std::vector<Eigen::MatrixXf> PRO3p1_3B::get_probs_grad(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const {
    // Parameters: Δm², smm = sin²2θ_μμ, sB. With
    //   Ue = |Ue4|² = sB (1 + √(1 − smm)) / 2   ⇒  ∂Ue/∂smm = −sB / (4 √(1 − smm)),  ∂Ue/∂sB = (1 + √(1 − smm)) / 2
    // the probabilities are (x = k · Δm² · L/E)
    //   P_mumu = 1 − smm · sin²x
    //   P_mue  = sB · smm · sin²x
    //   P_ee   = 1 − 4 Ue (1 − Ue) sin²x   ⇒  ∂P_ee/∂Ue = −4 (1 − 2 Ue) sin²x, chained through ∂Ue/∂smm, ∂Ue/∂sB
    const auto &le_arr = var_arrs[0];
    float dmsq = maybe_convert_log("dmsq", phys(0));
    float smm  = maybe_convert_log("sinsq2thmumu", phys(1));
    float sB   = maybe_convert_log("sB", phys(2));
    float root = std::sqrt(std::max(0.0f, 1.0f - smm));
    float Ue4sq = (sB / 2.0f) * (1.0f + root);

    constexpr float LN10 = 2.302585093f;
    float ddm  = is_log10[0] ? LN10 * dmsq : 1.0f;
    float dsmm = is_log10[1] ? LN10 * smm  : 1.0f;
    float dsB  = is_log10[2] ? LN10 * sB   : 1.0f;
    float dUe_dsmm = (root > 0.0f ? -sB / (4.0f * root) : 0.0f) * dsmm;
    float dUe_dsB  = (1.0f + root) / 2.0f * dsB;

    constexpr float k = 1.266932679f;
    std::vector<Eigen::MatrixXf> grads(3, Eigen::MatrixXf::Zero(le_arr.size(), model_functions.size()));
    for(size_t i = 0; i < le_arr.size(); ++i) {
        float x = k * dmsq * le_arr[i];
        float sinterm = std::sin(x);
        float s2 = sinterm * sinterm;
        float dsin2_ddm = std::sin(2.0f*x) * k * le_arr[i] * ddm;

        // col 1: P_mumu = 1 - smm sin^2
        grads[0](i, 1) = -smm * dsin2_ddm;
        grads[1](i, 1) = -s2 * dsmm;
        // col 2: P_mue = sB smm sin^2
        grads[0](i, 2) = sB * smm * dsin2_ddm;
        grads[1](i, 2) = sB * s2 * dsmm;
        grads[2](i, 2) = smm * s2 * dsB;
        // col 3: P_ee = 1 - 4 Ue(1-Ue) sin^2
        grads[0](i, 3) = -4.0f * Ue4sq * (1.0f - Ue4sq) * dsin2_ddm;
        grads[1](i, 3) = -4.0f * (1.0f - 2.0f*Ue4sq) * s2 * dUe_dsmm;
        grads[2](i, 3) = -4.0f * (1.0f - 2.0f*Ue4sq) * s2 * dUe_dsB;
    }
    return grads;
}

// ------------------------------------------------------------------
// PRO3p1_3C
// ------------------------------------------------------------------

PRO3p1_3C::PRO3p1_3C(const PROpeller &prop, const std::map<std::string,int> &parameter_map) {

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
}

int PRO3p1_3C::UnitarityConstraint(const Eigen::VectorXf &v){
    const float sinsq2thmue = maybe_convert_log("sinsq2thmue", v(param_name_to_index.at("sinsq2thmue")));
    const float xi = maybe_convert_log("xi", v(param_name_to_index.at("xi")));
    return   (std::sqrt(sinsq2thmue)*std::cosh(xi)<0.999 ? 1 : 0);
}

float PRO3p1_3C::Pmue(float dmsq, float sinsq2thmue, [[maybe_unused]]float xi, float le) const{
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

float PRO3p1_3C::Pmumu(float dmsq, float sinsq2thmue, float xi, float le) const{
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

float PRO3p1_3C::Pee(float dmsq, float sinsq2thmue, float xi, float le) const{
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

Eigen::MatrixXf PRO3p1_3C::get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const {
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

std::vector<Eigen::MatrixXf> PRO3p1_3C::get_probs_grad(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const {
    // Parameters: Δm², smue = sin²2θ_μe (log10), ξ (linear). With
    //   Um = |Um4|² = e^{−ξ} √smue / 2,   Ue = |Ue4|² = e^{+ξ} √smue / 2
    //   ∂Um/∂ξ = −Um,  ∂Ue/∂ξ = +Ue
    //   ∂Um/∂smue = e^{−ξ} · d√smue/dsmue / 2  (and likewise for Ue); for the log10
    //   parameter d√smue/dθ = ln10 · √smue / 2, which stays finite as smue → 0
    // the probabilities are (x = k · Δm² · L/E)
    //   P_mumu = 1 − 4 Um (1 − Um) sin²x   ⇒  ∂P_mumu/∂Um = −4 (1 − 2 Um) sin²x
    //   P_mue  = smue · sin²x
    //   P_ee   = 1 − 4 Ue (1 − Ue) sin²x   ⇒  ∂P_ee/∂Ue   = −4 (1 − 2 Ue) sin²x
    const auto &le_arr = var_arrs[0];
    float dmsq = maybe_convert_log("dmsq", phys(0));
    float smue = maybe_convert_log("sinsq2thmue", phys(1));
    float xi   = maybe_convert_log("xi", phys(2));
    float sqrtsin = std::sqrt(std::max(0.0f, smue));
    float Um4sq = std::exp(-xi) * sqrtsin / 2.0f;
    float Ue4sq = std::exp( xi) * sqrtsin / 2.0f;

    constexpr float LN10 = 2.302585093f;
    float ddm   = is_log10[0] ? LN10 * dmsq : 1.0f;
    float dsmue = is_log10[1] ? LN10 * smue : 1.0f;
    float dxi   = is_log10[2] ? LN10 * xi   : 1.0f;
    // d sqrt(smue) / d(internal smue): for log10 params LN10*sqrt/2 (finite at 0).
    float dsqrt = is_log10[1] ? LN10 * sqrtsin / 2.0f : (sqrtsin > 0.0f ? 1.0f / (2.0f * sqrtsin) : 0.0f);
    float dUm_dsmue = std::exp(-xi) * dsqrt / 2.0f, dUm_dxi = -Um4sq * dxi;
    float dUe_dsmue = std::exp( xi) * dsqrt / 2.0f, dUe_dxi =  Ue4sq * dxi;

    constexpr float k = 1.266932679f;
    std::vector<Eigen::MatrixXf> grads(3, Eigen::MatrixXf::Zero(le_arr.size(), model_functions.size()));
    for(size_t i = 0; i < le_arr.size(); ++i) {
        float x = k * dmsq * le_arr[i];
        float sinterm = std::sin(x);
        float s2 = sinterm * sinterm;
        float dsin2_ddm = std::sin(2.0f*x) * k * le_arr[i] * ddm;

        // col 1: P_mumu = 1 - 4 Um(1-Um) sin^2
        grads[0](i, 1) = -4.0f * Um4sq * (1.0f - Um4sq) * dsin2_ddm;
        grads[1](i, 1) = -4.0f * (1.0f - 2.0f*Um4sq) * s2 * dUm_dsmue;
        grads[2](i, 1) = -4.0f * (1.0f - 2.0f*Um4sq) * s2 * dUm_dxi;
        // col 2: P_mue = smue sin^2
        grads[0](i, 2) = smue * dsin2_ddm;
        grads[1](i, 2) = s2 * dsmue;
        // col 3: P_ee = 1 - 4 Ue(1-Ue) sin^2
        grads[0](i, 3) = -4.0f * Ue4sq * (1.0f - Ue4sq) * dsin2_ddm;
        grads[1](i, 3) = -4.0f * (1.0f - 2.0f*Ue4sq) * s2 * dUe_dsmue;
        grads[2](i, 3) = -4.0f * (1.0f - 2.0f*Ue4sq) * s2 * dUe_dxi;
    }
    return grads;
}
}
