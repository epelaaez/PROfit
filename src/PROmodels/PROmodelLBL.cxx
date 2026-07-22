/**
 * @file PROmodelLBL.cxx
 * @brief Implementation of the PROLBL three-flavour long-baseline model.
 * @author PROfit Collaboration
 */
#include "PROmodels/PROmodelLBL.h"
#include "NuFastLBL.h"

#include <cmath>

namespace PROfit {

PROLBL::PROLBL(const PROpeller &prop, const std::map<std::string,int> &parameter_map) {

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

float PROLBL::Pee(const Eigen::VectorXf &params, float le) {
    float probs_returned[3][3];
    NuFastLBL::Probability_Matter_LBL(params(2), params(3), params(4), params(5),
              params(0), params(1), 1300, le, rho_earth, Ye_earth, 0, &probs_returned);
    return probs_returned[0][0];
}

float PROLBL::Pemu(const Eigen::VectorXf &params, float le) {
    float probs_returned[3][3];
    NuFastLBL::Probability_Matter_LBL(params(2), params(3), params(4), params(5),
              params(0), params(1), 1300, le, rho_earth, Ye_earth, 0, &probs_returned);
    return probs_returned[0][1];
}

float PROLBL::Petau(const Eigen::VectorXf &params, float le) {
    float probs_returned[3][3];
    NuFastLBL::Probability_Matter_LBL(params(2), params(3), params(4), params(5),
              params(0), params(1), 1300, le, rho_earth, Ye_earth, 0, &probs_returned);
    return probs_returned[0][2];
}

float PROLBL::Pmue(const Eigen::VectorXf &params, float le) {
    float probs_returned[3][3];
    NuFastLBL::Probability_Matter_LBL(params(2), params(3), params(4), params(5),
              params(0), params(1), 1300, le, rho_earth, Ye_earth, 0, &probs_returned);
    return probs_returned[1][0];
}

float PROLBL::Pmumu(const Eigen::VectorXf &params, float le) {
    float probs_returned[3][3];
    NuFastLBL::Probability_Matter_LBL(params(2), params(3), params(4), params(5),
              params(0), params(1), 1300, le, rho_earth, Ye_earth, 0, &probs_returned);
    return probs_returned[1][1];
}

float PROLBL::Pmutau(const Eigen::VectorXf &params, float le) {
    float probs_returned[3][3];
    NuFastLBL::Probability_Matter_LBL(params(2), params(3), params(4), params(5),
              params(0), params(1), 1300, le, rho_earth, Ye_earth, 0, &probs_returned);
    return probs_returned[1][2];
}

float PROLBL::Ptaue(const Eigen::VectorXf &params, float le) {
    float probs_returned[3][3];
    NuFastLBL::Probability_Matter_LBL(params(2), params(3), params(4), params(5),
              params(0), params(1), 1300, le, rho_earth, Ye_earth, 0, &probs_returned);
    return probs_returned[2][0];
}

float PROLBL::Ptaumu(const Eigen::VectorXf &params, float le) {
    float probs_returned[3][3];
    NuFastLBL::Probability_Matter_LBL(params(2), params(3), params(4), params(5),
              params(0), params(1), 1300, le, rho_earth, Ye_earth, 0, &probs_returned);
    return probs_returned[2][1];
}

float PROLBL::Ptautau(const Eigen::VectorXf &params, float le) {
    float probs_returned[3][3];
    NuFastLBL::Probability_Matter_LBL(params(2), params(3), params(4), params(5),
              params(0), params(1), 1300, le, rho_earth, Ye_earth, 0, &probs_returned);
    return probs_returned[2][2];
}

Eigen::MatrixXf PROLBL::get_probs(const Eigen::VectorXf &phys, const std::vector<std::vector<float>> &var_arrs) const {
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

}
