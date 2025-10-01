#ifndef PROMODEL_H
#define PROMODEL_H

#include "PROconfig.h"
#include "PROpeller.h"

#include <Eigen/Eigen>

#include <Eigen/src/Core/Matrix.h>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace PROfit {

class PROmodel {
public:
    size_t nparams;
    int ivar; //TODO, this should be a string like "true" and then in config we have a map to that variable. error if not found. 
    std::vector<std::string> param_names;
    std::vector<std::string> pretty_param_names;
    std::vector<std::string> pretty_param_units;
    Eigen::VectorXf lb, ub, default_val;
    std::vector<std::function<float(const Eigen::VectorXf&, float)>> model_functions;
    std::function<int(const Eigen::VectorXf&)> model_constraint;
    std::vector<std::vector<Eigen::MatrixXf>> hists; //2D hists for binned oscilattion, one for each model function, and the N-variables 
                                        //Todo: make this a vector of length n_variables, and fill them all. For now 1 is "special". 
};

class NullModel : public PROmodel {
public:
    NullModel(const PROpeller &prop) {
        nparams = 0;
        ivar = 1;
        model_functions.push_back([](const Eigen::VectorXf &, float){ return 1.0f; });
       
        size_t nvar = prop.variable_mc_stat_err.size();
        hists.resize(nvar);
        for(size_t v = 0; v <nvar ;v++){
            for(size_t m = 0; m < model_functions.size(); ++m) {
                hists.at(v).emplace_back(Eigen::MatrixXf::Constant(prop.variable_hist_storage(ivar,v).rows(), prop.variable_hist_storage(ivar,v).cols(),0.0));
                Eigen::MatrixXf &h = hists.at(v).back();
                for(size_t i = 0; i < prop.NEvent(); ++i) {
                    int tbin = prop.VariableBinIndex(ivar, i), rbin = prop.VariableBinIndex(v, i);
                    if(tbin<0 || rbin<0)continue;
                    h(tbin, rbin) += prop.added_weights[i];
                }
            }
        }
    }
};

class PROnumudis : public PROmodel {
public:
    PROnumudis(const PROpeller &prop,const std::map<std::string,int> &parameter_map) {
        model_functions.push_back([this]([[maybe_unused]] const Eigen::VectorXf &v, float) {(void)this; return 1.0;});
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pmumu(v(0),v(1),le);});
        

        if(parameter_map.find("L/E") == parameter_map.end()) {
            log<LOG_ERROR>(L"%1%, %2% || Missing expected parameter: 'L/E'.Make sure its in your model section of XML. ") % __func__ % __LINE__;
            throw std::runtime_error("Missing parameter: L/E");
        }
        ivar = parameter_map.at("L/E");

        size_t nvar = prop.variable_mc_stat_err.size();
        hists.resize(nvar);
        for(size_t v = 0; v <nvar ;v++){
            for(size_t m = 0; m < model_functions.size(); ++m) {
                hists.at(v).emplace_back(Eigen::MatrixXf::Constant(prop.variable_hist_storage(ivar,v).rows(), prop.variable_hist_storage(ivar,v).cols(),0.0));
                Eigen::MatrixXf &h = hists.at(v).back();
                for(size_t i = 0; i < prop.NEvent(); ++i) {
                    if(prop.model_rule[i] != (int)m) continue;
                    int tbin = prop.VariableBinIndex(ivar, i), rbin = prop.VariableBinIndex(v, i);
                    if(tbin<0 || rbin<0) continue;
                    h(tbin, rbin) += prop.added_weights[i];
                }
            }
        }
        nparams = 2;
        param_names = {"dmsq", "sinsq2thmm"}; 
        pretty_param_names = {"#Deltam^{2}", "sin^{2}2#theta_{#mu#mu}"}; 
        pretty_param_units = {"eV^{2}", ""}; 
        lb = Eigen::VectorXf(2);
        ub = Eigen::VectorXf(2);
        default_val = Eigen::VectorXf(2);
        lb << -2, -std::numeric_limits<float>::infinity();
        ub << 2, 0;
        default_val << -10, -10;

    };

    /* Function: 3+1 numu->numue disapperance prob in SBL approx */
    float Pmumu(float dmsq, float sinsq2thmumu, float le) const{
        dmsq = std::pow(10.0f, dmsq);
        sinsq2thmumu = std::pow(10.0f, sinsq2thmumu);

        if(sinsq2thmumu > 1) {
            //log<LOG_ERROR>(L"%1% || sinsq2thmumu is %2% which is greater than 1. Setting to 1.")     % __func__ % sinsq2thmumu;
            sinsq2thmumu = 1;
        }
        if(sinsq2thmumu < 0) {
            log<LOG_ERROR>(L"%1% || sinsq2thmumu is %2% which is less than 0. Setting to 0.")
                % __func__ % sinsq2thmumu;
            sinsq2thmumu = 0;
        }

        float sinterm = std::sin(1.27f*dmsq*(le));
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
};

class PROnueapp : public PROmodel {
public:
    PROnueapp(const PROpeller &prop,const std::map<std::string,int> &parameter_map) {
        model_functions.push_back([this]([[maybe_unused]] const Eigen::VectorXf &v, float) {(void)this; return 1.0;});
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pmue(v(0),v(1),le);});
        if(parameter_map.find("L/E") == parameter_map.end()) {
            log<LOG_ERROR>(L"%1%, %2% || Missing expected parameter: 'L/E'. Make sure its in your model section of XML.") % __func__ % __LINE__;
            throw std::runtime_error("Missing parameter: L/E");
        }
        ivar = parameter_map.at("L/E");

        size_t nvar = prop.variable_mc_stat_err.size();
        hists.resize(nvar);
        for(size_t v = 0; v <nvar ;v++){
            for(size_t m = 0; m < model_functions.size(); ++m) {
                hists.at(v).emplace_back(Eigen::MatrixXf::Constant(prop.variable_hist_storage(ivar,v).rows(), prop.variable_hist_storage(ivar,v).cols(),0.0));
                Eigen::MatrixXf &h = hists.at(v).back();
                for(size_t i = 0; i < prop.NEvent(); ++i) {
                    if(prop.model_rule[i] != (int)m) continue;
                    int tbin = prop.VariableBinIndex(ivar, i), rbin = prop.VariableBinIndex(v, i);
                    if(tbin<0 || rbin<0)continue;
                    h(tbin, rbin) += prop.added_weights[i];
                }
            }
        }
         nparams = 2;
        param_names = {"dmsq", "sinsq2thme"}; 
        pretty_param_names = {"#Deltam^{2}", "sin^{2}2#theta_{#mue}"}; 
        pretty_param_units = {"eV^{2}", ""}; 
        lb = Eigen::VectorXf(2);
        ub = Eigen::VectorXf(2);
        default_val = Eigen::VectorXf(2);
        lb << -2, -std::numeric_limits<float>::infinity();
        ub << 2, 0;
        default_val << -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity();
    
        log<LOG_INFO>(L"%1% || setting up a model nueapp, with  %2% params.")     % __func__ % nparams;
        for(size_t i=0; i< nparams;i++){
            log<LOG_INFO>(L"%1% || Param %2% is %3% with lower bound/upper bound of %4%/%5% and default %6%")     % __func__ % i % param_names[i].c_str() % lb[i] % ub[i] % default_val[i];
        }

    };

    float Pmue(float dmsq, float sinsq2thmue, float le) const{
        dmsq = std::pow(10.0f, dmsq);
        sinsq2thmue = std::pow(10.0f, sinsq2thmue);

        if(sinsq2thmue > 1) {
            //log<LOG_ERROR>(L"%1% || sinsq2thmue is %2% which is greater than 1. Setting to 1.")  % __func__ % sinsq2thmue;
            sinsq2thmue = 1;
        }
        if(sinsq2thmue < 0) {
            log<LOG_ERROR>(L"%1% || sinsq2thmue is %2% which is less than 0. Setting to 0.")
                % __func__ % sinsq2thmue;
            sinsq2thmue = 0;
        }

        float sinterm = std::sin(1.27f*dmsq*(le));
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
};

class PRO3p1 : public PROmodel {
public:
    PRO3p1(const PROpeller &prop, const std::map<std::string,int> &parameter_map) {

        model_functions.push_back([this]([[maybe_unused]] const Eigen::VectorXf &v, float) {(void)this; return 1.0; });
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pmumu(v(0),v(1),v(2),le); });
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pmue(v(0),v(1),v(2),le); });
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pee(v(0),v(1),v(2),le); });
        if(parameter_map.find("L/E") == parameter_map.end()) {
            log<LOG_ERROR>(L"%1%, %2% || Missing expected parameter: 'L/E'. Make sure its in your model section of XML.") % __func__ % __LINE__;
            throw std::runtime_error("Missing parameter: L/E");
        }
        ivar = parameter_map.at("L/E");

        //constraints
        model_constraint = [this](const Eigen::VectorXf &v){return this->UnitarityConstraint(v);};


         size_t nvar = prop.variable_mc_stat_err.size();
        hists.resize(nvar);
        for(size_t v = 0; v <nvar ;v++){
            for(size_t m = 0; m < model_functions.size(); ++m) {
                hists.at(v).emplace_back(Eigen::MatrixXf::Constant(prop.variable_hist_storage(ivar,v).rows(), prop.variable_hist_storage(ivar,v).cols(),0.0));
                Eigen::MatrixXf &h = hists.at(v).back();
                for(size_t i = 0; i < prop.NEvent(); ++i) {
                    if(prop.model_rule[i] != (int)m) continue;
                    int tbin = prop.VariableBinIndex(ivar, i), rbin = prop.VariableBinIndex(v, i);
                    if(tbin<0 || rbin<0)continue;
                    h(tbin, rbin) += prop.added_weights[i];
                }
            }
        }


        nparams = 3;
        param_names = {"dmsq", "Ue4^2", "Um4^2"}; 
        pretty_param_names = {"#Deltam^{2}", "|U_{e4}|^{2}", "|U_{#mu4}|^{2}"}; 
        pretty_param_units = {"eV^{2}", "",""}; 
        lb = Eigen::VectorXf(3);
        ub = Eigen::VectorXf(3);
        default_val = Eigen::VectorXf(3);
        lb << -2, -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity();
        ub << 2, -1e-4, -1e-4;
        //default_val << -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity();
        default_val << -2, -8, -8;
    };

    int UnitarityConstraint(const Eigen::VectorXf &v){
        return   (pow(10,v(1))+pow(10,v(2))<1 ? 1 : 0);      
    }

    float Pmue(float dmsq, float Ue4sq, float Um4sq, float le) const{
        dmsq = std::pow(10.0f, dmsq);
        Ue4sq = std::pow(10.0f, Ue4sq);
        Um4sq = std::pow(10.0f, Um4sq);

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

        float sinterm = std::sin(1.27f*dmsq*(le));
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

    float Pmumu(float dmsq, [[maybe_unused]]float Ue4sq, float Um4sq, float le) const{
        dmsq = std::pow(10.0f, dmsq);
        Um4sq = std::pow(10.0f, Um4sq);

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

        float sinterm = std::sin(1.27*dmsq*(le));
        float prob    = 1.0f - 4.0f*Um4sq*(1.0f-Um4sq)*sinterm*sinterm;

        if(prob<0.0 || prob >1.0){
            log<LOG_ERROR>(L"%1% || Your probability %2% is outside the bounds of math. dmsq = %3%, Um4sq = %4%, L/E = %5%") % __func__ % prob % dmsq % Um4sq % le;
            log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
            exit(EXIT_FAILURE);
        }

        return prob;
    }

    float Pee(float dmsq, float Ue4sq, [[maybe_unused]]float Um4sq, float le) const{
        dmsq = std::pow(10.0f, dmsq);
        Ue4sq = std::pow(10.0f, Ue4sq);

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

        float sinterm = std::sin(1.27*dmsq*(le));
        float prob    = 1.0f - 4.0f*Ue4sq*(1.0f-Ue4sq)*sinterm*sinterm;

        if(prob<0.0 || prob >1.0){
            log<LOG_ERROR>(L"%1% || Your probability %2% is outside the bounds of math. dmsq = %3%, Ue4sq = %4%, L/E = %5%") % __func__ % prob % dmsq % Ue4sq % le;
            log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
            exit(EXIT_FAILURE);
        }

        return prob;
    }
};


class PRO3p1_3C : public PROmodel {
public:
    PRO3p1_3C(const PROpeller &prop, const std::map<std::string,int> &parameter_map) {

        model_functions.push_back([this]([[maybe_unused]] const Eigen::VectorXf &v, float) {(void)this; return 1.0; });
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pmumu(v(0),v(1),v(2),le); });
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pmue(v(0),v(1),v(2),le); });
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pee(v(0),v(1),v(2),le); });
        if(parameter_map.find("L/E") == parameter_map.end()) {
            log<LOG_ERROR>(L"%1%, %2% || Missing expected parameter: 'L/E'. Make sure its in your model section of XML.") % __func__ % __LINE__;
            throw std::runtime_error("Missing parameter: L/E");
        }
        ivar = parameter_map.at("L/E");

        //constraints
        model_constraint = [this](const Eigen::VectorXf &v){return this->UnitarityConstraint(v);};


         size_t nvar = prop.variable_mc_stat_err.size();
        hists.resize(nvar);
        for(size_t v = 0; v <nvar ;v++){
            for(size_t m = 0; m < model_functions.size(); ++m) {
                hists.at(v).emplace_back(Eigen::MatrixXf::Constant(prop.variable_hist_storage(ivar,v).rows(), prop.variable_hist_storage(ivar,v).cols(),0.0));
                Eigen::MatrixXf &h = hists.at(v).back();
                for(size_t i = 0; i < prop.NEvent(); ++i) {
                    if(prop.model_rule[i] != (int)m) continue;
                    int tbin = prop.VariableBinIndex(ivar, i), rbin = prop.VariableBinIndex(v, i);
                    if(tbin<0 || rbin<0)continue;
                    h(tbin, rbin) += prop.added_weights[i];
                }
            }
        }


        nparams = 3;
        param_names = {"dmsq", "sinsq2thmue", "xi"}; 
        pretty_param_names = {"#Deltam^{2}", "sin^{2}2#theta_{#mue}", "#xi"}; 
        lb = Eigen::VectorXf(3);
        ub = Eigen::VectorXf(3);
        default_val = Eigen::VectorXf(3);
        lb << -2, -std::numeric_limits<float>::infinity(), -10;
        ub << 2, -1e-3, 10;
        //default_val << -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity();
        default_val << -2, -8, 0;
    };

    int UnitarityConstraint(const Eigen::VectorXf &v){
        return   (sqrt(pow(10,v(1)))*cosh(v(2))<1 ? 1 : 0);      
    }

    float Pmue(float dmsq, float sinsq2thmue, [[maybe_unused]]float xi, float le) const{
        dmsq = std::pow(10.0f, dmsq);
        sinsq2thmue = std::pow(10.0f, sinsq2thmue);
        
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

        float sinterm = std::sin(1.27f*dmsq*(le));
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
        dmsq = std::pow(10.0f, dmsq);
        sinsq2thmue = std::pow(10.0f, sinsq2thmue);

        float Um4sq=(std::exp(xi) * std::sqrt(sinsq2thmue)) / 2.0;
        
        float sinterm = std::sin(1.27*dmsq*(le));
        float prob    = 1.0f - 4.0f*Um4sq*(1.0f-Um4sq)*sinterm*sinterm;

        if(prob<0.0 || prob >1.0){
            log<LOG_ERROR>(L"%1% || Your probability %2% is outside the bounds of math. dmsq = %3%, Um4sq = %4%, L/E = %5%") % __func__ % prob % dmsq % Um4sq % le;
            log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
            exit(EXIT_FAILURE);
        }

        return prob;
    }

    float Pee(float dmsq, float sinsq2thmue, float xi, float le) const{
        dmsq = std::pow(10.0f, dmsq);
        sinsq2thmue = std::pow(10.0f, sinsq2thmue);

        float Ue4sq=(std::exp(-xi) * std::sqrt(sinsq2thmue)) / 2.0;

        float sinterm = std::sin(1.27*dmsq*(le));
        float prob    = 1.0f - 4.0f*Ue4sq*(1.0f-Ue4sq)*sinterm*sinterm;

        if(prob<0.0 || prob >1.0){
            log<LOG_ERROR>(L"%1% || Your probability %2% is outside the bounds of math. dmsq = %3%, Ue4sq = %4%, L/E = %5%") % __func__ % prob % dmsq % Ue4sq % le;
            log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
            exit(EXIT_FAILURE);
        }

        return prob;
    }
};


// Main interface to different models
static inline
std::unique_ptr<PROmodel> get_model_from_string(const PROconfig& config, const PROpeller &prop) {
     std::string name = config.m_model_tag;

     if(name == "numudis") {
        return std::unique_ptr<PROmodel>(new PROnumudis(prop,config.m_model_parameter_map));
    } else if(name == "nueapp") {
        return std::unique_ptr<PROmodel>(new PROnueapp(prop,config.m_model_parameter_map));
    } else if(name == "3+1") {
        return std::unique_ptr<PROmodel>(new PRO3p1(prop,config.m_model_parameter_map));
    } else if(name == "3+1_3C") {
        return std::unique_ptr<PROmodel>(new PRO3p1_3C(prop,config.m_model_parameter_map));
    }
    log<LOG_ERROR>(L"%1% || Unrecognized model name %2%. Try numudis, nueapp, 3+1 or 3+1_3C for now. Terminating.") % __func__ % name.c_str();
    exit(EXIT_FAILURE);
}

}

#endif

