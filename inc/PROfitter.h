#ifndef PROFITTER_H
#define PROFITTER_H

#include "PROmetric.h"
#include "PROgress.h"

#include <Eigen/Eigen>
#include "LBFGSB.h"

namespace PROfit {

    struct PROfitterConfig {
        
        //Parameters from LBFGSB only
        LBFGSpp::LBFGSBParam<float> param;
        
        /***PROfit fitter parameters**/
    
        //Number of latin hypercube points to sample across ALL parameters, physics and otherwise
        int n_latin_points = 1500;
        float latin_diversity_factor =0.5;

        //Number of partile swarm particels, and how many iterations. 
        //taken from best (aka lowest chi, with iverse greedy seleciton) hypercube points
        int n_swarm_particles = 1;
        int n_swarm_iterations=1;
        int n_swarm_max_stagnent_iterations=50;
        float swarm_inertia_start = 0.9; 
        float swarm_inertia_end= 0.6; 
        float swarm_cognitive_score= 2.0; 
        float swarm_social_score = 2.0; 
        float swarm_convergence_theshold = 1e-4;

        //Total number of LBFGSB fits to do after PSO.
        int n_localfit=10;
        //Now many times to retry if LBFBSG has exception. Currently not really worth it until we cod ebetter logic for "changing" things after a fail. 
        size_t n_max_local_retries = 1;

        //MCMC parameters
        size_t MCMCiter = 20'000;
        size_t MCMCburn = 25'000;

        //Parameters in frequency based harmonic seed search
        size_t harmonic_min_num_seeds = 2;//4
        size_t harmonic_max_num_seeds = 4;//15
        size_t harmonic_num_test_points = 125;
        size_t harmonic_raw_max_tests = 60;
        float harmonic_prominence_threshold = 0.5;
        float harmonic_prominence_threshold_shift = 0.2;
        float harmonic_min_spacing_log = 0.025;
        float harmonic_prominence_threshold_minimum = 1e-5;
        float harmonic_seed_norm_tolerance = 1e-4;  
        float harmonic_seed_chi_tolerence = 1e-6;   
        bool harmonic_scan_fit  = false; 

        bool progress_bar = false;

        PROfitterConfig(){};
        PROfitterConfig(std::map<std::string, float> input_fit_options, std::string fit_preset, bool isScan){

            if(fit_preset == "good"){
                param.epsilon = 1e-5;
                param.epsilon_rel = 1e-5;
                param.max_iterations = 150;
                param.max_linesearch = 20;
                param.delta = 1e-6;
                param.wolfe = 0.90;
                param.ftol = 1e-4;
                param.m = 6;
                param.max_submin =10;
                param.min_step = std::numeric_limits<float>::epsilon();

                n_latin_points = 2000;
                n_swarm_particles = 25;
                n_swarm_iterations = 100;
                n_localfit=8;
                 
            }else if (fit_preset == "fast"){
                //typically used for scans
                param.epsilon = 1e-5;
                param.epsilon_rel = 1e-5;
                param.max_iterations = 100;
                param.max_linesearch = 20;
                param.delta = 1e-6;
                param.wolfe = 0.90;
                param.ftol = 1e-4;
                param.m = 6;
                param.max_submin =10;
                param.min_step = std::numeric_limits<float>::epsilon();

                n_latin_points = 750;
                n_swarm_particles = 15;
                n_swarm_iterations = 100;
                n_localfit=4;

            }else if(fit_preset == "overkill"){
                param.epsilon = 1e-5;
                param.epsilon_rel = 1e-5;
                param.max_iterations = 250;
                param.max_linesearch = 30;
                param.delta = 1e-6;
                param.wolfe = 0.90;
                param.ftol = 1e-4;
                param.m = 6;
                param.max_submin =10;
                param.min_step = std::numeric_limits<float>::epsilon();

                n_latin_points = 5000;
                n_swarm_particles = 100;
                n_swarm_iterations = 300;
                n_localfit=15;

                harmonic_min_num_seeds = 3;//4
                harmonic_max_num_seeds = 8;
                harmonic_num_test_points = 200;
            }



            std::string whichFit = ( isScan? "Simplier Scan" : "Detailed Global");
            log<LOG_INFO>(L"%1% ||Fit and  L-BFGS-B parameters for the %2% minimia finder.  ") % __func__ % whichFit.c_str();
            for(const auto &[param_name, value]: input_fit_options) {
                log<LOG_WARNING>(L"%1% || L-BFGS-B %2% set to %3% ") % __func__% param_name.c_str() % value ;
                
                // L-BFGS-B parameters
                if(param_name == "epsilon") {
                    param.epsilon = value;
                } else if(param_name == "delta") {
                    param.delta = value;
                } else if(param_name == "m") {
                    param.m = value;
                    if(value < 3) {
                        log<LOG_WARNING>(L"%1% || Number of corrections to approximate inverse Hessian in"
                                L" L-BFGS-B is recommended to be at least 3, provided value is %2%."
                                L" Note: this is controlled via --fit-options m.")
                            % __func__ % value;
                    }
                } else if(param_name == "epsilon_rel") {
                    param.epsilon_rel = value;
                } else if(param_name == "past") {
                    param.past = value;
                    if(value == 0) {
                        log<LOG_WARNING>(L"%1% || L-BFGS-B 'past' parameter set to 0. This will disable delta convergence test")
                            % __func__;
                    }
                } else if(param_name == "max_iterations") {
                    param.max_iterations = value;
                } else if(param_name == "max_submin") {
                    param.max_submin = value;
                } else if(param_name == "max_linesearch") {
                    param.max_linesearch = value;
                } else if(param_name == "min_step") {
                    param.min_step = value;
                    log<LOG_WARNING>(L"%1% || Modifying the minimum step size in the line search to be %2%."
                            L" This is not usually needed according to the LBFGSpp documentation.")
                        % __func__ % value;
                } else if(param_name == "max_step") {
                    param.max_step = value;
                    log<LOG_WARNING>(L"%1% || Modifying the maximum step size in the line search to be %2%."
                            L" This is not usually needed according to the LBFGSpp documentation.")
                        % __func__ % value;
                } else if(param_name == "ftol") {
                    param.ftol = value;
                } else if(param_name == "wolfe") {
                    param.wolfe = value;
                    
                // PROfitter specific parameters
                } else if(param_name == "n_latin_points") {
                    n_latin_points = value;
                    if(n_latin_points < 1) {
                        log<LOG_ERROR>(L"%1% || Expected to run at least 1 multistart point. Provided value is %2%.")
                            % __func__ % value;
                        exit(EXIT_FAILURE);
                    }
                } else if(param_name == "latin_diversity_factor") {
                    latin_diversity_factor = value;
                    if(latin_diversity_factor > 1 || latin_diversity_factor<0) {
                        log<LOG_ERROR>(L"%1% || Latin Diversity Factor must be between 0 and 1.  Provided value is %2%.")
                            % __func__ % value;
                        exit(EXIT_FAILURE);
                    }

                } else if(param_name == "n_localfit") {
                    n_localfit = value;
                    if(n_localfit < 1) {
                        log<LOG_ERROR>(L"%1% || Expected to run at least 1 local fit point. Provided value is %2%.")
                            % __func__ % value;
                        exit(EXIT_FAILURE);
                    }
                } else if(param_name == "n_max_local_retries") {
                    n_max_local_retries = value;
                    
                // Particle Swarm Optimization parameters
                } else if(param_name == "n_swarm_particles") {
                    n_swarm_particles = value;
                    if(n_swarm_particles < 1) {
                        log<LOG_ERROR>(L"%1% || Expected to run at least 1 PSO swarm particle point. Provided value is %2%.")
                            % __func__ % value;
                        exit(EXIT_FAILURE);
                    }
                } else if(param_name == "n_swarm_iterations") {
                    n_swarm_iterations = value;
                    if(n_swarm_iterations < 1) {
                        log<LOG_ERROR>(L"%1% || Expected to run at least 1 swarm_iterations point. Provided value is %2%.")
                            % __func__ % value;
                        exit(EXIT_FAILURE);
                    }
                } else if(param_name == "n_swarm_max_stagnent_iterations") {
                    n_swarm_max_stagnent_iterations = value;
                } else if(param_name == "swarm_inertia_start") {
                    swarm_inertia_start = value;
                } else if(param_name == "swarm_inertia_end") {
                    swarm_inertia_end = value;
                } else if(param_name == "swarm_cognitive_score") {
                    swarm_cognitive_score = value;
                } else if(param_name == "swarm_social_score") {
                    swarm_social_score = value;
                } else if(param_name == "swarm_convergence_threshold") {
                    swarm_convergence_theshold = value;
                    
                // MCMC parameters
                } else if(param_name == "MCMC-Burnin") {
                    MCMCburn = value;
                    if(MCMCburn < 1) {
                        log<LOG_WARNING>(L"%1% || Warning: Running without any burnin for MCMC.") % __func__ ;
                    }
                } else if(param_name == "MCMC-Iterations") {
                    MCMCiter = value;
                    if(MCMCiter < 1) {
                        log<LOG_ERROR>(L"%1% || Requested to run MCMC with no iterations.") % __func__ ;
                    }
                    
                // Harmonic seed search parameters
                } else if(param_name == "harmonic_min_num_seeds") {
                    harmonic_min_num_seeds = value;
                } else if(param_name == "harmonic_max_num_seeds") {
                    harmonic_max_num_seeds = value;
                } else if(param_name == "harmonic_num_test_points") {
                    harmonic_num_test_points = value;
                } else if(param_name == "harmonic_raw_max_tests") {
                    harmonic_raw_max_tests = value;
                } else if(param_name == "harmonic_prominence_threshold") {
                    harmonic_prominence_threshold = value;
                } else if(param_name == "harmonic_prominence_threshold_shift") {
                    harmonic_prominence_threshold_shift = value;
                } else if(param_name == "harmonic_min_spacing_log") {
                    harmonic_min_spacing_log = value;
                } else if(param_name == "harmonic_prominence_threshold_minimum") {
                    harmonic_prominence_threshold_minimum = value;
                } else if(param_name == "harmonic_seed_norm_tolerance") {
                    harmonic_seed_norm_tolerance = value;
                } else if(param_name == "harmonic_seed_chi_tolerence") {
                    harmonic_seed_chi_tolerence = value;
                } else if(param_name == "harmonic_scan_fit") {
                    harmonic_scan_fit = bool(value);
                } else {
                    log<LOG_WARNING>(L"%1% || Unrecognized parameter %2%. Will ignore.") 
                        % __func__ % param_name.c_str();
                }
            }
            try {
                print();
            } catch(std::invalid_argument &except) {
                log<LOG_ERROR>(L"%1% || Invalid L-BFGS-B parameters: %2%") % __func__ % except.what();
                log<LOG_ERROR>(L"Terminating.");
                exit(EXIT_FAILURE);
            }

        }

        void print(){
            log<LOG_INFO>(L"%1% || Printing PROfitterConfig Values.") % __func__;
            
            log<LOG_INFO>(L"%1% || ------------ PROfitter specific -------------- ") % __func__ ;
            log<LOG_INFO>(L"%1% || n_latin_points: %2% (default 1500) ") % __func__ % n_latin_points;
            log<LOG_INFO>(L"%1% || latin_diversity_factor: %2% (default 0.5) ") % __func__ % latin_diversity_factor;
            log<LOG_INFO>(L"%1% || n_localfit: %2% (default 10) ") % __func__ % n_localfit;
            log<LOG_INFO>(L"%1% || n_max_local_retries: %2% (default 1) ") % __func__ % n_max_local_retries;
            
            log<LOG_INFO>(L"%1% || ------------ Particle Swarm Optimization -------------- ") % __func__ ;
            log<LOG_INFO>(L"%1% || n_swarm_particles: %2% (default 25) ") % __func__ % n_swarm_particles;
            log<LOG_INFO>(L"%1% || n_swarm_iterations: %2% (default 250) ") % __func__ % n_swarm_iterations;
            log<LOG_INFO>(L"%1% || n_swarm_max_stagnent_iterations: %2% (default 50) ") % __func__ % n_swarm_max_stagnent_iterations;
            log<LOG_INFO>(L"%1% || swarm_inertia_start: %2% (default 0.9) ") % __func__ % swarm_inertia_start;
            log<LOG_INFO>(L"%1% || swarm_inertia_end: %2% (default 0.6) ") % __func__ % swarm_inertia_end;
            log<LOG_INFO>(L"%1% || swarm_cognitive_score: %2% (default 2.0) ") % __func__ % swarm_cognitive_score;
            log<LOG_INFO>(L"%1% || swarm_social_score: %2% (default 2.0) ") % __func__ % swarm_social_score;
            log<LOG_INFO>(L"%1% || swarm_convergence_threshold: %2% (default 1e-4) ") % __func__ % swarm_convergence_theshold;
            
            log<LOG_INFO>(L"%1% || ------------ MCMC Parameters -------------- ") % __func__ ;
            log<LOG_INFO>(L"%1% || MCMCiter: %2% (default 20000) ") % __func__ % MCMCiter;
            log<LOG_INFO>(L"%1% || MCMCburn: %2% (default 25000) ") % __func__ % MCMCburn;
            
            log<LOG_INFO>(L"%1% || ------------ Harmonic Seed Search -------------- ") % __func__ ;
            log<LOG_INFO>(L"%1% || harmonic_min_num_seeds: %2% (default 2 ) ") % __func__ % harmonic_min_num_seeds;
            log<LOG_INFO>(L"%1% || harmonic_max_num_seeds: %2% (default 4 ) ") % __func__ % harmonic_max_num_seeds;
            log<LOG_INFO>(L"%1% || harmonic_num_test_points: %2% (default 100) ") % __func__ % harmonic_num_test_points;
            log<LOG_INFO>(L"%1% || harmonic_raw_max_tests: %2% (default 60) ") % __func__ % harmonic_raw_max_tests;
            log<LOG_INFO>(L"%1% || harmonic_prominence_threshold: %2% (default 0.5) ") % __func__ % harmonic_prominence_threshold;
            log<LOG_INFO>(L"%1% || harmonic_prominence_threshold_shift: %2% (default 0.05) ") % __func__ % harmonic_prominence_threshold_shift;
            log<LOG_INFO>(L"%1% || harmonic_min_spacing_log: %2% (default 0.025) ") % __func__ % harmonic_min_spacing_log;
            log<LOG_INFO>(L"%1% || harmonic_prominence_threshold_minimum: %2% (default 1e-7) ") % __func__ % harmonic_prominence_threshold_minimum;
            log<LOG_INFO>(L"%1% || harmonic_seed_norm_tolerance: %2% (default 1e-4) ") % __func__ % harmonic_seed_norm_tolerance;
            log<LOG_INFO>(L"%1% || harmonic_seed_chi_tolerence: %2% (default 1e-6) ") % __func__ % harmonic_seed_chi_tolerence;
            log<LOG_INFO>(L"%1% || harmonic_scan_fit: %2% (default false) ") % __func__ % harmonic_scan_fit;
            
            log<LOG_INFO>(L"%1% || ------------ LBFGSBParam -------------- ") % __func__ ;
            log<LOG_INFO>(L"%1% || m: %2%  (default %3%) ") % __func__ % param.m % 6;
            log<LOG_INFO>(L"%1% || epsilon: %2%  (default %3%) ") % __func__ % param.epsilon % 1e-5;
            log<LOG_INFO>(L"%1% || epsilon_rel: %2%  (default %3%) ") % __func__ % param.epsilon_rel % 1e-5;
            log<LOG_INFO>(L"%1% || past: %2%  (default %3%) ") % __func__ % param.past % 1;
            log<LOG_INFO>(L"%1% || delta: %2%  (default %3%) ") % __func__ % param.delta % 1e-10;
            log<LOG_INFO>(L"%1% || max_iterations: %2%  (default %3%) ") % __func__ % param.max_iterations % 0;
            log<LOG_INFO>(L"%1% || max_submin: %2%  (default %3%) ") % __func__ % param.max_submin % 20;
            log<LOG_INFO>(L"%1% || max_linesearch: %2%  (default %3%) ") % __func__ % param.max_linesearch % 20;
            log<LOG_INFO>(L"%1% || min_step: %2%  (default %3%) ") % __func__ % param.min_step % 1e-20;
            log<LOG_INFO>(L"%1% || max_step: %2%  (default %3%) ") % __func__ % param.max_step % 1e20;
            log<LOG_INFO>(L"%1% || ftol: %2%  (default %3%) ") % __func__ % param.ftol % 1e-4;
            log<LOG_INFO>(L"%1% || wolfe: %2%  (default %3%) ") % __func__ % param.wolfe % 0.9;
            log<LOG_INFO>(L"%1% || ------------ Check Param LBFGSBParam Below -------------- ") % __func__ ;
            param.check_param();

        }
        
        static void PrintHelp(){
            log<LOG_INFO>(L"==================================================================");
            log<LOG_INFO>(L"                  PROfitter Configuration Help                    ");
            log<LOG_INFO>(L"==================================================================");
            
            log<LOG_INFO>(L"");
            log<LOG_INFO>(L"------ PROfitter Specific Parameters ------");
            log<LOG_INFO>(L"  n_latin_points                       : Number of Latin hypercube points to sample across all parameters");
            log<LOG_INFO>(L"  latin_diversity_factor               : Diversity of latin points, 0: no distance weighting, 1: select most diverse far away points");
            log<LOG_INFO>(L"  n_localfit                           : Total number of L-BFGS-B fits to do after PSO");
            log<LOG_INFO>(L"  n_max_local_retries                  : Maximum retries if L-BFGS-B throws an exception");
            
            log<LOG_INFO>(L"");
            log<LOG_INFO>(L"------ Particle Swarm Optimization (PSO) Parameters ------");
            log<LOG_INFO>(L"  n_swarm_particles                    : Number of particles in the swarm (taken from best hypercube points)");
            log<LOG_INFO>(L"  n_swarm_iterations                   : Maximum number of PSO iterations");
            log<LOG_INFO>(L"  n_swarm_max_stagnent_iterations      : Stop PSO after this many iterations without improvement");
            log<LOG_INFO>(L"  swarm_inertia_start                  : Initial inertia weight for PSO velocity update");
            log<LOG_INFO>(L"  swarm_inertia_end                    : Final inertia weight (linearly decreased during optimization)");
            log<LOG_INFO>(L"  swarm_cognitive_score                : Weight for particle's personal best in velocity update");
            log<LOG_INFO>(L"  swarm_social_score                   : Weight for global best in velocity update");
            log<LOG_INFO>(L"  swarm_convergence_threshold          : Convergence criterion for PSO");
            
            log<LOG_INFO>(L"");
            log<LOG_INFO>(L"------ MCMC Parameters ------");
            log<LOG_INFO>(L"  MCMC-Iterations                      : Number of MCMC iterations to run");
            log<LOG_INFO>(L"  MCMC-Burnin                          : Number of burn-in iterations for MCMC");
            
            log<LOG_INFO>(L"");
            log<LOG_INFO>(L"------ Harmonic Seed Search Parameters ------");
            log<LOG_INFO>(L"  harmonic_min_num_seeds               : Minimum number of seed points for harmonic search");
            log<LOG_INFO>(L"  harmonic_max_num_seeds               : Maximum number of seed points for harmonic search");
            log<LOG_INFO>(L"  harmonic_num_test_points             : Number of test points in frequency space");
            log<LOG_INFO>(L"  harmonic_raw_max_tests               : Max number of iterations to find significant minima.");
            log<LOG_INFO>(L"  harmonic_prominence_threshold        : Threshold for peak prominence in harmonic search");
            log<LOG_INFO>(L"  harmonic_prominence_threshold_shift  : Shift amount for adjusting prominence threshold");
            log<LOG_INFO>(L"  harmonic_min_spacing_log             : Minimum spacing between peaks in log space");
            log<LOG_INFO>(L"  harmonic_prominence_threshold_minimum: Absolute minimum for prominence threshold");
            log<LOG_INFO>(L"  harmonic_seed_norm_tolerance         : Tolerance for seed point norm convergence");
            log<LOG_INFO>(L"  harmonic_seed_chi_tolerence          : Tolerance for chi-squared convergence in seed search");
            log<LOG_INFO>(L"  harmonic_scan_fit                    : During harmonic scan, fit per point (true) or hold at BF (false/default)");
            
            log<LOG_INFO>(L"");
            log<LOG_INFO>(L"------ L-BFGS-B Parameters ------");
            log<LOG_INFO>(L"  m                                    : Number of corrections to approximate inverse Hessian (recommend >= 3)");
            log<LOG_INFO>(L"  epsilon                              : Tolerance for gradient norm convergence test");
            log<LOG_INFO>(L"  epsilon_rel                          : Relative tolerance for gradient norm");
            log<LOG_INFO>(L"  past                                 : Number of past iterations for delta convergence test (0 disables)");
            log<LOG_INFO>(L"  delta                                : Tolerance for delta convergence test");
            log<LOG_INFO>(L"  max_iterations                       : Maximum number of L-BFGS-B iterations (0 = unlimited)");
            log<LOG_INFO>(L"  max_submin                           : Maximum number of trials for subspace minimization");
            log<LOG_INFO>(L"  max_linesearch                       : Maximum number of line search iterations");
            log<LOG_INFO>(L"  min_step                             : Minimum step size in line search");
            log<LOG_INFO>(L"  max_step                             : Maximum step size in line search");
            log<LOG_INFO>(L"  ftol                                 : Tolerance for sufficient decrease condition in line search");
            log<LOG_INFO>(L"  wolfe                                : Coefficient for Wolfe condition (0 < wolfe < 1)");
            
            log<LOG_INFO>(L"");
            log<LOG_INFO>(L"==================================================================");
            log<LOG_INFO>(L"Note: Use --fit-options <param_name> <value> to set parameters");
            log<LOG_INFO>(L"Example: --fit-options n_latin_points 2000 max_iterations 5000");
            log<LOG_INFO>(L"==================================================================");
        }
    };

    class PROfitter {
        public:
            Eigen::VectorXf ub, lb, best_fit;
            PROfitterConfig fitconfig;
            LBFGSpp::LBFGSBSolver<float> solver;
            uint32_t seed;
            std::map<std::string,size_t> exception_string_map;
            MultiPROgressBar * progress;
            bool run_progress;


            std::vector<Eigen::VectorXf> freq_seed_points;
            std::vector<float> freq_seed_values;

            PROfitter(const Eigen::VectorXf ub, const Eigen::VectorXf lb, PROfitterConfig fitconfig_ = {}, uint32_t inseed = 0)
                : ub(ub), lb(lb), fitconfig(fitconfig_), solver(fitconfig.param), seed(inseed) {}

            float Fit(PROmetric &metric, const Eigen::VectorXf &seed_pt = Eigen::VectorXf());
            float Fit(PROmetric &metric, const std::vector<Eigen::VectorXf> &seed_points );

            int calcFreqSeedPoints(PROmetric &metric);
            std::vector<std::pair<float, float>> findSignificantMinima(  const std::vector<float>& x_values,const std::vector<float>& y_values, bool use_log_spacing = true);
            

            void setProgressBar(MultiPROgressBar* pin){
                    run_progress = true;
                    progress = pin;
                    return;
            }
            Eigen::VectorXf FinalGradient() const {return solver.final_grad();}
            float FinalGradientNorm() const {return solver.final_grad_norm();}
            Eigen::MatrixXf Hessian() const {return solver.final_approx_hessian();}
            Eigen::MatrixXf InverseHessian() const {return solver.final_approx_inverse_hessian();}
            Eigen::MatrixXf Covariance() const {return InverseHessian();}
            Eigen::VectorXf BestFit() const {return best_fit;}

            // If you don't belive the uncertainties on the parameters, you can use the final fit value to estimate the variance
            Eigen::MatrixXf ScaledCovariance(float chi2, int n_datapoint) const {return Covariance()*chi2/float(n_datapoint-best_fit.size());}


    };

}

#endif
