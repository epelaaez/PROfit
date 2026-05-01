#include "PROfitter.h"
#include "PROlog.h"
#include "PROmetric.h"
#include "PROswarm.h"
#include <Eigen/Eigen>

#include <random>

using namespace PROfit;

std::vector<std::vector<float>> PROfit::latin_hypercube_sampling(size_t num_samples, size_t dimensions, std::uniform_real_distribution<float>&dis, std::mt19937 &gen) {
    std::vector<std::vector<float>> samples(num_samples, std::vector<float>(dimensions));

    for (size_t d = 0; d < dimensions; ++d) {

        std::vector<float> perm(num_samples);
        for (size_t i = 0; i < num_samples; ++i) {
            perm[i] = (i + dis(gen)) / num_samples;  
        }
        std::shuffle(perm.begin(), perm.end(), gen);  
        for (size_t i = 0; i < num_samples; ++i) {
            samples[i][d] = perm[i]; 
        }
    }

    return samples;
}

void PROfit::recenter_latin_samples(std::vector<std::vector<float>> &samples, const Eigen::VectorXf &ub, const Eigen::VectorXf &lb) {
    for(std::vector<float> &pt: samples) {
        for(size_t i = 0; i < pt.size(); ++i) {
            if(ub(i) != 3 || lb(i) != -3) {
                float width = std::isinf(ub(i)) || std::isinf(lb(i)) ? 4 : ub(i) - lb(i);
                float center = std::isinf(ub(i)) ? lb(i) + width/2.0 :
                    std::isinf(lb(i)) ? ub(i) - width/2.0 :
                    (ub(i) + lb(i)) / 2.0;
                float randpt = pt[i] / 4.0;
                pt[i] = center + randpt * width;
            }
        }
    }
}

static inline
std::vector<int> sorted_indices(const std::vector<float>& vec) {
    std::vector<int> indices(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        indices[i] = i;
    }
    std::sort(indices.begin(), indices.end(), [&vec](int i1, int i2) { return vec[i1] < vec[i2]; });
    return indices;
}



static inline
std::vector<int> select_diverse_best(const std::vector<float>& chi2s, const std::vector<std::vector<float>>& points,size_t n_select,
                                     const Eigen::VectorXf& ub,const Eigen::VectorXf& lb,float diversity_factor = 0.3f) {
    
    if(n_select >= chi2s.size()) {
        return sorted_indices(chi2s);
    }
    
    std::vector<int> sorted_idx = sorted_indices(chi2s);
    
    std::vector<int> selected;
    selected.reserve(n_select);
    
    // always include the best point
    selected.push_back(sorted_idx[0]);
    
    // Use diversity_factor * average normalized dimension range
    size_t ndim = points[0].size();
    float threshold = diversity_factor / std::sqrt(static_cast<float>(ndim));
    
    for(size_t i = 1; i < sorted_idx.size() && selected.size() < n_select; ++i) {
        int candidate_idx = sorted_idx[i];
        bool is_diverse = true;
        
        for(int sel_idx : selected) {
            float normalized_dist_sq = 0.0;
            
            // Calculate normalized Euclidean distance
            for(size_t d = 0; d < ndim; ++d) {
                float range = (ub(d) - lb(d));
                if(std::isinf(range) || range <= 0) {
                    range = 1.0;  // Default normalization for unbounded dimensions
                }
                float diff = (points[candidate_idx][d] - points[sel_idx][d]) / range;
                normalized_dist_sq += diff * diff;
            }
            
            float normalized_dist = std::sqrt(normalized_dist_sq);
            
            if(normalized_dist < threshold) {
                is_diverse = false;
                break;
            }
        }
        
        if(is_diverse) {
            selected.push_back(candidate_idx);
        }
    }
    
    // i not enough, fill with best remaining
    if(selected.size() < n_select) {
        log<LOG_DEBUG>(L"%1% || Only found %2% diverse points out of requested %3% with diversity_factor=%4%")%  __func__ %  selected.size() % n_select % diversity_factor;
        
        for(size_t i = 0; i < sorted_idx.size() && selected.size() < n_select; ++i) {
            int idx = sorted_idx[i];
            if(std::find(selected.begin(), selected.end(), idx) == selected.end()) {
                selected.push_back(idx);
            }
        }
    }
    
    return selected;
}


float PROfitter::Fit(PROmetric &metric, const Eigen::VectorXf &seed_pt ) {
    const std::vector<Eigen::VectorXf> seed_points = seed_pt.size() > 0 ? std::vector<Eigen::VectorXf>{seed_pt} : std::vector<Eigen::VectorXf>{};
    return Fit(metric, seed_points);

}

float PROfitter::Fit(PROmetric &metric, const std::vector<Eigen::VectorXf> &seed_points ) {

    std::mt19937 rng;
    rng.seed(seed);
    std::normal_distribution<float> d;
    std::uniform_real_distribution<float> d_uni(-2.0, 2.0);

    //n_latin_points is how many initial latin cube points
    std::vector<std::vector<float>> latin_samples = latin_hypercube_sampling(fitconfig.n_latin_points, ub.size(), d_uni,rng);
    //Rescale the latin hypercube now at -2 to 2, scale to real bounds.
    for(std::vector<float> &pt: latin_samples) {
        for(size_t i = 0; i < pt.size(); ++i) {
            if(ub(i) != 3 || lb(i) != -3) {
                float width = std::isinf(ub(i)) || std::isinf(lb(i)) ? 4 : ub(i) - lb(i);
                float center = std::isinf(ub(i)) ? lb(i) + width/2.0 :
                    std::isinf(lb(i)) ? ub(i) - width/2.0 :
                    (ub(i) + lb(i)) / 2.0;
                float randpt = pt[i] / 4.0;
                pt[i] = center + randpt * width;
            }
        }
    }
    if(seed_points.size()>0 && seed_points.front().size()>0){
        log<LOG_INFO>(L"%1% || Seed point passed in. Being included.") % __func__  ;
        for(auto & pt: seed_points){
            std::vector<float> std_vec(pt.data(), pt.data() + pt.size());
            latin_samples.push_back(std_vec);
        }
    }else{
        log<LOG_INFO>(L"%1% || No seed point passed in. ") % __func__  ;
    }



    std::vector<float> chi2s_multistart;
    chi2s_multistart.reserve(fitconfig.n_latin_points);

    log<LOG_INFO>(L"%1% || Starting MultiGlobal runs (i.e latin hypercube runs, pure chi^2 no grad) : %2%") % __func__ % fitconfig.n_latin_points ;
    for(int s = 0; s < fitconfig.n_latin_points; s++){
        Eigen::VectorXf x = Eigen::Map<Eigen::VectorXf>(latin_samples[s].data(), latin_samples[s].size());
        Eigen::VectorXf grad = Eigen::VectorXf::Constant(x.size(), 0);
        float fx =  metric(x, grad, false);
        chi2s_multistart.push_back(fx);
        if(run_progress){progress->increment_bar(0);}

    }
    //Sort so we can take the best N_localfits for further zoning with a PSO
    //std::vector<int> best_multistart = sorted_indices(chi2s_multistart);    
    std::vector<int> best_multistart = select_diverse_best(chi2s_multistart, latin_samples, fitconfig.n_swarm_particles, ub, lb, fitconfig.latin_diversity_factor); 

    log<LOG_INFO>(L"%1% || Best Point after latin hypercube has chi^2 %2% with pts  : %3% ") % __func__ % chi2s_multistart[best_multistart[0]] % latin_samples[best_multistart[0]];

    std::string swarm_string = "";
    std::vector<std::vector<float>> swarm_start_points;
    int niter=0;
    float fx;
    if(fitconfig.n_swarm_particles < 1){
        fitconfig.n_swarm_particles = 1;
    }

    for(int s = 0; s < fitconfig.n_swarm_particles; s++){
        swarm_string += " " + std::to_string(chi2s_multistart[best_multistart[s]]);
        swarm_start_points.push_back(latin_samples[best_multistart[s]]);
    }
    log<LOG_INFO>(L"%1% || Will swarm with %2% swarm points chis of %3% ") % __func__ % fitconfig.n_swarm_particles % swarm_string.c_str();

    PROswarm PSO(metric, rng, swarm_start_points, lb, ub , fitconfig.n_swarm_iterations);
    if(run_progress){
        PSO.runSwarm(metric,rng,fitconfig, progress);
    }else{
        PSO.runSwarm(metric,rng,fitconfig);
    }


    Eigen::VectorXf x;  

    float chimin = 9999999;
    std::vector<float> chi2s_localfits;
    niter=0;

    bool success = false;
    best_fit = PSO.getGlobalBestPosition();
    chimin = PSO.getGlobalBestScore();


    log<LOG_INFO>(L"%1% || Starting local fit of best swarm point. ") % __func__ ;

    for (size_t attempt = 1; attempt <= fitconfig.n_max_local_retries; ++attempt) {
        if(run_progress)progress->increment_bar(2);

        try {
            x = PSO.getGlobalBestPosition();
            log<LOG_INFO>(L"%1% || Starting local minimization attempt %2%/%3%") % __func__ % attempt % fitconfig.n_max_local_retries;

            try{
                niter = solver.minimize(metric, x, fx, lb, ub);
            } catch(const std::exception &e) {
                std::string msg = e.what();
                exception_string_map[msg]++;
            }


            chi2s_localfits.push_back(fx);

            if (fx < chimin) {
                best_fit = x;
                chimin = fx;
            }

            log<LOG_INFO>(L"%1% || Minimization successful, chi %2% after %3% iterations") % __func__ % fx % niter;

            std::string spec_string = "";
            for (auto &f : x) spec_string += " " + std::to_string(f);
            log<LOG_DEBUG>(L"%1% || Best Point after minimization: %2%") % __func__ % spec_string.c_str();

            success = true;
            break;

        } catch (const std::runtime_error &except) {
            log<LOG_WARNING>(L"%1% || Minimization attempt %2%/%3% failed: %4%") % __func__ % attempt % fitconfig.n_max_local_retries % except.what();
        }
    }

    if (!success) {
        log<LOG_WARNING>(L"%1% || All minimization attempts failed, checking how good we got, otherwise falling back to PSO best") % __func__;

        log<LOG_WARNING>(L"%1% || PSO chi %2%  and local: %3% ") % __func__ % PSO.getGlobalBestScore() % fx;
        if (fx < chimin) {
             best_fit = x;
             chimin = fx;
        }
        if(PSO.getGlobalBestScore()< chimin){
            best_fit = PSO.getGlobalBestPosition();
            chimin = PSO.getGlobalBestScore();
        }
    }



    int fudge = 0;
    if(seed_points.size()>0 && seed_points.front().size()>0){
        fudge = seed_points.size();
        log<LOG_INFO>(L"%1% || Starting local fit of seed point. ") % __func__ ;

        if(run_progress)progress->increment_bar(2);
        for(size_t s = 0; s < seed_points.size();s++){
            for (size_t attempt = 1; attempt <= fitconfig.n_max_local_retries; ++attempt) {
                try {

                    x = seed_points.at(s);
                log<LOG_INFO>(L"%1% || Starting local minimization attempt %2%/%3%") % __func__ % attempt % fitconfig.n_max_local_retries;
            try{
                niter = solver.minimize(metric, x, fx, lb, ub);
            } catch(const std::exception &e) {
                std::string msg = e.what();
                exception_string_map[msg]++;
            }




                chi2s_localfits.push_back(fx);

                if (fx < chimin) {
                    best_fit = x;
                    chimin = fx;
                }

                log<LOG_INFO>(L"%1% || Minimization successful, chi %2% after %3% iterations") % __func__ % fx % niter;

                std::string spec_string = "";
                for (auto &f : x) spec_string += " " + std::to_string(f);
                log<LOG_DEBUG>(L"%1% || Best Point after minimization: %2%") % __func__ % spec_string.c_str();

                success = true;
                break;

            } catch (const std::runtime_error &except) {
                log<LOG_WARNING>(L"%1% || Minimization attempt %2%/%3% failed: %4%") % __func__ % attempt % fitconfig.n_max_local_retries % except.what();
            }
        }

        if (!success) {
            log<LOG_WARNING>(L"%1% || All minimization attempts failed, falling back to PSO best") % __func__;
            if (fx < chimin) {
                    best_fit = x;
                    chimin = fx;
            }


        }
    }
    }


    for(int i=0; i< fitconfig.n_localfit-1-fudge; i++){
        success = false;

        //After the best best fit, do you want to do more of the latin ones?
        x = Eigen::Map<Eigen::VectorXf>(latin_samples[best_multistart[i+1]].data(), latin_samples[best_multistart[i+1]].size());
        log<LOG_INFO>(L"%1% || Starting n_localfit local fit number %2%/%3% ") % __func__ % i  % fitconfig.n_localfit;

        if(run_progress)progress->increment_bar(2);

        for (size_t attempt = 1; attempt <= fitconfig.n_max_local_retries; ++attempt) {
            try {
                log<LOG_INFO>(L"%1% || Starting local minimization attempt %2%/%3%") % __func__ % attempt % fitconfig.n_max_local_retries;
                
                try{
                niter = solver.minimize(metric, x, fx, lb, ub);
            } catch(const std::exception &e) {
                std::string msg = e.what();
                exception_string_map[msg]++;
            }
                chi2s_localfits.push_back(fx);

                if (fx < chimin) {
                    best_fit = x;
                    chimin = fx;
                }

                log<LOG_INFO>(L"%1% || Minimization successful, chi %2% after %3% iterations") % __func__ % fx % niter;

                std::string spec_string = "";
                for (auto &f : x) spec_string += " " + std::to_string(f);
                log<LOG_DEBUG>(L"%1% || Best Point after minimization: %2%") % __func__ % spec_string.c_str();

                success = true;
                break;

            } catch (const std::runtime_error &except) {
                log<LOG_WARNING>(L"%1% || Minimization attempt %2%/%3% failed: %4%") % __func__ % attempt % fitconfig.n_max_local_retries % except.what();
            }
        }

        if (!success) {
            log<LOG_WARNING>(L"%1% || All minimization attempts failed. Hopefully the PSO worked above.") % __func__;
              if (fx < chimin) {
                    best_fit = x;
                    chimin = fx;
                }


        }
    }

    if (fx < chimin) {
        best_fit = x;
        chimin = fx;
    }

    log<LOG_INFO>(L"%1% || FINAL has a chi %2%") % __func__ %  chimin;
    std::string spec_string = "";
    for(auto &f : best_fit) spec_string+=" "+std::to_string(f); 
    log<LOG_INFO>(L"%1% || FINAL is  : %2% ") % __func__ % spec_string.c_str();

    return chimin;
}
int PROfitter::calcFreqSeedPoints(PROmetric &metric) {
    freq_seed_points.clear();
    freq_seed_values.clear();

    if(best_fit.size()==0){
        log<LOG_WARNING>(L"%1% || WARNING need to run a global fit using this PROfitter before asking to calculate Frequencey Seed Points.  ") %__func__;
        return 0;
    }

    size_t nparams = metric.GetModel().nparams + metric.GetSysts().GetNSplines();
    size_t nphys = metric.GetModel().nparams;

    if(nphys == 0){
        log<LOG_INFO>(L"%1% || No physics parameters in model, skipping frequency seed point calculation.") % __func__;
        return 0;
    }

    Eigen::VectorXf lb = Eigen::VectorXf::Constant(nparams, -3.0);
    Eigen::VectorXf ub = Eigen::VectorXf::Constant(nparams, 3.0);

    std::vector<float> chivalues;
    std::vector<float> chipos;

    //STEP 1, fix all syst values at best fit, and vary only physics
    Eigen::VectorXf local_candidate ;
    Eigen::VectorXf grad = Eigen::VectorXf::Constant(best_fit.size(),0);

    //TODO hardcoded 1 mass splittin for now
    size_t osc_par = 0;
    for(size_t i=0; i<nphys;i++){
        lb(i)=metric.GetModel().lb(i);
        ub(i)=metric.GetModel().ub(i);
    }

    //staggered test points, focus on active refion for efficiency
    float s1=0.0, s2=1.5;
    float w2= fabs(s2-s1)/float(fitconfig.harmonic_num_test_points);
    float w1=w2*10.0; 
    float w3 =w2*5.0;
    std::vector<float> test_p;
    for (float val = lb(osc_par); val <= s1; val += w1) test_p.push_back(val);
    for (float val = s1 + w2; val <= s2; val += w2) test_p.push_back(val);
    for (float val = s2 + w3; val <= ub(osc_par); val += w3) test_p.push_back(val);

    

    for(float &k : test_p){
        local_candidate = best_fit;
        local_candidate(osc_par) =k;
        Eigen::VectorXf temp_lb = local_candidate;
        Eigen::VectorXf temp_ub = local_candidate;

        //vary ONLY dm manually, and fit other_physics. Nuisence at BF
        for(size_t i=0; i<nphys;i++){
            if(i==osc_par) continue;
            temp_lb(i)=metric.GetModel().lb(i);
            temp_ub(i)=metric.GetModel().ub(i);
        }
        metric.setBounds(temp_lb,temp_ub);

        float fx = -9;

        if(fitconfig.harmonic_scan_fit){
            try{
                solver.minimize(metric, local_candidate, fx, temp_lb, temp_ub);
            } catch (const std::exception &except) {
                std::string msg = except.what();
                exception_string_map[msg]++;
            }
        }else{
           fx = metric(local_candidate,grad,false);
        }

        chivalues.push_back(fx);
        chipos.push_back(k);

        if(run_progress)progress->increment_bar(3);
       // log<LOG_INFO>(L"%1% || PARG  %2% %3% ") %__func__% k %  fx;
    }

    //#STEP 2 From the scan above, find local minima in freq
    std::vector<std::pair<float,float>> minima = findSignificantMinima(chipos, chivalues, true);


    //STEP 3, loop over all mimima and do twofold minimzation. 
    //First with DM minima fixed to get BF of pull terms, then fully free to optimize the mass splitting to high precisin


    for(size_t p=0;p<minima.size();p++){
        log<LOG_INFO>(L"%1% || ##################  ") %__func__;

        for(size_t i = 0; i < nphys; ++i) {
            lb(i) = metric.GetModel().lb(i);
            ub(i) = metric.GetModel().ub(i);
        }
        for(size_t i = nphys; i < nparams; ++i) {
            size_t si = i - nphys;
            lb(i) = metric.GetSysts().spline_has_restrict[si] ? metric.GetSysts().spline_restrict_lo[si] : metric.GetSysts().spline_lo[si];
            ub(i) = metric.GetSysts().spline_has_restrict[si] ? metric.GetSysts().spline_restrict_hi[si] : metric.GetSysts().spline_hi[si];
        }

        //fix dm at minima
        lb(osc_par)=minima.at(p).first;
        ub(osc_par)=minima.at(p).first;
        metric.setBounds(lb,ub);

        //Best fit, with minima points
        Eigen::VectorXf test_minima = best_fit;
        test_minima(osc_par) = minima.at(p).first;
        test_minima(1) = minima.at(p).second;

        float fx;
        LBFGSpp::LBFGSBSolver<float> solver(fitconfig.param);
        try{
            solver.minimize(metric, test_minima, fx, lb, ub);
        } catch (const std::exception &except) {
            std::string msg = except.what();
            exception_string_map[msg]++;
        }

        log<LOG_INFO>(L"%1% || Local min num (%2%) with FIXED frequency (%3%) has chi %4% ") %__func__% p % minima.at(p).first % fx;
        log<LOG_INFO>(L"%1% || -- at bf pt %2%  ") %__func__%  test_minima;

        //free the freq and repeat 
        lb(osc_par)=metric.GetModel().lb(osc_par);
        ub(osc_par)=metric.GetModel().ub(osc_par);
        metric.setBounds(lb,ub);

        try{
            solver.minimize(metric, test_minima, fx, lb, ub);
        } catch (const std::exception &except) {
            std::string msg = except.what();
            exception_string_map[msg]++;
        }

        log<LOG_INFO>(L"%1% || Local min num (%2%) with floating frequency (%3%) has chi %4% (but bf freq was %5%) ") %__func__% p % minima.at(p).first % fx % test_minima(osc_par);
        log<LOG_INFO>(L"%1% || -- at bf pt %2%  ") %__func__%  test_minima;

        freq_seed_points.push_back(test_minima);
        freq_seed_values.push_back(fx);
        if(run_progress){
            for(int jj=0;jj<std::ceil(100/minima.size())+1; jj++)progress->increment_bar(4);
        }
    }

    //ensure best fit is in the seeds, most likely mreoved in next step
    freq_seed_points.push_back(best_fit);
    float chibf = metric(best_fit, grad, false);
    freq_seed_values.push_back(chibf);

    log<LOG_INFO>(L"%1% || ##################  ") %__func__;
    log<LOG_DEBUG>(L"%1% || We have calculated %2% frequency seed points in total, plus 1 for global BF. Checking norm for differences! ") %__func__% freq_seed_values.size();

    std::vector<bool> keep(freq_seed_values.size(), true);
    float norm_tol = fitconfig.harmonic_seed_norm_tolerance; 
    float chi_tol = fitconfig.harmonic_seed_chi_tolerence;    

    for(size_t p = 0; p < freq_seed_values.size(); p++){
        if(!keep[p]) continue;  // needa to skip if already marked for del

        for(size_t q = p + 1; q < freq_seed_values.size(); q++){  
            if(!keep[q]) continue;

            float chip = freq_seed_values.at(p);
            float chiq = freq_seed_values.at(q);
            float normpq = (freq_seed_points.at(p) - freq_seed_points.at(q)).norm();

            log<LOG_DEBUG>(L"%1% || NORM freq seed (%2%,%3%) is : %4%, chi_diff: %5%") % __func__ % p % q % normpq % std::abs(chip - chiq);

            if(normpq < norm_tol && std::abs(chip - chiq) < chi_tol) {
                if(chip <= chiq) {
                    keep[q] = false;
                    log<LOG_DEBUG>(L"%1% || Removing duplicate %2% (keeping %3%)") % __func__ % q % p;
                } else {
                    keep[p] = false;
                    log<LOG_DEBUG>(L"%1% || Removing duplicate %2% (keeping %3%)") % __func__ % p % q;
                }
            }
        }
    }

    std::vector<Eigen::VectorXf> unique_points;
    std::vector<float> unique_values;
    for(size_t i = 0; i < keep.size(); i++) {
        if(keep[i]) {
            unique_points.push_back(freq_seed_points[i]);
            unique_values.push_back(freq_seed_values[i]);
        }
    }

    // Sort
    std::vector<size_t> sort_indices(unique_values.size());
    std::iota(sort_indices.begin(), sort_indices.end(), 0);
    std::sort(sort_indices.begin(), sort_indices.end(),[&](size_t i, size_t j) { return unique_values[i] < unique_values[j]; });

    std::vector<Eigen::VectorXf> sorted_points;
    std::vector<float> sorted_values;
    for(size_t idx : sort_indices) {
        sorted_points.push_back(unique_points[idx]);
        sorted_values.push_back(unique_values[idx]);
    }

    // Replace originals
    freq_seed_points = sorted_points;
    freq_seed_values = sorted_values;

    log<LOG_INFO>(L"%1% || Reduced from %2% to %3% unique seed points that are kept for future use!")   % __func__ % keep.size() % unique_points.size();

    return freq_seed_values.size();
}

std::vector<std::pair<float, float>> PROfitter::findSignificantMinima(const std::vector<float>& x_values, const std::vector<float>& y_values, bool use_log_spacing){
    if (x_values.size() != y_values.size() || x_values.size() < 3)
        return {};
    
    // Precompute derivative sign changes
    std::vector<float> dy(y_values.size() - 1);
    for (size_t i = 0; i < dy.size(); ++i)
        dy[i] = y_values[i + 1] - y_values[i];
    
    //Keep absolute for now
    //const float y_range = *std::max_element(y_values.begin(), y_values.end())- *std::min_element(y_values.begin(), y_values.end());
    //if (y_range == 0) return {};
    
    // Adaptive threshold loop
    float prominence_threshold = fitconfig.harmonic_prominence_threshold;//* y_range;
    float min_spacing = fitconfig.harmonic_min_spacing_log;
    
    std::vector<std::pair<float, float>> minima;
    size_t max_iter = fitconfig.harmonic_raw_max_tests;
    size_t iter = 0;
    
    // Keep trying with adjusted thresholds until we get the right number
    while (iter++ < max_iter) {
        minima.clear();
        
        std::vector<size_t> candidates;
        for (size_t i = 1; i < dy.size(); ++i) {
            if (dy[i - 1] < 0 && dy[i] > 0)
                candidates.push_back(i);
        }
        
        // now fiilter found minima by prominence
        std::vector<std::pair<size_t, float>> prominent_minima;
        
        for (size_t idx : candidates) {
            float y0 = y_values[idx];
            
            float left_peak = y0;
            for (int j = static_cast<int>(idx) - 1; j >= 0; --j) {
                left_peak = std::max(left_peak, y_values[j]);
                if (y_values[j] > y0 + 3 * prominence_threshold)
                    break;
            }
            
            float right_peak = y0;
            for (size_t j = idx + 1; j < y_values.size(); ++j) {
                right_peak = std::max(right_peak, y_values[j]);
                if (y_values[j] > y0 + 3 * prominence_threshold)
                    break;
            }
            
            float prominence = std::min(left_peak - y0, right_peak - y0);
            if (prominence >= prominence_threshold) {
                prominent_minima.emplace_back(idx, prominence);
            }
        }
        
        // remove minima that are too close together
        std::sort(prominent_minima.begin(), prominent_minima.end(),[&y_values](const auto& a, const auto& b) { 
                return y_values[a.first] < y_values[b.first]; 
        });
        
        std::vector<bool> keep(prominent_minima.size(), true);
        
        for (size_t i = 0; i < prominent_minima.size(); ++i) {
            if (!keep[i]) continue;
            
            size_t idx_i = prominent_minima[i].first;
            float x_i = x_values[idx_i];
            
            for (size_t j = i + 1; j < prominent_minima.size(); ++j) {
                if (!keep[j]) continue;
                
                size_t idx_j = prominent_minima[j].first;
                float x_j = x_values[idx_j];
                
                float spacing;
                if (use_log_spacing && x_i > 0 && x_j > 0) {
                    spacing = std::abs(std::log10(x_i) - std::log10(x_j));
                } else {
                    spacing = std::abs(x_i - x_j);
                }
                
                if (spacing < min_spacing) {
                    keep[j] = false;
                }
            }
        }
        
        // Collect final minima
        for (size_t i = 0; i < prominent_minima.size(); ++i) {
            if (keep[i]) {
                size_t idx = prominent_minima[i].first;
                minima.emplace_back(x_values[idx], y_values[idx]);
            }
        }
        
        // Check if we have enough minima
        if (minima.size() >= fitconfig.harmonic_min_num_seeds) {
           break;  // We have enough! If too many, we'll truncate later
         }
         
        // otheriwwse lets adjust thresholds for next iteration
        if (minima.size() < fitconfig.harmonic_min_num_seeds) {
            // Need more minima - relax thresholds
            prominence_threshold *= (1.0f - fitconfig.harmonic_prominence_threshold_shift);
            min_spacing *= (1.0f - fitconfig.harmonic_prominence_threshold_shift * 0.5f);
        };
    }
    
    std::sort(minima.begin(), minima.end(), [](const auto& a, const auto& b) { return a.second < b.second; });
   
    // Limit to maximum number if we still have too many
    if (minima.size() > fitconfig.harmonic_max_num_seeds) {
         minima.resize(fitconfig.harmonic_max_num_seeds);
         log<LOG_INFO>(L"%1% || Truncated to %2% best minima from %3% found")      % __func__ % fitconfig.harmonic_max_num_seeds % minima.size();
    }
    // Log results
    log<LOG_INFO>(L"%1% || Found %2% significant minima with prominence threshold %3%")  % __func__ % minima.size() % prominence_threshold;
    
    return minima;
}

