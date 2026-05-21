#include "PROsurf.h"
#include "PROfitter.h"
#include "PROlog.h"

#include <Eigen/Eigen>

#include <cmath>
#include <future>
#include <algorithm>
#include <functional>

#include "TGraph.h"
#include "TLatex.h"
#include "TLine.h"
#include "TMarker.h"
#include "TPaveText.h"
#include "TH2D.h"

using namespace PROfit;

std::vector<float> linspace(float start, float end, int N, bool endpoint = true) {
    std::vector<float> result;
    result.reserve(N);
    if (N == 0) return result;
    if (N == 1) {
        result.push_back(start);
        return result;
    }

    float step = (end - start) / (endpoint ? (N - 1) : N);
    for (int i = 0; i < N; ++i) {
        result.push_back(start + i * step);
    }
    return result;
}
std::vector<float> combined_sparse_dense(float Amin, float Amax, float CV, int Nsparse, int Ndense, float dense_width) {
    std::vector<float> sparse = linspace(Amin, Amax, Nsparse);  // global scan
    std::vector<float> dense = linspace(CV - dense_width / 2.0, CV + dense_width / 2.0, Ndense-1);  // local dense
    dense.push_back(CV);

    sparse.insert(sparse.end(), dense.begin(), dense.end());
    std::sort(sparse.begin(), sparse.end());

    sparse.erase(std::unique(sparse.begin(), sparse.end(),[](float a, float b) { return std::fabs(a - b) < 1e-8; }), sparse.end());

    return sparse;
}
std::vector<float> combined_sparse_seed(float Amin, float Amax, std::vector<float> seeds, int Nsparse, int Nocal) {
    std::vector<float> sparse = linspace(Amin, Amax, Nsparse);  // global scan

    std::vector<float> local_points;
    for (float seed : seeds) {
        local_points.push_back(seed);

        std::vector<float> log_offsets = linspace(-1.5f, -0.5f, Nocal);

        for (float log_offset : log_offsets) {
            float offset = std::pow(10.0f, log_offset);

            // Add both positive and negative offsets around seed
            float pos_point = seed + offset;
            float neg_point = seed - offset;

            if (pos_point >= Amin && pos_point <= Amax) {
                local_points.push_back(pos_point);
            }
            if (neg_point >= Amin && neg_point <= Amax) {
                local_points.push_back(neg_point);
            }
        }
    }

    sparse.insert(sparse.end(), local_points.begin(), local_points.end());

    // Remove duplicates with tolerance and sort
    std::sort(sparse.begin(), sparse.end());
    sparse.erase(std::unique(sparse.begin(), sparse.end(), 
                [](float a, float b) { return std::abs(a - b) < 0.015f; }), 
            sparse.end());

    return sparse;
}



PROsurf::PROsurf(PROmetric &metric,  size_t x_idx, size_t y_idx, size_t nbinsx, LogLin llx, float x_lo, float x_hi, size_t nbinsy, LogLin lly, float y_lo, float y_hi) : metric(metric), x_idx(x_idx), y_idx(y_idx), nbinsx(nbinsx), nbinsy(nbinsy), edges_x(Eigen::VectorXf::Constant(nbinsx + 1, 0)), edges_y(Eigen::VectorXf::Constant(nbinsy + 1, 0)), surface(nbinsx, nbinsy) {
          
    // If it's a log axis, we always convert to log space in order to define the grid
    if(llx == LogAxis) {
        x_lo = std::log10(x_lo);
        x_hi = std::log10(x_hi);
    }
    if(lly == LogAxis) {
        y_lo = std::log10(y_lo);
        y_hi = std::log10(y_hi);
    }
    
    for(size_t i = 0; i < nbinsx + 1; i++) {
        float edge_val = x_lo + i * (x_hi - x_lo) / nbinsx;
        if (llx == LogAxis) edge_val = std::pow(10, edge_val); // Now bin edges are back to linear space
        if (metric.GetModel().is_log10[x_idx])
            edges_x(i) = std::log10(edge_val);
        else
            edges_x(i) = edge_val;
    }
    for(size_t i = 0; i < nbinsy + 1; i++) {
        float edge_val = y_lo + i * (y_hi - y_lo) / nbinsy;
        if (lly == LogAxis) edge_val = std::pow(10, edge_val); // Now bin edges are back to linear space
        if (metric.GetModel().is_log10[y_idx])
            edges_y(i) = std::log10(edge_val);
        else
            edges_y(i) = edge_val;
    }

    /*
    for(size_t i = 0; i < nbinsx + 1; i++)
        log<LOG_ERROR>(L"%1% || xbin edge %2%: %3%") % __func__ % i % edges_x(i);
    for(size_t i = 0; i < nbinsy + 1; i++)
        log<LOG_ERROR>(L"%1% || ybin edge %2%: %3%") % __func__ % i % edges_y(i);
    */

}

void PROsurf::FillSurfaceStat(const PROconfig &config, const PROfitterConfig &fitconfig, std::string filename, const Eigen::VectorXf &cv_params, uint32_t seed) {
    std::ofstream chi_file;
    if(!filename.empty()){
        chi_file.open(filename);
        chi_file << "Dimensions: " << nbinsx << " " << nbinsy << "\n";
        chi_file << "Fixed indices: " << x_idx << " " << y_idx << "\n";
        chi_file << "Parameters:\n";
        for(const auto &name: metric.GetModel().param_names) chi_file << name << "\n";

        chi_file << "xval yval chi2";
        // TODO: Not saving this info for stats only right now (but we should)
        //for(size_t i = 0; i < metric.GetModel().nparams; ++i)
        //    chi_file << " p" << i;
        chi_file << "\n";
    }

    PROmetric *local_metric = metric.Clone();
    PROsyst newsyst = local_metric->GetSysts();
    newsyst.fractional_covariance = Eigen::MatrixXf::Constant(config.m_num_variable_bins_total[config.i_prime], config.m_num_variable_bins_total[config.i_prime], 0);
    local_metric->override_systs(newsyst);
    // When doing actual fits here we need to set the bounds of the metric
    // so the bounds of the systs are the cv values
    float min_chi = 1e9;
    Eigen::VectorXf dummy_grad = cv_params;
    Eigen::VectorXf params = cv_params;
    Eigen::VectorXf last;

    // TODO: If x_idx or y_idx refers to a systematic I think this may or may not be what we want
    // TODO: If one of the model parameters is fixed (using --fix) this will not fix that parameter (Will the regular surface fit fix that parameter? Or fix a systematic we wanted to fix?)

    if(local_metric->GetModel().nparams - 2 == 0) {
        for(size_t i = 0; i < nbinsx; i++) {
            for(size_t j = 0; j < nbinsy; j++) {
                params(y_idx) = (float)edges_y(j);
                params(x_idx) = (float)edges_x(i);
                float fx = (*local_metric)(params, dummy_grad, false);
                if(fx < min_chi) min_chi = fx;
                surface(i, j) = fx;
            }
        }
    } else {
        Eigen::VectorXf lb = params;
        for(size_t i = 0; i < local_metric->GetModel().nparams; ++i)
            lb(i) = local_metric->GetModel().lb(i);
        Eigen::VectorXf ub = params;
        for(size_t i = 0; i < local_metric->GetModel().nparams; ++i)
            ub(i) = local_metric->GetModel().ub(i);

        for(size_t i = 0; i < nbinsx; i++) {
            for(size_t j = 0; j < nbinsy; j++) {
                lb(y_idx) = (float)edges_y(j);
                ub(y_idx) = (float)edges_y(j);
                lb(x_idx) = (float)edges_x(i);
                ub(x_idx) = (float)edges_x(i);
                local_metric->setBounds(lb,ub);
                PROfitter fitter(ub, lb, fitconfig, seed+i+nbinsx*j);
                float fx;
                if(i != 0 || j != 0){
                    fx = fitter.Fit(*local_metric, last);
                }else{
                    fx = fitter.Fit(*local_metric);
                }
                last = fitter.best_fit;
                if(fx < min_chi) min_chi = fx;
                surface(i, j) = fx;
            }
        }
    }
    for(size_t i = 0; i < nbinsx; ++i) {
        for(size_t j = 0; j < nbinsy; ++j) {
            float fx = surface(i,j); 
            fx -= min_chi;
            surface(i,j) = fx;
            if(!filename.empty()){
                chi_file<<"\n"<<edges_x(i)<<" "<<edges_y(j)<<" "<<fx<<std::flush;
            }
        }
    }
    delete local_metric;
}

std::vector<profOut> PROfile::PROfilePointHelper(const PROsyst *systs, const PROfitterConfig &fitconfig, int offset, int stride, float minchi, bool with_osc, MultiPROgressBar& progressbar, const std::vector<Eigen::VectorXf> &seed_points, uint32_t seed) {


    std::vector<profOut> outs;
    // Make a local copy for this thread
    PROmetric *local_metric = metric.Clone();
    int nparams = local_metric->GetModel().nparams + systs->GetNSplines();
    int nstep = 18;

    Eigen::VectorXf ub, lb, tub, tlb;


    if(with_osc) {
        lb = Eigen::VectorXf::Constant(nparams, -3.0);
        ub = Eigen::VectorXf::Constant(nparams, 3.0);
        size_t nphys = local_metric->GetModel().nparams;
        //set physics to correct values
        for(size_t j=0; j<nphys; j++){
            ub(j) = local_metric->GetModel().ub(j);
            lb(j) = local_metric->GetModel().lb(j); 
        }
        //upper lower bounds for splines
        for(int j = nphys; j < nparams; ++j) {
            size_t si = j - nphys;
            lb(j) = systs->spline_has_restrict[si] ? systs->spline_restrict_lo[si] : systs->spline_lo[si];
            ub(j) = systs->spline_has_restrict[si] ? systs->spline_restrict_hi[si] : systs->spline_hi[si];
        }
    } else {
        // Syst-only mode: create full-size bounds but fix physics params at seed values
        lb = Eigen::VectorXf::Constant(nparams, -3.0);
        ub = Eigen::VectorXf::Constant(nparams, 3.0);
        size_t nphys = local_metric->GetModel().nparams;
        // Fix physics parameters at model default values
        for(size_t j=0; j<nphys; j++){
            float fixed_val = local_metric->GetModel().default_val(j);
            ub(j) = fixed_val;
            lb(j) = fixed_val;
        }
        // Spline bounds as normal
        for(int j = nphys; j < nparams; ++j) {
            size_t si = j - nphys;
            lb(j) = systs->spline_has_restrict[si] ? systs->spline_restrict_lo[si] : systs->spline_lo[si];
            ub(j) = systs->spline_has_restrict[si] ? systs->spline_restrict_hi[si] : systs->spline_hi[si];
        }
    }

    // In syst-only mode, start loop at first spline index (nphys), not 0
    int loop_start = with_osc ? 0 : (int)local_metric->GetModel().nparams;

    //loop over this threads todo list
    for(int i=loop_start+offset; i<nparams;i+=stride) {
        tlb = lb;
        tub = ub;

        local_metric->reset();

        size_t which_spline = i;
        bool isphys = which_spline < local_metric->GetModel().nparams;
        profOut output;

        log<LOG_INFO>(L"%1% || THREADS %2% in this batch if ( %3%,%4% )") % __func__ %  i % offset % stride;


        Eigen::VectorXf last_bf;

        //first get what values to sample
        std::vector<float> test_values;

        //if not physis do normal
        
        
        if(lb(which_spline)==ub(which_spline)){//its fixed. Dont run it
            test_values.push_back(ub(which_spline));
        }else{
            if(!isphys){
                //if lower bound is 0 or both pos/both negative (aka not sym around zero)
                if(lb(which_spline)==0 || (lb(which_spline)*ub(which_spline) >0) ){
                    for (int j = 0; j <= nstep; ++j) {
                        float which_value =  std::isinf(lb(which_spline)) ? -3 + (ub(which_spline) - (-3)) * j / (float)nstep :   lb(which_spline) + (ub(which_spline) - lb(which_spline)) * j / (float)nstep;
                        test_values.push_back(which_value);       
                    }
                }else{
                    for (int j = 0; j <= nstep; ++j) {
                        int k;
                        if (j <= nstep - nstep / 2) {
                            k = nstep / 2 + j;  // Forward direction
                        } else {
                            k = nstep - j;  // Backward direction
                        }
                        float which_value =  std::isinf(lb(which_spline)) ? -3 + (ub(which_spline) - (-3)) * k / (float)nstep :   lb(which_spline) + (ub(which_spline) - lb(which_spline)) * k / (float)nstep;
                        test_values.push_back(which_value);       
                    }
                }
            }else{
                //if its physics, grab seed points, need to include those
                std::vector<float> seed_values(seed_points.size());
                std::transform(seed_points.begin(), seed_points.end(), seed_values.begin(), [which_spline](const auto& vec) { return vec[which_spline]; });
                float mod = 0.5;
                while(test_values.size()<nstep*1.5){
                    test_values = combined_sparse_seed(std::isinf(lb(which_spline)) ? -3 : lb(which_spline), ub(which_spline), seed_values, nstep*mod, 2);
                    mod=mod*1.2;
                }
                log<LOG_INFO>(L"%1% || PROfileing over physics parameter number %2% has %3% uniform points, and %4% local ones for a total of %5% points. ") % __func__ %  which_spline % int(nstep*0.5) %int((2*2+1)*seed_values.size()) % test_values.size();
            }
        }

        //log<LOG_INFO>(L"%1% || PLONK which_spline %2% has testpt order %3% ") % __func__ %  which_spline % test_values;
        //and minimize

        Eigen::VectorXf cv_best_fit;
        float last_val = 0;
        for(auto &which_value: test_values){
            float fx;
            output.knob_vals.push_back(which_value);

            tlb[which_spline] = which_value;
            tub[which_spline] = which_value;

            local_metric->setBounds(tlb,tub);
            local_metric->fixSpline(which_spline,which_value);

            PROfitter fitter(tub, tlb, fitconfig, seed+i);

            std::vector<Eigen::VectorXf> test_seeds = seed_points;
            if(last_bf.size()>0){
                if(which_value*last_val<0){//have we flipper around the CV, if so lets take the best value from that and NOT init. 
                    test_seeds.push_back(cv_best_fit);
                }else{
                    test_seeds.push_back(last_bf);
                }
            }

            fx = fitter.Fit(*local_metric, test_seeds);
            output.knob_chis.push_back(fx - minchi);
            output.knob_bfs.push_back(fitter.best_fit);
            last_bf = fitter.best_fit;
            if(cv_best_fit.size()==0){cv_best_fit=last_bf;}

            std::string spec_string = "";
            for(auto &f : fitter.best_fit) spec_string+=" "+std::to_string(f); 
            log<LOG_INFO>(L"%1% || Fixed value of spline # %2% is value %3%, has a chi post of : %4% (i %5% nstep %6% ") % __func__ % which_spline % which_value % fx % i % nstep;
            log<LOG_INFO>(L"%1% || at a BF param value of @ %2%") % __func__ %  spec_string.c_str();

            last_val = which_value;
            if(fitconfig.progress_bar)progressbar.increment_bar(which_spline);

        }    //end step loop        
        output.sort();
        outs.push_back(output);

    }//end thread

    delete local_metric;

    return outs;
}



std::vector<surfOut> PROsurf::PointHelper(const PROfitterConfig &fitconfig, std::vector<surfOut> multi_physics_params, int start, int end, uint32_t seed, const Eigen::VectorXf &seed_pt){

    std::vector<surfOut> outs;

    // Make a local copy for this thread
    PROmetric *local_metric = metric.Clone();

    for(int i=start; i<end;i++){
        local_metric->reset();

        surfOut output;
        std::vector<float> physics_params = multi_physics_params[i].grid_val;
        output.grid_val = physics_params;
        output.grid_index = multi_physics_params[i].grid_index;

        int nparams = local_metric->GetModel().nparams + local_metric->GetSysts().GetNSplines() - 2;

        if(nparams == 0) {
            Eigen::VectorXf empty_vec, 
                params = Eigen::VectorXf::Map(physics_params.data(), physics_params.size());
            output.chi = (*local_metric)(params, empty_vec, false);
            output.best_fit = params; 
            outs.push_back(output);
            continue;
        }

        Eigen::VectorXf lb(nparams+2);
        lb << local_metric->GetModel().lb, Eigen::VectorXf::Map(local_metric->GetSysts().spline_lo.data(), local_metric->GetSysts().spline_lo.size());
        Eigen::VectorXf ub(nparams+2);
        ub << local_metric->GetModel().ub, Eigen::VectorXf::Map(local_metric->GetSysts().spline_hi.data(), local_metric->GetSysts().spline_hi.size());

        lb(x_idx) = multi_physics_params[i].grid_val[1];
        ub(x_idx) = multi_physics_params[i].grid_val[1];
        lb(y_idx) = multi_physics_params[i].grid_val[0];
        ub(y_idx) = multi_physics_params[i].grid_val[0];

        for(size_t j = local_metric->GetModel().nparams; j < nparams+2; ++j) {
            if(local_metric->GetSysts().spline_has_restrict[j-local_metric->GetModel().nparams]) {
                lb(j) = local_metric->GetSysts().spline_restrict_lo[j-local_metric->GetModel().nparams];
                ub(j) = local_metric->GetSysts().spline_restrict_hi[j-local_metric->GetModel().nparams];
            }
        }

        local_metric->setBounds(lb,ub);
        std::vector<Eigen::VectorXf> seeds;
        seeds.push_back(seed_pt);
        if(i != start) seeds.push_back(outs.back().best_fit);

        PROfitter fitter(ub, lb, fitconfig, seed+i);
        output.chi = fitter.Fit(*local_metric, seeds);
        output.best_fit = fitter.best_fit;
        outs.push_back(output);
    }

    delete local_metric;

    return outs;
}


void PROsurf::FillSurface(const PROfitterConfig &fitconfig, std::string filename, PROseed &proseed, float min_chi, const Eigen::VectorXf &seed_pt, int nThreads) {
    std::ofstream chi_file;
    if(!filename.empty()){
        chi_file.open(filename);
    }

    std::vector<surfOut> grid;
    for(size_t i = 0; i < nbinsx; i++) {
        for(size_t j = 0; j < nbinsy; j++) {
            std::vector<int> grid_pts = {(int)i,(int)j};
            std::vector<float> physics_params = {(float)edges_y(j), (float)edges_x(i)};  //deltam^2, sin^22thetamumu
            surfOut pt; pt.grid_val = physics_params; pt.grid_index = grid_pts;
            grid.push_back(pt);
        }
    }

    int loopSize = grid.size();
    int chunkSize = loopSize / nThreads;
    int remainder = loopSize % nThreads;

    std::vector<std::future<std::vector<surfOut>>> futures; 

    log<LOG_INFO>(L"%1% || Starting THREADS  : %2% , Loops %3%, Chunks %4%") % __func__ % nThreads % loopSize % chunkSize;

    int start = 0;
    int end = 0;
    for (int t = 0; t < nThreads; ++t) {
        start = end;
        end = start + chunkSize;
        if(t < remainder) end++;
        // make new vars for async, avoid mem issues
        int thread_start = start;
        int thread_end = end;
        futures.emplace_back(std::async(std::launch::async, [&, thread_start, thread_end]() {
                    return this->PointHelper(fitconfig, grid, thread_start, thread_end, proseed.getThreadSeeds()->at(t), seed_pt);
                    }));

    }

    std::vector<surfOut> combinedResults;
    for (auto& fut : futures) {
        std::vector<surfOut> result = fut.get();
        combinedResults.insert(combinedResults.end(), result.begin(), result.end());
    }

    if(filename != "") {
        chi_file << "Dimensions: " << nbinsx << " " << nbinsy << "\n";
        chi_file << "Fixed indices: " << x_idx << " " << y_idx << "\n";
        chi_file << "Parameters:\n";
        for(const auto &name: metric.GetModel().param_names) chi_file << name << "\n";
        for(const auto &name: metric.GetSysts().spline_names) chi_file << name << "\n";

        chi_file << "\nxval yval chi2";
        for(size_t i = 0; i < metric.GetModel().nparams + metric.GetSysts().GetNSplines(); ++i)
            chi_file << " p" << i;
    }
    float orig_chi = min_chi;
    for(const auto &item: combinedResults) {
        if(item.chi < orig_chi) {
            log<LOG_WARNING>(L"%1% || Found a point in the surface, index (%2%, %3%) with value (%4%, %5%), with chi^2 %6% which is lower than the minimum chi^2 passed into the function %7%. We will use this new value or the lowest other value in the surface instead of the min_chi passed in.") 
                % __func__ % item.grid_index[0] % item.grid_index[1] % item.grid_val[0] % item.grid_val[1] % item.chi % orig_chi;
        }
        if(item.chi < min_chi) {
            min_chi = item.chi;
        }
    }
    for (const auto& item : combinedResults) {
        log<LOG_INFO>(L"%1% || Finished  : %2% %3% %4%") % __func__ % item.grid_val[1] % item.grid_val[0] % (item.chi - min_chi);
        surface(item.grid_index[0], item.grid_index[1]) = item.chi - min_chi;
        results.push_back({item.grid_index[0], item.grid_index[1], item.best_fit, (item.chi-min_chi)});
        if(filename != "") {
            chi_file<<"\n"<<item.grid_val[1]<<" "<<item.grid_val[0]<<" "<<(item.chi-min_chi);
            for(float val: item.best_fit)
                chi_file << " " << val;
        }
    }
}



std::vector<surfOut> PROsurf::FillCurve(const PROfitterConfig &fitconfig, PROseed &proseed, float min_chi, const Eigen::VectorXf &seed_pt, int nThreads, std::vector<float> &A, std::vector<float> &B, size_t n_points) {


    std::vector<surfOut> grid;
    for(size_t i = 0; i < n_points; i++) {
        float t = (n_points == 1) ? 0.0f : static_cast<float>(i) / static_cast<float>(n_points - 1);

        std::vector<float> physics_params(A.size());
        for(size_t k = 0; k < A.size(); k++) {
            physics_params[k] = (1.0f - t) * A[k] + t * B[k];
        }

        std::vector<int> grid_pts = {(int)i, 0};
        surfOut pt;
        pt.grid_val = physics_params;
        pt.grid_index = grid_pts;
        grid.push_back(pt);
    }

    int loopSize = grid.size();
    int chunkSize = loopSize / nThreads;

    std::vector<std::future<std::vector<surfOut>>> futures; 

    log<LOG_INFO>(L"%1% || Starting THREADS  : %2% , Loops %3%, Chunks %4%") % __func__ % nThreads % loopSize % chunkSize;

    for (int t = 0; t < nThreads; ++t) {
        int start = t * chunkSize;
        int end = (t == nThreads - 1) ? loopSize : start + chunkSize;
        futures.emplace_back(std::async(std::launch::async, [&, start, end]() {
                    return this->PointHelper(fitconfig, grid, start, end, proseed.getThreadSeeds()->at(t), seed_pt);
                    }));
    }

    std::vector<surfOut> combinedResults;
    for (auto& fut : futures) {
        std::vector<surfOut> result = fut.get();
        combinedResults.insert(combinedResults.end(), result.begin(), result.end());
    }

    float orig_chi = min_chi;
    for(const auto &item: combinedResults) {
        if(item.chi < orig_chi) {
            log<LOG_WARNING>(L"%1% || Found a point in the surface, index (%2%, %3%) with value (%4%, %5%), with chi^2 %6% which is lower than the minimum chi^2 passed into the function %7%. We will use this new value or the lowest other value in the surface instead of the min_chi passed in.") 
                % __func__ % item.grid_index[0] % item.grid_index[1] % item.grid_val[0] % item.grid_val[1] % item.chi % orig_chi;
        }
        if(item.chi < min_chi) {
            min_chi = item.chi;
        }
    }

    for (auto& item : combinedResults) {
        log<LOG_INFO>(L"%1% || Finished  : %2% %3% %4%") % __func__ % item.grid_val[1] % item.grid_val[0] % (item.chi - min_chi);
        item.chi -= min_chi;
    }

    return combinedResults;
}

void PROsurf::PlotCurve(const PROconfig &config, const PROmodel &model, const PROsyst &syst, const std::vector<surfOut> & cpoints, std::string final_output_tag, bool logx, bool logy, size_t xaxis_idx,size_t yaxis_idx, std::vector<float> &A, std::vector<float> &B, [[maybe_unused]] size_t n_points){

    std::vector<float> binedges_x, binedges_y;
    // Edges are stored in model's native space (log if is_log10, linear otherwise)
    // Convert to linear for ROOT histogram bin edges
    for(size_t i = 0; i < this->nbinsx+1; i++)
        binedges_x.push_back(model.is_log10[xaxis_idx] ? std::pow(10, this->edges_x(i)) : this->edges_x(i));
    for(size_t i = 0; i < this->nbinsy+1; i++)
        binedges_y.push_back(model.is_log10[yaxis_idx] ? std::pow(10, this->edges_y(i)) : this->edges_y(i));

     std::string xlabel = xaxis_idx < model.nparams ? model.pretty_param_names.at(xaxis_idx) : 
            config.m_mcgen_variation_plotname_map.at(syst.spline_names.at(xaxis_idx));
     std::string ylabel = yaxis_idx < model.nparams ? model.pretty_param_names.at(yaxis_idx) : 
            config.m_mcgen_variation_plotname_map.at(syst.spline_names.at(yaxis_idx));
    TH2D surf("surf", (";"+xlabel+";"+ylabel).c_str(), this->nbinsx, binedges_x.data(), this->nbinsy, binedges_y.data());

    TCanvas c1("c1", "PROcurve Analysis", 1600, 800);
    TPad *p1 = new TPad("p1", "left", 0.0, 0.0, 0.3, 1.0);  // 30% width
    TPad *p2 = new TPad("p2", "right", 0.3, 0.0, 1.0, 1.0); // 70% width
    p1->Draw();
    p2->Draw();


    p1->cd();
    p1->SetLeftMargin(0.12);
    p1->SetRightMargin(0.15);
    if(logx) p1->SetLogx();
    if(logy) p1->SetLogy();
    surf.Draw("COLZ");
    TMarker *markerA = new TMarker(pow(10,A[xaxis_idx]), pow(10,A[yaxis_idx]), 29); 
    markerA->SetMarkerColor(kBlack);
    markerA->SetMarkerSize(3);
    markerA->Draw();
    TMarker *markerB = new TMarker(pow(10,B[xaxis_idx]), pow(10,B[yaxis_idx]), 29);
    markerB->SetMarkerColor(kBlack);
    markerB->SetMarkerSize(3);
    markerB->Draw();


    TArrow *arrow = new TArrow(pow(10,A[xaxis_idx]), pow(10,A[yaxis_idx]), pow(10,B[xaxis_idx]), pow(10,B[yaxis_idx]),0.01, "|>");
    arrow->SetLineStyle(2);  // Dashed
    arrow->SetLineWidth(2);
    arrow->SetLineColor(kBlack);
    arrow->SetFillColor(kBlack);  
    arrow->Draw();

    TPaveText *textbox = new TPaveText(0.18, 0.75, 0.55, 0.92, "NDC"); 
    textbox->SetFillColor(kWhite);
    textbox->SetBorderSize(1);
    textbox->SetTextAlign(12);  // Left align
    textbox->SetTextSize(0.04);

    textbox->AddText(("Start: " + xlabel + " = " + to_string_prec(A[0],3)).c_str());
    textbox->AddText(("       " + ylabel + " = " + to_string_prec(A[1],3)).c_str());
    textbox->AddText("");  // Blank line
    textbox->AddText(("End:   " + xlabel + " = " + to_string_prec(B[0],3)).c_str());
    textbox->AddText(("       " + ylabel + " = " + to_string_prec(B[1],3)).c_str());
    textbox->Draw();

    p1->RedrawAxis();


    p2->cd();
    p2->SetLeftMargin(0.12);
    p2->SetRightMargin(0.05);

    size_t nparams =  syst.GetNSplines();
    //Eigen::VectorXf subvector2 = param.segment();

    TMultiGraph *mg = new TMultiGraph();
    TLegend *leg = new TLegend(0.18,0.69,0.89,0.89);
    int colors[] = {kBlack, kRed, kBlue, kGreen+2, kMagenta, kOrange, kCyan+2, kViolet, kYellow+2, kPink};

    double ymin = 1e10, ymax = -1e10;

    for(size_t iparam = 0; iparam < nparams; iparam++) {
        std::vector<double> x_points, y_points;

        log<LOG_INFO>(L"%1% || Curve plot  %2% ") % __func__ % iparam;

        for(size_t i = 0; i < cpoints.size(); i++) {
            x_points.push_back(i);
            float val = cpoints[i].best_fit(iparam+model.nparams);
            y_points.push_back(val);
            if(val < ymin) ymin = val;
            if(val > ymax) ymax = val;

        }
        log<LOG_INFO>(L"Parameter %1%: min=%2%, max=%3%") % iparam % *std::min_element(y_points.begin(), y_points.end()) % *std::max_element(y_points.begin(), y_points.end());

        TGraph *gr = new TGraph(x_points.size(), x_points.data(), y_points.data());
        gr->SetLineColor(colors[iparam % 10]);
        gr->SetLineWidth(2);
        gr->SetMarkerColor(colors[iparam % 10]);
        gr->SetMarkerStyle(20);
        gr->SetMarkerSize(0.8);

        mg->Add(gr, "LP");  

        std::string param_name = config.m_mcgen_variation_plotname_map.at(syst.spline_names[iparam]);
        leg->AddEntry(gr, param_name.c_str(), "lp");
    }

    mg->Draw("A");
    double margin = (ymax - ymin) * 0.05;
    mg->GetYaxis()->SetRangeUser(ymin - margin, ymax + 5*margin);

    mg->GetXaxis()->SetTitle("Curve Point Index");
    mg->GetYaxis()->SetTitle("Parameter Value");
    mg->SetTitle("Parameter Evolution Along Curve");

    leg->SetNColumns(4);
    leg->SetFillStyle(0);
    leg->SetLineWidth(0);
    leg->Draw();

    c1.SaveAs((final_output_tag+"_PROcurve.pdf").c_str(), "pdf");

}



std::vector<float> findMinAndBounds(TGraph *g, float val, float lo, float hi) {
    float step = 0.001;
    int n = g->GetN();
    float minY = 1e9, minX = 0;
    for (int i = 0; i < n; ++i) {
        double x, y;
        g->GetPoint(i, x,y);
        if (y < minY) {
            minY = y;
            minX = x;
        }
    }
    //..ok so minX is the min and Currentl minY is the chi^2. Want this to be delta chi^2

    float leftX = minX, rightX = minX;

    // Search to the left of the minimum
    for (float x = minX; x >= lo; x -= step) {
        float y = g->Eval(x) - minY; //DeltaChi^2
        if (y >= val) {
            leftX = x;
            break;
        } else if(x - step < lo) {
            // If at end of loop and haven't found left side
            leftX = lo;
        }
    }

    // Search to the right of the minimum
    for (float x = minX; x <= hi; x += step) {
        float y = g->Eval(x)-minY;
        if (y >= val) {
            rightX = x;
            break;
        } else if(x + step > hi) {
            // If at end of loop and haven't found right side
            rightX = hi;
        }
    }

    return {minX,leftX,rightX};
}


PROfile::PROfile(const PROconfig &config, const PROsyst &systs, const PROmodel &model, PROmetric &metric, PROseed &proseed, const PROfitterConfig &fitconfig, [[maybe_unused]] std::string filename, float minchi, bool with_osc, int nThreads, const std::vector<Eigen::VectorXf> &seed_points, [[maybe_unused]] const Eigen::VectorXf & true_params) : metric(metric) {
    LBFGSpp::LBFGSBSolver<float> solver(fitconfig.param);
    int nparams = systs.GetNSplines() + model.nparams*with_osc;
    std::vector<float> physics_params; 

    //hack
    std::vector<float> priorX;
    std::vector<float> priorY;

    for(int i=0; i<=30;i++){
        float which_value = -3.0+0.2*i;
        priorX.push_back(which_value);
        priorY.push_back(which_value*which_value);

    }
    std::unique_ptr<TGraph> gprior = std::make_unique<TGraph>(priorX.size(), priorX.data(), priorY.data());

    std::vector<std::string> names;
    if(with_osc) for(const auto& name: model.pretty_param_names) names.push_back(name);
    for(const auto &name: systs.spline_names) names.push_back(name);

    int loopSize = nparams;
    if(nThreads>loopSize){
        nThreads = loopSize;
        log<LOG_INFO>(L"%1% || nThreads is < loopSize (nparams) : %2% <  %3%. Setting equal ") % __func__ % nThreads % loopSize ;
    }

    int chunkSize = loopSize / nThreads;

    std::vector<std::future<std::vector<profOut>>> futures; 

    std::vector<std::pair<int, std::string>> prof_PB_configs;
    for (int i = 0; i < nparams; ++i) {
        std::string nam;
        if(i < (long)metric.GetModel().nparams){
            nam = metric.GetModel().pretty_param_names[i];
            prof_PB_configs.push_back({9+3*seed_points.size(), nam});
        }else{
            long idx =  i - metric.GetModel().nparams ;
            std::string bad_name = metric.GetSysts().spline_names[idx];
            nam = config.m_mcgen_variation_plotname_map.at(bad_name);
            prof_PB_configs.push_back({19, nam});
        }
    }

    MultiPROgressBar prof_progress(prof_PB_configs);
    if(fitconfig.progress_bar){
        prof_progress.initialize_display();
        prof_progress.start_display_thread(); 
    }


    log<LOG_INFO>(L"%1% || Starting THREADS  : %2% , Loops %3%, Chunks %4%") % __func__ % nThreads % loopSize % chunkSize;

    for (int t = 0; t < nThreads; ++t) {
        std::string  strD = "";
        for(int i=t; i<nparams;i+=nThreads) {
            strD+=std::to_string(i);
        }
        log<LOG_INFO>(L"%1% || THREAD #%2% runs pts: %3% ") % __func__ % t % strD.c_str();

        futures.emplace_back(std::async(std::launch::async, [&, t]() {
                    return this->PROfilePointHelper(&systs, fitconfig, t, nThreads, minchi, with_osc, std::ref(prof_progress), seed_points, proseed.getThreadSeeds()->at(t));
                    }));

    }

    std::vector<profOut> combinedResults(nparams);
    int offset = 0;
    int stride = nThreads;
    for (auto& fut : futures) {
        std::vector<profOut> result = fut.get();
        for(size_t i = 0; i < result.size(); ++i)
            combinedResults.at(offset+i*stride) = result.at(i);
        ++offset;
    }
    if(fitconfig.progress_bar) prof_progress.finish_all();

    //create all graphs, used directly in first setion
    for(auto & out: combinedResults){
        log<LOG_INFO>(L"%1% || Knob Values: %2%") % __func__ %  out.knob_vals;
        log<LOG_INFO>(L"%1% || Knob Chis: %2%") % __func__ %  out.knob_chis;
        std::unique_ptr<TGraph> g = std::make_unique<TGraph>(out.knob_vals.size(), out.knob_vals.data(), out.knob_chis.data());
        graphs.push_back(std::move(g));
        for(size_t u=0; u< out.knob_vals.size(); u++){
            if(out.knob_chis.at(u)<0){
                float chidiff = fabs(out.knob_chis.at(u));
                float bfnorm_without_phys = seed_points.size() ? seed_points.front().tail(seed_points.front().size() - model.nparams).norm() : 1;

                Eigen::VectorXf param_diff = seed_points.size() ? out.knob_bfs.at(u)-seed_points.front() : out.knob_bfs.at(u);
                float norm_without_phys = param_diff.tail(param_diff.size() - model.nparams).norm();

                float relative_tol1 = 1e-3;

                if( (norm_without_phys < relative_tol1*bfnorm_without_phys)  &&  (chidiff< relative_tol1*minchi) ){
                    log<LOG_WARNING>(L"%1% || Warning. A lower global best fit was found during PROfile, but less than relatively 1e-3f from global best fit (%2%), and pull parameters norm less than 1e-3 (%3%)from best_fit nuisence values. AKA same point.") % __func__ % float(chidiff/minchi) % float(norm_without_phys/bfnorm_without_phys);
                }else if((chidiff< relative_tol1*minchi)){
                    log<LOG_WARNING>(L"%1% || Warning. A lower global best fit was found during PROfile, but less than relatively 1e-3f from global best fit (%2%), although pull parameters norm more than 1e-3 (%3%) from best_fit nuisence values. Not uncommon in degenerate phase space.") % __func__ % float(chidiff/minchi) % float(norm_without_phys/bfnorm_without_phys); 
                }else{
                    log<LOG_WARNING>(L"%1% || Warning. A lower global best fit was found during PROfile. Difference greater than relative 1e-3f from global best fit (%2%). Rel Difference pull parameters norm (%3%). This is enough that could consider updating global best fit values and restarting.") % __func__ % float(chidiff/minchi) % float(norm_without_phys/bfnorm_without_phys)  ;
                    log<LOG_WARNING>(L"%1% || -- minchi %2% and chidiff %3% ") % __func__ % minchi % chidiff  ;
                    // log<LOG_ERROR>(L"%1% || TEMP chi glob %2% new %3% ") % __func__ % minchi % out.knob_chis.at(u);
                    // log<LOG_ERROR>(L"%1% || TEMP norm %2% pull norm %3% ") % __func__ % norm % norm_without_phys;
                    // log<LOG_ERROR>(L"%1% || TEMP chi param %2% ") % __func__ % seed_points.front() ;
                    // log<LOG_ERROR>(L"%1% || TEMP new param %2% ") % __func__ % out.knob_bfs.at(u);


                    newglob=out.knob_chis.at(u)+minchi;
                    newglob_param = out.knob_bfs.at(u);

                }
            }
        }
    }

    //Analyze them, used in later section
    //plot 2sigma also? default no, as its messier
    bool twosig = false;

    std::vector<float> values1_errup;
    std::vector<float> values1_errdown;

    std::vector<float> barvalues_err;

    std::vector<float> values2_up;
    std::vector<float> values2_down;


    log<LOG_INFO>(L"%1% || Getting BF, +/- one sigma ranges. Is Two sigma turned on? : %2% ") % __func__ % twosig;

    size_t count = 0;
    for(auto &g:graphs){
        //if(metric->GetModel().nparams)continue;
        float lo, hi;
        if (with_osc && count < metric.GetModel().nparams) {
            lo = metric.GetModel().lb(count);
            hi = metric.GetModel().ub(count);
        } else {
            size_t spline_idx = with_osc ? count - metric.GetModel().nparams : count;
            lo = metric.GetSysts().spline_lo[spline_idx];
            hi = metric.GetSysts().spline_hi[spline_idx];
        }
        if(std::isinf(lo)) lo = lo < 0 ? -5 : 5;
        if(std::isinf(hi)) hi = hi < 0 ? -5 : 5;
        std::vector<float> tmp = findMinAndBounds(g.get(),1.0, lo, hi);
        barvalues.push_back(float(count)+0.5);
        barvalues_err.push_back(0.3);
        bfvalues.push_back(tmp[0]);
        values1_down.push_back(tmp[1]);
        values1_up.push_back(tmp[2]);
        values1_errdown.push_back(abs(tmp[1]-tmp[0]));
        values1_errup.push_back(abs(tmp[2]-tmp[0]));
        log<LOG_DEBUG>(L"%1% || Results of findMinAndBounds : %2% %3% %4% ") % __func__ % tmp[0] % tmp[1] % tmp[2];
        log<LOG_DEBUG>(L"%1% || Barvalues : %2% %3% %4% %5%") % __func__ % count % barvalues[count] % barvalues_err[count] % barvalues_err[count];
        log<LOG_DEBUG>(L"%1% || Bfvalues : %2% %3% ") % __func__ % count % bfvalues[count];
        log<LOG_DEBUG>(L"%1% || RangeValues : %2% %3% %4% ") % __func__ % count % values1_down[count] % values1_up[count];
        log<LOG_DEBUG>(L"%1% || ErrValues : %2% %3% %4% ") % __func__ % count % values1_errdown[count] % values1_errup[count];
        if(twosig){
            std::vector<float> tmp2 = findMinAndBounds(g.get(),4.0,lo, hi);
            values2_down.push_back(abs(tmp2[1]-tmp[0]));
            values2_up.push_back(abs(tmp2[2]-tmp[0]));
        }
        count++;
    }
    onesig = TGraphAsymmErrors(barvalues.size(),barvalues.data(), bfvalues.data(), barvalues_err.data(), barvalues_err.data(), values1_errdown.data(), values1_errup.data());

}

void PROfile::Plot(const PROconfig &config, const PROsyst &systs, const PROmodel &model, [[maybe_unused]] PROmetric &metric, [[maybe_unused]] PROseed &proseed, std::string filename, bool with_osc, const Eigen::VectorXf& init_seed, const Eigen::VectorXf & true_params, const Eigen::MatrixXf& spline_covariance, const Eigen::VectorXf& param_err_lo, const Eigen::VectorXf& param_err_hi, bool mask_osc) {

    int nparams = systs.GetNSplines() + model.nparams*with_osc;
    int nBins = nparams;
    std::vector<std::string> names;
    if(with_osc) for(const auto& name: model.pretty_param_names) names.push_back(name);
    for(const auto &name: systs.spline_names) names.push_back(name);

    std::vector<float> priorX;
    std::vector<float> priorY;

    for(int i=0; i<=30;i++){
        float which_value = -3.0+0.2*i;
        priorX.push_back(which_value);
        priorY.push_back(which_value*which_value);

    }
    std::unique_ptr<TGraph> gprior = std::make_unique<TGraph>(priorX.size(), priorX.data(), priorY.data());

    //First plot - use multi-page PDF with 4 columns x 3 rows per page
    const int nCols = 4;
    const int nRows = 3;
    const int plotsPerPage = nCols * nRows;

    TCanvas *c = new TCanvas(filename.c_str(), filename.c_str(), 350*nCols, 350*nRows);
    c->Divide(nCols, nRows);

    // Open multi-page PDF
    c->Print((filename+".pdf[").c_str());

    // First, collect all plots to draw (main plots + zoomed physics plots)
    std::vector<std::function<void()>> plotFunctions;

    for(size_t w = 0; w < graphs.size(); w++) {
        if(mask_osc && w < model.nparams) continue;

        // Capture w by value for the lambda
        size_t idx = w;
        plotFunctions.push_back([&, idx]() {
            // Check if this is a physics param (only possible when with_osc=true)
            bool is_physics = with_osc && idx < model.nparams;
            std::string xval = is_physics ? "Log_{10}(" + model.pretty_param_names[idx]+")" : "#sigma Shift";
            std::string tit = (is_physics ? names[idx] : config.m_mcgen_variation_plotname_map.at(names[idx])) + ";" + xval + "; #Delta#Chi^{2}";
            graphs[idx]->SetTitle(tit.c_str());
            graphs[idx]->Draw("AL");
            graphs[idx]->SetLineWidth(1);
            graphs[idx]->GetYaxis()->SetTitleSize(0.05);
            graphs[idx]->GetYaxis()->SetLabelSize(0.04);
            graphs[idx]->GetXaxis()->SetTitleSize(0.05);
            graphs[idx]->GetXaxis()->SetLabelSize(0.04);
            graphs[idx]->GetYaxis()->SetRangeUser(0, graphs[idx]->GetHistogram()->GetMaximum());

            TLine* line = new TLine(graphs[idx]->GetXaxis()->GetXmin(), 1, graphs[idx]->GetXaxis()->GetXmax(), 1);
            line->SetLineStyle(3);
            line->SetLineWidth(1);
            line->SetLineColor(kBlack);
            line->Draw();

            if(is_physics) graphs[idx]->SetLineColor(kBlue-7);

            if(graphs[idx]->GetN() == 1) {
                float x_val = graphs[idx]->GetPointX(0);
                TLine* linet = new TLine(x_val, 0, x_val, 1);
                linet->SetLineStyle(1);
                linet->SetLineWidth(1);
                linet->SetLineColor(kGreen+2);
                linet->Draw();
            }
            else if(!is_physics) {
                gprior->Draw("L same");
                gprior->SetLineStyle(2);
                gprior->SetLineWidth(1);
                gprior->SetLineColor(kRed-7);
                graphs[idx]->GetYaxis()->SetRangeUser(0, std::min(graphs[idx]->GetHistogram()->GetMaximum(), 10.0));
            }
        });

        // Add zoomed plots for physics parameters after the last physics parameter (only when physics params are included)
        if(with_osc && w == model.nparams - 1) {
            for(int zs = model.nparams - 1; zs >= 0; zs--) {
                size_t zoomIdx = w - zs;
                plotFunctions.push_back([&, zoomIdx]() {
                    TGraph* graphClone = new TGraph(*graphs[zoomIdx]);
                    graphClone->Draw("AL");
                    std::string newTitle = std::string(graphClone->GetTitle()) + " Zoomed 1#sigma";
                    graphClone->SetTitle(newTitle.c_str());
                    graphClone->SetLineColor(kViolet);
                    graphClone->SetLineWidth(1);
                    float vd = std::min(values1_down[zoomIdx], values1_up[zoomIdx]);
                    float vu = std::max(values1_down[zoomIdx], values1_up[zoomIdx]);
                    float pd = (vd > 0 ? vd * 0.9 : vd * 1.1);
                    float pu = (vu > 0 ? vu * 1.1 : vu * 0.9);
                    graphClone->GetXaxis()->SetLimits(pd, pu);
                    graphClone->GetYaxis()->SetRangeUser(0, std::max(graphClone->Eval(pu), graphClone->Eval(pd)) * 1.1);
                    graphClone->GetYaxis()->SetTitleSize(0.05);
                    graphClone->GetYaxis()->SetLabelSize(0.04);
                    graphClone->GetXaxis()->SetTitleSize(0.05);
                    graphClone->GetXaxis()->SetLabelSize(0.04);

                    if(graphClone->GetN() == 1) {
                        float x_val = graphClone->GetPointX(0);
                        TLine* linet = new TLine(x_val, 0, x_val, 1);
                        linet->SetLineStyle(1);
                        linet->SetLineWidth(1);
                        linet->SetLineColor(kGreen+2);
                        linet->Draw();
                    }

                    log<LOG_INFO>(L"%1% || Zoom boundaries X %2% %3% Y %4% %5%  ") % __func__ % pd % pu % 0.0 % (std::max(graphClone->Eval(pu), graphClone->Eval(pd)) * 1.1);

                    TLine* line1 = new TLine(pd, 1, pu, 1);
                    line1->SetLineStyle(3);
                    line1->SetLineWidth(1);
                    line1->SetLineColor(kBlack);
                    line1->Draw();

                    TLine* line2 = new TLine(vd, graphClone->Eval(vd), vd, 0);
                    line2->SetLineStyle(3);
                    line2->SetLineWidth(1);
                    line2->SetLineColor(kBlack);
                    line2->Draw();

                    TLine* line3 = new TLine(vu, graphClone->Eval(vu), vu, 0);
                    line3->SetLineStyle(3);
                    line3->SetLineWidth(1);
                    line3->SetLineColor(kBlack);
                    line3->Draw();
                });
            }
        }
    }

    // Now draw plots page by page
    for(size_t i = 0; i < plotFunctions.size(); i++) {
        int padIdx = (i % plotsPerPage) + 1;

        // Start of a new page (except for the first plot)
        if(i > 0 && padIdx == 1) {
            c->Print((filename+".pdf").c_str());
            c->Clear();
            c->Divide(nCols, nRows);
        }

        c->cd(padIdx);
        plotFunctions[i]();
    }

    // Print the last page and close the PDF
    c->Print((filename+".pdf").c_str());
    c->Print((filename+".pdf]").c_str());

    delete c;

    // Summary bar chart - scales from 1 to dozens of parameters
    log<LOG_DEBUG>(L"%1% || Are all lines the same : %2% %3% %4% %5% %6%") % __func__ % nBins % barvalues.size() % bfvalues.size() % values1_down.size() % values1_up.size() ;

    // Y-axis range: include everything within ±5σ (band + markers), nothing beyond.
    // Clamp band extremes to ±5σ so the axis never extends past that.
    float minVal = std::max(*std::min_element(values1_down.begin(), values1_down.end()), -5.0f);
    float maxVal = std::min(*std::max_element(values1_up.begin(),   values1_up.end()),    5.0f);
    for(int i = 0; i < nBins; ++i) {
        if(mask_osc && i < (int)model.nparams) continue;
        int vec_idx = with_osc ? i : (i + model.nparams);
        // Collect all marker values for this bin
        std::vector<float> markers;
        if(i        < (int)bfvalues.size())    markers.push_back(bfvalues[i]);
        if(vec_idx  < (int)init_seed.size())   markers.push_back((float)init_seed[vec_idx]);
        if(vec_idx  < (int)true_params.size()) markers.push_back((float)true_params[vec_idx]);
        // Expand range to include each marker, but only if it's within ±5σ
        for(float v : markers) {
            if(v >= -5.0f && v <= 5.0f) {
                minVal = std::min(minVal, v);
                maxVal = std::max(maxVal, v);
            }
        }
    }

    { // _1sigma_detailed.pdf scope
    // 50px/bin keeps the aspect ratio reasonable for large parameter counts;
    // height is fixed so NDC font/line sizes stay consistent.
    int c2_width  = std::max(600, std::min(5000, 50 * nBins));
    int c2_height = 500;
    float axis_label_size = std::max(0.030f, std::min(0.045f, 1.8f / nBins));
    float x_label_size    = std::max(0.015f, std::min(0.030f, 1.2f / nBins));
    int   profile_lw      = (nBins > 20) ? 1 : 2;
    float bar_halfwidth   = std::max(0.08f, std::min(0.4f, 4.0f / nBins));
    float marker_offset   = bar_halfwidth * 0.6f;

    TCanvas *c2 = new TCanvas((filename+"1sigma_detailed").c_str(), (filename+"1sigma_detailed").c_str(), c2_width, c2_height);
    c2->cd();
    c2->SetLeftMargin(0.09);
    c2->SetBottomMargin(0.30);
    c2->SetRightMargin(0.28);
    c2->SetTopMargin(0.08);

    // Derived error vectors for profile_pts
    std::vector<float> values1_errdown, values1_errup;
    for(int i = 0; i < nBins; ++i) {
        values1_errdown.push_back(std::abs(values1_down[i] - bfvalues[i]));
        values1_errup.push_back(std::abs(values1_up[i]   - bfvalues[i]));
    }

    // Y-axis range: cover the post-fit band; always show ±1 (prior reference)
    float minVal_det = minVal;
    float maxVal_det = maxVal;
    minVal_det = std::min(minVal_det, -1.2f);
    maxVal_det = std::max(maxVal_det, 1.2f);
    float y_axis_min = minVal_det * 1.15f;
    float y_axis_max = maxVal_det * 1.15f;

    // TH1F frame to establish axes (persists for the lifetime of the canvas)
    TH1F *frame = new TH1F((filename+"_frame").c_str(), "", nBins, 0, nBins);
    frame->SetMinimum(y_axis_min);
    frame->SetMaximum(y_axis_max);
    frame->SetStats(0);
    frame->GetXaxis()->SetLabelSize(0);
    frame->GetXaxis()->SetTickLength(0);
    frame->GetYaxis()->SetTitle("Parameter value (prior #kern[0.3]{} #sigma = 1)");
    frame->GetYaxis()->SetTitleSize(axis_label_size);
    frame->GetYaxis()->SetLabelSize(axis_label_size);
    frame->GetYaxis()->SetTitleOffset(1.0);
    frame->Draw("AXIS");

    // Full-width gray band for pre-fit ±1σ (prior spans entire plot)
    TBox *prior_band = new TBox(0.0f, -1.0f, (float)nBins, 1.0f);
    prior_band->SetFillColor(kGray);
    prior_band->SetFillStyle(1001);
    prior_band->SetLineColor(kGray+1);
    prior_band->SetLineWidth(1);
    prior_band->Draw("same");

    // Post-fit ±1σ bars (blue): centered on the global best-fit, width from MCMC
    // 16th/84th percentile quantiles.  For oscillation physics parameters (no entry
    // in param_err_lo/hi), fall back to the profile interval.
    TGraphAsymmErrors todraw = onesig;
    for(int i = 0; i < nBins; ++i) {
        int vec_idx = with_osc ? i : (i + (int)model.nparams);
        float center = (vec_idx < (int)init_seed.size()) ? (float)init_seed[vec_idx] : bfvalues[i];
        int syst_idx = with_osc ? (i - (int)model.nparams) : i;
        bool is_syst = !with_osc || (i >= (int)model.nparams);
        float err_lo, err_hi;
        if(is_syst && syst_idx >= 0 && syst_idx < param_err_lo.size() && syst_idx < param_err_hi.size()) {
            err_lo = param_err_lo(syst_idx);
            err_hi = param_err_hi(syst_idx);
        } else {
            err_lo = todraw.GetErrorYlow(i);
            err_hi = todraw.GetErrorYhigh(i);
        }
        todraw.SetPoint(i, todraw.GetPointX(i), center);
        todraw.SetPointError(i, bar_halfwidth, bar_halfwidth, err_lo, err_hi);
    }
    if(mask_osc) {
        for(size_t i = 0; i < model.nparams; ++i) {
            todraw.SetPoint(i, 0, 0);
            todraw.SetPointError(i, 0, 0, 0, 0);
        }
    }
    todraw.SetFillColor(kBlue-7);
    todraw.SetFillStyle(1001);
    todraw.SetLineColor(kBlue-8);
    todraw.SetLineWidth(1);
    todraw.Draw("2 same");

    // Horizontal reference lines
    TLine l_zero(0, 0, nBins, 0);
    l_zero.SetLineStyle(2);
    l_zero.SetLineColor(kGray+2);
    l_zero.SetLineWidth(1);
    l_zero.Draw();
    TLine l_pm1(0, 1, nBins, 1);
    l_pm1.SetLineStyle(3);
    l_pm1.SetLineColor(kGray+2);
    l_pm1.SetLineWidth(1);
    l_pm1.Draw();
    TLine l_mm1(0, -1, nBins, -1);
    l_mm1.SetLineStyle(3);
    l_mm1.SetLineColor(kGray+2);
    l_mm1.SetLineWidth(1);
    l_mm1.Draw();

    // Profile scan best-fit ±1σ: black circles with error bars
    std::vector<float> zeroes_err(nBins, 0.0f);
    TGraphAsymmErrors profile_pts(nBins, barvalues.data(), bfvalues.data(),
                                   zeroes_err.data(), zeroes_err.data(),
                                   values1_errdown.data(), values1_errup.data());
    if(mask_osc) {
        for(size_t i = 0; i < model.nparams; ++i) {
            profile_pts.SetPoint(i, barvalues[i], y_axis_min - 999.0f);
            profile_pts.SetPointError(i, 0, 0, 0, 0);
        }
    }
    float marker_size = std::max(0.5f, std::min(1.4f, 6.0f / std::sqrt((float)nBins)));
    profile_pts.SetMarkerStyle(20);
    profile_pts.SetMarkerSize(marker_size);
    profile_pts.SetMarkerColor(kBlack);
    profile_pts.SetLineColor(kBlack);
    profile_pts.SetLineWidth(profile_lw);
    profile_pts.Draw("P same");

    // Arrow/marker helper for out-of-range values
    float y_range_size = y_axis_max - y_axis_min;
    float arrow_margin = y_range_size * 0.07f;
    float arrow_length = y_range_size * 0.05f;
    log<LOG_INFO>(L"%1% || _1sigma plot y-axis range: min=%2%, max=%3%") % __func__ % y_axis_min % y_axis_max;

    auto drawMarkerWithArrow2 = [&](float x, float y, int color, int marker_style, float msize) {
        bool clamp_below = y < -5.0f;
        bool clamp_above = y > 5.0f;
        if(!clamp_below && !clamp_above && (y < y_axis_min || y > y_axis_max)) return;

        float draw_y = clamp_below ? y_axis_min + arrow_margin : (clamp_above ? y_axis_max - arrow_margin : y);

        TMarker* marker = new TMarker(x, draw_y, marker_style);
        marker->SetMarkerSize(msize);
        marker->SetMarkerColor(color);
        marker->Draw();

        if(clamp_below) {
            TArrow* arr = new TArrow(x, draw_y - arrow_length * 0.3f, x, y_axis_min + arrow_length * 0.2f, 0.008, "|>");
            arr->SetLineColor(color); arr->SetFillColor(color); arr->SetLineWidth(1);
            arr->Draw();
        } else if(clamp_above) {
            TArrow* arr = new TArrow(x, draw_y + arrow_length * 0.3f, x, y_axis_max - arrow_length * 0.2f, 0.008, "|>");
            arr->SetLineColor(color); arr->SetFillColor(color); arr->SetLineWidth(1);
            arr->Draw();
        }
    };

    for(int i = 0; i < nBins; ++i) {
        if(mask_osc && i < (int)model.nparams) continue;
        int vec_idx = with_osc ? i : (i + model.nparams);
        float x_center = i + 0.5f;

        // Red square: global best-fit (init_seed)
        if(vec_idx < init_seed.size()) {
            drawMarkerWithArrow2(x_center - marker_offset, (float)init_seed[vec_idx], kRed, 21, marker_size);
        }

        // Orange diamond: injected true values (true_params)
        if(vec_idx < (int)true_params.size()) {
            log<LOG_INFO>(L"%1% || _1sigma marker i=%2%: true_params[%3%]=%4%") % __func__ % i % vec_idx % true_params[vec_idx];
            drawMarkerWithArrow2(x_center + marker_offset, (float)true_params[vec_idx], kOrange+7, 33, marker_size * 1.1f);
        }
    }

    // Parameter labels (rotated -45 degrees)
    float text_size = x_label_size;
    float label_y = y_axis_min - y_range_size * 0.04f;
    for(size_t i = 0; i < barvalues.size(); ++i) {
        if(mask_osc && with_osc && i < model.nparams) continue;
        std::string label;
        if(with_osc && i < model.nparams) {
            label = "Log_{10}(" + model.pretty_param_names[i] + ")";
        } else {
            label = config.m_mcgen_variation_plotname_map.at(names[i]);
        }
        TLatex* text = new TLatex(barvalues[i], label_y, label.c_str());
        text->SetTextAlign(13);
        text->SetTextSize(text_size);
        text->SetTextAngle(-45);
        text->Draw();
    }

    // Legend (NDC coordinates, top-right)
    TLegend *leg = new TLegend(0.73, 0.55, 0.99, 0.92);
    leg->SetFillStyle(1001);
    leg->SetBorderSize(1);
    leg->SetTextSize(std::max(0.022f, std::min(0.030f, axis_label_size * 0.85f)));
    leg->AddEntry(prior_band,   "Pre-fit #pm1#sigma (prior)", "f");
    leg->AddEntry(&todraw,      "Post-fit #pm1#sigma (MCMC)", "f");
    leg->AddEntry(&profile_pts, "Profile scan #pm1#sigma", "pe");
    TGraph *leg_global = new TGraph(1);
    leg_global->SetPoint(0, 0, 0);
    leg_global->SetMarkerStyle(21);
    leg_global->SetMarkerColor(kRed);
    leg_global->SetMarkerSize(marker_size);
    leg->AddEntry(leg_global, "Global best-fit", "p");
    TGraph *leg_inj = new TGraph(1);
    leg_inj->SetPoint(0, 0, 0);
    leg_inj->SetMarkerStyle(33);
    leg_inj->SetMarkerColor(kOrange+7);
    leg_inj->SetMarkerSize(marker_size * 1.1f);
    leg->AddEntry(leg_inj, "Injected values", "p");
    leg->Draw();

    // Version label
    TText *t = new TText();
    t->SetNDC();
    t->SetTextFont(42);
    t->SetTextSize(0.028f);
    t->SetTextAlign(33);
    std::string pv = "PROfit v"+std::string(PROJECT_VERSION_STR);
    t->DrawText(0.96, 0.97, pv.c_str());

    c2->Update();
    c2->SaveAs((filename+"_1sigma_detailed.pdf").c_str(),"pdf");
    delete c2;
    } // end _1sigma_detailed.pdf scope

    //Next version
    TCanvas *c2 =  new TCanvas((filename+"1sigma").c_str(), (filename+"1sigma").c_str() , 20*nparams, 400);
    c2->cd();
    c2->SetBottomMargin(0.25);
    c2->SetRightMargin(0.05);

    onesig.SetFillColor(kBlue-7);
    onesig.SetStats(0);
    //onesig.SetMinimum(min(-1.2,minVal*1.2));
    onesig.SetMinimum(minVal*1.1);
    onesig.SetMaximum(maxVal*1.1);

    onesig.GetXaxis()->SetNdivisions(barvalues.size());  // Set number of tick marks
    onesig.GetXaxis()->SetLabelSize(0);  // Hide default numerical labels

    onesig.SetTitle("");
    TGraphAsymmErrors todraw = onesig;
    if(mask_osc) {
        for(size_t i = 0; i < model.nparams; ++i) {
            todraw.SetPoint(i, 0,0);
            todraw.SetPointError(i, 0, 0, 0, 0);
        }
    }
    todraw.Draw("A2");
    //onesig.Draw("A2");
    //onesig.GetYaxis()->SetTitle("#sigma Shift");
    todraw.GetYaxis()->SetTitle("Posterior 1#sigma Error");
    todraw.GetYaxis()->SetTitleOffset(0.8);

    float y_min = todraw.GetMinimum();
    for (size_t i = 0; i < barvalues.size(); ++i) {
        // In syst-only mode (with_osc=false), all entries are splines
        // In with_osc mode, first model.nparams entries are physics, rest are splines
        std::string label;
        if (with_osc && i < model.nparams) {
            label = "Log_{10}(" + model.pretty_param_names[i] + ")";
        } else {
            label = config.m_mcgen_variation_plotname_map.at(names[i]);
        }
        TLatex* text = new TLatex(barvalues[i], y_min - 0.05, label.c_str());  // Position text below axis
        text->SetTextAlign(13);
        text->SetTextSize(0.03);
        text->SetTextAngle(-45);
        text->Draw();
    }
    TText *t = new TText();
    t->SetNDC();
    t->SetTextFont(42);
    t->SetTextSize(0.03);
    t->SetTextAlign(33);
    std::string pv = "PROfit v"+std::string(PROJECT_VERSION_STR);
    t->DrawText(0.895, 0.955, pv.c_str());

    c2->Update();

    //if (twosig) {
    //    TGraphAsymmErrors *h2 = new TGraphAsymmErrors(barvalues.size(),barvalues.data(), bfvalues.data(), barvalues_err.data(), barvalues_err.data(), values2_down.data(), values2_up.data());
    //    h2->SetFillColor(38);
    //    h2->Se


    TLine l(0,0,nBins+0.5,0);
    l.SetLineStyle(2);
    l.SetLineColor(kBlack);
    l.SetLineWidth(1);
    l.Draw();
    TLine l2(0,-1,nBins+0.5,-1);
    l2.SetLineStyle(3);
    l2.SetLineColor(kBlack);
    l2.SetLineWidth(1);
    l2.Draw();
    TLine l3(0,1,nBins+0.5,1);
    l3.SetLineStyle(3);
    l3.SetLineColor(kBlack);
    l3.SetLineWidth(1);
    l3.Draw();



    // Get y-axis range for out-of-bounds handling
    float y_axis_min = minVal * 1.1;
    float y_axis_max = maxVal * 1.1;
    float y_range = y_axis_max - y_axis_min;
    log<LOG_INFO>(L"%1% || _1sigma plot y-axis range: min=%2%, max=%3%") % __func__ % y_axis_min % y_axis_max;
    float arrow_margin = y_range * 0.08;  // margin from edge for out-of-range markers
    float arrow_length = y_range * 0.06;  // length of the arrow

    // Horizontal offsets to prevent marker overlap
    float offset_blue = -0.12;   // init_seed (blue) - left
    float offset_red = 0.0;      // true_params (red) - center
    float offset_black = 0.12;   // best fit (black) - right

    // Helper lambda to draw a marker with out-of-range arrow handling
    auto drawMarkerWithArrow = [&](float x, float y, int color, float marker_size) {
        bool clamp_below = y < -5.0f;
        bool clamp_above = y > 5.0f;
        if(!clamp_below && !clamp_above && (y < y_axis_min || y > y_axis_max)) return;

        float draw_y = clamp_below ? y_axis_min + arrow_margin : (clamp_above ? y_axis_max - arrow_margin : y);

        TMarker* marker = new TMarker(x, draw_y, 29);
        marker->SetMarkerSize(marker_size);
        marker->SetMarkerColor(color);
        marker->Draw();

        if(clamp_below) {
            TArrow* arr = new TArrow(x, draw_y - arrow_length * 0.3, x, y_axis_min + arrow_length * 0.2, 0.008, "|>");
            arr->SetLineColor(color);
            arr->SetFillColor(color);
            arr->SetLineWidth(1);
            arr->Draw();
        } else if(clamp_above) {
            TArrow* arr = new TArrow(x, draw_y + arrow_length * 0.3, x, y_axis_max - arrow_length * 0.2, 0.008, "|>");
            arr->SetLineColor(color);
            arr->SetFillColor(color);
            arr->SetLineWidth(1);
            arr->Draw();
        }
    };

    for (int i = 0; i < nBins; ++i) {
        if(mask_osc && i < (int)model.nparams) continue;
        // In syst-only mode, init_seed/true_params are full-size but plot indices are spline-only
        int vec_idx = with_osc ? i : (i + model.nparams);
        float x_center = i + 0.5;

        // Blue star: init_seed (best fit seed)
        drawMarkerWithArrow(x_center + offset_blue, init_seed[vec_idx], kBlue, 0.6);

        // Red star: true_params (injected truth)
        if (vec_idx < true_params.size()) {
            if (i < 3) {  // Log first few for debugging
                log<LOG_INFO>(L"%1% || _1sigma marker i=%2%: true_params[%3%]=%4%, below_range=%5%")
                    % __func__ % i % vec_idx % true_params[vec_idx] % (true_params[vec_idx] < y_axis_min);
            }
            drawMarkerWithArrow(x_center + offset_red, true_params[vec_idx], kRed, 0.5);
        }

        // Black star: best fit value
        drawMarkerWithArrow(x_center + offset_black, bfvalues[i], kBlack, 0.5);
    }



    t->DrawText(0.895, 0.955, pv.c_str());
    c2->SaveAs((filename+"_1sigma.pdf").c_str(),"pdf");
    delete c2;

    return;
}

