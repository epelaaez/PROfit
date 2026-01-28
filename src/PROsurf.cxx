#include "PROsurf.h"
#include "PROfitter.h"
#include "PROlog.h"

#include <Eigen/Eigen>

#include <cmath> 
#include <future>
#include <algorithm>

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

void PROsurf::FillSurfaceStat(const PROconfig &config, const PROfitterConfig &fitconfig, std::string filename) {
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

    // I think this will be needed for stat fits with more than 2 physics parameters
    (void)fitconfig;

    PROsyst dummy_syst;
    dummy_syst.fractional_covariance = Eigen::MatrixXf::Constant(config.m_num_variable_bins_total[config.i_prime], config.m_num_variable_bins_total[config.i_prime], 0);
    Eigen::VectorXf empty_vec;

    PROmetric *local_metric = metric.Clone();
    local_metric->override_systs(dummy_syst);
    float min_chi = 1e9;

    for(size_t i = 0; i < nbinsx; i++) {
        for(size_t j = 0; j < nbinsy; j++) {
            Eigen::VectorXf physics_params{{(float)edges_y(j), (float)edges_x(i)}};
            float fx = (*local_metric)(physics_params, empty_vec, false);
            if(fx < min_chi) min_chi = fx;
            surface(i, j) = fx;
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
            lb(j) = systs->spline_lo[j-nphys];
            ub(j) = systs->spline_hi[j-nphys];
        }
    } else {
        // Syst-only mode: create full-size bounds but fix physics params at seed values
        lb = Eigen::VectorXf::Constant(nparams, -3.0);
        ub = Eigen::VectorXf::Constant(nparams, 3.0);
        size_t nphys = local_metric->GetModel().nparams;
        // Fix physics parameters at seed values
        for(size_t j=0; j<nphys; j++){
            float fixed_val = (seed_points.size() > 0) ? seed_points.front()(j) : 0.0f;
            ub(j) = fixed_val;
            lb(j) = fixed_val;
        }
        // Spline bounds as normal
        for(int j = nphys; j < nparams; ++j) {
            lb(j) = systs->spline_lo[j-nphys];
            ub(j) = systs->spline_hi[j-nphys];
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

        int cnt=0;
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

            cnt++;
            last_val = which_value;
            if(fitconfig.progress_bar)progressbar.increment_bar(which_spline);

        }    //end step loop        
        output.sort();
        outs.push_back(output);

    }//end thread

    delete local_metric;

    return outs;
}



std::vector<surfOut> PROsurf::PointHelper(const PROfitterConfig &fitconfig, std::vector<surfOut> multi_physics_params, int start, int end, uint32_t seed){

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

        local_metric->setBounds(lb,ub);

        PROfitter fitter(ub, lb, fitconfig, seed+i);
        if(i!=start){
            output.chi = fitter.Fit(*local_metric, outs.back().best_fit);
        }else{
            output.chi = fitter.Fit(*local_metric);
        }
        output.best_fit = fitter.best_fit;
        outs.push_back(output);
    }

    delete local_metric;

    return outs;
}


void PROsurf::FillSurface(const PROfitterConfig &fitconfig, std::string filename, PROseed &proseed, int nThreads) {
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

    std::vector<std::future<std::vector<surfOut>>> futures; 

    log<LOG_INFO>(L"%1% || Starting THREADS  : %2% , Loops %3%, Chunks %4%") % __func__ % nThreads % loopSize % chunkSize;

    for (int t = 0; t < nThreads; ++t) {
        int start = t * chunkSize;
        int end = (t == nThreads - 1) ? loopSize : start + chunkSize;
        futures.emplace_back(std::async(std::launch::async, [&, start, end]() {
                    return this->PointHelper(fitconfig, grid, start, end, proseed.getThreadSeeds()->at(t));
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
    float min_chi = 1e9;
    for(const auto &item: combinedResults) {
        if(item.chi < min_chi) min_chi = item.chi;
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



std::vector<surfOut> PROsurf::FillCurve(const PROfitterConfig &fitconfig, PROseed &proseed, int nThreads, std::vector<float> &A, std::vector<float> &B, size_t n_points) {


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
                    return this->PointHelper(fitconfig, grid, start, end, proseed.getThreadSeeds()->at(t));
                    }));

    }

    std::vector<surfOut> combinedResults;
    for (auto& fut : futures) {
        std::vector<surfOut> result = fut.get();
        combinedResults.insert(combinedResults.end(), result.begin(), result.end());
    }

    return combinedResults;



}


void PROsurf::PlotCurve(const PROconfig &config, const PROmodel &model, const PROsyst &syst, const std::vector<surfOut> & cpoints, std::string final_output_tag, bool logx, bool logy, size_t xaxis_idx,size_t yaxis_idx, std::vector<float> &A, std::vector<float> &B, size_t n_points){

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


PROfile::PROfile(const PROconfig &config, const PROsyst &systs, const PROmodel &model, PROmetric &metric, PROseed &proseed, const PROfitterConfig &fitconfig, std::string filename, float minchi, bool with_osc, int nThreads, const std::vector<Eigen::VectorXf> &seed_points, const Eigen::VectorXf & true_params) : metric(metric) {
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
                float bfnorm = seed_points.size() ? seed_points.front().norm() : 1;
                float bfnorm_without_phys = seed_points.size() ? seed_points.front().tail(seed_points.front().size() - model.nparams).norm() : 1;

                Eigen::VectorXf param_diff = seed_points.size() ? out.knob_bfs.at(u)-seed_points.front() : out.knob_bfs.at(u);
                float norm = param_diff.norm();
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
        float lo = count < metric.GetModel().nparams ? metric.GetModel().lb(count) :
            metric.GetSysts().spline_lo[count - metric.GetModel().nparams];
        if(std::isinf(lo)) lo = lo < 0 ? -5 : 5;
        float hi = count < metric.GetModel().nparams ? metric.GetModel().ub(count) :
            metric.GetSysts().spline_hi[count - metric.GetModel().nparams];
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

void PROfile::Plot(const PROconfig &config, const PROsyst &systs, const PROmodel &model, PROmetric &metric, PROseed &proseed, std::string filename, bool with_osc, const Eigen::VectorXf& init_seed, const Eigen::VectorXf & true_params, bool mask_osc) {

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

    //First plot
    int depth = std::ceil((nparams+model.nparams)/4.0);
    TCanvas *c =  new TCanvas(filename.c_str(), filename.c_str() , 350*4, 350*depth);
    c->Divide(4,depth);


    size_t zoom_shift = 0;
    for(size_t w = 0; w< graphs.size(); w++ ){
        if(mask_osc && w < model.nparams) continue;

        c->cd(w+1+zoom_shift);
        std::string xval = w < model.nparams ? "Log_{10}(" + model.pretty_param_names[w]+")" :"#sigma Shift"  ;
        std::string tit = (w < model.nparams ? names[w] :config.m_mcgen_variation_plotname_map.at(names[w]))+ ";"+xval+"; #Delta#Chi^{2}";
        graphs[w]->SetTitle(tit.c_str());
        graphs[w]->Draw("AL");
        graphs[w]->SetLineWidth(2);
        graphs[w]->GetYaxis()->SetTitleSize(0.04);             
        graphs[w]->GetYaxis()->SetLabelSize(0.04);            
        graphs[w]->GetXaxis()->SetTitleSize(0.04);             
        graphs[w]->GetXaxis()->SetLabelSize(0.04);            
        graphs[w]->GetYaxis()->SetRangeUser(0, graphs[w]->GetHistogram()->GetMaximum());

        TLine* line = new TLine(graphs[w]->GetXaxis()->GetXmin(), 1, graphs[w]->GetXaxis()->GetXmax(), 1);
        line->SetLineStyle(3);  // Dotted line style (1 is solid, 2 is dashed, 3 is dotted)
        line->SetLineWidth(1);  // Thin line
        line->SetLineColor(kBlack);  // Set color (black for visibility)
        line->Draw();

        if(w<model.nparams) graphs[w]->SetLineColor(kBlue-7);

        if(graphs[w]->GetN()==1){//1 point, its been fixed. Just draw a line
            float x_val = graphs[w]->GetPointX(0);
            TLine* linet = new TLine(x_val, 0, x_val, 1);
            linet->SetLineStyle(1);  // Dotted line style (1 is solid, 2 is dashed, 3 is dotted)
            linet->SetLineWidth(2);  // Thin line
            linet->SetLineColor(kGreen+2);  // Set color (black for visibility)
            linet->Draw();
        }
        else if(w>=model.nparams){
            gprior->Draw("L same");
            gprior->SetLineStyle(2);
            gprior->SetLineWidth(2);
            gprior->SetLineColor(kRed-7);
            graphs[w]->GetYaxis()->SetRangeUser(0, std::min(graphs[w]->GetHistogram()->GetMaximum(),10.0));
        }



        if(w==model.nparams-1){
            //on past physics param, lets do a quick zoom, stepping back though the physics param
            for(int zs = model.nparams-1; zs>=0; zs--){
                c->cd(w+1+zs+1);
                TGraph * graphClone = new TGraph(*graphs[w-zs]);
                graphClone->Draw("AL");
                std::string newTitle = std::string(graphClone->GetTitle()) + " Zoomed 1#sigma";
                graphClone->SetTitle(newTitle.c_str());
                graphClone->SetLineColor(kViolet);
                float vd = std::min(values1_down[w-zs],values1_up[w-zs]) ;
                float vu = std::max(values1_down[w-zs],values1_up[w-zs]) ;
                float pd = (vd>0 ? vd*0.9 : vd*1.1);
                float pu = (vu >0 ? vu*1.1 : vu*0.9);
                graphClone->GetXaxis()->SetLimits(pd,pu); 
                graphClone->GetYaxis()->SetRangeUser(0, std::max(graphClone->Eval(pu),graphClone->Eval(pd))*1.1) ;
                graphClone->GetYaxis()->SetTitleSize(0.04);             
                graphClone->GetYaxis()->SetLabelSize(0.04);            
                graphClone->GetXaxis()->SetTitleSize(0.04);             
                graphClone->GetXaxis()->SetLabelSize(0.04);            

                if(graphClone->GetN()==1){//1 point, its been fixed. Just draw a line
                    float x_val = graphClone->GetPointX(0);
                    TLine* linet = new TLine(x_val, 0, x_val, 1);
                    linet->SetLineStyle(1);  // Dotted line style (1 is solid, 2 is dashed, 3 is dotted)
                    linet->SetLineWidth(2);  // Thin line
                    linet->SetLineColor(kGreen+2);  // Set color (black for visibility)
                    linet->Draw();
                }

                log<LOG_INFO>(L"%1% || Zoom boundaries X %2% %3% Y %4% %5%  ") % __func__ % pd % pu % 0.0 % (std::max(graphClone->Eval(pu),graphClone->Eval(pd))*1.1)  ;

                TLine *line1 = new TLine(pd, 1, pu, 1);
                line1->SetLineStyle(3);  
                line1->SetLineWidth(1);  
                line1->SetLineColor(kBlack); 
                line1->Draw();

                TLine* line2 = new TLine(vd, graphClone->Eval(vd) ,vd, 0);
                line2->SetLineStyle(3);  
                line2->SetLineWidth(1);  
                line2->SetLineColor(kBlack); 
                line2->Draw();

                TLine *line3 = new TLine(vu, graphClone->Eval(vu) ,vu, 0);
                line3->SetLineStyle(3);  
                line3->SetLineWidth(1);  
                line3->SetLineColor(kBlack); 
                line3->Draw();
            }
            zoom_shift=model.nparams;
        }
    }

    c->SaveAs((filename+".pdf").c_str(),"pdf");

    delete c;

    //Next version
    TCanvas *c2 =  new TCanvas((filename+"1sigma").c_str(), (filename+"1sigma").c_str() , 20*nparams, 400);
    c2->cd();
    c2->SetBottomMargin(0.25);
    c2->SetRightMargin(0.05);

    log<LOG_DEBUG>(L"%1% || Are all lines the same : %2% %3% %4% %5% %6%") % __func__ % nBins % barvalues.size() % bfvalues.size() % values1_down.size() % values1_up.size() ;

    float minVal = *std::min_element(values1_down.begin(), values1_down.end());
    float maxVal = *std::max_element(values1_up.begin(), values1_up.end());

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
        std::string label = i < model.nparams ? "Log_{10}(" + model.pretty_param_names[i]+")" : config.m_mcgen_variation_plotname_map.at(names[i]);
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



    for (int i = 0; i < nBins; ++i) {
        if(mask_osc && i < model.nparams) continue;
        // In syst-only mode, init_seed/true_params are full-size but plot indices are spline-only
        int vec_idx = with_osc ? i : (i + model.nparams);
        TMarker* initstar = new TMarker(i+0.5, init_seed[vec_idx], 29);
        initstar->SetMarkerSize(0.6); 
        initstar->SetMarkerColor(kBlue); 
        initstar->Draw();

        if (vec_idx < true_params.size()) {

            TMarker* truestar = new TMarker(i+0.5, true_params[vec_idx], 29);
            truestar->SetMarkerSize(0.5); 
            truestar->SetMarkerColor(kRed); 
            truestar->Draw();
        }

        TMarker* star = new TMarker(i+0.5, bfvalues[i], 29);
        star->SetMarkerSize(0.5); 
        star->SetMarkerColor(kBlack); 
        star->Draw();
    }



    t->DrawText(0.895, 0.955, pv.c_str()); 
    c2->SaveAs((filename+"_1sigma.pdf").c_str(),"pdf");
    delete c2;

    return;
}

