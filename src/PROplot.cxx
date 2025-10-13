#include "PROplot.h"

namespace PROfit{


    std::map<std::string, std::unique_ptr<TH1D>> getCV1DHists(const PROspec &spec, const PROconfig& inconfig, bool scale, int other_index) {
        std::map<std::string, std::unique_ptr<TH1D>> hists;  

        size_t global_subchannel_index = 0;
        size_t global_channel_index = 0;
        for(size_t im = 0; im < inconfig.m_num_modes; im++){
            for(size_t id =0; id < inconfig.m_num_detectors; id++){
                for(size_t ic = 0; ic < inconfig.m_num_channels; ic++){
                    for(size_t sc = 0; sc < inconfig.m_num_subchannels[ic]; sc++){
                        const std::string& subchannel_name  = inconfig.m_fullnames[global_subchannel_index];
                        const std::string& color = inconfig.m_subchannel_colors[ic][sc];
                        int rcolor = color == "NONE" ? kRed - 7 : inconfig.HexToROOTColor(color);
                        std::unique_ptr<TH1D> htmp = std::make_unique<TH1D>(spec.toTH1D(inconfig, global_subchannel_index, other_index));
                        htmp->SetLineWidth(1);
                        htmp->SetLineColor(kBlack);
                        htmp->SetFillColor(rcolor);
                        if(scale) htmp->Scale(1,"width");
                        hists[subchannel_name] = std::move(htmp);

                        log<LOG_DEBUG>(L"%1% || Printot %2% %3% %4% %5% %6% : Integral %7% ") % __func__ % global_channel_index % global_subchannel_index % subchannel_name.c_str() % sc % ic % hists[subchannel_name]->Integral();
                        ++global_subchannel_index;
                    }//end subchan
                    ++global_channel_index;
                }//end chan
            }//end det
        }//end mode
        return hists;
    }

    std::map<std::string, std::unique_ptr<TH2D>> getCV2DHists(const PROspec &spec, const PROconfig& inconfig, bool scale, int other_index) {
        std::map<std::string, std::unique_ptr<TH2D>> hists;

        size_t global_subchannel_index = 0;
        size_t global_channel_index = 0;
        for(size_t im = 0; im < inconfig.m_num_modes; im++){
            for(size_t id =0; id < inconfig.m_num_detectors; id++){
                for(size_t ic = 0; ic < inconfig.m_num_channels; ic++){
                    for(size_t sc = 0; sc < inconfig.m_num_subchannels[ic]; sc++){
                        if(inconfig.m_variable_dims.at(ic) == 2){
                            const std::string& subchannel_name  = inconfig.m_fullnames[global_subchannel_index];
                            const std::string& color = inconfig.m_subchannel_colors[ic][sc];
                            int rcolor = color == "NONE" ? kRed - 7 : inconfig.HexToROOTColor(color);
                            std::unique_ptr<TH2D> htmp = std::make_unique<TH2D>(spec.toTH2D(inconfig, global_subchannel_index, other_index));
                            if(scale) htmp->Scale(1,"width");
                            hists[subchannel_name] = std::move(htmp);
                            log<LOG_DEBUG>(L"%1% || Printot %2% %3% %4% %5% %6% : Integral %7% ") % __func__ % global_channel_index % global_subchannel_index % subchannel_name.c_str() % sc % ic % hists[subchannel_name]->Integral();
                            }
                        ++global_subchannel_index;
                        }//end subchan
                    ++global_channel_index;
                }//end chan
            }//end det
        }//end mode
        return hists;
    }

    std::map<std::string, std::unique_ptr<TH2D>> covarianceTH2D(const PROsyst &syst, const PROconfig &config, const PROspec &cv) {
        std::map<std::string, std::unique_ptr<TH2D>> ret;
        Eigen::MatrixXf fractional_cov = syst.fractional_covariance;
        Eigen::MatrixXf diag = cv.Spec().array().matrix().asDiagonal(); 
        Eigen::MatrixXf full_covariance =  diag*fractional_cov*diag;
        Eigen::MatrixXf collapsed_full_covariance =  CollapseMatrix(config,full_covariance);  
        Eigen::VectorXf collapsed_cv = CollapseMatrix(config, cv.Spec());
        Eigen::MatrixXf collapsed_cv_inv_diag = collapsed_cv.asDiagonal().inverse();
        Eigen::MatrixXf collapsed_frac_cov = collapsed_cv_inv_diag * collapsed_full_covariance * collapsed_cv_inv_diag;

        std::unique_ptr<TH2D> cov_hist = std::make_unique<TH2D>("cov", "Fractional Covariance Matrix;Bin # ;Bin #", config.m_num_variable_bins_total[config.i_prime], 0, config.m_num_variable_bins_total[config.i_prime], config.m_num_variable_bins_total[config.i_prime], 0, config.m_num_variable_bins_total[config.i_prime]);
        std::unique_ptr<TH2D> collapsed_cov_hist = std::make_unique<TH2D>("ccov", "Collapsed Fractional Covariance Matrix;Bin # ;Bin #", config.m_num_variable_bins_total_collapsed[config.i_prime], 0, config.m_num_variable_bins_total_collapsed[config.i_prime], config.m_num_variable_bins_total_collapsed[config.i_prime], 0, config.m_num_variable_bins_total_collapsed[config.i_prime]);

        std::unique_ptr<TH2D> cor_hist = std::make_unique<TH2D>("cor", "Correlation Matrix;Bin # ;Bin #", config.m_num_variable_bins_total[config.i_prime], 0, config.m_num_variable_bins_total[config.i_prime], config.m_num_variable_bins_total[config.i_prime], 0, config.m_num_variable_bins_total[config.i_prime]);
        std::unique_ptr<TH2D> collapsed_cor_hist = std::make_unique<TH2D>("ccor", "Collapsed Correlation Matrix;Bin # ;Bin #", config.m_num_variable_bins_total_collapsed[config.i_prime], 0, config.m_num_variable_bins_total_collapsed[config.i_prime], config.m_num_variable_bins_total_collapsed[config.i_prime], 0, config.m_num_variable_bins_total_collapsed[config.i_prime]);

        for(size_t i = 0; i < config.m_num_variable_bins_total[config.i_prime]; ++i)
            for(size_t j = 0; j < config.m_num_variable_bins_total[config.i_prime]; ++j){
                cov_hist->SetBinContent(i+1,j+1,fractional_cov(i,j));
                cor_hist->SetBinContent(i+1,j+1,fractional_cov(i,j)/(sqrt(fractional_cov(i,i))*sqrt(fractional_cov(j,j))));
            }

        for(size_t i = 0; i < config.m_num_variable_bins_total_collapsed[config.i_prime]; ++i)
            for(size_t j = 0; j < config.m_num_variable_bins_total_collapsed[config.i_prime]; ++j){
                collapsed_cov_hist->SetBinContent(i+1,j+1,collapsed_frac_cov(i,j));
                collapsed_cor_hist->SetBinContent(i+1,j+1,collapsed_frac_cov(i,j)/(sqrt(collapsed_frac_cov(i,i))*sqrt(collapsed_frac_cov(j,j))));
            }

        float cov_max_abs = std::max(cov_hist->GetMaximum(), std::abs(cov_hist->GetMinimum()));
        cov_hist->SetMaximum(cov_max_abs);
        cov_hist->SetMinimum(-cov_max_abs);
        float coll_cov_max_abs = std::max(collapsed_cov_hist->GetMaximum(), std::abs(collapsed_cov_hist->GetMinimum()));
        collapsed_cov_hist->SetMaximum(coll_cov_max_abs);
        collapsed_cov_hist->SetMinimum(-coll_cov_max_abs);
        cor_hist->SetMaximum(1);
        cor_hist->SetMinimum(-1);
        collapsed_cor_hist->SetMaximum(1);
        collapsed_cor_hist->SetMinimum(-1);

        ret["total_frac_cov"] = std::move(cov_hist);
        ret["collapsed_total_frac_cov"] = std::move(collapsed_cov_hist);
        ret["total_cor"] = std::move(cor_hist);
        ret["collapsed_total_cor"] = std::move(collapsed_cor_hist);

        for(const auto &name: syst.covar_names) {
            const Eigen::MatrixXf &covar = syst.GrabMatrix(name);
            const Eigen::MatrixXf &corr = syst.GrabCorrMatrix(name);

            std::unique_ptr<TH2D> cov_h = std::make_unique<TH2D>(("cov"+name).c_str(), (name+" Fractional Covariance;Bin # ;Bin #").c_str(), config.m_num_variable_bins_total[config.i_prime], 0, config.m_num_variable_bins_total[config.i_prime], config.m_num_variable_bins_total[config.i_prime], 0, config.m_num_variable_bins_total[config.i_prime]);
            std::unique_ptr<TH2D> corr_h = std::make_unique<TH2D>(("cor"+name).c_str(), (name+" Correlation;Bin # ;Bin #").c_str(), config.m_num_variable_bins_total[config.i_prime], 0, config.m_num_variable_bins_total[config.i_prime], config.m_num_variable_bins_total[config.i_prime], 0, config.m_num_variable_bins_total[config.i_prime]);
            for(size_t i = 0; i < config.m_num_variable_bins_total[config.i_prime]; ++i){
                for(size_t j = 0; j < config.m_num_variable_bins_total[config.i_prime]; ++j){
                    cov_h->SetBinContent(i+1,j+1,covar(i,j));
                    corr_h->SetBinContent(i+1,j+1,corr(i,j));
                }
            }

            float cov_max_abs = std::max(cov_h->GetMaximum(), std::abs(cov_h->GetMinimum()));
            cov_h->SetMaximum(cov_max_abs);
            cov_h->SetMinimum(-cov_max_abs);
            corr_h->SetMaximum(1);
            corr_h->SetMinimum(-1);

            ret[name+"_cov"] = std::move(cov_h);
            ret[name+"_corr"] = std::move(corr_h);
        }

        return ret;
    }
/*
    std::map<std::string, std::vector<std::pair<std::unique_ptr<TGraph>,std::unique_ptr<TGraph>>>> 
        getSplineGraphs(const PROsyst &systs, const PROconfig &config) {
            std::map<std::string, std::vector<std::pair<std::unique_ptr<TGraph>,std::unique_ptr<TGraph>>>> spline_graphs;

            for(size_t i = 0; i < systs.GetNSplines(); ++i) {
                const std::string &name = systs.spline_names[i];
                const Spline &spline = systs.GrabSpline(name);
                //using Spline = std::vector<std::vector<std::pair<float, std::array<float, 4>>>>;
                std::vector<std::pair<std::unique_ptr<TGraph>,std::unique_ptr<TGraph>>> bin_graphs;
                size_t nbins =  config.m_num_variable_bins_total[systs.spline_binnings[i]];
                bin_graphs.reserve(nbins);

                for(size_t j = 0; j < nbins; ++j) {
                    const std::vector<std::pair<float, std::array<float, 4>>> &spline_for_bin = spline[j];
                    std::unique_ptr<TGraph> curve = std::make_unique<TGraph>();
                    std::unique_ptr<TGraph> fixed_pts = std::make_unique<TGraph>();
                    for(size_t k = 0; k < spline_for_bin.size(); ++k) {
                        //const auto &[lo, coeffs] = spline_for_bin[k];
                        float lo = spline_for_bin[k].first;
                        std::array<float, 4> coeffs = spline_for_bin[k].second;
                        float hi = k < spline_for_bin.size() - 1 ? spline_for_bin[k+1].first : systs.spline_hi[i];
                        auto fn = [coeffs](float shift){
                            return coeffs[0] + coeffs[1]*shift + coeffs[2]*shift*shift + coeffs[3]*shift*shift*shift;
                        };
                        fixed_pts->SetPoint(fixed_pts->GetN(), lo, fn(0)); 
                        if(k == spline_for_bin.size() - 1)
                            fixed_pts->SetPoint(fixed_pts->GetN(), hi, fn(hi - lo));
                        float width = (hi - lo) / 20;
                        for(size_t l = 0; l < 20; ++l)
                            curve->SetPoint(curve->GetN(), lo + l * width, fn(l * width));
                    }
                    bin_graphs.push_back(std::make_pair(std::move(fixed_pts), std::move(curve)));
                }
                spline_graphs[name] = std::move(bin_graphs);
            }

            return spline_graphs;
        }
*/
std::map<std::string, std::vector<std::pair<std::unique_ptr<TGraph>, std::unique_ptr<TGraph>>>>
getSplineGraphs(const PROsyst &systs, const PROconfig &config) {
    std::map<std::string, std::vector<std::pair<std::unique_ptr<TGraph>, std::unique_ptr<TGraph>>>> spline_graphs;

    for (size_t i = 0; i < systs.GetNSplines(); ++i) {
        const std::string &name = systs.spline_names[i];
        const Spline &spline = systs.GrabSpline(name);
        std::vector<std::pair<std::unique_ptr<TGraph>, std::unique_ptr<TGraph>>> bin_graphs;
        size_t nbins = config.m_num_variable_bins_total.at(systs.spline_binnings[i]);
        int nsegs = spline.segments_per_bin;
        bin_graphs.reserve(nbins);

        for (size_t j = 0; j < nbins; ++j) {
            std::unique_ptr<TGraph> curve = std::make_unique<TGraph>();
            std::unique_ptr<TGraph> fixed_pts = std::make_unique<TGraph>();

            // Access the segment range for this bin
            size_t seg_offset = j * nsegs;

            for (int k = 0; k < nsegs; ++k) {
                const SplineSegment &seg = spline.segments[seg_offset + k];
                float lo = seg.knot;
                std::array<float, 4> coeffs = seg.coeffs;
                // Determine hi for this segment
                float hi;
                if (k < nsegs - 1) {
                    hi = spline.segments[seg_offset + k + 1].knot;
                } else {
                    hi = systs.spline_hi[i];
                }
                auto fn = [coeffs](float shift) {
                    return coeffs[0] + coeffs[1] * shift + coeffs[2] * shift * shift + coeffs[3] * shift * shift * shift;
                };
                fixed_pts->SetPoint(fixed_pts->GetN(), lo, fn(0));
                if (k == nsegs - 1)
                    fixed_pts->SetPoint(fixed_pts->GetN(), hi, fn(hi - lo));
                float width = (hi - lo) / 20.0f;
                for (int l = 0; l < 20; ++l)
                    curve->SetPoint(curve->GetN(), lo + l * width, fn(l * width));
            }
            bin_graphs.push_back(std::make_pair(std::move(fixed_pts), std::move(curve)));
        }
        spline_graphs[name] = std::move(bin_graphs);
    }

    return spline_graphs;
}
    PROerrorbar getErrorBand(const PROconfig &config, const PROpeller &prop, const PROsyst &syst, const PROmodel &model, const PROspec &cv_spec, const Eigen::VectorXf &cvparams,bool scale, int other_index) {

        //cv_spec.Print();
        //   log<LOG_INFO>(L"%1% ||ARSE %2% ") % __func__ % cv_spec.Spec();
        //    log<LOG_INFO>(L"%1% ||AddressSE %2% ") % __func__ % &cv_spec;

        Eigen::VectorXf cv = CollapseMatrix(config, cv_spec.Spec(), other_index);
   
        std::vector<float> centers;
        size_t global_channel_index = 0;
        for(size_t mode = 0; mode < config.m_num_modes; ++mode) {
            for(size_t det = 0; det < config.m_num_detectors; ++det) {
                for(size_t channel = 0; channel < config.m_num_channels; ++channel) {
                    std::vector<float> tedges =  config.GetChannelVariableBins(global_channel_index, other_index).Edges();
                    global_channel_index++;
                    for(size_t p=0; p<tedges.size(); p++){
                        if(p<tedges.size()-1){
                            centers.push_back((tedges[p+1]+tedges[p])/2.0);
                        }
                    }
                    
                }
            }
        }

        log<LOG_DEBUG>(L"%1% || PARK cv is %2% and the centers are %3%") % __func__ %  cv.size() % centers.size();
        log<LOG_DEBUG>(L"%1% || For other var %2% the cv is %3% and the centers are %4%") % __func__ % other_index % cv % centers;
        size_t nerrorsample = 2500;
        
        std::vector<Eigen::VectorXf> specs;
        std::uniform_int_distribution<uint32_t> dseed(0, std::numeric_limits<uint32_t>::max());
        
        //Fills already collapsed
        for(size_t i = 0; i < nerrorsample; ++i){
            Eigen::VectorXf var = FillSystRandomThrow(config, prop, syst, model,cv_spec, cvparams, dseed(PROseed::global_rng), other_index).Spec();
            specs.push_back(var);

        }

        PROerrorbar ebar(cv.size());
        for(int i = 0; i < cv.size(); ++i) {
            std::vector<float> binconts(nerrorsample);
            for(size_t j = 0; j < nerrorsample; ++j) {
                binconts[j] = specs[j](i);
            }
            float scale_factor = scale ? 1.0/config.collapsed_bin_widths.at(other_index)(i) :  1.0;
            if(std::isnan(scale_factor)) scale_factor = 1;
            std::sort(binconts.begin(), binconts.end());
            float ehi = std::abs((binconts[2.5*840] - cv(i))*scale_factor);
            float elo = std::abs((cv(i) - binconts[2.5*160])*scale_factor);
            ebar.error_up(i) =  ehi;
            ebar.error_down(i) =  elo;
            ebar.error_point(i) = cv(i)*scale_factor;
            log<LOG_DEBUG>(L"%1% || ErrorBand bin %2% %3% %4% %5% %6% ") % __func__ % i % cv(i) % ehi % elo % scale_factor ;
        }
        return ebar;
    }


    void plot_channels(const std::string &filename, const PROconfig &config, std::optional<PROspec> cv, std::optional<PROspec> best_fit, std::optional<PROdata> data, std::optional<PROerrorbar> errband, std::optional<PROerrorbar> posterrband, std::vector<TPaveText> &texts, PlotBounds &bounds, PlotOptions opt, int other_index) {
        TCanvas c;
        c.Print((filename+"[").c_str());

        std::map<std::string, std::unique_ptr<TH1D>> cv1dhists;
        std::map<std::string, std::unique_ptr<TH2D>> cv2dhists;
        if(cv){
            cv1dhists = getCV1DHists(*cv, config, (bool)(opt & PlotOptions::BinWidthScaled), other_index);
            cv2dhists = getCV2DHists(*cv, config, (bool)(opt & PlotOptions::BinWidthScaled), other_index);
        }

        std::string ytitle = bool(opt&PlotOptions::AreaNormalized)
            ? "Area Normalized"
            : bool(opt&PlotOptions::BinWidthScaled) 
            ? "Events/GeV" 
            : "Events";

        size_t global_subchannel_index = 0;
        size_t global_subchannel_index_2d = 0;
        size_t global_channel_index = 0;
        int channel_start = 0;
        for(size_t mode = 0; mode < config.m_num_modes; ++mode) {
            for(size_t det = 0; det < config.m_num_detectors; ++det) {
                for(size_t channel = 0; channel < config.m_num_channels; ++channel) {

                    size_t channel_nbins = config.m_channel_variable_bins[channel][other_index].NBinsAlong(0);
                    Eigen::VectorXf bf_spec;
                    if(best_fit) {
                        bf_spec = config.GetChannelVariableBins(0, other_index).ProjectSpectra(CollapseMatrix(config, best_fit->Spec(), other_index), 0);//TODO, get N-dim compatability?                        
                    }

                    if(config.m_variable_dims.at(channel) == 2){
                        std::string joined_title = config.m_channel_variable_units[channel][other_index];
                        string del = ",";
                        auto pos = joined_title.find(del);
                        std::string xtitle2d = joined_title.substr(0, pos);
                        std::string ytitle2d = joined_title.erase(0, pos + del.length());
                        std::string hist_title = config.m_detector_plotnames[det]  + " "+ config.m_channel_plotnames[channel]+" CV;"+xtitle2d+";"+ytitle2d;
                        size_t channel_nbins_x = config.m_channel_variable_bins[channel][other_index].NBinsAlong(0);
                        std::vector<float> edges_x = config.m_channel_variable_bins[channel][other_index].Edges(0);
                        size_t channel_nbins_y = config.m_channel_variable_bins[channel][other_index].NBinsAlong(1);
                        std::vector<float> edges_y = config.m_channel_variable_bins[channel][other_index].Edges(1);
                        const std::string& subchannel_name  = config.m_fullnames[global_subchannel_index_2d];
                        TH2D* cv_hist = cv2dhists[subchannel_name].get();
                        TPad p2d("p2d", "p2d", 0, 0, 1, 1);
                        p2d.cd();
                        cv_hist->SetTitle(hist_title.c_str());
                        cv_hist->Draw("colz");
                        c.cd();
                        p2d.Draw();
                        c.Print(filename.c_str());

                        hist_title = config.m_detector_plotnames[det]  + " "+ config.m_channel_plotnames[channel]+" Best-Fit;"+xtitle2d+";"+ytitle2d;
                        TH2D bf_hist(hist_title.c_str(),hist_title.c_str(), channel_nbins_x, edges_x.data(), channel_nbins_y, edges_y.data());
                        Eigen::VectorXf tmp_bf = best_fit->Spec();
                        for(int xbin = 0; xbin < channel_nbins_x; xbin++){
                            for(size_t ybin = 0; ybin < channel_nbins_y; ybin++) {
                                bf_hist.SetBinContent(xbin+1, ybin, tmp_bf(xbin*channel_nbins_y+ybin));
                            }
                        }

                        TPad pbfd("pbfd", "pbfd", 0, 0, 1, 1);
                        pbfd.cd();
                        bf_hist.SetTitle(hist_title.c_str());
                        bf_hist.Draw("colz");
                        c.cd();
                        pbfd.Draw();
                        c.Print(filename.c_str());
                        global_subchannel_index_2d += config.m_num_subchannels[channel];
                    }

                    std::vector<float> edges = config.m_channel_variable_bins[channel][other_index].Edges();
                    std::string xtitle = config.m_channel_variable_units[channel][other_index];

                    Color_t bfcol = TColor::GetColor(234, 67, 53);//ncie red
                    Color_t cvcol =  TColor::GetColor(66, 103, 210);//nice blue :)
                    if(!best_fit)cvcol=kBlack;

                    std::string hist_titles = config.m_mode_plotnames[mode]+" "+config.m_detector_plotnames[det]  + " "+ config.m_channel_plotnames[channel]+";"+xtitle+";"+ytitle;
                    //std::unique_ptr<TLegend> leg = std::make_unique<TLegend>(0.11,0.75,0.89,0.89); 4
                    std::unique_ptr<TLegend> leg = std::make_unique<TLegend>(0.38,0.69,0.89,0.89);
                    leg->SetNColumns(2);
                    leg->SetFillStyle(0);
                    leg->SetLineWidth(0);
                    TH1D cv_hist(("cv_hist"+std::to_string(global_channel_index)).c_str(), hist_titles.c_str(), channel_nbins, edges.data());
                    cv_hist.SetLineWidth(2);
                    cv_hist.SetLineColor(cvcol);
                    cv_hist.SetFillStyle(0);
                    for(size_t bin = 0; bin < channel_nbins; ++bin) {
                        cv_hist.SetBinContent(bin+1, 0);
                    }
                    if(bool(opt&PlotOptions::BinWidthScaled))
                        cv_hist.Scale(1, "width");

                    // Set up TPads for ratios, unused if ratio option not chosen
                    TPad p1("p1", "p1", 0, 0.25, 1, 1);
                    p1.SetBottomMargin(0);

                    TPad p2("p2", "p2", 0, 0, 1, 0.25);
                    p2.SetTopMargin(0);
                    p2.SetBottomMargin(0.3);

                    THStack *cvstack = NULL;
                    if(cv) {
                        if(bool(opt&PlotOptions::CVasStack)) cvstack = new THStack(std::to_string(global_channel_index).c_str(), "");
                        std::vector<std::pair<std::string, const char*>> subplots;
                        for(size_t subchannel = 0; subchannel < config.m_num_subchannels[channel]; ++subchannel){
                            const std::string& subchannel_name  = config.m_fullnames[global_subchannel_index];
                            if(bool(opt&PlotOptions::CVasStack)) {
                                cvstack->Add(cv1dhists[subchannel_name].get());
                                subplots.push_back({subchannel_name, config.m_subchannel_plotnames[channel][subchannel].c_str()});
                            }
                            cv_hist.Add(cv1dhists[subchannel_name].get());
                            ++global_subchannel_index;
                        }
                        if(bool(opt&PlotOptions::CVasStack)) {
                            for(size_t sc = subplots.size(); sc > 0; --sc)
                                leg->AddEntry(cv1dhists[subplots[sc-1].first].get(), subplots[sc-1].second ,"f");
                        }
                        if(bool(opt&PlotOptions::AreaNormalized)) {
                            float integral = cv_hist.Integral();
                            cv_hist.Scale(1 / integral);
                            if(bool(opt&PlotOptions::CVasStack)) {
                                TList *stlists = (TList*)cvstack->GetHists();
                                for(const auto&& obj: *stlists){
                                    ((TH1*)obj)->Scale(1/integral);
                                }
                            }
                        }

                        TH1 *leg_hack = (TH1*)cv_hist.Clone((std::string(cv_hist.GetTitle())+"leg_hack").c_str());
                        leg_hack->SetFillStyle(3144);
                        leg_hack->SetFillColorAlpha(cvcol, 0.2);
                        //leg_hack->SetFillColorAlpha(kGray+2, 0.2);
                        leg_hack->SetLineColor(cvcol);
                        leg_hack->SetLineWidth(2);

                        if(errband){
                            leg->AddEntry(leg_hack,"CV Prediction #pm 1#sigma" ,"fl"); 
                        }else{
                            leg->AddEntry(&cv_hist, "CV Prediction #pm 1#sigma", "l");
                        }
                    }

                    TGraphAsymmErrors *channel_errband = NULL;
                    if(errband) {
                        channel_errband = new TGraphAsymmErrors(&cv_hist);
                        // int channel_start =  config.GetCollapsedGlobalVariableBinStart(global_channel_index, other_index);

                        for(size_t bin = 0; bin < channel_nbins; ++bin) {
                            float scale = 1.0;

                            //log<LOG_DEBUG>(L"%1% || PARK IN bin %2% channel_nbins %3% channel_start %4% bin+channel_start %5% erriorN %6%") % __func__ % bin % channel_nbins % channel_start % int(bin+channel_start) % errband->error_point.size();
                            if(bool(opt&PlotOptions::AreaNormalized) || bool(opt&PlotOptions::BinWidthScaled)) {
                                scale = channel_errband->GetPointY(bin) / errband->error_point(bin+channel_start);
                            }
                            channel_errband->SetPointEYhigh(bin, scale*(errband->error_up(bin+channel_start)));
                            channel_errband->SetPointEYlow(bin, scale*(errband->error_down(bin+channel_start)));
                        }
                        channel_errband->SetFillStyle(3144);
                        //channel_errband->SetFillColorAlpha(kGray+2, 0.2);
                        channel_errband->SetFillColorAlpha(cvcol, 0.2);
                        channel_errband->SetLineColor(cvcol);
                        channel_errband->SetLineWidth(1);
                        //leg->AddEntry(channel_errband, "#pm 1#sigma", "f");
                    }

                    TH1D bf_hist(("bf"+std::to_string(global_channel_index)).c_str(), "", channel_nbins, edges.data());
                    if(best_fit) {
                        // int channel_start =  config.GetCollapsedGlobalVariableBinStart(global_channel_index, other_index);
                        size_t total_bins = config.GetChannelVariableBins(global_channel_index, other_index).NBins();
                        // Eigen::VectorXf this_bf_spec = config.GetChannelVariableBins(global_channel_index, other_index).ProjectSpectra(bf_spec(Eigen::seqN(channel_start, total_bins)), 0);

                        for(size_t bin = 0; bin < channel_nbins; ++bin) {
                            bf_hist.SetBinContent(bin+1, bf_spec(bin+channel_start));
                        }
                        //bf_hist.SetLineColor(TColor::GetColor(234, 67, 53)); // pastel red
                        bf_hist.SetLineColor(bfcol); 
                        bf_hist.SetLineStyle(kDashed); 
                        bf_hist.SetLineWidth(2);
                        //leg->AddEntry(&bf_hist, "Best Fit", "l");


                        TH1 *leg_hack = (TH1*)bf_hist.Clone((std::string(bf_hist.GetTitle())+"bf").c_str());
                        leg_hack->SetFillStyle(3254);
                        leg_hack->SetFillColor(bfcol);
                        leg_hack->SetLineColor(bfcol);
                        leg_hack->SetLineWidth(2);

                        if(errband){
                            leg->AddEntry(leg_hack,"Best Fit #pm 1#sigma (post-fit)" ,"fl"); 
                        }else{
                            leg->AddEntry(&bf_hist, "Best Fit #pm 1#sigma (post-fit)", "l");
                        }
                        //cv_hist.Draw("hist");

                        if(bool(opt&PlotOptions::BinWidthScaled))
                            bf_hist.Scale(1, "width");
                        if(bool(opt&PlotOptions::AreaNormalized))
                            bf_hist.Scale(1.0/bf_hist.Integral());
                    }

                    TGraphAsymmErrors *post_channel_errband = NULL;
                    if(posterrband) {
                        post_channel_errband = new TGraphAsymmErrors(&bf_hist);
                        // int channel_start =  config.GetCollapsedGlobalVariableBinStart(global_channel_index, other_index);
                        for(size_t bin = 0; bin < channel_nbins; ++bin) {
                            float scale = 1.0;
                            if(bool(opt&PlotOptions::AreaNormalized) || bool(opt&PlotOptions::BinWidthScaled)) {
                                scale = post_channel_errband->GetPointY(bin) / (posterrband->error_point(bin+channel_start));
                            }
                            post_channel_errband->SetPointEYhigh(bin, scale*(posterrband->error_up(bin+channel_start)));
                            post_channel_errband->SetPointEYlow(bin, scale*(posterrband->error_down(bin+channel_start)));
                        }
                        post_channel_errband->SetFillColor(bfcol);
                        post_channel_errband->SetFillStyle(3254);
                        post_channel_errband->SetLineColor(bfcol);
                        post_channel_errband->SetLineWidth(1);
                        //leg->AddEntry(post_channel_errband, "post-fit #pm 1#sigma", "f");
                    }
                    TH1D data_hist;
                    if(data) {

                        data_hist = data->toTH1D(config, global_channel_index, other_index);
                        data_hist.SetLineColor(kBlack);
                        data_hist.SetLineWidth(2);
                        data_hist.SetMarkerStyle(kFullCircle);
                        data_hist.SetMarkerColor(kBlack);
                        data_hist.SetMarkerSize(1);
                        std::string dat_str = "Data: ";
                        std::ostringstream oss;
                        int exponent = static_cast<int>(std::log10(std::abs(config.m_det_pot[det])));
                        float mantissa = config.m_det_pot[det]/ std::pow(10, exponent);
                        oss << std::fixed << std::setprecision(2) << mantissa << "x10^{" << exponent << "} POT";
                        dat_str+= oss.str();
                        leg->AddEntry(&data_hist,dat_str.c_str(), "lp");
                        if(bool(opt&PlotOptions::BinWidthScaled))
                            data_hist.Scale(1, "width");
                        if(bool(opt&PlotOptions::AreaNormalized))
                            data_hist.Scale(1.0/data_hist.Integral());
                    }


                    /*******************/
                    /* Draw everything */
                    /*******************/
                    double top_modifier = 1.35;

                    if(bool(opt&PlotOptions::DataMCRatio) || bool(opt&PlotOptions::DataPostfitRatio))
                        p1.cd();


                    if(cv) {
                        if(bool(opt&PlotOptions::CVasStack)) {
                            cvstack->SetMaximum(  bounds.hasBound("ymax") ? bounds.getBound("ymax") : std::max(top_modifier*cvstack->GetMaximum(), top_modifier*data_hist.GetMaximum()));
                            cvstack->Draw("hist");
                            cvstack->SetTitle(hist_titles.c_str());
                            cv_hist.Draw("same hist");

                        } else {
                            cv_hist.SetMaximum( bounds.hasBound("ymax") ? bounds.getBound("ymax") :  top_modifier*cv_hist.GetMaximum());

                            cv_hist.Draw("hist");
                            cv_hist.GetYaxis()->SetTitleSize(0.06);  
                            cv_hist.GetYaxis()->SetTitleOffset(0.75);
                            cv_hist.GetYaxis()->SetLabelSize(0.05);
                            cv_hist.SetMinimum(0.01);
                        }
                    }

                    if(errband) channel_errband->Draw("2 same");

                    if(best_fit) {
                        bf_hist.SetTitle(hist_titles.c_str());
                        if(cv) bf_hist.Draw("hist same");
                        else bf_hist.Draw("hist");
                    }

                    if(posterrband) post_channel_errband->Draw("2 same");

                    if(data) {
                        TGraphErrors *g = new TGraphErrors(data_hist.GetNbinsX());
                        float datmax =-999;
                        for (int i = 1; i <= data_hist.GetNbinsX(); ++i) {
                            double x = data_hist.GetBinCenter(i);
                            double y = data_hist.GetBinContent(i);
                            double ex = 0;
                            double ey = data_hist.GetBinError(i);
                            if(y>datmax){
                                datmax=y;
                            }
                            g->SetPoint(i - 1, x, y);
                            g->SetPointError(i - 1, ex, ey);
                            g->SetLineColor(kBlack);
                            g->SetLineWidth(2);
                            g->SetMarkerStyle(kFullCircle);
                            g->SetMarkerColor(kBlack);
                            g->SetMarkerSize(1);

                        }


                        if(cv || best_fit) {
                            g->Draw("PE1 same");
                            cv_hist.SetMaximum( bounds.hasBound("ymax") ? bounds.getBound("ymax") : std::max(cv_hist.GetMaximum(),top_modifier*datmax));

                        } else {
                            g->Draw("PE1");
                        }
                    }

                    if(texts.size()!=0) {
                        TLine *dummy_line = new TLine(0,0,0.1,0);
                        dummy_line->SetLineColor(kWhite);
                        dummy_line->SetLineWidth(0);
                        if(texts.size()==1){
                            //texts.front().Draw("same");
                            TText* text = (TText*)texts.front().GetListOfLines()->First();
                            const char* label = text->GetTitle(); 
                            leg->AddEntry(dummy_line,label,"l"); 
                        }else{
                            //texts[global_channel_index].Draw("same");
                            TText* text = (TText*)texts.at(global_channel_index).GetListOfLines()->First();
                            const char* label = text->GetTitle(); 
                            leg->AddEntry(dummy_line,label,"l"); 
                        }
                    }

                    leg->Draw("same");

                    TH1D *ratio, *one;
                    TGraphAsymmErrors *ratio_err;
                    if(bool(opt&PlotOptions::DataMCRatio) || bool(opt&PlotOptions::DataPostfitRatio)) {
                        p2.cd();

                        std::string y_title = bool(opt&PlotOptions::DataMCRatio) ? "Data/MC" : "Data/Best-Fit";
                        ratio = new TH1D(("rat"+std::to_string(global_channel_index)).c_str(), (";"+xtitle+";"+y_title).c_str(), channel_nbins, edges.data());
                        one = new TH1D(("one"+std::to_string(global_channel_index)).c_str(), (";"+xtitle+";"+y_title).c_str(), channel_nbins, edges.data());
                        ratio_err = new TGraphAsymmErrors(); 
                        *ratio_err = bool(opt&PlotOptions::DataMCRatio)
                            ? *channel_errband
                            : *post_channel_errband;


                       
                        float ymin = 1e9, ymax = -1e9;

                        for(size_t i = 0; i < channel_nbins; ++i) {
                            float numerator = data_hist.GetBinContent(i+1);
                            float denonminator = 
                                bool(opt&PlotOptions::DataMCRatio)
                                ? cv_hist.GetBinContent(i+1)
                                : bf_hist.GetBinContent(i+1);
                            float rat = numerator/denonminator;
                            if(isnan(rat)) rat = 1;
                            ratio->SetBinError(i+1, data_hist.GetBinError(i+1)/denonminator);
                            ratio->SetBinContent(i+1, rat);
                            one->SetBinContent(i+1, 1.0);
                            ratio_err->SetPointEYhigh(i, ratio_err->GetErrorYhigh(i)/ratio_err->GetPointY(i));
                            ratio_err->SetPointEYlow(i, ratio_err->GetErrorYlow(i)/ratio_err->GetPointY(i));
                            ratio_err->SetPointY(i, 1.0);
                            ymin = std::min(ymin, rat);
                            ymax = std::max(ymax, rat);


                        }
                        for (int i = 0; i < ratio_err->GetN(); ++i) {
                            float y, eyh, eyl;
                            y = ratio_err->GetPointY(i);
                            eyh = ratio_err->GetErrorYhigh(i);
                            eyl = ratio_err->GetErrorYlow(i);
                            ymin = std::min(ymin, y - eyl);
                            ymax = std::max(ymax, y + eyh);

                        }
                        float yrange = ymax - ymin;
                        float ylow = ymin - 0.05 * yrange;  // 15% padding below
                        float yhigh = ymax + 0.05 * yrange; // 15% padding above

                        float kmin = bounds.hasBound("ratmin") ? bounds.getBound("ratmin") : std::min(ylow,0.85f);
                        float kmax = bounds.hasBound("ratmax") ? bounds.getBound("ratmax") : std::max(yhigh,1.148f);
                        log<LOG_DEBUG>(L"%1% || RatioMin %2%  Max %3% (ylow %4%) %5% val %6%") % __func__ % kmin % kmax % ylow % int(bounds.hasBound("ratmin")) % bounds.getBound("ratmin");

                        if(!posterrband){
                            one->SetMinimum(kmin);
                            one->SetMaximum(kmax);

                        }else{
                           one->SetMinimum(kmin);
                           one->SetMaximum(kmax);
                        }

                        one->SetLineColor(kBlack);
                        one->SetLineStyle(kDashed);
                        one->Draw("hist");
                        one->SetTitle("");
                        one->GetYaxis()->SetTitleSize(0.15);
                        one->GetYaxis()->SetLabelSize(0.12);
                        one->GetXaxis()->SetTitleSize(0.14);
                        one->GetXaxis()->SetLabelSize(0.14);
                        one->GetYaxis()->SetTitleOffset(0.21);
                        one->GetXaxis()->SetTitleOffset(0.85);
                        ratio->SetLineColor(kBlack);
                        ratio->SetLineWidth(2);
                        ratio->SetMarkerStyle(kFullCircle);
                        ratio->SetMarkerColor(kBlack);
                        ratio->SetMarkerSize(1);

                        //ratio_err->SetFillColor(kRed);
                        //ratio_err->SetFillStyle(3345);
                        ratio_err->Draw("2 same");

                        ratio->Draw("PE1 E0 same");

                        c.cd();
                        p1.Draw();
                        p2.Draw();
                    }


                    TText *t = new TText();
                    t->SetNDC();                
                    t->SetTextFont(42);                          
                    t->SetTextSize(0.03);      
                    t->SetTextAlign(33);        
                    std::string pv = "PROfit v"+std::string(PROJECT_VERSION_STR);
                    t->DrawText(0.895, 0.955, pv.c_str()); 

                    c.Print(filename.c_str());

                    ++global_channel_index;
                    channel_start += config.m_channel_variable_bins[channel][other_index].NBinsAlong(0);
                }
            }
        }
        c.Print((filename+"]").c_str());
    }


    int plotPriorFractionalSystematicBreakdown(const PROconfig &config, const PROspec &spec, const PROsyst &allsplinesyst, std::string filename, int other_index) {
        //Input PROsyst needs to be the allsplinesyst for now

        std::vector<int> colors = {
            kAzure+1,      // Light blue
            kRed+1,        // Bright red
            kGreen+3,      // Medium green
            kOrange+7,      // Deep orange
            kBlue+2,        // Darker blue
            kViolet+2,      // Purple/violet
            kGray+1,         // Light gray
            kYellow+2,      // Golden yellow
            kTeal+3,        // Teal
            kPink+2,        // Pink
            kMagenta+2,     // Magenta
            kSpring+5      // Blue-green
        };

        std::vector<int> line_styles = {
            1,  // Solid (base style)
            1,  // Dashed
        };



        //some testing
        for (const auto& [syst_name, tags] : config.m_mcgen_variation_tags) {
            std::string tag_list;
            for (const auto& tag : tags) {
                tag_list += tag + ", ";
            }
            if (!tag_list.empty()) {
                tag_list.erase(tag_list.size() - 2);  // Remove trailing ", "
            }
            log<LOG_INFO>(L"Systematic: %1% | Tags: [%2%]") 
                % syst_name.c_str() 
                % tag_list.c_str();
        }

        //This is for prior everthing of course
        std::map<std::string,std::vector<std::string>> used_tags;
        for(const auto &name: allsplinesyst.covar_names){
            log<LOG_INFO>(L"%1% || Systematic %2% ") % __func__ % name.c_str();
            auto it = config.m_mcgen_variation_tags.find(name);
            if (it == config.m_mcgen_variation_tags.end()) {
                log<LOG_WARNING>(L"%1% || Systematic %2% not in tags map") % __func__ % name.c_str();
                continue;
            }
            const vector<std::string>& mtags = it->second;
            log<LOG_INFO>(L"%1% || -- has tags %2%") % __func__ %  mtags;
            for(auto &t: mtags){
                used_tags[t].push_back(name);
            }
        }
        for(const auto &[tag, vec]: used_tags) {
            log<LOG_INFO>(L"%1% || So for tag %2% we include %3%") % __func__ % tag.c_str() % vec;
        }



        int nTags = used_tags.size()+1;
        int gridCols = std::ceil(std::sqrt(nTags));
        int gridRows = std::ceil(nTags / float(gridCols));

        TCanvas c("c", "Systematics Comparison", gridCols*1600, gridRows*1200);  
        c.Print((filename+"[").c_str());
        c.Divide(gridCols, gridRows);

        Eigen::MatrixXf diag = spec.Spec().array().matrix().asDiagonal(); 
        Eigen::MatrixXf collapsed_diag = CollapseMatrix(config, diag);

        size_t global_channel_index = 0;
        for(size_t mode = 0; mode < config.m_num_modes; ++mode) {
            for(size_t det = 0; det < config.m_num_detectors; ++det) {
                for(size_t channel = 0; channel < config.m_num_channels; ++channel) {

                    std::string name = config.m_mode_plotnames[mode]+" "+config.m_detector_plotnames[det]+" "+config.m_channel_plotnames[channel]; 
                    c.Clear();
                    c.Divide(gridCols, gridRows);

                    int padIndex = 1;

                    std::vector<float> bin_edges = config.GetChannelVariableBins(global_channel_index,other_index).Edges();
                    size_t binstart = config.GetCollapsedGlobalVariableBinStart(global_channel_index,other_index);
                    size_t nbins = config.m_channel_variable_bins[channel][other_index].NBins();
                    std::vector<int> channel_bins(nbins);
                    std::iota(channel_bins.begin(), channel_bins.end(), binstart);

                    std::vector<TH1F*> vsums;
                    std::vector<std::string> vnames;
                    for (const auto &[tag, vec] : used_tags) {

                        c.cd(padIndex++);

                        TLegend* leg = new TLegend(0.11, 0.6, 0.89, 0.89);
                        leg->SetNColumns(3);
                        //leg->SetHeader(tag.c_str(), "C");  // Center-aligned header


                        TH1F* hsum = new TH1F( ("Sum_"+tag+"_"+std::to_string(global_channel_index)).c_str(), tag.c_str(), bin_edges.size()-1, bin_edges.data());
                        hsum->Reset();
                        std::vector<TH1F*> hvec;
                        int i =0;
                        for(const auto & systname:vec){

                            Eigen::MatrixXf frac_covariance = allsplinesyst.GrabMatrix(systname);
                            Eigen::MatrixXf full_covariance = diag*(frac_covariance)*diag;
                            Eigen::MatrixXf collapsed_full_covariance = CollapseMatrix(config, full_covariance);
                            Eigen::MatrixXf collapsed_frac_covariance = collapsed_diag.inverse()*collapsed_full_covariance*collapsed_diag.inverse();


                            //submatix ffractional
                            Eigen::MatrixXf channel_cov = collapsed_frac_covariance(channel_bins, channel_bins);

                            log<LOG_INFO>(L"%1% || Channel: %2%/%3% | Det: %4%/%5% | Mode: %6%/%7% | Tag: %8% | Syst: %9% | Bins: %10% [%11%:%12%]") 
                                % __func__ 
                                % channel % config.m_num_channels
                                % det % config.m_num_detectors
                                % mode % config.m_num_modes
                                % tag.c_str() 
                                % systname.c_str()
                                % nbins
                                % binstart % (binstart + nbins - 1);

                            int color_idx = i % colors.size();
                            int style_idx = (i / 4) % line_styles.size();  
                            i++;

                            TH1F* h = new TH1F((tag+"_Channel_"+std::to_string(global_channel_index)+"_"+std::to_string(i)).c_str(), tag.c_str(), bin_edges.size()-1, bin_edges.data());

                            for (size_t i = 0; i < nbins; ++i) {
                                h->SetBinContent(i+1, sqrt(channel_cov(i,i)));
                                hsum->SetBinContent(i+1, hsum->GetBinContent(i+1)+channel_cov(i,i));
                            }

                            const std::string &plotname = config.m_mcgen_variation_plotname_map.at(systname);
                            leg->AddEntry(h, plotname.c_str(), "l");
                            h->SetLineColor(colors[color_idx]);
                            h->SetLineStyle(line_styles[style_idx]);
                            hvec.push_back(h);

                        }//end syst
                        for (size_t i = 0; i < nbins; ++i) {
                            hsum->SetBinContent(i+1, sqrt(hsum->GetBinContent(i+1)));
                        }
                        leg->AddEntry(hsum,"Sum","l");

                        hsum->SetXTitle(config.m_channel_plotnames[channel].c_str());
                        hsum->SetYTitle("Fractional Uncertainty");
                        hsum->SetLineColor(kBlack);
                        hsum->SetLineWidth(2);
                        hsum->SetLineStyle(1);
                        hsum->SetMinimum(0);
                        hsum->SetStats(0);  
                        hsum->Draw("HIST");
                        hsum->SetMaximum(hsum->GetMaximum()*1.7);
                        gPad->Modified();
                        gPad->Update();


                        vsums.push_back(hsum);
                        vnames.push_back(tag);
                        for(auto &h:hvec) h->Draw("HIST SAME");

                        leg->Draw();
                        
                        TText *t = new TText();
                        t->SetNDC();                
                        t->SetTextFont(42);                          
                        t->SetTextSize(0.03);      
                        t->SetTextAlign(33);        
                        std::string pv = "PROfit v"+std::string(PROJECT_VERSION_STR);
                        t->DrawText(0.895, 0.945, pv.c_str()); 

                    }//end tag


                    //and each sum of sums to wrap it off!
                    c.cd(padIndex++);

                    TH1F* hsum = new TH1F( ("USum_"+std::to_string(global_channel_index)).c_str(),("Summary! "+name).c_str(), bin_edges.size()-1, bin_edges.data());
                    hsum->Reset();
                    TLegend* leg = new TLegend(0.11, 0.6, 0.89, 0.89);
                    leg->SetNColumns(3);
                    std::vector<TH1F*> hvec;
                    for(size_t t=0; t< vsums.size(); t++){
                        int color_idx = t % colors.size();
                        for (size_t i = 0; i < nbins; ++i) {
                            hsum->SetBinContent(i+1, hsum->GetBinContent(i+1)+pow(vsums.at(t)->GetBinContent(i+1),2));
                        }
                        TH1F * h = (TH1F*)vsums.at(t)->Clone((to_string(global_channel_index)+vnames[t]).c_str());
                        leg->AddEntry(h, vnames[t].c_str(), "l");
                        h->SetLineColor(colors[color_idx]);
                        h->SetLineStyle(1);
                        h->SetLineWidth(1);
                        hvec.push_back(h);
                    }

                    for (size_t i = 0; i < nbins; ++i) {
                        hsum->SetBinContent(i+1, sqrt(hsum->GetBinContent(i+1)));
                    }
                    leg->AddEntry(hsum,"Sum","l");
                    hsum->SetXTitle(config.m_channel_plotnames[channel].c_str());
                    hsum->SetTitle(("Summary: "+name).c_str());
                    hsum->SetYTitle("Fractional Uncertainty");
                    hsum->SetLineColor(kBlack);
                    hsum->SetLineWidth(2);
                    hsum->SetLineStyle(1);
                    hsum->SetMinimum(0);
                    hsum->SetStats(0);  
                    hsum->Draw("HIST");
                    hsum->SetMaximum(hsum->GetMaximum()*1.7);
                    gPad->Modified();
                    gPad->Update();
                    for(auto &h:hvec) h->Draw("HIST SAME");
                    leg->Draw();

                    TText *t = new TText();
                    t->SetNDC();                
                    t->SetTextFont(42);                          
                    t->SetTextSize(0.03);      
                    t->SetTextAlign(33);        
                    std::string pv = "PROfit v"+std::string(PROJECT_VERSION_STR);
                    t->DrawText(0.895, 0.945, pv.c_str()); 


                    c.Print(filename.c_str());
                    global_channel_index++;
                }
            }
        }



        c.Print((filename+"]").c_str());
        return 0;
    };
}
