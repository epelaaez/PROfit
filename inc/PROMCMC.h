#ifndef PROMCMC_H
#define PROMCMC_H

#include "PROmetric.h"
#include "PROgress.h"
#include <Eigen/Eigen>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <optional>

#include "TStyle.h"
#include "TVirtualFFT.h"

namespace PROfit {

    // In PROplot.h, but I get a weird error when I include here because of recursive inclusion
    // Not sure why, that should be taken care of by include guards.
    void set_matrix_palette();
        
    template<class Target_FN, class Proposal_FN>
        class Metropolis {
            private:
                std::mt19937 rng;
                std::uniform_real_distribution<float> uniform;
                uint32_t seed;
                bool save_chain;

            public:
                Target_FN target;
                Proposal_FN proposal;
                Eigen::VectorXf current;
                std::vector<Eigen::VectorXf> chain;
                size_t naccept;

                Metropolis(Target_FN target, Proposal_FN proposal, const Eigen::VectorXf &initial, uint32_t seed, bool save_chain = true) 
                    : seed(seed), target(target), proposal(proposal), current(initial), save_chain(save_chain) {
                        rng.seed(seed);
                        naccept = 0;
                    }

                bool step() {
                    Eigen::VectorXf p = proposal(current);
                    float acceptance;
                    acceptance = proposal.within_bound(p) ? std::min(1.0f, target(p)/target(current) * proposal.P(current, p)/proposal.P(p, current)) : 0;
                    float u = uniform(rng);

                    if(u <= acceptance) {
                        //                      log<LOG_DEBUG>(L"%1% || APPROVED acc %2%, rng %3% and proposal: %4%  ") % __func__ % acceptance % u %p;
                        current = p;
                        naccept += 1;
                        return true;
                    }else{
                        //                      log<LOG_DEBUG>(L"%1% || REJECTED acc %2%, rng %3% and proposal: %4%  ") % __func__ % acceptance % u %p;
                    }
                    return false;
                }

                void run(size_t burnin, size_t steps, std::optional<std::function<void(const Eigen::VectorXf&)>> action = {}, PROgressBar *pbar = nullptr) {
                    const size_t pbar_stride = std::max<size_t>(1, (burnin + steps) / 10000);
                    for(size_t i = 0; i < burnin; i++) {
                        if constexpr(Proposal_FN::has_tune) {
                            proposal.tune(step());
                        } else {
                            step();
                        }
                        if(pbar && (i + 1) % pbar_stride == 0) pbar->set_progress(i + 1);
                    }
                    proposal.tune_mode = false;
                    for(size_t i = 0; i < steps; i++) {
                        step();
                        if(save_chain) chain.push_back(current);
                        if(action) (*action)(current);
                        if(pbar && (i + 1) % pbar_stride == 0) pbar->set_progress(burnin + i + 1);
                    }
                    if(pbar) {
                        pbar->finish();
                        std::cerr << std::endl;
                    }
                }

                void plot_autocorrelation(const std::string &filename, const std::vector<std::string> &param_names, size_t max_lag = 1000) const {
                    if(chain.size() == 0) {
                        log<LOG_ERROR>(L"%1% || Error: cannot calculate autocorrelation without a saved chain."
                                       L" Did you forget to run the Metropolis object, or tell the Metropolis"
                                       L" object to not save the chain?")
                            % __func__;
                        log<LOG_ERROR>(L"%1% || Not saving autocorrelations as a result.");
                        return;
                    }
                    long nparam = chain[0].size();
                    if(param_names.size() < (size_t)nparam) {
                        log<LOG_ERROR>(L"%1% || Passed in parameter names is not the same size (%2%) "
                                       L"as the number of parameters in each step of the chain (%3%).")
                            % __func__ % param_names.size() % nparam;
                        log<LOG_ERROR>(L"%1% || Not saving autocorrelations as a result.");
                        return;
                    }
                    std::vector<std::pair<TH1D*,TH1D*>> hs;
                    TH1D zero("z","",max_lag, 0, max_lag);
                    for(size_t i = 1; i <= max_lag; ++i) zero.SetBinContent(i, 0);
                    zero.SetLineStyle(kDashed);
                    TCanvas c;
                    c.Divide(2);
                    c.Print((filename + "[").c_str());
                    for(long i = 0; i < nparam; ++i) {
                        hs.emplace_back(new TH1D(("hautoc"+std::to_string(i)).c_str(), (param_names[i]+";lag;abs(autocorrelation)").c_str(), max_lag, 0, max_lag), new TH1D(("h2autoc"+std::to_string(i)).c_str(), (param_names[i]+";lag;autocorrelation").c_str(), max_lag, 0, max_lag));
                        int n = chain.size();
                        std::vector<double> values;
                        values.reserve(n);
                        float mean = 0;
                        for(const auto &step : chain) {
                            values.push_back(step(i));
                            mean += step(i);
                        }
                        mean /= n;
                        for(double &v : values) v -= mean;
                        TVirtualFFT *fft = TVirtualFFT::FFT(1, &n, "R2C");
                        fft->SetPoints(values.data());
                        fft->Transform();
                        std::vector<double> fft_pts;
                        std::vector<double> ims(n, 0); // Dummy vector to hold imag value of 0 for each point.
                        for(int k = 0; k < n; ++k) {
                            double re, im;
                            fft->GetPointComplex(k, re, im);
                            fft_pts.push_back(re*re+im*im);
                        }
                        TVirtualFFT *ifft = TVirtualFFT::FFT(1, &n, "C2R");
                        ifft->SetPointsComplex(fft_pts.data(), ims.data());
                        ifft->Transform();
                        double lag0, klag;
                        lag0 = ifft->GetPointReal(0);
                        for(int k = 0; k < n && k < max_lag; ++k) {
                            klag = ifft->GetPointReal(k);
                            hs.back().first->SetBinContent(k+1, std::abs(klag/lag0));
                            hs.back().second->SetBinContent(k+1, klag/lag0);
                        }
                        log<LOG_INFO>(L"%1% || Lag %2% autocorrelation for parameter %3% is %4%.")
                            % __func__ % std::min(max_lag, (size_t)n) % param_names[i].c_str() % (klag/lag0);
                        c.cd(1);
                        gPad->SetLogy(1);
                        hs.back().first->Draw("l");
                        c.cd(2);
                        gPad->SetLogy(0);
                        hs.back().second->Draw("l");
                        zero.Draw("l same");
                        c.Print(filename.c_str());
                    }
                    c.Clear();
                    gStyle->SetPalette(kCool);
                    TLegend leg(0.3, 0.6, 0.89, 0.89);
                    leg.SetNColumns(2);
                    leg.SetFillStyle(0);
                    leg.SetLineWidth(0);
                    int i = 0;
                    for(auto [h, _] : hs) {
                        gPad->SetLogy(1);
                        h->SetTitle(";lag;abs(autocorrelation)");
                        h->SetMinimum(1e-5);
                        h->Draw("plclsame");
                        leg.AddEntry(h, param_names[i++].c_str(), "l");
                    }
                    leg.Draw("same");
                    c.Print(filename.c_str());
                    gPad->SetLogy(0);
                    i = 0;
                    TLegend leg2(0.3, 0.6, 0.89, 0.89);
                    leg2.SetNColumns(2);
                    leg2.SetFillStyle(0);
                    leg2.SetLineWidth(0);
                    zero.SetTitle(";lag;autocorrelation");
                    zero.SetMinimum(-0.07);
                    zero.SetMaximum(1.05);
                    zero.Draw("l");
                    for(auto [_, h] : hs) {
                        gPad->SetLogy(0);
                        h->SetTitle(";lag;autocorrelation");
                        h->Draw("plclsame");
                        leg2.AddEntry(h, param_names[i++].c_str(), "l");
                    }
                    leg2.Draw("same");
                    c.Print(filename.c_str());
                    c.Print((filename + "]").c_str());
                    set_matrix_palette();
                }

        };

    struct simple_target {
        PROmetric &metric;

        float operator()(Eigen::VectorXf &value) {
            Eigen::VectorXf empty = value;
            return std::exp(-0.5f*metric(value, empty, false));
        }
    };

    struct prior_only_target {
        PROmetric &metric;

        float operator()(Eigen::VectorXf &value) {
            Eigen::VectorXf nuisance = value.segment(metric.GetModel().nparams, metric.GetSysts().GetNSplines());
            return std::exp(-0.5f*metric.Pull(nuisance));
        }
    };

    struct simple_proposal {
        PROmetric &metric;
        uint32_t seed;
        float width;
        std::vector<int> fixed;
        std::mt19937 rng;
        static constexpr bool has_tune = true;
        std::vector<bool> accepted_list;
        size_t tune_calls = 0;
        float last_acceptance = -1;
        float last_shift;

        simple_proposal(PROmetric &metric, uint32_t seed, float width = 0.2, std::vector<int> fixed = {}) 
            : metric(metric), seed(seed), width(width), fixed(fixed), rng(seed), last_shift(width) {
                accepted_list = std::vector(1000, false);
            }

        Eigen::VectorXf operator()(Eigen::VectorXf &current) {
            Eigen::VectorXf ret = current;
            long int nparams = metric.GetModel().nparams;
            for(long int i = 0; i < ret.size(); ++i) {
                if(std::find(fixed.begin(), fixed.end(), i) != std::end(fixed)) continue;
                if(i < nparams) {
                    float lo = ret(i) - width;
                    float hi = ret(i) + width;
                    std::uniform_real_distribution<float> ud(lo, hi);
                    ret(i) = ud(rng);
                } else if(metric.GetSysts().spline_lo[i-nparams] == 0) {
                    // Currently there's some weird behavior with the 0-1 systematics
                    // which using a uniform distribution seems to fix
                    // TODO: How to use width with a uniform distribution
                    //float lo = metric.GetSysts().spline_lo[i-nparams];
                    //float hi = metric.GetSysts().spline_hi[i-nparams];
                    float lo = ret(i) - width;
                    float hi = ret(i) + width;
                    std::uniform_real_distribution<float> ud(lo, hi);
                    ret(i) = ud(rng);
                } else {
                    std::normal_distribution<float> nd(current(i), width);
                    float proposed_value = nd(rng);
                    ret(i) = proposed_value;
                    //ret(i) = std::clamp(proposed_value, metric.GetSysts().spline_lo[i-nparams], metric.GetSysts().spline_hi[i-nparams]);
                }
            }
            return ret;
        }

        float P(const Eigen::VectorXf &value, const Eigen::VectorXf &given) {
            float prob = 1.0;
            long int nparams = metric.GetModel().nparams;
            for(long int i = 0; i < value.size(); ++i) {
                if(std::find(fixed.begin(), fixed.end(), i) != std::end(fixed)) continue;
                if(i < nparams) {
                    //float diff = metric.GetModel().ub(i) - metric.GetModel().lb(i);
                    //if(std::isinf(diff)) diff = 5;
                    //prob *= 1.0f / diff;
                    prob *= 1.0f / (2 * width);
                }else if(metric.GetSysts().spline_lo[i-nparams] == 0) {
                    //float lo = metric.GetSysts().spline_lo[i-nparams];
                    //float hi = metric.GetSysts().spline_hi[i-nparams];
                    //prob *= 1.0f / (hi - lo);
                    prob *= 1.0f / (2 * width);
                } else {
                    //if(value(i) <= metric.GetSysts().spline_lo[i-nparams] || value(i) >= metric.GetSysts().spline_hi[i-nparams] || 
                    //   given(i) <= metric.GetSysts().spline_lo[i-nparams] || given(i) >= metric.GetSysts().spline_hi[i-nparams]) {
                    //    // Due to bounds, use CDF to get total probability value is <= bound
                    //    // Symmetry makes this work for upper bound as well
                    //    float v = std::clamp(value(i), metric.GetSysts().spline_lo[i-nparams], metric.GetSysts().spline_hi[i-nparams]);
                    //    float g = std::clamp(given(i), metric.GetSysts().spline_lo[i-nparams], metric.GetSysts().spline_hi[i-nparams]);
                    //    prob *= 0.5f * (1.0f + std::erff((v - g)/(std::sqrt(2.0f)*width)));
                    //    //prob = 0;
                    //} else {
                    prob *= (1.0f / std::sqrt(2 * M_PI * width * width))
                        * std::exp(-(value(i) - given(i))*(value(i) - given(i))/(2 * width * width));
                    //}
                }
            }
            return prob;
        }

        bool within_bound(const Eigen::VectorXf &value) {
            long int nparams = metric.GetModel().nparams;
            for(long int i = 0; i < value.size(); ++i) {
                if(std::find(fixed.begin(), fixed.end(), i) != std::end(fixed)) continue;
                if(i < nparams) {
                    if(value(i) > metric.GetModel().ub(i) || value(i) < metric.GetModel().lb(i) || value(i) < -5.0f)
                        return false;
                } else {
                    size_t si = i - nparams;
                    float lo, hi;
                    if(metric.GetSysts().spline_has_restrict[si]) {
                        lo = metric.GetSysts().spline_restrict_lo[si];
                        hi = metric.GetSysts().spline_restrict_hi[si];
                    } else if(metric.GetSysts().spline_hi[si] == 1.0f) {
                        lo = -1.0f; hi = 1.0f;
                    } else {
                        lo = metric.GetSysts().spline_lo[si];
                        hi = metric.GetSysts().spline_hi[si];
                    }
                    if(value(i) < lo || value(i) > hi) return false;
                }
            }
            return true;
        }

        void tune(bool accepted) {
            accepted_list[tune_calls % 1000] = accepted;
            if(++tune_calls % 1000 == 0) {
                float acceptance = std::count(accepted_list.begin(), accepted_list.end(), true) / 1000.0f;
                if(acceptance < 0.20 || acceptance > 0.30) {
                    if(std::abs(acceptance - 0.234) < std::abs(last_acceptance - 0.234)) {
                        if(last_acceptance < 0) {
                            width *= 1.25; // Default first step
                            last_shift = 0.25 * width;
                        } else {
                            width += last_shift;
                        }
                    } else { // Moved too far
                        width += -0.5 * last_shift;
                        last_shift *= -0.5;
                    }
                }
                for(size_t i = 0; i < 1000; ++i) accepted_list[i] = false;
                last_acceptance = acceptance;
            }
        }
    };

    struct adaptive_proposal {
        PROmetric &metric;
        uint32_t seed;
        Eigen::MatrixXf width;
        std::vector<int> fixed;//fixed and active are opposite. usually active is 
        std::vector<int> active;
        std::mt19937 rng;
        static constexpr bool has_tune = true;
        std::vector<Eigen::VectorXf> proposed;
        Eigen::VectorXf last_proposed;
        Eigen::VectorXf last_accepted;
        Eigen::VectorXf mean;
        Eigen::MatrixXf cov;
        size_t tune_calls = 0;
        float scale = 5.66;
        float beta = 1.0;
        
        float diag_scale = 0.01;
        Eigen::MatrixXf diagL;
        Eigen::MatrixXf sub_diagL;

        // Adaptive scaling state
        std::vector<bool> accept_history;
        size_t adapt_window = 1000;  // window size for adaptation
        float target_accept = 0.234;
        float adapt_factor = 1.1;


        Eigen::MatrixXf sub_L;
        bool tune_mode;

        adaptive_proposal(PROmetric &metric, uint32_t seed, std::vector<int> fixed = {}) 
            : metric(metric), seed(seed), fixed(fixed), rng(seed) {
                int nparams = metric.GetModel().nparams + metric.GetSysts().GetNSplines();
                scale /= nparams - fixed.size();
                diag_scale /= nparams - fixed.size();
                width = Eigen::MatrixXf::Identity(nparams, nparams);
                mean = Eigen::VectorXf::Constant(nparams, 0);
                cov = Eigen::MatrixXf::Identity(nparams, nparams);
                Eigen::MatrixXf diag = Eigen::MatrixXf::Identity(nparams, nparams);
                Eigen::LLT<Eigen::MatrixXf> llt(diag_scale * diag);
                diagL = llt.matrixL();

                tune_mode = true;
                for (int i = 0; i < nparams; ++i) {
                    if (std::find(fixed.begin(), fixed.end(), i) == fixed.end()) {
                        active.push_back(i);
                    }
                }
                //grab the bits that correspond to the active only.
                sub_diagL = Eigen::MatrixXf::Identity(active.size(), active.size());
                sub_diagL = diagL(active, active);  
                last_accepted = Eigen::VectorXf::Zero(nparams);

            }

        Eigen::VectorXf operator()(Eigen::VectorXf &current) {

            Eigen::VectorXf sub_throw1(active.size());
            Eigen::VectorXf sub_throw2(active.size());
            std::normal_distribution<float> nd(0.0f, 1.0f);
            for (size_t i = 0; i < active.size(); ++i) {
                sub_throw1(i) = nd(rng);
                sub_throw2(i) = nd(rng);
            }
            
            Eigen::MatrixXf inp = scale*width;//fulldim

            if(tune_mode)sub_L = ComputeSquareRootCovariance(inp(active, active)); //magic eigen indexing

            last_proposed = current;//fulldim
            last_proposed(active) += (1.0f - beta) * sub_L * sub_throw1 + beta * sub_diagL * sub_throw2; //More index nonsesne?!

            return last_proposed;
        }

        float P(const Eigen::VectorXf &, const Eigen::VectorXf &) {
            return 1;
        }


        bool within_bound(const Eigen::VectorXf &value) {
            long int nparams = metric.GetModel().nparams;
            for(long int i = 0; i < value.size(); ++i) {
                if(std::find(fixed.begin(), fixed.end(), i) != std::end(fixed)) continue;
                if(i < nparams) {
                    if(value(i) > metric.GetModel().ub(i) || value(i) < std::max(metric.GetModel().lb(i),-5.0f))
                        return false;
                } else {
                    size_t si = i - nparams;
                    float lo, hi;
                    if(metric.GetSysts().spline_has_restrict[si]) {
                        lo = metric.GetSysts().spline_restrict_lo[si];
                        hi = metric.GetSysts().spline_restrict_hi[si];
                    } else if(metric.GetSysts().spline_hi[si] == 1.0f) {
                        lo = -1.0f; hi = 1.0f;
                    } else {
                        lo = metric.GetSysts().spline_lo[si];
                        hi = metric.GetSysts().spline_hi[si];
                    }
                    if(value(i) < lo || value(i) > hi) return false;
                }
            }
            return true;
        }


        void tune(bool accepted) {

            //if (accepted) {
                ++tune_calls;
                if(accepted) last_accepted = last_proposed;
                Eigen::VectorXf delta = last_accepted - mean;
                mean += delta / tune_calls;
                if (tune_calls > 1) {
                    cov += (delta * (last_accepted - mean).transpose() - cov) / tune_calls;
                }
                for (int idx : fixed) {
                    cov.row(idx).setZero();
                    cov.col(idx).setZero();
                }

            //}
            accept_history.push_back(accepted);
            if (accept_history.size() == adapt_window) {
                float acc_rate = std::count(accept_history.begin(), accept_history.end(), true) / float(adapt_window);

                if (acc_rate > target_accept) {
                    scale *= adapt_factor;
                } else {
                    scale /= adapt_factor;
                }
                accept_history.clear();
                log<LOG_DEBUG>(L"%1% || Adaptive scale updated to %2%, acceptance rate = %3%") % __func__ % scale % acc_rate;
            }

            if(tune_calls > (size_t)(4*last_proposed.size())) {
                Eigen::MatrixXf cov_pd = cov;
                for(long int i = 0; i < cov_pd.rows(); ++i)
                    cov_pd(i, i) += 1e-8;
                width = cov_pd;
                beta = tune_calls > (size_t)(100*last_proposed.size()) ? 0.05 : 0.5;
            }


        }

    };
};

#endif

