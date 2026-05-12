#include "PROcess.h"
#include "PROlog.h"
#include "PROspec.h"
#include "PROsyst.h"
#include "PROtocall.h"
#include "TH2D.h"

#include <Eigen/Eigen>

#include <algorithm>
#include <random>


namespace PROfit {

    PROspec FillSpectra(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst, const PROmodel &inmodel, const std::map<std::string, float> &inparams, bool binned, size_t var_index) {
        // default parameters
        Eigen::VectorXf params = Eigen::VectorXf::Zero(inmodel.nparams + insyst.GetNSplines());

        // default pulls are all 0. Set the default model parameters
        for (size_t ind = 0; ind < inmodel.nparams; ind++) params[ind] = inmodel.default_val[ind];

        // set parameters configured by user
        for (auto const &pair: inparams) {
            auto it1 = std::find(insyst.spline_names.begin(), insyst.spline_names.end(), pair.first);
            auto it2 = std::find(inmodel.param_names.begin(), inmodel.param_names.end(), pair.first);
            if (it1 != insyst.spline_names.end()) {
                int ind = std::distance(insyst.spline_names.begin(), it1);
                params[ind + inmodel.nparams] = pair.second;
            }
            else if (it2 != inmodel.param_names.end()) {
                int ind = std::distance(inmodel.param_names.begin(), it2);
                params[ind] = pair.second;
            }
            else {
                log<LOG_WARNING>(L"%1% | unable to find parameters %2% . Skipping.") % __func__ % pair.first.c_str();
            }
        }


        return FillSpectra(inconfig, inprop, insyst, inmodel, params, binned, var_index);
    }


    PROspec FillSpectra(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst, const PROmodel &inmodel, const Eigen::VectorXf &params, FillSpectraCache &cache, bool binned, size_t var_index) {
        // Unbinned path can't be cleanly split phys/syst (per-event Fill mixes them).
        // Fall back, invalidate cache.
        if(!binned) {
            cache.invalidate();
            return FillSpectra(inconfig, inprop, insyst, inmodel, params, binned, var_index);
        }

        Eigen::VectorXf phys = params.segment(0, inmodel.nparams);
        Eigen::VectorXf shifts = params.segment(inmodel.nparams, params.size() - inmodel.nparams);

        const bool ctx_changed = (cache.last_var_index != (int)var_index ||
                                  cache.last_syst_ptr != &insyst ||
                                  cache.last_model_ptr != &inmodel);

        // ---- Systematic-weights half (depends only on shifts) ----
        // Three branches for the systw vector:
        //   * full_systw_hit    : shifts identical to cached -> reuse cache.last_systw.
        //   * try_incremental   : shifts differs in EXACTLY one spline-range element
        //                          -> Tier 1.3: divide out old factor, multiply in new
        //                          factor for that one spline. Cache stays pinned to
        //                          its "central" state (no write-back).
        //   * full recompute    : everything else (size mismatch, ctx change, multi-diff,
        //                          or incremental gave a divide-by-near-zero on some bin).
        //                          Re-runs the full loop AND populates central_factors.
        const size_t       nbins_var      = inconfig.m_num_variable_bins_total[var_index];
        const Eigen::Index nsplines_e     = (Eigen::Index)insyst.GetNSplines();
        const bool         size_match     = (cache.last_shifts.size() == shifts.size());
        const bool         factors_valid  = (cache.central_factors.cols() == nsplines_e &&
                                             cache.central_factors.rows() == (Eigen::Index)nbins_var);
        const bool         diff_check_ok  = !ctx_changed && size_match && factors_valid;

        int diff_count = 0;
        int diff_idx   = -1;
        if(diff_check_ok) {
            for(Eigen::Index i = 0; i < shifts.size(); ++i) {
                if(cache.last_shifts(i) != shifts(i)) {
                    diff_idx = (int)i;
                    if(++diff_count > 1) break;
                }
            }
        }
        const bool full_systw_hit  = diff_check_ok && (diff_count == 0);
        const bool try_incremental = diff_check_ok && (diff_count == 1)
                                                   && (diff_idx >= 0)
                                                   && (diff_idx < (int)insyst.GetNSplines());

        Eigen::VectorXf       systw_local;            // populated only on the incremental path
        const Eigen::VectorXf *systw_to_use = nullptr; // points to whichever vector we'll combine

        if(full_systw_hit) {
            systw_to_use = &cache.last_systw;
        } else if(try_incremental) {
            const size_t j       = (size_t)diff_idx;
            const size_t binning = insyst.spline_binnings[j];

            Eigen::VectorXf new_factor_j(nbins_var);
            if(binning == var_index) {
                for(size_t k = 0; k < nbins_var; ++k)
                    new_factor_j(k) = insyst.GetSplineShift((int)j, shifts(j), (int)k);
            } else {
                const size_t nbins_binning = inconfig.m_num_variable_bins_total[binning];
                Eigen::VectorXf spline_shifts_one(nbins_binning);
                for(size_t b = 0; b < nbins_binning; ++b)
                    spline_shifts_one(b) = insyst.GetSplineShift((int)j, shifts(j), (int)b);
                const auto &hist = inprop.variable_hist_storage(binning, var_index);
                Eigen::VectorXf weighted_sum   = hist.transpose() * spline_shifts_one;
                Eigen::VectorXf unweighted_sum = hist.colwise().sum().transpose();
                for(size_t k = 0; k < nbins_var; ++k)
                    new_factor_j(k) = (unweighted_sum(k) > 0) ? weighted_sum(k) / unweighted_sum(k)
                                                              : 1.0f;
            }

            // Apply division+multiplication into a fresh local systw without touching cache.
            // If any cached factor for spline j is too small, fall back to full recompute.
            constexpr float kTiny = 1e-30f;
            systw_local.resize(nbins_var);
            bool ok = true;
            for(size_t k = 0; k < nbins_var; ++k) {
                const float old_f = cache.central_factors(k, (Eigen::Index)j);
                if(std::abs(old_f) < kTiny) { ok = false; break; }
                systw_local(k) = cache.last_systw(k) / old_f * new_factor_j(k);
            }

            if(ok) {
                systw_to_use = &systw_local;
                // CRITICAL: do NOT update cache.last_shifts, cache.last_systw, or
                // cache.central_factors. Cache stays pinned to the central point so
                // subsequent single-shift gradient calls also hit the incremental path.
            }
            // else fall through to full recompute below
        }

        if(systw_to_use == nullptr) {
            // Full recompute: rebuild systw AND populate per-spline factors. The
            // existing math is preserved bit-for-bit; we just stash each spline's
            // contribution as we go.
            Eigen::VectorXf systw = Eigen::VectorXf::Constant(nbins_var, 1);
            Eigen::MatrixXf factors(nbins_var, insyst.GetNSplines());
            for(int i = 0; i < (int)insyst.GetNSplines(); ++i) {
                size_t binning = insyst.spline_binnings[i];
                if(binning == var_index) {
                    for(size_t k = 0; k < nbins_var; ++k) {
                        const float f = insyst.GetSplineShift(i, shifts(i), (int)k);
                        factors(k, i) = f;
                        systw(k) *= f;
                    }
                } else {
                    const size_t nbins_binning = inconfig.m_num_variable_bins_total[binning];
                    Eigen::VectorXf spline_shifts_loc(nbins_binning);
                    for(size_t b = 0; b < nbins_binning; ++b)
                        spline_shifts_loc(b) = insyst.GetSplineShift(i, shifts(i), (int)b);
                    const auto &hist = inprop.variable_hist_storage(binning, var_index);
                    Eigen::VectorXf weighted_sum   = hist.transpose() * spline_shifts_loc;
                    Eigen::VectorXf unweighted_sum = hist.colwise().sum().transpose();
                    for(size_t k = 0; k < nbins_var; ++k) {
                        const float f = (unweighted_sum(k) > 0) ? weighted_sum(k) / unweighted_sum(k)
                                                                : 1.0f;
                        factors(k, i) = f;
                        systw(k) *= f;
                    }
                }
            }
            cache.last_systw      = std::move(systw);
            cache.last_shifts     = shifts;
            cache.central_factors = std::move(factors);
            systw_to_use          = &cache.last_systw;
        }

        // ---- Physics-result half (depends only on phys) ----
        const bool phys_changed = ctx_changed ||
                                  cache.last_phys.size() != phys.size() ||
                                  cache.last_phys != phys;
        if(phys_changed) {
            Eigen::VectorXf result;
            if(inmodel.is_trivial) {
                result = inmodel.H_combined[var_index].col(0);
            } else {
                const size_t N_ivars = inmodel.ivars.size();
                std::vector<size_t> ivar_sizes(N_ivars);
                for(size_t k = 0; k < N_ivars; ++k)
                    ivar_sizes[k] = inprop.variable_midbin[inmodel.ivars[k]].size();

                std::vector<std::vector<float>> var_arrs(N_ivars, std::vector<float>(inmodel.n_phys_bins));
                for(long int flat = 0; flat < inmodel.n_phys_bins; ++flat) {
                    long int rem = flat;
                    for(int k = (int)N_ivars - 1; k >= 0; --k) {
                        var_arrs[k][flat] = inprop.variable_midbin[inmodel.ivars[k]][rem % ivar_sizes[k]];
                        rem /= (long int)ivar_sizes[k];
                    }
                }

                auto probs = inmodel.get_probs(phys, var_arrs);
                Eigen::Map<const Eigen::VectorXf> probs_flat(probs.data(), probs.size());
                result = inmodel.H_combined[var_index] * probs_flat;
            }
            cache.last_result = std::move(result);
            cache.last_phys = phys;
        }

        cache.last_var_index = (int)var_index;
        cache.last_syst_ptr = &insyst;
        cache.last_model_ptr = &inmodel;

        // systw_to_use points to either cache.last_systw (full hit / full recompute) or
        // to the local perturbed vector built by the Tier 1.3 incremental path.
        Eigen::VectorXf final_spec  = systw_to_use->cwiseProduct(cache.last_result);
        Eigen::VectorXf final_error = final_spec.array().abs().sqrt();
        return PROspec(final_spec, final_error);
    }


    PROspec FillSpectra(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst, const PROmodel &inmodel, const Eigen::VectorXf &params, bool binned, size_t var_index){
        PROspec myspectrum(inconfig.m_num_variable_bins_total[var_index]);
        Eigen::VectorXf phys   = params.segment(0, inmodel.nparams);
        Eigen::VectorXf shifts = params.segment(inmodel.nparams, params.size() - inmodel.nparams);

        if(binned) {
            //log<LOG_INFO>(L"%1% || Starting systw calculation %2%") % __func__ % var_index;
            //auto start_systw = std::chrono::high_resolution_clock::now();

            const size_t nbins_var = inconfig.m_num_variable_bins_total[var_index];

            // Whether spline_is_pre_migration has been populated (requires the full-syst constructor).
            // If not populated, treat all splines as post-migration (backward-compatible behaviour).
            const bool has_pre_mig_info = (insyst.spline_is_pre_migration.size() == insyst.GetNSplines());

            // Only visible-decay models (which override get_counts) have truth-E migration.
            // For non-decay models, all splines are applied at reco level as before.
            const bool model_has_migration = inmodel.uses_get_counts();

            // pre_mig_weight: per flat physics grid point, flux reweighting applied BEFORE the
            // decay energy migration (used to pre-scale N_truth for get_counts).  Length = n_phys_bins.
            // n_E (last phys_grid_sizes entry) = number of truth-E bins; E is the last (fastest-
            // varying) ivar, so E_idx = flat % n_E.
            Eigen::VectorXf pre_mig_weight = Eigen::VectorXf::Constant(inmodel.n_phys_bins, 1.0f);

            // post_mig_systw: per reco bin, cross-section/detector systematic weight applied
            // AFTER the migration.  Replaces the old monolithic systw.
            Eigen::VectorXf post_mig_systw = Eigen::VectorXf::Constant(nbins_var, 1.0f);

            // Iterate up to insyst.GetNSplines(), not shifts.size(): params may be
            // over-sized when shared across variables with different spline counts.
            for(int i = 0; i < (int)insyst.GetNSplines(); ++i) {
                size_t binning = insyst.spline_binnings[i];

                bool is_pre_migration = has_pre_mig_info && model_has_migration
                                        && insyst.spline_is_pre_migration[i];

                if(is_pre_migration) {
                    // Pre-migration flux spline: apply to the flat (L, E) physics grid.
                    // The spline must be binned in the truth-E variable (the last ivar);
                    // n_spline_bins must equal n_E (the number of truth-E bins).
                    const size_t n_E           = inmodel.phys_grid_sizes.back();
                    const size_t n_spline_bins = inconfig.m_num_variable_bins_total[binning];
                    if(n_spline_bins != n_E) {
                        log<LOG_ERROR>(L"%1% || Pre-migration flux spline '%2%' has %3% bins but "
                            L"the model has %4% truth-E bins. The spline must be binned in the "
                            L"truth-E variable (the last model ivar).") % __func__
                            % insyst.spline_names[i].c_str() % n_spline_bins % n_E;
                        exit(EXIT_FAILURE);
                    }
                    // Each flat bin inherits the flux weight from its truth-E index.
                    for(long int flat = 0; flat < inmodel.n_phys_bins; ++flat) {
                        size_t e_idx = flat % n_E;
                        pre_mig_weight(flat) *= insyst.GetSplineShift(i, shifts(i), e_idx);
                    }
                } else if(binning == var_index) {
                    // Post-migration, same binning — direct multiplication on reco bins.
                    for(size_t k = 0; k < nbins_var; ++k) {
                        post_mig_systw(k) *= insyst.GetSplineShift(i, shifts(i), k);
                    }
                } else {
                    // Post-migration, different binning — use matrix-vector multiplication.
                    const size_t nbins_binning = inconfig.m_num_variable_bins_total[binning];

                    Eigen::VectorXf spline_shifts(nbins_binning);
                    for(size_t j = 0; j < nbins_binning; ++j) {
                        spline_shifts(j) = insyst.GetSplineShift(i, shifts(i), j);
                    }
                    const auto& hist = inprop.variable_hist_storage(binning, var_index);

                    // weighted_sum[k] = sum_j(spline_shifts[j] * hist(j, k))
                    // unweighted_sum[k] = sum_j(hist(j, k))
                    Eigen::VectorXf weighted_sum   = hist.transpose() * spline_shifts;
                    Eigen::VectorXf unweighted_sum = hist.colwise().sum().transpose();

                    for(size_t k = 0; k < nbins_var; ++k) {
                        if(unweighted_sum(k) > 0) {
                            post_mig_systw(k) *= weighted_sum(k) / unweighted_sum(k);
                        }
                    }
                }
            }

            //auto end_systw = std::chrono::high_resolution_clock::now();
            //std::chrono::duration<double> duration_systw = end_systw - start_systw;
            //log<LOG_INFO>(L"%1% || systw calculation took %2% ms") % __func__ % (duration_systw.count() * 1000.);

            //log<LOG_INFO>(L"%1% || Starting le_arr building %2%") % __func__ % var_index;
            //auto start_le = std::chrono::high_resolution_clock::now();

            // Build var_arrs and the prediction.  Trivial models (empty ivars) short-circuit:
            // probs ≡ 1 and H_combined[var_index] has shape (n_reco, 1) already holding the
            // per-reco-bin event-weight sum, with no physics-grid coupling.
            Eigen::VectorXf result;
            if(inmodel.is_trivial) {
                result = inmodel.H_combined[var_index].col(0);
            } else {
                // var_arrs: one entry per ivar, each of length n_phys_bins (flat grid).
                // For 1-var: var_arrs[0] = midbin values of that var (n_phys_bins = n_ivar_bins).
                // For N-var: row-major product grid — var_arrs[k][flat] = midbin of ivar[k] at flat index.
                const size_t N_ivars = inmodel.ivars.size();
                std::vector<size_t> ivar_sizes(N_ivars);
                for(size_t k = 0; k < N_ivars; ++k)
                    ivar_sizes[k] = inprop.variable_midbin[inmodel.ivars[k]].size();

                std::vector<std::vector<float>> var_arrs(N_ivars, std::vector<float>(inmodel.n_phys_bins));
                for(long int flat = 0; flat < inmodel.n_phys_bins; ++flat) {
                    long int rem = flat;
                    for(int k = (int)N_ivars - 1; k >= 0; --k) {
                        var_arrs[k][flat] = inprop.variable_midbin[inmodel.ivars[k]][rem % ivar_sizes[k]];
                        rem /= (long int)ivar_sizes[k];
                    }
                }

                // Effective per-physics-bin oscillation probabilities.  Two paths:
                //   - Visible-decay models (uses_get_counts): pre-scale N_truth rows by
                //     pre_mig_weight so the flux reweight is applied at the PARENT neutrino's
                //     truth energy, then call get_counts and divide by the ORIGINAL N_truth.
                //     Pre-scaling N_truth weights the parent in both the oscillation term (A)
                //     at flat_dst and the decay-migration term (B) at each flat_src, since
                //     get_counts reads the passed N_truth at exactly those bins.
                //   - All other models: the simple get_probs path.  pre_mig_weight is always
                //     all-ones here (model_has_migration == false), so no flux reweight is lost.
                Eigen::MatrixXf probs;
                if(inmodel.uses_get_counts()) {
                    const Eigen::MatrixXf& N_truth = inmodel.get_N_truth();
                    Eigen::MatrixXf N_weighted = N_truth.array().colwise() * pre_mig_weight.array();
                    auto counts = inmodel.get_counts(phys, var_arrs, N_weighted);
                    auto N_arr  = N_truth.array();
                    probs = (N_arr > 0.0f).select(counts.array() / N_arr, 0.0f);
                } else {
                    probs = inmodel.get_probs(phys, var_arrs);
                }

                // Single GEMV: H_combined[var_index] has shape (n_reco, n_phys*J).
                // probs is (n_phys, J) column-major, so probs.data() = [col0 | col1 | ...] = probs_flat.
                Eigen::Map<const Eigen::VectorXf> probs_flat(probs.data(), probs.size());
                result = inmodel.H_combined[var_index] * probs_flat;
            }

            // Apply post-migration systematic weights (cross-section, detector) at reco level.
            Eigen::VectorXf final_spec = post_mig_systw.cwiseProduct(result);
            Eigen::VectorXf final_error = final_spec.array().abs().sqrt();
            myspectrum = PROspec(final_spec, final_error);

        } else {

            // Unbinned path: per-event var values, one vector per ivar.
            // For trivial models (empty ivars), skip var_arrs / get_probs entirely and use oscw = 1.
            Eigen::MatrixXf probs;
            if(!inmodel.is_trivial) {
                std::vector<std::vector<float>> var_arrs(inmodel.ivars.size(), std::vector<float>(inprop.NEvent()));
                for(size_t k = 0; k < inmodel.ivars.size(); ++k)
                    for(size_t i = 0; i < inprop.NEvent(); ++i)
                        var_arrs[k][i] = inprop.VariableValue(inmodel.ivars[k], i);

                probs = inmodel.get_probs(phys, var_arrs);
            }

            for(size_t i = 0; i < inprop.NEvent(); ++i) {
                float oscw = inmodel.is_trivial ? 1.0f : probs(i, inprop.model_rule[i]);
                float add_w = inprop.added_weights[i];
                const int reco_bin = inprop.VariableBinIndex(var_index, i);

                float systw = 1;
                for(int j = 0; j < shifts.size(); ++j) {
                    int binning = insyst.spline_binnings[j];
                    const int spline_bin = inprop.VariableBinIndex(binning, i);
                    systw *= insyst.GetSplineShift(j, shifts[j], spline_bin);
                }
                float finalw = oscw * systw * add_w;
                myspectrum.Fill(reco_bin, finalw);
            }
        }
        return myspectrum;
    }




    //*********************************************************************************************

    PROspec FillWeightedSpectrumFromHist(const PROconfig &inconfig, const PROpeller &inprop, std::vector<TH2D*> inweighthists, const PROmodel &inmodel, const Eigen::VectorXf &params, bool binned){
        PROspec myspectrum(inconfig.m_num_variable_bins_total[inconfig.i_prime]);
        Eigen::VectorXf phys   = params.segment(0, inmodel.nparams);
        Eigen::VectorXf shifts = params.segment(inmodel.nparams, params.size() - inmodel.nparams);

        int i_osc_tmp =1;
        if (binned) {
            std::vector<float> hist_w_arr;
            std::vector<float> le_arr;
            for(long int i = 0; i < inprop.variable_hist_storage(i_osc_tmp,inconfig.i_prime).rows(); ++i) {
                float le = inprop.variable_midbin[i_osc_tmp][i];
                float hist_w = 1.0 ;

                //Figure out what subchannel the event is in
                size_t subchan = inconfig.GetSubchannelIndexFromVariableGlobalBin(inprop.VariableBinIndex(i_osc_tmp, i), i_osc_tmp);
                std::string name = inconfig.m_fullnames[subchan];

                //Put name for ICARUS study here. How to handle more generically?
                if (name == "nu_ICARUS_numu_numucc") {
                    float pmom = 2;//static_cast<float>(inprop.pmom[i]);
                    float pcosth = 3;// static_cast<float>(inprop.pcosth[i]);
                    for (size_t j = 0; j<inweighthists.size(); ++j){
                        TH2D h = *inweighthists[j];
                        int bin = h.FindBin(pmom,pcosth);
                        hist_w *= h.GetBinContent(bin);
                    }
                }

                hist_w_arr.push_back(hist_w);
                le_arr.push_back(le);
            }

            auto probs = inmodel.get_probs(phys, {le_arr});

            // Weight each column of probs by hist_w, flatten, then single GEMV.
            Eigen::Map<const Eigen::VectorXf> hist_w_vec(hist_w_arr.data(), hist_w_arr.size());
            const long int n_phys = inmodel.n_phys_bins;
            const size_t   J      = inmodel.prob_types.size();
            Eigen::VectorXf probs_flat(n_phys * J);
            for(size_t j = 0; j < J; ++j)
                probs_flat.segment(j * n_phys, n_phys) = hist_w_vec.cwiseProduct(probs.col(j));

            Eigen::VectorXf result = inmodel.H_combined[inconfig.i_prime] * probs_flat;
            Eigen::VectorXf final_error = result.array().abs().sqrt();
            myspectrum = PROspec(result, final_error);

        } else {
            std::vector<float> le_arr;
            std::vector<float> hist_w_arr;
            std::vector<float> add_w_arr;
            for(size_t i = 0; i < inprop.NEvent(); ++i) {
                le_arr.push_back(inprop.VariableValue(inmodel.ivars[0], i));
                add_w_arr.push_back(inprop.added_weights[i]);

                float hist_w = 1.0;

                //Figure out what subchannel the event is in
                size_t subchan = inconfig.GetSubchannelIndexFromVariableGlobalBin(inprop.VariableBinIndex(i_osc_tmp, i), i_osc_tmp);
                std::string name = inconfig.m_fullnames[subchan];

                //Put name for ICARUS study here. How to handle more generically?
                if (name == "nu_ICARUS_numu_numucc") {
                    float pmom = 2;//static_cast<float>(inprop.pmom[i]);
                    float pcosth = 2;//static_cast<float>(inprop.pcosth[i]);

                    for (size_t j = 0; j<inweighthists.size(); ++j){
                        TH2D h = *inweighthists[j];
                        int bin = h.FindBin(pmom,pcosth);
                        hist_w *= h.GetBinContent(bin);
                    }
                }

                hist_w_arr.push_back(hist_w);
            }

            auto probs = inmodel.get_probs(phys, {le_arr});

            for(size_t i = 0; i < inprop.NEvent(); ++i) {
                float oscw = phys.size() != 0 ? probs(i, inprop.model_rule[i]) : 1.0f;

                float finalw = oscw * add_w_arr[i] * hist_w_arr[i];
                myspectrum.Fill(inprop.VariableBinIndex(inconfig.i_prime, i), finalw);
            }
        }
        return myspectrum;
    }

    PROspec FillSystRandomThrow(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst, const PROmodel &model, const PROspec &cvspec, const Eigen::VectorXf &cvparams, uint32_t seed, int var_index) {
        int nbins = inconfig.m_num_variable_bins_total[var_index],
        nbins_collapsed = inconfig.m_num_variable_bins_total_collapsed[var_index];
        Eigen::VectorXf spec = Eigen::VectorXf::Constant(nbins, 0);

        Eigen::VectorXf params = cvparams;

        // TODO: We should think about centralizing rng in a thread-safe/thread-aware way
        static std::mt19937 rng{seed};
        std::vector<std::normal_distribution<float>> d_spline;
        for(size_t i = 0; i < insyst.GetNSplines(); ++i)
            d_spline.emplace_back(insyst.spline_centers(i), insyst.spline_priors(i));
        std::normal_distribution<float> d_cov;
        std::vector<float> throws;
        //Eigen::VectorXf throwC = Eigen::VectorXf::Constant(inconfig.m_num_variable_bins_total[inconfig.i_prime], 0);
        Eigen::VectorXf throwC = Eigen::VectorXf::Constant(nbins_collapsed, 0);
        for(size_t i = 0; i < insyst.GetNSplines(); i++) {
            throws.push_back(d_spline[i](rng));
        }
        for(int i = 0; i < nbins_collapsed; i++)
            throwC(i) = d_cov(rng);

        //get actual splines 
        for(size_t i=0;i<throws.size();i++){
            params(i+model.nparams) = throws.at(i);
        }

        bool binned = true;//dont want to faf around with event by event here lets be honst
        if (binned){
          spec = FillSpectra(inconfig, inprop, insyst, model, params, binned, var_index).Spec();

        }else{//currently never run
            for(size_t i = 0; i<inprop.NEvent(); ++i){
                float add_w = inprop.added_weights[i]; 
                float systw = 1;
                for(size_t j = 0; j < throws.size(); ++j) {
                    int binning = insyst.spline_binnings[j];
                    const int spline_bin = inprop.VariableBinIndex(binning, i);
                    systw *= insyst.GetSplineShift(j, throws[j], spline_bin);
                }
                if(inprop.VariableBinIndex(var_index, i) >= 0) {
                    float finalw = systw * add_w;
                    spec(inprop.VariableBinIndex(var_index, i)) += finalw;
                }
            }
        }

        if(insyst.GetNCovar() == 0) {
            Eigen::VectorXf final_spec = CollapseMatrix(inconfig, spec, var_index);
            return PROspec(final_spec, final_spec.array().sqrt());
        }

        Eigen::MatrixXf decomp_cov = insyst.DecomposeFractionalCovariance(inconfig, cvspec.Spec());
        Eigen::VectorXf collapsed_spec = CollapseMatrix(inconfig, spec, var_index);
        Eigen::VectorXf final_spec = collapsed_spec + decomp_cov * throwC;

        //std::vector<float> stdVec(final_spec.data(), final_spec.data() + final_spec.size());
        //log<LOG_INFO>(L"%1% | final_spec is %2% ") % __func__ % stdVec;


        return PROspec(final_spec, final_spec.array().sqrt());
    }

    PROspec FillSplineRandomThrow(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst,  const PROmodel &model,  const Eigen::VectorXf &cvparams, int spline, uint32_t seed, int other_index) {
        int nbins =  inconfig.m_num_variable_bins_total[other_index];
        Eigen::VectorXf spec = Eigen::VectorXf::Constant(nbins, 0);

        // TODO: We should think about centralizing rng in a thread-safe/thread-aware way
        static std::mt19937 rng{seed};
        std::normal_distribution<float> d(insyst.spline_centers(spline), insyst.spline_priors(spline));
        float spline_throw = d(rng);
        Eigen::VectorXf params = cvparams;
        params(spline+model.nparams) = spline_throw;

        bool binned = true;//dont want to faf around with event by event here lets be honst
        if (binned){
            spec = FillSpectra(inconfig, inprop, insyst, model, params, binned, other_index).Spec();
        } 

        return PROspec(spec, spec.array().sqrt());
    }
};
