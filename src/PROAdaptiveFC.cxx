/**
 * @file PROAdaptiveFC.cxx
 * @brief Adaptive Feldman-Cousins pipeline — top-level mode dispatcher.
 *
 * run_adaptive_fc() routes each AdaptiveFCMode (build-mesh, init-bank,
 * print-bank, asimov, brazil) to the pipeline stages implemented in the
 * sibling translation units:
 *   - src/PROAdaptiveFCmesh.cxx  — throws, Wilks AMR prepass, meta-mesh
 *   - src/PROAdaptiveFCbank.cxx  — serialisation, PE worker/scheduler,
 *                                  asimov observables, classification
 *   - src/PROAdaptiveFCplot.cxx  — ROOT PDF / artifact output
 * Cross-TU internals are declared in src/PROAdaptiveFCinternal.h
 * (namespace PROfit::afc). The AMR per-point fit body and the mesh drawing
 * are shared with PROsurf via inc/PROmeshEval.h and inc/PROmeshPlot.h.
 */
#include "PROAdaptiveFC.h"
#include "PROAdaptiveFCinternal.h"

#include "PROlog.h"
#include "PROmodel.h"
#include "PROspec.h"
#include "PROcess.h"
#include "PROtocall.h"

#include <Eigen/Eigen>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace PROfit {

using namespace afc;

// ====================================================================
//  Helpers — axis name → parameter/spline index resolution.
//  Mirrors the lookup at bin/PROfit.cxx:1425-1457 used by the `surface`
//  command. Kept local because the parent function is large and not yet
//  factored out.
// ====================================================================

static size_t resolve_axis_index(const std::string &name,
                                 const PROmodel &model,
                                 const PROsyst  &systs,
                                 const PROconfig &config,
                                 size_t fallback)
{
    if (auto loc = std::find(model.param_names.begin(), model.param_names.end(), name);
        loc != model.param_names.end()) {
        return std::distance(model.param_names.begin(), loc);
    }
    // Spline axes index the FULL [physics | spline] parameter vector that the
    // callers pin/bound, so the spline position must be offset by nparams.
    if (auto loc = std::find(systs.spline_names.begin(), systs.spline_names.end(), name);
        loc != systs.spline_names.end()) {
        return model.nparams + std::distance(systs.spline_names.begin(), loc);
    }
    for (const auto &[xml_name, plot_name] : config.m_mcgen_variation_plotname_map) {
        if (name == plot_name) {
            if (auto loc = std::find(systs.spline_names.begin(), systs.spline_names.end(), xml_name);
                loc != systs.spline_names.end()) {
                return model.nparams + std::distance(systs.spline_names.begin(), loc);
            }
        }
    }
    log<LOG_WARNING>(L"%1% || resolve_axis_index: axis variable '%2%' not found; falling back to %3%.")
        % __func__ % name.c_str() % fallback;
    return fallback;
}

// ====================================================================
//  Section 6 — run_adaptive_fc (top-level dispatcher)
// ====================================================================

AdaptiveFCResult run_adaptive_fc(
    const PROconfig &config,
    const PROpeller &prop,
    const PROsyst   &systs,
    const PROfitterConfig &fitconfig,
    PROseed         &proseed,
    const Eigen::VectorXf &fakeDataParams,
    const PROdata   &data,
    const AdaptiveFCConfig &acfg,
    int nthreads,
    MultiPROgressBar &progress)
{
    AdaptiveFCResult res;

    log<LOG_INFO>(L"%1% || mode=%2%, throws=%3%, p_thresh=%4%, baseline_level=%5%, "
                  L"prepass=%6%x%7%/levels=%8%, stat_only=%9%.")
        % __func__ % (int)acfg.mode % acfg.n_throws % acfg.p_thresh % acfg.baseline_level
        % acfg.prepass_amr_initial_x % acfg.prepass_amr_initial_y % acfg.prepass_amr_levels
        % (int)acfg.stat_only_throws;

    std::unique_ptr<PROmodel> model = get_model_from_string(config, prop);

    // Resolve axis names → indices using the same lookup the surface command uses.
    const size_t xaxis_idx = resolve_axis_index(acfg.xvar, *model, systs, config, 1);
    const size_t yaxis_idx = resolve_axis_index(acfg.yvar, *model, systs, config, 0);
    log<LOG_INFO>(L"%1% || resolved xvar='%2%' -> idx=%3%; yvar='%4%' -> idx=%5%.")
        % __func__ % acfg.xvar.c_str() % (int)xaxis_idx % acfg.yvar.c_str() % (int)yaxis_idx;

    // ---- Mode: print-bank ---------------------------------------------------
    // Loads an existing bank artifact and writes a summary PDF. No fitting.
    if (acfg.mode == AdaptiveFCMode::PrintBank) {
        const std::string bank_in = acfg.output_tag + "_bank.bin";
        PEBank bank;
        if (!load_bank(bank, bank_in)) {
            log<LOG_ERROR>(L"%1% || print-bank: failed to load %2%.") % __func__ % bank_in.c_str();
            return res;
        }

        const std::string out_pdf = acfg.output_tag + "_bank_summary.pdf";
        const bool xlog_axis = (xaxis_idx < model->is_log10.size()) ? model->is_log10[xaxis_idx] : acfg.logx;
        const bool ylog_axis = (yaxis_idx < model->is_log10.size()) ? model->is_log10[yaxis_idx] : acfg.logy;
        const std::string xlabel = xaxis_idx < model->nparams
            ? model->pretty_param_names.at(xaxis_idx) : std::string("x");
        const std::string ylabel = yaxis_idx < model->nparams
            ? model->pretty_param_names.at(yaxis_idx) : std::string("y");

        plot_pebank_summary_pdf(bank, out_pdf, bank_in, xlabel, ylabel,
                                acfg.logx, acfg.logy, xlog_axis, ylog_axis);

        const std::string per_cell_pdf = acfg.output_tag + "_bank_per_cell.pdf";
        plot_pebank_pes_multipage_pdf(bank, per_cell_pdf, xlabel, ylabel,
                                      xlog_axis, ylog_axis);

        // Populate result for the caller (no PE generation in this mode).
        res.bank_path        = bank_in;
        res.n_meta_cells     = bank.n_cells;
        int64_t total_pes = 0;
        for (const auto &v : bank.cell_pes) total_pes += (int64_t)v.size();
        res.total_pes_generated = total_pes;
        res.mean_pes_per_cell   = bank.n_cells > 0 ? (float)total_pes / (float)bank.n_cells : 0.0f;
        return res;
    }

    // ---- Mode: asimov -------------------------------------------------------
    // Loads an existing bank, builds the asimov dataset from fakeDataParams
    // (noise-free expected counts under the injected truth — same convention
    // as surface/fc/profile), classifies every cell against the bank's
    // per-cell critical Δχ² at each requested CL, and writes a verdict PDF.
    // No PE generation. Bank is read-only.
    if (acfg.mode == AdaptiveFCMode::Asimov) {
        const std::string bank_in = acfg.output_tag + "_bank.bin";
        PEBank bank;
        if (!load_bank(bank, bank_in)) {
            log<LOG_ERROR>(L"%1% || asimov: failed to load %2%.") % __func__ % bank_in.c_str();
            return res;
        }

        // Use the data spectrum PROfit assembled — that's where --inject,
        // --use-fake-data, --poisson-throw, --pseudo-experiment, and real data
        // all flow into via bin/PROfit.cxx:993 (`data = variable_data[i_prime]`).
        // Classifying against that here gives the user "FC contour for whatever
        // dataset is in the chain". For null-asimov use, the user runs PROfit
        // with --use-fake-data and no --inject (so data == CV-spectrum
        // collapsed).
        const PROdata &asimov_data = data;
        log<LOG_INFO>(L"%1% || asimov: classifying %2% cells against bank %3%; data nbins=%4%.")
            % __func__ % bank.n_cells % bank_in.c_str() % (int)asimov_data.Spec().size();

        // Outer throws bar wasn't displayed in asimov mode (see PROfit.cxx),
        // so finish_all(false) — no display refresh on a never-shown bar.
        progress.finish_all(false);
        std::vector<std::pair<int, std::string>> ab_cfg;
        ab_cfg.push_back({bank.n_cells, "AFC asimov cells"});
        MultiPROgressBar asimov_progress(ab_cfg);
        asimov_progress.initialize_display();
        asimov_progress.start_display_thread();

        AsimovObs obs = compute_asimov_obs(
            config, prop, systs, *model, fitconfig, asimov_data,
            acfg.chi2, acfg.binned, xaxis_idx, yaxis_idx,
            bank.cell_center_x, bank.cell_center_y,
            proseed, nthreads, asimov_progress, 0);

        asimov_progress.finish_all(true);

        // Classify against the bank for every requested CL.
        const int min_pes_for_decision = std::max(10, acfg.n_pe_min);
        std::vector<std::vector<CellVerdict>> verdicts =
            classify_against_bank(bank, obs, acfg.cl_targets, min_pes_for_decision);

        // PDF (multipage, one page per CL).
        const std::string out_pdf = acfg.output_tag + "_asimov_verdict.pdf";
        const bool xlog_axis = (xaxis_idx < model->is_log10.size()) ? model->is_log10[xaxis_idx] : acfg.logx;
        const bool ylog_axis = (yaxis_idx < model->is_log10.size()) ? model->is_log10[yaxis_idx] : acfg.logy;
        const std::string xlabel = xaxis_idx < model->nparams
            ? model->pretty_param_names.at(xaxis_idx) : std::string("x");
        const std::string ylabel = yaxis_idx < model->nparams
            ? model->pretty_param_names.at(yaxis_idx) : std::string("y");
        plot_asimov_verdict_pdf(bank, obs, acfg.cl_targets, verdicts, out_pdf, bank_in,
                                xlabel, ylabel, acfg.logx, acfg.logy, xlog_axis, ylog_axis);

        // Clean publication-style contour overlay — the main asimov deliverable.
        // Inject-truth marker: pulled from fakeDataParams in model space, mapped
        // to physical for the canvas (same convention as the bank summary).
        const float truth_x_phys = xlog_axis
            ? std::pow(10.0f, fakeDataParams((int)xaxis_idx))
            : fakeDataParams((int)xaxis_idx);
        const float truth_y_phys = ylog_axis
            ? std::pow(10.0f, fakeDataParams((int)yaxis_idx))
            : fakeDataParams((int)yaxis_idx);
        const std::string contour_pdf = acfg.output_tag + "_asimov_contour.pdf";
        plot_asimov_contour_pdf(bank, obs, acfg.cl_targets, verdicts, contour_pdf,
                                xlabel, ylabel, acfg.logx, acfg.logy,
                                xlog_axis, ylog_axis,
                                /*draw_truth_marker=*/ true,
                                truth_x_phys, truth_y_phys);

        // ROOT artifact: contour TGraphs + per-cell TTree for downstream use.
        const std::string asimov_root = acfg.output_tag + "_asimov_contours.root";
        save_asimov_root(bank, obs, acfg.cl_targets, verdicts, asimov_root,
                         xlog_axis, ylog_axis);

        // Populate result.
        res.bank_path     = bank_in;
        res.n_meta_cells  = bank.n_cells;
        int64_t total_pes = 0;
        for (const auto &v : bank.cell_pes) total_pes += (int64_t)v.size();
        res.total_pes_generated = total_pes;
        res.mean_pes_per_cell   = bank.n_cells > 0 ? (float)total_pes / (float)bank.n_cells : 0.0f;
        // Summary log lines per CL.
        for (size_t k = 0; k < acfg.cl_targets.size(); ++k) {
            int n_in = 0, n_out = 0, n_undec = 0;
            for (const auto &v : verdicts[k]) {
                if (!v.decidable) ++n_undec;
                else if (v.included) ++n_in;
                else ++n_out;
            }
            log<LOG_INFO>(L"%1% || asimov verdict CL=%2%: inside=%3%, outside=%4%, undecidable=%5%.")
                % __func__ % acfg.cl_targets[k] % n_in % n_out % n_undec;
        }
        return res;
    }

    // ---- Mode: brazil -------------------------------------------------------
    // Additive Brazil-band construction. Each invocation:
    //   1. Loads <tag>_brazil.bin if it exists and matches the bank footprint.
    //   2. Generates --n-brazil-throws NEW throws, appending observables to the
    //      archive (chi2_osc_global + per-cell dchi2_obs per throw).
    //   3. Saves the merged archive.
    //   4. Re-classifies ALL throws (old + new) against the current bank +
    //      cl_targets to derive per-throw verdicts. The archive deliberately
    //      stores only observables, not verdicts, so growing the bank or
    //      changing CLs between brazil runs reuses existing throw data.
    //   5. Aggregates per-cell inclusion fractions across all throws and emits
    //      the brazil-band PDF + ROOT TGraphs.
    //
    // (Substage 2 — adaptive bank top-up triggered by undecidable verdicts —
    // is a separate slice on top of this.)
    if (acfg.mode == AdaptiveFCMode::Brazil) {
        const std::string bank_in = acfg.output_tag + "_bank.bin";
        const std::string brazil_bin = acfg.output_tag + "_brazil.bin";
        PEBank bank;
        if (!load_bank(bank, bank_in)) {
            log<LOG_ERROR>(L"%1% || brazil: failed to load %2%.") % __func__ % bank_in.c_str();
            return res;
        }

        const int n_new    = std::max(1, acfg.n_brazil_throws);
        const int n_cells  = bank.n_cells;
        const int n_cl     = (int)acfg.cl_targets.size();
        const int min_pes  = std::max(10, acfg.n_pe_min);

        // (1) Load existing archive if compatible. Otherwise initialise fresh.
        BrazilArchive arc;
        arc.finest_nx = bank.finest_nx;
        arc.finest_ny = bank.finest_ny;
        arc.n_cells   = n_cells;
        {
            std::ifstream test(brazil_bin, std::ios::binary);
            if (test.is_open()) {
                test.close();
                BrazilArchive existing;
                if (load_brazil_archive(existing, brazil_bin)
                    && existing.n_cells   == n_cells
                    && existing.finest_nx == bank.finest_nx
                    && existing.finest_ny == bank.finest_ny) {
                    arc.per_throw_global_chi2 = std::move(existing.per_throw_global_chi2);
                    arc.per_throw_dchi2       = std::move(existing.per_throw_dchi2);
                    log<LOG_INFO>(L"%1% || brazil: loaded existing archive %2% (%3% prior throws).")
                        % __func__ % brazil_bin.c_str() % (int)arc.per_throw_dchi2.size();
                } else {
                    log<LOG_WARNING>(L"%1% || brazil: existing archive %2% has mismatched footprint; ignoring and starting fresh.")
                        % __func__ % brazil_bin.c_str();
                }
            }
        }
        const int n_existing = (int)arc.per_throw_dchi2.size();
        const int n_total    = n_existing + n_new;

        log<LOG_INFO>(L"%1% || brazil: bank=%2% (%3% cells, %4% CLs, min_pes=%5%); existing=%6% throws, new=%7%, total after=%8%.")
            % __func__ % bank_in.c_str() % n_cells % n_cl % min_pes
            % n_existing % n_new % n_total;

        // Reserve room for new throws so we can index by absolute throw_idx.
        arc.per_throw_global_chi2.resize((size_t)n_total, 0.0f);
        arc.per_throw_dchi2.resize((size_t)n_total);
        for (int t = n_existing; t < n_total; ++t) {
            arc.per_throw_dchi2[(size_t)t].assign((size_t)n_cells, 0.0f);
        }

        // (2) Run new throws. Outer bar wasn't displayed for brazil mode
        // (see PROfit.cxx), so finish_all(false) — no spurious refresh.
        progress.finish_all(false);
        std::vector<std::pair<int, std::string>> bar_cfg;
        bar_cfg.push_back({n_new, "Brazil throws (new)"});
        MultiPROgressBar brazil_progress(bar_cfg);
        brazil_progress.initialize_display();
        brazil_progress.start_display_thread();

        std::vector<std::pair<int, std::string>> silent_cfg;
        silent_cfg.push_back({n_cells, "_silent"});
        MultiPROgressBar silent_progress(silent_cfg);

        // Throw-invariant: CV spectrum + covariance decomposition (an SVD)
        // hoisted out of the per-throw loop.
        PROspec brazil_cv = FillSpectra(config, prop, systs, *model, fakeDataParams,
                                        acfg.binned, config.i_prime);
        Eigen::MatrixXf brazil_L = systs.DecomposeFractionalCovariance(config, brazil_cv.Spec());

        for (int t_new = 0; t_new < n_new; ++t_new) {
            const int t_abs = n_existing + t_new;
            PROdata throw_data = generate_pseudo_experiment_data(
                config, prop, systs, *model, fakeDataParams, acfg.binned, brazil_L, proseed);

            AsimovObs obs = compute_asimov_obs(
                config, prop, systs, *model, fitconfig, throw_data,
                acfg.chi2, acfg.binned, xaxis_idx, yaxis_idx,
                bank.cell_center_x, bank.cell_center_y,
                proseed, nthreads, silent_progress, 0);

            arc.per_throw_global_chi2[(size_t)t_abs] = obs.chi2_osc_global;
            for (int c = 0; c < n_cells; ++c) {
                arc.per_throw_dchi2[(size_t)t_abs][(size_t)c] = obs.dchi2_obs[(size_t)c];
            }

            log<LOG_INFO>(L"%1% || brazil throw %2%/%3% (abs=%4%): chi2_osc_global=%5%.")
                % __func__ % (t_new + 1) % n_new % t_abs % obs.chi2_osc_global;
            brazil_progress.increment_bar(0);
        }
        brazil_progress.finish_all(true);

        // (3) Persist the merged archive.
        save_brazil_archive(arc, brazil_bin);

        // (4) Re-classify ALL throws (existing + new) against the current bank.
        //     Verdicts are derived afresh — archive stored only observables.
        //     Track BOTH `included` and `decidable` per throw so the aggregation
        //     can distinguish "throw said outside" from "bank too sparse to say":
        //     treating undecidable as "outside" creates a phantom Brazil-band
        //     ring at the AMR refined↔baseline boundary (deep-basin cells get
        //     painted as excluded when they're really just unsampled).
        std::vector<std::vector<std::vector<uint8_t>>> per_throw_verdicts(
            (size_t)n_total,
            std::vector<std::vector<uint8_t>>(
                (size_t)n_cl, std::vector<uint8_t>((size_t)n_cells, 0)));
        std::vector<std::vector<std::vector<uint8_t>>> per_throw_decidable(
            (size_t)n_total,
            std::vector<std::vector<uint8_t>>(
                (size_t)n_cl, std::vector<uint8_t>((size_t)n_cells, 0)));
        for (int t = 0; t < n_total; ++t) {
            AsimovObs obs_t;
            obs_t.chi2_osc_global = arc.per_throw_global_chi2[(size_t)t];
            obs_t.dchi2_obs       = arc.per_throw_dchi2[(size_t)t];
            obs_t.chi2_syst.assign((size_t)n_cells, 0.0f); // not stored / not needed for classify
            auto verdicts = classify_against_bank(bank, obs_t, acfg.cl_targets, min_pes);
            for (int k = 0; k < n_cl; ++k) {
                for (int c = 0; c < n_cells; ++c) {
                    const auto &v = verdicts[(size_t)k][(size_t)c];
                    per_throw_decidable[(size_t)t][(size_t)k][(size_t)c] = v.decidable ? 1 : 0;
                    per_throw_verdicts [(size_t)t][(size_t)k][(size_t)c] =
                        (v.decidable && v.included) ? 1 : 0;
                }
            }
        }

        // (5) Aggregate inclusion fractions across all throws.
        //     Sentinel: inclusion_frac < 0 marks a cell where no throw was ever
        //     decidable. build_inclusion_th2d skips such cells from the IDW
        //     interpolation, so the surface is filled by neighbouring decided
        //     cells — no phantom contour ringing the AMR boundary.
        constexpr float kUndecidedSentinel = -1.0f;
        std::vector<std::vector<float>> inclusion_frac(
            (size_t)n_cl, std::vector<float>((size_t)n_cells, kUndecidedSentinel));
        int total_undecidable_cells = 0;
        for (int k = 0; k < n_cl; ++k) {
            int undecidable_this_cl = 0;
            for (int c = 0; c < n_cells; ++c) {
                int n_in = 0, n_decided = 0;
                for (int t = 0; t < n_total; ++t) {
                    if (per_throw_decidable[(size_t)t][(size_t)k][(size_t)c]) {
                        ++n_decided;
                        if (per_throw_verdicts[(size_t)t][(size_t)k][(size_t)c]) ++n_in;
                    }
                }
                if (n_decided > 0) {
                    inclusion_frac[(size_t)k][(size_t)c] = (float)n_in / (float)n_decided;
                } else {
                    inclusion_frac[(size_t)k][(size_t)c] = kUndecidedSentinel;
                    ++undecidable_this_cl;
                }
            }
            log<LOG_INFO>(L"%1% || brazil aggregation CL=%2%: %3% / %4% cells undecidable (bank too sparse).")
                % __func__ % acfg.cl_targets[(size_t)k] % undecidable_this_cl % n_cells;
            total_undecidable_cells += undecidable_this_cl;
        }
        if (total_undecidable_cells > 0) {
            log<LOG_INFO>(L"%1% || brazil: %2% undecidable (cell, CL) entries skipped from IDW interpolation; "
                          L"deep-basin / sparse-bank regions are filled by neighbouring decided cells.")
                % __func__ % total_undecidable_cells;
        }

        // Outputs.
        const std::string brazil_pdf  = acfg.output_tag + "_brazil_band.pdf";
        const std::string brazil_root = acfg.output_tag + "_brazil.root";
        const bool xlog_axis = (xaxis_idx < model->is_log10.size()) ? model->is_log10[xaxis_idx] : acfg.logx;
        const bool ylog_axis = (yaxis_idx < model->is_log10.size()) ? model->is_log10[yaxis_idx] : acfg.logy;
        const std::string xlabel = xaxis_idx < model->nparams
            ? model->pretty_param_names.at(xaxis_idx) : std::string("x");
        const std::string ylabel = yaxis_idx < model->nparams
            ? model->pretty_param_names.at(yaxis_idx) : std::string("y");
        const float truth_x_phys = xlog_axis
            ? std::pow(10.0f, fakeDataParams((int)xaxis_idx))
            : fakeDataParams((int)xaxis_idx);
        const float truth_y_phys = ylog_axis
            ? std::pow(10.0f, fakeDataParams((int)yaxis_idx))
            : fakeDataParams((int)yaxis_idx);

        plot_brazil_band_pdf(bank, inclusion_frac, acfg.cl_targets, brazil_pdf, bank_in,
                              xlabel, ylabel, acfg.logx, acfg.logy,
                              xlog_axis, ylog_axis,
                              /*draw_truth_marker=*/ true,
                              truth_x_phys, truth_y_phys, n_total);

        save_brazil_root(bank, per_throw_verdicts, arc.per_throw_dchi2,
                          arc.per_throw_global_chi2, inclusion_frac, acfg.cl_targets,
                          brazil_root, xlog_axis, ylog_axis);

        // Populate result.
        res.bank_path    = bank_in;
        res.n_meta_cells = n_cells;
        int64_t total_pes = 0;
        for (const auto &v : bank.cell_pes) total_pes += (int64_t)v.size();
        res.total_pes_generated = total_pes;
        res.mean_pes_per_cell   = n_cells > 0 ? (float)total_pes / (float)n_cells : 0.0f;

        log<LOG_INFO>(L"%1% || brazil done: %2% new + %3% existing = %4% total throws across %5% cells; outputs %6%, %7%, %8%.")
            % __func__ % n_new % n_existing % n_total % n_cells
            % brazil_bin.c_str() % brazil_pdf.c_str() % brazil_root.c_str();
        return res;
    }

    const std::string mesh_path = acfg.output_tag + "_mesh.bin";
    const std::string bank_path = acfg.output_tag + "_bank.bin";

    // ---- Mode: build-mesh ---------------------------------------------------
    // Wilks prepass + meta-mesh + diagnostics. Saves <tag>_mesh.bin and the
    // slice-1 PDFs. Does NOT generate PEs. Always rebuilds — running this mode
    // is the explicit request to (re)compute the mesh.
    if (acfg.mode == AdaptiveFCMode::BuildMesh) {
        std::vector<PROmesh::AMRResult> per_throw_meshes =
            generate_throws(config, prop, systs, *model, fitconfig, proseed,
                            fakeDataParams, acfg, nthreads, xaxis_idx, yaxis_idx, progress);
        res.n_throws_done = (int)per_throw_meshes.size();
        res.leaves_per_throw.reserve(per_throw_meshes.size());
        for (const auto &amr : per_throw_meshes) res.leaves_per_throw.push_back((int)amr.leaves.size());

        MetaMesh mm = build_meta_mesh(per_throw_meshes, acfg.p_thresh, acfg.baseline_level);
        res.n_meta_cells     = (int)mm.cells.size();
        res.n_baseline_cells = mm.n_baseline_cells;
        res.n_refined_cells  = mm.n_refined_cells;

        write_slice1_diagnostics(per_throw_meshes, mm, *model, systs, acfg,
                                 xaxis_idx, yaxis_idx, res);
        save_mesh(mm, mesh_path);

        progress.finish_all(true);
        log<LOG_INFO>(L"%1% || build-mesh done: cells=%2% (baseline=%3%, refined=%4%), saved %5%. "
                      L"Next step: --mode init-bank.")
            % __func__ % (int)mm.cells.size() % mm.n_baseline_cells % mm.n_refined_cells
            % mesh_path.c_str();
        return res;
    }

    // ---- Mode: init-bank ----------------------------------------------------
    // Loads <tag>_mesh.bin and generates the PE bank. Errors out if the mesh
    // doesn't exist — the user is expected to run --mode build-mesh first.
    if (acfg.mode == AdaptiveFCMode::InitBank) {
        MetaMesh mm;
        if (!load_mesh(mm, mesh_path)) {
            log<LOG_ERROR>(L"%1% || init-bank: required mesh artifact %2% not found. "
                           L"Run --mode build-mesh first.") % __func__ % mesh_path.c_str();
            progress.finish_all(false);
            return res;
        }
        log<LOG_INFO>(L"%1% || init-bank: loaded meta-mesh from %2% (cells=%3%).")
            % __func__ % mesh_path.c_str() % (int)mm.cells.size();
        res.n_meta_cells     = (int)mm.cells.size();
        res.n_baseline_cells = mm.n_baseline_cells;
        res.n_refined_cells  = mm.n_refined_cells;

        if (mm.cells.empty()) {
            log<LOG_WARNING>(L"%1% || empty meta-mesh; nothing to bank.") % __func__;
            progress.finish_all(false);
            return res;
        }

        const bool xlog = (xaxis_idx < model->is_log10.size()) ? model->is_log10[xaxis_idx] : acfg.logx;
        const bool ylog = (yaxis_idx < model->is_log10.size()) ? model->is_log10[yaxis_idx] : acfg.logy;

        std::vector<float> cell_x_model, cell_y_model;
        compute_cell_centers(mm, xlog, ylog, cell_x_model, cell_y_model);

        // Cholesky factor of the total covariance — built once, reused across all
        // cells and all PEs. Same construction as the brazil-band path
        // (bin/PROfit.cxx:1586) and the existing fc block (bin/PROfit.cxx:2511).
        PROspec cv = FillSpectra(config, prop, systs, *model, fakeDataParams,
                                 acfg.binned, config.i_prime);
        Eigen::MatrixXf L = systs.DecomposeFractionalCovariance(config, cv.Spec());

        log<LOG_INFO>(L"%1% || init-bank: starting PE generation for %2% cells "
                      L"(n_pe_min=%3%, n_pe_max=%4%, wilson_eps=%5%).")
            % __func__ % (int)mm.cells.size() % acfg.n_pe_min % acfg.n_pe_max % acfg.wilson_eps;

        PEBank bank;
        bank.finest_nx = mm.finest_nx;
        bank.finest_ny = mm.finest_ny;
        bank.max_levels = mm.max_levels;
        bank.x_lo = mm.x_lo; bank.x_hi = mm.x_hi;
        bank.y_lo = mm.y_lo; bank.y_hi = mm.y_hi;
        bank.n_cells = (int)mm.cells.size();
        bank.cell_center_x = cell_x_model;
        bank.cell_center_y = cell_y_model;
        bank.cell_i_bl.reserve(mm.cells.size());
        bank.cell_j_bl.reserve(mm.cells.size());
        bank.cell_step.reserve(mm.cells.size());
        bank.cell_level.reserve(mm.cells.size());
        for (const auto &c : mm.cells) {
            bank.cell_i_bl.push_back(c.i_bl);
            bank.cell_j_bl.push_back(c.j_bl);
            bank.cell_step.push_back(c.step);
            bank.cell_level.push_back(c.level);
        }

        // Top-up: try to carry forward an existing <tag>_bank.bin if its mesh
        // footprint matches. Each cell's PE list is preserved; schedule_pes
        // only generates additional PEs for cells not yet at the criterion.
        {
            std::ifstream test(bank_path, std::ios::binary);
            if (test.is_open()) {
                test.close();
                PEBank existing;
                if (load_bank(existing, bank_path)
                    && existing.n_cells   == bank.n_cells
                    && existing.finest_nx == bank.finest_nx
                    && existing.finest_ny == bank.finest_ny
                    && existing.max_levels == bank.max_levels) {
                    bank.cell_pes = std::move(existing.cell_pes);
                    int64_t existing_total = 0;
                    for (const auto &v : bank.cell_pes) existing_total += (int64_t)v.size();
                    log<LOG_INFO>(L"%1% || init-bank top-up: loaded existing bank %2% (%3% PEs / %4% cells). Generating additional PEs to meet criterion.")
                        % __func__ % bank_path.c_str() % (long long)existing_total % bank.n_cells;
                } else {
                    log<LOG_WARNING>(L"%1% || init-bank: existing bank at %2% has mismatched footprint; ignoring and regenerating.")
                        % __func__ % bank_path.c_str();
                }
            }
        }

        // Stop the throws progress bar (never displayed in init-bank). Launch a
        // PE-counted bar: bar size = total PEs to be added across all cells.
        // Updating per-PE (rather than per-cell) makes the bar smooth on
        // large meshes where each cell takes minutes.
        progress.finish_all(false);
        int total_pes_to_add = 0;
        for (int c = 0; c < bank.n_cells; ++c) {
            const int level = bank.cell_level[(size_t)c];
            // Mirror compute_to_add's level eligibility: only_layer >= 0 wins.
            if (acfg.only_layer >= 0) {
                if (level != acfg.only_layer) continue;
            } else {
                if (level < acfg.update_layer) continue;
            }
            const int n_existing = (c < (int)bank.cell_pes.size())
                ? (int)bank.cell_pes[(size_t)c].size() : 0;
            if (n_existing >= acfg.n_pe_max) continue;
            int to_add_raw;
            if (acfg.only_layer >= 0) {
                to_add_raw = acfg.n_pe_min;  // no doubling in only-layer mode
            } else {
                const int delta = std::max(0, level - acfg.update_layer);
                to_add_raw = (delta > 20)
                    ? acfg.n_pe_max
                    : (int)((int64_t)acfg.n_pe_min << delta);
            }
            total_pes_to_add += std::min(to_add_raw, acfg.n_pe_max - n_existing);
        }
        log<LOG_INFO>(L"%1% || init-bank: will add %2% PEs total across %3% cells.")
            % __func__ % total_pes_to_add % bank.n_cells;
        std::vector<std::pair<int, std::string>> cells_bar_cfg;
        cells_bar_cfg.push_back({std::max(1, total_pes_to_add), "AFC PEs added"});
        MultiPROgressBar cells_progress(cells_bar_cfg);
        cells_progress.initialize_display();
        cells_progress.start_display_thread();

        schedule_pes(acfg, config, prop, systs, *model, fitconfig, proseed, L,
                     xaxis_idx, yaxis_idx, cell_x_model, cell_y_model,
                     bank, nthreads, cells_progress, 0, res);

        cells_progress.finish_all(true);

        if (save_bank(bank, bank_path)) {
            res.bank_path = bank_path;
        }

        // Auto-print bank summary + per-cell PDFs.
        {
            const std::string summary_pdf = acfg.output_tag + "_bank_summary.pdf";
            const bool xlog_axis = xlog;
            const bool ylog_axis = ylog;
            const std::string xlabel = xaxis_idx < model->nparams
                ? model->pretty_param_names.at(xaxis_idx) : std::string("x");
            const std::string ylabel = yaxis_idx < model->nparams
                ? model->pretty_param_names.at(yaxis_idx) : std::string("y");
            plot_pebank_summary_pdf(bank, summary_pdf, bank_path, xlabel, ylabel,
                                    acfg.logx, acfg.logy, xlog_axis, ylog_axis);

            const std::string per_cell_pdf = acfg.output_tag + "_bank_per_cell.pdf";
            plot_pebank_pes_multipage_pdf(bank, per_cell_pdf, xlabel, ylabel,
                                          xlog_axis, ylog_axis);
        }

        return res;
    }

    // Classify / unknown — not yet implemented.
    log<LOG_ERROR>(L"%1% || mode '%2%' is not yet implemented.")
        % __func__ % (int)acfg.mode;
    progress.finish_all(false);
    return res;
}

} // namespace PROfit
