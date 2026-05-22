/**
 * @file PROAdaptiveFC.cxx
 * @brief Adaptive Feldman-Cousins pipeline implementation. Slice 1 only.
 *
 * Slice 1 covers: Wilks pre-pass over N independent throws + meta-mesh
 * aggregation + diagnostic ROOT artifact. The PE bank, sequential stopping
 * rule, scheduler, and data-classification step are deferred to slice 2.
 *
 * This file deliberately *duplicates* code from src/PROfc.cxx and from
 * src/PROsurf.cxx::FillSurfaceAMR rather than refactoring shared helpers out
 * of either. The per-PE worker body and the AMR EvalFn lambda are both copied
 * with banner comments marking the duplication.
 */
#include "PROAdaptiveFC.h"

#include "PROlog.h"
#include "PROmodel.h"
#include "PROchi.h"
#include "PROCNP.h"
#include "PROpoisson.h"
#include "PROmetric.h"
#include "PROspec.h"
#include "PROcess.h"
#include "PROtocall.h"
#include "PROserial.h"

#include <Eigen/Eigen>

#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <fstream>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>

#include "TBox.h"
#include "TCanvas.h"
#include "TDirectory.h"
#include "TFile.h"
#include "TGraph.h"
#include "TH1F.h"
#include "TH2D.h"
#include "TLegend.h"
#include "TLine.h"
#include "TList.h"
#include "TMarker.h"
#include "TObjArray.h"
#include "TPad.h"
#include "TPaveText.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TTree.h"

namespace PROfit {

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
    if (auto loc = std::find(systs.spline_names.begin(), systs.spline_names.end(), name);
        loc != systs.spline_names.end()) {
        return std::distance(systs.spline_names.begin(), loc);
    }
    for (const auto &[xml_name, plot_name] : config.m_mcgen_variation_plotname_map) {
        if (name == plot_name) {
            if (auto loc = std::find(systs.spline_names.begin(), systs.spline_names.end(), xml_name);
                loc != systs.spline_names.end()) {
                return std::distance(systs.spline_names.begin(), loc);
            }
        }
    }
    log<LOG_WARNING>(L"%1% || resolve_axis_index: axis variable '%2%' not found; falling back to %3%.")
        % __func__ % name.c_str() % fallback;
    return fallback;
}

// ====================================================================
//  Section 1 — run_wilks_prepass
//
//  Copied/adapted from PROsurf::FillSurfaceAMR EvalFn (src/PROsurf.cxx:758-795).
//  Keep parallel until the adaptive pipeline is validated, then refactor.
//
//  The only structural difference from PROsurf's version: this one closes over
//  a *PROdata* (the per-throw fake dataset) and builds a fresh PROmetric per
//  thread from it, since PROsurf takes the metric pre-built from outside.
// ====================================================================

static PROmesh::AMRResult run_wilks_prepass(
    const PROconfig &config,
    const PROpeller &prop,
    const PROsyst   &systs,
    const PROmodel  &model,
    const PROdata   &data,
    const PROfitterConfig &fitconfig,
    const std::string &chi2_kind,
    bool eventbyevent,
    size_t xaxis_idx,
    size_t yaxis_idx,
    float x_lo, float x_hi,
    float y_lo, float y_hi,
    const PROmesh::AMROptions &opts_in,
    int nthreads)
{
    PROmesh::AMROptions opts = opts_in;
    opts.nthreads = nthreads;

    // Thread-local metric clones. The first call on each thread allocates;
    // subsequent calls reuse. Mirrors PROsurf.cxx:761-764.
    auto eval_fn = [&config, &prop, &systs, &model, &data, &fitconfig,
                    chi2_kind, eventbyevent, xaxis_idx, yaxis_idx]
        (const PROmesh::EvalRequest &req) -> PROmesh::EvalResult
    {
        thread_local std::unique_ptr<PROmetric> tls_metric;
        if (!tls_metric) {
            // Construct a per-thread metric matching the chi2_kind choice.
            // The lifetime of `data`, `systs`, etc. is the lifetime of
            // run_wilks_prepass — fine because the AMR scheduler joins all
            // workers before run_amr returns.
            PROmetric::EvalStrategy strat = eventbyevent
                ? PROmetric::EventByEvent : PROmetric::BinnedChi2;
            if (chi2_kind == "PROchi") {
                tls_metric.reset(new PROchi("", config, prop, &systs, model, data, strat));
            } else if (chi2_kind == "PROCNP") {
                tls_metric.reset(new PROCNP("", config, prop, &systs, model, data, strat));
            } else if (chi2_kind == "Poisson") {
                tls_metric.reset(new PROpoisson("", config, prop, &systs, model, data, strat));
            } else {
                log<LOG_ERROR>(L"%1% || run_wilks_prepass: unknown chi2 kind '%2%'.")
                    % __func__ % chi2_kind.c_str();
                abort();
            }
        }
        PROmetric *m = tls_metric.get();
        m->reset();

        const int nphys   = (int)m->GetModel().nparams;
        const int nspline = (int)m->GetSysts().GetNSplines();
        const int n_full  = nphys + nspline;
        Eigen::VectorXf lb(n_full), ub(n_full);
        lb << m->GetModel().lb,
              Eigen::VectorXf::Map(m->GetSysts().spline_lo.data(), m->GetSysts().spline_lo.size());
        ub << m->GetModel().ub,
              Eigen::VectorXf::Map(m->GetSysts().spline_hi.data(), m->GetSysts().spline_hi.size());

        // Pin the two scanned coords, optimise the rest.
        lb((int)xaxis_idx) = req.x_phys;
        ub((int)xaxis_idx) = req.x_phys;
        lb((int)yaxis_idx) = req.y_phys;
        ub((int)yaxis_idx) = req.y_phys;
        m->setBounds(lb, ub);

        const uint32_t fseed = static_cast<uint32_t>(req.key & 0xffffffffu);
        PROfitter fitter(ub, lb, fitconfig, fseed);

        PROmesh::EvalResult out;
        if (req.seeds.empty()) {
            out.chi2 = fitter.Fit(*m);
        } else {
            out.chi2 = fitter.Fit(*m, req.seeds);
        }
        out.best_fit = fitter.best_fit;
        return out;
    };

    log<LOG_INFO>(L"%1% || run_wilks_prepass starting AMR on [%2%, %3%] x [%4%, %5%], "
                  L"initial=%6%x%7%, levels=%8%, nthreads=%9%.")
        % __func__ % x_lo % x_hi % y_lo % y_hi
        % opts.initial_nx % opts.initial_ny % opts.max_levels % opts.nthreads;

    PROmesh::AMRResult amr = PROmesh::run_amr(eval_fn, x_lo, x_hi, y_lo, y_hi, opts);

    log<LOG_INFO>(L"%1% || AMR done: %2% fits across %3% levels, min_chi2=%4%, %5% leaves, %6% contour levels.")
        % __func__ % amr.total_fits % opts.max_levels % amr.min_chi2
        % (int)amr.leaves.size() % (int)amr.polylines.size();

    return amr;
}

// ====================================================================
//  Section 2 — generate_throws
//
//  Mirrors the brazil-band throw loop at bin/PROfit.cxx:1582-1620 but writes
//  each throw's result into our own vector of AMRResults instead of into
//  PROsurf::surface. Honours `stat_only_throws`.
// ====================================================================

static std::vector<PROmesh::AMRResult> generate_throws(
    const PROconfig &config,
    const PROpeller &prop,
    const PROsyst   &systs,
    const PROmodel  &model,
    const PROfitterConfig &fitconfig,
    PROseed &proseed,
    const Eigen::VectorXf &fakeDataParams,
    const AdaptiveFCConfig &acfg,
    int nthreads,
    size_t xaxis_idx, size_t yaxis_idx,
    MultiPROgressBar &progress)
{
    const size_t N_phys_params = model.nparams;
    const size_t nbins_collapsed = (size_t)config.m_num_variable_bins_total_collapsed[config.i_prime];

    // Build the CV spectrum and the Cholesky factor once — both are throw-independent.
    PROspec cv = FillSpectra(config, prop, systs, model, fakeDataParams, !acfg.binned ? false : true, config.i_prime);
    PROspec collapsed_cv = PROspec(CollapseMatrix(config, cv.Spec()),
                                   CollapseMatrix(config, cv.Error()));
    Eigen::MatrixXf L = systs.DecomposeFractionalCovariance(config, cv.Spec());

    // Convert acfg axis bounds back to *transformed* (log/lin) space — that's
    // what PROmesh expects and what PROsurf::FillSurfaceAMR uses (see line
    // 747-750 of PROsurf.cxx).
    const float x_lo_t = acfg.logx ? std::log10(acfg.x_lo) : acfg.x_lo;
    const float x_hi_t = acfg.logx ? std::log10(acfg.x_hi) : acfg.x_hi;
    const float y_lo_t = acfg.logy ? std::log10(acfg.y_lo) : acfg.y_lo;
    const float y_hi_t = acfg.logy ? std::log10(acfg.y_hi) : acfg.y_hi;

    PROmesh::AMROptions opts;
    opts.initial_nx     = acfg.prepass_amr_initial_x;
    opts.initial_ny     = acfg.prepass_amr_initial_y;
    opts.max_levels     = acfg.prepass_amr_levels;
    opts.delta_widen    = acfg.prepass_delta_widen;
    opts.contour_levels = acfg.prepass_contour_levels;
    opts.produce_dense  = true;

    std::vector<PROmesh::AMRResult> results;
    results.reserve(acfg.n_throws);

    // RNG plumbing — mirrors fc_worker's pattern (PROfc.cxx:36-37).
    std::uniform_int_distribution<uint32_t> dseed(0, std::numeric_limits<uint32_t>::max());

    std::normal_distribution<float> d;

    for (int t = 0; t < acfg.n_throws; ++t) {
        // Build this throw's parameter vector and stat-throw vector.
        Eigen::VectorXf throwp = fakeDataParams;
        Eigen::VectorXf throwC = Eigen::VectorXf::Constant((int)nbins_collapsed, 0);

        if (!acfg.stat_only_throws) {
            for (size_t i = 0; i < systs.GetNSplines(); ++i) {
                throwp((int)(i + N_phys_params)) = d(PROseed::global_rng);
            }
        }
        for (size_t i = 0; i < nbins_collapsed; ++i) {
            throwC((int)i) = d(PROseed::global_rng);
        }

        const bool binned_flag = acfg.binned;
        PROspec shifted = FillSpectra(config, prop, systs, model, throwp, binned_flag, config.i_prime);

        PROspec newSpec = acfg.stat_only_throws
            ? PROspec::PoissonVariation(collapsed_cv, dseed(proseed.global_rng))
            : PROspec::PoissonVariation(
                  PROspec(CollapseMatrix(config, shifted.Spec()) + L * throwC,
                          CollapseMatrix(config, shifted.Error())),
                  dseed(proseed.global_rng));
        PROdata data(newSpec.Spec(), newSpec.Error());

        log<LOG_INFO>(L"%1% || throw %2%/%3%: stat_only=%4%, n_splines_thrown=%5%, n_stat_bins=%6%.")
            % __func__ % t % acfg.n_throws % (int)acfg.stat_only_throws
            % (int)(acfg.stat_only_throws ? 0 : systs.GetNSplines()) % (int)nbins_collapsed;

        PROmesh::AMRResult amr = run_wilks_prepass(
            config, prop, systs, model, data, fitconfig,
            acfg.chi2, !acfg.binned ? true : false,
            xaxis_idx, yaxis_idx,
            x_lo_t, x_hi_t, y_lo_t, y_hi_t,
            opts, nthreads);

        results.push_back(std::move(amr));
        progress.increment_bar(0);
    }

    return results;
}

// ====================================================================
//  Section 3 — build_meta_mesh
//
//  Aggregates per-throw AMRResults into a MetaMesh by counting, at each
//  finest-grid coordinate, how many throws produced a leaf containing that
//  coordinate at refinement depth ≥ L for each L.
//
//  A cell at level L is "kept" in the meta-mesh if either
//    (a) L < baseline_level (always kept — uniform baseline), OR
//    (b) refine_count[L] / n_throws ≥ p_thresh.
// ====================================================================

static MetaMesh build_meta_mesh(const std::vector<PROmesh::AMRResult> &throws,
                                float p_thresh,
                                int baseline_level)
{
    MetaMesh mm;
    if (throws.empty()) {
        log<LOG_WARNING>(L"%1% || build_meta_mesh: empty throw list.") % __func__;
        return mm;
    }

    // Sanity: all throws must share the same finest grid + AMR depth + bounds.
    mm.finest_nx  = throws.front().finest_nx;
    mm.finest_ny  = throws.front().finest_ny;
    mm.x_lo = throws.front().x_lo;  mm.x_hi = throws.front().x_hi;
    mm.y_lo = throws.front().y_lo;  mm.y_hi = throws.front().y_hi;
    int max_level_seen = 0;
    for (const auto &amr : throws) {
        if (amr.finest_nx != mm.finest_nx || amr.finest_ny != mm.finest_ny) {
            log<LOG_ERROR>(L"%1% || build_meta_mesh: throws disagree on finest grid (%2%x%3% vs %4%x%5%).")
                % __func__ % amr.finest_nx % amr.finest_ny % mm.finest_nx % mm.finest_ny;
            return mm;
        }
        for (const auto &leaf : amr.leaves) max_level_seen = std::max(max_level_seen, leaf.level);
    }
    mm.max_levels = max_level_seen;
    const int n_levels = max_level_seen + 1;
    const int n_throws = (int)throws.size();

    // For each finest-grid point (i, j), count refinement-depth tallies across throws.
    // Storage: dense grid (finest_nx * finest_ny) of vector<int> with n_levels slots.
    // For SBN-class grids (typical finest=80x80 with max_levels=3, n_levels=4) this is
    // 80*80*4 = 25600 ints = ~100 KB. Acceptable.
    const size_t W = (size_t)mm.finest_nx;
    const size_t H = (size_t)mm.finest_ny;
    std::vector<std::vector<int>> tally(W * H, std::vector<int>(n_levels, 0));

    // Each leaf in a throw covers a square region in finest-integer coords from
    // (i_bl, j_bl) to (i_bl + step, j_bl + step). For depth tallying, we credit
    // every finest-grid point inside that square with "refined to depth = leaf.level".
    // A cell at depth L was produced because the throw's AMR refined depth (L-1),
    // (L-2), ..., 0 to reach this point — so we credit all levels ≤ leaf.level.
    for (const auto &amr : throws) {
        for (const auto &leaf : amr.leaves) {
            const int i0 = std::max(0, leaf.i_bl);
            const int j0 = std::max(0, leaf.j_bl);
            const int i1 = std::min((int)W, leaf.i_bl + leaf.step);
            const int j1 = std::min((int)H, leaf.j_bl + leaf.step);
            for (int i = i0; i < i1; ++i) {
                for (int j = j0; j < j1; ++j) {
                    auto &v = tally[(size_t)i * H + (size_t)j];
                    for (int L = 0; L <= leaf.level && L < n_levels; ++L) v[L] += 1;
                }
            }
        }
    }

    // Build MetaCells. A cell at level L is described by its bottom-left
    // finest-integer coord and its step = 2^(max_levels - L). We sweep the
    // finest grid by step at level L and decide inclusion based on the
    // tally count for the top-left (i_bl, j_bl) of each candidate cell.
    // For mixed-level meta-meshes, the rule is: a cell at level L exists iff
    //   (a) L < baseline_level, OR
    //   (b) tally[L][center] / n_throws ≥ p_thresh, AND no cell at deeper
    //       level covering the same area also passes (b).
    //
    // For slice 1 we keep the policy simple: emit cells at the *finest* level
    // that passes the threshold at each finest-grid coordinate. If no level
    // passes and L < baseline_level coverage applies, fall back to baseline.
    //
    // We materialise this by sweeping the finest grid and, for each finest
    // coordinate, finding the deepest level L* where either condition holds.
    // Then we deduplicate by the (i_bl, j_bl, step) coordinate of the cell
    // that contains that finest coordinate at level L*.
    const int threshold_count = std::max(1, (int)std::ceil(p_thresh * (float)n_throws));

    std::vector<MetaCell> emitted;
    emitted.reserve(W * H / 4);
    std::vector<uint8_t> seen_finest(W * H, 0); // marks finest points already covered by an emitted deeper cell

    // First pass: emit deep (refined) cells where p_thresh is met.
    // Walk from deepest level outward.
    for (int level = mm.max_levels; level >= baseline_level; --level) {
        const int step = 1 << std::max(0, mm.max_levels - level);
        for (int i = 0; i < (int)W; i += step) {
            for (int j = 0; j < (int)H; j += step) {
                if (seen_finest[(size_t)i * H + (size_t)j]) continue;
                const int count = tally[(size_t)i * H + (size_t)j][std::min(level, n_levels - 1)];
                if (count >= threshold_count) {
                    MetaCell c;
                    c.i_bl = i; c.j_bl = j; c.step = step; c.level = level;
                    c.per_level_refine_count.assign(n_levels, 0);
                    // Aggregate per-level counts across the cell footprint (max).
                    for (int L = 0; L < n_levels; ++L) {
                        int peak = 0;
                        for (int ii = i; ii < i + step && ii < (int)W; ++ii) {
                            for (int jj = j; jj < j + step && jj < (int)H; ++jj) {
                                peak = std::max(peak, tally[(size_t)ii * H + (size_t)jj][L]);
                            }
                        }
                        c.per_level_refine_count[L] = peak;
                    }
                    emitted.push_back(std::move(c));
                    mm.n_refined_cells++;
                    // Mark this whole footprint as already covered.
                    for (int ii = i; ii < i + step && ii < (int)W; ++ii) {
                        for (int jj = j; jj < j + step && jj < (int)H; ++jj) {
                            seen_finest[(size_t)ii * H + (size_t)jj] = 1;
                        }
                    }
                }
            }
        }
    }

    // Second pass: fill remaining area with baseline-level cells.
    {
        const int level = std::max(0, baseline_level - 1);
        const int step = 1 << std::max(0, mm.max_levels - level);
        for (int i = 0; i < (int)W; i += step) {
            for (int j = 0; j < (int)H; j += step) {
                bool any_covered = false;
                for (int ii = i; ii < i + step && ii < (int)W && !any_covered; ++ii) {
                    for (int jj = j; jj < j + step && jj < (int)H && !any_covered; ++jj) {
                        if (seen_finest[(size_t)ii * H + (size_t)jj]) any_covered = true;
                    }
                }
                if (any_covered) continue;

                MetaCell c;
                c.i_bl = i; c.j_bl = j; c.step = step; c.level = level;
                c.per_level_refine_count.assign(n_levels, 0);
                for (int L = 0; L < n_levels; ++L) {
                    int peak = 0;
                    for (int ii = i; ii < i + step && ii < (int)W; ++ii) {
                        for (int jj = j; jj < j + step && jj < (int)H; ++jj) {
                            peak = std::max(peak, tally[(size_t)ii * H + (size_t)jj][L]);
                        }
                    }
                    c.per_level_refine_count[L] = peak;
                }
                emitted.push_back(std::move(c));
                mm.n_baseline_cells++;
                for (int ii = i; ii < i + step && ii < (int)W; ++ii) {
                    for (int jj = j; jj < j + step && jj < (int)H; ++jj) {
                        seen_finest[(size_t)ii * H + (size_t)jj] = 1;
                    }
                }
            }
        }
    }

    mm.cells = std::move(emitted);

    log<LOG_INFO>(L"%1% || build_meta_mesh: aggregated %2% throws, finest=%3%x%4%, "
                  L"meta_cells=%5% (baseline=%6%, refined=%7%), threshold_count=%8%/%9%.")
        % __func__ % n_throws % mm.finest_nx % mm.finest_ny
        % (int)mm.cells.size() % mm.n_baseline_cells % mm.n_refined_cells
        % threshold_count % n_throws;

    return mm;
}

// ====================================================================
//  Section 4 — write_slice1_diagnostics
//
//  Writes one ROOT file containing per-throw and aggregate diagnostics.
//  The per-throw mesh visualisation mirrors PROsurf::PlotAMRMesh
//  (src/PROsurf.cxx:856-957) — *copied* as a free helper to avoid
//  constructing a temporary PROsurf instance.
// ====================================================================

namespace {

// Coordinate transform from finest-integer (i, j) to physical (x, y), honouring
// per-axis log10 flag from the model.
struct AxisXform {
    float x_lo, x_hi, y_lo, y_hi;
    int   finest_nx, finest_ny;
    bool  xlog, ylog;

    float i_to_x(int i) const {
        const float t = x_lo + (float)i / (float)finest_nx * (x_hi - x_lo);
        return xlog ? std::pow(10.0f, t) : t;
    }
    float j_to_y(int j) const {
        const float t = y_lo + (float)j / (float)finest_ny * (y_hi - y_lo);
        return ylog ? std::pow(10.0f, t) : t;
    }
};

// Draw one AMR mesh into the given canvas. Adapted from PROsurf::PlotAMRMesh
// (src/PROsurf.cxx:856-957) — pulled apart so the same body can be reused as
// the per-page draw routine in the multipage throws PDF.
//
// Does NOT call Print(); the caller owns the canvas lifecycle.
static void draw_amr_mesh_on_canvas(TCanvas &c,
                                    const PROmesh::AMRResult &amr,
                                    const PROmodel &model,
                                    bool logx, bool logy,
                                    size_t xaxis_idx, size_t yaxis_idx,
                                    const std::string &title_prefix = "AMR mesh")
{
    if (amr.leaves.empty() || amr.finest_nx <= 0 || amr.finest_ny <= 0) return;

    c.Clear();
    if (logx) c.SetLogx(); else c.SetLogx(0);
    if (logy) c.SetLogy(); else c.SetLogy(0);

    const bool xlog = (xaxis_idx < model.is_log10.size()) ? model.is_log10[xaxis_idx] : false;
    const bool ylog = (yaxis_idx < model.is_log10.size()) ? model.is_log10[yaxis_idx] : false;
    AxisXform A{amr.x_lo, amr.x_hi, amr.y_lo, amr.y_hi, amr.finest_nx, amr.finest_ny, xlog, ylog};

    const float xmin = A.i_to_x(0);
    const float xmax = A.i_to_x(amr.finest_nx);
    const float ymin = A.j_to_y(0);
    const float ymax = A.j_to_y(amr.finest_ny);

    int max_lvl = 0;
    for (const auto &leaf : amr.leaves) max_lvl = std::max(max_lvl, leaf.level);

    std::string xlabel = xaxis_idx < model.nparams ? model.pretty_param_names.at(xaxis_idx) : std::string("x");
    std::string ylabel = yaxis_idx < model.nparams ? model.pretty_param_names.at(yaxis_idx) : std::string("y");
    const std::string title = title_prefix + ";" + xlabel + ";" + ylabel;
    // Canvas-owned frame: heap-allocated, ROOT cleans up when canvas is cleared/destroyed.
    TH1F *frame = new TH1F("amr_frame", title.c_str(), 1, xmin, xmax);
    frame->SetMinimum(ymin);
    frame->SetMaximum(ymax);
    frame->SetStats(0);
    frame->Draw();

    const int level_palette[6] = { kAzure - 9, kAzure - 4, kAzure + 1, kViolet - 4, kViolet + 1, kRed + 1 };
    for (const auto &leaf : amr.leaves) {
        const float xlo = A.i_to_x(leaf.i_bl);
        const float xhi = A.i_to_x(leaf.i_bl + leaf.step);
        const float ylo = A.j_to_y(leaf.j_bl);
        const float yhi = A.j_to_y(leaf.j_bl + leaf.step);
        TBox *box = new TBox(xlo, ylo, xhi, yhi);
        const int idx = std::min(leaf.level, 5);
        box->SetFillColorAlpha(level_palette[idx], 0.25f);
        box->SetLineColor(kBlack);
        box->SetLineWidth(1);
        box->Draw();
    }

    const int contour_colors[5] = { kRed + 1, kOrange + 7, kGreen + 2, kMagenta, kBlack };
    for (size_t k = 0; k < amr.polylines.size(); ++k) {
        const int col = contour_colors[k % 5];
        for (const auto &seg : amr.polylines[k]) {
            float x0 = seg.p0.first,  x1 = seg.p1.first;
            float y0 = seg.p0.second, y1 = seg.p1.second;
            if (xlog) { x0 = std::pow(10.0f, x0); x1 = std::pow(10.0f, x1); }
            if (ylog) { y0 = std::pow(10.0f, y0); y1 = std::pow(10.0f, y1); }
            TLine *line = new TLine(x0, y0, x1, y1);
            line->SetLineColor(col);
            line->SetLineWidth(2);
            line->Draw();
        }
    }

    TPaveText *info = new TPaveText(0.1, 0.1, 0.3, 0.3, "NDC");
    info->SetFillColor(kWhite);
    info->SetBorderSize(1);
    info->SetTextSize(0.025);
    info->SetTextAlign(12);
    info->AddText(Form("AMR levels: 0..%d", max_lvl));
    info->AddText(Form("Total fits: %d", amr.total_fits));
    info->AddText(Form("Leaf cells: %d", (int)amr.leaves.size()));
    info->Draw();

    c.Update();
}

// Multi-page PDF of every per-throw AMR mesh, one page per throw. Uses the
// same open/append/close pattern as the *_Covar.pdf output in bin/PROfit.cxx
// (lines 2282/2296/2310/2312): single TCanvas reused across pages with
// c.Print(path + "[", "pdf") / c.Print(path, "pdf") / c.Print(path + "]", "pdf").
static void plot_amr_throws_multipage_pdf(
    const std::vector<PROmesh::AMRResult> &throws,
    const PROmodel &model,
    const std::string &filename,
    bool logx, bool logy,
    size_t xaxis_idx, size_t yaxis_idx)
{
    if (throws.empty()) return;

    TCanvas c("amr_mesh_multipage", "AMR Mesh Throws", 800, 700);
    c.Print((filename + "[").c_str(), "pdf");
    for (size_t t = 0; t < throws.size(); ++t) {
        const std::string page_title = "AMR mesh - throw " + std::to_string(t);
        draw_amr_mesh_on_canvas(c, throws[t], model, logx, logy,
                                xaxis_idx, yaxis_idx, page_title);
        c.Print(filename.c_str(), "pdf");
    }
    c.Print((filename + "]").c_str(), "pdf");
    log<LOG_INFO>(L"%1% || wrote multipage throw PDF %2% (%3% pages).")
        % __func__ % filename.c_str() % (int)throws.size();
}

// One-PDF visualisation of the *merged* meta-mesh. Same visual idiom as
// plot_amr_mesh_pdf: TBox per cell with the level palette + black borders.
// Difference: alpha is modulated by agreement strength (how many throws
// refined this cell at its assigned level), so cells where the throws
// strongly agreed appear saturated and cells that barely cleared p_thresh
// appear translucent — at a glance you see *where* the throws gathered.
static void plot_metamesh_pdf(const MetaMesh &mm,
                              const PROmodel &model,
                              const std::string &filename,
                              int n_throws,
                              float p_thresh,
                              int baseline_level,
                              bool logx, bool logy,
                              size_t xaxis_idx, size_t yaxis_idx)
{
    if (mm.cells.empty() || mm.finest_nx <= 0 || mm.finest_ny <= 0) {
        log<LOG_WARNING>(L"%1% || plot_metamesh_pdf: empty mesh, skipping.") % __func__;
        return;
    }

    const bool xlog = (xaxis_idx < model.is_log10.size()) ? model.is_log10[xaxis_idx] : false;
    const bool ylog = (yaxis_idx < model.is_log10.size()) ? model.is_log10[yaxis_idx] : false;
    AxisXform A{mm.x_lo, mm.x_hi, mm.y_lo, mm.y_hi, mm.finest_nx, mm.finest_ny, xlog, ylog};

    const float xmin = A.i_to_x(0);
    const float xmax = A.i_to_x(mm.finest_nx);
    const float ymin = A.j_to_y(0);
    const float ymax = A.j_to_y(mm.finest_ny);

    int max_lvl = 0;
    for (const auto &c : mm.cells) max_lvl = std::max(max_lvl, c.level);

    // Side-by-side layout: left pad = mesh, right pad = info panel.
    TCanvas c("metamesh", "Meta-Mesh", 1400, 800);
    TPad left_pad("mm_left", "", 0.00, 0.00, 0.66, 1.00);
    TPad right_pad("mm_right", "", 0.66, 0.00, 1.00, 1.00);
    left_pad.SetLeftMargin(0.13);
    left_pad.SetRightMargin(0.03);
    left_pad.SetTopMargin(0.08);
    left_pad.SetBottomMargin(0.12);
    if (logx) left_pad.SetLogx();
    if (logy) left_pad.SetLogy();
    right_pad.SetLeftMargin(0.02);
    right_pad.SetRightMargin(0.02);
    right_pad.SetTopMargin(0.04);
    right_pad.SetBottomMargin(0.04);
    left_pad.Draw();
    right_pad.Draw();

    // ---- Left pad: the mesh itself, no overlays. -----------------------------
    left_pad.cd();

    std::string xlabel = xaxis_idx < model.nparams ? model.pretty_param_names.at(xaxis_idx) : std::string("x");
    std::string ylabel = yaxis_idx < model.nparams ? model.pretty_param_names.at(yaxis_idx) : std::string("y");
    const std::string title = std::string("Meta-mesh (merged over throws);") + xlabel + ";" + ylabel;
    TH1F frame("mm_frame", title.c_str(), 1, xmin, xmax);
    frame.SetMinimum(ymin);
    frame.SetMaximum(ymax);
    frame.SetStats(0);
    frame.GetXaxis()->SetTitleSize(0.045);
    frame.GetYaxis()->SetTitleSize(0.045);
    frame.Draw();

    // Level → colour. Same palette as the per-throw plots.
    const int level_palette[6] = { kAzure - 9, kAzure - 4, kAzure + 1, kViolet - 4, kViolet + 1, kRed + 1 };

    // Draw shallowest-first so deeper cells overlay cleanly on shared edges.
    std::vector<const MetaCell*> sorted_cells;
    sorted_cells.reserve(mm.cells.size());
    for (const auto &mc : mm.cells) sorted_cells.push_back(&mc);
    std::sort(sorted_cells.begin(), sorted_cells.end(),
              [](const MetaCell *a, const MetaCell *b){ return a->level < b->level; });

    for (const MetaCell *mc : sorted_cells) {
        const float xlo = A.i_to_x(mc->i_bl);
        const float xhi = A.i_to_x(mc->i_bl + mc->step);
        const float ylo = A.j_to_y(mc->j_bl);
        const float yhi = A.j_to_y(mc->j_bl + mc->step);

        const int palette_idx = std::min(mc->level, 5);
        int refine_count_at_level = (mc->level < (int)mc->per_level_refine_count.size())
            ? mc->per_level_refine_count[mc->level] : 0;
        const float agreement = std::min(1.0f, (float)refine_count_at_level / (float)std::max(1, n_throws));

        TBox *box = new TBox(xlo, ylo, xhi, yhi);
        if (mc->level < baseline_level) {
            box->SetFillColorAlpha(kGray + 1, 0.15f);
        } else {
            const float alpha = std::min(1.0f, std::max(0.35f, agreement));
            box->SetFillColorAlpha(level_palette[palette_idx], alpha);
        }
        box->SetLineColor(kBlack);
        box->SetLineWidth(1);
        box->Draw();
    }

    // ---- Right pad: info only, no axes. --------------------------------------
    right_pad.cd();
    TPaveText *info = new TPaveText(0.02, 0.05, 0.98, 0.97, "NDC");
    info->SetFillColor(kWhite);
    info->SetBorderSize(1);
    info->SetTextSize(0.038);
    info->SetTextAlign(12);
    info->AddText("Meta-mesh summary");
    info->AddText("");
    info->AddText(Form("Throws merged: %d", n_throws));
    info->AddText(Form("p_{thresh}: %.3f", p_thresh));
    info->AddText(Form("  threshold count: #geq %d / %d throws",
                        std::max(1, (int)std::ceil(p_thresh * (float)n_throws)), n_throws));
    info->AddText(Form("Baseline level: %d", baseline_level));
    info->AddText(Form("Levels present: 0..%d", max_lvl));
    info->AddText(Form("Total cells: %d", (int)mm.cells.size()));
    info->AddText(Form("  refined : %d", mm.n_refined_cells));
    info->AddText(Form("  baseline: %d", mm.n_baseline_cells));
    info->AddText(Form("Finest grid: %d x %d", mm.finest_nx, mm.finest_ny));
    info->Draw();

    c.cd();
    c.Print(filename.c_str());
    log<LOG_INFO>(L"%1% || meta-mesh plot written to %2% (%3% cells, max level %4%).")
        % __func__ % filename.c_str() % (int)mm.cells.size() % max_lvl;
}

// One-PDF summary of a PEBank. Left pad: a TH2D heatmap (colz) at finest-grid
// resolution where the colour is the number of PEs banked at each cell,
// painted across that cell's footprint. Cell borders overlaid in black so the
// adaptive structure remains visible. Right pad: text panel with bank stats.
static void plot_pebank_summary_pdf(const PEBank &bank,
                                    const std::string &filename,
                                    const std::string &bank_path,
                                    const std::string &xlabel,
                                    const std::string &ylabel,
                                    bool logx, bool logy,
                                    bool xlog_axis, bool ylog_axis)
{
    if (bank.n_cells <= 0 || bank.finest_nx <= 0 || bank.finest_ny <= 0) {
        log<LOG_WARNING>(L"%1% || plot_pebank_summary_pdf: empty bank, skipping.") % __func__;
        return;
    }

    AxisXform A{bank.x_lo, bank.x_hi, bank.y_lo, bank.y_hi,
                bank.finest_nx, bank.finest_ny, xlog_axis, ylog_axis};

    // ---- Stats over the bank --------------------------------------------------
    int64_t total_pes = 0;
    int     min_pe = INT_MAX, max_pe = 0;
    std::vector<int> per_cell_counts;
    per_cell_counts.reserve((size_t)bank.n_cells);
    for (const auto &v : bank.cell_pes) {
        const int n = (int)v.size();
        total_pes += (int64_t)n;
        min_pe = std::min(min_pe, n);
        max_pe = std::max(max_pe, n);
        per_cell_counts.push_back(n);
    }
    if (per_cell_counts.empty()) { min_pe = max_pe = 0; }
    const float mean_pe = bank.n_cells > 0
        ? (float)total_pes / (float)bank.n_cells : 0.0f;
    int median_pe = 0;
    if (!per_cell_counts.empty()) {
        std::vector<int> sorted_counts = per_cell_counts;
        std::sort(sorted_counts.begin(), sorted_counts.end());
        median_pe = sorted_counts[sorted_counts.size() / 2];
    }
    int cells_at_max = 0;
    for (int n : per_cell_counts) if (n == max_pe) ++cells_at_max;

    // ---- Side-by-side layout: left mesh+heatmap, right info ------------------
    TCanvas c("pebank_summary", "PE-bank summary", 1400, 800);
    TPad left_pad("pb_left", "", 0.00, 0.00, 0.66, 1.00);
    TPad right_pad("pb_right", "", 0.66, 0.00, 1.00, 1.00);
    left_pad.SetLeftMargin(0.13);
    left_pad.SetRightMargin(0.14); // leave room for colz Z-axis
    left_pad.SetTopMargin(0.08);
    left_pad.SetBottomMargin(0.12);
    if (logx) left_pad.SetLogx();
    if (logy) left_pad.SetLogy();
    right_pad.SetLeftMargin(0.02);
    right_pad.SetRightMargin(0.02);
    right_pad.SetTopMargin(0.04);
    right_pad.SetBottomMargin(0.04);
    left_pad.Draw();
    right_pad.Draw();

    // ---- Left pad: PE-count heatmap + cell-border overlay --------------------
    left_pad.cd();
    gStyle->SetPalette(kBird); // standard ROOT modern palette (cool→warm).
    gStyle->SetOptStat(0);

    const int W = bank.finest_nx;
    const int H = bank.finest_ny;
    std::vector<double> ex(W + 1), ey(H + 1);
    for (int i = 0; i <= W; ++i) ex[(size_t)i] = (double)A.i_to_x(i);
    for (int j = 0; j <= H; ++j) ey[(size_t)j] = (double)A.j_to_y(j);
    TH2D *hmap = new TH2D("pebank_heatmap",
                          (std::string("PE-bank counts;") + xlabel + ";" + ylabel + ";N_{PE}").c_str(),
                          W, ex.data(), H, ey.data());
    hmap->SetStats(0);

    // Paint each cell's PE-count across its footprint.
    for (int c_idx = 0; c_idx < bank.n_cells; ++c_idx) {
        const int i0 = bank.cell_i_bl[(size_t)c_idx];
        const int j0 = bank.cell_j_bl[(size_t)c_idx];
        const int sp = bank.cell_step [(size_t)c_idx];
        const int cnt = per_cell_counts[(size_t)c_idx];
        for (int ii = i0; ii < i0 + sp && ii < W; ++ii) {
            for (int jj = j0; jj < j0 + sp && jj < H; ++jj) {
                hmap->SetBinContent(ii + 1, jj + 1, (double)cnt);
            }
        }
    }
    hmap->Draw("colz");

    // Overlay cell borders so the adaptive structure stays readable.
    for (int c_idx = 0; c_idx < bank.n_cells; ++c_idx) {
        const int i0 = bank.cell_i_bl[(size_t)c_idx];
        const int j0 = bank.cell_j_bl[(size_t)c_idx];
        const int sp = bank.cell_step [(size_t)c_idx];
        const float xlo = A.i_to_x(i0);
        const float xhi = A.i_to_x(i0 + sp);
        const float ylo = A.j_to_y(j0);
        const float yhi = A.j_to_y(j0 + sp);
        TBox *box = new TBox(xlo, ylo, xhi, yhi);
        box->SetFillStyle(0);      // transparent
        box->SetLineColor(kBlack);
        box->SetLineWidth(1);
        box->Draw();
    }

    // ---- Right pad: bank stats only ------------------------------------------
    right_pad.cd();
    TPaveText *info = new TPaveText(0.02, 0.05, 0.98, 0.97, "NDC");
    info->SetFillColor(kWhite);
    info->SetBorderSize(1);
    info->SetTextSize(0.038);
    info->SetTextAlign(12);
    info->AddText("PE-bank summary");
    info->AddText("");
    info->AddText(Form("Source: %s", bank_path.c_str()));
    info->AddText(Form("Cells in bank: %d", bank.n_cells));
    info->AddText(Form("Total PEs: %lld", (long long)total_pes));
    info->AddText(Form("PEs/cell  mean : %.1f", mean_pe));
    info->AddText(Form("          median: %d", median_pe));
    info->AddText(Form("          min/max: %d / %d", min_pe, max_pe));
    info->AddText(Form("Cells at max (%d): %d", max_pe, cells_at_max));
    info->AddText(Form("Finest grid: %d x %d", bank.finest_nx, bank.finest_ny));
    info->AddText(Form("Max AMR level: %d", bank.max_levels));
    info->Draw();

    c.cd();
    c.Print(filename.c_str());
    log<LOG_INFO>(L"%1% || PE-bank summary written to %2% (cells=%3%, total_pes=%4%).")
        % __func__ % filename.c_str() % bank.n_cells % (long long)total_pes;
}

// Multi-page PDF: one page per meta-mesh cell, three histograms per page —
// χ²_syst / χ²_osc / Δχ² for the PEs banked at that cell. Uses the same
// open/append/close pattern as plot_amr_throws_multipage_pdf (mirroring the
// *_PROplot_Covar.pdf idiom at bin/PROfit.cxx:2282/2296/2312).
static void plot_pebank_pes_multipage_pdf(const PEBank &bank,
                                          const std::string &filename,
                                          const std::string &xlabel,
                                          const std::string &ylabel,
                                          bool xlog_axis, bool ylog_axis)
{
    if (bank.n_cells <= 0) {
        log<LOG_WARNING>(L"%1% || plot_pebank_pes_multipage_pdf: empty bank, skipping.") % __func__;
        return;
    }

    TCanvas c("pebank_pes_multipage", "PE-bank per-cell histograms", 1500, 550);
    c.Print((filename + "[").c_str(), "pdf");

    int pages_written = 0;
    int skipped_empty = 0;

    for (int idx = 0; idx < bank.n_cells; ++idx) {
        const auto &pes = bank.cell_pes[(size_t)idx];
        if (pes.empty()) { ++skipped_empty; continue; }

        c.Clear();

        // Pad layout: thin header strip on top (10%), three equal panels below.
        TPad *top = new TPad("hdr", "", 0.0, 0.88, 1.0, 1.00);
        TPad *p1  = new TPad("p1",  "", 0.000, 0.00, 0.333, 0.88);
        TPad *p2  = new TPad("p2",  "", 0.333, 0.00, 0.666, 0.88);
        TPad *p3  = new TPad("p3",  "", 0.666, 0.00, 1.000, 0.88);
        top->Draw(); p1->Draw(); p2->Draw(); p3->Draw();

        // Find min/max of each variable for auto-binned histograms.
        float syst_lo =  std::numeric_limits<float>::infinity();
        float syst_hi = -std::numeric_limits<float>::infinity();
        float osc_lo  =  std::numeric_limits<float>::infinity();
        float osc_hi  = -std::numeric_limits<float>::infinity();
        float dch_lo  =  std::numeric_limits<float>::infinity();
        float dch_hi  = -std::numeric_limits<float>::infinity();
        for (const auto &r : pes) {
            syst_lo = std::min(syst_lo, r.chi2_syst);
            syst_hi = std::max(syst_hi, r.chi2_syst);
            osc_lo  = std::min(osc_lo,  r.chi2_osc);
            osc_hi  = std::max(osc_hi,  r.chi2_osc);
            dch_lo  = std::min(dch_lo,  r.dchi2);
            dch_hi  = std::max(dch_hi,  r.dchi2);
        }
        // Pad the ranges so histogram doesn't clip at the edges.
        auto pad_range = [](float &lo, float &hi) {
            if (hi <= lo) { hi = lo + 1.0f; }
            const float w = hi - lo;
            lo -= 0.05f * w;
            hi += 0.05f * w;
        };
        pad_range(syst_lo, syst_hi);
        pad_range(osc_lo,  osc_hi);
        pad_range(dch_lo,  dch_hi);

        const int n_bins = std::max(20, (int)pes.size() / 5);

        // Histograms — allocated with new; ROOT will free them when canvas clears.
        TH1F *h_syst = new TH1F(Form("h_syst_%d", idx),
                                (std::string(";#chi^{2}_{syst};entries")).c_str(),
                                n_bins, syst_lo, syst_hi);
        TH1F *h_osc  = new TH1F(Form("h_osc_%d",  idx),
                                (std::string(";#chi^{2}_{osc};entries")).c_str(),
                                n_bins, osc_lo, osc_hi);
        TH1F *h_dch  = new TH1F(Form("h_dch_%d",  idx),
                                (std::string(";#Delta#chi^{2};entries")).c_str(),
                                n_bins, dch_lo, dch_hi);
        h_syst->SetFillColorAlpha(kAzure + 1, 0.4f);
        h_osc ->SetFillColorAlpha(kGreen + 2, 0.4f);
        h_dch ->SetFillColorAlpha(kRed   + 1, 0.4f);
        h_syst->SetLineColor(kAzure + 1);
        h_osc ->SetLineColor(kGreen + 2);
        h_dch ->SetLineColor(kRed   + 1);

        for (const auto &r : pes) {
            h_syst->Fill(r.chi2_syst);
            h_osc ->Fill(r.chi2_osc);
            h_dch ->Fill(r.dchi2);
        }

        // ---- Header: cell metadata --------------------------------------------
        top->cd();
        const float xphys = xlog_axis ? std::pow(10.0f, bank.cell_center_x[(size_t)idx])
                                      : bank.cell_center_x[(size_t)idx];
        const float yphys = ylog_axis ? std::pow(10.0f, bank.cell_center_y[(size_t)idx])
                                      : bank.cell_center_y[(size_t)idx];
        TPaveText *hdr = new TPaveText(0.02, 0.10, 0.98, 0.95, "NDC");
        hdr->SetFillColor(kWhite);
        hdr->SetBorderSize(0);
        hdr->SetTextAlign(12);
        hdr->SetTextSize(0.45);
        hdr->AddText(Form("Cell %d / %d   level %d   step %d   N_{PE}=%d   "
                          "%s=%.4g   %s=%.4g",
                          idx, bank.n_cells,
                          bank.cell_level[(size_t)idx],
                          bank.cell_step[(size_t)idx],
                          (int)pes.size(),
                          xlabel.c_str(), xphys,
                          ylabel.c_str(), yphys));
        hdr->Draw();

        // ---- Three histograms -------------------------------------------------
        p1->cd(); h_syst->Draw();
        p2->cd(); h_osc ->Draw();
        p3->cd(); h_dch ->Draw();

        c.cd();
        c.Print(filename.c_str(), "pdf");
        ++pages_written;
    }

    c.Print((filename + "]").c_str(), "pdf");
    log<LOG_INFO>(L"%1% || wrote per-cell PE histograms PDF %2% (%3% pages; skipped %4% empty cells).")
        % __func__ % filename.c_str() % pages_written % skipped_empty;
}

// Build a TH2D from a dense reconstructed Δχ² matrix, with physical-coord
// (log10-aware) bin edges. Used for the per-throw χ² heatmap.
static TH2D make_th2d_from_dense(const Eigen::MatrixXf &dense,
                                 const AxisXform &A,
                                 const std::string &name,
                                 const std::string &title)
{
    const int nx = (int)dense.cols();
    const int ny = (int)dense.rows();
    std::vector<double> ex(nx + 1), ey(ny + 1);
    for (int i = 0; i <= nx; ++i) ex[(size_t)i] = (double)A.i_to_x(i * A.finest_nx / std::max(1, nx));
    for (int j = 0; j <= ny; ++j) ey[(size_t)j] = (double)A.j_to_y(j * A.finest_ny / std::max(1, ny));
    TH2D h(name.c_str(), title.c_str(), nx, ex.data(), ny, ey.data());
    for (int ix = 0; ix < nx; ++ix) {
        for (int iy = 0; iy < ny; ++iy) {
            h.SetBinContent(ix + 1, iy + 1, (double)dense(iy, ix));
        }
    }
    return h;
}

} // anonymous

static void write_slice1_diagnostics(
    const std::vector<PROmesh::AMRResult> &throws,
    const MetaMesh &mm,
    const PROmodel &model,
    const PROsyst  & /*systs*/,
    const AdaptiveFCConfig &acfg,
    size_t xaxis_idx, size_t yaxis_idx,
    AdaptiveFCResult &result_out)
{
    const std::string root_path = acfg.output_tag + "_afc_slice1.root";
    TFile fout(root_path.c_str(), "RECREATE");
    if (fout.IsZombie()) {
        log<LOG_ERROR>(L"%1% || write_slice1_diagnostics: could not open %2% for writing.")
            % __func__ % root_path.c_str();
        return;
    }

    const bool xlog = (xaxis_idx < model.is_log10.size()) ? model.is_log10[xaxis_idx] : acfg.logx;
    const bool ylog = (yaxis_idx < model.is_log10.size()) ? model.is_log10[yaxis_idx] : acfg.logy;

    // ---- Per-throw subdirectory ------------------------------------------------
    TTree summary("summary", "per-throw AMR summary");
    int t_idx = 0, t_total_fits = 0, t_leaves = 0, t_contour_segs = 0;
    float t_min_chi2 = 0.0f;
    summary.Branch("throw_idx", &t_idx);
    summary.Branch("total_fits", &t_total_fits);
    summary.Branch("leaves", &t_leaves);
    summary.Branch("contour_segs", &t_contour_segs);
    summary.Branch("min_chi2", &t_min_chi2);

    for (size_t t = 0; t < throws.size(); ++t) {
        const auto &amr = throws[t];
        std::string dname = "throw_" + std::to_string(t);
        TDirectory *d = fout.mkdir(dname.c_str());
        d->cd();

        AxisXform A{amr.x_lo, amr.x_hi, amr.y_lo, amr.y_hi, amr.finest_nx, amr.finest_ny, xlog, ylog};

        if (amr.reconstructed_dense.size() > 0) {
            TH2D h = make_th2d_from_dense(amr.reconstructed_dense, A,
                                          "chi2_dense", "throw " + std::to_string(t) + " #Delta#chi^{2}");
            h.Write();
        }

        // Leaves overlay as one TGraph per cell (closed rectangle).
        for (size_t k = 0; k < amr.leaves.size(); ++k) {
            const auto &leaf = amr.leaves[k];
            const float xlo = A.i_to_x(leaf.i_bl);
            const float xhi = A.i_to_x(leaf.i_bl + leaf.step);
            const float ylo = A.j_to_y(leaf.j_bl);
            const float yhi = A.j_to_y(leaf.j_bl + leaf.step);
            const double xs[5] = {xlo, xhi, xhi, xlo, xlo};
            const double ys[5] = {ylo, ylo, yhi, yhi, ylo};
            TGraph g(5, xs, ys);
            g.SetName(("leaf_" + std::to_string(k)).c_str());
            g.SetTitle(("level " + std::to_string(leaf.level)).c_str());
            g.Write();
        }

        // Contour polylines per CL level (one TGraph per segment).
        for (size_t cl = 0; cl < amr.polylines.size(); ++cl) {
            for (size_t s = 0; s < amr.polylines[cl].size(); ++s) {
                const auto &seg = amr.polylines[cl][s];
                float x0 = seg.p0.first, x1 = seg.p1.first;
                float y0 = seg.p0.second, y1 = seg.p1.second;
                if (xlog) { x0 = std::pow(10.0f, x0); x1 = std::pow(10.0f, x1); }
                if (ylog) { y0 = std::pow(10.0f, y0); y1 = std::pow(10.0f, y1); }
                const double xs[2] = {x0, x1};
                const double ys[2] = {y0, y1};
                TGraph g(2, xs, ys);
                g.SetName(Form("contour_cl%zu_seg%zu", cl, s));
                g.Write();
            }
        }

        fout.cd();
        t_idx          = (int)t;
        t_total_fits   = amr.total_fits;
        t_leaves       = (int)amr.leaves.size();
        t_contour_segs = 0;
        for (const auto &poly : amr.polylines) t_contour_segs += (int)poly.size();
        t_min_chi2     = amr.min_chi2;
        summary.Fill();
    }
    summary.Write();

    // ---- Aggregate (meta-mesh) subdirectory ------------------------------------
    TDirectory *mdir = fout.mkdir("metamesh");
    mdir->cd();

    AxisXform Ag{mm.x_lo, mm.x_hi, mm.y_lo, mm.y_hi, mm.finest_nx, mm.finest_ny, xlog, ylog};

    // Per-level refine-count heatmap at the finest grid resolution.
    if (mm.finest_nx > 0 && mm.finest_ny > 0) {
        const int W = mm.finest_nx;
        const int H = mm.finest_ny;
        std::vector<double> ex(W + 1), ey(H + 1);
        for (int i = 0; i <= W; ++i) ex[(size_t)i] = (double)Ag.i_to_x(i);
        for (int j = 0; j <= H; ++j) ey[(size_t)j] = (double)Ag.j_to_y(j);
        for (int L = 0; L <= mm.max_levels; ++L) {
            TH2D h(Form("refine_count_level%d", L),
                   Form("throws refining to level #geq %d;%s;%s", L,
                        model.pretty_param_names.size() > xaxis_idx ? model.pretty_param_names[xaxis_idx].c_str() : "x",
                        model.pretty_param_names.size() > yaxis_idx ? model.pretty_param_names[yaxis_idx].c_str() : "y"),
                   W, ex.data(), H, ey.data());
            // Each meta-cell carries its peak per-level refine count over its footprint;
            // paint that value across the cell.
            for (const auto &c : mm.cells) {
                const int cnt = (L < (int)c.per_level_refine_count.size()) ? c.per_level_refine_count[L] : 0;
                for (int ii = c.i_bl; ii < c.i_bl + c.step && ii < W; ++ii) {
                    for (int jj = c.j_bl; jj < c.j_bl + c.step && jj < H; ++jj) {
                        h.SetBinContent(ii + 1, jj + 1, (double)cnt);
                    }
                }
            }
            h.Write();
        }
    }

    // Meta-mesh cell overlay (one TGraph per cell border, named by level).
    for (size_t k = 0; k < mm.cells.size(); ++k) {
        const auto &c = mm.cells[k];
        const float xlo = Ag.i_to_x(c.i_bl);
        const float xhi = Ag.i_to_x(c.i_bl + c.step);
        const float ylo = Ag.j_to_y(c.j_bl);
        const float yhi = Ag.j_to_y(c.j_bl + c.step);
        const double xs[5] = {xlo, xhi, xhi, xlo, xlo};
        const double ys[5] = {ylo, ylo, yhi, yhi, ylo};
        TGraph g(5, xs, ys);
        g.SetName(Form("metacell_%zu_level%d", k, c.level));
        g.Write();
    }

    fout.cd();
    fout.Close();

    // All per-throw AMR meshes collected into one multipage PDF (one page per
    // throw). Pattern lifted from the *_PROplot_Covar.pdf output in
    // bin/PROfit.cxx:2282/2296/2312.
    const std::string throws_pdf = acfg.output_tag + "_throws.pdf";
    plot_amr_throws_multipage_pdf(throws, model, throws_pdf,
                                  acfg.logx, acfg.logy, xaxis_idx, yaxis_idx);

    // Single-page PDF of the merged meta-mesh — the "look at this" view.
    const std::string metamesh_pdf = acfg.output_tag + "_metamesh.pdf";
    plot_metamesh_pdf(mm, model, metamesh_pdf,
                      (int)throws.size(), acfg.p_thresh, acfg.baseline_level,
                      acfg.logx, acfg.logy, xaxis_idx, yaxis_idx);

    result_out.diag_root_path = root_path;
    log<LOG_INFO>(L"%1% || wrote diagnostics ROOT=%2% (throws=%3%, meta_cells=%4%); throws PDF=%5%; meta-mesh PDF=%6%.")
        % __func__ % root_path.c_str() % (int)throws.size() % (int)mm.cells.size()
        % throws_pdf.c_str() % metamesh_pdf.c_str();
}

// ====================================================================
//  Section 5 — Slice 2a: PEBank boost serialisation, worker, scheduler,
//  and the --mode init-bank driver.
//
//  Layout note: boost::serialization overloads must live in the boost
//  namespace, so we close PROfit, declare the overloads, then reopen.
// ====================================================================

} // namespace PROfit
namespace boost { namespace serialization {

template <class Archive>
void serialize(Archive &ar, PROfit::PEBankRecord &r, [[maybe_unused]] const unsigned int v) {
    ar & r.chi2_syst;
    ar & r.chi2_osc;
    ar & r.dchi2;
    ar & r.seed;
}

template <class Archive>
void serialize(Archive &ar, PROfit::PEBank &b, [[maybe_unused]] const unsigned int v) {
    ar & b.finest_nx;
    ar & b.finest_ny;
    ar & b.max_levels;
    ar & b.x_lo & b.x_hi & b.y_lo & b.y_hi;
    ar & b.n_cells;
    ar & b.cell_pes;
    ar & b.cell_center_x;
    ar & b.cell_center_y;
    ar & b.cell_i_bl;
    ar & b.cell_j_bl;
    ar & b.cell_step;
    ar & b.cell_level;
}

template <class Archive>
void serialize(Archive &ar, PROfit::MetaCell &c, [[maybe_unused]] const unsigned int v) {
    ar & c.i_bl;
    ar & c.j_bl;
    ar & c.step;
    ar & c.level;
    ar & c.per_level_refine_count;
}

template <class Archive>
void serialize(Archive &ar, PROfit::MetaMesh &m, [[maybe_unused]] const unsigned int v) {
    ar & m.cells;
    ar & m.finest_nx;
    ar & m.finest_ny;
    ar & m.max_levels;
    ar & m.x_lo & m.x_hi & m.y_lo & m.y_hi;
    ar & m.n_baseline_cells;
    ar & m.n_refined_cells;
}

}} // namespace boost::serialization

namespace PROfit {

bool save_bank(const PEBank &bank, const std::string &path) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open()) {
        log<LOG_ERROR>(L"%1% || save_bank: could not open %2% for writing.") % __func__ % path.c_str();
        return false;
    }
    try {
        boost::archive::binary_oarchive oa(ofs);
        uint32_t magic = PEBank::MAGIC;
        uint16_t version = PEBank::VERSION;
        oa & magic;
        oa & version;
        oa & const_cast<PEBank &>(bank); // boost serialize() takes non-const ref
    } catch (const std::exception &e) {
        log<LOG_ERROR>(L"%1% || save_bank: serialisation error: %2%") % __func__ % e.what();
        return false;
    }
    int64_t total_pes = 0;
    for (const auto &v : bank.cell_pes) total_pes += (int64_t)v.size();
    log<LOG_INFO>(L"%1% || save_bank: wrote %2% (cells=%3%, total_pes=%4%).")
        % __func__ % path.c_str() % bank.n_cells % total_pes;
    return true;
}

bool load_bank(PEBank &bank_out, const std::string &path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
        log<LOG_ERROR>(L"%1% || load_bank: could not open %2% for reading.") % __func__ % path.c_str();
        return false;
    }
    try {
        boost::archive::binary_iarchive ia(ifs);
        uint32_t magic = 0;
        uint16_t version = 0;
        ia & magic;
        ia & version;
        if (magic != PEBank::MAGIC) {
            log<LOG_ERROR>(L"%1% || load_bank: bad magic in %2% (got 0x%3$08x, expected 0x%4$08x).")
                % __func__ % path.c_str() % magic % (uint32_t)PEBank::MAGIC;
            return false;
        }
        if (version != PEBank::VERSION) {
            log<LOG_ERROR>(L"%1% || load_bank: version mismatch in %2% (got %3%, expected %4%).")
                % __func__ % path.c_str() % (int)version % (int)PEBank::VERSION;
            return false;
        }
        ia & bank_out;
    } catch (const std::exception &e) {
        log<LOG_ERROR>(L"%1% || load_bank: deserialisation error: %2%") % __func__ % e.what();
        return false;
    }
    int64_t total_pes = 0;
    for (const auto &v : bank_out.cell_pes) total_pes += (int64_t)v.size();
    log<LOG_INFO>(L"%1% || load_bank: read %2% (cells=%3%, total_pes=%4%).")
        % __func__ % path.c_str() % bank_out.n_cells % total_pes;
    return true;
}

bool save_mesh(const MetaMesh &mm, const std::string &path) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open()) {
        log<LOG_ERROR>(L"%1% || save_mesh: could not open %2% for writing.") % __func__ % path.c_str();
        return false;
    }
    try {
        boost::archive::binary_oarchive oa(ofs);
        uint32_t magic = MetaMesh::MAGIC;
        uint16_t version = MetaMesh::VERSION;
        oa & magic;
        oa & version;
        oa & const_cast<MetaMesh &>(mm);
    } catch (const std::exception &e) {
        log<LOG_ERROR>(L"%1% || save_mesh: serialisation error: %2%") % __func__ % e.what();
        return false;
    }
    log<LOG_INFO>(L"%1% || save_mesh: wrote %2% (cells=%3%, finest=%4%x%5%).")
        % __func__ % path.c_str() % (int)mm.cells.size() % mm.finest_nx % mm.finest_ny;
    return true;
}

bool load_mesh(MetaMesh &mm_out, const std::string &path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) return false; // Silent: caller decides whether the absence is an error.
    try {
        boost::archive::binary_iarchive ia(ifs);
        uint32_t magic = 0;
        uint16_t version = 0;
        ia & magic;
        ia & version;
        if (magic != MetaMesh::MAGIC) {
            log<LOG_ERROR>(L"%1% || load_mesh: bad magic in %2% (got 0x%3$08x, expected 0x%4$08x).")
                % __func__ % path.c_str() % magic % (uint32_t)MetaMesh::MAGIC;
            return false;
        }
        if (version != MetaMesh::VERSION) {
            log<LOG_ERROR>(L"%1% || load_mesh: version mismatch in %2% (got %3%, expected %4%).")
                % __func__ % path.c_str() % (int)version % (int)MetaMesh::VERSION;
            return false;
        }
        ia & mm_out;
    } catch (const std::exception &e) {
        log<LOG_ERROR>(L"%1% || load_mesh: deserialisation error: %2%") % __func__ % e.what();
        return false;
    }
    log<LOG_INFO>(L"%1% || load_mesh: read %2% (cells=%3%, finest=%4%x%5%).")
        % __func__ % path.c_str() % (int)mm_out.cells.size() % mm_out.finest_nx % mm_out.finest_ny;
    return true;
}

// --------------------------------------------------------------------
//  Per-cell PE worker — *duplicated* from src/PROfc.cxx::fc_worker
//  (lines 5-134). Kept parallel until adaptive pipeline is validated.
//
//  Differences from fc_worker:
//    • Throws *one* PE per call (not args.todo) — outer loop lives in
//      the scheduler so a per-cell stop flag can interrupt mid-batch.
//    • Uses std::unique_ptr<PROmetric> instead of raw new/delete so an
//      early break doesn't leak.
//    • Pinned coordinates: phy_params[xaxis_idx] = cell_x_model,
//      phy_params[yaxis_idx] = cell_y_model — both in *model space*
//      (log10 of the physical value for log-axis params). Other physics
//      params set to model->default_val(i).
// --------------------------------------------------------------------

namespace {

// Bundle of inputs to one adaptive PE call.
struct AdaptivePEArgs {
    const PROconfig *config;
    const PROpeller *prop;
    const PROsyst   *systs;
    const PROmodel  *model;
    const Eigen::MatrixXf *L;    ///< Cholesky factor of total covariance.
    const PROfitterConfig *fitconfig;
    std::string chi2_kind;       ///< "PROchi" | "PROCNP" | "Poisson"
    bool   binned;
    size_t xaxis_idx, yaxis_idx;
    float  cell_x_model, cell_y_model; ///< Cell-center coords in *model space* (log10(phys) for log-axis params).
    uint32_t seed;
};

// Run a single PE at the pinned cell coords. Returns the PEBankRecord with
// chi2_syst, chi2_osc, dchi2, and the seed used.
//
// Adapted body of fc_worker's per-PE loop (PROfc.cxx:40-131). Each call here
// is one iteration of that loop, with phy_params pinned to the cell.
static PEBankRecord run_one_pe(const AdaptivePEArgs &args)
{
    PEBankRecord rec;
    rec.seed = args.seed;
    std::mt19937 rng{args.seed};
    std::uniform_int_distribution<uint32_t> dseed(0, std::numeric_limits<uint32_t>::max());
    std::normal_distribution<float> d;

    const PROmodel &model = *args.model;
    const PROsyst  &systs = *args.systs;
    const PROconfig &config = *args.config;
    const PROpeller &prop = *args.prop;

    const size_t nphys   = model.nparams;
    const size_t nspline = systs.GetNSplines();
    const size_t nparams = nphys + nspline;

    // Build throw vector. Physics params: all at default, *except* the two
    // axis params pinned to the cell center.
    Eigen::VectorXf throws = Eigen::VectorXf::Zero((int)nparams);
    for (size_t i = 0; i < nphys; ++i) throws((int)i) = model.default_val(i);
    throws((int)args.xaxis_idx) = args.cell_x_model;
    throws((int)args.yaxis_idx) = args.cell_y_model;

    // Bounds for the syst-only fit: pin all physics params; let splines vary.
    Eigen::VectorXf lb_syst = Eigen::VectorXf::Constant((int)nparams, -3.0f);
    Eigen::VectorXf ub_syst = Eigen::VectorXf::Constant((int)nparams,  3.0f);
    for (size_t j = 0; j < nphys; ++j) {
        lb_syst((int)j) = throws((int)j);
        ub_syst((int)j) = throws((int)j);
    }
    Eigen::VectorXf lb_osc(lb_syst), ub_osc(ub_syst);
    for (size_t j = 0; j < nphys; ++j) {
        lb_osc((int)j) = model.lb(j);
        ub_osc((int)j) = model.ub(j);
    }
    for (size_t j = nphys; j < nparams; ++j) {
        const size_t si = j - nphys;
        const float lo = systs.spline_has_restrict[si] ? systs.spline_restrict_lo[si] : systs.spline_lo[si];
        const float hi = systs.spline_has_restrict[si] ? systs.spline_restrict_hi[si] : systs.spline_hi[si];
        lb_syst((int)j) = lo; ub_syst((int)j) = hi;
        lb_osc((int)j)  = lo; ub_osc((int)j)  = hi;
    }

    // Throw splines (Gaussian, respecting restrict ranges).
    for (size_t i = 0; i < nspline; ++i) {
        const float tlo = systs.spline_has_restrict[i] ? systs.spline_restrict_lo[i] : systs.spline_lo[i];
        const float thi = systs.spline_has_restrict[i] ? systs.spline_restrict_hi[i] : systs.spline_hi[i];
        do {
            throws((int)(i + nphys)) = d(rng);
        } while (throws((int)(i + nphys)) < tlo || throws((int)(i + nphys)) > thi);
    }

    // Stat throw vector.
    const int nbins_coll = config.m_num_variable_bins_total_collapsed[config.i_prime];
    Eigen::VectorXf throwC(nbins_coll);
    for (int i = 0; i < nbins_coll; ++i) throwC(i) = d(rng);

    // Build fake-data spectrum.
    PROchi::EvalStrategy strat = args.binned ? PROchi::BinnedChi2 : PROchi::EventByEvent;
    PROspec shifted = FillSpectra(config, prop, systs, model, throws, strat);
    PROspec newSpec = PROspec::PoissonVariation(
        PROspec(CollapseMatrix(config, shifted.Spec()) + (*args.L) * throwC,
                CollapseMatrix(config, shifted.Error())),
        dseed(rng));
    PROdata data(newSpec.Spec(), newSpec.Error());

    // Build metric (unique_ptr — early-stop safe).
    PROmetric::EvalStrategy mstrat = args.binned ? PROmetric::BinnedChi2 : PROmetric::EventByEvent;
    std::unique_ptr<PROmetric> metric;
    if (args.chi2_kind == "PROchi") {
        metric.reset(new PROchi("", config, prop, &systs, model, data, mstrat));
    } else if (args.chi2_kind == "PROCNP") {
        metric.reset(new PROCNP("", config, prop, &systs, model, data, mstrat));
    } else if (args.chi2_kind == "Poisson") {
        metric.reset(new PROpoisson("", config, prop, &systs, model, data, mstrat));
    } else {
        log<LOG_ERROR>(L"%1% || run_one_pe: unknown chi2 kind '%2%'.") % __func__ % args.chi2_kind.c_str();
        return rec;
    }

    // Syst-only fit (physics params pinned at cell center).
    PROfitter fitter_syst(ub_syst, lb_syst, *args.fitconfig, dseed(rng));
    metric->setBounds(lb_syst, ub_syst);
    rec.chi2_syst = fitter_syst.Fit(*metric);
    metric->freeParams();

    // Full (syst + osc) fit.
    PROfitter fitter_osc(ub_osc, lb_osc, *args.fitconfig, dseed(rng));
    metric->setBounds(lb_osc, ub_osc);
    std::vector<Eigen::VectorXf> seed_pts = {fitter_syst.best_fit};
    rec.chi2_osc = fitter_osc.Fit(*metric, seed_pts);
    fitter_osc.calcFreqSeedPoints(*metric);
    for (size_t i = 0; i < fitter_osc.freq_seed_points.size(); ++i) {
        const float c = fitter_osc.freq_seed_values.at(i);
        if (c < rec.chi2_osc) rec.chi2_osc = c;
    }

    rec.dchi2 = rec.chi2_syst - rec.chi2_osc;
    return rec;
}

// --------------------------------------------------------------------
//  schedule_pes — owns a threadpool, hands out cell-jobs to workers,
//  polls SequentialFCTest on every completed PE, stops cells once
//  Wilson half-width < eps, caps at n_pe_max.
// --------------------------------------------------------------------

struct CellState {
    int     cell_idx;
    int     n_done = 0;
    int     n_above_obs = 0;
    bool    stopped = false;
    std::vector<PEBankRecord> records; // local accumulator; copied into PEBank when stopped or capped
};

// Drive PE generation for every MetaCell. Slice-2a stopping rule: Wilson
// half-width < acfg.wilson_eps, with a floor of acfg.n_pe_min and a cap of
// acfg.n_pe_max. The Δχ²_obs threshold against which `n_above_obs` is
// counted is, for init-bank mode, the *median* Δχ² of PEs already at this
// cell — i.e. we're asking "is the sample stable enough to call the median
// to within wilson_eps?" Later modes (asimov/brazil/classify) supply a
// specific Δχ²_obs and re-run the classification.
static void schedule_pes(const AdaptiveFCConfig &acfg,
                         const PROconfig &config,
                         const PROpeller &prop,
                         const PROsyst   &systs,
                         const PROmodel  &model,
                         const PROfitterConfig &fitconfig,
                         PROseed &proseed,
                         const Eigen::MatrixXf &L,
                         size_t xaxis_idx, size_t yaxis_idx,
                         const std::vector<float> &cell_x_model,
                         const std::vector<float> &cell_y_model,
                         PEBank &bank_out,
                         int nthreads,
                         MultiPROgressBar &progress,
                         int progress_bar_idx,
                         AdaptiveFCResult &result_out)
{
    const int n_cells = (int)cell_x_model.size();
    if (n_cells == 0) return;

    bank_out.cell_pes.assign(n_cells, {});

    std::uniform_int_distribution<uint32_t> dseed(0, std::numeric_limits<uint32_t>::max());

    // Cell-job queue: simple atomic index + mutexed cell state map. For
    // n_cells ~ hundreds this is plenty; revisit if we ever go to N_cells > 10k.
    std::atomic<int> next_cell{0};
    std::mutex log_mutex;
    std::atomic<int64_t> total_pes{0};
    std::atomic<int> cells_capped{0};

    SequentialFCTest stop_rule;

    auto worker = [&](int thread_idx) {
        std::mt19937 thread_rng((*proseed.getThreadSeeds())[thread_idx]);
        while (true) {
            int c = next_cell.fetch_add(1);
            if (c >= n_cells) break;

            std::vector<PEBankRecord> local_pes;
            local_pes.reserve((size_t)acfg.n_pe_min);

            // Per-PE loop with Wilson-stop polling. The Bernoulli statistic
            // used for stopping is "fraction of PEs with dchi2 >= sample
            // median" — a stable proxy when we don't yet have a specific
            // Δχ²_obs to classify against.
            int n_above = 0;
            float running_median = 0.0f;
            std::vector<float> sorted_buf;
            sorted_buf.reserve((size_t)acfg.n_pe_max);

            for (int i = 0; i < acfg.n_pe_max; ++i) {
                AdaptivePEArgs args{};
                args.config = &config;
                args.prop   = &prop;
                args.systs  = &systs;
                args.model  = &model;
                args.L      = &L;
                args.fitconfig = &fitconfig;
                args.chi2_kind = acfg.chi2;
                args.binned    = acfg.binned;
                args.xaxis_idx = xaxis_idx;
                args.yaxis_idx = yaxis_idx;
                args.cell_x_model = cell_x_model[c];
                args.cell_y_model = cell_y_model[c];
                args.seed = dseed(thread_rng);

                PEBankRecord rec = run_one_pe(args);
                local_pes.push_back(rec);

                // Insert into sorted buffer (for running median).
                auto it = std::upper_bound(sorted_buf.begin(), sorted_buf.end(), rec.dchi2);
                sorted_buf.insert(it, rec.dchi2);
                running_median = sorted_buf[sorted_buf.size() / 2];
                n_above = 0;
                for (float v : sorted_buf) if (v >= running_median) ++n_above;

                const int n_done = i + 1;
                if (n_done >= acfg.n_pe_min) {
                    if (stop_rule.should_stop(n_done, n_above, acfg.wilson_eps)) {
                        std::lock_guard<std::mutex> lg(log_mutex);
                        log<LOG_INFO>(L"%1% || cell %2%/%3% stopped: n=%4%, halfwidth=%5%.")
                            % __func__ % c % n_cells % n_done
                            % stop_rule.wilson_halfwidth(n_done, n_above);
                        break;
                    }
                }
            }

            if ((int)local_pes.size() >= acfg.n_pe_max) {
                cells_capped.fetch_add(1);
                std::lock_guard<std::mutex> lg(log_mutex);
                log<LOG_WARNING>(L"%1% || cell %2%/%3% hit n_pe_max=%4% without Wilson-stopping.")
                    % __func__ % c % n_cells % acfg.n_pe_max;
            }

            total_pes.fetch_add((int64_t)local_pes.size());
            bank_out.cell_pes[c] = std::move(local_pes);
            progress.increment_bar(progress_bar_idx);
        }
    };

    std::vector<std::thread> tpool;
    tpool.reserve((size_t)std::max(1, nthreads));
    for (int t = 0; t < std::max(1, nthreads); ++t) tpool.emplace_back(worker, t);
    for (auto &th : tpool) th.join();

    result_out.total_pes_generated = total_pes.load();
    result_out.cells_hit_n_pe_max  = cells_capped.load();
    result_out.mean_pes_per_cell   = n_cells > 0 ? (float)total_pes.load() / (float)n_cells : 0.0f;

    log<LOG_INFO>(L"%1% || schedule_pes done: cells=%2%, total_pes=%3%, mean=%4%, capped=%5%.")
        % __func__ % n_cells % (int64_t)total_pes.load()
        % result_out.mean_pes_per_cell % (int)cells_capped.load();
}

// Compute per-cell (x, y) coords for every MetaCell — Option A: PE bank at
// each cell center.
//
// Returns values in *model space* — i.e. log10(physical) for log-axis params
// and linear for lin-axis params. This matches the convention used by
// FillSpectra / fc_worker / PROsurf::FillSurfaceAMR's EvalFn: the model
// always wants log10(sinsq2thmm) for a log-axis sinsq2thmm. Display code
// (slice-1 diagnostic plots) applies pow(10) when needed for ROOT axes.
//
// The xlog/ylog arguments are unused now but kept in the signature for
// readability — if we ever need physical-coord output again it goes here.
static void compute_cell_centers(const MetaMesh &mm,
                                 bool /*xlog*/, bool /*ylog*/,
                                 std::vector<float> &cx_out,
                                 std::vector<float> &cy_out)
{
    const int W = mm.finest_nx;
    const int H = mm.finest_ny;
    cx_out.clear(); cy_out.clear();
    cx_out.reserve(mm.cells.size());
    cy_out.reserve(mm.cells.size());
    for (const auto &c : mm.cells) {
        const float fi = (float)c.i_bl + 0.5f * (float)c.step;
        const float fj = (float)c.j_bl + 0.5f * (float)c.step;
        cx_out.push_back(mm.x_lo + fi / (float)W * (mm.x_hi - mm.x_lo));
        cy_out.push_back(mm.y_lo + fj / (float)H * (mm.y_hi - mm.y_lo));
    }
}

// --------------------------------------------------------------------
//  Asimov-mode helpers (slice 2b, asimov only).
// --------------------------------------------------------------------

// Observed Δχ² on the asimov dataset at every meta-mesh cell.
//
// The global (syst+osc) fit is computed once — physics floats freely, so the
// result is μ-independent. Per-cell fits then re-pin the two scanned axes at
// each cell center and run a syst-only fit (splines float; non-scanned
// physics params pinned at model->default_val(i)).
struct AsimovObs {
    float chi2_osc_global = 0.0f;        ///< Single global best fit.
    std::vector<float> chi2_syst;        ///< Per-cell syst-only chi^2.
    std::vector<float> dchi2_obs;        ///< chi2_syst[c] - chi2_osc_global per cell.
};

static AsimovObs compute_asimov_obs(
    const PROconfig &config,
    const PROpeller &prop,
    const PROsyst   &systs,
    const PROmodel  &model,
    const PROfitterConfig &fitconfig,
    const PROdata   &asimov_data,
    const std::string &chi2_kind,
    bool binned,
    size_t xaxis_idx, size_t yaxis_idx,
    const std::vector<float> &cell_x_model,
    const std::vector<float> &cell_y_model,
    PROseed &proseed,
    int nthreads,
    MultiPROgressBar &progress,
    int bar_idx)
{
    AsimovObs obs;
    const int n_cells = (int)cell_x_model.size();
    obs.chi2_syst.assign(n_cells, 0.0f);
    obs.dchi2_obs.assign(n_cells, 0.0f);
    if (n_cells == 0) return obs;

    const size_t nphys   = model.nparams;
    const size_t nspline = systs.GetNSplines();
    const size_t nparams = nphys + nspline;

    auto make_metric = [&](const PROdata &d) -> std::unique_ptr<PROmetric> {
        PROmetric::EvalStrategy mstrat = binned ? PROmetric::BinnedChi2 : PROmetric::EventByEvent;
        if (chi2_kind == "PROchi")    return std::unique_ptr<PROmetric>(new PROchi   ("", config, prop, &systs, model, d, mstrat));
        if (chi2_kind == "PROCNP")    return std::unique_ptr<PROmetric>(new PROCNP   ("", config, prop, &systs, model, d, mstrat));
        if (chi2_kind == "Poisson")   return std::unique_ptr<PROmetric>(new PROpoisson("", config, prop, &systs, model, d, mstrat));
        log<LOG_ERROR>(L"%1% || compute_asimov_obs: unknown chi2 kind '%2%'.") % __func__ % chi2_kind.c_str();
        return nullptr;
    };

    std::uniform_int_distribution<uint32_t> dseed(0, std::numeric_limits<uint32_t>::max());

    // ---- 1. Global (syst + physics) fit, once. -------------------------------
    {
        auto metric = make_metric(asimov_data);
        if (!metric) return obs;
        Eigen::VectorXf lb_osc((int)nparams), ub_osc((int)nparams);
        for (size_t j = 0; j < nphys; ++j) {
            lb_osc((int)j) = model.lb(j);
            ub_osc((int)j) = model.ub(j);
        }
        for (size_t j = nphys; j < nparams; ++j) {
            const size_t si = j - nphys;
            const float lo = systs.spline_has_restrict[si] ? systs.spline_restrict_lo[si] : systs.spline_lo[si];
            const float hi = systs.spline_has_restrict[si] ? systs.spline_restrict_hi[si] : systs.spline_hi[si];
            lb_osc((int)j) = lo; ub_osc((int)j) = hi;
        }
        metric->setBounds(lb_osc, ub_osc);
        std::mt19937 main_rng((*proseed.getThreadSeeds())[0]);
        PROfitter fitter_osc(ub_osc, lb_osc, fitconfig, dseed(main_rng));
        obs.chi2_osc_global = fitter_osc.Fit(*metric);
        // Try the freq-seed alternatives — same trick as fc_worker.
        fitter_osc.calcFreqSeedPoints(*metric);
        for (size_t i = 0; i < fitter_osc.freq_seed_points.size(); ++i) {
            const float c = fitter_osc.freq_seed_values.at(i);
            if (c < obs.chi2_osc_global) obs.chi2_osc_global = c;
        }
        log<LOG_INFO>(L"%1% || asimov global fit: chi2_osc=%2%.") % __func__ % obs.chi2_osc_global;
    }

    // ---- 2. Per-cell syst-only fits, parallel. -------------------------------
    std::atomic<int> next_cell{0};

    auto worker = [&](int thread_idx) {
        std::mt19937 thread_rng((*proseed.getThreadSeeds())[thread_idx]);
        auto metric = make_metric(asimov_data);
        if (!metric) return;

        // Pre-build the spline bounds once (same for every cell).
        Eigen::VectorXf lb((int)nparams), ub((int)nparams);
        for (size_t j = 0; j < nphys; ++j) {
            lb((int)j) = model.default_val(j);
            ub((int)j) = model.default_val(j);
        }
        for (size_t j = nphys; j < nparams; ++j) {
            const size_t si = j - nphys;
            const float lo = systs.spline_has_restrict[si] ? systs.spline_restrict_lo[si] : systs.spline_lo[si];
            const float hi = systs.spline_has_restrict[si] ? systs.spline_restrict_hi[si] : systs.spline_hi[si];
            lb((int)j) = lo; ub((int)j) = hi;
        }

        while (true) {
            int c = next_cell.fetch_add(1);
            if (c >= n_cells) break;

            // Pin the two scanned axes at this cell's center.
            lb((int)xaxis_idx) = cell_x_model[(size_t)c];
            ub((int)xaxis_idx) = cell_x_model[(size_t)c];
            lb((int)yaxis_idx) = cell_y_model[(size_t)c];
            ub((int)yaxis_idx) = cell_y_model[(size_t)c];

            metric->setBounds(lb, ub);
            PROfitter fitter(ub, lb, fitconfig, dseed(thread_rng));
            const float chi2_s = fitter.Fit(*metric);
            obs.chi2_syst[(size_t)c] = chi2_s;
            obs.dchi2_obs[(size_t)c] = chi2_s - obs.chi2_osc_global;
            progress.increment_bar(bar_idx);
        }
    };

    std::vector<std::thread> tpool;
    tpool.reserve((size_t)std::max(1, nthreads));
    for (int t = 0; t < std::max(1, nthreads); ++t) tpool.emplace_back(worker, t);
    for (auto &th : tpool) th.join();

    return obs;
}

// Per-cell, per-CL verdict map produced by classify_against_bank.
struct CellVerdict {
    float crit_dchi2 = 0.0f;  ///< Empirical critical Δχ² at the bank's α-quantile.
    bool  included  = false;  ///< dchi2_obs ≤ crit_dchi2 at this CL.
    bool  decidable = false;  ///< Bank had enough PEs to give a stable quantile.
};

// Classify every cell at every requested CL, given asimov observations and a
// bank. Returns verdicts indexed by [cl_idx][cell_idx]. A cell with fewer than
// `min_pes_for_decision` PEs in the bank is marked `decidable = false`.
static std::vector<std::vector<CellVerdict>> classify_against_bank(
    const PEBank &bank,
    const AsimovObs &obs,
    const std::vector<float> &cl_targets,
    int min_pes_for_decision)
{
    const int n_cells = bank.n_cells;
    std::vector<std::vector<CellVerdict>> verdicts(cl_targets.size());
    for (auto &v : verdicts) v.assign(n_cells, CellVerdict{});

    for (int c = 0; c < n_cells; ++c) {
        const auto &pes = bank.cell_pes[(size_t)c];
        if ((int)pes.size() < min_pes_for_decision) continue; // leave decidable = false

        std::vector<float> sorted;
        sorted.reserve(pes.size());
        for (const auto &r : pes) sorted.push_back(r.dchi2);
        std::sort(sorted.begin(), sorted.end());

        for (size_t k = 0; k < cl_targets.size(); ++k) {
            const float crit = SequentialFCTest::crit_dchi2_at_cl(sorted, cl_targets[k]);
            CellVerdict v;
            v.crit_dchi2 = crit;
            v.included   = (obs.dchi2_obs[(size_t)c] <= crit);
            v.decidable  = true;
            verdicts[k][(size_t)c] = v;
        }
    }
    return verdicts;
}

// Multipage PDF: one page per CL. Left pad shows the meta-mesh with cells
// coloured by verdict (green = included, red = excluded, grey = undecidable).
// Right pad shows per-CL stats. Same idiom as plot_pebank_summary_pdf.
static void plot_asimov_verdict_pdf(
    const PEBank &bank,
    const AsimovObs &obs,
    const std::vector<float> &cl_targets,
    const std::vector<std::vector<CellVerdict>> &verdicts,
    const std::string &filename,
    const std::string &bank_path,
    const std::string &xlabel,
    const std::string &ylabel,
    bool logx, bool logy,
    bool xlog_axis, bool ylog_axis)
{
    if (bank.n_cells <= 0 || cl_targets.empty()) {
        log<LOG_WARNING>(L"%1% || plot_asimov_verdict_pdf: empty input, skipping.") % __func__;
        return;
    }
    AxisXform A{bank.x_lo, bank.x_hi, bank.y_lo, bank.y_hi,
                bank.finest_nx, bank.finest_ny, xlog_axis, ylog_axis};

    TCanvas c("asimov_verdict", "Asimov verdict", 1400, 800);
    c.Print((filename + "[").c_str(), "pdf");

    for (size_t k = 0; k < cl_targets.size(); ++k) {
        const float cl = cl_targets[k];
        const auto &verd = verdicts[k];

        c.Clear();
        TPad left("av_left",  "", 0.00, 0.00, 0.66, 1.00);
        TPad right("av_right","", 0.66, 0.00, 1.00, 1.00);
        left .SetLeftMargin(0.13);
        left .SetRightMargin(0.03);
        left .SetTopMargin(0.08);
        left .SetBottomMargin(0.12);
        if (logx) left.SetLogx();
        if (logy) left.SetLogy();
        right.SetLeftMargin(0.02);
        right.SetRightMargin(0.02);
        right.SetTopMargin(0.04);
        right.SetBottomMargin(0.04);
        left.Draw(); right.Draw();

        // ---- Left pad: cells coloured by verdict ----
        left.cd();
        const float xmin = A.i_to_x(0);
        const float xmax = A.i_to_x(bank.finest_nx);
        const float ymin = A.j_to_y(0);
        const float ymax = A.j_to_y(bank.finest_ny);
        TH1F *frame = new TH1F(Form("av_frame_%zu", k),
                               (std::string("Asimov verdict, CL=") + Form("%.3f", cl) +
                                ";" + xlabel + ";" + ylabel).c_str(),
                               1, xmin, xmax);
        frame->SetMinimum(ymin);
        frame->SetMaximum(ymax);
        frame->SetStats(0);
        frame->Draw();

        int n_in = 0, n_out = 0, n_undec = 0;
        for (int idx = 0; idx < bank.n_cells; ++idx) {
            const int i0 = bank.cell_i_bl[(size_t)idx];
            const int j0 = bank.cell_j_bl[(size_t)idx];
            const int sp = bank.cell_step [(size_t)idx];
            const float xlo = A.i_to_x(i0);
            const float xhi = A.i_to_x(i0 + sp);
            const float ylo = A.j_to_y(j0);
            const float yhi = A.j_to_y(j0 + sp);
            TBox *box = new TBox(xlo, ylo, xhi, yhi);
            const auto &v = verd[(size_t)idx];
            if (!v.decidable) {
                box->SetFillColorAlpha(kGray + 1, 0.30f);
                ++n_undec;
            } else if (v.included) {
                box->SetFillColorAlpha(kGreen + 2, 0.45f);
                ++n_in;
            } else {
                box->SetFillColorAlpha(kRed   + 1, 0.45f);
                ++n_out;
            }
            box->SetLineColor(kBlack);
            box->SetLineWidth(1);
            box->Draw();
        }

        // ---- Right pad: stats ----
        right.cd();
        TPaveText *info = new TPaveText(0.02, 0.05, 0.98, 0.97, "NDC");
        info->SetFillColor(kWhite);
        info->SetBorderSize(1);
        info->SetTextSize(0.038);
        info->SetTextAlign(12);
        info->AddText("Asimov verdict");
        info->AddText("");
        info->AddText(Form("Bank: %s", bank_path.c_str()));
        info->AddText(Form("CL: %.4f", cl));
        info->AddText(Form("Total cells: %d", bank.n_cells));
        info->AddText(Form("  inside  (green): %d", n_in));
        info->AddText(Form("  outside (red)  : %d", n_out));
        info->AddText(Form("  undecidable    : %d", n_undec));
        info->AddText("");
        info->AddText(Form("global #chi^{2}_{osc}: %.4f", obs.chi2_osc_global));
        info->Draw();

        c.cd();
        c.Print(filename.c_str(), "pdf");
    }

    c.Print((filename + "]").c_str(), "pdf");
    log<LOG_INFO>(L"%1% || asimov verdict PDF written to %2% (%3% CL pages).")
        % __func__ % filename.c_str() % (int)cl_targets.size();
}

// Build the finest-grid TH2D of `dchi2_obs - crit_dchi2` for a given CL by
// painting each cell's value across its footprint. Bins inside undecidable
// cells are left at 0 (the contour treats them as boundary cases — usually
// fine because undecidable cells are at the periphery of the active region).
static TH2D *build_fc_deviation_th2d(const PEBank &bank,
                                     const AsimovObs &obs,
                                     const std::vector<CellVerdict> &verd_cl,
                                     const AxisXform &A,
                                     const std::string &name)
{
    const int W = bank.finest_nx, H = bank.finest_ny;
    std::vector<double> ex(W + 1), ey(H + 1);
    for (int i = 0; i <= W; ++i) ex[(size_t)i] = (double)A.i_to_x(i);
    for (int j = 0; j <= H; ++j) ey[(size_t)j] = (double)A.j_to_y(j);
    TH2D *h = new TH2D(name.c_str(), ";;;#Delta#chi^{2}_{obs} - #Delta#chi^{2}_{c}",
                       W, ex.data(), H, ey.data());
    for (int idx = 0; idx < bank.n_cells; ++idx) {
        const auto &v = verd_cl[(size_t)idx];
        if (!v.decidable) continue;
        const float dev = obs.dchi2_obs[(size_t)idx] - v.crit_dchi2;
        const int i0 = bank.cell_i_bl[(size_t)idx];
        const int j0 = bank.cell_j_bl[(size_t)idx];
        const int sp = bank.cell_step [(size_t)idx];
        for (int ii = i0; ii < i0 + sp && ii < W; ++ii) {
            for (int jj = j0; jj < j0 + sp && jj < H; ++jj) {
                h->SetBinContent(ii + 1, jj + 1, (double)dev);
            }
        }
    }
    return h;
}

// Extract contour TGraphs at `level` from a TH2D using ROOT's CONT LIST
// machinery. Returns a vector of newly-allocated TGraph copies (ownership
// transferred to caller). May return empty if no contour exists at that level.
//
// Reference: standard ROOT idiom around TH2::Draw("CONT Z LIST") + the
// gROOT->GetListOfSpecials()->FindObject("contours") TObjArray.
static std::vector<TGraph*> extract_contour_graphs(TH2D *h, double level)
{
    std::vector<TGraph*> out;
    if (!h) return out;
    const double levels[1] = {level};
    h->SetContour(1, levels);

    // Draw onto a hidden temp canvas to populate gROOT's contour list.
    TCanvas tmp("tmp_contour_extract", "", 200, 200);
    tmp.cd();
    h->Draw("CONT Z LIST");
    tmp.Update();

    TObjArray *contours_array = (TObjArray*)gROOT->GetListOfSpecials()->FindObject("contours");
    if (!contours_array || contours_array->GetSize() < 1) return out;
    TList *level_contours = (TList*)contours_array->At(0);
    if (!level_contours) return out;

    TIter next(level_contours);
    while (TObject *o = next()) {
        TGraph *g_in = dynamic_cast<TGraph*>(o);
        if (!g_in || g_in->GetN() <= 0) continue;
        // Deep copy — the originals are owned by ROOT's special list and may
        // be invalidated on subsequent Draw calls.
        out.push_back(new TGraph(*g_in));
    }
    return out;
}

// Clean publication-style contour overlay: single canvas, axes only, one
// contour line per requested CL, optional injected-truth marker. The
// equivalent of the per-CL "verdict map" boundary, in one composite figure.
static void plot_asimov_contour_pdf(
    const PEBank &bank,
    const AsimovObs &obs,
    const std::vector<float> &cl_targets,
    const std::vector<std::vector<CellVerdict>> &verdicts,
    const std::string &filename,
    const std::string &xlabel,
    const std::string &ylabel,
    bool logx, bool logy,
    bool xlog_axis, bool ylog_axis,
    bool draw_truth_marker,
    float truth_x_phys,
    float truth_y_phys)
{
    if (bank.n_cells <= 0 || cl_targets.empty()) {
        log<LOG_WARNING>(L"%1% || plot_asimov_contour_pdf: empty input, skipping.") % __func__;
        return;
    }
    AxisXform A{bank.x_lo, bank.x_hi, bank.y_lo, bank.y_hi,
                bank.finest_nx, bank.finest_ny, xlog_axis, ylog_axis};

    TCanvas c("asimov_contour", "Asimov FC contour", 900, 800);
    if (logx) c.SetLogx();
    if (logy) c.SetLogy();
    c.SetLeftMargin(0.13);
    c.SetRightMargin(0.04);
    c.SetTopMargin(0.06);
    c.SetBottomMargin(0.12);

    const float xmin = A.i_to_x(0);
    const float xmax = A.i_to_x(bank.finest_nx);
    const float ymin = A.j_to_y(0);
    const float ymax = A.j_to_y(bank.finest_ny);

    TH1F *frame = new TH1F("asimov_contour_frame",
                           (std::string("FC contour (asimov);") + xlabel + ";" + ylabel).c_str(),
                           1, xmin, xmax);
    frame->SetMinimum(ymin);
    frame->SetMaximum(ymax);
    frame->SetStats(0);
    frame->GetXaxis()->SetTitleSize(0.045);
    frame->GetYaxis()->SetTitleSize(0.045);
    frame->Draw();

    const int contour_palette[5] = {kRed + 1, kAzure + 1, kGreen + 2, kMagenta + 1, kBlack};
    TLegend *leg = new TLegend(0.65, 0.75, 0.95, 0.93);
    leg->SetBorderSize(1);
    leg->SetFillColor(kWhite);
    leg->SetTextSize(0.030);

    // Stash extracted graphs so they live until c.Print returns.
    std::vector<TGraph*> all_segments;

    for (size_t k = 0; k < cl_targets.size(); ++k) {
        const int col = contour_palette[k % 5];
        TH2D *h_dev = build_fc_deviation_th2d(bank, obs, verdicts[k], A,
                                              Form("dev_cl%zu", k));
        auto segs = extract_contour_graphs(h_dev, 0.0);
        delete h_dev;
        if (segs.empty()) {
            log<LOG_WARNING>(L"%1% || no contour found at CL=%2% (deviation never crosses zero?).")
                % __func__ % cl_targets[k];
            continue;
        }
        for (TGraph *g : segs) {
            g->SetLineColor(col);
            g->SetLineWidth(3);
            g->Draw("L SAME");
            all_segments.push_back(g);
        }
        leg->AddEntry(segs.front(), Form("CL = %.3f", cl_targets[k]), "l");
    }

    if (draw_truth_marker) {
        TMarker *truth = new TMarker((double)truth_x_phys, (double)truth_y_phys, 29); // star
        truth->SetMarkerColor(kBlack);
        truth->SetMarkerSize(2.2);
        truth->Draw();
        leg->AddEntry(truth, "Injected truth", "p");
    }

    leg->Draw();

    c.Print(filename.c_str());
    log<LOG_INFO>(L"%1% || asimov contour PDF written to %2% (%3% CLs).")
        % __func__ % filename.c_str() % (int)cl_targets.size();

    for (TGraph *g : all_segments) delete g;
}

// Persist the contour TGraphs + a per-cell TTree to a ROOT file so the
// asimov results can be loaded in downstream analyses / notebooks without
// re-running the dispatcher.
static void save_asimov_root(
    const PEBank &bank,
    const AsimovObs &obs,
    const std::vector<float> &cl_targets,
    const std::vector<std::vector<CellVerdict>> &verdicts,
    const std::string &filename,
    bool xlog_axis, bool ylog_axis)
{
    TFile fout(filename.c_str(), "RECREATE");
    if (fout.IsZombie()) {
        log<LOG_ERROR>(L"%1% || save_asimov_root: could not open %2%.") % __func__ % filename.c_str();
        return;
    }

    AxisXform A{bank.x_lo, bank.x_hi, bank.y_lo, bank.y_hi,
                bank.finest_nx, bank.finest_ny, xlog_axis, ylog_axis};

    // Per-cell TTree with one row per cell and per-CL crit/verdict columns.
    TTree t("cells", "asimov per-cell results");
    int    cell_idx = 0, cell_i_bl = 0, cell_j_bl = 0, cell_level = 0, cell_step = 0;
    float  cell_x_phys = 0.0f, cell_y_phys = 0.0f;
    float  dchi2_obs = 0.0f, chi2_syst = 0.0f;
    const float chi2_osc_global = obs.chi2_osc_global;
    t.Branch("cell_idx",  &cell_idx);
    t.Branch("i_bl",      &cell_i_bl);
    t.Branch("j_bl",      &cell_j_bl);
    t.Branch("level",     &cell_level);
    t.Branch("step",      &cell_step);
    t.Branch("x_phys",    &cell_x_phys);
    t.Branch("y_phys",    &cell_y_phys);
    t.Branch("dchi2_obs", &dchi2_obs);
    t.Branch("chi2_syst", &chi2_syst);
    t.Branch("chi2_osc_global", const_cast<float*>(&chi2_osc_global));

    // Per-CL columns; one float (crit) + one int (verdict) per CL target.
    std::vector<float> crit_per_cl(cl_targets.size(), 0.0f);
    std::vector<int>   verdict_per_cl(cl_targets.size(), -1); // -1 = undecidable
    for (size_t k = 0; k < cl_targets.size(); ++k) {
        const std::string cl_label = Form("%.4f", cl_targets[k]);
        t.Branch(("crit_dchi2_cl_" + cl_label).c_str(), &crit_per_cl[k]);
        t.Branch(("verdict_cl_"    + cl_label).c_str(), &verdict_per_cl[k]);
    }

    for (int idx = 0; idx < bank.n_cells; ++idx) {
        cell_idx   = idx;
        cell_i_bl  = bank.cell_i_bl[(size_t)idx];
        cell_j_bl  = bank.cell_j_bl[(size_t)idx];
        cell_level = bank.cell_level[(size_t)idx];
        cell_step  = bank.cell_step [(size_t)idx];
        const float xm = bank.cell_center_x[(size_t)idx];
        const float ym = bank.cell_center_y[(size_t)idx];
        cell_x_phys = xlog_axis ? std::pow(10.0f, xm) : xm;
        cell_y_phys = ylog_axis ? std::pow(10.0f, ym) : ym;
        dchi2_obs   = obs.dchi2_obs[(size_t)idx];
        chi2_syst   = obs.chi2_syst[(size_t)idx];
        for (size_t k = 0; k < cl_targets.size(); ++k) {
            const auto &v = verdicts[k][(size_t)idx];
            crit_per_cl[k]    = v.crit_dchi2;
            verdict_per_cl[k] = !v.decidable ? -1 : (v.included ? 1 : 0);
        }
        t.Fill();
    }
    t.Write();

    // Per-CL contour TGraphs.
    for (size_t k = 0; k < cl_targets.size(); ++k) {
        TH2D *h_dev = build_fc_deviation_th2d(bank, obs, verdicts[k], A,
                                              Form("dev_save_cl%zu", k));
        auto segs = extract_contour_graphs(h_dev, 0.0);
        for (size_t s = 0; s < segs.size(); ++s) {
            segs[s]->SetName(Form("contour_cl_%.4f_seg%zu", cl_targets[k], s));
            segs[s]->Write();
        }
        for (TGraph *g : segs) delete g;
        delete h_dev;
    }

    fout.Close();
    log<LOG_INFO>(L"%1% || asimov ROOT artifact written to %2%.")
        % __func__ % filename.c_str();
}

} // anonymous

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
        const std::string bank_in = acfg.bank_path.empty()
            ? (acfg.output_tag + "_bank.bin")
            : acfg.bank_path;
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
        const std::string bank_in = acfg.bank_path.empty()
            ? (acfg.output_tag + "_bank.bin")
            : acfg.bank_path;
        PEBank bank;
        if (!load_bank(bank, bank_in)) {
            log<LOG_ERROR>(L"%1% || asimov: failed to load %2%.") % __func__ % bank_in.c_str();
            return res;
        }

        // Build the asimov dataset: noise-free expected counts under
        // fakeDataParams. CVParams flows through the metric automatically via
        // the existing PROfit infrastructure (same as surface / fc).
        PROspec asimov_spec = FillSpectra(config, prop, systs, *model,
                                          fakeDataParams, acfg.binned, config.i_prime);
        PROspec asimov_collapsed = PROspec(CollapseMatrix(config, asimov_spec.Spec()),
                                           CollapseMatrix(config, asimov_spec.Error()));
        PROdata asimov_data(asimov_collapsed.Spec(), asimov_collapsed.Error());

        log<LOG_INFO>(L"%1% || asimov: built data from fakeDataParams (no throws), classifying %2% cells against %3%.")
            % __func__ % bank.n_cells % bank_in.c_str();

        // Stop the outer throws progress bar (it was set up by PROfit.cxx for
        // a phase we don't run) and launch a cells bar for the per-cell fits.
        progress.finish_all(true);
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

    // Slice 2b (partial): init-bank + asimov are wired; brazil / classify
    // remain placeholders.
    if (acfg.mode != AdaptiveFCMode::InitBank) {
        log<LOG_WARNING>(L"%1% || mode '%2%' not yet implemented; treating as init-bank.")
            % __func__ % (int)acfg.mode;
    }

    // ---- Slice 1: prepass + meta-mesh + diagnostics. ------------------------
    // Cache check: if <output_tag>_mesh.bin exists and --rebuild-mesh wasn't
    // passed, skip the slow prepass+aggregation+diagnostics step entirely.
    const std::string mesh_path = acfg.output_tag + "_mesh.bin";
    MetaMesh mm;
    bool mesh_loaded_from_cache = false;
    if (!acfg.rebuild_mesh && load_mesh(mm, mesh_path)) {
        mesh_loaded_from_cache = true;
        log<LOG_INFO>(L"%1% || using cached meta-mesh from %2% (cells=%3%); "
                      L"skipping prepass + diagnostics. Pass --rebuild-mesh to force rebuild.")
            % __func__ % mesh_path.c_str() % (int)mm.cells.size();
        res.n_throws_done    = 0;
        res.n_meta_cells     = (int)mm.cells.size();
        res.n_baseline_cells = mm.n_baseline_cells;
        res.n_refined_cells  = mm.n_refined_cells;
    } else {
        std::vector<PROmesh::AMRResult> per_throw_meshes =
            generate_throws(config, prop, systs, *model, fitconfig, proseed,
                            fakeDataParams, acfg, nthreads, xaxis_idx, yaxis_idx, progress);
        res.n_throws_done = (int)per_throw_meshes.size();
        res.leaves_per_throw.reserve(per_throw_meshes.size());
        for (const auto &amr : per_throw_meshes) res.leaves_per_throw.push_back((int)amr.leaves.size());

        mm = build_meta_mesh(per_throw_meshes, acfg.p_thresh, acfg.baseline_level);
        res.n_meta_cells     = (int)mm.cells.size();
        res.n_baseline_cells = mm.n_baseline_cells;
        res.n_refined_cells  = mm.n_refined_cells;

        write_slice1_diagnostics(per_throw_meshes, mm, *model, systs, acfg,
                                 xaxis_idx, yaxis_idx, res);

        // Persist the fresh mesh so future invocations can skip the prepass.
        save_mesh(mm, mesh_path);
    }
    (void)mesh_loaded_from_cache; // currently informational only

    // ---- Slice 2a: per-cell PE bank generation + save. ----------------------
    if (mm.cells.empty()) {
        log<LOG_WARNING>(L"%1% || empty meta-mesh; skipping PE-bank generation.") % __func__;
        return res;
    }

    // Determine per-axis log10 flag the same way the diagnostic plotter does
    // (model overrides CLI when in conflict).
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

    // Dedicated progress bar tracking cells. We piggyback on the same
    // MultiPROgressBar already in flight (the throws bar at index 0) — add a
    // second slot via a new instance is harder, so for slice 2a we reuse bar 0
    // and the caller's progress display continues. (If you want a separate
    // bar, plumb a second MultiPROgressBar through.)
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

    // Stop the throws progress bar before launching the cells one, so the two
    // ANSI cursor-driven refresh loops don't fight. finish_all() rounds the
    // throws bar to 100% (true regardless of whether it ran or was cached).
    progress.finish_all(true);

    // Cells progress bar — sized for the actual number of cells, not n_throws.
    std::vector<std::pair<int, std::string>> cells_bar_cfg;
    cells_bar_cfg.push_back({(int)mm.cells.size(), "AFC cells (PEs)"});
    MultiPROgressBar cells_progress(cells_bar_cfg);
    cells_progress.initialize_display();
    cells_progress.start_display_thread();

    schedule_pes(acfg, config, prop, systs, *model, fitconfig, proseed, L,
                 xaxis_idx, yaxis_idx, cell_x_model, cell_y_model,
                 bank, nthreads, cells_progress, 0, res);

    cells_progress.finish_all(true);

    // Persist the bank. Default path is <output_tag>_bank.bin unless
    // --bank explicitly set.
    const std::string bank_path = acfg.bank_path.empty()
        ? (acfg.output_tag + "_bank.bin")
        : acfg.bank_path;
    if (save_bank(bank, bank_path)) {
        res.bank_path = bank_path;
    }

    // Auto-print the bank summary PDF as a side effect of init-bank. The same
    // helper is invoked standalone from --mode print-bank for later inspection.
    {
        const std::string summary_pdf = acfg.output_tag + "_bank_summary.pdf";
        const bool xlog_axis = (xaxis_idx < model->is_log10.size()) ? model->is_log10[xaxis_idx] : acfg.logx;
        const bool ylog_axis = (yaxis_idx < model->is_log10.size()) ? model->is_log10[yaxis_idx] : acfg.logy;
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

} // namespace PROfit
