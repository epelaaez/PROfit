/**
 * @file PROAdaptiveFCplot.cxx
 * @brief Adaptive Feldman-Cousins: all ROOT PDF and artifact output
 *        (per-throw meshes, meta-mesh, PE bank summaries, asimov verdicts
 *        and contours, brazil bands).
 * @author PROfit Collaboration
 *
 * @details Part of the adaptive FC pipeline (see inc/PROAdaptiveFC.h and
 * src/PROAdaptiveFCinternal.h for the file layout). Mesh pages are rendered
 * with the shared PROmesh::draw_amr_mesh_on_canvas (inc/PROmeshPlot.h).
 */
#include "PROAdaptiveFCinternal.h"

#include "PROlog.h"
#include "PROmeshPlot.h"

#include <Eigen/Eigen>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "TBox.h"
#include "TColor.h"
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
#include "TKey.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TTree.h"

namespace PROfit {
namespace afc {

// ====================================================================
//  Section 4 — write_slice1_diagnostics
//
//  Writes one ROOT file containing per-throw and aggregate diagnostics.
//  The per-throw mesh visualisation uses the shared
//  PROmesh::draw_amr_mesh_on_canvas (inc/PROmeshPlot.h), which is also
//  what PROsurf::PlotAMRMesh renders with.
// ====================================================================

using PROmesh::AxisXform;
using PROmesh::draw_amr_mesh_on_canvas;

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
//
// n_throws <= 0 means "no throw tallies available" (a mesh loaded from disk
// via --mode print-mesh, or a derived merge/cleanup mesh whose counters are
// zero): alpha modulation and the throw/p_thresh info lines are skipped.
void plot_metamesh_pdf(const MetaMesh &mm,
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
        const float agreement = n_throws > 0
            ? std::min(1.0f, (float)refine_count_at_level / (float)n_throws)
            : 1.0f; // no tallies: flat saturation

        TBox *box = new TBox(xlo, ylo, xhi, yhi);
        // n_throws <= 0 (loaded/derived mesh): baseline_level is not known,
        // so colour every cell by level instead of graying "baseline" cells.
        if (n_throws > 0 && mc->level < baseline_level) {
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
    if (n_throws > 0) {
        info->AddText(Form("Throws merged: %d", n_throws));
        info->AddText(Form("p_{thresh}: %.3f", p_thresh));
        info->AddText(Form("  threshold count: #geq %d / %d throws",
                            std::max(1, (int)std::ceil(p_thresh * (float)n_throws)), n_throws));
    } else {
        info->AddText("(loaded from file: throw tallies not shown)");
    }
    if (n_throws > 0) info->AddText(Form("Baseline level: %d", baseline_level));
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
void plot_pebank_summary_pdf(const PEBank &bank,
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
void plot_pebank_pes_multipage_pdf(const PEBank &bank,
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

void write_slice1_diagnostics(
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

// Multipage PDF: one page per CL. Left pad shows the meta-mesh with cells
// coloured by verdict (green = included, red = excluded, grey = undecidable).
// Right pad shows per-CL stats. Same idiom as plot_pebank_summary_pdf.
void plot_asimov_verdict_pdf(
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
// Build the FC deviation TH2D by *interpolating* between cell-center samples.
//
// Treats each meta-cell as a point sample at its center (in finest-grid
// integer coords). For every finest-grid bin, finds the K=4 nearest decidable
// cell-center samples by Euclidean distance and computes their inverse-
// distance-squared weighted average — the standard Shepard-style IDW with p=2.
//
// Why this is the right thing (vs cell-painting + TH2::Smooth(N)):
//   - Painted cells produce piecewise-constant footprints; the contour
//     extracted at iso-level 0 then hugs cell edges and looks stair-stepped.
//     TH2::Smooth(N) blurs in finest-bin space with an arbitrary kernel — the
//     result depends on N, and large cells get smeared more than small ones.
//   - IDW with K=4 nearest samples gives a smooth surface whose interpolation
//     resolution matches the local mesh density: refined regions (small cells,
//     close-packed centers) get tight interpolation, baseline regions (wide
//     centers) get a smoother fill. No arbitrary smoothing parameter.
//   - Surface passes through cell-center values to high precision (exact at
//     center bins; ~0 error at finer resolution).
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

    // Collect (cell_center_finest_coords, deviation_value) for decidable cells.
    struct Sample { float cx, cy, val; };
    std::vector<Sample> samples;
    samples.reserve((size_t)bank.n_cells);
    for (int idx = 0; idx < bank.n_cells; ++idx) {
        const auto &v = verd_cl[(size_t)idx];
        if (!v.decidable) continue;
        const float dev = obs.dchi2_obs[(size_t)idx] - v.crit_dchi2;
        const float fcx = (float)bank.cell_i_bl[(size_t)idx]
                        + 0.5f * (float)bank.cell_step[(size_t)idx];
        const float fcy = (float)bank.cell_j_bl[(size_t)idx]
                        + 0.5f * (float)bank.cell_step[(size_t)idx];
        samples.push_back({fcx, fcy, dev});
    }
    if (samples.empty()) return h;

    // For each finest-grid bin, IDW interpolation over the K=4 nearest samples.
    constexpr int K = 4;
    for (int i = 0; i < W; ++i) {
        const float bx = (float)i + 0.5f;
        for (int j = 0; j < H; ++j) {
            const float by = (float)j + 0.5f;

            // Track the K smallest squared distances (linear scan; K=4 makes
            // the worst-of-K replacement trivial). For n_cells ~ 1000 and
            // n_bins ~ 4096 this is ~16M float ops — sub-second.
            std::array<std::pair<float, size_t>, K> nearest;
            for (int k = 0; k < K; ++k) {
                nearest[k] = {std::numeric_limits<float>::infinity(), 0};
            }
            for (size_t s = 0; s < samples.size(); ++s) {
                const float dx = samples[s].cx - bx;
                const float dy = samples[s].cy - by;
                const float d2 = dx*dx + dy*dy;
                int worst = 0;
                for (int k = 1; k < K; ++k) {
                    if (nearest[k].first > nearest[worst].first) worst = k;
                }
                if (d2 < nearest[worst].first) nearest[worst] = {d2, s};
            }

            // IDW with p=2 (inverse squared distance). Exact pass-through at
            // sample locations — guard against d=0 explicitly.
            float num = 0.0f, den = 0.0f;
            bool exact_hit = false;
            for (int k = 0; k < K; ++k) {
                if (std::isinf(nearest[k].first)) continue;
                if (nearest[k].first < 1e-6f) {
                    h->SetBinContent(i + 1, j + 1,
                                      (double)samples[nearest[k].second].val);
                    exact_hit = true;
                    break;
                }
                const float w = 1.0f / nearest[k].first;
                num += w * samples[nearest[k].second].val;
                den += w;
            }
            if (!exact_hit && den > 0.0f) {
                h->SetBinContent(i + 1, j + 1, (double)(num / den));
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

    // Drop any stale "contours" list from a previous extraction: if the
    // current level yields no contour, ROOT leaves the old object in the
    // specials list and we would return the previous call's curves.
    if (TObject *stale = gROOT->GetListOfSpecials()->FindObject("contours")) {
        gROOT->GetListOfSpecials()->Remove(stale);
    }

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
void plot_asimov_contour_pdf(
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
    TLegend *leg = new TLegend(0.15, 0.13, 0.45, 0.30);
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetTextSize(0.030);

    // Stash extracted graphs so they live until c.Print returns.
    std::vector<TGraph*> all_segments;

    for (size_t k = 0; k < cl_targets.size(); ++k) {
        const int col = contour_palette[k % 5];
        // build_fc_deviation_th2d does IDW interpolation between cell-center
        // samples — smooth surface by construction, no Smooth() needed.
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
void save_asimov_root(
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

    // Per-CL contour TGraphs (smoothed to match the plotted curves).
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

// --------------------------------------------------------------------
//  Brazil-band helpers.
// --------------------------------------------------------------------

// Build a TH2D of per-cell inclusion fractions, IDW-interpolated between
// cell centers. Same IDW algorithm as build_fc_deviation_th2d, but the
// per-cell values are inclusion fractions in [0, 1].
//
// `upsample` (>=1, default 1) multiplies the TH2D resolution per axis on top
// of the bank's finest grid. Higher resolution → smoother marching-squares
// contour polylines extracted afterwards. Brazil-band plotting uses upsample
// 4 (256×256 from a 64×64 finest grid). IDW cost scales linearly with W*H
// and n_samples, so the 16× extra bins are still sub-second.
//
// Cells whose inclusion_frac is < 0 (the "permanently undecidable" sentinel)
// are skipped from the sample set — IDW then fills those locations from the
// nearest decided cells, which is the correct behaviour for the deep-basin
// region (no PEs available → the band is genuinely the "inside the FC
// contour" interior).
static TH2D *build_inclusion_th2d(const PEBank &bank,
                                  const std::vector<float> &inclusion_frac_per_cell,
                                  const AxisXform &A,
                                  const std::string &name,
                                  int upsample = 1)
{
    const int up = std::max(1, upsample);
    const int W = bank.finest_nx * up;
    const int H = bank.finest_ny * up;

    // Bin edges at the upsampled resolution, same physical bounds.
    AxisXform A_up{A.x_lo, A.x_hi, A.y_lo, A.y_hi, W, H, A.xlog, A.ylog};
    std::vector<double> ex(W + 1), ey(H + 1);
    for (int i = 0; i <= W; ++i) ex[(size_t)i] = (double)A_up.i_to_x(i);
    for (int j = 0; j <= H; ++j) ey[(size_t)j] = (double)A_up.j_to_y(j);
    TH2D *h = new TH2D(name.c_str(), ";;;P(included)", W, ex.data(), H, ey.data());

    // Collect samples at upsampled-grid sample coordinates (original cell
    // centers scaled by `up`). Skip undecidable cells.
    struct Sample { float cx, cy, val; };
    std::vector<Sample> samples;
    samples.reserve((size_t)bank.n_cells);
    for (int idx = 0; idx < bank.n_cells; ++idx) {
        const float v = inclusion_frac_per_cell[(size_t)idx];
        if (v < 0.0f) continue;
        const float fcx = ((float)bank.cell_i_bl[(size_t)idx]
                        + 0.5f * (float)bank.cell_step[(size_t)idx]) * (float)up;
        const float fcy = ((float)bank.cell_j_bl[(size_t)idx]
                        + 0.5f * (float)bank.cell_step[(size_t)idx]) * (float)up;
        samples.push_back({fcx, fcy, v});
    }
    if (samples.empty()) return h;

    constexpr int K = 4;
    for (int i = 0; i < W; ++i) {
        const float bx = (float)i + 0.5f;
        for (int j = 0; j < H; ++j) {
            const float by = (float)j + 0.5f;
            std::array<std::pair<float, size_t>, K> nearest;
            for (int k = 0; k < K; ++k) nearest[k] = {std::numeric_limits<float>::infinity(), 0};
            for (size_t s = 0; s < samples.size(); ++s) {
                const float dx = samples[s].cx - bx;
                const float dy = samples[s].cy - by;
                const float d2 = dx*dx + dy*dy;
                int worst = 0;
                for (int k = 1; k < K; ++k) {
                    if (nearest[k].first > nearest[worst].first) worst = k;
                }
                if (d2 < nearest[worst].first) nearest[worst] = {d2, s};
            }
            float num = 0.0f, den = 0.0f;
            bool exact_hit = false;
            for (int k = 0; k < K; ++k) {
                if (std::isinf(nearest[k].first)) continue;
                if (nearest[k].first < 1e-6f) {
                    h->SetBinContent(i + 1, j + 1, (double)samples[nearest[k].second].val);
                    exact_hit = true;
                    break;
                }
                const float w = 1.0f / nearest[k].first;
                num += w * samples[nearest[k].second].val;
                den += w;
            }
            if (!exact_hit && den > 0.0f) {
                h->SetBinContent(i + 1, j + 1, (double)(num / den));
            }
        }
    }
    return h;
}

// Flag the finest-grid bins traversed by the SAVED Brazil quantile contours:
// the TGraphs save_brazil_root wrote to <tag>_brazil.root
// (brazil_cl_<CL>_<qlabel>_seg<N>) — the exact curve objects the band PDF
// drew. Nothing is recomputed here: the brazil archive grows on every brazil
// invocation and the inclusion surface depends on the current bank and
// min-PE settings, so re-deriving contours at cleanup time can silently
// disagree with the band the user is looking at. Reading the artifact makes
// mesh and plot consistent by construction.
//
// `quantiles` must be among the five levels save_brazil_root stores
// (0.025, 0.16, 0.5, 0.84, 0.975); others are warned about and skipped.
// Graphs are used for every CL in `cl_targets` (empty = all CLs present).
// Each polyline segment is rasterized at sub-bin steps (no gaps at bin
// corners); `halo` then dilates the flagged set by that many finest bins
// (Chebyshev) so the mesh brackets the curve on both sides.
std::vector<uint8_t> flag_bins_from_saved_brazil_contours(
    const PEBank &bank,
    const std::string &brazil_root_path,
    const std::vector<float> &cl_targets,
    const std::vector<float> &quantiles,
    bool xlog_axis, bool ylog_axis,
    int halo,
    int &n_curves_used)
{
    const int W = bank.finest_nx, H = bank.finest_ny;
    std::vector<uint8_t> flags((size_t)W * (size_t)H, 0);
    n_curves_used = 0;

    // Quantile -> saved-label mapping; must match save_brazil_root.
    static const struct { float q; const char *label; } kQuantileTable[5] = {
        {0.025f, "q025"}, {0.16f, "q16"}, {0.5f, "median"},
        {0.84f, "q84"}, {0.975f, "q975"}};
    std::vector<std::string> want_substr;
    for (float q : quantiles) {
        bool found = false;
        for (const auto &e : kQuantileTable) {
            if (std::fabs(q - e.q) < 1e-4f) {
                want_substr.push_back(std::string("_") + e.label + "_seg");
                found = true;
                break;
            }
        }
        if (!found) {
            log<LOG_WARNING>(L"%1% || quantile %2% has no saved contour "
                             L"(saved levels: 0.025 0.16 0.5 0.84 0.975); skipping.")
                % __func__ % q;
        }
    }
    if (want_substr.empty()) return flags;

    TFile fin(brazil_root_path.c_str(), "READ");
    if (fin.IsZombie()) {
        log<LOG_ERROR>(L"%1% || could not open %2%.") % __func__ % brazil_root_path.c_str();
        return flags;
    }

    // Physical -> finest-grid fractional coordinate (inverse of i_to_x/j_to_y).
    auto phys_to_fi = [&](float x) -> float {
        const float t = xlog_axis ? std::log10(std::max(x, 1e-30f)) : x;
        return (t - bank.x_lo) / (bank.x_hi - bank.x_lo) * (float)W;
    };
    auto phys_to_fj = [&](float y) -> float {
        const float t = ylog_axis ? std::log10(std::max(y, 1e-30f)) : y;
        return (t - bank.y_lo) / (bank.y_hi - bank.y_lo) * (float)H;
    };
    auto flag_at = [&](float fi, float fj) {
        const int i = std::min(W - 1, std::max(0, (int)std::floor(fi)));
        const int j = std::min(H - 1, std::max(0, (int)std::floor(fj)));
        flags[(size_t)i * (size_t)H + (size_t)j] = 1;
    };

    TIter it(fin.GetListOfKeys());
    while (TKey *key = (TKey*)it()) {
        if (std::strcmp(key->GetClassName(), "TGraph") != 0) continue;
        TString nm(key->GetName());
        if (!nm.BeginsWith("brazil_cl_")) continue;
        float cl = -1.0f;
        if (std::sscanf(nm.Data(), "brazil_cl_%f_", &cl) != 1) continue;
        bool cl_ok = cl_targets.empty();
        for (float c : cl_targets) if (std::fabs(c - cl) < 1e-3f) { cl_ok = true; break; }
        if (!cl_ok) continue;
        bool label_ok = false;
        for (const auto &s : want_substr) if (nm.Contains(s.c_str())) { label_ok = true; break; }
        if (!label_ok) continue;

        TGraph *g = (TGraph*)key->ReadObj();
        if (!g || g->GetN() <= 0) { delete g; continue; }
        float prev_fi = 0.0f, prev_fj = 0.0f;
        for (int p = 0; p < g->GetN(); ++p) {
            const float fi = phys_to_fi((float)g->GetPointX(p));
            const float fj = phys_to_fj((float)g->GetPointY(p));
            if (p == 0) {
                flag_at(fi, fj);
            } else {
                const float d = std::max(std::fabs(fi - prev_fi),
                                         std::fabs(fj - prev_fj));
                const int nstep = std::max(1, (int)std::ceil(d * 2.0f));
                for (int s = 1; s <= nstep; ++s) {
                    const float t = (float)s / (float)nstep;
                    flag_at(prev_fi + t * (fi - prev_fi),
                            prev_fj + t * (fj - prev_fj));
                }
            }
            prev_fi = fi; prev_fj = fj;
        }
        delete g;
        ++n_curves_used;
    }
    fin.Close();

    if (halo > 0) {
        std::vector<uint8_t> dilated = flags;
        for (int i = 0; i < W; ++i) {
            for (int j = 0; j < H; ++j) {
                if (!flags[(size_t)i * (size_t)H + (size_t)j]) continue;
                for (int di = -halo; di <= halo; ++di) {
                    for (int dj = -halo; dj <= halo; ++dj) {
                        const int ni = i + di, nj = j + dj;
                        if (ni < 0 || nj < 0 || ni >= W || nj >= H) continue;
                        dilated[(size_t)ni * (size_t)H + (size_t)nj] = 1;
                    }
                }
            }
        }
        flags.swap(dilated);
    }
    return flags;
}

// Multi-page PDF, one page per requested CL. Each page draws the median
// (P=0.5), ±1σ (P=0.16, 0.84), and ±2σ (P=0.025, 0.975) contours of the
// per-cell inclusion-fraction field — the classic Brazil-band visualisation.
void plot_brazil_band_pdf(
    const PEBank &bank,
    const std::vector<std::vector<float>> &inclusion_frac, // [cl_idx][cell_idx]
    const std::vector<float> &cl_targets,
    const std::string &filename,
    const std::string & /*bank_path*/,
    const std::string &xlabel,
    const std::string &ylabel,
    bool logx, bool logy,
    bool xlog_axis, bool ylog_axis,
    bool draw_truth_marker,
    float truth_x_phys,
    float truth_y_phys,
    const std::vector<int> &n_throws_kept,
    const std::vector<int> &n_throws_dropped)
{
    if (bank.n_cells <= 0 || cl_targets.empty()) {
        log<LOG_WARNING>(L"%1% || plot_brazil_band_pdf: empty input, skipping.") % __func__;
        return;
    }
    AxisXform A{bank.x_lo, bank.x_hi, bank.y_lo, bank.y_hi,
                bank.finest_nx, bank.finest_ny, xlog_axis, ylog_axis};

    TCanvas c("brazil_band", "Brazil band", 900, 800);
    c.Print((filename + "[").c_str(), "pdf");

    const float xmin = A.i_to_x(0);
    const float xmax = A.i_to_x(bank.finest_nx);
    const float ymin = A.j_to_y(0);
    const float ymax = A.j_to_y(bank.finest_ny);

    // Style mirrors the CMS/ATLAS Brazil-band convention: filled ±2σ (pale yellow)
    // and ±1σ (pale green) regions, dashed black median line on top. Realised via
    // TH2::SetContour(4) + custom 5-color palette + Draw("CONT3"); median extracted
    // separately as TGraphs.
    const int brazil_yellow = TColor::GetColor(244, 229, 160);  // ±2σ band fill
    const int brazil_green  = TColor::GetColor(152, 215, 152);  // ±1σ band fill

    for (size_t k = 0; k < cl_targets.size(); ++k) {
        c.Clear();
        if (logx) c.SetLogx();
        if (logy) c.SetLogy();
        c.SetLeftMargin(0.13);
        c.SetRightMargin(0.04);
        c.SetTopMargin(0.06);
        c.SetBottomMargin(0.12);

        const float cl = cl_targets[k];
        // Empty frame title (CL info appears in the legend header instead).
        TH1F *frame = new TH1F(Form("brazil_frame_%zu", k),
                               (std::string(";") + xlabel + ";" + ylabel).c_str(),
                               1, xmin, xmax);
        frame->SetMinimum(ymin);
        frame->SetMaximum(ymax);
        frame->SetStats(0);
        frame->GetXaxis()->SetTitleSize(0.045);
        frame->GetYaxis()->SetTitleSize(0.045);
        frame->Draw();

        // High-res IDW surface so marching-squares contours are smooth.
        // 4× upsampling → 256×256 from a 64×64 finest grid (sub-second cost).
        TH2D *h_incl = build_inclusion_th2d(bank, inclusion_frac[k], A,
                                            Form("incl_cl%zu", k), /*upsample=*/ 4);

        // Extract each contour level as TGraphs once and reuse for both the
        // filled band and the outline overlay. extract_contour_graphs mutates
        // h_incl's contour level on each call, so we extract everything
        // up-front before any drawing.
        auto segs_q025 = extract_contour_graphs(h_incl, 0.025);
        auto segs_q16  = extract_contour_graphs(h_incl, 0.16);
        auto segs_med  = extract_contour_graphs(h_incl, 0.50);
        auto segs_q84  = extract_contour_graphs(h_incl, 0.84);
        auto segs_q975 = extract_contour_graphs(h_incl, 0.975);

        // (1) Brazil-band fills via cell-based, per-row run-length-encoded
        //   TBoxes.
        //
        //   Earlier attempts used h_incl (the IDW-smoothed surface) for the
        //   per-bin color decision. That gave smooth visual transitions but
        //   bled the ±2σ yellow into the deep basin: IDW averages a bin's
        //   value from its 4 nearest decided cells, so a bin sitting between
        //   a decided "deep-basin" cell (v≈1) and a decided "boundary" cell
        //   (v≈0.5) gets an interpolated value in the 0.84–0.975 band, even
        //   though no actual cell *believes* it sits in the inner-2σ rim.
        //   With large baseline cells in sparse-bank regions, the IDW
        //   smoothing zone covers a huge swath of the plot.
        //
        //   Fix: for the *fills*, each bin inherits the value of the
        //   meta-cell that physically contains it — no IDW. Cell-edges in
        //   the fills are then hidden by the smooth IDW-based contour
        //   outlines drawn in step (2). Result: bands appear *only* where
        //   some actual cell has its own inclusion fraction in the band
        //   range. Undecidable cells (sentinel v<0) are skipped entirely
        //   and stay white, which is the correct "unknown" presentation
        //   instead of being smeared into a band colour.
        //
        //   Five-region classification of P(included):
        //       v < 0.025 or v > 0.975          → outside bands (no fill)
        //       0.025 ≤ v < 0.16, 0.84 < v ≤ 0.975 → ±2σ outer/inner (yellow)
        //       0.16 ≤ v ≤ 0.84                 → ±1σ (green)
        const int W_up = h_incl->GetNbinsX();
        const int H_up = h_incl->GetNbinsY();
        // upsample factor relative to bank.finest_nx (must match the call
        // to build_inclusion_th2d above).
        const int up = std::max(1, W_up / std::max(1, bank.finest_nx));

        // Precompute a [W_up × H_up] lookup mapping each upsampled bin to
        // the index of the meta-cell that contains it. -1 = no cell covers
        // this bin (shouldn't happen if the meta-mesh tiles the parameter
        // space, but we guard for safety).
        std::vector<int> cell_id_at((size_t)W_up * (size_t)H_up, -1);
        for (int c = 0; c < bank.n_cells; ++c) {
            const int i0  = bank.cell_i_bl[(size_t)c] * up;
            const int j0  = bank.cell_j_bl[(size_t)c] * up;
            const int len = bank.cell_step[(size_t)c] * up;
            const int i_end = std::min(W_up, i0 + len);
            const int j_end = std::min(H_up, j0 + len);
            const int i_beg = std::max(0, i0);
            const int j_beg = std::max(0, j0);
            for (int ii = i_beg; ii < i_end; ++ii) {
                for (int jj = j_beg; jj < j_end; ++jj) {
                    cell_id_at[(size_t)ii * (size_t)H_up + (size_t)jj] = c;
                }
            }
        }

        auto color_for_v = [&](float v) -> int {
            if (v < 0.0f) return -1; // undecidable cell: leave white
            if      (v >= 0.025f && v < 0.16f)   return brazil_yellow;
            else if (v >= 0.16f  && v <= 0.84f)  return brazil_green;
            else if (v > 0.84f   && v <= 0.975f) return brazil_yellow;
            return -1; // outside bands
        };

        // RLE per row over cell-based v lookups.
        for (int j = 0; j < H_up; ++j) {
            const float ylo = (float)h_incl->GetYaxis()->GetBinLowEdge(j + 1);
            const float yhi = (float)h_incl->GetYaxis()->GetBinUpEdge(j + 1);
            int run_color = -1;  // -1 means "no active fillable run"
            int run_start = 0;
            for (int i = 0; i <= W_up; ++i) {
                int color = -2; // sentinel: forces emit at i == W_up
                if (i < W_up) {
                    const int cid = cell_id_at[(size_t)i * (size_t)H_up + (size_t)j];
                    color = (cid >= 0) ? color_for_v(inclusion_frac[k][(size_t)cid]) : -1;
                }
                if (color != run_color) {
                    if (run_color >= 0) {
                        const float xlo = (float)h_incl->GetXaxis()->GetBinLowEdge(run_start + 1);
                        const float xhi = (float)h_incl->GetXaxis()->GetBinUpEdge(i);
                        TBox *box = new TBox(xlo, ylo, xhi, yhi);
                        box->SetFillColor(run_color);
                        box->SetLineColor(run_color);
                        box->SetLineWidth(0);
                        box->Draw();
                    }
                    run_color = color;
                    run_start = i;
                }
            }
        }

        // (2) Thin black outlines on the band boundaries. The same extracted
        // TGraphs are restyled (line attrs swapped) and redrawn as polylines.
        auto outline_at = [&](std::vector<TGraph*> &segs) {
            for (TGraph *g : segs) {
                g->SetLineColor(kBlack);
                g->SetLineWidth(1);
                g->SetLineStyle(1); // solid
                g->Draw("L SAME");
            }
        };
        outline_at(segs_q025);
        outline_at(segs_q16);
        outline_at(segs_q84);
        outline_at(segs_q975);

        // (3) Median dashed black line on top.
        for (TGraph *g : segs_med) {
            g->SetLineColor(kBlack);
            g->SetLineStyle(2);
            g->SetLineWidth(2);
            g->Draw("L SAME");
        }

        // (4) Truth marker.
        if (draw_truth_marker) {
            TMarker *truth = new TMarker((double)truth_x_phys, (double)truth_y_phys, 29);
            truth->SetMarkerColor(kBlack);
            truth->SetMarkerSize(2.2);
            truth->Draw();
        }

        // (5) Legend with proxy entries for the filled bands. Header carries the CL.
        TLegend *leg = new TLegend(0.15, 0.13, 0.50, 0.34);
        leg->SetBorderSize(0);
        leg->SetFillStyle(0);
        leg->SetTextSize(0.030);
        const int n_kept = k < n_throws_kept.size()    ? n_throws_kept[k]    : 0;
        const int n_drop = k < n_throws_dropped.size() ? n_throws_dropped[k] : 0;
        if (n_drop > 0) {
            leg->SetHeader(Form("CL = %.1f%%  (N_{throws} = %d, %d closed removed)",
                                cl * 100.0f, n_kept, n_drop), "L");
        } else {
            leg->SetHeader(Form("CL = %.1f%%  (N_{throws} = %d)", cl * 100.0f, n_kept), "L");
        }

        // Median proxy (dashed black line).
        TGraph *median_proxy = new TGraph();
        median_proxy->SetLineColor(kBlack);
        median_proxy->SetLineStyle(2);
        median_proxy->SetLineWidth(2);
        leg->AddEntry(median_proxy, "Median Exclusion", "l");

        // ±1σ band proxy (green fill).
        TBox *box_1sig = new TBox();
        box_1sig->SetFillColor(brazil_green);
        leg->AddEntry(box_1sig, "#pm 1#sigma", "f");

        // ±2σ band proxy (yellow fill).
        TBox *box_2sig = new TBox();
        box_2sig->SetFillColor(brazil_yellow);
        leg->AddEntry(box_2sig, "#pm 2#sigma", "f");

        if (draw_truth_marker) {
            TMarker *truth_proxy = new TMarker(0, 0, 29);
            truth_proxy->SetMarkerColor(kBlack);
            truth_proxy->SetMarkerSize(1.8);
            leg->AddEntry(truth_proxy, "Injected truth", "p");
        }

        leg->Draw();

        c.RedrawAxis(); // keep axes on top of the filled regions

        c.Print(filename.c_str(), "pdf");

        delete h_incl;
        for (TGraph *g : segs_q025) delete g;
        for (TGraph *g : segs_q16)  delete g;
        for (TGraph *g : segs_med)  delete g;
        for (TGraph *g : segs_q84)  delete g;
        for (TGraph *g : segs_q975) delete g;
    }

    c.Print((filename + "]").c_str(), "pdf");
    log<LOG_INFO>(L"%1% || brazil band PDF written to %2% (%3% CLs).")
        % __func__ % filename.c_str() % (int)cl_targets.size();
}

// Save brazil-band artifacts to a ROOT file:
//   per_throw — TTree with one row per throw (throw_idx, global chi2_osc, mean
//               dchi2 across cells, fraction of cells inside the contour, and
//               per-CL kept_cl_* flag: 0 = dropped by the closed-contour
//               filter, so excluded from the inclusion_frac aggregation).
//   incl_clN  — TH2D of per-cell inclusion fraction (IDW-interpolated) per CL,
//               aggregated over KEPT throws only.
//   brazil_cl_*_q* — TGraph contour segments at each Brazil quantile.
// (Per-(throw, cell) observables are not duplicated here — they live in the
// <tag>_brazil.bin archive.)
void save_brazil_root(
    const PEBank &bank,
    const std::vector<std::vector<std::vector<uint8_t>>> &per_throw_verdicts, // [t][cl][cell]
    const std::vector<std::vector<float>> &per_throw_dchi2,                   // [t][cell]
    const std::vector<float> &per_throw_global_chi2,                          // [t]
    const std::vector<std::vector<float>> &inclusion_frac,                    // [cl][cell]
    const std::vector<std::vector<uint8_t>> &throw_kept,                      // [cl][t]
    const std::vector<float> &cl_targets,
    const std::string &filename,
    bool xlog_axis, bool ylog_axis)
{
    TFile fout(filename.c_str(), "RECREATE");
    if (fout.IsZombie()) {
        log<LOG_ERROR>(L"%1% || save_brazil_root: could not open %2%.") % __func__ % filename.c_str();
        return;
    }
    AxisXform A{bank.x_lo, bank.x_hi, bank.y_lo, bank.y_hi,
                bank.finest_nx, bank.finest_ny, xlog_axis, ylog_axis};

    const int n_throws = (int)per_throw_dchi2.size();
    const int n_cells  = bank.n_cells;
    const int n_cl     = (int)cl_targets.size();

    // Per-throw TTree.
    TTree t_throw("per_throw", "Brazil per-throw summary");
    int throw_idx = 0;
    float chi2_osc_global = 0.0f;
    float mean_dchi2 = 0.0f;
    std::vector<float> frac_in_per_cl(n_cl, 0.0f);
    std::vector<int>   kept_per_cl(n_cl, 1);
    t_throw.Branch("throw_idx", &throw_idx);
    t_throw.Branch("chi2_osc_global", &chi2_osc_global);
    t_throw.Branch("mean_dchi2", &mean_dchi2);
    for (int k = 0; k < n_cl; ++k) {
        const std::string lbl = Form("%.4f", cl_targets[(size_t)k]);
        t_throw.Branch(("frac_in_cl_" + lbl).c_str(), &frac_in_per_cl[(size_t)k]);
        t_throw.Branch(("kept_cl_"    + lbl).c_str(), &kept_per_cl[(size_t)k]);
    }
    for (int t = 0; t < n_throws; ++t) {
        throw_idx = t;
        chi2_osc_global = per_throw_global_chi2[(size_t)t];
        double sum_d = 0;
        for (int c = 0; c < n_cells; ++c) sum_d += per_throw_dchi2[(size_t)t][(size_t)c];
        mean_dchi2 = n_cells > 0 ? (float)(sum_d / (double)n_cells) : 0.0f;
        for (int k = 0; k < n_cl; ++k) {
            int n_in = 0;
            for (int c = 0; c < n_cells; ++c) n_in += per_throw_verdicts[(size_t)t][(size_t)k][(size_t)c];
            frac_in_per_cl[(size_t)k] = (float)n_in / (float)std::max(1, n_cells);
            kept_per_cl[(size_t)k] = ((size_t)k < throw_kept.size()
                                      && (size_t)t < throw_kept[(size_t)k].size())
                                     ? (int)throw_kept[(size_t)k][(size_t)t] : 1;
        }
        t_throw.Fill();
    }
    t_throw.Write();

    // Per-cell aggregate inclusion-fraction TH2Ds and Brazil-quantile contours.
    const float quantiles[5] = {0.025f, 0.16f, 0.5f, 0.84f, 0.975f};
    const char *q_labels[5]  = {"q025", "q16", "median", "q84", "q975"};
    for (int k = 0; k < n_cl; ++k) {
        // upsample 4 matches plot_brazil_band_pdf so the saved contour TGraphs
        // are the same curves as the plotted ones.
        TH2D *h_incl = build_inclusion_th2d(bank, inclusion_frac[(size_t)k], A,
                                            Form("incl_cl_%.4f", cl_targets[(size_t)k]),
                                            /*upsample=*/ 4);
        h_incl->Write();
        for (int q = 0; q < 5; ++q) {
            auto segs = extract_contour_graphs(h_incl, (double)quantiles[q]);
            for (size_t s = 0; s < segs.size(); ++s) {
                segs[s]->SetName(Form("brazil_cl_%.4f_%s_seg%zu",
                                       cl_targets[(size_t)k], q_labels[q], s));
                segs[s]->Write();
            }
            for (TGraph *g : segs) delete g;
        }
        delete h_incl;
    }

    fout.Close();
    log<LOG_INFO>(L"%1% || brazil ROOT artifact written to %2%.")
        % __func__ % filename.c_str();
}

} // namespace afc
} // namespace PROfit
