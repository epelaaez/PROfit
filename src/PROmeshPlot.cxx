/**
 * @file PROmeshPlot.cxx
 * @brief Implementation of the shared AMR mesh visualisation helpers.
 * @author PROfit Collaboration
 *
 * @details The mesh drawer originated in PROsurf::PlotAMRMesh and was
 * improved in the adaptive-FC diagnostics (leaf sorting, two-TBox
 * rendering); this file is the single shared home for that version.
 */
#include "PROmeshPlot.h"

#include "TBox.h"
#include "TCanvas.h"
#include "TColor.h"
#include "TH1F.h"
#include "TLine.h"
#include "TPaveText.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace PROfit {
namespace PROmesh {

// Does NOT call Print(); the caller owns the canvas lifecycle.
void draw_amr_mesh_on_canvas(TCanvas &c,
                             const AMRResult &amr,
                             const PROmodel &model,
                             bool logx, bool logy,
                             size_t xaxis_idx, size_t yaxis_idx,
                             const std::string &title_prefix)
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
    // Draw shallowest-first so deeper (smaller, more numerous) cells layer
    // cleanly over shared edges of their parent. With cells of mixed sizes,
    // a deep cell's left edge can sit on a shallow cell's interior; without
    // ordering, the painter clobbers the deeper border.
    std::vector<const PROmesh::MeshCell*> sorted_leaves;
    sorted_leaves.reserve(amr.leaves.size());
    for (const auto &leaf : amr.leaves) sorted_leaves.push_back(&leaf);
    std::sort(sorted_leaves.begin(), sorted_leaves.end(),
              [](const PROmesh::MeshCell *a, const PROmesh::MeshCell *b){
                  return a->level < b->level;
              });
    // Two-TBox-per-leaf rendering: one for the translucent level fill, one
    // for the opaque black outline. The single-TBox path (set fill + line,
    // call Draw()) drops the border in this ROOT build whenever the fill
    // uses SetFillColorAlpha — Draw() short-circuits to "f only" mode.
    // Splitting into a Draw("F") fill followed by a Draw("L") outline with
    // SetFillStyle(0) forces both to render unconditionally.
    for (const PROmesh::MeshCell *leaf : sorted_leaves) {
        const float xlo = A.i_to_x(leaf->i_bl);
        const float xhi = A.i_to_x(leaf->i_bl + leaf->step);
        const float ylo = A.j_to_y(leaf->j_bl);
        const float yhi = A.j_to_y(leaf->j_bl + leaf->step);
        const int idx = std::min(leaf->level, 5);

        // (a) Translucent level fill — explicit "F" so it never draws a line.
        TBox *fill = new TBox(xlo, ylo, xhi, yhi);
        fill->SetFillColorAlpha(level_palette[idx], 0.20f);
        fill->SetLineColor(level_palette[idx]); // border matches fill if rendered
        fill->SetLineWidth(0);
        fill->Draw("F");

        // (b) Outline-only TBox with SetFillStyle(0) — guaranteed no fill, so
        // Draw("L") never blends with the level color and the black border
        // is opaque regardless of the underlying fill's alpha.
        TBox *border = new TBox(xlo, ylo, xhi, yhi);
        border->SetFillStyle(0);
        border->SetLineColor(kGray + 1);
        border->SetLineWidth(1);
        border->SetLineStyle(1);
        border->Draw("L");
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
    for (int L = 0; L <= max_lvl && L < 8; ++L) {
        info->AddText(Form("  level %d: %d fits", L, amr.fits_by_level[L]));
    }
    info->Draw();

    c.Update();
}

}  // namespace PROmesh
}  // namespace PROfit
