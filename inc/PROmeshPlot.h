/**
 * @file PROmeshPlot.h
 * @brief Shared visualisation helpers for AMR meshes.
 * @author PROfit Collaboration
 *
 * @details Companion to PROmesh.h. Provides the coordinate transform from
 * finest-integer mesh indices to physical axis values (AxisXform) and the
 * canvas-level AMR mesh drawer used by both PROsurf::PlotAMRMesh and the
 * adaptive-FC diagnostics (src/PROAdaptiveFCplot.cxx). Kept separate from
 * PROmesh.h so the core AMR engine stays free of ROOT dependencies.
 */
#ifndef PROMESHPLOT_H
#define PROMESHPLOT_H

#include "PROmesh.h"
#include "PROmodel.h"

#include <cmath>
#include <string>

class TCanvas;

namespace PROfit {
namespace PROmesh {

    /// Coordinate transform from finest-integer (i, j) to physical (x, y),
    /// honouring per-axis log10 flag from the model.
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

    /**
     * @brief Draw one AMR mesh into the given canvas.
     * @details Renders each leaf cell as a level-coloured translucent box with
     * an opaque outline, overlays the contour polylines, and adds an info box
     * with total/per-level fit counts. Leaves are drawn shallowest-first so
     * deeper (smaller) cells layer cleanly over shared edges of their parents.
     * Each leaf uses a two-TBox fill ("F") + outline ("L") pair to work around
     * a ROOT painter behavior where SetFillColorAlpha suppresses the border.
     *
     * Does NOT call Print(); the caller owns the canvas lifecycle, so the same
     * body can serve single-plot and multipage-PDF outputs.
     */
    void draw_amr_mesh_on_canvas(TCanvas &c,
                                 const AMRResult &amr,
                                 const PROmodel &model,
                                 bool logx, bool logy,
                                 size_t xaxis_idx, size_t yaxis_idx,
                                 const std::string &title_prefix = "AMR mesh");

}  // namespace PROmesh
}  // namespace PROfit

#endif
