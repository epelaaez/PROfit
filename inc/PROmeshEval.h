/**
 * @file PROmeshEval.h
 * @brief Shared single-point evaluation body for AMR chi² surface scans.
 * @author PROfit Collaboration
 *
 * @details pinned_scan_eval() is the common per-point fit body used by both
 * PROsurf::FillSurfaceAMR and the adaptive-FC Wilks prepass
 * (src/PROAdaptiveFCmesh.cxx). It assembles the full (physics + spline)
 * bounds from the metric, pins the two scanned coordinates to the request
 * point, and runs PROfitter with reproducible per-key seeding
 * (fseed = req.key & 0xffffffff).
 *
 * Metric acquisition (thread_local clone of a pre-built metric vs per-thread
 * construction from a per-throw dataset) stays at the call site; this
 * function is called once per EvalRequest with a ready-to-use metric. It
 * lives in a separate header so PROmesh.h keeps zero dependency on
 * PROmetric/PROfitter and remains a pure callback-driven engine.
 */
#ifndef PROMESHEVAL_H
#define PROMESHEVAL_H

#include "PROmesh.h"
#include "PROmetric.h"
#include "PROfitter.h"

namespace PROfit {
namespace PROmesh {

    /**
     * @brief Fit one AMR grid point with the two scanned coordinates pinned.
     * @details Resets the metric, builds full lower/upper bounds from the
     * model's physics parameters plus the systematics' spline ranges, pins
     * parameters `x_idx` and `y_idx` to (req.x_phys, req.y_phys), and
     * minimises the remaining parameters with PROfitter. The fitter RNG is
     * seeded from the grid-point key so results are reproducible regardless
     * of thread scheduling. Warm-start seeds carried by the request are
     * forwarded to PROfitter::Fit when present.
     *
     * @param metric     Ready-to-use, thread-private metric (caller manages
     *                   thread_local acquisition).
     * @param fitconfig  Minimiser configuration.
     * @param x_idx      Full-parameter-vector index of the scanned x axis.
     * @param y_idx      Full-parameter-vector index of the scanned y axis.
     * @param req        AMR evaluation request (point, key, seeds).
     * @return           EvalResult with the minimum χ² and best-fit vector.
     */
    EvalResult pinned_scan_eval(PROmetric &metric,
                                const PROfitterConfig &fitconfig,
                                size_t x_idx, size_t y_idx,
                                const EvalRequest &req);

}  // namespace PROmesh
}  // namespace PROfit

#endif
