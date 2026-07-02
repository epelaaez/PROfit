/**
 * @file PROmesh.h
 * @brief Adaptive-mesh-refinement (AMR) contour finder for 2D chi² surfaces.
 * @author PROfit Collaboration
 *
 * @details PROmesh::run_amr replaces fixed dense-grid χ²-surface scans (e.g., the
 * 60×60 grid in PROsurf::FillSurface) with an adaptive scheme that concentrates
 * evaluations near the target contour level(s). For each refinement level, cells
 * that *straddle* a target Δχ² are subdivided 2×1 in each direction; cells in the
 * deep basin or far tails are left coarse. Effective resolution along the contour
 * grows as `initial_grid × 2^max_levels`, while typical total fit count is O(N)
 * per level (the contour's perimeter), giving 6–8× wall-time reduction over a
 * fixed dense grid for equivalent contour quality.
 *
 * The algorithm is purely user-callback driven: the caller supplies an `EvalFn`
 * that maps a single (x_phys, y_phys) request to a (χ², best_fit) pair. PROmesh
 * manages the integer-coordinate grid, parallel dispatch (a producer-consumer
 * queue across `nthreads` workers), 2:1 neighbour-balance enforcement, marching
 * squares contour extraction, and an optional bilinear-reconstructed dense
 * matrix for backwards-compat plotting paths.
 *
 * Threading is pipelined: workers consume the shared queue continuously across
 * level transitions. As soon as a cell's last corner is evaluated, that worker
 * classifies the cell (straddle test against any of `contour_levels`) and, if
 * straddling, generates child cells and pushes their new edge-midpoint and
 * center evaluations onto the queue — without waiting for the whole level to
 * complete. This avoids the 20–30% tail-thread idle a strict level-by-level
 * barrier would cause on sparse refinement levels.
 *
 * Warm-start seeds: each `EvalRequest` carries a `seeds` list. PROmesh fills
 * this from the cell-local corner best_fits already in `bestfit_map` — for an
 * edge midpoint, the two endpoint corners; for a cell center, all four corners.
 * For initial-level points (no prior context), the caller's `seed_points`
 * (typically `freq_seed_points` from a prior global fit) are used.
 */
#ifndef PROMESH_H
#define PROMESH_H

#include <Eigen/Eigen>

#include <cstdint>
#include <functional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace PROfit {
namespace PROmesh {

    /// Single point requested for evaluation by the AMR scheduler.
    struct EvalRequest {
        uint64_t key;                                ///< (i << 32) | j  in finest-integer coords; identifies the grid point.
        float    x_phys;                             ///< Physical x-coordinate (after axis log/lin transform).
        float    y_phys;                             ///< Physical y-coordinate.
        std::vector<Eigen::VectorXf> seeds;          ///< Warm-start LBFGS seed list. Cell-local corners when available; caller's globals otherwise.
    };

    /// Result of one fit at one (x, y) point.
    struct EvalResult {
        float            chi2;                       ///< Minimum χ² found at this point (full param vector minimised over).
        Eigen::VectorXf  best_fit;                   ///< Best-fit parameter vector. Stored for warm-starting neighbouring cells.
    };

    /// Single line segment of an extracted contour, in physical coordinates.
    struct ContourSegment {
        std::pair<float, float> p0;
        std::pair<float, float> p1;
    };

    /// One leaf (un-refined) cell of the AMR mesh, exposed for visualization.
    /// Coordinates are in the *finest-integer* grid, i.e. (i_bl, j_bl) is the
    /// bottom-left corner index in [0, finest_nx] × [0, finest_ny] and the cell
    /// extends to (i_bl + step, j_bl + step). Convert to physical coords via
    /// `phys = bound_lo + (i / float(finest_n)) * (bound_hi - bound_lo)`.
    struct MeshCell {
        int i_bl;
        int j_bl;
        int step;     ///< Cell side in finest-integer units (= 2^(max_levels - level)).
        int level;    ///< Refinement depth (0 = coarsest grid).
    };

    /// Tunable knobs.
    struct AMROptions {
        int   initial_nx     = 10;        ///< Coarsest x-grid resolution (number of cells along x).
        int   initial_ny     = 10;        ///< Coarsest y-grid resolution.
        int   max_levels     = 3;         ///< Number of refinement levels above the initial grid.
        float delta_widen    = 0.5f;      ///< Straddle-band widening (χ² units) to absorb noise/interp error.
        bool  balance_2to1   = true;      ///< Enforce 2:1 neighbour balance before subdividing.
        std::vector<float> contour_levels = {5.99f}; ///< Δχ² target levels (default 95% CL at 2 dof). Multi-target supported in one pass.
        int   nthreads       = 1;         ///< Worker thread count.
        bool  produce_dense  = true;      ///< If true, fill `reconstructed_dense` via bilinear interpolation for plot-compat.
        int   dense_nx       = 60;        ///< Dense reconstruction columns (x).
        int   dense_ny       = 60;        ///< Dense reconstruction rows (y).
        std::vector<Eigen::VectorXf> initial_seed_points; ///< Warm-start seeds for initial-level points (e.g., global-fit freq_seed_points).
    };

    /// Output bundle from run_amr.
    struct AMRResult {
        std::unordered_map<uint64_t, float>           chi2_map;     ///< χ² value at every evaluated grid point. Keyed by (i<<32)|j.
        std::unordered_map<uint64_t, Eigen::VectorXf> bestfit_map;  ///< Best-fit vectors at every evaluated point, parallel to chi2_map.
        std::vector<std::vector<ContourSegment>>      polylines;    ///< Contour line segments, one outer-vector entry per target in opts.contour_levels.
        std::vector<MeshCell>                         leaves;       ///< Un-refined cells of the final mesh — drives AMR visualisation.
        int              finest_nx = 0;               ///< Total finest-integer span on x  (= initial_nx × 2^max_levels).
        int              finest_ny = 0;               ///< Total finest-integer span on y.
        int              max_levels = 0;              ///< AMR refinement depth actually used (clamped opts.max_levels). finest_nx = initial_nx << max_levels.
        float            x_lo = 0.0f, x_hi = 0.0f;    ///< Scan bounds in transformed (log/lin) space, as passed to run_amr.
        float            y_lo = 0.0f, y_hi = 0.0f;
        float            min_chi2 = 0.0f;             ///< Global min χ² seen across all evaluations (NOT offset).
        Eigen::MatrixXf  reconstructed_dense;         ///< Bilinearly interpolated (dense_ny × dense_nx) Δχ² matrix; empty when produce_dense is false. Δχ² (offset by min_chi2).
        int              total_fits  = 0;             ///< Number of EvalFn calls dispatched.
        int              fits_by_level[8] = {0,0,0,0,0,0,0,0}; ///< Per-level fit count for diagnostics (truncated at 8 levels).
    };

    /// Single-point evaluation callback. Invoked concurrently from many threads
    /// inside run_amr; implementation must be thread-safe (e.g. via thread_local
    /// metric clones).
    using EvalFn = std::function<EvalResult(const EvalRequest&)>;

    /**
     * @brief Run adaptive-mesh-refinement contour finding on a 2D χ² surface.
     * @details The scanned region is the rectangle [x_lo, x_hi] × [y_lo, y_hi]
     * in *transformed* coordinates (i.e., if an axis is log10, the bounds are
     * passed in log10 space — this matches `PROsurf::edges_x/edges_y`).
     *
     * @param eval     Single-point evaluation function. Called concurrently
     *                 from up to `opts.nthreads` workers.
     * @param x_lo     Lower bound of the x scan range (transformed coords).
     * @param x_hi     Upper bound of the x scan range.
     * @param y_lo     Lower bound of the y scan range.
     * @param y_hi     Upper bound of the y scan range.
     * @param opts     Tuning knobs; see AMROptions.
     * @return         AMRResult with the sparse χ² map, polyline contours per
     *                 target level, optional bilinear dense matrix, and
     *                 per-level diagnostic counts.
     */
    AMRResult run_amr(EvalFn eval,
                      float x_lo, float x_hi,
                      float y_lo, float y_hi,
                      const AMROptions &opts);

}  // namespace PROmesh
}  // namespace PROfit

#endif
