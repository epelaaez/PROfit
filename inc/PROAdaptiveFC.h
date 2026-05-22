/**
 * @file PROAdaptiveFC.h
 * @brief Adaptive Feldman-Cousins pipeline — public surface.
 *
 * Slice 1 (current): Wilks pre-pass over N independent throws + aggregated meta-mesh
 * + diagnostic ROOT artifact. Subsequent slices will add the sequential per-cell PE
 * bank, the data-classification step, and the bank-top-up loop during Brazil-band
 * construction.
 *
 * The implementation lives entirely in src/PROAdaptiveFC.cxx and is kept *parallel*
 * to src/PROfc.cxx (the existing brute-force FC). Code duplicated from there or
 * from PROsurf::FillSurfaceAMR is annotated in the .cxx with banner comments.
 */
#ifndef PRO_ADAPTIVE_FC_H
#define PRO_ADAPTIVE_FC_H

#include "PROconfig.h"
#include "PROpeller.h"
#include "PROsyst.h"
#include "PROseed.h"
#include "PROfitter.h"
#include "PROgress.h"
#include "PROmesh.h"

#include <Eigen/Eigen>

#include <string>
#include <vector>

namespace PROfit {

    /// Pipeline mode. Slice 1 only implements InitBank (which stops after the
    /// meta-mesh is built and diagnostics are written). Others are placeholders
    /// reserved for follow-up slices.
    enum class AdaptiveFCMode {
        InitBank,
        Asimov,
        Brazil,
        Classify,
    };

    /**
     * @brief CLI-driven configuration for the adaptive FC pipeline.
     *
     * Slice-1-relevant fields populated by the fc-adaptive subcommand. Slice-2
     * fields (bank path, CL targets, Wilson eps, PE bounds, ROI band) are declared
     * for forward CLI compat but ignored by slice-1 logic.
     */
    struct AdaptiveFCConfig {
        AdaptiveFCMode mode = AdaptiveFCMode::InitBank;

        // Wilks pre-pass.
        int   n_throws            = 200;
        int   prepass_amr_initial_x = 10;
        int   prepass_amr_initial_y = 10;
        int   prepass_amr_levels    = 3;
        float prepass_delta_widen   = 0.5f;
        std::vector<float> prepass_contour_levels = {2.30f, 5.99f}; ///< Wilks Δχ² target levels (1σ, 2σ at 2 dof by default).
        bool  stat_only_throws    = false;

        // Scan axes (mirror the `surface` subcommand semantics).
        std::string xvar = "sinsq2thmm";
        std::string yvar = "dmsq";
        float x_lo = 1e-4f, x_hi = 1.0f;
        float y_lo = 1e-2f, y_hi = 1e2f;
        bool  logx = true, logy = true;

        // Meta-mesh build.
        float p_thresh       = 0.05f;   ///< Refine cell if fraction of throws refining it ≥ p_thresh.
        int   baseline_level = 2;       ///< Levels < baseline_level are kept regardless of p_thresh.

        // Output naming.
        std::string output_tag = "afc_slice1";

        // chi^2 metric name and binned/eventbyevent flags (passed straight through
        // to the inner per-throw fits — mirrors fc_args.chi2 / fc_args.binned).
        std::string chi2 = "PROchi";
        bool        binned = true;

        // ---- Slice-2 placeholders (declared, not used in slice 1). ----
        std::string bank_path = "";
        std::vector<float> cl_targets = {0.683f, 0.90f, 0.954f};
        float wilson_eps = 0.05f;
        int   n_pe_min = 30;
        int   n_pe_max = 1000;
        float roi_band = 8.0f;
    };

    /**
     * @brief One aggregated meta-mesh cell.
     *
     * Coordinates are in the finest-integer system shared by every per-throw
     * AMRResult (see PROmesh::MeshCell). `per_level_refine_count[L]` is the
     * number of throws that refined this cell to depth ≥ L.
     */
    struct MetaCell {
        int i_bl   = 0;
        int j_bl   = 0;
        int step   = 0;
        int level  = 0;
        std::vector<int> per_level_refine_count;
    };

    /**
     * @brief Aggregated meta-mesh from N per-throw Wilks AMR meshes.
     *
     * `finest_nx`, `finest_ny` and the (x_lo, x_hi, y_lo, y_hi) box must be
     * identical across every per-throw AMRResult that contributed.
     */
    struct MetaMesh {
        std::vector<MetaCell> cells;
        int   finest_nx = 0;
        int   finest_ny = 0;
        int   max_levels = 0;
        float x_lo = 0.0f, x_hi = 0.0f;
        float y_lo = 0.0f, y_hi = 0.0f;

        // Cell counters for quick logging.
        int n_baseline_cells = 0;
        int n_refined_cells  = 0;
    };

    /**
     * @brief Output bundle from run_adaptive_fc (slice-1 fields).
     */
    struct AdaptiveFCResult {
        int  n_throws_done   = 0;
        int  n_meta_cells    = 0;
        int  n_baseline_cells = 0;
        int  n_refined_cells = 0;
        std::vector<int> leaves_per_throw;
        std::string diag_root_path;
    };

    /**
     * @brief Adaptive FC entry point. Slice 1 stops after the meta-mesh is
     * built and the diagnostic ROOT artifact is written.
     */
    AdaptiveFCResult run_adaptive_fc(
        const PROconfig &config,
        const PROpeller &prop,
        const PROsyst   &systs,
        const PROfitterConfig &fitconfig,
        PROseed         &proseed,
        const Eigen::VectorXf &fakeDataParams,
        const AdaptiveFCConfig &acfg,
        int nthreads,
        MultiPROgressBar &progress);

} // namespace PROfit

#endif
