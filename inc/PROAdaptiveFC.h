/**
 * @file PROAdaptiveFC.h
 * @brief Adaptive Feldman-Cousins pipeline — public surface.
 *
 * Slice 1 (done): Wilks pre-pass over N independent throws + aggregated meta-mesh
 * + diagnostic ROOT artifact.
 *
 * Slice 2a (current): bookkeeping (PEBank + boost serialisation), sequential
 * Wilson stopping rule, and the `--mode init-bank` end-to-end path that throws
 * PEs at each MetaCell center, stops via the sequential rule, and saves a
 * versioned bank artifact.
 *
 * Slice 2b (deferred): asimov / brazil / classify modes.
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
#include "PROdata.h"

#include <Eigen/Eigen>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace PROfit {

    /// Pipeline mode. Slice 1 only implements InitBank (which stops after the
    /// meta-mesh is built and diagnostics are written). Others are placeholders
    /// reserved for follow-up slices.
    enum class AdaptiveFCMode {
        BuildMesh, ///< Wilks prepass + meta-mesh build + diagnostics, then save <tag>_mesh.bin and exit.
        InitBank,  ///< Load <tag>_mesh.bin, generate PE bank, save <tag>_bank.bin. Errors if mesh missing.
        PrintBank, ///< Load <tag>_bank.bin and write summary PDF(s). No fitting.
        Asimov,    ///< Load <tag>_bank.bin, classify the asimov dataset, produce contour PDF + ROOT.
        Brazil,    ///< (deferred) Brazil-band loop with bank top-up.
        Classify,  ///< (deferred) classify real data against bank.
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
        float prepass_delta_widen   = 0.05f; ///< Tiny by design: per-throw global-fit warm-starts make AMR fitter scatter sub-χ², so the "halo" of cells just inside/outside the contour is wasted work for meta-mesh aggregation. (PROsurf surface-amr keeps its 0.5 default — different use case.)
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

        // ---- PE-bank generation knobs ----
        // Per-cell PE generation follows an additive doubling rule:
        //   to_add = n_pe_min * 2^max(0, cell.level - update_layer)
        // Each run ADDS this many PEs to the cell on top of whatever's already
        // there (existing PEs always preserved). Cells with level < update_layer
        // are not touched. n_pe_max is a total-per-cell safety cap — no cell
        // ever exceeds it, even across many runs.
        std::vector<float> cl_targets = {0.683f, 0.90f, 0.954f};
        int   n_pe_min = 50;     ///< PEs added per run at level == update_layer; doubles for each deeper level.
        int   n_pe_max = 5000;   ///< Hard total cap per cell (cumulative across runs).
        int   update_layer = 0;  ///< Cells below this AMR level are skipped entirely.
        float wilson_eps = 0.05f; ///< Unused for bank generation now; kept for slice 2c classification.
        float roi_band = 8.0f;   ///< (slice 2c, unused for now).
        int   n_brazil_throws = 100; ///< Number of pseudo-experiment throws for --mode brazil.
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
        static constexpr uint32_t MAGIC = 0x41464D45;  ///< 'AFME' (Adaptive FC MEsh).
        static constexpr uint16_t VERSION = 1;

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
     * @brief Persist a MetaMesh to disk via Boost binary archive.
     *
     * Sibling artifact to the PEBank — meant as a dev-time cache so the
     * prepass+meta-mesh build doesn't have to re-run when iterating on PE-bank
     * parameters. Same magic+version layout as save_bank.
     */
    bool save_mesh(const MetaMesh &mm, const std::string &path);

    /// Inverse of save_mesh. Returns false (and leaves `mm_out` indeterminate)
    /// on missing file, bad magic, or version mismatch.
    bool load_mesh(MetaMesh &mm_out, const std::string &path);

    /**
     * @brief Wilson-interval sequential stopping rule. Pure utility — no PROfit deps.
     *
     * Slice-2a v1 rule: stop when the Wilson 95% half-width on the Bernoulli
     * proportion estimate falls below `eps`. The spec-faithful rule (Wilson CI
     * vs target alpha with Bonferroni-corrected milestones) is deferred to
     * slice 2b's validation pass.
     *
     * All methods are constexpr-free but inlined so the helper has no link
     * dependency on the .cxx (useful for unit-testing).
     */
    struct SequentialFCTest {
        float z = 1.959963984540054f; ///< Wilson z-score; 1.96 for 95% nominal coverage.

        /**
         * @brief Wilson score interval on a Bernoulli proportion p = k/n.
         * @return (lo, hi) bounds with nominal coverage 1 - alpha controlled by `z`.
         */
        std::pair<float, float> wilson_interval(int n, int k) const {
            if (n <= 0) return {0.0f, 1.0f};
            const float nf = (float)n;
            const float p = (float)k / nf;
            const float z2 = z * z;
            const float denom = 1.0f + z2 / nf;
            const float center = (p + z2 / (2.0f * nf)) / denom;
            const float radical = std::sqrt(std::max(0.0f, p * (1.0f - p) / nf + z2 / (4.0f * nf * nf)));
            const float half = (z * radical) / denom;
            return {center - half, center + half};
        }

        /**
         * @brief Half-width of the Wilson interval (handy for the v1 stop rule).
         */
        float wilson_halfwidth(int n, int k) const {
            const auto [lo, hi] = wilson_interval(n, k);
            return 0.5f * (hi - lo);
        }

        /**
         * @brief Slice-2a stop rule: stop when half-width below `eps`.
         *
         * `k` is the count of PEs with Δχ²(syst) − Δχ²(syst+osc) ≥ Δχ²_obs(μ).
         * `n` is the total PEs thrown at this cell so far.
         */
        bool should_stop(int n, int k, float eps) const {
            if (n <= 0) return false;
            return wilson_halfwidth(n, k) < eps;
        }

        /**
         * @brief Empirical Δχ² critical value at a given confidence level.
         *
         * Returns the `cl`-quantile of the supplied (sorted) sample. Used by
         * the classify path to convert a per-cell PE distribution into a
         * critical Δχ²_c against which the observed Δχ² is compared.
         *
         * @param sorted_samples  Δχ² values in ascending order.
         * @param cl              Target confidence level in (0, 1).
         */
        static float crit_dchi2_at_cl(const std::vector<float> &sorted_samples, float cl) {
            if (sorted_samples.empty()) return 0.0f;
            const float p = std::min(1.0f, std::max(0.0f, cl));
            // Linear-interpolated quantile (type-7 in R's parlance).
            const float pos = p * ((float)sorted_samples.size() - 1.0f);
            const int   lo  = (int)std::floor(pos);
            const int   hi  = (int)std::ceil(pos);
            if (lo == hi) return sorted_samples[lo];
            const float frac = pos - (float)lo;
            return (1.0f - frac) * sorted_samples[lo] + frac * sorted_samples[hi];
        }
    };

    /**
     * @brief One pseudo-experiment record in the PE bank.
     *
     * Mirrors fc_out (inc/PROfc.h:34) but keeps only the fields slice-2a needs
     * to drive classification (the syst-only and syst+osc chi^2s and their
     * difference). Best-fit vectors are dropped to keep the bank small; they
     * can be re-derived if ever needed by re-running the throw with the same seed.
     */
    struct PEBankRecord {
        float    chi2_syst = 0.0f; ///< Best-fit chi^2 from the syst-only minimisation.
        float    chi2_osc  = 0.0f; ///< Best-fit chi^2 from the full (syst + osc) minimisation.
        float    dchi2     = 0.0f; ///< chi2_syst - chi2_osc.
        uint32_t seed      = 0;    ///< RNG seed used to generate this PE (for reproducibility).
    };

    /**
     * @brief Pseudo-experiment bank attached to a meta-mesh.
     *
     * One slot per MetaCell (Option A: PE bank at each cell center). Cell index
     * is the position of the MetaCell in `MetaMesh::cells`. Each slot holds the
     * raw Δχ² outcomes from `n_pes_at_cell[c]` PEs thrown at that cell.
     *
     * The bank is the pre-unblinding artifact: classification at any threshold
     * is a sort+search over the per-cell vector, so the *same* bank serves the
     * Asimov sensitivity, Brazil-band, and real-data analyses without
     * regenerating PEs.
     */
    struct PEBank {
        static constexpr uint32_t MAGIC = 0x41464342;  ///< 'AFCB' (Adaptive FC Bank).
        static constexpr uint16_t VERSION = 2;
        ///< v2: added cell footprint arrays (i_bl, j_bl, step, level) so the
        ///<     bank is self-describing without needing the MetaMesh sidecar.

        // Mesh footprint the bank was generated against. Used as a provenance
        // check at load time so we can refuse a (bank, mesh) mismatch.
        int   finest_nx = 0;
        int   finest_ny = 0;
        int   max_levels = 0;
        float x_lo = 0.0f, x_hi = 0.0f;
        float y_lo = 0.0f, y_hi = 0.0f;
        int   n_cells = 0;

        // Per-cell PE storage. Outer index = cell index into MetaMesh::cells;
        // inner vector = PEs thrown at that cell in generation order.
        std::vector<std::vector<PEBankRecord>> cell_pes;

        // Per-cell coordinates in *model space* (log10 of physical for
        // log-axis params; linear otherwise). Slice-2b classify code applies
        // pow(10) when mapping back to physical for plotting.
        std::vector<float> cell_center_x;
        std::vector<float> cell_center_y;

        // Per-cell footprint copied from MetaMesh::cells. Lets the bank be
        // plotted / inspected without re-loading the mesh sidecar.
        std::vector<int> cell_i_bl;
        std::vector<int> cell_j_bl;
        std::vector<int> cell_step;
        std::vector<int> cell_level;
    };

    /**
     * @brief Persist a PEBank to disk via Boost binary archive.
     *
     * On-disk layout: magic + version + the serialised PEBank. Mirrors the
     * pattern in src/PROcreate.cxx:25-69.
     * @return true on success, false on any I/O or serialisation error.
     */
    bool save_bank(const PEBank &bank, const std::string &path);

    /**
     * @brief Inverse of save_bank. Verifies magic + version on the way in.
     * @return true on success and `bank_out` populated; false otherwise (and
     *         `bank_out` left in an indeterminate state).
     */
    bool load_bank(PEBank &bank_out, const std::string &path);

    /**
     * @brief Output bundle from run_adaptive_fc.
     */
    struct AdaptiveFCResult {
        // Slice 1 fields.
        int  n_throws_done   = 0;
        int  n_meta_cells    = 0;
        int  n_baseline_cells = 0;
        int  n_refined_cells = 0;
        std::vector<int> leaves_per_throw;
        std::string diag_root_path;

        // Slice 2a fields (populated by --mode init-bank).
        std::string bank_path;        ///< Path of the saved PEBank artifact, if any.
        int64_t     total_pes_generated = 0;
        int         cells_hit_n_pe_max  = 0; ///< Number of cells that reached n_pe_max without Wilson-stopping.
        float       mean_pes_per_cell   = 0.0f;
    };

    /**
     * @brief Adaptive FC entry point.
     *
     * @param data  The data spectrum PROfit assembled (real data OR injected fake OR
     *              `--poisson-throw`'d OR `--pseudo-experiment`'d). Asimov / classify
     *              modes operate on this; build-mesh / init-bank don't read it.
     *              Without this we'd silently ignore the user's `--use-fake-data`
     *              choices in asimov mode.
     */
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
        MultiPROgressBar &progress);

} // namespace PROfit

#endif
