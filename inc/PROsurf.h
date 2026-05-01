/**
 * @file PROsurf.h
 * @brief Chi-squared surface, profile likelihood, and sensitivity curve computation.
 * @author PROfit Collaboration
 *
 * @details Defines PROsurf (2D chi-squared surface scanning) and PROfile (1D profile
 * likelihood scanning) for producing exclusion contours and confidence regions.
 * Both classes take a PROmetric and scan physics parameter combinations, running the
 * full PROfitter pipeline at each grid point in parallel threads.
 *
 * Also defines the output structs surfOut and profOut that carry per-grid-point results.
 */
#ifndef PROSURF_H
#define PROSURF_H

#include "PROfitter.h"
#include "PROconfig.h"
#include "PROsyst.h"
#include "PROseed.h"
#include "PROmetric.h"
#include "PROgress.h"
#include "PROversion.h"

#include <Eigen/Eigen>

#include "TGraphAsymmErrors.h"
#include "TMarker.h"
#include "TMultiGraph.h"
#include "TArrow.h"

namespace PROfit {

    /**
     * @brief Output record for a single grid point in a 2D chi-squared surface scan.
     */
    struct surfOut{
        std::vector<int> grid_index;   ///< Indices [ix, iy] of this grid point in the surface matrix.
        std::vector<float> grid_val;   ///< Parameter values [x, y] at this grid point.
        Eigen::VectorXf best_fit;      ///< Best-fit full parameter vector (physics + splines) at this point.
        float chi;                     ///< Minimum chi-squared found at this physics point.
    };

    /**
     * @brief Output record for a 1D profile likelihood scan.
     */
    struct profOut{
        std::vector<float> knob_vals; ///< Parameter values at each scan point.
        std::vector<float> knob_chis; ///< Profile chi-squared at each scan point.
        std::vector<Eigen::VectorXf> knob_bfs; ///< Best-fit parameter vectors at each scan point.
        float chi; ///< Global minimum chi-squared found during the profile scan.

        void sort(){
            std::vector<size_t> indices(knob_vals.size());
            std::iota(indices.begin(), indices.end(), 0);

            // Sort indices based on knob_vals
            std::sort(indices.begin(), indices.end(),
                    [this](size_t i, size_t j) { return knob_vals[i] < knob_vals[j]; });

            std::vector<float> sorted_vals(knob_vals.size());
            std::vector<float> sorted_chis(knob_chis.size());
            std::vector<Eigen::VectorXf> sorted_bfs(knob_bfs.size());

            for(size_t i = 0; i < indices.size(); ++i) {
                sorted_vals[i] = knob_vals[indices[i]];
                sorted_chis[i] = knob_chis[indices[i]];
                sorted_bfs[i] = knob_bfs[indices[i]];
            }

            knob_vals = std::move(sorted_vals);
            knob_chis = std::move(sorted_chis);
            knob_bfs = std::move(sorted_bfs);
        }
    };

    /**
     * @brief 1D profile likelihood scanner producing exclusion bands for individual parameters.
     * @details Scans one physics parameter at a time while minimising over all others using
     * PROfitter.  Produces 1-sigma error bands and stores the result in ROOT TGraph objects.
     * Supports optional oscillation parameter inclusion and multi-threaded point evaluation.
     */
    class PROfile {

        public:
            PROmetric &metric;                          ///< Reference to the chi-squared metric being scanned.
            TGraphAsymmErrors onesig;                   ///< ROOT graph of the 1-sigma asymmetric error band.
            std::vector<std::unique_ptr<TGraph>> graphs; ///< Per-systematic profile chi-squared graphs.
            std::vector<float> bfvalues;   ///< Best-fit parameter values at each scan point.
            std::vector<float> barvalues;  ///< Bar (central) values at each scan point.
            std::vector<float> values1_up;   ///< Upper 1-sigma boundary values.
            std::vector<float> values1_down; ///< Lower 1-sigma boundary values.

            float newglob;                  ///< Updated global minimum chi-squared found during the scan.
            Eigen::VectorXf newglob_param;  ///< Full parameter vector at the updated global minimum.

            PROfile(const PROconfig &config, const PROsyst &systs, const PROmodel &model, PROmetric &metric, PROseed &proseed, const PROfitterConfig &fitconfig, std::string filename, float minchi = 0, bool with_osc = false, int nThreads = 1, const std::vector<Eigen::VectorXf> &seed_points = {}, const Eigen::VectorXf& true_params = Eigen::VectorXf() ) ;

            void Plot(const PROconfig &config, const PROsyst &systs, const PROmodel &model, PROmetric &metric, PROseed &proseed, std::string filename, bool with_osc = false, const Eigen::VectorXf& init_seed = Eigen::VectorXf(), const Eigen::VectorXf& true_params = Eigen::VectorXf(), const Eigen::MatrixXf& spline_covariance = Eigen::MatrixXf{}, const Eigen::VectorXf& param_err_lo = Eigen::VectorXf{}, const Eigen::VectorXf& param_err_hi = Eigen::VectorXf{}, bool mask_osc = false) ;

            std::vector<profOut> PROfilePointHelper(const PROsyst *systs, const PROfitterConfig &fitconfig, int offset, int stride, float minchi, bool with_osc, MultiPROgressBar& progressbar, const std::vector<Eigen::VectorXf> &seed_points = {}, uint32_t seed=0);
    };

    /**
     * @brief 2D chi-squared surface scanner for two-parameter exclusion contours.
     * @details Evaluates the profile chi-squared on a 2D grid of (x, y) physics parameter
     * values, minimising over all other parameters at each grid point via PROfitter.
     * Results are stored in the surface matrix for subsequent contour plotting.
     * Supports linear and logarithmic axis spacing.
     */
    class PROsurf {
        public:
            PROmetric &metric;     ///< Reference to the chi-squared metric being evaluated.
            size_t x_idx;          ///< Index of the x-axis physics parameter.
            size_t y_idx;          ///< Index of the y-axis physics parameter.
            size_t nbinsx;         ///< Number of grid points along the x axis.
            size_t nbinsy;         ///< Number of grid points along the y axis.
            Eigen::VectorXf edges_x; ///< Grid edges along the x axis (length nbinsx+1).
            Eigen::VectorXf edges_y; ///< Grid edges along the y axis (length nbinsy+1).
            Eigen::MatrixXf surface; ///< Profile chi-squared matrix (nbinsx × nbinsy).

            /**
             * @brief Per-grid-point result record.
             */
            struct SurfPointResult {
                int binx;              ///< x grid index.
                int biny;              ///< y grid index.
                Eigen::VectorXf best_fit; ///< Full best-fit parameter vector at this point.
                float chi2;            ///< Profile chi-squared at this point.
            };

            std::vector<SurfPointResult> results; ///< All grid-point results (filled by FillSurface).

            /**
             * @brief Axis spacing mode for surface grid construction.
             */
            enum LogLin {
                LinAxis, ///< Uniform linear spacing.
                LogAxis, ///< Uniform logarithmic spacing.
            };

            PROsurf(PROmetric &metric,  size_t x_idx, size_t y_idx, size_t nbinsx, const Eigen::VectorXf &edges_x, size_t nbinsy, const Eigen::VectorXf &edges_y) : metric(metric), x_idx(x_idx), y_idx(y_idx), nbinsx(nbinsx), nbinsy(nbinsy), edges_x(edges_x), edges_y(edges_y), surface(nbinsx, nbinsy) { }

            PROsurf(PROmetric &metric, size_t x_idx, size_t y_idx, size_t nbinsx, LogLin llx, float x_lo, float x_hi, size_t nbinsy, LogLin lly, float y_lo, float y_hi);

            std::vector<surfOut> PointHelper(const PROfitterConfig &fitconfig, std::vector<surfOut> multi_physics_params, int start, int end, uint32_t seed);

            void FillSurfaceStat(const PROconfig &config, const PROfitterConfig &fitconfig, std::string filename, const Eigen::VectorXf &cv_params, uint32_t seed);
            void FillSurface(const PROfitterConfig &fitconfig, std::string filename, PROseed & proseed, int nthreads = 1);
            std::vector<surfOut> FillCurve(const PROfitterConfig &fitconfig, PROseed &proseed, int nThreads, std::vector<float> &A, std::vector<float> &B, size_t n_points);
            void PlotCurve(const PROconfig &config, const PROmodel &model, const PROsyst &syst, const std::vector<surfOut> & cpoints, std::string final_output_tag, bool logx, bool logy,size_t xaxis_idx,size_t yaxis_idx,std::vector<float> &A, std::vector<float> &B, size_t n_points);

    };

}

#endif

