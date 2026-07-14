/**
 * @file PROmeshEval.cxx
 * @brief Implementation of the shared AMR pinned-scan fit body.
 * @author PROfit Collaboration
 *
 * @details Extracted verbatim from the formerly duplicated EvalFn lambda
 * bodies in PROsurf::FillSurfaceAMR and the adaptive-FC Wilks prepass.
 */
#include "PROmeshEval.h"

namespace PROfit {
namespace PROmesh {

EvalResult pinned_scan_eval(PROmetric &metric,
                            const PROfitterConfig &fitconfig,
                            size_t x_idx, size_t y_idx,
                            const EvalRequest &req)
{
    PROmetric *m = &metric;
    m->reset();

    const int nphys    = (int)m->GetModel().nparams;
    const int nspline  = (int)m->GetSysts().GetNSplines();
    const int n_full   = nphys + nspline;
    Eigen::VectorXf lb(n_full), ub(n_full);
    lb << m->GetModel().lb,
          Eigen::VectorXf::Map(m->GetSysts().spline_lo.data(), m->GetSysts().spline_lo.size());
    ub << m->GetModel().ub,
          Eigen::VectorXf::Map(m->GetSysts().spline_hi.data(), m->GetSysts().spline_hi.size());

    // Pin the two scanned coordinates; the remaining n_full-2 parameters
    // are optimised by PROfitter.
    lb((int)x_idx) = req.x_phys;
    ub((int)x_idx) = req.x_phys;
    lb((int)y_idx) = req.y_phys;
    ub((int)y_idx) = req.y_phys;
    m->setBounds(lb, ub);

    EvalResult out;

    // Zero free parameters (covariance-only systematics, both physics axes
    // pinned): there is nothing to minimise — the profiled chi2 IS the chi2
    // at the pinned point. One metric evaluation instead of the full
    // multistart (which would burn ~n_latin_points evaluations of the
    // identical point and LBFGS attempts that can only fail). This also
    // covers the adaptive-FC Wilks prepass, which shares this evaluator.
    if (n_full == 2) {
        Eigen::VectorXf params(n_full);
        params((int)x_idx) = req.x_phys;
        params((int)y_idx) = req.y_phys;
        Eigen::VectorXf grad = Eigen::VectorXf::Zero(n_full);
        out.chi2 = (*m)(params, grad, false);
        out.best_fit = params;
        return out;
    }

    // Reproducible per-key seeding.
    const uint32_t fseed = static_cast<uint32_t>(req.key & 0xffffffffu);
    PROfitter fitter(ub, lb, fitconfig, fseed);

    if (req.seeds.empty()) {
        out.chi2 = fitter.Fit(*m);
    } else {
        out.chi2 = fitter.Fit(*m, req.seeds);
    }
    out.best_fit = fitter.best_fit;
    return out;
}

}  // namespace PROmesh
}  // namespace PROfit
