#ifndef PROMETRIC_H
#define PROMETRIC_H

#include "PROsyst.h"
#include "PROmodel.h"

#include <Eigen/Eigen>

namespace PROfit {

class PROmetric {
public:
    enum EvalStrategy {
        EventByEvent,
        BinnedGrad,
        BinnedChi2
    };

    virtual void override_systs(const PROsyst &new_syst) = 0;
    virtual float operator()(const Eigen::VectorXf &param, Eigen::VectorXf &gradient) = 0;
    virtual float operator()(const Eigen::VectorXf &param, Eigen::VectorXf &gradient, bool nograd) = 0;
    virtual void reset() = 0;
    virtual PROmetric *Clone() const = 0;
    virtual const PROmodel &GetModel() const = 0;
    virtual const PROsyst  &GetSysts() const = 0;
    virtual float getSingleChannelChi(size_t channel_index) = 0;
    virtual ~PROmetric() {}
    virtual void fixSpline(int,float)  = 0;
    virtual float Pull(const Eigen::VectorXf &systs) = 0;

    size_t nParams() const {return GetModel().nparams + GetSysts().GetNSplines();}

    Eigen::VectorXf LowerBound() const {
      size_t nphys = GetModel().nparams;
      size_t nparams = nParams();
      Eigen::VectorXf lb = Eigen::VectorXf::Constant(nparams, -3.0);
      for (size_t i = 0; i < nphys; ++i) {
        lb(i) = GetModel().lb(i);
      }
      for(size_t i = nphys; i < nparams; ++i) {
        lb(i) = GetSysts().spline_lo[i-nphys];
      }
      return lb;
    }

    Eigen::VectorXf UpperBound() const {
      size_t nphys = GetModel().nparams;
      size_t nparams = nParams();
      Eigen::VectorXf ub = Eigen::VectorXf::Constant(nparams, 3.0);
      for (size_t i = 0; i < nphys; ++i) {
        ub(i) = GetModel().ub(i);
      }
      for(size_t i = nphys; i < nparams; ++i) {
        ub(i) = GetSysts().spline_hi[i-nphys];
      }
      return ub;
    }

};

};

#endif

