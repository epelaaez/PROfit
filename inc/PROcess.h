#ifndef PROCESS_H
#define PROCESS_H

#include <Eigen/Eigen>
#include <cstdint>

// PROfit include 
#include "PROconfig.h"
#include "PROmodel.h"
#include "PROpeller.h"
#include "PROspec.h"
#include "PROsyst.h"

#include "TH2D.h"
#include <vector>

namespace PROfit{

    /* Function: 
     *  The master weighting function that combines all weights and fills into spectrum PROspec, event-by-event
     */

    PROspec FillCVSpectrum(const PROconfig &inconfig, const PROpeller &inprop, bool binned = false);
    PROspec FillOtherCVSpectrum(const PROconfig &inconfig, const PROpeller &inprop, size_t other_index);


  //Added overloading for FillRecoSpectra
  PROspec FillRecoSpectra(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst, const PROmodel &inmodel, const Eigen::VectorXf &params, const std::vector<TH2D*> inweighthists = {}, bool binned = true);
  PROspec FillRecoSpectra(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst, const PROmodel &inmodel, const Eigen::VectorXf &params, bool binned = true);  
  PROspec FillOtherRecoSpectra(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst, const PROmodel &inmodel, const Eigen::VectorXf &params, size_t other_index, const std::vector<TH2D*> inweighthists = {});
    PROspec FillSystRandomThrow(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst, uint32_t seed, int other_index = -1);
    PROspec FillSplineRandomThrow(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst, int spline, uint32_t seed, int other_index = -1);
};

#endif
