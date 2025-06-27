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

namespace PROfit{

    /* Function: 
     *  The master weighting function that combines all weights and fills into spectrum PROspec, event-by-event or binned, for any of the variables in the XML
     *  There is one for CV and one for model-applied or splines, but could be reduced to just the "Reco"
     */
    PROspec FillCVSpectra(const PROconfig &inconfig, const PROpeller &inprop, bool binned = true, size_t var_index = 0);
    PROspec FillSpectra(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst, const PROmodel &inmodel, const Eigen::VectorXf &params, bool binned = true, size_t var_index =0);
    PROspec FillSpectra(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst, const PROmodel &inmodel, const std::map<std::string, float> &pulls, bool binned =true, size_t var_index=0);


    //Below are depreciated, slightly
  //ETW 1/22/2025 Add function to fill spectrum using weights from input histogram
    PROspec FillWeightedSpectrumFromHist(const PROconfig &inconfig, const PROpeller &inprop, std::vector<TH2D*> inweighthists, const PROmodel &inmodel, const Eigen::VectorXf &params, bool binned = false);
    PROspec FillSystRandomThrow(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst, uint32_t seed, int other_index = 0);
    PROspec FillSplineRandomThrow(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst, int spline, uint32_t seed, int other_index = 0);

};

#endif
