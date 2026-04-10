/**
 * @file PROcess.h
 * @brief Spectrum-filling functions combining MC events, systematics, and physics models.
 * @author PROfit Collaboration
 *
 * @details Declares the family of FillSpectra functions that are the primary entry points
 * for computing a predicted event spectrum from Monte Carlo events stored in a PROpeller,
 * applying oscillation weights from a PROmodel, and applying systematic spline weights
 * from a PROsyst.  Both event-by-event and pre-binned modes are supported.
 */
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

    /**
     * @brief Master spectrum-filling function combining oscillation weights and systematic spline weights.
     * @details Iterates over MC events (or uses pre-binned histograms when @p binned is true),
     * computes oscillation probabilities via @p inmodel, applies spline shifts from @p insyst,
     * and accumulates a predicted spectrum for the analysis variable indexed by @p var_index.
     * @param inconfig   Configuration object describing binning, channels, and subchannels.
     * @param inprop     MC event store (PROpeller).
     * @param insyst     Systematic object holding spline and covariance information.
     * @param inmodel    Physics model providing oscillation probability weights.
     * @param params     Combined parameter vector: physics parameters followed by spline nuisance parameters.
     * @param binned     If true (default), use pre-binned mode (fast); if false, iterate event-by-event.
     * @param var_index  Index of the analysis variable to fill (default 0, i.e. the primary reco variable).
     * @return A PROspec containing the predicted event spectrum.
     */
    PROspec FillSpectra(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst, const PROmodel &inmodel, const Eigen::VectorXf &params, bool binned = true, size_t var_index =0);

    /**
     * @brief Overload of FillSpectra accepting systematic pulls as a name-to-value map.
     * @details Converts the named pull map to an ordered parameter vector and delegates to the
     * primary FillSpectra overload.
     * @param inconfig   Configuration object.
     * @param inprop     MC event store.
     * @param insyst     Systematic object.
     * @param inmodel    Physics model.
     * @param pulls      Map from spline systematic name to pull value (in units of sigma).
     * @param binned     If true (default), use pre-binned mode.
     * @param var_index  Index of the analysis variable to fill (default 0).
     * @return A PROspec containing the predicted event spectrum.
     */
    PROspec FillSpectra(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst, const PROmodel &inmodel, const std::map<std::string, float> &pulls, bool binned =true, size_t var_index=0);


    //Below are depreciated, slightly

    /**
     * @brief Fill a spectrum using per-event oscillation weights derived from 2D input histograms.
     * @details Deprecated in favour of the standard FillSpectra path.  Applies weights read from
     * @p inweighthists rather than computing them analytically.
     * @param inconfig      Configuration object.
     * @param inprop        MC event store.
     * @param inweighthists Vector of TH2D weight histograms, one per probability type.
     * @param inmodel       Physics model (used for event classification only).
     * @param params        Physics parameter vector.
     * @param binned        If false (default for this overload), iterate event-by-event.
     * @return A PROspec with weights applied from the input histograms.
     */
    PROspec FillWeightedSpectrumFromHist(const PROconfig &inconfig, const PROpeller &inprop, std::vector<TH2D*> inweighthists, const PROmodel &inmodel, const Eigen::VectorXf &params, bool binned = false);

    /**
     * @brief Generate a spectrum with a single random systematic throw applied.
     * @details Draws a random Gaussian universe for all systematics and fills a spectrum for
     * use in covariance-matrix estimation or toy-MC studies.
     * @param inconfig   Configuration object.
     * @param inprop     MC event store.
     * @param insyst     Systematic object.
     * @param model      Physics model.
     * @param cvspec     Central-value spectrum (used as denominator for fractional shifts).
     * @param cvparams   Central-value physics parameter vector.
     * @param seed       Random seed.
     * @param var_index  Variable index to fill (default 0).
     * @return A PROspec with one random systematic throw applied.
     */
    PROspec FillSystRandomThrow(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst, const PROmodel &model, const PROspec &cvspec, const Eigen::VectorXf &cvparams, uint32_t seed, int var_index=0);

    /**
     * @brief Generate a spectrum with a single named spline systematic randomly shifted.
     * @details Throws that spline's shift from a Gaussian distribution and fills the spectrum.
     * @param inconfig    Configuration object.
     * @param inprop      MC event store.
     * @param insyst      Systematic object.
     * @param model       Physics model.
     * @param cvparams    Central-value physics parameter vector.
     * @param spline      0-based index of the spline systematic to vary.
     * @param seed        Random seed.
     * @param other_index Variable index (default 0).
     * @return A PROspec with the specified spline thrown.
     */
    PROspec FillSplineRandomThrow(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst,  const PROmodel &model,  const Eigen::VectorXf &cvparams, int spline, uint32_t seed, int other_index=0);

};

#endif
