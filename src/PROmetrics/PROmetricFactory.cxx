#include "PROmetrics/PROmetricFactory.h"
#include "PROmetrics/PROchi_neyman.h"
#include "PROmetrics/PROchi_pearson.h"
#include "PROmetrics/PROchi_CNP.h"
#include "PROmetrics/PROpoisson.h"
#include "PROlog.h"

namespace PROfit {

    std::unique_ptr<PROmetric> MakeMetric(const std::string &chi2_kind,
                                          const PROconfig &config, const PROpeller &prop,
                                          const PROsyst *systs, const PROmodel &model,
                                          const PROdata &data, PROmetric::EvalStrategy strat,
                                          bool shape_only,
                                          std::vector<float> physics_param_fixed) {
        // Defensive canonicalization: idempotent for PROfit-binary callers (main already
        // canonicalized), protects direct library callers.
        const std::string kind = PROmetric::canonicalizeMetricName(chi2_kind);
        if(kind == "neyman")
            return std::unique_ptr<PROmetric>(new PROchi("", config, prop, systs, model, data, strat, shape_only, physics_param_fixed));
        if(kind == "pearson")
            return std::unique_ptr<PROmetric>(new PROchi_pearson("", config, prop, systs, model, data, strat, shape_only, physics_param_fixed));
        if(kind == "CNP")
            return std::unique_ptr<PROmetric>(new PROCNP("", config, prop, systs, model, data, strat, shape_only, physics_param_fixed));
        if(kind == "poisson")
            return std::unique_ptr<PROmetric>(new PROpoisson("", config, prop, systs, model, data, strat, shape_only, physics_param_fixed));
        log<LOG_ERROR>(L"%1% || Unrecognized chi2 function %2%. Options: neyman (default), pearson, CNP, poisson (legacy aliases: PROchi, PROCNP, Poisson).") % __func__ % chi2_kind.c_str();
        return nullptr;
    }

}
