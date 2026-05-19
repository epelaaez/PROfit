#include "PROconfig.h"
#include "PROdata.h"
#include "PROlog.h"
#include "PROmetric.h"
#include "PROspec.h"
#include "PROsyst.h"
#include "PROcreate.h"
#include "PROpeller.h"
#include "PROchi.h"
#include "PROCNP.h"
#include "PROpoisson.h"
#include "PROcess.h"
#include "PROsurf.h"
#include "PROfc.h"
#include "PROfitter.h"
#include "PROmodel.h"
#include "PROMCMC.h"
#include "PROtocall.h"
#include "PROseed.h"
#include "PROversion.h"
#include "PROplot.h"
#include "PRObench.h"

#include "CLI11.h"
#include "LBFGSB.h"

#include <Eigen/Eigen>

#include <Eigen/src/Core/Matrix.h>
#include <LBFGSpp/Param.h>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <future>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <set>
#include <vector>
#include <chrono>
#include "TMath.h"

using namespace PROfit;

// Unique key for DetVar propeller maps (names can be reused across sections).
static std::string DetVarKey(const PROconfig& config, size_t file_index) {
    const auto& dv = config.m_detvar_files[file_index];
    return "sec" + std::to_string(dv.section_index) + "::" + dv.name + "." + std::to_string(dv.knobval);
}

// Build a collision-free composite key for event i_event from its matching_var_values.
// Returns a vector of integer-cast values, one per matching variable (e.g. run, subrun, event).
static std::vector<int> DetVarMatchingKey(const PROpeller& prop, size_t i_event) {
    std::vector<int> key;
    key.reserve(prop.matching_var_values.size());
    for(const auto& vals : prop.matching_var_values)
        key.push_back(static_cast<int>(std::round(vals[i_event])));
    return key;
}

// Build PROspec objects for CV and variation using only events whose matching keys appear
// in both propellers. var_idx selects which variable's bin indices to use.
// Returns false (leaving out_cv/out_var unchanged) if either propeller lacks matching vars.
static bool BuildDetVarMatchedSpecs(
        const PROpeller& cvprop, const std::map<int, const PROpeller*> &varprop,
        int var_idx, int spec_size,
        PROspec& out_cv, std::map<int, PROspec> &out_var) {

    if(!cvprop.has_matching_vars || 
            !std::all_of(varprop.begin(), varprop.end(), [](const auto &p){ return p.second->has_matching_vars;})) 
        return false;
    if(!std::all_of(varprop.begin(), varprop.end(), 
                [&cvprop](const auto &p){ 
                    return cvprop.matching_var_values.size() == p.second->matching_var_values.size(); 
                }))
        return false;

    // Step 1: build lookup key -> list of event indices for CV.
    // Use std::map with vector<int> keys for guaranteed collision-free RSE matching.
    std::map<std::vector<int>, std::vector<size_t>> cv_key_map;
    for(size_t i = 0; i < cvprop.NEvent(); ++i)
        cv_key_map[DetVarMatchingKey(cvprop, i)].push_back(i);

    // Step 2: find the set of keys present in both CV and variation;
    // track unique var keys to compute CV-only / var-only / overlapping counts.
    std::map<int, std::set<std::vector<int>>> var_key_set;
    std::set<std::vector<int>> common_keys;
    for(const auto &[kv, prop] : varprop) {
        for(size_t j = 0; j < prop->NEvent(); ++j) {
            auto k = DetVarMatchingKey(*prop, j);
            var_key_set[kv].insert(k);
        }
    }
    for(const auto &cv_key : cv_key_map) {
        if(std::all_of(var_key_set.begin(), var_key_set.end(),
                    [&cv_key](const auto &s) {
                        return s.second.count(cv_key.first) > 0;
                    }))
            common_keys.insert(cv_key.first);
    }

    const size_t n_cv_unique    = cv_key_map.size();
    const size_t n_overlapping  = common_keys.size();
    const size_t n_cv_only      = n_cv_unique - n_overlapping;
    // Count total unique keys across all variations
    std::set<std::vector<int>> all_var_keys;
    for(const auto &[kv, ks] : var_key_set) all_var_keys.insert(ks.begin(), ks.end());
    const size_t n_var_unique   = all_var_keys.size();
    const size_t n_var_only     = n_var_unique > n_overlapping ? n_var_unique - n_overlapping : 0;
    log<LOG_INFO>(L"DetVar matching: total CV unique keys: %1%, total var unique keys: %2%")
        % n_cv_unique % n_var_unique;
    log<LOG_INFO>(L"DetVar matching: num CV only: %1%, num var only: %2%, num overlapping: %3%")
        % n_cv_only % n_var_only % n_overlapping;

    // Step 3: fill matched CV spec
    PROspec matched_cv(spec_size);
    size_t n_cv_prop_matched = 0;
    for(size_t i = 0; i < cvprop.NEvent(); ++i) {
        if(!common_keys.count(DetVarMatchingKey(cvprop, i))) continue;
        ++n_cv_prop_matched;
        int bin = cvprop.variable_bin_indices[var_idx][i];
        if(bin >= 0) matched_cv.QuickFill(bin, cvprop.added_weights[i]);
    }

    // Step 4: fill matched var spec
    std::map<int, PROspec> matched_var;
    Eigen::VectorXf n_var_prop_matched = Eigen::VectorXf::Zero(varprop.size());
    Eigen::VectorXf n_var_evt = Eigen::VectorXf::Zero(varprop.size());
    size_t prop_i = 0;
    for(const auto &[kv, prop] : varprop) {
        matched_var[kv] = PROspec(spec_size);
        n_var_evt(prop_i) = prop->NEvent();
        for(size_t j = 0; j < prop->NEvent(); ++j) {
            if(!common_keys.count(DetVarMatchingKey(*prop, j))) continue;
            n_var_prop_matched(prop_i) += 1;
            int bin = prop->variable_bin_indices[var_idx][j];
            if(bin >= 0) matched_var[kv].QuickFill(bin, prop->added_weights[j]);
        }
        prop_i++;
    }
    log<LOG_INFO>(L"DetVar matching: matched propeller events CV: %1%, var: %2% (total propeller events CV: %3%, var: %4%)")
        % n_cv_prop_matched % n_var_prop_matched % cvprop.NEvent() % n_var_evt;


    out_cv  = std::move(matched_cv);
    out_var = std::move(matched_var);
    return true;
}

log_level_t GLOBAL_LEVEL = LOG_INFO;
log_level_t FILE_LEVEL = LOG_INFO;

std::wostream *OSTREAM = &wcout;

std::wofstream LOG_FILE_STREAM;
bool LOGGING_TO_FILE = false;

void mcmc_worker(std::vector<Metropolis<simple_target, adaptive_proposal>> &mets, Eigen::VectorXf initial, PROmetric *metric, uint32_t seed, size_t nchains, size_t burnin, size_t steps);

struct GlobalFitResult {
    PROfitter fitter;
    std::optional<Metropolis<simple_target, adaptive_proposal>> mh;

    std::vector<TH1D> priors, posteriors;
    Eigen::VectorXf prior_param_lo, prior_param_hi, post_param_lo, post_param_hi;
    Eigen::MatrixXf covmat, fraccovmat, corrmat, prior_covariance, spline_covariance;

    std::optional<PROerrorbar> err_band, post_err_band;

    float chi2;

    GlobalFitResult(const Eigen::VectorXf &ub, const Eigen::VectorXf &lb, const PROfitterConfig &config)
        : fitter(ub, lb, config) {}
};
enum struct GlobalFitOptions {
    Default             = 0,
    Progress            = 1 << 0,
    FreqSeedPts         = 1 << 1,
    Correlations        = 1 << 2,
    PrefitErrorBand     = 1 << 3,
    MCMCPrefitErrorBand = 1 << 4,
    PostFitErrorBand    = 1 << 5,
    BinWidthScaled      = 1 << 6,
};
GlobalFitOptions operator|(GlobalFitOptions lhs, GlobalFitOptions rhs) { 
    return static_cast<GlobalFitOptions>(static_cast<int>(lhs) | static_cast<int>(rhs)); 
}
GlobalFitOptions operator|=(GlobalFitOptions &lhs, GlobalFitOptions rhs) {
    lhs = lhs | rhs;
    return lhs;
}
GlobalFitOptions operator&(GlobalFitOptions lhs, GlobalFitOptions rhs) { 
    return static_cast<GlobalFitOptions>(static_cast<int>(lhs) & static_cast<int>(rhs)); 
}
GlobalFitOptions operator&=(GlobalFitOptions &lhs, GlobalFitOptions rhs) {
    lhs = lhs & rhs;
    return lhs;
}
GlobalFitResult do_a_fit(const PROconfig &config, const PROpeller &prop, const PROdata &data, PROmetric *metric, const Eigen::VectorXf &ub, const Eigen::VectorXf &lb, const PROfitterConfig &fit_config, const Eigen::VectorXf &CVParams, const PROspec &cv, const std::vector<int> &global_fixed, GlobalFitOptions opt);

// Walks the collapsed reco bins and logs any with prediction < threshold,
// printing the channel, bin index/edges, prediction, and data count side-by-side.
static void logLowPredictionBins(const PROconfig &config, const Eigen::VectorXf &pred_collapsed, const Eigen::VectorXf &data_collapsed, float threshold = 1.0f, size_t var_index = 0) {
    log<LOG_INFO>(L"%1% || ----- Low-prediction reco bins (pred < %2%) -----") % __func__ % threshold;
    log<LOG_INFO>(L"%1% || %2$-30s  %3$-12s  %4$-18s  %5$-12s  %6$-12s")
        % __func__ % "channel" % "bin_in_chan" % "edges [MeV]" % "pred" % "data";
    size_t flat = 0;
    size_t n_low = 0;
    size_t global_channel_index = 0;
    for(size_t mode = 0; mode < config.m_num_modes; ++mode) {
        for(size_t det = 0; det < config.m_num_detectors; ++det) {
            for(size_t channel = 0; channel < config.m_num_channels; ++channel) {
                std::vector<float> edges = config.GetChannelVariableBins(global_channel_index, var_index).Edges();
                const std::string &cname = config.m_channel_names[channel];
                for(size_t b = 0; b + 1 < edges.size(); ++b, ++flat) {
                    if(flat >= (size_t)pred_collapsed.size()) break;
                    float p = pred_collapsed(flat);
                    float d = data_collapsed(flat);
                    if(p < threshold) {
                        ++n_low;
                        char edge_str[32];
                        std::snprintf(edge_str, sizeof(edge_str), "[%.0f, %.0f)", edges[b], edges[b+1]);
                        log<LOG_INFO>(L"%1% || %2$-30s  %3$-12zu  %4$-18s  %5$-12.4f  %6$-12.2f")
                            % __func__ % cname.c_str() % b % edge_str % p % d;
                    }
                }
                ++global_channel_index;
            }
        }
    }
    log<LOG_INFO>(L"%1% || %2% / %3% reco bins have prediction < %4%.")
        % __func__ % n_low % (size_t)pred_collapsed.size() % threshold;
}

int main(int argc, char* argv[])
{
    auto start_time = std::chrono::high_resolution_clock::now();

    gStyle->SetOptStat(0);
    CLI::App app{"PROfit: a PROfessional, PROductive fitting and oscillation framework. Together let's minimize PROfit!"}; 

    // Define options
    std::string xmlname = "NULL.xml"; 
    std::string data_xml = "";
    std::string analysis_tag = "PROfit";
    std::string output_tag = "v1";
    std::string chi2 = "PROchi";
    bool show_fit_help = false;
    bool eventbyevent=false;
    bool shapeonly = false;
    bool rateonly = false;
    bool force = false;
    bool noxrootd = false;
    bool poisson_throw = false;
    bool progress_bar = false;
    std::vector<std::string> scale_arg;
    std::map<std::string, float> scale_map;
    size_t nthread = 1;
    std::map<std::string, float> scan_fit_options;
    std::map<std::string, float> global_fit_options;
    size_t maxevents;
    int global_seed = -1;
    std::string log_file = "";
    std::vector<std::string> fit_preset = {"good","fast"};
    static const std::unordered_set<std::string> allowed_preset = {"good","fast","overkill","sensitivity"};
    bool with_splines = false, binwidth_scale = false, area_normalized = false, data_mc_ratio = false;
    std::map<std::string, float> fake_data_osc_params;
    std::map<std::string, float> cv_osc_params;
    std::map<std::string, float> injected_systs;
    std::map<std::string, float> cv_injected_systs;
    std::vector<std::string> fixed_params;
    std::vector<std::string> syst_list, systs_excluded;
    bool MCMC_prefit_errors = false;
    bool systs_only = false;
    bool use_fake_data = false;

    float xlo, xhi, ylo, yhi;
    std::array<float, 2> xlims, ylims;
    std::vector<int> grid_size;
    bool statonly = false, logx=true, logy=true;
    std::string xlabel, ylabel;
    std::string xvar = "sinsq2thmm", yvar = "dmsq";
    bool run_brazil = false;
    bool statonly_brazil = false;
    bool single_brazil = false;
    bool only_brazil = false;
    std::vector<std::string> brazil_throws;
    std::vector<float> procurve_points;

    std::string reweights_file;
    std::vector<std::string> mockreweights;
    std::vector<TH2D*> weighthists;

    std::vector<std::string> mcmc_vars;
    size_t mcmc_chains;

    std::map<std::string, float> bound_list;
    PlotBounds pbounds; 
    size_t nuniv;
    bool gof_pvalue = false;
    bool pvalue = false;


    //Global Arguments for all PROfit enables subcommands.
    app.add_option("-x,--xml", xmlname, "Input PROfit XML configuration file.")->required();
    app.add_option("-t,--tag", analysis_tag, "Analysis Tag used for output identification.")->default_str("PROfit");
    app.add_option("-v,--verbosity", GLOBAL_LEVEL, "Verbosity Level [1-4]->[Error,Warning,Info,Debug].")->default_val(GLOBAL_LEVEL);
    app.add_option("-l,--log", log_file, "File to save log to. Warning: Will overwrite this file.");
    app.add_option("-w,--file-verbosity", FILE_LEVEL, "File (log) Verbosity Level [1-4]->[Error,Warning,Info,Debug].")->default_val(static_cast<log_level_t>(-1));
    app.add_flag("-b,--progress", progress_bar, "Use a progress bar when applicable.");
    app.add_option("-o,--output",output_tag,"Additional output filename quantifier")->default_str("v1");
    app.add_option("-n, --nthread",   nthread, "Number of threads to parallelize over.")->default_val(1);
    app.add_option("-m,--max", maxevents, "Max number of events to run over.");
    app.add_option("-c, --chi2", chi2, "Which chi2 function to use. Options are PROchi or PROCNP")->default_str("PROchi");
    app.add_option("-d, --data", data_xml, "Load from a seperate data xml/data file instead of signal injection. Only used with plot subcommand.")->default_str("");
    app.add_option("-i, --inject", fake_data_osc_params, "Physics parameters to inject as fake-data true signal. Example: dmsq 3 sinsq2thmm 0.25")->expected(-1);
    app.add_option("--inject-cv", cv_osc_params, "Physics parameters to inject as CV. Example: dmsq 3 sinsq2thmm 0.25")->expected(-1);
    app.add_option("--fix", fixed_params, "Fix Certain Physics or Systematics parameters. Fixed to CV.");
    app.add_option("-s, --seed", global_seed, "A global seed for PROseed rng. Default to -1 for hardware rng seed.")->default_val(-1);
    app.add_option("-p,--preset", fit_preset, "Preset fitting params. Available `fast`, `good` and `overkill` Takes up to a vector of 2, first for global. 2nd for scan.");
    app.add_option("--fit-options", global_fit_options, "Parameters for single, detailed global best fit LBFGSB. See PROfitter.h or run --fit-help for available settings.");
    app.add_option("--scan-fit-options", scan_fit_options, "Parameters for simpier, multiple best fits in PROfile/surface LBFGSB.");
    app.add_flag("--fit-help", show_fit_help, "Show detailed help for all fitting parameters (L-BFGS-B, PSO, MCMC, etc.)");
    std::string gradient_mode_str = "central-full";
    app.add_option("--grad-mode", gradient_mode_str,
                   "Gradient evaluation strategy passed to the metric. One of: "
                   "central-full (default; central FD on full chi^2), "
                   "one-sided-full (forward FD on full chi^2; ~2x faster, O(h)), "
                   "central-lin (central FD on delta only, M frozen at base; Gauss-Newton, ~5-10x), "
                   "one-sided-lin (forward FD on delta only, M frozen at base; ~10-20x).")
        ->default_str("central-full");

    app.add_option("--inject-systs", injected_systs, "Systematic shifts to inject. Map of name and shift value in sigmas. Only spline systs are supported right now.");
    app.add_option("--inject-systs-cv", cv_injected_systs, "Systematic shifts to inject.  as CV Map of name and shift value in sigmas. Only spline systs are supported right now.");
    app.add_option("--syst-list", syst_list, "Override list of systematics to use (note: all systs must be in the xml).");
    app.add_option("--exclude-systs", systs_excluded, "List of systematics to exclude.")->excludes("--syst-list"); 

    app.add_flag("--use-fake-data", use_fake_data, "Ignore any data XML or embedded <data> section and use fake (MC) data instead.");
    app.add_flag("--poisson-throw", poisson_throw, "Do a Poisson stats throw of fake data.");
    app.add_flag("--scale-by-width", binwidth_scale, "Scale histgrams by 1/(bin width).");
    app.add_flag("--data-mc-ratio", data_mc_ratio, "For ratio plots, use data/pre-fit mc instead of data/best-fit mc.");
    app.add_option("--scale", scale_arg, "Scale detector POT by a given value.");
    app.add_option("--plot-bounds", bound_list, "Plot bounds, set by  string float pairs. Available strings are ymax,ratmin,ratmax."); 

    //app.add_option("-f, --rwfile", reweights_file, "File containing histograms for reweighting");//deprociated, add back in later
    //app.add_option("-r, --mockrw",   mockreweights, "Vector of reweights to use for mock data");
    app.add_flag("--event-by-event", eventbyevent, "Do you want to weight event-by-event?");
    app.add_flag("--statonly", statonly, "Run a stats only surface instead of fitting systematics");
    app.add_flag("--force",force,"Force loading binary data even if hash is incorrect (Be Careful!)");
    app.add_flag("--no-xrootd",noxrootd,"Do not use XRootD, which is enabled by default");
    app.add_flag("--syst-only", systs_only, "Force fitting over nuisance parameters only, currently just --fix's them");
    app.add_flag("--area-norm", area_normalized, "Make area normalized histograms.");

    auto* shape_flag = app.add_flag("--shapeonly", shapeonly, "Run a shape only analysis");
    auto* rate_flag = app.add_flag("--rateonly", rateonly, "Run a rate only analysis");
    shape_flag->excludes(rate_flag);   

    //PROcess, into binary data [Do this once first!]
    CLI::App *process_command = app.add_subcommand("process", "PROcess the MC and systematics in root files into binary data for future rapid loading.");

    //PROsurf, make a 2D surface scan of physics parameters
    CLI::App *surface_command = app.add_subcommand("surface", "Make a 2D surface scan of two physics parameters, profiling over all others.");
    surface_command->add_option("-g, --grid", grid_size, "Set grid size. If one dimension passed, grid assumed to be square, else rectangular")->expected(0, 2)->default_val(40);
    surface_command->add_option("--xvar", xvar, "Name of variable to put on x-axis")->default_val("sinsq2thmm");
    surface_command->add_option("--yvar", yvar, "Name of variable to put on x-axis")->default_val("dmsq");
    CLI::Option *xlim_opt = surface_command->add_option("--xlims", xlims, "Limits for x-axis");
    CLI::Option *ylim_opt = surface_command->add_option("--ylims", ylims, "Limits for y-axis");
    surface_command->add_option("--xlo", xlo, "Lower limit for x-axis")->excludes(xlim_opt)->default_val(1e-4);
    surface_command->add_option("--xhi", xhi, "Upper limit for x-axis")->excludes(xlim_opt)->default_val(1);
    surface_command->add_option("--ylo", ylo, "Lower limit for y-axis")->excludes(ylim_opt)->default_val(1e-2);
    surface_command->add_option("--yhi", yhi, "Upper limit for y-axis")->excludes(ylim_opt)->default_val(1e2);
    surface_command->add_option("--xlabel", xlabel, "X-axis label");
    surface_command->add_option("--ylabel", ylabel, "Y-axis label");
    surface_command->add_flag("--logx,!--linx", logx, "Specify if x-axis is logarithmic or linear (default log)");
    surface_command->add_flag("--logy,!--liny", logy, "Specify if y-axis is logarithmic or linear (default log)");
    surface_command->add_flag("--brazil-band", run_brazil, "Run 1000 throws of stats+systs and draw 1 sigma and 2 sigma Brazil bands");
    surface_command->add_flag("--stat-throws", statonly_brazil, "Only do stat throws for the Brazil band")->needs("--brazil-band");
    surface_command->add_flag("--single-throw", single_brazil, "Only run a single iteration of the Brazil band")->needs("--brazil-band");
    surface_command->add_flag("--only-throw", only_brazil, "Only run Brazil band throws and not the nominal surface")->needs("--brazil-band");
    surface_command->add_option("--from-many", brazil_throws, "Make Brazil band from many provided throws")->needs("--brazil-band");
    surface_command->add_option("--curve-mode", procurve_points , "Make a PROcurve plot from param A to param B.");

    //PROfile, make N profile'd chi^2 for each physics and nuisence parameters
    CLI::App *profile_command = app.add_subcommand("profile", "Make a 1D profiled chi2 for each physics and nuisence parameter.");
    profile_command->add_flag("--mcmc-prefit", MCMC_prefit_errors, "Use MCMC to sample the systematic priors for the pre-fit error band.");

    //PROplot, plot things
    CLI::App *proplot_command = app.add_subcommand("plot", "Make plots of CV, or injected point with error bars and covariance.");
    proplot_command->add_flag("--with-splines", with_splines, "Include graphs of splines in output.");
    std::string bkg_subtract_pattern = "";
    proplot_command->add_option("--bkg-subtract", bkg_subtract_pattern,
        "Wildcard (substring) matching one or more subchannel names; that "
        "background's central-value prediction is subtracted from data, CV, "
        "best-fit, and the error band points at plot time. The error band's "
        "spread/covariance is unchanged (Var(X - constant) = Var(X)), so the "
        "systematic uncertainty on the background continues to appear in the "
        "band. Example: --bkg-subtract numu_bkg matches every "
        "<detector>_numu_bkg subchannel.");

    //PROfc, Feldmand-Cousins
    CLI::App *profc_command = app.add_subcommand("fc", "Run Feldman-Cousins for this injected signal");
    profc_command->add_option("-u,--universes", nuniv, "Number of Feldman Cousins universes to throw")->default_val(1000);
    profc_command->add_flag("--gof", gof_pvalue, "Get GOF pvalue");
    profc_command->add_flag("--pval", pvalue, "Get FC pvalue")->excludes("--gof");

    //PROglobal
    CLI::App *proglobal_command = app.add_subcommand("global", "Just do a single global fit.");

    CLI::App *promcmc_command = app.add_subcommand("mcmc", "Get bayesian posteriors using MCMC");
    promcmc_command->add_option("--vars", mcmc_vars, "Variables to find posteriors of.");
    promcmc_command->add_option("--nchains", mcmc_chains, "Number of chains to run with MCMC.")->default_val(1);

    //PROtest, test things
    CLI::App *protest_command = app.add_subcommand("protest", "Testing ground for rapid quick tests.");

    //PRObench, scaling/timing benchmarks. Loud greppable LOG output via [SCALETEST] tag.
    // Uses the live PROmetric built by the main chain (PROchi/PROCNP/PROpoisson)
    // — no separate metric-class flag needed here.
    int    bench_N           = 1000;
    std::string bench_tests_str = "all";
    CLI::App *bench_command = app.add_subcommand("scale-test", "Run timing benchmarks for FillSpectra / metric / fit hot paths and emit greppable [SCALETEST] LOG lines.");
    bench_command->add_option("-N,--n", bench_N, "Base call count: FillSpectra=N, metric=N/10, fit=N/100.")->default_val(1000);
    bench_command->add_option("--tests", bench_tests_str, "Comma-separated subset of {a..n} or {fillspectra,metric,metricgrad,fit,pseudo,collapse,mcmc,all}. Default 'all'.")->default_val("all");

    app.set_config("--config");
    surface_command->configurable(true);
    process_command->configurable(true);
    profile_command->configurable(true);
    protest_command->configurable(true);
    proglobal_command->configurable(true);
    profc_command->configurable(true);
    proplot_command->configurable(true);
    promcmc_command->configurable(true);
    bench_command->configurable(true);

    //Parse inputs. 
    CLI11_PARSE(app, argc, argv);

    if(show_fit_help) {
        PROfit::PROfitterConfig::PrintHelp();
        return 0;
    }

    if(log_file != "") {

        if(FILE_LEVEL == static_cast<log_level_t>(-1)) {
            FILE_LEVEL = GLOBAL_LEVEL;
        }

        log_impl::EnableFileLogging(log_file, FILE_LEVEL);
    }

    if(shapeonly) area_normalized = true;

    pbounds.Load(bound_list);

    log<LOG_WARNING>(L" %1% ") % getIcon().c_str()  ;
    std::string final_output_tag =analysis_tag +"_"+output_tag;




    log<LOG_WARNING>(L"%1% || ##################################################################") % __func__  ;
    log<LOG_WARNING>(L"%1% || ####################### PROfit version v%2% ######################") % __func__ % PROJECT_VERSION_STR ;
    log<LOG_WARNING>(L"%1% || ##################################################################") % __func__  ;
    log<LOG_WARNING>(L"%1% || PROfit commandline input arguments. xml: %2%, tag: %3%, output %4%, nthread: %5% ") % __func__ % xmlname.c_str() % analysis_tag.c_str() % output_tag.c_str() % nthread ;

    //Initilize configuration from the XML;
    PROconfig config(xmlname, rateonly);

    //Inititilize PROpeller to keep MC
    PROpeller prop;

    //Initilize objects for systematics storage
    std::vector<std::vector<SystStruct>> systsstructs;

    //input/output logic
    std::string propBinName = analysis_tag+"_prop.bin";
    std::string systBinName = analysis_tag+"_syst.bin";

    bool need_main_process = (*process_command) || (!std::filesystem::exists(systBinName) || !std::filesystem::exists(propBinName));

    if(need_main_process){
        log<LOG_INFO>(L"%1% || Processing PROpeller and PROsysts from XML defined root files, and saving to binary output also: %2%") % __func__ % propBinName.c_str();
        //Process the CAF files to grab and fill all SystStructs and PROpeller
        PROcess_CAFAna(config, systsstructs, prop,noxrootd);
        prop.save(propBinName);
        saveSystStructVector(systsstructs, systBinName);
        log<LOG_INFO>(L"%1% || Done processing PROpeller and PROsysts from XML defined root files, and saving to binary output also: %2%") % __func__ % propBinName.c_str();

    }else{
        log<LOG_INFO>(L"%1% || Loading PROpeller and PROsysts from precalc binary input: %2%") % __func__ % propBinName.c_str();
        prop.load(propBinName);
        loadSystStructVector(systsstructs, systBinName);

        //is hash right for PROpeller first?
        log<LOG_INFO>(L"%1% || Done loading. Config hash (%2%) and binary loaded PROpeller (%3%) are here. ") % __func__ %  config.hash % prop.hash ;
        if(config.hash!=prop.hash){
            if(force){
                log<LOG_WARNING>(L"%1% || WARNING config hash (%2%) and binary loaded PROpeller (%3%)  not compatable! ") % __func__ %  config.hash % prop.hash ;
                log<LOG_WARNING>(L"%1% || WARNING But we are forcing ahead, be SUPER clear and happy you understand what your doing.  ") % __func__;
            }else{
                log<LOG_ERROR>(L"%1% || ERROR config hash (%2%) and binary loaded PROpeller (%3%)  not compatable! ") % __func__ %  config.hash % prop.hash ;
                return 1;
            }
        }
        //Now check syststructs, if there is any!
        if(systsstructs.front().size()>0){
            log<LOG_INFO>(L"%1% || Done loading. Config hash (%2%) and binary loaded PROsyst hash(%3%) are here. ") % __func__ %  config.hash % systsstructs[0][0].hash;
            if( config.hash!=systsstructs.front().front().hash){
                if(force){
                    log<LOG_WARNING>(L"%1% || WARNING config hash (%2%) and binary loaded PROsyst hash(%3%) not compatable! ") % __func__ %  config.hash %  systsstructs.front().front().hash;
                    log<LOG_WARNING>(L"%1% || WARNING But we are forcing ahead, be SUPER clear and happy you understand what your doing.  ") % __func__;
                }else{
                    log<LOG_ERROR>(L"%1% || ERROR config hash (%2%) and binary loaded PROsyst hash(%3%) not compatable! ") % __func__ %  config.hash %  systsstructs.front().front().hash;
                    return 1;
                }
            }
        }

    }

    // Combined DetVar propeller binary: one file for all DetVar files, keyed by section+name.
    // Uses detvar_hash (binning + DetVar section only), so changes to <DetVarFiles> or
    // top-level binning trigger reprocessing without invalidating the main prop/syst binaries.
    if(config.m_has_detvar_section) {
        std::string dvAllPropsBin = analysis_tag + "_detvar_props.bin";
        bool need_detvar_process = (*process_command) || !std::filesystem::exists(dvAllPropsBin);

        std::map<std::string, PROpeller> dvprops;

        if(need_detvar_process) {
            log<LOG_INFO>(L"%1% || Processing all DetVar files into combined binary: %2%") % __func__ % dvAllPropsBin.c_str();
            for(size_t idv = 0; idv < config.GetNumDetVarFiles(); ++idv) {
                const std::string& name = config.m_detvar_files[idv].name;
                const std::string key = DetVarKey(config, idv);
                log<LOG_INFO>(L"%1% || Processing DetVar file '%2%'") % __func__ % name.c_str();
                PROconfig dvconfig = config.BuildDetVarConfig(idv);
                PROpeller dvprop;
                std::vector<std::vector<SystStruct>> dvsystsstructs;
                PROcess_CAFAna(dvconfig, dvsystsstructs, dvprop, noxrootd);
                dvprops[key] = std::move(dvprop);
                log<LOG_INFO>(L"%1% || Done processing DetVar file '%2%'") % __func__ % name.c_str();
            }
            saveDetVarProps(dvprops, config.detvar_hash, dvAllPropsBin);
        } else {
            log<LOG_INFO>(L"%1% || Loading DetVar props from combined binary: %2%") % __func__ % dvAllPropsBin.c_str();
            uint32_t loaded_detvar_hash = loadDetVarProps(dvprops, dvAllPropsBin);
            log<LOG_INFO>(L"%1% || Config detvar_hash (%2%) and binary detvar_hash (%3%).") % __func__ % config.detvar_hash % loaded_detvar_hash;
            if(config.detvar_hash != loaded_detvar_hash) {
                if(force) {
                    log<LOG_WARNING>(L"%1% || WARNING config detvar_hash (%2%) and binary detvar_hash (%3%) not compatible!") % __func__ % config.detvar_hash % loaded_detvar_hash;
                } else {
                    log<LOG_ERROR>(L"%1% || ERROR config detvar_hash (%2%) and binary detvar_hash (%3%) not compatible!") % __func__ % config.detvar_hash % loaded_detvar_hash;
                    return 1;
                }
            }
        }

        // Build DetVar SystStructs in memory from dvprops (not stored in syst.bin —
        // they live in the DetVar binary so either binary can be regenerated independently).
        log<LOG_INFO>(L"%1% || Building DetVar SystStructs from DetVar props...") % __func__;
        PROsyst emptySyst;

        for(size_t isec = 0; isec < config.GetNumDetVarSections(); ++isec) {

            // Find CV index for this section
            size_t cv_idx = SIZE_MAX;
            for(size_t i = 0; i < config.m_detvar_files.size(); ++i) {
                if(config.m_detvar_files[i].is_cv && config.m_detvar_files[i].section_index == isec) {
                    cv_idx = i; break;
                }
            }
            if(cv_idx == SIZE_MAX) {
                log<LOG_ERROR>(L"%1% || ERROR: No CV file found for DetVar section %2%") % __func__ % isec;
                continue;
            }

            PROpeller& cvprop = dvprops.at(DetVarKey(config, cv_idx));
            PROconfig cvconfig = config.BuildDetVarConfig(cv_idx);
            NullModel cvmodel(cvprop);
            Eigen::VectorXf cvparams = Eigen::VectorXf::Constant(cvmodel.nparams, 0);
            int cv_binning = cvconfig.i_prime;
            if(cv_binning < 0 || cv_binning >= (int)config.m_num_variables)
                cv_binning = config.i_prime;
            PROspec cvSpec = FillSpectra(cvconfig, cvprop, emptySyst, cvmodel, cvparams, true, cv_binning);

            std::vector<size_t> skip;
            for(size_t idv = 0; idv < config.m_detvar_files.size(); ++idv) {
                if(skip.size() && std::find(skip.begin(), skip.end(), idv) != skip.end()) continue;
                if(config.m_detvar_files[idv].section_index != isec) continue;
                if(config.m_detvar_files[idv].is_cv) continue;

                const std::string& varName = config.m_detvar_files[idv].name;

                if(config.m_mcgen_variation_type_map.count(varName) == 0) {
                    log<LOG_INFO>(L"%1% || Skipping DetVar '%2%' — no matching entry in <systematics> section.") % __func__ % varName.c_str();
                    continue;
                }
                std::map<int, size_t> syst_files;
                auto find_fn = [&varName](const PROconfig::DetVarFile &dvf) { return dvf.name == varName; };
                auto it = config.m_detvar_files.begin() + idv;
                while((it = std::find_if(it, config.m_detvar_files.end(), find_fn))
                        != std::end(config.m_detvar_files)) {
                    size_t i = std::distance(config.m_detvar_files.begin(), it);
                    syst_files[it->knobval] = i;
                    skip.push_back(i);
                    it++;
                }

                const std::string& systType = config.m_mcgen_variation_type_map.at(varName);
                int binningIndex = config.m_mcgen_variation_binning_map.count(varName) ? config.m_mcgen_variation_binning_map.at(varName) : config.i_prime;
                if(binningIndex < 0 || binningIndex >= (int)config.m_num_variables)
                    binningIndex = config.i_prime;

                PROspec cvSpec = FillSpectra(cvconfig, cvprop, emptySyst, cvmodel, cvparams, true, binningIndex);
                std::map<int, PROspec> specs;
                std::map<int, const PROpeller*> props;
                for(const auto &[k, i] : syst_files) {
                    PROpeller& dvprop = dvprops.at(DetVarKey(config, i));
                    PROconfig dvconfig = config.BuildDetVarConfig(i);
                    NullModel dvmodel(dvprop);
                    Eigen::VectorXf dvparams = Eigen::VectorXf::Constant(dvmodel.nparams, 0);
                    specs[k] = FillSpectra(dvconfig, dvprop, emptySyst, dvmodel, dvparams, true, binningIndex);
                    props[k] = &dvprop;
                }

                // Attempt to build matched specs using only common (run,subrun,event) events.
                // If both propellers have matching vars stored, replace cvSpec/varSpec for this pair.
                PROspec matchedCvSpec = cvSpec;
                const bool matched = BuildDetVarMatchedSpecs(
                    cvprop, props, binningIndex, (int)config.m_num_variable_bins_total[binningIndex],
                    matchedCvSpec, specs);
                if(matched) {
                    log<LOG_INFO>(L"%1% || DetVar '%2%': using event-matched spectra for spline building") % __func__ % varName.c_str();
                    // When cv_variation_matching_vars is used, undo the POT scaling that was
                    // applied during propeller filling so both CV and variation matched spectra
                    // are in raw event-weight units. The spline ratio then reflects only detector
                    // shape/efficiency effects, not any POT normalization artifact.
                    const double det_pot = config.m_det_pot[0];
                    const double cv_pot_dv = config.m_detvar_files[cv_idx].pot;
                    if(det_pot > 0.0 && cv_pot_dv > 0.0) {
                        const float cv_unscale = (float)(cv_pot_dv / det_pot);
                        matchedCvSpec.Spec() *= cv_unscale;
                        matchedCvSpec.Error() *= cv_unscale;
                        for(const auto &[kv, var_file_idx] : syst_files) {
                            const double var_pot_dv = config.m_detvar_files[var_file_idx].pot;
                            if(var_pot_dv > 0.0) {
                                const float var_unscale = (float)(var_pot_dv / det_pot);
                                specs[kv].Spec() *= var_unscale;
                                specs[kv].Error() *= var_unscale;
                            }
                        }
                    }
                } else {
                    log<LOG_INFO>(L"%1% || DetVar '%2%': no matching vars stored, using full spectra") % __func__ % varName.c_str();
                }

                {
                    // Zero out bins where CV is 0 to avoid division by zero when constructing splines
                    Eigen::ArrayXf mask = (matchedCvSpec.Spec().array() != 0.0f).cast<float>();
                    for(auto &[_, spec] : specs) {
                        spec.Spec() = spec.Spec().array() * mask;
                        spec.Error() = spec.Error().array() * mask;
                    }
                }

                {
                    std::vector<eweight_type> knobvals;
                    std::transform(specs.begin(), specs.end(), std::back_inserter(knobvals),
                            [](const auto &p){ return p.first; });
                    std::sort(knobvals.begin(), knobvals.end());
                    SystStruct ss(varName, specs.size(), systType, "1",
                                  knobvals, knobvals, 0);
                    ss.binning = binningIndex;
                    ss.CreateSpecs(matchedCvSpec.Spec().size());
                    ss.p_cv = std::make_shared<PROspec>(matchedCvSpec);
                    for(const auto &[kv, spec] : specs) {
                        size_t idx = std::distance(knobvals.begin(), std::find(knobvals.begin(), knobvals.end(), kv));
                        ss.p_multi_spec[idx] = std::make_shared<PROspec>(specs[kv]);
                    }
                    ss.SetHash(config.hash);
                    for(auto &ssv : systsstructs) ssv.push_back(ss);
                }
                log<LOG_INFO>(L"%1% || Added DetVar SystStruct '%2%' (section %3%, binning=%4%, mode=%5%)") % __func__ % varName.c_str() % isec % binningIndex % systType.c_str();
            }
        }
    }

    // For process-only command, exit early after MC processing is complete
    // This avoids unnecessary setup and potential cleanup issues with ROOT
    //if(*process_command && !*profile_command && !*surface_command && !*protest_command && !*proglobal_command && !*proplot_command && !*profc_command) {
    //    log<LOG_WARNING>(L"%1% || Process command complete. Binary files saved successfully.") % __func__;
    //    return 0;
    //}

    //Scale events by some percentage of total detector POT
    if(scale_arg.size()) {
        if (scale_arg.size() % 2 != 0) {
            log<LOG_ERROR>(L"%1% || Expected pairs of detector and scaling values (e.g., ICARUS 0.5)") % __func__;
            exit(EXIT_FAILURE);
        }
        for (size_t i = 0; i < scale_arg.size(); i += 2) {
            scale_map[scale_arg[i]] = std::stof(scale_arg[i + 1]);
        }
        prop.scale(config, scale_map);
    }

    //Before building, if we fixed a nuisence parameter to a value, lets shift to that value.
    for(auto &sys: fixed_params){
           float def = config.m_mcgen_variation_prior_centers.count(sys) ? config.m_mcgen_variation_prior_centers[sys] : 0.0;
           config.m_mcgen_variation_prior_centers[sys]= cv_injected_systs.count(sys) ? cv_injected_systs.at(sys) : def ;
    }

    //Seed time
    PROseed myseed(nthread, global_seed);
    std::uniform_int_distribution<uint32_t> dseed(0, std::numeric_limits<uint32_t>::max());



    std::unique_ptr<PROmodel> model = get_model_from_string(config, prop);
    std::unique_ptr<PROmodel> null_model = std::make_unique<NullModel>(prop);

    //Build a PROsyst to sort and analyze all systematics
    //PROsyst systs(prop, config, systsstructs.front(), shapeonly);
    std::vector<PROsyst> variable_systs;
    for(size_t i = 0; i < config.m_num_variables; ++i){

        if(config.m_channel_variable_plot_bool.at(i) || i == config.i_prime){ 
            variable_systs.emplace_back(prop, config, systsstructs.at(i), shapeonly, i, model.get(), nullptr);
        }else{
            variable_systs.emplace_back();
        }
        //variable_systs.back().PrintSplines();
        //return 0;
    }

    Eigen::VectorXf fake_data_osc_param_vector = model->default_val;
    Eigen::VectorXf cv_osc_param_vector = model->default_val;

    //loop over input fake data physics params and check/set
    for(const auto &[name, value]: fake_data_osc_params) {
        const auto it = std::find(model->param_names.begin(), model->param_names.end(), name);
        if(it == std::end(model->param_names)) {
            log<LOG_ERROR>(L"%1% || Unrecognized model parameter name %2%.\n"
                    L"Valid names for model %3% are %4%") %
                __func__% name.c_str()% config.m_model_tag.c_str()%
                model->param_names;
            return 1;
        }
        int loc = std::distance(model->param_names.begin(), it);
        fake_data_osc_param_vector(loc) = model->is_log10[loc] ? std::log10(value) : value;
        log<LOG_INFO>(L"%1% Set fake data injected parameter %2% to value %3%, internally %4%") % __func__ % name.c_str() % value % fake_data_osc_param_vector(loc);
    }

    //loop over input CV physics params and check/set
    for(const auto &[name, value]: cv_osc_params) {
        const auto it = std::find(model->param_names.begin(), model->param_names.end(), name);
        if(it == std::end(model->param_names)) {
            log<LOG_ERROR>(L"%1% || Unrecognized model parameter name %2%.\n"
                    L"Valid names for model %3% are %4%") %
                __func__% name.c_str()% config.m_model_tag.c_str()%
                model->param_names;
            return 1;
        }
        int loc = std::distance(model->param_names.begin(), it);
        cv_osc_param_vector(loc) = model->is_log10[loc] ? std::log10(value) : value;
        log<LOG_INFO>(L"%1% Set CV injected parameter %2% to value %3%, internally %4%") % __func__ % name.c_str() % value % fake_data_osc_param_vector(loc);
    }


    //Spline fake data injection studies
    Eigen::VectorXf fakedataparams = Eigen::VectorXf::Constant(model->nparams + variable_systs[config.i_prime].GetNSplines(), 0);
    for(size_t i = 0; i < model->nparams; ++i) fakedataparams(i) = fake_data_osc_param_vector(i);
    log<LOG_INFO>(L"%1% || model->default_val: %2%") % __func__ % model->default_val;
    log<LOG_INFO>(L"%1% || fake_data_osc_param_vector: %2%") % __func__ % fake_data_osc_param_vector;
    log<LOG_INFO>(L"%1% || fakedataparams (physics portion): %2% %3%") % __func__ % fakedataparams(0) % fakedataparams(1);
    for(const auto& [name, shift]: injected_systs) {
        log<LOG_INFO>(L"%1% || Injected syst: %2% shifted by %3%") % __func__ % name.c_str() % shift;

        auto it = std::find(variable_systs[config.i_prime].spline_names.begin(), variable_systs[config.i_prime].spline_names.end(), name);
        if(it == variable_systs[config.i_prime].spline_names.end()) {
            for(const auto &[xml_name, plot_name]: config.m_mcgen_variation_plotname_map) {
                if(name == plot_name) {
                    it = std::find(variable_systs[config.i_prime].spline_names.begin(), variable_systs[config.i_prime].spline_names.end(), xml_name);
                    break;
                }
            }
            if(it == variable_systs[config.i_prime].spline_names.end()) {
                log<LOG_ERROR>(L"%1% || Error: Unrecognized spline %2%. Ignoring this injected shift.") % __func__ % name.c_str();
                continue;
            }

        }
        int idx = std::distance(variable_systs[config.i_prime].spline_names.begin(), it);
        fakedataparams(idx+model->nparams) = shift;
    }

    

    //Some logic for EITHER injecting fake/mock data of oscillated signal/syst shifts OR using real data
    //Main data for the i_prime fitting.
    PROdata data;

    //We will load all other variables too, but many are truth level so data won't be as common.
    std::vector<PROdata> variable_data;
    // Support data from either --data flag or embedded <data> section in the XML
    bool use_real_data = (!data_xml.empty() || config.m_has_data_section) && !use_fake_data;
    if(use_real_data){
        PROconfig dataconfig;
        if(!data_xml.empty()){
            // Explicit --data flag takes precedence
            dataconfig = PROconfig(data_xml);
        } else {
            // Use embedded <data> section from the unified XML
            log<LOG_INFO>(L"%1% || Using embedded <data> section from XML for data config") % __func__;
            dataconfig = config.BuildDataConfig();
        }
        std::string dataBinName = analysis_tag+"_data.bin";
        for(size_t i = 0; i < dataconfig.m_num_channels; ++i) {
            size_t nsubch = dataconfig.m_num_subchannels[i];
            if(nsubch != 1) {
                log<LOG_ERROR>(L"%1% || Data xml required to have exactly 1 subchannel per channel. Found %2% for channel %3%")
                    % __func__ % nsubch % i;
                log<LOG_ERROR>(L"Terminating.");
                exit(EXIT_FAILURE);
            }
            std::string &subchname = dataconfig.m_subchannel_names[i][0];
            if(subchname != "data") {
                log<LOG_ERROR>(L"%1% || Data subchannel required to be called \"data.\" Found name %2% for channel %3%")
                    % __func__ % subchname.c_str() % i;
                log<LOG_ERROR>(L"Terminating.");
                exit(EXIT_FAILURE);
            }
        }
        if(!PROconfig::SameChannels(config, dataconfig)) {
            log<LOG_ERROR>(L"%1% || Require data and MC to have same channels. A difference was found, check messages above.")
                % __func__;
            log<LOG_ERROR>(L"Terminating.");
            exit(EXIT_FAILURE);
        }

        if((*process_command) || (!std::filesystem::exists(dataBinName))  ){
            log<LOG_INFO>(L"%1% || Processing Data Spectrum and saving to binary output also: %2%") % __func__ % dataBinName.c_str();

            //Process the CAF files to grab and fill spectrum directly
            std::vector<PROdata> alldata = CreatePROdata(dataconfig);
            PROdata::saveVector(dataconfig, alldata, dataBinName);
            data = alldata[config.i_prime];
            //data.save(dataconfig,dataBinName);
            
            for(size_t io = 0; io < dataconfig.m_num_variables; ++io)
                variable_data.push_back(alldata[io]);

            log<LOG_INFO>(L"%1% || Done processing Data from XML defined root files, and saving to binary output also: %2%") % __func__ % dataBinName.c_str();
        }else{
            log<LOG_INFO>(L"%1% || Loading Data from precalc binary input: %2%") % __func__ % dataBinName.c_str();
            //data.load(dataBinName);
            std::vector<PROdata> alldata;
            PROdata::loadVector(alldata, dataBinName);
            data = alldata[config.i_prime];
            //data.save(dataconfig,dataBinName);

            for(size_t io = 0; io < dataconfig.m_num_variables; ++io)
                variable_data.push_back(alldata[io]);

            log<LOG_INFO>(L"%1% || Done loading. Config hash (%2%) and binary loaded Data (%3%) hash are here. ") % __func__ %  dataconfig.hash % data.hash;
            if(dataconfig.hash!=data.hash){
                if(force){
                    log<LOG_WARNING>(L"%1% || WARNING config hash (%2%) and binary loaded data (%3%) hash not compatable! ") % __func__ %  dataconfig.hash % data.hash ;
                    log<LOG_WARNING>(L"%1% || WARNING But we are forcing ahead, be SUPER clear and happy you understand what your doing.  ") % __func__;
                }else{
                    log<LOG_ERROR>(L"%1% || ERROR config hash (%2%) and binary loaded data (%3%) hash not compatable! ") % __func__ %  dataconfig.hash % data.hash ;
                    return 1;
                }
            }
        }

        /*if(*profile_command || *surface_command || *protest_command){
            log<LOG_ERROR>(L"%1% || ERROR --data can only be used with plot subcommand! ") % __func__  ;
            return 1;
        }*/


    }//if no data, use injected or fake data;
    else{
        log<LOG_INFO>(L"%1% || Going to get fake data set up for each variable.") % __func__ ;
        for(size_t io = 0; io < config.m_num_variables; ++io) {
            PROspec data_spec = config.m_channel_variable_plot_bool.at(io) ?  FillSpectra(config, prop, variable_systs[io], *model, fakedataparams, !eventbyevent, io) : PROspec(config.m_num_variable_bins_total[io]) ;
            if(poisson_throw) data_spec = PROspec::PoissonVariation(data_spec, dseed(myseed.global_rng));
            Eigen::VectorXf data_vec = CollapseMatrix(config, data_spec.Spec(), io);
            variable_data.push_back(PROdata(data_vec, data_vec.array().sqrt()));
        }
    }

    data = variable_data[config.i_prime];

    // Leave this after creating fake data so we can make fake data using systs that aren't
    // included in the fit.

    //for(auto & systs : variable_systs){
    {
        if(syst_list.size()) {

            std::vector<std::string> systs_to_include;
            for(const auto &s: syst_list) {
                bool istag = false;
                for(const auto &[syst, tags]: config.m_mcgen_variation_tags) {
                    if(std::find(tags.begin(), tags.end(), s) != std::end(tags)) {
                        istag = true;
                        systs_to_include.push_back(syst);
                    }
                }
                if(!istag) systs_to_include.push_back(s);
            }
            for(std::string &name: systs_to_include) {
                for(const auto &[xml_name, plot_name]: config.m_mcgen_variation_plotname_map) {
                    if(name == plot_name) {
                        name = xml_name;
                    }
                }
            }
            //systs = systs.subset(systs_to_include);
            int io=0;
            for(PROsyst &syst: variable_systs){
                if(config.m_channel_variable_plot_bool.at(io)){
                    syst = syst.subset(systs_to_include);
                }
                io++;
            }
        } else if(systs_excluded.size()) {

            std::vector<std::string> systs_to_exclude;
            for(const auto &s: systs_excluded) {
                log<LOG_INFO>(L"%1% || Excluding systematic %2% by command line argument.") % __func__ % s.c_str();
                bool istag = false;
                for(const auto &[syst, tags]: config.m_mcgen_variation_tags) {
                    if(std::find(tags.begin(), tags.end(), s) != std::end(tags)) {
                        istag = true;
                        systs_to_exclude.push_back(syst);
                    }
                }
                if(!istag) systs_to_exclude.push_back(s);
            }
            for(std::string &name: systs_to_exclude) {
                for(const auto &[xml_name, plot_name]: config.m_mcgen_variation_plotname_map) {
                    if(name == plot_name) {
                        name = xml_name;
                    }
                }
            }
            //systs = systs.excluding(systs_to_exclude);
            int io=0;
            for(PROsyst &syst: variable_systs){
                if(config.m_channel_variable_plot_bool.at(io)){
                    syst = syst.excluding(systs_to_exclude);
                }
                io++;
            }
        }
    }

    // Empty-bin sanity check: build a default-CV spectrum and look for collapsed bins
    // that would make stat-only / CNP statistics singular (CV<=0) or untrustworthy (CV<1).
    {
        const size_t io = config.i_prime;
        Eigen::VectorXf cv_check_params =
            Eigen::VectorXf::Zero(model->nparams + variable_systs[io].GetNSplines());
        for(size_t i = 0; i < model->nparams; ++i)
            cv_check_params(i) = model->default_val(i);

        PROspec cv_check_spec = FillSpectra(config, prop, variable_systs[io], *model,
                                            cv_check_params, !eventbyevent, io);
        Eigen::VectorXf collapsed_cv = CollapseMatrix(config, cv_check_spec.Spec(), io);

        int n_zero = 0, n_tiny = 0;
        for(Eigen::Index b = 0; b < collapsed_cv.size(); ++b) {
            if(collapsed_cv(b) <= 0.0f) {
                log<LOG_ERROR>(L"%1% || Default-CV collapsed bin %2% has %3% expected events (<=0). Empty-bin would make CNP/stat covariance singular.") % __func__ % (long)b % collapsed_cv(b);
                ++n_zero;
            } else if(collapsed_cv(b) < 1.0f) {
                log<LOG_WARNING>(L"%1% || Default-CV collapsed bin %2% has %3% expected events (<1). Low-stat region; results in this bin may be unreliable.") % __func__ % (long)b % collapsed_cv(b);
                ++n_tiny;
            }
        }
        if(n_zero > 0) {
            if(!force) {
                log<LOG_ERROR>(L"%1% || %2% collapsed CV bins are empty. Aborting. Re-run with --force to override (NOT recommended).") % __func__ % n_zero;
                return 1;
            }
            log<LOG_WARNING>(L"%1% || %2% collapsed CV bins are empty but --force was set; proceeding at user's risk.") % __func__ % n_zero;
        }
        if(n_tiny > 0)
            log<LOG_WARNING>(L"%1% || %2% collapsed CV bins have <1 expected event.") % __func__ % n_tiny;

        if(use_real_data) {
            int n_nan = 0, n_neg = 0;
            const Eigen::VectorXf &dvec = data.Spec();
            for(Eigen::Index b = 0; b < dvec.size(); ++b) {
                if(std::isnan(dvec(b)) || std::isinf(dvec(b))) {
                    log<LOG_ERROR>(L"%1% || Data bin %2% is NaN/inf.") % __func__ % (long)b;
                    ++n_nan;
                } else if(dvec(b) < 0.0f) {
                    log<LOG_WARNING>(L"%1% || Data bin %2% is negative (%3%).") % __func__ % (long)b % dvec(b);
                    ++n_neg;
                }
            }
            if(n_nan > 0 && !force) {
                log<LOG_ERROR>(L"%1% || %2% data bins are NaN/inf. Aborting. Re-run with --force to override.") % __func__ % n_nan;
                return 1;
            }
        }
    }

    //Pysics parameter input
    Eigen::VectorXf fakeDataParams = Eigen::VectorXf::Constant(model->nparams + variable_systs[config.i_prime].GetNSplines(), 0);
    Eigen::VectorXf CVParams = Eigen::VectorXf::Constant(model->nparams + variable_systs[config.i_prime].GetNSplines(), 0);

    //Spline CV  injection studies [NEED TO GO AFTER the remove exclude systs]
    for(const auto& [name, shift]: cv_injected_systs) {
        log<LOG_INFO>(L"%1% || Injected syst: %2% shifted by %3%") % __func__ % name.c_str() % shift;

        auto it = std::find(variable_systs[config.i_prime].spline_names.begin(), variable_systs[config.i_prime].spline_names.end(), name);
        if(it == variable_systs[config.i_prime].spline_names.end()) {
            for(const auto &[xml_name, plot_name]: config.m_mcgen_variation_plotname_map) {
                if(name == plot_name) {
                    it = std::find(variable_systs[config.i_prime].spline_names.begin(), variable_systs[config.i_prime].spline_names.end(), xml_name);
                    break;
                }
            }
            if(it == variable_systs[config.i_prime].spline_names.end()) {
                log<LOG_ERROR>(L"%1% || Error: Unrecognized spline %2%. Ignoring this injected shift.") % __func__ % name.c_str();
                continue;
            }

        }
        int idx = std::distance(variable_systs[config.i_prime].spline_names.begin(), it);
        CVParams(idx+model->nparams) = shift;
    }

    for(long i = 0; i < fake_data_osc_param_vector.size(); ++i) {
        fakeDataParams(i) = fake_data_osc_param_vector(i);
        CVParams(i) = model->default_val(i);
    }
    for(long i = 0; i < cv_osc_param_vector.size(); ++i) {
        CVParams(i) = cv_osc_param_vector(i);
    }


    log<LOG_INFO>(L"%1% || Starting from fit preset :  %2%.")% __func__ % fit_preset;
    for(auto &fit_pre: fit_preset){
        if (allowed_preset.find(fit_pre) == allowed_preset.end()) {
            log<LOG_ERROR>(L"%1% || ERROR allowed fit_presets are good, fast, sensitivity or overkill. You entred : %2%.")% __func__ % fit_pre.c_str();
            return 1;
        }
    }
    //Some global minimizer params
    // This runs for the single best gobal fit
    PROfitterConfig fitConfig(global_fit_options, fit_preset.front(), false);



    //Some Scan minimizer params.
    // This runs lots during PROfile and surface.
    PROfitterConfig scanFitConfig(scan_fit_options, fit_preset.back(), true);

    // Apply --grad-mode to BOTH fit configurations. PROfitter::Fit calls
    // metric.setGradientMode(...) at the start of every fit, so the same flag
    // controls global fits, profile fits, surface fits, and FC fits uniformly.
    // The double-parse here detects an unrecognised token: the parser returns
    // the fallback for unknown input, so calling it with two different
    // sentinels and comparing flags any input that wasn't matched against
    // either of them.
    {
        const PROmetric::GradientMode gmode_a =
            PROmetric::parseGradientMode(gradient_mode_str, PROmetric::GradientCentralFull);
        const PROmetric::GradientMode gmode_b =
            PROmetric::parseGradientMode(gradient_mode_str, PROmetric::GradientOneSidedLin);
        if (gmode_a != gmode_b) {
            log<LOG_WARNING>(L"%1% || Unknown --grad-mode '%2%'; falling back to central-full.")
                % __func__ % gradient_mode_str.c_str();
        }
        fitConfig.gradient_mode     = gmode_a;
        scanFitConfig.gradient_mode = gmode_a;
        log<LOG_INFO>(L"%1% || Gradient mode: %2%") % __func__ % PROmetric::gradientModeName(gmode_a);
    }



    //Section to set bounds, as well as fix 
    size_t N_phys_params = model->nparams;
    size_t N_syst_params = variable_systs[config.i_prime].GetNSplines();
    size_t N_params = N_phys_params+N_syst_params;

    Eigen::VectorXf global_lb = Eigen::VectorXf::Constant(N_params, -3.0);
    Eigen::VectorXf global_ub = Eigen::VectorXf::Constant(N_params, 3.0);
    std::vector<int> global_fixed(N_params,0); 
    log<LOG_INFO>(L"%1% || We are hoping to FIX : %2% ") % __func__ % fixed_params;
    for(size_t i = 0; i < N_phys_params; ++i) {
                std::string name = model->param_names[i];
                std::string pname = model->pretty_param_names[i];
                if( systs_only || std::find(fixed_params.begin(), fixed_params.end(), pname) != fixed_params.end() || std::find(fixed_params.begin(), fixed_params.end(), name) != fixed_params.end()){
                    log<LOG_INFO>(L"%1% || We are FIXING physics parameter %2% (%3%) at value %4% ") % __func__ % i % name.c_str() % CVParams(i);  
                    model->lb(i) = CVParams(i);
                    model->ub(i) = CVParams(i);
                    global_fixed.at(i)=1;
                }
                global_lb(i) = model->lb(i);
                global_ub(i) = model->ub(i);

           
    }
    for(size_t i = N_phys_params; i < N_params; ++i) {
                std::string name = variable_systs[config.i_prime].spline_names[i-N_phys_params];
                std::string pname =config.m_mcgen_variation_plotname_map.at(name); 

                if( std::find(fixed_params.begin(), fixed_params.end(), name) != fixed_params.end() || std::find(fixed_params.begin(), fixed_params.end(), pname) != fixed_params.end()){
                    log<LOG_INFO>(L"%1% || We are FIXING syst parameter %2% (%3%) at value %4% ") % __func__ % i % name.c_str() % CVParams(i);  
                    variable_systs[config.i_prime].spline_hi[i-N_phys_params] = CVParams(i);
                    variable_systs[config.i_prime].spline_lo[i-N_phys_params] = CVParams(i);
                    global_fixed.at(i)=1;
                }
                global_lb(i) = variable_systs[config.i_prime].spline_lo[i-N_phys_params];
                global_ub(i) = variable_systs[config.i_prime].spline_hi[i-N_phys_params];

    }
    if( (fixed_params.size()!=std::accumulate(global_fixed.begin(), global_fixed.end(), (size_t)0)) && !systs_only ){
            log<LOG_ERROR>(L"%1% || ERROR. The fixed parameters you passed, check they exist? the number of fixed params is not the same as input params.") % __func__;
            log<LOG_ERROR>(L"%1% || ERROR. fixed_params %2% ") % __func__ % fixed_params;
            log<LOG_ERROR>(L"%1% || ERROR. global_fixed %2% : sum %3% ") % __func__ % global_fixed % ((int)std::accumulate(global_fixed.begin(), global_fixed.end(), 0)) ;
            exit(EXIT_FAILURE);
    }


    //Metric Time
    //Metrics are for i_prime only for now
    PROmetric *metric;
    if(chi2 == "PROchi") {
        metric = new PROchi("", config, prop, &(variable_systs[config.i_prime]), *model, data, eventbyevent ? PROmetric::EventByEvent : PROmetric::BinnedChi2,shapeonly);
    } else if(chi2 == "PROCNP") {
        metric = new PROCNP("", config, prop, &(variable_systs[config.i_prime]), *model, data, eventbyevent ? PROmetric::EventByEvent : PROmetric::BinnedChi2,shapeonly);
    } else if(chi2 == "Poisson") {
        metric = new PROpoisson("", config, prop, &(variable_systs[config.i_prime]), *model, data, eventbyevent ? PROmetric::EventByEvent : PROmetric::BinnedChi2,shapeonly);
    } else {
        log<LOG_ERROR>(L"%1% || Unrecognized chi2 function %2%") % __func__ % chi2.c_str();
        abort();
    }

    // Set color palette for covar and correlation matrices
    set_matrix_palette();

    Eigen::VectorXf global_fit_result, global_fit_result_surf;
    float global_fit_chi2 = -1, global_fit_chi2_surf = -1;

    //***********************************************************************
    //***********************************************************************
    //******************** PROfile PROfile PROfile **************************
    //***********************************************************************
    //***********************************************************************

    if(*profile_command){

        GlobalFitOptions opt = GlobalFitOptions::Default;
        if(progress_bar) opt |= GlobalFitOptions::Progress;
        if(binwidth_scale) opt |= GlobalFitOptions::BinWidthScaled;
        if(!global_fixed[0] || !systs_only) opt |= GlobalFitOptions::FreqSeedPts;
        opt |= MCMC_prefit_errors ? GlobalFitOptions::MCMCPrefitErrorBand : GlobalFitOptions::PrefitErrorBand;
        opt |= GlobalFitOptions::PostFitErrorBand;
        opt |= GlobalFitOptions::Correlations;
        PROspec cv = FillSpectra(config, prop, metric->GetSysts(), metric->GetModel(), CVParams , true ,config.i_prime);
        GlobalFitResult fitres = do_a_fit(config, prop, data, metric, global_ub, global_lb, fitConfig, CVParams, cv, global_fixed, opt); 
        global_fit_chi2 = fitres.chi2;
        global_fit_result = fitres.fitter.best_fit;

        TH2D corrhist("crh", "", N_params, 0, N_params, N_params, 0, N_params);
        TH2D fraccovhist("fch", "", N_params, 0, N_params, N_params, 0, N_params);
        TH2D covhist("ch", "", N_params, 0, N_params, N_params, 0, N_params);
        std::vector<std::string> param_names;
        for(size_t i = 0; i < N_params; ++i) {
            std::string label = i < N_phys_params 
                ? metric->GetModel().pretty_param_names[i]
                : config.m_mcgen_variation_plotname_map[metric->GetSysts().spline_names[i-N_phys_params]].c_str();
            param_names.push_back(label);
            covhist.GetXaxis()->SetBinLabel(i+1, label.c_str());
            covhist.GetYaxis()->SetBinLabel(i+1, label.c_str());
            fraccovhist.GetXaxis()->SetBinLabel(i+1, label.c_str());
            fraccovhist.GetYaxis()->SetBinLabel(i+1, label.c_str());
            corrhist.GetXaxis()->SetBinLabel(i+1, label.c_str());
            corrhist.GetYaxis()->SetBinLabel(i+1, label.c_str());
            for(size_t j = 0; j < N_params; ++j) {
                covhist.SetBinContent(i+1, j+1, fitres.covmat(i,j));
                fraccovhist.SetBinContent(i+1, j+1, fitres.fraccovmat(i,j));
                corrhist.SetBinContent(i+1, j+1, fitres.corrmat(i,j));
            }
        }
        TCanvas c1;
        corrhist.SetMaximum(1);
        corrhist.SetMinimum(-1);
        covhist.SetMaximum(1);
        covhist.SetMinimum(-1);
        fraccovhist.SetMaximum(100);
        fraccovhist.SetMinimum(-100);
        //covhist.Draw("colz");
        //c1.Print((final_output_tag+"_postfit_cov.pdf").c_str());
        //fraccovhist.Draw("colz");
        //c1.Print((final_output_tag+"_postfit_fraccov.pdf").c_str());
        c1.SetLeftMargin(0.18);   
        corrhist.SetTitle("Post-Fit Correlation Matrix");
        corrhist.Draw("colz");
        gPad->Update();

        TLine line;
        line.SetLineColor(kBlack);
        line.SetLineWidth(2);
        line.DrawLine(N_phys_params, 0, N_phys_params, N_params);
        line.DrawLine(0, N_phys_params, N_params, N_phys_params);
        c1.Print((final_output_tag+"_postfit_correlation_matrix.pdf").c_str());

        log<LOG_INFO>(L"%1% || MCMC acceptance is  %2%. ") % __func__% ((double)fitres.mh->naccept /fitConfig.MCMCiter);
        fitres.mh->plot_autocorrelation(final_output_tag+"_PROfile_corrmat_mcmc_autocorrelation.pdf", param_names);

        std::string hname = "#chi^{2}/nbins = " + to_string(fitres.chi2) + "/" + to_string(config.m_num_variable_bins_total_collapsed[config.i_prime]);
        PROspec bf = FillSpectra(config, prop, metric->GetSysts(), metric->GetModel(), fitres.fitter.best_fit, true,config.i_prime);
        // Concatenated bins across all channels share no common x-axis, so use bin-index axis.
        TH1D post_hist("ph", hname.c_str(), config.m_num_variable_bins_total_collapsed[config.i_prime], 0, config.m_num_variable_bins_total_collapsed[config.i_prime]);
        TH1D pre_hist("prh", hname.c_str(), config.m_num_variable_bins_total_collapsed[config.i_prime], 0, config.m_num_variable_bins_total_collapsed[config.i_prime]);
        for(size_t i = 0; i < config.m_num_variable_bins_total_collapsed[config.i_prime]; ++i) {
            post_hist.SetBinContent(i+1, bf.Spec()(i));
            pre_hist.SetBinContent(i+1, cv.Spec()(i));
        }

        std::vector<TPaveText> texts;
        TPaveText chi2text(0.55, 0.50, 0.85, 0.58, "NDC");
        chi2text.AddText(hname.c_str());
        chi2text.SetFillColor(0);
        chi2text.SetBorderSize(0);
        chi2text.SetTextAlign(12);
        //chi2text.SetTextSize(0.035); 
        texts.push_back(chi2text);

        PlotOptions popt; 
        if(data_mc_ratio){
            popt = PlotOptions::DataMCRatio;
        } else {
            popt = PlotOptions::DataPostfitRatio;
        }
        if(binwidth_scale) popt |= PlotOptions::BinWidthScaled;
        if(area_normalized) popt |= PlotOptions::AreaNormalized;
        plot_channels((final_output_tag+"_PROfile_hists.pdf"), config, cv, bf, data, fitres.err_band, fitres.post_err_band, texts, pbounds, popt);

        TCanvas c;
        c.Print((final_output_tag+"_postfit_posteriors.pdf[").c_str());
        for(auto &h: fitres.posteriors) {
            h.Draw("hist");
            c.Print((final_output_tag+"_postfit_posteriors.pdf").c_str());
        }
        c.Print((final_output_tag+"_postfit_posteriors.pdf]").c_str());

        Eigen::VectorXf inv_sqrt_diag_nuis = fitres.spline_covariance.diagonal().array().abs().max(1e-10f).sqrt().inverse();
        Eigen::MatrixXf corrmat_nuis = inv_sqrt_diag_nuis.asDiagonal() * fitres.spline_covariance * inv_sqrt_diag_nuis.asDiagonal();

        TH2F spline_cov("postfit_corr_nuisance_only", "", corrmat_nuis.cols(), 0, corrmat_nuis.cols(), corrmat_nuis.rows(), 0, corrmat_nuis.rows());
        TH2F spline_cov_cov("postfit_cov_nuisance_only", "", fitres.spline_covariance.cols(), 0, fitres.spline_covariance.cols(), fitres.spline_covariance.rows(), 0, fitres.spline_covariance.rows());
        for(int i = 0; i < corrmat_nuis.cols(); ++i) {
            spline_cov.GetXaxis()->SetBinLabel(i+1, config.m_mcgen_variation_plotname_map[metric->GetSysts().spline_names[i]].c_str());
            spline_cov.GetYaxis()->SetBinLabel(i+1, config.m_mcgen_variation_plotname_map[metric->GetSysts().spline_names[i]].c_str());
            spline_cov_cov.GetXaxis()->SetBinLabel(i+1, config.m_mcgen_variation_plotname_map[metric->GetSysts().spline_names[i]].c_str());
            spline_cov_cov.GetYaxis()->SetBinLabel(i+1, config.m_mcgen_variation_plotname_map[metric->GetSysts().spline_names[i]].c_str());
            for(int j = 0; j < corrmat_nuis.rows(); ++j) {
                spline_cov.SetBinContent(i+1, j+1, corrmat_nuis(i,j));
                spline_cov_cov.SetBinContent(i+1, j+1, fitres.spline_covariance(i,j));
            }
        }
        spline_cov.Draw("colz");
        spline_cov.SetMaximum(1);
        spline_cov.SetMinimum(-1);

        c.Print((final_output_tag+"_postfit_correlation_matrix_nuisance_only.pdf").c_str());

        plot_mcmc_1sigma(final_output_tag+"_PROfile", config, metric->GetSysts(), metric->GetModel(), fitres.fitter.best_fit, fitres.post_param_lo, fitres.post_param_hi, !systs_only, fakedataparams);

        log<LOG_INFO>(L"%1% ||  Beginning full PROfile ") % __func__;

        if(progress_bar)scanFitConfig.progress_bar = true;

        std::vector<Eigen::VectorXf> seeds = fitres.fitter.freq_seed_points;//to be updated to v1.1.5 harmoincs [DONE]
        if(!seeds.size()) seeds.push_back(fitres.fitter.best_fit);
        PROfile profile(config, metric->GetSysts(), metric->GetModel(), *metric, myseed, scanFitConfig,
                final_output_tag+"_PROfile", fitres.chi2, !systs_only, nthread, seeds,
                fakedataparams);
        log<LOG_INFO>(L"%1% || fakedataparams for Plot (true_params/red stars): %2%") % __func__ % fakedataparams;
        profile.Plot(config, metric->GetSysts(), metric->GetModel(), *metric, myseed,
                final_output_tag+"_PROfile", !systs_only, fitres.fitter.best_fit,
                fakedataparams, fitres.spline_covariance, fitres.post_param_lo, fitres.post_param_hi);
        TFile fout((final_output_tag+"_PROfile.root").c_str(), "RECREATE");
        profile.onesig.Write("one_sigma_errs");
        pre_hist.Write("cv");
        size_t tot_offset = 0;
        for(size_t mode = 0; mode < config.m_num_modes; ++mode) {
            for(size_t det = 0; det < config.m_num_detectors; ++det) {
                for(size_t channel = 0; channel < config.m_num_channels; ++channel) {
                    size_t channel_nbins_x = config.m_channel_variable_bins[channel][config.i_prime].NBinsAlong(0);
                    // default is 1d, but catch 2d case for ybins
                    size_t channel_nbins_y = 1;
                    if(config.m_channel_variable_dims[channel][config.i_prime] == 2)  channel_nbins_y = config.m_channel_variable_bins[channel][config.i_prime].NBinsAlong(1);
                    size_t nbins_p_2dchan = channel_nbins_y*channel_nbins_x;
                    TGraphAsymmErrors preband(nbins_p_2dchan);
                    TGraphAsymmErrors postband(nbins_p_2dchan);
                    for(size_t i = 0; i < nbins_p_2dchan; ++i) {
                        float x = channel_nbins_y == 1 ? 
                            (config.m_channel_variable_bins[channel][config.i_prime].Edges(0)[i+1] + config.m_channel_variable_bins[channel][config.i_prime].Edges(0)[i])/2 :
                            i;
                        float xerr = channel_nbins_y == 1 ?
                            (config.m_channel_variable_bins[channel][config.i_prime].Edges(0)[i+1] - config.m_channel_variable_bins[channel][config.i_prime].Edges(0)[i])/2 :
                            0.5;
                        preband.SetPoint(i, x, fitres.err_band->error_point(tot_offset + i));
                        preband.SetPointEYhigh(i, fitres.err_band->error_up(tot_offset + i));
                        preband.SetPointEYlow(i, fitres.err_band->error_down(tot_offset + i));
                        preband.SetPointEXhigh(i, xerr);
                        preband.SetPointEXlow(i, xerr);
                        postband.SetPoint(i, x, fitres.post_err_band->error_point(tot_offset + i));
                        postband.SetPointEYhigh(i, fitres.post_err_band->error_up(tot_offset + i));
                        postband.SetPointEYlow(i, fitres.post_err_band->error_down(tot_offset + i));
                        postband.SetPointEXhigh(i, xerr);
                        postband.SetPointEXlow(i, xerr);
                    }
                    std::string name = config.m_mode_names[mode]+"_"+config.m_detector_names[det]+"_"+config.m_channel_names[channel];
                    preband.Write((name+"_prefit_err").c_str());
                    postband.Write((name+"_postfit_err").c_str());
                    tot_offset += nbins_p_2dchan;
                }
            }
        }
        //err_band->Write("prefit_errband");
        //post_err_band->Write("postfit_errband");
        post_hist.Write("best_fit");
        spline_cov.Write();
        spline_cov_cov.Write();
        if(global_fit_result.size() > 0) {
            bool use_phys_gfr = (size_t)global_fit_result.size() == N_phys_params + metric->GetSysts().GetNSplines();
            TH1D global_fit_hist("global_fit_result", "Global Best Fit Parameters", global_fit_result.size(), 0, global_fit_result.size());
            for(long i = 0; i < global_fit_result.size(); i++) {
                std::string pname;
                if(use_phys_gfr && i < (long)N_phys_params)
                    pname = metric->GetModel().param_names[i];
                else {
                    long idx = use_phys_gfr ? i - N_phys_params : i;
                    pname = config.m_mcgen_variation_plotname_map.at(metric->GetSysts().spline_names[idx]);
                }
                global_fit_hist.GetXaxis()->SetBinLabel(i+1, pname.c_str());
                global_fit_hist.SetBinContent(i+1, global_fit_result(i));
            }
            global_fit_hist.Write();
        }

        //***********************************************************************
        //***********************************************************************
        //******************** PROsurf PROsurf PROsurf **************************
        //***********************************************************************
        //***********************************************************************
    }
    if(*surface_command ){

        // If we haven't run global fit yet, or we did, but it excluded some parameters
        // Not sure global fits will exclude parameters now that we have the fixed pars for syst only though
        // So this logic may not work anymore.
        if(global_fit_result.size() == 0 || global_fit_result.size() != (int)N_params) {
            GlobalFitOptions opt = GlobalFitOptions::Default;
            if(progress_bar) opt |= GlobalFitOptions::Progress;
            opt |= GlobalFitOptions::FreqSeedPts;
            PROspec cv = FillSpectra(config, prop, metric->GetSysts(), metric->GetModel(), CVParams , true ,config.i_prime);
            // Should we pass in global fixed here? This mostly gets used with syst_only which would not make sense for a surface.
            GlobalFitResult fitres = do_a_fit(config, prop, data, metric, global_ub, global_lb, fitConfig, CVParams, cv, global_fixed, opt); 
            global_fit_chi2_surf = fitres.chi2;
            global_fit_result_surf = fitres.fitter.best_fit;
        } else {
            global_fit_chi2_surf = global_fit_chi2;
            global_fit_result_surf = global_fit_result;
        }
        if (grid_size.empty()) {
            grid_size = {40, 40};
        }
        if (grid_size.size() == 1) {
            grid_size.push_back(grid_size[0]); //make it square
        }

        if(*xlim_opt) {
            xlo = xlims[0];
            xhi = xlims[1];
        }
        if(*ylim_opt) {
            ylo = ylims[0];
            yhi = ylims[1];
        }

        //Define grid and Surface
        size_t xaxis_idx = 1, yaxis_idx = 0;
        if(const auto loc = std::find(model->param_names.begin(), model->param_names.end(), xvar); loc != model->param_names.end()) {
            xaxis_idx = std::distance(model->param_names.begin(), loc);

        } else if(const auto loc = std::find(variable_systs[config.i_prime].spline_names.begin(), variable_systs[config.i_prime].spline_names.end(), xvar); loc != variable_systs[config.i_prime].spline_names.end()) {
            xaxis_idx = std::distance(variable_systs[config.i_prime].spline_names.begin(), loc);
        } else {
            for(const auto &[xml_name, plot_name]: config.m_mcgen_variation_plotname_map) {
                if(xvar == plot_name) {
                    const auto loc = std::find(variable_systs[config.i_prime].spline_names.begin(), variable_systs[config.i_prime].spline_names.end(), xml_name);
                    if(loc != variable_systs[config.i_prime].spline_names.end()) {
                        xaxis_idx = std::distance(variable_systs[config.i_prime].spline_names.begin(), loc);
                    }
                    break;
                }
            }
        }
        if(const auto loc = std::find(model->param_names.begin(), model->param_names.end(), yvar); loc != model->param_names.end()) {
            yaxis_idx = std::distance(model->param_names.begin(), loc);
        } else if(const auto loc = std::find(variable_systs[config.i_prime].spline_names.begin(),variable_systs[config.i_prime].spline_names.end(), yvar); loc != variable_systs[config.i_prime].spline_names.end()) {
            yaxis_idx = std::distance(variable_systs[config.i_prime].spline_names.begin(), loc);
        } else {
            for(const auto &[xml_name, plot_name]: config.m_mcgen_variation_plotname_map) {
                if(yvar == plot_name) {
                    const auto loc = std::find(variable_systs[config.i_prime].spline_names.begin(), variable_systs[config.i_prime].spline_names.end(), xml_name);
                    if(loc != variable_systs[config.i_prime].spline_names.end()) {
                        yaxis_idx = std::distance(variable_systs[config.i_prime].spline_names.begin(), loc);
                    }
                    break;
                }
            }

        }
        size_t nbinsx = grid_size[0], nbinsy = grid_size[1];
        PROsurf surface(*metric, xaxis_idx, yaxis_idx, nbinsx, logx ? PROsurf::LogAxis : PROsurf::LinAxis, xlo, xhi,
                nbinsy, logy ? PROsurf::LogAxis : PROsurf::LinAxis, ylo, yhi);

        if(procurve_points.size()!=0){

            size_t mid = procurve_points.size() / 2;
            std::vector<float> A(procurve_points.begin(), procurve_points.begin() + mid);
            std::vector<float> B(procurve_points.begin() + mid, procurve_points.end());
            size_t Ncurvep = grid_size.front();
            log<LOG_INFO>(L"%1% || Running a PROcurve from %2% to point %3% with %4% points") % __func__ % A% B %Ncurvep;
            
            std::vector<surfOut> cpoints = surface.FillCurve(fitConfig, myseed, global_fit_chi2_surf, global_fit_result_surf, nthread, A, B, Ncurvep);
            surface.PlotCurve(config,*model,variable_systs[config.i_prime],cpoints,final_output_tag,logx,logy,xaxis_idx,yaxis_idx,A, B, Ncurvep); 
            return 0;
        }


        if(!only_brazil) {
            if(statonly)
                surface.FillSurfaceStat(config, scanFitConfig, final_output_tag+"_statonly_surface.txt", CVParams, dseed(myseed.global_rng));
            else
                surface.FillSurface(scanFitConfig, final_output_tag+"_surface.txt",myseed, global_fit_chi2_surf, global_fit_result_surf, nthread);
        }

        std::vector<float> binedges_x, binedges_y;
        // Edges are stored in model's native space (log if is_log10, linear otherwise)
        // Convert to linear for ROOT histogram bin edges
        for(size_t i = 0; i < surface.nbinsx+1; i++)
            binedges_x.push_back(model->is_log10[xaxis_idx] ? std::pow(10, surface.edges_x(i)) : surface.edges_x(i));
        for(size_t i = 0; i < surface.nbinsy+1; i++)
            binedges_y.push_back(model->is_log10[yaxis_idx] ? std::pow(10, surface.edges_y(i)) : surface.edges_y(i));

        if(xlabel == "") 
            xlabel = xaxis_idx < N_phys_params ? model->pretty_param_names[xaxis_idx] : 
                config.m_mcgen_variation_plotname_map[variable_systs[config.i_prime].spline_names[xaxis_idx]];
        if(ylabel == "") 
            ylabel = yaxis_idx < N_phys_params ? model->pretty_param_names[yaxis_idx] : 
                config.m_mcgen_variation_plotname_map[variable_systs[config.i_prime].spline_names[yaxis_idx]];
        TH2D surf("surf", (";"+xlabel+";"+ylabel).c_str(), surface.nbinsx, binedges_x.data(), surface.nbinsy, binedges_y.data());

        for(size_t i = 0; i < surface.nbinsx; i++) {
            for(size_t j = 0; j < surface.nbinsy; j++) {
                surf.SetBinContent(i+1, j+1, surface.surface(i, j));
            }
        }

        log<LOG_INFO>(L"%1% || Saving surface to %2% as TH2D named \"surf.\"") % __func__ % final_output_tag.c_str();
        TFile fout((final_output_tag+"_surf.root").c_str(), "RECREATE");
        if(!only_brazil) {
            surf.Write();
            float chisq;
            int xbin, ybin;
            std::map<std::string, float> best_fit;
            TTree tree("tree", "BestFitTree");
            tree.Branch("chi2", &chisq); 
            tree.Branch("xbin", &xbin); 
            tree.Branch("ybin", &ybin); 
            tree.Branch("best_fit", &best_fit); 

            for(const auto &res: surface.results) {
                chisq = res.chi2;
                xbin = res.binx;
                ybin = res.biny;
                // If all fit points fail
                if(!res.best_fit.size()) { tree.Fill(); continue; }
                for(size_t i = 0; i < N_phys_params; ++i) {
                    best_fit[model->param_names[i]] = res.best_fit(i);
                }
                for(size_t i = 0; i < variable_systs[config.i_prime].GetNSplines(); ++i) {
                    best_fit[variable_systs[config.i_prime].spline_names[i]] = res.best_fit(i + N_phys_params);
                }
                tree.Fill();
            }
            // TODO: Should we save the spectra as TH1s?

            tree.Write();

            TCanvas c;
            if(logy)
                c.SetLogy();
            if(logx)
                c.SetLogx();
            c.SetLogz();
            gStyle->SetPalette(kViridis);
            surf.Draw("colz");
            c.Print((final_output_tag+"_surface.pdf").c_str());
        }

        std::vector<PROsurf> brazil_band_surfaces;
        if(run_brazil && brazil_throws.size() == 0) {
            std::normal_distribution<float> d;
            PROspec cv = FillSpectra(config, prop, metric->GetSysts(), metric->GetModel(), CVParams , true,config.i_prime);

            PROspec collapsed_cv = PROspec(CollapseMatrix(config, cv.Spec()), CollapseMatrix(config, cv.Error()));
            Eigen::MatrixXf L = metric->GetSysts().DecomposeFractionalCovariance(config, cv.Spec());
            for(size_t i = 0; i < 1000; ++i) {
                Eigen::VectorXf throwp = fakeDataParams;
                Eigen::VectorXf throwC = Eigen::VectorXf::Constant(config.m_num_variable_bins_total_collapsed[config.i_prime], 0);
                for(size_t i = 0; i < metric->GetSysts().GetNSplines(); i++)
                    throwp(i+N_phys_params) = d(PROseed::global_rng);
                for(size_t i = 0; i < config.m_num_variable_bins_total_collapsed[config.i_prime]; i++)
                    throwC(i) = d(PROseed::global_rng);
                bool binned = (eventbyevent ? PROmetric::EventByEvent : PROmetric::BinnedChi2) != 0;
                PROspec shifted = FillSpectra(config, prop, metric->GetSysts(), metric->GetModel(), throwp, binned);
                PROspec newSpec = statonly_brazil ? PROspec::PoissonVariation(collapsed_cv, dseed(myseed.global_rng)) :
                    PROspec::PoissonVariation(PROspec(CollapseMatrix(config, shifted.Spec()) + L * throwC, CollapseMatrix(config, shifted.Error())), dseed(myseed.global_rng));
                PROdata data(newSpec.Spec(), newSpec.Error());
                PROmetric *metric;
                if(chi2 == "PROchi") {
                    metric = new PROchi("", config, prop, &variable_systs[config.i_prime], *model, data, eventbyevent ? PROmetric::EventByEvent : PROmetric::BinnedChi2);
                } else if(chi2 == "PROCNP") {
                    metric = new PROCNP("", config, prop, &variable_systs[config.i_prime], *model, data, eventbyevent ? PROmetric::EventByEvent : PROmetric::BinnedChi2);
                } else if(chi2 == "Poisson") {
                    metric = new PROpoisson("", config, prop, &variable_systs[config.i_prime], *model, data, eventbyevent ? PROmetric::EventByEvent : PROmetric::BinnedChi2);
                } else {
                    log<LOG_ERROR>(L"%1% || Unrecognized chi2 function %2%") % __func__ % chi2.c_str();
                    abort();
                }
                GlobalFitOptions opt = GlobalFitOptions::Default;
                if(progress_bar) opt |= GlobalFitOptions::Progress;
                opt |= GlobalFitOptions::FreqSeedPts;
                PROspec cv = FillSpectra(config, prop, metric->GetSysts(), metric->GetModel(), CVParams , true ,config.i_prime);
                GlobalFitResult fitres = do_a_fit(config, prop, data, metric, global_ub, global_lb, fitConfig, CVParams, cv, global_fixed, opt); 

                brazil_band_surfaces.emplace_back(*metric, xaxis_idx, yaxis_idx, nbinsx, logx ? PROsurf::LogAxis : PROsurf::LinAxis, xlo, xhi,
                        nbinsy, logy ? PROsurf::LogAxis : PROsurf::LinAxis, ylo, yhi);

                if(statonly)
                    brazil_band_surfaces.back().FillSurfaceStat(config, scanFitConfig, "", CVParams, dseed(myseed.global_rng));
                else
                    brazil_band_surfaces.back().FillSurface(scanFitConfig, "", myseed, fitres.chi2, fitres.fitter.best_fit, nthread);

                TH2D surf("surf", (";"+xlabel+";"+ylabel).c_str(), surface.nbinsx, binedges_x.data(), surface.nbinsy, binedges_y.data());

                for(size_t i = 0; i < surface.nbinsx; i++) {
                    for(size_t j = 0; j < surface.nbinsy; j++) {
                        surf.SetBinContent(i+1, j+1, brazil_band_surfaces.back().surface(i, j));
                    }
                }
                surf.Write(("brazil_throw_surf_"+std::to_string(i)).c_str());

                // WARNING: Metric reference stored in surface. DO NOT USE IT AFTER THIS POINT.
                delete metric;
                if(single_brazil) break;
            }
        } else if(run_brazil) { // if brazil_thows.size() > 0
            for(const std::string &in: brazil_throws) {
                brazil_band_surfaces.emplace_back(*metric, xaxis_idx, yaxis_idx, nbinsx, logx ? PROsurf::LogAxis : PROsurf::LinAxis, xlo, xhi,
                        nbinsy, logy ? PROsurf::LogAxis : PROsurf::LinAxis, ylo, yhi);

                TFile fin(in.c_str());
                // TODO: Check that axes and labels are the same
                TH2D *surf = fin.Get<TH2D>("brazil_throw_surf_0");
                if(!surf) {
                    log<LOG_ERROR>(L"%1% || Could not find a TH2D called 'surf' in the file %2%. Skipping this file.")
                        % __func__ % in.c_str();
                    continue;
                    //return EXIT_FAILURE;
                }
                for(size_t i = 0; i < surface.nbinsx; ++i) {
                    for(size_t j = 0; j < surface.nbinsy; ++j) {
                        brazil_band_surfaces.back().surface(i,j) = surf->GetBinContent(i+1,j+1);
                    }
                }
            }
        }

        if(run_brazil && !single_brazil) {
            TH2D surf16("surf16", (";"+xlabel+";"+ylabel).c_str(), surface.nbinsx, binedges_x.data(), surface.nbinsy, binedges_y.data());
            TH2D surf84("surf84", (";"+xlabel+";"+ylabel).c_str(), surface.nbinsx, binedges_x.data(), surface.nbinsy, binedges_y.data());
            TH2D surf98("surf98", (";"+xlabel+";"+ylabel).c_str(), surface.nbinsx, binedges_x.data(), surface.nbinsy, binedges_y.data());
            TH2D surf02("surf02", (";"+xlabel+";"+ylabel).c_str(), surface.nbinsx, binedges_x.data(), surface.nbinsy, binedges_y.data());
            TH2D surf50("surf50", (";"+xlabel+";"+ylabel).c_str(), surface.nbinsx, binedges_x.data(), surface.nbinsy, binedges_y.data());

            for(size_t i = 0; i < surface.nbinsx; ++i) {
                for(size_t j = 0; j < surface.nbinsy; ++j) {
                    std::vector<float> values;
                    for(const auto &bbsurf: brazil_band_surfaces)
                        values.push_back(bbsurf.surface(i,j));
                    std::sort(values.begin(), values.end());
                    surf02.SetBinContent(i+1, j+1, values[(size_t)(0.023 * values.size())]);
                    surf16.SetBinContent(i+1, j+1, values[(size_t)(0.159 * values.size())]);
                    surf50.SetBinContent(i+1, j+1, values[(size_t)(0.500 * values.size())]);
                    surf84.SetBinContent(i+1, j+1, values[(size_t)(0.841 * values.size())]);
                    surf98.SetBinContent(i+1, j+1, values[(size_t)(0.977 * values.size())]);
                }
            }

            fout.cd();
            surf02.Write();
            surf16.Write();
            surf50.Write();
            surf84.Write();
            surf98.Write();
            if(brazil_throws.size() != 0) {
                int i = 0;
                for(const auto &bbsurf: brazil_band_surfaces) {
                    TH2D *surf = new TH2D(("brazil_throw_surf_"+std::to_string(i)).c_str(),(";"+xlabel+";"+ylabel).c_str(),surface.nbinsx, binedges_x.data(), surface.nbinsy, binedges_y.data())     ;
                    for(size_t j = 0; j < surface.nbinsx; ++j) {
                        for(size_t k = 0; k < surface.nbinsy; ++k) {
                            surf->SetBinContent(j+1,k+1, bbsurf.surface(j,k));
                        }
                    }
                    surf->Write();
                    ++i;
                }
            }
        }

        //***********************************************************************
        //***********************************************************************
        //******************** PROplot PROplot PROplot **************************
        //***********************************************************************
        //***********************************************************************
    }
    if(*proplot_command){

        log<LOG_INFO>(L"%1% || Making a PROsyst thats full covariance for future error bar creation (might be slow) ")% __func__ ;
        PROsyst allcovsyst = variable_systs[config.i_prime].allsplines2cov(config, prop, *model, CVParams, dseed(PROseed::global_rng));

        // --bkg-subtract: resolve the wildcard once. The same matched subchannel
        // list is used for every variable in this block; bkg_full / bkg_collapsed
        // are rebuilt per variable just before each plot_channels call. Empty
        // bkg_subchannels short-circuits all subtraction below.
        std::vector<size_t> bkg_subchannels;
        if (!bkg_subtract_pattern.empty()) {
            bkg_subchannels = find_subchannels_by_pattern(config, bkg_subtract_pattern);
            if (bkg_subchannels.empty()) {
                log<LOG_WARNING>(L"%1% || --bkg-subtract pattern '%2%' matched no subchannels; ignoring.")
                    % __func__ % bkg_subtract_pattern.c_str();
            } else {
                log<LOG_INFO>(L"%1% || --bkg-subtract '%2%' matched %3% subchannel(s).")
                    % __func__ % bkg_subtract_pattern.c_str() % bkg_subchannels.size();
            }
        }
        const bool do_bkg_subtract = !bkg_subchannels.empty();

        PlotOptions opt = PlotOptions::CVasStack;
        std::vector<TPaveText> notext;
        if(binwidth_scale) opt |= PlotOptions::BinWidthScaled;
        if(area_normalized) opt |= PlotOptions::AreaNormalized;
        std::vector<PROspec> variable_cvs;
        for(size_t io = 0; io < config.m_num_variables; ++io) {

            variable_cvs.push_back(FillSpectra(config, prop, variable_systs[config.i_prime],*model,CVParams, !eventbyevent, io));

            // Local subtracted copy for the CV plot. variable_cvs is preserved
            // intact so downstream consumers (fractional-systematics breakdown,
            // error-band plot at L2156) see the unsubtracted CV unless they
            // also subtract.
            PROspec cv_plot = variable_cvs.back();
            if (do_bkg_subtract) {
                Eigen::VectorXf bkg_full = build_subchannel_mask_spec(
                    config, cv_plot, bkg_subchannels, io);
                cv_plot.Spec() -= bkg_full;
            }
            plot_channels(final_output_tag+"_PROplot_Variable_"+std::to_string(io)+"_CV.pdf", config, cv_plot, {}, {}, {}, {}, notext, pbounds, opt, io);
        }

        std::string filename = final_output_tag+"_fractional_systematics.pdf";
        plotPriorFractionalSystematicBreakdown(config, variable_cvs[config.i_prime], allcovsyst, filename,config.i_prime);
        std::string rfilename = final_output_tag+"_ratio_fractional_systematics.pdf";
        if(config.m_num_detectors > 1)
            plotPriorFractionalSystematicRatios(config, variable_cvs[config.i_prime], allcovsyst, rfilename,config.i_prime);

        std::vector<std::map<std::string, std::unique_ptr<TH1D>>> other_hists;
        for(size_t io = 0; io < config.m_num_variables; ++io) {
            other_hists.push_back(getCV1DHists(variable_cvs[io], config, binwidth_scale, io));
        }
        
        // DetVar plotting (uses combined DetVar propellers binary)
        if(config.m_has_detvar_section) {
            log<LOG_INFO>(L"%1% || Plotting detector variations...") % __func__;

            std::string dvAllPropsBin = analysis_tag + "_detvar_props.bin";
            std::map<std::string, PROpeller> plot_dvprops;
            std::vector<PROspec> detvar_specs;
            std::vector<std::string> detvar_names;
            std::vector<int> detvar_binning;
            // Matched pairs for _DetVarOverlapping PDF (var file index -> matched cv+var specs)
            struct MatchedPair { PROspec cv; std::map<int, PROspec> vars; };
            std::map<size_t, MatchedPair> matched_pairs;

            if(!std::filesystem::exists(dvAllPropsBin)) {
                log<LOG_ERROR>(L"%1% || DetVar combined binary not found: %2%. Run 'process' first.") % __func__ % dvAllPropsBin.c_str();
            } else {
                uint32_t loaded_detvar_hash = loadDetVarProps(plot_dvprops, dvAllPropsBin);
                if(config.detvar_hash != loaded_detvar_hash) {
                    log<LOG_WARNING>(L"%1% || WARNING config detvar_hash (%2%) and binary detvar_hash (%3%) not compatible. DetVar plots may be stale.") % __func__ % config.detvar_hash % loaded_detvar_hash;
                }

                // Precompute section CV index by section for matching
                std::map<size_t, size_t> plot_cv_idx_by_section; // section_idx -> detvar file index
                for(size_t idv = 0; idv < config.GetNumDetVarFiles(); ++idv) {
                    if(config.m_detvar_files[idv].is_cv)
                        plot_cv_idx_by_section[config.m_detvar_files[idv].section_index] = idv;
                }

                std::vector<size_t> skip;
                for(size_t idv = 0; idv < config.GetNumDetVarFiles(); ++idv) {
                    if(skip.size() && std::find(skip.begin(), skip.end(), idv) != skip.end()) continue;
                    const std::string& name = config.m_detvar_files[idv].name;
                    const std::string key = DetVarKey(config, idv);
                    if(plot_dvprops.count(key) == 0) {
                        log<LOG_ERROR>(L"%1% || DetVar entry '%2%' not found in combined binary. Run 'process' first.") % __func__ % name.c_str();
                        break;
                    }
                    std::map<int, size_t> syst_files;
                    auto find_fn = [&name](const PROconfig::DetVarFile &dvf) { return dvf.name == name; };
                    auto it = config.m_detvar_files.begin() + idv;
                    while((it = std::find_if(it, config.m_detvar_files.end(), find_fn))
                            != std::end(config.m_detvar_files)) {
                        size_t i = std::distance(config.m_detvar_files.begin(), it);
                        syst_files[it->knobval] = i;
                        skip.push_back(i);
                        it++;
                    }

                    int binningIndex = config.m_mcgen_variation_binning_map.count(name) ? config.m_mcgen_variation_binning_map.at(name) : config.i_prime;
                    if(binningIndex < 0 || binningIndex >= (int)config.m_num_variables)
                        binningIndex = config.i_prime;

                    std::map<int, const PROpeller*> props;
                    MatchedPair mp;
                    for(auto &[kv, f] : syst_files) {
                        PROconfig dvconfig = config.BuildDetVarConfig(f);
                        const std::string key = DetVarKey(config, f);
                        PROpeller& dvprop = plot_dvprops.at(key);

                        std::unique_ptr<PROmodel> dv_model = std::make_unique<NullModel>(dvprop);
                        PROsyst dvsysts;
                        Eigen::VectorXf dvparams = Eigen::VectorXf::Constant(dv_model->nparams, 0);

                        // Always use full spec for _DetVarFull PDF
                        PROspec full_spec = FillSpectra(dvconfig, dvprop, dvsysts, *dv_model, dvparams, !eventbyevent, binningIndex);
                        mp.vars[kv] = full_spec;
                        props[kv] = &dvprop;
                        detvar_specs.push_back(full_spec);
                        detvar_names.push_back(name);
                        detvar_binning.push_back(binningIndex);
                    }

                    // For variation files, build matched pair for _DetVarOverlapping PDF
                    if(!config.m_detvar_files[idv].is_cv) {
                        size_t sec = config.m_detvar_files[idv].section_index;
                        auto cv_it = plot_cv_idx_by_section.find(sec);
                        if(cv_it != plot_cv_idx_by_section.end()) {
                            PROpeller& cvprop_plot = plot_dvprops.at(DetVarKey(config, cv_it->second));
                            PROconfig cvconfig = config.BuildDetVarConfig(cv_it->second);
                            std::unique_ptr<PROmodel> cv_model = std::make_unique<NullModel>(cvprop_plot);
                            Eigen::VectorXf cvparams = Eigen::VectorXf::Constant(cv_model->nparams, 0);
                            mp.cv = FillSpectra(cvconfig, cvprop_plot, PROsyst(), *cv_model, cvparams, !eventbyevent, binningIndex);
                            if(BuildDetVarMatchedSpecs(cvprop_plot, props, binningIndex,
                                                       (int)config.m_num_variable_bins_total[binningIndex],
                                                       mp.cv, mp.vars)) {
                                // Undo POT scaling from both CV and variation matched spectra so
                                // the overlapping plot shows raw event-weight units (no POT scaling)
                                const double det_pot_ov = config.m_det_pot[0];
                                const double cv_pot_ov = config.m_detvar_files[cv_it->second].pot;
                                if(det_pot_ov > 0.0 && cv_pot_ov > 0.0) {
                                    const float cv_unscale_ov = (float)(cv_pot_ov / det_pot_ov);
                                    mp.cv.Spec() *= cv_unscale_ov;
                                    mp.cv.Error() *= cv_unscale_ov;
                                    for(auto &[kv_plot, var_file_idx_plot] : syst_files) {
                                        const double var_pot_ov = config.m_detvar_files[var_file_idx_plot].pot;
                                        if(var_pot_ov > 0.0) {
                                            const float var_unscale_ov = (float)(var_pot_ov / det_pot_ov);
                                            mp.vars[kv_plot].Spec() *= var_unscale_ov;
                                            mp.vars[kv_plot].Error() *= var_unscale_ov;
                                        }
                                    }
                                }
                                matched_pairs[idv] = std::move(mp);
                                log<LOG_INFO>(L"%1% || DetVar plot '%2%': matched pair built for Overlapping PDF") % __func__ % name.c_str();
                            }
                        }
                    }
                }
            }

            if(detvar_specs.size() == config.GetNumDetVarFiles()) {
                // Build per-section CV hist maps (each DetVarSection has its own CV)
                std::map<size_t, std::map<std::string, std::unique_ptr<TH1D>>> cv_hists_by_section;
                bool any_cv_missing = false;
                for(size_t i = 0; i < config.m_detvar_files.size(); ++i) {
                    if(config.m_detvar_files[i].is_cv) {
                        size_t sec = config.m_detvar_files[i].section_index;
                        cv_hists_by_section[sec] = getCV1DHists(detvar_specs[i], config, binwidth_scale, detvar_binning[i]);
                    }
                }
                if(cv_hists_by_section.empty()) {
                    log<LOG_ERROR>(L"%1% || ERROR: No CV files found in DetVar files!") % __func__;
                    any_cv_missing = true;
                }
                if(!any_cv_missing) {
                    TCanvas detvar_canvas;
                    std::string detvar_pdf = final_output_tag + "_PROplot_DetVarFull.pdf";
                    detvar_canvas.Print((detvar_pdf + "[").c_str(), "pdf");

                    size_t global_subchannel_index = 0;
                        for(size_t im = 0; im < config.m_num_modes; im++){
                            for(size_t id = 0; id < config.m_num_detectors; id++){
                                for(size_t ic = 0; ic < config.m_num_channels; ic++){
                                    // Direct DetVar plot: show detvar_cv and detvar_variation spectra directly
                                    int colors[] = {kRed, kBlue, kGreen+2, kMagenta, kCyan+1, kOrange+1, kViolet+1, kTeal+1};
                                    int ncolors = sizeof(colors)/sizeof(colors[0]);

                                    // Per-section pages: draw only variations belonging to the same section as the CV
                                    for(const auto& [sec_idx, sec_cv_hists] : cv_hists_by_section) {
                                        TH1D* cv_total = nullptr;
                                        for(size_t sc = 0; sc < config.m_num_subchannels[ic]; sc++) {
                                            const std::string& subchannel_name = config.m_fullnames[global_subchannel_index + sc];
                                            auto cv_it = sec_cv_hists.find(subchannel_name);
                                            if(cv_it != sec_cv_hists.end()) {
                                                if(!cv_total)
                                                    cv_total = (TH1D*)cv_it->second->Clone(("cv_total_sec" + std::to_string(sec_idx)).c_str());
                                                else
                                                    cv_total->Add(&*(cv_it->second));
                                            }
                                        }
                                        if(!cv_total) continue;

                                        cv_total->SetLineColor(kBlack);
                                        cv_total->SetLineWidth(3);
                                        cv_total->SetFillColor(kWhite);
                                        cv_total->SetFillStyle(0);
                                        cv_total->SetTitle((config.m_mode_names[im] + " " + config.m_detector_names[id] + " " + config.m_channel_names[ic] + " DetVar (sec " + std::to_string(sec_idx) + ")").c_str());
                                        {
                                            std::string chan_unit = config.GetChannelUnit(ic, config.i_prime);
                                            std::string ytitle;
                                            if(!binwidth_scale) {
                                                ytitle = "Events";
                                            } else if(chan_unit.empty()) {
                                                ytitle = "Events/unit";
                                            } else {
                                                ytitle = "Events/" + chan_unit;
                                            }
                                            cv_total->GetYaxis()->SetTitle(ytitle.c_str());
                                        }

                                        float ymax = cv_total->GetMaximum();

                                        // Build variation totals for this section only
                                        std::vector<TH1D*> var_totals;
                                        std::vector<std::string> var_labels;
                                        int color_idx = 0;

                                        for(size_t idv = 0; idv < detvar_specs.size(); ++idv) {
                                            if(config.m_detvar_files[idv].is_cv) continue;
                                            if(config.m_detvar_files[idv].section_index != sec_idx) continue;

                                            std::map<std::string, std::unique_ptr<TH1D>> var_hists = getCV1DHists(detvar_specs[idv], config, binwidth_scale, detvar_binning[idv]);

                                            TH1D* var_total = nullptr;
                                            for(size_t sc = 0; sc < config.m_num_subchannels[ic]; sc++) {
                                                const std::string& subchannel_name = config.m_fullnames[global_subchannel_index + sc];
                                                auto var_it = var_hists.find(subchannel_name);
                                                if(var_it != var_hists.end()) {
                                                    if(!var_total)
                                                        var_total = (TH1D*)var_it->second->Clone(("var_" + detvar_names[idv]).c_str());
                                                    else
                                                        var_total->Add(&*(var_it->second));
                                                }
                                            }

                                            if(var_total) {
                                                var_total->SetLineColor(colors[color_idx % ncolors]);
                                                var_total->SetLineWidth(2);
                                                var_total->SetFillColor(kWhite);
                                                var_total->SetFillStyle(0);
                                                if(var_total->GetMaximum() > ymax) ymax = var_total->GetMaximum();
                                                var_totals.push_back(var_total);
                                                var_labels.push_back(detvar_names[idv]);
                                            }
                                            color_idx++;
                                        }

                                        cv_total->SetMaximum(ymax * 1.15);
                                        cv_total->Draw("hist");

                                        std::unique_ptr<TLegend> leg = std::make_unique<TLegend>(0.55, 0.65, 0.89, 0.89);
                                        leg->SetFillStyle(0);
                                        leg->SetLineWidth(0);
                                        leg->AddEntry(cv_total, ("DetVar CV (sec " + std::to_string(sec_idx) + ")").c_str(), "l");

                                        for(size_t vi = 0; vi < var_totals.size(); ++vi) {
                                            var_totals[vi]->Draw("hist same");
                                            leg->AddEntry(var_totals[vi], var_labels[vi].c_str(), "l");
                                        }
                                        leg->Draw("same");

                                        detvar_canvas.Print(detvar_pdf.c_str(), "pdf");

                                        delete cv_total;
                                        for(auto* h : var_totals) delete h;
                                    }

                                    global_subchannel_index += config.m_num_subchannels[ic];
                                }
                            }
                        }
                    detvar_canvas.Print((detvar_pdf + "]").c_str(), "pdf");
                    log<LOG_INFO>(L"%1% || DetVar full plots saved to %2%") % __func__ % detvar_pdf.c_str();

                    // _DetVarOverlapping PDF: one page per channel × variation, matched CV vs matched var
                    if(!matched_pairs.empty()) {
                        TCanvas ov_canvas;
                        std::string ov_pdf = final_output_tag + "_PROplot_DetVarOverlapping.pdf";
                        ov_canvas.Print((ov_pdf + "[").c_str(), "pdf");

                        size_t ov_global_subchannel_index = 0;
                        for(size_t im = 0; im < config.m_num_modes; im++){
                            for(size_t id = 0; id < config.m_num_detectors; id++){
                                for(size_t ic = 0; ic < config.m_num_channels; ic++){
                                    for(size_t idv = 0; idv < detvar_specs.size(); ++idv) {
                                        if(config.m_detvar_files[idv].is_cv) continue;
                                        auto mp_it = matched_pairs.find(idv);
                                        if(mp_it == matched_pairs.end()) continue;

                                        const MatchedPair& mp = mp_it->second;
                                        std::map<std::string, std::unique_ptr<TH1D>> cv_hists_ov = getCV1DHists(mp.cv, config, binwidth_scale, detvar_binning[idv]);
                                        std::map<int, std::map<std::string, std::unique_ptr<TH1D>>> var_hists_ov;
                                        for(auto &[kv, vspec] : mp.vars)
                                            var_hists_ov[kv] = getCV1DHists(vspec, config, binwidth_scale, detvar_binning[idv]);

                                        TH1D* cv_total_ov = nullptr;
                                        std::map<int, TH1D*> var_total_ov;
                                        for(size_t sc = 0; sc < config.m_num_subchannels[ic]; sc++) {
                                            const std::string& subchannel_name = config.m_fullnames[ov_global_subchannel_index + sc];
                                            auto cv_hit = cv_hists_ov.find(subchannel_name);
                                            if(cv_hit != cv_hists_ov.end()) {
                                                if(!cv_total_ov) cv_total_ov = (TH1D*)cv_hit->second->Clone("cv_matched_ov_total");
                                                else cv_total_ov->Add(&*(cv_hit->second));
                                            }
                                            for(auto &[kv, specs] : var_hists_ov) {
                                                auto var_hit = specs.find(subchannel_name);
                                                if(var_hit != specs.end()) {
                                                    if(var_total_ov.count(kv) == 0) 
                                                        var_total_ov[kv] 
                                                            = (TH1D*)var_hit->second->Clone("var_matched_ov_total");
                                                    else 
                                                        var_total_ov[kv]->Add(&*(var_hit->second));
                                                }
                                            }
                                        }

                                        if(cv_total_ov && var_total_ov.size()) {
                                            cv_total_ov->SetLineColor(kBlack);
                                            cv_total_ov->SetLineWidth(3);
                                            cv_total_ov->SetFillColor(kWhite);
                                            cv_total_ov->SetFillStyle(0);
                                            std::vector<double> maxs;
                                            for(auto &[kv, h] : var_total_ov) {
                                                h->SetLineWidth(2);
                                                h->SetFillColor(kWhite);
                                                h->SetFillStyle(0);
                                                maxs.push_back(h->GetMaximum());
                                            }

                                            float ymax_ov = std::max(cv_total_ov->GetMaximum(), *std::max_element(maxs.begin(), maxs.end()));
                                            cv_total_ov->SetMaximum(ymax_ov * 1.15);
                                            std::string ov_title = config.m_mode_names[im] + " " + config.m_detector_names[id] + " " + config.m_channel_names[ic] + " " + detvar_names[idv] + " (Matched)";
                                            cv_total_ov->SetTitle(ov_title.c_str());
                                            {
                                                std::string chan_unit = config.GetChannelUnit(ic, config.i_prime);
                                                std::string ytitle;
                                                if(!binwidth_scale) {
                                                    ytitle = "Events";
                                                } else if(chan_unit.empty()) {
                                                    ytitle = "Events/unit";
                                                } else {
                                                    ytitle = "Events/" + chan_unit;
                                                }
                                                cv_total_ov->GetYaxis()->SetTitle(ytitle.c_str());
                                            }

                                            std::unique_ptr<TLegend> ov_leg = std::make_unique<TLegend>(0.55, 0.75, 0.89, 0.89);
                                            ov_leg->SetFillStyle(0);
                                            ov_leg->SetLineWidth(0);

                                            int ov_var_colors[] = {kRed, kBlue, kGreen+2, kMagenta, kCyan+1, kOrange+1, kViolet+1, kTeal+1};
                                            int n_ov_var_colors = sizeof(ov_var_colors)/sizeof(ov_var_colors[0]);
                                            int ov_color_idx = 0;
                                            cv_total_ov->Draw("hist");
                                            ov_leg->AddEntry(cv_total_ov, "Matched CV", "l");
                                            for(auto &[kv, h] : var_total_ov) {
                                                h->SetLineColor(ov_var_colors[ov_color_idx % n_ov_var_colors]);
                                                ++ov_color_idx;
                                                h->Draw("hist same");
                                                ov_leg->AddEntry(h, (detvar_names[idv]+" "+std::to_string(kv)).c_str(), "l");
                                            }

                                            ov_leg->Draw("same");

                                            ov_canvas.Print(ov_pdf.c_str(), "pdf");

                                            delete cv_total_ov;
                                            //delete var_total_ov;
                                        } else {
                                            if(cv_total_ov) delete cv_total_ov;
                                            //if(var_total_ov) delete var_total_ov;
                                        }
                                    }
                                    ov_global_subchannel_index += config.m_num_subchannels[ic];
                                }
                            }
                        }
                        ov_canvas.Print((ov_pdf + "]").c_str(), "pdf");
                        log<LOG_INFO>(L"%1% || DetVar overlapping plots saved to %2%") % __func__ % ov_pdf.c_str();
                        set_matrix_palette();
                    }
                }
            }
        }

        TCanvas c;
        if(fake_data_osc_params.size()) {

            c.Print((final_output_tag +"_PROplot_Osc.pdf"+ "[").c_str(), "pdf");

            PROspec osc_spec = FillSpectra(config, prop, variable_systs[config.i_prime], *model, fakeDataParams, !eventbyevent,config.i_prime );
            std::map<std::string, std::unique_ptr<TH1D>> osc_hists = getCV1DHists(osc_spec, config, binwidth_scale);
            size_t global_subchannel_index = 0;
            for(size_t im = 0; im < config.m_num_modes; im++){
                for(size_t id =0; id < config.m_num_detectors; id++){
                    for(size_t ic = 0; ic < config.m_num_channels; ic++){
                        TH1D* osc_hist = NULL;
                        TH1D* cv_hist = NULL;
                        for(size_t sc = 0; sc < config.m_num_subchannels[ic]; sc++){
                            const std::string& subchannel_name  = config.m_fullnames[global_subchannel_index];
                            const auto &h = other_hists[config.i_prime][subchannel_name];
                            const auto &o = osc_hists[subchannel_name];
                            if(sc == 0) {
                                cv_hist = (TH1D*)h->Clone();
                                osc_hist = (TH1D*)o->Clone();
                            } else {
                                cv_hist->Add(&*h);
                                osc_hist->Add(&*o);
                            }
                            ++global_subchannel_index;
                        }
                        {
                            std::string chan_unit = config.GetChannelUnit(ic, config.i_prime);
                            std::string ytitle;
                            if(!binwidth_scale) {
                                ytitle = "Events";
                            } else if(chan_unit.empty()) {
                                ytitle = "Events/unit";
                            } else {
                                ytitle = "Events/" + chan_unit;
                            }
                            cv_hist->GetYaxis()->SetTitle(ytitle.c_str());
                        }
                        if(area_normalized) {
                            cv_hist->GetYaxis()->SetTitle("Area Normalized");
                            cv_hist->Scale(1.0 / cv_hist->Integral());
                            osc_hist->Scale(1.0 / osc_hist->Integral());
                        }
                        cv_hist->SetTitle((config.m_mode_names[im]  +" "+ config.m_detector_names[id]+" "+ config.m_channel_names[ic]).c_str());
                        cv_hist->GetXaxis()->SetTitle("");
                        cv_hist->SetLineColor(kBlack);
                        cv_hist->SetFillColor(kWhite);
                        cv_hist->SetFillStyle(0);
                        osc_hist->SetLineColor(kBlue);
                        osc_hist->SetFillColor(kWhite);
                        osc_hist->SetFillStyle(0);
                        cv_hist->SetLineWidth(3);
                        osc_hist->SetLineWidth(3);
                        TH1D *rat = (TH1D*)osc_hist->Clone();
                        rat->Divide(cv_hist);
                        rat->SetTitle("");
                        rat->GetYaxis()->SetTitle("Ratio");
                        TH1D *one = (TH1D*)rat->Clone();
                        one->Divide(one);
                        one->SetLineColor(kBlack);
                        one->GetYaxis()->SetTitle("Ratio");

                        std::unique_ptr<TLegend> leg = std::make_unique<TLegend>(0.59,0.89,0.59,0.89);
                        leg->SetFillStyle(0);
                        leg->SetLineWidth(0);
                        leg->AddEntry(cv_hist, "No Oscillations", "l");
                        std::string oscstr = "";//"#splitline{Oscilations:}{";
                        for(size_t j=0;j<model->nparams;j++){
                            float val_maybe_log = model->is_log10[j] ? std::pow(10.0f, fake_data_osc_param_vector(j)) : fake_data_osc_param_vector(j);
                            oscstr+=model->pretty_param_names[j]+ " : "+ to_string_prec(val_maybe_log,3) +" "+model->pretty_param_units[j] + (j==0 ? ", " : "" );
                        }
                        //oscstr+="}";

                        leg->AddEntry(osc_hist, oscstr.c_str(), "l");

                        TPad p1("p1", "p1", 0, 0.25, 1, 1);
                        p1.SetBottomMargin(0);
                        p1.cd();
                        cv_hist->Draw("hist");
                        osc_hist->Draw("hist same");
                        cv_hist->SetMaximum(std::max(cv_hist->GetMaximum(),osc_hist->GetMaximum())*1.1);
                        leg->Draw("same");

                        TPad p2("p2", "p2", 0, 0, 1, 0.25);
                        p2.SetTopMargin(0);
                        p2.SetBottomMargin(0.3);
                        p2.cd();
                        one->GetYaxis()->SetTitleSize(0.1);
                        one->GetYaxis()->SetLabelSize(0.1);
                        one->GetXaxis()->SetTitleSize(0.1);
                        one->GetXaxis()->SetLabelSize(0.1);
                        one->GetYaxis()->SetTitleOffset(0.5);
                        one->Draw("hist");
                        one->SetMaximum(rat->GetMaximum()*1.2);
                        one->SetMinimum(rat->GetMinimum()*0.8);
                        rat->Draw("hist same");

                        c.cd();
                        p1.Draw();
                        p2.Draw();

                        c.Print((final_output_tag+"_PROplot_Osc.pdf").c_str(), "pdf");

                        delete cv_hist;
                        delete osc_hist;
                    }
                }
            }
            c.Print((final_output_tag+"_PROplot_Osc.pdf" + "]").c_str(), "pdf");

        }

        //Now some covariances
        std::map<std::string, std::unique_ptr<TH2D>> matrices = covarianceTH2D(allcovsyst, config, variable_cvs[config.i_prime]);
        c.Print((final_output_tag+"_PROplot_Covar.pdf" + "[").c_str(), "pdf");

        std::vector<std::string> first_plots = {"collapsed_total_cor","collapsed_total_frac_cov","total_cor","total_frac_cov"};

        for(const auto &name: first_plots){
            auto &mat = matrices.at(name);
            mat->Draw("colz");
            TText *t = new TText();
            t->SetNDC();                
            t->SetTextFont(42);                          
            t->SetTextSize(0.03);      
            t->SetTextAlign(33);        
            std::string pv = "PROfit v"+std::string(PROJECT_VERSION_STR);
            t->DrawText(0.895, 0.955, pv.c_str()); 
            c.Print((final_output_tag+"_PROplot_Covar.pdf").c_str(), "pdf");
        }


        for(const auto &[name, mat]: matrices) {
            if (std::find(first_plots.begin(), first_plots.end(), name) != first_plots.end())continue;
            mat->Draw("colz");
            TText *t = new TText();
            t->SetNDC();                
            t->SetTextFont(42);                          
            t->SetTextSize(0.03);      
            t->SetTextAlign(33);        
            std::string pv = "PROfit v"+std::string(PROJECT_VERSION_STR);
            t->DrawText(0.895, 0.955, pv.c_str()); 
            c.Print((final_output_tag+"_PROplot_Covar.pdf").c_str(), "pdf");
        }
        c.Print((final_output_tag+"_PROplot_Covar.pdf" + "]").c_str(), "pdf");

        //errorband
        std::unique_ptr<PROmetric> allcov_metric(metric->Clone());
        allcov_metric->override_systs(allcovsyst);
        std::vector<std::vector<TPaveText>> other_channel_chitexts; 

        for(size_t io = 0; io < config.m_num_variables; ++io) {

            log<LOG_INFO>(L"%1% || On Variable number %2%:") % __func__ % io ;
            int global_channel_index = 0;
            std::vector<TPaveText> channel_chitexts;

            //Currently metrics are for i_prime only. TO BE DONE

            for(size_t im = 0; im < config.m_num_modes; im++){
                for(size_t id =0; id < config.m_num_detectors; id++){
                    for(size_t ic = 0; ic < config.m_num_channels; ic++){
                        TPaveText chi2text(0.59, 0.50, 0.89, 0.59, "NDC");
                        if(io==config.i_prime){
                            log<LOG_INFO>(L"%1% || On channel %2%:") % __func__ % global_channel_index ;
                            double chival = allcov_metric->getSingleChannelChi(global_channel_index, variable_cvs[io], io);
                            int ndf = config.m_channel_variable_bins[ic][io].NBinsAlong(0) - bool(opt&PlotOptions::AreaNormalized);
                            log<LOG_INFO>(L"%1% || -- the datamc chi^2/ndof is %2%/%3% .") % __func__ % chival % ndf;
                            chi2text.AddText(("#chi^{2}/ndf = "+to_string_prec(chival,2)+"/"+std::to_string(ndf)).c_str());
                            chi2text.SetFillColor(0);
                            chi2text.SetBorderSize(0);
                            chi2text.SetTextAlign(12);
                            channel_chitexts.push_back(chi2text);
                        }else{
                            chi2text.AddText("");
                        }
                        // For now don't add chi2text to non-prime variables
                        // We just use an empty string anyway and there's a weird
                        // bug that shows up in the ErrorBand plots with this.
                        // (They show "A line segment" in the space where the chi2 would be.)
                        //chi2text.SetFillColor(0);
                        //chi2text.SetBorderSize(0);
                        //chi2text.SetTextAlign(12);
                        //channel_chitexts.push_back(chi2text);
                        global_channel_index++;
                    }
                }
            }
            other_channel_chitexts.push_back(channel_chitexts);
        }


        std::vector<PROerrorbar> other_err_bands;
        for(size_t io = 0; io < config.m_num_variables; ++io) {
            if(!config.m_channel_variable_plot_bool.at(io)) continue; // For now skip the L/E 250 bin. 
            other_err_bands.push_back(getErrorBand(config, prop, variable_systs[io], *model, variable_cvs[io], CVParams, binwidth_scale, io));

            // --bkg-subtract: shift CV, data, and error band by the bkg CV.
            // Var(X - const) = Var(X), so covariance / band width are unchanged;
            // only the central positions slide. The bkg's systematic uncertainty
            // remains embedded in `covariance` and therefore in error_up/down
            // around the shifted point.
            PROspec     cv_plot      = variable_cvs[io];
            PROdata     data_plot    = variable_data[io];
            PROerrorbar errband_plot = other_err_bands.back();
            if (do_bkg_subtract) {
                Eigen::VectorXf bkg_full      = build_subchannel_mask_spec(
                    config, cv_plot, bkg_subchannels, io);
                // Pass `io` explicitly: the 1-arg CollapseMatrix uses
                // config.i_prime, which is wrong when this loop iterates over
                // multiple plottable variables (io may differ from i_prime).
                Eigen::VectorXf bkg_collapsed = CollapseMatrix(config, bkg_full, io);
                cv_plot.Spec() -= bkg_full;
                // PROdata holds a const spec; rebuild via the (spec, error) ctor.
                data_plot = PROdata(Eigen::VectorXf(data_plot.Spec() - bkg_collapsed),
                                    data_plot.Error());
                errband_plot.error_point -= bkg_collapsed;
            }
            plot_channels(final_output_tag+"_PROplot_Variable_"+std::to_string(io)+"_ErrorBand.pdf", config, cv_plot, {}, data_plot,
                    errband_plot, {}, other_channel_chitexts[io], pbounds, opt | PlotOptions::DataMCRatio, io);
        }



        if(with_splines) {

            c.Print((final_output_tag+"_PROplot_Spline.pdf" + "[").c_str(), "pdf");

            std::map<std::string, std::vector<std::pair<std::unique_ptr<TGraph>,std::unique_ptr<TGraph>>>> spline_graphs = getSplineGraphs(variable_systs[config.i_prime], config);
            c.Clear();
            c.Divide(4,4);
            int chan=0, snum = 0;
            for(const auto &[syst_name, syst_bins]: spline_graphs) {
                int bin = 0;
                int binning = variable_systs[config.i_prime].spline_binnings[snum++];
                bool unprinted = true;
                chan++;
                int col = chan%2==0 ? kRed: kBlue;

                for(const auto &[fixed_pts, curve]: syst_bins) {

                    unprinted = true;
                    c.cd(bin%16+1);
                    size_t sbi = config.GetSubchannelIndexFromVariableGlobalBin(bin,binning);
                    std::string nsubchannel = config.GetSubchannelName(sbi);
                    size_t local_channel_index = config.GetLocalChannelIndexFromGlobalSubchannelIndex(sbi);
                    std::string chan_units = config.GetChannelXAxisTitle(local_channel_index, binning);
                    int edges_vec_sz = (int)config.m_variable_bin_to_edges[binning].size();
                    size_t safe_bin = (edges_vec_sz>0) ? std::min((size_t)bin, (size_t)edges_vec_sz-1) : 0;
                    std::pair<float,float> edg = config.m_variable_bin_to_edges[binning][safe_bin];

                    fixed_pts->SetMarkerColor(col);
                    fixed_pts->SetMarkerStyle(kFullCircle);
                    fixed_pts->GetXaxis()->SetTitle("#sigma");
                    fixed_pts->GetYaxis()->SetTitle("Weight");
                    fixed_pts->SetTitle(("#splitline{"+syst_name+"}{#splitline{"+nsubchannel+" "+std::to_string(bin)+"}{"+chan_units+" ["+to_string_prec(edg.first,2)+"->"+to_string_prec(edg.second,2)+ "]}}").c_str());
                    fixed_pts->Draw("PA");
                    double max_y = TMath::MaxElement(fixed_pts->GetN(), fixed_pts->GetY());
                    double min_y = TMath::MinElement(fixed_pts->GetN(), fixed_pts->GetY());
                    double range = max_y - min_y;
                    fixed_pts->SetMaximum(max_y + 0.2 * range);
                    fixed_pts->SetMinimum(min_y);

                    curve->Draw("C same");
                    ++bin;
                    if(bin % 16 == 0) {
                        c.Print((final_output_tag+"_PROplot_Spline.pdf").c_str(), "pdf");
                        c.Clear();
                        c.Divide(4,4);
                        unprinted = false;
                    }
                }
                if(unprinted)
                    c.Print((final_output_tag+"_PROplot_Spline.pdf").c_str(), "pdf");
            }

            c.Print((final_output_tag+"_PROplot_Spline.pdf" + "]").c_str(), "pdf");
            c.Clear();

        }

        //now onto root files
        TFile fout((final_output_tag+"_PROplot.root").c_str(), "RECREATE");

        int io = 0;
        for(const auto &other: other_hists) {
            for(const auto &[name, hist]: other) {
                hist->Write(("Variable_"+std::to_string(io)+name).c_str());
            }
            io++;
        }

        if((fake_data_osc_params.size())) {
            PROspec osc_spec = FillSpectra(config, prop, variable_systs[config.i_prime], *model, fakeDataParams, !eventbyevent,config.i_prime);
            std::map<std::string, std::unique_ptr<TH1D>> osc_hists = getCV1DHists(osc_spec, config, binwidth_scale);
            fout.mkdir("Osc_hists");
            fout.cd("Osc_hists");
            for(const auto &[name, hist]: osc_hists) {
                hist->Write(name.c_str());
            }
        }

        fout.mkdir("Covariance");
        fout.cd("Covariance");
        for(const auto &[name, mat]: matrices)
            mat->Write(name.c_str());

        fout.mkdir("ErrorBand");
        fout.cd("ErrorBand");
        size_t eind = 0;
        for(size_t io = 0; io < config.m_num_variables; ++io) {
            if(!config.m_channel_variable_plot_bool.at(io))continue;// For now skip the L/E 250 bin. 
            size_t tot_offset = 0;
            // We need to use a different index here since we don't add error bars to the vector if
            // plot="false" in the xml
            const PROerrorbar &err = other_err_bands.at(eind++);
            for(size_t mode = 0; mode < config.m_num_modes; ++mode) {
                for(size_t det = 0; det < config.m_num_detectors; ++det) {
                    for(size_t channel = 0; channel < config.m_num_channels; ++channel) {
                        size_t channel_nbins_x = config.m_channel_variable_bins[channel][io].NBinsAlong(0);
                        // default is 1d, but catch 2d case for ybins
                        size_t channel_nbins_y = 1;
                        if(config.m_channel_variable_dims[channel][io] == 2)  channel_nbins_y = config.m_channel_variable_bins[channel][io].NBinsAlong(1);
                        size_t nbins_p_2dchan = channel_nbins_y*channel_nbins_x;
                        TGraphAsymmErrors eband(nbins_p_2dchan);
                        for(size_t i = 0; i < nbins_p_2dchan; ++i) {
                            float x = channel_nbins_y == 1 ? 
                                (config.m_channel_variable_bins[channel][io].Edges(0)[i+1] + config.m_channel_variable_bins[channel][io].Edges(0)[i])/2 :
                                i;
                            float xerr = channel_nbins_y == 1 ?
                                (config.m_channel_variable_bins[channel][io].Edges(0)[i+1] - config.m_channel_variable_bins[channel][io].Edges(0)[i])/2 :
                                0.5;
                            eband.SetPoint(i, x, err.error_point(tot_offset + i));
                            eband.SetPointEYhigh(i, err.error_up(tot_offset + i));
                            eband.SetPointEYlow(i, err.error_down(tot_offset + i));
                            eband.SetPointEXhigh(i, xerr);
                            eband.SetPointEXlow(i, xerr);
                        }
                        std::string name = config.m_mode_names[mode]+"_"+config.m_detector_names[det]+"_"+config.m_channel_names[channel]+"_var"+std::to_string(io);
                        eband.Write(name.c_str());
                        tot_offset += nbins_p_2dchan;
                    }
                }
            }
        }

        if((with_splines)) {
            std::map<std::string, std::vector<std::pair<std::unique_ptr<TGraph>,std::unique_ptr<TGraph>>>> spline_graphs = getSplineGraphs(variable_systs[config.i_prime], config);
            fout.mkdir("Splines");
            fout.cd("Splines");
            for(const auto &[name, syst_splines]: spline_graphs) {
                size_t bin = 0;
                for(const auto &[fixed_pts, curve]: syst_splines) {
                    fixed_pts->Write((name+"_fixedpts_"+std::to_string(bin)).c_str());
                    curve->Write((name+"_curve_"+std::to_string(bin)).c_str());
                    bin++;
                }
            }
        }

        fout.Close();
    }

    //***********************************************************************
    //***********************************************************************
    //********************     Feldman-Cousins    ***************************
    //***********************************************************************
    //***********************************************************************

    if(*profc_command) {
        float global_chi2 = 0, null_chi2 = 0;
        if(gof_pvalue || pvalue) {
            // Nominal Fit with all parameters
            GlobalFitOptions opt = GlobalFitOptions::Default;
            if(progress_bar) opt |= GlobalFitOptions::Progress;
            if(!global_fixed[0] || !systs_only) opt |= GlobalFitOptions::FreqSeedPts;
            PROspec cv = FillSpectra(config, prop, metric->GetSysts(), metric->GetModel(), CVParams , true ,config.i_prime);
            GlobalFitResult fitres = do_a_fit(config, prop, data, metric, global_ub, global_lb, fitConfig, CVParams, cv, global_fixed, opt); 
            global_chi2 = fitres.chi2;
        }
        if(pvalue) {
            // Fit with fixed osc parameters
            size_t nphys = metric->GetModel().nparams;
            Eigen::VectorXf lb = global_lb;
            Eigen::VectorXf ub = global_ub;
            std::vector<int> fixed = global_fixed;
            for(size_t i = 0; i < nphys; ++i) {
                lb(i) = metric->GetModel().default_val(i);
                ub(i) = metric->GetModel().default_val(i);
                fixed[i] = 1;
            }
            
            GlobalFitOptions opt = GlobalFitOptions::Default;
            if(progress_bar) opt |= GlobalFitOptions::Progress;
            PROspec cv = FillSpectra(config, prop, metric->GetSysts(), metric->GetModel(), CVParams , true ,config.i_prime);
            GlobalFitResult fitres = do_a_fit(config, prop, data, metric, ub, lb, fitConfig, CVParams, cv, fixed, opt); 
            null_chi2 = fitres.chi2;
        }

        size_t FCthreads = nthread > nuniv ? nuniv : nthread;
        Eigen::MatrixXf cv_vec = FillSpectra(config, prop, metric->GetSysts(), metric->GetModel(), CVParams , true,config.i_prime).Spec();
        Eigen::MatrixXf L = variable_systs[config.i_prime].DecomposeFractionalCovariance(config, cv_vec);

        std::vector<std::vector<float>> dchi2s;
        dchi2s.reserve(FCthreads);
        std::vector<std::vector<fc_out>> outs;
        outs.reserve(FCthreads);
        std::vector<std::thread> threads;
        size_t todo = nuniv/FCthreads;
        size_t addone = FCthreads - nuniv%FCthreads;
        bool gof_mode = gof_pvalue;

        std::vector<std::pair<int, std::string>> fc_PB_configs;
        for (size_t i = 0; i < FCthreads; ++i) {
                fc_PB_configs.push_back({int(nuniv/FCthreads), "Thread " + std::to_string(i)});
        }
        MultiPROgressBar fc_progress(fc_PB_configs);
        fc_progress.initialize_display();
        fc_progress.start_display_thread(); 


        for(size_t i = 0; i < FCthreads; i++) {
            dchi2s.emplace_back();
            outs.emplace_back();
            fc_args args{todo + (i >= addone), &dchi2s.back(), &outs.back(), config, prop, variable_systs[config.i_prime], chi2, fakeDataParams, L, scanFitConfig,(*myseed.getThreadSeeds())[i], (int)i, !eventbyevent, gof_mode};


            threads.emplace_back([args, &fc_progress]() {
                        PROfit::fc_worker(args, std::ref(fc_progress));
                        });
        }
        for(auto&& t: threads) {
            t.join();
        }
        fc_progress.finish_all();

        std::vector<float> flattened_dchi2s;
        for(const auto& v: dchi2s) for(const auto& dchi2: v) flattened_dchi2s.push_back(dchi2);
        std::sort(flattened_dchi2s.begin(), flattened_dchi2s.end());
        log<LOG_INFO>(L"%1% || 90%% Feldman-Cousins delta chi2 after throwing %2% universes is %3%") 
            % __func__ % nuniv % flattened_dchi2s[0.9*flattened_dchi2s.size()];
        if(gof_pvalue) {
            std::vector<float> flattened_syst_chi2;
            for(const auto &out : outs) for(const auto &fco : out) flattened_syst_chi2.push_back(fco.chi2_syst);
            std::sort(flattened_syst_chi2.begin(), flattened_syst_chi2.end());
            log<LOG_ERROR>(L"%1% || All: %2% ") % __func__ % flattened_syst_chi2;
            log<LOG_ERROR>(L"%1% || chi: %2% ") % __func__ % global_chi2;
            auto it = std::lower_bound(flattened_syst_chi2.begin(), flattened_syst_chi2.end(), global_chi2);
            size_t index =  std::distance(flattened_syst_chi2.begin(),it);
            size_t count_above = flattened_syst_chi2.size()-index;
            float pval = (float)count_above/(float)nuniv;
            log<LOG_ERROR>(L"%1% || Finished throws. %2% %3%") % __func__ % index % count_above;
            log<LOG_ERROR>(L"%1% || GOF pval after throwing %2% universes is %3%") % __func__ % nuniv % pval ;
        }
        if(pvalue) {
            log<LOG_ERROR>(L"%1% || All Delta Chis: %2% ") % __func__ % flattened_dchi2s;
            log<LOG_ERROR>(L"%1% || Delta chi bkg-min: %2% ") % __func__ % float(null_chi2 - global_chi2);
            auto itFC = std::lower_bound(flattened_dchi2s.begin(), flattened_dchi2s.end(), float(null_chi2-global_chi2));
            size_t indexFC =  std::distance(flattened_dchi2s.begin(),itFC);
            size_t count_aboveFC = flattened_dchi2s.size()-indexFC;
            float pvalFC = (float)count_aboveFC/(float)nuniv;

            log<LOG_ERROR>(L"%1% || Finished throws. %2% %3%") % __func__ % indexFC % count_aboveFC;
            log<LOG_ERROR>(L"%1% || FC Corrected pval after throwing %2% universes is %3%") % __func__ % nuniv % pvalFC ;
        }

        {
            TFile fout((final_output_tag+"_FC.root").c_str(), "RECREATE");
            fout.cd();
            float chi2_osc, chi2_syst;
            // One float per physics parameter — plain branches named "best_<param_name>".
            // Vector kept alive for the full lifetime of the TTree.
            std::vector<float> best_phys_vals(model->nparams, 0.0f);
            std::map<std::string, float> best_systs_osc, best_systs, syst_throw;
            TTree tree("tree", "tree");
            tree.Branch("chi2_osc",  &chi2_osc);
            tree.Branch("chi2_syst", &chi2_syst);
            for(size_t i = 0; i < model->nparams; ++i)
                tree.Branch(("best_" + model->param_names[i]).c_str(), &best_phys_vals[i]);
            tree.Branch("best_systs_osc", &best_systs_osc);
            tree.Branch("best_systs",     &best_systs);
            tree.Branch("syst_throw",     &syst_throw);

            for(const auto &out: outs) {
                for(const auto &fco: out) {
                    chi2_osc  = fco.chi2_osc;
                    chi2_syst = fco.chi2_syst;
                    for(size_t i = 0; i < model->nparams; ++i) {
                        float raw = fco.best_phys_osc.size() > (Eigen::Index)i ? fco.best_phys_osc(i) : 0.0f;
                        best_phys_vals[i] = model->is_log10[i] ? std::pow(10.0f, raw) : raw;
                    }
                    for(size_t i = 0; i < variable_systs[config.i_prime].GetNSplines(); ++i) {
                        if(!gof_pvalue) best_systs_osc[variable_systs[config.i_prime].spline_names[i]] = fco.best_fit_osc(i);
                        best_systs[variable_systs[config.i_prime].spline_names[i]] = fco.best_fit_syst(i);
                        syst_throw[variable_systs[config.i_prime].spline_names[i]] = fco.syst_throw(i);
                    }
                    tree.Fill();
                }
            }

            tree.Write();
        }
        {
            ofstream fcout(final_output_tag+"_FC.csv");
            fcout << "chi2_osc,chi2_syst";
            for(const std::string &name: model->param_names)
                fcout << ",best_" << name;
            for(const std::string &name: variable_systs[config.i_prime].spline_names)
                fcout << ",best_" << name << "_osc,best_" << name << "," << name << "_throw";
            fcout << "\r\n";

            for(const auto &out: outs) {
                for(const auto &fco: out) {
                    fcout << fco.chi2_osc << "," << fco.chi2_syst;
                    for(size_t i = 0; i < model->nparams; ++i) {
                        float raw = fco.best_phys_osc.size() > (Eigen::Index)i ? fco.best_phys_osc(i) : 0.0f;
                        float val = model->is_log10[i] ? std::pow(10.0f, raw) : raw;
                        fcout << "," << val;
                    }
                    for(size_t i = 0; i < variable_systs[config.i_prime].GetNSplines(); ++i)
                        fcout << "," << (gof_pvalue ? 0 : fco.best_fit_osc(i)) << "," << fco.best_fit_syst(i) << "," << fco.syst_throw(i);
                    fcout << "\r\n";
                }
            }
        }
    }


    //***********************************************************************
    //***********************************************************************
    //******************** global       **************************
    //***********************************************************************
    //***********************************************************************
    if(*proglobal_command){

        GlobalFitOptions opt = GlobalFitOptions::Default;
        if(progress_bar) opt |= GlobalFitOptions::Progress;
        if(binwidth_scale) opt |= GlobalFitOptions::BinWidthScaled;
        if(!global_fixed[0] || !systs_only) opt |= GlobalFitOptions::FreqSeedPts;
        opt |= MCMC_prefit_errors ? GlobalFitOptions::MCMCPrefitErrorBand : GlobalFitOptions::PrefitErrorBand;
        opt |= GlobalFitOptions::PostFitErrorBand;
        opt |= GlobalFitOptions::Correlations;
        PROspec cv = FillSpectra(config, prop, metric->GetSysts(), metric->GetModel(), CVParams , true ,config.i_prime);
        GlobalFitResult fitres = do_a_fit(config, prop, data, metric, global_ub, global_lb, fitConfig, CVParams, cv, global_fixed, opt); 
        global_fit_chi2 = fitres.chi2;
        global_fit_result = fitres.fitter.best_fit;

        TH2D corrhist("crh", "", N_params, 0, N_params, N_params, 0, N_params);
        TH2D fraccovhist("fch", "", N_params, 0, N_params, N_params, 0, N_params);
        TH2D covhist("ch", "", N_params, 0, N_params, N_params, 0, N_params);
        std::vector<std::string> param_names;
        for(size_t i = 0; i < N_params; ++i) {
            std::string label = i < N_phys_params 
                ? metric->GetModel().pretty_param_names[i]
                : config.m_mcgen_variation_plotname_map[metric->GetSysts().spline_names[i-N_phys_params]].c_str();
            param_names.push_back(label);
            covhist.GetXaxis()->SetBinLabel(i+1, label.c_str());
            covhist.GetYaxis()->SetBinLabel(i+1, label.c_str());
            fraccovhist.GetXaxis()->SetBinLabel(i+1, label.c_str());
            fraccovhist.GetYaxis()->SetBinLabel(i+1, label.c_str());
            corrhist.GetXaxis()->SetBinLabel(i+1, label.c_str());
            corrhist.GetYaxis()->SetBinLabel(i+1, label.c_str());
            for(size_t j = 0; j < N_params; ++j) {
                covhist.SetBinContent(i+1, j+1, fitres.covmat(i,j));
                fraccovhist.SetBinContent(i+1, j+1, fitres.fraccovmat(i,j));
                corrhist.SetBinContent(i+1, j+1, fitres.corrmat(i,j));
            }
        }
        TCanvas c1;
        corrhist.SetMaximum(1);
        corrhist.SetMinimum(-1);
        covhist.SetMaximum(1);
        covhist.SetMinimum(-1);
        fraccovhist.SetMaximum(100);
        fraccovhist.SetMinimum(-100);
        c1.SetLeftMargin(0.18);   
        corrhist.SetTitle("Post-Fit Correlation Matrix");
        corrhist.Draw("colz");
        gPad->Update();

        TLine line;
        line.SetLineColor(kBlack);
        line.SetLineWidth(2);
        line.DrawLine(N_phys_params, 0, N_phys_params, N_params);
        line.DrawLine(0, N_phys_params, N_params, N_phys_params);
        c1.Print((final_output_tag+"_postfit_correlation_matrix.pdf").c_str());

        log<LOG_INFO>(L"%1% || MCMC acceptance is  %2%. ") % __func__% ((double)fitres.mh->naccept /fitConfig.MCMCiter);

        fitres.mh->plot_autocorrelation(final_output_tag+"_PROglobal_corrmat_mcmc_autocorrelation.pdf", param_names);

        std::string hname = "#chi^{2}/nbins = " + to_string(fitres.chi2) + "/" + to_string(config.m_num_variable_bins_total_collapsed[config.i_prime]);
        PROspec bf = FillSpectra(config, prop, metric->GetSysts(), metric->GetModel(), fitres.fitter.best_fit, true ,config.i_prime);

        // Concatenated bins across all channels share no common x-axis, so use bin-index axis.
        TH1D post_hist("ph", hname.c_str(), config.m_num_variable_bins_total_collapsed[config.i_prime], 0, config.m_num_variable_bins_total_collapsed[config.i_prime]);
        TH1D pre_hist("prh", hname.c_str(), config.m_num_variable_bins_total_collapsed[config.i_prime], 0, config.m_num_variable_bins_total_collapsed[config.i_prime]);
        for(size_t i = 0; i < config.m_num_variable_bins_total_collapsed[config.i_prime]; ++i) {
            post_hist.SetBinContent(i+1, bf.Spec()(i));
            pre_hist.SetBinContent(i+1, cv.Spec()(i));
        }

        std::vector<TPaveText> texts;
        TPaveText chi2text(0.55, 0.50, 0.85, 0.58, "NDC");
        chi2text.AddText(hname.c_str());
        chi2text.SetFillColor(0);
        chi2text.SetBorderSize(0);
        chi2text.SetTextAlign(12);
        //chi2text.SetTextSize(0.035); 
        texts.push_back(chi2text);

        PlotOptions popt; 
        if(data_mc_ratio){
            popt = PlotOptions::DataMCRatio;
        } 
        else{
            popt = PlotOptions::DataPostfitRatio;
        }
        if(binwidth_scale) popt |= PlotOptions::BinWidthScaled;
        if(area_normalized) popt |= PlotOptions::AreaNormalized;
        plot_channels((final_output_tag+"_PROglobal_hists.pdf"), config, cv, bf, data, fitres.err_band, fitres.post_err_band, texts, pbounds, popt, 0, true);
    }

    if(*promcmc_command) {
        metric->setBounds(global_ub, global_lb);
        size_t nparams = metric->GetModel().nparams + metric->GetSysts().GetNSplines();
        std::uniform_real_distribution<float> latin_distribution(-2, 2);
        std::vector<std::vector<float>> samples = latin_hypercube_sampling(mcmc_chains, nparams, latin_distribution, myseed.global_rng);
        recenter_latin_samples(samples, global_ub, global_lb);
        std::vector<Eigen::VectorXf> samples_eigen; 
        for(size_t i = 0; i < samples.size(); ++i)
            samples_eigen.push_back(Eigen::VectorXf::Map(samples[i].data(), samples[i].size()));
        size_t mcmc_threads = mcmc_chains >= nthread ? nthread : mcmc_chains;
        std::vector<std::vector<Metropolis<simple_target, adaptive_proposal>>> mets;
        mets.reserve(mcmc_threads);
        std::vector<std::thread> threads;
        size_t chains_per_thread = mcmc_chains / mcmc_threads;
        size_t addone = mcmc_threads - mcmc_chains%mcmc_threads;
        for(size_t i = 0; i < mcmc_threads; ++i) {
            mets.emplace_back();
            threads.emplace_back(
                    [&, i](){
                        mcmc_worker(mets[i], samples_eigen[i], metric->Clone(), myseed.getThreadSeeds()->at(i), chains_per_thread + (i >= addone), fitConfig.MCMCburn, fitConfig.MCMCiter);
                    });
        }
        for(auto&& t : threads) {
            t.join();
        }

        //std::vector<TH2D> twod;
        //std::vector<TH1D> oned;
        // TODO: This is hardcoded for numu disappearance right now
        // TODO: How do we input binnings for these plots in a nice way?
        // TODO: Is it better to write out the chain to a cvs/root file and plot externally?
        //twod.push_back(TH2D("two", ";sin^{2}2#theta_{#mu#mu};#Deltam^{2}_{41} [eV^{2}];MCMC Points",
        //                     200, -3, 0, 200, -2, 2));
        //oned.push_back(TH1D("one1", ";sin^{2}2#theta_{#mu#mu};Posterior PDF", 200, -3, 0));
        //oned.push_back(TH1D("one2", ";#Deltam^{2}_{41} [eV^{2}];Posterior PDF", 200, -2, 2));
        TFile fout((final_output_tag+"_PROMCMC_chains.root").c_str(), "RECREATE");
        size_t chain_counter = 0;
        for(const auto &tmets : mets) {
            for(const auto &met : tmets) {
                chain_counter++;
                std::string name = "chain"+std::to_string(chain_counter);
                TTree tree(name.c_str(), name.c_str());
                Eigen::VectorXf v = Eigen::VectorXf::Zero(nparams);
                std::vector<std::string> param_names;
                for(size_t i = 0; i < metric->GetModel().nparams; ++i) {
                    tree.Branch(metric->GetModel().param_names[i].c_str(), &v(i));
                    param_names.push_back(metric->GetModel().pretty_param_names[i]);
                }
                for(size_t i = metric->GetModel().nparams; i < nparams; ++i) {
                    const std::string &sname = metric->GetSysts().spline_names[i-metric->GetModel().nparams];
                    std::string::size_type l = sname.find(':');
                    // TODO: This only handles names with a single colon in them. I don't think we ever have more than that, it's really just meant for the 'flat' and 'norm' systs.
                    if(l != std::string::npos) {
                        std::string bname = sname;
                        bname[l] = '_';
                        tree.Branch(bname.c_str(), &v(i));
                    } else {
                        tree.Branch(sname.c_str(), &v(i));
                    }
                    param_names.push_back(config.m_mcgen_variation_plotname_map.at(metric->GetSysts().spline_names[i-N_phys_params]));
                }
                for(const auto &p : met.chain) {
                //    twod[0].Fill(p(1), p(0));
                //    oned[0].Fill(p(1));
                //    oned[1].Fill(p(0));
                    v = p;
                    tree.Fill();
                }
                tree.Write();
                met.plot_autocorrelation((final_output_tag+"_PROMCMC_autocorrelation_chain"+std::to_string(chain_counter)+".pdf").c_str(), param_names);
            }
        }
        //TCanvas c;
        //c.Divide(2,2);
        //c.cd(1);
        //oned[0].Scale(1.0/oned[0].Integral());
        //oned[0].Draw("hist");
        //c.cd(3);
        //twod[0].Draw("colz");
        //c.cd(4);
        //oned[1].Scale(1.0/oned[1].Integral());
        //oned[1].Draw("hist");
        //c.Print("mcmc_corner_plot.pdf");
    }

    //***********************************************************************
    //***********************************************************************
    //******************** TEST AREA TEST AREA     **************************
    //***********************************************************************
    //***********************************************************************

    if(*bench_command) {
        // Parse the comma-separated --tests selector. "all"/"fillspectra"/
        // "metric"/"fit" expand to the corresponding group; individual letters
        // a–i map to single tests.
        unsigned bench_mask = 0u;
        auto match_token = [&bench_mask](const std::string &tok) {
            using namespace PROfit::PRObench;
            if      (tok == "all")         bench_mask |= Bench_All;
            else if (tok == "fillspectra") bench_mask |= Bench_FillSpectra_Group;
            else if (tok == "metric")      bench_mask |= Bench_Metric_Group;
            else if (tok == "metricgrad")  bench_mask |= Bench_MetricGrad_Group;
            else if (tok == "fit")         bench_mask |= Bench_Fit_Group;
            else if (tok == "pseudo")      bench_mask |= Bench_PseudoUniverse;
            else if (tok == "collapse")    bench_mask |= Bench_Collapse;
            else if (tok == "mcmc")        bench_mask |= Bench_MCMC_Group;
            else if (tok == "a")           bench_mask |= Bench_FillSpectra_All;
            else if (tok == "b")           bench_mask |= Bench_FillSpectra_Phys;
            else if (tok == "c")           bench_mask |= Bench_FillSpectra_Nuis;
            else if (tok == "d")           bench_mask |= Bench_Metric_All;
            else if (tok == "e")           bench_mask |= Bench_Metric_Phys;
            else if (tok == "f")           bench_mask |= Bench_Metric_Nuis;
            else if (tok == "g")           bench_mask |= Bench_Fit;
            else if (tok == "h")           bench_mask |= Bench_PseudoUniverse;
            else if (tok == "i")           bench_mask |= Bench_Collapse;
            else if (tok == "j")           bench_mask |= Bench_MetricGrad_All;
            else if (tok == "k")           bench_mask |= Bench_MetricGrad_Phys;
            else if (tok == "l")           bench_mask |= Bench_MetricGrad_Nuis;
            else if (tok == "m")           bench_mask |= Bench_MCMC_Burnin;
            else if (tok == "n")           bench_mask |= Bench_MCMC_Post;
            else log<LOG_WARNING>(L"%1% || scale-test: unknown --tests token '%2%' (ignored)") % __func__ % tok.c_str();
        };
        const std::string &s = bench_tests_str;
        size_t pos = 0;
        while (pos < s.size()) {
            size_t comma = s.find(',', pos);
            std::string tok = s.substr(pos, (comma == std::string::npos ? s.size() : comma) - pos);
            while (!tok.empty() && std::isspace(static_cast<unsigned char>(tok.front()))) tok.erase(tok.begin());
            while (!tok.empty() && std::isspace(static_cast<unsigned char>(tok.back())))  tok.pop_back();
            if (!tok.empty()) match_token(tok);
            if (comma == std::string::npos) break;
            pos = comma + 1;
        }
        if (bench_mask == 0u) bench_mask = PROfit::PRObench::Bench_All;

        PROfit::PRObench::BenchOptions bopts;
        bopts.N      = bench_N;
        bopts.tests  = bench_mask;
        bopts.binned = !eventbyevent;

        PROfit::PRObench::run_scale_test(config, prop, *metric, fitConfig, bopts);
    }

    if(*protest_command){
        log<LOG_INFO>(L"%1% || PROtest: Testing FillSpectra with fixed seed random spline throws") % __func__;
        
        size_t n_splines = metric->GetSysts().GetNSplines();
        size_t n_tests = 3;
        
        // Fixed seed for reproducibility across code versions
        std::mt19937 test_rng(12345);
        std::normal_distribution<float> d(0.0f, 1.0f);
        
        for(size_t test = 0; test < n_tests; ++test) {
            log<LOG_INFO>(L"%1% || ===== TEST %2% =====") % __func__ % (test + 1);
            
            // Create params: physics from CVParams, random splines
            Eigen::VectorXf testParams = Eigen::VectorXf::Zero(N_phys_params + n_splines);
            for(size_t i = 0; i < N_phys_params; ++i) {
                testParams(i) = CVParams(i);
            }
            for(size_t i = 0; i < n_splines; ++i) {
                testParams(N_phys_params + i) = d(test_rng);
            }

            log<LOG_INFO>(L"%1% || Test %2% Parameters:") % __func__ % (test + 1);
            for(Eigen::Index i = 0; i < testParams.size(); ++i) {
                log<LOG_INFO>(L"%1% || Test %2% Parameter %3%: %4%") % __func__ % (test + 1) % i % testParams(i);
            }
            
            // Fill spectrum
            PROspec spec = FillSpectra(config, prop, metric->GetSysts(), metric->GetModel(), testParams, true, config.i_prime);
            
            // Print all bin values
            log<LOG_INFO>(L"%1% || Test %2% Spectrum (%3% bins):") % __func__ % (test + 1) % spec.Spec().size();
            for(long b = 0; b < spec.Spec().size(); ++b) {
                log<LOG_INFO>(L"%1% || Test %2% bin %3%: %4%") % __func__ % (test + 1) % b % spec.Spec()(b);
            }
            log<LOG_INFO>(L"%1% || Test %2% Total: %3%") % __func__ % (test + 1) % spec.Spec().sum();
        }

        // ---- chi2 algorithm comparison: 4 paths exercising covariance-build and inverse-vs-solve ----
        log<LOG_INFO>(L"%1% || ===== chi2 algorithm comparison =====") % __func__;
        const PROsyst &cmp_syst = metric->GetSysts();
        const Eigen::MatrixXf &cmp_fcov = cmp_syst.fractional_covariance;
        const Eigen::VectorXf cmp_data = data.Spec();

        std::vector<Eigen::Index> cmp_nonempty;
        for(Eigen::Index i = 0; i < cmp_data.size(); ++i)
            if(cmp_data(i) > 0) cmp_nonempty.push_back(i);
        const size_t cmp_red = cmp_nonempty.size();
        if(cmp_red == 0) {
            log<LOG_ERROR>(L"%1% || All data bins empty, skipping chi2 comparison.") % __func__;
        } else {
            const size_t cmp_N = 200;
            std::vector<Eigen::VectorXf> cmp_params;
            cmp_params.reserve(cmp_N);
            std::mt19937 cmp_rng(67890);
            std::normal_distribution<float> cmp_d(0.0f, 1.0f);
            for(size_t k = 0; k < cmp_N; ++k) {
                Eigen::VectorXf p = Eigen::VectorXf::Zero(N_phys_params + n_splines);
                for(size_t i = 0; i < N_phys_params; ++i) p(i) = CVParams(i);
                for(size_t i = 0; i < n_splines; ++i) p(N_phys_params + i) = cmp_d(cmp_rng);
                cmp_params.push_back(p);
            }

            std::vector<float> chi2_old(cmp_N), chi2_mid(cmp_N), chi2_new(cmp_N), chi2_alt(cmp_N);
            Eigen::MatrixXf cmp_red_stat(cmp_red, cmp_red);

            // OLD: dense diag*F*diag + .inverse()
            auto cmp_t0 = std::chrono::high_resolution_clock::now();
            for(size_t k = 0; k < cmp_N; ++k) {
                PROspec r = FillSpectra(config, prop, cmp_syst, metric->GetModel(), cmp_params[k], true, config.i_prime);
                Eigen::MatrixXf diag = r.Spec().array().matrix().asDiagonal();
                Eigen::MatrixXf full_cov = diag * cmp_fcov * diag;
                Eigen::MatrixXf coll = CollapseMatrix(config, full_cov);
                Eigen::MatrixXf stat = cmp_data.matrix().asDiagonal();
                Eigen::MatrixXf red_full(cmp_red, cmp_red);
                for(size_t i = 0; i < cmp_red; ++i)
                    for(size_t j = 0; j < cmp_red; ++j) {
                        red_full(i,j) = coll(cmp_nonempty[i], cmp_nonempty[j]);
                        cmp_red_stat(i,j) = stat(cmp_nonempty[i], cmp_nonempty[j]);
                    }
                Eigen::MatrixXf inv = (cmp_red_stat + red_full).inverse();
                Eigen::VectorXf coll_mc = CollapseMatrix(config, r.Spec());
                Eigen::VectorXf delta(cmp_red);
                for(size_t i = 0; i < cmp_red; ++i)
                    delta(i) = coll_mc(cmp_nonempty[i]) - cmp_data(cmp_nonempty[i]);
                chi2_old[k] = (delta.transpose() * inv * delta)(0,0);
            }
            auto cmp_t1 = std::chrono::high_resolution_clock::now();

            // INTERMEDIATE: cwiseProduct (fast covariance build) + .inverse() (slow inversion)
            for(size_t k = 0; k < cmp_N; ++k) {
                PROspec r = FillSpectra(config, prop, cmp_syst, metric->GetModel(), cmp_params[k], true, config.i_prime);
                const Eigen::VectorXf &s = r.Spec();
                Eigen::MatrixXf full_cov = (s * s.transpose()).cwiseProduct(cmp_fcov);
                Eigen::MatrixXf coll = CollapseMatrix(config, full_cov);
                Eigen::MatrixXf red_full(cmp_red, cmp_red);
                for(size_t i = 0; i < cmp_red; ++i)
                    for(size_t j = 0; j < cmp_red; ++j) {
                        red_full(i,j) = coll(cmp_nonempty[i], cmp_nonempty[j]);
                        cmp_red_stat(i,j) = (i == j) ? cmp_data(cmp_nonempty[i]) : 0.0f;
                    }
                Eigen::MatrixXf inv = (cmp_red_stat + red_full).inverse();
                Eigen::VectorXf coll_mc = CollapseMatrix(config, r.Spec());
                Eigen::VectorXf delta(cmp_red);
                for(size_t i = 0; i < cmp_red; ++i)
                    delta(i) = coll_mc(cmp_nonempty[i]) - cmp_data(cmp_nonempty[i]);
                chi2_mid[k] = (delta.transpose() * inv * delta)(0,0);
            }
            auto cmp_t2 = std::chrono::high_resolution_clock::now();

            // NEW: cwiseProduct + .llt().solve()
            for(size_t k = 0; k < cmp_N; ++k) {
                PROspec r = FillSpectra(config, prop, cmp_syst, metric->GetModel(), cmp_params[k], true, config.i_prime);
                const Eigen::VectorXf &s = r.Spec();
                Eigen::MatrixXf full_cov = (s * s.transpose()).cwiseProduct(cmp_fcov);
                Eigen::MatrixXf coll = CollapseMatrix(config, full_cov);
                Eigen::MatrixXf red_full(cmp_red, cmp_red);
                for(size_t i = 0; i < cmp_red; ++i)
                    for(size_t j = 0; j < cmp_red; ++j) {
                        red_full(i,j) = coll(cmp_nonempty[i], cmp_nonempty[j]);
                        cmp_red_stat(i,j) = (i == j) ? cmp_data(cmp_nonempty[i]) : 0.0f;
                    }
                Eigen::MatrixXf M = cmp_red_stat + red_full;
                Eigen::VectorXf coll_mc = CollapseMatrix(config, r.Spec());
                Eigen::VectorXf delta(cmp_red);
                for(size_t i = 0; i < cmp_red; ++i)
                    delta(i) = coll_mc(cmp_nonempty[i]) - cmp_data(cmp_nonempty[i]);
                chi2_new[k] = delta.dot(M.llt().solve(delta));
            }
            auto cmp_t3 = std::chrono::high_resolution_clock::now();

            // ALT: inline asDiagonal()*F*asDiagonal() (no materialization) + .llt().solve() (matches upstream's covariance build)
            for(size_t k = 0; k < cmp_N; ++k) {
                PROspec r = FillSpectra(config, prop, cmp_syst, metric->GetModel(), cmp_params[k], true, config.i_prime);
                Eigen::MatrixXf full_cov = r.Spec().asDiagonal() * cmp_fcov * r.Spec().asDiagonal();
                Eigen::MatrixXf coll = CollapseMatrix(config, full_cov);
                Eigen::MatrixXf red_full(cmp_red, cmp_red);
                for(size_t i = 0; i < cmp_red; ++i)
                    for(size_t j = 0; j < cmp_red; ++j) {
                        red_full(i,j) = coll(cmp_nonempty[i], cmp_nonempty[j]);
                        cmp_red_stat(i,j) = (i == j) ? cmp_data(cmp_nonempty[i]) : 0.0f;
                    }
                Eigen::MatrixXf M = cmp_red_stat + red_full;
                Eigen::VectorXf coll_mc = CollapseMatrix(config, r.Spec());
                Eigen::VectorXf delta(cmp_red);
                for(size_t i = 0; i < cmp_red; ++i)
                    delta(i) = coll_mc(cmp_nonempty[i]) - cmp_data(cmp_nonempty[i]);
                chi2_alt[k] = delta.dot(M.llt().solve(delta));
            }
            auto cmp_t4 = std::chrono::high_resolution_clock::now();

            double cmp_old_s = std::chrono::duration<double>(cmp_t1 - cmp_t0).count();
            double cmp_mid_s = std::chrono::duration<double>(cmp_t2 - cmp_t1).count();
            double cmp_new_s = std::chrono::duration<double>(cmp_t3 - cmp_t2).count();
            double cmp_alt_s = std::chrono::duration<double>(cmp_t4 - cmp_t3).count();

            auto worst_diff = [&](const std::vector<float> &ref, const std::vector<float> &test) {
                double max_abs = 0, max_rel = 0;
                size_t max_rel_k = 0;
                for(size_t k = 0; k < cmp_N; ++k) {
                    double a = std::abs(double(test[k]) - double(ref[k]));
                    double denom = std::max(std::abs(double(ref[k])), 1e-30);
                    double r = a / denom;
                    if(a > max_abs) max_abs = a;
                    if(r > max_rel) { max_rel = r; max_rel_k = k; }
                }
                return std::make_tuple(max_abs, max_rel, max_rel_k);
            };
            auto [mid_abs, mid_rel, mid_k] = worst_diff(chi2_old, chi2_mid);
            auto [new_abs, new_rel, new_k] = worst_diff(chi2_old, chi2_new);
            auto [alt_abs, alt_rel, alt_k] = worst_diff(chi2_old, chi2_alt);

            log<LOG_INFO>(L"%1% || ----- chi2 algorithm comparison: 4 paths, identical parameter sets -----") % __func__;
            log<LOG_INFO>(L"%1% || N = %2% chi2 evaluations per path; reduced bin count = %3% (after dropping empty-data bins)") % __func__ % cmp_N % cmp_red;
            log<LOG_INFO>(L"%1% ||") % __func__;
            log<LOG_INFO>(L"%1% || Path 1 (OLD)         : full_cov = diag(spec) * F * diag(spec)        [dense diag materialized -> O(N^3) matmuls]") % __func__;
            log<LOG_INFO>(L"%1% ||                        chi2     = delta^T * (M).inverse() * delta    [O(N^3) inverse + 2 matvec]") % __func__;
            log<LOG_INFO>(L"%1% || Path 2 (INTERMEDIATE): full_cov = (spec * spec^T).cwiseProduct(F)    [O(N^2): outer product + Hadamard]") % __func__;
            log<LOG_INFO>(L"%1% ||                        chi2     = delta^T * (M).inverse() * delta    [same inverse step as OLD]") % __func__;
            log<LOG_INFO>(L"%1% || Path 3 (NEW)         : full_cov = (spec * spec^T).cwiseProduct(F)    [same as INTERMEDIATE]") % __func__;
            log<LOG_INFO>(L"%1% ||                        chi2     = delta . M.llt().solve(delta)       [Cholesky solve, no explicit inverse]") % __func__;
            log<LOG_INFO>(L"%1% || Path 4 (ALT)         : full_cov = spec.asDiagonal()*F*spec.asDiagonal() [Eigen DiagonalWrapper, matches upstream build]") % __func__;
            log<LOG_INFO>(L"%1% ||                        chi2     = delta . M.llt().solve(delta)       [same solve as NEW]") % __func__;
            log<LOG_INFO>(L"%1% ||") % __func__;
            log<LOG_INFO>(L"%1% || Path 1 (OLD)         : %2% s total, %3% ms/call (baseline)")
                % __func__ % cmp_old_s % (cmp_old_s / cmp_N * 1e3);
            log<LOG_INFO>(L"%1% || Path 2 (INTERMEDIATE): %2% s total, %3% ms/call (%4%x vs OLD)  -> isolates the covariance-build speedup")
                % __func__ % cmp_mid_s % (cmp_mid_s / cmp_N * 1e3) % (cmp_old_s / std::max(cmp_mid_s, 1e-12));
            log<LOG_INFO>(L"%1% || Path 3 (NEW)         : %2% s total, %3% ms/call (%4%x vs OLD)  -> cwiseProduct build + Cholesky solve")
                % __func__ % cmp_new_s % (cmp_new_s / cmp_N * 1e3) % (cmp_old_s / std::max(cmp_new_s, 1e-12));
            log<LOG_INFO>(L"%1% || Path 4 (ALT)         : %2% s total, %3% ms/call (%4%x vs OLD)  -> asDiagonal build + Cholesky solve (= current code)")
                % __func__ % cmp_alt_s % (cmp_alt_s / cmp_N * 1e3) % (cmp_old_s / std::max(cmp_alt_s, 1e-12));
            log<LOG_INFO>(L"%1% ||") % __func__;
            log<LOG_INFO>(L"%1% || NEW vs ALT covariance-build comparison: %2%x (>1 means ALT/asDiagonal is faster)")
                % __func__ % (cmp_new_s / std::max(cmp_alt_s, 1e-12));
            log<LOG_INFO>(L"%1% ||") % __func__;
            log<LOG_INFO>(L"%1% || Numerical agreement (chi2 differences from OLD path):") % __func__;
            log<LOG_INFO>(L"%1% || INTERMEDIATE vs OLD: worst abs diff %2%, worst rel diff %3% (at k=%4%, old=%5%, mid=%6%)")
                % __func__ % mid_abs % mid_rel % mid_k % chi2_old[mid_k] % chi2_mid[mid_k];
            log<LOG_INFO>(L"%1% || NEW          vs OLD: worst abs diff %2%, worst rel diff %3% (at k=%4%, old=%5%, new=%6%)")
                % __func__ % new_abs % new_rel % new_k % chi2_old[new_k] % chi2_new[new_k];
            log<LOG_INFO>(L"%1% || ALT          vs OLD: worst abs diff %2%, worst rel diff %3% (at k=%4%, old=%5%, alt=%6%)")
                % __func__ % alt_abs % alt_rel % alt_k % chi2_old[alt_k] % chi2_alt[alt_k];
        }

    }

    /*
    if(*protest_command){
        log<LOG_INFO>(L"%1% || PROtest. Place anything here, a playground for testing things.") % __func__;
        //PrintVariableInfo(config);
        auto start = std::chrono::high_resolution_clock::now();
        int N=1000;
        for(int i=0; i< N; i++){
            FillSpectra(config, prop, metric->GetSysts(), metric->GetModel(), fakeDataParams , true,config.i_prime);
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        log<LOG_INFO>(L"%1% || PROtest took %2% seconds total, or %3% per call of FillSpectra; ") % __func__ % duration.count() % float(duration.count()/(double(N)));
        // *************************** END *********************************
    }
    */

    /*
    if(*protest_command){
        log<LOG_INFO>(L"%1% || PROtest. Place anything here, a playground for testing things .") % __func__;
        //PrintVariableInfo(config);


        auto start = std::chrono::high_resolution_clock::now();
        int N=1000;
        for(int i=0; i< N; i++){
            FillSpectra(config, prop, metric->GetSysts(), metric->GetModel(), fakeDataParams , true,config.i_prime);
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        log<LOG_INFO>(L"%1% || PROtest took %2% sseconds total, or %3% per call of FillSPectra; ") % __func__ % duration.count() % float(duration.count()/(double(N)));

        // *************************** END *********************************
    }
    */

    std::ofstream global_fit_out;
    if(global_fit_result.size() > 0) {
        global_fit_out.open(final_output_tag+"_global_fit.txt");
        float chi2 = global_fit_chi2 >= 0 ? global_fit_chi2 : global_fit_chi2_surf;
        log<LOG_INFO>(L"%1% || ################################################") % __func__;
        log<LOG_INFO>(L"%1% || ########### Global Best Fit Results ############") % __func__;
        log<LOG_INFO>(L"%1% || ################################################") % __func__;
        log<LOG_INFO>(L"%1% || Global Best Fit chi^2: %2%") %__func__ % chi2;
        log<LOG_INFO>(L"%1% || at paramters: ") % __func__;

        global_fit_out << "Global best fit:\n";

        bool use_phys = (size_t)global_fit_result.size() == N_phys_params + metric->GetSysts().GetNSplines();
        for(long i = 0; i < global_fit_result.size(); i++){

            if(use_phys && i < (long)N_phys_params){
                log<LOG_INFO>(L"%1% || %2%  : %3% (log) %4% (nonlog) ") % __func__ % metric->GetModel().pretty_param_names[i].c_str() % global_fit_result(i) % pow(10,global_fit_result(i));
                global_fit_out << metric->GetModel().param_names[i] << " : " << global_fit_result(i) << "\n";
            }else{
                long idx = use_phys ? i - N_phys_params : i;
                log<LOG_INFO>(L"%1% || %2%  :  %3% ") % __func__ % config.m_mcgen_variation_plotname_map.at(metric->GetSysts().spline_names[idx]).c_str() % global_fit_result(i);

                global_fit_out <<  config.m_mcgen_variation_plotname_map.at(metric->GetSysts().spline_names[idx])
                    << " : " << global_fit_result(i) << "\n";
            }
        }
        log<LOG_INFO>(L"%1% || ################################################") % __func__;
    }
    if(global_fit_result_surf.size() > 0) {
        if(!global_fit_out.is_open()) {
            global_fit_out.open(final_output_tag+"_global_fit.txt");
            global_fit_out << "Global best fit:\n";
        } else {
            global_fit_out << "\nSurface global best fit:\n";
        }
        log<LOG_INFO>(L"%1% || ########################################################") % __func__;
        log<LOG_INFO>(L"%1% || ########### Surface Global Best Fit Results ############") % __func__;
        log<LOG_INFO>(L"%1% || ########################################################") % __func__;
        log<LOG_INFO>(L"%1% || Global Best Fit chi^2: %2%") %__func__ % global_fit_chi2_surf;
        log<LOG_INFO>(L"%1% || at paramters: ") % __func__;

        for(long i = 0; i < global_fit_result.size(); i++){
            if(i < (long)N_phys_params){
                log<LOG_INFO>(L"%1% || %2%  : %3% (log) %4% (nonlog) ") % __func__ % metric->GetModel().pretty_param_names[i].c_str() % global_fit_result(i) % pow(10,global_fit_result(i));
                global_fit_out << metric->GetModel().param_names[i] << " : " << global_fit_result(i) << "\n";
            }else{
                log<LOG_INFO>(L"%1% || %2%  :  %3% ") % __func__ % metric->GetSysts().spline_names[i - metric->GetModel().nparams].c_str() % global_fit_result(i);
                global_fit_out << metric->GetSysts().spline_names[i - metric->GetModel().nparams]
                    << " : " << global_fit_result(i) << "\n";
            }
        }
        log<LOG_INFO>(L"%1% || ########################################################") % __func__;
    }
    if(global_fit_out.is_open()) global_fit_out.close();

    delete metric;
    auto stop_time = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time);
    log<LOG_INFO>(L"%1% || Total run time: %2%") % __func__ % duration.count();

    return 0;
}


void mcmc_worker(std::vector<Metropolis<simple_target, adaptive_proposal>> &mets, Eigen::VectorXf initial, PROmetric *metric, uint32_t seed, size_t nchains, size_t burnin, size_t steps) {
    std::uniform_int_distribution<uint32_t> dseed(0, std::numeric_limits<uint32_t>::max());
    std::mt19937 rng(seed);
    for(size_t i = 0; i < nchains; ++i) {
        simple_target target{*metric};
        adaptive_proposal proposal(*metric, dseed(rng));
        mets.emplace_back(target, proposal, initial, dseed(rng));
        mets.back().run(burnin, steps);
    }
}

GlobalFitResult do_a_fit(const PROconfig &config, const PROpeller &prop, const PROdata &data, PROmetric *metric, const Eigen::VectorXf &ub, const Eigen::VectorXf &lb, const PROfitterConfig &fit_config, const Eigen::VectorXf &CVParams, const PROspec &cv, const std::vector<int> &global_fixed, GlobalFitOptions opt) {
    GlobalFitResult res(ub, lb, fit_config);
    metric->setBounds(lb, ub);

    log<LOG_INFO>(L"%1% || ########### Starting Global Best Fit Minimizing ############") % __func__;

    std::vector<std::pair<int, std::string>> progress_configs;
    progress_configs.push_back({fit_config.n_latin_points, "(1) LatinHyperCube"});
    progress_configs.push_back({fit_config.n_swarm_iterations, "(2) ParticleSwarm"});
    progress_configs.push_back({fit_config.n_localfit, "(3) BestLBFGSB"});
    progress_configs.push_back({fit_config.harmonic_num_test_points, "(4) HarmonicScan"});
    progress_configs.push_back({100, "(5) HarmonicLBFGSB"});
    MultiPROgressBar progress(progress_configs);

    bool progress_bar = (opt & GlobalFitOptions::Progress) != GlobalFitOptions::Default;
    if(progress_bar){
        progress.initialize_display();
        progress.start_display_thread(); 
        res.fitter.setProgressBar(&progress);
    }

    float best_chi2 = res.fitter.Fit(*metric, CVParams); 
    Eigen::VectorXf best_fit = res.fitter.best_fit;
    if((opt & GlobalFitOptions::FreqSeedPts) != GlobalFitOptions::Default) res.fitter.calcFreqSeedPoints(*metric);

    for(size_t i=0; i< res.fitter.freq_seed_points.size(); i++){
        float chi_freq = res.fitter.freq_seed_values.at(i);
        if( chi_freq < best_chi2){
            log<LOG_INFO>(L"%1% || One of the harmonics of first pass best fit, is a lower chi :  %2% ") % __func__ % res.fitter.freq_seed_values.at(i);
            log<LOG_INFO>(L"%1% || -- at params:  %2% ") % __func__ % res.fitter.freq_seed_points.at(i);
            best_chi2 = chi_freq;
            best_fit = res.fitter.freq_seed_points.at(i);
        }
    }
    res.chi2 = best_chi2;   
    if(progress_bar) progress.finish_all();

    if (res.fitter.exception_string_map.empty()) {
        log<LOG_INFO>(L"%1% || No exceptions were caught from LBFGSB [ --INFO-- ]") % __func__;
    } else {
        log<LOG_INFO>(L"%1% || Some exceptions were caught in LBFGSB [ --INFO-- ]") % __func__;
        for (const auto &[msg, count] : res.fitter.exception_string_map) {
            log<LOG_INFO>(L"%1% ||  -- Exception \"%2%\" occurred %3% time(s)") % __func__ % msg.c_str() % count;
        }
    }

    log<LOG_INFO>(L"%1% || ################################################") % __func__;
    log<LOG_INFO>(L"%1% || ########### Global Best Fit Results ############") % __func__;
    log<LOG_INFO>(L"%1% || ################################################") % __func__;
    log<LOG_INFO>(L"%1% || Global Best Fit chi^2: %2%") %__func__ % best_chi2;
    log<LOG_INFO>(L"%1% || at paramters: ") % __func__;

    size_t N_phys_params = metric->GetModel().nparams;
    size_t N_nuisance = metric->GetSysts().GetNSplines();
    size_t N_params = N_phys_params + N_nuisance;

    for(size_t i = 0; i< N_params; i++){

        if(i<N_phys_params){
            log<LOG_INFO>(L"%1% || %2%  : %3% (log) %4% (nonlog) ") % __func__ % metric->GetModel().pretty_param_names[i].c_str() % best_fit(i) % pow(10,best_fit(i));
        }else{
            log<LOG_INFO>(L"%1% || %2%  :  %3% ") % __func__ % config.m_mcgen_variation_plotname_map.at(metric->GetSysts().spline_names[i-N_phys_params]).c_str() % best_fit(i);
        }
    }
    log<LOG_INFO>(L"%1% || ################################################") % __func__;

    {
        Eigen::VectorXf bf_spec_full = FillSpectra(config, prop, metric->GetSysts(), metric->GetModel(), best_fit, true, config.i_prime).Spec();
        Eigen::VectorXf bf_spec_coll = CollapseMatrix(config, bf_spec_full);
        logLowPredictionBins(config, bf_spec_coll, data.Spec(), 1.0f, config.i_prime);
    }

    std::uniform_int_distribution<uint32_t> dseed(0, std::numeric_limits<uint32_t>::max());
    if((opt & GlobalFitOptions::Correlations) != GlobalFitOptions::Default) {
        log<LOG_INFO>(L"%1% || Starting a metropolis hastings chain to estimate the covariance matrix aroud the above best fit. Run and Burn is (%2%,%3%);") % __func__%fit_config.MCMCiter % fit_config.MCMCburn;

        std::vector<int> fixed;
        for(size_t i = 0; i< global_fixed.size();i++){
            if(global_fixed.at(i) == 1)
                fixed.push_back(i);
        }
        res.mh.emplace(simple_target{*metric}, adaptive_proposal(*metric, dseed(PROseed::global_rng), fixed), best_fit, dseed(PROseed::global_rng));

        res.covmat = Eigen::MatrixXf::Constant(N_params, N_params, 0);
        size_t count = 0;
        const auto action = [&](const Eigen::VectorXf &value) {
            res.covmat += (value-best_fit) * (value-best_fit).transpose();
            count += 1;
        };
        std::optional<PROgressBar> mh_pbar;
        if((opt & GlobalFitOptions::Progress) != GlobalFitOptions::Default) 
            mh_pbar.emplace(int(fit_config.MCMCburn + fit_config.MCMCiter), 30, "MCMC postfit");
        res.mh->run(fit_config.MCMCburn,fit_config.MCMCiter, action, mh_pbar ? &*mh_pbar : nullptr);

        res.covmat /= count;
        Eigen::VectorXf inv_best_fit = best_fit.array().abs().max(1e-10f).inverse();
        res.fraccovmat = inv_best_fit.asDiagonal() * res.covmat * inv_best_fit.asDiagonal();

        Eigen::VectorXf inv_sqrt_diag = res.fraccovmat.diagonal().array().abs().max(1e-10f).sqrt().inverse();
        res.corrmat = inv_sqrt_diag.asDiagonal() * res.fraccovmat * inv_sqrt_diag.asDiagonal();

        log<LOG_INFO>(L"%1% || Finished the metropolis hastings chain ") % __func__;
    }

    bool preerr = (opt & GlobalFitOptions::PrefitErrorBand) != GlobalFitOptions::Default;
    bool mcmcpre = (opt & GlobalFitOptions::MCMCPrefitErrorBand) != GlobalFitOptions::Default;
    bool binwidth_scale = (opt & GlobalFitOptions::BinWidthScaled) != GlobalFitOptions::Default;
    if(preerr || mcmcpre) {
        // Fix physics parameters
        std::vector<int> fixed_pars;
        for(size_t i = 0; i < N_phys_params; ++i) fixed_pars.push_back(i);
        for(size_t i = N_phys_params; i< global_fixed.size();i++){
            if(global_fixed.at(i)==1)fixed_pars.push_back(i);
        }

        log<LOG_INFO>(L"%1% || Starting global getErrorBand() ") % __func__;
        Metropolis mh_pre(prior_only_target{*metric}, adaptive_proposal(*metric, dseed(PROseed::global_rng), fixed_pars), best_fit, dseed(PROseed::global_rng));

        std::optional<PROgressBar> errband_pre_pbar;
        if(progress_bar && mcmcpre) errband_pre_pbar.emplace(int(fit_config.MCMCburn + fit_config.MCMCiter), 30, "MCMC prefit");
        res.err_band =
            mcmcpre
            ? getMCMCErrorBand(mh_pre, fit_config.MCMCburn, fit_config.MCMCiter, config, prop, *metric, best_fit, res.priors, res.prior_covariance, res.prior_param_lo, res.prior_param_hi, binwidth_scale, config.i_prime, errband_pre_pbar ? &*errband_pre_pbar : nullptr)
            : getErrorBand(config, prop, metric->GetSysts(), metric->GetModel(), cv ,CVParams, binwidth_scale, config.i_prime);
    }

    if((opt & GlobalFitOptions::PostFitErrorBand) != GlobalFitOptions::Default) {
        // Fix physics parameters
        std::vector<int> fixed_pars;
        for(size_t i = 0; i < N_phys_params; ++i) fixed_pars.push_back(i);
        for(size_t i = N_phys_params; i< global_fixed.size();i++){
            if(global_fixed.at(i)==1)fixed_pars.push_back(i);
        }

        Metropolis mh_post(simple_target{*metric}, adaptive_proposal(*metric, dseed(PROseed::global_rng), fixed_pars), best_fit, dseed(PROseed::global_rng));
        log<LOG_INFO>(L"%1% || Starting global getPostFitErrorBand() ") % __func__;
        std::optional<PROgressBar> errband_post_pbar;
        if(progress_bar) errband_post_pbar.emplace(int(fit_config.MCMCburn + fit_config.MCMCiter), 30, "MCMC postfit band");
        res.post_err_band = getMCMCErrorBand(mh_post, fit_config.MCMCburn, fit_config.MCMCiter, config, prop, *metric, best_fit, res.posteriors, res.spline_covariance, res.post_param_lo, res.post_param_hi, binwidth_scale,config.i_prime, errband_post_pbar ? &*errband_post_pbar : nullptr);
    }
    
    return res;
}

