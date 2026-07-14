# PROfit v2.X Walk-Through Tutorial

This tutorial walks through the full PROfit v2.X workflow, from the conceptual
building blocks and XML configuration all the way to Feldman-Cousins and the
PROjector two-stage fit. Every plot shown here can be regenerated with the
companion script [`make_tutorial_plots.sh`](make_tutorial_plots.sh) in this
directory — each plot placeholder below is labelled with the exact output file
that script produces.

> **Note:** This walk-through was developed and tested on the `project-SBN-dev`
> branch of PROfit v2.X. If commands here disagree with `PROfit --help` on your
> checkout, trust the binary.

The configuration used throughout is `working_dir/Neutrino2026/fake_sbn_v2.xml`:
a **fake** two-detector (ND + FD) SBN-style setup with a νe appearance search
(`nueapp` model: `dmsq`, `sinsq2thme`), two channels (`nue`, `numu`), and a
realistic mix of spline, covariance, flat-normalisation, and MC-stat
systematics. The MC files (`fake_sbn_mc_ND.root`, `fake_sbn_mc_FD.root`, ~1 GB
each) live alongside the XML, or ping Mark Ross-Lonergan on Slack for a
tarball. **These are toy files for teaching — do not use for physics results.**

The XML ships pointing at a gpvm path, so localize it once before starting:

```bash
sed "s|/exp/uboone/data/users/markross|/path/to/your/mc/files|g" \
    working_dir/Neutrino2026/fake_sbn_v2.xml > tutorial.xml
```

All commands below use `tutorial.xml`, the tag `TUT`, and a fixed seed so your
numbers should match.

---

## Table of contents

1. [PROfit conceptual introduction](#1-profit-conceptual-introduction)
2. [The PROfit XML](#2-the-profit-xml)
3. [General arguments and how stuff works](#3-general-arguments-and-how-stuff-works)
4. [Subcommand `plot` — exploring your spectra](#4-subcommand-plot--exploring-your-spectra)
5. [Subcommand `global` — fitting and fitter configuration](#5-subcommand-global--fitting-and-fitter-configuration)
6. [Subcommand `profile` — 1D profiled Δχ²](#6-subcommand-profile--1d-profiled-χ²)
7. [Subcommand `surface` — 2D Wilks surfaces and AMR](#7-subcommand-surface--2d-wilks-surfaces-and-amr)
8. [Feldman-Cousins: `fc` and `fc-adaptive`](#8-feldman-cousins-fc-and-fc-adaptive)
9. [PROjector — two-stage pre-fit / projected fits](#9-projector--two-stage-pre-fit--projected-fits)

---

# 1. PROfit conceptual introduction

PROfit is a frequentist fitting framework for short-baseline neutrino
oscillation analyses. You describe your entire analysis — MC files, event
selections, binning, oscillation model, and systematics — in a single XML
file, and PROfit turns that into spectra, covariance matrices, response
splines, and fits.

The data flow, start to finish:

```
XML config
   │  (parsed by PROconfig — owns all binning/bookkeeping)
   ▼
CAF/MC ROOT files  ──►  process  ──►  <TAG>_prop.bin + <TAG>_syst.bin
   │                              (event store + systematics, read ONCE,
   │                               hash-checked against the XML forever after)
   ▼
PROsyst (splines + fractional covariances)      PROpeller (per-event MC)
   │                                                 │
   └───────────────┬─────────────────────────────────┘
                   ▼
   χ² metric (PROchi / PROCNP / Poisson)  ⇐ binds config, MC, systs, model, data
                   ▼
   PROfitter (Latin hypercube → particle swarm → L-BFGS-B)
                   ▼
   global fit / profile / surface / FC / adaptive-FC / PROjector
                   ▼
   <TAG>_<out>_*.root, *.pdf, *.txt
```

A few concepts you should internalize before anything else:

**Channels, subchannels, and fullnames.** A *channel* is a selection with a
binning (e.g. the reconstructed-energy νμ selection); a *subchannel* is a
truth-level component stacked inside it (signal, background, cosmics, …). Each
subchannel has a *fullname* of the form

```
<mode>_<detector>_<channel>_<subchannel>     e.g.  nu_FD_numu_signal
```

Everywhere PROfit accepts a "pattern" (background subtraction, POT scaling,
PROjector channel selection, flat systematics) it does **plain substring
matching** against these fullnames — no regex. `"_ND_"` matches everything in
the near detector; `background` matches every background subchannel in every
channel and detector.

**Variables.** Each channel can carry several binnings (`<bins>` entries):
reconstructed energy, true L/E, true energy, and so on. One of them is the
*fitting variable* (the reconstructed one); the others are used for the
oscillation model (true L/E) and diagnostics. Plots are made for all of them.

**The parameter vector.** A fit point is
`[physics parameters, one parameter per spline systematic]`, in that order.
Physics parameters are mostly in **log10 space** (for `nueapp`:
`log10(Δm²) ∈ [-2, 2]`, `log10(sin²2θμe) ∈ [-10, 0]`); spline parameters are
in **σ units** of their knobs. Covariance-type systematics are *not* fit
parameters — they are marginalized analytically inside the χ² covariance
matrix.

**The χ².** For the default `PROchi`:

```
χ² = Δᵀ M⁻¹ Δ  +  Σ pulls(θ)
M  = stat  +  (collapsed, prediction-scaled fractional covariance)
```

with a Gaussian pull term on every spline parameter (priors and centers
configurable per systematic in the XML). `PROCNP` swaps the statistical term
for the combined-Neyman-Pearson variance, and `Poisson` uses the
Baker-Cousins likelihood-ratio sum (and ignores covariance systematics — it
warns).

**Asimov vs fake data.** Unless you say otherwise, the "data" in every fit is
the central-value expectation itself (Asimov). You can inject an oscillation
signal, shift systematics, Poisson-fluctuate, or generate a full FC-style
pseudo-experiment — all from the command line, no XML edits needed.

---

# 2. The PROfit XML

The XML defines the *entire* analysis. Let's walk through
`fake_sbn_v2.xml` block by block. (For deeper background see the
[XML configuration wiki page](https://github.com/markrosslonergan/Elephant_Vanishes/wiki/Minimizing-PROfit:-XML-configuration).)

### Mode and detectors

```xml
<mode name="nu" />
<detector name="ND" pot="1e+21"/>
<detector name="FD" pot="1e+21"/>
```

Multi-detector configs are first-class: each detector gets its own contiguous
block of bins, and every channel is replicated per detector. The `pot` sets
the target exposure each MC file is scaled to.

### Channels, binnings, and subchannels

```xml
<channel name="nue" plotname="Fake CC #nu_{e} Selection">
    <bins unit="Reconstructed Neutrino Energy [GeV]" min="0.1" max="3.0" nbins="16"/>
    <bins unit="True L/E [km/GeV]" min="0" max="2.5" nbins="200" plot="false"/>
    <bins unit="True Neutrino Energy [GeV]" min="0" max="3" nbins="20" />
    <bins unit="Random Value" min="0" max="1" nbins="50"/>
    <subchannel name="intrinsic" plotname="Intrinsic #nu_{e} CC" color="#34A853"/>
    <subchannel name="background" plotname="#nu_{e} Backgrounds" color="#FF6961"/>
    <subchannel name="fullosc" plotname="#nu_{#mu}#rightarrow#nu_{e} (full osc)" color="#4285F4"/>
    <subchannel name="cosmic" plotname="Cosmics" color="#E37400"/>
</channel>
```

Each `<bins>` entry is one *variable*. The **first** `<bins>` entry is the
fitting variable (reconstructed energy here); the rest are extra binnings
that PROfit tracks and plots for you (`plot="false"` suppresses plotting of a
variable, useful for the 200-bin true-L/E binning that only exists for the
oscillation model). Subchannel `plotname` and `color` control the stacked
histograms in `plot`.

### The oscillation model

```xml
<model tag="nueapp">
    <rule index="0" name="No Osc"/>
    <rule index="1" name="Nue Appearance"/>
    <parameter name="L/E" variable_index="1"/>
</model>
```

`tag` selects the physics model implemented in `inc/PROmodel.h`
(`numudis` = 3+1 νμ disappearance with parameters `dmsq`, `sinsq2thmm`;
`nueapp` = 3+1 νμ→νe appearance with `dmsq`, `sinsq2thme`). Each *rule* is an
oscillation-weight function; every MC branch declares which rule applies to
it. The `<parameter>` line tells the model which channel variable holds true
L/E (variable index 1, i.e. the second `<bins>` entry).

### MC files and branches

```xml
<MCFile treename="events/selected" filename=".../fake_sbn_mc_ND.root" scale="1.0" pot="1e+21">
    <friend treename="events/multisigmaTree" />
    <friend treename="events/multisimTree" />
    <friend treename="events/variationTree" />
    <branch
        associated_subchannel = "nu_ND_nue_intrinsic"
        model_rule            = "0"
        additional_weight     = "5*mcweight*(category == 0)"
    >
        <variable>reco_visible_energy</variable>
        <variable>true_baseline/(1000*true_neutrino_energy)</variable>
        <variable>true_neutrino_energy</variable>
        <variable>random_value</variable>
    </branch>
    ...
</MCFile>
```

Each `<branch>` fills one subchannel: `additional_weight` is an arbitrary
TTree formula (here also doing the truth-category selection), `model_rule`
picks the oscillation rule, and the `<variable>` list maps one formula to
each `<bins>` entry of the channel, **in order**. `<friend>` trees carry the
systematic weight branches. Branches with `incl_systematics="false"` (the
cosmics here) get no systematic variations at all.

### Systematics: the `<variation_list>`

```xml
<variation_list>
    <allowlist type="mcstat" plotname="MC Stats" tag="other">MCStat</allowlist>
    <allowlist type="spline" binning="var0" plotname="Flux1" tag="flux">Flux1</allowlist>
    ...
    <allowlist type="covariance" plotname="RPA_CCQE" tag="QE-MEC">RPA_CCQE</allowlist>
    ...
    <allowlist type="norm" plotname="FluxNorm_ND" tag="other">nu_ND_numu:0.02</allowlist>
</variation_list>
```

The main `type`s:

| type | Source | Becomes | Fit parameter? |
|---|---|---|---|
| `spline` | multisigma knobs (±1,2,3σ universes) | per-bin cubic response spline | yes, one per systematic, in σ units |
| `covariance` | multisim universes | fractional covariance matrix | no — marginalized analytically |
| `norm` / `flat` | you, in the XML | flat normalisation covariance on matching subchannels (`pattern:fraction`) | no |
| `mcstat` | MC statistics | diagonal MC-stat covariance | no |

Extra per-systematic attributes worth knowing: `prior=` / `center=` override
the default N(0,1) Gaussian pull on a spline, `<correlation>` blocks make
spline priors correlated, `binning=` lets a spline live on its own coarser
binning, and `mode="covariance_to_spline"` with `num_decomp_knobs=` promotes
a covariance to its leading eigenmode splines (the same machinery PROjector
uses — see section 9).

Note the flat norms use the substring convention: `nu_ND_numu:0.02` is a 2%
normalisation on every subchannel whose fullname contains `nu_ND_numu`.

---

# 3. General arguments and how stuff works

The standard invocation is always

```bash
PROfit -x tutorial.xml -t TUT [GLOBAL OPTIONS] SUBCOMMAND [SUBCOMMAND OPTIONS]
```

Run `PROfit --help` for the full list. The subcommands in v2.X:

```
Subcommands:
  process      PROcess the MC and systematics in root files into binary data for future rapid loading.
  surface      Make a 2D surface scan of two physics parameters, profiling over all others.
  profile      Make a 1D profiled chi2 for each physics and nuisence parameter.
  plot         Make plots of CV, or injected point with error bars and covariance.
  fc           Run Feldman-Cousins for this injected signal
  fc-adaptive  Adaptive Feldman-Cousins. Sub-modes: build-mesh, init-bank, print-bank, asimov, brazil.
  global       Just do a single global fit.
  mcmc         Get bayesian posteriors using MCMC
  scale-test   Run timing benchmarks for FillSpectra / metric / fit hot paths.
```

### Tags, outputs, and the binary cache

- `-t/--tag TUT` names one *processing* of the ROOT files. The first time any
  subcommand runs it creates `TUT_prop.bin` (the event store) and
  `TUT_syst.bin` (the systematics) in the current directory; every later run
  with the same tag loads these instead of touching ROOT files. You almost
  never need to run `process` explicitly.
- A MurmurHash of the XML is stored inside the binaries. Change the XML and
  keep the tag, and PROfit refuses to load with a hash-mismatch ERROR — rerun
  `process` (or change the tag). `--force` overrides the check; be careful.
- `-o/--output v1` is a *secondary* label that goes into every output
  filename (`<tag>_<out>_...`) but does **not** touch the binaries. Use it to
  run many studies off one processing without overwriting results.

```bash
PROfit -x tutorial.xml -t TUT --log process.log process
# → TUT_prop.bin  TUT_syst.bin   (~a minute for this config, once)
```

### Data, injection, and fake-data options

By default the fit data is the Asimov CV. You can build essentially any fake
dataset from the command line:

| Option | What it does |
|---|---|
| `-i/--inject dmsq 1 sinsq2thme 0.01` | inject an oscillation signal as truth (**name value pairs** in v2.X, in linear units) |
| `--inject-systs Flux1 1.0 DetSys2 -2.0` | shift spline systematics (in σ) into the fake data |
| `--inject-cv` / `--inject-systs-cv` | same, but shift the *prediction* CV instead of the data |
| `--poisson-throw` | Poisson-fluctuate the fake data |
| `--pseudo-experiment` | full FC-style throw: spline pulls + covariance Cholesky shift + Poisson stats (combines with `--inject`) |
| `-d/--data file` | load real data from a separate file/XML (plot subcommand) |
| `--use-fake-data` | ignore any embedded/external data and force MC fake data |
| `--scale ND 0.5` | scale POT of subchannels matching a pattern |
| `--seed 405` | fix the RNG seed (default -1 = hardware random) |

Reproducibility rule of thumb: `--seed N` plus `-n 1` is bit-reproducible;
multithreaded runs are statistically equivalent but not byte-identical.

### Controlling the systematics and parameters

| Option | What it does |
|---|---|
| `--syst-list Flux1 Flux2` | use ONLY these systematics |
| `--exclude-systs RPA_CCQE` | use everything except these |
| `--fix dmsq Flux1` | fix parameters at CV (physics or splines) |
| `--syst-only` | fix ALL physics parameters (nuisance-only fit) |
| `--statonly` | drop systematics entirely |
| `--shapeonly` / `--rateonly` | shape-only or single-bin-normalisation analysis |
| `-c/--chi2 PROchi\|PROCNP\|Poisson` | χ² metric (default PROchi) |
| `--grad-mode central-lin` | gradient strategy: `central-full` (most accurate) / `one-sided-full` / `central-lin` (default, Gauss-Newton, 5-10× faster) / `one-sided-lin` |

`--grad-mode central-lin` is exact at minima and fine for scans; use
`central-full` for final publication-quality runs.

### Housekeeping

`-n/--nthread N` parallelizes all fitting code. `-l/--log file.log` +
`-w/--file-verbosity 4` writes a full debug log while keeping the terminal at
`-v 3`. `-b/--progress` adds a progress bar where applicable. `-m/--max N`
truncates the MC event loop (quick tests).

---

# 4. Subcommand `plot` — exploring your spectra

```
Usage: PROfit plot [OPTIONS]
Options:
  --with-splines              Include graphs of splines in output.
  --bkg-subtract TEXT         Substring pattern; that background's CV is subtracted
                              from data and CV at plot time (publication convention).
```

Plus the relevant global options: `--area-norm`, `--scale-by-width`,
`--plot-bounds ymax 100 ratmin 0.5 ratmax 1.5`, and all the injection
machinery from section 3.

### The CV and error band

```bash
PROfit -x tutorial.xml -t TUT -o plotcv --seed 405 plot
```

Outputs (one `Variable_<i>` set per plotted binning of each channel):

* `TUT_plotcv_PROplot_Variable_0_CV.pdf` — stacked CV spectra, fitting variable
* `TUT_plotcv_PROplot_Variable_0_ErrorBand.pdf` — CV + full systematic band
* `TUT_plotcv_PROplot_Variable_2_*.pdf`, ... — same for the other variables
* `TUT_plotcv_PROplot_Covar.pdf` — all covariance matrices, per systematic and total
* `TUT_plotcv_fractional_systematics.pdf` — fractional uncertainty per bin, broken down by systematic `tag`
* `TUT_plotcv_ratio_fractional_systematics.pdf` — same as a ratio
* `TUT_plotcv_PROplot.root` — everything above as ROOT objects

> 📷 **PLOT PLACEHOLDER** — `TUT_plotcv_PROplot_Variable_0_CV.pdf`
> (stacked CV, reconstructed energy, ND+FD νe and νμ channels)
> <!-- <img src="UPLOAD_URL_HERE" width="600"/> -->

> 📷 **PLOT PLACEHOLDER** — `TUT_plotcv_PROplot_Variable_0_ErrorBand.pdf`
> (CV + total systematic error band; "data" points are the Asimov CV since we injected nothing)
> <!-- <img src="UPLOAD_URL_HERE" width="600"/> -->

> 📷 **PLOT PLACEHOLDER** — `TUT_plotcv_PROplot_Covar.pdf` (page with the total collapsed covariance/correlation)
> <!-- <img src="UPLOAD_URL_HERE" width="600"/> -->

> 📷 **PLOT PLACEHOLDER** — `TUT_plotcv_fractional_systematics.pdf`
> (per-bin fractional uncertainty by systematic tag: flux / xsec / det / QE-MEC / RES / other)
> <!-- <img src="UPLOAD_URL_HERE" width="600"/> -->

### Injecting a signal

Inject a 1 eV² sterile with sin²2θμe = 0.01 (note the v2.X **name value**
pair syntax, linear units):

```bash
PROfit -x tutorial.xml -t TUT -o plotinj --seed 405 -i dmsq 1 sinsq2thme 0.01 plot
```

You now additionally get `TUT_plotinj_PROplot_Osc.pdf` (oscillated vs
unoscillated spectra) and the "data" points in the error-band plot become the
injected fake data.

> 📷 **PLOT PLACEHOLDER** — `TUT_plotinj_PROplot_Osc.pdf`
> <!-- <img src="UPLOAD_URL_HERE" width="600"/> -->

> 📷 **PLOT PLACEHOLDER** — `TUT_plotinj_PROplot_Variable_0_ErrorBand.pdf`
> (error band with injected-signal fake data)
> <!-- <img src="UPLOAD_URL_HERE" width="600"/> -->

### Injecting systematic shifts

Shift `Flux1` up by 1σ and `DetSys2` down by 2σ on top of the signal:

```bash
PROfit -x tutorial.xml -t TUT -o plotsyst --seed 405 \
    -i dmsq 1 sinsq2thme 0.01 --inject-systs Flux1 1.0 DetSys2 -2.0 plot
```

> 📷 **PLOT PLACEHOLDER** — `TUT_plotsyst_PROplot_Variable_0_ErrorBand.pdf`
> (fake data now includes both the signal and the systematic shifts)
> <!-- <img src="UPLOAD_URL_HERE" width="600"/> -->

### Looking at the splines themselves

```bash
PROfit -x tutorial.xml -t TUT -o plotspl --seed 405 plot --with-splines
```

adds `TUT_plotspl_PROplot_Spline.pdf`: per-bin response splines for every
spline systematic, with the knob points overlaid — the single most useful
plot for debugging a suspicious multisigma input.

> 📷 **PLOT PLACEHOLDER** — `TUT_plotspl_PROplot_Spline.pdf` (example page)
> <!-- <img src="UPLOAD_URL_HERE" width="600"/> -->

### Background subtraction

`--bkg-subtract PATTERN` applies the publication convention: the matched
background's CV is subtracted from both data and prediction, the error band
becomes signal-only (each throw's own background is subtracted, so background
variations cancel), and the background's uncertainty moves onto the data
points as √(N + σ²_bkg-syst + σ²_bkg-MCstat).

```bash
PROfit -x tutorial.xml -t TUT -o plotbsub --seed 405 plot --bkg-subtract background
```

> 📷 **PLOT PLACEHOLDER** — `TUT_plotbsub_PROplot_Variable_0_ErrorBand.pdf`
> (background-subtracted spectra; compare with the plotcv version)
> <!-- <img src="UPLOAD_URL_HERE" width="600"/> -->

Other useful toggles to try yourself: `--area-norm`, `--scale-by-width`,
`--scale FD 0.5` (half the far-detector POT), `--poisson-throw`.

---

# 5. Subcommand `global` — fitting and fitter configuration

`global` performs one full global best fit of all physics + spline parameters
and draws the post-fit results. It takes no subcommand options of its own —
everything is controlled by the global arguments.

```bash
PROfit -x tutorial.xml -t TUT -o glob1 --seed 405 -n 8 \
    -i dmsq 1 sinsq2thme 0.01 --log glob1.log global
```

Outputs:

* `TUT_glob1_global_fit.txt` — best-fit χ² and every parameter value in a plain-text table
* `TUT_glob1_PROglobal_hists.pdf` — pre-fit (blue/gray) vs post-fit (red) spectra + error bands, with a data/fit ratio panel
* `TUT_glob1_PROglobal_postfit_correlation_matrix.pdf` (+ `_nuisance_only` version) — post-fit parameter correlations
* `TUT_glob1_PROglobal_postfit_posteriors.pdf` — post-fit parameter constraints
* `TUT_glob1_PROglobal.root` — all of the above as ROOT objects

> 📷 **PLOT PLACEHOLDER** — `TUT_glob1_PROglobal_hists.pdf`
> (pre-fit vs post-fit spectra with ratio panel — your at-a-glance goodness-of-fit)
> <!-- <img src="UPLOAD_URL_HERE" width="700"/> -->

> 📷 **PLOT PLACEHOLDER** — `TUT_glob1_PROglobal_postfit_correlation_matrix.pdf`
> <!-- <img src="UPLOAD_URL_HERE" width="600"/> -->

> 📷 **PLOT PLACEHOLDER** — `TUT_glob1_PROglobal_postfit_posteriors.pdf`
> <!-- <img src="UPLOAD_URL_HERE" width="600"/> -->

The text output looks like:

```
################################################
########### Global Best Fit Results ############
################################################
Global Best Fit chi^2: <...>
at paramters:
#Deltam^{2}          :  <log10 value>
sin^{2}2#theta_{#mue}:  <log10 value>
Flux1                :  <sigma>
...
```

Remember the physics values are printed in **log10 space** for this model.

### How the fit actually works

Every fit in PROfit is a three-stage pipeline:

1. **Latin hypercube sampling** (`n_latin_points`): random-but-space-filling
   starting points across all parameters, filtered for diversity
   (`latin_diversity_factor`).
2. **Particle swarm optimization** (`n_swarm_particles`,
   `n_swarm_iterations`): the best hypercube points explore globally.
3. **L-BFGS-B local fits** (`n_localfit`): full gradient-based minimizations
   from the best swarm point and each seed point; best result wins.

After the global fit, PROfit runs a **harmonic seed search** over Δm² (the χ²
is quasi-periodic in log Δm², so degenerate local minima are found by a
frequency scan) — those seeds are handed to every subsequent profile/surface
fit, which is a big part of why scans are robust to the multi-modal
oscillation landscape.

### Fit presets and fit options

There are two independent fitter configurations: the **global** fit (done
once — be careful) and the **scan** fit (done thousands of times in
profile/surface — be fast). Configure them with:

```bash
--preset fast              # ONE value sets both global and scan config
--preset good overkill     # first = global, second = scan
--fit-options n_latin_points 2000 max_iterations 5000     # global fit knobs
--scan-fit-options n_localfit 2 ...                        # scan fit knobs
```

Presets are `fast`, `good` (default), `overkill`, and `sensitivity`.

> ⚠️ **CLI11 trap:** if you pass a single preset (`--preset fast`) make sure
> the *next* token is a flag or the subcommand is protected — greedy vector
> parsing can eat a following bare word. `--preset fast --seed 405 ... global`
> is safe; when in doubt put `--preset` before other options.

Run `PROfit --fit-help -x anything` for the full annotated parameter list; the
highlights:

```
------ PROfitter Specific Parameters ------
  n_latin_points          : Number of Latin hypercube points sampled across all parameters
  latin_diversity_factor  : 0 = no distance weighting, 1 = most diverse points
  n_localfit              : Total number of L-BFGS-B fits after PSO
  n_max_local_retries     : Retries if L-BFGS-B throws
------ Particle Swarm Optimization ------
  n_swarm_particles, n_swarm_iterations, n_swarm_max_stagnent_iterations,
  swarm_inertia_start/end, swarm_cognitive_score, swarm_social_score, ...
------ Harmonic Seed Search ------
  harmonic_min/max_num_seeds, harmonic_num_test_points, ...
------ L-BFGS-B ------
  m, epsilon, epsilon_rel, past, delta, max_iterations, max_linesearch, ...
```

(L-BFGS-B parameter meanings: see the
[optimizer wiki page](https://github.com/markrosslonergan/Elephant_Vanishes/wiki/L%E2%80%90BFGS%E2%80%90B-Optimizer-Parameter-Descriptions).)

One benign scary-looking thing you WILL see in logs: L-BFGS-B throwing
`"line search step became smaller than minimum"` on near-optimal starting
points. This is routine (the seed's own χ² is kept as a candidate), not a
failure.

### Metric choice matters

```bash
PROfit -x tutorial.xml -t TUT -o globcnp --seed 405 -n 8 -c PROCNP global
PROfit -x tutorial.xml -t TUT -o globpoi --seed 405 -n 8 -c Poisson global
```

`PROCNP` is recommended when bins can be low-statistics; `Poisson` ignores
covariance systematics entirely (it will warn you).

---

# 6. Subcommand `profile` — 1D profiled Δχ²

```
Usage: PROfit profile [OPTIONS]
Options:
  --mcmc-prefit        Use MCMC to sample the systematic priors for the pre-fit error band.
  --probe              Use PRObe adaptive importance sampling instead of the legacy 18-uniform scan.
  --probe-chunks INT   With --probe, split each physics scan into N parallel chunks.
  --profile-timing     Emit a scan-timing summary (diagnostic).
```

`profile` first runs the full global fit (identical to `global`), then scans
**every parameter one at a time** — each physics parameter and each spline —
profiling (re-minimizing) over all the others at each scan point.

```bash
PROfit -x tutorial.xml -t TUT -o prof1 --seed 405 -n 8 \
    -i dmsq 1 sinsq2thme 0.01 --log prof1.log profile
```

Outputs:

* `TUT_prof1_PROfile.pdf` — the Δχ² curve for every parameter (physics first, then all nuisances with the prior in dashed red)
* `TUT_prof1_PROfile_1sigma.pdf` (+ `_1sigma_detailed.pdf`) — post-fit ±1σ summary for all nuisance parameters at a glance
* `TUT_prof1_PROfile_hists.pdf` — pre-fit vs post-fit spectra (same as global's)
* `TUT_prof1_PROfile_postfit_correlation_matrix.pdf`, `_postfit_posteriors.pdf`
* `TUT_prof1_PROfile.root` — every profile as a TGraph, plus the 1σ summary
* `TUT_prof1_global_fit.txt` — the global best fit table

> 📷 **PLOT PLACEHOLDER** — `TUT_prof1_PROfile.pdf`
> (per-parameter profiled Δχ²; top row = physics, rest = nuisances vs their priors)
> <!-- <img src="UPLOAD_URL_HERE" width="1100"/> -->

> 📷 **PLOT PLACEHOLDER** — `TUT_prof1_PROfile_1sigma.pdf`
> (±1σ nuisance summary; star = injected, black = best fit)
> <!-- <img src="UPLOAD_URL_HERE" width="800"/> -->

> 📷 **PLOT PLACEHOLDER** — `TUT_prof1_PROfile_hists.pdf`
> <!-- <img src="UPLOAD_URL_HERE" width="700"/> -->

How to read `PROfile.pdf`: each nuisance panel shows the profiled Δχ²
(black) against the dashed-red prior (a 1σ Gaussian pull by construction).
A post-fit curve narrower than the prior means the data constrains that
systematic; a shifted minimum means the fit pulled it (watch for correlated
pairs sharing a pull, e.g. two normalisations covering for each other).

### PRObe: the adaptive scan

The legacy scan evaluates a fixed 18-point grid per parameter. `--probe`
replaces this with PRObe, an adaptive algorithm that anchors the minimum,
fits a quadratic surrogate, and bisects to the Δχ²=1 crossings directly —
usually fewer fits *and* more accurate 1σ bounds, especially for strongly
constrained nuisances. For physics parameters (which can be spiky in Δm²) it
uses a spike-safe coarse grid + refinement instead.

```bash
PROfit -x tutorial.xml -t TUT -o probe1 --seed 405 -n 8 \
    -i dmsq 1 sinsq2thme 0.01 profile --probe
```

> 📷 **PLOT PLACEHOLDER** — `TUT_probe1_PROfile.pdf`
> (PRObe version — compare point placement with the legacy scan above)
> <!-- <img src="UPLOAD_URL_HERE" width="1100"/> -->

If the two physics-parameter scans are your wall-time bottleneck and you have
threads to spare, `--probe-chunks 4` splits each physics scan across threads
(the cross-thread seed bank keeps the chunks cooperating).

### Nuisance-only profiling

`--syst-only` fixes the physics at CV and profiles just the nuisances —
useful for constraint studies:

```bash
PROfit -x tutorial.xml -t TUT -o profso --seed 405 -n 8 --syst-only profile --probe
```

---

# 7. Subcommand `surface` — 2D Wilks surfaces and AMR

`surface` maps Δχ² over a 2D grid of two physics parameters, profiling over
everything else at each point. Contours at Wilks-theorem critical values
(Δχ²=2.30/5.99/... for 2 dof) give your confidence regions — see section 8
for when Wilks isn't good enough.

Selected options (run `PROfit ... surface --help` for all):

```
-g,--grid INT [40]        Grid size (one value = square, two = rectangular)
--xvar / --yvar           Which physics parameters on which axes
--xlo --xhi --ylo --yhi   Axis ranges (or --xlims/--ylims)
--logx/--linx --logy/--liny   Axis scaling (default log)
--xlabel --ylabel         Axis labels
--brazil-band             1000 stats+systs throws → median ±1σ/±2σ sensitivity bands
  --stat-throws / --single-throw / --only-throw / --from-many
--curve-mode FLOAT ...    1D PROcurve from param A to param B
--surface-amr             Adaptive mesh refinement instead of the dense grid
  --amr-initial INT [10]  AMR coarsest grid (NxN)
  --amr-levels INT [3]    Refinement depth (resolution ≈ initial * 2^levels)
  --amr-delta FLOAT [0.5] Straddle-band widening in chi^2 units
  --amr-levels-chi2 ...   Target Delta-chi^2 contours (default 5.99)
```

**For the `nueapp` model you must set the axes** — the built-in defaults are
for the νμ-disappearance model:

```bash
AXES="--xvar sinsq2thme --yvar dmsq --xlo 1e-4 --xhi 1 --ylo 1e-2 --yhi 1e2"
```

### A dense Asimov sensitivity surface

```bash
PROfit -x tutorial.xml -t TUT -o surf1 --seed 405 -n 8 --log surf1.log \
    surface -g 30 $AXES
```

This is 900 profiled fits, so it's the first genuinely slow thing in the
tutorial — grid size and thread count are your levers. Outputs:

* `TUT_surf1_surface.txt` — plain-text grid: `xval yval chi2 p0 p1 ...` (the best-fit nuisance values at every point)
* `TUT_surf1_surf.root` — the same as a TTree plus a TH2D
* `TUT_surf1_surface.pdf` — a quick-look contour plot
* `TUT_surf1_global_fit.txt` — the global best fit

> 📷 **PLOT PLACEHOLDER** — `TUT_surf1_surface.pdf`
> (Asimov sensitivity, 30×30 grid, 90%/95% CL contours)
> <!-- <img src="UPLOAD_URL_HERE" width="600"/> -->

We stress: PROfit's job is to give you the *surface data*; ROOT is not the
place to make pretty contour plots. The `.txt` output loads trivially into
matplotlib — see `working_dir/Tutorial_V2.0.1_allincl/PROfit_Contour_plotter.ipynb`
for a ready-made notebook, and everything you learned in the old v1 tutorial
about overlaying `--syst-list` / `--exclude-systs` / `--statonly` variants
still applies:

```bash
PROfit -x tutorial.xml -t TUT -o surfstat --seed 405 -n 8 --statonly surface -g 30 $AXES
PROfit -x tutorial.xml -t TUT -o surfnoflux --seed 405 -n 8 --exclude-systs Flux1 Flux2 Flux3 surface -g 30 $AXES
```

### Adaptive mesh refinement: `--surface-amr`

A dense grid wastes almost all its fits far from the contour you care about.
`--surface-amr` starts from a coarse grid and recursively refines only the
cells that straddle (within `--amr-delta`) one of your target Δχ² levels:

```bash
PROfit -x tutorial.xml -t TUT -o surfamr --seed 405 -n 8 --log surfamr.log \
    surface $AXES --surface-amr --amr-initial 10 --amr-levels 3 \
    --amr-levels-chi2 2.30 5.99
```

Effective resolution along the contour is `amr_initial × 2^amr_levels`
(here 80×80) for roughly the cost of the coarse grid plus a band around the
contours — typically a **6-8× wall-time win** at equivalent contour quality.
The scan writes `TUT_surfamr_surface_amr.txt` (same column format, one row
per evaluated mesh point) alongside the usual `_surf.root` / `_surface.pdf`.

> 📷 **PLOT PLACEHOLDER** — `TUT_surfamr_surface.pdf`
> (AMR surface; note the refined mesh hugging the 1σ/2σ contours)
> <!-- <img src="UPLOAD_URL_HERE" width="600"/> -->

### Brazil bands

`--brazil-band` repeats the sensitivity estimate over 1000 statistical +
systematic throws of the data and draws the median and ±1σ/±2σ envelope of
the resulting contours — the classic sensitivity band. This is ~1000× the
cost of one surface, so use AMR, threads, and patience (or `--stat-throws`
for stat-only bands, `--single-throw` to test the machinery, and
`--from-many file1 file2 ...` to merge throw files from separate jobs run in
parallel on a cluster).

```bash
PROfit -x tutorial.xml -t TUT -o surfbrz --seed 405 -n 16 --log surfbrz.log \
    surface $AXES --surface-amr --amr-initial 10 --amr-levels 2 --brazil-band
```

> 📷 **PLOT PLACEHOLDER** — `TUT_surfbrz_surface.pdf`
> (median sensitivity with ±1σ/±2σ Brazil bands)
> <!-- <img src="UPLOAD_URL_HERE" width="600"/> -->

---

# 8. Feldman-Cousins: `fc` and `fc-adaptive`

Wilks' theorem (Δχ² cuts of 2.30/5.99/...) assumes Gaussian-land: no physical
boundaries, no degenerate minima. Oscillation fits violate both, so for
publication contours you calibrate the Δχ² cut empirically with
Feldman-Cousins pseudo-experiments.

### Classic FC: `fc`

```
Usage: PROfit fc [OPTIONS]
Options:
  -u,--universes UINT [1000]  Number of Feldman Cousins universes to throw
  --gof                       Get GOF pvalue
  --pval                      Get FC pvalue
```

At the injected point (`-i`, or CV if none), `fc` throws `-u` universes —
each one a full pseudo-experiment: Gaussian spline pulls sampled from the
priors, covariance-systematic shifts via Cholesky decomposition, and Poisson
statistics — and for each universe runs both a nuisance-only fit and a free
fit. The distribution of `Δχ² = χ²(fixed) - χ²(free)` is your empirical
critical-value distribution at that point.

```bash
PROfit -x tutorial.xml -t TUT -o fc1 --seed 405 -n 8 \
    -i dmsq 1 sinsq2thme 0.01 --log fc1.log fc -u 500
```

Output is `TUT_fc1_FC.root` containing a TTree with, per universe, the two
χ² values, Δχ², and the best-fit parameters — from which you extract the
90%/95% quantiles and compare to the Wilks values. This is the honest but
brute-force approach: to calibrate a whole *contour* you would repeat it at
every grid point, which is exactly what `fc-adaptive` automates.

### Adaptive FC: `fc-adaptive`

The adaptive-FC pipeline concentrates pseudo-experiments where they matter —
near the contour — instead of uniformly across the plane. It runs in stages
that communicate through binary artifacts, **all keyed by the `-o` output
tag**, so every stage of one study must share the same `-o`:

```
build-mesh  →  <tag>_<out>_mesh.bin      (Wilks prepass: N throws, each an AMR mesh;
                                          cells that many throws refine form the meta-mesh)
init-bank   →  <tag>_<out>_bank.bin      (pseudo-experiment bank: PEs per meta-mesh cell,
                                          doubling with refinement level; re-running ADDS PEs)
print-bank  →  summary PDFs               (bank occupancy diagnostics)
asimov      →  FC contour + verdict PDFs  (classify the Asimov data against the bank)
brazil      →  Brazil-band PDFs           (throw pseudo-data, classify each against the bank)
```

A full small-scale run (bump `--throws` and `--n-pe-min` for real studies):

```bash
AXES="--xvar sinsq2thme --yvar dmsq --xlo 1e-4 --xhi 1 --ylo 1e-2 --yhi 1e2"
AFC="fc-adaptive --throws 25 --prepass-amr-initial 8 8 --prepass-amr-levels 2 $AXES"

# Stage 1: Wilks prepass → meta-mesh
PROfit -x tutorial.xml -t TUT -o afc --seed 405 -n 8 $AFC --mode build-mesh
# Stage 2: fill the PE bank (repeat to add more PEs, capped at --n-pe-max)
PROfit -x tutorial.xml -t TUT -o afc --seed 405 -n 8 $AFC --mode init-bank --n-pe-min 25 --n-pe-max 400
# Stage 3: inspect
PROfit -x tutorial.xml -t TUT -o afc --seed 405 -n 8 $AFC --mode print-bank
# Stage 4: FC-corrected Asimov contour
PROfit -x tutorial.xml -t TUT -o afc --seed 405 -n 8 $AFC --mode asimov
# Stage 5: FC-corrected Brazil band
PROfit -x tutorial.xml -t TUT -o afc --seed 405 -n 8 $AFC --mode brazil --n-brazil-throws 50
```

Key knobs: `--p-thresh` (fraction of prepass throws that must refine a cell
for it to enter the meta-mesh), `--baseline-level` (levels always kept),
`--cl` (target CLs), `--update-layer` / `--update-only-layer` (target which
refinement layers get new PEs on an `init-bank` re-run), `--stat-only-throws`.

Outputs along the way:

> 📷 **PLOT PLACEHOLDER** — `TUT_afc_metamesh.pdf`
> (the meta-mesh: cell refinement levels, concentrated where throws put the contour)
> <!-- <img src="UPLOAD_URL_HERE" width="600"/> -->

> 📷 **PLOT PLACEHOLDER** — `TUT_afc_throws.pdf`
> (the Wilks-prepass throw contours that built the mesh)
> <!-- <img src="UPLOAD_URL_HERE" width="600"/> -->

> 📷 **PLOT PLACEHOLDER** — `TUT_afc_bank_summary.pdf` (PE bank occupancy per level)
> <!-- <img src="UPLOAD_URL_HERE" width="600"/> -->

> 📷 **PLOT PLACEHOLDER** — `TUT_afc_asimov_contour.pdf`
> (FC-corrected contour vs the Wilks contour)
> <!-- <img src="UPLOAD_URL_HERE" width="600"/> -->

> 📷 **PLOT PLACEHOLDER** — `TUT_afc_asimov_verdict.pdf`
> (per-cell FC vs Wilks verdict map)
> <!-- <img src="UPLOAD_URL_HERE" width="600"/> -->

> 📷 **PLOT PLACEHOLDER** — `TUT_afc_brazil_band.pdf`
> (FC-corrected Brazil band from the bank)
> <!-- <img src="UPLOAD_URL_HERE" width="600"/> -->

Determinism note: with `-n 1` and a fixed `--seed` the entire pipeline is
bit-reproducible; multithreaded runs are statistically equivalent.

---

# 9. PROjector — two-stage pre-fit / projected fits

PROjector answers "what does my near detector buy me?" properly. Instead of
fitting ND and FD simultaneously every time, you (1) fit **only** the ND
channels once and save the nuisance posterior, then (2) run any FD study with
those channels masked out and the saved posterior installed as a correlated
prior. Same statistical content as the joint fit (to the Gaussian
approximation), at a fraction of the per-fit cost — which matters enormously
for FC studies.

### Stage 1: the pre-fit

```bash
PROfit -x tutorial.xml -t TUT -o pj --seed 405 -n 8 \
    --projector-prefit "_ND_" global
# → TUT_pj_PROjector_constraint.bin
```

What happens: every subchannel matching `_ND_` (substring, and matches must
cover **whole channels** — χ² lives in collapsed space) is selected; all
covariance-type systematics are *promoted* to their eigenmode splines so the
fit has explicit parameters for them (`--projector-knobs N` limits to the top
N modes + a residual covariance; the default -1 keeps all modes, exact);
physics is fixed at CV; and the fit's posterior — best-fit point θ̂ plus the
full correlated covariance Σ from a finite-difference Hessian — is written to
the constraint file.

Useful stage-1 options:

* `--projector-knobs 20` — promote only the leading 20 eigenmodes per covariance (remainder stays as a residual covariance)
* `--projector-keep-cov MCStat DetSys...` — covariances to *not* promote (e.g. detector-local systematics with no ND/FD correlation; MC-stat is never promoted)
* `--projector-float-physics` — float physics in the pre-fit; the saved posterior is then the physics-marginalized nuisance covariance

### Stage 2: the projected fit

Any subcommand — `global`, `profile`, `surface`, `fc`, `fc-adaptive` — can
run projected:

```bash
PROfit -x tutorial.xml -t TUT -o pjglob --seed 405 -n 8 \
    --projector TUT_pj_PROjector_constraint.bin global

PROfit -x tutorial.xml -t TUT -o pjprof --seed 405 -n 8 \
    --projector TUT_pj_PROjector_constraint.bin profile --probe

PROfit -x tutorial.xml -t TUT -o pjsurf --seed 405 -n 8 \
    --projector TUT_pj_PROjector_constraint.bin \
    surface --surface-amr --amr-initial 10 --amr-levels 3 \
    --xvar sinsq2thme --yvar dmsq --xlo 1e-4 --xhi 1 --ylo 1e-2 --yhi 1e2
```

Stage 2 re-derives the identical eigenmode promotion (name-checked against
the constraint file), masks the pre-fit channels OUT of the χ² (active-bins
mask + zeroed data), and installs (θ̂, Σ) as a fully correlated Gaussian prior
on the promoted spline parameters.

> 📷 **PLOT PLACEHOLDER** — `TUT_pjprof_PROfile_1sigma.pdf`
> (projected nuisance constraints — compare against the joint-fit `TUT_prof1_PROfile_1sigma.pdf`)
> <!-- <img src="UPLOAD_URL_HERE" width="800"/> -->

> 📷 **PLOT PLACEHOLDER** — `TUT_pjsurf_surface.pdf`
> (projected FD-only sensitivity with the ND constraint as prior, vs the joint surface `TUT_surfamr_surface.pdf`)
> <!-- <img src="UPLOAD_URL_HERE" width="600"/> -->

### Rules and closure checks

* **Same XML, same binaries, same systematic selection in both stages** —
  hash-enforced; the constraint file also records the χ² metric (`-c`) and
  refuses a mismatch.
* The pattern must cover whole channels: `--projector-prefit fullosc` (one
  subchannel) or `nu_` (everything) are refused loudly. Detector patterns
  like `"_ND_"` are the intended use.
* Closure signatures worth checking on any new setup: a projected Asimov
  `global` fit gives χ² = 0 at CV; a pre-fit with only FD-blind systematics
  gives posterior widths of exactly 1.
* FC/Brazil throws in projected mode sample the *marginal* widths of the
  constraint only — correlations enter the pull term, not the throws (PROfit
  prints a runtime warning to remind you).

---

# Appendix: regenerating every plot in this tutorial

```bash
# from the repository root, after building:
docs/tutorials/make_tutorial_plots.sh
# heavy extras (Brazil bands, larger FC banks):
RUN_EXPENSIVE=1 docs/tutorials/make_tutorial_plots.sh
```

The script localizes the XML, processes once, and runs every command shown
above with fixed seeds. Outputs land in `docs/tutorials/tutorial_run/` (set
`TUTORIAL_OUTDIR` to change). See the script header for the environment
overrides (`PROFIT_BIN`, `PROFIT_TEST_MCDIR`, `NTHREADS`, `SEED`).
