# PROfit Code Review — `markross_AdaptiveFC` branch

**Scope:** (a) bugs in core code, (b) performance of the hot path — `FillSpectra` and the
`PROmetric::operator()` (`Metric()`) implementations — and (c) the new adaptive
Feldman-Cousins codebase (`PROAdaptiveFC.{h,cxx}`, `PROmesh.{h,cxx}`).

**Method:** three independent deep passes (hot path; adaptive FC; core support) over
`inc/`, `src/`, `bin/PROfit.cxx` and the build system, followed by direct line-level
verification of every high/medium finding. Line numbers refer to the branch state at the
time of review (commit `7d22ba9`); the fixes referenced in "Status" columns are applied on
this branch in commit groups so they can be tested/bisected independently.

**Fix status legend:** `FIXED` = implemented on this branch · `FIXED*` = implemented with a
noted behavioral decision · `DOC` = documented only (physics/API decision left to
maintainers).

---

## Executive summary

The framework is in good shape structurally — clear separation between config, event
storage, systematics, metrics, fitter, and the FC driver — but the review found several
correctness bugs that can silently corrupt physics results (not just crash), and the hot
path spends most of its time on avoidable work.

**The ten most important findings:**

1. Latin-hypercube multistart samples only `[center, center + width/4]` of every bounded
   parameter (and raw `[0,1]` for the spline dimensions) — the global-fit seeding never
   explores most of the parameter box (T1.3).
2. `PROfitter::Fit` can propagate an *uninitialized* chi² into the best-fit result when
   the first L-BFGS-B attempt throws, and its retry logic is dead code because an inner
   `catch` swallows every exception (T1.2).
3. `PROpoisson` returns NaN for any dataset with an empty bin — unguarded `0·log(0)` (T1.4).
4. The adaptive-FC pseudo-experiment generators re-introduce the unbounded rejection loop
   that commit `000b3d0` already fixed in `bin/PROfit.cxx` — covariance-to-spline configs
   can hang `init-bank`/`brazil` forever or read out of bounds (T1.5).
5. `build_meta_mesh` neither tiles nor partitions the parameter plane: it leaves *holes*
   (no PE bank cell) and emits *overlapping* cells, exactly along contours (T1.7).
6. A data race on `corner_counts` in the PROmesh worker pool, and an O(N²)-under-lock
   global-min scan that serializes the whole AMR thread pool (T1.6, T1.11).
7. The default event-by-event `FillSpectra` path writes to bin index −1 for
   out-of-range events (heap corruption) and reads `spline_binnings` out of bounds for
   over-sized parameter vectors (T1.8).
8. The two random-throw helpers use a function-local `static std::mt19937`, so their
   `seed` argument is ignored after the first call and the generator is shared across
   threads (T1.9).
9. In the default `GradientCentralFull` mode, every L-BFGS-B iteration performs
   `1 + 2·nparams` full chi² evaluations, each materializing a dense N×N covariance and
   an O(m³) Cholesky — ~100 dense covariance builds *per iteration* for a 50-spline fit.
   The existing linearised gradient modes avoid almost all of it (PERF-H2).
10. The "release" build compiles without `-DNDEBUG`, leaving Eigen's per-access assertions
    enabled in the hottest loops (PERF-H1).

Deprecated functionality removed at the maintainer's request: the XML `use="false"`
channel switch (it was broken — see T1.1 — and is no longer honored; configs using it now
fail with a clear error).

---

## Tier 1 — critical correctness (silently wrong results or crashes)

### T1.1 — `remove_unused_channel` corrupts binning; `use="false"` removed entirely
`src/PROconfig.cxx:2021–2049`. When any `<channel use="false">` was present, the compacted
copies of the per-variable binning arrays (`temp_channel_other_bins`, per-variable labels
and units) were built but **never assigned back** — only the four scalar name arrays were.
Every downstream consumer (`GetChannelVariableBins`, `CalcTotalBins`,
`construct_variable_collapsing_matrices`) then indexed the *uncompacted* arrays with
*compacted* local channel indices: wrong bin counts, wrong collapse matrices, silently
wrong physics for any non-trailing disabled channel. (`temp_variable_dims` was declared and
never even filled.)

**Resolution (per maintainer):** `use="false"` is deprecated and unused — the
functionality is removed rather than repaired. Channels and subchannels no longer honor
`use`; an XML that sets `use="false"` on either now fails at parse time with an explicit
"deprecated — remove from your XML" error. The `m_channel_bool`/compaction machinery is
deleted. **Status: FIXED\*** (behavioral: configs with `use="false"` now error out).

### T1.2 — `PROfitter::Fit`: uninitialized `fx`, dead retry loop, false success
`src/PROfitter.cxx:211, 255–300` (and the seed-point loop at `:310–345`).
`float fx;` is uninitialized; the local-fit loop wraps `solver.minimize` in an inner
`try { … } catch(const std::exception&)` that records the message and *continues*, so
(1) the outer `catch(const std::runtime_error&)` retry handler can never fire — the retry
loop is dead code; (2) `success = true` is reached even when minimize threw; (3) on a
first-attempt throw, the uninitialized `fx` is pushed into `chi2s_localfits` and compared
`fx < chimin`, so garbage can become `best_fit`/`chimin` (undefined behavior; a bogus
"minimum" can be returned). **Status: FIXED** — `fx` initialized to `+inf`, exceptions
propagate to the retry handler, chi² recorded only on success.

### T1.3 — Latin-hypercube multistart never explores most of the box
`src/PROfitter.cxx:26–42` builds `perm[i] = (i + U(-2,2))/n`, i.e. samples ≈ `[0,1]`
(the ±2-cell jitter also breaks stratification); the comment at `:158` believes the values
are in `[-2,2]`. The rescale (`:159–168`, duplicated in `recenter_latin_samples`
`:44–56`) computes `randpt = pt/4 ∈ [0, 0.25]` → bounded parameters are sampled only in
`[center, center + width/4]`; parameters whose bounds are exactly `(−3,3)` — the nuisance
splines — skip rescaling entirely and are "sampled" in raw `[0,1]`. The multistart
therefore never places a start point below the midpoint of *any* dimension; PSO has to
rescue every fit, and FC critical values inherit whatever bias survives.
**Status: FIXED** — proper LHS strata `(i + U(0,1))/n`, then a single affine map
`[0,1] → [lb,ub]` for every dimension (infinite bounds fall back to a ±2 window around 0,
matching the old effective behavior for splines).

### T1.4 — `PROpoisson` NaN on any empty data bin
`src/PROpoisson.cxx:88` (and `getSingleChannelChi`, `:267`):
`vdata·log(vdata/vmc)` evaluates `0·log(0) = NaN` for a zero-count bin (IEEE: `0·(−inf)`),
and `vmc = 0` produces ±inf. PROchi (`:154`) and PROCNP (`:118`) both guard/throw; PROpoisson
has no guard, so NaN silently reaches L-BFGS-B. Correct Baker–Cousins limit: a zero-data
bin contributes `2·vmc`. **Status: FIXED** — per-bin guarded sum (`n==0 → μ` term;
`μ<=0` with `n>0` → large finite penalty + warning).

### T1.5 — Adaptive-FC PE generators regress the `000b3d0` pseudo-universe fix
`src/PROAdaptiveFC.cxx:1668–1674` (`run_one_pe`) and `:1946–1954`
(`generate_pseudo_experiment_data`) throw nuisances with
`do { θ = N(0,1) } while(θ < tlo || θ > thi)` and unguarded
`spline_has_restrict[i]` access. Commit `000b3d0` fixed exactly this pattern in
`bin/PROfit.cxx:1017` (bounded attempts, size guard, inverted-bounds swap) but the two
adaptive-FC copies were missed: covariance-to-spline systematics (short restrict arrays /
unreachable or inverted bounds) hang `init-bank`/`brazil` forever or read OOB. Meanwhile
`generate_throws` (`:356–359`) applied *no* restriction at all — statistically inconsistent
with the PE throws. **Status: FIXED** — one shared bounded/guarded thrower
(`throw_restricted_spline`) used by all sites, mirroring the `000b3d0` semantics.

### T1.6 — PROmesh: data race on `corner_counts`
`src/PROmesh.cxx:456–457`. Workers execute `corner_counts[cidx]->fetch_sub(…)` *outside*
`state_mu` while other workers `push_back` new cells *under* it (`:111`).
`std::deque::operator[]` navigates the deque's internal block map, which `push_back` may
reallocate → data race/UB (the `fetch_sub` itself is atomic; the container navigation is
not). `cells` itself is safe (copied under lock, `:288–293`). **Status: FIXED** — the
`std::atomic<int>*` pointers are snapshotted under `state_mu` (pointees are stable —
`unique_ptr` targets never move), `fetch_sub` runs on the raw pointer outside the lock.

### T1.7 — `build_meta_mesh` neither tiles nor partitions the plane
`src/PROAdaptiveFC.cxx:551–619`.
*Holes:* the baseline pass skips an entire coarse block if **any** finest point inside it
is covered (`if(any_covered) continue;`, `:590–596`) — but the sub-threshold points inside
partially-covered blocks were never emitted by the refined pass either, so they end up in
**no cell at all** → no PE bank → permanently undecidable regions, precisely along
contours (the common case), papered over by IDW interpolation.
*Overlaps:* the refined pass skips candidates by their **anchor point only** (`:555`) yet
marks whole footprints as seen — a coarser cell whose anchor is unseen but whose footprint
crosses an already-emitted deeper cell is emitted overlapping it (double coverage).
**Status: FIXED** — emission rewritten as an explicit uncovered-region sweep: refined
cells are emitted only when their full footprint is uncovered, and after the refined pass
every still-uncovered finest point is covered exactly once by descending from the baseline
block to the largest uncovered sub-cell containing it. A tiling assertion
(Σ step² == W·H, no double-marks) now runs after the build.

### T1.8 — Unbinned (event-by-event) `FillSpectra`: OOB write and OOB read
`src/PROcess.cxx:324–337`. `reco_bin = VariableBinIndex(...)` is −1 for out-of-range
events (`inc/PROpeller.h:226`); `PROspec::Fill` does no bounds check
(`inc/PROspec.h:191`) → `spec(-1) += w` is UB. (`FillSystRandomThrow` guards this exact
case at `:474`; the main path didn't.) The spline loop iterates `j < shifts.size()` while
the binned path deliberately iterates `GetNSplines()` because "params may be over-sized"
(`:229–231`) → OOB read of `spline_binnings[j]`. `GetSplineShift` also returns **−1** for
a negative spline bin (`src/PROsyst.cxx:667`), which was silently multiplied into the
event weight. Event-by-event mode is kept (maintainer decision) — just made safe.
**Status: FIXED** — out-of-range events are skipped, the loop bound is `GetNSplines()`,
and negative spline bins contribute a factor of 1.

### T1.9 — `static std::mt19937 rng{seed}`: seed ignored after first call, cross-thread race
`src/PROcess.cxx:442` (`FillSystRandomThrow`), `:502` (`FillSplineRandomThrow`).
A function-local `static` RNG is seeded only on the first-ever call — every later `seed`
argument is silently ignored — and the single generator is shared across threads (the
`TODO` comment acknowledges it). Note `PROsyst::spline2cov` *relied* on the static
advancing across its 500 calls, so the naive fix (fresh `rng{seed}` per call with a fixed
seed) would have produced 500 identical throws. **Status: FIXED** — both functions take
the seed per call and construct a local generator; all in-tree callers
(`spline2cov`, `allsplines2cov`, plotting paths) now pass a distinct per-iteration seed
(`seed + i`). Reproducibility: a given (seed, iteration) is deterministic; thread-safety:
no shared RNG state.

### T1.10 — `process_cafana_event`: unchecked `eventweight_map.find()`
`src/PROcreate.cxx:1192` → dereferenced at `:1205`, `:1235`, `:1249`. A
spline/covariance/covariance-to-spline systematic whose weight name is missing from the
event weight map (branch-name typo, absent friend tree) dereferences the end iterator →
segfault mid MC load with no diagnostic. **Status: FIXED** — explicit check with a fatal
log naming the offending systematic and the event entry.

### T1.11 — PROmesh: O(N²) global-min recomputation under the global lock
`src/PROmesh.cxx:326–331`. Every cell classification re-scans the entire `chi2_map` under
`state_mu` to find the global minimum — O(points) work × O(points) classifications, all
serialized on the one mutex every worker needs (the abandoned intent is visible in the
dead code at `:314–317`). This dominates prepass wall-time and defeats `nthreads`.
**Status: FIXED** — `std::atomic<float>` running minimum maintained with a CAS loop at
result-insertion time; classification reads it lock-free. (This is also PERF-F1.)

---

## Tier 2 — mode-specific correctness, races, robustness

| # | Finding | Location | Status |
|---|---------|----------|--------|
| T2.12 | `BinnedGrad` evaluates the chi² **value** event-by-event (`strat == BinnedChi2`) but the FD **gradient** binned (`strat != EventByEvent`): L-BFGS-B sees inconsistent f/∇f (line-search/Wolfe breakage), and the unbinned base call `cache.invalidate()`s the fill cache every iteration, which the binned FD closures then rebuild — worst of both worlds. | `PROchi.cxx:104` vs `:223/:236`; `PROCNP.cxx:94` vs `:183/:194`; `PROpoisson.cxx:82` vs `:146/:157` | FIXED\* — base condition now `strat != EventByEvent` in all three metrics: value and gradient are consistent and the cache is effective. `BinnedChi2` and `EventByEvent` behavior unchanged. If a mixed mode is ever wanted it should get a dedicated second cache. |
| T2.13 | `PROpoisson::getSingleChannelChi` indexes `m_channel_variable_bins` with a **global** channel index; the container is local-indexed (PROchi/PROCNP convert). Multi-mode/detector configs → wrong bin count or OOB. | `PROpoisson.cxx:258` | FIXED |
| T2.14 | `run_one_pe` builds fake data with `FillSpectra(..., strat)` — the enum lands on the `bool binned` parameter (works by accident) and `var_index` defaults to **0** — while the 2-arg `CollapseMatrix` uses **`i_prime`**: wrong-variable physics or dimension crash when `i_prime != 0`. Mirrors the legacy `fc_worker` (`PROfc.cxx:53`), fixed there too. | `PROAdaptiveFC.cxx:1682–1686`; `PROfc.cxx:53–60` | FIXED |
| T2.15 | `build_meta_mesh` sets `mm.max_levels = max_level_seen`, but the finest grid is always `initial × 2^opts.max_levels`: if no throw reached full depth, every cell `step` is wrong by `2^(opts.max_levels − max_level_seen)` (mis-sized cells, mislabeled levels → wrong PE-doubling budgets). | `PROAdaptiveFC.cxx:492–494` (+`PROmesh.cxx:79–81`) | FIXED — AMR depth carried in `AMRResult::max_levels` and used directly; consistency-checked against `finest_nx`. |
| T2.16 | PE bank is not reproducible across runs even with a fixed `--global-seed`: cells are claimed via `next_cell.fetch_add` and each PE's seed is drawn from the **claiming thread's** RNG, so seeds depend on OS scheduling. | `PROAdaptiveFC.cxx:1816, 1856` | FIXED — deterministic per-PE seed `MurmurHash3(global_seed, cell_idx, pe_index)`; independent of thread assignment. |
| T2.17 | `PROdata::operator/=` overwrites `spec` with the ratio **before** computing the relative error, so errors divide by the ratio instead of the original spectrum (const `operator/` is correct). Silent wrong uncertainties. | `PROdata.cxx:172–177` | FIXED |
| T2.18 | `PROdata::toTH1D` default `other_index = -1` → `m_channel_variable_bins[local][-1]` UB; both internal callers (`toROOT`, `plotSpectrum`) use the default. `PROspec::toTH1D` uses 0. | `inc/PROdata.h:102`; `PROdata.cxx:50–52` | FIXED — default 0. |
| T2.19 | CNP stat covariance `3/(1/n + 2/μ)` with μ from the spline-zeroed CV: `μ == 0` zeroes the diagonal → singular M → garbage/NaN from the Cholesky solve. Also the zero-data test uses `data.Spec()` in the base but `normdata` in the gradient rebuild (differs under `shape_only`). | `PROCNP.cxx:97–104`, `:167` | FIXED — μ floored at a small ε with a one-time warning; zero-test unified on `normdata`. The choice of μ (physics-only CV, i.e. stat term does not respond to nuisances) is **DOC** — flagged for physics review below. |
| T2.20 | `m_variable_bin_to_edges` built with `push_back` **inside the mode loop** and a never-cleared accumulator: with >1 mode the vector has `nvars×nmodes` cumulatively-growing entries; the consumer indexes it by variable index (wrong bin-edge labels, clamped OOB). | `PROconfig.cxx:1836–1854` | FIXED — one entry per variable. |
| T2.21 | NaN/inf universe weights guarded for splines (`w=1`) but **not** for covariance / covariance-to-spline universes — one bad weight silently NaNs the covariance (later zeroed by `toFiniteMatrix`, hiding the input problem). | `PROcreate.cxx:1235–1240, 1249–1253` | FIXED — same guard + a per-systematic warning counter so bad inputs are visible. |
| T2.22 | `ComputeSquareRootCovariance` LDLT path: LDLT "succeeds" on indefinite matrices, then `sqrt(negative D)` → silent NaNs; `svd_tol` is `[[maybe_unused]]`, the documented SVD filtering is commented out. Breaks PROMCMC proposal tuning on semi-definite sample covariances. | `PROtocall.cxx:117–157` | FIXED — negative D/singular values clamped to 0 with a warning; `svd_tol` wired up in the SVD branch. |
| T2.23 | `resolve_axis_index` returns a spline's position **within the spline list**, but callers index the full `[physics | spline]` parameter vector — a spline-valued `--xvar/--yvar` silently pins the wrong parameter. | `PROAdaptiveFC.cxx:86–97` (callers `:3011`) | FIXED — offset by `model.nparams`. |
| T2.24 | `best_multistart` indexing can run past the end for custom `--fit-options` (`n_swarm_particles > n_latin_points`, or `n_localfit − fudge > n_swarm_particles`). | `PROfitter.cxx:204, 216–218, 360–364` | FIXED — loops clamped to `.size()`. |
| T2.25 | XML bins parsing: only the all-attributes-missing case is guarded; a *partially* specified `<bins>` reaches `strtod(NULL, …)` (UB), and `nbins == 0` divides by zero. Same for both `bins2D` blocks. | `PROconfig.cxx:485` (+`:423–425`, `:445–447`, `:515`) | FIXED — per-attribute presence validation with clear errors; `nbins >= 1` enforced. |
| T2.26 | `DecomposeFractionalCovariance` writes `mutable last_decomp_*` from a `const` method; the `PROsyst` is shared by pointer across metric clones and throw helpers → data race if those paths ever run threaded. | `PROsyst.cxx:1020–1022, 1080–1081` | FIXED — cache guarded by a mutex (cold path; contention irrelevant). |
| T2.27 | Constraint-violation early return `return 1e10;` leaves `gradient` untouched — L-BFGS-B consumes a stale/garbage direction. | `PROchi.cxx:96`; `PROCNP.cxx:87`; `PROpoisson.cxx:76` | FIXED — gradient zeroed before returning. |

---

## Tier 3 — low severity / hygiene

| # | Finding | Location | Status |
|---|---------|----------|--------|
| T3.28 | Only PROCNP increments `call_count`; `getCallCount()` reports 0 for PROchi/PROpoisson. | `PROCNP.cxx:77` | FIXED — all three increment. |
| T3.29 | `cells_hit_n_pe_max` repurposed to hold the topped-up count but printed as `capped=` — misleading run summary. | `PROAdaptiveFC.cxx:1874`; `bin/PROfit.cxx:2883` | FIXED — both counters reported under honest names. |
| T3.30 | Malformed Boost.Format strings (`%2 vs. %3`, missing `%`) make the *error path itself* throw `bad_format_string`. | `PROdata.cxx:85`; `PROspec.cxx:218` | FIXED |
| T3.31 | `FindSubchannelIndexFromVariableGlobalBin` declared but never defined; `FindSubchannelIndexFromGlobalBin` defined but never declared/used — dead pair + latent linker trap. | `inc/PROtocall.h:51`; `PROtocall.cxx:36` | FIXED — removed. |
| T3.32 | `PROsyst::excluding` iterates `std::map` (alphabetical) while `subset` preserves caller order — spline/parameter-vector misalignment hazard for callers assuming insertion order. | `PROsyst.cxx:229` | FIXED — iterates `spline_names` order like `subset`. |
| T3.33 | `calcFreqSeedPoints` initializes a physics parameter with a **chi² value** (`minima.at(p).second` is the chi², not the parameter). Bad seed, not wrong result. | `PROfitter.cxx:546` | FIXED |
| T3.34 | FC bank stores `chi2_syst/chi2_osc/dchi2` as `float`; `dchi2` is a difference of large near-equal values, so quantile comparisons ride on the float precision of the chi² values near critical thresholds. | `inc/PROAdaptiveFC.h:239–241` | DOC — deliberately not changed: the fitter and every metric compute chi² in `float` end-to-end (`Eigen::VectorXf`, `float` returns), so double-storing the bank adds no real precision (subtraction of two nearby floats is already exact by the Sterbenz lemma) while breaking the artifact format. The real precision limit is the float chi² pipeline itself — flagged as a possible future migration if Δχ² resolution near critical values ever becomes a limiting systematic. |
| T3.35 | Dead "sequential Wilson stopping rule" machinery (`SequentialFCTest`, `CellState` with uninitialized `cell_idx`) advertised in the header, never used by `schedule_pes`. | `inc/PROAdaptiveFC.h:53–58, 169–205`; `.cxx:1731–1751` | FIXED — removed (doubling rule is the real policy; header comment updated). |
| T3.36 | Misc: `HexToROOTColor` uses uninitialized r/g/b on parse failure; `convertToXRootD` can `erase(npos)`; `find_equal_index` logs `.back()` on empty input; `ScaledCovariance` divides by `n − nparams` unguarded; `SystStruct::norm_value` uninitialized for non-norm modes; `PROlog` repeat-suppression map keyed on the fully formatted message (unbounded growth + double formatting). | various (see plan) | FIXED |

### Physics-review flags (documented, intentionally not changed)

- **PROchi drops zero-data bins and uses Neyman (data-count) statistical covariance**
  (`PROchi.cxx:53–64, 141–147`): an over-predicting MC in an empty data bin contributes
  *nothing* to the chi², and data-based variance biases fits low. PROCNP exists to address
  this; the behavior is presumably intentional for PROchi but should be a documented,
  deliberate choice.
- **CNP μ convention** (T2.19): the CNP statistical term uses the *physics-only* CV
  (splines zeroed), so the stat covariance does not respond to nuisance parameters. Fine
  if intentional; different from using the shifted prediction.
- **`variable_mc_stat_err` counts raw events** (`PROcreate.cxx:1174` `+= 1`), so the MC
  stat systematic is exact only for unweighted MC; for weighted MC it approximates the
  effective count. Working as designed, worth documenting.

---

## Performance — hot path (`FillSpectra` / `Metric()`)

Structure at review time: default gradient mode `GradientCentralFull`
(`PROmetric.h:244`, `PROfitter.h:117`) ⇒ each L-BFGS-B iteration costs `1 + 2·nparams`
full chi² evaluations; each full evaluation = FillSpectra + dense N×N
`diag(s)·F·diag(s)` + `CollapseMatrix` + O(m³) Cholesky. For ~50 splines that is ~100
dense covariance builds + factorizations per iteration, before which each `Fit()` already
does 750–5000 Latin-hypercube evaluations and 15–100 × 100–300 PSO evaluations.

| # | Item | Status |
|---|------|--------|
| H1 | `-DNDEBUG` absent from `CMAKE_CXX_FLAGS` (and the `set()` clobbers user flags): Eigen asserts on every coefficient access in "release" builds. | FIXED — `-O3 -DNDEBUG` appended (not overwritten); optional `PROFIT_MARCH_NATIVE` switch; `-ffast-math` deliberately **not** used (code depends on NaN/Inf semantics). |
| H2 | Default gradient mode rebuilds the dense covariance + Cholesky `2·nparams` times per gradient. The linearised modes (`GradientCentralLin`/`OneSidedLin`) freeze M at the base point and use the Gauss–Newton chain rule — exact at the minimum, dropping the dominant cost. | FIXED\* — **default flipped to `GradientCentralLin` in its own, clearly-labeled commit** so local tests can isolate or revert it (`git revert` that one commit restores `GradientCentralFull`). All modes remain selectable via `--fit-options grad=...`. |
| H3 | Base evaluation materializes the dense N×N `diag(s)·F·diag(s)` (twice, with the collapse) although only the collapsed m×m block is used. | FIXED — collapse computed as `Sᵀ F S` with `S = diag(s)·T` built sparse per call: no full-binning dense temporary. Bit-identical algebra. |
| H4 | Finite-difference loop copies the full parameter vector twice per parameter (O(nparams²) copying per gradient). | FIXED — perturb-in-place with restore. |
| H5 | FillSpectra waste: `final_error = sqrt(abs(spec))` computed every call and never read by any metric; the flat physics grid `var_arrs` (constant per fit) rebuilt on every physics-changed call; `unweighted_sum` (constant) recomputed per cross-binning spline per call; `PROhistStorage::operator()` returns an `Eigen::Ref` bound to `.transpose()`, forcing a full-matrix copy on every cross-binning access. | FIXED — error vector computed lazily only where consumed; physics grid cached in `PROmodel` (built once); per-(binning,var) `unweighted_sum` cached in the fill cache; hist storage returns a lightweight transpose-aware view. |
| H6 | Event-by-event strategy is O(NEvent·nsplines) per call with the fill cache force-invalidated. **Kept by maintainer decision** (rarely used but wanted). | DOC — cost documented here; correctness fixed in T1.8. Binned remains the recommended strategy for fits. |
| H7 | Per-call allocation trims: PROchi wrote a dense N×N stat covariance member every call (used only in NaN diagnostics — and the overwrite also dropped the ctor's `cwiseMax(1)` zero-bin guard that `getSingleChannelChi` relies on); PROCNP allocated a dense zero N×N every call for a diagonal; `maybe_convert_log` did a string-temporary + hash-map lookup per call; `Pull()` re-derives priors² per call; `get_probs` allocates a fresh matrix per call. | FIXED (first three) — the PROchi per-call overwrite is gone (also restoring the guarded member for `getSingleChannelChi`), CNP stat covariance is a vector added onto the collapsed matrix diagonal, and `maybe_convert_log` gained an allocation-free `const char*` overload (literals bind to it; no call-site changes). The `Pull()` temp and the `get_probs` return buffer were measured against the GEMV they sit next to and judged negligible — left as-is. |
| H8 | `PROlog` repeat-suppression keyed on the fully formatted message: unbounded map growth over long fits and double formatting per log call. | FIXED — keyed on the format template, single format pass. |

## Performance — adaptive FC / PROmesh

| # | Item | Status |
|---|------|--------|
| F1 | Global-min full-map rescan under the global mutex per classification (= T1.11). | FIXED |
| F2 | `enqueue_eval` dedups by linear scan of the whole work deque (O(queue²)); `enforce_balance_around` scans **all** cells per subdivision; the dense chi² reconstruction runs for every throw even when diagnostics are off. | FIXED — pending-key `unordered_set`; balance check via `corner_to_cells`; dense reconstruction gated on the diagnostics flag. |
| F3 | `generate_throws` retains every throw's full `AMRResult` (including `bestfit_map`, pure warm-start scratch) in RAM for the whole run — linear memory growth in `n_throws`. | FIXED — `bestfit_map` freed per throw; leaves/chi² retained only as needed by the meta-mesh tally and (if enabled) diagnostics. |
| F4 | Brazil throws rebuild the CV spectrum + `DecomposeFractionalCovariance` (a factorization) **inside** the per-throw loop although they are throw-invariant (`generate_throws` hoists the same computation); IDW surfaces recomputed per CL page and again on ROOT save. | FIXED — hoisted; IDW computed once per page set and reused. |
| F5 | `run_one_pe` heap-allocates a fresh metric for every PE (thousands per cell); the wilks prepass already demonstrates `thread_local` metric reuse. | FIXED — per-thread metric reuse in `schedule_pes` with per-PE `reset`/data swap. |

---

## Code-quality observations (not all addressed on this branch)

- `run_adaptive_fc` is a ~560-line five-mode dispatcher; the pad/frame/log-axis plotting
  boilerplate is copy-pasted ~6×; the two IDW builders are ~60 near-identical lines — all
  candidates for extraction. The file banner itself acknowledges wholesale duplication of
  `fc_worker` and `FillSurfaceAMR`; T1.5/T2.14 fixes reduce but do not remove it.
- Pervasive raw `new` for ROOT plot objects relying on canvas ownership; fine by ROOT
  idiom, but leaks accumulate over multi-page PDF production.
- Magic numbers: nuisance fit bounds `±3.0`, IDW exact-hit `1e-6`, `fits_by_level[8]`.
- `const_cast` in `save_bank/save_mesh/save_brazil_archive` — would be cleaner with
  Boost `split_free`.
- The `asimov` mode name no longer matches behavior (it classifies whatever dataset is
  passed, per `75391f7`) — rename or document.

## Verification guidance

- Build mirrors `.github/workflows/cmake-single-platform.yml` (ROOT 6.28+, Boost ≥ 1.71
  with serialization, HDF5, Eigen). Smoke test:
  `cd ci && PROfit -x ci_quick.xml -t quick_v0.1_FullSBN_nue_app plot`.
- Hot-path refactors (H3–H7) are algebra-preserving: compare chi² at fixed parameter
  points before/after (bit-level for H3/H4/H7). The gradient-default flip (H2) changes
  optimizer trajectories — compare *converged minima*, not paths, and revert its single
  commit to A/B test.
- FC determinism (T2.16): run `--mode init-bank` twice with the same `--global-seed` and
  diff the bank files — they should now be identical.
- A one-off `-fsanitize=address,undefined` build of the CI smoke test is recommended; it
  would have caught T1.2, T1.8, T2.18, T2.24, T2.25 directly.
