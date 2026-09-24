#!/usr/bin/env bash
#
# PROfit short test suite.
#
# Runs `process` and then one short instance of every major PROfit workflow
# (global fits in all three metrics, profile, surface incl. AMR, plot variants,
# FC, the full adaptive-FC pipeline, MCMC, benchmarks, and the PROjector
# two-stage pre-fit/projected fit) against a fixed fake ND+FD SBN config.
# Everything is seeded and single-threaded so two runs of the same code are
# bit-reproducible and two tags can be compared with compare_tags.sh.
#
# Usage:
#   tests/run_short_tests.sh <TAG> [XML]
#
#   TAG   Analysis tag; all outputs land in tests/runs/<TAG>/.
#   XML   Config to use. Default:
#         working_dir/Neutrino2026/fake_sbn_v2.xml
#
# Environment overrides:
#   PROFIT_BIN            PROfit executable   (default: <repo>/build/bin/PROfit)
#   PROFIT_TEST_MCDIR     Directory holding fake_sbn_mc_{ND,FD}.root referenced
#                         by the XML (default: directory containing the XML).
#                         The XML's hardcoded /exp/... path is rewritten to it.
#   PROFIT_TEST_OUTDIR    Where run directories go (default: <repo>/tests/runs)
#   PROFIT_TEST_TIMEOUT   Per-test timeout in seconds (default: 1800)
#
# Exit code: number of failed tests (0 = all passed).

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TAG="${1:?usage: run_short_tests.sh <TAG> [XML]}"
XML_IN="${2:-$REPO/working_dir/Neutrino2026/fake_sbn_v2.xml}"
BIN="${PROFIT_BIN:-$REPO/build/bin/PROfit}"
MCDIR="${PROFIT_TEST_MCDIR:-$(cd "$(dirname "$XML_IN")" && pwd)}"
OUTBASE="${PROFIT_TEST_OUTDIR:-$REPO/tests/runs}"
TIMEOUT="${PROFIT_TEST_TIMEOUT:-1800}"
RUNDIR="$OUTBASE/$TAG"

[ -x "$BIN" ]    || { echo "ERROR: PROfit binary not found/executable: $BIN"; exit 99; }
[ -f "$XML_IN" ] || { echo "ERROR: XML not found: $XML_IN"; exit 99; }

mkdir -p "$RUNDIR/logs"
cd "$RUNDIR"

# Localize the XML: the reference config points its MCFile entries at a fixed
# /exp/... path; rewrite that directory to wherever the fake MC actually lives.
# Both tags of a comparison must use the same MCDIR or the config hash differs.
sed "s|/exp/uboone/data/users/markross|$MCDIR|g" "$XML_IN" > local_test.xml

for f in "$MCDIR/fake_sbn_mc_ND.root" "$MCDIR/fake_sbn_mc_FD.root"; do
    [ -f "$f" ] || { echo "ERROR: fake MC file missing: $f (set PROFIT_TEST_MCDIR)"; exit 99; }
done

# Fixed, deterministic base arguments. -n 1 is REQUIRED for bit-reproducibility
# (thread scheduling changes AMR warm-start ordering); --preset fast keeps every
# individual fit sub-second (a single value sets both the global and scan
# presets; do NOT pass two values — CLI11's greedy vector parsing would eat the
# subcommand name).
COMMON=(-x local_test.xml -t "$TAG" -n 1 -v 2 --seed 405 --preset fast)

# Physics axes for the nueapp model of this config.
AXES=(--xvar sinsq2thme --yvar dmsq --xlo 1e-4 --xhi 1 --ylo 1e-2 --yhi 1e2)

PASS=0; FAIL=0
SUMMARY="$RUNDIR/summary.txt"
: > "$SUMMARY"

note() { echo "$*" | tee -a "$SUMMARY"; }

# run_test <name> <args...>  — expects exit 0.
run_test() {
    local name="$1"; shift
    local t0=$SECONDS
    if timeout "$TIMEOUT" "$BIN" "${COMMON[@]}" -o "$name" "$@" > "logs/$name.log" 2>&1; then
        local nerr; nerr=$(grep -c "ERROR" "logs/$name.log" || true)
        note "PASS  $name  ($((SECONDS-t0))s, ${nerr} ERROR lines)"
        PASS=$((PASS+1))
    else
        local rc=$?
        note "FAIL  $name  ($((SECONDS-t0))s, exit $rc) -- see logs/$name.log"
        tail -n 5 "logs/$name.log" | sed 's/^/        /'
        FAIL=$((FAIL+1))
    fi
}

# expect_fail <name> <args...> — passes only if PROfit exits NONzero (validation
# paths must refuse bad input loudly, not limp on).
expect_fail() {
    local name="$1"; shift
    if timeout "$TIMEOUT" "$BIN" "${COMMON[@]}" -o "$name" "$@" > "logs/$name.log" 2>&1; then
        note "FAIL  $name  (expected a refusal but PROfit exited 0)"
        FAIL=$((FAIL+1))
    else
        note "PASS  $name  (correctly refused)"
        PASS=$((PASS+1))
    fi
}

note "PROfit short test suite"
note "  tag: $TAG"
note "  xml: $XML_IN"
note "  bin: $BIN"
note "  git: $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown) $(git -C "$REPO" diff --quiet 2>/dev/null || echo '(dirty)')"
note "  date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
note "----------------------------------------------------------------------"

# --- 0. MC processing (creates <TAG>_prop.bin / <TAG>_syst.bin) --------------
run_test t00process process
[ -f "${TAG}_prop.bin" ] || { note "FATAL t00process produced no ${TAG}_prop.bin; aborting."; exit 98; }

# --- 1. Global fits across metrics and data options --------------------------
run_test t01global        --use-fake-data global
run_test t02globalcnp     --use-fake-data -c PROCNP global
run_test t03globalpoisson --use-fake-data -c Poisson global
run_test t03bglobalpearson --use-fake-data -c pearson global
run_test t04statonly      --use-fake-data --statonly global
run_test t05inject        --use-fake-data -i dmsq 1 sinsq2thme 0.01 global
run_test t06pseudoexp     --use-fake-data --pseudo-experiment global

# --- 2. Profile (legacy 18-point scan, and the PRObe adaptive scan) -----------
run_test t07profile       --use-fake-data profile
run_test t07probe         --use-fake-data profile --probe

# --- 3. Surfaces --------------------------------------------------------------
run_test t08surface       --use-fake-data surface -g 4 "${AXES[@]}"
run_test t09surfaceamr    --use-fake-data surface -g 4 "${AXES[@]}" --surface-amr --amr-initial 4 --amr-levels 1

# --- 4. Plotting variants -----------------------------------------------------
# t10 opts into the covariance plots (off by default since --with-covar);
# t11/t12 run the default (no Covar.pdf / ROOT Covariance dir).
run_test t10plot          --use-fake-data plot --with-splines --with-covar
run_test t11plotwidth     --use-fake-data --scale-by-width plot
run_test t12plotbkgsub    --use-fake-data plot --bkg-subtract background

# --- 4b. Shape-only (prediction rescaled onto the data per channel; every
# systematic shape-projected). Covers all three chi2 families that differ in
# their statistical term, the scan warm-start path, FC (flag must reach the
# per-universe metrics), and the width+area plot combination.
run_test t12sglobal       --use-fake-data --shapeonly global
run_test t12sglobalcnp    --use-fake-data --shapeonly -c CNP global
run_test t12sglobalpois   --use-fake-data --shapeonly -c poisson global
run_test t12sprofile      --use-fake-data --shapeonly profile
run_test t12sfc           --use-fake-data --shapeonly fc -u 2
run_test t12splotwidth    --use-fake-data --shapeonly --scale-by-width plot

# --- 5. Feldman-Cousins -------------------------------------------------------
run_test t13fc            --use-fake-data fc -u 2

# --- 6. Adaptive FC pipeline (stages share -o so artifacts chain together) ----
AFC=(fc-adaptive --throws 2 --prepass-amr-initial 4 4 --prepass-amr-levels 1
     --p-thresh 0.33 --baseline-level 1 "${AXES[@]}")
run_test t14afcmesh   --use-fake-data "${AFC[@]}" --mode build-mesh
run_test t14afcbank   --use-fake-data "${AFC[@]}" --mode init-bank --n-pe-min 1 --n-pe-max 2
run_test t14afcprint  --use-fake-data "${AFC[@]}" --mode print-bank
run_test t14afcasimov --use-fake-data "${AFC[@]}" --mode asimov
run_test t14afcbrazil --use-fake-data "${AFC[@]}" --mode brazil --n-brazil-throws 2

# --- 7. MCMC + benchmark smoke ------------------------------------------------
run_test t15mcmc          --use-fake-data mcmc --nchains 1
run_test t16scaletest     --use-fake-data scale-test -N 50 --tests fillspectra,metric

# --- 8. PROjector two-stage pre-fit / projected fit ---------------------------
run_test t17pjprefit      --use-fake-data --projector-prefit "_ND_" global
CONSTRAINT="${TAG}_t17pjprefit_PROjector_constraint.bin"
if [ -f "$CONSTRAINT" ]; then
    run_test t18pjglobal  --use-fake-data --projector "$CONSTRAINT" global
    run_test t19pjfc      --use-fake-data --projector "$CONSTRAINT" fc -u 2
else
    note "FAIL  t18pjglobal (missing $CONSTRAINT)"; FAIL=$((FAIL+1))
    note "FAIL  t19pjfc     (missing $CONSTRAINT)"; FAIL=$((FAIL+1))
fi
# Partial-channel pattern (fullosc is one subchannel of the nue channel) and a
# match-everything pattern must both be refused.
expect_fail t20pjpartial  --use-fake-data --projector-prefit "fullosc" global
expect_fail t21pjall      --use-fake-data --projector-prefit "nu_" global

# --- 9. apply_to_subchannel (per-subchannel systematic scoping) ---------------
# DetSys1 (spline) restricted to ND, RPA_CCQE (covariance) restricted to FD.
# process must not require (or read) their weight branches in non-matching
# files; outside the match splines are flat at 1 and covariance blocks exactly
# zero — asserted numerically by check_applyto.C on the t23 plot output.
# Also: a flat systematic restricted to FD and a norm_to_covariance restricted to
# the ND numu channel (both matrix-only types, scoped by the post-build stage
# PROsyst::ApplySubchannelScopes rather than at fill time).
sed -e 's|plotname="DetSys1" tag="det"|plotname="DetSys1" tag="det" apply_to_subchannel="_ND_"|' \
    -e 's|plotname="RPA_CCQE" tag="QE-MEC"|plotname="RPA_CCQE" tag="QE-MEC" apply_to_subchannel="_FD_"|' \
    local_test.xml \
  | awk '{print} /FiducialVol_FD/ && !d {print "    <allowlist type=\"flat\" plotname=\"FlatFD\" tag=\"other\" apply_to_subchannel=\"_FD_\">nu_:0.05</allowlist>"; print "    <allowlist type=\"norm_to_covariance\" plotname=\"NormCovNDnumu\" tag=\"other\" apply_to_subchannel=\"_ND_numu\">nu_:0.03</allowlist>"; d=1}' \
  > local_applyto.xml
SAVED_COMMON=("${COMMON[@]}")
COMMON=(-x local_applyto.xml -t "${TAG}apt" -n 1 -v 2 --seed 405 --preset fast)
run_test t22aptprocess process
# --with-covar: t26aptzero asserts on the Covariance dir in this ROOT file.
run_test t23aptplot   --use-fake-data plot --with-splines --with-covar
run_test t24aptglobal --use-fake-data global
# A wildcard matching no subchannel fullname must be refused loudly.
sed 's|apply_to_subchannel="_ND_"|apply_to_subchannel="_TYPO_"|' local_applyto.xml > local_applyto_typo.xml
COMMON=(-x local_applyto_typo.xml -t "${TAG}apttypo" -n 1 -v 2 --seed 405 --preset fast)
expect_fail t25apttypo    process
COMMON=("${SAVED_COMMON[@]}")
# Numeric assertion: non-matching covariance blocks exactly zero, non-matching
# splines exactly flat (needs root; skipped silently if unavailable).
ROOTEXE="${ROOTEXE:-$(command -v root || true)}"
[ -z "$ROOTEXE" ] && [ -x /usr/local/root/root/bin/root ] && ROOTEXE=/usr/local/root/root/bin/root
if [ -n "$ROOTEXE" ]; then
    # ND numu block = bins [64,166): nue 4x16 then numu 3x34 in the ND half.
    if "$ROOTEXE" -l -b -q "$REPO/tests/check_applyto.C(\"${TAG}apt_t23aptplot_PROplot.root\",\"RPA_CCQE;cov;FD,DetSys1;spline;ND,nu_:0.05;cov;FD,nu_:0.03;cov;64-166\")" > logs/t26aptzero.log 2>&1; then
        note "PASS  t26aptzero  (non-matching cov blocks zero, splines flat)"
        PASS=$((PASS+1))
    else
        note "FAIL  t26aptzero -- see logs/t26aptzero.log"
        FAIL=$((FAIL+1))
    fi
else
    note "SKIP  t26aptzero  (no root executable for the numeric assertion)"
fi

# --- 9b. apply_to_subchannel: every type, DetVar, and branch order --------------
# (a) EVERY systematic (splines, covariances, mcstat, norm, flat, norm_to_covariance)
#     restricted to _FD_: the TOTAL fractional covariance must have an exactly-zero
#     ND block. mcstat is not plotted on its own, so this is what covers it.
sed 's|\(<allowlist [^>]*\)>|\1 apply_to_subchannel="_FD_">|' local_test.xml \
  | awk '{print} /FiducialVol_FD/ && !d {print "    <allowlist type=\"flat\" plotname=\"FlatFD\" tag=\"other\" apply_to_subchannel=\"_FD_\">nu_:0.05</allowlist>"; print "    <allowlist type=\"norm_to_covariance\" plotname=\"NormCovFD\" tag=\"other\" apply_to_subchannel=\"_FD_\">nu_:0.03</allowlist>"; d=1}' \
  > local_applyto_fd.xml
COMMON=(-x local_applyto_fd.xml -t "${TAG}aptfd" -n 1 -v 2 --seed 405 --preset fast)
run_test t26bfdprocess process
run_test t26cfdplot    --use-fake-data plot --with-splines --with-covar
if [ -n "$ROOTEXE" ]; then
    if "$ROOTEXE" -l -b -q "$REPO/tests/check_applyto.C(\"${TAG}aptfd_t26cfdplot_PROplot.root\",\"total_frac_cov;total;FD,nu_:0.05;cov;FD,nu_:0.03;cov;FD,Flux1;spline;FD\")" > logs/t26dfdzero.log 2>&1; then
        note "PASS  t26dfdzero  (all-FD scope: total covariance ND block exactly zero)"
        PASS=$((PASS+1))
    else
        note "FAIL  t26dfdzero -- see logs/t26dfdzero.log"
        FAIL=$((FAIL+1))
    fi
fi
# (b) DetVar systematic: the same ND file as CV and as a variation at half the POT
#     (ratio 2 everywhere the section covers), restricted to the ND nue channel via
#     apply_to_subchannel: spline varying only in bins [0,64), flat everywhere else.
awk -v mc="$MCDIR" '
  /<variation_list>/ && !dv {
    print "<DetVarFiles>";
    print "  <DetVarSection treename=\"events/selected\" scale=\"1.0\">";
    print "    <cv filename=\"" mc "/fake_sbn_mc_ND.root\" pot=\"1e+21\"/>";
    print "    <variation name=\"DetVarTest\" filename=\"" mc "/fake_sbn_mc_ND.root\" pot=\"5e+20\"/>";
    print "    <subchannel>nu_ND_nue_intrinsic</subchannel>";
    print "    <subchannel>nu_ND_nue_background</subchannel>";
    print "    <subchannel>nu_ND_nue_fullosc</subchannel>";
    print "    <subchannel>nu_ND_numu_signal</subchannel>";
    print "    <subchannel>nu_ND_numu_background</subchannel>";
    print "  </DetVarSection>";
    print "</DetVarFiles>";
    dv=1 }
  {print}
  /FiducialVol_FD/ && !sy { print "    <allowlist type=\"spline\" name=\"DetVarTest\" plotname=\"DetVarTest\" tag=\"det\" apply_to_subchannel=\"_ND_nue_\"/>"; sy=1 }
' local_applyto.xml > local_applyto_detvar.xml
COMMON=(-x local_applyto_detvar.xml -t "${TAG}aptdv" -n 1 -v 2 --seed 405 --preset fast)
run_test t26edvprocess process
run_test t26fdvplot    --use-fake-data plot --with-splines --with-covar
if [ -n "$ROOTEXE" ]; then
    if "$ROOTEXE" -l -b -q "$REPO/tests/check_applyto.C(\"${TAG}aptdv_t26fdvplot_PROplot.root\",\"DetVarTest;spline;0-64,RPA_CCQE;cov;FD\")" > logs/t26gdvzero.log 2>&1; then
        note "PASS  t26gdvzero  (DetVar systematic scoped to the ND nue channel)"
        PASS=$((PASS+1))
    else
        note "FAIL  t26gdvzero -- see logs/t26gdvzero.log"
        FAIL=$((FAIL+1))
    fi
fi
# (c) Branch order: an incl_systematics="false" branch listed FIRST in a file whose
#     later branches carry systematics (zero-weight, so the physics is unchanged).
#     The friend-tree weight binding used to be decided by that first branch and
#     every later branch then died at fill time; the fit must match t24aptglobal.
awk '/<friend treename="events\/variationTree" \/>/ && !d { print;
       print "    <branch associated_subchannel=\"nu_ND_nue_cosmic\" incl_systematics=\"false\" model_rule=\"0\" additional_weight=\"0.0*mcweight\">";
       print "        <variable>reco_visible_energy</variable>";
       print "        <variable>true_baseline/(1000*true_neutrino_energy)</variable>";
       print "        <variable>true_neutrino_energy</variable>";
       print "        <variable>random_value</variable>";
       print "    </branch>"; d=1; next } {print}' local_applyto.xml > local_applyto_order.xml
COMMON=(-x local_applyto_order.xml -t "${TAG}aptord" -n 1 -v 2 --seed 405 --preset fast)
run_test t26hordprocess process
run_test t26iordglobal  --use-fake-data global
if cmp -s "${TAG}aptord_t26iordglobal_global_fit.txt" "${TAG}apt_t24aptglobal_global_fit.txt"; then
    note "PASS  t26jordsame  (cosmic-first branch order gives the identical global fit)"
    PASS=$((PASS+1))
else
    note "FAIL  t26jordsame  (global fit differs from t24aptglobal)"
    FAIL=$((FAIL+1))
fi
COMMON=("${SAVED_COMMON[@]}")

# --- 10. regex wildcards (patterns are unanchored ECMAScript regexes) ---------
# Plain substrings keep their old meaning (every test above covers that); here
# a genuine regex alternation must be accepted end-to-end. NAME:percent splits
# on the LAST colon, so regex constructs containing ':' survive too.
sed 's#>nu_ND_numu:0.02<#>nu_(ND|FD)_numu:0.02<#' local_test.xml > local_regex.xml
SAVED_COMMON=("${COMMON[@]}")
COMMON=(-x local_regex.xml -t "${TAG}rgx" -n 1 -v 2 --seed 405 --preset fast)
run_test t27regexprocess process
run_test t28regexglobal --use-fake-data global
# An invalid regex must be refused loudly (CompilePattern fatal)...
sed 's#>nu_ND:0.01<#>*bad:0.01<#' local_test.xml > local_regex_bad.xml
COMMON=(-x local_regex_bad.xml -t "${TAG}rgxbad" -n 1 -v 2 --seed 405 --preset fast)
expect_fail t29badregex   process
# ...and so must a valid regex that matches no subchannel (zero-match fatal).
sed 's#>nu_ND:0.01<#>^nomatch$:0.01<#' local_test.xml > local_regex_none.xml
COMMON=(-x local_regex_none.xml -t "${TAG}rgxnone" -n 1 -v 2 --seed 405 --preset fast)
expect_fail t30nomatch    process
COMMON=("${SAVED_COMMON[@]}")

# --- 11. LBL 3nu models (matter + vacuum) -------------------------------------
# Matter needs per-event L and E <parameter>s, so its XML replaces the L/E
# variable with a true-baseline one (hash changes -> own tag + process). The
# L/E and E binnings are cut down hard: the model grid is n_L x n_E GLOBAL bins
# x 10 components x every variable's reco bins, which OOMs at the base 200x20.
# Vacuum's single signed L/E swaps only the model block (not hashed) and reuses
# the t00 caches. Binaries predating the vacuum tag fail t31c by design.
sed -e 's|<bins unit="True L/E \[km/GeV\]" min="0" max="2.5" nbins="200" plot="false"/>|<bins unit="True Baseline [km]" min="0" max="1" nbins="4" plot="false"/>|' \
    -e 's|<bins unit="True Neutrino Energy \[GeV\]" min="0" max="3" nbins="20" />|<bins unit="True Neutrino Energy [GeV]" min="0" max="3" nbins="5" />|' \
    -e 's|<variable>true_baseline/(1000\*true_neutrino_energy)</variable>|<variable>true_baseline/1000</variable>|' \
    -e 's|<model tag="nueapp">|<model tag="LBL_3nu-matter_angles">|' \
    -e 's|<parameter name="L/E" variable_index="1"/>|<parameter name="L" variable_index="1"/><parameter name="E" variable_index="2"/>|' \
    local_test.xml > local_lbl_matter.xml
sed 's|tag="LBL_3nu-matter_angles"|tag="LBL_3nu-matter_angles" density="3" electron_fraction="0.5" n_newton="0"|' local_lbl_matter.xml > local_lbl_explicit.xml
sed 's|<model tag="nueapp">|<model tag="LBL_3nu-vacuum_angles">|' local_test.xml > local_lbl_vacuum.xml
sed -e 's|<model tag="nueapp">|<model tag="LBL_3nu-matter_angles" baseline="1300">|' \
    -e 's|<parameter name="L/E" variable_index="1"/>|<parameter name="E" variable_index="2"/>|' \
    local_test.xml > local_lbl_eonly.xml
sed 's|<model tag="nueapp">|<model tag="nueapp" density="3">|'    local_test.xml > local_lbl_sblopt.xml
COMMON=(-x local_lbl_matter.xml -t "${TAG}lbl" -n 1 -v 2 --seed 405 --preset fast)
run_test t31lblprocess process
run_test t31almatter   --use-fake-data global
COMMON=(-x local_lbl_explicit.xml -t "${TAG}lbl" -n 1 -v 2 --seed 405 --preset fast)
run_test t31blexplicit --use-fake-data global
# Explicit legacy-default attributes must be bitwise identical to no attributes.
if cmp -s "${TAG}lbl_t31almatter_global_fit.txt" "${TAG}lbl_t31blexplicit_global_fit.txt"; then
    note "PASS  t31dlbldefault  (explicit default attributes bitwise-identical)"
    PASS=$((PASS+1))
else
    note "FAIL  t31dlbldefault  (explicit LBL defaults changed the fit)"
    FAIL=$((FAIL+1))
fi
COMMON=(-x local_lbl_vacuum.xml -t "$TAG" -n 1 -v 2 --seed 405 --preset fast)
run_test t31clvacuum   --use-fake-data global
# E-only + baseline= (fixed-baseline mode, 1D grid; also reuses the t00 caches).
COMMON=(-x local_lbl_eonly.xml -t "$TAG" -n 1 -v 2 --seed 405 --preset fast)
run_test t31eleonly    --use-fake-data global
# Model options on a non-LBL tag must be refused loudly.
COMMON=(-x local_lbl_sblopt.xml -t "$TAG" -n 1 -v 2 --seed 405 --preset fast)
expect_fail t32lbadopt  --use-fake-data global
COMMON=("${SAVED_COMMON[@]}")

note "----------------------------------------------------------------------"
note "RESULT: $PASS passed, $FAIL failed  (outputs in $RUNDIR)"
exit "$FAIL"
