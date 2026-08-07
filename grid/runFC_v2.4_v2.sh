#!/bin/bash
# Worker-node script. Do NOT set -e: we want to control what happens after a
# PROfit failure ourselves (log, list the dir, exit non-zero so condor sees it).
set -uo pipefail

say()  { echo "[$(date +%H:%M:%S)] $*"; }
fail() { echo "[$(date +%H:%M:%S)] FATAL: $*" >&2; }

echo "======================================================================"
say "host $(hostname) | $(date) | cluster ${CLUSTER:-?} process ${PROCESS:-?}"
say "jobsub id  : ${JOBSUBJOBID:-unset}"
say "site       : ${GLIDEIN_Site:-unknown} / ${GLIDEIN_ResourceName:-unknown}"
say "os release : $(cat /etc/redhat-release 2>/dev/null || echo unknown)"
say "user       : $(id -un) ($(id -u))"
say "cwd        : $(pwd)"
echo "======================================================================"

# --- environment sanity before we do anything --------------------------------
for v in INPUT_TAR_DIR_LOCAL _CONDOR_SCRATCH_DIR; do
    if [ -z "${!v:-}" ]; then
        fail "\$$v is not set — is this actually running under jobsub?"
        exit 59
    fi
    say "$v = ${!v}"
done

say "scratch space available:"
df -h "$_CONDOR_SCRATCH_DIR" | sed 's/^/    /'

say "contents of input tar dir:"
ls -l "$INPUT_TAR_DIR_LOCAL"/grid_dir/ | sed 's/^/    /' || {
    fail "cannot list $INPUT_TAR_DIR_LOCAL/grid_dir — did the tarball unpack?"
    ls -lR "$INPUT_TAR_DIR_LOCAL" | head -50
    exit 59
}

# --- UPS setup ---------------------------------------------------------------
# UPS scripts poke at unset variables; relax -u across the setup block only.
say "sourcing larsoft setup..."
set +u
source /cvmfs/larsoft.opensciencegrid.org/setup_larsoft.sh
src_rc=$?
set -u
[ $src_rc -ne 0 ] && { fail "setup_larsoft.sh returned $src_rc"; exit 58; }

# setup <product> failing is a silent killer: it returns non-zero but nothing
# downstream notices until the binary can't find a library.
for prod in "root v6_28_12 -q e26:p3915:prof" \
            "cmake v3_27_4" \
            "hdf5 v1_12_2b -q e26:prof" \
            "boost v1_82_0 -q e26:prof" \
            "ifdhc v2_8_0 -q e26:p3915:prof"; do
    set +u
    # shellcheck disable=SC2086
    setup $prod
    rc=$?
    set -u
    if [ $rc -ne 0 ]; then
        fail "setup $prod returned $rc"
        exit 58
    fi
    say "  setup ok: $prod"
done

say "=== Setup's complete ==="
say "  ROOTSYS   = ${ROOTSYS:-unset}"
say "  root-config --version = $(root-config --version 2>/dev/null || echo 'NOT FOUND')"
say "  which root = $(command -v root || echo 'NOT ON PATH')"

# --- stage into scratch ------------------------------------------------------
say "copying payload into scratch..."
cp -r "$INPUT_TAR_DIR_LOCAL"/grid_dir/* "$_CONDOR_SCRATCH_DIR"/ || { fail "cp of payload failed"; exit 60; }
cd "$_CONDOR_SCRATCH_DIR" || { fail "cd to scratch failed"; exit 61; }

say "scratch dir now contains:"
ls -l | sed 's/^/    /'

# --- the binary --------------------------------------------------------------
if [ ! -f ./PROfit ]; then
    fail "./PROfit is not in the scratch dir — check PROFIT_BIN in the submit script"
    exit 64
fi

say "PROfit found: $(ls -l ./PROfit | awk '{print $1, $5" bytes", $NF}')"
say "  md5    : $(md5sum ./PROfit | cut -d' ' -f1)"
say "  type   : $(file -b ./PROfit 2>/dev/null || echo 'file(1) unavailable')"

chmod +x ./PROfit
if [ ! -x ./PROfit ]; then
    fail "./PROfit is still not executable after chmod (noexec mount on scratch?)"
    mount | grep -E "$(df --output=target ./ | tail -1)" | sed 's/^/    /'
    exit 65
fi
say "  executable bit: OK"

# The single most common grid failure: binary built against libs that aren't in
# the container. This turns a bare exit 127 into an actionable message.
say "checking shared library resolution..."
if command -v ldd >/dev/null 2>&1; then
    missing=$(ldd ./PROfit 2>&1 | grep -i "not found" || true)
    if [ -n "$missing" ]; then
        fail "PROfit has unresolved shared libraries:"
        echo "$missing" | sed 's/^/    /'
        say "full ldd output:"
        ldd ./PROfit | sed 's/^/    /'
        say "LD_LIBRARY_PATH = ${LD_LIBRARY_PATH:-unset}"
        exit 66
    fi
    say "  all $(ldd ./PROfit | wc -l) shared libs resolved"
else
    say "  ldd unavailable, skipping"
fi

# Smoke test: can it start at all? Cheap, and separates "won't launch" from
# "launched and then died on the physics".
say "smoke test: ./PROfit --help"
./PROfit --help >/dev/null 2>&1
smoke_rc=$?
if [ $smoke_rc -ne 0 ]; then
    fail "./PROfit --help exited $smoke_rc — binary won't launch"
    ./PROfit --help 2>&1 | head -20 | sed 's/^/    /'
    exit 67
fi
say "  smoke test passed"

# --- job parameters ----------------------------------------------------------
RNG=$((PROCESS + 1))
SEED=$((190000 + RNG))          # unique per process, and won't collide once RNG>9
XML=PROfit_Tutorial_Oct2025v1_SPINE_ICARUS_numu_dis.xml
NTHROWS=2                       # <-- bump this for a real bank

# Namespace by cluster so a resubmit doesn't silently clobber the last one.
# The username MUST be the job's mapped grid identity (jobsub sets GRID_USER):
# writing into anyone else's scratch tree gets a dCache 403 that gfal-copy
# misreports as "File exists" (error 17).
OUTDIR=/pnfs/sbnd/scratch/users/${GRID_USER:-$(whoami)}/PROfit_Grid/${CLUSTER:-manual}

say "job parameters:"
say "  PROCESS  = ${PROCESS}"
say "  RNG      = ${RNG}"
say "  SEED     = ${SEED}"
say "  XML      = ${XML}"
say "  NTHROWS  = ${NTHROWS}"
say "  OUTDIR   = ${OUTDIR}"

[ -f "$XML" ] || { fail "$XML not in the tarball"; ls -l; exit 62; }
say "XML found: $(ls -l "$XML" | awk '{print $5" bytes"}'), md5 $(md5sum "$XML" | cut -d' ' -f1)"


# --- locate the pre-built mesh and rename it to what init-bank expects --------
# PROfit looks for ${tag}_${output}_mesh.bin, i.e. GRID_fc_2_mesh.bin for PROCESS=1
MESH_EXPECTED="GRID_fc_${RNG}_mesh.bin"

shopt -s nullglob
mesh_candidates=( *mesh*.bin )
shopt -u nullglob

if [ ${#mesh_candidates[@]} -eq 0 ]; then
    fail "no *mesh*.bin in $(pwd) — was the mesh actually added to the tarball?"
    ls -l | sed 's/^/    /'
    exit 68
elif [ ${#mesh_candidates[@]} -gt 1 ]; then
    fail "expected exactly 1 mesh file, found ${#mesh_candidates[@]}: ${mesh_candidates[*]}"
    say "refusing to guess which one to use"
    exit 69
fi

MESH_FOUND=${mesh_candidates[0]}

if [ "$MESH_FOUND" = "$MESH_EXPECTED" ]; then
    say "mesh already correctly named: $MESH_EXPECTED"
else
    say "renaming mesh: $MESH_FOUND -> $MESH_EXPECTED"
    mv "$MESH_FOUND" "$MESH_EXPECTED" || { fail "mesh rename failed"; exit 70; }
fi

say "  mesh: $(stat -c%s "$MESH_EXPECTED") bytes, md5 $(md5sum "$MESH_EXPECTED" | cut -d' ' -f1)"

# --- run ---------------------------------------------------------------------
say "launching PROfit..."
echo "----------------------------------------------------------------------"
t0=$SECONDS

./PROfit -x "$XML" -t GRID -o "fc_${RNG}" -s "$SEED" -v 2 -n 1 --log "fc.${RNG}.meta" \
         fc-adaptive \
         --cl 90 \
         --mode init-bank \
         --n-pe-min "$NTHROWS" --n-pe-max "$NTHROWS" 
rc=$?

echo "----------------------------------------------------------------------"
say "=== PROfit done: exit $rc after $((SECONDS - t0))s ==="

if [ $rc -ne 0 ]; then
    fail "PROfit exited $rc"
    say "scratch dir contents at time of failure:"
    ls -l | sed 's/^/    /'
    say "tail of meta log:"
    tail -n 100 "fc.${RNG}.meta" 2>/dev/null | sed 's/^/    /'
    exit $rc
fi

# --- copy back ---------------------------------------------------------------
# Artifacts are prefixed ${tag}_${output}_ = GRID_fc_${RNG}_ (bank, mesh, PDFs).
# Trailing underscore keeps RNG=1 from matching a hypothetical fc_10.
shopt -s nullglob
outputs=( GRID_fc_${RNG}_* )
[ -f "fc.${RNG}.meta" ] && outputs+=( "fc.${RNG}.meta" )

say "scratch dir after run:"
ls -l | sed 's/^/    /'

if [ ${#outputs[@]} -eq 0 ]; then
    fail "PROfit succeeded but produced no output files"
    exit 63
fi

say "copying back ${#outputs[@]} file(s): ${outputs[*]}"

export IFDH_CP_MAXRETRIES=3
# Don't discard mkdir errors: a failed mkdir here is the first visible symptom
# of a wrong/unwritable OUTDIR, long before the copy's misleading error 17.
ifdh mkdir_p "$OUTDIR" || say "WARNING: ifdh mkdir_p $OUTDIR returned $? (may just already exist)"

# dCache refuses overwrites (gfal error 17 / HTTP 403), so a re-executed job
# (eviction, FERMIHTC auto-release) must clear its own leftovers first to make
# the copy idempotent. Re-runs produce identical files anyway (-n 1, fixed seed).
for f in "${outputs[@]}"; do
    ifdh rm "$OUTDIR/$f" >/dev/null 2>&1 || true
done

ifdh cp -D "${outputs[@]}" "$OUTDIR/"
crc=$?
if [ $crc -ne 0 ]; then
    fail "ifdh cp failed ($crc)"
    say "destination listing (does it exist? do you own it?):"
    ifdh ls "$OUTDIR" 2>&1 | sed 's/^/    /'
    say "GRID_USER = ${GRID_USER:-unset}, whoami = $(whoami)"
    say "X509_USER_PROXY = ${X509_USER_PROXY:-unset}"
    exit $crc
fi

say "=== done, all outputs in $OUTDIR ==="
exit 0
