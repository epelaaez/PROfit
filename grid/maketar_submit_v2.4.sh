#!/bin/bash
#
# DEPRECATED: superseded by `PROfit -x cfg.xml -t TAG proletariat --script <worker.sh> ...`
# (see inc/PROletariat.h). Kept until the subcommand is validated on the grid.
#
# Usage:
#   ./mktar_and_submit.sh -n 500 path/to/grid_script.sh [other_file ...]
#
#   -n N   number of grid jobs        (default 2)   
#   -l T   expected lifetime          (default 2d, 3d is the ceiling before rejection)
#   -m MB  requested memory in MB     (default 4000)
#   -d MB  requested scratch disk     (default 10000)
#   -T     dry run: build the tarball, print the jobsub command, don't submit

set -euo pipefail

NJOBS=2
LIFETIME=2d
MEMORY=4000
DISK=10000
DRYRUN=0

PROFIT_BIN=${PROFIT_BIN:-/exp/sbnd/app/users/markross/PROfit_Dev/Dev25_Tutpush/Elephant_Vanishes/build/bin/PROfit}
SIMG=${SIMG:-/cvmfs/singularity.opensciencegrid.org/fermilab/fnal-wn-sl7:latest}

usage() { echo "usage: $0 [-n njobs] [-l lifetime] [-m mem_MB] [-d disk_MB] [-T] script.sh [files...]" >&2; exit 1; }

while getopts "n:l:m:d:T" opt; do
    case "$opt" in
        n) NJOBS=$OPTARG ;;
        l) LIFETIME=$OPTARG ;;
        m) MEMORY=$OPTARG ;;
        d) DISK=$OPTARG ;;
        T) DRYRUN=1 ;;
        *) usage ;;
    esac
done
shift $((OPTIND - 1))

[ $# -ge 1 ] || usage

SCRIPT_PATH=$1; shift
SCRIPT=$(basename "$SCRIPT_PATH")

[ -f "$SCRIPT_PATH" ] || { echo "no such script: $SCRIPT_PATH" >&2; exit 1; }
[ -x "$PROFIT_BIN"  ] || { echo "PROfit binary missing or not executable: $PROFIT_BIN" >&2; exit 1; }

# --- stage -------------------------------------------------------------------
# Fresh staging area every time, so nothing from a previous run leaks into the tar.
STAGE=$(mktemp -d "$PWD/.grid_stage.XXXXXX")
trap 'rm -rf "$STAGE"' EXIT

mkdir -p "$STAGE/grid_dir"
cp "$PROFIT_BIN"  "$STAGE/grid_dir/"
cp "$SCRIPT_PATH" "$STAGE/grid_dir/"
# cp "$(dirname "$PROFIT_BIN")"/*.pcm "$STAGE/grid_dir/"   # uncomment if you need the dict pcm

for f in "$@"; do
    [ -f "$f" ] || { echo "missing input file: $f" >&2; exit 1; }
    cp "$f" "$STAGE/grid_dir/"
done

chmod +x "$STAGE/grid_dir/PROfit" "$STAGE/grid_dir/$SCRIPT"

TARBALL=$PWD/grid_dir.tar
rm -f "$TARBALL"
tar cf "$TARBALL" -C "$STAGE" grid_dir

echo "--- tarball contents ($(du -h "$TARBALL" | cut -f1)) ---"
tar tvf "$TARBALL"
echo "-------------------------------------------------------"
echo "submitting $NJOBS job(s), lifetime $LIFETIME, ${MEMORY}MB, ${DISK}MB disk"




# --- submit ------------------------------------------------------------------
CMD=(
    jobsub_submit
      -G sbnd
      -N "$NJOBS"
      --role=Analysis
      --expected-lifetime="$LIFETIME"
      --memory="${MEMORY}MB"
      --disk="${DISK}MB"
      --lines '+FERMIHTC_AutoRelease=True'
      --lines '+FERMIHTC_GraceMemory=4000'
      --lines '+FERMIHTC_GraceLifetime=7200'
      --resource-provides="usage_model=DEDICATED,OPPORTUNISTIC,OFFSITE"
      -l "+SingularityImage=\\\"${SIMG}\\\""
      --append_condor_requirements='(TARGET.HAS_SINGULARITY=?=true)'
      --tar_file_name "dropbox://${TARBALL}"
      "file://${PWD}/${SCRIPT}"
)

if [ "$DRYRUN" -eq 1 ]; then
    printf '%q ' "${CMD[@]}"; echo
    echo "(dry run — not submitted; tarball left at $TARBALL)"
    trap - EXIT; rm -rf "$STAGE"
    exit 0
fi

"${CMD[@]}"
