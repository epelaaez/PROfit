#!/usr/bin/env bash
#
# Compare the outputs of two run_short_tests.sh runs, file by file, and write
# a markdown summary document.
#
# Usage:
#   tests/compare_tags.sh <TAG1> <TAG2> [RUNSBASE]
#
#   TAG1/TAG2  Tags previously run with run_short_tests.sh.
#   RUNSBASE   Directory holding the run dirs (default: <repo>/tests/runs).
#
# Comparison policy per file type:
#   *.bin *.txt *.xml   byte-level (cmp). These are expected to be identical
#                       for identical code (fixed seed, -n 1).
#   *.root              semantic dump via dump_root.C (ROOT files embed
#                       timestamps, so bytes always differ) then text diff.
#   *.pdf               size note only (PDFs embed creation timestamps and are
#                       not meaningfully comparable) — never fails the compare.
#   logs/, *.dump       skipped.
#   summary.txt         test PASS/FAIL statuses compared (timings ignored).
#
# Environment:
#   ROOTEXE   root executable (default: `root` in PATH, then
#             /usr/local/root/root/bin/root). Without it, .root files fall
#             back to a size comparison marked NODUMP.
#
# Exit code: 0 if no differences (PDF size deltas and skips do not count),
# otherwise the number of DIFFER/MISSING findings.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG1="${1:?usage: compare_tags.sh <TAG1> <TAG2> [RUNSBASE]}"
TAG2="${2:?usage: compare_tags.sh <TAG1> <TAG2> [RUNSBASE]}"
BASE="${3:-$REPO/tests/runs}"
DIR1="$BASE/$TAG1"
DIR2="$BASE/$TAG2"
DUMPER="$REPO/tests/dump_root.C"

[ -d "$DIR1" ] || { echo "ERROR: no run dir $DIR1"; exit 99; }
[ -d "$DIR2" ] || { echo "ERROR: no run dir $DIR2"; exit 99; }

ROOTEXE="${ROOTEXE:-$(command -v root || true)}"
[ -z "$ROOTEXE" ] && [ -x /usr/local/root/root/bin/root ] && ROOTEXE=/usr/local/root/root/bin/root

MD="$BASE/compare_${TAG1}_vs_${TAG2}.md"
N_IDENT=0; N_MATCH=0; N_DIFF=0; N_MISS=0; N_INFO=0; N_SKIP=0
ROWS=""

row() { ROWS+="| \`$1\` | $2 | $3 | $4 |"$'\n'; }

human_size() { numfmt --to=iec "$1" 2>/dev/null || echo "$1"; }

dump_root_file() {  # <file> <outdump>
    ( cd "$(dirname "$1")" && "$ROOTEXE" -l -b -q "$DUMPER(\"$(basename "$1")\")" ) 2>/dev/null \
        | grep -vE "^Processing |^$" > "$2"
}

# ---------------------------------------------------------------------------
shopt -s nullglob
mkdir -p "$DIR1/.dumps" "$DIR2/.dumps"

for f1 in "$DIR1"/*; do
    b1="$(basename "$f1")"
    [ -d "$f1" ] && continue                     # skip logs/, .dumps/
    case "$b1" in *.dump) continue;; esac
    b2="${b1/#$TAG1/$TAG2}"                      # translate leading tag prefix
    f2="$DIR2/$b2"

    if [ ! -f "$f2" ]; then
        row "$b1" "-" "**MISSING in $TAG2**" ""
        N_MISS=$((N_MISS+1))
        continue
    fi

    case "$b1" in
        summary.txt)
            # Compare test verdicts only (strip timings / ERROR-line counts).
            s1=$(awk '$1=="PASS"||$1=="FAIL"{print $1, $2}' "$f1")
            s2=$(awk '$1=="PASS"||$1=="FAIL"{print $1, $2}' "$f2")
            if [ "$s1" = "$s2" ]; then
                row "$b1" "statuses" "MATCH" "same PASS/FAIL verdicts"
                N_MATCH=$((N_MATCH+1))
            else
                d=$(diff <(echo "$s1") <(echo "$s2") | head -6 | tr '\n' ';')
                row "$b1" "statuses" "**DIFFER**" "$d"
                N_DIFF=$((N_DIFF+1))
            fi
            ;;
        *.pdf)
            sz1=$(stat -c%s "$f1"); sz2=$(stat -c%s "$f2")
            row "$b1" "pdf" "INFO" "sizes $(human_size "$sz1") vs $(human_size "$sz2") (PDFs not comparable)"
            N_INFO=$((N_INFO+1))
            ;;
        *.log)
            N_SKIP=$((N_SKIP+1))
            ;;
        *.root)
            if [ -z "$ROOTEXE" ]; then
                sz1=$(stat -c%s "$f1"); sz2=$(stat -c%s "$f2")
                row "$b1" "root" "NODUMP" "no root executable; sizes $sz1 vs $sz2"
                N_INFO=$((N_INFO+1))
            else
                d1="$DIR1/.dumps/$b1.dump"; d2="$DIR2/.dumps/$b2.dump"
                # Regenerate stale dumps (older than the file OR the dumper macro).
                [ -s "$d1" ] && [ "$d1" -nt "$f1" ] && [ "$d1" -nt "$DUMPER" ] || dump_root_file "$f1" "$d1"
                [ -s "$d2" ] && [ "$d2" -nt "$f2" ] && [ "$d2" -nt "$DUMPER" ] || dump_root_file "$f2" "$d2"
                # Object names inside the files embed the analysis tag (e.g. the
                # mcmc_chain TTree name); neutralize both tags before diffing.
                sed "s/$TAG1/@TAG@/g" "$d1" > "$d1.norm"
                sed "s/$TAG2/@TAG@/g" "$d2" > "$d2.norm"
                if diff -q "$d1.norm" "$d2.norm" > /dev/null; then
                    row "$b1" "root" "MATCH" "semantic dump identical ($(wc -l < "$d1") lines)"
                    N_MATCH=$((N_MATCH+1))
                else
                    nd=$(diff "$d1.norm" "$d2.norm" | grep -c '^[<>]')
                    first=$(diff "$d1.norm" "$d2.norm" | grep '^[<>]' | head -1 | cut -c1-90)
                    row "$b1" "root" "**DIFFER**" "$nd differing dump lines; first: \`$first\`"
                    N_DIFF=$((N_DIFF+1))
                fi
            fi
            ;;
        *)
            # Binary-level comparison for everything else (.bin, .txt, .xml, ...)
            if cmp -s "$f1" "$f2"; then
                row "$b1" "bytes" "IDENTICAL" ""
                N_IDENT=$((N_IDENT+1))
            else
                sz1=$(stat -c%s "$f1"); sz2=$(stat -c%s "$f2")
                row "$b1" "bytes" "**DIFFER**" "sizes $sz1 vs $sz2"
                N_DIFF=$((N_DIFF+1))
            fi
            ;;
    esac
done

# Files present only in TAG2.
for f2 in "$DIR2"/*; do
    b2="$(basename "$f2")"
    [ -d "$f2" ] && continue
    case "$b2" in *.dump|*.log) continue;; esac
    b1="${b2/#$TAG2/$TAG1}"
    if [ ! -f "$DIR1/$b1" ]; then
        row "$b2" "-" "**EXTRA in $TAG2**" ""
        N_MISS=$((N_MISS+1))
    fi
done

# ---------------------------------------------------------------------------
VERDICT="PASS — no differences"
[ $((N_DIFF+N_MISS)) -gt 0 ] && VERDICT="FAIL — $N_DIFF differing, $N_MISS missing/extra"

{
    echo "# PROfit test comparison: \`$TAG1\` vs \`$TAG2\`"
    echo
    echo "- Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "- Run dirs: \`$DIR1\` vs \`$DIR2\`"
    echo "- Verdict: **$VERDICT**"
    echo "- Counts: $N_IDENT byte-identical, $N_MATCH semantic match, $N_DIFF differ, $N_MISS missing/extra, $N_INFO info-only, $N_SKIP logs skipped"
    echo
    echo "| File ($TAG1 side) | Compared as | Status | Detail |"
    echo "|---|---|---|---|"
    printf '%s' "$ROWS"
    echo
    echo "Legend: IDENTICAL = byte-equal; MATCH = ROOT semantic dump equal (bytes"
    echo "differ only by embedded timestamps); INFO = not meaningfully comparable"
    echo "(PDFs); NODUMP = no root executable available for dumping."
} > "$MD"

echo "----------------------------------------------------------------------"
echo "$VERDICT"
echo "Summary written to $MD"
exit $((N_DIFF+N_MISS))
