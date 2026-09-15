#!/usr/bin/env bash
# Build every report for Study C, in dependency order.
#
#   ./analyze_all.sh
#
# analyze.py is per (corruption, study): C and C_full share one results tree, so
# each is named explicitly rather than inferred from results/study.json, which
# only ever holds whichever study finished last.
set -euo pipefail
cd "$(dirname "$0")"

CORRUPTIONS=(large_multigoal action_gain_jitter)

for corr in "${CORRUPTIONS[@]}"; do
  echo "=== analyze $corr (study C, actor copy only) ==="
  python3 analyze.py --corruption "$corr" --study C --out "METRICS_${corr}"
  echo "=== analyze $corr (study C_full, full model) ==="
  python3 analyze.py --corruption "$corr" --study C_full --out "METRICS_${corr}_full"
done

# combine_metrics.py is deliberately absent. It is Study B's report writer:
# its arm names, its negate_both reference column and most of its prose are
# hard-wired to GFlowNet -> PPO. Repointing it would have produced a report
# whose narrative no longer matched its numbers, so Study C's cross-corruption
# view lives in compare_studies.py, which recomputes everything from the
# per-seed files.
echo "=== cross-study comparison (A, B, C) ==="
python3 compare_studies.py

echo "done -> METRICS_<corruption>.md, COMPARISON.md, overleaf_study_c.tex"
