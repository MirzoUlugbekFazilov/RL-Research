#!/usr/bin/env bash
# Run both corruption studies: 3 arms x 30 seeds each.
#
#   ./run_all.sh [JOBS]        # default 6 concurrent workers
#
# Seeds are sharded across worker processes.  One shard owns a whole seed
# (pretrain -> finetune -> scratch) because the fine-tuning arm needs that
# seed's pretrain checkpoint, and every seed writes into its own
# results/<arm>/seed_<n>/ directory, so shards never touch the same file.
#
# Each worker is pinned to one BLAS/OMP thread: the models are tiny (two 128-unit
# layers) so intra-op parallelism buys nothing and 6 workers each spawning 8
# threads would oversubscribe the 8 cores and run slower than serial.
#
# Runs are cached -- a seed whose final_eval.json already exists is skipped --
# so this script is safe to re-run after an interruption, and the final pass
# below re-runs with the full seed list purely to write results/study.json.
set -uo pipefail
cd "$(dirname "$0")"

JOBS="${1:-6}"
SEEDS=(42 123 456 789 1000 $(seq 2000 2024))
CORRUPTIONS=(large_multigoal action_gain_jitter)
NSHARDS=$JOBS

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Seeds are dealt round-robin so no shard gets a contiguous block: run times
# vary by seed, and a contiguous split tends to leave one worker running long
# after the others have finished.
#
# Plain background jobs and a `wait` barrier per corruption, rather than
# `xargs -P -I{}`: BSD xargs caps the length of an -I replacement and refuses
# the whole pipeline ("command line cannot be assembled, too long") once the
# seed list is this long.  `wait -n` is avoided too -- macOS ships bash 3.2,
# which does not have it.  With one shard per worker the barrier costs only the
# tail of the slowest shard.
for corr in "${CORRUPTIONS[@]}"; do
  mkdir -p "runs/$corr/logs"
  echo "=== $corr : ${#SEEDS[@]} seeds over $NSHARDS shards, $JOBS concurrent ==="
  for ((s = 0; s < NSHARDS; s++)); do
    shard=()
    for ((i = s; i < ${#SEEDS[@]}; i += NSHARDS)); do shard+=("${SEEDS[i]}"); done
    (
      python3 run_study.py --study B --corruption "$corr" --seeds "${shard[@]}" \
        --no-study-json > "runs/$corr/logs/shard_$s.log" 2>&1
      echo "  finished $corr shard $s (exit $?)"
    ) &
  done
  wait
  echo "=== $corr : all shards finished ==="
done

# Everything is cached now; this pass only writes results/study.json.
for corr in "${CORRUPTIONS[@]}"; do
  echo "=== writing runs/$corr/results/study.json ==="
  python3 run_study.py --study B --corruption "$corr" --seeds "${SEEDS[@]}" \
    > "runs/$corr/logs/study_json.log" 2>&1
  tail -2 "runs/$corr/logs/study_json.log"
done

echo "=== done; now run: python3 analyze.py --corruption <name> ==="
