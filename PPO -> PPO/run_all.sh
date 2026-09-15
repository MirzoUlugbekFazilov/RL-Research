#!/usr/bin/env bash
# Study C: PPO on corrupted -> PPO fine-tuned on original, 2 corruptions x 30 seeds.
#
#   ./run_all.sh [JOBS]        # default 6 concurrent workers
#
# Two fine-tuning surfaces are run over the same pretrained checkpoints:
#
#   --study C        actor copy only: critic random, Adam moments fresh. This is
#                    exactly what Study B's GFlowNet can hand to PPO, so the
#                    contrast against Study B varies only which algorithm did
#                    the pretraining. Primary.
#   --study C_full   PPO.load: actor, critic and optimiser moments all carry
#                    over -- ordinary PPO fine-tuning. Secondary; its gap over
#                    C prices the critic the cross-algorithm arms cannot get.
#
# ppo_pretrain_corrupted and ppo_scratch_original are imported, not retrained
# (see PROVENANCE.md and verify_import.py), so run_study.py finds them cached
# and only the two fine-tuning arms actually train: 2 corruptions x 30 seeds x
# 2 surfaces = 120 runs of 300,000 environment steps.
#
# Seeds are sharded across worker processes; every seed writes into its own
# results/<arm>/seed_<n>/ directory, so shards never touch the same file. Each
# worker is pinned to one BLAS/OMP thread: the models are two 128-unit layers,
# intra-op parallelism buys nothing, and 6 workers x 8 threads would
# oversubscribe the 8 cores and run slower than serial.
#
# Runs are cached -- a seed whose final_eval.json already exists is skipped --
# so this script is safe to re-run after an interruption.
set -uo pipefail
cd "$(dirname "$0")"

JOBS="${1:-6}"
SEEDS=(42 123 456 789 1000 $(seq 2000 2024))
CORRUPTIONS=(large_multigoal action_gain_jitter)
STUDIES=(C C_full)
NSHARDS=$JOBS

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Seeds are dealt round-robin so no shard gets a contiguous block: run times
# vary by seed, and a contiguous split tends to leave one worker running long
# after the others have finished.
#
# Plain background jobs and a `wait` barrier, rather than `xargs -P -I{}`: BSD
# xargs caps the length of an -I replacement and refuses the whole pipeline once
# the seed list is this long. `wait -n` is avoided too -- macOS ships bash 3.2,
# which does not have it.
for study in "${STUDIES[@]}"; do
  for corr in "${CORRUPTIONS[@]}"; do
    mkdir -p "runs/$corr/logs"
    echo "=== study $study / $corr : ${#SEEDS[@]} seeds over $NSHARDS shards ==="
    for ((s = 0; s < NSHARDS; s++)); do
      shard=()
      for ((i = s; i < ${#SEEDS[@]}; i += NSHARDS)); do shard+=("${SEEDS[i]}"); done
      (
        python3 run_study.py --study "$study" --corruption "$corr" \
          --seeds "${shard[@]}" --no-study-json \
          > "runs/$corr/logs/${study}_shard_$s.log" 2>&1
        echo "  finished $study/$corr shard $s (exit $?)"
      ) &
    done
    wait
    echo "=== study $study / $corr : all shards finished ==="
  done
done

# Everything is cached now; this pass only writes results/study_<name>.json.
for study in "${STUDIES[@]}"; do
  for corr in "${CORRUPTIONS[@]}"; do
    echo "=== writing runs/$corr/results/study_$study.json ==="
    python3 run_study.py --study "$study" --corruption "$corr" \
      --seeds "${SEEDS[@]}" > "runs/$corr/logs/${study}_study_json.log" 2>&1
    tail -2 "runs/$corr/logs/${study}_study_json.log"
  done
done

echo "=== done; now run: ./analyze_all.sh ==="
