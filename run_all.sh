#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Full experiment suite for the XM-MADRL paper.
#
#   bash run_all.sh              # full: 5 seeds, all methods + ablations
#   QUICK=1 bash run_all.sh      # fast sanity pass (tiny steps, 1 seed)
#
# Every run checkpoints to results/, so the script is safe to stop and re-run:
# completed runs are skipped automatically.
# ---------------------------------------------------------------------------
set -e

SEEDS=(11 22 33 44 55)
STEPS=300000
BASELINES=(PPO A2C DDPG MADDPG)
ABLATIONS=(XM-noMAML XM-noGNN XM-noTrans XM-noXAI)
RESULTS=results

if [ "${QUICK:-0}" = "1" ]; then
  echo ">>> QUICK MODE: 1 seed, tiny budget (sanity check only)"
  SEEDS=(11); STEPS=4000
fi

mkdir -p "$RESULTS"

run () {   # run <method> <seed>
  local method=$1 seed=$2
  local tag="${RESULTS}/${method}_seed${seed}_eval.json"
  if [ -f "$tag" ]; then
    echo "skip  ${method} seed ${seed} (already done)"; return
  fi
  echo ">>> ${method} seed ${seed}"
  python train.py --method "$method" --seed "$seed" --steps "$STEPS" --results "$RESULTS"
}

# 1) proposed method -------------------------------------------------------
for s in "${SEEDS[@]}"; do run XM-MADRL "$s"; done
# 2) baselines -------------------------------------------------------------
for m in "${BASELINES[@]}"; do for s in "${SEEDS[@]}"; do run "$m" "$s"; done; done
# 3) ablations -------------------------------------------------------------
for m in "${ABLATIONS[@]}"; do for s in "${SEEDS[@]}"; do run "$m" "$s"; done; done

# 4) few-shot signal-classification MAML experiment ------------------------
for s in "${SEEDS[@]}"; do
  [ -f "${RESULTS}/signal_maml_seed${s}.json" ] || python signal_maml.py --seed "$s" --results "$RESULTS"
done

# 5) statistics, SHAP explainability, figures/tables -----------------------
python stats.py --results "$RESULTS" --baseline PPO
[ -f "${RESULTS}/XM-MADRL_seed${SEEDS[0]}.pt" ] && \
  python run_shap.py --weights "${RESULTS}/XM-MADRL_seed${SEEDS[0]}.pt" --seed "${SEEDS[0]}" --results "$RESULTS"
python make_figures.py --results "$RESULTS" --out figures

echo ">>> DONE. See results/ and figures/."
