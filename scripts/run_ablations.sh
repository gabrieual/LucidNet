#!/usr/bin/env bash
# Automated Ablation Runner for Variants A, B, C, D, E and Diagnostic E-ZeroWiener
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

ITERS="${1:-200000}"
SEED="${2:-42}"

CONFIGS=(
  "configs/ablations/A_baseline_proposta.yaml"
  "configs/ablations/B_frf_only.yaml"
  "configs/ablations/C_daab_mtcsa_no_wiener.yaml"
  "configs/ablations/D_twin_decoder_no_wiener.yaml"
  "configs/ablations/E_cwdnet_full.yaml"
)

echo "=================================================================="
echo " Running CWD-Net Ablation Suite (A -> E) | iters=${ITERS} | seed=${SEED}"
echo "=================================================================="

for cfg in "${CONFIGS[@]}"; do
  exp_name="$(basename "${cfg}" .yaml)"
  exp_dir="experiments/ablation_${exp_name}_seed${SEED}"
  echo ""
  echo ">>> [Train] Variant: ${exp_name} (${cfg})"
  python3 scripts/train.py \
    --config "${cfg}" \
    --exp-dir "${exp_dir}" \
    --opts "train.total_iters=${ITERS}" "experiment.seed=${SEED}"

  echo ">>> [Evaluate] Variant: ${exp_name}"
  python3 scripts/evaluate.py \
    --config "${cfg}" \
    --checkpoint "${exp_dir}/checkpoints/best_psnr.pt" \
    --output-json "${exp_dir}/eval_metrics.json"
done

# Run zero-shot Wiener diagnostic (E-ZeroWiener) on trained Variant E checkpoint
E_DIR="experiments/ablation_E_cwdnet_full_seed${SEED}"
if [[ -f "${E_DIR}/checkpoints/best_psnr.pt" ]]; then
  echo ""
  echo ">>> [Evaluate] Diagnostic Variant: E-ZeroWiener (beta_deconv = 0)"
  python3 scripts/evaluate.py \
    --config "configs/ablations/E_cwdnet_full.yaml" \
    --checkpoint "${E_DIR}/checkpoints/best_psnr.pt" \
    --zero-wiener \
    --output-json "${E_DIR}/eval_metrics_zero_wiener.json"
fi

echo ""
echo "All ablation experiments completed successfully."
