#!/usr/bin/env bash
# Evaluate a trained antibody-codesign checkpoint on the test set
# (antibody_codesign_test_set -> PROTENIX_ROOT_DIR/test_mmcif via test_indices.csv).
#
# Runs runner/train.py in --eval_only mode: it builds the model, loads the given
# checkpoint, runs one pass over the test dataloader, and prints/logs metrics
# (structure lDDT + codesign sequence accuracy/loss). No training, no gradients.
#
# Usage:
#   bash antibody_codesign_eval_stage_1.sh                 # eval the EMA checkpoint
#   CKPT=.../checkpoints/stage_1.pt bash antibody_codesign_eval_stage_1.sh   # eval raw weights
#   GPU=0 bash antibody_codesign_eval_stage_1.sh           # pick a different GPU
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

# fast_layernorm's fused CUDA kernel is NOT built for Blackwell (sm_120) and there
# silently returns its input UNNORMALIZED -> the pair rep explodes to NaN. Force the
# native torch LayerNorm (same as the training scripts).
export LAYERNORM_TYPE=torch

# Data root (mmcif / bioassembly / indices live under here).
export PROTENIX_ROOT_DIR="/home/dinge/data/proj/protenix_codesign/data"

# Reduce CUDA fragmentation OOMs on the large uncropped test complexes (the
# allocator can grow segments instead of failing on a big contiguous request).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- what to evaluate (CHANGE L30) -------------------------------------------------------
RUN_DIR="./output/protenix_antibody_codesign_stage_1_20260729_103800"
# Default to the final EMA weights (what you'd normally report); override with CKPT.
# Raw (non-EMA) final weights are at ${RUN_DIR}/checkpoints/stage_1.pt
CKPT="${CKPT:-${RUN_DIR}/checkpoints/stage_1.pt}"

# GPU 0 is usually occupied (ComfyUI); default to the free GPU 1.
GPU="${GPU:-1}"

if [[ ! -f "${CKPT}" ]]; then
  echo "[error] checkpoint not found: ${CKPT}" >&2
  exit 1
fi
echo "Evaluating checkpoint: ${CKPT}"
echo "On test set: antibody_codesign_test_set  (${PROTENIX_ROOT_DIR}/test_mmcif)"
echo "GPU: ${GPU}"

LOG="${RUN_DIR}/eval_$(basename "${CKPT}" .pt).log"

CUDA_VISIBLE_DEVICES="${GPU}" torchrun --standalone --nproc_per_node=1 \
  /home/dinge/Protenix/runner/train.py \
  --model_name "protenix_base_default_v1.0.0_codesign" \
  --run_name protenix_antibody_codesign_stage_1_eval \
  --seed 42 \
  --base_dir ./output \
  --dtype bf16 \
  --eval_only true \
  --eval_structure_inpainting "${INPAINT:-true}" \
  --ema_decay 0 \
  --use_wandb false \
  --load_checkpoint_path "${CKPT}" \
  --data.num_dl_workers 2 \
  --model.N_cycle 4 \
  --sample_diffusion.N_step 200 \
  --sample_diffusion.N_sample "${N_SAMPLE:-20}" \
  --skip_amp.sample_diffusion false \
  --triangle_attention "cuequivariance" \
  --triangle_multiplicative "cuequivariance" \
  --data.train_sets antibody_codesign_train_set \
  --data.test_sets antibody_codesign_test_set \
  --data.template.enable_prot_template false \
  --data.msa.enable_prot_msa true \
  --data.msa.enable_rna_msa false \
  --loss.weight.alpha_sequence 2.0 \
  --antibody_add_antigen false \
  --antibody_min_neighborhood 0 \
  --antibody_max_neighborhood 40 \
  --antibody_mixed_prob 0.0 \
  --test_max_n_token 3840
  2>&1 | tee "${LOG}"

echo
echo "Eval finished. Per-test-set metrics are printed above (line: 'eval antibody_codesign_test_set: {...}')."
echo "Full log saved to: ${LOG}"
