#!/usr/bin/env bash
# Full test-set codesign eval on the `codesign` branch.
#
# Runs runner/train.py in --eval_only mode over the ENTIRE antibody codesign test
# set, with replacement sampling (structural inpainting at every denoising step)
# and N_sample diffusion samples per target. Dumps, per target:
#   structures/<test_set>_step0_raw/<pid>_sample{0..N-1}.cif   predicted
#   structures/<test_set>_step0_raw/<pid>_native.cif           ground truth
#   structures/<test_set>_step0_raw/<pid>_regions.json         CDR/framework map
#   predictions/<test_set>_step0_raw.fasta                     designed sequences
#
# CDR RMSD is NOT computed here -- it is a separate offline pass over the dumped
# structures (scripts/cdr_rmsd_benchmark.py); the command is printed at the end.
#
# RUNTIME: at N_SAMPLE=20 this is roughly 2-3 DAYS on one GPU. The 200-step
# diffusion loop dominates and scales with N_SAMPLE; inpainting adds an fp32
# Kabsch SVD at every step. Start it under tmux/nohup:
#
#     nohup bash antibody_codesign_full_eval.sh > full_eval.out 2>&1 &
#
# Usage:
#   bash antibody_codesign_full_eval.sh
#   N_SAMPLE=1 bash antibody_codesign_full_eval.sh    # ~4-5h, matches MFDesign's
#                                                     # single-sample protocol
#   GPU=1 CKPT=/path/to/ckpt.pt bash antibody_codesign_full_eval.sh
set -euo pipefail
cd /home/dinge/Protenix

# ---- knobs -------------------------------------------------------------------
GPU="${GPU:-0}"
N_SAMPLE="${N_SAMPLE:-20}"
INPAINT="${INPAINT:-true}"
MAX_TOKEN="${MAX_TOKEN:-3840}"
RUN_DIR="${RUN_DIR:-./output/protenix_antibody_codesign_stage_4_20260802_140148}"
CKPT="${CKPT:-${RUN_DIR}/checkpoints/stage_4.pt}"
RUN_NAME="${RUN_NAME:-protenix_codesign_fulltest}"
# Pin the interpreter: under nohup/cron the conda env is usually NOT activated,
# and a bare `torchrun` then fails with "command not found" after preflight has
# already passed.
TORCHRUN="${TORCHRUN:-/home/dinge/miniconda3/envs/protenix/bin/torchrun}"

# ---- preflight ---------------------------------------------------------------
# A 2-3 day run must not die on something knowable in the first second.

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ "${BRANCH}" != "codesign" ]]; then
  echo "[error] on branch '${BRANCH}', expected 'codesign'." >&2
  echo "        run: git checkout codesign" >&2
  exit 1
fi

if [[ ! -f "${CKPT}" ]]; then
  echo "[error] checkpoint not found: ${CKPT}" >&2
  exit 1
fi

if [[ ! -x "${TORCHRUN}" ]]; then
  if command -v torchrun >/dev/null 2>&1; then
    TORCHRUN="$(command -v torchrun)"
  else
    echo "[error] torchrun not found at ${TORCHRUN} and not on PATH." >&2
    echo "        activate the protenix env or set TORCHRUN=<path>." >&2
    exit 1
  fi
fi

# Single GPU deliberately: _log_sequence_design/_log_structure_design both
# early-return on rank != 0, so a multi-rank run would silently dump only the
# rank-0 share of structures and quietly halve the benchmark.
USED_MIB="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU}" 2>/dev/null || echo 0)"
if (( USED_MIB > 2000 )); then
  echo "[warn] GPU ${GPU} already has ${USED_MIB} MiB in use -- risk of OOM." >&2
  echo "       ctrl-C within 10s to abort, or set GPU=<other>." >&2
  sleep 10
fi

# ---- environment -------------------------------------------------------------
export PYTHONPATH="${PYTHONPATH:-}:/home/dinge/Protenix"
export PROTENIX_ROOT_DIR="/home/dinge/data/proj/protenix_codesign/data"
# The fused fast_layernorm CUDA kernel is NOT built for Blackwell (sm_120) and
# silently returns its input UNNORMALIZED there -> every loss goes NaN.
export LAYERNORM_TYPE=torch
# Let the allocator grow segments rather than fail on a big contiguous request;
# the uncropped test complexes vary hugely in size.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${RUN_DIR}/full_eval_${STAMP}.log"
mkdir -p "$(dirname "${LOG}")"

echo "branch     : ${BRANCH}"
echo "checkpoint : ${CKPT}"
echo "GPU        : ${GPU}  (single-rank; see note above)"
echo "N_sample   : ${N_SAMPLE}    inpainting: ${INPAINT}    max tokens: ${MAX_TOKEN}"
echo "log        : ${LOG}"
echo

# ---- run ---------------------------------------------------------------------
# Note: --ema_decay 0 means "no EMA wrapper"; if CKPT is itself an *_ema_*.pt
# file its weights are still used verbatim, they just are not re-averaged.
CUDA_VISIBLE_DEVICES="${GPU}" "${TORCHRUN}" --standalone --nproc_per_node=1 \
  /home/dinge/Protenix/runner/train.py \
  --model_name "protenix_base_default_v1.0.0_codesign" \
  --run_name "${RUN_NAME}" \
  --seed 42 \
  --base_dir ./output \
  --dtype bf16 \
  --eval_only true \
  --eval_structure_inpainting "${INPAINT}" \
  --ema_decay 0 \
  --use_wandb true \
  --load_checkpoint_path "${CKPT}" \
  --data.num_dl_workers 2 \
  --model.N_cycle 4 \
  --sample_diffusion.N_step 200 \
  --sample_diffusion.N_sample "${N_SAMPLE}" \
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
  --test_max_n_token "${MAX_TOKEN}" \
  2>&1 | tee "${LOG}"

# ---- next step ---------------------------------------------------------------
OUT="$(ls -dt ./output/${RUN_NAME}_* 2>/dev/null | head -1)"
PRED_DIR="${OUT}/structures/antibody_codesign_test_set_step0_raw"

echo
echo "=============================================================="
echo "Eval finished. AAR / lDDT are in the log line 'eval antibody_codesign_test_set: {...}'."
echo "Any structures that failed are logged as 'SKIPPING <pid>' -- grep for them:"
echo "    grep SKIPPING ${LOG}"
echo
echo "Structures dumped to:"
echo "    ${PRED_DIR}"
echo
echo "Now compute CDR RMSD offline (geometry only, fast):"
echo "    python scripts/cdr_rmsd_benchmark.py \\"
echo "        --pred_dir ${PRED_DIR} \\"
echo "        --out_csv  cdr_rmsd_${STAMP}.csv --no_relax"
echo
echo "NOTE: the --relax path uses an UNCONSTRAINED FastRelax, which measurably"
echo "      WORSENS Ca RMSD (H3 6.11 -> 7.42 A on the structures tested) because"
echo "      it minimises Rosetta energy over the full backbone. Use --no_relax,"
echo "      or add coordinate constraints first."
echo "=============================================================="
