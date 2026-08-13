#!/usr/bin/env bash
# Full test-set codesign eval.
#
# Runs runner/train.py in --eval_only mode over the ENTIRE antibody codesign test
# set, with replacement sampling (structural inpainting at every denoising step)
# and N_sample diffusion samples per target. Each sample now rolls out its OWN
# sequence, so N_SAMPLE=20 yields 20 DISTINCT designs per target -- previously the
# per-sample logits were averaged and 20 structures collapsed onto 1 sequence.
# AAR is reported best-of-N (seq_acc), with the average also logged as
# seq_acc_mean. NOTE best-of-N is NOT comparable to MFDesign's published AAR:
# their predict.py runs --diffusion_samples 20 but eval_codesign.py scores rank 0
# only. Quote best-of-N as its own number, or compare using seq_acc_mean.
# Dumps, per target:
#   structures/<test_set>_step0_raw/<pid>_sample{0..N-1}.cif   predicted
#   structures/<test_set>_step0_raw/<pid>_native.cif           ground truth
#   structures/<test_set>_step0_raw/<pid>_regions.json         CDR/framework map
#   predictions/<test_set>_step0_raw.fasta                     designed sequences
#
# CDR RMSD is NOT computed here -- it is a separate offline pass over the dumped
# structures (scripts/cdr_rmsd_benchmark.py); the command is printed at the end.
#
# RUNTIME: at N_SAMPLE=20 this is roughly 2-3 DAYS on ONE GPU, ~half that split
# across two (default GPU=0,1 -> one rank per GPU). The 200-step
# diffusion loop dominates and scales with N_SAMPLE; inpainting adds an fp32
# Kabsch SVD at every step. Start it under tmux/nohup:
#
#     nohup bash antibody_codesign_full_eval.sh > full_eval.out 2>&1 &
#
# Usage:
#   bash antibody_codesign_full_eval.sh
#   N_SAMPLE=1 bash antibody_codesign_full_eval.sh    # ~4-5h, one design per target
#   GPU=1 CKPT=/path/to/ckpt.pt bash antibody_codesign_full_eval.sh
set -euo pipefail
cd /home/dinge/Protenix

# ---- knobs -------------------------------------------------------------------
# Comma-separated GPU list; one rank per GPU. Safe under DDP now that the dump
# functions are no longer rank-gated (each rank writes its own disjoint shard).
GPU="${GPU:-0,1}"
NPROC="$(awk -F, '{print NF}' <<< "${GPU}")"
N_SAMPLE="${N_SAMPLE:-20}"   # 20 designs per target, matching MFDesign predict.py
INPAINT="${INPAINT:-true}"   # replacement sampling (MFDesign --structure_inpainting)
# MUST match what the checkpoint was TRAINED with -- the sampler branches on this.
# Omitting the flag silently falls back to the config default (discrete_uniform),
# which rolls the sequence out with a different reverse process than training used
# and voids every metric. Stages 3 and 4 both train with discrete_absorb.
SEQ_NOISE="${SEQ_NOISE:-discrete_absorb}"
# MAX_TOKEN="${MAX_TOKEN:-3840}"
MAX_TOKEN="3840"
RUN_DIR="${RUN_DIR:-./output/protenix_antibody_codesign_stage_4_20260802_140148}"
CKPT="${CKPT:-${RUN_DIR}/checkpoints/stage_4.pt}"
RUN_NAME="${RUN_NAME:-protenix_codesign_fulltest}"
# Pin the interpreter: under nohup/cron the conda env is usually NOT activated,
# and a bare `torchrun` then fails with "command not found" after preflight has
# already passed.
TORCHRUN="${TORCHRUN:-/home/dinge/miniconda3/envs/protenix/bin/torchrun}"

# ---- preflight ---------------------------------------------------------------
# A 2-3 day run must not die on something knowable in the first second.

# This script lives on base_codesign, angular_diffusion and structural_tokendenoiser,
# which train different models -- so the guard is against launching a multi-day
# run from whichever branch happened to be checked out, not against any one
# branch. Override when you genuinely mean to evaluate another one.
EXPECT_BRANCH="${EXPECT_BRANCH:-base_codesign}"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ "${BRANCH}" != "${EXPECT_BRANCH}" ]]; then
  echo "[error] on branch '${BRANCH}', expected '${EXPECT_BRANCH}'." >&2
  echo "        git checkout ${EXPECT_BRANCH}    (to evaluate that branch)" >&2
  echo "        EXPECT_BRANCH=${BRANCH} bash $0  (to evaluate this one)" >&2
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

# Cross-check the sampler against what the checkpoint was trained with. A
# mismatch here is silent at runtime and voids every metric, so refuse to start.
TRAIN_SH="$(ls antibody_codesign*stage_4*.sh 2>/dev/null | head -1)"
if [[ -n "${TRAIN_SH}" ]]; then
  TRAINED_NOISE="$(grep -oE 'sequence_noise_type[= ][^ \\]*' "${TRAIN_SH}" \
    | head -1 | awk '{print $NF}' | tr -d '"')"
  if [[ -n "${TRAINED_NOISE}" && "${TRAINED_NOISE}" != "${SEQ_NOISE}" ]]; then
    echo "[error] SEQ_NOISE='${SEQ_NOISE}' but ${TRAIN_SH} trained with" >&2
    echo "        '${TRAINED_NOISE}'. Evaluating with a different reverse" >&2
    echo "        process than training voids the metrics. Set SEQ_NOISE=${TRAINED_NOISE}." >&2
    exit 1
  fi
fi

for g in ${GPU//,/ }; do
  USED_MIB="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${g}" 2>/dev/null || echo 0)"
  if (( USED_MIB > 2000 )); then
    echo "[warn] GPU ${g} already has ${USED_MIB} MiB in use -- risk of OOM." >&2
    echo "       ctrl-C within 10s to abort, or set GPU=<other>." >&2
    sleep 10
  fi
done

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
echo "GPU        : ${GPU}  (${NPROC} rank(s))"
echo "N_sample   : ${N_SAMPLE}    inpainting: ${INPAINT}    max tokens: ${MAX_TOKEN}"
echo "seq noise  : ${SEQ_NOISE}  (must match training)"
echo "log        : ${LOG}"
echo

# ---- run ---------------------------------------------------------------------
# Note: --ema_decay 0 means "no EMA wrapper"; if CKPT is itself an *_ema_*.pt
# file its weights are still used verbatim, they just are not re-averaged.
CUDA_VISIBLE_DEVICES="${GPU}" "${TORCHRUN}" --standalone --nproc_per_node="${NPROC}" \
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
  --model.diffusion_module.sequence_noise_type "${SEQ_NOISE}" \
  --infer_setting.sample_diffusion_chunk_size 2 \
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
if (( NPROC > 1 )); then
  echo "Multi-rank run: sequences are sharded per rank. Merge them with"
  echo "    cat ${OUT}/predictions/*_rank*.fasta > ${OUT}/predictions/all.fasta"
  echo "(structures are per-PDB and need no merging)."
  echo
fi
# ---- CDR RMSD (automatic) -----------------------------------------------------
# Runs here so a 2-3 day eval yields BOTH numbers without a second manual step.
# Geometry only, no GPU, minutes. Non-fatal: a failure here must not obscure the
# eval that just finished, so the command is echoed for a manual retry.
RMSD_CSV="${OUT}/cdr_rmsd_${STAMP}.csv"
if [[ -d "${PRED_DIR}" ]]; then
  echo "Computing CDR RMSD -> ${RMSD_CSV}"
  if "${PYBIN:-/home/dinge/miniconda3/envs/protenix/bin/python}" \
        scripts/cdr_rmsd_benchmark.py \
        --pred_dir "${PRED_DIR}" \
        --out_csv  "${RMSD_CSV}" --no_relax; then
    echo "CDR RMSD written to ${RMSD_CSV}"
    echo "RANK0 is the MFDesign-comparable column (designs are confidence-sorted,"
    echo "so sample 0 is the highest-confidence design)."
  else
    echo "[warn] CDR RMSD failed; the eval itself is unaffected. Retry with:"
    echo "    python scripts/cdr_rmsd_benchmark.py --pred_dir ${PRED_DIR} \\"
    echo "        --out_csv ${RMSD_CSV} --no_relax"
  fi
else
  echo "[warn] no structures at ${PRED_DIR}; skipping CDR RMSD."
fi
echo
echo "NOTE: the --relax path uses an UNCONSTRAINED FastRelax, which measurably"
echo "      WORSENS Ca RMSD (H3 6.11 -> 7.42 A on the structures tested) because"
echo "      it minimises Rosetta energy over the full backbone. Use --no_relax,"
echo "      or add coordinate constraints first."
echo "=============================================================="
