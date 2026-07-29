# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# fast_layernorm is used by default, no need to set explicitly. Set LAYERNORM_TYPE=torch to disable.
# NOTE: the fused fast_layernorm CUDA kernel is NOT built for Blackwell (sm_120) and
# silently returns its input UNNORMALIZED there, so the pair rep explodes to NaN.
# Force the native torch LayerNorm on this GPU.
export LAYERNORM_TYPE=torch
# Kernel options:
# - triangle_attention: supports 'triattention', 'cuequivariance', 'deepspeed', 'torch'
# - triangle_multiplicative: supports 'cuequivariance', 'torch'

# Specify your data root directory by uncommenting the following line.
export PROTENIX_ROOT_DIR="/home/dinge/data/proj/protenix_codesign/data"
# wget -P $PROTENIX_ROOT_DIR/checkpoint/ https://protenix.tos-cn-beijing.volces.com/checkpoint/protenix_base_default_v1.0.0.pt
checkpoint_path="${PROTENIX_ROOT_DIR}/checkpoint/protenix_base_default_v1.0.0.pt"
torchrun --standalone --nproc_per_node=1 /home/dinge/Protenix/runner/train.py \
--model_name "protenix_base_default_v1.0.0_codesign" \
--run_name protenix_antibody_codesign_stage_1 \
--seed 42 \
--base_dir ./output \
--dtype fp32 \
--project protenix \
--use_wandb true \
--diffusion_batch_size 48 \
--eval_interval 5000 \
--log_interval 50 \
--checkpoint_interval 400 \
--ema_decay 0.999 \
--train_crop_size 256 \
--max_steps 100000 \
--warmup_steps 2000 \
--lr 0.001 \
--model.N_cycle 4 \
--sample_diffusion.N_step 200 \
--triangle_attention "torch" \
--triangle_multiplicative "torch" \
--load_checkpoint_path ${checkpoint_path} \
--load_ema_checkpoint_path ${checkpoint_path} \
--data.train_sets antibody_codesign_train_set \
--data.test_sets antibody_codesign_test_set \
--data.template.enable_prot_template false \
--data.msa.enable_prot_msa true \
--data.msa.enable_rna_msa false \
--loss.weight.alpha_sequence 2.0 \
--antibody_add_antigen false \
--antibody_min_neighborhood 0 \
--antibody_max_neighborhood 40 \
--antibody_mixed_prob 0.0
