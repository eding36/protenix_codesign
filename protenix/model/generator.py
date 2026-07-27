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

from typing import Any, Callable, Optional

import torch
from torch.distributions.categorical import Categorical
from torch.nn.functional import one_hot
from protenix.data import constants
from einops import rearrange

from protenix.model.sequence_decode import decode_sequence
from protenix.model.utils import centre_random_augmentation
from protenix.tfg import parse_tfg_config, TFGEngine
from protenix.utils.logger import get_logger

logger = get_logger(__name__)


class TrainingNoiseSampler:
    """
    Sample the noise-level of training samples.

    Args:
        p_mean (float, optional): gaussian mean. Defaults to -1.2.
        p_std (float, optional): gaussian std. Defaults to 1.5.
        sigma_data (float, optional): scale. Defaults to 16.0, but this is 1.0 in EDM.
    """

    def __init__(
        self,
        p_mean: float = -1.2,
        p_std: float = 1.5,
        sigma_data: float = 16.0,  # NOTE: in EDM, this is 1.0
    ) -> None:
        self.sigma_data = sigma_data
        self.p_mean = p_mean
        self.p_std = p_std
        print(f"train scheduler {self.sigma_data}")

    def __call__(
        self, size: torch.Size, device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        """Sampling

        Args:
            size (torch.Size): the target size
            device (torch.device, optional): target device. Defaults to torch.device("cpu").

        Returns:
            torch.Tensor: sampled noise-level
        """
        rnd_normal = torch.randn(size=size, device=device)
        noise_level = (rnd_normal * self.p_std + self.p_mean).exp() * self.sigma_data
        return noise_level

#TODO: implement SequenceNoiseSampler, modeling /home/dinge/MFDesign/src/boltz/data/mask/masker.py
class SequenceNoiseSampler:
    """Token masker"""

    def __init__(
        self, 
        noise_token_id,
        N_tokens,
        timesteps=200, 
        noise_schedule="linear",
        noise_type="discrete_absorb"
    ):
        self.timesteps = timesteps
        self.noise_type = noise_type
        self.n_tokens = N_tokens
        if noise_type == "discrete_absorb":
            self.noise_token_id = noise_token_id
            self.mask_rates = torch.linspace(
                0, 1, timesteps
            )
        elif noise_type == "discrete_uniform": #D3PM diffusion module
            #D3PM uniform noising parameters initialization
            s = 0.008
            t = torch.linspace(0, timesteps, timesteps + 1) / timesteps
            f_t = torch.cos(((t + s) / (1 + s)) * torch.pi / 2) ** 2
            self.alpha_bar = f_t / f_t[0] #signal retention parameter; how much of true residue is encapsulated at timestep t
            uniform_protein_dist = torch.zeros(self.n_tokens)
            # Protenix amino acids occupy token ids 0-19 (PRO_STD_RESIDUES); uniform
            # over the 20 standard amino acids. (MFDesign's boltz vocab uses 2-21.)
            uniform_protein_dist[0:20] = 1.0 / 20.0
            self.uniform_protein = uniform_protein_dist

    def __call__(self, seq, noise, seq_mask=None):
        return self.corrupt(seq, noise, seq_mask)

    def convert_noise_level(self,c_skip):
        if self.noise_type == "discrete_absorb":
            return 1 - c_skip
        elif self.noise_type == "discrete_uniform":
            return torch.minimum(torch.tensor(1.0), (1 - c_skip) * 20.0 / 19.0)

    def corrupt(self, seq, noise, seq_mask=None):
        if self.noise_type == "discrete_absorb":
            return self.absorb_corrupt(seq, noise, seq_mask)
        elif self.noise_type == "discrete_uniform":
            return self.uniform_corrupt(seq, noise, seq_mask)
        elif self.noise_type == "continuous":
            return self.continuous_corrupt(seq, noise, seq_mask)
        else:
            raise ValueError(f"No implementations for {self.noise_type} noise type")

    def absorb_corrupt(self, seq, noise_level, seq_mask=None):
        device = seq.device
        self.mask_rates = self.mask_rates.to(device)
        
        batch_mask_rates = self.mask_rates[noise_level].unsqueeze(-1).to(device)
        mask = torch.rand(seq.shape, device=device) < batch_mask_rates
        
        if seq_mask is not None:
            mask = mask & seq_mask.to(torch.bool)

        res = torch.where(mask, self.noise_token_id, seq)
        return res, mask
    
    def uniform_corrupt(self, seq, timesteps, seq_mask=None):
        device = seq.device
        self.alpha_bar = self.alpha_bar.to(device)
        self.uniform_protein = self.uniform_protein.to(device)
        
        if len(seq.shape) == 2: #guard against 1D sequences, only allows one-hot
            seqs = one_hot(seq, num_classes=self.n_tokens).float()
        else:
            seqs = seq.float()
        alpha_bar = self.alpha_bar[timesteps].view(seq.size(0), 1, 1) #fetch signal retention at specified timestep 
        res = alpha_bar * seqs + (1.0 - alpha_bar) * self.uniform_protein #aka q(x_t|x_0) 
        if seq_mask is not None: #noise only sequences labeled True under seq_mask
            res = torch.where(seq_mask.unsqueeze(-1), res, seqs) 
        return res #[1,N_tokens, 32], probability distribution of all 32 identities at each residue in N_tokens
    
    def uniform_posterior(self, seq_t, seq, timesteps, seq_mask=None):
        """
        seq_t: noised sequence at time t
        seq: model predicted gt sequence logits
        timesteps: time t
        """
        device = seq.device
        self.alpha_bar = self.alpha_bar.to(device)
        self.uniform_protein = self.uniform_protein.to(device)
        if len(seq_t.shape) == 2:
            x_t = one_hot(seq_t, num_classes=self.n_tokens).float()
        else:
            x_t = seq_t.float()
        if len(seq.shape) == 2:
            x_0 = one_hot(seq, num_classes=self.n_tokens).float()
        else:
            x_0 = seq.float()
            
        alpha = self.alpha_bar[timesteps] / (self.alpha_bar[timesteps - 1] + 1e-8)
        alpha_bar = self.alpha_bar[timesteps - 1]
        
        alpha = alpha.view(seq.size(0), 1, 1)
        alpha_bar = alpha_bar.view(seq.size(0), 1, 1)
        
        q_x_t_from_x_t_minus_1 = alpha * x_t + (1.0 - alpha) * self.uniform_protein # q( x_t )| q( x_t-1 )
        q_x_t_minus_1_from_x_0 = alpha_bar * x_0 + (1.0 - alpha_bar) * self.uniform_protein # q( x_t-1 ) | x_0 
        res = q_x_t_from_x_t_minus_1 * q_x_t_minus_1_from_x_0 # q( x_t-1 ) | q ( x_t ) ; single denoising step
        res = res / (res.sum(dim=-1, keepdim=True) + 1e-8)
        if seq_mask is not None: #denoise only sequences labeled True under seq_mask
            res = torch.where(seq_mask.unsqueeze(-1), res, x_t)
        return res  

    def continuous_corrupt(self, seq, sigmas, seq_mask, omega=0.25, clamp_v=3.0):
        if len(seq.shape) == 2:
            seqs = one_hot(seq, num_classes=self.n_tokens).float()
        else:
            seqs = seq.float()
        seq_noise = torch.randn_like(seqs) * seq_mask.unsqueeze(-1)
        res = seqs + omega * sigmas * seq_noise
        res = torch.clamp(res, min=-clamp_v, max=clamp_v)
        return res
class InferenceNoiseScheduler:
    """
    Scheduler for noise-level (time steps).

    Args:
        s_max (float, optional): maximal noise level. Defaults to 160.0.
        s_min (float, optional): minimal noise level. Defaults to 4e-4.
        rho (float, optional): the exponent numerical part. Defaults to 7.
        sigma_data (float, optional): scale. Defaults to 16.0, but this is 1.0 in EDM.
    """

    def __init__(
        self,
        s_max: float = 160.0,
        s_min: float = 4e-4,
        rho: float = 7,
        sigma_data: float = 16.0,  # NOTE: in EDM, this is 1.0
    ) -> None:
        self.sigma_data = sigma_data
        self.s_max = s_max
        self.s_min = s_min
        self.rho = rho
        print(f"inference scheduler {self.sigma_data}")

    def __call__(
        self,
        N_step: int = 200,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Schedule the noise-level (time steps). No sampling is performed.

        Args:
            N_step (int, optional): number of time steps. Defaults to 200.
            device (torch.device, optional): target device. Defaults to torch.device("cpu").
            dtype (torch.dtype, optional): target dtype. Defaults to torch.float32.

        Returns:
            torch.Tensor: noise-level (time_steps)
                [N_step+1]
        """
        step_size = 1 / N_step
        step_indices = torch.arange(N_step + 1, device=device, dtype=dtype)
        t_step_list = (
            self.sigma_data
            * (
                self.s_max ** (1 / self.rho)
                + step_indices
                * step_size
                * (self.s_min ** (1 / self.rho) - self.s_max ** (1 / self.rho))
            )
            ** self.rho
        )
        # replace the last time step by 0
        t_step_list[..., -1] = 0  # t_N = 0

        return t_step_list


# ---------------------------------------------------------------------------
# Antibody-codesign inference: iterative sequence (D3PM) reverse process, run
# in unison with the coordinate sampler (MFDesign AtomDiffusion.sample analog).
# All state is unbatched [N_token]; a batch dim of 1 is added for the masker.
# ---------------------------------------------------------------------------


def _splice_restype_slice(s_inputs, masked_seq, offset, width):
    """Replace the ``[offset : offset+width]`` restype slice of ``s_inputs`` with
    the one-hot of ``masked_seq`` (MFDesign "s_replaced")."""
    new_restype = one_hot(masked_seq, num_classes=width).to(s_inputs.dtype)
    return torch.cat(
        [s_inputs[..., :offset], new_restype, s_inputs[..., offset + width :]], dim=-1
    )


def _seed_sequence(cdr_corrupter, gt_ids, cdr_mask, noise_type):
    """Seed the noised sequence at the maximum timestep (CDRs fully corrupted)."""
    seq = gt_ids.unsqueeze(0)  # [1, N_token]
    cdr = cdr_mask.unsqueeze(0)
    t = torch.full(
        (1,), cdr_corrupter.timesteps - 1, device=gt_ids.device, dtype=torch.long
    )
    res = cdr_corrupter(seq, t, cdr)
    if noise_type == "discrete_absorb":
        return res[0].squeeze(0)  # [N_token] ids (UNK at CDRs)
    return Categorical(probs=res).sample().squeeze(0)  #sample from Categorical distribution at each corrupted residue


def _sequence_reverse_step(
    cdr_corrupter, seq_noisy, seq_logits, gt_ids, cdr_mask, t_seq, noise_type,
    temperature, sample,
):
    """One reverse (t -> t-1) sequence-diffusion step.

    ``seq_noisy``/``gt_ids``/``cdr_mask``: [N_token]; ``seq_logits``: [N_token, 20].
    Returns the updated noised sequence ids [N_token] after one step of denoising
    seq_noisy: current noised sequence
    seq_logits: sequence model's sequence logit predictions
    gt_ids: ground truth sequence
    t_seq: current time step of denoising
    temperature: higher = less random, lower = more random

    """
    device = seq_noisy.device
    cdr = cdr_mask.unsqueeze(0).to(torch.bool)  # [1, N_token]
    gt = gt_ids.unsqueeze(0)
    if noise_type == "discrete_absorb":
        denoised = (
            Categorical(logits=seq_logits * temperature).sample() # for each residue, sample from a distribution of model predicted logits
            if sample
            else seq_logits.argmax(dim=-1) #seq_logits condensed into token indices
        ).unsqueeze(0)
        tm1 = torch.full((1,), t_seq - 1, device=device, dtype=torch.long)
        seq_new = cdr_corrupter(denoised, tm1, cdr)[0] #corrupt fully denoised seq to t-1
        # Outside the design mask nothing is ever corrupted, so the framework /
        # antigen are simply pinned to ground truth.
        return torch.where(cdr, seq_new, gt).squeeze(0)
    # discrete_uniform: x0 distribution over the 20 AA slots, D3PM posterior to t-1.
    probs = seq_logits.new_zeros(seq_logits.shape[0], cdr_corrupter.n_tokens) #[N_tokens, 32]
    probs[:, 0:20] = torch.softmax(seq_logits * temperature, dim=-1) #only fill the first 20 columns with logits, since the sequence prediction head only predicts the 20 AAs
    probs = probs.unsqueeze(0)  # [1, N_token, 32]
    t_t = torch.full((1,), t_seq, device=device, dtype=torch.long) #fetch current timestep of denoising
    post = cdr_corrupter.uniform_posterior(seq_noisy.unsqueeze(0), probs, t_t, cdr) # denoise seq_noisy for 1 timestep, conditioned on model's predicted gt sequence distribution logits
    gt_onehot = one_hot(gt, num_classes=cdr_corrupter.n_tokens).to(post.dtype)
    post = torch.where(cdr.unsqueeze(-1), post, gt_onehot) #only corrupt CDR tokens, rest is ground truth seq
    return Categorical(probs=post).sample().squeeze(0)  # ids, no +2


def sample_diffusion(
    denoise_net: Callable,
    input_feature_dict: dict[str, Any],
    s_inputs: torch.Tensor,
    s_trunk: torch.Tensor,
    z_trunk: torch.Tensor,
    pair_z: torch.Tensor,
    p_lm: torch.Tensor,
    c_l: torch.Tensor,
    noise_schedule: torch.Tensor,
    N_sample: int = 1,
    gamma0: float = 0.8,
    gamma_min: float = 1.0,
    noise_scale_lambda: float = 1.003,
    step_scale_eta: float = 1.5,
    diffusion_chunk_size: Optional[int] = None,
    inplace_safe: bool = False,
    attn_chunk_size: Optional[int] = None,
    enable_efficient_fusion: bool = False,
    guidance_configs: Optional[dict[str, Any]] = None,
    sequence_train: bool = False,
    cdr_corrupter: Optional["SequenceNoiseSampler"] = None,
    noise_type: str = "discrete_uniform",
    restype_offset: Optional[int] = None,
    restype_width: Optional[int] = None,
    seq_temperature: float = 1.0,
    seq_sample: bool = True,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Implements Algorithm 18 in AF3.
    It performances denoising steps from time 0 to time T.
    This outer func performs denoising for all samples for all batches 
    The time steps (=noise levels) are given by noise_schedule.

    Args:
        denoise_net (Callable): the network that performs the denoising step.
        input_feature_dict (dict[str, Any]): input meta feature dict
        s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
            [..., N_tokens, c_s_inputs]
        s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
            [..., N_tokens, c_s]
        z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
            [..., N_tokens, N_tokens, c_z]
        pair_z (torch.Tensor): pair feature embedding from InputFeatureEmbedder
            [..., N_tokens, N_tokens, c_z_inputs]
        p_lm (torch.Tensor): MSA embedding
            [..., N_tokens, c_p_lm]
        c_l (torch.Tensor): ligand embedding
            [..., N_tokens, c_c_l]
        noise_schedule (torch.Tensor): noise-level schedule (which is also the time steps) since sigma=t.
            [N_iterations]
        N_sample (int): number of generated samples
        gamma0 (float): params in Alg.18.
        gamma_min (float): params in Alg.18.
        noise_scale_lambda (float): params in Alg.18.
        step_scale_eta (float): params in Alg.18.
        diffusion_chunk_size (Optional[int]): Chunk size for diffusion operation. Defaults to None.
        inplace_safe (bool): Whether to use inplace operations safely. Defaults to False.
        attn_chunk_size (Optional[int]): Chunk size for attention operation. Defaults to None.
        enable_efficient_fusion (bool): Whether to enable efficient fusion. Defaults to False.
        guidance_configs (Optional[dict[str, Any]]): training free guidance configs. Defaults to None.

    Returns:
        tuple[torch.Tensor, Optional[torch.Tensor]]:
            x_l: the denoised coordinates of x in inference stage
                [..., N_sample, N_atom, 3]
            k_denoised: the final-step sequence prediction when the denoiser is a
                sequence model (sequence_train), else None
                [..., N_sample, N_tokens, vocab_size]
    """
    N_atom = input_feature_dict["atom_to_token_idx"].size(-1)
    batch_shape = s_inputs.shape[:-2]
    device = s_inputs.device
    dtype = s_inputs.dtype
    tfg_cfg = parse_tfg_config(guidance_configs)
    if tfg_cfg.enable:
        logger.info("Guidance is enabled.")
        tfg = TFGEngine(tfg_cfg, device=device, dtype=dtype)

    def _chunk_sample_diffusion(chunk_n_sample, inplace_safe):
        """
        Denoises seq & struct for all items in each chunk (batch)
        """
        # init noise
        # [..., N_sample, N_atom, 3]
        x_l = noise_schedule[0] * torch.randn(
            size=(*batch_shape, chunk_n_sample, N_atom, 3), device=device, dtype=dtype
        )  # NOTE: set seed in distributed training

        # Sequence prediction from the denoiser; stays None for non-sequence
        # models and holds the final (cleanest) step's prediction otherwise.
        k_denoised = None

        # Antibody-codesign inference: iterative sequence D3PM, one shared sequence
        # per structure, updated in lock-step with the coordinate sampler. Requires
        # the masker (cdr_corrupter) and the restype-slice location in s_inputs.
        seq_rollout = (
            sequence_train
            and cdr_corrupter is not None
            and restype_offset is not None
            and restype_width is not None
            and "seq" in input_feature_dict
        )
        n_coord_steps = len(noise_schedule) - 1
        if seq_rollout:
            # The sequence D3PM reverse chain must advance exactly one timestep per
            # coordinate step, so the coordinate sampler's step count must equal the
            # masker's timesteps (MFDesign asserts the same). Set the inference
            # sample_diffusion.N_step equal to model.diffusion_module.N_steps_seq.
            assert n_coord_steps == cdr_corrupter.timesteps, (
                f"sequence rollout requires N_step ({n_coord_steps}) == "
                f"n_steps_seq ({cdr_corrupter.timesteps}); set "
                f"sample_diffusion.N_step to N_steps_seq for codesign inference."
            )
            gt_ids = input_feature_dict["seq"]
            cdr_mask = input_feature_dict["cdr_mask"]
            seq_noisy = _seed_sequence(cdr_corrupter, gt_ids, cdr_mask, noise_type) #seq with fully noised CDR region 

        for step_i, (c_tau_last, c_tau) in enumerate( #iterative denoising loop
            zip(noise_schedule[:-1], noise_schedule[1:])
        ):
            # [..., N_sample, N_atom, 3]
            x_l = (
                centre_random_augmentation(x_input_coords=x_l, N_sample=1)
                .squeeze(dim=-3)
                .to(dtype)
            )

            # Denoise with a predictor-corrector sampler
            # 1. Add noise to move x_{c_tau_last} to x_{t_hat}
            gamma = float(gamma0) if c_tau > gamma_min else 0
            t_hat = c_tau_last * (gamma + 1)

            delta_noise_level = torch.sqrt(t_hat**2 - c_tau_last**2)
            x_noisy = x_l + noise_scale_lambda * delta_noise_level * torch.randn(
                size=x_l.shape, device=device, dtype=dtype
            )

            # 2. Denoise from x_{t_hat} to x_{c_tau}
            # Euler step only
            t_hat = (
                t_hat.reshape((1,) * (len(batch_shape) + 1))
                .expand(*batch_shape, chunk_n_sample)
                .to(dtype)
            )

            # Splice the current noised sequence into the restype slice of s_inputs
            # (MFDesign "s_replaced"); plain s_inputs otherwise.
            s_inputs_step = (
                _splice_restype_slice(s_inputs, seq_noisy, restype_offset, restype_width)
                if seq_rollout
                else s_inputs
            )

            if tfg_cfg.enable:
                x_l, k_denoised = tfg.step(
                    denoise_net,
                    x=x_noisy,
                    t_hat=t_hat,
                    input_feature_dict=input_feature_dict,
                    s_inputs=s_inputs_step,
                    s_trunk=s_trunk,
                    z_trunk=z_trunk,
                    pair_z=pair_z,
                    p_lm=p_lm,
                    c_l=c_l,
                    chunk_size=attn_chunk_size,
                    inplace_safe=inplace_safe,
                    enable_efficient_fusion=enable_efficient_fusion,
                    c_tau=c_tau,
                    step_i=step_i,
                    num_diffusion_steps=len(noise_schedule) - 1,
                    step_scale_eta=step_scale_eta,
                )
            else:
                x_denoised, k_denoised = denoise_net(
                    x_noisy=x_noisy,
                    t_hat_noise_level=t_hat,
                    input_feature_dict=input_feature_dict,
                    s_inputs=s_inputs_step,
                    s_trunk=s_trunk,
                    z_trunk=z_trunk,
                    pair_z=pair_z,
                    p_lm=p_lm,
                    c_l=c_l,
                    chunk_size=attn_chunk_size,
                    inplace_safe=inplace_safe,
                    enable_efficient_fusion=enable_efficient_fusion,
                )

                delta = (x_noisy - x_denoised) / t_hat[
                    ..., None, None
                ]  # Line 9 of AF3 uses 'x_l_hat' instead, which we believe  is a typo.
                dt = c_tau - t_hat
                x_l = x_noisy + step_scale_eta * dt[..., None, None] * delta

            # Sequence reverse step (t -> t-1), skipped on the final step (decode).
            if seq_rollout and k_denoised is not None:
                # fetch model denoised pred logits at t_hat
                seq_logits = k_denoised.mean(dim=-3) if k_denoised.dim() >= 3 else k_denoised
                # One coordinate step == one sequence step (N == T is asserted), so
                # the sequence timestep descends T-1 -> 0 exactly (MFDesign's
                # seq_timesteps = range(T)[::-1]).
                t_seq = (cdr_corrupter.timesteps - 1) - step_i
                if t_seq > 0:
                    #denoise sequence by 1 step
                    seq_noisy = _sequence_reverse_step(
                        cdr_corrupter, seq_noisy, seq_logits, gt_ids, cdr_mask,
                        t_seq, noise_type, seq_temperature, seq_sample,
                    )

        if seq_rollout and k_denoised is not None: #at final time step t_0, decode sequence logits -> final designed CDR sequence
            # Final decode: designed residues from the last logits, framework from GT.
            seq_logits = k_denoised.mean(dim=-3) if k_denoised.dim() >= 3 else k_denoised
            seq_ids = decode_sequence(
                seq_logits, cdr_mask, gt_ids,
                temperature=seq_temperature, sample=seq_sample,
            )
            return x_l, seq_ids  # k slot now carries decoded token ids [N_token]

        return x_l, k_denoised

    if diffusion_chunk_size is None:
        x_l, k_denoised = _chunk_sample_diffusion(N_sample, inplace_safe=inplace_safe)
    else:
        x_l = []
        k_chunks = []
        no_chunks = N_sample // diffusion_chunk_size + (
            N_sample % diffusion_chunk_size != 0
        ) #number of batches/chunks to denoise
        for i in range(no_chunks):
            chunk_n_sample = (
                diffusion_chunk_size
                if i < no_chunks - 1
                else N_sample - i * diffusion_chunk_size
            )
            chunk_x_l, chunk_k = _chunk_sample_diffusion(
                chunk_n_sample, inplace_safe=inplace_safe
            )
            x_l.append(chunk_x_l)
            k_chunks.append(chunk_k)
        x_l = torch.cat(x_l, -3)  # [..., N_sample, N_atom, 3]
        # k is None for non-sequence models; only aggregate real predictions.
        if not all(k is not None for k in k_chunks):
            k_denoised = None
        elif k_chunks[0].dim() == 1:
            # Sequence rollout: each chunk yields one decoded sequence [N_token];
            # stack into [n_chunks, N_token] (one design per chunk).
            k_denoised = torch.stack(k_chunks, dim=0)
        else:
            k_denoised = torch.cat(k_chunks, -3)  # per-sample logits
    return x_l, k_denoised


def sample_diffusion_training(
    noise_sampler: TrainingNoiseSampler,
    cdr_corrupter: SequenceNoiseSampler,
    denoise_net: Callable,
    label_dict: dict[str, Any],
    input_feature_dict: dict[str, Any],
    s_inputs: torch.Tensor,
    s_trunk: torch.Tensor,
    z_trunk: torch.Tensor,
    pair_z: torch.Tensor,
    p_lm: torch.Tensor,
    c_l: torch.Tensor,
    c_s: int = 384,
    N_tokens: int = 384,
    N_sample: int = 1,
    diffusion_chunk_size: Optional[int] = None,
    use_conditioning: bool = True,
    enable_efficient_fusion: bool = False,
    sequence_train: bool = False,
    noise_type: str = "discrete_uniform",
    n_steps_seq: int = 200,
    seq_sigma_schedule: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Implements diffusion training as described in AF3 Appendix at page 23.
    It performances denoising steps from time 0 to time T.
    The time steps (=noise levels) are given by noise_schedule.

    Args:
        noise_sampler (TrainingNoiseSampler): sampler for training noise-level.
        denoise_net (Callable): the network that performs the denoising step.
        label_dict (dict[str, Any]) : a dictionary containing the followings.
            "coordinate": the ground-truth coordinates
                [..., N_atom, 3]
            "coordinate_mask": whether true coordinates exist.
                [..., N_atom]
        input_feature_dict (dict[str, Any]): input meta feature dict
        s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
            [..., N_tokens, c_s_inputs]
        s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
            [..., N_tokens, c_s]
        z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
            [..., N_tokens, N_tokens, c_z]
        pair_z (torch.Tensor): pair feature embedding from InputFeatureEmbedder
            [..., N_tokens, N_tokens, c_z_inputs]
        p_lm (torch.Tensor): MSA embedding
            [..., N_tokens, c_p_lm]
        c_l (torch.Tensor): ligand embedding
            [..., N_tokens, c_c_l]
        N_sample (int): number of training samples
        diffusion_chunk_size (Optional[int]): Chunk size for diffusion operation. Defaults to None.
        use_conditioning (bool): Whether to use conditioning. Defaults to True.
        enable_efficient_fusion (bool): Whether to enable efficient fusion. Defaults to False.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            x_gt_augment: the augmented ground-truth coordinates [..., N_sample, N_atom, 3]
            x_denoised: the denoised coordinates [..., N_sample, N_atom, 3]
            sigma: the sampled noise-level [..., N_sample]
    """
    batch_size_shape = label_dict["coordinate"].shape[:-2]
    device = label_dict["coordinate"].device
    dtype = label_dict["coordinate"].dtype
    # Areate N_sample versions of the input structure by randomly rotating and translating
    x_gt_augment = centre_random_augmentation(
        x_input_coords=label_dict["coordinate"],
        N_sample=N_sample,
        mask=label_dict["coordinate_mask"],
    ).to(
        dtype
    )  # [..., N_sample, N_atom, 3]

    # Sample the structure noise level (sigma). For sequence codesign it is
    # coupled to the sequence noise level (seq_t).
    # sigma = schedule[T-1-seq_t] so high sequence noise <-> high structure noise (the
    # model never sees mismatched noise levels). 
    coupled_seq = sequence_train and noise_type != "continuous"
    seq_t = None
    if coupled_seq:
        assert seq_sigma_schedule is not None, (
            "seq_sigma_schedule required for coupled discrete sequence training"
        )
        seq_t = torch.randint(n_steps_seq, size=(1,), device=device)  # sampled sequence noise level
        coupled_sigma = seq_sigma_schedule.to(device=device, dtype=dtype)[
            n_steps_seq - 1 - seq_t
        ]  # corresponding structure noise level 
        sigma = (
            coupled_sigma.reshape((1,) * len(batch_size_shape) + (1,))
            .expand(*batch_size_shape, N_sample)
            .to(dtype)
        )  # corresponding structure noise level 
    else: #if sequence codesign does not occur
        sigma = noise_sampler(size=(*batch_size_shape, N_sample), device=device).to(dtype)
    # noise: [..., N_sample, N_atom, 3]
    noise = torch.randn_like(x_gt_augment, dtype=dtype) * sigma[..., None, None]

    """Sequence noising process, only at CDR residues"""
    if sequence_train:
        # One diffusion timestep per SEQUENCE (MFDesign samples per-sequence, not
        # per-token). Protenix runs unbatched (seq is [N_token]); add a batch dim of
        # 1 so the masker's [B, L] kernels apply, then squeeze it back off.
        seq = input_feature_dict["seq"].unsqueeze(0)            # [1, N_token]
        cdr_mask = input_feature_dict["cdr_mask"].unsqueeze(0)  # [1, N_token]
        # Reuse the coupled t for discrete; continuous has no discrete timestep.
        t = (
            seq_t
            if seq_t is not None
            else torch.randint(n_steps_seq, size=(1,), device=seq.device)
        )
        input_feature_dict["time"] = t
        # Corrupt only the CDR (design) residues at timestep t (uses the masker built
        # once in Protenix.__init__ and threaded in as cdr_corrupter).
        res = cdr_corrupter(seq, t, cdr_mask)
        if noise_type == "discrete_absorb":
            masked_seq, seq_mask = res
            input_feature_dict["masked_seq"] = masked_seq.squeeze(0)   # [N_token]
            input_feature_dict["seq_mask"] = seq_mask.squeeze(0)
        elif noise_type == "discrete_uniform":
            # res is a per-token distribution [1, N_token, vocab]; sample token ids.
            # No boltz "+2" offset: Protenix amino acids already occupy ids 0-19.
            input_feature_dict["masked_seq"] = Categorical(probs=res).sample().squeeze(0)

        # Overwrite the restype slice of s_inputs with the corrupted one-hot
        # (MFDesign "s_replaced"). Width = restype vocab (N_tokens=32), not token count.
        new_restype = one_hot(
            input_feature_dict["masked_seq"], num_classes=N_tokens
        ).to(s_inputs.dtype)
        s_inputs = torch.cat([
            s_inputs[..., :c_s],                # atom feature block, kept
            new_restype,                        # noised restype slice
            s_inputs[..., c_s + N_tokens:],     # profile / deletion, kept
        ], dim=-1)

    # Get denoising outputs [..., N_sample, N_atom, 3]. denoise_net always returns
    # (x_denoised, k_denoised); k_denoised is None unless the model is sequence_train.
    if diffusion_chunk_size is None:
        x_denoised, k_denoised = denoise_net(
            x_noisy=x_gt_augment + noise,
            t_hat_noise_level=sigma,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            pair_z=pair_z,
            p_lm=p_lm,
            c_l=c_l,
            use_conditioning=use_conditioning,
            enable_efficient_fusion=enable_efficient_fusion,
        )
    else:
        x_denoised = []
        k_chunks = []
        no_chunks = N_sample // diffusion_chunk_size + (
            N_sample % diffusion_chunk_size != 0
        )
        for i in range(no_chunks):
            x_noisy_i = (x_gt_augment + noise)[
                ..., i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size, :, :
            ]
            t_hat_noise_level_i = sigma[
                ..., i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size
            ]
            x_denoised_i, k_denoised_i = denoise_net(
                x_noisy=x_noisy_i,
                t_hat_noise_level=t_hat_noise_level_i,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                pair_z=pair_z,
                p_lm=p_lm,
                c_l=c_l,
                use_conditioning=use_conditioning,
                enable_efficient_fusion=enable_efficient_fusion,
            )
            x_denoised.append(x_denoised_i)
            k_chunks.append(k_denoised_i)
        x_denoised = torch.cat(x_denoised, dim=-3)
        # k is None for non-sequence models; only concatenate real predictions.
        k_denoised = (
            torch.cat(k_chunks, dim=-3)
            if all(k is not None for k in k_chunks)
            else None
        )
    return x_gt_augment, x_denoised, k_denoised, sigma
