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

import datetime
import hashlib
import logging
import os
import time
from argparse import Namespace
from contextlib import nullcontext
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.distributed as dist
import wandb

from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_model_type import model_configs
from protenix.config.config import parse_configs, parse_sys_args, save_config
from protenix.data.pipeline.dataloader import get_dataloaders
from protenix.metrics.lddt_metrics import LDDTMetrics
from protenix.model.loss import ProtenixLoss
from protenix.model.protenix import Protenix
from protenix.utils.distributed import DIST_WRAPPER
from protenix.utils.lr_scheduler import FinetuneLRScheduler, get_lr_scheduler
from protenix.utils.metrics import SimpleMetricAggregator
from protenix.utils.permutation.permutation import SymmetricPermutation
from protenix.utils.seed import seed_everything
from protenix.utils.torch_utils import autocasting_disable_decorator, to_device
from protenix.utils.training import get_optimizer, is_loss_nan_check
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from runner.ema import EMAWrapper

# Disable WANDB's console output capture to reduce unnecessary logging
os.environ["WANDB_CONSOLE"] = "off"

torch.serialization.add_safe_globals([Namespace])


class AF3Trainer(object):
    """
    Trainer class for the Alphafold3 model.

    Args:
        configs (Any): Configuration object for training.
    """

    def __init__(self, configs: Any) -> None:
        self.configs = configs
        self.init_env()
        self.init_basics()
        self.init_log()
        self.init_model()
        self.init_loss()
        self.init_data()
        self.try_load_checkpoint()

    def init_basics(self) -> None:
        """
        Initialize basic training parameters and directory structure.
        """
        # Step means effective step considering accumulation
        self.step = 0
        # Global_step equals to self.step * self.iters_to_accumulate
        self.global_step = 0
        self.start_step = 0
        # Add for grad accumulation, it can increase real batch size
        self.iters_to_accumulate = self.configs.iters_to_accumulate

        self.run_name = self.configs.run_name + "_" + time.strftime("%Y%m%d_%H%M%S")
        run_names = DIST_WRAPPER.all_gather_object(
            self.run_name if DIST_WRAPPER.rank == 0 else None
        )
        self.run_name = [name for name in run_names if name is not None][0]
        self.run_dir = f"{self.configs.base_dir}/{self.run_name}"
        self.checkpoint_dir = f"{self.run_dir}/checkpoints"
        self.prediction_dir = f"{self.run_dir}/predictions"
        self.structure_dir = f"{self.run_dir}/structures"
        self.dump_dir = f"{self.run_dir}/dumps"
        self.error_dir = f"{self.run_dir}/errors"

        if DIST_WRAPPER.rank == 0:
            os.makedirs(self.run_dir)
            os.makedirs(self.checkpoint_dir)
            os.makedirs(self.prediction_dir)
            os.makedirs(self.structure_dir)
            os.makedirs(self.dump_dir)
            os.makedirs(self.error_dir)
            save_config(
                self.configs,
                os.path.join(self.configs.base_dir, self.run_name, "config.yaml"),
            )

        self.print(
            f"Using run name: {self.run_name}, run dir: {self.run_dir}, "
            f"checkpoint_dir: {self.checkpoint_dir}, "
            f"prediction_dir: {self.prediction_dir}, "
            f"structure_dir: {self.structure_dir}, "
            f"error_dir: {self.error_dir}"
        )

    def init_log(self) -> None:
        """
        Initialize logging and metrics aggregation.
        """
        if self.configs.use_wandb and DIST_WRAPPER.rank == 0:
            wandb.init(
                project=self.configs.project,
                name=self.run_name,
                config=vars(self.configs),
                id=self.configs.wandb_id or None,
            )
        self.train_metric_wrapper = SimpleMetricAggregator(["avg"])

    def init_env(self) -> None:
        """
        Initialize the PyTorch and CUDA environment for training.
        Sets up distributed training if world_size > 1 and sets random seeds.
        """
        logging.info(
            f"Distributed environment: world size: {DIST_WRAPPER.world_size}, "
            f"global rank: {DIST_WRAPPER.rank}, local rank: {DIST_WRAPPER.local_rank}"
        )
        self.use_cuda = torch.cuda.device_count() > 0
        if self.use_cuda:
            self.device = torch.device(f"cuda:{DIST_WRAPPER.local_rank}")
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            all_gpu_ids = ",".join(str(x) for x in range(torch.cuda.device_count()))
            devices = os.getenv("CUDA_VISIBLE_DEVICES", all_gpu_ids)
            logging.info(
                f"LOCAL_RANK: {DIST_WRAPPER.local_rank} - CUDA_VISIBLE_DEVICES: [{devices}]"
            )
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        if DIST_WRAPPER.world_size > 1:
            timeout_seconds = int(os.environ.get("NCCL_TIMEOUT_SECOND", 600))
            dist.init_process_group(
                backend="nccl", timeout=datetime.timedelta(seconds=timeout_seconds)
            )

        if not self.configs.deterministic_seed:
            # use rank-specific seed
            hash_string = f"({self.configs.seed},{DIST_WRAPPER.rank},init_seed)"
            rank_seed = int(hashlib.sha256(hash_string.encode("utf8")).hexdigest(), 16)
            rank_seed = rank_seed % (2**32)
        else:
            rank_seed = self.configs.seed

        seed_everything(
            seed=rank_seed,
            deterministic=self.configs.deterministic,
        )  # Different DDP processes get different seeds

        if self.configs.triangle_attention == "deepspeed":
            env = os.getenv("CUTLASS_PATH", None)
            print(f"env: {env}")
            assert env is not None, (
                "If use deepspeed (ds4sci), set CUTLASS_PATH env as per "
                "https://www.deepspeed.ai/tutorials/ds4sci_evoformerattention/"
            )
        logging.info("Finished environment initialization.")

    def init_loss(self) -> None:
        """
        Initialize loss functions, metrics, and permutation utilities.
        """
        self.loss = ProtenixLoss(self.configs)
        self.symmetric_permutation = SymmetricPermutation(
            self.configs, error_dir=self.error_dir
        )
        self.lddt_metrics = LDDTMetrics(self.configs)

    def init_model(self) -> None:
        """
        Initialize the Protenix model, optimizer, and exponential moving average (EMA).
        Sets up DistributedDataParallel (DDP) if multiple GPUs are used.
        """
        self.raw_model = Protenix(self.configs).to(self.device)
        self.use_ddp = False
        if DIST_WRAPPER.world_size > 1:
            self.print("Using DistributedDataParallel (DDP)")
            self.use_ddp = True
            # Fix DDP/checkpoint compatibility:
            # https://discuss.pytorch.org/t/ddp-and-gradient-checkpointing/132244
            self.model = DDP(
                self.raw_model,
                find_unused_parameters=self.configs.find_unused_parameters,
                device_ids=[DIST_WRAPPER.local_rank],
                output_device=DIST_WRAPPER.local_rank,
                static_graph=True,
            )
        else:
            self.model = self.raw_model

        def count_parameters(model: torch.nn.Module) -> float:
            """Count total parameters in millions."""
            total_params = sum(p.numel() for p in model.parameters())
            return total_params / 1e6

        self.print(f"Model Parameters: {count_parameters(self.model):.2f}M")
        if self.configs.get("ema_decay", -1) > 0:
            assert self.configs.ema_decay < 1
            self.ema_wrapper = EMAWrapper(
                self.model,
                self.configs.ema_decay,
                self.configs.ema_mutable_param_keywords,
            )
            self.ema_wrapper.register()

        torch.cuda.empty_cache()
        self.optimizer = get_optimizer(
            self.configs,
            self.model,
            param_names=self.configs.get("finetune_params_with_substring", [""]),
        )
        self.init_scheduler()

    def init_scheduler(self, **kwargs: Any) -> None:
        """
        Initialize the learning rate scheduler.
        Supports both standard and fine-tuning schedulers.

        Args:
            **kwargs: Additional arguments passed to the scheduler.
        """
        # init finetune lr scheduler if available
        finetune_params = self.configs.get("finetune_params_with_substring", [""])
        is_finetune = len(finetune_params[0]) > 0

        if is_finetune:
            self.lr_scheduler = FinetuneLRScheduler(
                self.optimizer,
                self.configs,
                self.configs.finetune,
                **kwargs,
            )
        else:
            self.lr_scheduler = get_lr_scheduler(self.configs, self.optimizer, **kwargs)

    def init_data(self) -> None:
        """
        Initialize training and test dataloaders.
        """
        self.train_dl, self.test_dls = get_dataloaders(
            self.configs,
            DIST_WRAPPER.world_size,
            seed=self.configs.seed,
            error_dir=self.error_dir,
        )

    def save_checkpoint(self, ema_suffix: str = "") -> None:
        """
        Save the current model state, optimizer, and scheduler to a checkpoint file.

        Args:
            ema_suffix (str): Optional suffix for EMA checkpoints.
        """
        if DIST_WRAPPER.rank == 0:
            path = f"{self.checkpoint_dir}/{self.step}{ema_suffix}.pt"
            checkpoint = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": (
                    self.lr_scheduler.state_dict()
                    if self.lr_scheduler is not None
                    else None
                ),
                "step": self.step,
            }
            torch.save(checkpoint, path)
            self.print(f"Saved checkpoint to {path}")

    def try_load_checkpoint(self) -> None:
        """
        Attempt to load model and training state from specified checkpoint paths.
        Loads both standard and EMA checkpoints if configured.
        """

        def _load_checkpoint(
            checkpoint_path: str,
            load_params_only: bool,
            skip_load_optimizer: bool = False,
            skip_load_step: bool = False,
            skip_load_scheduler: bool = False,
            load_step_for_scheduler: bool = True,
        ) -> None:
            """
            Internal helper to load a single checkpoint.

            Args:
                checkpoint_path (str): Path to the checkpoint file.
                load_params_only (bool): If True, only load model parameters.
                skip_load_optimizer (bool): If True, do not load optimizer state.
                skip_load_step (bool): If True, do not load training step.
                skip_load_scheduler (bool): If True, do not load scheduler state.
                load_step_for_scheduler (bool): If True, re-initialize scheduler with loaded step.
            """
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(
                    f"Given checkpoint path does not exist: [{checkpoint_path}]"
                )
            self.print(
                f"Loading from {checkpoint_path}, strict: {self.configs.load_strict}"
            )
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )
            sample_key = list(checkpoint["model"].keys())[0]
            self.print(f"Sampled key: {sample_key}")
            # Align the checkpoint's key namespace with the model's. DDP wraps the
            # model before this runs, so self.model.state_dict() is "module."-prefixed
            # while a checkpoint saved from a single-GPU run is not (and vice versa).
            # With load_strict=False a mismatch here silently loads NOTHING.
            ckpt_is_ddp = sample_key.startswith("module.")
            if ckpt_is_ddp and not self.use_ddp:
                checkpoint["model"] = {
                    k[len("module.") :]: v for k, v in checkpoint["model"].items()
                }
            elif self.use_ddp and not ckpt_is_ddp:
                checkpoint["model"] = {
                    f"module.{k}": v for k, v in checkpoint["model"].items()
                }

            incompatible = self.model.load_state_dict(
                state_dict=checkpoint["model"],
                strict=self.configs.load_strict,
            )
            n_missing = len(incompatible.missing_keys)
            n_unexpected = len(incompatible.unexpected_keys)
            if n_missing or n_unexpected:
                # Expected for the codesign head on top of the base checkpoint; a
                # count in the thousands means the namespaces did not line up.
                self.print(
                    f"load_state_dict: {n_missing} missing / {n_unexpected} unexpected keys, "
                    f"e.g. missing={incompatible.missing_keys[:3]}, "
                    f"unexpected={incompatible.unexpected_keys[:3]}"
                )
            n_loaded = len(self.model.state_dict()) - n_missing
            assert n_loaded > 0, (
                f"No parameters were loaded from {checkpoint_path} "
                f"({n_unexpected} unexpected keys) -- key namespaces do not match."
            )
            if not load_params_only:
                if not skip_load_optimizer:
                    self.print("Loading optimizer state")
                    self.optimizer.load_state_dict(checkpoint["optimizer"])
                if not skip_load_step:
                    self.print("Loading checkpoint step")
                    self.step = checkpoint["step"] + 1
                    self.start_step = self.step
                    self.global_step = self.step * self.iters_to_accumulate
                if not skip_load_scheduler:
                    self.print("Loading scheduler state")
                    self.lr_scheduler.load_state_dict(checkpoint["scheduler"])
                elif load_step_for_scheduler:
                    assert (
                        not skip_load_step
                    ), "If load_step_for_scheduler is True, you must load step first"
                    # Reinitialize LR scheduler using the updated optimizer and step
                    self.init_scheduler(last_epoch=self.step - 1)

            self.print(f"Finish loading checkpoint, current step: {self.step}")

        # Load EMA model parameters if configured
        if self.configs.load_ema_checkpoint_path:
            _load_checkpoint(
                self.configs.load_ema_checkpoint_path,
                load_params_only=True,
            )
            self.ema_wrapper.register()

        # Load standard model checkpoint if configured
        if self.configs.load_checkpoint_path:
            _load_checkpoint(
                self.configs.load_checkpoint_path,
                self.configs.load_params_only,
                skip_load_optimizer=self.configs.skip_load_optimizer,
                skip_load_scheduler=self.configs.skip_load_scheduler,
                skip_load_step=self.configs.skip_load_step,
                load_step_for_scheduler=self.configs.load_step_for_scheduler,
            )

    def print(self, msg: str) -> None:
        """
        Print message to log only on the master rank (rank 0).

        Args:
            msg (str): The message to log.
        """
        if DIST_WRAPPER.rank == 0:
            logging.info(msg)

    def model_forward(
        self, batch: Dict[str, Any], mode: str = "train"
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Perform a forward pass of the model.

        Args:
            batch (Dict[str, Any]): Input batch containing features and labels.
            mode (str): Execution mode, either 'train' or 'eval'.

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: Updated batch with
                predictions and logging dictionary.
        """
        assert mode in ["train", "eval"]
        batch["pred_dict"], batch["label_dict"], log_dict = self.model(
            input_feature_dict=batch["input_feature_dict"],
            label_dict=batch["label_dict"],
            label_full_dict=batch["label_full_dict"],
            mode=mode,
            current_step=self.step if mode == "train" else None,
            symmetric_permutation=self.symmetric_permutation,
            mc_dropout_apply_rate=(
                0 if mode == "train" else self.configs.mc_dropout_apply_rate
            ),
        )
        return batch, log_dict

    def get_loss(
        self, batch: Dict[str, Any], mode: str = "train"
    ) -> Tuple[torch.Tensor, Dict[str, Any], Dict[str, Any]]:
        """
        Compute the loss for a given batch.

        Args:
            batch (Dict[str, Any]): Batch containing features, labels, and predictions.
            mode (str): Execution mode, either 'train' or 'eval'.

        Returns:
            Tuple[torch.Tensor, Dict[str, Any], Dict[str, Any]]:
                - Total loss tensor.
                - Dictionary containing individual loss components.
                - Updated batch dictionary.
        """
        assert mode in ["train", "eval"]

        loss, loss_dict = autocasting_disable_decorator(self.configs.skip_amp.loss)(
            self.loss
        )(
            feat_dict=batch["input_feature_dict"],
            pred_dict=batch["pred_dict"],
            label_dict=batch["label_dict"],
            mode=mode,
        )
        return loss, loss_dict, batch

    @torch.no_grad()
    def get_metrics(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute evaluation metrics (e.g., lDDT) for a batch.

        Args:
            batch (Dict[str, Any]): Batch containing predictions and labels.

        Returns:
            Dict[str, Any]: Dictionary containing computed metrics.
        """
        lddt_dict = self.lddt_metrics.compute_lddt(
            batch["pred_dict"], batch["label_dict"]
        )
        return lddt_dict

    @torch.no_grad()
    def aggregate_metrics(
        self, lddt_dict: Dict[str, Any], batch: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Aggregate metrics across the batch.

        Args:
            lddt_dict (Dict[str, Any]): Dictionary of per-structure metrics.
            batch (Dict[str, Any]): Batch containing confidence summaries.

        Returns:
            Dict[str, Any]: Aggregated metrics.
        """
        simple_metrics, _ = self.lddt_metrics.aggregate_lddt(
            lddt_dict, batch["pred_dict"]["summary_confidence"]
        )
        return simple_metrics

    @torch.no_grad()
    def evaluate(self, mode: str = "eval") -> None:
        """
        Evaluate the model on all test datasets.
        Handles both standard and EMA model evaluation if applicable.

        Args:
            mode (str): Execution mode, typically 'eval'.
        """
        if not self.configs.eval_ema_only:
            self._evaluate(mode=mode)
        if hasattr(self, "ema_wrapper"):
            self.ema_wrapper.apply_shadow()
            self._evaluate(ema_suffix=f"ema{self.ema_wrapper.decay}_", mode=mode)
            self.ema_wrapper.restore()

    # Antibody region bookkeeping for the codesign eval metrics.
    # chain_type: 1=Heavy, 2=Light, 3=Antigen. region_type: Chothia fr1..fr4 = 1,3,5,7
    # and cdr1..cdr3 = 2,4,6 (see protenix/data/antibody_cdr.py).
    _CDR_REGIONS = {
        "H1": (1, 2), "H2": (1, 4), "H3": (1, 6),
        "L1": (2, 2), "L2": (2, 4), "L3": (2, 6),
    }
    _FRAMEWORK_REGIONS = (1, 3, 5, 7)
    # Loop-RMSD: number of stem/anchor residues trimmed from each end of CDR-H3 so
    # only the central (apex) loop residues remain. ASSUMPTION -- adjust to match the
    # loop definition of the numbering scheme being reported.
    _H3_LOOP_STEM = 2

    def _log_structure_design(
        self,
        batch: Dict[str, Any],
        simple_metrics: Dict[str, Any],
        test_name: str,
        ema_suffix: str = "",
    ) -> None:
        """Write every predicted structure plus the region metadata for Benchmark 2.

        No metrics are computed here ``scripts/cdr_rmsd_benchmark.py`` does the
        PyRosetta pack/relax, the framework-Ca superposition and then outputs per-CDR RMSD.

        """
        try:
            atom_array = batch.get("cropped_atom_array")
            pred = batch.get("pred_dict") or {}
            label = batch.get("label_dict") or {}
            pred_coord, gt_coord = pred.get("coordinate"), label.get("coordinate")
            ifd = batch.get("input_feature_dict") or {}
            if atom_array is None or pred_coord is None or gt_coord is None:
                return
            if "region_type" not in ifd or "chain_type" not in ifd:
                return

            import numpy as np

            n_atom = gt_coord.shape[-2]
            all_p = pred_coord.reshape(-1, n_atom, 3).float()  # one row per sample
            g = gt_coord.reshape(-1, n_atom, 3)[0].float()
            gm = label.get("coordinate_mask")
            resolved = (
                gm.reshape(-1, n_atom)[0].bool()
                if gm is not None
                else torch.ones(n_atom, dtype=torch.bool, device=all_p.device)
            )
            a2t = ifd["atom_to_token_idx"].reshape(-1).long()
            region = ifd["region_type"].reshape(-1).long()[a2t]  # per atom
            chain = ifd["chain_type"].reshape(-1).long()[a2t]
            is_ca = torch.as_tensor(
                np.asarray(atom_array.atom_name) == "CA", device=all_p.device
            )
            if is_ca.shape[0] != n_atom:
                return

            # No RMSD is computed here. Benchmark 2 is a separate offline step:
            # this pass only writes every predicted structure (already conditioned by
            # replacement sampling) plus the region metadata, and
            # scripts/cdr_rmsd_benchmark.py does the PyRosetta pack/relax, the
            # framework-Ca superposition and the per-CDR RMSD.
            # NOT rank-gated: under DDP the test set is split across ranks, so
            # gating here would silently dump only rank 0's share and the offline
            # CDR RMSD benchmark would score a half-sized set without complaining.
            # Every file below is named per-PDB, so ranks write disjoint paths.
            import copy
            import json

            from biotite.structure.io.pdbx import CIFFile, set_structure

            pid = batch["basic"]["pdb_id"]
            out_dir = os.path.join(
                self.structure_dir, f"{test_name}_step{self.step}_{ema_suffix or 'raw'}"
            )
            os.makedirs(out_dir, exist_ok=True)

            # One CIF per diffusion sample, in the model's own frame (no alignment --
            # the benchmark script superposes after relax).
            for row in range(all_p.shape[0]):
                pred_array = copy.deepcopy(atom_array)
                pred_array.coord = all_p[row].detach().cpu().numpy()
                pred_array.bonds = None  # biotite bond export is not needed here
                cif = CIFFile()
                set_structure(cif, pred_array)
                cif.write(os.path.join(out_dir, f"{pid}_sample{row}.cif"))

            # Ground truth in the identical atom ordering, so the benchmark script
            # never has to re-derive the correspondence.
            gt_array = copy.deepcopy(atom_array)
            gt_array.coord = g.detach().cpu().numpy()
            gt_array.bonds = None
            cif = CIFFile()
            set_structure(cif, gt_array)
            cif.write(os.path.join(out_dir, f"{pid}_native.cif"))

            # Region metadata: which residues are framework (align on these) and which
            # are CDRs (score these), keyed by chain/res_id so it survives PyRosetta.
            ca_idx = is_ca.nonzero().flatten().tolist()
            centre = [
                {
                    "chain": str(atom_array.chain_id[i]),
                    "res_id": int(atom_array.res_id[i]),
                    "res_name": str(atom_array.res_name[i]),
                    "chain_type": int(chain[i]),      # 1=H, 2=L, 3=antigen
                    "region_type": int(region[i]),    # Chothia fr1..fr4=1,3,5,7; cdr1..3=2,4,6
                    "resolved": bool(resolved[i]),
                }
                for i in ca_idx
            ]
            with open(os.path.join(out_dir, f"{pid}_regions.json"), "w") as fh:
                json.dump(
                    {
                        "pdb_id": pid,
                        "n_samples": int(all_p.shape[0]),
                        "cdr_regions": self._CDR_REGIONS,
                        "framework_regions": list(self._FRAMEWORK_REGIONS),
                        "h3_loop_stem": self._H3_LOOP_STEM,
                        "residues": centre,
                    },
                    fh,
                )
        except Exception as e:  # noqa: BLE001 - never break an eval pass
            logging.warning("Structure-design logging failed: %s", e)

    def _log_sequence_design(
        self,
        batch: Dict[str, Any],
        simple_metrics: Dict[str, Any],
        test_name: str,
        ema_suffix: str = "",
    ) -> None:
        """Add per-CDR AAR metrics to training eval.

        Only active when the model returned
        a decoded sequence (``pred_dict["sequence"]``) together with the ground truth
        and the design mask. Adds ``aar/*`` entries to ``simple_metrics`` (so they
        flow into the normal aggregator and wandb) and appends the design to a
        per-eval FASTA + ``.seq`` TSV under ``prediction_dir``, matching MFDesign's
        writer output (Rank/Sequence/Total/H/L/PerCDR).

        Any failure here is logged and swallowed: a metrics/IO problem must never
        abort an evaluation pass.
        """
        try:
            pred = batch.get("pred_dict") or {}
            seq_pred = pred.get("sequence")
            seq_gt = pred.get("seq_gt")
            cdr_mask = pred.get("cdr_mask")
            if seq_pred is None or seq_gt is None or cdr_mask is None:
                return
            n_token = seq_gt.reshape(-1, seq_gt.shape[-1]).shape[-1]
            if not seq_pred.is_floating_point():
                # Eval emits decoded token ids, one row per diffusion sample.
                all_pred = seq_pred.reshape(-1, n_token)
            else:  # logits (shouldn't happen in eval, but stay defensive)
                all_pred = seq_pred.reshape(-1, n_token, seq_pred.shape[-1]).argmax(-1)
            gt_ids = seq_gt.reshape(-1, seq_gt.shape[-1])[0]
            cdr = cdr_mask.reshape(-1, cdr_mask.shape[-1])[0].bool()
            if int(cdr.sum()) == 0:
                return

            from protenix.metrics.sequence_recovery import calculate_aar
            from protenix.model.sequence_decode import token_ids_to_letters

            # With N_sample > 1 the model emits several independent designs. Report the
            # MEAN recovery over them, not the best: taking the best would select on
            # the very metric being reported and rise with N_sample for free.
            per_sample = [
                calculate_aar(all_pred[row], gt_ids, cdr)
                for row in range(all_pred.shape[0])
            ]
            n_s = len(per_sample)
            n_cdr = min(len(a["per_cdr"]) for a in per_sample)
            aar = {
                "total": sum(a["total"] for a in per_sample) / n_s,
                "heavy": sum(a["heavy"] for a in per_sample) / n_s,
                "light": sum(a["light"] for a in per_sample) / n_s,
                "per_cdr": [
                    sum(a["per_cdr"][i] for a in per_sample) / n_s
                    for i in range(n_cdr)
                ],
                "n_designed": per_sample[0]["n_designed"],
            }
            pred_ids = all_pred[0]  # representative design for the FASTA/.seq dump
            simple_metrics["aar/n_samples"] = float(n_s)
            simple_metrics["aar/total"] = aar["total"]
            simple_metrics["aar/heavy"] = aar["heavy"]
            simple_metrics["aar/light"] = aar["light"]
            # Name the loops when the segment count matches the canonical layout
            # (MFDesign's positional convention: first three heavy, last three light).
            per_cdr = aar["per_cdr"]
            if len(per_cdr) == 6:
                names = ["H1", "H2", "H3", "L1", "L2", "L3"]
            elif len(per_cdr) == 3:
                names = ["H1", "H2", "H3"]
            else:
                names = [f"cdr{i + 1}" for i in range(len(per_cdr))]
            for name, value in zip(names, per_cdr):
                simple_metrics[f"aar/{name}"] = value

            # NOT rank-gated, for the same reason as the structure dump: each rank
            # sees a disjoint shard of the test set. These two are shared *append*
            # targets though, so they get a per-rank suffix -- concurrent "a" writes
            # can interleave, and the truncating "w" below would race one rank's
            # header against another's accumulated rows. Concatenate the shards
            # afterwards: cat <tag>_rank*.fasta > <tag>.fasta
            pid = batch["basic"]["pdb_id"]
            tag = f"{test_name}_step{self.step}_{ema_suffix or 'raw'}"
            shard = (
                f"_rank{DIST_WRAPPER.rank}" if DIST_WRAPPER.world_size > 1 else ""
            )
            letters = token_ids_to_letters(pred_ids)
            per_cdr_str = "\t".join(f"{v:.3f}" for v in per_cdr)
            fasta_path = os.path.join(self.prediction_dir, f"{tag}{shard}.fasta")
            seq_path = os.path.join(self.prediction_dir, f"{tag}{shard}.seq")
            if not os.path.exists(seq_path):
                with open(seq_path, "w") as f:
                    f.write("PDB\tSequence\tTotal\tH\tL\tN_designed\tPerCDR\n")
            with open(fasta_path, "a") as f:
                f.write(f">{pid}\n{letters}\n")
            with open(seq_path, "a") as f:
                f.write(
                    f"{pid}\t{letters}\t{aar['total']:.3f}\t{aar['heavy']:.3f}\t"
                    f"{aar['light']:.3f}\t{aar['n_designed']}\t{per_cdr_str}\n"
                )
        except Exception as e:  # noqa: BLE001 - never break an eval pass
            logging.warning("Sequence-design logging failed: %s", e)

    @torch.no_grad()
    def _evaluate(self, ema_suffix: str = "", mode: str = "eval") -> None:
        """
        Internal evaluation loop for a specific model state.

        Args:
            ema_suffix (str): Suffix for metric names if using EMA.
            mode (str): Execution mode.
        """
        # Init Metric Aggregator
        simple_metric_wrapper = SimpleMetricAggregator(["avg"])
        eval_precision = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[self.configs.dtype]
        enable_amp = (
            torch.autocast(device_type="cuda", dtype=eval_precision)
            if torch.cuda.is_available()
            else nullcontext()
        )
        self.model.eval()

        for test_name, test_dl in self.test_dls.items():
            self.print(f"Testing on {test_name}")
            evaluated_pids = []
            total_batch_num = len(test_dl)
            for index, batch in enumerate(tqdm(test_dl)):
                batch = to_device(batch, self.device)
                pid = batch["basic"]["pdb_id"]

                if index + 1 == total_batch_num and DIST_WRAPPER.world_size > 1:
                    # Gather all pids across ranks to avoid duplicate evaluation
                    # when drop_last=False
                    all_data_ids = DIST_WRAPPER.all_gather_object(evaluated_pids)
                    dedup_ids = set(sum(all_data_ids, []))
                    if pid in dedup_ids:
                        print(
                            f"Rank {DIST_WRAPPER.rank}: Dropping data_id {pid} "
                            "as it is already evaluated."
                        )
                        break
                evaluated_pids.append(pid)

                # A single OOM (or any per-structure failure) must not abort the
                # whole pass -- a full-test-set eval at large N_sample runs for
                # many hours, and the uncropped complexes vary hugely in size.
                # Skip the offending structure and carry on; it is simply absent
                # from the aggregate, and the skip is logged loudly.
                try:
                    simple_metrics = {}
                    with enable_amp:
                        # Model forward
                        batch, _ = self.model_forward(batch, mode=mode)
                        # Loss forward
                        _, loss_dict, batch = self.get_loss(batch, mode="eval")
                        # lDDT metrics
                        lddt_dict = self.get_metrics(batch)
                        lddt_metrics = self.aggregate_metrics(lddt_dict, batch)
                        simple_metrics.update(
                            {k: v for k, v in lddt_metrics.items() if "diff" not in k}
                        )
                        simple_metrics.update(loss_dict)

                    # Both metrics come from this single design pass, matching the
                    # reference protocol (Alg. S3 emits one sequence + structure pair):
                    #   AAR  - designed sequence vs native sequence
                    #   RMSD - designed CDR backbone vs native, framework-aligned
                    # RMSD is Ca-only, and every amino acid has a Ca, so comparing the
                    # designed loop's trace against the native one is well defined even
                    # where the designed residues differ.
                    self._log_sequence_design(
                        batch, simple_metrics, test_name, ema_suffix
                    )
                    self._log_structure_design(
                        batch, simple_metrics, test_name, ema_suffix
                    )

                    # Update metric aggregator
                    for key, value in simple_metrics.items():
                        simple_metric_wrapper.add(
                            f"{ema_suffix}{key}", value, namespace=test_name
                        )
                except Exception as exc:  # noqa: BLE001 - report and keep going
                    logging.error(
                        f"Rank {DIST_WRAPPER.rank}: SKIPPING {pid} in {test_name} "
                        f"-- {type(exc).__name__}: {exc}"
                    )
                    simple_metrics = None

                del batch, simple_metrics
                # Every iteration, not every 5: at large N_sample the peak varies
                # enormously with complex size, and a stale cached block is what
                # turns a merely large next complex into an OOM.
                torch.cuda.empty_cache()

            metrics = simple_metric_wrapper.calc()
            self.print(f"Step {self.step}, eval {test_name}: {metrics}")
            if self.configs.use_wandb and DIST_WRAPPER.rank == 0:
                wandb.log(metrics, step=self.step)

    def update(self) -> None:
        """
        Apply gradient clipping to model parameters.
        """
        if self.configs.grad_clip_norm != 0.0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.configs.grad_clip_norm
            )

    def train_step(self, batch: Dict[str, Any]) -> None:
        """
        Perform a single training step including forward, backward, and optimization.

        Args:
            batch (Dict[str, Any]): Input batch for training.
        """
        self.model.train()
        train_precision = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[self.configs.dtype]
        enable_amp = (
            torch.autocast(
                device_type="cuda", dtype=train_precision, cache_enabled=False
            )
            if torch.cuda.is_available()
            else nullcontext()
        )

        scaler = torch.GradScaler(
            device="cuda" if torch.cuda.is_available() else "cpu",
            enabled=(self.configs.dtype == "float16"),
        )

        with enable_amp:
            batch, _ = self.model_forward(batch, mode="train")
            loss, loss_dict, _ = self.get_loss(batch, mode="train")

        if self.configs.dtype in ["bf16", "fp32"]:
            if is_loss_nan_check(loss):
                self.print(f"Skip iteration with NaN loss at step {self.step}")
                loss = torch.tensor(0.0, device=loss.device, requires_grad=True)
        scaler.scale(loss / self.iters_to_accumulate).backward()

        # Global training step used for accumulation logic
        if (self.global_step + 1) % self.iters_to_accumulate == 0:
            # Unscale gradients before clipping
            scaler.unscale_(self.optimizer)
            # Apply gradient clipping
            self.update()
            # Optimizer and scaler step
            scaler.step(self.optimizer)
            scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.lr_scheduler.step()

        for key, value in loss_dict.items():
            if "loss" not in key:
                continue
            self.train_metric_wrapper.add(key, value, namespace="train")
        torch.cuda.empty_cache()

    def progress_bar(self, desc: str = "") -> None:
        """
        Update the training progress bar on the master rank.

        Args:
            desc (str): Description to display on the progress bar.
        """
        if DIST_WRAPPER.rank != 0:
            return
        eval_steps = self.configs.eval_interval * self.iters_to_accumulate
        if self.global_step % eval_steps == 0 or (not hasattr(self, "_ipbar")):
            # Start a new progress bar
            self._pbar = tqdm(
                range(
                    self.global_step % eval_steps,
                    eval_steps,
                )
            )
            self._ipbar = iter(self._pbar)

        try:
            step = next(self._ipbar)
            self._pbar.set_description(
                f"[step {self.step}: {step}/{eval_steps}] {desc}"
            )
        except StopIteration:
            # If the iterator is exhausted, just return
            pass

    def run(self) -> None:
        """
        Main entry point for the AF3Trainer.
        Handles the complete training cycle including evaluation, logging, and checkpointing.
        """
        if self.configs.eval_only or self.configs.eval_first:
            self.evaluate()
            if self.configs.eval_only:
                return
        use_ema = hasattr(self, "ema_wrapper")
        self.print(f"Using EMA: {use_ema}")

        while True:
            for batch in self.train_dl:
                is_update_step = (self.global_step + 1) % self.iters_to_accumulate == 0
                is_last_step = (self.step + 1) == self.configs.max_steps
                step_need_log = (self.step + 1) % self.configs.log_interval == 0

                step_need_eval = (
                    self.configs.eval_interval > 0
                    and (self.step + 1) % self.configs.eval_interval == 0
                )
                step_need_save = (
                    self.configs.checkpoint_interval > 0
                    and (self.step + 1) % self.configs.checkpoint_interval == 0
                )

                is_last_step &= is_update_step
                step_need_log &= is_update_step
                step_need_eval &= is_update_step
                step_need_save &= is_update_step

                batch = to_device(batch, self.device)
                self.progress_bar()
                self.train_step(batch)
                if use_ema and is_update_step:
                    self.ema_wrapper.update()

                if step_need_log or is_last_step:
                    metrics = self.train_metric_wrapper.calc()
                    self.print(f"Step {self.step} train metrics: {metrics}")
                    last_lr = self.lr_scheduler.get_last_lr()
                    if DIST_WRAPPER.rank == 0:
                        if self.configs.use_wandb:
                            lr_dict = {"train/lr": last_lr[0]}
                            for group_i, group_lr in enumerate(last_lr):
                                lr_dict[f"train/group{group_i}_lr"] = group_lr
                            wandb.log(lr_dict, step=self.step)
                        self.print(f"Step {self.step}, learning rate: {last_lr}")
                        if self.configs.use_wandb:
                            wandb.log(metrics, step=self.step)

                if step_need_save or is_last_step:
                    self.save_checkpoint()
                    if use_ema:
                        self.ema_wrapper.apply_shadow()
                        self.save_checkpoint(
                            ema_suffix=f"_ema_{self.ema_wrapper.decay}"
                        )
                        self.ema_wrapper.restore()

                if step_need_eval or is_last_step:
                    self.evaluate()

                self.global_step += 1
                if self.global_step % self.iters_to_accumulate == 0:
                    self.step += 1

                if self.step >= self.configs.max_steps:
                    self.print(f"Finished training after {self.step} steps")
                    break
            if self.step >= self.configs.max_steps:
                break


def main() -> None:
    """
    Parse configurations and start the training process.
    """
    log_format = (
        "%(asctime)s,%(msecs)-3d %(levelname)-8s "
        "[%(filename)s:%(lineno)s %(funcName)s] %(message)s"
    )
    logging.basicConfig(
        format=log_format,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode="w",
    )
    configs_base["triangle_attention"] = os.environ.get(
        "TRIANGLE_ATTENTION", "cuequivariance"
    )
    configs_base["triangle_multiplicative"] = os.environ.get(
        "TRIANGLE_MULTIPLICATIVE", "cuequivariance"
    )
    arg_str = parse_sys_args()
    configs = {**configs_base, **{"data": data_configs}}
    # 1. First pass to get model_name
    configs = parse_configs(
        configs,
        arg_str=arg_str,
        fill_required_with_null=True,
    )
    model_name = configs.model_name

    # 2. Get model specifics and merge into base defaults
    base_configs = {**configs_base, **{"data": data_configs}}
    model_specfics_configs = model_configs[model_name]

    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, Mapping) and k in d and isinstance(d[k], Mapping):
                deep_update(d[k], v)
            else:
                d[k] = v
        return d

    deep_update(base_configs, model_specfics_configs)

    # 3. Second pass to apply sys_args with higher priority
    configs = parse_configs(
        configs=base_configs,
        arg_str=arg_str,
        fill_required_with_null=True,
    )
    logging.info(
        f"Using params for model {model_name}: "
        f"cycle={configs.model.N_cycle}, step={configs.sample_diffusion.N_step}"
    )
    print(f"Run Name: {configs.run_name}")
    trainer = AF3Trainer(configs)
    trainer.run()


if __name__ == "__main__":
    main()
