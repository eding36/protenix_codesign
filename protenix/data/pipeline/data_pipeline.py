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

import os
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Union

import biotite.structure.io as strucio
import numpy as np
import pandas as pd
import torch
from biotite.structure import AtomArray

from protenix.data.antibody_cdr import (
    add_chain_and_region_types_to_token_array,
    add_is_cdr_residue_to_token_array,
    resolve_sabdab_roles,
    strip_cdr_side_chains,
)
from protenix.data.core.parser import (
    AddAtomArrayAnnot,
    DistillationMMCIFParser,
    MMCIFParser,
    RecentPDB_MMCIFParser,
)
from protenix.data.msa.msa_featurizer import MSAFeaturizer
from protenix.data.template.template_featurizer import TemplateFeaturizer
from protenix.data.tokenizer import AtomArrayTokenizer, TokenArray
from protenix.utils.cropping import CropData
from protenix.utils.file_io import load_gzip_pickle
from protenix.utils.logger import get_logger

logger = get_logger(__name__)

torch.multiprocessing.set_sharing_strategy("file_system")


class DataPipeline(object):
    """
    DataPipeline class provides static methods to handle various data processing tasks related to bioassembly structures.
    """

    @staticmethod
    def get_data_from_mmcif(
        mmcif: Union[str, Path],
        pdb_cluster_file: Union[str, Path, None] = None,
        dataset: str = "WeightedPDB",
        strip_antibody_cdr: bool = False,
        sabdab_roles: Union[dict, None] = None,
        skip_assembly_expansion: bool = False,
        assembly_id: str = "1",
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        Get raw data from mmcif with tokenizer and a list of chains and interfaces for sampling.

        Args:
            mmcif (Union[str, Path]): The raw mmcif file.
            pdb_cluster_file (Union[str, Path, None], optional): Cluster info txt file. Defaults to None.
            dataset (str, optional): The dataset type, either "WeightedPDB" or "Distillation". Defaults to "WeightedPDB".
            interface_radius (float, optional): The radius of the interface. Defaults to 5.
            strip_antibody_cdr (bool, optional): If True, remove side-chain atoms of
                antibody variable-domain CDR residues (keeping backbone + CB) to
                prevent atom-count/reference-conformer leakage of the design target,
                add a per-atom ``is_cdr`` annotation on the AtomArray and a per-token
                ``is_cdr_residue`` annotation on the TokenArray. Defaults to False.
            sabdab_roles (Union[dict, None] = None, optional): Heavy/Light/Antigen chain to author_chain_id mappings
        Returns:
            tuple[list[dict[str, Any]], dict[str, Any]]:
                sample_indices_list (list[dict[str, Any]]): The sample indices list (each one is a chain or an interface).
                bioassembly_dict (dict[str, Any]): The bioassembly dict with sequence, atom_array, and token_array.
        """
        try:
            if dataset == "WeightedPDB":
                parser = MMCIFParser(mmcif_file=mmcif)
                bioassembly_dict = parser.get_bioassembly(
                    assembly_id=assembly_id,
                    skip_assembly_expansion=skip_assembly_expansion,
                )
            elif dataset == "Distillation":
                parser = DistillationMMCIFParser(mmcif_file=mmcif)
                bioassembly_dict = parser.get_structure_dict()
            elif dataset == "RecentPDB":
                parser = RecentPDB_MMCIFParser(mmcif_file=mmcif)
                bioassembly_dict = parser.get_bioassembly(
                    assembly_id=assembly_id,
                    skip_assembly_expansion=skip_assembly_expansion,
                )
            else:
                raise NotImplementedError(
                    'Unsupported "dataset", please input either "WeightedPDB" or "Distillation".'
                )

            sample_indices_list = parser.make_indices(
                bioassembly_dict=bioassembly_dict,
                pdb_cluster_file=pdb_cluster_file,
            )
            if len(sample_indices_list) == 0:
                # empty indices and AtomArray
                return [], bioassembly_dict

            atom_array = bioassembly_dict["atom_array"]
            atom_array.set_annotation(
                "resolution", [parser.resolution] * len(atom_array)
            )

            # Collect this PDB's curated Fv (reference, CDR-masked) sequence pairs
            # from the SAbDab summary so CDR boundaries come from MFDesign's masks
            # (no abnumber numbering, which fails on ~10% of chains). Extracted
            # before stripping since strip runs ahead of role resolution below.
            summary_seqs: list[tuple[str, str]] = []
            if sabdab_roles:
                _entries = sabdab_roles.get(str(bioassembly_dict["pdb_id"]).lower())
                for _e in _entries or []:
                    if _e.get("H_seq") and _e.get("H_masked"):
                        summary_seqs.append((_e["H_seq"], _e["H_masked"]))
                    if _e.get("L_seq") and _e.get("L_masked"):
                        summary_seqs.append((_e["L_seq"], _e["L_masked"]))

            if strip_antibody_cdr:
                # Strip antibody CDR side chains (keep backbone + CB) before
                # tokenization. Only ref_pos/ref_charge/ref_mask/res_perm depend on
                # the per-residue atom set, so recompute just those; all other
                # per-atom annotations survive boolean indexing (CA/CB are kept).
                atom_array, n_removed = strip_cdr_side_chains(
                    atom_array, summary_seqs=summary_seqs or None
                )
                if n_removed > 0:
                    atom_array = AddAtomArrayAnnot.add_ref_info_and_res_perm(atom_array)
                bioassembly_dict["atom_array"] = atom_array
                bioassembly_dict["num_tokens"] = int(
                    atom_array.centre_atom_mask.sum()
                )

            tokenizer = AtomArrayTokenizer(atom_array)
            token_array = tokenizer.get_token_array()
            if strip_antibody_cdr:
                # Propagate the per-atom ``is_cdr`` flag to a per-token
                # ``is_cdr_residue`` annotation so downstream code can identify
                # design-region tokens without re-inspecting the AtomArray.
                token_array = add_is_cdr_residue_to_token_array(
                    token_array, atom_array
                )
            bioassembly_dict["msa_features"] = None
            bioassembly_dict["template_features"] = None

            bioassembly_dict["token_array"] = token_array

            # Surface antibody chain roles onto every per-sample index row so the
            # indices CSV carries H/L/antigen chain ids (the MFDesign AntibodyInfo
            # analog). Roles come from MFDesign's curated SAbDab summary CSV; when no
            # table is supplied or this PDB is absent from it, the columns are written
            # empty to keep the CSV schema stable.
            heavy_ids: list[str] = []
            light_ids: list[str] = []
            antigen_ids: list[str] = []
            if sabdab_roles:
                entries = sabdab_roles.get(str(bioassembly_dict["pdb_id"]).lower())
                if entries:
                    heavy_ids, light_ids, antigen_ids = resolve_sabdab_roles(
                        atom_array, entries
                    )
                    # Bake per-token chain_type (H/L/Ag) and region_type (Chothia
                    # Fv regions + antigen/epitope) onto the token array so the
                    # sequence model's type/region embeddings can read them after
                    # cropping. Also returns the epitope token count.
                    token_array, epitope_count = (
                        add_chain_and_region_types_to_token_array(
                            token_array,
                            atom_array,
                            heavy_ids,
                            light_ids,
                            antigen_ids,
                        )
                    )
                    bioassembly_dict["token_array"] = token_array
                    # MFDesign-style epitope gate: an antibody-antigen complex with
                    # no antigen residue within the epitope cutoff carries no usable
                    # epitope signal for codesign -- drop it (no samples emitted).
                    if antigen_ids and epitope_count == 0:
                        logger.warning(
                            "No epitope residues within cutoff for %s; skipping.",
                            bioassembly_dict["pdb_id"],
                        )
                        return [], bioassembly_dict
            for row in sample_indices_list:
                # Single-antibody assumption: first heavy/light chain. Values are
                # asym_id_int chain indices (MFDesign-style), not label chain ids.
                row["H_chain_id"] = heavy_ids[0] if heavy_ids else ""
                row["L_chain_id"] = light_ids[0] if light_ids else ""
                row["antigen_chain_ids"] = ";".join(str(a) for a in antigen_ids)

            return sample_indices_list, bioassembly_dict

        except Exception as e:
            logger.warning("Gen data failed for %s due to %s", mmcif, e)
            traceback.print_exc()
            return [], {}

    @staticmethod
    def get_label_entity_id_to_asym_id_int(atom_array: AtomArray) -> dict[str, int]:
        """
        Get a dictionary that associates each label_entity_id with its corresponding asym_id_int.

        Args:
            atom_array (AtomArray): AtomArray object

        Returns:
            dict[str, int]: label_entity_id to its asym_id_int
        """
        entity_to_asym_id = defaultdict(set)
        for atom in atom_array:
            entity_id = atom.label_entity_id
            entity_to_asym_id[entity_id].add(atom.asym_id_int)
        return entity_to_asym_id

    @staticmethod
    def get_data_bioassembly(
        bioassembly_dict_fpath: Union[str, Path],
    ) -> dict[str, Any]:
        """
        Get the bioassembly dict.

        Args:
            bioassembly_dict_fpath (Union[str, Path]): The path to the bioassembly dictionary file.

        Returns:
            dict[str, Any]: The bioassembly dict with sequence, atom_array and token_array.

        Raises:
            AssertionError: If the bioassembly dictionary file does not exist.
        """
        assert os.path.exists(
            bioassembly_dict_fpath
        ), f"File not exists {bioassembly_dict_fpath}"
        bioassembly_dict = load_gzip_pickle(bioassembly_dict_fpath)

        return bioassembly_dict

    @staticmethod
    def _get_antibody_chains(
        one_sample: pd.Series, bioassembly_dict: dict[str, Any]
    ) -> tuple[Optional[int], Optional[int]]:
        """Read the heavy/light ``asym_id_int`` chains of an antibody sample.

        The ``H_chain_id`` / ``L_chain_id`` columns (asym_id_int values, written by
        :meth:`get_data_from_mmcif` from the SAbDab roles) are the Protenix analog of
        MFDesign's ``AntibodyInfo.H_chain_id`` / ``L_chain_id``. Returns ``(None, None)``
        for non-antibody samples (columns absent or empty), so ordinary complexes fall
        through to the standard crop path unchanged.

        Args:
            one_sample (pd.Series): A row from the indices list.
            bioassembly_dict (dict[str, Any]): The bioassembly dict.

        Returns:
            tuple[Optional[int], Optional[int]]: (H_chain_id, L_chain_id) as asym_id_int,
                each ``None`` when missing or not present in the structure.
        """
        chain_ids = np.unique(bioassembly_dict["atom_array"].asym_id_int)

        def _parse(field: str) -> Optional[int]:
            raw = one_sample.get(field, "") if hasattr(one_sample, "get") else ""
            raw = str(raw).strip()
            if raw in ("", "nan", "None"):
                return None
            try:
                val = int(float(raw))
            except (TypeError, ValueError):
                return None
            return val if val in chain_ids else None

        return _parse("H_chain_id"), _parse("L_chain_id")

    @staticmethod
    def _map_ref_chain(
        one_sample: pd.Series, bioassembly_dict: dict[str, Any]
    ) -> list[int]:
        """
        Map the chain or interface chain_x_id to the reference chain asym_id.

        Args:
            one_sample (pd.Series): A dict of one chain or interface from indices list.
            bioassembly_dict (dict[str, Any]): The bioassembly dict with sequence, atom_array and token_array.

        Returns:
            list[int]: A list of asym_id_lnt of the chosen chain or interface, length 1 or 2.
        """
        atom_array = bioassembly_dict["atom_array"]
        ref_chain_indices = []
        for chain_id_field in ["chain_1_id", "chain_2_id"]:
            chain_id = one_sample[chain_id_field]
            assert np.isin(
                chain_id, np.unique(atom_array.chain_id)
            ), f"PDB {bioassembly_dict['pdb_id']} {chain_id_field}:{chain_id} not in atom_array"
            chain_asym_id = atom_array[atom_array.chain_id == chain_id].asym_id_int[0]
            ref_chain_indices.append(chain_asym_id)
            if one_sample["type"] == "chain":
                break
        return ref_chain_indices

    @staticmethod
    def get_msa_raw_features(
        bioassembly_dict: dict[str, Any],
        selected_indices: np.ndarray,
        msa_featurizer: Optional[MSAFeaturizer],
    ) -> dict[str, np.ndarray]:
        """
        Get tokenized MSA features of the bioassembly

        Args:
            bioassembly_dict (Mapping[str, Any]): The bioassembly dict with sequence, atom_array and token_array.
            selected_indices (torch.Tensor): Cropped token indices.
            msa_featurizer (MSAFeaturizer): MSAFeaturizer instance.

        Returns:
            Optional[dict[str, np.ndarray]]: The tokenized MSA features of the bioassembly.
        """
        if msa_featurizer is None:
            return {}

        entity_to_asym_id_int = dict(
            DataPipeline.get_label_entity_id_to_asym_id_int(
                bioassembly_dict["atom_array"]
            )
        )

        msa_feats = msa_featurizer(
            bioassembly_dict=bioassembly_dict,
            selected_indices=selected_indices,
            entity_to_asym_id_int=entity_to_asym_id_int,
        )

        return msa_feats

    @staticmethod
    def get_template_raw_features(
        bioassembly_dict: dict[str, Any],
        selected_indices: np.ndarray,
        template_featurizer: None,
    ) -> Optional[dict[str, np.ndarray]]:
        """
        Get tokenized template features of the bioassembly.

        Args:
            bioassembly_dict (dict[str, Any]): The bioassembly dict with sequence, atom_array and token_array.
            selected_indices (np.ndarray): Cropped token indices.
            template_featurizer (None): Placeholder for the template featurizer.

        Returns:
            Optional[dict[str, np.ndarray]]: The tokenized template features of the bioassembly,
                or None if the template featurizer is not provided.
        """
        if template_featurizer is None:
            return {}

        entity_to_asym_id_int = dict(
            DataPipeline.get_label_entity_id_to_asym_id_int(
                bioassembly_dict["atom_array"]
            )
        )

        template_feats = template_featurizer(
            bioassembly_dict=bioassembly_dict,
            selected_indices=selected_indices,
            entity_to_asym_id_int=entity_to_asym_id_int,
        )
        return template_feats

    @staticmethod
    def crop(
        one_sample: pd.Series,
        bioassembly_dict: dict[str, Any],
        crop_size: int,
        msa_featurizer: Optional[MSAFeaturizer],
        template_featurizer: Optional[TemplateFeaturizer],
        method_weights: list[float] = [0.2, 0.4, 0.4],
        contiguous_crop_complete_lig: bool = False,
        spatial_crop_complete_lig: bool = False,
        drop_last: bool = False,
        remove_metal: bool = False,
        antibody_add_antigen: bool = True,
        antibody_min_neighborhood: int = 0,
        antibody_max_neighborhood: int = 40,
        antibody_mixed_prob: float = 1.0,
    ) -> tuple[str, TokenArray, AtomArray, dict[str, Any], dict[str, Any]]:
        """
        Crop data based on the crop size and reference chain indices.

        Args:
            one_sample (pd.Series): A dict of one chain or interface from indices list.
            bioassembly_dict (dict[str, Any]): A dict of bioassembly dict with sequence, atom_array and token_array.
            crop_size (int): the crop size.
            msa_featurizer (MSAFeaturizer): Default to an empty replacement for msa featurizer.
            template_featurizer (None): Placeholder for the template featurizer.
            method_weights (list[float]): The weights corresponding to these three cropping methods:
                                          ["ContiguousCropping", "SpatialCropping", "SpatialInterfaceCropping"].
            contiguous_crop_complete_lig (bool): Whether to crop the complete ligand in ContiguousCropping method.
            spatial_crop_complete_lig (bool): Whether to crop the complete ligand in SpatialCropping method.
            drop_last (bool): Whether to drop the last fragment in ContiguousCropping.
            remove_metal (bool): Whether to remove metal atoms from the crop.
            antibody_mixed_prob (float): For antibody samples, the probability of using
                the standard weighted crop (anywhere in the complex); with probability
                ``1 - antibody_mixed_prob`` the sample uses the antibody-centered
                AntibodyCropping. Defaults to 0.0 (always AntibodyCropping).
                MFDesign stage-4 MixedCropper ``probability: 0.5`` == 0.5 here.

        Returns:
            tuple[str, TokenArray, AtomArray, dict[str, Any], dict[str, Any]]:
                crop_method (str): The crop method.
                cropped_token_array (TokenArray): TokenArray after cropping.
                cropped_atom_array (AtomArray): AtomArray after cropping.
                cropped_msa_features (dict[str, Any]): The cropped msa features.
                cropped_template_features (dict[str, Any]): The cropped template features.
        """
        if crop_size <= 0:
            # No cropping (the test config's crop_size=-1) still has to deduplicate
            # antibody assemblies: an unmasked duplicate copy of the designed H/L
            # chain leaks the CDR sequence outright. This is the branch eval takes,
            # so the dedup in the cropping path below never runs here.
            selected_indices = None
            _h, _l = DataPipeline._get_antibody_chains(
                one_sample=one_sample, bioassembly_dict=bioassembly_dict
            )
            if _h is not None or _l is not None:
                _refs = [c for c in (_h, _l) if c is not None]
                _n_tok = len(bioassembly_dict["token_array"])
                _kept = np.asarray(
                    DataPipeline.drop_duplicate_assembly_copies(
                        bioassembly_dict=bioassembly_dict,
                        selected_indices=np.arange(_n_tok),
                        ref_chain_indices=_refs,
                    )
                )
                if _kept.size < _n_tok:
                    selected_indices = torch.as_tensor(_kept, dtype=torch.long)
            # Prepare msa
            msa_features = DataPipeline.get_msa_raw_features(
                bioassembly_dict=bioassembly_dict,
                selected_indices=selected_indices,
                msa_featurizer=msa_featurizer,
            )
            # Prepare template
            template_features = DataPipeline.get_template_raw_features(
                bioassembly_dict=bioassembly_dict,
                selected_indices=selected_indices,
                template_featurizer=template_featurizer,
            )
            if selected_indices is None:
                return (
                    "no_crop",
                    bioassembly_dict["token_array"],
                    bioassembly_dict["atom_array"],
                    msa_features or {},
                    template_features or {},
                    -1,
                )
            # Duplicates were removed: materialise the reduced arrays through the
            # same path the cropper uses, so token/atom bookkeeping stays consistent.
            dedup_crop = CropData(
                crop_size=len(bioassembly_dict["token_array"]),
                ref_chain_indices=_refs,
                token_array=bioassembly_dict["token_array"],
                atom_array=bioassembly_dict["atom_array"],
                method_weights=method_weights,
                contiguous_crop_complete_lig=contiguous_crop_complete_lig,
                spatial_crop_complete_lig=spatial_crop_complete_lig,
                drop_last=drop_last,
                remove_metal=remove_metal,
            )
            dedup_token_array, dedup_atom_array = dedup_crop.crop_by_indices(
                selected_token_indices=selected_indices,
            )
            return (
                "no_crop_dedup",
                dedup_token_array,
                dedup_atom_array,
                msa_features or {},
                template_features or {},
                -1,
            )

        # Antibody samples (H/L chain roles present) use a dedicated crop that keeps the
        # whole Fv and, optionally, a spatial antigen neighborhood -- the MFDesign
        # AntibodyCropper analog. Ordinary complexes keep the standard weighted methods.
        h_chain_id, l_chain_id = DataPipeline._get_antibody_chains(
            one_sample=one_sample, bioassembly_dict=bioassembly_dict
        )
        is_antibody = h_chain_id is not None or l_chain_id is not None

        # MFDesign MixedCropper analog: ``antibody_mixed_prob`` is the probability that
        # an antibody sample uses the standard weighted crop (seeded like an ordinary
        # complex -- a crop from anywhere in the complex, not forced around the Fv);
        # with probability ``1 - antibody_mixed_prob`` it uses the antibody-centered
        # AntibodyCropping. ``antibody_mixed_prob=0.0`` (the default) reproduces the
        # previous always-AntibodyCropping behaviour (MFDesign stages 1-3); MFDesign's
        # stage-4 ``MixedCropper(probability=0.5)`` maps to ``antibody_mixed_prob=0.5``.
        use_antibody_crop = is_antibody and (
            antibody_mixed_prob <= 0.0 or np.random.random() >= antibody_mixed_prob
        )

        if use_antibody_crop:
            ref_chain_indices = [c for c in (h_chain_id, l_chain_id) if c is not None]
        else:
            ref_chain_indices = DataPipeline._map_ref_chain(
                one_sample=one_sample, bioassembly_dict=bioassembly_dict
            )

        # Assembly-expanded duplicates are a sequence leak, not just wasted tokens.
        # A bioassembly often carries several copies of the same complex (8tg9: chains
        # D/F are byte-identical heavy chains, E/G identical light chains). Only one
        # copy is role-resolved and therefore CDR-masked, so the *other* copy sits in
        # the input with its ground-truth CDR sequence intact and the model can simply
        # read the answer off it. Cropping hides this during training (the Fv alone
        # nearly fills crop_size), but the test config uses crop_size=-1, so eval saw
        # the whole assembly. Keep one copy per entity.
        antibody_dedup = is_antibody

        crop = CropData(
            crop_size=crop_size,
            ref_chain_indices=ref_chain_indices,
            token_array=bioassembly_dict["token_array"],
            atom_array=bioassembly_dict["atom_array"],
            method_weights=method_weights,
            contiguous_crop_complete_lig=contiguous_crop_complete_lig,
            spatial_crop_complete_lig=spatial_crop_complete_lig,
            drop_last=drop_last,
            remove_metal=remove_metal,
            antibody_add_antigen=antibody_add_antigen,
            antibody_min_neighborhood=antibody_min_neighborhood,
            antibody_max_neighborhood=antibody_max_neighborhood,
        )
        # Get crop method
        crop_method = (
            "AntibodyCropping" if use_antibody_crop else crop.random_crop_method()
        )
        # Get crop indices based crop method
        selected_indices, reference_token_index = crop.get_crop_indices(
            crop_method=crop_method
        )
        if antibody_dedup:
            selected_indices = DataPipeline.drop_duplicate_assembly_copies(
                bioassembly_dict=bioassembly_dict,
                selected_indices=selected_indices,
                ref_chain_indices=ref_chain_indices,
            )
        # Prepare msa
        cropped_msa_features = DataPipeline.get_msa_raw_features(
            bioassembly_dict=bioassembly_dict,
            selected_indices=selected_indices,
            msa_featurizer=msa_featurizer,
        )
        # Prepare template
        cropped_template_features = DataPipeline.get_template_raw_features(
            bioassembly_dict=bioassembly_dict,
            selected_indices=selected_indices,
            template_featurizer=template_featurizer,
        )
        (
            cropped_token_array,
            cropped_atom_array,
        ) = crop.crop_by_indices(
            selected_token_indices=selected_indices,
        )

        if crop_method == "ContiguousCropping":
            resovled_atom_num = cropped_atom_array.is_resolved.sum()
            # The criterion of “more than 4 atoms” is chosen arbitrarily.
            assert (
                resovled_atom_num > 4
            ), f"{resovled_atom_num=} <= 4 after ContiguousCropping"

        return (
            crop_method,
            cropped_token_array,
            cropped_atom_array,
            cropped_msa_features,
            cropped_template_features,
            reference_token_index,
        )

    @staticmethod
    def drop_duplicate_assembly_copies(
        bioassembly_dict: dict,
        selected_indices,
        ref_chain_indices: list,
    ):
        """Keep one chain per entity, dropping assembly-expanded duplicates.

        Entities with only one chain are untouched. Returns ``selected_indices``
        unchanged if the antibody chains cannot be located, so this can never empty
        the crop.

        Args:
            bioassembly_dict: must carry ``token_array`` and ``atom_array``.
            selected_indices: token indices chosen by the cropper.
            ref_chain_indices: ``asym_id_int`` of the heavy/light chains.

        Returns:
            The filtered token indices, in the same order and type as the input.
        """
        token_array = bioassembly_dict["token_array"]
        atom_array = bioassembly_dict["atom_array"]

        sel = np.asarray(selected_indices)
        if sel.size == 0:
            return selected_indices

        centre = np.asarray(token_array.get_annotation("centre_atom_index"))[sel]
        asym = np.asarray(atom_array.asym_id_int)[centre]
        entity = np.asarray(atom_array.label_entity_id)[centre]
        coord = np.asarray(atom_array.coord)[centre]

        refs = {int(c) for c in ref_chain_indices if c is not None}
        anchor_mask = np.isin(asym, list(refs)) if refs else np.zeros_like(asym, bool)
        if not anchor_mask.any():
            # No antibody chain in the crop -- nothing to anchor "closest copy" on,
            # and no CDR mask to leak against. Leave the selection alone.
            return selected_indices
        anchor = coord[anchor_mask].mean(axis=0)

        keep_asym: set[int] = set()
        for ent in np.unique(entity):
            ent_mask = entity == ent
            chains = np.unique(asym[ent_mask])
            ref_chains = [int(a) for a in chains if int(a) in refs]
            if ref_chains:
                keep_asym.update(ref_chains)
            elif len(chains) == 1:
                keep_asym.add(int(chains[0]))
            else:
                closest = min(
                    (int(a) for a in chains),
                    key=lambda a: float(
                        np.linalg.norm(coord[asym == a].mean(axis=0) - anchor)
                    ),
                )
                keep_asym.add(closest)

        keep = np.isin(asym, list(keep_asym))
        if not keep.any():
            return selected_indices
        filtered = sel[keep]
        if isinstance(selected_indices, torch.Tensor):
            return torch.as_tensor(
                filtered, dtype=selected_indices.dtype, device=selected_indices.device
            )
        return filtered

    @staticmethod
    def save_atoms_to_cif(
        output_cif_file: str, atom_array: AtomArray, include_bonds: bool = False
    ) -> None:
        """
        Save atom array data to a CIF file.

        Args:
            output_cif_file (str): The output path for saving atom array in cif
            atom_array (AtomArray): The atom array to be saved
            include_bonds (bool): Whether to include bond information in the CIF file. Default is False.

        """
        strucio.save_structure(
            file_path=output_cif_file,
            array=atom_array,
            data_block=os.path.basename(output_cif_file).replace(".cif", ""),
            include_bonds=include_bonds,
        )
