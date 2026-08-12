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

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from protenix.data.antibody_cdr import load_sabdab_chain_roles
from protenix.data.pipeline.data_pipeline import DataPipeline
from protenix.data.utils import save_atoms_to_cif
from protenix.utils.file_io import dump_gzip_pickle


def gen_a_bioassembly_data(
    mmcif: Path,
    bioassembly_output_dir: Path,
    cluster_file: Optional[Path],
    distillation: bool = False,
    strip_antibody_cdr: bool = True,
    sabdab_roles: Optional[dict] = None,
) -> Optional[list[dict]]:
    """
    Generates bioassembly data from an mmCIF file and saves it to the specified output directory.

    Args:
        mmcif (Path): Path to the mmCIF file.
        bioassembly_output_dir (Path): Directory where the bioassembly data will be saved.
        cluster_file (Optional[Path]): Path to the cluster file, if available.
        distillation (bool, optional): Flag indicating whether to use the 'Distillation' setting. Defaults to False.
        strip_antibody_cdr (bool, optional): If True, strip antibody CDR side chains
            (keep backbone + CB) and add an ``is_cdr`` annotation. Defaults to False.

    Returns:
        Optional[list[dict]]: A list of sample indices if data is successfully generated, otherwise None.
    """
    if distillation:
        dataset = "Distillation"
    else:
        dataset = "WeightedPDB"

    sample_indices_list, bioassembly_dict = DataPipeline.get_data_from_mmcif(
        mmcif,
        cluster_file,
        dataset,
        strip_antibody_cdr=strip_antibody_cdr,
        sabdab_roles=sabdab_roles,
    )

    if sample_indices_list and bioassembly_dict:
        pdb_id = bioassembly_dict["pdb_id"]
        # save to output dir
        dump_gzip_pickle(bioassembly_dict, bioassembly_output_dir / f"{pdb_id}.pkl.gz")
        return sample_indices_list


def gen_data_from_mmcifs(
    mmcif_list: list[Path],
    output_indices_csv: Path,
    bioassembly_output_dir: Path,
    cluster_file: Optional[Path],
    distillation: bool = False,
    num_workers: int = 1,
    strip_antibody_cdr: bool = False,
    sabdab_roles: Optional[dict] = None,
):
    """
    Generates training data from a list of mmCIF files and saves the results to a CSV file.

    Args:
        mmcif_list (list[Path]): List of paths to mmCIF files.
        output_indices_csv (Path): Path to the output CSV file where the indices will be saved.
        bioassembly_output_dir (Path): Directory where the bioassembly output will be stored.
        cluster_file (Optional[Path]): Path to the cluster file. If None, clustering is not performed.
        distillation (bool, optional): Flag indicating whether to use the 'Distillation' setting. Defaults to False.
        num_workers (int, optional): Number of parallel workers to use. Defaults to 1.
        strip_antibody_cdr (bool, optional): If True, strip antibody CDR side chains
            (keep backbone + CB) and add an ``is_cdr`` annotation. Defaults to False.
    """
    random.shuffle(mmcif_list)

    all_sample_indices_list = [
        r
        for r in tqdm(
            Parallel(n_jobs=num_workers, return_as="generator_unordered")(
                delayed(gen_a_bioassembly_data)(
                    mmcif,
                    bioassembly_output_dir,
                    cluster_file,
                    distillation,
                    strip_antibody_cdr,
                    sabdab_roles,
                )
                for mmcif in mmcif_list
            ),
            total=len(mmcif_list),
        )
    ]

    merged_results = []
    for sample_indices_list in all_sample_indices_list:
        if sample_indices_list:
            merged_results += sample_indices_list
    df = pd.DataFrame(merged_results)

    df.to_csv(output_indices_csv, index=False, quoting=csv.QUOTE_NONNUMERIC)


# ---------------------------------------------------------------------------
# Per-complex generation (MFDesign step-7)
#
# MFDesign's antibody.py iterates once per antibody-antigen complex keyed by
# ``{pdb}_{H}_{L}_{antigen}``: several complexes may share one PDB (e.g.
# ``8r9y_B_C_A`` and ``8r9y_H_L_A``), each reads the same structure but selects
# only its own H/L/Ag chains, and each output is named by the composite id so nothing
# overwrites. Here we reproduce this by building bioassembly pkls and indices from a JSON dict of
# all distinct antibody-antigen complexes and calling ``get_data_from_mmcif`` once per complex
# with that single complex's role, which makes the pipeline's existing
# single-antibody labelling exactly correct.
# ---------------------------------------------------------------------------


def _clean_chain(v) -> Optional[str]:
    """Normalize a chain id to a str of either: pdb_id, "H", "L", or "A". Empty / NA-like values become ``None``."""
    if v is None:
        return None
    v = str(v).strip()
    if v == "" or v.lower() in {"na", "nan", "none"}:
        return None
    return v


def _resolve_complexes(input_obj, complexes_json: Optional[Path]) -> dict:
    """Uses the complexes_json file (e.g. "5hhv_H_L_A": {"pdb":"5hhv","H_chain_id":"H","L_chain_id":"L","antigen_chain_id":["A"]})
    to fetch all the pdb_id/H_chain/L_chain/Ag_chain to chain_id mappings of all the -i entries.
    """
    if isinstance(input_obj, dict):
        return input_obj
    if not isinstance(input_obj, list):
        raise ValueError(
            "-i JSON must be a dict {complex_id: {...}} or a list of complex ids."
        )
    if complexes_json is None:
        raise ValueError(
            "--complexes_json (e.g. MFDesign summary.json) is required when -i is a "
            "list of complex ids."
        )
    with open(complexes_json) as f:
        role_dict = json.load(f)
    if not isinstance(role_dict, dict):
        raise ValueError("--complexes_json must be a dict {complex_id: {...}}.")

    complexes = {}
    missing = []
    for cid in input_obj:
        if cid in role_dict:
            complexes[cid] = role_dict[cid]
        else:
            missing.append(cid)
    if missing:
        print(
            f"[warn] {len(missing)} id(s) from the split are absent from "
            f"--complexes_json; skipping (first few: {missing[:5]})"
        )
    return complexes


def _find_mmcif(mmcif_dir: Path, *stems: str) -> Optional[Path]:
    """Locate ``{stem}.cif`` / ``.cif.gz`` (case-insensitive) for the first stem
    that exists. Pass the composite id first then the bare PDB id, so both a
    per-complex layout (``8r9y_B_C_A.cif``) and a shared-structure layout
    (``8r9y.cif``) are supported."""
    for stem in stems:
        if not stem:
            continue
        for candidate in (stem, stem.lower(), stem.upper()):
            for ext in (".cif", ".cif.gz"):
                p = mmcif_dir / f"{candidate}{ext}"
                if p.exists():
                    return p
    return None


def _write_single_complex_cif(
    bioassembly_dict: dict, entry: dict, out_path: Path, complex_id: str
) -> bool:
    """Write a CIF of just this complex's chains, following MFDesign's convention.

    Mirrors MFDesign's mmcif.py chain handling:
      * keep **one chain per role** -- heavy, light, then each antigen author id,
        taking the first base copy only (assembly-expanded duplicates, whose label
        chain_id carries a ``.``, are dropped), matching MFDesign's ``next(...)``;
      * order the chains **H -> L -> antigen(s)**;
      * label the output chains with their **original SAbDab author ids**
        (e.g. ``H``/``L``/``W``) instead of the pipeline's internal ``A``/``B``/``C``.

    Also reflects the processed atom array (CDR side chains absent when
    ``--strip_antibody_cdr`` was set). Returns False if no role resolves.
    """
    atom_array = bioassembly_dict["atom_array"]
    cats = atom_array.get_annotation_categories()
    if "auth_asym_id" not in cats or "asym_id_int" not in cats:
        return False

    # author chain id -> first base-chain asym_id_int (skip assembly-expanded
    # copies whose label chain_id contains "."; keep first occurrence).
    auth_to_asym: dict = {}
    for auth, label, asym in zip(
        atom_array.auth_asym_id, atom_array.chain_id, atom_array.asym_id_int
    ):
        if "." in str(label):
            continue
        auth_to_asym.setdefault(str(auth), int(asym))

    # Roles in MFDesign order: H, L, antigen(s); one asym per author id, de-duped.
    role_names = []
    if entry.get("H"):
        role_names.append(str(entry["H"]))
    if entry.get("L"):
        role_names.append(str(entry["L"]))
    role_names.extend(str(a) for a in entry.get("antigen", []) if a)

    ordered: list = []  # (asym_id_int, output_chain_name)
    seen: set = set()
    for name in role_names:
        asym = auth_to_asym.get(name)
        if asym is None or asym in seen:
            continue
        seen.add(asym)
        ordered.append((asym, name))
    if not ordered:
        return False

    # Concatenate atoms in role order, carrying the original author id as the
    # new chain label.
    chain_dtype = atom_array.chain_id.dtype
    idx_parts, name_parts = [], []
    for asym, name in ordered:
        idx = np.where(atom_array.asym_id_int == asym)[0]
        if len(idx) == 0:
            continue
        idx_parts.append(idx)
        name_parts.append(np.full(len(idx), name, dtype=chain_dtype))
    if not idx_parts:
        return False
    order_idx = np.concatenate(idx_parts)
    new_names = np.concatenate(name_parts)

    sub = atom_array[order_idx]
    # Relabel every chain-id annotation the writer/entity block reads, so the CIF
    # is self-consistent under the original author ids.
    sub.set_annotation("chain_id", new_names)
    if "label_asym_id" in cats:
        sub.set_annotation("label_asym_id", new_names.copy())
    if "auth_asym_id" in cats:
        sub.set_annotation("auth_asym_id", new_names.copy())

    save_atoms_to_cif(
        str(out_path),
        sub,
        bioassembly_dict["entity_poly_type"],
        complex_id,
    )
    return True


def _assembly_ids(mmcif) -> list:
    """Biological-assembly ids declared in the mmCIF (e.g. ``['1', '2']``), in file
    order; empty when none are declared. Used to locate the assembly that actually
    contains an antibody complex's chains when assembly 1 does not (e.g. a second
    crystallographic antibody copy declared as assembly 2)."""
    try:
        from protenix.data.core.parser import MMCIFParser

        block = MMCIFParser(mmcif_file=str(mmcif)).cif.block
        if "pdbx_struct_assembly_gen" in block:
            ids = block["pdbx_struct_assembly_gen"]["assembly_id"].as_array(str)
            return list(dict.fromkeys(str(x) for x in ids))
    except Exception:
        pass
    return []


def gen_a_complex_data(
    complex_id: str,
    pdb_id: str,
    mmcif: Path,
    entry: dict,
    bioassembly_output_dir: Path,
    cluster_file: Optional[Path],
    distillation: bool = False,
    strip_antibody_cdr: bool = True,
    cif_output_dir: Optional[Path] = None,
    mfdesign_chain_subset: bool = False,
) -> Optional[list[dict]]:
    """Generate bioassembly data and processed cif file 
    containing selected H/L/Ag chains for a single antibody-antigen complex. 

    ``entry`` is one ``{"H", "L", "antigen"}`` role dict. The shared structure is
    parsed and only this complex's chains are labelled. The output pkl and the
    index rows' ``pdb_id`` lookup key are set to ``complex_id`` so complexes
    sharing a PDB never overwrite and the training dataset (which loads
    ``f"{pdb_id}.pkl.gz"``) resolves the correct per-complex file. The real
    4-char PDB id is preserved as ``source_pdb_id``.
    """
    dataset = "Distillation" if distillation else "WeightedPDB"
    # A single-entry role table -> the pipeline's single-antibody labelling picks
    # exactly this complex's heavy/light/antigen chains. get_data_from_mmcif looks
    # roles up by the parser's pdb_id, which is the mmCIF filename stem -- that is
    # the bare pdb for a {pdb}.cif layout but the composite id for a
    # {pdb}_{H}_{L}_{antigen}.cif layout. Key on both so either resolves.
    sabdab_roles = {pdb_id.lower(): [entry], complex_id.lower(): [entry]}

    sample_indices_list, bioassembly_dict = DataPipeline.get_data_from_mmcif(
        mmcif,
        cluster_file,
        dataset,
        strip_antibody_cdr=strip_antibody_cdr,
        mfdesign_chain_subset=mfdesign_chain_subset,
        sabdab_roles=sabdab_roles,
    )

    # Biological assembly 1 may exclude this complex's chains (e.g. a second
    # crystallographic antibody copy: 1ap2 assembly 1 = A,B, but the complex is
    # chains D,C, declared as assembly 2). When an antibody complex's H/L roles
    # fail to resolve, search the other declared assemblies for the one that
    # contains its chains (below); only if none do fall back to the full ASU.
    def _entry_is_antibody(e: dict) -> bool:
        return bool(e.get("H") or e.get("L"))

    def _roles_unresolved(rows: Optional[list]) -> bool:
        if not rows:
            return True
        r = rows[0]
        return not (
            str(r.get("H_chain_id", "")).strip() or str(r.get("L_chain_id", "")).strip()
        )

    if _entry_is_antibody(entry) and _roles_unresolved(sample_indices_list):
        # Assembly 1 didn't contain this complex's chains. Try each *other* declared
        # biological assembly and keep the first that resolves the H/L roles (e.g.
        # 1ap2's D,C copy lives in assembly 2), so the bioassembly stays minimal.
        # Only if no declared assembly contains them fall back to the full ASU.
        for aid in _assembly_ids(mmcif):
            if aid == "1":
                continue
            rows_a, bd_a = DataPipeline.get_data_from_mmcif(
                mmcif,
                cluster_file,
                dataset,
                strip_antibody_cdr=strip_antibody_cdr,
        mfdesign_chain_subset=mfdesign_chain_subset,
                sabdab_roles=sabdab_roles,
                assembly_id=aid,
            )
            if not _roles_unresolved(rows_a):
                sample_indices_list, bioassembly_dict = rows_a, bd_a
                break
        else:
            sample_indices_list, bioassembly_dict = DataPipeline.get_data_from_mmcif(
                mmcif,
                cluster_file,
                dataset,
                strip_antibody_cdr=strip_antibody_cdr,
        mfdesign_chain_subset=mfdesign_chain_subset,
                sabdab_roles=sabdab_roles,
                skip_assembly_expansion=True,
            )

    if sample_indices_list and bioassembly_dict:
        source_pdb_id = bioassembly_dict.get("pdb_id")
        bioassembly_dict["pdb_id"] = complex_id
        bioassembly_dict["source_pdb_id"] = source_pdb_id
        for row in sample_indices_list:
            row["source_pdb_id"] = source_pdb_id
            row["pdb_id"] = complex_id
        dump_gzip_pickle(
            bioassembly_dict, bioassembly_output_dir / f"{complex_id}.pkl.gz"
        )
        if cif_output_dir is not None:
            try:
                _write_single_complex_cif(
                    bioassembly_dict,
                    entry,
                    cif_output_dir / f"{complex_id}.cif",
                    complex_id,
                )
            except Exception as e:  # keep the pkl/indices even if CIF export fails
                print(f"[warn] CIF export failed for {complex_id}: {e}")
        return sample_indices_list


def _dedup_one_row_per_complex(df: "pd.DataFrame") -> "pd.DataFrame":
    """Reduce the indices to a single representative sample row per complex.

    In per-complex mode the ``pdb_id`` column holds the complex id, and each
    complex expands into many sample rows (one per chain / per pairwise interface,
    including ligand/ion interfaces). Evaluation iterates every row and averages
    the metric, so leaving them in both inflates eval cost (one 200-step rollout
    per row) and biases ``seq_acc`` toward many-chain complexes -- see the
    train-vs-test sampling asymmetry. Training is unaffected (weighted sampling),
    so this is applied to the test/eval indices only.

    The antibody crop is driven by the H/L roles on the token array, not by which
    interface row is sampled, so any row yields the same Fv crop; we deterministically
    keep the protein-protein interface row with the most tokens (the antibody-antigen
    interface, most codesign-relevant) as the representative.
    """
    if df.empty or "pdb_id" not in df.columns:
        return df
    n_rows, n_cplx = len(df), df["pdb_id"].nunique()
    df = df.copy()
    # Prefer a protein-protein *interface* row (mol type is encoded as "prot"):
    # for an antibody complex this is the antibody-antigen/antibody interface,
    # the most codesign-relevant anchor. Falls back to any row otherwise.
    is_pp = (df.get("mol_1_type") == "prot") & (df.get("mol_2_type") == "prot")
    df["_pp"] = is_pp.astype(int)
    sort_cols = ["pdb_id", "_pp"]
    ascending = [True, False]
    if "num_tokens" in df.columns:
        sort_cols.append("num_tokens")
        ascending.append(False)
    # Final deterministic tie-breaks so the kept row is reproducible.
    for tb in ("chain_1_id", "chain_2_id"):
        if tb in df.columns:
            sort_cols.append(tb)
            ascending.append(True)
    df = df.sort_values(by=sort_cols, ascending=ascending, kind="mergesort")
    df = df.drop_duplicates(subset=["pdb_id"], keep="first").drop(columns=["_pp"])
    print(
        f"[dedup] test indices: {n_rows} sample rows -> {len(df)} "
        f"(one representative row per complex; {n_cplx} complexes)."
    )
    return df


def gen_data_from_complexes(
    complexes: dict,
    mmcif_dir: Path,
    output_indices_csv: Path,
    bioassembly_output_dir: Path,
    cluster_file: Optional[Path],
    distillation: bool = False,
    num_workers: int = 1,
    strip_antibody_cdr: bool = True,
    cif_output_dir: Optional[Path] = None,
    one_row_per_complex: bool = False,
    mfdesign_chain_subset: bool = False,
):
    """Generate training data per antibody-antigen complex from a JSON dict containing all structures to be processed.

    Args:
        complexes (dict): ``{complex_id: {"pdb", "H_chain_id", "L_chain_id",
            "antigen_chain_id", ...}}`` -- MFDesign's distinct-complexes dict,
            keyed by ``{pdb}_{H}_{L}_{antigen}``.
        mmcif_dir (Path): Directory of shared ``{pdb}.cif`` / ``.cif.gz`` files.
        cif_output_dir (Optional[Path]): If given, also write a per-complex CIF
            trimmed to that complex's chains to ``{cif_output_dir}/{complex_id}.cif``.
    """
    if cif_output_dir is not None:
        cif_output_dir = Path(cif_output_dir)
        cif_output_dir.mkdir(parents=True, exist_ok=True)
    work = []
    missing = []
    for complex_id, val in complexes.items():
        val = val or {}
        pdb_id = _clean_chain(val.get("pdb")) or complex_id.split("_")[0]
        pdb_id = pdb_id.lower()
        # Try the composite-id filename first (per-complex layout, e.g.
        # 8r9y_B_C_A.cif), then the bare PDB id (shared-structure layout).
        mmcif = _find_mmcif(mmcif_dir, complex_id, pdb_id)
        if mmcif is None:
            missing.append(complex_id)
            continue
        entry = {
            "H": _clean_chain(val.get("H_chain_id")),
            "L": _clean_chain(val.get("L_chain_id")),
            "antigen": [
                str(s).strip()
                for s in (val.get("antigen_chain_id") or [])
                if str(s).strip()
            ],
            # MFDesign's curated Fv reference + CDR-masked (X at CDRs) sequences.
            # Carried through to strip_cdr_side_chains, which reads CDR boundaries
            # from these masks instead of re-numbering with abnumber (which fails
            # on ~10% of chains). Absent -> abnumber fallback.
            "H_seq": val.get("H_chain_seq"),
            "H_masked": val.get("H_chain_masked_seq"),
            "L_seq": val.get("L_chain_seq"),
            "L_masked": val.get("L_chain_masked_seq"),
        }
        work.append((complex_id, pdb_id, mmcif, entry))

    if missing:
        print(
            f"[warn] no mmCIF found in {mmcif_dir} for {len(missing)} complex(es); "
            f"skipping (first few: {missing[:5]})"
        )
    print(f"Processing {len(work)} antibody-antigen complexes.")

    random.shuffle(work)

    all_sample_indices_list = [
        r
        for r in tqdm(
            Parallel(n_jobs=num_workers, return_as="generator_unordered")(
                delayed(gen_a_complex_data)(
                    complex_id,
                    pdb_id,
                    mmcif,
                    entry,
                    bioassembly_output_dir,
                    cluster_file,
                    distillation,
                    strip_antibody_cdr,
                    cif_output_dir,
                    mfdesign_chain_subset,
                )
                for (complex_id, pdb_id, mmcif, entry) in work
            ),
            total=len(work),
        )
    ]

    merged_results = []
    for sample_indices_list in all_sample_indices_list:
        if sample_indices_list:
            merged_results += sample_indices_list
    df = pd.DataFrame(merged_results)

    # Test/eval set: keep one representative row per complex (see docstring).
    if one_row_per_complex:
        df = _dedup_one_row_per_complex(df)

    df.to_csv(output_indices_csv, index=False, quoting=csv.QUOTE_NONNUMERIC)


def run_gen_data(
    input_path: Path,
    output_indices_csv: Path,
    bioassembly_output_dir: Path,
    cluster_file: Optional[Path],
    distillation: bool = False,
    num_workers: int = 1,
    strip_antibody_cdr: bool = False,
    sabdab_summary: Optional[Path] = None,
    mmcif_dir: Optional[Path] = None,
    complexes_json: Optional[Path] = None,
    cif_output_dir: Optional[Path] = None,
    one_row_per_complex: bool = False,
    mfdesign_chain_subset: bool = False,
):
    """
    Generates data from MMCIF files and saves the output to specified locations.

    Args:
        input_path (str): Path to the input directory containing MMCIF files or a text file listing MMCIF file paths.
        output_indices_csv (str): Path to the output CSV file where indices will be saved.
        bioassembly_output_dir (str): Directory where bioassembly outputs will be saved.
        cluster_file (Optional[str]): Path to the cluster file, if any.
        distillation (bool, optional): Flag indicating whether to use the 'Distillation' setting. Defaults to False.
        num_workers (int, optional): Number of worker processes to use. Defaults to 1.

    Raises:
        NotImplementedError: If the input path is not a directory or a text file.
    """

    input_path = Path(input_path)
    bioassembly_output_dir = Path(bioassembly_output_dir)
    output_indices_csv = Path(output_indices_csv)

    # create directory for output
    output_indices_csv.parent.mkdir(parents=True, exist_ok=True)
    bioassembly_output_dir.mkdir(parents=True, exist_ok=True)

    # Per-complex mode: -i is a JSON dict of the distinct antibody-antigen
    # complexes (keyed by {pdb}_{H}_{L}_{antigen}) OR a list of complex ids (a
    # split file such as train_entry.json). For a list, roles come from
    # --complexes_json (e.g. step-1 summary.json) or are parsed from the ids.
    # Iterate once per complex, pulling the shared structure from --mmcif_dir.
    if input_path.suffix == ".json":
        if mmcif_dir is None:
            raise ValueError(
                "--mmcif_dir is required when -i is a distinct-complexes JSON."
            )
        if cif_output_dir is None:
            raise ValueError(
                "--cif_output_dir is required for antibody-codesign (per-complex) "
                "preprocessing: each complex is exported as a trimmed per-complex CIF."
            )
        with open(input_path) as f:
            input_obj = json.load(f)
        complexes = _resolve_complexes(input_obj, complexes_json)
        # Dedup to one row per complex for the test/eval split only. Auto-enabled
        # when the split file looks like a test entry (e.g. test_entry.json), or
        # forced via --one_row_per_complex. Train/val keep all rows (weighted
        # sampling relies on them).
        dedup = one_row_per_complex or ("test" in input_path.stem.lower())
        if dedup:
            print(
                f"[dedup] '{input_path.name}' treated as a test/eval split: "
                "writing one representative sample row per complex."
            )
        gen_data_from_complexes(
            complexes,
            Path(mmcif_dir),
            output_indices_csv,
            bioassembly_output_dir,
            cluster_file,
            distillation,
            num_workers,
            strip_antibody_cdr,
            cif_output_dir,
            one_row_per_complex=dedup,
            mfdesign_chain_subset=mfdesign_chain_subset,
        )
        return

    if input_path.is_dir():
        mmcif_list = list(input_path.glob("*.cif")) + list(input_path.glob("*.cif.gz"))
    elif input_path.suffix == ".txt":
        with open(input_path) as f:
            mmcif_list = [i.strip() for i in f.readlines()]
    else:
        raise NotImplementedError(f"Unsupported input path: {input_path}")

    # Curation-first H/L/antigen labels: reuse MFDesign's processed summary CSV
    # (already H/L-corrected and filtered) instead of inferring the antigen. Loaded
    # once here and broadcast to the workers.
    sabdab_roles = (
        load_sabdab_chain_roles(sabdab_summary) if sabdab_summary is not None else None
    )

    gen_data_from_mmcifs(
        mmcif_list,
        output_indices_csv,
        bioassembly_output_dir,
        cluster_file,
        distillation,
        num_workers,
        strip_antibody_cdr,
        sabdab_roles,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_path",
        type=Path,
        default=None,
        help=(
            "Input source. Either:"
                "(a: normal pdb preprocessing) a directory of mmCIF files, or "
                "(b: antibody-antigen complex preprocessing) a .json file from Step 6 of MFDesign preprocessing containing either the train/val/test split cifs. "
                "b) also requires --mmcif_dir)."
        ),
    )
    parser.add_argument(
        "-o",
        "--output_csv",
        type=Path,
        default=None,
        help="Path to the output CSV file where indices will be saved.",
    )
    parser.add_argument(
        "-b",
        "--bio_output_dir",
        type=Path,
        default=None,
        help="Directory where bioassembly outputs will be saved.",
    )
    parser.add_argument(
        "-c",
        "--cluster_file",
        type=Path,
        default=None,
        help="Path to the cluster txt file, if any",
    )

    parser.add_argument(
        "-d",
        "--distillation",
        action="store_true",
        help="Whether to use the 'Distillation' setting",
    )

    parser.add_argument(
        "-n",
        "--n_cpu",
        type=int,
        default=1,
        help="Number of worker processes to use. Defaults to 1.",
    )




    """Antibody codesign data processing arguments below, all are OPTIONAL."""

    parser.add_argument(
        "--strip_antibody_cdr",
        action="store_true",
        help=(
            "Strip antibody CDR side chains (keep backbone + CB) "
            "and add a per-atom 'is_cdr' annotation, to prevent leakage of the "
            "design target."
        ),
    )

    parser.add_argument(
        "--sabdab_summary",
        type=Path,
        default=None,
        help=(
            "Path to MFDesign's processed summary CSV (columns pdb, H_chain_id, "
            "L_chain_id, antigen_chain_id). When given, the indices CSV's "
            "H_chain_id/L_chain_id/antigen_chain_ids columns use these curated, "
            "H/L-corrected SAbDab labels instead of inferred roles. "
            "Used only when certain that your input cifs only have 1 of each: H/L/Ag. DO NOT use this flag when -i is a JSON (per-complex)."
        ),
    )

    parser.add_argument(
        "--mmcif_dir",
        type=Path,
        default=None,
        help=(
            "Directory of all mmCIF files used to build bioassemblies. "
            "Some of the mmCIFs have multiple antibody-antigen complexes ({pdb}.cif / .cif.gz). Required when "
            "Each complex reads its PDB's structure from here."
        ),
    )

    parser.add_argument(
        "--complexes_json",
        type=Path,
        default=None,
        help=(
            "JSON file containing all the . H/L/Ag to actual chain ID mappings."
        ),
    )

    parser.add_argument(
        "--one_row_per_complex",
        action="store_true",
        help=(
            "Write only one representative sample row per complex to the indices "
            "CSV (the antibody-antigen interface row). Intended for the TEST/EVAL "
            "split so each complex is evaluated once instead of once per "
            "chain/interface. Auto-enabled when -i looks like a test entry file "
            "(name contains 'test'). Do NOT use for train/val (weighted sampling "
            "needs all rows)."
        ),
    )

    parser.add_argument(
        "--mfdesign_chain_subset",
        action="store_true",
        help=(
            "Trim each structure to its H/L/antigen chains before tokenization, so "
            "the token count matches MFDesign's benchmark. The crystallographic "
            "assembly carries chains their YAML never sees, and glycans are atomised "
            "one token per heavy atom. Intended for the TEST split."
        ),
    )

    parser.add_argument(
        "--cif_output_dir",
        type=Path,
        default=None,
        help=(
            "Output directory for each CIF trimmed to each "
            "complex's heavy/light/antigen chains to "
            "{cif_output_dir}/{complex_id}.cif (one antibody-antigen complex per "
            "file). Reflects the processed atom array, so CDR side chains are "
            "absent when --strip_antibody_cdr is set."
        ),
    )

    args = parser.parse_args()

    run_gen_data(
        input_path=args.input_path,
        output_indices_csv=args.output_csv,
        bioassembly_output_dir=args.bio_output_dir,
        cluster_file=args.cluster_file,
        distillation=args.distillation,
        num_workers=args.n_cpu,
        strip_antibody_cdr=args.strip_antibody_cdr,
        sabdab_summary=args.sabdab_summary,
        mmcif_dir=args.mmcif_dir,
        complexes_json=args.complexes_json,
        cif_output_dir=args.cif_output_dir,
        one_row_per_complex=args.one_row_per_complex,
        mfdesign_chain_subset=args.mfdesign_chain_subset,
    )
