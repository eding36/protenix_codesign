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
import random
from pathlib import Path
from typing import Optional

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from protenix.data.antibody_cdr import load_sabdab_chain_roles
from protenix.data.pipeline.data_pipeline import DataPipeline
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


def run_gen_data(
    input_path: Path,
    output_indices_csv: Path,
    bioassembly_output_dir: Path,
    cluster_file: Optional[Path],
    distillation: bool = False,
    num_workers: int = 1,
    strip_antibody_cdr: bool = False,
    sabdab_summary: Optional[Path] = None,
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
        help="Path to the input directory containing MMCIF files or a .txt file listing MMCIF file paths.",
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

    parser.add_argument(
        "--strip_antibody_cdr",
        action="store_true",
        help=(
            "Strip antibody variable-domain CDR side chains (keep backbone + CB) "
            "and add a per-atom 'is_cdr' annotation, to prevent leakage of the "
            "design target. Requires 'abnumber' (ANARCI/anarcii)."
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
            "H/L-corrected SAbDab labels instead of abnumber-inferred roles."
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
    )
