#!/usr/bin/env python3
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

"""Step 2.5 -- CDR-similarity MSA filtering on a3m files.

This is a port of the *filtering part* of MFDesign's ``scripts/process/convert_msa.py``
(``boltz.data.parse.csv.filter_msa_df``), adapted to operate directly on ``.a3m`` files
instead of the intermediate ``.csv`` files, and to derive each query's CDR positions
with Protenix's abnumber-based detection instead of a precomputed ``summary.json``.

For every a3m file:

  1. The first record is the query. Its CDR1/CDR2/CDR3 spans are computed from the query
     sequence via :func:`protenix.data.antibody_cdr._chain_residue_region_labels`
     (Chothia numbering through abnumber/anarcii). Queries that do not parse as an
     antibody variable domain (i.e. antigen / non-antibody chains, or antibodies with an
     incomplete CDR) are copied through unchanged -- there is nothing to filter.
  2. Every homolog whose CDR identity to the query is ``>= threshold`` on *any* of
     CDR1/CDR2/CDR3 is dropped. This removes sequences that would leak the CDR being
     designed. Identity is computed on the match columns (a3m insertions -- lowercase
     letters -- are removed first so each homolog aligns 1:1 with the query), exactly as
     MFDesign does.
  3. The query is always kept. Surviving records are written to the output a3m with their
     original headers and (raw, insertion-preserving) sequences.

Before processing, all ``*.a3m`` files in ``--input_dir`` are moved into
``--prefiltered_dir`` (default ``<input_dir>/prefiltered``); filtered results are written
to ``--filtered_dir`` (default ``<input_dir>/filtered``). The move is idempotent: if the
input directory has already been drained, the script processes whatever is in the
prefiltered directory.

Example
-------
    python scripts/step2_5_filter_msa_cdr_similarity.py \
        --input_dir /home/dinge/data/proj/protenix_codesign/data/mmcif_msa_initial \
        --msa_filtering_threshold 0.2
"""

import argparse
import concurrent.futures
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Reuse Protenix's abnumber-based CDR detector so this script stays consistent with the
# rest of the antibody pipeline (protenix/data/antibody_cdr.py).
try:
    from protenix.data.antibody_cdr import (
        _CDR_LABEL_SET,
        _chain_residue_region_labels,
    )
except ImportError as e:  # pragma: no cover - environment guard
    print(
        "Error: could not import protenix.data.antibody_cdr. Run this script from the "
        "Protenix repo root with the protenix environment active.\n"
        f"Details: {e}"
    )
    sys.exit(1)


DEFAULT_INPUT_DIR = "/home/dinge/data/proj/protenix_codesign/data/mmcif_msa_initial"


# --------------------------------------------------------------------------------------
# a3m IO
# --------------------------------------------------------------------------------------
def read_a3m(path: Path) -> "list[tuple[str, str]]":
    """Parse an a3m into a list of ``(header, sequence)`` records (headers keep ``>``)."""
    records: list[tuple[str, str]] = []
    header = None
    seq_parts: list[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_parts)))
                header = line
                seq_parts = []
            elif line:
                seq_parts.append(line)
        if header is not None:
            records.append((header, "".join(seq_parts)))
    return records


def write_a3m(path: Path, records: "list[tuple[str, str]]") -> None:
    """Write ``(header, sequence)`` records to an a3m file."""
    with open(path, "w") as fh:
        for header, seq in records:
            fh.write(f"{header}\n{seq}\n")


# --------------------------------------------------------------------------------------
# Filtering primitives (faithful to MFDesign filter_msa_df / get_cdr_indices)
# --------------------------------------------------------------------------------------
def match_columns(seq: str) -> str:
    """Drop a3m insertions (lowercase) so the sequence aligns 1:1 with the query."""
    return "".join(c for c in seq if not c.islower())


def cdr_spans_from_flags(flags: "list[bool]") -> "list[tuple[int, int]]":
    """Convert a per-residue CDR boolean mask into contiguous ``(start, end)`` spans."""
    spans: list[tuple[int, int]] = []
    start = None
    for i, flag in enumerate(flags):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(flags)))
    return spans


def calculate_similarity(seq1: str, seq2: str) -> float:
    """Fractional identity of ``seq2`` against ``seq1`` (denominator = len(seq1))."""
    if len(seq1) == 0:
        return 0.0
    matches = sum(a == b for a, b in zip(seq1, seq2))
    return float(matches) / len(seq1)


def filter_records(
    records: "list[tuple[str, str]]",
    spans: "list[tuple[int, int]]",
    query_seq: str,
    threshold: float,
    max_seqs: "int | None",
) -> "list[tuple[str, str]]":
    """Keep the query and every homolog not too similar to the query CDRs."""
    query_cdrs = [query_seq[a:b] for a, b in spans]

    kept = [records[0]]  # query is always retained
    for header, seq in records[1:]:
        aligned = match_columns(seq)
        # Preserve exact query duplicates. ColabFold a3m repeats the query sequence to
        # mark the UniRef -> ColabFold-envdb boundary; step3 relies on that second
        # occurrence to delimit the pairing region. Such a record is identical to the
        # query (100% CDR identity), so the leakage filter below would otherwise drop it.
        if aligned == query_seq:
            kept.append((header, seq))
            continue
        drop = False
        for (a, b), q_cdr in zip(spans, query_cdrs):
            if calculate_similarity(q_cdr, aligned[a:b]) >= threshold:
                drop = True
                break
        if not drop:
            kept.append((header, seq))
        if max_seqs is not None and len(kept) >= max_seqs:
            break
    return kept


def process_single_a3m(
    a3m_path: Path,
    output_dir: Path,
    threshold: float,
    max_seqs: "int | None",
    mask_query_cdr: bool,
) -> dict:
    """Filter one a3m file and write the result to ``output_dir``."""
    try:
        out_path = output_dir / a3m_path.name
        records = read_a3m(a3m_path)

        if not records:
            write_a3m(out_path, records)
            return {"status": "empty", "file": a3m_path.name, "before": 0, "after": 0}

        query_header, query_seq = records[0]
        before = len(records)

        # CDR detection (abnumber/Chothia). None => not an antibody Fv (antigen or
        # incomplete CDR) => nothing to filter, copy through.
        # Region labels are 1-7 (fr1..fr4 / cdr1..cdr3), 0 outside the Fv, or None
        # for a non-antibody chain. Convert to a per-residue CDR bool for span
        # detection (CDR labels are cdr1/cdr2/cdr3 in _CDR_LABEL_SET).
        labels = _chain_residue_region_labels(query_seq, {})
        flags = (
            [lbl in _CDR_LABEL_SET for lbl in labels] if labels is not None else None
        )
        spans = cdr_spans_from_flags(flags) if flags else []
        if not spans:
            capped = records if max_seqs is None else records[:max_seqs]
            write_a3m(out_path, capped)
            return {
                "status": "copied",
                "file": a3m_path.name,
                "before": before,
                "after": len(capped),
            }

        kept = filter_records(records, spans, query_seq, threshold, max_seqs)

        if mask_query_cdr:
            masked = list(query_seq)
            for a, b in spans:
                for i in range(a, b):
                    masked[i] = "X"
            kept[0] = (query_header, "".join(masked))

        write_a3m(out_path, kept)
        return {
            "status": "filtered",
            "file": a3m_path.name,
            "before": before,
            "after": len(kept),
        }

    except Exception as e:  # noqa: BLE001 - report and continue with other files
        return {
            "status": "failure",
            "file": a3m_path.name,
            "message": f"Error processing '{a3m_path.name}': {e}",
        }


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def move_a3m_files(input_dir: Path, prefiltered_dir: Path) -> None:
    """Move ``*.a3m`` from ``input_dir`` into ``prefiltered_dir`` (idempotent)."""
    prefiltered_dir.mkdir(parents=True, exist_ok=True)
    to_move = sorted(input_dir.glob("*.a3m"))
    if not to_move:
        print(
            f"No .a3m files directly in '{input_dir}' -- assuming they were already "
            f"moved to '{prefiltered_dir}'."
        )
        return
    print(f"Moving {len(to_move)} .a3m files into '{prefiltered_dir}' ...")
    for src in tqdm(to_move, desc="Moving files"):
        dst = prefiltered_dir / src.name
        shutil.move(str(src), str(dst))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=DEFAULT_INPUT_DIR,
        help="Directory currently holding the .a3m files (they are moved to "
        "--prefiltered_dir before processing).",
    )
    parser.add_argument(
        "--prefiltered_dir",
        type=str,
        default=None,
        help="Where the original a3m files are moved. Default: <input_dir>/prefiltered.",
    )
    parser.add_argument(
        "--filtered_dir",
        type=str,
        default=None,
        help="Where filtered a3m files are written. Default: <input_dir>/filtered.",
    )
    parser.add_argument(
        "--msa_filtering_threshold",
        type=float,
        default=0.2,
        help="Drop a homolog if its CDR identity to the query is >= this on any CDR.",
    )
    parser.add_argument(
        "--max_seqs",
        type=int,
        default=None,
        help="Optional cap on sequences kept per file (query included). Default: no cap.",
    )
    parser.add_argument(
        "--mask_query_cdr",
        action="store_true",
        help="Replace the query's CDR residues with 'X' in the output (MFDesign "
        "filter_msa_df behaviour). Off by default so the a3m query stays intact.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=os.cpu_count(),
        help="Number of parallel worker processes.",
    )
    parser.add_argument(
        "--skip_move",
        action="store_true",
        help="Do not move input files; read directly from --prefiltered_dir.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    prefiltered_dir = (
        Path(args.prefiltered_dir)
        if args.prefiltered_dir
        else input_dir / "prefiltered"
    )
    filtered_dir = (
        Path(args.filtered_dir) if args.filtered_dir else input_dir / "filtered"
    )
    filtered_dir.mkdir(parents=True, exist_ok=True)

    # 1. Move originals into the prefiltered directory.
    if not args.skip_move:
        move_a3m_files(input_dir, prefiltered_dir)
    else:
        prefiltered_dir.mkdir(parents=True, exist_ok=True)

    a3m_files = sorted(prefiltered_dir.glob("*.a3m"))
    if not a3m_files:
        print(f"No .a3m files to process in '{prefiltered_dir}'.")
        return
    print(
        f"Filtering {len(a3m_files)} a3m files "
        f"(threshold={args.msa_filtering_threshold}) -> '{filtered_dir}'"
    )

    # 2. Filter each file in parallel.
    filtered_percentages: list[float] = []
    n_filtered = n_copied = n_empty = n_fail = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        futures = [
            ex.submit(
                process_single_a3m,
                path,
                filtered_dir,
                args.msa_filtering_threshold,
                args.max_seqs,
                args.mask_query_cdr,
            )
            for path in a3m_files
        ]
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Filtering",
        ):
            result = future.result()
            status = result["status"]
            if status == "filtered":
                n_filtered += 1
                before, after = result["before"], result["after"]
                if before > 1:  # exclude the query from the leakage-removal percentage
                    filtered_percentages.append(
                        (before - after) / (before - 1) * 100
                    )
            elif status == "copied":
                n_copied += 1
            elif status == "empty":
                n_empty += 1
            else:
                n_fail += 1
                print(result.get("message", result))

    print("\nProcessing complete.")
    print("\n--- Statistics ---")
    print(f"Antibody files filtered : {n_filtered}")
    print(f"Non-antibody copied     : {n_copied}")
    print(f"Empty files             : {n_empty}")
    print(f"Failures                : {n_fail}")
    if filtered_percentages:
        mean = np.mean(filtered_percentages)
        std = np.std(filtered_percentages)
        print(
            f"Homologs removed        : {mean:.2f} ± {std:.2f}% "
            f"(mean over {len(filtered_percentages)} antibody files)"
        )


if __name__ == "__main__":
    main()
