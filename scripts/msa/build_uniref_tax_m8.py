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

"""Reconstruct ``uniref_tax.m8`` (accession -> NCBI TaxID) from a local mmseqs
UniRef30 database, for when the ColabFold search was run *without* the taxonomy
``convertalis`` patch (so ``uniref_tax.m8`` was never produced).

The mapping is recovered directly from the UniRef30 mmseqs database instead of
re-running the search:

  * ``<db>_seq_h`` (+ ``.index``) -- the header database: internal key -> accession.
    Records are stored in key order (record ``i`` == key ``i``), each terminated by
    ``"\\n\\0"``.
  * ``<db>_mapping`` -- a 5-byte header followed by 8-byte ``(key:uint32, taxid:uint32)``
    little-endian records, i.e. internal key -> NCBI TaxID.

Both are keyed by the same sequential internal id, so joining them yields
``accession -> taxid`` for every sequence in the database. We only emit the accessions
that actually occur in the input a3m headers, producing a compact ``uniref_tax.m8`` whose
columns match what ``step3-uniref_add_taxid.py`` reads (``col1 = accession``,
``col2 = taxid``).

Example
-------
    python scripts/msa/build_uniref_tax_m8.py \
        --db_prefix /home/dinge/MMseqs2/ColabFold/uniref30_2202_db \
        --a3m_dir /home/dinge/data/proj/protenix_codesign/data/mmcif_msa_initial/filtered
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np


def collect_a3m_accessions(a3m_dir: Path) -> "set[bytes]":
    """Union of every non-query MSA header's accession token across all a3m files."""
    wanted: set[bytes] = set()
    files = sorted(a3m_dir.glob("*.a3m"))
    print(f"Scanning {len(files)} a3m files for accessions ...")
    t0 = time.time()
    for i, path in enumerate(files):
        with open(path, "rb") as fh:
            first = True
            for line in fh:
                if not line.startswith(b">"):
                    continue
                if first:
                    first = False  # query line, skip
                    continue
                # accession = header up to first whitespace/tab, sans '>'
                token = line[1:].split(maxsplit=1)[0].strip()
                if token:
                    wanted.add(token)
        if (i + 1) % 1000 == 0:
            print(
                f"  {i + 1}/{len(files)} files, {len(wanted):,} unique accessions "
                f"({time.time() - t0:.0f}s)"
            )
    print(f"Collected {len(wanted):,} unique accessions in {time.time() - t0:.0f}s")
    return wanted


def load_taxid_by_key(mapping_path: Path, header_offset: int = 5) -> np.ndarray:
    """Load the ``<db>_mapping`` file into an array indexed by internal key.

    The file is a ``header_offset``-byte header followed by 8-byte
    ``(key:uint32, taxid:uint32)`` little-endian records. Keys are sequential
    ``0..N-1``; we verify this and return ``taxid_by_key`` where index == key.
    """
    size = os.path.getsize(mapping_path)
    n_records = (size - header_offset) // 8
    if (size - header_offset) % 8 != 0:
        raise ValueError(
            f"{mapping_path} size {size} not consistent with a {header_offset}-byte "
            f"header + 8-byte records; check the header offset."
        )
    print(f"Loading mapping: {n_records:,} (key, taxid) records ...")
    with open(mapping_path, "rb") as fh:
        fh.seek(header_offset)
        arr = np.frombuffer(fh.read(n_records * 8), dtype=np.uint32).reshape(-1, 2)
    keys, taxids = arr[:, 0], arr[:, 1]
    # Validate sequential keys so we can treat position == key downstream.
    if not (keys[:1000] == np.arange(1000, dtype=np.uint32)).all() or keys[-1] != (
        n_records - 1
    ):
        raise ValueError(
            "Mapping keys are not the expected sequential 0..N-1; refusing to guess "
            "the join. First keys: " + str(keys[:8].tolist())
        )
    return np.ascontiguousarray(taxids)


def build_m8(
    header_data_path: Path,
    taxid_by_key: np.ndarray,
    wanted: "set[bytes]",
    output_path: Path,
    chunk_size: int = 64 * 1024 * 1024,
) -> None:
    """Stream the header DB in key order and write matched accession/taxid rows.

    Record ``i`` in ``header_data_path`` corresponds to internal key ``i``; its taxid is
    ``taxid_by_key[i]``. Only accessions present in ``wanted`` (and with a nonzero taxid)
    are written, as ``0\\t<accession>\\t<taxid>``.
    """
    n_keys = taxid_by_key.shape[0]
    key = 0
    written = 0
    missing_taxid = 0
    buf = b""
    print(f"Streaming header DB, writing -> {output_path}")
    t0 = time.time()
    with open(header_data_path, "rb") as fin, open(output_path, "w") as fout:
        while True:
            chunk = fin.read(chunk_size)
            if not chunk:
                break
            buf += chunk
            records = buf.split(b"\x00")
            buf = records.pop()  # last element may be an incomplete record
            for rec in records:
                if key >= n_keys:
                    key += 1
                    continue
                acc = rec.strip()  # drop trailing '\n'
                if acc in wanted:
                    taxid = int(taxid_by_key[key])
                    if taxid != 0:
                        fout.write(f"0\t{acc.decode()}\t{taxid}\n")
                        written += 1
                    else:
                        missing_taxid += 1
                key += 1
            if key % 20_000_000 < (chunk_size // 32):
                print(
                    f"  ~{key:,}/{n_keys:,} keys scanned, {written:,} written "
                    f"({time.time() - t0:.0f}s)"
                )
    # Any trailing record without a terminating null.
    if buf.strip():
        if key < n_keys:
            acc = buf.strip()
            if acc in wanted and int(taxid_by_key[key]) != 0:
                fout_taxid = int(taxid_by_key[key])
                with open(output_path, "a") as fout:
                    fout.write(f"0\t{acc.decode()}\t{fout_taxid}\n")
                written += 1
        key += 1

    print(
        f"Done: scanned {key:,} header records ({n_keys:,} keys expected), "
        f"wrote {written:,} rows, {missing_taxid:,} matched accessions had taxid 0."
    )
    if key < n_keys:
        print(
            f"WARNING: only {key:,} header records seen but {n_keys:,} keys expected; "
            "header DB may be truncated."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--db_prefix",
        type=str,
        default="/home/dinge/MMseqs2/ColabFold/uniref30_2202_db",
        help="mmseqs UniRef30 db prefix (expects <prefix>_seq_h, <prefix>_seq_h.index, "
        "<prefix>_mapping).",
    )
    parser.add_argument(
        "--a3m_dir",
        type=str,
        required=True,
        help="Directory of a3m files whose accessions need taxids (e.g. the filtered dir).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output m8 path. Default: <a3m_dir>/uniref_tax.m8 (where step3 looks).",
    )
    parser.add_argument(
        "--mapping_header_bytes",
        type=int,
        default=5,
        help="Byte offset of the first record in <prefix>_mapping.",
    )
    args = parser.parse_args()

    db_prefix = Path(args.db_prefix)
    a3m_dir = Path(args.a3m_dir)
    output_path = Path(args.output) if args.output else a3m_dir / "uniref_tax.m8"

    header_data = Path(f"{db_prefix}_seq_h")
    mapping = Path(f"{db_prefix}_mapping")
    for p in (header_data, mapping):
        if not p.exists():
            raise FileNotFoundError(f"Required db file not found: {p}")

    wanted = collect_a3m_accessions(a3m_dir)
    if not wanted:
        print("No accessions found in a3m files; nothing to do.")
        return

    taxid_by_key = load_taxid_by_key(mapping, header_offset=args.mapping_header_bytes)
    build_m8(header_data, taxid_by_key, wanted, output_path)
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
