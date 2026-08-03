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

"""Benchmark 2 -- CDR RMSD to the ground-truth structure.

Offline companion to the codesign eval. The eval writes, per test sample, the
N predicted structures produced with replacement sampling (structural inpainting
at every denoising step), the native structure in identical atom ordering, and a
``*_regions.json`` describing which residues are framework and which are CDRs.

This script then, for every predicted structure independently:

  1. packs side chains and relaxes with PyRosetta,
  2. superposes onto the native **framework** Ca atoms,
  3. computes per-CDR Ca RMSD (H1/H2/H3/L1/L2/L3) plus Loop-RMSD over the central
     CDR-H3 residues.

Results are written per structure, so downstream you can aggregate however you
like (mean over the N samples, best-of-N, distribution) without re-running.

Usage:
    python scripts/cdr_rmsd_benchmark.py \\
        --pred_dir output/<run>/structures/<test_set>_step<N>_<raw|ema> \\
        --out_csv  cdr_rmsd.csv

    # skip PyRosetta (geometry only -- see the note on Ca RMSD below)
    python scripts/cdr_rmsd_benchmark.py --pred_dir ... --out_csv ... --no_relax

NOTE ON RELAX: packing side chains cannot change a Ca-only RMSD at all -- it moves
rotamers, not backbone. Only ``FastRelax`` shifts Ca, and typically by a fraction
of an Angstrom. Pack/relax matters for full-atom or energy-based scoring; for the
Ca RMSD reported here expect it to be close to a no-op. ``--no_relax`` is provided
so you can quantify that difference rather than assume it.
"""

import argparse
import csv
import json
import os
from glob import glob
from typing import Optional

import numpy as np

# Chothia region ids shared with the eval (protenix/data/antibody_cdr.py).
FRAMEWORK_REGIONS = (1, 3, 5, 7)
CDR_REGIONS = {
    "H1": (1, 2), "H2": (1, 4), "H3": (1, 6),
    "L1": (2, 2), "L2": (2, 4), "L3": (2, 6),
}


def load_ca(path: str) -> "dict[tuple[str, int], np.ndarray]":
    """Map (chain, res_id) -> CA coordinate for one structure.

    Dispatches on extension: the eval writes CIF, but PyRosetta relax returns PDB,
    so both have to be readable here.
    """
    if path.lower().endswith((".pdb", ".ent")):
        import biotite.structure.io.pdb as pdb

        arr = pdb.PDBFile.read(path).get_structure(model=1)
    else:
        import biotite.structure.io.pdbx as pdbx

        arr = pdbx.get_structure(pdbx.CIFFile.read(path), model=1)
    sel = arr.atom_name == "CA"
    return {
        (str(c), int(r)): xyz
        for c, r, xyz in zip(arr.chain_id[sel], arr.res_id[sel], arr.coord[sel])
    }


def kabsch(mobile: np.ndarray, target: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """Rotation/translation superposing ``mobile`` onto ``target`` (both [N, 3])."""
    mc, tc = mobile.mean(0), target.mean(0)
    cov = (mobile - mc).T @ (target - tc)
    u, _, vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return rot, tc - rot @ mc


def relax_with_pyrosetta(cif_path: str, out_path: str) -> Optional[str]:
    """Pack side chains and FastRelax. Returns the relaxed path, or None on failure."""
    try:
        import pyrosetta
        from pyrosetta.rosetta.protocols.relax import FastRelax
    except ImportError:
        return None
    if not getattr(relax_with_pyrosetta, "_init", False):
        pyrosetta.init("-mute all -ex1 -ex2aro", silent=True)
        relax_with_pyrosetta._init = True

    # PyRosetta's mmCIF reader rejects the minimal CIF biotite emits ("There are
    # multiple blocks in the file"), so round-trip through PDB, which both agree on.
    import biotite.structure.io.pdb as biotite_pdb
    import biotite.structure.io.pdbx as biotite_pdbx

    tmp_pdb = out_path.replace(".pdb", "_in.pdb")
    arr = biotite_pdbx.get_structure(biotite_pdbx.CIFFile.read(cif_path), model=1)
    arr.bonds = None
    pdb_f = biotite_pdb.PDBFile()
    pdb_f.set_structure(arr)
    pdb_f.write(tmp_pdb)

    pose = pyrosetta.pose_from_file(tmp_pdb)
    scorefxn = pyrosetta.get_fa_scorefxn()
    fr = FastRelax(scorefxn, 1)  # one repeat: pack + minimise
    fr.apply(pose)
    pose.dump_pdb(out_path)
    return out_path


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).sum(-1).mean()))


def score_one(pred_cif, native_cif, meta, relax_dir=None) -> "dict[str, float]":
    """Relax, framework-align, and score one predicted structure."""
    path = pred_cif
    if relax_dir is not None:
        os.makedirs(relax_dir, exist_ok=True)
        relaxed = relax_with_pyrosetta(
            pred_cif,
            os.path.join(relax_dir, os.path.basename(pred_cif).replace(".cif", ".pdb")),
        )
        if relaxed is None:
            raise RuntimeError(
                "PyRosetta unavailable -- install it or pass --no_relax."
            )
        path = relaxed

    pred, native = load_ca(path), load_ca(native_cif)
    residues = [r for r in meta["residues"] if r["resolved"]]

    def coords(keep):
        keys = [
            (r["chain"], r["res_id"])
            for r in residues
            if keep(r) and (r["chain"], r["res_id"]) in pred
            and (r["chain"], r["res_id"]) in native
        ]
        if not keys:
            return None, None
        return (
            np.stack([pred[k] for k in keys]),
            np.stack([native[k] for k in keys]),
        )

    # 1. superpose on the antibody framework Ca atoms
    fw_p, fw_n = coords(
        lambda r: r["chain_type"] in (1, 2) and r["region_type"] in FRAMEWORK_REGIONS
    )
    if fw_p is None or len(fw_p) < 3:
        raise RuntimeError("fewer than 3 framework Ca atoms to align on")
    rot, trans = kabsch(fw_p, fw_n)
    xform = {k: rot @ v + trans for k, v in pred.items()}

    out = {"framework_n": len(fw_p), "framework_rmsd": rmsd(
        np.stack([xform[k] for k in pred if k in native and
                  (k in {(r["chain"], r["res_id"]) for r in residues
                         if r["chain_type"] in (1, 2)
                         and r["region_type"] in FRAMEWORK_REGIONS})]),
        fw_n,
    )}

    # 2. per-CDR Ca RMSD, no per-loop re-alignment
    for name, (ct, rt) in CDR_REGIONS.items():
        keys = [
            (r["chain"], r["res_id"])
            for r in residues
            if r["chain_type"] == ct and r["region_type"] == rt
            and (r["chain"], r["res_id"]) in xform
            and (r["chain"], r["res_id"]) in native
        ]
        if not keys:
            continue
        out[f"rmsd_{name}"] = rmsd(
            np.stack([xform[k] for k in keys]), np.stack([native[k] for k in keys])
        )
        out[f"n_{name}"] = len(keys)

    # 3. Loop-RMSD: central CDR-H3 residues, stem trimmed at both ends
    h3 = [
        (r["chain"], r["res_id"])
        for r in residues
        if r["chain_type"] == 1 and r["region_type"] == 6
        and (r["chain"], r["res_id"]) in xform
    ]
    stem = int(meta.get("h3_loop_stem", 2))
    if len(h3) > 2 * stem:
        loop = h3[stem:-stem]
        out["rmsd_H3_loop"] = rmsd(
            np.stack([xform[k] for k in loop]), np.stack([native[k] for k in loop])
        )
        out["n_H3_loop"] = len(loop)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred_dir", required=True, help="eval structures/ subdirectory")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--relax_dir", default=None, help="where to write relaxed PDBs")
    ap.add_argument("--no_relax", action="store_true", help="skip PyRosetta entirely")
    args = ap.parse_args()

    relax_dir = None if args.no_relax else (
        args.relax_dir or os.path.join(args.pred_dir, "relaxed")
    )

    rows = []
    for meta_path in sorted(glob(os.path.join(args.pred_dir, "*_regions.json"))):
        with open(meta_path) as fh:
            meta = json.load(fh)
        pid = meta["pdb_id"]
        native = os.path.join(args.pred_dir, f"{pid}_native.cif")
        if not os.path.exists(native):
            print(f"[skip] {pid}: no native structure")
            continue
        for sample in range(meta["n_samples"]):
            pred = os.path.join(args.pred_dir, f"{pid}_sample{sample}.cif")
            if not os.path.exists(pred):
                continue
            try:
                res = score_one(pred, native, meta, relax_dir)
            except Exception as exc:  # noqa: BLE001 - report and continue
                print(f"[warn] {pid} sample {sample}: {exc}")
                continue
            rows.append({"pdb_id": pid, "sample": sample, **res})
            print(f"{pid} s{sample}: " + "  ".join(
                f"{k}={v:.3f}" for k, v in res.items() if k.startswith("rmsd")
            ))

    if not rows:
        print("no structures scored")
        return
    keys = sorted({k for r in rows for k in r})
    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["pdb_id", "sample"] + [
            k for k in keys if k not in ("pdb_id", "sample")
        ])
        w.writeheader()
        w.writerows(rows)

    print(f"\nwrote {len(rows)} rows -> {args.out_csv}")
    print(f"{'metric':<16}{'mean':>9}{'median':>9}{'n':>6}")
    for k in keys:
        if not k.startswith("rmsd"):
            continue
        vals = np.array([r[k] for r in rows if k in r], dtype=float)
        print(f"{k:<16}{vals.mean():>9.3f}{np.median(vals):>9.3f}{len(vals):>6}")


if __name__ == "__main__":
    main()
