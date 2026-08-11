#!/usr/bin/env python3
"""Side-chain chi-angle error between predicted and native structures.

Motivation: none of the training losses tell you whether the model has learned to
recover side-chain *torsions*. ``mse_loss`` is Cartesian, so it conflates a wrong
twist with wrong local geometry; ``bond_loss`` measures covalent-geometry
violation, which torsion noising preserves by construction and therefore leaves
near zero regardless of what the model learned. This computes the quantity that
actually answers the question -- how far is each predicted dihedral from the
native one.

Reads the structures the eval already dumps (``<pid>_sample<i>.cif`` against
``<pid>_native.cif``), so it can be run retroactively on past evals as well as
new ones. Reports mean absolute error per chi level, in degrees.

Reference points: ~90 deg is what random guessing gives (errors are wrapped to
[0, 180]). chi1 should be markedly better than chi3/chi4, which are genuinely
floppy. Compare a torsion-noised run against a Gaussian-only baseline at the same
step -- the absolute number matters less than the difference.

Usage:
    python scripts/chi_angle_error.py --pred_dir output/<run>/structures/<eval_dir>
    python scripts/chi_angle_error.py --pred_dir ... --per_residue --out_csv chi.csv
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from glob import glob

import numpy as np

from protenix.data.constants import _CHI_ANGLES_ATOMS

# Chothia region ids shared with the eval (protenix/data/antibody_cdr.py).
CDR_REGIONS = {2, 4, 6}


def load_atoms(path: str) -> "dict[tuple[str, int], dict[str, np.ndarray]]":
    """Map (chain, res_id) -> {atom_name: xyz} for one structure."""
    if path.lower().endswith((".pdb", ".ent")):
        import biotite.structure.io.pdb as pdb

        arr = pdb.PDBFile.read(path).get_structure(model=1)
    else:
        import biotite.structure.io.pdbx as pdbx

        arr = pdbx.get_structure(pdbx.CIFFile.read(path), model=1)
    out: "dict[tuple[str, int], dict[str, np.ndarray]]" = defaultdict(dict)
    for ch, rid, nm, xyz in zip(arr.chain_id, arr.res_id, arr.atom_name, arr.coord):
        out[(str(ch), int(rid))][str(nm)] = xyz
    return out


def dihedral(p0, p1, p2, p3) -> float:
    """Signed dihedral about the p1-p2 axis, in degrees."""
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    n1 = np.cross(b0, b1)
    n2 = np.cross(b1, b2)
    m = np.cross(n1, b1 / (np.linalg.norm(b1) + 1e-12))
    x = float(np.dot(n1, n2))
    y = float(np.dot(m, n2))
    return float(np.degrees(np.arctan2(y, x)))


def wrap180(d: float) -> float:
    """Absolute angular difference wrapped to [0, 180]."""
    d = abs(d) % 360.0
    return 360.0 - d if d > 180.0 else d


def chi_errors(pred, native, res_name, key):
    """Per-chi |error| for one residue, or [] when atoms are missing."""
    quads = _CHI_ANGLES_ATOMS.get(res_name, [])
    out = []
    for i, names in enumerate(quads):
        pa, na = pred.get(key, {}), native.get(key, {})
        if any(n not in pa or n not in na for n in names):
            continue  # stripped or unresolved side chain
        dp = dihedral(*[pa[n] for n in names])
        dn = dihedral(*[na[n] for n in names])
        out.append((i, wrap180(dp - dn)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred_dir", required=True, help="eval structures/ subdirectory")
    ap.add_argument("--out_csv", default=None)
    ap.add_argument(
        "--cdr_only",
        action="store_true",
        help="restrict to CDR residues (needs <pid>_regions.json)",
    )
    ap.add_argument("--per_residue", action="store_true", help="also print by residue type")
    args = ap.parse_args()

    by_chi = defaultdict(list)
    by_res = defaultdict(list)
    rows = []
    n_struct = 0

    for meta_path in sorted(glob(os.path.join(args.pred_dir, "*_regions.json"))):
        with open(meta_path) as fh:
            meta = json.load(fh)
        pid = meta["pdb_id"]
        native_p = os.path.join(args.pred_dir, f"{pid}_native.cif")
        if not os.path.exists(native_p):
            continue
        native = load_atoms(native_p)
        # Residues to score; region info only needed for --cdr_only.
        keep = None
        if args.cdr_only:
            keep = {
                (r["chain"], r["res_id"])
                for r in meta["residues"]
                if r.get("region_type") in CDR_REGIONS and r.get("resolved")
            }
        rname = {
            (r["chain"], r["res_id"]): r["res_name"] for r in meta["residues"]
        }
        for pred_p in sorted(glob(os.path.join(args.pred_dir, f"{pid}_sample*.cif"))):
            pred = load_atoms(pred_p)
            n_struct += 1
            for key, rn in rname.items():
                if keep is not None and key not in keep:
                    continue
                for chi_idx, err in chi_errors(pred, native, rn, key):
                    by_chi[chi_idx].append(err)
                    by_res[rn].append(err)
                    rows.append(
                        {
                            "pdb_id": pid,
                            "sample": os.path.basename(pred_p),
                            "chain": key[0],
                            "res_id": key[1],
                            "res_name": rn,
                            "chi": chi_idx + 1,
                            "abs_err_deg": round(err, 3),
                        }
                    )

    if not rows:
        print("no chi angles measurable (side chains stripped, or no structures)")
        return

    print(f"structures: {n_struct}   measured angles: {len(rows)}")
    print(f"{'chi':<6}{'n':>9}{'MAE(deg)':>11}{'median':>9}{'<30deg':>9}")
    for i in sorted(by_chi):
        v = np.array(by_chi[i])
        print(
            f"chi{i+1:<3}{len(v):>9}{v.mean():>11.1f}{np.median(v):>9.1f}"
            f"{100*(v<30).mean():>8.0f}%"
        )
    allv = np.array([r["abs_err_deg"] for r in rows])
    print(f"{'ALL':<6}{len(allv):>9}{allv.mean():>11.1f}{np.median(allv):>9.1f}"
          f"{100*(allv<30).mean():>8.0f}%")
    print("\n(random guessing ~90 deg; lower is better)")

    if args.per_residue:
        print(f"\n{'res':<6}{'n':>8}{'MAE(deg)':>11}")
        for rn in sorted(by_res, key=lambda k: -np.mean(by_res[k])):
            v = np.array(by_res[rn])
            if len(v) >= 20:
                print(f"{rn:<6}{len(v):>8}{v.mean():>11.1f}")

    if args.out_csv:
        with open(args.out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {len(rows)} rows -> {args.out_csv}")


if __name__ == "__main__":
    main()
