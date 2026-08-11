#!/usr/bin/env python3
"""Build side chains onto designed CDRs, offline, after eval.

CDR side chains are stripped to backbone+CB during preprocessing, because a
residue's atom count and reference conformer would otherwise give away the
identity the model is designing. The model therefore predicts CDR sequence and
backbone but has no side-chain atoms to place. This adds them afterwards.

For each predicted structure: mutate every designed CDR position to the residue
the model chose, repack rotamers there with the backbone fixed, and optionally
run a coordinate-constrained relax.

ON RELAX: MFDesign's paper describes packing and relaxing with PyRosetta, but no
such step exists in their released code, so this is not a port -- it is the
standard protocol. Two flags:

  (default)        pack only. PackRotamersMover changes rotamers, never the
                   backbone, so Ca RMSD is provably unchanged.
  --relax          adds FastRelax with -relax:constrain_relax_to_start_coords,
                   which tethers the backbone to its input position. Fixes
                   clashes without remodelling.

Do NOT relax without constraints. Measured on this data, an unconstrained
FastRelax minimises Rosetta energy over the whole backbone and made CDR RMSD
WORSE (H3 6.11 -> 7.42 A), because a shallow loop landscape lets it drift.

Needs PyRosetta, which lives in its own env:
    /home/dinge/miniconda3/envs/pyrosetta/bin/python scripts/pack_cdr_sidechains.py ...

Usage:
    python scripts/pack_cdr_sidechains.py \\
        --pred_dir output/<run>/structures/<eval_dir> --out_dir packed/
    python scripts/pack_cdr_sidechains.py --pred_dir ... --out_dir ... --relax
"""

import argparse
import json
import os
from glob import glob

# Chothia region ids (protenix/data/antibody_cdr.py): cdr1/2/3 = 2/4/6.
CDR_REGIONS = {2, 4, 6}
# chain_type 1 = heavy, 2 = light; antigen is never designed.
AB_CHAINS = {1, 2}

ONE_TO_THREE = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE", "G": "GLY",
    "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU", "M": "MET", "N": "ASN",
    "P": "PRO", "Q": "GLN", "R": "ARG", "S": "SER", "T": "THR", "V": "VAL",
    "W": "TRP", "Y": "TYR",
}


def read_designed_sequences(seq_files) -> "dict[str, str]":
    """pdb_id -> designed sequence, from the eval's .seq TSVs (one per rank)."""
    out = {}
    for f in seq_files:
        with open(f) as fh:
            next(fh, None)  # header
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    out[parts[0]] = parts[1]
    return out


def cif_to_pdb(cif_path: str, pdb_path: str) -> None:
    """PyRosetta rejects the minimal CIF biotite emits; round-trip through PDB.

    Also drops everything that is not a standard amino acid. The eval dumps
    ligands, glycans and ions as per-atom UNK tokens, which PyRosetta refuses to
    read ("Unrecognized residue: X") and which would otherwise confuse the packer
    task. They are not part of the designed antibody, so removing them is safe.
    """
    import numpy as np
    import biotite.structure.io.pdb as biotite_pdb
    import biotite.structure.io.pdbx as biotite_pdbx

    arr = biotite_pdbx.get_structure(biotite_pdbx.CIFFile.read(cif_path), model=1)
    arr = arr[np.isin(arr.res_name, list(ONE_TO_THREE.values()))]
    arr.bonds = None
    f = biotite_pdb.PDBFile()
    f.set_structure(arr)
    f.write(pdb_path)


def pack_one(pred_cif, meta, designed_seq, out_pdb, tmp_pdb, repack_neighbors, do_relax):
    """Mutate designed CDR positions, repack, optionally relax. Returns #positions."""
    import pyrosetta
    from pyrosetta.rosetta.core.pack.task import TaskFactory
    from pyrosetta.rosetta.core.pack.task.operation import (
        IncludeCurrent,
        RestrictToRepacking,
    )
    from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover
    from pyrosetta.rosetta.protocols.simple_moves import MutateResidue

    residues = meta["residues"]
    cif_to_pdb(pred_cif, tmp_pdb)
    pose = pyrosetta.pose_from_file(tmp_pdb)
    info = pose.pdb_info()

    # regions.json lists residues in sequence order, so index i of the designed
    # sequence corresponds to residues[i].
    targets = []
    for i, r in enumerate(residues):
        if r.get("region_type") not in CDR_REGIONS or r.get("chain_type") not in AB_CHAINS:
            continue
        if i >= len(designed_seq):
            continue
        aa1 = designed_seq[i]
        if aa1 not in ONE_TO_THREE:
            continue  # X / non-standard
        seqpos = info.pdb2pose(str(r["chain"]), int(r["res_id"]))
        if seqpos == 0:
            continue  # absent from this structure
        targets.append((seqpos, ONE_TO_THREE[aa1]))

    if not targets:
        return 0

    for seqpos, aa3 in targets:
        MutateResidue(seqpos, aa3).apply(pose)

    designed = {s for s, _ in targets}
    keep = set(designed)
    if repack_neighbors:
        import pyrosetta.rosetta.core.select.residue_selector as rs

        sel = rs.ResidueIndexSelector(",".join(str(s) for s in sorted(designed)))
        nbr = rs.NeighborhoodResidueSelector(sel, 6.0, True)
        keep = {i + 1 for i, v in enumerate(nbr.apply(pose)) if v}

    tf = TaskFactory()
    tf.push_back(RestrictToRepacking())  # repack only -- never redesign
    tf.push_back(IncludeCurrent())
    task = tf.create_task_and_apply_taskoperations(pose)
    for i in range(1, pose.total_residue() + 1):
        if i not in keep:
            task.nonconst_residue_task(i).prevent_repacking()

    scorefxn = pyrosetta.get_fa_scorefxn()
    PackRotamersMover(scorefxn, task).apply(pose)

    if do_relax:
        from pyrosetta.rosetta.protocols.relax import FastRelax

        fr = FastRelax(scorefxn, 1)
        fr.constrain_relax_to_start_coords(True)  # tether backbone to input
        fr.ramp_down_constraints(False)           # keep the tether to the end
        fr.apply(pose)

    pose.dump_pdb(out_pdb)
    return len(targets)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred_dir", required=True, help="eval structures/ subdirectory")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--seq_file", nargs="*", default=None,
        help="designed-sequence .seq file(s); default: all under ../../predictions",
    )
    ap.add_argument(
        "--repack_neighbors", action="store_true",
        help="also repack side chains within 6 A of a designed position",
    )
    ap.add_argument(
        "--relax", action="store_true",
        help="coordinate-constrained FastRelax after packing (slow; backbone tethered)",
    )
    args = ap.parse_args()

    seq_files = args.seq_file
    if not seq_files:
        guess = os.path.join(os.path.dirname(os.path.dirname(args.pred_dir)), "predictions")
        seq_files = sorted(glob(os.path.join(guess, "*.seq")))
    if not seq_files:
        raise SystemExit("no .seq files found; pass --seq_file")
    designed = read_designed_sequences(seq_files)
    print(f"designed sequences: {len(designed)} complexes (from {len(seq_files)} file(s))")
    print(f"mode: pack{' + constrained relax' if args.relax else ' only'}")

    import pyrosetta

    pyrosetta.init("-mute all -ex1 -ex2aro", silent=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tmp = os.path.join(args.out_dir, "_tmp_in.pdb")

    n_done = n_skip = n_pos = 0
    for meta_path in sorted(glob(os.path.join(args.pred_dir, "*_regions.json"))):
        with open(meta_path) as fh:
            meta = json.load(fh)
        pid = meta["pdb_id"]
        if pid not in designed:
            print(f"[skip] {pid}: no designed sequence")
            n_skip += 1
            continue
        for pred in sorted(glob(os.path.join(args.pred_dir, f"{pid}_sample*.cif"))):
            out = os.path.join(
                args.out_dir, os.path.basename(pred).replace(".cif", "_packed.pdb")
            )
            try:
                k = pack_one(pred, meta, designed[pid], out, tmp,
                             args.repack_neighbors, args.relax)
            except Exception as exc:  # noqa: BLE001 - report and continue
                print(f"[warn] {os.path.basename(pred)}: {type(exc).__name__}: {exc}")
                n_skip += 1
                continue
            n_done += 1
            n_pos += k
            print(f"{os.path.basename(out)}: packed {k} CDR positions")

    if os.path.exists(tmp):
        os.remove(tmp)
    print(f"\npacked {n_done} structures ({n_pos} CDR positions), {n_skip} skipped")
    print(f"-> {args.out_dir}")
    if args.relax:
        print("Relax was coordinate-constrained; verify Ca RMSD is unchanged with")
        print("  python scripts/cdr_rmsd_benchmark.py --pred_dir <out_dir> --no_relax")
    else:
        print("Backbone untouched, so CDR RMSD is unchanged by construction.")


if __name__ == "__main__":
    main()
