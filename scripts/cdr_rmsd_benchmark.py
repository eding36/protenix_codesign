#!/usr/bin/env python3
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
like without re-running. The summary prints three per-target aggregates:
RANK0 (sample 0 -- the eval sorts designs by model confidence before writing, so
this is MFDesign's protocol and the number to compare against them), mean over
the N samples, and best-of-N (an oracle upper bound, since it selects using the
ground truth).

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


BACKBONE = {"N", "CA", "C", "O"}


def load_allatom(path: str) -> "dict[tuple[str, int], dict[str, np.ndarray]]":
    """Map (chain, res_id) -> {atom_name: coord} for ALL heavy atoms.

    Dihedrals need the backbone anchors too (chi1 is N-CA-CB-CG), so this cannot
    reuse load_sidechain, which deliberately drops backbone and CB.
    """
    import biotite.structure.io.pdb as biotite_pdb
    import biotite.structure.io.pdbx as biotite_pdbx

    if path.endswith(".pdb"):
        arr = biotite_pdb.PDBFile.read(path).get_structure(model=1)
    else:
        arr = biotite_pdbx.get_structure(biotite_pdbx.CIFFile.read(path), model=1)
    out: "dict[tuple[str, int], dict[str, np.ndarray]]" = {}
    for i in range(len(arr)):
        if str(arr.element[i]) == "H":
            continue
        c = arr.coord[i]
        if not np.isfinite(c).all() or float(np.linalg.norm(c)) < 1e-3:
            continue  # unresolved atoms are parked at the origin
        out.setdefault((str(arr.chain_id[i]), int(arr.res_id[i])), {})[
            str(arr.atom_name[i])
        ] = c
    return out


def load_sidechain(path: str) -> "dict[tuple[str, int], dict[str, np.ndarray]]":
    """Map (chain, res_id) -> {atom_name: coord} for SIDE-CHAIN atoms beyond CB.

    CB is excluded deliberately: preprocessing strips CDR side chains to
    backbone+CB, so including CB would let stripped CDR residues contribute a
    number that is really determined by the backbone. Beyond-CB atoms are exactly
    the ones a rotamer places, so their absence is what "stripped" means.
    """
    import biotite.structure.io.pdb as biotite_pdb
    import biotite.structure.io.pdbx as biotite_pdbx

    if path.endswith(".pdb"):
        arr = biotite_pdb.PDBFile.read(path).get_structure(model=1)
    else:
        arr = biotite_pdbx.get_structure(biotite_pdbx.CIFFile.read(path), model=1)
    out: "dict[tuple[str, int], dict[str, np.ndarray]]" = {}
    for i in range(len(arr)):
        name = str(arr.atom_name[i])
        if name in BACKBONE or name == "CB" or str(arr.element[i]) == "H":
            continue
        xyz = arr.coord[i]
        # The parser writes unresolved atoms at exactly (0,0,0) -- including one in
        # the RMSD puts a spurious ~30 A term in every affected residue. Dropping
        # them here is the same guard the chi metric needs.
        if not np.any(xyz) or np.isnan(xyz).any():
            continue
        out.setdefault((str(arr.chain_id[i]), int(arr.res_id[i])), {})[name] = xyz
    return out


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
    pred_sc, native_sc = load_sidechain(path), load_sidechain(native_cif)
    pred_all, native_all = load_allatom(path), load_allatom(native_cif)
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

    # 4. SIDE-CHAIN RMSD per region, on the same framework superposition.
    #
    # Atoms beyond CB only -- those are the ones a rotamer places. CDR side chains
    # are stripped to backbone+CB during preprocessing, so sc_n_cdr is expected to
    # be 0: the metric reports that explicitly rather than silently omitting the
    # region, because "we cannot measure designed side chains" is a result.
    # Framework / antigen / epitope are measurable and are where torsion noising
    # and the chi loss can actually act.
    groups = {
        "cdr": lambda r: r["chain_type"] in (1, 2) and r["region_type"] in (2, 4, 6),
        "framework": lambda r: r["chain_type"] in (1, 2)
        and r["region_type"] in FRAMEWORK_REGIONS,
        "antigen": lambda r: r["chain_type"] == 3,
        "epitope": lambda r: r["region_type"] == 9,
    }
    for gname, keep in groups.items():
        pv, nv = [], []
        nres = 0
        for r in residues:
            if not keep(r):
                continue
            k = (r["chain"], r["res_id"])
            pa, na = pred_sc.get(k), native_sc.get(k)
            if not pa or not na:
                continue
            shared = [a for a in pa if a in na]
            if not shared:
                continue
            nres += 1
            for a in shared:
                pv.append(rot @ pa[a] + trans)   # same superposition as the Ca metrics
                nv.append(na[a])
        out[f"sc_rmsd_{gname}"] = rmsd(np.stack(pv), np.stack(nv)) if pv else float("nan")
        out[f"sc_n_{gname}"] = len(pv)
        out[f"sc_res_{gname}"] = nres

    # 5. CHI MAE per region, degrees. Torsion noising and the chi loss act on side
    #    chain dihedrals, so this is the metric that shows whether they survived.
    #    Rotation-invariant: dihedrals need no superposition. Periodic chis (Asp1,
    #    Glu2, Phe1, Tyr1) are scored modulo 180 -- a flip is the same molecule.
    from protenix.data.constants import _CHI_ANGLES_ATOMS
    PI_PERIODIC = {("ASP", 1), ("GLU", 2), ("PHE", 1), ("TYR", 1)}

    def _dihedral(p0, p1, p2, p3):
        b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
        n1 = np.cross(b0, b1); n2 = np.cross(-b1, b2)
        n = np.linalg.norm(b1)
        if n < 1e-8:
            return None
        m = np.cross(n1, b1 / n)
        x = float(np.dot(n1, n2)); y = float(np.dot(m, n2))
        return np.degrees(np.arctan2(y, x))

    for gname, keep in groups.items():
        errs = []
        for r in residues:
            if not keep(r):
                continue
            rn = r.get("res_name")
            chis = _CHI_ANGLES_ATOMS.get(str(rn))
            if not chis:
                continue
            k = (r["chain"], r["res_id"])
            pa, na = pred_all.get(k), native_all.get(k)
            if not pa or not na:
                continue
            for ci, quad in enumerate(chis):
                if any(a not in pa or a not in na for a in quad):
                    continue
                dp = _dihedral(*[pa[a] for a in quad])
                dn = _dihedral(*[na[a] for a in quad])
                if dp is None or dn is None:
                    continue
                period = 180.0 if (str(rn), ci) in PI_PERIODIC else 360.0
                d = abs(dp - dn) % period
                errs.append(min(d, period - d))
        out[f"chi_mae_{gname}"] = float(np.mean(errs)) if errs else float("nan")
        out[f"chi_n_{gname}"] = len(errs)
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

    # Three per-target summaries, matching the three AAR numbers the eval reports.
    #
    #   RANK0  sample 0. The eval sorts designs by the model's confidence before
    #          writing them, so sample 0 is the highest-confidence design -- this is
    #          MFDesign's protocol (writer.py ranks by confidence_score,
    #          eval_codesign.py scores rank 0). Use THIS to compare against them.
    #   MEAN   average over the N samples of a target; no selection.
    #   BEST   minimum over the N samples. An ORACLE -- it picks using the answer,
    #          so it is an upper bound, not a protocol.
    by_target: "dict[str, dict[str, list]]" = {}
    for r in rows:
        by_target.setdefault(r["pdb_id"], {}).setdefault(int(r["sample"]), r)

    print(f"{'metric':<16}{'RANK0':>9}{'mean':>9}{'best':>9}{'n_tgt':>7}")
    print("-" * 50)
    for k in keys:
        if not (k.startswith("rmsd") or k.startswith("sc_rmsd")):
            continue
        # CDR side chains are stripped to backbone+CB during preprocessing, so
        # sc_rmsd_cdr can never be computed. The column stays in the CSV as an
        # explicit record; there is nothing to summarise.
        if k == "sc_rmsd_cdr":
            continue
        r0, mn, bs = [], [], []
        for _pid, samples in by_target.items():
            # NaN = that region is absent from this target (e.g. no antigen chain).
            # Averaging it in would poison the whole column.
            vals = [
                float(s[k]) for s in samples.values()
                if k in s and not np.isnan(float(s[k]))
            ]
            if not vals:
                continue
            if 0 in samples and k in samples[0] and not np.isnan(float(samples[0][k])):
                r0.append(float(samples[0][k]))
            mn.append(float(np.mean(vals)))
            bs.append(float(np.min(vals)))
        if not mn:
            continue
        r0v = np.mean(r0) if r0 else float("nan")
        print(f"{k:<16}{r0v:>9.3f}{np.mean(mn):>9.3f}{np.mean(bs):>9.3f}{len(mn):>7}")
    n_s = {len(v) for v in by_target.values()}
    print(f"\nsamples per target: {sorted(n_s)}"
          + ("   (RANK0 == mean == best with 1 sample)" if n_s == {1} else ""))


if __name__ == "__main__":
    main()
