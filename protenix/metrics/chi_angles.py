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

"""Side-chain chi-angle error, for judging whether the model learns torsions.

The structure losses cannot answer this on their own. ``mse_loss`` is Cartesian,
so a correct side chain rotated as a whole scores as badly as a mangled one, and
``bond_loss`` measures covalent-geometry violation -- which torsion noising
preserves by construction, leaving it near zero whatever the model learned. This
compares the dihedral itself.

Errors are wrapped to [0, 180] degrees, so ~90 is what random guessing gives.

Note on coverage: CDR side chains are stripped to backbone+CB during
preprocessing (their atom count would otherwise reveal the residue identity the
model is designing), so CDR positions have no chi to measure and ``chi/cdr_*``
is normally empty. What is measured is framework and antigen -- reported
separately, since the epitope side chains are the ones a CDR backbone packs
against.
"""

from typing import Optional

import numpy as np
import torch

from protenix.data.constants import _CHI_ANGLES_ATOMS

# Chothia region ids (protenix/data/antibody_cdr.py).
_CDR_REGIONS = (2, 4, 6)          # cdr1/2/3 -- side chains are stripped, so these
                                  # contribute no measurable chi (see module note)
_FRAMEWORK_REGIONS = (1, 3, 5, 7)  # fr1..fr4
_ANTIGEN_REGIONS = (8, 9)          # 8 = antigen, 9 = epitope
_EPITOPE_REGIONS = (9,)


def _dihedral(p: torch.Tensor) -> torch.Tensor:
    """Signed dihedral of ``[..., 4, 3]`` point sets, in degrees."""
    b0 = p[..., 0, :] - p[..., 1, :]
    b1 = p[..., 2, :] - p[..., 1, :]
    b2 = p[..., 3, :] - p[..., 2, :]
    n1 = torch.cross(b0, b1, dim=-1)
    n2 = torch.cross(b1, b2, dim=-1)
    b1n = b1 / b1.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    m = torch.cross(n1, b1n, dim=-1)
    return torch.rad2deg(
        torch.atan2((m * n2).sum(-1), (n1 * n2).sum(-1))
    )


def chi_angle_errors(
    atom_array,
    pred_coord: torch.Tensor,
    gt_coord: torch.Tensor,
    region_per_atom: Optional[torch.Tensor] = None,
) -> "dict[str, float]":
    """Mean |chi error| between predicted and ground-truth side chains.

    Args:
        atom_array: cropped AtomArray, for atom/residue names and residue ids.
        pred_coord: ``[..., N_atom, 3]``; leading dims are averaged over.
        gt_coord: ``[..., N_atom, 3]``.
        region_per_atom: optional ``[N_atom]`` Chothia region id, used to report
            the CDR-only subset separately.

    Returns:
        ``chi/mae``, ``chi/chi{1..4}``, ``chi/frac_below_30`` and, when regions are
        given, the same under ``chi/cdr_*``. Empty when no chi is measurable (all
        side chains stripped or unresolved).
    """
    names = np.asarray(atom_array.atom_name)
    resn = np.asarray(atom_array.res_name)
    resi = np.asarray(atom_array.res_id)
    chain = np.asarray(atom_array.chain_id)

    # (chain, res_id) -> {atom_name: atom_index}
    lookup: "dict[tuple, dict[str, int]]" = {}
    for i in range(len(names)):
        lookup.setdefault((chain[i], int(resi[i])), {})[str(names[i])] = i

    quads: "list[list[int]]" = []
    chi_idx: "list[int]" = []
    region_of: "list[int]" = []
    for key, amap in lookup.items():
        first = next(iter(amap.values()))
        for ci, atoms in enumerate(_CHI_ANGLES_ATOMS.get(str(resn[first]), [])):
            if any(a not in amap for a in atoms):
                continue  # stripped or unresolved side chain
            idx = [amap[a] for a in atoms]
            quads.append(idx)
            chi_idx.append(ci)
            # Bucket by the region of the atom the torsion pivots on.
            reg = int(region_per_atom[idx[2]]) if region_per_atom is not None else -1
            region_of.append(reg)
    if not quads:
        return {}

    q = torch.as_tensor(quads, dtype=torch.long, device=pred_coord.device)
    n_atom = gt_coord.shape[-2]
    p = pred_coord.reshape(-1, n_atom, 3).float()
    g = gt_coord.reshape(-1, n_atom, 3)[0].float()

    err = (_dihedral(p[:, q]) - _dihedral(g[q])).abs() % 360.0
    err = torch.where(err > 180.0, 360.0 - err, err).mean(dim=0)  # over samples

    ci = torch.as_tensor(chi_idx, device=err.device)
    reg = torch.as_tensor(region_of, device=err.device)
    in_ = lambda ids: torch.isin(reg, torch.as_tensor(ids, device=err.device))
    cdr, fw = in_(_CDR_REGIONS), in_(_FRAMEWORK_REGIONS)
    ag, ep = in_(_ANTIGEN_REGIONS), in_(_EPITOPE_REGIONS)

    def _summarise(mask: torch.Tensor, prefix: str) -> "dict[str, float]":
        if not bool(mask.any()):
            return {}
        sub = err[mask]
        out = {
            f"{prefix}mae": float(sub.mean()),
            f"{prefix}frac_below_30": float((sub < 30.0).float().mean()),
            f"{prefix}n": float(sub.numel()),
        }
        for k in range(4):
            m = mask & (ci == k)
            if bool(m.any()):
                out[f"{prefix}chi{k + 1}"] = float(err[m].mean())
        return out

    metrics = _summarise(torch.ones_like(cdr, dtype=torch.bool), "chi/")
    for mask, prefix in (
        (cdr, "chi/cdr_"),
        (fw, "chi/framework_"),
        (ag, "chi/antigen_"),
        (ep, "chi/epitope_"),
    ):
        metrics.update(_summarise(mask, prefix))
    return metrics
