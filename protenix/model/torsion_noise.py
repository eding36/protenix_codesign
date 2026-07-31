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

"""Torsion-space (bond-length preserving) noising for structure diffusion.

Isotropic Gaussian noise in Cartesian space destroys covalent geometry: bond
lengths, bond angles and chirality are all violated, so a large share of the
denoiser's capacity goes into restoring local geometry rather than modelling
conformational change.

This module noises structures by **rotating about rotatable bonds** instead. A
rotation of the downstream atom set about the axis through a bond changes only
torsion angles: every bond length is preserved exactly (the moved substructure is
rigid, and the one bond crossing the partition lies on the rotation axis), and
every bond angle is preserved for the same reason. The output is still a
coordinate tensor, so the rest of the diffusion pipeline is unchanged.

Scope: side-chain ``chi`` torsions of standard amino acids. Backbone ``phi``/``psi``
are deliberately excluded -- rotating a backbone torsion moves the entire
downstream chain, so the coordinate displacement per radian grows with the lever
arm and is wildly non-uniform along the sequence. Callers combine this with
ordinary Gaussian noise on the atoms this module does not move (see
``TorsionNoiser.residual_mask``).
"""

from typing import Optional

import numpy as np
import torch

from protenix.data.constants import ATOM14

BACKBONE_ATOMS = ("N", "CA", "C", "O")

# Side-chain atoms in canonical (proximal -> distal) order per residue.
SC_ORDER = {
    res: tuple(a for a in atoms if a not in BACKBONE_ATOMS)
    for res, atoms in ATOM14.items()
}


def build_torsion_index(
    atom_names: np.ndarray,
    res_names: np.ndarray,
    residue_starts: np.ndarray,
    n_atoms: int,
    bonds: Optional[np.ndarray] = None,
) -> "list[tuple[int, int, np.ndarray]]":
    """Enumerate rotatable side-chain bonds and the atoms they move.

    For a residue whose side chain is ``[CB, CG, CD, ...]`` (canonical order), the
    bond ``(a_i, a_{i+1})`` is rotatable and moves ``a_{i+2:}``. ``chi1`` is the
    CA-CB bond, which moves ``CG`` onward.

    Candidates are then **validated against the real bond graph**: rotating the
    downstream set is only bond-length preserving if every bond crossing the
    partition has its stationary endpoint *on the rotation axis* (i.e. it is one of
    the two pivots). This automatically rejects the cases where the canonical
    linear ordering is not the true topology:

    * proline -- the side chain rings back to the backbone N;
    * tryptophan / histidine / phenylalanine ring closures inside the side chain;
    * cystine -- a disulfide bonds the side chain to another residue.

    Without ``bonds`` the validation is skipped and those residues will have their
    rings torn open, so callers should always pass the connectivity.

    Args:
        atom_names / res_names: ``[N_atom]`` per-atom metadata.
        residue_starts: residue boundaries with an exclusive stop.
        n_atoms: total atom count.
        bonds: ``[N_bond, 2]`` atom-index pairs used to validate each candidate.

    Returns:
        List of ``(pivot_a, pivot_b, moved_indices)``: rotate ``moved_indices``
        about the axis through atoms ``pivot_a -> pivot_b``.
    """
    torsions = []
    candidates: "list[tuple[int, int]]" = []
    for r in range(len(residue_starts) - 1):
        start, stop = residue_starts[r], residue_starts[r + 1]
        order = SC_ORDER.get(res_names[start])
        if not order or len(order) < 2:
            continue  # Gly/Ala: nothing rotatable
        names = list(atom_names[start:stop])
        idx_of = {nm: start + k for k, nm in enumerate(names)}
        if "CA" not in idx_of:
            continue
        # Chain of pivots: (CA, CB), (CB, CG), (CG, CD), ...
        chain = ["CA"] + [a for a in order if a in idx_of]
        candidates.extend(
            (idx_of[chain[i]], idx_of[chain[i + 1]]) for i in range(len(chain) - 2)
        )

    if bonds is None or len(bonds) == 0:
        return []  # without connectivity we cannot build correct downstream sets
    adjacency = _adjacency(bonds, n_atoms)
    for a, b in candidates:
        moved = _downstream(a, b, adjacency)
        if moved is not None and moved.size:
            torsions.append((a, b, moved))
    return torsions


def _adjacency(bonds: np.ndarray, n_atoms: int) -> "list[list[int]]":
    adj: "list[list[int]]" = [[] for _ in range(n_atoms)]
    for i, j in bonds[:, :2]:
        adj[int(i)].append(int(j))
        adj[int(j)].append(int(i))
    return adj


def _downstream(
    a: int, b: int, adjacency: "list[list[int]]"
) -> Optional[np.ndarray]:
    """Atoms reachable from ``b`` without crossing back through ``a``.

    This is the true torsion subtree, so rotating it about the ``a -> b`` axis
    changes only the torsion: every bond length *and* every bond angle is
    preserved, because each moved atom keeps its distance to the axis and the only
    stationary neighbours of the moved set are the pivots themselves.

    Returns ``None`` when ``a`` is reachable from ``b`` without using the ``a-b``
    bond -- that means the bond lies in a ring (proline's N-CD closure, aromatic
    side chains, a disulfide bridged through the backbone) and cannot be rotated
    without tearing the ring open.
    """
    seen = {b}
    stack = [b]
    while stack:
        cur = stack.pop()
        for nxt in adjacency[cur]:
            if nxt == a and cur == b:
                continue  # the torsion bond itself
            if nxt == a:
                return None  # cycle back to the pivot -> ring bond, not rotatable
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    seen.discard(b)
    return np.fromiter(seen, dtype=np.int64, count=len(seen))


def rotate_about_axis(
    coords: torch.Tensor,
    pivot_a: torch.Tensor,
    pivot_b: torch.Tensor,
    angle: torch.Tensor,
) -> torch.Tensor:
    """Rodrigues rotation of ``coords`` about the line ``pivot_a -> pivot_b``.

    Args:
        coords: ``[..., K, 3]`` points to rotate.
        pivot_a / pivot_b: ``[..., 3]`` two points defining the axis.
        angle: ``[...]`` rotation angle in radians.

    Returns:
        Rotated ``[..., K, 3]``. Distances to the axis are preserved, so all bond
        lengths and bond angles involving the axis are exactly preserved.
    """
    axis = pivot_b - pivot_a
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    v = coords - pivot_b[..., None, :]
    k = axis[..., None, :]
    cos_t = torch.cos(angle)[..., None, None]
    sin_t = torch.sin(angle)[..., None, None]
    dot = (v * k).sum(-1, keepdim=True)
    rotated = v * cos_t + torch.cross(k.expand_as(v), v, dim=-1) * sin_t + k * dot * (1 - cos_t)
    return rotated + pivot_b[..., None, :]


class TorsionNoiser:
    """Apply torsion-space noise to a coordinate tensor.

    The noise magnitude is expressed in the same units the EDM schedule uses
    (Angstrom of resulting coordinate displacement) and converted to a per-torsion
    angular scale, so a given ``sigma`` produces roughly the same RMSD as Gaussian
    noise would. This keeps the EDM preconditioning (which assumes
    ``x_noisy ~ x + sigma * eps``) approximately valid when the two noising modes
    are mixed during training.

    Built once per sample from the atom array; the index is static.
    """

    def __init__(
        self,
        atom_names: np.ndarray,
        res_names: np.ndarray,
        residue_starts: np.ndarray,
        n_atoms: int,
        max_angle: float = np.pi,
        bonds: Optional[np.ndarray] = None,
    ) -> None:
        self.torsions = build_torsion_index(
            atom_names, res_names, residue_starts, n_atoms, bonds=bonds
        )
        self.n_atoms = n_atoms
        self.max_angle = max_angle
        moved = np.zeros(n_atoms, dtype=bool)
        for _, _, m in self.torsions:
            moved[m] = True
        # Atoms no torsion can move (backbone, Gly/Ala side chains, ligands, ions,
        # nucleic acids). Callers apply ordinary Gaussian noise to these.
        self.residual_mask = torch.from_numpy(~moved)
        self.moved_mask = torch.from_numpy(moved)

    def __len__(self) -> int:
        return len(self.torsions)

    def __call__(
        self,
        coords: torch.Tensor,
        sigma: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Noise ``coords`` by perturbing torsions.

        Args:
            coords: ``[..., N_atom, 3]``
            sigma: ``[...]`` noise level (Angstrom), broadcast over the batch.

        Returns:
            ``[..., N_atom, 3]`` with every bond length and bond angle preserved
            exactly for the atoms moved by torsions.
        """
        if not self.torsions:
            return coords
        out = coords.clone()
        # Map sigma (A) -> angular scale. A torsion rotation displaces a moved atom
        # by ~ r * theta with r its distance to the axis (~1.5-3 A), so theta ~
        # sigma / r_typ. Clamped so high-sigma steps stay within a full turn.
        theta_scale = (sigma / 2.0).clamp(max=self.max_angle)
        for pivot_a, pivot_b, moved in self.torsions:
            idx = torch.as_tensor(moved, device=coords.device)
            shape = theta_scale.shape
            angle = torch.randn(shape, device=coords.device, dtype=coords.dtype,
                                generator=generator) * theta_scale
            out[..., idx, :] = rotate_about_axis(
                out[..., idx, :],
                out[..., pivot_a, :],
                out[..., pivot_b, :],
                angle,
            )
        return out
