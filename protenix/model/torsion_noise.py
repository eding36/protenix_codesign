"""Torsion-space noising for structure diffusion.

Adding Gaussian noise to atom coordinates breaks the molecule: bonds stretch and
angles distort, so the denoiser spends much of its capacity just rebuilding valid
chemistry. This module instead noises a structure the way a real side chain moves, 
by twisting it around its rotatable bonds.

Bond lengths and angles don't change, so the noised structure is still physically viable.
Applying torsion noise still returns noised coordinates, so nothing downstream needs to change.

Only side-chain torsions are applied. Twisting a backbone bond would move the entire
rest of the chain, displacing distant atoms tremendously for a small angular change.
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
#e.g. {"ALA":("CB","CG",...all side chain atoms)}


def build_torsion_index(
    atom_names: np.ndarray,
    res_names: np.ndarray,
    residue_starts: np.ndarray,
    n_atoms: int,
    bonds: Optional[np.ndarray] = None,
) -> "list[tuple[int, int, np.ndarray]]":
    """Finds every rotatable side-chain bond, along with the specific atoms it moves in 3D space when rotated.

    A side chain is a short arm of atoms, so every bond along it is a candidate
    rotatable side chain bond. Disulfide bridges, prolines, and aromatics are exceptions that cannot
    actually rotate their bonds so they are excluded. Both are detected from the
    structure's real connectivity and dropped which is why ``bonds`` is
    required.

    Args:
        atom_names / res_names: ``[N_atom]`` per-atom metadata.
        residue_starts: list of atom indices marking the start of a new residue. len(residue_starts) = # residues
        n_atoms: total atom count.
        bonds: ``[N_bond, 2]`` atom-index pairs.

    Returns:
        One entry per usable torsion: the two atoms defining the rotation axis, the
        atoms that swing with it, and its depth (chi1, chi2, ...). Torsions at the
        same depth never overlap, so they can all be rotated together.
    """
    torsions = []
    candidates: "list[tuple[int, int, int]]" = []
    for r in range(len(residue_starts) - 1): 
        start, stop = residue_starts[r], residue_starts[r + 1] #fetches all atom indices for that residue
        order = SC_ORDER.get(res_names[start]) #get the side chain atom list for the selected residue
        if not order or len(order) < 2:
            continue  # Gly/Ala: nothing rotatable
        names = list(atom_names[start:stop]) #[CA,CB,CG,CD]
        idx_of = {nm: start + k for k, nm in enumerate(names)} #{CA:n, CB:n+1,...}
        if "CA" not in idx_of:
            continue
        
        chain = ["CA"] + [a for a in order if a in idx_of] #[CA,CB,CG,CD]
        # depth i == chi(i+1); torsions at one depth move disjoint atom sets.
        candidates.extend(
            (idx_of[chain[i]], idx_of[chain[i + 1]], i) for i in range(len(chain) - 2)
        ) # e.g: [(CA, CB, 0), (CB, CG, 1), (CG, CD,2), ...]

    if bonds is None or len(bonds) == 0:
        return []  # without connectivity information we cannot build correct downstream sets
    adjacency = _adjacency(bonds, n_atoms)
    # Per-atom residue index, so _downstream can reject subtrees that escape the
    # residue through a cross-link (disulfides) rather than walking the whole chain.
    atom_residue = np.zeros(n_atoms, dtype=np.int64)
    for r in range(len(residue_starts) - 1):
        atom_residue[residue_starts[r] : residue_starts[r + 1]] = r #[N_atoms], contains the residue index for each atom.
    for a, b, depth in candidates:
        moved = _downstream(a, b, adjacency, atom_residue, int(atom_residue[a]))
        if moved is not None and moved.size:
            torsions.append((a, b, moved, depth))
    return torsions


def torsion_index_to_tensors(
    torsions: "list[tuple[int, int, np.ndarray, int]]",
) -> "dict[str, torch.Tensor]":
    """This function is fed into the Featurizer.get_all_input_features() method, and turns torsions
    into tensors fed into input_feature_dict, features used during training through apply_torsion_noise() function below
    

    Returns:
        torsion_pivot   [T, 2]  atoms at the end of each rotatable bond
        torsion_depth   [T]     chi level (0 = chi1)
        torsion_atom    [K]     atom indices moved by torsion 
        torsion_atom_id [K]     which torsion T each ``torsion_atom`` entry belongs to
    """
    if not torsions:
        return {
            "torsion_pivot": torch.zeros((0, 2), dtype=torch.long),
            "torsion_depth": torch.zeros((0,), dtype=torch.long),
            "torsion_atom": torch.zeros((0,), dtype=torch.long),
            "torsion_atom_id": torch.zeros((0,), dtype=torch.long),
        }
    pivots, depths, atoms, atom_ids = [], [], [], []
    for tid, (a, b, moved, depth) in enumerate(torsions):
        pivots.append((a, b))
        depths.append(depth)
        atoms.append(moved)
        atom_ids.append(np.full(moved.shape, tid, dtype=np.int64))
    return {
        "torsion_pivot": torch.tensor(pivots, dtype=torch.long),
        "torsion_depth": torch.tensor(depths, dtype=torch.long),
        "torsion_atom": torch.from_numpy(np.concatenate(atoms)),
        "torsion_atom_id": torch.from_numpy(np.concatenate(atom_ids)),
    }


def apply_torsion_noise(
    coords: torch.Tensor,
    torsion_pivot: torch.Tensor,
    torsion_depth: torch.Tensor,
    torsion_atom: torch.Tensor,
    torsion_atom_id: torch.Tensor,
    sigma: torch.Tensor,
    max_angle: float = float(np.pi),
) -> torch.Tensor:
    """Twist every torsion by a random angle scaled to ``sigma``. 
    Used during /home/dinge/Protenix/protenix/model/generator.py:sample_diffusion_training()

    Hinges are nested like joints in an arm: the shoulder moves the elbow, so
    rotations are applied one level at a time (chi1, then chi2, ...) and each level
    twists about an axis the previous one has already repositioned. Within a level
    the joints are independent, so they all turn at once.

    Args:
        coords: ``[..., N_atom, 3]``
        sigma: ``[...]`` noise level in Angstrom, broadcast over leading dims.

    Returns:
        ``[..., N_atom, 3]`` with all bond lengths and bond angles preserved.
    """
    if torsion_pivot.numel() == 0:
        return coords
    out = coords.clone()
    n_torsion = torsion_pivot.shape[0]
    # sigma (A) -> angular scale: a moved atom sits ~2 A from its axis, so a
    # rotation of theta displaces it by ~2*theta.
    theta_scale = (sigma / 2.0).clamp(max=max_angle)
    angles = torch.randn(
        (*theta_scale.shape, n_torsion), device=coords.device, dtype=coords.dtype
    ) * theta_scale[..., None] #randomly sample rotational angle

    for depth in torsion_depth.unique(sorted=True):
        at_depth = torsion_depth == depth
        entries = at_depth[torsion_atom_id]
        if not bool(entries.any()):
            continue
        aidx = torsion_atom[entries]
        tid = torsion_atom_id[entries]
        pa = out[..., torsion_pivot[tid, 0], :]
        pb = out[..., torsion_pivot[tid, 1], :]
        ang = angles[..., tid]
        axis = pb - pa
        axis = axis / axis.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        v = out[..., aidx, :] - pb
        cos_t = torch.cos(ang)[..., None]
        sin_t = torch.sin(ang)[..., None]
        dot = (v * axis).sum(-1, keepdim=True)
        out[..., aidx, :] = (
            v * cos_t
            + torch.cross(axis, v, dim=-1) * sin_t
            + axis * dot * (1 - cos_t)
            + pb
        )
    return out


def _adjacency(bonds: np.ndarray, n_atoms: int) -> "list[list[int]]":
    """Builds list of atom indices bonded to each atom """
    adj: "list[list[int]]" = [[] for _ in range(n_atoms)]
    for i, j in bonds[:, :2]:
        adj[int(i)].append(int(j))
        adj[int(j)].append(int(i))
    return adj


def _downstream(
    a: int,
    b: int,
    adjacency: "list[list[int]]",
    atom_residue: Optional[np.ndarray] = None,
    residue: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Starting at ``b`` and never traversing back through ``a``, 
    this is a depth first search implementation that collects
    all the atoms that move in coordinate space when the ``a-b`` 
    bond is twisted. Returns ``None`` when the bond cannot rotate:

    * Two "None" cases: the walk loops back around to ``a`` -- the bond is part of a ring (proline / aromatics)
    * the walk reaches another residue: a disulfide bond connecting two different cysteines 
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
                if atom_residue is not None and atom_residue[nxt] != residue:
                    return None  # cross-link out of the residue (disulfide)
                seen.add(nxt)
                stack.append(nxt)
    seen.discard(b)
    return np.fromiter(seen, dtype=np.int64, count=len(seen))
