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

"""Antibody CDR side-chain stripping for training-data preprocessing.

This mirrors the design-region handling in MFDesign
(https://github.com/yangnianzu0515/MFDesign): for antibody variable-domain
CDR residues, only backbone + CB atoms are kept, so that the per-residue atom
count / reference conformer cannot leak the identity of a residue the model is
being asked to design.

CDR boundaries follow the Chothia scheme, computed with ``abnumber`` (backed by
ANARCI/anarcii), exactly as MFDesign does. Chains that do not parse as an
antibody heavy/light variable domain are left untouched.
"""

import logging

import numpy as np
from biotite.structure import AtomArray, get_residue_starts

from protenix.data.constants import mmcif_restype_3to1

logger = logging.getLogger(__name__)

# Atoms retained for a stripped CDR residue (glycine simply has no CB present).
# Matches MFDesign SKELETON_ATOMS = ["N", "CA", "C", "O", "CB"].
BACKBONE_CB_ATOMS = frozenset({"N", "CA", "C", "O", "CB"})

# Chothia region order; the concatenation of the per-region sequences is the
# numbered Fv sequence (a contiguous substring of the input chain sequence).
_CHOTHIA_RANGES = ["fr1", "cdr1", "fr2", "cdr2", "fr3", "cdr3", "fr4"]
_CDR_RANGES = frozenset({"cdr1", "cdr2", "cdr3"})

# anarcii is a torch model; instantiate one CPU/single-thread instance and reuse
# it for every chain to avoid the thread oversubscription that occurs when each
# call spins up ncpu=-1 threads inside a joblib worker.
_ANARCII_ARGS = {"cpu": True, "ncpu": 1}


def _fv_cdr_mask(chain) -> "tuple[str, list[bool]] | tuple[None, None]":
    """Return (Fv sequence, per-residue CDR mask) for an abnumber Chain.

    Faithful to MFDesign ``mask_cdr``: build the Fv sequence by concatenating the
    Chothia region sequences in order and mark the CDR regions. Returns
    ``(None, None)`` if any region is empty (incomplete/over-long CDR3), matching
    MFDesign's behaviour of discarding such chains (here: no stripping).
    """
    origin, mask = [], []
    for region in _CHOTHIA_RANGES:
        seq = getattr(chain, region + "_seq")
        if len(seq) == 0:
            return None, None
        origin += list(seq)
        mask += [region in _CDR_RANGES] * len(seq)
    return "".join(origin), mask


def _chain_residue_cdr_flags(struct_seq: str, seq_cache: dict) -> "list[bool] | None":
    """Map an antibody chain's structure sequence to a per-residue CDR flag list.

    Returns ``None`` when the chain is not an antibody variable domain (or cannot
    be numbered), in which case the caller strips nothing for that chain.
    """
    if struct_seq in seq_cache:
        return seq_cache[struct_seq]

    result = None
    try:
        # Lazy import: abnumber pulls in torch, keep it out of module import.
        import abnumber

        chain = abnumber.Chain(
            struct_seq,
            scheme="chothia",
            use_anarcii=True,
            anarcii_args=_ANARCII_ARGS,
        )
        if chain.is_heavy_chain() or chain.is_light_chain():
            fv_seq, cdr_mask = _fv_cdr_mask(chain)
            if fv_seq is not None:
                offset = struct_seq.find(fv_seq)
                if offset >= 0:
                    flags = [False] * len(struct_seq)
                    for i, is_cdr in enumerate(cdr_mask):
                        if is_cdr:
                            flags[offset + i] = True
                    result = flags
                else:
                    logger.warning(
                        "Antibody Fv sequence not found within chain sequence; "
                        "skipping CDR stripping for this chain."
                    )
    except Exception as e:  # noqa: BLE001 - non-antibody chains raise here
        logger.debug("abnumber could not number chain (treated as non-antibody): %s", e)

    seq_cache[struct_seq] = result
    return result


def strip_cdr_side_chains(atom_array: AtomArray) -> "tuple[AtomArray, int]":
    """Remove side-chain atoms of antibody CDR residues from an AtomArray.

    For every protein chain that parses as an antibody heavy/light variable
    domain (Chothia numbering via abnumber), residues in CDR1/2/3 are reduced to
    backbone + CB atoms. All other atoms/chains are left untouched. A boolean
    per-atom ``is_cdr`` annotation marking atoms that belong to a CDR residue is
    added (set before stripping, so the retained backbone/CB atoms of CDR residues
    are flagged True).

    Args:
        atom_array (AtomArray): Bioassembly AtomArray. Must carry ``mol_type``,
            ``chain_id``, ``res_name`` and ``atom_name`` annotations.

    Returns:
        tuple[AtomArray, int]: The (possibly reduced) AtomArray and the number of
        atoms removed. When nothing is stripped the array is returned unchanged.
    """
    n_atoms = len(atom_array)
    if n_atoms == 0:
        return atom_array, 0

    res_starts = get_residue_starts(atom_array, add_exclusive_stop=True)

    keep_mask = np.ones(n_atoms, dtype=bool)
    is_cdr_atom = np.zeros(n_atoms, dtype=bool)
    seq_cache: dict = {}

    chain_ids = atom_array.chain_id
    mol_types = atom_array.mol_type
    res_names = atom_array.res_name
    atom_names = atom_array.atom_name

    # Group residue index-ranges by chain, preserving order of appearance.
    chain_residues: dict = {}
    for r in range(len(res_starts) - 1):
        start = res_starts[r]
        chain_residues.setdefault(chain_ids[start], []).append(
            (start, res_starts[r + 1])
        )

    for chain_id, residues in chain_residues.items():
        # Only consider protein chains.
        if mol_types[residues[0][0]] != "protein":
            continue

        struct_seq = "".join(
            mmcif_restype_3to1.get(res_names[start], "X") for start, _ in residues
        )
        # Antibody variable domains are ~110 residues; skip obviously-too-short.
        if len(struct_seq) < 70:
            continue

        cdr_flags = _chain_residue_cdr_flags(struct_seq, seq_cache)
        if cdr_flags is None:
            continue

        for (start, stop), is_cdr in zip(residues, cdr_flags):
            if not is_cdr:
                continue
            is_cdr_atom[start:stop] = True
            for a in range(start, stop):
                if atom_names[a] not in BACKBONE_CB_ATOMS:
                    keep_mask[a] = False

    n_removed = int((~keep_mask).sum())

    atom_array.set_annotation("is_cdr", is_cdr_atom)

    if n_removed == 0:
        return atom_array, 0

    return atom_array[keep_mask], n_removed


def add_is_cdr_residue_to_token_array(token_array, atom_array):
    """Add a per-token boolean ``is_cdr_residue`` annotation to a TokenArray.

    The flag is derived from the per-atom ``is_cdr`` annotation added by
    :func:`strip_cdr_side_chains`, using each token's centre atom (the CA for a
    standard residue, the single atom for a ligand/atomised token). The AtomArray
    must already carry the ``is_cdr`` annotation (call this after
    :func:`strip_cdr_side_chains`).

    Args:
        token_array (TokenArray): TokenArray produced from ``atom_array``; must
            carry the ``centre_atom_index`` annotation.
        atom_array (AtomArray): The AtomArray the tokens index into; must carry
            the ``is_cdr`` annotation.

    Returns:
        TokenArray: The same TokenArray, with an ``is_cdr_residue`` annotation.
    """
    is_cdr_atom = atom_array.is_cdr
    centre_atom_indices = token_array.get_annotation("centre_atom_index")
    is_cdr_residue = [bool(is_cdr_atom[i]) for i in centre_atom_indices]
    token_array.set_annotation("is_cdr_residue", is_cdr_residue)
    return token_array


def load_sabdab_chain_roles(csv_path) -> "dict[str, list[dict]]":
    """Load H/L/antigen:author_chain_ID mappings from MFDesign's processed summary CSV.

    Reuses the SAbDab chain identity already curated by MFDesign's step-1
    ``summary.py`` (H/L corrected via abnumber, filtered, deduplicated) instead of
    re-parsing the raw SAbDab TSV. Expected columns: ``pdb``, ``H_chain_id``,
    ``L_chain_id``, ``antigen_chain_id`` (a Python-list string, e.g. ``"['A']"``).

    Returns ``{pdb_id_lower: [{"H": str|None, "L": str|None, "antigen": [str, ...]}, ...]}``
    with SAbDab *author* chain ids. A PDB may occur in several rows (multiple
    antibodies); all are kept and disambiguated against the structure in
    :func:`resolve_sabdab_roles`.
    """
    import ast
    import csv as _csv

    def _clean(v):
        if v is None:
            return None
        v = str(v).strip()
        if v == "" or v.lower() in {"na", "nan", "none"}:
            return None
        return v

    def _parse_antigen(v):
        v = _clean(v)
        if v is None:
            return []
        try:
            parsed = ast.literal_eval(v)  # MFDesign stores a Python list repr
        except (ValueError, SyntaxError):
            # tolerate a bare or '|'/','-delimited string as well
            return sorted(
                s.strip() for s in v.replace("|", ",").split(",") if s.strip()
            )
        if isinstance(parsed, str):
            parsed = [parsed]
        return sorted(str(s).strip() for s in parsed if str(s).strip())

    roles: dict = {}
    with open(csv_path, newline="") as fh:
        reader = _csv.DictReader(fh)
        for row in reader:
            pdb = _clean(row.get("pdb"))
            if pdb is None:
                continue
            roles.setdefault(pdb.lower(), []).append(
                {
                    "H": _clean(row.get("H_chain_id")),
                    "L": _clean(row.get("L_chain_id")),
                    "antigen": _parse_antigen(row.get("antigen_chain_id")),
                }
            )
    return roles


def resolve_sabdab_roles(
    atom_array, entries: "list[dict]"
) -> "tuple[list[int], list[int], list[int]]":
    """Find the mmCIF chain indices of author_chain_id chains.

    SAbDab records chains by *author* id (e.g. ``H``/``L``/``A``), which carries no
    role information -- the structure only knows a chain called ``H`` exists, not that
    it is the heavy chain. This function injects that external SAbDab role knowledge and
    maps each author id to Protenix's integer chain index (``asym_id_int``) via the
    ``auth_asym_id`` annotation -- the direct analog of MFDesign storing the
    ``enumerate(chains)`` asym_id for H/L/antigen (mmcif.py:1250-1260). Chain indices are
    also what the cropper's ``ref_chain_indices`` expect. Assembly-expanded copies
    (chain_id containing ``.``) are skipped so the mapping lands on the base chains.

    Args:
        atom_array: bioassembly AtomArray (must carry ``auth_asym_id`` and
            ``asym_id_int``).
        entries: list of ``{"H","L","antigen"}`` dicts for this PDB (author ids).

    Returns:
        ``(heavy_idx, light_idx, antigen_idx)`` as lists of int chain indices. Returns
        empty lists when the SAbDab entry cannot be matched to the structure.
    """
    cats = atom_array.get_annotation_categories()
    if "auth_asym_id" not in cats or "asym_id_int" not in cats or not entries:
        return [], [], []

    # author chain id -> sorted list of asym_id_int chain indices, base chains only
    a2i: dict = {}
    for auth, label, asym_int in zip(
        atom_array.auth_asym_id, atom_array.chain_id, atom_array.asym_id_int
    ):
        if "." in str(label):
            continue
        a2i.setdefault(str(auth), set()).add(int(asym_int))
    a2i = {k: sorted(v) for k, v in a2i.items()}
    if not a2i:
        return [], [], []

    def _map(auth_ids):
        out: list = []
        for a in auth_ids:
            out.extend(a2i.get(str(a), []))
        return list(dict.fromkeys(out))  # de-dup, preserve order

    # For PDBs with several antibodies, pick the entry whose heavy chain is present.
    chosen = next(
        (e for e in entries if e.get("H") and a2i.get(str(e["H"]))),
        entries[0],
    )
    heavy = _map([chosen["H"]] if chosen.get("H") else [])
    light = _map([chosen["L"]] if chosen.get("L") else [])
    antigen = _map(chosen.get("antigen", []))
    return heavy, light, antigen
