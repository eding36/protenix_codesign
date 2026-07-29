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

CDR boundaries follow the Chothia scheme. They are read from MFDesign's curated
per-chain CDR masks (``*_chain_masked_seq``: ``X`` at CDR positions) when a chain
aligns to a summary entry -- the same masks MFDesign trains on, and available for
every summary complex. Chains not covered by the summary fall back to numbering
with ``abnumber`` (backed by ANARCI/anarcii); abnumber fails to number ~10% of
chains, which is why the curated masks are preferred. Chains that neither align
to the summary nor parse as an antibody variable domain are left untouched.
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

# Per-region integer labels for the model's ``region_type`` feature. 0 is
# reserved for "not an antibody Fv position" (padding_idx of the region
# embedding); antigen tokens are labelled separately in
# :func:`add_chain_and_region_types_to_token_array`.
_REGION_LABELS = {name: i + 1 for i, name in enumerate(_CHOTHIA_RANGES)}
_CDR_LABEL_SET = frozenset(_REGION_LABELS[r] for r in _CDR_RANGES)

# ``chain_type`` / ``region_type`` label values shared with the sequence model
# (protenix/model/modules/diffusion.py:238-239). 0 = padding.
CHAIN_TYPE_HEAVY = 1
CHAIN_TYPE_LIGHT = 2
CHAIN_TYPE_ANTIGEN = 3
REGION_TYPE_ANTIGEN = 8  # antigen, non-epitope
REGION_TYPE_EPITOPE = 9  # antigen residue near the antibody (epitope)

# Antigen residue is an epitope if any of its atoms lies within this distance (A)
# of any antibody (H/L) atom. Matches MFDesign's default epitope cutoff.
EPITOPE_DISTANCE_CUTOFF = 10.0

# anarcii is a torch model; instantiate one CPU/single-thread instance and reuse
# it for every chain to avoid the thread oversubscription that occurs when each
# call spins up ncpu=-1 threads inside a joblib worker.
_ANARCII_ARGS = {"cpu": True, "ncpu": 1}


def _fv_region_labels(chain) -> "tuple[str, list[int]] | tuple[None, None]":
    """Return (Fv sequence, per-residue region labels) for an abnumber Chain.

    Faithful to MFDesign ``mask_cdr``: build the Fv sequence by concatenating the
    Chothia region sequences in order and label each residue with its region
    (``_REGION_LABELS``: fr1..fr4 / cdr1..cdr3, 1-7). Returns ``(None, None)`` if
    any region is empty (incomplete/over-long CDR3), matching MFDesign's behaviour
    of discarding such chains (here: no stripping).
    """
    origin, labels = [], []
    for region in _CHOTHIA_RANGES:
        seq = getattr(chain, region + "_seq")
        if len(seq) == 0:
            return None, None
        origin += list(seq)
        labels += [_REGION_LABELS[region]] * len(seq)
    return "".join(origin), labels


def _chain_residue_region_labels(struct_seq: str, seq_cache: dict) -> "list[int] | None":
    """Map an antibody chain's structure sequence to a per-residue region-label list.

    Each residue gets its Chothia region label (1-7; see ``_REGION_LABELS``), with
    0 for positions outside the numbered Fv (constant domain / unnumbered tails).
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
            fv_seq, region_labels = _fv_region_labels(chain)
            if fv_seq is not None:
                offset = struct_seq.find(fv_seq)
                if offset >= 0:
                    labels = [0] * len(struct_seq)
                    for i, label in enumerate(region_labels):
                        labels[offset + i] = label
                    result = labels
                else:
                    logger.warning(
                        "Antibody Fv sequence not found within chain sequence; "
                        "skipping CDR stripping for this chain."
                    )
    except Exception as e:  # noqa: BLE001 - non-antibody chains raise here
        logger.debug("abnumber could not number chain (treated as non-antibody): %s", e)

    seq_cache[struct_seq] = result
    return result


def _cdr_runs(masked_seq: str) -> "list[tuple[int, int]]":
    """Return the (start, stop) spans of maximal ``X`` runs in a masked sequence."""
    runs = []
    i, n = 0, len(masked_seq)
    while i < n:
        if masked_seq[i] == "X":
            j = i
            while j < n and masked_seq[j] == "X":
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def _ref_region_labels(masked_seq: str) -> "list[int] | None":
    """Per-position Chothia region labels (1-7) for a summary masked Fv sequence.

    MFDesign's ``*_masked_seq`` marks the three CDRs with ``X`` runs. The runs are
    cdr1/cdr2/cdr3 (labels 2/4/6); the framework segments around them are
    fr1..fr4 (labels 1/3/5/7) -- the same ``_REGION_LABELS`` scheme abnumber
    produces. Returns ``None`` unless there are exactly three CDR runs (so it can
    be mapped onto the seven-region Chothia layout).
    """
    runs = _cdr_runs(masked_seq)
    if len(runs) != 3:
        return None
    n = len(masked_seq)
    (c1s, c1e), (c2s, c2e), (c3s, c3e) = runs
    segments = [
        (0, c1s, _REGION_LABELS["fr1"]),
        (c1s, c1e, _REGION_LABELS["cdr1"]),
        (c1e, c2s, _REGION_LABELS["fr2"]),
        (c2s, c2e, _REGION_LABELS["cdr2"]),
        (c2e, c3s, _REGION_LABELS["fr3"]),
        (c3s, c3e, _REGION_LABELS["cdr3"]),
        (c3e, n, _REGION_LABELS["fr4"]),
    ]
    labels = [0] * n
    for start, stop, label in segments:
        for k in range(start, stop):
            labels[k] = label
    return labels


def _region_labels_from_summary(
    struct_seq: str, summary_seqs: "list[tuple[str, str]]"
) -> "list[int] | None":
    """Map a chain's structure sequence to region labels from SAbDab summary seqs.

    ``summary_seqs`` is a list of ``(ref_seq, masked_seq)`` pairs (H and/or L)
    from MFDesign's curated summary, where ``ref_seq`` is the Fv reference
    sequence and ``masked_seq`` is the same sequence with CDRs replaced by ``X``.
    For each pair whose reference aligns to ``struct_seq`` we transfer the
    masked-derived region labels (1-7) onto the matched structure positions,
    returning a per-residue label list (0 outside the Fv). Returns ``None`` if no
    pair aligns, in which case the caller falls back to abnumber numbering.

    This bypasses abnumber/ANARCI, which fails to number ~10% of chains; the
    summary CDR masks are pre-computed once by MFDesign and always available.
    """
    import difflib

    for ref_seq, masked_seq in summary_seqs:
        if not ref_seq or not masked_seq or len(ref_seq) != len(masked_seq):
            continue
        ref_labels = _ref_region_labels(masked_seq)
        if ref_labels is None:
            continue
        labels = [0] * len(struct_seq)
        # Fast path: the Fv reference is an exact substring of the structure seq.
        offset = struct_seq.find(ref_seq)
        if offset >= 0:
            for i, label in enumerate(ref_labels):
                labels[offset + i] = label
            return labels
        # Robust path: unresolved residues / point mutations break an exact find,
        # so align ref->struct and transfer labels only on matched blocks. Require
        # most of the Fv to match to avoid mislabelling an unrelated chain.
        matcher = difflib.SequenceMatcher(None, ref_seq, struct_seq, autojunk=False)
        blocks = matcher.get_matching_blocks()
        matched = sum(size for _, _, size in blocks)
        if matched >= int(0.9 * len(ref_seq)):
            for i, j, size in blocks:
                for k in range(size):
                    labels[j + k] = ref_labels[i + k]
            return labels
    return None


def strip_cdr_side_chains(
    atom_array: AtomArray, summary_seqs: "list[tuple[str, str]] | None" = None
) -> "tuple[AtomArray, int]":
    """Remove side-chain atoms of antibody CDR residues from an AtomArray.

    For every protein chain that parses as an antibody heavy/light variable
    domain, residues in CDR1/2/3 are reduced to backbone + CB atoms. CDR
    boundaries come from MFDesign's curated ``summary_seqs`` when a chain aligns
    to one (``_region_labels_from_summary``), falling back to Chothia numbering
    via abnumber otherwise. All other atoms/chains are left untouched. A boolean
    per-atom ``is_cdr`` annotation marking atoms that belong to a CDR residue is
    added (set before stripping, so the retained backbone/CB atoms of CDR residues
    are flagged True). An integer per-atom ``region_label`` annotation is added
    too (Chothia region 1-7 for Fv positions, 0 elsewhere; see ``_REGION_LABELS``),
    the basis of the per-token ``region_type`` model feature.

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
    region_label_atom = np.zeros(n_atoms, dtype=np.int64)
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

        # Prefer MFDesign's curated CDR masks (always available, no numbering);
        # fall back to abnumber only for chains not covered by the summary.
        region_labels = None
        if summary_seqs:
            region_labels = _region_labels_from_summary(struct_seq, summary_seqs)
        if region_labels is None:
            region_labels = _chain_residue_region_labels(struct_seq, seq_cache)
        if region_labels is None:
            continue

        for (start, stop), region_label in zip(residues, region_labels): #removing side chain atoms of CDR residues
            if not region_label:
                continue
            region_label_atom[start:stop] = region_label
            if region_label not in _CDR_LABEL_SET:
                continue
            is_cdr_atom[start:stop] = True
            for a in range(start, stop):
                if atom_names[a] not in BACKBONE_CB_ATOMS:
                    keep_mask[a] = False

    n_removed = int((~keep_mask).sum())

    atom_array.set_annotation("is_cdr", is_cdr_atom)
    atom_array.set_annotation("region_label", region_label_atom)

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


def _antigen_epitope_token_mask(
    token_array, atom_array, antibody_asym, antigen_asym, cutoff
) -> "tuple[list[bool], int]":
    """Per-token epitope flag for antigen tokens.

    An antigen token is an epitope if any of its (resolved) atoms lies within
    ``cutoff`` A of any (resolved) antibody atom -- the direct analog of MFDesign's
    :func:`get_epitope_token` all-atom distance test.

    Returns ``(epitope_mask, epitope_count)`` where ``epitope_mask`` is one bool
    per token (True only for epitope antigen tokens).
    """
    n_token = len(token_array.tokens)
    epitope = [False] * n_token
    if not antibody_asym or not antigen_asym:
        return epitope, 0

    coords = np.asarray(atom_array.coord)
    asym = np.asarray(atom_array.asym_id_int)
    cats = atom_array.get_annotation_categories()
    resolved = (
        np.asarray(atom_array.is_resolved).astype(bool)
        if "is_resolved" in cats
        else np.ones(len(atom_array), dtype=bool)
    )
    ab_mask = np.isin(asym, list(antibody_asym)) & resolved
    if not ab_mask.any():
        return epitope, 0

    from scipy.spatial import cKDTree

    tree = cKDTree(coords[ab_mask])
    centre_atom_indices = token_array.get_annotation("centre_atom_index")
    count = 0
    for tok_i, token in enumerate(token_array.tokens):
        if int(asym[centre_atom_indices[tok_i]]) not in antigen_asym:
            continue
        atom_idx = np.asarray(token.atom_indices)
        tok_resolved = atom_idx[resolved[atom_idx]]
        if tok_resolved.size == 0:
            continue
        # distance_upper_bound -> finite result only when a neighbour is within cutoff
        d, _ = tree.query(coords[tok_resolved], k=1, distance_upper_bound=cutoff)
        if np.any(np.isfinite(d)):
            epitope[tok_i] = True
            count += 1
    return epitope, count


def add_chain_and_region_types_to_token_array(
    token_array,
    atom_array,
    heavy_ids,
    light_ids,
    antigen_ids,
    epitope_cutoff: float = EPITOPE_DISTANCE_CUTOFF,
) -> "tuple[object, int]":
    """Add per-token ``chain_type`` and ``region_type`` annotations to a TokenArray.

    These feed the sequence model's type/region embeddings
    (protenix/model/modules/diffusion.py:238-239). Labels are assigned per token
    from its centre atom:

    * ``chain_type``: 1=Heavy, 2=Light, 3=Antigen (from the SAbDab chain roles
      matched against each token's ``asym_id_int``), 0 for everything else.
    * ``region_type``: 1-7 for the Chothia Fv regions (from the per-atom
      ``region_label`` set by :func:`strip_cdr_side_chains`), 8 for non-epitope
      antigen tokens, 9 for epitope antigen tokens (within ``epitope_cutoff`` of
      the antibody), 0 otherwise. Antigen wins over any residual ``region_label``.

    The ``region_label`` atom annotation is expected (present when CDR stripping
    ran); if absent, region defaults to 0 for non-antigen tokens.

    Args:
        token_array (TokenArray): must carry ``centre_atom_index``.
        atom_array (AtomArray): must carry ``asym_id_int`` and ``coord`` (and
            ``region_label`` / ``is_resolved`` when available).
        heavy_ids, light_ids, antigen_ids: ``asym_id_int`` chain indices for the
            heavy / light / antigen chains (from :func:`resolve_sabdab_roles`).
        epitope_cutoff (float): antibody-antigen distance cutoff for epitope tokens.

    Returns:
        tuple[TokenArray, int]: the TokenArray (with ``chain_type`` and
        ``region_type``) and the number of epitope antigen tokens found.
    """
    centre_atom_indices = token_array.get_annotation("centre_atom_index")
    asym_id_int = atom_array.asym_id_int
    has_region_label = "region_label" in atom_array.get_annotation_categories()
    region_label_atom = atom_array.region_label if has_region_label else None

    heavy_set, light_set, antigen_set = set(heavy_ids), set(light_ids), set(antigen_ids)

    epitope_mask, epitope_count = _antigen_epitope_token_mask(
        token_array, atom_array, heavy_set | light_set, antigen_set, epitope_cutoff
    ) #return epitope mask, which labels which residues are epitopes

    chain_type, region_type = [], [] #token level lists of chain and region type 
    for tok_i, i in enumerate(centre_atom_indices): #iterate through all residues of token_array
        asym = int(asym_id_int[i]) 
        #determine what chain type the residue is in
        if asym in heavy_set: 
            chain_type.append(CHAIN_TYPE_HEAVY)
        elif asym in light_set:
            chain_type.append(CHAIN_TYPE_LIGHT)
        elif asym in antigen_set:
            chain_type.append(CHAIN_TYPE_ANTIGEN)
        else:
            chain_type.append(0)

        #now populate region_type for each residue
        if asym in antigen_set: 
            region_type.append( #check if antigen is an epitope or not with epitope_mask
                REGION_TYPE_EPITOPE if epitope_mask[tok_i] else REGION_TYPE_ANTIGEN
            )
        elif region_label_atom is not None:
            region_type.append(int(region_label_atom[i]))
        else:
            region_type.append(0)

    token_array.set_annotation("chain_type", chain_type)
    token_array.set_annotation("region_type", region_type)
    return token_array, epitope_count


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
                    # Curated Fv reference + CDR-masked seqs (if present) so CDR
                    # boundaries come from the mask, not abnumber. See
                    # :func:`_region_labels_from_summary`.
                    "H_seq": _clean(row.get("H_chain_seq")),
                    "H_masked": _clean(row.get("H_chain_masked_seq")),
                    "L_seq": _clean(row.get("L_chain_seq")),
                    "L_masked": _clean(row.get("L_chain_masked_seq")),
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
