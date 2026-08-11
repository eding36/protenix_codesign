"""Antibody-codesign sequence-recovery (AAR) metrics.

Port of MFDesign's ``calculate_aar`` (boltz ``data/write/writer.py``): amino-acid
recovery over designed (CDR) residues, reported per-CDR and as total / heavy /
light aggregates. Operates on integer token ids (Protenix ``STD_RESIDUES_WITH_GAP``
space) plus a per-token CDR/design mask.
"""

from typing import Any

import numpy as np


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):  # torch tensor
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _contiguous_segments(mask: np.ndarray) -> "list[tuple[int, int]]":
    """Return inclusive ``(start, end)`` index ranges of contiguous True runs."""
    segments = []
    start = -1
    for i in range(len(mask)):
        if mask[i] and start == -1:
            start = i
        elif not mask[i] and start != -1:
            segments.append((start, i - 1))
            start = -1
    if start != -1:
        segments.append((start, len(mask) - 1))
    return segments


def calculate_aar(pred_ids: Any, gt_ids: Any, cdr_mask: Any) -> "dict[str, Any]":
    """Amino-acid recovery over designed (CDR) residues.

    Faithful to MFDesign: the CDR mask is split into contiguous segments (each is
    one CDR loop, in token order), the first three are treated as the heavy-chain
    CDRs and the last three as the light-chain CDRs. Assumes the standard ordering
    (heavy CDRs before light CDRs, <=3 each); with fewer than six segments the
    heavy/light aggregates overlap, matching MFDesign's positional convention.

    Args:
        pred_ids: predicted per-token amino-acid ids. [N_token]
        gt_ids: ground-truth per-token ids. [N_token]
        cdr_mask: per-token boolean design mask. [N_token]

    Returns:
        dict with ``per_cdr`` (list[float], one per CDR segment), ``total``,
        ``heavy``, ``light`` recovery fractions, and ``n_designed`` token count.
    """
    pred = _to_numpy(pred_ids)
    gt = _to_numpy(gt_ids)
    mask = _to_numpy(cdr_mask).astype(bool)
    assert len(pred) == len(gt) == len(mask), "pred/gt/mask length mismatch"

    segments = _contiguous_segments(mask)

    def _seg_acc(segs: "list[tuple[int, int]]") -> "tuple[int, int]":
        matches = sum(int((pred[s : e + 1] == gt[s : e + 1]).sum()) for s, e in segs)
        length = sum(e - s + 1 for s, e in segs)
        return matches, length

    per_cdr = []
    total_match, total_count = 0, 0
    for s, e in segments:
        m, n = _seg_acc([(s, e)])
        per_cdr.append(m / n if n else 0.0)
        total_match += m
        total_count += n

    total = total_match / total_count if total_count else 0.0
    h_m, h_n = _seg_acc(segments[:3])
    l_m, l_n = _seg_acc(segments[-3:])
    return {
        "per_cdr": per_cdr,
        "total": total,
        "heavy": h_m / h_n if h_n else 0.0,
        "light": l_m / l_n if l_n else 0.0,
        "n_designed": int(total_count),
    }
