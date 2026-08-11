"""Decode antibody-codesign sequence predictions.

Turns the sequence model's per-token logits (``pred_dict["sequence"]`` ==
``k_denoised``) into concrete amino-acid token ids and 1-letter strings. Mirrors
MFDesign's final decode: sample (or argmax) the designed residues, keep the
ground-truth residue everywhere outside the CDR/design mask. Protenix amino acids
already occupy ids 0-19, so -- unlike MFDesign's boltz vocab -- no ``+2`` shift.
"""

from typing import Any

import numpy as np
import torch
from torch.distributions.categorical import Categorical

from protenix.data.constants import (
    mmcif_restype_3to1,
    STD_RESIDUES_WITH_GAP_ID_TO_NAME,
)


def decode_sequence(
    seq_logits: torch.Tensor,
    cdr_mask: torch.Tensor,
    gt_ids: torch.Tensor,
    temperature: float = 1.0,
    sample: bool = False,
) -> torch.Tensor:
    """Decode per-token amino-acid ids from the sequence logits.

    Args:
        seq_logits (torch.Tensor): per-token logits over the 20 amino acids.
            [..., N_token, vocab_size]
        cdr_mask (torch.Tensor): per-token design mask; only these positions are
            taken from the prediction. [N_token]
        gt_ids (torch.Tensor): ground-truth token ids; used verbatim outside the
            design mask (framework / antigen stay fixed). [N_token]
        temperature (float): sharpness multiplier on the logits when ``sample`` is
            True (MFDesign multiplies logits by temperature). Defaults to 1.0.
        sample (bool): sample from the categorical (True) vs argmax (False).

    Returns:
        torch.Tensor: decoded token ids, prediction at CDR tokens and ground truth
        elsewhere. [..., N_token]
    """
    if sample:
        decoded = Categorical(logits=seq_logits * temperature).sample()
    else:
        decoded = seq_logits.argmax(dim=-1)
    cdr = cdr_mask.to(torch.bool)
    return torch.where(cdr, decoded.to(gt_ids.dtype), gt_ids)


def token_ids_to_letters(ids: Any) -> str:
    """Map ``STD_RESIDUES_WITH_GAP`` token ids to a 1-letter amino-acid string.

    Non-standard / unknown ids map to ``X``.
    """
    if hasattr(ids, "detach"):
        ids = ids.detach().cpu().numpy()
    ids = np.asarray(ids).astype(int).reshape(-1)
    letters = []
    for i in ids:
        resname = STD_RESIDUES_WITH_GAP_ID_TO_NAME.get(int(i), "UNK")
        letters.append(mmcif_restype_3to1.get(resname, "X"))
    return "".join(letters)
