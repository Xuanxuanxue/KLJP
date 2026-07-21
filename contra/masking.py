import torch


def normalize_attention_mask(input_ids, attention_mask=None, name="input"):
    """Return a boolean valid-token mask with the same shape as input_ids."""
    if input_ids.ndim != 2:
        raise ValueError(
            f"{name} input_ids must have shape (batch, sequence), "
            f"got {tuple(input_ids.shape)}"
        )
    if input_ids.shape[1] == 0:
        raise ValueError(f"{name} sequence length must be greater than zero")

    if attention_mask is None:
        valid_mask = input_ids.ne(0)
    else:
        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                f"{name} attention_mask must match input_ids shape "
                f"{tuple(input_ids.shape)}, got {tuple(attention_mask.shape)}"
            )
        valid_mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)

    return valid_mask


def require_nonempty_rows(valid_mask, name="input"):
    """Reject samples for which every token is masked."""
    empty_rows = ~valid_mask.any(dim=1)
    if empty_rows.any():
        indices = empty_rows.nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(f"{name} contains fully masked samples at indices {indices}")


def safe_key_padding_mask(valid_mask):
    """Build a key-padding mask that remains finite for fully empty rows.

    Empty label details are expected in this project. Temporarily exposing one
    zero embedding prevents MultiheadAttention from applying softmax to an
    all-masked row; the original valid_mask is still used during pooling.
    """
    safe_valid_mask = valid_mask.clone()
    empty_rows = ~safe_valid_mask.any(dim=1)
    if empty_rows.any():
        safe_valid_mask[empty_rows, 0] = True
    return ~safe_valid_mask


def masked_mean_pool(sequence, valid_mask):
    """Mean-pool valid positions only; empty rows become finite zero vectors."""
    if sequence.shape[:2] != valid_mask.shape:
        raise ValueError(
            "valid_mask must match the first two sequence dimensions, "
            f"got {tuple(valid_mask.shape)} for {tuple(sequence.shape)}"
        )
    weights = valid_mask.to(dtype=sequence.dtype).unsqueeze(-1)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    return (sequence * weights).sum(dim=1) / denominator
