"""One-hot and accessibility feature encoders.

All encoders return **position-major** arrays shaped ``(length, channels)``,
which is the layout ``torch.nn.LSTM(batch_first=True)`` consumes. The shipped
PNAS npz files use the channel-first ``(channels, length)`` convention for
their own CNN; :func:`channel_first` converts between the two.
"""

from __future__ import annotations

import numpy as np

from .config import (
    DOT_BRACKET_ORDER,
    N_ACCESSIBILITY_CHANNELS,
    NUCLEOTIDE_ORDER,
)

#: ``nucleotide -> channel index``. ``T`` and ``U`` share a channel; ``N`` is
#: deliberately absent so unknown bases become an all-zero column rather than
#: a silently wrong base.
NUCLEOTIDE_INDEX = {base: i for i, base in enumerate(NUCLEOTIDE_ORDER)}
NUCLEOTIDE_INDEX["T"] = NUCLEOTIDE_INDEX["U"]

#: ``dot-bracket symbol -> channel index``.
DOT_BRACKET_INDEX = {symbol: i for i, symbol in enumerate(DOT_BRACKET_ORDER)}


def one_hot_sequence(sequence: str, allow_unknown: bool = True) -> np.ndarray:
    """One-hot encode a nucleotide sequence.

    Channel order is :data:`structsplice.config.NUCLEOTIDE_ORDER`
    (``A``, ``C``, ``G``, ``U``). ``T`` is treated as ``U``.

    Args:
        sequence: Nucleotide sequence in either the DNA or RNA alphabet.
        allow_unknown: When true, a character outside ``ACGUTN`` (including
            ``N``) yields an all-zero column. When false, such a character
            raises.

    Returns:
        Float32 array shaped ``(len(sequence), 4)``. Every row sums to 1.0
        except columns for unknown bases, which sum to 0.0.

    Raises:
        ValueError: If the sequence is empty, or ``allow_unknown`` is false and
            an unknown base appears.
    """
    seq = sequence.strip().upper()
    if not seq:
        raise ValueError("cannot one-hot encode an empty sequence")
    out = np.zeros((len(seq), len(NUCLEOTIDE_ORDER)), dtype=np.float32)
    for pos, base in enumerate(seq):
        channel = NUCLEOTIDE_INDEX.get(base)
        if channel is None:
            if not allow_unknown:
                raise ValueError(
                    f"unknown nucleotide {base!r} at position {pos} of "
                    f"{sequence!r}"
                )
            continue
        out[pos, channel] = 1.0
    return out


def one_hot_dot_bracket(structure: str) -> np.ndarray:
    """One-hot encode an RNAfold dot-bracket secondary structure.

    Channel order is :data:`structsplice.config.DOT_BRACKET_ORDER`
    (``.``, ``(``, ``)``), matching the ``struct_oh`` convention of the shipped
    PNAS npz files. This encodes the *MFE secondary structure*, not
    base-pairing probabilities: each position is unpaired, paired with a
    downstream partner, or paired with an upstream partner.

    Args:
        structure: Dot-bracket string over ``.``, ``(`` and ``)``.

    Returns:
        Float32 array shaped ``(len(structure), 3)``. Every row sums to 1.0.

    Raises:
        ValueError: If the string holds a symbol outside ``.()`` or its
            brackets are unbalanced.
    """
    text = structure.strip()
    if not text:
        raise ValueError("cannot encode an empty dot-bracket string")
    bad = sorted(set(text) - set(DOT_BRACKET_INDEX))
    if bad:
        raise ValueError(
            f"dot-bracket string holds unsupported symbols {bad}; expected "
            f"only {list(DOT_BRACKET_ORDER)}"
        )
    depth = 0
    for symbol in text:
        depth += 1 if symbol == "(" else -1 if symbol == ")" else 0
        if depth < 0:
            raise ValueError(f"unbalanced dot-bracket string: {structure!r}")
    if depth != 0:
        raise ValueError(f"unbalanced dot-bracket string: {structure!r}")

    out = np.zeros((len(text), len(DOT_BRACKET_ORDER)), dtype=np.float32)
    for pos, symbol in enumerate(text):
        out[pos, DOT_BRACKET_INDEX[symbol]] = 1.0
    return out


def encode_accessibility(block: np.ndarray) -> np.ndarray:
    """Transpose a cached accessibility block to position-major layout.

    Args:
        block: Float array shaped ``(n_channels, length)`` as produced by
            :func:`structsplice.vienna.run_rnaplfold`.

    Returns:
        Float32 array shaped ``(length, n_channels)``.

    Raises:
        ValueError: If ``block`` is not two-dimensional, does not carry exactly
            :data:`structsplice.config.N_ACCESSIBILITY_CHANNELS` channels, or
            holds values outside ``[0, 1]``. The range check is what catches a
            mis-parsed ``_lunp`` file, in which a column index or an
            unconverted ``NA`` would show up as an impossible probability.
    """
    arr = np.asarray(block, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(
            f"accessibility block must be 2-D (channels, length); got shape "
            f"{arr.shape}"
        )
    if arr.shape[0] != N_ACCESSIBILITY_CHANNELS:
        raise ValueError(
            f"accessibility block has {arr.shape[0]} channels; the model "
            f"expects exactly {N_ACCESSIBILITY_CHANNELS} (unpaired stretch "
            f"lengths l = 1..{N_ACCESSIBILITY_CHANNELS})"
        )
    if arr.size:
        if not np.isfinite(arr).all():
            raise ValueError(
                "accessibility block holds non-finite values; NA entries must "
                "be filled by the cache's NA policy before encoding"
            )
        low, high = float(arr.min()), float(arr.max())
        if low < -1e-6 or high > 1.0 + 1e-6:
            raise ValueError(
                f"accessibility values must be probabilities in [0, 1]; got "
                f"range [{low:.6g}, {high:.6g}]"
            )
    return np.ascontiguousarray(arr.T)


def build_features(
    sequence: str,
    structure: str | None = None,
    accessibility: np.ndarray | None = None,
) -> np.ndarray:
    """Concatenate the per-position feature channels for one example.

    Channels are laid out in a fixed order so a checkpoint's input dimension is
    always unambiguous:

    ===========  ==================================================
    channels     content
    ===========  ==================================================
    ``0:4``      one-hot nucleotide (``A``, ``C``, ``G``, ``U``)
    ``4:7``      one-hot dot-bracket (``.``, ``(``, ``)``), if given
    next 8       RNAplfold accessibility for ``l = 1..8``, if given
    ===========  ==================================================

    Args:
        sequence: Nucleotide sequence.
        structure: Optional dot-bracket string of the same length.
        accessibility: Optional array shaped ``(n_channels, len(sequence))``.

    Returns:
        Float32 array shaped ``(len(sequence), n_total_channels)``.

    Raises:
        ValueError: If any feature length disagrees with the sequence length,
            or the accessibility block has an unexpected channel count.
    """
    parts = [one_hot_sequence(sequence)]
    length = len(sequence.strip())

    if structure is not None:
        if len(structure.strip()) != length:
            raise ValueError(
                f"structure length {len(structure.strip())} does not match "
                f"sequence length {length}"
            )
        parts.append(one_hot_dot_bracket(structure))

    if accessibility is not None:
        block = encode_accessibility(accessibility)
        if block.shape[0] != length:
            raise ValueError(
                f"accessibility covers {block.shape[0]} positions but the "
                f"sequence has {length} nucleotides"
            )
        if block.shape[1] != N_ACCESSIBILITY_CHANNELS:
            raise ValueError(
                f"expected {N_ACCESSIBILITY_CHANNELS} accessibility channels, "
                f"got {block.shape[1]}"
            )
        parts.append(block)

    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


def channel_first(features: np.ndarray) -> np.ndarray:
    """Convert a position-major array to the PNAS channel-first convention.

    Args:
        features: Array shaped ``(length, channels)``.

    Returns:
        Contiguous array shaped ``(channels, length)``.
    """
    return np.ascontiguousarray(np.asarray(features).T)
