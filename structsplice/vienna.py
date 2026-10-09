"""Safe subprocess wrappers around the ViennaRNA command-line tools.

Two entry points:

* :func:`run_rnafold`      -> minimum-free-energy dot-bracket structure + MFE
* :func:`run_rnaplfold`    -> unpaired-stretch accessibility matrix

Both take a list of sequences and return results in the same order. Neither
uses a shell, so sequence content can never be interpreted as a command.
RNAplfold writes its output as files named after the FASTA header, so every
call runs inside a fresh temporary directory with synthetic numeric IDs.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .config import (
    DEFAULT_NA_POLICY,
    NA_POLICIES,
    RNAfoldParams,
    RNAplfoldParams,
)

INSTALL_HINT = """\
ViennaRNA command-line tools were not found on PATH.

Install them with one of:

    conda install -c bioconda viennarna      # recommended on HPC
    brew install viennarna                   # macOS
    apt-get install vienna-rna               # Debian/Ubuntu

or build from source: https://github.com/ViennaRNA/ViennaRNA

Then confirm with:

    RNAfold --version
    RNAplfold --help

If the binaries live outside PATH, pass an explicit path with
--rnafold-bin / --rnaplfold-bin.
"""


class ViennaRNANotFound(RuntimeError):
    """Raised when a required ViennaRNA executable is unavailable."""


class ViennaRNAError(RuntimeError):
    """Raised when a ViennaRNA executable runs but produces unusable output."""


# ── Availability ────────────────────────────────────────────────────────────


def find_binary(name: str) -> str | None:
    """Return the resolved path to a ViennaRNA executable, or ``None``.

    Args:
        name: Executable name or an explicit path.

    Returns:
        Absolute path if the executable exists and is runnable, else ``None``.
    """
    resolved = shutil.which(name)
    if resolved:
        return resolved
    candidate = Path(name).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())
    return None


def require_binary(name: str) -> str:
    """Return the resolved path to an executable or raise with install help.

    Args:
        name: Executable name or an explicit path.

    Returns:
        Absolute path to the executable.

    Raises:
        ViennaRNANotFound: If the executable cannot be located.
    """
    resolved = find_binary(name)
    if resolved is None:
        raise ViennaRNANotFound(f"could not find {name!r}.\n\n{INSTALL_HINT}")
    return resolved


def vienna_available(
    rnafold_bin: str = "RNAfold", rnaplfold_bin: str = "RNAplfold"
) -> bool:
    """Return ``True`` when both ViennaRNA tools used here are available."""
    return (
        find_binary(rnafold_bin) is not None
        and find_binary(rnaplfold_bin) is not None
    )


def vienna_version(rnafold_bin: str = "RNAfold") -> str | None:
    """Return the reported ``RNAfold`` version string, or ``None``."""
    resolved = find_binary(rnafold_bin)
    if resolved is None:
        return None
    try:
        out = subprocess.run(
            [resolved, "--version"], capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


# ── Sequence hygiene ────────────────────────────────────────────────────────

_ALLOWED = set("ACGUTN")


def to_rna(seq: str) -> str:
    """Return ``seq`` uppercased with ``T`` rewritten as ``U``.

    The shipped PNAS datasets store DNA strings. ViennaRNA substitutes ``T``
    with ``U`` internally and echoes the substituted sequence, so every
    sequence is normalised to RNA on the way in. That keeps the echo check in
    :func:`_parse_rnafold_stdout` exact and makes cache keys independent of
    whether the caller supplied DNA or RNA.

    Args:
        seq: Nucleotide sequence in either alphabet.

    Returns:
        The uppercased RNA form of the sequence.
    """
    return seq.strip().upper().replace("T", "U")


def validate_sequence(seq: str) -> str:
    """Return the RNA form of ``seq`` after checking it is usable.

    Args:
        seq: Nucleotide sequence.

    Returns:
        The uppercased, ``T``-to-``U`` normalised sequence.

    Raises:
        ValueError: If the sequence is empty or holds unexpected characters.
    """
    upper = seq.strip().upper()
    if not upper:
        raise ValueError("empty sequence")
    bad = sorted(set(upper) - _ALLOWED)
    if bad:
        raise ValueError(
            f"sequence contains unsupported characters {bad}; "
            f"expected only {sorted(_ALLOWED)}"
        )
    return to_rna(upper)


# ── RNAfold ─────────────────────────────────────────────────────────────────


def run_rnafold(
    sequences: Sequence[str],
    params: RNAfoldParams | None = None,
) -> tuple[list[str], np.ndarray]:
    """Fold sequences with ``RNAfold`` and return dot-bracket strings and MFEs.

    This returns the *minimum-free-energy dot-bracket secondary structure*, not
    base-pairing probabilities. No ``-p`` flag is passed, so RNAfold performs no
    partition-function computation and emits exactly two lines per sequence.

    Args:
        sequences: Nucleotide sequences, all folded independently.
        params: RNAfold parameters; defaults to :class:`RNAfoldParams`.

    Returns:
        ``(structures, mfe)`` where ``structures[i]`` is the dot-bracket string
        for ``sequences[i]`` (same length as the sequence) and ``mfe`` is a
        float32 array of minimum free energies in kcal/mol.

    Raises:
        ViennaRNANotFound: If ``RNAfold`` is unavailable.
        ViennaRNAError: If the output cannot be parsed or is misaligned.
    """
    params = params or RNAfoldParams()
    if not sequences:
        return [], np.zeros(0, dtype=np.float32)

    binary = require_binary(params.binary)
    clean = [validate_sequence(s) for s in sequences]

    cmd = [binary, "--noPS", "-T", f"{params.temperature:g}"]
    # RNAfold writes .ps files into the CWD even with --noPS for some
    # sub-outputs; run in a scratch directory so the repo stays clean.
    with tempfile.TemporaryDirectory(prefix="structsplice_rnafold_") as tmp:
        proc = subprocess.run(
            cmd,
            input="\n".join(clean) + "\n",
            capture_output=True,
            text=True,
            cwd=tmp,
            check=False,
        )
    if proc.returncode != 0:
        raise ViennaRNAError(
            f"RNAfold exited with code {proc.returncode}: {proc.stderr.strip()}"
        )

    return _parse_rnafold_stdout(proc.stdout, clean)


def _parse_rnafold_stdout(
    stdout: str, sequences: Sequence[str]
) -> tuple[list[str], np.ndarray]:
    """Parse ``RNAfold`` stdout into dot-bracket strings and MFE values.

    RNAfold emits two lines per input sequence: the echoed sequence, then the
    structure followed by the free energy in parentheses, e.g.::

        GGGAAACCC
        (((...))) ( -1.20)

    Args:
        stdout: Raw ``RNAfold`` standard output.
        sequences: The input sequences, used to validate alignment and lengths.

    Returns:
        ``(structures, mfe)``.

    Raises:
        ViennaRNAError: On any structural mismatch with the inputs.
    """
    lines = [ln for ln in stdout.splitlines() if ln.strip()]
    if len(lines) != 2 * len(sequences):
        raise ViennaRNAError(
            f"RNAfold returned {len(lines)} non-empty lines for "
            f"{len(sequences)} sequences; expected {2 * len(sequences)}"
        )

    structures: list[str] = []
    energies = np.empty(len(sequences), dtype=np.float32)

    for i, seq in enumerate(sequences):
        echoed = lines[2 * i].strip()
        body = lines[2 * i + 1]
        if echoed != seq:
            raise ViennaRNAError(
                f"RNAfold output is misaligned at record {i}: echoed "
                f"{echoed!r} but expected {seq!r}"
            )
        structure = body[: len(seq)]
        remainder = body[len(seq) :].strip()
        bad = set(structure) - set("().")
        if bad:
            raise ViennaRNAError(
                f"RNAfold structure for record {i} holds unexpected "
                f"characters {sorted(bad)}: {structure!r}"
            )
        if structure.count("(") != structure.count(")"):
            raise ViennaRNAError(
                f"unbalanced dot-bracket structure at record {i}: {structure!r}"
            )
        try:
            energies[i] = float(remainder.strip("()").strip())
        except ValueError as exc:
            raise ViennaRNAError(
                f"could not read free energy for record {i} from {body!r}"
            ) from exc
        structures.append(structure)

    return structures, energies


# ── RNAplfold ───────────────────────────────────────────────────────────────


def run_rnaplfold(
    sequences: Sequence[str],
    params: RNAplfoldParams | None = None,
    na_policy: str = DEFAULT_NA_POLICY,
    n_channels: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute RNAplfold unpaired-stretch accessibility for each sequence.

    Each returned channel ``c`` (0-based) holds the probability that an
    unpaired stretch of length ``l = c + 1`` *ends* at that position, read from
    the ``l=1..`` columns of RNAplfold's ``_lunp`` output. This is the full
    multi-length accessibility profile, not the single-nucleotide (``l=1``)
    unpaired probability, which is only channel 0.

    Args:
        sequences: Nucleotide sequences. Lengths may differ.
        params: RNAplfold parameters; defaults to :class:`RNAplfoldParams`
            (``-W 200 -L 150 -u 31 -T 37``).
        na_policy: How to fill boundary ``NA`` entries; see
            :data:`structsplice.config.NA_POLICIES`.
        n_channels: Number of leading length columns to keep. Defaults to
            :data:`structsplice.config.N_ACCESSIBILITY_CHANNELS` (8).

    Returns:
        ``(values, na_mask)``. ``values`` has shape
        ``(n_sequences, n_channels, max_len)`` as float32 and ``na_mask`` the
        same shape as bool, ``True`` exactly where RNAplfold reported ``NA``.
        Positions beyond a sequence's own length are zero in ``values`` and
        ``True`` in ``na_mask``.

    Raises:
        ViennaRNANotFound: If ``RNAplfold`` is unavailable.
        ViennaRNAError: If output files are missing or malformed.
        ValueError: On a bad ``na_policy`` or an unusable sequence.
    """
    from .config import N_ACCESSIBILITY_CHANNELS

    params = params or RNAplfoldParams()
    n_channels = N_ACCESSIBILITY_CHANNELS if n_channels is None else n_channels
    if na_policy not in NA_POLICIES:
        raise ValueError(
            f"unknown na_policy {na_policy!r}; expected one of {list(NA_POLICIES)}"
        )
    if n_channels > params.ulength:
        raise ValueError(
            f"requested {n_channels} accessibility channels but RNAplfold was "
            f"asked for only -u {params.ulength}"
        )
    if not sequences:
        return (
            np.zeros((0, n_channels, 0), dtype=np.float32),
            np.zeros((0, n_channels, 0), dtype=bool),
        )

    binary = require_binary(params.binary)
    clean = [validate_sequence(s) for s in sequences]
    max_len = max(len(s) for s in clean)

    values = np.zeros((len(clean), n_channels, max_len), dtype=np.float32)
    na_mask = np.ones((len(clean), n_channels, max_len), dtype=bool)

    cmd = [
        binary,
        "-W", str(params.winsize),
        "-L", str(params.span),
        "-u", str(params.ulength),
        "-T", f"{params.temperature:g}",
    ]

    with tempfile.TemporaryDirectory(prefix="structsplice_rnaplfold_") as tmp:
        tmpdir = Path(tmp)
        # Synthetic, filesystem-safe IDs: RNAplfold names its output files
        # after the FASTA header, so headers must never come from input data.
        ids = [f"s{i:08d}" for i in range(len(clean))]
        fasta = "".join(f">{i}\n{s}\n" for i, s in zip(ids, clean))

        proc = subprocess.run(
            cmd,
            input=fasta,
            capture_output=True,
            text=True,
            cwd=tmpdir,
            check=False,
        )
        if proc.returncode != 0:
            raise ViennaRNAError(
                f"RNAplfold exited with code {proc.returncode}: "
                f"{proc.stderr.strip()}"
            )

        for i, (seq_id, seq) in enumerate(zip(ids, clean)):
            lunp = tmpdir / f"{seq_id}_lunp"
            if not lunp.is_file():
                raise ViennaRNAError(
                    f"RNAplfold produced no _lunp file for record {i}. "
                    f"stderr: {proc.stderr.strip()}"
                )
            # RNAplfold caps its column count at the sequence length: asking
            # for -u 31 on a 24 nt sequence yields 24 columns, not 31. Expect
            # the capped count, then treat any stretch length longer than the
            # sequence as undefined everywhere rather than as a measured zero.
            n_cols = min(params.ulength, len(seq))
            raw, raw_na = parse_lunp(lunp.read_text(), len(seq), n_cols)
            if n_cols < n_channels:
                pad = n_channels - n_cols
                raw = np.vstack(
                    [raw, np.zeros((pad, len(seq)), dtype=np.float32)]
                )
                raw_na = np.vstack(
                    [raw_na, np.ones((pad, len(seq)), dtype=bool)]
                )
            vals = raw[:n_channels]
            mask = raw_na[:n_channels]
            values[i, :, : len(seq)] = _apply_na_policy(vals, mask, na_policy)
            na_mask[i, :, : len(seq)] = mask

    return values, na_mask


def parse_lunp(
    text: str, seq_length: int, ulength: int
) -> tuple[np.ndarray, np.ndarray]:
    """Parse an RNAplfold ``_lunp`` file into an accessibility matrix.

    The file holds two header lines (``#unpaired probabilities`` and a
    ``#i$  l=1  2 ...`` column legend) followed by one row per sequence
    position. Row *i* (1-based) gives the probability that an unpaired stretch
    of length *l* ends at position *i*, for *l* = 1..``ulength``. Entries with
    *l > i* are ``NA``: such a stretch would extend past the 5' end of the
    molecule, so the event is undefined rather than zero.

    Args:
        text: Full contents of the ``_lunp`` file.
        seq_length: Expected number of position rows.
        ulength: Expected number of length columns.

    Returns:
        ``(values, na_mask)``, both shaped ``(ulength, seq_length)``.
        ``values`` is float32 and holds ``0.0`` wherever ``na_mask`` is
        ``True``; ``na_mask`` is ``True`` exactly at ``NA`` entries.

    Raises:
        ViennaRNAError: If the row or column counts or indices disagree with
            the expected shape, or a non-``NA`` field is not a float.
    """
    rows = [ln for ln in text.splitlines() if ln.strip()]
    body = [ln for ln in rows if not ln.lstrip().startswith("#")]
    if len(body) != seq_length:
        raise ViennaRNAError(
            f"_lunp holds {len(body)} position rows but the sequence has "
            f"{seq_length} nucleotides"
        )

    values = np.zeros((ulength, seq_length), dtype=np.float32)
    na_mask = np.zeros((ulength, seq_length), dtype=bool)

    for row_idx, line in enumerate(body):
        fields = line.split()
        position = fields[0]
        if position != str(row_idx + 1):
            raise ViennaRNAError(
                f"_lunp row {row_idx} reports position {position!r}; expected "
                f"{row_idx + 1!r}. Output may be misaligned."
            )
        entries = fields[1:]
        if len(entries) != ulength:
            raise ViennaRNAError(
                f"_lunp row {row_idx + 1} holds {len(entries)} value columns; "
                f"expected {ulength} (-u {ulength})"
            )
        for col_idx, field in enumerate(entries):
            if field.upper() == "NA":
                na_mask[col_idx, row_idx] = True
                continue
            try:
                values[col_idx, row_idx] = float(field)
            except ValueError as exc:
                raise ViennaRNAError(
                    f"_lunp row {row_idx + 1} column l={col_idx + 1} is not a "
                    f"number or NA: {field!r}"
                ) from exc

    _check_na_pattern(na_mask, seq_length, ulength)
    return values, na_mask


def _check_na_pattern(na_mask: np.ndarray, seq_length: int, ulength: int) -> None:
    """Verify ``NA`` entries appear exactly where RNAplfold must emit them.

    A stretch of length ``l`` ending at 1-based position ``i`` is defined iff
    ``i >= l``. Any other ``NA`` placement means the parse is misaligned.

    Args:
        na_mask: Boolean mask shaped ``(ulength, seq_length)``.
        seq_length: Sequence length.
        ulength: Number of length columns.

    Raises:
        ViennaRNAError: If the observed mask differs from the expected pattern.
    """
    lengths = np.arange(1, ulength + 1)[:, None]
    positions = np.arange(1, seq_length + 1)[None, :]
    expected = positions < lengths
    if not np.array_equal(na_mask, expected):
        n_bad = int((na_mask != expected).sum())
        raise ViennaRNAError(
            f"RNAplfold NA pattern is unexpected in {n_bad} cells. NA must "
            f"occur exactly where stretch length exceeds the 1-based position."
        )


def _apply_na_policy(
    values: np.ndarray, na_mask: np.ndarray, na_policy: str
) -> np.ndarray:
    """Fill ``NA`` cells in an accessibility block according to ``na_policy``.

    Args:
        values: Float array shaped ``(n_channels, seq_length)`` with ``0.0`` at
            ``NA`` cells.
        na_mask: Boolean array of the same shape, ``True`` at ``NA`` cells.
        na_policy: ``"zero"`` or ``"edge"``.

    Returns:
        A new float32 array with ``NA`` cells filled.
    """
    filled = values.astype(np.float32, copy=True)
    if na_policy == "zero":
        filled[na_mask] = 0.0
        return filled

    # "edge": at position i (1-based) every length l > i is undefined, so carry
    # the longest defined length, which is exactly l = i (channel index i - 1).
    n_channels, seq_length = filled.shape
    for pos in range(seq_length):
        undefined = np.nonzero(na_mask[:, pos])[0]
        if undefined.size == 0:
            continue
        deepest = int(undefined[0]) - 1
        filled[undefined, pos] = filled[deepest, pos] if deepest >= 0 else 0.0
    return filled


def iter_batches(items: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    """Yield consecutive slices of ``items`` of at most ``batch_size``."""
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]
