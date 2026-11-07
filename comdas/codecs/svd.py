"""
Truncated Singular Value Decomposition (SVD) codec for 2D arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from warnings import warn

import numpy as np

from comdas.codecs.base import Codec, CompressedPayload


@dataclass
class SVDPayload(CompressedPayload):
    """
    Truncated-SVD representation of a 2D array:
    ``A ~= (U * s) @ Vt``.

    :ivar U: Left singular vectors, shape ``(m, k)``.
    :vartype U: numpy.ndarray
    :ivar s: Singular values, shape ``(k,)``.
    :vartype s: numpy.ndarray
    :ivar Vt: Right singular vectors (transposed), shape ``(k, n)``.
    :vartype Vt: numpy.ndarray
    """

    U: np.ndarray
    s: np.ndarray
    Vt: np.ndarray

    @property
    def rank(self) -> int:
        """
        The truncation rank ``k`` used on this payload.

        :rtype: int
        """
        return self.s.shape[0]

    @property
    def compression_params(self) -> dict:
        """
        The compression parameters used for this payload.

        :rtype: dict
        """
        return {
            "rank": self.rank,
        }


class SVDCodec(Codec):
    """
    Truncated-Singular Value Decomposition (SVD) codec for 2D arrays.

    This codec compresses a 2D array by computing its truncated singular value
    decomposition (SVD) and storing the left singular vectors, singular
    values, and right singular vectors (transposed) as a :class:`SVDPayload`.

    Truncation rank can be chosen either as a fixed integer ``rank`` or
    as a retained energy fraction ``energy`` in ``(0,1]``. Exactly one option
    must be specified, either as a codec-level default or per-call parameter.

    SVD does not implement partial writes efficiently, so the full array is
    re-encoded every 10,000 pending writes (or on demand).

    :param rank: Default truncation rank ``k`` to use if not specified per-call.
    :type rank: int | None
    :param energy: Default retained energy fraction to use if not specified per-call.
    :type energy: float | None
    :raises ValueError: If both of ``rank`` and ``energy`` are specified.
    """

    name = "SVD"
    version = "1"

    # Writes are expensive with SVD
    max_pending_writes = 10_000
    exclusive_params = ("rank", "energy")

    def __init__(self, rank: int | None = None, energy: float | None = None):
        if rank is not None and energy is not None:
            raise ValueError(
                "Provide at most one of rank / energy to act as codec default."
            )
        self.default_rank = rank
        self.default_energy = energy

    def encode(
        self,
        array: np.ndarray,
        *,
        rank: int | None = None,
        energy: float | None = None,
        **kwargs,
    ) -> SVDPayload:
        """
        Compress a 2D array via truncated SVD.

        :param array: The dense 2D array to compress.
        :type array: numpy.ndarray
        :param rank: Fixed truncation rank. Overrides
            ``self.default_rank`` for this call.
        :type rank: int or None
        :param energy: Retained energy fraction in ``(0, 1]``.
            Overrides ``self.default_energy`` for this call.
        :type energy: float or None
        :param kwargs: Accepted and ignored for compatibility.
        :returns: The truncated-SVD payload.
        :rtype: SVDPayload
        :raises ValueError: If ``array`` is not 2D, or if both
            ``rank`` and ``energy`` are specified.
        """
        array = np.asarray(array)
        if array.ndim != 2:
            raise ValueError(
                f"SVDCodec only supports 2D arrays, got shape {array.shape}"
            )

        if rank is not None and energy is not None:
            raise ValueError(
                "Specify only one of `rank` or `energy` per call or use the codec default."
            )
        if rank is None and energy is None:
            rank = self.default_rank
            energy = self.default_energy
        if (rank is None) == (energy is None):
            raise ValueError(
                "Exactly one of `rank` or `energy` must be specified, "
                "either here or as a codec default."
            )

        U, s, Vt = np.linalg.svd(array, full_matrices=False)

        if rank is None:
            cumulative = np.cumsum(s**2)
            cumulative /= cumulative[-1]
            rank = int(np.searchsorted(cumulative, energy) + 1)
            rank = min(rank, s.shape[0])
        rank = max(1, min(rank, s.shape[0]))

        return SVDPayload(
            original_shape=array.shape,
            dtype=array.dtype,
            U=np.ascontiguousarray(U[:, :rank]),
            s=np.ascontiguousarray(s[:rank]),
            Vt=np.ascontiguousarray(Vt[:rank, :]),
        )

    def decode(self, payload: SVDPayload) -> np.ndarray:
        """
        Fully reconstruct the dense array as ``(U * s) @ Vt``.

        :param payload: A payload previously produced by :meth:`encode`.
        :type payload: SVDPayload
        :returns: The reconstructed dense array.
        :rtype: numpy.ndarray
        """
        out = (payload.U * payload.s) @ payload.Vt
        return out.astype(payload.dtype, copy=False)

    def decode_partial(self, payload: SVDPayload, key) -> np.ndarray:
        """
        Reconstruct only the requested rows/cols of ``payload``.

        :param payload: A payload previously produced by :meth:`encode`.
        :type payload: SVDPayload
        :param key: A NumPy-style index. Up to two ints/slices take the
            fast path; anything else falls back to a full decode.
        :returns: The requested subset of the reconstructed array.
        :rtype: numpy.ndarray
        :raises IndexError: If ``key`` is invalid for a 2D array (e.g. too
            many indices or out of bounds), as with NumPy.
        """
        if not isinstance(key, tuple):
            key = (key,)

        # Validate the key against the array shape using NumPy's rules.
        np.broadcast_to(np.empty((), dtype=bool), payload.original_shape)[key]

        def _is_simple(k) -> bool:
            return isinstance(k, (int, np.integer, slice)) and not isinstance(
                k, (bool, np.bool_)
            )

        # If using complex indexing (anything more than up to two simple
        # ints/slices, e.g. arrays, ellipsis, newaxis) fall back to full decode
        if len(key) > 2 or not all(_is_simple(k) for k in key):
            warn(
                "Only simple int/slice indexing is supported for partial decoding. Fully decoding entire array and then applying the index."
            )
            return self.decode(payload)[key]

        key = key + (slice(None),) * (2 - len(key))
        row_key, col_key = key
        row_is_int = isinstance(row_key, (int, np.integer))
        col_is_int = isinstance(col_key, (int, np.integer))

        U_sub = payload.U[row_key, :]
        Vt_sub = payload.Vt[:, col_key]
        if row_is_int:
            U_sub = U_sub[np.newaxis, :]
        if col_is_int:
            Vt_sub = Vt_sub[:, np.newaxis]

        out = (U_sub * payload.s) @ Vt_sub
        out = out.astype(payload.dtype, copy=False)

        if row_is_int:
            out = out[0]
        if col_is_int:
            out = out[..., 0]
        return out

    def compressed_size_bytes(self, payload: SVDPayload) -> int:
        """
        Get the total number of bytes used by ``U``, ``s``, and ``Vt``.

        :param payload: A payload previously produced by :meth:`encode`.
        :type payload: SVDPayload
        :returns: Combined byte size of the three stored arrays.
        :rtype: int
        """
        return payload.U.nbytes + payload.s.nbytes + payload.Vt.nbytes

    def payload_to_group(self, payload: SVDPayload, group) -> None:
        """
        Write ``U``, ``s``, ``Vt``, and ``rank`` into an open ``h5py.Group``.

        :param payload: The payload to serialize.
        :type payload: SVDPayload
        :param group: An open, writable ``h5py.Group`` dedicated to
            this one patch.
        :type group: h5py.Group
        """
        group.create_dataset("U", data=payload.U)
        group.create_dataset("s", data=payload.s)
        group.create_dataset("Vt", data=payload.Vt)
        group.attrs["rank"] = payload.rank

    def payload_from_group(
        self, group, *, original_shape: tuple[int, ...], dtype: np.dtype
    ) -> SVDPayload:
        """
        Construct an :class:`SVDPayload` from an HDF5 group.

        :param group: An open, readable ``h5py.Group`` previously
            written by :meth:`payload_to_group`.
        :type group: h5py.Group
        :param original_shape: The array shape, as read from the
            container's own attributes.
        :type original_shape: tuple[int, ...]
        :param dtype: The array dtype, as read from the container's
            own attributes.
        :type dtype: numpy.dtype
        :returns: The reconstructed payload.
        :rtype: SVDPayload
        """
        return SVDPayload(
            original_shape=original_shape,
            dtype=dtype,
            U=group["U"][:],
            s=group["s"][:],
            Vt=group["Vt"][:],
        )
