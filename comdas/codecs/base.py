"""
Base classes for compression codecs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class CompressedPayload:
    """
    Base container for a codec's compressed payload (representation of an array).

    Codecs should subclass this to hold the arrays/parameters the method requires.

    :ivar original_shape: The shape of the array before compression.
    :vartype original_shape: tuple[int, ...]
    :ivar dtype: The data type of the array before compression.
    :vartype dtype: numpy.dtype
    """

    original_shape: tuple[int, ...]
    dtype: np.dtype

    @property
    def compression_params(self) -> dict:
        """
        The parameters used for compression.

        The default implementation returns an empty dict. Codecs with
        parameters relevant to re-encoding should override this.

        :rtype: dict
        """
        return {}


class Codec(ABC):
    """
    Base class for all codecs.

    A codec is responsible for compressing and decompressing arrays. It knows
    how to turn a dense NumPy array into a :class:`CompressedPayload` and back,
    and how to (de)serialize that payload to/from an HDF5 file.

    Subclasses are auto-registered by name on class creation. This allows
    the reader to automatically recognize and use the appropriate codec by name,
    without having to import every codec.

    Most codecs do not support random access writes and algorithms can be
    inefficient at performing many small writes, so writes to the payload are
    buffered until max_pending_writes is reached. Once this value is exceeded,
    the entire array will be decompressed, the writes applied, and the
    array recompressed. By default, this value is set to 10,000.

    If a codec supports random access writes efficiently, it should
    implement the :meth:`partial_write` method. If this method is
    implemented, it will always be immediately used for applying writes,
    bypassing max_pending_writes entirely.

    :cvar name: Unique, human-readable name for the codec.
    :vartype name: str
    :cvar version: Codec format version, bumped if the codec's payload
        structure changes in a way that is not backward-compatible.
    :vartype version: str
    :cvar max_pending_writes: Maximum number of pending writes before automatic consolidation.
        This is only used by codecs that do not implement :meth:`partial_write`.
    :vartype max_pending_writes: int
    :cvar exclusive_params: Names of mutually exclusive ``encode`` parameters
        that select the compression mode (e.g. ``("rank", "energy")``). When
        a caller overrides one of them, the others are dropped from the
        payload's stored parameters. See :meth:`merge_compression_params`.
    :vartype exclusive_params: tuple[str, ...]
    """

    name: str = ""
    version: str = "1"
    max_pending_writes: int = 10_000
    exclusive_params: tuple[str, ...] = ()

    _registry: dict[str, type[Codec]] = {}

    def __init_subclass__(cls, **kwargs) -> None:
        """
        Register ``cls`` in the registry under ``cls.name``.

        :param kwargs: Forwarded to :meth:`object.__init_subclass__`.
        :raises ValueError: If a concrete subclass doesn't set a
            non-empty ``name``.
        """
        super().__init_subclass__(**kwargs)
        if getattr(cls, "__abstractmethods__", None):
            return
        if not cls.name:
            raise ValueError(
                f"{cls.__name__} must define a `name` for codec registration."
            )
        Codec._registry[cls.name] = cls

    @classmethod
    def get_registered(cls, name: str) -> type[Codec]:
        """
        Look up a codec class by name.

        :param name: The codec's ``name`` attribute.
        :type name: str
        :returns: The codec class registered under ``name``.
        :rtype: type[Codec]
        :raises KeyError: If no codec is registered under ``name``.
        """
        try:
            return cls._registry[name]
        except KeyError as err:
            raise KeyError(
                f"No codec registered under name {name!r}. Known codecs: {sorted(cls._registry)}"
            ) from err

    @abstractmethod
    def encode(self, array: np.ndarray, **kwargs) -> CompressedPayload:
        """
        Compress ``array`` into a :class:`CompressedPayload`.

        :param array: The dense array to compress.
        :type array: numpy.ndarray
        :param kwargs: Codec-specific compression parameters.
        :returns: The compressed payload.
        :rtype: CompressedPayload
        """

    @abstractmethod
    def decode(self, payload: CompressedPayload) -> np.ndarray:
        """
        Fully decompress the payload.

        :param payload: A payload previously produced by :meth:`encode`.
        :type payload: CompressedPayload
        :returns: The decompressed array.
        :rtype: numpy.ndarray
        """

    def decode_partial(self, payload: CompressedPayload, key) -> np.ndarray:
        """
        Partially decompress the payload, reconstructing only the elements
        addressed by ``key``.

        The default implementation uses a full decode. Codecs that support
        partial decompression or random access read/write should overwrite
        this method.

        :param payload: A payload previously produced by :meth:`encode`.
        :type payload: CompressedPayload
        :param key: A NumPy-style index/slice key.
        :returns: The requested subset of the reconstructed array.
        :rtype: numpy.ndarray
        """
        return self.decode(payload)[key]

    def partial_write(
        self, payload: CompressedPayload, key, value
    ) -> CompressedPayload:
        """
        Write ``value`` into the compressed representation at ``key``,
        without a full decode -> update -> re-encode round trip.

        This is an optional method, most codecs do not support random access
        writes. The default implementation always raises a
        :class:`NotImplementedError`, which
        :class:`~comdas.arrays.duck_array.DuckArray`
        interprets as "this codec or write attempt doesn't
        support true partial writes" and falls back to its overlay +
        :attr:`default_overlay_flush_threshold` strategy instead.

        A codec should implement this for the cases it can handle
        cheaply and raise :class:`NotImplementedError` for cases it can't.
        The fallback applies per call, not just
        per codec. See :meth:`supports_partial_write`.

        Implementations must apply exactly the same assignment
        semantics as ``dense_array[key] = value`` would on the fully
        decompressed array, including NumPy's broadcasting rules.

        :param payload: The payload before the write.
        :type payload: CompressedPayload
        :param key: A NumPy-style index/slice key.
        :param value: The value(s) to write, broadcast against the
            shape implied by ``key`` exactly as plain NumPy assignment
            would.
        :returns: The payload reflecting the write.
        :rtype: CompressedPayload
        :raises NotImplementedError: Always, unless overridden.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support partial_write()."
        )

    def supports_partial_write(self) -> bool:
        """
        Whether this codec has overridden :meth:`partial_write`.

        :returns: True if :meth:`partial_write` is overridden.
        :rtype: bool
        """
        return type(self).partial_write is not Codec.partial_write

    @abstractmethod
    def compressed_size_bytes(self, payload: CompressedPayload) -> int:
        """
        Get the total size, in bytes, of everything stored in ``payload``.

        :param payload: A payload previously produced by :meth:`encode`.
        :type payload: CompressedPayload
        :returns: Total size in bytes of the payload's stored arrays.
        :rtype: int
        """

    def merge_compression_params(self, stored: dict, overrides: dict) -> dict:
        """
        Combine a payload's stored compression parameters with explicit
        overrides, for re-encoding (e.g. in
        :meth:`~comdas.arrays.duck_array.DuckArray.consolidate`).

        Overrides win. If an override selects one of the mutually exclusive
        :attr:`exclusive_params`, the other exclusive params are dropped
        from ``stored`` so the old mode doesn't conflict with the new one.
        Conflicts *within* ``overrides`` are left for :meth:`encode` to
        reject.

        :param stored: The payload's ``compression_params``.
        :type stored: dict
        :param overrides: Explicit keyword arguments from the caller.
        :type overrides: dict
        :returns: The parameters to pass to :meth:`encode`.
        :rtype: dict
        """
        merged = dict(stored)
        if any(k in self.exclusive_params for k in overrides):
            for k in self.exclusive_params:
                merged.pop(k, None)
        merged.update(overrides)
        return merged

    def compression_ratio(self, payload: CompressedPayload) -> float:
        """
        Helper to determine the compression ratio of the payload.

        :param payload: A payload previously produced by :meth:`encode`.
        :type payload: CompressedPayload
        :returns: Original bytes divided by compressed bytes.
        :rtype: float
        """
        original_bytes = (
            int(np.prod(payload.original_shape))
            * np.dtype(payload.dtype).itemsize
        )
        return original_bytes / self.compressed_size_bytes(payload)

    @abstractmethod
    def payload_to_group(self, payload: CompressedPayload, group) -> None:
        """
        Write this codec's own arrays/params into an open HDF5 group.

        Only codec-specific data belongs here. The general ``COMDAS``
        container already writes ``original_shape``, ``dtype``,
        ``codec_name``, and ``codec_version`` at the container level
        before calling this. Don't duplicate those fields.

        :param payload: The payload to serialize.
        :type payload: CompressedPayload
        :param group: An open, writable ``h5py.Group`` dedicated to
            this one patch.
        :type group: h5py.Group
        """

    @abstractmethod
    def payload_from_group(
        self, group, *, original_shape: tuple[int, ...], dtype: np.dtype
    ) -> CompressedPayload:
        """
        Reconstruct this codec's payload from an HDF5 group.

        :param group: An open, readable ``h5py.Group`` previously
            written by :meth:`payload_to_group`.
        :type group: h5py.Group
        :param original_shape: The array shape, as already read from
            the container's own (codec-independent) attributes.
        :type original_shape: tuple[int, ...]
        :param dtype: The array dtype, as already read from the
            container's own (codec-independent) attributes.
        :type dtype: numpy.dtype
        :returns: The reconstructed payload.
        :rtype: CompressedPayload
        """
