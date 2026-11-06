"""
Implements a NumPy-like `duck array
<https://docs.xarray.dev/en/stable/user-guide/duckarrays.html>`_ for compressed
Patch data.
"""

from __future__ import annotations

import numpy as np

from comdas.codecs.base import Codec, CompressedPayload


def _materialize(obj):
    """
    Replace every :class:`DuckArray` in ``obj`` with a dense NumPy array,
    recursing into lists, tuples, and dicts (e.g. the list passed to
    ``np.concatenate``).

    :param obj: An argument (or container of arguments) to a NumPy call.
    :returns: ``obj`` with DuckArrays materialized.
    """
    if isinstance(obj, DuckArray):
        return np.asarray(obj)
    if type(obj) is list:
        return [_materialize(o) for o in obj]
    if type(obj) is tuple:
        return tuple(_materialize(o) for o in obj)
    if type(obj) is dict:
        return {k: _materialize(v) for k, v in obj.items()}
    return obj


def _densify_out(kwargs: dict) -> tuple[dict, list]:
    """
    Swap any DuckArray in an ``out=`` kwarg for a dense temporary buffer.

    The buffer starts as a copy of the DuckArray's current values, so
    elements a ``where=`` mask leaves untouched are preserved.

    :param kwargs: Keyword arguments to a NumPy function/ufunc.
    :returns: ``(new_kwargs, pairs)`` where ``pairs`` is a list of
        ``(duck_array, dense_buffer)`` to pass to :func:`_write_back_out`.
    """
    out = kwargs.get("out")
    if out is None:
        return kwargs, []
    is_tuple = isinstance(out, tuple)
    pairs = []
    new_out = []
    for o in out if is_tuple else (out,):
        if isinstance(o, DuckArray):
            dense = np.asarray(o)
            pairs.append((o, dense))
            new_out.append(dense)
        else:
            new_out.append(o)
    new_kwargs = dict(kwargs)
    new_kwargs["out"] = tuple(new_out) if is_tuple else new_out[0]
    return new_kwargs, pairs


def _write_back_out(result, pairs: list):
    """
    Fold dense ``out`` buffers back into their DuckArrays (re-encoding
    them) and return those DuckArrays in place of the buffers.

    :param result: The value returned by the NumPy call.
    :param pairs: As returned by :func:`_densify_out`.
    :returns: ``result``, with dense buffers replaced by their DuckArrays.
    """
    if not pairs:
        return result
    swap = {}
    for duck, dense in pairs:
        duck.consolidate(dense=dense)
        swap[id(dense)] = duck
    if isinstance(result, tuple):
        return tuple(swap.get(id(r), r) for r in result)
    return swap.get(id(result), result)


class DuckArray(np.lib.mixins.NDArrayOperatorsMixin):
    """
    A NumPy-like duck array backed by a compressed payload and codec defining
    how the data is compressed and decompressed.

    Reads are performed via ``codec.decode_partial``. Writes go into a
    sparse overlay dict rather than immediately forcing a complete recompression
    of the underlying compressed payload. :meth:`consolidate` can be called to
    merge the overlay into the compressed payload. :meth:`consolidate` will be
    called automatically when the number of pending writes exceeds the codec's
    specified maximum number of pending writes.

    If the specified codec implements partial writes efficiently, the codec's
    partial writes function will always be used to update the compressed
    payload without accumulation.

    :param payload: The compressed payload backing this duck array.
    :type payload: CompressedPayload
    :param codec: The codec defining how the data is compressed and decompressed.
    :type codec: Codec
    :param max_pending_writes: Maximum number of pending writes before automatic
        consolidation.
    :type max_pending_writes: int or None
    """

    def __init__(
        self,
        payload: CompressedPayload,
        codec: Codec,
        *,
        max_pending_writes: int | None = None,
    ):
        self._payload = payload
        self._codec = codec
        self._overlay: dict[tuple[int, ...], object] = {}
        self._max_pending_writes = (
            max_pending_writes
            if max_pending_writes is not None
            else codec.max_pending_writes
        )

    # Array basics
    @property
    def shape(self) -> tuple[int, ...]:
        """
        Get the shape of the decompressed/original array.

        :rtype: tuple[int, ...]
        """
        return self._payload.original_shape

    @property
    def dtype(self) -> np.dtype:
        """
        Get the data type of the decompressed/original array.

        :rtype: numpy.dtype
        """
        return self._payload.dtype

    @property
    def ndim(self) -> int:
        """
        Get the number of dimensions of the decompressed/original array.

        :rtype: int
        """
        return len(self.shape)

    def __len__(self) -> int:
        return self.shape[0]

    @property
    def payload(self) -> CompressedPayload:
        """
        Get the current compressed payload backing this array.

        :rtype: CompressedPayload
        """
        return self._payload

    @property
    def codec(self) -> Codec:
        """
        Get the codec used to encode/decode :attr:`payload`.

        :rtype: Codec
        """
        return self._codec

    @property
    def overlay_size(self) -> int:
        """
        Get the number of elements currently held in the write overlay.

        :rtype: int
        """
        return len(self._overlay)

    def __array__(self, dtype=None) -> np.ndarray:
        """
        Fully materialize this array as a dense NumPy array.

        :param dtype: Optional dtype to cast the result to.
        :type dtype: numpy.dtype or None
        :returns: The decompressed array, with any pending overlay
            writes applied.
        :rtype: numpy.ndarray
        """
        out = self._codec.decode(self._payload).copy()
        for idx, val in self._overlay.items():
            out[idx] = val
        if dtype is not None:
            out = out.astype(dtype, copy=False)
        return out

    def __array_function__(self, func, types, args, kwargs):
        """
        Apply a numpy function to this array. Decompresses any DuckArrays
        among the arguments (including inside lists/tuples/dicts, e.g. for
        ``np.concatenate``) before applying the NumPy function.

        If a DuckArray is given as ``out=``, the result is computed into a
        dense buffer and then re-encoded into that DuckArray (see
        :meth:`consolidate`), which is returned in place of the buffer.

        NOTE: Some codecs support efficient operations directly
        on the compressed representation. We should provide a way
        to implement these in the future.

        NOTE: Applying a chain of functions will decompress the array
        at each step. We should optimize chained operations to avoid
        this.

        :param func: The NumPy function being dispatched.
        :param types: Types of the arguments involved.
        :param args: Positional arguments to ``func``.
        :param kwargs: Keyword arguments to ``func``.
        :returns: The result of calling ``func`` on materialized arrays.
        """
        kwargs, out_pairs = _densify_out(kwargs)
        result = func(*_materialize(args), **_materialize(kwargs))
        return _write_back_out(result, out_pairs)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Materializes the array then applies the ufunc.

        DuckArrays given as ``out=`` (including via in-place operators such
        as ``+=``) or as the target of ``ufunc.at`` are updated by computing
        into a dense buffer and re-encoding it (see :meth:`consolidate`).
        Re-encoding with a lossy codec can introduce reconstruction error.

        NOTE: We should explore per-codec optimizations.

        :param ufunc: The NumPy ufunc being dispatched.
        :param method: The ufunc method being called (e.g. ``"__call__"``).
        :param inputs: Positional inputs to the ufunc.
        :param kwargs: Keyword arguments to the ufunc.
        :returns: The result of calling the ufunc on materialized arrays.
        """
        kwargs, out_pairs = _densify_out(kwargs)

        # ufunc.at modifies its first input in place.
        if method == "at" and isinstance(inputs[0], DuckArray):
            target = inputs[0]
            dense = np.asarray(target)
            ufunc.at(dense, *_materialize(inputs[1:]), **_materialize(kwargs))
            target.consolidate(dense=dense)
            return None

        result = getattr(ufunc, method)(
            *_materialize(inputs), **_materialize(kwargs)
        )
        return _write_back_out(result, out_pairs)

    def __repr__(self) -> str:
        ratio = self._codec.compression_ratio(self._payload)
        return (
            f"DuckArray(shape={self.shape}, dtype={self.dtype}, "
            f"codec={type(self._codec).__name__}, ratio={ratio:.1f}x, "
            f"overlay={self.overlay_size})"
        )

    def _selected_positions(self, key) -> tuple[np.ndarray, ...]:
        """
        Find the physical position of every element selected by ``key``.

        NumPy itself applies ``key`` (to zero-copy broadcast views of
        per-axis index grids), so the result follows NumPy's indexing
        semantics: negative indices, bounds checks, slices,
        broadcast/paired advanced indices, ellipsis, newaxis, and boolean
        masks.

        :param key: A NumPy-style index key.
        :returns: One integer array per axis, each with the shape of
            ``dense[key]``, giving the coordinates of each
            selected element.
        :rtype: tuple[numpy.ndarray, ...]
        :raises IndexError: If ``key`` is invalid for this array's shape.
        """
        grids = np.indices(self.shape, sparse=True)
        return tuple(np.broadcast_to(g, self.shape)[key] for g in grids)

    def __getitem__(self, key):
        """
        Read values from the array.

        :param key: A NumPy-style index/slice key.
        :returns: The requested subset of the array.
        :rtype: numpy.ndarray or scalar
        """
        if not self._overlay:
            return self._codec.decode_partial(self._payload, key)

        positions = self._selected_positions(key)
        base = np.array(
            self._codec.decode_partial(self._payload, key), copy=True
        )

        # Match pending overlay writes to the selected elements by flat
        # physical index.
        selected = np.ravel_multi_index(positions, self.shape)
        overlay_idx = np.ravel_multi_index(
            tuple(np.array(list(self._overlay)).T), self.shape
        )
        overlay_vals = np.array(list(self._overlay.values()), dtype=self.dtype)
        hit = np.isin(selected, overlay_idx)
        if hit.any():
            order = np.argsort(overlay_idx)
            loc = np.searchsorted(overlay_idx, selected[hit], sorter=order)
            base[hit] = overlay_vals[order[loc]]
        return base[()] if base.ndim == 0 else base

    def __setitem__(self, key, value) -> None:
        """
        Write a value. Attempts to not decompress the entire array if possible.

        Three strategies are tried, in order:

        1. If the codec implements :meth:`~comdas.codecs.base.Codec.partial_write`,
           it's used directly. In this case, the payload is patched in place.
           A codec may raise :class:`NotImplementedError` for a particular ``key``
           it can't handle (e.g. it only supports single elements), in
           this case it goes to strategy 2 for that call.
        2. Otherwise, small writes land in a sparse overlay dict,
           applied on read and folded in by :meth:`consolidate`.
        3. A write that touches more than
           :attr:`~comdas.codecs.base.Codec.max_pending_writes`
           elements or would push the overlay size past it auto-consolidates
           instead.

        :param key: A NumPy-style index/slice key.
        :param value: The value(s) to write; broadcast against the
            shape implied by ``key``, matching normal NumPy assignment.
        """
        if self._codec.supports_partial_write():
            try:
                self._payload = self._codec.partial_write(
                    self._payload, key, value
                )
                return
            except NotImplementedError:
                pass

        positions = self._selected_positions(key)
        target_shape = positions[0].shape
        size = positions[0].size

        if (
            size > self._max_pending_writes
            or (self.overlay_size + size) > self._max_pending_writes
        ):
            dense = self.__array__()
            dense[key] = value
            self.consolidate(dense=dense)
            return

        values = np.asarray(value).astype(self.dtype, copy=False)
        # Like NumPy assignment, allow extra leading length-1 dims on value.
        while values.ndim > len(target_shape) and values.shape[0] == 1:
            values = values[0]
        try:
            values = np.broadcast_to(values, target_shape)
        except ValueError:
            raise ValueError(
                f"could not broadcast input array from shape "
                f"{np.shape(value)} into shape {target_shape}"
            ) from None

        flat_positions = [p.ravel() for p in positions]
        for idx, val in zip(zip(*flat_positions), values.ravel()):
            self._overlay[tuple(int(i) for i in idx)] = val

    def consolidate(
        self,
        *,
        dense: np.ndarray | None = None,
        **encode_kwargs,
    ) -> None:
        """
        Fold the write overlay into a freshly-compressed payload.

        By default, re-encodes using whatever compression parameters the
        current payload already used. Any ``encode_kwargs`` given override
        those (see :meth:`~comdas.codecs.base.Codec.merge_compression_params`).

        :param dense: Precomputed dense array to encode, if the
            caller already has one.
        :type dense: numpy.ndarray or None
        :param encode_kwargs: Codec-specific kwargs forwarded to
            ``codec.encode``, overriding the current payload's parameters.
        """
        if dense is None:
            dense = self.__array__()

        params = self._codec.merge_compression_params(
            self._payload.compression_params, encode_kwargs
        )

        self._payload = self._codec.encode(dense, **params)
        self._overlay.clear()
