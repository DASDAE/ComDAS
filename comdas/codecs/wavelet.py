"""
Wavelet compression via PyWavelets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pywt

from comdas.codecs.base import Codec, CompressedPayload


@dataclass
class WaveletPayload(CompressedPayload):
    """
    Thresholded wavelet-coefficient representation of an array.

    Retained coefficients are stored as ``(indices, values)`` pairs.

    :ivar wavelet: Name of the PyWavelets wavelet.
    :vartype wavelet: str
    :ivar mode: PyWavelets signal extension mode.
    :vartype mode: str
    :ivar indices: Flat indices of retained coefficients.
    :vartype indices: numpy.ndarray
    :ivar values: Values of retained coefficients.
    :vartype values: numpy.ndarray
    :ivar threshold: The absolute threshold used to select coefficients, if
        encoded in threshold mode. ``None`` if encoded with ``keep_fraction``.
    :vartype threshold: float or None
    """

    wavelet: str
    mode: str
    indices: np.ndarray
    values: np.ndarray
    threshold: float | None = None

    @property
    def coeff_shapes(self) -> list:
        """
        Get the coefficient shapes of the decomposition before thresholding.

        :rtype: list
        """
        return pywt.wavedecn_shapes(
            self.original_shape, self.wavelet, mode=self.mode
        )

    @property
    def n_coeffs(self) -> int:
        """
        Total number of coefficients in the full decomposition without thresholding.

        :rtype: int
        """
        shapes = self.coeff_shapes
        total = int(np.prod(shapes[0]))
        for details in shapes[1:]:
            total += sum(int(np.prod(s)) for s in details.values())
        return total

    @property
    def compression_params(self) -> dict:
        """
        The compression parameters used for this payload.

        Threshold-encoded payloads report their ``threshold``, so re-encoding
        applies the same rule (a derived ``keep_fraction`` could be 0, which
        isn't a valid input). Otherwise the retained ``keep_fraction``.

        :rtype: dict
        """
        params = {"wavelet": self.wavelet, "mode": self.mode}
        if self.threshold is not None:
            params["threshold"] = self.threshold
        else:
            params["keep_fraction"] = self.values.size / self.n_coeffs
        return params


class WaveletCodec(Codec):
    """
    Wavelet-thresholding codec.

    The array is decomposed with a multilevel n-dimensional discrete wavelet
    transform (:func:`pywt.wavedecn`) at the maximum useful level for the
    array shape, and only the largest-magnitude coefficients are retained.

    Coefficients to keep can be chosen either as a fraction ``keep_fraction`` in
    ``(0, 1]`` of all coefficients, or as an absolute magnitude
    ``threshold`` where coefficients with ``|c| > threshold`` are kept. Exactly
    one option must be specified, either as a codec-level default or
    per-call parameter.

    :param wavelet: PyWavelets wavelet name. (default: ``"db4"``)
    :type wavelet: str
    :param mode: PyWavelets signal extension mode. (default: ``"symmetric"``)
    :type mode: str
    :param keep_fraction: Default fraction of coefficients to retain.
    :type keep_fraction: float | None
    :param threshold: Default absolute coefficient magnitude threshold.
    :type threshold: float | None
    :raises ValueError: If both of ``keep_fraction`` and ``threshold`` are specified.
    """

    name = "WAVELET"
    exclusive_params = ("keep_fraction", "threshold")
    version = "1"

    def __init__(
        self,
        wavelet: str = "db4",
        mode: str = "symmetric",
        keep_fraction: float | None = None,
        threshold: float | None = None,
    ):
        if keep_fraction is not None and threshold is not None:
            raise ValueError(
                "Provide at most one of keep_fraction / threshold to act as codec default."
            )
        self.default_wavelet = wavelet
        self.default_mode = mode
        self.default_keep_fraction = keep_fraction
        self.default_threshold = threshold

    def encode(
        self,
        array: np.ndarray,
        *,
        wavelet: str | None = None,
        mode: str | None = None,
        keep_fraction: float | None = None,
        threshold: float | None = None,
        **kwargs,
    ) -> WaveletPayload:
        """
        Compress an array via wavelet coefficient thresholding.

        :param array: The dense array to compress.
        :type array: numpy.ndarray
        :param wavelet: Overrides ``self.default_wavelet`` for this call.
        :type wavelet: str or None
        :param mode: Overrides ``self.default_mode`` for this call.
        :type mode: str or None
        :param keep_fraction: Fraction of coefficients to retain, in ``(0, 1]``.
            Overrides ``self.default_keep_fraction`` for this call.
        :type keep_fraction: float or None
        :param threshold: Absolute coefficient magnitude threshold.
            Overrides ``self.default_threshold`` for this call.
        :type threshold: float or None
        :param kwargs: Accepted and ignored for compatibility.
        :returns: The sparse wavelet payload.
        :rtype: WaveletPayload
        :raises ValueError: If both or neither of ``keep_fraction`` and
            ``threshold`` are specified, or either is out of range.
        """
        array = np.asarray(array)
        wavelet = wavelet if wavelet is not None else self.default_wavelet
        mode = mode if mode is not None else self.default_mode

        if keep_fraction is not None and threshold is not None:
            raise ValueError(
                "Specify only one of `keep_fraction` or `threshold` per call or use the codec default."
            )
        if keep_fraction is None and threshold is None:
            keep_fraction = self.default_keep_fraction
            threshold = self.default_threshold
        if (keep_fraction is None) == (threshold is None):
            raise ValueError(
                "Exactly one of `keep_fraction` or `threshold` must be specified, "
                "either here or as a codec default."
            )
        if keep_fraction is not None and not 0 < keep_fraction <= 1:
            raise ValueError(
                f"`keep_fraction` must be in (0, 1], got {keep_fraction}"
            )
        if threshold is not None and threshold < 0:
            raise ValueError(f"`threshold` must be >= 0, got {threshold}")

        coeffs = pywt.wavedecn(array, wavelet, mode=mode)
        flat = pywt.ravel_coeffs(coeffs)[0]
        magnitude = np.abs(flat)

        if keep_fraction is not None:
            k = max(1, min(flat.size, int(round(keep_fraction * flat.size))))
            indices = np.argpartition(magnitude, -k)[-k:]
            indices.sort()
        else:
            indices = np.flatnonzero(magnitude > threshold)

        index_dtype = (
            np.uint32 if flat.size <= np.iinfo(np.uint32).max else np.uint64
        )
        return WaveletPayload(
            original_shape=array.shape,
            dtype=array.dtype,
            wavelet=wavelet,
            mode=mode,
            indices=indices.astype(index_dtype),
            values=np.ascontiguousarray(flat[indices]),
            threshold=float(threshold) if threshold is not None else None,
        )

    def decode(self, payload: WaveletPayload) -> np.ndarray:
        """
        Reconstruct the dense array via the inverse wavelet transform.

        :param payload: A payload previously produced by :meth:`encode`.
        :type payload: WaveletPayload
        :returns: The reconstructed dense array.
        :rtype: numpy.ndarray
        """
        shapes = payload.coeff_shapes
        template = [np.zeros(shapes[0])] + [
            {key: np.zeros(shape) for key, shape in details.items()}
            for details in shapes[1:]
        ]
        _, slices, coeff_shapes = pywt.ravel_coeffs(template)
        flat = np.zeros(payload.n_coeffs, dtype=payload.values.dtype)
        flat[payload.indices] = payload.values

        coeffs = pywt.unravel_coeffs(
            flat, slices, coeff_shapes, output_format="wavedecn"
        )
        out = pywt.waverecn(coeffs, payload.wavelet, mode=payload.mode)
        out = out[tuple(slice(0, n) for n in payload.original_shape)]
        return out.astype(payload.dtype, copy=False)

    def compressed_size_bytes(self, payload: WaveletPayload) -> int:
        """
        Get payload size.

        :param payload: A payload previously produced by :meth:`encode`.
        :type payload: WaveletPayload
        :returns: Combined byte size of the stored arrays.
        :rtype: int
        """
        return payload.indices.nbytes + payload.values.nbytes

    def payload_to_group(self, payload: WaveletPayload, group) -> None:
        """
        Write ``indices``, ``values``, and transform parameters into an
        open ``h5py.Group``.

        :param payload: The payload to serialize.
        :type payload: WaveletPayload
        :param group: An open, writable ``h5py.Group`` dedicated to
            this one patch.
        :type group: h5py.Group
        """
        group.create_dataset("indices", data=payload.indices)
        group.create_dataset("values", data=payload.values)
        group.attrs["wavelet"] = payload.wavelet
        group.attrs["mode"] = payload.mode
        if payload.threshold is not None:
            group.attrs["threshold"] = payload.threshold

    def payload_from_group(
        self, group, *, original_shape: tuple[int, ...], dtype: np.dtype
    ) -> WaveletPayload:
        """
        Construct a :class:`WaveletPayload` from an HDF5 group.

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
        :rtype: WaveletPayload
        """
        return WaveletPayload(
            original_shape=original_shape,
            dtype=dtype,
            wavelet=str(group.attrs["wavelet"]),
            mode=str(group.attrs["mode"]),
            indices=group["indices"][:],
            values=group["values"][:],
            threshold=(
                float(group.attrs["threshold"])
                if "threshold" in group.attrs
                else None
            ),
        )
