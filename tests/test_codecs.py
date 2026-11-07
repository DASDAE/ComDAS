"""
Codec tests. Defines a base codec test class for tests all codecs should run and pass.

Each codec should create a concrete test class that inherits from ``CodecContract``,
implement make_codec, and include additional codec-specific tests.
"""

from abc import ABC, abstractmethod

import dascore as dc
import h5py
import numpy as np
import pytest

from comdas import write_compressed
from comdas.arrays.duck_array import DuckArray
from comdas.codecs.base import Codec
from comdas.codecs.svd import SVDCodec
from comdas.codecs.wavelet import WaveletCodec


def make_random_matrix(m=40, n=60, seed=42):
    """
    Build a random test matrix.
    """
    return np.random.default_rng(seed).normal(size=(m, n))


class CodecContract(ABC):
    """
    Tests inherited by every concrete codec test class.
    """

    @abstractmethod
    def make_codec(self) -> Codec:
        """
        Return a codec configured to encode the shared test array.
        """

    def test_name_registers(self):
        """
        Check the codec gets registered correctly.
        """
        codec = self.make_codec()
        assert codec.name
        assert Codec.get_registered(codec.name) is type(codec)

    def test_file_roundtrip(self, tmp_path):
        """
        Test that the codec read/writes to a file.
        """
        codec = self.make_codec()
        array = make_random_matrix()
        example = dc.get_example_patch()
        patch = example.new(
            data=array,
            coords={
                dim: example.coords.get_coord(dim)[:length]
                for dim, length in zip(example.dims, array.shape)
            },
        )
        path = tmp_path / "compressed.h5"

        write_compressed(patch, path, codec)
        read_back = list(dc.spool(path))

        assert len(read_back) == 1
        assert isinstance(read_back[0].data, DuckArray)
        assert read_back[0].data.codec.name == codec.name
        np.testing.assert_allclose(
            np.asarray(read_back[0].data), codec.decode(codec.encode(array))
        )

    def test_payload_group_roundtrip(self, tmp_path):
        """
        Test that the codec can write its payload to an HDF5 group and read it back.
        """
        codec = self.make_codec()
        array = make_random_matrix()
        payload = codec.encode(array)
        path = tmp_path / "payload.h5"

        with h5py.File(path, "w") as file:
            group = file.create_group("payload")
            codec.payload_to_group(payload, group)

        with h5py.File(path, "r") as file:
            restored = codec.payload_from_group(
                file["payload"],
                original_shape=payload.original_shape,
                dtype=payload.dtype,
            )

        assert restored.original_shape == array.shape
        assert restored.dtype == array.dtype
        np.testing.assert_allclose(
            codec.decode(restored), codec.decode(payload)
        )

    def test_decode_partial_matches_full_decode(self):
        """
        If partial decoding is implemented for the codec, check that it returns
        the same values as the full decode.
        """
        codec = self.make_codec()
        if type(codec).decode_partial is Codec.decode_partial:
            pytest.skip("codec does not implement decode_partial")

        payload = codec.encode(make_random_matrix())
        full = codec.decode(payload)
        for key in [
            (3, 4),
            (slice(2, 10), slice(5, 20)),
            (5, slice(None)),
            (slice(None), 7),
        ]:
            np.testing.assert_allclose(
                codec.decode_partial(payload, key), full[key]
            )


class TestSVDCodec(CodecContract):
    @staticmethod
    def make_low_rank(m=40, n=60, rank=5, seed=42):
        """
        Build a deterministic exactly-rank-``rank`` test matrix.
        """
        rng = np.random.default_rng(seed)
        array = rng.normal(size=(m, rank)) @ rng.normal(size=(rank, n))
        return array.astype(np.float64)

    def make_codec(self) -> SVDCodec:
        return SVDCodec(rank=6)

    def test_encode_requires_2d(self):
        """
        SVD only supports 2D arrays.
        """
        with pytest.raises(ValueError):
            self.make_codec().encode(np.zeros(10))

    def test_encode_requires_exactly_one_of_rank_or_energy(self):
        """
        SVDCodec requires exactly one of 'rank' or 'energy' to be specified.
        """
        codec = SVDCodec()
        array = make_random_matrix()
        with pytest.raises(ValueError):
            codec.encode(array)
        with pytest.raises(ValueError):
            codec.encode(array, rank=5, energy=0.9)

    def test_encode_decode_exact_rank(self):
        """
        Test that the rank 5 compressed version of a rank 5 matrix
        is the original matrix. i.e. that SVD is working correctly.
        """
        array = self.make_low_rank(rank=5)
        codec = SVDCodec(rank=5)
        payload = codec.encode(array)
        np.testing.assert_allclose(array, codec.decode(payload), atol=1e-8)

    def test_energy_truncation_picks_good_rank(self):
        """
        If we give SVD a rank 5 matrix and ask it to retain
        nearly all information, it should pick rank 5.
        """
        codec = SVDCodec(energy=0.999)
        payload = codec.encode(self.make_low_rank(rank=5))
        assert payload.rank <= 5

    def test_decode_partial_falls_back_for_fancy_indexing(self):
        """
        SVD codec falls back to full decoding when fancy indexing is used.
        This should give a warning so that the user is aware of the behavior.
        """
        codec = self.make_codec()
        payload = codec.encode(make_random_matrix())
        rows = np.array([1, 3, 5])
        cols = np.array([2, 4])
        key = np.ix_(rows, cols)
        with pytest.warns(UserWarning):
            partial = codec.decode_partial(payload, key)
        np.testing.assert_allclose(partial, codec.decode(payload)[key])

    def test_duck_array_with_svd_codec(self):
        """
        Test that a DuckArray wrapping an SVD-compressed payload behaves correctly.
        """
        codec = self.make_codec()
        array = make_random_matrix()
        payload = codec.encode(array)
        compressed = DuckArray(payload, codec)

        assert compressed.shape == array.shape
        np.testing.assert_allclose(
            np.asarray(compressed), codec.decode(payload)
        )
        np.testing.assert_allclose(
            compressed[3, 4], codec.decode(payload)[3, 4]
        )

    def test_consolidate_honors_rank_override(self):
        """
        Explicit encode kwargs passed to consolidate should be applied,
        while omitting them keeps the payload's existing settings.
        """
        array = make_random_matrix()
        codec = SVDCodec(rank=4)

        ca = DuckArray(codec.encode(array), codec)
        ca.consolidate()
        assert ca.payload.rank == 4

        ca.consolidate(rank=1)
        assert ca.payload.rank == 1

        # Switching to energy mode must drop the stored rank rather than
        # raising for specifying both.
        ca = DuckArray(codec.encode(array), codec)
        ca.consolidate(energy=0.5)
        assert ca.payload.rank < 4

        with pytest.raises(ValueError):
            ca.consolidate(rank=2, energy=0.9)

    def test_svd_rejects_excess_indices(self):
        """
        Indexing a 2D array with more than two indices should raise
        IndexError.
        """
        codec = self.make_codec()
        array = make_random_matrix()
        payload = codec.encode(array)
        dense = codec.decode(payload)

        with pytest.raises(IndexError):
            codec.decode_partial(payload, (slice(None), slice(None), 0))
        with pytest.raises(IndexError):
            DuckArray(payload, codec)[:, :, 0]

        for key in [(Ellipsis, 3), (np.newaxis, 2, slice(1, 4))]:
            with pytest.warns(UserWarning):
                partial = codec.decode_partial(payload, key)
            np.testing.assert_allclose(partial, dense[key])
        np.testing.assert_allclose(
            codec.decode_partial(payload, (-1, -2)), dense[-1, -2]
        )


class TestWaveletCodec(CodecContract):
    def make_codec(self) -> WaveletCodec:
        """
        Codec instance for testing. Keeps 20% of wavelet coefficients.
        """
        return WaveletCodec(wavelet="db4", keep_fraction=0.2)

    def test_encode_requires_exactly_one_of_keep_fraction_or_threshold(self):
        """
        WaveletCodec requires exactly one of 'keep_fraction' or 'threshold'.
        """
        codec = WaveletCodec()
        array = make_random_matrix()
        with pytest.raises(ValueError):
            codec.encode(array)
        with pytest.raises(ValueError):
            codec.encode(array, keep_fraction=0.5, threshold=0.1)
        with pytest.raises(ValueError):
            WaveletCodec(keep_fraction=0.5, threshold=0.1)

    def test_invalid_keep_fraction_raises(self):
        """
        Invalid keep_fraction values should raise a ValueError.
        """
        with pytest.raises(ValueError):
            WaveletCodec(keep_fraction=0.0).encode(make_random_matrix())
        with pytest.raises(ValueError):
            WaveletCodec(keep_fraction=1.5).encode(make_random_matrix())

    @pytest.mark.parametrize("shape", [(40, 60), (41, 63), (1, 100)])
    def test_keep_all_is_lossless(self, shape):
        """
        Keeping every coefficient should be lossless.
        """
        array = np.random.default_rng(0).normal(size=shape)
        codec = WaveletCodec(wavelet="db4", keep_fraction=1.0)
        np.testing.assert_allclose(
            codec.decode(codec.encode(array)), array, atol=1e-10
        )

    def test_keep_fraction_controls_coefficient_count(self):
        """
        Checks that keep fraction calculates the number of coeffs to keep correctly.
        """
        codec = WaveletCodec(keep_fraction=0.1)
        payload = codec.encode(make_random_matrix())
        assert payload.values.size == round(0.1 * payload.n_coeffs)
        assert codec.compression_ratio(payload) > 1

    def test_threshold_keeps_large_coefficients(self):
        """
        Only coefficients larger than the threshold should be kept.
        """
        codec = WaveletCodec(threshold=1.0)
        payload = codec.encode(make_random_matrix())
        assert np.all(np.abs(payload.values) > 1.0)

    def test_zero_wavelet_payload_has_usable_repr(self):
        """
        A payload with no retained coefficients (e.g. all-zero data with a
        threshold) should decode, have an infinite compression
        ratio, and have a working repr.
        """
        codec = WaveletCodec(threshold=1)
        payload = codec.encode(np.zeros((32, 32)))
        assert payload.values.size == 0

        ca = DuckArray(payload, codec)
        np.testing.assert_array_equal(np.asarray(ca), np.zeros((32, 32)))
        assert codec.compression_ratio(payload) == float("inf")
        assert "DuckArray" in repr(ca)

    def test_threshold_payload_consolidates_with_its_threshold(self, tmp_path):
        """
        A threshold-encoded payload should re-encode with the same threshold,
        including after a file roundtrip, and including when no
        coefficients were kept.
        """
        codec = WaveletCodec(threshold=1)
        payload = codec.encode(np.zeros((32, 32)))
        assert payload.compression_params["threshold"] == 1

        ca = DuckArray(payload, codec)
        ca.consolidate()
        np.testing.assert_array_equal(np.asarray(ca), np.zeros((32, 32)))

        # A write large enough to auto-consolidate.
        ca[:, :] = 5.0
        np.testing.assert_allclose(np.asarray(ca), 5.0)

        with h5py.File(tmp_path / "wavelet.h5", "w") as f:
            codec.payload_to_group(payload, f.create_group("p"))
            restored = codec.payload_from_group(
                f["p"],
                original_shape=payload.original_shape,
                dtype=payload.dtype,
            )
        assert restored.compression_params["threshold"] == 1

        ca.consolidate(keep_fraction=0.5)
        assert "threshold" not in ca.payload.compression_params


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
