"""
Tests for the ``COMDAS`` HDF5 container.
"""

import numpy as np
import dascore as dc
import h5py
import pytest

import comdas
from comdas import write_compressed

try:
    from _dummy_codec import DummyCodec
except ImportError:
    from tests._dummy_codec import DummyCodec


def test_write_read_roundtrip(tmp_path):
    """
    A patch written via write_compressed should read back, via
    dc.spool(), with matching shape/coords/attrs.
    """
    patch = dc.get_example_patch()
    out_path = tmp_path / "test.h5"

    write_compressed(patch, out_path, DummyCodec())

    # dc.spool() itself should read the format without a ComDAS specific call
    patches = list(dc.spool(out_path))
    assert len(patches) == 1
    p2 = patches[0]

    assert p2.dims == patch.dims
    assert p2.data.shape == patch.data.shape

    # DummyCodec doesn't introduce error
    assert np.array_equal(np.asarray(p2.data), patch.data)

    for dim in patch.dims:
        assert np.array_equal(
            p2.coords.get_array(dim), patch.coords.get_array(dim)
        )

    assert p2.attrs.tag == patch.attrs.tag


def test_write_compressed_from_existing_spool(tmp_path):
    """
    write_compressed should also accept a Spool.
    """
    p1 = dc.get_example_patch()
    spool = dc.spool([p1])
    out_path = tmp_path / "from_spool.h5"

    write_compressed(spool, out_path, DummyCodec())

    assert len(list(dc.spool(out_path))) == 1


def test_scan_does_not_load_arrays(tmp_path, monkeypatch):
    """
    dc.scan() should return correct attrs without reading the codec-specific arrays.
    """
    patch = dc.get_example_patch()
    out_path = tmp_path / "test_scan.h5"
    write_compressed(patch, out_path, DummyCodec())

    original_getitem = h5py.Dataset.__getitem__

    def reject_payload_read(dataset, key):
        if dataset.name.endswith("/data"):
            raise AssertionError("scan loaded the codec payload")
        return original_getitem(dataset, key)

    with monkeypatch.context() as patcher:
        patcher.setattr(h5py.Dataset, "__getitem__", reject_payload_read)
        attrs_list = dc.scan(out_path)

    assert len(attrs_list) == 1
    assert attrs_list[0].tag == patch.attrs.tag
    assert attrs_list[0].file_format == "COMDAS"


def test_append_and_dim_range_selection(tmp_path):
    """
    Writing twice to the same file should append, and dim-range selection should
    pick the right patch out of a multi-patch file rather than always the first.
    """
    p1 = dc.get_example_patch()
    p2 = (
        dc.get_example_patch()
        .update_attrs(tag="second")
        .update_coords(
            time=p1.coords.get_array("time") + np.timedelta64(1, "h")
        )
    )
    out_path = tmp_path / "multi.h5"

    codec = DummyCodec()
    write_compressed(p1, out_path, codec)
    write_compressed(p2, out_path, codec)

    read_back = list(dc.spool(out_path))
    assert len(read_back) == 2
    tags = {p.attrs.tag for p in read_back}
    assert tags == {"random", "second"}

    times = p2.coords.get_array("time")
    selected = list(dc.spool(out_path).select(time=(times[0], times[-1])))
    assert len(selected) == 1
    assert selected[0].attrs.tag == "second"


def test_codec_recorded(tmp_path):
    """
    The codec used should be discoverable from the file without additional info.
    """
    patch = dc.get_example_patch()
    out_path = tmp_path / "codec_check.h5"
    write_compressed(patch, out_path, DummyCodec())

    p2 = list(dc.spool(out_path))[0]
    assert isinstance(p2.data, comdas.DuckArray)
    assert p2.data.codec.name == DummyCodec.name


def _patch_with_aux_coords():
    """
    A small patch carrying a (irregular) latitude auxiliary coord
    associated with distance, plus a 2D auxiliary coord.
    """
    n_dist, n_time = 8, 32
    distance = np.arange(n_dist) * 2.0
    time = np.datetime64("2026-01-01", "ns") + np.arange(
        n_time
    ) * np.timedelta64(2, "ms")
    # Irregular spacing so the values can't be rebuilt from a min/max/step summary.
    latitude = 39.7 + np.cumsum(np.linspace(0.001, 0.02, n_dist))
    quality = np.arange(n_dist * n_time, dtype=np.float64).reshape(
        n_dist, n_time
    )
    return dc.Patch(
        data=np.arange(n_dist * n_time, dtype=np.float32).reshape(
            n_dist, n_time
        ),
        dims=("distance", "time"),
        coords={
            "distance": dc.get_coord(data=distance, units="m"),
            "time": dc.get_coord(data=time, units="s"),
            "latitude": (
                "distance",
                dc.get_coord(data=latitude, units="degree"),
            ),
            "quality": (("distance", "time"), quality),
        },
        attrs={"tag": "aux"},
    )


def _roundtrip(patch, tmp_path, name="aux.h5"):
    """Write ``patch`` with the DummyCodec and read it back via dc.spool."""
    out_path = tmp_path / name
    write_compressed(patch, out_path, DummyCodec())
    patches = list(dc.spool(out_path))
    assert len(patches) == 1
    return patches[0], out_path


def test_auxiliary_coords_roundtrip(tmp_path):
    """
    Non-dimensional coords (e.g. latitude associated with distance) should be
    stored and restored with their values, dtypes, and dim associations.
    """
    patch = _patch_with_aux_coords()
    p2, _ = _roundtrip(patch, tmp_path)

    assert set(p2.coords.coord_map) == set(patch.coords.coord_map)
    for name in patch.coords.coord_map:
        expected = patch.coords.get_array(name)
        got = p2.coords.get_array(name)
        assert got.dtype == expected.dtype, name
        assert np.array_equal(got, expected), name
        assert p2.coords.dim_map[name] == patch.coords.dim_map[name], name


def test_auxiliary_coord_follows_dim_selection(tmp_path):
    """
    After reading back, slicing distance should also slice latitude.
    """
    patch = _patch_with_aux_coords()
    p2, _ = _roundtrip(patch, tmp_path)

    dist = patch.coords.get_array("distance")
    lo, hi = dist[2], dist[5]
    expected = patch.select(distance=(lo, hi)).coords.get_array("latitude")
    got = p2.select(distance=(lo, hi)).coords.get_array("latitude")
    assert np.array_equal(got, expected)


def test_dim_coord_units_roundtrip(tmp_path):
    """
    Units of ordinary dimension coords (no auxiliary coords involved)
    should survive the roundtrip.
    """
    patch = dc.get_example_patch()
    p2, _ = _roundtrip(patch, tmp_path, name="units.h5")

    for dim in patch.dims:
        expected = patch.coords.get_coord(dim).units
        assert expected is not None
        assert p2.coords.get_coord(dim).units == expected


def test_coord_units_roundtrip(tmp_path):
    """
    Coordinate units should survive the write/read roundtrip.
    """
    patch = _patch_with_aux_coords()
    p2, _ = _roundtrip(patch, tmp_path)

    for name in ("distance", "time", "latitude"):
        assert (
            p2.coords.get_coord(name).units
            == patch.coords.get_coord(name).units
        )


def test_coord_without_units_roundtrip(tmp_path):
    """
    A coord with no units should read back with no units.
    """
    patch = _patch_with_aux_coords()
    p2, _ = _roundtrip(patch, tmp_path)

    assert patch.coords.get_coord("quality").units is None
    assert p2.coords.get_coord("quality").units is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
