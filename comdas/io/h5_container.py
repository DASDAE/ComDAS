"""
FiberIO plugin for the ComDAS HDF5 container format.

The ComDAS HDF5 container format is designed to store patches
in a codec-agnostic manner

Registers the ``COMDAS`` format with DASCore's FiberIO manager. Once
this module is imported (or the package is pip-installed, via the
``dascore.fiber_io`` entry point in ``pyproject.toml``), ``dc.spool(path)``
and ``dc.write(patches, path, file_format="COMDAS")`` work
like any other DASCore format regardless of which
:class:`~comdas.codecs.base.Codec` was used to compress each patch.

File layout
-----------
::

    /                                       (root)
      attrs:
        __format__ = "COMDAS"
        __COMDAS_version__ = "1"
    /patches
      /patch_0000
        attrs:
          codec_name         (str)   -- e.g. "SVD"; looked up in the codec registry
          codec_version      (str)   -- the codec's own version string
          compression_params (dict)  -- parameters used for compression
          original_shape     (tuple[int, ...])
          dtype              (str)   -- original array dtype
          dims               (str)   -- comma-separated dim names, in order
          attrs_json         (str)   -- PatchAttrs.model_dump_json()
        <codec-specific datasets/attrs, written by codec.payload_to_group>
        /coords
          <coord_name>            -- one dataset per coord (dimensional and
                                     auxiliary, e.g. latitude), full array
            attrs: dtype (str), units (str, empty if none),
                   dims_json (str) -- JSON list of the data dims this
                                      coord is associated with, e.g.
                                      ["distance"].
      /patch_0001
        ...

Everything above the codec-specific datasets is written and read by
this module. Only what's inside a patch's own group (beyond
``coords``) is delegated to ``codec.payload_to_group`` /
``codec.payload_from_group``.

Patches are appended, not overwritten: calling ``write`` again on an
existing file adds new ``patch_XXXX`` groups after the existing ones
(``H5Writer`` opens in ``"a"`` mode).

Once ``comdas`` is installed, plain ``dc.spool(path)`` finds and uses this
format automatically. Writing has one convenience function,
:func:`write_compressed`.
"""

from __future__ import annotations

import json

import numpy as np

import dascore as dc
from dascore.io import FiberIO
from dascore.utils.hdf5 import H5Reader, H5Writer

from comdas.arrays.duck_array import DuckArray
from comdas.codecs.base import Codec

FORMAT_NAME = "COMDAS"
FORMAT_VERSION = "1"
_PATCH_GROUP_PREFIX = "patch_"


def _patch_group_names(h5file) -> list[str]:
    """
    List patch group names in a COMDAS file, in write order.

    :param h5file: An open ``h5py.File`` for a COMDAS container.
    :returns: Sorted patch group names (empty if the file has none yet).
    :rtype: list[str]
    """
    if "patches" not in h5file:
        return []
    return sorted(h5file["patches"].keys())


def _next_patch_index(h5file) -> int:
    """
    The next unused ``patch_XXXX`` index for appending to ``h5file``.

    :param h5file: An open ``h5py.File`` for a COMDAS container.
    :returns: The next patch index to use.
    :rtype: int
    """
    names = _patch_group_names(h5file)
    if not names:
        return 0
    return int(names[-1][len(_PATCH_GROUP_PREFIX) :]) + 1


def _dump_coord_array(arr: np.ndarray) -> tuple[np.ndarray, str]:
    """
    Convert a coordinate array to something h5py can store directly.

    h5py can't store ``datetime64``/``timedelta64`` directly, so those
    are viewed as ``int64`` for storage; the exact original dtype
    string is returned so :func:`_load_coord_array` can cast back.

    :param arr: The coordinate array to prepare for storage.
    :type arr: numpy.ndarray
    :returns: A ``(storable_array, dtype_str)`` pair.
    :rtype: tuple[numpy.ndarray, str]
    """
    dtype_str = str(arr.dtype)
    if np.issubdtype(arr.dtype, np.datetime64) or np.issubdtype(
        arr.dtype, np.timedelta64
    ):
        return arr.astype("int64"), dtype_str
    return arr, dtype_str


def _load_coord_array(arr: np.ndarray, dtype_str: str) -> np.ndarray:
    """
    Invert :func:`_dump_coord_array`.

    :param arr: The array as read from the HDF5 dataset.
    :type arr: numpy.ndarray
    :param dtype_str: The original dtype string, as stored on write.
    :type dtype_str: str
    :returns: The coordinate array cast back to its original dtype.
    :rtype: numpy.ndarray
    """
    return arr.astype(dtype_str)


def _range_overlaps(requested, actual_min, actual_max) -> bool:
    """
    Whether a ``(lo, hi)`` request overlaps an ``[actual_min, actual_max]`` range.

    :param requested: A ``(lo, hi)`` tuple (either bound may be
        ``None`` for "unbounded"), or ``None`` for "no constraint".
    :param actual_min: The range's actual minimum, or ``None``.
    :param actual_max: The range's actual maximum, or ``None``.
    :returns: True if the two ranges overlap (or either is unbounded).
    :rtype: bool
    """
    if requested is None:
        return True
    lo, hi = requested
    if lo is not None and actual_max is not None and lo > actual_max:
        return False
    if hi is not None and actual_min is not None and hi < actual_min:
        return False
    return True


def _patch_matches_kwargs(attrs: dc.PatchAttrs, kwargs: dict) -> bool:
    """
    Whether a stored patch's coord ranges match DASCore's selection kwargs.

    :param attrs: The candidate patch's attrs (with a ``coords``
        summary mapping dim name to a min/max-bearing object).
    :type attrs: dascore.PatchAttrs
    :param kwargs: Keyword arguments as received by
        :meth:`ComdasV1.read`; only keys matching a dim name in
        ``attrs.coords`` are consulted.
    :returns: True if every dim-range constraint in ``kwargs`` overlaps
        this patch's own range for that dim.
    :rtype: bool
    """
    for dim, summary in attrs.coords.items():
        if dim not in kwargs or kwargs[dim] is None:
            continue
        if not _range_overlaps(kwargs[dim], summary.min, summary.max):
            return False
    return True


def _write_patch(
    patches_group, name: str, patch, codec: Codec, encode_kwargs: dict
) -> None:
    """
    Compress and write one patch into ``patches_group[name]``.

    :param patches_group: The open ``/patches`` HDF5 group.
    :type patches_group: h5py.Group
    :param name: The new patch group's name (e.g. ``"patch_0003"``).
    :type name: str
    :param patch: The (uncompressed) patch to write.
    :type patch: dascore.Patch
    :param codec: The codec to compress ``patch.data`` with.
    :type codec: Codec
    :param encode_kwargs: Forwarded to ``codec.encode``.
    :type encode_kwargs: dict
    :raises ValueError: If ``patch.data`` is not 2D (the only shape
        ComDAS codecs currently support).
    """
    dense = np.asarray(patch.data)
    if dense.ndim != 2:
        raise ValueError(
            "ComDAS codecs currently only support 2D patches (e.g. "
            f"channel x time), got shape {dense.shape} for dims {patch.dims}"
        )
    payload = codec.encode(dense, **encode_kwargs)

    grp = patches_group.create_group(name)
    grp.attrs["codec_name"] = codec.name
    grp.attrs["codec_version"] = codec.version
    grp.attrs["original_shape"] = payload.original_shape
    grp.attrs["dtype"] = str(payload.dtype)
    grp.attrs["dims"] = ",".join(patch.dims)
    grp.attrs["attrs_json"] = patch.attrs.model_dump_json()

    codec.payload_to_group(payload, grp)

    # Store every coord (dimensional and auxiliary, e.g. latitude), along
    # with which data dims it is associated with.
    coord_grp = grp.create_group("coords")
    for coord_name, coord in patch.coords.coord_map.items():
        arr = patch.coords.get_array(coord_name)
        storable, dtype_str = _dump_coord_array(arr)
        cds = coord_grp.create_dataset(coord_name, data=storable)
        cds.attrs["dtype"] = dtype_str
        cds.attrs["dims_json"] = json.dumps(
            list(patch.coords.dim_map[coord_name])
        )
        units = getattr(coord, "units", None)
        cds.attrs["units"] = str(units) if units is not None else ""


def _load_patch_attrs(raw_json: str) -> dc.PatchAttrs:
    """
    Load ``PatchAttrs`` from JSON written by ``PatchAttrs.model_dump_json``.

    DASCore serializes a missing coord ``step`` or ``units`` (e.g. for an
    irregular or unitless coordinate) as the string ``"None"``, which it
    then fails to parse back. Convert exactly those values to ``None``
    before validating.

    :param raw_json: The stored ``attrs_json`` string.
    :type raw_json: str
    :returns: The validated patch attrs.
    :rtype: dascore.PatchAttrs
    """
    values = json.loads(raw_json)
    for summary in (values.get("coords") or {}).values():
        if not isinstance(summary, dict):
            continue
        for field in ("step", "units"):
            if summary.get(field) == "None":
                summary[field] = None
    return dc.PatchAttrs.model_validate(values)


def _read_patch(patch_group) -> dc.Patch:
    """
    Read one patch group back into an ordinary compressed Patch.

    :param patch_group: An open patch group (e.g. ``f["patches"]["patch_0000"]``).
    :type patch_group: h5py.Group
    :returns: The reconstructed Patch, backed by a
        :class:`~comdas.arrays.duck_array.DuckArray`.
    :rtype: dascore.Patch
    """
    codec_cls = Codec.get_registered(patch_group.attrs["codec_name"])
    codec = codec_cls()
    original_shape = tuple(patch_group.attrs["original_shape"])
    dtype = np.dtype(patch_group.attrs["dtype"])
    payload = codec.payload_from_group(
        patch_group, original_shape=original_shape, dtype=dtype
    )
    compressed_array = DuckArray(payload, codec)

    dims = tuple(patch_group.attrs["dims"].split(","))
    coord_grp = patch_group["coords"]
    coords = {}
    for coord_name, ds in coord_grp.items():
        raw = ds[()]
        dtype_str = ds.attrs.get("dtype", str(raw.dtype))
        values = _load_coord_array(raw, dtype_str)
        units = ds.attrs.get("units", "") or None

        if "dims_json" in ds.attrs:
            coord_dims = tuple(json.loads(ds.attrs["dims_json"]))
        elif coord_name in dims:
            coord_dims = (coord_name,)
        else:
            raise ValueError(
                f"Missing dimension association for coordinate {coord_name!r}"
            )

        coords[coord_name] = (
            coord_dims,
            dc.get_coord(data=values, units=units),
        )

    attrs = _load_patch_attrs(patch_group.attrs["attrs_json"])

    return dc.Patch(
        data=compressed_array, coords=coords, dims=dims, attrs=attrs
    )


class ComdasV1(FiberIO):
    """
    FiberIO support for the codec-agnostic ComDAS HDF5 container, v1.

    Each patch stores which :class:`~comdas.codecs.base.Codec` it was
    compressed with. This means a single file can hold patches
    compressed with different codecs.
    """

    name = FORMAT_NAME
    version = FORMAT_VERSION
    preferred_extensions = ("h5", "hdf5")

    def get_format(
        self, resource: H5Reader, **kwargs
    ) -> tuple[str, str] | bool:
        """
        Identify a COMDAS file via its root-level magic attrs.

        :param resource: An open, readable HDF5 file (auto-opened by
            DASCore from a path via the ``H5Reader`` type hint).
        :type resource: h5py.File
        :param kwargs: Unused; accepted for FiberIO API compatibility.
        :returns: ``(format_name, format_version)`` if this file is a
            COMDAS container, else ``False``.
        :rtype: tuple[str, str] or bool
        """
        fmt = resource.attrs.get("__format__")
        if fmt == FORMAT_NAME:
            version = resource.attrs.get(
                f"__{FORMAT_NAME}_version__", FORMAT_VERSION
            )
            return FORMAT_NAME, str(version)
        return False

    def scan(self, resource: H5Reader, **kwargs) -> list[dc.PatchAttrs]:
        """
        Read every patch's attrs without touching datasets.

        :param resource: An open, readable COMDAS HDF5 file.
        :type resource: h5py.File
        :param kwargs: Unused; accepted for FiberIO API compatibility.
        :returns: One :class:`dascore.PatchAttrs` per stored patch.
        :rtype: list[dascore.PatchAttrs]
        """
        out = []
        for name in _patch_group_names(resource):
            grp = resource["patches"][name]
            attrs = _load_patch_attrs(grp.attrs["attrs_json"])
            attrs = attrs.update(
                file_format=self.name,
                file_version=self.version,
                path=str(resource.filename),
            )
            out.append(attrs)
        return out

    def read(self, resource: H5Reader, **kwargs) -> dc.BaseSpool:
        """
        Read patches.

        Honors dimension-range kwargs.

        :param resource: An open, readable COMDAS HDF5 file.
        :type resource: h5py.File
        :param kwargs: Optional dimension-range constraints (e.g.
            ``time=(t1, t2)``); any key not matching a coord's dim
            name is ignored.
        :returns: A spool of the matching patches.
        :rtype: dascore.BaseSpool
        """
        patches = []
        for name in _patch_group_names(resource):
            grp = resource["patches"][name]
            attrs = _load_patch_attrs(grp.attrs["attrs_json"])
            if _patch_matches_kwargs(attrs, kwargs):
                patches.append(_read_patch(grp))
        return dc.spool(patches)

    def write(
        self,
        spool,
        resource: H5Writer,
        *,
        codec: Codec | None = None,
        **encode_kwargs,
    ) -> None:
        """
        Compress and write a Patch/spool of Patches to the container.

        :param spool: A single Patch, or any iterable of Patches, to
            write.
        :type spool: dascore.Patch or dascore.BaseSpool or Sequence[dascore.Patch]
        :param resource: An open, writable HDF5 file (auto-opened by
            DASCore from a path via the ``H5Writer`` type hint, in
            append mode).
        :type resource: h5py.File
        :param codec: The codec to compress every patch in ``spool``
            with. (e.g. ``SVDCodec(rank=20)``.)
        :type codec: Codec
        :param encode_kwargs: Forwarded to ``codec.encode`` for every
            patch.
        :raises ValueError: If ``codec`` is not given.
        """
        if codec is None:
            raise ValueError(
                "A `codec` must be provided to write a COMDAS container "
                "(e.g. codec=SVDCodec(rank=20))."
            )
        resource.attrs["__format__"] = FORMAT_NAME
        resource.attrs[f"__{FORMAT_NAME}_version__"] = FORMAT_VERSION

        if "patches" not in resource:
            resource.create_group("patches")
        patches_group = resource["patches"]

        patches = [spool] if isinstance(spool, dc.Patch) else list(spool)
        idx = _next_patch_index(resource)
        for patch in patches:
            name = f"{_PATCH_GROUP_PREFIX}{idx:04d}"
            _write_patch(patches_group, name, patch, codec, encode_kwargs)
            idx += 1


def write_compressed(source, path, codec: Codec, **encode_kwargs) -> None:
    """
    Compress ``source`` and write it to ``path`` as a COMDAS container.

    This is the intended way to turn uncompressed DAS data
    into a compressed COMDAS file.

    :param source: Where the uncompressed data comes from (a file or
        directory path, a single Patch, a sequence of Patches, or an
        existing Spool).
    :type source: str or pathlib.Path or dascore.Patch or dascore.BaseSpool or Sequence[dascore.Patch]
    :param path: Destination path for the COMDAS container. If it
        already exists as a COMDAS file, patches are appended rather
        than overwriting existing ones.
    :type path: str or pathlib.Path
    :param codec: The codec to compress every patch with, e.g.
        ``SVDCodec(rank=20)``.
    :type codec: Codec
    :param encode_kwargs: Forwarded to ``codec.encode`` for every
        patch (e.g. ``rank=``/``energy=`` for
        :class:`~comdas.codecs.svd.SVDCodec`, overriding the codec's
        own defaults for this call).

    :Example:

    .. code-block:: python

        import dascore as dc
        from comdas import write_compressed, SVDCodec

        # from an existing uncompressed file
        write_compressed("raw.h5", "compressed.h5", SVDCodec(rank=20))

        # from a patch already in memory
        patch = dc.get_example_patch()
        write_compressed(patch, "compressed.h5", SVDCodec(rank=20))

        # reading back needs nothing ComDAS-specific
        read_back = dc.spool("compressed.h5")
    """
    spool = dc.spool(source)
    dc.write(
        spool,
        path,
        file_format=FORMAT_NAME,
        file_version=FORMAT_VERSION,
        codec=codec,
        **encode_kwargs,
    )
