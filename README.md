# ComDAS

![Python versions tested](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)

ComDAS (Compressed DAS) lets you store and work with compressed distributed acoustic sensing (DAS) data using ordinary [DASCore](https://dascore.org/) tools.

With ComDAS you can:

- **Shrink DAS data in memory.** Compress a DASCore patch and keep using it like any other.
- **Save compressed files.** Write patches, spools, or whole directories of raw data to a compressed HDF5 file or files.
- **Read compressed files with plain DASCore.** Once ComDAS is installed, `dascore.spool()` opens compressed files directly without needing to know how the data was compressed.
- **Choose your compression level.** Pick a codec and set how much detail to keep, from near-lossless to aggressive compression.
- **Implement your own algorithm.** Add a new compression method by writing a codec class.

> [!NOTE]
> ComDAS is in active development. The file format and public API may change between releases.

## Supported codecs

| Codec | Main setting | Reference |
| --- | --- | --- |
| [SVD](https://dasdae.github.io/ComDAS/codecs/svd.html) | `rank` or retained `energy` | [NumPy SVD](https://numpy.org/doc/1.22/reference/generated/numpy.linalg.svd.html) |
| [Wavelet](https://dasdae.github.io/ComDAS/codecs/wavelet.html) | `keep_fraction` or `threshold` | [PyWavelets](https://doi.org/10.21105/joss.01237); [DeVore et al. (1992)](https://doi.org/10.1109/18.119733) |

## Installation

ComDAS requires Python 3.10 or newer.

```bash
pip install comdas
```

## Documentation

Full documentation is available at [dasdae.github.io/ComDAS](https://dasdae.github.io/ComDAS/), including:

- [Quick start](https://dasdae.github.io/ComDAS/quickstart.html): compress a patch, save a compressed file, and read it back
- [Codecs](https://dasdae.github.io/ComDAS/codecs/svd.html): configuration options and examples for each codec
- [Examples](https://dasdae.github.io/ComDAS/examples/compression_comparison.html)
- [API reference](https://dasdae.github.io/ComDAS/api/index.html)

## License

ComDAS is licensed under the [GNU Lesser General Public License](LICENSE).
