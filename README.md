# PyTorchEMD

PyTorch wrapper for approximate point-cloud Earth Mover's Distance (EMD).

Key changes fo this fork:
- works for newer PyTorch/CUDA builds
- normalizes the returned cost per source point
- optional point limiting (memory scales quadratically with point sizes)


## Requirements

- PyTorch with CUDA support
- A CUDA toolkit compatible with the active PyTorch installation
- A compiler toolchain supported by `torch.utils.cpp_extension`

This fork has been used with PyTorch 2.4.1 and CUDA 12.x.

## Installation

From this repository root:

```bash
pip install -e .
```

For local source-tree use without installing the Python modules, build the CUDA
extension in place and keep this directory on `PYTHONPATH`:

```bash
python setup.py build_ext --inplace
```

The previous manual step of copying `emd_cuda*.so` out of `build/lib...` is no
longer required.

## Usage

```python
from earth_mover_distance import earth_mover_distance

d = earth_mover_distance(p1, p2, transpose=False, point_limit=2048)
```

Input shapes:

- `transpose=False`: `p1` and `p2` are expected as `B x N x 3`.
- `transpose=True`: `p1` and `p2` are expected as `B x 3 x N` and are
  transposed before calling the CUDA extension.

`point_limit` is optional. If set to a positive integer, point clouds larger
than the limit are deterministically downsampled before EMD is computed, and a
warning is emitted. Use `point_limit=None` or `point_limit <= 0` to disable
limiting.

The returned cost is normalized by the number of source points.

The legacy import path is still available:

```python
from emd import earth_mover_distance
```

Run the test with:

```bash
python test_emd_loss.py
```

## Attribution

This fork is based on the original
[daerduoCarey/PyTorchEMD](https://github.com/daerduoCarey/PyTorchEMD)
repository.

The CUDA code was originally written by Haoqiang Fan. The PyTorch wrapper was
written by Kaichun Mo, with help from Jiayuan Gu.

## License

MIT
