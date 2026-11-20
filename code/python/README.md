# torch-tansacnet

PyTorch implementation of the LSUN (locally structured unitary network)
layers and networks of the [TanSacNet](https://github.com/msiplab/TanSacNet)
MATLAB package (`+tansacnet/+lsun`).

## Installation

Requires Python 3.10 or later and PyTorch 2.3 or later. Install PyTorch
first if you need a specific CUDA build (see <https://pytorch.org>), then:

```sh
# From a clone of the repository
pip install ./code/python

# Or directly from GitHub
pip install "torch-tansacnet @ git+https://github.com/msiplab/TanSacNet.git@feature/pytorch#subdirectory=code/python"
```

The package is imported as `torch_tansacnet`.

Optional extras:

| Extra | Adds | For |
|---|---|---|
| `torch-tansacnet[test]` | pytest, parameterized, scipy, torch-dct | Running the tests |
| `torch-tansacnet[examples]` | torchvision, matplotlib, requests, torchinfo, ipykernel | The example notebook |
| `torch-tansacnet[dev]` | all of the above, build | Development |

## Quick start

```python
import torch
from torch_tansacnet import LsunAnalysis2dNetwork, OrthonormalTransform

analyzer = LsunAnalysis2dNetwork(
    input_size=[32, 32],  # [height, width]
    stride=[2, 2],  # decimation factors (vertical, horizontal)
    overlapping_factor=[3, 3],
    number_of_levels=0,  # 0: flat (single-level) structure
)

def init_angles(m):
    if isinstance(m, OrthonormalTransform):
        torch.nn.init.normal_(m.angles, mean=0.0, std=torch.pi / 6)

analyzer.apply(init_angles)
synthesizer = analyzer.T  # adjoint (synthesis) network

x = torch.rand(8, 1, 32, 32)  # N x C x H x W
with torch.no_grad():
    coefs = analyzer(x)  # N x (H/2) x (W/2) x 4
    y = synthesizer(coefs)
print(torch.dist(x, y))  # perfect reconstruction
```

### 3-D volumes

`LsunAnalysis3dNetwork` and `LsunSynthesis3dNetwork` are the 3-D
counterparts, which correspond to MATLAB `fcn_createlsunlgraph3d`. The sizes
are given in the order of `[vertical, horizontal, depth]`, and the atoms are
extended in the depth, horizontal and vertical directions in this order.

```python
import torch
from torch_tansacnet import LsunAnalysis3dNetwork, OrthonormalTransform

analyzer = LsunAnalysis3dNetwork(
    input_size=[16, 16, 16],  # [height, width, depth]
    stride=[2, 2, 2],  # decimation factors (vertical, horizontal, depth)
    overlapping_factor=[3, 3, 3],
    number_of_levels=2,  # tree structure of two levels
    boundary_operation="Termination",
)

def init_angles(m):
    if isinstance(m, OrthonormalTransform):
        torch.nn.init.normal_(m.angles, mean=0.0, std=torch.pi / 6)

analyzer.apply(init_angles)
synthesizer = analyzer.T  # adjoint (synthesis) network

x = torch.rand(4, 1, 16, 16, 16)  # N x C x H x W x D
with torch.no_grad():
    # (DC of level 2, AC of level 2, AC of level 1):
    # N x 4 x 4 x 4 x 1, N x 4 x 4 x 4 x 7, N x 8 x 8 x 8 x 7
    coefs = analyzer(x)
    y = synthesizer(coefs)
print(torch.dist(x, y))  # perfect reconstruction
```

With `number_of_levels=0`, the output is a single tensor of
`N x (H/2) x (W/2) x (D/2) x 8`. Only a single component (`C = 1`) is
supported, as in the 2-D networks.

### 1-D sequences

`LsunAnalysis1dNetwork` and `LsunSynthesis1dNetwork` are the 1-D
counterparts, which correspond to MATLAB `fcn_createlsunlgraph1d`. They have
the same structure as the 2-D and 3-D networks (not the CS type of MATLAB
`fcn_createcslsunlgraph1d`). `input_size`, `stride` and `overlapping_factor`
are integers.

```python
import torch
from torch_tansacnet import LsunAnalysis1dNetwork, OrthonormalTransform

analyzer = LsunAnalysis1dNetwork(
    input_size=64,  # sequence length
    stride=4,  # decimation factor (even)
    overlapping_factor=3,  # odd
    number_of_levels=2,  # tree structure of two levels
    boundary_operation="Termination",
)

def init_angles(m):
    if isinstance(m, OrthonormalTransform):
        torch.nn.init.normal_(m.angles, mean=0.0, std=torch.pi / 6)

analyzer.apply(init_angles)
synthesizer = analyzer.T  # adjoint (synthesis) network

x = torch.rand(4, 1, 64)  # N x C x L
with torch.no_grad():
    # (DC of level 2, AC of level 2, AC of level 1):
    # N x 4 x 1, N x 4 x 3, N x 16 x 3
    coefs = analyzer(x)
    y = synthesizer(coefs)
print(torch.dist(x, y))  # perfect reconstruction
```

With `number_of_levels=0`, the output is a single tensor of
`N x (L/4) x 4`. Only a single component (`C = 1`) is supported.

> **Note**
> A stride of 2 gives one symmetric and one antisymmetric channel, so the
> rotation layers have no angles. The network then always realizes the block
> DCT (Haar transform) regardless of `overlapping_factor`. Use a stride of 4
> or larger for a trainable overlapped transform. A `UserWarning` is issued
> for a stride of 2 with `overlapping_factor` greater than 1.

## Examples

| Example | MATLAB counterpart | Contents |
|---|---|---|
| [`main_lsun.ipynb`](../examples/lsun/main_lsun.ipynb) | `examples/lsun/main_lsun.m` | LSUN as a linear transform like PCA, and its training |

Install with `pip install "./code/python[examples]"`.

## Boundary operation

`boundary_operation` of `LsunAnalysis2dNetwork` and `LsunSynthesis2dNetwork`
corresponds to `'BoundaryOperation'` of MATLAB `fcn_createlsunlgraph2d`:

- `'Circular'` (default): the image is extended periodically.
- `'Termination'`: the intermediate rotations right after the shift of the
  difference channels fix the rotation matrices of the boundary blocks to -I,
  so that no coefficients wrap around to the opposite boundary. The perfect
  reconstruction is kept, and the boundary blocks do not contribute to the
  gradient of the angles.

```python
analyzer = LsunAnalysis2dNetwork(
    input_size=[32, 32], stride=[2, 2], overlapping_factor=[3, 3],
    boundary_operation="Termination",
)
synthesizer = analyzer.T  # inherits boundary_operation
```

`LsunIntermediateRotation2dLayer` has `boundary_operation` and `direction`
(`'Vertical'`: the first row of blocks, `'Horizontal'`: the first column),
as the MATLAB properties `BoundaryOperation` and `Direction`. Note that the
blocks of the PyTorch layers are in row-major order, while those of MATLAB
are in column-major order.

The 3-D networks and `LsunIntermediateRotation3dLayer` have the same
options, with the additional direction `'Depth'` (the first layer of blocks).
The 1-D networks and `LsunIntermediateRotation1dLayer` take
`boundary_operation` only, where the boundary block is the first block.

## Tests

```sh
pip install "./code/python[test]"
cd code/python
python -m pytest
```

The tests are placed next to the modules (`torch_tansacnet/test_*.py`)
and are not included in the wheel.

`torch_tansacnet/test_lsunMatlabRefdata.py` compares the PyTorch
implementation with the reference data in `refdata`, which are generated by
MATLAB with `tools/matlab/gen_refdata.m` (run it in the folder `code` after
`setpath`). The reference data cover the intermediate rotation layer with
both boundary operations, including the gradients, and the analysis and
synthesis networks with random angles and mus, in 1-D, 2-D and 3-D. The 1-D
and 3-D data also cover the block DCT layer and a network of two tree levels;
`gen_refdata('Dimensions',3)` regenerates only the 3-D data. The angles and mus are
converted from the MATLAB layout (`n x nBlocks`, column-major blocks) to the
PyTorch layout (`nBlocks x n`, row-major blocks).
