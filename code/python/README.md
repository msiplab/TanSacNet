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

## Examples

| Example | MATLAB counterpart | Contents |
|---|---|---|
| [`main_lsun.ipynb`](../examples/lsun/main_lsun.ipynb) | `examples/lsun/main_lsun.m` | LSUN as a linear transform like PCA, and its training |

Install with `pip install "./code/python[examples]"`.

## Tests

```sh
pip install "./code/python[test]"
cd code/python
python -m pytest
```

The tests are placed next to the modules (`torch_tansacnet/test_*.py`)
and are not included in the wheel.
