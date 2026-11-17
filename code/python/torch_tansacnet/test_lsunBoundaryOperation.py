"""BoundaryOperation ('Circular' | 'Termination') of the LSUN intermediate
rotation layer and the analysis/synthesis networks.

This file checks the same properties as the MATLAB test classes
lsunIntermediateRotation2dLayerBoundaryOperationTestCase and
fcn_createlsunlgraph2dBoundaryOperationTestCase. The values are compared with
MATLAB in test_lsunMatlabRefdata.py.

Data layout: nSamples x nRows x nCols x nChs. Blocks are in row-major order,
so that the boundary blocks are
    'Vertical'   : the first row,    iblk = 0, ..., nCols-1
    'Horizontal' : the first column, iblk = 0, nCols, 2*nCols, ...

Requirements: Python 3.10-12.x, PyTorch 2.3/4.x

Copyright (c) 2026, Shogo MURAMATSU

All rights reserved.

Contact address: Shogo MURAMATSU,
    Faculty of Engineering, Niigata University,
    8050 2-no-cho Ikarashi, Nishi-ku,
    Niigata, 950-2181, JAPAN

https://www.eng.niigata-u.ac.jp/~msiplab/
"""

import math

import pytest
import torch

from torch_tansacnet.lsunAnalysis2dNetwork import LsunAnalysis2dNetwork
from torch_tansacnet.lsunIntermediateRotation2dLayer import LsunIntermediateRotation2dLayer
from torch_tansacnet.lsunLayerExceptions import InvalidBoundaryOperation, InvalidDirection
from torch_tansacnet.orthonormalTransform import OrthonormalTransform

DT = torch.float64
CPU = torch.device("cpu")


def _layer(stride, nblocks, **kw):
    return LsunIntermediateRotation2dLayer(
        stride=stride, number_of_blocks=nblocks, dtype=DT, device=CPU, **kw
    )


def _randomize_layer(layer, seed=0):
    g = torch.Generator().manual_seed(seed)
    nblks, nangs = layer.angles.shape
    layer.angles = torch.randn(nblks, nangs, generator=g, dtype=DT)
    pa = layer.mus.shape[1]
    layer.mus = torch.where(torch.rand(nblks, pa, generator=g) > 0.5, 1.0, -1.0).to(DT)


def _boundary_slice(direction):
    # Index of the boundary blocks in nSamples x nRows x nCols x nChs
    return (slice(None), 0, slice(None)) if direction == "Vertical" else (slice(None), slice(None), 0)


# --- Layer --------------------------------------------------------------------


def test_default_boundary_operation():
    layer = _layer([2, 2], [4, 5])
    assert layer.boundary_operation == "Circular"


@pytest.mark.parametrize("direction", ["Vertical", "Horizontal"])
def test_constructor_with_boundary_operation(direction):
    layer = _layer([2, 2], [4, 5], boundary_operation="Termination", direction=direction)
    assert layer.boundary_operation == "Termination"
    assert layer.direction == direction


def test_invalid_boundary_operation():
    with pytest.raises(InvalidBoundaryOperation):
        _layer([2, 2], [4, 5], boundary_operation="InvalidBoundaryOperation")
    layer = _layer([2, 2], [4, 5])
    with pytest.raises(InvalidBoundaryOperation):
        layer.boundary_operation = "InvalidBoundaryOperation"


def test_invalid_direction():
    with pytest.raises(InvalidDirection):
        _layer([2, 2], [4, 5], boundary_operation="Termination", direction="InvalidDirection")
    layer = _layer([2, 2], [4, 5])
    with pytest.raises(InvalidDirection):
        layer.direction = "InvalidDirection"


def test_termination_without_direction():
    with pytest.raises(InvalidDirection):
        _layer([2, 2], [4, 5], boundary_operation="Termination")
    layer = _layer([2, 2], [4, 5])
    layer.boundary_operation = "Termination"
    with pytest.raises(InvalidDirection):
        layer(torch.zeros(1, 4, 5, 4, dtype=DT))


@pytest.mark.parametrize("stride", [[2, 2], [4, 4]])
@pytest.mark.parametrize("mode", ["Analysis", "Synthesis"])
@pytest.mark.parametrize("direction", ["Vertical", "Horizontal"])
def test_termination_boundary_blocks(stride, mode, direction):
    # Boundary blocks: -Ya, other blocks: the same as 'Circular'
    nblocks = [4, 5]
    circular = _layer(stride, nblocks, mode=mode)
    termination = _layer(stride, nblocks, mode=mode, boundary_operation="Termination", direction=direction)
    _randomize_layer(circular)
    termination.angles = circular.angles.detach().clone()
    termination.mus = circular.mus.clone()
    nDecs = math.prod(stride)
    ps = math.ceil(nDecs / 2)
    X = torch.randn(3, *nblocks, nDecs, dtype=DT)
    expected = circular(X).detach().clone()
    idx = _boundary_slice(direction) + (slice(ps, None),)
    expected[idx] = -X[idx]
    torch.testing.assert_close(termination(X), expected)


@pytest.mark.parametrize("direction", ["Vertical", "Horizontal"])
def test_circular_ignores_direction(direction):
    nblocks = [4, 5]
    reference = _layer([2, 2], nblocks, mode="Analysis")
    layer = _layer([2, 2], nblocks, mode="Analysis", boundary_operation="Circular", direction=direction)
    _randomize_layer(reference)
    layer.angles = reference.angles.detach().clone()
    layer.mus = reference.mus.clone()
    X = torch.randn(3, *nblocks, 4, dtype=DT)
    torch.testing.assert_close(layer(X), reference(X))


@pytest.mark.parametrize("direction", ["Vertical", "Horizontal"])
def test_set_after_construction(direction):
    nblocks = [4, 5]
    layer = _layer([2, 2], nblocks, mode="Analysis")
    _randomize_layer(layer)
    X = torch.randn(3, *nblocks, 4, dtype=DT)
    expected = layer(X).detach().clone()
    idx = _boundary_slice(direction) + (slice(2, None),)
    expected[idx] = -X[idx]
    layer.direction = direction
    layer.boundary_operation = "Termination"
    torch.testing.assert_close(layer(X), expected)


@pytest.mark.parametrize("mode", ["Analysis", "Synthesis"])
@pytest.mark.parametrize("direction", ["Vertical", "Horizontal"])
def test_termination_angle_gradients(mode, direction):
    # The boundary blocks do not contribute to the angle gradients, and the
    # other blocks have the same gradients as 'Circular'
    nblocks = [4, 5]
    circular = _layer([4, 4], nblocks, mode=mode)
    termination = _layer([4, 4], nblocks, mode=mode, boundary_operation="Termination", direction=direction)
    _randomize_layer(circular)
    termination.angles = circular.angles.detach().clone()
    termination.mus = circular.mus.clone()
    X = torch.randn(3, *nblocks, 16, dtype=DT)
    dLdZ = torch.randn(3, *nblocks, 16, dtype=DT)
    for layer in (circular, termination):
        (layer(X) * dLdZ).sum().backward()
    # The learnable angles are held by the OrthonormalTransform module
    grad = {id(layer): layer.orthTransUnx.orthonormalTransforms.angles.grad for layer in (circular, termination)}
    expected = grad[id(circular)].clone().reshape(*nblocks, -1)  # nRows x nCols x nAngles
    if direction == "Vertical":
        expected[0, :, :] = 0.0  # first row
    else:
        expected[:, 0, :] = 0.0  # first column
    torch.testing.assert_close(grad[id(termination)], expected.reshape(math.prod(nblocks), -1))


# --- Networks -----------------------------------------------------------------


def _network(stride=(4, 4), ovlp=(3, 5), size=(24, 32), nlevels=0, **kw):
    return LsunAnalysis2dNetwork(
        input_size=list(size),
        stride=list(stride),
        overlapping_factor=list(ovlp),
        number_of_levels=nlevels,
        dtype=DT,
        device=CPU,
        **kw,
    )


def _randomize_network(net, seed=0):
    torch.manual_seed(seed)

    def init(m):
        if isinstance(m, OrthonormalTransform):
            torch.nn.init.normal_(m.angles, mean=0.0, std=math.pi / 6)

    net.apply(init)


def _intermediate_layers(net):
    return {
        name: m
        for stage in net.layers
        for name, m in stage.named_children()
        if isinstance(m, LsunIntermediateRotation2dLayer)
    }


def test_network_default_boundary_operation():
    layers = _intermediate_layers(_network())
    assert layers
    assert all(m.boundary_operation == "Circular" for m in layers.values())


@pytest.mark.parametrize("boundary_operation", ["Circular", "Termination"])
@pytest.mark.parametrize("adjoint", [False, True])
def test_network_layer_settings(boundary_operation, adjoint):
    # Odd-order Vh/Vv get boundary_operation with 'Horizontal'/'Vertical';
    # even-order ones stay 'Circular'. The adjoint (.T) inherits them.
    ovlp = (3, 5)
    net = _network(ovlp=ovlp, boundary_operation=boundary_operation)
    if adjoint:
        net = net.T
    layers = _intermediate_layers(net)
    assert len(layers) == (ovlp[0] - 1) + (ovlp[1] - 1)
    for name, layer in layers.items():
        label = name.split("_V")[-1][0]
        order = int(name.split("_V")[-1][1:].lstrip("~"))
        if order % 2 == 1:
            assert layer.boundary_operation == boundary_operation, name
            assert layer.direction == {"h": "Horizontal", "v": "Vertical"}[label], name
        else:
            assert layer.boundary_operation == "Circular", name


def test_network_invalid_boundary_operation():
    with pytest.raises(InvalidBoundaryOperation):
        _network(boundary_operation="InvalidBoundaryOperation")


@pytest.mark.parametrize("nlevels", [0, 1, 2])
def test_network_perfect_reconstruction(nlevels):
    analyzer = _network(stride=(2, 2), size=(24, 32), nlevels=nlevels, boundary_operation="Termination")
    _randomize_network(analyzer)
    synthesizer = analyzer.T
    X = torch.randn(2, 1, 24, 32, dtype=DT)
    with torch.no_grad():
        torch.testing.assert_close(synthesizer(analyzer(X)), X, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("boundary_operation", ["Circular", "Termination"])
@pytest.mark.parametrize("nlevels", [0, 2])
def test_adjoint_copies_mus(boundary_operation, nlevels):
    # .T copies the mus as MATLAB fcn_cpparamsana2syn does, so that the
    # adjoint reconstructs the input even with random mus
    analyzer = _network(stride=(2, 2), size=(24, 32), nlevels=nlevels, boundary_operation=boundary_operation)
    _randomize_network(analyzer)
    g = torch.Generator().manual_seed(1)
    for layer in _intermediate_layers(analyzer).values():
        layer.mus = torch.where(torch.rand(layer.mus.shape, generator=g) > 0.5, 1.0, -1.0).to(DT)
    X = torch.randn(2, 1, 24, 32, dtype=DT)
    with torch.no_grad():
        torch.testing.assert_close(analyzer.T(analyzer(X)), X, rtol=1e-10, atol=1e-10)
        # and back from the synthesizer to an analyzer
        torch.testing.assert_close(analyzer.T.T(X), analyzer(X), rtol=1e-10, atol=1e-10)


def _edge_response(boundary_operation, axis):
    """Max |coefs| on the last block along axis (1: rows, 2: cols) for an
    input on the first block."""
    stride = (2, 2)
    analyzer = _network(stride=stride, ovlp=(5, 5), size=(16, 16), boundary_operation=boundary_operation)
    _randomize_network(analyzer)
    X = torch.zeros(1, 1, 16, 16, dtype=DT)
    X.narrow(axis + 1, 0, stride[axis - 1]).normal_()
    with torch.no_grad():
        coefs = analyzer(X)  # nSamples x nRows x nCols x nChs
    return coefs.narrow(axis, coefs.shape[axis] - 1, 1).abs().max().item()


@pytest.mark.parametrize("axis", [1, 2])
def test_network_no_wrap_around(axis):
    assert _edge_response("Termination", axis) == pytest.approx(0.0, abs=1e-10)
    assert _edge_response("Circular", axis) > 1e-4
