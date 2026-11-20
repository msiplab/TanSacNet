"""1-D LSUN layers and the analysis/synthesis networks.

This file checks the properties of the 1-D layers and of
LsunAnalysis1dNetwork and LsunSynthesis1dNetwork, as the MATLAB test class
fcn_createlsunlgraph1dTestCase does for fcn_createlsunlgraph1d. The values
are compared with MATLAB in test_lsunMatlabRefdata.py.

Data layout:
    sequences    : nSamples x 1 x Length
    coefficients : nSamples x nBlks x nChs

A stride of 2 gives (ps,pa) = (1,1), which has no angles and is always
the block DCT (Haar transform).

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

from torch_tansacnet import LsunAnalysis1dNetwork, LsunSynthesis1dNetwork
from torch_tansacnet.lsunAtomExtension1dLayer import LsunAtomExtension1dLayer
from torch_tansacnet.lsunBlockDct1dLayer import LsunBlockDct1dLayer
from torch_tansacnet.lsunBlockIdct1dLayer import LsunBlockIdct1dLayer
from torch_tansacnet.lsunChannelConcatenation1dLayer import LsunChannelConcatenation1dLayer
from torch_tansacnet.lsunChannelSeparation1dLayer import LsunChannelSeparation1dLayer
from torch_tansacnet.lsunFinalRotation1dLayer import LsunFinalRotation1dLayer
from torch_tansacnet.lsunInitialRotation1dLayer import LsunInitialRotation1dLayer
from torch_tansacnet.lsunIntermediateRotation1dLayer import LsunIntermediateRotation1dLayer
from torch_tansacnet.lsunLayerExceptions import (
    InvalidBoundaryOperation,
    InvalidDirection,
    InvalidInputSize,
    InvalidNumberOfLevels,
    InvalidOverlappingFactor,
    InvalidStride,
    InvalidTargetChannels,
)
from torch_tansacnet.lsunUtility import block_dct_matrix_1d, dctmtx
from torch_tansacnet.orthonormalTransform import OrthonormalTransform

DT = torch.float64
CPU = torch.device("cpu")
TOL = dict(rtol=1e-10, atol=1e-10)
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
STRIDES = [2, 4, 8]
NBLOCKS = 5


def _randomize(module, seed=0, std=math.pi / 6):
    torch.manual_seed(seed)

    def init(m):
        if isinstance(m, OrthonormalTransform):
            torch.nn.init.normal_(m.angles, mean=0.0, std=std)

    module.apply(init)
    return module


def _random_mus(shape, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.where(torch.rand(shape, generator=g) > 0.5, 1.0, -1.0).to(DT)


# --- Block DCT ----------------------------------------------------------------


@pytest.mark.parametrize("stride", STRIDES)
def test_block_dct_matrix_is_orthonormal(stride):
    C = block_dct_matrix_1d(stride, dtype=DT)
    assert C.shape == (stride, stride)
    torch.testing.assert_close(C @ C.T, torch.eye(stride, dtype=DT), **TOL)
    # The first row is the DC
    torch.testing.assert_close(C[0], torch.full((stride,), 1 / math.sqrt(stride), dtype=DT), **TOL)


@pytest.mark.parametrize("stride", STRIDES)
def test_block_dct_matrix_symmetry(stride):
    # The first half of the rows is symmetric and the second half is
    # antisymmetric with respect to the flip of the block
    C = block_dct_matrix_1d(stride, dtype=DT)
    ps = stride // 2
    torch.testing.assert_close(C[:ps].flip(dims=(1,)), C[:ps], **TOL)
    torch.testing.assert_close(C[ps:].flip(dims=(1,)), -C[ps:], **TOL)


@pytest.mark.parametrize("stride", STRIDES)
def test_block_dct_is_dct_of_each_block(stride):
    # Even frequencies first, then odd ones
    nSamples = 2
    X = torch.randn(nSamples, 1, NBLOCKS * stride, dtype=DT)
    Z = LsunBlockDct1dLayer(stride=stride)(X)
    assert Z.shape == (nSamples, NBLOCKS, stride)
    coefs = X.reshape(nSamples, NBLOCKS, stride) @ dctmtx(stride, dtype=DT).T
    expected = torch.cat((coefs[..., 0::2], coefs[..., 1::2]), dim=-1)
    torch.testing.assert_close(Z, expected, **TOL)


@pytest.mark.parametrize("stride", STRIDES)
def test_block_idct_inverts_block_dct(stride):
    X = torch.randn(2, 1, NBLOCKS * stride, dtype=DT)
    Z = LsunBlockDct1dLayer(stride=stride)(X)
    torch.testing.assert_close(LsunBlockIdct1dLayer(stride=stride)(Z), X, **TOL)


def test_block_dct_multiple_components():
    stride = 4
    X = torch.randn(2, 3, 20, dtype=DT)
    Z = LsunBlockDct1dLayer(stride=stride, number_of_components=3)(X)
    assert len(Z) == 3
    single = LsunBlockDct1dLayer(stride=stride)
    for iComponent in range(3):
        torch.testing.assert_close(Z[iComponent], single(X[:, [iComponent]]), **TOL)
    Y = LsunBlockIdct1dLayer(stride=stride, number_of_components=3)(*Z)
    torch.testing.assert_close(Y, X, **TOL)


def test_block_dct_invalid_input_size():
    with pytest.raises(InvalidInputSize):
        LsunBlockDct1dLayer(stride=4)(torch.zeros(1, 1, 10, dtype=DT))


# --- Channel separation and concatenation ---------------------------------------


def test_channel_separation_and_concatenation():
    X = torch.randn(2, NBLOCKS, 8, dtype=DT)
    ac, dc = LsunChannelSeparation1dLayer()(X)
    assert ac.shape == (2, NBLOCKS, 7)
    assert dc.shape == (2, NBLOCKS)
    torch.testing.assert_close(dc, X[..., 0])
    torch.testing.assert_close(ac, X[..., 1:])
    concat = LsunChannelConcatenation1dLayer()
    torch.testing.assert_close(concat(ac, dc), X)
    torch.testing.assert_close(concat(ac, dc.unsqueeze(-1)), X)


# --- Atom extension -----------------------------------------------------------


@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("target", ["Sum", "Difference"])
@pytest.mark.parametrize("direction, shift", [("Right", 1), ("Left", -1)])
def test_atom_extension(stride, target, direction, shift):
    ps = stride // 2
    X = torch.randn(2, NBLOCKS, stride, dtype=DT)
    Ys = X[..., :ps] + X[..., ps:]
    Ya = X[..., :ps] - X[..., ps:]
    if target == "Sum":
        Ys = torch.roll(Ys, shifts=shift, dims=1)
    else:
        Ya = torch.roll(Ya, shifts=shift, dims=1)
    expected = torch.cat((Ys + Ya, Ys - Ya), dim=-1) / 2
    layer = LsunAtomExtension1dLayer(stride=stride, direction=direction, target_channels=target)
    torch.testing.assert_close(layer(X), expected, **TOL)


@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("target", ["Sum", "Difference"])
def test_atom_extension_is_orthonormal(stride, target):
    # The layer of the reverse direction is the inverse and the adjoint
    forward = LsunAtomExtension1dLayer(stride=stride, direction="Right", target_channels=target)
    backward = LsunAtomExtension1dLayer(stride=stride, direction="Left", target_channels=target)
    X = torch.randn(2, NBLOCKS, stride, dtype=DT, requires_grad=True)
    Z = forward(X)
    torch.testing.assert_close(backward(Z), X, **TOL)
    dLdZ = torch.randn_like(Z)
    (Z * dLdZ).sum().backward()
    torch.testing.assert_close(X.grad, backward(dLdZ), **TOL)


def test_atom_extension_invalid_arguments():
    with pytest.raises(InvalidDirection):
        LsunAtomExtension1dLayer(stride=4, direction="Down", target_channels="Sum")
    with pytest.raises(InvalidTargetChannels):
        LsunAtomExtension1dLayer(stride=4, direction="Right", target_channels="Dummy")


# --- Initial and final rotations ------------------------------------------------


@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("cls", [LsunInitialRotation1dLayer, LsunFinalRotation1dLayer])
def test_rotation_is_identity_by_default(stride, cls):
    layer = cls(stride=stride, number_of_blocks=NBLOCKS, dtype=DT, device=CPU)
    ps = stride // 2
    assert layer.angles.shape == (NBLOCKS, ps * (ps - 1))
    assert layer.mus.shape == (NBLOCKS, stride)
    X = torch.randn(2, NBLOCKS, stride, dtype=DT)
    torch.testing.assert_close(layer(X), X, **TOL)


@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("no_dc_leakage", [False, True])
def test_final_rotation_is_adjoint_of_initial_rotation(stride, no_dc_leakage):
    opts = dict(stride=stride, number_of_blocks=NBLOCKS, no_dc_leakage=no_dc_leakage, dtype=DT, device=CPU)
    initial = _randomize(LsunInitialRotation1dLayer(**opts))
    initial.mus = _random_mus(initial.mus.shape)
    final = LsunFinalRotation1dLayer(**opts)
    final.angles = initial.angles.detach().clone()
    final.mus = initial.mus.clone()
    X = torch.randn(2, NBLOCKS, stride, dtype=DT)
    Z = initial(X)
    # Blockwise orthonormal
    torch.testing.assert_close(Z.pow(2).sum(dim=-1), X.pow(2).sum(dim=-1), **TOL)
    torch.testing.assert_close(final(Z), X, **TOL)


@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("cls", [LsunInitialRotation1dLayer, LsunFinalRotation1dLayer])
def test_rotation_no_dc_leakage(stride, cls):
    # The DC channel passes through the rotation
    layer = _randomize(cls(stride=stride, number_of_blocks=NBLOCKS, no_dc_leakage=True, dtype=DT, device=CPU))
    layer.mus = -1
    X = torch.zeros(2, NBLOCKS, stride, dtype=DT)
    X[..., 0] = torch.randn(2, NBLOCKS, dtype=DT)
    torch.testing.assert_close(layer(X), X, **TOL)


def test_initial_rotation_is_blockwise():
    # Only the block iBlk is rotated by its own angles
    stride, iBlk = 8, 3
    layer = LsunInitialRotation1dLayer(stride=stride, number_of_blocks=NBLOCKS, dtype=DT, device=CPU)
    angles = torch.zeros_like(layer.angles)
    angles[iBlk, :] = 1.0
    layer.angles = angles
    X = torch.randn(2, NBLOCKS, stride, dtype=DT)
    isChanged = (layer(X) - X).abs().amax(dim=(0, 2)) > 1e-6
    expected = torch.zeros(NBLOCKS, dtype=torch.bool)
    expected[iBlk] = True
    assert torch.equal(isChanged, expected)


# --- Intermediate rotation ------------------------------------------------------


def _intermediate(stride, **kw):
    return LsunIntermediateRotation1dLayer(stride=stride, number_of_blocks=NBLOCKS, dtype=DT, device=CPU, **kw)


def _randomize_intermediate(layer):
    _randomize(layer, std=1.0)
    layer.mus = _random_mus(layer.mus.shape)
    return layer


@pytest.mark.parametrize("stride", STRIDES)
def test_intermediate_rotation_default(stride):
    layer = _intermediate(stride, mus=-1)
    assert layer.boundary_operation == "Circular"
    assert layer.mode == "Synthesis"
    ps = stride // 2
    assert layer.angles.shape == (NBLOCKS, ps * (ps - 1) // 2)
    X = torch.randn(2, NBLOCKS, stride, dtype=DT)
    expected = torch.cat((X[..., :ps], -X[..., ps:]), dim=-1)
    torch.testing.assert_close(layer(X), expected, **TOL)


@pytest.mark.parametrize("stride", STRIDES)
def test_intermediate_rotation_synthesis_is_adjoint_of_analysis(stride):
    analysis = _randomize_intermediate(_intermediate(stride, mode="Analysis"))
    synthesis = _intermediate(stride, mode="Synthesis")
    synthesis.angles = analysis.angles.detach().clone()
    synthesis.mus = analysis.mus.clone()
    X = torch.randn(2, NBLOCKS, stride, dtype=DT)
    torch.testing.assert_close(synthesis(analysis(X)), X, **TOL)


def test_intermediate_rotation_invalid_arguments():
    with pytest.raises(InvalidBoundaryOperation):
        _intermediate(4, boundary_operation="Dummy")
    layer = _intermediate(4)
    with pytest.raises(InvalidBoundaryOperation):
        layer.boundary_operation = "Dummy"


@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("mode", ["Analysis", "Synthesis"])
def test_intermediate_rotation_termination(stride, mode):
    # First block: -Ya, other blocks: the same as 'Circular'
    circular = _randomize_intermediate(_intermediate(stride, mode=mode))
    termination = _intermediate(stride, mode=mode, boundary_operation="Termination")
    termination.angles = circular.angles.detach().clone()
    termination.mus = circular.mus.clone()
    ps = stride // 2
    X = torch.randn(3, NBLOCKS, stride, dtype=DT)
    expected = circular(X).detach().clone()
    expected[:, 0, ps:] = -X[:, 0, ps:]
    torch.testing.assert_close(termination(X), expected, **TOL)


@pytest.mark.parametrize("stride", [4, 8])
@pytest.mark.parametrize("mode", ["Analysis", "Synthesis"])
def test_intermediate_rotation_termination_angle_gradients(stride, mode):
    # The first block does not contribute to the angle gradients, and the
    # other blocks have the same gradients as 'Circular'
    circular = _randomize_intermediate(_intermediate(stride, mode=mode))
    termination = _intermediate(stride, mode=mode, boundary_operation="Termination")
    termination.angles = circular.angles.detach().clone()
    termination.mus = circular.mus.clone()
    X = torch.randn(3, NBLOCKS, stride, dtype=DT)
    dLdZ = torch.randn(3, NBLOCKS, stride, dtype=DT)
    grads = []
    for layer in (circular, termination):
        (layer(X) * dLdZ).sum().backward()
        grads.append(layer.orthTransUnx.orthonormalTransforms.angles.grad)
    expected = grads[0].clone()
    assert expected.abs().amin(dim=-1).min() > 0
    expected[0] = 0.0
    torch.testing.assert_close(grads[1], expected, **TOL)


# --- Networks -----------------------------------------------------------------


def _network(stride=4, ovlp=3, size=32, nlevels=0, device="cpu", **kw):
    return LsunAnalysis1dNetwork(
        input_size=size,
        stride=stride,
        overlapping_factor=ovlp,
        number_of_levels=nlevels,
        dtype=DT,
        device=torch.device(device),
        **kw,
    )


def _intermediate_layers(net):
    return {
        name: m
        for stage in net.layers
        for name, m in stage.named_children()
        if isinstance(m, LsunIntermediateRotation1dLayer)
    }


def _as_tuple(coefs):
    return coefs if isinstance(coefs, tuple) else (coefs,)


def test_network_layer_order():
    # The synthesizer is the reverse of the analyzer
    analyzer = _network(ovlp=5, nlevels=1)
    assert [name for name, _ in analyzer.layers[0].named_children()] == [
        "Lv1_E0", "Lv1_V0",
        "Lv1_Qx1rd", "Lv1_Vx1", "Lv1_Qx2ls", "Lv1_Vx2",
        "Lv1_Qx3rd", "Lv1_Vx3", "Lv1_Qx4ls", "Lv1_Vx4",
        "Lv1_Sp",
    ]  # fmt: skip
    assert [name for name, _ in analyzer.T.layers[0].named_children()] == [
        "Lv1_Cc",
        "Lv1_Vx~4", "Lv1_Qx~4ls", "Lv1_Vx~3", "Lv1_Qx~3rd",
        "Lv1_Vx~2", "Lv1_Qx~2ls", "Lv1_Vx~1", "Lv1_Qx~1rd",
        "Lv1_V0~", "Lv1_E0~",
    ]  # fmt: skip


@pytest.mark.parametrize("boundary_operation", ["Circular", "Termination"])
@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize("ovlp", [1, 3, 5])
def test_network_layer_settings(boundary_operation, adjoint, ovlp):
    # Odd-order Vx get boundary_operation; even-order ones stay 'Circular'.
    # The adjoint inherits them.
    net = _network(ovlp=ovlp, boundary_operation=boundary_operation)
    if adjoint:
        net = net.T
        assert isinstance(net, LsunSynthesis1dNetwork)
        assert net.boundary_operation == boundary_operation
    layers = _intermediate_layers(net)
    assert len(layers) == ovlp - 1
    for name, layer in layers.items():
        order = int(name.split("_Vx")[-1].lstrip("~"))
        assert layer.mode == ("Synthesis" if adjoint else "Analysis"), name
        if order % 2 == 1:
            assert layer.boundary_operation == boundary_operation, name
        else:
            assert layer.boundary_operation == "Circular", name


def test_network_number_of_blocks_per_level():
    size, stride, nlevels = 128, 4, 3
    for net in (_network(size=size, nlevels=nlevels), _network(size=size, nlevels=nlevels).T):
        layers = net.rotation_layers()
        assert len(layers) == nlevels * 3
        for name, layer in layers.items():
            ilevel = int(name[2 : name.index("_")])
            assert layer.number_of_blocks == size // stride**ilevel, name


def test_network_invalid_arguments():
    with pytest.raises(InvalidBoundaryOperation):
        _network(boundary_operation="Dummy")
    with pytest.raises(InvalidStride):
        _network(stride=3, size=27)
    with pytest.raises(InvalidStride):
        _network(stride=[2, 2])
    with pytest.raises(InvalidOverlappingFactor):
        _network(ovlp=2)
    with pytest.raises(InvalidInputSize):
        _network(size=30)
    with pytest.raises(InvalidInputSize):
        _network(size=32, nlevels=3)
    with pytest.raises(InvalidNumberOfLevels):
        _network(nlevels=-1)
    with pytest.raises(ValueError):
        _network(number_of_components=2)


@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("ovlp", [1, 3, 5])
@pytest.mark.parametrize("boundary_operation", ["Circular", "Termination"])
def test_network_is_block_dct_by_default(stride, ovlp, boundary_operation):
    size = 4 * stride
    analyzer = _network(stride=stride, ovlp=ovlp, size=size, boundary_operation=boundary_operation)
    X = torch.randn(2, 1, size, dtype=DT)
    with torch.no_grad():
        torch.testing.assert_close(analyzer(X), LsunBlockDct1dLayer(stride=stride)(X), **TOL)


def test_network_stride_2_is_always_block_dct():
    # (ps,pa) = (1,1) has no angles; the signs of Vx cancel out
    size = 16
    analyzer = _randomize(_network(stride=2, ovlp=5, size=size))
    assert all(p.numel() == 0 for p in analyzer.parameters())
    X = torch.randn(2, 1, size, dtype=DT)
    with torch.no_grad():
        torch.testing.assert_close(analyzer(X), LsunBlockDct1dLayer(stride=2)(X), **TOL)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("nlevels", [0, 1, 3])
@pytest.mark.parametrize("boundary_operation", ["Circular", "Termination"])
@pytest.mark.parametrize("stride, ovlp", [(2, 3), (4, 1), (4, 3), (4, 5), (8, 3), (8, 5)])
def test_network_perfect_reconstruction(device, nlevels, boundary_operation, stride, ovlp):
    size = 4 * stride ** max(nlevels, 1)
    analyzer = _randomize(
        _network(
            stride=stride, ovlp=ovlp, size=size, nlevels=nlevels, device=device, boundary_operation=boundary_operation
        )
    )
    for layer in _intermediate_layers(analyzer).values():
        layer.mus = _random_mus(layer.mus.shape)
    synthesizer = analyzer.T
    nSamples = 2
    X = torch.randn(nSamples, 1, size, dtype=DT, device=device)
    with torch.no_grad():
        coefs = _as_tuple(analyzer(X))
        Y = synthesizer(coefs if nlevels > 0 else coefs[0])
    # Sizes
    if nlevels == 0:
        assert coefs[0].shape == (nSamples, size // stride, stride)
    else:
        assert len(coefs) == nlevels + 1
        assert coefs[0].shape == (nSamples, size // stride**nlevels, 1)
        for iLevel in range(nlevels, 0, -1):
            assert coefs[nlevels - iLevel + 1].shape == (nSamples, size // stride**iLevel, stride - 1)
    assert all(c.device.type == device and c.dtype == DT for c in coefs)
    # Energy preservation and perfect reconstruction
    energy = sum(c.pow(2).sum() for c in coefs)
    torch.testing.assert_close(energy, X.pow(2).sum(), **TOL)
    torch.testing.assert_close(Y, X, **TOL)
    # The adjoint of the synthesizer is the analyzer
    with torch.no_grad():
        for actual, expected in zip(_as_tuple(synthesizer.T(X)), coefs):
            torch.testing.assert_close(actual, expected, **TOL)


@pytest.mark.parametrize("nlevels", [0, 2])
def test_synthesizer_is_adjoint_of_analyzer(nlevels):
    # <Ax,s> = <x,A^T s> for random coefficients s
    size = 64
    analyzer = _randomize(_network(ovlp=5, size=size, nlevels=nlevels, boundary_operation="Termination"))
    synthesizer = analyzer.T
    X = torch.randn(2, 1, size, dtype=DT)
    with torch.no_grad():
        coefs = _as_tuple(analyzer(X))
        S = tuple(torch.randn_like(c) for c in coefs)
        Y = synthesizer(S if nlevels > 0 else S[0])
    lhs = sum((c * s).sum() for c, s in zip(coefs, S))
    torch.testing.assert_close(lhs, (X * Y).sum(), **TOL)


def test_synthesizer_constructed_directly():
    # A synthesizer constructed directly is the inverse of the analyzer
    # with the default parameters, and its adjoint is that analyzer
    opts = dict(input_size=32, stride=4, overlapping_factor=3, number_of_levels=2, dtype=DT, device=CPU)
    synthesizer = LsunSynthesis1dNetwork(**opts)
    analyzer = LsunAnalysis1dNetwork(**opts)
    X = torch.randn(2, 1, 32, dtype=DT)
    with torch.no_grad():
        torch.testing.assert_close(synthesizer(analyzer(X)), X, **TOL)
        for actual, expected in zip(synthesizer.T(X), analyzer(X)):
            torch.testing.assert_close(actual, expected, **TOL)


def _edge_response(boundary_operation, stride, ovlp, nlevels, reverse=False):
    """Max |coefs| on the last block for an input on the first block of the
    coarsest level (or on the first block for an input on the last one)."""
    size = 8 * stride ** max(nlevels, 1)
    analyzer = _randomize(
        _network(stride=stride, ovlp=ovlp, size=size, nlevels=nlevels, boundary_operation=boundary_operation)
    )
    width = stride ** max(nlevels, 1)
    X = torch.zeros(1, 1, size, dtype=DT)
    X.narrow(2, size - width if reverse else 0, width).normal_()
    with torch.no_grad():
        coefs = _as_tuple(analyzer(X))
    return max(c[:, 0 if reverse else -1].abs().max().item() for c in coefs)


@pytest.mark.parametrize("stride", [4, 8])
@pytest.mark.parametrize("ovlp", [3, 5])
@pytest.mark.parametrize("nlevels", [0, 2])
@pytest.mark.parametrize("reverse", [False, True])
def test_network_no_wrap_around(stride, ovlp, nlevels, reverse):
    assert _edge_response("Termination", stride, ovlp, nlevels, reverse) == pytest.approx(0.0, abs=1e-10)
    assert _edge_response("Circular", stride, ovlp, nlevels, reverse) > 1e-4


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("boundary_operation", ["Circular", "Termination"])
def test_network_gradients(device, boundary_operation):
    # The angles of all the levels are trainable through the analyzer and
    # the synthesizer
    size = 64
    analyzer = _randomize(_network(stride=4, size=size, nlevels=2, device=device, boundary_operation=boundary_operation))
    synthesizer = analyzer.T
    X = torch.randn(2, 1, size, dtype=DT, device=device)
    coefs = analyzer(X)
    # Reconstruction from the DC and the AC of level 2 with the attenuated
    # AC of level 1 (zeros would give no gradient to the first rotation of
    # the level-1 stage of the synthesizer)
    attenuated = coefs[:2] + tuple(0.5 * c for c in coefs[2:])
    loss = (synthesizer(attenuated) - X).pow(2).sum() + sum(c.abs().sum() for c in coefs)
    loss.backward()
    for net in (analyzer, synthesizer):
        params = dict(net.named_parameters())
        assert len(params) == 2 * (2 + 2)  # per level: W0, U0 and two Un
        for name, param in params.items():
            assert param.grad is not None, name
            assert torch.isfinite(param.grad).all(), name
            assert param.grad.abs().max() > 0, name


@pytest.mark.parametrize("stride", STRIDES)
def test_network_no_dc_leakage(stride):
    # A constant input gives only the DC coefficients
    size = 4 * stride**2
    analyzer = _randomize(_network(stride=stride, size=size, nlevels=2, no_dc_leakage=True))
    X = torch.full((1, 1, size), 3.0, dtype=DT)
    with torch.no_grad():
        coefs = analyzer(X)
    torch.testing.assert_close(coefs[0], torch.full_like(coefs[0], 3.0 * math.sqrt(stride) ** 2), **TOL)
    for ac in coefs[1:]:
        torch.testing.assert_close(ac, torch.zeros_like(ac), **TOL)


def test_network_prefix():
    analyzer = _randomize(_network(nlevels=1, prefix="P_"))
    assert all(name.startswith("P_Lv1_") for name, _ in analyzer.layers[0].named_children())
    assert sorted(analyzer.rotation_layers()) == sorted(_network(nlevels=1).rotation_layers())
    X = torch.randn(1, 1, 32, dtype=DT)
    with torch.no_grad():
        torch.testing.assert_close(analyzer.T(analyzer(X)), X, **TOL)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_network_to_device():
    analyzer = _randomize(_network(nlevels=1))
    X = torch.randn(1, 1, 32, dtype=DT)
    with torch.no_grad():
        expected = analyzer(X)
        analyzer = analyzer.to(device=torch.device("cuda"))
        actual = analyzer(X.cuda())
        Y = analyzer.T(actual)
    assert Y.device.type == "cuda"
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a.cpu(), e, **TOL)
    torch.testing.assert_close(Y.cpu(), X, **TOL)
