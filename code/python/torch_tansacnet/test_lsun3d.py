"""3-D LSUN layers and the analysis/synthesis networks.

This file checks the properties of the 3-D layers and of
LsunAnalysis3dNetwork and LsunSynthesis3dNetwork, as the MATLAB test class
fcn_createlsunlgraph3dTestCase does for fcn_createlsunlgraph3d. The values
are compared with MATLAB in test_lsunMatlabRefdata.py.

Data layout:
    volumes      : nSamples x 1 x Height x Width x Depth
    coefficients : nSamples x nRows x nCols x nLays x nChs
Blocks are in row-major order of nRows x nCols x nLays.

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

from torch_tansacnet import LsunAnalysis3dNetwork, LsunSynthesis3dNetwork
from torch_tansacnet.lsunAtomExtension3dLayer import LsunAtomExtension3dLayer
from torch_tansacnet.lsunBlockDct3dLayer import LsunBlockDct3dLayer
from torch_tansacnet.lsunBlockIdct3dLayer import LsunBlockIdct3dLayer
from torch_tansacnet.lsunChannelConcatenation3dLayer import LsunChannelConcatenation3dLayer
from torch_tansacnet.lsunChannelSeparation3dLayer import LsunChannelSeparation3dLayer
from torch_tansacnet.lsunFinalRotation3dLayer import LsunFinalRotation3dLayer
from torch_tansacnet.lsunInitialRotation3dLayer import LsunInitialRotation3dLayer
from torch_tansacnet.lsunIntermediateRotation3dLayer import LsunIntermediateRotation3dLayer
from torch_tansacnet.lsunLayerExceptions import (
    InvalidBoundaryOperation,
    InvalidDirection,
    InvalidInputSize,
    InvalidNumberOfLevels,
    InvalidOverlappingFactor,
    InvalidStride,
    InvalidTargetChannels,
)
from torch_tansacnet.lsunUtility import block_dct_matrix_3d, dctmtx
from torch_tansacnet.orthonormalTransform import OrthonormalTransform

DT = torch.float64
CPU = torch.device("cpu")
TOL = dict(rtol=1e-10, atol=1e-10)
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
STRIDES = [[2, 2, 2], [1, 2, 2], [2, 4, 2]]
DIRECTIONS = ["Vertical", "Horizontal", "Depth"]
NBLOCKS = [2, 3, 4]  # non-cubic so that the directions cannot be confused


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
    C = block_dct_matrix_3d(stride, dtype=DT)
    nDecs = math.prod(stride)
    assert C.shape == (nDecs, nDecs)
    torch.testing.assert_close(C @ C.T, torch.eye(nDecs, dtype=DT), **TOL)
    # The first row is the DC
    torch.testing.assert_close(C[0], torch.full((nDecs,), 1 / math.sqrt(nDecs), dtype=DT), **TOL)


@pytest.mark.parametrize("stride", STRIDES)
def test_block_dct_matrix_symmetry(stride):
    # The first half of the rows is symmetric and the second half is
    # antisymmetric with respect to the flip of the block
    decV, decH, decD = stride
    nDecs = math.prod(stride)
    C = block_dct_matrix_3d(stride, dtype=DT).reshape(nDecs, decD, decH, decV)
    flipped = C.flip(dims=(1, 2, 3))
    ps = math.ceil(nDecs / 2)
    torch.testing.assert_close(flipped[:ps], C[:ps], **TOL)
    torch.testing.assert_close(flipped[ps:], -C[ps:], **TOL)


@pytest.mark.parametrize("stride", STRIDES)
def test_block_dct_is_separable_dct(stride):
    # The coefficients are those of the separable DCT of each block
    decV, decH, decD = stride
    nSamples = 2
    X = torch.randn(nSamples, 1, *[n * d for n, d in zip(NBLOCKS, stride)], dtype=DT)
    Z = LsunBlockDct3dLayer(stride=stride)(X)
    assert Z.shape == (nSamples, *NBLOCKS, math.prod(stride))
    blocks = X.reshape(nSamples, NBLOCKS[0], decV, NBLOCKS[1], decH, NBLOCKS[2], decD)
    expected = torch.einsum(
        "yv,xh,zd,nrvchld->nrclyxz", dctmtx(decV, dtype=DT), dctmtx(decH, dtype=DT), dctmtx(decD, dtype=DT), blocks
    )
    torch.testing.assert_close(Z.sort(dim=-1).values, expected.flatten(4).sort(dim=-1).values, **TOL)
    # DC
    torch.testing.assert_close(Z[..., 0], expected[..., 0, 0, 0], **TOL)


@pytest.mark.parametrize("stride", STRIDES)
def test_block_idct_inverts_block_dct(stride):
    X = torch.randn(2, 1, *[n * d for n, d in zip(NBLOCKS, stride)], dtype=DT)
    Z = LsunBlockDct3dLayer(stride=stride)(X)
    torch.testing.assert_close(LsunBlockIdct3dLayer(stride=stride)(Z), X, **TOL)


def test_block_dct_multiple_components():
    stride = [2, 2, 2]
    X = torch.randn(2, 3, 4, 6, 8, dtype=DT)
    Z = LsunBlockDct3dLayer(stride=stride, number_of_components=3)(X)
    assert len(Z) == 3
    single = LsunBlockDct3dLayer(stride=stride)
    for iComponent in range(3):
        torch.testing.assert_close(Z[iComponent], single(X[:, [iComponent]]), **TOL)
    Y = LsunBlockIdct3dLayer(stride=stride, number_of_components=3)(*Z)
    torch.testing.assert_close(Y, X, **TOL)


def test_block_dct_invalid_input_size():
    with pytest.raises(InvalidInputSize):
        LsunBlockDct3dLayer(stride=[2, 2, 2])(torch.zeros(1, 1, 4, 5, 6, dtype=DT))


# --- Channel separation and concatenation ---------------------------------------


def test_channel_separation_and_concatenation():
    X = torch.randn(2, *NBLOCKS, 8, dtype=DT)
    ac, dc = LsunChannelSeparation3dLayer()(X)
    assert ac.shape == (2, *NBLOCKS, 7)
    assert dc.shape == (2, *NBLOCKS)
    torch.testing.assert_close(dc, X[..., 0])
    torch.testing.assert_close(ac, X[..., 1:])
    concat = LsunChannelConcatenation3dLayer()
    torch.testing.assert_close(concat(ac, dc), X)
    torch.testing.assert_close(concat(ac, dc.unsqueeze(-1)), X)


# --- Atom extension -----------------------------------------------------------


@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("target", ["Sum", "Difference"])
@pytest.mark.parametrize(
    "direction, dim, shift",
    [("Down", 1, 1), ("Up", 1, -1), ("Right", 2, 1), ("Left", 2, -1), ("Back", 3, 1), ("Front", 3, -1)],
)
def test_atom_extension(stride, target, direction, dim, shift):
    nDecs = math.prod(stride)
    ps = nDecs // 2
    X = torch.randn(2, *NBLOCKS, nDecs, dtype=DT)
    Ys = X[..., :ps] + X[..., ps:]
    Ya = X[..., :ps] - X[..., ps:]
    if target == "Sum":
        Ys = torch.roll(Ys, shifts=shift, dims=dim)
    else:
        Ya = torch.roll(Ya, shifts=shift, dims=dim)
    expected = torch.cat((Ys + Ya, Ys - Ya), dim=-1) / 2
    layer = LsunAtomExtension3dLayer(stride=stride, direction=direction, target_channels=target)
    torch.testing.assert_close(layer(X), expected, **TOL)


@pytest.mark.parametrize("target", ["Sum", "Difference"])
@pytest.mark.parametrize("direction, reverse", [("Down", "Up"), ("Right", "Left"), ("Back", "Front")])
def test_atom_extension_is_orthonormal(target, direction, reverse):
    # The layer of the reverse direction is the inverse and the adjoint
    stride = [2, 2, 2]
    forward = LsunAtomExtension3dLayer(stride=stride, direction=direction, target_channels=target)
    backward = LsunAtomExtension3dLayer(stride=stride, direction=reverse, target_channels=target)
    X = torch.randn(2, *NBLOCKS, 8, dtype=DT, requires_grad=True)
    Z = forward(X)
    torch.testing.assert_close(backward(Z), X, **TOL)
    dLdZ = torch.randn_like(Z)
    (Z * dLdZ).sum().backward()
    torch.testing.assert_close(X.grad, backward(dLdZ), **TOL)


def test_atom_extension_invalid_arguments():
    with pytest.raises(InvalidDirection):
        LsunAtomExtension3dLayer(stride=[2, 2, 2], direction="Dummy", target_channels="Sum")
    with pytest.raises(InvalidTargetChannels):
        LsunAtomExtension3dLayer(stride=[2, 2, 2], direction="Back", target_channels="Dummy")


# --- Initial and final rotations ------------------------------------------------


@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("cls", [LsunInitialRotation3dLayer, LsunFinalRotation3dLayer])
def test_rotation_is_identity_by_default(stride, cls):
    layer = cls(stride=stride, number_of_blocks=NBLOCKS, dtype=DT, device=CPU)
    nDecs = math.prod(stride)
    ps = nDecs // 2
    assert layer.angles.shape == (math.prod(NBLOCKS), ps * (ps - 1))
    assert layer.mus.shape == (math.prod(NBLOCKS), nDecs)
    X = torch.randn(2, *NBLOCKS, nDecs, dtype=DT)
    torch.testing.assert_close(layer(X), X, **TOL)


@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("no_dc_leakage", [False, True])
def test_final_rotation_is_adjoint_of_initial_rotation(stride, no_dc_leakage):
    opts = dict(stride=stride, number_of_blocks=NBLOCKS, no_dc_leakage=no_dc_leakage, dtype=DT, device=CPU)
    initial = _randomize(LsunInitialRotation3dLayer(**opts))
    initial.mus = _random_mus(initial.mus.shape)
    final = LsunFinalRotation3dLayer(**opts)
    final.angles = initial.angles.detach().clone()
    final.mus = initial.mus.clone()
    X = torch.randn(2, *NBLOCKS, math.prod(stride), dtype=DT)
    Z = initial(X)
    # Blockwise orthonormal
    torch.testing.assert_close(Z.pow(2).sum(dim=-1), X.pow(2).sum(dim=-1), **TOL)
    torch.testing.assert_close(final(Z), X, **TOL)


@pytest.mark.parametrize("cls", [LsunInitialRotation3dLayer, LsunFinalRotation3dLayer])
def test_rotation_no_dc_leakage(cls):
    # The DC channel passes through the rotation
    stride = [2, 2, 2]
    layer = _randomize(cls(stride=stride, number_of_blocks=NBLOCKS, no_dc_leakage=True, dtype=DT, device=CPU))
    layer.mus = -1
    X = torch.zeros(2, *NBLOCKS, 8, dtype=DT)
    X[..., 0] = torch.randn(2, *NBLOCKS, dtype=DT)
    torch.testing.assert_close(layer(X), X, **TOL)


def test_initial_rotation_blocks_are_in_row_major_order():
    # Only the block (iRow,iCol,iLay) is rotated by its own angles
    stride = [2, 2, 2]
    iRow, iCol, iLay = 1, 2, 3
    layer = LsunInitialRotation3dLayer(stride=stride, number_of_blocks=NBLOCKS, dtype=DT, device=CPU)
    angles = torch.zeros_like(layer.angles)
    angles[(iRow * NBLOCKS[1] + iCol) * NBLOCKS[2] + iLay, :] = 1.0
    layer.angles = angles
    X = torch.randn(2, *NBLOCKS, 8, dtype=DT)
    isChanged = (layer(X) - X).abs().amax(dim=(0, 4)) > 1e-6
    expected = torch.zeros(*NBLOCKS, dtype=torch.bool)
    expected[iRow, iCol, iLay] = True
    assert torch.equal(isChanged, expected)


# --- Intermediate rotation ------------------------------------------------------


def _intermediate(stride, **kw):
    return LsunIntermediateRotation3dLayer(stride=stride, number_of_blocks=NBLOCKS, dtype=DT, device=CPU, **kw)


def _randomize_intermediate(layer):
    _randomize(layer, std=1.0)
    layer.mus = _random_mus(layer.mus.shape)
    return layer


def _boundary_slice(direction):
    idx = [slice(None)] * 4  # nSamples x nRows x nCols x nLays
    idx[1 + DIRECTIONS.index(direction)] = 0
    return tuple(idx)


@pytest.mark.parametrize("stride", STRIDES)
def test_intermediate_rotation_default(stride):
    layer = _intermediate(stride, mus=-1)
    assert layer.boundary_operation == "Circular"
    assert layer.mode == "Synthesis"
    nDecs = math.prod(stride)
    ps = nDecs // 2
    X = torch.randn(2, *NBLOCKS, nDecs, dtype=DT)
    expected = torch.cat((X[..., :ps], -X[..., ps:]), dim=-1)
    torch.testing.assert_close(layer(X), expected, **TOL)


@pytest.mark.parametrize("stride", STRIDES)
def test_intermediate_rotation_synthesis_is_adjoint_of_analysis(stride):
    analysis = _randomize_intermediate(_intermediate(stride, mode="Analysis"))
    synthesis = _intermediate(stride, mode="Synthesis")
    synthesis.angles = analysis.angles.detach().clone()
    synthesis.mus = analysis.mus.clone()
    X = torch.randn(2, *NBLOCKS, math.prod(stride), dtype=DT)
    torch.testing.assert_close(synthesis(analysis(X)), X, **TOL)


def test_intermediate_rotation_invalid_arguments():
    stride = [2, 2, 2]
    with pytest.raises(InvalidBoundaryOperation):
        _intermediate(stride, boundary_operation="Dummy")
    with pytest.raises(InvalidDirection):
        _intermediate(stride, boundary_operation="Termination", direction="Dummy")
    with pytest.raises(InvalidDirection):
        _intermediate(stride, boundary_operation="Termination")
    layer = _intermediate(stride)
    layer.boundary_operation = "Termination"
    with pytest.raises(InvalidDirection):
        layer(torch.zeros(1, *NBLOCKS, 8, dtype=DT))


@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("mode", ["Analysis", "Synthesis"])
@pytest.mark.parametrize("direction", DIRECTIONS)
def test_intermediate_rotation_termination(stride, mode, direction):
    # Boundary blocks: -Ya, other blocks: the same as 'Circular'
    circular = _randomize_intermediate(_intermediate(stride, mode=mode))
    termination = _intermediate(stride, mode=mode, boundary_operation="Termination", direction=direction)
    termination.angles = circular.angles.detach().clone()
    termination.mus = circular.mus.clone()
    nDecs = math.prod(stride)
    ps = nDecs // 2
    X = torch.randn(3, *NBLOCKS, nDecs, dtype=DT)
    expected = circular(X).detach().clone()
    idx = _boundary_slice(direction) + (slice(ps, None),)
    expected[idx] = -X[idx]
    torch.testing.assert_close(termination(X), expected, **TOL)


@pytest.mark.parametrize("mode", ["Analysis", "Synthesis"])
@pytest.mark.parametrize("direction", DIRECTIONS)
def test_intermediate_rotation_termination_angle_gradients(mode, direction):
    # The boundary blocks do not contribute to the angle gradients, and the
    # other blocks have the same gradients as 'Circular'
    stride = [2, 2, 2]
    circular = _randomize_intermediate(_intermediate(stride, mode=mode))
    termination = _intermediate(stride, mode=mode, boundary_operation="Termination", direction=direction)
    termination.angles = circular.angles.detach().clone()
    termination.mus = circular.mus.clone()
    X = torch.randn(3, *NBLOCKS, 8, dtype=DT)
    dLdZ = torch.randn(3, *NBLOCKS, 8, dtype=DT)
    grads = []
    for layer in (circular, termination):
        (layer(X) * dLdZ).sum().backward()
        grads.append(layer.orthTransUnx.orthonormalTransforms.angles.grad)
    expected = grads[0].clone().reshape(*NBLOCKS, -1)
    assert expected.abs().amin(dim=-1).min() > 0
    expected[_boundary_slice(direction)[1:]] = 0.0
    torch.testing.assert_close(grads[1], expected.reshape(math.prod(NBLOCKS), -1), **TOL)


# --- Networks -----------------------------------------------------------------


def _network(stride=(2, 2, 2), ovlp=(3, 3, 3), size=(8, 12, 16), nlevels=0, device="cpu", **kw):
    return LsunAnalysis3dNetwork(
        input_size=list(size),
        stride=list(stride),
        overlapping_factor=list(ovlp),
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
        if isinstance(m, LsunIntermediateRotation3dLayer)
    }


def _as_tuple(coefs):
    return coefs if isinstance(coefs, tuple) else (coefs,)


def test_network_layer_order():
    # The atoms are extended in the depth, horizontal and vertical
    # directions in this order, and the synthesizer is its reverse
    analyzer = _network(nlevels=1)
    assert [name for name, _ in analyzer.layers[0].named_children()] == [
        "Lv1_E0", "Lv1_V0",
        "Lv1_Qd1bd", "Lv1_Vd1", "Lv1_Qd2fs", "Lv1_Vd2",
        "Lv1_Qh1rd", "Lv1_Vh1", "Lv1_Qh2ls", "Lv1_Vh2",
        "Lv1_Qv1dd", "Lv1_Vv1", "Lv1_Qv2us", "Lv1_Vv2",
        "Lv1_Sp",
    ]  # fmt: skip
    assert [name for name, _ in analyzer.T.layers[0].named_children()] == [
        "Lv1_Cc",
        "Lv1_Vv~2", "Lv1_Qv~2us", "Lv1_Vv~1", "Lv1_Qv~1dd",
        "Lv1_Vh~2", "Lv1_Qh~2ls", "Lv1_Vh~1", "Lv1_Qh~1rd",
        "Lv1_Vd~2", "Lv1_Qd~2fs", "Lv1_Vd~1", "Lv1_Qd~1bd",
        "Lv1_V0~", "Lv1_E0~",
    ]  # fmt: skip


@pytest.mark.parametrize("boundary_operation", ["Circular", "Termination"])
@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize("ovlp", [(3, 3, 3), (1, 3, 5), (5, 1, 3)])
def test_network_layer_settings(boundary_operation, adjoint, ovlp):
    # Odd-order Vd/Vh/Vv get boundary_operation with 'Depth'/'Horizontal'/
    # 'Vertical'; even-order ones stay 'Circular'. The adjoint inherits them.
    net = _network(ovlp=ovlp, boundary_operation=boundary_operation)
    if adjoint:
        net = net.T
        assert isinstance(net, LsunSynthesis3dNetwork)
        assert net.boundary_operation == boundary_operation
    layers = _intermediate_layers(net)
    assert len(layers) == sum(ovlp) - 3
    for name, layer in layers.items():
        label = name.split("_V")[-1][0]
        order = int(name.split("_V")[-1][1:].lstrip("~"))
        assert layer.mode == ("Synthesis" if adjoint else "Analysis"), name
        if order % 2 == 1:
            assert layer.boundary_operation == boundary_operation, name
            assert layer.direction == {"d": "Depth", "h": "Horizontal", "v": "Vertical"}[label], name
        else:
            assert layer.boundary_operation == "Circular", name


def test_network_number_of_blocks_per_level():
    size, stride, nlevels = (8, 16, 24), (2, 2, 2), 3
    for net in (_network(size=size, nlevels=nlevels), _network(size=size, nlevels=nlevels).T):
        layers = net.rotation_layers()
        assert len(layers) == nlevels * 7
        for name, layer in layers.items():
            ilevel = int(name[2 : name.index("_")])
            assert list(layer.number_of_blocks) == [sz // dec**ilevel for sz, dec in zip(size, stride)], name


def test_network_invalid_arguments():
    with pytest.raises(InvalidBoundaryOperation):
        _network(boundary_operation="Dummy")
    with pytest.raises(InvalidStride):
        _network(stride=(1, 1, 1), size=(8, 8, 8))
    with pytest.raises(InvalidStride):
        _network(stride=(2, 2))
    with pytest.raises(InvalidOverlappingFactor):
        _network(ovlp=(3, 2, 3))
    with pytest.raises(InvalidInputSize):
        _network(size=(8, 12, 15))
    with pytest.raises(InvalidInputSize):
        _network(size=(8, 12, 16), nlevels=3)
    with pytest.raises(InvalidNumberOfLevels):
        _network(nlevels=-1)
    with pytest.raises(ValueError):
        _network(number_of_components=2)


@pytest.mark.parametrize("stride", [(2, 2, 2), (1, 2, 2)])
@pytest.mark.parametrize("ovlp", [(3, 3, 3), (1, 3, 5), (5, 1, 3)])
@pytest.mark.parametrize("boundary_operation", ["Circular", "Termination"])
def test_network_is_block_dct_by_default(stride, ovlp, boundary_operation):
    size = [4 * dec for dec in stride]
    analyzer = _network(stride=stride, ovlp=ovlp, size=size, boundary_operation=boundary_operation)
    X = torch.randn(2, 1, *size, dtype=DT)
    with torch.no_grad():
        torch.testing.assert_close(analyzer(X), LsunBlockDct3dLayer(stride=list(stride))(X), **TOL)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("nlevels", [0, 1, 2])
@pytest.mark.parametrize("boundary_operation", ["Circular", "Termination"])
@pytest.mark.parametrize("stride, ovlp", [((2, 2, 2), (3, 3, 3)), ((2, 2, 2), (1, 3, 5)), ((1, 2, 2), (5, 3, 1))])
def test_network_perfect_reconstruction(device, nlevels, boundary_operation, stride, ovlp):
    size = [4 * dec ** max(nlevels, 1) for dec in stride]
    analyzer = _randomize(
        _network(
            stride=stride, ovlp=ovlp, size=size, nlevels=nlevels, device=device, boundary_operation=boundary_operation
        )
    )
    for layer in _intermediate_layers(analyzer).values():
        layer.mus = _random_mus(layer.mus.shape)
    synthesizer = analyzer.T
    nSamples, nDecs = 2, math.prod(stride)
    X = torch.randn(nSamples, 1, *size, dtype=DT, device=device)
    with torch.no_grad():
        coefs = _as_tuple(analyzer(X))
        Y = synthesizer(coefs if nlevels > 0 else coefs[0])
    # Sizes
    if nlevels == 0:
        assert coefs[0].shape == (nSamples, *[sz // dec for sz, dec in zip(size, stride)], nDecs)
    else:
        assert len(coefs) == nlevels + 1
        nblocks = [sz // dec**nlevels for sz, dec in zip(size, stride)]
        assert coefs[0].shape == (nSamples, *nblocks, 1)
        for iLevel in range(nlevels, 0, -1):
            nblocks = [sz // dec**iLevel for sz, dec in zip(size, stride)]
            assert coefs[nlevels - iLevel + 1].shape == (nSamples, *nblocks, nDecs - 1)
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
    size = (8, 12, 16)
    analyzer = _randomize(_network(size=size, nlevels=nlevels, boundary_operation="Termination"))
    synthesizer = analyzer.T
    X = torch.randn(2, 1, *size, dtype=DT)
    with torch.no_grad():
        coefs = _as_tuple(analyzer(X))
        S = tuple(torch.randn_like(c) for c in coefs)
        Y = synthesizer(S if nlevels > 0 else S[0])
    lhs = sum((c * s).sum() for c, s in zip(coefs, S))
    torch.testing.assert_close(lhs, (X * Y).sum(), **TOL)


def test_synthesizer_constructed_directly():
    # A synthesizer constructed directly is the inverse of the analyzer
    # with the default parameters, and its adjoint is that analyzer
    opts = dict(
        input_size=[8, 12, 16], stride=[2, 2, 2], overlapping_factor=[3, 3, 3], number_of_levels=2, dtype=DT, device=CPU
    )
    synthesizer = LsunSynthesis3dNetwork(**opts)
    analyzer = LsunAnalysis3dNetwork(**opts)
    X = torch.randn(2, 1, 8, 12, 16, dtype=DT)
    with torch.no_grad():
        torch.testing.assert_close(synthesizer(analyzer(X)), X, **TOL)
        for actual, expected in zip(synthesizer.T(X), analyzer(X)):
            torch.testing.assert_close(actual, expected, **TOL)


def _edge_response(boundary_operation, axis, nlevels):
    """Max |coefs| on the last blocks along axis (1: rows, 2: cols, 3: lays)
    for an input on the first blocks."""
    stride = (2, 2, 2)
    size = (16, 16, 16)
    analyzer = _randomize(
        _network(stride=stride, ovlp=(3, 3, 3), size=size, nlevels=nlevels, boundary_operation=boundary_operation)
    )
    X = torch.zeros(1, 1, *size, dtype=DT)
    X.narrow(axis + 1, 0, stride[axis - 1] ** max(nlevels, 1)).normal_()
    with torch.no_grad():
        coefs = _as_tuple(analyzer(X))
    return max(c.narrow(axis, c.shape[axis] - 1, 1).abs().max().item() for c in coefs)


@pytest.mark.parametrize("axis", [1, 2, 3])
@pytest.mark.parametrize("nlevels", [0, 2])
def test_network_no_wrap_around(axis, nlevels):
    assert _edge_response("Termination", axis, nlevels) == pytest.approx(0.0, abs=1e-10)
    assert _edge_response("Circular", axis, nlevels) > 1e-4


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("boundary_operation", ["Circular", "Termination"])
def test_network_gradients(device, boundary_operation):
    # The angles of all the levels are trainable through the analyzer and
    # the synthesizer
    size = (8, 8, 8)
    analyzer = _randomize(_network(size=size, nlevels=2, device=device, boundary_operation=boundary_operation))
    synthesizer = analyzer.T
    X = torch.randn(2, 1, *size, dtype=DT, device=device)
    coefs = analyzer(X)
    # Reconstruction from the DC and the AC of level 2 with the attenuated
    # AC of level 1 (zeros would give no gradient to the first rotation of
    # the level-1 stage of the synthesizer)
    attenuated = coefs[:2] + tuple(0.5 * c for c in coefs[2:])
    loss = (synthesizer(attenuated) - X).pow(2).sum() + sum(c.abs().sum() for c in coefs)
    loss.backward()
    for net in (analyzer, synthesizer):
        params = dict(net.named_parameters())
        assert len(params) == 2 * (2 + 6)  # per level: W0, U0 and six Un
        for name, param in params.items():
            assert param.grad is not None, name
            assert torch.isfinite(param.grad).all(), name
            assert param.grad.abs().max() > 0, name


def test_network_no_dc_leakage():
    # A constant input gives only the DC coefficients
    size = (8, 12, 16)
    analyzer = _randomize(_network(size=size, nlevels=2, no_dc_leakage=True))
    X = torch.full((1, 1, *size), 3.0, dtype=DT)
    with torch.no_grad():
        coefs = analyzer(X)
    torch.testing.assert_close(coefs[0], torch.full_like(coefs[0], 3.0 * math.sqrt(8) ** 2), **TOL)
    for ac in coefs[1:]:
        torch.testing.assert_close(ac, torch.zeros_like(ac), **TOL)


def test_network_prefix():
    analyzer = _randomize(_network(nlevels=1, prefix="P_"))
    assert all(name.startswith("P_Lv1_") for name, _ in analyzer.layers[0].named_children())
    assert sorted(analyzer.rotation_layers()) == sorted(_network(nlevels=1).rotation_layers())
    X = torch.randn(1, 1, 8, 12, 16, dtype=DT)
    with torch.no_grad():
        torch.testing.assert_close(analyzer.T(analyzer(X)), X, **TOL)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_network_to_device():
    analyzer = _randomize(_network(nlevels=1))
    X = torch.randn(1, 1, 8, 12, 16, dtype=DT)
    with torch.no_grad():
        expected = analyzer(X)
        analyzer = analyzer.to(device=torch.device("cuda"))
        actual = analyzer(X.cuda())
        Y = analyzer.T(actual)
    assert Y.device.type == "cuda"
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a.cpu(), e, **TOL)
    torch.testing.assert_close(Y.cpu(), X, **TOL)
