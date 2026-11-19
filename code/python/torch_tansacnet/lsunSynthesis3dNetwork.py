import math
import torch
import torch.nn as nn
from .lsunBlockIdct3dLayer import LsunBlockIdct3dLayer
from .lsunFinalRotation3dLayer import LsunFinalRotation3dLayer
from .lsunAtomExtension3dLayer import LsunAtomExtension3dLayer
from .lsunIntermediateRotation3dLayer import LsunIntermediateRotation3dLayer
from .lsunChannelConcatenation3dLayer import LsunChannelConcatenation3dLayer
from .lsunLayerExceptions import InvalidOverlappingFactor, InvalidNoDcLeakage, InvalidNumberOfLevels, InvalidStride, InvalidInputSize, InvalidBoundaryOperation
from .lsunAnalysis3dNetwork import ATOM_EXTENSION_STEPS_3D

# Direction of the shift of the synthesizer for that of the analyzer
REVERSE_DIRECTION = {
    'Right': 'Left', 'Left': 'Right',
    'Down': 'Up', 'Up': 'Down',
    'Back': 'Front', 'Front': 'Back' }

class LsunSynthesis3dNetwork(nn.Module):
    """
    LSUNSYNTHESIS3DNETWORK

        3-D counterpart of LsunSynthesis2dNetwork, which corresponds to the
        synthesizer of MATLAB fcn_createlsunlgraph3d. It is the adjoint of
        LsunAnalysis3dNetwork with the same parameters.

        input_size, stride and overlapping_factor are given in the order
        of [vertical, horizontal, depth].

        Input (number_of_levels = 0):
            nSamples x nRows x nCols x nLays x nDecs

        Input (number_of_levels = L > 0): tuple of
            nSamples x nRows_L x nCols_L x nLays_L x 1          (DC of level L)
            nSamples x nRows_L x nCols_L x nLays_L x (nDecs-1)  (AC of level L)
            ...
            nSamples x nRows_1 x nCols_1 x nLays_1 x (nDecs-1)  (AC of level 1)

        Output:
            nSamples x 1 x Height x Width x Depth

    Requirements: Python 3.10-12.x, PyTorch 2.3/4.x

    Copyright (c) 2026, Shogo MURAMATSU

    All rights reserved.

    Contact address: Shogo MURAMATSU,
        Faculty of Engineering, Niigata University,
        8050 2-no-cho Ikarashi, Nishi-ku,
        Niigata, 950-2181, JAPAN

    https://www.eng.niigata-u.ac.jp/~msiplab/
    """
    def __init__(self,
        input_size=[2, 2, 2],
        number_of_components=1,
        stride=[2, 2, 2],
        overlapping_factor=[1, 1, 1],
        no_dc_leakage=True,
        number_of_levels=0,
        prefix='',
        dtype=torch.get_default_dtype(),
        device=torch.get_default_device(),
        boundary_operation='Circular'
        ):
        super(LsunSynthesis3dNetwork, self).__init__()

        # Check and set parameters
        self.dtype = dtype
        self.device = device
        self.prefix = prefix

        # Number of components
        if number_of_components != 1:
            raise ValueError(
                '%d : Currently, a single component is only supported.'\
                % number_of_components)
        self.number_of_components = number_of_components

        # Boundary operation of the odd-order intermediate rotation layers,
        # i.e., those right after the shift of the difference channels.
        # 'Termination' fixes the rotation matrices of their boundary blocks
        # to -I so that no coefficients wrap around to the opposite boundary
        # (MATLAB fcn_createlsunlgraph3d with 'BoundaryOperation').
        # The even-order layers are always 'Circular'.
        if boundary_operation not in ('Circular', 'Termination'):
            raise InvalidBoundaryOperation(
                f"{boundary_operation} : BoundaryOperation should be either of Circular or Termination")
        self.boundary_operation = boundary_operation

        # Stride
        if len(stride) != 3 or math.prod(stride)%2 != 0:
            raise InvalidStride(
                '%s : Currently, three strides of even product are only supported.'\
                % str(list(stride)))
        self.stride = stride

        # Overlapping factor
        if len(overlapping_factor) != 3 or any((ovlp-1)%2 for ovlp in overlapping_factor):
            raise InvalidOverlappingFactor(
                '%s : Currently, three odd overlapping factors are only supported.'\
                % str(list(overlapping_factor)))
        self.overlapping_factor = overlapping_factor

        # No DC leakage
        if not isinstance(no_dc_leakage, bool):
            raise InvalidNoDcLeakage("no_dc_leakage must be a boolean value")
        self.no_dc_leakage = no_dc_leakage

        # # of levels
        if not isinstance(number_of_levels, int):
            raise InvalidNumberOfLevels(
                '%f : The number of levels must be integer.'\
                % number_of_levels)
        if number_of_levels < 0:
            raise InvalidNumberOfLevels(
                '%d : The number of levels must be greater than or equal to 0.'\
                % number_of_levels)
        self.number_of_levels = number_of_levels
        nlevels = max(self.number_of_levels,1)

        # # of blocks
        if len(input_size) != 3 or any(sz < 1 for sz in input_size):
            raise InvalidInputSize(
                '%s : Three positive integers are only supported.'\
                % str(list(input_size)))
        if any(sz%(dec**nlevels) for sz, dec in zip(input_size,stride)):
            raise InvalidInputSize(
                '%s : Currently, multiples of stride**number_of_levels are only supported.'\
                % str(list(input_size)))
        self.input_size = input_size

        # Instantiation of layers
        nblocks = [ sz//(dec**nlevels) for sz, dec in zip(input_size,stride) ]
        stages = [ nn.Sequential() for iStage in range(nlevels) ]
        for iStage in range(len(stages)):
            iLevel = nlevels - iStage
            strLv = prefix+'Lv%0d_'%iLevel
            rotopts = dict(stride=self.stride,number_of_blocks=list(nblocks),
                dtype=self.dtype,device=self.device)

            # Channel concatenation for the tree structure
            if self.number_of_levels > 0:
                stages[iStage].add_module(strLv+'Cc',LsunChannelConcatenation3dLayer())

            # Atom concatenation in vertical, horizontal and depth
            for idx, lbl, dirs, sfx, bdir in reversed(ATOM_EXTENSION_STEPS_3D):
                for iOrder in range(overlapping_factor[idx]-1,1,-2):
                    stages[iStage].add_module(strLv+'V%s~%d'%(lbl,iOrder),LsunIntermediateRotation3dLayer(
                        mode='Synthesis',
                        mus=-1,
                        **rotopts))
                    stages[iStage].add_module(strLv+'Q%s~%d%s'%(lbl,iOrder,sfx[1]),LsunAtomExtension3dLayer(
                        stride=self.stride,
                        direction=REVERSE_DIRECTION[dirs[1]],
                        target_channels='Sum'))
                    stages[iStage].add_module(strLv+'V%s~%d'%(lbl,iOrder-1),LsunIntermediateRotation3dLayer(
                        mode='Synthesis',
                        mus=-1,
                        boundary_operation=self.boundary_operation,
                        direction=bdir,
                        **rotopts))
                    stages[iStage].add_module(strLv+'Q%s~%d%s'%(lbl,iOrder-1,sfx[0]),LsunAtomExtension3dLayer(
                        stride=self.stride,
                        direction=REVERSE_DIRECTION[dirs[0]],
                        target_channels='Difference'))

            # Final blocks
            stages[iStage].add_module(strLv+'V0~',LsunFinalRotation3dLayer(
                mus=1,
                no_dc_leakage=self.no_dc_leakage,
                **rotopts))
            stages[iStage].add_module(strLv+'E0~',LsunBlockIdct3dLayer(
                stride=self.stride))

            # Update size
            nblocks = [ nblk*dec for nblk, dec in zip(nblocks,stride) ]

        # Stack modules as a list
        self.layers = nn.ModuleList(stages)

    def forward(self,x):
        if self.number_of_levels == 0: # Flat structure
            return self.layers[0](x)
        else: # Tree structure
            xdc = x[0]
            for iStage, m in enumerate(self.layers):
                xac = x[iStage+1]
                y = m[0](xac,xdc)
                y = m[1:](y)
                # nSamples x 1 x Height x Width x Depth
                #   -> nSamples x nRows x nCols x nLays x 1
                xdc = y.squeeze(1).unsqueeze(-1)
            return y

    @property
    def T(self):
        from .lsunAnalysis3dNetwork import LsunAnalysis3dNetwork

        # Create analyzer as the adjoint of SELF
        analyzer = LsunAnalysis3dNetwork(
            input_size=self.input_size,
            number_of_components=self.number_of_components,
            stride=self.stride,
            overlapping_factor=self.overlapping_factor,
            no_dc_leakage=self.no_dc_leakage,
            number_of_levels=self.number_of_levels,
            prefix=self.prefix,
            dtype=self.dtype,
            device=self.device,
            boundary_operation=self.boundary_operation
            )
        return self._copy_parameters_to(analyzer)

    def rotation_layers(self):
        """
        Dictionary of the rotation layers. The keys are the layer names
        without '~' and the prefix, which are common to the analyzer and
        the synthesizer, e.g., 'Lv1_V0' and 'Lv1_Vd1'.
        """
        return { name[len(self.prefix):].replace('~',''): m
            for stage in self.layers for name, m in stage.named_children()
            if hasattr(m,'angles') }

    def _copy_parameters_to(self, adjoint):
        # Angles, mus and the boundary operation of the rotation layers
        dst_layers = adjoint.rotation_layers()
        for key, src in self.rotation_layers().items():
            dst = dst_layers[key]
            dst.angles = src.angles.detach().clone()
            dst.mus = src.mus.detach().clone()
            if hasattr(src,'boundary_operation'):
                dst.direction = src.direction
                dst.boundary_operation = src.boundary_operation
        return adjoint

    def to(self, device=None, dtype=None,*args, **kwargs):
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        super(LsunSynthesis3dNetwork, self).to(device=self.device,dtype=self.dtype,*args, **kwargs)
        for s in self.layers:
            for m in s:
                m.to(device=self.device,dtype=self.dtype,*args, **kwargs)
        return self
