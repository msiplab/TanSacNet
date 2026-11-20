import torch
import torch.nn as nn
from .lsunBlockDct1dLayer import LsunBlockDct1dLayer
from .lsunInitialRotation1dLayer import LsunInitialRotation1dLayer
from .lsunAtomExtension1dLayer import LsunAtomExtension1dLayer
from .lsunIntermediateRotation1dLayer import LsunIntermediateRotation1dLayer
from .lsunChannelSeparation1dLayer import LsunChannelSeparation1dLayer
from .lsunLayerExceptions import InvalidOverlappingFactor, InvalidNoDcLeakage, InvalidNumberOfLevels, InvalidStride, InvalidInputSize, InvalidBoundaryOperation

class LsunAnalysis1dNetwork(nn.Module):
    """
    LSUNANALYSIS1DNETWORK

        1-D counterpart of LsunAnalysis2dNetwork, which corresponds to the
        analyzer of MATLAB fcn_createlsunlgraph1d. It has the same
        structure as the 2-D and 3-D LSUNs (not the CS type).

        input_size (sequence length), stride and overlapping_factor are
        integers.

        Input:
            nSamples x 1 x Length

        Output (number_of_levels = 0):
            nSamples x nBlks x nDecs

        Output (number_of_levels = L > 0): tuple of
            nSamples x nBlks_L x 1          (DC of level L)
            nSamples x nBlks_L x (nDecs-1)  (AC of level L)
            ...
            nSamples x nBlks_1 x (nDecs-1)  (AC of level 1)

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
        input_size=2,
        number_of_components=1,
        stride=2,
        overlapping_factor=1,
        no_dc_leakage=True,
        number_of_levels=0,
        prefix='',
        dtype=torch.get_default_dtype(),
        device=torch.get_default_device(),
        boundary_operation='Circular'
        ):
        super(LsunAnalysis1dNetwork, self).__init__()

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
        # (MATLAB fcn_createlsunlgraph1d with 'BoundaryOperation').
        # The even-order layers are always 'Circular'.
        if boundary_operation not in ('Circular', 'Termination'):
            raise InvalidBoundaryOperation(
                f"{boundary_operation} : BoundaryOperation should be either of Circular or Termination")
        self.boundary_operation = boundary_operation

        # Stride
        if not isinstance(stride, int) or stride < 2 or stride%2 != 0:
            raise InvalidStride(
                '%s : Currently, an even stride is only supported.'\
                % str(stride))
        self.stride = stride

        # Overlapping factor
        if not isinstance(overlapping_factor, int) or overlapping_factor < 1 \
                or (overlapping_factor-1)%2 != 0:
            raise InvalidOverlappingFactor(
                '%s : Currently, an odd overlapping factor is only supported.'\
                % str(overlapping_factor))
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
        if not isinstance(input_size, int) or input_size < 1:
            raise InvalidInputSize(
                '%s : A positive integer is only supported.'\
                % str(input_size))
        if input_size%(stride**nlevels):
            raise InvalidInputSize(
                '%d : Currently, a multiple of stride**number_of_levels is only supported.'\
                % input_size)
        self.input_size = input_size

        # Instantiation of layers
        nblocks = input_size//stride
        stages = [ nn.Sequential() for iStage in range(nlevels) ]
        for iStage in range(len(stages)):
            iLevel = iStage + 1
            strLv = prefix+'Lv%0d_'%iLevel
            rotopts = dict(stride=self.stride,number_of_blocks=nblocks,
                dtype=self.dtype,device=self.device)

            # Initial blocks
            stages[iStage].add_module(strLv+'E0',LsunBlockDct1dLayer(
                stride=self.stride))
            stages[iStage].add_module(strLv+'V0',LsunInitialRotation1dLayer(
                mus=1,
                no_dc_leakage=self.no_dc_leakage,
                **rotopts))

            # Atom extension
            for iOrder in range(2,overlapping_factor,2):
                stages[iStage].add_module(strLv+'Qx%drd'%(iOrder-1),LsunAtomExtension1dLayer(
                    stride=self.stride,
                    direction='Right',
                    target_channels='Difference'))
                stages[iStage].add_module(strLv+'Vx%d'%(iOrder-1),LsunIntermediateRotation1dLayer(
                    mode='Analysis',
                    mus=-1,
                    boundary_operation=self.boundary_operation,
                    **rotopts))
                stages[iStage].add_module(strLv+'Qx%dls'%iOrder,LsunAtomExtension1dLayer(
                    stride=self.stride,
                    direction='Left',
                    target_channels='Sum'))
                stages[iStage].add_module(strLv+'Vx%d'%iOrder,LsunIntermediateRotation1dLayer(
                    mode='Analysis',
                    mus=-1,
                    **rotopts))

            # Channel separation for the tree structure
            if self.number_of_levels > 0:
                stages[iStage].add_module(strLv+'Sp',LsunChannelSeparation1dLayer())

            # Update size
            nblocks = nblocks//stride

        # Stack modules as a list
        self.layers = nn.ModuleList(stages)

    def forward(self,x):
        if self.number_of_levels == 0: # Flat structure
            return self.layers[0](x)
        else: # Tree structure
            y = []
            for m in self.layers:
                yac, ydc = m(x)
                y.insert(0,yac)
                # nSamples x nBlks -> nSamples x 1 x nBlks
                x = ydc.unsqueeze(1)
            y.insert(0,ydc.unsqueeze(-1))
            return tuple(y)

    @property
    def T(self):
        from .lsunSynthesis1dNetwork import LsunSynthesis1dNetwork

        # Create synthesizer as the adjoint of SELF
        synthesizer = LsunSynthesis1dNetwork(
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
        return self._copy_parameters_to(synthesizer)

    def rotation_layers(self):
        """
        Dictionary of the rotation layers. The keys are the layer names
        without '~' and the prefix, which are common to the analyzer and
        the synthesizer, e.g., 'Lv1_V0' and 'Lv1_Vx1'.
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
                dst.boundary_operation = src.boundary_operation
        return adjoint

    def to(self, device=None, dtype=None,*args, **kwargs):
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        super(LsunAnalysis1dNetwork, self).to(device=self.device,dtype=self.dtype,*args, **kwargs)
        for s in self.layers:
            for m in s:
                m.to(device=self.device,dtype=self.dtype,*args, **kwargs)
        return self
