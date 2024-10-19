import torch
import torch.nn as nn
from .lsunBlockIdct2dLayer import LsunBlockIdct2dLayer 
from .lsunFinalRotation2dLayer import LsunFinalRotation2dLayer 
from .lsunAtomExtension2dLayer import LsunAtomExtension2dLayer
from .lsunIntermediateRotation2dLayer import LsunIntermediateRotation2dLayer
from .lsunChannelConcatenation2dLayer import LsunChannelConcatenation2dLayer
from .lsunLayerExceptions import InvalidOverlappingFactor, InvalidNoDcLeakage, InvalidNumberOfLevels, InvalidStride, InvalidInputSize
from .lsunUtility import Direction

class LsunSynthesis2dNetwork(nn.Module):
    """
    LSUNSYNTHESIS2DNETWORK
    
    Requirements: Requirements: Python 3.10-12.x, PyTorch 2.3/4.x
    
    Copyright (c) 2024, Shogo MURAMATSU
    
    All rights reserved.
    
    Contact address: Shogo MURAMATSU,
        Faculty of Engineering, Niigata University,
        8050 2-no-cho Ikarashi, Nishi-ku,
        Niigata, 950-2181, JAPAN
    
        https://www.eng.niigata-u.ac.jp/~msiplab/
    """
    def __init__(self,
        input_size=[2, 2], 
        number_of_components=1,
        stride=[2, 2],
        overlapping_factor=[1,1],
        no_dc_leakage=True,
        number_of_levels=0,
        prefix='',
        dtype=torch.get_default_dtype(),
        device=torch.get_default_device()
        ):
        super(LsunSynthesis2dNetwork, self).__init__()
        
        # Check and set parameters
        self.dtype = dtype
        self.device = device
        
        # Stride
        nDecs = stride[Direction.VERTICAL]*stride[Direction.HORIZONTAL]
        if nDecs%2!=0:
            raise InvalidStride(
            '%d x %d : Currently, even product of strides is only supported.'\
            % (stride[Direction.VERTICAL], stride[Direction.HORIZONTAL]))        
        self.stride = stride
        
        # Overlapping factor
        if (overlapping_factor[Direction.VERTICAL]-1)%2 or (overlapping_factor[Direction.HORIZONTAL]-1)%2: 
             raise InvalidOverlappingFactor(
             '%d + %d : Currently, odd overlapping factors are only supported.'\
             % (overlapping_factor[Direction.VERTICAL], overlapping_factor[Direction.HORIZONTAL]))
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

        # # of blocks
        if input_size[Direction.VERTICAL]%stride[Direction.VERTICAL] != 0 or input_size[Direction.HORIZONTAL]%stride[Direction.HORIZONTAL] != 0:
            raise InvalidInputSize(
            '%d x %d : Currently, multiples of strides is only supported.'\
            % (input_size[Direction.VERTICAL], input_size[Direction.HORIZONTAL]))
        if input_size[Direction.VERTICAL] < 1 or input_size[Direction.HORIZONTAL] < 1:
            raise InvalidInputSize(
            '%d x %d : Positive integers are only supported.'\
            % (input_size[Direction.VERTICAL], input_size[Direction.HORIZONTAL]))
        if number_of_levels > 0:   
            nrows = input_size[Direction.VERTICAL]//(stride[Direction.VERTICAL]**number_of_levels)
            ncols = input_size[Direction.HORIZONTAL]//(stride[Direction.HORIZONTAL]**number_of_levels)
        else:
            nrows = input_size[Direction.VERTICAL]//stride[Direction.VERTICAL]
            ncols = input_size[Direction.HORIZONTAL]//stride[Direction.HORIZONTAL]
        self.input_size = input_size
        
        # Instantiation of layers
        if self.number_of_levels == 0:
            nlevels = 1
        else:
            nlevels = self.number_of_levels
            
        stages = [ nn.Sequential() for iStage in range(nlevels) ]
        for iStage in range(len(stages)):
            iLevel = nlevels - iStage
            strLv = 'Lv%0d_'%iLevel
            
            # Channel Concatanation 
            if self.number_of_levels > 0:
                stages[iStage].add_module(strLv+'Cc',LsunChannelConcatenation2dLayer())
            
            # Vertical concatenation
            for iOrderV in range(overlapping_factor[Direction.VERTICAL]-1,1,-2):  
                stages[iStage].add_module(strLv+'Vv~%d'%(iOrderV),LsunIntermediateRotation2dLayer(
                    stride=self.stride,
                    number_of_blocks=[nrows,ncols],
                    mode='Synthesis',
                    mus=-1,
                    dtype=self.dtype,device=self.device))
                stages[iStage].add_module(strLv+'Qv~%dus'%(iOrderV),LsunAtomExtension2dLayer(
                    stride=self.stride,
                    direction='Down',
                    target_channels='Sum'))
                stages[iStage].add_module(strLv+'Vv~%d'%(iOrderV-1),LsunIntermediateRotation2dLayer(
                    stride=self.stride,
                    number_of_blocks=[nrows,ncols],
                    mode='Synthesis',
                    mus=-1,
                    dtype=self.dtype,device=self.device))
                stages[iStage].add_module(strLv+'Qv~%ddd'%(iOrderV-1),LsunAtomExtension2dLayer(
                    stride=self.stride,
                    direction='Up',
                    target_channels='Difference'))
            
            # Horizontal concatenation
            for iOrderH in range(overlapping_factor[Direction.HORIZONTAL]-1,1,-2):
                stages[iStage].add_module(strLv+'Vh~%d'%(iOrderH),LsunIntermediateRotation2dLayer(
                    stride=self.stride,
                    number_of_blocks=[nrows,ncols],
                    mode='Synthesis',
                    mus=-1,
                    dtype=self.dtype,device=self.device))
                stages[iStage].add_module(strLv+'Qh~%dls'%(iOrderH),LsunAtomExtension2dLayer(
                    stride=self.stride,
                    direction='Right',
                    target_channels='Sum'))
                stages[iStage].add_module(strLv+'Vh~%d'%(iOrderH-1),LsunIntermediateRotation2dLayer(
                    stride=self.stride,
                    number_of_blocks=[nrows,ncols],
                    mode='Synthesis',
                    mus=-1,
                    dtype=self.dtype,device=self.device))
                stages[iStage].add_module(strLv+'Qh~%drd'%(iOrderH-1),LsunAtomExtension2dLayer(
                    stride=self.stride,
                    direction='Left',
                    target_channels='Difference'))
                
            stages[iStage].add_module(prefix+strLv+'V0~',LsunFinalRotation2dLayer(
                stride=self.stride,
                number_of_blocks=[nrows,ncols],
                mus=1,
                no_dc_leakage=self.no_dc_leakage,
                dtype=self.dtype,device=self.device))
            
            stages[iStage].add_module(prefix+strLv+'E0~',LsunBlockIdct2dLayer(
                stride=self.stride
                ))    
            
            # Update size
            nrows *= stride[Direction.VERTICAL]
            ncols *= stride[Direction.HORIZONTAL]   

        # Stack modules as a list
        self.layers = nn.ModuleList(stages)
            
    def forward(self,x):
        # TODO: #11 Stream processing

        if self.number_of_levels == 0: # Flat structure
            m = self.layers[0]
            x = m.forward(x)
            return x
        else: # tree structure
            stride = self.stride
            nSamples = x[0].size(0)
            nrows = x[0].size(1)
            ncols = x[0].size(2)
            iLevel = self.number_of_levels
            for m in self.layers:
                if iLevel == self.number_of_levels:
                    xdc = x[0]
                xac = x[self.number_of_levels-iLevel+1]
                y = m[0].forward(xac,xdc)
                y = m[1::].forward(y)
                nrows *= stride[Direction.VERTICAL]
                ncols *= stride[Direction.HORIZONTAL]
                xdc = y.reshape(nSamples,nrows,ncols,1)             
                iLevel -= 1
            return xdc.view(nSamples,1,nrows,ncols)

    @property
    def T(self):
        from .lsunAnalysis2dNetwork import LsunAnalysis2dNetwork
        import re

        # Create analyzer as the adjoint of SELF
        analyzer = LsunAnalysis2dNetwork(
            input_size=self.input_size,
            stride=self.stride,
            overlapping_factor=self.overlapping_factor,
            no_dc_leakage=self.no_dc_leakage,
            number_of_levels=self.number_of_levels,
            dtype=self.dtype,
            device=self.device            
        )

        if self.number_of_levels == 0:
            nlevels = 1
        else:
            nlevels = self.number_of_levels

        # Copy state dictionary
        syn_state_dict = self.state_dict()
        ana_state_dict = analyzer.state_dict()
        for key in syn_state_dict.keys():
            istage_ana = int(re.sub('^layers\.|\.Lv\d_.+$','',key))            
            istage_syn = (nlevels-1)-istage_ana
            angs = syn_state_dict[key]
            ana_state_dict[key.replace('layers.%d'%istage_ana,'layers.%d'%istage_syn).replace('~','').replace('T.orthonormalTransforms','.orthonormalTransforms') ] = angs

        # Load state dictionary
        analyzer.load_state_dict(ana_state_dict)

        # Return adjoint
        return analyzer #.to(dtype=self.dtype,device=self.device)
    
    
    def to(self, device=None, dtype=None,*args, **kwargs):
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        super(LsunSynthesis2dNetwork, self).to(device=self.device,dtype=self.dtype,*args, **kwargs)
        for s in self.layers:
            for m in s:
                m.to(device=self.device,dtype=self.dtype,*args, **kwargs)
        return self
