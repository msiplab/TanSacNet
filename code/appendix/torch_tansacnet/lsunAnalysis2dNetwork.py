import torch
import torch.nn as nn
from lsunBlockDct2dLayer import LsunBlockDct2dLayer 
from lsunInitialRotation2dLayer import LsunInitialRotation2dLayer 
#from lsunAtomExtension2dLayer import LsunAtomExtension2dLayer
#from lsunIntermediateRotation2dLayer import LsunIntermediateRotation2dLayer
#from lsunChannelSeparation2dLayer import LsunChannelSeparation2dLayer
from lsunLayerExceptions import InvalidOverlappingFactor, InvalidNoDcLeakage, InvalidNumberOfLevels, InvalidStride, InvalidInputSize
from lsunUtility import Direction

class LsunAnalysis2dNetwork(nn.Module):
    """
    LSUNANALYSIS2DNETWORK
    
    Requirements: Requirements: Python 3.10/11.x, PyTorch 2.3.x
    
    Copyright (c) 2024, Yasas Dulanjaya, Shogo MURAMATSU
    
    All rights reserved.
    
    Contact address: Shogo MURAMATSU,
        Faculty of Engineering, Niigata University,
        8050 2-no-cho Ikarashi, Nishi-ku,
        Niigata, 950-2181, JAPAN
    
        https://www.eng.niigata-u.ac.jp/~msiplab/
    """
    def __init__(self,
        input_size=[2, 2], 
        stride=[2, 2],
        overlapping_factor=[1,1],
        no_dc_leakage=True,
        number_of_levels=0
        ):
        super(LsunAnalysis2dNetwork, self).__init__()

        # Check and set parameters
     
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

        # Number of blocks
        if input_size[Direction.VERTICAL]%stride[Direction.VERTICAL] != 0 or input_size[Direction.HORIZONTAL]%stride[Direction.HORIZONTAL] != 0:
            raise InvalidInputSize(
            '%d x %d : Currently, multiples of strides is only supported.'\
            % (input_size[Direction.VERTICAL], input_size[Direction.HORIZONTAL]))
        if input_size[Direction.VERTICAL] < 1 or input_size[Direction.HORIZONTAL] < 1:
            raise InvalidInputSize(
            '%d x %d : Positive integers are only supported.'\
            % (input_size[Direction.VERTICAL], input_size[Direction.HORIZONTAL]))
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
            iLevel = iStage+1
            strLv = 'Lv%0d_'%iLevel

            # Initial blocks
            stages[iStage].add_module(strLv+'E0',LsunBlockDct2dLayer(
                decimation_factor=self.stride
                ))

            stages[iStage].add_module(strLv+'V0',LsunInitialRotation2dLayer(
                stride=self.stride,
                number_of_blocks=[nrows,ncols], 
                mus=1,               
                no_dc_leakage=self.no_dc_leakage
               ))

            # Horizontal extension
        #    for iOrderH in range(2,polyphase_order[Direction.HORIZONTAL]+1,2):
        #        stages[iStage].add_module(strLv+'Qh%drd'%(iOrderH-1),LsunAtomExtension2dLayer(
        #            number_of_channels=number_of_channels,
        #            direction='Right',
        #            target_channels='Difference'))
        #        stages[iStage].add_module(strLv+'Vh%d'%(iOrderH-1),LsunIntermediateRotation2dLayer(
        #            number_of_channels=number_of_channels,
        #            mode='Analysis',
        #            mus=-1))
        #        stages[iStage].add_module(strLv+'Qh%dls'%iOrderH,LsunAtomExtension2dLayer(
        #            number_of_channels=number_of_channels,
        #            direction='Left',
        #            target_channels='Sum'))
        #        stages[iStage].add_module(strLv+'Vh%d'%iOrderH,LsunIntermediateRotation2dLayer(
        #            number_of_channels=number_of_channels,
        #            mode='Analysis',
        #            mus=-1))
                
            # Vertical extension
        #    for iOrderV in range(2,polyphase_order[Direction.VERTICAL]+1,2):            
        #        stages[iStage].add_module(strLv+'Qv%ddd'%(iOrderV-1),LsunAtomExtension2dLayer(
        #            number_of_channels=number_of_channels,
        #            direction='Down',
        #            target_channels='Difference'))                
        #        stages[iStage].add_module(strLv+'Vv%d'%(iOrderV-1),LsunIntermediateRotation2dLayer(
        #            number_of_channels=number_of_channels,
        #            mode='Analysis',
        #            mus=-1))
        #        stages[iStage].add_module(strLv+'Qv%dus'%iOrderV,LsunAtomExtension2dLayer(
        #            number_of_channels=number_of_channels,
        #            direction='Up',
        #            target_channels='Sum'))                
        #        stages[iStage].add_module(strLv+'Vv%d'%iOrderV,LsunIntermediateRotation2dLayer(
        #            number_of_channels=number_of_channels,
        #            mode='Analysis',
        #            mus=-1))
            # Channel Separation for intermediate stages
        #    if self.number_of_levels > 0:
        #        stages[iStage].add_module(strLv+'Sp',LsunChannelSeparation2dLayer())

        # Stack modules as a list
        self.layers = nn.ModuleList(stages)

        # Set flag
        self.hasSetup_ = True
        
    def forward(self,x):

        if self.number_of_levels == 0: # Flat structure
            for m in self.layers:
                y = m.forward(x) 
            return y
        #else: # Tree structure
        #    stride = self.stride
        #    nSamples = x.size(0)
        #    nComponents = x.size(1)
        #    nrows = int(x.size(2)/stride[Direction.VERTICAL])
        #    ncols = int(x.size(3)/stride[Direction.HORIZONTAL])
        #    y = []
        #    iLevel = 1                   
        #    for m in self.layers:
        #        yac, ydc = m.forward(x)
        #        y.insert(0,yac)
        #        if iLevel < self.number_of_levels:
        #            x = ydc.view(nSamples,nComponents,nrows,ncols)
        #            nrows = int(nrows/stride[Direction.VERTICAL])
        #            ncols = int(ncols/stride[Direction.HORIZONTAL])     
        #            iLevel += 1             
        #        else:
        #            y.insert(0,ydc)
        #   return tuple(y)
    
    """
    @property
    def T(self):
        from lsunSynthesis2dNetwork import LsunSynthesis2dNetwork
        import re

        # Create synthesizer as the adjoint of SELF
        synthesizer = LsunSynthesis2dNetwork(
            stride=self.stride,
            overlapping_factor=self.overlapping_factor,
            no_dc_leakage=self.no_dc_leakage,
            number_of_levels=self.number_of_levels            
        )

        if self.number_of_levels == 0:
            nlevels = 1
        else:
            nlevels = self.number_of_levels

        # Copy state dictionary
        ana_state_dict = self.state_dict()
        syn_state_dict = synthesizer.state_dict()
        for key in syn_state_dict.keys():
            istage_ana = int(re.sub('^layers\.|\.Lv\d_.+$','',key))
            istage_syn = (nlevels-1)-istage_ana
            angs = ana_state_dict[key\
                .replace('layers.%d'%istage_ana,'layers.%d'%istage_syn)\
                .replace('~','')\
                .replace('T.angles','.angles') ] 
            syn_state_dict[key] = angs
        
        # Load state dictionary
        synthesizer.load_state_dict(syn_state_dict)

        # Return adjoint
        return synthesizer.to(angs.device)
    """