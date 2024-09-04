import torch
import torch.nn as nn
from lsunBlockIdct2dLayer import LsunBlockIdct2dLayer 
#from lsunFinalRotation2dLayer import LsunFinalRotation2dLayer 
#from lsunAtomExtension2dLayer import LsunAtomExtension2dLayer
#from lsunIntermediateRotation2dLayer import LsunIntermediateRotation2dLayer
#from lsunChannelConcatenation2dLayer import LsunChannelConcatenation2dLayer
from lsunLayerExceptions import InvalidOverlappingFactor, InvalidNoDcLeakage, InvalidNumberOfLevels, InvalidStride
from lsunUtility import Direction

class LsunSynthesis2dNetwork(nn.Module):
    """
    LSUNSYNTHESIS2DNETWORK
    
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
        stride=[2, 2],
        overlapping_factor=[1,1],
        no_dc_leakage=True,
        number_of_levels=0
        ):
        super(LsunSynthesis2dNetwork, self).__init__()
        
        # Check and set parameters
        
        # Stride
        nDecs = stride[Direction.VERTICAL]*stride[Direction.HORIZONTAL]
        if nDecs%2!=0:
            raise InvalidStride(
            '%d x %d : Currently, even product of strides is only supported.'\
            % (stride[Direction.VERTICAL], stride[Direction.HORIZONTAL]))        
        self.stride = stride
        
        # Overlapping factor
        if any((torch.tensor(overlapping_factor)-1)%2):
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
            #if self.number_of_levels > 0:
            #    stages[iStage].add_module(strLv+'Cc',LsunChannelConcatenation2dLayer())
            
            # Vertical concatenation
            #for iOrderV in range(polyphase_order[Direction.VERTICAL],1,-2):            
            #    stages[iStage].add_module(strLv+'Vv~%d'%(iOrderV),LsunIntermediateRotation2dLayer(
            #        number_of_channels=number_of_channels,
            #        mode='Synthesis',
            #       mus=-1))
            #    stages[iStage].add_module(strLv+'Qv~%dus'%(iOrderV),LsunAtomExtension2dLayer(
            #        number_of_channels=number_of_channels,
            #        direction='Down',
            #        target_channels='Sum'))
            #    stages[iStage].add_module(strLv+'Vv~%d'%(iOrderV-1),LsunIntermediateRotation2dLayer(
            #        number_of_channels=number_of_channels,
            #        mode='Synthesis',
            #        mus=-1))
            #    stages[iStage].add_module(strLv+'Qv~%ddd'%(iOrderV-1),LsunAtomExtension2dLayer(
            #        number_of_channels=number_of_channels,
            #        direction='Up',
            #        target_channels='Difference'))
            
            # Horizontal concatenation
            #for iOrderH in range(polyphase_order[Direction.HORIZONTAL],1,-2):
            #    stages[iStage].add_module(strLv+'Vh~%d'%(iOrderH),LsunIntermediateRotation2dLayer(
            #        number_of_channels=number_of_channels,
            #         mode='Synthesis',
            #        mus=-1))
            #    stages[iStage].add_module(strLv+'Qh~%dls'%(iOrderH),LsunAtomExtension2dLayer(
            #        number_of_channels=number_of_channels,
            #        direction='Right',
            #        target_channels='Sum'))
            #    stages[iStage].add_module(strLv+'Vh~%d'%(iOrderH-1),LsunIntermediateRotation2dLayer(
            #        number_of_channels=number_of_channels,
            #        mode='Synthesis',
            #        mus=-1))
            #    stages[iStage].add_module(strLv+'Qh~%drd'%(iOrderH-1),LsunAtomExtension2dLayer(
            #        number_of_channels=number_of_channels,
            #        direction='Left',
            #        target_channels='Difference'))
                
            #stages[iStage].add_module(strLv+'V0~',LsunFinalRotation2dLayer(
            #    number_of_channels=number_of_channels,
            #    decimation_factor=decimation_factor,
            #    no_dc_leakage=(self.number_of_vanishing_moments==1)))
            stages[iStage].add_module(strLv+'E0~',LsunBlockIdct2dLayer(
                decimation_factor=stride
                ))    
        
        # Stack modules as a list
        self.layers = nn.ModuleList(stages)
            
    def forward(self,x):
        if self.number_of_levels == 0: # Flat structure
            for m in self.layers:
                xdc = m.forward(x)
            return xdc
        #else: # tree structure
        #    stride = self.stride
        #    nSamples = x[0].size(0)
        #    nrows = x[0].size(1)
        #    ncols = x[0].size(2)
        #    iLevel = self.number_of_levels
        #    for m in self.layers:
        #        if iLevel == self.number_of_levels:
        #            xdc = x[0]
        #        xac = x[self.number_of_levels-iLevel+1]
        #        y = m[0].forward(xac,xdc)
        #        y = m[1::].forward(y)
        #        nrows *= stride[Direction.VERTICAL]
        #        ncols *= stride[Direction.HORIZONTAL]
        #        xdc = y.reshape(nSamples,nrows,ncols,1)             
        #        iLevel -= 1
        #    return xdc.view(nSamples,1,nrows,ncols)

    """
    @property
    def T(self):
        from lsunAnalysis2dNetwork import LsunAnalysis2dNetwork
        import re

        # Create analyzer as the adjoint of SELF
        analyzer = LsunAnalysis2dNetwork(
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
        syn_state_dict = self.state_dict()
        ana_state_dict = analyzer.state_dict()
        for key in syn_state_dict.keys():
            istage_ana = int(re.sub('^layers\.|\.Lv\d_.+$','',key))            
            istage_syn = (nlevels-1)-istage_ana
            angs = syn_state_dict[key]
            ana_state_dict[key\
                .replace('layers.%d'%istage_ana,'layers.%d'%istage_syn)\
                .replace('~','')\
                .replace('T.angles','.angles') ] = angs
        
        # Load state dictionary
        analyzer.load_state_dict(ana_state_dict)

        # Return adjoint
        return analyzer.to(angs.device)
    """

