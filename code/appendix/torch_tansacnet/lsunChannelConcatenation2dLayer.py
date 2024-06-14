import torch
import torch.nn as nn

class LsunChannelConcatenation2dLayer(nn.Module):
    """
    LSUNCHANNELCONCATENATION2DLAYER
    
       ２コンポーネント入力(nComponents=2のみサポート):
          nSamples x nRows x nCols x (nChsTotal-1) 
          nSamples x nRows x nCols
    
       １コンポーネント出力(nComponents=1のみサポート):
          nSamples x nRows x nCols x nChsTotal
    
     Requirements: Python 3.10/11.x, PyTorch 2.3.x
    
     Copyright (c) 2024, Shogo MURAMATSU
    
     All rights reserved.
    
     Contact address: Shogo MURAMATSU,
                    Faculty of Engineering, Niigata University,
                    8050 2-no-cho Ikarashi, Nishi-ku,
                    Niigata, 950-2181, JAPAN
    
    http://www.eng.niigata-u.ac.jp/~msiplab/
    """

    def __init__(self,
        name=''):
        super(LsunChannelConcatenation2dLayer, self).__init__()
        self.name = name
        self.description = "Channel concatenation"

    def forward(self,Xac,Xdc):
        """
        Forward input data through the layer at prediction time and
        output the result.
            
            Inputs:
                layer       - Layer to forward propagate through
                Xac, Xdc    - Input data (2 components)
            Outputs:
                Z           - Outputs of layer forward function
        """

        # Layer forward function for prediction goes here.
        if Xdc.dim() == 3:
            Xdc_ = Xdc.unsqueeze(dim=3)
        else:
            Xdc_ = Xdc
        if Xac.dim() == 3:
            Xac_ = Xac.unsqueeze(dim=3)
        else:
            Xac_ = Xac
        return torch.cat((Xdc_,Xac_),dim=3)
