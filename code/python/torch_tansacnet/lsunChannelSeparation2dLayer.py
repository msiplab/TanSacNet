#import torch
import torch.nn as nn

class LsunChannelSeparation2dLayer(nn.Module):
    """
    LSUNCHANNELSEPARATION2DLAYER

        １コンポーネント入力(nComponents=1のみサポート):
            nSamples x nRows x nCols x nChsTotal 

        ２コンポーネント出力(nComponents=2のみサポート):
            nSamples x nRows x nCols x (nChsTotal-1) 
            nSamples x nRows x nCols 
    
    Requirements: Python 3.10/11.x, PyTorch 2.3.x

    Copyright (c) 2024, Shogo MURAMATSU

    All rights reserved.

    Contact address: Shogo MURAMATSU,
                    Faculty of Engineering, Niigata University,
                    8050 2-no-cho Ikarashi, Nishi-ku,
                    Niigata, 950-2181, JAPAN

    https://www.eng.niigata-u.ac.jp/~msiplab/
    """

    def __init__(self,
        name=''):
        super(LsunChannelSeparation2dLayer, self).__init__()
        self.name = name
        self.description = "Channel separation"        
        #self.type = ''
        #self.input_names = [ 'ac', 'dc' ]

    def forward(self,X):
        return X[:,:,:,1:], X[:,:,:,0]
