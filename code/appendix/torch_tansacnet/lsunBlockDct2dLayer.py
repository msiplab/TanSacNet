import torch
import torch.nn as nn
import torch_dct as dct
import math
from .lsunUtility import Direction, permuteDctCoefs
    
class LsunBlockDct2dLayer(nn.Module):
    """
    LSUNBLOCKDCT2DLAYER
    
       ベクトル配列をブロック配列を入力:
          nSamples x nComponents x (Stride[0]xnRows) x (Stride[1]xnCols) 
    
       コンポーネント別に出力(nComponents):
          nSamples x nRows x nCols x nDecs
        
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
        name='',
        stride=[],
        number_of_components=1
        ):
        super(LsunBlockDct2dLayer, self).__init__()
        self.stride = stride
        self.name = name
        self.description = "Block DCT of size " \
            + str(self.stride[Direction.VERTICAL]) + "x" \
            + str(self.stride[Direction.HORIZONTAL])
        #self.type = ''
        self.num_outputs = number_of_components
        #self.num_inputs = 1

    def forward(self,X):
        nComponents = self.num_outputs
        nSamples = X.size(0)
        height = X.size(2)
        width = X.size(3)
        stride = self.stride        
        nrows = int(math.ceil(height/stride[Direction.VERTICAL]))
        ncols = int(math.ceil(width/stride[Direction.HORIZONTAL]))
        ndecs = stride[Direction.VERTICAL]*stride[Direction.HORIZONTAL] #math.prod(stride)
        # Block DCT (nSamples x nComponents x nrows x ncols) x decV x decH
        arrayshape = list(stride).copy()
        arrayshape.insert(0,-1)
        Y = dct.dct_2d(X.reshape(arrayshape),norm='ortho')
        # Rearrange the DCT Coefs. (nSamples x nComponents x nrows x ncols) x (decV x decH)
        #cee = Y[:,0::2,0::2].reshape(Y.size(0),-1)
        #coo = Y[:,1::2,1::2].reshape(Y.size(0),-1)
        #coe = Y[:,1::2,0::2].reshape(Y.size(0),-1)
        #ceo = Y[:,0::2,1::2].reshape(Y.size(0),-1)
        #A = torch.cat((cee,coo,coe,ceo),dim=-1)
        A = permuteDctCoefs(Y)
        Z = A.view(nSamples,nComponents,nrows,ncols,ndecs) 

        if nComponents<2:
            return torch.squeeze(Z,dim=1)
        else:
            return map(lambda x: torch.squeeze(x,dim=1),torch.chunk(Z,nComponents,dim=1))