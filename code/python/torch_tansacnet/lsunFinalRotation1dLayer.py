import torch
import torch.nn as nn
import math
from .lsunLayerExceptions import InvalidNumberOfBlocks, InvalidStride
from .orthonormalTransform import SetOfOrthonormalTransforms

class LsunFinalRotation1dLayer(nn.Module):
    """
    LSUNFINALROTATION1DLAYER

        コンポーネント別に入力(nComponents):
            nSamples x nBlks x nChs

        コンポーネント別に出力(nComponents):
            nSamples x nBlks x nChs

        stride and number_of_blocks are integers.

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
                 dtype=torch.get_default_dtype(),
                 device=torch.get_default_device(),
                 stride=None,
                 mus=None,
                 number_of_blocks=None,
                 no_dc_leakage=False,
                 name=''):
        super(LsunFinalRotation1dLayer, self).__init__()
        self.dtype = dtype
        self.device = device
        self.name = name

        # Stride
        if stride is None:
            raise InvalidStride("The stride must be specified.")
        self.stride = stride

        # Number of blocks
        if number_of_blocks is None:
            raise InvalidNumberOfBlocks("The number of blocks must be specified.")
        self.number_of_blocks = number_of_blocks
        self.__no_dc_leakage = no_dc_leakage

        # Number of channels
        ps = math.ceil(self.stride/2.0)
        pa = math.floor(self.stride/2.0)

        # Description
        self.description = "LSUN final rotation " \
                + "(ps,pa) = (" \
                + str(ps) + "," \
                + str(pa) + "), "  \
                + "m = " + str(self.stride)

        # Orthonormal matrix generation systems
        nblks = self.number_of_blocks
        self.orthTransW0T = SetOfOrthonormalTransforms(name=self.name+"_W0T",nblks=nblks,n=ps,mode='Synthesis',device=self.device,dtype=self.dtype)
        self.orthTransU0T = SetOfOrthonormalTransforms(name=self.name+"_U0T",nblks=nblks,n=pa,mode='Synthesis',device=self.device,dtype=self.dtype)
        self.orthTransW0T.angles = nn.init.zeros_(self.orthTransW0T.angles)
        self.orthTransU0T.angles = nn.init.zeros_(self.orthTransU0T.angles)

        # Update parameters
        self.mus = mus

    def forward(self,X):
        nblks = X.size(dim=1)
        nDecs = self.stride
        ps = math.ceil(nDecs/2.0)
        pa = math.floor(nDecs/2.0)

        if self.__no_dc_leakage:
            # Avoid in-place modification of mus, which is saved for backward
            if not torch.all(self.orthTransW0T.mus[:,0] == 1.0):
                mus_ = self.orthTransW0T.mus.clone()
                mus_[:,0] = 1.0
                self.orthTransW0T.mus = mus_
            #
            with torch.no_grad():
                angles_ = self.orthTransW0T.angles
                angles_[:,:(ps-1)] = 0.0
                self.orthTransW0T.angles = angles_

        # Process
        # nSamples x nBlks x nChs -> nBlks x nChs x nSamples
        Y = X.permute(1,2,0)
        Zs = self.orthTransW0T(Y[:,:ps,:])
        Za = self.orthTransU0T(Y[:,ps:,:])
        Z = torch.cat((Zs,Za),dim=1).permute(2,0,1)
        return Z

    @property
    def no_dc_leakage(self):
        return self.__no_dc_leakage

    @property
    def angles(self):
        return torch.cat((self.orthTransW0T.angles,self.orthTransU0T.angles),dim=1)

    @angles.setter
    def angles(self, angles):
        ps = math.ceil(self.stride/2.0)
        nAnglesW = ps*(ps-1)//2
        with torch.no_grad():
            self.orthTransW0T.angles = angles[:,:nAnglesW]
            self.orthTransU0T.angles = angles[:,nAnglesW:]

    @property
    def mus(self):
        return torch.cat((self.orthTransW0T.mus,self.orthTransU0T.mus),dim=1)

    @mus.setter
    def mus(self, mus):
        nBlocks = self.number_of_blocks
        nDecs = self.stride
        if mus is None:
            mus = torch.ones(nBlocks,nDecs,dtype=self.dtype)
        elif isinstance(mus, int) or isinstance(mus, float):
            mus = mus*torch.ones(nBlocks,nDecs,dtype=self.dtype)
        ps = math.ceil(nDecs/2.0)
        self.orthTransW0T.mus = mus[:,:ps]
        self.orthTransU0T.mus = mus[:,ps:]

    def to(self, device=None, dtype=None,*args, **kwargs):
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        super(LsunFinalRotation1dLayer, self).to(device=self.device,dtype=self.dtype,*args, **kwargs)
        for m in (self.orthTransW0T, self.orthTransU0T):
            m.to(device=self.device,dtype=self.dtype,*args, **kwargs)
        return self
