import torch
import torch.nn as nn
import math
from .lsunLayerExceptions import InvalidMode, InvalidNumberOfBlocks, InvalidStride, \
    InvalidBoundaryOperation
from .orthonormalTransform import SetOfOrthonormalTransforms

class LsunIntermediateRotation1dLayer(nn.Module):
    """
    LSUNINTERMEDIATEROTATION1DLAYER

        コンポーネント別に入力(nComponents):
            nSamples x nBlks x nChs

        コンポーネント別に出力(nComponents):
            nSamples x nBlks x nChs

        boundary_operation ('Circular' or 'Termination') corresponds to
        the MATLAB property BoundaryOperation. 'Termination' fixes the
        rotation matrix of the boundary block (the first block) to -I, so
        that the boundary block does not contribute to the gradient of the
        angles.

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
                 mode='Synthesis',
                 stride=None,
                 mus=None,
                 number_of_blocks=None,
                 name='',
                 boundary_operation='Circular'):
        super(LsunIntermediateRotation1dLayer, self).__init__()
        self.dtype = dtype
        self.device = device
        self.__mode = mode
        self.name = name

        # Stride
        if stride is None:
            raise InvalidStride("The stride must be specified.")
        self.stride = stride

        # Number of blocks
        if number_of_blocks is None:
            raise InvalidNumberOfBlocks("The number of blocks must be specified.")
        self.number_of_blocks = number_of_blocks

        # Number of channels
        ps = math.ceil(self.stride/2.0)
        pa = math.floor(self.stride/2.0)

        # Description
        self.description = self.__mode \
            + ' LSUN intermediate rotation ' \
            + '(ps,pa) = (' + str(ps) + ',' + str(pa) + ')'

        # Orthonormal matrix generation system
        if self.__mode != 'Analysis' and self.__mode != 'Synthesis':
            raise InvalidMode(f"{self.__mode}: Mode should be either of Synthesis or Analysis'")
        nblks = self.number_of_blocks
        if self.__mode == 'Synthesis':
            self.orthTransUnx = SetOfOrthonormalTransforms(name=self.name+"_UnT",nblks=nblks,n=pa,mode='Synthesis',device=self.device,dtype=self.dtype)
        else:
            self.orthTransUnx = SetOfOrthonormalTransforms(name=self.name+"_Un",nblks=nblks,n=pa,mode='Analysis',device=self.device,dtype=self.dtype)
        self.orthTransUnx.angles = nn.init.zeros_(self.orthTransUnx.angles)

        # Update parameters
        self.mus = mus

        # Boundary operation (see the MATLAB lsunIntermediateRotation1dLayer)
        self.boundary_operation = boundary_operation

    @property
    def boundary_operation(self):
        return self.__boundary_operation

    @boundary_operation.setter
    def boundary_operation(self, boundary_operation):
        if boundary_operation not in ('Circular', 'Termination'):
            raise InvalidBoundaryOperation(
                f"{boundary_operation} : BoundaryOperation should be either of Circular or Termination")
        self.__boundary_operation = boundary_operation

    def forward(self, X):
        nblks = X.size(dim=1)
        nDecs = self.stride
        ps = math.ceil(nDecs/2.0)
        pa = math.floor(nDecs/2.0)

        # Process
        # nSamples x nBlks x nChs -> nBlks x nChs x nSamples
        Y = X.permute(1,2,0)
        Zs = Y[:,:ps,:].clone()
        Za = self.orthTransUnx(Y[:,ps:,:])
        if self.__boundary_operation == 'Termination':
            # The boundary block is rotated by the fixed matrix -I
            isBoundary = torch.zeros(nblks, 1, 1, dtype=torch.bool, device=X.device)
            isBoundary[0] = True  # first block
            Za = torch.where(isBoundary, -Y[:,ps:,:], Za)
        Z = torch.cat((Zs,Za),dim=1).permute(2,0,1)
        return Z

    @property
    def mode(self):
        return self.__mode

    @property
    def angles(self):
        return self.orthTransUnx.angles

    @angles.setter
    def angles(self, angles):
        self.orthTransUnx.angles = angles

    @property
    def mus(self):
        return self.orthTransUnx.mus

    @mus.setter
    def mus(self, mus):
        nBlocks = self.number_of_blocks
        pa = math.floor(self.stride/2.0)
        if mus is None:
            mus = torch.ones(nBlocks,pa,dtype=self.dtype)
        elif isinstance(mus, int) or isinstance(mus, float):
            mus = mus*torch.ones(nBlocks,pa,dtype=self.dtype)
        self.orthTransUnx.mus = mus.to(self.device)

    def to(self, device=None, dtype=None,*args, **kwargs):
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        super(LsunIntermediateRotation1dLayer, self).to(device=self.device,dtype=self.dtype,*args, **kwargs)
        for m in (self.orthTransUnx,):
            m.to(device=self.device,dtype=self.dtype,*args, **kwargs)
        return self
