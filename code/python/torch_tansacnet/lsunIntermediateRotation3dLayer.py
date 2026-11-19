import torch
import torch.nn as nn
import math
from .lsunLayerExceptions import InvalidMode, InvalidNumberOfBlocks, InvalidStride, \
    InvalidBoundaryOperation, InvalidDirection
from .orthonormalTransform import SetOfOrthonormalTransforms

class LsunIntermediateRotation3dLayer(nn.Module):
    """
    LSUNINTERMEDIATEROTATION3DLAYER

        コンポーネント別に入力(nComponents):
            nSamples x nRows x nCols x nLays x nChs

        コンポーネント別に出力(nComponents):
            nSamples x nRows x nCols x nLays x nChs

        boundary_operation ('Circular' or 'Termination') and direction
        ('Vertical', 'Horizontal' or 'Depth') correspond to the MATLAB
        properties BoundaryOperation and Direction. 'Termination' fixes the
        rotation matrices of the boundary blocks (the first row for
        'Vertical', the first column for 'Horizontal', the first layer for
        'Depth') to -I, so that the boundary blocks do not contribute to
        the gradient of the angles.

        The angles and mus are those of the blocks in row-major order of
        nRows x nCols x nLays.

    Requirements: Python 3.10-12.x, PyTorch 2.3/4.x

    Copyright (c) 2026, Shogo MURAMATSU

    All rights reserved.

    Contact address: Shogo MURAMATSU,
        Faculty of Engineering, Niigata University,
        8050 2-no-cho Ikarashi, Nishi-ku,
        Niigata, 950-2181, JAPAN

    https://www.eng.niigata-u.ac.jp/~msiplab/
    """

    DIRECTIONS = ('Vertical', 'Horizontal', 'Depth')

    def __init__(self,
                 dtype=torch.get_default_dtype(),
                 device=torch.get_default_device(),
                 mode='Synthesis',
                 stride=None,
                 mus=None,
                 number_of_blocks=None,
                 name='',
                 boundary_operation='Circular',
                 direction=None):
        super(LsunIntermediateRotation3dLayer, self).__init__()
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
        ps = math.ceil(math.prod(self.stride)/2.0)
        pa = math.floor(math.prod(self.stride)/2.0)

        # Description
        self.description = self.__mode \
            + ' LSUN intermediate rotation ' \
            + '(ps,pa) = (' + str(ps) + ',' + str(pa) + ')'

        # Orthonormal matrix generation system
        if self.__mode != 'Analysis' and self.__mode != 'Synthesis':
            raise InvalidMode(f"{self.__mode}: Mode should be either of Synthesis or Analysis'")
        nblks = math.prod(self.number_of_blocks)
        if self.__mode == 'Synthesis':
            self.orthTransUnx = SetOfOrthonormalTransforms(name=self.name+"_UnT",nblks=nblks,n=pa,mode='Synthesis',device=self.device,dtype=self.dtype)
        else:
            self.orthTransUnx = SetOfOrthonormalTransforms(name=self.name+"_Un",nblks=nblks,n=pa,mode='Analysis',device=self.device,dtype=self.dtype)
        self.orthTransUnx.angles = nn.init.zeros_(self.orthTransUnx.angles)

        # Update parameters
        self.mus = mus

        # Boundary operation (see the MATLAB lsunIntermediateRotation3dLayer)
        self.boundary_operation = boundary_operation
        self.direction = direction
        self.__validate_direction()

    @property
    def boundary_operation(self):
        return self.__boundary_operation

    @boundary_operation.setter
    def boundary_operation(self, boundary_operation):
        if boundary_operation not in ('Circular', 'Termination'):
            raise InvalidBoundaryOperation(
                f"{boundary_operation} : BoundaryOperation should be either of Circular or Termination")
        self.__boundary_operation = boundary_operation

    @property
    def direction(self):
        return self.__direction

    @direction.setter
    def direction(self, direction):
        # None is allowed only for 'Circular'
        if direction in (None, ''):
            direction = None
        elif direction not in self.DIRECTIONS:
            raise InvalidDirection(
                f"{direction} : Direction should be either of Vertical, Horizontal or Depth")
        self.__direction = direction

    def __validate_direction(self):
        if self.__boundary_operation == 'Termination' and self.__direction is None:
            raise InvalidDirection(
                "Direction should be either of Vertical, Horizontal or Depth for Termination")

    def __boundary_block_mask(self, nrows, ncols, nlays, device):
        """(nRows x nCols x nLays) x 1 x 1 mask of the boundary blocks in row-major order."""
        self.__validate_direction()
        mask = torch.zeros(nrows, ncols, nlays, dtype=torch.bool, device=device)
        if self.__direction == 'Vertical':
            mask[0, :, :] = True  # first row
        elif self.__direction == 'Horizontal':
            mask[:, 0, :] = True  # first column
        else:
            mask[:, :, 0] = True  # first layer
        return mask.reshape(nrows*ncols*nlays, 1, 1)

    def forward(self, X):
        nSamples = X.size(dim=0)
        nrows = X.size(dim=1)
        ncols = X.size(dim=2)
        nlays = X.size(dim=3)
        nDecs = math.prod(self.stride)
        ps = math.ceil(nDecs/2.0)
        pa = math.floor(nDecs/2.0)

        # Process
        # nSamples x nRows x nCols x nLays x nChs
        #   -> (nRows x nCols x nLays) x nChs x nSamples
        Y = X.permute(1,2,3,4,0).reshape(nrows*ncols*nlays,ps+pa,nSamples)
        Zs = Y[:,:ps,:].clone()
        Za = self.orthTransUnx(Y[:,ps:,:])
        if self.__boundary_operation == 'Termination':
            # The boundary blocks are rotated by the fixed matrix -I
            isBoundary = self.__boundary_block_mask(nrows, ncols, nlays, X.device)
            Za = torch.where(isBoundary, -Y[:,ps:,:], Za)
        Z = torch.cat((Zs,Za),dim=1).reshape(nrows,ncols,nlays,ps+pa,nSamples).permute(4,0,1,2,3)
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
        nBlocks = math.prod(self.number_of_blocks)
        pa = math.floor(math.prod(self.stride)/2.0)
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
        super(LsunIntermediateRotation3dLayer, self).to(device=self.device,dtype=self.dtype,*args, **kwargs)
        for m in (self.orthTransUnx,):
            m.to(device=self.device,dtype=self.dtype,*args, **kwargs)
        return self
