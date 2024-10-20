import torch
import torch.nn as nn
import math

class Direction:
    VERTICAL = 0
    HORIZONTAL = 1
    DEPTH = 2

def rot_(vt, vb, angle):
    """ Planar rotation """
    c = torch.cos(angle)
    s = torch.sin(angle)
    u = s * (vt + vb)
    vt = (c + s) * vt
    vb = (c - s) * vb
    vt = vt - u
    vb = vb + u
    return vt, vb

class ForwardTruncationLayer(nn.Module):
    """
    FORWARDTRUNCATIONLAYER Forward truncation layer
    
    Requirements: Python 3.10-12.x, PyTorch 2.3/4.x
    """
    def __init__(self,
                 number_of_channels=1,
                 stride=[2,2],
                 nlevels=0):
        super(ForwardTruncationLayer, self).__init__()

        self.number_of_channels = number_of_channels
        self.stride = stride
        self.nlevels = nlevels

    def forward(self, X):
        #if self.nlevels == 0:
        Z = X[:,:,:,:self.number_of_channels]
        """
        else:
            nDecs = self.stride[Direction.VERTICAL]*self.stride[Direction.HORIZONTAL]
            target_stage = (self.number_of_channels-1)//(nDecs-1) 
            Z = []
            Z.append(X[0][:,:,:,:self.number_of_channels])
            if target_stage >= 0:
                number_of_channels_at_target_stage = self.number_of_channels - nDecs - (target_stage-1)*(nDecs-1)
                for istage in range(target_stage):
                    Z.append(X[istage])
                Z.append(X[target_stage][:,:,:,:number_of_channels_at_target_stage])
            Z = tuple(Z)
        """
        return Z

class AdjointTruncationLayer(nn.Module):
    """
    ADJOINTTRUNCATIONLAYER Adjoint truncation layer
    
    Requirements: Python 3.10/11.x, PyTorch 2.3.x
    """
    def __init__(self,
                 number_of_channels=1,
                 stride=[2,2],
                 nlevels=0):
        super(AdjointTruncationLayer, self).__init__()

        self.number_of_channels = number_of_channels
        self.stride = stride
        self.nlevels = nlevels

    def forward(self, X):
        nsamples = X.size(0)
        nrows = X.size(1)
        ncols = X.size(2)
        nDecs = self.stride[Direction.VERTICAL]*self.stride[Direction.HORIZONTAL]
        if self.number_of_channels == nDecs:
            Z = X
        else:
            Z = torch.cat((X,torch.zeros(nsamples,nrows,ncols,nDecs-self.number_of_channels,dtype=X.dtype,device=X.device)),dim=-1)

        return Z

class OrthonormalMatrixGenerationSystem:
    """
    ORTHONORMALMATRIXGENERATIONSYSTEM Orthonormal matrix generator
    
    Requirements: Python 3.10-12.x, PyTorch 2.3/4.x
    
    Copyright (c) 2024, Shogo MURAMATSU
    
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
                 partial_difference=False,
                 mode='normal'):
        super(OrthonormalMatrixGenerationSystem, self).__init__()
        self.dtype = dtype
        self.device = device
        self.partial_difference = partial_difference
        self.mode = mode

    def __call__(self,
                 angles=0, 
                 mus=1, 
                 index_pd_angle=None):
        """
        The output is set on the same device with the angles
        """

        # Number of angles
        if isinstance(angles,list) and len(angles) == 0:
            if isinstance(mus, int) or isinstance(mus, float):
                nMatrices_ = 1
                nDims_ = 1
            elif not torch.is_tensor(mus):
                nMatrices_ = len(mus)
                nDims_ = len(mus[0])
            else:
                nMatrices_ = mus.size(0)
                nDims_ = mus.size(1)
            nAngles_ = int(nDims_*(nDims_-1)/2)
            angles = torch.zeros(nMatrices_,nAngles_)   
        if isinstance(angles, int) or isinstance(angles, float):
            angle_ = angles
            angles = torch.zeros(1,1,dtype=self.dtype,device=self.device)
            angles[0,0] = angle_
        elif not torch.is_tensor(angles):
            angles = torch.tensor(angles,dtype=self.dtype,device=self.device)
        else:
            angles = angles.to(dtype=self.dtype,device=self.device) 
        nAngles = angles.size(1)
        nMatrices = angles.size(0)

        # Setup at the first call
        if not hasattr(self, 'number_of_dimensions'):
            self.number_of_dimensions = int(1+math.sqrt(1+8*nAngles)/2)
        if self.mode == 'sequential' and not hasattr(self, 'nextangle'):
            self.nextangle = 0
        
        # Number of dimensions
        nDims = self.number_of_dimensions

        # Setup of mus
        if isinstance(mus, int) or isinstance(mus, float):
            mu_ = mus
            mus = torch.ones(nMatrices,nDims,dtype=self.dtype,device=self.device)
            mus = mu_ * mus
        elif not torch.is_tensor(mus): #isinstance(mus, list):
            mus = torch.tensor(mus,dtype=self.dtype,device=self.device)
        else:
            mus = mus.to(dtype=self.dtype,device=self.device) 
        if mus.size(0) != nMatrices:
            if mus.size(0) == 1:
                mus = torch.tile(mus, (nMatrices,1))
            else:
                raise ValueError("The number of matrices must be the same between angles and mus.")
        elif mus.size(1) != nDims:
            if mus.size(1) == 1:
                mus = torch.tile(mus, (1,nDims))
            else:
                raise ValueError("The number of dimensions must be the same between angles and mus.")
        
        # Switch mode 
        if self.partial_difference and self.mode == 'normal':
            matrix = self.stepNormal_(angles, mus, index_pd_angle)
        elif self.partial_difference and self.mode == 'sequential':
            matrix = self.stepSequential_(angles, mus, index_pd_angle)
        else:
            matrix = self.stepNormal_(angles, mus, index_pd_angle=None)

        return matrix.clone()
    
    def stepNormal_(self, angles, mus, index_pd_angle):
        is_pd = self.partial_difference
        nDims = self.number_of_dimensions
        nMatrices = angles.size(0)
        matrix = torch.tile(torch.eye(nDims,dtype=self.dtype,device=self.device), (nMatrices,1, 1))
        # 
        for iMtx in range(nMatrices):
            iAng = 0
            matrix_iMtx = matrix[iMtx]
            angles_iMtx = angles[iMtx]
            mus_iMtx = mus[iMtx]
            for iTop in range(nDims - 1):
                vt = matrix_iMtx[iTop, :]
                for iBtm in range(iTop + 1, nDims):
                    angle = angles_iMtx[iAng]
                    if is_pd and iAng == index_pd_angle:
                        angle = angle + torch.pi / 2.0
                    vb = matrix_iMtx[iBtm, :]
                    vt, vb = rot_(vt, vb, angle)
                    if is_pd and iAng == index_pd_angle:
                        matrix_iMtx = torch.zeros_like(matrix_iMtx)
                    matrix_iMtx[iBtm, :] = vb
                    iAng += 1
                matrix_iMtx[iTop, :] = vt
            matrix[iMtx] = mus_iMtx.view(-1,1) * matrix_iMtx

        return matrix

    def stepSequential_(self, angles, mus, index_pd_angle):
        # Check index_pd_angle
        if index_pd_angle != None and index_pd_angle != self.nextangle:
            raise ValueError("Unable to proceed sequential differentiation. Index = %d is expected, but %d was given." % (self.nextangle, index_pd_angle))
        #
        nDims = self.number_of_dimensions
        nMatrices = angles.size(0)
        matrix = torch.tile(torch.eye(nDims,dtype=self.dtype,device=self.device), (nMatrices,1, 1))
        if index_pd_angle == None: # Initialization 
            self.matrixpst = torch.tile(torch.eye(nDims,dtype=self.dtype,device=self.device), (nMatrices,1, 1))
            self.matrixpre = torch.tile(torch.eye(nDims,dtype=self.dtype,device=self.device), (nMatrices,1, 1))
            for iMtx in range(nMatrices):
                iAng = 0
                matrixpst_iMtx = self.matrixpst[iMtx]
                angles_iMtx = angles[iMtx]
                mus_iMtx = mus[iMtx]
                for iTop in range(nDims - 1):
                    vt = matrixpst_iMtx[iTop, :]
                    for iBtm in range(iTop + 1, nDims):
                        angle = angles_iMtx[iAng]
                        vb = matrixpst_iMtx[iBtm, :]
                        vt, vb = rot_(vt, vb, angle)
                        matrixpst_iMtx[iBtm, :] = vb
                        iAng += 1
                    matrixpst_iMtx[iTop, :] = vt
                matrix[iMtx] = mus_iMtx.view(-1,1) * matrixpst_iMtx
                self.matrixpst[iMtx] = matrixpst_iMtx
            self.nextangle = 0
        else: # Sequential differentiation
            for iMtx in range(nMatrices):
                matrixrev = torch.eye(nDims,dtype=self.dtype,device=self.device)
                matrixdif = torch.zeros(nDims,nDims,dtype=self.dtype,device=self.device)
                iAng = 0
                matrixpst_iMtx = self.matrixpst[iMtx]
                matrixpre_iMtx = self.matrixpre[iMtx]
                angles_iMtx = angles[iMtx]
                mus_iMtx = mus[iMtx]
                for iTop in range(nDims - 1):
                    rt = matrixrev[iTop, :]
                    dt = torch.zeros(nDims,dtype=self.dtype,device=self.device)
                    dt[iTop] = 1
                    for iBtm in range(iTop + 1, nDims):
                        if iAng == index_pd_angle:
                            angle = angles_iMtx[iAng]
                            rb = matrixrev[iBtm, :]
                            rt, rb = rot_(rt, rb, -angle)
                            matrixrev[iTop, :] = rt
                            matrixrev[iBtm, :] = rb
                            db = torch.zeros(nDims,dtype=self.dtype,device=self.device)
                            db[iBtm] = 1
                            dangle = angle + torch.pi / 2.0
                            dt, db = rot_(dt, db, dangle)
                            matrixdif[iTop, :] = dt
                            matrixdif[iBtm, :] = db
                            matrixpst_iMtx = torch.matmul(matrixpst_iMtx, matrixrev)
                            matrix[iMtx] = torch.matmul(torch.matmul(matrixpst_iMtx, matrixdif), matrixpre_iMtx)
                            matrixpre_iMtx = torch.matmul(torch.transpose(matrixrev,0,1), matrixpre_iMtx)
                        iAng += 1
                matrix[iMtx] = mus_iMtx.view(-1,1) * matrix[iMtx]
                self.matrixpre[iMtx] = matrixpre_iMtx
                self.matrixpst[iMtx] = matrixpst_iMtx
            self.nextangle += 1

        return matrix
    
    def to(self, dtype=None, device=None):
        if dtype is not None:
            self.dtype = dtype
        if device is not None:
            self.device = device
        self.angle = self.angle.to(dtype=self.dtype,device=self.device)
        self.mus = self.mus.to(dtype=self.dtype,device=self.device)
        return self

def permuteDctCoefs(x):
    cee = x[:,0::2,0::2].reshape(x.size(0),-1)
    coo = x[:,1::2,1::2].reshape(x.size(0),-1)
    coe = x[:,1::2,0::2].reshape(x.size(0),-1)
    ceo = x[:,0::2,1::2].reshape(x.size(0),-1)
    return torch.cat((cee,coo,coe,ceo),dim=-1)

def permuteIdctCoefs(x,block_size):
    #coefs = x.view(-1,block_size[Direction.VERTICAL]*block_size[Direction.HORIZONTAL]) # x.view(-1,math.prod(block_size)) 
    coefs = x.reshape(-1,block_size[Direction.VERTICAL]*block_size[Direction.HORIZONTAL]) # x.view(-1,math.prod(block_size))     
    decY_ = block_size[Direction.VERTICAL]
    decX_ = block_size[Direction.HORIZONTAL]
    chDecY = int(math.ceil(decY_/2.)) #.astype(int)
    chDecX = int(math.ceil(decX_/2.)) #.astype(int)
    fhDecY = int(math.floor(decY_/2.)) #.astype(int)
    fhDecX = int(math.floor(decX_/2.)) #.astype(int)
    nQDecsee = chDecY*chDecX
    nQDecsoo = fhDecY*fhDecX
    nQDecsoe = fhDecY*chDecX
    cee = coefs[:,:nQDecsee]
    coo = coefs[:,nQDecsee:nQDecsee+nQDecsoo]
    coe = coefs[:,nQDecsee+nQDecsoo:nQDecsee+nQDecsoo+nQDecsoe]
    ceo = coefs[:,nQDecsee+nQDecsoo+nQDecsoe:]
    nBlocks = coefs.size(0)
    value = torch.zeros(nBlocks,decY_,decX_,dtype=x.dtype,device=x.device)
    value[:,0::2,0::2] = cee.view(nBlocks,chDecY,chDecX)
    value[:,1::2,1::2] = coo.view(nBlocks,fhDecY,fhDecX)
    value[:,1::2,0::2] = coe.view(nBlocks,fhDecY,chDecX)
    value[:,0::2,1::2] = ceo.view(nBlocks,chDecY,fhDecX)
    return value

"""
def block_butterfly(X,nchs):
    ps = nchs[0]
    Xs = X[:,:,:,:ps]
    Xa = X[:,:,:,ps:]
    return torch.cat((Xs+Xa,Xs-Xa),dim=-1)

def block_shift(X,nchs,target,shift):
    ps = nchs[0]
    if target == 0: # Difference channel
        X[:,:,:,ps:] = torch.roll(X[:,:,:,ps:],shifts=tuple(shift),dims=(0,1,2,3))
    else: # Sum channel
        X[:,:,:,:ps] = torch.roll(X[:,:,:,:ps],shifts=tuple(shift),dims=(0,1,2,3))
    return X

def intermediate_rotation(X,nchs,R):
    Y = X.clone()
    ps,pa = nchs
    nSamples = X.size(0)
    nrows = X.size(1)
    ncols = X.size(2)
    Za = R @ X[:,:,:,ps:].view(-1,pa).T 
    Y[:,:,:,ps:] = Za.T.view(nSamples,nrows,ncols,pa)
    return Y
"""