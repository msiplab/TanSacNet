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
                 nlevels=0,
                 mode='normal'):
        super(ForwardTruncationLayer, self).__init__()

        self.number_of_channels = number_of_channels
        self.stride = stride
        self.nlevels = nlevels
        self.mode = mode

        if self.mode not in ['normal', 'interleave']:
            raise ValueError("mode must be either 'normal' or 'interleave'")

        if self.mode == 'interleave' and self.nlevels != 0:
            raise ValueError("mode = 'interleave' is only supported when self.nlevels = 0")

    def forward(self, X):

        if self.nlevels == 0:
            if self.mode == 'interleave':
                nDecs = self.stride[Direction.VERTICAL]*self.stride[Direction.HORIZONTAL]
                indices = torch.arange(nDecs).view(2,nDecs//2).T.flatten()
                X = X[:,:,:,indices]
            Z = X[:,:,:,:self.number_of_channels]

        else: # TODO: INTERLEAVE mode
            nDecs = self.stride[Direction.VERTICAL]*self.stride[Direction.HORIZONTAL]
            if self.number_of_channels == 1:
                Z = X
            else:
                target_stage = (self.number_of_channels-1)//(nDecs-1) + 1
                Z = []
                number_of_channels_at_target_stage = (self.number_of_channels - 1) % (nDecs-1) 
                for istage in range(target_stage):
                    Z.append(X[istage])
                Z.append(X[target_stage][:,:,:,:number_of_channels_at_target_stage])
            Z = tuple(Z)
        return Z

    @property
    def T(self): # TODO: Unit test
        from . import AdjointTruncationLayer
        return AdjointTruncationLayer(
            stride=self.stride,
            nlevels=self.nlevels,
            mode=self.mode
        )

class AdjointTruncationLayer(nn.Module):
    """
    ADJOINTTRUNCATIONLAYER Adjoint truncation layer
    
    Requirements: Python 3.10/11.x, PyTorch 2.3.x
    """
    def __init__(self,
                 stride=[2,2],
                 nlevels=0,
                 mode='normal'):
        super(AdjointTruncationLayer, self).__init__()

        self.stride = stride
        self.nlevels = nlevels
        self.mode = mode

        if self.mode not in ['normal', 'interleave']:
            raise ValueError("mode must be either 'normal' or 'interleave'")

        if self.mode == 'interleave' and self.nlevels != 0:
            raise ValueError("mode = 'interleave' is only supported when self.nlevels = 0")

    def forward(self, X):
        nDecs = self.stride[Direction.VERTICAL]*self.stride[Direction.HORIZONTAL]
        if self.nlevels == 0:
            nsamples = X.size(0)
            nrows = X.size(1)
            ncols = X.size(2)
            number_of_channels = X.size(3)
            Z = torch.zeros(nsamples,nrows,ncols,nDecs,dtype=X.dtype,device=X.device,requires_grad=X.requires_grad)
            Z[:,:,:,:number_of_channels] = X
            if self.mode == 'interleave':
                indices = torch.arange(nDecs).view(-1,2).T.flatten()
                Z = Z[:,:,:,indices]

        else: # TODO: INTERLEAVE mode
            nstages = len(X)
            Z = []
            for istage in range(self.nlevels+1):
                if istage == 0 or istage < nstages-1:
                    X_i = X[istage]
                    nsamples = X_i.size(0)
                    nrows_ = X_i.size(1)
                    ncols_ = X_i.size(2)
                    Z.append(X_i) 
                elif istage == nstages-1:
                    X_i = X[istage]
                    nsamples = X_i.size(0)
                    nrows_ = X_i.size(1)
                    ncols_ = X_i.size(2)
                    number_of_channels = X_i.size(3)
                    if number_of_channels < nDecs -1:
                        Z.append(torch.cat((X_i,torch.zeros(nsamples,nrows_,ncols_,nDecs-1-number_of_channels,dtype=X_i.dtype,device=X_i.device,requires_grad=X_i.requires_grad)),dim=-1))
                    else:
                        Z.append(X_i)
                else:
                    if istage > 1:
                        nrows_ *= self.stride[Direction.VERTICAL]
                        ncols_ *= self.stride[Direction.HORIZONTAL]
                    Z.append(torch.zeros(nsamples,nrows_,ncols_,nDecs-1,dtype=X_i.dtype,device=X_i.device,requires_grad=X_i.requires_grad))                        
            Z = tuple(Z)
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

def dctmtx(n,dtype=None,device=None):
    """
    Orthonormal DCT-II matrix, equivalent to MATLAB dctmtx(n)
    """
    k = torch.arange(n,dtype=dtype,device=device).view(-1,1)
    i = torch.arange(n,dtype=dtype,device=device).view(1,-1)
    C = math.sqrt(2./n)*torch.cos(math.pi*(2*i+1)*k/(2*n))
    C[0,:] = C[0,:]/math.sqrt(2.)
    return C

def block_dct_matrix_2d(stride,dtype=None,device=None):
    """
    2-D block DCT matrix, equivalent to Cvh in MATLAB lsunBlockDct2dLayer

    Rows are ordered as [ Cee; Coo; Coe; Ceo ] and columns correspond to
    the pixels of a decV x decH block in column-major order (v + decV*h).
    """
    decV = stride[Direction.VERTICAL]
    decH = stride[Direction.HORIZONTAL]
    Cv = dctmtx(decV,dtype=dtype,device=device)
    Ch = dctmtx(decH,dtype=dtype,device=device)
    Cve, Cvo = Cv[0::2,:], Cv[1::2,:]
    Che, Cho = Ch[0::2,:], Ch[1::2,:]
    Cee = torch.kron(Che,Cve)
    Coo = torch.kron(Cho,Cvo)
    Coe = torch.kron(Che,Cvo)
    Ceo = torch.kron(Cho,Cve)
    return torch.cat((Cee,Coo,Coe,Ceo),dim=0)

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