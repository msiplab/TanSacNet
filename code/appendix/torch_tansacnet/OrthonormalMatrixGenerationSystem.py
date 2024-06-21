import torch
import math

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

class OrthonormalMatrixGenerationSystem:
    """
    ORTHONORMALMATRIXGENERATIONSYSTEM Orthonormal matrix generator
    
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
                 dtype=torch.get_default_dtype(),
                 partial_difference=False,
                 mode='normal'):
        super(OrthonormalMatrixGenerationSystem, self).__init__()
        self.dtype = dtype
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
            angles = torch.zeros(1,1,dtype=self.dtype)
            angles[0,0] = angle_
        elif not torch.is_tensor(angles):
            angles = torch.tensor(angles,dtype=self.dtype)
        else:
            angles = angles.to(dtype=self.dtype) 
        nAngles = angles.size(1)
        nMatrices = angles.size(0)

        # Setup at the first call
        if not hasattr(self, 'number_of_dimensions'):
            self.number_of_dimensions = int(1+math.sqrt(1+8*nAngles)/2)
        if self.mode == 'sequential' and not hasattr(self, 'nextangle'):
            self.nextangle = 0
        
        # Number of dimensions
        nDims = self.number_of_dimensions

        # Setup of mus, which is send to the same device with angles
        if isinstance(mus, int) or isinstance(mus, float):
            mu_ = mus
            mus = torch.ones(nMatrices,nDims,dtype=self.dtype,device=angles.device)
            mus = mu_ * mus
        elif not torch.is_tensor(mus): #isinstance(mus, list):
            mus = torch.tensor(mus,dtype=self.dtype,device=angles.device)
        else:
            mus = mus.to(dtype=self.dtype,device=angles.device) 
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
        matrix = torch.tile(torch.eye(nDims,dtype=self.dtype,device=angles.device), (nMatrices,1, 1))
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
        matrix = torch.tile(torch.eye(nDims,dtype=self.dtype,device=angles.device), (nMatrices,1, 1))
        if index_pd_angle == None: # Initialization 
            self.matrixpst = torch.tile(torch.eye(nDims,dtype=self.dtype,device=angles.device), (nMatrices,1, 1))
            self.matrixpre = torch.tile(torch.eye(nDims,dtype=self.dtype,device=angles.device), (nMatrices,1, 1))
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
                matrixrev = torch.eye(nDims,dtype=self.dtype,device=angles.device)
                matrixdif = torch.zeros(nDims,nDims,dtype=self.dtype,device=angles.device)
                iAng = 0
                matrixpst_iMtx = self.matrixpst[iMtx]
                matrixpre_iMtx = self.matrixpre[iMtx]
                angles_iMtx = angles[iMtx]
                mus_iMtx = mus[iMtx]
                for iTop in range(nDims - 1):
                    rt = matrixrev[iTop, :]
                    dt = torch.zeros(nDims,dtype=self.dtype,device=angles.device)
                    dt[iTop] = 1
                    for iBtm in range(iTop + 1, nDims):
                        if iAng == index_pd_angle:
                            angle = angles_iMtx[iAng]
                            rb = matrixrev[iBtm, :]
                            rt, rb = rot_(rt, rb, -angle)
                            matrixrev[iTop, :] = rt
                            matrixrev[iBtm, :] = rb
                            db = torch.zeros(nDims,dtype=self.dtype,device=angles.device)
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