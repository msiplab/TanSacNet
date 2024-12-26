import torch
import torch.nn as nn
import torch.autograd as autograd
#import torch.multiprocessing as mp
import math
#import numpy as np
from .lsunLayerExceptions import InvalidMode, InvalidMus, InvalidAngles

"""
import torch.multiprocessing as mp
def worker(layer, X_iblk, queue, iblk):
    Z_iblk = layer(X_iblk)
    queue.put((iblk, Z_iblk))
"""

class SetOfOrthonormalTransforms(nn.Module):
    """
    SETOFORTHONORMALTRANSFORMS
    
    Requirements: Python 3.10-12.x, PyTorch 2.3/4.x

    Copyright (c) 2024, Shogo MURAMATSU, Yasas GODAGE, and Takuma KUWABARA

    All rights reserved.

    Contact address: Shogo MURAMATSU,
        Faculty of Engineering, Niigata University,
        8050 2-no-cho Ikarashi, Nishi-ku,
        Niigata, 950-2181, JAPAN

        https://www.eng.niigata-u.ac.jp/~msiplab/
    """

    def __init__(self,
        n=2,
        nblks=1,
        name='SoOT',
        mode='Analysis',
        dtype=torch.get_default_dtype(),
        device=torch.get_default_device()): #device("cpu")):

        super(SetOfOrthonormalTransforms, self).__init__()
        self.dtype = dtype
        self.nPoints = n
        self.nblks = nblks
        self.device = device
        self.__name = name
                
        # Mode
        if mode in {'Analysis','Synthesis'}:
            self.__mode = mode
        else:
            raise InvalidMode(
                '%s : Mode should be either of Analysis or Synthesis'\
                % str(mode)
            )
        
        # Angles
        nAngs = int(n*(n-1)/2)
        self.__angles = torch.empty(self.nblks,nAngs,dtype=self.dtype,device=self.device)

        # Mus
        self.__mus = torch.empty(self.nblks,n,dtype=self.dtype,device=self.device)

        # OrthonormalTransforms
        self.orthonormalTransforms = OrthonormalTransform(n=self.nPoints, nblks=self.nblks,mode=self.mode,dtype=self.dtype,device=self.device)

    def forward(self,x):
        Z = torch.empty_like(x)
        Z = self.orthonormalTransforms(x)

        return Z

    @property
    def name(self):
        return self.__name
    
    @name.setter
    def name(self,name):
        if isinstance(name,str):
            self.__name = name

    @property
    def mode(self):
        return self.__mode 
    
    @mode.setter
    def mode(self,mode):
        if mode in {'Analysis','Synthesis'}:
            self.__mode = mode
        else:
            raise InvalidMode(
                '%s : Mode should be either of Analysis or Synthesis'\
                % str(mode)
            )
        
        self.orthonormalTransforms.mode = self.__mode

    @property
    def mus(self):
        self.__mus = self.orthonormalTransforms.mus
        return self.__mus
    
    @mus.setter
    def mus(self,mus):
        if torch.is_tensor(mus):
            self.__mus = mus.to(dtype=self.dtype,device=self.device)
        else:
            self.__mus = torch.tensor(mus,dtype=self.dtype,device=self.device)
        if self.__mus.size(0) != self.nblks:
            raise InvalidMus(
                '%s : The number of mus should be equal to the number of blocks'\
                % str(self.__mus)
            )
        
        self.orthonormalTransforms.mus = self.__mus

    @property
    def angles(self):
        self.__angles = self.orthonormalTransforms.angles.data
        return self.__angles

    @angles.setter
    def angles(self,angles):
        if torch.is_tensor(angles):
            self.__angles = angles.to(dtype=self.dtype,device=self.device)
        else:
            self.__angles = torch.tensor(angles,dtype=self.dtype,device=self.device)
        if self.__angles.size(0) != self.nblks:
            raise InvalidAngles(
                '%s : The number of angles should be equal to the number of blocks'\
                % str(self.__angles)
            )

        self.orthonormalTransforms.angles.data = self.__angles        

    def to(self, device=None, dtype=None,*args, **kwargs):
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        super(SetOfOrthonormalTransforms, self).to(device=self.device,dtype=self.dtype,*args, **kwargs)
        self.__angles.to(device=self.device,dtype=self.dtype,*args, **kwargs)
        self.__mus.to(device=self.device,dtype=self.dtype,*args, **kwargs)
        
        self.orthonormalTransforms.to(device=self.device, dtype=self.dtype, *args, **kwargs)
        return self

class OrthonormalTransform(nn.Module):
    """
    ORTHONORMALTRANSFORM
    
    Requirements: Python 3.10-12.x, PyTorch 2.3/4.x
    
    Copyright (c) 2021-2024, Shogo MURAMATSU
    
    All rights reserved.
    
    Contact address: Shogo MURAMATSU,
        Faculty of Engineering, Niigata University,
        8050 2-no-cho Ikarashi, Nishi-ku,
        Niigata, 950-2181, JAPAN
    
        http://www.eng.niigata-u.ac.jp/~msiplab/
    """

    def __init__(self,
        n=2,
        nblks = 1,
        name='OT',        
        mus=1,
        mode='Analysis',
        dtype=torch.get_default_dtype(),
        device=torch.get_default_device()): #device("cpu")):

        super(OrthonormalTransform, self).__init__()
        self.dtype = dtype
        self.device = device
        self.nPoints = n
        self.nblks = nblks
        self.__name = name

        # Mode
        if mode in {'Analysis','Synthesis'}:
            self.__mode = mode
        else:
            raise InvalidMode(
                '%s : Mode should be either of Analysis or Synthesis'\
                % str(mode)
            )

        # Angles
        nAngs = int(n*(n-1)/2)
        self.angles = nn.Parameter(torch.zeros(self.nblks, nAngs,dtype=self.dtype,device=self.device))

        # Mus
        if torch.is_tensor(mus):
            self.__mus = mus.to(dtype=self.dtype,device=self.device)
        elif mus == 1:
            self.__mus = torch.ones(nblks,self.nPoints,dtype=self.dtype,device=self.device)
        elif mus == -1:
            self.__mus = -torch.ones(nblks,self.nPoints,dtype=self.dtype,device=self.device)
        else:
            self.__mus = torch.tensor(mus,dtype=self.dtype,device=self.device)
        self.checkMus()

    def forward(self,X):

        angles = self.angles #.to(dtype=self.dtype,device=self.device)
        mus = self.__mus
        mode = self.__mode

        if angles.dim() == 1:
            angles = angles.unsqueeze(0)
        if mus.dim() == 1:
            mus = mus.unsqueeze(0)
        
        if mode=='Analysis':
            givensrots = GivensRotations4Analyzer.apply
        else:
            givensrots = GivensRotations4Synthesizer.apply
        return givensrots(X,angles,mus)           

    @property
    def name(self):
        return self.__name
    
    @name.setter
    def name(self,name):
        if isinstance(name,str):
            self.__name = name

    @property
    def mode(self):
        return self.__mode 

    @mode.setter
    def mode(self,mode):
        if mode in {'Analysis','Synthesis'}:
            self.__mode = mode
        else:
            raise InvalidMode(
                '%s : Mode should be either of Analysis or Synthesis'\
                % str(mode)
            )

    @property 
    def mus(self):
        return self.__mus
    
    @mus.setter
    def mus(self,mus):
        if torch.is_tensor(mus):
            self.__mus = mus.to(dtype=self.dtype,device=self.device)
        elif mus == 1:
            self.__mus = torch.ones(self.nblks, self.nPoints,dtype=self.dtype,device=self.device)
        elif mus == -1:
            self.__mus = -torch.ones(self.nblks, self.nPoints,dtype=self.dtype,device=self.device)
        else:
            self.__mus = torch.tensor(mus,dtype=self.dtype,device=self.device)
        self.checkMus()

    def checkMus(self):
        if torch.not_equal(torch.abs(self.__mus),torch.ones(self.nblks, self.nPoints,device=self.device)).any():
            raise InvalidMus(
                '%s : Elements in mus should be either of 1 or -1'\
                % str(self.__mus)
            )
        
    def to(self, device=None, dtype=None,*args, **kwargs):
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        super(OrthonormalTransform, self).to(device=self.device,dtype=self.dtype,*args, **kwargs)
        self.angles.to(device=self.device,dtype=self.dtype,*args, **kwargs)
        self.__mus.to(device=self.device,dtype=self.dtype,*args, **kwargs)
        return self
    

class GivensRotations4Analyzer(autograd.Function):
    """
    GIVENSROTATIONS4ANALYZER
    
    Requirements: Python 3.10-12.x, PyTorch 2.3/4.x
    
    Copyright (c) 2021-2024, Shogo MURAMATSU
    
    All rights reserved.
    
    Contact address: Shogo MURAMATSU,
        Faculty of Engineering, Niigata University,
        8050 2-no-cho Ikarashi, Nishi-ku,
        Niigata, 950-2181, JAPAN
    
        http://www.eng.niigata-u.ac.jp/~msiplab/
    """ 

    @staticmethod
    def forward(ctx, input, angles, mus):
        ctx.mark_non_differentiable(mus)
        R = fcn_orthonormalMatrixGeneration(angles,mus,partial_difference=False)
        ctx.save_for_backward(input,angles,mus,R)

        if input.dim() < 3:
            input = input.unsqueeze(0)

        return R @ input # TODO: Slice processing as MATLAB's PAGEFUN
    
    @staticmethod
    def backward(ctx, grad_output):
        input, angles, mus, R = ctx.saved_tensors
        grad_input = grad_angles = grad_mus = None

        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]: 
            dLdX = R.mT @ grad_output # dLdX = dZdX @ dLdZ # FIXME: #5 Userwarning on CUDA context setting
         
        if ctx.needs_input_grad[0]:
            grad_input = dLdX

        if ctx.needs_input_grad[1]:
            grad_angles = torch.zeros_like(angles,device=angles.device,requires_grad=False)
            #
            dRpre = torch.zeros_like(R) 
            dRpst = torch.eye(R.size(1),dtype=angles.dtype,device=angles.device,requires_grad=False).repeat(angles.size(0), 1, 1)
            for iAngle in range(grad_angles.size(1)):
                [dRi,dRpst,dRpre] = fcn_orthmtxgen_diff_seq(angles,mus,index_pd_angle=iAngle,matrixpre=dRpre,matrixpst=dRpst) 
                grad_angles[:,iAngle] = torch.sum((grad_output * (dRi @ input)),dim=(1,2)) # TODO: #9 Sequential processing

        if ctx.needs_input_grad[2]:
            grad_mus = torch.zeros_like(mus,device=angles.device,requires_grad=False)      

        return grad_input, grad_angles, grad_mus

class GivensRotations4Synthesizer(autograd.Function):
    """
    GIVENSROTATIONS4SYNTHESIZER
    
    Requirements: Python 3.10-12.x, PyTorch 2.3/4.x
    
    Copyright (c) 2021-2024, Shogo MURAMATSU, Yasas GODAGE, and Takuma KUWABARA
    
    All rights reserved.
    
    Contact address: Shogo MURAMATSU,
        Faculty of Engineering, Niigata University,
        8050 2-no-cho Ikarashi, Nishi-ku,
        Niigata, 950-2181, JAPAN
    
        http://www.eng.niigata-u.ac.jp/~msiplab/
    """ 

    @staticmethod
    def forward(ctx, input, angles, mus):
        ctx.mark_non_differentiable(mus)        
        R = fcn_orthonormalMatrixGeneration(angles,mus,partial_difference=False) 
        ctx.save_for_backward(input,angles,mus,R)
        
        if input.dim() < 3:
            input = input.unsqueeze(0)

        return torch.matmul(R.mT, input) # TODO: Slice processing as MATLAB's PAGEFUN
    
    @staticmethod
    def backward(ctx, grad_output):
        input, angles, mus, R = ctx.saved_tensors
        grad_input = grad_angles = grad_mus = None

        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            dLdX = R @ grad_output # dLdX = dZdX @ dLdZ
      
        if ctx.needs_input_grad[0]:
            grad_input = dLdX

        if ctx.needs_input_grad[1]:
            grad_angles = torch.zeros_like(angles,device=angles.device,requires_grad=False)
    
            # TODO: Initialize prematrix and pstmatrix for storing the state of the matrices
            dRi = torch.zeros_like(R) # TODO: Remove
            for iAngle in range(grad_angles.size(1)):
                # TODO: Modify to use prematrix and pstmatrix
                dRi = fcn_orthonormalMatrixGeneration(angles,mus,partial_difference=True,index_pd_angle=iAngle) # TODO: #8 Sequential processing
                # for iblks in range(grad_angles.size(0)):
                #     dRi[iblks] = fcn_orthonormalMatrixGeneration(angles[iblks],mus[iblks],partial_difference=True,index_pd_angle=iAngle)
                grad_angles[:,iAngle] = torch.sum((grad_output * (dRi.mT @ input)),dim=(1,2)) # TODO: #9 Sequential processing

        if ctx.needs_input_grad[2]:
            grad_mus = torch.zeros_like(mus,device=angles.device,requires_grad=False) 

        return grad_input, grad_angles, grad_mus

@torch.jit.script
def fcn_orthmtxgen_diff_seq(nDims: int, angles: torch.Tensor, index_pd_angle: int, matrixpre: torch.Tensor, matrixpst: torch.Tensor):
        
    nMatrices_ = angles.size(0)
    matrix = torch.eye(nDims,dtype=angles.dtype,device=angles.device).repeat(nMatrices_, 1, 1)
    #
    matrixrev = torch.eye(nDims,dtype=angles.dtype,device=angles.device).repeat(nMatrices_, 1, 1)   
    matrixdif = torch.zeros_like(matrix,dtype=angles.dtype,device=angles.device)
    #    
    if angles.device.type == "cuda":
        iAng = 0

        # TODO: Modify to utilize prematrix and pstmatrix to reduce the computation  
        for iTop in range(nDims-1):
            vt = matrix[:,iTop,:].unsqueeze(1)
            for iBtm in range(iTop+1,nDims):
                angle = angles[:,iAng].unsqueeze(1).unsqueeze(2)
                if iAng == index_pd_angle:
                    angle = angle + torch.pi/2. #math.pi/2.
                c = torch.cos(angle)
                s = torch.sin(angle)
                vb = matrix[:,iBtm,:].unsqueeze(1)
                #
                u = s * (vt + vb)
                vt = (c + s) * vt
                vb = (c - s) * vb
                vt = vt - u
                if iAng == index_pd_angle:
                    matrix = torch.zeros_like(matrix,dtype=angles.dtype,device=angles.device)

                matrix[:,iBtm,:] = (vb + u).squeeze()
                iAng = iAng + 1
            matrix[:,iTop,:] = vt.squeeze()

    else:
        for iMtx in range(nMatrices_):
            matrixrev = torch.eye(nDims,dtype=angles.dtype,device=angles.device)
            matrixdif = torch.zeros_like(matrix,dtype=angles.dtype,device=angles.device)
            iAng = 0
            for iTop in range(nDims-1):
                rt = matrixrev[iTop,:].clone()
                dt = matrixdif[iTop,:].clone()
                dt[iTop] = 1.
                for iBtm in range(iTop+1,nDims):
                    if iAng == index_pd_angle:
                        angle = angles[iMtx,iAng].clone()
                        #
                        rb = matrixrev[iBtm,:].clone()
                        db = matrixdif[iBtm,:].clone()
                        db[iBtm] = 1.
                        dangle = angle + torch.pi/2. 
                        #
                        vt = torch.cat((rt,dt),dim=0)
                        vb = torch.cat((rb,db),dim=0)
                        #
                        angle_ = torch.cat((-angle,dangle),dim=0).mT
                        c = torch.cos(angle_)
                        s = torch.sin(angle_)
                        #
                        u = s*(vt + vb)
                        vt = (c + s)*vt - u
                        vb = (c - s)*vb + u
                        #
                        matrixpst = matrixpst@matrixrev
                        matrix = matrixpst@matrixdif@matrixpre
                        matrixpre = matrixrev.mT@matrixpre
                    iAng = iAng + 1
                    
    return matrix

# FIXME: For multiple block case (nMatrices_ >1)
@torch.jit.script
def fcn_orthmtxgen_diff(nDims: int, angles: torch.Tensor, index_pd_angle: int):
    
    nMatrices_ = angles.size(0)
    matrix = torch.eye(nDims,dtype=angles.dtype,device=angles.device).repeat(nMatrices_, 1, 1)
    
    if angles.device.type == "cuda":
        iAng = 0

        # TODO: Modify to utilize prematrix and pstmatrix to reduce the computation  
        for iTop in range(nDims-1):
            vt = matrix[:,iTop,:].unsqueeze(1)
            for iBtm in range(iTop+1,nDims):
                angle = angles[:,iAng].unsqueeze(1).unsqueeze(2)
                if iAng == index_pd_angle:
                    angle = angle + torch.pi/2. #math.pi/2.
                c = torch.cos(angle)
                s = torch.sin(angle)
                vb = matrix[:,iBtm,:].unsqueeze(1)
                #
                u = s * (vt + vb)
                vt = (c + s) * vt
                vb = (c - s) * vb
                vt = vt - u
                if iAng == index_pd_angle:
                    matrix = torch.zeros_like(matrix,dtype=angles.dtype,device=angles.device)

                matrix[:,iBtm,:] = (vb + u).squeeze()
                iAng = iAng + 1
            matrix[:,iTop,:] = vt.squeeze()

            #torch.cuda.empty_cache()

        #print(f"matrix size: {matrix.size()}")
    else:
        # TODO: Modify to utilize prematrix and pstmatrix to reduce the computation
        for iMtx in range(nMatrices_):
            iAng = 0

            for iTop in range(nDims-1):
                vt = matrix[iMtx,iTop,:]
                for iBtm in range(iTop+1,nDims):
                    angle = angles[iMtx,iAng]
                    if iAng == index_pd_angle:
                        angle = angle + torch.pi/2. #math.pi/2.
                    c = torch.cos(angle)
                    s = torch.sin(angle)
                    vb = matrix[iMtx,iBtm,:]
                    #
                    u = s*(vt + vb)
                    vt = (c + s)*vt
                    vb = (c - s)*vb
                    vt = vt - u
                    if iAng == index_pd_angle:
                        matrix[iMtx] = torch.zeros(nDims,nDims,dtype=angles.dtype,device=angles.device)
                    matrix[iMtx,iBtm,:] = vb + u
                    iAng = iAng + 1
                matrix[iMtx,iTop,:] = vt
  
    return matrix
    
@torch.jit.script
def fcn_orthmtxgen(nDims: int, angles: torch.Tensor):
    nMatrices_ = angles.size(0)
    matrix = torch.eye(nDims,dtype=angles.dtype,device=angles.device).repeat(nMatrices_, 1, 1)
    
    if angles.device.type == "cuda":
        iAng = 0

        for iTop in range(nDims-1):
            vt = matrix[:,iTop,:].unsqueeze(1)
            for iBtm in range(iTop+1,nDims):
                angle = angles[:,iAng].unsqueeze(1).unsqueeze(2)
                c = torch.cos(angle)
                s = torch.sin(angle)
                vb = matrix[:,iBtm,:].unsqueeze(1)
                #
                u = s * (vt + vb)
                vt = (c + s) * vt
                vb = (c - s) * vb
                vt = vt - u

                matrix[:,iBtm,:] = (vb + u).squeeze()
                iAng = iAng + 1
            matrix[:,iTop,:] = vt.squeeze()

            #torch.cuda.empty_cache()

    else:
        for iMtx in range(nMatrices_):
            iAng = 0

            for iTop in range(nDims-1):
                vt = matrix[iMtx,iTop,:]
                for iBtm in range(iTop+1,nDims):
                    angle = angles[iMtx,iAng]
                    c = torch.cos(angle)
                    s = torch.sin(angle)
                    vb = matrix[iMtx,iBtm,:]
                    #
                    u = s*(vt + vb)
                    vt = (c + s)*vt
                    vb = (c - s)*vb
                    vt = vt - u
                    matrix[iMtx,iBtm,:] = vb + u
                    iAng = iAng + 1
                matrix[iMtx,iTop,:] = vt

    return matrix

# TODO: Define new function for partial difference w/ prematrix and pstmatrix
def fcn_orthonormalMatrixGeneration(
    angles: torch.Tensor,
    mus: torch.Tensor,
    partial_difference=False,
    index_pd_angle=None):
    """
    The output is set on the same device with the angles
    """

    # Number of angles
    if(angles.dim() == 1):
        angles = angles.unsqueeze(0)
    if mus.dim() == 1:
        mus = mus.unsqueeze(0)
    nAngles = angles.size(1)

    # Number of dimensions
    nDims = int((1+math.sqrt(1+8*nAngles))/2)

    # Setup of mus, which is send to the same device with angles
    mus = mus.to(device=angles.device,dtype=angles.dtype) 

    if partial_difference:
        matrix = fcn_orthmtxgen_diff(nDims, angles, index_pd_angle)            
    else:
        matrix = fcn_orthmtxgen(nDims, angles)            

    matrix = mus.unsqueeze(-1) * matrix
    return matrix 
