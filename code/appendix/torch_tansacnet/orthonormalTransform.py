import torch
import torch.nn as nn
import torch.autograd as autograd
import math
#import numpy as np
from .lsunLayerExceptions import InvalidMode, InvalidMus, InvalidAngles

class SetOfOrthonormalTransforms(nn.Module):
    """
    SETOFORTHONORMALTRANSFORMS
    
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
        n=2,
        nblks=1,
        name='SoOT',
        mode='Analysis',
        dtype=torch.get_default_dtype(),
        device=torch.get_default_device()): #device("cpu")):
        super(SetOfOrthonormalTransforms, self).__init__()
        self.dtype = dtype
        self.nPoints = n
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
        self.__angles = torch.empty(nblks,nAngs,dtype=self.dtype,device=self.device)

        # Mus
        self.__mus = torch.empty(nblks,n,dtype=self.dtype,device=self.device)

        # OrthonormalTransforms
        self.orthonormalTransforms = nn.ModuleList([OrthonormalTransform(n=self.nPoints,mode=self.mode,dtype=self.dtype,device=self.device) for _ in range(nblks)])

    def forward(self, X):
        Z = torch.empty_like(X)
        # TODO: Parallel processing, e.g., torch.nn.DataParallel
        # TODO: JIT compilation, e.g., torch.jit.script
        for iblk, layer in enumerate(self.orthonormalTransforms):
           X_iblk = X[iblk]
           Z_iblk = layer(X_iblk)
           Z[iblk] = Z_iblk
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
        for iblk in range(len(self.orthonormalTransforms)):
            self.orthonormalTransforms[iblk].mode = self.__mode

    @property
    def mus(self):
        for iblk in range(len(self.orthonormalTransforms)):
            self.__mus[iblk] = self.orthonormalTransforms[iblk].mus
        return self.__mus
    
    @mus.setter
    def mus(self,mus):
        if torch.is_tensor(mus):
            self.__mus = mus.to(dtype=self.dtype,device=self.device)
        else:
            self.__mus = torch.tensor(mus,dtype=self.dtype,device=self.device)
        if self.__mus.size(0) != len(self.orthonormalTransforms):
            raise InvalidMus(
                '%s : The number of mus should be equal to the number of blocks'\
                % str(self.__mus)
            )
        for iblk in range(len(self.orthonormalTransforms)):
            self.orthonormalTransforms[iblk].mus = self.__mus[iblk]

    @property
    def angles(self):
        for iblk in range(len(self.orthonormalTransforms)):
            self.__angles[iblk] = self.orthonormalTransforms[iblk].angles.data
        return self.__angles
    
    @angles.setter
    def angles(self,angles):
        if torch.is_tensor(angles):
            self.__angles = angles.to(dtype=self.dtype,device=self.device)
        else:
            self.__angles = torch.tensor(angles,dtype=self.dtype,device=self.device)
        if self.__angles.size(0) != len(self.orthonormalTransforms):
            raise InvalidAngles(
                '%s : The number of angles should be equal to the number of blocks'\
                % str(self.__angles)
            )
        for iblk in range(len(self.orthonormalTransforms)):
            self.orthonormalTransforms[iblk].angles.data = self.__angles[iblk]         

class OrthonormalTransform(nn.Module):
    """
    ORTHONORMALTRANSFORM
    
    Requirements: Python 3.10/11.x, PyTorch 2.3.x
    
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
        name='OT',        
        mus=1,
        mode='Analysis',
        dtype=torch.get_default_dtype(),
        device=torch.device("cpu")):

        super(OrthonormalTransform, self).__init__()
        self.dtype = dtype
        self.nPoints = n
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
        self.angles = nn.Parameter(torch.zeros(nAngs,dtype=self.dtype,device=self.device))

        # Mus
        if torch.is_tensor(mus):
            self.__mus = mus.to(dtype=self.dtype,device=self.device)
        elif mus == 1:
            self.__mus = torch.ones(1,self.nPoints,dtype=self.dtype,device=self.device)
        elif mus == -1:
            self.__mus = -torch.ones(1,self.nPoints,dtype=self.dtype,device=self.device)
        else:
            self.__mus = torch.tensor(mus,dtype=self.dtype,device=self.device)
        self.checkMus()

    def forward(self,X):
        angles = self.angles
        mus = self.__mus
        mode = self.__mode
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
            self.__mus = torch.ones(self.nPoints,dtype=self.dtype,device=self.device)
        elif mus == -1:
            self.__mus = -torch.ones(self.nPoints,dtype=self.dtype,device=self.device)
        else:
            self.__mus = torch.tensor(mus,dtype=self.dtype,device=self.device)
        self.checkMus()

    def checkMus(self):
        if torch.not_equal(torch.abs(self.__mus),torch.ones(self.nPoints,device=self.device)).any():
            raise InvalidMus(
                '%s : Elements in mus should be either of 1 or -1'\
                % str(self.__mus)
            )

class GivensRotations4Analyzer(autograd.Function):
    """
    GIVENSROTATIONS4ANALYZER
    
    Requirements: Python 3.10/11.x, PyTorch 2.3.x
    
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
        #ctx.save_for_backward(input,angles,mus)
        omgs = SingleOrthonormalMatrixGenerationSystem(dtype=input.dtype,partial_difference=False)
        R = omgs(angles,mus).to(input.device)
        ctx.save_for_backward(input,angles,mus,R)
        return R @ input
    
    @staticmethod
    def backward(ctx, grad_output):
        #input, angles, mus = ctx.saved_tensors
        input, angles, mus, R = ctx.saved_tensors
        grad_input = grad_angles = grad_mus = None
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:        
            #omgs = SingleOrthonormalMatrixGenerationSystem(dtype=grad_output.dtype,partial_difference=False)
            #R = omgs(angles,mus).to(grad_output.device)
            dLdX = R.T @ grad_output # dLdX = dZdX @ dLdZ
        # 
        if ctx.needs_input_grad[0]:
            grad_input = dLdX
        if ctx.needs_input_grad[1]:
            #omgs.partial_difference=True
            omgs = SingleOrthonormalMatrixGenerationSystem(dtype=grad_output.dtype,partial_difference=True)            
            grad_angles = torch.zeros_like(angles,dtype=input.dtype,requires_grad=False)
            for iAngle in range(len(grad_angles)):
                dRi = omgs(angles,mus,index_pd_angle=iAngle).to(grad_output.device)
                #grad_angles[iAngle] = torch.sum(dLdX * (dRi @ input))
                grad_angles[iAngle] = torch.sum(grad_output * (dRi @ input))
        if ctx.needs_input_grad[2]:
            grad_mus = torch.zeros_like(mus,dtype=grad_output.dtype,requires_grad=False)                
        return grad_input, grad_angles, grad_mus

class GivensRotations4Synthesizer(autograd.Function):
    """
    GIVENSROTATIONS4SYNTHESIZER
    
    Requirements: Python 3.10/11.x, PyTorch 2.3.x
    
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
        #ctx.save_for_backward(input,angles,mus)
        omgs = SingleOrthonormalMatrixGenerationSystem(dtype=input.dtype,partial_difference=False)
        R = omgs(angles,mus).to(input.device)
        ctx.save_for_backward(input,angles,mus,R)
        return R.T @ input
    
    @staticmethod
    def backward(ctx, grad_output):
        #input, angles, mus = ctx.saved_tensors
        input, angles, mus, R = ctx.saved_tensors
        grad_input = grad_angles = grad_mus = None
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            #omgs = SingleOrthonormalMatrixGenerationSystem(dtype=grad_output.dtype,partial_difference=False)
            #R = omgs(angles,mus).to(grad_output.device)
            dLdX = R @ grad_output # dLdX = dZdX @ dLdZ
        #            
        if ctx.needs_input_grad[0]:
            grad_input = dLdX
        if ctx.needs_input_grad[1]:
            #omgs.partial_difference=True
            omgs = SingleOrthonormalMatrixGenerationSystem(dtype=grad_output.dtype,partial_difference=True)
            grad_angles = torch.zeros_like(angles,dtype=input.dtype,requires_grad=False)
            for iAngle in range(len(grad_angles)):
                dRi = omgs(angles,mus,index_pd_angle=iAngle).to(grad_output.device)
                #grad_angles[iAngle] = torch.sum(dLdX * (dRi.T @ input))
                grad_angles[iAngle] = torch.sum(grad_output * (dRi.T @ input))
        if ctx.needs_input_grad[2]:
            grad_mus = torch.zeros_like(mus,dtype=grad_output.dtype,requires_grad=False)
        return grad_input, grad_angles, grad_mus

@torch.jit.script
def fcn_orthmtxgen_diff(nDims: int, angles: torch.Tensor, index_pd_angle: int):
    matrix = torch.eye(nDims,dtype=angles.dtype,device=angles.device)
    #matrix.requires_grad = False
    iAng = 0
    for iTop in range(nDims-1):
        vt = matrix[iTop,:]
        for iBtm in range(iTop+1,nDims):
            angle = angles[iAng]
            if iAng == index_pd_angle:
                angle = angle + torch.pi/2. #math.pi/2.
            c = torch.cos(angle)
            s = torch.sin(angle)
            vb = matrix[iBtm,:]
            #
            u  = s*(vt + vb)
            vt = (c + s)*vt
            vb = (c - s)*vb
            vt = vt - u
            if iAng == index_pd_angle:
                matrix = torch.zeros_like(matrix,dtype=angles.dtype,device=angles.device)
                #matrix.requires_grad = False
            matrix[iBtm,:] = vb + u
            iAng = iAng + 1
        matrix[iTop,:] = vt

    return matrix 
    
@torch.jit.script
def fcn_orthmtxgen(nDims: int, angles: torch.Tensor):
    matrix = torch.eye(nDims,dtype=angles.dtype,device=angles.device)
    #matrix.requires_grad = False
    iAng = 0
    for iTop in range(nDims-1):
        vt = matrix[iTop,:]
        for iBtm in range(iTop+1,nDims):
            angle = angles[iAng]
            c = torch.cos(angle)
            s = torch.sin(angle)
            vb = matrix[iBtm,:]
            #
            u  = s*(vt + vb)
            vt = (c + s)*vt
            vb = (c - s)*vb
            vt = vt - u
            matrix[iBtm,:] = vb + u
            iAng = iAng + 1
        matrix[iTop,:] = vt

    return matrix

class SingleOrthonormalMatrixGenerationSystem:
    """
    SINGLEORTHONORMALMATRIXGENERATIONSYSTEM
    
    Requirements: Python 3.10/11.x, PyTorch 2.3.x
    Copyright (c) 2021-2024, Shogo MURAMATSU
    
    All rights reserved.
    
    Contact address: Shogo MURAMATSU,
        Faculty of Engineering, Niigata University,
        8050 2-no-cho Ikarashi, Nishi-ku,
        Niigata, 950-2181, JAPAN
    
        http://www.eng.niigata-u.ac.jp/~msiplab/
    """

    def __init__(self,
        dtype=torch.get_default_dtype(),
        partial_difference=False):
        
        super(SingleOrthonormalMatrixGenerationSystem, self).__init__()
        self.dtype = dtype
        self.partial_difference = partial_difference


    def __call__(self,
        angles=0,
        mus=1,
        index_pd_angle=None):
        """
        The output is set on the same device with the angles
        """

        # Number of angles
        if isinstance(angles, int) or isinstance(angles, float):
            angles = torch.tensor([angles],dtype=self.dtype) 
        elif not torch.is_tensor(angles):
            angles = torch.tensor(angles,dtype=self.dtype)
        else:
            angles = angles.to(dtype=self.dtype) 
        nAngles = len(angles)

        # Number of dimensions
        nDims = int((1+math.sqrt(1+8*nAngles))/2)

        # Setup of mus, which is send to the same device with angles
        if isinstance(mus, int) or isinstance(mus, float):
            mus = mus * torch.ones(nDims,dtype=self.dtype,device=angles.device,requires_grad=False)
        elif not torch.is_tensor(mus): #isinstance(mus, list):
            mus = torch.tensor(mus,dtype=self.dtype,device=angles.device,requires_grad=False)
        else:
            mus = mus.to(dtype=self.dtype,device=angles.device)

        if self.partial_difference:
            #matrix = self.fcn_orthmtxgen_diff(nDims, angles, index_pd_angle)
            matrix = fcn_orthmtxgen_diff(nDims, angles, index_pd_angle)            
        else:
            #matrix = self.fcn_orthmtxgen(nDims, angles)
            matrix = fcn_orthmtxgen(nDims, angles)            

        matrix = mus.view(-1,1) * matrix

        return matrix 

    """
    @torch.jit.script
    def fcn_orthmtxgen_diff(self, nDims: int, angles: torch.Tensor, index_pd_angle: int):
        matrix = torch.eye(nDims,dtype=angles.dtype,device=angles.device)
        #matrix.requires_grad = False
        iAng = 0
        for iTop in range(nDims-1):
            vt = matrix[iTop,:]
            for iBtm in range(iTop+1,nDims):
                angle = angles[iAng]
                if iAng == index_pd_angle:
                    angle = angle + torch.pi/2. #math.pi/2.
                c = torch.cos(angle)
                s = torch.sin(angle)
                vb = matrix[iBtm,:]
                #
                u  = s*(vt + vb)
                vt = (c + s)*vt
                vb = (c - s)*vb
                vt = vt - u
                if iAng == index_pd_angle:
                    matrix = torch.zeros_like(matrix,dtype=angles.dtype,device=angles.device)
                    #matrix.requires_grad = False
                matrix[iBtm,:] = vb + u
                iAng = iAng + 1
            matrix[iTop,:] = vt

        return matrix 
    
    @torch.jit.script
    def fcn_orthmtxgen(self, nDims: int, angles: torch.Tensor):
        matrix = torch.eye(nDims,dtype=angles.dtype,device=angles.device)
        #matrix.requires_grad = False
        iAng = 0
        for iTop in range(nDims-1):
            vt = matrix[iTop,:]
            for iBtm in range(iTop+1,nDims):
                angle = angles[iAng]
                c = torch.cos(angle)
                s = torch.sin(angle)
                vb = matrix[iBtm,:]
                #
                u  = s*(vt + vb)
                vt = (c + s)*vt
                vb = (c - s)*vb
                vt = vt - u
                matrix[iBtm,:] = vb + u
                iAng = iAng + 1
            matrix[iTop,:] = vt

        return matrix
    """