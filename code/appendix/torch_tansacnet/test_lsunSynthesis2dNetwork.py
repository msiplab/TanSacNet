import itertools
import unittest
from parameterized import parameterized
import torch
import torch.nn as nn
import torch_dct as dct

import math
import random
from torch_tansacnet.lsunSynthesis2dNetwork import LsunSynthesis2dNetwork
from torch_tansacnet.lsunUtility import Direction, OrthonormalMatrixGenerationSystem
from torch_tansacnet.orthonormalTransform import OrthonormalTransform
from torch_tansacnet.lsunLayerExceptions import InvalidOverlappingFactor, InvalidNoDcLeakage, InvalidNumberOfLevels, InvalidStride, InvalidInputSize

stride = [ [2, 1], [1, 2], [2, 2], [2, 4], [4, 1], [4, 4] ]
ovlpfactor = [ [1, 1], [3, 3], [5, 5], [1, 3], [3, 1] ]
datatype = [ torch.float, torch.double ]
height = [ 8, 16, 32 ]
width = [ 8, 16, 32 ]
nodcleakage = [ False, True ]
nlevels = [ 1, 2, 3 ]
usegpu = [ True, False ]

class LsunSynthesis2dNetworkTestCase(unittest.TestCase):
    """
    LSUNSYNHESIS2DNETWORKTESTCASE Test cases for LsunSynthesis2dNetwork
    
    Requirements: Requirements: Python 3.10-12.x, PyTorch 2.3/4.x
    
    Copyright (c) 2024, Shogo MURAMATSU
    
    All rights reserved.
    
    Contact address: Shogo MURAMATSU,
        Faculty of Engineering, Niigata University,
        8050 2-no-cho Ikarashi, Nishi-ku,
        Niigata, 950-2181, JAPAN
    
        https://www.eng.niigata-u.ac.jp/~msiplab/
    """
    
    @parameterized.expand(
        list(itertools.product(stride,ovlpfactor,height,width,nodcleakage))
    )
    def testConstructor(self, stride,ovlpfactor,height,width,nodcleakage):

        # Expcted values
        expctdStride = stride
        expctdOvlpFactor = ovlpfactor
        expctdInputSize = [ height, width ]
        expctdNoDcLeakage = nodcleakage        

        # Instantiation of target class
        network = LsunSynthesis2dNetwork(
            input_size = [ height, width ],
            stride = stride,
            overlapping_factor = ovlpfactor,
            no_dc_leakage = nodcleakage
        )

        # Actual values
        actualStride = network.stride
        actualOvlpFactor = network.overlapping_factor
        actualInputSize = network.input_size
        actualNoDcLeakage = network.no_dc_leakage 

        # Evaluation
        self.assertTrue(isinstance(network, nn.Module))
        self.assertEqual(actualStride,expctdStride)
        self.assertEqual(actualOvlpFactor,expctdOvlpFactor)
        self.assertEqual(actualInputSize,expctdInputSize)
        self.assertEqual(actualNoDcLeakage,expctdNoDcLeakage)                

    @parameterized.expand(
        list(itertools.product(stride,height,width,datatype,usegpu))
    )
    def testForwardGrayScale(self,
            stride, height, width, datatype,usegpu):
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else:
                print('No GPU device was detected.')
                return
        else:
            device = torch.device("cpu")    
        rtol,atol = 1e-5,1e-8

        # Parameters
        nSamples = 8
        nComponents = 1
        nDecs = stride[Direction.VERTICAL]*stride[Direction.HORIZONTAL] #math.prod(stride)

        # nSamples x nRows x nCols x nDecs
        nrows = height//stride[Direction.VERTICAL]
        ncols = width//stride[Direction.HORIZONTAL]
        X = torch.randn(nSamples,nrows,ncols,nDecs,dtype=datatype,device=device,requires_grad=True)
        
        # Expected values        
        # nSamples x nComponents x (Stride[0]xnRows) x (Stride[1]xnCols)
        ps,pa = int(math.ceil(nDecs/2.)), int(math.floor(nDecs/2.)) 
        W0T = torch.eye(ps,dtype=datatype).to(device)
        U0T = torch.eye(pa,dtype=datatype).to(device)
        Ys = X[:,:,:,:ps].view(-1,ps).T
        if pa > 0:
            Ya = X[:,:,:,ps:].view(-1,pa).T
            Zsa = torch.cat(
                ( W0T @ Ys, 
                  U0T @ Ya ),dim=0)
        else:
            Zsa = W0T @ Ys
        V = Zsa.T.view(nSamples,nrows,ncols,nDecs)
        A = permuteIdctCoefs_(V,stride)        
        arrayshape = stride.copy() # FIXME
        arrayshape.insert(0,-1)
        expctdZ = dct.idct_2d(A.view(arrayshape),norm='ortho').reshape(nSamples,nComponents,height,width)        
        
        # Instantiation of target class
        network = LsunSynthesis2dNetwork(
                input_size = [ height, width ],
                stride=stride,
                dtype=datatype
            )
        network = network.to(device)

        # Actual values
        with torch.no_grad():
            actualZ = network.forward(X)

        # Evaluation
        self.assertEqual(actualZ.dtype,datatype)
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))
        self.assertFalse(actualZ.requires_grad)

    @parameterized.expand(
        list(itertools.product(stride,ovlpfactor))
        )
    def testOverlappingFactorException(self,
        stride,ovlpfactor):
        with self.assertRaises(InvalidOverlappingFactor):
            LsunSynthesis2dNetwork(
                overlapping_factor = [ ovlpfactor[Direction.VERTICAL]+1, ovlpfactor[Direction.HORIZONTAL] ],
                stride = stride
            )

        with self.assertRaises(InvalidOverlappingFactor):
            LsunSynthesis2dNetwork(
                overlapping_factor = [ ovlpfactor[Direction.VERTICAL], ovlpfactor[Direction.HORIZONTAL]+1 ],
                stride = stride
            )

        with self.assertRaises(InvalidOverlappingFactor):
            LsunSynthesis2dNetwork(
                overlapping_factor = [ ovlpfactor[Direction.VERTICAL]+1, ovlpfactor[Direction.HORIZONTAL]+1 ],
                stride = stride
            )


    @parameterized.expand(
        list(itertools.product(stride,ovlpfactor))
    )
    def testNumberOfVanishingMomentsException(self,
        stride,ovlpfactor):
        no_dc_leakage = 0
        with self.assertRaises(InvalidNoDcLeakage):
            LsunSynthesis2dNetwork(
                stride = stride,
                overlapping_factor = ovlpfactor,
                no_dc_leakage = no_dc_leakage
            )

        no_dc_leakage = 1
        with self.assertRaises(InvalidNoDcLeakage):
            LsunSynthesis2dNetwork(
                stride = stride,
                overlapping_factor = ovlpfactor,
                no_dc_leakage = no_dc_leakage
            )

    @parameterized.expand(
        list(itertools.product(stride,ovlpfactor))
    )
    def testNumberOfLevelsException(self,
        stride,ovlpfactor):
        nlevels = -1
        with self.assertRaises(InvalidNumberOfLevels):
            LsunSynthesis2dNetwork(
                stride = stride,
                overlapping_factor = ovlpfactor,
                number_of_levels = nlevels
            )

        nlevels = 0.5
        with self.assertRaises(InvalidNumberOfLevels):
            LsunSynthesis2dNetwork(
                stride = stride,
                overlapping_factor = ovlpfactor,
                number_of_levels = nlevels
            )


    @parameterized.expand(
        list(itertools.product(ovlpfactor))
    )
    def testStrideException(self,ovlpfactor):
        stride = [ 1, 1 ]
        with self.assertRaises(InvalidStride):
            LsunSynthesis2dNetwork(
                stride = stride,
                overlapping_factor = ovlpfactor
            )

        stride = [ 1, 3 ]
        with self.assertRaises(InvalidStride):
            LsunSynthesis2dNetwork(
                stride = stride,
                overlapping_factor = ovlpfactor
            )

        stride = [ 3, 1 ]
        with self.assertRaises(InvalidStride):
            LsunSynthesis2dNetwork(
                stride = stride,
                overlapping_factor = ovlpfactor
            )

        stride = [ 3, 3 ]
        with self.assertRaises(InvalidStride):
            LsunSynthesis2dNetwork(
                stride = stride,
                overlapping_factor = ovlpfactor
            )


    @parameterized.expand(
        list(itertools.product(stride))
    )
    def testInputSize(self,stride):
        input_size = [ 2*stride[Direction.VERTICAL]+1, 2*stride[Direction.HORIZONTAL]+1 ]
        with self.assertRaises(InvalidInputSize):
            LsunSynthesis2dNetwork(
                input_size = input_size,
                stride = stride
            )
    
        input_size = [ -2*stride[Direction.VERTICAL], -2*stride[Direction.HORIZONTAL] ]
        with self.assertRaises(InvalidInputSize):
            LsunSynthesis2dNetwork(
                input_size = input_size,
                stride = stride
            )


    @parameterized.expand(
        list(itertools.product(stride,height,width,datatype,usegpu))
    )
    def testForwardGrayScaleWithInitalization(self,
            stride, height, width, datatype,usegpu):
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else:
                print('No GPU device was detected.')
                return
        else:
            device = torch.device("cpu")    
        rtol,atol = 1e-4,1e-5

        genW = OrthonormalMatrixGenerationSystem(dtype=datatype)
        genU = OrthonormalMatrixGenerationSystem(dtype=datatype)

        # Initialization function of angle parameters
        angle0 = 2.0*math.pi*random.random()
        def init_angles(m):
            if type(m) == OrthonormalTransform:
                torch.nn.init.constant_(m.angles,angle0)        

        # Parameters
        isNoDcLeakage = False
        nSamples = 8
        nComponents = 1
        nDecs = stride[Direction.VERTICAL]*stride[Direction.HORIZONTAL]
        nrows = height//stride[Direction.VERTICAL]
        ncols = width//stride[Direction.HORIZONTAL]
        ps,pa = int(math.ceil(nDecs/2.)), int(math.floor(nDecs/2.))
        nAngles = (nDecs-2)*nDecs//4
        nAnglesH = nAngles//2
        angles = angle0*torch.ones(nrows*ncols,nAngles,dtype=datatype,device=device)
        W0T = genW(angles=angles[:,:nAnglesH]).transpose(1,2)
        U0T = genU(angles=angles[:,nAnglesH:]).transpose(1,2)

        # nSamples x nRows x nCols x nDecs
        X = torch.randn(nSamples,nrows,ncols,nDecs,dtype=datatype,device=device,requires_grad=True)

        # Expected values        
        V = torch.zeros_like(X)
        for iSample in range(nSamples):
            Xi = X[iSample,:,:,:].clone()
            Ys = Xi[:,:,:ps].view(-1,ps)
            Ya = Xi[:,:,ps:].view(-1,pa)
            for iblk in range(nrows*ncols):
                Ys[iblk,:] = W0T[iblk,:,:] @ Ys[iblk,:]
                Ya[iblk,:] = U0T[iblk,:,:] @ Ya[iblk,:]
            Yi = torch.cat((Ys,Ya),dim=1).view(nrows,ncols,nDecs)
            V[iSample,:,:,:] = Yi
        A = permuteIdctCoefs_(V,stride)
        arrayshape = stride.copy() 
        arrayshape.insert(0,-1)
        expctdZ = dct.idct_2d(A.view(arrayshape),norm='ortho').reshape(nSamples,nComponents,height,width)

        # Instantiation of target class
        network = LsunSynthesis2dNetwork(
            input_size=[height,width],
            stride=stride,
            no_dc_leakage=isNoDcLeakage,
            dtype=datatype
        )
        network = network.to(device)

        # Initialization of angle parameters
        network.apply(init_angles)

        # Actual values
        with torch.no_grad():
            actualZ = network.forward(X)

        # Evaluation
        self.assertEqual(actualZ.dtype,datatype)
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))
        self.assertFalse(actualZ.requires_grad)

    @parameterized.expand(
        list(itertools.product(stride,ovlpfactor,height,width,nodcleakage,datatype,usegpu))
    )
    def testForwardGrayScaleOverlappingDefault(self,
            stride, ovlpfactor, height, width, nodcleakage,datatype,usegpu):
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else:
                print('No GPU device was detected.')
                return
        else:
            device = torch.device("cpu")    
        rtol,atol = 1e-4,1e-5

        # Parameters
        isNoDcLeakage = nodcleakage
        nSamples = 8
        nComponents = 1
        nDecs = stride[Direction.VERTICAL]*stride[Direction.HORIZONTAL]
        nrows = height//stride[Direction.VERTICAL]
        ncols = width//stride[Direction.HORIZONTAL]

        # nSamples x nRows x nCols x nDecs
        X = torch.randn(nSamples,nrows,ncols,nDecs,dtype=datatype,device=device,requires_grad=True)

        # Expected values
        A = permuteIdctCoefs_(X,stride)
        Y = dct.idct_2d(A,norm='ortho')
        expctdZ = Y.reshape(nSamples,nComponents,height,width)

        # Instantiation of target class
        network = LsunSynthesis2dNetwork(
            input_size=[height,width],
            stride=stride,
            overlapping_factor=ovlpfactor,
            no_dc_leakage=isNoDcLeakage,
            dtype=datatype
        )
        network = network.to(device)

        # Actual values
        with torch.no_grad():
            actualZ = network.forward(X)

        # Evaluation
        self.assertEqual(actualZ.dtype,datatype)
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))
        self.assertFalse(actualZ.requires_grad)

    @parameterized.expand(
        list(itertools.product(stride,height,width,nodcleakage,datatype,usegpu))
    )
    def testForwardGrayScaleOvlpFactor11(self,
        stride, height, width, nodcleakage, datatype, usegpu):
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else:
                print('No GPU device was detected.')
                return
        else:
            device = torch.device("cpu")
        rtol, atol = 1e-4, 1e-5

        genW = OrthonormalMatrixGenerationSystem(dtype=datatype)
        genU = OrthonormalMatrixGenerationSystem(dtype=datatype)

        # Initialization function of angle parameters
        angle0 = 2.0*math.pi*random.random()
        def init_angles(m):
            if type(m) == OrthonormalTransform:
                torch.nn.init.constant_(m.angles, angle0)

        # Parameters
        ovlpFactor = [ 1, 1 ]
        isNoDcLeakage = nodcleakage # False # TODO: Handling no_dc_leakage
        nSamples = 8
        nComponents = 1
        nDecs = stride[Direction.VERTICAL]*stride[Direction.HORIZONTAL]
        nrows = height//stride[Direction.VERTICAL]
        ncols = width//stride[Direction.HORIZONTAL]

        # nSamples x nRows x nCols x nDecs
        X = torch.randn(nSamples,nrows,ncols,nDecs,dtype=datatype,device=device,requires_grad=True)
        nAngles = (nDecs-2)*nDecs//4
        angles = angle0*torch.ones(nrows*ncols,nAngles,dtype=datatype,device=device)

        # Expected values
        ps = math.ceil(nDecs/2)
        pa = math.floor(nDecs/2)
        #nchs = (ps, pa)
        nAnglesH = nAngles//2
        if isNoDcLeakage:
            angles[:,:(ps-1)] = 0
        W0T = genW(angles[:,:nAnglesH]).transpose(1,2) 
        U0T = genU(angles[:,nAnglesH:]).transpose(1,2)

        # Final rotation
        Z = X.clone()
        for iSample in range(nSamples):
            Vi = Z[iSample,:,:,:].clone()
            Ys = Vi[:,:,:ps].view(-1,ps)
            Ya = Vi[:,:,ps:].view(-1,pa)
            for iblk in range(nrows*ncols):
                Ys[iblk,:] = W0T[iblk,:] @ Ys[iblk,:]
                Ya[iblk,:] = U0T[iblk,:] @ Ya[iblk,:]
            Yi = torch.cat((Ys,Ya),dim=1).view(nrows,ncols,nDecs)
            Z[iSample,:,:,:] = Yi            
        A = permuteIdctCoefs_(Z,stride)
        Y = dct.idct_2d(A,norm='ortho')
        # Samples x nComponents x (nrows x decV)x (decH x decH)
        expctdZ = Y.reshape(nSamples,nComponents,height,width)

        # Instantiation of target class
        network = LsunSynthesis2dNetwork(
            input_size=[height,width],
            stride=stride,
            overlapping_factor=ovlpFactor,
            no_dc_leakage=isNoDcLeakage,
            dtype=datatype
        )
        network = network.to(device)

        # Actual values
        network.apply(init_angles)        
        with torch.no_grad():
            actualZ = network.forward(X)

        # Evaluation
        self.assertEqual(actualZ.dtype,datatype)
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))
        self.assertFalse(actualZ.requires_grad)

    @parameterized.expand(
        list(itertools.product(stride,height,width,nodcleakage, datatype,usegpu))
    )
    def testForwardGrayScaleOvlpFactor33(self,
        stride, height, width, nodcleakage, datatype, usegpu):
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else:
                print('No GPU device was detected.')
                return
        else:
            device = torch.device("cpu")
        rtol, atol = 1e-3, 1e-4

        genW = OrthonormalMatrixGenerationSystem(dtype=datatype)
        genU = OrthonormalMatrixGenerationSystem(dtype=datatype)

        # Initialization function of angle parameters
        angle0 = 2.0 * math.pi * random.random()
        def init_angles(m):
            if type(m) == OrthonormalTransform:
                torch.nn.init.constant_(m.angles, angle0)

        # Parameters
        ovlpFactor = [3, 3]
        isNoDcLeakage = nodcleakage # False # TODO: Handling no_dc_leakage
        nSamples = 8
        nComponents = 1
        nDecs = stride[Direction.VERTICAL] * stride[Direction.HORIZONTAL]
        nrows = height//stride[Direction.VERTICAL]
        ncols = width//stride[Direction.HORIZONTAL]

        # nSamples x nRows x nCols x nDecs
        X = torch.randn(nSamples,nrows,ncols,nDecs,dtype=datatype,device=device,requires_grad=True)
        nAngles = (nDecs-2)*nDecs//4
        angles = angle0*torch.ones(nrows*ncols,nAngles,dtype=datatype,device=device)

        # Expected values
        ps = math.ceil(nDecs/2)
        pa = math.floor(nDecs/2)
        nchs = (ps, pa)
        nAnglesH = nAngles//2
        if isNoDcLeakage:
            angles[:,:(ps-1)] = 0
        W0T = genW(angles[:,:nAnglesH]).transpose(1,2) 
        U0T = genU(angles[:,nAnglesH:]).transpose(1,2)
        Uh1T = -genU(angles[:,nAnglesH:]).transpose(1,2)
        Uh2T = -genU(angles[:,nAnglesH:]).transpose(1,2)
        Uv1T = -genU(angles[:,nAnglesH:]).transpose(1,2)
        Uv2T = -genU(angles[:,nAnglesH:]).transpose(1,2)

        # Vertical atom concatenation
        Z = X.clone()
        #for ordV in range(ovlpFactor[Direction.VERTICAL]//2):        
        Z = intermediate_rotation_(Z,nchs,Uv2T)
        Z = block_butterfly_(Z,nchs)
        Z = block_shift_(Z,nchs,1,[0,1,0,0]) # target=sum, shift=down
        Z = block_butterfly_(Z,nchs)/2.
        Z = intermediate_rotation_(Z,nchs,Uv1T)
        Z = block_butterfly_(Z,nchs)
        Z = block_shift_(Z,nchs,0,[0,-1,0,0]) # target=diff, shift=up
        Z = block_butterfly_(Z,nchs)/2.
        # Horizontal atom concatenation
        #for ordH in range(ovlpFactor[Direction.HORIZONTAL]//2):  
        Z = intermediate_rotation_(Z,nchs,Uh2T)
        Z = block_butterfly_(Z,nchs)
        Z = block_shift_(Z,nchs,1,[0,0,1,0]) # target=sum, shift=right
        Z = block_butterfly_(Z,nchs)/2.
        Z = intermediate_rotation_(Z,nchs,Uh1T)
        Z = block_butterfly_(Z,nchs)
        Z = block_shift_(Z,nchs,0,[0,0,-1,0]) # target=diff, shift=left
        Z = block_butterfly_(Z,nchs)/2. 
        # Final rotation
        for iSample in range(nSamples):
            Vi = Z[iSample,:,:,:].clone()
            Ys = Vi[:,:,:ps].view(-1,ps)
            Ya = Vi[:,:,ps:].view(-1,pa)
            for iblk in range(nrows*ncols):
                Ys[iblk,:] = W0T[iblk,:] @ Ys[iblk,:]
                Ya[iblk,:] = U0T[iblk,:] @ Ya[iblk,:]
            Yi = torch.cat((Ys,Ya),dim=1).view(nrows,ncols,nDecs)
            Z[iSample,:,:,:] = Yi                   
        # Block IDCT (nSamples x nRows x nCols x nDecs)
        A = permuteIdctCoefs_(Z,stride)
        Y = dct.idct_2d(A,norm='ortho')
        # Samples x nComponents x (nrows x decV)x (decH x decH)
        expctdZ = Y.reshape(nSamples,nComponents,height,width)

        # Instantiation of target class
        network = LsunSynthesis2dNetwork(
            input_size=[height,width],
            stride=stride,
            overlapping_factor=ovlpFactor,
            no_dc_leakage=isNoDcLeakage,
            dtype=datatype
        )
        network = network.to(device)

        # Actual values
        network.apply(init_angles)        
        with torch.no_grad():
            actualZ = network.forward(X)

        # Evaluation
        self.assertEqual(actualZ.dtype,datatype)
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))
        self.assertFalse(actualZ.requires_grad)
            
    @parameterized.expand(
        list(itertools.product(stride,ovlpfactor,height,width,datatype,usegpu))
    )
    def testForwardGrayScaleOvlpFactorXX(self,
        stride, ovlpfactor,height, width, datatype, usegpu):
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else:
                print('No GPU device was detected.')
                return
        else:
            device = torch.device("cpu")
        rtol, atol = 1e-3, 1e-4

        genW = OrthonormalMatrixGenerationSystem(dtype=datatype)
        genU = OrthonormalMatrixGenerationSystem(dtype=datatype)

        # Initialization function of angle parameters
        angle0 = 2.0 * math.pi * random.random()
        def init_angles(m):
            if type(m) == OrthonormalTransform:
                torch.nn.init.constant_(m.angles, angle0)

        # Parameters
        ovlpFactor = ovlpfactor
        isNoDcLeakage = False 
        nSamples = 8
        nComponents = 1
        nDecs = stride[Direction.VERTICAL] * stride[Direction.HORIZONTAL]
        nrows = height // stride[Direction.VERTICAL]
        ncols = width // stride[Direction.HORIZONTAL]

        # nSamples x nRows x nCols x nDecs
        X = torch.randn(nSamples,nrows,ncols,nDecs,dtype=datatype,device=device,requires_grad=True)
        nAngles = (nDecs-2)*nDecs//4
        angles = angle0*torch.ones(nrows*ncols,nAngles,dtype=datatype,device=device)

        # Expected values
        ps = math.ceil(nDecs/2)
        pa = math.floor(nDecs/2)
        nchs = (ps, pa)
        nAnglesH = nAngles//2
        W0T = genW(angles[:,:nAnglesH]).transpose(1,2) 
        U0T = genU(angles[:,nAnglesH:]).transpose(1,2)
        Uh1T = -genU(angles[:,nAnglesH:]).transpose(1,2)
        Uh2T = -genU(angles[:,nAnglesH:]).transpose(1,2)
        Uv1T = -genU(angles[:,nAnglesH:]).transpose(1,2)
        Uv2T = -genU(angles[:,nAnglesH:]).transpose(1,2)

        # Vertical atom concatenation
        Z = X.clone()
        for ordV in range(ovlpFactor[Direction.VERTICAL]//2):        
            Z = intermediate_rotation_(Z,nchs,Uv2T)
            Z = block_butterfly_(Z,nchs)
            Z = block_shift_(Z,nchs,1,[0,1,0,0]) # target=sum, shift=down
            Z = block_butterfly_(Z,nchs)/2.
            Z = intermediate_rotation_(Z,nchs,Uv1T)
            Z = block_butterfly_(Z,nchs)
            Z = block_shift_(Z,nchs,0,[0,-1,0,0]) # target=diff, shift=up
            Z = block_butterfly_(Z,nchs)/2.
        # Horizontal atom concatenation
        for ordH in range(ovlpFactor[Direction.HORIZONTAL]//2):        
            Z = intermediate_rotation_(Z,nchs,Uh2T)
            Z = block_butterfly_(Z,nchs)
            Z = block_shift_(Z,nchs,1,[0,0,1,0]) # target=sum, shift=right
            Z = block_butterfly_(Z,nchs)/2.
            Z = intermediate_rotation_(Z,nchs,Uh1T)
            Z = block_butterfly_(Z,nchs)
            Z = block_shift_(Z,nchs,0,[0,0,-1,0]) # target=diff, shift=left
            Z = block_butterfly_(Z,nchs)/2.
        # Final rotation
        for iSample in range(nSamples):
            Vi = Z[iSample,:,:,:].clone()
            Ys = Vi[:,:,:ps].view(-1,ps)
            Ya = Vi[:,:,ps:].view(-1,pa)
            for iblk in range(nrows*ncols):
                Ys[iblk,:] = W0T[iblk,:] @ Ys[iblk,:]
                Ya[iblk,:] = U0T[iblk,:] @ Ya[iblk,:]
            Yi = torch.cat((Ys,Ya),dim=1).view(nrows,ncols,nDecs)
            Z[iSample,:,:,:] = Yi                 
        # Block IDCT (nSamples x nRows x nCols x nDecs)
        A = permuteIdctCoefs_(Z,stride)
        Y = dct.idct_2d(A,norm='ortho')
        # Samples x nComponents x (nrows x decV)x (decH x decH)
        expctdZ = Y.reshape(nSamples,nComponents,height,width)

        # Instantiation of target class
        network = LsunSynthesis2dNetwork(
            input_size=[height,width],
            stride=stride,
            overlapping_factor=ovlpFactor,
            no_dc_leakage=isNoDcLeakage,
            dtype=datatype
        )
        network = network.to(device)

        # Actual values
        network.apply(init_angles)
        with torch.no_grad():
            actualZ = network.forward(X)

        # Evaluation
        self.assertEqual(actualZ.dtype,datatype)
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))
        self.assertFalse(actualZ.requires_grad)
    
    @parameterized.expand(
        list(itertools.product(height,width,nlevels,nodcleakage,datatype,usegpu))
    )
    def testForwardGrayScaleMultiLevels(self,
            height, width, nlevels, nodcleakage, datatype, usegpu):
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else:
                print('No GPU device was detected.')
                return
        else:
            device = torch.device("cpu")
        rtol, atol = 1e-3, 1e-4

        genW = OrthonormalMatrixGenerationSystem(dtype=datatype)
        genU = OrthonormalMatrixGenerationSystem(dtype=datatype)

        # Initialization function of angle parameters
        angle0 = 2.0 * math.pi * random.random()
        def init_angles(m):
            if type(m) == OrthonormalTransform:
                torch.nn.init.constant_(m.angles, angle0)

        # Parameters
        stride = [2, 2]
        ovlpFactor = [3, 3]
        isNoDcLeakage = nodcleakage # False 
        nSamples = 8
        nComponents = 1
        nDecs = stride[Direction.VERTICAL] * stride[Direction.HORIZONTAL]
        nrows = height // (stride[Direction.VERTICAL]**nlevels)
        ncols = width // (stride[Direction.HORIZONTAL]**nlevels)
        # nSamples x nRows x nCols x nDecs
        nrows_ = nrows
        ncols_ = ncols
        X = []
        for iLevel in range(1,nlevels+1):
            if iLevel == 1:
                X.append(torch.randn(nSamples,nrows_,ncols_,1,dtype=datatype,device=device,requires_grad=True)) 
            X.append(torch.randn(nSamples,nrows_,ncols_,nDecs-1,dtype=datatype,device=device,requires_grad=True))     
            nrows_ *= stride[Direction.VERTICAL]
            ncols_ *= stride[Direction.HORIZONTAL]
        X = tuple(X)
        nAngles = (nDecs-2)*nDecs//4
        angles = angle0*torch.ones(nrows_*ncols_,nAngles,dtype=datatype,device=device)            

        # Expected values
        ps = math.ceil(nDecs/2)
        pa = math.floor(nDecs/2)
        nchs = (ps, pa)
        if isNoDcLeakage:
            angles[:,:(ps-1)] = 0
        nAnglesH = nAngles//2
        W0T = genW(angles[:,:nAnglesH]).transpose(1,2)
        U0T = genU(angles[:,nAnglesH:]).transpose(1,2)
        Uh1T = -genU(angles[:,nAnglesH:]).transpose(1,2)
        Uh2T = -genU(angles[:,nAnglesH:]).transpose(1,2)
        Uv1T = -genU(angles[:,nAnglesH:]).transpose(1,2)
        Uv2T = -genU(angles[:,nAnglesH:]).transpose(1,2)
        
        # nSamples x nRows x nCols x nDecs
        # Multi-level reconstruction
        nrows_ = nrows
        ncols_ = ncols
        for iLevel in range(nlevels,0,-1):
            # Extract scale channel
            if iLevel == nlevels:
                Xdc = X[0]
            Xac = X[nlevels-iLevel+1]
            Z = torch.cat((Xdc,Xac),dim=3)
            # Vertical atom concatenation
            for ordV in range(ovlpFactor[Direction.VERTICAL]//2):        
                Z = intermediate_rotation_(Z,nchs,Uv2T)
                Z = block_butterfly_(Z,nchs)
                Z = block_shift_(Z,nchs,1,[0,1,0,0]) # target=sum, shift=down
                Z = block_butterfly_(Z,nchs)/2.
                Z = intermediate_rotation_(Z,nchs,Uv1T)
                Z = block_butterfly_(Z,nchs)
                Z = block_shift_(Z,nchs,0,[0,-1,0,0]) # target=diff, shift=up
                Z = block_butterfly_(Z,nchs)/2.
            # Horizontal atom concatenation
            for ordH in range(ovlpFactor[Direction.HORIZONTAL]//2):        
                Z = intermediate_rotation_(Z,nchs,Uh2T)
                Z = block_butterfly_(Z,nchs)
                Z = block_shift_(Z,nchs,1,[0,0,1,0]) # target=sum, shift=right
                Z = block_butterfly_(Z,nchs)/2.
                Z = intermediate_rotation_(Z,nchs,Uh1T)
                Z = block_butterfly_(Z,nchs)
                Z = block_shift_(Z,nchs,0,[0,0,-1,0]) # target=diff, shift=left
                Z = block_butterfly_(Z,nchs)/2.
            
            # Final rotation
            for iSample in range(nSamples):
                Vi = Z[iSample,:,:,:].clone()
                Ys = Vi[:,:,:ps].view(-1,ps)
                Ya = Vi[:,:,ps:].view(-1,pa)
                for iblk in range(nrows_*ncols_):
                    Ys[iblk,:] = W0T[iblk,:] @ Ys[iblk,:]
                    Ya[iblk,:] = U0T[iblk,:] @ Ya[iblk,:]
                Yi = torch.cat((Ys,Ya),dim=1).view(nrows_,ncols_,nDecs)
                Z[iSample,:,:,:] = Yi                 
            # Block IDCT (nSamples x nRows x nCols x nDecs)
            A = permuteIdctCoefs_(Z,stride)
            Y = dct.idct_2d(A,norm='ortho')
            # Update
            nrows_ *= stride[Direction.VERTICAL]
            ncols_ *= stride[Direction.HORIZONTAL]            
            Xdc = Y.reshape(nSamples,nrows_,ncols_,1)
        expctdZ = Xdc.view(nSamples,nComponents,height,width)

        # Instantiation of target class
        network = LsunSynthesis2dNetwork(
            input_size=[height,width],
            stride=stride,
            overlapping_factor=ovlpFactor,
            number_of_levels=nlevels,
            no_dc_leakage=isNoDcLeakage,
            dtype=datatype
        )
        network = network.to(device)

        # Initialization of angle parameters
        network.apply(init_angles)

        # Actual values
        with torch.no_grad():
            actualZ = network.forward(X)

        # Evaluation
        self.assertEqual(actualZ.dtype,datatype)
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))
        self.assertFalse(actualZ.requires_grad)

    @parameterized.expand(
        list(itertools.product(stride,ovlpfactor,height,width,nodcleakage,datatype,usegpu))
    )
    def testBackwardGrayScale(self,
        stride,ovlpfactor,height,width,nodcleakage,datatype,usegpu):
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else:
                print('No GPU device was detected.')
                return
        else:
            device = torch.device("cpu")
        rtol, atol = 1e-3, 1e-4

        # Initialization function of angle parameters
        angle0 = 2.0*math.pi*random.random()
        def init_angles(m):
            if type(m) == OrthonormalTransform:
                torch.nn.init.constant_(m.angles,angle0)       

        # Parameters
        stride_ = stride
        ovlpfactor_ = ovlpfactor
        isNoDcLeakage = nodcleakage 
        nlevels_ = 0
        nSamples = 8
        nComponents = 1
        nDecs = stride_[Direction.VERTICAL]*stride_[Direction.HORIZONTAL]
        nrows = height//stride_[Direction.VERTICAL]
        ncols = width//stride_[Direction.HORIZONTAL]

        # Coefficients nSamples x nRows x nCols x nDecs
        X = torch.randn(nSamples,nrows,ncols,nDecs,dtype=datatype,device=device,requires_grad=True)

        # nSamples x nComponents x height x width
        dLdZ = torch.randn(nSamples,nComponents,height,width,dtype=datatype,device=device)

        # Instantiation of target class
        network = LsunSynthesis2dNetwork(
            input_size=[height,width],
            stride=stride_,
            overlapping_factor=ovlpfactor_,
            number_of_levels=nlevels_,
            no_dc_leakage=isNoDcLeakage,
            dtype=datatype
        )
        network = network.to(device)

        # Initialization of angle parameters
        network.apply(init_angles)

        # Expected values
        adjoint = network.T           
        expctddLdX = adjoint(dLdZ)
        
        # Actual values
        Z = network(X)
        Z.backward(dLdZ,retain_graph=True)
        actualdLdX = X.grad
      
        # Evaluation
        self.assertEqual(actualdLdX.dtype,datatype)
        self.assertTrue(torch.allclose(actualdLdX,expctddLdX,rtol=rtol,atol=atol))
        self.assertTrue(Z.requires_grad)

    @parameterized.expand(
        list(itertools.product(nlevels,nodcleakage,datatype,usegpu))
    )
    def testBackwardGrayScaleMultiLevels(self,
        nlevels,nodcleakage,datatype,usegpu):
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else:
                print('No GPU device was detected.')
                return
        else:
            device = torch.device("cpu")
        rtol, atol = 1e-3, 1e-4

        # Initialization function of angle parameters
        angle0 = 2.0*math.pi*random.random()
        def init_angles(m):
            if type(m) == OrthonormalTransform:
                torch.nn.init.constant_(m.angles,angle0)       

        # Parameters
        stride_ = [2, 2]
        ovlpfactor_ = [3, 3]
        isNoDcLeakage = nodcleakage 
        nlevels_ = nlevels
        nSamples = 4
        nComponents = 1
        nDecs = stride_[Direction.VERTICAL]*stride_[Direction.HORIZONTAL]
        nrows = 2
        ncols = 3
        height_ = nrows*(stride_[Direction.VERTICAL]**nlevels_)
        width_ = ncols*(stride_[Direction.HORIZONTAL]**nlevels_) 

        # Coefficients nSamples x nRows x nCols x nDecs
        nrows_ = nrows
        ncols_ = ncols
        X = []
        for iLevel in range(1,nlevels_+1):
            if iLevel == 1:
                X.append(torch.randn(nSamples,nrows_,ncols_,1,dtype=datatype,device=device,requires_grad=True)) 
            X.append(torch.randn(nSamples,nrows_,ncols_,nDecs-1,dtype=datatype,device=device,requires_grad=True))     
            nrows_ *= stride_[Direction.VERTICAL]
            ncols_ *= stride_[Direction.HORIZONTAL]
        X = tuple(X)

        # nSamples x nComponents x height x width
        dLdZ = torch.randn(nSamples,nComponents,height_,width_,dtype=datatype,device=device)

        # Instantiation of target class
        network = LsunSynthesis2dNetwork(
            input_size=[height_,width_],
            stride=stride_,
            overlapping_factor=ovlpfactor_,
            number_of_levels=nlevels_,
            no_dc_leakage=isNoDcLeakage,
            dtype=datatype
        )
        network = network.to(device)

        # Initialization of angle parameters
        network.apply(init_angles)

        # Expected values
        adjoint = network.T           
        expctddLdX = adjoint(dLdZ)
        
        # Actual values
        Z = network(X)
        Z.backward(dLdZ,retain_graph=True)
        actualdLdX = []
        for iCh in range(len(X)):
            actualdLdX.append(X[iCh].grad)

      
        # Evaluation
        for iCh in range(len(X)):
            #print('actual(',iCh,'): ',actualdLdX[iCh].shape)
            #print('expctd(',iCh,'): ',expctddLdX[iCh].shape)
            self.assertEqual(actualdLdX[iCh].dtype,datatype)
            self.assertTrue(torch.allclose(actualdLdX[iCh],expctddLdX[iCh],rtol=rtol,atol=atol))
        self.assertTrue(Z.requires_grad)

"""
    Local functions
"""

def permuteIdctCoefs_(x,block_size):
    coefs = x.view(-1,block_size[Direction.VERTICAL]*block_size[Direction.HORIZONTAL]) # x.view(-1,math.prod(block_size)) 
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

def block_butterfly_(X,nchs):
    ps = nchs[0]
    Xs = X[:,:,:,:ps]
    Xa = X[:,:,:,ps:]
    return torch.cat((Xs+Xa,Xs-Xa),dim=-1)

def block_shift_(X,nchs,target,shift):
    ps = nchs[0]
    if target == 0: # Difference channel
        X[:,:,:,ps:] = torch.roll(X[:,:,:,ps:],shifts=tuple(shift),dims=(0,1,2,3))
    else: # Sum channel
        X[:,:,:,:ps] = torch.roll(X[:,:,:,:ps],shifts=tuple(shift),dims=(0,1,2,3))
    return X

def intermediate_rotation_(X,nchs,R):
    #Y = X.clone()
    ps,pa = nchs
    nSamples = X.size(0)
    nrows = X.size(1)
    ncols = X.size(2)
    #Za = R @ X[:,:,:,ps:].view(-1,pa).T 
    #Y[:,:,:,ps:] = Za.T.view(nSamples,nrows,ncols,pa)
    #return Y
    Y = X.clone()
    for iSample in range(nSamples):
        Ya = Y[iSample,:,:,ps:].view(-1,pa)
        for iblk in range(nrows*ncols):
            Ya[iblk,:] = R[iblk,:,:] @ Ya[iblk,:]
        Y[iSample,:,:,ps:] = Ya.clone().view(nrows,ncols,pa)
    return Y

if __name__ == '__main__':
    unittest.main(failfast=True)

    """
    # Create a test suite
    suite = unittest.TestSuite()

    # Add specific test methods to the suite

    of = [ 1159 ]
           
    for i in of:
        suite.addTest(LsunSynthesis2dNetworkTestCase('testBackwardGrayScale_{:02d}'.format(i)))
                                                    
    # Create a test runner
    runner = unittest.TextTestRunner()

    # Run the tests
    runner.run(suite)
    """
    