import itertools
import unittest
from parameterized import parameterized
import torch
import torch.nn as nn
import torch_dct as dct

import math
import random
from lsunAnalysis2dNetwork import LsunAnalysis2dNetwork
from lsunUtility import Direction, OrthonormalMatrixGenerationSystem
from orthonormalTransform import OrthonormalTransform
from lsunLayerExceptions import InvalidOverlappingFactor, InvalidNoDcLeakage, InvalidNumberOfLevels, InvalidStride, InvalidInputSize

stride = [ [2, 1], [1, 2], [2, 2], [2, 4], [4, 1], [4, 4] ]
ovlpfactor = [ [1, 1], [3, 3], [5, 5], [1, 3], [3, 1] ]
datatype = [ torch.float, torch.double ]
height = [ 8, 16, 32 ]
width = [ 8, 16, 32 ]
nodcleakage = [ False, True ]
#nlevels = [ 1, 2, 3 ]
usegpu = [ True, False ]

class LsunAnalysis2dNetworkTestCase(unittest.TestCase):
    """
    LSUNANLAYSIS2DNETWORKTESTCASE Test cases for LsunAnalysis2dNetwork
    
    Requirements: Requirements: Python 3.10/11.x, PyTorch 2.3.x
    
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
        network = LsunAnalysis2dNetwork(
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
            stride, height, width, datatype, usegpu):
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
        # Source (nSamples x nComponents x (Stride[0]xnRows) x (Stride[1]xnCols))
        X = torch.rand(nSamples,nComponents,height,width,dtype=datatype,device=device,requires_grad=True)
        
        # Expected values
        nrows = int(math.ceil(height/stride[Direction.VERTICAL])) #.astype(int)
        ncols = int(math.ceil(width/stride[Direction.HORIZONTAL])) #.astype(int)
        # Block DCT (nSamples x nComponents x nrows x ncols) x decV x decH
        arrayshape = stride.copy()
        arrayshape.insert(0,-1)
        Y = dct.dct_2d(X.view(arrayshape),norm='ortho')
        # Rearrange the DCT Coefs. (nSamples x nComponents x nrows x ncols) x (decV x decH)
        A = permuteDctCoefs_(Y)
        V = A.view(nSamples,nrows,ncols,nDecs)
        # nSamplex x nRows x nCols x nChs
        ps,pa = int(math.ceil(nDecs/2.)), int(math.floor(nDecs/2.))        
        W0 = torch.eye(ps,dtype=datatype).to(device)
        U0 = torch.eye(pa,dtype=datatype).to(device)
        Zsa = torch.zeros(nDecs,nrows*ncols*nSamples,dtype=datatype)
        Zsa = Zsa.to(device)        
        Ys = V[:,:,:,:ps].view(-1,ps).T
        Zsa[:ps,:] = W0 @ Ys
        if pa > 0:
            Ya = V[:,:,:,ps:].view(-1,pa).T
            Zsa[ps:,:] = U0 @ Ya
        expctdZ = Zsa.T.view(nSamples,nrows,ncols,nDecs)

        # Instantiation of target class
        network = LsunAnalysis2dNetwork(
                input_size=[height,width],
                stride=stride
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
            LsunAnalysis2dNetwork(
                overlapping_factor = [ ovlpfactor[Direction.VERTICAL]+1, ovlpfactor[Direction.HORIZONTAL] ],
                stride = stride
            )

        with self.assertRaises(InvalidOverlappingFactor):
            LsunAnalysis2dNetwork(
                overlapping_factor = [ ovlpfactor[Direction.VERTICAL], ovlpfactor[Direction.HORIZONTAL]+1 ],
                stride = stride
            )

        with self.assertRaises(InvalidOverlappingFactor):
            LsunAnalysis2dNetwork(
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
            LsunAnalysis2dNetwork(
                stride = stride,
                overlapping_factor = ovlpfactor,
                no_dc_leakage = no_dc_leakage
            )

        no_dc_leakage = 1
        with self.assertRaises(InvalidNoDcLeakage):
            LsunAnalysis2dNetwork(
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
            LsunAnalysis2dNetwork(
                stride = stride,
                overlapping_factor = ovlpfactor,
                number_of_levels = nlevels
            )

        nlevels = 0.5
        with self.assertRaises(InvalidNumberOfLevels):
            LsunAnalysis2dNetwork(
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
            LsunAnalysis2dNetwork(
                stride = stride,
                overlapping_factor = ovlpfactor
            )

        stride = [ 1, 3 ]
        with self.assertRaises(InvalidStride):
            LsunAnalysis2dNetwork(
                stride = stride,
                overlapping_factor = ovlpfactor
            )

        stride = [ 3, 1 ]
        with self.assertRaises(InvalidStride):
            LsunAnalysis2dNetwork(
                stride = stride,
                overlapping_factor = ovlpfactor
            )

        stride = [ 3, 3 ]
        with self.assertRaises(InvalidStride):
            LsunAnalysis2dNetwork(
                stride = stride,
                overlapping_factor = ovlpfactor
            )

    @parameterized.expand(
        list(itertools.product(stride))
    )
    def testInputSize(self,stride):
        input_size = [ 2*stride[Direction.VERTICAL]+1, 2*stride[Direction.HORIZONTAL]+1 ]
        with self.assertRaises(InvalidInputSize):
            LsunAnalysis2dNetwork(
                input_size = input_size,
                stride = stride
            )
    
        input_size = [ -2*stride[Direction.VERTICAL], -2*stride[Direction.HORIZONTAL] ]
        with self.assertRaises(InvalidInputSize):
            LsunAnalysis2dNetwork(
                input_size = input_size,
                stride = stride
            )

    @parameterized.expand(
        list(itertools.product(stride,height,width,datatype,usegpu))
    )
    def testForwardGrayScaleWithInitilization(self,
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
        W0 = genW(angles=angles[:,:nAnglesH])
        U0 = genU(angles=angles[:,nAnglesH:])    

        # Source (nSamples x nComponents x (Stride[0]xnRows) x (Stride[1]xnCols))
        X = torch.rand(nSamples,nComponents,height,width,dtype=datatype,device=device,requires_grad=True)

        # Expected values
        # Block DCT (nSamples x nComponents x nrows x ncols) x decV x decH)
        arrayshape = stride.copy()
        arrayshape.insert(0,-1)
        Y = dct.dct_2d(X.view(arrayshape),norm='ortho')
        Y = Y.to(device)
        # Rearrange the DCT Coefs. (nSamples x nComponents x nrows x ncols) x (decV x decH)
        A = permuteDctCoefs_(Y)
        # nSamples x nRows x nCols x nDecs        
        V = A.view(nSamples,nrows,ncols,nDecs)

        expctdZ = torch.zeros_like(V)
        for iSample in range(nSamples):
            Vi = V[iSample,:,:,:].clone()
            Ys = Vi[:,:,:ps].view(-1,ps)
            Ya = Vi[:,:,ps:].view(-1,pa)
            for iblk in range(nrows*ncols):
                Ys[iblk,:] = W0[iblk,:] @ Ys[iblk,:]
                Ya[iblk,:] = U0[iblk,:] @ Ya[iblk,:]
            Yi = torch.cat((Ys,Ya),dim=1).view(nrows,ncols,nDecs)
            expctdZ[iSample,:,:,:] = Yi

        # Instantiation of target class
        network = LsunAnalysis2dNetwork(
                input_size=[height,width],
                stride=stride,
                no_dc_leakage=isNoDcLeakage
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
    def testForwardGrayScaleOvlpFactorDefault(self,
            stride, ovlpfactor, height, width, nodcleakage, datatype, usegpu):
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

        # Source (nSamples x nComponents x (Stride[0]xnRows) x (Stride[1]xnCols))
        X = torch.rand(nSamples,nComponents,height,width,dtype=datatype,device=device,requires_grad=True)

        # Expected values
        # Block DCT (nSamples x nComponents x nrows x ncols) x decV x decH
        arrayshape = stride.copy()
        arrayshape.insert(0,-1)
        Y = dct.dct_2d(X.view(arrayshape),norm='ortho')
        Y = Y.to(device)
        # Rearrange the DCT Coefs. (nSamples x nComponents x nrows x ncols) x (decV x decH)
        A = permuteDctCoefs_(Y)
        # nSamples x nRows x nCols x nDecs
        expctdZ = A.view(nSamples,nrows,ncols,nDecs)

        # Instantiation of target class
        network = LsunAnalysis2dNetwork(
                input_size=[height,width],
                stride=stride,
                overlapping_factor=ovlpfactor,
                no_dc_leakage=isNoDcLeakage
            )
        network = network.to(device)

        # Actual values
        with torch.no_grad():
            actualZ = network.forward(X)

        # Evaluation
        self.assertEqual(actualZ.dtype,datatype) 
        msg = 'stride=%s, ovlpfactor=%s, height=%d, width=%d, nodcleakage=%s, datatype=%s' % (stride,ovlpfactor,height,width,nodcleakage,datatype)
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol),msg=msg)
        self.assertFalse(actualZ.requires_grad)

    @parameterized.expand(
        list(itertools.product(stride,height,width,datatype,usegpu))
    )
    def testForwardGrayScaleOvlpFactor33(self,
            stride, height, width, datatype, usegpu):
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
        ovlpFactor = [ 3, 3 ]
        isNoDcLeakage = False # TODO
        nSamples = 8
        nComponents = 1
        nDecs = stride[Direction.VERTICAL]*stride[Direction.HORIZONTAL]
        nrows = height//stride[Direction.VERTICAL]
        ncols = width//stride[Direction.HORIZONTAL]

        # Source (nSamples x nComponents x (Stride[0]xnRows) x (Stride[1]xnCols))
        X = torch.rand(nSamples,nComponents,height,width,dtype=datatype,device=device,requires_grad=True)
        nAngles = (nDecs-2)*nDecs//4
        angles = angle0*torch.ones(nrows*ncols,nAngles,dtype=datatype,device=device)

        # Expected values
        ps = math.ceil(nDecs/2)
        pa = math.floor(nDecs/2)
        nchs = (ps, pa)
        nAnglesH = nAngles//2
        W0 = genW(angles[:,:nAnglesH]) # TODO: Handling no_dc_leakage
        U0 = genU(angles[:,nAnglesH:])
        Uh1 = -genU(angles[:,nAnglesH:])
        Uh2 = -genU(angles[:,nAnglesH:])        
        Uv1 = -genU(angles[:,nAnglesH:])
        Uv2 = -genU(angles[:,nAnglesH:])        

        # Block DCT (nSamples x nComponents x nrows x ncols) x decV x decH
        arrayshape = stride.copy()
        arrayshape.insert(0,-1)
        Y = dct.dct_2d(X.view(arrayshape),norm='ortho')
        Y = Y.to(device)
        # Rearrange the DCT Coefs. (nSamples x nComponents x nrows x ncols) x (decV x decH)
        A = permuteDctCoefs_(Y)
        # nSamples x nRows x nCols x nDecs
        Z = A.view(nSamples,nrows,ncols,nDecs)
        for iSample in range(nSamples):
            Vi = Z[iSample,:,:,:].clone()
            Ys = Vi[:,:,:ps].view(-1,ps)
            Ya = Vi[:,:,ps:].view(-1,pa)
            for iblk in range(nrows*ncols):
                Ys[iblk,:] = W0[iblk,:] @ Ys[iblk,:]
                Ya[iblk,:] = U0[iblk,:] @ Ya[iblk,:]
            Yi = torch.cat((Ys,Ya),dim=1).view(nrows,ncols,nDecs)
            Z[iSample,:,:,:] = Yi

        # Horizontal atom extention
        for ordH in range(ovlpFactor[Direction.HORIZONTAL]//2):
            Z = block_butterfly_(Z,nchs)
            Z = block_shift_(Z,nchs,0,[0,0,1,0]) # target=diff, shift=right
            Z = block_butterfly_(Z,nchs)/2.
            Z = intermediate_rotation_(Z,nchs,Uh1)
            Z = block_butterfly_(Z,nchs)
            Z = block_shift_(Z,nchs,1,[0,0,-1,0]) # target=sum, shift=left
            Z = block_butterfly_(Z,nchs)/2.
            Z = intermediate_rotation_(Z,nchs,Uh2)
        # Vertical atom extention
        for ordV in range(ovlpFactor[Direction.VERTICAL]//2):
            Z = block_butterfly_(Z,nchs)
            Z = block_shift_(Z,nchs,0,[0,1,0,0]) # target=diff, shift=down
            Z = block_butterfly_(Z,nchs)/2.
            Z = intermediate_rotation_(Z,nchs,Uv1)
            Z = block_butterfly_(Z,nchs)
            Z = block_shift_(Z,nchs,1,[0,-1,0,0]) # target=sum, shift=up
            Z = block_butterfly_(Z,nchs)/2.
            Z = intermediate_rotation_(Z,nchs,Uv2)
        expctdZ = Z

        # Instantiation of target class
        network = LsunAnalysis2dNetwork(
                input_size=[height,width],
                stride=stride,
                overlapping_factor=ovlpFactor,
                no_dc_leakage=isNoDcLeakage
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

"""
    @parameterized.expand(
        list(itertools.product(nchs,stride,height,width,datatype))
    )
    def testForwardGrayScaleOvlpFactor2Ord20(self,
            nchs, stride, height, width, datatype):
        rtol,atol = 1e-3,1e-6
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")        

        # Parameters
        # Source (nSamples x nComponents x (Stride[0]xnRows) x (Stride[1]xnCols))
        X = torch.rand(nSamples,nComponents,height,width,dtype=datatype,device=device,requires_grad=True)
        
        # Expected values
        # Block DCT (nSamples x nComponents x nrows x ncols) x decV x decH
        arrayshape = stride.copy()
        arrayshape.insert(0,-1)
        Y = dct_2d(X.view(arrayshape))
        # Rearrange the DCT Coefs. (nSamples x nComponents x nrows x ncols) x (decV x decH)
        A = permuteDctCoefs_(Y)
        V = A.view(nSamples,nrows,ncols,nDecs)
        # nSamples x nRows x nCols x nChs
        ps, pa = nchs
        # Initial rotation
        W0 = torch.eye(ps,dtype=datatype).to(device)
        U0 = torch.eye(pa,dtype=datatype).to(device)
        ms,ma = int(math.ceil(nDecs/2.)), int(math.floor(nDecs/2.))        
        Zsa = torch.zeros(nChsTotal,nrows*ncols*nSamples,dtype=datatype)    
        Zsa = Zsa.to(device)    
        Ys = V[:,:,:,:ms].view(-1,ms).T
        Zsa[:ps,:] = W0[:,:ms] @ Ys
        if ma > 0:
            Ya = V[:,:,:,ms:].view(-1,ma).T
            Zsa[ps:,:] = U0[:,:ma] @ Ya
        Z = Zsa.T.view(nSamples,nrows,ncols,nChsTotal)
        # Vertical atom extention
        Z = block_butterfly(Z,nchs)
        Z = block_shift(Z,nchs,0,[0,1,0,0]) # target=diff, shift=down
        Z = block_butterfly(Z,nchs)/2.
        Uv1 = -torch.eye(pa,dtype=datatype).to(device)
        Z = intermediate_rotation(Z,nchs,Uv1)
        Z = block_butterfly(Z,nchs)
        Z = block_shift(Z,nchs,1,[0,-1,0,0]) # target=sum, shift=up
        Z = block_butterfly(Z,nchs)/2.
        Uv2 = -torch.eye(pa,dtype=datatype).to(device)
        Z = intermediate_rotation(Z,nchs,Uv2)
        expctdZ = Z

        # Instantiation of target class
        network = LsunAnalysis2dNetwork(
                number_of_channels=nchs,
                stride=stride,
                polyphase_order=ppOrd
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
        list(itertools.product(nchs,stride,height,width,datatype))
    )
    def testForwardGrayScaleOrd02(self,
            nchs, stride, height, width, datatype):
        rtol,atol = 1e-3,1e-6
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")      

        # Parameters
        ppOrd = [ 0, 2 ]
        nSamples = 8
        nComponents = 1
        nDecs = stride[0]*stride[1] #math.prod(stride)
        nChsTotal = sum(nchs)
        # Source (nSamples x nComponents x (Stride[0]xnRows) x (Stride[1]xnCols))
        X = torch.rand(nSamples,nComponents,height,width,dtype=datatype,device=device,requires_grad=True)
        
        # Expected values
        nrows = int(math.ceil(height/stride[Direction.VERTICAL])) #.astype(int)
        ncols = int(math.ceil(width/stride[Direction.HORIZONTAL])) #.astype(int)
        # Block DCT (nSamples x nComponents x nrows x ncols) x decV x decH
        arrayshape = stride.copy()
        arrayshape.insert(0,-1)
        Y = dct_2d(X.view(arrayshape))
        # Rearrange the DCT Coefs. (nSamples x nComponents x nrows x ncols) x (decV x decH)
        A = permuteDctCoefs_(Y)
        V = A.view(nSamples,nrows,ncols,nDecs)
        # nSamples x nRows x nCols x nChs
        ps, pa = nchs
        # Initial rotation
        W0 = torch.eye(ps,dtype=datatype).to(device)
        U0 = torch.eye(pa,dtype=datatype).to(device)
        ms,ma = int(math.ceil(nDecs/2.)), int(math.floor(nDecs/2.))        
        Zsa = torch.zeros(nChsTotal,nrows*ncols*nSamples,dtype=datatype)        
        Zsa = Zsa.to(device)
        Ys = V[:,:,:,:ms].view(-1,ms).T
        Zsa[:ps,:] = W0[:,:ms] @ Ys
        if ma > 0:
            Ya = V[:,:,:,ms:].view(-1,ma).T
            Zsa[ps:,:] = U0[:,:ma] @ Ya
        Z = Zsa.T.view(nSamples,nrows,ncols,nChsTotal)
        # Horizontal atom extention
        Z = block_butterfly(Z,nchs)
        Z = block_shift(Z,nchs,0,[0,0,1,0]) # target=diff, shift=right
        Z = block_butterfly(Z,nchs)/2.
        Uh1 = -torch.eye(pa,dtype=datatype).to(device)
        Z = intermediate_rotation(Z,nchs,Uh1)
        Z = block_butterfly(Z,nchs)
        Z = block_shift(Z,nchs,1,[0,0,-1,0]) # target=sum, shift=left
        Z = block_butterfly(Z,nchs)/2.
        Uh2 = -torch.eye(pa,dtype=datatype).to(device)
        Z = intermediate_rotation(Z,nchs,Uh2)
        expctdZ = Z

        # Instantiation of target class
        network = LsunAnalysis2dNetwork(
                number_of_channels=nchs,
                stride=stride,
                polyphase_order=ppOrd
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
        list(itertools.product(nchs,stride,ppord,datatype))
    )
    def testForwardGrayScaleOverlapping(self,
            nchs, stride, ppord, datatype):
        rtol,atol = 1e-3,1e-6
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")  

        # Parameters
        height = 8
        width = 16
        ppOrd = ppord
        nSamples = 8
        nComponents = 1
        nDecs = stride[0]*stride[1] #math.prod(stride)
        nChsTotal = sum(nchs)
        # Source (nSamples x nComponents x (Stride[0]xnRows) x (Stride[1]xnCols))
        X = torch.rand(nSamples,nComponents,height,width,dtype=datatype,device=device,requires_grad=True)
        
        # Expected values
        nrows = int(math.ceil(height/stride[Direction.VERTICAL])) #.astype(int)
        ncols = int(math.ceil(width/stride[Direction.HORIZONTAL])) #.astype(int)
        # Block DCT (nSamples x nComponents x nrows x ncols) x decV x decH
        arrayshape = stride.copy()
        arrayshape.insert(0,-1)
        Y = dct_2d(X.view(arrayshape))
        # Rearrange the DCT Coefs. (nSamples x nComponents x nrows x ncols) x (decV x decH)
        A = permuteDctCoefs_(Y)
        V = A.view(nSamples,nrows,ncols,nDecs)
        # nSamples x nRows x nCols x nChs
        ps, pa = nchs
        # Initial rotation
        W0 = torch.eye(ps,dtype=datatype).to(device)
        U0 = torch.eye(pa,dtype=datatype).to(device)
        ms,ma = int(math.ceil(nDecs/2.)), int(math.floor(nDecs/2.))        
        Zsa = torch.zeros(nChsTotal,nrows*ncols*nSamples,dtype=datatype)  
        Zsa = Zsa.to(device)      
        Ys = V[:,:,:,:ms].view(-1,ms).T
        Zsa[:ps,:] = W0[:,:ms] @ Ys
        if ma > 0:
            Ya = V[:,:,:,ms:].view(-1,ma).T
            Zsa[ps:,:] = U0[:,:ma] @ Ya
        Z = Zsa.T.view(nSamples,nrows,ncols,nChsTotal)
        # Horizontal atom extention
        for ordH in range(int(ppOrd[Direction.HORIZONTAL]/2)):
            Z = block_butterfly(Z,nchs)
            Z = block_shift(Z,nchs,0,[0,0,1,0]) # target=diff, shift=right
            Z = block_butterfly(Z,nchs)/2.
            Uh1 = -torch.eye(pa,dtype=datatype).to(device)
            Z = intermediate_rotation(Z,nchs,Uh1)
            Z = block_butterfly(Z,nchs)
            Z = block_shift(Z,nchs,1,[0,0,-1,0]) # target=sum, shift=left
            Z = block_butterfly(Z,nchs)/2.
            Uh2 = -torch.eye(pa,dtype=datatype).to(device)
            Z = intermediate_rotation(Z,nchs,Uh2)
        # Vertical atom extention
        for ordV in range(int(ppOrd[Direction.VERTICAL]/2)):
            Z = block_butterfly(Z,nchs)
            Z = block_shift(Z,nchs,0,[0,1,0,0]) # target=diff, shift=down
            Z = block_butterfly(Z,nchs)/2.
            Uv1 = -torch.eye(pa,dtype=datatype).to(device)
            Z = intermediate_rotation(Z,nchs,Uv1)
            Z = block_butterfly(Z,nchs)
            Z = block_shift(Z,nchs,1,[0,-1,0,0]) # target=sum, shift=up
            Z = block_butterfly(Z,nchs)/2.
            Uv2 = -torch.eye(pa,dtype=datatype).to(device)
            Z = intermediate_rotation(Z,nchs,Uv2)
        expctdZ = Z

        # Instantiation of target class
        network = LsunAnalysis2dNetwork(
                number_of_channels=nchs,
                stride=stride,
                polyphase_order=ppOrd
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
        list(itertools.product(nchs,stride,ppord,datatype))
    )
    def testForwardGrayScaleOverlappingWithInitialization(self,
            nchs, stride, ppord, datatype):
        rtol,atol = 1e-3,1e-6
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")                
        gen = OrthonormalMatrixGenerationSystem(dtype=datatype)

        # Initialization function of angle parameters
        angle0 = 2.0*math.pi*random.random()
        def init_angles(m):
            if type(m) == OrthonormalTransform:
                torch.nn.init.constant_(m.angles,angle0)

        # Parameters
        nVm = 0
        height = 8
        width = 16
        ppOrd = ppord
        nSamples = 8
        nComponents = 1
        nDecs = stride[0]*stride[1] #math.prod(stride)
        nChsTotal = sum(nchs)
        # Source (nSamples x nComponents x (Stride[0]xnRows) x (Stride[1]xnCols))
        X = torch.rand(nSamples,nComponents,height,width,dtype=datatype,device=device,requires_grad=True)
        
        # Expected values
        nrows = int(math.ceil(height/stride[Direction.VERTICAL])) #.astype(int)
        ncols = int(math.ceil(width/stride[Direction.HORIZONTAL])) #.astype(int)
        # Block DCT (nSamples x nComponents x nrows x ncols) x decV x decH
        arrayshape = stride.copy()
        arrayshape.insert(0,-1)
        Y = dct_2d(X.view(arrayshape))
        # Rearrange the DCT Coefs. (nSamples x nComponents x nrows x ncols) x (decV x decH)
        A = permuteDctCoefs_(Y)
        V = A.view(nSamples,nrows,ncols,nDecs)
        # nSamples x nRows x nCols x nChs
        ps, pa = nchs
        # Initial rotation
        angles = angle0*torch.ones(int((nChsTotal-2)*nChsTotal/4)).to(device) #,dtype=datatype)
        nAngsW = int(len(angles)/2)
        angsW,angsU = angles[:nAngsW],angles[nAngsW:]
        W0,U0 = gen(angsW),gen(angsU)        
        ms,ma = int(math.ceil(nDecs/2.)), int(math.floor(nDecs/2.))        
        Zsa = torch.zeros(nChsTotal,nrows*ncols*nSamples,dtype=datatype)
        Zsa = Zsa.to(device)        
        Ys = V[:,:,:,:ms].view(-1,ms).T
        Zsa[:ps,:] = W0[:,:ms] @ Ys
        if ma > 0:
            Ya = V[:,:,:,ms:].view(-1,ma).T
            Zsa[ps:,:] = U0[:,:ma] @ Ya
        Z = Zsa.T.view(nSamples,nrows,ncols,nChsTotal)
        # Horizontal atom extention
        for ordH in range(int(ppOrd[Direction.HORIZONTAL]/2)):
            Z = block_butterfly(Z,nchs)
            Z = block_shift(Z,nchs,0,[0,0,1,0]) # target=diff, shift=right
            Z = block_butterfly(Z,nchs)/2.
            Uh1 = -gen(angsU)
            Z = intermediate_rotation(Z,nchs,Uh1)
            Z = block_butterfly(Z,nchs)
            Z = block_shift(Z,nchs,1,[0,0,-1,0]) # target=sum, shift=left
            Z = block_butterfly(Z,nchs)/2.
            Uh2 = -gen(angsU)
            Z = intermediate_rotation(Z,nchs,Uh2)
        # Vertical atom extention
        for ordV in range(int(ppOrd[Direction.VERTICAL]/2)):
            Z = block_butterfly(Z,nchs)
            Z = block_shift(Z,nchs,0,[0,1,0,0]) # target=diff, shift=down
            Z = block_butterfly(Z,nchs)/2.
            Uv1 = -gen(angsU)
            Z = intermediate_rotation(Z,nchs,Uv1)
            Z = block_butterfly(Z,nchs)
            Z = block_shift(Z,nchs,1,[0,-1,0,0]) # target=sum, shift=up
            Z = block_butterfly(Z,nchs)/2.
            Uv2 = -gen(angsU)
            Z = intermediate_rotation(Z,nchs,Uv2)
        expctdZ = Z

        # Instantiation of target class
        network = LsunAnalysis2dNetwork(
                number_of_channels=nchs,
                stride=stride,
                polyphase_order=ppOrd,
                number_of_vanishing_moments=nVm
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
        list(itertools.product(nchs,stride,ppord,datatype))
    )
    def testForwardGrayScaleOverlappingWithNoDcLeakage(self,
            nchs, stride, ppord, datatype):
        rtol,atol = 1e-3,1e-6
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")                
        gen = OrthonormalMatrixGenerationSystem(dtype=datatype)

        # Initialization function of angle parameters
        def init_angles(m):
            if type(m) == OrthonormalTransform:
                torch.nn.init.normal_(m.angles)

        # Parameters
        nVm = 1
        height = 8
        width = 16
        ppOrd = ppord
        nSamples = 8
        nComponents = 1
        nDecs = stride[0]*stride[1] #math.prod(stride)
        nChsTotal = sum(nchs)
        # Source (nSamples x nComponents x (Stride[0]xnRows) x (Stride[1]xnCols))
        X = torch.ones(nSamples,nComponents,height,width,dtype=datatype,device=device,requires_grad=True)
        
        # Expected values
        nrows = int(math.ceil(height/stride[Direction.VERTICAL])) #.astype(int)
        ncols = int(math.ceil(width/stride[Direction.HORIZONTAL])) #.astype(int)
        # nSamples x nRows x nCols x nChs
        expctdZ = torch.cat(
                    [math.sqrt(nDecs)*torch.ones(nSamples,nrows,ncols,1,dtype=datatype,device=device),
                    torch.zeros(nSamples,nrows,ncols,nChsTotal-1,dtype=datatype,device=device)],
                    dim=3)

        # Instantiation of target class
        network = LsunAnalysis2dNetwork(
                number_of_channels=nchs,
                stride=stride,
                polyphase_order=ppOrd,
                number_of_vanishing_moments=nVm
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
        list(itertools.product(nchs,stride,nvm,nlevels,datatype))
    )
    def testForwardGrayScaleMultiLevels(self,
            nchs, stride, nvm, nlevels, datatype):
        rtol,atol = 1e-3,1e-6
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")               
        gen = OrthonormalMatrixGenerationSystem(dtype=datatype)

        # Initialization function of angle parameters
        angle0 = 2.0*math.pi*random.random()
        def init_angles(m):
            if type(m) == OrthonormalTransform:
                torch.nn.init.constant_(m.angles,angle0)

        # Parameters
        nVm = nvm
        height = 8 
        width = 16
        ppOrd = [ 2, 2 ]
        nSamples = 8
        nComponents = 1
        nDecs = stride[0]*stride[1] #math.prod(stride)
        nChsTotal = sum(nchs)

        # Source (nSamples x nComponents x ((Stride[0]**nlevels) x nRows) x ((Stride[1]**nlevels) x nCols))
        X = torch.randn(nSamples,nComponents,height,width,dtype=datatype,device=device,requires_grad=True)
        
        # Expected values
        nrows = int(math.ceil(height/(stride[Direction.VERTICAL]))) #.astype(int)
        ncols = int(math.ceil(width/(stride[Direction.HORIZONTAL]))) #.astype(int)
        # Block DCT (nSamples x nComponents x nrows x ncols) x decV x decH
        arrayshape = stride.copy() 
        arrayshape.insert(0,-1)
        # Multi-level decomposition
        coefs = []
        X_ = X
        for iStage in range(nlevels):
            iLevel = iStage+1
            Y = dct_2d(X_.view(arrayshape))
            # Rearrange the DCT Coefs. (nSamples x nComponents x nrows x ncols) x (decV x decH)
            A = permuteDctCoefs_(Y)
            V = A.view(nSamples,nrows,ncols,nDecs)
            # nSamples x nRows x nCols x nChs
            ps, pa = nchs
            # Initial rotation
            angles = angle0*torch.ones(int((nChsTotal-2)*nChsTotal/4)).to(device) #,dtype=datatype)
            nAngsW = int(len(angles)/2)
            angsW,angsU = angles[:nAngsW],angles[nAngsW:]
            if nVm > 0:
                angsW[:(ps-1)] = torch.zeros_like(angsW[:(ps-1)])
            W0,U0 = gen(angsW),gen(angsU)        
            ms,ma = int(math.ceil(nDecs/2.)), int(math.floor(nDecs/2.))        
            Zsa = torch.zeros(nChsTotal,nrows*ncols*nSamples,dtype=datatype)        
            Zsa = Zsa.to(device)
            Ys = V[:,:,:,:ms].view(-1,ms).T
            Zsa[:ps,:] = W0[:,:ms] @ Ys
            if ma > 0:
                Ya = V[:,:,:,ms:].view(-1,ma).T
                Zsa[ps:,:] = U0[:,:ma] @ Ya
            Z = Zsa.T.view(nSamples,nrows,ncols,nChsTotal)
            # Horizontal atom extention
            for ordH in range(int(ppOrd[Direction.HORIZONTAL]/2)):
                Z = block_butterfly(Z,nchs)
                Z = block_shift(Z,nchs,0,[0,0,1,0]) # target=diff, shift=right
                Z = block_butterfly(Z,nchs)/2.
                Uh1 = -gen(angsU)
                Z = intermediate_rotation(Z,nchs,Uh1)
                Z = block_butterfly(Z,nchs)
                Z = block_shift(Z,nchs,1,[0,0,-1,0]) # target=sum, shift=left
                Z = block_butterfly(Z,nchs)/2.
                Uh2 = -gen(angsU)
                Z = intermediate_rotation(Z,nchs,Uh2)
            # Vertical atom extention
            for ordV in range(int(ppOrd[Direction.VERTICAL]/2)):
                Z = block_butterfly(Z,nchs)
                Z = block_shift(Z,nchs,0,[0,1,0,0]) # target=diff, shift=down
                Z = block_butterfly(Z,nchs)/2.
                Uv1 = -gen(angsU)
                Z = intermediate_rotation(Z,nchs,Uv1)
                Z = block_butterfly(Z,nchs)
                Z = block_shift(Z,nchs,1,[0,-1,0,0]) # target=sum, shift=up
                Z = block_butterfly(Z,nchs)/2.
                Uv2 = -gen(angsU)
                Z = intermediate_rotation(Z,nchs,Uv2)
            # Xac
            coefs.insert(0,Z[:,:,:,1:])
            if iLevel < nlevels:
                X_ = Z[:,:,:,0].view(nSamples,nComponents,nrows,ncols)
                nrows = int(nrows/stride[Direction.VERTICAL])
                ncols = int(ncols/stride[Direction.HORIZONTAL])            
            else: # Xdc
                coefs.insert(0,Z[:,:,:,0])
        expctdZ = tuple(coefs)

        # Instantiation of target class
        network = LsunAnalysis2dNetwork(
                number_of_channels=nchs,
                stride=stride,
                polyphase_order=ppOrd,
                number_of_vanishing_moments=nVm,
                number_of_levels=nlevels
            )
        network = network.to(device)

        # Initialization of angle parameters
        network.apply(init_angles)

        # Actual values
        with torch.no_grad():
            actualZ = network.forward(X)

        # Evaluation
        for iStage in range(nlevels+1):
            self.assertEqual(actualZ[iStage].dtype,datatype)         
            self.assertTrue(torch.allclose(actualZ[iStage],expctdZ[iStage],rtol=rtol,atol=atol))
            self.assertFalse(actualZ[iStage].requires_grad) 


    @parameterized.expand(
        list(itertools.product(nchs,stride,nvm,nlevels,datatype))        
    )
    def testBackwardGrayScale(self,
        nchs,stride,nvm,nlevels,datatype):
        rtol,atol = 1e-3,1e-6
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")              

        # Initialization function of angle parameters
        def init_angles(m):
            if type(m) == OrthonormalTransform:
                torch.nn.init.zeros_(m.angles)             

        # Parameters
        nVm = nvm
        height = 8 
        width = 16
        ppOrd = [ 2, 2 ]
        nSamples = 8
        nrows = int(math.ceil(height/(stride[Direction.VERTICAL]**nlevels)))
        ncols = int(math.ceil(width/(stride[Direction.HORIZONTAL]**nlevels)))        
        nComponents = 1
        nDecs = stride[0]*stride[1] #math.prod(stride)
        nChsTotal = sum(nchs)

        # Source (nSamples x nComponents x ((Stride[0]**nlevels) x nRows) x ((Stride[1]**nlevels) x nCols))
        X = torch.randn(nSamples,nComponents,height,width,dtype=datatype,device=device,requires_grad=True)

        # Coefficients nSamples x nRows x nCols x nChsTotal
        nrows_ = nrows
        ncols_ = ncols
        dLdZ = []
        for iLevel in range(1,nlevels+1):
            if iLevel == 1:
                dLdZ.append(torch.randn(nSamples,nrows_,ncols_,dtype=datatype,device=device)) 
            dLdZ.append(torch.randn(nSamples,nrows_,ncols_,nChsTotal-1,dtype=datatype,device=device))     
            nrows_ *= stride[Direction.VERTICAL]
            ncols_ *= stride[Direction.HORIZONTAL]
        dLdZ = tuple(dLdZ)

        # Instantiation of target class
        network = LsunAnalysis2dNetwork(
                number_of_channels=nchs,
                stride=stride,
                polyphase_order=ppOrd,
                number_of_vanishing_moments=nVm,
                number_of_levels=nlevels
            ).to(device)

        # Initialization of angle parameters
        network.apply(init_angles)

        # Expected values
        adjoint = network.T
        expctddLdX = adjoint(dLdZ)
        
        # Actual values
        Z = network(X)
        for iCh in range(len(Z)):
            Z[iCh].backward(dLdZ[iCh],retain_graph=True)
        actualdLdX = X.grad

        # Evaluation
        self.assertEqual(actualdLdX.dtype,datatype)
        self.assertTrue(torch.allclose(actualdLdX,expctddLdX,rtol=rtol,atol=atol))
        for iCh in range(len(Z)):
            self.assertTrue(Z[iCh].requires_grad)
"""

def permuteDctCoefs_(x):
    cee = x[:,0::2,0::2].reshape(x.size(0),-1)
    coo = x[:,1::2,1::2].reshape(x.size(0),-1)
    coe = x[:,1::2,0::2].reshape(x.size(0),-1)
    ceo = x[:,0::2,1::2].reshape(x.size(0),-1)
    return torch.cat((cee,coo,coe,ceo),dim=-1)

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
        Y[iSample,:,:,ps:] = Ya.view(nrows,ncols,pa)
    return Y
    
if __name__ == '__main__':
    unittest.main()

    """
    # Create a test suite
    suite = unittest.TestSuite()

    # Add specific test methods to the suite
    suite.addTest(LsunAnalysis2dNetworkTestCase('testForwardGrayScaleOvlpFactor33Initial_215'))

    # Create a test runner
    runner = unittest.TextTestRunner()

    # Run the tests
    runner.run(suite)
    """