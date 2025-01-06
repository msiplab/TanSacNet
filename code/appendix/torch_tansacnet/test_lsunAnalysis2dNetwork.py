import itertools
import unittest
from parameterized import parameterized
import torch
import torch.nn as nn
import torch_dct as dct

import math
import random
from torch_tansacnet.lsunAnalysis2dNetwork import LsunAnalysis2dNetwork
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

class LsunAnalysis2dNetworkTestCase(unittest.TestCase):
    """
    LSUNANLAYSIS2DNETWORKTESTCASE Test cases for LsunAnalysis2dNetwork
    
    Requirements: Python 3.10-12.x, PyTorch 2.3/4.x
    
    Copyright (c) 2024, Shogo MURAMATSU
    
    All rights reserved.
    
    Contact address: Shogo MURAMATSU,
        Faculty of Engineering, Niigata University,
        8050 2-no-cho Ikarashi, Nishi-ku,
        Niigata, 950-2181, JAPAN
    
        https://www.eng.niigata-u.ac.jp/~msiplab/
    """

    @parameterized.expand(
        list(itertools.product(stride,ovlpfactor,height,width,nodcleakage,datatype))
    )
    def testConstructor(self, stride,ovlpfactor,height,width,nodcleakage,datatype):

        # Expcted values
        expctdStride = stride
        expctdOvlpFactor = ovlpfactor
        expctdInputSize = [ height, width ]
        expctdNoDcLeakage = nodcleakage
        expctdDtype = datatype
        expctdDevice = torch.device('cpu')       

        # Instantiation of target class
        network = LsunAnalysis2dNetwork(
            input_size = [ height, width ],
            stride = stride,
            overlapping_factor = ovlpfactor,
            no_dc_leakage = nodcleakage,
            dtype = datatype
        )

        # Actual values
        actualStride = network.stride
        actualOvlpFactor = network.overlapping_factor
        actualInputSize = network.input_size
        actualNoDcLeakage = network.no_dc_leakage 
        actualDtype = network.dtype
        actualDevice = network.device

        # Evaluation
        self.assertTrue(isinstance(network, nn.Module))
        self.assertEqual(actualStride,expctdStride)
        self.assertEqual(actualOvlpFactor,expctdOvlpFactor)
        self.assertEqual(actualInputSize,expctdInputSize)
        self.assertEqual(actualNoDcLeakage,expctdNoDcLeakage)       
        self.assertEqual(actualDtype,expctdDtype)     
        self.assertEqual(actualDevice,expctdDevice)    

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
        
        genW = OrthonormalMatrixGenerationSystem(device=device,dtype=datatype)
        genU = OrthonormalMatrixGenerationSystem(device=device,dtype=datatype)

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
                no_dc_leakage=isNoDcLeakage,
                dtype=datatype
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
        rtol,atol = 1e-4,1e-5

        genW = OrthonormalMatrixGenerationSystem(device=device,dtype=datatype)
        genU = OrthonormalMatrixGenerationSystem(device=device,dtype=datatype)

        # Initialization function of angle parameters
        angle0 = 2.0*math.pi*random.random()
        def init_angles(m):
            if type(m) == OrthonormalTransform:
                torch.nn.init.constant_(m.angles,angle0)        

        # Parameters
        ovlpFactor = [ 1, 1 ]
        isNoDcLeakage = nodcleakage # False # TODO: Handling no_dc_leakage
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
        #nchs = (ps, pa)
        nAnglesH = nAngles//2
        if isNoDcLeakage:
            angles[:,:(ps-1)] = 0
        W0 = genW(angles[:,:nAnglesH]) 
        U0 = genU(angles[:,nAnglesH:])
 
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
        expctdZ = Z

        # Instantiation of target class
        network = LsunAnalysis2dNetwork(
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

        relerr = torch.abs(actualZ-expctdZ)/torch.abs(expctdZ)
        abserr = torch.abs(actualZ-expctdZ)
        msg = 'relerr=%s, abserr=%s, stride=%s, ovlpfactor=%s, height=%d, width=%d, nodcleakage=%s, datatype=%s' % (relerr,abserr,stride,ovlpFactor,height,width,nodcleakage,datatype)

        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol),msg=msg)
        self.assertFalse(actualZ.requires_grad)

    @parameterized.expand(
        list(itertools.product(stride,height,width,nodcleakage,datatype,usegpu))
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
        rtol,atol = 1e-4,1e-5

        genW = OrthonormalMatrixGenerationSystem(device=device,dtype=datatype)
        genU = OrthonormalMatrixGenerationSystem(device=device,dtype=datatype)

        # Initialization function of angle parameters
        angle0 = 2.0*math.pi*random.random()
        def init_angles(m):
            if type(m) == OrthonormalTransform:
                torch.nn.init.constant_(m.angles,angle0)        

        # Parameters
        ovlpFactor = [ 3, 3 ]
        isNoDcLeakage = nodcleakage # False # TODO: Handling no_dc_leakage
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
        if isNoDcLeakage:
            angles[:,:(ps-1)] = 0
        W0 = genW(angles[:,:nAnglesH]) 
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
        #for ordH in range(ovlpFactor[Direction.HORIZONTAL]//2):
        Z = block_butterfly_(Z,nchs)
        Z = block_shift_(Z,nchs,0,[0,0,1,0]) # target=diff, shift=right
        Z = block_butterfly_(Z,nchs)/2.
        Z = intermediate_rotation_(Z,nchs,Uh1)
        Z = block_butterfly_(Z,nchs)
        Z = block_shift_(Z,nchs,1,[0,0,-1,0]) # target=sum, shift=left
        Z = block_butterfly_(Z,nchs)/2.
        Z = intermediate_rotation_(Z,nchs,Uh2)
        # Vertical atom extention
        #for ordV in range(ovlpFactor[Direction.VERTICAL]//2):
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

        relerr = torch.abs(actualZ-expctdZ)/torch.abs(expctdZ)
        abserr = torch.abs(actualZ-expctdZ)
        msg = 'relerr=%s, abserr=%s, stride=%s, ovlpfactor=%s, height=%d, width=%d, nodcleakage=%s, datatype=%s' % (relerr,abserr,stride,ovlpFactor,height,width,nodcleakage,datatype)

        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol),msg=msg)
        self.assertFalse(actualZ.requires_grad)

    @parameterized.expand(
        list(itertools.product(stride, ovlpfactor, height,width,datatype,usegpu))
    )
    def testForwardGrayScaleOvlpFactorXX(self,
            stride, ovlpfactor, height, width, datatype, usegpu):
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else:
                print('No GPU device was detected.')
                return
        else:
            device = torch.device("cpu")
        rtol,atol = 1e-4,1e-5

        genW = OrthonormalMatrixGenerationSystem(device=device,dtype=datatype)
        genU = OrthonormalMatrixGenerationSystem(device=device,dtype=datatype)

        # Initialization function of angle parameters
        angle0 = 2.0*math.pi*random.random()
        def init_angles(m):
            if type(m) == OrthonormalTransform:
                torch.nn.init.constant_(m.angles,angle0)        

        # Parameters
        ovlpFactor = ovlpfactor
        isNoDcLeakage = False 
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
        W0 = genW(angles[:,:nAnglesH]) 
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
        height, width, nlevels, nodcleakage,datatype, usegpu):
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else:
                print('No GPU device was detected.')
                return 
        else:
            device = torch.device("cpu")
        rtol, atol = 1e-3, 1e-4

        genW = OrthonormalMatrixGenerationSystem(device=device,dtype=datatype)
        genU = OrthonormalMatrixGenerationSystem(device=device,dtype=datatype)

        # Initialization function of angle parameters
        angle0 = 2.0*math.pi*random.random()
        def init_angles(m):
            if type(m) == OrthonormalTransform:
                torch.nn.init.constant_(m.angles,angle0)

        # Parameters
        stride_ = [2, 2]
        ovlpFactor = [ 3, 3 ]
        isNoDcLeakage = nodcleakage # False 
        nSamples = 8
        nComponents = 1
        nDecs = stride_[Direction.VERTICAL]*stride_[Direction.HORIZONTAL]
        nrows = height//stride_[Direction.VERTICAL]
        ncols = width//stride_[Direction.HORIZONTAL]

        # Source (nSamples x nComponents x (Stride[0]xnRows) x (Stride[1]xnCols))
        X = torch.rand(nSamples,nComponents,height,width,dtype=datatype,device=device,requires_grad=True)
        nAngles = (nDecs-2)*nDecs//4
        angles = angle0*torch.ones(nrows*ncols,nAngles,dtype=datatype,device=device)

        # Expected values
        ps = math.ceil(nDecs/2)
        pa = math.floor(nDecs/2)
        nchs = (ps, pa)
        nAnglesH = nAngles//2
        if isNoDcLeakage:
            angles[:,:(ps-1)] = 0
        W0 = genW(angles[:,:nAnglesH])
        U0 = genU(angles[:,nAnglesH:])
        Uh1 = -genU(angles[:,nAnglesH:])
        Uh2 = -genU(angles[:,nAnglesH:])
        Uv1 = -genU(angles[:,nAnglesH:])
        Uv2 = -genU(angles[:,nAnglesH:])

        # Block DCT (nSamples x nComponents x nrows x ncols) x decV x decH
        arrayshape = stride_.copy()
        arrayshape.insert(0,-1)
        # Multi-level decomposition
        coefs = []
        X_ = X.clone()
        for iStage in range(nlevels):
            Z = X_
            iLevel = iStage+1
            Y = dct.dct_2d(Z.view(arrayshape),norm='ortho')
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
            # Xac
            coefs.insert(0,Z[:,:,:,1:])
            if iLevel < nlevels:
                X_ = Z[:,:,:,0].reshape(nSamples,nComponents,nrows,ncols)
                nrows = nrows//stride_[Direction.VERTICAL]
                ncols = ncols//stride_[Direction.HORIZONTAL]
            else: # Xdc
                coefs.insert(0,Z[:,:,:,0].unsqueeze(3))              
        expctdZ = tuple(coefs)

        #for iStage in range(nlevels+1):
        #    print('expctdZ[',iStage,'].shape=',expctdZ[iStage].shape)

        # Instantiation of target class
        network = LsunAnalysis2dNetwork(
                input_size=[height,width],
                stride=stride_,
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
        for iStage in range(nlevels+1):
            self.assertEqual(actualZ[iStage].dtype,datatype)         
            self.assertTrue(torch.allclose(actualZ[iStage],expctdZ[iStage],rtol=rtol,atol=atol))
            self.assertFalse(actualZ[iStage].requires_grad)

    @parameterized.expand(
        list(itertools.product(stride,ovlpfactor,height,width,nodcleakage,datatype,usegpu))
    )
    def testBackwardGrayScale(self,
        stride,ovlpfactor,height,width,nodcleakage, datatype, usegpu):
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

        # Source (nSamples x nComponents x height_ x width_)
        X = torch.randn(nSamples,nComponents,height,width,dtype=datatype,device=device,requires_grad=True)

        # Coefficients nSamples x nRows x nCols x nDecs
        dLdZ = torch.randn(nSamples,nrows,ncols,nDecs,dtype=datatype,device=device) 

        # Instantiation of target class
        network = LsunAnalysis2dNetwork(
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
        nlevels, nodcleakage, datatype, usegpu):
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
        stride_ = [2,2]
        ovlpfactor_ = [3,3]
        isNoDcLeakage = nodcleakage
        nlevels_ = nlevels        
        nSamples = 4
        nComponents = 1
        nDecs = stride_[Direction.VERTICAL]*stride_[Direction.HORIZONTAL]
        nrows = 2
        ncols = 3
        height_ = nrows*(stride_[Direction.VERTICAL]**nlevels_)
        width_ = ncols*(stride_[Direction.HORIZONTAL]**nlevels_)
        
        # Source (nSamples x nComponents x height_ x width_)
        X = torch.randn(nSamples,nComponents,height_,width_,dtype=datatype,device=device,requires_grad=True)

        # Coefficients nSamples x nRows x nCols x nDecs
        nrows_ = nrows
        ncols_ = ncols
        dLdZ = []
        for iLevel in range(1,nlevels+1):
            if iLevel == 1:
                dLdZ.append(torch.randn(nSamples,nrows_,ncols_,1,dtype=datatype,device=device)) 
            dLdZ.append(torch.randn(nSamples,nrows_,ncols_,nDecs-1,dtype=datatype,device=device))     
            nrows_ *= stride_[Direction.VERTICAL]
            ncols_ *= stride_[Direction.HORIZONTAL]
        dLdZ = tuple(dLdZ)

        # Instantiation of target class
        network = LsunAnalysis2dNetwork(
                input_size=[height_,width_],
                stride=stride_,
                overlapping_factor=ovlpfactor_,
                number_of_levels=nlevels_,
                no_dc_leakage=isNoDcLeakage,
                dtype = datatype
            )
        network = network.to(device)

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
    Local functions
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
        Y[iSample,:,:,ps:] = Ya.clone().view(nrows,ncols,pa)
    return Y
    
if __name__ == '__main__':
    unittest.main() #failfast=True)

    """
    # Create a test suite
    suite = unittest.TestSuite()

    # Add specific test methods to the suite

    of = [ 23 ]

    for i in of:
        suite.addTest(LsunAnalysis2dNetworkTestCase('testBackwardGrayScaleMultiLevels_{:02d}'.format(i)))

    # Create a test runner
    runner = unittest.TextTestRunner()

    # Run the tests
    runner.run(suite)
    """