import itertools
import unittest
from parameterized import parameterized
#import random
import torch
import torch.nn as nn
import torch_dct as dct

import math
from lsunAnalysis2dNetwork import LsunAnalysis2dNetwork
from lsunUtility import Direction, permuteDctCoefs, permuteIdctCoefs

stride = [ [1, 1], [2, 2], [2, 4], [4, 1], [4, 4] ]
ovlpfactor = [ [1, 1], [1, 3], [3, 1], [3, 3], [5, 5] ]
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
        list(itertools.product(stride,ovlpfactor,nodcleakage))
    )
    def testConstructor(self, stride,ovlpfactor,nodcleakage):

        # Expcted values
        expctdStride = stride
        expctdOvlpFactor = ovlpfactor
        expctdNoDcLeakage = nodcleakage        

        # Instantiation of target class
        network = LsunAnalysis2dNetwork(
            stride = stride,
            overlapping_factor = ovlpfactor,
            no_dc_leakage = nodcleakage
        )

        # Actual values
        actualStride = network.stride
        actualOvlpFactor = network.overlapping_factor
        actualNoDcLeakage = network.no_dc_leakage 

        # Evaluation
        self.assertTrue(isinstance(network, nn.Module))
        self.assertEqual(actualStride,expctdStride)
        self.assertEqual(actualOvlpFactor,expctdOvlpFactor)
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
        A = permuteDctCoefs(Y)
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

    """
    @parameterized.expand(
        list(itertools.product(nchs,stride))
    )
    def testNumberOfChannelsException(self,
        nchs,stride):
        ps,pa = nchs
        with self.assertRaises(InvalidNumberOfChannels):
            LsunAnalysis2dNetwork(
                number_of_channels = [ps,ps+1],
                decimation_factor = stride
            )

        with self.assertRaises(InvalidNumberOfChannels):
            LsunAnalysis2dNetwork(
                number_of_channels = [pa+1,pa],
                decimation_factor = stride
            )

    @parameterized.expand(
        list(itertools.product(nchs,stride,ppord))
    )
    def testNumberOfPolyPhaseOrderException(self,
        nchs,stride,ppord):
        with self.assertRaises(InvalidPolyPhaseOrder):
            LsunAnalysis2dNetwork(
                polyphase_order = [ ppord[0]+1, ppord[1] ],
                number_of_channels = nchs,
                decimation_factor = stride
            )

        with self.assertRaises(InvalidPolyPhaseOrder):
            LsunAnalysis2dNetwork(
                polyphase_order = [ ppord[0], ppord[1]+1 ],
                number_of_channels = nchs,
                decimation_factor = stride
            )

        with self.assertRaises(InvalidPolyPhaseOrder):
            LsunAnalysis2dNetwork(
                polyphase_order = [ ppord[0]+1, ppord[1]+1 ],
                number_of_channels = nchs,
                decimation_factor = stride
            )

    @parameterized.expand(
        list(itertools.product(nchs,stride,ppord))
    )
    def testNumberOfVanishingMomentsException(self,
        nchs,stride,ppord):
        nVm = -1
        with self.assertRaises(InvalidNumberOfVanishingMoments):
            LsunAnalysis2dNetwork(
                number_of_channels = nchs,
                decimation_factor = stride,
                polyphase_order = ppord,
                number_of_vanishing_moments = nVm
            )

        nVm = 2
        with self.assertRaises(InvalidNumberOfVanishingMoments):
            LsunAnalysis2dNetwork(
                number_of_channels = nchs,
                decimation_factor = stride,
                polyphase_order = ppord,
                number_of_vanishing_moments = nVm
            )

    @parameterized.expand(
        list(itertools.product(nchs,stride,ppord))
    )
    def testNumberOfLevelsException(self,
        nchs,stride,ppord):
        nlevels = -1
        with self.assertRaises(InvalidNumberOfLevels):
            LsunAnalysis2dNetwork(
                number_of_channels = nchs,
                decimation_factor = stride,
                polyphase_order = ppord,
                number_of_levels = nlevels
            )

        nlevels = 0.5
        with self.assertRaises(InvalidNumberOfLevels):
            LsunAnalysis2dNetwork(
                number_of_channels = nchs,
                decimation_factor = stride,
                polyphase_order = ppord,
                number_of_levels = nlevels
            )


    @parameterized.expand(
        list(itertools.product(nchs,stride,height,width,datatype))
    )
    def testForwardGrayScaleWithInitilization(self,
            nchs,stride, height, width, datatype):
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
        expctdZ = Zsa.T.view(nSamples,nrows,ncols,nChsTotal)

        # Instantiation of target class
        network = LsunAnalysis2dNetwork(
                number_of_channels=nchs,
                decimation_factor=stride,
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
        list(itertools.product(nchs,stride,height,width,datatype))
    )
    def testForwardGrayScaleOrd22(self,
            nchs, stride, height, width, datatype):
        rtol,atol = 1e-3,1e-6
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")       

        # Parameters
        ppOrd = [ 2, 2 ]
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
                decimation_factor=stride,
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
    def testForwardGrayScaleOrd20(self,
            nchs, stride, height, width, datatype):
        rtol,atol = 1e-3,1e-6
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")        

        # Parameters
        ppOrd = [ 2, 0 ]
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
                decimation_factor=stride,
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
                decimation_factor=stride,
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
                decimation_factor=stride,
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
                decimation_factor=stride,
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
                decimation_factor=stride,
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
                decimation_factor=stride,
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
                decimation_factor=stride,
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

if __name__ == '__main__':
    unittest.main()