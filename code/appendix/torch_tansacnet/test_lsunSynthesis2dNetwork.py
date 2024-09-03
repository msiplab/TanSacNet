import itertools
import unittest
from parameterized import parameterized
#import random
import torch
import torch.nn as nn
import torch_dct as dct

import math
from lsunSynthesis2dNetwork import LsunSynthesis2dNetwork
from lsunUtility import Direction, permuteDctCoefs, permuteIdctCoefs

stride = [ [1, 1], [2, 2], [2, 4], [4, 1], [4, 4] ]
ovlpfactor = [ [1, 1], [1, 3], [3, 1], [3, 3], [5, 5] ]
datatype = [ torch.float, torch.double ]
height = [ 8, 16, 32 ]
width = [ 8, 16, 32 ]
nodcleakage = [ False, True ]
#nlevels = [ 1, 2, 3 ]
usegpu = [ True, False ]

class LsunSynthesis2dNetworkTestCase(unittest.TestCase):
    """
    LSUNSYNHESIS2DNETWORKTESTCASE Test cases for LsunSynthesis2dNetwork
    
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
        network = LsunSynthesis2dNetwork(
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
        nrows = int(math.ceil(height/stride[Direction.VERTICAL]))
        ncols = int(math.ceil(width/stride[Direction.HORIZONTAL]))
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
        A = permuteIdctCoefs(V,stride)
        Y = dct.idct_2d(A,norm='ortho')
        expctdZ = Y.reshape(nSamples,nComponents,height,width)
        
        # Instantiation of target class
        network = LsunSynthesis2dNetwork(
            stride=stride
            )
        network = network.to(device)

        # Actual values
        with torch.no_grad():
            actualZ = network.forward(X)

        # Evaluation
        self.assertEqual(actualZ.dtype,datatype)
        msg = ' '.join(['stride=',str(stride),', height=',str(height),', width=',str(width),', datatype=',str(datatype),', usegpu=',str(usegpu)])
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol),msg=msg)
        self.assertFalse(actualZ.requires_grad)

"""
    
    @parameterized.expand(  
        list(itertools.product(nchs,stride))
    )
    def testNumberOfChannelsException(self,
        nchs,stride):
        ps,pa = nchs
        with self.assertRaises(InvalidNumberOfChannels):
            LsunSynthesis2dNetwork(
                number_of_channels = [ps,ps+1],
                decimation_factor = stride
            )

        with self.assertRaises(InvalidNumberOfChannels):
            LsunSynthesis2dNetwork(
                number_of_channels = [pa+1,pa],
                decimation_factor = stride
            )

    @parameterized.expand(
        list(itertools.product(nchs,stride,ppord))
    )
    def testNumberOfPolyPhaseOrderException(self,
        nchs,stride,ppord):
        with self.assertRaises(InvalidPolyPhaseOrder):
            LsunSynthesis2dNetwork(
                polyphase_order = [ ppord[0]+1, ppord[1] ],
                number_of_channels = nchs,
                decimation_factor = stride
            )

        with self.assertRaises(InvalidPolyPhaseOrder):
            LsunSynthesis2dNetwork(
                polyphase_order = [ ppord[0], ppord[1]+1 ],
                number_of_channels = nchs,
                decimation_factor = stride
            )

        with self.assertRaises(InvalidPolyPhaseOrder):
            LsunSynthesis2dNetwork(
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
            LsunSynthesis2dNetwork(
                number_of_channels = nchs,
                decimation_factor = stride,
                polyphase_order = ppord,
                number_of_vanishing_moments = nVm
            )

        nVm = 2
        with self.assertRaises(InvalidNumberOfVanishingMoments):
            LsunSynthesis2dNetwork(
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
            LsunSynthesis2dNetwork(
                number_of_channels = nchs,
                decimation_factor = stride,
                polyphase_order = ppord,
                number_of_levels = nlevels
            )

        nlevels = 0.5
        with self.assertRaises(InvalidNumberOfLevels):
            LsunSynthesis2dNetwork(
                number_of_channels = nchs,
                decimation_factor = stride,
                polyphase_order = ppord,
                number_of_levels = nlevels
            )

    @parameterized.expand(
        list(itertools.product(nchs,stride,height,width,datatype))
    )
    def testForwardGrayScaleWithInitialization(self,
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
        nrows = int(math.ceil(height/stride[Direction.VERTICAL]))
        ncols = int(math.ceil(width/stride[Direction.HORIZONTAL]))
        nComponents = 1
        nDecs = stride[0]*stride[1] #math.prod(stride)
        nChsTotal = sum(nchs)

        # nSamples x nRows x nCols x nChsTotal
        X = torch.randn(nSamples,nrows,ncols,nChsTotal,dtype=datatype,device=device,requires_grad=True)
        
        # Expected values        
        # nSamples x nRows x nCols x nDecs
        ps,pa = nchs
        angles = angle0*torch.ones(int((nChsTotal-2)*nChsTotal/4)) #,dtype=datatype)
        nAngsW = int(len(angles)/2)
        angsW,angsU = angles[:nAngsW],angles[nAngsW:]
        W0T,U0T = gen(angsW).T.to(device),gen(angsU).T.to(device)
        Ys = X[:,:,:,:ps].view(-1,ps).T
        Ya = X[:,:,:,ps:].view(-1,pa).T
        ms,ma = int(math.ceil(nDecs/2.)),int(math.floor(nDecs/2.))        
        Zsa = torch.cat(
                ( W0T[:ms,:] @ Ys, 
                  U0T[:ma,:] @ Ya ),dim=0)
        V = Zsa.T.view(nSamples,nrows,ncols,nDecs)
        A = permuteIdctCoefs_(V,stride)
        Y = idct_2d(A)
        expctdZ = Y.reshape(nSamples,nComponents,height,width)
        
        # Instantiation of target class
        network = LsunSynthesis2dNetwork(
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
        nrows = int(math.ceil(height/stride[Direction.VERTICAL]))
        ncols = int(math.ceil(width/stride[Direction.HORIZONTAL]))
        nComponents = 1
        nDecs = stride[0]*stride[1] #math.prod(stride)
        nChsTotal = sum(nchs)

        # nSamples x nRows x nCols x nChsTotal
        X = torch.randn(nSamples,nrows,ncols,nChsTotal,dtype=datatype,device=device,requires_grad=True)

        # Expected values        
        # nSamples x nRows x nCols x nDecs
        ps,pa = nchs
        Z = X
        # Vertical atom concatenation
        Uv2T = -torch.eye(pa,dtype=datatype).to(device)
        Z = intermediate_rotation(Z,nchs,Uv2T)
        Z = block_butterfly(Z,nchs)
        Z = block_shift(Z,nchs,1,[0,1,0,0]) # target=sum, shift=down
        Z = block_butterfly(Z,nchs)/2.
        Uv1T = -torch.eye(pa,dtype=datatype).to(device)
        Z = intermediate_rotation(Z,nchs,Uv1T)
        Z = block_butterfly(Z,nchs)
        Z = block_shift(Z,nchs,0,[0,-1,0,0]) # target=diff, shift=up
        Z = block_butterfly(Z,nchs)/2.
        # Horizontal atom concatenation
        Uh2T = -torch.eye(pa,dtype=datatype).to(device)
        Z = intermediate_rotation(Z,nchs,Uh2T)
        Z = block_butterfly(Z,nchs)
        Z = block_shift(Z,nchs,1,[0,0,1,0]) # target=sum, shift=right
        Z = block_butterfly(Z,nchs)/2.
        Uh1T = -torch.eye(pa,dtype=datatype).to(device)
        Z = intermediate_rotation(Z,nchs,Uh1T)
        Z = block_butterfly(Z,nchs)
        Z = block_shift(Z,nchs,0,[0,0,-1,0]) # target=diff, shift=left
        Z = block_butterfly(Z,nchs)/2.
        # Final rotation
        W0T = torch.eye(ps,dtype=datatype).to(device)
        U0T = torch.eye(pa,dtype=datatype).to(device)
        Ys = Z[:,:,:,:ps].view(-1,ps).T
        Ya = Z[:,:,:,ps:].view(-1,pa).T
        ms,ma = int(math.ceil(nDecs/2.)),int(math.floor(nDecs/2.))        
        Zsa = torch.cat(
                ( W0T[:ms,:] @ Ys, 
                  U0T[:ma,:] @ Ya ),dim=0)
        V = Zsa.T.view(nSamples,nrows,ncols,nDecs)
        A = permuteIdctCoefs_(V,stride)
        Y = idct_2d(A)
        expctdZ = Y.reshape(nSamples,nComponents,height,width)
        
        # Instantiation of target class
        network = LsunSynthesis2dNetwork(
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
        nrows = int(math.ceil(height/stride[Direction.VERTICAL]))
        ncols = int(math.ceil(width/stride[Direction.HORIZONTAL]))
        nComponents = 1
        nDecs = stride[0]*stride[1] #math.prod(stride)
        nChsTotal = sum(nchs)

        # nSamples x nRows x nCols x nChsTotal
        X = torch.randn(nSamples,nrows,ncols,nChsTotal,dtype=datatype,device=device,requires_grad=True)

        # Expected values        
        # nSamples x nRows x nCols x nDecs
        ps,pa = nchs
        Z = X
        # Vertical atom concatenation
        Uv2T = -torch.eye(pa,dtype=datatype).to(device)
        Z = intermediate_rotation(Z,nchs,Uv2T)
        Z = block_butterfly(Z,nchs)
        Z = block_shift(Z,nchs,1,[0,1,0,0]) # target=sum, shift=down
        Z = block_butterfly(Z,nchs)/2.
        Uv1T = -torch.eye(pa,dtype=datatype).to(device)
        Z = intermediate_rotation(Z,nchs,Uv1T)
        Z = block_butterfly(Z,nchs)
        Z = block_shift(Z,nchs,0,[0,-1,0,0]) # target=diff, shift=up
        Z = block_butterfly(Z,nchs)/2.
        # Final rotation
        W0T = torch.eye(ps,dtype=datatype).to(device)
        U0T = torch.eye(pa,dtype=datatype).to(device)
        Ys = Z[:,:,:,:ps].view(-1,ps).T
        Ya = Z[:,:,:,ps:].view(-1,pa).T
        ms,ma = int(math.ceil(nDecs/2.)),int(math.floor(nDecs/2.))        
        Zsa = torch.cat(
                ( W0T[:ms,:] @ Ys, 
                  U0T[:ma,:] @ Ya ),dim=0)
        V = Zsa.T.view(nSamples,nrows,ncols,nDecs)
        A = permuteIdctCoefs_(V,stride)
        Y = idct_2d(A)
        expctdZ = Y.reshape(nSamples,nComponents,height,width)
        
        # Instantiation of target class
        network = LsunSynthesis2dNetwork(
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
        nrows = int(math.ceil(height/stride[Direction.VERTICAL]))
        ncols = int(math.ceil(width/stride[Direction.HORIZONTAL]))
        nComponents = 1
        nDecs = stride[0]*stride[1] #math.prod(stride)
        nChsTotal = sum(nchs)

        # nSamples x nRows x nCols x nChsTotal
        X = torch.randn(nSamples,nrows,ncols,nChsTotal,dtype=datatype,device=device,requires_grad=True)

        # Expected values        
        # nSamples x nRows x nCols x nDecs
        ps,pa = nchs
        Z = X
        # Horizontal atom concatenation
        Uh2T = -torch.eye(pa,dtype=datatype).to(device)
        Z = intermediate_rotation(Z,nchs,Uh2T)
        Z = block_butterfly(Z,nchs)
        Z = block_shift(Z,nchs,1,[0,0,1,0]) # target=sum, shift=right
        Z = block_butterfly(Z,nchs)/2.
        Uh1T = -torch.eye(pa,dtype=datatype).to(device)
        Z = intermediate_rotation(Z,nchs,Uh1T)
        Z = block_butterfly(Z,nchs)
        Z = block_shift(Z,nchs,0,[0,0,-1,0]) # target=diff, shift=left
        Z = block_butterfly(Z,nchs)/2.
        # Final rotation
        W0T = torch.eye(ps,dtype=datatype).to(device)
        U0T = torch.eye(pa,dtype=datatype).to(device)
        Ys = Z[:,:,:,:ps].view(-1,ps).T
        Ya = Z[:,:,:,ps:].view(-1,pa).T
        ms,ma = int(math.ceil(nDecs/2.)),int(math.floor(nDecs/2.))        
        Zsa = torch.cat(
                ( W0T[:ms,:] @ Ys, 
                  U0T[:ma,:] @ Ya ),dim=0)
        V = Zsa.T.view(nSamples,nrows,ncols,nDecs)
        A = permuteIdctCoefs_(V,stride)
        Y = idct_2d(A)
        expctdZ = Y.reshape(nSamples,nComponents,height,width)
        
        # Instantiation of target class
        network = LsunSynthesis2dNetwork(
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
        nrows = int(math.ceil(height/stride[Direction.VERTICAL]))
        ncols = int(math.ceil(width/stride[Direction.HORIZONTAL]))
        nComponents = 1
        nDecs = stride[0]*stride[1] #math.prod(stride)
        nChsTotal = sum(nchs)

        # nSamples x nRows x nCols x nChsTotal
        X = torch.randn(nSamples,nrows,ncols,nChsTotal,dtype=datatype,device=device,requires_grad=True)
        
        # Expected values        
        # nSamples x nRows x nCols x nDecs
        ps,pa = nchs
        Z = X
        # Vertical atom concatenation
        for ordV in range(int(ppOrd[Direction.VERTICAL]/2)):
            Uv2T = -torch.eye(pa,dtype=datatype).to(device)
            Z = intermediate_rotation(Z,nchs,Uv2T)
            Z = block_butterfly(Z,nchs)
            Z = block_shift(Z,nchs,1,[0,1,0,0]) # target=sum, shift=down
            Z = block_butterfly(Z,nchs)/2.
            Uv1T = -torch.eye(pa,dtype=datatype).to(device)
            Z = intermediate_rotation(Z,nchs,Uv1T)
            Z = block_butterfly(Z,nchs)
            Z = block_shift(Z,nchs,0,[0,-1,0,0]) # target=diff, shift=up
            Z = block_butterfly(Z,nchs)/2.
        # Horizontal atom concatenation
        for ordH in range(int(ppOrd[Direction.HORIZONTAL]/2)):
            Uh2T = -torch.eye(pa,dtype=datatype).to(device)
            Z = intermediate_rotation(Z,nchs,Uh2T)
            Z = block_butterfly(Z,nchs)
            Z = block_shift(Z,nchs,1,[0,0,1,0]) # target=sum, shift=right
            Z = block_butterfly(Z,nchs)/2.
            Uh1T = -torch.eye(pa,dtype=datatype).to(device)
            Z = intermediate_rotation(Z,nchs,Uh1T)
            Z = block_butterfly(Z,nchs)
            Z = block_shift(Z,nchs,0,[0,0,-1,0]) # target=diff, shift=left
            Z = block_butterfly(Z,nchs)/2.
        # Final rotation
        W0T = torch.eye(ps,dtype=datatype).to(device)
        U0T = torch.eye(pa,dtype=datatype).to(device)
        Ys = Z[:,:,:,:ps].view(-1,ps).T
        Ya = Z[:,:,:,ps:].view(-1,pa).T
        ms,ma = int(math.ceil(nDecs/2.)),int(math.floor(nDecs/2.))        
        Zsa = torch.cat(
                ( W0T[:ms,:] @ Ys, 
                  U0T[:ma,:] @ Ya ),dim=0)
        V = Zsa.T.view(nSamples,nrows,ncols,nDecs)
        A = permuteIdctCoefs_(V,stride)
        Y = idct_2d(A)
        expctdZ = Y.reshape(nSamples,nComponents,height,width)
        
        # Instantiation of target class
        network = LsunSynthesis2dNetwork(
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
    def testForwardGrayScaleOverlappingWithInitalization(self,
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
        nrows = int(math.ceil(height/stride[Direction.VERTICAL]))
        ncols = int(math.ceil(width/stride[Direction.HORIZONTAL]))
        nComponents = 1
        nDecs = stride[0]*stride[1] #math.prod(stride)
        nChsTotal = sum(nchs)

        # nSamples x nRows x nCols x nChsTotal
        X = torch.randn(nSamples,nrows,ncols,nChsTotal,dtype=datatype,device=device,requires_grad=True)

        # Expected values        
        # nSamples x nRows x nCols x nDecs
        ps,pa = nchs
        angles = angle0*torch.ones(int((nChsTotal-2)*nChsTotal/4)).to(device) #,dtype=datatype)
        nAngsW = int(len(angles)/2)
        angsW,angsU = angles[:nAngsW],angles[nAngsW:]
        Z = X
        # Vertical atom concatenation
        for ordV in range(int(ppOrd[Direction.VERTICAL]/2)):
            Uv2T = -gen(angsU).T
            Z = intermediate_rotation(Z,nchs,Uv2T)
            Z = block_butterfly(Z,nchs)
            Z = block_shift(Z,nchs,1,[0,1,0,0]) # target=sum, shift=down
            Z = block_butterfly(Z,nchs)/2.
            Uv1T = -gen(angsU).T
            Z = intermediate_rotation(Z,nchs,Uv1T)
            Z = block_butterfly(Z,nchs)
            Z = block_shift(Z,nchs,0,[0,-1,0,0]) # target=diff, shift=up
            Z = block_butterfly(Z,nchs)/2.
        # Horizontal atom concatenation
        for ordH in range(int(ppOrd[Direction.HORIZONTAL]/2)):
            Uh2T = -gen(angsU).T
            Z = intermediate_rotation(Z,nchs,Uh2T)
            Z = block_butterfly(Z,nchs)
            Z = block_shift(Z,nchs,1,[0,0,1,0]) # target=sum, shift=right
            Z = block_butterfly(Z,nchs)/2.
            Uh1T = -gen(angsU).T
            Z = intermediate_rotation(Z,nchs,Uh1T)
            Z = block_butterfly(Z,nchs)
            Z = block_shift(Z,nchs,0,[0,0,-1,0]) # target=diff, shift=left
            Z = block_butterfly(Z,nchs)/2.
        # Final rotation
        W0T,U0T = gen(angsW).T,gen(angsU).T        
        Ys = Z[:,:,:,:ps].view(-1,ps).T
        Ya = Z[:,:,:,ps:].view(-1,pa).T
        ms,ma = int(math.ceil(nDecs/2.)),int(math.floor(nDecs/2.))        
        Zsa = torch.cat(
                ( W0T[:ms,:] @ Ys, 
                  U0T[:ma,:] @ Ya ),dim=0)
        V = Zsa.T.view(nSamples,nrows,ncols,nDecs)
        A = permuteIdctCoefs_(V,stride)
        Y = idct_2d(A)
        expctdZ = Y.reshape(nSamples,nComponents,height,width)
        
        # Instantiation of target class
        network = LsunSynthesis2dNetwork(
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
    def testForwardGrayScaleOverlappingWithNoDcLeackage(self,
            nchs, stride, ppord, datatype):
        rtol,atol = 1e-3,1e-6
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")            
        #gen = OrthonormalMatrixGenerationSystem(dtype=datatype)

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
        nrows = int(math.ceil(height/stride[Direction.VERTICAL]))
        ncols = int(math.ceil(width/stride[Direction.HORIZONTAL]))
        nComponents = 1
        nDecs = stride[0]*stride[1] #math.prod(stride)
        nChsTotal = sum(nchs)

        # nSamples x nRows x nCols x nChsTotal
        X = torch.cat(
            [math.sqrt(nDecs)*torch.ones(nSamples,nrows,ncols,1,dtype=datatype,device=device,requires_grad=True),
            torch.zeros(nSamples,nrows,ncols,nChsTotal-1,dtype=datatype,device=device,requires_grad=True)],
            dim=3)

        # Expected values        
        # nSamples x nRows x nCols x nDecs
        expctdZ = torch.ones(nSamples,nComponents,height,width,dtype=datatype).to(device)
        
        # Instantiation of target class
        network = LsunSynthesis2dNetwork(
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
        nrows = int(math.ceil(height/(stride[Direction.VERTICAL]**nlevels)))
        ncols = int(math.ceil(width/(stride[Direction.HORIZONTAL]**nlevels)))        
        nComponents = 1
        nDecs = stride[0]*stride[1] #math.prod(stride)
        nChsTotal = sum(nchs)

        # nSamples x nRows x nCols x nChsTotal
        nrows_ = nrows
        ncols_ = ncols
        X = []
        for iLevel in range(1,nlevels+1):
            if iLevel == 1:
                X.append(torch.randn(nSamples,nrows_,ncols_,1,dtype=datatype,device=device,requires_grad=True)) 
            X.append(torch.randn(nSamples,nrows_,ncols_,nChsTotal-1,dtype=datatype,device=device,requires_grad=True))     
            nrows_ *= stride[Direction.VERTICAL]
            ncols_ *= stride[Direction.HORIZONTAL]
        X = tuple(X)

        # Expected values        
        # nSamples x nRows x nCols x nDecs
        ps,pa = nchs
        # Multi-level reconstruction
        for iLevel in range(nlevels,0,-1):
            angles = angle0*torch.ones(int((nChsTotal-2)*nChsTotal/4)).to(device) #,dtype=datatype)
            nAngsW = int(len(angles)/2)
            angsW,angsU = angles[:nAngsW],angles[nAngsW:]
            angsW,angsU = angles[:nAngsW],angles[nAngsW:]
            if nVm > 0:
                angsW[:(ps-1)] = torch.zeros_like(angsW[:(ps-1)])
            # Extract scale channel
            if iLevel == nlevels:
                Xdc = X[0]
            Xac = X[nlevels-iLevel+1]
            Z = torch.cat((Xdc,Xac),dim=3)
            # Vertical atom concatenation
            for ordV in range(int(ppOrd[Direction.VERTICAL]/2)):
                Uv2T = -gen(angsU).T
                Z = intermediate_rotation(Z,nchs,Uv2T)
                Z = block_butterfly(Z,nchs)
                Z = block_shift(Z,nchs,1,[0,1,0,0]) # target=sum, shift=down
                Z = block_butterfly(Z,nchs)/2.
                Uv1T = -gen(angsU).T
                Z = intermediate_rotation(Z,nchs,Uv1T)
                Z = block_butterfly(Z,nchs)
                Z = block_shift(Z,nchs,0,[0,-1,0,0]) # target=diff, shift=up
                Z = block_butterfly(Z,nchs)/2.
            # Horizontal atom concatenation
            for ordH in range(int(ppOrd[Direction.HORIZONTAL]/2)):
                Uh2T = -gen(angsU).T
                Z = intermediate_rotation(Z,nchs,Uh2T)
                Z = block_butterfly(Z,nchs)
                Z = block_shift(Z,nchs,1,[0,0,1,0]) # target=sum, shift=right
                Z = block_butterfly(Z,nchs)/2.
                Uh1T = -gen(angsU).T
                Z = intermediate_rotation(Z,nchs,Uh1T)
                Z = block_butterfly(Z,nchs)
                Z = block_shift(Z,nchs,0,[0,0,-1,0]) # target=diff, shift=left
                Z = block_butterfly(Z,nchs)/2.
            # Final rotation
            W0T,U0T = gen(angsW).T,gen(angsU).T        
            Ys = Z[:,:,:,:ps].view(-1,ps).T
            Ya = Z[:,:,:,ps:].view(-1,pa).T
            ms,ma = int(math.ceil(nDecs/2.)),int(math.floor(nDecs/2.))        
            Zsa = torch.cat(
                    ( W0T[:ms,:] @ Ys, 
                      U0T[:ma,:] @ Ya ),dim=0)
            V = Zsa.T.view(nSamples,nrows,ncols,nDecs)
            A = permuteIdctCoefs_(V,stride)
            Y = idct_2d(A)
            # Update
            nrows *= stride[Direction.VERTICAL]
            ncols *= stride[Direction.HORIZONTAL]            
            Xdc = Y.reshape(nSamples,nrows,ncols,1)
        expctdZ = Xdc.view(nSamples,nComponents,height,width)
        
        # Instantiation of target class
        network = LsunSynthesis2dNetwork(
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
        self.assertEqual(actualZ.dtype,datatype)
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))
        self.assertFalse(actualZ.requires_grad)

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

        # Coefficients nSamples x nRows x nCols x nChsTotal
        nrows_ = nrows
        ncols_ = ncols
        X = []
        for iLevel in range(1,nlevels+1):
            if iLevel == 1:
                X.append(torch.randn(nSamples,nrows_,ncols_,dtype=datatype,device=device,requires_grad=True)) 
            X.append(torch.randn(nSamples,nrows_,ncols_,nChsTotal-1,dtype=datatype,device=device,requires_grad=True))     
            nrows_ *= stride[Direction.VERTICAL]
            ncols_ *= stride[Direction.HORIZONTAL]
        X = tuple(X)

        # Source (nSamples x nComponents x ((Stride[0]**nlevels) x nRows) x ((Stride[1]**nlevels) x nCols))
        dLdZ = torch.randn(nSamples,nComponents,height,width,dtype=datatype,device=device)

        # Instantiation of target class
        network = LsunSynthesis2dNetwork(
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
        Z.backward(dLdZ,retain_graph=True)
        actualdLdX = []
        for iCh in range(len(X)):
            actualdLdX.append(X[iCh].grad)

        # Evaluation
        for iCh in range(len(X)):
            self.assertEqual(actualdLdX[iCh].dtype,datatype)
            self.assertTrue(torch.allclose(actualdLdX[iCh],expctddLdX[iCh],rtol=rtol,atol=atol))
        self.assertTrue(Z.requires_grad)

"""

if __name__ == '__main__':
    unittest.main()