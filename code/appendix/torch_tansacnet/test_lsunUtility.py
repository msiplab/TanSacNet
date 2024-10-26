import itertools
import unittest
from parameterized import parameterized
import torch
import torch.nn as nn
from torch_tansacnet.lsunUtility import Direction, ForwardTruncationLayer, AdjointTruncationLayer
from torch_tansacnet.lsunUtility import rot_

stride = [ [2, 1], [1, 2], [2, 2], [2, 4], [4, 1], [4, 4] ]
number_of_channels = [ 1, 2 ]
datatype = [ torch.float, torch.double ]
height = [ 8, 16, 32 ]
width = [ 8, 16, 32 ]
nsamples = [ 1, 2, 4 ]
nlevels = [ 1, 2, 3 ]
usegpu = [ True, False ]

class LsunUtilityTestCase(unittest.TestCase):
    """
    LSUNUTILITYTESTCASE
    
    Requirements: Python 3.10/11.x, PyTorch 2.3.x
    
    Copyright (c) 2024, Shogo MURAMATSU
    
    All rights reserved.
    
    Contact address: Shogo MURAMATSU,
        Faculty of Engineering, Niigata University,
        8050 2-no-cho Ikarashi, Nishi-ku,
        Niigata, 950-2181, JAPAN
    
        https://www.eng.niigata-u.ac.jp/~msiplab/
    """

    def testDirection(self):
        self.assertEqual(Direction.VERTICAL, 0)
        self.assertEqual(Direction.HORIZONTAL, 1)
        self.assertEqual(Direction.DEPTH, 2)

    def testRot_(self):
        vt = torch.tensor([1.0, 2.0])
        vb = torch.tensor([3.0, 4.0])
        angle = torch.tensor(0.5)  # 例として0.5ラジアン
        vt_rot, vb_rot = rot_(vt, vb, angle)
        
        # 期待される結果を計算
        c = torch.cos(angle)
        s = torch.sin(angle)
        u = s * (vt + vb)
        vt_expected = (c + s) * vt - u
        vb_expected = (c - s) * vb + u
        
        self.assertTrue(torch.allclose(vt_rot, vt_expected))
        self.assertTrue(torch.allclose(vb_rot, vb_expected))

    @parameterized.expand(
        list(itertools.product(stride,datatype,nsamples,number_of_channels,usegpu))
    )
    def testForwardTruncationLayer(self,
                                   stride,datatype,nsamples,number_of_channels,usegpu): 
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                print('No GPU device was detected.')
                return 
        else:
            device = torch.device('cpu')
        
        # Parameters       
        stride_ = stride
        datatype_ = datatype
        height_ = 4
        width_ = 4
        nlevels_ = 0
        nsamples_ = nsamples
        number_of_channels_ = number_of_channels

        # nSamples x nRows x nCols x nDecs
        nrows = height_//stride_[Direction.VERTICAL]
        ncols = width_//stride_[Direction.HORIZONTAL]
        nDecs = stride_[Direction.VERTICAL]*stride_[Direction.HORIZONTAL]
        X = torch.randn(nsamples_,nrows,ncols,nDecs,dtype=datatype_,device=device,requires_grad=True)

        # Expected values
        expctdZ = X[:,:,:,:number_of_channels_]

        # Instantiation of target class
        layer = ForwardTruncationLayer(
            number_of_channels = number_of_channels_,
            stride = stride_,
            nlevels = nlevels_
        )
        layer = layer.to(device)

        # Actual values
        with torch.no_grad():
            actualZ = layer(X)

        # Evaluation
        self.assertEqual(actualZ.dtype,datatype_)
        self.assertEqual(actualZ.shape,expctdZ.shape)
        self.assertTrue(torch.allclose(actualZ,expctdZ))

    @parameterized.expand(
        list(itertools.product(stride,datatype,nsamples,number_of_channels,usegpu))
    )
    def testForwardTruncationLayerMultiLevelsAtStage1(self,
                                   stride,datatype,nsamples,number_of_channels,usegpu): 
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                print('No GPU device was detected.')
                return 
        else:
            device = torch.device('cpu')
        
        # Parameters       
        stride_ = stride
        datatype_ = datatype
        height_ = 128
        width_ = 192
        nlevels_ = 3
        nsamples_ = nsamples
        number_of_channels_at_target_stage = number_of_channels

        # target_stage = 1  

        # nSamples x nRows x nCols x nDecs
        nDecs = stride_[Direction.VERTICAL] * stride_[Direction.HORIZONTAL]
        nrows = height_ // (stride_[Direction.VERTICAL]**nlevels_)
        ncols = width_ // (stride_[Direction.HORIZONTAL]**nlevels_)
        nrows_ = nrows
        ncols_ = ncols
        X = []
        for istage in range(nlevels_+1):
            if istage == 0:
                X.append(torch.randn(nsamples_,nrows_,ncols_,1,dtype=datatype_,device=device,requires_grad=True)) 
            X.append(torch.randn(nsamples_,nrows_,ncols_,nDecs-1,dtype=datatype_,device=device,requires_grad=True))     
            nrows_ *= stride_[Direction.VERTICAL]
            ncols_ *= stride_[Direction.HORIZONTAL]
        X = tuple(X)

        # Expected values
        number_of_channels_ = number_of_channels_at_target_stage 
        expctdZ = []
        expctdZ.append(X[0])
        if number_of_channels_at_target_stage > 1:
            expctdZ.append(X[1][:,:,:,:(number_of_channels_at_target_stage-1)])
        expctdZ = tuple(expctdZ)

        # Instantiation of target class
        layer = ForwardTruncationLayer(
            number_of_channels = number_of_channels_,
            stride = stride_,
            nlevels = nlevels_
        )

        # Actual values
        with torch.no_grad():
            actualZ = layer(X)

        # Evaluation
        self.assertEqual(actualZ[0].dtype,datatype)        
        self.assertEqual(actualZ[0].shape,expctdZ[0].shape) 
        self.assertTrue(torch.allclose(actualZ[0],expctdZ[0]))    
        if number_of_channels_at_target_stage > 1:
            self.assertEqual(actualZ[1].dtype,datatype)                
            self.assertEqual(actualZ[1].shape,expctdZ[1].shape) 
            self.assertTrue(torch.allclose(actualZ[1],expctdZ[1]))  

    @parameterized.expand(
        list(itertools.product(stride,datatype,nsamples,usegpu))
    )
    def testForwardTruncationLayerMultiLevelsAtStage2(self,
                                   stride,datatype,nsamples,usegpu): 
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                print('No GPU device was detected.')
                return 
        else:
            device = torch.device('cpu')
        
        # Parameters       
        stride_ = stride
        datatype_ = datatype
        height_ = 128
        width_ = 192
        nlevels_ = 3
        nsamples_ = nsamples
        number_of_channels_at_target_stage = 1

        target_stage = 2  

        # nSamples x nRows x nCols x nDecs
        nDecs = stride_[Direction.VERTICAL] * stride_[Direction.HORIZONTAL]
        nrows = height_ // (stride_[Direction.VERTICAL]**nlevels_)
        ncols = width_ // (stride_[Direction.HORIZONTAL]**nlevels_)
        nrows_ = nrows
        ncols_ = ncols
        X = []
        for istage in range(nlevels_+1):
            if istage == 0:
                X.append(torch.randn(nsamples_,nrows_,ncols_,1,dtype=datatype_,device=device,requires_grad=True)) 
            X.append(torch.randn(nsamples_,nrows_,ncols_,nDecs-1,dtype=datatype_,device=device,requires_grad=True))     
            nrows_ *= stride_[Direction.VERTICAL]
            ncols_ *= stride_[Direction.HORIZONTAL]
        X = tuple(X)

        # Expected values
        number_of_channels_ = number_of_channels_at_target_stage + nDecs 
        expctdZ = []
        expctdZ.append(X[0])
        expctdZ.append(X[1])
        expctdZ.append(X[2][:,:,:,:number_of_channels_at_target_stage])
        expctdZ = tuple(expctdZ)

        # Instantiation of target class
        layer = ForwardTruncationLayer(
            number_of_channels = number_of_channels_,
            stride = stride_,
            nlevels = nlevels_
        )

        # Actual values
        with torch.no_grad():
            actualZ = layer(X)

        # Evaluation
        self.assertEqual(actualZ[0].dtype,datatype)        
        self.assertEqual(actualZ[0].shape,expctdZ[0].shape) 
        self.assertTrue(torch.allclose(actualZ[0],expctdZ[0]))    
        for istage in range(1,target_stage+1): 
            self.assertEqual(actualZ[istage].dtype,datatype)                
            self.assertEqual(actualZ[istage].shape,expctdZ[istage].shape) 
            self.assertTrue(torch.allclose(actualZ[istage],expctdZ[istage]))  

    @parameterized.expand(
        list(itertools.product(stride,datatype,nsamples,usegpu))
    )
    def testForwardTruncationLayerMultiLevelsAtStage3(self,
                                   stride,datatype,nsamples,usegpu): 
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                print('No GPU device was detected.')
                return 
        else:
            device = torch.device('cpu')
        
        # Parameters       
        stride_ = stride
        datatype_ = datatype
        height_ = 128
        width_ = 192
        nlevels_ = 3
        nsamples_ = nsamples
        number_of_channels_at_target_stage = 1

        target_stage = 3  

        # nSamples x nRows x nCols x nDecs
        nDecs = stride_[Direction.VERTICAL] * stride_[Direction.HORIZONTAL]
        nrows = height_ // (stride_[Direction.VERTICAL]**nlevels_)
        ncols = width_ // (stride_[Direction.HORIZONTAL]**nlevels_)
        nrows_ = nrows
        ncols_ = ncols
        X = []
        for istage in range(nlevels_+1):
            if istage == 0:
                X.append(torch.randn(nsamples_,nrows_,ncols_,1,dtype=datatype_,device=device,requires_grad=True)) 
            X.append(torch.randn(nsamples_,nrows_,ncols_,nDecs-1,dtype=datatype_,device=device,requires_grad=True))     
            nrows_ *= stride[Direction.VERTICAL]
            ncols_ *= stride[Direction.HORIZONTAL]
        X = tuple(X)

        # Expected values
        number_of_channels_ = number_of_channels_at_target_stage + nDecs + (nDecs-1)
        expctdZ = []
        for istage in range(target_stage):
            expctdZ.append(X[istage])
        expctdZ.append(X[target_stage][:,:,:,:number_of_channels_at_target_stage])
        expctdZ = tuple(expctdZ)

        # Instantiation of target class
        layer = ForwardTruncationLayer(
            number_of_channels = number_of_channels_,
            stride = stride_,
            nlevels = nlevels_
        )

        # Actual values
        with torch.no_grad():
            actualZ = layer(X)

        # Evaluation
        self.assertEqual(actualZ[0].dtype,datatype)        
        self.assertEqual(actualZ[0].shape,expctdZ[0].shape) 
        self.assertTrue(torch.allclose(actualZ[0],expctdZ[0]))    
        for istage in range(1,target_stage+1): 
            self.assertEqual(actualZ[istage].dtype,datatype)                
            self.assertEqual(actualZ[istage].shape,expctdZ[istage].shape) 
            self.assertTrue(torch.allclose(actualZ[istage],expctdZ[istage]))  

    @parameterized.expand(
        list(itertools.product(number_of_channels,datatype,nsamples,usegpu))
    )
    def testForwardTruncationLayerMultiLevelsAtStage3nChsX(self,
                                   number_of_channels,datatype,nsamples,usegpu): 
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                print('No GPU device was detected.')
                return 
        else:
            device = torch.device('cpu')
        
        # Parameters       
        stride_ = [ 2, 2 ]
        datatype_ = datatype
        height_ = 128
        width_ = 192
        nlevels_ = 3
        nsamples_ = nsamples
        number_of_channels_at_target_stage = number_of_channels

        target_stage = 3  

        # nSamples x nRows x nCols x nDecs
        nDecs = stride_[Direction.VERTICAL] * stride_[Direction.HORIZONTAL]
        nrows = height_ // (stride_[Direction.VERTICAL]**nlevels_)
        ncols = width_ // (stride_[Direction.HORIZONTAL]**nlevels_)
        nrows_ = nrows
        ncols_ = ncols
        X = []
        for istage in range(nlevels_+1):
            if istage == 0:
                X.append(torch.randn(nsamples_,nrows_,ncols_,1,dtype=datatype_,device=device,requires_grad=True)) 
            X.append(torch.randn(nsamples_,nrows_,ncols_,nDecs-1,dtype=datatype_,device=device,requires_grad=True))     
            nrows_ *= stride_[Direction.VERTICAL]
            ncols_ *= stride_[Direction.HORIZONTAL]
        X = tuple(X)

        # Expected values
        number_of_channels_ = number_of_channels_at_target_stage + nDecs + (nDecs-1)
        expctdZ = []
        for istage in range(target_stage):
            expctdZ.append(X[istage])
        expctdZ.append(X[target_stage][:,:,:,:number_of_channels_at_target_stage])
        expctdZ = tuple(expctdZ)

        # Instantiation of target class
        layer = ForwardTruncationLayer(
            number_of_channels = number_of_channels_,
            stride = stride_,
            nlevels = nlevels_
        )

        # Actual values
        with torch.no_grad():
            actualZ = layer(X)

        # Evaluation
        self.assertEqual(actualZ[0].dtype,datatype)        
        self.assertEqual(actualZ[0].shape,expctdZ[0].shape) 
        self.assertTrue(torch.allclose(actualZ[0],expctdZ[0]))    
        for istage in range(1,target_stage+1): 
            self.assertEqual(actualZ[istage].dtype,datatype)                
            self.assertEqual(actualZ[istage].shape,expctdZ[istage].shape) 
            self.assertTrue(torch.allclose(actualZ[istage],expctdZ[istage]))  

    @parameterized.expand(
        list(itertools.product(stride,datatype,nsamples,number_of_channels,usegpu))
    )
    def testAdjointTruncationLayer(self,
                                   stride,datatype,nsamples,number_of_channels,usegpu): 
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                print('No GPU device was detected.')
                return 
        else:
            device = torch.device('cpu')
        
        # Parameters       
        stride_ = stride
        datatype_ = datatype
        height_ = 4
        width_ = 4
        nlevels_ = 0
        nsamples_ = nsamples
        number_of_channels_ = number_of_channels

        # nSamples x nRows x nCols x nDecs
        nrows = height_//stride_[Direction.VERTICAL]
        ncols = width_//stride_[Direction.HORIZONTAL]
        nDecs = stride_[Direction.VERTICAL]*stride_[Direction.HORIZONTAL]
        X = torch.randn(nsamples_,nrows,ncols,number_of_channels_,dtype=datatype_,device=device,requires_grad=True)

        # Expected values
        if number_of_channels_ == nDecs:
            expctdZ = X
        else:
            expctdZ = torch.cat((X,torch.zeros(nsamples_,nrows,ncols,nDecs-number_of_channels_,dtype=datatype_,device=device)),dim=3)

        # Instantiation of target class
        layer = AdjointTruncationLayer(
            stride = stride_,
            nlevels = nlevels_
        )
        layer = layer.to(device)

        # Actual values
        with torch.no_grad():
            actualZ = layer(X)

        # Evaluation
        self.assertEqual(actualZ.dtype,datatype_)
        self.assertEqual(actualZ.shape,expctdZ.shape)
        self.assertTrue(torch.allclose(actualZ,expctdZ))

    @parameterized.expand(
        list(itertools.product(stride,datatype,nsamples,number_of_channels,usegpu))
    )
    def testAdjointTruncationLayerMultiLevelsAtStage1(self,
                                   stride,datatype,nsamples,number_of_channels,usegpu): 
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                print('No GPU device was detected.')
                return 
        else:
            device = torch.device('cpu')
        
        # Parameters       
        stride_ = stride
        datatype_ = datatype
        height_ = 128
        width_ = 192
        nlevels_ = 3
        nsamples_ = nsamples
        number_of_channels_at_target_stage = number_of_channels - 1

        # target_stage = 1  

        # nSamples x nRows x nCols x nDecs
        nDecs = stride_[Direction.VERTICAL] * stride_[Direction.HORIZONTAL]
        nrows = height_ // (stride_[Direction.VERTICAL]**nlevels_)
        ncols = width_ // (stride_[Direction.HORIZONTAL]**nlevels_)
        nrows_ = nrows
        ncols_ = ncols
        X = []
        for istage in range(nlevels_+1):
            if istage == 0:
                X.append(torch.randn(nsamples_,nrows_,ncols_,1,dtype=datatype_,device=device,requires_grad=True)) 
            X.append(torch.randn(nsamples_,nrows_,ncols_,nDecs-1,dtype=datatype_,device=device,requires_grad=True))     
            nrows_ *= stride_[Direction.VERTICAL]
            ncols_ *= stride_[Direction.HORIZONTAL]
        X = tuple(X)

        # Input values
        Y = []
        Y.append(X[0])
        if number_of_channels_at_target_stage > 0:
            Y.append(X[1][:,:,:,:number_of_channels_at_target_stage])
        Y = tuple(Y)

        # Expected values
        nrows_ = nrows
        ncols_ = ncols
        expctdZ = []
        for istage in range(nlevels_+1):
            expctdZ.append(X[istage].clone())
            if istage == 1:
                if number_of_channels_at_target_stage > 0:
                    expctdZ[1][:,:,:,number_of_channels_at_target_stage:] *= 0
                else:   
                    expctdZ[1] *= 0
            elif istage > 1:   
                expctdZ[istage] *= 0 
        expctdZ = tuple(expctdZ)

        # Instantiation of target class
        layer = AdjointTruncationLayer(
            stride = stride_,
            nlevels = nlevels_
        )

        # Actual values
        with torch.no_grad():
            actualZ = layer(Y)

        # Evaluation
        for istage in range(nlevels_+1):
            self.assertEqual(actualZ[istage].dtype,datatype)                
            self.assertEqual(actualZ[istage].shape,expctdZ[istage].shape,istage) 
            self.assertTrue(torch.allclose(actualZ[istage],expctdZ[istage]),istage) 
    
    @parameterized.expand(
        list(itertools.product(stride,datatype,nsamples,number_of_channels,usegpu))
    )
    def testAdjointTruncationLayerMultiLevelsAtStage2(self,
                                   stride,datatype,nsamples,number_of_channels,usegpu): 
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                print('No GPU device was detected.')
                return 
        else:
            device = torch.device('cpu')
        
        # Parameters       
        stride_ = stride
        datatype_ = datatype
        height_ = 128
        width_ = 192
        nlevels_ = 3
        nsamples_ = nsamples
        number_of_channels_at_target_stage = number_of_channels 

        target_stage = 2  

        # nSamples x nRows x nCols x nDecs
        nDecs = stride_[Direction.VERTICAL] * stride_[Direction.HORIZONTAL]
        nrows = height_ // (stride_[Direction.VERTICAL]**nlevels_)
        ncols = width_ // (stride_[Direction.HORIZONTAL]**nlevels_)
        nrows_ = nrows
        ncols_ = ncols
        X = []
        for istage in range(nlevels_+1):
            if istage == 0:
                X.append(torch.randn(nsamples_,nrows_,ncols_,1,dtype=datatype_,device=device,requires_grad=True)) 
            X.append(torch.randn(nsamples_,nrows_,ncols_,nDecs-1,dtype=datatype_,device=device,requires_grad=True))     
            nrows_ *= stride_[Direction.VERTICAL]
            ncols_ *= stride_[Direction.HORIZONTAL]
        X = tuple(X)

        # Input values
        Y = []
        Y.append(X[0])
        Y.append(X[1])
        Y.append(X[2][:,:,:,:number_of_channels_at_target_stage])
        Y = tuple(Y)

        # Expected values
        nrows_ = nrows
        ncols_ = ncols
        expctdZ = []
        for istage in range(nlevels_+1):
            expctdZ.append(X[istage].clone())
            if istage == target_stage:
                if number_of_channels_at_target_stage > 0:
                    expctdZ[istage][:,:,:,number_of_channels_at_target_stage:] *= 0
                else:   
                    expctdZ[istage] *= 0
            elif istage > target_stage:   
                expctdZ[istage] *= 0 
        expctdZ = tuple(expctdZ)

        # Instantiation of target class
        layer = AdjointTruncationLayer(
            stride = stride_,
            nlevels = nlevels_
        )

        # Actual values
        with torch.no_grad():
            actualZ = layer(Y)

        # Evaluation
        for istage in range(nlevels_+1):
            self.assertEqual(actualZ[istage].dtype,datatype)                
            self.assertEqual(actualZ[istage].shape,expctdZ[istage].shape,istage) 
            self.assertTrue(torch.allclose(actualZ[istage],expctdZ[istage]),istage) 
    
    @parameterized.expand(
        list(itertools.product(stride,datatype,nsamples,number_of_channels,usegpu))
    )
    def testAdjointTruncationLayerMultiLevelsAtStage3(self,
                                   stride,datatype,nsamples,number_of_channels,usegpu): 
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                print('No GPU device was detected.')
                return 
        else:
            device = torch.device('cpu')
        
        # Parameters       
        stride_ = stride
        datatype_ = datatype
        height_ = 128
        width_ = 192
        nlevels_ = 3
        nsamples_ = nsamples
        number_of_channels_at_target_stage = number_of_channels 

        target_stage = 3  

        # nSamples x nRows x nCols x nDecs
        nDecs = stride_[Direction.VERTICAL] * stride_[Direction.HORIZONTAL]
        nrows = height_ // (stride_[Direction.VERTICAL]**nlevels_)
        ncols = width_ // (stride_[Direction.HORIZONTAL]**nlevels_)
        nrows_ = nrows
        ncols_ = ncols
        X = []
        for istage in range(nlevels_+1):
            if istage == 0:
                X.append(torch.randn(nsamples_,nrows_,ncols_,1,dtype=datatype_,device=device,requires_grad=True)) 
            X.append(torch.randn(nsamples_,nrows_,ncols_,nDecs-1,dtype=datatype_,device=device,requires_grad=True))     
            nrows_ *= stride_[Direction.VERTICAL]
            ncols_ *= stride_[Direction.HORIZONTAL]
        X = tuple(X)

        # Input values
        Y = []
        for istage in range(target_stage):
            Y.append(X[istage])
        Y.append(X[target_stage][:,:,:,:number_of_channels_at_target_stage])
        Y = tuple(Y)

        # Expected values
        nrows_ = nrows
        ncols_ = ncols
        expctdZ = []
        for istage in range(nlevels_+1):
            expctdZ.append(X[istage].clone())
            if istage == target_stage:
                if number_of_channels_at_target_stage > 0:
                    expctdZ[istage][:,:,:,number_of_channels_at_target_stage:] *= 0
                else:   
                    expctdZ[istage] *= 0
            elif istage > target_stage:   
                expctdZ[istage] *= 0 
        expctdZ = tuple(expctdZ)

        # Instantiation of target class
        layer = AdjointTruncationLayer(
            stride = stride_,
            nlevels = nlevels_
        )

        # Actual values
        with torch.no_grad():
            actualZ = layer(Y)

        # Evaluation
        for istage in range(nlevels_+1):
            self.assertEqual(actualZ[istage].dtype,datatype)                
            self.assertEqual(actualZ[istage].shape,expctdZ[istage].shape,istage) 
            self.assertTrue(torch.allclose(actualZ[istage],expctdZ[istage]),istage) 
    """
    @parameterized.expand(
        list(itertools.product(number_of_channels,datatype,nsamples,usegpu))
    )
    def testAdjointTruncationLayerMultiLevelsAtStage3nChsX(self,
                                   number_of_channels,datatype,nsamples,usegpu): 
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                print('No GPU device was detected.')
                return 
        else:
            device = torch.device('cpu')
        
        # Parameters       
        stride_ = [ 2, 2 ]
        datatype_ = datatype
        height_ = 128
        width_ = 192
        nlevels_ = 3
        nsamples_ = nsamples
        number_of_channels_at_target_stage = number_of_channels

        target_stage = 3  

        # nSamples x nRows x nCols x nDecs
        nDecs = stride_[Direction.VERTICAL] * stride_[Direction.HORIZONTAL]
        nrows = height_ // (stride_[Direction.VERTICAL]**nlevels_)
        ncols = width_ // (stride_[Direction.HORIZONTAL]**nlevels_)
        nrows_ = nrows
        ncols_ = ncols
        X = []
        for istage in range(nlevels_+1):
            if istage == 0:
                X.append(torch.randn(nsamples_,nrows_,ncols_,1,dtype=datatype_,device=device,requires_grad=True)) 
            X.append(torch.randn(nsamples_,nrows_,ncols_,nDecs-1,dtype=datatype_,device=device,requires_grad=True))     
            nrows_ *= stride_[Direction.VERTICAL]
            ncols_ *= stride_[Direction.HORIZONTAL]
        X = tuple(X)

        # Input values
        number_of_channels_ = number_of_channels_at_target_stage + nDecs + (nDecs-1)
        Y = []
        for istage in range(target_stage):
            Y.append(X[istage])
        Y.append(X[target_stage][:,:,:,:number_of_channels_at_target_stage])
        Y = tuple(Y)

        # Expceted values
        nrows_ = nrows
        ncols_ = ncols
        expctdZ = []
        for istage in range(nlevels_+1):
            if istage == 0:
                expctdZ.append(torch.zeros(nsamples_,nrows_,ncols_,1,dtype=datatype_,device=device,requires_grad=True)) 
            expctdZ.append(torch.zeros(nsamples_,nrows_,ncols_,nDecs-1,dtype=datatype_,device=device,requires_grad=True))     
            nrows_ *= stride_[Direction.VERTICAL]
            ncols_ *= stride_[Direction.HORIZONTAL]
        expctdZ[0] = Y[0]
        expctdZ[1] = Y[1]
        expctdZ[2] = Y[2]
        expctdZ[3][:,:,:,:number_of_channels_at_target_stage] = Y[3]
        expctdZ = tuple(expctdZ)

        # Instantiation of target class
        layer = AdjointTruncationLayer(
            stride = stride_,
            nlevels = nlevels_
        )

        # Actual values
        with torch.no_grad():
            actualZ = layer(Y)

        # Evaluation
        self.assertEqual(actualZ[0].dtype,datatype)        
        self.assertEqual(actualZ[0].shape,expctdZ[0].shape) 
        self.assertTrue(torch.allclose(actualZ[0],expctdZ[0]))    
        for istage in range(1,target_stage+1): 
            self.assertEqual(actualZ[istage].dtype,datatype)                
            self.assertEqual(actualZ[istage].shape,expctdZ[istage].shape) 
            self.assertTrue(torch.allclose(actualZ[istage],expctdZ[istage]))  
    """

if __name__ == '__main__':
    unittest.main()