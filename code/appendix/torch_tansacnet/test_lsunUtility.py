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
                device = torch.device('cpu')
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
            datatype = datatype_,
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

    """
    def testForwardTruncationLayerMultiLevels(self): # TODO: Implement ForwardTruncationLayer
    
    def testAdjointTruncationLayer(self): # TODO: Implement AdjointTrunationLayer
        layer = AdjointTruncationLayer()
        self.assertIsInstance(layer, nn.Module)
        self.assertIsNone(layer._forward_pre_hooks_with_kwargs)

    def testAdjointTruncationLayerMultiLevels(self): # TODO: Implement AdjointTrunationLayer

    """

if __name__ == '__main__':
    unittest.main()