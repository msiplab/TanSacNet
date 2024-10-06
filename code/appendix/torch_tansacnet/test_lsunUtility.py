import itertools
import unittest
from parameterized import parameterized
import torch
from torch_tansacnet.lsunUtility import Direction, ForwardTruncationLayer, AdjointTruncationLayer
from torch_tansacnet.lsunUtility import rot_

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

    """
    def testForwardTruncationLayer(self): # TODO: Implement ForwardTruncationLayer
        layer = ForwardTruncationLayer()
        self.assertIsInstance(layer, nn.Module)
        self.assertIsNone(layer._forward_pre_hooks_with_kwargs)

    def testAdjointTruncationLayer(self): # TODO: Implement AdjointTruncationLayer
        layer = AdjointTruncationLayer()
        self.assertIsInstance(layer, nn.Module)
        self.assertIsNone(layer._forward_pre_hooks_with_kwargs)
    """

if __name__ == '__main__':
    unittest.main()