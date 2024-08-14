import torch
import torch.nn as nn

class LsunIdentityLayer(nn.Module):
    """
    LSUNIDENTITYLAYER

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
        name=''):
        super(LsunIdentityLayer, self).__init__()
        self.name = name
        self.description = "Identity"        

    def forward(self,X):
        return X
