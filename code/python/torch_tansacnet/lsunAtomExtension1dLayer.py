import math
import torch
import torch.nn as nn
from .lsunLayerExceptions import InvalidDirection, InvalidTargetChannels

class LsunAtomExtension1dLayer(nn.Module):
    """
    LSUNATOMEXTENSION1DLAYER

        コンポーネント別に入力(nComponents=1のみサポート):
            nSamples x nBlks x nChsTotal

        コンポーネント別に出力(nComponents=1のみサポート):
            nSamples x nBlks x nChsTotal

    Requirements: Python 3.10-12.x, PyTorch 2.3/4.x

    Copyright (c) 2026, Shogo MURAMATSU

    All rights reserved.

    Contact address: Shogo MURAMATSU,
        Faculty of Engineering, Niigata University,
        8050 2-no-cho Ikarashi, Nishi-ku,
        Niigata, 950-2181, JAPAN

    https://www.eng.niigata-u.ac.jp/~msiplab/
    """

    # Shift of the blocks
    SHIFTS = { 'Right': 1, 'Left': -1 }

    def __init__(self,
            name='',
            stride=None,
            direction='',
            target_channels=''):
        super(LsunAtomExtension1dLayer, self).__init__()
        self.stride = stride
        self.name = name

        # Target channels
        if target_channels not in ('Sum', 'Difference'):
            raise InvalidTargetChannels(
                '%s : Target should be either of Sum or Difference'\
                % target_channels)
        self.target_channels = target_channels

        # Shift direction
        if direction not in self.SHIFTS:
            raise InvalidDirection(
                '%s : Direction should be either of Right or Left'\
                % direction)
        self.direction = direction

        # Description
        ps = math.ceil(self.stride/2.0)
        pa = math.floor(self.stride/2.0)
        self.description = direction \
            + " shift the " \
            + target_channels.lower() \
            + "-channel Coefs. " \
            + "(ps,pa) = (" \
            + str(ps) + "," \
            + str(pa) + ")"

    def forward(self,X):
        ps = math.ceil(self.stride/2.0)
        # Block butterfly
        Xs = X[...,:ps]
        Xa = X[...,ps:]
        Ys = Xs + Xa
        Ya = Xs - Xa
        # Block circular shift
        shift = self.SHIFTS[self.direction]
        if self.target_channels == 'Difference':
            Ya = torch.roll(Ya,shifts=shift,dims=1)
        else:
            Ys = torch.roll(Ys,shifts=shift,dims=1)
        # Block butterfly
        return torch.cat((Ys+Ya,Ys-Ya),dim=-1)/2.
