function nets = fcn_ssbuildsalsunsz(inputSize,stride,ovlp,acMask,sigma,bottleneck)
%FCN_SSBUILDSALSUNSZ SA-LSUN networks as fcn_ssbuildsalsun, with estimator size
%
%   nets = fcn_ssbuildsalsunsz(inputSize,stride,ovlp,acMask,sigma,bottleneck)
%   is the same as fcn_ssbuildsalsun, but passes 'BottleneckSize',bottleneck
%   to fcn_createsalsunlgraph2d. Every angle estimator (V0 and the
%   intermediate stages) then projects its standardized local state to
%   'bottleneck' features before the residual blocks.
%   bottleneck = 0 (no projection) gives exactly fcn_ssbuildsalsun
%   (488,717 estimator weights for 4x4 blocks, ovlp 3x3).
%
%   Estimator weights (4x4 blocks, ovlp 3x3, no DC leakage, 3 residual
%   blocks, width 2): 60*b^2 + 664*b + 161 for b > 0.

import tansacnet.salsun.*
import tansacnet.lsun.*

common = {'InputSize',inputSize,'Stride',stride,'OverlappingFactor',ovlp,...
    'NumberOfVanishingMoments',true,...
    'NumberOfNeighborBlocks',[3 3],'NumberOfResidualBlocks',3,'Width',2,...
    'BottleneckSize',bottleneck};
nChs = prod(stride);
strLv = 'Lv1_'; strCmp = 'Cmp1_';

% Step 1: analysis SA-LSUN with coefficient mask (training network)
alg = fcn_createsalsunlgraph2d([],common{:},'Mode','Analyzer');
tlg = alg.replaceLayer([strLv 'AcOut'],...
    maskLayer('Name',[strLv 'AcMask'],'Mask',acMask,...
    'NumberOfChannels',nChs-1));
tlg = tlg.addLayers(lsunChannelConcatenation2dLayer('Name',[strLv strCmp 'Cn']));
tlg = tlg.connectLayers([strLv 'AcMask'],[strLv strCmp 'Cn/ac']);
tlg = tlg.connectLayers([strLv 'DcOut'],[strLv strCmp 'Cn/dc']);
nets.train = dlnetwork(tlg);

% Whole, Reuse (synthesis reuses the analysis-side angles)
wlg = fcn_createsalsunlgraph2d([],common{:},'Mode','Whole','ThetaMode','Reuse');
nets.whole = dlnetwork(wlg);
end
