function nets = fcn_ssbuildsalsun(inputSize,stride,ovlp,acMask,sigma)
%FCN_SSBUILDSALSUN SA-LSUN networks for the two-step self-supervised denoiser
%
%   nets = fcn_ssbuildsalsun(inputSize,stride,ovlp,acMask,sigma) builds,
%   following examples/salsun/main_salsun2d.mlx:
%
%     nets.train  : Analyzer + maskLayer on AcOut + channel concatenation
%                   (step 1, loss ||v||^2 - ||F(v)||^2)
%     nets.shrink : Whole, ThetaMode 'Reuse', with sureShrinkage2dLayer
%                   'Lv1_Shrink' between Lv1_AcOut and Lv1_AcIn (step 2)
%     nets.whole  : Whole, ThetaMode 'Reuse', nothing in between
%                   (perfect-reconstruction check ||DEx - x||/||x||)
%
%   Estimator: Sensei's default (3x3 neighbour blocks, 3 residual blocks,
%   width 2), as in the example. Learnables of nets.train are copied to
%   the other nets by layer name with fcn_copylearnablesbyname.

import tansacnet.salsun.*
import tansacnet.lsun.*

common = {'InputSize',inputSize,'Stride',stride,'OverlappingFactor',ovlp,...
    'NumberOfVanishingMoments',true,...
    'NumberOfNeighborBlocks',[3 3],'NumberOfResidualBlocks',3,'Width',2};
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

% Step 2: Whole, Reuse, with shrinkage of the AC coefficients
slg = wlg.disconnectLayers([strLv 'AcOut'],[strLv 'AcIn']);
slg = slg.addLayers(sureShrinkage2dLayer('Name',[strLv 'Shrink'],...
    'NumberOfChannels',nChs-1,'Sigma',sigma));
slg = slg.connectLayers([strLv 'AcOut'],[strLv 'Shrink']);
slg = slg.connectLayers([strLv 'Shrink'],[strLv 'AcIn']);
nets.shrink = dlnetwork(slg);
end
