function c = fcn_ssbuildconstsalsun(inputSize,stride,ovlp,sigma)
%FCN_SSBUILDCONSTSALSUN SA-LSUN graphs with constant (frozen) per-block angles
%
%   Same structure as fcn_ssbuildsalsun, but every angle estimator is
%   Sensei's constant estimator (tansacnet.salsun.fcn_createconstestimatorlgraph):
%   per-block angles that do not depend on the input. The Theta layers
%   have the same names as in the estimator nets, so angles estimated by
%   a trained estimator can be loaded with fcn_sssetconstangles.
%
%     c.ana    : Analyzer (outputs Lv1_AcOut, Lv1_DcOut)
%     c.whole  : Whole, ThetaMode 'Reuse' (perfect-reconstruction check)
%     c.shrink : Whole, ThetaMode 'Reuse', sureShrinkage2dLayer 'Lv1_Shrink'
%                between Lv1_AcOut and Lv1_AcIn (denoiser)

import tansacnet.salsun.*
common = {'InputSize',inputSize,'Stride',stride,'OverlappingFactor',ovlp,...
    'NumberOfVanishingMoments',true,...
    'EstimatorFunction',@tansacnet.salsun.fcn_createconstestimatorlgraph};
nChs = prod(stride);
strLv = 'Lv1_';

c.ana = dlnetwork(fcn_createsalsunlgraph2d([],common{:},'Mode','Analyzer'));
wlg = fcn_createsalsunlgraph2d([],common{:},'Mode','Whole','ThetaMode','Reuse');
c.whole = dlnetwork(wlg);
slg = wlg.disconnectLayers([strLv 'AcOut'],[strLv 'AcIn']);
slg = slg.addLayers(sureShrinkage2dLayer('Name',[strLv 'Shrink'],...
    'NumberOfChannels',nChs-1,'Sigma',sigma));
slg = slg.connectLayers([strLv 'AcOut'],[strLv 'Shrink']);
slg = slg.connectLayers([strLv 'Shrink'],[strLv 'AcIn']);
c.shrink = dlnetwork(slg);
end
