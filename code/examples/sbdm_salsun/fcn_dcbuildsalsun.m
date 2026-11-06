function nets = fcn_dcbuildsalsun(inputSize,stride,ovlp,o)
%FCN_DCBUILDSALSUN SA-LSUN networks with the DC options
%
%   nets = fcn_dcbuildsalsun(inputSize,stride,ovlp,o)
%   o.estimator : "net"   -> angle estimator (3x3 neighbour blocks, 3
%                            residual blocks, width 2, BottleneckSize
%                            o.bottleneck), as fcn_ssbuildsalsunsz
%                 "const" -> constant per-block angles, as
%                            fcn_ssbuildconstsalsun
%   o.leak      : false = no DC leakage (as before), true = all angles free
%   o.shrink    : also build the denoiser net (Whole, Reuse, shrinkage)
%   o.dcShrink  : the denoiser net also shrinks the DC channel
%                 (sureShrinkage2dLayer 'Lv1_ShrinkDc', 1 channel, between
%                 Lv1_DcOut and Lv1_DcIn)
%   o.dcInit    : initial DC shrinkage, "ac" (kappa0 0.9, tau0 2) or
%                 "id" (kappa0 0.01, tau0 2); see fcn_dcinitab
%   o.sigma     : sigma of the shrinkage layers
%
%   Output, estimator "net" (same build order as fcn_ssbuildsalsunsz, so
%   with leak = false the step-1 net gets the same initial weights for
%   the same seed):
%     nets.train : Analyzer + maskLayer 'Lv1_AcMask' (all ones) + concat
%     nets.whole : Whole, ThetaMode 'Reuse'
%     nets.shrink: (o.shrink) Whole, Reuse, 'Lv1_Shrink' on AC
%                  [+ 'Lv1_ShrinkDc' on DC]
%   Output, estimator "const":
%     nets.ana, nets.whole, nets.shrink (o.shrink)

import tansacnet.salsun.*
import tansacnet.lsun.*
if ~isfield(o,'shrink'), o.shrink = false; end
if ~isfield(o,'dcShrink'), o.dcShrink = false; end
if ~isfield(o,'dcInit'), o.dcInit = "id"; end
nChs = prod(stride);
strLv = 'Lv1_'; strCmp = 'Cmp1_';
common = {'InputSize',inputSize,'Stride',stride,'OverlappingFactor',ovlp,...
    'NumberOfVanishingMoments',~logical(o.leak)};
switch o.estimator
    case "net"
        common = [common {'NumberOfNeighborBlocks',[3 3],'NumberOfResidualBlocks',3,'Width',2,...
            'BottleneckSize',o.bottleneck}];
        alg = fcn_createsalsunlgraph2d([],common{:},'Mode','Analyzer');
        tlg = alg.replaceLayer([strLv 'AcOut'],...
            maskLayer('Name',[strLv 'AcMask'],'Mask',ones(nChs-1,1,'single'),...
            'NumberOfChannels',nChs-1));
        tlg = tlg.addLayers(lsunChannelConcatenation2dLayer('Name',[strLv strCmp 'Cn']));
        tlg = tlg.connectLayers([strLv 'AcMask'],[strLv strCmp 'Cn/ac']);
        tlg = tlg.connectLayers([strLv 'DcOut'],[strLv strCmp 'Cn/dc']);
        nets.train = dlnetwork(tlg);
    case "const"
        common = [common {'EstimatorFunction',@tansacnet.salsun.fcn_createconstestimatorlgraph}];
        nets.ana = dlnetwork(fcn_createsalsunlgraph2d([],common{:},'Mode','Analyzer'));
    otherwise
        error("unknown estimator %s",o.estimator)
end
wlg = fcn_createsalsunlgraph2d([],common{:},'Mode','Whole','ThetaMode','Reuse');
nets.whole = dlnetwork(wlg);
if o.shrink
    slg = wlg.disconnectLayers([strLv 'AcOut'],[strLv 'AcIn']);
    slg = slg.addLayers(sureShrinkage2dLayer('Name',[strLv 'Shrink'],...
        'NumberOfChannels',nChs-1,'Sigma',o.sigma));
    slg = slg.connectLayers([strLv 'AcOut'],[strLv 'Shrink']);
    slg = slg.connectLayers([strLv 'Shrink'],[strLv 'AcIn']);
    if o.dcShrink
        switch o.dcInit
            case "ac", k0 = 0.9;
            case "id", k0 = 0.01;
            otherwise, error("unknown dcInit %s",o.dcInit)
        end
        slg = slg.disconnectLayers([strLv 'DcOut'],[strLv 'DcIn']);
        slg = slg.addLayers(sureShrinkage2dLayer('Name',[strLv 'ShrinkDc'],...
            'NumberOfChannels',1,'Sigma',o.sigma,'Kappa0',k0,'Tau0',2.0));
        slg = slg.connectLayers([strLv 'DcOut'],[strLv 'ShrinkDc']);
        slg = slg.connectLayers([strLv 'ShrinkDc'],[strLv 'DcIn']);
    end
    nets.shrink = dlnetwork(slg);
end
end
