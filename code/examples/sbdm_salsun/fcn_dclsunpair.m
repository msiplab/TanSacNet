function [anet,snet,info] = fcn_dclsunpair(inputSize,stride,ovlpFactor,leak)
%FCN_DCLSUNPAIR LSUN analysis/synthesis pair, with or without DC leakage
%
%   [anet,snet,info] = fcn_dclsunpair(inputSize,stride,ovlpFactor,leak)
%   is examples/sbdm/fcn_lsunpair with one option:
%     leak = false : 'NumberOfVanishingMoments' true (no DC leakage, the
%                    first nChs/2-1 angles of V0 and V0~ are forced to 0)
%                    -> exactly fcn_lsunpair
%     leak = true  : 'NumberOfVanishingMoments' false (DC leakage allowed,
%                    all angles free)
%   With zero angles both give the same transform (block DCT + lattice
%   with zero angles). Mus are 1 in both cases (layer default).
%   info.leak = leak.

import tansacnet.lsun.*
nvm = ~logical(leak);
alg = fcn_createlsunlgraph2d([],...
    'InputSize',inputSize,'Stride',stride,'OverlappingFactor',ovlpFactor,...
    'NumberOfVanishingMoments',nvm,'Mode','Analyzer');
slg = fcn_createlsunlgraph2d([],...
    'InputSize',inputSize,'Stride',stride,'OverlappingFactor',ovlpFactor,...
    'NumberOfVanishingMoments',nvm,'Mode','Synthesizer');
anet = dlnetwork(alg);
evalc('slg = fcn_cpparamsana2syn(slg,layerGraph(anet));');
snet = dlnetwork(slg);

info.nBlocks    = prod(inputSize./stride);
info.nChs       = prod(stride);
info.nAngles    = cellfun(@(v) size(v,1),anet.Learnables.Value);
info.nRotations = numel(info.nAngles);
info.inputSize  = inputSize;
info.stride     = stride;
info.ovlpFactor = ovlpFactor;
info.leak       = logical(leak);
end
