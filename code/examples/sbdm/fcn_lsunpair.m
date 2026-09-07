function [anet,snet,info] = fcn_lsunpair(inputSize,stride,ovlpFactor)
%FCN_LSUNPAIR Analysis/synthesis LSUN pair for the SBDM denoiser
%
%   [anet,snet,info] = fcn_lsunpair(inputSize,stride,ovlpFactor) returns a
%   pair of dlnetwork objects realising the LSUN analysis operator
%   E_theta and its adjoint D_theta = E_theta^T, together with a struct
%   INFO describing the angle layout:
%
%     info.nBlocks       # of blocks (= prod(inputSize./stride))
%     info.nChs          # of channels (= prod(stride))
%     info.nAngles       # of angles per block for each rotation layer
%     info.nRotations    # of rotation layers
%
%   The adjoint relation is structural: both networks are parameterised by
%   the same rotation angles, so E^T E = I holds exactly for any angle
%   vector, with no projection onto the Stiefel manifold.
%
% Requirements: MATLAB R2024b, Deep Learning Toolbox
%
% Copyright (c) 2026, Shogo MURAMATSU, All rights reserved.

import tansacnet.lsun.*

alg = fcn_createlsunlgraph2d([],...
    'InputSize',inputSize,...
    'Stride',stride,...
    'OverlappingFactor',ovlpFactor,...
    'NumberOfVanishingMoments',true,...
    'Mode','Analyzer');
slg = fcn_createlsunlgraph2d([],...
    'InputSize',inputSize,...
    'Stride',stride,...
    'OverlappingFactor',ovlpFactor,...
    'NumberOfVanishingMoments',true,...
    'Mode','Synthesizer');

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
end
