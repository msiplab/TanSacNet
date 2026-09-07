function [anet,snet,info] = fcn_nsoltpair(inputSize,decFactor,nChannels,ppOrder)
%FCN_NSOLTPAIR Analysis/synthesis NSOLT pair, the shift-invariant counterpart
%
%   [anet,snet,info] = fcn_nsoltpair(inputSize,decFactor,nChannels,ppOrder)
%   returns a pair of dlnetwork objects realising an NSOLT analysis operator
%   E_theta and its adjoint D_theta = E_theta^T, built with the SaivDr
%   package, together with a struct INFO of the same shape as the one
%   returned by FCN_LSUNPAIR.
%
%   NSOLT is the shift-invariant ancestor of the LSUN: it carries the same
%   rotation-angle parameterisation, so the dictionary is a tight frame for
%   every parameter value with no projection, but its angles are shared by
%   every block by construction -- the rotation layers have no
%   NumberOfBlocks property -- and the operator therefore has a polyphase
%   representation. Two consequences matter here:
%
%     * with sum(nChannels) == prod(decFactor) the transform is critically
%       sampled and unitary, and is then essentially the uniform LSUN;
%     * with sum(nChannels) > prod(decFactor) it is an oversampled tight
%       frame, so E^T*E = I still holds -- Propositions 1 and 3 survive --
%       but E*E^T is a projection rather than the identity, so white noise
%       of level sigma in the image does *not* map to white noise of level
%       sigma in the coefficients and the conditioning of Proposition 2 is
%       no longer exact.
%
%   Requires SaivDr on the path (run its setpath in the package root).
%
% Copyright (c) 2026, Shogo MURAMATSU, All rights reserved.

arguments
    inputSize (1,2) double
    decFactor (1,2) double
    nChannels (1,2) double
    ppOrder   (1,2) double
end

import saivdr.dcnn.*

args = {'InputSize',inputSize,...
    'NumberOfChannels',nChannels,...
    'DecimationFactor',decFactor,...
    'PolyPhaseOrder',ppOrder,...
    'NumberOfVanishingMoments',true};

alg = fcn_creatensoltlgraph2d([],args{:},'Mode','Analyzer');
slg = fcn_creatensoltlgraph2d([],args{:},'Mode','Synthesizer');

% the analyser is made the exact adjoint of the synthesiser
snet = dlnetwork(slg);
evalc('alg = fcn_cpparamssyn2ana(alg,layerGraph(snet));');
anet = dlnetwork(alg);

% Rows of the two Learnables tables that carry rotation angles, paired by
% name: the synthesis counterpart of analysis layer X is named X~. Pairing by
% name rather than by position is necessary because the synthesis network
% lists its rotation layers in the reverse order.
aL = anet.Learnables;
sL = snet.Learnables;
info.aRows = find(string(aL.Parameter) == "Angles");
info.sRows = zeros(numel(info.aRows),1);
for i = 1:numel(info.aRows)
    nm = string(aL.Layer(info.aRows(i))) + "~";
    j = find(string(sL.Layer) == nm & string(sL.Parameter) == "Angles");
    assert(isscalar(j),"no unique synthesis counterpart of %s",nm);
    info.sRows(i) = j;
end

info.nBlocks    = prod(inputSize./decFactor);
info.nChs       = sum(nChannels);
info.nAngles    = cellfun(@(v) numel(v),aL.Value(info.aRows));
info.nRotations = numel(info.nAngles);
info.inputSize  = inputSize;
info.stride     = decFactor;
info.ovlpFactor = ppOrder+1;
info.nChannels  = nChannels;
info.redundancy = sum(nChannels)/prod(decFactor);
end
