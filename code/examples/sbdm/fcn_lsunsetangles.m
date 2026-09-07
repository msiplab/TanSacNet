function [anet,snet] = fcn_lsunsetangles(anet,snet,info,th)
%FCN_LSUNSETANGLES Tie the analysis and synthesis rotation angles
%
%   [anet,snet] = fcn_lsunsetangles(anet,snet,info,th) writes the angle
%   set TH into both networks so that the synthesis operator is the exact
%   adjoint of the analysis operator. TH is a cell array of dlarrays, one
%   per rotation layer, each of size [nAngles(k) x 1] (uniform LSUN, the
%   angles are broadcast over all blocks) or [nAngles(k) x nBlocks]
%   (locally-structured LSUN).
%
%   The synthesis network lists its rotation layers in reverse order, so
%   layer k of the analysis network is paired with layer nRotations+1-k of
%   the synthesis network.
%
%   Called inside DLFEVAL, the assignment keeps the tracing of TH, so a
%   single DLGRADIENT call returns the sum of the analysis-side and
%   synthesis-side contributions, i.e. the exact gradient of the tied
%   parameter.
%
% Copyright (c) 2026, Shogo MURAMATSU, All rights reserved.

nR = info.nRotations;
aval = anet.Learnables.Value;
sval = snet.Learnables.Value;
for k = 1:nR
    nCol = size(th{k},2);
    if nCol == 1
        % uniform LSUN: one angle set broadcast over every block, so the same
        % parameter vector drives a transform of any image size
        angles = repmat(th{k},1,info.nBlocks);
    elseif nCol == info.nBlocks
        angles = th{k};
    else
        error("tansacnet:lsun:BlockCountMismatch",...
            ['A locally-structured angle set is tied to the block grid it was ' ...
             'trained on: rotation %d carries %d blocks but this network has ' ...
             '%d. Use a uniform angle set (one column) to change image size.'],...
            k,nCol,info.nBlocks);
    end
    aval{k} = angles;
    sval{nR+1-k} = angles;
end
anet.Learnables.Value = aval;
snet.Learnables.Value = sval;
end
