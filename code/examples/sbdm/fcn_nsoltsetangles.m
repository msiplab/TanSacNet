function [anet,snet] = fcn_nsoltsetangles(anet,snet,info,th)
%FCN_NSOLTSETANGLES Tie the analysis and synthesis rotation angles of an NSOLT
%
%   [anet,snet] = fcn_nsoltsetangles(anet,snet,info,th) writes the angle set
%   TH into both networks so that the synthesis operator is the exact adjoint
%   of the analysis operator. TH is a cell array of dlarrays, one per rotation
%   layer, each a single column: an NSOLT rotation layer has no
%   NumberOfBlocks property, so its angles are shared by every block by
%   construction and the operator is shift invariant.
%
%   INFO.aRows and INFO.sRows, built by FCN_NSOLTPAIR, pair the rows of the
%   two Learnables tables by layer name.
%
%   Called inside DLFEVAL, the assignment keeps the tracing of TH, so a single
%   DLGRADIENT call returns the sum of the analysis-side and synthesis-side
%   contributions, i.e. the exact gradient of the tied parameter.
%
% Copyright (c) 2026, Shogo MURAMATSU, All rights reserved.

aval = anet.Learnables.Value;
sval = snet.Learnables.Value;
for k = 1:numel(th)
    aval{info.aRows(k)} = th{k};
    sval{info.sRows(k)} = th{k};
end
anet.Learnables.Value = aval;
snet.Learnables.Value = sval;
end
