function [numResBlocks,width] = salsun2d_estimator_size(params)
%SALSUN2D_ESTIMATOR_SIZE Number of residual blocks and width of the estimators
%
%   [r,w] = salsun2d_estimator_size(params) returns the number of residual
%   blocks and the width (hidden size / features, as the Width option of
%   the network, hidden = round(w x features)) of the angle estimators in
%   params (salsun2d_extract_params), for salsun2d_hls_layout. It checks
%   that all estimators have the same size.
%
% Requirements: MATLAB R2026b
%
% Copyright (c) 2026, Shogo MURAMATSU
%
% All rights reserved.
%
% Contact address: Shogo MURAMATSU,
%    Faculty of Engineering, Niigata University,
%    8050 2-no-cho Ikarashi, Nishi-ku,
%    Niigata, 950-2181, JAPAN
%
% http://msiplab.eng.niigata-u.ac.jp/
%
ests = [params.V0.Estimator, params.Stages.Estimator];
numResBlocks = numel(ests(1).ResBlocks);
nF = arrayfun(@(e) size(e.Wo,2),ests);
nH = arrayfun(@(e) size(e.ResBlocks(1).W1,1),ests);
assert(all(arrayfun(@(e) numel(e.ResBlocks),ests) == numResBlocks), ...
    'salsun2d_estimator_size:blocks','The estimators differ in the number of residual blocks.')
% Widths used in training are simple fractions: find the one that gives
% every hidden size
candidates = unique([nH./nF, (1:64)/8]);
width = NaN;
for c = candidates
    if all(round(c*nF) == nH)
        width = c;
        break
    end
end
assert(~isnan(width),'salsun2d_estimator_size:width', ...
    'No common width gives the hidden sizes %s for the features %s.',mat2str(nH),mat2str(nF))
% Prefer the simplest fraction that matches
for c = (1:64)/8
    if all(round(c*nF) == nH)
        width = c;
        break
    end
end
end
