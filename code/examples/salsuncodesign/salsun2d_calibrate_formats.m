function [formats,ranges] = salsun2d_calibrate_formats(u,params,wordLengths,options)
%SALSUN2D_CALIBRATE_FORMATS Fixed-point formats of the estimator signals
%
%   formats = salsun2d_calibrate_formats(u,params,wl) runs salsun2d_infer
%   on the frames u(:,:,t) and chooses, for each estimator signal class
%   and estimator ('features:k', 'ln:k', 'z1:k', 'act:k', 'angles:k'), the
%   fraction length that holds the largest magnitude seen in wl bits:
%   fl = wl - 1 - ceil(log2(max|v| * Margin)). wl is a struct of word
%   lengths with fields features, ln, z1, act, angles (a missing field
%   leaves that class in floating point). formats is a containers.Map
%   for salsun2d_static_quantizer, ranges a containers.Map of the
%   largest magnitudes.
%
%   salsun2d_calibrate_formats(...,Margin=m) leaves headroom (default 1:
%   values beyond the calibration range saturate).
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
arguments
    u (:,:,:) {mustBeFloat}
    params (1,1) struct
    wordLengths (1,1) struct
    options.Margin (1,1) double {mustBePositive} = 1
end
ranges = containers.Map('KeyType','char','ValueType','double');
rec = @(v,tag) record(v,tag,ranges);
for t = 1:size(u,3)
    salsun2d_infer(u(:,:,t),params,Quantizer=rec,EstimatorTags=true);
end
formats = containers.Map('KeyType','char','ValueType','any');
for key = keys(ranges)
    k = key{1};
    cls = extractBefore(k,':');
    if isempty(cls) || ~isfield(wordLengths,cls)
        continue
    end
    wl = wordLengths.(cls);
    m = max(ranges(k)*options.Margin,eps);
    formats(k) = [wl, wl - 1 - ceil(log2(m))];
end
end

function v = record(v,tag,ranges)
m = double(max(abs(v(:))));
if isKey(ranges,tag)
    ranges(tag) = max(ranges(tag),m);
else
    ranges(tag) = m;
end
end
