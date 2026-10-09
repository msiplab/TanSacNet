function q = salsun2d_static_quantizer(formats)
%SALSUN2D_STATIC_QUANTIZER Quantizer handle with fixed fixed-point formats
%
%   q = salsun2d_static_quantizer(formats) returns q(v,tag) for the
%   Quantizer option of salsun2d_infer. formats is a containers.Map from
%   tags (e.g. 'z1:3', with EstimatorTags=true) to [wordLength
%   fractionLength]; signals with other tags are left unchanged. Each
%   value is rounded to the nearest multiple of 2^-fractionLength (ties
%   upward, as AP_RND of ap_fixed) and saturated to the signed range of
%   the word length (AP_SAT), as the casts at the layer outputs of the
%   fixed-point hardware.
%
%   See also salsun2d_calibrate_formats.
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
    formats containers.Map
end
q = @(v,tag) quantize(v,tag,formats);
end

function v = quantize(v,tag,formats)
if ~isKey(formats,tag)
    return
end
f = formats(tag);
v = salsun2d_round_fixed(v,f(1),f(2));
end
