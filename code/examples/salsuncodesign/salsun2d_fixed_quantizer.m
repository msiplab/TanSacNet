function q = salsun2d_fixed_quantizer(wordLengths,options)
%SALSUN2D_FIXED_QUANTIZER Quantizer handle for word-length studies
%
%   q = salsun2d_fixed_quantizer(wl) returns q(v,tag) for the Quantizer
%   option of salsun2d_infer. wl is a scalar word length for every tag,
%   or a struct with a word length per tag ('coefs', 'rotation',
%   'features', 'ln', 'z1', 'act', 'angles'); a missing tag or a value
%   of Inf leaves that signal unquantized.
%
%   Each tensor is rounded to a signed fixed-point number of the given
%   word length with a power-of-two scaling chosen from its own largest
%   magnitude (block floating point per tensor). This is the best case
%   for a fixed-point implementation whose ranges are known; fixed
%   scalings chosen from training data would be slightly worse.
%
%   salsun2d_fixed_quantizer(wl,RowScaling=tags) uses one scaling per
%   row (first dimension) instead of per tensor for the listed tags,
%   e.g. {'coefs'}: one scaling per block channel, as a fixed-point
%   data path with a word length per channel would have.
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
    wordLengths
    options.RowScaling cell = {}
end
q = @(v,tag) quantize(v,wordLengthOf(wordLengths,tag),any(strcmp(tag,options.RowScaling)));
end

function wl = wordLengthOf(wordLengths,tag)
if isstruct(wordLengths)
    if isfield(wordLengths,tag)
        wl = wordLengths.(tag);
    else
        wl = Inf;
    end
else
    wl = wordLengths;
end
end

function v = quantize(v,wl,perRow)
if isinf(wl) || isempty(v)
    return
end
if perRow
    m = max(abs(v),[],2:ndims(v));      % one scaling per row
else
    m = max(abs(v(:)));
end
m(m == 0) = 1;
scale = 2.^(wl - 1 - ceil(log2(m)));    % largest magnitude fits in wl-1 bits
limit = 2^(wl-1) - 1;
v = max(min(round(v.*scale),limit),-limit)./scale;
end
