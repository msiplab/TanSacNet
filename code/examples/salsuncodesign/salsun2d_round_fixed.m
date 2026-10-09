function v = salsun2d_round_fixed(v,wordLength,fractionLength)
%SALSUN2D_ROUND_FIXED Round to a signed fixed-point format
%
%   v = salsun2d_round_fixed(v,wl,fl) rounds v to the nearest multiple of
%   2^-fl, ties upward (AP_RND of ap_fixed), and saturates to the range
%   of a signed wl-bit number (AP_SAT). The result has the class of v.
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
scale = 2^fractionLength;
lo = -2^(wordLength-1);
hi = 2^(wordLength-1) - 1;
v = cast(min(max(floor(double(v)*scale + 0.5),lo),hi)/scale,'like',v);
end
