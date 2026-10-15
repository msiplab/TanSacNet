function theta = salsun2d_round_angle(theta,wordLength)
%SALSUN2D_ROUND_ANGLE Angles rounded as stored by the band design
%
%   theta = salsun2d_round_angle(theta,wordLength) rounds the angles theta
%   (radians) to the angle store of salsun2d_hls_band: theta/pi as a signed
%   fraction of wordLength bits (wordLength-1 fractional bits), rounded to
%   nearest and wrapped into [-1,1), computed as the design computes it. The result is in radians and has the
%   class of theta. See L.FixAngle in salsun2d_hls_layout.
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
% As the band design: the angle rounded and saturated to [27 21], times
% 1/pi rounded to [18 18] (unsigned), rounded and wrapped
a = salsun2d_round_fixed(double(theta),27,21);
c = round(2^18/pi)/2^18;
scale = 2^(wordLength-1);
k = floor(a*c*scale + 0.5);
k = mod(k + scale,2*scale) - scale;
theta = cast(k/scale*pi,'like',theta);
end
