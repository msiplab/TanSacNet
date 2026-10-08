function salsun2d_check_band_mask(w)
%SALSUN2D_CHECK_BAND_MASK Check that the mask suits the band design
%
%   salsun2d_check_band_mask(w) raises an error if the coefficient mask
%   in the packed parameters w keeps any antisymmetric channel other than
%   the first one (channel L.Ps+1). The band design salsun2d_hls_band
%   relies on this: it predicts and applies only the L.LastStageNAngles
%   rotations of the last stage that act on that channel.
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
L = salsun2d_hls_layout();
m = w(L.Mask + (1:L.NDec));
if any(m(L.Ps+2:end) ~= 0)
    error('salsun2d_check_band_mask:mask', ...
        ['The coefficient mask keeps antisymmetric channels %s; the band design ' ...
         'supports only channel %d of the antisymmetric part.'], ...
        mat2str(L.Ps + 1 + find(m(L.Ps+2:end) ~= 0)'), L.Ps+1)
end
end
