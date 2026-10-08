function [mu,sigma] = salsun2d_stats_from_sums(sum1,sum2,nBlocks,epsilon)
%SALSUN2D_STATS_FROM_SUMS Channel statistics from sums over the blocks of a frame
%
%   [mu,sigma] = salsun2d_stats_from_sums(sum1,sum2,nBlocks,epsilon)
%   converts the sums and sums of squares of the estimator input channels
%   (L.NDec x L.NEst, as returned by salsun2d_hls_band or the band kernel,
%   accumulated over a frame of nBlocks blocks) into the mean and the
%   standard deviation plus epsilon used by the standardization, in the
%   same form as salsun2d_band_frame.
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
    sum1 single
    sum2 single
    nBlocks (1,1) single
    epsilon (1,1) double
end
mu = sum1/nBlocks;
v = (sum2 - nBlocks*mu.^2)/(nBlocks-1);
sigma = sqrt(max(v,0)) + single(epsilon);
end
