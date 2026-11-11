function c = salsun2d_angle_centers(params,x,options)
%SALSUN2D_ANGLE_CENTERS Centers of the angles for their storage as offsets
%
%   c = salsun2d_angle_centers(params,x,AngleBits=n) returns the center of
%   every angle of the band design (147 integers, in units of 2^(1-n), the
%   layout of P_CTR in proto/band_df.h: the 28 of W and the 28 of U of the
%   initial rotation, the 28 of stages 1 to 3, the first 7 of stage 4):
%   the median over the blocks of the frame x of the angle rounded to n
%   bits (salsun2d_band_fixed_model with AngleBits=n, statistics of x
%   itself). The angles then go to the angle store as offsets from their
%   centers (salsun2d_band_fixed_model with AngleCenters=c and
%   OffsetBits; proto/band_df.cpp with QBITS=n and QOFS). The zero angles
%   of W have the center 0.
%
%   The angles as offsets from fixed centers are an idea of Shogo
%   Muramatsu.
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
    params (1,1) struct
    x (:,:) single
    options.AngleBits (1,1) double {mustBeInteger,mustBeInRange(options.AngleBits,2,18)} = 9
end
n = options.AngleBits;
[pQ,q] = salsun2d_band_fixed_model(params,AngleBits=n);
[~,~,th] = salsun2d_infer(x,pQ,Quantizer=q,EstimatorTags=true);
rows = {1:56, 1:28, 1:28, 1:28, 1:7};
c = cell(numel(rows),1);
for k = 1:numel(rows)
    A = round(double(th{k}(rows{k},:))/pi*2^(n-1));
    c{k} = round(median(A,2));
end
c{1}(1:7) = 0;                     % the zero angles of W
c = vertcat(c{:});
c = mod(c + 2^(n-1),2^n) - 2^(n-1);
end
