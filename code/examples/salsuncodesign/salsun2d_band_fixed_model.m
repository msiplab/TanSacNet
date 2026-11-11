function [paramsQ,q] = salsun2d_band_fixed_model(params,options)
%SALSUN2D_BAND_FIXED_MODEL Reference model of the fixed-point band design
%
%   [paramsQ,q] = salsun2d_band_fixed_model(params) returns the parameters
%   with the estimator weight matrices rounded to L.FixWeight and the
%   quantizer q that rounds the inputs of the fully connected layers of
%   every estimator ('features', 'ln', 'act') to L.FixSignal and the
%   angles ('angles') to the angle store (L.FixAngle,
%   salsun2d_round_angle), so that
%
%     salsun2d_infer(x,paramsQ,Quantizer=q,EstimatorTags=true,...)
%
%   computes what the fixed-point estimators of salsun2d_hls_band compute,
%   up to the rounding of the single-precision parts (standardization,
%   LayerNorm scaling, GELU), the biases rounded to the accumulator
%   format, and the order of the floating-point operations.
%
%   [...] = salsun2d_band_fixed_model(params,AngleBits=n) rounds the angles
%   to n bits instead of the 18 of L.FixAngle (the dataflow kernel of
%   proto/ with the angles on chip, QBITS).
%
%   [...] = salsun2d_band_fixed_model(params,AngleBits=n,AngleCenters=c,
%   OffsetBits=b) stores the angles as offsets of b bits from the centers
%   c (salsun2d_angle_centers), saturated: the angle is c + r with r the
%   offset of the n-bit angle from c (wrapped) limited to
%   [-2^(b-1), 2^(b-1)-1] (proto/band_df.cpp with QOFS=b). The centers and
%   offsets are an idea of Shogo Muramatsu.
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
    options.AngleBits (1,1) double {mustBeInteger,mustBeInRange(options.AngleBits,2,18)} = 18
    options.AngleCenters (:,1) double {mustBeInteger} = []
    options.OffsetBits (1,1) double {mustBeInteger,mustBePositive} = 18
end
if ~isempty(options.AngleCenters)
    assert(numel(options.AngleCenters) == 147,'147 centers (salsun2d_angle_centers)')
end
L = salsun2d_hls_layout();
r = @(W) cast(salsun2d_round_fixed(double(W),L.FixWeight(1),L.FixWeight(2)),'like',W);
paramsQ = params;
paramsQ.V0.Estimator = roundEstimator(paramsQ.V0.Estimator,r);
for s = 1:numel(paramsQ.Stages)
    paramsQ.Stages(s).Estimator = roundEstimator(paramsQ.Stages(s).Estimator,r);
end
formats = containers.Map('KeyType','char','ValueType','any');
for k = 1:L.NEst
    for tag = {'features','ln','act'}
        formats(sprintf('%s:%d',tag{1},k)) = L.FixSignal;
    end
end
qs = salsun2d_static_quantizer(formats);
q = @(v,tag) roundAngles(qs(v,tag),tag,options.AngleBits,options.AngleCenters,options.OffsetBits);
end

function v = roundAngles(v,tag,n,c,b)
if startsWith(tag,'angles')
    v = salsun2d_round_angle(v,n);
    if ~isempty(c) && b < n
        % centers of estimator k (rows of its angles; the others: 0)
        k = sscanf(tag,'angles:%d');
        first = [0 56 84 112 140]; count = [56 28 28 28 7];
        ck = zeros(size(v,1),1);
        m = min(count(k),size(v,1));
        ck(1:m) = c(first(k) + (1:m));
        wrapN = @(r) mod(r + 2^(n-1),2^n) - 2^(n-1);
        A = round(double(v)/pi*2^(n-1));
        r = min(max(wrapN(A - ck),-2^(b-1)),2^(b-1) - 1);
        v = cast(wrapN(ck + r)/2^(n-1)*pi,'like',v);
    end
end
end

function est = roundEstimator(est,r)
for i = 1:numel(est.ResBlocks)
    est.ResBlocks(i).W1 = r(est.ResBlocks(i).W1);
    est.ResBlocks(i).W2 = r(est.ResBlocks(i).W2);
end
est.Wo = r(est.Wo);
end
