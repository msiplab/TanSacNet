function [paramsQ,q] = salsun2d_band_fixed_model(params)
%SALSUN2D_BAND_FIXED_MODEL Reference model of the fixed-point band design
%
%   [paramsQ,q] = salsun2d_band_fixed_model(params) returns the parameters
%   with the estimator weight matrices rounded to L.FixWeight and the
%   quantizer q that rounds the inputs of the fully connected layers of
%   every estimator ('features', 'ln', 'act') to L.FixSignal, so that
%
%     salsun2d_infer(x,paramsQ,Quantizer=q,EstimatorTags=true,...)
%
%   computes what the fixed-point estimators of salsun2d_hls_band compute,
%   up to the rounding of the single-precision parts (standardization,
%   LayerNorm scaling, GELU), the biases rounded to the accumulator
%   format, and the order of the floating-point operations.
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
q = salsun2d_static_quantizer(formats);
end

function est = roundEstimator(est,r)
for i = 1:numel(est.ResBlocks)
    est.ResBlocks(i).W1 = r(est.ResBlocks(i).W1);
    est.ResBlocks(i).W2 = r(est.ResBlocks(i).W2);
end
est.Wo = r(est.Wo);
end
