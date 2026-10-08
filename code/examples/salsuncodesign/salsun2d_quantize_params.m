function params = salsun2d_quantize_params(params,wordLength)
%SALSUN2D_QUANTIZE_PARAMS Round the network parameters to a fixed-point word length
%
%   params = salsun2d_quantize_params(params,wl) rounds the weights and
%   biases of the estimators (W1, B1, W2, B2, Gamma, Beta, Wo, Bo) and
%   the block DCT matrix to signed fixed-point numbers of wl bits, each
%   tensor with its own power-of-two scaling (as salsun2d_fixed_quantizer).
%   The sign flips Mus (+-1) and the mask are exact and left unchanged.
%   The parameters are returned in double precision.
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
    wordLength (1,1) double
end
params = salsun2d_cast_params(params,'double');
q = salsun2d_fixed_quantizer(wordLength);
params.Cvh = q(params.Cvh,'coefs');
params.V0.Estimator = quantizeEstimator(params.V0.Estimator,q);
for k = 1:numel(params.Stages)
    params.Stages(k).Estimator = quantizeEstimator(params.Stages(k).Estimator,q);
end
end

function est = quantizeEstimator(est,q)
for i = 1:numel(est.ResBlocks)
    for f = {'Gamma','Beta','W1','B1','W2','B2'}
        est.ResBlocks(i).(f{1}) = q(est.ResBlocks(i).(f{1}),'weights');
    end
end
est.Wo = q(est.Wo,'weights');
est.Bo = q(est.Bo,'weights');
end
