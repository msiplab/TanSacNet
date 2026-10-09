function results = evaluate_salsun2d_fixed_estimator(options)
%EVALUATE_SALSUN2D_FIXED_ESTIMATOR Fixed-point estimators with static formats
%
%   results = evaluate_salsun2d_fixed_estimator() evaluates the trained
%   network with fixed-point angle estimators whose formats are fixed in
%   advance (salsun2d_calibrate_formats on the calibration frames), as a
%   hardware implementation needs, while the data path (block DCT,
%   rotations, atom extensions) stays in floating point. The weights of
%   the estimators are rounded per tensor. The MSE of the reconstruction
%   on the evaluation frames is reported relative to the floating-point
%   network with the same statistics.
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
    options.ParamsFile {mustBeTextScalar} = fullfile(fileparts(mfilename('fullpath')),'results','none.mat')
    options.CalibrationFrames (1,:) double = 1:10
    options.EvaluationFrames (1,:) double = 11:15
    options.SignalWordLengths (1,:) double = [10 12 14 16]
    options.WeightWordLengths (1,:) double = [10 12 16]
    options.AngleWordLengths (1,:) double = [12 14 16]
    options.Margin (1,1) double = 1
    % Common: one format for the inputs of the fully connected layers
    % (features, LayerNorm and GELU outputs of every estimator) and one
    % for all weight matrices, as one shared fixed-point layer needs; the
    % hidden layer before GELU and the angles stay in floating point.
    options.Common (1,1) logical = false
end
here = fileparts(mfilename('fullpath'));
addpath(here);
R = load(options.ParamsFile,'params');
params = salsun2d_cast_params(R.params,'double');
u = salsun2d_wave_data();
uCal = u(:,:,options.CalibrationFrames);
uEval = u(:,:,options.EvaluationFrames);

mseRef = mseOf(uEval,params,@(v,tag) v);
fprintf('floating point: MSE %.4g (frames %s)\n',mseRef,mat2str(options.EvaluationFrames([1 end])));
results = struct('Signals',{},'Weights',{},'Angles',{},'MseRatio',{});
fprintf('%8s %8s %8s %10s\n','signals','weights','angles','MSE/ref');
for wa = options.AngleWordLengths
    for ws = options.SignalWordLengths
        if options.Common
            wl = struct('features',ws,'ln',ws,'act',ws);
        else
            wl = struct('features',ws,'ln',ws,'z1',ws,'act',ws,'angles',wa);
        end
        formats = salsun2d_calibrate_formats(uCal,params,wl,Margin=options.Margin);
        if options.Common
            formats = commonFormat(formats);
        end
        q = salsun2d_static_quantizer(formats);
        for ww = options.WeightWordLengths
            if options.Common
                p = quantizeEstimatorWeightsCommon(params,ww);
            else
                p = quantizeEstimatorWeights(params,ww);
            end
            r = mseOf(uEval,p,q)/mseRef;
            fprintf('%8d %8d %8d %10.4f\n',ws,ww,wa,r);
            results(end+1) = struct('Signals',ws,'Weights',ww,'Angles',wa,'MseRatio',r); %#ok<AGROW>
        end
    end
end
end

function formats = commonFormat(formats)
% The smallest fraction length of all formats (the largest range) for all
fl = min(cellfun(@(f) f(2),values(formats)));
k = keys(formats);
fprintf('signals: format %d.%d (fraction lengths per signal %s)\n',formats(k{1})*[1;0],fl, ...
    strjoin(cellfun(@(x) sprintf('%s %d',x,formats(x)*[0;1]),k,'UniformOutput',false),', '));
for key = keys(formats)
    f = formats(key{1});
    formats(key{1}) = [f(1) fl];
end
end

function params = quantizeEstimatorWeightsCommon(params,wl)
% One format for all weight matrices W1, W2, Wo of all estimators, from
% the largest magnitude; the biases and LayerNorm parameters stay in
% floating point (added in the accumulator or applied in floating point)
ests = [params.V0.Estimator, params.Stages.Estimator];
m = 0;
for e = ests
    m = max([m, max(abs(e.Wo(:))), arrayfun(@(r) max(abs([r.W1(:); r.W2(:)])),e.ResBlocks)]);
end
fl = wl - 1 - ceil(log2(m));
fprintf('weights: max %.3g, format %d.%d\n',m,wl,fl);
r = @(v) salsun2d_round_fixed(v,wl,fl);
params.V0.Estimator = roundOne(params.V0.Estimator,r);
for s = 1:numel(params.Stages)
    params.Stages(s).Estimator = roundOne(params.Stages(s).Estimator,r);
end
end

function est = roundOne(est,r)
for i = 1:numel(est.ResBlocks)
    est.ResBlocks(i).W1 = r(est.ResBlocks(i).W1);
    est.ResBlocks(i).W2 = r(est.ResBlocks(i).W2);
end
est.Wo = r(est.Wo);
end

function m = mseOf(u,params,q)
m = 0;
for t = 1:size(u,3)
    y = salsun2d_infer(u(:,:,t),params,Quantizer=q,EstimatorTags=true);
    m = m + mean((u(:,:,t) - y).^2,'all');
end
m = m/size(u,3);
end

function params = quantizeEstimatorWeights(params,wl)
% Each weight tensor of the estimators with its own power-of-two scaling
q = salsun2d_fixed_quantizer(wl);
params.V0.Estimator = quantizeOne(params.V0.Estimator,q);
for s = 1:numel(params.Stages)
    params.Stages(s).Estimator = quantizeOne(params.Stages(s).Estimator,q);
end
end

function est = quantizeOne(est,q)
for i = 1:numel(est.ResBlocks)
    for f = {'Gamma','Beta','W1','B1','W2','B2'}
        est.ResBlocks(i).(f{1}) = q(est.ResBlocks(i).(f{1}),'weights');
    end
end
est.Wo = q(est.Wo,'weights');
est.Bo = q(est.Bo,'weights');
end
