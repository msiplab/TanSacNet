function [x,v,sigma] = fcn_minibatch(exemplars,sigmaRange,isCond,miniBatchSize)
%FCN_MINIBATCH Mini-batch of AWGN observations for score matching
%
%   [x,v,sigma] = fcn_minibatch(exemplars,sigmaRange,isCond,miniBatchSize)
%   draws MINIBATCHSIZE images with replacement from the cell array
%   EXEMPLARS and returns the clean batch X and the noisy batch
%   V = X + SIGMA.*randn, both as dlarrays of format 'SSCB'. SIGMA has size
%   [1 1 1 miniBatchSize], one noise level per batch element, drawn from a
%   log-uniform distribution over SIGMARANGE when ISCOND is true and fixed
%   at SIGMARANGE otherwise.
%
%   Drawing one noise level per image is what lets a single denoiser cover
%   the whole range of noise levels required by the score estimator of
%   Section 10.2.3 of Muramatsu (2026).
%
% Copyright (c) 2026, Shogo MURAMATSU, All rights reserved.

B = miniBatchSize;
sz = size(exemplars{1});
xb = zeros([sz 1 B],'like',exemplars{1});
for b = 1:B
    xb(:,:,1,b) = exemplars{randi(numel(exemplars))};
end
if isCond
    s = single(exp(log(sigmaRange(1)) + ...
        (log(sigmaRange(2))-log(sigmaRange(1)))*rand(1,1,1,B)));
else
    s = single(sigmaRange)*ones(1,1,1,B,'single');
end
if isa(exemplars{1},'gpuArray'), s = gpuArray(s); end
sigma = s;
x = dlarray(xb,'SSCB');
v = dlarray(xb + sigma.*randn(size(xb),'like',xb),'SSCB');
end
