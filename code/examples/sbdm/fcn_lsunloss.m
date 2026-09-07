function [loss,grads] = fcn_lsunloss(anet,snet,info,prm,v,xstar,sigma)
%FCN_LSUNLOSS MMSE loss and gradients of the LSUN denoiser
%
%   [loss,grads] = fcn_lsunloss(anet,snet,info,prm,v,xstar,sigma) returns
%   the mean-squared error between the output of FCN_LSUNDENOISE and the
%   reference XSTAR, together with the gradients with respect to PRM. Used
%   with DLFEVAL.
%
%   Minimising this loss over noise levels drawn from a log-uniform
%   distribution trains the MMSE estimator of Section 10.2.3, whose
%   residual gives the score by Tweedie's formula:
%
%     grad log p_sigma(u) = (f(u;sigma) - u)/sigma^2.
%
% Copyright (c) 2026, Shogo MURAMATSU, All rights reserved.

xh = fcn_lsundenoise(anet,snet,info,prm,v,sigma);
loss = mean((xstar-xh).^2,'all');
grads = dlgradient(loss,prm);
end
