function [loss,grads] = fcn_nsoltloss(anet,snet,info,prm,v,xstar,sigma)
%FCN_NSOLTLOSS MMSE loss and gradients of the NSOLT denoiser
%
%   The NSOLT counterpart of FCN_LSUNLOSS. Used with DLFEVAL.
%
% Copyright (c) 2026, Shogo MURAMATSU, All rights reserved.

xh = fcn_nsoltdenoise(anet,snet,info,prm,v,sigma);
loss = mean((xstar-xh).^2,'all');
grads = dlgradient(loss,prm);
end
