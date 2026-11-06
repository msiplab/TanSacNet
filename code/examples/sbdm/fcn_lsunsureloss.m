function [loss,grads] = fcn_lsunsureloss(anet,snet,info,prm,u,sigma)
%FCN_LSUNSURELOSS Self-supervised (SURE) loss and gradients, for DLFEVAL
%
%   Drop-in replacement of FCN_LSUNLOSS when no clean reference exists:
%   the argument XSTAR is gone, and the loss is the batch mean of
%   FCN_LSUNSURE. See FCN_LSUNSURE for the divergence.

sure  = fcn_lsunsure(anet,snet,info,prm,u,sigma);
loss  = mean(sure,'all');
grads = dlgradient(loss,prm);
end
