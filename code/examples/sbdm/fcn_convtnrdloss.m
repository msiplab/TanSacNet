function [loss,grads] = fcn_convtnrdloss(prm,v,xstar,sigma)
%FCN_CONVTNRDLOSS MMSE loss and gradients of the convolutional TNRD baseline
%
%   [loss,grads] = fcn_convtnrdloss(prm,v,xstar,sigma) returns the
%   mean-squared error between the output of FCN_CONVTNRDDENOISE and the
%   reference XSTAR, together with the gradients with respect to PRM. Used
%   with DLFEVAL.
%
% Copyright (c) 2026, Shogo MURAMATSU, All rights reserved.

xh = fcn_convtnrddenoise(prm,v,sigma);
loss = mean((xstar-xh).^2,'all');
grads = dlgradient(loss,prm);
end
