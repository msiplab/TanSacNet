function [sure,divN] = fcn_ssfastsure(ab,ac,sigma,N)
%FCN_SSFASTSURE SURE of the one-stage LSUN denoiser from its AC coefficients
%
%   [sure,divN] = fcn_ssfastsure(ab,ac,sigma,N)
%   ab.a, ab.b : logit of kappa_p, log of g_p (nAc x 1)
%   ac         : AC coefficients E v (H x W x nAc), angles frozen
%   N          : number of pixels
%   Uses only unitarity of E (see examples/sbdm/fcn_lsunsure, "exact"):
%     res  = ||sigma (kappa/g) tanh(g ac/sigma)||^2 / N
%     divN = 1 - sum kappa sech^2(g ac/sigma) / N
%     sure = res - sigma^2 + 2 sigma^2 divN
nAc = size(ac,3);
kappa = reshape(sigmoid(ab.a),1,1,nAc);
gain  = reshape(exp(ab.b),1,1,nAc);
th    = tanh(gain.*ac./sigma);
res   = sum((sigma.*(kappa./gain).*th).^2,[1 2 3])/N;
divN  = 1 - sum(kappa.*(1 - th.^2),[1 2 3])/N;
sure  = res - sigma.^2 + 2*sigma.^2.*divN;
end
