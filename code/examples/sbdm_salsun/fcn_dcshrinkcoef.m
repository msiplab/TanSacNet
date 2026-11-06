function [acS,dcS] = fcn_dcshrinkcoef(ab,ac,dc,sigma)
%FCN_DCSHRINKCOEF Shrinkage of the AC (and optionally the DC) coefficients
%
%   [acS,dcS] = fcn_dcshrinkcoef(ab,ac,dc,sigma)
%   AC channel p : c - sigma*(kappa_p/g_p)*tanh(g_p*c/sigma)
%                  kappa_p = sigmoid(ab.a(p)), g_p = exp(ab.b(p))
%                  (as examples/sbdm/fcn_lsundenoise)
%   DC           : the same map with kappa_0 = sigmoid(ab.a0),
%                  g_0 = exp(ab.b0), if ab has the field a0; otherwise the
%                  DC channel passes through.
%   sigma : scalar or [1 1 1 B].
nAc = size(ac,3);
kappa = reshape(sigmoid(ab.a),1,1,nAc);
gain  = reshape(exp(ab.b),1,1,nAc);
acS = ac - sigma.*(kappa./gain).*tanh(gain.*ac./sigma);
if isfield(ab,'a0')
    k0 = sigmoid(ab.a0); g0 = exp(ab.b0);
    dcS = dc - sigma.*(k0./g0).*tanh(g0.*dc./sigma);
else
    dcS = dc;
end
end
