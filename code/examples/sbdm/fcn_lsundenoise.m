function xh = fcn_lsundenoise(anet,snet,info,prm,v,sigma)
%FCN_LSUNDENOISE Noise-level conditional AWGN denoiser built with LSUN
%
%   xh = fcn_lsundenoise(anet,snet,info,prm,v,sigma) evaluates the
%   denoiser of Example 10.2 of Muramatsu (2026) with the synthesis
%   dictionary D_theta realised by an LSUN,
%
%     f(v;sigma) = v - D_theta*S_ac^T*phi_sigma(S_ac*E_theta*v),
%     E_theta    = D_theta^T,
%
%   where phi_sigma acts channel-wise on the AC coefficients as
%
%     phi_sigma,p(c) = sigma*(kappa_p/g_p)*tanh(g_p*c/sigma).
%
%   Because E_theta is unitary, the skip connection is realised exactly
%   inside the coefficient domain: with c = E_theta*v the update reads
%   c_ac <- c_ac - phi_sigma(c_ac) and the DC channel passes through, so
%   no explicit skip path is needed. The Jacobian is
%
%     J_f = D_theta*diag(1 - kappa_p*sech^2(g_p*c/sigma))*D_theta^T
%
%   which is exactly symmetric with spectrum in [1-max(kappa_p),1], i.e.
%   f is firmly non-expansive for every parameter value.
%
%   V must be a dlarray with format 'SSCB'. SIGMA is a positive scalar, or
%   an array of size [1 1 1 B] giving one noise level per batch element.
%
% Copyright (c) 2026, Shogo MURAMATSU, All rights reserved.

nStages = numel(prm.th);
nAc = info.nChs - 1;
xh = v;
for s = 1:nStages
    [anet,snet] = fcn_lsunsetangles(anet,snet,info,prm.th{s});
    [ac,dc] = forward(anet,xh);
    kappa = reshape(sigmoid(prm.a{s}),1,1,nAc);
    gain  = reshape(exp(prm.b{s}),1,1,nAc);
    ac = ac - sigma.*(kappa./gain).*tanh(gain.*ac./sigma);
    xh = forward(snet,ac,dc);
end
end
