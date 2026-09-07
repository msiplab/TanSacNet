function xh = fcn_nsoltdenoise(anet,snet,info,prm,v,sigma)
%FCN_NSOLTDENOISE The denoiser of Example 10.2 on an NSOLT dictionary
%
%   Identical in form to FCN_LSUNDENOISE, with the LSUN replaced by an NSOLT.
%   The structure carries over because an NSOLT is a tight frame,
%   E^T*E = I, so Proposition 1 (the skip connection is internal) and
%   Proposition 3 (the Jacobian is symmetric with spectrum in (0,1]) hold as
%   stated.
%
%   Proposition 2 does not. Conditioning on the noise level by dividing the
%   coefficients by sigma is exact only when E is unitary, i.e. when
%   E*E^T = I as well, which requires critical sampling. For an oversampled
%   NSOLT E*E^T is a projection, so white noise of level sigma in the image
%   does not map to white noise of level sigma in the coefficients and the
%   normalisation below is only nominal. That is the point of including it.
%
% Copyright (c) 2026, Shogo MURAMATSU, All rights reserved.

nStages = numel(prm.th);
nAc = info.nChs - 1;
xh = v;
for s = 1:nStages
    [anet,snet] = fcn_nsoltsetangles(anet,snet,info,prm.th{s});
    [ac,dc] = forward(anet,xh);
    kappa = reshape(sigmoid(prm.a{s}),1,1,nAc);
    gain  = reshape(exp(prm.b{s}),1,1,nAc);
    ac = ac - sigma.*(kappa./gain).*tanh(gain.*ac./sigma);
    xh = forward(snet,ac,dc);
end
end
