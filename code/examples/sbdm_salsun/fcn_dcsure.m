function [sure,divN,res] = fcn_dcsure(ab,ac,dc,sigma,N)
%FCN_DCSURE Exact SURE of the one-stage LSUN denoiser, with optional DC shrinkage
%
%   [sure,divN,res] = fcn_dcsure(ab,ac,dc,sigma,N)
%   ac, dc : coefficients E v (angles frozen), H x W x nAc (x B), H x W x 1 (x B)
%   ab     : ab.a, ab.b (AC); ab.a0, ab.b0 (DC, optional), see fcn_dcinitab
%   N      : number of pixels
%
%   f(v) = D S(E v), E unitary (D = E'), S coefficient-wise. Then
%     J_f = D diag(S') E,   div f = trace(J_f) = sum_k S'(c_k)
%   with S'(c) = 1 - kappa sech^2(g c/sigma) on each shrunk channel and
%   S' = 1 on a channel that passes through. So
%     div f = N - sum_AC kappa_p sech^2(g_p c/sigma)
%               - sum_DC kappa_0 sech^2(g_0 c/sigma)   (DC term only with DC shrinkage)
%     ||f(v) - v||^2 = ||S(c) - c||^2   (unitarity)
%   This uses only unitarity, so it holds with and without DC leakage.
%   Without ab.a0 this is exactly fcn_ssfastsure.
%   Outputs per batch element: sure, divN = div f / N, res = ||f(v)-v||^2/N.
nAc = size(ac,3);
kappa = reshape(sigmoid(ab.a),1,1,nAc);
gain  = reshape(exp(ab.b),1,1,nAc);
th    = tanh(gain.*ac./sigma);
rs = sum((sigma.*(kappa./gain).*th).^2,[1 2 3]);
sc = sum(kappa.*(1 - th.^2),[1 2 3]);
if isfield(ab,'a0')
    k0 = sigmoid(ab.a0); g0 = exp(ab.b0);
    t0 = tanh(g0.*dc./sigma);
    rs = rs + sum((sigma.*(k0./g0).*t0).^2,[1 2 3]);
    sc = sc + sum(k0.*(1 - t0.^2),[1 2 3]);
end
res  = rs/N;
divN = 1 - sc/N;
sure = res - sigma.^2 + 2*sigma.^2.*divN;
end
