function [sure,divMc,divFr,sureFr] = fcn_sspilotsure(ab,pre,sigma,epsFd)
%FCN_SSPILOTSURE SURE of the pilot SA-LSUN denoiser with fixed probes
%
%   f(v) = D_th S(E_th v), th = angles estimated from p = N(v), D_th = E_th'.
%   All coefficients are precomputed once (pre, see fcn_sspilotrunimage):
%     ac0, dc0 : E_th0 v,              th0 = th(N(v))
%     acJ, dcJ : E_thj (v + eps b_j),  thj = th(N(v + eps b_j))
%     eAc, eDc : E_thj b_j
%     e0Ac,e0Dc: E_th0 b_j
%   Since D is unitary, b_j'(f(v+eps b_j) - f(v))
%     = eJ' S(cJ) - e0J' S(c0)   (S = identity on DC),
%   so the Monte-Carlo divergence of the WHOLE chain (N -> angles ->
%   coefficients) costs only element-wise operations per iteration.
%   Also ||f(v) - v||^2 = ||S(ac0) - ac0||^2.
%
%   sure, divMc : fixed-probe MC (mean over the P probes), per pixel
%   divFr, sureFr : angles frozen at th0 (exact formula, ignores the
%                   dependence of the angles on v)

kappa = reshape(1./(1+exp(-ab.a)),1,1,[]);
gain  = reshape(exp(ab.b),1,1,[]);
S = @(c) c - sigma.*(kappa./gain).*tanh(gain.*c./sigma);
t0 = tanh(gain.*pre.ac0./sigma);
res = sum((sigma.*(kappa./gain).*t0).^2,'all')/pre.N;
tJ = sum(pre.eAc.*S(pre.acJ),[1 2 3]) + pre.eDcDc ...
   - sum(pre.e0Ac.*S(pre.ac0),[1 2 3]) - pre.e0DcDc;     % 1 x 1 x 1 x P
divMc = mean(tJ,4)/(epsFd*pre.N);
sure = res - sigma^2 + 2*sigma^2*divMc;
divFr = 1 - sum(kappa.*(1 - t0.^2),'all')/pre.N;
sureFr = res - sigma^2 + 2*sigma^2*divFr;
end
