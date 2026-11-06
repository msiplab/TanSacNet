function [sure,divMc,divFr,sureFr] = fcn_dcsurepilot(ab,pre,sigma,epsFd)
%FCN_DCSUREPILOT Fixed-probe SURE of the pilot SA-LSUN denoiser, optional DC shrinkage
%
%   fcn_sspilotsure with the DC channel. f(v) = D_th S(E_th v),
%   th = angles estimated from p = Pilot(v), D_th = E_th'. Precomputed
%   (fcn_dcsaprun): ac0/dc0 = E_th0 v; acJ/dcJ = E_thj (v + eps b_j);
%   eAc/eDc = E_thj b_j; e0Ac/e0Dc = E_th0 b_j.
%   Since D is unitary,
%     b_j'(f(v + eps b_j) - f(v)) = eJ' S(cJ) - e0J' S(c0)
%   on AC and DC (S = identity on DC without DC shrinkage), and
%     ||f(v) - v||^2 = ||S(c0) - c0||^2  (AC + DC).
%   sure, divMc   : fixed-probe MC (mean over the P probes), per pixel
%   divFr, sureFr : angles frozen at th0 (exact formula, fcn_dcsure)
%   Without ab.a0 this equals fcn_sspilotsure.

N = pre.N;
kappa = reshape(1./(1+exp(-ab.a)),1,1,[]);
gain  = reshape(exp(ab.b),1,1,[]);
S = @(c) c - sigma.*(kappa./gain).*tanh(gain.*c./sigma);
t0 = tanh(gain.*pre.ac0./sigma);
res = sum((sigma.*(kappa./gain).*t0).^2,'all');
sc  = sum(kappa.*(1 - t0.^2),'all');
if isfield(ab,'a0')
    k0 = 1./(1+exp(-ab.a0)); g0 = exp(ab.b0);
    S0 = @(c) c - sigma.*(k0./g0).*tanh(g0.*c./sigma);
    d0 = tanh(g0.*pre.dc0./sigma);
    res = res + sum((sigma.*(k0./g0).*d0).^2,'all');
    sc  = sc + sum(k0.*(1 - d0.^2),'all');
    tJ = sum(pre.eAc.*S(pre.acJ),[1 2 3]) + sum(pre.eDc.*S0(pre.dcJ),[1 2 3]) ...
       - sum(pre.e0Ac.*S(pre.ac0),[1 2 3]) - sum(pre.e0Dc.*S0(pre.dc0),[1 2 3]);
else
    tJ = sum(pre.eAc.*S(pre.acJ),[1 2 3]) + pre.eDcDc ...
       - sum(pre.e0Ac.*S(pre.ac0),[1 2 3]) - pre.e0DcDc;     % as fcn_sspilotsure
end
res = res/N;
divMc = mean(tJ,4)/(epsFd*N);
sure = res - sigma^2 + 2*sigma^2*divMc;
divFr = 1 - sc/N;
sureFr = res - sigma^2 + 2*sigma^2*divFr;
end
