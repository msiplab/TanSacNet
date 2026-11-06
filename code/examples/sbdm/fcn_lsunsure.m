function [sure,fu,divN] = fcn_lsunsure(anet,snet,info,prm,u,sigma,divMode,epsFd)
%FCN_LSUNSURE Stein's unbiased risk estimate of the LSUN denoiser (no clean image)
%
%   [sure,fu,divN] = fcn_lsunsure(anet,snet,info,prm,u,sigma) evaluates
%
%     SURE(u) = ||f(u)-u||^2/N - sigma^2 + 2*sigma^2*div f(u)/N
%
%   for each batch element of the noisy dlarray U ('SSCB'), whose noise level
%   is SIGMA (scalar or [1 1 1 B]). For u = x + sigma*zeta with Gaussian zeta,
%   E[SURE] = E[||f(u)-x||^2]/N, so minimising SURE trains the MMSE denoiser
%   from the noisy image alone: the clean image x is never used.
%
%   DIVMODE "auto" (default): exact divergence for one stage, Monte-Carlo
%   otherwise. "exact" / "mc" force one or the other ("exact" needs S = 1).
%
%   Exact divergence for one stage. With c = E*u and E unitary (E*D = I),
%     J_f = D*diag(d)*E,  d = 1 (DC),  d = 1 - kappa_p*sech^2(g_p*c/sigma) (AC),
%   so trace(J_f) = sum(d) = N - sum_AC kappa_p*sech^2(g_p*c/sigma).
%   This holds for uniform and for locally-structured (per-block) angles
%   alike, because only unitarity is used. It does not hold for a
%   non-tight dictionary such as the convolutional TNRD baseline.
%
%   Monte-Carlo divergence (S > 1): div f ~ b'*(f(u+eps*b)-f(u))/eps,
%   b ~ N(0,I) (Ramani, Blu and Unser, IEEE TIP 2008).
%
%   DIVN returns div f / N, size [1 1 1 B].

arguments
    anet, snet, info struct, prm struct
    u
    sigma
    divMode (1,1) string = "auto"
    epsFd (1,1) double = 1e-3
end

nStages = numel(prm.th);
N = prod(info.inputSize);       % pixels = coefficients (critically sampled)
if divMode == "auto"
    if nStages == 1, divMode = "exact"; else, divMode = "mc"; end
end

switch divMode
    case "exact"
        assert(nStages == 1,"Exact divergence is implemented for S = 1 only.")
        nAc = info.nChs - 1;
        [anet,snet] = fcn_lsunsetangles(anet,snet,info,prm.th{1});
        [ac,dc] = forward(anet,u);
        kappa = reshape(sigmoid(prm.a{1}),1,1,nAc);
        gain  = reshape(exp(prm.b{1}),1,1,nAc);
        th    = tanh(gain.*ac./sigma);
        fu    = forward(snet,ac - sigma.*(kappa./gain).*th,dc);
        % sech^2 = 1 - tanh^2; divide by N inside the sum for precision
        divN  = 1 - sum(kappa.*(1 - th.^2),[1 2 3])/N;
    case "mc"
        fu = fcn_lsundenoise(anet,snet,info,prm,u,sigma);
        b  = randn(size(u),'like',extractdata(u));
        fb = fcn_lsundenoise(anet,snet,info,prm,u + epsFd*b,sigma);
        divN = sum(b.*(fb - fu),[1 2 3])/(epsFd*N);
    otherwise
        error("divMode must be ""auto"", ""exact"" or ""mc""")
end

res  = sum((fu - u).^2,[1 2 3])/N;
sure = res - sigma.^2 + 2*sigma.^2.*divN;     % [1 1 1 B]
end
