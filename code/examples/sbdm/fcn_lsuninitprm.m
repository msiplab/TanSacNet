function prm = fcn_lsuninitprm(info,nStages,isLocal,kappa0,tau0)
%FCN_LSUNINITPRM Initial parameters of the LSUN denoiser
%
%   prm = fcn_lsuninitprm(info,nStages,isLocal,kappa0,tau0) initialises the
%   learnable parameters of the noise-level conditional LSUN denoiser:
%
%     prm.th{s}{k}  rotation angles of stage s, rotation layer k
%                   [nAngles(k) x 1]        if isLocal is false (uniform)
%                   [nAngles(k) x nBlocks]  if isLocal is true  (local)
%     prm.a{s}      logit of the contraction kappa_p in (0,1)
%     prm.b{s}      log of the gain g_p > 0
%
%   The influence function of channel p is
%
%     phi_p(z) = (kappa_p/g_p)*tanh(g_p*z),  kappa_p = sigmoid(a_p),
%                                            g_p     = exp(b_p),
%
%   so that |phi_p'| <= kappa_p < 1 holds for every parameter value. See
%   FCN_LSUNDENOISE for how phi_p enters the denoiser.
%
%   KAPPA0 is the initial contraction and TAU0 the initial normalised soft
%   threshold kappa_p/g_p (i.e. the threshold divided by the noise level).
%
% Copyright (c) 2026, Shogo MURAMATSU, All rights reserved.

arguments
    info struct
    nStages (1,1) double = 1
    isLocal (1,1) logical = false
    kappa0 (1,1) double = 0.9
    tau0 (1,1) double = 2.0
end

nAc = info.nChs - 1;
prm.th = cell(1,nStages);
prm.a  = cell(1,nStages);
prm.b  = cell(1,nStages);
for s = 1:nStages
    th = cell(info.nRotations,1);
    for k = 1:info.nRotations
        if isLocal
            th{k} = dlarray(zeros(info.nAngles(k),info.nBlocks,'single'));
        else
            th{k} = dlarray(zeros(info.nAngles(k),1,'single'));
        end
    end
    prm.th{s} = th;
    prm.a{s} = dlarray(single(log(kappa0/(1-kappa0)))*ones(nAc,1,'single'));
    prm.b{s} = dlarray(single(log(kappa0/tau0))*ones(nAc,1,'single'));
end
end
