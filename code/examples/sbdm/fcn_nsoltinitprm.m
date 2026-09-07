function prm = fcn_nsoltinitprm(info,nStages,kappa0,tau0)
%FCN_NSOLTINITPRM Initial parameters of the NSOLT denoiser
%
%   Same layout as FCN_LSUNINITPRM, except that every angle array is a single
%   column: an NSOLT shares its angles across blocks by construction.
%
% Copyright (c) 2026, Shogo MURAMATSU, All rights reserved.

arguments
    info struct
    nStages (1,1) double = 1
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
        th{k} = dlarray(zeros(info.nAngles(k),1,'single'));
    end
    prm.th{s} = th;
    prm.a{s} = dlarray(single(log(kappa0/(1-kappa0)))*ones(nAc,1,'single'));
    prm.b{s} = dlarray(single(log(kappa0/tau0))*ones(nAc,1,'single'));
end
end
