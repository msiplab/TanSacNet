function ab = fcn_dcinitab(nAc,dcInit,proto)
%FCN_DCINITAB Initial shrinkage parameters, optionally with the DC channel
%
%   ab = fcn_dcinitab(nAc,dcInit,proto)
%   AC (always): kappa0 = 0.9, tau0 = 2 (as fcn_lsuninitprm and
%                sureShrinkage2dLayer): ab.a = logit(0.9), ab.b = log(0.9/2)
%   DC (ab.a0, ab.b0), dcInit:
%     "none" : no DC shrinkage (no fields a0, b0)
%     "ac"   : the same init as AC (kappa0 = 0.9, tau0 = 2)
%     "id"   : kappa0 = 0.01, tau0 = 2 (almost the identity, i.e. almost
%              no DC shrinkage at the start)
%   proto : numeric array; the parameters get its class and device.
%   All fields are dlarrays.

ab.a = dlarray(cast(log(0.9/0.1)*ones(nAc,1),'like',proto));
ab.b = dlarray(cast(log(0.9/2.0)*ones(nAc,1),'like',proto));
switch dcInit
    case "none"
    case "ac"
        ab.a0 = dlarray(cast(log(0.9/0.1),'like',proto));
        ab.b0 = dlarray(cast(log(0.9/2.0),'like',proto));
    case "id"
        k0 = 0.01; t0 = 2;
        ab.a0 = dlarray(cast(log(k0/(1-k0)),'like',proto));
        ab.b0 = dlarray(cast(log(k0/t0),'like',proto));
    otherwise
        error("unknown dcInit %s",dcInit)
end
end
