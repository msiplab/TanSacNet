function [sure,fu,divN] = fcn_sssuresalsun(net,X,sigma,divMode,b,epsFd)
%FCN_SSSURESALSUN SURE of the SA-LSUN denoiser f(v) = D(v) S(E(v) v)
%
%   net : nets.shrink of fcn_ssbuildsalsun (Whole, Reuse, Lv1_Shrink)
%   X   : noisy image v, dlarray 'SSCB' (one sample)
%
%   divMode "mc"     : Monte-Carlo divergence of the WHOLE map. The angles
%                      are re-estimated for v + epsFd*b, so the dependence
%                      of the angles on v is included. b holds the probes
%                      (H x W x 1 x P, 'SSCB'); they run as one batch.
%   divMode "frozen" : exact divergence with the angles frozen at their
%                      values for v (E unitary, S = 1):
%                      div f = N - sum_AC kappa_p sech^2(g_p c / sigma).
%
%   Returns per-pixel SURE, f(v) and div f / N.

N = numel(X);
switch divMode
    case "mc"
        fu = forward(net,X);
        fb = forward(net,X + epsFd*b);
        divN = mean(sum(b.*(fb - fu),[1 2 3]),4)/(epsFd*N);
    case "frozen"
        [fu,ac] = forward(net,X,'Outputs',{'Lv1_Out','Lv1_AcOut'});
        L = net.Learnables;
        A = L.Value{strcmp(L.Layer,'Lv1_Shrink') & strcmp(L.Parameter,'A')};
        B = L.Value{strcmp(L.Layer,'Lv1_Shrink') & strcmp(L.Parameter,'B')};
        kappa = reshape(1./(1+exp(-A)),1,1,[]);
        gain  = reshape(exp(B),1,1,[]);
        th = tanh(gain.*ac./sigma);
        divN = 1 - sum(kappa.*(1 - th.^2),'all')/N;
    otherwise
        error('divMode must be "mc" or "frozen"')
end
res  = sum((fu - X).^2,'all')/N;
sure = res - sigma.^2 + 2*sigma.^2.*divN;
end
