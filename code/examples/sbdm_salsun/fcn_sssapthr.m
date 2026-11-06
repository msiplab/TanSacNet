function [thr,info] = fcn_sssapthr(v,cfg,qP,variant)
%FCN_SSSAPTHR Step-1 threshold on the pilot p for SAP-hard ("matched")
%
%   [thr,info] = fcn_sssapthr(v,cfg,qP,variant)
%   qP : N-hard result used as the pilot (fcn_ssnrunimage, mode "hard",
%        statistics with thresholds [2 3]*sigma), p = qP(v).
%   variant "a"  : thr such that p keeps the same fraction of AC
%                  coefficients as N-hard keeps on v (per image):
%                  f = fraction of |E_Nh v| > tau_N*sigma (qP.statV),
%                  thr = value between the k-th and (k+1)-th largest
%                  |E_Nh p| (AC), k = round(f*n), E_Nh = N-hard's transform.
%                  The threshold is fixed; the SA transform learned in
%                  step 1 changes, so the final fraction on p is reported.
%   variant "bT" : thr = T*sigma_p, sigma_p = sqrt(SURE of N-hard)
%                  (per-pixel exact SURE, an estimate of the MSE of p).
%   info.f      : target fraction (variant a), info.fAct = k/n
%   info.sigmaP : sqrt(SURE of N-hard)

tauN = [2 3];                          % thresholds of qP.statV (fcn_ssnrunimage, tauList [2 3])
sigma = double(cfg.sigma);
info.sigmaP = sqrt(max(double(qP.sure),eps));
info.f = NaN; info.fAct = NaN;
if variant == "a"
    jN = find(abs(tauN - qP.tau) < 1e-9,1);
    info.f = qP.statV.hardFrac(jN);
    useGpu = isa(v,'gpuArray');
    [anet,snet,info2] = fcn_lsunpair(size(v),cfg.stride,cfg.ovlp);
    prm = qP.prm;
    if useGpu, prm = dlupdate(@gpuArray,prm); end
    X = dlarray(v,'SSCB');
    P = fcn_lsundenoise(anet,snet,info2,prm,X,cfg.sigma);   % p = N-hard(v), as the pilot
    anet = fcn_lsunsetangles(anet,snet,info2,prm.th{1});
    [ac,~] = predict(anet,P);
    a = extractdata(ac);
    s = sort(abs(double(gather(a(:)))),'descend');
    n = numel(s); k = min(max(1,round(info.f*n)),n-1);
    thr = (s(k) + s(k+1))/2;
    info.fAct = k/n;
elseif startsWith(variant,"b")
    tau = str2double(extractAfter(variant,1));
    thr = tau*info.sigmaP;
else
    error("unknown variant %s",variant)
end
info.thrOverSigma = thr/sigma;
end
