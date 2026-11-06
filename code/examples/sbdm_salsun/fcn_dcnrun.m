function n = fcn_dcnrun(x,v,cfg,seed,leak,dcShrink,tau,th0)
%FCN_DCNRUN N-hard (shared angles, observation only) with the DC options
%
%   n = fcn_dcnrun(x,v,cfg,seed,leak,dcShrink,tau,th0)
%   leak     : false = no DC leakage (as before), true = all angles free
%              (fcn_dclsunpair)
%   dcShrink : false = DC passes through (as before), true = step 2 also
%              learns kappa_0, g_0 of the DC channel
%   tau      : step-1 threshold tau*sigma (hard mask, fcn_ssstep1lsunhard,
%              cfg.nIt1N its, lr cfg.lr1N, zero initial angles)
%   th0      : [] -> run step 1;
%              cell of angles -> step 1 is NOT run, these angles are used
%              (the step-1 result of a cached run with the same leak and
%              tau; step 1 does not depend on the shrinkage)
%   Step 2: SURE on v, angles frozen (fcn_dcstep2, cfg.nIt2 its, lr
%   cfg.lr2). With dcShrink, two DC inits are run ("ac": kappa0 = 0.9 as
%   the AC channels, "id": kappa0 = 0.01) and the one with the lower SURE
%   is kept (v only, no clean image). Without dcShrink: one run, AC only
%   (= fcn_ssnrunimage step 2).
%   x is used ONLY for evaluation (PSNR, SSIM, MSE, statistics, PR error).

sigma = cfg.sigma; sz = size(v); N = numel(v);
useGpu = isa(v,'gpuArray');
X = dlarray(v,'SSCB'); Xc = dlarray(x,'SSCB');
m = reshape(cast(fcn_ssacmask(prod(cfg.stride),cfg.K),'like',v),1,1,[]);
thrList = double(cfg.tauList)*double(sigma);
n.leak = logical(leak); n.dcShrink = logical(dcShrink); n.tau = tau; n.sigma = sigma;

rng(seed)
t = tic;
[anet,snet,info] = fcn_dclsunpair(sz,cfg.stride,cfg.ovlp,leak);
n.tBuild = toc(t);

t = tic;
if isempty(th0)
    prm = fcn_lsuninitprm(info,1,false);
    if useGpu, prm = dlupdate(@gpuArray,prm); end
    [prm,n.hist1,n.frac1] = fcn_ssstep1lsunhard(anet,snet,info,prm,v,...
        tau*double(sigma),cfg.nIt1N,cfg.lr1N(1),cfg.lr1N(2));
    th = prm.th{1}; n.step1 = "run";
else
    th = th0;
    if useGpu, th = dlupdate(@gpuArray,th); end
    n.hist1 = []; n.frac1 = []; n.step1 = "given";
end
sync(useGpu); n.t1 = toc(t);

[anetT,snetT] = fcn_lsunsetangles(anet,snet,info,th);
[ac,dc] = predict(anetT,X);
n.statV = fcn_sskeptstats(ac,dc,X,m,thrList);
[acx,dcx] = predict(anetT,Xc);                    % clean: evaluation only
n.statX = fcn_sskeptstats(acx,dcx,Xc,m,thrList);
n.prErrX = relerr(predict(snetT,acx,dcx),Xc);
n.prErrV = relerr(predict(snetT,ac,dc),X);

% --- step 2
t = tic;
A = extractdata(ac); D = extractdata(dc);
if dcShrink, inits = ["ac" "id"]; else, inits = "none"; end
best = inf;
for k = 1:numel(inits)
    ab0 = fcn_dcinitab(info.nChs-1,inits(k),A);
    [ab,hist] = fcn_dcstep2(A,D,sigma,N,ab0,cfg.nIt2,cfg.lr2);
    s = double(gather(extractdata(fcn_dcsure(ab,A,D,sigma,N))));
    fk = gather(extractdata(fcn_dcdenoise(anet,snet,info,th,ab,X,sigma)));
    n.init(k) = struct('name',inits(k),'sure',s,'psnr',psnr(fk,gather(x)),...   % psnr: evaluation only
        'hist2',hist,'ab',dlupdate(@gather,ab));
    if s < best, best = s; abB = ab; n.selInit = inits(k); n.hist2 = hist; end
end
sync(useGpu); n.t2 = toc(t);

% --- evaluation
t = tic;
fu = fcn_dcdenoise(anet,snet,info,th,abB,X,sigma);
[sure,divN,res] = fcn_dcsure(abB,ac,dc,sigma,N);
z = gather(extractdata(fu));
n.psnr = psnr(z,gather(x)); n.ssim = ssim(z,gather(x));
n.mse = double(mean((z - gather(x)).^2,'all')); n.img = single(z);
n.sure = double(gather(extractdata(sure)));
n.divN = double(gather(extractdata(divN)));
n.resCheck = abs(double(mean((z - gather(v)).^2,'all')) - double(gather(extractdata(res))));
n.kappa = gather(extractdata(sigmoid(abB.a)));
n.gain  = gather(extractdata(exp(abB.b)));
if dcShrink
    n.kappa0 = double(gather(extractdata(sigmoid(abB.a0))));
    n.gain0  = double(gather(extractdata(exp(abB.b0))));
else
    n.kappa0 = NaN; n.gain0 = NaN;
end
n.th = dlupdate(@gather,th);
n.ab = dlupdate(@gather,abB);
n.prm.th = {n.th};                                 % same layout as fcn_ssnrunimage
n.prm.a = {n.ab.a}; n.prm.b = {n.ab.b};
sync(useGpu); n.tEval = toc(t);
n.tTotal = n.tBuild + n.t1 + n.t2 + n.tEval;
end

function e = relerr(y,x)
e = double(gather(extractdata(sqrt(sum((y - x).^2,'all')/sum(x.^2,'all')))));
end

function sync(useGpu)
if useGpu, wait(gpuDevice); end
end
