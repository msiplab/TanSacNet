function [r,ctx] = fcn_dcsaprun(x,v,cfg,seed,Pf,pCheck,leak,dcShrink,thr,nIt1)
%FCN_DCSAPRUN SAP-hard (SA-LSUN, pilot p = Pf(v)) with the DC options
%
%   [r,ctx] = fcn_dcsaprun(x,v,cfg,seed,Pf,pCheck,leak,dcShrink,thr,nIt1)
%
%   A copy of fcn_ssphardrunimage (mode "hard") with three changes:
%     - the pilot is a given function Pf (the N-hard of the same variant,
%       recomputed for every SURE probe); pCheck = its cached output
%       (check only)
%     - leak = true: SA-LSUN without the no-DC-leakage constraint (the
%       estimator of V0 also predicts the first nChs/2-1 angles and sees
%       the DC channel)
%     - dcShrink = true: step 2 also learns kappa_0, g_0 of the DC
%       channel; two DC inits ("ac", "id"), the one with the lower
%       fixed-probe SURE is kept (v only)
%   thr  : step-1 threshold on p (absolute), mask |c| > thr, DC kept
%   nIt1 : step-1 iterations (lr cfg.lr1Sa); b = cfg.bottleneck
%   Angles from p frozen; step 2 by SURE on v with cfg.nProbePilot fixed
%   probes (cfg.nIt2 its, lr cfg.lr2). Checks: explicit chain with the
%   same probes, and cfg.nProbeFresh fresh probes.
%   With leak = false, dcShrink = false and the same pilot, this is
%   fcn_ssphardrunimage (same seed, same initial weights).
%   x is used ONLY for evaluation.

sigma = cfg.sigma; sz = size(v); N = numel(v);
useGpu = isa(v,'gpuArray');
X = dlarray(v,'SSCB'); Xc = dlarray(x,'SSCB');
nAc = prod(cfg.stride) - 1;
m = reshape(cast(fcn_ssacmask(prod(cfg.stride),cfg.K),'like',v),1,1,[]);
r.leak = logical(leak); r.dcShrink = logical(dcShrink); r.nIt1 = nIt1;
r.thr = thr; r.thrOverSigma = thr/double(sigma); r.bottleneck = cfg.bottleneck;

% --- pilot
t = tic;
P = Pf(X); p = extractdata(P);
r.pDiff = double(max(abs(gather(p) - pCheck),[],'all'));
r.psnrPilot = psnr(gather(p),gather(x));
r.ssimPilot = ssim(gather(p),gather(x));

% --- step 1 on p (same seed and build order as fcn_ssphardrunimage)
rng(seed)
if useGpu, gpurng(seed); end
o = struct('estimator',"net",'leak',leak,'bottleneck',cfg.bottleneck,'shrink',false);
nets = fcn_dcbuildsalsun(sz,cfg.stride,cfg.ovlp,o);
if useGpu
    nets.train = dlupdate(@gpuArray,nets.train);
    nets.whole = dlupdate(@gpuArray,nets.whole);
end
r.tBuild = toc(t);
t = tic;
[tnet,r.hist1,r.frac1] = fcn_ssstep1salsunhard(nets.train,p,thr,nIt1,cfg.lr1Sa(1),cfg.lr1Sa(2));
sync(useGpu); r.t1 = toc(t);
r.nEstW = sum(cellfun(@numel,tnet.Learnables.Value));

% --- angles from p, frozen
t = tic;
oc = struct('estimator',"const",'leak',leak,'shrink',true,'dcShrink',dcShrink,...
    'dcInit',"id",'sigma',sigma);
c = fcn_dcbuildsalsun(sz,cfg.stride,cfg.ovlp,oc);
if useGpu
    c.ana = dlupdate(@gpuArray,c.ana); c.whole = dlupdate(@gpuArray,c.whole);
    c.shrink = dlupdate(@gpuArray,c.shrink);
end
[th0,names] = fcn_ssgetthetas(tnet,P);
cana0 = fcn_sssetconstangles(c.ana,names,th0);
cwhole0 = fcn_sssetconstangles(c.whole,names,th0);

% check: constant-angle net on p == estimator net on p
wnet = fcn_copylearnablesbyname(nets.whole,tnet);
acE = predict(wnet,P,'Outputs','Lv1_AcOut');
acC = predict(cana0,P,'Outputs','Lv1_AcOut');
r.constCheck = double(gather(extractdata(max(abs(acE - acC),[],'all'))));

r.statV = stats(cana0,X,m,thr);
r.statP = stats(cana0,P,m,thr);
r.statX = stats(cana0,Xc,m,thr);            % evaluation only
r.prErrX = relerr(predict(cwhole0,Xc),Xc);
r.prErrV = relerr(predict(cwhole0,X),X);

% --- fixed probes: whole chain computed once per probe
rng(seed + 1000,'twister');
B = randn([sz 1 cfg.nProbePilot],'single');
if useGpu, B = gpuArray(B); end
dB = dlarray(B,'SSCB'); epsFd = cfg.epsFd;
[ac0,dc0] = predict(cana0,X,'Outputs',{'Lv1_AcOut','Lv1_DcOut'});
[e0Ac,e0Dc] = predict(cana0,dB,'Outputs',{'Lv1_AcOut','Lv1_DcOut'});
nP = cfg.nProbePilot;
acJ = zeros([size(ac0,1:2) nAc nP],'like',extractdata(ac0)); dcJ = zeros([size(dc0,1:2) 1 nP],'like',acJ);
eAc = acJ; eDc = dcJ;
for j = 1:nP
    Vj = X + epsFd*dB(:,:,:,j);
    thj = fcn_ssgetthetas(tnet,Pf(Vj),names);
    canaj = fcn_sssetconstangles(c.ana,names,thj);
    [a1,d1] = predict(canaj,Vj,'Outputs',{'Lv1_AcOut','Lv1_DcOut'});
    [a2,d2] = predict(canaj,dB(:,:,:,j),'Outputs',{'Lv1_AcOut','Lv1_DcOut'});
    acJ(:,:,:,j) = extractdata(a1); dcJ(:,:,:,j) = extractdata(d1);
    eAc(:,:,:,j) = extractdata(a2); eDc(:,:,:,j) = extractdata(d2);
end
pre.N = N;
pre.ac0 = double(extractdata(ac0)); pre.dc0 = double(extractdata(dc0));
pre.acJ = double(acJ); pre.eAc = double(eAc);
pre.dcJ = double(dcJ); pre.eDc = double(eDc);
pre.e0Ac = double(extractdata(e0Ac)); pre.e0Dc = double(extractdata(e0Dc));
pre.eDcDc  = sum(pre.eDc.*pre.dcJ,[1 2 3]);
pre.e0DcDc = sum(pre.e0Dc.*pre.dc0,[1 2 3]);
sync(useGpu); r.tPrep = toc(t);

% --- step 2 (same init, iterations and lr as before; DC inits if dcShrink)
t = tic;
sd = double(sigma);
if dcShrink, inits = ["ac" "id"]; else, inits = "none"; end
cshrink0 = fcn_sssetconstangles(c.shrink,names,th0);
best = inf;
for k = 1:numel(inits)
    ab = fcn_dcinitab(nAc,inits(k),pre.ac0);
    avgG = []; avgSqG = []; hist = zeros(cfg.nIt2,1);
    for it = 1:cfg.nIt2
        [loss,grads] = dlfeval(@lossfun,ab,pre,sd,epsFd);
        [ab,avgG,avgSqG] = adamupdate(ab,grads,avgG,avgSqG,it,cfg.lr2);
        hist(it) = gather(extractdata(loss));
    end
    s = double(gather(extractdata(fcn_dcsurepilot(ab,pre,sd,epsFd))));
    fk = gather(extractdata(predict(fcn_dcsetshrink(cshrink0,ab),X)));
    r.init(k) = struct('name',inits(k),'sure',s,'psnr',psnr(fk,gather(x)),...   % psnr: evaluation only
        'hist2',hist,'ab',dlupdate(@gather,ab));
    if s < best, best = s; abB = ab; r.selInit = inits(k); r.hist2 = hist; end
end
sync(useGpu); r.t2 = toc(t);

% --- evaluation
t = tic;
[s,dM,dF,sF] = fcn_dcsurepilot(abB,pre,sd,epsFd);
r.sure = gather(extractdata(s)); r.divMc = gather(extractdata(dM));
r.sureFrozen = gather(extractdata(sF)); r.divFrozen = gather(extractdata(dF));
cshrinkB = fcn_dcsetshrink(cshrink0,abB);
fu = predict(cshrinkB,X);
z = gather(extractdata(fu)); xg = gather(x);
r.psnr = psnr(z,xg); r.ssim = ssim(z,xg);
r.mse = double(mean((z - xg).^2,'all')); r.img = single(z);
r.kappa = gather(extractdata(1./(1+exp(-abB.a))));
r.gain  = gather(extractdata(exp(abB.b)));
if dcShrink
    r.kappa0 = double(gather(extractdata(1./(1+exp(-abB.a0)))));
    r.gain0  = double(gather(extractdata(exp(abB.b0))));
else
    r.kappa0 = NaN; r.gain0 = NaN;
end
r.ab = dlupdate(@gather,abB);

% checks: explicit chain with the same probes, and with fresh probes
ctx = struct('Pf',Pf,'tnet',tnet,'names',{names},'cshrink',c.shrink,...
    'X',X,'epsFd',epsFd,'sigma',sd);
divJ = fcn_dcsapexplicit(ctx,abB,B);
res = sum((extractdata(fu) - v).^2,'all')/N;
r.sureExplicitSame = double(gather(res)) - sd^2 + 2*sd^2*mean(divJ);
rng(seed + 2000,'twister');
Bf = randn([sz 1 cfg.nProbeFresh],'single');
if useGpu, Bf = gpuArray(Bf); end
divF = fcn_dcsapexplicit(ctx,abB,Bf);
sureF = double(gather(res)) - sd^2 + 2*sd^2*divF;
r.sureFresh = mean(sureF); r.sureFreshSe = std(sureF)/sqrt(numel(sureF));
sync(useGpu); r.tEval = toc(t);
r.tTotal = toc(t) + r.tBuild + r.t1 + r.tPrep + r.t2;
end

%%
function [loss,grads] = lossfun(ab,pre,sigma,epsFd)
loss = fcn_dcsurepilot(ab,pre,sigma,epsFd);
grads = dlgradient(loss,ab);
end

function k = stats(cana,Z,m,thrList)
[ac,dc] = predict(cana,Z,'Outputs',{'Lv1_AcOut','Lv1_DcOut'});
k = fcn_sskeptstats(ac,dc,Z,m,thrList);
end

function e = relerr(y,x)
e = double(gather(extractdata(sqrt(sum((y - x).^2,'all')/sum(x.^2,'all')))));
end

function sync(useGpu)
if useGpu, wait(gpuDevice); end
end
