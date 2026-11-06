function [r,ctx] = fcn_sspsrcrunimage(x,v,K,cfg,seed,qN,qW,bottleneck,nIt1,pilot)
%FCN_SSPSRCRUNIMAGE SA-LSUN pilot arm with a chosen pilot source
%
%   [r,ctx] = fcn_sspsrcrunimage(x,v,K,cfg,seed,qN,qW,bottleneck,nIt1,pilot)
%
%   Same as fcn_sspsizerunimage (step 1 on the pilot p, angles from p
%   frozen, step 2 by SURE on v with cfg.nProbePilot fixed probes), with
%   one change: the pilot p = P(v) is chosen by
%     pilot = "N"  : p = N(v), cached uniform LSUN (as fcn_sspsizerunimage)
%             "W"  : p = W(v), wiener2 5x5, known noise variance (as arm W)
%             "NW" : p = (N(v) + W(v))/2
%   P is a fixed function of its input. It is applied again to v + eps*b
%   for every SURE probe, so the divergence includes the dependence of
%   the angles on v through P.
%   qN : cached N result (parameters qN.prm, output qN.img for a check)
%   qW : cached W result (output qW.img for a check)
%   pilot = "N" is exactly fcn_sspsizerunimage.
%   x is used ONLY for evaluation.

sigma = cfg.sigma; sz = size(v); N = numel(v);
useGpu = isa(v,'gpuArray');
X = dlarray(v,'SSCB'); Xc = dlarray(x,'SSCB');
acMask = fcn_ssacmask(prod(cfg.stride),K);
m = reshape(cast(acMask,'like',v),1,1,[]);
r.K = K; r.bottleneck = bottleneck; r.nIt1 = nIt1;

% --- N as a fixed function (cached parameters of the same K)
t = tic;
[anetN,snetN,infoN] = fcn_lsunpair(sz,cfg.stride,cfg.ovlp);
prmN = qN.prm;
if useGpu, prmN = dlupdate(@gpuArray,prmN); end
Nf = @(Z) fcn_lsundenoise(anetN,snetN,infoN,prmN,Z,sigma);
Wf = @(Z) wienerdl(Z,sigma);
switch pilot
    case "N",  Pf = Nf;
    case "W",  Pf = Wf;
    case "NW", Pf = @(Z) (Nf(Z) + Wf(Z))/2;
    otherwise, error("unknown pilot %s",pilot)
end
r.pilot = pilot;
r.pDiffCacheN = double(max(abs(gather(extractdata(Nf(X))) - qN.img),[],'all'));  % vs cached N output
r.pDiffCacheW = double(max(abs(gather(extractdata(Wf(X))) - qW.img),[],'all'));  % vs cached W output
P = Pf(X); p = extractdata(P);
r.psnrPilot = psnr(gather(p),gather(x));
r.ssimPilot = ssim(gather(p),gather(x));

% --- Step 1 on p (same seed/initial weights as the old SA arm)
rng(seed)
if useGpu, gpurng(seed); end
nets = fcn_ssbuildsalsunsz(sz,cfg.stride,cfg.ovlp,acMask,sigma,bottleneck);
if useGpu
    nets.train = dlupdate(@gpuArray,nets.train);
    nets.whole = dlupdate(@gpuArray,nets.whole);
end
r.tBuild = toc(t);
t = tic;
[tnet,r.hist1] = fcn_ssstep1salsun(nets.train,p,nIt1,cfg.lr1Sa(1),cfg.lr1Sa(2));
sync(useGpu); r.t1 = toc(t);
r.nEstW = sum(cellfun(@numel,tnet.Learnables.Value));

% --- Angles from p, frozen
t = tic;
c = fcn_ssbuildconstsalsun(sz,cfg.stride,cfg.ovlp,sigma);
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

% kept AC energy with the angles used for denoising (from p)
[r.keptAcV,r.keptAllV] = kept(cana0,X,m);
[r.keptAcP,r.keptAllP] = kept(cana0,P,m);
[r.keptAcX,r.keptAllX] = kept(cana0,Xc,m);     % evaluation only

% perfect reconstruction with the frozen angles
r.prErrX = relerr(predict(cwhole0,Xc),Xc);
r.prErrV = relerr(predict(cwhole0,X),X);

% --- Fixed probes: whole chain computed once per probe
rng(seed + 1000,'twister');
B = randn([sz 1 cfg.nProbePilot],'single');
if useGpu, B = gpuArray(B); end
dB = dlarray(B,'SSCB'); epsFd = cfg.epsFd;
[ac0,dc0] = predict(cana0,X,'Outputs',{'Lv1_AcOut','Lv1_DcOut'});
[e0Ac,e0Dc] = predict(cana0,dB,'Outputs',{'Lv1_AcOut','Lv1_DcOut'});
nP = cfg.nProbePilot; nAc = numel(acMask);
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
pre.ac0 = double(extractdata(ac0));
pre.acJ = double(acJ); pre.eAc = double(eAc);
pre.e0Ac = double(extractdata(e0Ac));
pre.eDcDc  = sum(double(eDc).*double(dcJ),[1 2 3]);
pre.e0DcDc = sum(double(extractdata(e0Dc)).*double(extractdata(dc0)),[1 2 3]);
sync(useGpu); r.tPrep = toc(t);

% --- Step 2: only kappa_p, g_p (same init, iterations and lr as N / old SA)
t = tic;
ab.a = dlarray(log(0.9/0.1)*ones(nAc,1,'like',pre.ac0));
ab.b = dlarray(log(0.9/2.0)*ones(nAc,1,'like',pre.ac0));
sd = double(sigma);
avgG = []; avgSqG = []; r.hist2 = zeros(cfg.nIt2,1);
for it = 1:cfg.nIt2
    [loss,grads] = dlfeval(@lossfun,ab,pre,sd,epsFd);
    [ab,avgG,avgSqG] = adamupdate(ab,grads,avgG,avgSqG,it,cfg.lr2);
    r.hist2(it) = gather(extractdata(loss));
end
sync(useGpu); r.t2 = toc(t);

% --- Evaluation
t = tic;
[s,dM,dF,sF] = fcn_sspilotsure(ab,pre,sd,epsFd);
r.sure = gather(extractdata(s)); r.divMc = gather(extractdata(dM));
r.sureFrozen = gather(extractdata(sF)); r.divFrozen = gather(extractdata(dF));
cshrink0 = fcn_sssetconstangles(c.shrink,names,th0,ab.a,ab.b);
fu = predict(cshrink0,X);
r = mergestruct(r,metrics(gather(extractdata(fu)),gather(x)));
r.kappa = gather(extractdata(1./(1+exp(-ab.a))));
r.gain  = gather(extractdata(exp(ab.b)));

% checks on the final parameters: explicit chain with the same probes,
% and with fresh probes (per-iteration-style MC)
ctx = struct('Nf',Pf,... field name used by fcn_sspilotexplicit
    'tnet',tnet,'names',{names},'cshrink',c.shrink,...
    'cana',c.ana,'X',X,'epsFd',epsFd,'pre',pre,'B',B,'th0',{th0},'sigma',sd,'ab',ab);
divJ = fcn_sspilotexplicit(ctx,ab,B);
res = sum((extractdata(fu) - v).^2,'all')/N;
r.sureExplicitSame = double(gather(res)) - sd^2 + 2*sd^2*mean(divJ);
rng(seed + 2000,'twister');
Bf = randn([sz 1 cfg.nProbeFresh],'single');
if useGpu, Bf = gpuArray(Bf); end
divF = fcn_sspilotexplicit(ctx,ab,Bf);
sureF = double(gather(res)) - sd^2 + 2*sd^2*divF;
r.sureFresh = mean(sureF); r.sureFreshSe = std(sureF)/sqrt(numel(sureF));
sync(useGpu); r.tEval = toc(t);
end

%%
function [loss,grads] = lossfun(ab,pre,sigma,epsFd)
loss = fcn_sspilotsure(ab,pre,sigma,epsFd);
grads = dlgradient(loss,ab);
end

function [kAc,kAll] = kept(cana,Z,m)
[ac,dc] = predict(cana,Z,'Outputs',{'Lv1_AcOut','Lv1_DcOut'});
e0 = sum(Z.^2,'all'); eDc = sum(dc.^2,'all'); eAc = sum((m.*ac).^2,'all');
kAc  = double(gather(extractdata(eAc/(e0 - eDc))));
kAll = double(gather(extractdata((eDc + eAc)/e0)));
end

function q = metrics(z,x)
q.psnr = psnr(z,x); q.ssim = ssim(z,x);
q.mse = double(mean((z - x).^2,'all')); q.img = single(z);
end

function e = relerr(y,x)
e = double(gather(extractdata(sqrt(sum((y - x).^2,'all')/sum(x.^2,'all')))));
end

function a = mergestruct(a,b)
for f = fieldnames(b).', a.(f{1}) = b.(f{1}); end
end

function sync(useGpu)
if useGpu, wait(gpuDevice); end
end

function Y = wienerdl(Z,sigma)
% wiener2 5x5 with known noise variance, as arm W (double on the CPU),
% returned as a dlarray of the same type and device as Z
z = extractdata(Z);
w = wiener2(gather(double(z)),[5 5],double(sigma)^2);
Y = dlarray(cast(w,'like',z),'SSCB');
end
