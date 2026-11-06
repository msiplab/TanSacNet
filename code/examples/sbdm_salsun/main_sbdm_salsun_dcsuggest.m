%% main_sbdm_salsun_dcsuggest.m
% Sensei's suggestions (2026-11-06), one MATLAB process, two stages.
% Kodak kodim09/15/19/23, 256x256 centre crops, grayscale, sigma = 25/255
% and 10/255, same noise seeds as before (rng(image index)).
%
% Four variants of the transform / shrinkage:
%   R  = reference: no DC shrinkage, no DC leakage (as before)
%   D  = DC shrinkage: the DC channel is also shrunk (kappa_0, g_0 learned)
%   L  = DC leakage allowed: LSUN / SA-LSUN without the no-DC-leakage
%        constraint (all angles free)
%   LD = both
%
% Stage A: two-step design, observation only.
%   N-hard (shared angles): step 1 hard mask |c| > tau*sigma on v (500 its),
%     step 2 exact SURE on v. tau in {2,3} selected on kodim09 per variant.
%     R : caches.  D : cached R step-1 angles, new step 2 with DC.
%     L : new step 1 (leakage), step 2 AC only.  LD : L angles, step 2 with DC.
%   SAP-hard: pilot p = N-hard of the same variant (selected tau), b = 18,
%     step-1 threshold T*sigma_p (sigma_p = sqrt(SURE of the pilot); T = 2 at
%     sigma 25 (b2), T = 3 at sigma 10 (b3), as selected before), 150 its,
%     angles from p frozen, step 2 by SURE on v with 8 fixed probes
%     (check: 32 fresh probes). R : caches.
%   With DC shrinkage, step 2 runs from two DC inits (kappa0 0.9 = as AC,
%   kappa0 0.01 = almost none); the lower SURE is kept (v only).
% Stage B: fully supervised LSUN-AE, clean targets. 2048 patches 64x64 of
%   kodim01-08 (as examples/sbdm/main_sbdm.m: rng(1), 256 per image),
%   batch 32, fresh noise each iteration, fixed sigma, MSE, Adam with
%   cosine decay. N: shared angles + shrinkage (lr 5e-3 -> 5e-5).
%   SA: estimator (b = 18, sees v) + shrinkage (lr 1e-3 -> 1e-5 and
%   5e-3 -> 5e-5). 8 settings = {N, SA} x {R, D, L, LD}, per sigma.
%   Evaluated on the same crops and noise as stage A.
%
% Fair selection on kodim09 only; report on kodim15/19/23.
% Time budget: before each unit its time is estimated (latest measured
% unit of the same kind, priors from the probe); a unit that would end
% after the deadline is not started (SKIPPED in the log and the summary).
% One cache file per unit in data/sbdm_salsun_dc_*.mat (resumable).
% Supervised trainings also write a checkpoint every cfg0.sup.ckptEvery its.
% Summaries: logs/dcsuggest_stageA_summary.txt, logs/dcsuggest_stageB_summary.txt
%   nohup matlab -licmode onlinelicensing -batch "main_sbdm_salsun_dcsuggest" \
%       > ~/Documents/SBDMwLSUN/logs/dcsuggest.log 2>&1 &
% Quick test of all code paths (3 its, 2 images, separate folder):
%   matlab -licmode onlinelicensing -batch "onTest = true; onTestDir = '<dir>'; main_sbdm_salsun_dcsuggest"
% Options (set before the call): onDeadline = datetime(...), onSupIters = n.

here = fileparts(mfilename('fullpath'));
cd(fullfile(here,'..','..')); setpath; cd(here)
addpath(fullfile(here,'..','sbdm'))     % helpers (called, not edited)
addpath(fullfile(here,'..','salsun'))   % maskLayer
warning('off','MATLAB:rmpath:DirNotFound')
rmpath(fullfile('/home/shogo/Workspace/GitHub/SaivDr','mexcodes'))
warning('on','MATLAB:rmpath:DirNotFound')

cfg0 = fcn_ssconfig();
cfg0.K = 4;
cfg0.nProbePilot = 8;
cfg0.nProbeFresh = 32;
cfg0.bottleneck = 18;
cfg0.selImg = 9;
cfg0.tauList = [2 3];        % N-hard taus (and statistics thresholds)
cfg0.nItSa = 150;            % SAP-hard step-1 iterations
cfg0.sigmas = [25 10];       % x/255
cfg0.sapT = [2 3];           % SAP-hard threshold T*sigma_p per sigma (b2 at 25, b3 at 10)
cfg0.sup = struct('nIters',1500,'mb',32,'lrN',[5e-3 5e-5],'lrEst',[1e-3 1e-5],...
    'lrShr',[5e-3 5e-5],'seed',1,'patch',[64 64],'nCrops',256,'trainIdx',1:8,'ckptEvery',200);
dat = fullfile(here,'..','..','..','data');
ctx.dat = dat; ctx.cdir = dat;
ctx.sdir = fullfile(here,'..','..','..','..','logs');
ctx.deadline = datetime(2026,11,6,23,0,0);
ctx.margin = 1.05;
isTest = exist('onTest','var') && onTest;
if isTest
    cfg0.imgIdx = [9 15]; cfg0.nIt1N = 3; cfg0.nIt2 = 3; cfg0.nItSa = 3; cfg0.nProbeFresh = 2;
    cfg0.sup.nIters = 3; cfg0.sup.ckptEvery = 2;
    ctx.cdir = onTestDir; ctx.sdir = onTestDir;
    if ~isfolder(onTestDir), mkdir(onTestDir); end
end
if exist('onSupIters','var'), cfg0.sup.nIters = onSupIters; end
if exist('onDeadline','var'), ctx.deadline = onDeadline; end
ctx.useGpu = canUseGPU;
ctx.cfg0 = cfg0;
ctx.ST = containers.Map();
ctx.ST('skips') = strings(0,1);
ctx.V = struct('key',{"R","D","L","LD"},'leak',{false,false,true,true},'dcs',{false,true,false,true});
fprintf("%s  start. GPU: %d. Deadline %s. Test mode: %d\n",string(datetime('now')),ctx.useGpu,string(ctx.deadline),isTest)
disp(cfg0); disp(cfg0.sup)

%% ===== Stage A =====
nSg = numel(cfg0.sigmas);
TA = cell(1,nSg);
for k = 1:nSg
    TA{k} = saferun(@() runA(k,ctx),"A s" + cfg0.sigmas(k),ctx);
end
safesummary(@() summaryA(fullfile(ctx.sdir,'dcsuggest_stageA_summary.txt'),TA,ctx));
save(fullfile(ctx.cdir,'sbdm_salsun_dc_stageA_results.mat'),'TA','cfg0')

%% ===== Stage B =====
TB = cell(1,nSg);
try
    rng(1); if ctx.useGpu, gpurng(1); end            % as main_sbdm.m (fcn_seed(1))
    [~,trainSet] = fcn_loadkodak(dat,cfg0.sup.trainIdx,cfg0.sup.patch,cfg0.sup.nCrops,ctx.useGpu);
    fprintf("\n%s  training patches: %d of size %s\n",string(datetime('now')),numel(trainSet),mat2str(cfg0.sup.patch))
    for k = 1:nSg
        TB{k} = saferun(@() runB(k,ctx,trainSet),"B s" + cfg0.sigmas(k),ctx);
    end
catch ME
    addskip(ctx,sprintf("FAILED B (training data): %s",ME.message))
end
safesummary(@() summaryB(fullfile(ctx.sdir,'dcsuggest_stageB_summary.txt'),TB,TA,ctx));
save(fullfile(ctx.cdir,'sbdm_salsun_dc_stageB_results.mat'),'TB','cfg0')

sk = ctx.ST('skips');
fprintf("\nSkipped / failed / missing units (%d):\n",numel(sk))
fprintf("  %s\n",sk)
fprintf("\n%s  done.\n",string(datetime('now')))

%% ===================== Local functions =====================
function T = saferun(fn,tag,ctx)
try
    T = fn();
catch ME
    addskip(ctx,sprintf("FAILED %s (whole sub-stage, unexpected error): %s",tag,ME.message))
    fprintf("%s\n",getReport(ME,'extended','hyperlinks','off'))
    T = [];
end
end

function safesummary(fn)
try
    fn();
catch ME
    fprintf("Summary FAILED: %s\n",ME.message)
    fprintf("%s\n",getReport(ME,'extended','hyperlinks','off'))
end
end

%% ----- stage A -----
function A = runA(k,ctx)
cfg = ctx.cfg0; sg = cfg.sigmas(k); cfg.sigma = single(sg/255); T = cfg.sapT(k);
V = ctx.V; nV = numel(V); taus = cfg.tauList; nT = numel(taus);
imgs = cfg.imgIdx; nI = numel(imgs); iSel = find(imgs == cfg.selImg,1);
A = struct('sg',sg,'T',T,'imgs',imgs,'cfg',cfg,'V',V,'tStart',datetime('now'));
A.W = cell(nI,1); A.N = cell(nI,nT,nV); A.SA = cell(nI,nV); A.selTau = nan(1,nV);
df = @(name) fullfile(ctx.dat,name);
cf = @(name) fullfile(ctx.cdir,name);
if sg == 25
    fW = @(i) df(sprintf('sbdm_salsun_%02d_W.mat',i));
    fNR = @(i,t) df(sprintf('sbdm_salsun_hard_%02d_N4_t%d.mat',i,t));
else
    fW = @(i) df(sprintf('sbdm_salsun_s%02d_%02d_W.mat',sg,i));
    fNR = @(i,t) df(sprintf('sbdm_salsun_on_s%02d_%02d_Nh_t%d.mat',sg,i,t));
end
fSR = @(i,pt) df(sprintf('sbdm_salsun_on_s%02d_%02d_SAPh_p%d_b%d_it150.mat',sg,i,pt,T));
fN = @(i,key,t) cf(sprintf('sbdm_salsun_dc_s%02d_%02d_N%s_t%d.mat',sg,i,key,t));
fS = @(i,key,pt) cf(sprintf('sbdm_salsun_dc_s%02d_%02d_SA%s_p%d_b%d_it%d.mat',sg,i,key,pt,T,cfg.nItSa));
A.title = sprintf("Stage A, sigma = %d/255",sg);
fprintf("\n%s  ===== %s =====\n",string(datetime('now')),A.title)
% estimate (units without a cache; SA counted for both taus' worst case = 1 per variant)
e = 0;
for i = 1:nI
    for t = taus
        if ~isfile(fN(imgs(i),"D",t)), e = e + estget(ctx.ST,"N2"); end
        if ~isfile(fN(imgs(i),"L",t)), e = e + estget(ctx.ST,"N1"); end
        if ~isfile(fN(imgs(i),"LD",t)), e = e + estget(ctx.ST,"N2"); end
    end
    e = e + 3*estget(ctx.ST,"SA");
end
A.estSec = e;
fprintf("Estimate for this sub-stage (upper bound, SA counted as not cached): %.1f min -> would end %s (deadline %s)\n",...
    e/60,string(datetime('now') + seconds(e)),string(ctx.deadline))
for i = 1:nI
    idx = imgs(i); seed = cfg.seedBase + idx;
    lab = sprintf("A s%02d kodim%02d",sg,idx);
    [x,v] = fcn_ssloadimage(ctx.dat,idx,cfg,ctx.useGpu);
    tImg = tic;
    fprintf("\n%s  %s  noisy %.2f dB\n",string(datetime('now')),lab,psnr(gather(v),gather(x)))
    A.W{i} = loadopt(ctx,lab + " W",fW(idx));
    for j = 1:nT
        t = taus(j);
        qR = loadopt(ctx,lab + " N R t" + t,fNR(idx,t));
        if ~isempty(qR), qR.leak = false; qR.dcShrink = false; qR.kappa0 = NaN; qR.gain0 = NaN; end
        A.N{i,j,1} = qR;
        if ~isempty(qR)
            A.N{i,j,2} = unit(ctx,lab + " N D t" + t,fN(idx,"D",t),"N2",...
                @() fcn_dcnrun(x,v,cfg,seed,false,true,t,qR.prm.th{1}));
        else
            addskip(ctx,sprintf("SKIPPED %s N D t%d: R angles missing",lab,t))
        end
        qL = unit(ctx,lab + " N L t" + t,fN(idx,"L",t),"N1",...
            @() fcn_dcnrun(x,v,cfg,seed,true,false,t,[]));
        A.N{i,j,3} = qL;
        if ~isempty(qL)
            A.N{i,j,4} = unit(ctx,lab + " N LD t" + t,fN(idx,"LD",t),"N2",...
                @() fcn_dcnrun(x,v,cfg,seed,true,true,t,qL.th));
        else
            addskip(ctx,sprintf("SKIPPED %s N LD t%d: L angles missing",lab,t))
        end
    end
    if i == iSel
        for kv = 1:nV
            P = cellfun(@(q) getf(q,'psnr'),A.N(i,:,kv));
            if any(~isnan(P))
                [~,jb] = max(P); A.selTau(kv) = taus(jb);
                fprintf("  tau selection on kodim%02d, variant %s (PSNR):%s -> tau=%d\n",idx,V(kv).key,...
                    sprintf(" t%d %.4f",[taus; P]),A.selTau(kv))
            end
        end
    end
    % SAP-hard
    if isnan(A.selTau(1))
        addskip(ctx,sprintf("MISSING %s SA R: no tau selection for R",lab))
    else
        A.SA{i,1} = loadopt(ctx,lab + " SA R",fSR(idx,A.selTau(1)));
    end
    for kv = 2:nV
        if isnan(A.selTau(kv))
            addskip(ctx,sprintf("SKIPPED %s SA %s: no tau selection (kodim09 N missing)",lab,V(kv).key))
            continue
        end
        qP = A.N{i,taus == A.selTau(kv),kv};
        if isempty(qP)
            addskip(ctx,sprintf("SKIPPED %s SA %s: pilot N %s t%d missing",lab,V(kv).key,V(kv).key,A.selTau(kv)))
            continue
        end
        Vk = V(kv);
        A.SA{i,kv} = unit(ctx,lab + " SA " + Vk.key,fS(idx,Vk.key,A.selTau(kv)),"SA",...
            @() runsapv(x,v,cfg,seed,qP,Vk,T));
    end
    fprintf("  %s done (%.1f s for this image)\n",lab,toc(tImg))
end
A.tEnd = datetime('now');
fprintf("%s  %s finished (%.1f min)\n",string(A.tEnd),A.title,minutes(A.tEnd - A.tStart))
end

function q = runsapv(x,v,cfg,seed,qP,Vk,T)
[anet,snet,info] = fcn_dclsunpair(size(v),cfg.stride,cfg.ovlp,qP.leak);
th = qP.th; ab = qP.ab;
if isa(v,'gpuArray'), th = dlupdate(@gpuArray,th); ab = dlupdate(@gpuArray,ab); end
Pf = @(Z) fcn_dcdenoise(anet,snet,info,th,ab,Z,cfg.sigma);
q.sigmaP = sqrt(max(double(qP.sure),eps));
thr = T*q.sigmaP;
r = fcn_dcsaprun(x,v,cfg,seed,Pf,qP.img,Vk.leak,Vk.dcs,thr,cfg.nItSa);
r.sigmaP = q.sigmaP; r.T = T; r.pilotTau = qP.tau; r.variant = Vk.key;
q = r;
end

%% ----- stage B -----
function B = runB(k,ctx,trainSet)
cfg = ctx.cfg0; sg = cfg.sigmas(k); cfg.sigma = single(sg/255);
V = ctx.V; nV = numel(V); kinds = ["N" "SA"];
imgs = cfg.imgIdx;
B = struct('sg',sg,'imgs',imgs,'cfg',cfg,'V',V,'kinds',kinds,'tStart',datetime('now'));
B.Q = cell(2,nV);
B.title = sprintf("Stage B, sigma = %d/255",sg);
cf = @(name) fullfile(ctx.cdir,name);
fB = @(kind,key) cf(sprintf('sbdm_salsun_dc_sup_s%02d_%s%s_it%d.mat',sg,kind,key,cfg.sup.nIters));
fprintf("\n%s  ===== %s (%d its per training) =====\n",string(datetime('now')),B.title,cfg.sup.nIters)
e = 0;
for kk = 1:2
    for kv = 1:nV
        if ~isfile(fB(kinds(kk),V(kv).key)), e = e + estget(ctx.ST,"sup" + kinds(kk),cfg.sup.nIters); end
    end
end
B.estSec = e;
fprintf("Estimate for this sub-stage: %.1f min -> would end %s (deadline %s)\n",...
    e/60,string(datetime('now') + seconds(e)),string(ctx.deadline))
for kk = 1:2
    for kv = 1:nV
        kind = kinds(kk); Vk = V(kv);
        lab = sprintf("B s%02d sup %s %s",sg,kind,Vk.key);
        f = fB(kind,Vk.key);
        B.Q{kk,kv} = unit(ctx,lab,f,"sup" + kind,@() runsup(kind,Vk,cfg,trainSet,ctx,string(f) + "_ckpt.mat"),cfg.sup.nIters);
    end
end
B.tEnd = datetime('now');
fprintf("%s  %s finished (%.1f min)\n",string(B.tEnd),B.title,minutes(B.tEnd - B.tStart))
end

function q = runsup(kind,Vk,cfg,trainSet,ctx,ckpt)
o = cfg.sup;
o.sigma = cfg.sigma; o.leak = Vk.leak; o.dcShrink = Vk.dcs;
o.stride = cfg.stride; o.ovlp = cfg.ovlp; o.bottleneck = cfg.bottleneck;
o.ckptFile = char(ckpt);
t = tic;
[mdl,tr] = fcn_dcsuptrain(kind,trainSet,o);
q.mdl = mdl; q.tr = tr; q.variant = Vk.key; q.kind = kind;
nb = max(1,min(100,numel(tr.hist)));
q.lossEnd = mean(tr.hist(end-nb+1:end));
fprintf("    trained: %.1f s (%.3f s/it), loss (mean of last %d its) %.4e\n",tr.tTrain,tr.tPerIt,nb,q.lossEnd)
netsSz = [];
if kind == "SA"
    ob = struct('estimator',"net",'leak',Vk.leak,'bottleneck',cfg.bottleneck,...
        'shrink',true,'dcShrink',Vk.dcs,'dcInit',"id",'sigma',cfg.sigma);
    netsSz = fcn_dcbuildsalsun(cfg.cropSize,cfg.stride,cfg.ovlp,ob);
end
imgs = cfg.imgIdx;
for i = 1:numel(imgs)
    [x,v] = fcn_ssloadimage(ctx.dat,imgs(i),cfg,ctx.useGpu);
    r = fcn_dcsupeval(mdl,x,v,cfg,netsSz);
    q.res(i) = r;
    fprintf("    kodim%02d  %.4f dB  SSIM %.4f  prErrX %.2e\n",imgs(i),r.psnr,r.ssim,r.prErrX)
end
q.tTotal = toc(t);
if isfile(ckpt), delete(ckpt); end
end

%% ----- units, time estimates -----
function e = estget(ST,key,nIt)
if nargin < 3, nIt = 1; end
if isKey(ST,char(key)), e = ST(char(key)); return, end
switch key
    case "N1", e = 185;              % N-hard step 1 (500 its) + step 2 (overnight: 181 s)
    case "N2", e = 10;               % step 2 only, 2 DC inits (probe: 4.1 s)
    case "SA", e = 75;               % SAP-hard it150 (probe: 67.3 s) + DC inits
    case "supN", e = 0.53*nIt + 30;  % probe: 0.51-0.53 s/it (batch 32) + build/eval
    case "supSA", e = 0.70*nIt + 60; % probe: 0.69-0.70 s/it + build/eval (8.5 s per image)
    otherwise, e = 200;
end
end

function q = unit(ctx,lab,f,key,fn,nIt)
if nargin < 6, nIt = 1; end
if isfile(f)
    S = load(f); q = S.q;
    fprintf("  %-28s cached (%s)\n",lab,f)
    return
end
e = estget(ctx.ST,key,nIt)*ctx.margin;
tEnd = datetime('now') + seconds(e);
if tEnd > ctx.deadline
    addskip(ctx,sprintf("SKIPPED %s: estimate %.0f s, would end %s > deadline %s",...
        lab,e,string(tEnd),string(ctx.deadline)))
    q = [];
    return
end
fprintf("  %-28s start (estimate %.0f s)\n",lab,e)
t = tic;
try
    q = fn();
catch ME
    addskip(ctx,sprintf("FAILED %s: %s",lab,ME.message))
    fprintf("%s\n",getReport(ME,'extended','hyperlinks','off'))
    q = [];
    return
end
dt = toc(t);
q.tWall = dt;
if ~isfield(q,'tTotal'), q.tTotal = dt; end
ctx.ST(char(key)) = dt;
ftmp = string(f) + ".tmp.mat";
save(ftmp,'q','-v7.3'); movefile(ftmp,f);
printq(lab,q)
end

function q = loadopt(ctx,lab,f)
if isfile(f)
    S = load(f); q = S.q;
    fprintf("  %-28s cached (%s)\n",lab,f)
else
    addskip(ctx,sprintf("MISSING %s: cache %s not found",lab,f))
    q = [];
end
end

function addskip(ctx,msg)
ctx.ST('skips') = [ctx.ST('skips'); string(msg)];
fprintf("  %s\n",msg)
end

function printq(lab,q)
if isfield(q,'res')
    s = sprintf("  %-28s mean PSNR %.4f dB (%.1f s)",lab,mean([q.res.psnr]),q.tWall);
else
    s = sprintf("  %-28s %.4f dB  SSIM %.4f  1-SURE/MSE %.4f",lab,q.psnr,q.ssim,1 - q.sure/q.mse);
    if isfield(q,'selInit'), s = s + "  DC init " + q.selInit + sprintf(" kappa0 %.4f g0 %.4f",q.kappa0,q.gain0); end
    if isfield(q,'sureExplicitSame')
        s = s + sprintf("  thr/sigma %.3f  pilot %.4f dB  fixed-vs-explicit rel %.1e",q.thrOverSigma,q.psnrPilot,...
            abs(q.sure - q.sureExplicitSame)/abs(q.sureExplicitSame));
    end
    s = s + sprintf("  (%.1f s)",q.tWall);
end
fprintf("%s\n",s)
end

function y = getf(q,f)
y = NaN;
if isempty(q), return, end
try, y = double(q.(f)); catch, y = NaN; end
end

function M = getm(C,fn)
M = nan(size(C));
for k = 1:numel(C)
    if isempty(C{k}), continue, end
    try, M(k) = double(fn(C{k})); catch, M(k) = NaN; end
end
end

function t = ttot(q)
if isfield(q,'tWall'), t = q.tWall;
elseif isfield(q,'tTotal'), t = q.tTotal;
else, t = NaN;
end
end

%% ----- summaries -----
function C = nsel(A)
% N results at the selected tau, nI x nV cell
nI = numel(A.imgs); nV = numel(A.V);
C = cell(nI,nV);
for kv = 1:nV
    if isnan(A.selTau(kv)), continue, end
    j = find(A.cfg.tauList == A.selTau(kv),1);
    C(:,kv) = A.N(:,j,kv);
end
end

function summaryA(file,TA,ctx)
fid = fopen(file,'w'); cleaner = onCleanup(@() fclose(fid));
cfg = ctx.cfg0;
out(fid,"Stage A: two-step design, observation only. DC shrinkage (D), DC leakage allowed (L), both (LD), reference (R).\n");
out(fid,"Written %s. Deadline %s.\n",string(datetime('now')),string(ctx.deadline));
out(fid,"Kodak kodim09/15/19/23, 256x256 centre crops, grayscale, same noise seeds as before (rng(image index)).\n");
out(fid,"Fair selection: by PSNR on kodim09 only (tau of N-hard per variant); reported on kodim15/19/23 (test).\n");
out(fid,"DC init of step 2 (D, LD): two runs (ac = kappa0 0.9 as AC, id = kappa0 0.01); the lower SURE is kept (v only).\n");
out(fid,"N-hard: shared angles, 4x4 blocks, ovlp 3x3, S = 1, step 1 hard mask |c| > tau*sigma on v (%d its), step 2 exact SURE (%d its).\n",cfg.nIt1N,cfg.nIt2);
out(fid,"  R: caches. D: R step-1 angles (cache) + new step 2. L: new step 1 without the no-DC-leakage constraint. LD: L angles + step 2 with DC.\n");
out(fid,"SAP-hard: pilot = N-hard of the same variant (selected tau), b = %d, step 1 on p (%d its) with threshold T*sigma_p,\n",cfg.bottleneck,cfg.nItSa);
out(fid,"  sigma_p = sqrt(SURE of the pilot), T = 2 at sigma 25 (b2), 3 at sigma 10 (b3); angles from p frozen; step 2 SURE on v, %d fixed probes (check %d fresh). R: caches.\n",...
    cfg.nProbePilot,cfg.nProbeFresh);
out(fid,"W: wiener2 5x5, known noise variance (caches).\n");
out(fid,"mean(all) / mean(test) omit missing values ('-' = not run or skipped).\n");
for k = 1:numel(TA)
    A = TA{k};
    if isempty(A), out(fid,"\n(sigma %d/255: sub-stage failed, see the list at the end)\n",cfg.sigmas(k)); continue, end
    sumA(fid,A);
end
writeskips(fid,ctx,"A ");
end

function sumA(fid,A)
imgs = A.imgs(:); rows = compose("kodim%02d",imgs);
iTest = find(imgs ~= A.cfg.selImg); iSel = find(imgs == A.cfg.selImg,1);
rowsM = [rows; "mean(all)"; "mean(test)"];
addm = @(M) [M; mean(M,1,'omitnan'); mean(M(iTest,:),1,'omitnan')];
keys = [A.V.key]; nV = numel(keys); taus = A.cfg.tauList;
G = @getm;
CN = nsel(A); CS = A.SA;
pW = G(A.W,@(q) q.psnr); sW = G(A.W,@(q) q.ssim);
pN = G(CN,@(q) q.psnr); sN = G(CN,@(q) q.ssim);
pS = G(CS,@(q) q.psnr); sS = G(CS,@(q) q.ssim);
cN = "N " + keys; cS = "SA " + keys;
out(fid,"\n==================== %s ====================\n",A.title);
out(fid,"Run: %s -> %s (estimate before the sub-stage, upper bound: %.1f min).\n",string(A.tStart),string(A.tEnd),A.estSec/60);
out(fid,"\nN-hard tau selection on kodim09 (PSNR [dB]):\n");
for kv = 1:nV
    P = squeeze(G(A.N(iSel,:,kv),@(q) q.psnr)).';
    if isnan(A.selTau(kv)), s = "none"; else, s = "tau=" + A.selTau(kv); end
    out(fid,"  %-3s%s -> %s\n",keys(kv),sprintf(" t%d %.4f",[taus; P(:).']),s);
end
printtable(fid,"PSNR [dB], N at the selected tau",rowsM,["W" cN cS],addm([pW pN pS]),"%.4f")
printtable(fid,"SSIM",rowsM,["W" cN cS],addm([sW sN sS]),"%.4f")
printtable(fid,"PSNR difference to the reference R [dB] (N V - N R, SA V - SA R)",rowsM,...
    [cN(2:end) + "-R" cS(2:end) + "-R"],addm([pN(:,2:end) - pN(:,1) pS(:,2:end) - pS(:,1)]),"%.4f")
printtable(fid,"SSIM difference to the reference R",rowsM,...
    [cN(2:end) + "-R" cS(2:end) + "-R"],addm([sN(:,2:end) - sN(:,1) sS(:,2:end) - sS(:,1)]),"%.4f")
printtable(fid,"SA - N for each variant, PSNR [dB]",rowsM,"SA-N " + keys,addm(pS - pN),"%.4f")
printtable(fid,"SA - N for each variant, SSIM",rowsM,"SA-N " + keys,addm(sS - sN),"%.4f")
printtable(fid,"Difference to W, PSNR [dB]",rowsM,[cN cS] + "-W",addm([pN pS] - pW),"%.4f")
printtable(fid,"Difference to W, SSIM",rowsM,[cN cS] + "-W",addm([sN sS] - sW),"%.4f")
% all N settings
cAll = strings(1,0); pAll = []; sAll = [];
for kv = 1:nV
    for j = 1:numel(taus)
        cAll(end+1) = "N " + keys(kv) + "/t" + taus(j); %#ok<AGROW>
        pAll = [pAll G(A.N(:,j,kv),@(q) q.psnr)]; %#ok<AGROW>
        sAll = [sAll G(A.N(:,j,kv),@(q) q.ssim)]; %#ok<AGROW>
    end
end
out(fid,"\nAll N-hard settings, all images (information; choosing from these is not a fair selection):\n");
printtable(fid,"PSNR [dB]",rowsM,cAll,addm(pAll),"%.4f")
printtable(fid,"SSIM",rowsM,cAll,addm(sAll),"%.4f")
% SURE
printtable(fid,"1 - SURE/MSE (> 0: SURE too low). N: exact; SA: 8 fixed probes",rowsM,[cN cS],...
    addm(1 - [G(CN,@(q) q.sure/q.mse) G(CS,@(q) q.sure/q.mse)]),"%.4f")
printtable(fid,"1 - SURE/MSE, SA: 32 fresh probes (explicit chain)",rowsM,cS,addm(1 - G(CS,@(q) q.sureFresh/q.mse)),"%.4f")
printtable(fid,"Check: |SURE fixed probes - SURE explicit chain, same probes| / |explicit| (SA)",rowsM,cS,...
    addm(G(CS,@(q) abs(q.sure - q.sureExplicitSame)/abs(q.sureExplicitSame))),"%.2e")
printtable(fid,"Check (new N runs): | ||f(v)-v||^2 image domain - coefficient domain | / N",rowsM,cN,addm(G(CN,@(q) q.resCheck)),"%.2e")
% DC shrinkage
out(fid,"\nDC shrinkage (D, LD): selected DC init (by SURE), and learned kappa_0, g_0.\n");
iD = find(contains(keys,"D"));
for kv = iD
    for i = 1:numel(imgs)
        for typ = ["N" "SA"]
            if typ == "N", q = CN{i,kv}; else, q = CS{i,kv}; end
            if isempty(q), continue, end
            s1 = sprintf("ac: SURE %.6e PSNR %.4f",q.init(1).sure,q.init(1).psnr);
            s2 = sprintf("id: SURE %.6e PSNR %.4f",q.init(2).sure,q.init(2).psnr);
            out(fid,"  %-3s %-2s %s  sel %-2s  kappa0 %.4f g0 %.4f  tau0 = kappa0/g0 %.4f | %s | %s\n",keys(kv),typ,rows(i),...
                q.selInit,q.kappa0,q.gain0,q.kappa0/q.gain0,s1,s2);
        end
    end
end
% SA pilot / threshold
printtable(fid,"SAP-hard: pilot PSNR [dB] (= N of the same variant, selected tau)",rowsM,cS,addm(G(CS,@(q) q.psnrPilot)),"%.4f")
printtable(fid,"SAP-hard: step-1 threshold / sigma",rowsM,cS,addm(G(CS,@(q) q.thrOverSigma)),"%.4f")
M = [G(CN,@(q) q.prErrX) G(CS,@(q) q.prErrX)];
printtable(fid,"PR error ||DEx - x||/||x||, last row = max",[rows; "max"],[cN cS],[M; max(M,[],1)],"%.2e")
printtable(fid,"Time per unit [s] (N R: old run; N D/LD: step 2 only, step-1 angles reused)",rowsM,[cN cS],...
    addm([G(CN,@ttot) G(CS,@ttot)]),"%.1f")
end

function summaryB(file,TB,TA,ctx)
fid = fopen(file,'w'); cleaner = onCleanup(@() fclose(fid));
cfg = ctx.cfg0; o = cfg.sup;
out(fid,"Stage B: fully supervised LSUN-AE (clean targets). Variants R, D (DC shrinkage), L (DC leakage allowed), LD (both).\n");
out(fid,"Written %s. Deadline %s.\n",string(datetime('now')),string(ctx.deadline));
out(fid,"Training: %d patches %dx%d of kodim%02d-%02d (rng(1), as main_sbdm.m), batch %d, fresh noise each iteration, fixed sigma, MSE, %d Adam its, cosine decay.\n",...
    o.nCrops*numel(o.trainIdx),o.patch(1),o.patch(2),o.trainIdx(1),o.trainIdx(end),o.mb,o.nIters);
out(fid,"  N: shared angles + shrinkage, lr %g -> %g. SA: estimator (b = %d, sees v) lr %g -> %g, shrinkage lr %g -> %g.\n",...
    o.lrN(1),o.lrN(2),cfg.bottleneck,o.lrEst(1),o.lrEst(2),o.lrShr(1),o.lrShr(2));
out(fid,"  Init: zero angles / zero estimator output (block DCT start); AC kappa0 0.9 tau0 2; DC kappa0 0.01 tau0 2.\n");
out(fid,"Evaluation: the same 256x256 crops and noise as stage A. No choice is made in this stage.\n");
out(fid,"Observation-only (obs) = stage A result of the same variant (N at the selected tau, SAP-hard).\n");
for k = 1:numel(TB)
    B = TB{k};
    if isempty(B), out(fid,"\n(sigma %d/255: sub-stage failed, see the list at the end)\n",cfg.sigmas(k)); continue, end
    A = []; if k <= numel(TA), A = TA{k}; end
    sumB(fid,B,A);
end
writeskips(fid,ctx,"B ");
end

function sumB(fid,B,A)
imgs = B.imgs(:); rows = compose("kodim%02d",imgs); nI = numel(imgs);
iTest = find(imgs ~= B.cfg.selImg);
rowsM = [rows; "mean(all)"; "mean(test)"];
addm = @(M) [M; mean(M,1,'omitnan'); mean(M(iTest,:),1,'omitnan')];
keys = [B.V.key]; nV = numel(keys);
pick = @(fld) cellfun(@(q) pickres(q,fld,nI),B.Q,'UniformOutput',false);
Pc = pick('psnr'); Sc = pick('ssim');
pN = [Pc{1,:}]; pS = [Pc{2,:}]; sN = [Sc{1,:}]; sS = [Sc{2,:}];
cN = "supN " + keys; cS = "supSA " + keys;
pW = nan(nI,1); sW = pW; pNo = nan(nI,nV); pSo = pNo; sNo = pNo; sSo = pNo;
if ~isempty(A)
    pW = getm(A.W,@(q) q.psnr); sW = getm(A.W,@(q) q.ssim);
    CN = nsel(A);
    pNo = getm(CN,@(q) q.psnr); sNo = getm(CN,@(q) q.ssim);
    pSo = getm(A.SA,@(q) q.psnr); sSo = getm(A.SA,@(q) q.ssim);
end
out(fid,"\n==================== %s ====================\n",B.title);
out(fid,"Run: %s -> %s (estimate before the sub-stage %.1f min).\n",string(B.tStart),string(B.tEnd),B.estSec/60);
L = getm(B.Q,@(q) q.lossEnd); Tt = getm(B.Q,@(q) q.tr.tTrain); Ti = getm(B.Q,@(q) q.tr.tPerIt);
Np = getm(B.Q,@(q) q.tr.nParams);
out(fid,"\nTraining: final loss (mean of the last 100 its, MSE per pixel), training time [s], s/it, number of parameters\n");
for kk = 1:2
    for kv = 1:nV
        out(fid,"  %-6s %-3s loss %.4e  train %.1f s  %.3f s/it  params %d\n",B.kinds(kk),keys(kv),L(kk,kv),Tt(kk,kv),Ti(kk,kv),Np(kk,kv));
    end
end
printtable(fid,"PSNR [dB]",rowsM,["W" cN cS],addm([pW pN pS]),"%.4f")
printtable(fid,"SSIM",rowsM,["W" cN cS],addm([sW sN sS]),"%.4f")
printtable(fid,"PSNR difference to the supervised reference R [dB]",rowsM,...
    [cN(2:end) + "-R" cS(2:end) + "-R"],addm([pN(:,2:end) - pN(:,1) pS(:,2:end) - pS(:,1)]),"%.4f")
printtable(fid,"SSIM difference to the supervised reference R",rowsM,...
    [cN(2:end) + "-R" cS(2:end) + "-R"],addm([sN(:,2:end) - sN(:,1) sS(:,2:end) - sS(:,1)]),"%.4f")
printtable(fid,"SA - N for each variant (supervised), PSNR [dB]",rowsM,"SA-N " + keys,addm(pS - pN),"%.4f")
printtable(fid,"SA - N for each variant (supervised), SSIM",rowsM,"SA-N " + keys,addm(sS - sN),"%.4f")
printtable(fid,"Supervised - observation-only (same variant), PSNR [dB]",rowsM,[cN cS] + "-obs",addm([pN - pNo pS - pSo]),"%.4f")
printtable(fid,"Supervised - observation-only (same variant), SSIM",rowsM,[cN cS] + "-obs",addm([sN - sNo sS - sSo]),"%.4f")
printtable(fid,"Difference to W, PSNR [dB]",rowsM,[cN cS] + "-W",addm([pN pS] - pW),"%.4f")
printtable(fid,"Difference to W, SSIM",rowsM,[cN cS] + "-W",addm([sN sS] - sW),"%.4f")
K0 = getm(B.Q,@(q) q.res(1).kappa0); G0 = getm(B.Q,@(q) q.res(1).gain0);
out(fid,"\nLearned DC shrinkage (D, LD): kappa_0, g_0, tau_0 = kappa_0/g_0\n");
for kk = 1:2
    for kv = find(contains(keys,"D"))
        out(fid,"  %-6s %-3s kappa0 %.4f  g0 %.4f  tau0 %.4f\n",B.kinds(kk),keys(kv),K0(kk,kv),G0(kk,kv),K0(kk,kv)/G0(kk,kv));
    end
end
Pe = pick('prErrX'); M = [[Pe{1,:}] [Pe{2,:}]];
printtable(fid,"PR error ||DEx - x||/||x||, last row = max",[rows; "max"],[cN cS],[M; max(M,[],1)],"%.2e")
Su = pick('sure'); Mn = pick('mse');
printtable(fid,"supervised N: 1 - SURE/MSE (exact SURE, information only)",rowsM,cN,addm(1 - [Su{1,:}]./[Mn{1,:}]),"%.4f")
Te = pick('tEval');
printtable(fid,"Time [s]: evaluation per image",rowsM,[cN cS],addm([[Te{1,:}] [Te{2,:}]]),"%.1f")
printtable(fid,"Time [s]: whole unit (training + evaluation of 4 images)",["unit"],[cN cS],...
    reshape(getm(B.Q,@ttot).',1,[]),"%.1f")
end

function y = pickres(q,fld,nI)
y = nan(nI,1);
if isempty(q) || ~isfield(q,'res'), return, end
for i = 1:min(nI,numel(q.res))
    try, y(i) = double(q.res(i).(fld)); catch, y(i) = NaN; end
end
end

function writeskips(fid,ctx,stageKey)
sk = ctx.ST('skips');
sk = sk(startsWith(sk,["SKIPPED " "FAILED " "MISSING "] + stageKey));
out(fid,"\n---- Skipped / failed / missing units in this stage (%d) ----\n",numel(sk));
for k = 1:numel(sk), out(fid,"  %s\n",sk(k)); end
out(fid,"\n");
end

function out(fid,fmt,varargin)
fprintf(fid,fmt,varargin{:});
fprintf(1,fmt,varargin{:});
end

function printtable(fid,ttl,rows,cols,M,fmt)
out(fid,"\n%s\n",ttl);
w = max(10,max(strlength(cols))+1);
s = sprintf("%-11s","");
for j = 1:numel(cols), s = s + sprintf("%*s",w,cols(j)); end
out(fid,"%s\n",s);
for i = 1:numel(rows)
    s = sprintf("%-11s",rows(i));
    for j = 1:size(M,2)
        if isnan(M(i,j)), c = "-"; else, c = strtrim(sprintf(fmt,M(i,j))); end
        s = s + sprintf("%*s",w,c);
    end
    out(fid,"%s\n",s);
end
end
