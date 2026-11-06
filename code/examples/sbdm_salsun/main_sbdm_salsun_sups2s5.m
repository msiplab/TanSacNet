%% main_sbdm_salsun_sups2s5.m
% Stage B of main_sbdm_salsun_dcsuggest.m again, at sigma = 2/255 and
% 5/255, reference variant R only (no DC leakage, no DC shrinkage),
% supervised N and supervised SA. (2026-11-06)
%
% Exactly the same code, settings, training data and crops as stage B:
%   - training: fcn_dcsuptrain (unchanged), cfg0.sup copied from
%     main_sbdm_salsun_dcsuggest.m (1500 its, batch 32, lr, seed 1, ...)
%   - training patches: rng(1) (+ gpurng(1)) and fcn_loadkodak, as stage B
%   - evaluation: fcn_dcsupeval (unchanged), same 256x256 crops and noise
%     seeds (fcn_ssloadimage, rng(image index)), kodim09/15/19/23
%   - the local functions runsup / unit / printtable are copies of the
%     ones in main_sbdm_salsun_dcsuggest.m
% Only cfg0.sigmas = [2 5] and the variant list (R only) differ.
% W (wiener2 5x5, known noise variance) is computed here for both sigmas
% (no sigma 2 cache exists); for sigma 5 it is compared with the cache of
% the overnight run (sbdm_salsun_on_s05_<img>_W.mat).
%
% Check before the run: 20 its of sigma 25 N R and SA R; the first loss
% (before any update) must equal hist(1) of the stage B models
% (same crops, same noise, same init). Later its use a different lr
% schedule (nIters = 20), so they are shown for information only.
%
% Caches: data/sbdm_salsun_dc_sup_s%02d_{N,SA}R_it1500.mat (same pattern as
% stage B, new sigmas only), data/sbdm_salsun_sups2s5_s%02d_%02d_W.mat,
% data/sbdm_salsun_sups2s5_results.mat
% Summary: logs/sups2s5_summary.txt
%   nohup matlab -licmode onlinelicensing -batch "main_sbdm_salsun_sups2s5" \
%       > ~/Documents/SBDMwLSUN/logs/sups2s5.log 2>&1 &
% Quick test (3 its, separate folder):
%   matlab -licmode onlinelicensing -batch "onTest = true; onTestDir = '<dir>'; main_sbdm_salsun_sups2s5"

here = fileparts(mfilename('fullpath'));
cd(fullfile(here,'..','..')); setpath; cd(here)
addpath(fullfile(here,'..','sbdm'))     % helpers (called, not edited)
addpath(fullfile(here,'..','salsun'))   % maskLayer
warning('off','MATLAB:rmpath:DirNotFound')
rmpath(fullfile('/home/shogo/Workspace/GitHub/SaivDr','mexcodes'))
warning('on','MATLAB:rmpath:DirNotFound')

% ---- settings: identical to main_sbdm_salsun_dcsuggest.m except sigmas ----
cfg0 = fcn_ssconfig();
cfg0.K = 4;
cfg0.nProbePilot = 8;
cfg0.nProbeFresh = 32;
cfg0.bottleneck = 18;
cfg0.selImg = 9;
cfg0.tauList = [2 3];
cfg0.nItSa = 150;
cfg0.sigmas = [2 5];         % x/255  (stage B: [25 10])
cfg0.sapT = [2 3];
cfg0.sup = struct('nIters',1500,'mb',32,'lrN',[5e-3 5e-5],'lrEst',[1e-3 1e-5],...
    'lrShr',[5e-3 5e-5],'seed',1,'patch',[64 64],'nCrops',256,'trainIdx',1:8,'ckptEvery',200);
dat = fullfile(here,'..','..','..','data');
ctx.dat = dat; ctx.cdir = dat;
ctx.sdir = fullfile(here,'..','..','..','..','logs');
ctx.deadline = datetime('now') + hours(6);
ctx.margin = 1.05;
isTest = exist('onTest','var') && onTest;
nItCheck = 20;
if isTest
    cfg0.imgIdx = [9 15];
    cfg0.sup.nIters = 3; cfg0.sup.ckptEvery = 2; nItCheck = 3;
    ctx.cdir = onTestDir; ctx.sdir = onTestDir;
    if ~isfolder(onTestDir), mkdir(onTestDir); end
end
ctx.useGpu = canUseGPU;
ctx.cfg0 = cfg0;
ctx.ST = containers.Map();
ctx.ST('skips') = strings(0,1);
ctx.V = struct('key',"R",'leak',false,'dcs',false);   % reference variant only
fprintf("%s  start. GPU: %d. Deadline %s. Test mode: %d\n",string(datetime('now')),ctx.useGpu,string(ctx.deadline),isTest)
disp(cfg0); disp(cfg0.sup)

%% ===== training data (as stage B) =====
rng(1); if ctx.useGpu, gpurng(1); end            % as main_sbdm.m (fcn_seed(1))
[~,trainSet] = fcn_loadkodak(dat,cfg0.sup.trainIdx,cfg0.sup.patch,cfg0.sup.nCrops,ctx.useGpu);
fprintf("\n%s  training patches: %d of size %s\n",string(datetime('now')),numel(trainSet),mat2str(cfg0.sup.patch))

%% ===== check: same data / noise / init as stage B (sigma 25) =====
CHK = struct('kind',{},'hist1New',{},'hist1Old',{},'relMax2',{},'ok',{});
for kind = ["N" "SA"]
    try
        S = load(fullfile(dat,sprintf('sbdm_salsun_dc_sup_s25_%sR_it1500.mat',kind)));
        o = cfg0.sup; o.nIters = nItCheck; o.ckptFile = '';
        o.sigma = single(25/255); o.leak = false; o.dcShrink = false;
        o.stride = cfg0.stride; o.ovlp = cfg0.ovlp; o.bottleneck = cfg0.bottleneck;
        [~,trc] = fcn_dcsuptrain(kind,trainSet,o);
        hOld = S.q.tr.hist(1:nItCheck); hNew = trc.hist(:);
        c.kind = kind; c.hist1New = hNew(1); c.hist1Old = hOld(1);
        c.relMax2 = max(abs(hNew(2:end) - hOld(2:end))./abs(hOld(2:end)));
        c.ok = (hNew(1) == hOld(1));
        CHK(end+1) = c; %#ok<SAGROW>
        fprintf("CHECK sigma 25 %s R: hist(1) new %.10e  stage B %.10e  equal %d | its 2-%d (lr differs) max rel diff %.2e\n",...
            kind,c.hist1New,c.hist1Old,c.ok,nItCheck,c.relMax2)
        if ~c.ok, fprintf("WARNING: first loss differs from stage B (%s R)\n",kind), end
    catch ME
        fprintf("CHECK %s FAILED: %s\n",kind,ME.message)
    end
end

%% ===== runs =====
nSg = numel(cfg0.sigmas);
TB = cell(1,nSg);
for k = 1:nSg
    TB{k} = saferun(@() runB(k,ctx,trainSet),"B s" + cfg0.sigmas(k),ctx);
end
safesummary(@() summaryB(fullfile(ctx.sdir,'sups2s5_summary.txt'),TB,CHK,ctx));
save(fullfile(ctx.cdir,'sbdm_salsun_sups2s5_results.mat'),'TB','CHK','cfg0')

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

function B = runB(k,ctx,trainSet)
cfg = ctx.cfg0; sg = cfg.sigmas(k); cfg.sigma = single(sg/255);
V = ctx.V; nV = numel(V); kinds = ["N" "SA"];
imgs = cfg.imgIdx; nI = numel(imgs);
B = struct('sg',sg,'imgs',imgs,'cfg',cfg,'V',V,'kinds',kinds,'tStart',datetime('now'));
B.Q = cell(2,nV);
B.title = sprintf("Supervised, sigma = %d/255",sg);
cf = @(name) fullfile(ctx.cdir,name);
fB = @(kind,key) cf(sprintf('sbdm_salsun_dc_sup_s%02d_%s%s_it%d.mat',sg,kind,key,cfg.sup.nIters));
fprintf("\n%s  ===== %s (%d its per training) =====\n",string(datetime('now')),B.title,cfg.sup.nIters)
% W (wiener2 5x5, known noise variance), as runW of main_sbdm_salsun_overnight.m
B.W = cell(nI,1); B.Wold = nan(nI,2);
for i = 1:nI
    idx = imgs(i);
    [x,v] = fcn_ssloadimage(ctx.dat,idx,cfg,ctx.useGpu);
    f = cf(sprintf('sbdm_salsun_sups2s5_s%02d_%02d_W.mat',sg,idx));
    if isfile(f)
        S = load(f); q = S.q;
    else
        t = tic;
        z = single(wiener2(gather(double(v)),[5 5],double(cfg.sigma)^2));
        xg = gather(x);
        q = struct('psnr',psnr(z,xg),'ssim',ssim(z,xg),'mse',double(mean((z - xg).^2,'all')),'img',z);
        q.time = toc(t); q.tTotal = q.time;
        save(f,'q');
    end
    B.W{i} = q;
    s = sprintf("  W kodim%02d  noisy %.4f dB  W %.4f dB  SSIM %.4f",idx,psnr(gather(v),gather(x)),q.psnr,q.ssim);
    fo = fullfile(ctx.dat,sprintf('sbdm_salsun_on_s%02d_%02d_W.mat',sg,idx));
    if isfile(fo)
        So = load(fo); B.Wold(i,:) = [So.q.psnr So.q.ssim];
        s = s + sprintf("  | old cache %.4f dB SSIM %.4f, max |img diff| %.2e",So.q.psnr,So.q.ssim,...
            max(abs(So.q.img - q.img),[],'all'));
    end
    fprintf("%s\n",s)
end
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

function e = estget(ST,key,nIt)
if nargin < 3, nIt = 1; end
if isKey(ST,char(key)), e = ST(char(key)); return, end
switch key
    case "supN", e = 0.51*nIt + 10;  % stage B: 0.485-0.510 s/it + eval
    case "supSA", e = 0.65*nIt + 15; % stage B: 0.645-0.649 s/it + build/eval
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
fprintf("  %-28s mean PSNR %.4f dB (%.1f s)\n",lab,mean([q.res.psnr]),q.tWall);
end

function addskip(ctx,msg)
ctx.ST('skips') = [ctx.ST('skips'); string(msg)];
fprintf("  %s\n",msg)
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

function y = pickres(q,fld,nI)
y = nan(nI,1);
if isempty(q) || ~isfield(q,'res'), return, end
for i = 1:min(nI,numel(q.res))
    try, y(i) = double(q.res(i).(fld)); catch, y(i) = NaN; end
end
end

%% ----- summary (format of summaryB / sumB in main_sbdm_salsun_dcsuggest.m, R only) -----
function summaryB(file,TB,CHK,ctx)
fid = fopen(file,'w'); cleaner = onCleanup(@() fclose(fid));
cfg = ctx.cfg0; o = cfg.sup;
out(fid,"Supervised LSUN-AE (clean targets), stage B of main_sbdm_salsun_dcsuggest.m at sigma = 2/255 and 5/255, variant R only\n");
out(fid,"(no DC leakage, no DC shrinkage).\n");
out(fid,"Written %s.\n",string(datetime('now')));
out(fid,"Training: %d patches %dx%d of kodim%02d-%02d (rng(1), as main_sbdm.m), batch %d, fresh noise each iteration, fixed sigma, MSE, %d Adam its, cosine decay.\n",...
    o.nCrops*numel(o.trainIdx),o.patch(1),o.patch(2),o.trainIdx(1),o.trainIdx(end),o.mb,o.nIters);
out(fid,"  N: shared angles + shrinkage, lr %g -> %g. SA: estimator (b = %d, sees v) lr %g -> %g, shrinkage lr %g -> %g.\n",...
    o.lrN(1),o.lrN(2),cfg.bottleneck,o.lrEst(1),o.lrEst(2),o.lrShr(1),o.lrShr(2));
out(fid,"  Init: zero angles / zero estimator output (block DCT start); AC kappa0 0.9 tau0 2.\n");
out(fid,"Evaluation: 256x256 centre crops, kodim09/15/19/23, noise rng(image index). No choice is made.\n");
out(fid,"W: wiener2 5x5, known noise variance, computed in this run.\n");
out(fid,"\nCheck (sigma 25, %d units): first loss vs stage B model hist(1)\n",numel(CHK));   % (label fixed after the run: was "%d its")
for c = CHK
    out(fid,"  %-2s R: new %.10e  stage B %.10e  equal %d  (its 2+, other lr schedule: max rel diff %.2e)\n",...
        c.kind,c.hist1New,c.hist1Old,c.ok,c.relMax2);
end
for k = 1:numel(TB)
    B = TB{k};
    if isempty(B), out(fid,"\n(sigma %d/255: sub-stage failed, see the list at the end)\n",cfg.sigmas(k)); continue, end
    sumB(fid,B);
end
sk = ctx.ST('skips');
out(fid,"\n---- Skipped / failed / missing units (%d) ----\n",numel(sk));
for k = 1:numel(sk), out(fid,"  %s\n",sk(k)); end
out(fid,"\n");
end

function sumB(fid,B)
imgs = B.imgs(:); rows = compose("kodim%02d",imgs); nI = numel(imgs);
iTest = find(imgs ~= B.cfg.selImg);
rowsM = [rows; "mean(all)"; "mean(test)"];
addm = @(M) [M; mean(M,1,'omitnan'); mean(M(iTest,:),1,'omitnan')];
keys = [B.V.key]; nV = numel(keys);
pick = @(fld) cellfun(@(q) pickres(q,fld,nI),B.Q,'UniformOutput',false);
Pc = pick('psnr'); Sc = pick('ssim');
pN = [Pc{1,:}]; pS = [Pc{2,:}]; sN = [Sc{1,:}]; sS = [Sc{2,:}];
cN = "supN " + keys; cS = "supSA " + keys;
pW = getm(B.W,@(q) q.psnr); sW = getm(B.W,@(q) q.ssim);
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
printtable(fid,"SA - N for each variant (supervised), PSNR [dB]",rowsM,"SA-N " + keys,addm(pS - pN),"%.4f")
printtable(fid,"SA - N for each variant (supervised), SSIM",rowsM,"SA-N " + keys,addm(sS - sN),"%.4f")
printtable(fid,"Difference to W, PSNR [dB]",rowsM,[cN cS] + "-W",addm([pN pS] - pW),"%.4f")
printtable(fid,"Difference to W, SSIM",rowsM,[cN cS] + "-W",addm([sN sS] - sW),"%.4f")
if any(~isnan(B.Wold(:)))
    printtable(fid,"Check: W of this run - W cache of the overnight run (PSNR, SSIM)",rows,["dPSNR" "dSSIM"],...
        [pW sW] - B.Wold,"%.2e")
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
