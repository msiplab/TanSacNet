%% main_sbdm_salsun_supsize.m
% Is the gain of supervised SA-LSUN over supervised NSCLT (N) due to shift
% variance or to the many estimator parameters? (2026-11-06)
% Supervised SA, reference variant R (no DC leakage, no DC shrinkage), with
% a smaller angle estimator:
%   BottleneckSize b = 3 at sigma = 25, 10, 5, 2 /255
%   BottleneckSize b = 1 at sigma = 25, 10 /255
% N R and SA R b = 18 are NOT retrained: they are loaded from the caches of
% stage B (main_sbdm_salsun_dcsuggest.m, sigma 25/10) and run A
% (main_sbdm_salsun_sups2s5.m, sigma 2/5).
%
% Everything else is as stage B / run A: cfg0 and cfg0.sup are copied from
% main_sbdm_salsun_sups2s5.m (1500 its, batch 32, lrEst 1e-3 -> 1e-5,
% lrShr 5e-3 -> 5e-5, seed 1, 2048 patches 64x64 of kodim01-08, rng(1)),
% evaluation with fcn_dcsupeval (unchanged), same crops and noise seeds.
%
% Trainer: fcn_szsuptrain = exact copy of fcn_dcsuptrain + o.rngAfterBuild.
% The estimator init draws from the CPU stream and the number of draws
% depends on b; the crops are drawn with randi from the same stream. So the
% CPU stream is set, after the build, to the state that the b = 18 build
% leaves. Then every b gets the same crops and the same noise in every
% iteration as the stage B b = 18 training. Only the estimator differs.
%
% Checks before the runs (sigma 25, 20 its; the run stops if one fails):
%   1. N R, fcn_dcsuptrain (unchanged): first loss = stage B hist(1)
%   2. SA R b18, fcn_dcsuptrain (unchanged): first loss = stage B hist(1)
%   3. SA R b18, fcn_szsuptrain with rngAfterBuild: whole 20-it history
%      bitwise equal to check 2
%   4. SA R b3 / b1, fcn_szsuptrain: first loss = stage B SA hist(1)
%      (the estimator output is zero at the start, so the same transform;
%      equality means the same first mini-batch and noise)
% Also: the cached N / SA b18 models at sigma 25 are evaluated again and
% compared with their stored PSNR / SSIM.
%
% Caches: data/sbdm_salsun_supsize_s%02d_SAR_b%d_it1500.mat (one per run,
% checkpoint every 200 its, resumable), data/sbdm_salsun_supsize_results.mat
% Summary: logs/supsize_summary.txt
%   nohup matlab -licmode onlinelicensing -batch "main_sbdm_salsun_supsize" \
%       > ~/Documents/SBDMwLSUN/logs/supsize.log 2>&1 &
% Quick test (3 its, separate folder):
%   matlab -licmode onlinelicensing -batch "onTest = true; onTestDir = '<dir>'; main_sbdm_salsun_supsize"

here = fileparts(mfilename('fullpath'));
cd(fullfile(here,'..','..')); setpath; cd(here)
addpath(fullfile(here,'..','sbdm'))     % helpers (called, not edited)
addpath(fullfile(here,'..','salsun'))   % maskLayer
warning('off','MATLAB:rmpath:DirNotFound')
rmpath(fullfile('/home/shogo/Workspace/GitHub/SaivDr','mexcodes'))
warning('on','MATLAB:rmpath:DirNotFound')

% ---- settings: identical to main_sbdm_salsun_sups2s5.m except sigmas ----
cfg0 = fcn_ssconfig();
cfg0.K = 4;
cfg0.nProbePilot = 8;
cfg0.nProbeFresh = 32;
cfg0.bottleneck = 18;        % stage B / run A value (cached SA models)
cfg0.selImg = 9;
cfg0.tauList = [2 3];
cfg0.nItSa = 150;
cfg0.sigmas = [25 10 5 2];   % x/255, all four for the table
cfg0.sapT = [2 3];
cfg0.sup = struct('nIters',1500,'mb',32,'lrN',[5e-3 5e-5],'lrEst',[1e-3 1e-5],...
    'lrShr',[5e-3 5e-5],'seed',1,'patch',[64 64],'nCrops',256,'trainIdx',1:8,'ckptEvery',200);
bList = [18 3 1];            % columns of the tables (18 = cached)
RUNS = struct('sg',{25 25 10 10 5 2},'b',{3 1 3 1 3 3});   % new trainings
dat = fullfile(here,'..','..','..','data');
ctx.dat = dat; ctx.cdir = dat;
ctx.sdir = fullfile(here,'..','..','..','..','logs');
ctx.deadline = datetime('now') + hours(5);
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
fprintf("%s  start. GPU: %d. Deadline %s. Test mode: %d\n",string(datetime('now')),ctx.useGpu,string(ctx.deadline),isTest)
disp(cfg0); disp(cfg0.sup)
fprintf("New trainings (sigma, b):%s\n",sprintf(" (%d,%d)",[[RUNS.sg];[RUNS.b]]))
tRun0 = datetime('now');

%% ===== training data (as stage B) =====
rng(1); if ctx.useGpu, gpurng(1); end            % as main_sbdm.m (fcn_seed(1))
[~,trainSet] = fcn_loadkodak(dat,cfg0.sup.trainIdx,cfg0.sup.patch,cfg0.sup.nCrops,ctx.useGpu);
fprintf("\n%s  training patches: %d of size %s\n",string(datetime('now')),numel(trainSet),mat2str(cfg0.sup.patch))

%% ===== CPU rng state after the b = 18 build (as inside fcn_dcsuptrain) =====
rng(cfg0.sup.seed,'twister');
ob18 = struct('estimator',"net",'leak',false,'bottleneck',18,...
    'shrink',true,'dcShrink',false,'dcInit',"id",'sigma',single(25/255));
fcn_dcbuildsalsun(cfg0.sup.patch,cfg0.stride,cfg0.ovlp,ob18);
ctx.rng18 = rng;
clear ob18

%% ===== checks (sigma 25) =====
CHK = struct('name',{},'hist1New',{},'hist1Ref',{},'relMax2',{},'ok',{},'tPerIt',{},'nParams',{});
allOk = true;
SB = struct();
try
    for kind = ["N" "SA"]
        S = load(fullfile(dat,sprintf('sbdm_salsun_dc_sup_s25_%sR_it1500.mat',kind)));
        SB.(kind) = S.q;
    end
    oc = cfg0.sup; oc.nIters = nItCheck; oc.ckptFile = '';
    oc.sigma = single(25/255); oc.leak = false; oc.dcShrink = false;
    oc.stride = cfg0.stride; oc.ovlp = cfg0.ovlp;
    % 1, 2: unchanged trainer, b = 18
    h = struct();
    for kind = ["N" "SA"]
        o = oc; o.bottleneck = 18;
        [~,trc] = fcn_dcsuptrain(kind,trainSet,o);
        hRef = SB.(kind).tr.hist(1:nItCheck); hNew = trc.hist(:); h.(kind) = hNew;
        c = mkchk(sprintf("%s R b18, fcn_dcsuptrain: first loss vs stage B",kind),hNew,hRef,trc);
        c.ok = (hNew(1) == hRef(1));
        CHK(end+1) = c; %#ok<SAGROW>
        prchk(c,nItCheck,"lr differs")
    end
    % 3: new trainer, b = 18, rngAfterBuild -> whole history = check 2
    o = oc; o.bottleneck = 18; o.rngAfterBuild = ctx.rng18;
    [~,trc] = fcn_szsuptrain("SA",trainSet,o);
    hNew = trc.hist(:);
    c = mkchk("SA R b18, fcn_szsuptrain: whole history vs check 2",hNew,h.SA,trc);
    c.ok = isequal(hNew,h.SA);
    c.relMax2 = max(abs(hNew - h.SA)./abs(h.SA));   % here: over all its
    CHK(end+1) = c;
    fprintf("CHECK %s: hist(1) %.10e vs %.10e, all %d its bitwise equal %d (max rel diff %.2e)  params %d\n",...
        c.name,c.hist1New,c.hist1Ref,nItCheck,c.ok,c.relMax2,c.nParams)
    % 4: new trainer, b = 3 and 1 -> first loss = stage B SA
    for b = [3 1]
        o = oc; o.bottleneck = b; o.rngAfterBuild = ctx.rng18;
        [~,trc] = fcn_szsuptrain("SA",trainSet,o);
        hNew = trc.hist(:); hRef = SB.SA.tr.hist(1:nItCheck);
        c = mkchk(sprintf("SA R b%d, fcn_szsuptrain: first loss vs stage B SA b18",b),hNew,hRef,trc);
        c.ok = (hNew(1) == hRef(1));
        CHK(end+1) = c; %#ok<SAGROW>
        prchk(c,nItCheck,"other estimator")
        ctx.ST(char("supSA_b" + b)) = c.tPerIt*cfg0.sup.nIters + 15;   % estimate for the runs
    end
catch ME
    fprintf("CHECK FAILED (error): %s\n",getReport(ME,'extended','hyperlinks','off'))
    allOk = false;
end
allOk = allOk && ~isempty(CHK) && all([CHK.ok]);
if ~allOk
    fprintf("\n%s  A check failed. The runs are NOT started.\n",string(datetime('now')))
    save(fullfile(ctx.cdir,'sbdm_salsun_supsize_checks_failed.mat'),'CHK')
    return
end
fprintf("All %d checks passed.\n",numel(CHK))

% evaluation path: cached sigma 25 models evaluated again
EVC = evalcheck(ctx,SB);

%% ===== time estimate =====
e = 0;
for r = RUNS
    if ~isfile(fsz(ctx,r.sg,r.b,cfg0.sup.nIters)), e = e + estget(ctx.ST,"supSA_b" + r.b,cfg0.sup.nIters); end
end
fprintf("\n%s  Estimate for all new trainings: %.1f min -> would end %s (deadline %s)\n",...
    string(datetime('now')),e/60,string(datetime('now') + seconds(e)),string(ctx.deadline))

%% ===== runs =====
Q = cell(1,numel(RUNS));
for k = 1:numel(RUNS)
    r = RUNS(k);
    cfg = cfg0; cfg.sigma = single(r.sg/255); cfg.bottleneck = r.b;
    lab = sprintf("sup SA R b%d s%02d",r.b,r.sg);
    f = fsz(ctx,r.sg,r.b,cfg.sup.nIters);
    Q{k} = unit(ctx,lab,f,"supSA_b" + r.b,@() runsz(r.b,cfg,trainSet,ctx,string(f) + "_ckpt.mat"),cfg.sup.nIters);
end
tRun1 = datetime('now');

%% ===== summary =====
TB = [];
try
    TB = collect(ctx,RUNS,Q,bList);
catch ME
    fprintf("collect FAILED: %s\n",getReport(ME,'extended','hyperlinks','off'))
end
safesummary(@() summarySz(fullfile(ctx.sdir,'supsize_summary.txt'),TB,CHK,EVC,ctx,bList,tRun0,tRun1));
save(fullfile(ctx.cdir,'sbdm_salsun_supsize_results.mat'),'TB','CHK','EVC','cfg0','RUNS','bList')

sk = ctx.ST('skips');
fprintf("\nSkipped / failed / missing units (%d):\n",numel(sk))
fprintf("  %s\n",sk)
fprintf("\n%s  done.\n",string(datetime('now')))

%% ===================== Local functions =====================
function f = fsz(ctx,sg,b,nIt)
f = fullfile(ctx.cdir,sprintf('sbdm_salsun_supsize_s%02d_SAR_b%d_it%d.mat',sg,b,nIt));
end

function c = mkchk(name,hNew,hRef,trc)
c.name = string(name); c.hist1New = hNew(1); c.hist1Ref = hRef(1);
c.relMax2 = max(abs(hNew(2:end) - hRef(2:end))./abs(hRef(2:end)));
c.ok = false; c.tPerIt = trc.tPerIt; c.nParams = trc.nParams;
end

function prchk(c,nIt,why)
fprintf("CHECK %s: new %.10e  ref %.10e  equal %d | its 2-%d (%s) max rel diff %.2e | %.3f s/it  params %d\n",...
    c.name,c.hist1New,c.hist1Ref,c.ok,nIt,why,c.relMax2,c.tPerIt,c.nParams)
if ~c.ok, fprintf("WARNING: check failed: %s\n",c.name), end
end

function EVC = evalcheck(ctx,SB)
% evaluate the cached sigma 25 N / SA b18 models again (same code)
cfg = ctx.cfg0; cfg.sigma = single(25/255); cfg.bottleneck = 18;
imgs = cfg.imgIdx; nI = numel(imgs);
EVC = struct('kind',{},'dPsnr',{},'dSsim',{});
ob = struct('estimator',"net",'leak',false,'bottleneck',18,'shrink',true,...
    'dcShrink',false,'dcInit',"id",'sigma',cfg.sigma);
netsSz = fcn_dcbuildsalsun(cfg.cropSize,cfg.stride,cfg.ovlp,ob);
for kind = ["N" "SA"]
    q = SB.(kind); d = nan(nI,2);
    for i = 1:nI
        [x,v] = fcn_ssloadimage(ctx.dat,imgs(i),cfg,ctx.useGpu);
        if kind == "SA", r = fcn_dcsupeval(q.mdl,x,v,cfg,netsSz); else, r = fcn_dcsupeval(q.mdl,x,v,cfg); end
        d(i,:) = [r.psnr - q.res(i).psnr, r.ssim - q.res(i).ssim];
    end
    EVC(end+1) = struct('kind',kind,'dPsnr',d(:,1),'dSsim',d(:,2)); %#ok<AGROW>
    fprintf("EVAL CHECK sigma 25 %s R b18 (cached model, evaluated again): max |dPSNR| %.2e  max |dSSIM| %.2e\n",...
        kind,max(abs(d(:,1))),max(abs(d(:,2))))
end
end

function q = runsz(b,cfg,trainSet,ctx,ckpt)
o = cfg.sup;
o.sigma = cfg.sigma; o.leak = false; o.dcShrink = false;
o.stride = cfg.stride; o.ovlp = cfg.ovlp; o.bottleneck = b;
o.rngAfterBuild = ctx.rng18;
o.ckptFile = char(ckpt);
t = tic;
[mdl,tr] = fcn_szsuptrain("SA",trainSet,o);
q.mdl = mdl; q.tr = tr; q.variant = "R"; q.kind = "SA"; q.b = b;
nb = max(1,min(100,numel(tr.hist)));
q.lossEnd = mean(tr.hist(end-nb+1:end));
fprintf("    trained: %.1f s (%.3f s/it), loss (mean of last %d its) %.4e\n",tr.tTrain,tr.tPerIt,nb,q.lossEnd)
ob = struct('estimator',"net",'leak',false,'bottleneck',b,...
    'shrink',true,'dcShrink',false,'dcInit',"id",'sigma',cfg.sigma);
netsSz = fcn_dcbuildsalsun(cfg.cropSize,cfg.stride,cfg.ovlp,ob);
q.np = countsa(netsSz.shrink.Learnables);
L = mdl.learn; q.npModel = countsa(L);
fprintf("    params: training net %d | trained model %d (est %d + shrink %d) | evaluation net %d (est %d + shrink %d)\n",...
    tr.nParams,q.npModel.total,q.npModel.est,q.npModel.shr,q.np.total,q.np.est,q.np.shr)
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

function n = countsa(L)
v = cellfun(@numel,L.Value); isS = startsWith(L.Layer,'Lv1_Shrink');
n = struct('total',sum(v),'est',sum(v(~isS)),'shr',sum(v(isS)));
end

function n = countn(mdl)
% N: shared angles + AC shrinkage (a, b)
nTh = sum(cellfun(@numel,mdl.th)); nShr = numel(mdl.ab.a) + numel(mdl.ab.b);
n = struct('total',nTh + nShr,'est',nTh,'shr',nShr);
end

function T = saferun(fn,tag,ctx) %#ok<DEFNU>
try
    T = fn();
catch ME
    addskip(ctx,sprintf("FAILED %s (unexpected error): %s",tag,ME.message))
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

function e = estget(ST,key,nIt)
if nargin < 3, nIt = 1; end
if isKey(ST,char(key)), e = ST(char(key)); return, end
e = 0.65*nIt + 15;   % stage B SA b18: 0.645-0.649 s/it + build/eval (upper bound)
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
fprintf("\n%s  %-28s start (estimate %.0f s)\n",string(datetime('now')),lab,e)
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
fprintf("%s  %-28s mean PSNR %.4f dB (%.1f s)\n",string(datetime('now')),lab,mean([q.res.psnr]),q.tWall);
end

function addskip(ctx,msg)
ctx.ST('skips') = [ctx.ST('skips'); string(msg)];
fprintf("  %s\n",msg)
end

function t = ttot(q)
if isempty(q), t = NaN;
elseif isfield(q,'tWall'), t = q.tWall;
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

function TB = collect(ctx,RUNS,Q,bList)
% one entry per sigma: models N, SA b18 (cached) and SA b3 / b1 (new)
cfg = ctx.cfg0; nI = numel(cfg.imgIdx);
TB = struct('sg',{},'M',{},'names',{},'P',{},'S',{},'np',{},'loss',{},'tTrain',{},'tPerIt',{},'tEval',{},'tUnit',{},'prErr',{});
for sg = cfg.sigmas
    names = ["N" "SA b" + bList];
    M = cell(1,numel(names));
    for kind = ["N" "SA"]
        f = fullfile(ctx.dat,sprintf('sbdm_salsun_dc_sup_s%02d_%sR_it1500.mat',sg,kind));
        if isfile(f)
            S = load(f);
            if kind == "N", M{1} = S.q; else, M{2} = S.q; end
        else
            addskip(ctx,sprintf("MISSING cached model %s",f))
        end
    end
    for k = 1:numel(RUNS)
        if RUNS(k).sg ~= sg, continue, end
        j = find(bList == RUNS(k).b) + 1;
        M{j} = Q{k};
    end
    nM = numel(names);
    T.sg = sg; T.M = M; T.names = names;
    T.P = nan(nI,nM); T.S = nan(nI,nM); T.tEval = nan(nI,nM); T.prErr = nan(nI,nM);
    T.np = repmat(struct('total',NaN,'est',NaN,'shr',NaN,'trainNet',NaN),1,nM);
    T.loss = nan(1,nM); T.tTrain = nan(1,nM); T.tPerIt = nan(1,nM); T.tUnit = nan(1,nM);
    for j = 1:nM
        q = M{j};
        if isempty(q), continue, end
        T.P(:,j) = pickres(q,'psnr',nI); T.S(:,j) = pickres(q,'ssim',nI);
        T.tEval(:,j) = pickres(q,'tEval',nI); T.prErr(:,j) = pickres(q,'prErrX',nI);
        if j == 1, n = countn(q.mdl); else, n = countsa(q.mdl.learn); end
        n.trainNet = q.tr.nParams; T.np(j) = n;
        T.loss(j) = q.lossEnd; T.tTrain(j) = q.tr.tTrain; T.tPerIt(j) = q.tr.tPerIt; T.tUnit(j) = ttot(q);
    end
    TB(end+1) = T; %#ok<AGROW>
end
end

%% ----- summary -----
function summarySz(file,TB,CHK,EVC,ctx,bList,tRun0,tRun1)
fid = fopen(file,'w'); cleaner = onCleanup(@() fclose(fid));
cfg = ctx.cfg0; o = cfg.sup;
out(fid,"Supervised SA-LSUN with a smaller angle estimator (BottleneckSize b), variant R (no DC leakage, no DC shrinkage).\n");
out(fid,"Written %s. Run %s -> %s (%.1f min).\n",string(datetime('now')),string(tRun0),string(tRun1),minutes(tRun1 - tRun0));
out(fid,"Training as stage B / run A: %d patches %dx%d of kodim%02d-%02d (rng(1)), batch %d, fresh noise each it, fixed sigma, MSE, %d Adam its, cosine decay.\n",...
    o.nCrops*numel(o.trainIdx),o.patch(1),o.patch(2),o.trainIdx(1),o.trainIdx(end),o.mb,o.nIters);
out(fid,"  SA: estimator lr %g -> %g, shrinkage lr %g -> %g. N: lr %g -> %g. Init: zero estimator output / zero angles, AC kappa0 0.9 tau0 2.\n",...
    o.lrEst(1),o.lrEst(2),o.lrShr(1),o.lrShr(2),o.lrN(1),o.lrN(2));
out(fid,"  New SA b = 3 / 1: fcn_szsuptrain (= fcn_dcsuptrain + CPU rng state after the build set to that of the b = 18 build),\n");
out(fid,"  so the crops and the noise of every iteration are the same as for the stage B b = 18 training.\n");
out(fid,"  N and SA b = 18: cached models (stage B: sigma 25, 10; run A: sigma 5, 2), not retrained.\n");
out(fid,"Evaluation: 256x256 centre crops, kodim09/15/19/23, noise rng(image index). mean(test) = kodim15/19/23. No choice is made.\n");
out(fid,"\nChecks before the runs (sigma 25, short training):\n");
for c = CHK
    out(fid,"  %s: new %.10e  ref %.10e  equal %d  (its 2+: max rel diff %.2e)  %.3f s/it  params %d\n",...
        c.name,c.hist1New,c.hist1Ref,c.ok,c.relMax2,c.tPerIt,c.nParams);
end
out(fid,"  (check 3 compares the whole history; its max rel diff is over all its.)\n");
for e = EVC
    out(fid,"  Cached sigma 25 %s R b18 evaluated again: max |dPSNR| %.2e  max |dSSIM| %.2e\n",e.kind,max(abs(e.dPsnr)),max(abs(e.dSsim)));
end
if isempty(TB), out(fid,"\n(no results)\n"); return, end
imgs = cfg.imgIdx(:); rows = compose("kodim%02d",imgs);
iTest = find(imgs ~= cfg.selImg);
rowsM = [rows; "mean(all)"; "mean(test)"];
addm = @(M) [M; mean(M,1,'omitnan'); mean(M(iTest,:),1,'omitnan')];
WT = nan(numel(TB),numel(bList)); WA = WT; DP = WT; DS = WT;
for k = 1:numel(TB)
    T = TB(k); names = T.names; nM = numel(names);
    out(fid,"\n==================== Supervised, sigma = %d/255 ====================\n",T.sg);
    out(fid,"\nParameters (exact counts of the learnables) and training\n");
    for j = 1:nM
        out(fid,"  %-7s params %6s (angles/estimator %6s + shrinkage %3s; training net %6s)  loss %.4e  train %.1f s  %.3f s/it  unit %.1f s\n",...
            names(j),nstr(T.np(j).total),nstr(T.np(j).est),nstr(T.np(j).shr),nstr(T.np(j).trainNet),...
            T.loss(j),T.tTrain(j),T.tPerIt(j),T.tUnit(j));
    end
    printtable(fid,"PSNR [dB]",rowsM,names,addm(T.P),"%.4f")
    printtable(fid,"SSIM",rowsM,names,addm(T.S),"%.4f")
    dP = T.P(:,2:end) - T.P(:,1); dS = T.S(:,2:end) - T.S(:,1);
    printtable(fid,"SA - N, PSNR [dB]",rowsM,names(2:end) + "-N",addm(dP),"%.4f")
    printtable(fid,"SA - N, SSIM",rowsM,names(2:end) + "-N",addm(dS),"%.4f")
    wt = sum(dP(iTest,:) > 0,1); wa = sum(dP > 0,1); ok = ~all(isnan(dP),1);
    wt(~ok) = NaN; wa(~ok) = NaN;
    ws = sum(dS(iTest,:) > 0,1); ws(~ok) = NaN;
    printtable(fid,sprintf("Number of images where SA beats N (PSNR): test (of %d), all (of %d); SSIM test",numel(iTest),numel(imgs)),...
        ["PSNR test"; "PSNR all"; "SSIM test"],names(2:end),[wt; wa; ws],"%d")
    printtable(fid,"Time [s]: evaluation per image",rowsM,names,addm(T.tEval),"%.1f")
    printtable(fid,"PR error ||DEx - x||/||x||, last row = max",[rows; "max"],names,[T.prErr; max(T.prErr,[],1)],"%.2e")
    WT(k,:) = wt; WA(k,:) = wa;
    am = addm(dP); DP(k,:) = am(end,:); am = addm(dS); DS(k,:) = am(end,:);
end
sgRows = compose("sigma %d",[TB.sg].');
cols = "SA b" + bList + "-N";
out(fid,"\n==================== Over sigma ====================\n");
printtable(fid,"SA - N, PSNR [dB], mean(test)",sgRows,cols,DP,"%.4f")
printtable(fid,"SA - N, SSIM, mean(test)",sgRows,cols,DS,"%.4f")
printtable(fid,sprintf("Test images (of %d) where SA beats N, PSNR",numel(iTest)),sgRows,cols,WT,"%d")
printtable(fid,sprintf("All images (of %d) where SA beats N, PSNR",numel(imgs)),sgRows,cols,WA,"%d")
sk = ctx.ST('skips');
out(fid,"\n---- Skipped / failed / missing units (%d) ----\n",numel(sk));
for k = 1:numel(sk), out(fid,"  %s\n",sk(k)); end
out(fid,"\n");
end

function s = nstr(x)
if isnan(x), s = "-"; else, s = sprintf("%d",x); end
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
