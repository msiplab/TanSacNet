%% main_sbdm_salsun_overnight.m
% Overnight driver (2026-11-06). Observation only, kodim09/15/19/23,
% same noise seeds as before. All stages in one process, in order:
%
% Stage 1: SAP-hard with a matched step-1 threshold on the pilot p.
%   Variants: a  = threshold on p such that p keeps the same fraction of
%                  AC coefficients as N-hard keeps on v (per image)
%             b2, b3 = tau*sigma_p, sigma_p = sqrt(SURE of N-hard), tau = 2, 3
%   b = 18, step-1 its 150 and 500, angles from p frozen, step 2 by SURE
%   on v with 8 fixed probes (fcn_ssphardrunimage, unchanged).
%   sigma = 25/255: pilot = N-hard tau=3 (cache); refs N-hard, zonal N, W (caches)
%   sigma = 10/255: N-hard tau = 2, 3 first; pilot = N-hard selected on
%                   kodim09; refs zonal N and W (sigma10 caches)
% Stage 2: sigma = 5/255: W, zonal N K=4, N-hard (tau 2, 3), SAP-hard as
%   at sigma = 10.
% Stage 3: full-size images, sigma = 10/255: W, N-hard tau=3, and the best
%   SAP-hard setting of stage 1 (sigma 10) chosen on kodim09, with the
%   full-size N-hard tau=3 as the pilot. kodim09 first; every further image
%   runs only if the whole image fits before the deadline.
%
% Fair selection on kodim09 for every comparison, reported on kodim15/19/23.
%
% Time budget: before every (image, setting) unit, the run time is
% estimated from the measured speeds (latest measured unit of the same
% kind; priors from earlier logs). A unit that would end after the
% deadline is not started; it is listed as SKIPPED in the log and in the
% summary. A unit that fails is listed as FAILED; the run continues.
%
% One cache file per (image, setting) in data/sbdm_salsun_on_*.mat
% (resumable). After each stage: logs/overnight_stage<N>_summary.txt.
%   nohup matlab -licmode onlinelicensing -batch "main_sbdm_salsun_overnight" \
%       > ~/Documents/SBDMwLSUN/logs/overnight.log 2>&1 &
% Quick test of all code paths (3 its, 2 images, separate folder):
%   matlab -licmode onlinelicensing -batch "onTest = true; onTestDir = '<dir>'; main_sbdm_salsun_overnight"
% Options: onDeadline = datetime(...) changes the deadline.

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
cfg0.tauList = [2 3];        % statistics thresholds of N / N-hard (tau*sigma)
cfg0.itList = [150 500];     % SAP-hard step-1 iterations
dat = fullfile(here,'..','..','..','data');
ctx.dat = dat;                                        % images, old caches
ctx.cdir = dat;                                       % new caches
ctx.sdir = fullfile(here,'..','..','..','..','logs'); % summaries
ctx.deadline = datetime(2026,11,6,7,30,0);
ctx.margin = 1.05;           % safety factor on the time estimates
isTest = exist('onTest','var') && onTest;
if isTest
    cfg0.imgIdx = [9 15]; cfg0.nIt1N = 3; cfg0.nIt2 = 3; cfg0.itList = [3 4]; cfg0.nProbeFresh = 2;
    ctx.cdir = onTestDir; ctx.sdir = onTestDir;
    if ~isfolder(onTestDir), mkdir(onTestDir); end
end
if exist('onDeadline','var'), ctx.deadline = onDeadline; end
ctx.useGpu = canUseGPU;
ctx.cfg0 = cfg0;
ctx.ST = containers.Map();            % handle: time estimates + skip list
ctx.ST('skips') = strings(0,1);
fprintf("%s  start. GPU: %d. Deadline %s. Test mode: %d\n",string(datetime('now')),ctx.useGpu,string(ctx.deadline),isTest)
disp(cfg0)

vars = ["a" "b2" "b3"];
[vv,ii] = ndgrid(vars,cfg0.itList);
sapAll = struct('var',num2cell(vv(:).'),'it',num2cell(ii(:).'));
for s = 1:numel(sapAll), sapAll(s).name = sapAll(s).var + "/it" + sapAll(s).it; end
cf = @(name) fullfile(ctx.cdir,name);
df = @(name) fullfile(dat,name);

%% ===== Stage 1 =====
sp = struct();
sp.tag = "S1 s25"; sp.title = "Stage 1, sigma = 25/255"; sp.sigma = 25/255; sp.full = false;
sp.imgs = cfg0.imgIdx; sp.wholeImage = false;
sp.W  = struct('run',false,'file',@(i) df(sprintf('sbdm_salsun_%02d_W.mat',i)));
sp.N  = struct('run',false,'file',@(i) df(sprintf('sbdm_salsun_%02d_N4.mat',i)));
sp.Nh = struct('run',false,'taus',[2 3],'file',@(i,t) df(sprintf('sbdm_salsun_hard_%02d_N4_t%d.mat',i,t)));
sp.pilot = 3;
sp.sap = sapAll;
sp.sapFile = @(i,pt,st) cf(sprintf('sbdm_salsun_on_s25_%02d_SAPh_p%d_%s_it%d.mat',i,pt,st.var,st.it));
sp.notes = ["W, zonal N K=4, N-hard tau=2,3: caches of the earlier runs." ...
            "Pilot p = N-hard tau=3 (fixed, as requested)."];
T25 = saferun(sp,ctx);

sp.tag = "S1 s10"; sp.title = "Stage 1, sigma = 10/255"; sp.sigma = 10/255;
sp.W  = struct('run',false,'file',@(i) df(sprintf('sbdm_salsun_s10_%02d_W.mat',i)));
sp.N  = struct('run',false,'file',@(i) df(sprintf('sbdm_salsun_s10_%02d_N4.mat',i)));
sp.Nh = struct('run',true,'taus',[2 3],'file',@(i,t) cf(sprintf('sbdm_salsun_on_s10_%02d_Nh_t%d.mat',i,t)));
sp.pilot = NaN;   % select on kodim09
sp.sapFile = @(i,pt,st) cf(sprintf('sbdm_salsun_on_s10_%02d_SAPh_p%d_%s_it%d.mat',i,pt,st.var,st.it));
sp.notes = ["W, zonal N K=4: sigma = 10/255 caches of the last run (sbdm_salsun_s10_*)." ...
            "N-hard tau=2,3 run here; pilot p = N-hard with tau selected on kodim09."];
T10 = saferun(sp,ctx);
safesummary(fullfile(ctx.sdir,'overnight_stage1_summary.txt'),{T25,T10},ctx,...
    "Stage 1: SAP-hard with a matched threshold on the pilot","S1");
save(cf('sbdm_salsun_on_stage1_results.mat'),'T25','T10','cfg0')

%% ===== Stage 2: sigma = 5/255 =====
sp.tag = "S2 s05"; sp.title = "Stage 2, sigma = 5/255"; sp.sigma = 5/255;
sp.W  = struct('run',true,'file',@(i) cf(sprintf('sbdm_salsun_on_s05_%02d_W.mat',i)));
sp.N  = struct('run',true,'file',@(i) cf(sprintf('sbdm_salsun_on_s05_%02d_N4.mat',i)));
sp.Nh = struct('run',true,'taus',[2 3],'file',@(i,t) cf(sprintf('sbdm_salsun_on_s05_%02d_Nh_t%d.mat',i,t)));
sp.pilot = NaN;
sp.sapFile = @(i,pt,st) cf(sprintf('sbdm_salsun_on_s05_%02d_SAPh_p%d_%s_it%d.mat',i,pt,st.var,st.it));
sp.notes = ["W, zonal N K=4 (as before) and N-hard tau=2,3 run here." ...
            "Pilot p = N-hard with tau selected on kodim09."];
T05 = saferun(sp,ctx);
safesummary(fullfile(ctx.sdir,'overnight_stage2_summary.txt'),{T05},ctx,...
    "Stage 2: sigma = 5/255","S2");
save(cf('sbdm_salsun_on_stage2_results.mat'),'T05','cfg0')

%% ===== Stage 3: full size, sigma = 10/255 =====
sp.tag = "S3 full"; sp.title = "Stage 3, full-size images, sigma = 10/255"; sp.sigma = 10/255; sp.full = true;
sp.wholeImage = true;
sp.W  = struct('run',true,'file',@(i) cf(sprintf('sbdm_salsun_on_full_s10_%02d_W.mat',i)));
sp.N  = struct('run',false,'file',[]);
sp.Nh = struct('run',true,'taus',3,'file',@(i,t) cf(sprintf('sbdm_salsun_on_full_s10_%02d_Nh_t%d.mat',i,t)));
sp.pilot = 3;
if ~isempty(T10) && ~isnan(T10.selSa)
    sp.sap = T10.sap(T10.selSa);
    sp.notes = "SAP-hard setting " + sp.sap.name + " = best on kodim09 in stage 1 (sigma 10/255). Pilot = full-size N-hard tau=3. No zonal N at full size.";
else
    sp.sap = sapAll([]);
    addskip(ctx,"SKIPPED S3 full SAP-hard: no stage-1 (sigma 10) selection available")
    sp.notes = "No SAP-hard: the stage-1 selection (sigma 10) is not available.";
end
sp.sapFile = @(i,pt,st) cf(sprintf('sbdm_salsun_on_full_s10_%02d_SAPh_p%d_%s_it%d.mat',i,pt,st.var,st.it));
T3 = saferun(sp,ctx);
safesummary(fullfile(ctx.sdir,'overnight_stage3_summary.txt'),{T3},ctx,...
    "Stage 3: full-size images, sigma = 10/255","S3");
save(cf('sbdm_salsun_on_stage3_results.mat'),'T3','cfg0')

sk = ctx.ST('skips');
fprintf("\nSkipped / failed units (%d):\n",numel(sk))
fprintf("  %s\n",sk)
fprintf("\n%s  done.\n",string(datetime('now')))

%% ===================== Local functions =====================
function T = saferun(sp,ctx)
% runsub, but an unexpected error ends only this stage
try
    T = runsub(sp,ctx);
catch ME
    addskip(ctx,sprintf("FAILED %s (whole stage, unexpected error): %s",sp.tag,ME.message))
    fprintf("%s\n",getReport(ME,'extended','hyperlinks','off'))
    T = [];
end
end

function safesummary(file,Ts,ctx,ttl,stageKey)
try
    writesummary(file,Ts(~cellfun(@isempty,Ts)),ctx,ttl,stageKey);
catch ME
    fprintf("Summary %s FAILED: %s\n",file,ME.message)
    fprintf("%s\n",getReport(ME,'extended','hyperlinks','off'))
end
end

function T = runsub(sp,ctx)
cfg = ctx.cfg0; cfg.sigma = single(sp.sigma);
imgs = sp.imgs; nI = numel(imgs); nT = numel(sp.Nh.taus); nS = numel(sp.sap);
T = sp; T.cfg = cfg; T.nhTaus = sp.Nh.taus;
T.W = cell(nI,1); T.N = cell(nI,1); T.Nh = cell(nI,nT); T.Sa = cell(nI,nS);
T.pilotTau = sp.pilot; T.selNh = NaN; T.selSa = NaN; T.tStart = datetime('now');
pre = ""; if sp.full, pre = "full:"; end
fprintf("\n%s  ===== %s =====\n",string(datetime('now')),sp.title)

% stage estimate (units without a cache)
e = 0;
for i = 1:nI, e = e + imgest(sp,ctx,imgs(i),pre); end
T.estSec = e;
fprintf("Estimate for this stage (units not cached): %.1f min -> would end %s (deadline %s)\n",...
    e/60,string(datetime('now') + seconds(e)),string(ctx.deadline))

iSel = find(imgs == cfg.selImg,1);
for i = 1:nI
    idx = imgs(i); seed = cfg.seedBase + idx;
    lab = sp.tag + sprintf(" kodim%02d",idx);
    if sp.wholeImage && i > 1
        e = imgest(sp,ctx,idx,pre)*ctx.margin;
        tEnd = datetime('now') + seconds(e);
        if tEnd > ctx.deadline
            addskip(ctx,sprintf("SKIPPED %s (whole image): estimate %.0f s, would end %s > deadline %s",...
                lab,e,string(tEnd),string(ctx.deadline)))
            continue
        end
    end
    if sp.full
        [x,v] = fcn_ssloadfull(ctx.dat,idx,cfg,ctx.useGpu);
    else
        [x,v] = fcn_ssloadimage(ctx.dat,idx,cfg,ctx.useGpu);
    end
    tImg = tic;
    fprintf("\n%s  %s  size %dx%d  noisy %.2f dB\n",string(datetime('now')),lab,size(x,1),size(x,2),psnr(gather(v),gather(x)))
    % W
    if sp.W.run
        T.W{i} = unit(ctx,lab + " W",sp.W.file(idx),pre + "W",@() runW(x,v,cfg));
    else
        T.W{i} = loadopt(ctx,lab + " W",sp.W.file(idx));
    end
    % zonal N
    if sp.N.run
        T.N{i} = unit(ctx,lab + " N K=4",sp.N.file(idx),pre + "N",@() fcn_ssnrunimage(x,v,cfg,seed,cfg.K,"zonal",NaN));
    elseif ~isempty(sp.N.file)
        T.N{i} = loadopt(ctx,lab + " N K=4",sp.N.file(idx));
    end
    % N-hard
    for j = 1:nT
        tau = sp.Nh.taus(j);
        if sp.Nh.run
            T.Nh{i,j} = unit(ctx,lab + " Nh/t" + tau,sp.Nh.file(idx,tau),pre + "Nh",...
                @() fcn_ssnrunimage(x,v,cfg,seed,cfg.K,"hard",tau));
        else
            T.Nh{i,j} = loadopt(ctx,lab + " Nh/t" + tau,sp.Nh.file(idx,tau));
        end
    end
    % pilot selection on kodim09
    if isnan(sp.pilot) && i == iSel
        P = cellfun(@(q) getf(q,'psnr'),T.Nh(i,:));
        if any(~isnan(P))
            [~,T.selNh] = max(P); T.pilotTau = sp.Nh.taus(T.selNh);
            fprintf("  Pilot selection on kodim%02d (PSNR):%s -> N-hard tau=%d\n",idx,...
                sprintf(" Nh/t%d %.2f",[sp.Nh.taus; P]),T.pilotTau)
        end
    elseif ~isnan(sp.pilot)
        T.selNh = find(sp.Nh.taus == sp.pilot,1);
    end
    if nS == 0, fprintf("  kodim%02d done (%.1f s)\n",idx,toc(tImg)); continue, end
    if isnan(T.pilotTau)
        addskip(ctx,sprintf("SKIPPED %s SAP-hard (all settings): no pilot selection (kodim09 N-hard missing)",lab))
        continue
    end
    qP = T.Nh{i,sp.Nh.taus == T.pilotTau};
    if isempty(qP)
        addskip(ctx,sprintf("SKIPPED %s SAP-hard (all settings): pilot N-hard tau=%d missing",lab,T.pilotTau))
        continue
    end
    for s = 1:nS
        st = sp.sap(s);
        T.Sa{i,s} = unit(ctx,lab + " SAPh " + st.name,sp.sapFile(idx,T.pilotTau,st),pre + "SAPh" + st.it,...
            @() runsap(x,v,cfg,seed,qP,T.pilotTau,st));
    end
    fprintf("  %s done (%.1f s for this image)\n",lab,toc(tImg))
    if sp.full && i == 1
        fprintf("  Full-size kodim%02d time: %.1f s. Estimate per further image: %.1f s.\n",idx,toc(tImg),...
            imgest(sp,ctx,-1,pre))
    end
end
if ~isempty(iSel) && nS > 0
    P = cellfun(@(q) getf(q,'psnr'),T.Sa(iSel,:));
    if any(~isnan(P)), [~,T.selSa] = max(P); end
end
T.tEnd = datetime('now');
fprintf("%s  %s finished (%.1f min)\n",string(T.tEnd),sp.title,minutes(T.tEnd - T.tStart))
end

function e = imgest(sp,ctx,idx,pre)
% estimated time [s] of the units of one image that have no cache (idx < 0: no cache check)
has = @(f) idx >= 0 && isfile(f);
e = 0;
if sp.W.run && ~has(sp.W.file(max(idx,0))), e = e + estget(ctx.ST,pre + "W"); end
if sp.N.run && ~has(sp.N.file(max(idx,0))), e = e + estget(ctx.ST,pre + "N"); end
if sp.Nh.run
    for t = sp.Nh.taus
        if ~has(sp.Nh.file(max(idx,0),t)), e = e + estget(ctx.ST,pre + "Nh"); end
    end
end
for s = 1:numel(sp.sap)
    st = sp.sap(s);
    if isnan(sp.pilot) || ~has(sp.sapFile(max(idx,0),sp.pilot,st))
        e = e + estget(ctx.ST,pre + "SAPh" + st.it);
    end
end
end

function e = estget(ST,key)
if isKey(ST,char(key)), e = ST(char(key)); return, end
k = erase(key,"full:");
if k == "W", e = 2;
elseif k == "N", e = 178;            % salsun_sigma10_hard.log, N total
elseif k == "Nh", e = 182;           % Nh total
elseif startsWith(k,"SAPh")
    if str2double(extractAfter(k,4)) <= 150, e = 69; else, e = 173; end
else, e = 200;
end
if startsWith(key,"full:"), e = 6*e; end   % 6x pixels (prior only)
end

function q = unit(ctx,lab,f,key,fn)
if isfile(f)
    S = load(f); q = S.q;
    fprintf("  %-28s cached (%s)\n",lab,f)
    return
end
e = estget(ctx.ST,key)*ctx.margin;
tEnd = datetime('now') + seconds(e);
if tEnd > ctx.deadline
    addskip(ctx,sprintf("SKIPPED %s: estimate %.0f s, would end %s > deadline %s",...
        lab,e,string(tEnd),string(ctx.deadline)))
    q = [];
    return
end
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
ctx.ST(char(key)) = dt;            % latest measured speed of this kind
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

function q = runW(x,v,cfg)
t = tic;
z = single(wiener2(gather(double(v)),[5 5],double(cfg.sigma)^2));
xg = gather(x);
q = struct('psnr',psnr(z,xg),'ssim',ssim(z,xg),'mse',double(mean((z - xg).^2,'all')),'img',z);
q.time = toc(t); q.tTotal = q.time;
end

function q = runsap(x,v,cfg,seed,qP,pTau,st)
[thr,info] = fcn_sssapthr(v,cfg,qP,st.var);
tau = thr/double(cfg.sigma);
c = cfg; c.tauList = [tau pTau];          % statistics: own threshold, pilot's threshold
t = tic;
q = fcn_ssphardrunimage(x,v,cfg.K,c,seed,qP,cfg.bottleneck,st.it,"hard",tau);
q.tTotal = toc(t);
q.variant = st.var; q.thr = thr; q.thrOverSigma = tau; q.pilotTau = pTau; q.thrInfo = info;
end

function printq(lab,q)
s = sprintf("  %-28s %.2f dB  SSIM %.4f",lab,q.psnr,q.ssim);
if isfield(q,'variant')
    s = s + sprintf("  thr/sigma %.3f  kept frac v %.4f p %.4f x %.4f  1-SURE/MSE %.4f",...
        q.thrOverSigma,q.statV.hardFrac(1),q.statP.hardFrac(1),q.statX.hardFrac(1),1 - q.sure/q.mse);
elseif isfield(q,'statV') && ~isnan(q.tau)
    j = ownidx(q);
    s = s + sprintf("  kept frac v %.4f x %.4f  1-SURE/MSE %.4f",q.statV.hardFrac(j),q.statX.hardFrac(j),1 - q.sure/q.mse);
elseif isfield(q,'sure')
    s = s + sprintf("  1-SURE/MSE %.4f",1 - q.sure/q.mse);
end
s = s + sprintf("  (%.1f s)",q.tTotal);
fprintf("%s\n",s)
end

function j = ownidx(q)
% index of the own threshold in the statistics
if isfield(q,'variant'), j = 1; else, j = find(abs([2 3] - q.tau) < 1e-9,1); end
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
if isfield(q,'tTotal'), t = q.tTotal;
elseif isfield(q,'time'), t = q.time;
else, t = q.tBuild + q.t1 + q.t2 + q.tEval;
end
end

%% ----- summary -----
function writesummary(file,Ts,ctx,ttl,stageKey)
fid = fopen(file,'w');
cleaner = onCleanup(@() fclose(fid));
out(fid,"%s\n",ttl);
out(fid,"Written %s. Deadline %s.\n",string(datetime('now')),string(ctx.deadline));
out(fid,"Observation only, Kodak kodim09/15/19/23, grayscale, same noise seeds as before (rng(image index)).\n");
out(fid,"Fair selection: by PSNR on kodim09 only; reported on the other images (test).\n");
out(fid,"mean(all) / mean(test) omit missing values ('-' = not run or skipped; see the list at the end).\n");
for k = 1:numel(Ts), sumone(fid,Ts{k}); end
sk = ctx.ST('skips');
sk = sk(contains(sk,stageKey + " "));
out(fid,"\n---- Skipped / failed / missing units in this stage (%d) ----\n",numel(sk));
for k = 1:numel(sk), out(fid,"  %s\n",sk(k)); end
out(fid,"\n");
end

function sumone(fid,T)
cfg = T.cfg; imgs = T.imgs(:); nI = numel(imgs);
rows = compose("kodim%02d",imgs);
iSel = find(imgs == cfg.selImg,1); iTest = find(imgs ~= cfg.selImg);
rowsM = [rows; "mean(all)"; "mean(test)"];
addm = @(M) [M; mean(M,1,'omitnan'); mean(M(iTest,:),1,'omitnan')];
nhN = "Nh/t" + string(T.nhTaus); saN = [T.sap.name]; nS = numel(T.sap);
if isempty(saN), saN = strings(1,0); end
G = @getm;
pW = G(T.W,@(q) q.psnr); sW = G(T.W,@(q) q.ssim);
pN = G(T.N,@(q) q.psnr); sN = G(T.N,@(q) q.ssim);
pNh = G(T.Nh,@(q) q.psnr); sNh = G(T.Nh,@(q) q.ssim);
pSa = G(T.Sa,@(q) q.psnr); sSa = G(T.Sa,@(q) q.ssim);
if isempty(T.N), pN = nan(nI,1); sN = pN; end
out(fid,"\n==================== %s ====================\n",T.title);
out(fid,"Images: %s. Size: %s. sigma = %.4f (= %g/255).\n",strjoin(rows.',", "),...
    string(ifelse(T.full,"full size","256x256 centre crops")),T.sigma,round(T.sigma*255));
out(fid,"4x4 blocks, ovlp 3x3, S = 1. N-hard: shared angles, hard step 1 on v (|c| > tau*sigma, %d its), step 2 exact SURE.\n",cfg.nIt1N);
out(fid,"SAP-hard: b = %d, hard step 1 on p = N-hard(v), angles from p frozen, step 2 by SURE on v, %d fixed probes (check: %d fresh).\n",...
    cfg.bottleneck,cfg.nProbePilot,cfg.nProbeFresh);
out(fid,"  variant a: threshold on p so that E_Nh p keeps the same fraction of AC coefficients as N-hard keeps on v (fixed during step 1).\n");
out(fid,"  variant bT: threshold T*sigma_p, sigma_p = sqrt(SURE of N-hard).\n");
for k = 1:numel(T.notes), out(fid,"%s\n",T.notes(k)); end
out(fid,"Run: %s -> %s (estimate before the stage %.1f min).\n",string(T.tStart),string(T.tEnd),T.estSec/60);
jP = T.selNh;
if ~isnan(jP)
    if isnan(T.pilot)
        out(fid,"\nPilot selection on kodim09 (PSNR [dB]):%s -> pilot N-hard tau=%d\n",...
            sprintf(" %s %s",[nhN; compose("%.2f",pNh(iSel,:))]),T.pilotTau);
    else
        out(fid,"\nPilot: N-hard tau=%g (fixed).\n",T.pilotTau);
    end
    pP = pNh(:,jP); sP = sNh(:,jP);
else
    out(fid,"\nPilot: none available.\n");
    pP = nan(nI,1); sP = pP;
end
if ~isnan(T.selSa)
    out(fid,"SAP-hard selection on kodim09 (PSNR [dB]):%s -> %s\n",...
        sprintf(" %s %s",[saN; compose("%.2f",pSa(iSel,:))]),saN(T.selSa));
end
% test table
if ~isempty(iTest)
    if ~isnan(T.selSa), pS = pSa(:,T.selSa); sS = sSa(:,T.selSa); selName = saN(T.selSa);
    else, pS = nan(nI,1); sS = pS; selName = "-"; end
    cols = ["W" "N K=4" "Nh pilot" "SAPh sel" "SAPh-Nh" "Nh-W" "Nh-N" "SAPh-N" "SAPh-W"];
    it_ = iTest;
    M = [pW(it_) pN(it_) pP(it_) pS(it_) pS(it_)-pP(it_) pP(it_)-pW(it_) pP(it_)-pN(it_) pS(it_)-pN(it_) pS(it_)-pW(it_)];
    printtable(fid,sprintf("Test images, PSNR [dB]. Nh pilot = N-hard tau=%g, SAPh sel = %s",T.pilotTau,selName),...
        [rows(it_); "mean"],cols,[M; mean(M,1,'omitnan')],"%.4f")
    M = [sW(it_) sN(it_) sP(it_) sS(it_) sS(it_)-sP(it_) sP(it_)-sW(it_) sP(it_)-sN(it_) sS(it_)-sN(it_) sS(it_)-sW(it_)];
    printtable(fid,"Test images, SSIM",[rows(it_); "mean"],cols,[M; mean(M,1,'omitnan')],"%.4f")
end
out(fid,"\nAll settings, all images (information; choosing from these is not a fair selection):\n");
printtable(fid,"PSNR [dB]",rowsM,["W" "N K=4" nhN "SAPh " + saN],addm([pW pN pNh pSa]),"%.2f")
printtable(fid,"SSIM",rowsM,["W" "N K=4" nhN "SAPh " + saN],addm([sW sN sNh sSa]),"%.4f")
printtable(fid,"PSNR differences [dB]: SAP-hard - N-hard (pilot), N-hard - W, N-hard - zonal N",rowsM,...
    [saN + "-Nh" nhN + "-W" nhN + "-N"],addm([pSa - pP pNh - pW pNh - pN]),"%.2f")
printtable(fid,"SSIM differences: SAP-hard - N-hard (pilot), N-hard - W, N-hard - zonal N",rowsM,...
    [saN + "-Nh" nhN + "-W" nhN + "-N"],addm([sSa - sP sNh - sW sNh - sN]),"%.4f")
if nS > 0
    printtable(fid,"SAP-hard step-1 threshold / sigma",rowsM,saN,addm(G(T.Sa,@(q) q.thrOverSigma)),"%.4f")
    printtable(fid,"Variant a: target fraction f (= N-hard kept fraction on v); sigma_p*255 (sigma_p = sqrt(SURE of N-hard), used by b2/b3)",rowsM,...
        [saN + " f" saN + " sp*255"],addm([G(T.Sa,@(q) q.thrInfo.f) G(T.Sa,@(q) 255*q.thrInfo.sigmaP)]),"%.4f")
end
own = @(fld,what) G([T.Nh T.Sa],@(q) q.(fld).(what)(ownidx(q)));
cOwn = [nhN "SAPh " + saN];
out(fid,"\nKept statistics with each setting's own step-1 threshold (N-hard: tau*sigma; SAP-hard: its threshold on p). x = evaluation only.\n");
printtable(fid,"Kept fraction of AC coefficients on v",rowsM,cOwn,addm(own('statV','hardFrac')),"%.4f")
if nS > 0, printtable(fid,"Kept fraction on p",rowsM,"SAPh " + saN,addm(G(T.Sa,@(q) q.statP.hardFrac(1))),"%.4f"), end
printtable(fid,"Kept fraction on x",rowsM,cOwn,addm(own('statX','hardFrac')),"%.4f")
printtable(fid,"Kept AC energy on v (energy above threshold / AC energy)",rowsM,cOwn,addm(own('statV','hardAc')),"%.4f")
if nS > 0, printtable(fid,"Kept AC energy on p",rowsM,"SAPh " + saN,addm(G(T.Sa,@(q) q.statP.hardAc(1))),"%.4f"), end
printtable(fid,"Kept AC energy on x",rowsM,cOwn,addm(own('statX','hardAc')),"%.4f")
if nS > 0
    out(fid,"\nSame threshold as the pilot N-hard on v (tau_N*sigma), for a direct comparison with N-hard:\n");
    cP = ["Nh pilot" "SAPh " + saN];
    fP = @(fld,what) [G(T.Nh(:,max(jP,1)),@(q) q.(fld).(what)(ownidx(q))) G(T.Sa,@(q) q.(fld).(what)(2))];
    if isnan(jP), fP = @(fld,what) nan(nI,1+nS); end
    printtable(fid,"Kept fraction on v at tau_N*sigma",rowsM,cP,addm(fP('statV','hardFrac')),"%.4f")
    printtable(fid,"Kept fraction on x at tau_N*sigma",rowsM,cP,addm(fP('statX','hardFrac')),"%.4f")
    printtable(fid,"Kept AC energy on v at tau_N*sigma",rowsM,cP,addm(fP('statV','hardAc')),"%.4f")
    printtable(fid,"Kept AC energy on x at tau_N*sigma",rowsM,cP,addm(fP('statX','hardAc')),"%.4f")
end
printtable(fid,"1 - SURE/MSE (> 0: SURE too low). N, N-hard: exact; SAP-hard: 8 fixed probes",rowsM,...
    ["N K=4" nhN "SAPh " + saN],addm(1 - [G(T.N,@(q) q.sure/q.mse) G(T.Nh,@(q) q.sure/q.mse) G(T.Sa,@(q) q.sure/q.mse)]),"%.4f")
if nS > 0
    printtable(fid,"1 - SURE/MSE, SAP-hard: 32 fresh probes",rowsM,"SAPh " + saN,addm(1 - G(T.Sa,@(q) q.sureFresh/q.mse)),"%.4f")
end
M = [G(T.Nh,@(q) q.prErrX) G(T.Sa,@(q) q.prErrX)];
printtable(fid,"PR error ||DEx - x||/||x||, max row = max",[rows; "max"],[nhN "SAPh " + saN],[M; max(M,[],1)],"%.2e")
printtable(fid,"Time per setting [s] (whole unit)",rowsM,["W" "N K=4" nhN "SAPh " + saN],...
    addm([G(T.W,@ttot) G(T.N,@ttot) G(T.Nh,@ttot) G(T.Sa,@ttot)]),"%.1f")
end

function out(fid,fmt,varargin)
fprintf(fid,fmt,varargin{:});
fprintf(1,fmt,varargin{:});
end

function y = ifelse(c,a,b)
if c, y = a; else, y = b; end
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
