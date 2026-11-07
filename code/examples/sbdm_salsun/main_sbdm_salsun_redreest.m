%% main_sbdm_salsun_redreest.m
% RED with SA b18 where the angles are re-estimated from the current
% iterate every K its (and when the sigma model switches), constant in
% between. K = inf = Part 3 of main_sbdm_salsun_tnrdguar.m (angles from the
% observation v, frozen), cached. (2026-11-07)
%
% Everything else exactly as Part 3: 9x9 Gaussian blur std 2 (circular),
% noise 5/255 rng(1000 + image), 256x256 centre crops kodim09/15/19/23,
% z(0) = v, eps = 0, 300 its, sigma(t) 25 -> 2 /255 (log), bank of the
% fixed-sigma SA b18 models (nearest sigma), eta, beta in {0.5,1,2}, with
% and without clipping. Runner: fcn_tgredrunre.m (copy of fcn_tgredrun.m
% + re-estimation).
%
% Stages (variable rrStage):
%   'check' : (1) K = inf with fcn_tgredrunre reproduces the cached SA b18
%             unit sbdm_tg_red_kodim09_SAb18_c1.mat (bitwise);
%             (2) K = 1 denoiser = the trained SA net with input-dependent
%             angles (max |remake(k,z)(z) - net(z)|) at a few iterates;
%             (3) time of one re-estimation and of one RED step -> estimate.
%   'run'   : the units for K in rrKs (default [50 10 1]); clip 1 before
%             clip 0; skip a unit that would end after rrDeadline. Then the
%             summary (all cached units).
% New files only. Caches: data/sbdm_rr_red_kodim%02d_SAb18_K%d_c%d.mat.
% Logs: logs/redreest_*.log, summary logs/redreest_summary.txt.
%   nohup matlab -licmode onlinelicensing -batch "rrStage='check'; main_sbdm_salsun_redreest" > ~/Documents/SBDMwLSUN/logs/redreest_check.log 2>&1 &

here = fileparts(mfilename('fullpath'));
cd(fullfile(here,'..','..')); setpath; cd(here)
addpath(fullfile(here,'..','sbdm'))
addpath(fullfile(here,'..','salsun'))
warning('off','MATLAB:rmpath:DirNotFound')
rmpath(fullfile('/home/shogo/Workspace/GitHub/SaivDr','mexcodes'))
warning('on','MATLAB:rmpath:DirNotFound')

if ~exist('rrStage','var'), rrStage = 'check'; end
if ~exist('rrKs','var'), rrKs = [50 10 1]; end
if ~exist('rrDeadline','var'), rrDeadline = datetime('now') + hours(2); end
if ~exist('rrSummary','var'), rrSummary = true; end
rrStage = string(rrStage);

% ---- settings (as main_sbdm_salsun_tnrdguar.m, Part 3) ----
cfg0 = fcn_ssconfig();
cfg0.selImg = 9;
cfg0.sigmas = [25 10 5 2];
rd = struct('blurSize',9,'blurStd',2.0,'sigmaW',5/255,'nIters',300,'sgMax',25/255,'sgMin',2/255,...
    'etas',[0.5 1 2],'betas',[0.5 1 2],'seedBase',1000);
dat = fullfile(here,'..','..','..','data');
ctx.dat = dat; ctx.cdir = dat;
ctx.sdir = fullfile(here,'..','..','..','..','logs');
ctx.deadline = datetime(rrDeadline);
ctx.margin = 1.10;
isTest = exist('onTest','var') && onTest;
if isTest
    rd.nIters = 6; rd.etas = [0.5 2]; rd.betas = [0.5 2];
    ctx.cdir = onTestDir; ctx.sdir = onTestDir;
    if ~isfolder(onTestDir), mkdir(onTestDir); end
end
ctx.useGpu = canUseGPU;
if ctx.useGpu, deep.gpu.deterministicAlgorithms(true); end
fprintf("deterministic GPU algorithms: %d\n",ctx.useGpu && deep.gpu.deterministicAlgorithms);
ctx.cfg0 = cfg0;
ctx.ST = containers.Map();
ctx.ST('skips') = strings(0,1);
fprintf("%s  start. Stage %s, K %s, GPU %d, deadline %s, test %d\n",string(datetime('now')),rrStage,...
    mat2str(rrKs),ctx.useGpu,string(ctx.deadline),isTest)

blurKernel = fspecial('gaussian',rd.blurSize,rd.blurStd);
Hf  = @(z) imfilter(z,blurKernel,'circular');
Htf = @(z) imfilter(z,rot90(blurKernel,2),'circular');
schedule = exp(linspace(log(rd.sgMax),log(rd.sgMin),rd.nIters));
sgBank = cfg0.sigmas/255;
kk = nearestk(schedule,sgBank);
fprintf("RED model bank: sigma %s; iterations per model %s\n",mat2str(cfg0.sigmas),mat2str(accumarray(kk,1,[numel(sgBank) 1]).'))
MB = cell(1,numel(cfg0.sigmas));
for s = 1:numel(cfg0.sigmas), MB{s} = struct('SA18',loadsa(ctx,cfg0.sigmas(s))); end
BB = [];

%% ===== stage check =====
if rrStage == "check"
    BB = buildsa(ctx,[],18,cfg0.cropSize);
    % (1) K = inf reproduces the cached unit
    [X,V] = redobs(ctx,9,Hf,rd);
    t = tic;
    q = runredre(ctx,MB,BB,X,V,Hf,Htf,schedule,sgBank,rd,true,inf);
    tInf = toc(t);
    if isTest
        fprintf("CHECK 1 K = inf vs cached: test mode, not compared (%.1f s)\n",tInf)
    else
        C = load(fullfile(dat,'sbdm_tg_red_kodim09_SAb18_c1.mat')); Rc = C.q.R;
        same = true; dRes = 0; dP = 0; dS = 0;
        for c = 1:numel(q.R)
            rc = Rc([Rc.eta] == q.R(c).eta & [Rc.beta] == q.R(c).beta);
            dRes = max(dRes,max(abs(q.R(c).res - rc.res))); dP = max(dP,abs(q.R(c).psnrEnd - rc.psnrEnd));
            dS = max(dS,abs(q.R(c).ssimEnd - rc.ssimEnd));
            same = same && isequal(q.R(c).res,rc.res) && isequal(q.R(c).psnrEnd,rc.psnrEnd) && isequal(q.R(c).ssimEnd,rc.ssimEnd);
        end
        fprintf("CHECK 1 K = inf (fcn_tgredrunre) vs cached SA b18 kodim09 clip 1: 9 runs, max |d res| %.3e, max |d PSNR| %.3e, max |d SSIM| %.3e, bitwise equal %d (%.1f s)\n",...
            dRes,dP,dS,same,tInf)
        if ~same, error('CHECK 1 failed.'), end
    end
    % (2) K = 1 denoiser = trained net with input-dependent angles; (3) timing
    dev = @(z) z; if ctx.useGpu, dev = @(z) gpuArray(z); end
    D = mkdens(ctx,MB,BB,V,sgBank);
    rng(7,'twister');
    tRe = []; tF = []; dmx = 0;
    for s = 1:numel(sgBank)
        for j = 1:5
            z = min(max(V + 0.05*randn(size(V)),0),1);      % some iterate
            tr = tic; f = D.remake(s,z); tRe(end+1) = toc(tr); %#ok<SAGROW>
            tf = tic; y = f(z); tF(end+1) = toc(tf); %#ok<SAGROW>
            yN = double(gather(extractdata(predict(D.snet{s},dlarray(dev(single(z)),'SSCB')))));
            dmx = max(dmx,max(abs(y(:) - yN(:))));
        end
    end
    fprintf("CHECK 2 K = 1 denoiser vs trained net (input-dependent angles), 4 models x 5 iterates: max |diff| %.3e\n",dmx)
    tReM = median(tRe(2:end)); tFM = median(tF(2:end));
    tStep = tInf/(numel(q.R)*rd.nIters);
    fprintf("TIME one re-estimation %.4f s (median), one denoiser call %.4f s, one RED step (K = inf, incl. PSNR) %.4f s\n",tReM,tFM,tStep)
    nRe = @(K) sum(arrayfun(@(K1) nrecount(kk,K1),K));
    tot = 0;
    for K = rrKs
        tu = numel(rd.etas)*numel(rd.betas)*(rd.nIters*tStep + nrecount(kk,K)*tReM) + 20;
        tot = tot + 8*tu;
        fprintf("ESTIMATE K = %d: %d re-estimations per run, %.0f s per unit (9 runs), %.1f min for 8 units\n",K,nrecount(kk,K),tu,8*tu/60)
    end
    fprintf("ESTIMATE total %.1f min (+ about 1.5 min start and net build)\n",tot/60)
    fprintf("\n%s  done.\n",string(datetime('now')))
    return
end

%% ===== stage run =====
ctx.ST('tStep') = 0.0452; ctx.ST('tRe') = 0.1;     % replaced by measured unit times
R3 = struct('img',{},'method',{},'clip',{},'runs',{});
for K = rrKs
    key = sprintf("redK%d",K);
    ctx.ST(char(key)) = numel(rd.etas)*numel(rd.betas)*(rd.nIters*ctx.ST('tStep') + nrecount(kk,K)*ctx.ST('tRe')) + 20;
    for clip = [true false]
        for i = 1:numel(cfg0.imgIdx)
            idx = cfg0.imgIdx(i);
            f = fullfile(ctx.cdir,sprintf('sbdm_rr_red_kodim%02d_SAb18_K%d_c%d.mat',idx,K,clip));
            if ~isfile(f) && isempty(BB), BB = buildsa(ctx,[],18,cfg0.cropSize); end
            [X,V] = redobs(ctx,idx,Hf,rd);
            q = unit(ctx,sprintf("RED kodim%02d SA b18 K %d clip %d",idx,K,clip),f,key,...
                @() runredre(ctx,MB,BB,X,V,Hf,Htf,schedule,sgBank,rd,clip,K));
            R3(end+1) = struct('img',idx,'method',sprintf("SA b18 K=%d",K),'clip',clip,'runs',{q}); %#ok<SAGROW>
        end
    end
end
if ~rrSummary
    fprintf("\nSkipped / failed / missing units (%d):\n",numel(ctx.ST('skips'))); fprintf("  %s\n",ctx.ST('skips'))
    fprintf("\n%s  done.\n",string(datetime('now')))
    return
end

%% ===== cached: K = inf, TNRD L old / c4, TNRD S3 c3 =====
CM = struct('name',{"SA b18 K=inf","TNRD L old","TNRD L c4","TNRD S3 c3"},...
    'pat',{'sbdm_tg_red_kodim%02d_SAb18_c%d.mat','sbdm_tg_red_kodim%02d_TNRDL_c%d.mat',...
    'sbdm_tg2_red_kodim%02d_TNRDLc4_c%d.mat','sbdm_tg2_red_kodim%02d_TNRDS3c3_c%d.mat'});
for m = 1:numel(CM)
    for clip = [true false]
        for i = 1:numel(cfg0.imgIdx)
            f = fullfile(dat,sprintf(CM(m).pat,cfg0.imgIdx(i),clip));
            if isfile(f), S = load(f); q = S.q; else, q = []; addskip(ctx,sprintf("MISSING cached %s",f)); end
            R3(end+1) = struct('img',cfg0.imgIdx(i),'method',CM(m).name,'clip',clip,'runs',{q}); %#ok<SAGROW>
        end
    end
end
names = ["SA b18 K=inf" "SA b18 K=" + string(sort(rrKs,'descend')) "TNRD L old" "TNRD L c4" "TNRD S3 c3"];
cases = struct('name',{},'clip',{});
for nm = names
    for clip = [true false], cases(end+1) = struct('name',nm,'clip',clip); end %#ok<SAGROW>
end
try
    summaryrr(fullfile(ctx.sdir,'redreest_summary.txt'),R3,ctx,rd,schedule,sgBank,cases,rrKs);
catch ME
    fprintf("Summary FAILED: %s\n%s\n",ME.message,getReport(ME,'extended','hyperlinks','off'))
end
R3s = R3;
for k = 1:numel(R3s)
    if ~isempty(R3s(k).runs) && isfield(R3s(k).runs,'R') && isfield(R3s(k).runs.R,'est'), R3s(k).runs.R = rmfield(R3s(k).runs.R,'est'); end
end
save(fullfile(ctx.cdir,'sbdm_rr_red_results.mat'),'R3s','rd','schedule','sgBank','rrKs')
sk = ctx.ST('skips');
fprintf("\nSkipped / failed / missing units (%d):\n",numel(sk))
fprintf("  %s\n",sk)
fprintf("\n%s  done.\n",string(datetime('now')))

%% ===================== Local functions =====================
function n = nrecount(kk,K)
% number of re-estimations per run (see fcn_tgredrunre)
if isinf(K), n = 0; return, end
n = 0; kCur = 0; tLast = -inf;
for it = 1:numel(kk)
    if kk(it) ~= kCur
        if it > 1, n = n + 1; end
        kCur = kk(it); tLast = it;
    elseif it - tLast >= K
        n = n + 1; tLast = it;
    end
end
end

function q = loadsa(ctx,sg)
f = fullfile(ctx.dat,sprintf('sbdm_salsun_dc_sup_s%02d_SAR_it1500.mat',sg));
S = load(f); q = S.q;
end

function B = buildsa(ctx,sg,bList,sz)
% copy of main_sbdm_salsun_tnrdguar.m/buildsa
B = struct(); t = tic;
if isempty(sg), sgs = ctx.cfg0.sigmas; else, sgs = sg; end
for s = sgs
    for b = bList
        ob = struct('estimator',"net",'leak',false,'bottleneck',b,'shrink',true,...
            'dcShrink',false,'dcInit',"id",'sigma',single(s/255));
        nets = fcn_dcbuildsalsun(sz,ctx.cfg0.stride,ctx.cfg0.ovlp,ob);
        B.("est" + b + "_" + s) = nets.shrink;
    end
    c = fcn_ssbuildconstsalsun(sz,ctx.cfg0.stride,ctx.cfg0.ovlp,single(s/255));
    B.("const" + s) = c.shrink;
end
fprintf("  built SA nets (b %s, sigma %s): %.1f s\n",mat2str(bList),mat2str(sgs),toc(t))
end

function [cs,snet,dmax,kap,A,Bv] = safrozen(mdl,estNet,constNet,V,useGpu)
% copy of main_sbdm_salsun_tnrdguar.m/safrozen (+ outputs A, Bv)
snet = estNet; L = snet.Learnables; Ls = mdl.learn; n = 0;
for i = 1:height(Ls)
    idx = find(strcmp(L.Layer,Ls.Layer{i}) & strcmp(L.Parameter,Ls.Parameter{i}));
    if ~isempty(idx), L.Value{idx} = Ls.Value{i}; n = n + 1; end
end
assert(n == height(L),'Not all learnables were copied (%d/%d).',n,height(L))
snet.Learnables = L;
ly = snet.Layers(strcmp({snet.Layers.Name},'Lv1_Shrink'));
assert(abs(double(ly.Sigma) - double(mdl.sigma)) < 1e-7,'est net sigma %g ~= model sigma %g',ly.Sigma,mdl.sigma)
if useGpu, snet = dlupdate(@gpuArray,snet); end
A = Ls.Value{strcmp(Ls.Layer,'Lv1_Shrink') & strcmp(Ls.Parameter,'A')};
Bv = Ls.Value{strcmp(Ls.Layer,'Lv1_Shrink') & strcmp(Ls.Parameter,'B')};
[th,names] = fcn_ssgetthetas(snet,V);
cs = fcn_sssetconstangles(constNet,names,th,A,Bv);
if useGpu, cs = dlupdate(@gpuArray,cs); end
ly = cs.Layers(strcmp({cs.Layers.Name},'Lv1_Shrink'));
assert(abs(double(ly.Sigma) - double(mdl.sigma)) < 1e-7,'const net sigma %g ~= model sigma %g',ly.Sigma,mdl.sigma)
dmax = double(gather(max(abs(extractdata(predict(cs,V) - predict(snet,V))),[],'all')));
kap = 1./(1+exp(-double(gather(extractdata(A)))));
end

function D = mkdens(ctx,MB,BB,V,sgBank)
% denoisers with the angles from V (as Part 3) and the remake function
nS = numel(sgBank);
dev = @(z) z; if ctx.useGpu, dev = @(z) gpuArray(z); end
D.fb = cell(1,nS); D.cs = cell(1,nS); D.snet = cell(1,nS); D.A = cell(1,nS); D.B = cell(1,nS);
D.dFrozen = nan(1,nS); D.kappaMax = nan(1,nS);
for s = 1:nS
    sg = round(sgBank(s)*255);
    Vd = dlarray(dev(single(V)),'SSCB');
    [cs,snet,dmax,kap,A,Bv] = safrozen(MB{s}.SA18.mdl,BB.("est18_" + sg),BB.("const" + sg),Vd,ctx.useGpu);
    D.cs{s} = cs; D.snet{s} = snet; D.A{s} = A; D.B{s} = Bv;
    D.dFrozen(s) = dmax; D.kappaMax(s) = max(kap);
    D.fb{s} = @(z) double(gather(extractdata(predict(cs,dlarray(dev(single(z)),'SSCB')))));
end
names = {};
allNames = {D.snet{1}.Layers.Name}; names = allNames(endsWith(allNames,'Theta'));
D.remake = @(s,z) remakeden(D.snet{s},D.cs{s},names,D.A{s},D.B{s},z,dev,ctx.useGpu);
end

function f = remakeden(snet,cs,names,A,Bv,z,dev,useGpu)
% angles estimated from z by the trained estimator, then constant
Zd = dlarray(dev(single(z)),'SSCB');
th = fcn_ssgetthetas(snet,Zd,names);
c2 = fcn_sssetconstangles(cs,names,th,A,Bv);
if useGpu, c2 = dlupdate(@gpuArray,c2); end
f = @(x) double(gather(extractdata(predict(c2,dlarray(dev(single(x)),'SSCB')))));
end

function q = runredre(ctx,MB,BB,X,V,Hf,Htf,schedule,sgBank,rd,clip,K)
D = mkdens(ctx,MB,BB,V,sgBank);
q.dFrozen = D.dFrozen; q.kappaMax = D.kappaMax; q.K = K;
R = struct([]);
for e = rd.etas
    for b = rd.betas
        t = tic;
        r = fcn_tgredrunre(D.fb,D.remake,sgBank,V,X,Hf,Htf,schedule,e,b,rd.nIters,clip,K);
        r.eta = e; r.beta = b; r.t = toc(t);
        R = [R r]; %#ok<AGROW>
        fprintf("    SA b18 K %-4g clip %d eta %.1f beta %.1f: PSNR %.4f SSIM %.4f res(T) %.3e blowup %d  re-est %d (%.1f s) (%.1f s)\n",...
            K,clip,e,b,r.psnrEnd,r.ssimEnd,r.res(end),r.blowup,r.nRe,r.tRe,r.t)
    end
end
q.R = R;
end

function [X,V] = redobs(ctx,idx,Hf,rd)
cfg = ctx.cfg0;
x = fcn_ssloadimage(ctx.dat,idx,cfg,false);
X = double(x);
rng(rd.seedBase + idx,'twister');
V = Hf(X) + rd.sigmaW*randn(size(X));
end

function k = nearestk(schedule,sgBank)
[~,k] = min(abs(log(schedule(:)) - log(sgBank(:).')),[],2);
end

function e = estget(ST,key)
if isKey(ST,char(key)), e = ST(char(key)); return, end
e = 600;
end

function q = unit(ctx,lab,f,key,fn)
% copy of main_sbdm_salsun_tnrdguar.m/unit
if isfile(f)
    S = load(f); q = S.q;
    fprintf("  %-34s cached (%s)\n",lab,f)
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
fprintf("\n%s  %-34s start (estimate %.0f s)\n",string(datetime('now')),lab,e)
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
ctx.ST(char(key)) = dt;
ftmp = string(f) + ".tmp.mat";
save(ftmp,'q','-v7.3'); movefile(ftmp,f);
fprintf("%s  %-34s done (%.1f s)\n",string(datetime('now')),lab,q.tWall);
end

function addskip(ctx,msg)
ctx.ST('skips') = [ctx.ST('skips'); string(msg)];
fprintf("  %s\n",msg)
end

%% ----- summary -----
function [Pm,Sm,Rf,Rr,St,Bl,Tm,Nr] = getmat(R3,name,clip,imgs,EE,BB,N)
% as main_sbdm_salsun_tnrdguar.m/summary3 (+ run time, re-estimations)
nC = numel(EE); nI = numel(imgs);
Pm = nan(nC,nI); Sm = Pm; Rf = Pm; Rr = Pm; St = Pm; Bl = Pm; Tm = Pm; Nr = Pm;
for i = 1:nI
    q = R3(arrayfun(@(r) r.img == imgs(i) && r.method == name && r.clip == clip,R3));
    if isempty(q) || isempty(q.runs), continue, end
    for c = 1:nC
        r = q.runs.R([q.runs.R.eta] == EE(c) & [q.runs.R.beta] == BB(c));
        if isempty(r), continue, end
        Pm(c,i) = r.psnrEnd; Sm(c,i) = r.ssimEnd; Bl(c,i) = r.blowup;
        Rf(c,i) = r.res(end)/sqrt(N);
        tb = max(1,numel(r.res) - 50);
        Rr(c,i) = r.res(end)/r.res(tb);
        St(c,i) = ~r.blowup && all(isfinite(r.res)) && r.res(end) <= r.res(tb) && r.res(end)/sqrt(N) < 1e-3;
        if isfield(r,'t'), Tm(c,i) = r.t; end
        if isfield(r,'nRe'), Nr(c,i) = r.nRe; end
    end
end
end

function summaryrr(file,R3,ctx,rd,schedule,sgBank,cases,rrKs)
fid = fopen(file,'w'); cleaner = onCleanup(@() fclose(fid));
cfg = ctx.cfg0; imgs = cfg.imgIdx; nI = numel(imgs); N = prod(cfg.cropSize);
out(fid,"RED with SA b18, angles re-estimated from the current iterate every K its (and at each sigma-model switch). Written %s.\n",string(datetime('now')));
out(fid,"K = inf: Part 3 of main_sbdm_salsun_tnrdguar.m (angles from v, frozen), cached. TNRD L old/c4, TNRD S3 c3: cached (tnrdguar / tnrdretrain).\n");
out(fid,"As Part 3: blur %dx%d Gaussian std %.1f (circular), noise %.4f, rng(%d + image), 256x256 crops, z(0) = v, eps = 0, %d its, sigma(t) %g -> %g (log).\n",...
    rd.blurSize,rd.blurSize,rd.blurStd,rd.sigmaW,rd.seedBase,rd.nIters,rd.sgMax*255,rd.sgMin*255);
k = nearestk(schedule,sgBank);
out(fid,"Model bank: its per model sigma %s = %s. Re-estimations per run: %s for K = %s.\n",mat2str(round(sgBank*255)),...
    mat2str(accumarray(k,1,[numel(sgBank) 1]).'),mat2str(arrayfun(@(K) nrecount(k,K),rrKs)),mat2str(rrKs));
out(fid,"Stable := finite, r(T) <= r(T-50), r(T)/sqrt(N) < 1e-3 (T = %d). Blow-up := z not finite or max|z| > 1e6.\n",rd.nIters);
[EE,BB] = ndgrid(rd.etas,rd.betas);
combo = strings(0,1);
for c = 1:numel(EE), combo(c,1) = sprintf("eta %.1f beta %.1f",EE(c),BB(c)); end
i09 = find(imgs == 9); iT = find(imgs ~= 9);

out(fid,"\n==================== Paper block ====================\n");
out(fid,"beta: chosen at eta = 0.5 by the best PSNR on kodim09 (first one on a tie).\n");
out(fid,"mean4 = mean over kodim09/15/19/23; mean3 = mean over kodim15/19/23 (held out from the choice).\n");
out(fid,"stable: (setting, image) cases of %d. PSNR range: min and max over the settings of the mean4 PSNR.\n",numel(EE)*nI);
out(fid,"'r >= 1': settings where r(T)/sqrt(N) >= 1 on at least one image. time: sum of the run times of the %d runs (s); '-' if not recorded.\n",numel(EE)*nI);
out(fid,"%-26s %5s %10s %8s %10s %8s %8s %21s %12s %8s %9s  %s\n","method","beta","PSNR4","SSIM4","PSNR3","SSIM3","stable","PSNR range","max r/sqrtN","blowups","time [s]","settings with r >= 1");
for cc = cases
    [Pm,Sm,Rf,~,St,Bl,Tm] = getmat(R3,cc.name,cc.clip,imgs,EE,BB,N);
    lab = cc.name + " clip " + double(cc.clip);
    if all(isnan(Pm(:))), out(fid,"%-26s -\n",lab); continue, end
    c05 = find(EE == 0.5);
    [~,j] = max(Pm(c05,i09)); c = c05(j);
    m4 = mean(Pm,2);
    big = combo(any(Rf >= 1,2));
    if isempty(big), bs = "none"; else, bs = strjoin(big,", "); end
    if all(isnan(Tm(:))), ts = "-"; else, ts = sprintf("%.1f",sum(Tm(:),'omitnan')); end
    out(fid,"%-26s %5.1f %10.4f %8.4f %10.4f %8.4f %4d/%-3d %10.4f-%-10.4f %12.3e %8d %9s  %s\n",lab,BB(c),...
        mean(Pm(c,:)),mean(Sm(c,:)),mean(Pm(c,iT)),mean(Sm(c,iT)),sum(St(:)),numel(St),min(m4),max(m4),max(Rf(:)),sum(Bl(:),'omitnan'),ts,bs);
end
out(fid,"\nkodim09 PSNR at eta = 0.5 (for the choice of beta), beta %s:\n",mat2str(rd.betas));
for cc = cases
    Pm = getmat(R3,cc.name,cc.clip,imgs,EE,BB,N);
    out(fid,"  %-26s %s\n",cc.name + " clip " + double(cc.clip),strjoin(compose("%.4f",Pm(EE == 0.5,i09)).'," / "));
end

for cc = cases
    out(fid,"\n==================== %s, clip %d ====================\n",cc.name,cc.clip);
    [Pm,Sm,Rf,Rr,St,Bl,Tm,Nr] = getmat(R3,cc.name,cc.clip,imgs,EE,BB,N);
    cl = [compose("kodim%02d",imgs) "mean"]; cx = [compose("kodim%02d",imgs) "max"]; cn = [compose("kodim%02d",imgs) "count"];
    printtable(fid,"PSNR [dB] (last col: mean)",combo,cl,[Pm mean(Pm,2)],"%.4f")
    printtable(fid,"SSIM (last col: mean)",combo,cl,[Sm mean(Sm,2)],"%.4f")
    printtable(fid,"Final residual r(T)/sqrt(N) (last col: MAX over images)",combo,cx,[Rf max(Rf,[],2)],"%.3e")
    printtable(fid,"r(T)/r(T-50) (last col: MAX over images)",combo,cx,[Rr max(Rr,[],2)],"%.3e")
    printtable(fid,"Stable (1/0) (last col: number of images)",combo,cn,[St sum(St,2)],"%d")
    printtable(fid,"Blow-up (1/0) (last col: number of images)",combo,cn,[Bl sum(Bl,2)],"%d")
    if ~all(isnan(Tm(:))), printtable(fid,"Run time [s] (last col: mean)",combo,cl,[Tm mean(Tm,2)],"%.1f"), end
    if ~all(isnan(Nr(:))), printtable(fid,"Re-estimations per run (last col: max)",combo,cx,[Nr max(Nr,[],2)],"%d"), end
    out(fid,"  stable total: %d of %d; blow-up total: %d\n",sum(St(:)),numel(St),sum(Bl(:),'omitnan'));
end

out(fid,"\n==================== Residual r(t)/sqrt(N) at selected its (mean over images) ====================\n");
tl = unique([1 10 50 100 150 200 250 rd.nIters]); tl = tl(tl <= rd.nIters);
for cc = cases(arrayfun(@(c) startsWith(c.name,"SA b18"),cases))
    Mc = nan(numel(EE),numel(tl));
    for c = 1:numel(EE)
        acc = [];
        for i = 1:nI
            q = R3(arrayfun(@(r) r.img == imgs(i) && r.method == cc.name && r.clip == cc.clip,R3));
            if isempty(q) || isempty(q.runs), continue, end
            r = q.runs.R([q.runs.R.eta] == EE(c) & [q.runs.R.beta] == BB(c));
            acc = [acc r.res(tl).'/sqrt(N)]; %#ok<AGROW>
        end
        if ~isempty(acc), Mc(c,:) = mean(reshape(acc,numel(tl),[]).',1); end
    end
    printtable(fid,sprintf("%s, clip %d",cc.name,cc.clip),combo,compose("t=%d",tl),Mc,"%.3e")
end
sk = ctx.ST('skips');
out(fid,"\n---- Skipped / failed / missing units (%d) ----\n",numel(sk));
for k2 = 1:numel(sk), out(fid,"  %s\n",sk(k2)); end
end

function out(fid,fmt,varargin)
fprintf(fid,fmt,varargin{:});
fprintf(1,fmt,varargin{:});
end

function printtable(fid,ttl,rows,cols,M,fmt)
out(fid,"\n%s\n",ttl);
w = max(11,max(strlength(cols))+1);
s = sprintf("%-20s","");
for j = 1:numel(cols), s = s + sprintf("%*s",w,cols(j)); end
out(fid,"%s\n",s);
for i = 1:numel(rows)
    s = sprintf("%-20s",rows(i));
    for j = 1:size(M,2)
        if isnan(M(i,j)), c = "-"; else, c = strtrim(sprintf(fmt,M(i,j))); end
        s = s + sprintf("%*s",w,c);
    end
    out(fid,"%s\n",s);
end
end
