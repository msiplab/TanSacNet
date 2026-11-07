%% main_sbdm_salsun_redsmall.m
% RED deblurring for the small pair: SA b3 (SA-LSUN, 2723 params) run
% exactly as Part 3 of main_sbdm_salsun_tnrdguar.m ran SA b18, and a
% summary for the paper's RED block together with the cached runs
% (SA b18, TNRD S3 c3/c4, TNRD L c4). (2026-11-07)
%
% Setting (copied from main_sbdm_salsun_tnrdguar.m, Part 3):
%   9x9 Gaussian blur, std 2, circular; noise 5/255 with rng(1000 + image
%   index); 256x256 centre crops kodim09/15/19/23; z(0) = v; eps = 0;
%   300 its; sigma(t) = 25/255 -> 2/255 (log); model bank of the fixed-sigma
%   models (nearest sigma in log scale); SA: angles estimated once from the
%   observation v by each sigma model, then frozen; eta, beta in {0.5,1,2};
%   with and without clipping to [0,1].
%   Models: SA b3 = sbdm_salsun_supsize_s%02d_SAR_b3_it1500.mat (size sweep,
%   the same models as Parts 1 and 2).
% Check: the code here re-runs SA b18 on kodim09 (clipping) and compares
%   with the cached unit of tnrdguar (sbdm_tg_red_kodim09_SAb18_c1.mat).
%
% New file only. Caches: data/sbdm_rs_red_*.mat. Summary: logs/redsmall_summary.txt
%   nohup matlab -licmode onlinelicensing -batch "main_sbdm_salsun_redsmall" \
%       > ~/Documents/SBDMwLSUN/logs/redsmall.log 2>&1 &
% Quick test (tiny settings, separate folder):
%   matlab -licmode onlinelicensing -batch "onTest = true; onTestDir = '<dir>'; main_sbdm_salsun_redsmall"

here = fileparts(mfilename('fullpath'));
cd(fullfile(here,'..','..')); setpath; cd(here)
addpath(fullfile(here,'..','sbdm'))     % helpers (called, not edited)
addpath(fullfile(here,'..','salsun'))   % maskLayer
warning('off','MATLAB:rmpath:DirNotFound')
rmpath(fullfile('/home/shogo/Workspace/GitHub/SaivDr','mexcodes'))
warning('on','MATLAB:rmpath:DirNotFound')

% ---- settings (as main_sbdm_salsun_tnrdguar.m) ----
cfg0 = fcn_ssconfig();
cfg0.selImg = 9;
cfg0.sigmas = [25 10 5 2];
rd = struct('blurSize',9,'blurStd',2.0,'sigmaW',5/255,'nIters',300,'sgMax',25/255,'sgMin',2/255,...
    'etas',[0.5 1 2],'betas',[0.5 1 2],'seedBase',1000);
bNew = 3;                                    % SA b3
dat = fullfile(here,'..','..','..','data');
ctx.dat = dat; ctx.cdir = dat;
ctx.sdir = fullfile(here,'..','..','..','..','logs');
ctx.deadline = datetime('now') + hours(3);
ctx.margin = 1.10;
isTest = exist('onTest','var') && onTest;
if isTest
    rd.nIters = 4; rd.etas = [0.5 2]; rd.betas = [0.5 2];
    ctx.cdir = onTestDir; ctx.sdir = onTestDir;
    if ~isfolder(onTestDir), mkdir(onTestDir); end
end
ctx.useGpu = canUseGPU;
if ctx.useGpu, deep.gpu.deterministicAlgorithms(true); end   % as tnrdguar
fprintf("deterministic GPU algorithms: %d\n",ctx.useGpu && deep.gpu.deterministicAlgorithms);
ctx.cfg0 = cfg0;
ctx.ST = containers.Map();
ctx.ST('skips') = strings(0,1);
ctx.ST('redSA') = 9*rd.nIters*0.12 + 60;
fprintf("%s  start. GPU: %d. Deadline %s. Test mode: %d\n",string(datetime('now')),ctx.useGpu,string(ctx.deadline),isTest)
disp(rd)
tRun0 = datetime('now');

%% ===== RED setting =====
blurKernel = fspecial('gaussian',rd.blurSize,rd.blurStd);
Hf  = @(z) imfilter(z,blurKernel,'circular');
Htf = @(z) imfilter(z,rot90(blurKernel,2),'circular');
schedule = exp(linspace(log(rd.sgMax),log(rd.sgMin),rd.nIters));
sgBank = cfg0.sigmas/255;
fprintf("\nRED model bank: sigma %s; iterations per model %s\n",mat2str(cfg0.sigmas),...
    mat2str(accumarray(nearestk(schedule,sgBank),1,[numel(sgBank) 1]).'))
MB = cell(1,numel(cfg0.sigmas));
for s = 1:numel(cfg0.sigmas)
    MB{s} = struct('SA3',loadsa(ctx,cfg0.sigmas(s),3),'SA18',loadsa(ctx,cfg0.sigmas(s),18));
end
BB = buildsa(ctx,[],[bNew 18],cfg0.cropSize);

%% ===== check: this code reproduces the cached SA b18 unit (kodim09, clip 1) =====
chk = struct('ok',false,'info',"not run");
try
    [X,V] = redobs(ctx,9,Hf,rd);
    t = tic;
    q18 = runredsa(ctx,18,MB,BB,X,V,Hf,Htf,schedule,sgBank,rd,true);
    fC = fullfile(dat,'sbdm_tg_red_kodim09_SAb18_c1.mat');
    if isTest
        chk.info = sprintf("test mode: not compared (%d its); ran in %.1f s",rd.nIters,toc(t));
        chk.ok = true;
    else
        C = load(fC); Rc = C.q.R;
        dRes = 0; dP = 0; dS = 0; same = true;
        for c = 1:numel(q18.R)
            rc = Rc([Rc.eta] == q18.R(c).eta & [Rc.beta] == q18.R(c).beta);
            dRes = max(dRes,max(abs(q18.R(c).res - rc.res)));
            dP = max(dP,abs(q18.R(c).psnrEnd - rc.psnrEnd)); dS = max(dS,abs(q18.R(c).ssimEnd - rc.ssimEnd));
            same = same && isequal(q18.R(c).res,rc.res) && isequal(q18.R(c).psnrEnd,rc.psnrEnd) && isequal(q18.R(c).ssimEnd,rc.ssimEnd);
        end
        chk.ok = same || (dP < 1e-6 && dS < 1e-8);
        chk.info = sprintf("9 runs: max |d res| %.3e, max |d PSNR| %.3e, max |d SSIM| %.3e, bitwise equal %d (%.1f s)",dRes,dP,dS,same,toc(t));
    end
catch ME
    chk.info = "FAILED: " + string(ME.message);
    fprintf("%s\n",getReport(ME,'extended','hyperlinks','off'))
end
fprintf("CHECK SA b18 kodim09 clip 1, this code vs cached tnrdguar unit: ok %d  %s\n",chk.ok,chk.info)
if ~chk.ok
    error('The check failed: this code does not reproduce Part 3. Stop.')
end

%% ===== SA b3 units =====
R3 = struct('img',{},'method',{},'clip',{},'runs',{});
for i = 1:numel(cfg0.imgIdx)
    idx = cfg0.imgIdx(i);
    [X,V] = redobs(ctx,idx,Hf,rd);
    fprintf("\nkodim%02d: observation PSNR %.4f dB (clipped to [0,1] for the measure)\n",idx,psnr(min(max(V,0),1),X))
    for clip = [true false]
        f = fullfile(ctx.cdir,sprintf('sbdm_rs_red_kodim%02d_SAb%d_c%d.mat',idx,bNew,clip));
        q = unit(ctx,sprintf("RED kodim%02d SA b%d clip %d",idx,bNew,clip),f,"redSA",...
            @() runredsa(ctx,bNew,MB,BB,X,V,Hf,Htf,schedule,sgBank,rd,clip));
        R3(end+1) = struct('img',idx,'method',"SA b" + bNew,'clip',clip,'runs',{q}); %#ok<SAGROW>
    end
end

%% ===== cached runs (tnrdguar / tnrdretrain) =====
CM = struct('name',{"SA b18","TNRD S3 c3","TNRD S3 c4","TNRD L c4"},...
    'pat',{'sbdm_tg_red_kodim%02d_SAb18_c%d.mat','sbdm_tg2_red_kodim%02d_TNRDS3c3_c%d.mat',...
    'sbdm_tg2_red_kodim%02d_TNRDS3c4_c%d.mat','sbdm_tg2_red_kodim%02d_TNRDLc4_c%d.mat'});
for m = 1:numel(CM)
    for clip = [true false]
        for i = 1:numel(cfg0.imgIdx)
            f = fullfile(dat,sprintf(CM(m).pat,cfg0.imgIdx(i),clip));
            if isfile(f)
                S = load(f); q = S.q;
            else
                q = []; addskip(ctx,sprintf("MISSING cached %s",f));
            end
            R3(end+1) = struct('img',cfg0.imgIdx(i),'method',CM(m).name,'clip',clip,'runs',{q}); %#ok<SAGROW>
        end
    end
end

%% ===== summary =====
cases = struct('name',{},'clip',{});
for nm = ["SA b" + bNew "SA b18" "TNRD S3 c3" "TNRD S3 c4" "TNRD L c4"]
    for clip = [true false], cases(end+1) = struct('name',nm,'clip',clip); end %#ok<SAGROW>
end
try
    summaryrs(fullfile(ctx.sdir,'redsmall_summary.txt'),R3,ctx,rd,MB,schedule,sgBank,cases,chk,bNew);
catch ME
    fprintf("Summary FAILED: %s\n%s\n",ME.message,getReport(ME,'extended','hyperlinks','off'))
end
R3s = R3;
for k = 1:numel(R3s)
    if ~isempty(R3s(k).runs) && isfield(R3s(k).runs,'R') && isfield(R3s(k).runs.R,'est'), R3s(k).runs.R = rmfield(R3s(k).runs.R,'est'); end
end
save(fullfile(ctx.cdir,'sbdm_rs_red_results.mat'),'R3s','rd','schedule','sgBank','chk')

sk = ctx.ST('skips');
fprintf("\nTimes: start %s, end %s\n",string(tRun0),string(datetime('now')))
fprintf("\nSkipped / failed / missing units (%d):\n",numel(sk))
fprintf("  %s\n",sk)
fprintf("\n%s  done.\n",string(datetime('now')))

%% ===================== Local functions =====================
function q = loadsa(ctx,sg,b)
switch b
    case 18, f = fullfile(ctx.dat,sprintf('sbdm_salsun_dc_sup_s%02d_SAR_it1500.mat',sg));
    case 3,  f = fullfile(ctx.dat,sprintf('sbdm_salsun_supsize_s%02d_SAR_b3_it1500.mat',sg));
end
S = load(f); q = S.q;
fprintf("  loaded %s\n",f)
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

function [cs,snet,dmax,kap] = safrozen(mdl,estNet,constNet,V,useGpu)
% copy of main_sbdm_salsun_tnrdguar.m/safrozen: trained SA net and the same
% net with the angles estimated from V and then frozen
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

function [X,V] = redobs(ctx,idx,Hf,rd)
% copy of main_sbdm_salsun_tnrdguar.m/redobs
cfg = ctx.cfg0;
x = fcn_ssloadimage(ctx.dat,idx,cfg,false);
X = double(x);
rng(rd.seedBase + idx,'twister');
V = Hf(X) + rd.sigmaW*randn(size(X));
end

function k = nearestk(schedule,sgBank)
[~,k] = min(abs(log(schedule(:)) - log(sgBank(:).')),[],2);
end

function q = runredsa(ctx,b,MB,BB,X,V,Hf,Htf,schedule,sgBank,rd,clip)
% main_sbdm_salsun_tnrdguar.m/runred, SA branch, with the bottleneck b as
% an argument (there: b = 18 fixed). Angles from the observation, frozen.
nS = numel(sgBank); fb = cell(1,nS); q.dFrozen = nan(1,nS); q.kappaMax = nan(1,nS);
dev = @(z) z; if ctx.useGpu, dev = @(z) gpuArray(z); end
for s = 1:nS
    sg = round(sgBank(s)*255);
    Q = MB{s}.("SA" + b);
    Vd = dlarray(dev(single(V)),'SSCB');
    [cs,~,dmax,kap] = safrozen(Q.mdl,BB.("est" + b + "_" + sg),BB.("const" + sg),Vd,ctx.useGpu);
    q.dFrozen(s) = dmax; q.kappaMax(s) = max(kap);
    fb{s} = @(z) double(gather(extractdata(predict(cs,dlarray(dev(single(z)),'SSCB')))));
end
R = struct([]);
for e = rd.etas
    for bt = rd.betas
        t = tic;
        r = fcn_tgredrun(fb,sgBank,V,X,Hf,Htf,schedule,e,bt,rd.nIters,clip);
        r.eta = e; r.beta = bt; r.t = toc(t);
        R = [R r]; %#ok<AGROW>
        fprintf("    SA b%-3d clip %d eta %.1f beta %.1f: PSNR %.4f SSIM %.4f res(T) %.3e blowup %d (%.1f s)\n",...
            b,clip,e,bt,r.psnrEnd,r.ssimEnd,r.res(end),r.blowup,r.t)
    end
end
q.R = R;
end

function e = estget(ST,key)
if isKey(ST,char(key)), e = ST(char(key)); return, end
e = 600;
end

function q = unit(ctx,lab,f,key,fn)
% copy of main_sbdm_salsun_tnrdguar.m/unit
if isfile(f)
    S = load(f); q = S.q;
    fprintf("  %-32s cached (%s)\n",lab,f)
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
fprintf("\n%s  %-32s start (estimate %.0f s)\n",string(datetime('now')),lab,e)
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
fprintf("%s  %-32s done (%.1f s)\n",string(datetime('now')),lab,q.tWall);
end

function addskip(ctx,msg)
ctx.ST('skips') = [ctx.ST('skips'); string(msg)];
fprintf("  %s\n",msg)
end

%% ----- summary -----
function [Pm,Sm,Rf,Rr,St,Bl,Dfz] = getmat(R3,name,clip,imgs,EE,BB,N)
% the same computation as main_sbdm_salsun_tnrdguar.m/summary3
nC = numel(EE); nI = numel(imgs);
Pm = nan(nC,nI); Sm = Pm; Rf = Pm; Rr = Pm; St = Pm; Bl = Pm; Dfz = [];
for i = 1:nI
    q = R3(arrayfun(@(r) r.img == imgs(i) && r.method == name && r.clip == clip,R3));
    if isempty(q) || isempty(q.runs), continue, end
    if isfield(q.runs,'dFrozen'), Dfz = [Dfz q.runs.dFrozen]; end %#ok<AGROW>
    for c = 1:nC
        r = q.runs.R([q.runs.R.eta] == EE(c) & [q.runs.R.beta] == BB(c));
        if isempty(r), continue, end
        Pm(c,i) = r.psnrEnd; Sm(c,i) = r.ssimEnd; Bl(c,i) = r.blowup;
        Rf(c,i) = r.res(end)/sqrt(N);
        tb = max(1,numel(r.res) - 50);
        Rr(c,i) = r.res(end)/r.res(tb);
        St(c,i) = ~r.blowup && all(isfinite(r.res)) && r.res(end) <= r.res(tb) && r.res(end)/sqrt(N) < 1e-3;
    end
end
end

function summaryrs(file,R3,ctx,rd,MB,schedule,sgBank,cases,chk,bNew)
fid = fopen(file,'w'); cleaner = onCleanup(@() fclose(fid));
cfg = ctx.cfg0; imgs = cfg.imgIdx; nI = numel(imgs); N = prod(cfg.cropSize);
out(fid,"RED for the small pair (SA b%d new; SA b18, TNRD S3 c3/c4, TNRD L c4 cached). Written %s.\n",bNew,string(datetime('now')));
out(fid,"As main_sbdm_salsun_tnrdguar.m, Part 3: blur %dx%d Gaussian std %.1f (circular), noise %.4f (= 5/255), rng(%d + image index), 256x256 centre crops.\n",...
    rd.blurSize,rd.blurSize,rd.blurStd,rd.sigmaW,rd.seedBase);
out(fid,"z <- clip(z - eta*(H'(Hz - v) + beta*(z - f(z; sigma(t))))), z(0) = v, eps = 0, %d its, sigma(t) = %g -> %g (log).\n",rd.nIters,rd.sgMax*255,rd.sgMin*255);
k = nearestk(schedule,sgBank);
out(fid,"Model bank (nearest sigma in log scale): its per model sigma %s = %s. clip 0 = no clipping.\n",mat2str(round(sgBank*255)),mat2str(accumarray(k,1,[numel(sgBank) 1]).'));
out(fid,"SA b%d: models sbdm_salsun_supsize_s%%02d_SAR_b%d_it1500.mat; angles estimated once from the observation v by each sigma model, then frozen.\n",bNew,bNew);
out(fid,"Cached: SA b18 (tnrdguar, sbdm_tg_red_*), TNRD S3 c3/c4 and TNRD L c4 (tnrdretrain, sbdm_tg2_red_*).\n");
out(fid,"Residual r(t) = ||z(t) - z(t-1)||_2, shown as r/sqrt(N), N = %d.\n",N);
out(fid,"Stable := finite, r(T) <= r(T-50) and r(T)/sqrt(N) < 1e-3 (T = %d). Blow-up := z not finite or max|z| > 1e6.\n",rd.nIters);
out(fid,"\nCHECK SA b18 kodim09 clip 1, this code vs the cached tnrdguar unit: ok %d  %s\n",chk.ok,chk.info);
% theory for SA b3
out(fid,"\nTheory (SA frozen): eig(J) in [1 - kappaMax, 1], ||H'H|| = 1, so the step map is non-expansive (averaged) if eta*(1 + beta*kappaMax) < 2.\n");
kap = nan(1,numel(MB));
for s = 1:numel(MB)
    L = MB{s}.("SA" + bNew).mdl.learn;
    a = double(gather(extractdata(L.Value{strcmp(L.Layer,'Lv1_Shrink') & strcmp(L.Parameter,'A')})));
    kap(s) = max(1./(1+exp(-a)));
end
out(fid,"kappaMax of the SA b%d models (sigma %s): %s\n",bNew,mat2str(round(sgBank*255)),mat2str(kap,4));
km = max(kap); nb = 0;
for e = rd.etas
    v = e*(1 + rd.betas*km); nb = nb + sum(v < 2);
    out(fid,"  eta %.1f: eta*(1 + beta*kappaMax) for beta %s = %s\n",e,mat2str(rd.betas),mat2str(v,4));
end
out(fid,"  settings with eta*(1 + beta*kappaMax) < 2: %d of %d\n",nb,numel(rd.etas)*numel(rd.betas));

[EE,BB] = ndgrid(rd.etas,rd.betas);
combo = strings(0,1);
for c = 1:numel(EE), combo(c,1) = sprintf("eta %.1f beta %.1f",EE(c),BB(c)); end
i09 = find(imgs == 9); iT = find(imgs ~= 9);

% ---- paper block ----
out(fid,"\n==================== Paper block ====================\n");
out(fid,"beta: chosen at eta = 0.5 by the best PSNR on kodim09 (first one on a tie).\n");
out(fid,"mean4 = mean over kodim09/15/19/23 (includes kodim09); mean3 = mean over kodim15/19/23 (held out from the choice).\n");
out(fid,"stable: number of (setting, image) cases of %d. PSNR range: min and max over the settings of the mean4 PSNR.\n",numel(EE)*nI);
out(fid,"max r(T)/sqrt(N): max over all settings and images. 'r >= 1': settings where r(T)/sqrt(N) >= 1 on at least one image.\n");
out(fid,"(With clipping z stays in [0,1], so r(T)/sqrt(N) <= 1 and a blow-up cannot happen.)\n");
hdr = sprintf("%-24s %5s %10s %8s %10s %8s %8s %21s %12s %8s  %s","method","beta","PSNR4","SSIM4","PSNR3","SSIM3","stable","PSNR range","max r/sqrtN","blowups","settings with r >= 1");
out(fid,"%s\n",hdr);
for cc = cases
    [Pm,Sm,Rf,~,St,Bl] = getmat(R3,cc.name,cc.clip,imgs,EE,BB,N);
    if all(isnan(Pm(:))), out(fid,"%-24s -\n",cc.name + " clip " + double(cc.clip)); continue, end
    c05 = find(EE == 0.5);
    [~,j] = max(Pm(c05,i09)); c = c05(j);
    m4 = mean(Pm,2);
    big = combo(any(Rf >= 1,2));
    if isempty(big), bs = "none"; else, bs = strjoin(big,", "); end
    out(fid,"%-24s %5.1f %10.4f %8.4f %10.4f %8.4f %4d/%-3d %10.4f-%-10.4f %12.3e %8d  %s\n",cc.name + " clip " + double(cc.clip),BB(c),...
        mean(Pm(c,:)),mean(Sm(c,:)),mean(Pm(c,iT)),mean(Sm(c,iT)),sum(St(:)),numel(St),min(m4),max(m4),max(Rf(:)),sum(Bl(:)),bs);
end
out(fid,"\nkodim09 PSNR at eta = 0.5 (for the choice of beta), beta %s:\n",mat2str(rd.betas));
for cc = cases
    Pm = getmat(R3,cc.name,cc.clip,imgs,EE,BB,N);
    out(fid,"  %-24s %s\n",cc.name + " clip " + double(cc.clip),strjoin(compose("%.4f",Pm(EE == 0.5,i09)).'," / "));
end

% ---- per-method tables (as summary3) ----
for cc = cases
    out(fid,"\n==================== %s, clip %d ====================\n",cc.name,cc.clip);
    [Pm,Sm,Rf,Rr,St,Bl,Dfz] = getmat(R3,cc.name,cc.clip,imgs,EE,BB,N);
    cl = [compose("kodim%02d",imgs) "mean"];
    printtable(fid,"PSNR [dB] (last col: mean)",combo,cl,[Pm mean(Pm,2)],"%.4f")
    printtable(fid,"SSIM (last col: mean)",combo,cl,[Sm mean(Sm,2)],"%.4f")
    printtable(fid,"Final residual r(T)/sqrt(N) (last col: MAX over images)",combo,[compose("kodim%02d",imgs) "max"],[Rf max(Rf,[],2)],"%.3e")
    printtable(fid,"r(T)/r(T-50) (last col: MAX over images)",combo,[compose("kodim%02d",imgs) "max"],[Rr max(Rr,[],2)],"%.3e")
    printtable(fid,"Stable (1/0) (last col: number of images)",combo,[compose("kodim%02d",imgs) "count"],[St sum(St,2)],"%d")
    printtable(fid,"Blow-up (1/0) (last col: number of images)",combo,[compose("kodim%02d",imgs) "count"],[Bl sum(Bl,2)],"%d")
    out(fid,"  stable total: %d of %d; blow-up total: %d\n",sum(St(:)),numel(St),sum(Bl(:)));
    if ~isempty(Dfz), out(fid,"  frozen - not frozen at v, max over sigma models and images: %.2e\n",max(Dfz)); end
end

% ---- residual curves (mean over images) ----
out(fid,"\n==================== Residual r(t)/sqrt(N) at selected its (mean over images) ====================\n");
tl = unique([1 10 50 100 150 200 250 rd.nIters]); tl = tl(tl <= rd.nIters);
for cc = cases(arrayfun(@(c) startsWith(c.name,"SA b" + bNew),cases))
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
