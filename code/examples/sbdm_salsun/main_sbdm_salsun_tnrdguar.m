%% main_sbdm_salsun_tnrdguar.m
% Sensei's goal: show the advantage of a denoiser that GUARANTEES
% non-expansiveness and a symmetric Jacobian (SA-LSUN with frozen angles)
% over one that does not (the TNRD type of examples/sbdm/main_sbdm.m).
% (2026-11-07)
%
% Part 1  TNRD in the supervised setting of stage B / run A / size sweep:
%         same 2048 patches 64x64 of kodim01-08 (rng(1)), fixed sigma per
%         model, sigma = 25, 10, 5, 2 /255. TNRD recipe of main_sbdm.m:
%         "tied", Adam lr 1e-3 (constant), batch 32, 15000*S its,
%         unconditional (sigma absorbed into lambda, sigmaIn = 1).
%           S1 : 1 stage,  32 filters 5x5  (834 params)
%           S3 : 3 stages, 32 filters 5x5  (2502 params)
%           L  : 5 stages, 126 filters 7x7 (31510 params; SA b18: 31583)
%         Crops and noise: rng(1)/gpurng(1), init, then the CPU stream is
%         set to the state after the SA b18 build (as fcn_szsuptrain), so
%         its 1-1500 use the same crops and noise as the SA runs.
%         Also: SA b = 1 at sigma 5, 2 (missing so far), as the size sweep.
%         Evaluation: 256x256 centre crops kodim09/15/19/23, noise
%         rng(image index), PSNR/SSIM as fcn_dcsupeval (no clipping).
% Part 2  Guarantees: fcn_denoiserdiag (unchanged; v0 = rand) and
%         fcn_tgdenoiserdiag (v0 = the noisy test image), 256x256, nProbe 4,
%         nPower 60 (as main_sbdm.m), for TNRD S1/S3/L and SA b18/b3/b1 with
%         the angles estimated from the noisy test image and then frozen.
%         Info: SA b18 with the angles NOT frozen (the trained net itself).
% Part 3  RED deblurring as main_sbdm.m: 9x9 Gaussian, std 2, circular,
%         noise 5/255, eps = 0, 300 its, sigma(t) 25/255 -> 2/255 (log).
%         256x256 centre crops kodim09/15/19/23. The models are fixed-sigma,
%         so step t uses the model whose sigma is nearest to sigma(t) (log
%         scale), the same rule for all methods. SA b18: angles estimated
%         once from the observation (by each sigma model), then frozen.
%         eta, beta in {0.5, 1, 2}. Clipping to [0,1] as main_sbdm.m; extra
%         runs without clipping for SA b18 and TNRD L.
%
% New files only. Caches: data/sbdm_tg_*.mat. Logs/summaries:
%   logs/tnrdguar.log, logs/tnrdguar_part{1,2,3}_summary.txt
%   nohup matlab -licmode onlinelicensing -batch "main_sbdm_salsun_tnrdguar" \
%       > ~/Documents/SBDMwLSUN/logs/tnrdguar.log 2>&1 &
% Quick test of all code paths (tiny settings, separate folder):
%   matlab -licmode onlinelicensing -batch "onTest = true; onTestDir = '<dir>'; main_sbdm_salsun_tnrdguar"

here = fileparts(mfilename('fullpath'));
cd(fullfile(here,'..','..')); setpath; cd(here)
addpath(fullfile(here,'..','sbdm'))     % helpers (called, not edited)
addpath(fullfile(here,'..','salsun'))   % maskLayer
warning('off','MATLAB:rmpath:DirNotFound')
rmpath(fullfile('/home/shogo/Workspace/GitHub/SaivDr','mexcodes'))
warning('on','MATLAB:rmpath:DirNotFound')

% ---- settings ----
cfg0 = fcn_ssconfig();                 % imgIdx 9/15/19/23, crop 256, stride 4, ovlp 3, seedBase 0
cfg0.selImg = 9;
cfg0.sigmas = [25 10 5 2];
cfg0.sup = struct('nIters',1500,'mb',32,'lrN',[5e-3 5e-5],'lrEst',[1e-3 1e-5],...
    'lrShr',[5e-3 5e-5],'seed',1,'patch',[64 64],'nCrops',256,'trainIdx',1:8,'ckptEvery',200);
TN = struct('name',{"S1","S3","L"},'S',{1,3,5},'P',{32,32,126},'fs',{5,5,7});
tc = struct('nItPerStage',15000,'lr',1e-3,'mb',32,'ckptEvery',2000);   % main_sbdm.m nItersConv, lrConv, mbSize
dg = struct('nProbe',4,'nPower',60,'seedBase',500);                     % main_sbdm.m nProbe, nPower
rd = struct('blurSize',9,'blurStd',2.0,'sigmaW',5/255,'nIters',300,'sgMax',25/255,'sgMin',2/255,...
    'etas',[0.5 1 2],'betas',[0.5 1 2],'seedBase',1000);
RDM = struct('name',{"SA b18","TNRD L","TNRD S3"},'kind',{"SA","T","T"},'tn',{"","L","S3"},'noclip',{true,true,false});
bList = [18 3 1];
dat = fullfile(here,'..','..','..','data');
ctx.dat = dat; ctx.cdir = dat;
ctx.sdir = fullfile(here,'..','..','..','..','logs');
ctx.deadline = datetime('now') + hours(10);
ctx.margin = 1.10;
isTest = exist('onTest','var') && onTest;
nItCheck = 20;
if isTest
    tc.nItPerStage = 4; tc.ckptEvery = 2;
    cfg0.sup.nIters = 3; cfg0.sup.ckptEvery = 2; nItCheck = 4;
    dg.nProbe = 1; dg.nPower = 2;
    rd.nIters = 4; rd.etas = [0.5 2]; rd.betas = [2];
    ctx.cdir = onTestDir; ctx.sdir = onTestDir;
    if ~isfolder(onTestDir), mkdir(onTestDir); end
end
ctx.useGpu = canUseGPU;
% Deterministic GPU algorithms: without them Sensei's fcn_trainconvtnrd is
% not reproducible on this GPU (two identical 20-it runs differ, max rel.
% 5e-5, checked 2026-11-06), so a resumed run could not equal an unbroken one.
if ctx.useGpu, deep.gpu.deterministicAlgorithms(true); end
fprintf("deterministic GPU algorithms: %d\n",ctx.useGpu && deep.gpu.deterministicAlgorithms);
ctx.cfg0 = cfg0; ctx.tc = tc; ctx.TN = TN;
ctx.ST = containers.Map();
ctx.ST('skips') = strings(0,1);
fprintf("%s  start. GPU: %d. Deadline %s. Test mode: %d\n",string(datetime('now')),ctx.useGpu,string(ctx.deadline),isTest)
disp(cfg0); disp(cfg0.sup); disp(tc); disp(dg); disp(rd)
tRun0 = datetime('now');

%% ===== training data (as stage B) =====
rng(1); if ctx.useGpu, gpurng(1); end
[~,trainSet] = fcn_loadkodak(dat,cfg0.sup.trainIdx,cfg0.sup.patch,cfg0.sup.nCrops,ctx.useGpu);
fprintf("\n%s  training patches: %d of size %s\n",string(datetime('now')),numel(trainSet),mat2str(cfg0.sup.patch))

%% ===== CPU rng state after the b = 18 build (as fcn_dcsuptrain) =====
rng(cfg0.sup.seed,'twister');
ob18 = struct('estimator',"net",'leak',false,'bottleneck',18,...
    'shrink',true,'dcShrink',false,'dcInit',"id",'sigma',single(25/255));
fcn_dcbuildsalsun(cfg0.sup.patch,cfg0.stride,cfg0.ovlp,ob18);
ctx.rng18 = rng;

%% ===== parameter counts =====
for j = 1:numel(TN)
    p = fcn_convtnrdinitprm(TN(j).S,TN(j).P,TN(j).fs,"tied");
    TN(j).np = countT(p);
    fprintf("TNRD %-2s: %d stages, %d filters %dx%d -> params %d (Wa %d, ba %d, bs %d, loglam %d), its %d\n",...
        TN(j).name,TN(j).S,TN(j).P,TN(j).fs,TN(j).fs,TN(j).np,sum(cellfun(@numel,p.Wa)),...
        sum(cellfun(@numel,p.ba)),sum(cellfun(@numel,p.bs)),sum(cellfun(@numel,p.loglam)),TN(j).S*tc.nItPerStage);
end
ctx.TN = TN;

%% ===== checks (sigma 25) =====
CHK = struct('name',{},'vNew',{},'vRef',{},'ok',{},'info',{});
allOk = true;
try
    SB = load(fullfile(dat,'sbdm_salsun_dc_sup_s25_SAR_it1500.mat'));
    h1Ref = SB.q.tr.hist(1);
    sg = single(25/255);
    % 1: first mini-batch after (rng(1), gpurng(1), TNRD init, rng18) = the
    %    first batch of the SA b18 training: SA b18 initial net on it gives
    %    the stage B first loss. 2: the TNRD trainer sees that batch.
    rng(cfg0.sup.seed,'twister');
    nets0 = fcn_dcbuildsalsun(cfg0.sup.patch,cfg0.stride,cfg0.ovlp,ob18);
    net0 = nets0.shrink; if ctx.useGpu, net0 = dlupdate(@gpuArray,net0); end
    for j = 1:numel(TN)
        prm0 = tninit(TN(j),ctx);
        [xb,vb] = fcn_minibatch(trainSet,sg,false,cfg0.sup.mb);
        lSA = double(gather(extractdata(dlfeval(@lossSA,net0,vb,xb))));
        CHK(end+1) = mkchk(sprintf("TNRD %s: first batch, SA b18 init loss vs stage B hist(1)",TN(j).name),lSA,h1Ref); %#ok<SAGROW>
        lT = double(gather(extractdata(dlfeval(@fcn_convtnrdloss,prm0,vb,xb,single(1)))));
        prm0 = tninit(TN(j),ctx);
        [~,hT] = fcn_tgtrainconvtnrd(prm0,trainSet,sg,1,tc.lr,0,tc.mb,false);
        CHK(end+1) = mkchk(sprintf("TNRD %s: trainer hist(1) vs TNRD loss on that batch",TN(j).name),hT(1),lT); %#ok<SAGROW>
    end
    % 3: trainer copy = Sensei's fcn_trainconvtnrd (bitwise, nItCheck its)
    j = 2;
    prm0 = tninit(TN(j),ctx);
    [pA,hA] = fcn_trainconvtnrd(prm0,trainSet,sg,nItCheck,tc.lr,0,tc.mb,false);
    prm0 = tninit(TN(j),ctx);
    [pB,hB] = fcn_tgtrainconvtnrd(prm0,trainSet,sg,nItCheck,tc.lr,0,tc.mb,false);
    c = mkchk(sprintf("TNRD S3: fcn_tgtrainconvtnrd vs fcn_trainconvtnrd, %d its, history",nItCheck),hB(end),hA(end));
    c.ok = isequal(hA(:),hB(:)) && eqprm(pA,pB);
    c.info = "history and final parameters bitwise equal"; CHK(end+1) = c;
    % 4: resume (stopped twice by the time budget) = unbroken run
    ck = fullfile(ctx.cdir,'sbdm_tg_check_resume_ckpt.mat');
    if isfile(ck), delete(ck); end
    prm0 = tninit(TN(j),ctx);
    [~,~,d1] = fcn_tgtrainconvtnrd(prm0,trainSet,sg,nItCheck,tc.lr,0,tc.mb,false,ck,0,inf);
    rng(12345); if ctx.useGpu, gpurng(12345); end           % disturb the streams
    [~,~,d2] = fcn_tgtrainconvtnrd(prm0,trainSet,sg,nItCheck,tc.lr,0,tc.mb,false,ck,0,inf);
    rng(54321); if ctx.useGpu, gpurng(54321); end
    [pC,hC,d3] = fcn_tgtrainconvtnrd(prm0,trainSet,sg,nItCheck,tc.lr,0,tc.mb,false,ck,inf,inf);
    delete(ck);
    c = mkchk(sprintf("TNRD S3: resumed run (stops after its 1, 2) vs unbroken, %d its",nItCheck),hC(end),hB(end));
    c.ok = ~d1 && ~d2 && d3 && isequal(hC(:),hB(:)) && eqprm(pC,pB);
    c.info = "history and final parameters bitwise equal"; CHK(end+1) = c;
    % 5: SA b1 trainer (as the size sweep): first loss = stage B SA b18
    oc = cfg0.sup; oc.nIters = 3; oc.ckptFile = '';
    oc.sigma = sg; oc.leak = false; oc.dcShrink = false;
    oc.stride = cfg0.stride; oc.ovlp = cfg0.ovlp; oc.bottleneck = 1; oc.rngAfterBuild = ctx.rng18;
    [~,trc] = fcn_szsuptrain("SA",trainSet,oc);
    CHK(end+1) = mkchk("SA R b1, fcn_szsuptrain: first loss vs stage B SA b18",trc.hist(1),h1Ref);
    ctx.ST('saB1') = trc.tPerIt*cfg0.sup.nIters + 40;
catch ME
    fprintf("CHECK FAILED (error): %s\n",getReport(ME,'extended','hyperlinks','off'))
    allOk = false;
end
for c = CHK
    fprintf("CHECK %-75s new %.10e  ref %.10e  ok %d  %s\n",c.name,c.vNew,c.vRef,c.ok,c.info)
end
allOk = allOk && ~isempty(CHK) && all([CHK.ok]);
if ~allOk
    fprintf("\n%s  A check failed. The runs are NOT started.\n",string(datetime('now')))
    save(fullfile(ctx.cdir,'sbdm_tg_checks_failed.mat'),'CHK')
    return
end
fprintf("All %d checks passed.\n",numel(CHK))

%% ===== speed (for the estimates) =====
for j = 1:numel(TN)
    prm0 = tninit(TN(j),ctx);
    fcn_tgtrainconvtnrd(prm0,trainSet,single(25/255),5,tc.lr,0,tc.mb,false);
    if ctx.useGpu, wait(gpuDevice); end
    t = tic; fcn_tgtrainconvtnrd(prm0,trainSet,single(25/255),50,tc.lr,0,tc.mb,false);
    if ctx.useGpu, wait(gpuDevice); end
    spi = toc(t)/50;
    ctx.ST(char("tnrd" + TN(j).name)) = spi*TN(j).S*tc.nItPerStage + 20;
    fprintf("speed TNRD %-2s: %.4f s/it -> %.1f min per training\n",TN(j).name,spi,spi*TN(j).S*tc.nItPerStage/60)
end
e1 = 0;
for sgi = cfg0.sigmas
    for j = 1:numel(TN)
        if ~isfile(ftn(ctx,sgi,TN(j))), e1 = e1 + estget(ctx.ST,"tnrd" + TN(j).name); end
    end
end
for sgi = [5 2]
    if ~isfile(fb1(ctx,sgi)), e1 = e1 + estget(ctx.ST,"saB1"); end
end
fprintf("\n%s  Estimate part 1: %.1f min -> would end %s\n",string(datetime('now')),e1/60,string(datetime('now') + seconds(e1)))

%% ===== Part 1: training + denoising =====
QT = cell(numel(cfg0.sigmas),numel(TN));
for s = 1:numel(cfg0.sigmas)
    for j = 1:numel(TN)
        sgi = cfg0.sigmas(s);
        lab = sprintf("TNRD %s s%02d",TN(j).name,sgi);
        f = ftn(ctx,sgi,TN(j));
        QT{s,j} = unit(ctx,lab,f,"tnrd" + TN(j).name,@() runtnrd(TN(j),sgi,trainSet,ctx,string(f) + "_ckpt.mat"));
    end
end
QB1 = cell(1,2); sgB1 = [5 2];
for s = 1:2
    f = fb1(ctx,sgB1(s));
    QB1{s} = unit(ctx,sprintf("SA R b1 s%02d",sgB1(s)),f,"saB1",@() runb1(sgB1(s),trainSet,ctx,string(f) + "_ckpt.mat"));
end
P1 = [];
try
    P1 = collect1(ctx,QT,TN,bList);
catch ME
    fprintf("collect1 FAILED: %s\n",getReport(ME,'extended','hyperlinks','off'))
end
safesummary(@() summary1(fullfile(ctx.sdir,'tnrdguar_part1_summary.txt'),P1,CHK,ctx,TN,tRun0));
save(fullfile(ctx.cdir,'sbdm_tg_part1_results.mat'),'P1','CHK','cfg0','tc')
clear trainSet nets0 net0
tP1 = datetime('now');

%% ===== Part 2: guarantees =====
D2 = cell(numel(cfg0.sigmas),numel(cfg0.imgIdx));
ctx.ST('diag') = 300;
for s = 1:numel(cfg0.sigmas)
    sgi = cfg0.sigmas(s);
    M = loadmodels(ctx,sgi,TN,bList);
    B = [];
    for i = 1:numel(cfg0.imgIdx)
        f = fullfile(ctx.cdir,sprintf('sbdm_tg_diag_s%02d_kodim%02d.mat',sgi,cfg0.imgIdx(i)));
        if ~isfile(f) && isempty(B), B = buildsa(ctx,sgi,bList,cfg0.cropSize); end
        D2{s,i} = unit(ctx,sprintf("diag s%02d kodim%02d",sgi,cfg0.imgIdx(i)),f,"diag",...
            @() rundiag(ctx,sgi,cfg0.imgIdx(i),M,B,TN,bList,dg));
    end
end
safesummary(@() summary2(fullfile(ctx.sdir,'tnrdguar_part2_summary.txt'),D2,ctx,TN,bList,dg));
save(fullfile(ctx.cdir,'sbdm_tg_part2_results.mat'),'D2','dg')
tP2 = datetime('now');

%% ===== Part 3: RED deblurring =====
blurKernel = fspecial('gaussian',rd.blurSize,rd.blurStd);
Hf  = @(z) imfilter(z,blurKernel,'circular');
Htf = @(z) imfilter(z,rot90(blurKernel,2),'circular');
schedule = exp(linspace(log(rd.sgMax),log(rd.sgMin),rd.nIters));
sgBank = cfg0.sigmas/255;
fprintf("\nRED model bank: sigma %s; iterations per model %s\n",mat2str(cfg0.sigmas),...
    mat2str(accumarray(nearestk(schedule,sgBank),1,[numel(sgBank) 1]).'))
MB = cell(1,numel(cfg0.sigmas));
for s = 1:numel(cfg0.sigmas), MB{s} = loadmodels(ctx,cfg0.sigmas(s),TN,18); end
BB = [];
ctx.ST('redSA') = 9*rd.nIters*0.12 + 60; ctx.ST('redT') = 9*rd.nIters*0.03 + 10; ctx.ST('redNone') = 30;
R3 = struct('img',{},'method',{},'clip',{},'runs',{});
for i = 1:numel(cfg0.imgIdx)
    idx = cfg0.imgIdx(i);
    [X,V] = redobs(ctx,idx,Hf,rd);
    fprintf("\nkodim%02d: observation PSNR %.4f dB (clipped to [0,1] for the measure)\n",idx,psnr(min(max(V,0),1),X))
    % no regularisation (beta = 0), reference
    f = fullfile(ctx.cdir,sprintf('sbdm_tg_red_kodim%02d_none.mat',idx));
    q = unit(ctx,sprintf("RED kodim%02d none",idx),f,"redNone",@() runrednone(X,V,Hf,Htf,schedule,rd));
    R3(end+1) = struct('img',idx,'method',"none",'clip',true,'runs',{q}); %#ok<SAGROW>
    for m = 1:numel(RDM)
        for clip = [true false]
            if ~clip && ~RDM(m).noclip, continue, end
            f = fullfile(ctx.cdir,sprintf('sbdm_tg_red_kodim%02d_%s_c%d.mat',idx,strrep(RDM(m).name," ",""),clip));
            if ~isfile(f) && RDM(m).kind == "SA" && isempty(BB), BB = buildsa(ctx,[],18,cfg0.cropSize); end
            key = "redT"; if RDM(m).kind == "SA", key = "redSA"; end
            q = unit(ctx,sprintf("RED kodim%02d %s clip %d",idx,RDM(m).name,clip),f,key,...
                @() runred(ctx,RDM(m),MB,BB,X,V,Hf,Htf,schedule,sgBank,rd,clip));
            R3(end+1) = struct('img',idx,'method',RDM(m).name,'clip',clip,'runs',{q}); %#ok<SAGROW>
        end
    end
end
safesummary(@() summary3(fullfile(ctx.sdir,'tnrdguar_part3_summary.txt'),R3,ctx,rd,RDM,MB,schedule,sgBank));
R3s = R3;
for k = 1:numel(R3s)   % results file without the estimates (small)
    if ~isempty(R3s(k).runs) && isfield(R3s(k).runs,'R'), R3s(k).runs.R = rmfield(R3s(k).runs.R,'est'); end
end
save(fullfile(ctx.cdir,'sbdm_tg_part3_results.mat'),'R3s','rd','schedule','sgBank','RDM')

sk = ctx.ST('skips');
fprintf("\nTimes: part 1 %s -> %s, part 2 -> %s, part 3 -> %s\n",string(tRun0),string(tP1),string(tP2),string(datetime('now')))
fprintf("\nSkipped / failed / missing units (%d):\n",numel(sk))
fprintf("  %s\n",sk)
fprintf("\n%s  done.\n",string(datetime('now')))

%% ===================== Local functions =====================
function n = countT(p)
n = sum(cellfun(@numel,p.Wa)) + sum(cellfun(@numel,p.ba)) + sum(cellfun(@numel,p.bs)) + sum(cellfun(@numel,p.loglam));
end

function prm = tninit(T,ctx)
% seeds as the SA trainers (rng(1), gpurng(1)), TNRD init (CPU stream),
% then the CPU stream state after the SA b18 build (crops as the SA runs)
rng(ctx.cfg0.sup.seed,'twister');
if ctx.useGpu, gpurng(ctx.cfg0.sup.seed); end
prm = fcn_convtnrdinitprm(T.S,T.P,T.fs,"tied");
rng(ctx.rng18);
if ctx.useGpu, prm = dlupdate(@gpuArray,prm); end
end

function [loss] = lossSA(net,v,x)
xh = forward(net,v);
loss = mean((x - xh).^2,'all');
end

function ok = eqprm(a,b)
ok = true;
for f = string(fieldnames(a)).'
    for k = 1:numel(a.(f))
        ok = ok && isequal(gather(extractdata(a.(f){k})),gather(extractdata(b.(f){k})));
    end
end
end

function c = mkchk(name,vNew,vRef)
c = struct('name',string(name),'vNew',vNew,'vRef',vRef,'ok',vNew == vRef,'info',"equal");
end

function f = ftn(ctx,sg,T)
f = fullfile(ctx.cdir,sprintf('sbdm_tg_tnrd_s%02d_%s_it%d.mat',sg,T.name,T.S*ctx.tc.nItPerStage));
end

function f = fb1(ctx,sg)
f = fullfile(ctx.cdir,sprintf('sbdm_tg_sa_s%02d_SAR_b1_it%d.mat',sg,ctx.cfg0.sup.nIters));
end

function q = runtnrd(T,sg,trainSet,ctx,ckpt)
tc = ctx.tc; cfg = ctx.cfg0; cfg.sigma = single(sg/255);
t = tic;
prm0 = tninit(T,ctx);
nIt = T.S*tc.nItPerStage;
[prm,hist,done] = fcn_tgtrainconvtnrd(prm0,trainSet,cfg.sigma,nIt,tc.lr,0,tc.mb,true,char(ckpt),inf,tc.ckptEvery);
if ~done, error("training not finished"); end
if ctx.useGpu, wait(gpuDevice); end
q.tTrain = toc(t); q.tPerIt = q.tTrain/nIt;
q.prm = dlupdate(@gather,prm); q.hist = hist; q.T = T; q.sg = sg; q.np = countT(q.prm);
nb = min(100,numel(hist)); q.lossEnd = mean(hist(end-nb+1:end));
q.lam = cellfun(@(a) double(exp(extractdata(a))),q.prm.loglam);
fprintf("    trained: %.1f s (%.4f s/it), loss (mean of last %d its) %.4e, params %d, lambda %s\n",...
    q.tTrain,q.tPerIt,nb,q.lossEnd,q.np,mat2str(q.lam,4))
for i = 1:numel(cfg.imgIdx)
    [x,v] = fcn_ssloadimage(ctx.dat,cfg.imgIdx(i),cfg,ctx.useGpu);
    te = tic;
    z = gather(extractdata(fcn_convtnrddenoise(prm,dlarray(v,'SSCB'),single(1))));
    r.psnr = psnr(z,gather(x)); r.ssim = ssim(z,gather(x));
    r.mse = double(mean((z - gather(x)).^2,'all')); r.tEval = toc(te); r.img = single(z);
    q.res(i) = r;
    fprintf("    kodim%02d  %.4f dB  SSIM %.4f\n",cfg.imgIdx(i),r.psnr,r.ssim)
end
q.tTotal = toc(t);
if isfile(ckpt), delete(ckpt); end
end

function q = runb1(sg,trainSet,ctx,ckpt)
% as runsz in main_sbdm_salsun_supsize.m (b = 1)
cfg = ctx.cfg0; cfg.sigma = single(sg/255); cfg.bottleneck = 1;
o = cfg.sup;
o.sigma = cfg.sigma; o.leak = false; o.dcShrink = false;
o.stride = cfg.stride; o.ovlp = cfg.ovlp; o.bottleneck = 1;
o.rngAfterBuild = ctx.rng18;
o.ckptFile = char(ckpt);
t = tic;
[mdl,tr] = fcn_szsuptrain("SA",trainSet,o);
q.mdl = mdl; q.tr = tr; q.variant = "R"; q.kind = "SA"; q.b = 1;
nb = max(1,min(100,numel(tr.hist)));
q.lossEnd = mean(tr.hist(end-nb+1:end));
fprintf("    trained: %.1f s (%.3f s/it), loss (mean of last %d its) %.4e, params %d\n",tr.tTrain,tr.tPerIt,nb,q.lossEnd,tr.nParams)
ob = struct('estimator',"net",'leak',false,'bottleneck',1,...
    'shrink',true,'dcShrink',false,'dcInit',"id",'sigma',cfg.sigma);
netsSz = fcn_dcbuildsalsun(cfg.cropSize,cfg.stride,cfg.ovlp,ob);
for i = 1:numel(cfg.imgIdx)
    [x,v] = fcn_ssloadimage(ctx.dat,cfg.imgIdx(i),cfg,ctx.useGpu);
    r = fcn_dcsupeval(mdl,x,v,cfg,netsSz);
    q.res(i) = r;
    fprintf("    kodim%02d  %.4f dB  SSIM %.4f  prErrX %.2e\n",cfg.imgIdx(i),r.psnr,r.ssim,r.prErrX)
end
q.tTotal = toc(t);
if isfile(ckpt), delete(ckpt); end
end

function M = loadmodels(ctx,sg,TN,bList)
% cached SA (stage B / run A / size sweep / this run) and TNRD (part 1)
M = struct();
f = fullfile(ctx.dat,sprintf('sbdm_salsun_dc_sup_s%02d_NR_it1500.mat',sg));
if isfile(f), S = load(f); M.N = S.q; else, M.N = []; end
for b = bList
    switch b
        case 18, f = fullfile(ctx.dat,sprintf('sbdm_salsun_dc_sup_s%02d_SAR_it1500.mat',sg));
        case 3,  f = fullfile(ctx.dat,sprintf('sbdm_salsun_supsize_s%02d_SAR_b3_it1500.mat',sg));
        case 1
            f = fullfile(ctx.dat,sprintf('sbdm_salsun_supsize_s%02d_SAR_b1_it1500.mat',sg));
            if ~isfile(f), f = fb1(ctx,sg); end
    end
    if isfile(f)
        S = load(f); M.("SA" + b) = S.q;
    else
        M.("SA" + b) = []; addskip(ctx,sprintf("MISSING model %s",f));
    end
end
for j = 1:numel(TN)
    f = ftn(ctx,sg,TN(j));
    if isfile(f)
        S = load(f); M.("T" + TN(j).name) = S.q;
    else
        M.("T" + TN(j).name) = []; addskip(ctx,sprintf("MISSING model %s",f));
    end
end
end

function B = buildsa(ctx,sg,bList,sz)
% SA nets of the image size: estimator/shrink net per b (sigma sg) and the
% constant-angle shrink net per sigma (Sigma is a layer property)
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
% trained SA net (angles from its input) and the same net with the angles
% estimated from V and then frozen (constant per-block angles)
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

function q = rundiag(ctx,sg,idx,M,B,TN,bList,dg)
cfg = ctx.cfg0; cfg.sigma = single(sg/255); sz = cfg.cropSize;
[x,v] = fcn_ssloadimage(ctx.dat,idx,cfg,ctx.useGpu); %#ok<ASGLU>
V = dlarray(v,'SSCB');
gz = zeros(1,'single'); if ctx.useGpu, gz = gpuArray(gz); end
q = struct('model',{},'asymRand',{},'rhoRand',{},'asymImg',{},'rhoImg',{},'dFrozen',{},'kappaMax',{},'psnr',{},'t',{});
for j = 1:numel(TN)
    Q = M.("T" + TN(j).name);
    if isempty(Q), continue, end
    prm = Q.prm; if ctx.useGpu, prm = dlupdate(@gpuArray,prm); end
    f = @(z) fcn_convtnrddenoise(prm,z + gz,single(1));
    q(end+1) = measure(f,"TNRD " + TN(j).name,v,x,sz,dg,idx,NaN,NaN); %#ok<AGROW>
end
for b = bList
    Q = M.("SA" + b);
    if isempty(Q), continue, end
    [cs,snet,dmax,kap] = safrozen(Q.mdl,B.("est" + b + "_" + sg),B.("const" + sg),V,ctx.useGpu);
    f = @(z) predict(cs,z + gz);
    q(end+1) = measure(f,"SA b" + b + " frozen",v,x,sz,dg,idx,dmax,max(kap)); %#ok<AGROW>
    if b == 18
        f = @(z) predict(snet,z + gz);
        q(end+1) = measure(f,"SA b18 not frozen",v,x,sz,dg,idx,NaN,max(kap)); %#ok<AGROW>
    end
end
for k = 1:numel(q)
    fprintf("    %-18s rand v0: asym %.3e rho %.6f | v0 = v: asym %.3e rho %.6f | PSNR at v %.4f | frozen-unfrozen %.2e | kappaMax %.4f (%.1f s)\n",...
        q(k).model,q(k).asymRand,q(k).rhoRand,q(k).asymImg,q(k).rhoImg,q(k).psnr,q(k).dFrozen,q(k).kappaMax,q(k).t)
end
q = struct('m',q);
end

function r = measure(f,name,v,x,sz,dg,idx,dmax,kmax)
t = tic;
rng(dg.seedBase + idx,'twister');
d1 = fcn_denoiserdiag(f,[],sz,dg.nProbe,dg.nPower);
rng(dg.seedBase + idx,'twister');
d2 = fcn_tgdenoiserdiag(f,sz,dg.nProbe,dg.nPower,v);
z = gather(extractdata(f(dlarray(v,'SSCB'))));
r = struct('model',name,'asymRand',d1.jacAsym,'rhoRand',d1.jacRho,'asymImg',d2.jacAsym,...
    'rhoImg',d2.jacRho,'dFrozen',dmax,'kappaMax',kmax,'psnr',psnr(z,gather(x)),'t',toc(t));
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

function q = runrednone(X,V,Hf,Htf,schedule,rd)
fb = {@(z) z};   % beta = 0: the denoiser is not used
R = struct([]);
for e = rd.etas
    r = fcn_tgredrun(fb,schedule(1),V,X,Hf,Htf,schedule,e,0,rd.nIters,true);
    r.eta = e; r.beta = 0;
    R = [R r]; %#ok<AGROW>
    fprintf("    none eta %.1f: PSNR %.4f SSIM %.4f res(T) %.3e\n",e,r.psnrEnd,r.ssimEnd,r.res(end))
end
q.R = R;
end

function q = runred(ctx,Mth,MB,BB,X,V,Hf,Htf,schedule,sgBank,rd,clip)
nS = numel(sgBank); fb = cell(1,nS); q.dFrozen = nan(1,nS); q.kappaMax = nan(1,nS);
dev = @(z) z; if ctx.useGpu, dev = @(z) gpuArray(z); end
for s = 1:nS
    sg = round(sgBank(s)*255);
    if Mth.kind == "SA"
        Q = MB{s}.SA18;
        Vd = dlarray(dev(single(V)),'SSCB');
        [cs,~,dmax,kap] = safrozen(Q.mdl,BB.("est18_" + sg),BB.("const" + sg),Vd,ctx.useGpu);   % angles from the observation
        q.dFrozen(s) = dmax; q.kappaMax(s) = max(kap);
        fb{s} = @(z) double(gather(extractdata(predict(cs,dlarray(dev(single(z)),'SSCB')))));
    else
        Q = MB{s}.("T" + Mth.tn);
        prm = Q.prm; if ctx.useGpu, prm = dlupdate(@gpuArray,prm); end
        fb{s} = @(z) double(gather(extractdata(fcn_convtnrddenoise(prm,dlarray(dev(single(z)),'SSCB'),single(1)))));
    end
end
R = struct([]);
for e = rd.etas
    for b = rd.betas
        t = tic;
        r = fcn_tgredrun(fb,sgBank,V,X,Hf,Htf,schedule,e,b,rd.nIters,clip);
        r.eta = e; r.beta = b; r.t = toc(t);
        R = [R r]; %#ok<AGROW>
        fprintf("    %-7s clip %d eta %.1f beta %.1f: PSNR %.4f SSIM %.4f res(T) %.3e blowup %d (%.1f s)\n",...
            Mth.name,clip,e,b,r.psnrEnd,r.ssimEnd,r.res(end),r.blowup,r.t)
    end
end
q.R = R;
end

%% ----- units, skips -----
function safesummary(fn)
try
    fn();
catch ME
    fprintf("Summary FAILED: %s\n",ME.message)
    fprintf("%s\n",getReport(ME,'extended','hyperlinks','off'))
end
end

function e = estget(ST,key)
if isKey(ST,char(key)), e = ST(char(key)); return, end
e = 600;
end

function q = unit(ctx,lab,f,key,fn)
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

function y = pickres(q,fld,nI)
y = nan(nI,1);
if isempty(q) || ~isfield(q,'res'), return, end
for i = 1:min(nI,numel(q.res))
    try, y(i) = double(q.res(i).(fld)); catch, y(i) = NaN; end
end
end

%% ----- part 1 -----
function P1 = collect1(ctx,QT,TN,bList)
cfg = ctx.cfg0; nI = numel(cfg.imgIdx);
P1 = struct('sg',{},'names',{},'P',{},'S',{},'np',{},'loss',{},'tTrain',{},'tPerIt',{},'its',{},'lam',{});
for s = 1:numel(cfg.sigmas)
    sg = cfg.sigmas(s);
    M = loadmodels(ctx,sg,TN([]),bList);
    names = ["N" "SA b" + bList "TNRD " + [TN.name]];
    nM = numel(names);
    T.sg = sg; T.names = names;
    T.P = nan(nI,nM); T.S = nan(nI,nM); T.np = nan(1,nM); T.loss = nan(1,nM);
    T.tTrain = nan(1,nM); T.tPerIt = nan(1,nM); T.its = nan(1,nM); T.lam = cell(1,nM);
    Ms = [{M.N} arrayfun(@(b) M.("SA" + b),bList,'UniformOutput',false) QT(s,:)];
    for j = 1:nM
        q = Ms{j};
        if isempty(q), continue, end
        T.P(:,j) = pickres(q,'psnr',nI); T.S(:,j) = pickres(q,'ssim',nI);
        T.loss(j) = q.lossEnd;
        if j <= 1 + numel(bList)
            T.np(j) = q.tr.nParams; T.tTrain(j) = q.tr.tTrain; T.tPerIt(j) = q.tr.tPerIt; T.its(j) = numel(q.tr.hist);
        else
            T.np(j) = q.np; T.tTrain(j) = q.tTrain; T.tPerIt(j) = q.tPerIt; T.its(j) = numel(q.hist); T.lam{j} = q.lam;
        end
    end
    P1(end+1) = T; %#ok<AGROW>
end
end

function summary1(file,P1,CHK,ctx,TN,tRun0)
fid = fopen(file,'w'); cleaner = onCleanup(@() fclose(fid));
cfg = ctx.cfg0; o = cfg.sup; tc = ctx.tc;
out(fid,"Part 1: supervised TNRD vs supervised SA-LSUN (variant R) and N. Written %s (run start %s).\n",string(datetime('now')),string(tRun0));
out(fid,"Training data: %d patches %dx%d of kodim%02d-%02d (rng(1)), fixed sigma per model, fresh noise each it, MSE.\n",...
    o.nCrops*numel(o.trainIdx),o.patch(1),o.patch(2),o.trainIdx(1),o.trainIdx(end));
out(fid,"TNRD (examples/sbdm, 'tied', unconditional, sigmaIn = 1): Adam lr %g constant, batch %d, %d its per stage (main_sbdm.m recipe).\n",tc.lr,tc.mb,tc.nItPerStage);
for T = TN
    out(fid,"  TNRD %-2s: %d stages, %d filters %dx%d, params %d, its %d\n",T.name,T.S,T.P,T.fs,T.fs,T.np,T.S*tc.nItPerStage);
end
out(fid,"  Crops/noise: rng(1)/gpurng(1), init, CPU stream := state after the SA b18 build -> its 1-1500 = the same crops and noise as the SA runs.\n");
out(fid,"SA b18 / b3 / b1 and N: cached models (stage B, run A, size sweep); SA b1 at sigma 5, 2 trained in this run (fcn_szsuptrain, as the size sweep).\n");
out(fid,"Evaluation: 256x256 centre crops, kodim09/15/19/23, noise rng(image index), no clipping (as fcn_dcsupeval). mean(test) = kodim15/19/23.\n");
out(fid,"\nChecks before the runs:\n");
for c = CHK
    out(fid,"  %s: new %.10e  ref %.10e  ok %d (%s)\n",c.name,c.vNew,c.vRef,c.ok,c.info);
end
if isempty(P1), out(fid,"\n(no results)\n"); return, end
imgs = cfg.imgIdx(:); rows = compose("kodim%02d",imgs);
iTest = find(imgs ~= cfg.selImg);
rowsM = [rows; "mean(all)"; "mean(test)"];
addm = @(M) [M; mean(M,1,'omitnan'); mean(M(iTest,:),1,'omitnan')];
pairs = {"SA b18","TNRD L"; "SA b3","TNRD S3"; "SA b1","TNRD S1"};
DP = nan(numel(P1),3); DS = DP;
for k = 1:numel(P1)
    T = P1(k); names = T.names;
    out(fid,"\n==================== sigma = %d/255 ====================\n",T.sg);
    out(fid,"\nParameters and training (loss = mean of the last 100 its, MSE per pixel)\n");
    for j = 1:numel(names)
        lam = ""; if ~isempty(T.lam{j}), lam = "  lambda " + mat2str(T.lam{j},4); end
        out(fid,"  %-8s params %6s  its %6s  loss %.4e  train %.1f s  %.4f s/it%s\n",names(j),nstr(T.np(j)),nstr(T.its(j)),T.loss(j),T.tTrain(j),T.tPerIt(j),lam);
    end
    printtable(fid,"PSNR [dB]",rowsM,names,addm(T.P),"%.4f")
    printtable(fid,"SSIM",rowsM,names,addm(T.S),"%.4f")
    dP = nan(numel(imgs),3); dS = dP; cn = strings(1,3);
    for p = 1:3
        a = find(names == pairs{p,1}); b = find(names == pairs{p,2});
        dP(:,p) = T.P(:,a) - T.P(:,b); dS(:,p) = T.S(:,a) - T.S(:,b);
        cn(p) = pairs{p,1} + "-" + pairs{p,2};
    end
    printtable(fid,"SA - TNRD, PSNR [dB]",rowsM,cn,addm(dP),"%.4f")
    printtable(fid,"SA - TNRD, SSIM",rowsM,cn,addm(dS),"%.4f")
    am = addm(dP); DP(k,:) = am(end,:); am = addm(dS); DS(k,:) = am(end,:);
    out(fid,"Images where SA beats TNRD (PSNR): test (of %d) %s; all (of %d) %s\n",numel(iTest),...
        mat2str(sum(dP(iTest,:) > 0,1)),numel(imgs),mat2str(sum(dP > 0,1)));
end
sgRows = compose("sigma %d",[P1.sg].');
cn = string(pairs(:,1)) + "-" + string(pairs(:,2));
out(fid,"\n==================== Over sigma (mean(test)) ====================\n");
printtable(fid,"SA - TNRD, PSNR [dB]",sgRows,cn.',DP,"%.4f")
printtable(fid,"SA - TNRD, SSIM",sgRows,cn.',DS,"%.4f")
sk = ctx.ST('skips');
out(fid,"\n---- Skipped / failed / missing units so far (%d) ----\n",numel(sk));
for k = 1:numel(sk), out(fid,"  %s\n",sk(k)); end
end

%% ----- part 2 -----
function summary2(file,D2,ctx,TN,bList,dg)
fid = fopen(file,'w'); cleaner = onCleanup(@() fclose(fid));
cfg = ctx.cfg0;
out(fid,"Part 2: Jacobian asymmetry and spectral radius. Written %s.\n",string(datetime('now')));
out(fid,"256x256, nProbe %d, nPower %d (as main_sbdm.m). Probes: rng(%d + image index) before each measure (same probes for all models).\n",dg.nProbe,dg.nPower,dg.seedBase);
out(fid,"  'rand': fcn_denoiserdiag (unchanged), Jacobian at v0 = rand(256,256).\n");
out(fid,"  'img' : fcn_tgdenoiserdiag (copy), Jacobian at v0 = the noisy test image v at this sigma.\n");
out(fid,"  asym = mean |u'Jw - w'Ju|/(||J'u|| ||w||); rho = power iteration on J'.\n");
out(fid,"SA frozen: angles estimated by the trained estimator from the noisy test image v, then frozen (constant per-block angles).\n");
out(fid,"  'frozen - not frozen' = max |f_frozen(v) - f(v)| at v (should be 0). kappaMax = max_p kappa_p (theory: eig(J) in [1 - kappaMax, 1]).\n");
out(fid,"SA b18 not frozen: the trained net itself (angles depend on the input). Info only.\n");
models = ["TNRD " + [TN.name] "SA b" + bList + " frozen" "SA b18 not frozen"];
cols = ["asym rand" "rho rand" "asym img" "rho img"];
flds = ["asymRand" "rhoRand" "asymImg" "rhoImg"];
imgs = cfg.imgIdx; nI = numel(imgs);
OV = nan(numel(cfg.sigmas),numel(models),4);   % mean over images
OM = OV;                                         % max over images
for s = 1:numel(cfg.sigmas)
    out(fid,"\n==================== sigma = %d/255 ====================\n",cfg.sigmas(s));
    A = nan(numel(models),4,nI); K = nan(numel(models),2,nI); Pz = nan(numel(models),nI);
    for i = 1:nI
        if isempty(D2{s,i}), continue, end
        q = D2{s,i}.m;
        for k = 1:numel(q)
            m = find(models == q(k).model);
            for c = 1:4, A(m,c,i) = q(k).(flds(c)); end
            K(m,:,i) = [q(k).dFrozen q(k).kappaMax]; Pz(m,i) = q(k).psnr;
        end
    end
    for c = 1:4
        printtable(fid,cols(c) + " (per image)",models.',compose("kodim%02d",imgs),squeeze(A(:,c,:)),"%.4e")
    end
    printtable(fid,"Mean over the 4 images",models.',cols,mean(A,3,'omitnan'),"%.4e")
    printtable(fid,"Max over the 4 images",models.',cols,max(A,[],3,'omitnan'),"%.4e")
    printtable(fid,"frozen - not frozen at v (max over images), kappaMax (max over images), PSNR at v (mean over images)",...
        models.',["frozen-unfrozen" "kappaMax" "PSNR at v"],[max(K(:,1,:),[],3,'omitnan') max(K(:,2,:),[],3,'omitnan') mean(Pz,2,'omitnan')],"%.4e")
    OV(s,:,:) = mean(A,3,'omitnan'); OM(s,:,:) = max(A,[],3,'omitnan');
end
out(fid,"\n==================== Over sigma (mean over the 4 images) ====================\n");
sgRows = compose("sigma %d",cfg.sigmas(:));
for c = 1:4
    printtable(fid,cols(c),sgRows,models,squeeze(OV(:,:,c)),"%.4e")
end
out(fid,"\n==================== Over sigma (max over the 4 images) ====================\n");
for c = 1:4
    printtable(fid,cols(c),sgRows,models,squeeze(OM(:,:,c)),"%.4e")
end
sk = ctx.ST('skips');
out(fid,"\n---- Skipped / failed / missing units so far (%d) ----\n",numel(sk));
for k = 1:numel(sk), out(fid,"  %s\n",sk(k)); end
end

%% ----- part 3 -----
function summary3(file,R3,ctx,rd,RDM,MB,schedule,sgBank)
fid = fopen(file,'w'); cleaner = onCleanup(@() fclose(fid));
cfg = ctx.cfg0; imgs = cfg.imgIdx; nI = numel(imgs); N = prod(cfg.cropSize);
out(fid,"Part 3: RED deblurring. Written %s.\n",string(datetime('now')));
out(fid,"Blur %dx%d Gaussian std %.1f (circular), noise %.4f (= 5/255), observation noise rng(%d + image index). 256x256 centre crops.\n",...
    rd.blurSize,rd.blurSize,rd.blurStd,rd.sigmaW,rd.seedBase);
out(fid,"z <- clip(z - eta*(H'(Hz - v) + beta*(z - f(z; sigma(t))))), z(0) = v, eps = 0, %d its, sigma(t) = %g -> %g (log).\n",rd.nIters,rd.sgMax*255,rd.sgMin*255);
k = nearestk(schedule,sgBank);
out(fid,"Model bank (fixed-sigma models, nearest sigma in log scale): its per model sigma %s = %s.\n",mat2str(round(sgBank*255)),mat2str(accumarray(k,1,[numel(sgBank) 1]).'));
out(fid,"SA b18: angles estimated once from the observation v by each sigma model, then frozen.\n");
out(fid,"Residual r(t) = ||z(t) - z(t-1)||_2 (z(0) = v). Shown as RMS per pixel r/sqrt(N), N = %d.\n",N);
out(fid,"Stable := finite, r(T) <= r(T-50) and r(T)/sqrt(N) < 1e-3 (T = %d). (The last %d its use one model, sigma %d.)\n",rd.nIters,sum(k == k(end)),round(sgBank(k(end))*255));
out(fid,"Theory (SA frozen): eig(J) in [1 - kappaMax, 1] and ||H'H|| = 1, so the step map is non-expansive (averaged) if eta*(1 + beta*kappaMax) < 2.\n");
kap = [];
for s = 1:numel(MB)
    if isempty(MB{s}) || ~isfield(MB{s},'SA18') || isempty(MB{s}.SA18), continue, end
    L = MB{s}.SA18.mdl.learn;
    a = double(gather(extractdata(L.Value{strcmp(L.Layer,'Lv1_Shrink') & strcmp(L.Parameter,'A')})));
    kap(end+1) = max(1./(1+exp(-a))); %#ok<AGROW>
end
out(fid,"kappaMax of the SA b18 models (sigma %s): %s\n",mat2str(round(sgBank*255)),mat2str(kap,4));
km = max(kap);
for e = rd.etas
    out(fid,"  eta %.1f: eta*(1 + beta*kappaMax) for beta %s = %s\n",e,mat2str(rd.betas),mat2str(e*(1 + rd.betas*km),4));
end
% none
out(fid,"\nNo regularisation (beta = 0), PSNR [dB] per image (cols) and eta (rows):\n");
Mn = nan(numel(rd.etas),nI);
for i = 1:nI
    q = R3(arrayfun(@(r) r.img == imgs(i) && r.method == "none",R3));
    if isempty(q) || isempty(q.runs), continue, end
    Mn(:,i) = [q.runs.R.psnrEnd];
end
printtable(fid,"none",compose("eta %.1f",rd.etas(:)),[compose("kodim%02d",imgs) "mean"],[Mn mean(Mn,2)],"%.4f")
cases = struct('name',{},'clip',{});
for m = 1:numel(RDM)
    cases(end+1) = struct('name',RDM(m).name,'clip',true); %#ok<AGROW>
end
for m = 1:numel(RDM)
    if RDM(m).noclip, cases(end+1) = struct('name',RDM(m).name,'clip',false); end %#ok<AGROW>
end
combo = strings(0,1); [EE,BB] = ndgrid(rd.etas,rd.betas);
for c = 1:numel(EE), combo(c,1) = sprintf("eta %.1f beta %.1f",EE(c),BB(c)); end
nC = numel(combo);
for cc = cases
    out(fid,"\n==================== %s, clip %d ====================\n",cc.name,cc.clip);
    Pm = nan(nC,nI); Sm = Pm; Rf = Pm; Rr = Pm; St = Pm; Bl = Pm; Dfz = [];
    for i = 1:nI
        q = R3(arrayfun(@(r) r.img == imgs(i) && r.method == cc.name && r.clip == cc.clip,R3));
        if isempty(q) || isempty(q.runs), continue, end
        if isfield(q.runs,'dFrozen'), Dfz = [Dfz q.runs.dFrozen]; end %#ok<AGROW>
        for c = 1:nC
            r = q.runs.R([q.runs.R.eta] == EE(c) & [q.runs.R.beta] == BB(c));
            Pm(c,i) = r.psnrEnd; Sm(c,i) = r.ssimEnd; Bl(c,i) = r.blowup;
            Rf(c,i) = r.res(end)/sqrt(N);
            tb = max(1,numel(r.res) - 50);
            Rr(c,i) = r.res(end)/r.res(tb);
            St(c,i) = ~r.blowup && all(isfinite(r.res)) && r.res(end) <= r.res(tb) && r.res(end)/sqrt(N) < 1e-3;
        end
    end
    cl = [compose("kodim%02d",imgs) "mean"];
    printtable(fid,"PSNR [dB]",combo,cl,[Pm mean(Pm,2)],"%.4f")
    printtable(fid,"SSIM",combo,cl,[Sm mean(Sm,2)],"%.4f")
    printtable(fid,"Final residual r(T)/sqrt(N)",combo,cl,[Rf max(Rf,[],2)],"%.3e")
    printtable(fid,"r(T)/r(T-50)",combo,cl,[Rr max(Rr,[],2)],"%.3e")
    printtable(fid,"Stable (1/0), last col = number of images",combo,cl,[St sum(St,2)],"%d")
    printtable(fid,"Blow-up (1/0)",combo,cl,[Bl sum(Bl,2)],"%d")
    if ~isempty(Dfz), out(fid,"  frozen - not frozen at v, max over sigma models and images: %.2e\n",max(Dfz)); end
end
% residual curves at a few its (mean over images), for plots in the report
out(fid,"\n==================== Residual r(t)/sqrt(N) at selected its (mean over images) ====================\n");
tl = unique([1 10 50 100 150 200 250 rd.nIters]); tl = tl(tl <= rd.nIters);
for cc = cases
    Mc = nan(nC,numel(tl));
    for c = 1:nC
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

%% ----- printing -----
function s = nstr(x)
if isnan(x), s = "-"; else, s = sprintf("%d",x); end
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
