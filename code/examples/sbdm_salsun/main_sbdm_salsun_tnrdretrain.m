%% main_sbdm_salsun_tnrdretrain.m
% Retrain the TNRD models of main_sbdm_salsun_tnrdguar.m with a safer
% recipe, then redo Part 2 (guarantees) and Part 3 (RED) for them only.
% (2026-11-07)
%
% Why: with Adam lr 1e-3 constant (main_sbdm.m recipe) TNRD L never
% converged (loss jumps after it ~3000-10000), TNRD S3 at sigma 5 was below
% TNRD S1.
% Recipes (same mini-batches, same crop/noise streams as before and as SA):
%   c3 : Adam, cosine decay 1e-3 -> 1e-5 over the run (as the SA runs)
%   c4 : Adam, cosine decay 1e-4 -> 1e-6 (for TNRD L, if c3 still oscillates)
%   c3g/c4g : c3/c4 + global-norm gradient clipping at rtClip (if needed)
% Its: 15000 x S (as before). Models S1 (834), S3 (2502), L (31510 params).
%
% Stages (variable rtStage, default "A"):
%   "A": checks; S1, S3 with c3 at sigma 25, 10, 5, 2; TNRD L probes = the
%        first rtProbeIts its of the real L runs (sigma 25 and 2, c3 and c4),
%        kept as checkpoints so that stage B resumes them; probe summary.
%   "B": recipe rtRecL for L (default "c3"); if it is not c3, S1 and S3 are
%        also trained with it (same recipe for all sizes); L at sigma 25, 10,
%        then 5, 2 as long as the time allows (deadline 16:20 for training);
%        Part 1 summary, Part 2 and Part 3 for the new TNRD models (SA, N,
%        old TNRD: cached results of main_sbdm_salsun_tnrdguar.m).
%
% New files only. Caches: data/sbdm_tg2_*.mat. Logs: logs/tnrdretrain*.
%   nohup matlab -licmode onlinelicensing -batch "rtStage='A'; main_sbdm_salsun_tnrdretrain" \
%       > ~/Documents/SBDMwLSUN/logs/tnrdretrain_A.log 2>&1 &
% Quick test (tiny settings, separate folder):
%   matlab -licmode onlinelicensing -batch "onTest=true; onTestDir='<dir>'; rtStage='A'; main_sbdm_salsun_tnrdretrain"

here = fileparts(mfilename('fullpath'));
cd(fullfile(here,'..','..')); setpath; cd(here)
addpath(fullfile(here,'..','sbdm'))     % helpers (called, not edited)
addpath(fullfile(here,'..','salsun'))   % maskLayer
warning('off','MATLAB:rmpath:DirNotFound')
rmpath(fullfile('/home/shogo/Workspace/GitHub/SaivDr','mexcodes'))
warning('on','MATLAB:rmpath:DirNotFound')

% ---- settings ----
if ~exist('rtStage','var'), rtStage = "A"; end
if ~exist('rtRecL','var'), rtRecL = "c3"; end
if ~exist('rtClip','var'), rtClip = NaN; end
if ~exist('rtProbeIts','var'), rtProbeIts = 10000; end
rtStage = string(rtStage); rtRecL = string(rtRecL);
cfg0 = fcn_ssconfig();                 % imgIdx 9/15/19/23, crop 256, stride 4, ovlp 3, seedBase 0
cfg0.selImg = 9;
cfg0.sigmas = [25 10 5 2];
cfg0.sup = struct('nIters',1500,'mb',32,'lrN',[5e-3 5e-5],'lrEst',[1e-3 1e-5],...
    'lrShr',[5e-3 5e-5],'seed',1,'patch',[64 64],'nCrops',256,'trainIdx',1:8,'ckptEvery',200);
TN = struct('name',{"S1","S3","L"},'S',{1,3,5},'P',{32,32,126},'fs',{5,5,7});
tc = struct('nItPerStage',15000,'mb',32,'ckptEvery',2000);
REC = struct('name',{"c3","c4","c3g","c4g"},'lr',{[1e-3 1e-5],[1e-4 1e-6],[1e-3 1e-5],[1e-4 1e-6]},...
    'clip',{inf,inf,rtClip,rtClip});
dg = struct('nProbe',4,'nPower',60,'seedBase',500);
rd = struct('blurSize',9,'blurStd',2.0,'sigmaW',5/255,'nIters',300,'sgMax',25/255,'sgMin',2/255,...
    'etas',[0.5 1 2],'betas',[0.5 1 2],'seedBase',1000);
probeSg = [25 2]; probeRec = ["c3" "c4"];
dat = fullfile(here,'..','..','..','data');
ctx.dat = dat; ctx.cdir = dat;
ctx.sdir = fullfile(here,'..','..','..','..','logs');
ctx.deadlineTrain = datetime('today') + hours(16) + minutes(20);
ctx.deadlineAll = datetime('today') + hours(16) + minutes(55);
ctx.deadline = ctx.deadlineTrain;
ctx.margin = 1.10;
isTest = exist('onTest','var') && onTest;
nItCheck = 20;
if isTest
    tc.nItPerStage = 6; tc.ckptEvery = 2; rtProbeIts = 3;
    nItCheck = 4;
    dg.nProbe = 1; dg.nPower = 2;
    rd.nIters = 4; rd.etas = [0.5 2]; rd.betas = [2];
    ctx.cdir = onTestDir; ctx.sdir = onTestDir;
    ctx.deadlineTrain = datetime('now') + hours(1); ctx.deadlineAll = ctx.deadlineTrain; ctx.deadline = ctx.deadlineTrain;
    if ~isfolder(onTestDir), mkdir(onTestDir); end
end
ctx.isTest = isTest;
assert(any([REC.name] == rtRecL),'unknown recipe %s',rtRecL)
RL = REC([REC.name] == rtRecL);
assert(isfinite(RL.lr(1)) && ~isnan(RL.clip),'recipe %s needs rtClip',rtRecL)
ctx.useGpu = canUseGPU;
if ctx.useGpu, deep.gpu.deterministicAlgorithms(true); end   % as tnrdguar (bitwise resume)
fprintf("deterministic GPU algorithms: %d\n",ctx.useGpu && deep.gpu.deterministicAlgorithms);
ctx.cfg0 = cfg0; ctx.tc = tc; ctx.TN = TN; ctx.REC = REC;
ctx.ST = containers.Map();
ctx.ST('skips') = strings(0,1);
fprintf("%s  start. Stage %s, recipe for L %s, clip %g, probe its %d. GPU: %d. Training deadline %s, deadline %s. Test mode: %d\n",...
    string(datetime('now')),rtStage,rtRecL,rtClip,rtProbeIts,ctx.useGpu,string(ctx.deadlineTrain),string(ctx.deadlineAll),isTest)
for R = REC, fprintf("  recipe %-4s lr %s -> %s (cosine), clip %g\n",R.name,num2str(R.lr(1)),num2str(R.lr(2)),R.clip); end
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
for j = 1:numel(TN)
    TN(j).np = countT(fcn_convtnrdinitprm(TN(j).S,TN(j).P,TN(j).fs,"tied"));
    fprintf("TNRD %-2s: %d stages, %d filters %dx%d -> params %d, its %d\n",TN(j).name,TN(j).S,TN(j).P,TN(j).fs,TN(j).fs,TN(j).np,TN(j).S*tc.nItPerStage);
end
ctx.TN = TN;

%% ===== checks (sigma 25) =====
CHK = struct('name',{},'vNew',{},'vRef',{},'ok',{},'info',{});
allOk = true;
try
    SB = load(fullfile(dat,'sbdm_salsun_dc_sup_s25_SAR_it1500.mat'));
    h1Ref = SB.q.tr.hist(1);
    sg = single(25/255);
    rng(cfg0.sup.seed,'twister');
    nets0 = fcn_dcbuildsalsun(cfg0.sup.patch,cfg0.stride,cfg0.ovlp,ob18);
    net0 = nets0.shrink; if ctx.useGpu, net0 = dlupdate(@gpuArray,net0); end
    for j = 1:numel(TN)
        % 1: the first batch is the first batch of the SA b18 training
        prm0 = tninit(TN(j),ctx);
        [xb,vb] = fcn_minibatch(trainSet,sg,false,cfg0.sup.mb);
        lSA = double(gather(extractdata(dlfeval(@lossSA,net0,vb,xb))));
        CHK(end+1) = mkchk(sprintf("TNRD %s: first batch, SA b18 init loss vs stage B hist(1)",TN(j).name),lSA,h1Ref); %#ok<SAGROW>
        % 2: the new trainer (recipe c3) sees that batch
        lT = double(gather(extractdata(dlfeval(@fcn_convtnrdloss,prm0,vb,xb,single(1)))));
        prm0 = tninit(TN(j),ctx);
        [~,hT] = fcn_tgtrainconvtnrdcos(prm0,trainSet,sg,1,REC(1).lr,inf,tc.mb,false);
        CHK(end+1) = mkchk(sprintf("TNRD %s: new trainer (c3) hist(1) vs TNRD loss on that batch",TN(j).name),hT(1),lT); %#ok<SAGROW>
    end
    % 3: new trainer with constant lr and no clipping = Sensei's fcn_trainconvtnrd (bitwise)
    j = 2;
    prm0 = tninit(TN(j),ctx);
    [pA,hA] = fcn_trainconvtnrd(prm0,trainSet,sg,nItCheck,1e-3,0,tc.mb,false);
    prm0 = tninit(TN(j),ctx);
    [pB,hB] = fcn_tgtrainconvtnrdcos(prm0,trainSet,sg,nItCheck,[1e-3 1e-3],inf,tc.mb,false);
    c = mkchk(sprintf("TNRD S3: new trainer (lr 1e-3 const, no clip) vs fcn_trainconvtnrd, %d its",nItCheck),hB(end),hA(end));
    c.ok = isequal(hA(:),hB(:)) && eqprm(pA,pB);
    c.info = "history and final parameters bitwise equal"; CHK(end+1) = c;
    % 4: gradient clipping helper
    prm0 = tninit(TN(j),ctx);
    [xb,vb] = fcn_minibatch(trainSet,sg,false,cfg0.sup.mb);
    [~,g0] = dlfeval(@fcn_convtnrdloss,prm0,vb,xb,single(1));
    [gI,n0] = fcn_tgclipgrad(g0,inf);
    [gH,~] = fcn_tgclipgrad(g0,n0/2);
    [~,nH] = fcn_tgclipgrad(gH,inf);
    c = mkchk("TNRD S3: clipping at |g|/2 gives norm |g|/2 (rel. diff < 1e-5)",nH,n0/2);
    c.ok = abs(nH - n0/2)/(n0/2) < 1e-5 && eqprm(gI,g0);
    c.info = sprintf("|g| = %.6e; clip inf leaves g unchanged",n0); CHK(end+1) = c;
    ctx.gnorm0 = n0;
    % 5: resume (c3 + clipping at |g0|/2, stopped after it 1 and 2) = unbroken run
    ck = fullfile(ctx.cdir,'sbdm_tg2_check_resume_ckpt.mat');
    if isfile(ck), delete(ck); end
    prm0 = tninit(TN(j),ctx);
    [pU,hU] = fcn_tgtrainconvtnrdcos(prm0,trainSet,sg,nItCheck,REC(1).lr,n0/2,tc.mb,false);
    prm0 = tninit(TN(j),ctx);
    [~,~,d1] = fcn_tgtrainconvtnrdcos(prm0,trainSet,sg,nItCheck,REC(1).lr,n0/2,tc.mb,false,ck,inf,inf,1);
    rng(12345); if ctx.useGpu, gpurng(12345); end
    [~,~,d2] = fcn_tgtrainconvtnrdcos(prm0,trainSet,sg,nItCheck,REC(1).lr,n0/2,tc.mb,false,ck,0,inf);
    rng(54321); if ctx.useGpu, gpurng(54321); end
    [pC,hC,d3] = fcn_tgtrainconvtnrdcos(prm0,trainSet,sg,nItCheck,REC(1).lr,n0/2,tc.mb,false,ck,inf,inf);
    delete(ck);
    c = mkchk(sprintf("TNRD S3: c3 + clipping, resumed (stops after its 1, 2) vs unbroken, %d its",nItCheck),hC(end),hU(end));
    c.ok = ~d1 && ~d2 && d3 && isequal(hC(:),hU(:)) && eqprm(pC,pU);
    c.info = "history and final parameters bitwise equal"; CHK(end+1) = c;
catch ME
    fprintf("CHECK FAILED (error): %s\n",getReport(ME,'extended','hyperlinks','off'))
    allOk = false;
end
for c = CHK
    fprintf("CHECK %-80s new %.10e  ref %.10e  ok %d  %s\n",c.name,c.vNew,c.vRef,c.ok,c.info)
end
allOk = allOk && ~isempty(CHK) && all([CHK.ok]);
if ~allOk
    fprintf("\n%s  A check failed. The runs are NOT started.\n",string(datetime('now')))
    save(fullfile(ctx.cdir,'sbdm_tg2_checks_failed.mat'),'CHK')
    return
end
fprintf("All %d checks passed.\n",numel(CHK))
clear nets0 net0

%% ===== speed (for the estimates) =====
for j = 1:numel(TN)
    prm0 = tninit(TN(j),ctx);
    fcn_tgtrainconvtnrdcos(prm0,trainSet,single(25/255),5,REC(1).lr,inf,tc.mb,false);
    if ctx.useGpu, wait(gpuDevice); end
    t = tic; fcn_tgtrainconvtnrdcos(prm0,trainSet,single(25/255),50,REC(1).lr,inf,tc.mb,false);
    if ctx.useGpu, wait(gpuDevice); end
    ctx.spi.(TN(j).name) = toc(t)/50;
    fprintf("speed TNRD %-2s: %.4f s/it -> %.1f min per full training\n",TN(j).name,ctx.spi.(TN(j).name),ctx.spi.(TN(j).name)*TN(j).S*tc.nItPerStage/60)
end

%% ===== training =====
if rtStage == "A"
    recSmall = "c3";
    for R = REC(ismember([REC.name],recSmall))
        trainall(ctx,TN(1:2),R,cfg0.sigmas,trainSet);
    end
    % TNRD L probes = the first rtProbeIts its of the real runs (kept as checkpoints)
    PR = struct('sg',{},'rec',{},'hist',{},'gnorm',{},'t',{});
    for sgi = probeSg
        for rn = probeRec
            R = REC([REC.name] == rn); T = TN(3);
            f = fprobe(ctx,sgi,R.name,rtProbeIts);
            ckpt = string(ftn2(ctx,sgi,T,R.name)) + "_ckpt.mat";
            ctx.ST(char("probe")) = ctx.spi.L*rtProbeIts + 20;
            q = unit(ctx,sprintf("probe TNRD L s%02d %s",sgi,R.name),f,"probe",@() runprobe(T,R,sgi,trainSet,ctx,ckpt,rtProbeIts));
            if ~isempty(q), PR(end+1) = struct('sg',sgi,'rec',R.name,'hist',q.hist,'gnorm',q.gnorm,'t',q.tWall); end %#ok<SAGROW>
        end
    end
    safesummary(@() summaryprobe(fullfile(ctx.sdir,'tnrdretrain_probe_summary.txt'),PR,ctx,rtProbeIts));
    safesummary(@() plotprobe(fullfile(ctx.sdir,'tnrdretrain_probe.png'),PR,ctx));
    P1 = collect1(ctx,TN,REC);
    safesummary(@() summary1(fullfile(ctx.sdir,'tnrdretrain_part1_summary.txt'),P1,CHK,ctx,TN,tRun0));
    sk = ctx.ST('skips');
    fprintf("\nSkipped / failed / missing units (%d):\n",numel(sk)); fprintf("  %s\n",sk)
    fprintf("\n%s  stage A done.\n",string(datetime('now')))
    return
end

% ---- stage B ----
recEval = unique(["c3" rtRecL],'stable');
if rtRecL ~= "c3"
    trainall(ctx,TN(1:2),RL,cfg0.sigmas,trainSet);   % same recipe for all three sizes
end
trainall(ctx,TN(3),RL,cfg0.sigmas,trainSet);          % L: sigma 25, 10, 5, 2 (in this order)
clear trainSet
P1 = collect1(ctx,TN,REC);
safesummary(@() summary1(fullfile(ctx.sdir,'tnrdretrain_part1_summary.txt'),P1,CHK,ctx,TN,tRun0));
save(fullfile(ctx.cdir,'sbdm_tg2_part1_results.mat'),'P1','CHK','cfg0','tc','REC')
tP1 = datetime('now');
ctx.deadline = ctx.deadlineAll;

% new TNRD models to evaluate (all sizes, recipes in recEval, if trained at all four sigma)
TM = struct('T',{},'rec',{},'key',{});
for rn = recEval
    for j = 1:numel(TN)
        ok = all(arrayfun(@(s) isfile(ftn2(ctx,s,TN(j),rn)),cfg0.sigmas));
        if ok
            TM(end+1) = struct('T',TN(j),'rec',rn,'key',"TNRD " + TN(j).name + " " + rn); %#ok<SAGROW>
        else
            addskip(ctx,sprintf("MISSING: TNRD %s %s is not trained at all four sigma; Part 2/3 use the sigma that exist (Part 2) / skip it (Part 3)",TN(j).name,rn));
            TM(end+1) = struct('T',TN(j),'rec',rn,'key',"TNRD " + TN(j).name + " " + rn); %#ok<SAGROW>
        end
    end
end

%% ===== Part 2: guarantees (new TNRD only) =====
D2 = cell(numel(cfg0.sigmas),numel(cfg0.imgIdx));
ctx.ST('diag') = 60;
for s = 1:numel(cfg0.sigmas)
    sgi = cfg0.sigmas(s);
    for i = 1:numel(cfg0.imgIdx)
        f = fullfile(ctx.cdir,sprintf('sbdm_tg2_diag_s%02d_kodim%02d_%s.mat',sgi,cfg0.imgIdx(i),strjoin(recEval,"")));
        D2{s,i} = unit(ctx,sprintf("diag s%02d kodim%02d",sgi,cfg0.imgIdx(i)),f,"diag",...
            @() rundiag2(ctx,sgi,cfg0.imgIdx(i),TM,dg));
    end
end
safesummary(@() summary2(fullfile(ctx.sdir,'tnrdretrain_part2_summary.txt'),D2,ctx,TM,dg));
save(fullfile(ctx.cdir,'sbdm_tg2_part2_results.mat'),'D2','dg','TM')
tP2 = datetime('now');

%% ===== Part 3: RED deblurring (new TNRD only) =====
blurKernel = fspecial('gaussian',rd.blurSize,rd.blurStd);
Hf  = @(z) imfilter(z,blurKernel,'circular');
Htf = @(z) imfilter(z,rot90(blurKernel,2),'circular');
schedule = exp(linspace(log(rd.sgMax),log(rd.sgMin),rd.nIters));
sgBank = cfg0.sigmas/255;
RDM = struct('name',{},'tn',{},'rec',{});
for rn = recEval
    for tn = ["L" "S3"]
        RDM(end+1) = struct('name',"TNRD " + tn + " " + rn,'tn',tn,'rec',rn); %#ok<SAGROW>
    end
end
ctx.ST('redT') = 9*rd.nIters*0.04 + 10;
R3 = struct('img',{},'method',{},'clip',{},'runs',{});
for i = 1:numel(cfg0.imgIdx)
    idx = cfg0.imgIdx(i);
    [X,V] = redobs(ctx,idx,Hf,rd);
    fprintf("\nkodim%02d: observation PSNR %.4f dB (clipped to [0,1] for the measure)\n",idx,psnr(min(max(V,0),1),X))
    for m = 1:numel(RDM)
        T = TN([TN.name] == RDM(m).tn);
        fb = {};
        if all(arrayfun(@(s) isfile(ftn2(ctx,s,T,RDM(m).rec)),cfg0.sigmas))
            fb = tnbank(ctx,T,RDM(m).rec);
        end
        for clip = [true false]
            f = fullfile(ctx.cdir,sprintf('sbdm_tg2_red_kodim%02d_%s_c%d.mat',idx,strrep(RDM(m).name," ",""),clip));
            lab = sprintf("RED kodim%02d %s clip %d",idx,RDM(m).name,clip);
            if isempty(fb) && ~isfile(f)
                addskip(ctx,sprintf("MISSING %s: not all four sigma models",lab)); continue
            end
            q = unit(ctx,lab,f,"redT",@() runred2(fb,RDM(m).name,X,V,Hf,Htf,schedule,sgBank,rd,clip));
            R3(end+1) = struct('img',idx,'method',RDM(m).name,'clip',clip,'runs',{q}); %#ok<SAGROW>
        end
    end
end
R3s = R3;
for k = 1:numel(R3s)
    if ~isempty(R3s(k).runs) && isfield(R3s(k).runs,'R'), R3s(k).runs.R = rmfield(R3s(k).runs.R,'est'); end
end
save(fullfile(ctx.cdir,'sbdm_tg2_part3_results.mat'),'R3s','rd','schedule','sgBank','RDM')
safesummary(@() summary3(fullfile(ctx.sdir,'tnrdretrain_part3_summary.txt'),R3s,ctx,rd,RDM,schedule,sgBank));

sk = ctx.ST('skips');
fprintf("\nTimes: start %s, part 1 -> %s, part 2 -> %s, part 3 -> %s\n",string(tRun0),string(tP1),string(tP2),string(datetime('now')))
fprintf("\nSkipped / failed / missing units (%d):\n",numel(sk))
fprintf("  %s\n",sk)
fprintf("\n%s  done.\n",string(datetime('now')))

%% ===================== Local functions =====================
function n = countT(p)
n = sum(cellfun(@numel,p.Wa)) + sum(cellfun(@numel,p.ba)) + sum(cellfun(@numel,p.bs)) + sum(cellfun(@numel,p.loglam));
end

function prm = tninit(T,ctx)
% as main_sbdm_salsun_tnrdguar.m: rng(1), gpurng(1), TNRD init (CPU
% stream), then the CPU stream state after the SA b18 build
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

function f = ftn2(ctx,sg,T,rec)
f = fullfile(ctx.cdir,sprintf('sbdm_tg2_tnrd_s%02d_%s_%s_it%d.mat',sg,T.name,rec,T.S*ctx.tc.nItPerStage));
end

function f = ftnold(ctx,sg,T)
f = fullfile(ctx.dat,sprintf('sbdm_tg_tnrd_s%02d_%s_it%d.mat',sg,T.name,T.S*15000));
end

function f = fprobe(ctx,sg,rec,nIt)
f = fullfile(ctx.cdir,sprintf('sbdm_tg2_probe_s%02d_L_%s_it%d.mat',sg,rec,nIt));
end

function trainall(ctx,TNs,R,sigmas,trainSet)
for sgi = sigmas
    for T = TNs
        lab = sprintf("TNRD %s s%02d %s",T.name,sgi,R.name);
        f = ftn2(ctx,sgi,T,R.name);
        ckpt = string(f) + "_ckpt.mat";
        nIt = T.S*ctx.tc.nItPerStage; it0 = 0;
        if isfile(ckpt), C = load(ckpt,'it0'); it0 = C.it0; end
        key = "tnrd" + T.name + sgi + R.name;
        ctx.ST(char(key)) = ctx.spi.(T.name)*(nIt - it0) + 30;
        unit(ctx,lab,f,key,@() runtnrd2(T,R,sgi,trainSet,ctx,ckpt));
    end
end
end

function q = runtnrd2(T,R,sg,trainSet,ctx,ckpt)
tc = ctx.tc; cfg = ctx.cfg0; cfg.sigma = single(sg/255);
t = tic;
prm0 = tninit(T,ctx);
nIt = T.S*tc.nItPerStage; it0 = 0;
if isfile(ckpt), C = load(ckpt,'it0'); it0 = C.it0; end
[prm,hist,done,gnorm] = fcn_tgtrainconvtnrdcos(prm0,trainSet,cfg.sigma,nIt,R.lr,R.clip,tc.mb,true,char(ckpt),inf,tc.ckptEvery);
if ~done, error("training not finished"); end
if ctx.useGpu, wait(gpuDevice); end
q.tTrain = toc(t); q.itStart = it0; q.tPerIt = q.tTrain/(nIt - it0);
q.prm = dlupdate(@gather,prm); q.hist = hist; q.gnorm = gnorm; q.T = T; q.R = R; q.sg = sg; q.np = countT(q.prm);
nb = min(100,numel(hist)); q.lossEnd = mean(hist(end-nb+1:end));
q.lam = cellfun(@(a) double(exp(extractdata(a))),q.prm.loglam);
fprintf("    trained: %.1f s for its %d-%d (%.4f s/it), loss (mean of last %d its) %.4e, params %d, lambda %s\n",...
    q.tTrain,it0+1,nIt,q.tPerIt,nb,q.lossEnd,q.np,mat2str(q.lam,4))
for i = 1:numel(cfg.imgIdx)
    [x,v] = fcn_ssloadimage(ctx.dat,cfg.imgIdx(i),cfg,ctx.useGpu);
    z = gather(extractdata(fcn_convtnrddenoise(prm,dlarray(v,'SSCB'),single(1))));
    r.psnr = psnr(z,gather(x)); r.ssim = ssim(z,gather(x));
    r.mse = double(mean((z - gather(x)).^2,'all'));
    q.res(i) = r;
    fprintf("    kodim%02d  %.4f dB  SSIM %.4f\n",cfg.imgIdx(i),r.psnr,r.ssim)
end
q.tTotal = toc(t);
if isfile(ckpt), delete(ckpt); end
end

function q = runprobe(T,R,sg,trainSet,ctx,ckpt,nProbe)
% the first nProbe its of the real run; the checkpoint is kept for stage B
tc = ctx.tc;
prm0 = tninit(T,ctx);
nIt = T.S*tc.nItPerStage;
if isfile(ckpt)
    C = load(ckpt,'it0','hist','gnorm');
    if C.it0 >= nProbe
        q.hist = C.hist(1:nProbe); q.gnorm = C.gnorm(1:nProbe); q.rec = R.name; q.sg = sg; return
    end
end
[~,hist,~,gnorm] = fcn_tgtrainconvtnrdcos(prm0,trainSet,single(sg/255),nIt,R.lr,R.clip,tc.mb,true,char(ckpt),inf,tc.ckptEvery,nProbe);
q.hist = hist; q.gnorm = gnorm; q.rec = R.name; q.sg = sg;
end

function s = winstat(h,b,w)
% mean and max of h over [b-w+1, b] (b clipped to numel(h))
s = nan(2,numel(b));
for k = 1:numel(b)
    if b(k) > numel(h), continue, end
    seg = h(max(1,b(k)-w+1):b(k));
    s(:,k) = [mean(seg); max(seg)];
end
end

function summaryprobe(file,PR,ctx,nProbe)
fid = fopen(file,'w'); cleaner = onCleanup(@() fclose(fid));
out(fid,"TNRD L probes: the first %d its of the real runs (75000 its, cosine schedule over 75000). Written %s.\n",nProbe,string(datetime('now')));
out(fid,"old = the run of main_sbdm_salsun_tnrdguar.m (lr 1e-3 constant). Same crops and noise for all (same streams).\n");
out(fid,"Loss = MSE per pixel of the mini-batch. Window of 50 its ending at the shown it: mean and max.\n");
for R = ctx.REC, out(fid,"  %-4s lr %s -> %s (cosine), clip %g\n",R.name,num2str(R.lr(1)),num2str(R.lr(2)),R.clip); end
b = [50 100 200 300 500 1000 1500 2000 2500 3000 4000 5000 6000 7500 8500 10000];
b = b(b <= nProbe); if isempty(b), b = nProbe; end
cols = compose("it %d",b);
for sg = unique([PR.sg],'stable')
    out(fid,"\n==================== sigma = %d/255 ====================\n",sg);
    rows = strings(0,1); Mm = []; Mx = []; Gm = []; Gx = [];
    try
        S = load(ftnold(ctx,sg,ctx.TN(3))); ho = S.q.hist;
        s = winstat(ho,b,50); rows(end+1) = "old (lr 1e-3 const)"; Mm(end+1,:) = s(1,:); Mx(end+1,:) = s(2,:);
    catch
    end
    rowsG = strings(0,1);
    for p = PR([PR.sg] == sg)
        s = winstat(p.hist,b,50); rows(end+1) = p.rec; Mm(end+1,:) = s(1,:); Mx(end+1,:) = s(2,:); %#ok<AGROW>
        s = winstat(p.gnorm,b,50); rowsG(end+1) = p.rec; Gm(end+1,:) = s(1,:); Gx(end+1,:) = s(2,:); %#ok<AGROW>
    end
    printtable(fid,"loss, mean over the window",rows,cols,Mm,"%.3e")
    printtable(fid,"loss, max over the window",rows,cols,Mx,"%.3e")
    printtable(fid,"gradient norm |g| (before clipping), mean over the window",rowsG,cols,Gm,"%.3e")
    printtable(fid,"gradient norm |g| (before clipping), max over the window",rowsG,cols,Gx,"%.3e")
    for p = PR([PR.sg] == sg)
        h = p.hist; mm = movmedian(h,201);
        nSp = sum(h(201:end) > 3*mm(201:end));
        out(fid,"  %s: its with loss > 3x the moving median (201 its), its 201-%d: %d; max loss after it 1000: %.3e; final mean of 100 its: %.4e\n",...
            p.rec,numel(h),nSp,max(h(min(end,1001):end)),mean(h(max(1,end-99):end)));
    end
    if exist('ho','var')
        h = ho(1:min(end,nProbe)); mm = movmedian(h,201);
        out(fid,"  old: its with loss > 3x the moving median (201 its), its 201-%d: %d; max loss after it 1000: %.3e\n",numel(h),sum(h(201:end) > 3*mm(201:end)),max(h(min(end,1001):end)));
        clear ho
    end
end
end

function plotprobe(file,PR,ctx)
fg = figure('Visible','off','Position',[0 0 1400 520]);
sgs = unique([PR.sg],'stable');
tl = tiledlayout(fg,1,numel(sgs));
for k = 1:numel(sgs)
    ax = nexttile(tl); hold(ax,'on')
    lg = strings(0,1);
    try
        S = load(ftnold(ctx,sgs(k),ctx.TN(3))); n = max(arrayfun(@(p) numel(p.hist),PR));
        h = S.q.hist(1:min(end,n)); semilogy(ax,movmean(h,50),'Color',[0.6 0.6 0.6]); lg(end+1) = "old (lr 1e-3 const)";
    catch
    end
    for p = PR([PR.sg] == sgs(k))
        semilogy(ax,movmean(p.hist,50)); lg(end+1) = p.rec; %#ok<AGROW>
    end
    set(ax,'YScale','log'); grid(ax,'on'); xlabel(ax,'iteration'); ylabel(ax,'loss (moving mean of 50 its)')
    title(ax,sprintf("TNRD L, sigma = %d/255",sgs(k))); legend(ax,lg,'Location','northeast')
end
exportgraphics(fg,file,'Resolution',110); close(fg)
fprintf("probe plot: %s\n",file)
end

%% ----- part 1 -----
function P1 = collect1(ctx,TN,REC)
% old results (tnrdguar part 1: N, SA b18/b3/b1, old TNRD) + new TNRD per recipe
cfg = ctx.cfg0; nI = numel(cfg.imgIdx);
O = load(fullfile(ctx.dat,'sbdm_tg_part1_results.mat'),'P1'); O = O.P1;
P1 = struct('sg',{},'names',{},'P',{},'S',{},'np',{},'loss',{},'tTrain',{},'tPerIt',{},'its',{},'lam',{});
for s = 1:numel(cfg.sigmas)
    sg = cfg.sigmas(s);
    o = O([O.sg] == sg);
    names = o.names; names(startsWith(names,"TNRD")) = names(startsWith(names,"TNRD")) + " old";
    T = o; T.names = names; T.lam = o.lam;
    for R = REC
        for Tn = TN
            f = ftn2(ctx,sg,Tn,R.name);
            if ~isfile(f), continue, end
            S = load(f); q = S.q;
            T.names(end+1) = "TNRD " + Tn.name + " " + R.name;
            T.P(:,end+1) = arrayfun(@(r) r.psnr,q.res(:)); T.S(:,end+1) = arrayfun(@(r) r.ssim,q.res(:));
            T.np(end+1) = q.np; T.loss(end+1) = q.lossEnd; T.tTrain(end+1) = q.tTrain; T.tPerIt(end+1) = q.tPerIt;
            T.its(end+1) = numel(q.hist); T.lam{end+1} = q.lam;
        end
    end
    assert(size(T.P,1) == nI)
    P1(end+1) = T; %#ok<AGROW>
end
end

function summary1(file,P1,CHK,ctx,TN,tRun0)
fid = fopen(file,'w'); cleaner = onCleanup(@() fclose(fid));
cfg = ctx.cfg0; tc = ctx.tc;
out(fid,"Part 1 (retrain): supervised TNRD with new recipes vs old TNRD (lr 1e-3 const) and SA-LSUN. Written %s (run start %s).\n",string(datetime('now')),string(tRun0));
out(fid,"Training data, crops, noise streams, evaluation: as main_sbdm_salsun_tnrdguar.m (its 1-1500 = the same crops and noise as the SA runs).\n");
out(fid,"TNRD 'tied', unconditional (sigmaIn = 1), batch %d, %d its per stage, Adam:\n",tc.mb,tc.nItPerStage);
out(fid,"  old : lr 1e-3 constant (main_sbdm.m recipe), cached\n");
for R = ctx.REC, out(fid,"  %-4s: lr %s -> %s (cosine), gradient clipping %g\n",R.name,num2str(R.lr(1)),num2str(R.lr(2)),R.clip); end
for T = TN, out(fid,"  TNRD %-2s: %d stages, %d filters %dx%d, params %d, its %d\n",T.name,T.S,T.P,T.fs,T.fs,T.np,T.S*tc.nItPerStage); end
out(fid,"N, SA b18/b3/b1: cached (stage B, run A, size sweep, tnrdguar). Evaluation: 256x256 centre crops, no clipping. mean(test) = kodim15/19/23.\n");
out(fid,"\nChecks before the runs:\n");
for c = CHK, out(fid,"  %s: new %.10e  ref %.10e  ok %d (%s)\n",c.name,c.vNew,c.vRef,c.ok,c.info); end
imgs = cfg.imgIdx(:); rows = compose("kodim%02d",imgs);
iTest = find(imgs ~= cfg.selImg);
rowsM = [rows; "mean(all)"; "mean(test)"];
addm = @(M) [M; mean(M,1,'omitnan'); mean(M(iTest,:),1,'omitnan')];
allNames = P1(1).names;
for k = 2:numel(P1), allNames = union(allNames,P1(k).names,'stable'); end
nmNew = allNames(startsWith(allNames,"TNRD") & ~endsWith(allNames," old"));
recs = unique(regexprep(nmNew,'^.* ',''),'stable');
OVp = nan(numel(P1),numel(allNames)); OVs = OVp; OVl = OVp;
for k = 1:numel(P1)
    T = P1(k); names = T.names;
    out(fid,"\n==================== sigma = %d/255 ====================\n",T.sg);
    out(fid,"\nParameters and training (loss = mean of the last 100 its, MSE per pixel)\n");
    for j = 1:numel(names)
        lam = ""; if ~isempty(T.lam{j}), lam = "  lambda " + mat2str(T.lam{j},4); end
        out(fid,"  %-12s params %6s  its %6s  loss %.4e  train %.1f s  %.4f s/it%s\n",names(j),nstr(T.np(j)),nstr(T.its(j)),T.loss(j),T.tTrain(j),T.tPerIt(j),lam);
    end
    printtable(fid,"PSNR [dB]",rowsM,names,addm(T.P),"%.4f")
    printtable(fid,"SSIM",rowsM,names,addm(T.S),"%.4f")
    % pairs: SA - TNRD (old and new), new - old
    cn = strings(1,0); dP = zeros(numel(imgs),0); dS = dP;
    pairs = {"SA b18","L"; "SA b3","S3"; "SA b1","S1"};
    for r = ["old" recs]
        for p = 1:3
            a = find(names == pairs{p,1}); b = find(names == "TNRD " + pairs{p,2} + " " + r);
            if isempty(a) || isempty(b), continue, end
            cn(end+1) = pairs{p,1} + "-" + pairs{p,2} + " " + r; %#ok<AGROW>
            dP(:,end+1) = T.P(:,a) - T.P(:,b); dS(:,end+1) = T.S(:,a) - T.S(:,b); %#ok<AGROW>
        end
    end
    printtable(fid,"SA - TNRD, PSNR [dB]",rowsM,cn,addm(dP),"%.4f")
    printtable(fid,"SA - TNRD, SSIM",rowsM,cn,addm(dS),"%.4f")
    out(fid,"Images where SA beats TNRD (PSNR), columns as above: test (of %d) %s; all (of %d) %s\n",numel(iTest),...
        mat2str(sum(dP(iTest,:) > 0,1)),numel(imgs),mat2str(sum(dP > 0,1)));
    cn = strings(1,0); dN = zeros(numel(imgs),0);
    for r = recs
        for tn = ["S1" "S3" "L"]
            a = find(names == "TNRD " + tn + " " + r); b = find(names == "TNRD " + tn + " old");
            if isempty(a) || isempty(b), continue, end
            cn(end+1) = tn + " " + r + "-old"; dN(:,end+1) = T.P(:,a) - T.P(:,b); %#ok<AGROW>
        end
    end
    printtable(fid,"new TNRD - old TNRD, PSNR [dB]",rowsM,cn,addm(dN),"%.4f")
    am = addm(T.P); as = addm(T.S);
    [~,ia,ib] = intersect(allNames,names,'stable');
    OVp(k,ia) = am(end,ib); OVs(k,ia) = as(end,ib); OVl(k,ia) = T.loss(ib);
end
sgRows = compose("sigma %d",[P1.sg].');
out(fid,"\n==================== Over sigma ====================\n");
printtable(fid,"PSNR [dB], mean(test)",sgRows,allNames,OVp,"%.4f")
printtable(fid,"SSIM, mean(test)",sgRows,allNames,OVs,"%.4f")
printtable(fid,"Training loss (mean of the last 100 its)",sgRows,allNames,OVl,"%.4e")
sk = ctx.ST('skips');
out(fid,"\n---- Skipped / failed / missing units so far (%d) ----\n",numel(sk));
for k = 1:numel(sk), out(fid,"  %s\n",sk(k)); end
end

%% ----- part 2 -----
function q = rundiag2(ctx,sg,idx,TM,dg)
cfg = ctx.cfg0; cfg.sigma = single(sg/255); sz = cfg.cropSize;
[x,v] = fcn_ssloadimage(ctx.dat,idx,cfg,ctx.useGpu);
gz = zeros(1,'single'); if ctx.useGpu, gz = gpuArray(gz); end
q = struct('model',{},'asymRand',{},'rhoRand',{},'asymImg',{},'rhoImg',{},'psnr',{},'t',{});
for m = 1:numel(TM)
    f = ftn2(ctx,sg,TM(m).T,TM(m).rec);
    if ~isfile(f), continue, end
    S = load(f); prm = S.q.prm; if ctx.useGpu, prm = dlupdate(@gpuArray,prm); end
    fd = @(z) fcn_convtnrddenoise(prm,z + gz,single(1));
    t = tic;
    rng(dg.seedBase + idx,'twister');
    d1 = fcn_denoiserdiag(fd,[],sz,dg.nProbe,dg.nPower);
    rng(dg.seedBase + idx,'twister');
    d2 = fcn_tgdenoiserdiag(fd,sz,dg.nProbe,dg.nPower,v);
    z = gather(extractdata(fd(dlarray(v,'SSCB'))));
    q(end+1) = struct('model',TM(m).key,'asymRand',d1.jacAsym,'rhoRand',d1.jacRho,'asymImg',d2.jacAsym,...
        'rhoImg',d2.jacRho,'psnr',psnr(z,gather(x)),'t',toc(t)); %#ok<AGROW>
    fprintf("    %-14s rand v0: asym %.3e rho %.6f | v0 = v: asym %.3e rho %.6f | PSNR at v %.4f (%.1f s)\n",...
        q(end).model,q(end).asymRand,q(end).rhoRand,q(end).asymImg,q(end).rhoImg,q(end).psnr,q(end).t)
end
q = struct('m',q);
end

function summary2(file,D2,ctx,TM,dg)
fid = fopen(file,'w'); cleaner = onCleanup(@() fclose(fid));
cfg = ctx.cfg0;
out(fid,"Part 2 (retrain): Jacobian asymmetry and spectral radius. Written %s.\n",string(datetime('now')));
out(fid,"As main_sbdm_salsun_tnrdguar.m: 256x256, nProbe %d, nPower %d, probes rng(%d + image index), the same probes for all models.\n",dg.nProbe,dg.nPower,dg.seedBase);
out(fid,"  'rand': fcn_denoiserdiag (v0 = rand); 'img': fcn_tgdenoiserdiag (v0 = the noisy test image).\n");
out(fid,"Old TNRD (lr 1e-3 const) and SA (frozen angles, and b18 not frozen): cached results of tnrdguar part 2 (sbdm_tg_part2_results.mat).\n");
O = load(fullfile(ctx.dat,'sbdm_tg_part2_results.mat'),'D2'); O = O.D2;
oldNames = ["TNRD S1" "TNRD S3" "TNRD L" "SA b18 frozen" "SA b3 frozen" "SA b1 frozen" "SA b18 not frozen"];
models = [oldNames(1:3) + " old" [TM.key] oldNames(4:end)];
cols = ["asym rand" "rho rand" "asym img" "rho img"];
flds = ["asymRand" "rhoRand" "asymImg" "rhoImg"];
imgs = cfg.imgIdx; nI = numel(imgs);
OV = nan(numel(cfg.sigmas),numel(models),4); OM = OV; PZ = nan(numel(cfg.sigmas),numel(models));
for s = 1:numel(cfg.sigmas)
    out(fid,"\n==================== sigma = %d/255 ====================\n",cfg.sigmas(s));
    A = nan(numel(models),4,nI); Pz = nan(numel(models),nI);
    for i = 1:nI
        qs = {};
        if ~isempty(O{s,i}), qs{end+1} = O{s,i}.m; end %#ok<AGROW>
        if ~isempty(D2{s,i}), qs{end+1} = D2{s,i}.m; end %#ok<AGROW>
        for u = 1:numel(qs)
            q = qs{u}; isOld = u == 1 && ~isempty(O{s,i});
            for k = 1:numel(q)
                nm = string(q(k).model);
                if isOld && startsWith(nm,"TNRD"), nm = nm + " old"; end
                m = find(models == nm);
                if isempty(m), continue, end
                for c = 1:4, A(m,c,i) = q(k).(flds(c)); end
                Pz(m,i) = q(k).psnr;
            end
        end
    end
    for c = 1:4
        printtable(fid,cols(c) + " (per image)",models.',compose("kodim%02d",imgs),squeeze(A(:,c,:)),"%.4e")
    end
    printtable(fid,"Mean over the 4 images",models.',[cols "PSNR at v"],[mean(A,3,'omitnan') mean(Pz,2,'omitnan')],"%.4e")
    printtable(fid,"Max over the 4 images",models.',cols,max(A,[],3,'omitnan'),"%.4e")
    OV(s,:,:) = mean(A,3,'omitnan'); OM(s,:,:) = max(A,[],3,'omitnan'); PZ(s,:) = mean(Pz,2,'omitnan');
end
sgRows = compose("sigma %d",cfg.sigmas(:));
out(fid,"\n==================== Over sigma (mean over the 4 images; rows = models, cols = sigma) ====================\n");
for c = 1:4
    printtable(fid,cols(c),models.',compose("sigma %d",cfg.sigmas),squeeze(OV(:,:,c)).',"%.4e")
end
out(fid,"\n==================== Over sigma (max over the 4 images; rows = models, cols = sigma) ====================\n");
for c = 1:4
    printtable(fid,cols(c),models.',compose("sigma %d",cfg.sigmas),squeeze(OM(:,:,c)).',"%.4e")
end
printtable(fid,"PSNR at v (mean over the 4 images)",models.',compose("sigma %d",cfg.sigmas),PZ.',"%.4f")
sk = ctx.ST('skips');
out(fid,"\n---- Skipped / failed / missing units so far (%d) ----\n",numel(sk));
for k = 1:numel(sk), out(fid,"  %s\n",sk(k)); end
end

%% ----- part 3 -----
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

function fb = tnbank(ctx,T,rec)
dev = @(z) z; if ctx.useGpu, dev = @(z) gpuArray(z); end
sg = ctx.cfg0.sigmas; fb = cell(1,numel(sg));
for s = 1:numel(sg)
    S = load(ftn2(ctx,sg(s),T,rec)); prm = S.q.prm;
    if ctx.useGpu, prm = dlupdate(@gpuArray,prm); end
    fb{s} = @(z) double(gather(extractdata(fcn_convtnrddenoise(prm,dlarray(dev(single(z)),'SSCB'),single(1)))));
end
end

function q = runred2(fb,name,X,V,Hf,Htf,schedule,sgBank,rd,clip)
R = struct([]);
for e = rd.etas
    for b = rd.betas
        t = tic;
        r = fcn_tgredrun(fb,sgBank,V,X,Hf,Htf,schedule,e,b,rd.nIters,clip);
        r.eta = e; r.beta = b; r.t = toc(t);
        R = [R r]; %#ok<AGROW>
        fprintf("    %-12s clip %d eta %.1f beta %.1f: PSNR %.4f SSIM %.4f res(T) %.3e blowup %d (%.1f s)\n",...
            name,clip,e,b,r.psnrEnd,r.ssimEnd,r.res(end),r.blowup,r.t)
    end
end
q.R = R;
end

function summary3(file,R3n,ctx,rd,RDM,schedule,sgBank)
fid = fopen(file,'w'); cleaner = onCleanup(@() fclose(fid));
cfg = ctx.cfg0; imgs = cfg.imgIdx; nI = numel(imgs); N = prod(cfg.cropSize);
O = load(fullfile(ctx.dat,'sbdm_tg_part3_results.mat'),'R3s'); O = O.R3s;
for k = 1:numel(O)
    if startsWith(O(k).method,"TNRD"), O(k).method = O(k).method + " old"; end
end
R3 = [O R3n];
out(fid,"Part 3 (retrain): RED deblurring. Written %s.\n",string(datetime('now')));
out(fid,"As main_sbdm_salsun_tnrdguar.m: blur %dx%d Gaussian std %.1f (circular), noise 5/255 (rng(%d + image index)), 256x256 centre crops,\n",rd.blurSize,rd.blurSize,rd.blurStd,rd.seedBase);
out(fid,"z <- clip(z - eta*(H'(Hz - v) + beta*(z - f(z; sigma(t))))), z(0) = v, eps = 0, %d its, sigma(t) = 25 -> 2 (log), model bank (nearest sigma),\n",rd.nIters);
k = nearestk(schedule,sgBank);
out(fid,"its per model sigma %s = %s. clip 0 = no clipping.\n",mat2str(round(sgBank*255)),mat2str(accumarray(k,1,[numel(sgBank) 1]).'));
out(fid,"none, SA b18, old TNRD (lr 1e-3 const): cached results of tnrdguar part 3 (sbdm_tg_part3_results.mat). New: this run.\n");
out(fid,"Residual r(t) = ||z(t) - z(t-1)||_2, shown as r/sqrt(N), N = %d. Stable := finite, r(T) <= r(T-50), r(T)/sqrt(N) < 1e-3.\n",N);
out(fid,"In the per-method tables, the last column is: mean (PSNR, SSIM), MAX over images (residual, ratio), number of images (stable, blow-up).\n");
cases = struct('name',{"SA b18","SA b18","TNRD L old","TNRD L old","TNRD S3 old"},'clip',{true,false,true,false,true});
for m = 1:numel(RDM)
    for clip = [true false], cases(end+1) = struct('name',RDM(m).name,'clip',clip); end %#ok<AGROW>
end
combo = strings(0,1); [EE,BB] = ndgrid(rd.etas,rd.betas);
for c = 1:numel(EE), combo(c,1) = sprintf("eta %.1f beta %.1f",EE(c),BB(c)); end
nC = numel(combo);
cn = strings(1,0); MP = nan(nC,0); MS = MP; MR = MP; MST = MP; MBL = MP;
for cc = cases
    Pm = nan(nC,nI); Sm = Pm; Rf = Pm; Rr = Pm; St = Pm; Bl = Pm; found = false;
    for i = 1:nI
        q = R3(arrayfun(@(r) r.img == imgs(i) && r.method == cc.name && r.clip == cc.clip,R3));
        if isempty(q) || isempty(q(1).runs), continue, end
        q = q(1); found = true;
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
    if ~found, continue, end
    out(fid,"\n==================== %s, clip %d ====================\n",cc.name,cc.clip);
    cl = [compose("kodim%02d",imgs) "mean"];
    printtable(fid,"PSNR [dB]",combo,cl,[Pm mean(Pm,2)],"%.4f")
    printtable(fid,"SSIM",combo,cl,[Sm mean(Sm,2)],"%.4f")
    printtable(fid,"Final residual r(T)/sqrt(N)",combo,[compose("kodim%02d",imgs) "max"],[Rf max(Rf,[],2)],"%.3e")
    printtable(fid,"r(T)/r(T-50)",combo,[compose("kodim%02d",imgs) "max"],[Rr max(Rr,[],2)],"%.3e")
    printtable(fid,"Stable (1/0)",combo,[compose("kodim%02d",imgs) "count"],[St sum(St,2)],"%d")
    printtable(fid,"Blow-up (1/0)",combo,[compose("kodim%02d",imgs) "count"],[Bl sum(Bl,2)],"%d")
    cn(end+1) = cc.name + " c" + double(cc.clip); %#ok<AGROW>
    MP(:,end+1) = mean(Pm,2); MS(:,end+1) = mean(Sm,2); MR(:,end+1) = max(Rf,[],2); %#ok<AGROW>
    MST(:,end+1) = sum(St,2); MBL(:,end+1) = sum(Bl,2); %#ok<AGROW>
end
out(fid,"\n==================== All methods side by side (c1 = clip, c0 = no clip) ====================\n");
printtable(fid,"PSNR [dB], mean over the 4 images",combo,cn,MP,"%.4f")
printtable(fid,"SSIM, mean over the 4 images",combo,cn,MS,"%.4f")
printtable(fid,"Final residual r(T)/sqrt(N), MAX over the 4 images",combo,cn,MR,"%.3e")
printtable(fid,"Stable: number of images (of 4)",[combo; "total (of 36)"],cn,[MST; sum(MST,1)],"%d")
printtable(fid,"Blow-up: number of images (of 4)",[combo; "total (of 36)"],cn,[MBL; sum(MBL,1)],"%d")
out(fid,"\n==================== Residual r(t)/sqrt(N) at selected its (mean over images) ====================\n");
tl = unique([1 10 50 100 150 200 250 rd.nIters]); tl = tl(tl <= rd.nIters);
for cc = cases
    Mc = nan(nC,numel(tl)); found = false;
    for c = 1:nC
        acc = [];
        for i = 1:nI
            q = R3(arrayfun(@(r) r.img == imgs(i) && r.method == cc.name && r.clip == cc.clip,R3));
            if isempty(q) || isempty(q(1).runs), continue, end
            r = q(1).runs.R([q(1).runs.R.eta] == EE(c) & [q(1).runs.R.beta] == BB(c));
            if isempty(r) || numel(r.res) < max(tl), continue, end
            acc = [acc; r.res(tl).'/sqrt(N)]; %#ok<AGROW>
        end
        if ~isempty(acc), Mc(c,:) = mean(acc,1); found = true; end
    end
    if found, printtable(fid,sprintf("%s, clip %d",cc.name,cc.clip),combo,compose("t=%d",tl),Mc,"%.3e"), end
end
sk = ctx.ST('skips');
out(fid,"\n---- Skipped / failed / missing units (%d) ----\n",numel(sk));
for k2 = 1:numel(sk), out(fid,"  %s\n",sk(k2)); end
end

%% ----- units, skips, printing (as main_sbdm_salsun_tnrdguar.m) -----
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
ftmp = string(f) + ".tmp.mat";
save(ftmp,'q','-v7.3'); movefile(ftmp,f);
fprintf("%s  %-32s done (%.1f s)\n",string(datetime('now')),lab,q.tWall);
end

function addskip(ctx,msg)
ctx.ST('skips') = [ctx.ST('skips'); string(msg)];
fprintf("  %s\n",msg)
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
