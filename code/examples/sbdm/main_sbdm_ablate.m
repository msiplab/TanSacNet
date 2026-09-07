%[text] # Ablation: what does the nonlinearity buy?
%[text] The proposed denoiser applies $\phi_{\sigma,p}(c)=\sigma(\kappa_p/g_p)\tanh(g_pc/\sigma)$ to the AC coefficients. Removing that nonlinearity makes the whole denoiser a linear map, and Tweedie's formula then returns a score that is linear in $\mathbf{u}$ -- the score of a *Gaussian*. A score-based diffusion model built on it therefore has a Gaussian prior, whatever the data. The prediction is that the denoising PSNR degrades only moderately while the RED loop degrades far more.
%[text] Three variants share the LSUN basis and differ only in what acts on the AC coefficients:
%[text] - `N` nonlinear (the proposal): $\phi_{\sigma,p}$ as above, $2(P-1)$ parameters per stage;
%[text] - `L` linear gain: $c\mapsto(1-\kappa_p)c$, $P-1$ parameters, and $\sigma$-independent as a map;
%[text] - `T` truncation: keep the DC and the first $P'-1$ AC channels, zero the rest, no shrinkage parameters at all. This is the "narrow the channel count instead of shrinking" option; the angles are trained with the truncation in place, so the network learns to route the energy it needs into the channels it keeps.
%[text] `N` is read from the Experiment 2 cache of *main_sbdm.m*; `L` and `T` are trained here on the same patches with the same schedule.
%[text] Copyright (c) 2026, Shogo MURAMATSU, All rights reserved.
clc, clear, close all
here = pwd;
run(fullfile('..','..','setpath.m'))
cd(here)
warning('off','MATLAB:rmpath:DirNotFound')
rmpath(fullfile('/home/shogo/Workspace/GitHub/SaivDr','mexcodes'))
warning('on','MATLAB:rmpath:DirNotFound')
addpath(fullfile('..','..','mexcodes'))

datfolder = fullfile('..','..','..','data');
szOrg = [64 64]; stride = [4 4]; ovlp = [3 3];
sigmaMin = single(2/255); sigmaMax = single(25/255);
sigmaTest = single([2 5 10 15 25]/255);
nItersLsun = 1500; lrLsun = 5e-3; mbSize = 32;
trainIdx = 1:8; evalIdx = 9:24; nCrops = 256;
deblurIdx = 9;
useGpu = canUseGPU;

timeBudget = 800;
tRun = tic;
remaining = @() timeBudget - toc(tRun);
pending = false;

fcn_seed(0)
[anet,snet,info] = fcn_lsunpair(szOrg,stride,ovlp);
fcn_seed(1)
[~,trainSet] = fcn_loadkodak(datfolder,trainIdx,szOrg,nCrops,useGpu);
evalSet = fcn_loadkodak(datfolder,evalIdx,szOrg,0,useGpu);
fprintf("channels %d, training patches %d, evaluation images %d\n",...
    info.nChs,numel(trainSet),numel(evalSet))

%%
%[text] ## Training and conditional denoising
variants = { 'N',NaN ; 'L',NaN ; 'T',8 ; 'T',4 };
names = {'N nonlinear','L linear gain','T keep 8/16','T keep 4/16'};
abl = struct('name',{},'psnr',{},'ssim',{},'nPrm',{},'variant',{},'nKeep',{});
prmAll = cell(1,size(variants,1));
for k = 1:size(variants,1)
    vr = variants{k,1}; nKeep = variants{k,2};
    if strcmp(vr,'N')
        S = load(fullfile(datfolder,'sbdm_exp2_3.mat'));   % LSUN-U cond. (S=1)
        prm = fcn_lsunmovegpu(S.prm,useGpu);
        rec = struct('name',names{k},'psnr',S.rec.psnr,'ssim',S.rec.ssim,...
            'nPrm',S.rec.nPrm,'variant',vr,'nKeep',NaN);
        fprintf("[ablation] %s taken from the Experiment 2 cache\n",names{k});
    else
        f1 = fullfile(datfolder,sprintf('sbdm_abl_%s%d.mat',vr,...
            max(nKeep,0)*isfinite(nKeep)));
        if isfile(f1)
            S = load(f1); prm = S.prm; rec = S.rec;
            fprintf("[ablation] loaded %s\n",names{k});
        else
            fprintf("\n[ablation] %s\n",names{k});
            if remaining() <= 0
                pending = true; fprintf("  skipped: out of time budget\n"); continue
            end
            ck = fullfile(datfolder,sprintf('sbdm_ckpt_abl_%s%d.mat',vr,...
                max(nKeep,0)*isfinite(nKeep)));
            prm = fcn_lsunmovegpu(fcn_lsuninitprm(info,1,false,0.9,2.0),useGpu);
            [prm,~,isDone] = fcn_trainvar(anet,snet,info,prm,trainSet,...
                [sigmaMin sigmaMax],nItersLsun,lrLsun,mbSize,ck,remaining(),vr,nKeep);
            if ~isDone
                pending = true; fprintf("  unfinished, re-run to continue\n"); continue
            end
            nPrm = fcn_countangles(prm);
            if strcmp(vr,'L'), nPrm = nPrm + numel(prm.a{1}); end
            fdenSz = @(v,sg,sz) fcn_denoisesz(prm,v,sg,sz,stride,ovlp,vr,nKeep);
            [ps,ss] = fcn_evalfullres(fdenSz,evalSet,sigmaTest);
            rec = struct('name',names{k},'psnr',mean(ps,2),'ssim',mean(ss,2),...
                'nPrm',nPrm,'variant',vr,'nKeep',nKeep);
            save(f1,'rec','prm')
        end
    end
    fprintf("  PSNR = %s dB\n  SSIM = %s\n  params = %d\n",...
        mat2str(round(rec.psnr',2)),mat2str(round(rec.ssim',4)),rec.nPrm)
    abl(k) = rec; prmAll{k} = prm;
end

%%
%[text] ## The same estimators inside the score-based RED loop
%[text] If the linear variants are Gaussian priors in disguise, this is where it should show.
if ~pending
blurKernel = fspecial('gaussian',9,2.0);
Hf  = @(z) imfilter(z,blurKernel,'circular');
Htf = @(z) imfilter(z,rot90(blurKernel,2),'circular');
sigmaW = 5/255; mu3 = 0.5; nIters3 = 300;
lambdaGrid = [0.5 1.0 2.0];
schedule = exp(linspace(log(sigmaMax),log(sigmaMin),nIters3));

Xd = double(gather(evalSet{deblurIdx-evalIdx(1)+1}));
fcn_seed(4)
Vd = Hf(Xd) + sigmaW*randn(size(Xd));
fprintf("\nkodim%02d %s: observation %.2f dB\n",deblurIdx,mat2str(size(Xd)),...
    psnr(min(max(Vd,0),1),Xd))

red = struct('name',{},'lambda',{},'psnr',{},'ssim',{});
for k = 1:numel(abl)
    if isempty(abl(k).name), continue, end
    vr = abl(k).variant; nKeep = abl(k).nKeep; prm = prmAll{k};
    fden = @(z,sg) double(gather(fcn_denoisesz(prm,fcn_dev(single(z),useGpu),...
        single(sg),size(Xd),stride,ovlp,vr,nKeep)));
    fr = fullfile(datfolder,sprintf('sbdm_ablred_%d.mat',k));
    if isfile(fr)
        S = load(fr); rec = S.rec;
    else
        if remaining() <= 0
            pending = true; fprintf("  %s: out of time budget\n",abl(k).name); continue
        end
        best = -inf; bestLam = NaN; bestZ = [];
        for il = 1:numel(lambdaGrid)
            [p,~,z] = fcn_redrun(fden,Vd,Xd,Hf,Htf,schedule,mu3,...
                lambdaGrid(il),nIters3,4,sigmaMin,sigmaMax,sigmaW,size(Xd));
            if p > best, best = p; bestLam = lambdaGrid(il); bestZ = z; end
        end
        rec = struct('name',abl(k).name,'lambda',bestLam,'psnr',best,...
            'ssim',ssim(bestZ,Xd));
        save(fr,'rec')
    end
    fprintf("  %-14s annealed RED: %.2f dB / %.4f (lambda %.1f)\n",...
        rec.name,rec.psnr,rec.ssim,rec.lambda)
    red(k) = rec;
end
end

%%
%[text] ## Summary
if ~pending
    T = table(string({abl.name})',[abl.nPrm]',...
        round(cell2mat({abl.psnr})',2),round(cell2mat({abl.ssim})',4),...
        'VariableNames',{'variant','nPrm','PSNR_by_sigma','SSIM_by_sigma'});
    disp("conditional denoising, mean over kodim09-24"), disp(T)
    if ~isempty(red)
        disp("annealed score-based RED on kodim09"), disp(struct2table(red))
    end
else
    fprintf("\n*** work pending: run this script again ***\n");
end

%%
%[text] ## Definitions of local functions

function ac = fcn_shrink(ac,prm,s,sigma,nAc,variant,nKeep)
% what acts on the AC coefficients, and nothing else, distinguishes the variants
switch variant
    case 'N'
        kappa = reshape(sigmoid(prm.a{s}),1,1,nAc);
        gain  = reshape(exp(prm.b{s}),1,1,nAc);
        ac = ac - sigma.*(kappa./gain).*tanh(gain.*ac./sigma);
    case 'L'
        % a linear per-channel gain: scale equivariant for free, but the map
        % cannot depend on sigma at all
        kappa = reshape(sigmoid(prm.a{s}),1,1,nAc);
        ac = ac.*(1-kappa);
    case 'T'
        % keep the DC and the first nKeep-1 AC channels, zero the rest
        m = zeros(1,1,nAc,'like',extractdata(ac));
        m(1,1,1:min(nKeep-1,nAc)) = 1;
        ac = ac.*m;
    otherwise
        error("unknown variant %s",variant)
end
end

function xh = fcn_denoisevar(anet,snet,info,prm,v,sigma,variant,nKeep)
nStages = numel(prm.th);
nAc = info.nChs - 1;
xh = v;
for s = 1:nStages
    [anet,snet] = fcn_lsunsetangles(anet,snet,info,prm.th{s});
    [ac,dc] = forward(anet,xh);
    ac = fcn_shrink(ac,prm,s,sigma,nAc,variant,nKeep);
    xh = forward(snet,ac,dc);
end
end

function [loss,grads] = fcn_lossvar(anet,snet,info,prm,v,xstar,sigma,variant,nKeep)
xh = fcn_denoisevar(anet,snet,info,prm,v,sigma,variant,nKeep);
loss = mean((xstar-xh).^2,'all');
grads = dlgradient(loss,prm);
end

function [prm,hist,done] = fcn_trainvar(anet,snet,info,prm,exemplars,...
    sigmaRange,nIters,learnRate,miniBatchSize,ckptFile,maxSeconds,variant,nKeep)
avgG = []; avgSqG = []; it0 = 0; hist = zeros(nIters,1);
if ~isempty(ckptFile) && isfile(ckptFile)
    C = load(ckptFile);
    prm = C.prm; avgG = C.avgG; avgSqG = C.avgSqG; it0 = C.it0;
    hist(1:numel(C.hist)) = C.hist;
    fprintf("    resuming at iteration %d of %d\n",it0,nIters);
end
tStart = tic; done = true;
for it = it0+1:nIters
    [x,v,sigma] = fcn_minibatch(exemplars,sigmaRange,true,miniBatchSize);
    [loss,grads] = dlfeval(@fcn_lossvar,anet,snet,info,prm,v,x,sigma,variant,nKeep);
    [prm,avgG,avgSqG] = adamupdate(prm,grads,avgG,avgSqG,it,learnRate);
    hist(it) = double(gather(extractdata(loss)));
    if mod(it,max(1,round(nIters/10)))==0
        fprintf("    iter %5d/%d  loss = %.4e\n",it,nIters,hist(it));
    end
    if toc(tStart) > maxSeconds && it < nIters
        it0 = it; done = false; break
    end
    it0 = it;
end
if ~isempty(ckptFile)
    hist = hist(1:it0);
    save(ckptFile,'prm','avgG','avgSqG','it0','hist','-v7.3');
end
end

function y = fcn_denoisesz(prm,v,sigma,sz,stride,ovlp,variant,nKeep)
[anet,snet,info] = fcn_paircached(sz,stride,ovlp);
y = extractdata(fcn_denoisevar(anet,snet,info,prm,dlarray(v,'SSCB'),sigma,...
    variant,nKeep));
end

function [anet,snet,info] = fcn_paircached(sz,stride,ovlp)
persistent M
if isempty(M), M = containers.Map; end
key = mat2str(sz);
if ~isKey(M,key)
    [anet,snet,info] = fcn_lsunpair(sz,stride,ovlp);
    M(key) = {anet,snet,info};
else
    pr = M(key); anet = pr{1}; snet = pr{2}; info = pr{3};
end
end

function [p,curve,z] = fcn_redrun(fden,V,X,Hf,Htf,schedule,mu,lambda,nIters,...
    mode,sigmaMin,sigmaMax,sigmaW,sz,seed)
if nargin < 15, seed = 1; end
rng(seed)
z = V; curve = zeros(nIters,1);
for it = 1:nIters
    switch mode
        case 2, st = sigmaMin;
        case 3, st = sigmaMax;
        otherwise, st = schedule(it);
    end
    gData = Htf(Hf(z)-V);
    if mode == 1
        gReg = 0; lam = 0;
    else
        gReg = z - fden(z,st); lam = lambda;
    end
    z = z - mu*(gData + lam*gReg);
    if mode == 5
        z = z + sigmaW*sqrt(2*mu)*randn(sz);
    end
    z = min(max(z,0),1);
    curve(it) = psnr(z,X);
end
p = curve(end);
end

function fcn_seed(s)
rng(s)
if canUseGPU, gpurng(s); end
end

function y = fcn_dev(x,useGpu)
y = x; if useGpu, y = gpuArray(x); end
end

function prm = fcn_lsunmovegpu(prm,useGpu)
if useGpu, prm = dlupdate(@gpuArray,prm); end
end

function n = fcn_countangles(prm)
n = 0;
for s = 1:numel(prm.th)
    for k = 1:numel(prm.th{s}), n = n + numel(prm.th{s}{k}); end
end
end
