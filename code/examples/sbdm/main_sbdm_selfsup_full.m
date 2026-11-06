%[text] # Self-supervised LSUN at full resolution (SURE), kodim09-24
%[text] Full-resolution version of main_sbdm_selfsup.m for the paper.
%[text] Arms, all starting from the same noise-conditional uniform model U0 (S = 1), trained on patches of kodim01-08:
%[text] - `U0` the uniform model as is;
%[text] - `C` U0 adapted by SURE on the noisy image, angles shared by all blocks (NSCLT);
%[text] - `L` U0 expanded to per-block angles (identical denoiser at the start), adapted by SURE (LSUN).
%[text] The noise of image i is drawn with seed 7000+i, as fcn_evalfullres does, so U0 reproduces the LSUN-U cond. (S=1) row of Experiment 2.
%[text] The clean image is used only for PSNR, SSIM and the true MSE.
%[text] Every result is cached under data/sbdm_ssf_*.mat, and the SURE training is checkpointed, so the script can be stopped and run again.
%[text] Mode. By default the script only checks both image sizes and estimates the run time, then stops. For the full run, start MATLAB with the environment variable SSF_FULL=1:
%[text]   SSF_FULL=1 nohup matlab -licmode onlinelicensing -batch "setpath; cd examples/sbdm; main_sbdm_selfsup_full" > ~/Documents/SBDMwLSUN/logs/selfsup_full.log 2>&1 &
%[text] Run setpath in code/ first.

%%
clc, clear, close all
fcn_seed(0)
datfolder = fullfile('..','..','..','data');
warning('off','MATLAB:rmpath:DirNotFound')
rmpath(fullfile('/home/shogo/Workspace/GitHub/SaivDr','mexcodes'))
warning('on','MATLAB:rmpath:DirNotFound')
addpath(fullfile('..','..','mexcodes'))

isFull = strcmp(getenv('SSF_FULL'),'1');
szOrg     = [64 64];        % training patches of U0
stride    = [4 4];
ovlp      = [3 3];
trainIdx  = 1:8;  nCrops = 256;
evalIdx   = 9:24;
sigmaMin  = single(2/255);  % noise-level range of U0, as Experiment 2
sigmaMax  = single(25/255);
sigmaList = single([10 25]/255);
nItersLsun = 1500; lrLsun = 5e-3; mbSize = 32;   % U0, as main_sbdm.m
nItersSs   = 400;           % SURE adaptation, as main_sbdm_selfsup.m
seedNoise  = 7000;          % as fcn_evalfullres
chunkSec   = 600;           % write a checkpoint every 10 min of training
nTime      = 20;            % timed iterations per arm for the estimate
useGpu = canUseGPU;
fprintf("GPU: %d, mode: %s\n",useGpu,string(isFull).replace(["true","false"],["full run","estimate only"]))

%%
%[text] ## Data: kodim09-24, grayscale, full resolution
evalSet = fcn_loadkodak(datfolder,evalIdx,szOrg,0,useGpu);
szs  = cellfun(@size,evalSet,'UniformOutput',false);
szKeys = cellfun(@mat2str,szs,'UniformOutput',false);
uKeys = unique(szKeys);
for k = 1:numel(uKeys)
    fprintf("size %s: %s\n",uKeys{k},...
        strjoin(compose("kodim%02d",evalIdx(strcmp(szKeys,uKeys{k}))),' '))
end

%%
%[text] ## LSUN pairs, one per image size, and a check of both sizes
%[text] Each pair is checked for the block grid, for Parseval tightness with uniform and with per-block (randomly perturbed) angles, and by one SURE evaluation of each kind.
pairs = containers.Map;
for k = 1:numel(uKeys)
    i0 = find(strcmp(szKeys,uKeys{k}),1);
    sz = szs{i0};
    [an,sn,inf_] = fcn_lsunpair(sz,stride,ovlp);
    pairs(uKeys{k}) = {an,sn,inf_};
    assert(inf_.nBlocks == prod(sz./stride),"block grid mismatch for size %s",uKeys{k})
    prmT  = fcn_lsunmovegpu(fcn_lsuninitprm(inf_,1,false,0.9,2.0),useGpu);
    prmTL = fcn_lsunexpandprm(prmT,inf_);
    for r = 1:numel(prmTL.th{1})
        prmTL.th{1}{r} = prmTL.th{1}{r} + ...
            0.1*randn(size(prmTL.th{1}{r}),'like',extractdata(prmTL.th{1}{r}));
    end
    xd = dlarray(evalSet{i0},'SSCB');
    eU = fcn_relerr(fcn_lsunpairapply(an,sn,inf_,prmT.th{1},xd),xd);
    eL = fcn_relerr(fcn_lsunpairapply(an,sn,inf_,prmTL.th{1},xd),xd);
    vd = dlarray(evalSet{i0} + sigmaMax*randn(sz,'single','like',evalSet{i0}),'SSCB');
    sU = fcn_lsunsure(an,sn,inf_,prmT,vd,sigmaMax);
    sL = fcn_lsunsure(an,sn,inf_,prmTL,vd,sigmaMax);
    fprintf("size %s: %d blocks, ||DEx-x||/||x|| uniform %.2e, per-block %.2e; SURE uniform %.4e, per-block %.4e\n",...
        uKeys{k},inf_.nBlocks,eU,eL,gather(extractdata(sU)),gather(extractdata(sL)))
    assert(eU < 1e-4 && eL < 1e-4,"size %s: LSUN pair is not tight",uKeys{k})
end

%%
%[text] ## Run-time estimate from one image of each size
%[text] Times `nTime` SURE iterations of arms C and L (after 2 warm-up iterations) and one evaluation, at sigma = 25/255, with untrained parameters (the cost does not depend on their values). No file is written.
tIt  = zeros(numel(uKeys),2);   % s per iteration, arms C and L
tEv  = zeros(numel(uKeys),1);   % s per evaluation (SURE pass, PSNR, SSIM)
nImg = zeros(numel(uKeys),1);
for k = 1:numel(uKeys)
    i0 = find(strcmp(szKeys,uKeys{k}),1);
    nImg(k) = sum(strcmp(szKeys,uKeys{k}));
    pr = pairs(uKeys{k}); an = pr{1}; sn = pr{2}; inf_ = pr{3};
    x = evalSet{i0}; sz = size(x);
    v = x + sigmaMax*randn(sz,'single','like',x);
    prmT = fcn_lsunmovegpu(fcn_lsuninitprm(inf_,1,false,0.9,2.0),useGpu);
    starts = {prmT, fcn_lsunexpandprm(prmT,inf_)};
    for a = 1:2
        fcn_trainlsunsure(an,sn,inf_,starts{a},{v},sigmaMax,sigmaMax,2,lrLsun,1,false,'',inf);
        if useGpu, wait(gpuDevice); end
        t0 = tic;
        fcn_trainlsunsure(an,sn,inf_,starts{a},{v},sigmaMax,sigmaMax,nTime,lrLsun,1,false,'',inf);
        if useGpu, wait(gpuDevice); end
        tIt(k,a) = toc(t0)/nTime;
    end
    t0 = tic;
    [~,fu] = fcn_lsunsure(an,sn,inf_,starts{2},dlarray(v,'SSCB'),sigmaMax);
    z = min(max(double(gather(extractdata(fu))),0),1);
    ref = double(gather(x)); psnr(z,ref); ssim(z,ref);
    tEv(k) = toc(t0);
    fprintf("size %s (kodim%02d): %.3f s/iter C, %.3f s/iter L, %.2f s/eval => %.1f min C, %.1f min L per run\n",...
        uKeys{k},evalIdx(i0),tIt(k,1),tIt(k,2),tEv(k),tIt(k,1)*nItersSs/60,tIt(k,2)*nItersSs/60)
end
tTot = sum(nImg.*numel(sigmaList).*(nItersSs*(tIt(:,1)+tIt(:,2)) + 3*tEv));
fprintf("estimated total: %.1f h for %d images x %d noise levels x 3 arms (cached results are skipped)\n",...
    tTot/3600,numel(evalIdx),numel(sigmaList))
if ~isfile(fullfile(datfolder,'sbdm_exp2_3.mat')) && ~isfile(fullfile(datfolder,'sbdm_ssf_pre.mat'))
    fprintf("  plus training U0 (%d iterations, batch %d, 64x64), not included\n",nItersLsun,mbSize)
end
if ~isFull
    fprintf("estimate only: set SSF_FULL=1 for the full run\n")
    return
end

%%
%[text] ## U0: noise-conditional uniform LSUN (S = 1), as Experiment 2 "LSUN-U cond. (S=1)"
f0   = fullfile(datfolder,'sbdm_exp2_3.mat');
fpre = fullfile(datfolder,'sbdm_ssf_pre.mat');
rec2 = [];
if isfile(f0)
    S = load(f0); prm0 = S.prm; rec2 = S.rec; fprintf("U0 loaded from %s\n",f0)
elseif isfile(fpre)
    S = load(fpre); prm0 = S.prm0; fprintf("U0 loaded from %s\n",fpre)
else
    fprintf("training U0 on kodim01-08, sigma in [%g,%g]/255 (%d iterations)\n",...
        sigmaMin*255,sigmaMax*255,nItersLsun)
    [anet64,snet64,info64] = fcn_lsunpair(szOrg,stride,ovlp);
    fcn_seed(1)
    [~,trainSet] = fcn_loadkodak(datfolder,trainIdx,szOrg,nCrops,useGpu);
    prm0 = fcn_lsunmovegpu(fcn_lsuninitprm(info64,1,false,0.9,2.0),useGpu);
    ck = fullfile(datfolder,'sbdm_ckpt_ssf_pre.mat');
    done = false;
    while ~done
        [prm0,~,done] = fcn_trainlsun(anet64,snet64,info64,prm0,trainSet,...
            [sigmaMin sigmaMax],nItersLsun,lrLsun,mbSize,true,ck,chunkSec);
    end
    save(fpre,'prm0')
end
assert(numel(prm0.th) == 1 && size(prm0.th{1}{1},2) == 1,"U0 must be uniform with S = 1")
prm0 = fcn_lsunmovegpu(prm0,useGpu);

%%
%[text] ## SURE adaptation on each noisy image
arms = {'U0','C','L'};
nI = numel(evalIdx); nS = numel(sigmaList);
P = nan(nI,3,nS); Q = P; R = P; M = P; Tsec = P;   % PSNR, SSIM, SURE, true MSE, seconds
psNoisy = nan(nI,nS); ssNoisy = nan(nI,nS);
nPrm = nan(nI,3);
for s = 1:nS
    sg = sigmaList(s);
    for i = 1:nI
        x = evalSet{i}; sz = size(x); ref = double(gather(x));
        pr = pairs(mat2str(sz)); anet = pr{1}; snet = pr{2}; info = pr{3};
        fcn_seed(seedNoise+i)
        v  = x + sg*randn(sz,'single','like',x);
        vd = dlarray(v,'SSCB');
        zN = min(max(double(gather(v)),0),1);
        psNoisy(i,s) = psnr(zN,ref); ssNoisy(i,s) = ssim(zN,ref);
        fprintf("\nsigma %g/255, kodim%02d %s: noisy %.2f dB\n",sg*255,evalIdx(i),mat2str(sz),psNoisy(i,s))
        for a = 1:3
            tag = sprintf('%02d_s%02d_%s',evalIdx(i),round(sg*255),arms{a});
            f1 = fullfile(datfolder,['sbdm_ssf_' tag '.mat']);
            if isfile(f1)
                S = load(f1); rec = S.rec;
            else
                t0 = tic; hist = [];
                switch arms{a}
                    case 'U0', prm = prm0;
                    case 'C',  prm = prm0;
                    case 'L',  prm = fcn_lsunexpandprm(prm0,info);
                end
                ck = fullfile(datfolder,['sbdm_ckpt_ssf_' tag '.mat']);
                if a > 1
                    fcn_seed(100+i)
                    done = false;
                    while ~done   % resumes from ck after an interruption
                        [prm,hist,done] = fcn_trainlsunsure(anet,snet,info,prm,{v},sg,sg,...
                            nItersSs,lrLsun,1,false,ck,chunkSec);
                    end
                end
                [sr,fu] = fcn_lsunsure(anet,snet,info,prm,vd,sg);
                y = double(gather(extractdata(fu)));
                z = min(max(y,0),1);
                rec = struct('arm',arms{a},'psnr',psnr(z,ref),'ssim',ssim(z,ref),...
                    'sure',double(gather(extractdata(sr))),'mse',mean((y-ref).^2,'all'),...
                    'nPrm',fcn_countprm(prm),'hist',hist,'sec',toc(t0));
                prm = dlupdate(@gather,prm); %#ok<NASGU>
                save(f1,'rec','prm')
                if isfile(ck), delete(ck), end
            end
            P(i,a,s) = rec.psnr; Q(i,a,s) = rec.ssim; R(i,a,s) = rec.sure;
            M(i,a,s) = rec.mse;  Tsec(i,a,s) = rec.sec; nPrm(i,a) = rec.nPrm;
            fprintf("  %-2s PSNR %.2f dB  SSIM %.4f  SURE %.4e  true MSE %.4e  (%d prm, %.0f s)\n",...
                arms{a},rec.psnr,rec.ssim,rec.sure,rec.mse,rec.nPrm,rec.sec)
        end
    end
end

%%
%[text] ## Summary
rows = [compose("kodim%02d",evalIdx'); "mean"];
for s = 1:nS
    fprintf("\n===== sigma = %g/255, full resolution, kodim09-24 =====\n",sigmaList(s)*255)
    Ps = [psNoisy(:,s) P(:,:,s)]; Qs = [ssNoisy(:,s) Q(:,:,s)];
    disp("PSNR [dB]"), disp(array2table([Ps; mean(Ps,1)],'VariableNames',[{'noisy'},arms],'RowNames',rows))
    disp("SSIM"),      disp(array2table([Qs; mean(Qs,1)],'VariableNames',[{'noisy'},arms],'RowNames',rows))
    dLC = P(:,3,s)-P(:,2,s);
    fprintf("mean PSNR: U0 %.2f, C %.2f, L %.2f dB; mean SSIM: U0 %.4f, C %.4f, L %.4f\n",...
        mean(P(:,:,s),1),mean(Q(:,:,s),1))
    fprintf("shift variance, L - C: mean %+.2f dB, per image %s\n",mean(dLC),mat2str(round(dLC',2)))
    fprintf("SURE adaptation alone, C - U0: mean %+.2f dB\n",mean(P(:,2,s)-P(:,1,s)))
    fprintf("SURE vs true MSE, mean relative gap (SURE-MSE)/MSE: U0 %+.3f, C %+.3f, L %+.3f\n",...
        mean((R(:,:,s)-M(:,:,s))./M(:,:,s),1))
end
if ~isempty(rec2)
    [~,j] = ismember(round(sigmaList*255),round(single([2 5 10 15 25])));
    fprintf("\ncheck, U0 mean PSNR vs Experiment 2 (same noise): %s vs %s dB\n",...
        mat2str(round(squeeze(mean(P(:,1,:),1))',2)),mat2str(round(rec2.psnr(j)',2)))
end
fprintf("parameters: U0 %d, C %d, L %s (depends on the image size)\n",nPrm(1,1),nPrm(1,2),mat2str(unique(nPrm(:,3))'))
fprintf("training time per run: C %.0f s, L %.0f s (mean)\n",mean(Tsec(:,2,:),'all'),mean(Tsec(:,3,:),'all'))
save(fullfile(datfolder,'sbdm_ssf_results.mat'),'P','Q','R','M','Tsec','psNoisy','ssNoisy','nPrm','evalIdx','sigmaList','arms')

%%
%[text] ## Local functions
function fcn_seed(s)
rng(s)
if canUseGPU, gpurng(s); end
end

function y = fcn_lsunpairapply(anet,snet,info,th,v)
[anet,snet] = fcn_lsunsetangles(anet,snet,info,th);
[ac,dc] = forward(anet,v);
y = forward(snet,ac,dc);
end

function e = fcn_relerr(y,x)
e = sqrt(double(gather(extractdata(sum((y-x).^2,'all')))) / ...
         double(gather(extractdata(sum(x.^2,'all')))));
end

function prm = fcn_lsunmovegpu(prm,useGpu)
if useGpu, prm = dlupdate(@gpuArray,prm); end
end

function n = fcn_countprm(prm)
n = 0;
f = fieldnames(prm);
for i = 1:numel(f)
    v = prm.(f{i});
    if iscell(v)
        for j = 1:numel(v)
            if iscell(v{j})
                for k = 1:numel(v{j}), n = n + numel(v{j}{k}); end
            else
                n = n + numel(v{j});
            end
        end
    else
        n = n + numel(v);
    end
end
end
