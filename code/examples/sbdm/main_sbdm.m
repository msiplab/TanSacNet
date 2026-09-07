%[text] # Score-Based Diffusion Model (SBDM) with LSUN
%[text] Realisation of the AWGN denoising network of Example 10.2 and of the score-based diffusion model of Section 10.2.3 of Muramatsu (2026) with a locally-structured unitary network (LSUN).
%[text] The denoiser of Example 10.2 is $\mathbf{f}_{\bmTheta}(\mathbf{v})=\mathbf{v}-\lambda_\mathrm{g}\mathbf{g}_{\bmTheta}(\mathbf{v})$ with $\mathbf{g}_{\bmTheta}(\mathbf{v})=\mathbf{D}_{\bmtheta}\sum_p\mathbf{S}_{\mathcal{I}_p}^\top\phi_p(\mathbf{S}_{\mathcal{I}_p}\mathbf{E}_{\bmtheta}\mathbf{v})$ and $\mathbf{E}_{\bmtheta}=\mathbf{D}_{\bmtheta}^\top$. Taking an LSUN as $\mathbf{D}_{\bmtheta}$ makes the adjoint relation structural rather than a constraint to be imposed, so the dictionary stays Parseval tight throughout training and the noise level of the coefficients equals the noise level of the image exactly. The latter is what the score estimator of Section 10.2.3 needs.
%[text] Please do not forget to run *setpath* in the top directory of this package, and then return to this directory.
%[text] Requirements: MATLAB R2024b, Deep Learning Toolbox, Image Processing Toolbox
%[text] The oversampled-NSOLT row of Table 1 is produced by the companion script *main_sbdm_nsolt.m*, which needs SaivDr on the path. It is kept separate because both packages ship identically named mex kernels (`fcn_orthmtxgen_*_mex`) in non-package folders, so only one of the two dictionaries can use its compiled kernels in a given session.
%[text] Every experiment caches its result under *data*, one file per model, so the script can be run repeatedly and picks up where it stopped. Training uses mini-batches with one noise level per image; for the LSUN the per-block bookkeeping dominates the forward pass, so a batch of eight costs the same as a batch of one.
%[text] Source of this live script: *main_sbdm.m* (plain-text live code); the .mlx is generated from it.
%[text] 【References】
%[text] - S. Muramatsu, *Fundamentals and Developments of Multidimensional Signal and Image Processing*, Corona, 2026 (Example 10.2, Section 10.2.3).
%[text] - Y. Godage, E. Kobayashi and S. Muramatsu, "Locally-structured unitary network," APSIPA Trans. Signal Inf. Process., vol. 13, no. 1, e9, 2024.
%[text] - Y. Chen and T. Pock, "Trainable nonlinear reaction diffusion," IEEE TPAMI, vol. 39, no. 6, pp. 1256-1272, 2017.
%[text] - Y. Romano, M. Elad and P. Milanfar, "The little engine that could: regularization by denoising (RED)," SIAM J. Imaging Sci., vol. 10, no. 4, pp. 1804-1844, 2017.
%[text] - Y. Song and S. Ermon, "Generative modeling by estimating gradients of the data distribution," NeurIPS, 2019.
%[text] Copyright (c) 2026, Shogo MURAMATSU, All rights reserved.

%%
%[text] ## Configuration
clc, clear, close all
fcn_seed(0)

datfolder = fullfile('..','..','..','data');
resfolder = fullfile('..','..','..','results');

% If main_sbdm_nsolt.m has been run in this session, SaivDr's mexcodes folder
% is on the path in place of TanSacNet's: the two ship identically named
% kernels (fcn_orthmtxgen_*_mex) in non-package folders, and SaivDr's rejects
% the per-block angle layout of the LSUN. Restore ours.
warning('off','MATLAB:rmpath:DirNotFound')
rmpath(fullfile('/home/shogo/Workspace/GitHub/SaivDr','mexcodes'))
warning('on','MATLAB:rmpath:DirNotFound')
addpath(fullfile('..','..','mexcodes'))

szOrg    = [64 64];         % patch size used for training and for Experiment 1
stride   = [4 4];           % LSUN decimation factor
ovlp     = [3 3];           % LSUN overlapping factor (polyphase order plus one)
trainIdx = 1:8;             % kodim01..08 train the score estimator
evalIdx  = 9:24;            % kodim09..24 evaluate it, at full resolution
nCrops   = 256;             % training patches per training image
deblurIdx = [9 15];         % images used for the restoration experiment
intIdx   = [9 15 19 23];    % patches used for internal learning (Experiment 1b)

sigmaFix  = single(25/255);          % noise level of Experiment 1
sigmaMin  = single(2/255);           % noise-level range of the score estimator
sigmaMax  = single(25/255);
sigmaTest = single([2 5 10 15 25]/255);

nItersLsun = 1500;          % Adam iterations, LSUN
nItersConv = 15000;         % Adam iterations, convolutional baseline
lrLsun = 5e-3;
lrConv = 1e-3;
mbSize = 32;                % mini-batch size; a batch of 32 costs 1.2x a batch
                            % of 1, the per-block bookkeeping dominating
nItersInt     = 3000;       % Adam iterations, internal learning (Experiment 1b)
nItersIntLsun = 400;        % ditto, LSUN
nProbe = 4; nPower = 60;    % probes of the structural diagnostics

%[text] Training is checkpointed. Each run of this script spends at most `timeBudget` seconds on training and then stops with a notice; running it again resumes from the checkpoints under *data*, so a long schedule can be spread over several sessions.
timeBudget = 1500;          % seconds of training per run of this script
tRun = tic;
remaining = @() timeBudget - toc(tRun);
pending = false;

useGpu = canUseGPU;
fprintf("GPU: %d, image size: %s, stride: %s, overlapping factor: %s\n",...
    useGpu,mat2str(szOrg),mat2str(stride),mat2str(ovlp))

%%
%[text] ## Data
%[text] Kodak images at their native resolution, converted to grayscale. The score estimator is trained on random $64\times64$ patches of kodim01--08 and evaluated on the whole of kodim09--24, which are disjoint from it. Training on patches of full-resolution images rather than on downsampled images matters: downsampling removes the high-frequency content that the shrinkage has to learn to keep, and the models would then be evaluated out of their training distribution.
fcn_seed(1)
[~,trainSet] = fcn_loadkodak(datfolder,trainIdx,szOrg,nCrops,useGpu);
evalSet = fcn_loadkodak(datfolder,evalIdx,szOrg,0,useGpu);
fprintf("training patches: %d of size %s; evaluation images: %d at full resolution\n",...
    numel(trainSet),mat2str(szOrg),numel(evalSet))
montage(cellfun(@(z) gather(z),trainSet(1:16),'UniformOutput',false),'Size',[2 8])
title("Training patches (kodim01-08, full resolution)")

%%
%[text] ## LSUN analysis and synthesis pair
%[text] `fcn_lsunpair` returns the LSUN analysis operator $\mathbf{E}_{\bmtheta}$ and its adjoint $\mathbf{D}_{\bmtheta}=\mathbf{E}_{\bmtheta}^\top$ as two networks driven by one shared set of rotation angles, so the adjoint relation holds for every angle vector with no projection.
[anet,snet,info] = fcn_lsunpair(szOrg,stride,ovlp);
fprintf("blocks: %d, channels: %d, rotation layers: %d, angles/block: %s (total %d)\n",...
    info.nBlocks,info.nChs,info.nRotations,mat2str(info.nAngles'),sum(info.nAngles))

xchk = dlarray(single(evalSet{1}(1:szOrg(1),1:szOrg(2))),'SSCB');
[acchk,dcchk] = forward(anet,xchk);
fprintf("Parseval tightness error ||D*E*x-x||/||x|| = %.3e\n",...
    sqrt(double(gather(extractdata(sum((forward(snet,acchk,dcchk)-xchk).^2,'all'))))/...
         double(gather(extractdata(sum(xchk.^2,'all'))))))

%%
%[text] ## Experiment 1: the denoiser of Example 10.2, three dictionaries
%[text] The denoiser of \eqref{eq:tnrd} is trained at the single noise level $\sigma=25/255$ on patches of kodim01--08 and evaluated on patches of kodim09--24 that it has not seen. Example 10.2 trains on the image it denoises, which is instructive but degenerates into memorising that image once the schedule is long: the unconstrained convolutional dictionary then wins on capacity rather than on structure. The internal-learning regime is kept separate, below.
%[text] The diagnostics measure what the plug-and-play, RED and SBDM frameworks require of a denoiser: departure from Parseval tightness, asymmetry of the Jacobian, and its spectral radius. Tied weights already make the Jacobian symmetric in all cases, but only the LSUN parameterisation bounds the spectrum without a projection. For the LSUN the bound is attained: the DC channel is passed through, so $\rho(\bJ_f)=1$ exactly, and Proposition 3 says the rest of the spectrum lies above $1-\max_p\kappa_p$.
names1 = {'conv-TNRD (tied)','conv-TNRD (Parseval)','LSUN-U (uniform)'};
fcn_seed(3)
[~,evalPatches] = fcn_loadkodak(datfolder,evalIdx,szOrg,8,useGpu);
fprintf("held-out evaluation patches: %d\n",numel(evalPatches))
exp1 = struct('name',{},'psnr',{},'nPrm',{},'tightErr',{},'jacAsym',{},'jacRho',{});
for k = 1:3
    f1 = fullfile(datfolder,sprintf('sbdm_exp1_%d.mat',k));
    isCached = isfile(f1);
    if isCached
        S = load(f1); prm = S.prm; rec = S.rec;
        fprintf("[Experiment 1] loaded %s\n",names1{k});
    else
        fprintf("\n[Experiment 1] %s\n",names1{k});
    end
    tic
    switch k
        case {1,2}
            mode = "tied"; proj = 0;
            if k==2, mode = "parseval"; proj = 10; end
            if ~isCached
                if remaining() <= 0, pending = true; fprintf("  skipped: out of time budget\n"); continue, end
                ck = fullfile(datfolder,sprintf('sbdm_ckpt_exp1_%d.mat',k));
                prm = fcn_lsunmovegpu(fcn_convtnrdinitprm(1,32,5,mode),useGpu);
                [prm,~,isDone] = fcn_trainconvtnrd(prm,trainSet,sigmaFix,nItersConv,...
                    lrConv,proj,mbSize,true,ck,remaining());
                if ~isDone, pending = true; fprintf("  unfinished, re-run to continue\n"); continue, end
            end
            fden = @(v) fcn_convtnrddenoise(prm,v,single(1));
            zba = zeros(size(prm.Wa{1},4),1,'like',prm.Wa{1});
            zbs = zeros(1,1,'like',prm.Wa{1});
            fpair = @(v) dltranspconv(dlconv(v,prm.Wa{1},zba,'Padding','same'),...
                prm.Wa{1},zbs,'Cropping','same');
        case 3
            if ~isCached
                if remaining() <= 0, pending = true; fprintf("  skipped: out of time budget\n"); continue, end
                ck = fullfile(datfolder,sprintf('sbdm_ckpt_exp1_%d.mat',k));
                prm = fcn_lsunmovegpu(fcn_lsuninitprm(info,1,false,0.9,2.0),useGpu);
                [prm,~,isDone] = fcn_trainlsun(anet,snet,info,prm,trainSet,sigmaFix,...
                    nItersLsun,lrLsun,mbSize,true,ck,remaining());
                if ~isDone, pending = true; fprintf("  unfinished, re-run to continue\n"); continue, end
            end
            fden = @(v) fcn_lsundenoise(anet,snet,info,prm,v,sigmaFix);
            fpair = @(v) fcn_lsunpairapply(anet,snet,info,prm.th{1},v);
    end
    % the diagnostics are expensive and independent of the data, so they are
    % cached; the held-out PSNR is cheap and is recomputed every run
    if ~isCached
        d = fcn_denoiserdiag(fden,fpair,szOrg,nProbe,nPower);
        rec = struct('name',names1{k},'psnr',[],'nPrm',fcn_countprm(prm),...
            'tightErr',d.tightErr,'jacAsym',d.jacAsym,'jacRho',d.jacRho);
    end
    rec.psnr = fcn_psnrpatches(fden,evalPatches,sigmaFix,mbSize);
    fprintf("  held-out PSNR = %.2f dB, params = %d, tightErr = %.2e, jacAsym = %.2e, rho(J) = %.4f (%.0f s)\n",...
        rec.psnr,rec.nPrm,rec.tightErr,rec.jacAsym,rec.jacRho,toc)
    if ~isCached, save(f1,'rec','prm'), end
    exp1(k) = rec;
end
psnrNoisy1 = fcn_psnrpatches(@(v) v,evalPatches,sigmaFix,mbSize);
fprintf("\nnoisy patches: %.2f dB\n",psnrNoisy1)

%%
%[text] ## Experiment 1b: internal learning, where shift variance pays off
%[text] Each dictionary is trained on the very patch it denoises, as in Example 10.2. This is the regime in which the locally-structured LSUN, whose rotation angles vary from block to block, has something to offer that a shift-invariant dictionary cannot express.
%[text] Starting the locally-structured model from scratch does not test that: it pits $168\times$`nBlocks` angles against $168$ on the same budget and so measures the optimisation, not the parameterisation. The uniform parameterisation is exactly the subset of the locally-structured one on which every block carries the same angles, so the decisive comparison continues *one* trained uniform model for a further `nItersIntLsun` iterations in each of the two spaces:
%[text] - `U` uniform, `nItersIntLsun` iterations from scratch;
%[text] - `C` the same model continued for `nItersIntLsun` more, angles still shared;
%[text] - `L` the same model continued for `nItersIntLsun` more, angles freed per block.
%[text] `C` matches `L` in iterations and starts from the identical denoiser, so `L`-`C` isolates shift variance. A tied convolutional dictionary on the first patch is reported alongside as the memorisation reference: with a long schedule and unconstrained capacity it wins on capacity, not on structure, and carries none of the guarantees.
arms1b = {'U','C','L'};
psnrConv1b = NaN;
lbl1b = {'LSUN-U (uniform)','LSUN-U (continued)','LSUN-L (warm start)'};
res1b = nan(numel(intIdx),3);
nPrm1b = nan(1,3);
prmL1b = [];
for i = 1:numel(intIdx)
    fcn_seed(2)
    [~,ci] = fcn_loadkodak(datfolder,intIdx(i),szOrg,1,useGpu);
    imgi = ci{1};
    fcn_seed(11)
    vNoisy = single(imgi) + sigmaFix*randn(szOrg,'single','like',single(imgi));
    fprintf("\n[Experiment 1b] kodim%02d, noisy patch %.2f dB\n",intIdx(i),...
        fcn_psnr(dlarray(vNoisy,'SSCB'),imgi))

    % the first patch also carries the convolutional memorisation reference
    if i == 1
        fc = fullfile(datfolder,'sbdm_exp1b_1.mat');
        if isfile(fc)
            S = load(fc); prm = S.prm; rec = S.rec;
        elseif remaining() > 0
            ck = fullfile(datfolder,'sbdm_ckpt_exp1b_1.mat');
            prm = fcn_lsunmovegpu(fcn_convtnrdinitprm(1,32,5,"tied"),useGpu);
            [prm,~,isDone] = fcn_trainconvtnrd(prm,{imgi},sigmaFix,nItersInt,...
                lrConv,0,mbSize,false,ck,remaining());
            if isDone
                rec = struct('name','conv-TNRD (tied)','psnr',[],'nPrm',fcn_countprm(prm));
                rec.psnr = fcn_psnr(fcn_convtnrddenoise(prm,dlarray(vNoisy,'SSCB'),single(1)),imgi);
                save(fc,'rec','prm')
            else
                pending = true; rec = [];
            end
        else
            pending = true; rec = [];
        end
        if ~isempty(rec)
            psnrConv1b = rec.psnr;
            fprintf("  conv-TNRD (tied), memorisation reference: %.2f dB (%d prm)\n",...
                rec.psnr,rec.nPrm)
        end
    end

    % the kodim09 arms keep the original Experiment 1b file names
    if i == 1
        fs = {'sbdm_exp1b_2.mat','sbdm_exp1b_4.mat','sbdm_exp1b_5.mat'};
    else
        fs = arrayfun(@(a) sprintf('sbdm_exp1bm_%d_%s.mat',i,arms1b{a}),...
            1:3,'UniformOutput',false);
    end

    prmU = [];
    for a = 1:3
        f1 = fullfile(datfolder,fs{a});
        if isfile(f1)
            S = load(f1); prm = S.prm; rec = S.rec;
        else
            if a > 1 && isempty(prmU)
                pending = true;
                fprintf("  %s: skipped, arm U must finish first\n",arms1b{a}); continue
            end
            if remaining() <= 0
                pending = true;
                fprintf("  %s: skipped, out of time budget\n",arms1b{a}); continue
            end
            ck = fullfile(datfolder,sprintf('sbdm_ckpt_exp1bm_%d_%s.mat',i,arms1b{a}));
            switch arms1b{a}
                case 'U'
                    prm = fcn_lsunmovegpu(fcn_lsuninitprm(info,1,false,0.9,2.0),useGpu);
                case 'C'
                    prm = prmU;
                case 'L'
                    % the embedding leaves the denoiser unchanged, so this arm
                    % searches a strict superset of what arm C searches
                    prm = fcn_lsunexpandprm(prmU,info);
            end
            [prm,~,isDone] = fcn_trainlsun(anet,snet,info,prm,{imgi},sigmaFix,...
                nItersIntLsun,lrLsun,mbSize,false,ck,remaining());
            if ~isDone
                pending = true;
                fprintf("  %s: unfinished, re-run to continue\n",arms1b{a}); continue
            end
            rec = struct('name',arms1b{a},'psnr',[],'nPrm',fcn_countprm(prm));
            rec.psnr = fcn_psnr(fcn_lsundenoise(anet,snet,info,prm,...
                dlarray(vNoisy,'SSCB'),sigmaFix),imgi);
            save(f1,'rec','prm')
        end
        if strcmp(arms1b{a},'U'), prmU = prm; end
        if strcmp(arms1b{a},'L'), prmL1b = prm; end
        res1b(i,a) = rec.psnr;
        nPrm1b(a)  = rec.nPrm;
        fprintf("  %-2s %-20s %.2f dB (%d prm)\n",arms1b{a},lbl1b{a},rec.psnr,rec.nPrm)
    end
end

fprintf("\n[Experiment 1b] %d patches\n",numel(intIdx));
for a = 1:3
    fprintf("  %-20s mean %.2f dB (min %.2f, max %.2f), %d prm\n",lbl1b{a},...
        mean(res1b(:,a),'omitnan'),min(res1b(:,a)),max(res1b(:,a)),nPrm1b(a));
end
dLC = res1b(:,3)-res1b(:,2);
fprintf("  shift variance, L-C: mean %+.2f dB, per patch %s\n",...
    mean(dLC,'omitnan'),mat2str(round(dLC',2)));
fprintf("  extra iterations alone, C-U: mean %+.2f dB\n",...
    mean(res1b(:,2)-res1b(:,1),'omitnan'));
if ~isnan(psnrConv1b)
    fprintf("  conv-TNRD (tied) on kodim%02d, memorisation reference: %.2f dB\n",...
        intIdx(1),psnrConv1b);
end

%[text] The guarantees of Section 3 must survive the shift-variant angles, and they do: the locally-structured model is measured below with the same diagnostics as Experiment 1.
if ~isempty(prmL1b)
    fdenL = @(v) fcn_lsundenoise(anet,snet,info,prmL1b,v,sigmaFix);
    fpairL = @(x) fcn_lsunpairapply(anet,snet,info,prmL1b.th{1},x);
    dg1b = fcn_denoiserdiag(fdenL,fpairL,szOrg,nProbe,nPower);
    kap = max(double(gather(extractdata(sigmoid(prmL1b.a{1})))));
    fprintf("  locally-structured model: tightErr %.2e, jacAsym %.2e, rho(J) %.4f\n",...
        dg1b.tightErr,dg1b.jacAsym,dg1b.jacRho);
    fprintf("  max kappa %.4f, so Prop. 3 predicts the spectrum in [%.4f,1]\n",kap,1-kap);
end

%%
%[text] ## Experiment 2: noise-level conditional denoiser (score estimator)
%[text] By Tweedie's formula the score follows from the denoiser, $\nabla_\mathbf{u}\log p_\sigma(\mathbf{u})=(\mathbf{f}(\mathbf{u};\sigma)-\mathbf{u})/\sigma^2$, so a single denoiser covering a range of noise levels is all that is required. Each model is trained on patches of kodim01-08 with the noise level drawn from a log-uniform distribution over $[2/255,25/255]$, and evaluated on the whole of kodim09-24 by PSNR and SSIM.
%[text] The evaluation runs at full resolution although the training ran on $64\times64$ patches. Both models transfer: the convolutional one because it is shift invariant, the uniform LSUN because its angles are shared across blocks, so the same angle vector drives a lapped transform of any size. One forward pass costs about the same at $768\times512$ as at $64\times64$, the per-block bookkeeping being vectorised over blocks.
%[text] In the convolutional baseline the noise level multiplies the residual from outside the nonlinearity, so with one stage it merely rescales a fixed correction direction; stacking stages is what buys genuine noise-level dependence. In the LSUN realisation the noise level enters the argument of the nonlinearity, which is exact because the transform is unitary, so one stage already adapts.
% Cache slot k is fixed by history, so the models are listed in that order and
% only displayed by stage count later. Every dictionary is trained at S = 1, 2
% and 3, so Table 2 compares matched stage counts.
names2 = {'conv-TNRD cond. (S=1)','conv-TNRD cond. (S=3)',...
    'LSUN-U cond. (S=1)','LSUN-U cond. (S=2)',...
    'conv-TNRD cond. (S=2)','LSUN-U cond. (S=3)'};
nStages2 = [1 3 1 2 2 3];
isConv2  = [true true false false true false];
ord2 = [1 3 5 2 4 6];   % display order: conv/LSUN paired by S
exp2 = struct('name',{},'psnr',{},'ssim',{},'nPrm',{});
prm2 = cell(1,6);
for k = 1:6
    f2 = fullfile(datfolder,sprintf('sbdm_exp2_%d.mat',k));
    isCached = isfile(f2);
    if isCached
        S = load(f2); prm = S.prm;
        fprintf("[Experiment 2] loaded %s\n",names2{k});
    else
        fprintf("\n[Experiment 2] %s\n",names2{k});
    end
    tic
    nSt = nStages2(k);
    if isConv2(k)
        if ~isCached
            if remaining() <= 0, pending = true; fprintf("  skipped: out of time budget\n"); continue, end
            ck = fullfile(datfolder,sprintf('sbdm_ckpt_exp2_%d.mat',k));
            prm = fcn_lsunmovegpu(fcn_convtnrdinitprm(nSt,32,5,"tied"),useGpu);
            % the convolutional budget scales with the depth, the LSUN one does
            % not, which favours the baseline as S grows
            [prm,~,isDone] = fcn_trainconvtnrd(prm,trainSet,[sigmaMin sigmaMax],...
                nSt*nItersConv,lrConv,0,mbSize,true,ck,remaining());
            if ~isDone, pending = true; fprintf("  unfinished, re-run to continue\n"); continue, end
        end
    else
        if ~isCached
            if remaining() <= 0, pending = true; fprintf("  skipped: out of time budget\n"); continue, end
            ck = fullfile(datfolder,sprintf('sbdm_ckpt_exp2_%d.mat',k));
            prm = fcn_lsunmovegpu(fcn_lsuninitprm(info,nSt,false,0.9,2.0),useGpu);
            [prm,~,isDone] = fcn_trainlsun(anet,snet,info,prm,trainSet,...
                [sigmaMin sigmaMax],nItersLsun,lrLsun,mbSize,true,ck,remaining());
            if ~isDone, pending = true; fprintf("  unfinished, re-run to continue\n"); continue, end
        end
    end
    if isConv2(k)
        fdenSz = @(v,sg,sz) extractdata(fcn_convtnrddenoise(prm,dlarray(v,'SSCB'),sg));
    else
        fdenSz = @(v,sg,sz) fcn_lsundenoisesz(prm,v,sg,sz,stride,ovlp);
    end
    [ps,ss] = fcn_evalfullres(fdenSz,evalSet,sigmaTest);
    rec = struct('name',names2{k},'psnr',mean(ps,2),'ssim',mean(ss,2),...
        'nPrm',fcn_countprm(prm));
    fprintf("  PSNR = %s dB, SSIM = %s, params = %d (%.0f s)\n",...
        mat2str(round(mean(ps,2)',2)),mat2str(round(mean(ss,2)',4)),rec.nPrm,toc)
    if ~isCached, save(f2,'rec','prm'), end
    exp2(k) = rec; prm2{k} = prm;
end
[psn,ssn] = fcn_evalfullres(@(v,sg,sz) v,evalSet,sigmaTest);
psNoisy2 = mean(psn,2); ssNoisy2 = mean(ssn,2);
fprintf("noisy input: PSNR = %s dB, SSIM = %s\n",...
    mat2str(round(psNoisy2',2)),mat2str(round(ssNoisy2',4)))

%%
%[text] ## Learned normalised thresholds
%[text] For large coefficients the LSUN shrinkage tends to $c-\sigma(\kappa_p/g_p)\,\mathrm{sgn}(c)$, i.e. a soft threshold whose level is proportional to the noise level, as in wavelet shrinkage. The learned ratio $\tau_p=\kappa_p/g_p$ is that threshold in units of $\sigma$, and $\kappa_p$ bounds the contraction of the Jacobian: its spectrum lies in $[1-\max_p\kappa_p,1]$.
if ~pending
% the shrinkage of the first stage of the deepest LSUN, the model Table 3 uses
kappa = double(gather(extractdata(sigmoid(prm2{6}.a{1}))));
tau   = kappa./double(gather(extractdata(exp(prm2{6}.b{1}))));
bar([tau(:) kappa(:)])
legend(["\tau_p = \kappa_p/g_p","\kappa_p"],'Location','best')
xlabel("AC channel index p"), ylabel("value"), grid on
title("Learned normalised threshold and contraction")
fprintf("max kappa_p = %.4f  =>  spectrum of J in [%.4f, 1]\n",max(kappa),1-max(kappa))
fprintf("normalised threshold tau_p in [%.3f, %.3f], median %.3f\n",min(tau),max(tau),median(tau))
end

%%
%[text] ## Experiment 3: score-based RED restoration and Langevin sampling
%[text] Gaussian blur followed by AWGN on full-resolution images, restored by the score-based RED iteration of Section 10.2.3,
%[text] $$\mathbf{x}^{(t+1)}\gets\mathbf{x}^{(t)}-\eta\left(\mathbf{H}^\top(\mathbf{H}\mathbf{x}^{(t)}-\mathbf{v})+\lambda(\mathbf{x}^{(t)}-\mathbf{f}(\mathbf{x}^{(t)};\sigma^{(t)}))\right)+\varepsilon\,\sigma_\mathrm{w}\sqrt{2\eta}\,\bmzeta^{(t)},$$
%[text] whose last term is the Euler--Maruyama discretisation of the Langevin dynamics and is switched off, $\varepsilon=0$, for the point estimate.
if ~pending
blurKernel = fspecial('gaussian',9,2.0);
Hf  = @(z) imfilter(z,blurKernel,'circular');
Htf = @(z) imfilter(z,rot90(blurKernel,2),'circular');
sigmaW = 5/255;
mu3 = 0.5; nIters3 = 300;
lambdaGrid = [0.5 1.0 2.0];
schedule = exp(linspace(log(sigmaMax),log(sigmaMin),nIters3));
modeNames = {'no regularisation','fixed sigma_{min}','fixed sigma_{max}',...
    'annealed sigma^{(t)}','Langevin (annealed)'};
methodNames = {'conv-TNRD cond. (S=3)','LSUN-U cond. (S=1)',...
    'LSUN-U cond. (S=2)','LSUN-U cond. (S=3)'};
methodPrm = [2 3 4 6];
nMeth = numel(methodNames);
iLgv = 4;   % the model used for the posterior sampling below

% observations, one per restoration image
nDeb = numel(deblurIdx);
Xd = cell(1,nDeb); Vd = cell(1,nDeb); psnrObsd = zeros(1,nDeb);
for id = 1:nDeb
    Xd{id} = double(gather(evalSet{deblurIdx(id)-evalIdx(1)+1}));
    fcn_seed(3+id)
    Vd{id} = Hf(Xd{id}) + sigmaW*randn(size(Xd{id}));
    psnrObsd(id) = psnr(min(max(Vd{id},0),1),Xd{id});
    fprintf("kodim%02d %s : observation PSNR = %.2f dB\n",...
        deblurIdx(id),mat2str(size(Xd{id})),psnrObsd(id))
end

fden3all = cell(1,nMeth);
for m = 1:nMeth
    if m == 1
        fden3all{m} = @(z,sg,sz) double(gather(extractdata(fcn_convtnrddenoise(...
            prm2{methodPrm(m)},dlarray(fcn_dev(single(z),useGpu),'SSCB'),single(sg)))));
    else
        fden3all{m} = @(z,sg,sz) double(gather(fcn_lsundenoisesz(prm2{methodPrm(m)},...
            fcn_dev(single(z),useGpu),single(sg),sz,stride,ovlp)));
    end
end

%[text] The regularisation parameter is selected per method on the first image with the annealed schedule, so that the comparison is not an artefact of a shared value.
bestLam = zeros(1,nMeth);
sweep = struct('method',{},'lambda',{},'psnr',{});
q = 0;
for m = 1:nMeth
    for il = 1:numel(lambdaGrid)
        q = q+1;
        fs = fullfile(datfolder,sprintf('sbdm_exp3sw_%d_%d.mat',m,il));
        if isfile(fs), S = load(fs); sweep(q) = S.rec; continue, end
        if remaining() <= 0, pending = true; break, end
        rec = struct('method',methodNames{m},'lambda',lambdaGrid(il),...
            'psnr',fcn_redrun(@(z,sg) fden3all{m}(z,sg,size(Xd{1})),Vd{1},Xd{1},...
            Hf,Htf,schedule,mu3,lambdaGrid(il),nIters3,4,...
            sigmaMin,sigmaMax,sigmaW,size(Xd{1})));
        fprintf("  [sweep] %-22s lambda = %.1f  PSNR = %.2f dB\n",...
            rec.method,rec.lambda,rec.psnr);
        save(fs,'rec'), sweep(q) = rec;
    end
    ps = [sweep(arrayfun(@(r) strcmp(r.method,methodNames{m}),sweep)).psnr];
    if numel(ps) < numel(lambdaGrid)
        pending = true;
        fprintf("  %-22s sweep incomplete, re-run to continue\n",methodNames{m});
        break
    end
    [~,ib] = max(ps); bestLam(m) = lambdaGrid(ib);
    fprintf("  %-22s best lambda = %.1f\n",methodNames{m},bestLam(m));
end

exp3 = struct('image',{},'method',{},'mode',{},'lambda',{},'psnr',{},'ssim',{},...
    'curve',{},'est',{});
e = 0;
if ~pending
for id = 1:nDeb
    for m = 1:nMeth
        for mode = 1:5
            e = e+1;
            f3 = fullfile(datfolder,sprintf('sbdm_exp3_%d_%d_%d.mat',id,m,mode));
            if isfile(f3), S = load(f3); exp3(e) = S.rec; continue, end
            if remaining() <= 0, pending = true; break, end
            [p3,curve,z] = fcn_redrun(@(zz,sg) fden3all{m}(zz,sg,size(Xd{id})),...
                Vd{id},Xd{id},Hf,Htf,schedule,mu3,bestLam(m),nIters3,mode,...
                sigmaMin,sigmaMax,sigmaW,size(Xd{id}));
            rec = struct('image',deblurIdx(id),'method',methodNames{m},...
                'mode',modeNames{mode},'lambda',bestLam(m),'psnr',p3,...
                'ssim',ssim(z,Xd{id}),'curve',curve,'est',single(z));
            fprintf("  kodim%02d %-22s %-22s PSNR = %.2f dB, SSIM = %.4f\n",...
                deblurIdx(id),methodNames{m},modeNames{mode},p3,rec.ssim);
            save(f3,'rec'), exp3(e) = rec;
        end
    end
end
end
end


%[text] ## Posterior sampling and uncertainty
%[text] The Langevin iteration is a sampler, not a point estimator, so a single sample has a lower PSNR than the maximum-a-posteriori estimate by construction. Its purpose is to quantify the uncertainty of the estimate: the sample mean approaches the posterior mean and the pixelwise standard deviation localises where the reconstruction is not determined by the data.
if ~pending
nSamples = 8;   % posterior samples
lgvFile = fullfile(datfolder,'sbdm_langevin.mat');
if isfile(lgvFile)
    load(lgvFile,'smpMean','smpStd','psnrMean','psnrSample')
    fprintf("loaded %s\n",lgvFile)
else
    szL = size(Xd{1});
    acc = zeros(szL); acc2 = zeros(szL); psnrSample = zeros(nSamples,1);
    for r = 1:nSamples
        [psnrSample(r),~,zr] = fcn_redrun(@(zz,sg) fden3all{iLgv}(zz,sg,szL),...
            Vd{1},Xd{1},Hf,Htf,schedule,mu3,bestLam(iLgv),nIters3,5,...
            sigmaMin,sigmaMax,sigmaW,szL,1000+r);
        acc = acc + zr; acc2 = acc2 + zr.^2;
    end
    smpMean = acc/nSamples;
    smpStd  = sqrt(max(acc2/nSamples - smpMean.^2,0));
    psnrMean = psnr(min(max(smpMean,0),1),Xd{1});
    save(lgvFile,'smpMean','smpStd','psnrMean','psnrSample')
end
fprintf("Langevin: single sample %.2f +/- %.2f dB, sample mean of %d %.2f dB\n",...
    mean(psnrSample),std(psnrSample),nSamples,psnrMean)
fprintf("posterior std: mean %.4f, max %.4f\n",mean(smpStd(:)),max(smpStd(:)))
end

%%
%[text] ## Results
if pending
    fprintf("\n*** training is not finished: run this script again to continue ***\n")
else
%[text] ### Experiment 1: structure and held-out denoising
disp(struct2table(exp1))
fprintf("noisy patches: %.2f dB\n",psnrNoisy1)
%[text] ### Experiment 1b: internal learning, one model per patch
disp(array2table(res1b,'VariableNames',arms1b,...
    'RowNames',compose("kodim%02d",intIdx')))
fprintf("shift variance, L-C: mean %+.2f dB over %d patches\n",...
    mean(res1b(:,3)-res1b(:,2),'omitnan'),numel(intIdx))
%[text] ### Experiment 2: noise-level conditional denoising on full-resolution kodim09-24
T2 = table(double(sigmaTest')*255,psNoisy2,'VariableNames',{'sigma_x255','noisy'});
T2s = table(double(sigmaTest')*255,ssNoisy2,'VariableNames',{'sigma_x255','noisy'});
for k = ord2
    if k > numel(exp2) || isempty(exp2(k).name), continue, end
    nm = matlab.lang.makeValidName(exp2(k).name);
    T2.(nm) = exp2(k).psnr; T2s.(nm) = exp2(k).ssim;
end
disp("PSNR [dB], mean over kodim09-24 at full resolution"), disp(T2)
disp("SSIM, mean over kodim09-24 at full resolution"), disp(T2s)
%[text] ### Experiment 3: score-based RED restoration
disp(struct2table(rmfield(exp3,{'curve','est'})))
disp(struct2table(sweep))

%[text] ### Figures for the paper
iAnn = find(strcmp({exp3.mode},'annealed sigma^{(t)}') & [exp3.image]==deblurIdx(1));

figThr = figure('Units','centimeters','Position',[0 0 8.6 5.2]);
bar([tau(:) kappa(:)])
legend({'$\tau_p=\kappa_p/g_p$','$\kappa_p$'},'Interpreter','latex',...
    'Location','northwest','FontSize',8)
xlabel('AC channel index p','FontSize',8), ylabel('value','FontSize',8)
set(gca,'FontSize',7), grid on, ylim([0 1.05*max([tau(:);kappa(:)])*1.35])

figCurve = figure('Units','centimeters','Position',[0 0 8.6 6.2]);
% distinguishable in black and white: the paper may be printed in grey
styles = {'--',':','-.','-'};
widths = [1.0 1.4 1.0 1.1];
hold on
lg = strings(0);
for j = 1:numel(iAnn)
    plot(exp3(iAnn(j)).curve,styles{j},'Color','k','LineWidth',widths(j))
    lg(end+1) = string(exp3(iAnn(j)).method); %#ok<SAGROW>
end
yline(psnrObsd(1),'-','Color',[.6 .6 .6],'LineWidth',0.6)
hold off, grid on
xlabel('iteration t','FontSize',8), ylabel('PSNR [dB]','FontSize',8)
legend([lg,"observation"],'Location','southeast','FontSize',7,'Box','on')
set(gca,'FontSize',7)

figure
montage([{Xd{1},min(max(Vd{1},0),1)},{exp3(iAnn).est}],'Size',[2 2])
title("original / observation / annealed RED per method")

if isfolder(resfolder)
    exportgraphics(figThr,fullfile(resfolder,'sbdm_fig_thresholds.pdf'),'ContentType','vector')
    exportgraphics(figCurve,fullfile(resfolder,'sbdm_fig_redcurve.pdf'),'ContentType','vector')
    imwrite(min(max(Xd{1},0),1),fullfile(resfolder,'sbdm_img_orig.png'))
    imwrite(min(max(Vd{1},0),1),fullfile(resfolder,'sbdm_img_obs.png'))
    imwrite(exp3(iAnn(1)).est,fullfile(resfolder,'sbdm_img_conv.png'))
    imwrite(exp3(iAnn(end)).est,fullfile(resfolder,'sbdm_img_lsun.png'))
    imwrite(min(max(smpMean,0),1),fullfile(resfolder,'sbdm_img_postmean.png'))
    imwrite(ind2rgb(gray2ind(mat2gray(smpStd),256),parula(256)),...
        fullfile(resfolder,'sbdm_img_poststd.png'))
    save(fullfile(resfolder,'sbdm_results.mat'),'exp1','exp2','exp3','sweep',...
        'bestLam','res1b','nPrm1b','intIdx','psnrNoisy1','psNoisy2','ssNoisy2','psnrObsd','sigmaTest',...
        'kappa','tau','Xd','Vd','smpMean','smpStd','psnrMean','psnrSample')
    fprintf("figures and results written to %s\n",resfolder)
end
end

%%
%[text] ## Definitions of local functions

function y = fcn_lsundenoisesz(prm,v,sigma,sz,stride,ovlp)
% LSUN denoiser at an arbitrary image size: the uniform angle set is shared
% across blocks, so one trained parameter vector drives a transform of any
% size once the network of that size has been built
[anet,snet,info] = fcn_lsunpaircached(sz,stride,ovlp);
y = extractdata(fcn_lsundenoise(anet,snet,info,prm,dlarray(v,'SSCB'),sigma));
end

function [anet,snet,info] = fcn_lsunpaircached(sz,stride,ovlp)
% build one LSUN pair per distinct image size and keep it
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

function y = fcn_dev(x,useGpu)
% move an array to the device the parameters live on
y = x; if useGpu, y = gpuArray(x); end
end

function fcn_seed(s)
% seed both the host and the device generator: randn(...,'like',gpuArray)
% draws from the device stream, which rng alone does not reset, so without
% this the noise realisations differ from run to run
rng(s)
if canUseGPU, gpurng(s); end
end

function y = fcn_lsunpairapply(anet,snet,info,th,v)
% D*E*v with the tied angle set TH, used to measure Parseval tightness
[anet,snet] = fcn_lsunsetangles(anet,snet,info,th);
[ac,dc] = forward(anet,v);
y = forward(snet,ac,dc);
end

function prm = fcn_lsunmovegpu(prm,useGpu)
% move every learnable of a parameter struct to the GPU
if useGpu
    prm = dlupdate(@gpuArray,prm);
end
end

function p = fcn_psnrpatches(fden,patches,sigma,batchSize)
% mean PSNR of a denoiser over a set of equally sized patches, in batches
sz = size(patches{1});
n = numel(patches);
acc = 0;
for i0 = 1:batchSize:n
    idx = i0:min(i0+batchSize-1,n);
    xb = zeros([sz 1 numel(idx)],'like',patches{1});
    for j = 1:numel(idx), xb(:,:,1,j) = patches{idx(j)}; end
    fcn_seed(500+i0)
    vb = xb + sigma*randn(size(xb),'like',xb);
    yb = fden(dlarray(vb,'SSCB'));
    if isa(yb,'dlarray'), yb = extractdata(yb); end
    yb = min(max(double(gather(yb)),0),1);
    xb = double(gather(xb));
    for j = 1:numel(idx), acc = acc + psnr(yb(:,:,1,j),xb(:,:,1,j)); end
end
p = acc/n;
end

function p = fcn_psnr(xh,ref)
% PSNR of a dlarray estimate against a reference image, both possibly on GPU
a = min(max(double(gather(extractdata(xh))),0),1);
p = psnr(a,double(gather(ref)));
end

function [p,curve,z] = fcn_redrun(fden,V,X,Hf,Htf,schedule,mu,lambda,nIters,mode,sigmaMin,sigmaMax,sigmaW,sz,seed)
% score-based RED iteration of Section 10.2.3, optionally with the
% Euler--Maruyama noise injection of the Langevin dynamics (mode 5). SEED
% selects the realisation of the injected noise; it is what distinguishes
% one posterior sample from another.
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

function n = fcn_countprm(prm)
% number of learnable scalars in a parameter struct of (nested) cells
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

%[appendix]{"version":"1.0"}
