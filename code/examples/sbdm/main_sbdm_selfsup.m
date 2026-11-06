%[text] # Self-supervised shift-variant LSUN (SURE)
%[text] Answer to the ver3 comment: with angles shared by all blocks the dictionary is the NSCLT, not the LSUN. Here the per-block angles of a real LSUN are learned on the noisy test image itself, by SURE, so no clean image of the test scene is used.
%[text] Arms, all starting from the same uniform (NSCLT) model U0 trained on kodim01-08 (clean, external data):
%[text] - `U0` the uniform model as is (no adaptation);
%[text] - `C` U0 adapted by SURE on the noisy image, angles still shared (NSCLT);
%[text] - `L` U0 expanded to per-block angles (identical denoiser at the start), adapted by SURE (LSUN).
%[text] `C` and `L` use the same iterations from the same start, so `L`-`C` isolates the shift variance.
%[text] The image crops and noise realisations follow Experiment 1b of main_sbdm.m (same seeds), so the oracle results there (trained with the clean patch) are a direct reference.
%[text] The clean image is used only to report PSNR and the true MSE at the end.
%[text] Run setpath in code/ first. Results are cached under data/sbdm_ss_*.mat.

%%
clc, clear, close all
fcn_seed(0)
datfolder = fullfile('..','..','..','data');
warning('off','MATLAB:rmpath:DirNotFound')
rmpath(fullfile('/home/shogo/Workspace/GitHub/SaivDr','mexcodes'))
warning('on','MATLAB:rmpath:DirNotFound')
addpath(fullfile('..','..','mexcodes'))

szOrg    = [64 64];         % same patches as Experiment 1b
stride   = [4 4];
ovlp     = [3 3];
trainIdx = 1:8;  nCrops = 256;
intIdx   = [9 15 19 23];
sigmaFix = single(25/255);
nItersLsun = 1500; lrLsun = 5e-3; mbSize = 32;   % pre-training of U0, as main_sbdm.m
nItersSs   = 400;                                 % SURE adaptation, as nItersIntLsun
nProbe = 4; nPower = 60;
useGpu = canUseGPU;
fprintf("GPU: %d\n",useGpu)

[anet,snet,info] = fcn_lsunpair(szOrg,stride,ovlp);

%%
%[text] ## U0: uniform model trained on kodim01-08
%[text] Reuses Experiment 1 (LSUN-U) of main_sbdm.m when its cache exists.
f0   = fullfile(datfolder,'sbdm_exp1_3.mat');
fpre = fullfile(datfolder,'sbdm_ss_pre.mat');
if isfile(f0)
    S = load(f0); prm0 = S.prm; fprintf("U0 loaded from %s\n",f0)
elseif isfile(fpre)
    S = load(fpre); prm0 = S.prm0; fprintf("U0 loaded from %s\n",fpre)
else
    fprintf("training U0 on kodim01-08 (%d iterations)\n",nItersLsun)
    fcn_seed(1)
    [~,trainSet] = fcn_loadkodak(datfolder,trainIdx,szOrg,nCrops,useGpu);
    prm0 = fcn_lsunmovegpu(fcn_lsuninitprm(info,1,false,0.9,2.0),useGpu);
    ck = fullfile(datfolder,'sbdm_ckpt_ss_pre.mat');
    prm0 = fcn_trainlsun(anet,snet,info,prm0,trainSet,sigmaFix,...
        nItersLsun,lrLsun,mbSize,true,ck,inf);
    save(fpre,'prm0')
end
prm0 = fcn_lsunmovegpu(prm0,useGpu);

%%
%[text] ## SURE adaptation on each noisy image
arms = {'U0','C','L'};
res  = nan(numel(intIdx),3);      % PSNR [dB]
sre  = nan(numel(intIdx),3);      % final SURE
mse  = nan(numel(intIdx),3);      % true MSE (evaluation only)
nPrm = nan(1,3);
psnrNoisy = nan(numel(intIdx),1);
prmL1 = [];
for i = 1:numel(intIdx)
    fcn_seed(2)
    [~,ci] = fcn_loadkodak(datfolder,intIdx(i),szOrg,1,useGpu);
    x = ci{1};
    fcn_seed(11)
    v = single(x) + sigmaFix*randn(szOrg,'single','like',single(x));
    vd = dlarray(v,'SSCB');
    psnrNoisy(i) = fcn_psnr(vd,x);
    fprintf("\nkodim%02d, noisy patch %.2f dB\n",intIdx(i),psnrNoisy(i))
    for a = 1:3
        f1 = fullfile(datfolder,sprintf('sbdm_ss_%02d_%s.mat',intIdx(i),arms{a}));
        if isfile(f1)
            S = load(f1); rec = S.rec; prm = S.prm;
        else
            hist = [];
            switch arms{a}
                case 'U0', prm = prm0;
                case 'C',  prm = prm0;
                case 'L',  prm = fcn_lsunexpandprm(prm0,info);
            end
            if a > 1
                tic
                fcn_seed(100+i)
                ck = fullfile(datfolder,sprintf('sbdm_ckpt_ss_%02d_%s.mat',intIdx(i),arms{a}));
                % one observation at its own noise level: the batch would only
                % repeat it, and the divergence is exact for one stage, so B = 1
                [prm,hist] = fcn_trainlsunsure(anet,snet,info,prm,{v},sigmaFix,...
                    sigmaFix,nItersSs,lrLsun,1,false,ck,inf);
                fprintf("  %-2s adapted in %.0f s\n",arms{a},toc)
            end
            [sr,fu] = fcn_lsunsure(anet,snet,info,prm,vd,sigmaFix);
            rec = struct('arm',arms{a},'psnr',fcn_psnr(fu,x),...
                'sure',double(gather(extractdata(sr))),...
                'mse',mean((double(gather(extractdata(fu)))-double(gather(x))).^2,'all'),...
                'nPrm',fcn_countprm(prm),'hist',hist);
            save(f1,'rec','prm')
        end
        res(i,a) = rec.psnr; sre(i,a) = rec.sure; mse(i,a) = rec.mse; nPrm(a) = rec.nPrm;
        if i == 1 && strcmp(arms{a},'L'), prmL1 = prm; end
        fprintf("  %-2s PSNR %.2f dB  SURE %.4e  true MSE %.4e  (%d prm)\n",...
            arms{a},rec.psnr,rec.sure,rec.mse,rec.nPrm)
    end
end

%%
%[text] ## Guarantees for the per-block model
%[text] Proposition: symmetric Jacobian with spectrum in [1-max kappa, 1] for any angles.
if ~isempty(prmL1)
    fdenL  = @(z) fcn_lsundenoise(anet,snet,info,prmL1,z,sigmaFix);
    fpairL = @(z) fcn_lsunpairapply(anet,snet,info,prmL1.th{1},z);
    dg = fcn_denoiserdiag(fdenL,fpairL,szOrg,nProbe,nPower);
    kap = max(double(gather(extractdata(sigmoid(prmL1.a{1})))));
    fprintf("\nper-block model (kodim%02d): tightErr %.2e, jacAsym %.2e, rho(J) %.4f, predicted [%.4f,1]\n",...
        intIdx(1),dg.tightErr,dg.jacAsym,dg.jacRho,1-kap)
end

%%
%[text] ## Summary
T = array2table([psnrNoisy res],'VariableNames',[{'noisy'},arms],...
    'RowNames',compose("kodim%02d",intIdx'));
disp("PSNR [dB], 64x64 patches, sigma = 25/255"), disp(T)
fprintf("mean: U0 %.2f, C %.2f, L %.2f dB\n",mean(res,1))
fprintf("shift variance, L - C: mean %+.2f dB, per image %s\n",...
    mean(res(:,3)-res(:,2)),mat2str(round((res(:,3)-res(:,2))',2)))
fprintf("SURE adaptation alone, C - U0: mean %+.2f dB\n",mean(res(:,2)-res(:,1)))
fprintf("SURE vs true MSE (overfitting check), relative gap per arm: %s\n",...
    mat2str(round(mean((sre-mse)./mse,1),3)))
fprintf("parameters: U0 %d, C %d, L %d\n",nPrm)
save(fullfile(datfolder,'sbdm_ss_results.mat'),'res','sre','mse','nPrm','psnrNoisy','intIdx')

%%
%[text] ## Local functions (copied from main_sbdm.m)
function fcn_seed(s)
rng(s)
if canUseGPU, gpurng(s); end
end

function y = fcn_lsunpairapply(anet,snet,info,th,v)
[anet,snet] = fcn_lsunsetangles(anet,snet,info,th);
[ac,dc] = forward(anet,v);
y = forward(snet,ac,dc);
end

function prm = fcn_lsunmovegpu(prm,useGpu)
if useGpu, prm = dlupdate(@gpuArray,prm); end
end

function p = fcn_psnr(xh,ref)
a = min(max(double(gather(extractdata(xh))),0),1);
p = psnr(a,double(gather(ref)));
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
