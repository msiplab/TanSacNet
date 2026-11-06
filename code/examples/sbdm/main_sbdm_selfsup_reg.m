%[text] # Self-supervised LSUN with fewer free angles (plan B against overfitting)
%[text] Run only after main_sbdm_selfsup.m: it reuses its U0 and its C / L results.
%[text] A per-block LSUN on a 64x64 patch has about 10 angles per noisy pixel, so SURE may fit the noise.
%[text] Two remedies, both starting from the same U0 and using the same SURE schedule as arm L:
%[text] - groups: g x g neighbouring blocks share one angle set (g = 1 is arm L, g = 16 would be arm C);
%[text] - penalty: lambda * mean((theta - theta_U0)^2) keeps the local angles near the uniform ones.
%[text] `g1` (g = 1, lambda = 0) must reproduce arm L: a check of the new trainer.
%[text] Selection protocol (as for beta in the paper): choose the variant on kodim09, report kodim15/19/23.
%[text] The clean image is used only for PSNR and the true MSE.

%%
clc, clear, close all
datfolder = fullfile('..','..','..','data');
warning('off','MATLAB:rmpath:DirNotFound')
rmpath(fullfile('/home/shogo/Workspace/GitHub/SaivDr','mexcodes'))
warning('on','MATLAB:rmpath:DirNotFound')
addpath(fullfile('..','..','mexcodes'))

szOrg = [64 64]; stride = [4 4]; ovlp = [3 3];
intIdx = [9 15 19 23];
sigmaFix = single(25/255);
nItersSs = 400; lrLsun = 5e-3;
useGpu = canUseGPU;

vars = struct('name',{'g1','g2','g4','g8','g1-l1e-3','g1-l1e-2','g1-l1e-1'},...
              'gs',  { 1,   2,   4,   8,   1,         1,         1        },...
              'lam', { 0,   0,   0,   0,   1e-3,      1e-2,      1e-1     });
nV = numel(vars);

[anet,snet,info] = fcn_lsunpair(szOrg,stride,ovlp);

% the same U0 as main_sbdm_selfsup.m
f0   = fullfile(datfolder,'sbdm_exp1_3.mat');
fpre = fullfile(datfolder,'sbdm_ss_pre.mat');
if isfile(f0), S = load(f0); prm0 = S.prm;
elseif isfile(fpre), S = load(fpre); prm0 = S.prm0;
else, error("Run main_sbdm_selfsup.m first: U0 is missing.")
end
prm0 = fcn_lsunmovegpu(prm0,useGpu);

%%
psn = nan(numel(intIdx),nV); sre = psn; mse = psn; nFree = nan(1,nV);
ref = nan(numel(intIdx),2);          % PSNR of arms C and L from main_sbdm_selfsup.m
for i = 1:numel(intIdx)
    fcn_seed(2)
    [~,ci] = fcn_loadkodak(datfolder,intIdx(i),szOrg,1,useGpu);
    x = ci{1};
    fcn_seed(11)
    v = single(x) + sigmaFix*randn(szOrg,'single','like',single(x));
    vd = dlarray(v,'SSCB');
    for a = 1:2
        armRef = {'C','L'};
        fr = fullfile(datfolder,sprintf('sbdm_ss_%02d_%s.mat',intIdx(i),armRef{a}));
        if isfile(fr), S = load(fr); ref(i,a) = S.rec.psnr; end
    end
    fprintf("\nkodim%02d: C %.2f dB, L %.2f dB (from main_sbdm_selfsup.m)\n",intIdx(i),ref(i,1),ref(i,2))
    for j = 1:nV
        f1 = fullfile(datfolder,sprintf('sbdm_ssr_%02d_%s.mat',intIdx(i),vars(j).name));
        if isfile(f1)
            S = load(f1); rec = S.rec;
        else
            tic
            fcn_seed(100+i)
            ck = fullfile(datfolder,sprintf('sbdm_ckpt_ssr_%02d_%s.mat',intIdx(i),vars(j).name));
            [prm,hist] = fcn_trainlsunsurex(anet,snet,info,prm0,{v},sigmaFix,sigmaFix,...
                nItersSs,lrLsun,1,false,ck,inf,...
                'GroupSize',vars(j).gs*[1 1],'Lambda',vars(j).lam);
            [sr,fu] = fcn_lsunsure(anet,snet,info,prm,vd,sigmaFix);
            [~,nG] = fcn_lsungroupidx(info,vars(j).gs*[1 1]);
            rec = struct('name',vars(j).name,'psnr',fcn_psnr(fu,x),...
                'sure',double(gather(extractdata(sr))),...
                'mse',mean((double(gather(extractdata(fu)))-double(gather(x))).^2,'all'),...
                'nFree',sum(info.nAngles)*nG + 2*(info.nChs-1),'hist',hist,'sec',toc);
            save(f1,'rec','prm')
        end
        psn(i,j) = rec.psnr; sre(i,j) = rec.sure; mse(i,j) = rec.mse; nFree(j) = rec.nFree;
        fprintf("  %-9s PSNR %.2f dB  SURE %.4e  true MSE %.4e  gap %+.3f  (%d free prm)\n",...
            vars(j).name,rec.psnr,rec.sure,rec.mse,(rec.sure-rec.mse)/rec.mse,rec.nFree)
    end
end

%%
%[text] ## Summary
names = matlab.lang.makeValidName({vars.name});
T = array2table([ref psn],'VariableNames',[{'C','L'},names],...
    'RowNames',compose("kodim%02d",intIdx'));
disp("PSNR [dB], 64x64 patches, sigma = 25/255"), disp(T)
fprintf("check: g1 - L = %s dB (should be about 0)\n",mat2str(round((psn(:,1)-ref(:,2))',3)))
[~,jb] = max(psn(1,:));
fprintf("selected on kodim%02d: %s\n",intIdx(1),vars(jb).name)
fprintf("  on kodim15/19/23: selected - C = %s dB, mean %+.2f\n",...
    mat2str(round((psn(2:end,jb)-ref(2:end,1))',2)),mean(psn(2:end,jb)-ref(2:end,1)))
fprintf("SURE gap (SURE-MSE)/MSE per variant, mean: %s\n",mat2str(round(mean((sre-mse)./mse,1),3)))
fprintf("free parameters per variant: %s\n",mat2str(nFree))
save(fullfile(datfolder,'sbdm_ssr_results.mat'),'psn','sre','mse','ref','nFree','vars','intIdx')

%%
function fcn_seed(s)
rng(s)
if canUseGPU, gpurng(s); end
end

function prm = fcn_lsunmovegpu(prm,useGpu)
if useGpu, prm = dlupdate(@gpuArray,prm); end
end

function p = fcn_psnr(xh,ref)
a = min(max(double(gather(extractdata(xh))),0),1);
p = psnr(a,double(gather(ref)));
end
