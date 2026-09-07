%[text] # Experiment 1, NSOLT row: an oversampled tight frame that is not unitary
%
% Same protocol as Experiment 1 of main_sbdm.m -- trained at the single noise
% level sigma = 25/255 on 2048 patches of kodim01--08, evaluated on 128
% held-out patches of kodim09--24 -- so the row drops straight into Table 1.
% Checkpointed: re-run until it reports nothing pending.
clc, clear, close all
here = pwd;
saivdrRoot = '/home/shogo/Workspace/GitHub/SaivDr';
run(fullfile('..','..','setpath.m'))      % TanSacNet, for the example helpers
run(fullfile(saivdrRoot,'setpath.m'))     % SaivDr, for the NSOLT layers
cd(here)

% Both packages ship identically named mex kernels (fcn_orthmtxgen_*_mex) in
% non-package folders. Only NSOLT is exercised here, so TanSacNet's copy is
% dropped and SaivDr's is kept, which lets the NSOLT run at full speed.
warning('off','MATLAB:rmpath:DirNotFound')
rmpath(fullfile(here,'..','..','mexcodes'))
warning('on','MATLAB:rmpath:DirNotFound')

datfolder = fullfile('..','..','..','data');
szOrg   = [64 64];
dec     = [4 4];
ppo     = [2 2];                  % LSUN overlapping factor [3 3]
nChs    = [10 10];                % 20 channels, redundancy 20/16 = 1.25
sigmaFix = single(25/255);
nIters = 1500;                    % as for the LSUN in Experiment 1
lr     = 5e-3;
mbSize = 32;
nProbe = 4; nPower = 60;
trainIdx = 1:8; evalIdx = 9:24; nCrops = 256;
useGpu = canUseGPU;

timeBudget = 800;
tRun = tic;
remaining = @() timeBudget - toc(tRun);

fcn_seed(0)
[anet,snet,info] = fcn_nsoltpair(szOrg,dec,nChs,ppo);
fprintf("NSOLT: blocks %d, channels %d, redundancy %.2f, angles %d over %d layers\n",...
    info.nBlocks,info.nChs,info.redundancy,sum(info.nAngles),info.nRotations)

fcn_seed(1)
[~,trainSet] = fcn_loadkodak(datfolder,trainIdx,szOrg,nCrops,useGpu);
fcn_seed(3)
[~,evalPatches] = fcn_loadkodak(datfolder,evalIdx,szOrg,8,useGpu);
fprintf("training patches %d, held-out patches %d\n",numel(trainSet),numel(evalPatches))

f1 = fullfile(datfolder,'sbdm_exp1_nsolt.mat');
if isfile(f1)
    S = load(f1); prm = S.prm; rec = S.rec;
    fprintf("loaded %s\n",f1)
else
    ck = fullfile(datfolder,'sbdm_ckpt_exp1_nsolt.mat');
    prm = fcn_nsoltinitprm(info,1,0.9,2.0);
    if useGpu, prm = dlupdate(@gpuArray,prm); end
    [prm,~,isDone] = fcn_trainnsolt(anet,snet,info,prm,trainSet,sigmaFix,...
        nIters,lr,mbSize,true,ck,remaining());
    if ~isDone
        fprintf("\n*** unfinished: run this script again ***\n"); return
    end
    fden  = @(v) fcn_nsoltdenoise(anet,snet,info,prm,v,sigmaFix);
    fpair = @(v) fcn_nsoltpairapply(anet,snet,info,prm.th{1},v);
    d = fcn_denoiserdiag(fden,fpair,szOrg,nProbe,nPower);
    rec = struct('name','NSOLT, oversampled','nPrm',fcn_countprm(prm),...
        'tightErr',d.tightErr,'jacAsym',d.jacAsym,'jacRho',d.jacRho,'psnr',[]);
    rec.psnr = fcn_psnrpatches(fden,evalPatches,sigmaFix,mbSize);
    save(f1,'rec','prm')
end

fprintf("\n=== Table 1 row ===\n");
fprintf("%-20s #prm %5d  PSNR %6.2f  tight %8.1e  asym %8.1e  rho %.4f\n",...
    rec.name,rec.nPrm,rec.psnr,rec.tightErr,rec.jacAsym,rec.jacRho);

% how far white image noise is from white coefficient noise, the quantity
% Proposition 2 needs to be one in every channel
fcn_seed(7)
sAll = [];
for r = 1:8
    [ac,dc] = forward(anet,dlarray(randn(szOrg,'single'),'SSCB'));
    c = cat(3,double(gather(extractdata(dc))),double(gather(extractdata(ac))));
    s = zeros(size(c,3),1);
    for pp = 1:size(c,3), t = c(:,:,pp,:); s(pp) = std(t(:)); end
    sAll = [sAll s]; %#ok<AGROW>
end
sm = mean(sAll,2);
fprintf("coefficient noise level, 8 draws: mean %.4f, min %.4f, max %.4f, dead channels %d/%d\n",...
    mean(sm),min(sm),max(sm),sum(sm<1e-3),numel(sm));

% ---------------------------------------------------------------- helpers
function fcn_seed(s)
rng(s)
if canUseGPU, gpurng(s); end
end

function y = fcn_nsoltpairapply(anet,snet,info,th,v)
[anet,snet] = fcn_nsoltsetangles(anet,snet,info,th);
[ac,dc] = forward(anet,v);
y = forward(snet,ac,dc);
end

function p = fcn_psnrpatches(fden,patches,sigma,batchSize)
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
