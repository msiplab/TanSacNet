%% probe_sbdm_salsun_dcsuggest.m
% Checks before main_sbdm_salsun_dcsuggest (2026-11-06):
%  1. reference caches (fields used by the driver)
%  2. exact SURE divergence with DC shrinkage and DC leakage vs Monte-Carlo
%     (N, kodim09 crop, sigma 25/255)
%  3. SA fixed-probe SURE with DC shrinkage vs the explicit chain
%  4. regression: the new code with no DC shrink / no leakage reproduces
%     the cached N-hard tau=3 and SAP-hard b2/it150 (kodim09, sigma 25)
%  5. timing of the supervised training (N and SA, batch 32, 64x64)
% Options: probeSkip = [..] skips parts.
here = fileparts(mfilename('fullpath'));
cd(fullfile(here,'..','..')); setpath; cd(here)
addpath(fullfile(here,'..','sbdm'))
addpath(fullfile(here,'..','salsun'))
warning('off','MATLAB:rmpath:DirNotFound')
rmpath(fullfile('/home/shogo/Workspace/GitHub/SaivDr','mexcodes'))
warning('on','MATLAB:rmpath:DirNotFound')
if ~exist('probeSkip','var'), probeSkip = []; end
dat = fullfile(here,'..','..','..','data');
useGpu = canUseGPU;
cfg = fcn_ssconfig(); cfg.K = 4; cfg.nProbePilot = 8; cfg.nProbeFresh = 32;
cfg.bottleneck = 18; cfg.tauList = [2 3];
fprintf("%s  probe start, GPU %d\n",string(datetime('now')),useGpu)

%% 1. caches
S = load(fullfile(dat,'sbdm_salsun_hard_09_N4_t3.mat')); qR = S.q;
fprintf("Nh s25 cache: psnr %.4f sure %.4e mse %.4e  tau %g  #th %d  fields prm.th/a/b: %d %d %d\n",...
    qR.psnr,qR.sure,qR.mse,qR.tau,numel(qR.prm.th{1}),isfield(qR.prm,'th'),isfield(qR.prm,'a'),isfield(qR.prm,'b'))
S = load(fullfile(dat,'sbdm_salsun_on_s25_09_SAPh_p3_b2_it150.mat')); qS = S.q;
fprintf("SAPh s25 cache: psnr %.4f ssim %.4f sure %.4e fresh %.4e mse %.4e thr/sigma %.4f prErrX %.2e t %.1f s (t1 %.1f t2 %.1f)\n",...
    qS.psnr,qS.ssim,qS.sure,qS.sureFresh,qS.mse,qS.thrOverSigma,qS.prErrX,qS.tTotal,qS.t1,qS.t2)
S = load(fullfile(dat,'sbdm_salsun_on_s10_09_Nh_t2.mat'));
fprintf("Nh s10 cache: psnr %.4f fields prm %d\n",S.q.psnr,isfield(S.q,'prm'))
S = load(fullfile(dat,'sbdm_salsun_on_s10_09_SAPh_p2_b3_it150.mat'));
fprintf("SAPh s10 cache: psnr %.4f thr/sigma %.4f\n",S.q.psnr,S.q.thrOverSigma)
S = load(fullfile(dat,'sbdm_salsun_09_W.mat')); fprintf("W s25 cache: psnr %.4f\n",S.q.psnr)
S = load(fullfile(dat,'sbdm_salsun_s10_09_W.mat')); fprintf("W s10 cache: psnr %.4f\n",S.q.psnr)

cfg.sigma = single(25/255);
[x,v] = fcn_ssloadimage(dat,9,cfg,useGpu);
sigma = cfg.sigma; N = numel(v); X = dlarray(v,'SSCB');

%% 2. exact SURE divergence vs Monte-Carlo (N)
if ~ismember(2,probeSkip)
fprintf("\n--- 2. N: exact divergence vs MC (kodim09 crop, sigma 25)\n")
rng(7);
for leak = [false true]
    for dcs = [false true]
        [anet,snet,info] = fcn_dclsunpair(size(v),cfg.stride,cfg.ovlp,leak);
        th = qR.prm.th{1};
        if leak   % move every angle, also the ones that are 0 without leakage
            th = cellfun(@(t) t + dlarray(0.3*randn(size(t),'single')),th,'UniformOutput',false);
        end
        if useGpu, th = dlupdate(@gpuArray,th); end
        ab = struct('a',dlarray(cast(gather(extractdata(qR.prm.a{1})),'like',v)),...
                    'b',dlarray(cast(gather(extractdata(qR.prm.b{1})),'like',v)));
        if dcs
            ab.a0 = dlarray(cast(0,'like',v));             % kappa0 = 0.5
            ab.b0 = dlarray(cast(log(0.05),'like',v));     % g0 = 0.05: not saturated
        end
        [anetT,snetT] = fcn_lsunsetangles(anet,snet,info,th);
        [ac,dc] = predict(anetT,X);
        % leakage check: AC energy of a constant image / its energy
        cimg = dlarray(0.5*ones(size(v),'like',v),'SSCB');
        [acc,~] = predict(anetT,cimg);
        leakE = double(gather(extractdata(sum(acc.^2,'all')/sum(cimg.^2,'all'))));
        [sure,divN,res] = fcn_dcsure(ab,ac,dc,sigma,N);
        fu = fcn_dcdenoise(anet,snet,info,th,ab,X,sigma);
        resImg = double(gather(extractdata(sum((fu - X).^2,'all'))))/N;
        % per-probe: FD vs exact quadratic form b' J b = sum S'(c) (E b)^2
        nP = 64; epsFd = 1e-3; dFd = zeros(nP,1); dQ = dFd;
        nAc = info.nChs - 1;
        kap = reshape(sigmoid(ab.a),1,1,nAc); gn = reshape(exp(ab.b),1,1,nAc);
        sAc = 1 - kap.*(1 - tanh(gn.*ac./sigma).^2);
        if dcs
            sDc = 1 - sigmoid(ab.a0).*(1 - tanh(exp(ab.b0).*dc./sigma).^2);
        else
            sDc = ones(size(dc),'like',dc);
        end
        for j = 1:nP
            b = dlarray(randn(size(v),'like',v),'SSCB');
            fb = fcn_dcdenoise(anet,snet,info,th,ab,X + epsFd*b,sigma);
            dFd(j) = double(gather(extractdata(sum(b.*(fb - fu),'all'))))/(epsFd*N);
            [eb,ed] = predict(anetT,b);
            dQ(j) = double(gather(extractdata(sum(sAc.*eb.^2,'all') + sum(sDc.*ed.^2,'all'))))/N;
        end
        dEx = double(gather(extractdata(divN)));
        fprintf(" leak %d dcs %d: DC leak energy %.2e | divN exact %.6f  MC(64) %.6f +- %.6f (z %.2f) | per probe FD vs exact quad.: max rel %.2e | res coef %.6e img %.6e | SURE %.6e\n",...
            leak,dcs,leakE,dEx,mean(dFd),std(dFd)/sqrt(nP),(mean(dFd)-dEx)/(std(dFd)/sqrt(nP)),...
            max(abs(dFd - dQ)./abs(dQ)),double(gather(extractdata(res))),resImg,double(gather(extractdata(sure))))
        if ~leak && ~dcs
            prm = struct('th',{{th}},'a',{{ab.a}},'b',{{ab.b}});
            [s0,~,d0] = fcn_lsunsure(anet,snet,info,prm,X,sigma,"exact");
            fprintf("   regression vs examples/sbdm/fcn_lsunsure exact: SURE %.6e vs %.6e, divN %.6f vs %.6f\n",...
                double(gather(extractdata(s0))),double(gather(extractdata(sure))),...
                double(gather(extractdata(d0))),dEx)
        end
    end
end
end

%% 3./4. N and SA with the new code
if ~ismember(4,probeSkip)
fprintf("\n--- 4. regression, kodim09, sigma 25: new code with R settings vs cache\n")
t = tic;
nR = fcn_dcnrun(x,v,cfg,9,false,false,3,qR.prm.th{1});
fprintf(" N-hard t3 (cached angles, new step 2): psnr %.4f vs cache %.4f, sure %.6e vs %.6e (%.1f s)\n",...
    nR.psnr,qR.psnr,nR.sure,qR.sure,toc(t))
t = tic;
nD = fcn_dcnrun(x,v,cfg,9,false,true,3,qR.prm.th{1});
fprintf(" N-hard t3 + DC shrink: psnr %.4f, sure %.6e, init %s (ac: sure %.6e psnr %.4f / id: sure %.6e psnr %.4f) kappa0 %.4f g0 %.4f (%.1f s, t2 %.1f s)\n",...
    nD.psnr,nD.sure,nD.selInit,nD.init(1).sure,nD.init(1).psnr,nD.init(2).sure,nD.init(2).psnr,nD.kappa0,nD.gain0,toc(t),nD.t2)
end
if ~ismember(3,probeSkip)
fprintf("\n--- 3. SA fixed-probe SURE with DC shrinkage vs explicit chain (20 step-1 its)\n")
[anetP,snetP,infoP] = fcn_dclsunpair(size(v),cfg.stride,cfg.ovlp,false);
prmP = qR.prm; if useGpu, prmP = dlupdate(@gpuArray,prmP); end
abP = struct('a',prmP.a{1},'b',prmP.b{1});
Pf = @(Z) fcn_dcdenoise(anetP,snetP,infoP,prmP.th{1},abP,Z,sigma);
thr = 2*sqrt(qR.sure);
for leak = [false true]
    t = tic;
    r = fcn_dcsaprun(x,v,cfg,9,Pf,qR.img,leak,true,thr,20);
    fprintf(" leak %d dcs 1: SURE fixed %.6e explicit(same probes) %.6e (rel %.2e) fresh %.6e +- %.1e | MSE %.6e | init %s kappa0 %.4f g0 %.4f | pDiff %.2e constCheck %.2e prErrX %.2e (%.1f s)\n",...
        leak,r.sure,r.sureExplicitSame,abs(r.sure - r.sureExplicitSame)/abs(r.sureExplicitSame),...
        r.sureFresh,r.sureFreshSe,r.mse,r.selInit,r.kappa0,r.gain0,r.pDiff,r.constCheck,r.prErrX,toc(t))
end
end
if ~ismember(4,probeSkip)
t = tic;
r = fcn_dcsaprun(x,v,cfg,9,Pf,qR.img,false,false,thr,150);
fprintf(" SAP-hard b2/it150 (new code, R): psnr %.4f vs cache %.4f, ssim %.4f vs %.4f, sure %.6e vs %.6e, thr/sigma %.4f vs %.4f (%.1f s)\n",...
    r.psnr,qS.psnr,r.ssim,qS.ssim,r.sure,qS.sure,r.thrOverSigma,qS.thrOverSigma,toc(t))
end

%% 5. supervised training speed
if ~ismember(5,probeSkip)
fprintf("\n--- 5. supervised training speed (batch 32, 64x64)\n")
rng(1); if useGpu, gpurng(1); end
[~,trainSet] = fcn_loadkodak(dat,1:8,[64 64],256,useGpu);
fprintf(" training patches: %d\n",numel(trainSet))
o = struct('sigma',sigma,'nIters',30,'mb',32,'lrN',[5e-3 5e-5],'lrEst',[1e-3 1e-5],...
    'lrShr',[5e-3 5e-5],'seed',1,'stride',cfg.stride,'ovlp',cfg.ovlp,'bottleneck',cfg.bottleneck,...
    'patch',[64 64],'ckptFile','','ckptEvery',inf);
for kind = ["N" "SA"]
    for lk = [false true]
        o.leak = lk; o.dcShrink = lk;
        t = tic;
        [mdl,tr] = fcn_dcsuptrain(kind,trainSet,o);
        tTot = toc(t);
        t = tic;
        q = fcn_dcsupeval(mdl,x,v,cfg);
        fprintf(" %s leak %d dcs %d: %.3f s/it (30 its, incl. build: %.1f s), params %d, loss %.4e -> %.4e | eval 256x256 %.1f s psnr %.2f prErrX %.2e\n",...
            kind,lk,lk,tr.tPerIt,tTot,tr.nParams,tr.hist(1),tr.hist(end),toc(t),q.psnr,q.prErrX)
    end
end
end
fprintf("%s  probe done\n",string(datetime('now')))
