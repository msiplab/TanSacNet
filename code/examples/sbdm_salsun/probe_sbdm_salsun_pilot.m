%% probe_sbdm_salsun_pilot.m
% One image (kodim09, K = 2) of the pilot arm: time estimate and checks
%  - frozen-angle net on p == estimator net on p
%  - fixed-probe SURE (shortcut) == explicit chain with the same probes
%  - fixed-probe SURE ~ per-iteration-style MC SURE (fresh probes),
%    at the initial and at the final kappa_p, g_p
% Writes no cache. Run from this folder:
%   matlab -licmode onlinelicensing -batch "probe_sbdm_salsun_pilot"

here = fileparts(mfilename('fullpath'));
cd(fullfile(here,'..','..')); setpath; cd(here)
addpath(fullfile(here,'..','sbdm')); addpath(fullfile(here,'..','salsun'))
warning('off','MATLAB:rmpath:DirNotFound')
rmpath(fullfile('/home/shogo/Workspace/GitHub/SaivDr','mexcodes'))
warning('on','MATLAB:rmpath:DirNotFound')

cfg = fcn_ssconfig(); cfg.nProbePilot = 8; cfg.nProbeFresh = 32;
datfolder = fullfile(here,'..','..','..','data');
useGpu = canUseGPU; idx = 9; K = 2;
[x,v] = fcn_ssloadimage(datfolder,idx,cfg,useGpu);
S = load(fullfile(datfolder,sprintf('sbdm_salsun_%02d_N%d.mat',idx,K))); qN = S.q;

tAll = tic;
[r,ctx] = fcn_sspilotrunimage(x,v,K,cfg,cfg.seedBase + idx,qN);
tImg = toc(tAll);

fprintf("checks: p vs cached N output max diff %.2e; frozen-angle net vs estimator net on p max diff %.2e\n",...
    r.pDiffCache,r.constCheck)
fprintf("PR with frozen angles: x %.2e, v %.2e\n",r.prErrX,r.prErrV)

% SURE checks at the initial parameters
sd = ctx.sigma; N = numel(v);
ab0.a = dlarray(log(0.9/0.1)*ones(15,1,'like',ctx.pre.ac0));
ab0.b = dlarray(log(0.9/2.0)*ones(15,1,'like',ctx.pre.ac0));
s0 = gather(extractdata(fcn_sspilotsure(ab0,ctx.pre,sd,ctx.epsFd)));
[divJ,fu0] = fcn_sspilotexplicit(ctx,ab0,ctx.B);
res0 = double(gather(sum((extractdata(fu0) - v).^2,'all')))/N;
sSame0 = res0 - sd^2 + 2*sd^2*mean(divJ);
rng(12345,'twister'); Bf = randn([size(v) 1 cfg.nProbeFresh],'single');
if useGpu, Bf = gpuArray(Bf); end
divF = fcn_sspilotexplicit(ctx,ab0,Bf); sF0 = res0 - sd^2 + 2*sd^2*divF;
fprintf("\nSURE at INITIAL kappa,g (x1e3): fixed %.4f | explicit same probes %.4f | fresh (32) %.4f +- %.4f (se)\n",...
    1e3*s0,1e3*sSame0,1e3*mean(sF0),1e3*std(sF0)/sqrt(numel(sF0)))
fprintf("SURE at FINAL   kappa,g (x1e3): fixed %.4f | explicit same probes %.4f | fresh (32) %.4f +- %.4f (se) | frozen %.4f\n",...
    1e3*r.sure,1e3*r.sureExplicitSame,1e3*r.sureFresh,1e3*r.sureFreshSe,1e3*r.sureFrozen)

fprintf("\nTiming, kodim09 K=%d: build %.1f s | step1 %.1f s (%d its, %.3f s/it) | prep %.1f s | step2 %.1f s (%d its) | eval %.1f s | total %.0f s\n",...
    K,r.tBuild,r.t1,cfg.nIt1Sa,r.t1/cfg.nIt1Sa,r.tPrep,r.t2,cfg.nIt2,r.tEval,tImg)
fprintf("Estimate full run (%d images x %d K): %.0f s (%.1f min)\n",numel(cfg.imgIdx),numel(cfg.K),...
    tImg*numel(cfg.imgIdx)*numel(cfg.K),tImg*numel(cfg.imgIdx)*numel(cfg.K)/60)
