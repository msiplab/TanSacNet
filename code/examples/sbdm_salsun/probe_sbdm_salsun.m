%% probe_sbdm_salsun.m
% Sanity checks + timing on ONE image (kodim09). Short runs only; it
% prints no PSNR/SSIM. The full run is main_sbdm_salsun.m.
%
% Run from this folder:
%   matlab -licmode onlinelicensing -batch "probe_sbdm_salsun"

here = fileparts(mfilename('fullpath'));
cd(fullfile(here,'..','..')); setpath; cd(here)
addpath(fullfile(here,'..','sbdm'))     % helpers (called, not edited)
addpath(fullfile(here,'..','salsun'))   % maskLayer
warning('off','MATLAB:rmpath:DirNotFound')
rmpath(fullfile('/home/shogo/Workspace/GitHub/SaivDr','mexcodes'))
warning('on','MATLAB:rmpath:DirNotFound')

cfg = fcn_ssconfig();
datfolder = fullfile(here,'..','..','..','data');
useGpu = canUseGPU;
if useGpu, g = gpuDevice; fprintf("GPU: %s\n",g.Name); end
[x,v] = fcn_ssloadimage(datfolder,9,cfg,useGpu);
X = dlarray(v,'SSCB');
sz = size(v); sigma = cfg.sigma;

%% Sanity 1: SA-LSUN (zero angles at init) vs uniform LSUN (zero angles)
rng(0); if useGpu, gpurng(0); end
acMask = fcn_ssacmask(16,2);
fprintf("K = 2 AC mask (15 AC channels): %s\n",mat2str(acMask.'))
fprintf("K = 4 AC mask (15 AC channels): %s\n",mat2str(fcn_ssacmask(16,4).'))
nets = fcn_ssbuildsalsun(sz,cfg.stride,cfg.ovlp,acMask,sigma);
if useGpu
    nets.shrink = dlupdate(@gpuArray,nets.shrink);
    nets.whole  = dlupdate(@gpuArray,nets.whole);
end
[acS,dcS,fS] = predict(nets.shrink,X,'Outputs',{'Lv1_AcOut','Lv1_DcOut','Lv1_Out'});
[anet,snet,info] = fcn_lsunpair(sz,cfg.stride,cfg.ovlp);
prm = fcn_lsuninitprm(info,1,false);
if useGpu, prm = dlupdate(@gpuArray,prm); end
[anet,snet] = fcn_lsunsetangles(anet,snet,info,prm.th{1});
[acL,dcL] = predict(anet,X);
fL = fcn_lsundenoise(anet,snet,info,prm,X,sigma);
mx = @(a) double(gather(extractdata(max(abs(a),[],'all'))));
fprintf("[1] max|ac_SA - ac_LSUN| = %.2e, max|dc_SA - dc_LSUN| = %.2e\n",mx(acS-acL),mx(dcS-dcL))
fprintf("[1] max|f_SA(v) - f_LSUN(v)| at init = %.2e\n",mx(fS-fL))

%% Sanity 2: SURE at init (angles constant): SA MC vs SA frozen vs LSUN exact
b = dlarray(randn([sz 1 cfg.nProbeEval],'like',v),'SSCB');
[s1,~,d1] = fcn_sssuresalsun(nets.shrink,X,sigma,"mc",b,cfg.epsFd);
[s2,~,d2] = fcn_sssuresalsun(nets.shrink,X,sigma,"frozen");
[s3,~,d3] = fcn_lsunsure(anet,snet,info,prm,X,sigma,"exact");
g = @(a) double(gather(extractdata(a)));
fprintf("[2] div/N: SA-MC %.6f, SA-frozen %.6f, LSUN-exact %.6f\n",g(d1),g(d2),g(d3))
fprintf("[2] SURE : SA-MC %.4e, SA-frozen %.4e, LSUN-exact %.4e\n",g(s1),g(s2),g(s3))

%% Sanity 3: input-dependent angles (random estimator output weights)
L = nets.whole.Learnables;
for i = find(ismember(L.Parameter,{'Wo','Bo'})).'
    L.Value{i} = 0.1*randn(size(L.Value{i}),'like',L.Value{i});
end
nets.whole.Learnables = L;
y = predict(nets.whole,X);
fprintf("[3] random angles: ||DEv - v||/||v|| = %.2e\n",...
    g(sqrt(sum((y-X).^2,'all')/sum(X.^2,'all'))))
th = predict(nets.whole,X,'Outputs','Lv1_Cmp1_V0_Theta');
fprintf("[3] random angles: std over blocks of V0 angles = %.3f (should be > 0)\n",...
    g(mean(std(th,0,2),'all')))
nets.shrink = fcn_copylearnablesbyname(nets.shrink,nets.whole);
[s1,~,d1] = fcn_sssuresalsun(nets.shrink,X,sigma,"mc",b,cfg.epsFd);
[s2,~,d2] = fcn_sssuresalsun(nets.shrink,X,sigma,"frozen");
fprintf("[3] random angles: div/N MC %.6f vs frozen %.6f (gap = angle dependence + MC noise)\n",g(d1),g(d2))

%% Timing: short runs, extrapolated to the planned iterations
nP = 40;
cfgP = cfg; cfgP.nIt1Sa = nP; cfgP.nIt1N = nP; cfgP.nIt2 = nP;
tAll = tic;
r = fcn_ssrunimage(x,v,cfgP,9);
tProbe = toc(tAll);
fprintf("\nTiming probe on kodim09 (%d iterations per step, includes warm-up)\n",nP)
fprintf("W: %.2f s\n",r.W.time)
tot = r.W.time;
for arm = ["N","SA"]
    for k = 1:numel(cfg.K)
        q = r.(arm)(k);
        if arm == "N", n1 = cfg.nIt1N; else, n1 = cfg.nIt1Sa; end
        tFull = q.tBuild + q.t1*n1/nP + q.t2*cfg.nIt2/nP + q.tEval;
        tot = tot + tFull;
        fprintf("%-2s K=%d: build %.1f s | step1 %.3f s/it | step2 %.3f s/it | eval %.1f s | est. full %.0f s\n",...
            arm,cfg.K(k),q.tBuild,q.t1/nP,q.t2/nP,q.tEval,tFull)
    end
end
fprintf("Probe wall time %.0f s. Estimated full time: %.0f s per image, %.0f s (%.1f min) for %d images\n",...
    tProbe,tot,tot*numel(cfg.imgIdx),tot*numel(cfg.imgIdx)/60,numel(cfg.imgIdx))
fprintf("Estimator weights (SA): %d, shared angles (N): %d, pixels: %d\n",...
    r.SA(1).nEstW,r.N(1).nEstW,numel(v))
fprintf("SA PR check after %d step-1 its: ||DEx-x||/||x|| = %.2e\n",nP,r.SA(1).prErrX)
