%% main_sbdm_salsun.m
% Self-supervised SA-LSUN vs uniform LSUN for AWGN denoising.
% Sensei's decision: both steps use ONLY the observation v (no training
% images). The clean image is used only for evaluation.
%
%   Step 1: zonal-coding autoencoder (DC + K AC kept), loss ||v||^2 - ||F(v)||^2
%   Step 2: transform frozen, learn kappa_p, g_p (30 parameters) by SURE on v
%   Arms  : W (wiener2 5x5), N (uniform LSUN, exact SURE div),
%           SA (SA-LSUN, ThetaMode 'Reuse', Monte-Carlo SURE div)
%
% Resumable: one cache file per (image, arm, K) in data/sbdm_salsun_*.mat.
% Run from this folder:
%   nohup matlab -licmode onlinelicensing -batch "main_sbdm_salsun" \
%       > ~/Documents/SBDMwLSUN/logs/salsun_internal.log 2>&1 &

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
nI = numel(cfg.imgIdx); nK = numel(cfg.K);
fprintf("%s  start. GPU: %d\n",string(datetime),useGpu)
disp(cfg)

%% Run (or load) every (image, arm, K)
R = cell(nI,1);
for i = 1:nI
    idx = cfg.imgIdx(i);
    [x,v] = fcn_ssloadimage(datfolder,idx,cfg,useGpu);
    rec.idx = idx;
    rec.psnrNoisy = psnr(gather(v),gather(x));
    rec.ssimNoisy = ssim(gather(v),gather(x));
    fprintf("\n%s  kodim%02d  noisy %.2f dB\n",string(datetime),idx,rec.psnrNoisy)

    rec.W = runcached(datfolder,idx,"W",[],x,v,cfg);
    for k = 1:nK
        rec.N(k)  = runcached(datfolder,idx,"N", cfg.K(k),x,v,cfg);
        rec.SA(k) = runcached(datfolder,idx,"SA",cfg.K(k),x,v,cfg);
    end
    R{i} = rec;
end
R = [R{:}];
save(fullfile(datfolder,'sbdm_salsun_results.mat'),'R','cfg')

%% Summary
armNames = ["W"]; P = []; S = []; T = [];
P(:,1) = arrayfun(@(r) r.W.psnr,R)'; S(:,1) = arrayfun(@(r) r.W.ssim,R)'; T(:,1) = arrayfun(@(r) r.W.time,R)';
for a = ["N","SA"]
    for k = 1:nK
        armNames(end+1) = a + " K=" + cfg.K(k); %#ok<SAGROW>
        P(:,end+1) = arrayfun(@(r) r.(a)(k).psnr,R)'; %#ok<SAGROW>
        S(:,end+1) = arrayfun(@(r) r.(a)(k).ssim,R)'; %#ok<SAGROW>
        T(:,end+1) = arrayfun(@(r) r.(a)(k).tBuild + r.(a)(k).t1 + r.(a)(k).t2 + r.(a)(k).tEval,R)'; %#ok<SAGROW>
    end
end
rows = [compose("kodim%02d",cfg.imgIdx(:)); "mean"];

fprintf("\n================ Summary ================\n")
fprintf("Self-supervised (observation only), sigma = 25/255, %dx%d centre crops, 4x4 blocks, ovlp 3x3, S = 1\n",cfg.cropSize)
fprintf("Iterations: step1 N %d, SA %d; step2 %d. MC probes: train 1/iter, report %d.\n",...
    cfg.nIt1N,cfg.nIt1Sa,cfg.nIt2,cfg.nProbeEval)
pn = [arrayfun(@(r) r.psnrNoisy,R)'; mean(arrayfun(@(r) r.psnrNoisy,R))];
printtable("PSNR [dB]",rows,["noisy" armNames],[pn [P; mean(P,1)]],"%10.2f")
sn = [arrayfun(@(r) r.ssimNoisy,R)'; mean(arrayfun(@(r) r.ssimNoisy,R))];
printtable("SSIM",rows,["noisy" armNames],[sn [S; mean(S,1)]],"%10.4f")

D = []; dn = strings(0);
for k = 1:nK
    D(:,end+1) = P(:,1+nK+k) - P(:,1+k); dn(end+1) = "SA-N K=" + cfg.K(k); %#ok<SAGROW>  % columns: W | N K's | SA K's
end
for k = 1:nK
    D(:,end+1) = P(:,1+k) - P(:,1); dn(end+1) = "N-W K=" + cfg.K(k); %#ok<SAGROW>
end
printtable("PSNR differences [dB]",rows,dn,[D; mean(D,1)],"%10.2f")

% SURE vs true MSE (step 2, final parameters), x 1e3
for a = ["N","SA"]
    for k = 1:nK
        mseT = arrayfun(@(r) r.(a)(k).mse,R)'; sre = arrayfun(@(r) r.(a)(k).sure,R)';
        if a == "N"
            cols = ["SURE exact" "true MSE" "SURE-MSE"];
            M = [sre mseT sre-mseT];
        else
            sfr = arrayfun(@(r) r.SA(k).sureFrozen,R)';
            cols = ["SURE MC" "SURE frozen" "true MSE" "MC-MSE" "frozen-MSE"];
            M = [sre sfr mseT sre-mseT sfr-mseT];
        end
        printtable(a + " K=" + cfg.K(k) + ": SURE vs true MSE (x1e3)",rows,cols,1e3*[M; mean(M,1)],"%12.4f")
    end
end

% Step-1 kept-energy ratio, transform learned on v, applied to v and to x
for a = ["N","SA"]
    for k = 1:nK
        M = [arrayfun(@(r) r.(a)(k).keptAc,R)' arrayfun(@(r) r.(a)(k).keptAcX,R)' ...
             arrayfun(@(r) r.(a)(k).keptAll,R)' arrayfun(@(r) r.(a)(k).keptAllX,R)'];
        printtable(a + " K=" + cfg.K(k) + ": step-1 kept energy (AC part / all; x = evaluation only)",...
            rows,["AC on v" "AC on x" "all on v" "all on x"],[M; mean(M,1)],"%10.4f")
    end
end

% Model size and perfect reconstruction
fprintf("\nPixels per image: %d\n",prod(cfg.cropSize))
for k = 1:nK
    fprintf("K=%d: N shared angles %d; SA estimator weights %d (%.2f x pixels)\n",cfg.K(k),...
        R(1).N(k).nEstW,R(1).SA(k).nEstW,R(1).SA(k).nEstW/prod(cfg.cropSize))
end
for k = 1:nK
    M = [arrayfun(@(r) r.SA(k).prErrX,R)' arrayfun(@(r) r.SA(k).prErrV,R)'];
    printtable("SA K=" + cfg.K(k) + ": ||DEx-x||/||x|| (Whole, Reuse, learned estimator)",...
        [rows(1:end-1); "max"],["on x" "on v"],[M; max(M,[],1)],"%10.2e")
end

printtable("Run time per image [s] (build + step1 + step2 + eval)",rows,armNames,[T; mean(T,1)],"%10.1f")
fprintf("\n%s  done.\n",string(datetime))

%% Local functions
function q = runcached(datfolder,idx,arm,K,x,v,cfg)
if isempty(K), tag = arm; else, tag = arm + K; end
f = fullfile(datfolder,sprintf('sbdm_salsun_%02d_%s.mat',idx,tag));
if isfile(f)
    S = load(f); q = S.q;
    fprintf("  %-4s cached (%s)\n",tag,f)
    return
end
c = cfg;
if ~isempty(K), c.K = K; end
t = tic;
r = fcn_ssrunimage(x,v,c,cfg.seedBase + idx,arm);
q = r.(arm);
ftmp = f + ".tmp.mat";
save(ftmp,'q','-v7.3'); movefile(ftmp,f);
if arm == "W"
    fprintf("  %-4s %.2f dB  (%.0f s)\n",tag,q.psnr,toc(t))
elseif arm == "N"
    fprintf("  %-4s %.2f dB  SURE %.4e MSE %.4e  kept AC v %.4f x %.4f  (%.0f s)\n",...
        tag,q.psnr,q.sure,q.mse,q.keptAc,q.keptAcX,toc(t))
else
    fprintf("  %-4s %.2f dB  SURE MC %.4e frozen %.4e MSE %.4e  kept AC v %.4f x %.4f  PR %.1e  (%.0f s)\n",...
        tag,q.psnr,q.sure,q.sureFrozen,q.mse,q.keptAc,q.keptAcX,q.prErrX,toc(t))
end
end

function printtable(ttl,rows,cols,M,fmt)
fprintf("\n%s\n",ttl)
w = max(10,max(strlength(cols))+1);
fprintf("%-9s","");
for j = 1:numel(cols), fprintf("%*s",w,cols(j)); end
fprintf("\n")
for i = 1:numel(rows)
    fprintf("%-9s",rows(i));
    for j = 1:size(M,2)
        s = sprintf(fmt,M(i,j));
        fprintf("%*s",w,strtrim(s));
    end
    fprintf("\n")
end
end
