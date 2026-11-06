%% main_sbdm_salsun_pilot.m
% SA-LSUN pilot arm: step 1 on the pilot p = N(v) instead of v, angles
% estimated from p and frozen, step 2 (kappa_p, g_p) by SURE on v with
% fixed probes through the whole chain. Observation only; the clean
% image is used only for evaluation. W, N and SA (old) are read from the
% caches of main_sbdm_salsun.m.
%
% Resumable: one cache file per (image, K): data/sbdm_salsun_pilot_<img>_SAP<K>.mat
%   nohup matlab -licmode onlinelicensing -batch "main_sbdm_salsun_pilot" \
%       > ~/Documents/SBDMwLSUN/logs/salsun_pilot.log 2>&1 &

here = fileparts(mfilename('fullpath'));
cd(fullfile(here,'..','..')); setpath; cd(here)
addpath(fullfile(here,'..','sbdm'))     % helpers (called, not edited)
addpath(fullfile(here,'..','salsun'))   % maskLayer
warning('off','MATLAB:rmpath:DirNotFound')
rmpath(fullfile('/home/shogo/Workspace/GitHub/SaivDr','mexcodes'))
warning('on','MATLAB:rmpath:DirNotFound')

cfg = fcn_ssconfig();
cfg.nProbePilot = 8;     % fixed probes in step 2
cfg.nProbeFresh = 32;    % fresh probes for the explicit MC check
datfolder = fullfile(here,'..','..','..','data');
useGpu = canUseGPU;
nI = numel(cfg.imgIdx); nK = numel(cfg.K);
fprintf("%s  start. GPU: %d\n",string(datetime),useGpu)
disp(cfg)

%% Run (or load)
R = cell(nI,1);
for i = 1:nI
    idx = cfg.imgIdx(i);
    [x,v] = fcn_ssloadimage(datfolder,idx,cfg,useGpu);
    rec.idx = idx;
    rec.psnrNoisy = psnr(gather(v),gather(x)); rec.ssimNoisy = ssim(gather(v),gather(x));
    fprintf("\n%s  kodim%02d  noisy %.2f dB\n",string(datetime),idx,rec.psnrNoisy)
    rec.W = loadq(datfolder,idx,"W");
    for k = 1:nK
        K = cfg.K(k);
        rec.N(k)  = loadq(datfolder,idx,"N" + K);
        rec.SA(k) = loadq(datfolder,idx,"SA" + K);
        f = fullfile(datfolder,sprintf('sbdm_salsun_pilot_%02d_SAP%d.mat',idx,K));
        if isfile(f)
            S = load(f); q = S.q;
            fprintf("  SAP%d cached (%s)\n",K,f)
        else
            t = tic;
            q = fcn_sspilotrunimage(x,v,K,cfg,cfg.seedBase + idx,rec.N(k));
            ftmp = string(f) + ".tmp.mat";
            save(ftmp,'q','-v7.3'); movefile(ftmp,f);
            fprintf("  SAP%d %.2f dB  SURE fixed %.4e same-explicit %.4e fresh %.4e (se %.1e) frozen %.4e MSE %.4e  kept AC v %.4f p %.4f x %.4f  PR %.1e  (%.0f s)\n",...
                K,q.psnr,q.sure,q.sureExplicitSame,q.sureFresh,q.sureFreshSe,q.sureFrozen,q.mse,...
                q.keptAcV,q.keptAcP,q.keptAcX,q.prErrX,toc(t))
        end
        rec.SAP(k) = q;
    end
    R{i} = rec;
end
R = [R{:}];
save(fullfile(datfolder,'sbdm_salsun_pilot_results.mat'),'R','cfg')

%% Summary
armNames = "W"; P = []; S = []; T = [];
P(:,1) = arrayfun(@(r) r.W.psnr,R)'; S(:,1) = arrayfun(@(r) r.W.ssim,R)';
for a = ["N","SA","SAP"]
    for k = 1:nK
        armNames(end+1) = a + " K=" + cfg.K(k); %#ok<SAGROW>
        P(:,end+1) = arrayfun(@(r) r.(a)(k).psnr,R)'; %#ok<SAGROW>
        S(:,end+1) = arrayfun(@(r) r.(a)(k).ssim,R)'; %#ok<SAGROW>
    end
end
rows = [compose("kodim%02d",cfg.imgIdx(:)); "mean"];
iN = 1 + (1:nK); iSA = 1 + nK + (1:nK); iSAP = 1 + 2*nK + (1:nK);

fprintf("\n================ Summary (pilot) ================\n")
fprintf("Observation only, sigma = 25/255, %dx%d centre crops, 4x4 blocks, ovlp 3x3, S = 1\n",cfg.cropSize)
fprintf("SA = old arm (step 1 on v). SAP = pilot arm (step 1 on p = N(v), angles from p frozen).\n")
fprintf("SAP: step1 %d its on p; step2 %d its, %d fixed probes; check with %d fresh probes.\n",...
    cfg.nIt1Sa,cfg.nIt2,cfg.nProbePilot,cfg.nProbeFresh)
pn = arrayfun(@(r) r.psnrNoisy,R)'; sn = arrayfun(@(r) r.ssimNoisy,R)';
printtable("PSNR [dB]",rows,["noisy" armNames],[[pn; mean(pn)] [P; mean(P,1)]],"%10.2f")
printtable("SSIM",rows,["noisy" armNames],[[sn; mean(sn)] [S; mean(S,1)]],"%10.4f")
D = [P(:,iSAP) - P(:,iN), P(:,iSAP) - P(:,iSA)];
dn = ["SAP-N K=" + cfg.K, "SAP-SA K=" + cfg.K];
printtable("PSNR differences [dB]",rows,dn,[D; mean(D,1)],"%10.2f")
D = [S(:,iSAP) - S(:,iN), S(:,iSAP) - S(:,iSA)];
printtable("SSIM differences",rows,dn,[D; mean(D,1)],"%10.4f")

for k = 1:nK
    g = @(f) arrayfun(@(r) r.SAP(k).(f),R)';
    M = [g('keptAcV') g('keptAcP') g('keptAcX') g('keptAllV') g('keptAllP') g('keptAllX')];
    printtable("SAP K=" + cfg.K(k) + ": kept energy with the angles from p (AC part / all; x = evaluation only)",...
        rows,["AC on v" "AC on p" "AC on x" "all on v" "all on p" "all on x"],[M; mean(M,1)],"%10.4f")
end

for k = 1:nK
    g = @(f) arrayfun(@(r) r.SAP(k).(f),R)';
    M = 1e3*[g('sure') g('sureExplicitSame') g('sureFresh') g('sureFreshSe') g('sureFrozen') g('mse') g('sure')-g('mse') g('sureFrozen')-g('mse')];
    printtable("SAP K=" + cfg.K(k) + ": SURE vs true MSE (x1e3). fixed = 8 fixed probes (shortcut), same = explicit chain with the same probes, fresh = explicit, 32 fresh probes (se = std. error)",...
        rows,["fixed" "same" "fresh" "fresh se" "frozen" "true MSE" "fixed-MSE" "frozen-MSE"],[M; mean(M,1)],"%11.4f")
end

for k = 1:nK
    g = @(f) arrayfun(@(r) r.SAP(k).(f),R)';
    M = [g('prErrX') g('prErrV') g('constCheck') g('pDiffCache')];
    printtable("SAP K=" + cfg.K(k) + ": PR error ||DEz-z||/||z|| (angles from p) and checks",...
        [rows(1:end-1); "max"],["PR on x" "PR on v" "const-est" "p-cache"],[M; max(M,[],1)],"%11.2e")
end
fprintf("const-est = max|coefficients of the frozen-angle net - estimator net| on p; p-cache = max|N(v) - cached N output|\n")
fprintf("\nPixels per image: %d. SA estimator weights: %d (%.2f x pixels). Step 2 learns 30 parameters.\n",...
    prod(cfg.cropSize),R(1).SAP(1).nEstW,R(1).SAP(1).nEstW/prod(cfg.cropSize))

Tm = [];
cn = strings(0);
for k = 1:nK
    g = @(f) arrayfun(@(r) r.SAP(k).(f),R)';
    Tm = [Tm g('tBuild') g('t1') g('tPrep') g('t2') g('tEval')]; %#ok<AGROW>
    cn = [cn "K" + cfg.K(k) + " " + ["build" "step1" "prep" "step2" "eval"]]; %#ok<AGROW>
end
Tm(:,end+1) = sum(Tm,2); cn(end+1) = "total";
printtable("SAP run time per image [s] (prep = angles + fixed probes)",rows,cn,[Tm; mean(Tm,1)],"%11.1f")
fprintf("\n%s  done.\n",string(datetime))

%% Local functions
function q = loadq(datfolder,idx,tag)
S = load(fullfile(datfolder,sprintf('sbdm_salsun_%02d_%s.mat',idx,tag)));
q = S.q;
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
        fprintf("%*s",w,strtrim(sprintf(fmt,M(i,j))));
    end
    fprintf("\n")
end
end
