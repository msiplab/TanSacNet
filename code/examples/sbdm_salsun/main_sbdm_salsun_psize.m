%% main_sbdm_salsun_psize.m
% SA-LSUN pilot arm (SAP) with smaller angle estimators and shorter step 1,
% to reduce overfitting. Same setting as SAP (main_sbdm_salsun_pilot.m):
% observation only, p = N(v), K = 4, step 1 on p, angles from p frozen,
% step 2 (kappa_p, g_p) by SURE on v with 8 fixed probes.
%
% Settings: BottleneckSize b in {0, 40, 18} (0 = Sensei's default, 488,717
% weights; 40 ~ 1/4; 18 ~ 1/16) x step-1 iterations {150, 500} = 6.
% Fair selection: the best setting by PSNR on kodim09 only, reported on
% kodim15/19/23. All settings on all images are printed for information.
%
% Resumable: one cache file per (image, setting):
%   data/sbdm_salsun_psize_<img>_K4_b<b>_it<n>.mat
% Full run:
%   nohup matlab -licmode onlinelicensing -batch "main_sbdm_salsun_psize" \
%       > ~/Documents/SBDMwLSUN/logs/salsun_pilot_size.log 2>&1 &
% One image only (time estimate; writes the same caches):
%   matlab -licmode onlinelicensing -batch "psizeImgs = 9; main_sbdm_salsun_psize"

here = fileparts(mfilename('fullpath'));
cd(fullfile(here,'..','..')); setpath; cd(here)
addpath(fullfile(here,'..','sbdm'))     % helpers (called, not edited)
addpath(fullfile(here,'..','salsun'))   % maskLayer
warning('off','MATLAB:rmpath:DirNotFound')
rmpath(fullfile('/home/shogo/Workspace/GitHub/SaivDr','mexcodes'))
warning('on','MATLAB:rmpath:DirNotFound')

cfg = fcn_ssconfig();
cfg.K = 4;               % K = 4 only
cfg.nProbePilot = 8;     % fixed probes in step 2 (as SAP)
cfg.nProbeFresh = 32;    % fresh probes for the explicit MC check (as SAP)
cfg.bottleneck = [0 40 18];
cfg.nIt1List   = [150 500];
cfg.selImg     = 9;      % selection image
if exist('psizeImgs','var'), cfg.imgIdx = psizeImgs; end
K = cfg.K;
datfolder = fullfile(here,'..','..','..','data');
useGpu = canUseGPU;
[bb,ii] = ndgrid(cfg.bottleneck,cfg.nIt1List);
st.b = bb(:).'; st.it = ii(:).'; nS = numel(st.b);
st.name = compose("b%d/it%d",[st.b(:) st.it(:)]).';
nI = numel(cfg.imgIdx); nPix = prod(cfg.cropSize);
fprintf("%s  start. GPU: %d\n",string(datetime),useGpu)
disp(cfg)

%% Estimator weights per size (exact count from the built networks)
acMask = fcn_ssacmask(prod(cfg.stride),K);
nW = zeros(1,numel(cfg.bottleneck));
for j = 1:numel(cfg.bottleneck)
    b = cfg.bottleneck(j);
    nets = fcn_ssbuildsalsunsz(cfg.cropSize,cfg.stride,cfg.ovlp,acMask,cfg.sigma,b);
    nW(j) = sum(cellfun(@numel,nets.train.Learnables.Value));
    fprintf("BottleneckSize %2d: %d estimator weights (%.4f x pixels, 1/%.2f of b=0)\n",...
        b,nW(j),nW(j)/nPix,nW(1)/nW(j))
end
clear nets

%% Run (or load)
R = cell(nI,1);
for i = 1:nI
    idx = cfg.imgIdx(i);
    [x,v] = fcn_ssloadimage(datfolder,idx,cfg,useGpu);
    rec.idx = idx;
    fprintf("\n%s  kodim%02d  noisy %.2f dB\n",string(datetime),idx,psnr(gather(v),gather(x)))
    rec.N   = loadq(datfolder,sprintf('sbdm_salsun_%02d_N%d.mat',idx,K));
    rec.SAP = loadq(datfolder,sprintf('sbdm_salsun_pilot_%02d_SAP%d.mat',idx,K));
    for s = 1:nS
        f = fullfile(datfolder,sprintf('sbdm_salsun_psize_%02d_K%d_b%d_it%d.mat',idx,K,st.b(s),st.it(s)));
        if isfile(f)
            S = load(f); q = S.q;
            fprintf("  %-9s cached (%s)\n",st.name(s),f)
        else
            t = tic;
            q = fcn_sspsizerunimage(x,v,K,cfg,cfg.seedBase + idx,rec.N,st.b(s),st.it(s));
            q.tTotal = toc(t);
            ftmp = string(f) + ".tmp.mat";
            save(ftmp,'q','-v7.3'); movefile(ftmp,f);
            fprintf("  %-9s %.2f dB  SSIM %.4f  SURE fixed %.4e fresh %.4e MSE %.4e  kept AC v %.4f p %.4f x %.4f  PR %.1e  W %d  (step1 %.1f s, total %.1f s)\n",...
                st.name(s),q.psnr,q.ssim,q.sure,q.sureFresh,q.mse,...
                q.keptAcV,q.keptAcP,q.keptAcX,q.prErrX,q.nEstW,q.t1,q.tTotal)
        end
        rec.Q(s) = q;
    end
    % reproduction check: b0/it500 should equal the SAP cache
    s0 = find(st.b == 0 & st.it == cfg.nIt1Sa,1);
    if ~isempty(s0)
        rec.repPsnr = rec.Q(s0).psnr - rec.SAP.psnr;
        rec.repImg  = double(max(abs(rec.Q(s0).img - rec.SAP.img),[],'all'));
        fprintf("  check b0/it%d vs SAP cache: PSNR diff %.2e dB, max|img diff| %.2e\n",...
            cfg.nIt1Sa,rec.repPsnr,rec.repImg)
    end
    R{i} = rec;
end
R = [R{:}];
save(fullfile(datfolder,'sbdm_salsun_psize_results.mat'),'R','cfg','st','nW')

%% Summary
g = @(f) cell2mat(arrayfun(@(r) arrayfun(@(q) double(q.(f)),r.Q),R(:),'UniformOutput',false));  % nI x nS
pN = arrayfun(@(r) r.N.psnr,R)'; sN = arrayfun(@(r) r.N.ssim,R)';
pSAP = arrayfun(@(r) r.SAP.psnr,R)'; sSAP = arrayfun(@(r) r.SAP.ssim,R)';
P = g('psnr'); Sm = g('ssim');
iTest = find(cfg.imgIdx ~= cfg.selImg);
rows = compose("kodim%02d",cfg.imgIdx(:)); rowsM = rows;
addm = @(M) M;
if nI > 1
    rowsM = [rows; "mean(all)"];
    addm = @(M) [M; mean(M,1)];
    if ~isempty(iTest)
        rowsM = [rowsM; "mean(test)"];
        addm = @(M) [M; mean(M,1); mean(M(iTest,:),1)];
    end
end

fprintf("\n================ Summary (pilot, estimator size) ================\n")
fprintf("Observation only, sigma = 25/255, %dx%d centre crops, 4x4 blocks, ovlp 3x3, S = 1, K = %d\n",cfg.cropSize,K)
fprintf("SAP: p = N(v); step 1 on p (lr %.0e -> %.0e, cosine over the step-1 iterations); angles from p frozen;\n",cfg.lr1Sa)
fprintf("     step 2: %d its, lr %.0e, %d fixed probes; check with %d fresh probes.\n",cfg.nIt2,cfg.lr2,cfg.nProbePilot,cfg.nProbeFresh)
fprintf("Setting bX/itY = BottleneckSize X, Y step-1 iterations. b0/it%d = SAP of main_sbdm_salsun_pilot.m.\n",cfg.nIt1Sa)
fprintf("mean(test) = mean over kodim15, 19, 23 (selection image kodim%02d excluded).\n",cfg.selImg)

fprintf("\nEstimator weights (pixels per image: %d)\n",nPix)
for s = 1:nS
    w = nW(cfg.bottleneck == st.b(s));
    fprintf("  %-9s %7d weights  %.4f per pixel\n",st.name(s),w,w/nPix)
end

% --- Fair selection on kodim09
iSel = find(cfg.imgIdx == cfg.selImg,1);
if ~isempty(iSel)
    [~,sBest] = max(P(iSel,:));
    fprintf("\nSelection on kodim%02d (PSNR [dB]):",cfg.selImg)
    for s = 1:nS, fprintf("  %s %.2f",st.name(s),P(iSel,s)); end
    fprintf("\nSelected setting: %s\n",st.name(sBest))
    if ~isempty(iTest)
        M = [pN(iTest) pSAP(iTest) P(iTest,sBest) P(iTest,sBest)-pN(iTest) P(iTest,sBest)-pSAP(iTest) ...
             sN(iTest) sSAP(iTest) Sm(iTest,sBest) Sm(iTest,sBest)-sN(iTest) Sm(iTest,sBest)-sSAP(iTest)];
        printtable("Selected setting " + st.name(sBest) + " on the test images (PSNR [dB] | SSIM). SAP = b0/it500 (cache of the pilot run)",...
            [rows(iTest); "mean"],["N PSNR" "SAP PSNR" "sel PSNR" "sel-N" "sel-SAP" "N SSIM" "SAP SSIM" "sel SSIM" "sel-N" "sel-SAP"],...
            [M; mean(M,1)],"%10.4f")
    end
end

% --- All settings, all images (information)
printtable("PSNR [dB], all settings",rowsM,["N K=4" st.name],addm([pN P]),"%10.2f")
printtable("SSIM, all settings",rowsM,["N K=4" st.name],addm([sN Sm]),"%10.4f")
printtable("PSNR difference to N K=4 [dB] (SAP - N)",rowsM,st.name,addm(P - pN),"%10.2f")
printtable("SSIM difference to N K=4 (SAP - N)",rowsM,st.name,addm(Sm - sN),"%10.4f")
printtable("Kept AC energy on v (angles from p)",rowsM,st.name,addm(g('keptAcV')),"%10.4f")
printtable("Kept AC energy on p (angles from p)",rowsM,st.name,addm(g('keptAcP')),"%10.4f")
printtable("Kept AC energy on x (angles from p; x = evaluation only)",rowsM,st.name,addm(g('keptAcX')),"%10.4f")
printtable("Kept energy incl. DC on x (angles from p)",rowsM,st.name,addm(g('keptAllX')),"%10.4f")
printtable("Step-2 SURE (8 fixed probes) x1e3",rowsM,st.name,addm(1e3*g('sure')),"%10.4f")
printtable("True MSE x1e3",rowsM,st.name,addm(1e3*g('mse')),"%10.4f")
printtable("1 - SURE/MSE, SURE = 8 fixed probes (> 0: SURE too low)",rowsM,st.name,addm(1 - g('sure')./g('mse')),"%10.4f")
printtable("1 - SURE/MSE, SURE = 32 fresh probes",rowsM,st.name,addm(1 - g('sureFresh')./g('mse')),"%10.4f")
printtable("Step-1 final loss / ||p||^2 (energy not kept)",rowsM,st.name,...
    addm(cell2mat(arrayfun(@(r) arrayfun(@(q) q.hist1(end),r.Q),R(:),'UniformOutput',false))),"%10.2e")
printtable("PR error ||DEx - x||/||x|| (frozen angles), max row = max",[rows; "max"],st.name,...
    [g('prErrX'); max(g('prErrX'),[],1)],"%10.2e")
printtable("Step-1 time [s]",rowsM,st.name,addm(g('t1')),"%10.1f")
printtable("Total time per setting [s] (build + step1 + prep + step2 + eval)",rowsM,st.name,addm(g('tTotal')),"%10.1f")
if isfield(R,'repPsnr')
    printtable("Check: b0/it500 vs SAP cache",rows,["PSNR diff" "max|img|"],[[R.repPsnr]' [R.repImg]'],"%10.2e")
end
fprintf("\n%s  done.\n",string(datetime))

%% Local functions
function q = loadq(datfolder,name)
S = load(fullfile(datfolder,name));
q = S.q;
end

function printtable(ttl,rows,cols,M,fmt)
fprintf("\n%s\n",ttl)
w = max(10,max(strlength(cols))+1);
fprintf("%-11s","");
for j = 1:numel(cols), fprintf("%*s",w,cols(j)); end
fprintf("\n")
for i = 1:numel(rows)
    fprintf("%-11s",rows(i));
    for j = 1:size(M,2)
        fprintf("%*s",w,strtrim(sprintf(fmt,M(i,j))));
    end
    fprintf("\n")
end
end
