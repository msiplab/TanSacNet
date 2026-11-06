%% main_sbdm_salsun_psrc.m
% SA-LSUN pilot arm with pilots that do NOT come from LSUN.
% Idea: with p = N(v), step 1 may only learn N's transform again
% (kept AC on x: SA 0.83-0.85 vs N 0.85). Here the pilot is
%   pN  = N(v)              (cached uniform LSUN K=4; reference)
%   pW  = W(v)              (wiener2 5x5, known noise variance, as arm W)
%   pNW = (N(v) + W(v))/2
% all computed from v only. Same setting as the size sweep
% (main_sbdm_salsun_psize.m): observation only, K = 4, BottleneckSize 18,
% step 1 on p, angles from p frozen, step 2 (kappa_p, g_p) by SURE on v
% with 8 fixed probes. Step-1 iterations 150 and 500.
% 3 pilots x 2 iteration counts = 6 settings.
% pN settings = the size-sweep caches b18/it150 and b18/it500 (loaded).
% Fair selection: the best setting by PSNR on kodim09 only, reported on
% kodim15/19/23. All settings on all images are printed for information.
%
% Resumable: one cache file per (image, new setting):
%   data/sbdm_salsun_psrc_<img>_K4_b18_it<n>_p<pilot>.mat
% Full run:
%   nohup matlab -licmode onlinelicensing -batch "main_sbdm_salsun_psrc" \
%       > ~/Documents/SBDMwLSUN/logs/salsun_pilot_source.log 2>&1 &
% One image only (time estimate; writes the same caches), with a check
% that the new code with pilot N reproduces the size-sweep cache:
%   matlab -licmode onlinelicensing -batch "psrcImgs = 9; psrcCheck = true; main_sbdm_salsun_psrc"

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
cfg.bottleneck = 18;     % estimator size (as the size sweep, b = 18)
cfg.nIt1List   = [150 500];
cfg.pilots     = ["N" "W" "NW"];
cfg.selImg     = 9;      % selection image
if exist('psrcImgs','var'), cfg.imgIdx = psrcImgs; end
doCheck = exist('psrcCheck','var') && psrcCheck;
K = cfg.K; b = cfg.bottleneck;
datfolder = fullfile(here,'..','..','..','data');
useGpu = canUseGPU;
[pp,ii] = ndgrid(1:numel(cfg.pilots),cfg.nIt1List);
st.pilot = cfg.pilots(pp(:).'); st.it = ii(:).'; nS = numel(st.it);
st.name = "p" + st.pilot + "/it" + string(st.it);
nI = numel(cfg.imgIdx);
fprintf("%s  start. GPU: %d\n",string(datetime),useGpu)
disp(cfg)

%% Run (or load)
R = cell(nI,1);
for i = 1:nI
    idx = cfg.imgIdx(i);
    [x,v] = fcn_ssloadimage(datfolder,idx,cfg,useGpu);
    rec.idx = idx;
    fprintf("\n%s  kodim%02d  noisy %.2f dB\n",string(datetime),idx,psnr(gather(v),gather(x)))
    rec.N = loadq(datfolder,sprintf('sbdm_salsun_%02d_N%d.mat',idx,K));
    rec.W = loadq(datfolder,sprintf('sbdm_salsun_%02d_W.mat',idx));
    rec.Q = cell(1,nS);
    for s = 1:nS
        if st.pilot(s) == "N"   % size-sweep cache
            f = fullfile(datfolder,sprintf('sbdm_salsun_psize_%02d_K%d_b%d_it%d.mat',idx,K,b,st.it(s)));
        else
            f = fullfile(datfolder,sprintf('sbdm_salsun_psrc_%02d_K%d_b%d_it%d_p%s.mat',idx,K,b,st.it(s),st.pilot(s)));
        end
        if isfile(f)
            S = load(f); q = S.q;
            fprintf("  %-10s cached (%s)\n",st.name(s),f)
        elseif st.pilot(s) == "N"
            error("missing size-sweep cache %s",f)
        else
            q = runone(x,v,K,cfg,idx,rec,b,st.it(s),st.pilot(s));
            ftmp = string(f) + ".tmp.mat";
            save(ftmp,'q','-v7.3'); movefile(ftmp,f);
            fprintf("  %-10s %.2f dB  SSIM %.4f  pilot %.2f dB  SURE fixed %.4e fresh %.4e MSE %.4e  kept AC v %.4f p %.4f x %.4f  PR %.1e  pilot diff N %.1e W %.1e  (step1 %.1f s, total %.1f s)\n",...
                st.name(s),q.psnr,q.ssim,q.psnrPilot,q.sure,q.sureFresh,q.mse,...
                q.keptAcV,q.keptAcP,q.keptAcX,q.prErrX,q.pDiffCacheN,q.pDiffCacheW,q.t1,q.tTotal)
        end
        rec.Q{s} = q;
    end
    % check: the new code with pilot N reproduces the size-sweep cache
    if doCheck
        s0 = find(st.pilot == "N",1);
        q0 = runone(x,v,K,cfg,idx,rec,b,st.it(s0),"N");
        rec.repPsnr = q0.psnr - rec.Q{s0}.psnr;
        rec.repImg  = double(max(abs(q0.img - rec.Q{s0}.img),[],'all'));
        fprintf("  check %s new code vs size-sweep cache: PSNR diff %.2e dB, max|img diff| %.2e (total %.1f s)\n",...
            st.name(s0),rec.repPsnr,rec.repImg,q0.tTotal)
    end
    R{i} = rec;
end
R = [R{:}];
if nI == 4
    save(fullfile(datfolder,'sbdm_salsun_psrc_results.mat'),'R','cfg','st')
end

%% Summary
g = @(f) cell2mat(arrayfun(@(r) cellfun(@(q) double(q.(f)),r.Q),R(:),'UniformOutput',false));  % nI x nS
pN = arrayfun(@(r) r.N.psnr,R)'; sN = arrayfun(@(r) r.N.ssim,R)';
pW = arrayfun(@(r) r.W.psnr,R)'; sW = arrayfun(@(r) r.W.ssim,R)';
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

fprintf("\n================ Summary (pilot source) ================\n")
fprintf("Observation only, sigma = 25/255, %dx%d centre crops, 4x4 blocks, ovlp 3x3, S = 1, K = %d, BottleneckSize %d\n",cfg.cropSize,K,b)
fprintf("Pilots from v only: pN = N(v) (uniform LSUN K=4), pW = wiener2 5x5, pNW = (N(v)+W(v))/2.\n")
fprintf("Step 1 on p (lr %.0e -> %.0e, cosine over the step-1 iterations); angles from p frozen;\n",cfg.lr1Sa)
fprintf("step 2: %d its, lr %.0e, %d fixed probes (pilot recomputed per probe); check with %d fresh probes.\n",cfg.nIt2,cfg.lr2,cfg.nProbePilot,cfg.nProbeFresh)
fprintf("pN/itY = size-sweep setting b%d/itY (cache).\n",b)
fprintf("mean(test) = mean over the images except the selection image kodim%02d.\n",cfg.selImg)

% --- Fair selection on kodim09
iSel = find(cfg.imgIdx == cfg.selImg,1);
if ~isempty(iSel)
    [~,sBest] = max(P(iSel,:));
    fprintf("\nSelection on kodim%02d (PSNR [dB]):",cfg.selImg)
    for s = 1:nS, fprintf("  %s %.2f",st.name(s),P(iSel,s)); end
    fprintf("\nSelected setting: %s\n",st.name(sBest))
    if ~isempty(iTest)
        sRef = find(st.pilot == "N" & st.it == st.it(sBest),1);   % pN with the same iterations
        M = [pN(iTest) pW(iTest) P(iTest,sRef) P(iTest,sBest) P(iTest,sBest)-pN(iTest) P(iTest,sBest)-P(iTest,sRef)];
        printtable("Selected setting " + st.name(sBest) + " on the test images, PSNR [dB]. pN = " + st.name(sRef),...
            [rows(iTest); "mean"],["N K=4" "W" "pN" "sel" "sel-N" "sel-pN"],[M; mean(M,1)],"%10.4f")
        M = [sN(iTest) sW(iTest) Sm(iTest,sRef) Sm(iTest,sBest) Sm(iTest,sBest)-sN(iTest) Sm(iTest,sBest)-Sm(iTest,sRef)];
        printtable("Selected setting " + st.name(sBest) + " on the test images, SSIM. pN = " + st.name(sRef),...
            [rows(iTest); "mean"],["N K=4" "W" "pN" "sel" "sel-N" "sel-pN"],[M; mean(M,1)],"%10.4f")
    end
end

% --- All settings, all images (information)
printtable("Pilot PSNR [dB] (p vs x; evaluation only)",rowsM,st.name,addm(g('psnrPilot')),"%10.2f")
printtable("PSNR [dB], all settings",rowsM,["N K=4" "W" st.name],addm([pN pW P]),"%10.2f")
printtable("SSIM, all settings",rowsM,["N K=4" "W" st.name],addm([sN sW Sm]),"%10.4f")
printtable("PSNR difference to N K=4 [dB] (SA - N)",rowsM,st.name,addm(P - pN),"%10.2f")
printtable("SSIM difference to N K=4 (SA - N)",rowsM,st.name,addm(Sm - sN),"%10.4f")
kNv = arrayfun(@(r) r.N.keptAc,R)'; kNx = arrayfun(@(r) r.N.keptAcX,R)';
printtable("Kept AC energy on v (angles from p)",rowsM,["N K=4" st.name],addm([kNv g('keptAcV')]),"%10.4f")
printtable("Kept AC energy on p (angles from p)",rowsM,st.name,addm(g('keptAcP')),"%10.4f")
printtable("Kept AC energy on x (angles from p; x = evaluation only)",rowsM,["N K=4" st.name],addm([kNx g('keptAcX')]),"%10.4f")
printtable("Kept energy incl. DC on x (angles from p)",rowsM,st.name,addm(g('keptAllX')),"%10.4f")
printtable("Step-2 SURE (8 fixed probes) x1e3",rowsM,st.name,addm(1e3*g('sure')),"%10.4f")
printtable("True MSE x1e3",rowsM,st.name,addm(1e3*g('mse')),"%10.4f")
sNr = arrayfun(@(r) r.N.sure/r.N.mse,R)';
printtable("1 - SURE/MSE, SURE = 8 fixed probes (> 0: SURE too low); N K=4: exact-divergence SURE",...
    rowsM,["N K=4" st.name],addm(1 - [sNr g('sure')./g('mse')]),"%10.4f")
printtable("1 - SURE/MSE, SURE = 32 fresh probes",rowsM,st.name,addm(1 - g('sureFresh')./g('mse')),"%10.4f")
printtable("Step-1 final loss / ||p||^2 (energy not kept)",rowsM,st.name,...
    addm(cell2mat(arrayfun(@(r) cellfun(@(q) q.hist1(end),r.Q),R(:),'UniformOutput',false))),"%10.2e")
printtable("PR error ||DEx - x||/||x|| (frozen angles), max row = max",[rows; "max"],st.name,...
    [g('prErrX'); max(g('prErrX'),[],1)],"%10.2e")
sNew = find(st.pilot ~= "N");
printtable("Check: max|pilot part - cache| (N(v) vs N cache, W(v) vs W cache), new settings",...
    rows,[st.name(sNew)+" N" st.name(sNew)+" W"],[g2(R,sNew,'pDiffCacheN') g2(R,sNew,'pDiffCacheW')],"%10.2e")
printtable("Step-1 time [s]",rowsM,st.name,addm(g('t1')),"%10.1f")
printtable("Total time per setting [s] (build + step1 + prep + step2 + eval)",rowsM,st.name,addm(g('tTotal')),"%10.1f")
if isfield(R,'repPsnr')
    printtable("Check: new code (pilot N) vs size-sweep cache",rows,["PSNR diff" "max|img|"],[[R.repPsnr]' [R.repImg]'],"%10.2e")
end
fprintf("\n%s  done.\n",string(datetime))

%% Local functions
function q = runone(x,v,K,cfg,idx,rec,b,nIt1,pilot)
t = tic;
q = fcn_sspsrcrunimage(x,v,K,cfg,cfg.seedBase + idx,rec.N,rec.W,b,nIt1,pilot);
q.tTotal = toc(t);
end

function M = g2(R,sIdx,f)
M = cell2mat(arrayfun(@(r) cellfun(@(q) double(q.(f)),r.Q(sIdx)),R(:),'UniformOutput',false));
end

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
