%% main_sbdm_salsun_s10hard.m
% Two experiments, observation only, crops kodim09/15/19/23, same noise
% seeds as before (v = x + sigma*n, n from rng(image index)).
%
% Part 1: lower noise, sigma = 10/255. Same pipeline, only sigma changes.
%   W   : wiener2 5x5, known noise variance
%   N   : uniform LSUN K=4, zonal step 1 on v, step 2 by SURE on v
%         (exact frozen-transform shortcut, fcn_ssstep2lsunfast)
%   SAP : b = 18, pilot p = N(v) (the sigma = 10 N), step 1 (zonal) on p
%         with 150 or 500 its, angles from p frozen, step 2 by SURE on v
%         with 8 fixed probes (fcn_sspsrcrunimage, pilot "N", unchanged)
%   Fair selection: SAP it150 vs it500 by PSNR on kodim09 only.
%
% Part 2: hard thresholding in step 1, sigma = 25/255.
%   Step-1 loss: maximise the energy of the AC coefficients with
%   |c| > tau*sigma (mask without gradient, recomputed each iteration,
%   DC always kept). tau = 2, 3.
%   N-hard   : shared angles, step 1 on v (hard), step 2 by SURE on v
%   SAP-hard : b = 18, step 1 (hard) on p = N(v) (zonal N K=4 cache),
%              150 its, angles from p frozen, step 2 by SURE on v, 8 probes
%   Compared with the zonal N K=4, SAP b18/it150 and SAP (b0/it500) caches.
%   Fair selection: tau by PSNR on kodim09 only, for N-hard and SAP-hard.
%
% Resumable caches (data/):
%   sbdm_salsun_s10_<img>_W.mat, sbdm_salsun_s10_<img>_N4.mat,
%   sbdm_salsun_s10_<img>_K4_b18_it<n>.mat,
%   sbdm_salsun_hard_<img>_N4_t<tau>.mat,
%   sbdm_salsun_hard_<img>_K4_b18_it150_t<tau>.mat
% Full run:
%   nohup matlab -licmode onlinelicensing -batch "main_sbdm_salsun_s10hard" \
%       > ~/Documents/SBDMwLSUN/logs/salsun_sigma10_hard.log 2>&1 &
% One image (time estimate; writes the same caches) with the checks:
%   matlab -licmode onlinelicensing -batch "shImgs = 9; shCheck = true; main_sbdm_salsun_s10hard"

here = fileparts(mfilename('fullpath'));
cd(fullfile(here,'..','..')); setpath; cd(here)
addpath(fullfile(here,'..','sbdm'))     % helpers (called, not edited)
addpath(fullfile(here,'..','salsun'))   % maskLayer
warning('off','MATLAB:rmpath:DirNotFound')
rmpath(fullfile('/home/shogo/Workspace/GitHub/SaivDr','mexcodes'))
warning('on','MATLAB:rmpath:DirNotFound')

cfg0 = fcn_ssconfig();
cfg0.K = 4;
cfg0.nProbePilot = 8;     % fixed probes in step 2 (as SAP)
cfg0.nProbeFresh = 32;    % fresh probes for the explicit MC check (as SAP)
cfg0.bottleneck = 18;
cfg0.selImg = 9;
cfg0.tauList = [2 3];
if exist('shImgs','var'), cfg0.imgIdx = shImgs; end
doCheck = exist('shCheck','var') && shCheck;
K = cfg0.K; b = cfg0.bottleneck;
datfolder = fullfile(here,'..','..','..','data');
useGpu = canUseGPU;
nI = numel(cfg0.imgIdx);
rows = compose("kodim%02d",cfg0.imgIdx(:));
iSel = find(cfg0.imgIdx == cfg0.selImg,1);
iTest = find(cfg0.imgIdx ~= cfg0.selImg);
[rowsM,addm] = meanrows(rows,nI,iTest);
fprintf("%s  start. GPU: %d\n",string(datetime),useGpu)
disp(cfg0)

%% Checks (kodim09, sigma = 25/255)
if doCheck
    cfg = cfg0; idx = 9; seed = cfg.seedBase + idx;
    [x,v] = fcn_ssloadimage(datfolder,idx,cfg,useGpu);
    X = dlarray(v,'SSCB'); sigma = cfg.sigma;
    qN = loadq(datfolder,sprintf('sbdm_salsun_%02d_N%d.mat',idx,K));
    [anet,snet,info] = fcn_lsunpair(size(v),cfg.stride,cfg.ovlp);
    prmC = dlupdate(@gpuArrayIf,qN.prm);
    prmI = fcn_lsuninitprm(info,1,false); prmI = dlupdate(@gpuArrayIf,prmI);
    prm0 = prmC; prm0.a = prmI.a; prm0.b = prmI.b;     % step-1 angles, initial kappa, g
    [anet,snet] = fcn_lsunsetangles(anet,snet,info,prmC.th{1});
    [acC,~] = predict(anet,X); acC = extractdata(acC);
    fprintf("\n[check 1] fast SURE vs fcn_lsunsure exact (kodim09, sigma 25/255, N K=4 cache angles)\n")
    for pc = {prm0,prmC; "initial a,b","cached a,b"}
        p_ = pc{1}; ab.a = p_.a{1}; ab.b = p_.b{1};
        [s1,g1] = dlfeval(@gradfast,ab,acC,sigma,numel(v));
        [s2,g2] = dlfeval(@gradorig,ab,anet,snet,info,p_.th,X,sigma);
        s1 = double(gather(extractdata(s1))); s2 = double(gather(extractdata(s2)));
        ga = [extractdata(g1.a); extractdata(g1.b)]; gb = [extractdata(g2.a); extractdata(g2.b)];
        fprintf("  %s: SURE fast %.8e orig %.8e rel diff %.2e; grad rel diff %.2e\n",pc{2},...
            s1,s2,abs(s1-s2)/abs(s2),double(gather(norm(ga-gb)/norm(gb))))
    end
    t = tic; prmA = fcn_ssstep2lsun(anet,snet,info,prm0,v,sigma,cfg.nIt2,cfg.lr2); tA = toc(t);
    t = tic; prmB = fcn_ssstep2lsunfast(anet,snet,info,prm0,v,sigma,cfg.nIt2,cfg.lr2); tB = toc(t);
    [sA,fA] = fcn_lsunsure(anet,snet,info,prmA,X,sigma,"exact");
    [sB,fB] = fcn_lsunsure(anet,snet,info,prmB,X,sigma,"exact");
    fA = gather(extractdata(fA)); fB = gather(extractdata(fB)); xg = gather(x);
    fprintf("[check 2] step 2 from the cached step-1 angles (400 its)\n")
    fprintf("  original fcn_ssstep2lsun : PSNR %.6f dB (cache %.6f), max|img - cache| %.2e, SURE %.6e, %.1f s\n",...
        psnr(fA,xg),qN.psnr,max(abs(fA - qN.img),[],'all'),gather(extractdata(sA)),tA)
    fprintf("  fast fcn_ssstep2lsunfast : PSNR %.6f dB (cache %.6f), max|img - cache| %.2e, SURE %.6e, %.1f s\n",...
        psnr(fB,xg),qN.psnr,max(abs(fB - qN.img),[],'all'),gather(extractdata(sB)),tB)
    fprintf("  fast - original: PSNR diff %.2e dB, max|img diff| %.2e, max|a diff| %.2e, max|b diff| %.2e\n",...
        psnr(fB,xg) - psnr(fA,xg),max(abs(fB - fA),[],'all'),...
        gather(max(abs(extractdata(prmB.a{1} - prmA.a{1})))),gather(max(abs(extractdata(prmB.b{1} - prmA.b{1})))))
    % initial estimator weights: all-ones mask vs zonal mask
    acMask = fcn_ssacmask(prod(cfg.stride),K);
    rng(seed); n1 = fcn_ssbuildsalsunsz(size(v),cfg.stride,cfg.ovlp,acMask,sigma,b);
    rng(seed); n2 = fcn_ssbuildsalsunsz(size(v),cfg.stride,cfg.ovlp,ones(size(acMask),'like',acMask),sigma,b);
    dW = max(cellfun(@(a,c) double(max(abs(extractdata(a) - extractdata(c)),[],'all')),...
        n1.train.Learnables.Value,n2.train.Learnables.Value));
    fprintf("[check 3] initial estimator weights, all-ones mask vs zonal mask: max|diff| %.2e (%d tensors)\n",...
        dW,height(n1.train.Learnables))
    clear n1 n2
    qZ = loadq(datfolder,sprintf('sbdm_salsun_psize_%02d_K%d_b%d_it150.mat',idx,K,b));
    t = tic; q0 = fcn_ssphardrunimage(x,v,K,cfg,seed,qN,b,150,"zonal",NaN); t0 = toc(t);
    fprintf("[check 4] fcn_ssphardrunimage zonal vs size-sweep cache b18/it150: PSNR diff %.2e dB, max|img diff| %.2e (%.1f s)\n",...
        q0.psnr - qZ.psnr,double(max(abs(q0.img - qZ.img),[],'all')),t0)
    clear q0 qZ prmA prmB anet snet
end

%% ===================== Part 1: sigma = 10/255 =====================
cfg1 = cfg0; cfg1.sigma = single(10/255); cfg1.nIt1List = [150 500];
nS1 = numel(cfg1.nIt1List); nm1 = "SAP/it" + string(cfg1.nIt1List);
fprintf("\n%s  ===== Part 1: sigma = 10/255 =====\n",string(datetime))
R1 = cell(nI,1);
for i = 1:nI
    idx = cfg1.imgIdx(i); seed = cfg1.seedBase + idx;
    [x,v] = fcn_ssloadimage(datfolder,idx,cfg1,useGpu);
    rec = struct('idx',idx);
    rec.noisyPsnr = psnr(gather(v),gather(x)); rec.noisySsim = ssim(gather(v),gather(x));
    fprintf("\n%s  kodim%02d  sigma 10/255  noisy %.2f dB\n",string(datetime),idx,rec.noisyPsnr)
    % W
    f = fullfile(datfolder,sprintf('sbdm_salsun_s10_%02d_W.mat',idx));
    if isfile(f), rec.W = loadf(f,"W");
    else
        t = tic;
        z = single(wiener2(gather(double(v)),[5 5],double(cfg1.sigma)^2));
        q = struct('psnr',psnr(z,gather(x)),'ssim',ssim(z,gather(x)),...
            'mse',double(mean((z - gather(x)).^2,'all')),'img',z,'time',toc(t));
        savecache(f,q); rec.W = q;
        fprintf("  W          %.2f dB  SSIM %.4f\n",q.psnr,q.ssim)
    end
    % N K=4 (zonal)
    f = fullfile(datfolder,sprintf('sbdm_salsun_s10_%02d_N%d.mat',idx,K));
    if isfile(f), rec.N = loadf(f,"N K=4");
    else
        q = fcn_ssnrunimage(x,v,cfg1,seed,K,"zonal",NaN);
        savecache(f,q); rec.N = q;
        printn("N K=4",q)
    end
    % SAP
    rec.S = cell(1,nS1);
    for s = 1:nS1
        f = fullfile(datfolder,sprintf('sbdm_salsun_s10_%02d_K%d_b%d_it%d.mat',idx,K,b,cfg1.nIt1List(s)));
        if isfile(f), q = loadf(f,nm1(s));
        else
            t = tic;
            q = fcn_sspsrcrunimage(x,v,K,cfg1,seed,rec.N,rec.W,b,cfg1.nIt1List(s),"N");
            q.tTotal = toc(t);
            savecache(f,q);
            printsa(nm1(s),q)
        end
        rec.S{s} = q;
    end
    % sigma = 25/255 references (caches; for comparison only)
    rec.N25 = loadq(datfolder,sprintf('sbdm_salsun_%02d_N%d.mat',idx,K));
    rec.W25 = loadq(datfolder,sprintf('sbdm_salsun_%02d_W.mat',idx));
    rec.S25 = arrayfun(@(it) loadq(datfolder,sprintf('sbdm_salsun_psize_%02d_K%d_b%d_it%d.mat',idx,K,b,it)),...
        cfg1.nIt1List,'UniformOutput',false);
    R1{i} = rec;
end
R1 = [R1{:}];

%% Part 1 summary
c1 = @(fn) arrayfun(fn,R1(:));
g1 = @(f) cell2mat(arrayfun(@(r) cellfun(@(q) double(q.(f)),r.S),R1(:),'UniformOutput',false));
pW = c1(@(r) r.W.psnr); sW = c1(@(r) r.W.ssim); pN = c1(@(r) r.N.psnr); sN = c1(@(r) r.N.ssim);
P = g1('psnr'); Sm = g1('ssim');
fprintf("\n================ Summary Part 1 (sigma = 10/255) ================\n")
fprintf("Observation only, %dx%d centre crops, 4x4 blocks, ovlp 3x3, S = 1, K = %d, same noise seeds as before.\n",cfg1.cropSize,K)
fprintf("N: zonal step 1 (%d its, lr %.0e -> %.0e), step 2 by exact SURE (frozen-transform shortcut).\n",cfg1.nIt1N,cfg1.lr1N)
fprintf("SAP: BottleneckSize %d, p = N(v), step 1 on p (lr %.0e -> %.0e), angles frozen, step 2: %d its, lr %.0e, %d fixed probes; check %d fresh probes.\n",...
    b,cfg1.lr1Sa,cfg1.nIt2,cfg1.lr2,cfg1.nProbePilot,cfg1.nProbeFresh)
fprintf("mean(test) = mean over the images except the selection image kodim%02d.\n",cfg1.selImg)
if ~isempty(iSel)
    [~,sB] = max(P(iSel,:));
    fprintf("\nSelection on kodim%02d (PSNR [dB]):",cfg1.selImg)
    for s = 1:nS1, fprintf("  %s %.2f",nm1(s),P(iSel,s)); end
    fprintf("\nSelected setting: %s\n",nm1(sB))
    if ~isempty(iTest)
        it_ = iTest;
        M = [pW(it_) pN(it_) P(it_,sB) P(it_,sB)-pN(it_) pN(it_)-pW(it_)];
        printtable("Part 1: selected " + nm1(sB) + " on the test images, PSNR [dB]",[rows(it_); "mean"],...
            ["W" "N K=4" "SAP sel" "SA-N" "N-W"],[M; mean(M,1)],"%10.4f")
        M = [sW(it_) sN(it_) Sm(it_,sB) Sm(it_,sB)-sN(it_) sN(it_)-sW(it_)];
        printtable("Part 1: selected " + nm1(sB) + " on the test images, SSIM",[rows(it_); "mean"],...
            ["W" "N K=4" "SAP sel" "SA-N" "N-W"],[M; mean(M,1)],"%10.4f")
    end
end
printtable("Part 1: PSNR [dB], all settings",rowsM,["noisy" "W" "N K=4" nm1],addm([c1(@(r) r.noisyPsnr) pW pN P]),"%10.2f")
printtable("Part 1: SSIM, all settings",rowsM,["noisy" "W" "N K=4" nm1],addm([c1(@(r) r.noisySsim) sW sN Sm]),"%10.4f")
printtable("Part 1: PSNR differences [dB] (SA - N, N - W)",rowsM,[nm1+"-N" "N-W"],addm([P-pN pN-pW]),"%10.2f")
printtable("Part 1: SSIM differences (SA - N, N - W)",rowsM,[nm1+"-N" "N-W"],addm([Sm-sN sN-sW]),"%10.4f")
P25 = cell2mat(arrayfun(@(r) cellfun(@(q) q.psnr,r.S25),R1(:),'UniformOutput',false));
pN25 = c1(@(r) r.N25.psnr); pW25 = c1(@(r) r.W25.psnr);
printtable("For comparison, sigma = 25/255 (caches: N K=4, W, size sweep b18): PSNR differences [dB]",rowsM,...
    ["b18/it" + string(cfg1.nIt1List) + "-N" "N-W"],addm([P25-pN25 pN25-pW25]),"%10.2f")
printtable("Part 1: kept AC energy on v (angles from p for SAP)",rowsM,["N K=4" nm1],addm([c1(@(r) r.N.keptAc) g1('keptAcV')]),"%10.4f")
printtable("Part 1: kept AC energy on p",rowsM,nm1,addm(g1('keptAcP')),"%10.4f")
printtable("Part 1: kept AC energy on x (evaluation only)",rowsM,["N K=4" nm1],addm([c1(@(r) r.N.keptAcX) g1('keptAcX')]),"%10.4f")
printtable("Part 1: step-1 final loss / ||input||^2 (N: on v, SAP: on p)",rowsM,["N K=4" nm1],...
    addm([c1(@(r) r.N.hist1(end)) cell2mat(arrayfun(@(r) cellfun(@(q) q.hist1(end),r.S),R1(:),'UniformOutput',false))]),"%10.2e")
printtable("Part 1: SURE x1e3 (N: exact; SAP: 8 fixed probes)",rowsM,["N K=4" nm1],addm(1e3*[c1(@(r) r.N.sure) g1('sure')]),"%10.4f")
printtable("Part 1: true MSE x1e3",rowsM,["W" "N K=4" nm1],addm(1e3*[c1(@(r) r.W.mse) c1(@(r) r.N.mse) g1('mse')]),"%10.4f")
printtable("Part 1: 1 - SURE/MSE (> 0: SURE too low). N: exact; SAP: 8 fixed probes",rowsM,["N K=4" nm1],...
    addm(1 - [c1(@(r) r.N.sure/r.N.mse) g1('sure')./g1('mse')]),"%10.4f")
printtable("Part 1: 1 - SURE/MSE, SAP: 32 fresh probes",rowsM,nm1,addm(1 - g1('sureFresh')./g1('mse')),"%10.4f")
printtable("Part 1: PR error ||DEx - x||/||x||, max row = max",[rows; "max"],["N K=4" nm1],...
    [[c1(@(r) r.N.prErrX) g1('prErrX')]; max([c1(@(r) r.N.prErrX) g1('prErrX')],[],1)],"%10.2e")
printtable("Part 1 checks: max|N(v) - N|, max|W(v) - W| (pilot parts vs the sigma = 10 N, W results), const-angle check",rows,...
    [nm1+" N" nm1+" W" nm1+" const"],[g1('pDiffCacheN') g1('pDiffCacheW') g1('constCheck')],"%10.2e")
printtable("Part 1: time [s]",rowsM,["N step1" "N step2" "N total" nm1+" step1" nm1+" total"],...
    addm([c1(@(r) r.N.t1) c1(@(r) r.N.t2) c1(@(r) r.N.tTotal) g1('t1') g1('tTotal')]),"%10.1f")
if nI == 4
    save(fullfile(datfolder,'sbdm_salsun_s10_results.mat'),'R1','cfg1')
end

%% ===================== Part 2: hard thresholding, sigma = 25/255 =====================
cfg2 = cfg0; nT = numel(cfg2.tauList);
nmNh = "Nh/t" + string(cfg2.tauList); nmSh = "SAPh/t" + string(cfg2.tauList);
fprintf("\n%s  ===== Part 2: hard thresholding, sigma = 25/255 =====\n",string(datetime))
R2 = cell(nI,1);
for i = 1:nI
    idx = cfg2.imgIdx(i); seed = cfg2.seedBase + idx;
    [x,v] = fcn_ssloadimage(datfolder,idx,cfg2,useGpu);
    rec = struct('idx',idx);
    fprintf("\n%s  kodim%02d  sigma 25/255  noisy %.2f dB\n",string(datetime),idx,psnr(gather(v),gather(x)))
    rec.W = loadq(datfolder,sprintf('sbdm_salsun_%02d_W.mat',idx));
    rec.N = loadq(datfolder,sprintf('sbdm_salsun_%02d_N%d.mat',idx,K));
    rec.Z = loadq(datfolder,sprintf('sbdm_salsun_psize_%02d_K%d_b%d_it150.mat',idx,K,b));   % SAP zonal b18/it150
    rec.A = loadq(datfolder,sprintf('sbdm_salsun_psize_%02d_K%d_b0_it500.mat',idx,K));      % SAP (= pilot-run cache)
    [rec.NstatV,rec.NstatX] = nstats(rec.N.prm,x,v,cfg2);
    rec.Nh = cell(1,nT); rec.Sh = cell(1,nT);
    for j = 1:nT
        tau = cfg2.tauList(j);
        f = fullfile(datfolder,sprintf('sbdm_salsun_hard_%02d_N%d_t%d.mat',idx,K,tau));
        if isfile(f), q = loadf(f,nmNh(j));
        else
            q = fcn_ssnrunimage(x,v,cfg2,seed,K,"hard",tau);
            savecache(f,q); printn(nmNh(j),q)
        end
        rec.Nh{j} = q;
    end
    for j = 1:nT
        tau = cfg2.tauList(j);
        f = fullfile(datfolder,sprintf('sbdm_salsun_hard_%02d_K%d_b%d_it150_t%d.mat',idx,K,b,tau));
        if isfile(f), q = loadf(f,nmSh(j));
        else
            t = tic;
            q = fcn_ssphardrunimage(x,v,K,cfg2,seed,rec.N,b,150,"hard",tau);
            q.tTotal = toc(t);
            savecache(f,q); printsa(nmSh(j),q)
        end
        rec.Sh{j} = q;
    end
    R2{i} = rec;
end
R2 = [R2{:}];
if nI == 4
    save(fullfile(datfolder,'sbdm_salsun_hard_results.mat'),'R2','cfg2')
end

%% Part 2 summary
c2 = @(fn) arrayfun(fn,R2(:));
gh = @(fld,fn) cell2mat(arrayfun(@(r) cellfun(fn,r.(fld)),R2(:),'UniformOutput',false));  % nI x nT
jt = @(q) find(cfg2.tauList == q.tau,1);          % index of the setting's own tau
pW = c2(@(r) r.W.psnr); sW = c2(@(r) r.W.ssim); pN = c2(@(r) r.N.psnr); sN = c2(@(r) r.N.ssim);
pZ = c2(@(r) r.Z.psnr); sZ = c2(@(r) r.Z.ssim); pA = c2(@(r) r.A.psnr); sA = c2(@(r) r.A.ssim);
PNh = gh('Nh',@(q) q.psnr); SNh = gh('Nh',@(q) q.ssim);
PSh = gh('Sh',@(q) q.psnr); SSh = gh('Sh',@(q) q.ssim);
refN = ["W" "N K=4" "SAPz b18/it150" "SAP b0/it500"];
fprintf("\n================ Summary Part 2 (hard thresholding, sigma = 25/255) ================\n")
fprintf("Step-1 loss: energy of the AC coefficients with |c| > tau*sigma (mask without gradient, recomputed each iteration, DC kept).\n")
fprintf("Nh/tT  : shared angles, hard step 1 on v (%d its, lr %.0e -> %.0e), step 2 exact SURE (shortcut).\n",cfg2.nIt1N,cfg2.lr1N)
fprintf("SAPh/tT: BottleneckSize %d, hard step 1 on p = N(v) (zonal N K=4 cache), 150 its, angles frozen, step 2 with %d fixed probes.\n",b,cfg2.nProbePilot)
fprintf("References (caches): N K=4 zonal; SAPz b18/it150 = SAP zonal with the same size and its; SAP b0/it500 = pilot-run SAP.\n")
if ~isempty(iSel)
    [~,jN] = max(PNh(iSel,:)); [~,jS] = max(PSh(iSel,:));
    fprintf("\nSelection on kodim%02d (PSNR [dB]):",cfg2.selImg)
    for j = 1:nT, fprintf("  %s %.2f",nmNh(j),PNh(iSel,j)); end
    for j = 1:nT, fprintf("  %s %.2f",nmSh(j),PSh(iSel,j)); end
    fprintf("   (refs: N K=4 %.2f, SAPz %.2f, SAP %.2f, W %.2f)",pN(iSel),pZ(iSel),pA(iSel),pW(iSel))
    fprintf("\nSelected: %s and %s\n",nmNh(jN),nmSh(jS))
    if ~isempty(iTest)
        it_ = iTest;
        cols = ["W" "N K=4" "SAPz" "Nh sel" "SAPh sel" "Nh-N" "SAPh-N" "SAPh-SAPz" "SAPh-Nh" "Nh-W"];
        M = [pW(it_) pN(it_) pZ(it_) PNh(it_,jN) PSh(it_,jS) PNh(it_,jN)-pN(it_) PSh(it_,jS)-pN(it_) ...
            PSh(it_,jS)-pZ(it_) PSh(it_,jS)-PNh(it_,jN) PNh(it_,jN)-pW(it_)];
        printtable("Part 2: selected " + nmNh(jN) + ", " + nmSh(jS) + " on the test images, PSNR [dB]",[rows(it_); "mean"],cols,[M; mean(M,1)],"%10.4f")
        M = [sW(it_) sN(it_) sZ(it_) SNh(it_,jN) SSh(it_,jS) SNh(it_,jN)-sN(it_) SSh(it_,jS)-sN(it_) ...
            SSh(it_,jS)-sZ(it_) SSh(it_,jS)-SNh(it_,jN) SNh(it_,jN)-sW(it_)];
        printtable("Part 2: selected " + nmNh(jN) + ", " + nmSh(jS) + " on the test images, SSIM",[rows(it_); "mean"],cols,[M; mean(M,1)],"%10.4f")
    end
end
printtable("Part 2: PSNR [dB], all settings",rowsM,[refN nmNh nmSh],addm([pW pN pZ pA PNh PSh]),"%10.2f")
printtable("Part 2: SSIM, all settings",rowsM,[refN nmNh nmSh],addm([sW sN sZ sA SNh SSh]),"%10.4f")
printtable("Part 2: PSNR differences [dB]: SA - N K=4, SAPh - Nh (same tau), N - W",rowsM,...
    ["SAPz-N" "SAP-N" nmNh+"-N" nmSh+"-N" nmSh+"-Nh" "N-W" nmNh+"-W"],...
    addm([pZ-pN pA-pN PNh-pN PSh-pN PSh-PNh pN-pW PNh-pW]),"%10.2f")
printtable("Part 2: SSIM differences: SA - N K=4, SAPh - Nh (same tau), N - W",rowsM,...
    ["SAPz-N" "SAP-N" nmNh+"-N" nmSh+"-N" nmSh+"-Nh" "N-W" nmNh+"-W"],...
    addm([sZ-sN sA-sN SNh-sN SSh-sN SSh-SNh sN-sW SNh-sW]),"%10.4f")
% kept energy, zonal K=4 measure (same as earlier reports)
printtable("Part 2: kept AC energy on v, zonal K=4 measure (SA: angles from p)",rowsM,["N K=4" "SAPz" "SAP" nmNh nmSh],...
    addm([c2(@(r) r.N.keptAc) c2(@(r) r.Z.keptAcV) c2(@(r) r.A.keptAcV) gh('Nh',@(q) q.statV.zonalAc) gh('Sh',@(q) q.statV.zonalAc)]),"%10.4f")
printtable("Part 2: kept AC energy on p, zonal K=4 measure",rowsM,["SAPz" "SAP" nmSh],...
    addm([c2(@(r) r.Z.keptAcP) c2(@(r) r.A.keptAcP) gh('Sh',@(q) q.statP.zonalAc)]),"%10.4f")
printtable("Part 2: kept AC energy on x, zonal K=4 measure (evaluation only)",rowsM,["N K=4" "SAPz" "SAP" nmNh nmSh],...
    addm([c2(@(r) r.N.keptAcX) c2(@(r) r.Z.keptAcX) c2(@(r) r.A.keptAcX) gh('Nh',@(q) q.statX.zonalAc) gh('Sh',@(q) q.statX.zonalAc)]),"%10.4f")
% kept energy, hard measure (each setting at its own tau; N K=4 zonal at both taus)
nmNz = "N K=4@t" + string(cfg2.tauList);
hN = @(fld,what) cell2mat(arrayfun(@(r) r.(fld).(what),R2(:),'UniformOutput',false));   % nI x nT
printtable("Part 2: kept AC energy on v, hard measure (energy of |c| > tau*sigma / AC energy)",rowsM,[nmNz nmNh nmSh],...
    addm([hN('NstatV','hardAc') gh('Nh',@(q) q.statV.hardAc(jt(q))) gh('Sh',@(q) q.statV.hardAc(jt(q)))]),"%10.4f")
printtable("Part 2: kept AC energy on p, hard measure",rowsM,nmSh,addm(gh('Sh',@(q) q.statP.hardAc(jt(q)))),"%10.4f")
printtable("Part 2: kept AC energy on x, hard measure (evaluation only)",rowsM,[nmNz nmNh nmSh],...
    addm([hN('NstatX','hardAc') gh('Nh',@(q) q.statX.hardAc(jt(q))) gh('Sh',@(q) q.statX.hardAc(jt(q)))]),"%10.4f")
printtable("Part 2: fraction of AC coefficients kept by the hard mask, on v",rowsM,[nmNz nmNh nmSh],...
    addm([hN('NstatV','hardFrac') gh('Nh',@(q) q.statV.hardFrac(jt(q))) gh('Sh',@(q) q.statV.hardFrac(jt(q)))]),"%10.4f")
printtable("Part 2: fraction kept by the hard mask, on p",rowsM,nmSh,addm(gh('Sh',@(q) q.statP.hardFrac(jt(q)))),"%10.4f")
printtable("Part 2: fraction kept by the hard mask, on x (evaluation only)",rowsM,[nmNz nmNh nmSh],...
    addm([hN('NstatX','hardFrac') gh('Nh',@(q) q.statX.hardFrac(jt(q))) gh('Sh',@(q) q.statX.hardFrac(jt(q)))]),"%10.4f")
printtable("Part 2: fraction kept by the hard mask in the last step-1 iteration (Nh: on v, SAPh: on p)",rowsM,[nmNh nmSh],...
    addm([gh('Nh',@(q) q.frac1(end)) gh('Sh',@(q) q.frac1(end))]),"%10.4f")
printtable("Part 2: step-1 final loss / ||input||^2 (N, Nh: on v; SA: on p)",rowsM,["N K=4" "SAPz" nmNh nmSh],...
    addm([c2(@(r) r.N.hist1(end)) c2(@(r) r.Z.hist1(end)) gh('Nh',@(q) q.hist1(end)) gh('Sh',@(q) q.hist1(end))]),"%10.2e")
printtable("Part 2: SURE x1e3 (N, Nh: exact; SA: 8 fixed probes)",rowsM,["N K=4" "SAPz" "SAP" nmNh nmSh],...
    addm(1e3*[c2(@(r) r.N.sure) c2(@(r) r.Z.sure) c2(@(r) r.A.sure) gh('Nh',@(q) q.sure) gh('Sh',@(q) q.sure)]),"%10.4f")
printtable("Part 2: true MSE x1e3",rowsM,["N K=4" "SAPz" "SAP" nmNh nmSh],...
    addm(1e3*[c2(@(r) r.N.mse) c2(@(r) r.Z.mse) c2(@(r) r.A.mse) gh('Nh',@(q) q.mse) gh('Sh',@(q) q.mse)]),"%10.4f")
printtable("Part 2: 1 - SURE/MSE (> 0: SURE too low). N, Nh: exact; SA: 8 fixed probes",rowsM,["N K=4" "SAPz" "SAP" nmNh nmSh],...
    addm(1 - [c2(@(r) r.N.sure/r.N.mse) c2(@(r) r.Z.sure/r.Z.mse) c2(@(r) r.A.sure/r.A.mse) ...
    gh('Nh',@(q) q.sure/q.mse) gh('Sh',@(q) q.sure/q.mse)]),"%10.4f")
printtable("Part 2: 1 - SURE/MSE, SA: 32 fresh probes",rowsM,["SAPz" "SAP" nmSh],...
    addm(1 - [c2(@(r) r.Z.sureFresh/r.Z.mse) c2(@(r) r.A.sureFresh/r.A.mse) gh('Sh',@(q) q.sureFresh/q.mse)]),"%10.4f")
M = [gh('Nh',@(q) q.prErrX) gh('Sh',@(q) q.prErrX)];
printtable("Part 2: PR error ||DEx - x||/||x||, max row = max",[rows; "max"],[nmNh nmSh],[M; max(M,[],1)],"%10.2e")
printtable("Part 2 checks: max|N(v) - N cache|, const-angle check (SAPh)",rows,[nmSh+" N" nmSh+" const"],...
    [gh('Sh',@(q) q.pDiffCacheN) gh('Sh',@(q) q.constCheck)],"%10.2e")
printtable("Part 2: time [s]",rowsM,[nmNh+" step1" nmNh+" total" nmSh+" step1" nmSh+" total"],...
    addm([gh('Nh',@(q) q.t1) gh('Nh',@(q) q.tTotal) gh('Sh',@(q) q.t1) gh('Sh',@(q) q.tTotal)]),"%10.1f")
fprintf("\n%s  done.\n",string(datetime))

%% Local functions
function [rowsM,addm] = meanrows(rows,nI,iTest)
rowsM = rows; addm = @(M) M;
if nI > 1
    rowsM = [rows; "mean(all)"]; addm = @(M) [M; mean(M,1)];
    if ~isempty(iTest) && numel(iTest) < nI
        rowsM = [rowsM; "mean(test)"]; addm = @(M) [M; mean(M,1); mean(M(iTest,:),1)];
    end
end
end

function [sV,sX] = nstats(prm,x,v,cfg)
% kept-energy statistics of the cached N (shared angles) on v and x
[anet,snet,info] = fcn_lsunpair(size(v),cfg.stride,cfg.ovlp);
prm = dlupdate(@gpuArrayIf,prm);
anet = fcn_lsunsetangles(anet,snet,info,prm.th{1});
m = reshape(cast(fcn_ssacmask(prod(cfg.stride),cfg.K),'like',v),1,1,[]);
thr = double(cfg.tauList)*double(cfg.sigma);
X = dlarray(v,'SSCB'); [ac,dc] = predict(anet,X); sV = fcn_sskeptstats(ac,dc,X,m,thr);
X = dlarray(x,'SSCB'); [ac,dc] = predict(anet,X); sX = fcn_sskeptstats(ac,dc,X,m,thr);
end

function y = gpuArrayIf(y)
if canUseGPU, y = gpuArray(y); end
end

function [s,g] = gradfast(ab,ac,sigma,N)
s = fcn_ssfastsure(ab,ac,sigma,N);
g = dlgradient(s,ab);
end

function [s,g] = gradorig(ab,anet,snet,info,th,X,sigma)
p.th = th; p.a = {ab.a}; p.b = {ab.b};
s = mean(fcn_lsunsure(anet,snet,info,p,X,sigma,"exact"),'all');
g = dlgradient(s,ab);
end

function printn(name,q)
fprintf("  %-10s %.2f dB  SSIM %.4f  SURE %.4e MSE %.4e  kept AC (zonal) v %.4f x %.4f  frac1 %.4f  PR %.1e  (step1 %.1f s, step2 %.1f s, total %.1f s)\n",...
    name,q.psnr,q.ssim,q.sure,q.mse,q.statV.zonalAc,q.statX.zonalAc,q.frac1(end),q.prErrX,q.t1,q.t2,q.tTotal)
end

function printsa(name,q)
fprintf("  %-10s %.2f dB  SSIM %.4f  pilot %.2f dB  SURE fixed %.4e fresh %.4e MSE %.4e  kept AC (zonal) v %.4f p %.4f x %.4f  PR %.1e  (step1 %.1f s, total %.1f s)\n",...
    name,q.psnr,q.ssim,q.psnrPilot,q.sure,q.sureFresh,q.mse,q.keptAcV,q.keptAcP,q.keptAcX,q.prErrX,q.t1,q.tTotal)
end

function savecache(f,q)
ftmp = string(f) + ".tmp.mat";
save(ftmp,'q','-v7.3'); movefile(ftmp,f);
end

function q = loadf(f,name)
S = load(f); q = S.q;
fprintf("  %-10s cached (%s)\n",name,f)
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
