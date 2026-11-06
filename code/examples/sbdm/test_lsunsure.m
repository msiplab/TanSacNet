%% Sanity check of the SURE functions (a few minutes, run before any long job)
% Run setpath in the top directory first, then cd to code/examples/sbdm.
% Expected: (1) exact and Monte-Carlo divergence agree to ~1e-3 relative,
%           (2) mean SURE matches the true MSE (which uses the clean x),
%           (3) also for per-block (locally-structured) angles.
clc, clear
rng(0)
datfolder = fullfile('..','..','..','data');
sz = [64 64]; stride = [4 4]; ovlp = [3 3]; sigma = single(25/255);
[anet,snet,info] = fcn_lsunpair(sz,stride,ovlp);
[~,c] = fcn_loadkodak(datfolder,9,sz,1,false); x = single(c{1});

prmU = fcn_lsuninitprm(info,1,false,0.9,2.0);
prmL = fcn_lsunexpandprm(prmU,info);
for k = 1:numel(prmL.th{1})      % perturb per block: really shift variant
    prmL.th{1}{k} = prmL.th{1}{k} + 0.1*randn(size(prmL.th{1}{k}),'single');
end
names = ["uniform","per-block"]; P = {prmU,prmL};
nRep = 200;
for m = 1:2
    s = zeros(nRep,1); e = zeros(nRep,1); dEx = zeros(nRep,1); dMc = zeros(nRep,1);
    for r = 1:nRep
        u = dlarray(x + sigma*randn(sz,'single'),'SSCB');
        [sr,fu,dx] = fcn_lsunsure(anet,snet,info,P{m},u,sigma,"exact");
        [~,~,dm]   = fcn_lsunsure(anet,snet,info,P{m},u,sigma,"mc");
        s(r) = extractdata(sr); dEx(r) = extractdata(dx); dMc(r) = extractdata(dm);
        e(r) = mean((extractdata(fu) - x).^2,'all');
    end
    fprintf("%-9s  div/N exact %.5f  mc %.5f (rel. diff %.1e)\n",names(m),...
        mean(dEx),mean(dMc),abs(mean(dEx)-mean(dMc))/mean(dEx))
    fprintf("%-9s  mean SURE %.4e  true MSE %.4e  (rel. diff %.1e, std of SURE %.1e)\n",...
        names(m),mean(s),mean(e),abs(mean(s)-mean(e))/mean(e),std(s))
end
