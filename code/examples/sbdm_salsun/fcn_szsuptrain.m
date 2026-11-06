function [mdl,tr] = fcn_szsuptrain(kind,trainSet,o)
%FCN_SZSUPTRAIN fcn_dcsuptrain + fixed CPU random state after the build
%
%   [mdl,tr] = fcn_szsuptrain(kind,trainSet,o)
%   EXACT COPY of fcn_dcsuptrain.m (2026-11-06) with one added option:
%     o.rngAfterBuild (optional): a CPU rng state struct (from rng). If
%     given, rng(o.rngAfterBuild) is called right after the network is
%     built and before the first mini-batch.
%   Why: the weight init of the estimator draws from the CPU stream, and
%   the number of draws depends on the BottleneckSize. fcn_minibatch picks
%   the crops with randi on the same CPU stream. With o.rngAfterBuild =
%   the state after the b = 18 build, a net with another b gets the same
%   crops in every iteration as the stage B b = 18 run. The noise (randn
%   on the GPU, gpurng(seed)) is not used by the build, so it is the same
%   anyway. Without o.rngAfterBuild this function = fcn_dcsuptrain.
%   (Used by main_sbdm_salsun_supsize.m.)
%
%   Original help of fcn_dcsuptrain:
%   [mdl,tr] = fcn_dcsuptrain(kind,trainSet,o)
%   kind "N"  : uniform LSUN (shared angles) + shrinkage, f = D S E
%   kind "SA" : SA-LSUN (Whole, ThetaMode 'Reuse'; the angle estimator sees
%               the input v itself) + shrinkage
%   Loss: MSE ||f(v) - x||^2 on mini-batches of o.mb clean patches from
%   trainSet (cell, 64x64 patches of kodim01-08), v = x + sigma*noise,
%   fresh noise every iteration, fixed sigma (examples/sbdm/fcn_minibatch).
%   Adam, cosine decay of the learning rate lr0 -> lr1:
%     N  : angles and shrinkage o.lrN = [lr0 lr1]
%     SA : estimator weights o.lrEst, shrinkage layers o.lrShr
%   Initial shrinkage: AC kappa0 0.9, tau0 2; DC (o.dcShrink) kappa0 0.01,
%   tau0 2 ("id", almost no DC shrinkage at the start). Zero initial
%   angles (N); estimator output weights are zero at the start (SA), so
%   both start from the same transform.
%   o: sigma, leak, dcShrink, nIters, mb, lrN, lrEst, lrShr, seed,
%      stride, ovlp, bottleneck, patch, ckptFile, ckptEvery
%   Checkpoint (parameters, Adam moments, iteration, history, rng
%   states) every o.ckptEvery its; a run resumes from it.
%   mdl.kind, mdl.th / mdl.ab (N), mdl.learn (SA: Learnables table)
%   tr.hist (loss per it), tr.tTrain, tr.tPerIt, tr.nParams

useGpu = isa(trainSet{1},'gpuArray');
sigma = single(o.sigma);
rng(o.seed,'twister');
if useGpu, gpurng(o.seed); end
nAc = prod(o.stride) - 1;
switch kind
    case "N"
        [anet,snet,info] = fcn_dclsunpair(o.patch,o.stride,o.ovlp,o.leak);
        prm0 = fcn_lsuninitprm(info,1,false);
        P.th = prm0.th{1};
        if o.dcShrink, dcInit = "id"; else, dcInit = "none"; end
        P.ab = fcn_dcinitab(nAc,dcInit,zeros(1,'single'));
        if useGpu, P = dlupdate(@gpuArray,P); end
        st = struct('P',P,'avgG',[],'avgSqG',[]);
        tr.nParams = sum(cellfun(@numel,P.th)) + numel(P.ab.a) + numel(P.ab.b) + 2*o.dcShrink;
    case "SA"
        ob = struct('estimator',"net",'leak',o.leak,'bottleneck',o.bottleneck,...
            'shrink',true,'dcShrink',o.dcShrink,'dcInit',"id",'sigma',sigma);
        nets = fcn_dcbuildsalsun(o.patch,o.stride,o.ovlp,ob);
        net = nets.shrink;
        if useGpu, net = dlupdate(@gpuArray,net); end
        isS = startsWith(net.Learnables.Layer,'Lv1_Shrink');
        st = struct('net',net,'avgE',[],'sqE',[],'avgS',[],'sqS',[]);
        tr.nParams = sum(cellfun(@numel,net.Learnables.Value));
    otherwise
        error("unknown kind %s",kind)
end
if isfield(o,'rngAfterBuild') && ~isempty(o.rngAfterBuild)   % (added)
    rng(o.rngAfterBuild);
end
it0 = 0; hist = zeros(o.nIters,1); tPrev = 0;
if ~isempty(o.ckptFile) && isfile(o.ckptFile)
    C = load(o.ckptFile);
    st = C.st; it0 = C.it0; hist(1:it0) = C.hist(1:it0); tPrev = C.tTrain;
    rng(C.rngCpu);
    if useGpu, gpurng(C.rngGpu); end
    if useGpu
        for f2 = string(fieldnames(st)).'
            if ~isempty(st.(f2)), st.(f2) = dlupdate(@gpuArray,st.(f2)); end
        end
    end
    fprintf("    resuming at iteration %d of %d\n",it0,o.nIters);
end
lrAt = @(lr,it) lr(2) + (lr(1) - lr(2))*(1 + cos(pi*it/o.nIters))/2;
t = tic;
for it = it0+1:o.nIters
    [xb,vb] = fcn_minibatch(trainSet,sigma,false,o.mb);
    if kind == "N"
        [loss,grads] = dlfeval(@lossN,anet,snet,info,st.P,vb,xb,sigma);
        [st.P,st.avgG,st.avgSqG] = adamupdate(st.P,grads,st.avgG,st.avgSqG,it,lrAt(o.lrN,it));
    else
        [loss,G] = dlfeval(@lossSA,st.net,vb,xb);
        L = st.net.Learnables;
        [LE,st.avgE,st.sqE] = adamupdate(L(~isS,:),G(~isS,:),st.avgE,st.sqE,it,lrAt(o.lrEst,it));
        [LS,st.avgS,st.sqS] = adamupdate(L(isS,:),G(isS,:),st.avgS,st.sqS,it,lrAt(o.lrShr,it));
        L.Value(~isS) = LE.Value; L.Value(isS) = LS.Value;
        st.net.Learnables = L;
    end
    hist(it) = double(gather(extractdata(loss)));
    if mod(it,max(1,round(o.nIters/10))) == 0
        fprintf("    iter %5d/%d  loss %.4e  (%.1f s)\n",it,o.nIters,hist(it),tPrev + toc(t));
    end
    if ~isempty(o.ckptFile) && (mod(it,o.ckptEvery) == 0) && it < o.nIters
        savectx(o.ckptFile,st,it,hist,tPrev + toc(t),useGpu,kind);
    end
end
if useGpu, wait(gpuDevice); end
tr.tTrain = tPrev + toc(t);
tr.tPerIt = tr.tTrain/max(1,o.nIters);
tr.hist = hist;
mdl.kind = kind; mdl.leak = o.leak; mdl.dcShrink = o.dcShrink; mdl.sigma = sigma;
if kind == "N"
    mdl.th = dlupdate(@gather,st.P.th);
    mdl.ab = dlupdate(@gather,st.P.ab);
else
    mdl.learn = dlupdate(@gather,st.net.Learnables);
end
end

function savectx(f,st,it0,hist,tTrain,useGpu,kind) %#ok<INUSD>
if kind == "N", st.P = dlupdate(@gather,st.P); else, st.net = dlupdate(@gather,st.net); end
st.avgG = gat(st,'avgG'); st.avgSqG = gat(st,'avgSqG');
for f2 = ["avgE" "sqE" "avgS" "sqS"], st.(f2) = gat(st,f2); end
rngCpu = rng; rngGpu = []; %#ok<NASGU>
if useGpu, rngGpu = gpurng; end %#ok<NASGU>
ftmp = string(f) + ".tmp.mat";
save(ftmp,'st','it0','hist','tTrain','rngCpu','rngGpu','-v7.3'); movefile(ftmp,f);
end

function y = gat(st,f)
y = [];
if ~isfield(st,f) || isempty(st.(f)), return, end
y = dlupdate(@gather,st.(f));
end

function [loss,grads] = lossN(anet,snet,info,P,v,x,sigma)
xh = fcn_dcdenoise(anet,snet,info,P.th,P.ab,v,sigma);
loss = mean((x - xh).^2,'all');
grads = dlgradient(loss,P);
end

function [loss,G] = lossSA(net,v,x)
xh = forward(net,v);
loss = mean((x - xh).^2,'all');
G = dlgradient(loss,net.Learnables);
end
