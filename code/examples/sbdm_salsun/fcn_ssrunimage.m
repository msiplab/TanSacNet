function r = fcn_ssrunimage(x,v,cfg,seed,arms)
%FCN_SSRUNIMAGE All arms on one image, trained on the observation v only
%
%   r = fcn_ssrunimage(x,v,cfg,seed)
%
%   x : clean image (H x W, single). Used ONLY for evaluation
%       (PSNR, SSIM, true MSE, ||DEx - x||/||x||).
%   v : noisy observation x + sigma*n.
%
%   Arms
%     r.W      : wiener2, 5x5, known noise variance
%     r.N(k)   : uniform LSUN (shared angles), step 1 + step 2,
%                exact SURE divergence (angles fixed)
%     r.SA(k)  : SA-LSUN, ThetaMode 'Reuse', step 1 + step 2,
%                Monte-Carlo divergence of the whole map
%   with k over cfg.K (number of kept AC channels in step 1).
%   Times are in seconds (GPU synchronised).
%   arms (optional, default ["W" "N" "SA"]) selects the arms to run;
%   each arm reseeds with rng(seed), so arms run alone give the same result.

if nargin < 5, arms = ["W" "N" "SA"]; end
r = struct();

sigma = cfg.sigma;
sz = size(v);
useGpu = isa(v,'gpuArray');
X = dlarray(v,'SSCB');
Xc = dlarray(x,'SSCB');
nChs = prod(cfg.stride);

% --- W: Wiener filter
if any(arms == "W")
t = tic;
xw = wiener2(gather(double(v)),[5 5],double(sigma)^2);
r.W = metrics(single(xw),gather(x));
r.W.time = toc(t);
end

for k = 1:numel(cfg.K)
    K = cfg.K(k);
    acMask = fcn_ssacmask(nChs,K);
    m = reshape(cast(acMask,'like',v),1,1,[]);

    % --- N: uniform LSUN (tansacnet.lsun, shared angles)
    if any(arms == "N")
    rng(seed)
    t = tic;
    [anet,snet,info] = fcn_lsunpair(sz,cfg.stride,cfg.ovlp);
    prm = fcn_lsuninitprm(info,1,false);
    if useGpu, prm = dlupdate(@gpuArray,prm); end
    n.tBuild = toc(t);

    t = tic;
    [prm,n.hist1] = fcn_ssstep1lsun(anet,snet,info,prm,v,acMask,...
        cfg.nIt1N,cfg.lr1N(1),cfg.lr1N(2));
    sync(useGpu); n.t1 = toc(t);
    [anet,snet] = fcn_lsunsetangles(anet,snet,info,prm.th{1});
    [ac,dc] = predict(anet,X);
    [n.keptAll,n.keptAc] = kept(X,dc,ac,m);
    [acx,dcx] = predict(anet,Xc);                 % clean: evaluation only
    [n.keptAllX,n.keptAcX] = kept(Xc,dcx,acx,m);
    n.nEstW = sum(cellfun(@numel,prm.th{1}));

    t = tic;
    [prm,n.hist2] = fcn_ssstep2lsun(anet,snet,info,prm,v,sigma,cfg.nIt2,cfg.lr2);
    sync(useGpu); n.t2 = toc(t);

    t = tic;
    [sure,fu] = fcn_lsunsure(anet,snet,info,prm,X,sigma,"exact");
    n = mergestruct(n,metrics(gather(extractdata(fu)),gather(x)));
    n.sure = double(gather(extractdata(sure)));
    n.kappa = gather(extractdata(1./(1+exp(-prm.a{1}))));
    n.gain  = gather(extractdata(exp(prm.b{1})));
    n.prm = dlupdate(@gather,prm);
    sync(useGpu); n.tEval = toc(t);
    n.K = K;
    r.N(k) = n; %#ok<AGROW>
    clear n
    end

    % --- SA: SA-LSUN, ThetaMode 'Reuse'
    if any(arms == "SA")
    rng(seed)
    if useGpu, gpurng(seed); end
    t = tic;
    nets = fcn_ssbuildsalsun(sz,cfg.stride,cfg.ovlp,acMask,sigma);
    if useGpu
        nets.train = dlupdate(@gpuArray,nets.train);
        nets.whole = dlupdate(@gpuArray,nets.whole);
        nets.shrink = dlupdate(@gpuArray,nets.shrink);
    end
    s.tBuild = toc(t);

    t = tic;
    [tnet,s.hist1] = fcn_ssstep1salsun(nets.train,v,cfg.nIt1Sa,...
        cfg.lr1Sa(1),cfg.lr1Sa(2));
    sync(useGpu); s.t1 = toc(t);
    Y = predict(tnet,X);
    dc = predict(tnet,X,'Outputs','Lv1_DcOut');
    e0 = sum(X.^2,'all'); eY = sum(Y.^2,'all'); eDc = sum(dc.^2,'all');
    s.keptAll = double(gather(extractdata(eY/e0)));
    s.keptAc  = double(gather(extractdata((eY - eDc)/(e0 - eDc))));
    Yx = predict(tnet,Xc);                        % clean: evaluation only
    dcx = predict(tnet,Xc,'Outputs','Lv1_DcOut');
    e0 = sum(Xc.^2,'all'); eY = sum(Yx.^2,'all'); eDc = sum(dcx.^2,'all');
    s.keptAllX = double(gather(extractdata(eY/e0)));
    s.keptAcX  = double(gather(extractdata((eY - eDc)/(e0 - eDc))));
    s.nEstW = sum(cellfun(@numel,tnet.Learnables.Value));

    % Perfect reconstruction of Whole/Reuse with the learned estimator
    wnet = fcn_copylearnablesbyname(nets.whole,tnet);
    s.prErrX = relerr(predict(wnet,Xc),Xc);
    s.prErrV = relerr(predict(wnet,X),X);

    snet2 = fcn_copylearnablesbyname(nets.shrink,tnet);
    t = tic;
    [snet2,s.hist2] = fcn_ssstep2salsun(snet2,v,sigma,cfg.nIt2,cfg.lr2,cfg.epsFd);
    sync(useGpu); s.t2 = toc(t);

    t = tic;
    b = dlarray(randn([sz 1 cfg.nProbeEval],'like',v),'SSCB');
    [sureMc,fu,divMc] = fcn_sssuresalsun(snet2,X,sigma,"mc",b,cfg.epsFd);
    [sureFr,~,divFr] = fcn_sssuresalsun(snet2,X,sigma,"frozen");
    s = mergestruct(s,metrics(gather(extractdata(fu)),gather(x)));
    s.sure = double(gather(extractdata(sureMc)));
    s.sureFrozen = double(gather(extractdata(sureFr)));
    s.divMc = double(gather(extractdata(divMc)));
    s.divFrozen = double(gather(extractdata(divFr)));
    L = snet2.Learnables;
    s.kappa = gather(extractdata(1./(1+exp(-L.Value{strcmp(L.Layer,'Lv1_Shrink') & strcmp(L.Parameter,'A')}))));
    s.gain  = gather(extractdata(exp(L.Value{strcmp(L.Layer,'Lv1_Shrink') & strcmp(L.Parameter,'B')})));
    s.net = dlupdate(@gather,snet2);
    sync(useGpu); s.tEval = toc(t);
    s.K = K;
    r.SA(k) = s; %#ok<AGROW>
    clear s
    end
end
end

%%
function q = metrics(z,x)
z = gather(z); x = gather(x);
q.psnr = psnr(z,x);
q.ssim = ssim(z,x);
q.mse  = double(mean((z - x).^2,'all'));
q.img  = single(z);
end

function [kAll,kAc] = kept(X,dc,ac,m)
e0 = sum(X.^2,'all'); eDc = sum(dc.^2,'all'); eAc = sum((m.*ac).^2,'all');
kAll = double(gather(extractdata((eDc + eAc)/e0)));
kAc  = double(gather(extractdata(eAc/(e0 - eDc))));
end

function e = relerr(y,x)
e = double(gather(extractdata(sqrt(sum((y - x).^2,'all')/sum(x.^2,'all')))));
end

function a = mergestruct(a,b)
for f = fieldnames(b).'
    a.(f{1}) = b.(f{1});
end
end

function sync(useGpu)
if useGpu, wait(gpuDevice); end
end
