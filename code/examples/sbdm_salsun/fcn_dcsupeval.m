function q = fcn_dcsupeval(mdl,x,v,cfg,netsSz)
%FCN_DCSUPEVAL Evaluate a supervised model (fcn_dcsuptrain) on one image
%
%   q = fcn_dcsupeval(mdl,x,v,cfg,netsSz)
%   N  : the trained shared angles and shrinkage are used on a pair of
%        the image size (fcn_dclsunpair); also the exact SURE (fcn_dcsure,
%        information only).
%   SA : the trained estimator and shrinkage layers are copied by name to
%        SA-LSUN nets of the image size (the estimator weights do not
%        depend on the image size). netsSz (optional) = prebuilt nets of
%        the image size (fcn_dcbuildsalsun, estimator "net", shrink true,
%        same leak / dcShrink), to avoid rebuilding them for every image.
%   x is used for evaluation only. q.prErrX = ||D E x - x||/||x||.
sigma = single(mdl.sigma); sz = size(v); N = numel(v);
useGpu = isa(v,'gpuArray');
X = dlarray(v,'SSCB'); Xc = dlarray(x,'SSCB');
t = tic;
if mdl.kind == "N"
    [anet,snet,info] = fcn_dclsunpair(sz,cfg.stride,cfg.ovlp,mdl.leak);
    th = mdl.th; ab = mdl.ab;
    if useGpu, th = dlupdate(@gpuArray,th); ab = dlupdate(@gpuArray,ab); end
    [anetT,snetT] = fcn_lsunsetangles(anet,snet,info,th);
    [ac,dc] = predict(anetT,X);
    fu = fcn_dcdenoise(anet,snet,info,th,ab,X,sigma);
    q.sure = double(gather(extractdata(fcn_dcsure(ab,ac,dc,sigma,N))));
    [acx,dcx] = predict(anetT,Xc);
    q.prErrX = relerr(predict(snetT,acx,dcx),Xc);
    q.kappa = gather(extractdata(sigmoid(ab.a))); q.gain = gather(extractdata(exp(ab.b)));
    if isfield(ab,'a0')
        q.kappa0 = double(gather(extractdata(sigmoid(ab.a0)))); q.gain0 = double(gather(extractdata(exp(ab.b0))));
    else
        q.kappa0 = NaN; q.gain0 = NaN;
    end
else
    if nargin < 5 || isempty(netsSz)
        o = struct('estimator',"net",'leak',mdl.leak,'bottleneck',cfg.bottleneck,...
            'shrink',true,'dcShrink',mdl.dcShrink,'dcInit',"id",'sigma',sigma);
        netsSz = fcn_dcbuildsalsun(sz,cfg.stride,cfg.ovlp,o);
    end
    [snet,ns] = setlearn(netsSz.shrink,mdl.learn);
    [wnet,nw] = setlearn(netsSz.whole,mdl.learn);
    assert(ns == height(snet.Learnables) && nw == height(wnet.Learnables),...
        'Not all learnables were copied (%d/%d, %d/%d).',ns,height(snet.Learnables),nw,height(wnet.Learnables))
    if useGpu, snet = dlupdate(@gpuArray,snet); wnet = dlupdate(@gpuArray,wnet); end
    fu = predict(snet,X);
    q.sure = NaN;
    q.prErrX = relerr(predict(wnet,Xc),Xc);
    L = mdl.learn;
    g = @(ly,pr) double(gather(extractdata(L.Value{strcmp(L.Layer,ly) & strcmp(L.Parameter,pr)})));
    q.kappa = 1./(1+exp(-g('Lv1_Shrink','A'))); q.gain = exp(g('Lv1_Shrink','B'));
    if mdl.dcShrink
        q.kappa0 = 1./(1+exp(-g('Lv1_ShrinkDc','A'))); q.gain0 = exp(g('Lv1_ShrinkDc','B'));
    else
        q.kappa0 = NaN; q.gain0 = NaN;
    end
end
z = gather(extractdata(fu)); xg = gather(x);
q.psnr = psnr(z,xg); q.ssim = ssim(z,xg);
q.mse = double(mean((z - xg).^2,'all')); q.img = single(z);
if useGpu, wait(gpuDevice); end
q.tEval = toc(t);
end

function [net,n] = setlearn(net,Lsrc)
% copy learnables with the same layer/parameter name (fcn_copylearnablesbyname)
L = net.Learnables;
n = 0;
for i = 1:height(Lsrc)
    idx = find(strcmp(L.Layer,Lsrc.Layer{i}) & strcmp(L.Parameter,Lsrc.Parameter{i}));
    if ~isempty(idx)
        assert(isequal(size(L.Value{idx}),size(Lsrc.Value{i})),...
            'Size mismatch for %s/%s.',Lsrc.Layer{i},Lsrc.Parameter{i})
        L.Value{idx} = Lsrc.Value{i}; n = n + 1;
    end
end
net.Learnables = L;
end

function e = relerr(y,x)
e = double(gather(extractdata(sqrt(sum((y - x).^2,'all')/sum(x.^2,'all')))));
end
