function n = fcn_ssnrunimage(x,v,cfg,seed,K,mode,tau)
%FCN_SSNRUNIMAGE Uniform LSUN arm N (shared angles) on one image, v only
%
%   n = fcn_ssnrunimage(x,v,cfg,seed,K,mode,tau)
%   Same as arm N of fcn_ssrunimage (fcn_lsunpair, zero initial angles,
%   step 1 with cfg.nIt1N its, lr cfg.lr1N; step 2 with cfg.nIt2 its,
%   lr cfg.lr2), with two changes:
%     - step 1: mode "zonal" = fcn_ssstep1lsun (DC + K AC channels, as N)
%               mode "hard"  = fcn_ssstep1lsunhard, threshold tau*sigma
%     - step 2: fcn_ssstep2lsunfast (frozen-transform shortcut of the same
%               exact SURE)
%   The output image and the reported SURE come from
%   examples/sbdm/fcn_lsunsure(...,"exact") (original code).
%   x is used ONLY for evaluation. Kept-energy statistics with the zonal
%   K mask and with the hard thresholds cfg.tauList*sigma are stored for
%   v and x (fcn_sskeptstats).

sigma = cfg.sigma; sz = size(v);
useGpu = isa(v,'gpuArray');
X = dlarray(v,'SSCB'); Xc = dlarray(x,'SSCB');
acMask = fcn_ssacmask(prod(cfg.stride),K);
m = reshape(cast(acMask,'like',v),1,1,[]);
thrList = double(cfg.tauList)*double(sigma);
n.K = K; n.mode = mode; n.tau = tau; n.sigma = sigma;

rng(seed)
t = tic;
[anet,snet,info] = fcn_lsunpair(sz,cfg.stride,cfg.ovlp);
prm = fcn_lsuninitprm(info,1,false);
if useGpu, prm = dlupdate(@gpuArray,prm); end
n.tBuild = toc(t);

t = tic;
switch mode
    case "zonal"
        [prm,n.hist1] = fcn_ssstep1lsun(anet,snet,info,prm,v,acMask,...
            cfg.nIt1N,cfg.lr1N(1),cfg.lr1N(2));
        n.frac1 = NaN;
    case "hard"
        [prm,n.hist1,n.frac1] = fcn_ssstep1lsunhard(anet,snet,info,prm,v,...
            tau*double(sigma),cfg.nIt1N,cfg.lr1N(1),cfg.lr1N(2));
    otherwise
        error("unknown mode %s",mode)
end
sync(useGpu); n.t1 = toc(t);
[anet,snet] = fcn_lsunsetangles(anet,snet,info,prm.th{1});
[ac,dc] = predict(anet,X);
n.statV = fcn_sskeptstats(ac,dc,X,m,thrList);
[acx,dcx] = predict(anet,Xc);                    % clean: evaluation only
n.statX = fcn_sskeptstats(acx,dcx,Xc,m,thrList);
n.keptAc = n.statV.zonalAc; n.keptAcX = n.statX.zonalAc;   % as fcn_ssrunimage
n.prErrX = relerr(predict(snet,acx,dcx),Xc);
n.prErrV = relerr(predict(snet,ac,dc),X);
n.nEstW = sum(cellfun(@numel,prm.th{1}));

t = tic;
[prm,n.hist2] = fcn_ssstep2lsunfast(anet,snet,info,prm,v,sigma,cfg.nIt2,cfg.lr2);
sync(useGpu); n.t2 = toc(t);

t = tic;
[sure,fu] = fcn_lsunsure(anet,snet,info,prm,X,sigma,"exact");
z = gather(extractdata(fu));
n.psnr = psnr(z,gather(x)); n.ssim = ssim(z,gather(x));
n.mse = double(mean((z - gather(x)).^2,'all')); n.img = single(z);
n.sure = double(gather(extractdata(sure)));
n.kappa = gather(extractdata(1./(1+exp(-prm.a{1}))));
n.gain  = gather(extractdata(exp(prm.b{1})));
n.prm = dlupdate(@gather,prm);
sync(useGpu); n.tEval = toc(t);
n.tTotal = n.tBuild + n.t1 + n.t2 + n.tEval;
end

function e = relerr(y,x)
e = double(gather(extractdata(sqrt(sum((y - x).^2,'all')/sum(x.^2,'all')))));
end

function sync(useGpu)
if useGpu, wait(gpuDevice); end
end
