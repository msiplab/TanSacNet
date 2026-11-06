function [prm,hist,done] = fcn_trainlsunsure(anet,snet,info,prm,observations,sigmaObs,sigmaRange,nIters,learnRate,miniBatchSize,isVerbose,ckptFile,maxSeconds)
%FCN_TRAINLSUNSURE Self-supervised training of the LSUN denoiser by SURE
%
%   Same interface and checkpointing as FCN_TRAINLSUN, but OBSERVATIONS is a
%   cell array of NOISY images v = x + sigmaObs*zeta. No clean image is used.
%
%   SIGMARANGE scalar (= sigmaObs): train at the observation noise level.
%   SIGMARANGE = [lo hi]: draw sigma' log-uniformly from [max(lo,sigmaObs), hi]
%   and re-noise, u = v + sqrt(sigma'^2 - sigmaObs^2)*z. Then u = x + sigma'*zeta'
%   exactly, so SURE at sigma' is still unbiased: one noisy image trains the
%   noise-level conditional denoiser (score estimator) for every sigma' >= sigmaObs.
%
%   For a locally-structured (per-block) LSUN, every observation must have the
%   block grid the network was built for.

arguments
    anet, snet, info struct, prm struct
    observations cell
    sigmaObs (1,1) double
    sigmaRange double
    nIters (1,1) double = 400
    learnRate (1,1) double = 5e-3
    miniBatchSize (1,1) double = 8
    isVerbose (1,1) logical = true
    ckptFile (1,:) char = ''
    maxSeconds (1,1) double = inf
end

isCond = ~isscalar(sigmaRange);
lo = max(sigmaRange(1),sigmaObs);
avgG = []; avgSqG = []; it0 = 0; hist = zeros(nIters,1);
if ~isempty(ckptFile) && isfile(ckptFile)
    C = load(ckptFile);
    prm = C.prm; avgG = C.avgG; avgSqG = C.avgSqG; it0 = C.it0;
    hist(1:numel(C.hist)) = C.hist;
    if isVerbose, fprintf("    resuming at iteration %d of %d\n",it0,nIters); end
end
sz = size(observations{1});
tStart = tic; done = true;
for it = it0+1:nIters
    B = miniBatchSize;
    vb = zeros([sz 1 B],'like',observations{1});
    for b = 1:B, vb(:,:,1,b) = observations{randi(numel(observations))}; end
    if isCond
        s = exp(log(lo) + (log(sigmaRange(2))-log(lo))*rand(1,1,1,B,'like',vb));
        vb = vb + sqrt(max(s.^2 - sigmaObs^2,0)).*randn(size(vb),'like',vb);
    else
        s = sigmaObs*ones(1,1,1,B,'like',vb);
    end
    [loss,grads] = dlfeval(@fcn_lsunsureloss,anet,snet,info,prm,dlarray(vb,'SSCB'),s);
    [prm,avgG,avgSqG] = adamupdate(prm,grads,avgG,avgSqG,it,learnRate);
    hist(it) = double(gather(extractdata(loss)));
    if isVerbose && mod(it,max(1,round(nIters/10)))==0
        fprintf("    iter %5d/%d  SURE = %.4e\n",it,nIters,hist(it));
    end
    if toc(tStart) > maxSeconds && it < nIters
        it0 = it; done = false;
        if isVerbose, fprintf("    time budget reached at iteration %d of %d\n",it,nIters); end
        break
    end
    it0 = it;
end
if ~isempty(ckptFile)
    hist = hist(1:it0);
    save(ckptFile,'prm','avgG','avgSqG','it0','hist','-v7.3');
end
end
