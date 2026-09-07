function [prm,hist,done] = fcn_trainlsun(anet,snet,info,prm,exemplars,sigmaRange,nIters,learnRate,miniBatchSize,isVerbose,ckptFile,maxSeconds)
%FCN_TRAINLSUN Train the LSUN denoiser by denoising score matching
%
%   [prm,hist] = fcn_trainlsun(anet,snet,info,prm,exemplars,sigmaRange,...
%   nIters,learnRate,miniBatchSize,isVerbose,ckptFile,maxSeconds) minimises the MMSE loss of Section 10.2.3
%   of Muramatsu (2026) with Adam. EXEMPLARS is a cell array of reference
%   images. SIGMARANGE is either a scalar noise level or a pair
%   [sigmaMin sigmaMax], in which case the noise level of each iteration is
%   drawn from a log-uniform distribution over that interval, so that a
%   single denoiser covers the whole range.
%
%   No projection step is needed: the LSUN parameterisation keeps the
%   analysis and synthesis operators exactly adjoint at every iteration.
%
%   MINIBATCHSIZE images are drawn with replacement at each iteration, each
%   perturbed at its own noise level.
%
%   Long runs may be split across several calls: pass CKPTFILE to store the
%   parameters, the Adam moments and the iteration counter, and MAXSECONDS to
%   stop early. Calling again resumes where it stopped. DONE reports whether
%   NITERS has been reached.
%
%   HIST returns the loss history.
%
% Copyright (c) 2026, Shogo MURAMATSU, All rights reserved.

arguments
    anet, snet, info struct, prm struct
    exemplars cell
    sigmaRange double
    nIters (1,1) double = 1000
    learnRate (1,1) double = 1e-2
    miniBatchSize (1,1) double = 8
    isVerbose (1,1) logical = true
    ckptFile (1,:) char = ''
    maxSeconds (1,1) double = inf
end

isCond = ~isscalar(sigmaRange);
% resume from a checkpoint if one is there: long runs are split across calls
avgG = []; avgSqG = []; it0 = 0; hist = zeros(nIters,1);
if ~isempty(ckptFile) && isfile(ckptFile)
    C = load(ckptFile);
    prm = C.prm; avgG = C.avgG; avgSqG = C.avgSqG; it0 = C.it0;
    hist(1:numel(C.hist)) = C.hist;
    if isVerbose
        fprintf("    resuming at iteration %d of %d\n",it0,nIters);
    end
end
tStart = tic;
done = true;
for it = it0+1:nIters
    [x,v,sigma] = fcn_minibatch(exemplars,sigmaRange,isCond,miniBatchSize);
    [loss,grads] = dlfeval(@fcn_lsunloss,anet,snet,info,prm,v,x,sigma);
    [prm,avgG,avgSqG] = adamupdate(prm,grads,avgG,avgSqG,it,learnRate);
    hist(it) = double(gather(extractdata(loss)));
    if isVerbose && mod(it,max(1,round(nIters/10)))==0
        fprintf("    iter %5d/%d  loss = %.4e\n",it,nIters,hist(it));
    end
    if toc(tStart) > maxSeconds && it < nIters
        it0 = it; done = false;
        if isVerbose
            fprintf("    time budget reached at iteration %d of %d\n",it,nIters);
        end
        break
    end
    it0 = it;
end
if ~isempty(ckptFile)
    hist = hist(1:it0);
    save(ckptFile,'prm','avgG','avgSqG','it0','hist','-v7.3');
end
end
