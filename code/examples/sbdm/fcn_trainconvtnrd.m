function [prm,hist,done] = fcn_trainconvtnrd(prm,exemplars,sigmaRange,nIters,learnRate,projectEvery,miniBatchSize,isVerbose,ckptFile,maxSeconds)
%FCN_TRAINCONVTNRD Train the convolutional TNRD baseline
%
%   [prm,hist] = fcn_trainconvtnrd(prm,exemplars,sigmaRange,nIters,...
%   learnRate,projectEvery,miniBatchSize,isVerbose,ckptFile,maxSeconds) is the counterpart of FCN_TRAINLSUN
%   for the baseline of Example 10.2 of Muramatsu (2026). Set
%   PROJECTEVERY to a positive integer to project the dictionary onto the
%   Stiefel manifold every PROJECTEVERY iterations ("parseval" dictionary),
%   or to 0 to leave the weights unconstrained ("tied" dictionary).
%
%   Passing a scalar SIGMARANGE trains the unconditional denoiser at that
%   noise level; a pair [sigmaMin sigmaMax] trains the noise-level
%   conditional denoiser used as the score estimator.
%
%   Long runs may be split across several calls: pass CKPTFILE to store the
%   parameters, the Adam moments and the iteration counter, and MAXSECONDS to
%   stop early. Calling again resumes where it stopped. DONE reports whether
%   NITERS has been reached.
%
% Copyright (c) 2026, Shogo MURAMATSU, All rights reserved.

arguments
    prm struct
    exemplars cell
    sigmaRange double
    nIters (1,1) double = 1000
    learnRate (1,1) double = 1e-3
    projectEvery (1,1) double = 0
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
    % the unconditional baseline absorbs the noise level into lambda
    sigmaIn = single(1); if isCond, sigmaIn = sigma; end
    [loss,grads] = dlfeval(@fcn_convtnrdloss,prm,v,x,sigmaIn);
    [prm,avgG,avgSqG] = adamupdate(prm,grads,avgG,avgSqG,it,learnRate);
    if projectEvery > 0 && mod(it,projectEvery)==0
        prm = fcn_convtnrdproject(prm);
    end
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
