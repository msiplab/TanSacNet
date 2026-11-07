function [prm,hist,done,gnorm] = fcn_tgtrainconvtnrdcos(prm,exemplars,sigmaRange,nIters,lr,clipNorm,miniBatchSize,isVerbose,ckptFile,maxSeconds,ckptEvery,stopAfter)
%FCN_TGTRAINCONVTNRDCOS fcn_tgtrainconvtnrd + cosine lr decay + gradient clipping
%
%   [prm,hist,done,gnorm] = fcn_tgtrainconvtnrdcos(prm,exemplars,sigmaRange,nIters,...
%       lr,clipNorm,miniBatchSize,isVerbose,ckptFile,maxSeconds,ckptEvery,stopAfter)
%   COPY of fcn_tgtrainconvtnrd.m (itself a copy of Sensei's
%   examples/sbdm/fcn_trainconvtnrd.m, not changed) with these changes:
%     1. lr = [lr0 lr1]: cosine decay over nIters, as fcn_szsuptrain /
%        fcn_dcsuptrain:  lr(it) = lr1 + (lr0 - lr1)*(1 + cos(pi*it/nIters))/2.
%        lr0 = lr1 gives the constant lr of the original.
%     2. clipNorm: global-norm gradient clipping before the Adam update
%        (fcn_tgclipgrad). inf = no clipping (= the original).
%     3. gnorm(it): the global gradient norm before clipping (log only).
%     4. stopAfter: stop (with a checkpoint) after this iteration, so that
%        the first its of a run can be looked at and the run resumed later.
%   No projection (projectEvery = 0, as main_sbdm.m). Everything else (the
%   mini-batch, the loss, Adam, the checkpoint with the CPU/GPU rng states)
%   is the same code as fcn_tgtrainconvtnrd. The caller sets rng / gpurng
%   before the first call.
%   (Used by main_sbdm_salsun_tnrdretrain.m.)

arguments
    prm struct
    exemplars cell
    sigmaRange double
    nIters (1,1) double = 1000
    lr (1,2) double = [1e-3 1e-3]
    clipNorm (1,1) double = inf
    miniBatchSize (1,1) double = 8
    isVerbose (1,1) logical = true
    ckptFile (1,:) char = ''
    maxSeconds (1,1) double = inf
    ckptEvery (1,1) double = inf
    stopAfter (1,1) double = inf
end

isCond = ~isscalar(sigmaRange);
useGpu = isa(exemplars{1},'gpuArray');
lrAt = @(it) lr(2) + (lr(1) - lr(2))*(1 + cos(pi*it/nIters))/2;
if lr(1) == lr(2), lrAt = @(it) lr(1); end            % exactly the original constant lr
avgG = []; avgSqG = []; it0 = 0; hist = zeros(nIters,1); gnorm = zeros(nIters,1);
if ~isempty(ckptFile) && isfile(ckptFile)
    C = load(ckptFile);
    prm = C.prm; avgG = C.avgG; avgSqG = C.avgSqG; it0 = C.it0;
    hist(1:numel(C.hist)) = C.hist; gnorm(1:numel(C.gnorm)) = C.gnorm;
    rng(C.rngCpu);
    if useGpu
        gpurng(C.rngGpu);
        prm = dlupdate(@gpuArray,prm);
        avgG = dlupdate(@gpuArray,avgG); avgSqG = dlupdate(@gpuArray,avgSqG);
    end
    if isVerbose
        fprintf("    resuming at iteration %d of %d\n",it0,nIters);
    end
end
tStart = tic;
done = true;
for it = it0+1:nIters
    [x,v,sigma] = fcn_minibatch(exemplars,sigmaRange,isCond,miniBatchSize);
    sigmaIn = single(1); if isCond, sigmaIn = sigma; end
    [loss,grads] = dlfeval(@fcn_convtnrdloss,prm,v,x,sigmaIn);
    [grads,gnorm(it)] = fcn_tgclipgrad(grads,clipNorm);
    [prm,avgG,avgSqG] = adamupdate(prm,grads,avgG,avgSqG,it,lrAt(it));
    hist(it) = double(gather(extractdata(loss)));
    if isVerbose && mod(it,max(1,round(nIters/10)))==0
        fprintf("    iter %6d/%d  loss = %.4e  lr %.3e  |g| %.3e  (%.1f s)\n",it,nIters,hist(it),lrAt(it),gnorm(it),toc(tStart));
    end
    it0 = it;
    if (toc(tStart) > maxSeconds || it >= stopAfter) && it < nIters
        done = false;
        if isVerbose
            fprintf("    stopped at iteration %d of %d\n",it,nIters);
        end
        break
    end
    if ~isempty(ckptFile) && mod(it,ckptEvery) == 0 && it < nIters
        savectx(ckptFile,prm,avgG,avgSqG,it0,hist,gnorm,useGpu);
    end
end
hist = hist(1:it0); gnorm = gnorm(1:it0);
if ~isempty(ckptFile)
    savectx(ckptFile,prm,avgG,avgSqG,it0,hist,gnorm,useGpu);
end
end

function savectx(f,prm,avgG,avgSqG,it0,hist,gnorm,useGpu) %#ok<INUSD>
prm = dlupdate(@gather,prm); avgG = dlupdate(@gather,avgG); avgSqG = dlupdate(@gather,avgSqG); %#ok<NASGU>
hist = hist(1:it0); gnorm = gnorm(1:it0); %#ok<NASGU>
rngCpu = rng; rngGpu = []; %#ok<NASGU>
if useGpu, rngGpu = gpurng; end %#ok<NASGU>
ftmp = [f '.tmp.mat'];
save(ftmp,'prm','avgG','avgSqG','it0','hist','gnorm','rngCpu','rngGpu','-v7.3'); movefile(ftmp,f);
end
