function [prm,hist,done] = fcn_tgtrainconvtnrd(prm,exemplars,sigmaRange,nIters,learnRate,projectEvery,miniBatchSize,isVerbose,ckptFile,maxSeconds,ckptEvery)
%FCN_TGTRAINCONVTNRD fcn_trainconvtnrd + random states in the checkpoint
%
%   [prm,hist,done] = fcn_tgtrainconvtnrd(prm,exemplars,sigmaRange,nIters,...
%       learnRate,projectEvery,miniBatchSize,isVerbose,ckptFile,maxSeconds,ckptEvery)
%   COPY of examples/sbdm/fcn_trainconvtnrd.m (Sensei's, not changed) with
%   two additions, so that a resumed run is bitwise the same as an
%   unbroken run:
%     1. the checkpoint also stores the CPU rng state (crops, randi in
%        fcn_minibatch) and the GPU rng state (noise, randn 'like' gpuArray),
%        and a resume restores them;
%     2. a checkpoint is also written every CKPTEVERY iterations (default
%        inf = only at the end / at the time budget, as the original).
%   The update, the loss, the mini-batch and the learning rate are the same
%   code as the original (constant lr Adam, fcn_minibatch, fcn_convtnrdloss).
%   The caller sets rng / gpurng before the first call.
%   (Used by main_sbdm_salsun_tnrdguar.m.)

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
    ckptEvery (1,1) double = inf
end

isCond = ~isscalar(sigmaRange);
useGpu = isa(exemplars{1},'gpuArray');
avgG = []; avgSqG = []; it0 = 0; hist = zeros(nIters,1);
if ~isempty(ckptFile) && isfile(ckptFile)
    C = load(ckptFile);
    prm = C.prm; avgG = C.avgG; avgSqG = C.avgSqG; it0 = C.it0;
    hist(1:numel(C.hist)) = C.hist;
    rng(C.rngCpu);                                   % (added)
    if useGpu
        gpurng(C.rngGpu);                            % (added)
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
    [prm,avgG,avgSqG] = adamupdate(prm,grads,avgG,avgSqG,it,learnRate);
    if projectEvery > 0 && mod(it,projectEvery)==0
        prm = fcn_convtnrdproject(prm);
    end
    hist(it) = double(gather(extractdata(loss)));
    if isVerbose && mod(it,max(1,round(nIters/10)))==0
        fprintf("    iter %6d/%d  loss = %.4e  (%.1f s)\n",it,nIters,hist(it),toc(tStart));
    end
    it0 = it;
    if toc(tStart) > maxSeconds && it < nIters
        done = false;
        if isVerbose
            fprintf("    time budget reached at iteration %d of %d\n",it,nIters);
        end
        break
    end
    if ~isempty(ckptFile) && mod(it,ckptEvery) == 0 && it < nIters   % (added)
        savectx(ckptFile,prm,avgG,avgSqG,it0,hist,useGpu);
    end
end
hist = hist(1:it0);
if ~isempty(ckptFile)
    savectx(ckptFile,prm,avgG,avgSqG,it0,hist,useGpu);
end
end

function savectx(f,prm,avgG,avgSqG,it0,hist,useGpu) %#ok<INUSD>
prm = dlupdate(@gather,prm); avgG = dlupdate(@gather,avgG); avgSqG = dlupdate(@gather,avgSqG); %#ok<NASGU>
hist = hist(1:it0); %#ok<NASGU>
rngCpu = rng; rngGpu = []; %#ok<NASGU>
if useGpu, rngGpu = gpurng; end %#ok<NASGU>
ftmp = [f '.tmp.mat'];
save(ftmp,'prm','avgG','avgSqG','it0','hist','rngCpu','rngGpu','-v7.3'); movefile(ftmp,f);
end
