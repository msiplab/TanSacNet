function [prm,hist,done] = fcn_trainlsunsurex(anet,snet,info,prm,observations,sigmaObs,sigmaRange,nIters,learnRate,miniBatchSize,isVerbose,ckptFile,maxSeconds,opts)
%FCN_TRAINLSUNSUREX SURE training of a local LSUN with fewer free angles
%
%   As FCN_TRAINLSUNSURE, starting from a UNIFORM model PRM (one angle column),
%   with two ways to limit overfitting of the per-block angles:
%
%     'GroupSize' [gr gc]  blocks of one group share their angles
%                          ([1 1] = one set per block, the plain local LSUN)
%     'Lambda'    >= 0     adds Lambda*mean((theta - theta_U0).^2) to SURE,
%                          keeping every group close to the uniform angles
%
%   GroupSize [1 1] with Lambda 0 is the same model as FCN_LSUNEXPANDPRM
%   followed by FCN_TRAINLSUNSURE (a useful check).
%
%   The returned PRM has per-block angles [nAngles x nBlocks], so it works
%   with FCN_LSUNDENOISE and FCN_LSUNSURE directly.
%   HIST is [nIters x 2]: total loss and SURE.

arguments
    anet, snet, info struct, prm struct
    observations cell
    sigmaObs (1,1) double
    sigmaRange double
    nIters (1,1) double = 400
    learnRate (1,1) double = 5e-3
    miniBatchSize (1,1) double = 1
    isVerbose (1,1) logical = true
    ckptFile (1,:) char = ''
    maxSeconds (1,1) double = inf
    opts.GroupSize (1,2) double = [1 1]
    opts.Lambda (1,1) double {mustBeNonnegative} = 0
end

[idx,nG] = fcn_lsungroupidx(info,opts.GroupSize);

% anchor (uniform angles) and group-level learnables
anchor = prm.th;
pl = prm;
for s = 1:numel(prm.th)
    for k = 1:numel(prm.th{s})
        th = prm.th{s}{k};
        if size(th,2) ~= 1
            error("tansacnet:sbdm:NotUniform","Start from a uniform model (one angle column).")
        end
        pl.th{s}{k} = repmat(th,1,nG);
    end
end

isCond = ~isscalar(sigmaRange);
lo = max(sigmaRange(1),sigmaObs);
avgG = []; avgSqG = []; it0 = 0; hist = zeros(nIters,2);
if ~isempty(ckptFile) && isfile(ckptFile)
    C = load(ckptFile);
    pl = C.pl; avgG = C.avgG; avgSqG = C.avgSqG; it0 = C.it0;
    hist(1:size(C.hist,1),:) = C.hist;
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
    [loss,grads,sure] = dlfeval(@lossx,anet,snet,info,pl,dlarray(vb,'SSCB'),s,...
        anchor,idx,opts.Lambda);
    [pl,avgG,avgSqG] = adamupdate(pl,grads,avgG,avgSqG,it,learnRate);
    hist(it,:) = double(gather(extractdata([loss sure])));
    if isVerbose && mod(it,max(1,round(nIters/10)))==0
        fprintf("    iter %5d/%d  loss = %.4e  SURE = %.4e\n",it,nIters,hist(it,1),hist(it,2));
    end
    if toc(tStart) > maxSeconds && it < nIters
        it0 = it; done = false;
        if isVerbose, fprintf("    time budget reached at iteration %d of %d\n",it,nIters); end
        break
    end
    it0 = it;
end
if ~isempty(ckptFile)
    hist = hist(1:it0,:);
    save(ckptFile,'pl','avgG','avgSqG','it0','hist','-v7.3');
end
prm = expandgroups(pl,idx);
end

function prm = expandgroups(pl,idx)
prm = pl;
for s = 1:numel(pl.th)
    for k = 1:numel(pl.th{s})
        prm.th{s}{k} = pl.th{s}{k}(:,idx);
    end
end
end

function [loss,grads,sure] = lossx(anet,snet,info,pl,u,sigma,anchor,idx,lambda)
pen = 0; cnt = 0;
for s = 1:numel(pl.th)
    for k = 1:numel(pl.th{s})
        d = pl.th{s}{k} - anchor{s}{k};
        pen = pen + sum(d.^2,'all');
        cnt = cnt + numel(d);
    end
end
sure = mean(fcn_lsunsure(anet,snet,info,expandgroups(pl,idx),u,sigma),'all');
loss = sure + lambda*pen/cnt;
grads = dlgradient(loss,pl);
end
