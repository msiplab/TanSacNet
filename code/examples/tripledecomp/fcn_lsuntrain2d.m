function [net,info] = fcn_lsuntrain2d(data,opts)
%FCN_LSUNTRAIN2D Train a 2-D LSUN basis field on a snapshot sequence
%
%   [net,info] = FCN_LSUNTRAIN2D(data,opts) trains the analysis network of
%   a two-dimensional locally-structured unitary network on the snapshot
%   sequence data (ny x nx x nT) and returns the trained dlnetwork net
%   together with the bookkeeping struct info.
%
%   The training objective is the energy discarded by the channel
%   truncation,
%
%       loss(theta) = mean_t ( ||y_t||^2 - ||Gamma D_theta' y_t||^2 ) ,
%
%   which is non-negative by Parseval's identity because D_theta is
%   unitary by construction.  Unitarity is therefore never enforced by a
%   penalty; it is a structural property of the network and holds exactly
%   at every iteration.
%
%   opts is a struct with the fields
%       stride            block size, default [4 4]
%       ovlpFactor        number of overlapping blocks, default [3 3]
%       nCoefs            retained channels K per block, default 4
%       maxEpochs         default 8
%       miniBatchSize     default 10
%       initialLearnRate  default 1e-3
%       stdInitAng        std of the initial angle perturbation, 1e-6
%       noDcLeakage       structural no-DC-leakage constraint, default true
%       seed              rng seed, default 0
%       verbose           print the loss per epoch, default true
%
%   info carries szExt (the padded size), the truncation mask coefMask in
%   the [dc; ac] channel order, the stride and the option struct actually
%   used, so that FCN_LSUNCOEFS2D can analyse further data consistently.
%
%   See also FCN_LSUNCOEFS2D, FCN_ENERGYCONC.

arguments
    data (:,:,:) double
    opts struct = struct()
end
import tansacnet.lsun.*

%% Options
defaults = struct( ...
    'stride',[4 4], ...
    'ovlpFactor',[3 3], ...
    'nCoefs',4, ...
    'maxEpochs',8, ...
    'miniBatchSize',10, ...
    'initialLearnRate',1e-3, ...
    'stdInitAng',1e-6, ...
    'noDcLeakage',true, ...
    'seed',0, ...
    'verbose',true);
fn = fieldnames(defaults);
for k = 1:numel(fn)
    if ~isfield(opts,fn{k})
        opts.(fn{k}) = defaults.(fn{k});
    end
end

stride = opts.stride;
nChsTotal = prod(stride);
assert(opts.nCoefs <= nChsTotal, ...
    'nCoefs (%d) must not exceed the number of channels (%d).', ...
    opts.nCoefs,nChsTotal)

%% Reproducibility
rng(opts.seed)

%% Pad to a multiple of the block size
[nrows,ncols,nFrames] = size(data);
szExt = [ceil(nrows/stride(1))*stride(1) ceil(ncols/stride(2))*stride(2)];
dataExt = padarray(data,szExt-[nrows ncols],0,'post');

%% Analysis network
lgraph = fcn_createlsunlgraph2d([], ...
    'InputSize',szExt, ...
    'Stride',stride, ...
    'OverlappingFactor',opts.ovlpFactor, ...
    'NumberOfVanishingMoments',opts.noDcLeakage, ...
    'Mode','Analyzer');
net = dlnetwork(lgraph);

% The learnables default to single precision, at which the unitarity and
% Parseval invariants of the repository convention hold only to about 1e-6.
% Casting to double restores them to 1e-14.
net = dlupdate(@double,net);

% Perturb the initial angles away from the DCT initialisation
nLearnables = height(net.Learnables);
for iLearnable = 1:nLearnables
    if net.Learnables.Parameter(iLearnable) == "Angles"
        net.Learnables.Value(iLearnable) = cellfun( ...
            @(x) x + opts.stdInitAng*randn(size(x)), ...
            net.Learnables.Value(iLearnable),'UniformOutput',false);
    end
end

%% Truncation mask in the [dc; ac] channel order
% The natural mask keeps the first nCoefs channels; the LSUN arranges the
% channels as [symmetric; antisymmetric], hence the interleaving, which
% follows ../pidmd/fcn_pidmdvialsun.m.
maskNat = [ones(opts.nCoefs,1); zeros(nChsTotal-opts.nCoefs,1)];
coefMask = [maskNat(1:2:end); maskNat(2:2:end)];
dlMask = reshape(coefMask,1,1,[]);

useGPU = canUseGPU;
if useGPU
    dlMask = gpuArray(dlMask);
end

%% Training
averageGrad = [];
averageSqGrad = [];
iteration = 0;
tStart = tic;
lossHistory = zeros(opts.maxEpochs,1);
for epoch = 1:opts.maxEpochs
    perm = randperm(nFrames);
    epochLoss = 0;
    nBatch = 0;
    for iStart = 1:opts.miniBatchSize:nFrames
        idx = perm(iStart:min(iStart+opts.miniBatchSize-1,nFrames));
        X = reshape(dataExt(:,:,idx),szExt(1),szExt(2),1,numel(idx));
        if useGPU
            X = gpuArray(X);
        end
        dlX = dlarray(X,'SSCB');
        iteration = iteration + 1;
        [loss,grad] = dlfeval(@modelLoss_,net,dlX,dlMask);
        [net,averageGrad,averageSqGrad] = adamupdate(net,grad, ...
            averageGrad,averageSqGrad,iteration,opts.initialLearnRate);
        epochLoss = epochLoss + loss;
        nBatch = nBatch + 1;
    end
    lossHistory(epoch) = epochLoss/nBatch;
    if opts.verbose
        fprintf('  epoch %2d/%d  discarded energy = %.6g  (%.1f s elapsed)\n', ...
            epoch,opts.maxEpochs,lossHistory(epoch),toc(tStart));
    end
end

%% Bookkeeping
info = struct( ...
    'szExt',szExt, ...
    'szOrg',[nrows ncols], ...
    'stride',stride, ...
    'nChsTotal',nChsTotal, ...
    'coefMask',coefMask, ...
    'lossHistory',lossHistory, ...
    'useGPU',useGPU, ...
    'opts',opts);
end

%% Loss: energy discarded by the channel truncation
function [loss,gradients] = modelLoss_(net,dlX,dlMask)
[o1,o2] = forward(net,dlX);
if size(o1,3) == 1
    C = cat(3,o1,o2);
else
    C = cat(3,o2,o1);
end
nB = size(dlX,4);
loss = (sum(dlX.^2,'all') - sum((C.*dlMask).^2,'all'))/nB;
gradients = dlgradient(loss,net.Learnables);
loss = double(gather(extractdata(loss)));
end
