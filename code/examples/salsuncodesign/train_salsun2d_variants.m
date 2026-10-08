function results = train_salsun2d_variants(options)
%TRAIN_SALSUN2D_VARIANTS Train SA-LSUN on base-field variants and evaluate streaming options
%
%   results = train_salsun2d_variants() trains the 2-D SA-LSUN of
%   main_salsun2d.m (same settings and loss) on the wave equation data,
%   once per base field variant ('none', 'batch', 'iir'), then evaluates
%   every trained network with the reference implementation:
%
%   - reconstruction error with the standardization statistics of the
%     image (the original model) and with causal statistics
%     ('previous', 'ema', 'fir2'; see salsun2d_infer_sequence)
%   - amplification of a one-block perturbation through the estimators
%     (the leak via the whole-image statistics), with the trained
%     parameters
%
%   The trained parameters are saved to OutputDir/<variant>.mat together
%   with the packed vector w for the FPGA kernel.
%
%   Options: Variants (cell, default {'none','batch','iir'}), Rho (0.9),
%   Scope ('dc'), MaxEpochs (100), MiniBatchSize (64), NumCoefs (2),
%   Crop ([300 300], top-left crop of the frames), NumFrames (149, after
%   the first frame, which the IIR variant leaves at zero), OutputDir,
%   Seed (0), StatsRho (0.9), StatsModes (cell of statistics modes to
%   evaluate, default {'image','previous','ema','fir2'}; the evaluation
%   of each mode runs the reference on every frame, several minutes for
%   300 x 300 frames).
%
%   Requires a GPU for practical training times; runs on the CPU for
%   small settings, e.g. Crop=[32 32], NumFrames=8, MaxEpochs=2.
%
% Requirements: MATLAB R2026b, Deep Learning Toolbox
%
% Copyright (c) 2026, Shogo MURAMATSU
%
% All rights reserved.
%
% Contact address: Shogo MURAMATSU,
%    Faculty of Engineering, Niigata University,
%    8050 2-no-cho Ikarashi, Nishi-ku,
%    Niigata, 950-2181, JAPAN
%
% http://msiplab.eng.niigata-u.ac.jp/
%
arguments
    options.Variants cell = {'none','batch','iir'}
    options.Rho (1,1) double = 0.9
    options.Scope {mustBeMember(options.Scope,{'dc','full'})} = 'dc'
    options.MaxEpochs (1,1) double = 100
    options.MiniBatchSize (1,1) double = 64
    options.NumCoefs (1,1) double = 2
    options.Crop (1,2) double = [300 300]
    options.NumFrames (1,1) double = 149
    options.OutputDir = fullfile(fileparts(mfilename('fullpath')),'results')
    options.Seed (1,1) double = 0
    options.StatsRho (1,1) double = 0.9
    options.StatsModes cell = {'image','previous','ema','fir2'}
end
here = fileparts(mfilename('fullpath'));
addpath(here,fullfile(here,'..','..'),fullfile(here,'..','salsun'));
if ~isfolder(options.OutputDir)
    mkdir(options.OutputDir)
end
stride = [4 4];
nChsTotal = prod(stride);
coefMask = reshape([ones(options.NumCoefs,1); zeros(nChsTotal-options.NumCoefs,1)],2,[]).';
coefMask = coefMask(:);
statsModes = options.StatsModes;

%% Data
u = salsun2d_wave_data();
u = single(u(1:options.Crop(1),1:options.Crop(2),:));
[szy,szx,nTotal] = size(u);
frames = 2:min(1+options.NumFrames,nTotal);   % frame 1 is zero for the IIR variant
fprintf('Data: %d x %d, frames %d..%d, GPU: %d\n',szy,szx,frames(1),frames(end),canUseGPU);

results = struct('Variant',{},'Loss',{},'Mse',{},'MseDlnetwork',{},'MaxDiffDlnetwork',{}, ...
    'Floor',{},'TrainingTime',{});
for iVariant = 1:numel(options.Variants)
    variant = options.Variants{iVariant};
    fprintf('\n==== variant: base field %s\n',variant);
    [uf,~] = salsun2d_base_field(u,Method=variant,Rho=options.Rho,Scope=options.Scope,Stride=stride);

    %% Training (as in main_salsun2d.m)
    rng(options.Seed)
    tic
    [trainnet,loss] = trainAnalyzer(uf(:,:,frames),coefMask,options);
    trainingTime = toc;
    fprintf('training time %.0f s, final loss %.4g\n',trainingTime,loss(end));

    %% Reconstruction network on the CPU, parameters for the reference and the FPGA
    reconnet = reconstructionNetwork(trainnet,[szy szx],coefMask);
    params = salsun2d_extract_params(reconnet);
    w = salsun2d_pack_params(params);

    %% Evaluation with the reference implementation
    mse = struct();
    for iMode = 1:numel(statsModes)
        mode = statsModes{iMode};
        y = salsun2d_infer_sequence(u,params,BaseField=variant,Rho=options.Rho, ...
            Scope=options.Scope,Statistics=mode,StatsRho=options.StatsRho);
        mse.(mode) = mean((u(:,:,frames) - y(:,:,frames)).^2,'all');
        fprintf('  statistics %-8s  MSE %.4g\n',mode,mse.(mode));
    end

    % Sanity check against the dlnetwork itself (image statistics), 3 frames
    ufCheck = uf(:,:,frames(1:3));
    yNet = zeros(size(ufCheck),'single');
    yRef = zeros(size(ufCheck),'single');
    for k = 1:3
        yNet(:,:,k) = extractdata(reconnet.predict(dlarray(ufCheck(:,:,k),'SSCB')));
        yRef(:,:,k) = salsun2d_infer(ufCheck(:,:,k),params);
    end
    maxDiff = max(abs(yNet(:) - yRef(:)));
    mseNet = mean((ufCheck - yNet).^2,'all');
    fprintf('  dlnetwork vs reference (3 frames): max abs diff %.3g, dlnetwork MSE of fluctuation %.4g\n', ...
        maxDiff,mseNet);

    %% Perturbation leak with the trained parameters (double precision)
    iPerturbed = frames(min(10,numel(frames)));
    floorLevels = perturbationFloor(double(uf(:,:,iPerturbed)),salsun2d_cast_params(params,'double'));
    fprintf('  far-field change / perturbed-block change: est1 %.2g est2 %.2g est3 %.2g est4 %.2g est5 %.2g coefs %.2g output %.2g\n', ...
        floorLevels);

    %% Save
    results(iVariant).Variant = variant;
    results(iVariant).Loss = loss;
    results(iVariant).Mse = mse;
    results(iVariant).MseDlnetwork = mseNet;
    results(iVariant).MaxDiffDlnetwork = maxDiff;
    results(iVariant).Floor = floorLevels;
    results(iVariant).TrainingTime = trainingTime;
    save(fullfile(options.OutputDir,[variant '.mat']),'params','w','loss','mse','floorLevels','options');
end
save(fullfile(options.OutputDir,'results.mat'),'results','options');

%% Summary
fprintf('\n%-8s','variant');
fprintf(' %10s',statsModes{:});
fprintf(' | %8s %8s %8s\n','leak est5','coefs','output');
for iVariant = 1:numel(results)
    r = results(iVariant);
    fprintf('%-8s',r.Variant);
    for iMode = 1:numel(statsModes)
        fprintf(' %10.4g',r.Mse.(statsModes{iMode}));
    end
    fprintf(' | %8.2g %8.2g %8.2g\n',r.Floor(5),r.Floor(6),r.Floor(7));
end
end

%% Training of the analyzer with the coefficient mask (main_salsun2d.m)
function [trainnet,lossHistory] = trainAnalyzer(data,coefMask,options)
import tansacnet.salsun.* tansacnet.lsun.*
[szy,szx,szt] = size(data);
nChsTotal = numel(coefMask);
analysislgraph = fcn_createsalsunlgraph2d([], ...
    'InputSize',[szy szx],'Stride',[4 4],'OverlappingFactor',[3 3], ...
    'NumberOfVanishingMoments',true,'NumberOfNeighborBlocks',[3 3], ...
    'NumberOfResidualBlocks',3,'Width',2,'Mode','Analyzer');
trainlgraph = analysislgraph.replaceLayer('Lv1_AcOut', ...
    maskLayer('Name','Lv1_AcMask','Mask',coefMask(2:end),'NumberOfChannels',nChsTotal-1));
trainlgraph = trainlgraph.addLayers(lsunChannelConcatenation2dLayer('Name','Lv1_Cmp1_Cn'));
trainlgraph = trainlgraph.connectLayers('Lv1_AcMask','Lv1_Cmp1_Cn/ac');
trainlgraph = trainlgraph.connectLayers('Lv1_DcOut','Lv1_Cmp1_Cn/dc');

mbsize = min(options.MiniBatchSize,szt);
ds = arrayDatastore(data,'IterationDimension',3);
mbq = minibatchqueue(ds,'MiniBatchSize',mbsize,'MiniBatchFcn',@(c) cat(4,c{:}), ...
    'MiniBatchFormat','SSCB','OutputEnvironment','auto','PartialMiniBatch','discard');
dlX0 = next(mbq);
trainnet = dlnetwork(trainlgraph,dlX0);
assert(trainnet.Initialized)
reset(mbq);

numIterationsPerEpoch = floor(szt/mbsize);
numIterations = options.MaxEpochs*numIterationsPerEpoch;
B1 = 0.9; B2 = 0.999; ep = 1e-12;
initialLearnRate = 1e-3; finalLearnRate = 1e-5;
avgGrad = []; avgSqGrad = [];
iteration = 0;
lossHistory = zeros(numIterations,1);
tEpoch = tic;
for epoch = 1:options.MaxEpochs
    shuffle(mbq);
    while hasdata(mbq)
        iteration = iteration + 1;
        dlX = next(mbq);
        [gradients,loss] = dlfeval(@modelGradients,trainnet,dlX);
        learnRate = finalLearnRate + (initialLearnRate - finalLearnRate) ...
            *(1 + cos(pi*iteration/numIterations))/2;
        [trainnet,avgGrad,avgSqGrad] = adamupdate(trainnet,gradients, ...
            avgGrad,avgSqGrad,iteration,learnRate,B1,B2,ep);
        lossHistory(iteration) = loss;
    end
    if epoch == 1 || mod(epoch,10) == 0 || epoch == options.MaxEpochs
        fprintf('  epoch %3d/%d  loss %.4g  (%.1f s/epoch)\n',epoch,options.MaxEpochs,loss,toc(tEpoch)/epoch);
    end
end
end

function [gradients,loss] = modelGradients(dlnet,dlX)
% Energy compaction loss of main_salsun2d.m: ||x||^2 - ||F(x)||^2 (>= 0)
dlY = forward(dlnet,dlX);
loss = sum(dlX.^2,"all")/size(dlX,4) - sum(dlY.^2,"all")/size(dlY,4);
gradients = dlgradient(loss,dlnet.Learnables);
loss = double(gather(extractdata(loss)));
end

%% Reconstruction network (Whole, Reuse, with the mask) on the CPU
function reconnet = reconstructionNetwork(trainnet,inputSize,coefMask)
import tansacnet.salsun.* tansacnet.lsun.*
nChsTotal = numel(coefMask);
wholelgraph = fcn_createsalsunlgraph2d([], ...
    'InputSize',inputSize,'Stride',[4 4],'OverlappingFactor',[3 3], ...
    'NumberOfVanishingMoments',true,'NumberOfNeighborBlocks',[3 3], ...
    'NumberOfResidualBlocks',3,'Width',2,'Mode','Whole','ThetaMode','Reuse', ...
    'Device','cpu');
wholelgraph = wholelgraph.disconnectLayers('Lv1_AcOut','Lv1_AcIn');
wholelgraph = wholelgraph.addLayers(maskLayer('Name','Lv1_AcMask', ...
    'Mask',coefMask(2:end),'NumberOfChannels',nChsTotal-1));
wholelgraph = wholelgraph.connectLayers('Lv1_AcOut','Lv1_AcMask');
wholelgraph = wholelgraph.connectLayers('Lv1_AcMask','Lv1_AcIn');
reconnet = dlnetwork(wholelgraph);

tblDst = reconnet.Learnables;
tblSrc = trainnet.Learnables;
for i = 1:height(tblSrc)
    idx = find(strcmp(tblDst.Layer,tblSrc.Layer{i}) & strcmp(tblDst.Parameter,tblSrc.Parameter{i}));
    if ~isempty(idx)
        tblDst.Value{idx} = gather(tblSrc.Value{i});
    end
end
reconnet.Learnables = tblDst;
end

%% Far-field change caused by a one-block perturbation (see measure_receptive_field)
function floorLevels = perturbationFloor(x,params)
stride = params.Stride;
nBlk = size(x)./stride;
center = floor(nBlk/2) + 1;
xp = x;
rows = (center(1)-1)*stride(1) + (1:stride(1));
cols = (center(2)-1)*stride(2) + (1:stride(2));
rng(1)
xp(rows,cols) = xp(rows,cols) + 0.5*std(x(:))*(rand(stride) - 0.5);
[y0,c0,th0] = salsun2d_infer(x,params);
[y1,c1,th1] = salsun2d_infer(xp,params);
[R,C] = ndgrid(1:nBlk(1),1:nBlk(2));
dr = min(mod(R-center(1),nBlk(1)),mod(center(1)-R,nBlk(1)));
dc = min(mod(C-center(2),nBlk(2)),mod(center(2)-C,nBlk(2)));
dist = max(dr,dc);
far = dist >= max(dist(:)) - 3;
floorLevels = zeros(1,7);
% thetas are nAngles x nBlocks with the blocks in column-major order of
% the nRows x nCols grid, so they reshape to the grid
for k = 1:5
    d = reshape(max(abs(th1{k}-th0{k}),[],1),nBlk);
    floorLevels(k) = median(d(far))/d(center(1),center(2));
end
d = reshape(max(abs(c1-c0),[],1),nBlk);
floorLevels(6) = median(d(far))/d(center(1),center(2));
dy = abs(y1-y0);
d = squeeze(max(max(reshape(dy,stride(1),nBlk(1),stride(2),nBlk(2)),[],1),[],3));
floorLevels(7) = median(d(far))/d(center(1),center(2));
end
