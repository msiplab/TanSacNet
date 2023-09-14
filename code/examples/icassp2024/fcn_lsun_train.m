function [analsunnet,synlsunnet,coefMask] = fcn_lsun_train(DataT,stride,nCoefs,numEpochs)
nT = size(DataT,1);
nX = size(DataT,2);
%Stride (block size)
assert(mod(nX,stride)==0,'stride must be a divisor of nX.');
%Output dimension (per block)
assert(nCoefs<=stride,'nCoefs must be less than or equal to stride.')
%Number of overlapping blocks (number of shifts)
nof = 1;
kx = 2*nof+1; % Number of overlapping blocks (odd)
%Setting display
strbuf = "-- Settings --" + newline;
strbuf = strbuf.append("Data size (space): " + num2str(nX) + newline);
strbuf = strbuf.append("Data size (time): " + num2str(nT) + newline);
strbuf = strbuf.append("Block size: " + num2str(stride) + newline);
strbuf = strbuf.append("Output dimension (per block): " + num2str(nCoefs) + newline);
strbuf = strbuf.append("Number of overlapping blocks: " + num2str(kx) + newline);
disp(strbuf)

% One-dimensional locally structured unitary networks (1-D LSUNs)
% References.
%  Lu Gan and Kai-Kuang Ma, "On simplified order-one factorizations of paraunitary filterbanks," in  IEEE Transactions on Signal Processing, vol. 52, no. 3, pp. 674-686, March 2004, doi: 10.1109/ TSP.2003.822356.
% Original PUFB configuration
% Even-channel real coefficient symmetric delay decomposition (Real SDF) configuration [Fig. 7 (b), Gan et al,. IEEE T-SP, 2004]
% Number of channels M = 2m
% r_k = m
% Number of stages (k-1) (polyphase order N) set to even 2n, allowing for spatial non-causality
% Modified non-causal PUFB
% LSUN extension of modified non-causal PUFB
% Custom network construction
% - Defining custom deep learning layers - MATLAB & Simulink - MathWorks United Kingdom
import tansacnet.lsun.*
analysislgraph = fcn_createcslsunlgraph1d([],...
 'InputSize',nX,...
 'Stride',stride,...
 'OverlappingFactor',kx,...
 'Mode','Analyzer');
synthesislgraph = fcn_createcslsunlgraph1d([],...
 'InputSize',nX,...
 'Stride',stride,...
 'OverlappingFactor',kx,...
 'Mode','Synthesizer');
%
%{
figure
subplot(1,2,1)
plot(analysislgraph)
title('Analysis LSUN')
subplot(1,2,2)
plot(synthesislgraph)
title('Synthesis LSUN')
%}

%Initialisation of design parameters
% Standard deviation of initial angles
stdInitAng = 1e-9;
% Construction of synthesis network.
analysisnet = dlnetwork(analysislgraph); % <----
% Initialize
nLearnables = height(analysisnet.Learnables);
expanalyzer = '^Lv\d+_Cmp\d+_Q(\w\d|0)+(\w)+$';
nLayers = height(analysislgraph.Layers);
for iLearnable = 1:nLearnables
    if analysisnet.Learnables.Parameter(iLearnable)=="Angles"
        alayerName = analysisnet.Learnables.Layer(iLearnable);
        if ~isempty(regexp(alayerName,expanalyzer,'once'))
            disp("Angles in " + alayerName + " are set to N(-pi/2,"+num2str(stdInitAng^2)+")")
            analysisnet.Learnables.Value(iLearnable) = ...
                cellfun(@(x) double(x+stdInitAng*randn(size(x))-pi/2), ...
                analysisnet.Learnables.Value(iLearnable),'UniformOutput',false);
        else
            disp("Angles in " + alayerName + " are set to N(0,"+num2str(stdInitAng^2)+")")
            analysisnet.Learnables.Value(iLearnable) = ...
                cellfun(@(x) double(x+stdInitAng*randn(size(x))), ...
                analysisnet.Learnables.Value(iLearnable),'UniformOutput',false);
        end
    else
        analysisnet.Learnables.Value(iLearnable) = ...
            cellfun(@(x) double(x), ...
            analysisnet.Learnables.Value(iLearnable),'UniformOutput',false);
    end
end

%Establishment of concomitant relationships
%Copying design parameters
import tansacnet.lsun.*
% Construction of analysis network
analysislgraph = layerGraph(analysisnet);
synthesislgraph = fcn_cpparamsana2syn(synthesislgraph,analysislgraph);
synthesislgraph = fcn_cpparamsana2syn_csax_(synthesislgraph,analysislgraph);
synthesisnet = dlnetwork(synthesislgraph);
%Confirmation of the adjoint relationship (complete reconstruction).
nLearnables = height(synthesisnet.Learnables);
for iLearnable = 1:nLearnables
    synthesisnet.Learnables.Value(iLearnable) = ...
        cellfun(@(x) double(x), ...
        synthesisnet.Learnables.Value(iLearnable),'UniformOutput',false);
end

%
x = rand([1 nX 1 nT],'double');
dlx = dlarray(x,"SSCB"); % Deep learning array (SSCB)
[dls{1:2}] = analysisnet.predict(dlx);
dly = synthesisnet.predict(dls{:});
mse_ = mse(dlx,dly);
display("MSE: " + num2str(mse_))
assert(mse_<1e-6)

%Design parameter optimisation and signal approximation.
import tansacnet.lsun.*
analysislgraph = layerGraph(analysisnet);
% Coefficient masking
nChsTotal = prod(stride);
coefMask = reshape([ones(nCoefs,1); zeros(nChsTotal-nCoefs,1)],2,[]).';
coefMask = coefMask(:);
%nLevels = 1;
%for iLv = nLevels:-1:1
iLv = 1;
strLv = sprintf('Lv%0d_',iLv);
% For AC
analysislgraph = analysislgraph.replaceLayer([strLv 'AcOut'],...
 mask1dLayer('Name',[strLv 'AcMask'],'Mask',coefMask(2:end),...
 'NumberOfChannels',nChsTotal-1));
%strLvPre = strLv;
%end
% Output layer
iCmp = 1;
strCmp = sprintf('Cmp%0d_',iCmp);
%analysislgraph = analysislgraph.addLayers([...
% lsunChannelConcatenation1dLayer('Name',[strLv strCmp 'Cn']) ...
% lsunRegressionLayer('Coefficient output')
% ]);
analysislgraph = analysislgraph.addLayers(...
 lsunChannelConcatenation1dLayer('Name',[strLv strCmp 'Cn']));
analysislgraph = analysislgraph.connectLayers(...
 [strLv 'AcMask' ], [strLv strCmp 'Cn/ac']);
analysislgraph = analysislgraph.connectLayers(...
 [strLv 'DcOut' ], [strLv strCmp 'Cn/dc']);
%{
figure
plot(analysislgraph)
title('Analysis LSUN')
%}

%Reads numerical sequence data as 1-D images from a datastore.
% Load sequences

arrds = arrayDatastore(DataT,"ReadSize",1,"IterationDimension",1);
%arrds = transform(arrds,@(x) cell2mat(x));
%figure
arr = cell2mat(preview(arrds));
%{
for idx = 1:height(arr)
 plot(arr(idx,:))
 hold on
end
hold off
%}

%Design preparation
dlX = dlarray(gpuArray(arr(1,:)),"SSCB");
trainnet = dlnetwork(analysislgraph,dlX);
assert(trainnet.Initialized)
figure
monitor = trainingProgressMonitor(Metrics="Loss",Info="Epoch",XLabel="Iteration");

%Optimisation design
%Parameters for learning
%- Creating mini-batches for deep learning - MATLAB - MathWorks United Kingdom

%numEpochs = 2; % TODO !!!!!!!100; 

miniBatchSize = nT;
numObservationsTrain = nT;
numIterationsPerEpoch = ceil(numObservationsTrain / miniBatchSize);
numIterations = numEpochs * numIterationsPerEpoch;
% Minibatch
mbq = minibatchqueue(arrds,...
 "MinibatchSize",miniBatchSize,...
 "MiniBatchFcn",@(x) permute(cell2mat(x),[3 2 4 1]),...
 "OutputAsDlarray",1,...
 "OutputCast","double",...
 "MiniBatchFormat", "SSCB",...
 "OutputEnvironment", "gpu",...
 "PartialMiniBatch","discard");
% Training
averageGrad = [];
averageSqGrad = [];
iteration = 0;
epoch = 0;
start = tic;
% Loop over epochs.
while epoch < numEpochs && ~monitor.Stop
     epoch = epoch + 1;
     % Shuffle data.
     shuffle(mbq);
     % Loop over mini-batches.
     while hasdata(mbq) && ~monitor.Stop
     iteration = iteration + 1;
     % Read mini-batch of data.
     dlX = next(mbq);
     % Evaluate the model gradients, state, and loss using dlfeval and the
     % modelGradients function and update the network state.
    [loss,grad] = dlfeval(@modelLoss,trainnet,dlX);
     %Update the network parameters using the Adam optimizer.
     [trainnet,averageGrad,averageSqGrad] = ...
     adamupdate(trainnet,grad,averageGrad,averageSqGrad,iteration);
     % Display the training progress.
     recordMetrics(monitor,iteration,Loss=loss);
     updateInfo(monitor,Epoch=epoch + " of " + numEpochs);
     monitor.Progress = 100 * iteration/numIterations;
     end
end

%Trained Analysis LSUN
import tansacnet.lsun.*
analsunlgraph = fcn_createcslsunlgraph1d([],...
    'InputSize',nX,...
    'Stride',stride,...
    'OverlappingFactor',kx,...
    ...'NumberOfVanishingMoments',noDcLeakage,...
    'Mode','Analyzer');
synlsunlgraph = fcn_createcslsunlgraph1d([],...
    'InputSize',nX,...
    'Stride',stride,...
    'OverlappingFactor',kx,...
    ...'NumberOfVanishingMoments',noDcLeakage,...
    'Mode','Synthesizer');
trainlgraph = layerGraph(trainnet);
% Trained net -> Analyzer
synlsunlgraph = fcn_cpparamsana2syn(synlsunlgraph,trainlgraph);
synlsunlgraph = fcn_cpparamsana2syn_csax_(synlsunlgraph,trainlgraph); 
% Analyzer -> Synthesizer
analsunlgraph = fcn_cpparamssyn2ana(analsunlgraph,synlsunlgraph);
analsunlgraph = fcn_cpparamssyn2ana_csax_(analsunlgraph,synlsunlgraph); 

nLevels = 1;
%for iLv = nLevels:-1:1S
iLv = 1;

% !!! 完全再構成を確認するためマスク処理を無効化
% coefMask = ones(nChsTotal,1);

strLv = sprintf('Lv%0d_',iLv);
% For analyzer
analsunlgraph = analsunlgraph.replaceLayer([strLv 'DcOut'],...
    regressionLayer('Name',[strLv 'DcOut']));
%analsunlgraph = analsunlgraph.addLayers(...
%    mask1dLayer('Name',[strLv 'AcMask'],'Mask',coefMask(2:end),...
%    'NumberOfChannels',nChsTotal-1));
%analsunlgraph = analsunlgraph.connectLayers([strLv 'AcOut'],[strLv 'AcMask']);
analsunlgraph = analsunlgraph.replaceLayer([strLv 'AcOut'], ...
        mask1dLayer('Name',[strLv 'AcMask'],'Mask',coefMask(2:end),...
    'NumberOfChannels',nChsTotal-1));
analsunlgraph = analsunlgraph.addLayers(regressionLayer('Name',[strLv 'AcOut']));
analsunlgraph = analsunlgraph.connectLayers([strLv 'AcMask'],[strLv 'AcOut']);

% For synthesizer
synlsunlgraph = synlsunlgraph.replaceLayer([strLv 'Out'],...
    regressionLayer('Name',[strLv 'Out']));
%
synlsunlgraph = synlsunlgraph.disconnectLayers([strLv 'Ac feature input'],[strLv 'AcIn']);
synlsunlgraph = synlsunlgraph.addLayers(...
    mask1dLayer('Name',[strLv 'AcMask'],'Mask',coefMask(2:end),...
    'NumberOfChannels',nChsTotal-1));
synlsunlgraph = synlsunlgraph.connectLayers([strLv 'Ac feature input'],[strLv 'AcMask']);
synlsunlgraph = synlsunlgraph.connectLayers([strLv 'AcMask'],[strLv 'AcIn']);

%strLvPre = strLv;
%end

%

% Replace invalid linear layers with empty lattice parameters
fcn_replace_emptyangles_(analsunlgraph)
fcn_replace_emptyangles_(synlsunlgraph)


%analsunlgraph.Layers
%synlsunlgraph.Layers


analsunnet = assembleNetwork(analsunlgraph);
synlsunnet = assembleNetwork(synlsunlgraph);

%%%




end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [loss,gradients] = modelLoss(dlnet, dlX)
% Forward data through the dlnetwork object.
dlY = forward(dlnet,dlX); % F(x)
% Compute loss.
Nx = size(dlX,4);
Ny = size(dlY,4);
loss = sum(dlX.^2,"all")/Nx-sum(dlY.^2,"all")/Ny;
% Compute gradients.
gradients = dlgradient(loss,dlnet.Learnables);
loss = double(gather(extractdata(loss)));
end

%%%

function synthesislgraph = fcn_cpparamsana2syn_csax_(synthesislgraph,analysislgraph)
expanalyzer = '^Lv\d+_Cmp\d+_Q(\w\d|0)+(\w)+$';
nLayers = height(analysislgraph.Layers);
for iLayer = 1:nLayers
     alayer = analysislgraph.Layers(iLayer);
     alayerName = alayer.Name;
         if ~isempty(regexp(alayerName,expanalyzer,'once'))
         slayer = synthesislgraph.Layers({synthesislgraph.Layers.Name} == alayerName + "~");
         slayer.Angles = alayer.Angles;
         synthesislgraph = synthesislgraph.replaceLayer(slayer.Name,slayer);
         disp("Copy angles from " + alayerName + " to " + slayer.Name)
         end
end
end

%%%

function analysislgraph = fcn_cpparamssyn2ana_csax_(analysislgraph,synthesislgraph)
expanalyzer = '^Lv\d+_Cmp\d+_Q(\w\d|0)+(\w)+$';
nLayers = height(analysislgraph.Layers);
for iLayer = 1:nLayers
 alayer = analysislgraph.Layers(iLayer);
 alayerName = alayer.Name;
     if ~isempty(regexp(alayerName,expanalyzer,'once'))
     slayer = synthesislgraph.Layers({synthesislgraph.Layers.Name} == alayerName + "~");
     alayer.Angles = slayer.Angles;
     analysislgraph = analysislgraph.replaceLayer(alayerName,alayer);
     disp("Copy angles from " + slayer.Name + " to " + alayerName)
     end
end
end

%%%
function fcn_replace_emptyangles_(lsunlgraph)
explayer = '^Lv\d+_Cmp\d+_V(\w\d|0)+(~|)$';
nLayers = height(lsunlgraph.Layers);
for iLayer = 1:nLayers
    layer_ = lsunlgraph.Layers(iLayer);
    if ~isempty(regexp(layer_.Name,explayer,'once'))
        if isa(layer_,"tansacnet.lsun.lsunIntermediateFullRotation1dLayer") && isempty(layer_.Angles)
            newlayer = tansacnet.lsun.lsunSign1dLayer( ...
                'Name',layer_.Name, ...
                'Stride',layer_.Stride, ...
                'Mode',layer_.Mode, ...
                'NumberOfBlocks',layer_.NumberOfBlocks, ...
                'Mus',layer_.Mus);
            lsunlgraph = lsunlgraph.replaceLayer(layer_.Name,newlayer);
            display("Replaced " + layer_.Name + " to " + class(newlayer))
        end
    end
end
end