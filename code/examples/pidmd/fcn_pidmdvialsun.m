function [A,Vals,Vecs,projA,options,rt] = fcn_pidmdvialsun(obsData,dmdmethod,...
    nModes4PODtrunc,islsun,options)
%FCN_PIDMDVIALSUN この関数の概要をここに記述
%   詳細説明をここに記述
import tansacnet.lsun.*

%arguments
%    obsData (:,:,:) {mustBeNumeric}
%    dmdmethod (1,:) char {mustBeMember(dmdmethod,{'exact','orthogonal'})}
%    nModes4PODtrunc (1,1) {mustBePositive,mustBeInteger}
%    islsun logical = false
%end

% Options
if islsun
    if nargin < 5 % Default options for LSUN
        options.stride = [4 4];
        options.ovlpFactor = [5 5];
        options.nCoefs = 2;
        options.maxEpochs = 4;
        options.miniBatchSize = 5;
        options.stdInitAng = 1e-6;
        options.noDcLeakage = false;
        options.sgditeration = 0.01;
        options.sgdmomentum = 0.9;
        options.sgddecay = 0.01;
        options.initialLearnRate = 1e-4;
        options.lineLossTrain = [];
        options.useGPU = true;
    end
else
    options = [];
end

% Outputs
if isempty(obsData) && isempty(dmdmethod) && isempty(nModes4PODtrunc)
    A = [];
    Vals = [];
    Vecs = [];
    projA = [];
    return
end

% Data dimension
nrows = size(obsData,1);
ncols = size(obsData,2);
nFrames = size(obsData,3);

if islsun
    disp("LSUN training ...")
    %% Pad array
    stride = options.stride;
    szy = ceil(nrows/stride(1))*stride(1);
    szx = ceil(ncols/stride(2))*stride(2);
    ovlpFactor = options.ovlpFactor;
    noDcLeakage = options.noDcLeakage;
    stdInitAng = options.stdInitAng;
    nCoefs = options.nCoefs;
    miniBatchSize = options.miniBatchSize;
    maxEpochs = options.maxEpochs;

    %% Pad zeros to make the size multiples of nDecs.
    obsData = padarray(obsData,[szy szx]-[nrows ncols],0,'post');

    %% Construction of analysis network.
    analysislgraph = fcn_createlsunlgraph2d([],...
        'InputSize',[szy szx],...
        'Stride',stride,...
        'OverlappingFactor',ovlpFactor,...
        'NumberOfVanishingMoments',noDcLeakage,...
        'Mode','Analyzer');
    analysisnet = dlnetwork(analysislgraph);

    %% Initialize
    nLearnables = height(analysisnet.Learnables);
    for iLearnable = 1:nLearnables
        if analysisnet.Learnables.Parameter(iLearnable)=="Angles"
            analysisnet.Learnables.Value(iLearnable) = ...
                cellfun(@(x) x+stdInitAng*randn(size(x)), ...
                analysisnet.Learnables.Value(iLearnable),'UniformOutput',false);
        end
    end

    %% Parameter optimization and approximation
    % Create analysis layerGraph
    analysislgraph = layerGraph(analysisnet);
    clear analysisnet

    % Coefficient masking
    nChsTotal = prod(stride);
    coefMask = [ones(nCoefs,1); zeros(nChsTotal-nCoefs,1)];
    coefMask = [coefMask(1:2:end); coefMask(2:2:end)];
    nLevels = 1;
    %for iLv = nLevels:-1:1
    iLv = 1;
    strLv = sprintf('Lv%0d_',iLv);
    % For AC
    analysislgraph = analysislgraph.replaceLayer([strLv 'AcOut'],...
        maskLayer('Name',[strLv 'AcMask'],'Mask',coefMask(2:end),...
        'NumberOfChannels',nChsTotal-1));
    %strLvPre = strLv;
    %end

    % Output layer
    iCmp = 1;
    strCmp = sprintf('Cmp%0d_',iCmp);
    analysislgraph = analysislgraph.addLayers(...
        lsunChannelConcatenation2dLayer('Name',[strLv strCmp 'Cn']));
    analysislgraph = analysislgraph.connectLayers(...
        [strLv 'AcMask' ], [strLv strCmp 'Cn/ac']);
    analysislgraph = analysislgraph.connectLayers(...
        [strLv 'DcOut' ], [strLv strCmp 'Cn/dc']);

    % Setup for training
    arrds = arrayDatastore(obsData(:,:,1:end-1),'IterationDimension',3);
    mbq = minibatchqueue(arrds,...
        'MiniBatchSize',miniBatchSize,...
        'OutputAsDlarray',1,...
        'MiniBatchFcn',@preprocessMiniBatch,...
        'MiniBatchFormat','SSCB',...
        'OutputCast','double',...
        'OutputEnvironment','auto');
    shuffle(mbq)
    dlX = next(mbq);
    trainnet = dlnetwork(analysislgraph,dlX);
    if ~trainnet.Initialized
        warnign("Not initialized")
    end
    clear analysislgraph

    %% Configuration
    velocity = [];
    iteration = options.sgditeration; % 0.01
    momentum = options.sgdmomentum; % 0.9;
    decay = options.sgddecay; % 0.01;
    initialLearnRate = options.initialLearnRate; % 1e-4;

    %% Loop over epochs.
    %start = tic;
    if ~isempty(options.lineLossTrain)
        %D = duration(0,0,toc(start),'Format','hh:mm:ss');
        clearpoints(options.lineLossTrain)
    end
    for epoch = 1:maxEpochs

        % Shuffle data.
        shuffle(mbq)

        while hasdata(mbq)
            % Loop over mini-batches.
            iteration = iteration + 1;

            % Read mini-batch of data.
            dlX = next(mbq);

            % Evaluate the model gradients, state, and loss using dlfeval and the
            % modelGradients function and update the network state.
            [gradients,loss] = dlfeval(@modelGradients,trainnet,dlX);

            % Determine learning rate for time-based decay learning rate schedule.
            learnRate = initialLearnRate/(1 + decay*iteration);

            % Update the network parameters using the SGDM optimizer.
            [trainnet,velocity] = sgdmupdate(trainnet,gradients,velocity,learnRate,momentum);

            % Display the training progress.
            if ~isempty(options.lineLossTrain)
                %D = duration(0,0,toc(start),'Format','hh:mm:ss');
                addpoints(options.lineLossTrain,iteration,loss)
                title("Epoch: " + epoch) % + ", Elapsed: " + string(D))
                drawnow
            end
        end
    end

    %lsunresfile = "../../../results/lsun4pidmd"+string(datetime('now','TimeZone','local','Format','yyyy-M-dd-HH-mm-ssZ'))
    %save(lsunresfile,"trainnet","szy","szx","nCoefs");

    %%
    disp("LSUN encoding ...")
    szExt = [szy szx];
    lsunana4predict = fcn_createlsunlgraph2d([],...
        'InputSize',szExt,...
        'Stride',stride,...
        'OverlappingFactor',ovlpFactor,...
        'NumberOfVanishingMoments',noDcLeakage,...
        'Mode','Analyzer');
    lsunsyn4predict = fcn_createlsunlgraph2d([],...
        'InputSize',szExt,...
        'Stride',stride,...
        'OverlappingFactor',ovlpFactor,...
        'NumberOfVanishingMoments',noDcLeakage,...
        'Mode','Synthesizer');
    trainlgraph = layerGraph(trainnet);
    lsunsyn4predict = fcn_cpparamsana2syn(lsunsyn4predict,trainlgraph);
    lsunana4predict = fcn_cpparamssyn2ana(lsunana4predict,lsunsyn4predict);
    clear trainlgraph
    %

    %% Assemble analyzer
    iLv = 1;
    strLv = sprintf('Lv%0d_',iLv);
    % For AC
    lsunana4predict = lsunana4predict.disconnectLayers([strLv 'Cmp1_Sp/ac'],[strLv 'AcOut']);
    lsunana4predict = lsunana4predict.addLayers(...
        maskLayer('Name',[strLv 'AcMask'],'Mask',coefMask(2:end),...
        'NumberOfChannels',nChsTotal-1));
    lsunana4predict = lsunana4predict.connectLayers(...
        [strLv 'Cmp1_Sp/ac'],[strLv 'AcMask']);
    lsunana4predict = lsunana4predict.connectLayers(...
        [strLv 'AcMask'],[strLv 'AcOut']);
    for iLayer = 1:height(lsunana4predict.Layers)
        layer = lsunana4predict.Layers(iLayer);
        if contains(layer.Name,"Lv"+nLevels+"_DcOut") || ...
                ~isempty(regexp(layer.Name,'^Lv\d+_AcOut','once'))
            lsunana4predict = lsunana4predict.replaceLayer(layer.Name,...
                regressionLayer('Name',layer.Name));
        end
    end

    %% Assemble synthesizer
    lsunsyn4predict = lsunsyn4predict.replaceLayer('Lv1_Out',...
        regressionLayer('Name','Lv1_Out'));
    for iLayer = 1:height(lsunsyn4predict.Layers)
        layer = lsunsyn4predict.Layers(iLayer);
        if contains(layer.Name,'Ac feature input')
            iLv = str2double(layer.Name(3));
            sbSize = szExt.*(stride.^(-iLv));
            newlayer = ...
                imageInputLayer([sbSize (sum(nChsTotal)-1)],'Name',layer.Name,'Normalization','none');
            lsunsyn4predict = lsunsyn4predict.replaceLayer(...
                layer.Name,newlayer);
        elseif contains(layer.Name,sprintf('Lv%0d_Dc feature input',nLevels))
            iLv = str2double(layer.Name(3));
            sbSize = szExt.*(stride.^(-iLv));
            newlayer = ...
                imageInputLayer([sbSize 1],'Name',layer.Name,'Normalization','none');
            lsunsyn4predict = lsunsyn4predict.replaceLayer(...
                layer.Name,newlayer);
        end
    end

    %% Define the LSUN analyzer and synthesizer
    analysislsun4predict = assembleNetwork(lsunana4predict);
    synthesislsun4predict = assembleNetwork(lsunsyn4predict);
    clear lsunana4predict lsunsyn4predict

    nLayers = height(analysislsun4predict.Layers);
    for iLayer = 1:nLayers
        layer = analysislsun4predict.Layers(iLayer);
        if strcmp(layer.Name,"Lv1_AcMask")
            acMask = logical(squeeze(layer.Mask));
        end
    end

    analyzer_ = @(x) fcn_lsunanalyzer_(x,analysislsun4predict,acMask); % Analyze array x, and vectorize the result
    synthesizer_ = @(x) fcn_lsunsynthesizer_(x,synthesislsun4predict,acMask); % Reshape x and synthesize the result

    %% piDMD w/ POD mode truncation via LSUN
    % Project X and Y onto the LSUN feature space
    if options.useGPU
        obsData = gpuArray(obsData);
    end
    dataMatrix = analyzer_(obsData);

else % w/o LSUN
    dataMatrix = reshape(obsData,[],nFrames);
end

% piDMD
X = dataMatrix(:,1:end-1);
Y = dataMatrix(:,2:end);
if strcmp(string(dmdmethod),"orthogonal")
    [A, Vals, Vecs, projA] = piDMD(X,Y,dmdmethod,nModes4PODtrunc);
else
    [A, Vals, Vecs] = piDMD(X,Y,dmdmethod,nModes4PODtrunc);
    projA = [];
end

% Time evolution
if nargout > 5

    % Reconstruction
    x0 = X(:,1);
    %R = real(fcn_timeevoldmd(x0,Vals,Vecs,nFrames));  % Take real only for numerical error correction
    R = fcn_timeevoldmdwA(x0,A,nFrames);

    % Rendering
    if islsun
        % LSUN decoding
        disp("LSUN decoding ...")
        rt = synthesizer_(R);
        rt = rt(1:nrows,1:ncols,:);
    else
        rt = reshape(R,nrows,ncols,[]);
    end
end

end

%%
function X = preprocessMiniBatch(XCell)
% Extract image data from the cell array and concatenate over fourth
% dimension to add a third singleton dimension, as the channel
% dimension.
X = cat(4,XCell{:});
end

function [gradients, loss] = modelGradients(dlnet, dlX)
% Gradients w.r.t. LSUN design parameters
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

%% LSUN Analyzer
function Xlsun = fcn_lsunanalyzer_(x,analyzer,acMask)
nFrames = size(x,3);
nLevels = 1;
szy = size(x,1);
szx = size(x,2);
stride = extractstride(analyzer);
Xlsun = zeros(szy*szx*(1+sum(acMask))/prod(stride),nFrames,'like',x);
for iFrame = 1:nFrames
    [s{1:nLevels+1}] = analyzer.predict(x(:,:,iFrame));
    dc = s{1};
    ac = s{end};
    ac = ac(:,:,acMask); % Shrink
    Xlsun(:,iFrame) = reshape(cat(3,dc,ac),[],1);
end
end

%% LSUN Synthesizer
function x = fcn_lsunsynthesizer_(Xlsun,synthesizer,acMask)
nLevels = 1;
nLayers = height(synthesizer.Layers);
for iLayer = 1:nLayers
    layer = synthesizer.Layers(iLayer);
    if strcmp(layer.Name,"Lv1_Ac feature input")
        szac = layer.InputSize;
    end
end
nFrames = size(Xlsun,2);
stride = extractstride(synthesizer);
x = zeros(szac(1)*stride(1),szac(2)*stride(2),nFrames);
s = cell(nLevels+1,1);
ac = zeros(szac,'like',Xlsun);
for iFrame = 1:nFrames
    xlsun = reshape(Xlsun(:,iFrame),szac(1),szac(2),[]);
    s{1} = xlsun(:,:,1);
    % Extension
    ac(:,:,acMask) = xlsun(:,:,2:end);
    s{nLevels+1} = ac;
    x(:,:,iFrame) = synthesizer.predict(s{nLevels+1:-1:1});
end
end

%% Extract stride
function stride = extractstride(lsun)
import tansacnet.lsun.*

% Extraction of information
expfinallayer = '^Lv1_Cmp1+_V0~?$';
nLayers = height(lsun.Layers);
for iLayer = 1:nLayers
    layer = lsun.Layers(iLayer);
    if ~isempty(regexp(layer.Name,expfinallayer,'once'))
        stride = layer.Stride;
    end
end
end