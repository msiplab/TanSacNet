function salsunLgraph = ...
    fcn_createsalsunlgraph2d(salsunLgraph,varargin)
%FCN_CREATESALSUNLGRAPH2D
%
%   Angle estimator interface:
%     The angles of every rotation layer are given by an estimator
%     sub-graph, which 'EstimatorFunction' (default
%     @tansacnet.salsun.fcn_createparamestimator2dlgraph) adds as
%
%       lgraph = estimatorFunction(lgraph,<EstimatorOptions>,...
%           'Prefix',prefix,'TargetType',targetType,...
%           'NumberOfChannels',nChs,'NoDcLeakage',noDcLeakage,...
%           'NumberOfBlocks',nBlocks,'Stride',stride,...
%           'Side',side,'StageName',stageName)
%
%     with TargetType 'Initial' or 'Intermediate', Side 'Analysis' or
%     'Synthesis' and StageName 'V0', 'h1', 'h2', ..., 'v1', ... The
%     estimator must add a layer named [prefix 'Ext'], which receives the
%     block state before the rotation (nChs x nRows x nCols x nSamples),
%     and a layer named [prefix 'Theta'], which outputs the angles
%     (nAnglesFull x prod(nBlocks) x nSamples, blocks in column-major
%     order, the first nZeroPad angles zero; see fcn_salsunanglecounts).
%     The builder connects them to the rotation layer. With ThetaMode
%     'Reuse', the synthesis side has no estimators and reuses the
%     angles of the analysis side.
%
%     'EstimatorOptions' is a cell array of name-value arguments passed
%     to the estimator. For the default estimator, NumberOfNeighborBlocks,
%     NumberOfResidualBlocks, Width and BottleneckSize are passed before
%     them; other estimators receive EstimatorOptions only.
%
% Requirements: MATLAB R2022a
%
% Copyright (c) 2026, Motoyasu SUZUKI and Shogo MURAMATSU
%
% All rights reserved.
%
% Contact address: Shogo MURAMATSU,
%                Faculty of Engineering, Niigata University,
%                8050 2-no-cho Ikarashi, Nishi-ku,
%                Niigata, 950-2181, JAPAN
%
% http://msiplab.eng.niigata-u.ac.jp/

if isempty(salsunLgraph)
    salsunLgraph = layerGraph();
end

import tansacnet.lsun.*
import tansacnet.salsun.*

p = inputParser;
addParameter(p,'InputSize',[32 32])
addParameter(p,'NumberOfComponents',1)
addParameter(p,'Stride',[2 2])
addParameter(p,'OverlappingFactor',[1 1])
addParameter(p,'NumberOfLevels',1)
addParameter(p,'NumberOfVanishingMoments',[1 1])
addParameter(p,'Mode','Whole')
addParameter(p,'ThetaMode','Recompute')
addParameter(p,'Prefix','')
addParameter(p,'AppendInOutLayers',true)
if canUseGPU
    addParameter(p,'Device','cuda')
else
    addParameter(p,'Device','cpu')
end
addParameter(p,'DType','single')
addParameter(p,'NumberOfNeighborBlocks',[3 3])
addParameter(p,'NumberOfResidualBlocks',3)
addParameter(p,'Width',2)
addParameter(p,'BottleneckSize',0)
addParameter(p,'EstimatorFunction',@tansacnet.salsun.fcn_createparamestimator2dlgraph,...
    @(x) isa(x,'function_handle'))
addParameter(p,'EstimatorOptions',{},@iscell)
parse(p,varargin{:})

nComponents = p.Results.NumberOfComponents;
inputSize = [p.Results.InputSize nComponents];
stride = p.Results.Stride;
ovlpFactor = p.Results.OverlappingFactor;
nLevels = p.Results.NumberOfLevels;
device = p.Results.Device;
dtype = p.Results.DType;
prefix = p.Results.Prefix;
isapndinout = p.Results.AppendInOutLayers;

% Angle estimator (see the estimator interface above)
est.Function = p.Results.EstimatorFunction;
est.Options = p.Results.EstimatorOptions;
if endsWith(func2str(est.Function),'fcn_createparamestimator2dlgraph')
    est.Options = [{'NumberOfNeighborBlocks',p.Results.NumberOfNeighborBlocks,...
        'NumberOfResidualBlocks',p.Results.NumberOfResidualBlocks,...
        'Width',p.Results.Width,...
        'BottleneckSize',p.Results.BottleneckSize} est.Options(:).'];
end

noDcLeakage = p.Results.NumberOfVanishingMoments;
if isscalar(noDcLeakage)
    noDcLeakage = [1 1]*noDcLeakage;
end

mode = p.Results.Mode;
if strcmp(mode,'Whole')
    isAnalyzer = true;
    isSynthesizer = true;
elseif strcmp(mode,'Analyzer')
    isAnalyzer = true;
    isSynthesizer = false;
elseif strcmp(mode,'Synthesizer')
    isAnalyzer = false;
    isSynthesizer = true;
else
    error('Mode should be in { ''Whole'', ''Analyzer'', ''Synthesizer'' }');
end

thetaMode = p.Results.ThetaMode;
if ~any(strcmp(thetaMode,{'Recompute','Reuse'}))
    error('ThetaMode should be in { ''Recompute'', ''Reuse'' }');
end
if strcmp(thetaMode,'Reuse') && ~(isAnalyzer && isSynthesizer)
    error(['ThetaMode ''Reuse'' requires Mode ''Whole'' ' ...
        '(the analysis-side estimators must exist to reuse).']);
end

nDecs = prod(stride);
nChannels = [ceil(nDecs/2) floor(nDecs/2)];
if nChannels(1) ~= nChannels(2)
    throw(MException('SaLsunLayer:InvalidStride',...
        '[%d %d] : The product of stride should be even.',...
        stride(1),stride(2)))
end
if ~all(mod(ovlpFactor,2))
    throw(MException('SaLsunLayer:InvalidOverlappingFactor',...
        '[%d %d] : Currently, odd overlapping factors are only supported.',...
        ovlpFactor(1),ovlpFactor(2)))
end

nBlocksDeepest = inputSize(1:2)./(stride.^nLevels);
if any(mod(nBlocksDeepest,1))
    error('[%d %d] : Input size should be a multiple of stride^NumberOfLevels.',...
        stride(1),stride(2))
end

%% Build the (flat) list of atom-extension + intermediate-rotation
% stage specs, reused per (level, component). Unlike
% tansacnet.lsun.fcn_createlsunlgraph2d (which builds layers inline),
% each stage here also needs its own estimator sub-graph, so naming is
% separated from construction (see addAnalysisStage/addSynthesisStage).
% Stage order and Direction/TargetChannels pairing are unchanged.
stageSpecs = struct('name',{},'atomDir',{},'atomTarget',{},...
    'synAtomDir',{},'synAtomTarget',{});
for iOrderH = 2:2:ovlpFactor(2)-1
    stageSpecs(end+1) = mkstage(sprintf('h%d',iOrderH-1),... %#ok<AGROW>
        'Right','Difference','Left','Difference');
    stageSpecs(end+1) = mkstage(sprintf('h%d',iOrderH),... %#ok<AGROW>
        'Left','Sum','Right','Sum');
end
for iOrderV = 2:2:ovlpFactor(1)-1
    stageSpecs(end+1) = mkstage(sprintf('v%d',iOrderV-1),... %#ok<AGROW>
        'Down','Difference','Up','Difference');
    stageSpecs(end+1) = mkstage(sprintf('v%d',iOrderV),... %#ok<AGROW>
        'Up','Sum','Down','Sum');
end

%% Analysis
if isAnalyzer
    for iLv = 1:nLevels
        strLv = sprintf('Lv%0d_',iLv);
        strLvPre = sprintf('Lv%0d_',iLv-1);
        nBlocksLv = inputSize(1:2)./(stride.^iLv);

        if iLv == 1
            if isapndinout
                salsunLgraph = salsunLgraph.addLayers(...
                    [ imageInputLayer(inputSize,...
                    'Name',[prefix 'ImageInput'],...
                    'Normalization','none'),...
                    lsunIdentityLayer(...
                    'Name',[prefix strLv 'In'])
                    ]);
            else
                salsunLgraph = salsunLgraph.addLayers(...
                    lsunIdentityLayer(...
                    'Name',[prefix strLv 'In']));
            end
        else
            salsunLgraph = salsunLgraph.addLayers(...
                lsunIdentityLayer('Name',[prefix strLv 'In']));
            salsunLgraph = salsunLgraph.connectLayers(...
                [prefix strLvPre 'DcOut'],[prefix strLv 'In']);
        end
        salsunLgraph = salsunLgraph.addLayers(...
            lsunBlockDct2dLayer('Name',[prefix strLv 'E0'],...
            'Stride',stride,...
            'NumberOfComponents',nComponents));
        salsunLgraph = salsunLgraph.connectLayers(...
            [prefix strLv 'In'],[prefix strLv 'E0']);

        for iCmp = 1:nComponents
            if nComponents > 1
                strCmp = sprintf('Cmp%0d_',iCmp);
            else
                strCmp = 'Cmp1_';
            end
            compPrefix = [prefix strLv strCmp];

            % Initial rotation + its parameter estimator
            v0Name = [compPrefix 'V0'];
            estV0Prefix = [compPrefix 'V0_'];
            salsunLgraph = salsunLgraph.addLayers(...
                salsunInitialRotation2dLayer('Name',v0Name,...
                'Stride',stride,...
                'NumberOfBlocks',nBlocksLv,...
                'Device',device,...
                'DType',dtype));
            salsunLgraph = addEstimator(salsunLgraph,est,estV0Prefix,...
                'Initial',nDecs,logical(noDcLeakage(1)),nBlocksLv,stride,...
                'Analysis','V0');
            if nComponents > 1
                e0OutName = [prefix strLv 'E0/out' num2str(iCmp)];
            else
                e0OutName = [prefix strLv 'E0'];
            end
            salsunLgraph = salsunLgraph.connectLayers(...
                e0OutName,[v0Name '/x']);
            salsunLgraph = salsunLgraph.connectLayers(...
                e0OutName,[estV0Prefix 'Ext']);
            salsunLgraph = salsunLgraph.connectLayers(...
                [estV0Prefix 'Theta'],[v0Name '/theta']);

            lastName = v0Name;
            for iStage = 1:numel(stageSpecs)
                [salsunLgraph,lastName] = addAnalysisStage(salsunLgraph,...
                lastName,stageSpecs(iStage),...
                compPrefix,stride,nBlocksLv,nDecs,...
                est,logical(noDcLeakage(1)),device,dtype);
            end

            salsunLgraph = salsunLgraph.addLayers(...
                lsunChannelSeparation2dLayer('Name',[compPrefix 'Sp']));
            salsunLgraph = salsunLgraph.connectLayers(lastName,[compPrefix 'Sp']);
        end

        if nComponents > 1
            salsunLgraph = salsunLgraph.addLayers(...
                depthConcatenationLayer(nComponents,'Name',[prefix strLv 'AcOut']));
            salsunLgraph = salsunLgraph.addLayers(...
                depthConcatenationLayer(nComponents,'Name',[prefix strLv 'DcOut']));
            for iCmp = 1:nComponents
                strCmp = sprintf('Cmp%0d_',iCmp);
                salsunLgraph = salsunLgraph.connectLayers(...
                    [prefix strLv strCmp 'Sp/ac'],[prefix strLv 'AcOut/in' num2str(iCmp)]);
                salsunLgraph = salsunLgraph.connectLayers(...
                    [prefix strLv strCmp 'Sp/dc'],[prefix strLv 'DcOut/in' num2str(iCmp)]);
            end
        else
            salsunLgraph = salsunLgraph.addLayers(...
                lsunIdentityLayer('Name',[prefix strLv 'AcOut']));
            salsunLgraph = salsunLgraph.addLayers(...
                lsunIdentityLayer('Name',[prefix strLv 'DcOut']));
            salsunLgraph = salsunLgraph.connectLayers(...
                [prefix strLv strCmp 'Sp/ac'],[prefix strLv 'AcOut']);
            salsunLgraph = salsunLgraph.connectLayers(...
                [prefix strLv strCmp 'Sp/dc'],[prefix strLv 'DcOut']);
        end
    end
end

%% Synthesis
if isSynthesizer
    for iLv = nLevels:-1:1
        strLv = sprintf('Lv%0d_',iLv);
        strLvPre = sprintf('Lv%0d_',iLv+1);
        nBlocksLv = inputSize(1:2)./(stride.^iLv);

        if nComponents > 1
            salsunLgraph = salsunLgraph.addLayers(...
                lsunComponentSeparation2dLayer(nComponents,'Name',[prefix strLv 'AcIn']));
            salsunLgraph = salsunLgraph.addLayers(...
                lsunComponentSeparation2dLayer(nComponents,'Name',[prefix strLv 'DcIn']));
        else
            salsunLgraph = salsunLgraph.addLayers(...
                lsunIdentityLayer('Name',[prefix strLv 'AcIn']));
            salsunLgraph = salsunLgraph.addLayers(...
                lsunIdentityLayer('Name',[prefix strLv 'DcIn']));
        end

        if iLv < nLevels
            % DcIn is fed internally from the level above's reconstruction.
            salsunLgraph = salsunLgraph.connectLayers(...
                [prefix strLvPre 'Out'],[prefix strLv 'DcIn']);
        end

        if isapndinout && ~isAnalyzer
            salsunLgraph = salsunLgraph.addLayers(...
                imageInputLayer([nBlocksLv nComponents*(nDecs-1)],...
                'Name',[prefix strLv 'Ac feature Input'],'Normalization','none'));
            salsunLgraph = salsunLgraph.connectLayers(...
                [prefix strLv 'Ac feature Input'],[prefix strLv 'AcIn']);
            if iLv == nLevels
                salsunLgraph = salsunLgraph.addLayers(...
                    imageInputLayer([nBlocksLv nComponents],...
                    'Name',[prefix strLv 'Dc feature Input'],'Normalization','none'));
                salsunLgraph = salsunLgraph.connectLayers(...
                    [prefix strLv 'Dc feature Input'],[prefix strLv 'DcIn']);
            end
        end

        for iCmp = 1:nComponents
            if nComponents > 1
                strCmp = sprintf('Cmp%0d_',iCmp);
            else
                strCmp = 'Cmp1_';
            end
            compPrefix = [prefix strLv strCmp];

            salsunLgraph = salsunLgraph.addLayers(...
                lsunChannelConcatenation2dLayer('Name',[compPrefix 'Cn']));
            if nComponents > 1
                salsunLgraph = salsunLgraph.connectLayers(...
                    [prefix strLv 'AcIn/out' num2str(iCmp)],[compPrefix 'Cn/ac']);
                salsunLgraph = salsunLgraph.connectLayers(...
                    [prefix strLv 'DcIn/out' num2str(iCmp)],[compPrefix 'Cn/dc']);
            else
                salsunLgraph = salsunLgraph.connectLayers(...
                    [prefix strLv 'AcIn'],[compPrefix 'Cn/ac']);
                salsunLgraph = salsunLgraph.connectLayers(...
                    [prefix strLv 'DcIn'],[compPrefix 'Cn/dc']);
            end

            lastName = [compPrefix 'Cn'];
            for iStage = numel(stageSpecs):-1:1
                [salsunLgraph,lastName] = addSynthesisStage(salsunLgraph,lastName,...
                    stageSpecs(iStage),compPrefix,stride,nBlocksLv,nDecs,...
                    est,logical(noDcLeakage(2)),device,dtype,thetaMode);
            end

            v0sName = [compPrefix 'V0~'];
            salsunLgraph = salsunLgraph.addLayers(...
                salsunFinalRotation2dLayer('Name',v0sName,...
                'Stride',stride,'NumberOfBlocks',nBlocksLv,...
                'Device',device,'DType',dtype));
            salsunLgraph = salsunLgraph.connectLayers(lastName,[v0sName '/x']);
            if strcmp(thetaMode,'Reuse')
                % Reuse the analysis side's Initial-rotation estimator
                % output directly, instead of building an independent
                % synthesis-side estimator, so this stage exactly
                % inverts what the analysis side's V0 did (guaranteeing
                % unitarity/perfect reconstruction when no dimension
                % reduction occurs in between).
                estV0Prefix = [compPrefix 'V0_'];
                salsunLgraph = salsunLgraph.connectLayers(...
                    [estV0Prefix 'Theta'],[v0sName '/theta']);
            else
                estV0sPrefix = [compPrefix 'V0~_'];
                salsunLgraph = addEstimator(salsunLgraph,est,estV0sPrefix,...
                    'Initial',nDecs,logical(noDcLeakage(2)),nBlocksLv,stride,...
                    'Synthesis','V0');
                salsunLgraph = salsunLgraph.connectLayers(lastName,[estV0sPrefix 'Ext']);
                salsunLgraph = salsunLgraph.connectLayers(...
                    [estV0sPrefix 'Theta'],[v0sName '/theta']);
            end
        end

        salsunLgraph = salsunLgraph.addLayers(...
            lsunBlockIdct2dLayer('Name',[prefix strLv 'E0~'],...
            'Stride',stride,'NumberOfComponents',nComponents));
        for iCmp = 1:nComponents
            if nComponents > 1
                strCmp = sprintf('Cmp%0d_',iCmp);
                e0sInName = [prefix strLv 'E0~/in' num2str(iCmp)];
            else
                strCmp = 'Cmp1_';
                e0sInName = [prefix strLv 'E0~'];
            end
            compPrefix = [prefix strLv strCmp];
            salsunLgraph = salsunLgraph.connectLayers(...
                [compPrefix 'V0~'],e0sInName);
        end
        salsunLgraph = salsunLgraph.addLayers(...
            lsunIdentityLayer('Name',[prefix strLv 'Out']));
        salsunLgraph = salsunLgraph.connectLayers(...
            [prefix strLv 'E0~'],[prefix strLv 'Out']);
    end
end

%% Connect analyzer and synthesizer directly (Whole mode)
if isAnalyzer && isSynthesizer
    strLv = sprintf('Lv%0d_',nLevels);
    salsunLgraph = salsunLgraph.connectLayers(...
        [prefix strLv 'DcOut'],[prefix strLv 'DcIn']);
    for iLv = nLevels:-1:1
        strLv = sprintf('Lv%0d_',iLv);
        salsunLgraph = salsunLgraph.connectLayers(...
            [prefix strLv 'AcOut'],[prefix strLv 'AcIn']);
    end
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function s = mkstage(name,atomDir,atomTarget,synAtomDir,synAtomTarget)
s = struct('name',name,'atomDir',atomDir,'atomTarget',atomTarget,...
    'synAtomDir',synAtomDir,'synAtomTarget',synAtomTarget);
end

%%%

function lgraph = addEstimator(lgraph,est,estPrefix,targetType,nDecs,...
    noDcLeakage,nBlocks,stride,side,stageName)
% Adds the angle estimator est with the interface described above. The
% arguments of the builder come after EstimatorOptions, so they cannot
% be overridden by them.
lgraph = est.Function(lgraph,est.Options{:},...
    'Prefix',estPrefix,'TargetType',targetType,...
    'NumberOfChannels',nDecs,'NoDcLeakage',noDcLeakage,...
    'NumberOfBlocks',nBlocks,'Stride',stride,...
    'Side',side,'StageName',stageName);
layerNames = {lgraph.Layers.Name};
for name = {[estPrefix 'Ext'],[estPrefix 'Theta']}
    if ~any(strcmp(layerNames,name{1}))
        throw(MException('SaLsunLayer:InvalidEstimator',...
            '%s : The estimator %s must add a layer named %s.',...
            stageName,func2str(est.Function),name{1}))
    end
end
end

%%%

function [lgraph,lastName] = addAnalysisStage(lgraph,lastName,stage,prefix,...
    stride,nBlocks,nDecs,est,noDcLeakage,device,dtype)
import tansacnet.lsun.*
import tansacnet.salsun.*

if stage.atomDir == "Right"
    dirname = 'r';
elseif stage.atomDir == "Left"
    dirname = 'l';
elseif stage.atomDir == "Up"
    dirname = 'u';
elseif stage.atomDir == "Down"
    dirname = 'd';
else
error('Unknown direction: %s',stage.atomDir);
end
if stage.atomTarget == "Difference"
    targetname = 'd';
elseif stage.atomTarget == "Sum"
    targetname = 's';
else
error('Unknown target: %s',stage.atomTarget);
end
atomName = [prefix 'Q' stage.name dirname targetname];
lgraph = lgraph.addLayers(lsunAtomExtension2dLayer('Name',atomName,...
    'Stride',stride,'Direction',stage.atomDir,'TargetChannels',stage.atomTarget));
lgraph = lgraph.connectLayers(lastName,atomName);

rotName = [prefix 'V' stage.name '_'];
lgraph = lgraph.addLayers(salsunIntermediateRotation2dLayer('Name',rotName,...
    'Stride',stride,'NumberOfBlocks',nBlocks,'Mode','Analysis','Mus',-1,...
    'Device',device,'DType',dtype));
lgraph = addEstimator(lgraph,est,rotName,'Intermediate',nDecs,...
    noDcLeakage,nBlocks,stride,'Analysis',stage.name);
lgraph = lgraph.connectLayers(atomName,[rotName '/x']);
lgraph = lgraph.connectLayers(atomName,[rotName 'Ext']);
lgraph = lgraph.connectLayers([rotName 'Theta'],[rotName '/theta']);

lastName = rotName;
end

%%%

function [lgraph,lastName] = addSynthesisStage(lgraph,lastName,stage,prefix,...
    stride,nBlocks,nDecs,est,noDcLeakage,device,dtype,thetaMode)
import tansacnet.lsun.*
import tansacnet.salsun.*

if stage.atomDir == "Right"
    dirname = 'r';
elseif stage.atomDir == "Left"
    dirname = 'l';
elseif stage.atomDir == "Up"
    dirname = 'u';
elseif stage.atomDir == "Down"
    dirname = 'd';
else
error('Unknown direction: %s',stage.atomDir);
end
if stage.atomTarget == "Difference"
    targetname = 'd';
elseif stage.atomTarget == "Sum"
    targetname = 's';
else
error('Unknown target: %s',stage.atomTarget);
end
rotName = [prefix 'V' stage.name '~'];
lgraph = lgraph.addLayers(salsunIntermediateRotation2dLayer('Name',rotName,...
    'Stride',stride,'NumberOfBlocks',nBlocks,'Mode','Synthesis','Mus',-1,...
    'Device',device,'DType',dtype));
lgraph = lgraph.connectLayers(lastName,[rotName '/x']);
if strcmp(thetaMode,'Reuse')
    % Reuse the analysis side's estimator output for this same stage
    % directly, instead of building an independent synthesis-side
    % estimator, so this stage exactly inverts the analysis side's
    % corresponding stage (guaranteeing unitarity/perfect
    % reconstruction when no dimension reduction occurs in between).
    estPrefix = [prefix 'V' stage.name '_Theta'];
    lgraph = lgraph.connectLayers(estPrefix,[rotName '/theta']);
else
    estPrefix = [prefix 'V' stage.name '~_'];
    lgraph = addEstimator(lgraph,est,estPrefix,'Intermediate',nDecs,...
        noDcLeakage,nBlocks,stride,'Synthesis',stage.name);
    lgraph = lgraph.connectLayers(lastName,[estPrefix 'Ext']);
    lgraph = lgraph.connectLayers([estPrefix 'Theta'],[rotName '/theta']);
end

atomName = [prefix 'Q' stage.name dirname targetname '~'];
lgraph = lgraph.addLayers(lsunAtomExtension2dLayer('Name',atomName,...
    'Stride',stride,'Direction',stage.synAtomDir,'TargetChannels',stage.synAtomTarget));
lgraph = lgraph.connectLayers(rotName,atomName);

lastName = atomName;
end
