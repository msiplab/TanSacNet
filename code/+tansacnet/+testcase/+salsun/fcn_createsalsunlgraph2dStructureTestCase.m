classdef fcn_createsalsunlgraph2dStructureTestCase < matlab.unittest.TestCase
    %FCN_CREATESALSUNLGRAPH2DSTRUCTURETESTCASE
    %
    %   Structure of the graphs created by fcn_createsalsunlgraph2d with the
    %   default estimator: the expected estimators, their connections to
    %   the rotation layers and their sizes are derived from the
    %   configuration (stride, overlapping factor, levels, components,
    %   mode, ThetaMode and the estimator options).
    %
    % Requirements: MATLAB R2022a
    %
    % Copyright (c) 2026, Shogo MURAMATSU
    %
    % All rights reserved.
    %
    % Contact address: Shogo MURAMATSU,
    %                Faculty of Engineering, Niigata University,
    %                8050 2-no-cho Ikarashi, Nishi-ku,
    %                Niigata, 950-2181, JAPAN
    %
    % http://msiplab.eng.niigata-u.ac.jp/

    properties (TestParameter)
        config = struct(...
            'whole_o1x1', struct('InputSize',[16 16],'Stride',[2 2],...
                'OverlappingFactor',[1 1],'NumberOfLevels',1,...
                'NumberOfComponents',1,'NumberOfVanishingMoments',false,...
                'Mode','Whole','ThetaMode','Recompute',...
                'NumberOfNeighborBlocks',[3 3],'NumberOfResidualBlocks',3,...
                'Width',2,'BottleneckSize',0),...
            'whole_reuse_s4x4', struct('InputSize',[32 32],'Stride',[4 4],...
                'OverlappingFactor',[3 3],'NumberOfLevels',1,...
                'NumberOfComponents',1,'NumberOfVanishingMoments',true,...
                'Mode','Whole','ThetaMode','Reuse',...
                'NumberOfNeighborBlocks',[3 3],'NumberOfResidualBlocks',3,...
                'Width',2,'BottleneckSize',0),...
            'analyzer_lv2_o3x5', struct('InputSize',[32 32],'Stride',[2 2],...
                'OverlappingFactor',[3 5],'NumberOfLevels',2,...
                'NumberOfComponents',1,'NumberOfVanishingMoments',true,...
                'Mode','Analyzer','ThetaMode','Recompute',...
                'NumberOfNeighborBlocks',[3 3],'NumberOfResidualBlocks',2,...
                'Width',2,'BottleneckSize',0),...
            'synthesizer_cmp2', struct('InputSize',[16 16],'Stride',[2 2],...
                'OverlappingFactor',[3 3],'NumberOfLevels',1,...
                'NumberOfComponents',2,'NumberOfVanishingMoments',false,...
                'Mode','Synthesizer','ThetaMode','Recompute',...
                'NumberOfNeighborBlocks',[3 3],'NumberOfResidualBlocks',3,...
                'Width',2,'BottleneckSize',0),...
            'whole_bottleneck', struct('InputSize',[32 32],'Stride',[4 4],...
                'OverlappingFactor',[3 3],'NumberOfLevels',1,...
                'NumberOfComponents',1,'NumberOfVanishingMoments',true,...
                'Mode','Whole','ThetaMode','Recompute',...
                'NumberOfNeighborBlocks',[5 5],'NumberOfResidualBlocks',2,...
                'Width',1,'BottleneckSize',16));
    end

    methods (Test)

        function testEstimatorsAndConnections(testCase,config)
            lgraph = createGraph(config);
            names = {lgraph.Layers.Name};
            conns = lgraph.Connections;
            expctd = expectedEstimators(config);

            % Exactly the expected estimators exist
            thetaNames = names(endsWith(names,'Theta'));
            testCase.verifyEqual(sort(thetaNames),...
                sort(cellfun(@(p) [p 'Theta'],{expctd.Prefix},...
                'UniformOutput',false)));

            % Each rotation takes its angles from the expected estimator
            for k = 1:numel(expctd)
                for r = expctd(k).Rotations
                    src = conns.Source(strcmp(conns.Destination,[r{1} '/theta']));
                    testCase.verifyEqual(src,{[expctd(k).Prefix 'Theta']},...
                        "Angles of " + r{1});
                end
            end

            % Each estimator watches the input of its rotation
            for k = 1:numel(expctd)
                src = conns.Source(strcmp(conns.Destination,[expctd(k).Prefix 'Ext']));
                rot = expctd(k).Rotations{1};
                testCase.verifyEqual(src,...
                    conns.Source(strcmp(conns.Destination,[rot '/x'])),...
                    "Input of " + expctd(k).Prefix + "Ext");
            end
        end

        function testSizesOfDefaultEstimators(testCase,config)
            lgraph = createGraph(config);
            names = {lgraph.Layers.Name};
            layerOf = @(name) lgraph.Layers(strcmp(names,name));
            nChs = prod(config.Stride);
            nNeighbors = prod(config.NumberOfNeighborBlocks);

            for est = expectedEstimators(config)
                prefix = est.Prefix;
                % Channels watched by the estimator
                if strcmp(est.TargetType,'Initial')
                    nWatched = nChs - est.NoDcLeakage;
                else
                    nWatched = nChs/2;
                end
                nFeat = nWatched*nNeighbors;
                if config.BottleneckSize > 0
                    nInner = config.BottleneckSize;
                    proj = layerOf([prefix 'Proj']);
                    testCase.verifySize(proj.Wp,[nInner nFeat]);
                else
                    nInner = nFeat;
                    testCase.verifyEmpty(layerOf([prefix 'Proj']));
                end
                % Residual blocks
                nHidden = round(config.Width*nInner);
                for iBlk = 1:config.NumberOfResidualBlocks
                    blk = layerOf(sprintf('%sResBlk%d',prefix,iBlk));
                    testCase.verifySize(blk.W1,[nHidden nInner],prefix);
                    testCase.verifySize(blk.W2,[nInner nHidden],prefix);
                end
                testCase.verifyEmpty(layerOf(sprintf('%sResBlk%d',prefix,...
                    config.NumberOfResidualBlocks+1)));
                % Output layer
                [nFull,nPad] = tansacnet.salsun.fcn_salsunanglecounts(...
                    est.TargetType,nChs,est.NoDcLeakage);
                out = layerOf([prefix 'Theta']);
                testCase.verifySize(out.Wo,[nFull-nPad nInner],prefix);
                testCase.verifyEqual(out.NumberOfZeroPadAngles,nPad,prefix);
                ext = layerOf([prefix 'Ext']);
                testCase.verifyEqual(numel(ext.Channels),nWatched,prefix);
            end
        end

    end

end

function lgraph = createGraph(config)
args = namedargs2cell(config);
lgraph = tansacnet.salsun.fcn_createsalsunlgraph2d([],args{:},'Device','cpu');
end

function expctd = expectedEstimators(config)
% Estimators expected from the naming of fcn_createsalsunlgraph2d: their
% prefixes, the rotations they drive, their target types and whether no
% DC leakage applies
stageNames = {};
for iOrder = 1:config.OverlappingFactor(2)-1
    stageNames{end+1} = sprintf('h%d',iOrder); %#ok<AGROW>
end
for iOrder = 1:config.OverlappingFactor(1)-1
    stageNames{end+1} = sprintf('v%d',iOrder); %#ok<AGROW>
end
isAnalyzer = any(strcmp(config.Mode,{'Whole','Analyzer'}));
isSynthesizer = any(strcmp(config.Mode,{'Whole','Synthesizer'}));
isReuse = strcmp(config.ThetaMode,'Reuse');
noDc = logical(config.NumberOfVanishingMoments);

expctd = struct('Prefix',{},'Rotations',{},'TargetType',{},'NoDcLeakage',{});
for iLv = 1:config.NumberOfLevels
    for iCmp = 1:config.NumberOfComponents
        cmp = sprintf('Lv%d_Cmp%d_',iLv,iCmp);
        if isAnalyzer
            rotV0 = {[cmp 'V0']};
            if isSynthesizer && isReuse
                rotV0{end+1} = [cmp 'V0~']; %#ok<AGROW>
            end
            expctd(end+1) = struct('Prefix',[cmp 'V0_'],'Rotations',{rotV0},...
                'TargetType','Initial','NoDcLeakage',noDc); %#ok<AGROW>
            for s = stageNames
                rots = {[cmp 'V' s{1} '_']};
                if isSynthesizer && isReuse
                    rots{end+1} = [cmp 'V' s{1} '~']; %#ok<AGROW>
                end
                expctd(end+1) = struct('Prefix',[cmp 'V' s{1} '_'],...
                    'Rotations',{rots},'TargetType','Intermediate',...
                    'NoDcLeakage',noDc); %#ok<AGROW>
            end
        end
        if isSynthesizer && ~isReuse
            expctd(end+1) = struct('Prefix',[cmp 'V0~_'],...
                'Rotations',{{[cmp 'V0~']}},'TargetType','Initial',...
                'NoDcLeakage',noDc); %#ok<AGROW>
            for s = stageNames
                expctd(end+1) = struct('Prefix',[cmp 'V' s{1} '~_'],...
                    'Rotations',{{[cmp 'V' s{1} '~']}},...
                    'TargetType','Intermediate','NoDcLeakage',noDc); %#ok<AGROW>
            end
        end
    end
end
end
