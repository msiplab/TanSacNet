classdef fcn_createsalsunlgraph2dTestCase < matlab.unittest.TestCase
    %FCN_CREATESALSUNLGRAPH2DTESTCASE
    %
    %   Tests of the angle estimator interface of fcn_createsalsunlgraph2d
    %   (EstimatorFunction and EstimatorOptions).
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
        ovlpFactor = struct('o1x1',[1 1],'o3x3',[3 3],'o3x5',[3 5]);
        noDcLeakage = struct('false',false,'true',true);
    end

    methods (Test)

        function testDefaultEstimatorExplicitly(testCase)
            import tansacnet.salsun.*
            args = {'InputSize',[16 16],'Stride',[2 2],...
                'OverlappingFactor',[3 3],'Mode','Whole','Device','cpu'};

            % Expected values
            lgExpctd = fcn_createsalsunlgraph2d([],args{:});

            % Actual values
            lgActual = fcn_createsalsunlgraph2d([],args{:},...
                'EstimatorFunction',@fcn_createparamestimator2dlgraph);

            % Evaluation
            testCase.verifyEqual({lgActual.Layers.Name},{lgExpctd.Layers.Name});
            testCase.verifyEqual(lgActual.Connections,lgExpctd.Connections);
        end

        function testEstimatorOptionsOfDefaultEstimator(testCase)
            import tansacnet.salsun.*
            args = {'InputSize',[16 16],'Stride',[2 2],...
                'OverlappingFactor',[3 3],'Mode','Analyzer','Device','cpu'};

            % Expected values
            netExpctd = dlnetwork(fcn_createsalsunlgraph2d([],args{:},...
                'Width',1,'NumberOfResidualBlocks',2),'Initialize',false);

            % Actual values (EstimatorOptions come after the legacy options)
            netActual = dlnetwork(fcn_createsalsunlgraph2d([],args{:},...
                'Width',3,'EstimatorOptions',...
                {'Width',1,'NumberOfResidualBlocks',2}),'Initialize',false);

            % Evaluation
            testCase.verifyEqual(netActual.Learnables.Layer,netExpctd.Learnables.Layer);
            testCase.verifyEqual(cellfun(@size,netActual.Learnables.Value,'UniformOutput',false),...
                cellfun(@size,netExpctd.Learnables.Value,'UniformOutput',false));
        end

        function testConstantEstimatorPerfectReconstruction(testCase,...
                ovlpFactor,noDcLeakage)
            import tansacnet.salsun.*
            inputSize = [16 24];
            stride = [2 2];
            nSamples = 2;
            lgraph = fcn_createsalsunlgraph2d([],...
                'InputSize',inputSize,'Stride',stride,...
                'OverlappingFactor',ovlpFactor,...
                'NumberOfVanishingMoments',noDcLeakage,...
                'Mode','Whole','ThetaMode','Reuse','Device','cpu',...
                'EstimatorFunction',@fcn_createconstestimatorlgraph);
            net = dlnetwork(lgraph);
            rng(0)
            net = dlupdate(@(w) w + randn(size(w),'like',w),net);
            x = dlarray(rand([inputSize 1 nSamples],'single'),'SSCB');

            % Expected values
            yExpctd = extractdata(x);

            % Actual values
            yActual = extractdata(predict(net,x));

            % Evaluation
            testCase.verifyEqual(yActual,yExpctd,'AbsTol',single(1e-5));
            testCase.verifyTrue(all(net.Learnables.Parameter == "Angles"));
            nStages = (ovlpFactor(1)-1)+(ovlpFactor(2)-1);
            testCase.verifyEqual(height(net.Learnables),1+nStages);
        end

        function testConstantEstimatorIsAdaptiveOnlyByLocation(testCase)
            % With constant angles, the analysis is a linear transform
            import tansacnet.salsun.*
            inputSize = [16 16];
            lgraph = fcn_createsalsunlgraph2d([],...
                'InputSize',inputSize,'Stride',[2 2],...
                'OverlappingFactor',[3 3],'NumberOfVanishingMoments',true,...
                'Mode','Analyzer','Device','cpu',...
                'EstimatorFunction',@fcn_createconstestimatorlgraph);
            net = dlnetwork(lgraph);
            rng(1)
            net = dlupdate(@(w) w + randn(size(w),'like',w),net);
            x1 = dlarray(rand([inputSize 1],'single'),'SSCB');
            x2 = dlarray(rand([inputSize 1],'single'),'SSCB');

            % Expected values
            [ac1,dc1] = predict(net,x1);
            [ac2,dc2] = predict(net,x2);
            acExpctd = extractdata(ac1+2*ac2);
            dcExpctd = extractdata(dc1+2*dc2);

            % Actual values
            [ac,dc] = predict(net,x1+2*x2);

            % Evaluation
            testCase.verifyEqual(extractdata(ac),acExpctd,'AbsTol',single(1e-5));
            testCase.verifyEqual(extractdata(dc),dcExpctd,'AbsTol',single(1e-5));
        end

        function testEstimatorPerStage(testCase)
            % One function can choose the estimator by TargetType
            import tansacnet.salsun.*
            lgraph = fcn_createsalsunlgraph2d([],...
                'InputSize',[16 16],'Stride',[2 2],...
                'OverlappingFactor',[3 3],'Mode','Whole','Device','cpu',...
                'EstimatorFunction',@localEstimatorPerStage);
            names = {lgraph.Layers.Name};
            classOf = @(name) class(lgraph.Layers(strcmp(names,name)));

            % Evaluation
            testCase.verifyEqual(classOf('Lv1_Cmp1_V0_Theta'),...
                'tansacnet.salsun.salsunConstantAngleLayer');
            testCase.verifyEqual(classOf('Lv1_Cmp1_V0~_Theta'),...
                'tansacnet.salsun.salsunConstantAngleLayer');
            testCase.verifyEqual(classOf('Lv1_Cmp1_Vh1_Theta'),...
                'tansacnet.salsun.salsunAngleEstimatorOutput2dLayer');
            testCase.verifyEqual(classOf('Lv1_Cmp1_Vv2~_Theta'),...
                'tansacnet.salsun.salsunAngleEstimatorOutput2dLayer');
        end

        function testContextPassedToEstimator(testCase)
            import tansacnet.salsun.*
            stride = [2 2];
            inputSize = [16 24];
            lgraph = fcn_createsalsunlgraph2d([],...
                'InputSize',inputSize,'Stride',stride,...
                'OverlappingFactor',[3 3],'Mode','Whole','Device','cpu',...
                'EstimatorFunction',@localEstimatorRecordingContext);
            names = {lgraph.Layers.Name};
            descOf = @(name) lgraph.Layers(strcmp(names,name)).Description;

            % Evaluation
            nBlocks = inputSize./stride;
            testCase.verifyEqual(descOf('Lv1_Cmp1_V0_Ext'),...
                sprintf('Initial|Analysis|V0|%d %d|%d %d',nBlocks,stride));
            testCase.verifyEqual(descOf('Lv1_Cmp1_Vh2~_Ext'),...
                sprintf('Intermediate|Synthesis|h2|%d %d|%d %d',nBlocks,stride));
        end

        function testInvalidEstimator(testCase)
            import tansacnet.salsun.*
            testCase.verifyError(...
                @() fcn_createsalsunlgraph2d([],...
                    'InputSize',[16 16],'Stride',[2 2],'Device','cpu',...
                    'EstimatorFunction',@(lg,varargin) lg),...
                'SaLsunLayer:InvalidEstimator');
        end

    end

end

function lgraph = localEstimatorPerStage(lgraph,varargin)
import tansacnet.salsun.*
p = inputParser;
p.KeepUnmatched = true;
addParameter(p,'TargetType','Initial')
parse(p,varargin{:})
if strcmp(p.Results.TargetType,'Initial')
    lgraph = fcn_createconstestimatorlgraph(lgraph,varargin{:});
else
    lgraph = fcn_createparamestimator2dlgraph(lgraph,varargin{:});
end
end

function lgraph = localEstimatorRecordingContext(lgraph,varargin)
% The constant estimator, with the context recorded in the Description
% of the Ext layer
import tansacnet.salsun.*
p = inputParser;
p.KeepUnmatched = true;
addParameter(p,'Prefix','')
addParameter(p,'TargetType','')
addParameter(p,'Side','')
addParameter(p,'StageName','')
addParameter(p,'NumberOfBlocks',[])
addParameter(p,'Stride',[])
parse(p,varargin{:})
lgraph = fcn_createconstestimatorlgraph(lgraph,varargin{:});
r = p.Results;
ext = functionLayer(@(X) X,'Name',[r.Prefix 'Ext'],...
    'Description',sprintf('%s|%s|%s|%d %d|%d %d',r.TargetType,r.Side,...
    r.StageName,r.NumberOfBlocks,r.Stride));
lgraph = lgraph.replaceLayer([r.Prefix 'Ext'],ext);
end
