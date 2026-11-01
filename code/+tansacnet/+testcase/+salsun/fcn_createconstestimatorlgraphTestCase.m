classdef fcn_createconstestimatorlgraphTestCase < matlab.unittest.TestCase
    %FCN_CREATECONSTESTIMATORLGRAPHTESTCASE
    %
    %   Tests of the constant angle estimator and fcn_salsunanglecounts.
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
        nChs = {4,8,16};
        nBlocks = {[3 4],[2 3 4]};
    end

    methods (Test)

        function testAngleCounts(testCase,nChs)
            import tansacnet.salsun.*

            % Initial with and without no DC leakage, intermediate
            [nFull,nPad] = fcn_salsunanglecounts('Initial',nChs,true);
            testCase.verifyEqual([nFull nPad],[(nChs-2)*nChs/4 nChs/2-1]);
            [nFull,nPad] = fcn_salsunanglecounts('Initial',nChs,false);
            testCase.verifyEqual([nFull nPad],[(nChs-2)*nChs/4 0]);
            [nFull,nPad] = fcn_salsunanglecounts('Intermediate',nChs,true);
            testCase.verifyEqual([nFull nPad],[(nChs-2)*nChs/8 0]);
        end

        function testInvalidAngleCounts(testCase)
            import tansacnet.salsun.*
            testCase.verifyError(@() fcn_salsunanglecounts('Initial',7,false),...
                'SaLsunLayer:InvalidNumberOfChannels');
            testCase.verifyError(@() fcn_salsunanglecounts('Final',8,false),'');
        end

        function testOutputOfConstantEstimator(testCase,nChs,nBlocks)
            % The estimator does not depend on the number of spatial
            % dimensions
            import tansacnet.salsun.*
            nSamples = 3;
            lgraph = fcn_createconstestimatorlgraph([],...
                'Prefix','Est_','TargetType','Initial',...
                'NumberOfChannels',nChs,'NoDcLeakage',true,...
                'NumberOfBlocks',nBlocks);
            names = {lgraph.Layers.Name};
            testCase.verifyEqual(sort(names),{'Est_Ext','Est_Theta'});
            layer = lgraph.Layers(strcmp(names,'Est_Theta'));
            [nFull,nPad] = fcn_salsunanglecounts('Initial',nChs,true);
            angles = randn(nFull-nPad,prod(nBlocks));
            layer.Angles = angles;
            X = randn([nChs nBlocks nSamples]);

            % Expected values
            thetaExpctd = repmat([zeros(nPad,prod(nBlocks)); angles],1,1,nSamples);

            % Actual values
            thetaActual = layer.predict(X);

            % Evaluation
            testCase.verifyEqual(thetaActual,thetaExpctd);
        end

        function testInvalidNumberOfBlocks(testCase)
            import tansacnet.salsun.*
            layer = salsunConstantAngleLayer('NumberOfBlocks',[3 4],...
                'NumberOfAngles',6);
            testCase.verifyError(@() layer.predict(randn(8,5,5)),...
                'SaLsunLayer:InvalidNumberOfBlocks');
        end

    end

end
