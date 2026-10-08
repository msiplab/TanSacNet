classdef Salsun2dSequenceTestCase < matlab.unittest.TestCase
    %SALSUN2DSEQUENCETESTCASE Test case for salsun2d_base_field, salsun2d_infer_sequence
    %
    % Also covers the Statistics option of salsun2d_infer.
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

    properties (TestParameter)
        baseField = {'batch','iir'};
        scope = {'dc','full'};
        statistics = {'previous','ema','fir2'};
    end

    properties (Constant)
        FrameSize = [24 24];
    end

    methods (TestClassSetup)
        function addPaths(testCase)
            import matlab.unittest.fixtures.PathFixture
            here = fileparts(mfilename('fullpath'));
            testCase.applyFixture(PathFixture(fullfile(here,'..','..')));      % +tansacnet
            testCase.applyFixture(PathFixture(fullfile(here,'..','salsun')));  % maskLayer
        end
    end

    methods (Static, Access=private)
        function m = blockMeanField(frame,stride)
            [szy,szx] = size(frame);
            blocks = reshape(frame,stride(1),szy/stride(1),stride(2),szx/stride(2));
            m = reshape(repmat(mean(mean(blocks,1),3),[stride(1) 1 stride(2) 1]),szy,szx);
        end
    end

    methods (Test)

        function testBaseFieldIsAdditiveAndCausal(testCase)

            rng(21)
            u = rand([testCase.FrameSize 4],'single');

            % Actual values
            [ufDc,bDc] = salsun2d_base_field(u,Method='iir',Rho=0.5,Scope='dc');
            [ufFull,bFull] = salsun2d_base_field(u,Method='iir',Rho=0.5,Scope='full');
            [ufBatch,bBatch] = salsun2d_base_field(u,Method='batch',Scope='full');
            [ufNone,bNone] = salsun2d_base_field(u,Method='none');

            % Evaluation: u = uf + b
            testCase.verifyEqual(ufDc + bDc,u,'AbsTol',single(1e-6));
            testCase.verifyEqual(ufFull + bFull,u,'AbsTol',single(1e-6));
            testCase.verifyEqual(ufBatch + bBatch,u,'AbsTol',single(1e-6));
            testCase.verifyEqual(ufNone,u);
            testCase.verifyEqual(bNone,zeros(size(u),'single'));
            % Causal: frame 2 uses the base of frame 1 only, frame 3 the IIR of frames 1 and 2
            testCase.verifyEqual(bFull(:,:,1),u(:,:,1));
            testCase.verifyEqual(bFull(:,:,2),u(:,:,1));
            testCase.verifyEqual(bFull(:,:,3),0.5*u(:,:,1) + 0.5*u(:,:,2),'AbsTol',single(1e-6));
            testCase.verifyEqual(bDc(:,:,2), ...
                Salsun2dSequenceTestCase.blockMeanField(u(:,:,1),[4 4]),'AbsTol',single(1e-6));
            testCase.verifyEqual(bBatch(:,:,3),mean(u,3),'AbsTol',single(1e-6));
        end

        function testSequenceEqualsFrameByFrame(testCase)

            rng(22)
            net = salsun2d_create_test_network(testCase.FrameSize);
            params = salsun2d_extract_params(net);
            u = rand([testCase.FrameSize 3],'single');

            % Expected values
            yExpctd = zeros(size(u),'single');
            for t = 1:size(u,3)
                yExpctd(:,:,t) = salsun2d_infer(u(:,:,t),params);
            end

            % Actual values
            yActual = salsun2d_infer_sequence(u,params);

            % Evaluation
            testCase.verifyEqual(yActual,yExpctd);
        end

        function testPerfectReconstructionWithBaseField(testCase,baseField,scope)

            rng(23)
            net = salsun2d_create_test_network(testCase.FrameSize);   % no mask
            params = salsun2d_extract_params(net);
            u = rand([testCase.FrameSize 4],'single');

            % Actual values
            yActual = salsun2d_infer_sequence(u,params,BaseField=baseField,Scope=scope,Rho=0.7);

            % Evaluation
            testCase.verifyEqual(yActual,u,'AbsTol',single(1e-4));
        end

        function testGivenImageStatisticsReproduceImageMode(testCase)

            rng(24)
            net = salsun2d_create_test_network(testCase.FrameSize);
            params = salsun2d_extract_params(net);
            x = rand(testCase.FrameSize,'single');

            % Expected values
            [yExpctd,~,thExpctd,~,stats] = salsun2d_infer(x,params);

            % Actual values
            [yActual,~,thActual,~,statsAgain] = salsun2d_infer(x,params,Statistics=stats);

            % Evaluation
            testCase.verifyEqual(yActual,yExpctd);
            testCase.verifyEqual(thActual,thExpctd);
            testCase.verifyEqual(statsAgain,stats);
            testCase.verifyNumElements(stats,5);
            testCase.verifySize(stats(1).Mu,[135 1]);
            testCase.verifySize(stats(2).Sigma,[72 1]);
        end

        function testStatisticsModes(testCase,statistics)

            rng(25)
            nCoefs = 2;
            coefMask = reshape([ones(nCoefs,1); zeros(16-nCoefs,1)],2,[]).';
            net = salsun2d_create_test_network(testCase.FrameSize,coefMask(:));
            params = salsun2d_extract_params(net);
            u = rand([testCase.FrameSize 4],'single');

            % Expected values: the first frame falls back to image statistics
            yImage = salsun2d_infer_sequence(u,params);

            % Actual values
            [yActual,info] = salsun2d_infer_sequence(u,params,Statistics=statistics,StatsRho=0.5);

            % Evaluation
            testCase.verifySize(yActual,size(u));
            testCase.verifyTrue(all(isfinite(yActual(:))));
            testCase.verifyEqual(yActual(:,:,1),yImage(:,:,1));
            testCase.verifyNumElements(info.Statistics,4);
            % The later frames use other statistics, so they differ
            testCase.verifyNotEqual(yActual(:,:,end),yImage(:,:,end));
        end


        function testStreamEqualsWholeFrameWithFullHalo(testCase)
            % With a halo covering the structural receptive field (6) and
            % the synthesis (1), the band-wise result equals the whole
            % frame, given the same statistics

            rng(26)
            nCoefs = 2;
            coefMask = reshape([ones(nCoefs,1); zeros(16-nCoefs,1)],2,[]).';
            net = salsun2d_create_test_network([64 64],coefMask(:));   % 16 block rows
            params = salsun2d_extract_params(net);
            x = rand(64,64,'single');

            % Expected values
            [yExpctd,cExpctd,~,~,stats] = salsun2d_infer(x,params);

            % Actual values
            [yActual,cActual,statsActual] = salsun2d_infer_stream(x,params, ...
                Statistics=stats,Halo=7,BandRows=2);

            % Evaluation
            testCase.verifyEqual(yActual,yExpctd,'AbsTol',single(1e-5));
            testCase.verifyEqual(cActual,cExpctd,'AbsTol',single(1e-5));
            for k = 1:numel(stats)
                testCase.verifyEqual(statsActual(k).Mu,stats(k).Mu,'AbsTol',single(1e-5));
                testCase.verifyEqual(statsActual(k).Sigma,stats(k).Sigma,'RelTol',single(1e-4));
            end
        end

        function testStreamWithSmallHaloIsClose(testCase)
            % A small halo truncates the receptive field: the result is
            % close but not equal (random parameters leak strongly)

            rng(27)
            nCoefs = 2;
            coefMask = reshape([ones(nCoefs,1); zeros(16-nCoefs,1)],2,[]).';
            net = salsun2d_create_test_network([64 64],coefMask(:));
            params = salsun2d_extract_params(net);
            x = rand(64,64,'single');
            [yExpctd,~,~,~,stats] = salsun2d_infer(x,params);

            yActual = salsun2d_infer_stream(x,params,Statistics=stats,Halo=2,BandRows=4);

            testCase.verifySize(yActual,size(x));
            testCase.verifyTrue(all(isfinite(yActual(:))));
            testCase.verifyNotEqual(yActual,yExpctd);
        end

        function testSequenceStreaming(testCase)

            rng(28)
            nCoefs = 2;
            coefMask = reshape([ones(nCoefs,1); zeros(16-nCoefs,1)],2,[]).';
            net = salsun2d_create_test_network([32 32],coefMask(:));
            params = salsun2d_extract_params(net);
            u = rand(32,32,3,'single');

            % Expected values: whole-frame processing with the same statistics
            yWhole = salsun2d_infer_sequence(u,params,Statistics='previous');

            % Actual values: full halo (8 block rows), so identical
            yActual = salsun2d_infer_sequence(u,params,Statistics='previous',Halo=8,BandRows=2);

            % Evaluation. Frames 1 and 2 use the same statistics (frame 1
            % is processed as a whole in both). Frame 3 uses statistics
            % measured by the band-wise processing of frame 2, which
            % differ by rounding (about 2e-7 relative) from those of the
            % whole frame; with random parameters this is amplified to
            % about 1e-3 in the output (no such amplification with
            % trained parameters).
            testCase.verifyEqual(yActual(:,:,1:2),yWhole(:,:,1:2),'AbsTol',single(1e-5));
            testCase.verifyEqual(yActual(:,:,3),yWhole(:,:,3),'AbsTol',single(1e-2));
            testCase.verifyError(@() salsun2d_infer_sequence(u,params,Halo=2), ...
                'salsun2d_infer_sequence:imageStatistics');
        end

    end

end
