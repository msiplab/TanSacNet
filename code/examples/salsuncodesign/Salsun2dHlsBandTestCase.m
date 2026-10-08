classdef Salsun2dHlsBandTestCase < matlab.unittest.TestCase
    %SALSUN2DHLSBANDTESTCASE Test case for salsun2d_hls_band and salsun2d_band_frame
    %
    % The band design is compared with the reference salsun2d_infer_stream
    % and salsun2d_infer given the same statistics. With random parameters
    % the rounding differences between the loop implementation and the
    % vectorized reference reach about 1e-3 (see Salsun2dHlsTestCase), so
    % a tolerance of 1e-2 is used; with trained parameters (results/
    % none.mat, if present) the tolerance is 1e-4.
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

    methods (TestClassSetup)
        function addPaths(testCase)
            import matlab.unittest.fixtures.PathFixture
            here = fileparts(mfilename('fullpath'));
            testCase.applyFixture(PathFixture(fullfile(here,'..','..')));      % +tansacnet
            testCase.applyFixture(PathFixture(fullfile(here,'..','salsun')));  % maskLayer
        end
    end

    methods (Static, Access=private)
        function [mu,sigma] = channelStats(stats,params)
            % Per-channel statistics (NDec x NEst) from the per-feature
            % statistics of salsun2d_infer (the first nCh features are the
            % channels of the center block)
            L = salsun2d_hls_layout();
            ests = [params.V0.Estimator, params.Stages.Estimator];
            mu = zeros(L.NDec,L.NEst,'single');
            sigma = ones(L.NDec,L.NEst,'single');
            for k = 1:L.NEst
                ch = ests(k).Channels;
                mu(ch,k) = stats(k).Mu(1:numel(ch));
                sigma(ch,k) = stats(k).Sigma(1:numel(ch));
            end
        end

        function [params,w,x,tol] = testData(inputSize)
            here = fileparts(mfilename('fullpath'));
            trained = fullfile(here,'results','none.mat');
            if isfile(trained)
                R = load(trained,'params');
                params = R.params;
                u = salsun2d_wave_data();
                x = single(u(1:inputSize(1),1:inputSize(2),11));
                tol = single(1e-4);
            else
                rng(31)
                nCoefs = 2;
                coefMask = reshape([ones(nCoefs,1); zeros(16-nCoefs,1)],2,[]).';
                net = salsun2d_create_test_network(inputSize,coefMask(:));
                params = salsun2d_extract_params(net);
                x = rand(inputSize,'single');
                tol = single(1e-2);
            end
            w = salsun2d_pack_params(params);
        end
    end

    methods (Test)

        function testHaloIsStructuralReceptiveField(testCase)
            L = salsun2d_hls_layout();
            testCase.verifyEqual(L.Halo,7);
        end

        function testBandMatchesStreamReference(testCase)
            % One band of 2 block rows with its halo: 16 block rows in all

            [params,w,x,tol] = Salsun2dHlsBandTestCase.testData([64 32]);
            L = salsun2d_hls_layout();
            [~,~,~,~,stats] = salsun2d_infer(x,params);
            [mu,sigma] = Salsun2dHlsBandTestCase.channelStats(stats,params);

            % Expected values: the reference band-wise processing of the
            % same band, with the same statistics
            yRef = salsun2d_infer_stream(x,params,Statistics=stats,Halo=L.Halo,BandRows=2);

            % Actual values: x is the band (rows 8..9) with its halo (7 above, 7 below)
            bandRows = L.Halo + (1:2);
            pixelRows = (bandRows(1)-1)*4 + (1:8);
            [yActual,sum1,sum2] = salsun2d_hls_band(x,w,mu,sigma,int32(0));

            % Evaluation
            testCase.verifySize(yActual,[8 32]);
            testCase.verifyEqual(yActual,yRef(pixelRows,:),'AbsTol',tol);
            testCase.verifySize(sum1,[L.NDec L.NEst]);
            testCase.verifyTrue(all(sum2(:) >= 0));
            % The DC channel is not an input of any estimator
            testCase.verifyEqual(sum1(1,:),zeros(1,L.NEst,'single'));
        end

        function testFrameAssembly(testCase)
            % Bands of 4 block rows over 16 block rows (4 bands, no overlap).
            % The frame is narrow (4 block columns) because the MATLAB
            % execution of the HLS design takes about a minute per band.

            [params,w,x,tol] = Salsun2dHlsBandTestCase.testData([64 16]);
            [~,~,~,~,stats] = salsun2d_infer(x,params);
            [mu,sigma] = Salsun2dHlsBandTestCase.channelStats(stats,params);

            % Expected values: whole-frame reference with the same statistics
            [yExpctd,~,~,~,measured] = salsun2d_infer(x,params,Statistics=stats);
            [muExpctd,sigmaExpctd] = Salsun2dHlsBandTestCase.channelStats(measured,params);

            % Actual values
            [yActual,muActual,sigmaActual,bands] = salsun2d_band_frame(x,w,mu,sigma,4);

            % Evaluation
            testCase.verifyEqual(bands,[1 5 9 13]);
            testCase.verifyEqual(yActual,yExpctd,'AbsTol',tol);
            used = sigmaExpctd ~= 1;   % channels that are estimator inputs
            testCase.verifyEqual(muActual(used),muExpctd(used),'AbsTol',single(1e-5));
            testCase.verifyEqual(sigmaActual(used),sigmaExpctd(used),'RelTol',single(1e-3));
        end

        function testFrameAssemblyWithMovedLastBand(testCase)
            % Bands of 5 block rows over 16 block rows: the last band is
            % moved up to end at row 16

            [params,w,x,tol] = Salsun2dHlsBandTestCase.testData([64 16]);
            [~,~,~,~,stats] = salsun2d_infer(x,params);
            [mu,sigma] = Salsun2dHlsBandTestCase.channelStats(stats,params);
            [yExpctd,~,~,~,measured] = salsun2d_infer(x,params,Statistics=stats);
            [muExpctd,sigmaExpctd] = Salsun2dHlsBandTestCase.channelStats(measured,params);

            [yActual,muActual,sigmaActual,bands] = salsun2d_band_frame(x,w,mu,sigma,5);

            testCase.verifyEqual(bands,[1 6 11 12]);
            testCase.verifyEqual(yActual,yExpctd,'AbsTol',tol);
            % The repeated rows of the moved band are left out of the sums,
            % so the statistics are exact
            used = sigmaExpctd ~= 1;
            testCase.verifyEqual(muActual(used),muExpctd(used),'AbsTol',single(1e-5));
            testCase.verifyEqual(sigmaActual(used),sigmaExpctd(used),'RelTol',single(1e-3));
        end

    end

end
