classdef Salsun2dHlsTestCase < matlab.unittest.TestCase
    %SALSUN2DHLSTESTCASE Test case for salsun2d_hls and salsun2d_pack_params
    %
    % Compares the HLS version salsun2d_hls with the reference
    % salsun2d_infer on a network with randomly perturbed parameters (see
    % salsun2d_create_test_network). With the coefficient mask, both are
    % judged by their error from the double-precision reference, because
    % rounding errors in the estimated angles grow stage by stage.
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
        inputSize = {[32 32],[24 40]};
    end

    methods (TestClassSetup)
        function addPaths(testCase)
            import matlab.unittest.fixtures.PathFixture
            here = fileparts(mfilename('fullpath'));
            testCase.applyFixture(PathFixture(fullfile(here,'..','..')));      % +tansacnet
            testCase.applyFixture(PathFixture(fullfile(here,'..','salsun')));  % maskLayer
        end
    end

    methods (Test)

        function testPackedSize(testCase)

            rng(3)
            net = salsun2d_create_test_network([32 32]);
            L = salsun2d_hls_layout();

            % Actual values
            w = salsun2d_pack_params(salsun2d_extract_params(net));

            % Evaluation
            testCase.verifySize(w,[L.NParams 1]);
            testCase.verifyClass(w,'single');
            nLearnables = sum(cellfun(@numel,net.Learnables.Value));
            testCase.verifyGreaterThanOrEqual(L.NParams,nLearnables);
        end

        function testPerfectReconstruction(testCase,inputSize)

            rng(4)
            net = salsun2d_create_test_network(inputSize);
            x = rand(inputSize,'single');
            w = salsun2d_pack_params(salsun2d_extract_params(net));

            % Actual values
            yActual = salsun2d_hls(x,w);

            % Evaluation
            testCase.verifyClass(yActual,'single');
            testCase.verifyEqual(yActual,x,'AbsTol',single(1e-4));
        end

        function testMatchesReferenceWithMask(testCase,inputSize)

            rng(5)
            nCoefs = 2;
            coefMask = reshape([ones(nCoefs,1); zeros(16-nCoefs,1)],2,[]).';
            net = salsun2d_create_test_network(inputSize,coefMask(:));
            x = rand(inputSize,'single');
            params = salsun2d_extract_params(net);
            w = salsun2d_pack_params(params);

            % Expected values
            yDouble = salsun2d_infer(double(x),salsun2d_cast_params(params,'double'));
            yRef = salsun2d_infer(x,params);

            % Actual values
            yActual = salsun2d_hls(x,w);

            % Evaluation
            errRef = max(abs(double(yRef(:)) - yDouble(:)));
            errActual = max(abs(double(yActual(:)) - yDouble(:)));
            testCase.verifyLessThanOrEqual(errActual,2*errRef + 1e-6, ...
                sprintf('Error from double: HLS %g, reference %g',errActual,errRef));
            testCase.verifyEqual(yActual,yRef,'AbsTol',single(1e-2), ...
                'A large difference means an implementation error, not rounding.');
        end

    end

end
