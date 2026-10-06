classdef Salsun2dInferTestCase < matlab.unittest.TestCase
    %SALSUN2DINFERTESTCASE Test case for salsun2d_infer
    %
    % Compares the reference implementation salsun2d_infer with predict of
    % the dlnetwork it was extracted from. The learnable parameters are
    % perturbed randomly, because the initial angle estimators output zero
    % angles (Wo = 0), which would leave the rotations untested.
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

    methods (Static, Access=private)
        function net = createNetwork(inputSize,coefMask)
            % Same settings as main_salsun2d.m, on the CPU
            import tansacnet.salsun.*
            lg = fcn_createsalsunlgraph2d([], ...
                'InputSize',inputSize, ...
                'Stride',[4 4], ...
                'OverlappingFactor',[3 3], ...
                'NumberOfVanishingMoments',true, ...
                'NumberOfNeighborBlocks',[3 3], ...
                'NumberOfResidualBlocks',3, ...
                'Width',2, ...
                'Mode','Whole', ...
                'ThetaMode','Reuse', ...
                'Device','cpu');
            if ~isempty(coefMask)
                lg = lg.disconnectLayers('Lv1_AcOut','Lv1_AcIn');
                lg = lg.addLayers(maskLayer('Name','Lv1_AcMask', ...
                    'Mask',coefMask(2:end),'NumberOfChannels',numel(coefMask)-1));
                lg = lg.connectLayers('Lv1_AcOut','Lv1_AcMask');
                lg = lg.connectLayers('Lv1_AcMask','Lv1_AcIn');
            end
            net = dlnetwork(lg);
            % Perturb all learnables so that the angles are not zero
            net = dlupdate(@(w) w + 0.1*randn(size(w),'like',w),net);
        end
    end

    methods (Test)

        function testPerfectReconstruction(testCase,inputSize)

            rng(1)
            net = Salsun2dInferTestCase.createNetwork(inputSize,[]);
            x = rand(inputSize,'single');

            % Expected values
            yExpctd = extractdata(net.predict(dlarray(x,'SSCB')));

            % Actual values
            params = salsun2d_extract_params(net);
            yActual = salsun2d_infer(x,params);

            % Evaluation
            testCase.verifySize(yActual,size(x));
            testCase.verifyEqual(yActual,x,'AbsTol',single(1e-4), ...
                'The unmasked network must reconstruct its input.');
            testCase.verifyEqual(yActual,yExpctd,'AbsTol',single(1e-4));
        end

        function testMatchesDlnetworkWithMask(testCase,inputSize)

            rng(2)
            nCoefs = 2;
            nChsTotal = 16;
            coefMask = reshape([ones(nCoefs,1); zeros(nChsTotal-nCoefs,1)],2,[]).';
            coefMask = coefMask(:);
            net = Salsun2dInferTestCase.createNetwork(inputSize,coefMask);
            x = rand(inputSize,'single');

            % Expected values
            % With the mask, rounding errors in the estimated angles no
            % longer cancel between analysis and synthesis, and they grow
            % stage by stage. Both single-precision implementations are
            % therefore compared with the reference in double precision.
            yNet = extractdata(net.predict(dlarray(x,'SSCB')));
            params = salsun2d_extract_params(net);
            yDouble = salsun2d_infer(double(x), ...
                salsun2d_cast_params(params,'double'));

            % Actual values
            [yActual,coefs] = salsun2d_infer(x,params);

            % Evaluation
            testCase.verifyEqual(params.Mask,single(coefMask));
            testCase.verifyEqual(squeeze(any(coefs ~= 0,[2 3])),coefMask ~= 0);
            errNet = max(abs(double(yNet(:)) - yDouble(:)));
            errActual = max(abs(double(yActual(:)) - yDouble(:)));
            testCase.verifyLessThanOrEqual(errActual,2*errNet + 1e-6, ...
                sprintf('Error from double: reference %g, dlnetwork %g',errActual,errNet));
            testCase.verifyEqual(yActual,yNet,'AbsTol',single(1e-2), ...
                'A large difference means an implementation error, not rounding.');
        end

    end

end
