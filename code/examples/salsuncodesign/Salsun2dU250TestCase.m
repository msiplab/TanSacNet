classdef Salsun2dU250TestCase < matlab.unittest.TestCase
    %SALSUN2DU250TESTCASE Test case for salsun2d_u250 (Alveo U250 kernel)
    %
    % Runs salsun2d_kernel (design salsun2d_hls_opt) and compares the
    % result with the reference salsun2d_infer. As for the MATLAB versions,
    % the kernel is judged by its error from the double-precision
    % reference. The frames are 300 x 300 on the card and 32 x 32 in
    % emulation, where the kernel runs on one CPU thread.
    %
    % Each target is tested only when its xclbin and the MEX gateway are
    % built. One MATLAB session can use only one XRT target (see
    % salsun2d_u250): the first available one in the order hw, hw_emu,
    % sw_emu is tested and the others are skipped.
    %
    % Requirements: MATLAB R2026b, Deep Learning Toolbox, XRT
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
        target = {'hw','hw_emu','sw_emu'};
    end

    properties (Constant)
        Design = 'salsun2d_hls_opt';
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

        function testMatchesReferenceWithMask(testCase,target)

            if strcmp(target,'hw')
                frameSize = [300 300];
            else
                frameSize = [32 32];
            end
            here = fileparts(mfilename('fullpath'));
            xclbinName = sprintf('%s_%dx%d.xclbin',testCase.Design,frameSize);
            testCase.assumeTrue(isfile(fullfile(here,'build',target,xclbinName)) && ...
                exist('salsun2d_u250_mex','file') == 3, ...
                sprintf('xclbin for %s or MEX is not built.',target));
            sessionTarget = getenv('SALSUN2D_U250_TARGET');
            testCase.assumeTrue(isempty(sessionTarget) || strcmp(sessionTarget,target), ...
                sprintf('This session uses target %s. Restart MATLAB to test %s.', ...
                sessionTarget,target));

            rng(6)
            nCoefs = 2;
            coefMask = reshape([ones(nCoefs,1); zeros(16-nCoefs,1)],2,[]).';
            net = salsun2d_create_test_network(frameSize,coefMask(:));
            params = salsun2d_extract_params(net);
            w = salsun2d_pack_params(params);
            nFrames = 2;
            x = rand([frameSize nFrames],'single');

            % Expected values
            paramsDouble = salsun2d_cast_params(params,'double');
            yRef = zeros(size(x),'single');
            yDouble = zeros(size(x));
            for iFrame = 1:nFrames
                yRef(:,:,iFrame) = salsun2d_infer(x(:,:,iFrame),params);
                yDouble(:,:,iFrame) = salsun2d_infer(double(x(:,:,iFrame)),paramsDouble);
            end

            % Actual values
            yActual = salsun2d_u250(x,w,target,Design=testCase.Design);

            % Evaluation
            testCase.verifySize(yActual,size(x));
            errRef = max(abs(double(yRef(:)) - yDouble(:)));
            errActual = max(abs(double(yActual(:)) - yDouble(:)));
            testCase.log(1,sprintf('Error from double: U250 (%s) %g, reference %g', ...
                target,errActual,errRef));
            testCase.verifyLessThanOrEqual(errActual,2*errRef + 1e-6, ...
                sprintf('Error from double: U250 %g, reference %g',errActual,errRef));
            testCase.verifyEqual(yActual,yRef,'AbsTol',single(1e-2), ...
                'A large difference means an implementation error, not rounding.');
        end

    end

end
