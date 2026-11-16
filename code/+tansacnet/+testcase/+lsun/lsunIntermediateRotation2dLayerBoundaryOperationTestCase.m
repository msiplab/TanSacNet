classdef lsunIntermediateRotation2dLayerBoundaryOperationTestCase < matlab.unittest.TestCase
    %LSUNINTERMEDIATEROTATION2DLAYERBOUNDARYOPERATIONTESTCASE
    %
    %   Test cases for the BoundaryOperation option ('Circular' | 'Termination')
    %   of lsunIntermediateRotation2dLayer.
    %
    %   Termination fixes the rotation matrices of the boundary blocks
    %   (first column for horizontal, first row for vertical) to -I and
    %   excludes them from the angle gradients.
    %
    %   コンポーネント別に入力(nComponents):
    %      nChs x nRows x nCols x nSamples
    %
    %   コンポーネント別に出力(nComponents):
    %      nChs x nRows x nCols x nSamples
    %
    % References:
    %   S. Muramatsu, T. Kobayashi, M. Hiki and H. Kikuchi, "Boundary
    %   Operation of 2-D Nonseparable Linear-Phase Paraunitary Filter Banks,"
    %   IEEE Trans. Image Process., vol. 21, no. 4, pp. 2314-2318, 2012.
    %
    %   K. Furuya, S. Hara, K. Seino and S. Muramatsu, "Boundary operation of
    %   2D non-separable oversampled lapped transforms," APSIPA Trans. Signal
    %   Inf. Process., vol. 5, no. 1, pp. 1-9, 2016.
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
        stride = { [2 2], [4 4] };
        datatype = { 'single', 'double' };
        mus = { -1, 1 };
        nrows = struct('small', 2,'medium', 4, 'large', 8);
        ncols = struct('small', 2,'medium', 4, 'large', 8);
        usegpu = struct( 'true', true, 'false', false);
        mode = { 'Analysis', 'Synthesis' };
        bdir = { 'Horizontal', 'Vertical' }; % TODO: property name to be decided
    end

    methods (TestClassTeardown)
        function finalCheck(~)
            % TODO: checkLayer for a layer with 'BoundaryOperation','Termination'
        end
    end

    methods (Test)

        function testConstructorDefaultBoundaryOperation(testCase, stride)
            % Default BoundaryOperation is 'Circular' (backward compatibility)
            testCase.verifyFail('Not implemented yet.');
        end

        function testConstructorWithBoundaryOperation(testCase, stride, bdir)
            % BoundaryOperation and boundary direction are set by the constructor
            testCase.verifyFail('Not implemented yet.');
        end

        function testInvalidBoundaryOperation(testCase, stride)
            % An invalid BoundaryOperation value throws an exception
            testCase.verifyFail('Not implemented yet.');
        end

        function testPredictCircular(testCase, ...
                usegpu, stride, nrows, ncols, mus, mode, datatype)
            % 'Circular' gives the same output as the current layer
            testCase.verifyFail('Not implemented yet.');
        end

        function testPredictTermination(testCase, ...
                usegpu, stride, nrows, ncols, mus, mode, bdir, datatype)
            % Boundary blocks: Za = -Ya, other blocks: Za = Un(:,:,k)*Ya
            testCase.verifyFail('Not implemented yet.');
        end

        function testPredictTerminationWithRandomAngles(testCase, ...
                usegpu, stride, nrows, ncols, mode, bdir, datatype)
            % Boundary blocks stay -I for any Angles and Mus
            testCase.verifyFail('Not implemented yet.');
        end

        function testBackwardTermination(testCase, ...
                usegpu, stride, nrows, ncols, mode, bdir, datatype)
            % dLdX: boundary blocks give -dLdZ, other blocks give Un(:,:,k).'*dLdZ
            % (Un(:,:,k)*dLdZ in Synthesis mode)
            testCase.verifyFail('Not implemented yet.');
        end

        function testBackwardTerminationAngleGradients(testCase, ...
                usegpu, stride, nrows, ncols, mode, bdir, datatype)
            % dLdW: angle gradients of the boundary blocks are zero and
            % those of the other blocks equal the 'Circular' case
            testCase.verifyFail('Not implemented yet.');
        end

    end

end
