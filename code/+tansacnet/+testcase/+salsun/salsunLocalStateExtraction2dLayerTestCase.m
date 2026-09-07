classdef salsunLocalStateExtraction2dLayerTestCase < matlab.unittest.TestCase
    %SALSUNLOCALSTATEEXTRACTION2DLAYERTESTCASE
    %
    %   Input  'in'  : nChs x nRows x nCols x nSamples
    %   Output 'out' : (numel(Channels)*nv*nh) x nRows x nCols x nSamples
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

    methods (Test)

        function testInvalidNumberOfNeighborBlocks(testCase)
            import tansacnet.salsun.*
            testCase.verifyError(...
                @() salsunLocalStateExtraction2dLayer(...
                    'NumberOfNeighborBlocks',[2 3],'Channels',1:3), ...
                'SaLsunLayer:InvalidNumberOfNeighborBlocks');
        end

        function testPredictShapeAndValues(testCase)
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-12);
            import tansacnet.salsun.*

            nChs = 4;
            nRows = 4;
            nCols = 4;
            nSamples = 2;
            channels = 2:nChs; % exclude channel 1 (DC)
            nNeighbor = [3 3];

            X = reshape(1:(nChs*nRows*nCols*nSamples),nChs,nRows,nCols,nSamples);
            X = double(X);

            layer = salsunLocalStateExtraction2dLayer(...
                'Name','Extract',...
                'NumberOfNeighborBlocks',nNeighbor,...
                'Channels',channels);
            Z = layer.predict(X);

            % Expected shape
            testCase.verifySize(Z,[numel(channels)*prod(nNeighbor) nRows nCols nSamples]);

            % Reference: reproduce the reference gather explicitly
            Xc = X(channels,:,:,:);
            expctdZ = [];
            for vshift = 1:-1:-1
                for hshift = 1:-1:-1
                    expctdZ = cat(1,expctdZ,circshift(Xc,[0,vshift,hshift,0]));
                end
            end
            testCase.verifyThat(Z,IsEqualTo(expctdZ,'Within',tolObj));

            nSel = numel(channels);
            centerBlock = Z(4*nSel+1:5*nSel,:,:,:);
            testCase.verifyThat(centerBlock,IsEqualTo(Xc,'Within',tolObj));
        end

    end

end
