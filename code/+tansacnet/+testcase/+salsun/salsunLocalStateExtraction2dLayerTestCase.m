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

    methods (TestClassTeardown)

        function finalCheck(~)
            import tansacnet.salsun.*
            fprintf("\n --- Check layer for 2-D images (SA-LSUN) ---\n");
            nChs = 4;
            nRows = 4;
            nCols = 4;
            layer = salsunLocalStateExtraction2dLayer(...
                'Name','Extract','NumberOfNeighborBlocks',[3 3],...
                'Channels',2:nChs);
            checkLayer(layer,[nChs nRows nCols],...
                'ObservationDimension',4)
        end

    end

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

        function testPredictAndGradientWithDlarray(testCase)
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            import tansacnet.salsun.*

            nChs = 4;
            nRows = 3;
            nCols = 3;
            nSamples = 2;
            channels = 2:nChs;
            nNeighbor = [3 3];
            X0 = randn(nChs,nRows,nCols,nSamples);

            layer = salsunLocalStateExtraction2dLayer(...
                'Name','Extract',...
                'NumberOfNeighborBlocks',nNeighbor,...
                'Channels',channels);

            Zplain = layer.predict(X0);
            Zdl = layer.predict(dlarray(X0));
            testCase.verifyTrue(isdlarray(Zdl));
            testCase.verifyThat(extractdata(Zdl),...
                IsEqualTo(Zplain,'Within',AbsoluteTolerance(1e-12)));

            dLdX = dlfeval(@localModelGradient,layer,dlarray(X0));
            dLdX = extractdata(dLdX);

            h = 1e-5;
            tol = 1e-4;
            numGrad = zeros(size(X0));
            for idx = 1:numel(X0)
                Xp = X0; Xp(idx) = Xp(idx) + h;
                Xm = X0; Xm(idx) = Xm(idx) - h;
                Lp = sum(layer.predict(Xp).^2,'all');
                Lm = sum(layer.predict(Xm).^2,'all');
                numGrad(idx) = (Lp-Lm)/(2*h);
            end
            testCase.verifyThat(numGrad,...
                IsEqualTo(dLdX,'Within',AbsoluteTolerance(tol)));
        end

        function testBoundaryWraparoundIsToroidal(testCase)
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-12);
            import tansacnet.salsun.*

            nChs = 1;
            nRows = 3;
            nCols = 3;
            nSamples = 1;
            channels = 1;
            nNeighbor = [3 3];
            X = reshape(1:9,nChs,nRows,nCols,nSamples);

            layer = salsunLocalStateExtraction2dLayer(...
                'Name','Extract',...
                'NumberOfNeighborBlocks',nNeighbor,...
                'Channels',channels);
            Z = layer.predict(X);

            cornerNeighbor = Z(1,1,1,1); % block (row=1,col=1)
            testCase.verifyThat(cornerNeighbor,...
                IsEqualTo(X(1,3,3,1),'Within',tolObj)); % wraps to (3,3)

            centerNeighbor = Z(1,2,2,1); % block (row=2,col=2)
            testCase.verifyThat(centerNeighbor,...
                IsEqualTo(X(1,1,1,1),'Within',tolObj)); % ordinary, no wrap
        end

    end

end

function dLdX = localModelGradient(layer,X)
Z = layer.predict(X);
loss = sum(Z.^2,'all');
dLdX = dlgradient(loss,X);
end
