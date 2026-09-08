classdef salsunStateStandardization2dLayerTestCase < matlab.unittest.TestCase
    %SALSUNSTATESTANDARDIZATION2DLAYERTESTCASE
    %
    %   Input  'in'  : nFeat x nRows x nCols x nSamples
    %   Output 'out' : nFeat x nRows x nCols x nSamples
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
            nFeat = 5;
            nRows = 4;
            nCols = 4;
            layer = salsunStateStandardization2dLayer('Name','Std');
            checkLayer(layer,[nFeat nRows nCols],...
                'ObservationDimension',4)
        end

    end

    methods (Test)

        function testPredictZeroMeanUnitVariance(testCase)
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6);
            import tansacnet.salsun.*

            nFeat = 5;
            nRows = 6;
            nCols = 7;
            nSamples = 3;
            X = 10*randn(nFeat,nRows,nCols,nSamples) + 100;

            layer = salsunStateStandardization2dLayer('Name','Std');
            Z = layer.predict(X);

            testCase.verifySize(Z,size(X));
            nBlks = nRows*nCols;
            actualMean = mean(Z,[2 3]);
            actualVar = sum((Z-actualMean).^2,[2 3])/(nBlks-1);
            testCase.verifyThat(actualMean,IsEqualTo(zeros(nFeat,1,1,nSamples),'Within',tolObj));
            testCase.verifyThat(actualVar,IsEqualTo(ones(nFeat,1,1,nSamples),'Within',AbsoluteTolerance(1e-4)));
        end

        function testPredictMatchesReference(testCase)
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-10);
            import tansacnet.salsun.*

            nFeat = 4;
            nRows = 3;
            nCols = 3;
            nSamples = 2;
            X = randn(nFeat,nRows,nCols,nSamples);

            layer = salsunStateStandardization2dLayer('Name','Std');
            actualZ = layer.predict(X);

            % Reference computed by reshaping to (nFeat, nBlks, nSamples)
            % as in the original SA_LSUN.mlx script.
            nBlks = nRows*nCols;
            Xr = reshape(permute(X,[1 2 3 4]),nFeat,nBlks,nSamples); %#ok<NASGU>
            X_ = reshape(X,nFeat,nBlks,nSamples);
            mu = mean(X_,2);
            sigma = std(X_,0,2) + 1e-8;
            expctdZ = reshape((X_-mu)./sigma,nFeat,nRows,nCols,nSamples);

            testCase.verifyThat(actualZ,IsEqualTo(expctdZ,'Within',tolObj));
        end

        function testGradientCorrectnessByFiniteDifference(testCase)
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            import tansacnet.salsun.*

            nFeat = 3;
            nRows = 3;
            nCols = 3;
            nSamples = 2;
            X0 = 5*randn(nFeat,nRows,nCols,nSamples) + 10;

            layer = salsunStateStandardization2dLayer('Name','Std');

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
            testCase.verifyThat(numGrad,IsEqualTo(dLdX,'Within',AbsoluteTolerance(tol)));
        end

        function testZeroVarianceNoNanOrInf(testCase)
            import tansacnet.salsun.*

            nFeat = 3;
            nRows = 2;
            nCols = 2;
            nSamples = 2;
            baseVals = randn(nFeat,1,1,nSamples);
            X = repmat(baseVals,[1 nRows nCols 1]);

            layer = salsunStateStandardization2dLayer('Name','Std');
            Z = layer.predict(X);

            testCase.verifyTrue(all(isfinite(Z),'all'));
            
            % Due to zero variance, the output should be all zeros (after standardization).
            testCase.verifyEqual(Z,zeros(size(X)));
        end

    end

end

function dLdX = localModelGradient(layer,X)
Z = layer.predict(X);
loss = sum(Z.^2,'all');
dLdX = dlgradient(loss,X);
end
