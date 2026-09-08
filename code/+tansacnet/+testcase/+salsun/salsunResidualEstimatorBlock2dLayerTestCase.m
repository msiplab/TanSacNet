classdef salsunResidualEstimatorBlock2dLayerTestCase < matlab.unittest.TestCase
    %SALSUNRESIDUALESTIMATORBLOCK2DLAYERTESTCASE
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
            nFeat = 6;
            nRows = 4;
            nCols = 4;
            layer = salsunResidualEstimatorBlock2dLayer(...
                'Name','Res1','InputSize',nFeat,'Width',2);
            checkLayer(layer,[nFeat nRows nCols],...
                'ObservationDimension',4)
        end

    end

    methods (Test)

        function testPredictIsIdentityAtInitialization(testCase)
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-10);
            import tansacnet.salsun.*

            nFeat = 6;
            nRows = 2;
            nCols = 3;
            nSamples = 2;
            X = randn(nFeat,nRows,nCols,nSamples);

            layer = salsunResidualEstimatorBlock2dLayer(...
                'Name','Res1','InputSize',nFeat,'Width',2);
            Z = layer.predict(X);

            testCase.verifySize(Z,size(X));
            testCase.verifyThat(Z,IsEqualTo(X,'Within',tolObj));
        end

        function testPredictShapePreserved(testCase)
            import tansacnet.salsun.*
            nFeat = 5;
            nRows = 3;
            nCols = 4;
            nSamples = 2;
            X = randn(nFeat,nRows,nCols,nSamples);

            layer = salsunResidualEstimatorBlock2dLayer(...
                'Name','Res1','InputSize',nFeat,'Width',3);
            % Perturb W2 so the residual branch is nontrivial.
            layer.W2 = 0.01*randn(size(layer.W2));
            Z = layer.predict(X);

            testCase.verifySize(Z,size(X));
            testCase.verifyNotEqual(Z,X);
        end

        function testGradientsFlowToAllLearnables(testCase)
            import tansacnet.salsun.*
            nFeat = 4;
            nRows = 2;
            nCols = 2;
            nSamples = 2;
            X = dlarray(randn(nFeat,nRows,nCols,nSamples));

            layer = salsunResidualEstimatorBlock2dLayer(...
                'Name','Res1','InputSize',nFeat,'Width',2);
            layer.W2 = 0.1*randn(size(layer.W2)); % nonzero so gradients are nontrivial

            [dLdX,dLdGamma,dLdBeta,dLdW1,dLdB1,dLdW2,dLdB2] = dlfeval(...
                @localModelGradients,layer,X,...
                dlarray(layer.Gamma),dlarray(layer.Beta),...
                dlarray(layer.W1),dlarray(layer.B1),...
                dlarray(layer.W2),dlarray(layer.B2));

            testCase.verifyEqual(size(dLdX),size(X));
            testCase.verifyTrue(all(isfinite(extractdata(dLdX)),'all'));
            testCase.verifyTrue(all(isfinite(extractdata(dLdGamma)),'all'));
            testCase.verifyTrue(all(isfinite(extractdata(dLdBeta)),'all'));
            testCase.verifyTrue(all(isfinite(extractdata(dLdW1)),'all'));
            testCase.verifyTrue(all(isfinite(extractdata(dLdB1)),'all'));
            testCase.verifyTrue(all(isfinite(extractdata(dLdW2)),'all'));
            testCase.verifyTrue(all(isfinite(extractdata(dLdB2)),'all'));
        end

        function testPredictMatchesReferenceFormula(testCase)
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-10);
            import tansacnet.salsun.*

            nFeat = 4;
            nRows = 2;
            nCols = 3;
            nSamples = 2;
            width = 2;
            X = randn(nFeat,nRows,nCols,nSamples);

            layer = salsunResidualEstimatorBlock2dLayer(...
                'Name','Res1','InputSize',nFeat,'Width',width);
            layer.Gamma = 0.5+randn(size(layer.Gamma));
            layer.Beta  = 0.2*randn(size(layer.Beta));
            layer.W1    = 0.3*randn(size(layer.W1));
            layer.B1    = 0.1*randn(size(layer.B1));
            layer.W2    = 0.3*randn(size(layer.W2));
            layer.B2    = 0.1*randn(size(layer.B2));

            % Reference computation, written independently of predict().
            mu = mean(X,1);
            v = mean((X-mu).^2,1);
            xhat = (X-mu)./sqrt(v+1e-5);
            ln = layer.Gamma.*xhat + layer.Beta;
            lnFlat = reshape(ln,nFeat,nRows*nCols*nSamples);
            z1 = layer.W1*lnFlat + layer.B1;
            gelu = 0.5*z1.*(1+tanh(sqrt(2/pi)*(z1+0.044715*z1.^3)));
            z2 = layer.W2*gelu + layer.B2;
            residual = reshape(z2,nFeat,nRows,nCols,nSamples);
            Zexpected = X + residual;

            Z = layer.predict(X);
            testCase.verifySize(Z,size(X));
            testCase.verifyThat(Z,IsEqualTo(Zexpected,'Within',tolObj));
        end

        function testGradientCorrectnessByFiniteDifference(testCase)
            import tansacnet.salsun.*
            nFeat = 3;
            nRows = 2;
            nCols = 2;
            nSamples = 1;
            width = 2;

            X0 = randn(nFeat,nRows,nCols,nSamples);
            layer = salsunResidualEstimatorBlock2dLayer(...
                'Name','Res1','InputSize',nFeat,'Width',width);
            layer.Gamma = 0.5+randn(size(layer.Gamma));
            layer.Beta  = 0.2*randn(size(layer.Beta));
            layer.W1    = 0.3*randn(size(layer.W1));
            layer.B1    = 0.1*randn(size(layer.B1));
            layer.W2    = 0.3*randn(size(layer.W2));
            layer.B2    = 0.1*randn(size(layer.B2));

            [dLdX,dLdGamma,dLdBeta,dLdW1,dLdB1,dLdW2,dLdB2] = dlfeval(...
                @localModelGradients,layer,dlarray(X0),...
                dlarray(layer.Gamma),dlarray(layer.Beta),...
                dlarray(layer.W1),dlarray(layer.B1),...
                dlarray(layer.W2),dlarray(layer.B2));

            h = 1e-5;
            tol = 1e-4;
            localVerifyGradient(testCase,layer,X0,'',extractdata(dLdX),h,tol);
            localVerifyGradient(testCase,layer,X0,'Gamma',extractdata(dLdGamma),h,tol);
            localVerifyGradient(testCase,layer,X0,'Beta',extractdata(dLdBeta),h,tol);
            localVerifyGradient(testCase,layer,X0,'W1',extractdata(dLdW1),h,tol);
            localVerifyGradient(testCase,layer,X0,'B1',extractdata(dLdB1),h,tol);
            localVerifyGradient(testCase,layer,X0,'W2',extractdata(dLdW2),h,tol);
            localVerifyGradient(testCase,layer,X0,'B2',extractdata(dLdB2),h,tol);
        end

    end

end

function L = localLoss(layer,X)
Z = layer.predict(X);
L = sum(Z.^2,'all');
end

function localVerifyGradient(testCase,layer,X0,fieldName,analyticGrad,h,tol)
import matlab.unittest.constraints.IsEqualTo
import matlab.unittest.constraints.AbsoluteTolerance
tolObj = AbsoluteTolerance(tol);

if isempty(fieldName)
    arr0 = X0;
else
    arr0 = layer.(fieldName);
end
numGrad = zeros(size(arr0));
for idx = 1:numel(arr0)
    arrP = arr0; arrP(idx) = arrP(idx) + h;
    arrM = arr0; arrM(idx) = arrM(idx) - h;
    if isempty(fieldName)
        Lp = localLoss(layer,arrP);
        Lm = localLoss(layer,arrM);
    else
        layerP = layer; layerP.(fieldName) = arrP;
        layerM = layer; layerM.(fieldName) = arrM;
        Lp = localLoss(layerP,X0);
        Lm = localLoss(layerM,X0);
    end
    numGrad(idx) = (Lp-Lm)/(2*h);
end
testCase.verifyThat(numGrad,IsEqualTo(analyticGrad,'Within',tolObj));
end

function [dLdX,dLdGamma,dLdBeta,dLdW1,dLdB1,dLdW2,dLdB2] = localModelGradients(...
    layer,X,Gamma,Beta,W1,B1,W2,B2)
layer.Gamma = Gamma;
layer.Beta = Beta;
layer.W1 = W1;
layer.B1 = B1;
layer.W2 = W2;
layer.B2 = B2;
Z = layer.predict(X);
loss = sum(Z.^2,'all');
[dLdX,dLdGamma,dLdBeta,dLdW1,dLdB1,dLdW2,dLdB2] = dlgradient(loss, ...
    X,Gamma,Beta,W1,B1,W2,B2);
end
