classdef salsunAngleEstimatorOutput2dLayerTestCase < matlab.unittest.TestCase
    %SALSUNANGLEESTIMATOROUTPUT2DLAYERTESTCASE
    %
    %   Input  'in'    : nFeat x nRows x nCols x nSamples
    %   Output 'theta' : (NumberOfAngles+NumberOfZeroPadAngles) x
    %                    (nRows*nCols) x nSamples
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
            nAngles = 4;
            nRows = 4;
            nCols = 4;
            layer = salsunAngleEstimatorOutput2dLayer(...
                'Name','Theta','InputSize',nFeat,'NumberOfAngles',nAngles);
            checkLayer(layer,[nFeat nRows nCols],...
                'ObservationDimension',4)
        end

    end

    methods (Test)

        function testPredictZeroAtInitialization(testCase)
            import tansacnet.salsun.*
            nFeat = 6;
            nAngles = 4;
            nRows = 2;
            nCols = 3;
            nSamples = 2;
            X = randn(nFeat,nRows,nCols,nSamples);

            layer = salsunAngleEstimatorOutput2dLayer(...
                'Name','Theta','InputSize',nFeat,'NumberOfAngles',nAngles);
            Theta = layer.predict(X);

            testCase.verifySize(Theta,[nAngles nRows*nCols nSamples]);
            testCase.verifyEqual(Theta,zeros(nAngles,nRows*nCols,nSamples));
        end

        function testPredictWithZeroPad(testCase)
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-10);
            import tansacnet.salsun.*

            nFeat = 5;
            nAnglesPredicted = 3;
            nZeroPad = 2;
            nRows = 2;
            nCols = 2;
            nSamples = 2;
            X = randn(nFeat,nRows,nCols,nSamples);

            layer = salsunAngleEstimatorOutput2dLayer(...
                'Name','Theta','InputSize',nFeat,...
                'NumberOfAngles',nAnglesPredicted,...
                'NumberOfZeroPadAngles',nZeroPad);
            layer.Wo = randn(size(layer.Wo));
            layer.Bo = randn(size(layer.Bo));
            Theta = layer.predict(X);

            testCase.verifySize(Theta,[nAnglesPredicted+nZeroPad nRows*nCols nSamples]);
            testCase.verifyEqual(Theta(1:nZeroPad,:,:),...
                zeros(nZeroPad,nRows*nCols,nSamples));

            Xflat = reshape(X,nFeat,nRows*nCols*nSamples);
            expctdPredicted = reshape(layer.Wo*Xflat+layer.Bo,...
                nAnglesPredicted,nRows*nCols,nSamples);
            testCase.verifyThat(Theta(nZeroPad+1:end,:,:),...
                IsEqualTo(expctdPredicted,'Within',tolObj));
        end

        function testGradientCorrectnessByFiniteDifference(testCase)
            import tansacnet.salsun.*
            nFeat = 4;
            nAngles = 3;
            nRows = 2;
            nCols = 2;
            nSamples = 2;
            X0 = randn(nFeat,nRows,nCols,nSamples);

            layer = salsunAngleEstimatorOutput2dLayer(...
                'Name','Theta','InputSize',nFeat,'NumberOfAngles',nAngles);
            layer.Wo = 0.3*randn(size(layer.Wo));
            layer.Bo = 0.1*randn(size(layer.Bo));

            [dLdX,dLdWo,dLdBo] = dlfeval(@localModelGradients,layer,...
                dlarray(X0),dlarray(layer.Wo),dlarray(layer.Bo));
            dLdX = extractdata(dLdX);
            dLdWo = extractdata(dLdWo);
            dLdBo = extractdata(dLdBo);

            h = 1e-5;
            tol = 1e-4;
            localVerifyGradient(testCase,layer,X0,'',dLdX,h,tol);
            localVerifyGradient(testCase,layer,X0,'Wo',dLdWo,h,tol);
            localVerifyGradient(testCase,layer,X0,'Bo',dLdBo,h,tol);
        end

        function testZeroPadAnglesReceiveNoGradient(testCase)
            import tansacnet.salsun.*
            nFeat = 4;
            nAnglesPredicted = 2;
            nZeroPad = 2;
            nRows = 2;
            nCols = 2;
            nSamples = 2;
            X0 = randn(nFeat,nRows,nCols,nSamples);

            layer = salsunAngleEstimatorOutput2dLayer(...
                'Name','Theta','InputSize',nFeat,...
                'NumberOfAngles',nAnglesPredicted,...
                'NumberOfZeroPadAngles',nZeroPad);
            layer.Wo = 0.3*randn(size(layer.Wo));
            layer.Bo = 0.1*randn(size(layer.Bo));

            [dLdX,dLdWo,dLdBo] = dlfeval(@localPaddedLossGradients,layer,...
                dlarray(X0),dlarray(layer.Wo),dlarray(layer.Bo),nZeroPad);

            testCase.verifyEqual(extractdata(dLdX),zeros(size(X0)));
            testCase.verifyEqual(extractdata(dLdWo),zeros(size(layer.Wo)));
            testCase.verifyEqual(extractdata(dLdBo),zeros(size(layer.Bo)));
        end

    end

end

function [dLdX,dLdWo,dLdBo] = localModelGradients(layer,X,Wo,Bo)
layer.Wo = Wo;
layer.Bo = Bo;
Theta = layer.predict(X);
loss = sum(Theta.^2,'all');
[dLdX,dLdWo,dLdBo] = dlgradient(loss,X,Wo,Bo);
end

function [dLdX,dLdWo,dLdBo] = localPaddedLossGradients(layer,X,Wo,Bo,nZeroPad)
layer.Wo = Wo;
layer.Bo = Bo;
Theta = layer.predict(X);
loss = sum(Theta(1:nZeroPad,:,:).^2,'all');
[dLdX,dLdWo,dLdBo] = dlgradient(loss,X,Wo,Bo);
end

function L = localLoss(layer,X)
Theta = layer.predict(X);
L = sum(Theta.^2,'all');
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
