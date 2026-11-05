classdef lsunAtomExtension1dLayerTestCase < matlab.unittest.TestCase
    %LSUNATOMEXTENSION1DLAYERTESTCASE
    %
    %   コンポーネント別に入力(nComponents=1のみサポート):
    %      nChsTotal x 1 x nBlks x nSamples
    %
    %   コンポーネント別に出力(nComponents=1のみサポート):
    %      nChsTotal x 1 x nBlks x nSamples
    %
    % Requirements: MATLAB R2022b
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

    properties (TestParameter)
        stride = { 2, 4, 8 };
        datatype = { 'single', 'double' };
        nblks = struct('small', 4,'medium', 8, 'large', 16);
        dir = { 'Right', 'Left' };
        target = { 'Sum', 'Difference' }
    end

    methods (TestClassTeardown)
        function finalCheck(~)
            import tansacnet.lsun.*
            layer = lsunAtomExtension1dLayer(...
                'Stride',2,...
                'Direction','Right',...
                'TargetChannels','Difference');
            fprintf("\n --- Check layer for 1-D sequences ---\n");
            checkLayer(layer,[2 1 8],...
                'ObservationDimension',4,...
                'CheckCodegenCompatibility',true)
        end
    end

    methods (Test)

        function testConstructor(testCase, stride, target)

            % Parameters
            nChsTotal = stride;

            % Expected values
            expctdName = 'Qn';
            expctdDirection = 'Right';
            expctdTargetChannels = target;
            expctdDescription = "Right shift the " ...
                + lower(target) ...
                + "-channel Coefs. " ...
                + "(pt,pb) = (" ...
                + ceil(nChsTotal/2) + "," + floor(nChsTotal/2) + ")";

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunAtomExtension1dLayer(...
                'Stride',stride,...
                'Name',expctdName,...
                'Direction',expctdDirection,...
                'TargetChannels',expctdTargetChannels);

            % Actual values
            actualName = layer.Name;
            actualDirection = layer.Direction;
            actualTargetChannels = layer.TargetChannels;
            actualDescription = layer.Description;

            % Evaluation
            testCase.verifyEqual(actualName,expctdName);
            testCase.verifyEqual(actualDirection,expctdDirection);
            testCase.verifyEqual(actualTargetChannels,expctdTargetChannels);
            testCase.verifyEqual(actualDescription,expctdDescription);
        end

        function testPredictShiftDifferenceCoefs(testCase, ...
                stride, nblks, dir, datatype)

            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));

            % Parameters
            nSamples = 8;
            nChsTotal = stride;
            target_ = 'Difference';
            % nChsTotal x 1 x nBlks x nSamples
            X = randn(nChsTotal,1,nblks,nSamples,datatype);

            % Expected values
            if strcmp(dir,'Right')
                shift = [ 0 0  1 0 ];
            elseif strcmp(dir,'Left')
                shift = [ 0 0 -1 0 ];
            else
                shift = [ 0 0 0 0 ];
            end
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            % Block butterfly
            Yt = X(1:pt,:,:,:);
            Yb = X(pt+1:pt+pb,:,:,:);
            Y =  [ Yt+Yb ; Yt-Yb ]/sqrt(2);
            % Block circular shift
            Y(pt+1:pt+pb,:,:,:) = circshift(Y(pt+1:pt+pb,:,:,:),shift);
            % Block butterfly
            Yt = Y(1:pt,:,:,:);
            Yb = Y(pt+1:pt+pb,:,:,:);
            Y =  [ Yt+Yb ; Yt-Yb ]/sqrt(2);
            % Output
            expctdZ = Y;

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunAtomExtension1dLayer(...
                'Stride',stride,...
                'Name','Qn~',...
                'Direction',dir,...
                'TargetChannels',target_);

            % Actual values
            actualZ = layer.predict(X);

            % Evaluation
            testCase.verifyInstanceOf(actualZ,datatype);
            testCase.verifyThat(actualZ,...
                IsEqualTo(expctdZ,'Within',tolObj));

        end

        function testPredictShiftSumCoefs(testCase, ...
                stride, nblks, dir, datatype)

            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));

            % Parameters
            nSamples = 8;
            nChsTotal = stride;
            target_ = 'Sum';
            % nChsTotal x 1 x nBlks x nSamples
            X = randn(nChsTotal,1,nblks,nSamples,datatype);

            % Expected values
            if strcmp(dir,'Right')
                shift = [ 0 0  1 0 ];
            elseif strcmp(dir,'Left')
                shift = [ 0 0 -1 0 ];
            else
                shift = [ 0 0 0 0 ];
            end
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            % Block butterfly
            Yt = X(1:pt,:,:,:);
            Yb = X(pt+1:pt+pb,:,:,:);
            Y =  [ Yt+Yb ; Yt-Yb ]/sqrt(2);
            % Block circular shift
            Y(1:pt,:,:,:) = circshift(Y(1:pt,:,:,:),shift);
            % Block butterfly
            Yt = Y(1:pt,:,:,:);
            Yb = Y(pt+1:pt+pb,:,:,:);
            Y =  [ Yt+Yb ; Yt-Yb ]/sqrt(2);
            % Output
            expctdZ = Y;

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunAtomExtension1dLayer(...
                'Stride',stride,...
                'Name','Qn~',...
                'Direction',dir,...
                'TargetChannels',target_);

            % Actual values
            actualZ = layer.predict(X);

            % Evaluation
            testCase.verifyInstanceOf(actualZ,datatype);
            testCase.verifyThat(actualZ,...
                IsEqualTo(expctdZ,'Within',tolObj));

        end

        function testBackwardShiftDifferenceCoefs(testCase, ...
                stride, nblks, dir, datatype)

            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));

            % Parameters
            nSamples = 8;
            nChsTotal = stride;
            target_ = 'Difference';
            % nChsTotal x 1 x nBlks x nSamples
            dLdZ = randn(nChsTotal,1,nblks,nSamples,datatype);

            % Expected values
            if strcmp(dir,'Right')
                shift = [ 0 0 -1 0 ]; % Reverse
            elseif strcmp(dir,'Left')
                shift = [ 0 0  1 0 ]; % Reverse
            else
                shift = [ 0 0 0 0 ];
            end
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            Y = dLdZ;
            % Block butterfly
            Yt = Y(1:pt,:,:,:);
            Yb = Y(pt+1:pt+pb,:,:,:);
            Y =  [ Yt+Yb ; Yt-Yb ]/sqrt(2);
            % Block circular shift
            Y(pt+1:pt+pb,:,:,:) = circshift(Y(pt+1:pt+pb,:,:,:),shift);
            % Block butterfly
            Yt = Y(1:pt,:,:,:);
            Yb = Y(pt+1:pt+pb,:,:,:);
            Y =  [ Yt+Yb ; Yt-Yb ]/sqrt(2);
            % Output
            expctddLdX = Y;

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunAtomExtension1dLayer(...
                'Stride',stride,...
                'Name','Qn',...
                'Direction',dir,...
                'TargetChannels',target_);

            % Actual values
            actualdLdX = layer.backward([],[],dLdZ,[]);

            % Evaluation
            testCase.verifyInstanceOf(actualdLdX,datatype);
            testCase.verifyThat(actualdLdX,...
                IsEqualTo(expctddLdX,'Within',tolObj));

        end

        function testBackwardShiftSumCoefs(testCase, ...
                stride, nblks, dir, datatype)

            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));

            % Parameters
            nSamples = 8;
            nChsTotal = stride;
            target_ = 'Sum';
            % nChsTotal x 1 x nBlks x nSamples
            dLdZ = randn(nChsTotal,1,nblks,nSamples,datatype);

            % Expected values
            if strcmp(dir,'Right')
                shift = [ 0 0 -1 0 ]; % Reverse
            elseif strcmp(dir,'Left')
                shift = [ 0 0  1 0 ]; % Reverse
            else
                shift = [ 0 0 0 0 ];
            end
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            Y = dLdZ;
            % Block butterfly
            Yt = Y(1:pt,:,:,:);
            Yb = Y(pt+1:pt+pb,:,:,:);
            Y =  [ Yt+Yb ; Yt-Yb ]/sqrt(2);
            % Block circular shift
            Y(1:pt,:,:,:) = circshift(Y(1:pt,:,:,:),shift);
            % Block butterfly
            Yt = Y(1:pt,:,:,:);
            Yb = Y(pt+1:pt+pb,:,:,:);
            Y =  [ Yt+Yb ; Yt-Yb ]/sqrt(2);
            % Output
            expctddLdX = Y;

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunAtomExtension1dLayer(...
                'Stride',stride,...
                'Name','Qn',...
                'Direction',dir,...
                'TargetChannels',target_);

            % Actual values
            actualdLdX = layer.backward([],[],dLdZ,[]);

            % Evaluation
            testCase.verifyInstanceOf(actualdLdX,datatype);
            testCase.verifyThat(actualdLdX,...
                IsEqualTo(expctddLdX,'Within',tolObj));

        end
    end

end
