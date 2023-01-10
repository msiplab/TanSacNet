classdef lsunCSAtomExtension1dLayerTestCase < matlab.unittest.TestCase
    %LSUNCSATOMEXTENSION1DLAYERTESTCASE
    %
    %   コンポーネント別に入力(nComponents=1のみサポート):
    %      nChsTotal x nSamples x nBlks
    %
    %   コンポーネント別に出力(nComponents=1のみサポート):
    %      nChsTotal x nSamples x nBlks
    %
    % Requirements: MATLAB R2022b
    %
    % Copyright (c) 2023, Shogo MURAMATSU
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
        %{
        function finalCheck(~)
            import tansacnet.lsun.*
            layer = lsunCSAtomExtension1dLayer(...
                'Stride',2,...
                'Direction','Right',...
                'TargetChannels','Difference');
            fprintf("\n --- Check layer for 1-D sequences ---\n");
            checkLayer(layer,[2 8 8],...
                'ObservationDimension',2,...
                'CheckCodegenCompatibility',true)
        end
        %}
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
                + "(ps,pa) = (" ...
                + ceil(nChsTotal/2) + "," + floor(nChsTotal/2) + ")";
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunCSAtomExtension1dLayer(...
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
            % nChsTotal x nSamples x nBlks
            %X = randn(nrows,ncols,nChsTotal,nSamples,datatype);
            X = randn(nChsTotal,nSamples,nblks,datatype);
            
            % Expected values
            if strcmp(dir,'Right')
                shift = [ 0 0 1 ];
            elseif strcmp(dir,'Left')
                shift = [ 0 0 -1 ];
            else
                shift = [ 0 0 0 ];
            end
            %
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            % Block butterfly
            Ys = X(1:ps,:,:);
            Ya = X(ps+1:ps+pa,:,:);
            Y =  [ Ys+Ya ; Ys-Ya ]/sqrt(2);
            % Block circular shift
            Y(ps+1:ps+pa,:,:) = circshift(Y(ps+1:ps+pa,:,:),shift);
            % Block butterfly
            Ys = Y(1:ps,:,:);
            Ya = Y(ps+1:ps+pa,:,:);
            Y =  [ Ys+Ya ; Ys-Ya ]/sqrt(2);
            % Output
            expctdZ = Y; %ipermute(Y,[3 1 2 4]);
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunCSAtomExtension1dLayer(...
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
            % nChsTotal x nSamples x nBlks
            X = randn(nChsTotal,nSamples,nblks,datatype);
            
            % Expected values
            if strcmp(dir,'Right')
                shift = [ 0 0  1 ];
            elseif strcmp(dir,'Left')
                shift = [ 0 0 -1 ];
            else
                shift = [ 0 0 0 ];
            end
            % nChsTotal x nSamples x nblks
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            % Block butterfly
            Ys = X(1:ps,:,:);
            Ya = X(ps+1:ps+pa,:,:);
            Y =  [ Ys+Ya ; Ys-Ya ]/sqrt(2);
            % Block circular shift
            Y(1:ps,:,:) = circshift(Y(1:ps,:,:),shift);
            % Block butterfly
            Ys = Y(1:ps,:,:);
            Ya = Y(ps+1:ps+pa,:,:);
            Y =  [ Ys+Ya ; Ys-Ya ]/sqrt(2);
            % Output
            expctdZ = Y; %ipermute(Y,[3 1 2 4]);
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunCSAtomExtension1dLayer(...
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
            % nChsTotal x nSamples x nBlks
            dLdZ = randn(nChsTotal,nSamples,nblks,datatype);
            
            % Expected values
            if strcmp(dir,'Right')
                shift = [ 0 0 -1 ]; % Reverse
            elseif strcmp(dir,'Left')
                shift = [ 0 0 1 ]; % Reverse
            else
                shift = [ 0 0 0 ]; % Reverse
            end
            % nChsTotal x nSamples x nBlks
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            Y = dLdZ; %permute(dLdZ,[3 1 2 4]); % [ch ver hor smpl]
            % Block butterfly
            Ys = Y(1:ps,:,:);
            Ya = Y(ps+1:ps+pa,:,:);
            Y =  [ Ys+Ya ; Ys-Ya ]/sqrt(2);
            % Block circular shift
            Y(ps+1:ps+pa,:,:) = circshift(Y(ps+1:ps+pa,:,:),shift);
            % Block butterfly
            Ys = Y(1:ps,:,:);
            Ya = Y(ps+1:ps+pa,:,:);
            Y =  [ Ys+Ya ; Ys-Ya ]/sqrt(2);
            % Output
            expctddLdX = Y; %ipermute(Y,[3 1 2 4]);
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunCSAtomExtension1dLayer(...
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
            % nChsTotal x nSamples x nBlks
            dLdZ = randn(nChsTotal,nSamples,nblks,datatype);
            
            % Expected values
            if strcmp(dir,'Right')
                shift = [ 0 0 -1 ]; % Reverse
            elseif strcmp(dir,'Left')
                shift = [ 0 0  1 ]; % Reverse
            else
                shift = [ 0 0 0 ];
            end
            % nChsTotal x nSamples x nBlks
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            Y = dLdZ; %permute(dLdZ,[3 1 2 4]); % [ch ver hor smpl]
            % Block butterfly
            Ys = Y(1:ps,:,:);
            Ya = Y(ps+1:ps+pa,:,:);
            Y =  [ Ys+Ya ; Ys-Ya ]/sqrt(2);
            % Block circular shift
            Y(1:ps,:,:) = circshift(Y(1:ps,:,:),shift);
            % Block butterfly
            Ys = Y(1:ps,:,:);
            Ya = Y(ps+1:ps+pa,:,:);
            Y =  [ Ys+Ya ; Ys-Ya ]/sqrt(2);
            % Output
            expctddLdX = Y; %ipermute(Y,[3 1 2 4]);
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunCSAtomExtension1dLayer(...
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
        %{
        function testPredictShiftDiffCoefsWithRandomAngles(testCase, ...
                stride, nblks, dir, datatype)
            % TODO: random angles
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            
            % Parameters
            nSamples = 8;
            nChsTotal = stride;
            nAngles = nChsTotal/2
            target_ = 'Difference';
            % nChsTotal x nSamples x nBlks
            X = randn(nChsTotal,nSamples,nblks,datatype);
            
            % Expected values
            if strcmp(dir,'Right')
                shift = [ 0 0 1 ];
            elseif strcmp(dir,'Left')
                shift = [ 0 0 -1 ];
            else
                shift = [ 0 0 0 ];
            end
            %
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            % Block butterfly
            Ys = X(1:ps,:,:);
            Ya = X(ps+1:ps+pa,:,:);
            Y =  [ Ys+Ya ; Ys-Ya ]/sqrt(2);
            % Block circular shift
            Y(ps+1:ps+pa,:,:) = circshift(Y(ps+1:ps+pa,:,:),shift);
            % Block butterfly
            Ys = Y(1:ps,:,:);
            Ya = Y(ps+1:ps+pa,:,:);
            Y =  [ Ys+Ya ; Ys-Ya ]/sqrt(2);
            % Output
            expctdZ = Y; %ipermute(Y,[3 1 2 4]);
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunCSAtomExtension1dLayer(...
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

        function testPredictShiftSumCoefsWithRandomAngles(testCase, ...
                stride, nblks, dir, datatype)
            % TODO: random angles            
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            
            % Parameters
            nSamples = 8;
            nChsTotal = stride;
            target_ = 'Sum';
            % nChsTotal x nSamples x nBlks
            X = randn(nChsTotal,nSamples,nblks,datatype);
            
            % Expected values
            if strcmp(dir,'Right')
                shift = [ 0 0  1 ];
            elseif strcmp(dir,'Left')
                shift = [ 0 0 -1 ];
            else
                shift = [ 0 0 0 ];
            end
            % nChsTotal x nSamples x nblks
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            % Block butterfly
            Ys = X(1:ps,:,:);
            Ya = X(ps+1:ps+pa,:,:);
            Y =  [ Ys+Ya ; Ys-Ya ]/sqrt(2);
            % Block circular shift
            Y(1:ps,:,:) = circshift(Y(1:ps,:,:),shift);
            % Block butterfly
            Ys = Y(1:ps,:,:);
            Ya = Y(ps+1:ps+pa,:,:);
            Y =  [ Ys+Ya ; Ys-Ya ]/sqrt(2);
            % Output
            expctdZ = Y; %ipermute(Y,[3 1 2 4]);
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunCSAtomExtension1dLayer(...
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
        
        function testBackwardShiftDiffCoefsWithRandomAngles(testCase, ...
                stride, nblks, dir, datatype)
            % TODO: random angles
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            
            % Parameters
            nSamples = 8;
            nChsTotal = stride;
            target_ = 'Difference';
            % nChsTotal x nSamples x nBlks
            dLdZ = randn(nChsTotal,nSamples,nblks,datatype);
            
            % Expected values
            if strcmp(dir,'Right')
                shift = [ 0 0 -1 ]; % Reverse
            elseif strcmp(dir,'Left')
                shift = [ 0 0 1 ]; % Reverse
            else
                shift = [ 0 0 0 ]; % Reverse
            end
            % nChsTotal x nSamples x nBlks
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            Y = dLdZ; %permute(dLdZ,[3 1 2 4]); % [ch ver hor smpl]
            % Block butterfly
            Ys = Y(1:ps,:,:);
            Ya = Y(ps+1:ps+pa,:,:);
            Y =  [ Ys+Ya ; Ys-Ya ]/sqrt(2);
            % Block circular shift
            Y(ps+1:ps+pa,:,:) = circshift(Y(ps+1:ps+pa,:,:),shift);
            % Block butterfly
            Ys = Y(1:ps,:,:);
            Ya = Y(ps+1:ps+pa,:,:);
            Y =  [ Ys+Ya ; Ys-Ya ]/sqrt(2);
            % Output
            expctddLdX = Y; %ipermute(Y,[3 1 2 4]);
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunCSAtomExtension1dLayer(...
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

        function testBackwardShiftSumCoefsWithRandomAngles(testCase, ...
                stride, nblks, dir, datatype)
            % TODO: random angles            
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            
            % Parameters
            nSamples = 8;
            nChsTotal = stride;
            target_ = 'Sum';
            % nChsTotal x nSamples x nBlks
            dLdZ = randn(nChsTotal,nSamples,nblks,datatype);
            
            % Expected values
            if strcmp(dir,'Right')
                shift = [ 0 0 -1 ]; % Reverse
            elseif strcmp(dir,'Left')
                shift = [ 0 0  1 ]; % Reverse
            else
                shift = [ 0 0 0 ];
            end
            % nChsTotal x nSamples x nBlks
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            Y = dLdZ; %permute(dLdZ,[3 1 2 4]); % [ch ver hor smpl]
            % Block butterfly
            Ys = Y(1:ps,:,:);
            Ya = Y(ps+1:ps+pa,:,:);
            Y =  [ Ys+Ya ; Ys-Ya ]/sqrt(2);
            % Block circular shift
            Y(1:ps,:,:) = circshift(Y(1:ps,:,:),shift);
            % Block butterfly
            Ys = Y(1:ps,:,:);
            Ya = Y(ps+1:ps+pa,:,:);
            Y =  [ Ys+Ya ; Ys-Ya ]/sqrt(2);
            % Output
            expctddLdX = Y; %ipermute(Y,[3 1 2 4]);
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunCSAtomExtension1dLayer(...
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
        %}
    end
    
end