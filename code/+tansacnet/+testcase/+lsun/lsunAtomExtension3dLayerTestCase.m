classdef lsunAtomExtension3dLayerTestCase < matlab.unittest.TestCase
    %NSOLTATOMEXTENSION3DLAYERTESTCASE
    %
    %   コンポーネント別に入力(nComponents=1のみサポート):
    %      nChsTotal x nRows x nCols x nLays x nSamples
    %
    %   コンポーネント別に出力(nComponents=1のみサポート):
    %      nChsTotal x nRows x nCols x nLays x nSamples
    %
    %
    % Requirements: MATLAB R2020b
    %
    % Copyright (c) 2020-2022, Eisuke KOBAYASHI, Shogo MURAMATSU
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
        stride = { [2 2 2], [4 4 4] };
        datatype = { 'single', 'double' };
        nrows = struct('small', 4,'medium', 8, 'large', 16);
        ncols = struct('small', 4,'medium', 8, 'large', 16);
        nlays = struct('small', 4,'medium', 8, 'large', 16);
        dir = { 'Right', 'Left', 'Up', 'Down', 'Back','Front' };
        target = { 'Sum', 'Difference' }
    end
    
    methods (TestClassTeardown)
        function finalCheck(~)
            import tansacnet.lsun.*
            layer = lsunAtomExtension3dLayer(...
                'Stride',[2 2 2],...
                'Direction','Right',...
                'TargetChannels','Difference');
            fprintf("\n --- Check layer for 3-D images ---\n");
            checkLayer(layer,[8 8 8 8],...
                'ObservationDimension',5,...
                'CheckCodegenCompatibility',false)
        end
    end
    
    methods (Test)
        
        function testConstructor(testCase, stride, target)
            
            % Parameters
            nChsTotal = prod(stride);

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
            layer = lsunAtomExtension3dLayer(...
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
        
        function testPredictGrayscaleShiftDifferenceCoefs(testCase, ...
                stride, nrows, ncols, nlays, dir, datatype)
            
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            
            % Parameters
            nSamples = 8;
            nChsTotal = prod(stride);
            target_ = 'Difference';
            % nChsTotal x nRows x nCols x nSamples
            %X = randn(nrows,ncols,nChsTotal,nSamples,datatype);
            X = randn(nChsTotal,nrows,ncols,nlays,nSamples,datatype);
            
            % Expected values
            if strcmp(dir,'Right')
                shift = [ 0 0  1 0 0  ];
            elseif strcmp(dir,'Left')
                shift = [ 0 0 -1 0 0  ];
            elseif strcmp(dir,'Down')
                shift = [ 0  1 0 0 0  ];
            elseif strcmp(dir,'Up')
                shift = [ 0 -1 0 0 0  ];
            elseif strcmp(dir,'Back')
                shift = [ 0  0 0 1 0  ];
            elseif strcmp(dir,'Front')
                shift = [ 0 0 0 -1 0  ];
            else
                shift = [ 0 0 0 0 0 ];
            end
            % nRows x nCols x nChsTotal x nSamples
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            % Block butterfly
            Ys = X(1:ps,:,:,:,:);
            Ya = X(ps+1:ps+pa,:,:,:,:);
            Y =  [ Ys+Ya ; Ys-Ya ]/sqrt(2);
            % Block circular shift
            Y(ps+1:ps+pa,:,:,:,:) = circshift(Y(ps+1:ps+pa,:,:,:,:),shift);
            % Block butterfly
            Ys = Y(1:ps,:,:,:,:);
            Ya = Y(ps+1:ps+pa,:,:,:,:);
            Y =  [ Ys+Ya ; Ys-Ya ]/sqrt(2);
            % Output
            expctdZ = Y; %ipermute(Y,[4 1 2 3 5]);
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunAtomExtension3dLayer(...
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
         
        function testPredictGrayscaleShiftSumCoefs(testCase, ...
                stride, nrows, ncols, nlays, dir, datatype)
            
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            
            % Parameters
            nSamples = 8;
            nChsTotal = prod(stride);
            target_ = 'Sum';
            % nChsTotal x nRows x nCols x nLays x nSamples
            %X = randn(nrows,ncols,nChsTotal,nSamples,datatype);
            X = randn(nChsTotal,nrows,ncols,nlays,nSamples,datatype);
            
            % Expected values
            if strcmp(dir,'Right')
                shift = [ 0 0  1 0 0 ];
            elseif strcmp(dir,'Left')
                shift = [ 0 0 -1 0 0];
            elseif strcmp(dir,'Down')
                shift = [ 0  1 0 0 0];
            elseif strcmp(dir,'Up')
                shift = [ 0 -1 0 0 0];
            elseif strcmp(dir,'Back')
                shift = [ 0  0 0 1 0];
            elseif strcmp(dir,'Front')
                shift = [ 0 0 0 -1 0];
            else
                shift = [ 0 0 0 0 0];
            end
            % nChsTotal x nRows x nCols x nSamples
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            % Block butterfly
            Ys = X(1:ps,:,:,:,:);
            Ya = X(ps+1:ps+pa,:,:,:,:);
            Y =  [ Ys+Ya ; Ys-Ya ]/sqrt(2);
            % Block circular shift
            Y(1:ps,:,:,:,:) = circshift(Y(1:ps,:,:,:,:),shift);
            % Block butterfly
            Ys = Y(1:ps,:,:,:,:);
            Ya = Y(ps+1:ps+pa,:,:,:,:);
            Y =  [ Ys+Ya ; Ys-Ya ]/sqrt(2);
            % Output
            expctdZ = Y; %ipermute(Y,[3 1 2 4]);
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunAtomExtension3dLayer(...
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
        
        function testBackwardGrayscaleShiftDifferenceCoefs(testCase, ...
                stride, nrows, ncols, nlays, dir, datatype)
            
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            
            % Parameters
            nSamples = 8;
            nChsTotal = prod(stride);
            target_ = 'Difference';
            % nChsTotal x nRows x nCols x nSamples
            %dLdZ = randn(nrows,ncols,nChsTotal,nSamples,datatype);
            dLdZ = randn(nChsTotal,nrows,ncols,nlays,nSamples,datatype);
            
            % Expected values
            if strcmp(dir,'Right')
                shift = [ 0 0 -1 0 0  ]; % Reverse
            elseif strcmp(dir,'Left')
                shift = [ 0 0 1 0 0  ]; % Reverse
            elseif strcmp(dir,'Down')
                shift = [ 0  -1 0 0 0  ]; % Reverse
            elseif strcmp(dir,'Up')
                shift = [ 0 1 0 0 0  ]; % Reverse
            elseif strcmp(dir,'Back')
                shift = [ 0  0 0 -1 0  ]; % Reverse
            elseif strcmp(dir,'Front')
                shift = [ 0 0 0 1 0  ]; % Reverse
            else
                shift = [ 0 0 0 0 0 ];
            end
            % nChsTotal x nRows x nCols x nSamples
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            Y = dLdZ; %permute(dLdZ,[4 1 2 3 5]); % [ch ver hor dep smpl]
            % Block butterfly
            Ys = Y(1:ps,:,:,:,:);
            Ya = Y(ps+1:ps+pa,:,:,:,:);
            Y =  [ Ys+Ya ; Ys-Ya ]/sqrt(2);
            % Block circular shift
            Y(ps+1:ps+pa,:,:,:,:) = circshift(Y(ps+1:ps+pa,:,:,:,:),shift);
            % Block butterfly
            Ys = Y(1:ps,:,:,:,:);
            Ya = Y(ps+1:ps+pa,:,:,:,:);
            Y =  [ Ys+Ya ; Ys-Ya ]/sqrt(2);
            % Output
            expctddLdX = Y; %ipermute(Y,[3 1 2 4]);
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunAtomExtension3dLayer(...
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
        
        function testBackwardGrayscaleShiftSumCoefs(testCase, ...
                stride, nrows, ncols, nlays, dir, datatype)
            
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            
            % Parameters
            nSamples = 8;
            nChsTotal = prod(stride);
            target_ = 'Sum';
            % nChsTotal x nRows x nCols x nSamples
            %dLdZ = randn(nrows,ncols,nChsTotal,nSamples,datatype);
            dLdZ = randn(nChsTotal,nrows,ncols,nlays,nSamples,datatype);
            
            % Expected values
            if strcmp(dir,'Right')
                shift = [ 0 0 -1 0 0 ]; % Reverse
            elseif strcmp(dir,'Left')
                shift = [ 0 0 1 0 0]; % Reverse
            elseif strcmp(dir,'Down')
                shift = [ 0 -1 0 0 0]; % Reverse
            elseif strcmp(dir,'Up')
                shift = [ 0 1 0 0 0]; % Reverse
            elseif strcmp(dir,'Back')
                shift = [ 0  0 0 -1 0]; % Reverse
            elseif strcmp(dir,'Front')
                shift = [ 0 0 0 1 0]; % Reverse
            else
                shift = [ 0 0 0 0 0];
            end
            % nChsTotal x nRows x nCols x nSamples
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            Y = dLdZ; %permute(dLdZ,[3 1 2 4]); % [ch ver hor smpl]
            % Block butterfly
            Ys = Y(1:ps,:,:,:,:);
            Ya = Y(ps+1:ps+pa,:,:,:,:);
            Y =  [ Ys+Ya ; Ys-Ya ]/sqrt(2);
            % Block circular shift
            Y(1:ps,:,:,:,:) = circshift(Y(1:ps,:,:,:,:),shift);
            % Block butterfly
            Ys = Y(1:ps,:,:,:,:);
            Ya = Y(ps+1:ps+pa,:,:,:,:);
            Y =  [ Ys+Ya ; Ys-Ya ]/sqrt(2);
            % Output
            expctddLdX = Y; %ipermute(Y,[3 1 2 4]);
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunAtomExtension3dLayer(...
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