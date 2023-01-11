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
        target = { 'Top', 'Bottom' }
        usegpu = struct( 'true', true, 'false', false);        
    end
    
    methods (TestClassTeardown)
        %{
        function finalCheck(~)
            import tansacnet.lsun.*
            layer = lsunCSAtomExtension1dLayer(...
                'Stride',2,...
                'Direction','Right',...
                'TargetChannels','Bottom');
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
                + "(pt,pb) = (" ...
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

        function testPredictShiftBottomCoefs(testCase, ...
                usegpu, stride, nblks, dir, datatype)
            if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));

            % Parameters
            nSamples = 8;
            nChsTotal = stride;
            target_ = 'Bottom';
            % nChsTotal x nSamples x nBlks
            %X = randn(nrows,ncols,nChsTotal,nSamples,datatype);
            X = randn(nChsTotal,nSamples,nblks,datatype);
            if usegpu
                X = gpuArray(X);
            end

            % Expected values
            if strcmp(dir,'Right')
                shift = [ 0 0 1 ];
            elseif strcmp(dir,'Left')
                shift = [ 0 0 -1 ];
            else
                shift = [ 0 0 0 ];
            end
            %
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            % C-S block buttefly
            Yt = X(1:pt,:,:);
            Yb = X(pt+1:pt+pb,:,:);
            % Block circular shift
            Yb = circshift(Yb,shift);
            % Output
            expctdZ = cat(1,Yt,Yb);
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunCSAtomExtension1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'Name','Qn~',...
                'Direction',dir,...
                'TargetChannels',target_);
            
            % Actual values
            actualZ = layer.predict(X);

            % Evaluation
            if usegpu
                testCase.verifyClass(actualZ,'gpuArray')
                actualZ = gather(actualZ);
                expctdZ = gather(expctdZ);
            end
            testCase.verifyInstanceOf(actualZ,datatype);
            testCase.verifyThat(actualZ,...
                IsEqualTo(expctdZ,'Within',tolObj));

        end

        %{
        function testPredictShiftTopCoefs(testCase, ...
                usegpu, stride, nblks, dir, datatype)
            if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));

            % Parameters
            nSamples = 8;
            nChsTotal = stride;
            target_ = 'Top';
            % nChsTotal x nSamples x nBlks
            X = randn(nChsTotal,nSamples,nblks,datatype);
            if usegpu
                X = gpuArray(X);
            end

            % Expected values
            if strcmp(dir,'Right')
                shift = [ 0 0  1 ];
            elseif strcmp(dir,'Left')
                shift = [ 0 0 -1 ];
            else
                shift = [ 0 0 0 ];
            end
            % nChsTotal x nSamples x nblks
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            % C-S Block butterfly
            Yt = X(1:pt,:,:);
            Yb = X(pt+1:pt+pb,:,:);
            Y =  [ Yt ; Yb ];
            % Block circular shift
            Y(1:pt,:,:) = circshift(Y(1:pt,:,:),shift);
            % Output
            expctdZ = Y; %ipermute(Y,[3 1 2 4]);
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunCSAtomExtension1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'Name','Qn~',...
                'Direction',dir,...
                'TargetChannels',target_);
            
            % Actual values
            actualZ = layer.predict(X);
            
            % Evaluation 
            if usegpu
                testCase.verifyClass(actualZ,'gpuArray')
                actualZ = gather(actualZ);
                expctdZ = gather(expctdZ);
            end
            testCase.verifyInstanceOf(actualZ,datatype);
            testCase.verifyThat(actualZ,...
                IsEqualTo(expctdZ,'Within',tolObj));

        end
        
        function testPredictShiftBottomCoefsWithAngles(testCase, ...
                usegpu, stride, nblks, dir, datatype)
             if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            % TODO: random angles
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            
            % Parameters
            nSamples = 8;
            nChsTotal = stride;
            target_ = 'Bottom';
            % nChsTotal x nSamples x nBlks
            X = randn(nChsTotal,nSamples,nblks,datatype);
            angles = pi/4;
            if usegpu
                X = gpuArray(X);
            end
            
            % Expected values
            if strcmp(dir,'Right')
                shift = [ 0 0 1 ];
            elseif strcmp(dir,'Left')
                shift = [ 0 0 -1 ];
            else
                shift = [ 0 0 0 ];
            end
            %
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            c = cos(angles);
            s = sin(angles);
            %
            Y = permute(X,[1 3 2]);
            Yt = reshape(Y(1:pt,:,:),pt,nblks,nSamples);
            Yb = reshape(Y(pt+1:pt+pb,:,:),pb,nblks,nSamples);
            Zt = zeros(size(Yt),'like',Yt);
            Zb = zeros(size(Yb),'like',Yb);
            for iSample=1:nSamples
                for iblk = 1:nblks
                    Zt(:,iblk,iSample) = c*Yt(:,iblk,iSample)-s*Yb(:,iblk,iSample);
                    Zb(:,iblk,iSample) = s*Yt(:,iblk,iSample)+c*Yb(:,iblk,iSample);
                end
            end
            Y(1:pt,:,:) = Zt;
            Y(pt+1:pt+pb,:,:) = Zb;
            % Block circular shift
            Z = ipermute(Y,[1 3 2]);
            expctdZ(pt+1:pt+pb,:,:) = circshift(Z(pt+1:pt+pb,:,:),shift);            
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunCSAtomExtension1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'Name','Qn~',...
                'Direction',dir,...
                'TargetChannels',target_);
            layer.Angles = angles;
            
            % Actual values
            actualZ = layer.predict(X);
            
            % Evaluation
            if usegpu
                testCase.verifyClass(actualZ,'gpuArray')
                actualZ = gather(actualZ);
                expctdZ = gather(expctdZ);
            end       
            testCase.verifyInstanceOf(actualZ,datatype);
            testCase.verifyThat(actualZ,...
                IsEqualTo(expctdZ,'Within',tolObj));
            
        end
        %}
        %{
        function testPredictShiftBottomCoefsWithRandomAngles(testCase, ...
                usegpu, stride, nblks, dir, datatype)
             if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            % TODO: random angles
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            
            % Parameters
            nSamples = 8;
            nChsTotal = stride;
            nAngles = nChsTotal/2;
            target_ = 'Bottom';
            % nChsTotal x nSamples x nBlks
            X = randn(nChsTotal,nSamples,nblks,datatype);
            angles = randn(nAngles,nblks);
            if usegpu
                X = gpuArray(X);
                angles = gpuArray(angles);
            end
            
            % Expected values
            if strcmp(dir,'Right')
                shift = [ 0 0 1 ];
            elseif strcmp(dir,'Left')
                shift = [ 0 0 -1 ];
            else
                shift = [ 0 0 0 ];
            end
            %
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            C = cos(angles);
            S = sin(angles);
            % Block butterfly
            Yt = X(1:pt,:,:);
            Yb = X(pt+1:pt+pb,:,:);
            Y =  [ Yt+Yb ; Yt-Yb ]/sqrt(2);
            % Block circular shift
            Y(pt+1:pt+pb,:,:) = circshift(Y(pt+1:pt+pb,:,:),shift);
            % Block butterfly
            Yt = Y(1:pt,:,:);
            Yb = Y(pt+1:pt+pb,:,:);
            Y =  [ Yt+Yb ; Yt-Yb ]/sqrt(2);
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
            if usegpu
                testCase.verifyClass(actualZ,'gpuArray')
                actualZ = gather(actualZ);
                expctdZ = gather(expctdZ);
            end       
            testCase.verifyInstanceOf(actualZ,datatype);
            testCase.verifyThat(actualZ,...
                IsEqualTo(expctdZ,'Within',tolObj));
            
        end
        


        function testBackwardShiftBottomCoefs(testCase, ...
                usegpu, stride, nblks, dir, datatype)
             if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            
            % Parameters
            nSamples = 8;
            nChsTotal = stride;
            target_ = 'Bottom';
            % nChsTotal x nSamples x nBlks
            dLdZ = randn(nChsTotal,nSamples,nblks,datatype);
            if usegpu
                dLdZ = gpuArray(dLdZ);
            end
            % Expected values
            if strcmp(dir,'Right')
                shift = [ 0 0 -1 ]; % Reverse
            elseif strcmp(dir,'Left')
                shift = [ 0 0 1 ]; % Reverse
            else
                shift = [ 0 0 0 ]; % Reverse
            end
            % nChsTotal x nSamples x nBlks
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            Y = dLdZ; %permute(dLdZ,[3 1 2 4]); % [ch ver hor smpl]
            % Block butterfly
            Yt = Y(1:pt,:,:);
            Yb = Y(pt+1:pt+pb,:,:);
            Y =  [ Yt+Yb ; Yt-Yb ]/sqrt(2);
            % Block circular shift
            Y(pt+1:pt+pb,:,:) = circshift(Y(pt+1:pt+pb,:,:),shift);
            % Block butterfly
            Yt = Y(1:pt,:,:);
            Yb = Y(pt+1:pt+pb,:,:);
            Y =  [ Yt+Yb ; Yt-Yb ]/sqrt(2);
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
            if usegpu
                testCase.verifyClass(actualdLdX,'gpuArray')
                actualdLdX = gather(actualdLdX);
                expctddLdX = gather(expctddLdX);
            end            
            testCase.verifyInstanceOf(actualdLdX,datatype);
            testCase.verifyThat(actualdLdX,...
                IsEqualTo(expctddLdX,'Within',tolObj));
            
        end

        function testBackwardShiftTopCoefs(testCase, ...
                usegpu, stride, nblks, dir, datatype)
             if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            
            % Parameters
            nSamples = 8;
            nChsTotal = stride;
            target_ = 'Top';
            % nChsTotal x nSamples x nBlks
            dLdZ = randn(nChsTotal,nSamples,nblks,datatype);
                        if usegpu
                dLdZ = gpuArray(dLdZ);
            end
            % Expected values
            if strcmp(dir,'Right')
                shift = [ 0 0 -1 ]; % Reverse
            elseif strcmp(dir,'Left')
                shift = [ 0 0  1 ]; % Reverse
            else
                shift = [ 0 0 0 ];
            end
            % nChsTotal x nSamples x nBlks
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            Y = dLdZ; %permute(dLdZ,[3 1 2 4]); % [ch ver hor smpl]
            % Block butterfly
            Yt = Y(1:pt,:,:);
            Yb = Y(pt+1:pt+pb,:,:);
            Y =  [ Yt+Yb ; Yt-Yb ]/sqrt(2);
            % Block circular shift
            Y(1:pt,:,:) = circshift(Y(1:pt,:,:),shift);
            % Block butterfly
            Yt = Y(1:pt,:,:);
            Yb = Y(pt+1:pt+pb,:,:);
            Y =  [ Yt+Yb ; Yt-Yb ]/sqrt(2);
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
            if usegpu
                testCase.verifyClass(actualdLdX,'gpuArray')
                actualdLdX = gather(actualdLdX);
                expctddLdX = gather(expctddLdX);
            end            
            testCase.verifyInstanceOf(actualdLdX,datatype);
            testCase.verifyThat(actualdLdX,...
                IsEqualTo(expctddLdX,'Within',tolObj));
            
        end
        
        
        function testPredictShiftTopCoefsWithRandomAngles(testCase, ...
                usegpu, stride, nblks, dir, datatype)
             if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            % TODO: random angles            
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            
            % Parameters
            nSamples = 8;
            nChsTotal = stride;
            target_ = 'Top';
            % nChsTotal x nSamples x nBlks
            X = randn(nChsTotal,nSamples,nblks,datatype);
            if usegpu
                X = gpuArray(X);
            end

            % Expected values
            if strcmp(dir,'Right')
                shift = [ 0 0  1 ];
            elseif strcmp(dir,'Left')
                shift = [ 0 0 -1 ];
            else
                shift = [ 0 0 0 ];
            end
            % nChsTotal x nSamples x nblks
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            % Block butterfly
            Yt = X(1:pt,:,:);
            Yb = X(pt+1:pt+pb,:,:);
            Y =  [ Yt+Yb ; Yt-Yb ]/sqrt(2);
            % Block circular shift
            Y(1:pt,:,:) = circshift(Y(1:pt,:,:),shift);
            % Block butterfly
            Yt = Y(1:pt,:,:);
            Yb = Y(pt+1:pt+pb,:,:);
            Y =  [ Yt+Yb ; Yt-Yb ]/sqrt(2);
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
            if usegpu
                testCase.verifyClass(actualZ,'gpuArray')
                actualZ = gather(actualZ);
                expctdZ = gather(expctdZ);
            end       
            testCase.verifyInstanceOf(actualZ,datatype);
            testCase.verifyThat(actualZ,...
                IsEqualTo(expctdZ,'Within',tolObj));
            
        end
        
        function testBackwardShiftBottomCoefsWithRandomAngles(testCase, ...
                usegpu, stride, nblks, dir, datatype)
             if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            % TODO: random angles
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));


            % Parameters
            nSamples = 8;
            nChsTotal = stride;
            target_ = 'Bottom';
            % nChsTotal x nSamples x nBlks
            dLdZ = randn(nChsTotal,nSamples,nblks,datatype);
            if usegpu
                dLdZ = gpuArray(dLdZ);
            end

            % Expected values
            if strcmp(dir,'Right')
                shift = [ 0 0 -1 ]; % Reverse
            elseif strcmp(dir,'Left')
                shift = [ 0 0 1 ]; % Reverse
            else
                shift = [ 0 0 0 ]; % Reverse
            end
            % nChsTotal x nSamples x nBlks
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            Y = dLdZ; %permute(dLdZ,[3 1 2 4]); % [ch ver hor smpl]
            % Block butterfly
            Yt = Y(1:pt,:,:);
            Yb = Y(pt+1:pt+pb,:,:);
            Y =  [ Yt+Yb ; Yt-Yb ]/sqrt(2);
            % Block circular shift
            Y(pt+1:pt+pb,:,:) = circshift(Y(pt+1:pt+pb,:,:),shift);
            % Block butterfly
            Yt = Y(1:pt,:,:);
            Yb = Y(pt+1:pt+pb,:,:);
            Y =  [ Yt+Yb ; Yt-Yb ]/sqrt(2);
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
            if usegpu
                testCase.verifyClass(actualdLdX,'gpuArray')
                actualdLdX = gather(actualdLdX);
                expctddL = gather(expctddLdX);
            end       
            testCase.verifyInstanceOf(actualdLdX,datatype);
            testCase.verifyThat(actualdLdX,...
                IsEqualTo(expctddLdX,'Within',tolObj));
            
        end

        function testBackwardShiftTopCoefsWithRandomAngles(testCase, ...
                usegpu, stride, nblks, dir, datatype)
             if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            % TODO: random angles            
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            
            % Parameters
            nSamples = 8;
            nChsTotal = stride;
            target_ = 'Top';
            % nChsTotal x nSamples x nBlks
            dLdZ = randn(nChsTotal,nSamples,nblks,datatype);
            if usegpu
                dLdZ = gpuArray(dLdZ);
            end

            % Expected values
            if strcmp(dir,'Right')
                shift = [ 0 0 -1 ]; % Reverse
            elseif strcmp(dir,'Left')
                shift = [ 0 0  1 ]; % Reverse
            else
                shift = [ 0 0 0 ];
            end
            % nChsTotal x nSamples x nBlks
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            Y = dLdZ; %permute(dLdZ,[3 1 2 4]); % [ch ver hor smpl]
            % Block butterfly
            Yt = Y(1:pt,:,:);
            Yb = Y(pt+1:pt+pb,:,:);
            Y =  [ Yt+Yb ; Yt-Yb ]/sqrt(2);
            % Block circular shift
            Y(1:pt,:,:) = circshift(Y(1:pt,:,:),shift);
            % Block butterfly
            Yt = Y(1:pt,:,:);
            Yb = Y(pt+1:pt+pb,:,:);
            Y =  [ Yt+Yb ; Yt-Yb ]/sqrt(2);
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
            if usegpu
                testCase.verifyClass(actualdLdX,'gpuArray')
                actualdLdX = gather(actualdLdX);
                expctddLdX = gather(expctddLdX);
            end       
            testCase.verifyInstanceOf(actualdLdX,datatype);
            testCase.verifyThat(actualdLdX,...
                IsEqualTo(expctddLdX,'Within',tolObj));
            
        end
        %}
    end
    
end