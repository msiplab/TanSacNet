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
        mode = { 'Analysis', 'Synthesis' }
        usegpu = struct( 'true', true, 'false', false);        
    end
    
    methods (TestClassTeardown)

        function finalCheck(~)
            import tansacnet.lsun.*
            layer = lsunCSAtomExtension1dLayer(...
                'Stride',2,...
                'Direction','Right',...
                'NumberOfBlocks',4,...
                'Mode','Analysis',...
                'TargetChannels','Bottom');
            fprintf("\n --- Check layer for 1-D sequences ---\n");
            checkLayer(layer,[2 8 4],...
                'ObservationDimension',2,...
                'CheckCodegenCompatibility',true)

            layer = lsunCSAtomExtension1dLayer(...
                'Stride',2,...
                'Direction','Left',...
                'NumberOfBlocks',4,...
                'Mode','Synthesis',...
                'TargetChannels','Top');
            fprintf("\n --- Check layer for 1-D sequences ---\n");
            checkLayer(layer,[2 8 4],...
                'ObservationDimension',2,...
                'CheckCodegenCompatibility',true)

        end

    end
    
    methods (Test)
        
        function testConstructor(testCase, ...
                mode, stride, target)
            
            % Parameters
            nChsTotal = stride;

            % Expected values
            expctdName = 'Qn';
            expctdDirection = 'Right';
            expctdTargetChannels = target;
            expctdMode = mode;
            expctdDescription = expctdMode ...
                + " LSUN C-S transform w/ " ...
                + expctdDirection ... 
                + " shift the " ...
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
                'Mode',expctdMode,...
                'TargetChannels',expctdTargetChannels);
            
            % Actual values
            actualName = layer.Name;
            actualDirection = layer.Direction;
            actualTargetChannels = layer.TargetChannels;
            actualMode = layer.Mode;
            actualDescription = layer.Description;
            
            % Evaluation
            testCase.verifyEqual(actualName,expctdName);
            testCase.verifyEqual(actualDirection,expctdDirection);
            testCase.verifyEqual(actualTargetChannels,expctdTargetChannels);
            testCase.verifyEqual(actualMode,expctdMode);
            testCase.verifyEqual(actualDescription,expctdDescription);
        end

        function testPredictAnalysisShiftBottomCoefs(testCase, ...
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
            mode_ = 'Analysis';
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
                'Mode',mode_,...
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

        function testPredictAnalysisShiftTopCoefs(testCase, ...
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
            mode_ = 'Analysis';
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
            Yt = X(1:pt,:,:);
            Yb = X(pt+1:pt+pb,:,:);
            % Block circular shift
            Yt = circshift(Yt,shift);
            % Output
            expctdZ = cat(1,Yt,Yb);
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunCSAtomExtension1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'Name','Qn~',...
                'Direction',dir,...
                'Mode',mode_,...
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

        function testPredictAnalysisShiftBottomCoefsWithAnglePi4(testCase, ...
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
            mode_ = 'Analysis';
            % nChsTotal x nSamples x nBlks
            X = randn(nChsTotal,nSamples,nblks,datatype);
            angle = pi/4;
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
            Yt = X(1:pt,:,:);
            Yb = X(pt+1:pt+pb,:,:);
            % Block circular shift
            Yb = circshift(Yb,shift);
            % C-S block butterfly            
            c = cos(angle);
            s = sin(angle);
            Zct = zeros(size(Yt),'like',Yt);
            Zst = zeros(size(Yt),'like',Yt);
            Zcb = zeros(size(Yb),'like',Yb);
            Zsb = zeros(size(Yb),'like',Yb);            
            for iSample=1:nSamples
                for iblk = 1:nblks
                    Zct(:,iSample,iblk) = c*Yt(:,iSample,iblk);
                    Zst(:,iSample,iblk) = s*Yt(:,iSample,iblk);
                    Zcb(:,iSample,iblk) = c*Yb(:,iSample,iblk);
                    Zsb(:,iSample,iblk) = s*Yb(:,iSample,iblk);                    
                end
            end
            Yt = Zct-Zsb;
            Yb = Zst+Zcb;
            expctdZ = cat(1,Yt,Yb);
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunCSAtomExtension1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'Name','Qn~',...
                'Direction',dir,...
                'Mode',mode_,...
                'TargetChannels',target_);
            layer.Angles = angle;
            
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

        function testPredictAnalysisShiftBottomCoefsWithRandomAngles(testCase, ...
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
            nAngles = nChsTotal/2;
            target_ = 'Bottom';
            mode_ = 'Analysis';
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
            % 
            Yt = X(1:pt,:,:);
            Yb = X(pt+1:pt+pb,:,:);
            % Block circular shift
            Yb = circshift(Yb,shift);
            % C-S Block butterfly
            C = cos(angles);
            S = sin(angles);
            Zct = zeros(size(Yt),'like',Yt);
            Zst = zeros(size(Yt),'like',Yt);
            Zcb = zeros(size(Yb),'like',Yb);
            Zsb = zeros(size(Yb),'like',Yb);            
            for iSample=1:nSamples
                for iblk = 1:nblks
                    Zct(:,iSample,iblk) = C(:,iblk).*Yt(:,iSample,iblk);
                    Zst(:,iSample,iblk) = S(:,iblk).*Yt(:,iSample,iblk);
                    Zcb(:,iSample,iblk) = C(:,iblk).*Yb(:,iSample,iblk);
                    Zsb(:,iSample,iblk) = S(:,iblk).*Yb(:,iSample,iblk);                    
                end
            end
            Yt = Zct-Zsb;
            Yb = Zst+Zcb;
            expctdZ = cat(1,Yt,Yb);
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunCSAtomExtension1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...                
                'Name','Qn~',...
                'Direction',dir,...
                'Mode',mode_,...
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

        function testPredictAnalysisShiftTopCoefsWithRandomAngles(testCase, ...
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
            nAngles = nChsTotal/2;
            target_ = 'Top';
            mode_ = 'Analysis';
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
            % 
            Yt = X(1:pt,:,:);
            Yb = X(pt+1:pt+pb,:,:);
            % Block circular shift
            Yt = circshift(Yt,shift);
            % C-S Block butterfly
            C = cos(angles);
            S = sin(angles);
            Zct = zeros(size(Yt),'like',Yt);
            Zst = zeros(size(Yt),'like',Yt);
            Zcb = zeros(size(Yb),'like',Yb);
            Zsb = zeros(size(Yb),'like',Yb);            
            for iSample=1:nSamples
                for iblk = 1:nblks
                    Zct(:,iSample,iblk) = C(:,iblk).*Yt(:,iSample,iblk);
                    Zst(:,iSample,iblk) = S(:,iblk).*Yt(:,iSample,iblk);
                    Zcb(:,iSample,iblk) = C(:,iblk).*Yb(:,iSample,iblk);
                    Zsb(:,iSample,iblk) = S(:,iblk).*Yb(:,iSample,iblk);                    
                end
            end
            Yt = Zct-Zsb;
            Yb = Zst+Zcb;
            expctdZ = cat(1,Yt,Yb);
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunCSAtomExtension1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...                
                'Name','Qn~',...
                'Direction',dir,...
                'Mode',mode_,...
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

        function testBackwardAnalysisShiftBottomCoefs(testCase, ...
                usegpu, stride, nblks, dir, datatype)
             if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-5));
            
            % Parameters
            nSamples = 8;
            nChsTotal = stride;
            nAngles = nChsTotal/2;            
            target_ = 'Bottom';
            mode_ = 'Analysis';
            % nChsTotal x nSamples x nBlks
            X = randn(nChsTotal,nSamples,nblks,datatype);            
            dLdZ = randn(nChsTotal,nSamples,nblks,datatype);
            if usegpu
                X = gpuArray(X);
                dLdZ = gpuArray(dLdZ);
            end
            % Expected values
            if strcmp(dir,'Right')
                shift = [ 0 0 1 ];  
            elseif strcmp(dir,'Left')
                shift = [ 0 0 -1 ]; 
            else
                shift = [ 0 0 0 ]; 
            end
            % nChsTotal x nSamples x nBlks
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            % 
            Yt = dLdZ(1:pt,:,:);
            Yb = dLdZ(pt+1:pt+pb,:,:);
            % C-S Block butterfly
            %Yt = Yt;
            %Yb = Yb;
            % Block circular shift (Revserse)
            Yb = circshift(Yb,-shift); % Bottom, Revserse
            expctddLdX = cat(1,Yt,Yb);
            
            % dLdWi = <dLdZ,(dVdWi)X>
            expctddLdW = zeros(nAngles,nblks,datatype);
            c_top = X(1:pt,:,:);
            c_btm = X(pt+1:pt+pb,:,:);
            % Block circular shift
            c_btm = circshift(c_btm,shift); % Bottom
            % C-S differential
            for iAngle = 1:nAngles
                dS_ = zeros(nAngles,nblks);
                dS_(iAngle,:) = ones(1,nblks);
                for iblk = 1:nblks
                    c_top_iblk = -dS_(:,iblk).*c_btm(:,:,iblk);                    
                    c_btm_iblk =  dS_(:,iblk).*c_top(:,:,iblk);
                    c_iblk = cat(1,c_top_iblk,c_btm_iblk);                    
                    dldz_iblk = dLdZ(:,:,iblk);
                    expctddLdW(iAngle,iblk) = sum(dldz_iblk.*c_iblk,'all');
                end
            end
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunCSAtomExtension1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'Name','Qn',...
                'Direction',dir,...
                'Mode',mode_,...
                'TargetChannels',target_);
            
            % Actual values
            [actualdLdX,actualdLdW] = layer.backward(X,[],dLdZ,[]);
            
            % Evaluation
            if usegpu
                testCase.verifyClass(actualdLdX,'gpuArray')
                testCase.verifyClass(actualdLdW,'gpuArray')
                actualdLdX = gather(actualdLdX);
                expctddLdX = gather(expctddLdX);
                actualdLdW = gather(actualdLdW);
                expctddLdW = gather(expctddLdW);                
            end            
            testCase.verifyInstanceOf(actualdLdX,datatype);
            testCase.verifyInstanceOf(actualdLdW,datatype);            
            testCase.verifyThat(actualdLdX,...
                IsEqualTo(expctddLdX,'Within',tolObj));
            testCase.verifyThat(actualdLdW,...
                IsEqualTo(expctddLdW,'Within',tolObj));            
            
        end

        function testBackwardAnalysisShiftTopCoefs(testCase, ...
                usegpu, stride, nblks, dir, datatype)
             if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-5));
            
            % Parameters
            nSamples = 8;
            nChsTotal = stride;
            nAngles = nChsTotal/2;            
            target_ = 'Top';
            mode_ = 'Analysis';
            % nChsTotal x nSamples x nBlks
            X = randn(nChsTotal,nSamples,nblks,datatype);            
            dLdZ = randn(nChsTotal,nSamples,nblks,datatype);
            if usegpu
                X = gpuArray(X);
                dLdZ = gpuArray(dLdZ);
            end
            % Expected values
            if strcmp(dir,'Right')
                shift = [ 0 0 1 ];  
            elseif strcmp(dir,'Left')
                shift = [ 0 0 -1 ]; 
            else
                shift = [ 0 0 0 ]; 
            end
            % nChsTotal x nSamples x nBlks
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            % 
            Yt = dLdZ(1:pt,:,:);
            Yb = dLdZ(pt+1:pt+pb,:,:);
            % C-S Block butterfly
            %Yt = Yt;
            %Yb = Yb;
            % Block circular shift (Revserse)
            Yt = circshift(Yt,-shift); % Top, Revserse
            expctddLdX = cat(1,Yt,Yb);
            
            % dLdWi = <dLdZ,(dVdWi)X>
            expctddLdW = zeros(nAngles,nblks,datatype);
            c_top = X(1:pt,:,:);
            c_btm = X(pt+1:pt+pb,:,:);
            % Block circular shift
            c_top = circshift(c_top,shift); % Top
            % C-S differential
            for iAngle = 1:nAngles
                dS_ = zeros(nAngles,nblks);
                dS_(iAngle,:) = ones(1,nblks);
                for iblk = 1:nblks
                    c_top_iblk = -dS_(:,iblk).*c_btm(:,:,iblk);                    
                    c_btm_iblk =  dS_(:,iblk).*c_top(:,:,iblk);
                    c_iblk = cat(1,c_top_iblk,c_btm_iblk);                    
                    dldz_iblk = dLdZ(:,:,iblk);
                    expctddLdW(iAngle,iblk) = sum(dldz_iblk.*c_iblk,'all');
                end
            end
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunCSAtomExtension1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'Name','Qn',...
                'Direction',dir,...
                'Mode',mode_,...
                'TargetChannels',target_);
            
            % Actual values
            [actualdLdX,actualdLdW] = layer.backward(X,[],dLdZ,[]);
            
            % Evaluation
            if usegpu
                testCase.verifyClass(actualdLdX,'gpuArray')
                testCase.verifyClass(actualdLdW,'gpuArray')
                actualdLdX = gather(actualdLdX);
                expctddLdX = gather(expctddLdX);
                actualdLdW = gather(actualdLdW);
                expctddLdW = gather(expctddLdW);                
            end            
            testCase.verifyInstanceOf(actualdLdX,datatype);
            testCase.verifyInstanceOf(actualdLdW,datatype);            
            testCase.verifyThat(actualdLdX,...
                IsEqualTo(expctddLdX,'Within',tolObj));
            testCase.verifyThat(actualdLdW,...
                IsEqualTo(expctddLdW,'Within',tolObj));            
            
         end

        function testBackwardAnalysisShiftBottomCoefsWithAnglePi4(testCase, ...
                usegpu, stride, nblks, dir, datatype)
             if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-5));
            
            % Parameters
            nSamples = 8;
            nChsTotal = stride;
            nAngles = nChsTotal/2;            
            target_ = 'Bottom';
            mode_ = 'Analysis';
            % nChsTotal x nSamples x nBlks
            angle = pi/4;
            X = randn(nChsTotal,nSamples,nblks,datatype);            
            dLdZ = randn(nChsTotal,nSamples,nblks,datatype);
            if usegpu
                X = gpuArray(X);
                dLdZ = gpuArray(dLdZ);
            end
            % Expected values
            if strcmp(dir,'Right')
                shift = [ 0 0 1 ];  
            elseif strcmp(dir,'Left')
                shift = [ 0 0 -1 ]; 
            else
                shift = [ 0 0 0 ]; 
            end
            % nChsTotal x nSamples x nBlks
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            % 
            Zt = dLdZ(1:pt,:,:);
            Zb = dLdZ(pt+1:pt+pb,:,:);
            % C-S Block butterfly (Transpose)
            c = cos(angle);
            s = sin(angle);
            Yt =  c.*Zt + s.*Zb;
            Yb = -s.*Zt + c.*Zb;
            % Block circular shift (Revserse)
            Yb = circshift(Yb,-shift); % Bottom, Revserse
            expctddLdX = cat(1,Yt,Yb);
            
            % dLdWi = <dLdZ,(dVdWi)X>
            expctddLdW = zeros(nAngles,nblks,datatype);
            c_top = X(1:pt,:,:);
            c_btm = X(pt+1:pt+pb,:,:);
            % Block circular shift
            c_btm = circshift(c_btm,shift); % Bottom
            % C-S differential
            for iAngle = 1:nAngles
                dC_ = zeros(nAngles,nblks);
                dS_ = zeros(nAngles,nblks);
                dC_(iAngle,:) = -sin(angle)*ones(1,nblks);
                dS_(iAngle,:) =  cos(angle)*ones(1,nblks);
                for iblk = 1:nblks
                    c_top_iblk = dC_(:,iblk).*c_top(:,:,iblk) ...
                        - dS_(:,iblk).*c_btm(:,:,iblk);                    
                    c_btm_iblk = dS_(:,iblk).*c_top(:,:,iblk) ...
                        + dC_(:,iblk).*c_btm(:,:,iblk);
                    c_iblk = cat(1,c_top_iblk,c_btm_iblk);                    
                    dldz_iblk = dLdZ(:,:,iblk);
                    expctddLdW(iAngle,iblk) = sum(dldz_iblk.*c_iblk,'all');
                end
            end
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunCSAtomExtension1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'Name','Qn',...
                'Direction',dir,...
                'Mode',mode_,...
                'TargetChannels',target_);
            layer.Angles = angle;
            
            % Actual values
            [actualdLdX,actualdLdW] = layer.backward(X,[],dLdZ,[]);
            
            % Evaluation
            if usegpu
                testCase.verifyClass(actualdLdX,'gpuArray')
                testCase.verifyClass(actualdLdW,'gpuArray')
                actualdLdX = gather(actualdLdX);
                expctddLdX = gather(expctddLdX);
                actualdLdW = gather(actualdLdW);
                expctddLdW = gather(expctddLdW);                
            end            
            testCase.verifyInstanceOf(actualdLdX,datatype);
            testCase.verifyInstanceOf(actualdLdW,datatype);            
            testCase.verifyThat(actualdLdX,...
                IsEqualTo(expctddLdX,'Within',tolObj));
            testCase.verifyThat(actualdLdW,...
                IsEqualTo(expctddLdW,'Within',tolObj));            
            
        end
         
        function testBackwardAnalysisShiftBottomCoefsWithRandomAngles(testCase, ...
                usegpu, stride, nblks, dir, datatype)
             if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-5));
            
            % Parameters
            nSamples = 8;
            nChsTotal = stride;
            nAngles = nChsTotal/2;            
            target_ = 'Bottom';
            mode_ = 'Analysis';
            % nChsTotal x nSamples x nBlks
            angles = randn(nAngles,nblks);
            X = randn(nChsTotal,nSamples,nblks,datatype);            
            dLdZ = randn(nChsTotal,nSamples,nblks,datatype);
            if usegpu
                X = gpuArray(X);
                dLdZ = gpuArray(dLdZ);
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
            % nChsTotal x nSamples x nBlks
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            % 
            Yt = permute(dLdZ(1:pt,:,:),[1 3 2]);
            Yb = permute(dLdZ(pt+1:pt+pb,:,:),[1 3 2]);
            % C-S Block butterfly (Transpose)
            C_ = cos(angles);
            S_ = sin(angles);
            Zt = zeros(size(Yt),'like',Yt);
            Zb = zeros(size(Yb),'like',Yb);
            for iSample = 1:nSamples
                for iblk = 1:nblks
                    Zt(:,iblk,iSample) =  C_(:,iblk).*Yt(:,iblk,iSample) ...
                        + S_(:,iblk).*Yb(:,iblk,iSample);
                    Zb(:,iblk,iSample) = -S_(:,iblk).*Yt(:,iblk,iSample) ...
                        + C_(:,iblk).*Yb(:,iblk,iSample);
                end
            end
            Yt = ipermute(Zt,[1 3 2]);
            Yb = ipermute(Zb,[1 3 2]);
            % Block circular shift (Revserse)
            Yb = circshift(Yb,-shift); % Bottom, Revserse
            expctddLdX = cat(1,Yt,Yb);
            
            % dLdWi = <dLdZ,(dVdWi)X>
            expctddLdW = zeros(nAngles,nblks,datatype);
            c_top = X(1:pt,:,:);
            c_btm = X(pt+1:pt+pb,:,:);
            % Block circular shift
            c_btm = circshift(c_btm,shift); % Bottom
            % C-S differential
            for iAngle = 1:nAngles
                dC_ = zeros(nAngles,nblks);
                dS_ = zeros(nAngles,nblks);
                dC_(iAngle,:) = -sin(angles(iAngle,:));
                dS_(iAngle,:) =  cos(angles(iAngle,:));
                for iblk = 1:nblks
                    c_top_iblk = dC_(:,iblk).*c_top(:,:,iblk) ...
                        - dS_(:,iblk).*c_btm(:,:,iblk);                    
                    c_btm_iblk = dS_(:,iblk).*c_top(:,:,iblk) ...
                        + dC_(:,iblk).*c_btm(:,:,iblk);
                    c_iblk = cat(1,c_top_iblk,c_btm_iblk);                    
                    dldz_iblk = dLdZ(:,:,iblk);
                    expctddLdW(iAngle,iblk) = sum(dldz_iblk.*c_iblk,'all');
                end
            end
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunCSAtomExtension1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'Name','Qn',...
                'Direction',dir,...
                'Mode',mode_,...
                'TargetChannels',target_);
            layer.Angles = angles;
            
            % Actual values
            [actualdLdX,actualdLdW] = layer.backward(X,[],dLdZ,[]);
            
            % Evaluation
            if usegpu
                testCase.verifyClass(actualdLdX,'gpuArray')
                testCase.verifyClass(actualdLdW,'gpuArray')
                actualdLdX = gather(actualdLdX);
                expctddLdX = gather(expctddLdX);
                actualdLdW = gather(actualdLdW);
                expctddLdW = gather(expctddLdW);                
            end            
            testCase.verifyInstanceOf(actualdLdX,datatype);
            testCase.verifyInstanceOf(actualdLdW,datatype);            
            testCase.verifyThat(actualdLdX,...
                IsEqualTo(expctddLdX,'Within',tolObj));
            testCase.verifyThat(actualdLdW,...
                IsEqualTo(expctddLdW,'Within',tolObj));            
            
        end
         
        function testBackwardAnalysisShiftTopCoefsWithRandomAngles(testCase, ...
                usegpu, stride, nblks, dir, datatype)
             if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-5));
            
            % Parameters
            nSamples = 8;
            nChsTotal = stride;
            nAngles = nChsTotal/2;            
            target_ = 'Top';
            mode_ = 'Analysis';
            % nChsTotal x nSamples x nBlks
            angles = randn(nAngles,nblks);
            X = randn(nChsTotal,nSamples,nblks,datatype);            
            dLdZ = randn(nChsTotal,nSamples,nblks,datatype);
            if usegpu
                X = gpuArray(X);
                dLdZ = gpuArray(dLdZ);
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
            % nChsTotal x nSamples x nBlks
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            % 
            Yt = permute(dLdZ(1:pt,:,:),[1 3 2]);
            Yb = permute(dLdZ(pt+1:pt+pb,:,:),[1 3 2]);
            % C-S Block butterfly (Transpose)
            C_ = cos(angles);
            S_ = sin(angles);
            Zt = zeros(size(Yt),'like',Yt);
            Zb = zeros(size(Yb),'like',Yb);
            for iSample = 1:nSamples
                for iblk = 1:nblks
                    Zt(:,iblk,iSample) =  C_(:,iblk).*Yt(:,iblk,iSample) ...
                        + S_(:,iblk).*Yb(:,iblk,iSample);
                    Zb(:,iblk,iSample) = -S_(:,iblk).*Yt(:,iblk,iSample) ...
                        + C_(:,iblk).*Yb(:,iblk,iSample);
                end
            end
            Yt = ipermute(Zt,[1 3 2]);
            Yb = ipermute(Zb,[1 3 2]);
            % Block circular shift (Revserse)
            Yt = circshift(Yt,-shift); % Top, Revserse
            expctddLdX = cat(1,Yt,Yb);
            
            % dLdWi = <dLdZ,(dVdWi)X>
            expctddLdW = zeros(nAngles,nblks,datatype);
            c_top = X(1:pt,:,:);
            c_btm = X(pt+1:pt+pb,:,:);
            % Block circular shift
            c_top = circshift(c_top,shift); % Top
            % C-S differential
            for iAngle = 1:nAngles
                dC_ = zeros(nAngles,nblks);
                dS_ = zeros(nAngles,nblks);
                dC_(iAngle,:) = -sin(angles(iAngle,:));
                dS_(iAngle,:) =  cos(angles(iAngle,:));
                for iblk = 1:nblks
                    c_top_iblk = dC_(:,iblk).*c_top(:,:,iblk) ...
                        - dS_(:,iblk).*c_btm(:,:,iblk);                    
                    c_btm_iblk = dS_(:,iblk).*c_top(:,:,iblk) ...
                        + dC_(:,iblk).*c_btm(:,:,iblk);
                    c_iblk = cat(1,c_top_iblk,c_btm_iblk);                    
                    dldz_iblk = dLdZ(:,:,iblk);
                    expctddLdW(iAngle,iblk) = sum(dldz_iblk.*c_iblk,'all');
                end
            end
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunCSAtomExtension1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'Name','Qn',...
                'Direction',dir,...
                'Mode',mode_,...
                'TargetChannels',target_);
            layer.Angles = angles;
            
            % Actual values
            [actualdLdX,actualdLdW] = layer.backward(X,[],dLdZ,[]);
            
            % Evaluation
            if usegpu
                testCase.verifyClass(actualdLdX,'gpuArray')
                testCase.verifyClass(actualdLdW,'gpuArray')
                actualdLdX = gather(actualdLdX);
                expctddLdX = gather(expctddLdX);
                actualdLdW = gather(actualdLdW);
                expctddLdW = gather(expctddLdW);                
            end            
            testCase.verifyInstanceOf(actualdLdX,datatype);
            testCase.verifyInstanceOf(actualdLdW,datatype);            
            testCase.verifyThat(actualdLdX,...
                IsEqualTo(expctddLdX,'Within',tolObj));
            testCase.verifyThat(actualdLdW,...
                IsEqualTo(expctddLdW,'Within',tolObj));            
            
        end
                 
        function testPredictSynthesisShiftBottomCoefs(testCase, ...
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
            mode_ = 'Synthesis';
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
                'Mode',mode_,...
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

        function testPredictSynthesisShiftTopCoefs(testCase, ...
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
            mode_ = 'Synthesis';
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
            Yt = X(1:pt,:,:);
            Yb = X(pt+1:pt+pb,:,:);
            % Block circular shift
            Yt = circshift(Yt,shift);
            % Output
            expctdZ = cat(1,Yt,Yb);
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunCSAtomExtension1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'Name','Qn~',...
                'Direction',dir,...
                'Mode',mode_,...
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

        function testPredictSynthesisShiftBottomCoefsWithAnglePi4(testCase, ...
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
            mode_ = 'Synthesis';
            % nChsTotal x nSamples x nBlks
            X = randn(nChsTotal,nSamples,nblks,datatype);
            angle = pi/4;
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
            Yt = X(1:pt,:,:);
            Yb = X(pt+1:pt+pb,:,:);
            % C-S block butterfly            
            c = cos(angle);
            s = sin(angle);
            Zct = zeros(size(Yt),'like',Yt);
            Zst = zeros(size(Yt),'like',Yt);
            Zcb = zeros(size(Yb),'like',Yb);
            Zsb = zeros(size(Yb),'like',Yb);            
            for iSample=1:nSamples
                for iblk = 1:nblks
                    Zct(:,iSample,iblk) = c*Yt(:,iSample,iblk);
                    Zst(:,iSample,iblk) = s*Yt(:,iSample,iblk);
                    Zcb(:,iSample,iblk) = c*Yb(:,iSample,iblk);
                    Zsb(:,iSample,iblk) = s*Yb(:,iSample,iblk);                    
                end
            end
            Yt =  Zct+Zsb; % Inverted
            Yb = -Zst+Zcb; % Inverted
            % Block circular shift
            Yb = circshift(Yb,shift);
            expctdZ = cat(1,Yt,Yb);
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunCSAtomExtension1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'Name','Qn~',...
                'Direction',dir,...
                'Mode',mode_,...
                'TargetChannels',target_);
            layer.Angles = angle;
            
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

        function testPredictSynthesisShiftBottomCoefsWithRandomAngles(testCase, ...
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
            nAngles = nChsTotal/2;
            target_ = 'Bottom';
            mode_ = 'Synthesis';
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
            % 
            Yt = X(1:pt,:,:);
            Yb = X(pt+1:pt+pb,:,:);
            % C-S Block butterfly
            C = cos(angles);
            S = sin(angles);
            Zct = zeros(size(Yt),'like',Yt);
            Zst = zeros(size(Yt),'like',Yt);
            Zcb = zeros(size(Yb),'like',Yb);
            Zsb = zeros(size(Yb),'like',Yb);            
            for iSample=1:nSamples
                for iblk = 1:nblks
                    Zct(:,iSample,iblk) = C(:,iblk).*Yt(:,iSample,iblk);
                    Zst(:,iSample,iblk) = S(:,iblk).*Yt(:,iSample,iblk);
                    Zcb(:,iSample,iblk) = C(:,iblk).*Yb(:,iSample,iblk);
                    Zsb(:,iSample,iblk) = S(:,iblk).*Yb(:,iSample,iblk);                    
                end
            end
            Yt =  Zct+Zsb; % Inverted
            Yb = -Zst+Zcb; % Inverted
            % Block circular shift
            Yb = circshift(Yb,shift);
            expctdZ = cat(1,Yt,Yb);
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunCSAtomExtension1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...                
                'Name','Qn~',...
                'Direction',dir,...
                'Mode',mode_,...
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

        function testPredictSynthesisShiftTopCoefsWithRandomAngles(testCase, ...
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
            nAngles = nChsTotal/2;
            target_ = 'Top';
            mode_ = 'Synthesis';
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
            % 
            Yt = X(1:pt,:,:);
            Yb = X(pt+1:pt+pb,:,:);
            % C-S Block butterfly
            C = cos(angles);
            S = sin(angles);
            Zct = zeros(size(Yt),'like',Yt);
            Zst = zeros(size(Yt),'like',Yt);
            Zcb = zeros(size(Yb),'like',Yb);
            Zsb = zeros(size(Yb),'like',Yb);            
            for iSample=1:nSamples
                for iblk = 1:nblks
                    Zct(:,iSample,iblk) = C(:,iblk).*Yt(:,iSample,iblk);
                    Zst(:,iSample,iblk) = S(:,iblk).*Yt(:,iSample,iblk);
                    Zcb(:,iSample,iblk) = C(:,iblk).*Yb(:,iSample,iblk);
                    Zsb(:,iSample,iblk) = S(:,iblk).*Yb(:,iSample,iblk);                    
                end
            end
            Yt =  Zct+Zsb; % Inverted
            Yb = -Zst+Zcb; % Inverted
            % Block circular shift
            Yt = circshift(Yt,shift);
            expctdZ = cat(1,Yt,Yb);
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunCSAtomExtension1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...                
                'Name','Qn~',...
                'Direction',dir,...
                'Mode',mode_,...
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

        function testBackwardSynthesisShiftBottomCoefs(testCase, ...
                usegpu, stride, nblks, dir, datatype)
             if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-5));
            
            % Parameters
            nSamples = 8;
            nChsTotal = stride;
            nAngles = nChsTotal/2;            
            target_ = 'Bottom';
            mode_ = 'Synthesis';
            % nChsTotal x nSamples x nBlks
            X = randn(nChsTotal,nSamples,nblks,datatype);            
            dLdZ = randn(nChsTotal,nSamples,nblks,datatype);
            if usegpu
                X = gpuArray(X);
                dLdZ = gpuArray(dLdZ);
            end
            % Expected values
            if strcmp(dir,'Right')
                shift = [ 0 0 1 ];  
            elseif strcmp(dir,'Left')
                shift = [ 0 0 -1 ]; 
            else
                shift = [ 0 0 0 ]; 
            end
            % nChsTotal x nSamples x nBlks
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            % 
            Yt = dLdZ(1:pt,:,:);
            Yb = dLdZ(pt+1:pt+pb,:,:);
            % Block circular shift (Revserse)
            Yb = circshift(Yb,-shift); % Bottom, Revserse
            % C-S Block butterfly
            %Yt = Yt;
            %Yb = Yb;
            expctddLdX = cat(1,Yt,Yb);
            
            % dLdWi = <dLdZ,(dVdWi)X>
            expctddLdW = zeros(nAngles,nblks,datatype);
            c_top = X(1:pt,:,:);
            c_btm = X(pt+1:pt+pb,:,:);
            % C-S differential
            c_top_ = permute(c_top,[1 3 2]);
            c_btm_ = permute(c_btm,[1 3 2]);
            % Block circular shift (Revserse)
            Yt = dLdZ(1:pt,:,:);
            Yb = dLdZ(pt+1:pt+pb,:,:);
            Yb = circshift(Yb,-shift); % Bottom, Revserse            
            dldz_ = cat(1,Yt,Yb);
            for iAngle = 1:nAngles
                %dC_ = zeros(nAngles,nblks); % -sin(0) = 0
                dS_ = zeros(nAngles,nblks);  %  cos(0) = 1
                dS_(iAngle,:) = ones(1,nblks); 
                for iblk = 1:nblks
                    c_top_iblk_ =  dS_(:,iblk).*c_btm_(:,iblk,:); % [  C S ]            
                    c_btm_iblk_ = -dS_(:,iblk).*c_top_(:,iblk,:); % [ -S C ]
                    c_top_iblk = ipermute(c_top_iblk_,[1 3 2]); % iCh,iSample,iBlk
                    c_btm_iblk = ipermute(c_btm_iblk_,[1 3 2]);
                    c_iblk = cat(1,c_top_iblk,c_btm_iblk);                    
                    % Block circular shift
                    dldz_iblk = dldz_(:,:,iblk);
                    expctddLdW(iAngle,iblk) = sum(dldz_iblk.*c_iblk,'all');
                end
            end
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunCSAtomExtension1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'Name','Qn',...
                'Direction',dir,...
                'Mode',mode_,...
                'TargetChannels',target_);
            
            % Actual values
            [actualdLdX,actualdLdW] = layer.backward(X,[],dLdZ,[]);
            
            % Evaluation
            if usegpu
                testCase.verifyClass(actualdLdX,'gpuArray')
                testCase.verifyClass(actualdLdW,'gpuArray')
                actualdLdX = gather(actualdLdX);
                expctddLdX = gather(expctddLdX);
                actualdLdW = gather(actualdLdW);
                expctddLdW = gather(expctddLdW);                
            end            
            testCase.verifyInstanceOf(actualdLdX,datatype);
            testCase.verifyInstanceOf(actualdLdW,datatype);            
            testCase.verifyThat(actualdLdX,...
                IsEqualTo(expctddLdX,'Within',tolObj));
            testCase.verifyThat(actualdLdW,...
                IsEqualTo(expctddLdW,'Within',tolObj));            
            
        end

        %{
        function testBackwardSynthesisShiftTopCoefs(testCase, ...
                usegpu, stride, nblks, dir, datatype)
             if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-5));
            
            % Parameters
            nSamples = 8;
            nChsTotal = stride;
            nAngles = nChsTotal/2;            
            target_ = 'Top';
            mode_ = 'Synthesis';
            % nChsTotal x nSamples x nBlks
            X = randn(nChsTotal,nSamples,nblks,datatype);            
            dLdZ = randn(nChsTotal,nSamples,nblks,datatype);
            if usegpu
                X = gpuArray(X);
                dLdZ = gpuArray(dLdZ);
            end
            % Expected values
            if strcmp(dir,'Right')
                shift = [ 0 0 1 ];  
            elseif strcmp(dir,'Left')
                shift = [ 0 0 -1 ]; 
            else
                shift = [ 0 0 0 ]; 
            end
            % nChsTotal x nSamples x nBlks
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            %
            Yt = dLdZ(1:pt,:,:);
            Yb = dLdZ(pt+1:pt+pb,:,:);
            % Block circular shift (Revserse)
            Yt = circshift(Yt,-shift); % Top, Revserse
            % C-S Block butterfly
            %Yt = Yt;
            %Yb = Yb;
            expctddLdX = cat(1,Yt,Yb);

            % dLdWi = <dLdZ,(dVdWi)X>
            expctddLdW = zeros(nAngles,nblks,datatype);
            c_top = X(1:pt,:,:);
            c_btm = X(pt+1:pt+pb,:,:);
            % C-S differential
            for iAngle = 1:nAngles
                dS_ = zeros(nAngles,nblks);
                dS_(iAngle,:) = -ones(1,nblks);
                for iblk = 1:nblks
                    c_top_iblk = -dS_(:,iblk).*c_btm(:,:,iblk);
                    c_btm_iblk =  dS_(:,iblk).*c_top(:,:,iblk);
                    % Block circular shift
                    c_top_iblk = circshift(c_top_iblk,shift); % Top
                    c_iblk = cat(1,c_top_iblk,c_btm_iblk);
                    dldz_iblk = dLdZ(:,:,iblk);
                    expctddLdW(iAngle,iblk) = sum(dldz_iblk.*c_iblk,'all');
                end
            end

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunCSAtomExtension1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'Name','Qn',...
                'Direction',dir,...
                'Mode',mode_,...
                'TargetChannels',target_);
            
            % Actual values
            [actualdLdX,actualdLdW] = layer.backward(X,[],dLdZ,[]);
            
            % Evaluation
            if usegpu
                testCase.verifyClass(actualdLdX,'gpuArray')
                testCase.verifyClass(actualdLdW,'gpuArray')
                actualdLdX = gather(actualdLdX);
                expctddLdX = gather(expctddLdX);
                actualdLdW = gather(actualdLdW);
                expctddLdW = gather(expctddLdW);                
            end            
            testCase.verifyInstanceOf(actualdLdX,datatype);
            testCase.verifyInstanceOf(actualdLdW,datatype);            
            testCase.verifyThat(actualdLdX,...
                IsEqualTo(expctddLdX,'Within',tolObj));
            testCase.verifyThat(actualdLdW,...
                IsEqualTo(expctddLdW,'Within',tolObj));            
            
         end

        function testBackwardSynthesisShiftBottomCoefsWithAnglePi4(testCase, ...
                usegpu, stride, nblks, dir, datatype)
             if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-5));
            
            % Parameters
            nSamples = 8;
            nChsTotal = stride;
            nAngles = nChsTotal/2;            
            target_ = 'Bottom';
            mode_ = 'Synthesis';
            % nChsTotal x nSamples x nBlks
            angle = pi/4;
            X = randn(nChsTotal,nSamples,nblks,datatype);            
            dLdZ = randn(nChsTotal,nSamples,nblks,datatype);
            if usegpu
                X = gpuArray(X);
                dLdZ = gpuArray(dLdZ);
            end
            % Expected values
            if strcmp(dir,'Right')
                shift = [ 0 0 1 ];  
            elseif strcmp(dir,'Left')
                shift = [ 0 0 -1 ]; 
            else
                shift = [ 0 0 0 ]; 
            end
            % nChsTotal x nSamples x nBlks
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            %
            Zt = dLdZ(1:pt,:,:);
            Zb = dLdZ(pt+1:pt+pb,:,:);
            % Block circular shift (Revserse)
            Zb = circshift(Zb,-shift); % Bottom, Revserse
            % C-S Block butterfly (Transpose)
            c =  cos(angle);
            s = -sin(angle); 
            Yt =  c.*Zt + s.*Zb;
            Yb = -s.*Zt + c.*Zb;
            expctddLdX = cat(1,Yt,Yb);

            % dLdWi = <dLdZ,(dVdWi)X>
            expctddLdW = zeros(nAngles,nblks,datatype);
            c_top = permute(X(1:pt,:,:),[1 3 2]);
            c_btm = permute(X(pt+1:pt+pb,:,:),[1 3 2]);
            % C-S differential
            for iAngle = 1:nAngles
                dC_ = zeros(nAngles,nblks);
                dS_ = zeros(nAngles,nblks);
                dC_(iAngle,:) = -sin(angle)*ones(1,nblks);
                dS_(iAngle,:) = -cos(angle)*ones(1,nblks);
                for iblk = 1:nblks
                    c_top_iblk = dC_(:,iblk).*c_top(:,iblk,:) ...
                        - dS_(:,iblk).*c_btm(:,iblk,:);
                    c_btm_iblk = dS_(:,iblk).*c_top(:,iblk,:) ...
                        + dC_(:,iblk).*c_btm(:,iblk,:);
                    c_top_iblk = ipermute(c_top_iblk,[1 3 2]);
                    c_btm_iblk = ipermute(c_btm_iblk,[1 3 2]);
                    % Block circular shift
                    c_btm_iblk = circshift(c_btm_iblk,shift); % Bottom
                    c_iblk = cat(1,c_top_iblk,c_btm_iblk);
                    dldz_iblk = dLdZ(:,:,iblk);
                    expctddLdW(iAngle,iblk) = sum(dldz_iblk.*c_iblk,'all');
                end
            end

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunCSAtomExtension1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'Name','Qn',...
                'Direction',dir,...
                'Mode',mode_,...
                'TargetChannels',target_);
            layer.Angles = angle;
            
            % Actual values
            [actualdLdX,actualdLdW] = layer.backward(X,[],dLdZ,[]);
            
            % Evaluation
            if usegpu
                testCase.verifyClass(actualdLdX,'gpuArray')
                testCase.verifyClass(actualdLdW,'gpuArray')
                actualdLdX = gather(actualdLdX);
                expctddLdX = gather(expctddLdX);
                actualdLdW = gather(actualdLdW);
                expctddLdW = gather(expctddLdW);                
            end            
            testCase.verifyInstanceOf(actualdLdX,datatype);
            testCase.verifyInstanceOf(actualdLdW,datatype);            
            testCase.verifyThat(actualdLdX,...
                IsEqualTo(expctddLdX,'Within',tolObj));
            testCase.verifyThat(actualdLdW,...
                IsEqualTo(expctddLdW,'Within',tolObj));            
            
        end

        function testBackwardSynthesisShiftBottomCoefsWithRandomAngles(testCase, ...
                usegpu, stride, nblks, dir, datatype)
             if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-5));
            
            % Parameters
            nSamples = 8;
            nChsTotal = stride;
            nAngles = nChsTotal/2;            
            target_ = 'Bottom';
            mode_ = 'Synthesis';
            % nChsTotal x nSamples x nBlks
            angles = randn(nAngles,nblks);
            X = randn(nChsTotal,nSamples,nblks,datatype);            
            dLdZ = randn(nChsTotal,nSamples,nblks,datatype);
            if usegpu
                X = gpuArray(X);
                dLdZ = gpuArray(dLdZ);
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
            % nChsTotal x nSamples x nBlks
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            %
            Yt = dLdZ(1:pt,:,:);
            Yb = dLdZ(pt+1:pt+pb,:,:);
            % Block circular shift (Revserse)
            Yb = circshift(Yb,-shift); % Bottom, Revserse
            %
            Yt = permute(Yt,[1 3 2]);
            Yb = permute(Yb,[1 3 2]);
            % C-S Block butterfly (Transpose)
            C_ =  cos(angles);
            S_ = -sin(angles); % Inverted
            Zt = zeros(size(Yt),'like',Yt);
            Zb = zeros(size(Yb),'like',Yb);
            for iSample = 1:nSamples
                for iblk = 1:nblks
                    Zt(:,iblk,iSample) =  C_(:,iblk).*Yt(:,iblk,iSample) ...
                        + S_(:,iblk).*Yb(:,iblk,iSample);
                    Zb(:,iblk,iSample) = -S_(:,iblk).*Yt(:,iblk,iSample) ...
                        + C_(:,iblk).*Yb(:,iblk,iSample);
                end
            end
            Yt = ipermute(Zt,[1 3 2]);
            Yb = ipermute(Zb,[1 3 2]);
            expctddLdX = cat(1,Yt,Yb);
            
            % dLdWi = <dLdZ,(dVdWi)X>
            expctddLdW = zeros(nAngles,nblks,datatype);
            c_top = permute(X(1:pt,:,:),[1 3 2]);
            c_btm = permute(X(pt+1:pt+pb,:,:),[1 3 2]);
            % C-S differential
            for iAngle = 1:nAngles
                dC_ = zeros(nAngles,nblks);
                dS_ = zeros(nAngles,nblks);
                dC_(iAngle,:) = -sin(angles(iAngle,:));
                dS_(iAngle,:) = -cos(angles(iAngle,:)); % Inverted
                for iblk = 1:nblks
                    c_top_iblk = dC_(:,iblk).*c_top(:,iblk,:) ...
                        - dS_(:,iblk).*c_btm(:,iblk,:);
                    c_btm_iblk = dS_(:,iblk).*c_top(:,iblk,:) ...
                        + dC_(:,iblk).*c_btm(:,iblk,:);
                    c_top_iblk = ipermute(c_top_iblk,[1 3 2]);
                    c_btm_iblk = ipermute(c_btm_iblk,[1 3 2]);
                    % Block circular shift
                    c_btm_iblk = circshift(c_btm_iblk,shift); % Bottom
                    c_iblk = cat(1,c_top_iblk,c_btm_iblk);
                    dldz_iblk = dLdZ(:,:,iblk);
                    expctddLdW(iAngle,iblk) = sum(dldz_iblk.*c_iblk,'all');
                end
            end

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunCSAtomExtension1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'Name','Qn',...
                'Direction',dir,...
                'Mode',mode_,...
                'TargetChannels',target_);
            layer.Angles = angles;
            
            % Actual values
            [actualdLdX,actualdLdW] = layer.backward(X,[],dLdZ,[]);
            
            % Evaluation
            if usegpu
                testCase.verifyClass(actualdLdX,'gpuArray')
                testCase.verifyClass(actualdLdW,'gpuArray')
                actualdLdX = gather(actualdLdX);
                expctddLdX = gather(expctddLdX);
                actualdLdW = gather(actualdLdW);
                expctddLdW = gather(expctddLdW);                
            end            
            testCase.verifyInstanceOf(actualdLdX,datatype);
            testCase.verifyInstanceOf(actualdLdW,datatype);            
            testCase.verifyThat(actualdLdX,...
                IsEqualTo(expctddLdX,'Within',tolObj));
            testCase.verifyThat(actualdLdW,...
                IsEqualTo(expctddLdW,'Within',tolObj));            
            
        end

       function testBackwardSynthesisShiftTopCoefsWithRandomAngles(testCase, ...
                usegpu, stride, nblks, dir, datatype)
             if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-5));
            
            % Parameters
            nSamples = 8;
            nChsTotal = stride;
            nAngles = nChsTotal/2;            
            target_ = 'Top';
            mode_ = 'Synthesis';
            % nChsTotal x nSamples x nBlks
            angles = randn(nAngles,nblks);
            X = randn(nChsTotal,nSamples,nblks,datatype);            
            dLdZ = randn(nChsTotal,nSamples,nblks,datatype);
            if usegpu
                X = gpuArray(X);
                dLdZ = gpuArray(dLdZ);
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
            % nChsTotal x nSamples x nBlks
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            %
            Yt = dLdZ(1:pt,:,:);
            Yb = dLdZ(pt+1:pt+pb,:,:);
            % Block circular shift (Revserse)
            Yt = circshift(Yt,-shift); % Top, Revserse
            Yt = permute(Yt,[1 3 2]);
            Yb = permute(Yb,[1 3 2]);
            % C-S Block butterfly (Transpose)
            C_ =  cos(angles);
            S_ = -sin(angles); % Inverted
            Zt = zeros(size(Yt),'like',Yt);
            Zb = zeros(size(Yb),'like',Yb);
            for iSample = 1:nSamples
                for iblk = 1:nblks
                    Zt(:,iblk,iSample) =  C_(:,iblk).*Yt(:,iblk,iSample) ...
                        + S_(:,iblk).*Yb(:,iblk,iSample);
                    Zb(:,iblk,iSample) = -S_(:,iblk).*Yt(:,iblk,iSample) ...
                        + C_(:,iblk).*Yb(:,iblk,iSample);
                end
            end
            Yt = ipermute(Zt,[1 3 2]);
            Yb = ipermute(Zb,[1 3 2]);
            expctddLdX = cat(1,Yt,Yb);
            
            % dLdWi = <dLdZ,(dVdWi)X>
            expctddLdW = zeros(nAngles,nblks,datatype);
            c_top = X(1:pt,:,:);
            c_btm = X(pt+1:pt+pb,:,:);
            % C-S differential
            for iAngle = 1:nAngles
                dC_ = zeros(nAngles,nblks);
                dS_ = zeros(nAngles,nblks);
                dC_(iAngle,:) = -sin(angles(iAngle,:));
                dS_(iAngle,:) = -cos(angles(iAngle,:)); % Inverted
                for iblk = 1:nblks
                    c_top_iblk = dC_(:,iblk).*c_top(:,:,iblk) ...
                        - dS_(:,iblk).*c_btm(:,:,iblk);
                    c_btm_iblk = dS_(:,iblk).*c_top(:,:,iblk) ...
                        + dC_(:,iblk).*c_btm(:,:,iblk);
                    % Block circular shift
                    c_top_iblk = circshift(c_top_iblk,shift); % Top
                    c_iblk = cat(1,c_top_iblk,c_btm_iblk);
                    dldz_iblk = dLdZ(:,:,iblk);
                    expctddLdW(iAngle,iblk) = sum(dldz_iblk.*c_iblk,'all');
                end
            end

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunCSAtomExtension1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'Name','Qn',...
                'Direction',dir,...
                'Mode',mode_,...
                'TargetChannels',target_);
            layer.Angles = angles;
            
            % Actual values
            [actualdLdX,actualdLdW] = layer.backward(X,[],dLdZ,[]);
            
            % Evaluation
            if usegpu
                testCase.verifyClass(actualdLdX,'gpuArray')
                testCase.verifyClass(actualdLdW,'gpuArray')
                actualdLdX = gather(actualdLdX);
                expctddLdX = gather(expctddLdX);
                actualdLdW = gather(actualdLdW);
                expctddLdW = gather(expctddLdW);                
            end            
            testCase.verifyInstanceOf(actualdLdX,datatype);
            testCase.verifyInstanceOf(actualdLdW,datatype);            
            testCase.verifyThat(actualdLdX,...
                IsEqualTo(expctddLdX,'Within',tolObj));
            testCase.verifyThat(actualdLdW,...
                IsEqualTo(expctddLdW,'Within',tolObj));            
            
        end
        %}
    end
    
end