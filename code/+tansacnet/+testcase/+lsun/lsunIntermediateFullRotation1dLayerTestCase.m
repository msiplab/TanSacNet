classdef lsunIntermediateFullRotation1dLayerTestCase < matlab.unittest.TestCase
    %LSUNINTERMEDIATEFULLROTATION1DLAYERTESTCASE 
    %   
    %   コンポーネント別に入力(nComponents)
    %      nChs x 1 x nBlks x nSamples
    %
    %   コンポーネント別に出力(nComponents):
    %      nChs x 1 x nBlks x nSamples
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
        mus = { -1, 1 };
        nblks = struct('small', 2,'medium', 4, 'large', 8);
        usegpu = struct( 'true', true, 'false', false);        
    end

    methods (TestClassTeardown)

        function finalCheck(~)
            import tansacnet.lsun.*
            layer = lsunIntermediateFullRotation1dLayer(...
                'Stride',2,...
                'NumberOfBlocks',4);
            fprintf("\n --- Check layer for 1-D images ---\n");
            checkLayer(layer,[2 1 4 8],...
                'ObservationDimension',4,...
                'CheckCodegenCompatibility',true)      
        end

    end
    
    methods (Test)
        
        function testConstructor(testCase, stride)
            
            % Expected values
            expctdName = 'Vn~';
            expctdMode = 'Synthesis';
            expctdDescription = "Synthesis LSUN intermediate full rotation " ...
                + "(pt,pb) = (" ...
                + ceil(prod(stride)/2) + "," + floor(prod(stride)/2) + ")";
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunIntermediateFullRotation1dLayer(...
                'Stride',stride,...
                'Name',expctdName);
            
            % Actual values
            actualName = layer.Name;
            actualMode = layer.Mode;
            actualDescription = layer.Description;
            
            % Evaluation
            testCase.verifyEqual(actualName,expctdName);
            testCase.verifyEqual(actualMode,expctdMode);
            testCase.verifyEqual(actualDescription,expctdDescription);
        end

        function testConstructorWithDeviceAndDType(testCase, stride, usegpu, datatype)
            
            % Expected values
            expctdName = 'Vn~';
            %expctdMode = 'Synthesis';
     
            device_ = ["cpu", "cuda"];
            expctdDevice = device_(usegpu+1);
            expctdDType = datatype;
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunIntermediateFullRotation1dLayer(...
                'Stride',stride,...
                'Name',expctdName,...
                'Device',expctdDevice,...
                'DType',expctdDType);
            
            % Actual values
            actualName = layer.Name;
            actualDevice = layer.Device;
            actualDType = layer.DType;

            % Evaluation
            testCase.verifyEqual(actualName,expctdName);
            testCase.verifyEqual(actualDevice,expctdDevice);
            testCase.verifyEqual(actualDType,expctdDType);  
            
        end

        function testPredict(testCase, ...
                usegpu, stride, nblks, mus, datatype)

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
            % nChsTotal x 1 x nBlks x nSamples
            X = randn(nChsTotal,1,nblks,nSamples,datatype);
            if usegpu
                X = gpuArray(X);
            end

            % Expected values
            % nChsTotal x 1 x nBlks x nSamples
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            WnT = repmat(mus*eye(pt,datatype),[1 1 nblks]);
            UnT = repmat(mus*eye(pb,datatype),[1 1 nblks]);
            Y = X;
            Yt = reshape(Y(1:pt,:,:,:),pt,nblks,nSamples);
            Yb = reshape(Y(pt+1:pt+pb,:,:,:),pb,nblks,nSamples);
            Zt = zeros(size(Yt),'like',Yt);
            Zb = zeros(size(Yb),'like',Yb);
            for iSample=1:nSamples
                for iblk = 1:nblks
                    Zt(:,iblk,iSample) = WnT(:,:,iblk)*Yt(:,iblk,iSample);
                    Zb(:,iblk,iSample) = UnT(:,:,iblk)*Yb(:,iblk,iSample);
                end
            end
            Y(1:pt,:,:,:) = reshape(Zt,pt,1,nblks,nSamples);
            Y(pt+1:pt+pb,:,:,:) = reshape(Zb,pb,1,nblks,nSamples);
            expctdZ = Y;
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunIntermediateFullRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'Name','Vn~');
            
            % Actual values
            layer.Mus = mus;
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

        function testPredictWithDeviceAndDType(testCase, ...
                usegpu, stride, nblks, mus, datatype)

            if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));

            device_ = ["cpu", "cuda"];      
            expctdDevice = device_(usegpu+1);
            expctdDType = datatype;      
            
            % Parameters
            nSamples = 8;
            nChsTotal = stride;
            % nDecs x nRows x nCols x nSamples
            %X = randn(nrows,ncols,nDecs,nSamples,datatype);
            X = randn(nChsTotal,1,nblks,nSamples,datatype);
            if expctdDevice == "cuda"
                X = gpuArray(X);
            end
            
            % Expected values
            % nChsTotal x 1 x nBlks x nSamples
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            WnT = repmat(mus*eye(pt,datatype),[1 1 nblks]);
            UnT = repmat(mus*eye(pb,datatype),[1 1 nblks]);
            Y = X;
            Yt = reshape(Y(1:pt,:,:,:),pt,nblks,nSamples);
            Yb = reshape(Y(pt+1:pt+pb,:,:,:),pb,nblks,nSamples);
            Zt = zeros(size(Yt),'like',Yt);
            Zb = zeros(size(Yb),'like',Yb);
            for iSample=1:nSamples
                for iblk = 1:nblks
                    Zt(:,iblk,iSample) = WnT(:,:,iblk)*Yt(:,iblk,iSample);
                    Zb(:,iblk,iSample) = UnT(:,:,iblk)*Yb(:,iblk,iSample);
                end
            end
            Y(1:pt,:,:,:) = reshape(Zt,pt,1,nblks,nSamples);
            Y(pt+1:pt+pb,:,:,:) = reshape(Zb,pb,1,nblks,nSamples);
            expctdZ = Y;

            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunIntermediateFullRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'Name','Vn~',...
                'Device',expctdDevice,...
                'DType',expctdDType);
            
            % Actual values
            layer.Mus = mus;
            actualZ = layer.predict(X);
            actualDevice = layer.Device;
            
            % Evaluation
            testCase.verifyEqual(actualDevice,expctdDevice);
            if actualDevice == "cuda"
                testCase.verifyClass(actualZ,'gpuArray')
                actualZ = gather(actualZ);
                expctdZ = gather(expctdZ);
            end
            testCase.verifyInstanceOf(actualZ,expctdDType);
            testCase.verifyThat(actualZ,...
                IsEqualTo(expctdZ,'Within',tolObj));
            
        end

        function testPredictWithRandomAngles(testCase, ...
                usegpu, stride, nblks, mus, datatype)

            if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            import tansacnet.utility.*
            genW = OrthonormalMatrixGenerationSystem();
            genU = OrthonormalMatrixGenerationSystem();
            
            % Parameters
            nSamples = 8;
            nChsTotal = stride;
            % nChsTotal x 1 x nBlks x nSample
            X = randn(nChsTotal,1,nblks,nSamples,datatype);
            nAngles = (nChsTotal-2)*nChsTotal/4;
            angles = randn(nAngles,nblks);
            if usegpu
                X = gpuArray(X);
                angles = gpuArray(angles);
            end

            % Expected values
            % nChsTotal x 1 x nBlks x nSamples
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            anglesW = angles(1:nAngles/2,:);
            anglesU = angles(nAngles/2+1:nAngles,:);
            if nAngles == 0
                WnT = mus*ones(1,1,nblks);
                UnT = mus*ones(1,1,nblks);
            else
                WnT = permute(genW.step(anglesW,mus),[2 1 3]);
                UnT = permute(genU.step(anglesU,mus),[2 1 3]);
            end
            Y = X;
            Yt = reshape(Y(1:pt,:,:,:),pt,nblks,nSamples);
            Yb = reshape(Y(pt+1:pt+pb,:,:,:),pb,nblks,nSamples);
            Zt = zeros(size(Yt),'like',Yt);
            Zb = zeros(size(Yb),'like',Yb);
            for iSample=1:nSamples
                for iblk = 1:nblks
                    Zt(:,iblk,iSample) = WnT(:,:,iblk)*Yt(:,iblk,iSample);
                    Zb(:,iblk,iSample) = UnT(:,:,iblk)*Yb(:,iblk,iSample);
                end
            end
            Y(1:pt,:,:,:) = reshape(Zt,pt,1,nblks,nSamples);
            Y(pt+1:pt+pb,:,:,:) = reshape(Zb,pb,1,nblks,nSamples);
            expctdZ = Y;
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunIntermediateFullRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'Name','Vn~');
            
            % Actual values
            layer.Mus = mus;
            layer.Angles = angles;
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
        
        function testPredictAnalysisMode(testCase, ...
                usegpu, stride, nblks, mus, datatype)

            if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            import tansacnet.utility.*
            genW = OrthonormalMatrixGenerationSystem();
            genU = OrthonormalMatrixGenerationSystem();

            % Parameters
            nSamples = 8;
            nChsTotal = prod(stride);
            % nChsTotal x 1 x nBlks x nSamples
            X = randn(nChsTotal,1,nblks,nSamples,datatype);
            nAngles = (nChsTotal-2)*nChsTotal/4;
            angles = randn(nAngles,nblks);
            if usegpu
                X = gpuArray(X);
                angles = gpuArray(angles);
            end

            % Expected values
            % nChsTotal x 1 x nBlks x nSamples
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            anglesW = angles(1:nAngles/2,:);
            anglesU = angles(nAngles/2+1:nAngles,:);
            if nAngles == 0
                Wn = mus*ones(1,1,nblks);
                Un = mus*ones(1,1,nblks);
            else
                Wn = genW.step(anglesW,mus);
                Un = genU.step(anglesU,mus);
            end
            Y = X;
            Yt = reshape(Y(1:pt,:,:,:),pt,nblks,nSamples);
            Yb = reshape(Y(pt+1:pt+pb,:,:,:),pb,nblks,nSamples);
            Zt = zeros(size(Yt),'like',Yt);
            Zb = zeros(size(Yb),'like',Yb);
            for iSample=1:nSamples
                for iblk = 1:nblks
                    Zt(:,iblk,iSample) = Wn(:,:,iblk)*Yt(:,iblk,iSample);
                    Zb(:,iblk,iSample) = Un(:,:,iblk)*Yb(:,iblk,iSample);
                end
            end            
            Y(1:pt,:,:,:) = reshape(Zt,pt,1,nblks,nSamples);
            Y(pt+1:pt+pb,:,:,:) = reshape(Zb,pb,1,nblks,nSamples);
            expctdZ = Y;
            expctdDescription = "Analysis LSUN intermediate full rotation " ...
                + "(pt,pb) = (" ...
                + pt + "," + pb + ")";
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunIntermediateFullRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'Name','Vn',...
                'Mode','Analysis');
            
            % Actual values
            layer.Mus = mus;
            layer.Angles = angles;
            actualZ = layer.predict(X);
            actualDescription = layer.Description;
            
            % Evaluation
            if usegpu
                testCase.verifyClass(actualZ,'gpuArray')
                actualZ = gather(actualZ);
                expctdZ = gather(expctdZ);
            end
            testCase.verifyInstanceOf(actualZ,datatype);
            testCase.verifyThat(actualZ,...
                IsEqualTo(expctdZ,'Within',tolObj));
            testCase.verifyEqual(actualDescription,expctdDescription);

        end

        function testBackward(testCase, ...
                usegpu, stride, nblks, mus, datatype)
            
            if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-4,single(1e-4));
            import tansacnet.utility.*
            genW = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on');
            genU = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on');            
            
            % Parameters
            nSamples = 8;
            nChsTotal = stride;
            nAngles = (nChsTotal-2)*nChsTotal/4;
            angles = zeros(nAngles,nblks);
            
            % nChsTotal x 1 x nBlks x nSamples         
            X = randn(nChsTotal,1,nblks,nSamples,datatype);
            dLdZ = randn(nChsTotal,1,nblks,nSamples,datatype);
            if usegpu
                X = gpuArray(X);
                dLdZ = gpuArray(dLdZ);
                angles = gpuArray(angles);
            end

            % Expected values
            % nChsTotal x nRows x nCols x nSamples
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);

            % dLdX = dZdX x dLdZ
            anglesW = angles(1:nAngles/2,:);
            anglesU = angles(nAngles/2+1:nAngles,:);
            if nAngles == 0
                Wn = mus*ones(1,1,nblks);
                Un = mus*ones(1,1,nblks);
            else
                Wn = genW.step(anglesW,mus,0);
                Un = genU.step(anglesU,mus,0);
            end
            %
            adLd_ = dLdZ;
            cdLd_top = reshape(adLd_(1:pt,:,:,:),pt,nblks,nSamples);
            cdLd_btm = reshape(adLd_(pt+1:pt+pb,:,:,:),pb,nblks,nSamples);
            for iSample = 1:nSamples
                for iblk = 1:nblks
                    cdLd_top(:,iblk,iSample) = Wn(:,:,iblk)*cdLd_top(:,iblk,iSample);
                    cdLd_btm(:,iblk,iSample) = Un(:,:,iblk)*cdLd_btm(:,iblk,iSample);
                end
            end
            adLd_(1:pt,:,:,:) = reshape(cdLd_top,pt,1,nblks,nSamples);
            adLd_(pt+1:pt+pb,:,:,:) = reshape(cdLd_btm,pb,1,nblks,nSamples);
            expctddLdX = adLd_;
            
            % dLdWi = <dLdZ,(dVdWi)X>
            expctddLdW = zeros(nAngles,nblks,datatype);
            c_top = X(1:pt,:,:,:);
            c_btm = X(pt+1:pt+pb,:,:,:);
            dldz_top = dLdZ(1:pt,:,:,:);
            dldz_btm = dLdZ(pt+1:pt+pb,:,:,:);
            for iAngle = 1:nAngles/2
                dWn_T = permute(genW.step(anglesW,mus,iAngle),[2 1 3]);
                dUn_T = permute(genU.step(anglesU,mus,iAngle),[2 1 3]);                
                for iblk = 1:nblks
                    c_top_iblk = squeeze(c_top(:,:,iblk,:));                    
                    c_btm_iblk = squeeze(c_btm(:,:,iblk,:));
                    c_top_iblk = dWn_T(:,:,iblk)*c_top_iblk;                    
                    c_btm_iblk = dUn_T(:,:,iblk)*c_btm_iblk;
                    dldz_top_iblk = squeeze(dldz_top(:,:,iblk,:));
                    dldz_btm_iblk = squeeze(dldz_btm(:,:,iblk,:));
                    expctddLdW(iAngle,iblk) = sum(dldz_top_iblk.*c_top_iblk,'all');
                    expctddLdW(nAngles/2+iAngle,iblk) = sum(dldz_btm_iblk.*c_btm_iblk,'all');
                end
            end
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunIntermediateFullRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'Name','Vn~');
            layer.Mus = mus;
            
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

        function testBackwardWithRandomAngles(testCase, ...
                usegpu, stride, nblks, mus, datatype)
            if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-4,single(1e-4));
            import tansacnet.utility.*
            genW = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on');
            genU = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on');
            
            % Parameters
            nSamples = 8;
            nChsTotal = stride;
            nAngles = (nChsTotal-2)*nChsTotal/4;
            angles = randn(nAngles,nblks);
                 
            % nChsTotal x 1 x nBlks x nSamples
            X = randn(nChsTotal,1,nblks,nSamples,datatype);            
            dLdZ = randn(nChsTotal,1,nblks,nSamples,datatype);            
            if usegpu
                X = gpuArray(X);
                dLdZ = gpuArray(dLdZ);
                angles = gpuArray(angles);
            end

            % Expected values
            % nChsTotal x 1 x nBlks x nSamples
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            
            % dLdX = dZdX x dLdZ
            anglesW = angles(1:nAngles/2,:);
            anglesU = angles(nAngles/2+1:nAngles,:);
            if nAngles == 0
                Wn = mus*ones(1,1,nblks);
                Un = mus*ones(1,1,nblks);
            else
                Wn = genW.step(anglesW,mus,0);
                Un = genU.step(anglesU,mus,0);
            end
            %
            adLd_ = dLdZ;
            cdLd_top = adLd_(1:pt,:,:,:);
            cdLd_btm = adLd_(pt+1:pt+pb,:,:,:);
            for iSample = 1:nSamples
                for iblk = 1:nblks
                    cdLd_top(:,:,iblk,iSample) = Wn(:,:,iblk)*cdLd_top(:,:,iblk,iSample);
                    cdLd_btm(:,:,iblk,iSample) = Un(:,:,iblk)*cdLd_btm(:,:,iblk,iSample);                    
                end
            end
            adLd_(1:pt,:,:,:) = cdLd_top;
            adLd_(pt+1:pt+pb,:,:,:) = cdLd_btm;
            expctddLdX = adLd_;
            
            % dLdWi = <dLdZ,(dVdWi)X>
            expctddLdW = zeros(nAngles,nblks,datatype);
            c_top = X(1:pt,:,:,:);
            c_btm = X(pt+1:pt+pb,:,:,:);
            dldz_top = dLdZ(1:pt,:,:,:);
            dldz_btm = dLdZ(pt+1:pt+pb,:,:,:);
            for iAngle = 1:nAngles/2
                dWn_T = permute(genW.step(anglesW,mus,iAngle),[2 1 3]);
                dUn_T = permute(genU.step(anglesU,mus,iAngle),[2 1 3]);                
                for iblk = 1:nblks
                    c_top_iblk = squeeze(c_top(:,:,iblk,:));
                    c_btm_iblk = squeeze(c_btm(:,:,iblk,:));                    
                    c_top_iblk = dWn_T(:,:,iblk)*c_top_iblk;
                    c_btm_iblk = dUn_T(:,:,iblk)*c_btm_iblk;
                    dldz_top_iblk = squeeze(dldz_top(:,:,iblk,:));                    
                    dldz_btm_iblk = squeeze(dldz_btm(:,:,iblk,:));
                    expctddLdW(iAngle,iblk) = sum(dldz_top_iblk.*c_top_iblk,'all');
                    expctddLdW(nAngles/2+iAngle,iblk) = sum(dldz_btm_iblk.*c_btm_iblk,'all');                    
                end
            end
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunIntermediateFullRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'Name','Vn~');
            layer.Mus = mus;
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

        function testBackwardAnalysisMode(testCase, ...
                usegpu, stride, nblks, mus, datatype)
            if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-4,single(1e-4));
            import tansacnet.utility.*
            genW = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on');
            genU = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on');            
            
            % Parameters
            nSamples = 8;
            nChsTotal = stride;
            nAngles = (nChsTotal-2)*nChsTotal/4;
            angles = randn(nAngles,nblks);
            
            % nChsTotal x 1 x nBlks x nSamples
            X = randn(nChsTotal,1,nblks,nSamples,datatype);
            dLdZ = randn(nChsTotal,1,nblks,nSamples,datatype);
            if usegpu
                X = gpuArray(X);
                dLdZ = gpuArray(dLdZ);
                angles = gpuArray(angles);
            end

            % Expected values
            % nChsTotal x 1 x nBlks x nSamples
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            
            % dLdX = dZdX x dLdZ
            anglesW = angles(1:nAngles/2,:);
            anglesU = angles(nAngles/2+1:nAngles,:);
            if nAngles == 0
                WnT = mus*ones(1,1,nblks);
                UnT = mus*ones(1,1,nblks);
            else
                WnT = permute(genW.step(anglesW,mus,0),[2 1 3]);
                UnT = permute(genU.step(anglesU,mus,0),[2 1 3]);
            end
            %
            adLd_ = dLdZ;
            cdLd_top = adLd_(1:pt,:,:,:);
            cdLd_btm = adLd_(pt+1:pt+pb,:,:,:);
            for iSample = 1:nSamples
                for iblk = 1:nblks
                    cdLd_top(:,:,iblk,iSample) = WnT(:,:,iblk)*cdLd_top(:,:,iblk,iSample);
                    cdLd_btm(:,:,iblk,iSample) = UnT(:,:,iblk)*cdLd_btm(:,:,iblk,iSample);                    
                end
            end
            adLd_(1:pt,:,:,:) = cdLd_top;
            adLd_(pt+1:pt+pb,:,:,:) = cdLd_btm;
            expctddLdX = adLd_;

            % dLdWi = <dLdZ,(dVdWi)X>
            expctddLdW = zeros(nAngles,nblks,datatype);
            c_top = X(1:pt,:,:,:);
            c_btm = X(pt+1:pt+pb,:,:,:);
            dldz_top = dLdZ(1:pt,:,:,:);
            dldz_btm = dLdZ(pt+1:pt+pb,:,:,:);
            for iAngle = 1:nAngles/2
                dWn = genW.step(anglesW,mus,iAngle);
                dUn = genU.step(anglesU,mus,iAngle);
                for iblk = 1:nblks
                    c_top_iblk = squeeze(c_top(:,:,iblk,:));
                    c_btm_iblk = squeeze(c_btm(:,:,iblk,:));                    
                    c_top_iblk = dWn(:,:,iblk)*c_top_iblk;
                    c_btm_iblk = dUn(:,:,iblk)*c_btm_iblk;                    
                    dldz_top_iblk = squeeze(dldz_top(:,:,iblk,:));
                    dldz_btm_iblk = squeeze(dldz_btm(:,:,iblk,:));
                    expctddLdW(iAngle,iblk) = sum(dldz_top_iblk.*c_top_iblk,'all');
                    expctddLdW(nAngles/2+iAngle,iblk) = sum(dldz_btm_iblk.*c_btm_iblk,'all');                    
                end
            end
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunIntermediateFullRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'Name','Vn',...
                'Mode','Analysis');
            layer.Mus = mus;
            layer.Angles = angles;
            %expctdZ = layer.predict(X);
            
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

    end
    
end