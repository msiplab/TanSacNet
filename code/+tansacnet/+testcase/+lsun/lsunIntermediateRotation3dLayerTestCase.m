classdef lsunIntermediateRotation3dLayerTestCase < matlab.unittest.TestCase
    %LSUNINTERMEDIATEROTATION3DLAYERTESTCASE 
    %   
    %   コンポーネント別に入力(nComponents):
    %       nChsTotal x nRows x nCols x nLays xnSamples
    %
    %   コンポーネント別に出力(nComponents):
    %       nChsTotal x nRows x nCols x nLays x nSamples
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
    % http://msiplab.eng.niigata-u.ac.jp   
   
    properties (TestParameter)
        stride = { [2 2 2], [1 2 4] };
        datatype = { 'single', 'double' };
        mus = { -1, 1 };
        nrows = struct('small', 2,'medium', 4, 'large', 8);
        ncols = struct('small', 2,'medium', 4, 'large', 8);
        nlays = struct('small', 2,'medium', 4, 'large', 8);
        usegpu = struct( 'true', true, 'false', false);
    end

    methods (TestClassTeardown)
        function finalCheck(~)
            import tansacnet.lsun.*
            layer = lsunIntermediateRotation3dLayer(...
                'Stride',[2 2 2],...
                'NumberOfBlocks',[8 8 8]);
            fprintf("\n --- Check layer for 3-D images ---\n");
            checkLayer(layer,[8 8 8 8],...
                'ObservationDimension',5,...
                'CheckCodegenCompatibility',false)      
        end
    end
    
    methods (Test)
        
        function testConstructor(testCase, stride)
            
            % Expected values
            expctdName = 'Vn~';
            expctdMode = 'Synthesis';
            expctdDescription = "Synthesis LSUN intermediate rotation " ...
                + "(ps,pa) = (" ...
                + ceil(prod(stride)/2) + "," + floor(prod(stride)/2) + ")";
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunIntermediateRotation3dLayer(...
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
        
        function testPredictGrayscale(testCase, ...
                usegpu, stride, nrows, ncols, nlays, mus, datatype)

            if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end

            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));        
            
            % Parameters
            nSamples = 8;
            nChsTotal = prod(stride);
            % nChsTotal x nRows x nCols xnSamples
            %X = randn(nrows,ncols,nChsTotal,nSamples,datatype);
            X = randn(nChsTotal,nrows,ncols,nlays,nSamples,datatype);
            if usegpu
                X = gpuArray(X);
            end
            % Expected values
            % nChsTotal x nRows x nCols x nLays x nSamples
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            UnT = repmat(mus*eye(pa,datatype),[1 1 nrows*ncols*nlays]);
            Y = X; %permute(X,[4 1 2 3 5]);
            Ya = reshape(Y(ps+1:ps+pa,:,:,:,:),pa,nrows*ncols*nlays,nSamples);
            Za = zeros(size(Ya),'like',Ya);
            for iSample=1:nSamples
                for iblk = 1:(nrows*ncols*nlays)
                    Za(:,iblk,iSample) = UnT(:,:,iblk)*Ya(:,iblk,iSample);
                end
            end
            Y(ps+1:ps+pa,:,:,:,:) = reshape(Za,pa,nrows,ncols,nlays,nSamples);
            expctdZ = Y; % ipermute(Y,[4 1 2 3 5]);
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunIntermediateRotation3dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols nlays],...
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
        
        function testPredictGrayscaleWithRandomAngles(testCase, ...
                usegpu, stride, nrows, ncols, nlays, mus, datatype)

            if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            import tansacnet.utility.*
            genU = OrthonormalMatrixGenerationSystem();
            
            % Parameters
            nSamples = 8;
            nChsTotal = prod(stride);
            % nChs x nRows x nCols x nLays x nSamples
            %X = randn(nrows,ncols,nlays,nChsTotal,nSamples,datatype);
            X = randn(nChsTotal,nrows,ncols,nlays,nSamples,datatype);
            angles = randn((nChsTotal-2)*nChsTotal/8,nrows*ncols*nlays);
            if usegpu
                X = gpuArray(X);
                angles = gpuArray(angles);
            end

            % Expected values
            % nChsTotal x nRows x nCols x nLays x nSamples
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            UnT = permute(genU.step(angles,mus),[2 1 3]);
            Y = X; %permute(X,[4 1 2 3 5]);
            Ya = reshape(Y(ps+1:ps+pa,:,:,:,:),pa,nrows*ncols*nlays,nSamples);
            Za = zeros(size(Ya),'like',Ya);
            for iSample=1:nSamples
                for iblk = 1:(nrows*ncols*nlays)
                    Za(:,iblk,iSample) = UnT(:,:,iblk)*Ya(:,iblk,iSample);
                end
            end
            Y(ps+1:ps+pa,:,:,:,:) = reshape(Za,pa,nrows,ncols,nlays,nSamples);
            expctdZ = Y; %ipermute(Y,[3 1 2 4]);
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunIntermediateRotation3dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols nlays],...
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
        
        function testPredictGrayscaleAnalysisMode(testCase, ...
                usegpu, stride, nrows, ncols, nlays, mus, datatype)

            if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            import tansacnet.utility.*
            genU = OrthonormalMatrixGenerationSystem();

            % Parameters
            nSamples = 8;
            nChsTotal = prod(stride);
            % nChs x nRows x nCols x nLays x nSamples
            %X = randn(nrows,ncols,nlays,nChsTotal,nSamples,datatype);
            X = randn(nChsTotal,nrows,ncols,nlays,nSamples,datatype);
            angles = randn((nChsTotal-2)*nChsTotal/8,nrows*ncols*nlays);
            if usegpu
                X = gpuArray(X);
                angles = gpuArray(angles);
            end

            % Expected values
            % nChsTotal x nRows x nCols x nLays x nSamples
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            Un = genU.step(angles,mus);
            Y = X; % permute(X,[4 1 2 3 5]);
            Ya = reshape(Y(ps+1:ps+pa,:,:,:,:),pa,nrows*ncols*nlays,nSamples);
            Za = zeros(size(Ya),'like',Ya);
            for iSample=1:nSamples
                for iblk = 1:(nrows*ncols*nlays)
                    Za(:,iblk,iSample) = Un(:,:,iblk)*Ya(:,iblk,iSample);
                end
            end            
            Y(ps+1:ps+pa,:,:,:,:) = reshape(Za,pa,nrows,ncols,nlays,nSamples);
            expctdZ = Y; % ipermute(Y,[4 1 2 3 5]);
            expctdDescription = "Analysis LSUN intermediate rotation " ...
                + "(ps,pa) = (" ...
                + ps + "," + pa + ")";
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunIntermediateRotation3dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols nlays],...
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

        function testBackwardGrayscale(testCase, ...
                usegpu, stride, nrows, ncols, nlays, mus, datatype)
            
            if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-4,single(1e-4));
            import tansacnet.utility.*
            genU = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on');
            
            % Parameters
            nSamples = 8;
            nChsTotal = prod(stride);
            nAngles = (nChsTotal-2)*nChsTotal/8;
            angles = zeros(nAngles,nrows*ncols*nlays,datatype);
            
            % nChsTotal x nRows x nCols x nLays x nSamples
            %X = randn(nrows,ncols,nlays,nChsTotal,nSamples,datatype);            
            %dLdZ = randn(nrows,ncols,nlays,nChsTotal,nSamples,datatype);          
            X = randn(nChsTotal,nrows,ncols,nlays,nSamples,datatype);
            dLdZ = randn(nChsTotal,nrows,ncols,nlays,nSamples,datatype);
            if usegpu
                X = gpuArray(X);
                dLdZ = gpuArray(dLdZ);
                angles = gpuArray(angles);
            end

            % Expected values
            % nChsTotal x nRows x nCols x nSamples
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            
            % dLdX = dZdX x dLdZ
            Un = genU.step(angles,mus,0);
            adLd_ = dLdZ; %permute(dLdZ,[4 1 2 3 5]);
            cdLd_low = reshape(adLd_(ps+1:ps+pa,:,:,:,:),pa,nrows*ncols*nlays,nSamples);
            for iSample = 1:nSamples
                for iblk = 1:(nrows*ncols*nlays)
                    cdLd_low(:,iblk,iSample) = Un(:,:,iblk)*cdLd_low(:,iblk,iSample);
                end
            end
            adLd_(ps+1:ps+pa,:,:,:,:) = reshape(cdLd_low,pa,nrows,ncols,nlays,nSamples);
            expctddLdX = adLd_; %ipermute(adLd_,[4 1 2 3 5]);
            
            % dLdWi = <dLdZ,(dVdWi)X>
            expctddLdW = zeros(nAngles,nrows*ncols*nlays,datatype);
            c_low = reshape(X(ps+1:ps+pa,:,:,:,:),pa,nrows*ncols*nlays,nSamples);
            dldz_low = reshape(dLdZ(ps+1:ps+pa,:,:,:,:),pa,nrows*ncols*nlays,nSamples);
            for iAngle = 1:nAngles
                dUn_T = permute(genU.step(angles,mus,iAngle),[2 1 3]);
                for iblk = 1:(nrows*ncols*nlays)
                    c_low_iblk = squeeze(c_low(:,iblk,:));
                    c_low_iblk = dUn_T(:,:,iblk)*c_low_iblk;
                    dldz_iblk = squeeze(dldz_low(:,iblk,:));
                    expctddLdW(iAngle,iblk) = sum(dldz_iblk.*c_low_iblk,'all');
                end
            end
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunIntermediateRotation3dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols nlays],...
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

        function testBackwardGrayscaleWithRandomAngles(testCase, ...
                usegpu, stride, nrows, ncols, nlays, mus, datatype)
    
            if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-4,single(1e-4));
            import tansacnet.utility.*
            genU = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on');
            
            % Parameters
            nSamples = 8;
            nChsTotal = prod(stride);
            nAngles = (nChsTotal-2)*nChsTotal/8;
            angles = randn((nChsTotal-2)*nChsTotal/8,nrows*ncols*nlays);
                 
            % nChsTotal x nRows x nCols x nLays x nSamples
            %X = randn(nrows,ncols,nlays,nChsTotal,nSamples,datatype);            
            %dLdZ = randn(nrows,ncols,nlays,nChsTotal,nSamples,datatype);
            X = randn(nChsTotal,nrows,ncols,nlays,nSamples,datatype);            
            dLdZ = randn(nChsTotal,nrows,ncols,nlays,nSamples,datatype);            
            if usegpu
                X = gpuArray(X);
                dLdZ = gpuArray(dLdZ);
                angles = gpuArray(angles);
            end

            % Expected values
            % nChsTotal x nRows x nCols x nLays x nSamples
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            
            % dLdX = dZdX x dLdZ
            Un = genU.step(angles,mus,0);
            adLd_ = dLdZ; %permute(dLdZ,[4 1 2 3 5]);
            cdLd_low = reshape(adLd_(ps+1:ps+pa,:,:,:,:),pa,nrows*ncols*nlays,nSamples);
            for iSample = 1:nSamples
                for iblk = 1:(nrows*ncols*nlays)
                    cdLd_low(:,iblk,iSample) = Un(:,:,iblk)*cdLd_low(:,iblk,iSample);
                end
            end
            adLd_(ps+1:ps+pa,:,:,:,:) = reshape(cdLd_low,pa,nrows,ncols,nlays,nSamples);
            expctddLdX = adLd_; %ipermute(adLd_,[4 1 2 3 5]);
            
            % dLdWi = <dLdZ,(dVdWi)X>
            expctddLdW = zeros(nAngles,nrows*ncols*nlays,datatype);
            c_low = reshape(X(ps+1:ps+pa,:,:,:,:),pa,nrows*ncols*nlays,nSamples);
            dldz_low = reshape(dLdZ(ps+1:ps+pa,:,:,:,:),pa,nrows*ncols*nlays,nSamples);
            for iAngle = 1:nAngles
                dUn_T = permute(genU.step(angles,mus,iAngle),[2 1 3]);
                for iblk = 1:(nrows*ncols*nlays)
                    c_low_iblk = squeeze(c_low(:,iblk,:));
                    c_low_iblk = dUn_T(:,:,iblk)*c_low_iblk;
                    dldz_iblk = squeeze(dldz_low(:,iblk,:));
                    expctddLdW(iAngle,iblk) = sum(dldz_iblk.*c_low_iblk,'all');
                end
            end
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunIntermediateRotation3dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols nlays],...
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
        
        function testBackwardGrayscaleAnalysisMode(testCase, ...
                usegpu, stride, nrows, ncols, nlays, mus, datatype)

            if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-4,single(1e-4));
            import tansacnet.utility.*
            genU = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on');
            
            % Parameters
            nSamples = 8;
            nChsTotal = prod(stride);
            nAngles = (nChsTotal-2)*nChsTotal/8;
            angles = randn((nChsTotal-2)*nChsTotal/8,nrows*ncols*nlays);
            
            % nChsTotal x nRows x nCols x nLays x nSamples
            %X = randn(nrows,ncols,nlays,nChsTotal,nSamples,datatype);
            %dLdZ = randn(nrows,ncols,nlays,nChsTotal,nSamples,datatype);
            X = randn(nChsTotal,nrows,ncols,nlays,nSamples,datatype);
            dLdZ = randn(nChsTotal,nrows,ncols,nlays,nSamples,datatype);
            if usegpu
                X = gpuArray(X);
                dLdZ = gpuArray(dLdZ);
                angles = gpuArray(angles);
            end

            % Expected values
            % nChsTotal x nRows x nCols x nSamples
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            
            % dLdX = dZdX x dLdZ
            UnT = permute(genU.step(angles,mus,0),[2 1 3]);
            adLd_ = dLdZ; %permute(dLdZ,[4 1 2 3 5]);
            cdLd_low = reshape(adLd_(ps+1:ps+pa,:,:,:,:),pa,nrows*ncols*nlays,nSamples);
            for iSample = 1:nSamples
                for iblk = 1:(nrows*ncols*nlays)
                    cdLd_low(:,iblk,iSample) = UnT(:,:,iblk)*cdLd_low(:,iblk,iSample);
                end
            end
            adLd_(ps+1:ps+pa,:,:,:,:) = reshape(cdLd_low,pa,nrows,ncols,nlays,nSamples);
            expctddLdX = adLd_; %ipermute(adLd_,[4 1 2 3 5]);         
            
            
            % dLdWi = <dLdZ,(dVdWi)X>
            expctddLdW = zeros(nAngles,nrows*ncols*nlays,datatype);
            c_low = reshape(X(ps+1:ps+pa,:,:,:,:),pa,nrows*ncols*nlays,nSamples);
            dldz_low = reshape(dLdZ(ps+1:ps+pa,:,:,:,:),pa,nrows*ncols*nlays,nSamples);
            for iAngle = 1:nAngles
                dUn = genU.step(angles,mus,iAngle);
                for iblk = 1:(nrows*ncols*nlays)
                    c_low_iblk = squeeze(c_low(:,iblk,:));
                    c_low_iblk = dUn(:,:,iblk)*c_low_iblk;
                    dldz_iblk = squeeze(dldz_low(:,iblk,:));
                    expctddLdW(iAngle,iblk) = sum(dldz_iblk.*c_low_iblk,'all');
                end
            end
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunIntermediateRotation3dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols nlays],...
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