classdef lsunInitialRotation3dLayerTestCase < matlab.unittest.TestCase
    %LSUNINITIALROTATION3DLAYERTESTCASE
    %
    %   コンポーネント別に入力(nComponents=1のみサポート):
    %      nChs x nRows x nCols x nLays x nSamples
    %
    %   コンポーネント別に出力(nComponents=1のみサポート):
    %      nChs x nRows x nCols x nLays x nSamples
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
        stride = { [2 2 2], [ 1 2 4 ] };
        mus = { -1, 1 };
        datatype = { 'single', 'double' };
        nrows = struct('small', 2,'medium', 4, 'large', 8);
        ncols = struct('small', 2,'medium', 4, 'large', 8);
        nlays = struct('small', 4,'medium', 8, 'large', 16);     
        usegpu = struct( 'true', true, 'false', false);           
    end
    
    methods (TestClassTeardown)
        function finalCheck(~)
            import tansacnet.lsun.*
            layer = lsunInitialRotation3dLayer(...
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
            expctdName = 'V0';
            expctdDescription = "LSUN initial rotation " ...
                + "(ps,pa) = (" ...
                + ceil(prod(stride)/2) + "," ...
                + floor(prod(stride)/2) + "), "  ...
                + "(mv,mh,md) = (" ...
                + stride(1) + "," + stride(2) + "," + stride(3) + ")";
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunInitialRotation3dLayer(...
                'Stride',stride,...
                'Name',expctdName);
            
            % Actual values
            actualName = layer.Name;
            actualDescription = layer.Description;
            
            % Evaluation
            testCase.verifyEqual(actualName,expctdName);
            testCase.verifyEqual(actualDescription,expctdDescription);
        end
  
        function testPredictGrayscale(testCase, ...
                usegpu, stride, nrows, ncols, nlays, datatype)

            if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            
            % Parameters
            nSamples = 8;
            nDecs = prod(stride);
            nChsTotal = nDecs;
            % nDecs x nRows x nCols x nLays x nSamples
            %X = randn(nrows,ncols,nlays,nDecs,nSamples,datatype);
            X = randn(nDecs,nrows,ncols,nlays,nSamples,datatype);
            if usegpu
                X = gpuArray(X);
            end
            
            % Expected values
            % nChs x nRows x nCols x nLays x nSamples
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            W0 = repmat(eye(ps,datatype),[1 1 nrows*ncols*nlays]);
            U0 = repmat(eye(pa,datatype),[1 1 nrows*ncols*nlays]);
            %
            expctdZ = zeros(nChsTotal,nrows,ncols,nlays,nSamples,datatype);
            Y  = zeros(nChsTotal,nrows,ncols,nlays,datatype);
            for iSample=1:nSamples
                % Perumation in each block
                Ai = X(:,:,:,:,iSample); %Ai = permute(X(:,:,:,:,iSample),[4 1 2 3]);
                Yi = reshape(Ai,nDecs,nrows,ncols,nlays);
                %
                Ys = Yi(1:ps,:);
                Ya = Yi(ps+1:end,:);
                for iblk = 1:(nrows*ncols*nlays)
                    Ys(:,iblk) = W0(:,1:ps,iblk)*Ys(:,iblk);
                    Ya(:,iblk) = U0(:,1:pa,iblk)*Ya(:,iblk);
                end
                Y(1:ps,:,:,:) = reshape(Ys,ps,nrows,ncols,nlays);
                Y(ps+1:ps+pa,:,:,:) = reshape(Ya,pa,nrows,ncols,nlays);                
                expctdZ(:,:,:,:,iSample) = Y; 
            end
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunInitialRotation3dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols nlays],...
                'Name','V0');
            
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
        
        function testPredictGrayscaleWithRandomAngles(testCase, ...
                usegpu, stride, nrows, ncols, nlays ,datatype)
            
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
            nDecs = prod(stride);
            nChsTotal = nDecs;
            % nDecs x nRows x nCols x nLays x nSamples            
            %X = randn(nrows,ncols,nlays,nDecs,nSamples,datatype);
            X = randn(nDecs,nrows,ncols,nlays,nSamples,datatype);
            angles = randn((nChsTotal-2)*nChsTotal/4,nrows*ncols*nlays);
            if usegpu
                X = gpuArray(X);
                angles = gpuArray(angles);
            end            

            % Expected values
            % nChs x nRows x nCols x nLays x nSamples
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            W0 = genW.step(angles(1:size(angles,1)/2,:),1);
            U0 = genU.step(angles(size(angles,1)/2+1:end,:),1);
            %expctdZ = zeros(nrows,ncols,nlays,nChsTotal,nSamples,datatype);
            expctdZ = zeros(nChsTotal,nrows,ncols,nlays,nSamples,datatype);
            Y  = zeros(nChsTotal,nrows,ncols,nlays,datatype);
            for iSample=1:nSamples
                % Perumation in each block
                %Ai = permute(X(:,:,:,:,iSample),[4 1 2 3]);
                Ai = X(:,:,:,:,iSample);  
                Yi = reshape(Ai,nDecs,nrows,ncols,nlays);
                %
                Ys = Yi(1:ps,:);
                Ya = Yi(ps+1:end,:);
                for iblk = 1:(nrows*ncols*nlays)
                    Ys(:,iblk) = W0(:,1:ps,iblk)*Ys(:,iblk);
                    Ya(:,iblk) = U0(:,1:pa,iblk)*Ya(:,iblk);
                end
                Y(1:ps,:,:,:) = reshape(Ys,ps,nrows,ncols,nlays);
                Y(ps+1:ps+pa,:,:,:) = reshape(Ya,pa,nrows,ncols,nlays);
                expctdZ(:,:,:,:,iSample) = Y; %ipermute(Y,[4 1 2 3]);
            end
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunInitialRotation3dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols nlays],...
                'Name','V0');
            
            % Actual values
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
       
        function testPredictGrayscaleWithRandomAnglesNoDcLeackage(testCase, ...
                usegpu, stride, nrows, ncols, nlays, mus, datatype)
            
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
            nDecs = prod(stride);
            nChsTotal = nDecs;
            % nDecs x nRows x nCols x nLays x nSamples
            %X = randn(nrows,ncols,nlays,nDecs,nSamples,datatype);
            X = randn(nDecs,nrows,ncols,nlays,nSamples,datatype);
            angles = randn((nChsTotal-2)*nChsTotal/4,nrows*ncols*nlays);
            if usegpu
                X = gpuArray(X);
                angles = gpuArray(angles);
            end   

            % Expected values
            % nChs x nRows x nCols x nLays x nSamples
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            anglesNoDc = angles;
            anglesNoDc(1:ps-1,:)=zeros(ps-1,nrows*ncols*nlays);
            musW = mus*ones(ps,nrows*ncols*nlays);
            musW(1,1:end) = 1;
            musU = mus*ones(pa,nrows*ncols*nlays);
            W0 = genW.step(anglesNoDc(1:size(angles,1)/2,:),musW);
            U0 = genU.step(anglesNoDc(size(angles,1)/2+1:end,:),musU);
            %expctdZ = zeros(nrows,ncols,nlays,nChsTotal,nSamples,datatype);
            expctdZ = zeros(nChsTotal,nrows,ncols,nlays,nSamples,datatype);
            Y  = zeros(nChsTotal,nrows,ncols,nlays,datatype);
            for iSample=1:nSamples
                % Perumation in each block
                %Ai = permute(X(:,:,:,:,iSample),[4 1 2 3]);
                Ai = X(:,:,:,:,iSample); 
                Yi = reshape(Ai,nDecs,nrows*ncols*nlays);
                %
                Ys = Yi(1:ps,:);
                Ya = Yi(ps+1:end,:);
                for iblk = 1:(nrows*ncols*nlays)
                    Ys(:,iblk) = W0(:,1:ps,iblk)*Ys(:,iblk);
                    Ya(:,iblk) = U0(:,1:pa,iblk)*Ya(:,iblk);
                end
                Y(1:ps,:,:,:) = reshape(Ys,ps,nrows,ncols,nlays);
                Y(ps+1:end,:,:,:) = reshape(Ya,pa,nrows,ncols,nlays);                
                expctdZ(:,:,:,:,iSample) = Y; %ipermute(Y,[4 1 2 3]);
            end
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunInitialRotation3dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols nlays],...
                'NoDcLeakage',true,...
                'Name','V0');
            
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

        function testBackwardGrayscale(testCase, ...
                usegpu, stride, nrows, ncols, nlays, datatype)
            
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
            nDecs = prod(stride);
            nChsTotal = nDecs;
            nAnglesH = (nChsTotal-2)*nChsTotal/8;
            anglesW = zeros(nAnglesH,nrows*ncols*nlays,datatype);
            anglesU = zeros(nAnglesH,nrows*ncols*nlays,datatype);
            mus_ = 1;
            
            % nDecs x nRows x nCols x nLays x nSamples
            %X = randn(nrows,ncols,nlays,nDecs,nSamples,datatype);
            %dLdZ = randn(nrows,ncols,nlays,sum(nchs),nSamples,datatype);           
            X = randn(nDecs,nrows,ncols,nlays,nSamples,datatype);
            dLdZ = randn(nDecs,nrows,ncols,nlays,nSamples,datatype);                        
            if usegpu
                X = gpuArray(X);
                anglesW = gpuArray(anglesW);
                anglesU = gpuArray(anglesU);
                dLdZ = gpuArray(dLdZ);
            end

            % Expected values
            % nDecs x nRows x nCols x nSamples
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            
            % dLdX = dZdX x dLdZ
            W0T = permute(genW.step(anglesW,mus_,0),[2 1 3]);
            U0T = permute(genU.step(anglesU,mus_,0),[2 1 3]);
            Y = dLdZ; % permute(dLdZ,[4 1 2 3 5]);
            Ys = reshape(Y(1:ps,:,:,:,:),ps,nrows*ncols*nlays,nSamples);
            Ya = reshape(Y(ps+1:ps+pa,:,:,:,:),pa,nrows*ncols*nlays,nSamples);
            for iSample = 1:nSamples
                for iblk = 1:(nrows*ncols*nlays)
                    Ys(:,iblk,iSample) = W0T(1:ps,:,iblk)*Ys(:,iblk,iSample); 
                    Ya(:,iblk,iSample) = U0T(1:pa,:,iblk)*Ya(:,iblk,iSample);
                end
            end
            Zsa = cat(1,Ys,Ya);
            %expctddLdX = ipermute(reshape(Zsa,nDecs,nrows,ncols,nlays,nSamples),...
            %    [4 1 2 3 5]);
            expctddLdX = reshape(Zsa,nDecs,nrows,ncols,nlays,nSamples);
            
            % dLdWi = <dLdZ,(dVdWi)X>
            dldw_ = zeros(2*nAnglesH,nrows*ncols*nlays,1,datatype);
            dldz_ = dLdZ; %permute(dLdZ,[3 1 2 4]);
            dldz_upp = reshape(dldz_(1:ps,:,:,:,:),ps,nrows*ncols*nlays,nSamples);
            dldz_low = reshape(dldz_(ps+1:ps+pa,:,:,:,:),pa,nrows*ncols*nlays,nSamples);
            % (dVdWi)X
            a_ = X; %permute(X,[4 1 2 3 5]);
            c_upp = reshape(a_(1:ps,:,:,:,:),ps,nrows*ncols*nlays,nSamples);
            c_low = reshape(a_(ps+1:nDecs,:,:,:,:),pa,nrows*ncols*nlays,nSamples);
            for iAngle = 1:nAnglesH
                dW0 = genW.step(anglesW,mus_,iAngle);
                dU0 = genU.step(anglesU,mus_,iAngle);
                for iblk = 1:(nrows*ncols*nlays)                
                    dldz_upp_iblk = squeeze(dldz_upp(:,iblk,:));
                    dldz_low_iblk = squeeze(dldz_low(:,iblk,:));
                    c_upp_iblk = squeeze(c_upp(:,iblk,:));
                    c_low_iblk = squeeze(c_low(:,iblk,:));
                    d_upp_iblk = zeros(size(c_upp_iblk),'like',c_upp_iblk);
                    d_low_iblk = zeros(size(c_low_iblk),'like',c_low_iblk);
                    for iSample = 1:nSamples
                        d_upp_iblk(:,iSample) = dW0(:,1:ps,iblk)*c_upp_iblk(:,iSample);
                        d_low_iblk(:,iSample) = dU0(:,1:pa,iblk)*c_low_iblk(:,iSample);
                    end
                    dldw_(iAngle,iblk) = sum(dldz_upp_iblk.*d_upp_iblk,'all');
                    dldw_(nAnglesH+iAngle,iblk) = sum(dldz_low_iblk.*d_low_iblk,'all');
                end
            end
            expctddLdW = dldw_;

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunInitialRotation3dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols nlays],...
                'Name','V0');
            layer.Mus = mus_;
            %expctdZ = layer.predict(X);
            
            % Actual values
            [actualdLdX,actualdLdW] = layer.backward(X,[],dLdZ,[]);
            
            % Evaluation
            if usegpu
                testCase.verifyClass(actualdLdX,'gpuArray')
                actualdLdX = gather(actualdLdX);
                expctddLdX = gather(expctddLdX);
                testCase.verifyClass(actualdLdW,'gpuArray')
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
                usegpu, stride, nrows, ncols, nlays, datatype)

            if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-4,single(1e-3));
            import tansacnet.utility.*
            genW = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on');
            genU = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on');
            
            % Parameters
            nSamples = 8;
            nDecs = prod(stride);
            nChsTotal = nDecs;
            nAnglesH = (nChsTotal-2)*nChsTotal/8;
            anglesW = randn(nAnglesH,nrows*ncols*nlays,datatype);
            anglesU = randn(nAnglesH,nrows*ncols*nlays,datatype);
            mus_ = 1;
            
            % nDecs x nRows x nCols x nLays x nSamples
            %X = randn(nrows,ncols,nlays,nDecs,nSamples,datatype);
            %dLdZ = randn(nrows,ncols,nlays,sum(nchs),nSamples,datatype);
            X = randn(nDecs,nrows,ncols,nlays,nSamples,datatype);
            dLdZ = randn(nDecs,nrows,ncols,nlays,nSamples,datatype);
            if usegpu
                X = gpuArray(X);
                anglesW = gpuArray(anglesW);
                anglesU = gpuArray(anglesU);
                dLdZ = gpuArray(dLdZ);
            end

            % Expected values
            % nDecs x nRows x nCols x nLays x nSamples
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            
            % dLdX = dZdX x dLdZ
            W0T = permute(genW.step(anglesW,mus_,0),[2 1 3]);
            U0T = permute(genU.step(anglesU,mus_,0),[2 1 3]);
            Y = dLdZ; %permute(dLdZ,[3 1 2 4]);
            Ys = reshape(Y(1:ps,:,:,:,:),ps,nrows*ncols*nlays,nSamples);
            Ya = reshape(Y(ps+1:ps+pa,:,:,:,:),pa,nrows*ncols*nlays,nSamples);
            for iSample = 1:nSamples
                for iblk = 1:(nrows*ncols*nlays)
                    Ys(:,iblk,iSample) = W0T(1:ps,:,iblk)*Ys(:,iblk,iSample); 
                    Ya(:,iblk,iSample) = U0T(1:pa,:,iblk)*Ya(:,iblk,iSample);
                end
            end
            Zsa = cat(1,Ys,Ya);
            %expctddLdX = ipermute(reshape(Zsa,nDecs,nrows,ncols,nlays,nSamples),...
            %    [4 1 2 3 5]);
            expctddLdX = reshape(Zsa,nDecs,nrows,ncols,nlays,nSamples);
                        
            % dLdWi = <dLdZ,(dVdWi)X>
            dldw_ = zeros(2*nAnglesH,nrows*ncols*nlays,datatype);
            dldz_ = dLdZ; %permute(dLdZ,[4 1 2 3 5]);
            dldz_upp = reshape(dldz_(1:ps,:,:,:,:),ps,nrows*ncols*nlays,nSamples);
            dldz_low = reshape(dldz_(ps+1:ps+pa,:,:,:,:),pa,nrows*ncols*nlays,nSamples);
            % (dVdWi)X
            a_ = X; %permute(X,[4 1 2 3 5]);
            c_upp = reshape(a_(1:ps,:,:,:,:),ps,nrows*ncols*nlays,nSamples);
            c_low = reshape(a_(ps+1:nDecs,:,:,:,:),pa,nrows*ncols*nlays,nSamples);
            for iAngle = 1:nAnglesH
                dW0 = genW.step(anglesW,mus_,iAngle);
                dU0 = genU.step(anglesU,mus_,iAngle);
                for iblk = 1:(nrows*ncols*nlays)
                    dldz_upp_iblk = squeeze(dldz_upp(:,iblk,:));
                    dldz_low_iblk = squeeze(dldz_low(:,iblk,:));
                    c_upp_iblk = squeeze(c_upp(:,iblk,:));
                    c_low_iblk = squeeze(c_low(:,iblk,:));
                    d_upp_iblk = zeros(size(c_upp_iblk),'like',c_upp_iblk);
                    d_low_iblk = zeros(size(c_low_iblk),'like',c_low_iblk);
                    for iSample = 1:nSamples
                        d_upp_iblk(:,iSample) = dW0(:,1:ps,iblk)*c_upp_iblk(:,iSample);
                        d_low_iblk(:,iSample) = dU0(:,1:pa,iblk)*c_low_iblk(:,iSample);
                    end
                    dldw_(iAngle,iblk) = sum(dldz_upp_iblk.*d_upp_iblk,'all');
                    dldw_(nAnglesH+iAngle,iblk) = sum(dldz_low_iblk.*d_low_iblk,'all');
                end
            end
            expctddLdW = dldw_;

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunInitialRotation3dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols nlays],...
                'Name','V0');
            layer.Mus = mus_;
            layer.Angles = [anglesW; anglesU];
            %expctdZ = layer.predict(X);
            
            % Actual values
            [actualdLdX,actualdLdW] = layer.backward(X,[],dLdZ,[]);
            
            % Evaluation
            if usegpu
                testCase.verifyClass(actualdLdX,'gpuArray')
                actualdLdX = gather(actualdLdX);
                expctddLdX = gather(expctddLdX);
                testCase.verifyClass(actualdLdW,'gpuArray')
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

        function testBackwardGrayscaleWithRandomAnglesNoDcLeackage(testCase, ...
                usegpu, stride, nrows, ncols, nlays, mus, datatype)

            if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-4,single(1e-3));
            import tansacnet.utility.*
            genW = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on');
            genU = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on');
            
            % Parameters
            nSamples = 8;
            nDecs = prod(stride);
            nChsTotal = nDecs;
            nAnglesH = (nChsTotal-2)*nChsTotal/8;
            anglesW = randn(nAnglesH,nrows*ncols*nlays,datatype);
            anglesU = randn(nAnglesH,nrows*ncols*nlays,datatype);
            
            % nDecs x nRows x nCols x nLays x nSamples
            %X = randn(nrows,ncols,nlays,nDecs,nSamples,datatype);
            %dLdZ = randn(nrows,ncols,nlays,sum(nchs),nSamples,datatype);
            X = randn(nDecs,nrows,ncols,nlays,nSamples,datatype);
            dLdZ = randn(nDecs,nrows,ncols,nlays,nSamples,datatype);            
            if usegpu
                X = gpuArray(X);
                anglesW = gpuArray(anglesW);
                anglesU = gpuArray(anglesU);
                dLdZ = gpuArray(dLdZ);
            end

            % Expected values
            % nDecs x nRows x nCols x nLays x nSamples
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            
            % dLdX = dZdX x dLdZ
            anglesW_NoDc = anglesW;
            anglesW_NoDc(1:ps-1,:)=zeros(ps-1,nrows*ncols*nlays);
            musW = mus*ones(ps,nrows*ncols*nlays);
            musW(1,:) = ones(1,nrows*ncols*nlays);
            musU = mus*ones(pa,nrows*ncols*nlays);            
            W0T = permute(genW.step(anglesW_NoDc,musW,0),[2 1 3]);
            U0T = permute(genU.step(anglesU,musU,0),[2 1 3]);
            Y = dLdZ; %permute(dLdZ,[4 1 2 3 5]);
            Ys = reshape(Y(1:ps,:,:,:,:),ps,nrows*ncols*nlays,nSamples);
            Ya = reshape(Y(ps+1:ps+pa,:,:,:,:),pa,nrows*ncols*nlays,nSamples);
            for iSample = 1:nSamples
                for iblk = 1:(nrows*ncols*nlays)
                    Ys(:,iblk,iSample) = W0T(1:ps,:,iblk)*Ys(:,iblk,iSample);
                    Ya(:,iblk,iSample) = U0T(1:pa,:,iblk)*Ya(:,iblk,iSample);
                end
            end
            Zsa = cat(1,Ys,Ya);
            %expctddLdX = ipermute(reshape(Zsa,nDecs,nrows,ncols,nSamples),...
            %    [3 1 2 4]);
            expctddLdX = reshape(Zsa,nDecs,nrows,ncols,nlays,nSamples);
            
            % dLdWi = <dLdZ,(dVdWi)X>
            dldw_ = zeros(2*nAnglesH,nrows*ncols*nlays,datatype);
            dldz_ = dLdZ; %permute(dLdZ,[3 1 2 4]);
            dldz_upp = reshape(dldz_(1:ps,:,:,:,:),ps,nrows*ncols*nlays,nSamples);
            dldz_low = reshape(dldz_(ps+1:ps+pa,:,:,:,:),pa,nrows*ncols*nlays,nSamples);
            % (dVdWi)X
            a_ = X; %permute(X,[4 1 2 3 5]);
            c_upp = reshape(a_(1:ps,:,:,:,:),ps,nrows*ncols*nlays,nSamples);
            c_low = reshape(a_(ps+1:nDecs,:,:,:,:),pa,nrows*ncols*nlays,nSamples);
            for iAngle = 1:nAnglesH
                dW0 = genW.step(anglesW_NoDc,musW,iAngle);
                dU0 = genU.step(anglesU,musU,iAngle);
                for iblk = 1:(nrows*ncols*nlays)
                    dldz_upp_iblk = squeeze(dldz_upp(:,iblk,:));
                    dldz_low_iblk = squeeze(dldz_low(:,iblk,:));
                    c_upp_iblk = squeeze(c_upp(:,iblk,:));
                    c_low_iblk = squeeze(c_low(:,iblk,:));                    
                    d_upp_iblk = zeros(size(c_upp_iblk),'like',c_upp_iblk);
                    d_low_iblk = zeros(size(c_low_iblk),'like',c_low_iblk);
                    for iSample = 1:nSamples
                        d_upp_iblk(:,iSample) = dW0(:,1:ps,iblk)*c_upp_iblk(:,iSample);
                        d_low_iblk(:,iSample) = dU0(:,1:pa,iblk)*c_low_iblk(:,iSample);
                    end
                    dldw_(iAngle,iblk) = sum(dldz_upp_iblk.*d_upp_iblk,'all');
                    dldw_(nAnglesH+iAngle,iblk) = sum(dldz_low_iblk.*d_low_iblk,'all');
                end
            end
            expctddLdW = dldw_;
        
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunInitialRotation3dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols nlays],...
                'NoDcLeakage',true,...
                'Name','V0');
            layer.Mus = mus;
            layer.Angles = [anglesW; anglesU];
            %expctdZ = layer.predict(X);
            
            % Actual values
            [actualdLdX,actualdLdW] = layer.backward(X,[],dLdZ,[]);
            
            % Evaluation
            if usegpu
                testCase.verifyClass(actualdLdX,'gpuArray')
                actualdLdX = gather(actualdLdX);
                expctddLdX = gather(expctddLdX);
                testCase.verifyClass(actualdLdW,'gpuArray')
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