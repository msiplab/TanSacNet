classdef lsunFinalRotation2dLayerTestCase < matlab.unittest.TestCase
    %LSUNFINALROTATION2DLAYERTESTCASE 
    %
    %   コンポーネント別に入力(nComponents):
    %      nDecs x nRows x nCols x nSamples
    %
    %   コンポーネント別に出力(nComponents):
    %      nDecs x nRows x nCols x nSamples
    %
    % Requirements: MATLAB R2022a
    %
    % Copyright (c) 2022, Shogo MURAMATSU
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
        stride = { [2 2], [4 4] };
        mus = { -1, 1 };
        datatype = { 'single', 'double' };
        nrows = struct('small', 2,'medium', 4, 'large', 8);
        ncols = struct('small', 2,'medium', 4, 'large', 8);
        usegpu = struct( 'true', true, 'false', false);           
    end
    
    methods (TestClassTeardown)
        function finalCheck(~)
            import tansacnet.lsun.*
            layer = lsunFinalRotation2dLayer(...
                'Stride',[2 2],...
                'NumberOfBlocks',[8 8]);
            fprintf("\n --- Check layer for 2-D images ---\n");
            checkLayer(layer,[4 8 8],...
                'ObservationDimension',4,...                
                'CheckCodegenCompatibility',true)
        end
    end
    
    methods (Test)
        
        function testConstructor(testCase, stride)
            
            % Expected values
            expctdName = 'V0~';
            expctdDescription = "LSUN final rotation " ...
                + "(ps,pa) = (" ...
                + ceil(prod(stride)/2) + "," ...
                + floor(prod(stride)/2) + "), " ...               
                + "(mv,mh) = (" ...
                + stride(1) + "," + stride(2) + ")";
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunFinalRotation2dLayer(...
                'Stride',stride,...
                'Name',expctdName);
            
            % Actual values
            actualName = layer.Name;
            actualDescription = layer.Description;
            
            % Evaluation
            testCase.verifyEqual(actualName,expctdName);
            testCase.verifyEqual(actualDescription,expctdDescription);
        end

        function testConstructorWithDeviceAndDType(testCase, stride, usegpu, datatype)
            
            % Expected values
            expctdName = 'V0~';
     
            device_ = ["cpu", "cuda"];
            expctdDevice = device_(usegpu+1);
            expctdDType = datatype;
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunFinalRotation2dLayer(...
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

        function testPredictGrayscale(testCase, ...
                usegpu, stride, nrows, ncols, datatype)
            
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
            % nChs x nRows x nCols x nSamples
            %X = randn(nrows,ncols,sum(stride),nSamples,datatype);
            X = randn(nChsTotal,nrows,ncols,nSamples,datatype);
            if usegpu
                X = gpuArray(X);
            end

            % Expected values        
            % nDecs x nRows x nCols x nSamples
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            W0T = repmat(eye(ps,datatype),[1 1 nrows*ncols]);
            U0T = repmat(eye(pa,datatype),[1 1 nrows*ncols]);
            Y = X; %permute(X,[3 1 2 4]);
            Ys = reshape(Y(1:ps,:,:,:),ps,nrows*ncols,nSamples);
            Ya = reshape(Y(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            for iSample=1:nSamples
                for iblk = 1:(nrows*ncols)
                    Ys(:,iblk,iSample) = W0T(1:ps,:,iblk)*Ys(:,iblk,iSample); 
                    Ya(:,iblk,iSample) = U0T(1:pa,:,iblk)*Ya(:,iblk,iSample);
                end
            end
            Zsa = cat(1,Ys,Ya);
            %expctdZ = ipermute(reshape(Zsa,nDecs,nrows,ncols,nSamples),...
            %    [3 1 2 4]);
            expctdZ = reshape(Zsa,nDecs,nrows,ncols,nSamples);
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunFinalRotation2dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols],...
                'Name','V0~');
            
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

        function testPredictGrayscaleWithDeviceAndDType(testCase, ...
                usegpu, stride, nrows, ncols, datatype)

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
            %nrows_ = 2;
            %ncols_ = 2;
            nSamples = 8;
            nDecs = prod(stride);
            nChsTotal = nDecs;
            % nDecs x nRows x nCols x nSamples
            %X = randn(nrows,ncols,nDecs,nSamples,datatype);
            X = randn(nChsTotal,nrows,ncols,nSamples,datatype);
            if usegpu
                X = gpuArray(X);
            end
            
            % Expected values        
            % nDecs x nRows x nCols x nSamples
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            W0T = repmat(eye(ps,datatype),[1 1 nrows*ncols]);
            U0T = repmat(eye(pa,datatype),[1 1 nrows*ncols]);
            Y = X; %permute(X,[3 1 2 4]);
            Ys = reshape(Y(1:ps,:,:,:),ps,nrows*ncols,nSamples);
            Ya = reshape(Y(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            for iSample=1:nSamples
                for iblk = 1:(nrows*ncols)
                    Ys(:,iblk,iSample) = W0T(1:ps,:,iblk)*Ys(:,iblk,iSample); 
                    Ya(:,iblk,iSample) = U0T(1:pa,:,iblk)*Ya(:,iblk,iSample);
                end
            end
            Zsa = cat(1,Ys,Ya);
            %expctdZ = ipermute(reshape(Zsa,nDecs,nrows,ncols,nSamples),...
            %    [3 1 2 4]);
            expctdZ = reshape(Zsa,nDecs,nrows,ncols,nSamples);

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunFinalRotation2dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols],...
                'Name','V0~',...
                'Device',expctdDevice,...
                'DType',expctdDType);
            
            % Actual values
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
        
        function testPredictGrayscaleWithRandomAngles(testCase, ...
                usegpu, stride, nrows, ncols, datatype)
            
            if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end    
            if usegpu
                device_ = "cuda";
            else
                device_ = "cpu";
            end
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            import tansacnet.utility.*
            genW = OrthonormalMatrixGenerationSystem('Device','cpu');
            genU = OrthonormalMatrixGenerationSystem('Device','cpu');
            
            % Parameters
            nSamples = 8;
            nDecs = prod(stride);
            nChsTotal = nDecs;
            % nChs x nRows x nCols x nSamples
            X = randn(nDecs,nrows,ncols,nSamples,datatype);
            angles = randn((nChsTotal-2)*nChsTotal/4,nrows*ncols);            

            % Expected values
            % nDecs x nRows x nCols x nSamples
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            W0T = permute(genW.step(angles(1:size(angles,1)/2,:),1),[2 1 3]);
            U0T = permute(genU.step(angles(size(angles,1)/2+1:end,:),1),[2 1 3]);
            Y = X; %permute(X,[3 1 2 4]);
            Ys = reshape(Y(1:ps,:,:,:),ps,nrows*ncols,nSamples);
            Ya = reshape(Y(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            for iSample=1:nSamples
                for iblk = 1:(nrows*ncols)
                    Ys(:,iblk,iSample) = W0T(1:ps,:,iblk)*Ys(:,iblk,iSample);
                    Ya(:,iblk,iSample) = U0T(1:pa,:,iblk)*Ya(:,iblk,iSample);
                end
            end
            Zsa = cat(1,Ys,Ya);
            %expctdZ = ipermute(reshape(Zsa,nDecs,nrows,ncols,nSamples),...
            %    [3 1 2 4]);
            expctdZ = reshape(Zsa,nDecs,nrows,ncols,nSamples);
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunFinalRotation2dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols],...
                'Name','V0~', ...
                'Device',device_);
            
            % Actual values
            if usegpu
                X = gpuArray(X);
                angles = gpuArray(angles);
            end   
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
                usegpu, stride, nrows, ncols, mus, datatype)

            if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            if usegpu
                device_ = "cuda";
            else
                device_ = "cpu";
            end
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            import tansacnet.utility.*
            genW = OrthonormalMatrixGenerationSystem('Device','cpu');
            genU = OrthonormalMatrixGenerationSystem('Device','cpu');
            
            % Parameters
            nSamples = 8;
            nDecs = prod(stride);
            nChsTotal = nDecs;
            % nChs x nRows x nCols x nSamples
            %X = randn(nrows,ncols,sum(stride),nSamples,datatype);
            X = randn(nDecs,nrows,ncols,nSamples,datatype);
            angles = randn((nChsTotal-2)*nChsTotal/4,nrows*ncols);            

            % Expected values
            % nDecs x nRows x nCols x nSamples
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            anglesNoDc = angles;
            anglesNoDc(1:ps-1,:)=zeros(ps-1,nrows*ncols);
            musW = mus*ones(ps,nrows*ncols);
            musW(1,:) = 1;
            musU = mus*ones(pa,nrows*ncols);
            W0T = permute(genW.step(anglesNoDc(1:size(angles,1)/2,:),musW),[2 1 3]);
            U0T = permute(genU.step(anglesNoDc(size(angles,1)/2+1:end,:),musU),[2 1 3]);
            Y = X; %permute(X,[3 1 2 4]);
            Ys = reshape(Y(1:ps,:,:,:),ps,nrows*ncols,nSamples);
            Ya = reshape(Y(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            for iSample=1:nSamples
                for iblk = 1:(nrows*ncols)
                    Ys(:,iblk,iSample) = W0T(1:ps,:,iblk)*Ys(:,iblk,iSample);
                    Ya(:,iblk,iSample) = U0T(1:pa,:,iblk)*Ya(:,iblk,iSample);
                end
            end
            Zsa = cat(1,Ys,Ya);
            %expctdZ = ipermute(reshape(Zsa,nDecs,nrows,ncols,nSamples),...
            %    [3 1 2 4]);
            expctdZ = reshape(Zsa,nDecs,nrows,ncols,nSamples);

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunFinalRotation2dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols],...
                'NoDcLeakage',true,...
                'Name','V0~', ...
                'Device',device_);
            
            % Actual values
            if usegpu
                X = gpuArray(X);
                angles = gpuArray(angles);
            end
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
                usegpu, stride, nrows, ncols, datatype)
            
            if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            if usegpu
                device_ = "cuda";
            else
                device_ = "cpu";
            end
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-4,single(1e-4));
            import tansacnet.utility.*
            genW = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on',...
                'Device','cpu',...
                'DType',datatype);
            genU = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on',...
                'Device','cpu',...
                'DType',datatype);            
            
            % Parameters
            nSamples = 8;
            nDecs = prod(stride);
            nChsTotal = nDecs;
            nAnglesH = (nChsTotal-2)*nChsTotal/8;
            anglesW = zeros(nAnglesH,nrows*ncols,datatype);            
            anglesU = zeros(nAnglesH,nrows*ncols,datatype);  
            mus_ = cast(1,datatype);
            
            % nDecs x nRows x nCols x nSamples            
            %X = randn(nrows,ncols,sum(stride),nSamples,datatype); 
            %dLdZ = randn(nrows,ncols,nDecs,nSamples,datatype);            
            X = randn(nDecs,nrows,ncols,nSamples,datatype);            
            dLdZ = randn(nDecs,nrows,ncols,nSamples,datatype);
            
            % Expected values
            % nChs x nRows x nCols x nSamples
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            
            % dLdX = dZdX x dLdZ
            W0 = genW.step(anglesW,mus_,0);
            U0 = genU.step(anglesU,mus_,0);
            %expctddLdX = zeros(nrows,ncols,nChsTotal,nSamples,datatype);
            expctddLdX = zeros(nChsTotal,nrows,ncols,nSamples,datatype);
            Y  = zeros(nChsTotal,nrows,ncols,datatype);
            for iSample=1:nSamples
                % Perumation in each block
                %Ai = permute(dLdZ(:,:,:,iSample),[3 1 2]);
                Ai = dLdZ(:,:,:,iSample);
                Yi = reshape(Ai,nDecs,nrows*ncols);
                Ys = Yi(1:ps,:);
                Ya = Yi(ps+1:end,:);
                for iblk=1:(nrows*ncols)
                    Ys(:,iblk) = W0(:,1:ps,iblk)*Ys(:,iblk);
                    Ya(:,iblk) = U0(:,1:pa,iblk)*Ya(:,iblk);
                end
                Y(1:ps,:,:) = reshape(Ys,ps,nrows,ncols);
                Y(ps+1:end,:,:) = reshape(Ya,pa,nrows,ncols);
                %expctddLdX(:,:,:,iSample) = ipermute(Y,[3 1 2]);
                expctddLdX(:,:,:,iSample) = Y;
            end
            
            % dLdWi = <dLdZ,(dVdWi)X>
            dldw_ = zeros(2*nAnglesH,nrows*ncols,datatype);
            dldz_ = dLdZ; %permute(dLdZ,[3 1 2 4]);
            dldz_upp = reshape(dldz_(1:ps,:,:,:),ps,nrows*ncols,nSamples);
            dldz_low = reshape(dldz_(ps+1:nDecs,:,:,:),pa,nrows*ncols,nSamples);
            % (dVdWi)X
            a_ = X; %permute(X,[3 1 2 4]);
            c_upp = reshape(a_(1:ps,:,:,:),ps,nrows*ncols,nSamples);                
            c_low = reshape(a_(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            for iAngle = 1:nAnglesH
                dW0_T = permute(genW.step(anglesW,mus_,iAngle),[2 1 3]);
                dU0_T = permute(genU.step(anglesU,mus_,iAngle),[2 1 3]);
                for iblk=1:(nrows*ncols)
                    dldz_upp_iblk = squeeze(dldz_upp(:,iblk,:));
                    dldz_low_iblk = squeeze(dldz_low(:,iblk,:));
                    c_upp_iblk = squeeze(c_upp(:,iblk,:));
                    c_low_iblk = squeeze(c_low(:,iblk,:));
                    d_upp_iblk = zeros(size(c_upp_iblk),'like',c_upp_iblk);
                    d_low_iblk = zeros(size(c_low_iblk),'like',c_low_iblk);
                    for iSample = 1:nSamples
                        d_upp_iblk(:,iSample) = dW0_T(1:ps,:,iblk)*c_upp_iblk(:,iSample);
                        d_low_iblk(:,iSample) = dU0_T(1:pa,:,iblk)*c_low_iblk(:,iSample);
                    end
                    dldw_(iAngle,iblk) = sum(dldz_upp_iblk.*d_upp_iblk,'all');
                    dldw_(nAnglesH+iAngle,iblk) = sum(dldz_low_iblk.*d_low_iblk,'all');
                end
            end
            expctddLdW = dldw_;
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunFinalRotation2dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols],...
                'Name','V0~', ...
                'Device',device_);            
            %expctdZ = layer.predict(X);
            
            % Actual values
            if usegpu
                X = gpuArray(X);
                mus_ = gpuArray(mus_);
                %anglesW = gpuArray(anglesW);
                %anglesU = gpuArray(anglesU);
                dLdZ = gpuArray(dLdZ);
            end
            layer.Mus = mus_;
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

        function testBackwardGrayscaleWithDeviceAndDType(testCase, ...
                usegpu, stride, datatype)
            
            if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end

            device_ = ["cpu", "cuda"];      
            expctdDevice = device_(usegpu+1);
            expctdDType = datatype;      

            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-4,single(1e-4));
            import tansacnet.utility.*
            genW = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on',...
                'Device','cpu',...
                'DType',datatype);
            genU = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on',...
                'Device','cpu',...
                'DType',datatype);
            
            nrows_ = 8;
            ncols_ = 8;

            % Parameters
            nSamples = 8;
            nDecs = prod(stride);
            nChsTotal = nDecs;
            nAnglesH = (nChsTotal-2)*nChsTotal/8;
            anglesW = zeros(nAnglesH,nrows_*ncols_,datatype);
            anglesU = zeros(nAnglesH,nrows_*ncols_,datatype);
            mus_ = cast(1,datatype);
            
            % nDecs x nRows x nCols x nSamples
            %X = randn(nrows,ncols,nDecs,nSamples,datatype);
            %dLdZ = randn(nrows,ncols,sum(stride),nSamples,datatype);            
            X = randn(nDecs,nrows_,ncols_,nSamples,datatype);
            dLdZ = randn(nDecs,nrows_,ncols_,nSamples,datatype);   

            % Expected values
            % nChs x nRows x nCols x nSamples
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            
            % dLdX = dZdX x dLdZ
            W0 = genW.step(anglesW,mus_,0);
            U0 = genU.step(anglesU,mus_,0);
            %expctddLdX = zeros(nrows,ncols,nChsTotal,nSamples,datatype);
            expctddLdX = zeros(nChsTotal,nrows_,ncols_,nSamples,datatype);
            Y  = zeros(nChsTotal,nrows_,ncols_,datatype);
            for iSample=1:nSamples
                % Perumation in each block
                %Ai = permute(dLdZ(:,:,:,iSample),[3 1 2]);
                Ai = dLdZ(:,:,:,iSample);
                Yi = reshape(Ai,nDecs,nrows_*ncols_);
                Ys = Yi(1:ps,:);
                Ya = Yi(ps+1:end,:);
                for iblk=1:(nrows_*ncols_)
                    Ys(:,iblk) = W0(:,1:ps,iblk)*Ys(:,iblk);
                    Ya(:,iblk) = U0(:,1:pa,iblk)*Ya(:,iblk);
                end
                Y(1:ps,:,:) = reshape(Ys,ps,nrows_,ncols_);
                Y(ps+1:end,:,:) = reshape(Ya,pa,nrows_,ncols_);
                %expctddLdX(:,:,:,iSample) = ipermute(Y,[3 1 2]);
                expctddLdX(:,:,:,iSample) = Y;
            end
            
            % dLdWi = <dLdZ,(dVdWi)X>
            dldw_ = zeros(2*nAnglesH,nrows_*ncols_,datatype);
            dldz_ = dLdZ; %permute(dLdZ,[3 1 2 4]);
            dldz_upp = reshape(dldz_(1:ps,:,:,:),ps,nrows_*ncols_,nSamples);
            dldz_low = reshape(dldz_(ps+1:nDecs,:,:,:),pa,nrows_*ncols_,nSamples);
            % (dVdWi)X
            a_ = X; %permute(X,[3 1 2 4]);
            c_upp = reshape(a_(1:ps,:,:,:),ps,nrows_*ncols_,nSamples);                
            c_low = reshape(a_(ps+1:ps+pa,:,:,:),pa,nrows_*ncols_,nSamples);
            for iAngle = 1:nAnglesH
                dW0_T = permute(genW.step(anglesW,mus_,iAngle),[2 1 3]);
                dU0_T = permute(genU.step(anglesU,mus_,iAngle),[2 1 3]);
                for iblk=1:(nrows_*ncols_)
                    dldz_upp_iblk = squeeze(dldz_upp(:,iblk,:));
                    dldz_low_iblk = squeeze(dldz_low(:,iblk,:));
                    c_upp_iblk = squeeze(c_upp(:,iblk,:));
                    c_low_iblk = squeeze(c_low(:,iblk,:));
                    d_upp_iblk = zeros(size(c_upp_iblk),'like',c_upp_iblk);
                    d_low_iblk = zeros(size(c_low_iblk),'like',c_low_iblk);
                    for iSample = 1:nSamples
                        d_upp_iblk(:,iSample) = dW0_T(1:ps,:,iblk)*c_upp_iblk(:,iSample);
                        d_low_iblk(:,iSample) = dU0_T(1:pa,:,iblk)*c_low_iblk(:,iSample);
                    end
                    dldw_(iAngle,iblk) = sum(dldz_upp_iblk.*d_upp_iblk,'all');
                    dldw_(nAnglesH+iAngle,iblk) = sum(dldz_low_iblk.*d_low_iblk,'all');
                end
            end
            expctddLdW = dldw_;

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunFinalRotation2dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows_ ncols_],...
                'Name','V0~',...
                'Device',expctdDevice,...
                'DType',expctdDType);            
            %expctdZ = layer.predict(X);
            
            % Actual values
            if usegpu
                X = gpuArray(X);
                mus_ = gpuArray(mus_);
                %anglesW = gpuArray(anglesW);
                %anglesU = gpuArray(anglesU);
                dLdZ = gpuArray(dLdZ);
            end
            layer.Mus = mus_;
            [actualdLdX,actualdLdW] = layer.backward(X,[],dLdZ,[]);
            actualDevice = layer.Device;
            
            % Evaluation
            testCase.verifyEqual(actualDevice,expctdDevice);
            if actualDevice == "cuda"
                testCase.verifyClass(actualdLdX,'gpuArray')
                actualdLdX = gather(actualdLdX);
                expctddLdX = gather(expctddLdX);
                testCase.verifyClass(actualdLdW,'gpuArray')
                actualdLdW = gather(actualdLdW);
                expctddLdW = gather(expctddLdW);
            end
            testCase.verifyInstanceOf(actualdLdX,expctdDType);
            testCase.verifyInstanceOf(actualdLdW,expctdDType);
            testCase.verifyThat(actualdLdX,...
                IsEqualTo(expctddLdX,'Within',tolObj));
            testCase.verifyThat(actualdLdW,...
                IsEqualTo(expctddLdW,'Within',tolObj));
            
        end

        function testBackwardGayscaleWithRandomAngles(testCase, ...
                usegpu, stride, nrows, ncols, datatype)
            
            if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            if usegpu
                device_ = "cuda";
            else
                device_ = "cpu";
            end
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-4,single(1e-4));
            import tansacnet.utility.*
            genW = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on',...
                'Device','cpu',...
                'DType',datatype);
            genU = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on',...
                'Device','cpu',...
                'DType',datatype);
            
            % Parameters
            nSamples = 8;
            nDecs = prod(stride);
            nChsTotal = nDecs;
            nAnglesH = (nChsTotal-2)*nChsTotal/8;
            anglesW = randn(nAnglesH,nrows*ncols,datatype);
            anglesU = randn(nAnglesH,nrows*ncols,datatype);
            mus_ = cast(1,datatype);

            % nDecs x nRows x nCols x nSamples            
            %X = randn(nrows,ncols,sum(stride),nSamples,datatype); 
            %dLdZ = randn(nrows,ncols,nDecs,nSamples,datatype);            
            X = randn(nDecs,nrows,ncols,nSamples,datatype);            
            dLdZ = randn(nDecs,nrows,ncols,nSamples,datatype);            

            % Expected values
            % nChs x nRows x nCols x nSamples
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            
            % dLdX = dZdX x dLdZ
            W0 = genW.step(anglesW,mus_,0);
            U0 = genU.step(anglesU,mus_,0);
            %expctddLdX = zeros(nrows,ncols,nChsTotal,nSamples,datatype);
            expctddLdX = zeros(nChsTotal,nrows,ncols,nSamples,datatype);
            Y  = zeros(nChsTotal,nrows,ncols,datatype);
            for iSample=1:nSamples
                % Perumation in each block
                %Ai = permute(dLdZ(:,:,:,iSample),[3 1 2]);
                Ai = dLdZ(:,:,:,iSample);
                Yi = reshape(Ai,nDecs,nrows*ncols);
                Ys = Yi(1:ps,:);
                Ya = Yi(ps+1:end,:);
                for iblk=1:(nrows*ncols)
                    Ys(:,iblk) = W0(:,1:ps,iblk)*Ys(:,iblk);
                    Ya(:,iblk) = U0(:,1:pa,iblk)*Ya(:,iblk);
                end
                Y(1:ps,:,:) = reshape(Ys,ps,nrows,ncols);
                Y(ps+1:end,:,:) = reshape(Ya,pa,nrows,ncols);
                %expctddLdX(:,:,:,iSample) = ipermute(Y,[3 1 2]);
                expctddLdX(:,:,:,iSample) = Y;
            end
            
            % dLdWi = <dLdZ,(dVdWi)X>
            dldw_ = zeros(2*nAnglesH,nrows*ncols,datatype);
            dldz_ = dLdZ; %permute(dLdZ,[3 1 2 4]);
            dldz_upp = reshape(dldz_(1:ps,:,:,:),ps,nrows*ncols,nSamples);
            dldz_low = reshape(dldz_(ps+1:nDecs,:,:,:),pa,nrows*ncols,nSamples);
            % (dVdWi)X
            a_ = X; %permute(X,[3 1 2 4]);
            c_upp = reshape(a_(1:ps,:,:,:),ps,nrows*ncols,nSamples);                
            c_low = reshape(a_(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            for iAngle = 1:nAnglesH
                dW0_T = permute(genW.step(anglesW,mus_,iAngle),[2 1 3]);
                dU0_T = permute(genU.step(anglesU,mus_,iAngle),[2 1 3]);
                for iblk = 1:(nrows*ncols)
                    dldz_upp_iblk = squeeze(dldz_upp(:,iblk,:));
                    dldz_low_iblk = squeeze(dldz_low(:,iblk,:));
                    c_upp_iblk = squeeze(c_upp(:,iblk,:));
                    c_low_iblk = squeeze(c_low(:,iblk,:));
                    d_upp_iblk = zeros(size(c_upp_iblk),'like',c_upp_iblk);
                    d_low_iblk = zeros(size(c_low_iblk),'like',c_low_iblk);
                    for iSample = 1:nSamples
                        d_upp_iblk(:,iSample) = dW0_T(1:ps,:,iblk)*c_upp_iblk(:,iSample);
                        d_low_iblk(:,iSample) = dU0_T(1:pa,:,iblk)*c_low_iblk(:,iSample);
                    end
                    dldw_(iAngle,iblk) = sum(dldz_upp_iblk.*d_upp_iblk,'all');
                    dldw_(nAnglesH+iAngle,iblk) = sum(dldz_low_iblk.*d_low_iblk,'all');
                end
            end
            expctddLdW = dldw_;
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunFinalRotation2dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols],...
                'Name','V0~', ...
                'Device',device_);            
            %expctdZ = layer.predict(X);
            
            % Actual values
            if usegpu
                X = gpuArray(X);
                mus_ = gpuArray(mus_);
                %anglesW = gpuArray(anglesW);
                %anglesU = gpuArray(anglesU);
                dLdZ = gpuArray(dLdZ);
            end
            layer.Mus = mus_;
            layer.Angles = [anglesW; anglesU];
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

        function testBackwardWithRandomAnglesNoDcLeackage(testCase, ...
                usegpu, stride, nrows, ncols, mus, datatype)
            
            if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            if usegpu
                device_ = "cuda";
            else
                device_ = "cpu";
            end
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-4,single(1e-4));
            import tansacnet.utility.*
            genW = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on',...
                'Device','cpu',...
                'DType',datatype);
            genU = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on',...
                'Device','cpu',...
                'DType',datatype);
            
            % Parameters
            nSamples = 8;
            nDecs = prod(stride);
            nChsTotal = nDecs;
            nAnglesH = (nChsTotal-2)*nChsTotal/8;
            anglesW = randn(nAnglesH,nrows*ncols,datatype);
            anglesU = randn(nAnglesH,nrows*ncols,datatype);
            mus = cast(mus,datatype);
            
            % nDecs x nRows x nCols x nSamples
            %X = randn(nrows,ncols,sum(stride),nSamples,datatype);
            %dLdZ = randn(nrows,ncols,nDecs,nSamples,datatype);
            X = randn(nDecs,nrows,ncols,nSamples,datatype);
            dLdZ = randn(nDecs,nrows,ncols,nSamples,datatype);          
            
            % Expected values
            % nChs x nRows x nCols x nSamples
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            
            % dLdX = dZdX x dLdZ
            anglesW_NoDc = anglesW;
            anglesW_NoDc(1:ps-1,:)=zeros(ps-1,nrows*ncols);
            musW = mus*ones(ps,nrows*ncols);
            musW(1,:) = 1;
            musU = mus*ones(pa,nrows*ncols);
            W0 = genW.step(anglesW_NoDc,musW,0);
            U0 = genU.step(anglesU,musU,0);
            %expctddLdX = zeros(nrows,ncols,nChsTotal,nSamples,datatype);
            expctddLdX = zeros(nChsTotal,nrows,ncols,nSamples,datatype);
            Y  = zeros(nChsTotal,nrows,ncols,datatype);
            for iSample=1:nSamples
                % Perumation in each block
                %Ai = permute(dLdZ(:,:,:,iSample),[3 1 2]);
                Ai = dLdZ(:,:,:,iSample);
                Yi = reshape(Ai,nDecs,nrows,ncols);
                %
                Ys = Yi(1:ps,:);
                Ya = Yi(ps+1:end,:);
                for iblk=1:(nrows*ncols)
                    Ys(:,iblk) = W0(:,1:ps,iblk)*Ys(:,iblk);
                    Ya(:,iblk) = U0(:,1:pa,iblk)*Ya(:,iblk);
                end
                Y(1:ps,:,:) = reshape(Ys,ps,nrows,ncols);
                Y(ps+1:end,:,:) = reshape(Ya,pa,nrows,ncols);
                expctddLdX(:,:,:,iSample) = Y; %ipermute(Y,[3 1 2]);
            end
            
            % dLdWi = <dLdZ,(dVdWi)X>
            dldw_ = zeros(2*nAnglesH,nrows*ncols,datatype);
            dldz_ = dLdZ; %permute(dLdZ,[3 1 2 4]);
            dldz_upp = reshape(dldz_(1:ps,:,:,:),ps,nrows*ncols,nSamples);
            dldz_low = reshape(dldz_(ps+1:nDecs,:,:,:),pa,nrows*ncols,nSamples);
            % (dVdWi)X
            a_ = X; %permute(X,[3 1 2 4]);
            c_upp = reshape(a_(1:ps,:,:,:),ps,nrows*ncols,nSamples);
            c_low = reshape(a_(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            for iAngle = 1:nAnglesH
                dW0_T = permute(genW.step(anglesW_NoDc,musW,iAngle),[2 1 3]);
                dU0_T = permute(genU.step(anglesU,musU,iAngle),[2 1 3]);
                for iblk = 1:(nrows*ncols)
                    dldz_upp_iblk = squeeze(dldz_upp(:,iblk,:));
                    dldz_low_iblk = squeeze(dldz_low(:,iblk,:));
                    c_upp_iblk = squeeze(c_upp(:,iblk,:));
                    c_low_iblk = squeeze(c_low(:,iblk,:));
                    d_upp_iblk = zeros(size(c_upp_iblk),'like',c_upp_iblk);
                    d_low_iblk = zeros(size(c_low_iblk),'like',c_low_iblk);
                    for iSample = 1:nSamples
                        d_upp_iblk(:,iSample) = dW0_T(1:ps,:,iblk)*c_upp_iblk(:,iSample);
                        d_low_iblk(:,iSample) = dU0_T(1:pa,:,iblk)*c_low_iblk(:,iSample);
                    end
                    dldw_(iAngle,iblk) = sum(dldz_upp_iblk.*d_upp_iblk,'all');
                    dldw_(nAnglesH+iAngle,iblk) = sum(dldz_low_iblk.*d_low_iblk,'all');
                end
            end
            expctddLdW = dldw_;
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunFinalRotation2dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols],...
                'NoDcLeakage',true,...
                'Name','V0~', ...
                'Device',device_);
            layer.Mus = mus;
            layer.Angles = [anglesW; anglesU];
            %expctdZ = layer.predict(X);
            
            % Actual values
            if usegpu
                X = gpuArray(X);
                mus = gpuArray(mus);
                anglesW = gpuArray(anglesW);
                anglesU = gpuArray(anglesU);
                dLdZ = gpuArray(dLdZ);
            end
            layer.Mus = mus;
            layer.Angles = [anglesW; anglesU];
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

