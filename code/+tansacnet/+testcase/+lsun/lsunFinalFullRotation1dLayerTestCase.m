classdef lsunFinalFullRotation1dLayerTestCase < matlab.unittest.TestCase
    %LSUNFINALFULLROTATION1DLAYERTESTCASE
    %
    %   コンポーネント別に入力(nComponents=1のみサポート):
    %      nChs x 1 x nBlks x nSamples
    %
    %   コンポーネント別に出力(nComponents=1のみサポート):
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
        mus = { -1, 1 };
        datatype = { 'single', 'double' };
        nblks = struct('small', 2,'medium', 4, 'large', 8);
        usegpu = struct( 'true', true, 'false', false);           
    end
    
    methods (TestClassTeardown)

        function finalCheck(~)
            import tansacnet.lsun.*
            layer = lsunFinalFullRotation1dLayer(...
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
            expctdName = 'V0~';
            expctdDescription = "LSUN final full rotation " ...
                + "(pt,pb) = (" ...
                + ceil(prod(stride)/2) + "," ...
                + floor(prod(stride)/2) + "), " ...               
                + "m = " + stride;
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunFinalFullRotation1dLayer(...
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
            layer = lsunFinalFullRotation1dLayer(...
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

        function testinitialize(testCase,nblks,stride)

            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));

            seqlen = 32;
            datatype = 'double';
            nChsTotal = stride;
            nSamples = 8;
            nAngles = (nChsTotal-1)*nChsTotal/2;
            
            X = randn(nChsTotal,1,nblks,nSamples,datatype);
            layoutsize = [prod(stride) 1 seqlen./stride];
            anglesize = [nAngles prod(nblks)];
            expctdangles = zeros(anglesize,datatype);

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunInitialFullRotation1dLayer(...
                'Stride',stride, ...
                'NumberOfBlocks',nblks);
                

            layout = networkDataLayout(layoutsize,'SSC');
            layer_ = initialize(layer,layout);

            % Actual values
            actualZ = layer_.predict(X);
            actualangles = layer_.Angles;

            % Evaluation
            testCase.verifyInstanceOf(actualZ,datatype);
            testCase.verifyThat(actualangles,...
                IsEqualTo(expctdangles,'Within',tolObj));
        end

        function testPredict(testCase, ...
                usegpu, stride, nblks, datatype)
            
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
            % nChs x 1 x nBlks x nSamples
            X = randn(nChsTotal,1,nblks,nSamples,datatype);
            if usegpu
                X = gpuArray(X);
            end

            % Expected values        
            % nChsTotal x 1 x nBlks x nSamples
            V0T = repmat(eye(nChsTotal,datatype),[1 1 nblks]);
            Y = X;
            for iSample=1:nSamples
                for iblk = 1:nblks
                    Y(:,:,iblk,iSample) = V0T(:,:,iblk)*Y(:,:,iblk,iSample); 
                end
            end
            expctdZ = Y;
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunFinalFullRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
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

        function testPredictWithDeviceAndDType(testCase, ...
                usegpu, stride, nblks, datatype)

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
            % nChs x 1 x nBlks x nSamples
            X = randn(nChsTotal,1,nblks,nSamples,datatype);
            if expctdDevice == "cuda"
                X = gpuArray(X);
            end
            
            % Expected values        
            % nChsTotal x 1 x nBlks x nSamples
            V0T = repmat(eye(nChsTotal,datatype),[1 1 nblks]);
            Y = X;
            for iSample=1:nSamples
                for iblk = 1:nblks
                    Y(:,:,iblk,iSample) = V0T(:,:,iblk)*Y(:,:,iblk,iSample); 
                end
            end
            expctdZ = Y;
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunFinalFullRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
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

        function testPredictWithRandomAngles(testCase, ...
                usegpu, stride, nblks, datatype)
            
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
            gen = OrthonormalMatrixGenerationSystem('Device','cpu');
            
            % Parameters
            nSamples = 8;
            nChsTotal = stride;
            % nChs x 1 x nBlks x nSamples
            X = randn(nChsTotal,1,nblks,nSamples,datatype);
            angles = randn((nChsTotal-1)*nChsTotal/2,nblks);            

            % Expected values
            % nChsTotal x nSamples x nBlks
            V0T = permute(gen.step(angles,1),[2 1 3]);
            Y = X;
            for iSample=1:nSamples
                for iblk = 1:nblks
                    Y(:,:,iblk,iSample) = V0T(:,:,iblk)*Y(:,:,iblk,iSample);
                end
            end
            expctdZ = Y;
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunFinalFullRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
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
        
%{               
        function testPredictGrayscaleWithRandomAnglesNoDcLeackage(testCase, ...
                usegpu, stride, nrows, ncols, mus, datatype)
            % TODO:
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
            nChsTotal = nChsTotal;
            % nChs x nRows x nCols x nSamples
            %X = randn(nrows,ncols,sum(stride),nSamples,datatype);
            X = randn(nChsTotal,nrows,ncols,nSamples,datatype);
            angles = randn((nChsTotal-2)*nChsTotal/4,nrows*ncols);
            if usegpu
                X = gpuArray(X);
                angles = gpuArray(angles);
            end

            % Expected values
            % nChsTotal x nRows x nCols x nSamples
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            anglesNoDc = angles;
            anglesNoDc(1:pt-1,:)=zeros(pt-1,nrows*ncols);
            musW = mus*ones(pt,nrows*ncols);
            musW(1,:) = 1;
            musU = mus*ones(pb,nrows*ncols);
            W0T = permute(genW.step(anglesNoDc(1:size(angles,1)/2,:),musW),[2 1 3]);
            U0T = permute(genU.step(anglesNoDc(size(angles,1)/2+1:end,:),musU),[2 1 3]);
            Y = X; %permute(X,[3 1 2 4]);
            Yt = reshape(Y(1:pt,:,:,:),pt,nrows*ncols,nSamples);
            Yb = reshape(Y(pt+1:pt+pb,:,:,:),pb,nrows*ncols,nSamples);
            for iSample=1:nSamples
                for iblk = 1:(nrows*ncols)
                    Yt(:,iblk,iSample) = W0T(1:pt,:,iblk)*Yt(:,iblk,iSample);
                    Yb(:,iblk,iSample) = U0T(1:pb,:,iblk)*Yb(:,iblk,iSample);
                end
            end
            Zsa = cat(1,Yt,Yb);
            %expctdZ = ipermute(reshape(Zsa,nChsTotal,nrows,ncols,nSamples),...
            %    [3 1 2 4]);
            expctdZ = reshape(Zsa,nChsTotal,nrows,ncols,nSamples);

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunFinalFullRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols],...
                'NoDcLeakage',true,...
                'Name','V0~');
            
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
%}
          
        function testBackward(testCase, ...
                usegpu, stride, nblks, datatype)
            
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
            gen = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on',...
                'Device','cpu',...
                'DType',datatype);
            
            % Parameters
            nSamples = 8;
            nChsTotal = stride;
            nAngles = (nChsTotal-1)*nChsTotal/2;
            angles = zeros(nAngles,nblks,datatype);            
            mus_ = cast(1,datatype);
            
            % nChsTotal x nSamples x nBlks
            X = randn(nChsTotal,1,nblks,nSamples,datatype);            
            dLdZ = randn(nChsTotal,1,nblks,nSamples,datatype);            

            % Expected values
            % dLdX = dZdX x dLdZ
            V0 = gen.step(angles,mus_,0);
            expctddLdX = zeros(nChsTotal,1,nblks,nSamples,datatype);
            for iSample=1:nSamples
                % Perumation in each block
                Ai = dLdZ(:,:,:,iSample);
                Yi = reshape(Ai,nChsTotal,nblks);
                for iblk=1:nblks
                    Yi(:,iblk) = V0(:,:,iblk)*Yi(:,iblk);
                end
                expctddLdX(:,:,:,iSample) = Yi;
            end
            
            % dLdWi = <dLdZ,(dVdWi)X>
            dldw_ = zeros(nAngles,nblks,datatype);
            dldz_ = dLdZ;
            % (dVdWi)X
            c_ = X;
            for iAngle = 1:nAngles
                dV0_T = permute(gen.step(angles,mus_,iAngle),[2 1 3]);
                for iblk=1:nblks
                    dldz_iblk = squeeze(dldz_(:,:,iblk,:));
                    c_iblk = squeeze(c_(:,:,iblk,:));
                    d_iblk = zeros(size(c_iblk),'like',c_iblk);
                    for iSample = 1:nSamples
                        d_iblk(:,iSample) = dV0_T(:,:,iblk)*c_iblk(:,iSample);
                    end
                    dldw_(iAngle,iblk) = sum(dldz_iblk.*d_iblk,'all');
                end
            end
            expctddLdW = dldw_;
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunFinalFullRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'Name','V0~', ...
                'Device',device_);            
            
            % Actual values
            if usegpu
                X = gpuArray(X);
                %angles = gpuArray(angles);
                mus_ = gpuArray(mus_);
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

        function testBackwardWithDeviceAndDType(testCase, ...
                usegpu, stride, nblks, datatype)
            
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
            gen = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on',...
                'Device','cpu',...
                'DType',datatype);
            
            % Parameters
            nSamples = 8;
            nChsTotal = stride;
            nAngles = (nChsTotal-1)*nChsTotal/2;
            angles = zeros(nAngles,nblks,datatype);  
            mus_ = cast(1,datatype);
            
            % nChsTotal x 1 x nBlks x nSamples
            X = randn(nChsTotal,1,nblks,nSamples,datatype);
            dLdZ = randn(nChsTotal,1,nblks,nSamples,datatype); 

             % Expected values
            % dLdX = dZdX x dLdZ
            V0 = gen.step(angles,mus_,0);
            expctddLdX = zeros(nChsTotal,1,nblks,nSamples,datatype);
            for iSample=1:nSamples
                % Perumation in each block
                Ai = dLdZ(:,:,:,iSample);
                Yi = reshape(Ai,nChsTotal,nblks);
                for iblk=1:nblks
                    Yi(:,iblk) = V0(:,:,iblk)*Yi(:,iblk);
                end
                expctddLdX(:,:,:,iSample) = Yi;
            end
            
            % dLdWi = <dLdZ,(dVdWi)X>
            dldw_ = zeros(nAngles,nblks,datatype);
            dldz_ = dLdZ;
            % (dVdWi)X
            c_ = X;
            for iAngle = 1:nAngles
                dV0_T = permute(gen.step(angles,mus_,iAngle),[2 1 3]);
                for iblk=1:nblks
                    dldz_iblk = squeeze(dldz_(:,:,iblk,:));
                    c_iblk = squeeze(c_(:,:,iblk,:));
                    d_iblk = zeros(size(c_iblk),'like',c_iblk);
                    for iSample = 1:nSamples
                        d_iblk(:,iSample) = dV0_T(:,:,iblk)*c_iblk(:,iSample);
                    end
                    dldw_(iAngle,iblk) = sum(dldz_iblk.*d_iblk,'all');
                end
            end
            expctddLdW = dldw_;


            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunFinalFullRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'Name','V0~',...
                'Device',expctdDevice,...
                'DType',expctdDType);            
            %expctdZ = layer.predict(X);
            
            % Actual values
            if usegpu
                X = gpuArray(X);
                %angles = gpuArray(angles);
                mus_ = gpuArray(mus_);
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
       
        function testBackwardWithRandomAngles(testCase, ...
                usegpu, stride, nblks, datatype)
            
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
            gen = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on',...
                'Device','cpu',...
                'DType',datatype);
            
            % Parameters
            nSamples = 8;
            nChsTotal = stride;
            nAngles = (nChsTotal-1)*nChsTotal/2;
            mus_ = cast(1,datatype);

            % nChsTotal x nSamples x nBlks
            X = randn(nChsTotal,1,nblks,nSamples,datatype);            
            dLdZ = randn(nChsTotal,1,nblks,nSamples,datatype);
            angles = randn(nAngles,nblks,datatype); 

            % Expected values
            % dLdX = dZdX x dLdZ
            V0 = gen.step(angles,mus_,0);
            expctddLdX = zeros(nChsTotal,1,nblks,nSamples,datatype);
            for iSample=1:nSamples
                % Perumation in each block
                Ai = dLdZ(:,:,:,iSample);
                Yi = reshape(Ai,nChsTotal,nblks);
                for iblk=1:nblks
                    Yi(:,iblk) = V0(:,:,iblk)*Yi(:,iblk);
                end
                expctddLdX(:,:,:,iSample) = Yi;
            end
            
            % dLdWi = <dLdZ,(dVdWi)X>
            dldw_ = zeros(nAngles,nblks,datatype);
            dldz_ = dLdZ;
            % (dVdWi)X
            c_ = X;
            for iAngle = 1:nAngles
                dV0_T = permute(gen.step(angles,mus_,iAngle),[2 1 3]);
                for iblk = 1:nblks
                    dldz_iblk = squeeze(dldz_(:,:,iblk,:));
                    c_iblk = squeeze(c_(:,:,iblk,:));
                    d_iblk = zeros(size(c_iblk),'like',c_iblk);
                    for iSample = 1:nSamples
                        d_iblk(:,iSample) = dV0_T(:,:,iblk)*c_iblk(:,iSample);
                    end
                    dldw_(iAngle,iblk) = sum(dldz_iblk.*d_iblk,'all');
                end
            end
            expctddLdW = dldw_;
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunFinalFullRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'Name','V0~', ...
                'Device',device_);            
            %expctdZ = layer.predict(X);
            
            % Actual values
            if usegpu
                X = gpuArray(X);
                %angles = gpuArray(angles);
                mus_ = gpuArray(mus_);
                dLdZ = gpuArray(dLdZ);
            end
            layer.Mus = mus_;
            layer.Angles = angles;
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
%{
        function testBackwardWithRandomAnglesNoDcLeackage(testCase, ...
                usegpu, stride, nrows, ncols, mus, datatype)
            
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
            nChsTotal = prod(stride);
            nChsTotal = nChsTotal;
            nAnglesH = (nChsTotal-2)*nChsTotal/8;
            anglesW = randn(nAnglesH,nrows*ncols,datatype);
            anglesU = randn(nAnglesH,nrows*ncols,datatype);
            
            % nChsTotal x nRows x nCols x nSamples
            %X = randn(nrows,ncols,sum(stride),nSamples,datatype);
            %dLdZ = randn(nrows,ncols,nChsTotal,nSamples,datatype);
            X = randn(nChsTotal,nrows,ncols,nSamples,datatype);
            dLdZ = randn(nChsTotal,nrows,ncols,nSamples,datatype);            
            if usegpu
                X = gpuArray(X);
                anglesW = gpuArray(anglesW);
                anglesU = gpuArray(anglesU);
                dLdZ = gpuArray(dLdZ);
            end

            % Expected values
            % nChs x nRows x nCols x nSamples
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            
            % dLdX = dZdX x dLdZ
            anglesW_NoDc = anglesW;
            anglesW_NoDc(1:pt-1,:)=zeros(pt-1,nrows*ncols);
            musW = mus*ones(pt,nrows*ncols);
            musW(1,:) = 1;
            musU = mus*ones(pb,nrows*ncols);
            W0 = genW.step(anglesW_NoDc,musW,0);
            U0 = genU.step(anglesU,musU,0);
            %expctddLdX = zeros(nrows,ncols,nChsTotal,nSamples,datatype);
            expctddLdX = zeros(nChsTotal,nrows,ncols,nSamples,datatype);
            Y  = zeros(nChsTotal,nrows,ncols,datatype);
            for iSample=1:nSamples
                % Perumation in each block
                %Ai = permute(dLdZ(:,:,:,iSample),[3 1 2]);
                Ai = dLdZ(:,:,:,iSample);
                Yi = reshape(Ai,nChsTotal,nrows,ncols);
                %
                Yt = Yi(1:pt,:);
                Yb = Yi(pt+1:end,:);
                for iblk=1:(nrows*ncols)
                    Yt(:,iblk) = W0(:,1:pt,iblk)*Yt(:,iblk);
                    Yb(:,iblk) = U0(:,1:pb,iblk)*Yb(:,iblk);
                end
                Y(1:pt,:,:) = reshape(Yt,pt,nrows,ncols);
                Y(pt+1:end,:,:) = reshape(Yb,pb,nrows,ncols);
                expctddLdX(:,:,:,iSample) = Y; %ipermute(Y,[3 1 2]);
            end
            
            % dLdWi = <dLdZ,(dVdWi)X>
            dldw_ = zeros(2*nAnglesH,nrows*ncols,datatype);
            dldz_ = dLdZ; %permute(dLdZ,[3 1 2 4]);
            dldz_upp = reshape(dldz_(1:pt,:,:,:),pt,nrows*ncols,nSamples);
            dldz_low = reshape(dldz_(pt+1:nChsTotal,:,:,:),pb,nrows*ncols,nSamples);
            % (dVdWi)X
            a_ = X; %permute(X,[3 1 2 4]);
            c_upp = reshape(a_(1:pt,:,:,:),pt,nrows*ncols,nSamples);
            c_low = reshape(a_(pt+1:pt+pb,:,:,:),pb,nrows*ncols,nSamples);
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
                        d_upp_iblk(:,iSample) = dW0_T(1:pt,:,iblk)*c_upp_iblk(:,iSample);
                        d_low_iblk(:,iSample) = dU0_T(1:pb,:,iblk)*c_low_iblk(:,iSample);
                    end
                    dldw_(iAngle,iblk) = sum(dldz_upp_iblk.*d_upp_iblk,'all');
                    dldw_(nAnglesH+iAngle,iblk) = sum(dldz_low_iblk.*d_low_iblk,'all');
                end
            end
            expctddLdW = dldw_;
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunFinalFullRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols],...
                'NoDcLeakage',true,...
                'Name','V0~');
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
%}
    end

end

