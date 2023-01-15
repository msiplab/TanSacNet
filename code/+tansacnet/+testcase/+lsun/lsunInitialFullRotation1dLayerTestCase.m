classdef lsunInitialFullRotation1dLayerTestCase < matlab.unittest.TestCase
    %LSUNINITIALFULLROTATION1DLAYERTESTCASE
    %
    %  TODO: フォーマット変更 nChs x 1 x nBlks x nSamples
    %
    %   コンポーネント別に入力(nComponents=1のみサポート):
    %      nChs x nSamples x nBlks
    %
    %   コンポーネント別に出力(nComponents=1のみサポート):
    %      nChs x nSamples x nBlks
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
            layer = lsunInitialFullRotation1dLayer(...
                'Stride',2,...
                'NumberOfBlocks',4);
            fprintf("\n --- Check layer for 1-D sequences ---\n");
            checkLayer(layer,[2 8 4],...
                'ObservationDimension',2,...
                'CheckCodegenCompatibility',true)
        end

    end
    
    methods (Test)
        
        function testConstructor(testCase, stride)
            
            % Expected values
            expctdName = 'V0';
            expctdDescription = "LSUN initial full rotation " ...
                + "(pt,pb) = (" ...
                + ceil(prod(stride)/2) + "," ...
                + floor(prod(stride)/2) + "), "  ...
                + "m = " + stride;
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunInitialFullRotation1dLayer(...
                'Stride',stride,...
                'Name',expctdName);
            
            % Actual values
            actualName = layer.Name;
            actualDescription = layer.Description;
            
            % Evaluation
            testCase.verifyEqual(actualName,expctdName);
            testCase.verifyEqual(actualDescription,expctdDescription);
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
            nDecs = stride;
            nChsTotal = nDecs;
            % nDecs x nSamples x nBlks
            %X = randn(nrows,ncols,nDecs,nSamples,datatype);
            X = randn(nDecs,nSamples,nblks,datatype);
            if usegpu
                X = gpuArray(X);
            end
            
            % Expected values
            % nChs x nSamples x nBlks
            V0 = repmat(eye(nChsTotal,datatype),[1 1 nblks]);
            %expctdZ = zeros(nrows,ncols,nChsTotal,nSamples,datatype);
            expctdZ = zeros(nChsTotal,nSamples,nblks,datatype);
            for iSample=1:nSamples
                % Perumation in each block
                Ai = permute(X(:,iSample,:),[1 3 2]);
                Yi = reshape(Ai,nDecs,nblks);
                %
                for iblk = 1:nblks
                    Yi(:,iblk) = V0(:,:,iblk)*Yi(:,iblk);
                end
                expctdZ(:,iSample,:) = ipermute(Yi,[1 3 2]);
            end
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunInitialFullRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
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

        function testPredictWithRandomAngles(testCase, ...
                usegpu, stride, nblks, datatype)
            if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end            
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            import tansacnet.utility.*
            gen = OrthonormalMatrixGenerationSystem();
            
            % Parameters
            nSamples = 8;
            nDecs = stride;
            nChsTotal = nDecs;
            % nDecs x nSamples x nBlks
            %X = randn(nrows,ncols,nDecs,nSamples,datatype);
            X = randn(nDecs,nSamples,nblks,datatype);
            nAngles = (nChsTotal-1)*nChsTotal/2;
            angles = randn(nAngles,nblks);
            if usegpu
                X = gpuArray(X);
                angles = gpuArray(angles);
            end            

            % Expected values
            % nChs x nSamples x nBlks
            V0 = gen.step(angles,1);
            %expctdZ = zeros(nrows,ncols,nChsTotal,nSamples,datatype);
            expctdZ = zeros(nChsTotal,nSamples,nblks,datatype);
            for iSample=1:nSamples
                % Perumation in each block
                Ai = permute(X(:,iSample,:),[1 3 2]);
                Yi = reshape(Ai,nDecs,nblks);
                %
                for iblk = 1:nblks
                    Yi(:,iblk) = V0(:,:,iblk)*Yi(:,iblk);
                end
                expctdZ(:,iSample,:) = ipermute(Yi,[1 3 2]);
            end
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunInitialFullRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
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
        
        %{
        function testPredictWithRandomAnglesNoDcLeackage(testCase, ...
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
            nDecs = prod(stride);
            nChsTotal = nDecs;
            % nDecs x nRows x nCols x nDecs x nSamples
            %X = randn(nrows,ncols,nDecs,nSamples,datatype);
            X = randn(nDecs,nrows,ncols,nSamples,datatype);
            angles = randn((nChsTotal-2)*nChsTotal/4,nrows*ncols);
            if usegpu
                X = gpuArray(X);
                angles = gpuArray(angles);
            end   

            % Expected values
            % nChs x nRows x nCols x nSamples
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            anglesNoDc = angles;
            anglesNoDc(1:pt-1,:)=zeros(pt-1,nrows*ncols);
            musW = mus*ones(pt,nrows*ncols);
            musW(1,1:end) = 1;
            musU = mus*ones(pb,nrows*ncols);
            W0 = genW.step(anglesNoDc(1:size(angles,1)/2,:),musW);
            U0 = genU.step(anglesNoDc(size(angles,1)/2+1:end,:),musU);
            %expctdZ = zeros(nrows,ncols,nChsTotal,nSamples,datatype);
            expctdZ = zeros(nChsTotal,nrows,ncols,nSamples,datatype);
            Y  = zeros(nChsTotal,nrows,ncols,datatype);
            for iSample=1:nSamples
                % Perumation in each block
                Ai = X(:,:,:,iSample); %permute(X(:,:,:,iSample),[3 1 2]);
                Yi = reshape(Ai,nDecs,nrows*ncols);
                %
                Ys = Yi(1:pt,:);
                Ya = Yi(pt+1:end,:);
                for iblk = 1:(nrows*ncols)
                    Ys(:,iblk) = W0(:,1:pt,iblk)*Ys(:,iblk);
                    Ya(:,iblk) = U0(:,1:pb,iblk)*Ya(:,iblk);
                end
                Y(1:pt,:,:) = reshape(Ys,pt,nrows,ncols);
                Y(pt+1:end,:,:) = reshape(Ya,pb,nrows,ncols);                
                expctdZ(:,:,:,iSample) = Y; %ipermute(Y,[3 1 2]);
            end
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunInitialFullRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols],...
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
        %}

        function testBackward(testCase, ...
                usegpu, stride, nblks, datatype)
            
            if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-4,single(1e-4));
            import tansacnet.utility.*
            gen = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on');
            
            % Parameters
            nSamples = 8;
            nDecs = stride;
            nChsTotal = nDecs;
            nAngles = (nChsTotal-1)*nChsTotal/2;
            angles = zeros(nAngles,nblks);
            mus_ = 1;
            
            % nDecs x nSamples x nBlks
            %X = randn(nrows,ncols,nDecs,nSamples,datatype);
            X = randn(nDecs,nSamples,nblks,datatype);
            dLdZ = randn(nDecs,nSamples,nblks,datatype);                        
            if usegpu
                X = gpuArray(X);
                angles = gpuArray(angles);
                dLdZ = gpuArray(dLdZ);
            end

            % Expected values
            % nDecs x nSamples x nBlks
            % dLdX = dZdX x dLdZ
            V0T = permute(gen.step(angles,mus_,0),[2 1 3]);
            Y = permute(dLdZ,[1 3 2]);
            for iSample = 1:nSamples
                for iblk = 1:nblks
                    Y(:,iblk,iSample) = V0T(:,:,iblk)*Y(:,iblk,iSample); 
                end
            end
            expctddLdX = ipermute(reshape(Y,nDecs,nblks,nSamples),[1 3 2]);
            
            % dLdWi = <dLdZ,(dVdWi)X>
            dldw_ = zeros(nAngles,nblks,1,datatype);
            dldz_ = permute(dLdZ,[1 3 2]);
            % (dVdWi)X
            c_ = permute(X,[1 3 2]);
            for iAngle = 1:nAngles
                dV0 = gen.step(angles,mus_,iAngle);
                for iblk = 1:nblks
                    dldz_iblk = squeeze(dldz_(:,iblk,:));
                    c_iblk = squeeze(c_(:,iblk,:));
                    d_iblk = zeros(size(c_iblk),'like',c_iblk);
                    for iSample = 1:nSamples
                        d_iblk(:,iSample) = dV0(:,:,iblk)*c_iblk(:,iSample);
                    end
                    dldw_(iAngle,iblk) = sum(dldz_iblk.*d_iblk,'all');
                end
            end
            expctddLdW = dldw_;

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunInitialFullRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
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

        function testBackwardWithRandomAngles(testCase, ...
                usegpu, stride, nblks, datatype)

            if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-4,single(1e-3));
            import tansacnet.utility.*
            gen = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on');
            
            % Parameters
            nSamples = 8;
            nDecs = stride;
            nChsTotal = nDecs;
            nAngles = (nChsTotal-1)*nChsTotal/2;
            angles = randn(nAngles,nblks,datatype);
            mus_ = 1;
            
            % nDecs x nRows x nCols x nSamples
            %X = randn(nrows,ncols,nDecs,nSamples,datatype);
            %dLdZ = randn(nrows,ncols,sum(stride),nSamples,datatype);
            X = randn(nDecs,nSamples,nblks,datatype);
            dLdZ = randn(nDecs,nSamples,nblks,datatype);
            if usegpu
                X = gpuArray(X);
                angles = gpuArray(angles);
                dLdZ = gpuArray(dLdZ);
            end

            % Expected values
            % nDecs x nSamples x nBlks
            % dLdX = dZdX x dLdZ
            V0T = permute(gen.step(angles,mus_,0),[2 1 3]);
            Y = permute(dLdZ,[1 3 2]);
            for iSample = 1:nSamples
                for iblk = 1:nblks
                    Y(:,iblk,iSample) = V0T(:,:,iblk)*Y(:,iblk,iSample); 
                end
            end
            expctddLdX = ipermute(reshape(Y,nDecs,nblks,nSamples),[1 3 2]);
                        
            % dLdWi = <dLdZ,(dVdWi)X>
            dldw_ = zeros(nAngles,nblks,1,datatype);
            dldz_ = permute(dLdZ,[1 3 2]);
            % (dVdWi)X
            c_ = permute(X,[1 3 2]);
            for iAngle = 1:nAngles
                dV0 = gen.step(angles,mus_,iAngle);
                for iblk = 1:nblks
                    dldz_iblk = squeeze(dldz_(:,iblk,:));
                    c_iblk = squeeze(c_(:,iblk,:));
                    d_iblk = zeros(size(c_iblk),'like',c_iblk);
                    for iSample = 1:nSamples
                        d_iblk(:,iSample) = dV0(:,:,iblk)*c_iblk(:,iSample);
                    end
                    dldw_(iAngle,iblk) = sum(dldz_iblk.*d_iblk,'all');
                end
            end
            expctddLdW = dldw_;

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunInitialFullRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'Name','V0');
            layer.Mus = mus_;
            layer.Angles = angles; 
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
        %{
        function testBackwardWithRandomAnglesNoDcLeackage(testCase, ...
                usegpu, stride, nrows, ncols, mus, datatype)
            % TODO:
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
            anglesW = randn(nAnglesH,nrows*ncols,datatype);
            anglesU = randn(nAnglesH,nrows*ncols,datatype);
            
            % nDecs x nRows x nCols x nSamples
            %X = randn(nrows,ncols,nDecs,nSamples,datatype);
            %dLdZ = randn(nrows,ncols,sum(stride),nSamples,datatype);
            X = randn(nDecs,nrows,ncols,nSamples,datatype);
            dLdZ = randn(nDecs,nrows,ncols,nSamples,datatype);            
            if usegpu
                X = gpuArray(X);
                anglesW = gpuArray(anglesW);
                anglesU = gpuArray(anglesU);
                dLdZ = gpuArray(dLdZ);
            end

            % Expected values
            % nDecs x nRows x nCols x nSamples
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            
            % dLdX = dZdX x dLdZ
            anglesW_NoDc = anglesW;
            anglesW_NoDc(1:pt-1,:)=zeros(pt-1,nrows*ncols);
            musW = mus*ones(pt,nrows*ncols);
            musW(1,:) = ones(1,nrows*ncols);
            musU = mus*ones(pb,nrows*ncols);            
            W0T = permute(genW.step(anglesW_NoDc,musW,0),[2 1 3]);
            U0T = permute(genU.step(anglesU,musU,0),[2 1 3]);
            Y = dLdZ; %permute(dLdZ,[3 1 2 4]);
            Ys = reshape(Y(1:pt,:,:,:),pt,nrows*ncols,nSamples);
            Ya = reshape(Y(pt+1:pt+pb,:,:,:),pb,nrows*ncols,nSamples);
            for iSample = 1:nSamples
                for iblk = 1:(nrows*ncols)
                    Ys(:,iblk,iSample) = W0T(1:pt,:,iblk)*Ys(:,iblk,iSample);
                    Ya(:,iblk,iSample) = U0T(1:pb,:,iblk)*Ya(:,iblk,iSample);
                end
            end
            Zsa = cat(1,Ys,Ya);
            %expctddLdX = ipermute(reshape(Zsa,nDecs,nrows,ncols,nSamples),...
            %    [3 1 2 4]);
            expctddLdX = reshape(Zsa,nDecs,nrows,ncols,nSamples);
            
            % dLdWi = <dLdZ,(dVdWi)X>
            dldw_ = zeros(2*nAnglesH,nrows*ncols,datatype);
            dldz_ = dLdZ; %permute(dLdZ,[3 1 2 4]);
            dldz_upp = reshape(dldz_(1:pt,:,:,:),pt,nrows*ncols,nSamples);
            dldz_low = reshape(dldz_(pt+1:pt+pb,:,:,:),pb,nrows*ncols,nSamples);
            % (dVdWi)X
            a_ = X; %permute(X,[3 1 2 4]);
            c_upp = reshape(a_(1:pt,:,:,:),pt,nrows*ncols,nSamples);
            c_low = reshape(a_(pt+1:nDecs,:,:,:),pb,nrows*ncols,nSamples);
            for iAngle = 1:nAnglesH
                dW0 = genW.step(anglesW_NoDc,musW,iAngle);
                dU0 = genU.step(anglesU,musU,iAngle);
                for iblk = 1:(nrows*ncols)
                    dldz_upp_iblk = squeeze(dldz_upp(:,iblk,:));
                    dldz_low_iblk = squeeze(dldz_low(:,iblk,:));
                    c_upp_iblk = squeeze(c_upp(:,iblk,:));
                    c_low_iblk = squeeze(c_low(:,iblk,:));                    
                    d_upp_iblk = zeros(size(c_upp_iblk),'like',c_upp_iblk);
                    d_low_iblk = zeros(size(c_low_iblk),'like',c_low_iblk);
                    for iSample = 1:nSamples
                        d_upp_iblk(:,iSample) = dW0(:,1:pt,iblk)*c_upp_iblk(:,iSample);
                        d_low_iblk(:,iSample) = dU0(:,1:pb,iblk)*c_low_iblk(:,iSample);
                    end
                    dldw_(iAngle,iblk) = sum(dldz_upp_iblk.*d_upp_iblk,'all');
                    dldw_(nAnglesH+iAngle,iblk) = sum(dldz_low_iblk.*d_low_iblk,'all');
                end
            end
            expctddLdW = dldw_;
        
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunInitialFullRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols],...
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
    %}
    end
    
end

