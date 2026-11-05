classdef salsunIntermediateRotation2dLayerTestCase < matlab.unittest.TestCase
    %SALSUNINTERMEDIATEROTATION2DLAYERTESTCASE
    %
    %   Data-path input  'x'     : nChsTotal x nRows x nCols x nSamples
    %
    %   Control-path input 'theta': nAngles x (nRows*nCols) x nSamples
    %                               where nAngles = (nChsTotal-2)*nChsTotal/4
    %                               (first half: anglesW, second half: anglesU)
    %
    %   Output                   : nChsTotal x nRows x nCols x nSamples
    %
    % Requirements: MATLAB R2022a
    %
    % Copyright (c) 2026, Motoyasu SUZUKI and Shogo MURAMATSU
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
        datatype = { 'single', 'double' };
        mus = { -1, 1 };
        nrows = struct('small', 2,'medium', 4, 'large', 8);
        ncols = struct('small', 2,'medium', 4, 'large', 8);
        usegpu = struct( 'true', true, 'false', false);
    end

    methods (TestClassTeardown)
        function finalCheck(~)
            import tansacnet.salsun.*
            stride_ = [2 2];
            nrows_ = 8;
            ncols_ = 8;
            nChsTotal = prod(stride_);
            nAngles = (nChsTotal-2)*nChsTotal/8;
            layer = salsunIntermediateRotation2dLayer(...
                'Stride',stride_,...
                'NumberOfBlocks',[nrows_ ncols_]);
            fprintf("\n --- Check layer for 2-D images (SA-LSUN) ---\n");
            checkLayer(layer,{[nChsTotal nrows_ ncols_],[nAngles nrows_*ncols_]},...
                'ObservationDimension',[4 3])
        end
    end

    methods (Test)

        function testConstructor(testCase, stride)

            % Expected values
            expctdName = 'Vn~';
            expctdMode = 'Synthesis';
            expctdDescription = "Synthesis SA-LSUN intermediate rotation " ...
                + "(ps,pa) = (" ...
                + ceil(prod(stride)/2) + "," + floor(prod(stride)/2) + ")";

            % Instantiation of target class
            import tansacnet.salsun.*
            layer = salsunIntermediateRotation2dLayer(...
                'Stride',stride,...
                'Name',expctdName);

            % Actual values
            actualName = layer.Name;
            actualMode = layer.Mode;
            actualDescription = layer.Description;
            actualInputNames = layer.InputNames;

            % Evaluation
            testCase.verifyEqual(actualName,expctdName);
            testCase.verifyEqual(actualMode,expctdMode);
            testCase.verifyEqual(actualDescription,expctdDescription);
            testCase.verifyEqual(actualInputNames,{'x','theta'});
        end

        function testConstructorWithDeviceAndDType(testCase, stride, usegpu, datatype)

            % Expected values
            expctdName = 'Vn~';
            expctdMode = 'Synthesis';

            device_ = ["cpu", "cuda"];
            expctdDevice = device_(usegpu+1);
            expctdDType = datatype;

            % Instantiation of target class
            import tansacnet.salsun.*
            layer = salsunIntermediateRotation2dLayer(...
                'Stride',stride,...
                'Name',expctdName,...
                'Device',expctdDevice,...
                'DType',expctdDType);

            % Actual values
            actualName = layer.Name;
            actualMode = layer.Mode;
            actualDevice = layer.Device;
            actualDType = layer.DType;

            % Evaluation
            testCase.verifyEqual(actualName,expctdName);
            testCase.verifyEqual(actualMode,expctdMode);
            testCase.verifyEqual(actualDevice,expctdDevice);
            testCase.verifyEqual(actualDType,expctdDType);
        end

        function testPredictGrayscaleWithZeroAngles(testCase, ...
                usegpu, stride, nrows, ncols, mus, datatype)

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
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            nBlks = nrows*ncols;
            nAngles = (nChsTotal-2)*nChsTotal/8;

            X = randn(nChsTotal,nrows,ncols,nSamples,datatype);
            Theta = zeros(nAngles,nBlks,nSamples,datatype);
            if usegpu
                X = gpuArray(X);
                Theta = gpuArray(Theta);
            end

            % Expected values: at Theta==0, Un == mus*eye(pa) (default
            % Mode is 'Synthesis', for which Un' is applied, but a scaled
            % identity is self-transpose).
            UnT = repmat(mus*eye(pa,datatype),[1 1 nBlks]);
            Y = X;
            Ya = reshape(Y(ps+1:ps+pa,:,:,:),pa,nBlks,nSamples);
            Za = zeros(size(Ya),'like',Ya);
            for iSample = 1:nSamples
                for iblk = 1:nBlks
                    Za(:,iblk,iSample) = UnT(:,:,iblk)*Ya(:,iblk,iSample);
                end
            end
            Y(ps+1:ps+pa,:,:,:) = reshape(Za,pa,nrows,ncols,nSamples);
            expctdZ = Y;

            % Instantiation of target class
            import tansacnet.salsun.*
            layer = salsunIntermediateRotation2dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols],...
                'Name','Vn~');

            % Actual values
            layer.Mus = mus;
            actualZ = layer.predict(X,Theta);

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
                usegpu, stride, nrows, ncols, mus, datatype)

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
            nChsTotal = prod(stride);
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            nBlks = nrows*ncols;
            nAngles = (nChsTotal-2)*nChsTotal/8;

            X = randn(nChsTotal,nrows,ncols,nSamples,datatype);
            Theta = zeros(nAngles,nBlks,nSamples,datatype);
            if usegpu
                X = gpuArray(X);
                Theta = gpuArray(Theta);
            end

            % Expected values
            UnT = repmat(mus*eye(pa,datatype),[1 1 nBlks]);
            Y = X;
            Ya = reshape(Y(ps+1:ps+pa,:,:,:),pa,nBlks,nSamples);
            Za = zeros(size(Ya),'like',Ya);
            for iSample = 1:nSamples
                for iblk = 1:nBlks
                    Za(:,iblk,iSample) = UnT(:,:,iblk)*Ya(:,iblk,iSample);
                end
            end
            Y(ps+1:ps+pa,:,:,:) = reshape(Za,pa,nrows,ncols,nSamples);
            expctdZ = Y;

            % Instantiation of target class
            import tansacnet.salsun.*
            layer = salsunIntermediateRotation2dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols],...
                'Name','Vn~',...
                'Device',expctdDevice,...
                'DType',expctdDType);

            % Actual values
            layer.Mus = mus;
            actualZ = layer.predict(X,Theta);
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
            genU = OrthonormalMatrixGenerationSystem('Device','cpu');

            % Parameters
            nSamples = 8;
            nChsTotal = prod(stride);
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            nBlks = nrows*ncols;
            nAngles = (nChsTotal-2)*nChsTotal/8;

            % nChsTotal x nRows x nCols x nSamples
            X = randn(nChsTotal,nrows,ncols,nSamples,datatype);
            % Angles vary block by block AND sample by sample (the
            % SA-LSUN feature under test).
            Theta = randn(nAngles,nBlks,nSamples);

            % Expected values (reference computed sample by sample)
            expctdZ = zeros(nChsTotal,nrows,ncols,nSamples,datatype);
            for iSample = 1:nSamples
                UnT = permute(genU.step(Theta(:,:,iSample),mus),[2 1 3]);
                Xi = reshape(X(:,:,:,iSample),nChsTotal,nBlks);
                Ya = Xi(ps+1:ps+pa,:);
                for iblk = 1:nBlks
                    Ya(:,iblk) = UnT(:,:,iblk)*Ya(:,iblk);
                end
                expctdZ(:,:,:,iSample) = reshape([Xi(1:ps,:);Ya],nChsTotal,nrows,ncols);
            end

            % Instantiation of target class
            import tansacnet.salsun.*
            layer = salsunIntermediateRotation2dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols],...
                'Name','Vn~',...
                'Device',device_);

            % Actual values
            if usegpu
                X = gpuArray(X);
                Theta = gpuArray(Theta);
                mus = gpuArray(mus);
            end
            layer.Mus = mus;
            actualZ = layer.predict(X,Theta);

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
            genU = OrthonormalMatrixGenerationSystem('Device','cpu');

            % Parameters
            nSamples = 8;
            nChsTotal = prod(stride);
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            nBlks = nrows*ncols;
            nAngles = (nChsTotal-2)*nChsTotal/8;

            X = randn(nChsTotal,nrows,ncols,nSamples,datatype);
            Theta = randn(nAngles,nBlks,nSamples);

            % Expected values
            expctdZ = zeros(nChsTotal,nrows,ncols,nSamples,datatype);
            for iSample = 1:nSamples
                Un = genU.step(Theta(:,:,iSample),mus);
                Xi = reshape(X(:,:,:,iSample),nChsTotal,nBlks);
                Ya = Xi(ps+1:ps+pa,:);
                for iblk = 1:nBlks
                    Ya(:,iblk) = Un(:,:,iblk)*Ya(:,iblk);
                end
                expctdZ(:,:,:,iSample) = reshape([Xi(1:ps,:);Ya],nChsTotal,nrows,ncols);
            end
            expctdDescription = "Analysis SA-LSUN intermediate rotation " ...
                + "(ps,pa) = (" ...
                + ps + "," + pa + ")";

            % Instantiation of target class
            import tansacnet.salsun.*
            layer = salsunIntermediateRotation2dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols],...
                'Name','Vn',...
                'Mode','Analysis',...
                'Device',device_);

            % Actual values
            if usegpu
                X = gpuArray(X);
                Theta = gpuArray(Theta);
            end
            layer.Mus = mus;
            actualZ = layer.predict(X,Theta);
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

        function testBackwardGrayscaleWithZeroAngles(testCase, ...
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
            genU = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on',...
                'Device','cpu',...
                'DType',datatype);

            % Parameters
            nSamples = 8;
            nChsTotal = prod(stride);
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            nBlks = nrows*ncols;
            nAngles = (nChsTotal-2)*nChsTotal/8;
            ang = zeros(nAngles,nBlks,datatype);
            mus_ = cast(mus,datatype);
            Theta = repmat(ang,[1 1 nSamples]);

            X = randn(nChsTotal,nrows,ncols,nSamples,datatype);
            dLdZ = randn(nChsTotal,nrows,ncols,nSamples,datatype);

            % Expected values (default Mode is 'Synthesis': forward
            % applies Un', so backward applies Un for dLdX)
            Un = genU.step(ang,mus_,0);
            expctddLdX = zeros(nChsTotal,nrows,ncols,nSamples,datatype);
            expctddLdTheta = zeros(nAngles,nBlks,nSamples,datatype);
            for iSample = 1:nSamples
                dLdZi = reshape(dLdZ(:,:,:,iSample),nChsTotal,nBlks);
                Ya = dLdZi(ps+1:ps+pa,:);
                for iblk = 1:nBlks
                    Ya(:,iblk) = Un(:,:,iblk)*Ya(:,iblk);
                end
                expctddLdX(:,:,:,iSample) = reshape([dLdZi(1:ps,:);Ya],nChsTotal,nrows,ncols);

                % dLdTheta_i = <dLdZ,(dVdTheta_i)X>
                Xi = reshape(X(:,:,:,iSample),nChsTotal,nBlks);
                c_low = Xi(ps+1:ps+pa,:);
                dldz_low = dLdZi(ps+1:ps+pa,:);
                for iAngle = 1:nAngles
                    dUn_T = permute(genU.step(ang,mus_,iAngle),[2 1 3]);
                    for iblk = 1:nBlks
                        d_low = dUn_T(:,:,iblk)*c_low(:,iblk);
                        expctddLdTheta(iAngle,iblk,iSample) = sum(dldz_low(:,iblk).*d_low,'all');
                    end
                end
            end

            % Instantiation of target class
            import tansacnet.salsun.*
            layer = salsunIntermediateRotation2dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols],...
                'Name','Vn~',...
                'Device',device_);

            % Actual values
            if usegpu
                X = gpuArray(X);
                dLdZ = gpuArray(dLdZ);
                Theta = gpuArray(Theta);
                mus_ = gpuArray(mus_);
            end
            layer.Mus = mus_;
            [actualdLdX,actualdLdTheta] = layer.backward(X,Theta,[],dLdZ,[]);

            % Evaluation
            if usegpu
                testCase.verifyClass(actualdLdX,'gpuArray')
                actualdLdX = gather(actualdLdX);
                expctddLdX = gather(expctddLdX);
                testCase.verifyClass(actualdLdTheta,'gpuArray')
                actualdLdTheta = gather(actualdLdTheta);
                expctddLdTheta = gather(expctddLdTheta);
            end
            testCase.verifyInstanceOf(actualdLdX,datatype);
            testCase.verifyInstanceOf(actualdLdTheta,datatype);
            testCase.verifyThat(actualdLdX,...
                IsEqualTo(expctddLdX,'Within',tolObj));
            testCase.verifyThat(actualdLdTheta,...
                IsEqualTo(expctddLdTheta,'Within',tolObj));
        end

        function testBackwardGrayscaleWithDeviceAndDType(testCase, ...
                usegpu, stride, nrows, ncols, mus, datatype)

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
            genU = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on',...
                'Device','cpu',...
                'DType',datatype);

            % Parameters
            nSamples = 8;
            nChsTotal = prod(stride);
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            nBlks = nrows*ncols;
            nAngles = (nChsTotal-2)*nChsTotal/8;
            ang = zeros(nAngles,nBlks,datatype);
            mus_ = cast(mus,datatype);
            Theta = repmat(ang,[1 1 nSamples]);

            X = randn(nChsTotal,nrows,ncols,nSamples,datatype);
            dLdZ = randn(nChsTotal,nrows,ncols,nSamples,datatype);

            % Expected values
            Un = genU.step(ang,mus_,0);
            expctddLdX = zeros(nChsTotal,nrows,ncols,nSamples,datatype);
            expctddLdTheta = zeros(nAngles,nBlks,nSamples,datatype);
            for iSample = 1:nSamples
                dLdZi = reshape(dLdZ(:,:,:,iSample),nChsTotal,nBlks);
                Ya = dLdZi(ps+1:ps+pa,:);
                for iblk = 1:nBlks
                    Ya(:,iblk) = Un(:,:,iblk)*Ya(:,iblk);
                end
                expctddLdX(:,:,:,iSample) = reshape([dLdZi(1:ps,:);Ya],nChsTotal,nrows,ncols);

                Xi = reshape(X(:,:,:,iSample),nChsTotal,nBlks);
                c_low = Xi(ps+1:ps+pa,:);
                dldz_low = dLdZi(ps+1:ps+pa,:);
                for iAngle = 1:nAngles
                    dUn_T = permute(genU.step(ang,mus_,iAngle),[2 1 3]);
                    for iblk = 1:nBlks
                        d_low = dUn_T(:,:,iblk)*c_low(:,iblk);
                        expctddLdTheta(iAngle,iblk,iSample) = sum(dldz_low(:,iblk).*d_low,'all');
                    end
                end
            end

            % Instantiation of target class
            import tansacnet.salsun.*
            layer = salsunIntermediateRotation2dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols],...
                'Name','Vn~',...
                'Device',expctdDevice,...
                'DType',expctdDType);

            % Actual values
            if usegpu
                X = gpuArray(X);
                dLdZ = gpuArray(dLdZ);
                Theta = gpuArray(Theta);
                mus_ = gpuArray(mus_);
            end
            layer.Mus = mus_;
            [actualdLdX,actualdLdTheta] = layer.backward(X,Theta,[],dLdZ,[]);
            actualDevice = layer.Device;

            % Evaluation
            testCase.verifyEqual(actualDevice,expctdDevice);
            if actualDevice == "cuda"
                testCase.verifyClass(actualdLdX,'gpuArray')
                actualdLdX = gather(actualdLdX);
                expctddLdX = gather(expctddLdX);
                testCase.verifyClass(actualdLdTheta,'gpuArray')
                actualdLdTheta = gather(actualdLdTheta);
                expctddLdTheta = gather(expctddLdTheta);
            end
            testCase.verifyInstanceOf(actualdLdX,expctdDType);
            testCase.verifyInstanceOf(actualdLdTheta,expctdDType);
            testCase.verifyThat(actualdLdX,...
                IsEqualTo(expctddLdX,'Within',tolObj));
            testCase.verifyThat(actualdLdTheta,...
                IsEqualTo(expctddLdTheta,'Within',tolObj));
        end

        function testBackwardGrayscaleWithRandomAngles(testCase, ...
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
            genU = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on',...
                'Device','cpu',...
                'DType',datatype);

            % Parameters
            nSamples = 8;
            nChsTotal = prod(stride);
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            nBlks = nrows*ncols;
            nAngles = (nChsTotal-2)*nChsTotal/8;
            Theta = randn(nAngles,nBlks,nSamples,datatype);
            mus_ = cast(mus,datatype);

            X = randn(nChsTotal,nrows,ncols,nSamples,datatype);
            dLdZ = randn(nChsTotal,nrows,ncols,nSamples,datatype);

            % Expected values (reference computed sample by sample)
            expctddLdX = zeros(nChsTotal,nrows,ncols,nSamples,datatype);
            expctddLdTheta = zeros(nAngles,nBlks,nSamples,datatype);
            for iSample = 1:nSamples
                ang = Theta(:,:,iSample);

                % dLdX = dZdX x dLdZ (Synthesis forward uses Un', so
                % backward applies Un)
                Un = genU.step(ang,mus_,0);
                dLdZi = reshape(dLdZ(:,:,:,iSample),nChsTotal,nBlks);
                Ya = dLdZi(ps+1:ps+pa,:);
                for iblk = 1:nBlks
                    Ya(:,iblk) = Un(:,:,iblk)*Ya(:,iblk);
                end
                expctddLdX(:,:,:,iSample) = reshape([dLdZi(1:ps,:);Ya],nChsTotal,nrows,ncols);

                % dLdTheta_i = <dLdZ,(dVdTheta_i)X>
                Xi = reshape(X(:,:,:,iSample),nChsTotal,nBlks);
                c_low = Xi(ps+1:ps+pa,:);
                dldz_low = dLdZi(ps+1:ps+pa,:);
                for iAngle = 1:nAngles
                    dUn_T = permute(genU.step(ang,mus_,iAngle),[2 1 3]);
                    for iblk = 1:nBlks
                        d_low = dUn_T(:,:,iblk)*c_low(:,iblk);
                        expctddLdTheta(iAngle,iblk,iSample) = sum(dldz_low(:,iblk).*d_low,'all');
                    end
                end
            end

            % Instantiation of target class
            import tansacnet.salsun.*
            layer = salsunIntermediateRotation2dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols],...
                'Name','Vn~',...
                'Device',device_);

            % Actual values
            if usegpu
                X = gpuArray(X);
                dLdZ = gpuArray(dLdZ);
                Theta = gpuArray(Theta);
                mus_ = gpuArray(mus_);
            end
            layer.Mus = mus_;
            [actualdLdX,actualdLdTheta] = layer.backward(X,Theta,[],dLdZ,[]);

            % Evaluation
            if usegpu
                testCase.verifyClass(actualdLdX,'gpuArray')
                actualdLdX = gather(actualdLdX);
                expctddLdX = gather(expctddLdX);
                testCase.verifyClass(actualdLdTheta,'gpuArray')
                actualdLdTheta = gather(actualdLdTheta);
                expctddLdTheta = gather(expctddLdTheta);
            end
            testCase.verifyInstanceOf(actualdLdX,datatype);
            testCase.verifyInstanceOf(actualdLdTheta,datatype);
            testCase.verifyThat(actualdLdX,...
                IsEqualTo(expctddLdX,'Within',tolObj));
            testCase.verifyThat(actualdLdTheta,...
                IsEqualTo(expctddLdTheta,'Within',tolObj));
        end

        function testBackwardGrayscaleAnalysisMode(testCase, ...
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
            genU = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on',...
                'Device','cpu',...
                'DType',datatype);

            % Parameters
            nSamples = 8;
            nChsTotal = prod(stride);
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            nBlks = nrows*ncols;
            nAngles = (nChsTotal-2)*nChsTotal/8;
            Theta = randn(nAngles,nBlks,nSamples,datatype);
            mus_ = cast(mus,datatype);

            X = randn(nChsTotal,nrows,ncols,nSamples,datatype);
            dLdZ = randn(nChsTotal,nrows,ncols,nSamples,datatype);

            % Expected values (reference computed sample by sample)
            expctddLdX = zeros(nChsTotal,nrows,ncols,nSamples,datatype);
            expctddLdTheta = zeros(nAngles,nBlks,nSamples,datatype);
            for iSample = 1:nSamples
                ang = Theta(:,:,iSample);

                % dLdX = dZdX x dLdZ (Analysis forward uses Un, so
                % backward applies Un')
                UnT = permute(genU.step(ang,mus_,0),[2 1 3]);
                dLdZi = reshape(dLdZ(:,:,:,iSample),nChsTotal,nBlks);
                Ya = dLdZi(ps+1:ps+pa,:);
                for iblk = 1:nBlks
                    Ya(:,iblk) = UnT(:,:,iblk)*Ya(:,iblk);
                end
                expctddLdX(:,:,:,iSample) = reshape([dLdZi(1:ps,:);Ya],nChsTotal,nrows,ncols);

                % dLdTheta_i = <dLdZ,(dVdTheta_i)X>
                Xi = reshape(X(:,:,:,iSample),nChsTotal,nBlks);
                c_low = Xi(ps+1:ps+pa,:);
                dldz_low = dLdZi(ps+1:ps+pa,:);
                for iAngle = 1:nAngles
                    dUn = genU.step(ang,mus_,iAngle);
                    for iblk = 1:nBlks
                        d_low = dUn(:,:,iblk)*c_low(:,iblk);
                        expctddLdTheta(iAngle,iblk,iSample) = sum(dldz_low(:,iblk).*d_low,'all');
                    end
                end
            end

            % Instantiation of target class
            import tansacnet.salsun.*
            layer = salsunIntermediateRotation2dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols],...
                'Name','Vn',...
                'Mode','Analysis',...
                'Device',device_);

            % Actual values
            if usegpu
                X = gpuArray(X);
                dLdZ = gpuArray(dLdZ);
                Theta = gpuArray(Theta);
                mus_ = gpuArray(mus_);
            end
            layer.Mus = mus_;
            [actualdLdX,actualdLdTheta] = layer.backward(X,Theta,[],dLdZ,[]);

            % Evaluation
            if usegpu
                testCase.verifyClass(actualdLdX,'gpuArray')
                actualdLdX = gather(actualdLdX);
                expctddLdX = gather(expctddLdX);
                testCase.verifyClass(actualdLdTheta,'gpuArray')
                actualdLdTheta = gather(actualdLdTheta);
                expctddLdTheta = gather(expctddLdTheta);
            end
            testCase.verifyInstanceOf(actualdLdX,datatype);
            testCase.verifyInstanceOf(actualdLdTheta,datatype);
            testCase.verifyThat(actualdLdX,...
                IsEqualTo(expctddLdX,'Within',tolObj));
            testCase.verifyThat(actualdLdTheta,...
                IsEqualTo(expctddLdTheta,'Within',tolObj));
        end

    end

end
