classdef lsunFinalRotation1dLayerTestCase < matlab.unittest.TestCase
    %LSUNFINALROTATION1DLAYERTESTCASE
    %
    %   コンポーネント別に入力(nComponents):
    %      nChs x 1 x nBlks x nSamples
    %
    %   コンポーネント別に出力(nComponents):
    %      nChs x 1 x nBlks x nSamples
    %
    % Requirements: MATLAB R2022b
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
        stride = { 2, 4, 8 };
        mus = { -1, 1 };
        datatype = { 'single', 'double' };
        nblks = struct('small', 2,'medium', 4, 'large', 8);
        usegpu = struct( 'true', true, 'false', false);
    end

    methods (TestClassTeardown)
        function finalCheck(~)
            import tansacnet.lsun.*
            layer = lsunFinalRotation1dLayer(...
                'Stride',2,...
                'NumberOfBlocks',8);
            fprintf("\n --- Check layer for 1-D sequences ---\n");
            checkLayer(layer,[2 1 8],...
                'ObservationDimension',4,...
                'CheckCodegenCompatibility',true)
        end
    end

    methods (Test)

        function testConstructor(testCase, stride)

            % Expected values
            expctdName = 'V0~';
            expctdDescription = "LSUN final rotation " ...
                + "(pt,pb) = (" ...
                + ceil(stride/2) + "," ...
                + floor(stride/2) + "), "  ...
                + "m = " + stride;

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunFinalRotation1dLayer(...
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
            layer = lsunFinalRotation1dLayer(...
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
            X = randn(nChsTotal,1,nblks,nSamples,datatype);
            if usegpu
                X = gpuArray(X);
            end

            % Expected values (identity matrix is self-transpose)
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            W0T = repmat(eye(pt,datatype),[1 1 nblks]);
            U0T = repmat(eye(pb,datatype),[1 1 nblks]);
            expctdZ = zeros(nChsTotal,1,nblks,nSamples,datatype);
            for iSample = 1:nSamples
                Yi = X(:,:,:,iSample);
                for iblk = 1:nblks
                    Yi(1:pt,:,iblk) = W0T(:,:,iblk)*Yi(1:pt,:,iblk);
                    Yi(pt+1:pt+pb,:,iblk) = U0T(:,:,iblk)*Yi(pt+1:pt+pb,:,iblk);
                end
                expctdZ(:,:,:,iSample) = Yi;
            end

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunFinalRotation1dLayer(...
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
                usegpu, stride, datatype)

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
            nblks_ = 4;
            nSamples = 8;
            nChsTotal = stride;
            X = randn(nChsTotal,1,nblks_,nSamples,datatype);
            if usegpu
                X = gpuArray(X);
            end

            % Expected values
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            W0T = repmat(eye(pt,datatype),[1 1 nblks_]);
            U0T = repmat(eye(pb,datatype),[1 1 nblks_]);
            expctdZ = zeros(nChsTotal,1,nblks_,nSamples,datatype);
            for iSample = 1:nSamples
                Yi = X(:,:,:,iSample);
                for iblk = 1:nblks_
                    Yi(1:pt,:,iblk) = W0T(:,:,iblk)*Yi(1:pt,:,iblk);
                    Yi(pt+1:pt+pb,:,iblk) = U0T(:,:,iblk)*Yi(pt+1:pt+pb,:,iblk);
                end
                expctdZ(:,:,:,iSample) = Yi;
            end

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunFinalRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks_,...
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
            genW = OrthonormalMatrixGenerationSystem('Device','cpu');
            genU = OrthonormalMatrixGenerationSystem('Device','cpu');

            % Parameters
            nSamples = 8;
            nChsTotal = stride;
            X = randn(nChsTotal,1,nblks,nSamples,datatype);
            angles = randn((nChsTotal-2)*nChsTotal/4,nblks);

            % Expected values
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            if nChsTotal == 2
                W0T = ones(1,1,nblks);
                U0T = ones(1,1,nblks);
            else
                W0T = permute(genW.step(angles(1:size(angles,1)/2,:),1),[2 1 3]);
                U0T = permute(genU.step(angles(size(angles,1)/2+1:end,:),1),[2 1 3]);
            end
            expctdZ = zeros(nChsTotal,1,nblks,nSamples,datatype);
            for iSample = 1:nSamples
                Yi = X(:,:,:,iSample);
                for iblk = 1:nblks
                    Yi(1:pt,:,iblk) = W0T(:,:,iblk)*Yi(1:pt,:,iblk);
                    Yi(pt+1:pt+pb,:,iblk) = U0T(:,:,iblk)*Yi(pt+1:pt+pb,:,iblk);
                end
                expctdZ(:,:,:,iSample) = Yi;
            end

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunFinalRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'Name','V0~',...
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

        function testPredictWithRandomAnglesNoDcLeackage(testCase, ...
                usegpu, stride, nblks, mus, datatype)

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
            nChsTotal = stride;
            X = randn(nChsTotal,1,nblks,nSamples,datatype);
            angles = randn((nChsTotal-2)*nChsTotal/4,nblks);

            % Expected values
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            anglesNoDc = angles;
            anglesNoDc(1:pt-1,:) = zeros(pt-1,nblks);
            musW = mus*ones(pt,nblks);
            musW(1,:) = 1;
            musU = mus*ones(pb,nblks);
            if nChsTotal == 2
                W0T = reshape(musW,1,1,nblks);
                U0T = reshape(musU,1,1,nblks);
            else
                W0T = permute(genW.step(anglesNoDc(1:size(angles,1)/2,:),musW),[2 1 3]);
                U0T = permute(genU.step(anglesNoDc(size(angles,1)/2+1:end,:),musU),[2 1 3]);
            end
            expctdZ = zeros(nChsTotal,1,nblks,nSamples,datatype);
            for iSample = 1:nSamples
                Yi = X(:,:,:,iSample);
                for iblk = 1:nblks
                    Yi(1:pt,:,iblk) = W0T(:,:,iblk)*Yi(1:pt,:,iblk);
                    Yi(pt+1:pt+pb,:,iblk) = U0T(:,:,iblk)*Yi(pt+1:pt+pb,:,iblk);
                end
                expctdZ(:,:,:,iSample) = Yi;
            end

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunFinalRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
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
            nChsTotal = stride;
            nAnglesH = (nChsTotal-2)*nChsTotal/8;
            anglesW = zeros(nAnglesH,nblks,datatype);
            anglesU = zeros(nAnglesH,nblks,datatype);
            mus_ = cast(1,datatype);

            X = randn(nChsTotal,1,nblks,nSamples,datatype);
            dLdZ = randn(nChsTotal,1,nblks,nSamples,datatype);

            % Expected values
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);

            % dLdX = dZdX x dLdZ: predict applied W0'/U0', so backward
            % applies (W0')' = W0 (untransposed).
            if nAnglesH == 0
                W0 = repmat(mus_,[1 1 nblks]);
                U0 = repmat(mus_,[1 1 nblks]);
            else
                W0 = genW.step(anglesW,mus_,0);
                U0 = genU.step(anglesU,mus_,0);
            end
            expctddLdX = dLdZ;
            for iSample = 1:nSamples
                for iblk = 1:nblks
                    expctddLdX(1:pt,:,iblk,iSample) = W0(:,:,iblk)*expctddLdX(1:pt,:,iblk,iSample);
                    expctddLdX(pt+1:pt+pb,:,iblk,iSample) = U0(:,:,iblk)*expctddLdX(pt+1:pt+pb,:,iblk,iSample);
                end
            end

            % dLdWi = <dLdZ,(dVdWi)X> (transposed per-angle derivative,
            % matching predict)
            dldw_ = zeros(2*nAnglesH,nblks,datatype);
            dldz_top = dLdZ(1:pt,:,:,:);
            dldz_btm = dLdZ(pt+1:pt+pb,:,:,:);
            c_top = X(1:pt,:,:,:);
            c_btm = X(pt+1:pt+pb,:,:,:);
            for iAngle = 1:nAnglesH
                dW0_T = permute(genW.step(anglesW,mus_,iAngle),[2 1 3]);
                dU0_T = permute(genU.step(anglesU,mus_,iAngle),[2 1 3]);
                for iblk = 1:nblks
                    dldz_top_iblk = squeeze(dldz_top(:,:,iblk,:));
                    dldz_btm_iblk = squeeze(dldz_btm(:,:,iblk,:));
                    c_top_iblk = squeeze(c_top(:,:,iblk,:));
                    c_btm_iblk = squeeze(c_btm(:,:,iblk,:));
                    d_top_iblk = zeros(size(c_top_iblk),'like',c_top_iblk);
                    d_btm_iblk = zeros(size(c_btm_iblk),'like',c_btm_iblk);
                    for iSample = 1:nSamples
                        d_top_iblk(:,iSample) = dW0_T(:,:,iblk)*c_top_iblk(:,iSample);
                        d_btm_iblk(:,iSample) = dU0_T(:,:,iblk)*c_btm_iblk(:,iSample);
                    end
                    dldw_(iAngle,iblk) = sum(dldz_top_iblk.*d_top_iblk,'all');
                    dldw_(nAnglesH+iAngle,iblk) = sum(dldz_btm_iblk.*d_btm_iblk,'all');
                end
            end
            expctddLdW = dldw_;

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunFinalRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'Name','V0~',...
                'Device',device_);

            % Actual values
            if usegpu
                X = gpuArray(X);
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

            nblks_ = 8;

            % Parameters
            nSamples = 8;
            nChsTotal = stride;
            nAnglesH = (nChsTotal-2)*nChsTotal/8;
            anglesW = zeros(nAnglesH,nblks_,datatype);
            anglesU = zeros(nAnglesH,nblks_,datatype);
            mus_ = cast(1,datatype);

            X = randn(nChsTotal,1,nblks_,nSamples,datatype);
            dLdZ = randn(nChsTotal,1,nblks_,nSamples,datatype);

            % Expected values
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);

            if nAnglesH == 0
                W0 = repmat(mus_,[1 1 nblks_]);
                U0 = repmat(mus_,[1 1 nblks_]);
            else
                W0 = genW.step(anglesW,mus_,0);
                U0 = genU.step(anglesU,mus_,0);
            end
            expctddLdX = dLdZ;
            for iSample = 1:nSamples
                for iblk = 1:nblks_
                    expctddLdX(1:pt,:,iblk,iSample) = W0(:,:,iblk)*expctddLdX(1:pt,:,iblk,iSample);
                    expctddLdX(pt+1:pt+pb,:,iblk,iSample) = U0(:,:,iblk)*expctddLdX(pt+1:pt+pb,:,iblk,iSample);
                end
            end

            dldw_ = zeros(2*nAnglesH,nblks_,datatype);
            dldz_top = dLdZ(1:pt,:,:,:);
            dldz_btm = dLdZ(pt+1:pt+pb,:,:,:);
            c_top = X(1:pt,:,:,:);
            c_btm = X(pt+1:pt+pb,:,:,:);
            for iAngle = 1:nAnglesH
                dW0_T = permute(genW.step(anglesW,mus_,iAngle),[2 1 3]);
                dU0_T = permute(genU.step(anglesU,mus_,iAngle),[2 1 3]);
                for iblk = 1:nblks_
                    dldz_top_iblk = squeeze(dldz_top(:,:,iblk,:));
                    dldz_btm_iblk = squeeze(dldz_btm(:,:,iblk,:));
                    c_top_iblk = squeeze(c_top(:,:,iblk,:));
                    c_btm_iblk = squeeze(c_btm(:,:,iblk,:));
                    d_top_iblk = zeros(size(c_top_iblk),'like',c_top_iblk);
                    d_btm_iblk = zeros(size(c_btm_iblk),'like',c_btm_iblk);
                    for iSample = 1:nSamples
                        d_top_iblk(:,iSample) = dW0_T(:,:,iblk)*c_top_iblk(:,iSample);
                        d_btm_iblk(:,iSample) = dU0_T(:,:,iblk)*c_btm_iblk(:,iSample);
                    end
                    dldw_(iAngle,iblk) = sum(dldz_top_iblk.*d_top_iblk,'all');
                    dldw_(nAnglesH+iAngle,iblk) = sum(dldz_btm_iblk.*d_btm_iblk,'all');
                end
            end
            expctddLdW = dldw_;

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunFinalRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks_,...
                'Name','V0~',...
                'Device',expctdDevice,...
                'DType',expctdDType);

            % Actual values
            if usegpu
                X = gpuArray(X);
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
            nChsTotal = stride;
            nAnglesH = (nChsTotal-2)*nChsTotal/8;
            anglesW = randn(nAnglesH,nblks,datatype);
            anglesU = randn(nAnglesH,nblks,datatype);
            mus_ = cast(1,datatype);

            X = randn(nChsTotal,1,nblks,nSamples,datatype);
            dLdZ = randn(nChsTotal,1,nblks,nSamples,datatype);

            % Expected values
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);

            if nAnglesH == 0
                W0 = repmat(mus_,[1 1 nblks]);
                U0 = repmat(mus_,[1 1 nblks]);
            else
                W0 = genW.step(anglesW,mus_,0);
                U0 = genU.step(anglesU,mus_,0);
            end
            expctddLdX = dLdZ;
            for iSample = 1:nSamples
                for iblk = 1:nblks
                    expctddLdX(1:pt,:,iblk,iSample) = W0(:,:,iblk)*expctddLdX(1:pt,:,iblk,iSample);
                    expctddLdX(pt+1:pt+pb,:,iblk,iSample) = U0(:,:,iblk)*expctddLdX(pt+1:pt+pb,:,iblk,iSample);
                end
            end

            dldw_ = zeros(2*nAnglesH,nblks,datatype);
            dldz_top = dLdZ(1:pt,:,:,:);
            dldz_btm = dLdZ(pt+1:pt+pb,:,:,:);
            c_top = X(1:pt,:,:,:);
            c_btm = X(pt+1:pt+pb,:,:,:);
            for iAngle = 1:nAnglesH
                dW0_T = permute(genW.step(anglesW,mus_,iAngle),[2 1 3]);
                dU0_T = permute(genU.step(anglesU,mus_,iAngle),[2 1 3]);
                for iblk = 1:nblks
                    dldz_top_iblk = squeeze(dldz_top(:,:,iblk,:));
                    dldz_btm_iblk = squeeze(dldz_btm(:,:,iblk,:));
                    c_top_iblk = squeeze(c_top(:,:,iblk,:));
                    c_btm_iblk = squeeze(c_btm(:,:,iblk,:));
                    d_top_iblk = zeros(size(c_top_iblk),'like',c_top_iblk);
                    d_btm_iblk = zeros(size(c_btm_iblk),'like',c_btm_iblk);
                    for iSample = 1:nSamples
                        d_top_iblk(:,iSample) = dW0_T(:,:,iblk)*c_top_iblk(:,iSample);
                        d_btm_iblk(:,iSample) = dU0_T(:,:,iblk)*c_btm_iblk(:,iSample);
                    end
                    dldw_(iAngle,iblk) = sum(dldz_top_iblk.*d_top_iblk,'all');
                    dldw_(nAnglesH+iAngle,iblk) = sum(dldz_btm_iblk.*d_btm_iblk,'all');
                end
            end
            expctddLdW = dldw_;

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunFinalRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'Name','V0~', ...
                'Device',device_);

            % Actual values
            if usegpu
                X = gpuArray(X);
                anglesW = gpuArray(anglesW);
                anglesU = gpuArray(anglesU);
                mus_ = gpuArray(mus_);
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
                usegpu, stride, nblks, mus, datatype)

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
            nChsTotal = stride;
            nAnglesH = (nChsTotal-2)*nChsTotal/8;
            anglesW = randn(nAnglesH,nblks,datatype);
            anglesU = randn(nAnglesH,nblks,datatype);

            X = randn(nChsTotal,1,nblks,nSamples,datatype);
            dLdZ = randn(nChsTotal,1,nblks,nSamples,datatype);

            % Expected values
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);

            % dLdX = dZdX x dLdZ: predict applied W0'/U0', so backward
            % applies (W0')' = W0 (untransposed).
            anglesW_NoDc = anglesW;
            anglesW_NoDc(1:pt-1,:) = zeros(pt-1,nblks,datatype);
            musW = mus*ones(pt,nblks,datatype);
            musW(1,:) = ones(1,nblks,datatype);
            musU = mus*ones(pb,nblks,datatype);
            if nAnglesH == 0
                W0 = reshape(musW,1,1,nblks);
                U0 = reshape(musU,1,1,nblks);
            else
                W0 = genW.step(anglesW_NoDc,musW,0);
                U0 = genU.step(anglesU,musU,0);
            end
            expctddLdX = dLdZ;
            for iSample = 1:nSamples
                for iblk = 1:nblks
                    expctddLdX(1:pt,:,iblk,iSample) = W0(:,:,iblk)*expctddLdX(1:pt,:,iblk,iSample);
                    expctddLdX(pt+1:pt+pb,:,iblk,iSample) = U0(:,:,iblk)*expctddLdX(pt+1:pt+pb,:,iblk,iSample);
                end
            end

            % dLdWi = <dLdZ,(dVdWi)X> (transposed per-angle derivative,
            % matching predict)
            dldw_ = zeros(2*nAnglesH,nblks,datatype);
            dldz_top = dLdZ(1:pt,:,:,:);
            dldz_btm = dLdZ(pt+1:pt+pb,:,:,:);
            c_top = X(1:pt,:,:,:);
            c_btm = X(pt+1:pt+pb,:,:,:);
            for iAngle = 1:nAnglesH
                dW0_T = permute(genW.step(anglesW_NoDc,musW,iAngle),[2 1 3]);
                dU0_T = permute(genU.step(anglesU,musU,iAngle),[2 1 3]);
                for iblk = 1:nblks
                    dldz_top_iblk = squeeze(dldz_top(:,:,iblk,:));
                    dldz_btm_iblk = squeeze(dldz_btm(:,:,iblk,:));
                    c_top_iblk = squeeze(c_top(:,:,iblk,:));
                    c_btm_iblk = squeeze(c_btm(:,:,iblk,:));
                    d_top_iblk = zeros(size(c_top_iblk),'like',c_top_iblk);
                    d_btm_iblk = zeros(size(c_btm_iblk),'like',c_btm_iblk);
                    for iSample = 1:nSamples
                        d_top_iblk(:,iSample) = dW0_T(:,:,iblk)*c_top_iblk(:,iSample);
                        d_btm_iblk(:,iSample) = dU0_T(:,:,iblk)*c_btm_iblk(:,iSample);
                    end
                    dldw_(iAngle,iblk) = sum(dldz_top_iblk.*d_top_iblk,'all');
                    dldw_(nAnglesH+iAngle,iblk) = sum(dldz_btm_iblk.*d_btm_iblk,'all');
                end
            end
            expctddLdW = dldw_;

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunFinalRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'NoDcLeakage',true,...
                'Name','V0~', ...
                'Device',device_);

            % Actual values
            if usegpu
                X = gpuArray(X);
                anglesW = gpuArray(anglesW);
                anglesU = gpuArray(anglesU);
                mus = gpuArray(mus);
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
