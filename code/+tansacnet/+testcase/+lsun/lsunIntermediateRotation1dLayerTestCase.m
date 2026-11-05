classdef lsunIntermediateRotation1dLayerTestCase < matlab.unittest.TestCase
    %LSUNINTERMEDIATEROTATION1DLAYERTESTCASE
    %
    %   コンポーネント別に入力(nComponents)
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
        datatype = { 'single', 'double' };
        mus = { -1, 1 };
        nblks = struct('small', 2,'medium', 4, 'large', 8);
        usegpu = struct( 'true', true, 'false', false);
    end

    methods (TestClassTeardown)
        function finalCheck(~)
            import tansacnet.lsun.*
            layer = lsunIntermediateRotation1dLayer(...
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
            expctdName = 'Vn~';
            expctdMode = 'Synthesis';
            expctdDescription = "Synthesis LSUN intermediate rotation " ...
                + "(pt,pb) = (" ...
                + ceil(stride/2) + "," + floor(stride/2) + ")";

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunIntermediateRotation1dLayer(...
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
            expctdMode = 'Synthesis';

            device_ = ["cpu", "cuda"];
            expctdDevice = device_(usegpu+1);
            expctdDType = datatype;

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunIntermediateRotation1dLayer(...
                'Stride',stride,...
                'Name',expctdName,...
                'Mode',expctdMode,...
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
            X = randn(nChsTotal,1,nblks,nSamples,datatype);
            if usegpu
                X = gpuArray(X);
            end
            % Expected values
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            UnT = repmat(mus*eye(pb,datatype),[1 1 nblks]);
            Y = X;
            Yb = reshape(Y(pt+1:pt+pb,:,:,:),pb,nblks,nSamples);
            Zb = zeros(size(Yb),'like',Yb);
            for iSample = 1:nSamples
                for iblk = 1:nblks
                    Zb(:,iblk,iSample) = UnT(:,:,iblk)*Yb(:,iblk,iSample);
                end
            end
            Y(pt+1:pt+pb,:,:,:) = reshape(Zb,pb,1,nblks,nSamples);
            expctdZ = Y;

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunIntermediateRotation1dLayer(...
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
                usegpu, stride, mus, datatype)

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
            UnT = repmat(mus*eye(pb,datatype),[1 1 nblks_]);
            Y = X;
            Yb = reshape(Y(pt+1:pt+pb,:,:,:),pb,nblks_,nSamples);
            Zb = zeros(size(Yb),'like',Yb);
            for iSample = 1:nSamples
                for iblk = 1:nblks_
                    Zb(:,iblk,iSample) = UnT(:,:,iblk)*Yb(:,iblk,iSample);
                end
            end
            Y(pt+1:pt+pb,:,:,:) = reshape(Zb,pb,1,nblks_,nSamples);
            expctdZ = Y;

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunIntermediateRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks_,...
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
            nChsTotal = stride;
            X = randn(nChsTotal,1,nblks,nSamples,datatype);
            angles = randn((nChsTotal-2)*nChsTotal/8,nblks);

            % Expected values
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            if nChsTotal == 2
                UnT = repmat(mus,[1 1 nblks]);
            else
                UnT = permute(genU.step(angles,mus),[2 1 3]);
            end
            Y = X;
            Yb = reshape(Y(pt+1:pt+pb,:,:,:),pb,nblks,nSamples);
            Zb = zeros(size(Yb),'like',Yb);
            for iSample = 1:nSamples
                for iblk = 1:nblks
                    Zb(:,iblk,iSample) = UnT(:,:,iblk)*Yb(:,iblk,iSample);
                end
            end
            Y(pt+1:pt+pb,:,:,:) = reshape(Zb,pb,1,nblks,nSamples);
            expctdZ = Y;

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunIntermediateRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'Name','Vn~', ...
                'Device',device_);

            % Actual values
            if usegpu
                X = gpuArray(X);
                angles = gpuArray(angles);
                mus = gpuArray(mus);
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

        function testPredictAnalysisMode(testCase, ...
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
            genU = OrthonormalMatrixGenerationSystem('Device','cpu');

            % Parameters
            nSamples = 8;
            nChsTotal = stride;
            X = randn(nChsTotal,1,nblks,nSamples,datatype);
            angles = randn((nChsTotal-2)*nChsTotal/8,nblks);

            % Expected values
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            if nChsTotal == 2
                Un = repmat(mus,[1 1 nblks]);
            else
                Un = genU.step(angles,mus);
            end
            Y = X;
            Yb = reshape(Y(pt+1:pt+pb,:,:,:),pb,nblks,nSamples);
            Zb = zeros(size(Yb),'like',Yb);
            for iSample = 1:nSamples
                for iblk = 1:nblks
                    Zb(:,iblk,iSample) = Un(:,:,iblk)*Yb(:,iblk,iSample);
                end
            end
            Y(pt+1:pt+pb,:,:,:) = reshape(Zb,pb,1,nblks,nSamples);
            expctdZ = Y;
            expctdDescription = "Analysis LSUN intermediate rotation " ...
                + "(pt,pb) = (" ...
                + pt + "," + pb + ")";

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunIntermediateRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'Name','Vn',...
                'Mode','Analysis', ...
                'Device',device_);

            % Actual values
            if usegpu
                X = gpuArray(X);
                angles = gpuArray(angles);
            end
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
            nChsTotal = stride;
            nAngles = (nChsTotal-2)*nChsTotal/8;
            angles = zeros(nAngles,nblks,datatype);
            mus_ = cast(mus,datatype);

            X = randn(nChsTotal,1,nblks,nSamples,datatype);
            dLdZ = randn(nChsTotal,1,nblks,nSamples,datatype);

            % Expected values
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);

            % dLdX = dZdX x dLdZ
            if nAngles == 0
                Un = repmat(mus_,[1 1 nblks]);
            else
                Un = genU.step(angles,mus_,0);
            end
            adLd_ = dLdZ;
            cdLd_low = reshape(adLd_(pt+1:pt+pb,:,:,:),pb,nblks,nSamples);
            for iSample = 1:nSamples
                for iblk = 1:nblks
                    cdLd_low(:,iblk,iSample) = Un(:,:,iblk)*cdLd_low(:,iblk,iSample);
                end
            end
            adLd_(pt+1:pt+pb,:,:,:) = reshape(cdLd_low,pb,1,nblks,nSamples);
            expctddLdX = adLd_;

            % dLdWi = <dLdZ,(dVdWi)X>
            expctddLdW = zeros(nAngles,nblks,datatype);
            c_low = reshape(X(pt+1:pt+pb,:,:,:),pb,nblks,nSamples);
            dldz_low = reshape(dLdZ(pt+1:pt+pb,:,:,:),pb,nblks,nSamples);
            for iAngle = 1:nAngles
                dUn_T = permute(genU.step(angles,mus_,iAngle),[2 1 3]);
                for iblk = 1:nblks
                    c_low_iblk = squeeze(c_low(:,iblk,:));
                    c_low_iblk = dUn_T(:,:,iblk)*c_low_iblk;
                    dldz_iblk = squeeze(dldz_low(:,iblk,:));
                    expctddLdW(iAngle,iblk) = sum(dldz_iblk.*c_low_iblk,'all');
                end
            end

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunIntermediateRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'Name','Vn~', ...
                'Device',device_);

            % Actual values
            if usegpu
                X = gpuArray(X);
                dLdZ = gpuArray(dLdZ);
                mus_ = gpuArray(mus_);
            end
            layer.Mus = mus_;
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

        function testBackwardWithDeviceAndDType(testCase, ...
                usegpu, stride, mus, datatype)

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

            nblks_ = 8;

            % Parameters
            nSamples = 8;
            nChsTotal = stride;
            nAngles = (nChsTotal-2)*nChsTotal/8;
            angles = zeros(nAngles,nblks_,datatype);
            mus_ = cast(mus,datatype);

            X = randn(nChsTotal,1,nblks_,nSamples,datatype);
            dLdZ = randn(nChsTotal,1,nblks_,nSamples,datatype);

            % Expected values
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);

            % dLdX = dZdX x dLdZ
            if nAngles == 0
                Un = repmat(mus_,[1 1 nblks_]);
            else
                Un = genU.step(angles,mus_,0);
            end
            adLd_ = dLdZ;
            cdLd_low = reshape(adLd_(pt+1:pt+pb,:,:,:),pb,nblks_,nSamples);
            for iSample = 1:nSamples
                for iblk = 1:nblks_
                    cdLd_low(:,iblk,iSample) = Un(:,:,iblk)*cdLd_low(:,iblk,iSample);
                end
            end
            adLd_(pt+1:pt+pb,:,:,:) = reshape(cdLd_low,pb,1,nblks_,nSamples);
            expctddLdX = adLd_;

            % dLdWi = <dLdZ,(dVdWi)X>
            expctddLdW = zeros(nAngles,nblks_,datatype);
            c_low = reshape(X(pt+1:pt+pb,:,:,:),pb,nblks_,nSamples);
            dldz_low = reshape(dLdZ(pt+1:pt+pb,:,:,:),pb,nblks_,nSamples);
            for iAngle = 1:nAngles
                dUn_T = permute(genU.step(angles,mus_,iAngle),[2 1 3]);
                for iblk = 1:nblks_
                    c_low_iblk = squeeze(c_low(:,iblk,:));
                    c_low_iblk = dUn_T(:,:,iblk)*c_low_iblk;
                    dldz_iblk = squeeze(dldz_low(:,iblk,:));
                    expctddLdW(iAngle,iblk) = sum(dldz_iblk.*c_low_iblk,'all');
                end
            end

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunIntermediateRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks_,...
                'Name','Vn~',...
                'Device',expctdDevice,...
                'DType',expctdDType);

            % Actual values
            if usegpu
                X = gpuArray(X);
                dLdZ = gpuArray(dLdZ);
                mus_ = gpuArray(mus_);
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
            genU = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on',...
                'Device','cpu',...
                'DType',datatype);

            % Parameters
            nSamples = 8;
            nChsTotal = stride;
            nAngles = (nChsTotal-2)*nChsTotal/8;
            angles = randn(nAngles,nblks,datatype);
            mus_ = cast(mus,datatype);

            X = randn(nChsTotal,1,nblks,nSamples,datatype);
            dLdZ = randn(nChsTotal,1,nblks,nSamples,datatype);

            % Expected values
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);

            % dLdX = dZdX x dLdZ
            if nAngles == 0
                Un = repmat(mus_,[1 1 nblks]);
            else
                Un = genU.step(angles,mus_,0);
            end
            adLd_ = dLdZ;
            cdLd_low = reshape(adLd_(pt+1:pt+pb,:,:,:),pb,nblks,nSamples);
            for iSample = 1:nSamples
                for iblk = 1:nblks
                    cdLd_low(:,iblk,iSample) = Un(:,:,iblk)*cdLd_low(:,iblk,iSample);
                end
            end
            adLd_(pt+1:pt+pb,:,:,:) = reshape(cdLd_low,pb,1,nblks,nSamples);
            expctddLdX = adLd_;

            % dLdWi = <dLdZ,(dVdWi)X>
            expctddLdW = zeros(nAngles,nblks,datatype);
            c_low = reshape(X(pt+1:pt+pb,:,:,:),pb,nblks,nSamples);
            dldz_low = reshape(dLdZ(pt+1:pt+pb,:,:,:),pb,nblks,nSamples);
            for iAngle = 1:nAngles
                dUn_T = permute(genU.step(angles,mus_,iAngle),[2 1 3]);
                for iblk = 1:nblks
                    c_low_iblk = squeeze(c_low(:,iblk,:));
                    c_low_iblk = dUn_T(:,:,iblk)*c_low_iblk;
                    dldz_iblk = squeeze(dldz_low(:,iblk,:));
                    expctddLdW(iAngle,iblk) = sum(dldz_iblk.*c_low_iblk,'all');
                end
            end

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunIntermediateRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'Name','Vn~', ...
                'Device',device_);

            % Actual values
            if usegpu
                X = gpuArray(X);
                dLdZ = gpuArray(dLdZ);
                angles = gpuArray(angles);
                mus_ = gpuArray(mus_);
            end
            layer.Mus = mus_;
            layer.Angles = angles;
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
            nChsTotal = stride;
            nAngles = (nChsTotal-2)*nChsTotal/8;
            angles = randn(nAngles,nblks,datatype);
            mus_ = cast(mus,datatype);

            X = randn(nChsTotal,1,nblks,nSamples,datatype);
            dLdZ = randn(nChsTotal,1,nblks,nSamples,datatype);

            % Expected values
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);

            % dLdX = dZdX x dLdZ
            if nAngles == 0
                UnT = repmat(mus_,[1 1 nblks]);
            else
                UnT = permute(genU.step(angles,mus_,0),[2 1 3]);
            end
            adLd_ = dLdZ;
            cdLd_low = reshape(adLd_(pt+1:pt+pb,:,:,:),pb,nblks,nSamples);
            for iSample = 1:nSamples
                for iblk = 1:nblks
                    cdLd_low(:,iblk,iSample) = UnT(:,:,iblk)*cdLd_low(:,iblk,iSample);
                end
            end
            adLd_(pt+1:pt+pb,:,:,:) = reshape(cdLd_low,pb,1,nblks,nSamples);
            expctddLdX = adLd_;

            % dLdWi = <dLdZ,(dVdWi)X>
            expctddLdW = zeros(nAngles,nblks,datatype);
            c_low = reshape(X(pt+1:pt+pb,:,:,:),pb,nblks,nSamples);
            dldz_low = reshape(dLdZ(pt+1:pt+pb,:,:,:),pb,nblks,nSamples);
            for iAngle = 1:nAngles
                dUn = genU.step(angles,mus_,iAngle);
                for iblk = 1:nblks
                    c_low_iblk = squeeze(c_low(:,iblk,:));
                    c_low_iblk = dUn(:,:,iblk)*c_low_iblk;
                    dldz_iblk = squeeze(dldz_low(:,iblk,:));
                    expctddLdW(iAngle,iblk) = sum(dldz_iblk.*c_low_iblk,'all');
                end
            end

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunIntermediateRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'Name','Vn',...
                'Mode','Analysis',...
                'Device',device_);

            % Actual values
            if usegpu
                X = gpuArray(X);
                dLdZ = gpuArray(dLdZ);
                angles = gpuArray(angles);
                mus_ = gpuArray(mus_);
            end
            layer.Mus = mus_;
            layer.Angles = angles;
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
