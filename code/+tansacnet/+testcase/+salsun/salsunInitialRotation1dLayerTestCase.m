classdef salsunInitialRotation1dLayerTestCase < matlab.unittest.TestCase
    %SALSUNINITIALROTATION1DLAYERTESTCASE
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
        stride = { 2, 4, 6 };
        mus = { -1, 1 };
        datatype = { 'single', 'double' };
        nblks = struct('small', 2,'medium', 4, 'large', 8);
        usegpu = struct('true', true, 'false', false);
    end

    methods (TestClassTeardown)

        function finalCheck(~)
            import tansacnet.salsun.*
            fprintf("\n --- Check layer for 1-D sequences (SA-LSUN) ---\n");
            stride_ = 4;
            nChsTotal = stride_;
            nAngles = (nChsTotal-2)*nChsTotal/4;
            nBlks = 8;
            layer = salsunInitialRotation1dLayer(...
                'Stride',stride_,...
                'NumberOfBlocks',nBlks,...
                'Name','V0',...
                'Mus',-1);
            % CheckCodegenCompatibility omitted: Theta's observation
            % (sample) dimension is at position 3, not 2 or 4.
            checkLayer(layer,{[nChsTotal 1 nBlks],[nAngles nBlks]},...
                'ObservationDimension',[4 3])
        end

    end

    methods (Test)

        function testConstructor(testCase, stride)

            % Expected values
            expctdName = 'V0';
            expctdDescription = "SA-LSUN initial rotation " ...
                + "(pt,pb) = (" ...
                + ceil(stride/2) + "," ...
                + floor(stride/2) + "), "  ...
                + "m = " + stride;

            % Instantiation of target class
            import tansacnet.salsun.*
            layer = salsunInitialRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',3,...
                'Name',expctdName);

            % Actual values
            actualName = layer.Name;
            actualDescription = layer.Description;

            % Evaluation
            testCase.verifyEqual(actualName,expctdName);
            testCase.verifyEqual(actualDescription,expctdDescription);
            testCase.verifyEqual(layer.Stride,stride);
            testCase.verifyEqual(layer.InputNames,{'x','theta'});
        end

        function testConstructorWithDeviceAndDType(testCase, stride, usegpu, datatype)

            % Expected values
            expctdName = 'V0';
            device_ = ["cpu", "cuda"];
            expctdDevice = device_(usegpu+1);
            expctdDType = datatype;

            % Instantiation of target class
            import tansacnet.salsun.*
            layer = salsunInitialRotation1dLayer(...
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
            nSamples = 2;
            nChsTotal = stride;
            nAngles = (nChsTotal-2)*nChsTotal/4;
            X = randn(nChsTotal,1,nblks,nSamples,datatype);
            Theta = zeros(nAngles,nblks,nSamples,datatype);
            if usegpu
                X = gpuArray(X);
                Theta = gpuArray(Theta);
            end

            % Expected values
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            W0 = repmat(eye(pt,datatype),[1 1 nblks nSamples]);
            U0 = repmat(eye(pb,datatype),[1 1 nblks nSamples]);
            expctdZ = zeros(nChsTotal,1,nblks,nSamples,datatype);
            for iSample = 1:nSamples
                Yi = X(:,:,:,iSample);
                W0i = W0(:,:,:,iSample);
                U0i = U0(:,:,:,iSample);
                for iblk = 1:nblks
                    Yi(1:pt,:,iblk) = W0i(:,:,iblk)*Yi(1:pt,:,iblk);
                    Yi(pt+1:pt+pb,:,iblk) = U0i(:,:,iblk)*Yi(pt+1:pt+pb,:,iblk);
                end
                expctdZ(:,:,:,iSample) = Yi;
            end

            % Instantiation of target class
            import tansacnet.salsun.*
            layer = salsunInitialRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'Name','V0');

            % Actual values
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
            nSamples = 2;
            nChsTotal = stride;
            nAngles = (nChsTotal-2)*nChsTotal/4;
            X = randn(nChsTotal,1,nblks_,nSamples,datatype);
            Theta = zeros(nAngles,nblks_,nSamples,datatype);
            if usegpu
                X = gpuArray(X);
                Theta = gpuArray(Theta);
            end

            % Expected values
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            W0 = repmat(eye(pt,datatype),[1 1 nblks_ nSamples]);
            U0 = repmat(eye(pb,datatype),[1 1 nblks_ nSamples]);
            expctdZ = zeros(nChsTotal,1,nblks_,nSamples,datatype);
            for iSample = 1:nSamples
                Yi = X(:,:,:,iSample);
                W0i = W0(:,:,:,iSample);
                U0i = U0(:,:,:,iSample);
                for iblk = 1:nblks_
                    Yi(1:pt,:,iblk) = W0i(:,:,iblk)*Yi(1:pt,:,iblk);
                    Yi(pt+1:pt+pb,:,iblk) = U0i(:,:,iblk)*Yi(pt+1:pt+pb,:,iblk);
                end
                expctdZ(:,:,:,iSample) = Yi;
            end

            % Instantiation of target class
            import tansacnet.salsun.*
            layer = salsunInitialRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks_,...
                'Name','V0',...
                'Device',expctdDevice,...
                'DType',expctdDType);

            % Actual values
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

        function testPredictRoundTripInitialThenFinal(testCase, stride)

            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-10);

            % Parameters
            nSamples = 2;
            nChsTotal = stride;
            nAngles = (nChsTotal-2)*nChsTotal/4;
            nBlks = 3;
            X = randn(nChsTotal,1,nBlks,nSamples);
            Theta = 0.5*randn(nAngles,nBlks,nSamples);

            import tansacnet.salsun.*
            layerV0 = salsunInitialRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nBlks,...
                'Name','V0');
            layerV0T = salsunFinalRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nBlks,...
                'Name','V0T');

            Z = layerV0.predict(X,Theta);
            Xrec = layerV0T.predict(Z,Theta);

            testCase.verifyThat(Xrec,IsEqualTo(X,'Within',tolObj));
        end

        function testPredictWithRandomAnglesDataTypeAndDevice(testCase, ...
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
            nSamples = 2;
            nChsTotal = stride;
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            nAngles = (nChsTotal-2)*nChsTotal/4;
            nAnglesH = nAngles/2;
            mus_ = mus;
            X = randn(nChsTotal,1,nblks,nSamples,datatype);
            Theta = randn(nAngles,nblks,nSamples);

            % Expected values (reference computed sample by sample)
            expctdZ = zeros(nChsTotal,1,nblks,nSamples,datatype);
            for iSample = 1:nSamples
                angles = Theta(:,:,iSample);
                anglesW = angles(1:nAnglesH,:);
                anglesU = angles(nAnglesH+1:nAngles,:);
                if nAnglesH == 0
                    W0 = repmat(mus_,[1 1 nblks]);
                    U0 = repmat(mus_,[1 1 nblks]);
                else
                    W0 = genW.step(anglesW,mus_);
                    U0 = genU.step(anglesU,mus_);
                end
                Yi = X(:,:,:,iSample);
                for iblk = 1:nblks
                    Yi(1:pt,:,iblk) = W0(:,:,iblk)*Yi(1:pt,:,iblk);
                    Yi(pt+1:pt+pb,:,iblk) = U0(:,:,iblk)*Yi(pt+1:pt+pb,:,iblk);
                end
                expctdZ(:,:,:,iSample) = Yi;
            end

            % Instantiation of target class
            import tansacnet.salsun.*
            layer = salsunInitialRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'Name','V0',...
                'Mus',mus_,...
                'Device',device_);

            % Actual values
            if usegpu
                X = gpuArray(X);
                Theta = gpuArray(Theta);
            end
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

        function testBackwardWithZeroAngles(testCase, ...
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
            nSamples = 2;
            nChsTotal = stride;
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            nAngles = (nChsTotal-2)*nChsTotal/4;
            nAnglesH = nAngles/2;
            mus_ = cast(1,datatype);
            angles = zeros(nAngles,nblks,datatype);
            anglesW = angles(1:nAnglesH,:);
            anglesU = angles(nAnglesH+1:nAngles,:);
            Theta = repmat(angles,[1 1 nSamples]);

            X = randn(nChsTotal,1,nblks,nSamples,datatype);
            dLdZ = randn(nChsTotal,1,nblks,nSamples,datatype);

            % Expected values
            if nAnglesH == 0
                W0T = repmat(mus_,[1 1 nblks]);
                U0T = repmat(mus_,[1 1 nblks]);
            else
                W0T = permute(genW.step(anglesW,mus_,0),[2 1 3]);
                U0T = permute(genU.step(anglesU,mus_,0),[2 1 3]);
            end
            expctddLdX = dLdZ;
            for iSample = 1:nSamples
                for iblk = 1:nblks
                    expctddLdX(1:pt,:,iblk,iSample) = W0T(:,:,iblk)*expctddLdX(1:pt,:,iblk,iSample);
                    expctddLdX(pt+1:pt+pb,:,iblk,iSample) = U0T(:,:,iblk)*expctddLdX(pt+1:pt+pb,:,iblk,iSample);
                end
            end

            % dLdTheta_i = <dLdZ,(dVdTheta_i)X>
            expctddLdTheta = zeros(nAngles,nblks,nSamples,datatype);
            for iAngle = 1:nAnglesH
                dW = genW.step(anglesW,mus_,iAngle);
                dU = genU.step(anglesU,mus_,iAngle);
                for iblk = 1:nblks
                    for iSample = 1:nSamples
                        d_top = dW(:,:,iblk)*X(1:pt,:,iblk,iSample);
                        d_btm = dU(:,:,iblk)*X(pt+1:pt+pb,:,iblk,iSample);
                        expctddLdTheta(iAngle,iblk,iSample) = sum(dLdZ(1:pt,:,iblk,iSample).*d_top,'all');
                        expctddLdTheta(nAnglesH+iAngle,iblk,iSample) = sum(dLdZ(pt+1:pt+pb,:,iblk,iSample).*d_btm,'all');
                    end
                end
            end

            % Instantiation of target class
            import tansacnet.salsun.*
            layer = salsunInitialRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'Name','V0',...
                'Mus',mus_,...
                'Device',device_);

            % Actual values
            if usegpu
                X = gpuArray(X);
                dLdZ = gpuArray(dLdZ);
                Theta = gpuArray(Theta);
            end
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
            genW = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on',...
                'Device','cpu',...
                'DType',datatype);
            genU = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on',...
                'Device','cpu',...
                'DType',datatype);

            % Parameters
            nSamples = 2;
            nChsTotal = stride;
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            nAngles = (nChsTotal-2)*nChsTotal/4;
            nAnglesH = nAngles/2;
            mus_ = cast(1,datatype);
            angles = zeros(nAngles,nblks,datatype);
            anglesW = angles(1:nAnglesH,:);
            anglesU = angles(nAnglesH+1:nAngles,:);
            Theta = repmat(angles,[1 1 nSamples]);

            X = randn(nChsTotal,1,nblks,nSamples,datatype);
            dLdZ = randn(nChsTotal,1,nblks,nSamples,datatype);

            % Expected values
            if nAnglesH == 0
                W0T = repmat(mus_,[1 1 nblks]);
                U0T = repmat(mus_,[1 1 nblks]);
            else
                W0T = permute(genW.step(anglesW,mus_,0),[2 1 3]);
                U0T = permute(genU.step(anglesU,mus_,0),[2 1 3]);
            end
            expctddLdX = dLdZ;
            for iSample = 1:nSamples
                for iblk = 1:nblks
                    expctddLdX(1:pt,:,iblk,iSample) = W0T(:,:,iblk)*expctddLdX(1:pt,:,iblk,iSample);
                    expctddLdX(pt+1:pt+pb,:,iblk,iSample) = U0T(:,:,iblk)*expctddLdX(pt+1:pt+pb,:,iblk,iSample);
                end
            end

            expctddLdTheta = zeros(nAngles,nblks,nSamples,datatype);
            for iAngle = 1:nAnglesH
                dW = genW.step(anglesW,mus_,iAngle);
                dU = genU.step(anglesU,mus_,iAngle);
                for iblk = 1:nblks
                    for iSample = 1:nSamples
                        d_top = dW(:,:,iblk)*X(1:pt,:,iblk,iSample);
                        d_btm = dU(:,:,iblk)*X(pt+1:pt+pb,:,iblk,iSample);
                        expctddLdTheta(iAngle,iblk,iSample) = sum(dLdZ(1:pt,:,iblk,iSample).*d_top,'all');
                        expctddLdTheta(nAnglesH+iAngle,iblk,iSample) = sum(dLdZ(pt+1:pt+pb,:,iblk,iSample).*d_btm,'all');
                    end
                end
            end

            % Instantiation of target class
            import tansacnet.salsun.*
            layer = salsunInitialRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'Name','V0',...
                'Mus',mus_,...
                'Device',expctdDevice,...
                'DType',expctdDType);

            % Actual values
            if expctdDevice == "cuda"
                X = gpuArray(X);
                dLdZ = gpuArray(dLdZ);
                Theta = gpuArray(Theta);
            end
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

        function testBackwardWithRandomAnglesDataTypeAndDevice(testCase, ...
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
            tolObj = AbsoluteTolerance(1e-4,single(1e-3));
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
            nSamples = 2;
            nChsTotal = stride;
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            nAngles = (nChsTotal-2)*nChsTotal/4;
            nAnglesH = nAngles/2;
            mus_ = cast(mus,datatype);
            Theta = randn(nAngles,nblks,nSamples,datatype);

            X = randn(nChsTotal,1,nblks,nSamples,datatype);
            dLdZ = randn(nChsTotal,1,nblks,nSamples,datatype);

            % Expected values (reference computed sample by sample)
            expctddLdX = zeros(nChsTotal,1,nblks,nSamples,datatype);
            expctddLdTheta = zeros(nAngles,nblks,nSamples,datatype);
            for iSample = 1:nSamples
                ang = Theta(:,:,iSample);
                anglesW = ang(1:nAnglesH,:);
                anglesU = ang(nAnglesH+1:nAngles,:);
                if nAnglesH == 0
                    W0T = repmat(mus_,[1 1 nblks]);
                    U0T = repmat(mus_,[1 1 nblks]);
                else
                    W0T = permute(genW.step(anglesW,mus_,0),[2 1 3]);
                    U0T = permute(genU.step(anglesU,mus_,0),[2 1 3]);
                end
                dLdZi = dLdZ(:,:,:,iSample);
                for iblk = 1:nblks
                    dLdZi(1:pt,:,iblk) = W0T(:,:,iblk)*dLdZi(1:pt,:,iblk);
                    dLdZi(pt+1:pt+pb,:,iblk) = U0T(:,:,iblk)*dLdZi(pt+1:pt+pb,:,iblk);
                end
                expctddLdX(:,:,:,iSample) = dLdZi;

                % dLdTheta_i = <dLdZ,(dVdTheta_i)X>
                Xi = X(:,:,:,iSample);
                dldzi = dLdZ(:,:,:,iSample);
                for iAngle = 1:nAnglesH
                    dW = genW.step(anglesW,mus_,iAngle);
                    dU = genU.step(anglesU,mus_,iAngle);
                    for iblk = 1:nblks
                        d_top = dW(:,:,iblk)*Xi(1:pt,:,iblk);
                        d_btm = dU(:,:,iblk)*Xi(pt+1:pt+pb,:,iblk);
                        expctddLdTheta(iAngle,iblk,iSample) = sum(dldzi(1:pt,:,iblk).*d_top,'all');
                        expctddLdTheta(nAnglesH+iAngle,iblk,iSample) = sum(dldzi(pt+1:pt+pb,:,iblk).*d_btm,'all');
                    end
                end
            end

            % Instantiation of target class
            import tansacnet.salsun.*
            layer = salsunInitialRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'Name','V0',...
                'Mus',mus_,...
                'Device',device_);

            % Actual values
            if usegpu
                X = gpuArray(X);
                dLdZ = gpuArray(dLdZ);
                Theta = gpuArray(Theta);
            end
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

        function testBackwardGradientCorrectnessByFiniteDifference(testCase, mus)
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            import tansacnet.salsun.*

            nSamples = 2;
            stride_ = 6;
            nChsTotal = stride_;
            nAngles = (nChsTotal-2)*nChsTotal/4;
            nBlks = 2;
            X0 = randn(nChsTotal,1,nBlks,nSamples);
            Theta0 = 0.4*randn(nAngles,nBlks,nSamples);

            layer = salsunInitialRotation1dLayer(...
                'Stride',stride_,...
                'NumberOfBlocks',nBlks,...
                'Name','V0',...
                'Mus',mus);

            Z0 = layer.predict(X0,Theta0);
            dLdZ = randn(size(Z0));
            [dLdX,dLdTheta] = layer.backward(X0,Theta0,Z0,dLdZ,[]);

            h = 1e-5;
            tol = 1e-4;
            loss = @(X,Theta) sum(layer.predict(X,Theta).*dLdZ,'all');

            numGradX = zeros(size(X0));
            for idx = 1:numel(X0)
                Xp = X0; Xp(idx) = Xp(idx)+h;
                Xm = X0; Xm(idx) = Xm(idx)-h;
                numGradX(idx) = (loss(Xp,Theta0)-loss(Xm,Theta0))/(2*h);
            end
            testCase.verifyThat(numGradX,...
                IsEqualTo(dLdX,'Within',AbsoluteTolerance(tol)));

            numGradTheta = zeros(size(Theta0));
            for idx = 1:numel(Theta0)
                Tp = Theta0; Tp(idx) = Tp(idx)+h;
                Tm = Theta0; Tm(idx) = Tm(idx)-h;
                numGradTheta(idx) = (loss(X0,Tp)-loss(X0,Tm))/(2*h);
            end
            testCase.verifyThat(numGradTheta,...
                IsEqualTo(dLdTheta,'Within',AbsoluteTolerance(tol)));
        end

    end

end
