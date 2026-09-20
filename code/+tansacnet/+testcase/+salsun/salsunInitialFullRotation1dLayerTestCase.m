classdef salsunInitialFullRotation1dLayerTestCase < matlab.unittest.TestCase
    %SALSUNINITIALFULLROTATION1DLAYERTESTCASE
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
            nAngles = (nChsTotal-1)*nChsTotal/2;
            nBlks = 4;
            layer = salsunInitialFullRotation1dLayer('Name','V0',...
                'Stride',stride_,...
                'NumberOfBlocks',nBlks);
            % CheckCodegenCompatibility omitted: Theta's observation
            % (sample) dimension is at position 3, not 2 or 4, which
            % checkLayer's code generation compatibility check does not
            % support -- the rest of checkLayer's interface and
            % gradient-consistency validation still applies.
            checkLayer(layer,{[nChsTotal 1 nBlks],[nAngles nBlks]},...
                'ObservationDimension',[4 3])
        end

    end

    methods (Test)

        function testConstructor(testCase, stride)
            
            % Expected values
            expctdName = 'V0';
            expctdDescription = "SA-LSUN initial full rotation" ...
                + "(pt,pb) = (" ...
                + ceil(prod(stride)/2) + "," ...
                + floor(prod(stride)/2) + "), "  ...
                + "m = " + stride;
            
            % Instantiation of target class
            import tansacnet.salsun.*
            nChsTotal = stride;
            expctdPt = ceil(nChsTotal/2);
            expctdPb = floor(nChsTotal/2);

            layer = salsunInitialFullRotation1dLayer(...
                'Stride',stride,...
                'Name','V0',...
                'NumberOfBlocks',3);

            testCase.verifyEqual(layer.Name,expctdName);
            testCase.verifyEqual(layer.Description,expctdDescription);
            testCase.verifyEqual(layer.Stride,stride);
            testCase.verifyEqual(layer.InputNames,{'x','theta'});
            testCase.verifyEqual(size(layer.Mus),[expctdPt+expctdPb 3]);
            testCase.verifyEqual(layer.Mus,ones(expctdPt+expctdPb,3));
        end

        function testConstructorWithDeviceAndDType(testCase, stride, usegpu, datatype)

            % Expected values
            expctdName = 'V0';

            device_ = ["cpu", "cuda"];
            expctdDevice = device_(usegpu+1);
            expctdDType = datatype;

            % Instantiation of target class
            import tansacnet.salsun.*
            layer = salsunInitialFullRotation1dLayer(...
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

        function testPredictWithZeroAngles(testCase,...
                usegpu, stride, nblks, datatype)

            if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end

            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));

            % parameters
            nSamples = 2;
            nChsTotal = stride;

            nAngles = (nChsTotal-1)*nChsTotal/2;
            X = randn(nChsTotal,1,nblks,nSamples,datatype);
            Theta = zeros(nAngles,nblks,nSamples,datatype);
            if usegpu
                X = gpuArray(X);
                Theta = gpuArray(Theta);
            end

            % Expected values: with zero angles, V0 = diag(mus)
            % (all-ones by default), so the layer must act as the
            % identity.
            % nChs x 1 x nBlks x nSamples
            expctdZ = X;

            % Instantiation of target class
            import tansacnet.salsun.*
            layer = salsunInitialFullRotation1dLayer(...
                'Name','V0',...
                'Stride',stride,...
                'NumberOfBlocks',nblks);

            % Actual values
            actualZ = layer.predict(X,Theta);

            if usegpu
                testCase.verifyClass(actualZ,'gpuArray')
                actualZ = gather(actualZ);
                expctdZ = gather(expctdZ);
            end
            testCase.verifyInstanceOf(actualZ,datatype);
            testCase.verifyThat(actualZ,IsEqualTo(expctdZ,'Within',tolObj));
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
            nSamples = 2;
            nChsTotal = stride;

            nAngles = (nChsTotal-1)*nChsTotal/2;
            X = randn(nChsTotal,1,nblks,nSamples,datatype);
            Theta = zeros(nAngles,nblks,nSamples,datatype);
            if expctdDevice == "cuda"
                X = gpuArray(X);
                Theta = gpuArray(Theta);
            end

            % Expected values: with zero angles, V0 = diag(mus)
            % (all-ones by default), so the layer must act as the
            % identity.
            expctdZ = X;

            % Instantiation of target class
            import tansacnet.salsun.*
            layer = salsunInitialFullRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
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

        function testPredictPreservesEnergy(testCase, stride)
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-10);

            % Parameters
            nSamples = 2;
            nChsTotal = stride;
            nAngles = (nChsTotal-1)*nChsTotal/2;
            nBlks = 3;

            X = randn(nChsTotal,1,nBlks,nSamples);
            Theta = 0.5*randn(nAngles,nBlks,nSamples);

            import tansacnet.salsun.*
            layer = salsunInitialFullRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nBlks,...
                'Name','V0');
            Z = layer.predict(X,Theta);

            energyX = sum(X.^2,1);
            energyZ = sum(Z.^2,1);
            testCase.verifyThat(energyZ,IsEqualTo(energyX,'Within',tolObj));
        end

        function testPredictWithRandomAnglesDataTypeAndDevice(testCase, ...
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
            nSamples = 2;
            nChsTotal = stride;

            nAngles = (nChsTotal-1)*nChsTotal/2;
            X = randn(nChsTotal,1,nblks,nSamples,datatype);
            Theta = randn(nAngles,nblks,nSamples);

            % Expected values (reference computed sample by sample)
            expctdZ = zeros(nChsTotal,1,nblks,nSamples,datatype);
            for iSample = 1:nSamples
                V0 = gen.step(Theta(:,:,iSample),1);
                Xi = X(:,:,:,iSample);
                for iblk = 1:nblks
                    Xi(:,:,iblk) = V0(:,:,iblk)*Xi(:,:,iblk);
                end
                expctdZ(:,:,:,iSample) = Xi;
            end

            % Instantiation of target class
            import tansacnet.salsun.*
            layer = salsunInitialFullRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'Name','V0',...
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
            gen = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on',...
                'Device','cpu',...
                'DType',datatype);

            % Parameters
            nSamples = 2;
            nChsTotal = stride;
            nAngles = (nChsTotal-1)*nChsTotal/2;
            angles = zeros(nAngles,nblks,datatype);
            mus_ = cast(1,datatype);
            Theta = repmat(angles,[1 1 nSamples]);

            X = randn(nChsTotal,1,nblks,nSamples,datatype);
            dLdZ = randn(nChsTotal,1,nblks,nSamples,datatype);

            % Expected values (dLdX = dZdX x dLdZ: predict applies V0, so
            % backward applies V0')
            V0T = permute(gen.step(angles,mus_,0),[2 1 3]);
            expctddLdX = dLdZ;
            for iSample = 1:nSamples
                for iblk = 1:nblks
                    expctddLdX(:,:,iblk,iSample) = V0T(:,:,iblk)*expctddLdX(:,:,iblk,iSample);
                end
            end

            % dLdTheta_i = <dLdZ,(dVdTheta_i)X>
            expctddLdTheta = zeros(nAngles,nblks,nSamples,datatype);
            for iAngle = 1:nAngles
                dV0 = gen.step(angles,mus_,iAngle);
                for iblk = 1:nblks
                    for iSample = 1:nSamples
                        d_ = dV0(:,:,iblk)*X(:,:,iblk,iSample);
                        expctddLdTheta(iAngle,iblk,iSample) = sum(dLdZ(:,:,iblk,iSample).*d_,'all');
                    end
                end
            end

            % Instantiation of target class
            import tansacnet.salsun.*
            layer = salsunInitialFullRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'Name','V0',...
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
            gen = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on',...
                'Device','cpu',...
                'DType',datatype);

            % Parameters
            nSamples = 2;
            nChsTotal = stride;
            nAngles = (nChsTotal-1)*nChsTotal/2;
            angles = zeros(nAngles,nblks,datatype);
            mus_ = cast(1,datatype);
            Theta = repmat(angles,[1 1 nSamples]);

            X = randn(nChsTotal,1,nblks,nSamples,datatype);
            dLdZ = randn(nChsTotal,1,nblks,nSamples,datatype);

            % Expected values
            V0T = permute(gen.step(angles,mus_,0),[2 1 3]);
            expctddLdX = dLdZ;
            for iSample = 1:nSamples
                for iblk = 1:nblks
                    expctddLdX(:,:,iblk,iSample) = V0T(:,:,iblk)*expctddLdX(:,:,iblk,iSample);
                end
            end

            expctddLdTheta = zeros(nAngles,nblks,nSamples,datatype);
            for iAngle = 1:nAngles
                dV0 = gen.step(angles,mus_,iAngle);
                for iblk = 1:nblks
                    for iSample = 1:nSamples
                        d_ = dV0(:,:,iblk)*X(:,:,iblk,iSample);
                        expctddLdTheta(iAngle,iblk,iSample) = sum(dLdZ(:,:,iblk,iSample).*d_,'all');
                    end
                end
            end

            % Instantiation of target class
            import tansacnet.salsun.*
            layer = salsunInitialFullRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'Name','V0',...
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
            tolObj = AbsoluteTolerance(1e-4,single(1e-3));
            import tansacnet.utility.*
            gen = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on',...
                'Device','cpu',...
                'DType',datatype);

            % Parameters
            nSamples = 2;
            nChsTotal = stride;
            nAngles = (nChsTotal-1)*nChsTotal/2;
            Theta = randn(nAngles,nblks,nSamples,datatype);
            mus_ = cast(1,datatype);

            X = randn(nChsTotal,1,nblks,nSamples,datatype);
            dLdZ = randn(nChsTotal,1,nblks,nSamples,datatype);

            % Expected values (reference computed sample by sample)
            expctddLdX = zeros(nChsTotal,1,nblks,nSamples,datatype);
            expctddLdTheta = zeros(nAngles,nblks,nSamples,datatype);
            for iSample = 1:nSamples
                ang = Theta(:,:,iSample);

                % dLdX = dZdX x dLdZ (predict applies V0, so backward
                % applies V0')
                V0T = permute(gen.step(ang,mus_,0),[2 1 3]);
                dLdZi = dLdZ(:,:,:,iSample);
                for iblk = 1:nblks
                    dLdZi(:,:,iblk) = V0T(:,:,iblk)*dLdZi(:,:,iblk);
                end
                expctddLdX(:,:,:,iSample) = dLdZi;

                % dLdTheta_i = <dLdZ,(dVdTheta_i)X>
                Xi = X(:,:,:,iSample);
                dldzi = dLdZ(:,:,:,iSample);
                for iAngle = 1:nAngles
                    dV0 = gen.step(ang,mus_,iAngle);
                    for iblk = 1:nblks
                        d_ = dV0(:,:,iblk)*Xi(:,:,iblk);
                        expctddLdTheta(iAngle,iblk,iSample) = sum(dldzi(:,:,iblk).*d_,'all');
                    end
                end
            end

            % Instantiation of target class
            import tansacnet.salsun.*
            layer = salsunInitialFullRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'Name','V0',...
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

    end

end
