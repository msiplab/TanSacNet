classdef salsunIntermediateRotation1dLayerTestCase < matlab.unittest.TestCase
    %SALSUNINTERMEDIATEROTATION1DLAYERTESTCASE
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
        mode = { 'Analysis', 'Synthesis' };
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
            nAngles = (nChsTotal-2)*nChsTotal/8;
            nBlks = 4;
            layer = salsunIntermediateRotation1dLayer(...
                'Stride',stride_,...
                'NumberOfBlocks',nBlks,...
                'Mode','Analysis',...
                'Name','Un',...
                'Mus',-1);
            % CheckCodegenCompatibility omitted: Theta's observation
            % (sample) dimension is at position 3, not 2 or 4.
            checkLayer(layer,{[nChsTotal 1 nBlks],[nAngles nBlks]},...
                'ObservationDimension',[4 3])
        end

    end

    methods (Test)

        function testConstructor(testCase, stride, mode)

            % Expected values
            expctdName = 'Un';
            expctdMode = mode;
            expctdDescription = mode ...
                + " SA-LSUN intermediate rotation " ...
                + "(pt,pb) = (" ...
                + ceil(stride/2) + "," ...
                + floor(stride/2) + "), "  ...
                + "m = " + stride;

            % Instantiation of target class
            import tansacnet.salsun.*
            layer = salsunIntermediateRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',3,...
                'Mode',mode,...
                'Name',expctdName);

            % Actual values
            actualName = layer.Name;
            actualMode = layer.Mode;
            actualDescription = layer.Description;

            % Evaluation
            testCase.verifyEqual(actualName,expctdName);
            testCase.verifyEqual(actualMode,expctdMode);
            testCase.verifyEqual(actualDescription,expctdDescription);
            testCase.verifyEqual(layer.Stride,stride);
            testCase.verifyEqual(layer.InputNames,{'x','theta'});
        end

        function testConstructorWithDeviceAndDType(testCase, stride, mode, usegpu, datatype)

            % Expected values
            expctdName = 'Un';
            device_ = ["cpu", "cuda"];
            expctdDevice = device_(usegpu+1);
            expctdDType = datatype;

            % Instantiation of target class
            import tansacnet.salsun.*
            layer = salsunIntermediateRotation1dLayer(...
                'Stride',stride,...
                'Name',expctdName,...
                'Mode',mode,...
                'Device',expctdDevice,...
                'DType',expctdDType);

            % Actual values
            actualName = layer.Name;
            actualMode = layer.Mode;
            actualDevice = layer.Device;
            actualDType = layer.DType;

            % Evaluation
            testCase.verifyEqual(actualName,expctdName);
            testCase.verifyEqual(actualMode,mode);
            testCase.verifyEqual(actualDevice,expctdDevice);
            testCase.verifyEqual(actualDType,expctdDType);
        end

        function testInvalidMode(testCase)
            import tansacnet.salsun.*
            nSamples = 2;
            nChsTotal = 4;
            nAngles = (nChsTotal-2)*nChsTotal/8;
            nBlks = 2;
            X = randn(nChsTotal,1,nBlks,nSamples);
            Theta = randn(nAngles,nBlks,nSamples);
            layer = salsunIntermediateRotation1dLayer(...
                'Stride',nChsTotal,...
                'NumberOfBlocks',nBlks,...
                'Mode','Invalid',...
                'Name','Un');
            testCase.verifyError(@() layer.predict(X,Theta),...
                'SaLsunLayer:InvalidMode');
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
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            nAngles = (nChsTotal-2)*nChsTotal/8;
            X = randn(nChsTotal,1,nblks,nSamples,datatype);
            Theta = zeros(nAngles,nblks,nSamples,datatype);
            if usegpu
                X = gpuArray(X);
                Theta = gpuArray(Theta);
            end

            % Expected values
            Un = repmat(eye(pb,datatype),[1 1 nblks nSamples]);
            expctdZ = X;
            for iSample = 1:nSamples
                Yi = X(:,:,:,iSample);
                Uni = Un(:,:,:,iSample);
                for iblk = 1:nblks
                    Yi(pt+1:pt+pb,:,iblk) = Uni(:,:,iblk)*Yi(pt+1:pt+pb,:,iblk);
                end
                expctdZ(:,:,:,iSample) = Yi;
            end

            % Instantiation of target class
            import tansacnet.salsun.*
            layer = salsunIntermediateRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'Mode','Analysis',...
                'Name','Un');

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
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            nAngles = (nChsTotal-2)*nChsTotal/8;
            X = randn(nChsTotal,1,nblks_,nSamples,datatype);
            Theta = zeros(nAngles,nblks_,nSamples,datatype);
            if usegpu
                X = gpuArray(X);
                Theta = gpuArray(Theta);
            end

            % Expected values
            Un = repmat(eye(pb,datatype),[1 1 nblks_ nSamples]);
            expctdZ = X;
            for iSample = 1:nSamples
                Yi = X(:,:,:,iSample);
                Uni = Un(:,:,:,iSample);
                for iblk = 1:nblks_
                    Yi(pt+1:pt+pb,:,iblk) = Uni(:,:,iblk)*Yi(pt+1:pt+pb,:,iblk);
                end
                expctdZ(:,:,:,iSample) = Yi;
            end

            % Instantiation of target class
            import tansacnet.salsun.*
            layer = salsunIntermediateRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks_,...
                'Mode','Analysis',...
                'Name','Un',...
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

        function testPredictRoundTripAnalysisThenSynthesis(testCase, stride, mus)

            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-10);
            
            % Parameters
            nSamples = 2;
            nChsTotal = stride;
            nAngles = (nChsTotal-2)*nChsTotal/8;
            nBlks = 3;
            X = randn(nChsTotal,1,nBlks,nSamples);
            Theta = 0.5*randn(nAngles,nBlks,nSamples);

            import tansacnet.salsun.*
            layerAna = salsunIntermediateRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nBlks,...
                'Mode','Analysis',...
                'Name','Un',...
                'Mus',mus);
            layerSyn = salsunIntermediateRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nBlks,...
                'Mode','Synthesis',...
                'Name','Uns',...
                'Mus',mus);

            Z = layerAna.predict(X,Theta);
            Xrec = layerSyn.predict(Z,Theta);

            testCase.verifyThat(Xrec,IsEqualTo(X,'Within',tolObj));
        end

        function testPredictWithRandomAnglesDataTypeAndDevice(testCase, ...
                usegpu, stride, mode, nblks, mus, datatype)

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
            nSamples = 2;
            nChsTotal = stride;
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            nAngles = (nChsTotal-2)*nChsTotal/8;
            mus_ = mus;
            X = randn(nChsTotal,1,nblks,nSamples,datatype);
            Theta = randn(nAngles,nblks,nSamples);

            isAnalysis = strcmp(mode,'Analysis');

            % Expected values (reference computed sample by sample)
            expctdZ = zeros(nChsTotal,1,nblks,nSamples,datatype);
            for iSample = 1:nSamples
                anglesU = Theta(:,:,iSample);
                if nAngles == 0
                    Un = repmat(mus_,[1 1 nblks]);
                else
                    Un = genU.step(anglesU,mus_);
                end
                if isAnalysis
                    A_ = Un;
                else
                    A_ = permute(Un,[2 1 3]);
                end
                Yi = X(:,:,:,iSample);
                for iblk = 1:nblks
                    Yi(pt+1:pt+pb,:,iblk) = A_(:,:,iblk)*Yi(pt+1:pt+pb,:,iblk);
                end
                expctdZ(:,:,:,iSample) = Yi;
            end

            % Instantiation of target class
            import tansacnet.salsun.*
            layer = salsunIntermediateRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'Mode',mode,...
                'Name','Un',...
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
            genU = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on',...
                'Device','cpu',...
                'DType',datatype);

            % Parameters
            nSamples = 2;
            nChsTotal = stride;
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            nAngles = (nChsTotal-2)*nChsTotal/8;
            mus_ = cast(1,datatype);
            anglesU = zeros(nAngles,nblks,datatype);
            Theta = repmat(anglesU,[1 1 nSamples]);

            X = randn(nChsTotal,1,nblks,nSamples,datatype);
            dLdZ = randn(nChsTotal,1,nblks,nSamples,datatype);

            % Expected values (Mode fixed to 'Analysis')
            if nAngles == 0
                UT_= repmat(mus_,[1 1 nblks]);
            else
                UT_ = permute(genU.step(anglesU,mus_,0),[2 1 3]);
            end
            expctddLdX = dLdZ;
            for iSample = 1:nSamples
                for iblk = 1:nblks
                    expctddLdX(pt+1:pt+pb,:,iblk,iSample) = UT_(:,:,iblk)*expctddLdX(pt+1:pt+pb,:,iblk,iSample);
                end
            end

            % dLdTheta_i = <dLdZ,(dVdTheta_i)X>
            expctddLdTheta = zeros(nAngles,nblks,nSamples,datatype);
            for iAngle = 1:nAngles
                dU_ = genU.step(anglesU,mus_,iAngle);
                for iblk = 1:nblks
                    for iSample = 1:nSamples
                        d_btm = dU_(:,:,iblk)*X(pt+1:pt+pb,:,iblk,iSample);
                        expctddLdTheta(iAngle,iblk,iSample) = sum(dLdZ(pt+1:pt+pb,:,iblk,iSample).*d_btm,'all');
                    end
                end
            end

            % Instantiation of target class
            import tansacnet.salsun.*
            layer = salsunIntermediateRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'Mode','Analysis',...
                'Name','Un',...
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
            genU = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on',...
                'Device','cpu',...
                'DType',datatype);

            % Parameters
            nSamples = 2;
            nChsTotal = stride;
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            nAngles = (nChsTotal-2)*nChsTotal/8;
            mus_ = cast(1,datatype);
            anglesU = zeros(nAngles,nblks,datatype);
            Theta = repmat(anglesU,[1 1 nSamples]);

            X = randn(nChsTotal,1,nblks,nSamples,datatype);
            dLdZ = randn(nChsTotal,1,nblks,nSamples,datatype);

            % Expected values (Mode fixed to 'Analysis')
            if nAngles == 0
                UT_ = repmat(mus_,[1 1 nblks]);
            else
                UT_ = permute(genU.step(anglesU,mus_,0),[2 1 3]);
            end
            expctddLdX = dLdZ;
            for iSample = 1:nSamples
                for iblk = 1:nblks
                    expctddLdX(pt+1:pt+pb,:,iblk,iSample) = UT_(:,:,iblk)*expctddLdX(pt+1:pt+pb,:,iblk,iSample);
                end
            end

            expctddLdTheta = zeros(nAngles,nblks,nSamples,datatype);
            for iAngle = 1:nAngles
                dU_ = genU.step(anglesU,mus_,iAngle);
                for iblk = 1:nblks
                    for iSample = 1:nSamples
                        d_btm = dU_(:,:,iblk)*X(pt+1:pt+pb,:,iblk,iSample);
                        expctddLdTheta(iAngle,iblk,iSample) = sum(dLdZ(pt+1:pt+pb,:,iblk,iSample).*d_btm,'all');
                    end
                end
            end

            % Instantiation of target class
            import tansacnet.salsun.*
            layer = salsunIntermediateRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'Mode','Analysis',...
                'Name','Un',...
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
                usegpu, stride, mode, nblks, mus, datatype)

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
            genU = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on',...
                'Device','cpu',...
                'DType',datatype);

            % Parameters
            nSamples = 2;
            nChsTotal = stride;
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            nAngles = (nChsTotal-2)*nChsTotal/8;
            mus_ = cast(mus,datatype);
            Theta = randn(nAngles,nblks,nSamples,datatype);

            X = randn(nChsTotal,1,nblks,nSamples,datatype);
            dLdZ = randn(nChsTotal,1,nblks,nSamples,datatype);

            isAnalysis = strcmp(mode,'Analysis');

            % Expected values (reference computed sample by sample)
            expctddLdX = zeros(nChsTotal,1,nblks,nSamples,datatype);
            expctddLdTheta = zeros(nAngles,nblks,nSamples,datatype);
            for iSample = 1:nSamples
                anglesU = Theta(:,:,iSample);
                if nAngles == 0
                    Un = repmat(mus_,[1 1 nblks]);
                else
                    Un = genU.step(anglesU,mus_,0);
                end
                if isAnalysis
                    UT_ = permute(Un,[2 1 3]);
                else
                    UT_ = Un;
                end
                dLdZi = dLdZ(:,:,:,iSample);
                for iblk = 1:nblks
                    dLdZi(pt+1:pt+pb,:,iblk) = UT_(:,:,iblk)*dLdZi(pt+1:pt+pb,:,iblk);
                end
                expctddLdX(:,:,:,iSample) = dLdZi;

                % dLdTheta_i = <dLdZ,(dVdTheta_i)X>
                Xi = X(:,:,:,iSample);
                dldzi = dLdZ(:,:,:,iSample);
                for iAngle = 1:nAngles
                    dU = genU.step(anglesU,mus_,iAngle);
                    if isAnalysis
                        dU_ = dU;
                    else
                        dU_ = permute(dU,[2 1 3]);
                    end
                    for iblk = 1:nblks
                        d_btm = dU_(:,:,iblk)*Xi(pt+1:pt+pb,:,iblk);
                        expctddLdTheta(iAngle,iblk,iSample) = sum(dldzi(pt+1:pt+pb,:,iblk).*d_btm,'all');
                    end
                end
            end

            % Instantiation of target class
            import tansacnet.salsun.*
            layer = salsunIntermediateRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'Mode',mode,...
                'Name','Un',...
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

        function testBackwardGradientCorrectnessByFiniteDifference(testCase, mode, mus)
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            import tansacnet.salsun.*

            nSamples = 2;
            stride_ = 6;
            nChsTotal = stride_;
            nAngles = (nChsTotal-2)*nChsTotal/8;
            nBlks = 2;
            X0 = randn(nChsTotal,1,nBlks,nSamples);
            Theta0 = 0.4*randn(nAngles,nBlks,nSamples);

            layer = salsunIntermediateRotation1dLayer(...
                'Stride',stride_,...
                'NumberOfBlocks',nBlks,...
                'Mode',mode,...
                'Name','Un',...
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
