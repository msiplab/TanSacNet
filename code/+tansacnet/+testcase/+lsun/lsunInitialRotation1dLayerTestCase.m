classdef lsunInitialRotation1dLayerTestCase < matlab.unittest.TestCase
    %LSUNINITIALROTATION1DLAYERTESTCASE
    %
    %   コンポーネント別に入力(nComponents):
    %      nChs x 1 x nBlks x nSamples
    %
    %   コンポーネント別に出力(nComponents):
    %      nChs x 1 x nBlks x nSamples
    %
    %   The expected values are given by an independent implementation
    %   (Givens rotations and central differences) and by the 2-D layer
    %   tansacnet.lsun.lsunInitialRotation2dLayer with stride [1 Stride].
    %
    % Requirements: MATLAB R2022b
    %
    % Copyright (c) 2026, Shogo MURAMATSU
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
        nblks = struct('small', 2,'medium', 4);
        datatype = { 'single', 'double' };
        mus = { -1, 1 };
        nodcleak = struct( 'true', true, 'false', false);
        usegpu = struct( 'true', true, 'false', false);
    end

    methods (TestClassTeardown)
        function finalCheck(~)
            import tansacnet.lsun.*
            layer = lsunInitialRotation1dLayer(...
                'Stride',4,...
                'NumberOfBlocks',8,...
                'Device','cpu');
            fprintf("\n --- Check layer for 1-D sequences ---\n");
            checkLayer(layer,[4 1 8],...
                'ObservationDimension',4,...
                'CheckCodegenCompatibility',false)
        end
    end

    methods (Test)

        function testConstructor(testCase, stride)

            % Expected values
            expctdName = 'V0';
            expctdDescription = "LSUN initial rotation " ...
                + "(ps,pa) = (" + stride/2 + "," + stride/2 + "), " ...
                + "m = " + stride;

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunInitialRotation1dLayer(...
                'Stride',stride,...
                'Name',expctdName,...
                'Device','cpu');

            % Evaluation
            testCase.verifyEqual(layer.Name,expctdName);
            testCase.verifyEqual(layer.Description,expctdDescription);
            testCase.verifyEqual(layer.NumberOfBlocks,1);
            testCase.verifySize(layer.Angles,[(stride-2)*stride/4 1]);
            testCase.verifyEqual(layer.Mus,ones(stride,1));
        end

        function testInvalidNumberOfAngles(testCase)
            import tansacnet.lsun.*
            testCase.verifyError(@() lsunInitialRotation1dLayer(...
                'Stride',4,'NumberOfBlocks',2,'Angles',zeros(3,2),...
                'Device','cpu'),?MException);
        end

        function testPredictDefault(testCase, usegpu, stride, nblks, datatype)
            % The default parameters give the identity mapping

            testCase.assumeGpu_(usegpu,datatype);
            nSamples = 3;
            X = randn(stride,1,nblks,nSamples,datatype);
            if usegpu
                X = gpuArray(X);
            end

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunInitialRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'Device',testCase.device_(usegpu),...
                'DType',datatype);

            % Actual values
            actualZ = layer.predict(X);

            % Evaluation
            if usegpu
                testCase.verifyClass(actualZ,'gpuArray');
            end
            testCase.verifyInstanceOf(gather(actualZ),datatype);
            testCase.verifyEqual(gather(actualZ),gather(X),...
                'AbsTol',testCase.tol_(datatype));
        end

        function testPredictWithRandomAngles(testCase, ...
                usegpu, stride, nblks, mus, nodcleak, datatype)

            testCase.assumeGpu_(usegpu,datatype);
            nSamples = 3;
            ps = stride/2;
            nAngles = (stride-2)*stride/4;
            X = randn(stride,1,nblks,nSamples,datatype);
            angles = randn(nAngles,nblks,datatype);
            mus_ = mus*ones(stride,nblks);
            if usegpu
                X = gpuArray(X);
                angles = gpuArray(angles);
            end

            % Expected values
            V = testCase.blockMatrices_(gather(angles),mus_,ps,nodcleak);
            expctdZ = zeros(size(X),datatype);
            for iblk = 1:nblks
                expctdZ(:,1,iblk,:) = ...
                    V(:,:,iblk)*reshape(gather(X(:,1,iblk,:)),stride,nSamples);
            end

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunInitialRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'NoDcLeakage',nodcleak,...
                'Device',testCase.device_(usegpu),...
                'DType',datatype);
            layer.Mus = mus_;
            layer.Angles = angles;

            % Actual values
            actualZ = layer.predict(X);

            % Evaluation
            if usegpu
                testCase.verifyClass(actualZ,'gpuArray');
            end
            testCase.verifyInstanceOf(gather(actualZ),datatype);
            testCase.verifyEqual(gather(actualZ),expctdZ,...
                'AbsTol',testCase.tol_(datatype));
        end

        function testBackwardWithRandomAngles(testCase, ...
                usegpu, stride, nblks, mus, nodcleak)
            % dLdX is given by the transposed matrices and dLdW agrees
            % with central differences

            datatype_ = 'double';
            testCase.assumeGpu_(usegpu,datatype_);
            nSamples = 3;
            ps = stride/2;
            nAngles = (stride-2)*stride/4;
            X = randn(stride,1,nblks,nSamples,datatype_);
            dLdZ = randn(stride,1,nblks,nSamples,datatype_);
            angles = randn(nAngles,nblks,datatype_);
            mus_ = mus*ones(stride,nblks);

            % Expected values
            V = testCase.blockMatrices_(angles,mus_,ps,nodcleak);
            expctddLdX = zeros(size(X),datatype_);
            for iblk = 1:nblks
                expctddLdX(:,1,iblk,:) = ...
                    V(:,:,iblk).'*reshape(dLdZ(:,1,iblk,:),stride,nSamples);
            end
            expctddLdW = zeros(nAngles,nblks,datatype_);
            delta = 1e-6;
            for iblk = 1:nblks
                x = reshape(X(:,1,iblk,:),stride,nSamples);
                g = reshape(dLdZ(:,1,iblk,:),stride,nSamples);
                for iAngle = 1:nAngles
                    if nodcleak && iAngle < ps
                        % Fixed to zero for no DC leakage
                        continue
                    end
                    da = zeros(nAngles,nblks);
                    da(iAngle,iblk) = delta;
                    Vp = testCase.blockMatrices_(angles+da,mus_,ps,nodcleak);
                    Vm = testCase.blockMatrices_(angles-da,mus_,ps,nodcleak);
                    dV = (Vp(:,:,iblk)-Vm(:,:,iblk))/(2*delta);
                    expctddLdW(iAngle,iblk) = sum(g.*(dV*x),'all');
                end
            end

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunInitialRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'NoDcLeakage',nodcleak,...
                'Device',testCase.device_(usegpu),...
                'DType',datatype_);
            layer.Mus = mus_;
            if usegpu
                X = gpuArray(X);
                dLdZ = gpuArray(dLdZ);
                angles = gpuArray(angles);
            end
            layer.Angles = angles;

            % Actual values
            [actualdLdX,actualdLdW] = layer.backward(X,[],dLdZ,[]);

            % Evaluation
            if usegpu
                testCase.verifyClass(actualdLdX,'gpuArray');
                testCase.verifyClass(actualdLdW,'gpuArray');
            end
            actualdLdW = gather(actualdLdW);
            if nodcleak
                % The gradients w.r.t. the fixed angles are not used
                actualdLdW(1:ps-1,:) = 0;
            end
            testCase.verifyEqual(gather(actualdLdX),expctddLdX,'AbsTol',1e-10);
            testCase.verifySize(actualdLdW,[nAngles nblks]);
            testCase.verifyEqual(actualdLdW,expctddLdW,'AbsTol',1e-6);
        end

        function testEquivalenceTo2dLayer(testCase, ...
                usegpu, nblks, mus, nodcleak, datatype)
            % The 1-D layer equals the 2-D layer with stride [1 Stride]
            % and nRows = 1

            testCase.assumeGpu_(usegpu,datatype);
            stride_ = 4;
            nSamples = 3;
            nAngles = (stride_-2)*stride_/4;
            X = randn(stride_,1,nblks,nSamples,datatype);
            dLdZ = randn(stride_,1,nblks,nSamples,datatype);
            angles = randn(nAngles,nblks,datatype);
            if usegpu
                X = gpuArray(X);
                dLdZ = gpuArray(dLdZ);
                angles = gpuArray(angles);
            end
            opts = {'NoDcLeakage',nodcleak,...
                'Device',testCase.device_(usegpu),'DType',datatype};

            % Expected values
            import tansacnet.lsun.*
            layer2d = lsunInitialRotation2dLayer(...
                'Stride',[1 stride_],'NumberOfBlocks',[1 nblks],opts{:});
            layer2d.Mus = mus;
            layer2d.Angles = angles;
            expctdZ = layer2d.predict(X);
            [expctddLdX,expctddLdW] = layer2d.backward(X,[],dLdZ,[]);

            % Instantiation of target class
            layer = lsunInitialRotation1dLayer(...
                'Stride',stride_,'NumberOfBlocks',nblks,opts{:});
            layer.Mus = mus;
            layer.Angles = angles;

            % Actual values
            actualZ = layer.predict(X);
            [actualdLdX,actualdLdW] = layer.backward(X,[],dLdZ,[]);

            % Evaluation
            tol = testCase.tol_(datatype);
            testCase.verifyEqual(gather(actualZ),gather(expctdZ),'AbsTol',tol);
            testCase.verifyEqual(gather(actualdLdX),gather(expctddLdX),'AbsTol',tol);
            testCase.verifyEqual(gather(actualdLdW),gather(expctddLdW),'AbsTol',tol);
        end

    end

    methods (Access = private)

        function assumeGpu_(testCase,usegpu,datatype)
            if usegpu
                testCase.assumeTrue(canUseGPU,'No GPU device was detected.')
                if strcmp(datatype,'double')
                    testCase.assumeTrue(tansacnet.utility.isFp64GpuAvailable(), ...
                        'No GPU device with full-rate FP64 arithmetic was detected.')
                end
            end
        end

    end

    methods (Static, Access = private)

        function device = device_(usegpu)
            if usegpu
                device = 'cuda';
            else
                device = 'cpu';
            end
        end

        function tol = tol_(datatype)
            if strcmp(datatype,'double')
                tol = 1e-10;
            else
                tol = single(1e-4);
            end
        end

        function V = blockMatrices_(angles,mus,ps,nodcleak)
            % Matrices of the layer for every block
            %   Initial: blkdiag(W0,U0), Final: blkdiag(W0.',U0.')
            import tansacnet.testcase.lsun.lsunInitialRotation1dLayerTestCase
            nAngles = size(angles,1);
            nblks = size(angles,2);
            V = zeros(2*ps,2*ps,nblks,'like',angles);
            for iblk = 1:nblks
                anglesW = angles(1:nAngles/2,iblk);
                anglesU = angles(nAngles/2+1:end,iblk);
                musW = mus(1:ps,iblk);
                musU = mus(ps+1:end,iblk);
                if nodcleak
                    anglesW(1:ps-1) = 0;
                    musW(1) = 1;
                end
                W0 = lsunInitialRotation1dLayerTestCase.orthmtx_(anglesW,musW);
                U0 = lsunInitialRotation1dLayerTestCase.orthmtx_(anglesU,musU);
                V(:,:,iblk) = blkdiag(W0,U0);
            end
        end

        function matrix = orthmtx_(angles,mus)
            % Product of Givens rotations followed by sign changes
            nDim = numel(mus);
            matrix = eye(nDim,'like',angles);
            iAng = 1;
            for iTop = 1:nDim-1
                for iBtm = iTop+1:nDim
                    c = cos(angles(iAng));
                    s = sin(angles(iAng));
                    G = eye(nDim,'like',angles);
                    G([iTop iBtm],[iTop iBtm]) = [ c -s ; s c ];
                    matrix = G*matrix;
                    iAng = iAng + 1;
                end
            end
            matrix = diag(mus)*matrix;
        end

    end

end
