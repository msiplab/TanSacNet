classdef lsunIntermediateRotation1dLayerTestCase < matlab.unittest.TestCase
    %LSUNINTERMEDIATEROTATION1DLAYERTESTCASE
    %
    %   コンポーネント別に入力(nComponents):
    %      nChs x 1 x nBlks x nSamples
    %
    %   コンポーネント別に出力(nComponents):
    %      nChs x 1 x nBlks x nSamples
    %
    %   The expected values are given by an independent implementation
    %   (Givens rotations and central differences) and by the 2-D layer
    %   tansacnet.lsun.lsunIntermediateRotation2dLayer with stride
    %   [1 Stride].
    %
    %   'Termination' of BoundaryOperation fixes the rotation matrix of the
    %   first block to -I.
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
        mode = { 'Analysis', 'Synthesis' };
        boundaryOperation = { 'Circular', 'Termination' };
        usegpu = struct( 'true', true, 'false', false);
    end

    methods (TestClassTeardown)
        function finalCheck(~)
            import tansacnet.lsun.*
            for bo = { 'Circular', 'Termination' }
                layer = lsunIntermediateRotation1dLayer(...
                    'Stride',4,...
                    'NumberOfBlocks',8,...
                    'BoundaryOperation',bo{1},...
                    'Device','cpu');
                fprintf("\n --- Check layer for 1-D sequences (%s) ---\n",bo{1});
                checkLayer(layer,[4 1 8],...
                    'ObservationDimension',4,...
                    'CheckCodegenCompatibility',false)
            end
        end
    end

    methods (Test)

        function testConstructor(testCase, stride)

            % Expected values
            expctdName = 'Vn~';
            expctdMode = 'Synthesis';
            expctdDescription = "Synthesis LSUN intermediate rotation " ...
                + "(ps,pa) = (" + stride/2 + "," + stride/2 + ")";

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunIntermediateRotation1dLayer(...
                'Stride',stride,...
                'Name',expctdName,...
                'Device','cpu');

            % Evaluation
            testCase.verifyEqual(layer.Name,expctdName);
            testCase.verifyEqual(layer.Mode,expctdMode);
            testCase.verifyEqual(layer.Description,expctdDescription);
            testCase.verifyEqual(layer.BoundaryOperation,'Circular');
            testCase.verifyEqual(layer.NumberOfBlocks,1);
            testCase.verifySize(layer.Angles,[(stride-2)*stride/8 1]);
            testCase.verifyEqual(layer.Mus,ones(stride/2,1));
        end

        function testInvalidBoundaryOperation(testCase)
            import tansacnet.lsun.*
            testCase.verifyError(@() lsunIntermediateRotation1dLayer(...
                'Stride',4,'BoundaryOperation','Invalid','Device','cpu'),...
                'LsunLayer:InvalidBoundaryOperation');
        end

        function testInvalidMode(testCase)
            import tansacnet.lsun.*
            layer = lsunIntermediateRotation1dLayer(...
                'Stride',4,'Mode','Invalid','Device','cpu');
            testCase.verifyError(@() layer.predict(randn(4,1,1,1)),...
                'LsunLayer:InvalidMode');
        end

        function testPredictDefault(testCase, usegpu, stride, nblks, mus, datatype)
            % Zero angles give the sign changes of the difference channels

            testCase.assumeGpu_(usegpu,datatype);
            nSamples = 3;
            ps = stride/2;
            X = randn(stride,1,nblks,nSamples,datatype);
            if usegpu
                X = gpuArray(X);
            end

            % Expected values
            expctdZ = gather(X);
            expctdZ(ps+1:end,:,:,:) = mus*expctdZ(ps+1:end,:,:,:);

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunIntermediateRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'Mus',mus,...
                'Device',testCase.device_(usegpu),...
                'DType',datatype);

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

        function testPredictWithRandomAngles(testCase, ...
                usegpu, stride, nblks, mus, mode, boundaryOperation, datatype)

            testCase.assumeGpu_(usegpu,datatype);
            nSamples = 3;
            pa = stride/2;
            nAngles = (stride-2)*stride/8;
            X = randn(stride,1,nblks,nSamples,datatype);
            angles = randn(nAngles,nblks,datatype);
            mus_ = mus*ones(pa,nblks);
            if usegpu
                X = gpuArray(X);
                angles = gpuArray(angles);
            end

            % Expected values
            V = testCase.blockMatrices_(gather(angles),mus_,mode,boundaryOperation);
            expctdZ = zeros(size(X),datatype);
            for iblk = 1:nblks
                expctdZ(:,1,iblk,:) = ...
                    V(:,:,iblk)*reshape(gather(X(:,1,iblk,:)),stride,nSamples);
            end

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunIntermediateRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'Mode',mode,...
                'BoundaryOperation',boundaryOperation,...
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
                usegpu, stride, nblks, mus, mode, boundaryOperation)
            % dLdX is given by the transposed matrices and dLdW agrees
            % with central differences. The boundary block fixed to -I
            % has no angle gradients

            datatype_ = 'double';
            testCase.assumeGpu_(usegpu,datatype_);
            nSamples = 3;
            pa = stride/2;
            nAngles = (stride-2)*stride/8;
            X = randn(stride,1,nblks,nSamples,datatype_);
            dLdZ = randn(stride,1,nblks,nSamples,datatype_);
            angles = randn(nAngles,nblks,datatype_);
            mus_ = mus*ones(pa,nblks);

            % Expected values
            V = testCase.blockMatrices_(angles,mus_,mode,boundaryOperation);
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
                    da = zeros(nAngles,nblks);
                    da(iAngle,iblk) = delta;
                    Vp = testCase.blockMatrices_(angles+da,mus_,mode,boundaryOperation);
                    Vm = testCase.blockMatrices_(angles-da,mus_,mode,boundaryOperation);
                    dV = (Vp(:,:,iblk)-Vm(:,:,iblk))/(2*delta);
                    expctddLdW(iAngle,iblk) = sum(g.*(dV*x),'all');
                end
            end

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunIntermediateRotation1dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nblks,...
                'Mode',mode,...
                'BoundaryOperation',boundaryOperation,...
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
            testCase.verifyEqual(gather(actualdLdX),expctddLdX,'AbsTol',1e-10);
            testCase.verifySize(actualdLdW,[nAngles nblks]);
            testCase.verifyEqual(gather(actualdLdW),expctddLdW,'AbsTol',1e-6);
            if strcmp(boundaryOperation,'Termination')
                testCase.verifyEqual(gather(actualdLdW(:,1)),zeros(nAngles,1));
            end
        end

        function testEquivalenceTo2dLayer(testCase, ...
                usegpu, nblks, mus, mode, boundaryOperation, datatype)
            % The 1-D layer equals the 2-D layer with stride [1 Stride],
            % nRows = 1 and Direction 'Horizontal'

            testCase.assumeGpu_(usegpu,datatype);
            stride_ = 8;
            nSamples = 3;
            nAngles = (stride_-2)*stride_/8;
            X = randn(stride_,1,nblks,nSamples,datatype);
            dLdZ = randn(stride_,1,nblks,nSamples,datatype);
            angles = randn(nAngles,nblks,datatype);
            if usegpu
                X = gpuArray(X);
                dLdZ = gpuArray(dLdZ);
                angles = gpuArray(angles);
            end
            opts = {'Mode',mode,'BoundaryOperation',boundaryOperation,...
                'Device',testCase.device_(usegpu),'DType',datatype};

            % Expected values
            import tansacnet.lsun.*
            layer2d = lsunIntermediateRotation2dLayer(...
                'Stride',[1 stride_],'NumberOfBlocks',[1 nblks],...
                'Direction','Horizontal',opts{:});
            layer2d.Mus = mus;
            layer2d.Angles = angles;
            expctdZ = layer2d.predict(X);
            [expctddLdX,expctddLdW] = layer2d.backward(X,[],dLdZ,[]);

            % Instantiation of target class
            layer = lsunIntermediateRotation1dLayer(...
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

        function testSetBoundaryOperation(testCase, mode)
            % Changing BoundaryOperation updates the rotation matrices

            stride_ = 4;
            nblks_ = 4;
            X = randn(stride_,1,nblks_,2);
            import tansacnet.lsun.*
            layer = lsunIntermediateRotation1dLayer(...
                'Stride',stride_,'NumberOfBlocks',nblks_,...
                'Mode',mode,'Angles',pi/3,'Device','cpu');
            Zc = layer.predict(X);
            layer.BoundaryOperation = 'Termination';
            Zt = layer.predict(X);

            % Evaluation
            testCase.verifyEqual(Zt(3:4,1,1,:),-X(3:4,1,1,:),'AbsTol',1e-10);
            testCase.verifyEqual(Zt(:,1,2:end,:),Zc(:,1,2:end,:),'AbsTol',1e-10);
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

        function V = blockMatrices_(angles,mus,mode,boundaryOperation)
            % Matrices of the layer for every block
            %   Analysis: blkdiag(I,Un), Synthesis: blkdiag(I,Un.')
            %   Un of the first block is -I for 'Termination'
            import tansacnet.testcase.lsun.lsunIntermediateRotation1dLayerTestCase
            pa = size(mus,1);
            nblks = size(mus,2);
            V = zeros(2*pa,2*pa,nblks,'like',angles);
            for iblk = 1:nblks
                Un = lsunIntermediateRotation1dLayerTestCase.orthmtx_(...
                    angles(:,iblk),mus(:,iblk));
                if strcmp(boundaryOperation,'Termination') && iblk == 1
                    Un = -eye(pa,'like',angles);
                end
                if strcmp(mode,'Synthesis')
                    Un = Un.';
                end
                V(:,:,iblk) = blkdiag(eye(pa,'like',angles),Un);
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
