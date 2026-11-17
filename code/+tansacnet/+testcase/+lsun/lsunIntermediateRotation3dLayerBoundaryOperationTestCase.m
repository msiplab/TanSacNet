classdef lsunIntermediateRotation3dLayerBoundaryOperationTestCase < matlab.unittest.TestCase
    %LSUNINTERMEDIATEROTATION3DLAYERBOUNDARYOPERATIONTESTCASE
    %
    %   Test cases for the BoundaryOperation option ('Circular' | 'Termination')
    %   of lsunIntermediateRotation3dLayer.
    %
    %   Termination fixes the rotation matrices of the boundary blocks
    %   to -I and excludes them from the angle gradients.
    %   The boundary is specified by Direction:
    %      'Vertical'   : first row blocks
    %      'Horizontal' : first column blocks
    %      'Depth'      : first layer blocks
    %   where the blocks are ordered in column-major order of
    %   nRows x nCols x nLays.
    %
    %   コンポーネント別に入力(nComponents):
    %      nChs x nRows x nCols x nLays x nSamples
    %
    %   コンポーネント別に出力(nComponents):
    %      nChs x nRows x nCols x nLays x nSamples
    %
    % References:
    %   S. Muramatsu, T. Kobayashi, M. Hiki and H. Kikuchi, "Boundary
    %   Operation of 2-D Nonseparable Linear-Phase Paraunitary Filter Banks,"
    %   IEEE Trans. Image Process., vol. 21, no. 4, pp. 2314-2318, 2012.
    %
    %   K. Furuya, S. Hara, K. Seino and S. Muramatsu, "Boundary operation of
    %   2D non-separable oversampled lapped transforms," APSIPA Trans. Signal
    %   Inf. Process., vol. 5, no. 1, pp. 1-9, 2016.
    %
    % Requirements: MATLAB R2022a
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
        stride = { [2 2 2], [1 2 4] };
        datatype = { 'single', 'double' };
        mus = { -1, 1 };
        nrows = struct('small', 2,'medium', 4);
        ncols = struct('small', 2,'medium', 4);
        nlays = struct('small', 2,'medium', 4);
        usegpu = struct( 'true', true, 'false', false);
        mode = { 'Analysis', 'Synthesis' };
        bdir = { 'Vertical', 'Horizontal', 'Depth' };
    end

    methods (TestClassTeardown)
        function finalCheck(~)
            % checkLayer is applied to 'Depth' only, since it takes a
            % long time for 3-D LSUN layers. The other directions are
            % covered by the predict and backward tests.
            import tansacnet.lsun.*
            layer = lsunIntermediateRotation3dLayer(...
                'Stride',[2 2 2],...
                'NumberOfBlocks',[8 8 8],...
                'BoundaryOperation','Termination',...
                'Direction','Depth');
            fprintf("\n --- Check layer for 3-D images (Termination, Depth) ---\n");
            checkLayer(layer,[8 8 8 8],...
                'ObservationDimension',5,...
                'CheckCodegenCompatibility',false)
        end
    end

    methods (Test)

        function testConstructorDefaultBoundaryOperation(testCase, stride)
            % Default BoundaryOperation is 'Circular' (backward compatibility)

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunIntermediateRotation3dLayer(...
                'Stride',stride);

            % Evaluation
            testCase.verifyEqual(layer.BoundaryOperation,'Circular');
        end

        function testConstructorWithBoundaryOperation(testCase, stride, bdir)
            % BoundaryOperation and Direction are set by the constructor

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunIntermediateRotation3dLayer(...
                'Stride',stride,...
                'BoundaryOperation','Termination',...
                'Direction',bdir);

            % Evaluation
            testCase.verifyEqual(layer.BoundaryOperation,'Termination');
            testCase.verifyEqual(layer.Direction,bdir);
        end

        function testInvalidBoundaryOperation(testCase, stride)
            % An invalid BoundaryOperation value throws an exception

            import tansacnet.lsun.*
            testCase.verifyError(@() lsunIntermediateRotation3dLayer(...
                'Stride',stride,...
                'BoundaryOperation','InvalidBoundaryOperation'),...
                'LsunLayer:InvalidBoundaryOperation');
        end

        function testInvalidDirection(testCase, stride)
            % An invalid Direction value for 'Termination' throws an exception

            import tansacnet.lsun.*
            testCase.verifyError(@() lsunIntermediateRotation3dLayer(...
                'Stride',stride,...
                'BoundaryOperation','Termination',...
                'Direction','InvalidDirection'),...
                'LsunLayer:InvalidDirection');
        end

        function testPredictCircular(testCase, ...
                usegpu, stride, nrows, ncols, nlays, mus, mode, bdir, datatype)
            % 'Circular' rotates all blocks by Un(:,:,k), regardless of Direction

            testCase.assumeGpu_(usegpu,datatype);
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            import tansacnet.utility.*
            genU = OrthonormalMatrixGenerationSystem('Device','cpu');

            % Parameters
            nSamples = 8;
            nChsTotal = prod(stride);
            nBlocks = nrows*ncols*nlays;
            X = randn(nChsTotal,nrows,ncols,nlays,nSamples,datatype);
            angles = randn((nChsTotal-2)*nChsTotal/8,nBlocks);

            % Expected values
            Un = genU.step(angles,mus);
            expctdZ = testCase.rotateDifferenceChannels_(X,Un,mode);

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunIntermediateRotation3dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols nlays],...
                'Mode',mode,...
                'BoundaryOperation','Circular',...
                'Direction',bdir,...
                'Device',testCase.device_(usegpu));

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
            end
            testCase.verifyInstanceOf(actualZ,datatype);
            testCase.verifyThat(actualZ,...
                IsEqualTo(expctdZ,'Within',tolObj));
        end

        function testPredictTermination(testCase, ...
                usegpu, stride, nrows, ncols, nlays, mus, mode, bdir, datatype)
            % Boundary blocks: Za = -Ya, other blocks: Za = Un(:,:,k)*Ya
            % (Un(:,:,k).'*Ya in Synthesis mode) with default angles

            testCase.assumeGpu_(usegpu,datatype);
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));

            % Parameters
            nSamples = 8;
            nChsTotal = prod(stride);
            pa = floor(nChsTotal/2);
            nBlocks = nrows*ncols*nlays;
            X = randn(nChsTotal,nrows,ncols,nlays,nSamples,datatype);

            % Expected values
            Un = repmat(mus*eye(pa,datatype),[1 1 nBlocks]);
            bblks = testCase.boundaryBlocks_(nrows,ncols,nlays,bdir);
            Un(:,:,bblks) = repmat(-eye(pa,datatype),[1 1 numel(bblks)]);
            expctdZ = testCase.rotateDifferenceChannels_(X,Un,mode);

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunIntermediateRotation3dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols nlays],...
                'Mode',mode,...
                'Mus',mus,...
                'BoundaryOperation','Termination',...
                'Direction',bdir,...
                'Device',testCase.device_(usegpu));

            % Actual values
            if usegpu
                X = gpuArray(X);
            end
            actualZ = layer.predict(X);

            % Evaluation
            if usegpu
                testCase.verifyClass(actualZ,'gpuArray')
                actualZ = gather(actualZ);
            end
            testCase.verifyInstanceOf(actualZ,datatype);
            testCase.verifyThat(actualZ,...
                IsEqualTo(expctdZ,'Within',tolObj));
        end

        function testPredictTerminationWithRandomAngles(testCase, ...
                usegpu, stride, nrows, ncols, nlays, mode, bdir, datatype)
            % Boundary blocks stay -I for any Angles and Mus

            testCase.assumeGpu_(usegpu,datatype);
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            import tansacnet.utility.*
            genU = OrthonormalMatrixGenerationSystem('Device','cpu');

            % Parameters
            nSamples = 8;
            nChsTotal = prod(stride);
            pa = floor(nChsTotal/2);
            nBlocks = nrows*ncols*nlays;
            X = randn(nChsTotal,nrows,ncols,nlays,nSamples,datatype);
            angles = randn((nChsTotal-2)*nChsTotal/8,nBlocks);
            mus_ = 2*(rand(pa,nBlocks)>0.5)-1;

            % Expected values
            Un = cast(genU.step(angles,mus_),datatype);
            bblks = testCase.boundaryBlocks_(nrows,ncols,nlays,bdir);
            Un(:,:,bblks) = repmat(-eye(pa,datatype),[1 1 numel(bblks)]);
            expctdZ = testCase.rotateDifferenceChannels_(X,Un,mode);

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunIntermediateRotation3dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols nlays],...
                'Mode',mode,...
                'BoundaryOperation','Termination',...
                'Direction',bdir,...
                'Device',testCase.device_(usegpu));

            % Actual values
            if usegpu
                X = gpuArray(X);
                angles = gpuArray(angles);
                mus_ = gpuArray(mus_);
            end
            layer.Mus = mus_;
            layer.Angles = angles;
            actualZ = layer.predict(X);

            % Evaluation
            if usegpu
                testCase.verifyClass(actualZ,'gpuArray')
                actualZ = gather(actualZ);
            end
            testCase.verifyInstanceOf(actualZ,datatype);
            testCase.verifyThat(actualZ,...
                IsEqualTo(expctdZ,'Within',tolObj));
        end

        function testBackwardTermination(testCase, ...
                usegpu, stride, nrows, ncols, nlays, mode, bdir, datatype)
            % dLdX: boundary blocks give -dLdZ, other blocks give
            % Un(:,:,k).'*dLdZ (Un(:,:,k)*dLdZ in Synthesis mode)

            testCase.assumeGpu_(usegpu,datatype);
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-4,single(1e-4));
            import tansacnet.utility.*
            genU = OrthonormalMatrixGenerationSystem(...
                'Device','cpu',...
                'DType',datatype);

            % Parameters
            nSamples = 8;
            nChsTotal = prod(stride);
            pa = floor(nChsTotal/2);
            nBlocks = nrows*ncols*nlays;
            X = randn(nChsTotal,nrows,ncols,nlays,nSamples,datatype);
            dLdZ = randn(nChsTotal,nrows,ncols,nlays,nSamples,datatype);
            angles = randn((nChsTotal-2)*nChsTotal/8,nBlocks,datatype);
            mus_ = cast(2*(rand(pa,nBlocks)>0.5)-1,datatype);

            % Expected values
            % The adjoint of the forward rotation: Un.' in Analysis, Un in Synthesis
            Un = cast(genU.step(angles,mus_),datatype);
            bblks = testCase.boundaryBlocks_(nrows,ncols,nlays,bdir);
            Un(:,:,bblks) = repmat(-eye(pa,datatype),[1 1 numel(bblks)]);
            if strcmp(mode,'Analysis')
                adjointMode = 'Synthesis';
            else
                adjointMode = 'Analysis';
            end
            expctddLdX = testCase.rotateDifferenceChannels_(dLdZ,Un,adjointMode);

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunIntermediateRotation3dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols nlays],...
                'Mode',mode,...
                'BoundaryOperation','Termination',...
                'Direction',bdir,...
                'Device',testCase.device_(usegpu));

            % Actual values
            if usegpu
                X = gpuArray(X);
                dLdZ = gpuArray(dLdZ);
                angles = gpuArray(angles);
                mus_ = gpuArray(mus_);
            end
            layer.Mus = mus_;
            layer.Angles = angles;
            [actualdLdX,~] = layer.backward(X,[],dLdZ,[]);

            % Evaluation
            if usegpu
                testCase.verifyClass(actualdLdX,'gpuArray')
                actualdLdX = gather(actualdLdX);
            end
            testCase.verifyInstanceOf(actualdLdX,datatype);
            testCase.verifyThat(actualdLdX,...
                IsEqualTo(expctddLdX,'Within',tolObj));
        end

        function testBackwardTerminationAngleGradients(testCase, ...
                usegpu, stride, nrows, ncols, nlays, mode, bdir, datatype)
            % dLdW: angle gradients of the boundary blocks are zero and
            % those of the other blocks equal the 'Circular' case

            testCase.assumeGpu_(usegpu,datatype);
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
            nBlocks = nrows*ncols*nlays;
            nAngles = (nChsTotal-2)*nChsTotal/8;
            X = randn(nChsTotal,nrows,ncols,nlays,nSamples,datatype);
            dLdZ = randn(nChsTotal,nrows,ncols,nlays,nSamples,datatype);
            angles = randn(nAngles,nBlocks,datatype);
            mus_ = cast(2*(rand(pa,nBlocks)>0.5)-1,datatype);

            % Expected values
            % dLdWi = <dLdZ,(dVdWi)X> for non-boundary blocks, 0 for boundary blocks
            expctddLdW = zeros(nAngles,nBlocks,datatype);
            c_low = reshape(X(ps+1:ps+pa,:,:,:,:),pa,nBlocks,nSamples);
            dldz_low = reshape(dLdZ(ps+1:ps+pa,:,:,:,:),pa,nBlocks,nSamples);
            for iAngle = 1:nAngles
                dUn = genU.step(angles,mus_,iAngle);
                if strcmp(mode,'Synthesis')
                    dUn = permute(dUn,[2 1 3]);
                end
                for iblk = 1:nBlocks
                    c_low_iblk = dUn(:,:,iblk)*reshape(c_low(:,iblk,:),pa,nSamples);
                    dldz_iblk = reshape(dldz_low(:,iblk,:),pa,nSamples);
                    expctddLdW(iAngle,iblk) = sum(dldz_iblk.*c_low_iblk,'all');
                end
            end
            bblks = testCase.boundaryBlocks_(nrows,ncols,nlays,bdir);
            expctddLdW(:,bblks) = 0;

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunIntermediateRotation3dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols nlays],...
                'Mode',mode,...
                'BoundaryOperation','Termination',...
                'Direction',bdir,...
                'Device',testCase.device_(usegpu));

            % Actual values
            if usegpu
                X = gpuArray(X);
                dLdZ = gpuArray(dLdZ);
                angles = gpuArray(angles);
                mus_ = gpuArray(mus_);
            end
            layer.Mus = mus_;
            layer.Angles = angles;
            [~,actualdLdW] = layer.backward(X,[],dLdZ,[]);

            % Evaluation
            if usegpu
                testCase.verifyClass(actualdLdW,'gpuArray')
                actualdLdW = gather(actualdLdW);
            end
            testCase.verifyInstanceOf(actualdLdW,datatype);
            testCase.verifyThat(actualdLdW,...
                IsEqualTo(expctddLdW,'Within',tolObj));
        end

        function testPredictTerminationSetAfterConstruction(testCase, ...
                stride, nrows, ncols, nlays, mode, bdir)
            % Setting BoundaryOperation and Direction after construction
            % takes effect on the next predict

            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            import tansacnet.utility.*
            genU = OrthonormalMatrixGenerationSystem('Device','cpu');

            % Parameters
            nSamples = 8;
            nChsTotal = prod(stride);
            pa = floor(nChsTotal/2);
            nBlocks = nrows*ncols*nlays;
            X = randn(nChsTotal,nrows,ncols,nlays,nSamples);
            angles = randn((nChsTotal-2)*nChsTotal/8,nBlocks);
            mus_ = 2*(rand(pa,nBlocks)>0.5)-1;

            % Expected values
            Un = genU.step(angles,mus_);
            bblks = testCase.boundaryBlocks_(nrows,ncols,nlays,bdir);
            Un(:,:,bblks) = repmat(-eye(pa),[1 1 numel(bblks)]);
            expctdZ = testCase.rotateDifferenceChannels_(X,Un,mode);

            % Instantiation of target class with 'Circular'
            % Un is generated for 'Circular' at the end of construction
            import tansacnet.lsun.*
            layer = lsunIntermediateRotation3dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols nlays],...
                'Mode',mode,...
                'Angles',angles,...
                'Mus',mus_,...
                'Device','cpu');

            % Switch to 'Termination' after construction
            layer.Direction = bdir;
            layer.BoundaryOperation = 'Termination';

            % Actual values
            actualZ = layer.predict(X);

            % Evaluation
            testCase.verifyThat(actualZ,...
                IsEqualTo(expctdZ,'Within',tolObj));
        end

        function testSetInvalidBoundaryOperation(testCase, stride)
            % Setting an invalid BoundaryOperation after construction
            % throws an exception

            import tansacnet.lsun.*
            layer = lsunIntermediateRotation3dLayer(...
                'Stride',stride,...
                'Device','cpu');
            testCase.verifyError(@() testCase.setProperty_(layer,...
                'BoundaryOperation','InvalidBoundaryOperation'),...
                'LsunLayer:InvalidBoundaryOperation');
        end

        function testSetInvalidDirection(testCase, stride)
            % Setting an invalid Direction after construction throws an
            % exception

            import tansacnet.lsun.*
            layer = lsunIntermediateRotation3dLayer(...
                'Stride',stride,...
                'Device','cpu');
            testCase.verifyError(@() testCase.setProperty_(layer,...
                'Direction','InvalidDirection'),...
                'LsunLayer:InvalidDirection');
        end

        function testPredictTerminationWithoutDirection(testCase, stride)
            % Switching to 'Termination' without Direction after
            % construction throws an exception on predict

            % Parameters
            nBlocks_ = [2 2 2];
            nSamples = 1;
            nChsTotal = prod(stride);
            X = zeros([nChsTotal nBlocks_ nSamples]);

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunIntermediateRotation3dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',nBlocks_,...
                'Device','cpu');
            layer.BoundaryOperation = 'Termination';

            % Evaluation
            testCase.verifyError(@() layer.predict(X),...
                'LsunLayer:InvalidDirection');
        end

    end

    methods (Access = private)

        function assumeGpu_(testCase,usegpu,datatype)
            if usegpu
                testCase.assumeTrue(tansacnet.utility.isGpuAvailable(datatype), ...
                    'No GPU device supporting '+string(datatype)+' was detected.')
            end
        end

    end

    methods (Static, Access = private)

        function layer = setProperty_(layer,name,value)
            % Set a property of a layer (value object) in a function
            layer.(name) = value;
        end

        function bblks = boundaryBlocks_(nrows,ncols,nlays,bdir)
            % Indices of the boundary blocks in column-major block order
            % of nRows x nCols x nLays
            mask = false(nrows,ncols,nlays);
            if strcmp(bdir,'Vertical')
                mask(1,:,:) = true;   % first row
            elseif strcmp(bdir,'Horizontal')
                mask(:,1,:) = true;   % first column
            elseif strcmp(bdir,'Depth')
                mask(:,:,1) = true;   % first layer
            else
                error('Invalid direction: %s',bdir)
            end
            bblks = find(mask(:)).';
        end

        function Z = rotateDifferenceChannels_(X,Un,mode)
            % Rotate the difference channels block-wise by
            % Un(:,:,k) in Analysis mode or Un(:,:,k).' in Synthesis mode
            nChsTotal = size(X,1);
            nrows_ = size(X,2);
            ncols_ = size(X,3);
            nlays_ = size(X,4);
            nSamples = size(X,5);
            nBlocks = nrows_*ncols_*nlays_;
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            if strcmp(mode,'Synthesis')
                Un = permute(Un,[2 1 3]);
            end
            Un = cast(Un,'like',X);
            Z = X;
            Ya = reshape(X(ps+1:ps+pa,:,:,:,:),pa,nBlocks,nSamples);
            Za = zeros(size(Ya),'like',Ya);
            for iSample = 1:nSamples
                for iblk = 1:nBlocks
                    Za(:,iblk,iSample) = Un(:,:,iblk)*Ya(:,iblk,iSample);
                end
            end
            Z(ps+1:ps+pa,:,:,:,:) = reshape(Za,pa,nrows_,ncols_,nlays_,nSamples);
        end

        function device = device_(usegpu)
            if usegpu
                device = "cuda";
            else
                device = "cpu";
            end
        end

    end

end
