classdef salsunInitialRotation2dLayerTestCase < matlab.unittest.TestCase
    %SALSUNINITIALROTATION2DLAYERTESTCASE
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
            nAngles = (nChsTotal-2)*nChsTotal/4;
            layer = salsunInitialRotation2dLayer(...
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
            expctdName = 'V0~';
            expctdDescription = "SA-LSUN initial rotation " ...
                + "(ps,pa) = (" ...
                + ceil(prod(stride)/2) + "," + floor(prod(stride)/2) + "), " ...
                + "(mv,mh) = (" + stride(1) + "," + stride(2) + ")";

            % Instantiation of target class
            import tansacnet.salsun.*
            layer = salsunInitialRotation2dLayer(...
                'Stride',stride,...
                'Name',expctdName);

            % Actual values
            actualName = layer.Name;
            actualDescription = layer.Description;
            actualInputNames = layer.InputNames;
            actualMode = layer.Mode;

            % Evaluation
            testCase.verifyEqual(actualName,expctdName);
            testCase.verifyEqual(actualDescription,expctdDescription);
            testCase.verifyEqual(actualInputNames,{'x','theta'});
            testCase.verifyEqual(actualMode,'Analysis');
        end

        function testConstructorWithDeviceAndDType(testCase, stride, usegpu, datatype)

            % Expected values
            expctdName = 'V0~';

            device_ = ["cpu", "cuda"];
            expctdDevice = device_(usegpu+1);
            expctdDType = datatype;

            % Instantiation of target class
            import tansacnet.salsun.*
            layer = salsunInitialRotation2dLayer(...
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

        function testPredictGrayscaleWithZeroAngles(testCase, ...
                usegpu, stride, nrows, ncols, datatype)

            if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end

            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));

            % Parameters
            nSamples = 8;
            nDecs = prod(stride);
            nChsTotal = nDecs;
            nBlks = nrows*ncols;
            nAnglesH = (nChsTotal-2)*nChsTotal/8;

            X = randn(nDecs,nrows,ncols,nSamples,datatype);
            % With Theta == 0 and default Mus == 1, W0 and U0 are both
            % identity, so the layer must act as a pass-through.
            Theta = zeros(2*nAnglesH,nBlks,nSamples,datatype);
            if usegpu
                X = gpuArray(X);
                Theta = gpuArray(Theta);
            end

            % Expected values
            % nChs x nRows x nCols x nSamples
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            W0 = repmat(eye(ps,datatype),[1 1 nrows*ncols]);
            U0 = repmat(eye(pa,datatype),[1 1 nrows*ncols]);
            %expctdZ = zeros(nrows,ncols,nChsTotal,nSamples,datatype);
            expctdZ = zeros(nChsTotal,nrows,ncols,nSamples,datatype);
            Y  = zeros(nChsTotal,nrows,ncols,datatype);
            for iSample=1:nSamples
                % Perumation in each block
                Ai = X(:,:,:,iSample); %permute(X(:,:,:,iSample),[3 1 2]);
                Yi = reshape(Ai,nDecs,nrows,ncols);
                %
                Ys = Yi(1:ps,:);
                Ya = Yi(ps+1:end,:);
                for iblk = 1:(nrows*ncols)
                    Ys(:,iblk) = W0(:,1:ps,iblk)*Ys(:,iblk);
                    Ya(:,iblk) = U0(:,1:pa,iblk)*Ya(:,iblk);
                end
                Y(1:ps,:,:) = reshape(Ys,ps,nrows,ncols);
                Y(ps+1:ps+pa,:,:) = reshape(Ya,pa,nrows,ncols);                
                expctdZ(:,:,:,iSample) = Y; %ipermute(Y,[3 1 2]);
            end

            % Instantiation of target class
            import tansacnet.salsun.*
            layer = salsunInitialRotation2dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols],...
                'Name','V0~');

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

        function testPredictGrayscaleWithDeviceAndDType(testCase, ...
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
            nrows_ = 2;
            ncols_ = 2;
            nSamples = 8;
            nDecs = prod(stride);
            nChsTotal = nDecs;
            nBlks = nrows_*ncols_;
            nAnglesH = (nChsTotal-2)*nChsTotal/8;

            X = randn(nDecs,nrows_,ncols_,nSamples,datatype);
            Theta = zeros(2*nAnglesH,nBlks,nSamples,datatype);
            if usegpu
                X = gpuArray(X);
                Theta = gpuArray(Theta);
            end

            % Expected values
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            W0 = repmat(eye(ps,datatype),[1 1 nrows_*ncols_]);
            U0 = repmat(eye(pa,datatype),[1 1 nrows_*ncols_]);
            %expctdZ = zeros(nrows,ncols,nChsTotal,nSamples,datatype);
            expctdZ = zeros(nChsTotal,nrows_,ncols_,nSamples,datatype);
            Y  = zeros(nChsTotal,nrows_,ncols_,datatype);
            for iSample=1:nSamples
                % Perumation in each block
                Ai = X(:,:,:,iSample); %permute(X(:,:,:,iSample),[3 1 2]);
                Yi = reshape(Ai,nDecs,nrows_,ncols_);
                %
                Ys = Yi(1:ps,:);
                Ya = Yi(ps+1:end,:);
                for iblk = 1:(nrows_*ncols_)
                    Ys(:,iblk) = W0(:,1:ps,iblk)*Ys(:,iblk);
                    Ya(:,iblk) = U0(:,1:pa,iblk)*Ya(:,iblk);
                end
                Y(1:ps,:,:) = reshape(Ys,ps,nrows_,ncols_);
                Y(ps+1:ps+pa,:,:) = reshape(Ya,pa,nrows_,ncols_);                
                expctdZ(:,:,:,iSample) = Y; %ipermute(Y,[3 1 2]);
            end

            % Instantiation of target class
            import tansacnet.salsun.*
            layer = salsunInitialRotation2dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows_ ncols_],...
                'Name','V0~',...
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
            genW = OrthonormalMatrixGenerationSystem('Device','cpu');
            genU = OrthonormalMatrixGenerationSystem('Device','cpu');

            % Parameters
            nSamples = 8;
            nDecs = prod(stride);
            nChsTotal = nDecs;
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            nBlks = nrows*ncols;
            nAnglesH = (nChsTotal-2)*nChsTotal/8;

            % nChsTotal x nRows x nCols x nSamples
            X = randn(nDecs,nrows,ncols,nSamples,datatype);
            % Angles vary block by block AND sample by sample (the
            % SA-LSUN feature under test).
            anglesW = randn(nAnglesH,nBlks,nSamples);
            anglesU = randn(nAnglesH,nBlks,nSamples);
            Theta = cat(1,anglesW,anglesU);

            % Expected values (reference computed sample by sample)
            expctdZ = zeros(nChsTotal,nrows,ncols,nSamples,datatype);
            for iSample = 1:nSamples
                W0 = genW.step(anglesW(:,:,iSample),mus);
                U0 = genU.step(anglesU(:,:,iSample),mus);
                Xi = reshape(X(:,:,:,iSample),nDecs,nBlks);
                Ys = Xi(1:ps,:);
                Ya = Xi(ps+1:end,:);
                for iblk = 1:nBlks
                    Ys(:,iblk) = W0(:,1:ps,iblk)*Ys(:,iblk);
                    Ya(:,iblk) = U0(:,1:pa,iblk)*Ya(:,iblk);
                end
                expctdZ(:,:,:,iSample) = reshape([Ys;Ya],nChsTotal,nrows,ncols);
            end

            % Instantiation of target class
            import tansacnet.salsun.*
            layer = salsunInitialRotation2dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols],...
                'Name','V0~',...
                'Device',device_);

            % Actual values
            if usegpu
                X = gpuArray(X);
                Theta = gpuArray(Theta);
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

        function testBackwardGrayscaleWithZeroAngles(testCase, ...
                usegpu, stride, nrows, ncols, datatype)

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
            nDecs = prod(stride);
            nChsTotal = nDecs;
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            nBlks = nrows*ncols;
            nAnglesH = (nChsTotal-2)*nChsTotal/8;
            angW = zeros(nAnglesH,nBlks,datatype);
            angU = zeros(nAnglesH,nBlks,datatype);
            mus_ = cast(1,datatype);
            Theta = repmat(cat(1,angW,angU),[1 1 nSamples]);

            X = randn(nDecs,nrows,ncols,nSamples,datatype);
            dLdZ = randn(nDecs,nrows,ncols,nSamples,datatype);

            % Expected values
            % dLdX = dZdX x dLdZ
            W0T = permute(genW.step(angW,mus_,0),[2 1 3]);
            U0T = permute(genU.step(angU,mus_,0),[2 1 3]);
            expctddLdX = zeros(nDecs,nrows,ncols,nSamples,datatype);
            expctddLdW = zeros(nAnglesH,nBlks,nSamples,datatype);
            expctddLdU = zeros(nAnglesH,nBlks,nSamples,datatype);
            for iSample = 1:nSamples
                dLdZi = reshape(dLdZ(:,:,:,iSample),nDecs,nBlks);
                Ys = dLdZi(1:ps,:);
                Ya = dLdZi(ps+1:end,:);
                for iblk = 1:nBlks
                    Ys(:,iblk) = W0T(1:ps,:,iblk)*Ys(:,iblk);
                    Ya(:,iblk) = U0T(1:pa,:,iblk)*Ya(:,iblk);
                end
                expctddLdX(:,:,:,iSample) = reshape([Ys;Ya],nDecs,nrows,ncols);

                % dLdWi = <dLdZ,(dVdWi)X>
                Xi = reshape(X(:,:,:,iSample),nDecs,nBlks);
                c_upp = Xi(1:ps,:);
                c_low = Xi(ps+1:end,:);
                dldz_upp = dLdZi(1:ps,:);
                dldz_low = dLdZi(ps+1:end,:);
                for iAngle = 1:nAnglesH
                    dW0 = genW.step(angW,mus_,iAngle);
                    dU0 = genU.step(angU,mus_,iAngle);
                    for iblk = 1:nBlks
                        d_upp = dW0(:,1:ps,iblk)*c_upp(:,iblk);
                        d_low = dU0(:,1:pa,iblk)*c_low(:,iblk);
                        expctddLdW(iAngle,iblk,iSample) = sum(dldz_upp(:,iblk).*d_upp,'all');
                        expctddLdU(iAngle,iblk,iSample) = sum(dldz_low(:,iblk).*d_low,'all');
                    end
                end
            end
            expctddLdTheta = cat(1,expctddLdW,expctddLdU);

            % Instantiation of target class
            import tansacnet.salsun.*
            layer = salsunInitialRotation2dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols],...
                'Name','V0~',...
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

            % Parameters
            nrows_ = 8;
            ncols_ = 8;
            nSamples = 8;
            nDecs = prod(stride);
            nChsTotal = nDecs;
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            nBlks = nrows_*ncols_;
            nAnglesH = (nChsTotal-2)*nChsTotal/8;
            angW = zeros(nAnglesH,nBlks,datatype);
            angU = zeros(nAnglesH,nBlks,datatype);
            mus_ = cast(1,datatype);
            Theta = repmat(cat(1,angW,angU),[1 1 nSamples]);

            X = randn(nDecs,nrows_,ncols_,nSamples,datatype);
            dLdZ = randn(nDecs,nrows_,ncols_,nSamples,datatype);

            % Expected values
            W0T = permute(genW.step(angW,mus_,0),[2 1 3]);
            U0T = permute(genU.step(angU,mus_,0),[2 1 3]);
            expctddLdX = zeros(nDecs,nrows_,ncols_,nSamples,datatype);
            expctddLdW = zeros(nAnglesH,nBlks,nSamples,datatype);
            expctddLdU = zeros(nAnglesH,nBlks,nSamples,datatype);
            for iSample = 1:nSamples
                dLdZi = reshape(dLdZ(:,:,:,iSample),nDecs,nBlks);
                Ys = dLdZi(1:ps,:);
                Ya = dLdZi(ps+1:end,:);
                for iblk = 1:nBlks
                    Ys(:,iblk) = W0T(1:ps,:,iblk)*Ys(:,iblk);
                    Ya(:,iblk) = U0T(1:pa,:,iblk)*Ya(:,iblk);
                end
                expctddLdX(:,:,:,iSample) = reshape([Ys;Ya],nDecs,nrows_,ncols_);

                Xi = reshape(X(:,:,:,iSample),nDecs,nBlks);
                c_upp = Xi(1:ps,:);
                c_low = Xi(ps+1:end,:);
                dldz_upp = dLdZi(1:ps,:);
                dldz_low = dLdZi(ps+1:end,:);
                for iAngle = 1:nAnglesH
                    dW0 = genW.step(angW,mus_,iAngle);
                    dU0 = genU.step(angU,mus_,iAngle);
                    for iblk = 1:nBlks
                        d_upp = dW0(:,1:ps,iblk)*c_upp(:,iblk);
                        d_low = dU0(:,1:pa,iblk)*c_low(:,iblk);
                        expctddLdW(iAngle,iblk,iSample) = sum(dldz_upp(:,iblk).*d_upp,'all');
                        expctddLdU(iAngle,iblk,iSample) = sum(dldz_low(:,iblk).*d_low,'all');
                    end
                end
            end
            expctddLdTheta = cat(1,expctddLdW,expctddLdU);

            % Instantiation of target class
            import tansacnet.salsun.*
            layer = salsunInitialRotation2dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows_ ncols_],...
                'Name','V0~',...
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
            nDecs = prod(stride);
            nChsTotal = nDecs;
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            nBlks = nrows*ncols;
            nAnglesH = (nChsTotal-2)*nChsTotal/8;
            anglesW = randn(nAnglesH,nBlks,nSamples,datatype);
            anglesU = randn(nAnglesH,nBlks,nSamples,datatype);
            Theta = cat(1,anglesW,anglesU);
            mus_ = cast(mus,datatype);

            X = randn(nDecs,nrows,ncols,nSamples,datatype);
            dLdZ = randn(nDecs,nrows,ncols,nSamples,datatype);

            % Expected values (reference computed sample by sample)
            expctddLdX = zeros(nDecs,nrows,ncols,nSamples,datatype);
            expctddLdW = zeros(nAnglesH,nBlks,nSamples,datatype);
            expctddLdU = zeros(nAnglesH,nBlks,nSamples,datatype);
            for iSample = 1:nSamples
                angW = anglesW(:,:,iSample);
                angU = anglesU(:,:,iSample);

                % dLdX = dZdX x dLdZ
                W0T = permute(genW.step(angW,mus_,0),[2 1 3]);
                U0T = permute(genU.step(angU,mus_,0),[2 1 3]);
                dLdZi = reshape(dLdZ(:,:,:,iSample),nDecs,nBlks);
                Ys = dLdZi(1:ps,:);
                Ya = dLdZi(ps+1:end,:);
                for iblk = 1:nBlks
                    Ys(:,iblk) = W0T(1:ps,:,iblk)*Ys(:,iblk);
                    Ya(:,iblk) = U0T(1:pa,:,iblk)*Ya(:,iblk);
                end
                expctddLdX(:,:,:,iSample) = reshape([Ys;Ya],nDecs,nrows,ncols);

                % dLdWi = <dLdZ,(dVdWi)X>
                Xi = reshape(X(:,:,:,iSample),nDecs,nBlks);
                c_upp = Xi(1:ps,:);
                c_low = Xi(ps+1:end,:);
                dldz_upp = dLdZi(1:ps,:);
                dldz_low = dLdZi(ps+1:end,:);
                for iAngle = 1:nAnglesH
                    dW0 = genW.step(angW,mus_,iAngle);
                    dU0 = genU.step(angU,mus_,iAngle);
                    for iblk = 1:nBlks
                        d_upp = dW0(:,1:ps,iblk)*c_upp(:,iblk);
                        d_low = dU0(:,1:pa,iblk)*c_low(:,iblk);
                        expctddLdW(iAngle,iblk,iSample) = sum(dldz_upp(:,iblk).*d_upp,'all');
                        expctddLdU(iAngle,iblk,iSample) = sum(dldz_low(:,iblk).*d_low,'all');
                    end
                end
            end
            expctddLdTheta = cat(1,expctddLdW,expctddLdU);

            % Instantiation of target class
            import tansacnet.salsun.*
            layer = salsunInitialRotation2dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols],...
                'Name','V0~');

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
