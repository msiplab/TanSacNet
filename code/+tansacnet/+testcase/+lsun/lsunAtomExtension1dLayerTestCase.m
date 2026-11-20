classdef lsunAtomExtension1dLayerTestCase < matlab.unittest.TestCase
    %LSUNATOMEXTENSION1DLAYERTESTCASE
    %
    %   コンポーネント別に入力(nComponents=1のみサポート):
    %      nChsTotal x 1 x nBlks x nSamples
    %
    %   コンポーネント別に出力(nComponents=1のみサポート):
    %      nChsTotal x 1 x nBlks x nSamples
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
        nblks = struct('small', 4,'medium', 8);
        datatype = { 'single', 'double' };
        dir = { 'Right', 'Left' };
        target = { 'Sum', 'Difference' };
        usegpu = struct( 'true', true, 'false', false);
    end

    methods (TestClassTeardown)
        function finalCheck(~)
            import tansacnet.lsun.*
            layer = lsunAtomExtension1dLayer(...
                'Stride',4,...
                'Direction','Right',...
                'TargetChannels','Difference');
            fprintf("\n --- Check layer for 1-D sequences ---\n");
            checkLayer(layer,[4 1 8],...
                'ObservationDimension',4,...
                'CheckCodegenCompatibility',false)
        end
    end

    methods (Test)

        function testConstructor(testCase, stride)

            % Expected values
            expctdName = 'Qn';
            expctdDirection = 'Right';
            expctdTargetChannels = 'Difference';
            expctdDescription = "Right shift the difference-channel Coefs. " ...
                + "(ps,pa) = (" + stride/2 + "," + stride/2 + ")";

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunAtomExtension1dLayer(...
                'Stride',stride,...
                'Name',expctdName,...
                'Direction',expctdDirection,...
                'TargetChannels',expctdTargetChannels);

            % Evaluation
            testCase.verifyEqual(layer.Name,expctdName);
            testCase.verifyEqual(layer.Direction,expctdDirection);
            testCase.verifyEqual(layer.TargetChannels,expctdTargetChannels);
            testCase.verifyEqual(layer.Description,expctdDescription);
        end

        function testPredict(testCase, usegpu, stride, nblks, dir, target, datatype)

            testCase.assumeGpu_(usegpu,datatype);
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));

            % Parameters
            nSamples = 8;
            ps = stride/2;
            % nChsTotal x 1 x nBlks x nSamples
            X = randn(stride,1,nblks,nSamples,datatype);
            if usegpu
                X = gpuArray(X);
            end

            % Expected values
            expctdZ = testCase.atomext_(X,ps,testCase.shift_(dir),target);

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunAtomExtension1dLayer(...
                'Stride',stride,...
                'Name','Qx1',...
                'Direction',dir,...
                'TargetChannels',target);

            % Actual values
            actualZ = layer.predict(X);

            % Evaluation
            if usegpu
                testCase.verifyClass(actualZ,'gpuArray');
                actualZ = gather(actualZ);
                expctdZ = gather(expctdZ);
            end
            testCase.verifyInstanceOf(actualZ,datatype);
            testCase.verifyThat(actualZ,IsEqualTo(expctdZ,'Within',tolObj));
        end

        function testBackward(testCase, usegpu, stride, nblks, dir, target, datatype)

            testCase.assumeGpu_(usegpu,datatype);
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));

            % Parameters
            nSamples = 8;
            ps = stride/2;
            % nChsTotal x 1 x nBlks x nSamples
            dLdZ = randn(stride,1,nblks,nSamples,datatype);
            if usegpu
                dLdZ = gpuArray(dLdZ);
            end

            % Expected values (shift in the reverse direction)
            expctddLdX = testCase.atomext_(dLdZ,ps,-testCase.shift_(dir),target);

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunAtomExtension1dLayer(...
                'Stride',stride,...
                'Name','Qx1',...
                'Direction',dir,...
                'TargetChannels',target);

            % Actual values
            actualdLdX = layer.backward([],[],dLdZ,[]);

            % Evaluation
            if usegpu
                testCase.verifyClass(actualdLdX,'gpuArray');
                actualdLdX = gather(actualdLdX);
                expctddLdX = gather(expctddLdX);
            end
            testCase.verifyInstanceOf(actualdLdX,datatype);
            testCase.verifyThat(actualdLdX,IsEqualTo(expctddLdX,'Within',tolObj));
        end

        function testAdjoint(testCase, stride, nblks, dir, target)
            % backward is the adjoint of predict: <predict(X),Y> = <X,backward(Y)>

            X = randn(stride,1,nblks,2);
            Y = randn(stride,1,nblks,2);
            import tansacnet.lsun.*
            layer = lsunAtomExtension1dLayer(...
                'Stride',stride,...
                'Direction',dir,...
                'TargetChannels',target);

            % Evaluation
            testCase.verifyEqual(sum(layer.predict(X).*Y,'all'),...
                sum(X.*layer.backward([],[],Y,[]),'all'),'AbsTol',1e-10);
        end

        function testInvalidDirection(testCase)
            import tansacnet.lsun.*
            layer = lsunAtomExtension1dLayer(...
                'Stride',4,...
                'Direction','Down',...
                'TargetChannels','Sum');
            testCase.verifyError(@() layer.predict(randn(4,1,4,1)),...
                'LsunLayer:InvalidDirection');
        end

        function testInvalidTargetChannels(testCase)
            import tansacnet.lsun.*
            layer = lsunAtomExtension1dLayer(...
                'Stride',4,...
                'Direction','Right',...
                'TargetChannels','Invalid');
            testCase.verifyError(@() layer.predict(randn(4,1,4,1)),...
                'LsunLayer:InvalidTargetChannels');
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

        function shift = shift_(dir)
            if strcmp(dir,'Right')
                shift = [ 0 0 1 0 ];
            else
                shift = [ 0 0 -1 0 ];
            end
        end

        function Z = atomext_(X,ps,shift,target)
            % Block butterfly, block circular shift and block butterfly
            Xs = X(1:ps,:,:,:);
            Xa = X(ps+1:end,:,:,:);
            Ys = Xs + Xa;
            Ya = Xs - Xa;
            if strcmp(target,'Difference')
                Ya = circshift(Ya,shift);
            else
                Ys = circshift(Ys,shift);
            end
            Z = cat(1,Ys+Ya,Ys-Ya)/2;
        end

    end

end
