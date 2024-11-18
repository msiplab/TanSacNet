classdef OrthonormalMatrixGenerationSystemTestCase < matlab.unittest.TestCase
    %ORTHONORMALMATRIXGENERATIONSYSTEMTESTCASE Test case for OrthonormalMatrixGenerationSystem
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2014-2024, Shogo MURAMATSU, Yasas GODAGE
    %
    % All rights reserved.
    %
    % Contact address: Shogo MURAMATSU,
    %                Faculty of Engineering, Niigata University,
    %                8050 2-no-cho Ikarashi, Nishi-ku,
    %                Niigata, 950-2181, JAPAN
    %
    % http://msiplab.eng.niigata-u.ac.jp/
    %

    properties (TestParameter)
        nblks = { 1, 2, 4 };
        datatype =  {'single', 'double'};
        usegpu = struct( 'true', true, 'false', false );
    end

    properties
        omgs
    end

    methods (TestMethodTeardown)
        function deleteObject(testCase)
            delete(testCase.omgs);
        end
    end


    methods (Test)

        % Test for default construction
        function testConstructor(testCase)

            % Expected values
            coefExpctd = [
                1 0 ;
                0 1 ];

            % Instantiation of target class
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem( ...
                'Device','cpu');

            % Actual values
            coefActual = step(testCase.omgs,0,1);

            % Evaluation
            testCase.verifyEqual(coefActual,coefExpctd);
        end

        function testMismatchDeviceDTypeOnGpu(testCase,datatype)

            expctdDevice = "cuda";
            expctdDType = datatype;
            if datatype == "double"
                anotherDType = "single";
            else
                anotherDType = "double";
            end

            % Instantiation of target class
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem( ...
                'Device',expctdDevice,...
                'DType',expctdDType);

            % Exception test for device
            try
                step(testCase.omgs,0,1);
            catch ME
                testCase.verifyEqual(ME.identifier,'LSUN:DeviceMismatch');
                testCase.verifyEqual(ME.message,'ANGLES should be gpuArray')
            end

            % Use gpu
            if gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end

            try
                step(testCase.omgs,gpuArray(0),1);
            catch ME
                testCase.verifyEqual(ME.identifier,'LSUN:DeviceMismatch');
                testCase.verifyEqual(ME.message,'MUS should be gpuArray')
            end

            % Exception test for dtype
            try
                step(testCase.omgs,gpuArray(cast(0,anotherDType)),gpuArray(cast(1,expctdDType)))
            catch ME
                testCase.verifyEqual(ME.identifier,'LSUN:DTypeMismatch');
                testCase.verifyEqual(ME.message,char("ANGLES should be "+expctdDType))
            end

            try
                step(testCase.omgs,gpuArray(cast(0,expctdDType)),gpuArray(cast(1,anotherDType)))
            catch ME
                testCase.verifyEqual(ME.identifier,'LSUN:DTypeMismatch');
                testCase.verifyEqual(ME.message,char("MUS should be "+expctdDType))
            end


        end

        function testMismatchDeviceDTypeCpu(testCase,datatype)

            expctdDevice = "cpu";
            expctdDType = datatype;
            if datatype == "double"
                anotherDType = "single";
            else
                anotherDType = "double";
            end

            % Instantiation of target class
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem( ...
                'Device',expctdDevice,...
                'DType',expctdDType);


            % Exception test for dtype
            try
                step(testCase.omgs,cast(0,anotherDType),cast(1,expctdDType))
            catch ME
                testCase.verifyEqual(ME.identifier,'LSUN:DTypeMismatch');
                testCase.verifyEqual(ME.message,char("ANGLES should be "+expctdDType))
            end

            try
                step(testCase.omgs,cast(0,expctdDType),cast(1,anotherDType))
            catch ME
                testCase.verifyEqual(ME.identifier,'LSUN:DTypeMismatch');
                testCase.verifyEqual(ME.message,char("MUS should be "+expctdDType))
            end

            % Use gpu
            if gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end

            % Exception test for device
            try
                step(testCase.omgs,gpuArray(0),gpuArray(1));
            catch ME
                testCase.verifyEqual(ME.identifier,'LSUN:DeviceMismatch');
                testCase.verifyEqual(ME.message,'ANGLES should be on CPU')
            end

            try
                step(testCase.omgs,0,gpuArray(1));
            catch ME
                testCase.verifyEqual(ME.identifier,'LSUN:DeviceMismatch');
                testCase.verifyEqual(ME.message,'MUS should be on CPU')
            end

        end

        % Test for default construction
        function testConstructorWithDeviceAndDType(testCase, usegpu, datatype)
            if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            % Expected values
            coefExpctd = [
                1 0 ;
                0 1 ];

            device_ = [ "cpu", "cuda" ];
            expctdDevice = device_(usegpu+1);
            expctdDType = datatype;

            % Instantiation of target class
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem(...
                'Device', expctdDevice, ...
                'DType', expctdDType ...
                );

            % Actual values
            coefActual = step(testCase.omgs,0,1);
            actualDevice = testCase.omgs.Device;
            actualDType = testCase.omgs.DType;

            % Evaluation
            if usegpu
                testCase.verifyClass(coefActual,'gpuArray')
                coefActual = gather(coefActual);
            end
            testCase.verifyEqual(actualDevice,expctdDevice);
            testCase.verifyEqual(actualDType,expctdDType);
            testCase.verifyEqual(coefActual,coefExpctd);            
        end


        % Test for default construction
        function testConstructorWithAngles(testCase)

            % Expected values
            coefExpctd = [
                cos(pi/4) -sin(pi/4) ;
                sin(pi/4)  cos(pi/4) ];

            % Instantiation of target class
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem();

            % Actual values
            coefActual = step(testCase.omgs,pi/4,1);

            % Evaluation
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-10);
        end

        % Test for default construction
        function testConstructorWithAnglesDeviceAndDType(testCase, usegpu, datatype)
            if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            device_ = [ "cpu", "cuda" ];
            expctdDevice = device_(usegpu+1);
            expctdDType = datatype;

            % Expected values
            coefExpctd = cast([
                cos(pi/4) -sin(pi/4) ;
                sin(pi/4)  cos(pi/4) ],expctdDType);

            % Instantiation of target class
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem(...
                'Device', expctdDevice, ...
                'DType', expctdDType ...
                );

            % Actual values
            coefActual = step(testCase.omgs,pi/4,1);

            % Evaluation
            if usegpu
                testCase.verifyClass(coefActual,'gpuArray');
                coefActual = gather(coefActual);
            end
            testCase.verifyInstanceOf(coefActual,expctdDType);
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-10);
        end

        % Test for default construction
        function testConstructorWithAnglesMultiple(testCase)

            % Expected values
            coefExpctd(:,:,1) = [
                cos(pi/4) -sin(pi/4) ;
                sin(pi/4)  cos(pi/4) ];
            coefExpctd(:,:,2) = [
                cos(pi/6) -sin(pi/6) ;
                sin(pi/6)  cos(pi/6) ];

            % Instantiation of target class
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem();

            % Actual values
            angles = [pi/4 pi/6];
            coefActual = step(testCase.omgs,angles,1);

            % Evaluation
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-10);
        end

        % Test for default construction
        function testConstructorWithAnglesAndMus(testCase)

            % Expected values
            coefExpctd = [
                cos(pi/4) -sin(pi/4) ;
                -sin(pi/4) -cos(pi/4) ];

            % Instantiation of target class
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem();

            % Actual values
            coefActual = step(testCase.omgs,pi/4,[ 1 -1 ]);

            % Evaluation
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-10);

        end

        % Test for default construction
        function testConstructorWithMus(testCase)

            % Expected values
            coefExpctd = [
                1 0 ;
                0 -1 ];

            % Instantiation of target class
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem();

            % Actual values
            coefActual = step(testCase.omgs,[],[ 1 -1 ]);

            % Evaluation
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-10);

        end

        function testConstructorWithMusMultiple(testCase)

            % Expected values
            coefExpctd(:,:,1) = [
                1 0 ;
                0 -1 ];
            coefExpctd(:,:,2) = [
                -1 0 ;
                0 1 ];

            % Instantiation of target class
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem();

            % Actual values
            mus = [  1 -1 ;
                -1  1 ];
            coefActual = step(testCase.omgs,[],mus);

            % Evaluation
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-10);

        end

        function testConstructorWithAnglesAndMusMultiple(testCase)

            % Expected values
            coefExpctd(:,:,1) = [
                cos(pi/4) -sin(pi/4) ;
                -sin(pi/4) -cos(pi/4) ];
            coefExpctd(:,:,2) = [
                -cos(pi/6) sin(pi/6) ;
                sin(pi/6) cos(pi/6) ];

            % Instantiation of target class
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem();

            % Actual values
            angles = [pi/4 pi/6];
            mus = [  1 -1 ;
                -1  1 ];
            coefActual = step(testCase.omgs,angles,mus);

            % Evaluation
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-10);

        end

        % Test for set angle
        function testSetAngles(testCase)

            % Expected values
            coefExpctd = [
                1 0 ;
                0 1 ];

            % Instantiation of target class
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem();

            % Actual values
            coefActual = step(testCase.omgs,0,1);

            % Evaluation
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-10);

            % Expected values
            coefExpctd = [
                cos(pi/4) -sin(pi/4) ;
                sin(pi/4)  cos(pi/4) ];

            % Actual values
            coefActual = step(testCase.omgs,pi/4,1);

            % Evaluation
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-10);
        end

        % Test for set angle
        function testSetAnglesMultiple(testCase)

            % Expected values
            coefExpctd(:,:,1) = [
                1 0 ;
                0 1 ];
            coefExpctd(:,:,2) = [
                1 0;
                0 1 ];

            % Instantiation of target class
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem();

            % Actual values
            angles = [0 0];
            coefActual = step(testCase.omgs,angles,1);

            % Evaluation
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-10);

            % Expected values
            coefExpctd(:,:,1) = [
                cos(pi/4) -sin(pi/4) ;
                sin(pi/4)  cos(pi/4) ];
            coefExpctd(:,:,2) = [
                cos(pi/6) -sin(pi/6) ;
                sin(pi/6)  cos(pi/6) ];


            % Actual values
            angles = [pi/4 pi/6];
            coefActual = step(testCase.omgs,angles,1);

            % Evaluation
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-10);
        end

        % Test for set angle
        function test2x2Multiple(testCase,nblks)

            % Expected values
            normExpctd = ones(1,2,nblks);

            % Instantiation of target class
            angs = 2*pi*rand(1,nblks);
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem();

            % Actual values
            matrices = step(testCase.omgs,angs,1);
            normActual = vecnorm(matrices,2,1);

            % Evaluation
            message = ...
                sprintf('normActual=%g differs from 1',normActual);
            testCase.verifyEqual(normActual,normExpctd,'RelTol',1e-15,message);

        end

        % Test for set angle
        function test4x4(testCase)

            % Expected values
            normExpctd = 1;

            % Instantiation of target class
            ang = 2*pi*rand(6,1);
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem();

            % Actual values
            normActual = norm(step(testCase.omgs,ang,1)*[1 0 0 0].');

            % Evaluation
            message = ...
                sprintf('normActual=%g differs from 1',normActual);
            testCase.verifyEqual(normActual,normExpctd,'RelTol',1e-15,message);

        end

        % Test for set angle
        function test4x4Multiple(testCase,nblks)

            % Expected values
            normExpctd = ones(1,4,nblks);

            % Instantiation of target class
            angs = 2*pi*rand(6,nblks);
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem();

            % Actual values
            matrices = step(testCase.omgs,angs,1);
            normActual = vecnorm(matrices,2,1);

            % Evaluation
            message = ...
                sprintf('normActual=%g differs from 1',normActual);
            testCase.verifyEqual(normActual,normExpctd,'RelTol',1e-15,message);

        end


        % Test for set angle
        function test8x8(testCase)

            % Expected values
            normExpctd = 1;

            % Instantiation of target class
            ang = 2*pi*rand(28,1);
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem();

            % Actual values
            normActual = norm(step(testCase.omgs,ang,1)*[1 0 0 0 0 0 0 0].');

            % Evaluation
            message = ...
                sprintf('normActual=%g differs from 1',normActual);
            testCase.verifyEqual(normActual,normExpctd,'RelTol',1e-15,message);
        end

        % Test for set angle
        function test8x8Multiple(testCase,nblks)

            % Expected values
            normExpctd = ones(1,8,nblks);

            % Instantiation of target class
            ang = 2*pi*rand(28,nblks);
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem();

            % Actual values
            matrices = step(testCase.omgs,ang,1);
            normActual = vecnorm(matrices,2,1);

            % Evaluation
            message = ...
                sprintf('normActual=%g differs from 1',normActual);
            testCase.verifyEqual(normActual,normExpctd,'RelTol',1e-15,message);
        end

        % Test for set angle
        function test4x4red(testCase)

            % Expected values
            ltExpctd = 1;

            % Instantiation of target class
            ang = 2*pi*rand(6,1);
            nSize = 4;
            ang(1:nSize-1,1) = zeros(nSize-1,1);
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem();

            % Actual values
            matrix = step(testCase.omgs,ang,1);
            ltActual = matrix(1,1);

            % Evaluation
            message = ...
                sprintf('ltActual=%g differs from 1',ltActual);
            testCase.verifyEqual(ltActual,ltExpctd,'RelTol',1e-15,message);
        end

        % Test for set angle
        function test4x4redMultiple(testCase,nblks)

            % Expected values
            ltExpctd = ones(1,1,nblks);

            % Instantiation of target class
            angs = 2*pi*rand(6,nblks);
            nSize = 4;
            angs(1:nSize-1,:) = zeros(nSize-1,nblks);
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem();

            % Actual values
            matrix = step(testCase.omgs,angs,1);
            ltActual = matrix(1,1,:);

            % Evaluation
            message = ...
                sprintf('ltActual=%g differs from 1',ltActual);
            testCase.verifyEqual(ltActual,ltExpctd,'RelTol',1e-15,message);
        end

        % Test for set angle
        function test8x8red(testCase)

            % Expected values
            ltExpctd = 1;

            % Instantiation of target class
            ang = 2*pi*rand(28,1);
            nSize = 8;
            ang(1:nSize-1,1) = zeros(nSize-1,1);
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem();

            % Actual values
            matrix = step(testCase.omgs,ang,1);
            ltActual = matrix(1,1);

            % Evaluation
            message = ...
                sprintf('ltActual=%g differs from 1',ltActual);
            testCase.verifyEqual(ltActual,ltExpctd,'RelTol',1e-15,message);
        end

        % Test for set angle
        function test8x8redMultiple(testCase,nblks)

            % Expected values
            ltExpctd = ones(1,1,nblks);

            % Instantiation of target class
            angs = 2*pi*rand(28,nblks);
            nSize = 8;
            angs(1:nSize-1,:) = zeros(nSize-1,nblks);
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem();

            % Actual values
            matrix = step(testCase.omgs,angs,1);
            ltActual = matrix(1,1,:);

            % Evaluation
            message = ...
                sprintf('ltActual=%g differs from 1',ltActual);
            testCase.verifyEqual(ltActual,ltExpctd,'RelTol',1e-15,message);
        end

        % Test for set angle
        function testPartialDifference(testCase)

            % Expected values
            coefExpctd = [
                0 -1 ;
                1  0 ];

            % Instantiation of target class
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on');

            % Actual values
            coefActual = step(testCase.omgs,0,1,1);

            % Evaluation
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-10);
        end

        % Test for set angle
        function testPartialDifferenceMultiple(testCase)

            % Expected values
            coefExpctd(:,:,1) = [
                0 -1 ;
                1  0 ];
            coefExpctd(:,:,2) = [
                0 -1 ;
                1  0 ];

            % Instantiation of target class
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on');

            % Actual values
            coefActual = step(testCase.omgs,[0 0],1,1);

            % Evaluation
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-10);
        end

        % Test for set angle
        function testPartialDifferenceAngs(testCase)

            % Expected values
            coefExpctd = [
                cos(pi/4+pi/2) -sin(pi/4+pi/2) ;
                sin(pi/4+pi/2)  cos(pi/4+pi/2) ];

            % Instantiation of target class
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on');

            % Actual values
            coefActual = step(testCase.omgs,pi/4,1,1);

            % Evaluation
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-10);
        end

        % Test for set angle
        function testPartialDifferenceAngsMultiple(testCase)

            % Expected values
            coefExpctd(:,:,1) = [
                cos(pi/4+pi/2) -sin(pi/4+pi/2) ;
                sin(pi/4+pi/2)  cos(pi/4+pi/2) ];
            coefExpctd(:,:,2) = [
                cos(pi/6+pi/2) -sin(pi/6+pi/2) ;
                sin(pi/6+pi/2)  cos(pi/6+pi/2) ];

            % Instantiation of target class
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on');

            % Actual values
            angs = [pi/4 pi/6];
            coefActual = step(testCase.omgs,angs,1,1);

            % Evaluation
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-10);
        end

        % Test for default construction
        function testPartialDifferenceWithAnglesAndMus(testCase)

            % Expected values
            coefExpctd = [
                cos(pi/4+pi/2) -sin(pi/4+pi/2) ;
                -sin(pi/4+pi/2) -cos(pi/4+pi/2) ];

            % Instantiation of target class
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on');

            % Actual values
            coefActual = step(testCase.omgs,pi/4,[ 1 -1 ],1);

            % Evaluation
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-10);

        end

        % Test for default construction
        function testPartialDifferenceWithAnglesAndMusMultiple(testCase)

            % Expected values
            coefExpctd(:,:,1) = [
                cos(pi/4+pi/2) -sin(pi/4+pi/2) ;
                -sin(pi/4+pi/2) -cos(pi/4+pi/2) ];
            coefExpctd(:,:,2) = [
                -cos(pi/6+pi/2) sin(pi/6+pi/2) ;
                sin(pi/6+pi/2) cos(pi/6+pi/2) ];

            % Instantiation of target class
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on');

            % Actual values
            angs = [pi/4 pi/6];
            mus = [1 -1 ; -1 1];
            coefActual = step(testCase.omgs,angs,mus,1);

            % Evaluation
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-10);

        end

        % Test for set angle
        function testPartialDifferenceSetAngles(testCase)

            % Expected values
            coefExpctd = [
                0 -1 ;
                1  0 ];

            % Instantiation of target class
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on');

            % Actual values
            coefActual = step(testCase.omgs,0,1,1);

            % Evaluation
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-10);

            % Expected values
            coefExpctd = [
                cos(pi/4+pi/2) -sin(pi/4+pi/2) ;
                sin(pi/4+pi/2)  cos(pi/4+pi/2) ];

            % Actual values
            coefActual = step(testCase.omgs,pi/4,1,1);

            % Evaluation
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-10);
        end

        % Test for set angle
        function testPartialDifferenceSetAnglesMultiple(testCase)

            % Expected values
            coefExpctd(:,:,1) = [
                0 -1 ;
                1  0 ];
            coefExpctd(:,:,2) = [
                0 -1 ;
                1  0 ];

            % Instantiation of target class
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on');

            % Actual values
            angs = [0 0];
            coefActual = step(testCase.omgs,angs,1,1);

            % Evaluation
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-10);

            % Expected values
            coefExpctd(:,:,1) = [
                cos(pi/4+pi/2) -sin(pi/4+pi/2) ;
                sin(pi/4+pi/2)  cos(pi/4+pi/2) ];
            coefExpctd(:,:,2) = [
                cos(pi/6+pi/2) -sin(pi/6+pi/2) ;
                sin(pi/6+pi/2)  cos(pi/6+pi/2) ];

            % Actual values
            angs = [pi/4 pi/6];
            coefActual = step(testCase.omgs,angs,1,1);

            % Evaluation
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-10);
        end


        % Test for set angle
        function test4x4RandAngs(testCase)

            % Expected values
            mus = [ -1 1 -1 1 ];
            angs = 2*pi*rand(6,1);
            coefExpctd = ...
                diag(mus) * ...
                [ 1  0   0             0            ;
                0  1   0             0            ;
                0  0   cos(angs(6)) -sin(angs(6)) ;
                0  0   sin(angs(6))  cos(angs(6)) ] *...
                [ 1  0            0  0            ;
                0  cos(angs(5)) 0 -sin(angs(5)) ;
                0  0            1  0            ;
                0  sin(angs(5)) 0 cos(angs(5))  ] *...
                [ 1  0             0            0 ;
                0  cos(angs(4)) -sin(angs(4)) 0 ;
                0  sin(angs(4))  cos(angs(4)) 0 ;
                0  0             0            1 ] *...
                [ cos(angs(3)) 0 0 -sin(angs(3))  ;
                0            1 0  0             ;
                0            0 1  0             ;
                sin(angs(3)) 0 0  cos(angs(3)) ] *...
                [ cos(angs(2)) 0 -sin(angs(2)) 0  ;
                0            1  0            0  ;
                sin(angs(2)) 0  cos(angs(2)) 0  ;
                0            0  0            1 ] *...
                [ cos(angs(1)) -sin(angs(1)) 0 0  ;
                sin(angs(1)) cos(angs(1))  0 0  ;
                0            0             1 0  ;
                0            0             0 1 ];

            % Instantiation of target class
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem();

            % Actual values
            coefActual = step(testCase.omgs,angs,mus);

            % Evaluation
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-10);

        end


        % Test for set angle
        function testPartialDifference4x4RandAngPdAng3(testCase)

            % Expected values
            mus = [ -1 1 -1 1 ];
            angs = 2*pi*rand(6,1);
            pdAng = 3;
            coefExpctd = ...
                diag(mus) * ...
                [ 1  0   0             0            ;
                0  1   0             0            ;
                0  0   cos(angs(6)) -sin(angs(6)) ;
                0  0   sin(angs(6))  cos(angs(6)) ] *...
                [ 1  0            0  0            ;
                0  cos(angs(5)) 0 -sin(angs(5)) ;
                0  0            1  0            ;
                0  sin(angs(5)) 0 cos(angs(5))  ] *...
                [ 1  0             0            0 ;
                0  cos(angs(4)) -sin(angs(4)) 0 ;
                0  sin(angs(4))  cos(angs(4)) 0 ;
                0  0             0            1 ] *...
                [ cos(angs(3)+pi/2) 0 0 -sin(angs(3)+pi/2)  ; % Partial Diff.
                0            0 0  0             ;
                0            0 0  0             ;
                sin(angs(3)+pi/2) 0 0  cos(angs(3)+pi/2) ] *...
                [ cos(angs(2)) 0 -sin(angs(2)) 0  ;
                0            1  0            0  ;
                sin(angs(2)) 0  cos(angs(2)) 0  ;
                0            0  0            1 ] *...
                [ cos(angs(1)) -sin(angs(1)) 0 0  ;
                sin(angs(1)) cos(angs(1))  0 0  ;
                0            0             1 0  ;
                0            0             0 1 ];

            % Instantiation of target class
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on');

            % Actual values
            coefActual = step(testCase.omgs,angs,mus,pdAng);

            % Evaluation
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-10);

        end

        % Test for set angle
        function testPartialDifference4x4RandAngPdAng3Multiple(testCase,nblks)

            % Expected values
            mus = [ -1 1 -1 1 ];
            angs = 2*pi*rand(6,nblks);
            pdAng = 3;
            coefExpctd = zeros(4,4,nblks);
            for iblk = 1:nblks
                coefExpctd(:,:,iblk) = ...
                    diag(mus) * ...
                    [ 1  0   0             0            ;
                    0  1   0             0            ;
                    0  0   cos(angs(6,iblk)) -sin(angs(6,iblk)) ;
                    0  0   sin(angs(6,iblk))  cos(angs(6,iblk)) ] *...
                    [ 1  0            0  0            ;
                    0  cos(angs(5,iblk)) 0 -sin(angs(5,iblk)) ;
                    0  0            1  0            ;
                    0  sin(angs(5,iblk)) 0 cos(angs(5,iblk))  ] *...
                    [ 1  0             0            0 ;
                    0  cos(angs(4,iblk)) -sin(angs(4,iblk)) 0 ;
                    0  sin(angs(4,iblk))  cos(angs(4,iblk)) 0 ;
                    0  0             0            1 ] *...
                    [ cos(angs(3,iblk)+pi/2) 0 0 -sin(angs(3,iblk)+pi/2)  ; % Partial Diff.
                    0            0 0  0             ;
                    0            0 0  0             ;
                    sin(angs(3,iblk)+pi/2) 0 0  cos(angs(3,iblk)+pi/2) ] *...
                    [ cos(angs(2,iblk)) 0 -sin(angs(2,iblk)) 0  ;
                    0            1  0            0  ;
                    sin(angs(2,iblk)) 0  cos(angs(2,iblk)) 0  ;
                    0            0  0            1 ] *...
                    [ cos(angs(1,iblk)) -sin(angs(1,iblk)) 0 0  ;
                    sin(angs(1,iblk)) cos(angs(1,iblk))  0 0  ;
                    0            0             1 0  ;
                    0            0             0 1 ];
            end

            % Instantiation of target class
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on');

            % Actual values
            coefActual = step(testCase.omgs,angs,mus,pdAng);

            % Evaluation
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-10);

        end

        % Test for set angle
        function testPartialDifference4x4RandAngPdAng6(testCase)

            % Expected values
            mus = [ 1 1 -1 -1 ];
            angs = 2*pi*rand(6,1);
            pdAng = 6;
            coefExpctd = ...
                diag(mus) * ...
                [ 0  0   0             0            ;
                0  0   0             0            ;
                0  0   cos(angs(6)+pi/2) -sin(angs(6)+pi/2) ; % Partial Diff.
                0  0   sin(angs(6)+pi/2)  cos(angs(6)+pi/2) ] *...
                [ 1  0            0  0            ;
                0  cos(angs(5)) 0 -sin(angs(5)) ;
                0  0            1  0            ;
                0  sin(angs(5)) 0 cos(angs(5))  ] *...
                [ 1  0             0            0 ;
                0  cos(angs(4)) -sin(angs(4)) 0 ;
                0  sin(angs(4))  cos(angs(4)) 0 ;
                0  0             0            1 ] *...
                [ cos(angs(3)) 0 0 -sin(angs(3))  ;
                0            1 0  0             ;
                0            0 1  0             ;
                sin(angs(3)) 0 0  cos(angs(3)) ] *...
                [ cos(angs(2)) 0 -sin(angs(2)) 0  ;
                0            1  0            0  ;
                sin(angs(2)) 0  cos(angs(2)) 0  ;
                0            0  0            1 ] *...
                [ cos(angs(1)) -sin(angs(1)) 0 0  ;
                sin(angs(1)) cos(angs(1))  0 0  ;
                0            0             1 0  ;
                0            0             0 1 ];

            % Instantiation of target class
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on');

            % Actual values
            coefActual = step(testCase.omgs,angs,mus,pdAng);

            % Evaluation
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-10);

        end

        % Test for set angle
        function testPartialDifference4x4RandAngPdAng6Multiple(testCase,nblks)

            % Expected values
            mus = [ 1 1 -1 -1 ];
            angs = 2*pi*rand(6,nblks);
            pdAng = 6;
            coefExpctd = zeros(4,4,nblks);
            for iblk = 1:nblks
                coefExpctd(:,:,iblk) = ...
                    diag(mus) * ...
                    [ 0  0   0             0            ;
                    0  0   0             0            ;
                    0  0   cos(angs(6,iblk)+pi/2) -sin(angs(6,iblk)+pi/2) ; % Partial Diff.
                    0  0   sin(angs(6,iblk)+pi/2)  cos(angs(6,iblk)+pi/2) ] *...
                    [ 1  0            0  0            ;
                    0  cos(angs(5,iblk)) 0 -sin(angs(5,iblk)) ;
                    0  0            1  0            ;
                    0  sin(angs(5,iblk)) 0 cos(angs(5,iblk))  ] *...
                    [ 1  0             0            0 ;
                    0  cos(angs(4,iblk)) -sin(angs(4,iblk)) 0 ;
                    0  sin(angs(4,iblk))  cos(angs(4,iblk)) 0 ;
                    0  0             0            1 ] *...
                    [ cos(angs(3,iblk)) 0 0 -sin(angs(3,iblk))  ;
                    0            1 0  0             ;
                    0            0 1  0             ;
                    sin(angs(3,iblk)) 0 0  cos(angs(3,iblk)) ] *...
                    [ cos(angs(2,iblk)) 0 -sin(angs(2,iblk)) 0  ;
                    0            1  0            0  ;
                    sin(angs(2,iblk)) 0  cos(angs(2,iblk)) 0  ;
                    0            0  0            1 ] *...
                    [ cos(angs(1,iblk)) -sin(angs(1,iblk)) 0 0  ;
                    sin(angs(1,iblk)) cos(angs(1,iblk))  0 0  ;
                    0            0             1 0  ;
                    0            0             0 1 ];
            end

            % Instantiation of target class
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on');

            % Actual values
            coefActual = step(testCase.omgs,angs,mus,pdAng);

            % Evaluation
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-10);

        end

        % Test for set angle
        function testPartialDifference4x4RandAngPdAng2(testCase)

            % Expected values
            mus = [ -1 -1 -1 -1 ];
            angs = 2*pi*rand(6,1);
            pdAng = 2;
            delta = 1e-10;
            coefExpctd = 1/delta * ...
                diag(mus) * ...
                [ 1  0   0             0            ;
                0  1   0             0            ;
                0  0   cos(angs(6)) -sin(angs(6)) ;
                0  0   sin(angs(6))  cos(angs(6)) ] *...
                [ 1  0            0  0            ;
                0  cos(angs(5)) 0 -sin(angs(5)) ;
                0  0            1  0            ;
                0  sin(angs(5)) 0 cos(angs(5))  ] *...
                [ 1  0             0            0 ;
                0  cos(angs(4)) -sin(angs(4)) 0 ;
                0  sin(angs(4))  cos(angs(4)) 0 ;
                0  0             0            1 ] *...
                [ cos(angs(3)) 0 0 -sin(angs(3))  ;
                0            1 0  0             ;
                0            0 1  0             ;
                sin(angs(3)) 0 0  cos(angs(3)) ] * ...
                ( ...
                [ cos(angs(2)+delta) 0 -sin(angs(2)+delta) 0  ;
                0            1  0            0  ;
                sin(angs(2)+delta) 0  cos(angs(2)+delta) 0  ;
                0            0  0            1 ] - ...
                [ cos(angs(2)) 0 -sin(angs(2)) 0  ;
                0            1  0            0  ;
                sin(angs(2)) 0  cos(angs(2)) 0  ;
                0            0  0            1 ] ...
                ) *...
                [ cos(angs(1)) -sin(angs(1)) 0 0  ;
                sin(angs(1)) cos(angs(1))  0 0  ;
                0            0             1 0  ;
                0            0             0 1 ];

            % Instantiation of target class
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on');

            % Actual values
            coefActual = step(testCase.omgs,angs,mus,pdAng);

            % Evaluation
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-5);

        end

        % Test for set angle
        function testPartialDifference4x4RandAngPdAng2Multiple(testCase,nblks)

            % Expected values
            mus = [ -1 -1 -1 -1 ];
            angs = 2*pi*rand(6,nblks);
            pdAng = 2;
            delta = 1e-10;
            coefExpctd = zeros(4,4,nblks);
            for iblk = 1:nblks
                coefExpctd(:,:,iblk) = 1/delta * ...
                    diag(mus) * ...
                    [ 1  0   0             0            ;
                    0  1   0             0            ;
                    0  0   cos(angs(6,iblk)) -sin(angs(6,iblk)) ;
                    0  0   sin(angs(6,iblk))  cos(angs(6,iblk)) ] *...
                    [ 1  0            0  0            ;
                    0  cos(angs(5,iblk)) 0 -sin(angs(5,iblk)) ;
                    0  0            1  0            ;
                    0  sin(angs(5,iblk)) 0 cos(angs(5,iblk))  ] *...
                    [ 1  0             0            0 ;
                    0  cos(angs(4,iblk)) -sin(angs(4,iblk)) 0 ;
                    0  sin(angs(4,iblk))  cos(angs(4,iblk)) 0 ;
                    0  0             0            1 ] *...
                    [ cos(angs(3,iblk)) 0 0 -sin(angs(3,iblk))  ;
                    0            1 0  0             ;
                    0            0 1  0             ;
                    sin(angs(3,iblk)) 0 0  cos(angs(3,iblk)) ] * ...
                    ( ...
                    [ cos(angs(2,iblk)+delta) 0 -sin(angs(2,iblk)+delta) 0  ;
                    0            1  0            0  ;
                    sin(angs(2,iblk)+delta) 0  cos(angs(2,iblk)+delta) 0  ;
                    0            0  0            1 ] - ...
                    [ cos(angs(2,iblk)) 0 -sin(angs(2,iblk)) 0  ;
                    0            1  0            0  ;
                    sin(angs(2,iblk)) 0  cos(angs(2,iblk)) 0  ;
                    0            0  0            1 ] ...
                    ) *...
                    [ cos(angs(1,iblk)) -sin(angs(1,iblk)) 0 0  ;
                    sin(angs(1,iblk)) cos(angs(1,iblk))  0 0  ;
                    0            0             1 0  ;
                    0            0             0 1 ];
            end

            % Instantiation of target class
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on');

            % Actual values
            coefActual = step(testCase.omgs,angs,mus,pdAng);

            % Evaluation
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-5);

        end

        % Test for set angle
        function testPartialDifference8x8RandAngPdAng2(testCase)

            % Expected values
            pdAng = 14;
            delta = 1e-10;
            angs0 = 2*pi*rand(28,1);
            angs1 = angs0;
            angs1(pdAng) = angs1(pdAng)+delta;

            % Instantiation of target class
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','off');
            coefExpctd = 1/delta * ...
                ( step(testCase.omgs,angs1,1) ...
                - step(testCase.omgs,angs0,1));

            % Instantiation of target class
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on');

            % Actual values
            coefActual = step(testCase.omgs,angs0,1,pdAng);

            % Evaluation
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-5);

        end

        % Test for set angle
        function testPartialDifference8x8RandAngPdAng2Multiple(testCase,nblks)

            % Expected values
            pdAng = 14;
            delta = 1e-10;
            angs0 = 2*pi*rand(28,nblks);
            angs1 = angs0;
            angs1(pdAng,:) = angs1(pdAng,:)+delta;

            % Instantiation of target class
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','off');
            coefExpctd = 1/delta * ...
                ( step(testCase.omgs,angs1,1) ...
                - step(testCase.omgs,angs0,1));

            % Instantiation of target class
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on');

            % Actual values
            coefActual = step(testCase.omgs,angs0,1,pdAng);

            % Evaluation
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-5);

        end

        %
        function testPartialDifferenceInSequentialMode(testCase)

            % Expected values
            coefExpctd = [
                -1 0 ;
                0 -1];

            % Instantiation of target class
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','sequential');

            % Actual values
            coefActual = testCase.omgs.step(0,-1,0);

            % Evaluation
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-10);
        end

        function testPartialDifferenceInSequentialModeMultiple(testCase)

            % Expected values
            coefExpctd(:,:,1) = [
                -1 0 ;
                0 -1];
            coefExpctd(:,:,2) = [
                -1 0 ;
                0 -1];

            % Instantiation of target class
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','sequential');

            % Actual values
            coefActual = testCase.omgs.step([0 0],-1,0);

            % Evaluation
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-10);
        end

        function testPartialDifferenceAngsInSequentialMode(testCase)

            % Configuratin
            pdAng = 1;

            % Expected values
            coefExpctd = [
                cos(pi/4+pi/2) -sin(pi/4+pi/2) ;
                sin(pi/4+pi/2)  cos(pi/4+pi/2) ];

            % Instantiation of target class
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','sequential');

            % Actual values
            step(testCase.omgs,pi/4,1,0);
            coefActual = step(testCase.omgs,pi/4,1,pdAng);

            % Evaluation
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-10);

        end

        function testPartialDifferenceAngsInSequentialModeMultiple(testCase)

            % Configuratin
            pdAng = 1;

            % Expected values
            coefExpctd(:,:,1) = [
                cos(pi/4+pi/2) -sin(pi/4+pi/2) ;
                sin(pi/4+pi/2)  cos(pi/4+pi/2) ];
            coefExpctd(:,:,2) = [
                cos(pi/6+pi/2) -sin(pi/6+pi/2) ;
                sin(pi/6+pi/2)  cos(pi/6+pi/2) ];

            % Instantiation of target class
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','sequential');

            % Actual values
            angs = [pi/4 pi/6];
            step(testCase.omgs,angs,1,0);
            coefActual = step(testCase.omgs,angs,1,pdAng);

            % Evaluation
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-10);

        end

        %
        function testPartialDifferenceWithAnglesAndMusInSequentialMode(testCase)

            % Configuration
            pdAng = 1;

            % Expected values
            coefExpctd = [
                cos(pi/4+pi/2) -sin(pi/4+pi/2) ;
                -sin(pi/4+pi/2) -cos(pi/4+pi/2) ];

            % Instantiation of target class
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','sequential');

            % Actual values
            step(testCase.omgs,pi/4,[ 1 -1 ],0);
            coefActual = step(testCase.omgs,pi/4,[ 1 -1 ],pdAng);

            % Evaluation
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-10);

        end

        %
        function testPartialDifferenceWithAnglesAndMusInSequentialModeMultiple(testCase)

            % Configuration
            pdAng = 1;

            % Expected values
            coefExpctd(:,:,1) = [
                cos(pi/4+pi/2) -sin(pi/4+pi/2) ;
                -sin(pi/4+pi/2) -cos(pi/4+pi/2) ];
            coefExpctd(:,:,2) = [
                cos(pi/6+pi/2) -sin(pi/6+pi/2) ;
                -sin(pi/6+pi/2) -cos(pi/6+pi/2) ];

            % Instantiation of target class
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','sequential');

            % Actual values
            angs = [pi/4 pi/6];
            mus = [1 -1];
            step(testCase.omgs,angs,mus,0);
            coefActual = step(testCase.omgs,angs,mus,pdAng);

            % Evaluation
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-10);

        end

        function testPartialDifference4x4RandAngPdAng3InSequentialMode(testCase)

            % Expected values
            mus = [ -1 1 -1 1 ];
            angs = 2*pi*rand(6,1);
            pdAng = 3;
            coefExpctd = ...
                diag(mus) * ...
                [ 1  0   0             0            ;
                0  1   0             0            ;
                0  0   cos(angs(6)) -sin(angs(6)) ;
                0  0   sin(angs(6))  cos(angs(6)) ] *...
                [ 1  0            0  0            ;
                0  cos(angs(5)) 0 -sin(angs(5)) ;
                0  0            1  0            ;
                0  sin(angs(5)) 0 cos(angs(5))  ] *...
                [ 1  0             0            0 ;
                0  cos(angs(4)) -sin(angs(4)) 0 ;
                0  sin(angs(4))  cos(angs(4)) 0 ;
                0  0             0            1 ] *...
                [ cos(angs(3)+pi/2) 0 0 -sin(angs(3)+pi/2)  ; % Partial Diff.
                0            0 0  0             ;
                0            0 0  0             ;
                sin(angs(3)+pi/2) 0 0  cos(angs(3)+pi/2) ] *...
                [ cos(angs(2)) 0 -sin(angs(2)) 0  ;
                0            1  0            0  ;
                sin(angs(2)) 0  cos(angs(2)) 0  ;
                0            0  0            1 ] *...
                [ cos(angs(1)) -sin(angs(1)) 0 0  ;
                sin(angs(1)) cos(angs(1))  0 0  ;
                0            0             1 0  ;
                0            0             0 1 ];

            % Instantiation of target class
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','sequential');

            % Actual values
            for iAng = 0:pdAng-1
                testCase.omgs.step(angs,mus,iAng);
            end
            coefActual = testCase.omgs.step(angs,mus,pdAng);

            % Evaluation
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-10);

        end


        function testPartialDifference4x4RandAngPdAng3InSequentialModeMultiple(testCase,nblks)

            % Expected values
            mus = [ -1 1 -1 1 ];
            angs = 2*pi*rand(6,nblks);
            pdAng = 3;
            coefExpctd = zeros(4,4,nblks);
            for iblk = 1:nblks
                coefExpctd(:,:,iblk) = ...
                    diag(mus) * ...
                    [ 1  0   0             0            ;
                    0  1   0             0            ;
                    0  0   cos(angs(6,iblk)) -sin(angs(6,iblk)) ;
                    0  0   sin(angs(6,iblk))  cos(angs(6,iblk)) ] *...
                    [ 1  0            0  0            ;
                    0  cos(angs(5,iblk)) 0 -sin(angs(5,iblk)) ;
                    0  0            1  0            ;
                    0  sin(angs(5,iblk)) 0 cos(angs(5,iblk))  ] *...
                    [ 1  0             0            0 ;
                    0  cos(angs(4,iblk)) -sin(angs(4,iblk)) 0 ;
                    0  sin(angs(4,iblk))  cos(angs(4,iblk)) 0 ;
                    0  0             0            1 ] *...
                    [ cos(angs(3,iblk)+pi/2) 0 0 -sin(angs(3,iblk)+pi/2)  ; % Partial Diff.
                    0            0 0  0             ;
                    0            0 0  0             ;
                    sin(angs(3,iblk)+pi/2) 0 0  cos(angs(3,iblk)+pi/2) ] *...
                    [ cos(angs(2,iblk)) 0 -sin(angs(2,iblk)) 0  ;
                    0            1  0            0  ;
                    sin(angs(2,iblk)) 0  cos(angs(2,iblk)) 0  ;
                    0            0  0            1 ] *...
                    [ cos(angs(1,iblk)) -sin(angs(1,iblk)) 0 0  ;
                    sin(angs(1,iblk)) cos(angs(1,iblk))  0 0  ;
                    0            0             1 0  ;
                    0            0             0 1 ];
            end

            % Instantiation of target class
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','sequential');

            % Actual values
            for iAng = 0:pdAng-1
                testCase.omgs.step(angs,mus,iAng);
            end
            coefActual = testCase.omgs.step(angs,mus,pdAng);

            % Evaluation
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-10);

        end

        %
        function testPartialDifference4x4RandAngPdAng6InSequentialMode(testCase)

            % Expected values
            mus = [ 1 1 -1 -1 ];
            angs = 2*pi*rand(6,1);
            pdAng = 6;
            coefExpctd = ...
                diag(mus) * ...
                [ 0  0   0             0            ;
                0  0   0             0            ;
                0  0   cos(angs(6)+pi/2) -sin(angs(6)+pi/2) ; % Partial Diff.
                0  0   sin(angs(6)+pi/2)  cos(angs(6)+pi/2) ] *...
                [ 1  0            0  0            ;
                0  cos(angs(5)) 0 -sin(angs(5)) ;
                0  0            1  0            ;
                0  sin(angs(5)) 0 cos(angs(5))  ] *...
                [ 1  0             0            0 ;
                0  cos(angs(4)) -sin(angs(4)) 0 ;
                0  sin(angs(4))  cos(angs(4)) 0 ;
                0  0             0            1 ] *...
                [ cos(angs(3)) 0 0 -sin(angs(3))  ;
                0            1 0  0             ;
                0            0 1  0             ;
                sin(angs(3)) 0 0  cos(angs(3)) ] *...
                [ cos(angs(2)) 0 -sin(angs(2)) 0  ;
                0            1  0            0  ;
                sin(angs(2)) 0  cos(angs(2)) 0  ;
                0            0  0            1 ] *...
                [ cos(angs(1)) -sin(angs(1)) 0 0  ;
                sin(angs(1)) cos(angs(1))  0 0  ;
                0            0             1 0  ;
                0            0             0 1 ];

            % Instantiation of target class
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','sequential');

            % Actual values
            for iAng = 0:pdAng-1
                step(testCase.omgs,angs,mus,iAng);
            end
            coefActual = testCase.omgs.step(angs,mus,pdAng);

            % Evaluation
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-10);

        end

        function testPartialDifference4x4RandAngPdAng6InSequentialModeMultiple(testCase,nblks)

            % Expected values
            mus = [ 1 1 -1 -1 ];
            angs = 2*pi*rand(6,nblks);
            pdAng = 6;
            coefExpctd = zeros(4,4,nblks);
            for iblk = 1:nblks
                coefExpctd(:,:,iblk) = ...
                    diag(mus) * ...
                    [ 0  0   0             0            ;
                    0  0   0             0            ;
                    0  0   cos(angs(6,iblk)+pi/2) -sin(angs(6,iblk)+pi/2) ; % Partial Diff.
                    0  0   sin(angs(6,iblk)+pi/2)  cos(angs(6,iblk)+pi/2) ] *...
                    [ 1  0            0  0            ;
                    0  cos(angs(5,iblk)) 0 -sin(angs(5,iblk)) ;
                    0  0            1  0            ;
                    0  sin(angs(5,iblk)) 0 cos(angs(5,iblk))  ] *...
                    [ 1  0             0            0 ;
                    0  cos(angs(4,iblk)) -sin(angs(4,iblk)) 0 ;
                    0  sin(angs(4,iblk))  cos(angs(4,iblk)) 0 ;
                    0  0             0            1 ] *...
                    [ cos(angs(3,iblk)) 0 0 -sin(angs(3,iblk))  ;
                    0            1 0  0             ;
                    0            0 1  0             ;
                    sin(angs(3,iblk)) 0 0  cos(angs(3,iblk)) ] *...
                    [ cos(angs(2,iblk)) 0 -sin(angs(2,iblk)) 0  ;
                    0            1  0            0  ;
                    sin(angs(2,iblk)) 0  cos(angs(2,iblk)) 0  ;
                    0            0  0            1 ] *...
                    [ cos(angs(1,iblk)) -sin(angs(1,iblk)) 0 0  ;
                    sin(angs(1,iblk)) cos(angs(1,iblk))  0 0  ;
                    0            0             1 0  ;
                    0            0             0 1 ];
            end

            % Instantiation of target class
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','sequential');

            % Actual values
            for iAng = 0:pdAng-1
                step(testCase.omgs,angs,mus,iAng);
            end
            coefActual = testCase.omgs.step(angs,mus,pdAng);

            % Evaluation
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-10);

        end

        %
        function testPartialDifference4x4RandAngPdAng2InSequentialMode(testCase)

            % Expected values
            mus = [ -1 -1 -1 -1 ];
            angs = 2*pi*rand(6,1);
            pdAng = 2;
            delta = 1e-10;
            coefExpctd = 1/delta * ...
                diag(mus) * ...
                [ 1  0   0             0            ;
                0  1   0             0            ;
                0  0   cos(angs(6)) -sin(angs(6)) ;
                0  0   sin(angs(6))  cos(angs(6)) ] *...
                [ 1  0            0  0            ;
                0  cos(angs(5)) 0 -sin(angs(5)) ;
                0  0            1  0            ;
                0  sin(angs(5)) 0 cos(angs(5))  ] *...
                [ 1  0             0            0 ;
                0  cos(angs(4)) -sin(angs(4)) 0 ;
                0  sin(angs(4))  cos(angs(4)) 0 ;
                0  0             0            1 ] *...
                [ cos(angs(3)) 0 0 -sin(angs(3))  ;
                0            1 0  0             ;
                0            0 1  0             ;
                sin(angs(3)) 0 0  cos(angs(3)) ] * ...
                ( ...
                [ cos(angs(2)+delta) 0 -sin(angs(2)+delta) 0  ;
                0            1  0            0  ;
                sin(angs(2)+delta) 0  cos(angs(2)+delta) 0  ;
                0            0  0            1 ] - ...
                [ cos(angs(2)) 0 -sin(angs(2)) 0  ;
                0            1  0            0  ;
                sin(angs(2)) 0  cos(angs(2)) 0  ;
                0            0  0            1 ] ...
                ) *...
                [ cos(angs(1)) -sin(angs(1)) 0 0  ;
                sin(angs(1)) cos(angs(1))  0 0  ;
                0            0             1 0  ;
                0            0             0 1 ];

            % Instantiation of target class
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','sequential');

            % Actual values
            for iAng = 0:pdAng-1
                step(testCase.omgs,angs,mus,iAng);
            end
            coefActual = step(testCase.omgs,angs,mus,pdAng);

            % Evaluation
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-5);

        end

        function testPartialDifference4x4RandAngPdAng2InSequentialModeMultiple(testCase,nblks)

            % Expected values
            mus = [ -1 -1 -1 -1 ];
            angs = 2*pi*rand(6,nblks);
            pdAng = 2;
            delta = 1e-10;
            coefExpctd = zeros(4,4,nblks);
            for iblk = 1:nblks
                coefExpctd(:,:,iblk) = 1/delta * ...
                    diag(mus) * ...
                    [ 1  0   0             0            ;
                    0  1   0             0            ;
                    0  0   cos(angs(6,iblk)) -sin(angs(6,iblk)) ;
                    0  0   sin(angs(6,iblk))  cos(angs(6,iblk)) ] *...
                    [ 1  0            0  0            ;
                    0  cos(angs(5,iblk)) 0 -sin(angs(5,iblk)) ;
                    0  0            1  0            ;
                    0  sin(angs(5,iblk)) 0 cos(angs(5,iblk))  ] *...
                    [ 1  0             0            0 ;
                    0  cos(angs(4,iblk)) -sin(angs(4,iblk)) 0 ;
                    0  sin(angs(4,iblk))  cos(angs(4,iblk)) 0 ;
                    0  0             0            1 ] *...
                    [ cos(angs(3,iblk)) 0 0 -sin(angs(3,iblk))  ;
                    0            1 0  0             ;
                    0            0 1  0             ;
                    sin(angs(3,iblk)) 0 0  cos(angs(3,iblk)) ] * ...
                    ( ...
                    [ cos(angs(2,iblk)+delta) 0 -sin(angs(2,iblk)+delta) 0  ;
                    0            1  0            0  ;
                    sin(angs(2,iblk)+delta) 0  cos(angs(2,iblk)+delta) 0  ;
                    0            0  0            1 ] - ...
                    [ cos(angs(2,iblk)) 0 -sin(angs(2,iblk)) 0  ;
                    0            1  0            0  ;
                    sin(angs(2,iblk)) 0  cos(angs(2,iblk)) 0  ;
                    0            0  0            1 ] ...
                    ) *...
                    [ cos(angs(1,iblk)) -sin(angs(1,iblk)) 0 0  ;
                    sin(angs(1,iblk)) cos(angs(1,iblk))  0 0  ;
                    0            0             1 0  ;
                    0            0             0 1 ];
            end

            % Instantiation of target class
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','sequential');

            % Actual values
            for iAng = 0:pdAng-1
                step(testCase.omgs,angs,mus,iAng);
            end
            coefActual = step(testCase.omgs,angs,mus,pdAng);

            % Evaluation
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-5);

        end


        % TODO test for multiple blocks


        %
        function testPartialDifference8x8RandAngPdAng2InSequentialMode(testCase)

            % Expected values
            pdAng = 14;
            delta = 1e-10;
            angs0 = 2*pi*rand(28,1);
            angs1 = angs0;
            angs1(pdAng) = angs1(pdAng)+delta;

            % Instantiation of target class
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','off');
            coefExpctd = 1/delta * ...
                ( step(testCase.omgs,angs1,1) ...
                - step(testCase.omgs,angs0,1));

            % Instantiation of target class
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','sequential');

            % Actual values
            for iAng = 0:pdAng-1
                step(testCase.omgs,angs0,1,iAng);
            end
            coefActual = step(testCase.omgs,angs0,1,pdAng);

            % Evaluation
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-5);

        end

        %
        function testPartialDifference8x8RandAngPdAng2InSequentialModeMultiple(testCase,nblks)

            % Expected values
            pdAng = 14;
            delta = 1e-10;
            angs0 = 2*pi*rand(28,nblks);
            angs1 = angs0;
            angs1(pdAng,:) = angs1(pdAng,:)+delta;

            % Instantiation of target class
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','off');
            coefExpctd = 1/delta * ...
                ( step(testCase.omgs,angs1,1) ...
                - step(testCase.omgs,angs0,1));

            % Instantiation of target class
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','sequential');

            % Actual values
            for iAng = 0:pdAng-1
                step(testCase.omgs,angs0,1,iAng);
            end
            coefActual = step(testCase.omgs,angs0,1,pdAng);

            % Evaluation
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-5);

        end

        %
        function testPartialDifference8x8RandMusAngPdAng2InSequentialMode(testCase,nblks)

            % Expected values
            mus = 2*round(rand(8,nblks))-1;
            pdAng = 14;
            delta = 1e-10;
            angs0 = 2*pi*rand(28,nblks);
            angs1 = angs0;
            angs1(pdAng,:) = angs1(pdAng,:)+delta;

            % Instantiation of target class
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','off');
            coefExpctd = 1/delta * ...
                ( step(testCase.omgs,angs1,mus) ...
                - step(testCase.omgs,angs0,mus));

            % Instantiation of target class
            import tansacnet.utility.*
            testCase.omgs = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','sequential');

            % Actual values
            for iAng = 0:pdAng-1
                step(testCase.omgs,angs0,mus,iAng);
            end
            coefActual = step(testCase.omgs,angs0,mus,pdAng);

            % Evaluation
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-5);

        end
    end
end
