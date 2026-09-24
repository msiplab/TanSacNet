classdef ComplexButterflyCoefficientGenerationSystemTestCase < matlab.unittest.TestCase
    %COMPLEXBUTTERFLYCOEFFICIENTGENERATIONSYSTEMTESTCASE Test case for
    %ComplexButterflyCoefficientGenerationSystem
    %
    %   Target class (to be implemented, Step 2 Green):
    %       tansacnet.utility.ComplexButterflyCoefficientGenerationSystem
    %
    %   This generates the complex butterfly matrix B_hat_k (eq. 9 of
    %   the paper "局所構造化ユニタリネットワークの複素拡張に関する検討",
    %   Ohata & Muramatsu), confirmed with the paper's author as follows
    %   (theta real, p = 0,...,floor(m/2)-1):
    %
    %       chat(theta) = [ -1j*cos(theta)  -1j*sin(theta) ;
    %                         cos(theta)      -sin(theta)   ]
    %
    %       shat(theta) = [   sin(theta)      cos(theta)    ;
    %                        1j*sin(theta)  -1j*cos(theta)  ]
    %
    %       Chat_k = blkdiag( chat(theta_0), ..., chat(theta_{floor(m/2)-1}) )
    %                  [ + scalar 1  appended as last diagonal entry, if m is odd ]
    %       Shat_k = blkdiag( shat(theta_0), ..., shat(theta_{floor(m/2)-1}) )
    %                  [ + scalar 1j appended as last diagonal entry, if m is odd ]
    %
    %       Bhat_k = [ Chat_k,  conj(Chat_k) ;
    %                  Shat_k,  conj(Shat_k) ]          (size 2m x 2m)
    %
    %   Branch condition is the parity of m itself (NOT floor(m/2), which
    %   was a transcription slip in the paper's text, confirmed with the
    %   author). Number of angles per block is nAngles = floor(m/2).
    %
    %   Key property (verified by hand for m=1 and m=2, both at theta=0
    %   and theta=pi/4, consistent with the (1/2) prefactor of Q_k(z) in
    %   eq. 7 of the paper): Bhat_k is unitary up to a scale of 2, i.e.
    %
    %       Bhat_k * Bhat_k^H = 2 * eye(2m)
    %
    %   PartialDifference: since Chat_k/Shat_k are BLOCK-DIAGONAL (no
    %   cascade/product between different p-blocks, unlike
    %   OrthonormalMatrixGenerationSystem's Givens-rotation cascade),
    %   d(Bhat_k)/d(theta_p) is simply zero everywhere except the p-th
    %   2x2 sub-blocks of Chat_k/Shat_k (and their conjugates), where the
    %   same chat/shat formulas apply with theta_p -> theta_p + pi/2
    %   (since d/dtheta[cos]=-sin=cos(theta+pi/2) and
    %   d/dtheta[sin]=cos=sin(theta+pi/2), and the 1j factors are
    %   constants w.r.t. theta). This is an ordinary complex derivative
    %   of a real-parameterized matrix -- no Wirtinger-convention
    %   ambiguity arises here (that issue, found in the Step 0 spike,
    %   is specific to a layer's backward() w.r.t. a real loss, not to
    %   this generation system's own partial derivative).
    %
    % Requirements: MATLAB R2022a

    properties (TestParameter)
        m        = struct('m1',1,'m2',2,'m3',3,'m4',4,'m5',5);
        nblks    = struct('single',1,'multiple',3);
        datatype = {'single','double'};
        usegpu   = struct('true',true,'false',false);
    end

    properties
        cbgs
    end

    methods (TestMethodTeardown)
        function deleteObject(testCase)
            delete(testCase.cbgs);
        end
    end

    methods (Test)

        %% Basic construction / hand-derived closed forms

        function testConstructorM1(testCase)
            % m=1: floor(m/2)=0, no angle, odd-m extra scalars only.
            % Bhat = [1, 1 ; 1j, -1j]

            bhatExpctd = [ 1, 1 ; 1j, -1j ];

            import tansacnet.utility.*
            testCase.cbgs = ComplexButterflyCoefficientGenerationSystem(...
                'NumberOfHalfChannels',1);

            angles = zeros(0,1); % no angles for m=1
            if canUseGPU
                angles = gpuArray(angles);
            end
            bhatActual = step(testCase.cbgs,angles);

            if canUseGPU
                testCase.verifyClass(bhatActual,'gpuArray');
                bhatActual = gather(bhatActual);
            end
            testCase.verifyEqual(bhatActual,bhatExpctd,'AbsTol',1e-10);
        end

        function testConstructorM2ThetaZero(testCase)
            % m=2: floor(m/2)=1, one angle, m even -> no extra scalar.
            % theta=0: chat=[-1j,0;1,0], shat=[0,1;0,-1j]

            bhatExpctd = [ ...
                -1j,  0,  1j,  0 ; ...
                  1,  0,   1,  0 ; ...
                  0,  1,   0,  1 ; ...
                  0,-1j,   0, 1j ];

            import tansacnet.utility.*
            testCase.cbgs = ComplexButterflyCoefficientGenerationSystem(...
                'NumberOfHalfChannels',2);

            angles = 0;
            if canUseGPU
                angles = gpuArray(angles);
            end
            bhatActual = step(testCase.cbgs,angles);

            if canUseGPU
                testCase.verifyClass(bhatActual,'gpuArray');
                bhatActual = gather(bhatActual);
            end
            testCase.verifyEqual(bhatActual,bhatExpctd,'AbsTol',1e-10);
        end

        function testConstructorM2ThetaPiOver4(testCase)
            % m=2, theta=pi/4: cos=sin=1/sqrt(2)

            c = cos(pi/4); s = sin(pi/4); %#ok<NASGU> % s==c here
            chat = [ -1j*c, -1j*s ; c, -s ];
            shat = [ s, c ; 1j*s, -1j*c ];
            bhatExpctd = [ chat, conj(chat) ; shat, conj(shat) ];

            import tansacnet.utility.*
            testCase.cbgs = ComplexButterflyCoefficientGenerationSystem(...
                'NumberOfHalfChannels',2);

            angles = pi/4;
            if canUseGPU
                angles = gpuArray(angles);
            end
            bhatActual = step(testCase.cbgs,angles);

            if canUseGPU
                testCase.verifyClass(bhatActual,'gpuArray');
                bhatActual = gather(bhatActual);
            end
            testCase.verifyEqual(bhatActual,bhatExpctd,'AbsTol',1e-10);
        end

        function testConstructorM3WithExtraScalar(testCase)
            % m=3: floor(m/2)=1 (one 2x2 block) + one extra scalar
            % (m is odd), theta=pi/6.

            theta = pi/6;
            c = cos(theta); s = sin(theta);
            chat = [ -1j*c, -1j*s ; c, -s ];
            shat = [ s, c ; 1j*s, -1j*c ];
            Chat = blkdiag(chat,1);
            Shat = blkdiag(shat,1j);
            bhatExpctd = [ Chat, conj(Chat) ; Shat, conj(Shat) ];

            import tansacnet.utility.*
            testCase.cbgs = ComplexButterflyCoefficientGenerationSystem(...
                'NumberOfHalfChannels',3);

            angles = theta;
            if canUseGPU
                angles = gpuArray(angles);
            end
            bhatActual = step(testCase.cbgs,angles);

            if canUseGPU
                testCase.verifyClass(bhatActual,'gpuArray');
                bhatActual = gather(bhatActual);
            end
            testCase.verifySize(bhatActual,[6 6]);
            testCase.verifyEqual(bhatActual,bhatExpctd,'AbsTol',1e-10);
        end

        function testConstructorM4TwoAngles(testCase)
            % m=4: floor(m/2)=2 (two 2x2 blocks), m even -> no extra scalar.

            theta1 = pi/5; theta2 = pi/7;
            chat1 = [ -1j*cos(theta1), -1j*sin(theta1) ; cos(theta1), -sin(theta1) ];
            chat2 = [ -1j*cos(theta2), -1j*sin(theta2) ; cos(theta2), -sin(theta2) ];
            shat1 = [ sin(theta1), cos(theta1) ; 1j*sin(theta1), -1j*cos(theta1) ];
            shat2 = [ sin(theta2), cos(theta2) ; 1j*sin(theta2), -1j*cos(theta2) ];
            Chat = blkdiag(chat1,chat2);
            Shat = blkdiag(shat1,shat2);
            bhatExpctd = [ Chat, conj(Chat) ; Shat, conj(Shat) ];

            import tansacnet.utility.*
            testCase.cbgs = ComplexButterflyCoefficientGenerationSystem(...
                'NumberOfHalfChannels',4);

            angles = [theta1; theta2];
            if canUseGPU
                angles = gpuArray(angles);
            end
            bhatActual = step(testCase.cbgs,angles);

            if canUseGPU
                testCase.verifyClass(bhatActual,'gpuArray');
                bhatActual = gather(bhatActual);
            end
            testCase.verifySize(bhatActual,[8 8]);
            testCase.verifyEqual(bhatActual,bhatExpctd,'AbsTol',1e-10);
        end

        %% Structural properties (parametrized over m)

        function testOutputSize(testCase,m,nblks)
            nAngles = floor(m/2);
            angles = 2*pi*rand(max(nAngles,0),nblks);
            if nAngles == 0
                angles = zeros(0,nblks);
            end
            if canUseGPU
                angles = gpuArray(angles);
            end

            import tansacnet.utility.*
            testCase.cbgs = ComplexButterflyCoefficientGenerationSystem(...
                'NumberOfHalfChannels',m);
            bhatActual = step(testCase.cbgs,angles);

            if canUseGPU
                bhatActual = gather(bhatActual);
            end
            testCase.verifySize(bhatActual,[2*m 2*m nblks]);
        end

        function testExtraScalarEntriesForOddM(testCase)
            % For odd m, the last diagonal entries of Chat/Shat (i.e.
            % Bhat(m,m) and Bhat(2m,m)) must be exactly 1 and 1j,
            % regardless of the (random) angles feeding the other
            % blocks.
            mOdd = 5; % floor(5/2)=2 blocks + 1 extra scalar
            nAngles = floor(mOdd/2);
            angles = 2*pi*rand(nAngles,1);
            if canUseGPU
                angles = gpuArray(angles);
            end

            import tansacnet.utility.*
            testCase.cbgs = ComplexButterflyCoefficientGenerationSystem(...
                'NumberOfHalfChannels',mOdd);
            bhatActual = step(testCase.cbgs,angles);

            if canUseGPU
                bhatActual = gather(bhatActual);
            end
            testCase.verifyEqual(bhatActual(mOdd,mOdd),complex(1,0),'AbsTol',1e-10);
            testCase.verifyEqual(bhatActual(2*mOdd,mOdd),complex(0,1),'AbsTol',1e-10);
        end

        %% Scaled-unitarity: Bhat_k * Bhat_k^H = 2*I  (the key invariant)

        function testScaledUnitarity(testCase,m,nblks,datatype)
            nAngles = floor(m/2);
            angles = cast(2*pi*rand(max(nAngles,1),nblks),datatype);
            if nAngles == 0
                angles = zeros(0,nblks,datatype);
            end
            if canUseGPU
                angles = gpuArray(angles);
            end

            import tansacnet.utility.*
            testCase.cbgs = ComplexButterflyCoefficientGenerationSystem(...
                'NumberOfHalfChannels',m,'DType',datatype);
            bhat = step(testCase.cbgs,angles);

            if canUseGPU
                bhat = gather(bhat);
            end

            expctd = cast(2*eye(2*m),'like',complex(cast(0,datatype)));
            for iBlk = 1:nblks
                actual = bhat(:,:,iBlk)*bhat(:,:,iBlk)';
                testCase.verifyEqual(actual,expctd,'AbsTol', ...
                    cast(1e-6,datatype));
            end
        end

        %% Device / DType

        function testConstructorWithDeviceAndDType(testCase,usegpu,datatype,m)
            if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end

            device_ = [ "cpu", "cuda" ];
            expctdDevice = convertStringsToChars(device_(usegpu+1));
            expctdDType = datatype;

            nAngles = floor(m/2);
            angles = cast(2*pi*rand(max(nAngles,1),1),expctdDType);
            if nAngles == 0
                angles = zeros(0,1,expctdDType);
            end
            if usegpu
                angles = gpuArray(angles);
            end

            import tansacnet.utility.*
            testCase.cbgs = ComplexButterflyCoefficientGenerationSystem(...
                'NumberOfHalfChannels',m, ...
                'Device',expctdDevice, ...
                'DType',expctdDType);
            bhatActual = step(testCase.cbgs,angles);

            if usegpu
                testCase.verifyClass(bhatActual,'gpuArray');
                bhatActual = gather(bhatActual);
            end
            testCase.verifyEqual(testCase.cbgs.Device,expctdDevice);
            testCase.verifyEqual(testCase.cbgs.DType,expctdDType);
            testCase.verifyEqual(class(real(bhatActual)),expctdDType);
        end

        function testMismatchDeviceOnGpu(testCase)
            import tansacnet.utility.*
            testCase.cbgs = ComplexButterflyCoefficientGenerationSystem(...
                'NumberOfHalfChannels',2, ...
                'Device','cuda');

            try
                step(testCase.cbgs,0);
                testCase.verifyFail('Expected CLSUN:DeviceMismatch to be thrown');
            catch ME
                testCase.verifyEqual(ME.identifier,'CLSUN:DeviceMismatch');
            end
        end

        function testMismatchDTypeOnCpu(testCase)
            import tansacnet.utility.*
            testCase.cbgs = ComplexButterflyCoefficientGenerationSystem(...
                'NumberOfHalfChannels',2, ...
                'Device','cpu', ...
                'DType','double');

            try
                step(testCase.cbgs,single(0));
                testCase.verifyFail('Expected CLSUN:DTypeMismatch to be thrown');
            catch ME
                testCase.verifyEqual(ME.identifier,'CLSUN:DTypeMismatch');
            end
        end

        %% PartialDifference (ordinary complex derivative w.r.t. real theta)

        function testPartialDifferenceM2(testCase)
            % m=2, single angle: d(Bhat)/d(theta) via theta+pi/2 trick.
            theta = pi/5;
            chatDiff = [ -1j*cos(theta+pi/2), -1j*sin(theta+pi/2) ; ...
                          cos(theta+pi/2),    -sin(theta+pi/2)   ];
            shatDiff = [  sin(theta+pi/2),     cos(theta+pi/2)   ; ...
                        1j*sin(theta+pi/2),  -1j*cos(theta+pi/2) ];
            dBhatExpctd = [ chatDiff, conj(chatDiff) ; shatDiff, conj(shatDiff) ];

            import tansacnet.utility.*
            testCase.cbgs = ComplexButterflyCoefficientGenerationSystem(...
                'NumberOfHalfChannels',2, ...
                'PartialDifference','on');

            angles = theta;
            if canUseGPU
                angles = gpuArray(angles);
            end
            dBhatActual = step(testCase.cbgs,angles,1);

            if canUseGPU
                dBhatActual = gather(dBhatActual);
            end
            testCase.verifyEqual(dBhatActual,dBhatExpctd,'AbsTol',1e-10);
        end

        function testPartialDifferenceM4ZerosOtherBlock(testCase)
            % m=4, two angles: differentiating w.r.t. theta_1 must leave
            % the theta_2 block (and its conjugate) exactly zero, since
            % Chat_k/Shat_k are block-diagonal (no cross term).
            theta1 = pi/5; theta2 = pi/7;

            import tansacnet.utility.*
            testCase.cbgs = ComplexButterflyCoefficientGenerationSystem(...
                'NumberOfHalfChannels',4, ...
                'PartialDifference','on');

            angles = [theta1; theta2];
            if canUseGPU
                angles = gpuArray(angles);
            end
            dBhatActual = step(testCase.cbgs,angles,1);

            if canUseGPU
                dBhatActual = gather(dBhatActual);
            end
            % Second 2x2 block (rows/cols 3:4 of Chat, i.e. Bhat rows
            % 3:4, cols 3:4 and 7:8) must be all zero.
            testCase.verifyEqual(dBhatActual(3:4,3:4),zeros(2,2),'AbsTol',1e-10);
            testCase.verifyEqual(dBhatActual(3:4,7:8),zeros(2,2),'AbsTol',1e-10);
            testCase.verifyEqual(dBhatActual(7:8,3:4),zeros(2,2),'AbsTol',1e-10);
            testCase.verifyEqual(dBhatActual(7:8,7:8),zeros(2,2),'AbsTol',1e-10);
        end

        function testPartialDifferenceMatchesNumericDerivative(testCase,m)
            % Independent cross-check: compare the analytic
            % PartialDifference output against a plain central-difference
            % numerical derivative of Bhat w.r.t. one angle. This is an
            % ordinary complex derivative of a function of a real
            % variable -- unlike a layer's backward() w.r.t. a real
            % loss, there is no Wirtinger/conjugate-convention ambiguity
            % here (see clsun-tdd-plan.md Step 0 for that separate issue).
            nAngles = floor(m/2);
            testCase.assumeGreaterThan(nAngles,0, ...
                'm too small to have a differentiable angle');

            rng(0);
            angles = 2*pi*rand(nAngles,1);
            pdAng = 1;
            h = 1e-6;
            if canUseGPU
                angles = gpuArray(angles);
            end

            import tansacnet.utility.*
            cbgsOn = ComplexButterflyCoefficientGenerationSystem(...
                'NumberOfHalfChannels',m,'PartialDifference','on');
            dBhatAnalytic = step(cbgsOn,angles,pdAng);

            cbgsOff = ComplexButterflyCoefficientGenerationSystem(...
                'NumberOfHalfChannels',m);
            anglesP = angles; anglesP(pdAng) = anglesP(pdAng) + h;
            anglesM = angles; anglesM(pdAng) = anglesM(pdAng) - h;
            bhatP = step(cbgsOff,anglesP);
            bhatM = step(cbgsOff,anglesM);
            dBhatNumeric = (bhatP - bhatM)/(2*h);

            if canUseGPU
                dBhatAnalytic = gather(dBhatAnalytic);
                dBhatNumeric = gather(dBhatNumeric);
            end
            testCase.verifyEqual(dBhatAnalytic,dBhatNumeric,'AbsTol',1e-6);

            delete(cbgsOn);
            delete(cbgsOff);
        end

    end

end
