classdef lsunAnalysis1dNetworkTestCase < matlab.unittest.TestCase
    %LSUNANALYSIS2dNETWORKTESTCASE 
    %
    % Requirements: MATLAB R2024a
    %
    % Copyright (c) 2025, Shogo MURAMATSU, Yasas GODAGE
    %
    % All rights reserved.
    %
    % Contact address: Shogo MURAMATSU
    %                Faculty of Engineering, Niigata University,
    %                8050 2-no-cho Ikarashi, Nishi-ku,
    %                Niigata, 950-2181, JAPAN
    %
    % http://msiplab.eng.niigata-u.ac.jp/ 
    
    properties (TestParameter)
        
        inputSize = {32}
        Stride = {4}; % [1 2] 
        OverlappingFactor = {1, 3, 5};
        %nodcleakage = struct( 'true', true, 'false', false);
        datatype = { 'single' };
        device = {'cpu', 'cuda'};

        
    end

    methods (TestClassSetup)
        % テスト クラス全体の共有セットアップ
    end

    methods (TestMethodSetup)
        % 各テストのセットアップ
    end

    methods (Test)
        % テスト メソッド

        function testDefaultConstructor(testCase)
            % Configuration
            inputSize = 32;
            numberOfComponents = 1;

            % Expcted values
            expctdTensorSize = [1 inputSize numberOfComponents];
            expctdStride = 1;
            expctdOverlappingFactor = 1;
            expctdNumberOfLevels = 1;
            expctdPrefix = '';
            expctdDevice = 'cuda';
            expctdDType = 'double';

            % Instantiation of target class
            import tansacnet.lsun.*
            net = lsunAnalysis1dNetwork();

            % Verify that net is a subclass of dlnetwork
            testCase.verifyTrue(isa(net, 'dlnetwork'));

            % Actual values
            actualTensorSize = net.TensorSize;
            actualStride = net.Stride;
            actualOverlappingFactor = net.OverlappingFactor;
            actualNumberOfLevels = net.NumberOfLevels;
            actualPrefix = net.Prefix;
            actualDevice = net.Device;
            actualDType = net.DType;

            % Evaluation
            testCase.verifyEqual(actualTensorSize,expctdTensorSize);
            testCase.verifyEqual(actualStride,expctdStride);
            testCase.verifyEqual(actualOverlappingFactor,expctdOverlappingFactor);
            testCase.verifyEqual(actualNumberOfLevels,expctdNumberOfLevels);
            testCase.verifyEqual(actualPrefix,expctdPrefix);
            testCase.verifyEqual(actualDevice,expctdDevice);
            testCase.verifyEqual(actualDType,expctdDType);
        end

        function testConstructorWithProperties(testCase, Stride, OverlappingFactor, ...
                inputSize, datatype, device)
            % Configuration
            numberOfComponents = 1;
            
            % Expcted values
            expctdTensorSize = [1 inputSize numberOfComponents];
            expctdStride = Stride;
            expctdOverlappingFactor = OverlappingFactor;
            expctdNumberOfLevels = 1;
            %expcedNoDcLeakage = nodcleakage;
            expctdPrefix = '';
            expctdDevice = device;
            expctdDType = datatype;

            % Instantiation of target class
            import tansacnet.lsun.*
            net = lsunAnalysis1dNetwork('InputSize',inputSize, ...
                'Stride',Stride, ...
                'OverlappingFactor',OverlappingFactor, ...
                'DType',datatype, ...
                'Device',device);

            % Verify that net is a subclass of dlnetwork
            testCase.verifyTrue(isa(net, 'dlnetwork'));

            % Actual values
            actualTensorSize = net.TensorSize;
            actualStride = net.Stride;
            actualOverlappingFactor = net.OverlappingFactor;
            actualNumberOfLevels = net.NumberOfLevels;
            %actualNoDcLeakage = net.NoDcLeakage;
            actualPrefix = net.Prefix;
            actualDevice = net.Device;
            actualDType = net.DType;

            % Evaluation
            testCase.verifyEqual(actualTensorSize,expctdTensorSize);
            testCase.verifyEqual(actualStride,expctdStride);
            testCase.verifyEqual(actualOverlappingFactor,expctdOverlappingFactor);
            testCase.verifyEqual(actualNumberOfLevels,expctdNumberOfLevels);
            %testCase.verifyEqual(actualNoDcLeakage,expcedNoDcLeakage);
            testCase.verifyEqual(actualPrefix,expctdPrefix);
            testCase.verifyEqual(actualDevice,expctdDevice);
            testCase.verifyEqual(actualDType,expctdDType);
        end

        function testNetwork(testCase, Stride, OverlappingFactor, datatype, device)
            % TODO : Double precision fix
            seqlen = 100;
            nSamples = 8;
            nComponents = 1;
            import tansacnet.lsun.*
            net = lsunAnalysis1dNetwork('InputSize',seqlen, ...
                'Stride',Stride, ...
                'OverlappingFactor',OverlappingFactor,...
                'DType',datatype, ...
                'Device',device);
            dlnet = net.dlnetwork();
            analyzeNetwork(dlnet)
            dlnet_ = initialize(dlnet);
            
            X = rand([1, 100, 1, 1], datatype);
            %disp(whos('X'))
            X = dlarray(X, 'SSCB');
            
            [Zac,Zdc] = forward(dlnet_, X);

            testCase.verifyInstanceOf(extractdata(Zac),datatype);
            testCase.verifyInstanceOf(extractdata(Zdc),datatype);
        end

        function testForward(testCase, Stride, datatype, device)
            import tansacnet.utility.Direction
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-5,single(1e-5));

            % Parameters
            stride = Stride;
            nSamples = 8;
            seqlen = 100;
            nDecs = prod(stride);
            nChsTotal = nDecs;
            
     % Expected Values
            X = rand(1, seqlen, 1, nSamples, datatype);

            %
            nblks = ceil(seqlen/stride);
            DCT_Z = zeros(stride,1,nblks,nSamples,datatype);
            for iSample = 1:nSamples
            % Block DCT
                U = reshape(X(:,:,:,iSample),stride,[]);
                if stride > 1
                    Y = dct(U);
                    % Rearrange the DCT Coefs.
                    A = testCase.permuteDctCoefs_(Y);
                    DCT_Z(:,:,:,iSample) = ...
                        reshape(A,stride,1,nblks,1);
                else
                    DCT_Z(:,:,:,iSample) = ...
                        reshape(U,stride,1,nblks,1);
                end
            end

            %Initial Rotation
            V0 = repmat(eye(nChsTotal,datatype),[1 1 nblks]);
            INIT_Z = zeros(nChsTotal,1,nblks,nSamples,datatype);

            for iSample=1:nSamples
                % Perumation in each block
                Ai = DCT_Z(:,:,:,iSample); %permute(X(:,:,:,iSample),[3 1 2]);
                Yi = reshape(Ai,nChsTotal,nblks);
                %       
                for iblk = 1:nblks
                    Yi(:,iblk) = V0(:,:,iblk)*Yi(:,iblk);
                end               
                INIT_Z(:,:,:,iSample) = reshape(Yi,nChsTotal,1,nblks);
            end
            % Channel Separation
            % (nChsTotal-1) x 1 x nBlks x nSamples 
            expctdZac = INIT_Z(2:end,:,:,:);
            % 1 x 1 x nBlks x nSamples
            expctdZdc = INIT_Z(1,:,:,:);

            import tansacnet.lsun.*
            net = lsunAnalysis1dNetwork('InputSize',seqlen, ...
                'Stride',Stride, ...
                'DType',datatype, ...
                'Device',device);
            dlnet = net.dlnetwork();
            analyzeNetwork(dlnet)
            dlnet_ = initialize(dlnet);

            X = dlarray(X, 'SSCB');

            [Zac,Zdc] = forward(dlnet_, X);

            testCase.verifyInstanceOf((extractdata(Zac)),datatype);
            testCase.verifyThat(extractdata(Zdc),...
                IsEqualTo(expctdZdc,'Within',tolObj));
            testCase.verifyThat(extractdata(Zac),...
                IsEqualTo(expctdZac,'Within',tolObj));
        end

        function testForward_With_intermediateROT(testCase, Stride, OverlappingFactor, datatype, device)
            import tansacnet.utility.Direction
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-5,single(1e-5));
            
            % Parameters
            stride = Stride;
            nSamples = 8;
            seqlen = 100;
            nComponents = 1;
            nDecs = prod(stride);
            nChsTotal = nDecs;
            ovlpFactor = OverlappingFactor;
            mus = 1;
            
     % Expected Values
            X = rand(1, seqlen, 1, nSamples, datatype);
            X_ = gpuArray(X);
            
            nblks = ceil(seqlen/stride);
            DCT_Z = zeros(stride,1,nblks,nSamples,datatype);
            for iSample = 1:nSamples
            % Block DCT
                U = reshape(X_(:,:,:,iSample),stride,[]);
                if stride > 1
                    Y = dct(U);
                    % Rearrange the DCT Coefs.
                    A = testCase.permuteDctCoefs_(Y);
                    DCT_Z(:,:,:,iSample) = ...
                        reshape(A,stride,1,nblks,1);
                else
                    DCT_Z(:,:,:,iSample) = ...
                        reshape(U,stride,1,nblks,1);
                end
            end

            %Initial Rotation
            V0 = repmat(eye(nChsTotal,datatype),[1 1 nblks]);
            INIT_Z = zeros(nChsTotal,1,nblks,nSamples,datatype);

            for iSample=1:nSamples
                % Perumation in each block
                Ai = DCT_Z(:,:,:,iSample); %permute(X(:,:,:,iSample),[3 1 2]);
                Yi = reshape(Ai,nChsTotal,nblks);
                %       
                for iblk = 1:nblks
                    Yi(:,iblk) = V0(:,:,iblk)*Yi(:,iblk);
                end               
                INIT_Z(:,:,:,iSample) = reshape(Yi,nChsTotal,1,nblks);
            end
            Zi = INIT_Z;
            for iCmp = 1:nComponents
                for iOrderV = 2:2:ovlpFactor-1
                    % Right Shift
                    Zi = testCase.AtomEXT_(Zi,'Right','Bottom',nChsTotal);
                    % Intermediate rotation
                    Zi = testCase.IntermediatROT_(Zi,nChsTotal,nblks,nSamples,mus,datatype);
                    % Left shift
                    Zi = testCase.AtomEXT_(Zi,'Left','Top',nChsTotal);
                    % Intermediate rotation
                    Zi = testCase.IntermediatROT_(Zi,nChsTotal,nblks,nSamples,mus,datatype);
                end
                % Channel Separation
                % (nChsTotal-1) x 1 x nBlks x nSamples
                expctdZac = Zi(2:end,:,:,:);
                % 1 x 1 x nBlks x nSamples
                expctdZdc = Zi(1,:,:,:);
            end
            import tansacnet.lsun.*
            net = lsunAnalysis1dNetwork('InputSize',seqlen, ...
                'Stride',Stride, ...
                'OverlappingFactor',OverlappingFactor,...
                'DType',datatype, ...
                'Device',device);
            dlnet = net.dlnetwork();
            analyzeNetwork(dlnet)
            dlnet_ = initialize(dlnet);

            X = dlarray(X, 'SSCB');

            [Zac,Zdc] = forward(dlnet_, X);

            testCase.verifyInstanceOf((extractdata(Zac)),datatype);
            testCase.verifyThat(extractdata(Zdc),...
                IsEqualTo(expctdZdc,'Within',tolObj));
            testCase.verifyThat(extractdata(Zac),...
                IsEqualTo(expctdZac,'Within',tolObj));

        end

    end
    methods (Static, Access = private)

        function value = permuteDctCoefs_(x)
            coefs = x;
            ce = coefs(1:2:end,:);
            co = coefs(2:2:end,:);
            value = [ ce ; co ];
        end

        function Z = AtomEXT_(X,dir,target,nChsTotal)
            if strcmp(dir,'Right')
                shift = [ 0 0 1 ];
            elseif strcmp(dir,'Left')
                shift = [ 0 0 -1 ];
            else
                shift = [ 0 0 0 ];
            end
            %
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            Yt = X(1:pt,:,:,:);
            Yb = X(pt+1:pt+pb,:,:,:);
            % Block circular shift
            if strcmp(target,'Bottom')
                Yb = circshift(Yb,shift);
            elseif strcmp(target,'Top')
                Yt = circshift(Yt,shift);
            end
                % Output
            Z = cat(1,Yt,Yb);
        end

        function Z = IntermediatROT_(X,nChsTotal,nblks,nSamples,mus,datatype)

            % nChsTotal x 1 x nBlks x nSamples
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            WnT = repmat(mus*eye(pt,datatype),[1 1 nblks]);
            UnT = repmat(mus*eye(pb,datatype),[1 1 nblks]);
            Y = X;
            Yt = reshape(Y(1:pt,:,:,:),pt,nblks,nSamples);
            Yb = reshape(Y(pt+1:pt+pb,:,:,:),pb,nblks,nSamples);
            Zt = zeros(size(Yt),'like',Yt);
            Zb = zeros(size(Yb),'like',Yb);
            for iSample=1:nSamples
                for iblk = 1:nblks
                    Zt(:,iblk,iSample) = WnT(:,:,iblk)*Yt(:,iblk,iSample);
                    Zb(:,iblk,iSample) = UnT(:,:,iblk)*Yb(:,iblk,iSample);
                end
            end
            Y(1:pt,:,:,:) = reshape(Zt,pt,1,nblks,nSamples);
            Y(pt+1:pt+pb,:,:,:) = reshape(Zb,pb,1,nblks,nSamples);
            Z = Y;
        end


    end
end