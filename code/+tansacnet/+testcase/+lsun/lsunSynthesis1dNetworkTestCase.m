classdef lsunSynthesis1dNetworkTestCase < matlab.unittest.TestCase
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
        
        inputSize = {32,64}
        Stride = {2,4}; % [1 2] 
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
            net = lsunSynthesis1dNetwork();

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
            net = lsunSynthesis1dNetwork('InputSize',inputSize, ...
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
            nDecs = prod(Stride);
            nChsTotal = nDecs;
            nblks = ceil(seqlen/Stride);
            
            import tansacnet.lsun.*
            Xac = dlarray(randn(nChsTotal-1,1,nblks,nSamples,datatype),'SSCB');
            % 1 x 1 x nBlks x nSamples
            Xdc = dlarray(randn(1,1,nblks,nSamples,datatype),'SSCB');

            SYNnet = lsunSynthesis1dNetwork('InputSize',seqlen, ...
                'Stride',Stride, ...
                'OverlappingFactor',OverlappingFactor,...
                'DType',datatype, ...
                'Device',device);
            SYNdlnet = SYNnet.dlnetwork();
            analyzeNetwork(SYNdlnet)
            SYNdlnet_ = initialize(SYNdlnet);
            Z = forward(SYNdlnet_, Xac, Xdc);

            testCase.verifyInstanceOf(extractdata(Z),datatype);  
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
            nblks = ceil(seqlen/stride);
            
     % Expected Values
            Xac = randn(nChsTotal-1,1,nblks,nSamples,datatype);
            % 1 x 1 x nBlks x nSamples
            Xdc = randn(1,1,nblks,nSamples,datatype);

            % Channel Concatanation
            Z_CC = cat(1,Xdc,Xac);

            % Final Rotation
            V0T = repmat(eye(nChsTotal,datatype),[1 1 nblks]);
            Y_ = Z_CC;
            for iSample=1:nSamples
                for iblk = 1:nblks
                    Y_(:,:,iblk,iSample) = V0T(:,:,iblk)*Y_(:,:,iblk,iSample); 
                end
            end
            Z_FR = Y_;

            % Block IDCT
            Z_IDCT = zeros(1,seqlen,1,nSamples,datatype);
            for iSample = 1:nSamples
                A = reshape(Z_FR(:,:,:,iSample),stride,[]);
                if stride > 1
                    Yi = testCase.permuteIdctCoefs_(A,stride);
                    Z_IDCT(:,:,:,iSample) = ...
                        reshape(idct(Yi),1,seqlen,1);
                else
                    Z_IDCT(:,:,:,iSample) = ...
                        reshape(A,1,seqlen,1);                    
                end
            end
            expctdZ = Z_IDCT;

            import tansacnet.lsun.*
            net = lsunSynthesis1dNetwork('InputSize',seqlen, ...
                'Stride',Stride, ...
                'DType',datatype, ...
                'Device',device);
            dlnet = net.dlnetwork();
            analyzeNetwork(dlnet)
            dlnet_ = initialize(dlnet);

            Xac = dlarray(Xac,'SSCB');
            Xdc = dlarray(Xdc,'SSCB');

            actualZ = forward(dlnet_, Xac,Xdc);

            testCase.verifyInstanceOf((extractdata(actualZ)),datatype);
            testCase.verifyThat(extractdata(actualZ),...
                IsEqualTo(expctdZ,'Within',tolObj));
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
            mus = 1;
            nDecs = prod(stride);
            nChsTotal = nDecs;
            nblks = ceil(seqlen/stride);
            ovlpFactor = OverlappingFactor;
            
     % Expected Values
            Xac = randn(nChsTotal-1,1,nblks,nSamples,datatype);
            % 1 x 1 x nBlks x nSamples
            Xdc = randn(1,1,nblks,nSamples,datatype);

            
            if nComponents > 1
                numOutputs = nComponents;
                nChsPerCmp = size(Xac,1)/numOutputs;
                Xac_ = cell(numOutputs,1);
                for idx = 1:numOutputs
                    Xac_{idx} = ...
                        Xac((idx-1)*nChsPerCmp+1:idx*nChsPerCmp,:,:,:);
                end
                nChsPerCmp = size(Xdc,1)/numOutputs;
                Xdc_ = cell(numOutputs,1);
                for idx = 1:numOutputs
                    Xdc_{idx} = ...
                        Xdc((idx-1)*nChsPerCmp+1:idx*nChsPerCmp,:,:,:);
                end
            end

            for iCmp = 1:nComponents
                % Channel Concatanation
                Z_CC = cat(1,Xdc,Xac);
                Zi = Z_CC;
                for iOrderV = 2:2:ovlpFactor-1
                    % Intermediate rotation
                    Zi = testCase.IntermediatROT_(Zi,nChsTotal,nblks,nSamples,mus,datatype);
                    % Right Shift
                    Zi = testCase.AtomEXT_(Zi,'Right','Top',nChsTotal);
                    % Intermediate rotation
                    Zi = testCase.IntermediatROT_(Zi,nChsTotal,nblks,nSamples,mus,datatype);
                     % Left shift
                    Zi = testCase.AtomEXT_(Zi,'Left','Bottom',nChsTotal);
                end 
            end

            % Final Rotation
            V0T = repmat(eye(nChsTotal,datatype),[1 1 nblks]);
            Y_ = Zi;
            for iSample=1:nSamples
                for iblk = 1:nblks
                    Y_(:,:,iblk,iSample) = V0T(:,:,iblk)*Y_(:,:,iblk,iSample); 
                end
            end
            Z_FR = Y_;

            % Block IDCT
            Z_IDCT = zeros(1,seqlen,1,nSamples,datatype);
            for iSample = 1:nSamples
                A = reshape(Z_FR(:,:,:,iSample),stride,[]);
                if stride > 1
                    Yi = testCase.permuteIdctCoefs_(A,stride);
                    Z_IDCT(:,:,:,iSample) = ...
                        reshape(idct(Yi),1,seqlen,1);
                else
                    Z_IDCT(:,:,:,iSample) = ...
                        reshape(A,1,seqlen,1);                    
                end
            end
            expctdZ = Z_IDCT;

            import tansacnet.lsun.*
            net = lsunSynthesis1dNetwork('InputSize',seqlen, ...
                'Stride',Stride, ...
                'OverlappingFactor',OverlappingFactor,...
                'DType',datatype, ...
                'Device',device);
            dlnet = net.dlnetwork();
            analyzeNetwork(dlnet)
            dlnet_ = initialize(dlnet);

            Xac = dlarray(Xac,'SSCB');
            Xdc = dlarray(Xdc,'SSCB');

            actualZ = forward(dlnet_, Xac,Xdc);

            testCase.verifyInstanceOf((extractdata(actualZ)),datatype);
            testCase.verifyThat(extractdata(actualZ),...
                IsEqualTo(expctdZ,'Within',tolObj));

        end

    end
    methods (Static, Access = private)

        function value = permuteDctCoefs_(x)
            coefs = x;
            ce = coefs(1:2:end,:);
            co = coefs(2:2:end,:);
            value = [ ce ; co ];
        end

        function value = permuteIdctCoefs_(x,stride)
            coefs = x;
            nHDecse = ceil(stride/2);
            nHDecso = floor(stride/2);
            ce = coefs(         1:  nHDecse,:);
            co = coefs(nHDecse+1:nHDecse+nHDecso,:);
            value = zeros(stride,numel(coefs)/stride,'like',x);
            value(1:2:stride,:) = ce;
            value(2:2:stride,:) = co;
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