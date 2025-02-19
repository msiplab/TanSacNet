classdef lsunAnalysis2dNetworkTestCase < matlab.unittest.TestCase
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
        
        inputSize = {[8 8], [16 8], [8 32], [16 32], [16 16], [32 32]}
        Stride = {[2 2], [2 4], [4 1], [4 4]}; % [1 2] 
        OverlappingFactor = {[1 1], [3 3], [5 5], [1 3], [3 1]};
        nodcleakage = struct( 'true', true, 'false', false);
        datatype = { 'single', 'double' };
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
            inputSize = [32 32];
            numberOfComponents = 1;

            % Expcted values
            expctdTensorSize = [inputSize numberOfComponents];
            expctdStride = [2 2];
            expctdOverlappingFactor = [1 1];
            expctdNumberOfLevels = 1;
            expcedNoDcLeakage = false;
            expctdPrefix = '';
            expctdDevice = 'cuda';
            expctdDType = 'double';

            % Instantiation of target class
            import tansacnet.lsun.*
            net = lsunAnalysis2dNetwork();

            % Verify that net is a subclass of dlnetwork
            testCase.verifyTrue(isa(net, 'dlnetwork'));

            % Actual values
            actualTensorSize = net.TensorSize;
            actualStride = net.Stride;
            actualOverlappingFactor = net.OverlappingFactor;
            actualNumberOfLevels = net.NumberOfLevels;
            actualNoDcLeakage = net.NoDcLeakage;
            actualPrefix = net.Prefix;
            actualDevice = net.Device;
            actualDType = net.DType;

            % Evaluation
            testCase.verifyEqual(actualTensorSize,expctdTensorSize);
            testCase.verifyEqual(actualStride,expctdStride);
            testCase.verifyEqual(actualOverlappingFactor,expctdOverlappingFactor);
            testCase.verifyEqual(actualNumberOfLevels,expctdNumberOfLevels);
            testCase.verifyEqual(actualNoDcLeakage,expcedNoDcLeakage);
            testCase.verifyEqual(actualPrefix,expctdPrefix);
            testCase.verifyEqual(actualDevice,expctdDevice);
            testCase.verifyEqual(actualDType,expctdDType);
        end

        function testConstructorWithProperties(testCase, Stride, OverlappingFactor, ...
                inputSize, nodcleakage, datatype, device)
            % Configuration
            numberOfComponents = 1;
            

            % Expcted values
            expctdTensorSize = [inputSize numberOfComponents];
            expctdStride = Stride;
            expctdOverlappingFactor = OverlappingFactor;
            expctdNumberOfLevels = 1;
            expcedNoDcLeakage = nodcleakage;
            expctdPrefix = '';
            expctdDevice = device;
            expctdDType = datatype;

            % Instantiation of target class
            import tansacnet.lsun.*
            net = lsunAnalysis2dNetwork('InputSize',inputSize, ...
                'Stride',Stride, ...
                'OverlappingFactor',OverlappingFactor, ...
                'DType',datatype,'NoDcLeakage',nodcleakage, ...
                'Device',device);

            % Verify that net is a subclass of dlnetwork
            testCase.verifyTrue(isa(net, 'dlnetwork'));

            % Actual values
            actualTensorSize = net.TensorSize;
            actualStride = net.Stride;
            actualOverlappingFactor = net.OverlappingFactor;
            actualNumberOfLevels = net.NumberOfLevels;
            actualNoDcLeakage = net.NoDcLeakage;
            actualPrefix = net.Prefix;
            actualDevice = net.Device;
            actualDType = net.DType;

            % Evaluation
            testCase.verifyEqual(actualTensorSize,expctdTensorSize);
            testCase.verifyEqual(actualStride,expctdStride);
            testCase.verifyEqual(actualOverlappingFactor,expctdOverlappingFactor);
            testCase.verifyEqual(actualNumberOfLevels,expctdNumberOfLevels);
            testCase.verifyEqual(actualNoDcLeakage,expcedNoDcLeakage);
            testCase.verifyEqual(actualPrefix,expctdPrefix);
            testCase.verifyEqual(actualDevice,expctdDevice);
            testCase.verifyEqual(actualDType,expctdDType);
        end

        function testNetwork(testCase, Stride, datatype, device)

            height = 16;
            width = 16;
            nSamples = 8;
            nComponents = 1;
            import tansacnet.lsun.*
            net = lsunAnalysis2dNetwork('InputSize',[height width], ...
                'Stride',Stride, ...
                'DType',datatype, ...
                'Device',device);
            dlnet = net.dlnetwork();
            %analyzeNetwork(dlnet)
            dlnet_ = initialize(dlnet);
            
            X = rand([height, width, nComponents, nSamples], datatype);
            X = dlarray(X, 'SSCB');
            
            [Zac,Zdc] = forward(dlnet_, X);

            testCase.verifyInstanceOf(extractdata(Zac),datatype);
            testCase.verifyInstanceOf(extractdata(Zdc),datatype);
        end

        function testNetwork_with_Overlap(testCase, Stride,OverlappingFactor, datatype, device)

            height = 16;
            width = 16;
            nSamples = 8;
            nComponents = 1;
            %OverlappingFactor_ = [3 3];
            import tansacnet.lsun.*
            net = lsunAnalysis2dNetwork('InputSize',[height width], ...
                'Stride',Stride, ...
                'OverlappingFactor', OverlappingFactor, ...
                'DType',datatype, ...
                'Device',device);
            dlnet = net.dlnetwork();
            %analyzeNetwork(dlnet)
            dlnet_ = initialize(dlnet);
            
            X = rand([height, width, nComponents, nSamples], datatype);
            X = dlarray(X, 'SSCB');
            
            [Zac,Zdc] = forward(dlnet_, X);

            testCase.verifyInstanceOf(extractdata(Zac),datatype);
            testCase.verifyInstanceOf(extractdata(Zdc),datatype);
        end

        function testForward(testCase, datatype, device)
            import tansacnet.utility.Direction
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-5,single(1e-5));

            Stride = [2 2];
            nSamples = 8;
            nComponents = 1;
            height = 32;
            width = 32;
            nDecs = Stride(Direction.VERTICAL) * Stride(Direction.HORIZONTAL);
            nChsTotal = nDecs;

            X = rand([height, width, nComponents, nSamples], datatype);
          
            % Expected values
            nrows = ceil(height/Stride(Direction.VERTICAL));
            ncols = ceil(width/Stride(Direction.HORIZONTAL));
            ndecs = prod(Stride);
            %expctdZ = zeros(nrows,ncols,ndecs,nSamples,datatype);
            expctdZ_ = zeros(ndecs,nrows,ncols,nSamples,datatype);
            for iSample = 1:nSamples
                % Block DCT
                Y = blockproc(X(:,:,nComponents,iSample),...
                    Stride,@(x) dct2(x.data));
                % Rearrange the DCT Coefs.
                A = blockproc(Y,...
                    Stride,@testCase.permuteDctCoefs_);
                expctdZ_(:,:,:,iSample) = ...
                    ...permute(reshape(A,ndecs,nrows,ncols),[2 3 1]);
                    reshape(A,ndecs,nrows,ncols);
            end
            X_ = expctdZ_;
            %expctdZ = zeros(nChsTotal,nrows,ncols,nSamples,datatype);
            %expctdZ = zeros(nrows,ncols,nChsTotal,nSamples,datatype);

            % initial rotation
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            W0 = repmat(eye(ps,datatype),[1 1 nrows*ncols]);
            U0 = repmat(eye(pa,datatype),[1 1 nrows*ncols]);
            Y_  = zeros(nChsTotal,nrows,ncols,datatype);
            for iSample=1:nSamples
                % Perumation in each block
                Ai = X_(:,:,:,iSample); %permute(X(:,:,:,iSample),[3 1 2]);
                Yi = reshape(Ai,nDecs,nrows,ncols);
                %
                Ys = Yi(1:ps,:);
                Ya = Yi(ps+1:end,:);
                for iblk = 1:(nrows*ncols)
                    Ys(:,iblk) = W0(:,1:ps,iblk)*Ys(:,iblk);
                    Ya(:,iblk) = U0(:,1:pa,iblk)*Ya(:,iblk);
                end
                Y_(1:ps,:,:) = reshape(Ys,ps,nrows,ncols);
                Y_(ps+1:ps+pa,:,:) = reshape(Ya,pa,nrows,ncols);                
                expctdZ(:,:,:,iSample) = Y_; %ipermute(Y,[3 1 2]);
            end
            
            % Channel separation
            expctdZac = permute(expctdZ(2:end,:,:,:),[2 3 1 4]);
            expctdZdc = permute(expctdZ(1,:,:,:),[2 3 1 4]);
            
            %disp(size(expctdZac))

            import tansacnet.lsun.*
            net = lsunAnalysis2dNetwork('InputSize',[height width], ...
                'Stride',Stride, ...
                'DType',datatype, ...
                'Device',device);
            dlnet = net.dlnetwork();
            dlnet_ = initialize(dlnet);
            %analyzeNetwork(dlnet)
            
            X = dlarray(X, 'SSCB');
            [Zac,Zdc] = forward(dlnet_, X);
            
            testCase.verifyInstanceOf((extractdata(Zac)),datatype);
            testCase.verifyThat(extractdata(Zdc),...
                IsEqualTo(expctdZdc,'Within',tolObj));
            testCase.verifyThat(extractdata(Zac),...
                IsEqualTo(expctdZac,'Within',tolObj));
        end

        function testForward_with_Overlap(testCase,Stride,OverlappingFactor,datatype,device)
            import tansacnet.utility.Direction
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-5,single(1e-5));

            nSamples = 8;
            nComponents = 1;
            height = 32;
            width = 32;
            mus = 1;
            nDecs = Stride(Direction.VERTICAL) * Stride(Direction.HORIZONTAL);
            nChsTotal = nDecs;

            X = rand([height, width, nComponents, nSamples], datatype);
            angles = [];
            % Expected values
            nrows = ceil(height/Stride(Direction.VERTICAL));
            ncols = ceil(width/Stride(Direction.HORIZONTAL));
            ndecs = prod(Stride);
            %expctdZ = zeros(nrows,ncols,ndecs,nSamples,datatype);
            expctdZ_ = zeros(ndecs,nrows,ncols,nSamples,datatype);
            for iSample = 1:nSamples
                % Block DCT
                Y = blockproc(X(:,:,nComponents,iSample),...
                    Stride,@(x) dct2(x.data));
                % Rearrange the DCT Coefs.
                A = blockproc(Y,...
                    Stride,@testCase.permuteDctCoefs_);
                expctdZ_(:,:,:,iSample) = ...
                    ...permute(reshape(A,ndecs,nrows,ncols),[2 3 1]);
                    reshape(A,ndecs,nrows,ncols);
            end
            X_ = expctdZ_;
            %expctdZ = zeros(nChsTotal,nrows,ncols,nSamples,datatype);
            %expctdZ = zeros(nrows,ncols,nChsTotal,nSamples,datatype);

            % initial rotation
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            W0 = repmat(eye(ps,datatype),[1 1 nrows*ncols]);
            U0 = repmat(eye(pa,datatype),[1 1 nrows*ncols]);
            Y_  = zeros(nChsTotal,nrows,ncols,datatype);
            for iSample=1:nSamples
                % Perumation in each block
                Ai = X_(:,:,:,iSample); %permute(X(:,:,:,iSample),[3 1 2]);
                Yi = reshape(Ai,nDecs,nrows,ncols);
                %
                Ys = Yi(1:ps,:);
                Ya = Yi(ps+1:end,:);
                for iblk = 1:(nrows*ncols)
                    Ys(:,iblk) = W0(:,1:ps,iblk)*Ys(:,iblk);
                    Ya(:,iblk) = U0(:,1:pa,iblk)*Ya(:,iblk);
                end
                Y_(1:ps,:,:) = reshape(Ys,ps,nrows,ncols);
                Y_(ps+1:ps+pa,:,:) = reshape(Ya,pa,nrows,ncols);    
                eY(:,:,:,iSample) = Y_; %ipermute(Y,[3 1 2]);
            end

            for iCmp = 1:nComponents
                for iOrderH = 2:2:OverlappingFactor(2)-1
                    % Atom extension (Right shift)
                    eY = testCase.AtomExt('Right',eY,pa,ps);
                   
                    % Intermediate rotation
                    eY = testCase.intRotation(eY,mus,pa,ps,nrows,ncols,nSamples,nChsTotal,angles,datatype,device);
                    
                    % Atom extension (Left shift)
                    eY = testCase.AtomExt('Left',eY,pa,ps);
                    
                    % Intermediate rotation
                    eY = testCase.intRotation(eY,mus,pa,ps,nrows,ncols,nSamples,nChsTotal,angles,datatype,device);
                end
                
                for iOrderV = 2:2:OverlappingFactor(1)-1
                    % Atom extension (Down shift)
                    eY = testCase.AtomExt('Down',eY,pa,ps);
               
                    % Intermediate rotation
                    eY = testCase.intRotation(eY,mus,pa,ps,nrows,ncols,nSamples,nChsTotal,angles,datatype,device);
                    
                    % Atom extension (Up shift)
                    eY = testCase.AtomExt('Up',eY,pa,ps);
                  
                    % Intermediate rotation
                    eY = testCase.intRotation(eY,mus,pa,ps,nrows,ncols,nSamples,nChsTotal,angles,datatype,device);
                end
            
            % Channel separation
            expctdZac = permute(eY(2:end,:,:,:),[2 3 1 4]);
            expctdZdc = permute(eY(1,:,:,:),[2 3 1 4]);
            end
            
            %disp(expctdZdc)

            import tansacnet.lsun.*
            net = lsunAnalysis2dNetwork('InputSize',[height width], ...
                'Stride',Stride, ...
                'OverlappingFactor', OverlappingFactor, ...
                'DType',datatype, ...
                'Device',device);
            dlnet = net.dlnetwork();
            dlnet_ = initialize(dlnet);
            %analyzeNetwork(dlnet)
            
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
            coefs = x.data;
            cee = coefs(1:2:end,1:2:end);
            coo = coefs(2:2:end,2:2:end);
            coe = coefs(2:2:end,1:2:end);
            ceo = coefs(1:2:end,2:2:end);
            value = [ cee(:) ; coo(:) ; coe(:) ; ceo(:) ];
        end

        function eY = AtomExt(dir,Yn,pa,ps)
            if strcmp(dir,'Right')
                shift = [ 0 0  1 0 ];
            elseif strcmp(dir,'Left')
                shift = [ 0 0 -1 0 ];
            elseif strcmp(dir,'Down')
                shift = [ 0  1 0 0 ];
            elseif strcmp(dir,'Up')
                shift = [ 0 -1 0 0 ];
            else
                shift = [ 0 0 0 0 ];
            end
            eYs = Yn(1:ps,:,:,:);
            eYa = Yn(ps+1:ps+pa,:,:,:);
            eY =  [ eYs+eYa ; eYs-eYa ]/sqrt(2);
            % Block circular shift
            eY(ps+1:ps+pa,:,:,:) = circshift(eY(ps+1:ps+pa,:,:,:),shift);
            % Block butterfly
            eYs = eY(1:ps,:,:,:);
            eYa = eY(ps+1:ps+pa,:,:,:);
            eY =  [ eYs+eYa ; eYs-eYa ]/sqrt(2);
        end

        function irY = intRotation(eY,mus,pa,ps,nrows,ncols,nSamples,nChsTotal,angles,datatype,device)
            import tansacnet.utility.*
            genU = OrthonormalMatrixGenerationSystem('DType',datatype,'Device',device);
            %angles = zeros((nChsTotal-2)*nChsTotal/8,nrows*ncols,datatype);
            if isempty(angles)
                angles = zeros((nChsTotal-2)*nChsTotal/8,nrows*ncols,datatype);
            elseif isscalar(angles)
                angles = angles*ones((nChsTotal-2)*nChsTotal/8,nrows*ncols,'like',angles); 
            end
            if isempty(mus)
                mus = ones(pa,nrows*ncols,datatype);   
            elseif isscalar(mus)
                mus = mus*ones(pa,nrows*ncols,datatype);  
            end
            if device == "cuda"
                angles = gpuArray(angles);
                mus = gpuArray(mus);
            end
            Un = genU.step(angles,mus);
            irY = eY;

            %UnT = repmat(mus*eye(pa,datatype),[1 1 nrows*ncols]);
            iYa = reshape(irY(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            Za = zeros(size(iYa),'like',iYa);
            for iSample=1:nSamples
                for iblk = 1:(nrows*ncols)
                    Za(:,iblk,iSample) = Un(:,:,iblk)*iYa(:,iblk,iSample);
                end
            end
            irY(ps+1:ps+pa,:,:,:) = reshape(Za,pa,nrows,ncols,nSamples);
        end
    end

end