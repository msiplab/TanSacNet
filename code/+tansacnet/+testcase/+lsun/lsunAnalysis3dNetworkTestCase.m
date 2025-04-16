classdef lsunAnalysis3dNetworkTestCase < matlab.unittest.TestCase
    %LSUNANALYSIS3dNETWORKTESTCASE 
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

        inputSize = {[8 8 8], [16 8 8], [16 16 8], [8 16 16], [8 8 16], [16 16 16], [32 32 32]};
        Stride = { [2 2 2], [4 2 2], [4 4 2], [2 4 4], [2 2 4], [4 4 4] };
        OverlappingFactor = { [1 1 1], [3 3 3], [5 5 5], [3 1 1], [3 3 1], [1 3 3], [1 1 3]};
        nodcleakage = struct( 'true',true , 'false',false);
        datatype = {'single','double'};
        device = {'cpu','cuda'};


    end

    methods (TestClassSetup)

    end

    methods (TestMethodSetup)

    end

    methods (Test)


        function testDefaultConstructor(testCase)
            % Configuration
            inputSize = [32 32 32];
            numberofComponents = 1;

            % Expected values
            expctdTensorSize = [inputSize numberofComponents];
            expctdStride = [2 2 2];
            expctdOverlappingFactor = [1 1 1];
            expctdNumberOfLevels = 1;
            expctdNoDcLeakage = false;
            expctdPrefix = '';
            expctdDevice = 'cuda';
            expctdDType = 'double';

            % Instantiation of target class
            import tansacnet.lsun.*
            net = lsunAnalysis3dNetwork();

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
            testCase.verifyEqual(actualNoDcLeakage,expctdNoDcLeakage);
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
            expctdNoDcLeakage = nodcleakage;
            expctdPrefix = '';
            expctdDevice = device;
            expctdDType = datatype;

            % Instantiation of target class
            import tansacnet.lsun.*
            net = lsunAnalysis3dNetwork('InputSize',inputSize, ...
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
            testCase.verifyEqual(actualNoDcLeakage,expctdNoDcLeakage);
            testCase.verifyEqual(actualPrefix,expctdPrefix);
            testCase.verifyEqual(actualDevice,expctdDevice);
            testCase.verifyEqual(actualDType,expctdDType);
        end

        function testNetwork(testCase, Stride, datatype, device)

            height = 16;
            width = 16;
            depth = 16;
            nSamples = 8;
            nComponents = 1;
            import tansacnet.lsun.*
            net = lsunAnalysis3dNetwork('InputSize',[height width depth], ...
                'Stride',Stride, ...
                'DType',datatype, ...
                'Device',device);
            dlnet = net.dlnetwork();
            %analyzeNetwork(dlnet)
            dlnet_ = initialize(dlnet);
            
            X = rand([height, width, depth, nComponents, nSamples], datatype);
            X = dlarray(X, 'SSSCB');
            
            [Zac,Zdc] = forward(dlnet_, X);

            testCase.verifyInstanceOf(extractdata(Zac),datatype);
            testCase.verifyInstanceOf(extractdata(Zdc),datatype);
        end

        function testNetwork_with_Overlap(testCase, Stride, OverlappingFactor, datatype, device)

            height = 16;
            width = 16;
            depth = 16;
            nSamples = 8;
            nComponents = 1;
            import tansacnet.lsun.*
            net = lsunAnalysis3dNetwork('InputSize',[height width depth], ...
                'Stride',Stride, ...
                'OverlappingFactor', OverlappingFactor, ...
                'DType',datatype, ...
                'Device',device);
            dlnet = net.dlnetwork();
            %analyzeNetwork(dlnet)
            dlnet_ = initialize(dlnet);
            
            X = rand([height, width, depth, nComponents, nSamples], datatype);
            X = dlarray(X, 'SSSCB');
            
            [Zac,Zdc] = forward(dlnet_, X);

            testCase.verifyInstanceOf(extractdata(Zac),datatype);
            testCase.verifyInstanceOf(extractdata(Zdc),datatype);
        end

        function testForward(testCase, datatype, device)
            import tansacnet.utility.Direction
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-5,single(1e-5));

            Stride = [2 2 2];
            nSamples = 8;
            nComponents = 1;
            height = 32;
            width = 32;
            depth = 32;
            nDecs = Stride(Direction.VERTICAL) * Stride(Direction.HORIZONTAL) * Stride(Direction.DEPTH);
            nChsTotal = nDecs;

            X = rand([height, width, depth, nComponents, nSamples], datatype);
          
            % Expected values
            nrows = ceil(height/Stride(Direction.VERTICAL));
            ncols = ceil(width/Stride(Direction.HORIZONTAL));
            nlays = ceil(depth/Stride(Direction.DEPTH));
            ndecs = prod(Stride);
            %expctdZ = zeros(nrows,ncols,nlays,ndecs,nSamples,datatype);
            expctdZ_ = zeros(ndecs,nrows,ncols,nlays,nSamples,datatype);
            E0 = testCase.getMatrixE0_(Stride);
            for iSample = 1:nSamples
                for i = size(X,3)
                    % Block DCT
                    Y = testCase.vol2col_(X(:,:,:,1,iSample),Stride,...
                    [nrows,ncols,nlays]);
                % Rearrange the DCT Coefs.
                A = E0*Y;
                expctdZ_(:,:,:,:,iSample) = ...
                    ...permute(reshape(A,ndecs,nrows,ncols,nlays),[2 3 4 1]);
                    reshape(A,ndecs,nrows,ncols,nlays);
                end
            end
            X_ = expctdZ_;
            %expctdZ = zeros(nChsTotal,nrows,ncols,nlays,nSamples,datatype);
            %expctdZ = zeros(nrows,ncols,nlays,nChsTotal,nSamples,datatype);

            % initial rotation
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            W0 = repmat(eye(ps,datatype),[1 1 nrows*ncols*nlays]);
            U0 = repmat(eye(pa,datatype),[1 1 nrows*ncols*nlays]);
            Y_  = zeros(nChsTotal,nrows,ncols,nlays,datatype);
            for iSample=1:nSamples
                % Perumation in each block
                Ai = X_(:,:,:,:,iSample); %permute(X(:,:,:,:,iSample),[4 1 2 3]);
                Yi = reshape(Ai,nDecs,nrows,ncols,nlays);
                %
                Ys = Yi(1:ps,:);
                Ya = Yi(ps+1:end,:);
                for iblk = 1:(nrows*ncols*nlays)
                    Ys(:,iblk) = W0(:,1:ps,iblk)*Ys(:,iblk);
                    Ya(:,iblk) = U0(:,1:pa,iblk)*Ya(:,iblk);
                end
                Y_(1:ps,:,:,:) = reshape(Ys,ps,nrows,ncols,nlays);
                Y_(ps+1:ps+pa,:,:,:) = reshape(Ya,pa,nrows,ncols,nlays);                
                expctdZ(:,:,:,:,iSample) = Y_; %ipermute(Y,[4 1 2 3]);
            end
            
            % Channel separation
            expctdZac = permute(expctdZ(2:end,:,:,:,:),[2 3 4 1 5]);
            expctdZdc = permute(expctdZ(1,:,:,:,:),[2 3 4 1 5]);

            import tansacnet.lsun.*
            net = lsunAnalysis3dNetwork('InputSize',[height width depth], ...
                'Stride',Stride, ...
                'DType',datatype, ...
                'Device',device);
            dlnet = net.dlnetwork();
            dlnet_ = initialize(dlnet);
            % analyzeNetwork(dlnet)
            
            X = dlarray(X, 'SSSCB');
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
            depth = 32;
            mus = 1;
            nDecs = Stride(Direction.VERTICAL) * Stride(Direction.HORIZONTAL)*Stride(Direction.DEPTH);
            nChsTotal = nDecs;

            X = rand([height, width, depth, nComponents, nSamples], datatype);
            angles = [];
            % Expected values
            nrows = ceil(height/Stride(Direction.VERTICAL));
            ncols = ceil(width/Stride(Direction.HORIZONTAL));
            nlays = ceil(depth/Stride(Direction.DEPTH));
            ndecs = prod(Stride);
            %expctdZ = zeros(nrows,ncols,nlays,ndecs,nSamples,datatype);
            expctdZ_ = zeros(ndecs,nrows,ncols,nlays,nSamples,datatype);
            E0 = testCase.getMatrixE0_(Stride);
            for iSample = 1:nSamples
                for i = size(X,3)
                    % Block DCT
                    Y = testCase.vol2col_(X(:,:,:,1,iSample),Stride,...
                    [nrows,ncols,nlays]);
                % Rearrange the DCT Coefs.
                A = E0*Y;
                expctdZ_(:,:,:,:,iSample) = ...
                    ...permute(reshape(A,ndecs,nrows,ncols,nlays),[2 3 4 1]);
                    reshape(A,ndecs,nrows,ncols,nlays);
                end
            end
            X_ = expctdZ_;
            %expctdZ = zeros(nChsTotal,nrows,ncols,nlays,nSamples,datatype);
            %expctdZ = zeros(nrows,ncols,nlays,nChsTotal,nSamples,datatype);

            % initial rotation
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            W0 = repmat(eye(ps,datatype),[1 1 nrows*ncols*nlays]);
            U0 = repmat(eye(pa,datatype),[1 1 nrows*ncols*nlays]);
            Y_ = zeros(nChsTotal,nrows,ncols,nlays,datatype);
            for iSample=1:nSamples
                % Perumation in each block
                Ai = X_(:,:,:,:,iSample); %permute(X(:,:,:,:,iSample),[4 1 2 3]);
                Yi = reshape(Ai,nDecs,nrows,ncols,nlays);
                %
                Ys = Yi(1:ps,:);
                Ya = Yi(ps+1:end,:);
                for iblk = 1:(nrows*ncols*nlays)
                    Ys(:,iblk) = W0(:,1:ps,iblk)*Ys(:,iblk);
                    Ya(:,iblk) = U0(:,1:pa,iblk)*Ya(:,iblk);
                end
                Y_(1:ps,:,:,:) = reshape(Ys,ps,nrows,ncols,nlays);
                Y_(ps+1:ps+pa,:,:,:) = reshape(Ya,pa,nrows,ncols,nlays);    
                eY(:,:,:,:,iSample) = Y_; %ipermute(Y,[4 1 2 3]);
            end

            for iCmp = 1:nComponents
                for iOrderH = 2:2:OverlappingFactor(2)-1
                    % Atom extension (Right shift)
                    eY = testCase.AtomExt('Right',eY,pa,ps);
                   
                    % Intermediate rotation
                    eY = testCase.intRotation(eY,mus,pa,ps,nrows,ncols,nlays,nSamples,nChsTotal,angles,datatype,device);
                    
                    % Atom extension (Left shift)
                    eY = testCase.AtomExt('Left',eY,pa,ps);
                    
                    % Intermediate rotation
                    eY = testCase.intRotation(eY,mus,pa,ps,nrows,ncols,nlays,nSamples,nChsTotal,angles,datatype,device);
                end
                
                for iOrderV = 2:2:OverlappingFactor(1)-1
                    % Atom extension (Down shift)
                    eY = testCase.AtomExt('Down',eY,pa,ps);
               
                    % Intermediate rotation
                    eY = testCase.intRotation(eY,mus,pa,ps,nrows,ncols,nlays,nSamples,nChsTotal,angles,datatype,device);
                    
                    % Atom extension (Up shift)
                    eY = testCase.AtomExt('Up',eY,pa,ps);
                  
                    % Intermediate rotation
                    eY = testCase.intRotation(eY,mus,pa,ps,nrows,ncols,nlays,nSamples,nChsTotal,angles,datatype,device);
                end

                for iOrderD = 2:2:OverlappingFactor(3)-1
                    % Atom extension (Back shift)
                    eY = testCase.AtomExt('Back',eY,pa,ps);
               
                    % Intermediate rotation
                    eY = testCase.intRotation(eY,mus,pa,ps,nrows,ncols,nlays,nSamples,nChsTotal,angles,datatype,device);
                    
                    % Atom extension (Front shift)
                    eY = testCase.AtomExt('Front',eY,pa,ps);
                  
                    % Intermediate rotation
                    eY = testCase.intRotation(eY,mus,pa,ps,nrows,ncols,nlays,nSamples,nChsTotal,angles,datatype,device);
                end
            
            % Channel separation
            expctdZac = permute(eY(2:end,:,:,:,:),[2 3 4 1 5]);
            expctdZdc = permute(eY(1,:,:,:,:),[2 3 4 1 5]);
            end
            
            %disp(expctdZdc)

            import tansacnet.lsun.*
            net = lsunAnalysis3dNetwork('InputSize',[height width depth], ...
                'Stride',Stride, ...
                'OverlappingFactor', OverlappingFactor, ...
                'DType',datatype, ...
                'Device',device);
            dlnet = net.dlnetwork();
            dlnet_ = initialize(dlnet);
            %analyzeNetwork(dlnet)
            
            X = dlarray(X, 'SSSCB');
            [Zac,Zdc] = forward(dlnet_, X);
            
            testCase.verifyInstanceOf((extractdata(Zac)),datatype);
            testCase.verifyThat(extractdata(Zdc),...
                IsEqualTo(expctdZdc,'Within',tolObj));
            testCase.verifyThat(extractdata(Zac),...
                IsEqualTo(expctdZac,'Within',tolObj));
        end

        function testBackword_st(testCase, Stride, datatype, device)
            import tansacnet.utility.Direction
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-5,single(1e-5));
            import tansacnet.utility.*

            nSamples = 8;
            height = 32;
            width = 32;
            depth = 32;
            nrows = height/Stride(Direction.VERTICAL);
            ncols = width/Stride(Direction.HORIZONTAL);
            nDecs = prod(Stride);
            numSteps = 5;
            prevLoss = inf;
            isLossDecreasing = false;
            vel=[];

            import tansacnet.lsun.*
            net = lsunAnalysis3dNetwork('InputSize',[height width depth], ...
                'Stride',Stride, ...
                'DType',datatype, ...
                'Device',device);
            dlnet = net.dlnetwork();
            dlnet_ = initialize(dlnet);

            dlX = dlarray(gpuArray(randn(32,32,32)),'SSCB');
            % 
             for step = 1:numSteps
                
                [gradients,loss] = dlfeval(@testCase.modelGradients,dlnet_,dlX);
                [dlnet_,vel] = sgdmupdate(dlnet_,gradients,vel);
                if loss < prevLoss
                    isLossDecreasing = true;
                end
                prevLoss = loss;
                isNonZero = false;
                for igrad = 1:size(gradients,1)
                    gradData = extractdata(gradients.Value{igrad});
                    if any(gradData(:) ~= 0)
                        isNonZero = true;
                        break;
                    end
                end
                testCase.verifyTrue(isNonZero, 'Gradient contains only zeros!');
             end
            testCase.verifyTrue(isLossDecreasing, 'Loss did not decrease over steps!');

        end
    end

    methods (Static, Access = private)

        function eY = AtomExt(dir,Yn,pa,ps)
            if strcmp(dir,'Right')
                shift = [ 0 0  1 0 0];
            elseif strcmp(dir,'Left')
                shift = [ 0 0 -1 0 0];
            elseif strcmp(dir,'Down')
                shift = [ 0  1 0 0 0];
            elseif strcmp(dir,'Up')
                shift = [ 0 -1 0 0 0];
            elseif strcmp(dir,'Back')
                shift = [ 0 0 0  1 0];
            elseif strcmp(dir,'Front')
                shift = [ 0 0 0 -1 0];
            else
                shift = [ 0 0 0 0 0];
            end
            eYs = Yn(1:ps,:,:,:,:);
            eYa = Yn(ps+1:ps+pa,:,:,:,:);
            eY =  [ eYs+eYa ; eYs-eYa ]/sqrt(2);
            % Block circular shift
            eY(ps+1:ps+pa,:,:,:,:) = circshift(eY(ps+1:ps+pa,:,:,:,:),shift);
            % Block butterfly
            eYs = eY(1:ps,:,:,:,:);
            eYa = eY(ps+1:ps+pa,:,:,:,:);
            eY =  [ eYs+eYa ; eYs-eYa ]/sqrt(2);
        end

        function irY = intRotation(eY,mus,pa,ps,nrows,ncols,nlays,nSamples,nChsTotal,angles,datatype,device)
            import tansacnet.utility.*
            genU = OrthonormalMatrixGenerationSystem('DType',datatype,'Device',device);
            %angles = zeros((nChsTotal-2)*nChsTotal/8,nrows*ncols*nlays,datatype);
            if isempty(angles)
                angles = zeros((nChsTotal-2)*nChsTotal/8,nrows*ncols*nlays,datatype);
            elseif isscalar(angles)
                angles = angles*ones((nChsTotal-2)*nChsTotal/8,nrows*ncols*nlays,'like',angles); 
            end
            if isempty(mus)
                mus = ones(pa,nrows*ncols*nlays,datatype);   
            elseif isscalar(mus)
                mus = mus*ones(pa,nrows*ncols*nlays,datatype);  
            end
            if device == "cuda"
                angles = gpuArray(angles);
                mus = gpuArray(mus);
            end
            Un = genU.step(angles,mus);
            irY = eY;

            %UnT = repmat(mus*eye(pa,datatype),[1 1 nrows*ncols*nlays]);
            iYa = reshape(irY(ps+1:ps+pa,:,:,:,:),pa,nrows*ncols*nlays,nSamples);
            Za = zeros(size(iYa),'like',iYa);
            for iSample=1:nSamples
                for iblk = 1:(nrows*ncols*nlays)
                    Za(:,iblk,iSample) = Un(:,:,iblk)*iYa(:,iblk,iSample);
                end
            end
            irY(ps+1:ps+pa,:,:,:,:) = reshape(Za,pa,nrows,ncols,nlays,nSamples);
        end

        function x = col2vol_(y,decFactor,nBlocks)
            import tansacnet.utility.Direction
            decY = decFactor(Direction.VERTICAL);
            decX = decFactor(Direction.HORIZONTAL);
            decZ = decFactor(Direction.DEPTH);
            nRows_ = nBlocks(Direction.VERTICAL);
            nCols_ = nBlocks(Direction.HORIZONTAL);
            nLays_ = nBlocks(Direction.DEPTH);
            
            idx = 0;
            x = zeros(decY*nRows_,decX*nCols_,decZ*nLays_);
            for iLay = 1:nLays_
                idxZ = iLay*decZ;
                for iCol = 1:nCols_
                    idxX = iCol*decX;
                    for iRow = 1:nRows_
                        idxY = iRow*decY;
                        idx = idx + 1;
                        blockData = y(:,idx);
                        x(idxY-decY+1:idxY,...
                            idxX-decX+1:idxX,...
                            idxZ-decZ+1:idxZ) = ...
                            reshape(blockData,decY,decX,decZ);
                    end
                end
            end
            
        end
            
        function y = vol2col_(x,decFactor,nBlocks)
            import tansacnet.utility.Direction
            decY = decFactor(Direction.VERTICAL);
            decX = decFactor(Direction.HORIZONTAL);
            decZ = decFactor(Direction.DEPTH);
            nRows_ = nBlocks(Direction.VERTICAL);
            nCols_ = nBlocks(Direction.HORIZONTAL);
            nLays_ = nBlocks(Direction.DEPTH);
            
            idx = 0;
            y = zeros(decY*decX*decZ,nRows_*nCols_*nLays_);
            for iLay = 1:nLays_
                idxZ = iLay*decZ;
                for iCol = 1:nCols_
                    idxX = iCol*decX;
                    for iRow = 1:nRows_
                        idxY = iRow*decY;
                        idx = idx + 1;
                        blockData = x(...
                            idxY-decY+1:idxY,...
                            idxX-decX+1:idxX,...
                            idxZ-decZ+1:idxZ);
                        y(:,idx) = blockData(:);
                    end
                end
            end
            
        end
        

        function value = getMatrixE0_(decFactor)
            import tansacnet.utility.Direction
            decY = decFactor(Direction.VERTICAL);
            decX = decFactor(Direction.HORIZONTAL);
            decZ = decFactor(Direction.DEPTH);

            % Generate DCT matrices
            Cv_ = dctmtx(decY);
            Ch_ = dctmtx(decX);
            Cd_ = dctmtx(decZ);

            % Reorder rows using a single matrix operation
            reorder = @(C) C([1:2:end, 2:2:end], :);
            Cv_ = reorder(Cv_);
            Ch_ = reorder(Ch_);
            Cd_ = reorder(Cd_);

            % Split matrices into even and odd parts
            split = @(C, n) deal(C(1:ceil(n/2), :), C(ceil(n/2)+1:end, :));
            [Cve, Cvo] = split(Cv_, decY);
            [Che, Cho] = split(Ch_, decX);
            [Cde, Cdo] = split(Cd_, decZ);

            % Compute Kronecker products
            kron3 = @(A, B, C) kron(kron(A, B), C);
            value = [
                kron3(Cde, Che, Cve);
                kron3(Cdo, Cho, Cve);
                kron3(Cde, Cho, Cvo);
                kron3(Cdo, Che, Cvo);
                kron3(Cdo, Che, Cve);
                kron3(Cde, Cho, Cve);
                kron3(Cdo, Cho, Cvo);
                kron3(Cde, Che, Cvo)
            ];
        end
    end

end