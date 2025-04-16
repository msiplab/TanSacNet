classdef lsunSynthesis3dNetworkTestCase < matlab.unittest.TestCase
    %LSUNSYNTHESIS3dNETWORKTESTCASE 
    %
    % Requirements: MATLAB R2024a
    %
    % Copyright (c) 2025, Shogo MURAMATSU
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
            numberOfComponents = 1;

            % Expected values
            expctdTensorSize = [inputSize numberOfComponents];
            expctdStride = [2 2 2];
            expctdOverlappingFactor = [1 1 1];
            expctdNumberOfLevels = 1;
            expctdNoDcLeakage = false;
            expctdPrefix = '';
            expctdDevice = 'cuda';
            expctdDType = 'double';

            % Instantiation of target class
            import tansacnet.lsun.*
            net = lsunSynthesis3dNetwork();

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
            net = lsunSynthesis3dNetwork('InputSize',inputSize, ...
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

        function testNetwork(testCase,Stride,device,datatype)
                
            height = 16;
            width = 16;
            depth = 16;
            nSamples = 8;
            nComponents = 1;

            import tansacnet.lsun.*
            AnaNet = lsunAnalysis3dNetwork('InputSize',[height width depth], ...
                'Stride',Stride, ...
                'DType',datatype, ...
                'Device',device);
            Anadlnet = AnaNet.dlnetwork();
            %analyzeNetwork(dlnet)
            Anadlnet_ = initialize(Anadlnet);
            
            X = rand([height, width, depth, nComponents, nSamples], datatype);
            X = dlarray(X, 'SSSCB');
            [Zac,Zdc] = forward(Anadlnet_, X);
    
            SynNet = lsunSynthesis3dNetwork('InputSize',[height width depth], ...
                'Stride',Stride, ...
                'DType',datatype, ...
                'Device',device);
            Syndlnet = SynNet.dlnetwork();
            %analyzeNetwork(Syndlnet)
            Syndlnet_ = initialize(Syndlnet);

            Z = forward(Syndlnet_, Zac, Zdc);

            testCase.verifyInstanceOf(extractdata(Z),datatype);
        end

        function testNetwork_with_Overlap(testCase,inputSize,Stride,OverlappingFactor, datatype, device)

            height = inputSize(1);
            width = inputSize(2);
            depth = inputSize(3);
            nSamples = 8;
            nComponents = 1;
            import tansacnet.lsun.*

            AnaNet = lsunAnalysis3dNetwork('InputSize',[height width depth], ...
                'Stride',Stride, ...
                'OverlappingFactor', OverlappingFactor, ...
                'DType',datatype, ...
                'Device',device);
            Anadlnet = AnaNet.dlnetwork();
            %analyzeNetwork(Anadlnet)
            Anadlnet_ = initialize(Anadlnet);
            
            X = rand([height, width, depth, nComponents, nSamples], datatype);
            X = dlarray(X, 'SSSCB');
            [Zac,Zdc] = forward(Anadlnet_, X);

            SynNet = lsunSynthesis3dNetwork('InputSize',[height width depth], ...
                'Stride',Stride, ...
                'OverlappingFactor', OverlappingFactor, ...
                'DType',datatype, ...
                'Device',device);
            Syndlnet = SynNet.dlnetwork();
            %analyzeNetwork(Syndlnet)
            Syndlnet_ = initialize(Syndlnet);

            Z = forward(Syndlnet_, Zac, Zdc);

            testCase.verifyInstanceOf(extractdata(Z),datatype);
            %testCase.verifyInstanceOf(extractdata(Zdc),datatype);
        end

        function testForward(testCase, Stride, device, datatype)

            import tansacnet.utility.Direction
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-5,single(1e-5));
            
            height = 16;
            width = 16;
            depth = 16;
            nSamples = 8;
            nComponents = 1;
            nDecs = prod(Stride);
            nChsTotal = nDecs;
            nrows = height/Stride(Direction.VERTICAL);
            ncols = width/Stride(Direction.HORIZONTAL);
            nlays = depth/Stride(Direction.DEPTH);

            import tansacnet.lsun.*
            AnaNet = lsunAnalysis3dNetwork('InputSize',[height width depth], ...
                'Stride',Stride, ...
                'DType',datatype, ...
                'Device',device);
            Anadlnet = AnaNet.dlnetwork();
            %analyzeNetwork(Anadlnet)
            Anadlnet_ = initialize(Anadlnet);
            
            X = rand([height, width, depth, nComponents, nSamples], datatype);
            X = dlarray(X, 'SSSCB');
            [Xac,Xdc] = forward(Anadlnet_, X);
        
            if nComponents > 1
                numOutputs = nComponents;
                nChsPerCmp = size(Xac,4)/numOutputs;
                Xac_ = cell(numOutputs,1);
                for idx = 1:numOutputs
                    Xac_{idx} = Xac(:,:,:,(idx-1)*nChsPerCmp+1:idx*nChsPerCmp,:);
                end
                nChsPerCmp = size(Xdc,4)/numOutputs;
                Xdc_ = cell(numOutputs,1);
                for idx = 1:numOutputs
                    Xdc_{idx} = Xdc(:,:,:,(idx-1)*nChsPerCmp+1:idx*nChsPerCmp,:);
                end
                Xac = Xac_;
                Xdc = Xdc_;
            end
        
            Z = permute(cat(4, extractdata(Xdc), extractdata(Xac)),[4 1 2 3 5]);
            %Z = permute(cat(4,Xdc,Xac),[4 1 2 3 5]);

            % Expected values        
            % nDecs x nRows x nCols x nLays x nSamples
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            W0T = repmat(eye(ps,datatype),[1 1 nrows*ncols*nlays]);
            U0T = repmat(eye(pa,datatype),[1 1 nrows*ncols*nlays]);
            Y = Z; %permute(X,[4 1 2 3 5]);
            Ys = reshape(Y(1:ps,:,:,:,:),ps,nrows*ncols*nlays,nSamples);
            Ya = reshape(Y(ps+1:ps+pa,:,:,:,:),pa,nrows*ncols*nlays,nSamples);
            for iSample=1:nSamples
                for iblk = 1:(nrows*ncols*nlays)
                    Ys(:,iblk,iSample) = W0T(1:ps,:,iblk)*Ys(:,iblk,iSample); 
                    Ya(:,iblk,iSample) = U0T(1:pa,:,iblk)*Ya(:,iblk,iSample);
                end
            end
            Zsa = cat(1,Ys,Ya);
            %expctdZ = ipermute(reshape(Zsa,nDecs,nrows,ncols,nlays,nSamples),...
            %    [4 1 2 3 5]);
            expctdZd = reshape(Zsa,nDecs,nrows,ncols,nlays,nSamples);
            E0_T = transpose(testCase.getMatrixE0_(Stride));
            expctdZ = zeros(height,width,depth,datatype);
            for iSample = 1:nSamples
                %A = reshape(permute(X(:,:,:,:,iSample),[4 1 2 3]),...
                %    nDecs*nrows,ncols,nlays);
                A = reshape(expctdZd(:,:,:,:,iSample),nDecs,nrows*ncols*nlays);
                Y = E0_T*A;
                expctdZ(:,:,:,1,iSample) = testCase.col2vol_(Y,Stride,...
                    [nrows,ncols,nlays]);
            end

            SynNet = lsunSynthesis3dNetwork('InputSize',[height width depth], ...
                'Stride',Stride, ...
                'DType',datatype, ...
                'Device',device);
            Syndlnet = SynNet.dlnetwork();
            %analyzeNetwork(Syndlnet)
            Syndlnet_ = initialize(Syndlnet);

            actualZ = forward(Syndlnet_, Xac, Xdc);

            testCase.verifyInstanceOf(extractdata(actualZ),datatype);
            testCase.verifyThat(extractdata(actualZ),...
                IsEqualTo(expctdZ,'Within',tolObj));
        end

        function testForward_with_overlap(testCase,Stride,OverlappingFactor,datatype,device)

            import tansacnet.utility.Direction
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-5,single(1e-5));
    
            mus = 1;
            height = 16;
            width = 16;
            depth = 16;
            nSamples = 8;
            nComponents = 1;
            ovlpFactor = OverlappingFactor;
            nDecs = prod(Stride);
            nChsTotal = nDecs;
            nrows = height/Stride(Direction.VERTICAL);
            ncols = width/Stride(Direction.HORIZONTAL);
            nlays = depth/Stride(Direction.DEPTH);

            import tansacnet.lsun.*
            AnaNet = lsunAnalysis3dNetwork('InputSize',[height width depth], ...
                'Stride',Stride, ...
                'OverlappingFactor', ovlpFactor, ...
                'DType',datatype, ...
                'Device',device);
            Anadlnet = AnaNet.dlnetwork();
            %analyzeNetwork(Anadlnet)
            Anadlnet_ = initialize(Anadlnet);
            
            X = rand([height, width, depth, nComponents, nSamples], datatype);
            X = dlarray(X, 'SSSCB');
            [Xac,Xdc] = forward(Anadlnet_, X);
            Xac_ = Xac;
            Xdc_ = Xdc;
            % Component separation
            if nComponents > 1
                numOutputs = nComponents;
                nChsPerCmp = size(Xac,4)/numOutputs;
                Xac_ = cell(numOutputs,1);
                for idx = 1:numOutputs
                    Xac_{idx} = Xac(:,:,:,(idx-1)*nChsPerCmp+1:idx*nChsPerCmp,:);
                end
                nChsPerCmp = size(Xdc,4)/numOutputs;
                Xdc_ = cell(numOutputs,1);
                for idx = 1:numOutputs
                    Xdc_{idx} = Xdc(:,:,:,(idx-1)*nChsPerCmp+1:idx*nChsPerCmp,:);
                end 
            end
        
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            angles = [];    
            %Intermediate rotation and Shift
            for iCmp = 1:nComponents
                % Channel concatanation
                eY = permute(cat(4, extractdata(Xdc_), extractdata(Xac_)),[4 1 2 3 5]);
                %Z = permute(cat(4,Xdc,Xac),[4 1 2 3 5]);

                % Atom extension in vertical
                for iOrderV = 2:2:ovlpFactor(1)-1
                    % Intermediate rotation
                    eY = testCase.intRotation(eY,mus,pa,ps,nrows,ncols,nlays,nSamples,nChsTotal,angles,datatype,device);
                    % Atom extension (Down shift)
                    eY = testCase.AtomExt('Down',eY,pa,ps);
                    % Intermediate rotation
                    eY = testCase.intRotation(eY,mus,pa,ps,nrows,ncols,nlays,nSamples,nChsTotal,angles,datatype,device);
                    % Atom extension (Up shift)
                    eY = testCase.AtomExt('Up',eY,pa,ps);
                end
                
                % Atom extension in horizontal
                for iOrderH = 2:2:ovlpFactor(2)-1
                    % Intermediate rotation
                    eY = testCase.intRotation(eY,mus,pa,ps,nrows,ncols,nlays,nSamples,nChsTotal,angles,datatype,device);
                    % Atom extension (Right shift)
                    eY = testCase.AtomExt('Right',eY,pa,ps);
                    % Intermediate rotation
                    eY = testCase.intRotation(eY,mus,pa,ps,nrows,ncols,nlays,nSamples,nChsTotal,angles,datatype,device);
                    % Atom extension (Left shift)
                    eY = testCase.AtomExt('Left',eY,pa,ps);
                end

                % Atom extension in depth
                for iOrderD = 2:2:ovlpFactor(2)-1
                    % Intermediate rotation
                    eY = testCase.intRotation(eY,mus,pa,ps,nrows,ncols,nlays,nSamples,nChsTotal,angles,datatype,device);
                    % Atom extension (Right shift)
                    eY = testCase.AtomExt('Back',eY,pa,ps);
                    % Intermediate rotation
                    eY = testCase.intRotation(eY,mus,pa,ps,nrows,ncols,nlays,nSamples,nChsTotal,angles,datatype,device);
                    % Atom extension (Left shift)
                    eY = testCase.AtomExt('Front',eY,pa,ps);
                end
            end

            %Final rotation 
            W0T = repmat(eye(ps,datatype),[1 1 nrows*ncols*nlays]);
            U0T = repmat(eye(pa,datatype),[1 1 nrows*ncols*nlays]);
            Y = eY; %permute(X,[4 1 2 3 5]);
            Ys = reshape(Y(1:ps,:,:,:,:),ps,nrows*ncols*nlays,nSamples);
            Ya = reshape(Y(ps+1:ps+pa,:,:,:,:),pa,nrows*ncols*nlays,nSamples);
            for iSample=1:nSamples
                for iblk = 1:(nrows*ncols*nlays)
                    Ys(:,iblk,iSample) = W0T(1:ps,:,iblk)*Ys(:,iblk,iSample); 
                    Ya(:,iblk,iSample) = U0T(1:pa,:,iblk)*Ya(:,iblk,iSample);
                end
            end
            Zsa = cat(1,Ys,Ya);
            %expctdZ = ipermute(reshape(Zsa,nDecs,nrows,ncols,nlays,nSamples),...
            %    [4 1 2 3 5]);
            expctdZd = reshape(Zsa,nDecs,nrows,ncols,nlays,nSamples);
            E0_T = transpose(testCase.getMatrixE0_(Stride));
            expctdZ = zeros(height,width,depth,datatype);
            for iSample = 1:nSamples
                %A = reshape(permute(X(:,:,:,:,iSample),[4 1 2 3]),...
                %    nDecs*nrows,ncols,nlays);
                A = reshape(expctdZd(:,:,:,:,iSample),nDecs,nrows*ncols*nlays);
                Y = E0_T*A;
                expctdZ(:,:,:,1,iSample) = testCase.col2vol_(Y,Stride,...
                    [nrows,ncols,nlays]);
            end

            SynNet = lsunSynthesis3dNetwork('InputSize',[height width depth], ...
                'Stride',Stride, ...
                'OverlappingFactor', ovlpFactor, ...
                'DType',datatype, ...
                'Device',device);
            Syndlnet = SynNet.dlnetwork();
            %analyzeNetwork(Syndlnet)
            Syndlnet_ = initialize(Syndlnet);

            actualZ = forward(Syndlnet_, Xac, Xdc);

            testCase.verifyInstanceOf(extractdata(actualZ),datatype);
            testCase.verifyThat(extractdata(actualZ),...
                IsEqualTo(expctdZ,'Within',tolObj));
        end

    end
    
    methods (Static, Access = private)
        function eY = AtomExt(dir,Yn,pa,ps)
            if strcmp(dir,'Right')
                shift = [ 0 0  1 0 0 ];
            elseif strcmp(dir,'Left')
                shift = [ 0 0 -1 0 0 ];
            elseif strcmp(dir,'Down')
                shift = [ 0  1 0 0 0 ];
            elseif strcmp(dir,'Up')
                shift = [ 0 -1 0 0 0 ];
            elseif strcmp(dir,'Back')
                shift = [ 0  0 0 1 0 ];
            elseif strcmp(dir,'Front')
                shift = [ 0 0 0 -1 0 ];
            else
                shift = [ 0 0 0 0 0 ];
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
        % function Y = intRotation(eY,mus,pa,ps,nrows,ncols,nSamples,datatype)
        %     UnT = repmat(mus*eye(pa,datatype),[1 1 nrows*ncols]);
        %     UnT_ = permute(UnT,[2 1 3]);
        %     iYa = reshape(eY(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
        %     Za = zeros(size(iYa),'like',iYa);
        %     for iSample=1:nSamples
        %         for iblk = 1:(nrows*ncols)
        %             Za(:,iblk,iSample) = UnT_(:,:,iblk)*iYa(:,iblk,iSample);
        %         end
        %     end
        %     Y(ps+1:ps+pa,:,:,:) = reshape(Za,pa,nrows,ncols,nSamples);
        % end
        function irY = intRotation(eY,mus,pa,ps,nrows,ncols,nlays,nSamples,nChsTotal,angles,datatype,device)
            import tansacnet.utility.*
            genU = OrthonormalMatrixGenerationSystem('DType',datatype,'Device',device);
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

            %angles = randn((nChsTotal-2)*nChsTotal/8,nrows*ncols);
            %Un = genU.step(angles,mus);
            UnT = permute(genU.step(angles,mus),[2 1 3]);
            irY = eY;

            %UnT = repmat(mus*eye(pa,datatype),[1 1 nrows*ncols]);
            iYa = reshape(eY(ps+1:ps+pa,:,:,:,:),pa,nrows*ncols*nlays,nSamples);
            Za = zeros(size(iYa),'like',iYa);
            for iSample=1:nSamples
                for iblk = 1:(nrows*ncols*nlays)
                    Za(:,iblk,iSample) = UnT(:,:,iblk)*iYa(:,iblk,iSample);
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