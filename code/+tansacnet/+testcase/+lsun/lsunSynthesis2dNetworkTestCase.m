classdef lsunSynthesis2dNetworkTestCase < matlab.unittest.TestCase
    %LSUNSYNTHESIS2dNETWORKTESTCASE 
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
            net = lsunSynthesis2dNetwork();

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
            net = lsunSynthesis2dNetwork('InputSize',inputSize, ...
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

        function testNetwork(testCase,Stride,device,datatype)
            
            height = 16;
            width = 16;
            nSamples = 8;
            nComponents = 1;

            import tansacnet.lsun.*
            AnaNet = lsunAnalysis2dNetwork('InputSize',[height width], ...
                'Stride',Stride, ...
                'DType',datatype, ...
                'Device',device);
            Anadlnet = AnaNet.dlnetwork();
            %analyzeNetwork(dlnet)
            Anadlnet_ = initialize(Anadlnet);
            
            X = rand([height, width, nComponents, nSamples], datatype);
            X = dlarray(X, 'SSCB');
            [Zac,Zdc] = forward(Anadlnet_, X);
       
            SynNet = lsunSynthesis2dNetwork('InputSize',[height width], ...
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
            nSamples = 8;
            nComponents = 1;
            import tansacnet.lsun.*

            AnaNet = lsunAnalysis2dNetwork('InputSize',[height width], ...
                'Stride',Stride, ...
                'OverlappingFactor', OverlappingFactor, ...
                'DType',datatype, ...
                'Device',device);
            Anadlnet = AnaNet.dlnetwork();
            %analyzeNetwork(Anadlnet)
            Anadlnet_ = initialize(Anadlnet);
            
            X = rand([height, width, nComponents, nSamples], datatype);
            X = dlarray(X, 'SSCB');
            [Zac,Zdc] = forward(Anadlnet_, X);

            SynNet = lsunSynthesis2dNetwork('InputSize',[height width], ...
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
            nSamples = 8;
            nComponents = 1;
            nDecs = prod(Stride);
            nChsTotal = nDecs;
            nrows = height/Stride(Direction.VERTICAL);
            ncols = width/Stride(Direction.HORIZONTAL);

            import tansacnet.lsun.*
            AnaNet = lsunAnalysis2dNetwork('InputSize',[height width], ...
                'Stride',Stride, ...
                'DType',datatype, ...
                'Device',device);
            Anadlnet = AnaNet.dlnetwork();
            %analyzeNetwork(Anadlnet)
            Anadlnet_ = initialize(Anadlnet);
            
            X = rand([height, width, nComponents, nSamples], datatype);
            X = dlarray(X, 'SSCB');
            [Xac,Xdc] = forward(Anadlnet_, X);
           
            if nComponents > 1
                numOutputs = nComponents;
                nChsPerCmp = size(Xac,3)/numOutputs;
                Xac_ = cell(numOutputs,1);
                for idx = 1:numOutputs
                    Xac_{idx} = Xac(:,:,(idx-1)*nChsPerCmp+1:idx*nChsPerCmp,:);
                end
                nChsPerCmp = size(Xdc,3)/numOutputs;
                Xdc_ = cell(numOutputs,1);
                for idx = 1:numOutputs
                    Xdc_{idx} = Xdc(:,:,(idx-1)*nChsPerCmp+1:idx*nChsPerCmp,:);
                end
                Xac = Xac_;
                Xdc = Xdc_;
            end
           
            Z = permute(cat(3, extractdata(Xdc), extractdata(Xac)),[3 1 2 4]);
            %Z = permute(cat(3,Xdc,Xac),[3 1 2 4]);

            % Expected values        
            % nDecs x nRows x nCols x nSamples
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            W0T = repmat(eye(ps,datatype),[1 1 nrows*ncols]);
            U0T = repmat(eye(pa,datatype),[1 1 nrows*ncols]);
            Y = Z; %permute(X,[3 1 2 4]);
            Ys = reshape(Y(1:ps,:,:,:),ps,nrows*ncols,nSamples);
            Ya = reshape(Y(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            for iSample=1:nSamples
                for iblk = 1:(nrows*ncols)
                    Ys(:,iblk,iSample) = W0T(1:ps,:,iblk)*Ys(:,iblk,iSample); 
                    Ya(:,iblk,iSample) = U0T(1:pa,:,iblk)*Ya(:,iblk,iSample);
                end
            end
            Zsa = cat(1,Ys,Ya);
            %expctdZ = ipermute(reshape(Zsa,nDecs,nrows,ncols,nSamples),...
            %    [3 1 2 4]);
            expctdZd = reshape(Zsa,nDecs,nrows,ncols,nSamples);
            for iSample = 1:nSamples
                %A = reshape(permute(X(:,:,:,iSample),[3 1 2]),...
                %    nDecs*nrows,ncols);
                A = reshape(expctdZd(:,:,:,iSample),nDecs*nrows,ncols);
                dY = blockproc(A,[nDecs 1],...
                    @(x) testCase.permuteIdctCoefs_(x.data,Stride));
                expctdZ(:,:,nComponents,iSample) = ...
                    cast(blockproc(dY,...
                    Stride,...
                    @(x) idct2(x.data)),datatype);
            end

            SynNet = lsunSynthesis2dNetwork('InputSize',[height width], ...
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
            nSamples = 8;
            nComponents = 1;
            ovlpFactor = OverlappingFactor;
            nDecs = prod(Stride);
            nChsTotal = nDecs;
            nrows = height/Stride(Direction.VERTICAL);
            ncols = width/Stride(Direction.HORIZONTAL);

            import tansacnet.lsun.*
            AnaNet = lsunAnalysis2dNetwork('InputSize',[height width], ...
                'Stride',Stride, ...
                'OverlappingFactor', ovlpFactor, ...
                'DType',datatype, ...
                'Device',device);
            Anadlnet = AnaNet.dlnetwork();
            %analyzeNetwork(Anadlnet)
            Anadlnet_ = initialize(Anadlnet);
            
            X = rand([height, width, nComponents, nSamples], datatype);
            X = dlarray(X, 'SSCB');
            [Xac,Xdc] = forward(Anadlnet_, X);
            Xac_ = Xac;
            Xdc_ = Xdc;
            % Component separation
            if nComponents > 1
                numOutputs = nComponents;
                nChsPerCmp = size(Xac,3)/numOutputs;
                Xac_ = cell(numOutputs,1);
                for idx = 1:numOutputs
                    Xac_{idx} = Xac(:,:,(idx-1)*nChsPerCmp+1:idx*nChsPerCmp,:);
                end
                nChsPerCmp = size(Xdc,3)/numOutputs;
                Xdc_ = cell(numOutputs,1);
                for idx = 1:numOutputs
                    Xdc_{idx} = Xdc(:,:,(idx-1)*nChsPerCmp+1:idx*nChsPerCmp,:);
                end 
            end
           
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            angles = [];    
            %Intermediate rotation and Shift
            for iCmp = 1:nComponents
                % Channel concatanation
                eY = permute(cat(3, extractdata(Xdc_), extractdata(Xac_)),[3 1 2 4]);
                %Z = permute(cat(3,Xdc,Xac),[3 1 2 4]);

                for iOrderV = 2:2:ovlpFactor(1)-1
                    % Intermediate rotation
                    eY = testCase.intRotation(eY,mus,pa,ps,nrows,ncols,nSamples,nChsTotal,angles,datatype,device);
                    % Atom extension (Down shift)
                    eY = testCase.AtomExt('Down',eY,pa,ps);
                    % Intermediate rotation
                    eY = testCase.intRotation(eY,mus,pa,ps,nrows,ncols,nSamples,nChsTotal,angles,datatype,device);
                    % Atom extension (Up shift)
                    eY = testCase.AtomExt('Up',eY,pa,ps);
                end
                
                % Atom extension in horizontal
                for iOrderH = 2:2:ovlpFactor(2)-1
                    % Intermediate rotation
                    eY = testCase.intRotation(eY,mus,pa,ps,nrows,ncols,nSamples,nChsTotal,angles,datatype,device);
                    % Atom extension (Right shift)
                    eY = testCase.AtomExt('Right',eY,pa,ps);
                    % Intermediate rotation
                    eY = testCase.intRotation(eY,mus,pa,ps,nrows,ncols,nSamples,nChsTotal,angles,datatype,device);
                    % Atom extension (Left shift)
                    eY = testCase.AtomExt('Left',eY,pa,ps);
                end
            end

            %Final rotation 
            W0T = repmat(eye(ps,datatype),[1 1 nrows*ncols]);
            U0T = repmat(eye(pa,datatype),[1 1 nrows*ncols]);
            Y = eY; %permute(X,[3 1 2 4]);
            Ys = reshape(Y(1:ps,:,:,:),ps,nrows*ncols,nSamples);
            Ya = reshape(Y(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            for iSample=1:nSamples
                for iblk = 1:(nrows*ncols)
                    Ys(:,iblk,iSample) = W0T(1:ps,:,iblk)*Ys(:,iblk,iSample); 
                    Ya(:,iblk,iSample) = U0T(1:pa,:,iblk)*Ya(:,iblk,iSample);
                end
            end
            Zsa = cat(1,Ys,Ya);
            %expctdZ = ipermute(reshape(Zsa,nDecs,nrows,ncols,nSamples),...
            %    [3 1 2 4]);
            expctdZd = reshape(Zsa,nDecs,nrows,ncols,nSamples);
            for iSample = 1:nSamples
                %A = reshape(permute(X(:,:,:,iSample),[3 1 2]),...
                %    nDecs*nrows,ncols);
                A = reshape(expctdZd(:,:,:,iSample),nDecs*nrows,ncols);
                dY = blockproc(A,[nDecs 1],...
                    @(x) testCase.permuteIdctCoefs_(x.data,Stride));
                expctdZ(:,:,nComponents,iSample) = ...
                    cast(blockproc(dY,...
                    Stride,...
                    @(x) idct2(x.data)),datatype);
            end

            SynNet = lsunSynthesis2dNetwork('InputSize',[height width], ...
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
        function value = permuteDctCoefs_(x)
            coefs = x.data;
            cee = coefs(1:2:end,1:2:end);
            coo = coefs(2:2:end,2:2:end);
            coe = coefs(2:2:end,1:2:end);
            ceo = coefs(1:2:end,2:2:end);
            value = [ cee(:) ; coo(:) ; coe(:) ; ceo(:) ];
        end
        function value = permuteIdctCoefs_(x,blockSize)
            import tansacnet.utility.Direction
            coefs = x;
            decY_ = blockSize(Direction.VERTICAL);
            decX_ = blockSize(Direction.HORIZONTAL);
            nQDecsee = ceil(decY_/2)*ceil(decX_/2);
            nQDecsoo = floor(decY_/2)*floor(decX_/2);
            nQDecsoe = floor(decY_/2)*ceil(decX_/2);
            cee = coefs(         1:  nQDecsee);
            coo = coefs(nQDecsee+1:nQDecsee+nQDecsoo);
            coe = coefs(nQDecsee+nQDecsoo+1:nQDecsee+nQDecsoo+nQDecsoe);
            ceo = coefs(nQDecsee+nQDecsoo+nQDecsoe+1:end);
            value = zeros(decY_,decX_,'like',x);
            value(1:2:decY_,1:2:decX_) = reshape(cee,ceil(decY_/2),ceil(decX_/2));
            value(2:2:decY_,2:2:decX_) = reshape(coo,floor(decY_/2),floor(decX_/2));
            value(2:2:decY_,1:2:decX_) = reshape(coe,floor(decY_/2),ceil(decX_/2));
            value(1:2:decY_,2:2:decX_) = reshape(ceo,ceil(decY_/2),floor(decX_/2));
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
        function irY = intRotation(eY,mus,pa,ps,nrows,ncols,nSamples,nChsTotal,angles,datatype,device)
            import tansacnet.utility.*
            genU = OrthonormalMatrixGenerationSystem('DType',datatype,'Device',device);
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

            %angles = randn((nChsTotal-2)*nChsTotal/8,nrows*ncols);
            %Un = genU.step(angles,mus);
            UnT = permute(genU.step(angles,mus),[2 1 3]);
            irY = eY;

            %UnT = repmat(mus*eye(pa,datatype),[1 1 nrows*ncols]);
            iYa = reshape(eY(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            Za = zeros(size(iYa),'like',iYa);
            for iSample=1:nSamples
                for iblk = 1:(nrows*ncols)
                    Za(:,iblk,iSample) = UnT(:,:,iblk)*iYa(:,iblk,iSample);
                end
            end
            irY(ps+1:ps+pa,:,:,:) = reshape(Za,pa,nrows,ncols,nSamples);
        end
        
    end

end