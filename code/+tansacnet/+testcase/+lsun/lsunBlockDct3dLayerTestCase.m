classdef lsunBlockDct3dLayerTestCase < matlab.unittest.TestCase
    %NSOLTBLOCKDCT3DLAYERTESTCASE
    %
    %   ベクトル配列をブロック配列を入力:
    %      (Stride(1)xnRows) x (Stride(2)xnCols) x nComponents x nSamples
    %
    %   コンポーネント別に出力:
    %      nDecs x nRows x nCols x nLays x nSamples
    %
    %
    % Requirements: MATLAB R2020b
    %
    % Copyright (c) 2020-2022, Eisuke KOBAYASHI, Shogo MURAMATSU
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
        stride = { [1 1 1], [2 2 2], [1 2 4], [4 4 4] };
        datatype = { 'single', 'double' };
        height = struct('small', 8,'medium', 16, 'large', 32);
        width = struct('small', 8,'medium', 16, 'large', 32);
        depth = struct('small', 8,'medium', 16, 'large', 32);
    end
    
    methods (TestClassTeardown)
        function finalCheck(~)
            import tansacnet.lsun.*
            fprintf("\n --- Check layer for 3-D images ---\n");
            layer = lsunBlockDct3dLayer(...
                'Stride',[2 2 2]);
            checkLayer(layer,[8 8 8 1],'ObservationDimension',5,...
                'CheckCodegenCompatibility',false)
        end
    end
    
    methods (Test)
        
        function testConstructor(testCase, stride)
            
            % Expected values
            expctdName = 'E0';
            expctdDescription = "Block DCT of size " ...
                + stride(1) + "x" + stride(2) + "x" + stride(3);
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunBlockDct3dLayer(...
                'Stride',stride,...
                'Name',expctdName);
            
            % Actual values
            actualName = layer.Name;
            actualDescription = layer.Description;
            
            % Evaluation
            testCase.verifyEqual(actualName,expctdName);
            testCase.verifyEqual(actualDescription,expctdDescription);
        end
        
        
        function testPredict(testCase, ...
                stride, height, width, depth, datatype)
            import tansacnet.utility.Direction
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-5,single(1e-5));
            
            % Parameters
            nSamples = 8;
            nComponents = 1;
            X = rand(height,width,depth, nComponents,nSamples, datatype);
            
            % Expected values
            nrows = ceil(height/stride(Direction.VERTICAL));
            ncols = ceil(width/stride(Direction.HORIZONTAL));
            nlays = ceil(depth/stride(Direction.DEPTH));
            ndecs = prod(stride);
            %expctdZ = zeros(nrows,ncols,ndecs,nSamples,datatype);
            expctdZ = zeros(ndecs,nrows,ncols,nlays,nSamples,datatype);
            E0 = testCase.getMatrixE0_(stride);
            for iSample = 1:nSamples
                % Block DCT
                 Y = testCase.vol2col_(X(:,:,:,1,iSample),stride,...
                    [nrows,ncols,nlays]);
                % Rearrange the DCT Coefs.
                A = E0*Y;
                expctdZ(:,:,:,:,iSample) = ...
                    ...permute(reshape(A,ndecs,nrows,ncols,nlays),[2 3 4 1]);
                    reshape(A,ndecs,nrows,ncols,nlays);
            end
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunBlockDct3dLayer(...
                'Stride',stride,...
                'Name','E0');
            
            % Actual values
            actualZ = layer.predict(X);
            
            % Evaluation
            testCase.verifyInstanceOf(actualZ,datatype);
            testCase.verifyThat(actualZ,...
                IsEqualTo(expctdZ,'Within',tolObj));
            
        end
        
        
        function testBackward(testCase, ...
                stride, height, width, depth, datatype)
            import tansacnet.utility.Direction
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-5,single(1e-5));
            
            % Parameters
            nSamples = 8;
            nrows = height/stride(Direction.VERTICAL);
            ncols = width/stride(Direction.HORIZONTAL);
            nlays = depth/stride(Direction.DEPTH);
            nDecs = prod(stride);
            %dLdZ = rand(nrows,ncols,nDecs,nSamples,datatype);
            dLdZ = rand(nDecs,nrows,ncols,nlays,nSamples,datatype);
            
            % Expected values
            expctddLdX = zeros(height,width,depth,datatype);
            E0_T = transpose(testCase.getMatrixE0_(stride));
            for iSample = 1:nSamples
                %A = reshape(permute(dLdZ(:,:,:,:,iSample),[4 1 2 3]),...
                %    nDecs,nrows*ncols*nlays);
                A = reshape(dLdZ(:,:,:,:,iSample),nDecs,nrows*ncols*nlays);               
                % Block IDCT
                Y = E0_T*A;
                expctddLdX(:,:,:,1,iSample) = testCase.col2vol_(Y,stride,...
                    [nrows,ncols,nlays]);
            end
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunBlockDct3dLayer(...
                'Stride',stride,...
                'Name','E0');
            
            % Actual values
            actualdLdX = layer.backward([],[],dLdZ,[]);
            
            % Evaluation
            testCase.verifyInstanceOf(actualdLdX,datatype);
            testCase.verifyThat(actualdLdX,...
                IsEqualTo(expctddLdX,'Within',tolObj));
            
        end

    end
    
    methods (Static, Access = private)

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