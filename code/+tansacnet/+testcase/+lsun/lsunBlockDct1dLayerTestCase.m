classdef lsunBlockDct1dLayerTestCase < matlab.unittest.TestCase
    %NSOLTBLOCKDCT1DLAYERTESTCASE
    %
    %   ベクトル配列をブロック配列を入力:
    %      nComponents x nSamples x (Stride(1)xnBlks) 
    %
    %   コンポーネント別に出力:
    %      nDecs x nSamples x nBlks 
    %
    % Requirements: MATLAB R2022b
    %
    % Copyright (c) 2023, Shogo MURAMATSU
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
        stride = { 1, 2, 4, 8 };
        datatype = { 'single', 'double' };
        seqlen = struct('small', 8,'medium', 16, 'large', 32);
    end
    
    methods (TestClassTeardown)
      
        function finalCheck(~)
            import tansacnet.lsun.*
            fprintf("\n --- Check layer for 1-D sequences ---\n");
            layer = lsunBlockDct1dLayer('Stride',2);
            checkLayer(layer,[1 8 8],'ObservationDimension',2,...
                'CheckCodegenCompatibility',false)
        end

    end
    
    methods (Test)
        
        function testConstructor(testCase, stride)
            
            % Expected values
            expctdName = 'E0';
            expctdDescription = "Block DCT of size " ...
                + stride(1);
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunBlockDct1dLayer(...
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
                stride, seqlen, datatype)
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-5,single(1e-5));
            
            % Parameters
            nSamples = 8;
            nComponents = 1;
            X = rand(nComponents, nSamples, seqlen, datatype);
            
            % Expected values
            nblks = ceil(seqlen/stride);
            expctdZ = zeros(stride,nSamples,nblks,datatype);
            for iSample = 1:nSamples
                % Block DCT
                U = reshape(permute(X(nComponents,iSample,:),[3 1 2]),...
                    stride,[]);
                if stride > 1
                    Y = dct(U);
                    % Rearrange the DCT Coefs.
                    A = testCase.permuteDctCoefs_(Y);
                    expctdZ(:,iSample,:) = ...
                        reshape(A,stride,1,nblks);
                else
                    expctdZ(:,iSample,:) = ...
                        reshape(U,stride,1,nblks);
                end
            end
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunBlockDct1dLayer(...
                'Stride',stride,...
                'Name','E0');
            
            % Actual values
            actualZ = layer.predict(X);
            
            % Evaluation
            testCase.verifyInstanceOf(actualZ,datatype);
            testCase.verifyThat(actualZ,...
                IsEqualTo(expctdZ,'Within',tolObj));
            
        end
        
        function testForward(testCase, ...
                stride, seqlen, datatype)
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-5,single(1e-5));
            
            % Parameters
            nSamples = 8;
            nComponents = 1;
            X = rand(nComponents, nSamples, seqlen, datatype);
            
            % Expected values
            nblks = ceil(seqlen/stride);
            expctdZ = zeros(stride,nSamples,nblks,datatype);
            for iSample = 1:nSamples
                % Block DCT
                U = reshape(permute(X(nComponents,iSample,:),[3 1 2]),...
                    stride,[]);
                if stride > 1
                    Y = dct(U);
                    % Rearrange the DCT Coefs.
                    A = testCase.permuteDctCoefs_(Y);
                    expctdZ(:,iSample,:) = ...
                        reshape(A,stride,1,nblks);
                else
                    expctdZ(:,iSample,:) = ...
                        reshape(U,stride,1,nblks);
                end
            end
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunBlockDct1dLayer(...
                'Stride',stride,...
                'Name','E0');
            
            % Actual values
            actualZ = layer.forward(X);
            
            % Evaluation
            testCase.verifyInstanceOf(actualZ,datatype);
            testCase.verifyThat(actualZ,...
                IsEqualTo(expctdZ,'Within',tolObj));
            
        end
        
        function testBackward(testCase, ...
                stride, seqlen, datatype)
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-5,single(1e-5));
            
            % Parameters
            nSamples = 8;
            nblks = seqlen/stride;
            nComponents = 1;
            %dLdZ = rand(nrows,ncols,nDecs,nSamples,datatype);
            dLdZ = rand(stride,nSamples,nblks,datatype);
            
            % Expected values
            expctddLdX = zeros(nComponents,nSamples,seqlen,datatype);
            for iSample = 1:nSamples
                A = reshape(dLdZ(:,iSample,:),stride,[]);
                if stride > 1
                    Y = testCase.permuteIdctCoefs_(A,stride);
                    expctddLdX(nComponents,iSample,:) = ...
                        reshape(idct(Y),nComponents,1,seqlen);
                else
                    expctddLdX(nComponents,iSample,:) = ...
                        reshape(A,nComponents,1,seqlen);
                end
            end

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunBlockDct1dLayer(...
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

    end
        
end
