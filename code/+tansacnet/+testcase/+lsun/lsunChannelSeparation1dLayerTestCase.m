classdef lsunChannelSeparation1dLayerTestCase < matlab.unittest.TestCase
    %NSOLTCHANNELSEPARATION1DLAYERTESTCASE
    %
    %   １コンポーネント入力(nComponents=1のみサポート):
    %      nChsTotal x 1 x nBlks x nSamples
    %
    %   ２コンポーネント出力(nComponents=2のみサポート): 
    %      1 x 1 x nBlks x nSamples
    %      (nChsTotal-1) x 1 x nBlks x nSamples
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
        nchs = { 2, 4, 8 };
        datatype = { 'single', 'double' };
        nblks = struct('small', 1,'medium', 4, 'large', 16);
        batch = { 1, 8 };
    end
    
    methods (TestClassTeardown)

        function finalCheck(~)
            import tansacnet.lsun.*
            layer = lsunChannelSeparation1dLayer();
            fprintf("\n --- Check layer for 1-D images ---\n");
            checkLayer(layer,[4 1 4 8],...
                'ObservationDimension',4,...
                'CheckCodegenCompatibility',false) %true)
        end

    end
    
    methods (Test)
        
        function testConstructor(testCase)
            
            % Expected values
            expctdName = 'Sp';
            expctdDescription = "Channel separation";
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunChannelSeparation1dLayer('Name',expctdName);
            
            % Actual values
            actualName = layer.Name;
            actualDescription = layer.Description;
            
            % Evaluation
            testCase.verifyEqual(actualName,expctdName);    
            testCase.verifyEqual(actualDescription,expctdDescription);
        end
        
        function testPredict(testCase,nchs,nblks,batch,datatype)
            
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            
            % Parameters
            nSamples = batch;
            nChsTotal = sum(nchs);
            % nChsTotal x 1 x nBlks x nSamples
            X = randn(nChsTotal,1,nblks,nSamples,datatype);
            
            % Expected values
            % (nChsTotal-1) x 1 x nBlks x nSamples 
            expctdZac = X(2:end,:,:,:);
            % 1 x 1 x nBlks x nSamples
            expctdZdc = X(1,:,:,:);
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunChannelSeparation1dLayer('Name','Sp');
            
            % Actual values
            [actualZac,actualZdc] = layer.predict(X);
            
            % Evaluation
            testCase.verifyInstanceOf(actualZdc,datatype);
            testCase.verifyInstanceOf(actualZac,datatype);            
            testCase.verifyThat(actualZdc,...
                IsEqualTo(expctdZdc,'Within',tolObj));
            testCase.verifyThat(actualZac,...
                IsEqualTo(expctdZac,'Within',tolObj));            
            
        end
        
        function testBackward(testCase,nchs,nblks,batch,datatype)
            
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            
            % Parameters
            nSamples = batch;
            nChsTotal = sum(nchs);
            % (nChsTotal-1) x 1 x nBlks x nSamples  
            dLdZac = randn(nChsTotal-1,nblks,nSamples,datatype);
            % 1 x 1 x nBlks x nSamples 
            dLdZdc = randn(1,nblks,nSamples,datatype);
            
            % Expected values
            % nChsTotal x 1 x nBlks x nSamples
            expctddLdX = cat(1,dLdZdc,dLdZac);
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunChannelSeparation1dLayer('Name','Sp');
            
            % Actual values
            actualdLdX = layer.backward([],[],[],dLdZac,dLdZdc,[]);
            
            % Evaluation
            testCase.verifyInstanceOf(actualdLdX,datatype);
            testCase.verifyThat(actualdLdX,...
                IsEqualTo(expctddLdX,'Within',tolObj));
            
        end

        function testDefaultChannelDimension(testCase)
            import tansacnet.lsun.*
            layer = lsunChannelSeparation1dLayer();
            testCase.verifyEqual(layer.ChannelDimension,1);
        end

        function testInvalidChannelDimension(testCase)
            import tansacnet.lsun.*
            testCase.verifyError(...
                @() lsunChannelSeparation1dLayer('ChannelDimension',2),...
                'LsunLayer:InvalidChannelDimension');
        end

        function testPredictChannelDimension3(testCase,nchs,nblks,batch,datatype)
            % AC and DC are arranged as the 2-D/3-D layers do

            % Parameters
            % nChsTotal x 1 x nBlks x nSamples
            X = randn(nchs,1,nblks,batch,datatype);

            % Expected values
            % 1 x nBlks x (nChsTotal-1) x nSamples
            expctdZac = permute(X(2:end,:,:,:),[2 3 1 4]);
            % 1 x nBlks x 1 x nSamples
            expctdZdc = permute(X(1,:,:,:),[2 3 1 4]);

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunChannelSeparation1dLayer('Name','Sp',...
                'ChannelDimension',3);

            % Actual values
            [actualZac,actualZdc] = layer.predict(X);

            % Evaluation
            testCase.verifyInstanceOf(actualZdc,datatype);
            testCase.verifyInstanceOf(actualZac,datatype);
            testCase.verifySize(actualZac,[1 nblks nchs-1 batch]);
            testCase.verifySize(actualZdc,[1 nblks 1 batch]);
            testCase.verifyEqual(actualZdc,expctdZdc);
            testCase.verifyEqual(actualZac,expctdZac);
        end

        function testBackwardChannelDimension3(testCase,nchs,nblks,batch,datatype)

            % Parameters
            % 1 x nBlks x (nChsTotal-1) x nSamples
            dLdZac = randn(1,nblks,nchs-1,batch,datatype);
            % 1 x nBlks x 1 x nSamples
            dLdZdc = randn(1,nblks,1,batch,datatype);

            % Expected values
            % nChsTotal x 1 x nBlks x nSamples
            expctddLdX = ipermute(cat(3,dLdZdc,dLdZac),[2 3 1 4]);

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunChannelSeparation1dLayer('Name','Sp',...
                'ChannelDimension',3);

            % Actual values
            actualdLdX = layer.backward([],[],[],dLdZac,dLdZdc,[]);

            % Evaluation
            testCase.verifyInstanceOf(actualdLdX,datatype);
            testCase.verifySize(actualdLdX,[nchs 1 nblks batch]);
            testCase.verifyEqual(actualdLdX,expctddLdX);
        end

    end

end
