classdef lsunChannelSeparation1dLayerTestCase < matlab.unittest.TestCase
    %NSOLTCHANNELSEPARATION1DLAYERTESTCASE
    %
    %   １コンポーネント入力(nComponents=1のみサポート):
    %      nChsTotal x nSamples x nBlks
    %
    %   ２コンポーネント出力(nComponents=2のみサポート): "CBT"
    %      1 x nSamples x nBlks
    %      (nChsTotal-1) x nSamples x nBlks
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
            fprintf("\n --- Check layer for 1-D sequences ---\n");
            checkLayer(layer,[4 8 4],...
                'ObservationDimension',2,...
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
            % nChsTotal x nSamples x nBlks
            X = randn(nChsTotal,nSamples,nblks,datatype);
            
            % Expected values
            % nBlks x (nChsTotal-1) x nSamples 
            %expctdZ2 = X(:,2:end,:);
            expctdZac = X(2:end,:,:);
            % nBlks x 1 x nSamples
            %expctdZ1 = X(:,1,:);
            expctdZdc = X(1,:,:);
            
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
            % (nChsTotal-1) x nBlks x nSamples  
            dLdZac = randn(nChsTotal-1,nblks,nSamples,datatype);
            % 1 x nBlks x nSamples 
            dLdZdc = randn(1,nblks,nSamples,datatype);
            
            % Expected values
            % nChsTotal x nSamples x nBlks
            %expctddLdX = cat(1,dLdZ1,dLdZ2);
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
        
    end
    
end
