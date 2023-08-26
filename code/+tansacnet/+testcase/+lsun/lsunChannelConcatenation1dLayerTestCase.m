classdef lsunChannelConcatenation1dLayerTestCase < matlab.unittest.TestCase
    %LSUNCHANNELCONCATENATION1DLAYERTESTCASE
    %
    %   ２コンポーネント出力(nComponents=2のみサポート):
    %      1 x 1 x nBlks x nSamples 
    %      (nChsTotal-1) x 1 x nBlks x nSamples
    %
    %   １コンポーネント入力(nComponents=1のみサポート):
    %      nChsTotal x 1 x nBlks x nSamples
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
            layer = lsunChannelConcatenation1dLayer();
            fprintf("\n --- Check layer for 1-D images ---\n");
            checkLayer(layer,{[3 1 4 8], [1 1 4 8]},...
                'ObservationDimension',4,...
                'CheckCodegenCompatibility',false) %true)
        end
    end
    
    methods (Test)
        
        function testConstructor(testCase)
            
            % Expected values
            expctdName = 'Cn';
            expctdDescription = "Channel concatenation";
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunChannelConcatenation1dLayer('Name',expctdName);
            
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
            % (nChsTotal-1) x 1 x nBlks x nSamples
            Xac = randn(nChsTotal-1,1,nblks,nSamples,datatype);
            % 1 x 1 x nBlks x nSamples
            Xdc = randn(1,1,nblks,nSamples,datatype);

            
            % Expected values
            % nChsTotal x 1 x nBlks x nSamples
            expctdZ = cat(1,Xdc,Xac);
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunChannelConcatenation1dLayer('Name','Cn');
            
            % Actual values
            actualZ = layer.predict(Xac,Xdc);
            
            % Evaluation
            testCase.verifyInstanceOf(actualZ,datatype);
            testCase.verifyThat(actualZ,...
                IsEqualTo(expctdZ,'Within',tolObj));
            
        end
                
        function testBackward(testCase,nchs,nblks,batch,datatype)
            
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            
            % Parameters
            nSamples = batch;
            nChsTotal = sum(nchs);
            % nChsTotal x 1 x nBlks x nSamples
            dLdZ = randn(nChsTotal,1,nblks,nSamples,datatype);
            
            % Expected values
            % (nChsTotal-1) x 1 x nBlks x nSamples
            expctddLdXac = dLdZ(2:end,:,:,:);
            % 1 x 1 x nBlks x nSamples
            expctddLdXdc = dLdZ(1,:,:,:);
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunChannelConcatenation1dLayer('Name','Cn');
            
            % Actual values
            [actualdLdXac,actualdLdXdc] = layer.backward([],[],[],dLdZ,[]);
            
            % Evaluation
            testCase.verifyInstanceOf(actualdLdXdc,datatype);
            testCase.verifyInstanceOf(actualdLdXac,datatype);            
            testCase.verifyThat(actualdLdXdc,...
                IsEqualTo(expctddLdXdc,'Within',tolObj));
            testCase.verifyThat(actualdLdXac,...
                IsEqualTo(expctddLdXac,'Within',tolObj));            
            
        end
        
    end
    
end

