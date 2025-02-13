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

    end

end