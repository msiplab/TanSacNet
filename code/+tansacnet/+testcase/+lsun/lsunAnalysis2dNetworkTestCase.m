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
        Stride = {[2 1], [1 2], [2 2], [2 4], [4 1], [4 4]};
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

        function testNetwork(testCase, datatype, device)

            stride = [2 2];
            height = 16;
            width = 16;
            %nSamples = 8;
            %nComponents = 1;
            import tansacnet.lsun.*
            net = lsunAnalysis2dNetwork('InputSize',[height width], ...
                'Stride',stride, ...
                'DType',datatype, ...
                'Device',device);
            dlnet = net.dlnetwork();
            %analyzeNetwork(dlnet)
            dlnet_ = initialize(dlnet);
            
            X = rand([height, width], datatype);
            %X = rand([nSamples, nComponents, height, width], datatype);
            X = dlarray(X, 'SSCB');
            
            actualZ = forward(dlnet_, X);
            testCase.verifyInstanceOf(extractdata(actualZ),datatype);
     
        end





        % function testForward(testCase, datatype, device)
        %     import tansacnet.utility.Direction
        %     import matlab.unittest.constraints.IsEqualTo
        %     import matlab.unittest.constraints.AbsoluteTolerance
        %     tolObj = AbsoluteTolerance(1e-5,single(1e-5));
        % 
        %     Stride = [2 2];
        %     nSamples = 8;
        %     %nComponents = 1;
        %     height = 64;
        %     width = 32;
        %     nDecs = Stride(Direction.VERTICAL) * Stride(Direction.HORIZONTAL);
        % 
        %     X = rand([height, width, nSamples], datatype);
        %     X = dlarray(X, 'SSBC');
        %     %X = rand([nSamples, nComponents, height, width], datatype);
        %     nrows = ceil(height / Stride(Direction.VERTICAL));
        %     ncols = ceil(width / Stride(Direction.HORIZONTAL));
        % 
        %     arrayshape = numel(X)/prod(Stride);
        %     X_reshaped =  reshape(X, [arrayshape, Stride]);
        % 
        %     Y = zeros(size(X_reshaped));
        %     for i = 1:size(X_reshaped, 1)
        %         Y(i, :, :) = blockproc(squeeze(extractdata(X_reshaped(i, :, :))), Stride, @(x) dct2(x.data));
        %     end
        % 
        %     A = testCase.permuteDctCoefs_(Y); 
        %     disp(size(A))
        %     disp([nSamples, nrows, ncols, nDecs])
        %     V = reshape(A, [nSamples, nrows, ncols, nDecs]); 
        % 
        %     ps = ceil(nDecs / 2);
        %     pa = floor(nDecs / 2);
        % 
        %     W0 = eye(ps, datatype);
        %     U0 = eye(pa, datatype);
        % 
        %     Zsa = zeros(nDecs, nrows * ncols * nSamples, datatype);
        %     Ys = permute(V(:, :, :, 1:ps), [4, 1, 2, 3]);
        %     Ys = reshape(Ys, ps, []);
        %     Zsa(1:ps, :) = W0 * Ys;
        % 
        %     if pa > 0
        %         Ya = permute(V(:, :, :, ps+1:end), [4, 1, 2, 3]);
        %         Ya = reshape(Ya, pa, []);
        %         Zsa(ps+1:end, :) = U0 * Ya;
        %     end
        % 
        %     expctdZ = permute(reshape(Zsa', [nrows, ncols, nSamples, nDecs]), ...
        %         [3, 1, 2, 4]);
        % 
        %     import tansacnet.lsun.*
        %     net = lsunAnalysis2dNetwork('InputSize',[height width], ...
        %         'Stride',Stride, ...
        %         'DType',datatype, ...
        %         'Device',device);
        %     dlnet = net.dlnetwork();
        %     dlnet_ = initialize(dlnet);
        %     analyzeNetwork(dlnet)
        % 
        %     actualZ = forward(dlnet_, X);
        %     disp(class(extractdata(actualZ)))
        %     disp(datatype)
        % 
        %     testCase.verifyInstanceOf((extractdata(actualZ)),datatype);
        %     testCase.verifyThat(permute(extractdata(actualZ),[4, 1, 2, 3]),...
        %         IsEqualTo(expctdZ(:,:,:,[1,2,4]),'Within',tolObj));
        % 
        % end
    end

    % methods (Static, Access = private)
    %     function value = permuteDctCoefs_(coefs)
    %         cee = coefs(:,1:2:end,1:2:end);
    %         coo = coefs(:,2:2:end,2:2:end);
    %         coe = coefs(:,2:2:end,1:2:end);
    %         ceo = coefs(:,1:2:end,2:2:end);
    %         value = cat(2, cee(:) , coo(:) , coe(:) , ceo(:) );
    %     end
    % end

end