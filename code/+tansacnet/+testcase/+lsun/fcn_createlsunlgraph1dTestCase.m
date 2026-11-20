classdef fcn_createlsunlgraph1dTestCase < matlab.unittest.TestCase
    %FCN_CREATELSUNLGRAPH1DTESTCASE
    %
    %   Test cases for tansacnet.lsun.fcn_createlsunlgraph1d, including
    %   its BoundaryOperation option ('Circular' | 'Termination') and
    %   copying the parameters by tansacnet.lsun.fcn_cpparamsana2syn and
    %   fcn_cpparamssyn2ana.
    %
    %   'Termination' is given to the odd-order intermediate rotation
    %   layers, i.e., those right after the shift of the difference
    %   channels, in both the analyzer and the synthesizer. The even-order
    %   layers stay 'Circular'.
    %
    %   As a result, the coefficients never wrap around to the opposite
    %   boundary, while the perfect reconstruction is kept.
    %
    %   The networks are evaluated in double precision. Since dlnetwork
    %   casts the learnable parameters to single, the random angles are
    %   given in single, copied to the synthesizer, and then both networks
    %   are cast to double so that they are exactly adjoint. The tests on
    %   GPU run only if a GPU with full-rate FP64 arithmetic is available.
    %
    % References:
    %   S. Muramatsu, T. Kobayashi, M. Hiki and H. Kikuchi, "Boundary
    %   Operation of 2-D Nonseparable Linear-Phase Paraunitary Filter Banks,"
    %   IEEE Trans. Image Process., vol. 21, no. 4, pp. 2314-2318, 2012.
    %
    % Requirements: MATLAB R2022b
    %
    % Copyright (c) 2026, Shogo MURAMATSU
    %
    % All rights reserved.
    %
    % Contact address: Shogo MURAMATSU,
    %                Faculty of Engineering, Niigata University,
    %                8050 2-no-cho Ikarashi, Nishi-ku,
    %                Niigata, 950-2181, JAPAN
    %
    % http://msiplab.eng.niigata-u.ac.jp/

    properties (Constant)
        NBLOCKS = 8      % Number of blocks
        STDANG = pi/6    % Standard deviation of random angles
        TOL = 1e-10      % Absolute tolerance in double precision
    end

    properties (TestParameter)
        stride = { 2, 4, 8 };
        ovlpFactor = { 1, 3, 5 };
        boundaryOperation = { 'Circular', 'Termination' };
        usegpu = struct( 'true', true, 'false', false);
        nLevels = { 1, 2, 3 };
    end

    methods (Test)

        function testDefaultBoundaryOperation(testCase, stride, ovlpFactor)
            % All intermediate rotation layers are 'Circular' by default

            % Instantiation of target
            lgraph = testCase.createLgraph_(stride,ovlpFactor,'Whole',false);

            % Evaluation
            layers = testCase.intermediateRotationLayers_(lgraph);
            for iLayer = 1:numel(layers)
                testCase.verifyEqual(layers(iLayer).BoundaryOperation,'Circular',...
                    layers(iLayer).Name);
            end
        end

        function testLayerSettings(testCase, stride, ovlpFactor, boundaryOperation)
            % Odd-order Vx layers get the given BoundaryOperation.
            % Even-order ones stay 'Circular'

            % Instantiation of target
            lgraph = testCase.createLgraph_(stride,ovlpFactor,'Whole',false,...
                'BoundaryOperation',boundaryOperation);

            % Evaluation
            layers = testCase.intermediateRotationLayers_(lgraph);
            nExpctdLayers = 2*(ovlpFactor-1);
            testCase.verifyNumElements(layers,nExpctdLayers);
            for iLayer = 1:numel(layers)
                layer = layers(iLayer);
                tokens = regexp(layer.Name,'_Vx(\d+)~?$','tokens','once');
                order = str2double(tokens{1});
                if mod(order,2) == 1
                    expctdBoundaryOperation = boundaryOperation;
                else
                    expctdBoundaryOperation = 'Circular';
                end
                testCase.verifyEqual(layer.BoundaryOperation,...
                    expctdBoundaryOperation,layer.Name);
            end
        end

        function testInvalidBoundaryOperation(testCase)
            % An invalid BoundaryOperation throws an exception
            testCase.verifyError(@() testCase.createLgraph_(2,3,'Whole',false,...
                'BoundaryOperation','InvalidBoundaryOperation'),...
                'LsunLayer:InvalidBoundaryOperation');
        end

        function testLayerOrder(testCase)
            % Layers of the analyzer and the synthesizer

            % Instantiation of target
            lgraph = testCase.createLgraph_(2,5,'Analyzer',false);

            % Evaluation
            testCase.verifyEqual({lgraph.Layers.Name},...
                {'Sequence input','Lv1_In','Lv1_E0','Lv1_Cmp1_V0',...
                'Lv1_Cmp1_Qx1rd','Lv1_Cmp1_Vx1',...
                'Lv1_Cmp1_Qx2ls','Lv1_Cmp1_Vx2',...
                'Lv1_Cmp1_Qx3rd','Lv1_Cmp1_Vx3',...
                'Lv1_Cmp1_Qx4ls','Lv1_Cmp1_Vx4',...
                'Lv1_Cmp1_Sp','Lv1_AcOut','Lv1_DcOut'});

            % Instantiation of target
            lgraph = testCase.createLgraph_(2,3,'Synthesizer',false,...
                'AppendInOutLayers',false);

            % Evaluation
            testCase.verifyEqual({lgraph.Layers.Name},...
                {'Lv1_DcIn','Lv1_AcIn','Lv1_Cmp1_Cn',...
                'Lv1_Cmp1_Vx2~','Lv1_Cmp1_Qx2ls~',...
                'Lv1_Cmp1_Vx1~','Lv1_Cmp1_Qx1rd~',...
                'Lv1_Cmp1_V0~','Lv1_E0~','Lv1_Out'});
        end

        function testInvalidStride(testCase)
            % An odd stride throws an exception
            testCase.verifyError(@() testCase.createLgraph_(3,3,'Whole',false),...
                'LsunLayer:InvalidStride');
        end

        function testInvalidOverlappingFactor(testCase)
            % An even overlapping factor throws an exception
            testCase.verifyError(@() testCase.createLgraph_(2,2,'Whole',false),...
                'LsunLayer:InvalidOverlappingFactor');
        end

        function testInvalidNumberOfComponents(testCase)
            % Multiple components throw an exception
            testCase.verifyError(@() testCase.createLgraph_(2,3,'Whole',false,...
                'NumberOfComponents',2),...
                'LsunLayer:InvalidNumberOfComponents');
        end

        function testInvalidInputSizeMultiLevel(testCase)
            % The input size should be a multiple of stride^nLevels
            import tansacnet.lsun.*
            testCase.verifyError(@() fcn_createlsunlgraph1d([],...
                'InputSize',12,'Stride',2,'NumberOfLevels',3),?MException);
        end

        function testNumberOfBlocksMultiLevel(testCase, stride, nLevels)
            % The number of blocks is reduced by the stride level by level
            import tansacnet.lsun.*
            szInput = 2*stride^nLevels;
            lgraph = fcn_createlsunlgraph1d([],...
                'InputSize',szInput,'Stride',stride,...
                'OverlappingFactor',3,'NumberOfLevels',nLevels,...
                'Device','cpu');
            names = string({lgraph.Layers.Name});
            for iLv = 1:nLevels
                for name = ["V0","Vx1","Vx2","Vx2~","Vx1~","V0~"]
                    layer = lgraph.Layers(names == "Lv" + iLv + "_Cmp1_" + name);
                    testCase.verifyEqual(layer.NumberOfBlocks,...
                        szInput/stride^iLv,layer.Name);
                end
            end
        end

        function testBlockDctByDefault(testCase, ...
                usegpu, stride, ovlpFactor, boundaryOperation)
            % The default design parameters realize a block DCT

            testCase.assumeFp64Gpu_(usegpu);
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(testCase.TOL);
            szInput = testCase.NBLOCKS*stride;

            % Expected values (DCT coefficients with even ones first)
            X = randn(1,szInput);
            C = dctmtx(stride);
            C = [ C(1:2:end,:) ; C(2:2:end,:) ];
            expctd = permute(C*reshape(X,stride,[]),[3 2 1]);

            % Instantiation of target
            analysisnet = dlupdate(@double,dlnetwork(...
                testCase.createLgraph_(stride,ovlpFactor,'Analyzer',usegpu,...
                'BoundaryOperation',boundaryOperation)));
            if usegpu
                analysisnet = dlupdate(@gpuArray,analysisnet);
                X = gpuArray(X);
            end

            % Evaluation
            [ac,dc] = analysisnet.predict(dlarray(X,'SSCB'));
            if usegpu
                testCase.verifyClass(extractdata(ac),'gpuArray');
            end
            actual = gather(cat(3,extractdata(dc),extractdata(ac)));
            testCase.verifyClass(actual,'double');
            testCase.verifyThat(actual,IsEqualTo(expctd,'Within',tolObj));
        end

        function testPerfectReconstructionMultiLevel(testCase, ...
                usegpu, stride, boundaryOperation, nLevels)
            % The perfect reconstruction and the energy preservation hold
            % for multiple tree levels

            testCase.assumeFp64Gpu_(usegpu);
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(testCase.TOL);
            ovlpFactor_ = 3;
            szInput = 2*stride^nLevels;
            opts = {'InputSize',szInput,'NumberOfLevels',nLevels};

            % Instantiation of target
            [analysisnet,synthesisnet] = testCase.createNetworks_(...
                stride,ovlpFactor_,boundaryOperation,boundaryOperation,usegpu,...
                opts{:});

            % Evaluation
            X = randn([1 szInput 1 2]);
            if usegpu
                X = gpuArray(X);
            end
            s = cell(1,nLevels+1);
            [s{:}] = analysisnet.predict(dlarray(X,'SSCB'));
            for iLv = 1:nLevels
                testCase.verifySize(s{iLv},...
                    [1 szInput/stride^iLv stride-1 2]);
            end
            testCase.verifySize(s{end},[1 szInput/stride^nLevels 1 2]);
            if usegpu
                testCase.verifyClass(extractdata(s{1}),'gpuArray');
            end
            X = gather(X);
            energy = sum(cellfun(@(x) gather(sum(extractdata(x).^2,'all')),s));
            testCase.verifyEqual(energy,sum(X.^2,'all'),'RelTol',testCase.TOL);
            actualY = gather(extractdata(synthesisnet.predict(s{:})));
            testCase.verifyClass(actualY,'double');
            testCase.verifyThat(actualY,IsEqualTo(X,'Within',tolObj));
        end

        function testNoDcLeakage(testCase, usegpu, stride, ovlpFactor, ...
                boundaryOperation, nLevels)
            % A constant sequence gives no AC coefficients with a vanishing
            % moment (default)

            testCase.assumeFp64Gpu_(usegpu);
            szInput = 2*stride^nLevels;

            % Instantiation of target
            analysisnet = testCase.createNetworks_(...
                stride,ovlpFactor,boundaryOperation,boundaryOperation,usegpu,...
                'InputSize',szInput,'NumberOfLevels',nLevels);

            % Evaluation
            X = ones(1,szInput);
            if usegpu
                X = gpuArray(X);
            end
            s = cell(1,nLevels+1);
            [s{:}] = analysisnet.predict(dlarray(X,'SSCB'));
            for iLv = 1:nLevels
                testCase.verifyEqual(gather(extractdata(s{iLv})),...
                    zeros(size(s{iLv})),'AbsTol',testCase.TOL);
            end
            testCase.verifyEqual(gather(extractdata(s{end})),...
                sqrt(stride)^nLevels*ones(size(s{end})),'AbsTol',testCase.TOL);
        end

        function testWhole(testCase, usegpu, nLevels)
            % The whole network reconstructs its input

            testCase.assumeFp64Gpu_(usegpu);
            import tansacnet.lsun.*
            if usegpu
                device = 'cuda';
            else
                device = 'cpu';
            end
            stride_ = 4;
            szInput = 2*stride_^nLevels;
            lgraph = fcn_createlsunlgraph1d([],...
                'InputSize',szInput,...
                'Stride',stride_,...
                'OverlappingFactor',3,...
                'NumberOfLevels',nLevels,...
                'AppendInOutLayers',false,...
                'Device',device,...
                'DType','double');
            X = randn([1 szInput 1 2]);
            net = dlupdate(@double,dlnetwork(lgraph,dlarray(X,'SSCB')));
            if usegpu
                net = dlupdate(@gpuArray,net);
                X = gpuArray(X);
            end

            % Evaluation
            actualY = extractdata(net.predict(dlarray(X,'SSCB')));
            if usegpu
                testCase.verifyClass(actualY,'gpuArray');
            end
            actualY = gather(actualY);
            testCase.verifyClass(actualY,'double');
            testCase.verifyEqual(actualY,gather(X),'AbsTol',testCase.TOL);
        end

        function testBackwardMultiLevel(testCase, usegpu, boundaryOperation)
            % The gradients w.r.t. the angles are computed through
            % multiple tree levels and agree with numerical differences

            testCase.assumeFp64Gpu_(usegpu);
            stride_ = 4;
            nLevels_ = 2;
            szInput = 2*stride_^nLevels_;
            analysisnet = testCase.createNetworks_(...
                stride_,3,boundaryOperation,boundaryOperation,usegpu,...
                'InputSize',szInput,'NumberOfLevels',nLevels_);
            X = randn([1 szInput 1 2]);
            if usegpu
                X = gpuArray(X);
            end
            dlX = dlarray(X,'SSCB');

            % Actual values
            [loss,grads] = dlfeval(@testCase.gradients_,analysisnet,dlX);
            testCase.verifyTrue(isfinite(gather(extractdata(loss))));

            % Expected values by central differences
            isAngles = find(analysisnet.Learnables.Parameter == "Angles").';
            delta = 1e-6;
            for iPrm = isAngles
                value = analysisnet.Learnables.Value{iPrm};
                grad = gather(extractdata(grads.Value{iPrm}));
                testCase.verifySize(grad,size(value));
                % The first ps-1 angles of the initial rotation are fixed
                % to zero for no DC leakage
                iRow = randi(size(value,1));
                if endsWith(analysisnet.Learnables.Layer(iPrm),"V0")
                    iRow = randi([stride_/2 size(value,1)]);
                end
                iElm = sub2ind(size(value),iRow,randi(size(value,2)));
                netp = analysisnet;
                netp.Learnables.Value{iPrm}(iElm) = value(iElm) + delta;
                netm = analysisnet;
                netm.Learnables.Value{iPrm}(iElm) = value(iElm) - delta;
                expctd = gather(extractdata(...
                    testCase.gradients_(netp,dlX) ...
                    - testCase.gradients_(netm,dlX)))/(2*delta);
                testCase.verifyEqual(grad(iElm),expctd,'AbsTol',1e-6,...
                    char(analysisnet.Learnables.Layer(iPrm)));
            end
        end

        function testPerfectReconstruction(testCase, ...
                usegpu, stride, ovlpFactor, boundaryOperation)
            % The synthesizer reconstructs the input of the analyzer

            testCase.assumeFp64Gpu_(usegpu);
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(testCase.TOL);

            % Instantiation of target
            [analysisnet,synthesisnet] = testCase.createNetworks_(...
                stride,ovlpFactor,boundaryOperation,boundaryOperation,usegpu);

            % Evaluation
            [actualY,X] = testCase.reconstruct_(analysisnet,synthesisnet,...
                testCase.NBLOCKS*stride,usegpu);
            testCase.verifyClass(actualY,'double');
            testCase.verifyThat(actualY,IsEqualTo(X,'Within',tolObj));
        end

        function testNoWrapAroundTermination(testCase, ...
                usegpu, stride, ovlpFactor)
            % With 'Termination', an input on the left (right) edge block
            % never reaches the coefficients of the right (left) edge block

            testCase.assumeFp64Gpu_(usegpu);
            testCase.assumeGreaterThan(ovlpFactor,1,'No overlap.')
            [leftToRight,rightToLeft] = testCase.edgeResponses_(...
                stride,ovlpFactor,'Termination',usegpu);

            % Evaluation
            testCase.verifyEqual(leftToRight,0,'AbsTol',testCase.TOL,...
                'Left edge input wraps around to the right edge.');
            testCase.verifyEqual(rightToLeft,0,'AbsTol',testCase.TOL,...
                'Right edge input wraps around to the left edge.');
        end

        function testWrapAroundCircular(testCase, usegpu, stride, ovlpFactor)
            % With 'Circular', an input on an edge reaches the opposite
            % edge (sensitivity check of the no-wrap-around test)

            testCase.assumeFp64Gpu_(usegpu);
            testCase.assumeGreaterThan(ovlpFactor,1,'No overlap.')
            % (ps,pa) = (1,1) has no angles and is always the block DCT
            testCase.assumeGreaterThan(stride,2,'No design parameters.')
            [fwd,bwd] = testCase.edgeResponses_(...
                stride,ovlpFactor,'Circular',usegpu);
            testCase.verifyGreaterThan(fwd,1e-4,'No wrap-around.');
            testCase.verifyGreaterThan(bwd,1e-4,'No wrap-around.');
        end

        function testCopyParamsAna2SynBoundaryOperation(testCase, ...
                stride, ovlpFactor, boundaryOperation)
            % fcn_cpparamsana2syn copies BoundaryOperation
            % so that the synthesizer created with the other mode becomes
            % the adjoint of the analyzer

            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(testCase.TOL);
            otherOperation = testCase.otherOperation_(boundaryOperation);

            % Instantiation of target
            [analysisnet,synthesisnet] = testCase.createNetworks_(...
                stride,ovlpFactor,boundaryOperation,otherOperation,false);

            % Evaluation of the layer settings
            testCase.verifyCopiedSettings_(...
                layerGraph(analysisnet),layerGraph(synthesisnet),'~');

            % Evaluation of the perfect reconstruction
            [actualY,X] = testCase.reconstruct_(analysisnet,synthesisnet,...
                testCase.NBLOCKS*stride,false);
            testCase.verifyThat(actualY,IsEqualTo(X,'Within',tolObj));
        end

        function testCopyParamsSyn2AnaBoundaryOperation(testCase, ...
                stride, ovlpFactor, boundaryOperation)
            % fcn_cpparamssyn2ana copies BoundaryOperation
            % from the synthesizer to the analyzer

            otherOperation = testCase.otherOperation_(boundaryOperation);
            synthesislgraph = testCase.createLgraph_(stride,ovlpFactor,...
                'Synthesizer',false,'BoundaryOperation',boundaryOperation);
            analysislgraph = testCase.createLgraph_(stride,ovlpFactor,...
                'Analyzer',false,'BoundaryOperation',otherOperation); %#ok<NASGU> used in evalc

            % Actual values
            import tansacnet.lsun.*
            [~,analysislgraph] = evalc(...
                'fcn_cpparamssyn2ana(analysislgraph,synthesislgraph)');

            % Evaluation
            testCase.verifyCopiedSettings_(synthesislgraph,analysislgraph,'');
        end

    end

    methods (Access = private)

        function assumeFp64Gpu_(testCase,usegpu)
            if usegpu
                testCase.assumeTrue(tansacnet.utility.isFp64GpuAvailable(), ...
                    'No GPU device with full-rate FP64 arithmetic was detected.')
            end
        end

        function lgraph = createLgraph_(testCase,stride,ovlpFactor,mode,usegpu,varargin)
            import tansacnet.lsun.*
            if usegpu
                device = 'cuda';
            else
                device = 'cpu';
            end
            lgraph = fcn_createlsunlgraph1d([],...
                'InputSize',testCase.NBLOCKS*stride,...
                'Stride',stride,...
                'OverlappingFactor',ovlpFactor,...
                'NumberOfLevels',1,...
                'Mode',mode,...
                'Device',device,...
                'DType','double',...
                varargin{:});
        end

        function [analysisnet,synthesisnet] = createNetworks_(testCase,...
                stride,ovlpFactor,anaOperation,synOperation,usegpu,varargin)
            % Analyzer with random angles and the synthesizer with the
            % angles (and the boundary operation) copied from the analyzer
            import tansacnet.lsun.*
            analysislgraph = testCase.createLgraph_(stride,ovlpFactor,...
                'Analyzer',usegpu,'BoundaryOperation',anaOperation,...
                varargin{:});
            synthesislgraph = testCase.createLgraph_(stride,ovlpFactor,...
                'Synthesizer',usegpu,'BoundaryOperation',synOperation,...
                varargin{:}); %#ok<NASGU> used in evalc
            % Random angles in single, since dlnetwork casts learnables to single
            analysisnet = dlnetwork(analysislgraph);
            isAngles = analysisnet.Learnables.Parameter == "Angles";
            analysisnet.Learnables.Value(isAngles) = ...
                cellfun(@(x) x+testCase.STDANG*randn(size(x),'like',x), ...
                analysisnet.Learnables.Value(isAngles),'UniformOutput',false);
            % evalc suppresses the messages of fcn_cpparamsana2syn
            [~,synthesislgraph] = evalc(...
                'fcn_cpparamsana2syn(synthesislgraph,layerGraph(analysisnet))');
            synthesisnet = dlnetwork(synthesislgraph);
            % Cast both networks to double after copying the angles
            analysisnet = dlupdate(@double,analysisnet);
            synthesisnet = dlupdate(@double,synthesisnet);
            if usegpu
                analysisnet = dlupdate(@gpuArray,analysisnet);
                synthesisnet = dlupdate(@gpuArray,synthesisnet);
            end
        end

        function [fwd,bwd] = edgeResponses_(testCase,stride,ovlpFactor,...
                boundaryOperation,usegpu)
            % fwd: max |coefs| on the last block for an input on the
            %      first block
            % bwd: max |coefs| on the first block for an input on the
            %      last block
            szInput = testCase.NBLOCKS*stride;
            analysisnet = testCase.createNetworks_(stride,ovlpFactor,...
                boundaryOperation,boundaryOperation,usegpu);
            %
            X = zeros(1,szInput);
            X(1:stride) = randn(1,stride);
            fwd = testCase.edgeCoefMax_(analysisnet,X,'last',usegpu);
            %
            X = zeros(1,szInput);
            X(end-stride+1:end) = randn(1,stride);
            bwd = testCase.edgeCoefMax_(analysisnet,X,'first',usegpu);
        end

        function verifyCopiedSettings_(testCase,srclgraph,dstlgraph,suffix)
            % The intermediate rotation layers of dstlgraph have the same
            % BoundaryOperation as those of srclgraph
            srcLayers = testCase.intermediateRotationLayers_(srclgraph);
            dstNames = string({dstlgraph.Layers.Name});
            for iLayer = 1:numel(srcLayers)
                srcLayer = srcLayers(iLayer);
                dstName = erase(string(srcLayer.Name),"~") + suffix;
                dstLayer = dstlgraph.Layers(dstNames == dstName);
                testCase.verifyEqual(dstLayer.BoundaryOperation,...
                    srcLayer.BoundaryOperation,dstName);
            end
        end

    end

    methods (Static, Access = private)

        function [Y,X] = reconstruct_(analysisnet,synthesisnet,szInput,usegpu)
            % Analysis followed by synthesis of a random input in double
            X = randn(1,szInput);
            if usegpu
                X = gpuArray(X);
            end
            s = cell(1,2);
            [s{1:2}] = analysisnet.predict(dlarray(X,'SSCB'));
            Y = gather(extractdata(synthesisnet.predict(s{:})));
            X = gather(X);
        end

        function value = edgeCoefMax_(analysisnet,X,edge,usegpu)
            % Maximum absolute value of the AC and DC coefficients on the
            % first or last block
            if usegpu
                X = gpuArray(X);
            end
            s = cell(1,2);
            [s{1:2}] = analysisnet.predict(dlarray(X,'SSCB'));
            value = 0;
            for iCoef = 1:2
                coefs = gather(extractdata(s{iCoef}));
                if strcmp(edge,'first')
                    iBlk = 1;
                else
                    iBlk = size(coefs,2);
                end
                value = max(value,max(abs(coefs(:,iBlk,:,:)),[],'all'));
            end
        end

        function layers = intermediateRotationLayers_(lgraph)
            isTarget = arrayfun(@(x) ...
                isa(x,'tansacnet.lsun.lsunIntermediateRotation1dLayer'),...
                lgraph.Layers);
            layers = lgraph.Layers(isTarget);
        end

        function [loss,grads] = gradients_(net,dlX)
            % Loss on all outputs and its gradients
            s = cell(1,numel(net.OutputNames));
            [s{:}] = net.forward(dlX);
            loss = 0;
            for iOut = 1:numel(s)
                loss = loss + sum(stripdims(s{iOut}).^3,'all');
            end
            if nargout > 1
                grads = dlgradient(loss,net.Learnables);
            end
        end

        function value = otherOperation_(boundaryOperation)
            if strcmp(boundaryOperation,'Circular')
                value = 'Termination';
            else
                value = 'Circular';
            end
        end

    end

end
