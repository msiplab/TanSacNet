classdef fcn_createlsunlgraph3dTestCase < matlab.unittest.TestCase
    %FCN_CREATELSUNLGRAPH3DTESTCASE
    %
    %   Test cases for tansacnet.lsun.fcn_createlsunlgraph3d, including
    %   its BoundaryOperation option ('Circular' | 'Termination') and
    %   copying the parameters by tansacnet.lsun.fcn_cpparamsana2syn and
    %   fcn_cpparamssyn2ana.
    %
    %   'Termination' is given to the odd-order intermediate rotation
    %   layers, i.e., those right after the shift of the difference
    %   channels, with Direction 'Depth' (Vd), 'Horizontal' (Vh) or
    %   'Vertical' (Vv) in both the analyzer and the synthesizer. The even-order layers stay
    %   'Circular'.
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
    %   K. Furuya, S. Hara, K. Seino and S. Muramatsu, "Boundary operation of
    %   2D non-separable oversampled lapped transforms," APSIPA Trans. Signal
    %   Inf. Process., vol. 5, no. 1, pp. 1-9, 2016.
    %
    % Requirements: MATLAB R2022a
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
        NBLOCKS = [4 4 4] % Number of blocks [vertical horizontal depth]
        STDANG = pi/6    % Standard deviation of random angles
        TOL = 1e-10      % Absolute tolerance in double precision
    end

    properties (TestParameter)
        stride = { [2 2 2], [1 2 2] };
        ovlpFactor = { [3 3 3], [1 3 5], [5 1 3], [3 5 1] };
        boundaryOperation = { 'Circular', 'Termination' };
        usegpu = struct( 'true', true, 'false', false);
        nLevels = { 1, 2 };
        nComponents = { 1, 2 };
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
            % Odd-order Vd/Vh/Vv layers get the given BoundaryOperation
            % with Direction 'Depth'/'Horizontal'/'Vertical'. Even-order
            % ones stay 'Circular'

            % Instantiation of target
            lgraph = testCase.createLgraph_(stride,ovlpFactor,'Whole',false,...
                'BoundaryOperation',boundaryOperation);

            % Evaluation
            layers = testCase.intermediateRotationLayers_(lgraph);
            nExpctdLayers = 2*sum(ovlpFactor-1);
            testCase.verifyNumElements(layers,nExpctdLayers);
            for iLayer = 1:numel(layers)
                layer = layers(iLayer);
                tokens = regexp(layer.Name,'_V([dhv])(\d+)~?$','tokens','once');
                order = str2double(tokens{2});
                if mod(order,2) == 1
                    expctdBoundaryOperation = boundaryOperation;
                    if strcmp(tokens{1},'d')
                        expctdDirection = 'Depth';
                    elseif strcmp(tokens{1},'h')
                        expctdDirection = 'Horizontal';
                    else
                        expctdDirection = 'Vertical';
                    end
                else
                    expctdBoundaryOperation = 'Circular';
                    expctdDirection = '';
                end
                testCase.verifyEqual(layer.BoundaryOperation,...
                    expctdBoundaryOperation,layer.Name);
                if strcmp(expctdBoundaryOperation,'Termination')
                    testCase.verifyEqual(layer.Direction,...
                        expctdDirection,layer.Name);
                end
            end
        end

        function testInvalidBoundaryOperation(testCase)
            % An invalid BoundaryOperation throws an exception
            testCase.verifyError(@() testCase.createLgraph_([2 2 2],[3 3 3],'Whole',false,...
                'BoundaryOperation','InvalidBoundaryOperation'),...
                'LsunLayer:InvalidBoundaryOperation');
        end

        function testLayerOrder(testCase)
            % The atoms are extended in the depth, horizontal and vertical
            % directions in this order

            % Instantiation of target
            lgraph = testCase.createLgraph_([2 2 2],[3 3 3],'Analyzer',false);

            % Evaluation
            layers = testCase.intermediateRotationLayers_(lgraph);
            testCase.verifyEqual({layers.Name},...
                {'Lv1_Cmp1_Vd1','Lv1_Cmp1_Vd2',...
                'Lv1_Cmp1_Vh1','Lv1_Cmp1_Vh2',...
                'Lv1_Cmp1_Vv1','Lv1_Cmp1_Vv2'});
        end

        function testInvalidStride(testCase)
            % An odd product of stride throws an exception
            testCase.verifyError(@() testCase.createLgraph_([1 1 1],[3 3 3],'Whole',false),...
                'LsunLayer:InvalidStride');
        end

        function testInvalidOverlappingFactor(testCase)
            % An even overlapping factor throws an exception
            testCase.verifyError(@() testCase.createLgraph_([2 2 2],[3 2 3],'Whole',false),...
                'LsunLayer:InvalidOverlappingFactor');
        end

        function testBlockDctByDefault(testCase, ...
                usegpu, stride, ovlpFactor, boundaryOperation)
            % The default design parameters realize a block DCT

            testCase.assumeFp64Gpu_(usegpu);
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(testCase.TOL);
            szInput = testCase.NBLOCKS.*stride;

            % Expected values
            X = randn(szInput);
            dctnet = dlnetwork(tansacnet.lsun.lsunBlockDct3dLayer(...
                'Stride',stride),dlarray(X,'SSSCB'));
            expctd = permute(extractdata(dctnet.predict(dlarray(X,'SSSCB'))),...
                [2 3 4 1]);

            % Instantiation of target
            analysisnet = dlupdate(@double,dlnetwork(...
                testCase.createLgraph_(stride,ovlpFactor,'Analyzer',usegpu,...
                'BoundaryOperation',boundaryOperation)));
            if usegpu
                analysisnet = dlupdate(@gpuArray,analysisnet);
                X = gpuArray(X);
            end

            % Evaluation
            [ac,dc] = analysisnet.predict(dlarray(X,'SSSCB'));
            if usegpu
                testCase.verifyClass(extractdata(ac),'gpuArray');
            end
            actual = gather(cat(4,extractdata(dc),extractdata(ac)));
            testCase.verifyClass(actual,'double');
            testCase.verifyThat(actual,IsEqualTo(expctd,'Within',tolObj));
        end

        function testPerfectReconstructionMultiLevelMultiComponent(testCase, ...
                usegpu, boundaryOperation, nLevels, nComponents)
            % The perfect reconstruction and the energy preservation hold
            % for multiple tree levels and components

            testCase.assumeFp64Gpu_(usegpu);
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(testCase.TOL);
            stride_ = [2 2 2];
            ovlpFactor_ = [3 3 3];
            szInput = 2*stride_.^nLevels;
            opts = {'InputSize',szInput,'NumberOfLevels',nLevels,...
                'NumberOfComponents',nComponents};

            % Instantiation of target
            [analysisnet,synthesisnet] = testCase.createNetworks_(...
                stride_,ovlpFactor_,boundaryOperation,boundaryOperation,usegpu,...
                opts{:});

            % Evaluation
            X = randn([szInput nComponents 2]);
            if usegpu
                X = gpuArray(X);
            end
            s = cell(1,nLevels+1);
            [s{:}] = analysisnet.predict(dlarray(X,'SSSCB'));
            for iLv = 1:nLevels
                testCase.verifySize(s{iLv},...
                    [szInput./(stride_.^iLv) nComponents*(prod(stride_)-1) 2]);
            end
            testCase.verifySize(s{end},...
                [szInput./(stride_.^nLevels) nComponents 2]);
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

        function testWhole(testCase, usegpu, nLevels, nComponents)
            % The whole network reconstructs its input

            testCase.assumeFp64Gpu_(usegpu);
            import tansacnet.lsun.*
            if usegpu
                device = 'cuda';
            else
                device = 'cpu';
            end
            stride_ = [2 2 2];
            szInput = 2*stride_.^nLevels;
            lgraph = fcn_createlsunlgraph3d([],...
                'InputSize',szInput,...
                'Stride',stride_,...
                'OverlappingFactor',[3 3 3],...
                'NumberOfLevels',nLevels,...
                'NumberOfComponents',nComponents,...
                'AppendInOutLayers',false,...
                'Device',device,...
                'DType','double');
            X = randn([szInput nComponents 2]);
            net = dlupdate(@double,dlnetwork(lgraph,dlarray(X,'SSSCB')));
            if usegpu
                net = dlupdate(@gpuArray,net);
                X = gpuArray(X);
            end

            % Evaluation
            actualY = extractdata(net.predict(dlarray(X,'SSSCB')));
            if usegpu
                testCase.verifyClass(actualY,'gpuArray');
            end
            actualY = gather(actualY);
            testCase.verifyClass(actualY,'double');
            testCase.verifyEqual(actualY,gather(X),'AbsTol',testCase.TOL);
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
                testCase.NBLOCKS.*stride,usegpu);
            testCase.verifyClass(actualY,'double');
            testCase.verifyThat(actualY,IsEqualTo(X,'Within',tolObj));
        end

        function testNoWrapAroundHorizontalTermination(testCase, ...
                usegpu, stride, ovlpFactor)
            % With 'Termination', an input on the left (right) edge blocks
            % never reaches the coefficients of the right (left) edge blocks

            testCase.assumeFp64Gpu_(usegpu);
            testCase.assumeGreaterThan(ovlpFactor(2),1,...
                'No horizontal overlap.')
            [leftToRight,rightToLeft] = testCase.edgeResponses_(...
                stride,ovlpFactor,'Termination',2,usegpu);

            % Evaluation
            testCase.verifyEqual(leftToRight,0,'AbsTol',testCase.TOL,...
                'Left edge input wraps around to the right edge.');
            testCase.verifyEqual(rightToLeft,0,'AbsTol',testCase.TOL,...
                'Right edge input wraps around to the left edge.');
        end

        function testNoWrapAroundVerticalTermination(testCase, ...
                usegpu, stride, ovlpFactor)
            % With 'Termination', an input on the top (bottom) edge blocks
            % never reaches the coefficients of the bottom (top) edge blocks

            testCase.assumeFp64Gpu_(usegpu);
            testCase.assumeGreaterThan(ovlpFactor(1),1,...
                'No vertical overlap.')
            [topToBottom,bottomToTop] = testCase.edgeResponses_(...
                stride,ovlpFactor,'Termination',1,usegpu);

            % Evaluation
            testCase.verifyEqual(topToBottom,0,'AbsTol',testCase.TOL,...
                'Top edge input wraps around to the bottom edge.');
            testCase.verifyEqual(bottomToTop,0,'AbsTol',testCase.TOL,...
                'Bottom edge input wraps around to the top edge.');
        end

        function testNoWrapAroundDepthTermination(testCase, ...
                usegpu, stride, ovlpFactor)
            % With 'Termination', an input on the front (back) edge blocks
            % never reaches the coefficients of the back (front) edge blocks

            testCase.assumeFp64Gpu_(usegpu);
            testCase.assumeGreaterThan(ovlpFactor(3),1,...
                'No overlap in depth.')
            [frontToBack,backToFront] = testCase.edgeResponses_(...
                stride,ovlpFactor,'Termination',3,usegpu);

            % Evaluation
            testCase.verifyEqual(frontToBack,0,'AbsTol',testCase.TOL,...
                'Front edge input wraps around to the back edge.');
            testCase.verifyEqual(backToFront,0,'AbsTol',testCase.TOL,...
                'Back edge input wraps around to the front edge.');
        end

        function testWrapAroundCircular(testCase, usegpu, stride, ovlpFactor)
            % With 'Circular', an input on an edge reaches the opposite
            % edge in the overlapping direction (sensitivity check of the
            % no-wrap-around tests)

            testCase.assumeFp64Gpu_(usegpu);
            for dim = find(ovlpFactor > 1)
                [fwd,bwd] = testCase.edgeResponses_(...
                    stride,ovlpFactor,'Circular',dim,usegpu);
                testCase.verifyGreaterThan(fwd,1e-4,...
                    sprintf('No wrap-around in dimension %d.',dim));
                testCase.verifyGreaterThan(bwd,1e-4,...
                    sprintf('No wrap-around in dimension %d.',dim));
            end
        end

        function testCopyParamsAna2SynBoundaryOperation(testCase, ...
                stride, ovlpFactor, boundaryOperation)
            % fcn_cpparamsana2syn copies BoundaryOperation and Direction
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
                testCase.NBLOCKS.*stride,false);
            testCase.verifyThat(actualY,IsEqualTo(X,'Within',tolObj));
        end

        function testCopyParamsSyn2AnaBoundaryOperation(testCase, ...
                stride, ovlpFactor, boundaryOperation)
            % fcn_cpparamssyn2ana copies BoundaryOperation and Direction
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
            lgraph = fcn_createlsunlgraph3d([],...
                'InputSize',testCase.NBLOCKS.*stride,...
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
                boundaryOperation,dim,usegpu)
            % fwd: max |coefs| on the last edge blocks for an input on the
            %      first edge blocks along dimension dim
            % bwd: max |coefs| on the first edge blocks for an input on the
            %      last edge blocks along dimension dim
            szInput = testCase.NBLOCKS.*stride;
            analysisnet = testCase.createNetworks_(stride,ovlpFactor,...
                boundaryOperation,boundaryOperation,usegpu);
            edgeWidth = stride(dim);
            %
            X = zeros(szInput);
            idx = {':',':',':'};
            idx{dim} = 1:edgeWidth;
            X(idx{:}) = randn(size(X(idx{:})));
            fwd = testCase.edgeCoefMax_(analysisnet,X,dim,'last',usegpu);
            %
            X = zeros(szInput);
            idx = {':',':',':'};
            idx{dim} = szInput(dim)-edgeWidth+1:szInput(dim);
            X(idx{:}) = randn(size(X(idx{:})));
            bwd = testCase.edgeCoefMax_(analysisnet,X,dim,'first',usegpu);
        end

        function verifyCopiedSettings_(testCase,srclgraph,dstlgraph,suffix)
            % The intermediate rotation layers of dstlgraph have the same
            % BoundaryOperation and Direction as those of srclgraph
            srcLayers = testCase.intermediateRotationLayers_(srclgraph);
            dstNames = string({dstlgraph.Layers.Name});
            for iLayer = 1:numel(srcLayers)
                srcLayer = srcLayers(iLayer);
                dstName = erase(string(srcLayer.Name),"~") + suffix;
                dstLayer = dstlgraph.Layers(dstNames == dstName);
                testCase.verifyEqual(dstLayer.BoundaryOperation,...
                    srcLayer.BoundaryOperation,dstName);
                testCase.verifyEqual(dstLayer.Direction,...
                    srcLayer.Direction,dstName);
            end
        end

    end

    methods (Static, Access = private)

        function [Y,X] = reconstruct_(analysisnet,synthesisnet,szInput,usegpu)
            % Analysis followed by synthesis of a random input in double
            X = randn(szInput);
            if usegpu
                X = gpuArray(X);
            end
            s = cell(1,2);
            [s{1:2}] = analysisnet.predict(dlarray(X,'SSSCB'));
            Y = gather(extractdata(synthesisnet.predict(s{:})));
            X = gather(X);
        end

        function value = edgeCoefMax_(analysisnet,X,dim,edge,usegpu)
            % Maximum absolute value of the AC and DC coefficients on the
            % first or last blocks along dimension dim
            if usegpu
                X = gpuArray(X);
            end
            s = cell(1,2);
            [s{1:2}] = analysisnet.predict(dlarray(X,'SSSCB'));
            value = 0;
            for iCoef = 1:2
                coefs = gather(extractdata(s{iCoef}));
                idx = {':',':',':',':',':'};
                if strcmp(edge,'first')
                    idx{dim} = 1;
                else
                    idx{dim} = size(coefs,dim);
                end
                value = max(value,max(abs(coefs(idx{:})),[],'all'));
            end
        end

        function layers = intermediateRotationLayers_(lgraph)
            isTarget = arrayfun(@(x) ...
                isa(x,'tansacnet.lsun.lsunIntermediateRotation3dLayer'),...
                lgraph.Layers);
            layers = lgraph.Layers(isTarget);
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
