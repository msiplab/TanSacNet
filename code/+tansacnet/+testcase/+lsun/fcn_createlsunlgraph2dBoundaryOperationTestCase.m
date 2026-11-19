classdef fcn_createlsunlgraph2dBoundaryOperationTestCase < matlab.unittest.TestCase
    %FCN_CREATELSUNLGRAPH2DBOUNDARYOPERATIONTESTCASE
    %
    %   Test cases for the BoundaryOperation option ('Circular' | 'Termination')
    %   of tansacnet.lsun.fcn_createlsunlgraph2d, and for copying it by
    %   tansacnet.lsun.fcn_cpparamsana2syn and fcn_cpparamssyn2ana.
    %
    %   'Termination' is given to the odd-order intermediate rotation
    %   layers, i.e., those right after the shift of the difference
    %   channels, with Direction 'Horizontal' (Vh) or 'Vertical' (Vv) in
    %   both the analyzer and the synthesizer. The even-order layers stay
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
        NBLOCKS = [8 8]  % Number of blocks [vertical horizontal]
        STDANG = pi/6    % Standard deviation of random angles
        TOL = 1e-10      % Absolute tolerance in double precision
    end

    properties (TestParameter)
        stride = { [2 2], [4 4] };
        ovlpFactor = { [3 3], [5 5], [1 3], [3 1] };
        boundaryOperation = { 'Circular', 'Termination' };
        usegpu = struct( 'true', true, 'false', false);
        isLegacy = struct( 'true', true, 'false', false);
        dim = { 1, 2 };
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
            % Odd-order Vh/Vv layers get the given BoundaryOperation with
            % Direction 'Horizontal'/'Vertical'. Even-order ones stay
            % 'Circular'

            % Instantiation of target
            lgraph = testCase.createLgraph_(stride,ovlpFactor,'Whole',false,...
                'BoundaryOperation',boundaryOperation);

            % Evaluation
            layers = testCase.intermediateRotationLayers_(lgraph);
            nExpctdLayers = 2*((ovlpFactor(1)-1)+(ovlpFactor(2)-1));
            testCase.verifyNumElements(layers,nExpctdLayers);
            for iLayer = 1:numel(layers)
                layer = layers(iLayer);
                tokens = regexp(layer.Name,'_V([hv])(\d+)~?$','tokens','once');
                order = str2double(tokens{2});
                if mod(order,2) == 1
                    expctdBoundaryOperation = boundaryOperation;
                    if strcmp(tokens{1},'h')
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
            testCase.verifyError(@() testCase.createLgraph_([2 2],[3 3],'Whole',false,...
                'BoundaryOperation','InvalidBoundaryOperation'),...
                'LsunLayer:InvalidBoundaryOperation');
        end

        function testNumberOfBlocksMultiLevel(testCase, boundaryOperation, isLegacy)
            % The rotation layers of level n have InputSize./(Stride.^n)
            % blocks. With 'LegacyNumberOfBlocks', all the levels have
            % those of level 1

            import tansacnet.lsun.*
            stride_ = [2 2];
            nLevels = 3;
            szInput = [32 48];
            lgraph = fcn_createlsunlgraph2d([],...
                'InputSize',szInput,'Stride',stride_,...
                'OverlappingFactor',[3 3],'NumberOfLevels',nLevels,...
                'BoundaryOperation',boundaryOperation,...
                'LegacyNumberOfBlocks',isLegacy,...
                'Device','cpu','DType','double');

            % Evaluation
            isTarget = arrayfun(@(x) isprop(x,'NumberOfBlocks'),lgraph.Layers);
            layers = lgraph.Layers(isTarget);
            testCase.verifyNumElements(layers,nLevels*2*(1+4));
            for iLayer = 1:numel(layers)
                layer = layers(iLayer);
                iLv = str2double(regexp(layer.Name,'^Lv(\d+)_','tokens','once'));
                if isLegacy
                    expctdNumberOfBlocks = szInput./stride_;
                else
                    expctdNumberOfBlocks = szInput./(stride_.^iLv);
                end
                testCase.verifyEqual(layer.NumberOfBlocks,...
                    expctdNumberOfBlocks,layer.Name);
                testCase.verifySize(layer.Angles,...
                    [size(layer.Angles,1) prod(expctdNumberOfBlocks)],layer.Name);
            end
        end

        function testLegacyNumberOfBlocksSingleLevel(testCase, stride, ovlpFactor)
            % 'LegacyNumberOfBlocks' makes no difference for a single level

            lgraph = testCase.createLgraph_(stride,ovlpFactor,'Whole',false);
            legacy = testCase.createLgraph_(stride,ovlpFactor,'Whole',false,...
                'LegacyNumberOfBlocks',true);

            % Evaluation
            testCase.verifyEqual({legacy.Layers.Name},{lgraph.Layers.Name});
            isTarget = arrayfun(@(x) isprop(x,'NumberOfBlocks'),lgraph.Layers);
            testCase.verifyEqual(...
                arrayfun(@(x) x.NumberOfBlocks,legacy.Layers(isTarget),'UniformOutput',false),...
                arrayfun(@(x) x.NumberOfBlocks,lgraph.Layers(isTarget),'UniformOutput',false));
        end

        function testInvalidInputSizeMultiLevel(testCase)
            % The input size should be a multiple of Stride.^NumberOfLevels
            import tansacnet.lsun.*
            testCase.verifyError(@() fcn_createlsunlgraph2d([],...
                'InputSize',[12 12],'Stride',[2 2],'NumberOfLevels',3),?MException);
        end

        function testPerfectReconstructionMultiLevel(testCase, ...
                usegpu, boundaryOperation)
            % The perfect reconstruction and the energy preservation hold
            % for multiple tree levels

            testCase.assumeFp64Gpu_(usegpu);
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(testCase.TOL);
            nLevels = 2;
            stride_ = [2 2];
            szInput = [16 24];

            % Instantiation of target
            [analysisnet,synthesisnet] = testCase.createNetworks_(...
                stride_,[3 3],boundaryOperation,boundaryOperation,usegpu,...
                'InputSize',szInput,'NumberOfLevels',nLevels);

            % Evaluation
            X = randn(szInput);
            if usegpu
                X = gpuArray(X);
            end
            s = cell(1,nLevels+1);
            [s{:}] = analysisnet.predict(dlarray(X,'SSCB'));
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

        function testNoWrapAroundTerminationMultiLevel(testCase, usegpu, dim)
            % With 'Termination', an input on the first (last) edge never
            % reaches the coefficients of the last (first) edge blocks in
            % any tree level

            testCase.assumeFp64Gpu_(usegpu);
            nLevels = 2;
            stride_ = [2 2];
            szInput = [32 32];
            analysisnet = testCase.createNetworks_(...
                stride_,[3 3],'Termination','Termination',usegpu,...
                'InputSize',szInput,'NumberOfLevels',nLevels);
            edgeWidth = stride_(dim)^nLevels;
            edges = {'last','first'};
            ranges = {1:edgeWidth, szInput(dim)-edgeWidth+1:szInput(dim)};
            for iEdge = 1:2
                X = zeros(szInput);
                idx = {':',':'};
                idx{dim} = ranges{iEdge};
                X(idx{:}) = randn(size(X(idx{:})));
                if usegpu
                    X = gpuArray(X);
                end
                s = cell(1,nLevels+1);
                [s{:}] = analysisnet.predict(dlarray(X,'SSCB'));
                for iCoef = 1:nLevels+1
                    coefs = gather(extractdata(s{iCoef}));
                    idx = {':',':',':',':'};
                    if strcmp(edges{iEdge},'first')
                        idx{dim} = 1;
                    else
                        idx{dim} = size(coefs,dim);
                    end
                    testCase.verifyEqual(max(abs(coefs(idx{:})),[],'all'),0,...
                        'AbsTol',testCase.TOL,...
                        sprintf('Wrap-around to the %s edge in output %d.',...
                        edges{iEdge},iCoef));
                end
            end
        end

        function testBackwardMultiLevel(testCase, usegpu, boundaryOperation)
            % The gradients with respect to the angles of all the levels
            % are available

            testCase.assumeFp64Gpu_(usegpu);
            nLevels = 2;
            analysisnet = testCase.createNetworks_(...
                [2 2],[3 3],boundaryOperation,boundaryOperation,usegpu,...
                'InputSize',[16 16],'NumberOfLevels',nLevels);
            X = randn(16,16);
            if usegpu
                X = gpuArray(X);
            end
            X = dlarray(X,'SSCB');

            % Evaluation
            grads = dlfeval(@testCase.gradients_,analysisnet,X);
            isAngles = grads.Parameter == "Angles";
            testCase.verifyEqual(...
                cellfun(@(x) size(x,2),grads.Value(isAngles)),...
                cellfun(@(x) size(x,2),analysisnet.Learnables.Value(isAngles)));
            for iLv = 1:nLevels
                isLv = isAngles & startsWith(grads.Layer,"Lv"+iLv+"_");
                testCase.verifyTrue(any(cellfun(...
                    @(x) gather(any(extractdata(x) ~= 0,'all')),grads.Value(isLv))),...
                    sprintf('No gradient in level %d.',iLv));
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
            lgraph = fcn_createlsunlgraph2d([],...
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
            idx = {':',':'};
            idx{dim} = 1:edgeWidth;
            X(idx{:}) = randn(size(X(idx{:})));
            fwd = testCase.edgeCoefMax_(analysisnet,X,dim,'last',usegpu);
            %
            X = zeros(szInput);
            idx = {':',':'};
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

        function grads = gradients_(analysisnet,X)
            % Gradients of the l1-norm of all the outputs
            s = cell(1,numel(analysisnet.OutputNames));
            [s{:}] = analysisnet.forward(X);
            loss = 0;
            for iCoef = 1:numel(s)
                loss = loss + sum(abs(s{iCoef}),'all');
            end
            grads = dlgradient(loss,analysisnet.Learnables);
        end

        function [Y,X] = reconstruct_(analysisnet,synthesisnet,szInput,usegpu)
            % Analysis followed by synthesis of a random input in double
            X = randn(szInput);
            if usegpu
                X = gpuArray(X);
            end
            s = cell(1,2);
            [s{1:2}] = analysisnet.predict(dlarray(X,'SSCB'));
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
            [s{1:2}] = analysisnet.predict(dlarray(X,'SSCB'));
            value = 0;
            for iCoef = 1:2
                coefs = gather(extractdata(s{iCoef}));
                idx = {':',':',':',':'};
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
                isa(x,'tansacnet.lsun.lsunIntermediateRotation2dLayer'),...
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
