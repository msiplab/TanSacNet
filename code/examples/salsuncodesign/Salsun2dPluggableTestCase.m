classdef Salsun2dPluggableTestCase < matlab.unittest.TestCase
    %SALSUN2DPLUGGABLETESTCASE Test case for plug-in estimators and termination
    %
    % Checks salsun2d_extract_params and salsun2d_infer with the boundary
    % operation 'Termination' of fcn_createsalsunlgraph2d (as in the NSOLT
    % of SaivDr), with the estimator of constant angles
    % (fcn_createconstestimatorlgraph) and with a plug-in estimator given
    % by EstimatorExtractor: the reference matches predict of the
    % dlnetwork, reconstructs its input, and with 'Termination' no basis
    % function wraps around the boundary. Also checks that the functions
    % that support only circular boundaries and the default estimator
    % refuse such parameters.
    %
    % Requirements: MATLAB R2026b, Deep Learning Toolbox
    %
    % Copyright (c) 2026, Shogo MURAMATSU
    %
    % All rights reserved.
    %
    % Contact address: Shogo MURAMATSU,
    %    Faculty of Engineering, Niigata University,
    %    8050 2-no-cho Ikarashi, Nishi-ku,
    %    Niigata, 950-2181, JAPAN
    %
    % http://msiplab.eng.niigata-u.ac.jp/
    %

    properties (TestParameter)
        inputSize = {[32 32],[24 40]};
        boundary = {'Circular','Termination'};
    end

    methods (TestClassSetup)
        function addPaths(testCase)
            import matlab.unittest.fixtures.PathFixture
            here = fileparts(mfilename('fullpath'));
            testCase.applyFixture(PathFixture(fullfile(here,'..','..')));      % +tansacnet
            testCase.applyFixture(PathFixture(fullfile(here,'..','salsun')));  % maskLayer
        end
    end

    methods (Test)

        function testTerminationParams(testCase)

            rng(1)
            net = salsun2d_create_test_network([32 32],[],BoundaryOperation='Termination');

            % Actual values
            params = salsun2d_extract_params(net);

            % Evaluation: the stages with the target 'Difference' are
            % terminated in the direction of their atom extension, and
            % the estimators take zeros beyond the boundary
            testCase.verifyEqual(params.BoundaryOperation,'Termination');
            testCase.verifyEqual({params.Stages.Termination},{'Right','','Down',''});
            testCase.verifyEqual(params.V0.Estimator.Padding,'zero');
            testCase.verifyEqual(unique(arrayfun(@(s) string(s.Estimator.Padding),params.Stages)),"zero");
        end

        function testCircularParams(testCase)

            rng(1)
            net = salsun2d_create_test_network([32 32],[]);

            % Actual values
            params = salsun2d_extract_params(net);

            % Evaluation
            testCase.verifyEqual(params.BoundaryOperation,'Circular');
            testCase.verifyEqual({params.Stages.Termination},{'','','',''});
            testCase.verifyEqual(params.V0.Estimator.Padding,'circular');
        end

        function testTerminationMatchesDlnetwork(testCase,inputSize)

            rng(2)
            net = salsun2d_create_test_network(inputSize,[],BoundaryOperation='Termination');
            x = rand(inputSize,'single');

            % Expected values
            yExpctd = extractdata(net.predict(dlarray(x,'SSCB')));

            % Actual values
            params = salsun2d_extract_params(net);
            [yActual,~,thetas] = salsun2d_infer(x,params);

            % Evaluation
            testCase.verifyEqual(yActual,x,'AbsTol',single(1e-4), ...
                'The unmasked network must reconstruct its input.');
            testCase.verifyEqual(yActual,yExpctd,'AbsTol',single(1e-4));
            % the angles of the terminated blocks are zero (Vh1: first
            % block column, Vv1: first block row)
            nBlk = inputSize/4;
            th = reshape(thetas{2},[],nBlk(1),nBlk(2));
            testCase.verifyEqual(nnz(th(:,:,1)),0);
            testCase.verifyGreaterThan(nnz(th(:,:,2)),0);
            th = reshape(thetas{4},[],nBlk(1),nBlk(2));
            testCase.verifyEqual(nnz(th(:,1,:)),0);
        end

        function testTerminationMatchesDlnetworkWithMask(testCase)

            rng(3)
            inputSize = [24 40];
            coefMask = reshape([ones(2,1); zeros(14,1)],2,[]).';
            coefMask = coefMask(:);
            net = salsun2d_create_test_network(inputSize,coefMask,BoundaryOperation='Termination');
            x = rand(inputSize,'single');

            % Expected values (see Salsun2dInferTestCase)
            yNet = extractdata(net.predict(dlarray(x,'SSCB')));
            params = salsun2d_extract_params(net);
            yDouble = salsun2d_infer(double(x),salsun2d_cast_params(params,'double'));

            % Actual values
            yActual = salsun2d_infer(x,params);

            % Evaluation
            errNet = max(abs(double(yNet(:)) - yDouble(:)));
            errActual = max(abs(double(yActual(:)) - yDouble(:)));
            testCase.verifyLessThanOrEqual(errActual,2*errNet + 1e-6, ...
                sprintf('Error from double: reference %g, dlnetwork %g',errActual,errNet));
            testCase.verifyEqual(yActual,yNet,'AbsTol',single(1e-2));
        end

        function testTerminationDoesNotWrap(testCase,boundary)

            % Frames much wider (taller) than the receptive field of the
            % adaptive estimators (layer normalization of the state: no
            % statistics of the frame), so that the pixels at the right
            % (bottom) boundary can reach the first block column (row)
            % only by wrapping around
            estOptions = {'StandardizationScope','layer','ResidualLayerNorm',false, ...
                'Activation','tanh','OutputMapping','asintanh'};
            rng(4)
            netW = salsun2d_create_test_network([16 256],[],BoundaryOperation=boundary, ...
                EstimatorOptions=estOptions);
            rng(4)
            netT = salsun2d_create_test_network([256 16],[],BoundaryOperation=boundary, ...
                EstimatorOptions=estOptions);
            paramsW = salsun2d_cast_params(salsun2d_extract_params(netW),'double');
            paramsT = salsun2d_cast_params(salsun2d_extract_params(netT),'double');
            x = rand(16,256);
            xp = x;
            xp(:,end-3:end) = xp(:,end-3:end) + rand(16,4);      % the last block column
            z = rand(256,16);
            zp = z;
            zp(end-3:end,:) = zp(end-3:end,:) + rand(4,16);      % the last block row

            % Actual values
            [~,c0] = salsun2d_infer(x,paramsW);
            [~,cp] = salsun2d_infer(xp,paramsW);
            [~,d0] = salsun2d_infer(z,paramsT);
            [~,dp] = salsun2d_infer(zp,paramsT);
            dCol = max(abs(cp(:,:,1) - c0(:,:,1)),[],'all');
            dRow = max(abs(dp(:,1,:) - d0(:,1,:)),[],'all');

            % Evaluation (in double precision: up to rounding, as the
            % coefficients that went around come back through sums and
            % differences)
            if strcmp(boundary,'Termination')
                testCase.verifyLessThan(dCol,1e-10,'The first block column depends on the last.');
                testCase.verifyLessThan(dRow,1e-10,'The first block row depends on the last.');
            else
                testCase.verifyGreaterThan(dCol,1e-6);
                testCase.verifyGreaterThan(dRow,1e-6);
            end
        end

        function testConstantEstimator(testCase,boundary)

            rng(5)
            inputSize = [24 40];
            net = salsun2d_create_test_network(inputSize,[],BoundaryOperation=boundary, ...
                EstimatorFunction=@tansacnet.salsun.fcn_createconstestimatorlgraph);
            x = rand(inputSize,'single');

            % Expected values
            yExpctd = extractdata(net.predict(dlarray(x,'SSCB')));

            % Actual values
            params = salsun2d_extract_params(net);
            yActual = salsun2d_infer(x,params);

            % Evaluation
            testCase.verifyEqual(params.V0.Estimator.Type,'constant');
            testCase.verifyEqual(yActual,x,'AbsTol',single(1e-4));
            testCase.verifyEqual(yActual,yExpctd,'AbsTol',single(1e-4));
        end

        function testPluginEstimator(testCase,boundary)

            % The constant estimator again, as a plug-in: its own
            % extractor and inference function
            rng(6)
            inputSize = [24 40];
            net = salsun2d_create_test_network(inputSize,[],BoundaryOperation=boundary, ...
                EstimatorFunction=@tansacnet.salsun.fcn_createconstestimatorlgraph);
            x = rand(inputSize,'single');
            extractor = @(getLayer,prefix) struct( ...
                'Function',@(Y,est,q) q(cast(est.Angles,'like',Y),'angles'), ...
                'Angles',[zeros(getLayer([prefix 'Theta']).NumberOfZeroPadAngles, ...
                size(getLayer([prefix 'Theta']).Angles,2),'single'); ...
                single(extractdata(dlarray(getLayer([prefix 'Theta']).Angles)))]);

            % Expected values
            yExpctd = salsun2d_infer(x,salsun2d_extract_params(net));

            % Actual values
            params = salsun2d_extract_params(net,EstimatorExtractor=extractor);
            [yActual,~,thetas] = salsun2d_infer(x,params);

            % Evaluation
            testCase.verifyTrue(isfield(params.V0.Estimator,'Function'));
            testCase.verifyEqual(yActual,yExpctd);
            testCase.verifyEqual(numel(thetas),5);
        end

        function testUnknownEstimatorNeedsExtractor(testCase)

            % An estimator that is neither the default nor the constant
            % one (zero angles from a function layer)
            rng(7)
            inputSize = [24 40];
            net = salsun2d_create_test_network(inputSize,[], ...
                EstimatorFunction=@salsun2dTestEstimator);

            % Evaluation
            testCase.verifyError(@() salsun2d_extract_params(net), ...
                'salsun2d_extract_params:estimator');
        end

        function testUnsupportedPathsRefuse(testCase)

            rng(8)
            net = salsun2d_create_test_network([32 32],[],BoundaryOperation='Termination');
            params = salsun2d_extract_params(net);
            netC = salsun2d_create_test_network([32 32],[], ...
                EstimatorFunction=@tansacnet.salsun.fcn_createconstestimatorlgraph);
            paramsC = salsun2d_extract_params(netC);

            % Evaluation
            testCase.verifyError(@() salsun2d_assert_plain(params,'test'),'salsun2d:unsupported');
            testCase.verifyError(@() salsun2d_assert_plain(paramsC,'test'),'salsun2d:unsupported');
            testCase.verifyError(@() salsun2d_pack_params(params),'salsun2d:unsupported');
            rng(8)
            paramsP = salsun2d_extract_params(salsun2d_create_test_network([32 32],[]));
            salsun2d_assert_plain(paramsP,'test');         % no error
        end

    end

end

function lgraph = salsun2dTestEstimator(lgraph,varargin)
% A plug-in estimator for the tests that salsun2d_extract_params does not
% know: zero angles from a function layer
p = inputParser;
p.KeepUnmatched = true;
addParameter(p,'Prefix','')
addParameter(p,'TargetType','Initial')
addParameter(p,'NumberOfChannels',[])
addParameter(p,'NoDcLeakage',false)
parse(p,varargin{:})
nA = tansacnet.salsun.fcn_salsunanglecounts(p.Results.TargetType, ...
    p.Results.NumberOfChannels,p.Results.NoDcLeakage);
lgraph = lgraph.addLayers([
    tansacnet.lsun.lsunIdentityLayer('Name',[p.Results.Prefix 'Ext'])
    functionLayer(@(X) zeros(nA,size(X,2)*size(X,3),size(X,4),'like',X), ...
        'Name',[p.Results.Prefix 'Theta'])
    ]);
end
