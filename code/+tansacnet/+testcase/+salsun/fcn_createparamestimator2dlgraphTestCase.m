classdef fcn_createparamestimator2dlgraphTestCase < matlab.unittest.TestCase
    %FCN_CREATEPARAMESTIMATOR2DLGRAPHTESTCASE
    %
    %   Input (unconnected):
    %     [Prefix 'Extract'] : p x nRows x nCols x nSamples -- the block
    %                          state to be watched by the estimator (e.g.
    %                          the output of the input-layer orthonormal
    %                          transform).
    %   Output (unconnected):
    %     [Prefix 'Theta']   : nAngles x (nRows*nCols) x nSamples, to be
    %                          connected to the 'theta' input of the
    %                          corresponding rotation layer.
    %
    %
    % Requirements: MATLAB R2022a
    %
    % Copyright (c) 2026, Motoyasu SUZUKI and Shogo MURAMATSU
    %
    % All rights reserved.
    %
    % Contact address: Shogo MURAMATSU,
    %                Faculty of Engineering, Niigata University,
    %                8050 2-no-cho Ikarashi, Nishi-ku,
    %                Niigata, 950-2181, JAPAN
    %
    % http://msiplab.eng.niigata-u.ac.jp/

    methods (Test)

        function testConstructionIntermediate(testCase)
            import tansacnet.salsun.*
            nChs = 8; % p
            lgraph = fcn_createparamestimator2dlgraph([],...
                'Prefix','Est_',...
                'TargetType','Intermediate',...
                'NumberOfChannels',nChs,...
                'NumberOfNeighborBlocks',[3 3],...
                'NumberOfResidualBlocks',2,...
                'Width',2);

            outLayer = lgraph.Layers(strcmp({lgraph.Layers.Name},'Est_Theta'));
            testCase.verifyEqual(outLayer.NumberOfAngles,(nChs-2)*nChs/8);
            testCase.verifyEqual(outLayer.NumberOfZeroPadAngles,0);

            extractLayer = lgraph.Layers(strcmp({lgraph.Layers.Name},'Est_Ext'));
            testCase.verifyEqual(extractLayer.Channels,nChs/2+1:nChs);

            % 2 residual blocks + extract + standardize + output = 5
            testCase.verifyEqual(numel(lgraph.Layers),5);
        end

        function testConstructionInitialWithNoDcLeakage(testCase)
            import tansacnet.salsun.*
            nChs = 8;
            lgraph = fcn_createparamestimator2dlgraph([],...
                'Prefix','Est_',...
                'TargetType','Initial',...
                'NumberOfChannels',nChs,...
                'NumberOfNeighborBlocks',[3 3],...
                'NumberOfResidualBlocks',3,...
                'Width',2,...
                'NoDcLeakage',true);

            outLayer = lgraph.Layers(strcmp({lgraph.Layers.Name},'Est_Theta'));
            nAnglesFull = (nChs-2)*nChs/4;
            testCase.verifyEqual(outLayer.NumberOfZeroPadAngles,nChs/2-1);
            testCase.verifyEqual(outLayer.NumberOfAngles,nAnglesFull-(nChs/2-1));

            extractLayer = lgraph.Layers(strcmp({lgraph.Layers.Name},'Est_Ext'));
            testCase.verifyEqual(extractLayer.Channels,2:nChs);
        end

        function testConstructionInitialWithoutNoDcLeakage(testCase)
            import tansacnet.salsun.*
            nChs = 8;
            lgraph = fcn_createparamestimator2dlgraph([],...
                'Prefix','Est_',...
                'TargetType','Initial',...
                'NumberOfChannels',nChs,...
                'NoDcLeakage',false);

            outLayer = lgraph.Layers(strcmp({lgraph.Layers.Name},'Est_Theta'));
            nAnglesFull = (nChs-2)*nChs/4;
            testCase.verifyEqual(outLayer.NumberOfZeroPadAngles,0);
            testCase.verifyEqual(outLayer.NumberOfAngles,nAnglesFull);
        end

        function testInvalidNumberOfChannels(testCase)
            import tansacnet.salsun.*
            testCase.verifyError(...
                @() fcn_createparamestimator2dlgraph([],...
                    'Prefix','Est_','TargetType','Intermediate',...
                    'NumberOfChannels',7), ...
                'SaLsunLayer:InvalidNumberOfChannels');
        end

        function testInvalidTargetType(testCase)
            import tansacnet.salsun.*
            testCase.verifyError(...
                @() fcn_createparamestimator2dlgraph([],...
                    'Prefix','Est_','TargetType','Final',...
                    'NumberOfChannels',8), ...
                '');
        end

        function testNoNameCollisionWithMultiplePrefixes(testCase)
            import tansacnet.salsun.*
            nChs = 8;
            lgraph = fcn_createparamestimator2dlgraph([],...
                'Prefix','A_','TargetType','Intermediate',...
                'NumberOfChannels',nChs,'NumberOfNeighborBlocks',[3 3],...
                'NumberOfResidualBlocks',2,'Width',2);
            lgraph = fcn_createparamestimator2dlgraph(lgraph,...
                'Prefix','B_','TargetType','Intermediate',...
                'NumberOfChannels',nChs,'NumberOfNeighborBlocks',[3 3],...
                'NumberOfResidualBlocks',2,'Width',2);

            layerNames = {lgraph.Layers.Name};
            testCase.verifyEqual(numel(layerNames),numel(unique(layerNames)));
            testCase.verifyTrue(any(strcmp(layerNames,'A_Theta')));
            testCase.verifyTrue(any(strcmp(layerNames,'B_Theta')));
            % 2 residual blocks + extract + standardize + output = 5,
            % times 2 prefixes
            testCase.verifyEqual(numel(lgraph.Layers),10);
        end

        function testIntegrationWithInitialRotationLayer(testCase)
            import tansacnet.lsun.*
            import tansacnet.salsun.*

            stride = [2 2];
            nChs = prod(stride);
            nrows = 3;
            ncols = 3;
            nSamples = 2;
            imgRows = nrows*stride(1);
            imgCols = ncols*stride(2);

            lgraph = layerGraph();
            lgraph = lgraph.addLayers(imageInputLayer([imgRows imgCols 1],...
                'Name','ImgIn','Normalization','none'));
            lgraph = lgraph.addLayers(lsunBlockDct2dLayer('Name','E0',...
                'Stride',stride,'NumberOfComponents',1));
            lgraph = lgraph.connectLayers('ImgIn','E0');
            lgraph = lgraph.addLayers(salsunInitialRotation2dLayer(...
                'Stride',stride,'NumberOfBlocks',[nrows ncols],'Name','V0'));
            lgraph = fcn_createparamestimator2dlgraph(lgraph,...
                'Prefix','V0_',...
                'TargetType','Initial',...
                'NumberOfChannels',nChs,...
                'NumberOfNeighborBlocks',[3 3],...
                'NumberOfResidualBlocks',2,...
                'Width',2,...
                'NoDcLeakage',false);
            lgraph = lgraph.connectLayers('E0','V0/x');
            lgraph = lgraph.connectLayers('E0','V0_Ext');
            lgraph = lgraph.connectLayers('V0_Theta','V0/theta');

            dlImg = dlarray(randn(imgRows,imgCols,1,nSamples),'SSCB');
            net = dlnetwork(lgraph,dlImg);

            % Forward pass shape check
            Y = predict(net,dlImg);
            testCase.verifySize(Y,[nChs nrows ncols nSamples]);

            % At initialization, every estimator output layer predicts
            % all-zero angles (Wo/Bo are zero-initialized), so the
            % rotation layer must act as if driven by Theta == 0.
            dctLayer = lsunBlockDct2dLayer('Name','E0',...
                'Stride',stride,'NumberOfComponents',1);
            Xdct = dctLayer.predict(extractdata(dlImg));
            nAnglesFull = (nChs-2)*nChs/4;
            expctdY = predict(net.Layers(strcmp({net.Layers.Name},'V0')),...
                Xdct,...
                zeros(nAnglesFull,nrows*ncols,nSamples));
            % dlnetwork/predict casts activations to single precision
            % internally, so allow for single-precision rounding rather
            % than exact double-precision equality.
            testCase.verifyEqual(extractdata(Y),expctdY,'AbsTol',1e-6);

            % Gradient check: confirm every Learnable parameter of the
            % estimator receives a finite gradient, and that the
            % rotation layer itself contributes no Learnables.
            testCase.verifyTrue(all(startsWith(net.Learnables.Layer,'V0_')));

            [~,grads] = dlfeval(@localModelLoss,net,dlImg);
            for i = 1:height(grads)
                g = grads.Value{i};
                testCase.verifyTrue(all(isfinite(extractdata(g)),'all'),...
                    "Non-finite gradient for " + grads.Layer(i) + "/" + grads.Parameter(i));
            end
        end

        function testIntegrationWithIntermediateRotationLayer(testCase)
            import tansacnet.lsun.*
            import tansacnet.salsun.*

            stride = [2 2];
            nChs = prod(stride);
            nrows = 3;
            ncols = 3;
            nSamples = 2;
            imgRows = nrows*stride(1);
            imgCols = ncols*stride(2);

            lgraph = layerGraph();
            lgraph = lgraph.addLayers(imageInputLayer([imgRows imgCols 1],...
                'Name','ImgIn','Normalization','none'));
            lgraph = lgraph.addLayers(lsunBlockDct2dLayer('Name','E0',...
                'Stride',stride,'NumberOfComponents',1));
            lgraph = lgraph.connectLayers('ImgIn','E0');
            lgraph = lgraph.addLayers(salsunIntermediateRotation2dLayer(...
                'Stride',stride,'NumberOfBlocks',[nrows ncols],...
                'Name','Vn','Mode','Analysis'));
            lgraph = fcn_createparamestimator2dlgraph(lgraph,...
                'Prefix','Vn_',...
                'TargetType','Intermediate',...
                'NumberOfChannels',nChs,...
                'NumberOfNeighborBlocks',[3 3],...
                'NumberOfResidualBlocks',2,...
                'Width',2);
            lgraph = lgraph.connectLayers('E0','Vn/x');
            lgraph = lgraph.connectLayers('E0','Vn_Ext');
            lgraph = lgraph.connectLayers('Vn_Theta','Vn/theta');

            dlImg = dlarray(randn(imgRows,imgCols,1,nSamples),'SSCB');
            net = dlnetwork(lgraph,dlImg);

            % Forward pass shape check
            Y = predict(net,dlImg);
            testCase.verifySize(Y,[nChs nrows ncols nSamples]);

            % At initialization, every estimator output layer predicts
            % all-zero angles (Wo/Bo are zero-initialized), so the
            % rotation layer must act as if driven by Theta == 0.
            dctLayer = lsunBlockDct2dLayer('Name','E0',...
                'Stride',stride,'NumberOfComponents',1);
            Xdct = dctLayer.predict(extractdata(dlImg));
            expctdY = predict(net.Layers(strcmp({net.Layers.Name},'Vn')),...
                Xdct,...
                zeros((nChs-2)*nChs/8,nrows*ncols,nSamples));
            % dlnetwork/predict casts activations to single precision
            % internally, so allow for single-precision rounding rather
            % than exact double-precision equality.
            testCase.verifyEqual(extractdata(Y),expctdY,'AbsTol',1e-6);

            % Gradient check: confirm every Learnable parameter of the
            % estimator receives a finite gradient, and that the
            % rotation layer itself contributes no Learnables.
            testCase.verifyTrue(all(startsWith(net.Learnables.Layer,'Vn_')));

            [~,grads] = dlfeval(@localModelLoss,net,dlImg);
            for i = 1:height(grads)
                g = grads.Value{i};
                testCase.verifyTrue(all(isfinite(extractdata(g)),'all'),...
                    "Non-finite gradient for " + grads.Layer(i) + "/" + grads.Parameter(i));
            end
        end

    end

end

function [loss,grads] = localModelLoss(net,dlX)
Y = forward(net,dlX);
loss = sum(Y.^2,'all');
grads = dlgradient(loss,net.Learnables);
end
