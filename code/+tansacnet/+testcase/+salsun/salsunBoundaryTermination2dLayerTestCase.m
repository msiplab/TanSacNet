classdef salsunBoundaryTermination2dLayerTestCase < matlab.unittest.TestCase
    %SALSUNBOUNDARYTERMINATION2DLAYERTESTCASE
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

    properties (TestParameter)
        direction = {'Right','Left','Down','Up'};
    end

    methods (TestClassTeardown)
        function finalCheck(~)
            import tansacnet.salsun.*
            fprintf("\n --- Check layer for 2-D images (SA-LSUN) ---\n");
            layer = salsunBoundaryTermination2dLayer('Name','Trm',...
                'NumberOfBlocks',[3 4],'Direction','Right','Input','state');
            checkLayer(layer,[4 3 4],'ObservationDimension',4)
        end
    end

    methods (Test)

        function testAngles(testCase,direction)
            import tansacnet.salsun.*
            nBlocks = [3 4];
            nAngles = 6;
            nSamples = 2;
            Theta = randn(nAngles,prod(nBlocks),nSamples);
            layer = salsunBoundaryTermination2dLayer('Name','Trm',...
                'NumberOfBlocks',nBlocks,'Direction',direction);

            % Expected values
            T = reshape(Theta,nAngles,nBlocks(1),nBlocks(2),nSamples);
            T = zeroBoundary(T,direction);
            expctdZ = reshape(T,nAngles,prod(nBlocks),nSamples);

            % Actual values
            Z = layer.predict(Theta);

            % Evaluation
            testCase.verifyEqual(Z,expctdZ);
            testCase.verifyEqual(layer.Input,'angles');
        end

        function testState(testCase,direction)
            import tansacnet.salsun.*
            nBlocks = [3 4];
            nChs = 8;
            nSamples = 2;
            X = randn(nChs,nBlocks(1),nBlocks(2),nSamples);
            layer = salsunBoundaryTermination2dLayer('Name','TrmX',...
                'NumberOfBlocks',nBlocks,'Direction',direction,'Input','state');

            % Actual values
            Z = layer.predict(X);

            % Evaluation
            testCase.verifyEqual(Z,zeroBoundary(X,direction));
        end

    end

end

function X = zeroBoundary(X,direction)
% The blocks that receive the coefficients of the opposite boundary
switch direction
    case 'Right', X(:,:,1,:) = 0;
    case 'Left',  X(:,:,end,:) = 0;
    case 'Down',  X(:,1,:,:) = 0;
    case 'Up',    X(:,end,:,:) = 0;
end
end
