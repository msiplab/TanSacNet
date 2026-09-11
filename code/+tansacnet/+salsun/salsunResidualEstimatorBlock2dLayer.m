classdef salsunResidualEstimatorBlock2dLayer < nnet.layer.Layer
    %SALSUNRESIDUALESTIMATORBLOCK2DLAYER
    %
    %   Input  'in'  : nFeat x nRows x nCols x nSamples
    %   Output 'out' : nFeat x nRows x nCols x nSamples
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

    properties
        InputSize % nFeat
        Width     % expansion factor d (hidden = round(Width*InputSize))
    end

    properties (Learnable)
        Gamma
        Beta
        W1
        B1
        W2
        B2
    end

    methods
        function layer = salsunResidualEstimatorBlock2dLayer(varargin)
            p = inputParser;
            addParameter(p,'Name','')
            addParameter(p,'InputSize',[])
            addParameter(p,'Width',2)
            parse(p,varargin{:})

            layer.Name = p.Results.Name;
            layer.InputSize = p.Results.InputSize;
            layer.Width = p.Results.Width;

            nFeat = layer.InputSize;
            hidden = round(layer.Width*nFeat);

            layer.Gamma = ones(nFeat,1);
            layer.Beta = zeros(nFeat,1);
            layer.W1 = sqrt(2/nFeat)*randn(hidden,nFeat);
            layer.B1 = zeros(hidden,1);
            layer.W2 = zeros(nFeat,hidden);
            layer.B2 = zeros(nFeat,1);

            layer.Description = "SA-LSUN residual estimator block " ...
                + "(nFeat,hidden) = (" + nFeat + "," + hidden + ")";
        end

        function Z = predict(layer, X)
            nFeat = size(X,1);
            nRows = size(X,2);
            nCols = size(X,3);
            nSamples = size(X,4);

            % Layer normalization over the feature dimension
            mu = mean(X,1);
            v = mean((X-mu).^2,1);
            xhat = (X-mu)./sqrt(v+1e-5);
            ln = layer.Gamma.*xhat + layer.Beta;

            % First fully-connected layer (expand) + GELU
            lnFlat = reshape(ln,nFeat,nRows*nCols*nSamples);
            z1 = layer.W1*lnFlat + layer.B1;
            a = 0.5.*z1.*(1+tanh(sqrt(2/pi).*(z1+0.044715.*z1.^3)));

            % Second fully-connected layer (contract) + residual add
            z2 = layer.W2*a + layer.B2;
            residual = reshape(z2,nFeat,nRows,nCols,nSamples);
            Z = X + residual;
        end

        %   Because layer normalization, the fully connected layers, and the
        %   GELU nonlinearity are all expressed with dlarray-differentiable
        %   operations, no custom backward method is defined; MATLAB's
        %   automatic differentiation supplies the gradients with respect to
        %   X and to every Learnable property.

    end

end
