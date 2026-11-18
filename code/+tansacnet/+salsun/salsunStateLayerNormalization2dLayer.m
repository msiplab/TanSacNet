classdef salsunStateLayerNormalization2dLayer < nnet.layer.Layer
    %SALSUNSTATELAYERNORMALIZATION2DLAYER
    %
    %   Input  'in'  : nFeat x nRows x nCols x nSamples
    %   Output 'out' : nFeat x nRows x nCols x nSamples
    %
    %   Layer normalization of the local state of every block, in place of
    %   the state standardization: the features of a block are normalized
    %   with their mean and variance over the features (no statistics of
    %   the image), then scaled and shifted per feature by the learnable
    %   Gamma and Beta:
    %
    %     Z = Gamma .* (X - mean(X,1)) ./ sqrt(var(X,1,1) + Epsilon) + Beta
    %
    %   Unlike the LayerNorm of a residual block, its output replaces the
    %   state, so that the skip paths and the output layer see the
    %   normalized state too.
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

    properties
        Epsilon = 1e-8
        Scope = 'layer'
    end

    properties (Learnable)
        Gamma
        Beta
    end

    methods
        function layer = salsunStateLayerNormalization2dLayer(varargin)
            p = inputParser;
            addParameter(p,'Name','')
            addParameter(p,'InputSize',[])
            addParameter(p,'Epsilon',1e-8)
            parse(p,varargin{:})

            nFeat = p.Results.InputSize;
            layer.Name = p.Results.Name;
            layer.Epsilon = p.Results.Epsilon;
            layer.Gamma = ones(nFeat,1);
            layer.Beta = zeros(nFeat,1);
            layer.Description = "SA-LSUN state layer normalization (nFeat = " + nFeat + ")";
        end

        function Z = predict(layer, X)
            mu = mean(X,1);
            v = mean((X-mu).^2,1);
            Z = layer.Gamma.*((X-mu)./sqrt(v+layer.Epsilon)) + layer.Beta;
        end
    end

end
