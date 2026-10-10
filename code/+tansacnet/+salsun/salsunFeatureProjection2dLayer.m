classdef salsunFeatureProjection2dLayer < nnet.layer.Layer
    %SALSUNFEATUREPROJECTION2DLAYER
    %
    %   Linear projection of the standardized local state of each block
    %   to fewer features (bottleneck of the angle estimator):
    %   Z = Wp*X + Bp per block.
    %
    %   Input  'in'  : InputSize x nRows x nCols x nSamples
    %   Output 'out' : OutputSize x nRows x nCols x nSamples
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
        InputSize
        OutputSize
    end

    properties (Learnable)
        Wp
        Bp
    end

    methods
        function layer = salsunFeatureProjection2dLayer(varargin)
            p = inputParser;
            addParameter(p,'Name','')
            addParameter(p,'InputSize',[])
            addParameter(p,'OutputSize',[])
            parse(p,varargin{:})

            layer.Name = p.Results.Name;
            layer.InputSize = p.Results.InputSize;
            layer.OutputSize = p.Results.OutputSize;
            % Scale-preserving random initialization
            layer.Wp = sqrt(1/layer.InputSize)*randn(layer.OutputSize,layer.InputSize);
            layer.Bp = zeros(layer.OutputSize,1);
            layer.Description = "SA-LSUN feature projection " ...
                + "(nIn,nOut) = (" + layer.InputSize + "," + layer.OutputSize + ")";
        end

        function Z = predict(layer, X)
            nRows = size(X,2);
            nCols = size(X,3);
            nSamples = size(X,4);
            Xflat = reshape(X,size(X,1),nRows*nCols*nSamples);
            Z = reshape(layer.Wp*Xflat + layer.Bp,layer.OutputSize,nRows,nCols,nSamples);
        end
    end

end
