classdef salsunStateStandardization2dLayer < nnet.layer.Layer
    %SALSUNSTATESTANDARDIZATION2DLAYER
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
        Epsilon = 1e-8
    end

    methods
        function layer = salsunStateStandardization2dLayer(varargin)
            p = inputParser;
            addParameter(p,'Name','')
            addParameter(p,'Epsilon',1e-8)
            parse(p,varargin{:})

            layer.Name = p.Results.Name;
            layer.Epsilon = p.Results.Epsilon;
            layer.Description = "SA-LSUN state standardization";
        end

        function Z = predict(layer, X)
            nBlks = size(X,2)*size(X,3);
            mu = mean(X,[2 3]);
            v = sum((X-mu).^2,[2 3])/(nBlks-1);
            sigma = sqrt(v) + layer.Epsilon;
            Z = (X - mu) ./ sigma;
        end

    end

end
