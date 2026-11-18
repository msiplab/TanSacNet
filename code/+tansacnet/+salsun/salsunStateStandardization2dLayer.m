classdef salsunStateStandardization2dLayer < nnet.layer.Layer
    %SALSUNSTATESTANDARDIZATION2DLAYER
    %
    %   Input  'in'  : nFeat x nRows x nCols x nSamples
    %   Output 'out' : nFeat x nRows x nCols x nSamples
    %
    %   Scope 'image' (default): every feature is standardized with its
    %   mean and standard deviation over all blocks of the image (sigma
    %   plus Epsilon). Scope 'neighborhood': the features are the
    %   NumberOfChannels channels of the neighbor blocks (channels inner,
    %   as salsunLocalStateExtraction2dLayer stacks them), and every
    %   channel of a block is standardized with the mean and variance of
    %   that channel over the neighbor blocks of the block (unbiased,
    %   sigma = sqrt(variance + Epsilon)), so that no statistics of the
    %   whole image are needed. Scope 'none': no standardization (identity).
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
        Scope = 'image'
        NumberOfChannels = []
    end

    methods
        function layer = salsunStateStandardization2dLayer(varargin)
            p = inputParser;
            addParameter(p,'Name','')
            addParameter(p,'Epsilon',1e-8)
            addParameter(p,'Scope','image')
            addParameter(p,'NumberOfChannels',[])
            parse(p,varargin{:})

            layer.Name = p.Results.Name;
            layer.Epsilon = p.Results.Epsilon;
            layer.Scope = validatestring(p.Results.Scope,{'image','neighborhood','none'});
            layer.NumberOfChannels = p.Results.NumberOfChannels;
            if strcmp(layer.Scope,'neighborhood') && isempty(layer.NumberOfChannels)
                error('SaLsunLayer:NumberOfChannels', ...
                    'Scope ''neighborhood'' requires NumberOfChannels.')
            end
            layer.Description = "SA-LSUN state standardization";
        end

        function Z = predict(layer, X)
            if strcmp(layer.Scope,'none')
                Z = X;
                return
            end
            if strcmp(layer.Scope,'neighborhood')
                sz = size(X,1:4);
                nCh = layer.NumberOfChannels;
                nNb = sz(1)/nCh;
                Xr = reshape(X,nCh,nNb,sz(2),sz(3),sz(4));
                mu = mean(Xr,2);
                v = sum((Xr-mu).^2,2)/(nNb-1);
                Z = reshape((Xr-mu)./sqrt(v + layer.Epsilon),sz);
                return
            end
            nBlks = size(X,2)*size(X,3);
            mu = mean(X,[2 3]);
            v = sum((X-mu).^2,[2 3])/(nBlks-1);
            sigma = sqrt(v) + layer.Epsilon;
            Z = (X - mu) ./ sigma;
        end

    end

end
