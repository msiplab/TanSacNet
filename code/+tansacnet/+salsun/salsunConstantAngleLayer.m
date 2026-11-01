classdef salsunConstantAngleLayer < nnet.layer.Layer
    %SALSUNCONSTANTANGLELAYER
    %
    %   Output layer of the constant angle estimator: learnable angles per
    %   block that do not depend on the input. With it, each rotation of
    %   SA-LSUN has fixed angles per block as in LSUN. The input only gives
    %   the number of samples, so the layer does not depend on the number
    %   of spatial dimensions.
    %
    %   Input  'in'  : p x (block dimensions) x nSamples, with
    %                  prod(block dimensions) = prod(NumberOfBlocks)
    %   Output 'out' : (NumberOfAngles+NumberOfZeroPadAngles)
    %                  x prod(NumberOfBlocks) x nSamples, the first
    %                  NumberOfZeroPadAngles angles being zero
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
        NumberOfBlocks        % [nRows nCols] (or more dimensions)
        NumberOfAngles        % # learnable angles per block
        NumberOfZeroPadAngles % # fixed-zero angles prepended (0 = none)
    end

    properties (Learnable)
        Angles
    end

    methods
        function layer = salsunConstantAngleLayer(varargin)
            p = inputParser;
            addParameter(p,'Name','')
            addParameter(p,'NumberOfBlocks',[])
            addParameter(p,'NumberOfAngles',[])
            addParameter(p,'NumberOfZeroPadAngles',0)
            parse(p,varargin{:})

            layer.Name = p.Results.Name;
            layer.NumberOfBlocks = p.Results.NumberOfBlocks;
            layer.NumberOfAngles = p.Results.NumberOfAngles;
            layer.NumberOfZeroPadAngles = p.Results.NumberOfZeroPadAngles;
            % Zero angles, as the output layer of the default estimator
            layer.Angles = zeros(layer.NumberOfAngles,prod(layer.NumberOfBlocks));
            layer.Description = "SA-LSUN constant angles " ...
                + "#angles = " + (layer.NumberOfAngles+layer.NumberOfZeroPadAngles) ...
                + " (learnable " + layer.NumberOfAngles ...
                + ", zero-padded " + layer.NumberOfZeroPadAngles + ")";
        end

        function Theta = predict(layer, X)
            nBlocks = prod(layer.NumberOfBlocks);
            nSamples = numel(X)/(size(X,1)*nBlocks);
            if nSamples ~= fix(nSamples)
                error('SaLsunLayer:InvalidNumberOfBlocks',...
                    'The input does not have %d blocks per sample.',nBlocks)
            end
            angles = repmat(cast(layer.Angles,'like',X),1,1,nSamples);
            if layer.NumberOfZeroPadAngles > 0
                padZeros = zeros(layer.NumberOfZeroPadAngles,nBlocks,nSamples,'like',angles);
                Theta = cat(1,padZeros,angles);
            else
                Theta = angles;
            end
        end
    end

end
