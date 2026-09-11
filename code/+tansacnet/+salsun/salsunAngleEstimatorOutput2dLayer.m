classdef salsunAngleEstimatorOutput2dLayer < nnet.layer.Layer
    %SALSUNANGLEESTIMATOROUTPUT2DLAYER
    %
    %   Input  'in'    : nFeat x nRows x nCols x nSamples
    %   Output 'theta' : (NumberOfAngles+NumberOfZeroPadAngles) x
    %                    (nRows*nCols) x nSamples
    %
    % Reference:
    %  M. Suzuki and S. Muramatsu, "2-D State-Attentive Locally-Structured
    %  Unitary Networks for Tangent Space Learning," APSIPA ASC 2026.
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
        InputSize             % nFeat
        NumberOfAngles        % # angles predicted by Wo/Bo
        NumberOfZeroPadAngles % # fixed-zero angles prepended (0 = none)
    end

    properties (Learnable)
        Wo
        Bo
    end

    methods
        function layer = salsunAngleEstimatorOutput2dLayer(varargin)
            p = inputParser;
            addParameter(p,'Name','')
            addParameter(p,'InputSize',[])
            addParameter(p,'NumberOfAngles',[])
            addParameter(p,'NumberOfZeroPadAngles',0)
            parse(p,varargin{:})

            layer.Name = p.Results.Name;
            layer.InputSize = p.Results.InputSize;
            layer.NumberOfAngles = p.Results.NumberOfAngles;
            layer.NumberOfZeroPadAngles = p.Results.NumberOfZeroPadAngles;

            layer.Wo = zeros(layer.NumberOfAngles,layer.InputSize);
            layer.Bo = zeros(layer.NumberOfAngles,1);

            layer.Description = "SA-LSUN angle estimator output " ...
                + "#angles = " + (layer.NumberOfAngles+layer.NumberOfZeroPadAngles) ...
                + " (predicted " + layer.NumberOfAngles ...
                + ", zero-padded " + layer.NumberOfZeroPadAngles + ")";
        end

        function Theta = predict(layer, X)
            nFeat = size(X,1);
            nRows = size(X,2);
            nCols = size(X,3);
            nSamples = size(X,4);

            Xflat = reshape(X,nFeat,nRows*nCols*nSamples);
            angFlat = layer.Wo*Xflat + layer.Bo;
            angFlat = reshape(angFlat,layer.NumberOfAngles,nRows*nCols,nSamples);

            if layer.NumberOfZeroPadAngles > 0
                padZeros = zeros(layer.NumberOfZeroPadAngles,nRows*nCols,nSamples,'like',angFlat);
                Theta = cat(1,padZeros,angFlat);
            else
                Theta = angFlat;
            end
        end

    end

end
