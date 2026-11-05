classdef mask1dLayer < nnet.layer.Layer
    % MASKLAYER
    %
    % Copyright (c) Shogo MURAMATSU, 2023
    % All rights reserved.
    %
    
    properties
        NumberOfChannels
        Mask
    end
        
    methods
        function layer = mask1dLayer(varargin) 
            p = inputParser;
            addParameter(p,'Name','')
            addParameter(p,'NumberOfChannels',1)
            addParameter(p,'Mask',0)
            parse(p,varargin{:})
            
            % Set layer name.
            layer.Name = p.Results.Name;
            layer.NumberOfChannels = p.Results.NumberOfChannels;
            mask_ = p.Results.Mask;
            layer.Mask = mask_(:);

            % Set layer description.
            layer.Description = "MASK for " + layer.NumberOfChannels + " channels";
        end
        
        function Z = predict(layer, X)
            Z = layer.Mask.*X;
        end
    end
end