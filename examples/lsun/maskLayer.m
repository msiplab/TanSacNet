classdef maskLayer < nnet.layer.Layer
    % MASKLAYER
    %
    % Copyright (c) Shogo MURAMATSU, 2022
    % All rights reserved.
    %
    
    properties
        NumberOfChannels
        Mask
    end
        
    methods
        function layer = maskLayer(varargin) 
            p = inputParser;
            addParameter(p,'Name','')
            addParameter(p,'NumberOfChannels',1)
            addParameter(p,'Mask',0)
            parse(p,varargin{:})
            
            % Set layer name.
            layer.Name = p.Results.Name;
            layer.NumberOfChannels = p.Results.NumberOfChannels;
            mask_ = p.Results.Mask;
            layer.Mask = permute(mask_(:).*ones(layer.NumberOfChannels,1),[2 3 1]);

            % Set layer description.
            layer.Description = "MASK for " + layer.NumberOfChannels + " channels";
        end
        
        function Z = predict(layer, X)
            Z = layer.Mask.*X;
        end
    end
end