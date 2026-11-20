classdef lsunChannelSeparation1dLayer < nnet.layer.Layer %#codegen
    %LSUNCHANNELSEPARATION1DLAYER
    %
    %   １コンポーネント入力(nComponents=1のみサポート):
    %      nChsTotal x 1 x nBlks x nSamples
    %
    %   ２コンポーネント出力(nComponents=2のみサポート):
    %      (nChsTotal-1) x 1 x nBlks x nSamples
    %      1 x 1 x nBlks x nSamples
    %
    %   'ChannelDimension' が 3 のとき(2-D/3-D のレイヤーと同じ配置):
    %      1 x nBlks x (nChsTotal-1) x nSamples
    %      1 x nBlks x 1 x nSamples
    %
    % Requirements: MATLAB R2022b
    %
    % Copyright (c) 2023, Shogo MURAMATSU
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
        % Dimension of the channels of AC and DC, 1 (default) or 3
        ChannelDimension
    end
    
    methods
        function layer = lsunChannelSeparation1dLayer(varargin)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            p = inputParser;
            addParameter(p,'Name','')
            addParameter(p,'ChannelDimension',1)
            parse(p,varargin{:})

            % Dimension of the channels of AC and DC, 1 (default) or 3
            layer.ChannelDimension = p.Results.ChannelDimension;
            if ~isscalar(layer.ChannelDimension) ...
                    || ~any(layer.ChannelDimension == [1 3])
                throw(MException('LsunLayer:InvalidChannelDimension',...
                    'ChannelDimension should be either of 1 or 3'))
            end

            % Layer constructor function goes here.
            layer.Name = p.Results.Name;
            layer.Description =  "Channel separation";
            layer.Type = '';
            %layer.NumOutputs = 2;
            layer.OutputNames = { 'ac', 'dc' };            
        end
        
        function [Zac,Zdc] = predict(layer, X)
            % Forward input data through the layer at prediction time and
            % output the result.
            %
            % Inputs:
            %         layer       - Layer to forward propagate through
            %         X           - Input data (1 component)
            % Outputs:
            %         Z1, Z2      - Outputs of layer forward function
            %  
            
            % Layer forward function for prediction goes here.
            Zac = X(2:end,:,:,:);
            Zdc = X(1,:,:,:);
            if layer.ChannelDimension == 3
                Zac = permute(Zac,[2 3 1 4]);
                Zdc = permute(Zdc,[2 3 1 4]);
            end
            
            if isdlarray(X)
                Zac = dlarray(Zac,'SSCB');
                Zdc = dlarray(Zdc,'SSCB');
            end

        end
        
        function dLdX = backward(layer, ~, ~, ~, dLdZac,dLdZdc,~)
            % (Optional) Backward propagate the derivative of the loss  
            % function through the layer.
            %
            % Inputs:
            %         layer             - Layer to backward propagate through
            %         X1, ..., Xn       - Input data
            %         Z1, ..., Zm       - Outputs of layer forward function            
            %         dLdZ1, ..., dLdZm - Gradients propagated from the next layers
            %         memory            - Memory value from forward function
            % Outputs:
            %         dLdX1, ..., dLdXn - Derivatives of the loss with respect to the
            %                             inputs
            %         dLdW1, ..., dLdWk - Derivatives of the loss with respect to each
            %                             learnable parameter
            
            % Layer forward function for prediction goes here.
            if layer.ChannelDimension == 3
                dLdX = ipermute(cat(3,dLdZdc,dLdZac),[2 3 1 4]);
            else
                dLdX = cat(1,dLdZdc,dLdZac);
            end
        end
    end
    
end

