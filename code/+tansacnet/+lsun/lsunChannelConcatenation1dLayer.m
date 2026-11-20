classdef lsunChannelConcatenation1dLayer < nnet.layer.Layer %#codegen
    %LSUNCHANNELCONCATENATION1DLAYER
    %
    %   ２コンポーネント出力(nComponents=2のみサポート):
    %      1 x 1 x nBlks x nSamples 
    %      (nChsTotal-1) x 1 x nBlks x nSamples
    %
    %   'ChannelDimension' が 3 のとき(2-D/3-D のレイヤーと同じ配置):
    %      1 x nBlks x (nChsTotal-1) x nSamples
    %      1 x nBlks x 1 x nSamples
    %
    %   １コンポーネント入力(nComponents=1のみサポート):
    %      nChsTotal x 1 x nBlks x nSamples
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
        function layer = lsunChannelConcatenation1dLayer(varargin)
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
            layer.Description =  "Channel concatenation";
            layer.Type = '';
            %layer.NumInputs = 2;
            layer.InputNames = { 'ac', 'dc' };
            
        end
        
        function Z = predict(layer, Xac,Xdc)
            % Forward input data through the layer at prediction time and
            % output the result.
            %
            % Inputs:
            %         layer       - Layer to forward propagate through
            %         X1, X2      - Input data (2 components)
            % Outputs:
            %         Z           - Outputs of layer forward function
            %  
            
            if isdlarray(Xac) 
                Xac = stripdims(Xac);
            end                        
            
            if isdlarray(Xdc) 
                Xdc = stripdims(Xdc);
            end                                    
            
            % Layer forward function for prediction goes here.
            if layer.ChannelDimension == 3
                Z = permute(cat(3,Xdc,Xac),[3 1 2 4]);
            else
                Z = cat(1,Xdc,Xac);
            end
        end
        
        function [dLdXac,dLdXdc] = backward(layer,~, ~, ~, dLdZ, ~)
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
            dLdXac = dLdZ(2:end,:,:,:);
            dLdXdc = dLdZ(1,:,:,:);
            if layer.ChannelDimension == 3
                dLdXac = ipermute(dLdXac,[3 1 2 4]);
                dLdXdc = ipermute(dLdXdc,[3 1 2 4]);
            end
        end
    end
    
end

