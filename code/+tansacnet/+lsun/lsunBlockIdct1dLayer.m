classdef lsunBlockIdct1dLayer < nnet.layer.Layer %#codegen
    %NSOLTBLOCKIDCT1DLAYER
    %
    %   入力:
    %      Stride x nBlks x nSamples
    %
    %   ベクトル配列をブロック配列にして出力:
    %      1 x (Stride x nBlks) x 1 x nSamples
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
        % (Optional) Layer properties.
        Stride
        
        % Layer properties go here.
    end
    
    properties (Access = private)
        C
    end
    
    methods
        function layer = lsunBlockIdct1dLayer(varargin)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            %import tansacnet.utility.Direction

            p = inputParser;
            addParameter(p,'Stride',[])
            addParameter(p,'Name','')
            addParameter(p,'NumberOfComponents',1)
            parse(p,varargin{:})
            
            % Layer constructor function goes here.
            layer.Stride = p.Results.Stride;
            layer.Name = p.Results.Name;
            layer.Description = "Block IDCT of size " ...
                + layer.Stride;
            layer.Type = '';
            layer.NumInputs = 1; %p.Results.NumberOfComponents;
            layer.NumOutputs = 1;
            
            dec = layer.Stride;
                        
            C_ = dctmtx(dec);
            C_ = [ C_(1:2:end,:) ; C_(2:2:end,:) ];
            %
            Ce = C_(1:ceil(dec/2),:);
            Co = C_(ceil(dec/2)+1:end,:);
            layer.C = [Ce; Co];
            
        end
        
        function Z = predict(layer, X) %varargin)
            % Forward input data through the layer at prediction time and
            % output the result.
            %
            % Inputs:
            %         layer       - Layer to forward propagate through
            %         X1, ..., Xn - Input data
            % Outputs:
            %         Z1, ..., Zm - Outputs of layer forward function
            %import tansacnet.utility.Direction
            
            % Layer forward function for prediction goes here.
            %nComponents = layer.NumInputs;
            stride = layer.Stride;
            C_T = layer.C.';
            %
            %X = varargin{1};
            nSamples = size(X,4);
            nBlks = size(X,3);
            seqlen = stride*nBlks;
            %
            arrayY = C_T*reshape(X,stride,[]);
            Z = reshape(arrayY,1,seqlen,1,nSamples);
            if isdlarray(X)
                Z = dlarray(Z,"SSCB");
            end
            
        end
        
        function dLdX = backward(layer,varargin)
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
            %import tansacnet.utility.Direction
            
            %nComponents = layer.NumInputs;
            dLdZ = varargin{layer.NumInputs+layer.NumOutputs+1};
            %varargout = cell(1,nComponents);
            
            % Layer forward function for prediction goes here.
            stride = layer.Stride;
            %
            C_ = layer.C;
            %
            nSamples = size(dLdZ,4);            
            seqlen = size(dLdZ,2);
            nBlks = seqlen/stride;
            %
            dLdX = reshape(C_*reshape(dLdZ,stride,[]),...
                stride,1,nBlks,nSamples);
        end
    end
    
end

