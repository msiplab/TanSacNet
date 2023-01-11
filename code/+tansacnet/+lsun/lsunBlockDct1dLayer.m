classdef lsunBlockDct1dLayer < nnet.layer.Layer %#codegen
    %NSOLTBLOCKDCT1DLAYER
    %
    %   ベクトル配列をブロック配列を入力:
    %      nComponents x nSamples x (Stride(1)xnCols) 
    %
    %   コンポーネント別に出力(nComponents):
    %      nDecs x nSamples x nBlks 
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
        function layer = lsunBlockDct1dLayer(varargin)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            %import tansacnet.utility.Direction
            p = inputParser;
            addParameter(p,'Stride',[])
            addParameter(p,'Name','')
            addParameter(p,'NumberOfComponents',1);
            parse(p,varargin{:})
            
            % Layer constructor function goes here.
            layer.Stride = p.Results.Stride;
            layer.Name = p.Results.Name;
            layer.Description = "Block DCT of size " ...
                + layer.Stride;
            layer.Type = '';
            layer.NumOutputs = p.Results.NumberOfComponents;
            layer.NumInputs = 1;
            
            dec = layer.Stride;
            %
            C_ = dctmtx(dec);
            C_ = [ C_(1:2:end,:) ; C_(2:2:end,:) ];
            %
            Ce = C_(1:ceil(dec/2),:);
            Co = C_(ceil(dec/2)+1:end,:);
            layer.C = [Ce; Co];
            
        end
        
        function varargout = predict(layer, X)
            % Forward input data through the layer at prediction time and
            % output the result.
            %
            % Inputs:
            %         layer       - Layer to forward propagate through
            %         X           - Input data
            % Outputs:
            %         Z1, ..., Zm - Outputs of layer forward function
            %import tansacnet.utility.Direction

            nComponents = layer.NumOutputs;
            varargout = cell(1,nComponents);
            
            if isdlarray(X)
                X = stripdims(X);
            end
            
            % Layer forward function for prediction goes here.
            decFactor = layer.Stride;
            %
            C_ = layer.C;
            %
            nSamples = size(X,2);            
            seqlen = size(X,3);
            nBlks = seqlen/decFactor;
            %
            for iComponent = 1:nComponents
               
                arrayX = permute(reshape(X(iComponent,:,:),...
                    nSamples,decFactor,nBlks),[2 1 3]);

                if isgpuarray(X)
                    varargout{iComponent} = ...
                        pagefun(@mtimes,C_,arrayX);
                else
                    varargout{iComponent} = reshape(...
                        C_*reshape(arrayX,decFactor,[]),...
                        decFactor,nSamples,nBlks);
                end
            end

        end
        
        function dLdX = backward(layer, varargin)
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
            %import tansacnet.utility.Direction
            
            nComponents = layer.NumOutputs;
            decFactor = layer.Stride;
            C_T = layer.C.';
            %
            dLdZ = varargin{layer.NumInputs+layer.NumOutputs+1};
            nSamples = size(dLdZ,2);
            nBlks = size(dLdZ,3);
            seqlen = decFactor*nBlks;
            dLdX = zeros(nComponents,nSamples,seqlen,'like',dLdZ);
            %
            for iComponent = 1:nComponents
                dLdZ = varargin{layer.NumInputs+layer.NumOutputs+iComponent};
                if isgpuarray(dLdZ)
                    arrayX = pagefun(@mtimes,C_T,dLdZ);
                else
                    arrayX = C_T*reshape(dLdZ,decFactor,[]);
                end
                dLdX(iComponent,:,:) = reshape(...
                    permute(reshape(arrayX,decFactor,nSamples,[]),[2 1 3]),...
                    1,nSamples,[]);
            end
        end
    end
    
end

