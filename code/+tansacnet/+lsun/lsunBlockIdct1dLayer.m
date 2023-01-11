classdef lsunBlockIdct1dLayer < nnet.layer.Layer %#codegen
    %NSOLTBLOCKIDCT1DLAYER
    %
    %   コンポーネント別に入力:
    %      nDecs x nSamples x nBlks 
    %
    %   ベクトル配列をブロック配列にして出力:
    %      nComponents x nSamples x (Stride(1)xnBlks) 
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
            layer.NumInputs = p.Results.NumberOfComponents;
            layer.NumOutputs = 1;
            
            dec = layer.Stride;
                        
            C_ = dctmtx(dec);
            C_ = [ C_(1:2:end,:) ; C_(2:2:end,:) ];
            %
            Ce = C_(1:ceil(dec/2),:);
            Co = C_(ceil(dec/2)+1:end,:);
            layer.C = [Ce; Co];
            
        end
        
        function Z = predict(layer, varargin)
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
            nComponents = layer.NumInputs;
            decFactor = layer.Stride;
            C_T = layer.C.';
            %
            X = varargin{1};
            nSamples = size(X,2);
            nBlks = size(X,3);
            seqlen = decFactor*nBlks;
            Z = zeros(nComponents,nSamples,seqlen,'like',X);
            %
            for iComponent = 1:nComponents
                X = varargin{iComponent};
                if isgpuarray(X)
                    arrayY = pagefun(@mtimes,C_T,X);
                else
                    arrayY = C_T*reshape(X,decFactor,[]);
                end
                Z(iComponent,:,:) = reshape(...
                    permute(reshape(arrayY,decFactor,nSamples,[]),[2 1 3]),...
                    1,nSamples,[]);
            end
            if isdlarray(X)
                Z = dlarray(Z,'CTB');
            end
            
        end
        
        function varargout = backward(layer,varargin)
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
            
            nComponents = layer.NumInputs;
            dLdZ = varargin{layer.NumInputs+layer.NumOutputs+1};
            varargout = cell(1,nComponents);
            
            % Layer forward function for prediction goes here.
            decFactor = layer.Stride;
            %
            C_ = layer.C;
            %
            nSamples = size(dLdZ,2);            
            seqlen = size(dLdZ,3);
            nBlks = seqlen/decFactor;
            
            for iComponent = 1:nComponents

                arrayX = permute(reshape(dLdZ(iComponent,:,:),...
                    nSamples,decFactor,nBlks),[2 1 3]);

                if isgpuarray(dLdZ)
                    varargout{iComponent} = ...
                        pagefun(@mtimes,C_,arrayX);
                else
                    varargout{iComponent} = reshape(...
                        C_*reshape(arrayX,decFactor,[]),...
                        decFactor,nSamples,nBlks);
                end
            end
        end
    end
    
end

