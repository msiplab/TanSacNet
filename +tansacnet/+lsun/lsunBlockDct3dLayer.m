classdef lsunBlockDct3dLayer < nnet.layer.Layer %#codegen
    %NSOLTBLOCKDCT3DLAYER
    %
    %   ベクトル配列をブロック配列を入力:
    %      (Stride(1)xnRows) x (Stride(2)xnCols) x (Stride(3)xnLays)  x nComponents x nSamples
    %
    %   コンポーネント別に出力(nComponents):
    %      nDecs x nRows x nCols x nLays x nSamples
    %
    
    properties
        % (Optional) Layer properties.
        Stride
        
        % Layer properties go here.
    end
    
    properties (Access = private)
        Cvhd
    end
    
    methods
        function layer = lsunBlockDct3dLayer(varargin)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            %import tansacnet.utility.Direction
            Direction = tansacnet.utility.Direction;
            p = inputParser;
            addParameter(p,'Stride',[])
            addParameter(p,'Name','')
            addParameter(p,'NumberOfComponents',1);
            parse(p,varargin{:})
            
            % Layer constructor function goes here.
            layer.Stride = p.Results.Stride;
            layer.Name = p.Results.Name;
            layer.Description = "Block DCT of size " ...
                + layer.Stride(Direction.VERTICAL) + "x" ...
                + layer.Stride(Direction.HORIZONTAL) + "x" ...
                + layer.Stride(Direction.DEPTH);
            layer.Type = '';
            layer.NumOutputs = p.Results.NumberOfComponents;
            layer.NumInputs = 1;
            
            decV = layer.Stride(Direction.VERTICAL);
            decH = layer.Stride(Direction.HORIZONTAL);
            decD = layer.Stride(Direction.DEPTH);
            %
            Cv_ = dctmtx(decV);
            Ch_ = dctmtx(decH);
            Cd_ = dctmtx(decD);
            Cv_ = [ Cv_(1:2:end,:) ; Cv_(2:2:end,:) ];
            Ch_ = [ Ch_(1:2:end,:) ; Ch_(2:2:end,:) ];
            Cd_ = [ Cd_(1:2:end,:) ; Cd_(2:2:end,:) ];
            %
            Cve = Cv_(1:ceil(decV/2),:);
            Cvo = Cv_(ceil(decV/2)+1:end,:);
            Che = Ch_(1:ceil(decH/2),:);
            Cho = Ch_(ceil(decH/2)+1:end,:);
            Cde = Cd_(1:ceil(decD/2),:);
            Cdo = Cd_(ceil(decD/2)+1:end,:);
            %
            Cee = kron(Che,Cve);
            Coo = kron(Cho,Cvo);
            Coe = kron(Che,Cvo);
            Ceo = kron(Cho,Cve);
            %
            Ceee = kron(Cde,Cee);
            Ceoo = kron(Cdo,Ceo);
            Cooe = kron(Cde,Coo);
            Coeo = kron(Cdo,Coe);
            Ceeo = kron(Cdo,Cee);
            Ceoe = kron(Cde,Ceo);
            Cooo = kron(Cdo,Coo);
            Coee = kron(Cde,Coe);
            layer.Cvhd = ...
                [ Ceee; Ceoo; Cooe; Coeo; Ceeo; Ceoe; Cooo; Coee ]; % Cyxz
            
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
            Direction = tansacnet.utility.Direction;

            nComponents = layer.NumOutputs;
            varargout = cell(1,nComponents);
            
            if isdlarray(X)
                X = stripdims(X);
            end
            
            % Layer forward function for prediction goes here.
            decFactor = layer.Stride;
            decV = decFactor(Direction.VERTICAL);
            decH = decFactor(Direction.HORIZONTAL);
            decD = decFactor(Direction.DEPTH);
            nDec = decV*decH*decD;
            %
            Cvhd_ = layer.Cvhd;
            %
            height = size(X,1);
            width = size(X,2);
            depth = size(X,3);
            nRows = height/decV;
            nCols = width/decH;
            nLays = depth/decD;
            nSamples = size(X,5);
            %
            for iComponent = 1:nComponents
                arrayX = permute(reshape(X(:,:,:,iComponent,:),...
                    decV,nRows,decH,nCols,decD,nLays,nSamples),[1 3 5 2 4 6 7]);
                if isgpuarray(X)
                    varargout{iComponent} = pagefun(@mtimes,Cvhd_,...
                        reshape(arrayX,nDec,nRows,nCols,nLays,nSamples));
                else
                    varargout{iComponent} = reshape(...
                        Cvhd_*reshape(arrayX,nDec,[]),...
                        nDec,nRows,nCols,nLays,nSamples);
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
            Direction = tansacnet.utility.Direction;
            
            nComponents = layer.NumOutputs;
            decFactor = layer.Stride;
            decV = decFactor(Direction.VERTICAL);
            decH = decFactor(Direction.HORIZONTAL);
            decD = decFactor(Direction.DEPTH);
            nDec = decV*decH*decD;
            Cvhd_T = layer.Cvhd.';
            %
            dLdZ = varargin{layer.NumInputs+layer.NumOutputs+1};
            nRows = size(dLdZ,2);
            nCols = size(dLdZ,3);
            nLays = size(dLdZ,4);
            height = decV*nRows;
            width = decH*nCols;
            depth = decD*nLays;
            nSamples = size(dLdZ,5);
            dLdX = zeros(height,width,depth,nComponents,nSamples,'like',dLdZ);
            %
            for iComponent = 1:nComponents
                dLdZ = varargin{layer.NumInputs+layer.NumOutputs+iComponent};
                if isgpuarray(dLdZ)
                    arrayX = pagefun(@mtimes,Cvhd_T,dLdZ);
                else
                    arrayX = Cvhd_T*reshape(dLdZ,nDec,[]);
                end
                dLdX(:,:,:,iComponent,:) = ...
                    reshape(ipermute(reshape(arrayX,...
                    decV,decH,decD,nRows,nCols,nLays,nSamples),[1 3 5 2 4 6 7]),...
                    height,width,depth,1,nSamples);
            end
        end
    end
    
end