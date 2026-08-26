classdef lsunInitialRotation2dLayer < tansacnet.lsun.lsunRotation2dLayerBase %#codegen
    %LSUNINITIALROTATION2DLAYER
    %
    %   コンポーネント別に入力(nComponents):
    %      nChs x nRows x nCols x nSamples
    %
    %   コンポーネント別に出力(nComponents):
    %      nChs x nRows x nCols x nSamples
    %
    % Requirements: MATLAB R2022a
    %
    % Copyright (c) 2022-2024, Shogo MURAMATSU
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
        NumberOfBlocks
        Device
        DType
    end
    
    properties (Dependent)
        NoDcLeakage
    end
    
    properties (Dependent)
        Mus
    end
    
    properties (Learnable,Dependent)
        Angles
    end
    
    properties (Access = private)
        PrivateNumberOfChannels
        PrivateNoDcLeakage
        PrivateAngles
        PrivateMus
        isUpdateRequested
    end
    
    properties (Hidden)
        W0
        U0
    end

    properties (Hidden, Constant)
        Mode = 'Analysis'
    end

    methods
        
        function layer = lsunInitialRotation2dLayer(varargin)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            p = inputParser;
            addParameter(p,'Stride',[])
            addParameter(p,'Name','')
            addParameter(p,'Mus',[])
            addParameter(p,'Angles',[])
            addParameter(p,'NoDcLeakage',false)
            addParameter(p,'NumberOfBlocks',[1 1])
            addParameter(p,'DType','double')
            addParameter(p,'Device','cuda')
            parse(p,varargin{:})
            
            % Layer constructor function goes here.
            layer.Stride = p.Results.Stride;
            layer.NumberOfBlocks = p.Results.NumberOfBlocks;
            layer.PrivateNumberOfChannels = [ceil(prod(layer.Stride)/2) floor(prod(layer.Stride)/2)];
            layer.Name = p.Results.Name;
            layer.Mus = p.Results.Mus;
            layer.Angles = p.Results.Angles;
            layer.NoDcLeakage = p.Results.NoDcLeakage;
            layer.Description = "LSUN initial rotation " ...
                + "(ps,pa) = (" ...
                + layer.PrivateNumberOfChannels(1) + "," ...
                + layer.PrivateNumberOfChannels(2) + "), "  ...
                + "(mv,mh) = (" ...
                + layer.Stride(1) + "," ...
                + layer.Stride(2) + ")";
            %layer.Type = '';
            layer.Device = p.Results.Device;
            layer.DType = p.Results.DType;

            nChsTotal = sum(layer.PrivateNumberOfChannels);            
            nAngles = (nChsTotal-2)*nChsTotal/4;
            if size(layer.PrivateAngles,1)~=nAngles
                error('Invalid # of angles')
            end
            
            layer = layer.updateParameters();
        end
        
        function Z = predict(layer, X)
            % Forward input data through the layer at prediction time and
            % output the result.
            %
            % Inputs:
            %         layer       - Layer to forward propagate through
            %         X1, ..., Xn - Input data (n: # of components)
            % Outputs:
            %         Z           - Outputs of layer forward function
            %     
            
            % Layer forward function for prediction goes here.
            
            nrows = size(X,2);
            ncols = size(X,3);            
            nSamples = size(X,4);
            ps = layer.PrivateNumberOfChannels(1);
            pa = layer.PrivateNumberOfChannels(2);
            % Update parameters
            if layer.isUpdateRequested
                layer = layer.updateParameters();
            end
            %
            W0_ = layer.W0;
            U0_ = layer.U0;
            Y = reshape(X,ps+pa,nrows*ncols,nSamples);
            Ys = Y(1:ps,:,:);
            Ya = Y(ps+1:end,:,:);
            Zs = layer.applyBlockwiseMatrix(W0_(:,1:ps,:),Ys);
            Za = layer.applyBlockwiseMatrix(U0_(:,1:pa,:),Ya);
            Z = reshape([Zs;Za],ps+pa,nrows,ncols,nSamples);

        end
        
        function [dLdX, dLdW] = backward(layer, X, ~, dLdZ, ~)
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
            %
            %import tansacnet.lsun.get_fcn_orthmtxgen_diff
            
            nrows = size(dLdZ,2);
            ncols = size(dLdZ,3);            
            nSamples = size(dLdZ,4);
            ps = layer.PrivateNumberOfChannels(1);
            pa = layer.PrivateNumberOfChannels(2);
            %{
            if isempty(layer.Mus)
                layer.Mus = ones(ps+pa,1);
            elseif isscalar(layer.Mus)
                layer.Mus = layer.Mus*ones(ps+pa,1);
            end
            if layer.NoDcLeakage
                layer.Mus(1) = 1;
                layer.Angles(1:ps-1) = ...
                    zeros(ps-1,1,'like',layer.Angles);
            end
            % Extend Angle paremeters for every block
            if size(layer.PrivateAngles,2) == 1
                layer.Angles = repmat(layer.PrivateAngles,[1 (nrows*ncols)]);
            end
            if size(layer.PrivateMus,2) == 1
                layer.Mus = repmat(layer.PrivateMus,[1 (nrows*ncols)]);
            end
            %}
            if layer.isUpdateRequested
                layer = layer.updateParameters();
            end
            angles = layer.PrivateAngles;
            nAngles = size(angles,1);
            mus = cast(layer.Mus,'like',angles);            
            muW = mus(1:ps,:);
            muU = mus(ps+1:end,:);
            anglesW = angles(1:nAngles/2,:);
            anglesU = angles(nAngles/2+1:end,:);
            %[W0_,dW0Pst,dW0Pre] = fcn_orthmtxgen_diff(anglesW,muW,0,[],[]);
            %[U0_,dU0Pst,dU0Pre] = fcn_orthmtxgen_diff(anglesU,muU,0,[],[]);
            W0_ = layer.W0;
            U0_ = layer.U0;
            W0T = permute(W0_,[2 1 3]);
            U0T = permute(U0_,[2 1 3]);

            % Layer backward function goes here.
            % dLdX = dZdX x dLdZ
            Ys = reshape(dLdZ(1:ps,:,:,:),ps,nrows*ncols,nSamples);
            Ya = reshape(dLdZ(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            Ys = layer.applyBlockwiseMatrix(W0T(1:ps,:,:),Ys);
            Ya = layer.applyBlockwiseMatrix(U0T(1:pa,:,:),Ya);
            dLdX = reshape(cat(1,Ys,Ya),ps+pa,nrows,ncols,nSamples);

            % dLdWi = <dLdZ,(dVdWi)X>
            dldz_upp = reshape(dLdZ(1:ps,:,:,:),ps,nrows*ncols,nSamples);
            dldz_low = reshape(dLdZ(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            c_upp = reshape(X(1:ps,:,:,:),ps,nrows*ncols,nSamples);
            c_low = reshape(X(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            dLdW_upp = layer.computeAngleGradient(W0_,anglesW,muW,c_upp,dldz_upp);
            dLdW_low = layer.computeAngleGradient(U0_,anglesU,muU,c_low,dldz_low);
            dLdW = cat(1,dLdW_upp,dLdW_low);
        end

        function nodcleak = get.NoDcLeakage(layer)
            nodcleak = layer.PrivateNoDcLeakage;
        end

        function angles = get.Angles(layer)
            angles = layer.PrivateAngles;
        end
        
        function mus = get.Mus(layer)
            mus = layer.PrivateMus;
        end
        
        function layer = set.NoDcLeakage(layer,nodcleak)
            layer.PrivateNoDcLeakage = nodcleak;
            %
            layer.isUpdateRequested = true;
        end
        
        function layer = set.Angles(layer,angles)
            nBlocks = prod(layer.NumberOfBlocks);
            nChsTotal = sum(layer.PrivateNumberOfChannels);
            nAngles = (nChsTotal-2)*nChsTotal/4;
            if isempty(angles)
                angles = zeros(nAngles,nBlocks);
            elseif isscalar(angles)
                angles = angles*ones(nAngles,nBlocks,'like',angles);
            end
            %
            layer.PrivateAngles = angles;
            %layer = layer.updateParameters();
            layer.isUpdateRequested = true;
        end
        
        function layer = set.Mus(layer,mus)
            nBlocks = prod(layer.NumberOfBlocks);            
            ps = layer.PrivateNumberOfChannels(1);
            pa = layer.PrivateNumberOfChannels(2);
            %
            if isempty(mus)
                mus = ones(ps+pa,nBlocks);
            elseif isscalar(mus)
                mus = mus*ones(ps+pa,nBlocks);
            end
            %
            layer.PrivateMus = mus;
            %layer = layer.updateParameters();
            layer.isUpdateRequested = true;
        end
        
        function layer = updateParameters(layer)
            %import tansacnet.lsun.get_fcn_orthmtxgen
            ps = layer.PrivateNumberOfChannels(1);
            %
            if layer.NoDcLeakage
                layer.PrivateMus(1,:) = ones(1,size(layer.PrivateMus,2));                
                layer.PrivateAngles(1:ps-1,:) = ...
                    zeros(ps-1,size(layer.PrivateAngles,2),'like',layer.PrivateAngles);
            end            
            %
            angles = layer.PrivateAngles;            
            mus = cast(layer.PrivateMus,'like',angles);
            if isvector(angles)
                nAngles = length(angles);
            else
                nAngles = size(angles,1);
            end
            if isrow(mus)
                mus = mus.';
            end
            muW = mus(1:ps,:);
            muU = mus(ps+1:end,:);
            anglesW = angles(1:nAngles/2,:);
            anglesU = angles(nAngles/2+1:end,:);
            fcn_orthmtxgen = tansacnet.lsun.get_fcn_orthmtxgen(angles);                        
            layer.W0 = fcn_orthmtxgen(anglesW,muW);
            layer.U0 = fcn_orthmtxgen(anglesU,muU);
            layer.isUpdateRequested = false;
        end
        
    end
    
end

