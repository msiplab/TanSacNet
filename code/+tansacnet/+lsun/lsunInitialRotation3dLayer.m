classdef lsunInitialRotation3dLayer < nnet.layer.Layer %#codegen
    %LSUNINITIALROTATION3DLAYER
    %
    %   コンポーネント別に入力(nComponents):
    %      nChs x nRows x nCols x nLays x nSamples
    %
    %   コンポーネント別に出力(nComponents):
    %      nChs x nRows x nCols x nLays x nSamples
    %
    %
    % Requirements: MATLAB R2020b
    %
    % Copyright (c) 2020-2022, Eisuke KOBAYASHI, Shogo MURAMATSU
    %
    % All rights reserved.
    %
    % Contact address: Shogo MURAMATSU,
    %                Faculty of Engineering, Niigata University,
    %                8050 2-no-cho Ikarashi, Nishi-ku,
    %                Niigata, 950-2181, JAPAN
    %
    % http://msiplab.eng.niigata-u.ac.jp   

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
        
    methods
        
        function layer = lsunInitialRotation3dLayer(varargin)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            p = inputParser;
            addParameter(p,'Stride',[])
            addParameter(p,'Name','')
            addParameter(p,'Mus',[])
            addParameter(p,'Angles',[])
            addParameter(p,'NoDcLeakage',false)
            addParameter(p,'NumberOfBlocks',[1 1 1])
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
                + "(mv,mh,md) = (" ...
                + layer.Stride(1) + "," ...
                + layer.Stride(2) + "," ...
                + layer.Stride(3) + ")";
            layer.Type = '';
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
            nlays = size(X,4);
            nSamples = size(X,5);
            ps = layer.PrivateNumberOfChannels(1);
            pa = layer.PrivateNumberOfChannels(2);
            % Update parameters
            if layer.isUpdateRequested
                layer = layer.updateParameters();
            end
            %
            W0_ = layer.W0;
            U0_ = layer.U0;
            %Y = reshape(permute(X,[3 1 2 4]),ps+pa,nrows*ncols*nSamples);
            Y = reshape(X,ps+pa,nrows*ncols*nlays,nSamples);
            Zs = zeros(ps,nrows*ncols*nlays,nSamples,'like',Y);
            Za = zeros(pa,nrows*ncols*nlays,nSamples,'like',Y);
            for iSample = 1:nSamples
                if isgpuarray(X)
                    Ys_iSample = permute(Y(1:ps,:,iSample),[1 4 2 3]);
                    Ya_iSample = permute(Y(ps+1:end,:,iSample),[1 4 2 3]);
                    Zs_iSample = pagefun(@mtimes,W0_,Ys_iSample);
                    Za_iSample = pagefun(@mtimes,U0_,Ya_iSample);
                    Zs(:,:,iSample) = ipermute(Zs_iSample,[1 4 2 3]);
                    Za(:,:,iSample) = ipermute(Za_iSample,[1 4 2 3]);
                else
                    for iblk = 1:(nrows*ncols*nlays)
                        Zs(:,iblk,iSample) = W0_(:,1:ps,iblk)*Y(1:ps,iblk,iSample);
                        Za(:,iblk,iSample) = U0_(:,1:pa,iblk)*Y(ps+1:end,iblk,iSample);
                    end
                end
            end
            %Z = ipermute(reshape([Zs;Za],nChsTotal,nrows,ncols,nSamples),...
            %    [3 1 2 4]);
            Z = reshape([Zs;Za],ps+pa,nrows,ncols,nlays,nSamples);
            
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
            nlays = size(dLdZ,4);
            nSamples = size(dLdZ,5);
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
            W0_ = layer.W0; %transpose(fcn_orthmtxgen(anglesW,muW,0));
            U0_ = layer.U0; %transpose(fcn_orthmtxgen(anglesU,muU,0));
            W0T = permute(W0_,[2 1 3]);
            U0T = permute(U0_,[2 1 3]);
            dW0Pst = zeros(size(W0_),'like',W0_);
            dU0Pst = zeros(size(U0_),'like',U0_);
            for iblk = 1:(nrows*ncols*nlays)
                dW0Pst(:,:,iblk) = bsxfun(@times,muW(:,iblk),W0_(:,:,iblk));
                dU0Pst(:,:,iblk) = bsxfun(@times,muU(:,iblk),U0_(:,:,iblk));
            end
            dW0Pre = repmat(eye(ps,'like',W0_),[1 1 (nrows*ncols*nlays)]);
            dU0Pre = repmat(eye(pa,'like',U0_),[1 1 (nrows*ncols*nlays)]);
            
            % Layer backward function goes here.
            % dLdX = dZdX x dLdZ
            %Y = dLdZ; %permute(dLdZ,[3 1 2 4]);
            Ys = reshape(dLdZ(1:ps,:,:,:),ps,nrows*ncols*nlays,nSamples);
            Ya = reshape(dLdZ(ps+1:ps+pa,:,:,:),pa,nrows*ncols*nlays,nSamples);
            for iSample = 1:nSamples
                if isgpuarray(X)
                    Ys_iSample = permute(Ys(:,:,iSample),[1 4 2 3]);
                    Ya_iSample = permute(Ya(:,:,iSample),[1 4 2 3]);
                    Ys_iSample = pagefun(@mtimes,W0T(1:ps,:,:),Ys_iSample);
                    Ya_iSample = pagefun(@mtimes,U0T(1:pa,:,:),Ya_iSample);
                    Ys(:,:,iSample) = ipermute(Ys_iSample,[1 4 2 3]);
                    Ya(:,:,iSample) = ipermute(Ya_iSample,[1 4 2 3]);
                else
                    for iblk = 1:(nrows*ncols*nlays)
                        Ys(:,iblk,iSample) = W0T(1:ps,:,iblk)*Ys(:,iblk,iSample);
                        Ya(:,iblk,iSample) = U0T(1:pa,:,iblk)*Ya(:,iblk,iSample);
                    end
                end
            end
            Zsa = cat(1,Ys,Ya);
            %dLdX = ipermute(reshape(Zsa,ps+pa,nrows,ncols,nSamples),...
            %    [3 1 2 4]);
            dLdX = reshape(Zsa,ps+pa,nrows,ncols,nlays,nSamples);

            % dLdWi = <dLdZ,(dVdWi)X>
            fcn_orthmtxgen_diff = tansacnet.lsun.get_fcn_orthmtxgen_diff(angles);                        
            dLdW = zeros(nAngles,nrows*ncols*nlays,'like',dLdZ);
            dldz_upp = reshape(dLdZ(1:ps,:,:,:),ps,nrows*ncols*nlays,nSamples);
            dldz_low = reshape(dLdZ(ps+1:ps+pa,:,:,:),pa,nrows*ncols*nlays,nSamples);
            % (dVdWi)X
            %a_ = X; %permute(X,[3 1 2 4]);
            c_upp = reshape(X(1:ps,:,:,:),ps,nrows*ncols*nlays,nSamples);
            c_low = reshape(X(ps+1:ps+pa,:,:,:),pa,nrows*ncols*nlays,nSamples);
            for iAngle = uint32(1:nAngles/2)
                %dW0 = fcn_orthmtxgen(anglesW,muW,iAngle);
                %dU0 = fcn_orthmtxgen(anglesU,muU,iAngle);
                [dW0,dW0Pst,dW0Pre] = fcn_orthmtxgen_diff(anglesW,muW,iAngle,dW0Pst,dW0Pre);
                [dU0,dU0Pst,dU0Pre] = fcn_orthmtxgen_diff(anglesU,muU,iAngle,dU0Pst,dU0Pre);
                if isgpuarray(X)
                    c_upp_ext = permute(c_upp,[1 4 2 3]); % idx 1 iblk iSample
                    c_low_ext = permute(c_low,[1 4 2 3]); % idx 1 iblk iSample
                    d_upp_ext = pagefun(@mtimes,dW0(:,1:ps,:),c_upp_ext); % idx 1 iblk iSample
                    d_low_ext = pagefun(@mtimes,dU0(:,1:pa,:),c_low_ext); % idx 1 iblk iSample                    
                    d_upp = ipermute(d_upp_ext,[1 4 2 3]);
                    d_low = ipermute(d_low_ext,[1 4 2 3]);                    
                    dLdW(iAngle,:) = sum(bsxfun(@times,dldz_upp,d_upp),[1 3]);
                    dLdW(nAngles/2+iAngle,:) = sum(bsxfun(@times,dldz_low,d_low),[1 3]);                    
                else
                    for iblk = 1:(nrows*ncols*nlays)
                        dldz_upp_iblk = squeeze(dldz_upp(:,iblk,:));
                        dldz_low_iblk = squeeze(dldz_low(:,iblk,:));
                        c_upp_iblk = squeeze(c_upp(:,iblk,:));
                        c_low_iblk = squeeze(c_low(:,iblk,:));
                        d_upp_iblk = zeros(size(c_upp_iblk),'like',c_upp_iblk);
                        d_low_iblk = zeros(size(c_low_iblk),'like',c_low_iblk);
                        for iSample = 1:nSamples
                            d_upp_iblk(:,iSample) = dW0(:,1:ps,iblk)*c_upp_iblk(:,iSample);
                            d_low_iblk(:,iSample) = dU0(:,1:pa,iblk)*c_low_iblk(:,iSample);
                        end
                        dLdW(iAngle,iblk) = sum(bsxfun(@times,dldz_upp_iblk,d_upp_iblk),'all');
                        dLdW(nAngles/2+iAngle,iblk) = sum(bsxfun(@times,dldz_low_iblk,d_low_iblk),'all');
                    end
                end
            end
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