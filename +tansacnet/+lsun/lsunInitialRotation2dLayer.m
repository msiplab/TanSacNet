classdef lsunInitialRotation2dLayer < nnet.layer.Layer %#codegen
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
    % Copyright (c) 2022, Shogo MURAMATSU
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
            layer.Type = '';

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
            if layer.NumberOfBlocks(1) ~= nrows || ...
                    layer.NumberOfBlocks(2) ~= ncols
                layer.NumberOfBlocks = [nrows ncols];
            end
            ps = layer.PrivateNumberOfChannels(1);
            pa = layer.PrivateNumberOfChannels(2);
            nSamples = size(X,4);
            stride = layer.Stride;
            nDecs = prod(stride);
            nChsTotal = ps + pa;
            % Extend Angle paremeters for every block
            if size(layer.PrivateAngles,2) == 1
                layer.Angles = repmat(layer.PrivateAngles,[1 (nrows*ncols)]);
            end
            % Update parameters
            if layer.isUpdateRequested
                layer = layer.updateParameters();
            end
            %
            W0_ = layer.W0;
            U0_ = layer.U0;
            %Y = reshape(permute(X,[3 1 2 4]),nDecs,nrows*ncols*nSamples);
            Y = reshape(X,nDecs,nrows*ncols,nSamples);
            Zs = zeros(ceil(nDecs/2),nrows*ncols,nSamples,'like',Y);
            Za = zeros(floor(nDecs/2),nrows*ncols,nSamples,'like',Y);
            for iSample = 1:nSamples
                for iblk = 1:(nrows*ncols)
                    Zs(:,iblk,iSample) = W0_(:,1:ceil(nDecs/2),iblk)*Y(1:ceil(nDecs/2),iblk,iSample);
                    Za(:,iblk,iSample) = U0_(:,1:floor(nDecs/2),iblk)*Y(ceil(nDecs/2)+1:end,iblk,iSample);
                end
            end
            %Z = ipermute(reshape([Zs;Za],nChsTotal,nrows,ncols,nSamples),...
            %    [3 1 2 4]);
            Z = reshape([Zs;Za],nChsTotal,nrows,ncols,nSamples);
            
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
            ps = layer.PrivateNumberOfChannels(1);
            pa = layer.PrivateNumberOfChannels(2);
            nAngles = size(layer.PrivateAngles,1);
            nSamples = size(dLdZ,4);
            stride = layer.Stride;
            nDecs = prod(stride);
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
            %}
            % Extend Angle paremeters for every block
            if size(layer.PrivateAngles,2) == 1
                layer.Angles = repmat(layer.PrivateAngles,[1 (nrows*ncols)]);
            end
            if size(layer.PrivateMus,2) == 1
                layer.Mus = repmat(layer.PrivateMus,[1 (nrows*ncols)]);
            end
            if layer.isUpdateRequested
                layer = layer.updateParameters();
            end
            angles = layer.PrivateAngles;
            mus = cast(layer.Mus,'like',angles);            
            anglesW = angles(1:nAngles/2,:);
            anglesU = angles(nAngles/2+1:end,:);
            muW = mus(1:ps,:);
            muU = mus(ps+1:end,:);
            %[W0_,dW0Pst,dW0Pre] = fcn_orthmtxgen_diff(anglesW,muW,0,[],[]);
            %[U0_,dU0Pst,dU0Pre] = fcn_orthmtxgen_diff(anglesU,muU,0,[],[]);
            W0_ = layer.W0; %transpose(fcn_orthmtxgen(anglesW,muW,0));
            U0_ = layer.U0; %transpose(fcn_orthmtxgen(anglesU,muU,0));
            W0T = permute(W0_,[2 1 3]);
            U0T = permute(U0_,[2 1 3]);
            %if isdlarray(W0_)
            %    dW0Pst = dlarray(muW(:).*W0_);
            %    dU0Pst = dlarray(muU(:).*U0_);
            %    dW0Pre = dlarray(eye(ps,W0_.underlyingType));
            %    dU0Pre = dlarray(eye(pa,U0_.underlyingType));
            %else
            dW0Pst = zeros(size(W0_),'like',W0_);
            dU0Pst = zeros(size(U0_),'like',U0_);
            for iblk = 1:(nrows*ncols)
                dW0Pst(:,:,iblk) = bsxfun(@times,muW(:,iblk),W0_(:,:,iblk));
                dU0Pst(:,:,iblk) = bsxfun(@times,muU(:,iblk),U0_(:,:,iblk));
            end
            dW0Pre = repmat(eye(ps,'like',W0_),[1 1 (nrows*ncols)]);
            dU0Pre = repmat(eye(pa,'like',U0_),[1 1 (nrows*ncols)]);
            %end
            
            % Layer backward function goes here.
            % dLdX = dZdX x dLdZ
            Y = dLdZ; %permute(dLdZ,[3 1 2 4]);
            Ys = reshape(Y(1:ps,:,:,:),ps,nrows*ncols,nSamples);
            Ya = reshape(Y(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            for iSample = 1:nSamples
                for iblk = 1:(nrows*ncols)
                    Ys(:,iblk,iSample) = W0T(1:ceil(nDecs/2),:,iblk)*Ys(:,iblk,iSample); 
                    Ya(:,iblk,iSample) = U0T(1:floor(nDecs/2),:,iblk)*Ya(:,iblk,iSample);
                end
            end
            Zsa = cat(1,Ys,Ya);
            %dLdX = ipermute(reshape(Zsa,nDecs,nrows,ncols,nSamples),...
            %    [3 1 2 4]);
            dLdX = reshape(Zsa,nDecs,nrows,ncols,nSamples);
            fcn_orthmtxgen_diff = tansacnet.lsun.get_fcn_orthmtxgen_diff(angles);                        
            % dLdWi = <dLdZ,(dVdWi)X>
            dLdW = zeros(nAngles,nrows*ncols,'like',dLdZ);
            dldz_ = dLdZ; %permute(dLdZ,[3 1 2 4]);
            dldz_upp = reshape(dldz_(1:ps,:,:,:),ps,nrows*ncols,nSamples);
            dldz_low = reshape(dldz_(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            % (dVdWi)X
            a_ = X; %permute(X,[3 1 2 4]);
            c_upp = reshape(a_(1:ceil(nDecs/2),:,:,:),ceil(nDecs/2),nrows*ncols,nSamples);
            c_low = reshape(a_(ceil(nDecs/2)+1:nDecs,:,:,:),floor(nDecs/2),nrows*ncols,nSamples);
            for iAngle = uint32(1:nAngles/2)
                %dW0 = fcn_orthmtxgen(anglesW,muW,iAngle);
                %dU0 = fcn_orthmtxgen(anglesU,muU,iAngle);
                [dW0,dW0Pst,dW0Pre] = fcn_orthmtxgen_diff(anglesW,muW,iAngle,dW0Pst,dW0Pre);
                [dU0,dU0Pst,dU0Pre] = fcn_orthmtxgen_diff(anglesU,muU,iAngle,dU0Pst,dU0Pre);
                for iblk = 1:(nrows*ncols)
                    dldz_upp_iblk = squeeze(dldz_upp(:,iblk,:));
                    dldz_low_iblk = squeeze(dldz_low(:,iblk,:));
                    c_upp_iblk = squeeze(c_upp(:,iblk,:));
                    c_low_iblk = squeeze(c_low(:,iblk,:));
                    d_upp_iblk = zeros(size(c_upp_iblk),'like',c_upp_iblk);
                    d_low_iblk = zeros(size(c_low_iblk),'like',c_low_iblk);
                    for iSample = 1:nSamples
                        d_upp_iblk(:,iSample) = dW0(:,1:ceil(nDecs/2),iblk)*c_upp_iblk(:,iSample);
                        d_low_iblk(:,iSample) = dU0(:,1:floor(nDecs/2),iblk)*c_low_iblk(:,iSample);
                    end
                    dLdW(iAngle,iblk) = sum(bsxfun(@times,dldz_upp_iblk,d_upp_iblk),'all');
                    dLdW(nAngles/2+iAngle,iblk) = sum(bsxfun(@times,dldz_low_iblk,d_low_iblk),'all');
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

