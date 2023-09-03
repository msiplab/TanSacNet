classdef lsunIntermediateFullRotation1dLayer < nnet.layer.Layer %#codegen
    %LSUNINTERMEDIATEDUALROTATION1DLAYER
    %   
    %   コンポーネント別に入力(nComponents)
    %      nChs x 1 x nBlks x nSamples
    %
    %   コンポーネント別に出力(nComponents):
    %      nChs x 1 x nBlks x nSamples
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
        Mode
        NumberOfBlocks
    end
    
    properties (Dependent)
        Mus
    end
    
    properties (Learnable,Dependent)
        Angles
    end
    
    properties (Access = private)
        PrivateNumberOfChannels
        PrivateAngles
        PrivateMus
        isUpdateRequested
    end
    
    properties (Hidden)
        Wn
        Un
    end
    
    methods
        function layer = lsunIntermediateFullRotation1dLayer(varargin)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            p = inputParser;
            addParameter(p,'Stride',[])
            addParameter(p,'Angles',[])
            addParameter(p,'Mus',[])
            addParameter(p,'Mode','Synthesis')
            addParameter(p,'Name','')
            addParameter(p,'NumberOfBlocks',1)
            parse(p,varargin{:})
            
            % Layer constructor function goes here.
            layer.Stride = p.Results.Stride;
            layer.NumberOfBlocks = p.Results.NumberOfBlocks;
            layer.PrivateNumberOfChannels = [ceil(layer.Stride/2) floor(layer.Stride/2)];
            layer.Name = p.Results.Name;
            layer.Mode = p.Results.Mode;
            layer.Angles = p.Results.Angles;
            layer.Mus = p.Results.Mus;
            layer.Description = layer.Mode ...
                + " LSUN intermediate full rotation " ...
                + "(pt,pb) = (" ...
                + layer.PrivateNumberOfChannels(1) + "," ...
                + layer.PrivateNumberOfChannels(2) + ")";
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

            nSamples = size(X,4);                        
            nblks = size(X,3);
            pt = layer.PrivateNumberOfChannels(1);
            pb = layer.PrivateNumberOfChannels(2);
            if layer.isUpdateRequested
                layer = layer.updateParameters();
            end
            %
            Wn_ = layer.Wn;            
            Un_ = layer.Un;
            if strcmp(layer.Mode,'Analysis')
                W_ = Wn_;
                U_ = Un_;                
            elseif strcmp(layer.Mode,'Synthesis')
                W_ = permute(Wn_,[2 1 3]);
                U_ = permute(Un_,[2 1 3]);
            else
                throw(MException('LsunLayer:InvalidMode',...
                    '%s : Mode should be either of Synthesis or Analysis',...
                    layer.Mode))
            end

            Y = X;
            Zt = zeros(pt,1,nblks,nSamples,'like',Y);
            Zb = zeros(pb,1,nblks,nSamples,'like',Y);
            for iSample = 1:nSamples
                Yt_iSample = Y(1:pt,:,:,iSample);
                Yb_iSample = Y(pt+1:pt+pb,:,:,iSample);
                if isgpuarray(X)
                    Zt_iSample = pagefun(@mtimes,W_,Yt_iSample);
                    Zb_iSample = pagefun(@mtimes,U_,Yb_iSample);
                else
                    Zt_iSample = zeros(size(Yt_iSample),'like',Yt_iSample);
                    Zb_iSample = zeros(size(Yb_iSample),'like',Yb_iSample);
                    for iblk = 1:nblks
                        Zt_iSample(:,:,iblk) = W_(:,:,iblk)*Yt_iSample(:,:,iblk);
                        Zb_iSample(:,:,iblk) = U_(:,:,iblk)*Yb_iSample(:,:,iblk);
                    end
                end
                Zt(:,:,:,iSample) = Zt_iSample;
                Zb(:,:,:,iSample) = Zb_iSample;
            end
            Z = cat(1,Zt,Zb);
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
            %                             learnable parameter
            %import tansacnet.lsun.get_fcn_orthmtxgen_diff
            
            nSamples = size(dLdZ,4);            
            nblks = size(dLdZ,3);
            
            pt = layer.PrivateNumberOfChannels(1);
            pb = layer.PrivateNumberOfChannels(2);            
            %
            if layer.isUpdateRequested
                layer = layer.updateParameters();
            end
            angles = layer.PrivateAngles;
            nAngles = size(angles,1);
            mus = cast(layer.PrivateMus,'like',angles);
            muW = mus(1:pt,:);
            muU = mus(pt+1:pt+pb,:);
            anglesW = angles(1:nAngles/2,:);
            anglesU = angles(nAngles/2+1:nAngles,:);

            % dLdX = dZdX x dLdZ
            Wn_ = layer.Wn;
            Un_ = layer.Un;            
            %dUnPst = zeros(size(Un_),'like',Un_);
            dWnPst = bsxfun(@times,permute(muW,[1 3 2]),Wn_);
            dUnPst = bsxfun(@times,permute(muU,[1 3 2]),Un_);            
            %for iblk = 1:(nrows*ncols)
            %    dUnPst(:,:,iblk) = bsxfun(@times,musU(:,iblk),Un_(:,:,iblk));
            %end
            dWnPre = repmat(eye(pt,'like',Wn_),[1 1 nblks]);
            dUnPre = repmat(eye(pb,'like',Un_),[1 1 nblks]);            
            
            %
            dLdX = cast(dLdZ,'like',X);
            %cdLd_low = reshape(dLdZ(pt+1:pt+pb,:,:,:),pb,nrows*ncols,nSamples);
            if strcmp(layer.Mode,'Analysis')
                W_ = permute(Wn_,[2 1 3]);
                U_ = permute(Un_,[2 1 3]);
            else
                W_ = Wn_;
                U_ = Un_;
            end
            cdLd_top = dLdX(1:pt,:,:,:);
            cdLd_btm = dLdX(pt+1:pt+pb,:,:,:);
            for iSample = 1:nSamples
                cdLd_top_iSample = cdLd_top(:,:,:,iSample);
                cdLd_btm_iSample = cdLd_btm(:,:,:,iSample);
                if isgpuarray(X)
                    cdLd_top_iSample = pagefun(@mtimes,W_,cdLd_top_iSample);
                    cdLd_btm_iSample = pagefun(@mtimes,U_,cdLd_btm_iSample);
                else
                    for iblk = 1:nblks
                        cdLd_top_iSample(:,:,iblk) = W_(:,1:pt,iblk)*cdLd_top_iSample(:,:,iblk);
                        cdLd_btm_iSample(:,:,iblk) = U_(:,1:pb,iblk)*cdLd_btm_iSample(:,:,iblk);
                    end
                end
                cdLd_top(:,:,:,iSample) = cdLd_top_iSample;                    
                cdLd_btm(:,:,:,iSample) = cdLd_btm_iSample;
            end
            dLdX(1:pt,:,:,:) = cdLd_top;
            dLdX(pt+1:pt+pb,:,:,:) = cdLd_btm;

            % dLdWi = <dLdZ,(dVdWi)X>
            fcn_orthmtxgen_diff = tansacnet.lsun.get_fcn_orthmtxgen_diff(angles);
            dLdW = zeros(nAngles,nblks,'like',dLdZ);
            dldz_top = dLdZ(1:pt,:,:,:);
            dldz_btm = dLdZ(pt+1:pt+pb,:,:,:);
            c_top = X(1:pt,:,:,:);
            c_btm = X(pt+1:pt+pb,:,:,:);
            for iAngle = uint32(1:nAngles/2)
                [dWn,dWnPst,dWnPre] = fcn_orthmtxgen_diff(anglesW,muW,iAngle,dWnPst,dWnPre);
                [dUn,dUnPst,dUnPre] = fcn_orthmtxgen_diff(anglesU,muU,iAngle,dUnPst,dUnPre);                
                if strcmp(layer.Mode,'Analysis')
                    dW_ = dWn;
                    dU_ = dUn;
                else
                    dW_ = permute(dWn,[2 1 3]);
                    dU_ = permute(dUn,[2 1 3]);
                end
                if isgpuarray(X)
                    c_top_ext = c_top; % idx 1 iblk iSample
                    c_btm_ext = c_btm; % idx 1 iblk iSample      
                    d_top_ext = pagefun(@mtimes,dW_,c_top_ext); % idx 1 iblk iSample                    
                    d_btm_ext = pagefun(@mtimes,dU_,c_btm_ext); % idx 1 iblk iSample
                    d_top = d_top_ext;
                    d_btm = d_btm_ext;
                    dLdW(iAngle,:) = sum(bsxfun(@times,dldz_top,d_top),[1 4]);
                    dLdW(nAngles/2+iAngle,:) = sum(bsxfun(@times,dldz_btm,d_btm),[1 4]);                    
                else
                    for iblk = 1:nblks
                        dW_iblk = dW_(:,:,iblk);                        
                        dU_iblk = dU_(:,:,iblk);
                        dldz_top_iblk = squeeze(dldz_top(:,:,iblk,:));                        
                        dldz_btm_iblk = squeeze(dldz_btm(:,:,iblk,:));
                        c_top_iblk = squeeze(c_top(:,:,iblk,:));                        
                        c_btm_iblk = squeeze(c_btm(:,:,iblk,:));
                        d_top_iblk = zeros(size(c_top_iblk),'like',c_top_iblk);                        
                        d_btm_iblk = zeros(size(c_btm_iblk),'like',c_btm_iblk);
                        for iSample = 1:nSamples
                            d_top_iblk(:,iSample) = dW_iblk*c_top_iblk(:,iSample);
                            d_btm_iblk(:,iSample) = dU_iblk*c_btm_iblk(:,iSample);                            
                        end
                        dLdW(iAngle,iblk) = sum(bsxfun(@times,dldz_top_iblk,d_top_iblk),'all');
                        dLdW(nAngles/2+iAngle,iblk) = sum(bsxfun(@times,dldz_btm_iblk,d_btm_iblk),'all');                        
                    end
                end
            end
        end

        function angles = get.Angles(layer)
            angles = layer.PrivateAngles;
        end
        
        function mus = get.Mus(layer)
            mus = layer.PrivateMus;
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
            pt = layer.PrivateNumberOfChannels(1);
            pb = layer.PrivateNumberOfChannels(2);
            if isempty(mus)
                mus = ones(pt+pb,nBlocks);
            elseif isscalar(mus)
                mus = mus*ones(pt+pb,nBlocks);
            end
            %
            layer.PrivateMus = mus;
            %layer = layer.updateParameters();
            layer.isUpdateRequested = true;
        end
        
        function layer = updateParameters(layer)
            %import tansacnet.lsun.get_fcn_orthmtxgen
            pt = layer.PrivateNumberOfChannels(1);
            pb = layer.PrivateNumberOfChannels(2);
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
            muW = mus(1:pt,:);
            muU = mus(pt+1:pt+pb,:);
            if nAngles > 0
                anglesW = angles(1:nAngles/2,:);
                anglesU = angles(nAngles/2+1:nAngles,:);
                fcn_orthmtxgen = tansacnet.lsun.get_fcn_orthmtxgen(angles);
                layer.Wn = fcn_orthmtxgen(anglesW,muW);
                layer.Un = fcn_orthmtxgen(anglesU,muU);
            else
                layer.Wn = reshape(muW,1,1,[]);
                layer.Un = reshape(muU,1,1,[]);
            end
            layer.isUpdateRequested = false;
        end
        
    end

end

