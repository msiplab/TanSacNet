classdef lsunIntermediateFullRotation1dLayer < nnet.layer.Layer %#codegen
    %LSUNINTERMEDIATEDUALROTATION1DLAYER
    %   
    %   コンポーネント別に入力(nComponents)
    %      nChs x nSamples x nBlks
    %
    %   コンポーネント別に出力(nComponents):
    %      nChs x nSamples x nBlks
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
                + "(ps,pa) = (" ...
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

            nSamples = size(X,2);                        
            nblks = size(X,3);
            ps = layer.PrivateNumberOfChannels(1);
            pa = layer.PrivateNumberOfChannels(2);
            if layer.isUpdateRequested
                layer = layer.updateParameters();
            end
            %
            Wn_ = layer.Wn;            
            Un_ = layer.Un;
            if strcmp(layer.Mode,'Analysis')
                S_ = Wn_;
                A_ = Un_;                
            elseif strcmp(layer.Mode,'Synthesis')
                S_ = permute(Wn_,[2 1 3]);
                A_ = permute(Un_,[2 1 3]);
            else
                throw(MException('LsunLayer:InvalidMode',...
                    '%s : Mode should be either of Synthesis or Analysis',...
                    layer.Mode))
            end

            Y = permute(X,[1 3 2]);
            Zs = zeros(ps,nblks,nSamples,'like',Y);
            Za = zeros(pa,nblks,nSamples,'like',Y);            
            for iSample = 1:nSamples
                if isgpuarray(X) 
                    Ys_iSample = permute(Y(1:ps,:,iSample),[1 4 2 3]);
                    Ya_iSample = permute(Y(ps+1:ps+pa,:,iSample),[1 4 2 3]);
                    Zs_iSample = pagefun(@mtimes,S_,Ys_iSample);
                    Za_iSample = pagefun(@mtimes,A_,Ya_iSample);
                    Zs(:,:,iSample) = ipermute(Zs_iSample,[1 4 2 3]);
                    Za(:,:,iSample) = ipermute(Za_iSample,[1 4 2 3]);
                else
                    for iblk = 1:nblks
                        Zs(:,iblk,iSample) = S_(:,:,iblk)*Y(1:ps,iblk,iSample);
                        Za(:,iblk,iSample) = A_(:,:,iblk)*Y(ps+1:ps+pa,iblk,iSample);
                    end
                end
            end
            Y(1:ps,:,:) = Zs;
            Y(ps+1:ps+pa,:,:) = Za;
            Z = ipermute(Y,[1 3 2]);
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
            
            nSamples = size(dLdZ,2);            
            nblks = size(dLdZ,3);
            
            ps = layer.PrivateNumberOfChannels(1);
            pa = layer.PrivateNumberOfChannels(2);            
            %
            if layer.isUpdateRequested
                layer = layer.updateParameters();
            end
            angles = layer.PrivateAngles;
            nAngles = size(angles,1);
            mus = cast(layer.PrivateMus,'like',angles);
            muW = mus(1:ps,:);
            muU = mus(ps+1:ps+pa,:);
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
            dWnPre = repmat(eye(ps,'like',Wn_),[1 1 nblks]);
            dUnPre = repmat(eye(pa,'like',Un_),[1 1 nblks]);            
            
            %
            dLdX = dLdZ;
            %cdLd_low = reshape(dLdZ(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            if strcmp(layer.Mode,'Analysis')
                S_ = permute(Wn_,[2 1 3]);
                A_ = permute(Un_,[2 1 3]);
            else
                S_ = Wn_;
                A_ = Un_;
            end
            cdLd_top = permute(dLdX(1:ps,:,:),[1 3 2]);
            cdLd_btm = permute(dLdX(ps+1:ps+pa,:,:),[1 3 2]);
            for iSample = 1:nSamples
                if isgpuarray(X)
                    cdLd_top_iSample = permute(cdLd_top(:,:,iSample),[1 4 2 3]);
                    cdLd_btm_iSample = permute(cdLd_btm(:,:,iSample),[1 4 2 3]);
                    cdLd_top_iSample = pagefun(@mtimes,S_,cdLd_top_iSample);
                    cdLd_btm_iSample = pagefun(@mtimes,A_,cdLd_btm_iSample);
                    cdLd_top(:,:,iSample) = ipermute(cdLd_top_iSample,[1 4 2 3]);                    
                    cdLd_btm(:,:,iSample) = ipermute(cdLd_btm_iSample,[1 4 2 3]);                                        
                else
                    for iblk = 1:nblks
                        cdLd_top(:,iblk,iSample) = S_(:,1:ps,iblk)*cdLd_top(:,iblk,iSample);
                        cdLd_btm(:,iblk,iSample) = A_(:,1:pa,iblk)*cdLd_btm(:,iblk,iSample);
                    end
                end
            end
            dLdX(1:ps,:,:) = ipermute(reshape(cdLd_top,ps,nblks,nSamples),[1 3 2]);
            dLdX(ps+1:ps+pa,:,:) = ipermute(reshape(cdLd_btm,pa,nblks,nSamples),[1 3 2]);
            %dLdX = dLdX; %ipermute(adLd_,[3 1 2 4]);

            % dLdWi = <dLdZ,(dVdWi)X>
            fcn_orthmtxgen_diff = tansacnet.lsun.get_fcn_orthmtxgen_diff(angles);
            dLdW = zeros(nAngles,nblks,'like',dLdZ);
            dldz_top = permute(dLdZ(1:ps,:,:),[1 3 2]);                        
            dldz_btm = permute(dLdZ(ps+1:ps+pa,:,:),[1 3 2]);
            c_top = permute(X(1:ps,:,:),[1 3 2]);              
            c_btm = permute(X(ps+1:ps+pa,:,:),[1 3 2]);  
            for iAngle = uint32(1:nAngles/2)
                [dWn,dWnPst,dWnPre] = fcn_orthmtxgen_diff(anglesW,muW,iAngle,dWnPst,dWnPre);
                [dUn,dUnPst,dUnPre] = fcn_orthmtxgen_diff(anglesU,muU,iAngle,dUnPst,dUnPre);                
                if strcmp(layer.Mode,'Analysis')
                    dS_ = dWn;
                    dA_ = dUn;
                else
                    dS_ = permute(dWn,[2 1 3]);
                    dA_ = permute(dUn,[2 1 3]);
                end
                if isgpuarray(X)
                    c_top_ext = permute(c_top,[1 4 2 3]); % idx 1 iblk iSample
                    c_btm_ext = permute(c_btm,[1 4 2 3]); % idx 1 iblk iSample      
                    d_top_ext = pagefun(@mtimes,dS_,c_top_ext); % idx 1 iblk iSample                    
                    d_btm_ext = pagefun(@mtimes,dA_,c_btm_ext); % idx 1 iblk iSample
                    d_top = ipermute(d_top_ext,[1 4 2 3]);
                    d_btm = ipermute(d_btm_ext,[1 4 2 3]);                    
                    dLdW(iAngle,:) = sum(bsxfun(@times,dldz_top,d_top),[1 3]);
                    dLdW(nAngles/2+iAngle,:) = sum(bsxfun(@times,dldz_btm,d_btm),[1 3]);                    
                else
                    for iblk = 1:nblks
                        dS_iblk = dS_(:,:,iblk);                        
                        dA_iblk = dA_(:,:,iblk);
                        dldz_top_iblk = squeeze(dldz_top(:,iblk,:));                        
                        dldz_btm_iblk = squeeze(dldz_btm(:,iblk,:));
                        c_top_iblk = squeeze(c_top(:,iblk,:));                        
                        c_btm_iblk = squeeze(c_btm(:,iblk,:));
                        d_top_iblk = zeros(size(c_top_iblk),'like',c_top_iblk);                        
                        d_btm_iblk = zeros(size(c_btm_iblk),'like',c_btm_iblk);
                        for iSample = 1:nSamples
                            d_top_iblk(:,iSample) = dS_iblk*c_top_iblk(:,iSample);
                            d_btm_iblk(:,iSample) = dA_iblk*c_btm_iblk(:,iSample);                            
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
            ps = layer.PrivateNumberOfChannels(1);
            pa = layer.PrivateNumberOfChannels(2);
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
            pa = layer.PrivateNumberOfChannels(2);
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
            muU = mus(ps+1:ps+pa,:);
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
