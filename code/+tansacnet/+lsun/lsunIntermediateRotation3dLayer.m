classdef lsunIntermediateRotation3dLayer < nnet.layer.Layer %#codegen
    %LSUNINTERMEDIATEROTATION3DLAYER
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
        Un
    end
    
    methods
        function layer = lsunIntermediateRotation3dLayer(varargin)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            p = inputParser;
            addParameter(p,'Stride',[])
            addParameter(p,'Angles',[])
            addParameter(p,'Mus',[])
            addParameter(p,'Mode','Synthesis')
            addParameter(p,'Name','')
            addParameter(p,'NumberOfBlocks',[1 1 1])
            parse(p,varargin{:})
            
            % Layer constructor function goes here.
            layer.Stride = p.Results.Stride;
            layer.NumberOfBlocks = p.Results.NumberOfBlocks;
            layer.PrivateNumberOfChannels = [ceil(prod(layer.Stride)/2) floor(prod(layer.Stride)/2)];
            layer.Name = p.Results.Name;
            layer.Mode = p.Results.Mode;
            layer.Angles = p.Results.Angles;
            layer.Mus = p.Results.Mus;
            layer.Description = layer.Mode ...
                + " LSUN intermediate rotation " ...
                + "(ps,pa) = (" ...
                + layer.PrivateNumberOfChannels(1) + "," ...
                + layer.PrivateNumberOfChannels(2) + ")";
            layer.Type = '';
            
            nChsTotal = sum(layer.PrivateNumberOfChannels);
            nAngles = (nChsTotal-2)*nChsTotal/8;
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
            if layer.isUpdateRequested
                layer = layer.updateParameters();
            end
            %
            Un_ = layer.Un;
            Y = X; %permute(X,[4 1 2 3 5]);
            Ya = reshape(Y(ps+1:ps+pa,:,:,:,:),pa,nrows*ncols*nlays,nSamples);
            if strcmp(layer.Mode,'Analysis')
                A_ = Un_;
            elseif strcmp(layer.Mode,'Synthesis')
                A_ = permute(Un_,[2 1 3]);
            else
                throw(MException('NsoltLayer:InvalidMode',...
                    '%s : Mode should be either of Synthesis or Analysis',...
                    layer.Mode))
            end

            Za = zeros(pa,nrows*ncols*nlays,nSamples,'like',Y);
            for iSample = 1:nSamples
                if isgpuarray(X)
                    Ya_iSample = permute(Ya(:,:,iSample),[1 4 2 3]);
                    Za_iSample = pagefun(@mtimes,A_,Ya_iSample);
                    Za(:,:,iSample) = ipermute(Za_iSample,[1 4 2 3]);
                else
                    for iblk = 1:(nrows*ncols*nlays)
                        Za(:,iblk,iSample) = A_(:,:,iblk)*Ya(:,iblk,iSample);
                    end
                end
            end
            Y(ps+1:ps+pa,:,:,:,:) = reshape(Za,pa,nrows,ncols,nlays,nSamples);
            Z = Y; %ipermute(Y,[4 1 2 3 5]);
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
            
            nrows = size(dLdZ,2);
            ncols = size(dLdZ,3);
            nlays = size(dLdZ,4);
            nSamples = size(dLdZ,5);            
            ps = layer.PrivateNumberOfChannels(1);
            pa = layer.PrivateNumberOfChannels(2);            
            %
            if layer.isUpdateRequested
                layer = layer.updateParameters();
            end
            anglesU = layer.PrivateAngles;
            musU = cast(layer.PrivateMus,'like',anglesU);
            
            % dLdX = dZdX x dLdZ
            %Un = fcn_orthmtxgen(anglesU,musU,0);
            %[Un_,dUnPst,dUnPre] = fcn_orthmtxgen_diff(anglesU,musU,0,[],[]);
            Un_ = layer.Un;
            %dUnPst = zeros(size(Un_),'like',Un_);
            dUnPst = bsxfun(@times,permute(musU,[1 3 2]),Un_);
            %for iblk = 1:(nrows*ncols*nlays)
            %    dUnPst(:,:,iblk) = bsxfun(@times,musU(:,iblk),Un_(:,:,iblk));
            %end
            dUnPre = repmat(eye(pa,'like',Un_),[1 1 (nrows*ncols*nlays)]);
            
            %
            dLdX = reshape(dLdZ,ps+pa,nrows,ncols,nlays,nSamples); 
            %cdLd_low = reshape(dLdZ(ps+1:ps+pa,:,:,:),pa,nrows*ncols*nlays,nSamples);
            if strcmp(layer.Mode,'Analysis')
                A_ = permute(Un_,[2 1 3]);
            else
                A_ = Un_;
            end
            cdLd_low = reshape(dLdX(ps+1:ps+pa,:,:,:,:),pa,nrows*ncols*nlays,nSamples);
            for iSample = 1:nSamples
                if isgpuarray(X)
                    cdLd_low_iSample = permute(cdLd_low(:,:,iSample),[1 4 2 3]);
                    cdLd_low_iSample = pagefun(@mtimes,A_,cdLd_low_iSample);
                    cdLd_low(:,:,iSample) = ipermute(cdLd_low_iSample,[1 4 2 3]);                    
                else
                    for iblk = 1:(nrows*ncols*nlays)
                        cdLd_low(:,iblk,iSample) = A_(:,:,iblk)*cdLd_low(:,iblk,iSample);
                    end
                end
            end
            dLdX(ps+1:ps+pa,:,:,:,:) = reshape(cdLd_low,pa,nrows,ncols,nlays,nSamples);
            %dLdX = dLdX; %ipermute(adLd_,[3 1 2 4]);

            % dLdWi = <dLdZ,(dVdWi)X>
            fcn_orthmtxgen_diff = tansacnet.lsun.get_fcn_orthmtxgen_diff(anglesU);
            nAngles = size(anglesU,1);
            dLdW = zeros(nAngles,nrows*ncols*nlays,'like',dLdZ);
            dldz_low = reshape(dLdZ(ps+1:ps+pa,:,:,:,:),pa,nrows*ncols*nlays,nSamples);                        
            c_low = reshape(X(ps+1:ps+pa,:,:,:,:),pa,nrows*ncols*nlays,nSamples);  
            for iAngle = uint32(1:nAngles)
                [dUn,dUnPst,dUnPre] = fcn_orthmtxgen_diff(anglesU,musU,iAngle,dUnPst,dUnPre);
                if strcmp(layer.Mode,'Analysis')
                    dA_ = dUn;
                else
                    dA_ = permute(dUn,[2 1 3]);
                end
                if isgpuarray(X)
                    c_low_ext = permute(c_low,[1 4 2 3]); % idx 1 iblk iSample
                    d_low_ext = pagefun(@mtimes,dA_,c_low_ext); % idx 1 iblk iSample
                    d_low = ipermute(d_low_ext,[1 4 2 3]);
                    dLdW(iAngle,:) = sum(bsxfun(@times,dldz_low,d_low),[1 3]);
                else
                    for iblk = 1:(nrows*ncols*nlays)
                        dA_iblk = dA_(:,:,iblk);
                        dldz_low_iblk = squeeze(dldz_low(:,iblk,:));
                        c_low_iblk = squeeze(c_low(:,iblk,:));
                        d_low_iblk = zeros(size(c_low_iblk),'like',c_low_iblk);
                        for iSample = 1:nSamples
                            d_low_iblk(:,iSample) = dA_iblk*c_low_iblk(:,iSample);
                        end
                        dLdW(iAngle,iblk) = sum(bsxfun(@times,dldz_low_iblk,d_low_iblk),'all');
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
            nAngles = (nChsTotal-2)*nChsTotal/8;
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
            pa = layer.PrivateNumberOfChannels(2);
            if isempty(mus)
                mus = ones(pa,nBlocks);   
            elseif isscalar(mus)
                mus = mus*ones(pa,nBlocks,'like',mus);   
            end
            %
            layer.PrivateMus = mus;
            %layer = layer.updateParameters();
            layer.isUpdateRequested = true;
        end
        
        function layer = updateParameters(layer)
            %import tansacnet.lsun.get_fcn_orthmtxgen
            anglesU = layer.PrivateAngles;
            musU = cast(layer.PrivateMus,'like',anglesU);
            if isrow(musU)
                musU = musU.';
            end
            fcn_orthmtxgen = tansacnet.lsun.get_fcn_orthmtxgen(anglesU);
            layer.Un = fcn_orthmtxgen(anglesU,musU);
            layer.isUpdateRequested = false;
        end
        
    end

end