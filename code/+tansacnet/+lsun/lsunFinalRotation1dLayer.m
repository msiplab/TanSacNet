classdef lsunFinalRotation1dLayer < nnet.layer.Layer %#codegen
    %LSUNFINALROTATION1DLAYER
    %
    %   コンポーネント別に入力(nComponents):
    %      nChs x 1 x nBlks x nSamples
    %
    %   コンポーネント別に出力(nComponents):
    %      nChs x 1 x nBlks x nSamples
    %
    % Requirements: MATLAB R2022b
    %
    % Copyright (c) 2026, Motoyasu SUZUKI and Shogo MURAMATSU
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
        W0T
        U0T
    end

    methods
        function layer = lsunFinalRotation1dLayer(varargin)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            p = inputParser;
            addParameter(p,'Stride',[])
            addParameter(p,'Mus',[])
            addParameter(p,'Angles',[])
            addParameter(p,'Name','')
            addParameter(p,'NoDcLeakage',false)
            addParameter(p,'NumberOfBlocks',1)
            addParameter(p,'DType','double')
            addParameter(p,'Device','cuda')
            parse(p,varargin{:})

            % Layer constructor function goes here.
            layer.Stride = p.Results.Stride;
            layer.NumberOfBlocks = p.Results.NumberOfBlocks;
            layer.PrivateNumberOfChannels = [ceil(layer.Stride/2) floor(layer.Stride/2)];
            layer.Mus = p.Results.Mus;
            layer.Angles = p.Results.Angles;
            layer.NoDcLeakage = p.Results.NoDcLeakage;
            layer.Name = p.Results.Name;
            layer.Description = "LSUN final rotation " ...
                + "(pt,pb) = (" ...
                + layer.PrivateNumberOfChannels(1) + "," ...
                + layer.PrivateNumberOfChannels(2) + "), "  ...
                + "m = " + layer.Stride;
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

            nblks = size(X,3);
            nSamples = size(X,4);
            pt = layer.PrivateNumberOfChannels(1);
            pb = layer.PrivateNumberOfChannels(2);
            if layer.isUpdateRequested
                layer = layer.updateParameters();
            end
            %
            W0T_ = layer.W0T;
            U0T_ = layer.U0T;
            Yt = X(1:pt,:,:,:);
            Yb = X(pt+1:pt+pb,:,:,:);
            if isgpuarray(X)
                Zt = pagefun(@mtimes,W0T_,Yt);
                Zb = pagefun(@mtimes,U0T_,Yb);
            else
                Zt = zeros(pt,1,nblks,nSamples,'like',X);
                Zb = zeros(pb,1,nblks,nSamples,'like',X);
                for iSample = 1:nSamples
                    for iblk = 1:nblks
                        Zt(:,:,iblk,iSample) = W0T_(:,:,iblk)*Yt(:,:,iblk,iSample);
                        Zb(:,:,iblk,iSample) = U0T_(:,:,iblk)*Yb(:,:,iblk,iSample);
                    end
                end
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

            nblks = size(dLdZ,3);
            nSamples = size(dLdZ,4);
            pt = layer.PrivateNumberOfChannels(1);
            pb = layer.PrivateNumberOfChannels(2);
            if layer.isUpdateRequested
                layer = layer.updateParameters();
            end
            angles = layer.PrivateAngles;
            nAngles = size(angles,1);
            mus = cast(layer.Mus,'like',angles);
            muW = mus(1:pt,:);
            muU = mus(pt+1:end,:);
            anglesW = angles(1:nAngles/2,:);
            anglesU = angles(nAngles/2+1:end,:);
            W0_T = layer.W0T;
            U0_T = layer.U0T;
            W0_ = permute(W0_T,[2 1 3]);
            U0_ = permute(U0_T,[2 1 3]);
            dW0Pst = bsxfun(@times,permute(muW,[1 3 2]),W0_);
            dU0Pst = bsxfun(@times,permute(muU,[1 3 2]),U0_);
            dW0Pre = repmat(eye(pt,'like',W0_),[1 1 nblks]);
            dU0Pre = repmat(eye(pb,'like',U0_),[1 1 nblks]);

            % Layer backward function goes here.
            % dLdX = dZdX x dLdZ
            dLdX = dLdZ;
            if isgpuarray(X)
                dLdX(1:pt,:,:,:) = pagefun(@mtimes,W0_,dLdZ(1:pt,:,:,:));
                dLdX(pt+1:pt+pb,:,:,:) = pagefun(@mtimes,U0_,dLdZ(pt+1:pt+pb,:,:,:));
            else
                for iSample = 1:nSamples
                    for iblk = 1:nblks
                        dLdX(1:pt,:,iblk,iSample) = W0_(:,:,iblk)*dLdZ(1:pt,:,iblk,iSample);
                        dLdX(pt+1:pt+pb,:,iblk,iSample) = U0_(:,:,iblk)*dLdZ(pt+1:pt+pb,:,iblk,iSample);
                    end
                end
            end

            % dLdWi = <dLdZ,(dVdWi)X>
            fcn_orthmtxgen_diff = tansacnet.lsun.get_fcn_orthmtxgen_diff(angles);
            dLdW = zeros(nAngles,nblks,'like',dLdZ);
            dldz_top = dLdZ(1:pt,:,:,:);
            dldz_btm = dLdZ(pt+1:pt+pb,:,:,:);
            c_top = X(1:pt,:,:,:);
            c_btm = X(pt+1:pt+pb,:,:,:);
            for iAngle = uint32(1:nAngles/2)
                [dW0,dW0Pst,dW0Pre] = fcn_orthmtxgen_diff(anglesW,muW,iAngle,dW0Pst,dW0Pre);
                [dU0,dU0Pst,dU0Pre] = fcn_orthmtxgen_diff(anglesU,muU,iAngle,dU0Pst,dU0Pre);
                dW0_T = permute(dW0,[2 1 3]);
                dU0_T = permute(dU0,[2 1 3]);
                if isgpuarray(X)
                    d_top = pagefun(@mtimes,dW0_T,c_top);
                    d_btm = pagefun(@mtimes,dU0_T,c_btm);
                    dLdW(iAngle,:) = sum(bsxfun(@times,dldz_top,d_top),[1 4]);
                    dLdW(nAngles/2+iAngle,:) = sum(bsxfun(@times,dldz_btm,d_btm),[1 4]);
                else
                    for iblk = 1:nblks
                        dldz_top_iblk = squeeze(dldz_top(:,:,iblk,:));
                        dldz_btm_iblk = squeeze(dldz_btm(:,:,iblk,:));
                        c_top_iblk = squeeze(c_top(:,:,iblk,:));
                        c_btm_iblk = squeeze(c_btm(:,:,iblk,:));
                        d_top_iblk = zeros(size(c_top_iblk),'like',c_top_iblk);
                        d_btm_iblk = zeros(size(c_btm_iblk),'like',c_btm_iblk);
                        for iSample = 1:nSamples
                            d_top_iblk(:,iSample) = dW0_T(:,:,iblk)*c_top_iblk(:,iSample);
                            d_btm_iblk(:,iSample) = dU0_T(:,:,iblk)*c_btm_iblk(:,iSample);
                        end
                        dLdW(iAngle,iblk) = sum(bsxfun(@times,dldz_top_iblk,d_top_iblk),'all');
                        dLdW(nAngles/2+iAngle,iblk) = sum(bsxfun(@times,dldz_btm_iblk,d_btm_iblk),'all');
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
            layer.isUpdateRequested = true;
        end

        function layer = set.Mus(layer,mus)
            nBlocks = prod(layer.NumberOfBlocks);
            pt = layer.PrivateNumberOfChannels(1);
            pb = layer.PrivateNumberOfChannels(2);
            %
            if isempty(mus)
                mus = ones(pt+pb,nBlocks);
            elseif isscalar(mus)
                mus = mus*ones(pt+pb,nBlocks);
            end
            %
            layer.PrivateMus = mus;
            layer.isUpdateRequested = true;
        end

        function layer = updateParameters(layer)
            pt = layer.PrivateNumberOfChannels(1);
            %
            if layer.NoDcLeakage
                layer.PrivateMus(1,:) = ones(1,size(layer.PrivateMus,2));
                layer.PrivateAngles(1:pt-1,:) = ...
                    zeros(pt-1,size(layer.PrivateAngles,2),'like',layer.PrivateAngles);
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
            muW = mus(1:pt,:);
            muU = mus(pt+1:end,:);
            anglesW = angles(1:nAngles/2,:);
            anglesU = angles(nAngles/2+1:end,:);
            if nAngles > 0
                fcn_orthmtxgen = tansacnet.lsun.get_fcn_orthmtxgen(angles);
                layer.W0T = permute(fcn_orthmtxgen(anglesW,muW),[2 1 3]);
                layer.U0T = permute(fcn_orthmtxgen(anglesU,muU),[2 1 3]);
            else
                layer.W0T = reshape(muW,1,1,[]);
                layer.U0T = reshape(muU,1,1,[]);
            end
            layer.isUpdateRequested = false;
        end

    end

end
