classdef salsunInitialRotation1dLayer < nnet.layer.Layer %#codegen
    %SALSUNINITIALROTATION1DLAYER
    %
    %   Data-path input  'x'     : nChsTotal x 1 x nBlks x nSamples
    %
    %   Control-path input 'theta': nAngles x nBlks x nSamples
    %                               where nAngles = (nChsTotal-2)*nChsTotal/4
    %                               (first half: anglesW for the top
    %                               (pt) group, second half: anglesU for
    %                               the bottom (pb) group; angles may
    %                               vary from sample to sample, unlike
    %                               tansacnet.lsun's Angles, which only
    %                               vary block by block)
    %
    %   Output                   : nChsTotal x 1 x nBlks x nSamples
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
        Mus
    end

    properties (Access = private)
        PrivateNumberOfChannels
        PrivateMus
    end

    methods
        function layer = salsunInitialRotation1dLayer(varargin)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            p = inputParser;
            addParameter(p,'Stride',[])
            addParameter(p,'Name','')
            addParameter(p,'Mus',[])
            addParameter(p,'NumberOfBlocks',1)
            addParameter(p,'DType','double')
            addParameter(p,'Device','cuda')
            parse(p,varargin{:})

            % Layer constructor function goes here.
            layer.Stride = p.Results.Stride;
            layer.NumberOfBlocks = p.Results.NumberOfBlocks;
            layer.PrivateNumberOfChannels = [ceil(layer.Stride/2) floor(layer.Stride/2)];
            layer.Name = p.Results.Name;
            layer.Mus = p.Results.Mus;
            layer.Description = "SA-LSUN initial rotation " ...
                + "(pt,pb) = (" ...
                + layer.PrivateNumberOfChannels(1) + "," ...
                + layer.PrivateNumberOfChannels(2) + "), "  ...
                + "m = " + layer.Stride;
            layer.Type = '';
            layer.Device = p.Results.Device;
            layer.DType = p.Results.DType;

            % Two named inputs: data path 'x' and control path 'theta'.
            layer.InputNames = { 'x', 'theta' };
        end

        function Z = predict(layer, X, Theta)
            % Forward input data through the layer at prediction time and
            % output the result.
            %
            % Inputs:
            %         layer - Layer to forward propagate through
            %         X     - Data-path input
            %         Theta - Control-path input (rotation angles)
            % Outputs:
            %         Z     - Output of layer forward function

            nblks = size(X,3);
            nSamples = size(X,4);
            pt = layer.PrivateNumberOfChannels(1);
            pb = layer.PrivateNumberOfChannels(2);
            nAngles = size(Theta,1);
            nAnglesH = nAngles/2;
            %
            mus = cast(layer.PrivateMus,'like',Theta);
            muW = mus(1:pt,:);
            muU = mus(pt+1:pt+pb,:);

            Yt = X(1:pt,:,:,:);
            Yb = X(pt+1:pt+pb,:,:,:);
            if isgpuarray(X)
                angles_ = reshape(Theta,nAngles,nblks*nSamples);
                anglesW_ = angles_(1:nAnglesH,:);
                anglesU_ = angles_(nAnglesH+1:nAngles,:);
                muW_ = repmat(muW,[1 nSamples]);
                muU_ = repmat(muU,[1 nSamples]);
                Yt_ext = reshape(Yt,pt,1,nblks*nSamples);
                Yb_ext = reshape(Yb,pb,1,nblks*nSamples);
                if nAnglesH > 0
                    fcn_orthmtxgenW = tansacnet.lsun.get_fcn_orthmtxgen(anglesW_);
                    fcn_orthmtxgenU = tansacnet.lsun.get_fcn_orthmtxgen(anglesU_);
                    W0 = fcn_orthmtxgenW(anglesW_,muW_);
                    U0 = fcn_orthmtxgenU(anglesU_,muU_);
                else
                    W0 = reshape(muW_,pt,1,[]);
                    U0 = reshape(muU_,pb,1,[]);
                end
                Zt = reshape(pagefun(@mtimes,W0,Yt_ext),pt,1,nblks,nSamples);
                Zb = reshape(pagefun(@mtimes,U0,Yb_ext),pb,1,nblks,nSamples);
            else
                Zt = zeros(pt,1,nblks,nSamples,'like',X);
                Zb = zeros(pb,1,nblks,nSamples,'like',X);
                for iSample = 1:nSamples
                    angles = Theta(:,:,iSample);
                    anglesW = angles(1:nAnglesH,:);
                    anglesU = angles(nAnglesH+1:nAngles,:);
                    if nAnglesH > 0
                        fcn_orthmtxgenW = tansacnet.lsun.get_fcn_orthmtxgen(anglesW);
                        fcn_orthmtxgenU = tansacnet.lsun.get_fcn_orthmtxgen(anglesU);
                        W0_i = fcn_orthmtxgenW(anglesW,muW);
                        U0_i = fcn_orthmtxgenU(anglesU,muU);
                    else
                        W0_i = reshape(muW,pt,1,[]);
                        U0_i = reshape(muU,pb,1,[]);
                    end
                    for iblk = 1:nblks
                        Zt(:,:,iblk,iSample) = W0_i(:,:,iblk)*Yt(:,:,iblk,iSample);
                        Zb(:,:,iblk,iSample) = U0_i(:,:,iblk)*Yb(:,:,iblk,iSample);
                    end
                end
            end
            Z = cat(1,Zt,Zb);
        end

        function [dLdX, dLdTheta] = backward(layer, X, Theta, ~, dLdZ, ~)
            % (Optional) Backward propagate the derivative of the loss
            % function through the layer.
            %
            % Inputs:
            %         layer       - Layer to backward propagate through
            %         X, Theta    - Layer inputs (data path, control path)
            %         Z           - Output of layer forward function
            %         dLdZ        - Gradient propagated from the next layer
            %         memory      - Memory value from forward function
            % Outputs:
            %         dLdX        - Derivative of the loss w.r.t. X
            %         dLdTheta    - Derivative of the loss w.r.t. Theta
            %                       (fed back into the control-path
            %                       parameter estimator)

            nblks = size(dLdZ,3);
            nSamples = size(dLdZ,4);
            pt = layer.PrivateNumberOfChannels(1);
            pb = layer.PrivateNumberOfChannels(2);
            nAngles = size(Theta,1);
            nAnglesH = nAngles/2;
            %
            mus = cast(layer.PrivateMus,'like',Theta);
            muW = mus(1:pt,:);
            muU = mus(pt+1:pt+pb,:);
            if isgpuarray(X) && ~isgpuarray(dLdZ)
                dLdZ = gpuArray(dLdZ);
            end
            dLdX = dLdZ;
            dLdTheta = zeros(nAngles,nblks,nSamples,'like',dLdZ);
            if isgpuarray(X)
                angles_ = reshape(Theta,nAngles,nblks*nSamples);
                anglesW_ = angles_(1:nAnglesH,:);
                anglesU_ = angles_(nAnglesH+1:nAngles,:);
                muW_ = repmat(muW,[1 nSamples]);
                muU_ = repmat(muU,[1 nSamples]);
                cdLd_top_ext = reshape(dLdZ(1:pt,:,:,:),pt,1,nblks*nSamples);
                cdLd_btm_ext = reshape(dLdZ(pt+1:pt+pb,:,:,:),pb,1,nblks*nSamples);
                if nAnglesH > 0
                    fcn_orthmtxgenW = tansacnet.lsun.get_fcn_orthmtxgen(anglesW_);
                    fcn_orthmtxgenU = tansacnet.lsun.get_fcn_orthmtxgen(anglesU_);
                    W0 = fcn_orthmtxgenW(anglesW_,muW_);
                    U0 = fcn_orthmtxgenU(anglesU_,muU_);
                    W0T = permute(W0,[2 1 3]);
                    U0T = permute(U0,[2 1 3]);
                    dLdX(1:pt,:,:,:) = reshape(pagefun(@mtimes,W0T,cdLd_top_ext),pt,1,nblks,nSamples);
                    dLdX(pt+1:pt+pb,:,:,:) = reshape(pagefun(@mtimes,U0T,cdLd_btm_ext),pb,1,nblks,nSamples);

                    % dLdTheta_i = <dLdZ,(dVdTheta_i)X>, batched across
                    % blocks and samples together.
                    c_top_ext = reshape(X(1:pt,:,:,:),pt,1,nblks*nSamples);
                    c_btm_ext = reshape(X(pt+1:pt+pb,:,:,:),pb,1,nblks*nSamples);
                    dldz_top = dLdZ(1:pt,:,:,:);
                    dldz_btm = dLdZ(pt+1:pt+pb,:,:,:);
                    fcn_orthmtxgen_diffW = tansacnet.lsun.get_fcn_orthmtxgen_diff(anglesW_);
                    fcn_orthmtxgen_diffU = tansacnet.lsun.get_fcn_orthmtxgen_diff(anglesU_);
                    dWPst = bsxfun(@times,permute(muW_,[1 3 2]),W0);
                    dWPre = repmat(eye(pt,'like',W0),[1 1 nblks*nSamples]);
                    dUPst = bsxfun(@times,permute(muU_,[1 3 2]),U0);
                    dUPre = repmat(eye(pb,'like',U0),[1 1 nblks*nSamples]);
                    for iAngle = uint32(1:nAnglesH)
                        [dW,dWPst,dWPre] = fcn_orthmtxgen_diffW(anglesW_,muW_,iAngle,dWPst,dWPre);
                        [dU,dUPst,dUPre] = fcn_orthmtxgen_diffU(anglesU_,muU_,iAngle,dUPst,dUPre);
                        d_top_ext = pagefun(@mtimes,dW,c_top_ext);
                        d_btm_ext = pagefun(@mtimes,dU,c_btm_ext);
                        d_top = reshape(d_top_ext,pt,1,nblks,nSamples);
                        d_btm = reshape(d_btm_ext,pb,1,nblks,nSamples);
                        dLdTheta(iAngle,:,:) = sum(bsxfun(@times,dldz_top,d_top),1);
                        dLdTheta(nAnglesH+iAngle,:,:) = sum(bsxfun(@times,dldz_btm,d_btm),1);
                    end
                else
                    W0T = reshape(muW_,pt,1,[]);
                    U0T = reshape(muU_,pb,1,[]);
                    dLdX(1:pt,:,:,:) = reshape(pagefun(@mtimes,W0T,cdLd_top_ext),pt,1,nblks,nSamples);
                    dLdX(pt+1:pt+pb,:,:,:) = reshape(pagefun(@mtimes,U0T,cdLd_btm_ext),pb,1,nblks,nSamples);
                    % nAnglesH==0: no angle derivatives to accumulate.
                end
            else
                for iSample = 1:nSamples
                    angles = Theta(:,:,iSample);
                    anglesW = angles(1:nAnglesH,:);
                    anglesU = angles(nAnglesH+1:nAngles,:);
                    if nAnglesH > 0
                        fcn_orthmtxgenW = tansacnet.lsun.get_fcn_orthmtxgen(anglesW);
                        fcn_orthmtxgenU = tansacnet.lsun.get_fcn_orthmtxgen(anglesU);
                        W0_i = fcn_orthmtxgenW(anglesW,muW);
                        U0_i = fcn_orthmtxgenU(anglesU,muU);
                        W0T_i = permute(W0_i,[2 1 3]);
                        U0T_i = permute(U0_i,[2 1 3]);
                        for iblk = 1:nblks
                            dLdX(1:pt,:,iblk,iSample) = W0T_i(:,:,iblk)*dLdZ(1:pt,:,iblk,iSample);
                            dLdX(pt+1:pt+pb,:,iblk,iSample) = U0T_i(:,:,iblk)*dLdZ(pt+1:pt+pb,:,iblk,iSample);
                        end

                        % dLdTheta_i = <dLdZ,(dVdTheta_i)X>
                        fcn_orthmtxgen_diffW = tansacnet.lsun.get_fcn_orthmtxgen_diff(anglesW);
                        fcn_orthmtxgen_diffU = tansacnet.lsun.get_fcn_orthmtxgen_diff(anglesU);
                        dWPst = bsxfun(@times,permute(muW,[1 3 2]),W0_i);
                        dWPre = repmat(eye(pt,'like',W0_i),[1 1 nblks]);
                        dUPst = bsxfun(@times,permute(muU,[1 3 2]),U0_i);
                        dUPre = repmat(eye(pb,'like',U0_i),[1 1 nblks]);
                        for iAngle = uint32(1:nAnglesH)
                            [dW,dWPst,dWPre] = fcn_orthmtxgen_diffW(anglesW,muW,iAngle,dWPst,dWPre);
                            [dU,dUPst,dUPre] = fcn_orthmtxgen_diffU(anglesU,muU,iAngle,dUPst,dUPre);
                            for iblk = 1:nblks
                                d_top = dW(:,:,iblk)*X(1:pt,:,iblk,iSample);
                                d_btm = dU(:,:,iblk)*X(pt+1:pt+pb,:,iblk,iSample);
                                dLdTheta(iAngle,iblk,iSample) = sum(dLdZ(1:pt,:,iblk,iSample).*d_top,'all');
                                dLdTheta(nAnglesH+iAngle,iblk,iSample) = sum(dLdZ(pt+1:pt+pb,:,iblk,iSample).*d_btm,'all');
                            end
                        end
                    else
                        W0T_i = reshape(muW,pt,1,[]);
                        U0T_i = reshape(muU,pb,1,[]);
                        for iblk = 1:nblks
                            dLdX(1:pt,:,iblk,iSample) = W0T_i(:,:,iblk)*dLdZ(1:pt,:,iblk,iSample);
                            dLdX(pt+1:pt+pb,:,iblk,iSample) = U0T_i(:,:,iblk)*dLdZ(pt+1:pt+pb,:,iblk,iSample);
                        end
                    end
                end
            end
        end

        function mus = get.Mus(layer)
            mus = layer.PrivateMus;
        end

        function layer = set.Mus(layer,mus)
            nBlocks = prod(layer.NumberOfBlocks);
            pt = layer.PrivateNumberOfChannels(1);
            pb = layer.PrivateNumberOfChannels(2);
            if isempty(mus)
                mus = ones(pt+pb,nBlocks);
            elseif isscalar(mus)
                mus = mus*ones(pt+pb,nBlocks,'like',mus);
            end
            %
            layer.PrivateMus = mus;
        end

    end

end
