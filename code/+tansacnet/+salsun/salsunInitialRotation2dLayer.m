classdef salsunInitialRotation2dLayer < nnet.layer.Layer %#codegen
    %SALSUNINITIALROTATION2DLAYER
    %
    %   Data-path input  'x'     : nChsTotal x nRows x nCols x nSamples
    %
    %   Control-path input 'theta': nAngles x (nRows*nCols) x nSamples
    %                               where nAngles = (nChsTotal-2)*nChsTotal/4
    %                               (first half: anglesW, second half: anglesU)
    %
    %   Output                   : nChsTotal x nRows x nCols x nSamples
    %
    % Requirements: MATLAB R2022a
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
        function layer = salsunInitialRotation2dLayer(varargin)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            p = inputParser;
            addParameter(p,'Stride',[])
            addParameter(p,'Name','')
            addParameter(p,'Mus',[])
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
            layer.Description = "SA-LSUN initial rotation " ...
                + "(ps,pa) = (" ...
                + layer.PrivateNumberOfChannels(1) + "," ...
                + layer.PrivateNumberOfChannels(2) + "), "  ...
                + "(mv,mh) = (" ...
                + layer.Stride(1) + "," ...
                + layer.Stride(2) + ")";
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
            %         Theta - Control-path input: [anglesW; anglesU],
            %                 nAngles x (nRows*nCols) x nSamples
            % Outputs:
            %         Z     - Output of layer forward function

            % Layer forward function for prediction goes here.

            nrows = size(X,2);
            ncols = size(X,3);
            nSamples = size(X,4);
            ps = layer.PrivateNumberOfChannels(1);
            pa = layer.PrivateNumberOfChannels(2);
            %
            nAngles = size(Theta,1);
            anglesW = Theta(1:nAngles/2,:,:);
            anglesU = Theta(nAngles/2+1:end,:,:);
            mus = cast(layer.Mus,'like',Theta);
            muW = mus(1:ps,:);
            muU = mus(ps+1:end,:);
            %
            Y = reshape(X,ps+pa,nrows*ncols,nSamples);
            Zs = zeros(ps,nrows*ncols,nSamples,'like',Y);
            Za = zeros(pa,nrows*ncols,nSamples,'like',Y);
            if isgpuarray(X)
                anglesW_ = reshape(anglesW,nAngles/2,nrows*ncols*nSamples);
                anglesU_ = reshape(anglesU,nAngles/2,nrows*ncols*nSamples);
                muW_ = repmat(muW,[1 nSamples]);
                muU_ = repmat(muU,[1 nSamples]);
                fcn_orthmtxgen = tansacnet.lsun.get_fcn_orthmtxgen(anglesW_);
                W0 = fcn_orthmtxgen(anglesW_,muW_);
                U0 = fcn_orthmtxgen(anglesU_,muU_);
                Ys_ext = reshape(Y(1:ps,:,:),ps,1,nrows*ncols*nSamples);
                Ya_ext = reshape(Y(ps+1:end,:,:),pa,1,nrows*ncols*nSamples);
                Zs = reshape(pagefun(@mtimes,W0,Ys_ext),ps,nrows*ncols,nSamples);
                Za = reshape(pagefun(@mtimes,U0,Ya_ext),pa,nrows*ncols,nSamples);
            else
                for iSample = 1:nSamples
                    fcn_orthmtxgen = tansacnet.lsun.get_fcn_orthmtxgen(anglesW(:,:,iSample));
                    W0_i = fcn_orthmtxgen(anglesW(:,:,iSample),muW);
                    U0_i = fcn_orthmtxgen(anglesU(:,:,iSample),muU);
                    for iblk = 1:nrows*ncols
                        Zs(:,iblk,iSample) = W0_i(:,:,iblk)*Y(1:ps,iblk,iSample);
                        Za(:,iblk,iSample) = U0_i(:,:,iblk)*Y(ps+1:end,iblk,iSample);
                    end
                end
            end
            Z = reshape([Zs;Za],ps+pa,nrows,ncols,nSamples);
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
            %                       ([dLdW; dLdU], fed back into the
            %                       control-path parameter estimator)

            nrows = size(dLdZ,2);
            ncols = size(dLdZ,3);
            nSamples = size(dLdZ,4);
            ps = layer.PrivateNumberOfChannels(1);
            pa = layer.PrivateNumberOfChannels(2);
            %
            if isgpuarray(X) && ~isgpuarray(dLdZ)
                dLdZ = gpuArray(dLdZ);
            end
            nAngles = size(Theta,1)/2;
            anglesW = Theta(1:nAngles,:,:);
            anglesU = Theta(nAngles+1:end,:,:);
            mus = cast(layer.Mus,'like',Theta);
            muW = mus(1:ps,:);
            muU = mus(ps+1:end,:);

            % dLdX = dZdX x dLdZ
            dldz_upp = reshape(dLdZ(1:ps,:,:,:),ps,nrows*ncols,nSamples);
            dldz_low = reshape(dLdZ(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            c_upp = reshape(X(1:ps,:,:,:),ps,nrows*ncols,nSamples);
            c_low = reshape(X(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);

            Zs = zeros(ps,nrows*ncols,nSamples,'like',dLdZ);
            Za = zeros(pa,nrows*ncols,nSamples,'like',dLdZ);
            dLdW = zeros(nAngles,nrows*ncols,nSamples,'like',dLdZ);
            dLdU = zeros(nAngles,nrows*ncols,nSamples,'like',dLdZ);
            if isgpuarray(X)
                anglesW_ = reshape(anglesW,nAngles,nrows*ncols*nSamples);
                anglesU_ = reshape(anglesU,nAngles,nrows*ncols*nSamples);
                muW_ = repmat(muW,[1 nSamples]);
                muU_ = repmat(muU,[1 nSamples]);
                fcn_orthmtxgen = tansacnet.lsun.get_fcn_orthmtxgen(anglesW_);
                W0 = fcn_orthmtxgen(anglesW_,muW_);
                U0 = fcn_orthmtxgen(anglesU_,muU_);
                W0T = permute(W0,[2 1 3]);
                U0T = permute(U0,[2 1 3]);
                dldz_upp_ext = reshape(dldz_upp,ps,1,nrows*ncols*nSamples);
                dldz_low_ext = reshape(dldz_low,pa,1,nrows*ncols*nSamples);
                Zs = reshape(pagefun(@mtimes,W0T,dldz_upp_ext),ps,nrows*ncols,nSamples);
                Za = reshape(pagefun(@mtimes,U0T,dldz_low_ext),pa,nrows*ncols,nSamples);

                % dLdWi = <dLdZ,(dVdWi)X>, batched across blocks and
                % samples together.
                c_upp_ext = reshape(c_upp,ps,1,nrows*ncols*nSamples);
                c_low_ext = reshape(c_low,pa,1,nrows*ncols*nSamples);
                fcn_orthmtxgen_diff = tansacnet.lsun.get_fcn_orthmtxgen_diff(anglesW_);
                dWPst = bsxfun(@times,permute(muW_,[1 3 2]),W0);
                dWPre = repmat(eye(ps,'like',W0),[1 1 nrows*ncols*nSamples]);
                dUPst = bsxfun(@times,permute(muU_,[1 3 2]),U0);
                dUPre = repmat(eye(pa,'like',U0),[1 1 nrows*ncols*nSamples]);
                for iAngle = uint32(1:nAngles)
                    [dW,dWPst,dWPre] = fcn_orthmtxgen_diff(anglesW_,muW_,iAngle,dWPst,dWPre);
                    [dU,dUPst,dUPre] = fcn_orthmtxgen_diff(anglesU_,muU_,iAngle,dUPst,dUPre);
                    d_upp = reshape(pagefun(@mtimes,dW,c_upp_ext),ps,nrows*ncols,nSamples);
                    d_low = reshape(pagefun(@mtimes,dU,c_low_ext),pa,nrows*ncols,nSamples);
                    dLdW(iAngle,:,:) = sum(bsxfun(@times,dldz_upp,d_upp),1);
                    dLdU(iAngle,:,:) = sum(bsxfun(@times,dldz_low,d_low),1);
                end
            else
                for iSample = 1:nSamples
                    angW = anglesW(:,:,iSample);
                    angU = anglesU(:,:,iSample);
                    fcn_orthmtxgen = tansacnet.lsun.get_fcn_orthmtxgen(angW);
                    W0_i = fcn_orthmtxgen(angW,muW);
                    U0_i = fcn_orthmtxgen(angU,muU);
                    W0T = permute(W0_i,[2 1 3]);
                    U0T = permute(U0_i,[2 1 3]);
                    for iblk = 1:nrows*ncols
                        Zs(:,iblk,iSample) = W0T(:,:,iblk)*dldz_upp(:,iblk,iSample);
                        Za(:,iblk,iSample) = U0T(:,:,iblk)*dldz_low(:,iblk,iSample);
                    end

                    % dLdWi = <dLdZ,(dVdWi)X>
                    fcn_orthmtxgen_diff = tansacnet.lsun.get_fcn_orthmtxgen_diff(angW);
                    dWPst = bsxfun(@times,permute(muW,[1 3 2]),W0_i);
                    dWPre = repmat(eye(ps,'like',W0_i),[1 1 nrows*ncols]);
                    dUPst = bsxfun(@times,permute(muU,[1 3 2]),U0_i);
                    dUPre = repmat(eye(pa,'like',U0_i),[1 1 nrows*ncols]);
                    for iAngle = uint32(1:nAngles)
                        [dW,dWPst,dWPre] = fcn_orthmtxgen_diff(angW,muW,iAngle,dWPst,dWPre);
                        [dU,dUPst,dUPre] = fcn_orthmtxgen_diff(angU,muU,iAngle,dUPst,dUPre);
                        for iblk = 1:nrows*ncols
                            d_upp = dW(:,:,iblk)*c_upp(:,iblk,iSample);
                            d_low = dU(:,:,iblk)*c_low(:,iblk,iSample);
                            dLdW(iAngle,iblk,iSample) = sum(dldz_upp(:,iblk,iSample).*d_upp);
                            dLdU(iAngle,iblk,iSample) = sum(dldz_low(:,iblk,iSample).*d_low);
                        end
                    end
                end
            end
            dLdX = reshape([Zs;Za],ps+pa,nrows,ncols,nSamples);
            dLdTheta = cat(1,dLdW,dLdU);
        end

        function mus = get.Mus(layer)
            mus = layer.PrivateMus;
        end

        function layer = set.Mus(layer,mus)
            nBlocks = prod(layer.NumberOfBlocks);
            ps = layer.PrivateNumberOfChannels(1);
            pa = layer.PrivateNumberOfChannels(2);
            %
            if isempty(mus)
                mus = ones(ps+pa,nBlocks);
            elseif isscalar(mus)
                mus = mus*ones(ps+pa,nBlocks,'like',mus);
            end
            %
            layer.PrivateMus = mus;
        end

    end

end
