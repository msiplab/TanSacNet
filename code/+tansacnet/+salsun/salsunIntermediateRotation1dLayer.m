classdef salsunIntermediateRotation1dLayer < nnet.layer.Layer %#codegen
    %SALSUNINTERMEDIATEROTATION1DLAYER
    %
    %   Data-path input  'x'     : nChsTotal x 1 x nBlks x nSamples
    %
    %   Control-path input 'theta': nAngles x nBlks x nSamples
    %                               where nAngles = (nChsTotal-2)*nChsTotal/8
    %                               (angles may vary from sample to
    %                               sample, unlike tansacnet.lsun's
    %                               Angles, which only vary block by
    %                               block)
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
        Mode
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
        function layer = salsunIntermediateRotation1dLayer(varargin)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            p = inputParser;
            addParameter(p,'Stride',[])
            addParameter(p,'Name','')
            addParameter(p,'Mode','Synthesis')
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
            layer.Mode = p.Results.Mode;
            layer.Mus = p.Results.Mus;
            layer.Description = layer.Mode ...
                + " SA-LSUN intermediate rotation " ...
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
            %
            if ~strcmp(layer.Mode,'Analysis') && ~strcmp(layer.Mode,'Synthesis')
                throw(MException('SaLsunLayer:InvalidMode',...
                    '%s : Mode should be either of Synthesis or Analysis',...
                    layer.Mode))
            end
            isAnalysis = strcmp(layer.Mode,'Analysis');
            musU = cast(layer.PrivateMus,'like',Theta);

            Y = X;
            Yb = Y(pt+1:pt+pb,:,:,:);
            if isgpuarray(X)
                angles_ = reshape(Theta,nAngles,nblks*nSamples);
                musU_ = repmat(musU,[1 nSamples]);
                Yb_ext = reshape(Yb,pb,1,nblks*nSamples);
                if nAngles > 0
                    fcn_orthmtxgen = tansacnet.lsun.get_fcn_orthmtxgen(angles_);
                    Un = fcn_orthmtxgen(angles_,musU_);
                else
                    Un = reshape(musU_,pb,1,[]);
                end
                if isAnalysis
                    A_ = Un;
                else
                    A_ = permute(Un,[2 1 3]);
                end
                Zb = reshape(pagefun(@mtimes,A_,Yb_ext),pb,1,nblks,nSamples);
            else
                Zb = zeros(pb,1,nblks,nSamples,'like',Y);
                for iSample = 1:nSamples
                    anglesU = Theta(:,:,iSample);
                    if nAngles > 0
                        fcn_orthmtxgen = tansacnet.lsun.get_fcn_orthmtxgen(anglesU);
                        Un_i = fcn_orthmtxgen(anglesU,musU);
                    else
                        Un_i = reshape(musU,pb,1,[]);
                    end
                    if isAnalysis
                        A_ = Un_i;
                    else
                        A_ = permute(Un_i,[2 1 3]);
                    end
                    for iblk = 1:nblks
                        Zb(:,:,iblk,iSample) = A_(:,:,iblk)*Yb(:,:,iblk,iSample);
                    end
                end
            end
            Y(pt+1:pt+pb,:,:,:) = Zb;
            Z = Y;
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
            %
            isAnalysis = strcmp(layer.Mode,'Analysis');
            musU = cast(layer.PrivateMus,'like',Theta);
            if isgpuarray(X) && ~isgpuarray(dLdZ)
                dLdZ = gpuArray(dLdZ);
            end
            dLdX = dLdZ;
            dLdTheta = zeros(nAngles,nblks,nSamples,'like',dLdZ);
            cdLd_btm = dLdX(pt+1:pt+pb,:,:,:);
            dldz_btm = dLdZ(pt+1:pt+pb,:,:,:);
            c_btm = X(pt+1:pt+pb,:,:,:);
            if isgpuarray(X)
                angles_ = reshape(Theta,nAngles,nblks*nSamples);
                musU_ = repmat(musU,[1 nSamples]);
                cdLd_btm_ext = reshape(cdLd_btm,pb,1,nblks*nSamples);
                if nAngles > 0
                    fcn_orthmtxgen = tansacnet.lsun.get_fcn_orthmtxgen(angles_);
                    Un = fcn_orthmtxgen(angles_,musU_);
                    if isAnalysis
                        A_ = permute(Un,[2 1 3]);
                    else
                        A_ = Un;
                    end
                    cdLd_btm = reshape(pagefun(@mtimes,A_,cdLd_btm_ext),pb,1,nblks,nSamples);

                    % dLdTheta_i = <dLdZ,(dVdTheta_i)X>, batched across
                    % blocks and samples together.
                    c_btm_ext = reshape(c_btm,pb,1,nblks*nSamples);
                    fcn_orthmtxgen_diff = tansacnet.lsun.get_fcn_orthmtxgen_diff(angles_);
                    dUPst = bsxfun(@times,permute(musU_,[1 3 2]),Un);
                    dUPre = repmat(eye(pb,'like',Un),[1 1 nblks*nSamples]);
                    for iAngle = uint32(1:nAngles)
                        [dU,dUPst,dUPre] = fcn_orthmtxgen_diff(angles_,musU_,iAngle,dUPst,dUPre);
                        if isAnalysis
                            dU_ = dU;
                        else
                            dU_ = permute(dU,[2 1 3]);
                        end
                        d_btm_ext = pagefun(@mtimes,dU_,c_btm_ext);
                        d_btm = reshape(d_btm_ext,pb,1,nblks,nSamples);
                        dLdTheta(iAngle,:,:) = sum(bsxfun(@times,dldz_btm,d_btm),1);
                    end
                else
                    A_ = reshape(musU_,pb,1,[]);
                    cdLd_btm = reshape(pagefun(@mtimes,A_,cdLd_btm_ext),pb,1,nblks,nSamples);
                    % nAngles==0: no angle derivatives to accumulate.
                end
            else
                for iSample = 1:nSamples
                    anglesU = Theta(:,:,iSample);
                    if nAngles > 0
                        fcn_orthmtxgen = tansacnet.lsun.get_fcn_orthmtxgen(anglesU);
                        Un_i = fcn_orthmtxgen(anglesU,musU);
                        if isAnalysis
                            A_ = permute(Un_i,[2 1 3]);
                        else
                            A_ = Un_i;
                        end
                        for iblk = 1:nblks
                            cdLd_btm(:,:,iblk,iSample) = A_(:,:,iblk)*cdLd_btm(:,:,iblk,iSample);
                        end

                        % dLdTheta_i = <dLdZ,(dVdTheta_i)X>
                        fcn_orthmtxgen_diff = tansacnet.lsun.get_fcn_orthmtxgen_diff(anglesU);
                        dUPst = bsxfun(@times,permute(musU,[1 3 2]),Un_i);
                        dUPre = repmat(eye(pb,'like',Un_i),[1 1 nblks]);
                        for iAngle = uint32(1:nAngles)
                            [dU,dUPst,dUPre] = fcn_orthmtxgen_diff(anglesU,musU,iAngle,dUPst,dUPre);
                            if isAnalysis
                                dU_ = dU;
                            else
                                dU_ = permute(dU,[2 1 3]);
                            end
                            for iblk = 1:nblks
                                d_btm = dU_(:,:,iblk)*c_btm(:,:,iblk,iSample);
                                dLdTheta(iAngle,iblk,iSample) = sum(dldz_btm(:,:,iblk,iSample).*d_btm,'all');
                            end
                        end
                    else
                        A_ = reshape(musU,pb,1,[]);
                        for iblk = 1:nblks
                            cdLd_btm(:,:,iblk,iSample) = A_(:,:,iblk)*cdLd_btm(:,:,iblk,iSample);
                        end
                    end
                end
            end
            dLdX(pt+1:pt+pb,:,:,:) = cdLd_btm;
        end

        function mus = get.Mus(layer)
            mus = layer.PrivateMus;
        end

        function layer = set.Mus(layer,mus)
            nBlocks = prod(layer.NumberOfBlocks);
            pb = layer.PrivateNumberOfChannels(2);
            if isempty(mus)
                mus = ones(pb,nBlocks);
            elseif isscalar(mus)
                mus = mus*ones(pb,nBlocks,'like',mus);
            end
            %
            layer.PrivateMus = mus;
        end

    end

end
