classdef salsunInitialFullRotation1dLayer < nnet.layer.Layer %#codegen
    %SALSUNINITIALFULLROTATION1DLAYER
    %
    %   Data-path input  'x'     : nChsTotal x 1 x nBlks x nSamples
    %
    %   Control-path input 'theta': nAngles x nBlks x nSamples
    %                               where nAngles = (nChsTotal-1)*nChsTotal/2
    %                               The 1-D initial/final rotation is not
    %                               split into separate ps/pa blocks)
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
        function layer = salsunInitialFullRotation1dLayer(varargin)
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
            layer.Description = "SA-LSUN initial full rotation" ...
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
            %         Theta - Control-path input: full angle vector,
            %                 nAngles x nBlks x nSamples
            % Outputs:
            %         Z     - Output of layer forward function

            nChsTotal = sum(layer.PrivateNumberOfChannels);
            nblks = size(X,3);
            nSamples = size(X,4);
            nAngles = size(Theta,1);
            mus = cast(layer.Mus,'like',Theta);

            if isgpuarray(X)
                angles_ = reshape(Theta,nAngles,nblks*nSamples);
                mus_ = repmat(mus,[1 nSamples]);
                fcn_orthmtxgen = tansacnet.lsun.get_fcn_orthmtxgen(angles_);
                V0 = fcn_orthmtxgen(angles_,mus_);
                X_ext = reshape(X,nChsTotal,1,nblks*nSamples);
                Z = reshape(pagefun(@mtimes,V0,X_ext),nChsTotal,1,nblks,nSamples);
            else
                Z = zeros(nChsTotal,1,nblks,nSamples,'like',X);
                for iSample = 1:nSamples
                    angles = Theta(:,:,iSample);
                    fcn_orthmtxgen = tansacnet.lsun.get_fcn_orthmtxgen(angles);
                    V0_i = fcn_orthmtxgen(angles,mus);
                    for iblk = 1:nblks
                        Z(:,:,iblk,iSample) = V0_i(:,:,iblk)*X(:,:,iblk,iSample);
                    end
                end
            end
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

            nChsTotal = sum(layer.PrivateNumberOfChannels);
            nblks = size(dLdZ,3);
            nSamples = size(dLdZ,4);
            nAngles = size(Theta,1);
            mus = cast(layer.Mus,'like',Theta);

            if isgpuarray(X) && ~isgpuarray(dLdZ)
                dLdZ = gpuArray(dLdZ);
            end

            dLdX = zeros(nChsTotal,1,nblks,nSamples,'like',dLdZ);
            dLdTheta = zeros(nAngles,nblks,nSamples,'like',dLdZ);
            if isgpuarray(X)
                angles_ = reshape(Theta,nAngles,nblks*nSamples);
                mus_ = repmat(mus,[1 nSamples]);
                fcn_orthmtxgen = tansacnet.lsun.get_fcn_orthmtxgen(angles_);
                V0 = fcn_orthmtxgen(angles_,mus_);
                V0T = permute(V0,[2 1 3]);
                dLdZ_ext = reshape(dLdZ,nChsTotal,1,nblks*nSamples);
                dLdX = reshape(pagefun(@mtimes,V0T,dLdZ_ext),nChsTotal,1,nblks,nSamples);

                % dLdTheta_i = <dLdZ,(dVdTheta_i)X>, batched across blocks
                % and samples together.
                X_ext = reshape(X,nChsTotal,1,nblks*nSamples);
                fcn_orthmtxgen_diff = tansacnet.lsun.get_fcn_orthmtxgen_diff(angles_);
                dPst = bsxfun(@times,permute(mus_,[1 3 2]),V0);
                dPre = repmat(eye(nChsTotal,'like',V0),[1 1 nblks*nSamples]);
                for iAngle = uint32(1:nAngles)
                    [dA,dPst,dPre] = fcn_orthmtxgen_diff(angles_,mus_,iAngle,dPst,dPre);
                    d_ext = pagefun(@mtimes,dA,X_ext);
                    d_ = reshape(d_ext,nChsTotal,1,nblks,nSamples);
                    dLdTheta(iAngle,:,:) = sum(bsxfun(@times,dLdZ,d_),1);
                end
            else
                for iSample = 1:nSamples
                    angles = Theta(:,:,iSample);
                    fcn_orthmtxgen = tansacnet.lsun.get_fcn_orthmtxgen(angles);
                    V0_i = fcn_orthmtxgen(angles,mus);
                    V0T = permute(V0_i,[2 1 3]);
                    for iblk = 1:nblks
                        dLdX(:,:,iblk,iSample) = V0T(:,:,iblk)*dLdZ(:,:,iblk,iSample);
                    end

                    % dLdTheta_i = <dLdZ,(dVdTheta_i)X>
                    fcn_orthmtxgen_diff = tansacnet.lsun.get_fcn_orthmtxgen_diff(angles);
                    dPst = bsxfun(@times,permute(mus,[1 3 2]),V0_i);
                    dPre = repmat(eye(nChsTotal,'like',V0_i),[1 1 nblks]);
                    for iAngle = uint32(1:nAngles)
                        [dA,dPst,dPre] = fcn_orthmtxgen_diff(angles,mus,iAngle,dPst,dPre);
                        for iblk = 1:nblks
                            d_ = dA(:,:,iblk)*X(:,:,iblk,iSample);
                            dLdTheta(iAngle,iblk,iSample) = sum(dLdZ(:,:,iblk,iSample).*d_,'all');
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
            nChsTotal = sum(layer.PrivateNumberOfChannels);
            %
            if isempty(mus)
                mus = ones(nChsTotal,nBlocks);
            elseif isscalar(mus)
                mus = mus*ones(nChsTotal,nBlocks,'like',mus);
            end
            %
            layer.PrivateMus = mus;
        end

    end

end
