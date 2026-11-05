classdef salsunIntermediateRotation2dLayer < nnet.layer.Layer %#codegen
    %SALSUNINTERMEDIATEROTATION2DLAYER
    %
    %   Data-path input  'x'     : nChsTotal x nRows x nCols x nSamples
    %
    %   Control-path input 'theta': nAngles x (nRows*nCols) x nSamples
    %                               where nAngles = (nChsTotal-2)*nChsTotal/8
    %                               (angles may vary from sample to
    %                               sample, unlike tansacnet.lsun's
    %                               Angles, which only vary block by
    %                               block)
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
        function layer = salsunIntermediateRotation2dLayer(varargin)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            p = inputParser;
            addParameter(p,'Stride',[])
            addParameter(p,'Mus',[])
            addParameter(p,'Mode','Synthesis')
            addParameter(p,'Name','')
            addParameter(p,'NumberOfBlocks',[1 1])
            addParameter(p,'DType','double')
            addParameter(p,'Device','cuda')
            parse(p,varargin{:})

            % Layer constructor function goes here.
            layer.Stride = p.Results.Stride;
            layer.NumberOfBlocks = p.Results.NumberOfBlocks;
            layer.PrivateNumberOfChannels = [ceil(prod(layer.Stride)/2) floor(prod(layer.Stride)/2)];
            layer.Name = p.Results.Name;
            layer.Mode = p.Results.Mode;
            layer.Mus = p.Results.Mus;
            layer.Description = layer.Mode ...
                + " SA-LSUN intermediate rotation " ...
                + "(ps,pa) = (" ...
                + layer.PrivateNumberOfChannels(1) + "," ...
                + layer.PrivateNumberOfChannels(2) + ")";
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

            nrows = size(X,2);
            ncols = size(X,3);
            nSamples = size(X,4);
            ps = layer.PrivateNumberOfChannels(1);
            pa = layer.PrivateNumberOfChannels(2);
            %
            if ~strcmp(layer.Mode,'Analysis') && ~strcmp(layer.Mode,'Synthesis')
                throw(MException('LsunLayer:InvalidMode',...
                    '%s : Mode should be either of Synthesis or Analysis',...
                    layer.Mode))
            end
            isAnalysis = strcmp(layer.Mode,'Analysis');
            musU = cast(layer.PrivateMus,'like',Theta);

            Y = X;
            Ya = reshape(Y(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            Za = zeros(pa,nrows*ncols,nSamples,'like',Y);
            if isgpuarray(X)
                nAngles = size(Theta,1);
                angles_ = reshape(Theta,nAngles,nrows*ncols*nSamples);
                mus_ = repmat(musU,[1 nSamples]);
                fcn_orthmtxgen = tansacnet.lsun.get_fcn_orthmtxgen(angles_);
                Un = fcn_orthmtxgen(angles_,mus_);
                if isAnalysis
                    A_ = Un;
                else
                    A_ = permute(Un,[2 1 3]);
                end
                Ya_ext = reshape(Ya,pa,1,nrows*ncols*nSamples);
                Za = reshape(pagefun(@mtimes,A_,Ya_ext),pa,nrows*ncols,nSamples);
            else
                for iSample = 1:nSamples
                    anglesU = Theta(:,:,iSample);
                    fcn_orthmtxgen = tansacnet.lsun.get_fcn_orthmtxgen(anglesU);
                    Un_i = fcn_orthmtxgen(anglesU,musU);
                    if isAnalysis
                        A_ = Un_i;
                    else
                        A_ = permute(Un_i,[2 1 3]);
                    end
                    for iblk = 1:nrows*ncols
                        Za(:,iblk,iSample) = A_(:,:,iblk)*Ya(:,iblk,iSample);
                    end
                end
            end
            Y(ps+1:ps+pa,:,:,:) = reshape(Za,pa,nrows,ncols,nSamples);
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

            nrows = size(dLdZ,2);
            ncols = size(dLdZ,3);
            nSamples = size(dLdZ,4);
            ps = layer.PrivateNumberOfChannels(1);
            pa = layer.PrivateNumberOfChannels(2);
            %
            isAnalysis = strcmp(layer.Mode,'Analysis');
            musU = cast(layer.PrivateMus,'like',Theta);

            if isgpuarray(X) && ~isgpuarray(dLdZ)
                dLdZ = gpuArray(dLdZ);
            end

            % dLdX = dZdX x dLdZ
            dLdX = reshape(dLdZ,ps+pa,nrows,ncols,nSamples);
            cdLd_low = reshape(dLdX(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            dldz_low = reshape(dLdZ(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            c_low = reshape(X(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            nAngles = size(Theta,1);
            dLdTheta = zeros(nAngles,nrows*ncols,nSamples,'like',dLdZ);
            if isgpuarray(X)
                angles_ = reshape(Theta,nAngles,nrows*ncols*nSamples);
                mus_ = repmat(musU,[1 nSamples]);
                fcn_orthmtxgen = tansacnet.lsun.get_fcn_orthmtxgen(angles_);
                Un = fcn_orthmtxgen(angles_,mus_);
                if isAnalysis
                    A_ = permute(Un,[2 1 3]);
                else
                    A_ = Un;
                end
                cdLd_low_ext = reshape(cdLd_low,pa,1,nrows*ncols*nSamples);
                cdLd_low = reshape(pagefun(@mtimes,A_,cdLd_low_ext),pa,nrows*ncols,nSamples);

                % dLdTheta_i = <dLdZ,(dVdTheta_i)X>, batched across
                % blocks and samples together.
                c_low_ext = reshape(c_low,pa,1,nrows*ncols*nSamples);
                fcn_orthmtxgen_diff = tansacnet.lsun.get_fcn_orthmtxgen_diff(angles_);
                dUPst = bsxfun(@times,permute(mus_,[1 3 2]),Un);
                dUPre = repmat(eye(pa,'like',Un),[1 1 nrows*ncols*nSamples]);
                for iAngle = uint32(1:nAngles)
                    [dU,dUPst,dUPre] = fcn_orthmtxgen_diff(angles_,mus_,iAngle,dUPst,dUPre);
                    if isAnalysis
                        dU_ = dU;
                    else
                        dU_ = permute(dU,[2 1 3]);
                    end
                    d_low = reshape(pagefun(@mtimes,dU_,c_low_ext),pa,nrows*ncols,nSamples);
                    dLdTheta(iAngle,:,:) = sum(bsxfun(@times,dldz_low,d_low),1);
                end
            else
                for iSample = 1:nSamples
                    anglesU = Theta(:,:,iSample);
                    fcn_orthmtxgen = tansacnet.lsun.get_fcn_orthmtxgen(anglesU);
                    Un_i = fcn_orthmtxgen(anglesU,musU);
                    if isAnalysis
                        A_ = permute(Un_i,[2 1 3]);
                    else
                        A_ = Un_i;
                    end
                    for iblk = 1:nrows*ncols
                        cdLd_low(:,iblk,iSample) = A_(:,:,iblk)*cdLd_low(:,iblk,iSample);
                    end

                    % dLdTheta_i = <dLdZ,(dVdTheta_i)X>
                    fcn_orthmtxgen_diff = tansacnet.lsun.get_fcn_orthmtxgen_diff(anglesU);
                    dUPst = bsxfun(@times,permute(musU,[1 3 2]),Un_i);
                    dUPre = repmat(eye(pa,'like',Un_i),[1 1 nrows*ncols]);
                    for iAngle = uint32(1:nAngles)
                        [dU,dUPst,dUPre] = fcn_orthmtxgen_diff(anglesU,musU,iAngle,dUPst,dUPre);
                        if isAnalysis
                            dU_ = dU;
                        else
                            dU_ = permute(dU,[2 1 3]);
                        end
                        for iblk = 1:nrows*ncols
                            d_low = dU_(:,:,iblk)*c_low(:,iblk,iSample);
                            dLdTheta(iAngle,iblk,iSample) = sum(dldz_low(:,iblk,iSample).*d_low);
                        end
                    end
                end
            end
            dLdX(ps+1:ps+pa,:,:,:) = reshape(cdLd_low,pa,nrows,ncols,nSamples);
        end

        function mus = get.Mus(layer)
            mus = layer.PrivateMus;
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
        end

    end

end
