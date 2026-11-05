classdef lsunIntermediateRotation1dLayer < nnet.layer.Layer %#codegen
    %LSUNINTERMEDIATEROTATION1DLAYER
    %
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
        function layer = lsunIntermediateRotation1dLayer(varargin)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            p = inputParser;
            addParameter(p,'Stride',[])
            addParameter(p,'Angles',[])
            addParameter(p,'Mus',[])
            addParameter(p,'Mode','Synthesis')
            addParameter(p,'Name','')
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
            layer.Angles = p.Results.Angles;
            layer.Mus = p.Results.Mus;
            layer.Description = layer.Mode ...
                + " LSUN intermediate rotation " ...
                + "(pt,pb) = (" ...
                + layer.PrivateNumberOfChannels(1) + "," ...
                + layer.PrivateNumberOfChannels(2) + ")";
            layer.Type = '';
            layer.Device = p.Results.Device;
            layer.DType = p.Results.DType;

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
            nblks = size(X,3);
            nSamples = size(X,4);
            pt = layer.PrivateNumberOfChannels(1);
            pb = layer.PrivateNumberOfChannels(2);
            if layer.isUpdateRequested
                layer = layer.updateParameters();
            end
            %
            Un_ = layer.Un;
            Y = X;
            Yb = Y(pt+1:pt+pb,:,:,:);
            if strcmp(layer.Mode,'Analysis')
                A_ = Un_;
            elseif strcmp(layer.Mode,'Synthesis')
                A_ = permute(Un_,[2 1 3]);
            else
                throw(MException('LsunLayer:InvalidMode',...
                    '%s : Mode should be either of Synthesis or Analysis',...
                    layer.Mode))
            end

            if isgpuarray(X)
                Zb = pagefun(@mtimes,A_,Yb);
            else
                Zb = zeros(pb,1,nblks,nSamples,'like',Y);
                for iSample = 1:nSamples
                    for iblk = 1:nblks
                        Zb(:,:,iblk,iSample) = A_(:,:,iblk)*Yb(:,:,iblk,iSample);
                    end
                end
            end
            Y(pt+1:pt+pb,:,:,:) = Zb;
            Z = Y;
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
            %
            if layer.isUpdateRequested
                layer = layer.updateParameters();
            end
            anglesU = layer.PrivateAngles;
            musU = cast(layer.PrivateMus,'like',anglesU);

            % dLdX = dZdX x dLdZ
            Un_ = layer.Un;
            dUnPst = bsxfun(@times,permute(musU,[1 3 2]),Un_);
            dUnPre = repmat(eye(pb,'like',Un_),[1 1 nblks]);

            %
            dLdX = reshape(dLdZ,pt+pb,1,nblks,nSamples);
            if strcmp(layer.Mode,'Analysis')
                A_ = permute(Un_,[2 1 3]);
            else
                A_ = Un_;
            end
            cdLd_low = dLdX(pt+1:pt+pb,:,:,:);
            if isgpuarray(X)
                cdLd_low = pagefun(@mtimes,A_,cdLd_low);
            else
                for iSample = 1:nSamples
                    for iblk = 1:nblks
                        cdLd_low(:,:,iblk,iSample) = A_(:,:,iblk)*cdLd_low(:,:,iblk,iSample);
                    end
                end
            end
            dLdX(pt+1:pt+pb,:,:,:) = cdLd_low;

            % dLdWi = <dLdZ,(dVdWi)X>
            fcn_orthmtxgen_diff = tansacnet.lsun.get_fcn_orthmtxgen_diff(anglesU);
            nAngles = size(anglesU,1);
            dLdW = zeros(nAngles,nblks,'like',dLdZ);
            dldz_low = dLdZ(pt+1:pt+pb,:,:,:);
            c_low = X(pt+1:pt+pb,:,:,:);
            for iAngle = uint32(1:nAngles)
                [dUn,dUnPst,dUnPre] = fcn_orthmtxgen_diff(anglesU,musU,iAngle,dUnPst,dUnPre);
                if strcmp(layer.Mode,'Analysis')
                    dA_ = dUn;
                else
                    dA_ = permute(dUn,[2 1 3]);
                end
                if isgpuarray(X)
                    d_low = pagefun(@mtimes,dA_,c_low);
                    dLdW(iAngle,:) = sum(bsxfun(@times,dldz_low,d_low),[1 4]);
                else
                    for iblk = 1:nblks
                        dA_iblk = dA_(:,:,iblk);
                        dldz_low_iblk = squeeze(dldz_low(:,:,iblk,:));
                        c_low_iblk = squeeze(c_low(:,:,iblk,:));
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
            layer.isUpdateRequested = true;
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
            layer.isUpdateRequested = true;
        end

        function layer = updateParameters(layer)
            anglesU = layer.PrivateAngles;
            musU = cast(layer.PrivateMus,'like',anglesU);
            if isrow(musU)
                musU = musU.';
            end
            if size(anglesU,1) > 0
                fcn_orthmtxgen = tansacnet.lsun.get_fcn_orthmtxgen(anglesU);
                layer.Un = fcn_orthmtxgen(anglesU,musU);
            else
                layer.Un = reshape(musU,1,1,[]);
            end
            layer.isUpdateRequested = false;
        end

    end

end
