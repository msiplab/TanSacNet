classdef salsunInitialRotation2dLayer < tansacnet.lsun.lsunRotation2dLayerBase %#codegen
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

    properties (Hidden, Constant)
        Mode = 'Analysis'
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
            layer.Description = "SA-LSUN initial rotation (state-controlled) " ...
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
            [W0_,U0_] = layer.buildRotationMatrices(Theta);
            %
            Y = reshape(X,ps+pa,nrows*ncols,nSamples);
            Zs = zeros(ps,nrows*ncols,nSamples,'like',Y);
            Za = zeros(pa,nrows*ncols,nSamples,'like',Y);
            for iSample = 1:nSamples
                Zs(:,:,iSample) = layer.applyBlockwiseMatrix( ...
                    W0_(:,:,:,iSample), Y(1:ps,:,iSample));
                Za(:,:,iSample) = layer.applyBlockwiseMatrix( ...
                    U0_(:,:,:,iSample), Y(ps+1:end,:,iSample));
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
            nBlks = nrows*ncols;
            %
            [W0_,U0_,muW,muU,anglesW,anglesU] = layer.buildRotationMatrices(Theta);
            nAnglesH = size(anglesW,1);

            % dLdX = dZdX x dLdZ
            dldz_upp = reshape(dLdZ(1:ps,:,:,:),ps,nBlks,nSamples);
            dldz_low = reshape(dLdZ(ps+1:ps+pa,:,:,:),pa,nBlks,nSamples);
            c_upp = reshape(X(1:ps,:,:,:),ps,nBlks,nSamples);
            c_low = reshape(X(ps+1:ps+pa,:,:,:),pa,nBlks,nSamples);

            Zs = zeros(ps,nBlks,nSamples,'like',dLdZ);
            Za = zeros(pa,nBlks,nSamples,'like',dLdZ);
            dLdW = zeros(nAnglesH,nBlks,nSamples,'like',dLdZ);
            dLdU = zeros(nAnglesH,nBlks,nSamples,'like',dLdZ);
            for iSample = 1:nSamples
                W0T = permute(W0_(:,:,:,iSample),[2 1 3]);
                U0T = permute(U0_(:,:,:,iSample),[2 1 3]);
                Zs(:,:,iSample) = layer.applyBlockwiseMatrix( ...
                    W0T, dldz_upp(:,:,iSample));
                Za(:,:,iSample) = layer.applyBlockwiseMatrix( ...
                    U0T, dldz_low(:,:,iSample));

                % dLdWi = <dLdZ,(dVdWi)X>
                dLdW(:,:,iSample) = layer.computeAngleGradient( ...
                    W0_(:,:,:,iSample), anglesW(:,:,iSample), muW, ...
                    c_upp(:,:,iSample), dldz_upp(:,:,iSample));
                dLdU(:,:,iSample) = layer.computeAngleGradient( ...
                    U0_(:,:,:,iSample), anglesU(:,:,iSample), muU, ...
                    c_low(:,:,iSample), dldz_low(:,:,iSample));
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

    methods (Access = private)
        function [W0_,U0_,muW,muU,anglesW,anglesU] = buildRotationMatrices(layer,Theta)
            ps = layer.PrivateNumberOfChannels(1);
            pa = layer.PrivateNumberOfChannels(2);
            nAngles = size(Theta,1);
            nBlks = size(Theta,2);
            nSamples = size(Theta,3);
            anglesW = Theta(1:nAngles/2,:,:);
            anglesU = Theta(nAngles/2+1:end,:,:);
            mus = cast(layer.Mus,'like',Theta);
            muW = mus(1:ps,:);
            muU = mus(ps+1:end,:);
            %
            W0_ = zeros(ps,ps,nBlks,nSamples,'like',Theta);
            U0_ = zeros(pa,pa,nBlks,nSamples,'like',Theta);
            for iSample = 1:nSamples
                fcn_orthmtxgen = tansacnet.lsun.get_fcn_orthmtxgen(anglesW(:,:,iSample));
                W0_(:,:,:,iSample) = fcn_orthmtxgen(anglesW(:,:,iSample),muW);
                U0_(:,:,:,iSample) = fcn_orthmtxgen(anglesU(:,:,iSample),muU);
            end
        end
    end

end
