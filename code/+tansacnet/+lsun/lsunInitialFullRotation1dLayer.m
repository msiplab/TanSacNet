classdef lsunInitialFullRotation1dLayer < nnet.layer.Layer %#codegen
    %LSUNINITIALFULLROTATION1DLAYER
    %
    %   コンポーネント別に入力(nComponents=1のみサポート):
    %      nChs x 1 x nBlks x nSamples
    %
    %   コンポーネント別に出力(nComponents=1のみサポート):
    %      nChs x 1 x nBlks x nSamples
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
        NumberOfBlocks
    end
    
    properties (Dependent)
        %NoDcLeakage
    end
    
    properties (Dependent)
        Mus
    end
    
    properties (Learnable,Dependent)
        Angles
    end
    
    properties (Access = private)
        PrivateNumberOfChannels
        %PrivateNoDcLeakage
        PrivateAngles
        PrivateMus
        isUpdateRequested
    end
    
    properties (Hidden)
        V0
    end
        
    methods
        
        function layer = lsunInitialFullRotation1dLayer(varargin)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            p = inputParser;
            addParameter(p,'Stride',[])
            addParameter(p,'Name','')
            addParameter(p,'Mus',[])
            addParameter(p,'Angles',[])
            %addParameter(p,'NoDcLeakage',false)
            addParameter(p,'NumberOfBlocks',1)
            parse(p,varargin{:})
            
            % Layer constructor function goes here.
            layer.Stride = p.Results.Stride;
            layer.NumberOfBlocks = p.Results.NumberOfBlocks;
            layer.PrivateNumberOfChannels = [ceil(layer.Stride/2) floor(layer.Stride/2)];
            layer.Name = p.Results.Name;
            layer.Mus = p.Results.Mus;
            layer.Angles = p.Results.Angles;
            %layer.NoDcLeakage = p.Results.NoDcLeakage;
            layer.Description = "LSUN initial full rotation " ...
                + "(pt,pb) = (" ...
                + layer.PrivateNumberOfChannels(1) + "," ...
                + layer.PrivateNumberOfChannels(2) + "), "  ...
                + "m = " + layer.Stride;
            layer.Type = '';

            nChsTotal = sum(layer.PrivateNumberOfChannels);            
            nAngles = (nChsTotal-1)*nChsTotal/2;
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
            
            nChsTotal = sum(layer.PrivateNumberOfChannels);
            nSamples = size(X,4);
            nblks = size(X,3);            
            % Update parameters
            if layer.isUpdateRequested
                layer = layer.updateParameters();
            end
            %
            V0_ = layer.V0;
            %Y = reshape(permute(X,[3 1 2 4]),pt+pb,nrows*ncols*nSamples);
            %Y = permute(X,[1 3 2]);
            Z = zeros(nChsTotal,1,nblks,nSamples,'like',X);
            for iSample = 1:nSamples
                if isgpuarray(X)
                    Y_iSample = X(:,:,:,iSample);
                    Z_iSample = pagefun(@mtimes,V0_,Y_iSample);
                    Z(:,:,:,iSample) = reshape(Z_iSample,nChsTotal,1,nblks);
                else
                    for iblk = 1:nblks
                        Z(:,:,iblk,iSample) = V0_(:,:,iblk)*X(:,:,iblk,iSample);
                    end
                end
            end
            
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
            %
            %import tansacnet.lsun.get_fcn_orthmtxgen_diff

            nChsTotal = sum(layer.PrivateNumberOfChannels);
            nSamples = size(dLdZ,4);            
            nblks = size(dLdZ,3);            
            %{
            if isempty(layer.Mus)
                layer.Mus = ones(pt+pb,1);
            elseif isscalar(layer.Mus)
                layer.Mus = layer.Mus*ones(pt+pb,1);
            end
            if layer.NoDcLeakage
                layer.Mus(1) = 1;
                layer.Angles(1:pt-1) = ...
                    zeros(pt-1,1,'like',layer.Angles);
            end
            % Extend Angle paremeters for every block
            if size(layer.PrivateAngles,2) == 1
                layer.Angles = repmat(layer.PrivateAngles,[1 (nrows*ncols)]);
            end
            if size(layer.PrivateMus,2) == 1
                layer.Mus = repmat(layer.PrivateMus,[1 (nrows*ncols)]);
            end
            %}
            if layer.isUpdateRequested
                layer = layer.updateParameters();
            end
            angles = layer.PrivateAngles;
            nAngles = size(angles,1);
            mus = cast(layer.Mus,'like',angles);            
            V0_ = layer.V0; %transpose(fcn_orthmtxgen(anglesW,muW,0));
            V0T = permute(V0_,[2 1 3]);
            dV0Pst = bsxfun(@times,permute(mus,[1 3 2]),V0_);                        
            %dV0Pst = zeros(size(V0_),'like',V0_);
            %for iblk = 1:nblks
            %    dV0Pst(:,:,iblk) = bsxfun(@times,mus(:,iblk),V0_(:,:,iblk));
            %end
            dV0Pre = repmat(eye(nChsTotal,'like',V0_),[1 1 nblks]);

            % Layer backward function goes here.
            % dLdX = dZdX x dLdZ
            Y = dLdZ;
            for iSample = 1:nSamples
                Y_iSample = Y(:,:,:,iSample);
                if isgpuarray(X)
                    Y_iSample = pagefun(@mtimes,V0T,Y_iSample);
                else
                    for iblk = 1:nblks
                        Y_iSample(:,:,iblk) = V0T(:,:,iblk)*Y_iSample(:,:,iblk);
                    end
                end
                Y(:,:,:,iSample) = Y_iSample;
            end
            dLdX = Y;

            % dLdWi = <dLdZ,(dVdWi)X>
            fcn_orthmtxgen_diff = tansacnet.lsun.get_fcn_orthmtxgen_diff(angles);                        
            dLdW = zeros(nAngles,nblks,'like',dLdZ);
            dldz_ = dLdZ;
            % (dVdWi)X
            c_ = X;
            for iAngle = uint32(1:nAngles)
                [dV0,dV0Pst,dV0Pre] = fcn_orthmtxgen_diff(angles,mus,iAngle,dV0Pst,dV0Pre);
                if isgpuarray(X)
                    c_ext = c_;
                    d_ext = pagefun(@mtimes,dV0,c_ext); % idx 1 iblk iSample
                    d_ = d_ext;
                    dLdW(iAngle,:) = sum(bsxfun(@times,dldz_,d_),[1 4]);
                else
                    for iblk = 1:nblks
                        dldz_iblk = squeeze(dldz_(:,:,iblk,:));
                        c_iblk = squeeze(c_(:,:,iblk,:));
                        d_iblk = zeros(size(c_iblk),'like',c_iblk);
                        for iSample = 1:nSamples
                            d_iblk(:,iSample) = dV0(:,:,iblk)*c_iblk(:,iSample);
                        end
                        dLdW(iAngle,iblk) = sum(bsxfun(@times,dldz_iblk,d_iblk),'all');
                    end
                end
            end
        end

        %function nodcleak = get.NoDcLeakage(layer)
        %    nodcleak = layer.PrivateNoDcLeakage;
        %end

        function angles = get.Angles(layer)
            angles = layer.PrivateAngles;
        end
        
        function mus = get.Mus(layer)
            mus = layer.PrivateMus;
        end
        
        %function layer = set.NoDcLeakage(layer,nodcleak)
        %    layer.PrivateNoDcLeakage = nodcleak;
            %
        %    layer.isUpdateRequested = true;
        %end
        
        function layer = set.Angles(layer,angles)
            nBlocks = prod(layer.NumberOfBlocks);
            nChsTotal = sum(layer.PrivateNumberOfChannels);
            nAngles = (nChsTotal-1)*nChsTotal/2;
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
            nChsTotal = sum(layer.PrivateNumberOfChannels);
            %
            if isempty(mus)
                mus = ones(nChsTotal,nBlocks);
            elseif isscalar(mus)
                mus = mus*ones(nChsTotal,nBlocks);
            end
            %
            layer.PrivateMus = mus;
            %layer = layer.updateParameters();
            layer.isUpdateRequested = true;
        end
        
        function layer = updateParameters(layer)
            %import tansacnet.lsun.get_fcn_orthmtxgen
            %pt = layer.PrivateNumberOfChannels(1);
            %pb = layer.PrivateNumberOfChannels(2);
            %{
            if layer.NoDcLeakage
                layer.PrivateMus(1,:) = ones(1,size(layer.PrivateMus,2));                
                layer.PrivateAngles(1:pt-1,:) = ...
                    zeros(pt-1,size(layer.PrivateAngles,2),'like',layer.PrivateAngles);
            end            
            %}
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
            if nAngles > 0
                fcn_orthmtxgen = tansacnet.lsun.get_fcn_orthmtxgen(angles);
                layer.V0 = fcn_orthmtxgen(angles,mus);
            else
                layer.V0 = reshape(mus,1,1,[]);
            end
            %size(layer.V0,3) 
            %disp(layer.NumberOfBlocks)
            %assert(size(layer.V0,3) == layer.NumberOfBlocks)
            layer.isUpdateRequested = false;
        end
        
    end
    
end

