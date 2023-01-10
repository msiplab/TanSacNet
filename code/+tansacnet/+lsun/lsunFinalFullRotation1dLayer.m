classdef lsunFinalFullRotation1dLayer < nnet.layer.Layer %#codegen
    %LSUNFINALFULLROTATION1DLAYER
    %
    %   コンポーネント別に入力(nComponents=1のみサポート):
    %      nChs x nSamples x nBlks
    %
    %   コンポーネント別に出力(nComponents=1のみサポート):
    %      nChs x nSamples x nBlks
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
        V0T
    end
    
    methods
        function layer = lsunFinalFullRotation1dLayer(varargin)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            p = inputParser;
            addParameter(p,'Stride',[])
            addParameter(p,'Mus',[])
            addParameter(p,'Angles',[])
            addParameter(p,'Name','')
            %addParameter(p,'NoDcLeakage',false)
            addParameter(p,'NumberOfBlocks',1)
            parse(p,varargin{:})

            % Layer constructor function goes here.
            layer.Stride = p.Results.Stride;
            layer.NumberOfBlocks = p.Results.NumberOfBlocks;
            layer.PrivateNumberOfChannels = [ceil(prod(layer.Stride)/2) floor(prod(layer.Stride)/2)];
            layer.Mus = p.Results.Mus;
            layer.Angles = p.Results.Angles;
            %layer.NoDcLeakage = p.Results.NoDcLeakage;
            layer.Name = p.Results.Name;
            layer.Description = "LSUN final full rotation " ...
                + "(ps,pa) = (" ...
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
            
            nSamples = size(X,2);            
            nblks = size(X,3);    
            nChsTotal = sum(layer.PrivateNumberOfChannels);
            if layer.isUpdateRequested
                layer = layer.updateParameters();
            end
            %
            V0T_ = layer.V0T;
            Y = permute(X,[1 3 2]);
            Z_ = zeros(nChsTotal,nblks,nSamples,'like',Y);
            for iSample = 1:nSamples
                if isgpuarray(X)
                    Y_iSample = permute(Y(:,:,iSample),[1 4 2 3]);
                    Z_iSample = pagefun(@mtimes,V0T_,Y_iSample);
                    Z_(:,:,iSample) = ipermute(Z_iSample,[1 4 2 3]);
                else
                    for iblk = 1:nblks
                        Z_(:,iblk,iSample) = V0T_(:,:,iblk)*Y(:,iblk,iSample);
                    end
                end
            end
            Z = ipermute(reshape(Z_,nChsTotal,nblks,nSamples),...
                [1 3 2]);

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
            
            nSamples = size(dLdZ,2);
            nblks = size(dLdZ,3);            
            nChsTotal = sum(layer.PrivateNumberOfChannels);            
            %{
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
            V0_T = layer.V0T; %transpose(fcn_orthmtxgen(anglesW,muW,0));
            V0 = permute(V0_T,[2 1 3]);
            dV0Pst = zeros(size(V0),'like',V0);
            for iblk = 1:nblks
                dV0Pst(:,:,iblk) = bsxfun(@times,mus(:,iblk),V0(:,:,iblk));
            end
            dV0Pre = repmat(eye(nChsTotal,'like',V0),[1 1 nblks]);
            
            % Layer backward function goes here.
            % dLdX = dZdX x dLdZ
            adldz_ = permute(dLdZ,[1 3 2]);
            cdLd_ = reshape(adldz_,nChsTotal,nblks,nSamples);
            for iSample = 1:nSamples
                if isgpuarray(X)
                    cdLd_iSample = permute(cdLd_(:,:,iSample),[1 4 2 3]);
                    cdLd_iSample = pagefun(@mtimes,V0,cdLd_iSample);
                    cdLd_(:,:,iSample) = ipermute(cdLd_iSample,[1 4 2 3]);
                else
                    for iblk = 1:nblks
                        cdLd_(:,iblk,iSample) = V0(:,:,iblk)*cdLd_(:,iblk,iSample);
                    end
                end
            end
            dLdX = ipermute(cdLd_,[1 3 2]);
            
            % dLdWi = <dLdZ,(dVdWi)X>
            fcn_orthmtxgen_diff = tansacnet.lsun.get_fcn_orthmtxgen_diff(angles);            
            dLdW = zeros(nAngles,nblks,'like',dLdZ);
            dldz_ = permute(dLdZ,[1 3 2]);
            c_ = permute(X,[1 3 2]);
            for iAngle = uint32(1:nAngles)
                [dV0,dV0Pst,dV0Pre] = fcn_orthmtxgen_diff(angles,mus,iAngle,dV0Pst,dV0Pre);
                dV0_T = permute(dV0,[2 1 3]);
                if isgpuarray(X)
                    c_ext = permute(c_,[1 4 2 3]); % idx 1 iblk iSample
                    d_ext = pagefun(@mtimes,dV0_T,c_ext); % idx 1 iblk iSample
                    d_ = ipermute(d_ext,[1 4 2 3]);
                    dLdW(iAngle,:) = sum(bsxfun(@times,dldz_,d_),[1 3]);
                else
                    for iblk = 1:nblks
                        dldz_iblk = squeeze(dldz_(:,iblk,:));
                        c_iblk = squeeze(c_(:,iblk,:));
                        d_iblk = zeros(size(c_iblk),'like',c_iblk);
                        for iSample = 1:nSamples
                            d_iblk(:,iSample) = dV0_T(:,:,iblk)*c_iblk(:,iSample);
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
            ps = layer.PrivateNumberOfChannels(1);
            pa = layer.PrivateNumberOfChannels(2);
            %
            if isempty(mus)
                mus = ones(ps+pa,nBlocks);
            elseif isscalar(mus)
                mus = mus*ones(ps+pa,nBlocks);
            end
            %
            layer.PrivateMus = mus;
            %layer = layer.updateParameters();
            layer.isUpdateRequested = true;
        end
        
        function layer = updateParameters(layer)
            %import tansacnet.lsun.get_fcn_orthmtxgen
            ps = layer.PrivateNumberOfChannels(1);
            %{
            if layer.NoDcLeakage
                layer.PrivateMus(1,:) = ones(1,size(layer.PrivateMus,2));           
                layer.PrivateAngles(1:ps-1,:) = ...
                    zeros(ps-1,size(layer.PrivateAngles,2),'like',layer.PrivateAngles);
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
            fcn_orthmtxgen = tansacnet.lsun.get_fcn_orthmtxgen(angles);                                    
            layer.V0T = permute(fcn_orthmtxgen(angles,mus),[2 1 3]);
            layer.isUpdateRequested = false;
        end
        
    end
    
end

