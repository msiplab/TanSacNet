classdef lsunFinalFullRotation1dLayer < nnet.layer.Layer %#codegen
    %LSUNFINALFULLROTATION1DLAYER
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
        Device
        DType
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
            addParameter(p,'DType','double')
            addParameter(p,'Device','cuda')
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
                + "(pt,pb) = (" ...
                + layer.PrivateNumberOfChannels(1) + "," ...
                + layer.PrivateNumberOfChannels(2) + "), "  ...
                + "m = " + layer.Stride;
            layer.Type = '';
            layer.Device = p.Results.Device;
            layer.DType = p.Results.DType;
            
            nChsTotal = sum(layer.PrivateNumberOfChannels);            
            nAngles = (nChsTotal-1)*nChsTotal/2;
            if size(layer.PrivateAngles,1)~=nAngles
                error('Invalid # of angles')
            end
            
            layer = layer.updateParameters();
        end

        function layer = initialize(layer,layout)
            % (Optional) Initialize layer learnable and state parameters.
            %
            % Inputs:
            %         layer  - Layer to initialize
            %         layout - Data layout, specified as a networkDataLayout
            %                  object
            %
            % Outputs:
            %         layer - Initialized layer
            %
            %  - For layers with multiple inputs, replace layout with
            %    layout1,...,layoutN, where N is the number of inputs.

            % Define layer initialization function here.
            %fprintf('Layout size: [%s]\n', sprintf('%d ', layout.Size));

            % LAYOUT
            nCols = layout.Size(3);
            %disp(size(layer.PrivateAngles,1))
            % layout - [prod(Stride) nRows/Stride(Dir.VERTICLE) nCols/Stride(Dir.HORIZONTAL)]
            inputSize =  nCols*layer.Stride;
            layer.NumberOfBlocks = inputSize./layer.Stride;
            layoutsize = [size(layer.PrivateAngles,1) prod(layer.NumberOfBlocks)];
            %layout = networkDataLayout(layoutsize,'SS');
            if isempty(layer.Angles)
                angles = zeros(layoutsize,layer.DType);
                layer.Angles = angles;
            end
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
            
            nSamples = size(X,4);            
            nblks = size(X,3);    
            %nChsTotal = sum(layer.PrivateNumberOfChannels);
            if layer.isUpdateRequested
                layer = layer.updateParameters();
            end
            %
            V0T_ = layer.V0T;
            Y = X;
            for iSample = 1:nSamples
                Y_iSample = Y(:,:,:,iSample);
                if isgpuarray(X)
                    Y_iSample = pagefun(@mtimes,V0T_,Y_iSample);
                else
                    for iblk = 1:nblks
                        Y_iSample(:,:,iblk) = V0T_(:,:,iblk)*Y_iSample(:,:,iblk);
                    end
                end
                Y(:,:,:,iSample) = Y_iSample;
            end
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
            %import tansacnet.lsun.get_fcn_orthmtxgen_diff
            
            nSamples = size(dLdZ,4);
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
            adldz_ = dLdZ;
            cdLd_ = adldz_;
            for iSample = 1:nSamples
                cdLd_iSample = cdLd_(:,:,:,iSample);                
                if isgpuarray(X)
                    cdLd_iSample = pagefun(@mtimes,V0,cdLd_iSample);
                else
                    for iblk = 1:nblks
                        cdLd_iSample(:,:,iblk) = V0(:,:,iblk)*cdLd_iSample(:,:,iblk);
                    end
                end
                cdLd_(:,:,:,iSample) = cdLd_iSample;
                
            end
            dLdX = cdLd_;
            
            % dLdWi = <dLdZ,(dVdWi)X>
            fcn_orthmtxgen_diff = tansacnet.lsun.get_fcn_orthmtxgen_diff(angles);            
            dLdW = zeros(nAngles,nblks,'like',dLdZ);
            dldz_ = dLdZ;
            c_ = X;
            for iAngle = uint32(1:nAngles)
                [dV0,dV0Pst,dV0Pre] = fcn_orthmtxgen_diff(angles,mus,iAngle,dV0Pst,dV0Pre);
                dV0_T = permute(dV0,[2 1 3]);
                if isgpuarray(X)
                    c_ext = c_; % idx 1 iblk iSample
                    d_ext = pagefun(@mtimes,dV0_T,c_ext); % idx 1 iblk iSample
                    d_ = d_ext;
                    dLdW(iAngle,:) = sum(bsxfun(@times,dldz_,d_),[1 4]);
                else
                    for iblk = 1:nblks
                        dldz_iblk = squeeze(dldz_(:,:,iblk,:));
                        c_iblk = squeeze(c_(:,:,iblk,:));
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
            %layer = layer.updateParameters();
            layer.isUpdateRequested = true;
        end
        
        function layer = updateParameters(layer)
            %import tansacnet.lsun.get_fcn_orthmtxgen
            %pt = layer.PrivateNumberOfChannels(1);
            %{
            if layer.NoDcLeakage
                layer.PrivateMus(1,:) = ones(1,size(layer.PrivateMus,2));           
                layer.PrivateAngles(1:pt-1,:) = ...
                    zeros(pt-1,size(layer.PrivateAngles,2),'like',layer.PrivateAngles);
            end      
            %}
            angles = layer.PrivateAngles;
            mus = cast(layer.PrivateMus,'like',angles);
            %if isvector(angles)
            %    nAngles = length(angles);
            %else
            %    nAngles = size(angles,1);
            %end
            if isrow(mus)
                mus = mus.';
            end
            fcn_orthmtxgen = tansacnet.lsun.get_fcn_orthmtxgen(angles);                                    
            layer.V0T = permute(fcn_orthmtxgen(angles,mus),[2 1 3]);
            layer.isUpdateRequested = false;
        end
        
    end
    
end

