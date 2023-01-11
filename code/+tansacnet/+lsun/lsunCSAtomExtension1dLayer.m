classdef lsunCSAtomExtension1dLayer < nnet.layer.Layer %#codegen
    %LSUNCSATOMEXTENSION1DLAYER
    %
    %   コンポーネント別に入力(nComponents=1のみサポート):
    %      nChsTotal x nSamples x nBlks
    %
    %   コンポーネント別に出力(nComponents=1のみサポート):
    %      nChsTotal x nSamples x nBlks
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
        Direction
        TargetChannels
        Mode
        NumberOfBlocks
        % Layer properties go here.
    end

    properties (Learnable, Dependent)
        Angles
    end

    properties (Access = private)
        PrivateNumberOfChannels
        PrivateAngles
        isUpdateRequested
    end

    properties (Hidden)
        Ck
        Sk
    end
    
    methods
        function layer = lsunCSAtomExtension1dLayer(varargin)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            p = inputParser;
            addParameter(p,'Name','')
            addParameter(p,'Stride',[])
            addParameter(p,'Angles',[])
            addParameter(p,'Mode','Synthesis')
            addParameter(p,'Direction','')
            addParameter(p,'TargetChannels','')
            addParameter(p,'NumberOfBlocks',1)
            parse(p,varargin{:})
            
            % Layer constructor function goes here.
            layer.Stride = p.Results.Stride;
            layer.NumberOfBlocks = p.Results.NumberOfBlocks;
            layer.PrivateNumberOfChannels = [ceil(layer.Stride/2) floor(layer.Stride/2)];
            layer.Name = p.Results.Name;
            layer.Angles = p.Results.Angles;
            layer.Mode = p.Results.Mode;
            layer.Direction = p.Results.Direction;
            layer.TargetChannels = p.Results.TargetChannels;
            layer.Description = layer.Mode ...
                + " LSUN C-S transform w/ " ...
                + layer.Direction ...
                + " shift the " ...
                + lower(layer.TargetChannels) ...
                + "-channel Coefs. " ...
                + "(pt,pb) = (" ...
                + layer.PrivateNumberOfChannels(1) + "," ...
                + layer.PrivateNumberOfChannels(2) + ")";
            layer.Type = '';            

            nChsTotal = sum(layer.PrivateNumberOfChannels);
            nAngles = nChsTotal/2;
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
            dir = layer.Direction;
            %
            if strcmp(dir,'Right')
                shift = [ 0 0 1 ];
            elseif strcmp(dir,'Left')
                shift = [ 0 0 -1 ];
            else
                throw(MException('NsoltLayer:InvalidDirection',...
                    '%s : Direction should be either of Right, Left, Down or Up',...
                    layer.Direction))
            end
            %
            Z = layer.atomext_(X,shift);
        end
        
        function [dLdX,dLdW] = backward(layer, X, ~, dLdZ, ~)
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
            
            % Layer forward function for prediction goes here.
            dir = layer.Direction;

            %
            if strcmp(dir,'Right')
                shift = [ 0 0 1 ]; 
            elseif strcmp(dir,'Left')
                shift = [ 0 0 -1 ];  
            else
                throw(MException('NsoltLayer:InvalidDirection',...
                    '%s : Direction should be either of Right, Left, Down or Up',...
                    layer.Direction))
            end
            
            % dLdX = dZdX x dLdZ
            dLdX = layer.atomextbp_(dLdZ,shift);

            % dLdWi = <dLdZ,(dVdWi)X>       
            dLdW = layer.atomextdiff_(X,dLdZ,shift);

        end
        
        function Z = atomext_(layer,X,shift)
            nSamples = size(X,2);
            nblks = size(X,3);
            nChsTotal = sum(layer.PrivateNumberOfChannels);
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            target = layer.TargetChannels;            
            %
            if layer.isUpdateRequested
                layer = layer.updateParameters();
            end
            % Block circular shift for Analysis Mode
            Yt = X(1:pt,:,:);
            Yb = X(pt+1:pt+pb,:,:);
            if strcmp(layer.Mode,'Analysis')
                if strcmp(target,'Bottom')
                    Yb = circshift(Yb,shift);
                elseif strcmp(target,'Top')
                    Yt = circshift(Yt,shift);
                else
                    throw(MException('LsunLayer:InvalidTargetChannels',...
                        '%s : TaregetChannels should be either of Top or Bottom',...
                        layer.TargetChannels))
                end
            end
            % C-S block butterfly
            C_ = layer.Ck;
            if strcmp(layer.Mode,'Analysis')
                S_ = layer.Sk;
            elseif strcmp(layer.Mode,'Synthesis')
                S_ = -layer.Sk;
            else
                throw(MException('LsunLayer:InvalidMode',...
                    '%s : Mode should be either of Synthesis or Analysis',...
                    layer.Mode))
            end
            Zct = zeros(size(Yt),'like',Yt);
            Zst = zeros(size(Yt),'like',Yt);            
            Zcb = zeros(size(Yb),'like',Yb);
            Zsb = zeros(size(Yb),'like',Yb);
            %
            if isgpuarray(X)
                Yt_ = permute(Yt,[1 3 2]);
                Yb_ = permute(Yb,[1 3 2]);
                Zct_ = pagefun(@times,C_,Yt_);
                Zst_ = pagefun(@times,S_,Yt_);
                Zcb_ = pagefun(@times,C_,Yb_);
                Zsb_ = pagefun(@times,S_,Yb_);
                Yt = ipermute(pagefun(@minus,Zct_,Zsb_),[1 3 2]);
                Yb = ipermute(pagefun(@plus,Zst_,Zcb_),[1 3 2]);                
            else
                for iSample = 1:nSamples
                    for iblk = 1:nblks
                        Zct(:,iSample,iblk) = C_(:,iblk).*Yt(:,iSample,iblk);
                        Zst(:,iSample,iblk) = S_(:,iblk).*Yt(:,iSample,iblk);
                        Zcb(:,iSample,iblk) = C_(:,iblk).*Yb(:,iSample,iblk);
                        Zsb(:,iSample,iblk) = S_(:,iblk).*Yb(:,iSample,iblk);
                    end
                end
                Yt = Zct-Zsb;
                Yb = Zst+Zcb;
            end
            %
            Z = cat(1,Yt,Yb);
        end

        function Z = atomextbp_(layer,X,shift)
            nSamples = size(X,2);
            nblks = size(X,3);
            nChsTotal = sum(layer.PrivateNumberOfChannels);
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            target = layer.TargetChannels;
            %
            if layer.isUpdateRequested
                layer = layer.updateParameters();
            end
            
            % C-S differential
            Yt = X(1:pt,:,:);
            Yb = X(pt+1:pt+pb,:,:);
            C_ = layer.Ck;
            if strcmp(layer.Mode,'Analysis')
                S_ = layer.Sk;
            elseif strcmp(layer.Mode,'Synthesis')
                S_ = -layer.Sk;
            else
                throw(MException('LsunLayer:InvalidMode',...
                    '%s : Mode should be either of Synthesis or Analysis',...
                    layer.Mode))
            end
            Zct = zeros(size(Yt),'like',Yt);
            Zst = zeros(size(Yt),'like',Yt);
            Zcb = zeros(size(Yb),'like',Yb);
            Zsb = zeros(size(Yb),'like',Yb);
            if isgpuarray(X)
                Yt_ = permute(Yt,[1 3 2]);
                Yb_ = permute(Yb,[1 3 2]);
                Zct_ = pagefun(@times,C_,Yt_);
                Zst_ = pagefun(@times,S_,Yt_);
                Zcb_ = pagefun(@times,C_,Yb_);
                Zsb_ = pagefun(@times,S_,Yb_);
                Yt = ipermute(pagefun(@minus,Zct_,Zsb_),[1 3 2]);
                Yb = ipermute(pagefun(@plus,Zst_,Zcb_),[1 3 2]);
            else
                for iSample = 1:nSamples
                    for iblk = 1:nblks
                        Zct(:,iSample,iblk) = C_(:,iblk).*Yt(:,iSample,iblk);
                        Zst(:,iSample,iblk) = S_(:,iblk).*Yt(:,iSample,iblk);
                        Zcb(:,iSample,iblk) = C_(:,iblk).*Yb(:,iSample,iblk);
                        Zsb(:,iSample,iblk) = S_(:,iblk).*Yb(:,iSample,iblk);
                    end
                end
                Yt =  Zct+Zsb; % Transposed
                Yb = -Zst+Zcb; % Transposed
            end
            %
            % Block circular shift for Analysis Mode (Reverse)
            if strcmp(layer.Mode,'Analysis')
                if strcmp(target,'Bottom')
                    Yb = circshift(Yb,-shift); % Reverse
                elseif strcmp(target,'Top')
                    Yt = circshift(Yt,-shift); % Reverse
                else
                    throw(MException('LsunLayer:InvalidTargetChannels',...
                        '%s : TaregetChannels should be either of Top or Bottom',...
                        layer.TargetChannels))
                end
            end
            %
            Z = cat(1,Yt,Yb);
        end

        function dLdW = atomextdiff_(layer,X,dLdZ,shift)
            nChsTotal = sum(layer.PrivateNumberOfChannels);
            %nSamples = size(dLdZ,2);            
            nblks = size(dLdZ,3);            

            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            %target = layer.TargetChannels;                        
            %
            angles = layer.PrivateAngles;
            nAngles = size(angles,1);
            dLdW = zeros(nAngles,nblks,'like',dLdZ);
            %
            c_top = X(1:pt,:,:);
            c_btm = X(pt+1:pt+pb,:,:);
            % Block circular shift
            if strcmp(layer.Mode,'Analysis')
                %if strcmp(target,'Bottom')
                    c_btm = circshift(c_btm,shift);
                %elseif strcmp(target,'Top')
                %    c_top = circshift(c_top,shift);
                %else
                %    throw(MException('LsunLayer:InvalidTargetChannels',...
                %        '%s : TaregetChannels should be either of Top or Bottom',...
                %        layer.TargetChannels))
                %end
            end
            % C-S diferential
            for iAngle = uint32(1:nAngles)
                dCk_ = zeros(pt,nblks);
                dSk_ = zeros(pb,nblks);
                %if strcmp(layer.Mode,'Analysis')
                    dCk_(iAngle,:) = zeros(1,nblks); %-sin(angles(iAngle,:));
                    dSk_(iAngle,:) = ones(1,nblks); %cos(angles(iAngle,:));
                %else
                %end
                %{
                if isgpuarray(X)
                else
                %}
                    for iblk = 1:nblks
                        dCk_iblk = dCk_(:,iblk);
                        dSk_iblk = dSk_(:,iblk);
                        c_top_iblk = c_top(:,:,iblk);
                        c_btm_iblk = c_btm(:,:,iblk);
                        d_top_iblk = dCk_iblk.*c_top_iblk - dSk_iblk.*c_btm_iblk;
                        d_btm_iblk = dSk_iblk.*c_top_iblk + dCk_iblk.*c_btm_iblk;
                        d_iblk = cat(1,d_top_iblk,d_btm_iblk);
                        dldz_iblk = dLdZ(:,:,iblk);
                        dLdW(iAngle,iblk) = sum(bsxfun(@times,dldz_iblk,d_iblk),'all');
                    end
                %end
            end
        end

        function angles = get.Angles(layer)
            angles = layer.PrivateAngles;
        end

        function layer = set.Angles(layer,angles)
            nBlocks = layer.NumberOfBlocks;
            nChsTotal = sum(layer.PrivateNumberOfChannels);
            nAngles = nChsTotal/2;
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
        

        function layer = updateParameters(layer)
            angles = layer.PrivateAngles;
            nBlocks = layer.NumberOfBlocks;
            if isvector(angles)
                nAngles = length(angles);
            else
                nAngles = size(angles,1);
            end
            if nAngles > 0
                layer.Ck = cos(angles);
                layer.Sk = sin(angles);
            else
                layer.Ck = ones(1,nBlocks);
                layer.Sk = zeros(1,nBlocks);
            end
            layer.isUpdateRequested = false;
        end

    end

end

