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
            %nSamples = size(X,2);
            %nblks = size(X,3);
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
            %
            Yt_ = permute(Yt,[1 3 2]);
            Yb_ = permute(Yb,[1 3 2]);
            %
            if isgpuarray(X)
                Zct_ = pagefun(@times,C_,Yt_);
                Zst_ = pagefun(@times,S_,Yt_);
                Zcb_ = pagefun(@times,C_,Yb_);
                Zsb_ = pagefun(@times,S_,Yb_);
                Yt_ = pagefun(@minus,Zct_,Zsb_);
                Yb_ = pagefun(@plus,Zst_,Zcb_);
            else
                Zct_ = C_.*Yt_;
                Zst_ = S_.*Yt_;
                Zcb_ = C_.*Yb_;
                Zsb_ = S_.*Yb_;
                %
                Yt_ = Zct_-Zsb_;
                Yb_ = Zst_+Zcb_;
            end
            Yt = ipermute(Yt_,[1 3 2]);
            Yb = ipermute(Yb_,[1 3 2]);
            %
            if strcmp(layer.Mode,'Synthesis')
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
            %
            Z = cat(1,Yt,Yb);
        end

        function dLdX = atomextbp_(layer,dLdZ,shift)
            %nblks = size(X,3);
            nChsTotal = sum(layer.PrivateNumberOfChannels);
            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            target = layer.TargetChannels;
            %
            if layer.isUpdateRequested
                layer = layer.updateParameters();
            end
            
            % Block circular shift for Synthesis Mode (Reverse)
            Yt = dLdZ(1:pt,:,:);
            Yb = dLdZ(pt+1:pt+pb,:,:);
            if strcmp(layer.Mode,'Synthesis')
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
            
            % C-S differential
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
            %
            Yt_ = permute(Yt,[1 3 2]);
            Yb_ = permute(Yb,[1 3 2]);
            if isgpuarray(dLdZ)
                Zct_ = pagefun(@times,C_,Yt_);
                Zst_ = pagefun(@times,S_,Yt_);
                Zcb_ = pagefun(@times,C_,Yb_);
                Zsb_ = pagefun(@times,S_,Yb_);
                Zt = pagefun(@plus,Zct_,Zsb_);
                Zb = pagefun(@minus,Zcb_,Zst_);
            else
                Zt =  C_.*Yt_ + S_.*Yb_; % Transposed
                Zb = -S_.*Yt_ + C_.*Yb_; % Transposed
            end
            Yt = ipermute(Zt,[1 3 2]);
            Yb = ipermute(Zb,[1 3 2]);

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
            dLdX = cat(1,Yt,Yb);
        end

        function dLdW = atomextdiff_(layer,X,dLdZ,shift)
            nChsTotal = sum(layer.PrivateNumberOfChannels);
            %nSamples = size(dLdZ,2);
            nblks = size(dLdZ,3);

            pt = ceil(nChsTotal/2);
            pb = floor(nChsTotal/2);
            target = layer.TargetChannels;
            %
            if layer.isUpdateRequested
                layer = layer.updateParameters();
            end
            %
            angles = layer.PrivateAngles;
            nAngles = size(angles,1);
            dLdW = zeros(nAngles,nblks,'like',dLdZ);
            %
            c_top = X(1:pt,:,:);
            c_btm = X(pt+1:pt+pb,:,:);
            if strcmp(layer.Mode,'Analysis')
                % Block circular shift for Analysis mode
                if strcmp(target,'Bottom')
                    c_btm = circshift(c_btm,shift);
                elseif strcmp(target,'Top')
                    c_top = circshift(c_top,shift);
                else
                    throw(MException('LsunLayer:InvalidTargetChannels',...
                        '%s : TaregetChannels should be either of Top or Bottom',...
                        layer.TargetChannels))
                end
                % C-S differential
                c_top_ = permute(c_top,[1 3 2]);
                c_btm_ = permute(c_btm,[1 3 2]);
                for iAngle = uint32(1:nAngles)
                    dCk_ = zeros(pt,nblks);
                    dSk_ = zeros(pb,nblks);
                    dCk_(iAngle,:) = -sin(angles(iAngle,:));
                    dSk_(iAngle,:) =  cos(angles(iAngle,:));
                    %
                    if isgpuarray(X)
                        dc_top_ = pagefun(@times,dCk_,c_top_);
                        ds_top_ = pagefun(@times,dSk_,c_top_);
                        dc_btm_ = pagefun(@times,dCk_,c_btm_);
                        ds_btm_ = pagefun(@times,dSk_,c_btm_);
                        d_top_ = pagefun(@minus,dc_top_,ds_btm_);
                        d_btm_ = pagefun(@plus,ds_top_,dc_btm_);
                    else
                        d_top_ = dCk_.*c_top_ - dSk_.*c_btm_;
                        d_btm_ = dSk_.*c_top_ + dCk_.*c_btm_;
                    end
                    d_top = ipermute(d_top_,[1 3 2]);
                    d_btm = ipermute(d_btm_,[1 3 2]);
                    d = cat(1,d_top,d_btm);
                    dLdW(iAngle,:) = squeeze(sum(sum(bsxfun(@times,dLdZ,d),1),2));
                end
            elseif strcmp(layer.Mode,'Synthesis')
                % Block circular shift for Synthesis mode
                dldz_top = dLdZ(1:pt,:,:);
                dldz_btm = dLdZ(pt+1:pt+pb,:,:);
                if strcmp(target,'Bottom')
                    dldz_btm = circshift(dldz_btm,-shift);
                elseif strcmp(target,'Top')
                    dldz_top = circshift(dldz_top,-shift);
                else
                    throw(MException('LsunLayer:InvalidTargetChannels',...
                        '%s : TaregetChannels should be either of Top or Bottom',...
                        layer.TargetChannels))
                end
                dldz_ = cat(1,dldz_top,dldz_btm);
                % C-S differential
                c_top_ = permute(c_top,[1 3 2]);
                c_btm_ = permute(c_btm,[1 3 2]);
                for iAngle = uint32(1:nAngles)
                    dCk_ = zeros(pt,nblks);
                    dSk_ = zeros(pb,nblks);
                    dCk_(iAngle,:) = -sin(angles(iAngle,:));
                    dSk_(iAngle,:) =  cos(angles(iAngle,:));
                    if isgpuarray(X)
                        dc_top_ = pagefun(@times,dCk_,c_top_);
                        ds_top_ = pagefun(@times,dSk_,c_top_);
                        dc_btm_ = pagefun(@times,dCk_,c_btm_);
                        ds_btm_ = pagefun(@times,dSk_,c_btm_);
                        d_top_ = pagefun(@plus,dc_top_,ds_btm_);
                        d_btm_ = pagefun(@minus,dc_btm_,ds_top_);
                    else
                        d_top_ =  dCk_.*c_top_ + dSk_.*c_btm_;
                        d_btm_ = -dSk_.*c_top_ + dCk_.*c_btm_;
                    end
                    d_top = ipermute(d_top_,[1 3 2]);
                    d_btm = ipermute(d_btm_,[1 3 2]);
                    d = cat(1,d_top,d_btm);
                    dLdW(iAngle,:) = squeeze(sum(sum(bsxfun(@times,dldz_,d),1),2));
                end
            else
                throw(MException('LsunLayer:InvalidMode',...
                    '%s : Mode should be either of Synthesis or Analysis',...
                    layer.Mode))
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

