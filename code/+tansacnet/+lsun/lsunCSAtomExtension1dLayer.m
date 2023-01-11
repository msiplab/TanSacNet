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
            layer.Direction = p.Results.Direction;
            layer.TargetChannels = p.Results.TargetChannels;
            layer.Description =  layer.Direction ...
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
        
        function dLdX = backward(layer, ~, ~, dLdZ, ~)
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
                shift = [ 0 0 -1 ]; % Reverse
            elseif strcmp(dir,'Left')
                shift = [ 0 0 1 ];  % Reverse
            else
                throw(MException('NsoltLayer:InvalidDirection',...
                    '%s : Direction should be either of Right, Left, Down or Up',...
                    layer.Direction))
            end
            %
            dLdX = layer.atomext_(dLdZ,shift);
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
            % Block circular shift
            Yt = X(1:pt,:,:);
            Yb = X(pt+1:pt+pb,:,:);
            if strcmp(target,'Bottom')
                Yb = circshift(Yb,shift);
            elseif strcmp(target,'Top')
                Yt = circshift(Yt,shift);
            else
                throw(MException('NsoltLayer:InvalidTargetChannels',...
                    '%s : TaregetChannels should be either of Top or Bottom',...
                    layer.TargetChannels))
            end
            % C-S block butterfly            
            Ck_ = layer.Ck;
            Sk_ = layer.Sk;
            Zct = zeros(size(Yt),'like',Yt);
            Zst = zeros(size(Yt),'like',Yt);            
            Zcb = zeros(size(Yb),'like',Yb);
            Zsb = zeros(size(Yb),'like',Yb);
            %
            if isgpuarray(X)
                Yt_ = permute(Yt,[1 3 2]);
                Yb_ = permute(Yb,[1 3 2]);
                Zct_ = pagefun(@times,Ck_,Yt_);
                Zst_ = pagefun(@times,Sk_,Yt_);
                Zcb_ = pagefun(@times,Ck_,Yb_);
                Zsb_ = pagefun(@times,Sk_,Yb_);
                Yt = ipermute(pagefun(@minus,Zct_,Zsb_),[1 3 2]);
                Yb = ipermute(pagefun(@plus,Zst_,Zcb_),[1 3 2]);                
            else
                for iSample = 1:nSamples
                    for iblk = 1:nblks
                        Zct(:,iSample,iblk) = Ck_(:,iblk).*Yt(:,iSample,iblk);
                        Zst(:,iSample,iblk) = Sk_(:,iblk).*Yt(:,iSample,iblk);
                        Zcb(:,iSample,iblk) = Ck_(:,iblk).*Yb(:,iSample,iblk);
                        Zsb(:,iSample,iblk) = Sk_(:,iblk).*Yb(:,iSample,iblk);
                    end
                end
                Yt = Zct-Zsb;
                Yb = Zst+Zcb;
            end
            %
            Z = cat(1,Yt,Yb);
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

