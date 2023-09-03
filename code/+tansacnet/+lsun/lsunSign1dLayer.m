classdef lsunSign1dLayer < nnet.layer.Layer %#codegen
    %LSUNINTERMEDIATEDUALROTATION1DLAYER
    %   
    %   コンポーネント別に入力(nComponents)
    %      nChs x 1 x nBlks x nSamples
    %
    %   コンポーネント別に出力(nComponents):
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
        Mode
        NumberOfBlocks
    end
    
    properties (Dependent)
        Mus
    end
    
    properties (Access = private)
        PrivateNumberOfChannels
        PrivateAngles
        PrivateMus
        isUpdateRequested
    end
    
    properties (Hidden)
        Wn
        Un
    end
    
    methods
        function layer = lsunSign1dLayer(varargin)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            p = inputParser;
            addParameter(p,'Stride',[])
            %addParameter(p,'Angles',[])
            addParameter(p,'Mus',[])
            addParameter(p,'Mode','Synthesis')
            addParameter(p,'Name','')
            addParameter(p,'NumberOfBlocks',1)
            parse(p,varargin{:})
            
            % Layer constructor function goes here.
            layer.Stride = p.Results.Stride;
            layer.NumberOfBlocks = p.Results.NumberOfBlocks;
            layer.PrivateNumberOfChannels = [ceil(layer.Stride/2) floor(layer.Stride/2)];
            layer.Name = p.Results.Name;
            layer.Mode = p.Results.Mode;
            %layer.Angles = p.Results.Angles;
            layer.Mus = p.Results.Mus;
            layer.Description = layer.Mode ...
                + " LSUN intermediate full rotation " ...
                + "(pt,pb) = (" ...
                + layer.PrivateNumberOfChannels(1) + "," ...
                + layer.PrivateNumberOfChannels(2) + ")";
            layer.Type = '';
            
            nChsTotal = sum(layer.PrivateNumberOfChannels);
            nAngles = (nChsTotal-2)*nChsTotal/4;
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

            nSamples = size(X,4);                        
            nblks = size(X,3);
            pt = layer.PrivateNumberOfChannels(1);
            pb = layer.PrivateNumberOfChannels(2);
            if layer.isUpdateRequested
                layer = layer.updateParameters();
            end
            %
            Wn_ = layer.Wn;            
            Un_ = layer.Un;
            if strcmp(layer.Mode,'Analysis')
                W_ = Wn_;
                U_ = Un_;                
            elseif strcmp(layer.Mode,'Synthesis')
                W_ = permute(Wn_,[2 1 3]);
                U_ = permute(Un_,[2 1 3]);
            else
                throw(MException('LsunLayer:InvalidMode',...
                    '%s : Mode should be either of Synthesis or Analysis',...
                    layer.Mode))
            end

            Y = X;
            Zt = zeros(pt,1,nblks,nSamples,'like',Y);
            Zb = zeros(pb,1,nblks,nSamples,'like',Y);
            for iSample = 1:nSamples
                Yt_iSample = Y(1:pt,:,:,iSample);
                Yb_iSample = Y(pt+1:pt+pb,:,:,iSample);
                if ~isdlarray(X) && isgpuarray(X)
                    Zt_iSample = pagefun(@mtimes,W_,Yt_iSample);
                    Zb_iSample = pagefun(@mtimes,U_,Yb_iSample);
                else
                    Zt_iSample = zeros(size(Yt_iSample),'like',Yt_iSample);
                    Zb_iSample = zeros(size(Yb_iSample),'like',Yb_iSample);
                    for iblk = 1:nblks
                        Zt_iSample(:,:,iblk) = W_(:,:,iblk)*Yt_iSample(:,:,iblk);
                        Zb_iSample(:,:,iblk) = U_(:,:,iblk)*Yb_iSample(:,:,iblk);
                    end
                end
                Zt(:,:,:,iSample) = Zt_iSample;
                Zb(:,:,:,iSample) = Zb_iSample;
            end
            Z = cat(1,Zt,Zb);
        end

        function mus = get.Mus(layer)
            mus = layer.PrivateMus;
        end

        function layer = set.Mus(layer,mus)
            nBlocks = prod(layer.NumberOfBlocks);
            pt = layer.PrivateNumberOfChannels(1);
            pb = layer.PrivateNumberOfChannels(2);
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
            pt = layer.PrivateNumberOfChannels(1);
            pb = layer.PrivateNumberOfChannels(2);
            %
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
            muW = mus(1:pt,:);
            muU = mus(pt+1:pt+pb,:);
            if nAngles > 0
                anglesW = angles(1:nAngles/2,:);
                anglesU = angles(nAngles/2+1:nAngles,:);
                fcn_orthmtxgen = tansacnet.lsun.get_fcn_orthmtxgen(angles);
                layer.Wn = fcn_orthmtxgen(anglesW,muW);
                layer.Un = fcn_orthmtxgen(anglesU,muU);
            else
                layer.Wn = reshape(muW,1,1,[]);
                layer.Un = reshape(muU,1,1,[]);
            end
            layer.isUpdateRequested = false;
        end
        
    end

end

