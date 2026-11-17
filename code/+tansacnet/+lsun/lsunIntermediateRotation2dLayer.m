classdef lsunIntermediateRotation2dLayer < nnet.layer.Layer %#codegen
    %LSUNINTERMEDIATEROTATION2DLAYER
    %
    %
    % Requirements: MATLAB R2022a
    %
    % Copyright (c) 2022, Shogo MURAMATSU
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
        BoundaryOperation % 'Circular' or 'Termination'
        Direction         % 'Horizontal' or 'Vertical' (boundary for 'Termination')
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
        function layer = lsunIntermediateRotation2dLayer(varargin)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            p = inputParser;
            addParameter(p,'Stride',[])
            addParameter(p,'Angles',[])
            addParameter(p,'Mus',[])
            addParameter(p,'Mode','Synthesis')
            addParameter(p,'Name','')
            addParameter(p,'NumberOfBlocks',[1 1])
            addParameter(p,'DType','double')
            addParameter(p,'Device','cuda')
            addParameter(p,'BoundaryOperation','Circular')
            addParameter(p,'Direction','')
            parse(p,varargin{:})
            
            % Boundary operation
            % 'Termination' fixes the rotation matrices of the boundary
            % blocks (first column for 'Horizontal', first row for
            % 'Vertical') to -I. See
            % S. Muramatsu et al., IEEE Trans. Image Process., vol. 21,
            % no. 4, pp. 2314-2318, 2012.
            % Each value is validated by its set method.
            layer.BoundaryOperation = p.Results.BoundaryOperation;
            layer.Direction = p.Results.Direction;
            layer.validateDirection_();

            % Layer constructor function goes here.
            layer.Stride = p.Results.Stride;
            layer.NumberOfBlocks = p.Results.NumberOfBlocks;
            layer.PrivateNumberOfChannels = [ceil(prod(layer.Stride)/2) floor(prod(layer.Stride)/2)];
            layer.Name = p.Results.Name;
            layer.Mode = p.Results.Mode;
            layer.Angles = p.Results.Angles;
            layer.Mus = p.Results.Mus;
            layer.Description = layer.Mode ...
                + " LSUN intermediate rotation " ...
                + "(ps,pa) = (" ...
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
            
            nrows = size(X,2);
            ncols = size(X,3);            
            nSamples = size(X,4);            
            ps = layer.PrivateNumberOfChannels(1);
            pa = layer.PrivateNumberOfChannels(2);
            if layer.isUpdateRequested
                layer = layer.updateParameters();
            end
            %
            Un_ = layer.Un;
            Y = X; %permute(X,[3 1 2 4]);
            Ya = reshape(Y(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            if strcmp(layer.Mode,'Analysis')
                A_ = Un_;
            elseif strcmp(layer.Mode,'Synthesis')
                A_ = permute(Un_,[2 1 3]);
            else
                throw(MException('LsunLayer:InvalidMode',...
                    '%s : Mode should be either of Synthesis or Analysis',...
                    layer.Mode))
            end

            Za = zeros(pa,nrows*ncols,nSamples,'like',Y);
            for iSample = 1:nSamples
                if isgpuarray(X)
                    Ya_iSample = permute(Ya(:,:,iSample),[1 4 2 3]);
                    Za_iSample = pagefun(@mtimes,A_,Ya_iSample);
                    Za(:,:,iSample) = ipermute(Za_iSample,[1 4 2 3]);
                else
                    for iblk = 1:(nrows*ncols)
                        Za(:,iblk,iSample) = A_(:,:,iblk)*Ya(:,iblk,iSample);
                    end
                end
            end
            Y(ps+1:ps+pa,:,:,:) = reshape(Za,pa,nrows,ncols,nSamples);
            Z = Y; %ipermute(Y,[3 1 2 4]);
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
            
            nrows = size(dLdZ,2);
            ncols = size(dLdZ,3);
            nSamples = size(dLdZ,4);            
            ps = layer.PrivateNumberOfChannels(1);
            pa = layer.PrivateNumberOfChannels(2);            
            %
            if layer.isUpdateRequested
                layer = layer.updateParameters();
            end
            anglesU = layer.PrivateAngles;
            musU = cast(layer.PrivateMus,'like',anglesU);
            
            % dLdX = dZdX x dLdZ
            %Un = fcn_orthmtxgen(anglesU,musU,0);
            %[Un_,dUnPst,dUnPre] = fcn_orthmtxgen_diff(anglesU,musU,0,[],[]);
            Un_ = layer.Un;
            %dUnPst = zeros(size(Un_),'like',Un_);
            dUnPst = bsxfun(@times,permute(musU,[1 3 2]),Un_);
            %for iblk = 1:(nrows*ncols)
            %    dUnPst(:,:,iblk) = bsxfun(@times,musU(:,iblk),Un_(:,:,iblk));
            %end
            dUnPre = repmat(eye(pa,'like',Un_),[1 1 (nrows*ncols)]);
            
            %
            dLdX = reshape(dLdZ,ps+pa,nrows,ncols,nSamples); 
            %cdLd_low = reshape(dLdZ(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            if strcmp(layer.Mode,'Analysis')
                A_ = permute(Un_,[2 1 3]);
            else
                A_ = Un_;
            end
            cdLd_low = reshape(dLdX(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            for iSample = 1:nSamples
                if isgpuarray(X)
                    cdLd_low_iSample = permute(cdLd_low(:,:,iSample),[1 4 2 3]);
                    cdLd_low_iSample = pagefun(@mtimes,A_,cdLd_low_iSample);
                    cdLd_low(:,:,iSample) = ipermute(cdLd_low_iSample,[1 4 2 3]);                    
                else
                    for iblk = 1:(nrows*ncols)
                        cdLd_low(:,iblk,iSample) = A_(:,:,iblk)*cdLd_low(:,iblk,iSample);
                    end
                end
            end
            dLdX(ps+1:ps+pa,:,:,:) = reshape(cdLd_low,pa,nrows,ncols,nSamples);
            %dLdX = dLdX; %ipermute(adLd_,[3 1 2 4]);

            % dLdWi = <dLdZ,(dVdWi)X>
            fcn_orthmtxgen_diff = tansacnet.lsun.get_fcn_orthmtxgen_diff(anglesU);
            nAngles = size(anglesU,1);
            dLdW = zeros(nAngles,nrows*ncols,'like',dLdZ);
            dldz_low = reshape(dLdZ(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);                        
            c_low = reshape(X(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);  
            for iAngle = uint32(1:nAngles)
                [dUn,dUnPst,dUnPre] = fcn_orthmtxgen_diff(anglesU,musU,iAngle,dUnPst,dUnPre);
                if strcmp(layer.Mode,'Analysis')
                    dA_ = dUn;
                else
                    dA_ = permute(dUn,[2 1 3]);
                end
                if isgpuarray(X)
                    c_low_ext = permute(c_low,[1 4 2 3]); % idx 1 iblk iSample
                    d_low_ext = pagefun(@mtimes,dA_,c_low_ext); % idx 1 iblk iSample
                    d_low = ipermute(d_low_ext,[1 4 2 3]);
                    dLdW(iAngle,:) = sum(bsxfun(@times,dldz_low,d_low),[1 3]);
                else
                    for iblk = 1:(nrows*ncols)
                        dA_iblk = dA_(:,:,iblk);
                        dldz_low_iblk = squeeze(dldz_low(:,iblk,:));
                        c_low_iblk = squeeze(c_low(:,iblk,:));
                        d_low_iblk = zeros(size(c_low_iblk),'like',c_low_iblk);
                        for iSample = 1:nSamples
                            d_low_iblk(:,iSample) = dA_iblk*c_low_iblk(:,iSample);
                        end
                        dLdW(iAngle,iblk) = sum(bsxfun(@times,dldz_low_iblk,d_low_iblk),'all');
                    end
                end
            end
            % The boundary blocks fixed to -I have no angle gradients
            if strcmp(layer.BoundaryOperation,'Termination')
                dLdW(:,layer.boundaryBlocks_()) = 0;
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
            %layer = layer.updateParameters();
            layer.isUpdateRequested = true;
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
            %layer = layer.updateParameters();
            layer.isUpdateRequested = true;
        end
        
        function layer = updateParameters(layer)
            %import tansacnet.lsun.get_fcn_orthmtxgen
            anglesU = layer.PrivateAngles;
            musU = cast(layer.PrivateMus,'like',anglesU);
            if isrow(musU)
                musU = musU.';
            end
            fcn_orthmtxgen = tansacnet.lsun.get_fcn_orthmtxgen(anglesU);
            Un_ = fcn_orthmtxgen(anglesU,musU);
            % Fix the boundary blocks to -I for 'Termination'
            if strcmp(layer.BoundaryOperation,'Termination')
                bblks = layer.boundaryBlocks_();
                pa = layer.PrivateNumberOfChannels(2);
                Un_(:,:,bblks) = repmat(-eye(pa,'like',Un_),[1 1 numel(bblks)]);
            end
            layer.Un = Un_;
            layer.isUpdateRequested = false;
        end
        
    end

    methods

        function layer = set.BoundaryOperation(layer,boundaryOperation)
            if ~any(strcmp(boundaryOperation,{'Circular','Termination'}))
                throw(MException('LsunLayer:InvalidBoundaryOperation',...
                    '%s : BoundaryOperation should be either of Circular or Termination',...
                    boundaryOperation))
            end
            layer.BoundaryOperation = boundaryOperation;
            layer.isUpdateRequested = true; %#ok<MCSUP>
        end

        function layer = set.Direction(layer,direction)
            % Empty Direction is allowed only for 'Circular'
            if ~isempty(direction) ...
                    && ~any(strcmp(direction,{'Horizontal','Vertical'}))
                throw(MException('LsunLayer:InvalidDirection',...
                    '%s : Direction should be either of Horizontal or Vertical',...
                    direction))
            end
            layer.Direction = direction;
            layer.isUpdateRequested = true; %#ok<MCSUP>
        end

    end

    methods (Access = private)

        function validateDirection_(layer)
            % 'Termination' requires Direction
            if strcmp(layer.BoundaryOperation,'Termination') ...
                    && ~any(strcmp(layer.Direction,{'Horizontal','Vertical'}))
                throw(MException('LsunLayer:InvalidDirection',...
                    'Direction should be either of Horizontal or Vertical for Termination'))
            end
        end

        function bblks = boundaryBlocks_(layer)
            % Indices of the boundary blocks in column-major block order
            layer.validateDirection_();
            nrows = layer.NumberOfBlocks(1);
            ncols = layer.NumberOfBlocks(2);
            if strcmp(layer.Direction,'Horizontal')
                bblks = 1:nrows;             % first column
            else % 'Vertical'
                bblks = 1:nrows:nrows*ncols; % first row
            end
        end

    end

end

