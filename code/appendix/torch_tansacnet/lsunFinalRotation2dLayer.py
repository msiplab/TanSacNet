import torch
import torch.nn as nn
import math
from lsunUtility import Direction
from orthonormalTransform import SetOfOrthonormalTransforms

class LsunFinalRotation2dLayer(nn.Module):
    """
    LSUNFINALROTATION2DLAYER

    コンポーネント別に入力(nComponents):
        nSamples x nRows x nCols x nChs

    コンポーネント別に出力(nComponents):
        nSamples x nRows x nCols x nChs

    Requirements: Python 3.10/11.x, PyTorch 2.3.x

    Copyright (c) 2024, Shogo MURAMATSU

    All rights reserved.

    Contact address: Shogo MURAMATSU,
        Faculty of Engineering, Niigata University,
        8050 2-no-cho Ikarashi, Nishi-ku,
        Niigata, 950-2181, JAPAN

    https://www.eng.niigata-u.ac.jp/~msiplab/
    """

    def __init__(self, 
                 dtype=torch.get_default_dtype(),
                 device=torch.device('cpu'),
                 stride=None, 
                 number_of_blocks=[1,1], 
                 no_dc_leakage=False, 
                 name=''):
        super(LsunFinalRotation2dLayer, self).__init__()
        self.dtype = dtype
        self.device = device
        self.stride = stride
        self.number_of_blocks = number_of_blocks
        self.name = name        
        self.mus = None
        self.angles = None
        self.no_dc_leakage = no_dc_leakage
        ps = math.ceil(math.prod(self.stride)/2)
        pa = math.floor(math.prod(self.stride)/2)
        self.description = "LSUN final rotation " \
                           + "(ps,pa) = (" \
                           + str(ps) + "," \
                           + str(pa) + "), " \
                           + "(mv,mh) = (" \
                           + str(self.stride[Direction.VERTICAL]) + "," \
                           + str(self.stride[Direction.HORIZONTAL]) + ")"
        
        # Orthonormal matrix generation systems 
        nblks = self.number_of_blocks[Direction.VERTICAL]*self.number_of_blocks[Direction.HORIZONTAL]
        self.orthTransW0T = SetOfOrthonormalTransforms(name=self.name+"_W0T",nblks=nblks,n=ps,mode='Synthesis',device=self.device,dtype=self.dtype)
        self.orthTransU0T = SetOfOrthonormalTransforms(name=self.name+"_U0T",nblks=nblks,n=pa,mode='Synthesis',device=self.device,dtype=self.dtype) 
        self.orthTransW0T.angles = nn.init.zeros_(self.orthTransW0T.angles).to(self.device)
        self.orthTransU0T.angles = nn.init.zeros_(self.orthTransU0T.angles).to(self.device)

        # Update parameters
        self.update_parameters()

    def forward(self,X):
        nSamples = X.size(dim=0)
        nrows = X.size(dim=1)
        ncols = X.size(dim=2)
        nDecs = math.prod(self.stride)
        ps = math.ceil(nDecs/2.0)
        pa = math.floor(nDecs/2.0)

        # Update parameters
        if self.is_update_requested:
            self.number_of_blocks = [ nrows, ncols ]
            self.update_parameters()

        # Process
        # nSamples x nRows x nCols x nChs -> (nRows x nCols) x nChs x nSamples
        Y = X.permute(1,2,3,0).reshape(nrows*ncols,ps+pa,nSamples)
        Zs = self.orthTransW0T(Y[:,:ps,:])
        Za = self.orthTransU0T(Y[:,ps:,:])
        Z = torch.cat((Zs,Za),dim=1).reshape(nrows,ncols,ps+pa,nSamples).permute(3,0,1,2)

        return Z
    
    @property
    def angles(self):
        return torch.cat((self.orthTransW0T.angles,self.orthTransU0T.angles),dim=1)

    @angles.setter
    def angles(self, angles):
        self.__angles = angles
        self.is_update_requested = True

    @property
    def mus(self):
        return self.__mus
    
    @mus.setter
    def mus(self, mus):
        nBlocks = math.prod(self.number_of_blocks)
        nDecs = math.prod(self.stride)
        if mus is None:
            mus = torch.ones(nBlocks,nDecs,dtype=self.dtype)
        elif isinstance(mus, int) or isinstance(mus, float):
            mus = mus*torch.ones(nBlocks,nDecs,dtype=self.dtype)
        self.__mus = mus.to(self.device)
        self.is_update_requested = True

    def update_parameters(self):
        angles = self.__angles
        mus = self.__mus
        if angles is None:
            self.orthTransW0T.angles = nn.init.zeros_(self.orthTransW0T.angles).to(self.device)
            self.orthTransU0T.angles = nn.init.zeros_(self.orthTransU0T.angles).to(self.device)
        else:
            ps = int(math.ceil(math.prod(self.stride)/2.0))
            if self.no_dc_leakage:
                mus[:,0] = 1.0
                self.__mus = mus
                angles[:,:(ps-1)] = 0.0
                self.__angles = angles
            nAngles = angles.size(1)
            anglesW = angles[:,:nAngles//2]
            anglesU = angles[:,nAngles//2:]
            musW = mus[:,:ps]
            musU = mus[:,ps:]       
            #
            self.orthTransW0T.angles = anglesW
            self.orthTransW0T.mus = musW
            self.orthTransU0T.angles = anglesU
            self.orthTransU0T.mus = musU
        self.is_update_requested = False

"""
classdef lsunFinalRotation2dLayer < nnet.layer.Layer %#codegen
    %LSUNFINALROTATION2DLAYER
    %
    %   コンポーネント別に入力(nComponents):
    %      nChs x nRows x nCols x nSamples
    %
    %   コンポーネント別に出力(nComponents):
    %      nChs x nRows x nCols x nSamples
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
        NumberOfBlocks
    end
    
    properties (Dependent)
        NoDcLeakage
    end
    
    properties (Dependent)
        Mus
    end
    
    properties (Learnable,Dependent)
        Angles
    end
    
    properties (Access = private)
        PrivateNumberOfChannels
        PrivateNoDcLeakage
        PrivateAngles
        PrivateMus
        isUpdateRequested
    end
    
    properties (Hidden)
        W0T
        U0T
    end
    
    methods
        function layer = lsunFinalRotation2dLayer(varargin)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            p = inputParser;
            addParameter(p,'Stride',[])
            addParameter(p,'Mus',[])
            addParameter(p,'Angles',[])
            addParameter(p,'Name','')
            addParameter(p,'NoDcLeakage',false)
            addParameter(p,'NumberOfBlocks',[1 1])
            parse(p,varargin{:})

            % Layer constructor function goes here.
            layer.Stride = p.Results.Stride;
            layer.NumberOfBlocks = p.Results.NumberOfBlocks;
            layer.PrivateNumberOfChannels = [ceil(prod(layer.Stride)/2) floor(prod(layer.Stride)/2)];
            layer.Mus = p.Results.Mus;
            layer.Angles = p.Results.Angles;
            layer.NoDcLeakage = p.Results.NoDcLeakage;
            layer.Name = p.Results.Name;
            layer.Description = "LSUN final rotation " ...
                + "(ps,pa) = (" ...
                + layer.PrivateNumberOfChannels(1) + "," ...
                + layer.PrivateNumberOfChannels(2) + "), "  ...
                + "(mv,mh) = (" ...
                + layer.Stride(1) + "," ...
                + layer.Stride(2) + ")";
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
            
            nrows = size(X,2);
            ncols = size(X,3);    
            nSamples = size(X,4);
            ps = layer.PrivateNumberOfChannels(1);
            pa = layer.PrivateNumberOfChannels(2);
            if layer.isUpdateRequested
                layer = layer.updateParameters();
            end
            %
            W0T_ = layer.W0T;
            U0T_ = layer.U0T;
            %Y = X; %permute(X,[3 1 2 4]);
            Y = reshape(X,ps+pa,nrows*ncols,nSamples);
            %Ys = reshape(X(1:ps,:,:,:),ps,nrows*ncols,nSamples);
            %Ya = reshape(X(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            Zs = zeros(ps,nrows*ncols,nSamples,'like',Y);
            Za = zeros(pa,nrows*ncols,nSamples,'like',Y);
            for iSample = 1:nSamples
                if isgpuarray(X)
                    Ys_iSample = permute(Y(1:ps,:,iSample),[1 4 2 3]);
                    Ya_iSample = permute(Y(ps+1:end,:,iSample),[1 4 2 3]);
                    Zs_iSample = pagefun(@mtimes,W0T_,Ys_iSample);
                    Za_iSample = pagefun(@mtimes,U0T_,Ya_iSample);
                    Zs(:,:,iSample) = ipermute(Zs_iSample,[1 4 2 3]);
                    Za(:,:,iSample) = ipermute(Za_iSample,[1 4 2 3]);
                else
                    for iblk = 1:(nrows*ncols)
                        Zs(:,iblk,iSample) = W0T_(1:ps,:,iblk)*Y(1:ps,iblk,iSample);
                        Za(:,iblk,iSample) = U0T_(1:pa,:,iblk)*Y(ps+1:end,iblk,iSample);
                    end
                end
            end
            Zsa = cat(1,Zs,Za);
            %Z = ipermute(reshape(Zsa,nDecs,nrows,ncols,nSamples),...
            %    [3 1 2 4]);
            Z = reshape(Zsa,ps+pa,nrows,ncols,nSamples);

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
            muW = mus(1:ps,:);
            muU = mus(ps+1:end,:);
            anglesW = angles(1:nAngles/2,:);
            anglesU = angles(nAngles/2+1:end,:);
            %W0 = fcn_orthmtxgen(anglesW,muW,0);
            %U0 = fcn_orthmtxgen(anglesU,muU,0);
            %[W0,dW0Pst,dW0Pre] = fcn_orthmtxgen_diff(anglesW,muW,0,[],[]);            
            %[U0,dU0Pst,dU0Pre] = fcn_orthmtxgen_diff(anglesU,muU,0,[],[]);            
            W0_T = layer.W0T; %transpose(fcn_orthmtxgen(anglesW,muW,0));
            U0_T = layer.U0T; %transpose(fcn_orthmtxgen(anglesU,muU,0));
            W0 = permute(W0_T,[2 1 3]);
            U0 = permute(U0_T,[2 1 3]);
            dW0Pst = zeros(size(W0),'like',W0);
            dU0Pst = zeros(size(U0),'like',U0);
            for iblk = 1:(nrows*ncols)
                dW0Pst(:,:,iblk) = bsxfun(@times,muW(:,iblk),W0(:,:,iblk));
                dU0Pst(:,:,iblk) = bsxfun(@times,muU(:,iblk),U0(:,:,iblk));
            end
            dW0Pre = repmat(eye(ps,'like',W0),[1 1 (nrows*ncols)]);
            dU0Pre = repmat(eye(pa,'like',U0),[1 1 (nrows*ncols)]);
            
            % Layer backward function goes here.
            % dLdX = dZdX x dLdZ
            %adldz_ = dLdZ; %permute(dLdZ,[3 1 2 4]);
            %cdLd_ = reshape(adldz_,ps+pa,nrows*ncols,nSamples);
            cdLd_ = reshape(dLdZ,ps+pa,nrows*ncols,nSamples);
            cdLd_upp = zeros(ps,nrows*ncols,nSamples,'like',cdLd_);
            cdLd_low = zeros(pa,nrows*ncols,nSamples,'like',cdLd_);
            for iSample = 1:nSamples
                if isgpuarray(X)
                    cdLd_upp_iSample = permute(cdLd_(1:ps,:,iSample),[1 4 2 3]);
                    cdLd_low_iSample = permute(cdLd_(ps+1:end,:,iSample),[1 4 2 3]);
                    cdLd_upp_iSample = pagefun(@mtimes,W0(:,1:ps,:),cdLd_upp_iSample);
                    cdLd_low_iSample = pagefun(@mtimes,U0(:,1:pa,:),cdLd_low_iSample);
                    cdLd_upp(:,:,iSample) = ipermute(cdLd_upp_iSample,[1 4 2 3]);
                    cdLd_low(:,:,iSample) = ipermute(cdLd_low_iSample,[1 4 2 3]);
                else
                    for iblk = 1:(nrows*ncols)
                        cdLd_upp(:,iblk,iSample) = W0(:,1:ps,iblk)*cdLd_(1:ps,iblk,iSample);
                        cdLd_low(:,iblk,iSample) = U0(:,1:pa,iblk)*cdLd_(ps+1:ps+pa,iblk,iSample);
                    end
                end
            end
            %adLd_ = reshape(cat(1,cdLd_upp,cdLd_low),pa+ps,nrows,ncols,nSamples);
            %dLdX = adLd_; %ipermute(adLd_,[3 1 2 4]);
            dLdX = reshape(cat(1,cdLd_upp,cdLd_low),pa+ps,nrows,ncols,nSamples);
            
            % dLdWi = <dLdZ,(dVdWi)X>
            fcn_orthmtxgen_diff = tansacnet.lsun.get_fcn_orthmtxgen_diff(angles);            
            dLdW = zeros(nAngles,nrows*ncols,'like',dLdZ);
            %dldz_ = dLdZ; %permute(dLdZ,[3 1 2 4]);
            dldz_upp = reshape(dLdZ(1:ps,:,:,:),ps,nrows*ncols,nSamples);
            dldz_low = reshape(dLdZ(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            %a_ = X; %permute(X,[3 1 2 4]);
            c_upp = reshape(X(1:ps,:,:,:),ps,nrows*ncols,nSamples);
            c_low = reshape(X(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            for iAngle = uint32(1:nAngles/2)
                %dW0_T = transpose(fcn_orthmtxgen(anglesW,muW,iAngle));
                %dU0_T = transpose(fcn_orthmtxgen(anglesU,muU,iAngle));
                [dW0,dW0Pst,dW0Pre] = fcn_orthmtxgen_diff(anglesW,muW,iAngle,dW0Pst,dW0Pre);
                [dU0,dU0Pst,dU0Pre] = fcn_orthmtxgen_diff(anglesU,muU,iAngle,dU0Pst,dU0Pre);
                dW0_T = permute(dW0,[2 1 3]);
                dU0_T = permute(dU0,[2 1 3]);
                if isgpuarray(X)
                    c_upp_ext = permute(c_upp,[1 4 2 3]); % idx 1 iblk iSample
                    c_low_ext = permute(c_low,[1 4 2 3]); % idx 1 iblk iSample
                    d_upp_ext = pagefun(@mtimes,dW0_T(1:ps,:,:),c_upp_ext); % idx 1 iblk iSample
                    d_low_ext = pagefun(@mtimes,dU0_T(1:pa,:,:),c_low_ext); % idx 1 iblk iSample
                    d_upp = ipermute(d_upp_ext,[1 4 2 3]);
                    d_low = ipermute(d_low_ext,[1 4 2 3]);
                    dLdW(iAngle,:) = sum(bsxfun(@times,dldz_upp,d_upp),[1 3]);
                    dLdW(nAngles/2+iAngle,:) = sum(bsxfun(@times,dldz_low,d_low),[1 3]);
                else
                    for iblk = 1:(nrows*ncols)
                        dldz_upp_iblk = squeeze(dldz_upp(:,iblk,:));
                        dldz_low_iblk = squeeze(dldz_low(:,iblk,:));
                        c_upp_iblk = squeeze(c_upp(:,iblk,:));
                        c_low_iblk = squeeze(c_low(:,iblk,:));
                        d_upp_iblk = zeros(size(c_upp_iblk),'like',c_upp_iblk);
                        d_low_iblk = zeros(size(c_low_iblk),'like',c_low_iblk);
                        for iSample = 1:nSamples
                            d_upp_iblk(:,iSample) = dW0_T(1:ps,:,iblk)*c_upp_iblk(:,iSample);
                            d_low_iblk(:,iSample) = dU0_T(1:pa,:,iblk)*c_low_iblk(:,iSample);
                        end
                        dLdW(iAngle,iblk) = sum(bsxfun(@times,dldz_upp_iblk,d_upp_iblk),'all');
                        dLdW(nAngles/2+iAngle,iblk) = sum(bsxfun(@times,dldz_low_iblk,d_low_iblk),'all');
                    end
                end
            end
        end
        
        function nodcleak = get.NoDcLeakage(layer)
            nodcleak = layer.PrivateNoDcLeakage;
        end        
        
        function angles = get.Angles(layer)
            angles = layer.PrivateAngles;
        end
        
        function mus = get.Mus(layer)
            mus = layer.PrivateMus;
        end
        
        function layer = set.NoDcLeakage(layer,nodcleak)
            layer.PrivateNoDcLeakage = nodcleak;
            %
            layer.isUpdateRequested = true;
        end                
        
        function layer = set.Angles(layer,angles)
            nBlocks = prod(layer.NumberOfBlocks);
            nChsTotal = sum(layer.PrivateNumberOfChannels);
            nAngles = (nChsTotal-2)*nChsTotal/4;
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
            %
            if layer.NoDcLeakage
                layer.PrivateMus(1,:) = ones(1,size(layer.PrivateMus,2));           
                layer.PrivateAngles(1:ps-1,:) = ...
                    zeros(ps-1,size(layer.PrivateAngles,2),'like',layer.PrivateAngles);
            end      
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
            muW = mus(1:ps,:);
            muU = mus(ps+1:end,:);
            anglesW = angles(1:nAngles/2,:);
            anglesU = angles(nAngles/2+1:end,:);
            fcn_orthmtxgen = tansacnet.lsun.get_fcn_orthmtxgen(angles);                                    
            layer.W0T = permute(fcn_orthmtxgen(anglesW,muW),[2 1 3]);
            layer.U0T = permute(fcn_orthmtxgen(anglesU,muU),[2 1 3]);
            layer.isUpdateRequested = false;
        end
        
    end
    
end
"""
