import torch
import torch.nn as nn
import math
from lsunUtility import Direction
from orthonormalTransform import SetOfOrthonormalTransforms

class LsunIntermediateRotation2dLayer(nn.Module):
    """
    LSUNINTERMEDIATEROTATION2DLAYER

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
                 device=torch.get_default_device(),
                 mode='Synthesis',
                 stride=None,
                 number_of_blocks=[1,1],
                 no_dc_leakage=False,
                 name=''):
        super(LsunIntermediateRotation2dLayer, self).__init__()
        self.dtype = dtype
        self.device = device
        self.mode = mode
        self.stride = stride
        self.number_of_blocks = number_of_blocks
        self.name = name
        self.mus = None
        self.angles = None
        self.no_dc_leakage = no_dc_leakage
        ps = math.ceil(math.prod(self.stride)/2.0)
        pa = math.floor(math.prod(self.stride)/2.0)
        self.description = self.mode \
            + ' LSUN intermediate rotation ' \
            + '(ps,pa) = (' + str(ps) + ',' + str(pa) + ')'
        
        # Orthonormal matrix generation system
        nblks = self.number_of_blocks[Direction.VERTICAL]*self.number_of_blocks[Direction.HORIZONTAL]
        if self.mode == 'Synthesis':
            self.orthTransUnx = SetOfOrthonormalTransforms(name=self.name+"_UnT",nblks=nblks,n=ps,mode='Synthesis',device=self.device,dtype=self.dtype)
        else:
            self.orthTransUnx = SetOfOrthonormalTransforms(name=self.name+"_Un",nblks=nblks,n=ps,mode='Analysis',device=self.device,dtype=self.dtype)
        self.orthTransUnx.angles = nn.init.zeros_(self.orthTransUnx.angles).to(self.device)

        # Update parameters
        self.update_parameters()

    def forward(self, X):
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
        Zs = Y[:,:ps,:].clone()
        Za = self.orthTransUnx(Y[:,ps:,:])
        Z = torch.cat((Zs,Za),dim=1).reshape(nrows,ncols,ps+pa,nSamples).permute(3,0,1,2)

        return Z
    
    @property
    def angles(self):
        return self.orthTransUnx.angles
    
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
        ps = math.ceil(nDecs/2.0)
        if mus is None:
            mus = torch.ones(nBlocks,ps,dtype=self.dtype)
        elif isinstance(mus, int) or isinstance(mus, float):
            mus = mus*torch.ones(nBlocks,ps,dtype=self.dtype)
        self.__mus = mus.to(self.device)
        self.is_update_requested = True
    
    def update_parameters(self):
        angles = self.__angles
        mus = self.__mus
        if angles is None:
            self.orthTransUnx.angles = nn.init.zeros_(self.orthTransUnx.angles).to(self.device)
        else:
            self.orthTransUnx.angles = angles
        self.orthTransUnx.mus = mus
        #
        self.is_update_requested = False
        
"""
    
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
            parse(p,varargin{:})
            
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
                throw(MException('NsoltLayer:InvalidMode',...
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
            layer.Un = fcn_orthmtxgen(anglesU,musU);
            layer.isUpdateRequested = false;
        end
        
    end

end
"""
