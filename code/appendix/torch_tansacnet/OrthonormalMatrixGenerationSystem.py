import torch
import math

def rot_(vt, vb, angle):
    """ Planar rotation """
    c = torch.cos(angle)
    s = torch.sin(angle)
    u = s * (vt + vb)
    vt = (c + s) * vt
    vb = (c - s) * vb
    vt = vt - u
    vb = vb + u
    return vt, vb

class OrthonormalMatrixGenerationSystem:
    """
    ORTHONORMALMATRIXGENERATIONSYSTEM Orthonormal matrix generator
    
    Requirements: Python 3.10/10.x, PyTorch 2.3.x
    
    Copyright (c) 2024, Shogo MURAMATSU
    
    All rights reserved.
    
    Contact address: Shogo MURAMATSU,
                    Faculty of Engineering, Niigata University,
                    8050 2-no-cho Ikarashi, Nishi-ku,
                    Niigata, 950-2181, JAPAN
    
    http://www.eng.niigata-u.ac.jp/~msiplab/
    """

    def __init__(self,
                 dtype=torch.get_default_dtype(),
                 partial_difference=False):
        super(OrthonormalMatrixGenerationSystem, self).__init__()
        self.dtype = dtype
        self.partial_difference = partial_difference

    def __call__(self,
                 angles=0, 
                 mus=1, 
                 index_pd_angle=None):
        """
        The output is set on the same device with the angles
        """

        # Number of angles
        if isinstance(angles,list) and len(angles) == 0:
            if isinstance(mus, int) or isinstance(mus, float):
                nMatrices_ = 1
                nDims_ = 1
            elif not torch.is_tensor(mus):
                nMatrices_ = len(mus)
                nDims_ = len(mus[0])
            else:
                nMatrices_ = mus.size(0)
                nDims_ = mus.size(1)
            nAngles_ = int(nDims_*(nDims_-1)/2)
            angles = torch.zeros(nMatrices_,nAngles_)   
        if isinstance(angles, int) or isinstance(angles, float):
            angle_ = angles
            angles = torch.zeros(1,1,dtype=self.dtype)
            angles[0,0] = angle_
        elif not torch.is_tensor(angles):
            angles = torch.tensor(angles,dtype=self.dtype)
        else:
            angles = angles.to(dtype=self.dtype) 
        nAngles = angles.size(1)
        nMatrices = angles.size(0)

        # Number of dimensions
        if not hasattr(self, 'number_of_dimensions'):
            self.number_of_dimensions = int(1+math.sqrt(1+8*nAngles)/2)
        nDims = self.number_of_dimensions

        # Setup of mus, which is send to the same device with angles
        if isinstance(mus, int) or isinstance(mus, float):
            mu_ = mus
            mus = torch.ones(nMatrices,nDims,dtype=self.dtype,device=angles.device)
            mus = mu_ * mus
        elif not torch.is_tensor(mus): #isinstance(mus, list):
            mus = torch.tensor(mus,dtype=self.dtype,device=angles.device)
        else:
            mus = mus.to(dtype=self.dtype,device=angles.device) 

        # 
        matrix = self.stepNormal_(angles,mus) #, mus, index_pd_angle)
        
        return matrix.clone()
    
    def stepNormal_(self, angles, mus): # index_pd_angle):
        nDims = self.number_of_dimensions
        nMatrices = angles.size(0)
        matrix = torch.tile(torch.eye(nDims,dtype=self.dtype,device=angles.device), (nMatrices,1, 1))
        # 
        for iMtx in range(nMatrices):
            iAng = 0
            matrix_iMtx = matrix[iMtx]
            angles_iMtx = angles[iMtx]
            mus_iMtx = mus[iMtx]
            for iTop in range(nDims - 1):
                vt = matrix_iMtx[iTop, :]
                for iBtm in range(iTop + 1, nDims):
                    angle = angles_iMtx[iAng]
                    #if iAng == index_pd_angle:
                    #    angle = angle + torch.pi / 2
                    vb = matrix_iMtx[iBtm, :]
                    vt, vb = rot_(vt, vb, angle)
                    #if iAng == index_pd_angle:
                    #   matrix[:, :, iMtx] = torch.zeros_like(matrix[:, :, iMtx])
                    matrix_iMtx[iBtm, :] = vb
                    iAng += 1
                matrix_iMtx[iTop, :] = vt
            matrix[iMtx] = mus_iMtx.view(-1,1) * matrix_iMtx

        return matrix

    """
    def stepNormal_(self, angles, mus, index_pd_angle):
        nDim_ = self.NumberOfDimensions
        nMatrices_ = angles.shape[1]
        matrix = torch.tile(torch.eye(nDim_), (1, 1, nMatrices_))
        if mus.ndim == 1:
            mus = mus[:, torch.newaxis]
        for iMtx in range(nMatrices_):
            iAng = 0
            for iTop in range(nDim_ - 1):
                vt = matrix[iTop, :, iMtx]
                for iBtm in range(iTop + 1, nDim_):
                    angle = angles[iAng, iMtx]
                    if iAng == pdAng:
                        angle = angle + torch.pi / 2
                    vb = matrix[iBtm, :, iMtx]
                    vt, vb = self.rot_(vt, vb, angle)
                    if iAng == pdAng:
                        matrix[:, :, iMtx] = torch.zeros_like(matrix[:, :, iMtx])
                    matrix[iBtm, :, iMtx] = vb
                    iAng += 1
                matrix[iTop, :, iMtx] = vt
            if mus.ndim == 2 or mus.size == 1:
                matrix[:, :, iMtx] = mus * matrix[:, :, iMtx]
            else:
                matrix[:, :, iMtx] = mus[:, iMtx] * matrix[:, :, iMtx]
        return matrix

    def stepSequential_(self, angles, mus, pdAng):
        if pdAng != self.nextangle:
            raise ValueError("Unable to proceed sequential differentiation. Index = %d is expected, but %d was given." % (self.nextangle, pdAng))
        nDim_ = self.NumberOfDimensions
        nMatrices_ = angles.shape[1]
        matrix = torch.tile(torch.eye(nDim_), (1, 1, nMatrices_))
        if mus.ndim == 1:
            mus = mus[:, torch.newaxis]
        if pdAng < 1:
            self.matrixpst = torch.tile(torch.eye(nDim_), (1, 1, nMatrices_))
            self.matrixpre = torch.tile(torch.eye(nDim_), (1, 1, nMatrices_))
            for iMtx in range(nMatrices_):
                iAng = 0
                for iTop in range(nDim_ - 1):
                    vt = self.matrixpst[iTop, :, iMtx]
                    for iBtm in range(iTop + 1, nDim_):
                        angle = angles[iAng, iMtx]
                        vb = self.matrixpst[iBtm, :, iMtx]
                        vt, vb = self.rot_(vt, vb, angle)
                        self.matrixpst[iBtm, :, iMtx] = vb
                        iAng += 1
                    self.matrixpst[iTop, :, iMtx] = vt
                if mus.ndim == 2:
                    matrix[:, :, iMtx] = mus * self.matrixpst[:, :, iMtx]
                else:
                    matrix[:, :, iMtx] = mus[:, iMtx] * self.matrixpst[:, :, iMtx]
            self.nextangle = 1
        else:
            for iMtx in range(nMatrices_):
                matrixrev = torch.eye(nDim_)
                matrixdif = torch.zeros((nDim_, nDim_))
                iAng = 0
                for iTop in range(nDim_ - 1):
                    rt = matrixrev[iTop, :]
                    dt = torch.zeros(nDim_)
                    dt[iTop] = 1
                    for iBtm in range(iTop + 1, nDim_):
                        if iAng == pdAng:
                            angle = angles[iAng, iMtx]
                            rb = matrixrev[iBtm, :]
                            rt, rb = self.rot_(rt, rb, -angle)
                            matrixrev[iTop, :] = rt
                            matrixrev[iBtm, :] = rb
                            db = torch.zeros(nDim_)
                            db[iBtm] = 1
                            dangle = angle + torch.pi / 2
                            dt, db = self.rot_(dt, db, dangle)
                            matrixdif[iTop, :] = dt
                            matrixdif[iBtm, :] = db
                            self.matrixpst[:, :, iMtx] = self.matrixpst[:, :, iMtx] @ matrixrev
                            matrix[:, :, iMtx] = self.matrixpst[:, :, iMtx] @ matrixdif @ self.matrixpre[:, :, iMtx]
                            self.matrixpre[:, :, iMtx] = matrixrev.T @ self.matrixpre[:, :, iMtx]
                        iAng += 1
                if mus.ndim == 2:
                    matrix[:, :, iMtx] = mus * matrix[:, :, iMtx]
                else:
                    matrix[:, :, iMtx] = mus[:, iMtx] * matrix[:, :, iMtx]
            self.nextangle += 1
        return matrix
"""

"""
classdef OrthonormalMatrixGenerationSystem < matlab.System %#codegen
    %ORTHONORMALMATRIXGENERATIONSYSTEM Orthonormal matrix generator
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2014-2022, Shogo MURAMATSU
    %
    % All rights reserved.
    %
    % Contact address: Shogo MURAMATSU,
    %                Faculty of Engineering, Niigata University,
    %                8050 2-no-cho Ikarashi, Nishi-ku,
    %                Niigata, 950-2181, JAPAN
    %
    % http://msiplab.eng.niigata-u.ac.jp/
    %
    
    properties (Nontunable)
        PartialDifference = 'off'
    end
    
    properties (Hidden, Transient)
        PartialDifferenceSet = ...
            matlab.system.StringSet({'on','off','sequential'});
    end
    
    properties
        NumberOfDimensions
    end
    
    properties (Access = private)
        matrixpst
        matrixpre
        nextangle
    end
    
    methods
        function obj = OrthonormalMatrixGenerationSystem(varargin)
            % Support name-value pair arguments
            setProperties(obj,nargin,varargin{:});
        end
    end
    
    methods (Access = protected)
        
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@matlab.System(obj);
            s.NumberOfDimensions = obj.NumberOfDimensions;
            s.PartialDifference = obj.PartialDifference;
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            if isfield(s,'PartialDifference')
                obj.PartialDifference = s.PartialDifference;
            else
                obj.PartialDifference = 'off';
            end
            obj.NumberOfDimensions = s.NumberOfDimensions;
            loadObjectImpl@matlab.System(obj,s,wasLocked);
        end
        
        function setupImpl(obj,angles,~,~)
            if isempty(obj.NumberOfDimensions)
                obj.NumberOfDimensions = (1+sqrt(1+8*size(angles,1)))/2;
            end
            if strcmp(obj.PartialDifference,'sequential')
                obj.nextangle = uint32(0);            
            end
        end
        
        function resetImpl(obj)
            obj.nextangle = uint32(0);            
        end
        
        function validateInputsImpl(~,~,mus,~)
            if ~isempty(mus) && any(abs(mus(:))~=1)
                error('All entries of mus must be 1 or -1.');
            end
        end
        
        function matrix = stepImpl(obj,angles,mus,pdAng)
            
            if nargin < 4
                pdAng = 0;
            end
            
            if isempty(angles)
                if isvector(mus) % Single case
                    matrix = diag(mus); 
                else % Multiple case
                    matrix = zeros(size(mus,1),size(mus,1),size(mus,2));
                    for idx = 1:size(mus,2)
                        matrix(:,:,idx) = diag(mus(:,idx));
                    end
                end
            elseif strcmp(obj.PartialDifference,'sequential')
                % Sequential mode
                matrix = obj.stepSequential_(angles,mus,pdAng);
            else
                % Normal mode
                matrix = obj.stepNormal_(angles,mus,pdAng);
            end
        end
        
        function N = getNumInputsImpl(obj)
            if strcmp(obj.PartialDifference,'on') || ...
                    strcmp(obj.PartialDifference,'sequential')
                N = 3;
            else
                N = 2;
            end
        end
        
        function N = getNumOutputsImpl(~)
            N = 1;
        end
    end
    
    methods (Access = private)
        
        function matrix = stepNormal_(obj,angles,mus,pdAng)
            nDim_ = obj.NumberOfDimensions;
            nMatrices_ = size(angles,2);
            matrix = repmat(eye(nDim_),[1 1 nMatrices_]);
            if isrow(mus)
                mus = mus.';
            end
            for iMtx = 1:nMatrices_
                iAng = 1;
                for iTop=1:nDim_-1
                    vt = matrix(iTop,:,iMtx);
                    for iBtm=iTop+1:nDim_
                        angle = angles(iAng,iMtx);
                        if iAng == pdAng
                            angle = angle + pi/2;
                        end
                        vb = matrix(iBtm,:,iMtx);
                        [vt,vb] = obj.rot_(vt,vb,angle);
                        if iAng == pdAng
                            matrix(:,:,iMtx) = 0*matrix(:,:,iMtx);
                        end
                        matrix(iBtm,:,iMtx) = vb;
                        %
                        iAng = iAng + 1;
                    end
                    matrix(iTop,:,iMtx) = vt;
                end
                if iscolumn(mus) || isscalar(mus)
                    matrix(:,:,iMtx) = mus.*matrix(:,:,iMtx);
                else
                    matrix(:,:,iMtx) = mus(:,iMtx).*matrix(:,:,iMtx);
                end
            end
        end

        function matrix = stepSequential_(obj,angles,mus,pdAng)
            % Check pdAng
            if pdAng ~= obj.nextangle
                error("Unable to proceed sequential differentiation. Index = %d is expected, but %d was given.", obj.nextangle, pdAng);
            end
            %
            nDim_ = obj.NumberOfDimensions;
            nMatrices_ = size(angles,2);
            matrix = repmat(eye(nDim_),[1 1 nMatrices_]);
            if isrow(mus)
                mus = mus.';
            end
            if pdAng < 1 % Initialization
                obj.matrixpst = repmat(eye(nDim_),[1 1 nMatrices_]);
                obj.matrixpre = repmat(eye(nDim_),[1 1 nMatrices_]);
                %
                for iMtx = 1:nMatrices_
                    iAng = 1;
                    for iTop=1:nDim_-1
                        vt = obj.matrixpst(iTop,:,iMtx);
                        for iBtm=iTop+1:nDim_
                            angle = angles(iAng,iMtx);
                            vb = obj.matrixpst(iBtm,:,iMtx);
                            [vt,vb] = obj.rot_(vt,vb,angle);
                            obj.matrixpst(iBtm,:,iMtx) = vb;
                            iAng = iAng + 1;
                        end
                        obj.matrixpst(iTop,:,iMtx) = vt;
                    end
                    if iscolumn(mus)
                        matrix(:,:,iMtx) = mus.*obj.matrixpst(:,:,iMtx);
                    else
                        matrix(:,:,iMtx) = mus(:,iMtx).*obj.matrixpst(:,:,iMtx);
                    end
                end
                obj.nextangle = uint32(1);
            else % Sequential differentiation
                %
                %matrix = 1;
                for iMtx = 1:nMatrices_
                    matrixrev = eye(nDim_);
                    matrixdif = zeros(nDim_);
                    %
                    iAng = 1;
                    for iTop=1:nDim_-1
                        rt = matrixrev(iTop,:);
                        dt = zeros(1,nDim_);
                        dt(iTop) = 1;
                        for iBtm=iTop+1:nDim_
                            if iAng == pdAng
                                angle = angles(iAng,iMtx);
                                %
                                rb = matrixrev(iBtm,:);
                                [rt,rb] = obj.rot_(rt,rb,-angle);
                                matrixrev(iTop,:) = rt;
                                matrixrev(iBtm,:) = rb;
                                %
                                db = zeros(1,nDim_);
                                db(iBtm) = 1;
                                dangle = angle + pi/2;
                                [dt,db] = obj.rot_(dt,db,dangle);
                                matrixdif(iTop,:) = dt;
                                matrixdif(iBtm,:) = db;
                                %
                                obj.matrixpst(:,:,iMtx) = obj.matrixpst(:,:,iMtx)*matrixrev;
                                matrix(:,:,iMtx) = obj.matrixpst(:,:,iMtx)*matrixdif*obj.matrixpre(:,:,iMtx);
                                obj.matrixpre(:,:,iMtx) = matrixrev.'*obj.matrixpre(:,:,iMtx);
                            end
                            iAng = iAng + 1;
                        end
                    end
                    if iscolumn(mus)
                        matrix(:,:,iMtx) = mus.*matrix(:,:,iMtx);
                    else
                        matrix(:,:,iMtx) = mus(:,iMtx).*matrix(:,:,iMtx);
                    end                    
                end
                obj.nextangle = obj.nextangle + 1;
            end
        end
    end
    
    methods (Static, Access = private)
        function [vt,vb] = rot_(vt,vb,angle)
            c = cos(angle);
            s = sin(angle);
            %
            u  = s*(vt + vb);
            vt = (c + s)*vt;
            vb = (c - s)*vb;
            vt = vt - u;
            vb = vb + u;
        end
    end
end
"""