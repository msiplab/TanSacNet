classdef OrthonormalMatrixGenerationSystem < matlab.System %#codegen
    %ORTHONORMALMATRIXGENERATIONSYSTEM Orthonormal matrix generator
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2014-2024, Shogo MURAMATSU, Yasas GODAGE
    %
    % All rights reserved.
    %
    % Contact address: Shogo MURAMATSU, Yasas GODAGE
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
        DeviceSet = ...
            matlab.system.StringSet({'cpu','cuda'});
        DTypeSet = ...
            matlab.system.StringSet({'single','double'});
    end
    
    properties
        NumberOfDimensions
        Device = 'cuda'
        DType = 'double'
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
            s.Device = obj.Device;
            s.DType = obj.DType;
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            if isfield(s,'PartialDifference')
                obj.PartialDifference = s.PartialDifference;
            else
                obj.PartialDifference = 'off';
            end
            obj.DType = s.DType;
            obj.Device = s.Device;            
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

        function validateInputsImpl(obj,angles,mus,~)
            % Check device of angles
            if obj.Device == "cuda" && ~isgpuarray(angles)
                errID = 'LSUN:DeviceMismatch';
                msg = 'ANGLES should be gpuArray';
                throw(MException(errID,msg));
            elseif obj.Device == "cpu" && isgpuarray(angles)
                errID = 'LSUN:DeviceMismatch';
                msg = 'ANGLES should be on CPU';
                throw(MException(errID,msg));
            end

            % Check device of mus
            if obj.Device == "cuda" && ~isgpuarray(mus)
                errID = 'LSUN:DeviceMismatch';
                msg = 'MUS should be gpuArray';
                throw(MException(errID,msg));
            elseif obj.Device == "cpu" && isgpuarray(mus)
                errID = 'LSUN:DeviceMismatch';
                msg = 'MUS should be on CPU';
                throw(MException(errID,msg));
            end

            % Check dtype of angles
            if isgpuarray(angles)
                angles = gather(angles);
            end
            if ~strcmp(obj.DType,class(angles))
                errID = 'LSUN:DTypeMismatch';
                msg = char("ANGLES should be " + obj.DType);
                throw(MException(errID,msg));
            end

            % Check dtype of angles
            if isgpuarray(mus)
                mus = gather(mus);
            end
            if ~strcmp(obj.DType,class(mus))
                errID = 'LSUN:DTypeMismatch';
                msg = char("MUS should be " + obj.DType);
                throw(MException(errID,msg));
            end

            if ~isempty(mus) && any(abs(mus(:))~=1)
                error('All entries of mus must be 1 or -1.');
            end
        end
        
        function matrix = stepImpl(obj,angles,mus,pdAng)
            device_ = obj.Device;
            dtype_ = obj.DType;

            if nargin < 4
                pdAng = 0;
            end
            
            if isempty(angles)
                if isvector(mus) % Single case
                    matrix = diag(mus); 
                else % Multiple case
                    if device_ == "cuda"
                        matrix = zeros(size(mus,1),size(mus,1),size(mus,2),dtype_,"gpuArray");
                    else
                        matrix = zeros(size(mus,1),size(mus,1),size(mus,2),dtype_);
                    end
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
            device_ = obj.Device;
            dtype_ = obj.DType;
            nDim_ = obj.NumberOfDimensions;
            nMatrices_ = size(angles,2);
            if device_ == "cuda"
                eye_ = eye(nDim_,dtype_,"gpuArray");
            else
                eye_ = eye(nDim_,dtype_);
            end
            matrix = repmat(eye_,[1 1 nMatrices_]); 
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
                            matrix(:,:,iMtx) = cast(0,dtype_)*matrix(:,:,iMtx);
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
            device_ = obj.Device;
            dtype_ = obj.DType;
            % Check pdAng
            if pdAng ~= obj.nextangle
                error("Unable to proceed sequential differentiation. Index = %d is expected, but %d was given.", obj.nextangle, pdAng);
            end
            %
            nDim_ = obj.NumberOfDimensions;
            nMatrices_ = size(angles,2);
            if device_ == "cuda"
                eye_ = eye(nDim_,dtype_,"gpuArray"); 
                zeros_ = zeros(nDim_,dtype_,"gpuArray"); 
            else
                eye_ = eye(nDim_,dtype_); 
                zeros_ = zeros(nDim_,dtype_); 
            end
            matrix = repmat(eye_,[1 1 nMatrices_]);
            if isrow(mus)
                mus = mus.';
            end
            if pdAng < 1 % Initialization
                obj.matrixpst = repmat(eye_,[1 1 nMatrices_]); 
                obj.matrixpre = repmat(eye_,[1 1 nMatrices_]); 
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
                if device_ == "cuda"
                    zeros_1xD_ = zeros(1,nDim_,dtype_,"gpuArray");
                else
                    zeros_1xD_ = zeros(1,nDim_,dtype_);
                end
                %
                %matrix = 1;
                for iMtx = 1:nMatrices_
                    matrixrev = eye_; 
                    matrixdif = zeros_; 
                    %
                    iAng = 1;
                    for iTop=1:nDim_-1
                        rt = matrixrev(iTop,:);
                        dt = zeros_1xD_; 
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
                                db = zeros_1xD_; 
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
