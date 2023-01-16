function [matrix,matrixpst,matrixpre] = fcn_orthmtxgen_diff(...
    angles,mus,pdAng,matrixpst,matrixpre,useGpu,isLessThanR2021b) %#codegen
%FCN_ORTHMTXGEN_DIFF
%
% Function realization of
% saivdr.dictionary.utility.OrthonormalMatrixGenerationSystem
% for supporting dlarray (Deep learning array for custom training
% loops)
%
% Requirements: MATLAB R2020a
%
% Copyright (c) 2020-2022, Shogo MURAMATSU
%
% All rights reserved.
%
% Contact address: Shogo MURAMATSU,
%                Faculty of Engineering, Niigata University,
%                8050 2-no-cho Ikarashi, Nishi-ku,
%                Niigata, 950-2181, JAPAN
%
% http://msiplab.eng.niigata-u.ac.jp/

if nargin < 7
    isLessThanR2021b = false;
end
if nargin < 6
    useGpu = isgpuarray(angles);
end
if nargin < 3
    pdAng = 0;
end

nDim_ = (1+sqrt(1+8.*size(angles,1)))/2;
nMatrices_ = size(angles,2);
matrix = repmat(eye(nDim_,'like',angles),[1 1 nMatrices_]);
if useGpu
    %for iMtx = 1:nMatrices_
    matrixrev = repmat(eye(nDim_,'like',angles),[1 1 nMatrices_]);
    matrixdif = zeros(nDim_,nDim_,nMatrices_,'like',angles);
    iAng = uint32(1);
    for iTop=1:nDim_-1
        rt = matrixrev(iTop,:,:);
        dt = zeros(1,nDim_,nMatrices_,'like',angles);
        dt(1,iTop,:) = 1;
        for iBtm=iTop+1:nDim_
            if iAng == pdAng
                angle = angles(iAng,:); 
                %
                rb = matrixrev(iBtm,:,:);
                db = zeros(1,nDim_,nMatrices_,'like',angles);
                db(1,iBtm,:) = 1;
                dangle = angle + pi/2;
                %
                %[rt,rb] = rot_(rt,rb,-angle);
                %[dt,db] = rot_(dt,db,dangle);
                %[vt,vb] = rot_([rt;dt],[rb;db],[-angle;dangle],...
                %    useGpu,isLessThanR2021b);
                vt = [rt;dt];
                vb = [rb;db];
                angle_ = permute([-angle;dangle],[1 3 2]);
                c = cos(angle_);
                s = sin(angle_);
                %u  = arrayfun(@(s,vt,vb) s.*(vt+vb),s,vt,vb);
                %vt = arrayfun(@(c,s,vt,u) (c+s).*vt-u,c,s,vt,u);
                %vb = arrayfun(@(c,s,vb,u) (c-s).*vb+u,c,s,vb,u);
                u  = pagefun(@times,s,bsxfun(@plus,vt,vb));
                vt = bsxfun(@minus,pagefun(@times,bsxfun(@plus,c,s),vt),u);
                vb = bsxfun(@plus,pagefun(@times,bsxfun(@minus,c,s),vb),u);
                %
                matrixrev(iTop,:,:) = vt(1,:,:); %rt;
                matrixrev(iBtm,:,:) = vb(1,:,:); %rb;
                matrixdif(iTop,:,:) = vt(2,:,:); %dt;
                matrixdif(iBtm,:,:) = vb(2,:,:); %db;
                %
                matrixpst = pagefun(@mtimes,matrixpst,matrixrev);
                matrix = pagefun(@mtimes,pagefun(@mtimes,matrixpst,matrixdif),matrixpre);
                matrixpre = pagefun(@mtimes,pagefun(@transpose,matrixrev),matrixpre);
            end
            iAng = iAng + 1;
        end
    end
    %end
    if isvector(mus) || isscalar(mus)
        matrix = pagefun(@times,mus(:),matrix);
    else
        matrix = pagefun(@times,permute(mus,[1 3 2]),matrix);
    end
else
    for iMtx = 1:nMatrices_
        matrixrev = eye(nDim_,'like',angles);
        matrixdif = zeros(nDim_,'like',angles);
        iAng = uint32(1);
        for iTop=1:nDim_-1
            rt = matrixrev(iTop,:);
            dt = zeros(1,nDim_,'like',angles);
            dt(iTop) = 1;
            for iBtm=iTop+1:nDim_
                if iAng == pdAng
                    angle = angles(iAng,iMtx);
                    %
                    rb = matrixrev(iBtm,:);
                    db = zeros(1,nDim_,'like',angles);
                    db(iBtm) = 1;
                    dangle = angle + pi/2;
                    %
                    %[rt,rb] = rot_(rt,rb,-angle);
                    %[dt,db] = rot_(dt,db,dangle);
                    [vt,vb] = rot_([rt;dt],[rb;db],[-angle;dangle],...
                        isLessThanR2021b);
                    %
                    matrixrev(iTop,:) = vt(1,:); %rt;
                    matrixrev(iBtm,:) = vb(1,:); %rb;
                    matrixdif(iTop,:) = vt(2,:); %dt;
                    matrixdif(iBtm,:) = vb(2,:); %db;
                    %
                    matrixpst(:,:,iMtx) = matrixpst(:,:,iMtx)*matrixrev;
                    matrix(:,:,iMtx) = matrixpst(:,:,iMtx)*matrixdif*matrixpre(:,:,iMtx);
                    matrixpre(:,:,iMtx) = matrixrev.'*matrixpre(:,:,iMtx);
                end
                iAng = iAng + 1;
            end
        end
        if isvector(mus) || isscalar(mus)
            if isLessThanR2021b % on CPU
                matrix(:,:,iMtx) = bsxfun(@times,mus(:),matrix(:,:,iMtx));
            else % on CPU
                matrix(:,:,iMtx) = mus(:).*matrix(:,:,iMtx);
            end
        else
            if isLessThanR2021b % on CPU
                matrix(:,:,iMtx) = bsxfun(@times,mus(:,iMtx),matrix(:,:,iMtx));
            else % on CPU
                matrix(:,:,iMtx) = mus(:,iMtx).*matrix(:,:,iMtx);
            end
        end
    end
end
end

function [vt,vb] = rot_(vt,vb,angle,isLessThanR2021b)
c = cos(angle);
s = sin(angle);
if isLessThanR2021b
    u  = bsxfun(@times,s,bsxfun(@plus,vt,vb));
    vt = bsxfun(@minus,bsxfun(@times,c+s,vt),u);
    vb = bsxfun(@plus,bsxfun(@times,c-s,vb),u);
else
    u  = s.*(vt+vb);
    vt = (c+s).*vt-u;
    vb = (c-s).*vb+u;
end
end

