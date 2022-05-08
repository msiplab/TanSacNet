function matrix = fcn_orthmtxgen(angles,mus,useGpu,isLessThanR2021b) %#codegen
%FCN_ORTHMTXGEN
%
% Function realization of
% tansacnet.utility.OrthonormalMatrixGenerationSystem
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
if nargin < 4
    isLessThanR2021b = false;
end
if nargin < 3
    useGpu = isgpuarray(angles);
end
nDim_ = (1+sqrt(1+8*size(angles,1)))/2;
nMatrices_ = size(angles,2);
matrix = repmat(eye(nDim_,'like',angles),[1 1 nMatrices_]);
if useGpu
    if ~isempty(angles)
        %for iMtx = 1:nMatrices_
        iAng = uint32(1);
        for iTop=1:nDim_-1
            vt = matrix(iTop,:,:);
            for iBtm=iTop+1:nDim_
                angle = permute(angles(iAng,:),[1 3 2]);
                %
                c = cos(angle);
                s = sin(angle);
                vb = matrix(iBtm,:,:);
                %
                u = pagefun(@times,s,bsxfun(@plus,vt,vb));
                vt = bsxfun(@minus,pagefun(@times,bsxfun(@plus,c,s),vt),u);
                matrix(iBtm,:,:) = bsxfun(@plus,pagefun(@times,bsxfun(@minus,c,s),vb),u);
                %u  = arrayfun(@(s,vt,vb) s.*(vt+vb),s,vt,vb);
                %vt = arrayfun(@(c,s,vt,u) (c+s).*vt-u,c,s,vt,u);
                %matrix(iBtm,:,:) = arrayfun(@(c,s,vb,u) (c-s).*vb+u,c,s,vb,u);
                %
                iAng = iAng + 1;
            end
            matrix(iTop,:,:) = vt;
        end
        %end
    end
    if isvector(mus) || isscalar(mus)
        matrix = pagefun(@times,permute(mus(:),[1 3 2]),matrix);
    else
        matrix = pagefun(@times,permute(mus,[1 3 2]),matrix);
    end
else
    if ~isempty(angles)
        for iMtx = 1:nMatrices_
            iAng = uint32(1);
            for iTop=1:nDim_-1
                vt = matrix(iTop,:,iMtx);
                for iBtm=iTop+1:nDim_
                    angle = angles(iAng,iMtx);
                    if angle ~= 0
                        c = cos(angle);
                        s = sin(angle);
                        vb = matrix(iBtm,:,iMtx);
                        if isLessThanR2021b % on CPU
                            u  = bsxfun(@times,s,bsxfun(@plus,vt,vb));
                            vt = bsxfun(@minus,bsxfun(@times,c+s,vt),u);
                            matrix(iBtm,:,iMtx) = bsxfun(@plus,bsxfun(@times,c-s,vb),u);
                        else % on CPU
                            u  = s.*(vt+vb);
                            vt = (c+s).*vt-u;
                            matrix(iBtm,:,iMtx) = (c-s).*vb+u;
                        end
                    end
                    %
                    iAng = iAng + 1;
                end
                matrix(iTop,:,iMtx) = vt;
            end
        end
    end
    for iMtx = 1:nMatrices_
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
