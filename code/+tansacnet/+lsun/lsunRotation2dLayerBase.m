classdef (Abstract) lsunRotation2dLayerBase < nnet.layer.Layer %#codegen
    %LSUNROTATION2DLAYERBASE
    %
    % Requirements: MATLAB R2022a
    %
    % Copyright (c) 2022-2026, Shogo MURAMATSU
    %
    % All rights reserved.
    %
    % Contact address: Shogo MURAMATSU,
    %                Faculty of Engineering, Niigata University,
    %                8050 2-no-cho Ikarashi, Nishi-ku,
    %                Niigata, 950-2181, JAPAN
    %
    % http://msiplab.eng.niigata-u.ac.jp/

    methods (Access = protected)

        function Zout = applyBlockwiseMatrix(~, A_, Vin)
            %APPLYBLOCKWISEMATRIX
            %   Apply a per-block matrix A_ (p x p x nBlks) to a per-block,
            %   per-sample array Vin (p x nBlks x　nSamples), block by block.

            pa = size(Vin,1);
            nBlks = size(Vin,2);
            nSamples = size(Vin,3);
            Zout = zeros(pa,nBlks,nSamples,'like',Vin);
            for iSample = 1:nSamples
                if isgpuarray(Vin)
                    Vin_iSample = permute(Vin(:,:,iSample),[1 4 2 3]);
                    Zout_iSample = pagefun(@mtimes,A_,Vin_iSample);
                    Zout(:,:,iSample) = ipermute(Zout_iSample,[1 4 2 3]);
                else
                    for iblk = 1:nBlks
                        Zout(:,iblk,iSample) = A_(:,:,iblk)*Vin(:,iblk,iSample);
                    end
                end
            end
        end

        function dLdTheta = computeAngleGradient(layer, A_, angles, mus, c_low, dldz_low)
            %COMPUTEANGLEGRADIENT Accumulate
            %   dL/dTheta_i = <dLdZ,(dU/dTheta_i)X>
            %   for every rotation angle Theta_i, given the orthonormal
            %   matrix A_ built from (angles, mus). layer.Mode
            %   ('Analysis' or 'Synthesis') selects the orientation of
            %   the per-angle derivative, matching the layer's own
            %   forward-application convention.

            pa = size(A_,1);
            nBlks = size(c_low,2);
            nSamples = size(c_low,3);
            dAPst = bsxfun(@times,permute(mus,[1 3 2]),A_);
            dAPre = repmat(eye(pa,'like',A_),[1 1 nBlks]);
            fcn_orthmtxgen_diff = tansacnet.lsun.get_fcn_orthmtxgen_diff(angles);
            nAngles = size(angles,1);
            dLdTheta = zeros(nAngles,nBlks,'like',dldz_low);
            isAnalysis = strcmp(layer.Mode,'Analysis');
            for iAngle = uint32(1:nAngles)
                [dA,dAPst,dAPre] = fcn_orthmtxgen_diff(angles,mus,iAngle,dAPst,dAPre);
                if isAnalysis
                    dA_ = dA;
                else
                    dA_ = permute(dA,[2 1 3]);
                end
                if isgpuarray(c_low)
                    c_low_ext = permute(c_low,[1 4 2 3]); % idx 1 iblk iSample
                    d_low_ext = pagefun(@mtimes,dA_,c_low_ext); % idx 1 iblk iSample
                    d_low = ipermute(d_low_ext,[1 4 2 3]);
                    dLdTheta(iAngle,:) = sum(bsxfun(@times,dldz_low,d_low),[1 3]);
                else
                    for iblk = 1:nBlks
                        dA_iblk = dA_(:,:,iblk);
                        dldz_low_iblk = squeeze(dldz_low(:,iblk,:));
                        c_low_iblk = squeeze(c_low(:,iblk,:));
                        d_low_iblk = zeros(size(c_low_iblk),'like',c_low_iblk);
                        for iSample = 1:nSamples
                            d_low_iblk(:,iSample) = dA_iblk*c_low_iblk(:,iSample);
                        end
                        dLdTheta(iAngle,iblk) = sum(bsxfun(@times,dldz_low_iblk,d_low_iblk),'all');
                    end
                end
            end
        end

    end

end
