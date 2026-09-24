classdef ComplexButterflyCoefficientGenerationSystem < matlab.System %#codegen
    %COMPLEXBUTTERFLYCOEFFICIENTGENERATIONSYSTEM Complex butterfly matrix generator
    %
    %   Generates the complex butterfly matrix B_hat_k (eq. 9 of the
    %   paper "局所構造化ユニタリネットワークの複素拡張に関する検討",
    %   Ohata & Muramatsu), for 1-D CLSUN:
    %
    %       chat(theta) = [ -1j*cos(theta)  -1j*sin(theta) ;
    %                         cos(theta)      -sin(theta)   ]
    %
    %       shat(theta) = [   sin(theta)      cos(theta)    ;
    %                        1j*sin(theta)  -1j*cos(theta)  ]
    %
    %       Chat_k = blkdiag( chat(theta_0), ..., chat(theta_{floor(m/2)-1}) )
    %                  [ + scalar 1  appended as last diagonal entry, if m is odd ]
    %       Shat_k = blkdiag( shat(theta_0), ..., shat(theta_{floor(m/2)-1}) )
    %                  [ + scalar 1j appended as last diagonal entry, if m is odd ]
    %
    %       Bhat_k = [ Chat_k,  conj(Chat_k) ;
    %                  Shat_k,  conj(Shat_k) ]          (size 2m x 2m)
    %
    %   Bhat_k satisfies Bhat_k * Bhat_k^H = 2*eye(2m) (a scaled unitary
    %   matrix, consistent with the (1/2) prefactor of Q_k(z) in eq. 7
    %   of the paper).
    %
    %   ANGLES input: (floor(m/2) x nBlks) real matrix.
    %   Output: (2m x 2m x nBlks) complex array.
    %
    %   PartialDifference: since Chat_k/Shat_k are block-diagonal (no
    %   cascade between different angle indices), d(Bhat_k)/d(theta_p)
    %   is zero everywhere except the p-th 2x2 sub-blocks (and their
    %   conjugates), computed via the same chat/shat formulas with
    %   theta_p -> theta_p + pi/2.
    %
    % Requirements: MATLAB R2022a
    %
    % Copyright (c) 2026, Shogo MURAMATSU, Kohei OHATA
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
            matlab.system.StringSet({'on','off'});
        DeviceSet = ...
            matlab.system.StringSet({'cpu','cuda'});
        DTypeSet = ...
            matlab.system.StringSet({'single','double'});
    end

    properties
        NumberOfHalfChannels
        Device = 'cpu'
        DType = 'double'
    end

    methods
        function obj = ComplexButterflyCoefficientGenerationSystem(varargin)
            if canUseGPU
                obj.Device = 'cuda';
            end
            setProperties(obj,nargin,varargin{:});
        end
    end

    methods (Access = protected)

        function s = saveObjectImpl(obj)
            s = saveObjectImpl@matlab.System(obj);
            s.NumberOfHalfChannels = obj.NumberOfHalfChannels;
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
            obj.NumberOfHalfChannels = s.NumberOfHalfChannels;
            loadObjectImpl@matlab.System(obj,s,wasLocked);
        end

        function setupImpl(obj,~,~)
            if isempty(obj.NumberOfHalfChannels)
                errID = 'CLSUN:InvalidNumberOfHalfChannels';
                msg = 'NumberOfHalfChannels must be specified.';
                throw(MException(errID,msg));
            end
        end

        function validateInputsImpl(obj,angles,~)
            if obj.Device == "cuda" && ~isgpuarray(angles)
                errID = 'CLSUN:DeviceMismatch';
                msg = 'ANGLES should be gpuArray';
                throw(MException(errID,msg));
            elseif obj.Device == "cpu" && isgpuarray(angles)
                errID = 'CLSUN:DeviceMismatch';
                msg = 'ANGLES should be on CPU';
                throw(MException(errID,msg));
            end

            if isgpuarray(angles)
                angles = gather(angles);
            end
            if ~isempty(angles) && ~strcmp(obj.DType,class(angles))
                errID = 'CLSUN:DTypeMismatch';
                msg = char("ANGLES should be " + obj.DType);
                throw(MException(errID,msg));
            end
        end

        function matrix = stepImpl(obj,angles,pdAng)
            if nargin < 3
                pdAng = 0;
            end
            matrix = obj.generate_(angles,pdAng);
        end

        function N = getNumInputsImpl(obj)
            if strcmp(obj.PartialDifference,'on')
                N = 2;
            else
                N = 1;
            end
        end

        function N = getNumOutputsImpl(~)
            N = 1;
        end

    end

    methods (Access = private)

        function matrix = generate_(obj,angles,pdAng)
            device_ = obj.Device;
            dtype_ = obj.DType;
            m = obj.NumberOfHalfChannels;
            nAngles = floor(m/2);
            isOddM = (mod(m,2) == 1);
            nBlks = size(angles,2);

            if device_ == "cuda"
                matrix = complex(zeros(2*m,2*m,nBlks,dtype_,"gpuArray"));
            else
                matrix = complex(zeros(2*m,2*m,nBlks,dtype_));
            end

            for iBlk = 1:nBlks
                if device_ == "cuda"
                    Chat = complex(zeros(m,m,dtype_,"gpuArray"));
                    Shat = complex(zeros(m,m,dtype_,"gpuArray"));
                else
                    Chat = complex(zeros(m,m,dtype_));
                    Shat = complex(zeros(m,m,dtype_));
                end
                for p = 1:nAngles
                    if pdAng == 0 || p == pdAng
                        theta = angles(p,iBlk);
                        if p == pdAng
                            theta = theta + pi/2;
                        end
                        [chatBlk,shatBlk] = obj.chatshat_(theta);
                        idx = (2*p-1):(2*p);
                        Chat(idx,idx) = chatBlk;
                        Shat(idx,idx) = shatBlk;
                    end
                    % else: left as zero (does not depend on theta_pdAng)
                end
                if isOddM && pdAng == 0
                    Chat(m,m) = 1;
                    Shat(m,m) = 1j;
                end
                % else (isOddM && pdAng~=0): the extra scalar entries are
                % constants w.r.t. any angle, so their derivative is
                % zero -- already the case since Chat/Shat were
                % initialized to zero.
                matrix(:,:,iBlk) = [Chat, conj(Chat) ; Shat, conj(Shat)];
            end
        end

    end

    methods (Static, Access = private)
        function [chatBlk,shatBlk] = chatshat_(theta)
            c = cos(theta);
            s = sin(theta);
            chatBlk = [ -1j*c, -1j*s ; c, -s ];
            shatBlk = [    s,     c ; 1j*s, -1j*c ];
        end
    end

end
