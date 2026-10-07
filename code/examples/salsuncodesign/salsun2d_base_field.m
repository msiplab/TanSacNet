function [uf,b] = salsun2d_base_field(u,options)
%SALSUN2D_BASE_FIELD Separate a sequence into a base field and fluctuations
%
%   [uf,b] = salsun2d_base_field(u) splits the frames u (szy x szx x N)
%   into a base field b and the fluctuation uf = u - b, so that u can be
%   recovered as uf + b. The SA-LSUN processes uf; the reconstruction adds
%   b back.
%
%   Options:
%     Method - 'iir' (default): causal first-order IIR (leaky integrator)
%              of the base, s_t = Rho*s_(t-1) + (1-Rho)*base(u_t). Frame t
%              uses the state before its own update, b_t = s_(t-1), so
%              that b_t depends only on frames 1..t-1 and can be computed
%              by a streaming analyzer and by the synthesizer (open loop:
%              from the original frames). The state starts from the first
%              frame, so uf(:,:,1) is zero.
%              'batch': the time average of the base over all frames
%              (not causal; needs the whole sequence).
%              'none': b = 0.
%     Rho    - forgetting factor of the IIR, 0 <= Rho < 1 (default 0.9);
%              the time constant is about 1/(1-Rho) frames.
%     Scope  - 'dc' (default): the base is the block-DC part of the frame
%              (the mean of each Stride block, repeated over the block),
%              which carries almost all of the energy of the time average
%              of the wave data and costs 1/prod(Stride) of the full field.
%              'full': the base is the whole frame.
%     Stride - block size for Scope 'dc' (default [4 4]).
%
% Requirements: MATLAB R2026b
%
% Copyright (c) 2026, Shogo MURAMATSU
%
% All rights reserved.
%
% Contact address: Shogo MURAMATSU,
%    Faculty of Engineering, Niigata University,
%    8050 2-no-cho Ikarashi, Nishi-ku,
%    Niigata, 950-2181, JAPAN
%
% http://msiplab.eng.niigata-u.ac.jp/
%
arguments
    u (:,:,:) {mustBeFloat}
    options.Method {mustBeMember(options.Method,{'none','batch','iir'})} = 'iir'
    options.Rho (1,1) double {mustBeGreaterThanOrEqual(options.Rho,0),mustBeLessThan(options.Rho,1)} = 0.9
    options.Scope {mustBeMember(options.Scope,{'dc','full'})} = 'dc'
    options.Stride (1,2) double {mustBePositive,mustBeInteger} = [4 4]
end

nFrames = size(u,3);
if strcmp(options.Scope,'dc')
    baseOf = @(frame) blockMeanField(frame,options.Stride);
else
    baseOf = @(frame) frame;
end

b = zeros(size(u),'like',u);
switch options.Method
    case 'none'
        % b stays zero
    case 'batch'
        b = repmat(baseOf(mean(u,3)),[1 1 nFrames]);
    case 'iir'
        rho = cast(options.Rho,'like',u);
        s = baseOf(u(:,:,1));
        for t = 1:nFrames
            b(:,:,t) = s;
            s = rho*s + (1-rho)*baseOf(u(:,:,t));
        end
end
uf = u - b;
end

function m = blockMeanField(frame,stride)
% Mean of every stride(1) x stride(2) block, repeated over the block
[szy,szx] = size(frame);
blocks = reshape(frame,stride(1),szy/stride(1),stride(2),szx/stride(2));
m = mean(mean(blocks,1),3);
m = reshape(repmat(m,[stride(1) 1 stride(2) 1]),szy,szx);
end
