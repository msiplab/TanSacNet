%SALSUN2D_HLS_TB Testbench for salsun2d_hls (used by HDL Coder HLS workflow)
%
% The image size is set by hlsInputSize and the design by hlsDesignName
% in the base workspace (defaults [32 32] and 'salsun2d_hls');
% run_salsun2d_hls_codegen sets them.
%
% Requirements: MATLAB R2026b, Deep Learning Toolbox
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

if ~exist('hlsInputSize','var')
    hlsInputSize = [32 32];
end
if ~exist('hlsDesignName','var')
    hlsDesignName = 'salsun2d_hls';
end
rng(0)
nCoefs = 2;
coefMask = reshape([ones(nCoefs,1); zeros(16-nCoefs,1)],2,[]).';
net = salsun2d_create_test_network(hlsInputSize,coefMask(:));
params = salsun2d_extract_params(net);
w = salsun2d_pack_params(params);
for iFrame = 1:2
    x = rand(hlsInputSize,'single');
    if strcmp(hlsDesignName,'salsun2d_hls_band')
        % One frame band by band (host logic of the streaming kernel)
        L = salsun2d_hls_layout();
        if ~exist('hlsBandRows','var'), hlsBandRows = 31; end
        [~,~,~,~,stats] = salsun2d_infer(x,params);
        ests = [params.V0.Estimator, params.Stages.Estimator];
        mu = zeros(L.NDec,L.NEst,'single'); sigma = ones(L.NDec,L.NEst,'single');
        for k = 1:L.NEst
            ch = ests(k).Channels;
            mu(ch,k) = stats(k).Mu(1:numel(ch)); sigma(ch,k) = stats(k).Sigma(1:numel(ch));
        end
        y = salsun2d_band_frame(x,w,mu,sigma,hlsBandRows);
        yRef = salsun2d_infer(x,params,Statistics=stats);
    elseif strcmp(hlsDesignName,'salsun2d_hls_opt')
        L = salsun2d_hls_layout();
        thetaBuf = zeros([L.NThetaRows hlsInputSize./L.Stride],'single');
        y = salsun2d_hls_opt(x,w,thetaBuf);
    else
        y = salsun2d_hls(x,w);
    end
    if ~strcmp(hlsDesignName,'salsun2d_hls_band')
        yRef = salsun2d_infer(x,params);
    end
    fprintf('salsun2d_hls_tb: frame %d, max abs difference from reference = %g\n', ...
        iFrame,max(abs(y(:)-yRef(:))));
end
