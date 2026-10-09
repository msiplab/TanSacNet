function w = salsun2d_pack_params(params,options)
%SALSUN2D_PACK_PARAMS Pack SA-LSUN parameters into a flat vector for HLS
%
%   w = salsun2d_pack_params(params) packs params (from
%   salsun2d_extract_params) into a single column vector in the layout of
%   salsun2d_hls_layout. It checks that the network has the configuration
%   that the HLS implementation is built for.
%
%   w = salsun2d_pack_params(params,RowMajorWeights=true) stores the
%   weight matrices W1, W2, Wo of the estimators row by row (transposed)
%   in the same places, as the fixed-point fully connected layers of the
%   band design (salsun2d_hls_band) read them.
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
    params (1,1) struct
    options.RowMajorWeights (1,1) logical = false
end
if options.RowMajorWeights
    order = @(W) W.';
else
    order = @(W) W;
end
L = salsun2d_hls_layout();

%% Check the configuration
check(isequal(params.Stride(:).',L.Stride),'Stride must be [%d %d].',L.Stride)
check(numel(params.Stages) == L.NStages,'%d intermediate stages are required.',L.NStages)
targetCode = @(t) 1 + strcmp(t,'Sum');
for iStage = 1:L.NStages
    s = params.Stages(iStage);
    check(isequal(s.Shift,L.Shift(iStage,:)) && targetCode(s.Target) == L.Target(iStage) && ...
        isequal(s.SynShift,L.SynShift(iStage,:)) && targetCode(s.SynTarget) == L.SynTarget(iStage), ...
        'Stage %d does not match the HLS configuration.',iStage)
    check(isUniform(s.Mus) && isUniform(s.SynMus), ...
        'Stage %d: Mus must be the same for all blocks.',iStage)
end
check(isUniform(params.V0.MusW) && isUniform(params.V0.MusU) && ...
    isUniform(params.V0t.MusW) && isUniform(params.V0t.MusU), ...
    'Initial/final rotation: Mus must be the same for all blocks.')
ests = [params.V0.Estimator, params.Stages.Estimator];
for iEst = 1:L.NEst
    e = ests(iEst);
    check(isequal(e.Channels(:).',L.EstChannelFirst(iEst):L.EstChannelLast(iEst)) && ...
        isequal(e.Neighbor(:).',L.Neighbor) && numel(e.ResBlocks) == L.NResBlocks && ...
        size(e.ResBlocks(1).W1,1) == L.EstNHidden(iEst) && ...
        size(e.Wo,1) == L.EstNAngles(iEst) && e.NumberOfZeroPadAngles == L.EstNZeroPad(iEst) && ...
        abs(double(e.Epsilon) - L.Epsilon) <= 1e-6*L.Epsilon, ...
        'Estimator %d does not match the HLS configuration.',iEst)
end

%% Pack
w = zeros(L.NParams,1,'single');
    function put(offset,value)
        w(offset + (1:numel(value))) = single(value(:));
    end
put(L.Cvh,params.Cvh)
put(L.Mask,params.Mask)
% Mus may be stored per block; all columns are equal (checked above)
put(L.V0MusW,params.V0.MusW(:,1))
put(L.V0MusU,params.V0.MusU(:,1))
put(L.V0tMusW,params.V0t.MusW(:,1))
put(L.V0tMusU,params.V0t.MusU(:,1))
for iStage = 1:L.NStages
    put(L.StageMus(iStage),params.Stages(iStage).Mus(:,1))
    put(L.StageSynMus(iStage),params.Stages(iStage).SynMus(:,1))
end
for iEst = 1:L.NEst
    e = ests(iEst);
    for iRes = 1:L.NResBlocks
        r = e.ResBlocks(iRes);
        put(L.Gamma(iEst,iRes),r.Gamma)
        put(L.Beta(iEst,iRes),r.Beta)
        put(L.W1(iEst,iRes),order(r.W1))
        put(L.B1(iEst,iRes),r.B1)
        put(L.W2(iEst,iRes),order(r.W2))
        put(L.B2(iEst,iRes),r.B2)
    end
    put(L.Wo(iEst),order(e.Wo))
    put(L.Bo(iEst),e.Bo)
end
end

function tf = isUniform(mus)
tf = all(mus == mus(:,1),'all');
end

function check(cond,varargin)
if ~cond
    error('salsun2d_pack_params:config',varargin{:})
end
end
