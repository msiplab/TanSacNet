function export_est_df(outDir,options)
%EXPORT_EST_DF Test data for the dataflow estimator prototype (est_df.cpp)
%
%   export_est_df(outDir) writes, as little-endian doubles, the parameters
%   of the five estimators of a network (ParamsFile, by default
%   results/c_tanh_asin_800ep_seed0.mat) in the memory layout of est_df.h
%   (wmem.bin, bmem.bin, ln.bin), and for each estimator k the state of
%   NumGroups groups of 5 blocks of a wave frame times 2^6 (feat<k>.bin,
%   rounded to [18 11]) with the angles of the fixed-point model
%   (ref<k>.bin) and of the floating-point network (flt<k>.bin). It also
%   writes act_table.h, the tables of the prototype, and the fixed-point
%   model uses them as est_df.cpp does.
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
    outDir {mustBeTextScalar} = fullfile(fileparts(mfilename('fullpath')),'testdata')
    options.NumGroups (1,1) double {mustBeInteger,mustBePositive} = 4
    options.ParamsFile {mustBeTextScalar} = ''
end
here = fileparts(fileparts(mfilename('fullpath')));
addpath(here,fullfile(here,'..','..'),fullfile(here,'..','salsun'));
if ~isfolder(outDir), mkdir(outDir); end
if isempty(options.ParamsFile)
    options.ParamsFile = fullfile(here,'results','c_tanh_asin_800ep_seed0.mat');
end
R = load(options.ParamsFile,'params'); P = R.params;
L = salsun2d_hls_layout();
[pQ,~] = salsun2d_band_fixed_model(P);
ests = [P.V0.Estimator, P.Stages.Estimator];
estsQ = [pQ.V0.Estimator, pQ.Stages.Estimator];
nEst = numel(ests); NB = 5; scale = 2^6;   % est_df.h NB, LN_SCALE_BITS
rX = @(v) salsun2d_round_fixed(double(v),L.FixSignal(1),L.FixSignal(2));
rAcc = @(v) salsun2d_round_fixed(double(v),L.FixAcc(1),L.FixAcc(2));

[w,b,ln,T] = pack_est_df(P);
writeBin(fullfile(outDir,'wmem.bin'),w.');      % word by word, 24 values each
writeBin(fullfile(outDir,'bmem.bin'),b);
writeBin(fullfile(outDir,'ln.bin'),ln.');       % (gamma, beta) per feature
fprintf('wmem %d words, bmem %d, ln %d\n',size(w,1),numel(b),size(ln,1));

% The state of the estimators on a frame (the 'input' of salsun2d_infer),
% and their angles
u = single(salsun2d_wave_data()); x = u(:,:,12);
feat = cell(nEst,1);
salsun2d_infer(x,P,Quantizer=@collect,EstimatorTags=true);
nBlk = options.NumGroups*NB;
for k = 1:nEst
    F = rX(scale*feat{k}(:,1:nBlk));
    writeBin(fullfile(outDir,sprintf('feat%d.bin',k-1)),reshape(permute(reshape(F,size(F,1),NB,[]),[2 1 3]),[],1));
    thQ = forwardFixed(F,ests(k),estsQ(k),rX,rAcc,T,scale);
    thF = forwardFloat(double(feat{k}(:,1:nBlk)),ests(k));
    writeBin(fullfile(outDir,sprintf('ref%d.bin',k-1)),reshape(permute(reshape(thQ,size(thQ,1),NB,[]),[2 1 3]),[],1));
    writeBin(fullfile(outDir,sprintf('flt%d.bin',k-1)),reshape(permute(reshape(thF,size(thF,1),NB,[]),[2 1 3]),[],1));
end

    function v = collect(v,tag)
        if startsWith(tag,'input:')
            feat{sscanf(tag,'input:%d')} = double(v);
        end
    end
end

function th = forwardFixed(F,est,estQ,rX,rAcc,T,scale)
% Estimator as est_df.cpp computes it, from the scaled state F
s1 = sum(F,1); s2 = sum(F.^2,1); n = size(F,1);
m = single(s1)/single(n); v = max(single(s2)/single(n) - m.*m,0);
rs = single(1)./sqrt(v + single(double(est.Epsilon)*scale^2));
mF = salsun2d_round_fixed(double(m),24,16);
rsF = salsun2d_round_fixed(double(rs),27,16);
t = salsun2d_round_fixed((F - mF).*rsF,24,16);
g = salsun2d_round_fixed(double(est.StdGamma(:)),18,15);
be = salsun2d_round_fixed(double(est.StdBeta(:)),18,15);
F = rX(t.*g + be);
for r = 1:numel(estQ.ResBlocks)
    rb = estQ.ResBlocks(r);
    z1 = double(rb.W1)*F + rAcc(est.ResBlocks(r).B1(:));
    a = rX(tableValue(z1,T.Tanh));
    F = rX(F + double(rb.W2)*a + rAcc(est.ResBlocks(r).B2(:)));
end
z = double(estQ.Wo)*F + rAcc(est.Bo(:));
th = floor(tableValue(z,T.Gd)*2^27)/2^27;      % truncated to the accumulator format
end

function th = forwardFloat(F,est)
m = mean(F,1); v = mean((F-m).^2,1);
F = double(est.StdGamma(:)).*((F-m)./sqrt(v+double(est.Epsilon))) + double(est.StdBeta(:));
for r = 1:numel(est.ResBlocks)
    rb = est.ResBlocks(r);
    F = F + double(rb.W2)*tanh(double(rb.W1)*F + double(rb.B1(:))) + double(rb.B2(:));
end
th = asin(tanh(double(est.Wo)*F + double(est.Bo(:))));
end

function a = tableValue(z,T)
% Linear interpolation of a table at x = -8 + i/64 with a 16-bit fraction
% (the end values beyond -8 and 8), as est_df.cpp interpolates
u = (z + 8)*64;
i = floor(u);
fr = floor((u - i)*2^16)/2^16;
i = min(max(i,0),1023);
a = T(i+1) + (T(i+2) - T(i+1)).*fr;
a(z <= -8) = T(1);
a(z >= 8) = T(end);
end

function writeBin(file,v)
fid = fopen(file,'w','ieee-le'); fwrite(fid,double(v(:)),'double'); fclose(fid);
end
