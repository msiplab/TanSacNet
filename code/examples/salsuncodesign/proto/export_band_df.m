function export_band_df(outDir,options)
%EXPORT_BAND_DF Test data for the band kernel prototype (band_df.cpp)
%
%   export_band_df(outDir) writes, as little-endian doubles, a frame
%   (x.bin, FrameSize, column-major), the small parameters (prm.bin, the
%   layout of band_df.h), the estimator parameters (wmem.bin, bmem.bin,
%   ln.bin, pack_est_df), the statistics of the estimator inputs measured
%   on the frame itself (stats.bin: mu then sigma, NDec x NEst each), and
%   the reconstruction of the fixed-point model with those statistics
%   (yref.bin) with the channel sums of the estimator inputs
%   (sums.bin: sums then sums of squares, NDec x NEst each). It also writes
%   sin_table.h for band_df.cpp.
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
    outDir {mustBeTextScalar} = fullfile(fileparts(mfilename('fullpath')),'testdata_band')
    options.FrameSize (1,2) double = [40 40]
end
here = fileparts(fileparts(mfilename('fullpath')));
addpath(here,fullfile(here,'..','..'),fullfile(here,'..','salsun'));
if ~isfolder(outDir), mkdir(outDir); end
R = load(fullfile(here,'results','r3w1_800ep_seed0.mat'),'params'); P = R.params;
L = salsun2d_hls_layout();

% Frame: a corner of a wave frame
u = single(salsun2d_wave_data());
x = u(91:90+options.FrameSize(1),91:90+options.FrameSize(2),12);
writeBin(fullfile(outDir,'x.bin'),x);

% Small parameters
prm = [P.Cvh(:); P.Mask(:); P.V0.MusW(:,1); P.V0.MusU(:,1); P.V0t.MusW(:,1); P.V0t.MusU(:,1)];
for s = 1:numel(P.Stages), prm = [prm; P.Stages(s).Mus(:,1)]; end %#ok<AGROW>
for s = 1:numel(P.Stages), prm = [prm; P.Stages(s).SynMus(:,1)]; end %#ok<AGROW>
assert(numel(prm) == 368)
writeBin(fullfile(outDir,'prm.bin'),prm);

% Estimator parameters
[w,b,ln] = pack_est_df(P);
writeBin(fullfile(outDir,'wmem.bin'),w.');
writeBin(fullfile(outDir,'bmem.bin'),b);
writeBin(fullfile(outDir,'ln.bin'),ln.');

% Statistics of the frame itself, and the fixed-point model with them
[~,~,~,mu,sg] = salsun2d_infer_fast(x,P,[],[]);
writeBin(fullfile(outDir,'stats.bin'),[mu(:); sg(:)]);
[pQ,q] = salsun2d_band_fixed_model(P);
ests = [P.V0.Estimator, P.Stages.Estimator];
st = struct('Mu',cell(L.NEst,1),'Sigma',cell(L.NEst,1));
for k = 1:L.NEst
    ch = ests(k).Channels; nn = prod(ests(k).Neighbor);
    st(k).Mu = repmat(mu(ch,k),nn,1); st(k).Sigma = repmat(sg(ch,k),nn,1);
end
yRef = salsun2d_infer(x,pQ,Statistics=st,Quantizer=q,EstimatorTags=true);
writeBin(fullfile(outDir,'yref.bin'),yRef);
[~,s1,s2] = salsun2d_infer_fast(x,P,mu,sg);
writeBin(fullfile(outDir,'sums.bin'),[s1(:); s2(:)]);

% sin(pi i/1024), i = 0..2048, rounded to [20 18]
T = floor(sin(pi*(0:2048)/1024)*2^18 + 0.5)/2^18;
fid = fopen(fullfile(fileparts(mfilename('fullpath')),'sin_table.h'),'w');
fprintf(fid,'// sin_table.h: sin(pi i/1024), i = 0..2048, in [20 18] (written by export_band_df.m)\n');
fprintf(fid,'#pragma once\nstatic const bdf::stab_t SIN_TABLE[bdf::SIN_N] = {\n');
fprintf(fid,'    %.17g,\n',T);
fprintf(fid,'};\n');
fclose(fid);
fprintf('frame %dx%d written to %s\n',size(x),outDir);
end

function writeBin(file,v)
fid = fopen(file,'w','ieee-le'); fwrite(fid,double(v(:)),'double'); fclose(fid);
end
