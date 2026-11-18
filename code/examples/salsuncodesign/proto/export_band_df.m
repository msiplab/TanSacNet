function export_band_df(outDir,options)
%EXPORT_BAND_DF Test data for the band kernel prototype (band_df.cpp)
%
%   export_band_df(outDir) writes, as little-endian doubles, a frame
%   (x.bin, FrameSize, column-major), the small parameters (prm.bin, the
%   layout of band_df.h), the estimator parameters (wmem.bin, bmem.bin,
%   ln.bin, pack_est_df) of a network (ParamsFile, by default
%   results/c_tanh_asin_800ep_seed0.mat), and the reconstruction of the
%   fixed-point model (yref.bin); stats.bin and sums.bin are zeros (the
%   estimators take no statistics of the frame). It also writes
%   sin_table.h for band_df.cpp.
%
%   export_band_df(...,AngleBits=n) rounds the angles of the model to n
%   bits (band_df.cpp built with QBITS=n; 18 by default).
%
%   export_band_df(...,AngleBits=n,OffsetBits=b), b < n, stores the angles
%   as offsets of b bits from centers (QOFS=b), the centers from another
%   frame (salsun2d_angle_centers). prm.bin ends with the 147 centers
%   (zeros for b >= n), also in centers.bin. The centers and offsets are
%   an idea of Shogo Muramatsu.
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
    options.AngleBits (1,1) double {mustBeInteger} = 18
    options.OffsetBits (1,1) double {mustBeInteger} = 18
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

% Frame: a corner of a wave frame, or a frame of a larger simulation
if all(options.FrameSize <= 210)
    u = single(salsun2d_wave_data());
    x = u(91:90+options.FrameSize(1),91:90+options.FrameSize(2),12);
    xc = u(91:90+options.FrameSize(1),91:90+options.FrameSize(2),11);   % for the centers
else
    u = single(salsun2d_wave_data(FrameSize=options.FrameSize,Frames=[401 402], ...
        Epicenters=[4 3; 10 6; 16 4; 7 8; 13 9; 2 9]*min(options.FrameSize)/1080));
    x = u(:,:,2);
    xc = u(:,:,1);
end
writeBin(fullfile(outDir,'x.bin'),x);

% Small parameters
prm = [P.Cvh(:); P.Mask(:); P.V0.MusW(:,1); P.V0.MusU(:,1); P.V0t.MusW(:,1); P.V0t.MusU(:,1)];
for s = 1:numel(P.Stages), prm = [prm; P.Stages(s).Mus(:,1)]; end %#ok<AGROW>
for s = 1:numel(P.Stages), prm = [prm; P.Stages(s).SynMus(:,1)]; end %#ok<AGROW>
assert(numel(prm) == 368)
if options.OffsetBits < options.AngleBits
    ctr = salsun2d_angle_centers(P,xc,AngleBits=options.AngleBits);
else
    ctr = zeros(147,1);
end
prm = [prm; ctr];
writeBin(fullfile(outDir,'prm.bin'),prm);
writeBin(fullfile(outDir,'centers.bin'),ctr);

% Estimator parameters
[w,b,ln] = pack_est_df(P);
writeBin(fullfile(outDir,'wmem.bin'),w.');
writeBin(fullfile(outDir,'bmem.bin'),b);
writeBin(fullfile(outDir,'ln.bin'),ln.');

% The fixed-point model (the estimators take no statistics of the frame;
% stats.bin and sums.bin are zeros, for the ports of the kernel)
writeBin(fullfile(outDir,'stats.bin'),zeros(2*L.NEst*16,1));
writeBin(fullfile(outDir,'sums.bin'),zeros(2*L.NEst*16,1));
[pQ,q] = salsun2d_band_fixed_model(P,AngleBits=options.AngleBits,AngleCenters=ctr,OffsetBits=options.OffsetBits);
yRef = salsun2d_infer(x,pQ,Quantizer=q,EstimatorTags=true);
writeBin(fullfile(outDir,'yref.bin'),yRef);
fprintf('MSE of the fixed-point model %.4g, of the floating-point network %.4g\n', ...
    mean((yRef-x).^2,'all'),mean((salsun2d_infer(x,P)-x).^2,'all'));

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
