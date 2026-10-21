function export_est_df(outDir,options)
%EXPORT_EST_DF Test data for the dataflow estimator prototype (est_df.cpp)
%
%   export_est_df(outDir) writes, as little-endian doubles, the parameters
%   of the five estimators of results/r3w1_800ep_seed0.mat in the memory
%   layout of est_df.h (wmem.bin, bmem.bin, ln.bin), and for each
%   estimator k the standardized features of NumGroups groups
%   of 5 blocks of a wave frame (feat<k>.bin, rounded to [18 11]) with the angles of
%   the fixed-point model (ref<k>.bin) and of the floating-point network
%   (flt<k>.bin). It also writes gelu_table.h, the GELU table of the
%   prototype, and the fixed-point model uses it as est_df.cpp does.
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
end
here = fileparts(fileparts(mfilename('fullpath')));
addpath(here,fullfile(here,'..','..'),fullfile(here,'..','salsun'));
if ~isfolder(outDir), mkdir(outDir); end
R = load(fullfile(here,'results','r3w1_800ep_seed0.mat'),'params'); P = R.params;
L = salsun2d_hls_layout();
[pQ,~] = salsun2d_band_fixed_model(P);
ests = [P.V0.Estimator, P.Stages.Estimator];
estsQ = [pQ.V0.Estimator, pQ.Stages.Estimator];
nEst = numel(ests); nRes = numel(ests(1).ResBlocks); NB = 5; Pw = 24;   % est_df.h NB, P
rX = @(v) salsun2d_round_fixed(double(v),L.FixSignal(1),L.FixSignal(2));
rAcc = @(v) salsun2d_round_fixed(double(v),L.FixAcc(1),L.FixAcc(2));

[w,b,ln,T] = pack_est_df(P);
gel = @(z) geluTable(z,T,rX);
writeBin(fullfile(outDir,'wmem.bin'),w.');      % word by word, P values each
writeBin(fullfile(outDir,'bmem.bin'),b);
writeBin(fullfile(outDir,'ln.bin'),ln.');       % (gamma, beta) per feature
fprintf('wmem %d words, bmem %d, ln %d\n',size(w,1),numel(b),size(ln,1));

% Features of the estimators on a frame, and their angles
u = single(salsun2d_wave_data()); x = u(:,:,12);
feat = collect_estimator_io(x,P);
nBlk = options.NumGroups*NB;
for k = 1:nEst
    F = rX(feat{k}.F(:,1:nBlk));
    writeBin(fullfile(outDir,sprintf('feat%d.bin',k-1)),reshape(permute(reshape(F,size(F,1),NB,[]),[2 1 3]),[],1));
    thQ = forward(F,estsQ(k),rX,rAcc,true,gel);
    thF = forward(double(feat{k}.F(:,1:nBlk)),ests(k),@(v) v,@(v) v,false,[]);
    writeBin(fullfile(outDir,sprintf('ref%d.bin',k-1)),reshape(permute(reshape(thQ,size(thQ,1),NB,[]),[2 1 3]),[],1));
    writeBin(fullfile(outDir,sprintf('flt%d.bin',k-1)),reshape(permute(reshape(thF,size(thF,1),NB,[]),[2 1 3]),[],1));
end
end

function th = forward(F,est,rX,rAcc,quantized,gel)
% Estimator as the MATLAB band design computes it (quantized) or in
% floating point
for r = 1:numel(est.ResBlocks)
    rb = est.ResBlocks(r);
    if quantized
        s1 = sum(F,1); s2 = sum(F.^2,1); n = size(F,1);
        m = single(s1)/single(n); v = max(single(s2)/single(n) - m.*m,0);
        rs = single(1)./sqrt(v + single(1e-5));
        ln = rX(single(rb.Gamma(:)).*((single(F) - m).*rs) + single(rb.Beta(:)));
        z1 = double(rb.W1)*ln + rAcc(rb.B1(:));
        a = gel(z1);
        F = rX(F + double(rb.W2)*a + rAcc(rb.B2(:)));
    else
        m = mean(F,1); v = mean((F-m).^2,1);
        ln = double(rb.Gamma(:)).*((F-m)./sqrt(v+1e-5)) + double(rb.Beta(:));
        z1 = double(rb.W1)*ln + double(rb.B1(:));
        a = 0.5*z1.*(1+tanh(sqrt(2/pi)*(z1+0.044715*z1.^3)));
        F = F + double(rb.W2)*a + double(rb.B2(:));
    end
end
th = double(est.Wo)*F + rAcc(est.Bo(:));
if ~quantized
    th = double(est.Wo)*F + double(est.Bo(:));
end
end

function y = geluTanh(x)
y = 0.5*x.*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x.^3)));
end

function a = geluTable(z,T,rX)
% GELU of the prototype: linear interpolation of the table T with a
% 16-bit fraction, rounded to the signal format; 0 below -8, z above 8
u = (z + 8)*64;
i = floor(u);
fr = floor((u - i)*2^16)/2^16;
i = min(max(i,0),1023);
a = T(i+1) + (T(i+2) - T(i+1)).*fr;
a(z <= -8) = 0;
a(z >= 8) = z(z >= 8);
a = rX(a);
end

function writeBin(file,v)
fid = fopen(file,'w','ieee-le'); fwrite(fid,double(v(:)),'double'); fclose(fid);
end
