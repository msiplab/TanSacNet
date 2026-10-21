function [w,b,ln,T] = pack_est_df(params)
%PACK_EST_DF Parameters of the dataflow estimator (est_df.h layout)
%
%   [w,b,ln] = pack_est_df(params) returns the parameters of the five
%   estimators of params (3 residual blocks of width 1) in the memory
%   layout of est_df.h: w, the weight words (one row of P = 24 values per
%   word, rows padded with zeros, weights rounded to L.FixWeight); b, the
%   biases rounded to the accumulator format; ln, (gamma, beta) per
%   feature. [...,T] = pack_est_df(params) also returns the GELU table of
%   est_df.cpp (and writes gelu_table.h next to this file).
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
L = salsun2d_hls_layout();
[pQ,~] = salsun2d_band_fixed_model(params);
ests = [params.V0.Estimator, params.Stages.Estimator];
estsQ = [pQ.V0.Estimator, pQ.Stages.Estimator];
nEst = numel(ests); nRes = numel(ests(1).ResBlocks); Pw = 24;   % est_df.h P
rAcc = @(v) salsun2d_round_fixed(double(v),L.FixAcc(1),L.FixAcc(2));
w = []; b = []; ln = [];
for r = 1:nRes
    for k = 1:nEst, w = [w; padRows(estsQ(k).ResBlocks(r).W1,Pw)]; end %#ok<AGROW>
    for k = 1:nEst, w = [w; padRows(estsQ(k).ResBlocks(r).W2,Pw)]; end %#ok<AGROW>
    for k = 1:nEst, b = [b; rAcc(ests(k).ResBlocks(r).B1(:))]; end %#ok<AGROW>
    for k = 1:nEst, b = [b; rAcc(ests(k).ResBlocks(r).B2(:))]; end %#ok<AGROW>
end
for k = 1:nEst, w = [w; padRows(estsQ(k).Wo,Pw)]; end
for k = 1:nEst, b = [b; rAcc(ests(k).Bo(:))]; end
for r = 1:nRes
    for k = 1:nEst
        ln = [ln; double([ests(k).ResBlocks(r).Gamma(:) ests(k).ResBlocks(r).Beta(:)])]; %#ok<AGROW>
    end
end
% GELU table: GELU(x) at x = -8 + i/64, rounded to [24 19]
xs = -8 + (0:1024)/64;
T = floor(0.5*xs.*(1 + tanh(sqrt(2/pi)*(xs + 0.044715*xs.^3)))*2^19 + 0.5)/2^19;
fid = fopen(fullfile(fileparts(mfilename('fullpath')),'gelu_table.h'),'w');
fprintf(fid,'// gelu_table.h: GELU(x) at x = -8 + i/64, i = 0..1024, in [24 19]\n');
fprintf(fid,'// (written by pack_est_df.m)\n#pragma once\n');
fprintf(fid,'static const estdf::gtab_t GELU_TABLE[estdf::GELU_N] = {\n');
fprintf(fid,'    %.17g,\n',T);
fprintf(fid,'};\n');
fclose(fid);
end

function W = padRows(W,Pw)
% rows of W, each padded with zeros to a multiple of Pw, as words of Pw
W = double(W);
n = size(W,2); np = Pw*ceil(n/Pw);
W = [W zeros(size(W,1),np-n)];
W = reshape(W.',Pw,[]).';
end
