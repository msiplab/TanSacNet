function [w,b,ln,T] = pack_est_df(params)
%PACK_EST_DF Parameters of the dataflow estimator (est_df.h layout)
%
%   [w,b,ln] = pack_est_df(params) returns the parameters of the five
%   estimators of params (layer normalization of the state at the input,
%   residual blocks without normalization and with tanh,
%   outputs mapped by asin(tanh)) in the memory layout of est_df.h: w, the
%   weight words (one row of 24 values per word, rows padded with zeros,
%   weights rounded to L.FixWeight); b, the biases rounded to the
%   accumulator format; ln, (gamma, beta) per feature of the input
%   normalization. [...,T] = pack_est_df(params) also returns the tables
%   of est_df.cpp, T.Tanh and T.Gd (and writes act_table.h next to this
%   file).
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
nEst = numel(ests); nRes = numel(ests(1).ResBlocks); Pw = 24;   % est_df.h PH
for k = 1:nEst
    e = ests(k);
    assert(isfield(e,'StdScope') && strcmp(e.StdScope,'layer') && isempty(e.ResBlocks(1).Gamma) ...
        && strcmp(e.Activation,'tanh') && strcmp(e.OutputMapping,'asintanh'), ...
        'est_df.cpp: input layer normalization, residual blocks with tanh and without normalization, asin(tanh) outputs')
end
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
for k = 1:nEst
    ln = [ln; double([ests(k).StdGamma(:) ests(k).StdBeta(:)])]; %#ok<AGROW>
end
% Tables at x = -8 + i/64: tanh rounded to [24 22], asin(tanh) to [24 21]
xs = -8 + (0:1024)/64;
T.Tanh = floor(tanh(xs)*2^22 + 0.5)/2^22;
T.Gd = floor(2*atan(tanh(xs/2))*2^21 + 0.5)/2^21;
fid = fopen(fullfile(fileparts(mfilename('fullpath')),'act_table.h'),'w');
fprintf(fid,'// act_table.h: tanh(x) in [24 22] and asin(tanh(x)) in [24 21] at\n');
fprintf(fid,'// x = -8 + i/64, i = 0..1024 (written by pack_est_df.m)\n#pragma once\n');
fprintf(fid,'static const estdf::ttab_t TANH_TABLE[estdf::ACT_N] = {\n');
fprintf(fid,'    %.17g,\n',T.Tanh);
fprintf(fid,'};\nstatic const estdf::gdtab_t GD_TABLE[estdf::ACT_N] = {\n');
fprintf(fid,'    %.17g,\n',T.Gd);
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
