function [th,names] = fcn_ssgetthetas(net,X,names)
%FCN_SSGETTHETAS Angles of every rotation stage, estimated for input X
%
%   [th,names] = fcn_ssgetthetas(net,X) runs the estimator net (e.g. the
%   step-1 training net) on X (one sample, 'SSCB') and returns the output
%   of every layer whose name ends with 'Theta' (analysis side), each
%   nAnglesFull x nBlocks (the first nZeroPad rows are zero).
if nargin < 3
    allNames = {net.Layers.Name};
    names = allNames(endsWith(allNames,'Theta'));
end
th = cell(1,numel(names));
[th{:}] = predict(net,X,'Outputs',names);
for i = 1:numel(th)
    if isdlarray(th{i}), th{i} = extractdata(th{i}); end
    th{i} = reshape(th{i},size(th{i},1),[]);
end
end
