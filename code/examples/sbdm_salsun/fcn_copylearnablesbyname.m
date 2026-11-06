function netDst = fcn_copylearnablesbyname(netDst, netSrc)
%FCN_COPYLEARNABLESBYNAME Copy learnables with the same layer/parameter name
%   Same as the local function copyLearnablesByName in
%   examples/salsun/main_salsun2d.mlx.
tblDst = netDst.Learnables;
tblSrc = netSrc.Learnables;
for i = 1:height(tblSrc)
    idx = find(strcmp(tblDst.Layer,tblSrc.Layer{i}) & ...
        strcmp(tblDst.Parameter,tblSrc.Parameter{i}));
    if ~isempty(idx)
        tblDst.Value{idx} = tblSrc.Value{i};
    end
end
netDst.Learnables = tblDst;
end
