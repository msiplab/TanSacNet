function net = fcn_dcsetshrink(net,ab)
%FCN_DCSETSHRINK Load shrinkage parameters ab into the layers Lv1_Shrink(Dc)
%
%   net = fcn_dcsetshrink(net,ab) sets A = ab.a, B = ab.b of 'Lv1_Shrink'
%   and, if ab has a0, A = ab.a0, B = ab.b0 of 'Lv1_ShrinkDc'. The values
%   get the class and device of the values already in the net.
L = net.Learnables;
L = setone(L,'Lv1_Shrink','A',ab.a);
L = setone(L,'Lv1_Shrink','B',ab.b);
if isfield(ab,'a0')
    L = setone(L,'Lv1_ShrinkDc','A',ab.a0);
    L = setone(L,'Lv1_ShrinkDc','B',ab.b0);
end
net.Learnables = L;
end

function L = setone(L,layer,prm,val)
idx = strcmp(L.Layer,layer) & strcmp(L.Parameter,prm);
assert(nnz(idx) == 1,'Layer %s / %s not found.',layer,prm)
old = extractdata(L.Value{idx});
if isdlarray(val), val = extractdata(val); end
L.Value{idx} = dlarray(cast(reshape(gather(val),size(old)),'like',old));
end
