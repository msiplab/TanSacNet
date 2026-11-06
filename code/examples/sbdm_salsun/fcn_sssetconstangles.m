function net = fcn_sssetconstangles(net,names,th,A,B)
%FCN_SSSETCONSTANGLES Load estimated angles (and optionally the shrinkage)
%
%   net = fcn_sssetconstangles(net,names,th) sets the 'Angles' learnable of
%   each constant-angle layer names{i} of a net from fcn_ssbuildconstsalsun
%   to th{i} (from fcn_ssgetthetas), dropping the zero-padded first rows.
%   net = fcn_sssetconstangles(net,names,th,A,B) also sets Lv1_Shrink A, B.
L = net.Learnables;
for i = 1:numel(names)
    idx = strcmp(L.Layer,names{i}) & strcmp(L.Parameter,'Angles');
    assert(nnz(idx) == 1,'Layer %s not found.',names{i})
    nA = size(L.Value{idx},1);
    L.Value{idx} = dlarray(th{i}(end-nA+1:end,:));
end
if nargin > 3
    iA = strcmp(L.Layer,'Lv1_Shrink') & strcmp(L.Parameter,'A');
    iB = strcmp(L.Layer,'Lv1_Shrink') & strcmp(L.Parameter,'B');
    L.Value{iA} = dlarray(single(gather(extractdata(A))));
    L.Value{iB} = dlarray(single(gather(extractdata(B))));
    if isa(th{1},'gpuArray')
        L.Value{iA} = gpuArray(L.Value{iA}); L.Value{iB} = gpuArray(L.Value{iB});
    end
end
net.Learnables = L;
end
