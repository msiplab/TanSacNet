function fpath = fcn_findcylinderdata(datadir)
%FCN_FINDCYLINDERDATA Locate the cylinder wake data file
%
%   fpath = FCN_FINDCYLINDERDATA(datadir) returns the full path of the
%   cylinder wake snapshot file below datadir, or "" when it is absent.
%
%   The databook archive expands to DATA/<name>.mat, but a manually
%   extracted copy may sit directly in datadir, so both layouts are
%   accepted.  CYLINDER_ALL.mat (velocity and vorticity) is preferred over
%   VORTALL.mat (vorticity only) when both are present.
%
%   See also SETUP, FCN_LOADCYLINDER.

arguments
    datadir (1,1) string
end

names = ["CYLINDER_ALL.mat","VORTALL.mat"];
subs  = ["","DATA"];

fpath = "";
for name = names
    for sub = subs
        cand = fullfile(datadir,sub,name);
        if exist(cand,'file') == 2
            fpath = string(cand);
            return
        end
    end
end
end
