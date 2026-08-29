function [Y,gridSize] = fcn_loadcylinder(datadir)
%FCN_LOADCYLINDER Load the cylinder wake vorticity snapshots
%
%   [Y,gridSize] = FCN_LOADCYLINDER(datadir) returns the vorticity field of
%   the flow past a circular cylinder at Re = 100 as a snapshot matrix
%   Y (space x time), together with the spatial grid size [ny nx] needed to
%   fold a column of Y back into an image.
%
%   The data are the 151 snapshots covering five shedding periods on a
%   199 x 449 grid distributed with Kutz et al., Data-Driven Science and
%   Engineering (http://databookuw.com/DATA.zip).  Run SETUP once to fetch
%   them.
%
%   See also SETUP, FCN_FINDCYLINDERDATA.

arguments
    datadir (1,1) string = fullfile(fileparts(mfilename('fullpath')), ...
        '..','..','..','data','databook_DATA')
end

fpath = fcn_findcylinderdata(datadir);
assert(fpath ~= "", ...
    'Cylinder data not found under %s. Run setup first.',datadir)

S = load(fpath);
if isfield(S,'VORTALL')
    Y = double(S.VORTALL);
else
    error('tripledecomp:noVorticity', ...
        'No VORTALL variable in %s.',fpath)
end

% The databook snapshots are stored as (ny*nx) x nT with ny = 199, nx = 449.
gridSize = [199 449];
assert(size(Y,1) == prod(gridSize), ...
    'Unexpected snapshot length %d (expected %d).',size(Y,1),prod(gridSize))
end
