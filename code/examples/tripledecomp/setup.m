%SETUP Path and data setup for the triple decomposition x LSUN experiments
%
% Run this once per MATLAB session before any main_* script:
%
%   cd code/examples/tripledecomp
%   setup
%
% It performs three things:
%   1. adds the TanSacNet package to the path (via code/setpath.m),
%   2. adds ../pidmd and ../tddmd so that the existing helpers can be reused,
%   3. makes sure the cylinder wake data and the piDMD reference
%      implementation are available locally.
%
% The data and the downloaded third-party code live under the repository
% level data/ directory, which is git-ignored; nothing downloaded here is
% committed.
%
% See also FCN_PHASEAVG, FCN_DIVFREEPROJ, MAIN_BASEFIELD_COMPARE,
% MAIN_COEFFDYN_UNITCIRCLE.

thisdir = fileparts(mfilename('fullpath'));
if isempty(thisdir)
    thisdir = pwd;
end

%% TanSacNet package
ccd = pwd;
cd(fullfile(thisdir,'..','..'))   % code/
setpath
cd(ccd)

%% Sibling examples (fcn_pidmdvialsun, fcn_cylinderplot, maskLayer, ...)
addpath(fullfile(thisdir,'..','pidmd'))
addpath(fullfile(thisdir,'..','tddmd'))
addpath(thisdir)

%% Results directory
if ~exist(fullfile(thisdir,'results'),'dir')
    mkdir(fullfile(thisdir,'results'))
end
if ~exist(fullfile(thisdir,'results','figures'),'dir')
    mkdir(fullfile(thisdir,'results','figures'))
end

%% piDMD reference implementation (Baddoo et al.)
% https://github.com/baddoo/piDMD
pidmddir = fullfile(thisdir,'..','pidmd','piDMD-main');
if ~exist(pidmddir,'dir')
    disp("Downloading piDMD ...")
    unzip("https://github.com/baddoo/piDMD/archive/refs/heads/main.zip", ...
        fullfile(thisdir,'..','pidmd'))
end
addpath(fullfile(pidmddir,'src'))

%% Cylinder wake data
% http://databookuw.com/DATA.zip (Kutz et al., Data-Driven Science and
% Engineering).  Only CYLINDER_ALL.mat is needed; it holds the velocity
% components UALL, VALL and the vorticity VORTALL of the flow past a
% circular cylinder at Re = 100, sampled over five shedding periods.
datadir = fullfile(thisdir,'..','..','..','data','databook_DATA');
if ~exist(datadir,'dir')
    mkdir(datadir)
end
% The archive expands to DATA/CYLINDER_ALL.mat, but an already unpacked copy
% may sit directly in datadir; fcn_cylinderdata accepts either layout.
if isempty(fcn_findcylinderdata(datadir))
    disp("Downloading databook DATA (about 174 MB) ...")
    unzip("http://databookuw.com/DATA.zip",datadir)
end
assert(~isempty(fcn_findcylinderdata(datadir)), ...
    'CYLINDER_ALL.mat not found under %s.',datadir)

clear ccd pidmddir
disp("Setup done. Data directory: " + datadir)
