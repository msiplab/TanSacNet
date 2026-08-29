function fpath = fcn_exportvalues(expname,config,metrics,resultsdir)
%FCN_EXPORTVALUES Write experiment numbers to results/values.json
%
%   fpath = FCN_EXPORTVALUES(expname,config,metrics) writes the struct
%
%       {"exp": ..., "date": ..., "config": {...}, "metrics": {...}}
%
%   to results/values_<expname>.json and refreshes the merged
%   results/values.json that the manuscript repository consumes.  This is
%   the only sanctioned interface between the experiments and the paper:
%   numbers reach the text through macros generated from this file, never
%   by being typed into the manuscript.
%
%   metrics must contain scalars only; every key becomes one macro on the
%   manuscript side, so keys are camelCase and stable.
%
%   See also MAIN_BASEFIELD_COMPARE, MAIN_COEFFDYN_UNITCIRCLE.

arguments
    expname (1,1) string
    config struct
    metrics struct
    resultsdir (1,1) string = fullfile(fileparts(mfilename('fullpath')),'results')
end

if ~exist(resultsdir,'dir')
    mkdir(resultsdir)
end

record = struct( ...
    'exp',expname, ...
    'date',string(datetime('now','Format','yyyy-MM-dd')), ...
    'config',config, ...
    'metrics',metrics);

fpath = fullfile(resultsdir,"values_" + expname + ".json");
fid = fopen(fpath,'w');
assert(fid > 0,'Cannot write %s.',fpath)
cleaner = onCleanup(@() fclose(fid));
fprintf(fid,'%s\n',jsonencode(record,'PrettyPrint',true));
clear cleaner

%% Merge every per-experiment file into results/values.json
listing = dir(fullfile(resultsdir,'values_*.json'));
merged = struct();
for k = 1:numel(listing)
    txt = fileread(fullfile(listing(k).folder,listing(k).name));
    one = jsondecode(txt);
    merged.(matlab.lang.makeValidName(one.exp)) = one;
end
mergedpath = fullfile(resultsdir,'values.json');
fid = fopen(mergedpath,'w');
assert(fid > 0,'Cannot write %s.',mergedpath)
cleaner = onCleanup(@() fclose(fid));
fprintf(fid,'%s\n',jsonencode(merged,'PrettyPrint',true));
clear cleaner

fprintf('Wrote %s and %s\n',fpath,mergedpath);
end
