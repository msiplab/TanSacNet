%% https://github.com/dynamicslab/databook_matlab

if ~exist("databook_matlab-master","dir")
    disp("Downloading databook_matlab ... ")
    unzip("https://github.com/dynamicslab/databook_matlab/archive/refs/heads/master.zip")
else
    disp("databook_matlab-master exists.")
end

%databooksubfolders = { "CH07" };
%for subfolder = databooksubfolders
%    disp("Add path databook_matlab-master/"+subfolder)
%    addpath("databook_matlab-master/"+subfolder)
%end

%% http://databookuw.com/DATA.zip
ccd = pwd;
cd("../../../data")
if ~exist("databook_DATA","dir")
    disp("Downloading databook DATA ...")
    unzip("http://databookuw.com/DATA.zip","databook_DATA")
else
    disp("databook_DATA exists.")
end
cd(ccd)

cd("./databook_matlab-master")
if ~exist("DATA","dir")
    disp("Making DATA folder...")
    mkdir("DATA")
else
    disp("DATA exists.")
end
if ~exist("DATA/VORTALL.mat","file")
    disp("Copying VORTALL.mat ...")
    copyfile("../../../../data/databook_DATA/DATA/VORTALL.mat","DATA")
else
    disp("VORTALL.mat exists.")
end
cd(ccd)

%% https://github.com/baddoo/piDMD    
if ~exist("piDMD-main","dir")
    disp("Downloading piDMD ... ")
    unzip("https://github.com/baddoo/piDMD/archive/refs/heads/main.zip")
else
    disp("piDMD-main exists.")
end
 
pidmdsubfolders = { "src" };
for subfolder = pidmdsubfolders
    disp("Add path piDMD-main/"+subfolder)
    addpath("piDMD-main/"+subfolder)
end

%%