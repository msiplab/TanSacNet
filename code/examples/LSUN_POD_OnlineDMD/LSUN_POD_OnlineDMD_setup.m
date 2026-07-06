%
isCodegen = false;

% SaivDr パッケージバージョン
LSUN_POD_OnlineDMD_VER = "v1.0";
LSUN_POD_OnlineDMD_DIR = "LSUN_POD_OnlineDMD"+SAIVDR_VER;
if ~exist(LSUN_POD_OnlineDMD_DIR,"dir")
    unzip("https://github.com/msiplab/LSUN_POD_OnlineDMD/archive/refs/tags/"+LSUN_POD_OnlineDMD_VER+".zip")
else
    disp(LSUN_POD_OnlineDMD_DIR+" exists.")
end
