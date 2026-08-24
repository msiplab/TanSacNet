clc, clear
%% Loading Input Signal

Datasetfolder ='./Experience1_add/Dataset';
files = dir(fullfile(Datasetfolder,'*.mat'));
data = load(fullfile(Datasetfolder,files(2).name));
u = single(data.u);

if canUseGPU
    disp("Use GPU")
    u = gpuArray(u);
end
[szy,szx,szt] = size(u);
%% パラメータ設定

% rblk = imd';
% nblocks = 9;
% blk = im2col(rblk,[14 14] ,'distinct')
nof = 1;
ky = 2*nof+1; % overlap の程度, 奇数
kx = 2*nof+1;
disp([ky kx])
blksz = [4 4]; % ブロックサイズ
% 係数の数 K (どれだけ特徴量を残すか),これはブロックサイズあたりに用いる係数の数で,ブロックサイズが[4 4]ならmax16
nCoefs = 1;
%%
% % Create Sub-images
% s_img = createsubimg(blk);
lbpcaimg = zeros(szy,szx,szt);
for iBlkCol = 1:szx/blksz(2)
    for iBlkRow = 1:szy/blksz(1)
        % ky x kx blocks の抽出
        subblks = fcn_extract_blks_(u,[iBlkRow,iBlkCol],blksz,[ky,kx]);
        % Reshape
        subblks_split = reshape(subblks, [blksz(1), ky, blksz(2), kx, szt]);
        subblks_perm  = permute(subblks_split, [1, 3, 2, 4, 5]);
        colblks       = reshape(subblks_perm, [prod(blksz), ky*kx*szt]); % [16 x (ky*kx*szt)]
        % PCA
        %Vpca = pca(colblks.')
        mu = mean(colblks,2);
        colblkszm = colblks - mu;
        C = cov(colblkszm.');
        [~,S,V] = svd(C,"econ");
        [~,idxS] = sort(diag(S),"descend");
        V = V(:,idxS(1:nCoefs));
        %norm(Vpca - Vsvd,'fro')
        % Approxiamtion
        targetblk = fcn_extract_blks_(u,[iBlkRow,iBlkCol],blksz,[1 1]);
        targetblk_mat = reshape(targetblk, [prod(blksz), szt]);

        targetblk_recons = V * (V' * (targetblk_mat - mu)) + mu; % [16 x szt]
        targetblk = reshape(targetblk_recons, [blksz(1), blksz(2), szt]);

        % Place block
        lbpcaimg = fcn_place_blks_(lbpcaimg,targetblk,[iBlkRow,iBlkCol],blksz);
    end
end
%%
mse = gather(mean((u - lbpcaimg).^2,'all'))
mae = gather(mean(abs(u - lbpcaimg),'all'))
pass = "./Experience1_add/Result2/LBPCA/K_1/";
%% Save WorkSpace
writematrix(mse,pass + "mse.txt");
writematrix(mae,pass + "mae.txt");

save(pass + "u_reconstruct.mat","lbpcaimg");
%% local functions の定義
% グローバル配列からローカル（局所）パッチブロックを抽出する関数

function y = fcn_extract_blks_(x,iBlk,blksz,k)
% 配列 x の拡張
ky = k(1);
kx = k(2);
iBlkRow = iBlk(1);
iBlkCol = iBlk(2);
padsz = [(ky-1)/2, (kx-1)/2].*blksz;
xx = padarray(x,[padsz,0],"circular"); % 配列の境界の要素を繰り返すことでパディング
%
posy = (iBlkRow-1)*blksz(1)+1;
posx = (iBlkCol-1)*blksz(2)+1;
y = xx(posy:posy+ky*blksz(1)-1,posx:posx+kx*blksz(2)-1,:);
end
%% 
% ローカルパッチブロックをグローバル配列に配置する関数

function y = fcn_place_blks_(y,blk,iBlk,blksz)
% Extend array x
iBlkRow = iBlk(1);
iBlkCol = iBlk(2);
posy = (iBlkRow-1)*blksz(1)+1;
posx = (iBlkCol-1)*blksz(2)+1;
y(posy:posy+blksz(1)-1,posx:posx+blksz(2)-1,:) = blk;
end