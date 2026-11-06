function cfg = fcn_ssconfig()
%FCN_SSCONFIG Settings of the self-supervised SA-LSUN experiment
%
%   Sensei's decision: both steps use ONLY the observation v.

cfg.imgIdx   = [9 15 19 23];      % Kodak test images (grayscale)
cfg.cropSize = [256 256];         % centre crop
cfg.sigma    = single(25/255);    % AWGN, known
cfg.stride   = [4 4];             % 4x4 blocks
cfg.ovlp     = [3 3];             % overlapping factor
cfg.K        = [2 4];             % kept AC channels in step 1 (DC always kept)

% Step 1: zonal-coding autoencoder (Adam, cosine decay lr0 -> lr1)
cfg.nIt1Sa   = 500;  cfg.lr1Sa = [1e-3 1e-5];   % estimator weights (lr as main_salsun2d.mlx)
cfg.nIt1N    = 500;  cfg.lr1N  = [1e-2 1e-4];   % shared angles

% Step 2: SURE on v, transform frozen, 30 parameters (kappa_p, g_p)
cfg.nIt2     = 400;  cfg.lr2   = 5e-3;          % as fcn_trainlsunsure defaults
cfg.epsFd    = 1e-3;                            % MC divergence step
cfg.nProbeEval = 16;                            % MC probes for the reported SURE

cfg.seedBase = 0;                               % noise seed = seedBase + image index
end
