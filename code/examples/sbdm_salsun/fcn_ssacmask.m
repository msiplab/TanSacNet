function acMask = fcn_ssacmask(nChs,K)
%FCN_SSACMASK AC mask of the zonal coding in examples/salsun/main_salsun2d.mlx
%
%   acMask = fcn_ssacmask(nChs,K) keeps DC + K AC channels. It uses the
%   same rule as the example with nCoefs = K + 1 (the example's nCoefs
%   counts the DC channel), and returns the mask of the nChs-1 AC
%   channels (column vector, as given to maskLayer).
%
%   nChs = 16: K = 2 keeps channels {1,2,9}, K = 4 keeps {1,2,3,9,10}.
%   Channels 1..8 are the symmetric half, 9..16 the antisymmetric half.

nCoefs = K + 1;
coefMask = reshape([ones(nCoefs,1); zeros(nChs-nCoefs,1)],2,[]).';
coefMask = coefMask(:);
assert(coefMask(1) == 1, 'The DC channel must be kept.')
acMask = single(coefMask(2:end));
end
