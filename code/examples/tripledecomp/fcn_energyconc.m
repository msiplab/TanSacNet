function [ec,ecBlocks,dcNorm] = fcn_energyconc(C,coefMask)
%FCN_ENERGYCONC Block-wise energy concentration of LSUN coefficients
%
%   [ec,ecBlocks,dcNorm] = FCN_ENERGYCONC(C,coefMask) measures how much of
%   the coefficient energy the retained channels capture.  C is the
%   coefficient array (nBlocksY x nBlocksX x M x nT) produced by
%   FCN_LSUNCOEFS2D and coefMask is the M x 1 truncation mask.
%
%   ec        mean over blocks and frames of the retained energy ratio
%   ecBlocks  the ratio per block, averaged over frames (nBlocksY x nBlocksX)
%   dcNorm    energy fraction carried by the DC channel, that is the
%             projection of the residual onto the constant direction
%
%   Blocks whose total energy is negligible are excluded from the average,
%   since the ratio is undefined there; the padded margin of the cylinder
%   snapshots is such a region.
%
%   See also FCN_LSUNCOEFS2D, FCN_LSUNTRAIN2D.

arguments
    C (:,:,:,:) double
    coefMask (:,1) double
end

M = size(C,3);
assert(numel(coefMask) == M, ...
    'coefMask has %d entries but the coefficients have %d channels.', ...
    numel(coefMask),M)

E = sum(C.^2,3);                                  % total energy per block
Ekept = sum(C.*reshape(coefMask,1,1,[]).*C,3);    % retained energy
Edc = C(:,:,1,:).^2;                              % DC channel energy

Etot = sum(E,4);
valid = Etot > eps(max(Etot(:)))*numel(Etot);

ratio = sum(Ekept,4)./max(Etot,realmin);
ecBlocks = ratio;
ecBlocks(~valid) = NaN;
ec = mean(ratio(valid));

dcNorm = sum(Edc(:))/max(sum(E(:)),realmin);
end
