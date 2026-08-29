function [uP,vP] = fcn_divfreeproj(u,v)
%FCN_DIVFREEPROJ Discrete Leray projection onto the divergence-free subspace
%
%   [uP,vP] = FCN_DIVFREEPROJ(u,v) projects the two-dimensional velocity
%   field (u,v), given as ny x nx arrays on a uniform periodic grid, onto
%   its divergence-free part.  In the Fourier domain the projection reads
%
%       uhat <- uhat - k (k'*uhat)/|k|^2 ,
%
%   which is the discrete Leray projection.  It is linear, orthogonal and
%   idempotent, and FCN_SPECDIV of the result vanishes to machine
%   precision.  The zero wavenumber is left untouched, so the mean flow is
%   preserved, and so is the Nyquist mode, see FCN_SPECWAVENUMBERS.
%
%   The projection imposes incompressibility on an estimated base-point
%   field.  It applies to velocity observations; for a scalar observable
%   such as vorticity there is no corresponding constraint and the
%   projection is not used.
%
%   See also FCN_SPECDIV, FCN_SPECWAVENUMBERS, FCN_PHASEAVG.

arguments
    u (:,:) double
    v (:,:) double
end
assert(isequal(size(u),size(v)),'u and v must have the same size.')

[ny,nx] = size(u);
[KX,KY] = fcn_specwavenumbers(ny,nx);
K2 = KX.^2 + KY.^2;
K2(K2 == 0) = 1;      % modes with zero wavenumber are left unchanged

U = fft2(u);
V = fft2(v);

div = KX.*U + KY.*V;
uP = real(ifft2(U - KX.*div./K2));
vP = real(ifft2(V - KY.*div./K2));
end
