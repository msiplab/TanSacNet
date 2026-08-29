function d = fcn_specdiv(u,v)
%FCN_SPECDIV Spectral divergence of a periodic two-dimensional vector field
%
%   d = FCN_SPECDIV(u,v) returns the divergence of the velocity field
%   (u,v), each ny x nx on a uniform periodic grid, computed with the
%   spectral derivative of FCN_SPECWAVENUMBERS.  This is the discrete
%   divergence operator that FCN_DIVFREEPROJ annihilates exactly.
%
%   See also FCN_DIVFREEPROJ, FCN_SPECWAVENUMBERS.

arguments
    u (:,:) double
    v (:,:) double
end
assert(isequal(size(u),size(v)),'u and v must have the same size.')

[ny,nx] = size(u);
[KX,KY] = fcn_specwavenumbers(ny,nx);
d = real(ifft2(1i*KX.*fft2(u) + 1i*KY.*fft2(v)));
end
