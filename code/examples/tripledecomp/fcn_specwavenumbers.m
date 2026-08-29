function [KX,KY] = fcn_specwavenumbers(ny,nx)
%FCN_SPECWAVENUMBERS Wavenumber grids of the spectral derivative operator
%
%   [KX,KY] = FCN_SPECWAVENUMBERS(ny,nx) returns the ny x nx wavenumber
%   grids used by the spectral divergence and by the Leray projection.
%
%   On an even-length axis the Nyquist wavenumber is set to zero.  Without
%   that convention the symbol of the derivative is not odd-symmetric, the
%   projected spectrum is not Hermitian, and taking the real part of the
%   inverse transform destroys both the divergence-free property and the
%   idempotence of the projection.  Zeroing the Nyquist mode leaves it
%   untouched by the projection, which is the standard choice for spectral
%   differentiation on an even grid.
%
%   See also FCN_SPECDIV, FCN_DIVFREEPROJ.

arguments
    ny (1,1) double {mustBePositive,mustBeInteger}
    nx (1,1) double {mustBePositive,mustBeInteger}
end

kx = 2*pi*[0:floor(nx/2) -ceil(nx/2)+1:-1]/nx;
ky = 2*pi*[0:floor(ny/2) -ceil(ny/2)+1:-1]/ny;
if mod(nx,2) == 0
    kx(nx/2+1) = 0;
end
if mod(ny,2) == 0
    ky(ny/2+1) = 0;
end
[KX,KY] = meshgrid(kx,ky);
end
