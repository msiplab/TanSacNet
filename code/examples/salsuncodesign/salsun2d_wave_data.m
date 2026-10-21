function [u,t] = salsun2d_wave_data(options)
%SALSUN2D_WAVE_DATA 2-D wave equation data as in ../salsun/createdata_waveEq.m
%
%   [u,t] = salsun2d_wave_data() returns the frames u (300 x 300 x 150,
%   double) and their times t that createdata_waveEq.m produces with its
%   default settings (one epicenter at the center, amplitude 10, Dirichlet
%   boundary, frames 401 to 550 of the simulation), computed with the
%   same finite difference scheme but vectorized.
%
%   Like the original script, the second frame of the simulation is left
%   at zero by default, so the scheme effectively starts from
%   u(:,:,1) = u0 and u(:,:,2) = 0: a velocity impulse that also excites
%   the highest-frequency mode of the grid, which alternates in sign from
%   frame to frame. This is kept as the default so that the data match.
%
%   salsun2d_wave_data(SecondFrame='zero-velocity') instead starts from
%   zero initial velocity, u(:,:,2) = u0 + (r/2)*laplacian(u0), which is
%   the standard second-order start of the leapfrog scheme.
%
%   salsun2d_wave_data(Frames=[first last]) selects other frames of the
%   simulation (default [401 550]).
%
%   salsun2d_wave_data(Epicenters=P,Amplitudes=A) places Gaussian initial
%   displacements at the rows [x y] of P (in the domain [0,3] x [0,3],
%   default [1.5 1.5]) with the amplitudes A (default 10 for each), e.g.
%   for test data that differ from the training data in the source
%   positions; the frames then contain the superposition of the waves.
%
%   salsun2d_wave_data(FrameSize=[ny nx]) simulates a larger domain with
%   the same mesh spacing (0.01) and time step, ny x nx points ([0,
%   ny*0.01] x [0, nx*0.01], default [300 300]), e.g. [1080 1920] for
%   Full HD frames; the epicenters are then in that domain (default its
%   center). Only the frames returned are kept in memory.
%
% Requirements: MATLAB R2026b
%
% Copyright (c) 2026, Shogo MURAMATSU
%
% All rights reserved.
%
% Contact address: Shogo MURAMATSU,
%    Faculty of Engineering, Niigata University,
%    8050 2-no-cho Ikarashi, Nishi-ku,
%    Niigata, 950-2181, JAPAN
%
% http://msiplab.eng.niigata-u.ac.jp/
%
arguments
    options.Frames (1,2) double = [401 550]
    options.SecondFrame {mustBeMember(options.SecondFrame,{'zero','zero-velocity'})} = 'zero'
    options.Epicenters (:,2) double = [1.5 1.5]
    options.Amplitudes (:,1) double = []
    options.FrameSize (1,2) double {mustBeInteger,mustBePositive} = [300 300]
end

c = 0.6;                           % propagation velocity
d = 0.01;                          % spatial mesh (same in x and y)
Ny = options.FrameSize(1); Nx = options.FrameSize(2);
if isequal(options.FrameSize,[300 300])
    L = 3; N = floor(L/d);         % as createdata_waveEq.m: linspace(0,3,300)
    coordY = linspace(0,L,N); coordX = coordY;
else
    coordY = (0:Ny-1)*d; coordX = (0:Nx-1)*d;
end
Lt = 7; dt = 0.01; Nt = floor(Lt/dt);
[X,Y] = meshgrid(coordX,coordY);
epicenters = options.Epicenters;
if ~isequal(options.FrameSize,[300 300]) && isequal(epicenters,[1.5 1.5])
    epicenters = [coordX(end) coordY(end)]/2;   % center of the larger domain
end

sigma = 0.005;
A = options.Amplitudes;
if isempty(A)
    A = 10*ones(size(epicenters,1),1);
end
assert(numel(A) == size(epicenters,1),'One amplitude per epicenter is required.')
u0 = zeros(size(X));
for k = 1:size(epicenters,1)
    x0 = epicenters(k,1);
    y0 = epicenters(k,2);
    u0 = u0 + A(k)*exp(-((X-x0).^2 + (Y-y0).^2)/(2*sigma^2));
end

% Leapfrog scheme with three frames in memory; the frames asked for are kept
frames = options.Frames(1):options.Frames(2);
u = zeros(Ny,Nx,numel(frames));
r = (c*dt/d)^2;
iy = 2:Ny-1; ix = 2:Nx-1;
uPrev = u0;
uCur = zeros(Ny,Nx);
if strcmp(options.SecondFrame,'zero-velocity')
    lap0 = u0(iy+1,ix) + u0(iy-1,ix) + u0(iy,ix+1) + u0(iy,ix-1) - 4*u0(iy,ix);
    uCur(iy,ix) = u0(iy,ix) + (r/2)*lap0;
end
[tf,loc] = ismember(1,frames); if tf, u(:,:,loc) = uPrev; end
[tf,loc] = ismember(2,frames); if tf, u(:,:,loc) = uCur; end
for n = 2:Nt-1
    lap = uCur(iy+1,ix) + uCur(iy-1,ix) + uCur(iy,ix+1) + uCur(iy,ix-1) - 4*uCur(iy,ix);
    uNext = zeros(Ny,Nx);          % Dirichlet boundary (zero)
    uNext(iy,ix) = 2*uCur(iy,ix) - uPrev(iy,ix) + r*lap;
    [tf,loc] = ismember(n+1,frames);
    if tf
        u(:,:,loc) = uNext;
    end
    if n+1 >= frames(end)
        break
    end
    uPrev = uCur; uCur = uNext;
end

t = frames*dt;
end
