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
end

c = 0.6;                           % propagation velocity
L = 3;  d = 0.01;  N = floor(L/d); % spatial mesh (same in x and y)
coord = linspace(0,L,N);
Lt = 7; dt = 0.01; Nt = floor(Lt/dt);
[X,Y] = meshgrid(coord,coord);

A = 10;  sigma = 0.005;  x0 = 1.5;  y0 = 1.5;
u0 = A*exp(-((X-x0).^2 + (Y-y0).^2)/(2*sigma^2));

u = zeros(N,N,Nt);
u(:,:,1) = u0;
r = (c*dt/d)^2;
in = 2:N-1;
if strcmp(options.SecondFrame,'zero-velocity')
    lap0 = u0(in+1,in) + u0(in-1,in) + u0(in,in+1) + u0(in,in-1) - 4*u0(in,in);
    u(in,in,2) = u0(in,in) + (r/2)*lap0;
end
for n = 2:Nt-1
    un = u(:,:,n);
    lap = un(in+1,in) + un(in-1,in) + un(in,in+1) + un(in,in-1) - 4*un(in,in);
    u(in,in,n+1) = 2*un(in,in) - u(in,in,n-1) + r*lap;
    % Dirichlet boundary (already zero)
end

frames = options.Frames(1):options.Frames(2);
u = u(:,:,frames);
t = frames*dt;
end
