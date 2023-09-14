function main_burgerseq_gen(filename,nu)
mu = 1;                 % advection coefficient (fixed at 1)
%nu = 0.05;              % Viscosity coefficient (corresponding to ν in the differential equation)
% Spatial Mesh
L_x = 10;               % Maximum value in space direction
dx = 0.1;
N_x = floor(L_x/dx);    % Total number of meshes in spatial direction
X = linspace(0,L_x,N_x);% Coordinates
% Temporal Mesh
L_t = 10;               % Maximum value in time direction
dt = 0.1;
N_t = floor(L_t/dt);    % Total number of meshes in time direction
T = linspace(0,L_t,N_t);% Coordinates
% Wave number discretization
k = 2*pi*fftfreq(N_x, dx);
% initial condition
u0 = exp(-(X-3).^2/2);
%u0 = np.sin(2*np.pi*X/L_x)
%ndim = 100;
%Data preparation
%PDE resolution (ODE system resolution)
opt = odeset('MaxStep',5000);
[~,DataT] = ode45(@(t,u) burg_system(u,t,k,mu,nu),T,u0,opt);

save(filename+"nu_"+replace(num2str(nu),'.','_'),"T","X","DataT","dt","dx","nu")

end

function f=fftfreq(npts,dt,alias_dt)
% returns a vector of the frequencies corresponding to the length
% of the signal and the time step.
% specifying alias_dt > dt returns the frequencies that would
% result from subsampling the raw signal at alias_dt rather than
% dt.
 
 
 if (nargin < 3)
 alias_dt = dt;
 end
 fmin = -1/(2*dt);
 df = 1/(npts*dt);
 f0 = -fmin;
 alias_fmin = -1/(2*alias_dt);
 f0a = -alias_fmin;
 
 ff = mod(linspace(0, 2*f0-df, npts)+f0, 2*f0) - f0;
 fa = mod( ff+f0a, 2*f0a) - f0a;
 % return the aliased frequencies
 f = fa;
end

%Definition of ODE system (PDE ---(FFT)---> ODE system)
function u_t_real = burg_system(u, t, k,mu,nu)
 %Spatial derivative in the Fourier domain
 u_hat = fft(u);
 u_hat_x = 1j*k(:).*u_hat;
 u_hat_xx = -(k(:).^2).*u_hat;
 
 %Switching in the spatial domain
 u_x = ifft(u_hat_x);
 u_xx = ifft(u_hat_xx);
 
 %ODE resolution
 u_t = -mu*u.*u_x + nu*u_xx;
 u_t_real = real(u_t);
end