%% Create Data from Transport Equation
%% Preparation

clear
close all
%% PDE Definition and Condition Setting
% PDE(Transport equation)
% $$\frac{\partial u}{\partial t} + a\frac{\partial u} {\partial x} +b\frac{\partial 
% u}{\partial y}= 0$$

a = 0.5;                           % Transportation velocity
b = 0.5;

%% Spatial Mesh
% x
L_x = 3;                           % Maximum value in spatial direction
dx = 0.01;                         % Spatial displacement
N_x = floor(L_x/dx);               % Total number of spatial mesh
X_c = linspace(0,L_x,N_x);         % Coordinates (plot:N_x,range[0 L_x])
% y(same as x)
L_y = 3;
dy = 0.01;
N_y = floor(L_y/dy);
Y_c = linspace(0,L_y,N_y);

[X, Y] = meshgrid(X_c, Y_c);

%% Temporal Mesh
L_t = 2;                          % Maximum value in spatial direction
dt = 0.01;                         % Temporal displacement
N_t = floor(L_t/dt);               % Total number of temporal mesh
% Definition of Initial Conditions (Use Gaussian Wave Packets)

A = 10;          % Amplitude
sigma = 0.05;                           % Width of the initial wave packet

u0 =  A * exp(-((X - 1.5).^2 + (Y - 1.5).^2)/(2 .* sigma^2));
% Creating a Data from PDE Solutions

u = zeros(N_x,N_y,N_t);
u(:,:,1) = u0;                  

u_new = u0;

% Solving PDEs by finite difference method
for t = 1:N_t
    for i = 2:N_x-1
        for j = 2:N_y-1
            u_new(i,j) = u0(i,j) ...
                     - (a * dt / dx) * (u0(i,j) - u0(i-1,j)) ...
                     - (b * dt / dy) * (u0(i,j) - u0(i,j-1));
        end
    end 
% Dirichlet boundary condition
    u_new(1, :) = 0; u_new(end, :) = 0;
    u_new(:, 1) = 0; u_new(:, end) = 0;
    
% Update
    u0 = u_new;
    u(:,:,t) = u_new;
    

end

for n = 1:N_t
    surf(X, Y, u(:,:,n), 'EdgeColor', 'none');
    axis([0 L_x 0 L_y 0 10]);
    xlabel('x');
    ylabel('y');
    zlabel('u(x,y,t)');
    title(sprintf('Time: %.4f', n*dt));
    clim([0 1]);
    pause(0.001)
end

%% Save data
save('./Experience1_add/Dataset/data2.mat','u')