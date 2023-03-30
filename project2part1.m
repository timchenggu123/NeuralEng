% Define FitzHugh-Nagumo model parameters
alpha = 3;
a = 0.7;
b = 0.8;
w2 = 1;
x0 = 0;
y0 = 0;
x10 = 0;
y10 = 0;
xy0=[x0; y0; x10; y10];

%Define coupling parameters
k1 = 0.1;
k2 = 0.2;
c = 0.5;

%Define time span
tspan = [0 150];
samp_freq=80;
n_samples=samp_freq*tspan(2);
t_samp = linspace(tspan(1), tspan(2), n_samples);

% Set white noise generation
rng('default')
GWN_freq = samp_freq/2;
GWN = randn(1,n_samples/2)*0.2;

% Define options for the ODE solver
opts = odeset('RelTol', 1e-6, 'AbsTol', 1e-8, 'MaxStep', 1e-3);

% Solve the ODE system using the stiff solver ode15s
sol = ode15s(@(t,y) fitzhugh_nagumo_w_GWN(t, y,alpha , w2, a, b, c, k1, k2, GWN, GWN_freq), tspan, xy0, opts);

v = deval(sol,t_samp,1);

%Find the system bandwidth
GWN_upsample=interp1(linspace(tspan(1), tspan(2), n_samples/2),GWN, t_samp, 'previous');

%compute the first order kernel
%kernel1 = ifft(abs(y_fft)./abs(x_fft));
x = GWN_upsample;
y = v;
alphas=[0.01:0.01:0.99];
min_L = 0;
min_nmse = 999;
min_alpha = -1;
for L  = [5,6,7,8,9]
    for a = alphas
        [cest, kest, pred, nmse] = LET_1(x, y, a, L, 2, 1);
        if min_nmse > nmse
            min_nmse = nmse;
            min_alpha = a;
            min_L = L
        end
    end
end
LET_1(x, y, min_alpha, min_L, 2, 1);