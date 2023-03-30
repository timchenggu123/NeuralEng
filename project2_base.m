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

% %%% Find the intrinsic response first
% Set white noise generation
rng('default')
GWN_freq = samp_freq/2;
GWN = zeros(1, n_samples/2);

% Define options for the ODE solver
opts = odeset('RelTol', 1e-6, 'AbsTol', 1e-8, 'MaxStep', 1e-3);

% Solve the ODE system using the stiff solver ode15s
sol = ode15s(@(t,y) fitzhugh_nagumo_w_GWN(t, y,alpha , w2, a, b, c, k1, k2, GWN, GWN_freq), tspan, xy0, opts);

v = deval(sol,t_samp,1);
figure(3);
plot(t_samp,v)

%%% Find the GWN response now
% Set white noise generation
rng('default')
GWN_freq = samp_freq/2;
GWN = randn(1,n_samples/2)*0.2;

% Define options for the ODE solver
opts = odeset('RelTol', 1e-6, 'AbsTol', 1e-8, 'MaxStep', 1e-3);

% Solve the ODE system using the stiff solver ode15s
sol = ode15s(@(t,y) fitzhugh_nagumo_w_GWN(t, y,alpha , w2, a, b, c, k1, k2, GWN, GWN_freq), tspan, xy0, opts);

v = deval(sol,t_samp,1);
figure(1)
subplot(4,2,1)
plot(t_samp,v)

%Find the system bandwidth
GWN_upsample=interp1(linspace(tspan(1), tspan(2), n_samples/2),GWN, t_samp, 'previous');
h = hamming(length(v));
h = reshape(h,[1,length(h)]);
x_fft = fft(GWN_upsample.*h);
y_fft = fft(v.*h);
freqspace = (1:length(t_samp)) .* samp_freq ./ length(t_samp);
subplot(4,2,3);
plot(freqspace, x_fft);
subplot(4,2,4);
plot(freqspace, y_fft);

fprintf("max GWN bandwidth:\n");
[val, idx] = max(abs(x_fft));
idx = samp_freq * idx / n_samples
fprintf("max response bandwidth: \n");
[val, idx] = max(abs(y_fft));
idx = samp_freq * idx / n_samples

%compute the first order kernel
kernel1 = ifft(abs(y_fft)./abs(x_fft));
%kernel1 = lee_schetzen(GWN_upsample,v);
figure(2);
hold on
title('First order kernel')
plot(t_samp, kernel1);
plot(t_samp, zeros(length(t_samp)));
hold off