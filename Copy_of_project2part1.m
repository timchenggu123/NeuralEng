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
%k1 = -0.85;
k1=1;
k2 = -1.1;
c = -0.4;

%Define time span
tspan = [0 150];
samp_freq=80;
n_samples=samp_freq*tspan(2);
t_samp = linspace(tspan(1), tspan(2), n_samples);

% Set white noise generation
rng('default')
GWN_freq = samp_freq/2;
GWN = randn(1,n_samples/2)*0.1;

% Define options for the ODE solver
opts = odeset('RelTol', 1e-6, 'AbsTol', 1e-8, 'MaxStep', 1e-3);

% Solve the ODE system using the stiff solver ode15s
sol = ode15s(@(t,y) fitzhugh_nagumo_w_GWN(t, y,alpha , w2, a, b, c, k1, k2, GWN, GWN_freq), tspan, xy0, opts);

v = deval(sol,t_samp,1);    
% Get activation spikes
spike_thresh = 0.3;
tmp = v > spike_thresh;
spikes = zeros(1, length(tmp));
spikes(strfind(tmp, [0, 1])) = 1;

figure(1)
plot(t_samp, v)
hold on
plot(t_samp, spikes, 'r')
hold off

GWN_upsample=interp1(linspace(tspan(1), tspan(2), n_samples/2),GWN, t_samp, 'previous');
%compute the first order kernel
%kernel1 = ifft(abs(y_fft)./abs(x_fft));
x = GWN_upsample;

%Find non-osscilatory kernel
x = GWN_upsample;
alphas=[0.01:0.01:0.99];
min_L = 0;
min_nmse = 999;
min_alpha = -1;
for L  = [7]
    for alph = alphas
        [cest, kest, pred, nmse] = LET_1(x, v, alph, L, 2, 1);
        if min_nmse > nmse
            min_nmse = nmse
            min_alpha = alph;
            min_L = L;
        end
    end
end
LET_1(x, v, min_alpha, min_L, 2, 1);

%Check Oscilatory kernel
k1  = -0.85;
sol = ode15s(@(t,y) fitzhugh_nagumo_w_GWN(t, y,alpha , w2, a, b, c, k1, k2, GWN, GWN_freq), tspan, xy0, opts);
v = deval(sol,t_samp,1);

% Get activation spikes
spike_thresh = 0.3;
tmp = v > spike_thresh;
spikes = zeros(1, length(tmp));
spikes(strfind(tmp, [0, 1])) = 1;

figure(3)
plot(t_samp, v)
hold on
plot(t_samp, spikes, 'r')
hold off
  
%% 

max_sim=-1;
x = x - 1.85;
[cest, kest, pred, nmse] = LET_1(x, v, a, L, 2, 2);
for spike_thresh = [0:0.05:2.0]
    activations = pred > spike_thresh;
    activations=reshape(activations, 1, length(activations));
    similarity = length(activations) - sum(xor(activations, spikes));
    if max_sim < similarity
        max_sim = similarity;
        max_spike_thresh=spike_thresh;
        max_activations = activations;
    end
end


% alphas=[0.01:0.01:0.99];
% max_L = 0;
% max_sim=-1;
% max_alpha = -1;
% max_spike_thresh = -1;
% max_activations = -1;
% for L  = [6,7,8,9]
%     for a = alphas
%         [cest, kest, pred, nmse] = LET_1(x, y, a, L, 2, 2);
%         for spike_thresh = [0.1:0.1:1.0]
%             activations = pred > spike_thresh;
%             similarity = activations*spikes;
%             if max_sim < similarity
%                 max_sim = similarity;
%                 max_alpha = a;
%                 max_L = L
%                 max_spike_thresh=spike_thresh;
%                 max_activations = activations.*spikes;
%             end
%         end
%     end
% end
% [cest, kest, pred, nmse]=LET_1(x, y, min_alpha, min_L, 2, 2);
figure(5)
hold on
plot(t_samp, spikes, 'b')
plot(t_samp, max_activations, 'r')
hold off 