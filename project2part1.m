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
k1 = -0.85;
k2 = -1.1;
c = -0.4;

%Define time span
tspan = [0 150];
samp_freq=100;
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
spikes = v > spike_thresh;

figure(1);
plot(t_samp, v)
hold on
plot(t_samp, spikes, 'r')
hold off

GWN_upsample=interp1(linspace(tspan(1), tspan(2), n_samples/2),GWN, t_samp, 'previous');
%compute the first order kernel

x = GWN_upsample;
alphas=[0.5:0.01:0.99];
max_L = 0;
max_auc=-1;
max_alpha = -1;
max_spike_thresh = -1;
max_activations = -1;
for L  = [8, 9, 10, 11, 12]
     for a = alphas
        [cest, kest, pred, nmse] = LET_1(x, v, a, L, 2, 2);
        [fpr, tpr, auc_score] = calc_roc_curve(pred, spikes);
        if max_auc < auc_score
            max_auc = auc_score;
            min_alpha = a;
            min_L = L;
        end
     end
end
[cest, kest, pred, nmse] = LET_1(x, v, min_alpha, min_L, 2, 2);
[fpr, tpr, auc_score] = calc_roc_curve(pred, spikes);
% Plot ROC curve
figure(5);
plot(fpr, tpr); 
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve');

%%
