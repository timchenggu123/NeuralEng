% Define the FitzHugh-Nagumo ODE system
function out=fitzhugh_nagumo_w_GWN(t,xy,alpha , w2, a, b, c, k1, k2, GWN, GWN_freq)
    x = xy(1);
    y = xy(2);
    x1 = xy(3);
    y1 = xy(4);
    t_samp = floor(t/(1/GWN_freq)) + 1;
    if t_samp > length(GWN)
        t_samp=t_samp - 1;
    end
    noise = GWN(t_samp);
    out = zeros(4,1);
    out(1) = alpha*(y + x - (x^3)/3 + (k1 + c*x1)+noise);
    out(2) = -(1/alpha) * (w2*x - a + b*y);
    out(3) = alpha*(y1 + x1 - (x1^3)/3 + (k2 + c*x));
    out(4) = -(1/alpha) * (w2*x1 - a + b*y1);
end