function out=lee_schetzen(x,y)
   y_mean = mean(y);
   y=y-y_mean;
   r=xcorr(x, y);
   [m,n]=size(r);
   r=r(ceil(n/2):n);
   out=r;
end