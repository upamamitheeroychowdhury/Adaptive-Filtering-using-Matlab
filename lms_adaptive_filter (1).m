function [y, e, w_hist] = lms_adaptive_filter(x, d, L, mu)
% LMS_ADAPTIVE_FILTER  Standard LMS adaptive FIR filter.
%
% Inputs:
%   x   : reference input signal (Nx1)
%   d   : desired signal (Nx1)  (here: noisy speech)
%   L   : filter length
%   mu  : step size
%
% Outputs:
%   y       : filter output
%   e       : error signal, e(n) = d(n) - y(n)
%   w_hist  : L x N matrix of weight vectors over time

N = length(x);
x = x(:);          % ensure column vectors
d = d(:);

w = zeros(L,1);    % initial weights
w_hist = zeros(L,N);
y = zeros(N,1);
e = zeros(N,1);

for n = L:N
    % regressor vector x_n = [x(n), x(n-1), ..., x(n-L+1)]^T
    x_n = x(n:-1:n-L+1);
    
    % filter output
    y(n) = w.' * x_n;
    
    % error
    e(n) = d(n) - y(n);
    
    % weight update (LMS)
    w = w + mu * e(n) * x_n;
    
    % store weights
    w_hist(:, n) = w;
end

end
