function [y, e, w_hist] = leaky_lms_filter(x, d, L, mu, gamma)
% LEAKY_LMS_FILTER  Leaky LMS adaptive FIR filter.
% Implements the (1 - mu*gamma)*w update.
N = length(x);
x = x(:); d = d(:);
w = zeros(L,1); w_hist = zeros(L,N);
y = zeros(N,1); e = zeros(N,1);
for n = L:N
    x_n = x(n:-1:n-L+1);
    y(n) = w.' * x_n;
    e(n) = d(n) - y(n);
    % Leaky LMS Update: w = (1 - mu*gamma)*w + mu*e[n]*x[n]
    w = (1 - mu*gamma)*w + mu*e(n)*x_n;
    w_hist(:, n) = w;
end
end