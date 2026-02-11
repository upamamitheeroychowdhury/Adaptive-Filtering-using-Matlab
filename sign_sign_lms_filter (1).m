function [y, e, w_hist] = sign_sign_lms_filter(x, d, L, mu)
% SIGN_SIGN_LMS_FILTER  Sign-Sign LMS adaptive FIR filter.
% Both the error signal and the data vector are signed.
N = length(x);
x = x(:); d = d(:);
w = zeros(L,1); w_hist = zeros(L,N);
y = zeros(N,1); e = zeros(N,1);
for n = L:N
    x_n = x(n:-1:n-L+1);
    y(n) = w.' * x_n;
    e(n) = d(n) - y(n);
    % Sign-Sign LMS Update: w = w + mu * sign(e[n]) * sign(x[n])
    w = w + mu * sign(e(n)) .* sign(x_n);
    w_hist(:, n) = w;
end
end