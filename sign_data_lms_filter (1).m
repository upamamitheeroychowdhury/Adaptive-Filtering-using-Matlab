function [y, e, w_hist] = sign_data_lms_filter(x, d, L, mu)
% SIGN_DATA_LMS_FILTER  Sign-Data LMS adaptive FIR filter.
% Only the input data vector is signed element-wise.
N = length(x);
x = x(:); d = d(:);
w = zeros(L,1); w_hist = zeros(L,N);
y = zeros(N,1); e = zeros(N,1);
for n = L:N
    x_n = x(n:-1:n-L+1);
    y(n) = w.' * x_n;
    e(n) = d(n) - y(n);
    % Sign-Data LMS Update: w = w + mu * e[n] * sign(x[n])
    w = w + mu * e(n) * sign(x_n);
    w_hist(:, n) = w;
end
end