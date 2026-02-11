function w_opt = wiener_fir_solution(x, d, L)
% WIENER_FIR_SOLUTION  Empirical Wiener solution for order-L FIR filter.
%
% x : input (reference) signal
% d : desired signal
% L : filter order

x = x(:); d = d(:);
N = length(x);

R = zeros(L, L);
p = zeros(L, 1);

for n = L:N
    x_n = x(n:-1:n-L+1);
    R = R + x_n * x_n.';
    p = p + d(n) * x_n;
end

R = R / (N-L+1);
p = p / (N-L+1);

w_opt = R \ p;       % R^{-1} p
end
