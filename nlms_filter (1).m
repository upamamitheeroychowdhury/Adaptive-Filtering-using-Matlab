function [y, e, w_hist] = nlms_filter(x, d, L, mu, eps)
% Normalized LMS Filter

x = x(:); d = d(:);
N = length(x);

if nargin < 5
    eps = 1e-6;   % regularization to avoid division by zero
end

w = zeros(L,1);
w_hist = zeros(L,N);
y = zeros(N,1);
e = zeros(N,1);

for n = L:N
    x_n = x(n:-1:n-L+1);
    y(n) = w.' * x_n;
    e(n) = d(n) - y(n);

    % NLMS update
    power = (x_n.'*x_n) + eps;     % ||x_n||^2
    w = w + (mu/power) * e(n) * x_n;

    w_hist(:,n) = w;
end
end
