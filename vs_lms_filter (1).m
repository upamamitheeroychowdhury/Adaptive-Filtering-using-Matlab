function [y, e, w_hist] = vs_lms_filter(x, d, L, mu_init, mu_min, mu_max, alpha, beta)
% VS_LMS_FILTER  Variable Step-Size LMS (VS-LMS) filter (per-tap).
N = length(x);
x = x(:); d = d(:);
w = zeros(L,1);
mu_vec = mu_init * ones(L,1);   % per-tap step sizes
prev_sign = zeros(L,1);         % for tracking sign change
w_hist = zeros(L,N);
y = zeros(N,1);
e = zeros(N,1);
for n = L:N
    x_n = x(n:-1:n-L+1);
    y(n) = w.' * x_n;
    e(n) = d(n) - y(n);
    
    for k = 1:L
        % Sign of the instantaneous gradient (e[n] * x[n])
        sgn = sign(e(n)*x_n(k));
        
        if sgn * prev_sign(k) > 0       % Gradient sign unchanged -> Increase mu
            mu_vec(k) = min(mu_vec(k) + alpha, mu_max);
        elseif sgn * prev_sign(k) < 0   % Gradient sign changed -> Decrease mu
            mu_vec(k) = max(mu_vec(k) * beta, mu_min);
        end
        
        % Per-tap weight update
        w(k) = w(k) + mu_vec(k)*e(n)*x_n(k);
        prev_sign(k) = sgn;
    end
    w_hist(:,n) = w;
end
end