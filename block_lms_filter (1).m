function [y, e, w_hist] = block_lms_filter(x, d, L, mu, B)
% BLOCK_LMS_FILTER  Block LMS adaptive FIR filter (time-domain).
% B is the block size.
N = length(x);
x = x(:); d = d(:);
w = zeros(L,1); w_hist = zeros(L,N);
y = zeros(N,1); e = zeros(N,1);
for k = L : B : (N-B+1)
    grad = zeros(L,1);
    for l = 0 : B-1
        n = k + l;
        if n > N, break; end
        x_n = x(n:-1:n-L+1);
        y(n) = w.' * x_n;
        e(n) = d(n) - y(n);
        grad = grad + e(n)*x_n;
    end
    % Block Update: w_k+1 = w_k + (mu/B) * Grad_avg
    w = w + (mu/B)*grad;
    % Store weights for the entire block
    w_hist(:, k:min(k+B-1,N)) = repmat(w,1,min(B,N-k+1));
end
% Ensure remaining samples are processed with final weights
if k <= N
    w_hist(:, k:N) = repmat(w,1,N-k+1);
end
end