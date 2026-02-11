function tuning_experiment(x, d)
% TUNING_EXPERIMENT  Explore effect of mu and L for LMS on given (x,d).
%
%   tuning_experiment(x, d)
%
%   x : input / reference signal (e.g., noise reference to ANC)
%   d : desired signal (e.g., noisy speech)
%
%   Uses your lms_adaptive_filter(x, d, L, mu) function.

x = x(:);
d = d(:);
N = length(d);
Nmax = min(N, 5000);   % use at most 5000 samples for tuning
x = x(1:Nmax);
d = d(1:Nmax);
N = Nmax;

%% --------------- EXPT 1: Vary step size mu --------------------
mu_vec = [1e-4 5e-4 1e-3 5e-3 1e-2];   % you can refine these
L_fix  = 32;                           % fixed filter length (can change)

mse_mu = zeros(length(mu_vec), N);

for i = 1:length(mu_vec)
    mu = mu_vec(i);
    [~, e, ~] = lms_adaptive_filter(x, d, L_fix, mu);
    mse_mu(i,:) = e(:)'.^2;            % instantaneous squared error
end

% smooth MSE curves for readability
win = 200;
for i = 1:length(mu_vec)
    mse_mu(i,:) = movmean(mse_mu(i,:), win);
end

figure;
for i = 1:length(mu_vec)
    semilogy(mse_mu(i,:), 'LineWidth', 1.2); hold on;
end
grid on;
xlabel('Iteration n');
ylabel('Smoothed MSE  E\{e^2[n]\}');
title(sprintf('LMS tuning: effect of \\mu (L = %d)', L_fix));
legend(arrayfun(@(m) sprintf('\\mu = %.0e', m), mu_vec, ...
       'UniformOutput', false), 'Location','best');


%% --------------- EXPT 2: Vary filter length L -----------------
L_vec  = [4 8 16 32 64];        % try a range of lengths
mu_fix = 1e-3;                  % pick a "reasonable" step size

mse_L = zeros(length(L_vec), N);

for i = 1:length(L_vec)
    L = L_vec(i);
    [~, e, ~] = lms_adaptive_filter(x, d, L, mu_fix);
    mse_L(i,:) = e(:)'.^2;
end

for i = 1:length(L_vec)
    mse_L(i,:) = movmean(mse_L(i,:), win);
end

figure;
for i = 1:length(L_vec)
    semilogy(mse_L(i,:), 'LineWidth', 1.2); hold on;
end
grid on;
xlabel('Iteration n');
ylabel('Smoothed MSE  E\{e^2[n]\}');
title(sprintf('LMS tuning: effect of L (\\mu = %.0e)', mu_fix));
legend(arrayfun(@(L) sprintf('L = %d', L), L_vec, ...
       'UniformOutput', false), 'Location','best');


%% --------------- EXPT 3: Summary surface (optional) ----------
% final steady–state MSE as a function of (mu, L)

mu_vec2 = logspace(-4,-2,7);     % finer grid in log scale
L_vec2  = [4 8 16 32 64];

Jss = zeros(length(L_vec2), length(mu_vec2));  % steady-state MSE

N_ss = round(0.7*N);             % ignore first 70% (transient)

for iL = 1:length(L_vec2)
    for im = 1:length(mu_vec2)
        L  = L_vec2(iL);
        mu = mu_vec2(im);
        [~, e, ~] = lms_adaptive_filter(x, d, L, mu);
        Jss(iL, im) = mean(e(N_ss:end).^2);    % steady-state MSE
    end
end

figure;
surf(mu_vec2, L_vec2, Jss);
set(gca, 'XScale', 'log');
xlabel('\mu');
ylabel('L');
zlabel('Steady-state MSE');
title('Steady-state MSE vs. (\mu, L)');
shading interp; grid on; view(135, 30);

end
