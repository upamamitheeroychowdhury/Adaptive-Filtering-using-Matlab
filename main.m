%% adaptive_filter_project.m
% ECE-541 Final Project – Adaptive Filtering
% Noise Cancellation with LMS-family algorithms
% Upama Roy Chowdhury

clear; clc; close all;
rng(1);                            % for reproducibility
set(groot, 'defaultFigureRenderer', 'painters');

%% ------------------------------------------------------------------------
% 1. Load or generate a clean speech signal
% -------------------------------------------------------------------------
load handel                        % gives y, Fs
s = y;

% Normalize speech to unit variance
s = s - mean(s);
s = s / std(s);

N = length(s);                     % number of samples
xi_min_theory = var(s);            % Theoretical minimum MSE (MMSE) = Power of clean speech

%% ------------------------------------------------------------------------
% 2. Generate different types of noise (baseline mixture)
% -------------------------------------------------------------------------
% 2.1 White Gaussian noise (unit variance)
wgn = randn(N, 1);

% 2.2 AR(1) colored noise: v[n] = 0.9 v[n-1] + e[n]
ar_a = [1 -0.9];
ar_b = 1;
ar_exc = randn(N, 1);
ar_noise = filter(ar_b, ar_a, ar_exc);
ar_noise = ar_noise / std(ar_noise);   % normalize

% 2.3 Baseline sinusoidal interference at 400 Hz (used in all baseline WGN cases)
f0_base = 400;
t = (0:N-1)'/Fs;
sin_noise_base = sin(2*pi*f0_base*t);
sin_noise_base = sin_noise_base / std(sin_noise_base);

% Mix all three noises with desired relative powers (baseline mixture)
sigma_wgn   = 0.4;
sigma_ar    = 0.6;
sigma_sine  = 0.5;

noise_primary = sigma_wgn*wgn + sigma_ar*ar_noise + sigma_sine*sin_noise_base;

% Primary microphone: speech + noise (baseline)
d = s + noise_primary;                 % "desired" but noisy signal
d = d - mean(d);                       % Re-normalize d
d = d / std(d);

% Re-normalize clean speech again (consistent scaling)
s = s - mean(s);
s = s / std(s);

%% ------------------------------------------------------------------------
% 3. Build reference noise for adaptive canceller
% -------------------------------------------------------------------------
% Channel for reference noise
b_ref = [1 0.3 -0.2];                  % 3-tap FIR channel
ref_raw = noise_primary;
x = filter(b_ref, 1, ref_raw);         % reference input to adaptive filter
x = x / std(x);

%% ============= 3b. TUNING EXPERIMENT (OPTIONAL) =========================
% Commented out to focus on the final performance runs. Uncomment to run
% original tuning_experiment(x, d) function.
% tuning_experiment(x, d);

%% ------------------------------------------------------------------------
% 4. Adaptive Filters: LMS-family in parallel
% -------------------------------------------------------------------------
 L       = 32;       % filter length (based on tuning_experiment)
 mu_lms  = 0.01;
 mu_nlms = 0.8;

% Alternative settings (kept as comments)
%L       = 64;
%mu_lms  = 0.002;
%mu_nlms = 0.25;

mu_leaky    = 0.01;
gamma_leaky = 1e-3;   % leakage coefficient
mu_block    = 0.01;
B_block     = 32;      % block size
mu_sign_err  = 0.01;
mu_sign_data = 0.01;
mu_sign_sign = 0.01;
mu_vs_init = 0.005; mu_vs_min = 1e-4; mu_vs_max = 0.05; alpha_vs = 1e-4; beta_vs  = 0.5;

% 4.1 Standard LMS
[y_lms, e_lms, w_hist_lms] = lms_adaptive_filter(x, d, L, mu_lms);
s_hat_lms = e_lms;

% 4.2 NLMS
[y_nlms, e_nlms, w_hist_nlms] = nlms_filter(x, d, L, mu_nlms, 1e-6);
s_hat_nlms = e_nlms;

% 4.3 Leaky LMS
[y_leaky, e_leaky, w_hist_leaky] = leaky_lms_filter(x, d, L, mu_leaky, gamma_leaky);
s_hat_leaky = e_leaky;

% 4.4 Block LMS
[y_block, e_block, w_hist_block] = block_lms_filter(x, d, L, mu_block, B_block);
s_hat_block = e_block;

% 4.5 Sign-Error LMS
[y_se, e_se, w_hist_se] = sign_error_lms_filter(x, d, L, mu_sign_err);
s_hat_se = e_se;

% 4.6 Sign-Data LMS
[y_sd, e_sd, w_hist_sd] = sign_data_lms_filter(x, d, L, mu_sign_data);
s_hat_sd = e_sd;

% 4.7 Sign-Sign LMS
[y_ss, e_ss, w_hist_ss] = sign_sign_lms_filter(x, d, L, mu_sign_sign);
s_hat_ss = e_ss;

% 4.8 Variable Step-Size LMS (VS-LMS)
[y_vs, e_vs, w_hist_vs] = vs_lms_filter(x, d, L, mu_vs_init, mu_vs_min, mu_vs_max, alpha_vs, beta_vs);
s_hat_vs = e_vs;

%% ------------------------------------------------------------------------
% 5. Performance Evaluation and Analytical Comparison Table
% -------------------------------------------------------------------------
win = 200;   % smoothing window for learning curves
steady_state_idx = round(0.7*N) : N; % Use last 30% of data for steady state

alg_names = {'LMS','NLMS','Leaky','Block','SignErr','SignData','SignSign','VS-LMS'};
e_all = {e_lms, e_nlms, e_leaky, e_block, e_se, e_sd, e_ss, e_vs};
s_hat_all = {s_hat_lms, s_hat_nlms, s_hat_leaky, s_hat_block, ...
             s_hat_se, s_hat_sd, s_hat_ss, s_hat_vs};
w_final_all = {w_hist_lms(:,end), w_hist_nlms(:,end), w_hist_leaky(:,end), w_hist_block(:,end), ...
               w_hist_se(:,end), w_hist_sd(:,end), w_hist_ss(:,end), w_hist_vs(:,end)};

inst_mse_all    = cell(size(e_all));
mse_smooth_all  = cell(size(e_all));
snr_clean_all   = zeros(size(e_all));
mse_ss_all      = zeros(size(e_all));
misadjustment_all = zeros(size(e_all));

% Optimal Wiener solution (for Misalignment comparison)
w_wiener = wiener_fir_solution(x, d, L);

% Calculate metrics for all algorithms
for i = 1:numel(e_all)
    e_i = e_all{i};
    inst_mse_all{i}   = e_i.^2;
    mse_smooth_all{i} = filter(ones(win,1)/win,1,inst_mse_all{i});

    % Output SNR (using error between estimate and clean speech)
    snr_clean_all(i) = snr(s, s_hat_all{i} - s);

    % Steady-state MSE (xi_inf)
    mse_ss = mean(inst_mse_all{i}(steady_state_idx));
    mse_ss_all(i) = mse_ss;

    % Misadjustment M = (xi_inf - xi_min) / xi_min
    misadjustment_all(i) = (mse_ss - xi_min_theory) / xi_min_theory;
end

% Input SNR (noisy vs clean)
snr_noisy = snr(s, d - s);

% --- Print Analytical Comparison Table to Command Window ---
fprintf('\n----------------------------------------------------------------------------------\n');
fprintf('ANALYTICAL PERFORMANCE COMPARISON TABLE (L = %d)\n', L);
fprintf('MMSE (xi_min) is: %.4f\n', xi_min_theory);
fprintf('----------------------------------------------------------------------------------\n');
fprintf('  Algorithm | SNR Out (dB) | MSE_ss | Misalignment (||w_err||) | Misadjustment (%%)\n');
fprintf('  ----------|--------------|--------|---------------------------|--------------------\n');

for i = 1:numel(e_all)
    % Misalignment: || w_final - w_wiener ||
    wev_norm = norm(w_final_all{i} - w_wiener);

    fprintf('  %8s | %12.2f | %6.4f | %25.6f | %18.2f\n', ...
            alg_names{i}, snr_clean_all(i), mse_ss_all(i), wev_norm, misadjustment_all(i)*100);
end
fprintf('  Input     | %12.2f |\n', snr_noisy);
fprintf('----------------------------------------------------------------------------------\n');

% Save results to a .mat file (for later analysis or plotting)
save('adaptive_filter_results.mat', 's','d','x','s_hat_all','alg_names', ...
     'w_final_all', 'w_wiener','snr_noisy','snr_clean_all');

%% ------------------------------------------------------------------------
% 6. Plots 
% -------------------------------------------------------------------------

%% 6.1 Time-domain demo (first 1000 samples)
range = 1:1000;
figure('Position',[200 150 850 900]) % Large high-res figure

% (a) Clean Speech
subplot(5,1,1)
plot(t(range), s(range),'LineWidth',1.4)
ylim([min(s(range)) max(s(range))]*1.05)
title('(a) Clean Speech s[n]')
ylabel('Amplitude')
grid on

% (b) Noisy Speech (primary mic)
subplot(5,1,2)
plot(t(range), d(range),'LineWidth',1.4)
ylim([min(d(range)) max(d(range))]*1.05)
title('(b) Noisy Speech d[n]')
ylabel('Amplitude')
grid on

% (c) Reference Signal (used by LMS/NLMS)
subplot(5,1,3)
plot(t(range), x(range),'LineWidth',1.3)
ylim([min(x(range)) max(x(range))]*1.05)
title('(c) Reference Signal x[n]')
ylabel('Amplitude')
grid on

% (d) LMS Recovered
subplot(5,1,4)
plot(t(range), s_hat_lms(range),'b','LineWidth',1.4)
ylim([min(s_hat_lms(range)) max(s_hat_lms(range))]*1.05)
title('(d) LMS Recovered Speech')
ylabel('Amplitude')
grid on

% (e) NLMS Recovered
subplot(5,1,5)
plot(t(range), s_hat_nlms(range),'m','LineWidth',1.4)
ylim([min(s_hat_nlms(range)) max(s_hat_nlms(range))]*1.05)
title('(e) NLMS Recovered Speech')
xlabel('Time (s)')
ylabel('Amplitude')
grid on

%% 6.2 Learning curves (MSE) – all algorithms
figure;
hold on;
for i = 1:numel(alg_names)
    plot(10*log10(mse_smooth_all{i}), 'DisplayName', alg_names{i});
end
yline(10*log10(xi_min_theory), 'k--', 'DisplayName', 'Wiener MMSE');
grid on;
xlabel('Iteration n');
ylabel('Smoothed MSE (dB)');
title(sprintf('Learning Curves Comparison, L = %d', L));
legend('Location','best');

%% 6.3 Evolution of filter coefficients – LMS vs NLMS
figure;
subplot(2,1,1);
plot(w_hist_lms.'); grid on;
xlabel('Iteration n'); ylabel('Coefficient value');
title('Evolution of LMS Filter Weights');

subplot(2,1,2);
plot(w_hist_nlms.'); grid on;
xlabel('Iteration n'); ylabel('Coefficient value');
title('Evolution of NLMS Filter Weights');

%% 6.4 PSD comparison – clean vs noisy vs LMS vs NLMS
nfft = 2048;

% Use only samples after convergence for PSD (e.g., last 70% of data)
idx_psd     = round(0.3*N) : N;
s_psd       = s(idx_psd);
d_psd       = d(idx_psd);
s_lms_psd   = s_hat_lms(idx_psd);
s_nlms_psd  = s_hat_nlms(idx_psd);

[Sp,  f]   = pwelch(s_psd,      hamming(512),256,nfft,Fs);
[Dp,  ~]   = pwelch(d_psd,      hamming(512),256,nfft,Fs);
[Slms,~]   = pwelch(s_lms_psd,  hamming(512),256,nfft,Fs);
[Snlm,~]   = pwelch(s_nlms_psd, hamming(512),256,nfft,Fs);

figure;
plot(f,10*log10(Sp),'LineWidth',1.5); hold on;
plot(f,10*log10(Dp),'LineWidth',1.5);
plot(f,10*log10(Slms),'LineWidth',1.5);
plot(f,10*log10(Snlm),'LineWidth',1.5);
grid on;
xlabel('Frequency (Hz)');
ylabel('Power / Frequency (dB/Hz)');
title('PSD of Clean / Noisy / LMS / NLMS Recovered Speech');
legend('Clean speech','Noisy speech','LMS','NLMS','Location','best');
xlim([0 Fs/2]);

%% 6.4b Bilateral spectrum – clean vs noisy vs LMS vs NLMS
nfft_bi = 4096;

% Compute FFTs and shift zero frequency to center
S_clean = fftshift(fft(s,         nfft_bi));
S_noisy = fftshift(fft(d,         nfft_bi));
S_lms   = fftshift(fft(s_hat_lms, nfft_bi));
S_nlms  = fftshift(fft(s_hat_nlms,nfft_bi));

% Frequency axis for bilateral spectrum
f_bi = (-nfft_bi/2:nfft_bi/2-1) * (Fs/nfft_bi);

% Convert to dB magnitude
S_clean_db = 20*log10(abs(S_clean) + eps);
S_noisy_db = 20*log10(abs(S_noisy) + eps);
S_lms_db   = 20*log10(abs(S_lms)   + eps);
S_nlms_db  = 20*log10(abs(S_nlms)  + eps);

figure;
plot(f_bi, S_clean_db,'LineWidth',1.5); hold on;
plot(f_bi, S_noisy_db,'LineWidth',1.5);
plot(f_bi, S_lms_db,  'LineWidth',1.5);
plot(f_bi, S_nlms_db, 'LineWidth',1.5);
grid on;
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
title('Bilateral Spectrum of Clean / Noisy / LMS / NLMS Speech');
legend('Clean speech','Noisy speech','LMS','NLMS','Location','best');
xlim([-Fs/2 Fs/2]);

%% 6.5 Bar plot of SNR improvement
figure;
vals = [snr_noisy; snr_clean_all(:)];
bar(vals);
xticks(1:length(vals));
xticklabels([{'Input'}, alg_names]);  % cell array of labels
ylabel('SNR (dB)');
title('SNR of Input and All Adaptive Filters');
grid on;

%% ------------------------------------------------------------------------
% 7. (Optional) Listen to signals
% -------------------------------------------------------------------------
% soundsc(s, Fs);          % original clean speech
% pause(length(s)/Fs + 1);
% soundsc(d, Fs);          % noisy speech
% pause(length(s)/Fs + 1);
% soundsc(s_hat_lms, Fs);  % after LMS
% pause(length(s)/Fs + 1);
% soundsc(s_hat_nlms, Fs); % after NLMS

%% ------------------------------------------------------------------------
% 8. Two-tap LMS and NLMS weight-trajectory demos ("bowl" figures)
% -------------------------------------------------------------------------
N2    = 500;
L2    = 2;
x2    = randn(N2,1);
h_true = [0.8; -0.5];
d2 = filter(h_true,1,x2);

% --- 2-tap LMS ---
mu2_lms = 0.02;
[~, e2_lms, w_hist2_lms] = lms_adaptive_filter(x2, d2, L2, mu2_lms);
w0_lms = w_hist2_lms(1,:);
w1_lms = w_hist2_lms(2,:);

% --- 2-tap NLMS ---
mu2_nlms = 0.8;
[~, e2_nlms, w_hist2_nlms] = nlms_filter(x2, d2, L2, mu2_nlms, 1e-6);
w0_nlms = w_hist2_nlms(1,:);
w1_nlms = w_hist2_nlms(2,:);

% Cost-surface bowl (J as a function of w0, w1)
[w0g,w1g] = meshgrid(linspace(-1.5,1.5,60), linspace(-1.5,1.5,60));
J = (w0g - h_true(1)).^2 + (w1g - h_true(2)).^2;

figure;
contour(w0g,w1g,J,20); hold on; grid on;
plot(w0_lms,w1_lms,'r.-','LineWidth',1.2);
plot(w0_nlms,w1_nlms,'b.-','LineWidth',1.2);
plot(h_true(1),h_true(2),'kx','MarkerSize',10,'LineWidth',2);
xlabel('w(0)');
ylabel('w(1)');
title('2-tap LMS and NLMS Weight Trajectories in Weight Space');
legend('Error contours','LMS trajectory','NLMS trajectory','Wiener solution');

figure;
subplot(2,1,1);
plot(1:N2,w0_lms,1:N2,w1_lms,'LineWidth',1.5); hold on;
yline(h_true(1),'--'); yline(h_true(2),'--');
grid on;
xlabel('Iteration n'); ylabel('Coefficient value');
title('2-tap LMS Coefficient Evolution');
legend('w_0','w_1','Location','best');

subplot(2,1,2);
plot(1:N2,w0_nlms,1:N2,w1_nlms,'LineWidth',1.5); hold on;
yline(h_true(1),'--'); yline(h_true(2),'--');
grid on;
xlabel('Iteration n'); ylabel('Coefficient value');
title('2-tap NLMS Coefficient Evolution');
legend('w_0','w_1','Location','best');

%% ------------------------------------------------------------------------
% 8b. Explicit Steepest-Descent Demo (Batch R,p) on the 2-tap Example
% -------------------------------------------------------------------------
% Cost function: J(w) = E{ (d2[n] - w^T u[n])^2 }, u[n] = [x2[n]; x2[n-1]]
% Steepest descent:
%   w_{k+1} = w_k - 2 * mu_sd * (R_hat * w_k - p_hat)

% Build regression matrix U and desired vector d_seg
N_sd  = N2 - L2 + 1;                   % number of regression vectors
U     = zeros(N_sd, L2);
d_seg = d2(L2:end);                    % align with u[n]

for n = 1:N_sd
    U(n, :) = x2(n+L2-1:-1:n).';       % u[n]^T
end

% Sample estimates of R and p
R_hat = (U.' * U) / N_sd;              % L2 x L2 correlation matrix
p_hat = (U.' * d_seg) / N_sd;          % L2 x 1 cross-correlation vector

% Choose stable step size: 0 < mu_sd < 1 / lambda_max(R_hat)
lambda_max = max(eig(R_hat));
mu_sd      = 1 / (2 * lambda_max);     % conservative choice

% Steepest-descent iterations
K_sd       = 50;                       % number of SD iterations
w_sd       = zeros(L2, 1);             % initial guess
w_hist2_sd = zeros(L2, K_sd);          % weight trajectory
J_sd       = zeros(K_sd, 1);           % cost vs iteration

for k = 1:K_sd
    gradJ = 2 * (R_hat * w_sd - p_hat);   % exact gradient
    w_sd  = w_sd - mu_sd * gradJ;         % SD update

    w_hist2_sd(:, k) = w_sd;

    % Approximate MSE cost at iteration k
    err_k   = d_seg - U * w_sd;
    J_sd(k) = mean(err_k.^2);
end

% SD trajectory components for plotting
w0_sd = w_hist2_sd(1, :);
w1_sd = w_hist2_sd(2, :);

% Plot SD trajectory together with LMS and NLMS on the bowl
figure;
contour(w0g, w1g, J, 20); hold on; grid on;
plot(w0_lms,  w1_lms,  'r.-', 'LineWidth', 1.2);
plot(w0_nlms, w1_nlms, 'b.-', 'LineWidth', 1.2);
plot(w0_sd,   w1_sd,   'g.-', 'LineWidth', 1.5);
plot(h_true(1), h_true(2), 'kx', 'MarkerSize', 10, 'LineWidth', 2);
xlabel('w(0)');
ylabel('w(1)');
title('2-tap Steepest Descent vs LMS and NLMS on Cost Surface');
legend('Error contours', ...
       'LMS trajectory', ...
       'NLMS trajectory', ...
       'Steepest-descent trajectory', ...
       'Wiener solution', ...
       'Location', 'best');

% SD learning curve
figure;
plot(1:K_sd, 10*log10(J_sd), 'g-o', 'LineWidth', 1.5, 'MarkerSize', 4);
grid on;
xlabel('Iteration k');
ylabel('MSE J(w_k) (dB)');
title('Steepest-Descent Learning Curve (Batch R,p) for 2-tap Example');

%% ------------------------------------------------------------------------
% 9. LMS Step-Size Trade-off (Convergence vs Misadjustment)
% -------------------------------------------------------------------------
N_lc   = length(s);
R_lc   = 50;             % Number of independent runs for smoother average
L_comp = L;              % Use the filter length defined in Section 4

% Define mu values for comparison
mu_slow   = 0.001;
mu_medium = mu_lms;      % standard mu from Section 4
mu_fast   = 0.05;

% Pre-allocate storage for ensemble-averaged squared error
J_slow   = zeros(N_lc,1);
J_medium = zeros(N_lc,1);
J_fast   = zeros(N_lc,1);

for r = 1:R_lc
    [~, e_slow,   ~] = lms_adaptive_filter(x, d, L_comp, mu_slow);
    [~, e_medium, ~] = lms_adaptive_filter(x, d, L_comp, mu_medium);
    [~, e_fast,   ~] = lms_adaptive_filter(x, d, L_comp, mu_fast);

    J_slow   = J_slow   + e_slow.^2;
    J_medium = J_medium + e_medium.^2;
    J_fast   = J_fast   + e_fast.^2;
end

J_slow   = J_slow   / R_lc;
J_medium = J_medium / R_lc;
J_fast   = J_fast   / R_lc;

figure;
plot(10*log10(J_slow),   'r-',  'LineWidth', 1.5, 'DisplayName', sprintf('\\mu = %.4f (Slow)', mu_slow)); hold on;
plot(10*log10(J_medium), 'b--', 'LineWidth', 1.5, 'DisplayName', sprintf('\\mu = %.2f (Medium)', mu_medium));
plot(10*log10(J_fast),   'k:',  'LineWidth', 1.5, 'DisplayName', sprintf('\\mu = %.2f (Fast)', mu_fast));
yline(10*log10(xi_min_theory), 'g-.', 'LineWidth', 1.5, 'DisplayName', 'Wiener MMSE');
grid on;
xlabel('Iteration n');
ylabel('Ensemble Averaged MSE (dB)');
title(sprintf('LMS Convergence vs Step Size (L = %d)', L_comp));
legend('Location', 'best');
xlim([0 N_lc]);

%% ------------------------------------------------------------------------
% 10. Nonstationary-noise cancellation demo 
% -------------------------------------------------------------------------
N_ns = 1000;
n_ns = (0:N_ns-1)';

% (a) Desired sinusoid to be estimated
f_des = 0.02*pi;                       % rad/sample (slow sinusoid)
s_ns  = 2 * sin(f_des*n_ns);           % desired clean sinusoid

% Time-varying (nonstationary) noise: variance slowly increases with n
sigma_ns = linspace(0.5, 2.5, N_ns)';  % time-varying std
v_ns     = sigma_ns .* randn(N_ns,1);  % nonstationary noise

% (b) Noise-corrupted sinusoid at primary sensor
d_ns = s_ns + v_ns;

% (c) Reference noise at secondary sensor (correlated with v_ns)
b_ref_ns = [1 0.5 -0.3];               % some secondary-path filter
x_ns = filter(b_ref_ns, 1, v_ns);      % reference signal

% NLMS adaptive noise canceller (12th-order)
L_ns   = 12;
beta_ns = 0.25;                        % normalized step size beta
eps_ns  = 1e-6;
[~, e_ns, ~] = nlms_filter(x_ns, d_ns, L_ns, beta_ns, eps_ns);
y_ns = e_ns;                           % NLMS output (estimate of sinusoid)

figure;
subplot(4,1,1);
plot(n_ns, s_ns); grid on;
ylabel('Amp');
title('(a) Desired signal s_{ns}[n]');

subplot(4,1,2);
plot(n_ns, d_ns); grid on;
ylabel('Amp');
title('(b) Noise-corrupted sinusoid d_{ns}[n]');

subplot(4,1,3);
plot(n_ns, x_ns); grid on;
ylabel('Amp');
title('(c) Reference signal x_{ns}[n]');

subplot(4,1,4);
plot(n_ns, y_ns); grid on;
xlabel('Sample index n');
ylabel('Amp');
title('(d) Output of 12th-order NLMS noise canceller (\beta = 0.25)');
%% ------------------------------------------------------------------------
% 10b. Nonstationary-noise cancellation demo (time-varying variance, LMS)
% -------------------------------------------------------------------------

% Use the same desired signal s_ns, nonstationary noise v_ns,
% primary d_ns = s_ns + v_ns, and reference x_ns from Section 10.

mu_ns_lms = 0.005;                 % LMS step size (small for stability)

% LMS adaptive noise canceller on the same nonstationary data
[~, e_ns_lms, ~] = lms_adaptive_filter(x_ns, d_ns, L_ns, mu_ns_lms);
y_ns_lms = e_ns_lms;               % LMS output (estimate of sinusoid)

% --- Time-domain comparison of outputs ---
figure;
plot(n_ns, s_ns, 'k--', 'LineWidth', 1.5); hold on;
plot(n_ns, y_ns_lms, 'r',  'LineWidth', 1.2);
plot(n_ns, y_ns,     'b',  'LineWidth', 1.2);   % y_ns from NLMS section
grid on;
xlabel('Sample index n');
ylabel('Amplitude');
title('Nonstationary Noise Cancellation: LMS vs NLMS (Time Domain)');
legend('Desired s_{ns}[n]', ...
       sprintf('LMS (\\mu = %.4f)', mu_ns_lms), ...
       sprintf('NLMS (\\beta = %.2f)', beta_ns), ...
       'Location','best');

% --- Learning-curve comparison (MSE vs n) ---
win_ns = 40;  % short moving-average window

% MSE between estimate and true sinusoid
mse_lms_ns  = movmean((s_ns - y_ns_lms).^2, win_ns);
mse_nlms_ns = movmean((s_ns - y_ns    ).^2, win_ns);

% Output SNRs
snr_lms_ns  = snr(s_ns, y_ns_lms - s_ns);
snr_nlms_ns = snr(s_ns, y_ns     - s_ns);

figure;
plot(10*log10(mse_lms_ns),  'r', 'LineWidth', 1.5, ...
    'DisplayName', sprintf('LMS (\\mu = %.4f, SNR = %.2f dB)', ...
                           mu_ns_lms, snr_lms_ns)); hold on;
plot(10*log10(mse_nlms_ns), 'b', 'LineWidth', 1.5, ...
    'DisplayName', sprintf('NLMS (\\beta = %.2f, SNR = %.2f dB)', ...
                           beta_ns, snr_nlms_ns));
grid on;
xlabel('Sample index n');
ylabel('MSE (dB)');
title('Nonstationary Noise Cancellation (Time-Varying Variance): LMS vs NLMS');
legend('Location','best');

%% ------------------------------------------------------------------------
% 11. Noise cancellation WITHOUT a reference signal
% -------------------------------------------------------------------------
N_sc  = 1000;
n_sc  = (0:N_sc-1)';

% Underlying sinusoid (narrowband component)
omega0 = 0.06*pi;                  % rad/sample
s_sc   = 2 * sin(omega0 * n_sc);   % clean sinusoid (hidden)

% Additive wideband noise
v_sc   = randn(N_sc,1);            % zero-mean white noise
x_sc   = s_sc + v_sc;              % noisy process x(n)  (no reference mic)

% NLMS noise canceller using delayed version of x_sc as "reference"
L_sc    = 12;                      % filter order
beta_sc = 0.25;                    % normalized step size β
n0      = 25;                      % delay
eps_sc  = 1e-6;

% Delayed reference signal
ref_sc = [zeros(n0,1); x_sc(1:end-n0)];
w_sc = zeros(L_sc,1);
y_sc = zeros(N_sc,1);              % NLMS output (noise estimate)
e_sc = zeros(N_sc,1);              % error = x_sc - y_sc (enhanced signal)

for n = L_sc:N_sc
    u = ref_sc(n:-1:n-L_sc+1);     % input vector from delayed x_sc
    y_sc(n) = w_sc.' * u;          % estimated noise component
    e_sc(n) = x_sc(n) - y_sc(n);   % output of noise canceller
    mu_n = beta_sc / (eps_sc + (u.'*u));
    w_sc = w_sc + mu_n * e_sc(n) * u;
end

figure;
subplot(2,1,1);
plot(n_sc, x_sc, 'k'); grid on;
ylabel('Amplitude');
title('(a) Noisy process x_{sc}[n]');
ylim([-3 3]);

subplot(2,1,2);
plot(n_sc, e_sc, 'k'); grid on;
xlabel('Sample index n');
ylabel('Amplitude');
title('(b) Output of 12th-order NLMS canceller (\beta = 0.25, n_0 = 25)');
ylim([-3 3]);

%% ------------------------------------------------------------------------
% 12. Gradient Adaptive Lattice (GAL) demo for an AR(1) process
% -------------------------------------------------------------------------
N_gal  = 3000;          % samples for clearer convergence
a_true = 0.9;           % true AR(1) coefficient
v_gal  = randn(N_gal,1);
x_gal  = filter(1, [1 -a_true], v_gal);   % AR(1) process

Gamma1      = 0;                       % initial reflection coefficient
mu_gal      = 1e-6;                    % small step for stability
Gamma_hist  = zeros(N_gal,1);
J_gal       = zeros(N_gal,1);          % instantaneous Burg cost

for n = 1:N_gal
    if n==1
        e0p = x_gal(n);   % forward error at stage 0
        e0m = 0;          % no past sample yet
    else
        e0p = x_gal(n);        % e_0^+(n)
        e0m = x_gal(n-1);      % e_0^-(n-1)
    end

    % Stage-1 forward/backward errors from lattice recursions
    e1p = e0p + Gamma1 * e0m;   % e_1^+(n)
    e1m = e0m + Gamma1 * e0p;   % e_1^-(n)

    % Instantaneous Burg cost ξ_1^B(n) ≈ |e1p|^2 + |e1m|^2
    J_gal(n) = e1p^2 + e1m^2;

    % Gradient of Burg cost wrt Γ_1 (real case)
    gradGamma = -2 * (e0p*e1m + e0m*e1p);

    % Gradient-descent update for Γ_1
    Gamma1 = Gamma1 - mu_gal * gradGamma;

    % Clipping to maintain stability: reflection coefficients ∈ (-1,1)
    Gamma1 = max(-0.999, min(0.999, Gamma1));

    Gamma_hist(n) = Gamma1;
end

% Plot convergence of Burg cost and reflection coefficient
n_axis = 1:N_gal;
figure('Position',[100 100 700 800]);

subplot(2,1,1);
plot(n_axis, 10*log10(J_gal + eps), 'b', 'LineWidth', 1.5);
grid on;
set(gca, 'FontSize', 12, 'FontName', 'Times New Roman');
xlabel('Iteration n', 'FontSize', 14, 'FontName', 'Times New Roman');
ylabel('Instantaneous Burg Cost \xi_1^B(n) (dB)', 'FontSize', 14, 'FontName', 'Times New Roman');
title('GAL Lattice: Burg Cost for AR(1) Process', 'FontSize', 14, 'FontName', 'Times New Roman');

subplot(2,1,2);
plot(n_axis, Gamma_hist, 'b', 'LineWidth', 1.5); hold on;
yline(-a_true, 'k--', 'LineWidth', 1.5);
grid on;
set(gca, 'FontSize', 12, 'FontName', 'Times New Roman');
xlabel('Iteration n', 'FontSize', 14, 'FontName', 'Times New Roman');
ylabel('Reflection Coefficient \Gamma_1(n)', 'FontSize', 14, 'FontName', 'Times New Roman');
title('GAL Lattice: Adaptation of Reflection Coefficient \Gamma_1', 'FontSize', 14, 'FontName', 'Times New Roman');
legend('{\Gamma_1(n)}','Theoretical Value ({-a}_{true})','Location','southeast', 'FontSize', 12, 'FontName', 'Times New Roman');

%% ------------------------------------------------------------------------
% 13. Robustness Test: LMS vs Sign-Error LMS under Impulsive Noise
% -------------------------------------------------------------------------
N_imp  = length(s);
L_imp  = L;
mu_imp = 0.01;  % common mu for fair comparison

% --- 13.1 Generate Impulsive Noise (Non-Gaussian) ---
impulse_prob = 0.005;
impulse_amp  = 8;
spikes = impulse_amp * (rand(N_imp, 1) < impulse_prob) .* randn(N_imp, 1);
spikes = spikes / std(spikes);

% Primary microphone signal with impulsive noise (baseline + spikes)
d_impulsive = s + noise_primary + spikes;
d_impulsive = d_impulsive - mean(d_impulsive);
d_impulsive = d_impulsive / std(d_impulsive);
x_imp = x; % Use the original reference signal

% --- 13.2 Run LMS and Sign-Error LMS ---
[~, e_lms_imp, ~] = lms_adaptive_filter(x_imp, d_impulsive, L_imp, mu_imp);
[~, e_se_imp,  ~] = sign_error_lms_filter(x_imp, d_impulsive, L_imp, mu_imp);

% --- 13.3 Performance Calculation and Plot ---
win_imp     = 200;
mse_lms_imp = filter(ones(win_imp,1)/win_imp,1,e_lms_imp.^2);
mse_se_imp  = filter(ones(win_imp,1)/win_imp,1,e_se_imp.^2);
snr_lms_imp = snr(s, e_lms_imp - s);
snr_se_imp  = snr(s, e_se_imp - s);

figure;
plot(10*log10(mse_lms_imp), 'r', 'LineWidth', 1.5, ...
    'DisplayName', sprintf('LMS (SNR: %.2f dB)', snr_lms_imp)); hold on;
plot(10*log10(mse_se_imp), 'b', 'LineWidth', 1.5, ...
    'DisplayName', sprintf('Sign-Error LMS (SNR: %.2f dB)', snr_se_imp));
grid on;
xlabel('Iteration n');
ylabel('MSE (dB)');
title('LMS vs. Sign-Error LMS under Impulsive Noise');
legend('Location', 'best');
xlim([0 N_imp]);

%% ------------------------------------------------------------------------
% 14. Effect of AR(1) Coefficient on LMS Convergence
% -------------------------------------------------------------------------
ar_coeffs = [0.2 0.5 0.9];   % weak, medium, strong correlation
mu_lms_ar = 0.01;            % fixed step size
L_ar      = L;
win_ar    = 200;

figure;
hold on;
leg_str = cell(numel(ar_coeffs),1);

for k = 1:numel(ar_coeffs)
    a1 = ar_coeffs(k);

    % Regenerate AR(1) noise with new coefficient
    ar_a_k   = [1 -a1];
    ar_exc_k = randn(N,1);
    ar_nk    = filter(1, ar_a_k, ar_exc_k);
    ar_nk    = ar_nk / std(ar_nk);

    % Use same WGN + baseline sinusoid + new AR(1)
    noise_k = sigma_wgn*wgn + sigma_sine*sin_noise_base + sigma_ar*ar_nk;

    % Primary mic: speech + this new noise
    d_k = s + noise_k;
    d_k = d_k - mean(d_k);
    d_k = d_k / std(d_k);

    % Reference channel for this noise
    x_k = filter(b_ref, 1, noise_k);
    x_k = x_k / std(x_k);

    % Run LMS with same mu for all a1
    [~, e_lms_k, ~] = lms_adaptive_filter(x_k, d_k, L_ar, mu_lms_ar);

    % Smooth MSE
    mse_k = filter(ones(win_ar,1)/win_ar, 1, e_lms_k.^2);
    plot(10*log10(mse_k),'LineWidth',1.2);

    leg_str{k} = sprintf('a = %.2f', a1);
end

yline(10*log10(xi_min_theory),'k--','Wiener MMSE');
grid on;
xlabel('Iteration n');
ylabel('Smoothed MSE (dB)');
title(sprintf('Effect of AR(1) Coefficient on LMS Convergence (\\mu = %.3f)', mu_lms_ar));
legend(leg_str,'Location','best');

%% ------------------------------------------------------------------------
% 15. Effect of Sinusoidal Interference Frequency on LMS Convergence
% -------------------------------------------------------------------------
freqs  = [200 400 800];  % Hz
mu_sin = 0.01;
L_sin  = 32;

figure;
hold on;

for k = 1:length(freqs)
    f0_k = freqs(k);
    sin_noise_k = sin(2*pi*f0_k*t);
    sin_noise_k = sin_noise_k / std(sin_noise_k);

    noise_k = sigma_wgn*wgn + sigma_ar*ar_noise + sigma_sine*sin_noise_k;

    d_k = s + noise_k;
    d_k = d_k - mean(d_k);
    d_k = d_k / std(d_k);

    x_k = filter(b_ref,1,noise_k);
    x_k = x_k/std(x_k);

    [~, e_k, ~] = lms_adaptive_filter(x_k, d_k, L_sin, mu_sin);
    mse_k = movmean(e_k.^2, 200);

    plot(10*log10(mse_k), 'LineWidth', 1.2);
end

xlabel('Iteration n');
ylabel('Smoothed MSE (dB)');
title('Effect of Sinusoidal Frequency on LMS Convergence');
legend('200 Hz','400 Hz','800 Hz','Location','best');
grid on;

%% ------------------------------------------------------------------------
% 16. Effect of WGN Power (σ) on LMS Convergence
% -------------------------------------------------------------------------
sigmas = [0.1 0.5 1.0];

figure; hold on;
for k = 1:length(sigmas)
    wgn_k = sigmas(k) * randn(N,1);

    % Keep AR noise and baseline 400 Hz sinusoid fixed
    noise_k = wgn_k + sigma_ar*ar_noise + sigma_sine*sin_noise_base;

    d_k = s + noise_k;
    d_k = d_k - mean(d_k);
    d_k = d_k / std(d_k);

    x_k = filter(b_ref,1,noise_k);
    x_k = x_k/std(x_k);

    [~, e_k, ~] = lms_adaptive_filter(x_k, d_k, L, mu_lms);
    mse_k = movmean(e_k.^2,200);

    plot(10*log10(mse_k),'LineWidth',1.2);
end

legend('\sigma=0.1','\sigma=0.5','\sigma=1.0','Location','best');
title('Effect of WGN Power on LMS Convergence');
xlabel('Iteration n'); ylabel('Smoothed MSE (dB)');
grid on;

%% ------------------------------------------------------------------------
% 17. Effect of Impulsive Noise Probability on Sign-Error LMS
% -------------------------------------------------------------------------
probs = [0.001 0.005 0.01];

figure; hold on;
for k = 1:length(probs)
    spikes_k = impulse_amp * (rand(N,1) < probs(k)) .* randn(N,1);
    spikes_k = spikes_k / std(spikes_k);

    d_k = s + noise_primary + spikes_k;
    d_k = (d_k - mean(d_k)) / std(d_k);

    [~, e_k, ~] = sign_error_lms_filter(x, d_k, L, mu_lms);
    mse_k = movmean(e_k.^2,200);

    plot(10*log10(mse_k),'LineWidth',1.2);
end

legend('p=0.001','p=0.005','p=0.01','Location','best');
title('Effect of Impulsive Noise Probability on Sign-Error LMS');
xlabel('Iteration n'); ylabel('MSE (dB)');
grid on;

%% ------------------------------------------------------------------------
% 18. Eigenvalue & Condition-Number Analysis of R_xx (Baseline Reference)
% -------------------------------------------------------------------------
% This section estimates the input correlation matrix R_xx for the baseline
% reference signal x (used in LMS/NLMS) and analyzes its eigenvalues and
% condition number. This links directly to the LMS stability bound
%   0 < mu < 2 / lambda_max(R_xx)
% and explains why convergence speed depends on input correlation.

L_eig = L;                  % use same filter length as main experiment
N_eig = length(x);
N_reg = N_eig - L_eig + 1;

% Build regression matrix X_reg whose rows are x_n^T = [x(n) ... x(n-L+1)]
X_reg = zeros(N_reg, L_eig);
for n = L_eig:N_eig
    X_reg(n-L_eig+1, :) = x(n:-1:n-L_eig+1).';
end

% Sample correlation matrix R_xx ≈ E{x_n x_n^T}
R_xx_hat = (X_reg.' * X_reg) / N_reg;

% Eigenvalue decomposition of R_xx
[Vecs, D] = eig(R_xx_hat);
lambda = diag(D);

lambda_max = max(lambda);
lambda_min = min(lambda);
cond_Rxx   = lambda_max / lambda_min;

fprintf('\n==============================================================\n');
fprintf('Eigenvalue / Condition-Number Analysis of R_xx (L = %d)\n', L_eig);
fprintf('--------------------------------------------------------------\n');
fprintf('  lambda_max(R_xx) = %.4e\n', lambda_max);
fprintf('  lambda_min(R_xx) = %.4e\n', lambda_min);
fprintf('  cond(R_xx)       = %.4f\n', cond_Rxx);
fprintf('  LMS stability bound:  0 < mu < 2 / lambda_max ≈ %.4e\n', 2/lambda_max);
fprintf('  Your chosen mu_lms   = %.4e\n', mu_lms);
fprintf('==============================================================\n\n');

% Plot eigenvalue spectrum (log scale to show spread clearly)
figure;
stem(1:L_eig, lambda, 'filled','LineWidth',1.4);
grid on;
xlabel('Eigenvalue index k');
ylabel('\lambda_k(R_{xx})');
title(sprintf('Eigenvalue Spectrum of R_{xx} (cond = %.2f)', cond_Rxx));

% Also show in dB (relative)
figure;
stem(1:L_eig, 10*log10(lambda / max(lambda)), 'filled','LineWidth',1.4);
grid on;
xlabel('Eigenvalue index k');
ylabel('Eigenvalue magnitude (dB, normalized)');
title('Relative Eigenvalue Spectrum of R_{xx}');
%% ------------------------------------------------------------------------
% 19. Tracking Ability: Time-Varying AR(1) Coefficient (LMS vs VS-LMS)
% -------------------------------------------------------------------------
% Here the AR(1) coefficient slowly changes with time:
%   a[n] = 0.4 + 0.4 * sin(2*pi*n / N)
% so the correlation structure of the noise drifts. We test how well
% fixed-step LMS and VS-LMS track this time-varying environment.

N_tv  = N;                        % use same length as baseline signal
n_tv  = (0:N_tv-1)';

% Slowly time-varying AR(1) coefficient in [0.0, 0.8]
a_tv = 0.4 + 0.4 * sin(2*pi*n_tv / N_tv);

% Generate time-varying AR(1) noise: v[n] = a[n] v[n-1] + e[n]
v_tv = zeros(N_tv,1);
e_tv = randn(N_tv,1);
for n = 2:N_tv
    v_tv(n) = a_tv(n) * v_tv(n-1) + e_tv(n);
end
v_tv = v_tv / std(v_tv);

% Build noise mixture using same WGN + baseline 400 Hz sinusoid + AR_tv
noise_tv = sigma_wgn * wgn(1:N_tv) + ...
           sigma_sine * sin_noise_base(1:N_tv) + ...
           sigma_ar   * v_tv;

% Primary mic: speech + time-varying noise
d_tv = s(1:N_tv) + noise_tv;
d_tv = d_tv - mean(d_tv);
d_tv = d_tv / std(d_tv);

% Reference channel for this noise
x_tv = filter(b_ref, 1, noise_tv);
x_tv = x_tv / std(x_tv);

% Run LMS (fixed step) and VS-LMS on this nonstationary scenario
mu_lms_tv = mu_lms;   % reuse baseline LMS step size

[~, e_lms_tv, w_hist_lms_tv] = lms_adaptive_filter(x_tv, d_tv, L, mu_lms_tv);
[~, e_vs_tv,  w_hist_vs_tv]  = vs_lms_filter(x_tv, d_tv, L, ...
                                             mu_vs_init, mu_vs_min, ...
                                             mu_vs_max, alpha_vs, beta_vs);

% Smooth MSE for visualization
win_tv       = 200;
mse_lms_tv   = movmean(e_lms_tv.^2, win_tv);
mse_vs_tv    = movmean(e_vs_tv.^2,  win_tv);

figure;
subplot(2,1,1);
plot(10*log10(mse_lms_tv),'r','LineWidth',1.4); hold on;
plot(10*log10(mse_vs_tv),'b','LineWidth',1.4);
grid on;
xlabel('Sample index n');
ylabel('Smoothed MSE (dB)');
title('Tracking under Time-Varying AR(1) Coefficient');
legend('LMS (fixed \mu)','VS-LMS (variable \mu)','Location','best');

% Compare first coefficient against time-varying AR behavior (qualitative)
subplot(2,1,2);
plot(n_tv, a_tv, 'k--','LineWidth',1.2); hold on;
plot(n_tv, w_hist_lms_tv(1, :).', 'r','LineWidth',1.2);
plot(n_tv, w_hist_vs_tv(1,  :).', 'b','LineWidth',1.2);
grid on;
xlabel('Sample index n');
ylabel('Coefficient value');
title('First Filter Tap vs Time-Varying AR(1) Coefficient');
legend('a[n] (time-varying)','LMS: w_1[n]','VS-LMS: w_1[n]','Location','best');
%% ------------------------------------------------------------------------
% 20. Sensitivity to AR(1) Coefficient Across Algorithms (SNR Comparison)
% -------------------------------------------------------------------------
% We now study how the AR(1) coefficient affects the performance of several
% algorithms (LMS, NLMS, VS-LMS) using output SNR as a scalar performance
% metric. This extends the LMS-only plot in Section 14.

ar_coeffs = [0.2 0.5 0.9];          % weak, medium, strong correlation
num_a     = numel(ar_coeffs);

alg_subnames = {'LMS','NLMS','VS-LMS'};
num_alg      = numel(alg_subnames);

SNR_out_mat  = zeros(num_a, num_alg);   % rows: a, cols: alg

% Common parameters for all runs
L_sens      = L;
mu_lms_sens = mu_lms;
mu_nlms_sens = mu_nlms;

for ia = 1:num_a
    a1 = ar_coeffs(ia);

    % --- Generate AR(1) noise with coefficient a1 ---
    ar_a_k   = [1 -a1];
    ar_exc_k = randn(N,1);
    ar_nk    = filter(1, ar_a_k, ar_exc_k);
    ar_nk    = ar_nk / std(ar_nk);

    % Use same WGN + baseline 400 Hz sinusoid + new AR(1)
    noise_k = sigma_wgn*wgn + sigma_sine*sin_noise_base + sigma_ar*ar_nk;

    % Primary mic: speech + this new noise
    d_k = s + noise_k;
    d_k = d_k - mean(d_k);
    d_k = d_k / std(d_k);

    % Reference channel
    x_k = filter(b_ref, 1, noise_k);
    x_k = x_k / std(x_k);

    % --- Algorithm 1: LMS ---
    [~, e_lms_k, ~] = lms_adaptive_filter(x_k, d_k, L_sens, mu_lms_sens);
    s_hat_lms_k = e_lms_k;
    SNR_out_mat(ia,1) = snr(s, s_hat_lms_k - s);

    % --- Algorithm 2: NLMS ---
    [~, e_nlms_k, ~] = nlms_filter(x_k, d_k, L_sens, mu_nlms_sens, 1e-6);
    s_hat_nlms_k = e_nlms_k;
    SNR_out_mat(ia,2) = snr(s, s_hat_nlms_k - s);

    % --- Algorithm 3: VS-LMS ---
    [~, e_vs_k, ~] = vs_lms_filter(x_k, d_k, L_sens, ...
                                   mu_vs_init, mu_vs_min, ...
                                   mu_vs_max, alpha_vs, beta_vs);
    s_hat_vs_k = e_vs_k;
    SNR_out_mat(ia,3) = snr(s, s_hat_vs_k - s);
end

% Print SNR table to command window
fprintf('\n==============================================================\n');
fprintf('SNR_out Sensitivity to AR(1) Coefficient (L = %d)\n', L_sens);
fprintf('--------------------------------------------------------------\n');
fprintf('      a       |    LMS (dB)   |   NLMS (dB)   |  VS-LMS (dB)\n');
fprintf('--------------------------------------------------------------\n');
for ia = 1:num_a
    fprintf('   %.2f      |    %8.2f   |    %8.2f   |    %8.2f\n', ...
            ar_coeffs(ia), ...
            SNR_out_mat(ia,1), ...
            SNR_out_mat(ia,2), ...
            SNR_out_mat(ia,3));
end
fprintf('==============================================================\n\n');

% Grouped bar plot: SNR_out vs a for different algorithms
figure;
bar(SNR_out_mat);
grid on;
set(gca,'XTick',1:num_a,'XTickLabel',arrayfun(@(a)sprintf('a=%.2f',a),ar_coeffs,'UniformOutput',false));
xlabel('AR(1) Coefficient a');
ylabel('Output SNR (dB)');
title('Sensitivity of Output SNR to AR(1) Coefficient for Different Algorithms');
legend(alg_subnames,'Location','best');

