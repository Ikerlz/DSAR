function [R1_U1] = U1(real_beta, beta, D, D_dev1, W, S, X, Jk, M1, M2, proj_const, seed1)
N = length(W);
d = ceil(proj_const * log(N));
X_beta0 = X * real_beta';
X_beta = X * beta';
M=M1 * X_beta0 - D * S' * X_beta;
L=M2 * X_beta0 - (D_dev1 * S' - D * W') * X_beta;
U1_k = L' * Jk * M1 + M' * Jk * M2; % 1 x N
rng(seed1);
R1 = 1 / sqrt(d) * randn(d, N);
R1_U1 = 2 * R1 * U1_k';
end

