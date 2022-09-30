function [M1, M2, R1_Xi_R2, R2_Xi_R1] = cal_Xi(real_rho, W, S, D, D_dev1, Jk, proj_const, seed1, seed2)
N = length(W);
d = ceil(proj_const * log(N));
S0 = eye(N) - real_rho * W;
SS = S' * S;
M1 = D * SS / S0;
M2 = (D_dev1 * SS - D * W' * S - D * S' * W) / S0;
% nodes in this worker
% num_k = nnz(Jk);
Xi_k = 2 * M2' * Jk * M1;
rng(seed1);
R1 = 1 / sqrt(d) * randn(d, N);
rng(seed2);
R2 = 1 / sqrt(d) * randn(d, N);
R1_Xi_R2 = R1 * Xi_k * R2';
R2_Xi_R1 = R2 * Xi_k * R1';
% test no projection
end

