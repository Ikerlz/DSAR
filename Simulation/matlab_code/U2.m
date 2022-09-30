function [R1_U2] = U2(D, S, X, Jk, M1, proj_const, seed1)
N = length(Jk);
d = ceil(proj_const * log(N));
U2_k = - X' * S * D * Jk * M1;
rng(seed1);
R1 = 1 / sqrt(d) * randn(d, N);
R1_U2 = 2 * R1 * U2_k';
end
