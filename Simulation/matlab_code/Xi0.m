function [R1_Xi_R2, R2_Xi_R1] = Xi0(rho, W, S, D, D_dev1, Jk, proj_const, seed1, seed2)
N = length(W);
d = ceil(proj_const * log(N));
Xi=(S'*S*D_dev1-S'*W*D-W'*S*D)*Jk*D;
rng(seed1);
R1 = 1 / sqrt(d) * randn(d, N);
rng(seed2);
R2 = 1 / sqrt(d) * randn(d, N);
R1_Xi_R2=R1 * Xi * R2';
R2_Xi_R1=R2 * Xi * R1';
end

