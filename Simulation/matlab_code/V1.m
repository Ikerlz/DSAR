function [V1r] = V1(S, D, Jk, proj_const, seed1, seed2)
N = length(Jk);
d = ceil(proj_const * log(N));
V1=D*S'*Jk;
rng(seed1);
R1 = 1 / sqrt(d) * randn(d, N);
rng(seed2);
R2 = 1 / sqrt(d) * randn(d, N);
V1r=R1 * V1 * R2';
% test no projection
end

