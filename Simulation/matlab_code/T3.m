function [T3r] = T3(X, S, D, Jk, proj_const, seed1)
N = length(Jk);
d = ceil(proj_const * log(N));
T3=X'*S*D*Jk*D*S';
rng(seed1);
R1 = 1 / sqrt(d) * randn(d, N);
T3r=T3 * R1';
% test no projection
end

