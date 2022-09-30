function [T1r] = T1(y, W, S, D, Jk, proj_const, seed1)
N = length(W);
d = ceil(proj_const * log(N));
T1=y'*W'*S*D*Jk*D*S';
rng(seed1);
R1 = 1 / sqrt(d) * randn(d, N);
T1r=T1 * R1';
% test no projection
end

