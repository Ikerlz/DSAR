function [V2r] = V2(rho, W, S, D, D_dev1, Jk, proj_const, seed1, seed2)
N = length(W);
d = ceil(proj_const * log(N));
M=D_dev1*S'*S*D_dev1-D_dev1*S'*W*D-D_dev1*W'*S*D-D*W'*S*D_dev1+D*W'*W*D-D*S'*W*D_dev1;
V2=M*Jk*D*S';
rng(seed1);
R1 = 1 / sqrt(d) * randn(d, N);
rng(seed2);
R2 = 1 / sqrt(d) * randn(d, N);
V2r=R1 * V2 * R2';
end

