function [Sigma1_rho_beta] = partfour(D, S, y, beta, W, X, Jk, Jl)
temp = (beta'*beta)/(beta'*beta+1)*y;
Sigma1_rho_beta = -4*temp'*W'*S*D*Jk*D*S'*S*D*Jl*D*S'*X;
end
