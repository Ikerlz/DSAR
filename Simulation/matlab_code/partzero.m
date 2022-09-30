function [Sigma1_rho_term0] = partzero(D, S, y, beta, W, Jk, Jl)
temp = (beta'*beta)/(beta'*beta+1)*y;
Sigma1_rho_term0=4*temp'*W'*S*D*Jk*D*S'*S*D*Jl*D*S'*W*temp;
end
