function [Sigma1_rho_term1] = partone(D, S, D_dev1, W, Jk, Jl)
temp = D*(S'*S*D_dev1-S'*W*D-W'*S*D);
Sigma1_rho_term1=4*trace(temp*Jk*temp*Jl);
end
