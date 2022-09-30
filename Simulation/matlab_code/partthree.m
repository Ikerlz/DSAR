function [Sigma1_beta] = partthree(D, S, X, Jk, Jl)
temp1 = X'*S*D*Jk*D*S';
temp2 = S*D*Jl*D*S'*X;
Sigma1_beta=4*temp1*temp2;
end
