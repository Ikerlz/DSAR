function [Sigma1_rho_term2] = parttwo(D, S, D_dev1, W, beta, y, Jk, Jl)
M1=D*S';
temp1=trace(Jl*M1*M1'*Jk*(D_dev1*S'*(S*D_dev1-W*D)-D_dev1*W'*S*D-D*W'*S*D_dev1+D*W'*W*D-D*S'*W*D_dev1));
temp2_1=y'*(W'*S*D*Jl*M1*M1'*Jk*D*W'*S+S'*W*D*Jl*M1*M1'*Jk*D*S'*W+W'*S*D*Jl*M1*M1'*Jk*D*S'*W)*y;
temp2_2=beta'*beta+1;
% disp("=======================================================")
% disp(temp2_1);
% disp(temp2_2);
% temp2_2 = 1;
temp2=temp2_1/temp2_2;
Sigma1_rho_term2=4*(temp1+temp2);
end
