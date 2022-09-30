%warning off;
NN=10000;
%MM=[0.01, 0.02, 0.04];
MM=[20, 40, 80];
rep_num=500;
rho=0.4;
rho_init=0;
beta_init=[0.5,0.5,0.5,0.5,0.5];
beta=[0.2,0.4,0.6,0.8,1.0];
eps=1e-4;
tic
[a,b,c] = PowerLaw( NN, 2, rho, beta, 1);
toc