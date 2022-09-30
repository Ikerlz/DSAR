NN = [1:6]*5000;
rho=0.4;
beta=[0.2,0.4,0.6,0.8,1.0];
base_save_path="/Users/lizhe/Downloads/dist_data_0811/";
if ~exist(base_save_path,'dir')
    mkdir(base_save_path)
end
for n=1:length(NN)
    N=NN(n);
	disp(N);
	[res] = generate_dist_data(N, 20, rho, beta, 1, N);
	res_name=strcat(base_save_path, "N", mat2str(N), "_", "SBM", ".mat");
    save(res_name, "res");
end