% warning off;
% NN=[2000, 4000, 10000, 20000];
% MM=[0.01, 0.02, 0.04];
%MM=[20, 40, 80];
NN=[20000];
%MM=[20, 40, 80];
MM=[40];
rep_num=500;
rho=0.4;
rho_init=0.5;
beta_init=[0.5,0.5,0.5,0.5,0.5];
beta=[0.2,0.4,0.6,0.8,1.0];
eps=1e-4;
proj_const=40;
base_save_path="/Users/lizhe/Downloads/sar_res_0715/";
if ~exist(base_save_path,'dir')
    mkdir(base_save_path)
end
for n=1:length(NN)
    N=NN(n);
    for m=1:length(MM)
        %M=N*MM(m);
        M=MM(m);
        global_mat=zeros(rep_num,length(beta)+1);
        oneshot_mat=zeros(rep_num,length(beta)+1);
        onestep_mat=zeros(rep_num,length(beta)+1);
        twostep_mat=zeros(rep_num,length(beta)+1);
        onestep_se_mat=zeros(rep_num,length(beta)+1);
        twostep_se_mat=zeros(rep_num,length(beta)+1);
        tic
        s =0;
        for i=1:rep_num
            % [Y,W,X] = SBM(N, 20, rho, beta, 1, i);
            [Y,W,X] = PowerLaw( N, 3, rho, beta, 1);
            seed1=100+i;
            seed2=1000+i;
            [global_est, oneshot, onestep, twostep, se_onestep, se_twostep]=dsar(Y,W,X,N,M,rho_init,beta_init,eps,rho,beta,seed1,seed2,proj_const);
            global_mat(i,:)=global_est;
            oneshot_mat(i,:)=oneshot;
            onestep_mat(i,:)=onestep;
            twostep_mat(i,:)=twostep;
            onestep_se_mat(i,:)=se_onestep;
            twostep_se_mat(i,:)=se_twostep;
            s=s+1;
            disp(s);
        end
        toc
        disp(["运行时间：",toc])
        % global_name=strcat(base_save_path, "N", mat2str(N), "_K", mat2str(M), "_", "SBM_", "global", ".mat");
        % oneshot_name=strcat(base_save_path, "N", mat2str(N), "_K", mat2str(M), "_", "SBM_", "oneshot", ".mat");
        % onestep_name=strcat(base_save_path, "N", mat2str(N), "_K", mat2str(M), "_", "SBM_", "onestep", ".mat");
        % twostep_name=strcat(base_save_path, "N", mat2str(N), "_K", mat2str(M), "_", "SBM_", "twostep", ".mat");
        % onestep_se_name=strcat(base_save_path, "N", mat2str(N), "_K", mat2str(M), "_", "SBM_", "onestep_se", ".mat");
        % twostep_se_name=strcat(base_save_path, "N", mat2str(N), "_K", mat2str(M), "_", "SBM_", "twostep_se", ".mat");
        global_name=strcat(base_save_path, "N", mat2str(N), "_K", mat2str(M), "_", "PowerLaw_", "global", ".mat");
        oneshot_name=strcat(base_save_path, "N", mat2str(N), "_K", mat2str(M), "_", "PowerLaw_", "oneshot", ".mat");
        onestep_name=strcat(base_save_path, "N", mat2str(N), "_K", mat2str(M), "_", "PowerLaw_", "onestep", ".mat");
        twostep_name=strcat(base_save_path, "N", mat2str(N), "_K", mat2str(M), "_", "PowerLaw_", "twostep", ".mat");
        onestep_se_name=strcat(base_save_path, "N", mat2str(N), "_K", mat2str(M), "_", "PowerLaw_", "onestep_se", ".mat");
        twostep_se_name=strcat(base_save_path, "N", mat2str(N), "_K", mat2str(M), "_", "PowerLaw_", "twostep_se", ".mat");
        save(global_name, "global_mat");
        save(oneshot_name, "oneshot_mat");
        save(onestep_name, "onestep_mat");
        save(twostep_name, "twostep_mat");
        save(onestep_se_name, "onestep_se_mat");
        save(twostep_se_name, "twostep_se_mat");
    end
end
