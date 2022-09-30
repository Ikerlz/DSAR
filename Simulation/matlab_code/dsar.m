function[global_est, oneshot, onestep, twostep, se_onestep, se_twostep]=dsar(Y,W,X,N,M,init_rho,init_beta,thres,real_rho,real_beta,seed1,seed2,proj_const)
  %%%%% 全局估计 %%%%%
  % global estimate
  diff=1;
  Q_old=0;
  rho=init_rho;
  beta=init_beta;
  step=0;
  while diff>thres && step < 15
    step=step+1;
    omega=sum(W.^2);                                                         %Calculate summation of columns
    temp1=1+rho^2*omega;                                                     %Calculate denominator of d(rho)
    D_rho=sparse(1:N,1:N,1./temp1,N,N);                                      %d(rho)
    D_rho_dev1=sparse(1:N,1:N,-2*rho*omega./(temp1.^2),N,N);                 %First order derivative of d(rho)
    D_rho_dev2=sparse(1:N,1:N,(6*rho^2*(omega.^2)-2*omega)./(temp1.^3),N,N); %Second order derivative of d(rho)
    II=sparse(1:N,1:N,1,N,N);                                                %Identity matrix of dimension n                             
    W1=W+W';                                                                 %W1
    W2=W'*W;                                                                 %W2
    S=II-rho*W;                                                              %S matrix
    X_beta=X*beta';
    %temp2=rho*W1-rho^2*W2-II;
    temp2=-rho*W1+rho^2*W2+II;
    %temp3=W1-2*rho*W2;
    temp3=2*rho*W2-W1;
    D=D_rho*(temp2);                                                         %D
    F=D*Y-D_rho*S'*X_beta;                                                   %F
    D1=D_rho_dev1*(temp2)+D_rho*(temp3);                                     %To calculate M1, M2
    F_dev1_rho=D1*Y-D_rho_dev1*S'*X_beta+D_rho*W'*X_beta;
    F_dev1_beta=-D_rho*S'*X;
    F_dev1=[F_dev1_rho, F_dev1_beta];
    %D2=D_rho_dev2*(temp2)+2*D_rho_dev1*(temp3)-2*D_rho*W2;
    D2=D_rho_dev2*(temp2)+2*D_rho_dev1*(temp3)+2*D_rho*W2;
    F_dev2_rho=D2*Y-D_rho_dev2*S'*X_beta+2*D_rho_dev1*W'*X_beta;
    F_dev2_rho_beta=(D_rho*W'-D_rho_dev1*S')*X;
    Q=F'*F;                                                                  %Objective function
    Q_dev1=2*F_dev1'*F;                                                      %First order derivative of Q(rho)
    Q_dev2=zeros(length(beta)+1,length(beta)+1);
    temp4=[F_dev2_rho F_dev2_rho_beta];
    temp5=F'*temp4;
    Q_dev2(1,:)=temp5;
    Q_dev2(:,1)=temp5;
    Q_dev2=2*(Q_dev2+F_dev1'*F_dev1);
    update=(Q_dev2 \ Q_dev1)';
    if max(abs(update)) >= 5
      update = 0.01 * update;
    end
    % update params
    rho = rho - update(1);
    beta = beta - update(2:end);
    diff = abs(Q - Q_old);
    Q_old = Q;
  end
  global_est=[rho,beta];
  %%%%% 分布式算法 %%%%%

  %% 初始化储存 %%
  % parameter dimension
  p=length(init_beta)+1;
  % Split Worker Index
  woker_mat=reshape(randperm(N),M,N/M);
  % Store result
  sum_sigma2=zeros(p,p);
  sum_sigma2_mul_param=zeros(p,1);
  oneshot=zeros(1,p);                                    % store oneshot result
  % Newton-Rapson Algorithm on each worker
  for m=1:M
      Jk=sparse(N,N);
      for i=1:N/M
          Jk(woker_mat(m,i),woker_mat(m,i))=1;
      end
      Jk_Cell{m}=Jk;
      diff=1;
      Q_old=0;
      rho=init_rho;
      beta=init_beta;
      step=0;
      while diff>thres && step < 15
        step = step+1;
        omega=sum(W.^2);                                                         %Calculate summation of columns
        temp1=1+rho^2*omega;                                                     %Calculate denominator of d(rho)
        D_rho=sparse(1:N,1:N,1./temp1,N,N);                                      %d(rho)
        D_rho_dev1=sparse(1:N,1:N,-2*rho*omega./(temp1.^2),N,N);                 %First order derivative of d(rho)
        D_rho_dev2=sparse(1:N,1:N,(6*rho^2*(omega.^2)-2*omega)./(temp1.^3),N,N); %Second order derivative of d(rho)
        II=sparse(1:N,1:N,1,N,N);                                                %Identity matrix of dimension n                             
        W1=W+W';                                                                 %W1
        W2=W'*W;                                                                 %W2
        S=II-rho*W;                                                              %S matrix
        X_beta=X*beta';
        %temp2=rho*W1-rho^2*W2-II;
        temp2=-rho*W1+rho^2*W2+II;
        %temp3=W1-2*rho*W2;
        temp3=2*rho*W2-W1;
        D=D_rho*(temp2);                                                         %D
        F=D*Y-D_rho*S'*X_beta;                                                   %F
        D1=D_rho_dev1*(temp2)+D_rho*(temp3);                                     %To calculate M1, M2
        F_dev1_rho=D1*Y-D_rho_dev1*S'*X_beta+D_rho*W'*X_beta;
        F_dev1_beta=-D_rho*S'*X;
        F_dev1=[F_dev1_rho, F_dev1_beta];
        %D2=D_rho_dev2*(temp2)+2*D_rho_dev1*(temp3)-2*D_rho*W2;
        D2=D_rho_dev2*(temp2)+2*D_rho_dev1*(temp3)+2*D_rho*W2;
        F_dev2_rho=D2*Y-D_rho_dev2*S'*X_beta+2*D_rho_dev1*W'*X_beta;
        F_dev2_rho_beta=(D_rho*W'-D_rho_dev1*S')*X;
        Q=F'*F;                                                                  %Objective function
        Q_dev1=2*F_dev1'*Jk*F;                                                      %First order derivative of Q(rho)
        Q_dev2=zeros(length(beta)+1,length(beta)+1);
        temp4=[F_dev2_rho F_dev2_rho_beta];
        temp5=F'*Jk*temp4;
        Q_dev2(1,:)=temp5;
        Q_dev2(:,1)=temp5;
        Q_dev2=2*(Q_dev2+F_dev1'*Jk*F_dev1);
        update=(Q_dev2 \ Q_dev1)';
        if max(abs(update)) >= 5
          update = 0.01 * update;
        end
        % update params
        rho = rho - update(1);
        beta = beta - update(2:end);
        diff = abs(Q - Q_old);
        Q_old = Q;
      end
      params=[rho,beta];
      oneshot = oneshot + 1/M * params;
      sum_sigma2_mul_param=sum_sigma2_mul_param+Q_dev2*params';
      sum_sigma2=sum_sigma2+Q_dev2;
  end
  % get the one-step result
  onestep=sum_sigma2 \ sum_sigma2_mul_param;
  % Calculate SE
  R1_Xi_R2_cell=cell(M);
  R2_Xi_R1_cell=cell(M);
  V1_cell=cell(M);
  V2_cell=cell(M);
  T1_cell=cell(M);
  T2_cell=cell(M);
  T3_cell=cell(M);
  p = length(init_beta)+1;
  onestep_Sigma1 = zeros(p);
  rho_step1=onestep(1);
  beta_step1=reshape(onestep(2:p),p-1,1);
  tilde_sig_step1=beta_step1'*beta_step1+1;
  S=II-rho_step1*W;
  temp1=1+rho_step1^2*omega;
  D_rho_sigma1=sparse(1:N,1:N,1./(1+rho_step1^2*omega),N,N);
  D_rho_sigma1_dev1=sparse(1:N,1:N,-2*rho_step1*omega./(temp1.^2),N,N);
  for m=1:M
    % 2022.2.22 新版本
    Jk=Jk_Cell{m};
    [R1_Xik_R2, R2_Xik_R1]=Xi0(rho_step1, W, S, D_rho_sigma1, D_rho_sigma1_dev1, Jk, proj_const, seed1, seed2);
    V1k=V1(S, D_rho_sigma1, Jk, proj_const, seed1, seed2);
    V2k=V2(rho_step1, W, S, D_rho_sigma1, D_rho_sigma1_dev1, Jk, proj_const, seed1, seed2);
    T1k=T1(Y, W, S, D_rho_sigma1, Jk, proj_const, seed1);
    T2k=T2(Y, W, S, D_rho_sigma1, Jk, proj_const, seed1);
    T3k=T3(X, S, D_rho_sigma1, Jk, proj_const, seed1);
    R1_Xi_R2_cell{m}=R1_Xik_R2;
    R2_Xi_R1_cell{m}=R2_Xik_R1;
    V1_cell{m}=V1k;
    V2_cell{m}=V2k;
    T1_cell{m}=T1k;
    T2_cell{m}=T2k;
    T3_cell{m}=T3k;
  end
  % get the two-step result
  sum_sigma2_step2=zeros(p,p);
  sum_sigma2_mul_param_step2=zeros(p,1);
  for m=1:M
      Jk=sparse(N,N);
      for i=1:N/M
          Jk(woker_mat(m,i),woker_mat(m,i))=1;
      end
      rho=onestep(1);
      beta=onestep(2:end)';
      omega=sum(W.^2);                                                         %Calculate summation of columns
      temp1=1+rho^2*omega;                                                     %Calculate denominator of d(rho)
      D_rho=sparse(1:N,1:N,1./temp1,N,N);                                      %d(rho)
      D_rho_dev1=sparse(1:N,1:N,-2*rho*omega./(temp1.^2),N,N);                 %First order derivative of d(rho)
      D_rho_dev2=sparse(1:N,1:N,(6*rho^2*(omega.^2)-2*omega)./(temp1.^3),N,N); %Second order derivative of d(rho)
      II=sparse(1:N,1:N,1,N,N);                                                %Identity matrix of dimension n                             
      W1=W+W';                                                                 %W1
      W2=W'*W;                                                                 %W2
      S=II-rho*W;                                                              %S matrix
      X_beta=X*beta';
      %temp2=rho*W1-rho^2*W2-II;
      temp2=-rho*W1+rho^2*W2+II;
      %temp3=W1-2*rho*W2;
      temp3=2*rho*W2-W1;
      D=D_rho*(temp2);                                                         %D
      F=D*Y-D_rho*S'*X_beta;                                                   %F
      D1=D_rho_dev1*(temp2)+D_rho*(temp3);                                     %To calculate M1, M2
      F_dev1_rho=D1*Y-D_rho_dev1*S'*X_beta+D_rho*W'*X_beta;
      F_dev1_beta=-D_rho*S'*X;
      F_dev1=[F_dev1_rho, F_dev1_beta];
      %D2=D_rho_dev2*(temp2)+2*D_rho_dev1*(temp3)-2*D_rho*W2;
      D2=D_rho_dev2*(temp2)+2*D_rho_dev1*(temp3)+2*D_rho*W2;
      F_dev2_rho=D2*Y-D_rho_dev2*S'*X_beta+2*D_rho_dev1*W'*X_beta;
      F_dev2_rho_beta=(D_rho*W'-D_rho_dev1*S')*X;
      %Q=F'*F;                                                                  %Objective function
      Q_dev1=2*F_dev1'*Jk*F;                                                      %First order derivative of Q(rho)
      Q_dev2=zeros(length(beta)+1,length(beta)+1);
      temp4=[F_dev2_rho F_dev2_rho_beta];
      temp5=F'*Jk*temp4;
      Q_dev2(1,:)=temp5;
      Q_dev2(:,1)=temp5;
      Q_dev2=2*(Q_dev2+F_dev1'*Jk*F_dev1);
      update=(Q_dev2 \ Q_dev1)';
      if max(abs(update)) >= 5
        update = 0.01 * update;
      end
      % update params
      rho = rho - update(1);
      beta = beta - update(2:end);
      params=[rho,beta];
      sum_sigma2_mul_param_step2=sum_sigma2_mul_param_step2+Q_dev2*params';
      sum_sigma2_step2=sum_sigma2_step2+Q_dev2;
  end
  twostep=(sum_sigma2_step2 \ sum_sigma2_mul_param_step2)';

  % Calculate SE
  R1_Xi_R2_cell_step2=cell(M);
  R2_Xi_R1_cell_step2=cell(M);
  V1_cell_step2=cell(M);
  V2_cell_step2=cell(M);
  T1_cell_step2=cell(M);
  T2_cell_step2=cell(M);
  T3_cell_step2=cell(M);
  p = length(init_beta)+1;
  twostep_Sigma1 = zeros(p);
  rho_step2=twostep(1);
  beta_step2=reshape(twostep(2:p),p-1,1);
  tilde_sig_step2=beta_step2'*beta_step2+1;
  temp1_step2=1+rho_step2^2*omega;
  S_step2=II-rho_step2*W;
  D_rho_sigma1=sparse(1:N,1:N,1./(1+rho_step2^2*omega),N,N);
  D_rho_sigma1_dev1=sparse(1:N,1:N,-2*rho_step2*omega./(temp1_step2.^2),N,N);
  for m=1:M
    % 2022.2.22 新版本
    Jk=Jk_Cell{m};
    [R1_Xik_R2, R2_Xik_R1]=Xi0(rho_step2, W, S_step2, D_rho_sigma1, D_rho_sigma1_dev1, Jk, proj_const, seed1, seed2);
    V1k=V1(S_step2, D_rho_sigma1, Jk, proj_const, seed1, seed2);
    V2k=V2(rho_step2, W, S_step2, D_rho_sigma1, D_rho_sigma1_dev1, Jk, proj_const, seed1, seed2);
    T1k=T1(Y, W, S_step2, D_rho_sigma1, Jk, proj_const, seed1);
    T2k=T2(Y, W, S_step2, D_rho_sigma1, Jk, proj_const, seed1);
    T3k=T3(X, S_step2, D_rho_sigma1, Jk, proj_const, seed1);
    R1_Xi_R2_cell_step2{m}=R1_Xik_R2;
    R2_Xi_R1_cell_step2{m}=R2_Xik_R1;
    V1_cell_step2{m}=V1k;
    V2_cell_step2{m}=V2k;
    T1_cell_step2{m}=T1k;
    T2_cell_step2{m}=T2k;
    T3_cell_step2{m}=T3k;
  end

  %%% 正式计算Sigma1 %%% 

  for i=1:M
    R1_Xi_R2_i=R1_Xi_R2_cell{i};
    R2_Xi_R1_i=R2_Xi_R1_cell{i};
    V1_i=V1_cell{i};
    V2_i=V2_cell{i};
    T1_i=T1_cell{i};
    T2_i=T2_cell{i};
    T3_i=T3_cell{i};
    R1_Xi_R2_i_step2=R1_Xi_R2_cell_step2{i};
    R2_Xi_R1_i_step2=R2_Xi_R1_cell_step2{i};
    V1_i_step2=V1_cell_step2{i};
    V2_i_step2=V2_cell_step2{i};
    T1_i_step2=T1_cell_step2{i};
    T2_i_step2=T2_cell_step2{i};
    T3_i_step2=T3_cell_step2{i};
    for j=1:M
      R1_Xi_R2_j=R1_Xi_R2_cell{j};
      R2_Xi_R1_j=R2_Xi_R1_cell{j};
      V1_j=V1_cell{j};
      V2_j=V2_cell{j};
      T1_j=T1_cell{j};
      T2_j=T2_cell{j};
      T3_j=T3_cell{j};
      R1_Xi_R2_j_step2=R1_Xi_R2_cell_step2{j};
      R2_Xi_R1_j_step2=R2_Xi_R1_cell_step2{j};
      V1_j_step2=V1_cell_step2{j};
      V2_j_step2=V2_cell_step2{j};
      T1_j_step2=T1_cell_step2{j};
      T2_j_step2=T2_cell_step2{j};
      T3_j_step2=T3_cell_step2{j};
      p1_step1=4*(trace(R1_Xi_R2_i*R2_Xi_R1_j)+trace(V1_i'*V2_j)+1/tilde_sig_step1*(T1_i*T2_j'+T2_i*T1_j')+T1_i*T1_j'); 
      p1_step2=4*(trace(R1_Xi_R2_i_step2*R2_Xi_R1_j_step2)+trace(V1_i_step2'*V2_j_step2)+1/tilde_sig_step2*(T1_i_step2*T2_j_step2'+T2_i_step2*T1_j_step2')+T1_i_step2*T1_j_step2'); 
      p2_step1=-4*T1_i*T3_j';
      p2_step2=-4*T1_i_step2*T3_j_step2';
      p3_step1=4*T3_i*T3_j';
      p3_step2=4*T3_i_step2*T3_j_step2';
      onestep_Sigma1(1,1)=onestep_Sigma1(1,1)+p1_step1;
      % onestep_Sigma1(1,2:p)=onestep_Sigma1(1,2:p)+p2_step1;
      % onestep_Sigma1(2:p,1)=onestep_Sigma1(2:p,1)+p2_step1';
      onestep_Sigma1(2:p,2:p)=onestep_Sigma1(2:p,2:p)+p3_step1;
      twostep_Sigma1(1,1)=twostep_Sigma1(1,1)+p1_step2;
      % twostep_Sigma1(1,2:p)=twostep_Sigma1(1,2:p)+p2_step2;
      % twostep_Sigma1(2:p,1)=twostep_Sigma1(2:p,1)+p2_step2';
      twostep_Sigma1(2:p,2:p)=twostep_Sigma1(2:p,2:p)+p3_step2;
    end
  end
  Sigma_onestep = sum_sigma2 \ onestep_Sigma1 / sum_sigma2;
  Sigma_twostep = sum_sigma2_step2 \ twostep_Sigma1 / sum_sigma2_step2;
  se_onestep = sqrt(diag(Sigma_onestep)');
  se_twostep = sqrt(diag(Sigma_twostep)');
end


  
  