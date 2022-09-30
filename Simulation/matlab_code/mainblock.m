%--------------------------------------------------------------------------------------%
% Objective: Performance of LSE                                                        %
% Included function: simulator & estimate                                              %
% Author: Danyang Huang                                                                %
% Update Date: 2015.09.26                                                              %
%--------------------------------------------------------------------------------------%
 rand('state',1234);randn('state',11);                                         %Set the seed of random numbers
 sigma=1;                                                                  %True paramter of sigma
 nsimu=1000;                                                                %Number of simulations
 %NN=[2000,5000,10000,20000];                                          %Various sample sizes
 NN=[2000];  
 Rho=zeros(nsimu,length(NN));                                              %Collect rho hat estimated by Sample-LSE
 SD=zeros(nsimu,length(NN));                                               %Standard error estimated by LSE
 Den=zeros(nsimu,length(NN));                                              %Collect density of generatting network
 output=[];
 rho=0.5;                                                                  %True parameter of rho, could be set to 0 or 0.2
 K=20;
 fid=fopen('SimuResultBlock0.txt','wt');
 tic
 for nn=1:length(NN)
   N=NN(nn);
   for i=1:nsimu
       [N,i]
    [Y,A,W] = matrixAblock( N, K,rho,sigma );                                       %Generate Y, A, W accordingly
    den=mean(mean(A));                                                     %Calculate density of the network
    [rhohat,sdhat,sigmahat]=estimate(Y,W,N);                               %Estimate using LSE
    Den(i,nn)=mean(mean(A));
    Rho(i,nn)=rhohat;
    SD(i,nn)=sdhat;
   end
   den=mean(Den(:,nn));                                                    %Average density
   avgrho=mean(Rho(:,nn));                                                 %Average rhohat by Sample-LSE
   Sdhat=std(Rho(:,nn));                                                   %True SE by Sample-LSE
   Sdhats=mean(SD(:,nn));                                                  %Average SEhat by Sample-LSE
   Z=(Rho(:,nn))./SD(:,nn);                                                %Z by LSE
   erp=mean(abs(Z)>1.96);                                                  %ERP by LSE
   %temp=[rho,nn,den,avgrho,Sdhat,Sdhats,erp];
   fprintf(fid,'& %4.1f',rho); fprintf(fid,'& %5.0f',N);
   fprintf(fid,'& %6.4f',den);fprintf(fid,'& %6.3f',avgrho);
   fprintf(fid,'& %6.3f',Sdhat);fprintf(fid,'& %6.3f',Sdhats);
   fprintf(fid,'& %4.1f',erp*100);
   fprintf(fid,'\\\\ \r\n'); 
   %output=[output;temp];
 end
toc
disp(['运行时间: ',num2str(toc)]);
fclose(fid);


 
