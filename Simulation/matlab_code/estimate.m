function[rho,sdrho,sigma2hat]=estimate(Y,W,N)
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %%%%%%%%%%%%%%%     Estimate SAR model via LSE        %%%%%%%%%%%%%%%%%%%%
 %%%%%%%%%%%%%%%       Author:  Danyang Huang          %%%%%%%%%%%%%%%%%%%%
 %%%%%%%%%%%%%%%            2015.09                    %%%%%%%%%%%%%%%%%%%%
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  rho=0;                                                                   %Initial value of rho
  temp1=sum(W.^2);                                                         %Calculate summation of columns
  temp2=1+rho^2*temp1;                                                     %Calculate denominator of d(rho)
  dd1=sparse(1:N,1:N,1./temp2,N,N);                                        %d(rho)
  dd2=sparse(1:N,1:N,-2*rho*temp1./(temp2.^2),N,N);                        %First order derivative of d(rho)
  dd3=sparse(1:N,1:N,(6*rho^2*(temp1.^2)-2*temp1)./(temp2.^3),N,N);        %Second order derivative of d(rho)
  II=sparse(1:N,1:N,1,N,N);                                                %Identity matrix of dimension n                             
  W1=W+W';                                                                 %W1
  W2=W'*W;                                                                 %W2
  D=dd1*((rho*W1-rho^2*W2)-II);                                            %D
  Dc=D*Y;                                                                  %DY 
  D1=dd2*(rho*W1-rho^2*W2-II)+dd1*(W1-2*rho*W2);                           %To calculate M1, M2
  D1c=D1*Y;                    
  D2=dd3*(rho*W1-rho^2*W2-II)+2*dd2*(W1-2*rho*W2)-2*dd1*W2;
  D2c=D2*Y;
  Q=Dc'*Dc;                                                                %Objective function
  aQ=D1c'*Dc;                                                              %First order derivative of Q(rho)
  aQ=2*aQ;
  a2Q=2*(D2c'*Dc+D1c'*D1c);                                                %Second order derivative of Q(rho)
  thres=1;
 while thres>(1e-4) 
    rho=rho-aQ/a2Q;                                                        %Update rho
    temp1=sum(W.^2);                                                       %The rest in the Loop is the same with the above
    temp2=1+rho^2*temp1;
    dd1=sparse(1:N,1:N,1./temp2,N,N);
    dd2=sparse(1:N,1:N,-2*rho*temp1./(temp2.^2),N,N);
    dd3=sparse(1:N,1:N,(6*rho^2*(temp1.^2)-2*temp1)./(temp2.^3),N,N);
    II=sparse(1:N,1:N,1,N,N);  
    W1=W+W';
    W2=W'*W;
    D=dd1*((rho*W1-rho^2*W2)-II);    
    Dc=D*Y;
    D1=dd2*(rho*W1-rho^2*W2-II)+dd1*(W1-2*rho*W2);        
    D1c=D1*Y;
    D2=dd3*(rho*W1-rho^2*W2-II)+2*dd2*(W1-2*rho*W2)-2*dd1*W2;
    D2c=D2*Y;
    QQ=Dc'*Dc;
    aQ=D1c'*Dc;
    aQ=2*aQ;
    a2Q=2*(D2c'*Dc+D1c'*D1c);
    thres=abs(QQ-Q);
    Q=QQ;                                                                  %Update the value of Objective function Q(rho)
 end
 sigma2hat=((Y-rho*W*Y)'*(Y-rho*W*Y)/N);                                   %Estimate sigma^2
 pp1=(II+rho^2*W2-rho*W1)*(dd2.*dd1);                                      %Calculate 4 parts of sigma^2_1
 p1=4*ones(1,N)*(pp1.*pp1')*ones(N,1);
 pp2=(W1-2*rho*W2)*(dd1.^2);
 p2=2*ones(1,N)*(pp2.*pp2')*ones(N,1);
 p3=-8*ones(1,N)*(pp2.*pp1')*ones(N,1);
 p4=2*Y'*(W1-2*rho*W2)*(dd1.^2)*(II+rho^2*W2-rho*W1)*(dd1.^2)*(W1-2*rho*W2)*Y;
 var1=p1+p2+p3+sigma2hat^(-1)*p4;
 varrho1=2*var1/(sigma2hat^(-2)*(a2Q^2));                                  %Estimate SE of rhohat 
 sdrho=sqrt(varrho1);
 
 

 
 