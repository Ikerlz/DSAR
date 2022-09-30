%----------------------------------------------
% Description: A function to generate probability matrix A with block
% structure
% Input: N -- number of nodes
%        K -- number of blocks
% Output: A -- Adjacency matrix
% Author: Danyang Huang 
% Update Date: 27 Apr 2016
%----------------------------------------------

%设定K=20
function [ Y,A,W ] = matrixAblock( N, K, rho,sigma )
PI=ones(1,K)/K;                                                            % generate block probabiltiy
W=mnrnd(1,PI,N);                                                           % randomly assign each node to each block
P=2*ones(K)/N;                                                             % probability of a_{ij}=1 when i and j belongs to different block
P(logical(eye(K)))=20/N;                                            % probability of a_{ij}=1 when i and j belongs to the same blcok
B=W*P*W';                                                                  % probability matrix for a_{ij}=1
B=B-diag(diag(B));                                                         % probbablity for a_{ii}=1 is set to be 0
Z=1*(rand(N,N)<B);                                                         % generate Z
A=sparse(Z);
A(1,:)=0;
A(3,:)=0;
flag=find(sum(A,2)==0);
if ~isempty(flag)
    for s=1:length(flag)
        i=flag(s);
        sub=find(B(i,:)==max(B(i,:)));
        sub1=randsample(sub,1);
        A(i,sub1)=1;
    end
end
[rr,cc]=find(A);
nor=sum(A,2);
V=1./nor(rr);                                                              %normalize each row of A
W=sparse(rr,cc,V,N,N);        
epi=normrnd(0,sigma,N,1);                                                  %random episilon, normal distribution with mean 0 and sd sigma
Y=epi;                                                                     %generate Y iteratively
niter=5;
for k=1:niter
    epi=rho*W*epi;
    Y=Y+epi;
end
end




