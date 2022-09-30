%----------------------------------------------
% Description: A function to generate probability matrix A with Power law model
% structure
% Input: N -- number of nodes
%        alpha -- parameter
% Output: A -- Adjacency matrix
% Author: Zhe Li
% Update Date: 6 Dec 2021
%----------------------------------------------

%设定alpha=2
function [ Y,W,X ] = PowerLaw( N, alpha, rho, beta, sigma )
%alpha=2;
index_list = 1:N;
probs = zeros(1,N);
for k=1:N
    probs(k)=k^(-alpha);
end
probs = probs / sum(probs);
A=sparse(N,N);
for n=1:N
    degree=randsrc(1,1,[index_list; probs]);
    A(n,randsample(N,degree))=1;
% row_index=[];
% col_index=[];
% for n=1:N
%     degree=randsrc(1,1,[index_list; probs]);
%     row_index=[row_index, repelem(n,degree)];
%     col_index=[col_index, randsample(N,degree)'];
%A=sparse(row_index,col_index,1,N,N);
[rr,cc]=find(A);
nor=sum(A,2);
V=1./nor(rr);                                                              %normalize each row of A
W=sparse(rr,cc,V,N,N);        
epi0=normrnd(0,sigma,N,1);                                                  %random episilon, normal distribution with mean 0 and sd sigma
X=normrnd(0,sigma,N,length(beta));
epi=epi0+X*beta';
Y=epi;                                                                     %generate Y iteratively
niter=5;
for k=1:niter
    epi=rho*W*epi;
    Y=Y+epi;
end
end