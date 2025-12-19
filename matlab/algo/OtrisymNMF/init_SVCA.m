% Perform community detection using SVCA (Smooth VCA).
%
% function [w,v,S,error] = init_SVCA(X,r,varargin)
%
% Gives a first approximation of  Z >= 0 and S >= 0 such that X ≈ ZSZ' with Z'Z=I.
%
% INPUTS
%  X: symmetric nonnegative matrix nxn sparse 
%  r: number of columns of Z
%
% Options (varargin)
%  numTrials 1*(default*) :number of trials with different initializations
%  verbosity :1* to display messages, 0 no display
%
% OUTPUTS
%  v: vector of lenght n, v(i) gives the index of the columns of Z not nul
%     for the i-th row
%  w : vector of lenght n, w(i) gives the value of the non zero element of
%      the i-th row
%  S: central matrix rxr 
%  error: relative error ||X-ZSZ||_F/||X||_F
%  time_global: Total Runtime
%  time_iteration: time_iteration{t}{i} time of iteration i in trial t
%

% This code is a supplementary material to the paper
%  TOCOMPLETE 

function [w_best,v_best,S_best,erreur_best,time_global,time_trial] = init_SVCA(X,r,varargin)

if nargin <= 2
    options = [];
else
    for k = 1:2:length(varargin)
        options.(varargin{k}) = varargin{k+1};
    end
end
% Default Value 
if ~isfield(options,'numTrials')
    options.numTrials = 1;
end

if ~isfield(options,'verbosity')
    options.verbosity=1;
end 

time_trial={};
start=tic;
erreur_best='inf';

 if options.verbosity > 0
        fprintf('Running %u Trials in Series \n', options.numTrials);
 end
 
% PRECOMPUTATION 
n         = size(X,1);
normX     = norm(X,'fro');
normX2    = normX^2;
[I,J,V]   = find(X);    
[~, perm] = sort(I,'ascend'); % permutation to sort given Z rows
I         = I(perm);
J         = J(perm);
V         = V(perm);

for trials =1:options.numTrials
    start_trial=tic;

    % Estimation of ZO=ZS by SVCA
    p=max(2,floor(0.1*n/r));
    options1.average=1;
    [ZO,~] = SVCA(X,r,p,options1);
    norm2x = sqrt(sum(X.^2, 1));
    Xn = X .* (1 ./ (norm2x + 1e-16));
    % Compute Z>=0 s.t. min||X-ZOZ'||_F Z'Z=I
    HO = orthNNLS(X, ZO, Xn);
    Z = HO';
    
    % Construction of v and w given Z
    v=max((Z~=0).*(1:r),[],2);
    zero_idx = find(v == 0);
    v(zero_idx) = randi(r, size(zero_idx)); 
    w = Z(sub2ind(size(Z), (1:n)', v));
        
   
    % Normalization of w 
     colNorm = sqrt(accumarray(v,w.^2,[r 1]));
    w       = w./(colNorm(v)+1e-10);
    
    % Construction of S
    prodVal = w(I).*w(J).*V; % w_i*w_j*X(i,j)
    S       = accumarray([v(I),v(J)],prodVal,[r,r]);
    
    erreur_prec= sqrt(normX2-norm(S,'fro')^2)/normX;
    erreur=erreur_prec;
    time_trial{end+1}=toc(start_trial);
    time_global=toc(start);

    if erreur<=erreur_best
        w_best=w;
        v_best=v;
        S_best=S;
        erreur_best=erreur;
    end 
    if options.verbosity > 0
    
        fprintf('Trial %u of %u : %2.4e | Best: %2.4e \n',...
            trials,options.numTrials,erreur,erreur_best);
            
    end
    
end 