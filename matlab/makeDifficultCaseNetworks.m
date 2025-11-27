% This code creates "difficult case" bipartite networks that were used as a
% testbed by Daniel B. Larremore, Aaron Clauset, and Abigail Z. Jacobs in
% the paper "Efficiently inferring community structure in bipartite
% networks." 
%
% More information is available at danlarremore.com/bipartiteSBM
%
% Code by Daniel Larremore, March, 2014
% larremor@hsph.harvard.edu
% danlarremore.com

%% Define all parameters

% Number of nodes
N = 1000;
% Node types
types = ones(N,1);
types(701:end) = 2;

% Correct planted communities (a.k.a. correct partition)
% NB: K_a=2, K_b=3, so K=5.
g(1:350) = 1;
g(351:700) = 2;
g(701:800) = 3;
g(801:950) = 4;
g(951:1000) = 5;
% Define a member list for each community
for j=1:5
    members{j} = find(g==j);
end

% Block structure matrix, called OMEGA PLANTED in the paper. 
s = zeros(5,5);
s(1,3) = 2500;
s(2,4) = 2500;
s(1,5) = 1500;
s(2,5) = 1500;
% Make it symmetric
s = s+s';

% Create edge affinities, called THETA in the paper. 
d = 15*floor(2*(rand(N,1)))+10;
for i=1:max(g)
    t = find(g==i);
    d(t) = d(t)/sum(d(t));
end

% Number of edges from each community, called KAPPA in the paper.
k = sum(s,2);
% Total number of edges in network.
m = sum(k)/2;

% Degree-corrected null model, or random network, called OMEGA RANDOM in
% the paper.
n = zeros(5,5);
for i=1:2
    for j=3:5
        n(i,j) = k(i)*k(j)/m;
    end
end
% make symmetric
n = n+n';

% convex combination parameter set
lambda = 0.7:0.05:1;

%% Generate networks
for l=1:length(lambda)
    % Let L be the particular value for lambda, the planted/random mixing
    % parameter
    L = lambda(l);
    % Create the convex combination of planted/random structure, called
    % OMEGA in the paper.
    mix = L*s+(1-L)*n;
    
    % Draw Poisson numbers, i.e. choose the number of edges placed in each
    % block or edge bundle between communities.
    for i=1:2
        for j=3:5
            w(i,j) = poissrnd(mix(i,j));
        end
    end
    
    % Create the edges specified in w, and assign each end of each edge to
    % a vertex in the appropriate group w.p. d, the vector of edge 
    % affinities.
    r = [];
    c = [];
    for i=1:2
        for j=3:5
            R=[];
            C=[];
            
            to = cumsum(d(members{i}));
            dice = rand(w(i,j),1);
            for t = 1:length(dice)
                R(t) = find(to>dice(t),1,'first');
            end
            r = [r;members{i}(R)'];
            
            from = cumsum(d(members{j}));
            dice = rand(w(i,j),1);
            for t = 1:length(dice)
                C(t) = find(from>dice(t),1,'first');
            end
            c = [c;members{j}(C)'];
        end
    end
    
    % Make the adjacency matrix
    A{l} = sparse(r,c,ones(length(r),1),N,N);
    
    % Make the projection
    X = A{l}*A{l}';
    P{l} = X(types==1,types==1);
    A{l} = A{l} + A{l}';
    fprintf('Network %i of %i created. lambda = %i.\n',l,length(lambda),L);
end

% The cell array A has 21 adjacency matrices, each corresponding
% to a value of lambda. The cell array P contains projections.
nt=0;
for Ai=A
    nt=test+1;
    lambda(nt)
    test = all(any(Ai{1}, 2));
    if test==0
        continue
    end
    r=5;
    [w_best,v_best,S_best,erreur_best,time_global,time_iteration] = OtrisymNMF_CD(Ai{1},r,'time_limit',1000,'init',"SVCA","numTrials",50,'delta',1e-10);

    NMI = nmi(g,v_best)

end