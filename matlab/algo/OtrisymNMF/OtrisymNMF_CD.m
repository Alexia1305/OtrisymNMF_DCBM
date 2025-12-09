% Orthogonal Symmetric Nonnegative Matrix Trifactorization (OtrisymNMF)
% with a block coordinate descent approach
%
% function [w,v,S,error] = OtrisymNMF_CD(X,r,varargin)
%
% Heuristic to solve the following problem:
%   Given a symmetric nonnegative matrix X>=0, find matrices W>=0 and S>=0 
%   such that XWSW' with W'W=I,
%
% INPUTS
%   X: Symmetric nonnegative matrix (adjacency matrix of an undirected graph)
%   r: Number of columns of W (number of communities)
%
% Options (varargin)
%   numTrials: Number of trials with different initializations
%              Default: 1
%   init: Initialization method for W
%         Options: 'random', 'SSPA', 'SVCA' (default), 'SPA'
%   maxiter: Maximum number of iterations per trial
%            Default: 1000
%   delta: Convergence tolerance
%          Default: 1e-7
%          (Stop if the change in error between iterations is < delta or if error < delta)
%   time_limit: Time limit for a trial in seconds
%               Default: 60*5
%   verbosity: Display messages (1) or not (0)
%              Default: 1

%
% OUTPUTS
%
%   w_best: Vector of length n; w(i) gives the value of the non-zero element
%           in the i-th row of W
%   v_best: Vector of length n; v(i) gives the index of the column of W
%           corresponding to the non-zero element in the i-th row
%   S_best: Central matrix of size r x r 
%   error_best: Relative error ||X-WSW||_F/||X||_F
%   time_global: Total Runtime
%   time_iteration: time_iteration{t}{i} time of iteration i in trial t
%
% This code is a supplementary material to the paper
%  TOCOMPLETE 

function [w_best,v_best,S_best,error_best,time_global,time_iteration] = OtrisymNMF_CD(X,r,varargin)

if nargin <= 2
    options = [];
else
    for k = 1:2:length(varargin)
        options.(varargin{k}) = varargin{k+1};
    end
end
% Default Value 
if ~isfield(options, 'numTrials')
    options.numTrials = 1;
end
if ~isfield(options, 'time_limit')
    options.time_limit = 5*60;
end 
if ~isfield(options, 'delta')
    options.delta = 1e-5;
end 
if ~isfield(options, 'maxiter')
    options.maxiter = 1000;
end 
if ~isfield(options, 'verbosity')
    options.verbosity = 1;
end 
time_iteration = {};
start = tic;
error_best = 'inf';

if any(sum(X, 2) == 0)
    error(['The sparse matrix contains at least one zero row. ' ...
           'Please remove empty rows (nodes without connections) during preprocessing.']);
end

 if options.verbosity > 0
        fprintf('Running %u Trials in Series \n', options.numTrials);
 end
 
% PRECOMPUTATION 
n = size(X,1);
normX = norm(X, 'fro');
normX2 = normX^2;
[I,J,V] = find(X);    
[~, perm] = sort(I, 'ascend'); % permutation to sort given W rows
I = I(perm);
J = J(perm);
V = V(perm);
rowStart = [1; find(diff(I))+1; numel(I)+1];
 
 
for trials = 1:options.numTrials
    
    %INITIALISATION
    time = {};
    w = zeros(n,1);
    v = zeros(n,1);
    if isfield(options, 'init')
        init_algo = options.init;
    else
        init_algo = "SVCA";
    end 
    if init_algo == "random"
        for i = 1:n
            v(i) = randi([1,r]);
            w(i) = rand;
        end 
        
    elseif init_algo == "SSPA"
        % Estimation of WO=WS by SSPA
        p = max(2,floor(0.1*n/r));
        options1.average = 1;
        [WO,~] = SSPA(X,r,p,options1);
        % Compute W s.t. min ||X-WO*W'||_F W'W=I
        norm2x = sqrt(sum(X.^2, 1));
        Xn = X .* (1 ./ (norm2x + 1e-16));
        HO = orthNNLS(X, WO, Xn);
        W = HO';
        % Compute v and w given W
        v = max ((W ~= 0).*(1:r),[],2);
        zero_idx = find(v == 0);
        v(zero_idx) = randi(r, size(zero_idx)); 
        w = W(sub2ind(size(W), (1:n)', v));
        
    elseif init_algo == "SVCA"
        % Estimation of WO=WS by SVCA
        p = max(2, floor(0.1*n/r));
        options1.average = 1;
        [WO,~] = SVCA(X, r, p, options1);
        % Compute W s.t. min ||X-WO*W'||_F W'W=I
        norm2x = sqrt(sum(X.^2, 1));
        Xn = X .* (1 ./ (norm2x + 1e-16));
        HO = orthNNLS(X, WO, Xn);
        W = HO';
        % Compute v and w given W
        v = max((W ~= 0).*(1:r),[],2);
        zero_idx = find(v == 0);
        v(zero_idx) = randi(r, size(zero_idx)); 
        w = W(sub2ind(size(W), (1:n)', v));
                
    elseif init_algo == "SPA"
        % Estimation of WO=WS by SPA
        p = 1;
        options1.average = 1;
        [WO,~] = SSPA(X, r, p, options1);
        % Compute W s.t. min ||X-WO*W'||_F W'W=I
        norm2x = sqrt(sum(X.^2, 1));
        Xn = X .* (1 ./ (norm2x + 1e-16));
        HO = orthNNLS(X, WO, Xn);
        W = HO';
        % Compute v and w given W
        v = max((W~=0).*(1:r),[],2);
        zero_idx = find(v == 0);
        v(zero_idx) = randi(r, size(zero_idx)); 
        w = W(sub2ind(size(W), (1:n)', v));
    end
    
    % Normalization of w 
    colNorm = sqrt(accumarray(v,w.^2,[r 1]));
    w       = w./(colNorm(v)+1e-10);
    
    % Construction of S
    prodVal = w(I).*w(J).*V; %calcul des termes w_i*w_j*X(i,j)
    S       = accumarray([v(I),v(J)],prodVal,[r,r]);
    dgS     = diag(S)';

    % UPDATE error 
    error_pre = sqrt(normX2-norm(S,'fro')^2)/normX;
    error = error_pre;
    
    for itt = 1:options.maxiter
        start_it = tic;
        
        if toc(start) > options.time_limit
            disp('Time limit passed');
            break;
        end
        
        % Precomputation
        p  = zeros(1,r);
        S2 = S.^2;
        for k = 1:r
            p(k) = sum(w.^2 .* S2(v,k));
        end

        % UPDATE W
        for i = 1 : n
            % b coefficients of the r problems  min_x ax^4+bx^2+cx
            b = 2*(p-(w(i)*S(v(i),:)).^2)-2*X(i,i)*dgS; % O(r)

            % c coefficients of the r problems  min_x ax^4+bx^2+cx
            ind    = rowStart(i):rowStart(i+1)-1; % indices nonzeros X(i,:) 
            mask   = J(ind) ~= i;                 % remove i index
            cols_i = J(ind(mask));                % indices nonzeros de X(i,:) without i !
            xip    = V(ind(mask));                %  X values for non zeros entries indices X(i,:) without i !
            c      = -4 * ( (xip(:).*w(cols_i))' * S( v(cols_i) , : ) );   % O( nnz dans X(i,:) )

            % Solve r problems  min_x ax^4+bx^2+cx
            best_f = inf; best_x =sqrt(r/n) ; best_k = 1;
            for k = 1:r
                [x,f] = cardan_depressed(4*S2(k,k),2*b(k),c(k),sqrt(r/n));
                   
                if f < best_f
                    best_f = f; best_x = x; best_k = k;
                end
            end

            % Update of p before updating w(i) (O(r))
            p = p - (w(i)*S(v(i),:)).^2 + (best_x*S(best_k,:)).^2;

            % Update w(i)
            w(i) = best_x;
            v(i) = best_k;
        end

        % Normalization of w O(n)
        colNorm = sqrt(accumarray(v,w.^2,[r 1]));
        w       = w./(colNorm(v)+1e-10);

        % Construction of S O(nnz(X)) S = W^TXW when W^TW=I
        prodVal = w(I).*w(J).*V; % calcul w_i*w_j*X(i,j) O(nnz(X))
        S       = accumarray([v(I),v(J)],prodVal,[r,r]); % O(nnz(X) + r^2)
        dgS     = diag(S)';
    
        error_pre = error;
        error = sqrt(normX2-norm(S,'fro')^2)/normX;
        time{end+1} = toc(start_it);
        if error < options.delta
            break;
        end
        if abs(error_pre-error) < options.delta
            break;
        end
    end

    time_iteration{end+1} = time;
    time_global = toc(start);

    if error<=error_best
        w_best = w;
        v_best = v;
        S_best = S;
        error_best = error;
        if error_best <= options.delta
            break;
        end 
         if toc(start) > options.time_limit
            if options.verbosity > 0
                fprintf('Time_limit reached \n')
            end 
            break;
        end
    end 
    if options.verbosity > 0
        if itt == options.maxiter
                fprintf('Not converged \n')
        end 
        fprintf('Trial %u of %u with %s : %2.4e | Best: %2.4e \n',...
            trials, options.numTrials, init_algo, error, error_best);
            
    end
    
end 

