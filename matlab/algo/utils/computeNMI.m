function nmi = computeNMI(P_true, P_found)
    %NMI between two partitions P_true et P_found

    if length(P_true) ~= length(P_found)
        error('P_true et P_found doivent avoir la même longueur.');
    end

     tol = 1e-6;
    if any(abs(P_true - round(P_true)) > tol)
        error('P_true non integer values !');
    end
    if any(abs(P_found - round(P_found)) > tol)
        error('P_found non integer values');
    end
    
    % number of nodes
    N = length(P_true);

    
    % labels
    clusters_true = unique(P_true);
    clusters_found = unique(P_found);
    
    
    contingency_matrix = zeros(length(clusters_true), length(clusters_found));
    
    for i = 1:length(clusters_true)
        for j = 1:length(clusters_found)
            
            contingency_matrix(i, j) = sum(P_true == clusters_true(i) & P_found == clusters_found(j));
        end
    end
    
    
    sum_true = sum(contingency_matrix, 2);  % Totaux des lignes
    sum_found = sum(contingency_matrix, 1);  % Totaux des colonnes
    
    % (MI)
    MI = 0;
    for i = 1:length(clusters_true)
        for j = 1:length(clusters_found)
            if contingency_matrix(i, j) > 0
                MI = MI + (contingency_matrix(i, j) ) * ...
                     log((N * contingency_matrix(i, j)) / (sum_true(i) * sum_found(j)));
            end
        end
    end
    
    
    H_true = -sum((sum_true) .* log(sum_true / N));
    H_found = -sum((sum_found) .* log(sum_found / N));
    
    % Normalization 
    nmi = 2 * MI / (H_true + H_found);
    if length(clusters_true)==1 && length(clusters_found)==1
        nmi=1;
    end 
end