clear all; clc; close all;
n_liste = [20000];
output_file = 'resultats_LFR.csv';

if ~isfile(output_file)
    fid = fopen(output_file,'w'); 
    fprintf(fid, 'n,NMI_mean,NMI_std,Time_mean,Time_std,Iterations_mean,Iterations_std,Time_it_mean,Time_it_std\n');
    fclose(fid);
end

for n=n_liste
    NMI_list = zeros(1,10);      
    iter_list = zeros(1,10);   
    time_list = zeros(1,10);   
    time_it_list={};
    % Reading LFR network 
    for g=1:10
        
        % Reading the community of each node
        community_file = sprintf('../Data/LFR_N/n_%d/community_%d.dat',n,g);
        data = load(community_file);  % colonnes : node community
        labels = data(:,2);
        r = max(labels);
        network_file=sprintf('../Data/LFR_N/n_%d/network_%d.dat',n,g);
        edges = load(network_file);
        %adjacency matrix
        A = sparse(edges(:,1), edges(:,2), 1, n, n);

        %Test
        [w_best,v_best,S_best,erreur_best,time_global,time_iteration] = OtrisymNMF_CD(A,r,'time_limit',1000,'init',"SVCA");
        NMI_list(g) = computeNMI(labels,v_best);
        iter_list(g)=length(time_iteration{1});
        time_list(g)=time_global;
        time_it_list = [time_it_list,time_iteration{1}];


        
        

    end
    NMI_mean = mean(NMI_list);
    NMI_std = std(NMI_list);
    iter_mean = mean(iter_list);
    iter_std = std(iter_list);
    time_global_mean=mean(time_list);
    time_global_std=std(time_list);
    time_it_mean=mean(cell2mat(time_it_list));
    time_it_std=std(cell2mat(time_it_list));
    
    % Display
    fprintf('n=%d : NMI=%.4f±%.4f, Time=%.2f±%.2f, Iter=%.2f±%.2f, Time_it=%.2f±%.2f \n', n, NMI_mean, NMI_std,time_global_mean,time_global_std, iter_mean, iter_std,time_it_mean,time_it_std);

    % Writing file CSV
    fid = fopen(output_file,'a');
    fprintf(fid, '%d,%.4f,%.4f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n', n, NMI_mean, NMI_std, time_global_mean, time_global_std, iter_mean, iter_std,time_it_mean,time_it_std);
    fclose(fid);

 
end




