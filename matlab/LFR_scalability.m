%% Test OtrisymNMF on LFR benchmark graphs with different sizes n 

clear all; clc; close all;

% List of network sizes to test
n_liste = [1000,2000,5000,10000,20000,50000,100000];

% Output CSV file
output_file = 'resultats_LFRfinal.csv';
if ~isfile(output_file)
    fid = fopen(output_file,'w'); 
    fprintf(fid, 'algo,n,NMI_mean,NMI_std,Time_mean,Time_std,Iterations_mean,Iterations_std,Time_it_mean,Time_it_std\n');
    fclose(fid);
end

% List of algorithms to test
algos={'FROST','FROST_SVCA','SVCA'};

for n=n_liste

    for a=1:length(algos)

        NMI_list = zeros(1,10);      
        iter_list = zeros(1,10);   
        time_list = zeros(1,10);   
        time_it_list={};

        for g=1:10
            
            % Reading LFR network 
            community_file = sprintf('../Data/LFR_N/n_%d/community_%d.dat',n,g);
            data = load(community_file);  % colonnes : node community
            labels = data(:,2);
            r = max(labels);
            network_file=sprintf('../Data/LFR_N/n_%d/network_%d.dat',n,g);
            edges = load(network_file);
            % Adjacency matrix of the graph 
            A = sparse(edges(:,1), edges(:,2), 1, n, n);
    
            % Community detection 
            if strcmp(algos{a},'FROST')
                [w_best,v_best,S_best,erreur_best,time_global,time_iteration] = frost(A,r,'time_limit',1000,'init',"random",'verbosity',0);
                NMI_list(g) = nmi(labels,v_best);
                iter_list(g)=length(time_iteration{1});
                time_list(g)=time_global;
                time_it_list = [time_it_list,time_iteration{1}];

            elseif strcmp(algos{a},'FROST_SVCA')
                [w_best,v_best,S_best,erreur_best,time_global,time_iteration] = frost(A,r,'time_limit',1000,'init',"SVCA",'verbosity',0);
                NMI_list(g) = nmi(labels,v_best);
                iter_list(g)=length(time_iteration{1});
                time_list(g)=time_global;
                time_it_list = [time_it_list,time_iteration{1}];
            else
                [w_best,v_best,S_best,erreur_best,time_global,time_trial] = init_SVCA(A,r,'verbosity',0);      
                NMI_list(g) = nmi(labels,v_best);
                iter_list(g)=1;
                time_list(g)=time_global;
                time_it_list = [time_it_list,time_trial{1}];
            end

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
        fprintf('%s,n=%d : NMI=%.4f±%.4f, Time=%.2f±%.2f, Iter=%.2f±%.2f, Time_it=%.2f±%.2f \n',algos{a}, n, NMI_mean, NMI_std,time_global_mean,time_global_std, iter_mean, iter_std,time_it_mean,time_it_std);
    
        % Writing file CSV
        fid = fopen(output_file,'a');
        fprintf(fid,'%s,%d,%.4f,%.4f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n', algos{a},n, NMI_mean, NMI_std, time_global_mean, time_global_std, iter_mean, iter_std,time_it_mean,time_it_std);
        fclose(fid);

    end 
end




