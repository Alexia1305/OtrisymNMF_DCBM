
edges = load('malariaData/malaria.edgelist');
types = load('malariaData/malaria.types');
partition = load('malariaData/malaria.partition');
N = max(max(edges));
r = edges(:,1);
c = edges(:,2);
A = sparse(r,c,ones(size(edges,1),1),N,N);
A = (A+A');
r=6;
[w_best,v_best,S_best,erreur_best,time_global,time_iteration] = OtrisymNMF_CD(A,r,'time_limit',1000,'init',"SVCA","numTrials",10,'delta',1e-10);

NMI = nmi(partition,v_best)

[w_best,v_best2,S_best,erreur_best,time_global,time_iteration] = OtrisymNMF_CD(A,r,'time_limit',1000,'init',"random","numTrials",10,'delta',1e-10);

NMI = nmi(partition,v_best2)
