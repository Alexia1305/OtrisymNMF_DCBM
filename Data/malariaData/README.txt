You have downloaded the "malaria" data set that was used by Larremore, Clauset, and Jacobs in the paper "Efficiently inferring community structure in bipartite networks."

http://danlarremore.com/bipartiteSBM
larremor@hsph.harvard.edu

// FILE LIST

There are 4 files:
	1. malaria.edgelist - a tab-separated list of edges in the malaria network, in the form: i j w. In this case, all weights w are equal to 1.
	2. malaria.types - a list of the types of all vertices in the malaria network, which is bipartite and comprises genes (type 1) and substrings (type 2).
	3. malaria.partition - the partition shown in Figures 6 and 7 of the paper.
	4. malaria.mat - A MATLAB file that contains:
		A - the adjacency matrix
		B - the bipartite adjacency matrix
		N_a, N_b - the numbers of genes and substrings, respectively.
		P_a, P_b - both weighted one-mode projections.
		geneSequences - the genes themselves, at the amino acid level. See note below.
		geneSequenceHeaders - the names of the genes
		substrings - the substrings that were extracted from the sequences
		g - the partition shown in Figures 6 and 7 of the paper.

// A NOTE ABOUT SEQUENCE DATA

These sequences were initially published by Thomas S. Rask, et al. but were analyzed using more traditional genetic techniques:

	Rask, T. S., Hansen, D. A., Theander, T. G., Gorm Pedersen, A., & Lavstsen, T. (2010). Plasmodium falciparum Erythrocyte Membrane Protein 1 Diversity in Seven Genomes – Divide and Conquer. PLoS Computational Biology, 6(9), e1000933. doi:10.1371/journal.pcbi.1000933

The same sequences were reanalyzed using complex networks in their Highly Variable Regions by Daniel B. Larremore, Aaron Clauset, and Caroline O. Buckee. The sequence data provided here correspond to HVR6 of their paper.

	Larremore, D. B., Clauset, A., & Buckee, C. O. (2013). A Network Approach to Analyzing Highly Recombinant Malaria Parasite Genes. PLoS Computational Biology, 9(10), e1003268. doi:10.1371/journal.pcbi.1003268.s010