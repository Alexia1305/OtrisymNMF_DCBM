import numpy
from scipy.io import loadmat
import OtrisymNMF
# Charger le fichier .mat
data = loadmat('matrice_2000.mat')

# Accéder à une matrice particulière
X = data['X']
r=201
w_best, v_best, S_best, error_best=OtrisymNMF.OtrisymNMF_CD(X, r, numTrials=1, maxiter=1000, delta=1e-5, time_limit=60, init_method="SVCA",update_rule="S_direct")
print(error_best)
