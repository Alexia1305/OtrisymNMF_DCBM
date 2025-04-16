# Degree-corrected block model : une nouvelle approche et une initialisation efficace pour l’inférence

Ce code **Python** contient les tests réalisés pour un article soumis au **GRETSI 25**.  
Nous utilisons **OtrisymNMF** comme **degree corrected block model (DCBM)** pour détecter des communautés dans plusieurs réseaux de référence, y compris le benchmark LFR.  
Nous montrons également que notre initialisation, basée sur la NMF séparable, améliore significativement les résultats des méthodes d’inférence classiques pour le **DCBM de Karrer et Newman**.

Tous les outils pour **OtrisymNMF** sont disponibles dans le package Python **OtrisymNMF**.  
Pour le DCBM de Karrer et Newman et les méthodes d’inférence associées, nous avons utilisé le package **pysbm**.  
Le notebook **Karate** compare l’utilisation d’**OtrisymNMF** et du DCBM de Karrer et Newman sur le **réseau Karate Club**.  
Les autres fichiers contiennent des expériences montrant que l’initialisation **SVCA** améliore significativement les résultats des méthodes d’inférence.


# Reproduire les résultats

## 🔧 Prérequis

- **Python 3.9** (⚠️ Non compatible avec Python 3.10+ pour l’affichage des graphes)
- Il est recommandé d’utiliser un environnement virtuel.

## 📦 Installation

1. **Cloner le dépôt** :

```bash
git clone https://github.com/Alexia1305/DCBM_OtrisymNMF.git
cd DCBM_OtrisymNMF
```

2. **Créer et activer l'environnement virtuel**:

```bash
py -3.9 -m venv env
env\Scripts\activate        # on Windows
```

3. **Installer les dépendances**:

```bash
pip install -r requirements.txt
```


## 🚀 Lancer les tests

### Karate Club (Figure 3)

Pour exécuter le notebook `Karate.ipynb`:

1. Activer l'environnement virtuel.
2. Lancer Jupyter :

```bash
jupyter notebook
```

3. Ouvrez le fichier `Karate.ipynb` depuis l'interface et exécutez les cellules.

### Tests sur les graphes LFR

Lancez le script et sélectionnez la valeur souhaitée pour $mu.

```bash
python LFR_benchmark.py
```
### Test sur le Scotland Corporate Interlock Network
```bash
python Scotland_test.py
```

# OtrisymNMF
Ce package contient l'implémentation de l'algorithme **Orthogonal Symmetric Nonnegative Matrix Tri-Factorization** (OtrisymNMF) tel que proposé dans l'article:

**Dache, Alexandra, Arnaud Vandaele, and Nicolas Gillis.**  
*"Orthogonal Symmetric Nonnegative Matrix Tri-Factorization."*  
IEEE International Workshop on Machine Learning for Signal Processing (MLSP), 2024.  
Institute of Electrical and Electronics Engineers (IEEE), United States.

L’algorithme vise à résoudre le problème d’optimisation suivant :

\[
\min_{W \geq 0, S \geq 0} \|X - WSW^T\|_F^2 \quad \text{s.t.} \quad W^TW = I
\]

Où :
- **X** est une matrice symétrique non négative donnée (par exemple, une matrice d’adjacence d'un graphe non orienté).
- **W** est une matrice représentant l’affectation des éléments à **r** communautés.
- **S** est une matrice centrale décrivant les interactions entre les communautés.

Le package **OtrisymNMF** inclut également l’algorithme **SVCA** pour initialiser l’inférence.


# pysbm
Un package Python pour l’inférence de **Stochastic Block Models (SBM)** développé par :

Funke T, Becker T (2019) Stochastic block models: A comparison of variants and inference methods. 
PLoS ONE 14(4): e0215296. https://doi.org/10.1371/journal.pone.0215296

Disponible sous licence **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)**.  
Le code original est disponible ici : https://github.com/funket/pysbm.

Ils implémentent les variantes des stochastic block models issues des publications suivantes :

- Karrer B, Newman ME. Stochastic blockmodels and community structure in networks. Physical Review E. 2011; 83(1):016107. https://doi.org/10.1103/PhysRevE.83.016107 
- Peixoto TP. Entropy of stochastic blockmodel ensembles. Physical Review E. 2012; 85(5):056122. https://doi.org/10.1103/PhysRevE.85.056122
- Côme E, Latouche P. Model selection and clustering in stochastic block models based on the exact inte- grated complete data likelihood. Statistical Modelling. 2015; 15(6):564–589. https://doi.org/10.1177/1471082X15577017
- Newman MEJ, Reinert G. Estimating the Number of Communities in a Network. Phys Rev Lett. 2016; 117:078301. https://doi.org/10.1103/PhysRevLett.117.078301 PMID: 27564002
- Peixoto TP. Hierarchical block structures and high-resolution model selection in large networks. Physi- cal Review X. 2014; 4(1):011047. https://doi.org/10.1103/PhysRevX.4.011047
- Peixoto TP. Nonparametric Bayesian inference of the microcanonical stochastic block model. Physical
Review E. 2017; 95(1):012317. https://doi.org/10.1103/PhysRevE.95.012317 PMID: 28208453
