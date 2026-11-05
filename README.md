
# Matrix Factorization Framework for Community Detection under the Degree-Corrected Block Model

Code for the paper **" Matrix Factorization Framework for Community Detection under the Degree-Corrected Block Model"** by Alexandra Dache, Arnaud Vandaele and Nicolas Gillis.

This repository contains:
- Python and MATLAB implementations of **FROST**, our method for community detection and our initialization method based on SVCA.
    - `otrisymNMF/`: the Python package with frost and the initialization.
    - `matlab/`: the MATLAB implementation of FROST and an example script.
- Code to reproduce the numerical experiments presented in the paper.

## FROST for community detection

The algorithm FROST aims to solve the following optimization problem:

$$
\min_{Z \geq 0, S \geq 0} ||X - ZSZ^T||_F^2 \quad \text{s.t.} \quad Z^TZ = I
$$

Where:
- **X** is a given symmetric nonnegative matrix of size $n \times n$  (e.g., adjacency matrix of an undirected graph).
- **Z** is a $n \times r$ matrix encoding the assignment of each node into **r** communities, where $Z(i,k) \neq 0$ if node $i$ belongs to community $k$.
- **S** is a $r\times r$ central matrix describing interactions between communities.

## Python 

### 🔧 Requirements

This project was developed and tested with **Python 3.12**.

Dependencies are provided for both:

- **Conda users** (`environment.yml`)
- **pip users** (`requirements.txt`)

### 📦 Installation

**Clone the repository**:

```bash
git clone https://github.com/Alexia1305/OtrisymNMF_DCBM.git
cd OtrisymNMF_DCBM
```

**Option 1 — Using Conda**:

Create the environment:
```bash
conda env create -f environment.yml
```
Activate it:
```bash
conda activate otrisymnmf-dcbm
```
**Option 2 — Using pip**

1. Create a virtual environment with Python 3.12.
```bash
python -m venv env
```
2. Activate the environment:

Windows:
```bash
env\Scripts\activate
```
Linux / macOS:
```bash
source env/bin/activate
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
### 🚀 Run Tests

#### Karate Club 

To run the `karate.ipynb` notebook:

1. Activate your virtual environment.
2. Launch Jupyter:

```bash
jupyter notebook
```

3. Open the `karate.ipynb` file from the interface and execute the cells.

#### Political blog

run the `polblogs.ipynb` notebook

To test SVCA initialization, run:
```bash
python polblogs_test_init.py
```

#### Southern women

run the `southern_women.ipynb` notebook

#### Synthetic Experiments 

Run the scripts `dcbm_synthetic.py` and  `dcbm_scalability.py`for the scalability experiment.

```bash
python ldcbm_synthetic.py
```

```bash
dcbm_scalability.py
```

#### Real graphs Experiments 
Run the script `real_datasets.py` and select the desired real graph.

```bash
python real_datasets.py
```

### Other methods 
#### graph-tool
To run graph-tool, install graph-tool see documentation https://graph-tool.skewed.de/installation.html
#### pysbm
A Python Package for Stochastic Block Model Inference by 

Funke T, Becker T (2019) Stochastic block models: A comparison of variants and inference methods. 
PLoS ONE 14(4): e0215296. https://doi.org/10.1371/journal.pone.0215296

available under the Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0). 
The original code can be found at: https://github.com/funket/pysbm.

They implement the stochastic block model variants from the following publications:

- Karrer B, Newman ME. Stochastic blockmodels and community structure in networks. Physical Review E. 2011; 83(1):016107. https://doi.org/10.1103/PhysRevE.83.016107 
- Peixoto TP. Entropy of stochastic blockmodel ensembles. Physical Review E. 2012; 85(5):056122. https://doi.org/10.1103/PhysRevE.85.056122
- Côme E, Latouche P. Model selection and clustering in stochastic block models based on the exact inte- grated complete data likelihood. Statistical Modelling. 2015; 15(6):564–589. https://doi.org/10.1177/1471082X15577017
- Newman MEJ, Reinert G. Estimating the Number of Communities in a Network. Phys Rev Lett. 2016; 117:078301. https://doi.org/10.1103/PhysRevLett.117.078301 PMID: 27564002
- Peixoto TP. Hierarchical block structures and high-resolution model selection in large networks. Physi- cal Review X. 2014; 4(1):011047. https://doi.org/10.1103/PhysRevX.4.011047
- Peixoto TP. Nonparametric Bayesian inference of the microcanonical stochastic block model. Physical
Review E. 2017; 95(1):012317. https://doi.org/10.1103/PhysRevE.95.012317 PMID: 28208453
