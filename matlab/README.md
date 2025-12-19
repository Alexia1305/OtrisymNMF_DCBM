
# OtrisymNMF - Orthogonal Symmetric Nonnegative Matrix Trifactorization

## Overview

The **Orthogonal Symmetric Nonnegative Matrix Trifactorization** (OtrisymNMF) decomposes a symmetric nonnegative matrix $X \geq 0$ of size $n\times n$ into two matrices $Z \geq 0$ of size $n \times r$ and $S \geq 0$ of size $r\times r$, such that:

$$
X \approx Z S Z^T \quad \text{with} \quad Z^T Z = I
$$

## Folder Structure

```
- algo/
  - OtrisymNMF/
    - frost.m
    - init_SVCA.m
```

## Function: `frost`

The `frost` function solves the Orthogonal Symmetric Nonnegative Matrix Trifactorization with respect to the squared Frobenius norm using a block coordinate descent approach. It solves the following optimization problem:
$$
\min_{Z \geq 0, S \geq 0} \|\| X - Z S Z^T \|\|_F^2 \quad \text{subject to} \quad Z^T Z = I,
$$
given X and the rank r.

An example script demonstrating how to use the `FROST` function is included in the script`Exemple.m`.

## Function: `init_SVCA`

The `init_SVCA` function gives a first approximation of Z >= 0 and S >= 0 such that X ≈ ZSZ' with Z'Z=I.
$$
\min_{Z \geq 0, S \geq 0} \|\| X - Z S Z^T \|\|_F^2 \quad \text{subject to} \quad Z^T Z = I
$$


An example script demonstrating how to use the `FROST` function is included in the script`Exemple.m`.



