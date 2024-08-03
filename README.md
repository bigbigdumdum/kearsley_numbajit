# Kearsley's algorithm

Kearsley's algorithm calculates the rotaiton and translation required to superimposed two sets of cordinates by minimizing their RMSD. 

This is a faster implementation of the python implementation by Marcelo Moreno (martxelo) (https://github.com/martxelo/kearsley-algorithm) using numba njit.

Benchmarks by %timeit for fitting 10 atoms

Original version: 215 µs ± 27.9 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
This version: 9.62 µs ± 124 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each) 

A cpython implementation might be faster. 

# Usage

```
u,v = read_from_data()
so_dc = 1
# these have to be of the shape (N,3) where the N must be the same for both u and v
# so_dc is the distance cutoff used to compute structure overlap.

from kearsley import fit_transform

tra_v, rmsd , so, rot, trans = fit_transform(u, v, so_dc)

```
