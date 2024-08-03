# Kearsley's algorithm

Kearsley's algorithm calculates the rotaiton and translation required to superimposed two sets of cordinates by minimizing their RMSD. 

This is a faster implementation of the python implementation by Marcelo Moreno (martxelo) (https://github.com/martxelo/kearsley-algorithm) using numba njit.

Benchmarks by %timeit for fitting 10 atoms

Original version: 215 µs ± 27.9 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

This version: 9.62 µs ± 124 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each) 

A cpython implementation might be faster. 

# Usage

My application needed fit_tansform to also computer structure overlap, which is the percentage of atoms withing a distance cutoff.

```python3
u,v = read_from_data()
# these have to be of the numpy arrays of shape (N,3) where the N must be the same for both u and v

# so_dc is the distance cutoff used to compute structure overlap. Has to be a float. 
so_dc = 1.0

from kearsley import fit_transform

# get cordinates of v transformedt to u, RMSD, SO, and the rotation and translation matrices as numpy arrays
transformed_v, rmsd , so, rotation, translation = fit_transform(u, v, so_dc)
```
If only rmsd needs to be computed

```python3
from kearsley import fit

rmsd, q, centroid_u, centroid_v = fit(u, v)

# rotation and translation matrices can be computed by

from kearsley import fill_rot_and_trans

centroid_u_ = np.empty(3,dtype=float)
rotation = np.empty((3,3),dtype=float)

rotation,translation = fill_rot_and_trans(q,centroid_u,centroid_v,rot,centroid_u_))
```
If rotationand translation matrices are already computed transform function can transform an array of shape (M,3). Note that the M can be different to N. This function, for example, allows fitting a subset of coordinates and then applying that transformation to all coordinates.
```python3
from kearsley import fit_transform,transform

u,v = read_from_data() # shape (N,3)
w = read_all_cordinantes() # w is the superset of v. and is of the shape (M,3) M>N

transformed_v, rmsd , so, rotation, translation = fit_transform(u, v, so_dc)

# get transformed w
tra_v = transform(w,rotation,translation)
```

# Packages and version used

The following are the packages and version which this script was last tested with.
```
python = 3.12.2
numpy = 1.26.4
numba = 0.59.1
```


