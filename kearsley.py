import numpy as np
import numpy.linalg as la
from numba import njit

'''Modified from https://github.com/martxelo/kearsley-algorithm/blob/main/kearsley/kearsley.py by Mukundan S'''

centroid_u_ = np.empty(3,dtype=float)
rot = np.empty((3,3),dtype=float)

@njit(cache=True,fastmath=True)
def transform(v,rot,trans):
    return apply_rot_2d(rot,v - trans)

@njit(cache=True,fastmath=True)
def fit_transform(u: np.ndarray, v: np.ndarray, so_d:float):
    centroid_u_ = np.empty(3,dtype=float)
    rot = np.empty((3,3),dtype=float)
    rmsd,q,centroid_u,centroid_v = fit(u, v) # find rotation 
    rot,trans = fill_rot_and_trans(q,centroid_u,centroid_v,rot,centroid_u_) # compute rotation and translation
    tra_v = transform(v,rot,trans) # apply transformation
    so = get_so(u,tra_v, so_d)   # compute so  
    return tra_v, rmsd , so, rot,trans  # return

@njit(cache=True,fastmath=True)
def fit_and_save(u: np.ndarray, v: np.ndarray, rot,centroid_u_):
    rmsd,q,centroid_u,centroid_v = fit(u, v) # find rotation 
    rot,trans = fill_rot_and_trans(q,centroid_u,centroid_v,rot,centroid_u_) # compute rotation and translation
    return rot,trans

@njit(cache=True,fastmath=True)
def fit( u: np.ndarray, v: np.ndarray):  
    # centroids
    centroid_u = mean_numba(u)
    centroid_v = mean_numba(v)

    # center both sets of points
    x, y = u - centroid_u, v - centroid_v

    # calculate Kearsley matrix
    K = _kearsley_matrix(x, y)

    # diagonalize K
    eig_vals, eig_vecs = la.eigh(K)

    # first eig_vec minimizes the rmsd
    q = eig_vecs[:,0]
    # q = np.roll(q, shift=3)

    # calculate rmsd
    eig_val = abs(eig_vals[0])
    rmsd = (eig_val/u.shape[0])**0.5

    return rmsd,q,centroid_u,centroid_v

@njit(cache=True,fastmath=True)
def _kearsley_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    '''
    Calculates the Kearsley matrix.
    Parameters
    ----------
    x: ndarray, shape (N, 3)
        Input array with 3D points, centered at zero.
    y: ndarray, shape (N, 3)
        Input array with 3D points, centered at zero.
    
    Returns
    ----------
    K: ndarray, shape (4, 4)
        The Kearsley matrix.
    '''
    # diff and sum quantities
    d, s = x - y, x + y

    # extract columns to simplify notation
    d0, d1, d2 = d[:,0], d[:,1], d[:,2]
    s0, s1, s2 = s[:,0], s[:,1], s[:,2]

    # fill kearsley matrix
    K = np.empty((4, 4),dtype=float)
    K[0,0] = dot(d0, d0) + dot(d1, d1) + dot(d2, d2)
    K[1,0] = dot(s1, d2) - dot(d1, s2)
    K[2,0] = dot(d0, s2) - dot(s0, d2)
    K[3,0] = dot(s0, d1) - dot(d0, s1)
    K[1,1] = dot(s1, s1) + dot(s2, s2) + dot(d0, d0)
    K[2,1] = dot(d0, d1) - dot(s0, s1)
    K[3,1] = dot(d0, d2) - dot(s0, s2)
    K[2,2] = dot(s0, s0) + dot(s2, s2) + dot(d1, d1)
    K[3,2] = dot(d1, d2) - dot(s1, s2)
    K[3,3] = dot(s0, s0) + dot(s1, s1) + dot(d2, d2)

    return K

@njit(cache=True,fastmath=True)
def centroid(a):
    res = np.empty(3,dtype=float)
    for i in range(a.shape[1]):
        res.append(a[:, i].mean())
    return np.array(res)


@njit(cache=True,fastmath=True)
def mean_numba(a):
    res = []
    for i in range(a.shape[1]):
        res.append(a[:, i].mean())
    return np.array(res)

@njit(cache=True,fastmath=True)
def dot(a,b):
    s = 0
    for i in range(a.shape[0]):
        s += a[i] * b [i]
    return s

@njit(cache=True,fastmath=True)
def eucl_dist(p1,p2):
    return ((p2[0]-p1[0])**2+(p2[1]-p1[1])**2+(p2[2]-p1[2])**2)**0.5

@njit(cache=True,fastmath=True)
def get_so(u: np.ndarray,tra_v : np.ndarray, so_d:float):
    n = 0
    t = u.shape[0]
    for i in range(t):
        if eucl_dist(u[i],tra_v[i]) < so_d:
            n+=1
    return (n/t)*100
    
@njit(cache=True,fastmath=True)
def quaternion_rotation_matrix(rot,Q):
    # Extract the values from Q
    # https://danceswithcode.net/engineeringnotes/quaternions/quaternions.html
    # q0,q1,q2,q3 = Q
    
    rot[0][0] = (Q[0]**2)+(Q[1]**2)-(Q[2]**2)-(Q[3]**2)
    rot[0][1] = (2*Q[1]*Q[2])-(2*Q[0]*Q[3])
    rot[0][2] = (2*Q[1]*Q[3])+(2*Q[0]*Q[2])

    rot[1][0] = (2*Q[1]*Q[2])+(2*Q[0]*Q[3])
    rot[1][1] = (Q[0]**2)-(Q[1]**2)+(Q[2]**2)-(Q[3]**2)
    rot[1][2] = (2*Q[2]*Q[3])-(2*Q[0]*Q[1])

    rot[2][0] = (2*Q[1]*Q[3])-(2*Q[0]*Q[2])
    rot[2][1] = (2*Q[2]*Q[3])+(2*Q[0]*Q[1])
    rot[2][2] = (Q[0]**2)-(Q[1]**2)-(Q[2]**2)+(Q[3]**2)

@njit(cache=True,fastmath=True)
def inv(rot):
    rot[0][1],rot[1][0] = rot[1][0] , rot[0][1]
    rot[0][2],rot[2][0] = rot[2][0] , rot[0][2]
    rot[1][2],rot[2][1] = rot[2][1] , rot[1][2]

@njit(cache=True,fastmath=True)
def apply_rot_2d(rot,v):
    tra_v = np.empty(v.shape,dtype=float)
    for i in range(v.shape[0]):
        apply_rot_1d(rot,v[i],tra_v[i])
    return tra_v
    # return np.array([apply_rot_1d(rot,v[i, :]) for i in range(v.shape[0])])

@njit(cache=True,fastmath=True)
def apply_rot_1d(rot,a,a_rot):    
    a_rot[0] = rot[0][0]*a[0] + rot[0][1]*a[1]+ rot[0][2]*a[2]
    a_rot[1] = rot[1][0]*a[0] + rot[1][1]*a[1]+ rot[1][2]*a[2]
    a_rot[2] = rot[2][0]*a[0] + rot[2][1]*a[1]+ rot[2][2]*a[2]


@njit(cache=True,fastmath=True)
def fill_rot_and_trans(q,centroid_u,centroid_v,rot,centroid_u_):
    quaternion_rotation_matrix(rot,q)
    rot_inv = np.copy(rot)
    inv(rot_inv)

    apply_rot_1d(rot,centroid_u,centroid_u_)

    trans = centroid_v - centroid_u_
    return rot_inv,trans