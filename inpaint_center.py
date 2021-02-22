
import numpy as np
import scipy
from vec import vec
import matplotlib.pyplot as plt


def setup(x_shape, box_size, rho=None):


    mask = np.ones(x_shape)


    idx_row = np.round(float(x_shape[1]) / 2.0 - float(box_size) / 2.0).astype(int)
    idx_col = np.round(float(x_shape[2]) / 2.0 - float(box_size) / 2.0).astype(int)

    mask[0,idx_row:idx_row+box_size,idx_col:idx_col+box_size,:] = 0. # binary drop mask, A is a more complex variant of this and is omitted for simplicity

    #A = np.eye(x_shape[1] * x_shape[2] * x_shape[3])
    A = scipy.sparse.lil_matrix((x_shape[1] * x_shape[2] * x_shape[3], x_shape[1] * x_shape[2] * x_shape[3]), dtype=np.float32)
    A.setdiag(1.)
    A[np.array([i * (x_shape[2] * x_shape[3]) + j * x_shape[3] + k for i in range(idx_row, idx_row+box_size) for j in range(idx_col, idx_col+box_size) for k in range(x_shape[3])]), :]=0
    A = A.tocsc()
    
    if rho is not None:
        Q = scipy.sparse.hstack([A.T, np.sqrt(rho) * scipy.sparse.eye(A.shape[1], format='csc', dtype=np.float32)]).tocsc()
        Au, As, Avt = scipy.sparse.linalg.svds(A, k=400, ncv=1600)#(np.min(A.shape) - 1))
        Qu, Qs, Qvt = scipy.sparse.linalg.svds(Q, k=400, ncv=1600)#(np.min(Q.shape) - 1))
        #Au, As, Avt = np.linalg.svd(np.array(A.todense()), full_matrices=False)
        #Qu, Qs, Qvt = np.linalg.svd(np.array(Q.todense()), full_matrices=False)
        print(Qs)
        As_inv = (1. / As)
        Qs_inv = (1. / Qs)
        Av = Avt.T
        Qv = Qvt.T
        Au_T = Au.T
        Qu_T = Qu.T
        print('A', A.shape, Av.shape, As_inv.shape, Au_T.shape, 'Q', Q.shape, Qv.shape, Qs_inv.shape, Qu_T.shape)
        pinvA = np.dot(Av * As_inv[None, :], Au_T)
        if np.any(np.isnan(pinvA)) or np.any(np.isnan(Qv)) or np.any(np.isnan(Qs_inv)) or np.any(np.isnan(Qu_T)):
            raise ValueError('nan')

    def A_fun(x):
        y = np.multiply(x, mask); # elementwise product with mask
        return y

    def AT_fun(y):
        x = np.multiply(y, mask); # elementwise product with mask
        return x

    if rho is None:
        return (A_fun, AT_fun, mask, A)
    else:
        return (A_fun, AT_fun, mask, A, pinvA, Qv, Qs_inv, Qu_T, rho)


