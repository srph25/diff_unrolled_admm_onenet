
import numpy as np
import scipy
from vec import vec
import matplotlib.pyplot as plt



def setup(x_shape, compress_ratio, rho=None):

    d = np.prod(x_shape).astype(int)
    m = np.round(compress_ratio * d).astype(int)

    A = np.random.randn(m,d) / np.sqrt(m) # A is overcomplete random gaussian
    A = A.astype(np.float32)

    if rho is not None:
        Q = np.hstack([A.T, np.sqrt(rho) * np.eye(A.shape[1], dtype=np.float32)])
        Au, As, Avt = scipy.sparse.linalg.svds(A, k=400, ncv=1600)#(np.min(A.shape) - 1))
        Qu, Qs, Qvt = scipy.sparse.linalg.svds(Q, k=400, ncv=1600)#(np.min(Q.shape) - 1))
        #Au, As, Avt = np.linalg.svd(A, full_matrices=False)
        #Qu, Qs, Qvt = np.linalg.svd(Q, full_matrices=False)
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
        y = np.dot(A, x.ravel(order='F')) # dot product with A
        y = np.reshape(y, [1, m], order='F')
        return y

    def AT_fun(y):
        y = np.reshape(y, [m, 1], order='F')
        x = np.dot(A.T, y) # dot product with A_T
        x = np.reshape(x, x_shape, order='F')
        return x
    
    if rho is None:
        return (A_fun, AT_fun, A)
    else:
        return (A_fun, AT_fun, A, pinvA, Qv, Qs_inv, Qu_T, rho)
