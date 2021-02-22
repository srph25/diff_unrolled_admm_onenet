
import numpy as np
import scipy
from vec import vec
import matplotlib.pyplot as plt


""" currently only support width (and height) * resize_ratio is an interger! """
def setup(x_shape, resize_ratio, rho=None):

    box_size = 1.0 / resize_ratio
    if np.mod(x_shape[1], box_size) != 0 or np.mod(x_shape[2], box_size) != 0:
        print("only support width (and height) * resize_ratio is an interger!")

    im_row = x_shape[1]
    im_col = x_shape[2]
    channel = x_shape[3]
    out_row = np.floor(float(im_row) / float(box_size)).astype(int)
    out_col = np.floor(float(im_col) / float(box_size)).astype(int)
    total_i = int(im_row / box_size)
    total_j = int(im_col / box_size)

    #A = np.zeros((out_row * out_col * x_shape[3], x_shape[1] * x_shape[2] * x_shape[3]))
    A = scipy.sparse.lil_matrix((out_row * out_col * x_shape[3], x_shape[1] * x_shape[2] * x_shape[3]), dtype=np.float32)

    for i in range(total_i):
        for j in range(total_j):
            for c in range(channel):
                for i2 in range(i*int(box_size), (i+1)*int(box_size)):
                    for j2 in range(j*int(box_size), (j+1)*int(box_size)):
                        A[i * (out_col * x_shape[3]) + j * x_shape[3] + c, i2 * (x_shape[2] * x_shape[3]) + j2 * x_shape[3] + c] = (resize_ratio ** 2)
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
        y = box_average(x, int(box_size)) # downsampling via averaging, A is a more complex variant of this and is omitted for simplicity
        return y

    def AT_fun(y):
        x = box_repeat(y, int(box_size)) # upsampling via repetition, A_T is a more complex variant of this and is omitted for simplicity
        return x

    if rho is None:
        return (A_fun, AT_fun, A)
    else:
        return (A_fun, AT_fun, A, pinvA, Qv, Qs_inv, Qu_T, rho)



def box_average(x, box_size):
    """ x: [1, row, col, channel] """
    im_row = x.shape[1]
    im_col = x.shape[2]
    channel = x.shape[3]
    out_row = np.floor(float(im_row) / float(box_size)).astype(int)
    out_col = np.floor(float(im_col) / float(box_size)).astype(int)
    y = np.zeros((1,out_row,out_col,channel))
    total_i = int(im_row / box_size)
    total_j = int(im_col / box_size)

    for c in range(channel):
        for i in range(total_i):
            for j in range(total_j):
                avg = np.average(x[0, i*int(box_size):(i+1)*int(box_size), j*int(box_size):(j+1)*int(box_size), c], axis=None)
                y[0,i,j,c] = avg

    return y


def box_repeat(x, box_size):
    """ x: [1, row, col, channel] """
    im_row = x.shape[1]
    im_col = x.shape[2]
    channel = x.shape[3]
    out_row = np.floor(float(im_row) * float(box_size)).astype(int)
    out_col = np.floor(float(im_col) * float(box_size)).astype(int)
    y = np.zeros((1,out_row,out_col,channel))
    total_i = im_row
    total_j = im_col

    for c in range(channel):
        for i in range(total_i):
            for j in range(total_j):
                y[0, i*int(box_size):(i+1)*int(box_size), j*int(box_size):(j+1)*int(box_size), c] = x[0,i,j,c]
    return y
