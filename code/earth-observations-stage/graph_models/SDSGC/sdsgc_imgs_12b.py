#import modules
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.linalg import eigh

#====================================================
# Define SDSGC function to obtain k clusters using spectral graph theory
def SDSGC(X, W, k, gamma, eta, local):
    NITER = 300
    num = X.shape[1]

    D = np.diag(W.sum(axis=1))
    L = D - W

    F, _, evs = eig1(L, k, isMax=0)
    if evs[:k+1].sum() < 1e-11:
        raise ValueError(f"The original graph has more than {k} connected components")

    rho = 1.3
    mu = 1.0
    lam = np.zeros((num, num))

    distX = L2_distance_1(X, X)
    Para, idx = symNeighbors(distX, k)

    evs_hist = [evs]

    for _ in range(NITER):
        # Update F
        distF = L2_distance_1(F.T, F.T)
        Z = (mu * W.T
             - 0.5 * lam + 0.5 * lam.T
             - 0.5 * distX
             - 0.5 * eta * distF)

        # Update W
        W_new = np.zeros_like(W)
        for i in range(num):
            if local == 1:
                idxa0 = Para["kn"][i]
            else:
                idxa0 = idx[i, 1:num]

            ad = Z[i, idxa0] / (gamma + mu)
            W_new[i, idxa0] = EProjSimplex_new(ad)

        W = W_new

        # Update lambda
        h = W - W.T
        lam = lam + mu * h

        # Update mu
        mu *= rho

        W = 0.5 * (W + W.T)
        D = np.diag(W.sum(axis=1))

        F_old = F.copy()
        F, _, ev = eig1(D - W, k, isMax=0)
        evs_hist.append(ev)

        fn1 = ev[:k].sum()
        fn2 = ev[:k+1].sum()
        flag = np.sum(np.abs(np.diag(D - np.eye(num))) < 0.01)

        if fn1 > 1e-11:
            eta *= 2
        elif fn2 < 1e-11:
            eta /= 2
            F = F_old
        elif fn1 < 1e-11 and fn2 > 1e-11 and flag == num:
            break

    # Connected components
    graph = sparse.csr_matrix(W)
    clusternum, labels = connected_components(graph, directed=False)

    if clusternum != k:
        print(f"Can not find the correct cluster number: {k}")

    return labels, W, np.array(evs_hist)

#====================================================
# Normalize feature vectors using L2 norm
def NormalizeFea(fea, row=True):
    fea = fea.astype(float)
    if row:
        feaNorm = np.maximum(1e-14, np.sum(fea**2, axis=1))
        fea = fea / np.sqrt(feaNorm)[:, None]
    else:
        feaNorm = np.maximum(1e-14, np.sum(fea**2, axis=0))
        fea = fea / np.sqrt(feaNorm)[None, :]
    return fea

#====================================================
# Construct an initial graph using GLAN method
def GLAN(X, k=5):
    num = X.shape[1]

    distX = L2_distance_1(X, X)
    idx = np.argsort(distX, axis=1)
    distX1 = np.take_along_axis(distX, idx, axis=1)

    A = np.zeros((num, num))
    rr = np.zeros(num)

    for i in range(num):
        di = distX1[i, 1:k+2]
        id_ = idx[i, 1:k+2]
        rr[i] = 0.5 * (k * di[k] - di[:k].sum())
        A[i, id_] = (di[k] - di) / (k * di[k] - di[:k].sum() + np.finfo(float).eps)

    r = rr.mean()
    return A, r

#====================================================
# Compute pairwise squared Euclidean distances between two sets of vectors
def L2_distance_1(a, b):
    if a.shape[0] == 1:
        a = np.vstack([a, np.zeros((1, a.shape[1]))])
        b = np.vstack([b, np.zeros((1, b.shape[1]))])

    aa = np.sum(a * a, axis=0)
    bb = np.sum(b * b, axis=0)
    ab = a.T @ b

    d = aa[:, None] + bb[None, :] - 2 * ab
    return np.maximum(np.real(d), 0)

#====================================================
# Compute selected eigenvalues and eigenvectors of a matrix
def eig1(A, c=None, isMax=True, isSym=True):
    n = A.shape[0]
    if c is None or c > n:
        c = n

    if isSym:
        A = np.maximum(A, A.T)
        d, v = eigh(A)
    else:
        d, v = np.linalg.eig(A)

    idx = np.argsort(d)
    if isMax:
        idx = idx[::-1]

    idx1 = idx[:c]
    eigval = d[idx1]
    eigvec = v[:, idx1]
    eigval_full = d[idx]

    return eigvec, eigval, eigval_full

#====================================================
# Compute symmetric k-nearest neighbor relationships
def symNeighbors(distX, k):
    num = distX.shape[1]
    idx = np.argsort(distX, axis=1)

    index = np.zeros((num, num), dtype=int)
    for i in range(num):
        index[i, idx[i, 1:k+1]] = 1

    Para = {"k": [], "kn": []}
    for i in range(num):
        m = np.where(index[:, i] == 1)[0]
        aa = idx[i, 1:k+1]
        cc = np.setdiff1d(m, aa)
        Para["k"].append(len(np.concatenate([cc, aa])))
        Para["kn"].append(np.concatenate([cc, aa]))

    return Para, idx

#====================================================
# Project a vector onto the probability simplex
def EProjSimplex_new(v, k=1):
    n = len(v)
    v0 = v - v.mean() + k / n

    if v0.min() < 0:
        lam = 0.0
        for _ in range(100):
            v1 = v0 - lam
            pos = v1 > 0
            f = v1[pos].sum() - k
            if abs(f) < 1e-10:
                break
            lam -= f / (-pos.sum())
        return np.maximum(v1, 0)
    else:
        return v0

#====================================================
# Main script
if __name__ == "__main__":
    
    #load image features
    filename = '../data/x_imgs_feats.npy'
    X = np.load(filename)
    
    #SDSGC parameters
    num_samples, size_features = np.shape(X)
    c = 100
    num_clusters = np.arange(10, 101)
    
    #normalize data
    X = NormalizeFea(X, row=True)
    
    #initial graph
    W, gamma = GLAN(X.T, c)
    eta = gamma

    for num_cls in num_clusters:
        print(f'{num_cls} - {num_clusters}')
        if( num_cls==80 ):
            continue
        
        #apply SDSGC algorithm
        runtimes = 1
        for _ in range(runtimes):
            y_cls, W, _ = SDSGC(X.T, W, num_cls, gamma, eta, local=2)

        #save clusters
        params = (runtimes, num_cls, c, num_samples, size_features)
        np.save(f'results_imgs_12b/params_cls_{num_cls}.npy', params)
        np.save(f'results_imgs_12b/clusters_cls_{num_cls}.npy', y_cls)











