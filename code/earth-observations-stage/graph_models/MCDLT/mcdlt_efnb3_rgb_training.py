#import modules
import numpy as np
import pandas as pd
import time
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.linalg import eigh

#======================================================
# Compute the Euclidean distance between all pairs of input data points
def L2_distance_1(a, b):
    aa = np.sum(a * a, axis=0)
    bb = np.sum(b * b, axis=0)
    ab = a.T @ b
    d = aa[:, np.newaxis] + bb - 2 * ab
    return np.maximum(d, 0)

#======================================================
# Project a vector onto the probability simplex
def EProjSimplex_new(v, k=1):
    n = len(v)
    v0 = v - np.mean(v) + k/n
    vmin = np.min(v0)
    if vmin < 0:
        f = 1
        lambda_m = 0
        ft = 0
        while abs(f) > 10e-11 and ft < 100:
            v1 = v0 - lambda_m
            posidx = v1 > 0
            npos = np.sum(posidx)
            if npos>0:
                f = np.sum(v1[posidx]) - k
                lambda_m = lambda_m - f/(-npos)
            ft += 1
        return np.maximum(v1, 0)
    return v0

#======================================================
# Construct the initial nearest neighbor graph ($k$-NN)
def constructA_d(X, k, d_trans):
    n = X.shape[0]
    D = L2_distance_1(X.T, X.T)
    
    # Sort distances
    idx = np.argsort(D, axis=1)
    dumb = np.take_along_axis(D, idx, axis=1)
    
    W = np.zeros((n, n))
    rr = np.zeros(n)
    
    for i in range(n):
        di = dumb[i, 1:k+2]
        rr[i] = 0.5 * (k * di[k] - np.sum(di[:k]))
        id_nn = idx[i, 1:k+2]
        W[i, id_nn] = (di[k] - di) / (k * di[k] - np.sum(di[:k]) + 1e-12)
        
    eta = np.mean(rr)
    A_list = [None] * d_trans
    A_list[0] = (W + W.T) / 2
    
    for i in range(1, d_trans):
        A_list[i] = A_list[i-1] @ A_list[0]
        
    return A_list, eta

#======================================================
# Solves the final consensus matrix
def solveW(J_list, c, eta):
    n = J_list[0].shape[0]
    z = len(J_list)
    zr = 10e-11
    rho = 1.1
    mu = 1.0
    lambd = np.zeros((n, n))
    
    W_sum = np.sum(J_list, axis=0)
    W = (W_sum + W_sum.T) / 2
    
    L = np.diag(np.sum(W, axis=1)) - W
    evals, F = eigh(L, subset_by_index=[0, c-1])
    
    for iter in range(300):
        R = L2_distance_1(F.T, F.T)
        
        Z = W_sum + mu * W.T - 0.5 * eta * R.T + 0.5 * lambd.T - 0.5 * lambd
        
        tempW = np.zeros((n, n))
        for i in range(n):
            tempW[i, :] = EProjSimplex_new(Z[i, :] / (z + mu))
        
        W = tempW
        lambd = lambd + mu * (W - W.T)
        mu *= rho
        
        W_sym = (W + W.T) / 2
        L = np.diag(np.sum(W_sym, axis=1)) - W_sym
        
        F_old = F
        evals_all, F_full = eigh(L)
        F = F_full[:, :c]
        
        fn1 = np.sum(evals_all[:c])
        fn2 = np.sum(evals_all[:c+1])
        
        if fn1 > zr:
            eta *= 2
        elif fn2 < zr:
            eta /= 2
            F = F_old
        elif fn1 < 1e-11 and fn2 > 1e-11:
            break

    n_components, labels = connected_components(sparse.csr_matrix(W), directed=False)
    return labels, W

#======================================================
# Define the MCDLT function that coordinates the iterative process
def MCDLT(Avd, c, NITER, Z, alpha, beta, eta):
    n = Avd[0].shape[0]
    o = len(Avd)
    P = np.ones(o)
    W = np.sum(Avd, axis=0) / o
    
    y = None
    for i in range(NITER):
        selected_indices = np.where(P > 0)[0]
        J_list = [Avd[idx] for idx in selected_indices]
        
        # solveS simplificado (Proyección Simplex)
        S_list = []
        for J in J_list:
            PP = W + alpha * J
            tempS = np.array([EProjSimplex_new(PP[row, :] / (1 + alpha)) for row in range(n)])
            S_list.append(tempS)
            
        # solveW
        y, W = solveW(S_list, c, eta)
        
        # Update P (selec_min)
        Pold = P.copy()
        norm_WS = np.array([np.linalg.norm(W - A, 'fro')**2 for A in Avd])
        
        P = np.zeros(o)
        idx_min = np.argsort(norm_WS)[:Z]
        P[idx_min] = 1
        
        if np.linalg.norm(P - Pold)**2 == 0:
            break
            
    return y, W

#======================================================
# Main script
if __name__ == "__main__":
    
    #load EfficientNet-B3 embbedings from the RGB bands
    filename = '../data/x_efnb3_rgb_feats.npy'
    X = np.load(filename)
    
    #MCDLT parameters
    num_iters = 5
    num_clusters = np.arange(10, 101)
    k = 10
    o = 30
    Z = 10
    alpha = 100
    beta = 1000
    
    Ad, eta_val = constructA_d(X, k, o)

    for num_cls in num_clusters:
        print(num_cls)
        y, W_final = MCDLT(Ad, num_cls, num_iters, Z, alpha, beta, eta_val)
        
        #save clusters
        params = (num_iters, num_cls, k, o, Z, alpha, beta)
        np.save(f'results_efnb3_rgb/params_cls_{num_cls}.npy', params)
        np.save(f'results_efnb3_rgb/clusters_cls_{num_cls}.npy', y)
    
