import numpy as np
from fivepoint import five_point_algorithm

def det(X):
    return X[0][0]*(X[1][1]*X[2][2]-X[1][2]*X[2][1])+X[0][1]*(X[1][2]*X[2][0]-X[1][0]*X[2][2])+X[0][2]*(X[1][0]*X[2][1]-X[1][1]*X[2][0])

def error(X1, X2, E):
    n = X1.shape[1]
    theta = E.reshape((9))
    ones = np.ones_like(X1[0])
    epsilon = np.vstack([X1[0]*X2[0], X1[1]*X2[0], X2[0], X1[0]*X2[1], X1[1]*X2[1], X2[1], X1[0], X1[1], ones])
    Er = 0
    for i in range(n):
        Er += (np.sum((epsilon[:, i]*theta))**2)
    return Er/n

def sampson(X1, X2, E):
    n = X1.shape[1]
    theta = E.reshape((9))
    ones = np.ones_like(X1[0])
    zeros = np.zeros_like(X1[0])
    epsilon = np.vstack([X1[0]*X2[0], X1[1]*X2[0], X2[0], X1[0]*X2[1], X1[1]*X2[1], X2[1], X1[0], X1[1], ones])
    V = np.array([[X1[0]**2+X2[0]**2, X2[0]*X2[1], X2[0], X1[0]*X1[1], zeros, zeros, X1[0], zeros, zeros],
                  [X2[0]*X2[1], X1[0]**2+X2[1]**2, X2[1], zeros, X1[0]*X1[1], zeros, zeros, X1[0], zeros],
                  [X2[0], X2[1], ones, zeros, zeros, zeros, zeros, zeros, zeros],
                  [X1[0]*X1[1], zeros, zeros, X1[1]**2+X2[0]**2, X2[0]*X2[1], X2[0], X1[1], zeros, zeros],
                  [zeros, X1[0]*X1[1], zeros, X2[0]*X2[1], X1[1]**2+X2[1]**2, X2[1], zeros, X1[1], zeros],
                  [zeros, zeros, zeros, X2[0], X2[1], ones, zeros, zeros, zeros],
                  [X1[0], zeros, zeros, X1[1], zeros, zeros, ones, zeros, zeros],
                  [zeros, X1[0], zeros, zeros, X1[1], zeros, zeros, ones, zeros],
                  [zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros]])
    J = 0
    for i in range(n):
        J += (np.sum((epsilon[:, i]*theta))**2)/np.sum(theta*(V[:, :, i]@theta))
    return J/n

def select_E(kpts1, kpts2, E_list):
    minEr = float('inf')
    minIndex = -1
    for i in range(len(E_list)):
        Er = error(kpts1, kpts2, E_list[i]/E_list[i][2][2])
        if Er<minEr:
            minEr = Er
            minIndex = i
    return E_list[minIndex]/E_list[minIndex][2][2], minEr

def triangulate(kpts1, kpts2, P1, P2):
    x1 = kpts1[0]
    y1 = kpts1[1]
    x2 = kpts2[0]
    y2 = kpts2[1]
    A = np.array([[x1*P1[2][0]-P1[0][0], x1*P1[2][1]-P1[0][1], x1*P1[2][2]-P1[0][2], x1*P1[2][3]-P1[0][3]],
                  [y1*P1[2][0]-P1[1][0], y1*P1[2][1]-P1[1][1], y1*P1[2][2]-P1[1][2], y1*P1[2][3]-P1[1][3]],
                  [x2*P2[2][0]-P2[0][0], x2*P2[2][1]-P2[0][1], x2*P2[2][2]-P2[0][2], x2*P2[2][3]-P2[0][3]],
                  [y2*P2[2][0]-P2[1][0], y2*P2[2][1]-P2[1][1], y2*P2[2][2]-P2[1][2], y2*P2[2][3]-P2[1][3]]])
    _, _, V = np.linalg.svd(A)
    pos = V[3, :]
    return pos[:3]/pos[3]

def get_motion(E, K1, K2, kpts1, kpts2, index):
    U, _, V = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    T = U[:, 2]
    R1 = det(U@W@V)*(U@W@V)
    R2 = det(U@(W.T)@V)*(U@(W.T)@V)
    T_list = [T, T, -T, -T]
    R_list = [R1, R2, R1, R2]
    best_T = None
    best_R = None
    max_count = 0
    for T_, R_ in zip(T_list, R_list):
        P1 = K1@np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = K2@np.hstack([R_, T_.reshape(3, 1)])
        count = 0
        for i in range(len(index)):
            pos = triangulate(kpts1[:2, index[i]], kpts2[:2, index[i]], P1, P2)
            kpts1_r = P1@np.concatenate([pos, [1]])
            kpts2_r = P2@np.concatenate([pos, [1]])
            if (kpts1_r[2]>0)&(kpts2_r[2]>0):
                count += 1
        if max_count<count:
            max_count = count
            best_T = T_
            best_R = R_
    return best_R, best_T

def estimate_motion(kpts1, kpts2, K1, K2, max_iter=100):
    best_E = None
    min_Er = float('inf')
    index_inlier = None
    for i in range(max_iter):
        index = np.random.choice(kpts1.shape[1], 5)
        E_list = five_point_algorithm(kpts1[:2, index], kpts2[:2, index], K1, K2)
        if len(E_list)==0:
            continue
        E, Er = select_E(kpts1, kpts2, E_list)
        if Er<min_Er:
            best_E = E
            min_Er = Er
            index_inlier = index
    R, T = get_motion(best_E, K1, K2, kpts1, kpts2, index_inlier)
    return R, T
