import numpy as np

def cube3D(centre_3D, size):
    x = centre_3D[0]
    y = centre_3D[1]
    z = centre_3D[2]
    hs = size/2
    ret = np.array([[x+hs, x+hs, x-hs, x-hs, x+hs, x+hs, x-hs, x-hs],
                    [y+hs, y-hs, y+hs, y-hs, y+hs, y-hs, y+hs, y-hs],
                    [z+hs, z+hs, z+hs, z+hs, z-hs, z-hs, z-hs, z-hs]])
    return ret

def inv3x3(X):
    div = X[0][0]*(X[1][1]*X[2][2]-X[1][2]*X[2][1])+X[0][1]*(X[1][2]*X[2][0]-X[1][0]*X[2][2])+X[0][2]*(X[1][0]*X[2][1]-X[1][1]*X[2][0])
    ret = np.array([[X[1][1]*X[2][2]-X[1][2]*X[2][1], -(X[0][1]*X[2][2]-X[0][2]*X[2][1]), X[0][1]*X[1][2]-X[0][2]*X[1][1]],
                    [-(X[1][0]*X[2][2]-X[1][2]*X[2][0]), X[0][0]*X[2][2]-X[0][2]*X[2][0], -(X[0][0]*X[1][2]-X[0][2]*X[1][0])],
                    [X[1][0]*X[2][1]-X[1][1]*X[2][0], -(X[0][0]*X[2][1]-X[0][1]*X[2][0]), X[0][0]*X[1][1]-X[0][1]*X[1][0]]])/div
    return ret

def null(X, max=100, tol=1e-10):
    XTX = X.T@X
    R = np.eye(XTX.shape[0])
    mask = -(np.eye(XTX.shape[0])-1)
    for _ in range(max):
        abs = np.abs(mask*XTX)
        mv = np.max(abs)
        mi = np.argmax(abs)
        if mv<tol:
            break
        r = int(mi/XTX.shape[0])
        c = int(mi%XTX.shape[0])
        if XTX[r][r]-XTX[c][c]==0:
            theta = 0.25*np.pi
        else:
            theta = 0.5*np.arctan(-2.0*XTX[r,c]/(XTX[r,r]-XTX[c,c]))
        XTX_cp = XTX.copy()
        for k in range(XTX.shape[0]):
            XTX_cp[r,k] = XTX[r,k]*np.cos(theta) - XTX[c,k]*np.sin(theta)
            XTX_cp[k,r] = XTX_cp[r,k]
            XTX_cp[c,k] = XTX[r,k]*np.sin(theta) + XTX[c,k]*np.cos(theta)
            XTX_cp[k,c] = XTX_cp[c,k]
        XTX_cp[r,r] = (XTX[r,r]+XTX[c,c])/2+((XTX[r,r]-XTX[c,c])/2)*np.cos(2*theta)-XTX[r,c]*np.sin(2*theta)
        XTX_cp[c,c] = (XTX[r,r]+XTX[c,c])/2-((XTX[r,r]-XTX[c,c])/2)*np.cos(2*theta)+XTX[r,c]*np.sin(2*theta)
        XTX_cp[r,c] = 0.0
        XTX_cp[c,r] = 0.0
        XTX = XTX_cp.copy()
        G = np.identity(XTX.shape[0])
        G[r,r] = np.cos(theta)
        G[r,c] = np.sin(theta)
        G[c,r] = -np.sin(theta)
        G[c,c] = np.cos(theta)
        R = np.dot(R,G)
    index = np.argsort(np.diag(XTX))
    return R.T[index[:4]]

def p1p1(p1, p2):
    ret = np.array([p1[0]*p2[0], p1[1]*p2[1], p1[2]*p2[2],
                    p1[0]*p2[1]+p1[1]*p2[0], p1[0]*p2[2]+p1[2]*p2[0], p1[1]*p2[2]+p1[2]*p2[1],
                    p1[0]*p2[3]+p1[3]*p2[0], p1[1]*p2[3]+p1[3]*p2[1], p1[2]*p2[3]+p1[3]*p2[2],
                    p1[3]*p2[3]])
    return ret

def p2p1(p1, p2):
    ret = np.array([p1[0]*p2[0], p1[1]*p2[1], p1[2]*p2[2],
                    p1[0]*p2[1]+p1[3]*p2[0], p1[1]*p2[0]+p1[3]*p2[1], p1[0]*p2[2]+p1[4]*p2[0],
                    p1[2]*p2[0]+p1[4]*p2[2], p1[1]*p2[2]+p1[5]*p2[1], p1[2]*p2[1]+p1[5]*p2[2],
                    p1[3]*p2[2]+p1[4]*p2[1]+p1[5]*p2[0],
                    p1[0]*p2[3]+p1[6]*p2[0], p1[1]*p2[3]+p1[7]*p2[1], p1[2]*p2[3]+p1[8]*p2[2],
                    p1[3]*p2[3]+p1[6]*p2[1]+p1[7]*p2[0], p1[4]*p2[3]+p1[6]*p2[2]+p1[8]*p2[0], p1[5]*p2[3]+p1[7]*p2[2]+p1[8]*p2[1],
                    p1[6]*p2[3]+p1[9]*p2[0], p1[7]*p2[3]+p1[9]*p2[1], p1[8]*p2[3]+p1[9]*p2[2],
                    p1[9]*p2[3]])
    return ret

def gauss_forward(X):
    cp = X.copy()
    for i in range(cp.shape[0]):
        mi = i+np.argmax(np.abs(cp[i:, i]))
        order = np.arange(cp.shape[0])
        order[i] = mi
        order[mi] = i
        cp = cp[order, :]
        for j in range(cp.shape[0]-i-1):
            div = cp[i+j+1][i]/cp[i][i]
            cp[i+j+1, :] = cp[i+j+1, :] - div*cp[i, :]
    return cp

def gauss_pp(X):
    U = gauss_forward(X)
    B = np.zeros((10, 20))
    B[:4, :] = U[:4, :]
    B[9, :] = U[9, :]/U[9][9]
    B[8, :] = (U[8, :]-U[8][9]*B[9, :])/U[8][8]
    B[7, :] = (U[7, :]-U[7][8]*B[8, :]-U[7][9]*B[9, :])/U[7][7]
    B[6, :] = (U[6, :]-U[6][7]*B[7, :]-U[6][8]*B[8, :]-U[6][9]*B[9, :])/U[6][6]
    B[5, :] = (U[5, :]-U[5][6]*B[6, :]-U[5][7]*B[7, :]-U[5][8]*B[8, :]-U[5][9]*B[9, :])/U[5][5]
    B[4, :] = (U[4, :]-U[4][5]*B[5, :]-U[4][6]*B[6, :]-U[4][7]*B[7, :]-U[4][8]*B[8, :]-U[4][9]*B[9, :])/U[4][4]
    return B

def partial_sub(p1, p2):
    ret = np.array([-p2[0], p1[0]-p2[1], p1[1]-p2[2], p1[2], -p2[3], p1[3]-p2[4],
                    p1[4]-p2[5], p1[5], -p2[6], p1[6]-p2[7], p1[7]-p2[8], p1[8]-p2[9], p1[9]])
    return ret

def pz4pz3(p1, p2):
    ret = np.array([p1[0]*p2[0], p1[1]*p2[0]+p1[0]*p2[1], p1[0]*p2[2]+p1[1]*p2[1]+p1[2]*p2[0], p1[0]*p2[3]+p1[1]*p2[2]+p1[2]*p2[1]+p1[3]*p2[0],
                    p1[1]*p2[3]+p1[2]*p2[2]+p1[3]*p2[1]+p1[4]*p2[0], p1[2]*p2[3]+p1[3]*p2[2]+p1[4]*p2[1], p1[3]*p2[3]+p1[4]*p2[2], p1[4]*p2[3]])
    return ret

def pz3pz3(p1, p2):
    ret = np.array([p1[0]*p2[0], p1[0]*p2[1]+p1[1]*p2[0], p1[2]*p2[0]+p1[1]*p2[1]+p1[0]*p2[2],
                    p1[0]*p2[3]+p1[1]*p2[2]+p1[2]*p2[1]+p1[3]*p2[0],
                    p1[1]*p2[3]+p1[2]*p2[2]+p1[3]*p2[1], p1[2]*p2[3]+p1[3]*p2[2], p1[3]*p2[3]])
    return ret

def pz7pz3(p1, p2):
    ret = np.array([p1[0]*p2[0], p1[1]*p2[0]+p1[0]*p2[1], p1[2]*p2[0]+p1[1]*p2[1]+p1[0]*p2[2],
                    p1[3]*p2[0]+p1[2]*p2[1]+p1[1]*p2[2]+p1[0]*p2[3],
                    p1[1]*p2[3]+p1[2]*p2[2]+p1[3]*p2[1]+p1[4]*p2[0],
                    p1[2]*p2[3]+p1[3]*p2[2]+p1[4]*p2[1]+p1[5]*p2[0],
                    p1[3]*p2[3]+p1[4]*p2[2]+p1[5]*p2[1]+p1[6]*p2[0],
                    p1[4]*p2[3]+p1[5]*p2[2]+p1[6]*p2[1]+p1[7]*p2[0],
                    p1[5]*p2[3]+p1[6]*p2[2]+p1[7]*p2[1],
                    p1[6]*p2[3]+p1[7]*p2[2],
                    p1[7]*p2[3]])
    return ret

def pz6pz4(p1, p2):
    ret = np.array([p1[0]*p2[0], p1[1]*p2[0]+p1[0]*p2[1], p1[0]*p2[2]+p1[1]*p2[1]+p1[2]*p2[0],
                    p1[0]*p2[3]+p1[1]*p2[2]+p1[2]*p2[1]+p1[3]*p2[0],
                    p1[0]*p2[4]+p1[1]*p2[3]+p1[2]*p2[2]+p1[3]*p2[1]+p1[4]*p2[0],
                    p1[1]*p2[4]+p1[2]*p2[3]+p1[3]*p2[2]+p1[4]*p2[1]+p1[5]*p2[0],
                    p1[2]*p2[4]+p1[3]*p2[3]+p1[4]*p2[2]+p1[5]*p2[1]+p1[6]*p2[0],
                    p1[3]*p2[4]+p1[4]*p2[3]+p1[5]*p2[2]+p1[6]*p2[1],
                    p1[4]*p2[4]+p1[5]*p2[3]+p1[6]*p2[2], p1[5]*p2[4]+p1[6]*p2[3], p1[6]*p2[4]])
    return ret

def solvePoly(coef):
    temp1 = coef.copy()
    temp2 = np.hstack([np.eye(coef.shape[0]-1), np.zeros((coef.shape[0]-1, 1))])
    temp = np.vstack([-temp1, temp2])
    val, _ = np.linalg.eig(temp)
    return val

def five_point_algorithm(pts1, pts2, K1, K2):
    q1 = inv3x3(K1)@np.vstack([pts1, np.ones((1, 5))])
    q2 = inv3x3(K2)@np.vstack([pts2, np.ones((1, 5))])
    q = np.hstack([(q1[0, :]*q2[0, :]).reshape((5, 1)), (q1[1, :]*q2[0, :]).reshape((5, 1)), (q1[2, :]*q2[0, :]).reshape((5, 1)), (q1[0, :]*q2[1, :]).reshape((5, 1)), (q1[1, :]*q2[1, :]).reshape((5, 1)), (q1[2, :]*q2[1, :]).reshape((5, 1)), (q1[0, :]*q2[2, :]).reshape((5, 1)), (q1[1, :]*q2[2, :]).reshape((5, 1)), (q1[2, :]*q2[2, :]).reshape((5, 1))])
    #null_q = null(q)
    _, _, V = np.linalg.svd(q)
    null_q = V[5:]
    X = null_q[0].reshape((3, 3))
    Y = null_q[1].reshape((3, 3))
    Z = null_q[2].reshape((3, 3))
    W = null_q[3].reshape((3, 3))
    X_ = inv3x3(K2.T)@X@inv3x3(K1)
    Y_ = inv3x3(K2.T)@Y@inv3x3(K1)
    Z_ = inv3x3(K2.T)@Z@inv3x3(K1)
    W_ = inv3x3(K2.T)@W@inv3x3(K1)
    detF = p2p1(p1p1([X_[0][1], Y_[0][1], Z_[0][1], W_[0][1]], [X_[1][2], Y_[1][2], Z_[1][2], W_[1][2]])
                -p1p1([X_[0][2], Y_[0][2], Z_[0][2], W_[0][2]], [X_[1][1], Y_[1][1], Z_[1][1], W_[1][1]]),
                [X_[2][0], Y_[2][0], Z_[2][0], W_[2][0]])\
          +p2p1(p1p1([X_[0][2], Y_[0][2], Z_[0][2], W_[0][2]], [X_[1][0], Y_[1][0], Z_[1][0], W_[1][0]])
                -p1p1([X_[0][0], Y_[0][0], Z_[0][0], W_[0][0]], [X_[1][2], Y_[1][2], Z_[1][2], W_[1][2]]),
                [X_[2][1], Y_[2][1], Z_[2][1], W_[2][1]])\
          +p2p1(p1p1([X_[0][0], Y_[0][0], Z_[0][0], W_[0][0]], [X_[1][1], Y_[1][1], Z_[1][1], W_[1][1]])
                -p1p1([X_[0][1], Y_[0][1], Z_[0][1], W_[0][1]], [X_[1][0], Y_[1][0], Z_[1][0], W_[1][0]]),
                [X_[2][2], Y_[2][2], Z_[2][2], W_[2][2]])
    EEt11 = p1p1([X[0][0], Y[0][0], Z[0][0], W[0][0]], [X[0][0], Y[0][0], Z[0][0], W[0][0]])\
           +p1p1([X[0][1], Y[0][1], Z[0][1], W[0][1]], [X[0][1], Y[0][1], Z[0][1], W[0][1]])\
           +p1p1([X[0][2], Y[0][2], Z[0][2], W[0][2]], [X[0][2], Y[0][2], Z[0][2], W[0][2]])
    EEt12 = p1p1([X[0][0], Y[0][0], Z[0][0], W[0][0]], [X[1][0], Y[1][0], Z[1][0], W[1][0]])\
           +p1p1([X[0][1], Y[0][1], Z[0][1], W[0][1]], [X[1][1], Y[1][1], Z[1][1], W[1][1]])\
           +p1p1([X[0][2], Y[0][2], Z[0][2], W[0][2]], [X[1][2], Y[1][2], Z[1][2], W[1][2]])
    EEt13 = p1p1([X[0][0], Y[0][0], Z[0][0], W[0][0]], [X[2][0], Y[2][0], Z[2][0], W[2][0]])\
           +p1p1([X[0][1], Y[0][1], Z[0][1], W[0][1]], [X[2][1], Y[2][1], Z[2][1], W[2][1]])\
           +p1p1([X[0][2], Y[0][2], Z[0][2], W[0][2]], [X[2][2], Y[2][2], Z[2][2], W[2][2]])
    EEt22 = p1p1([X[1][0], Y[1][0], Z[1][0], W[1][0]], [X[1][0], Y[1][0], Z[1][0], W[1][0]])\
           +p1p1([X[1][1], Y[1][1], Z[1][1], W[1][1]], [X[1][1], Y[1][1], Z[1][1], W[1][1]])\
           +p1p1([X[1][2], Y[1][2], Z[1][2], W[1][2]], [X[1][2], Y[1][2], Z[1][2], W[1][2]])
    EEt23 = p1p1([X[1][0], Y[1][0], Z[1][0], W[1][0]], [X[2][0], Y[2][0], Z[2][0], W[2][0]])\
           +p1p1([X[1][1], Y[1][1], Z[1][1], W[1][1]], [X[2][1], Y[2][1], Z[2][1], W[2][1]])\
           +p1p1([X[1][2], Y[1][2], Z[1][2], W[1][2]], [X[2][2], Y[2][2], Z[2][2], W[2][2]])
    EEt33 = p1p1([X[2][0], Y[2][0], Z[2][0], W[2][0]], [X[2][0], Y[2][0], Z[2][0], W[2][0]])\
           +p1p1([X[2][1], Y[2][1], Z[2][1], W[2][1]], [X[2][1], Y[2][1], Z[2][1], W[2][1]])\
           +p1p1([X[2][2], Y[2][2], Z[2][2], W[2][2]], [X[2][2], Y[2][2], Z[2][2], W[2][2]])
    A11 = EEt11-0.5*(EEt11+EEt22+EEt33)
    A12 = EEt12
    A13 = EEt13
    A21 = A12
    A22 = EEt22-0.5*(EEt11+EEt22+EEt33)
    A23 = EEt23
    A31 = A13
    A32 = A23
    A33 = EEt33-0.5*(EEt11+EEt22+EEt33)
    AE11 = p2p1(A11, [X[0][0], Y[0][0], Z[0][0], W[0][0]])\
          +p2p1(A12, [X[1][0], Y[1][0], Z[1][0], W[1][0]])\
          +p2p1(A13, [X[2][0], Y[2][0], Z[2][0], W[2][0]])
    AE12 = p2p1(A11, [X[0][1], Y[0][1], Z[0][1], W[0][1]])\
          +p2p1(A12, [X[1][1], Y[1][1], Z[1][1], W[1][1]])\
          +p2p1(A13, [X[2][1], Y[2][1], Z[2][1], W[2][1]])
    AE13 = p2p1(A11, [X[0][2], Y[0][2], Z[0][2], W[0][2]])\
          +p2p1(A12, [X[1][2], Y[1][2], Z[1][2], W[1][2]])\
          +p2p1(A13, [X[2][2], Y[2][2], Z[2][2], W[2][2]])
    AE21 = p2p1(A21, [X[0][0], Y[0][0], Z[0][0], W[0][0]])\
          +p2p1(A22, [X[1][0], Y[1][0], Z[1][0], W[1][0]])\
          +p2p1(A23, [X[2][0], Y[2][0], Z[2][0], W[2][0]])
    AE22 = p2p1(A21, [X[0][1], Y[0][1], Z[0][1], W[0][1]])\
          +p2p1(A22, [X[1][1], Y[1][1], Z[1][1], W[1][1]])\
          +p2p1(A23, [X[2][1], Y[2][1], Z[2][1], W[2][1]])
    AE23 = p2p1(A21, [X[0][2], Y[0][2], Z[0][2], W[0][2]])\
          +p2p1(A22, [X[1][2], Y[1][2], Z[1][2], W[1][2]])\
          +p2p1(A23, [X[2][2], Y[2][2], Z[2][2], W[2][2]])
    AE31 = p2p1(A31, [X[0][0], Y[0][0], Z[0][0], W[0][0]])\
          +p2p1(A32, [X[1][0], Y[1][0], Z[1][0], W[1][0]])\
          +p2p1(A33, [X[2][0], Y[2][0], Z[2][0], W[2][0]])
    AE32 = p2p1(A31, [X[0][1], Y[0][1], Z[0][1], W[0][1]])\
          +p2p1(A32, [X[1][1], Y[1][1], Z[1][1], W[1][1]])\
          +p2p1(A33, [X[2][1], Y[2][1], Z[2][1], W[2][1]])
    AE33 = p2p1(A31, [X[0][2], Y[0][2], Z[0][2], W[0][2]])\
          +p2p1(A32, [X[1][2], Y[1][2], Z[1][2], W[1][2]])\
          +p2p1(A33, [X[2][2], Y[2][2], Z[2][2], W[2][2]])
    A = np.vstack([detF, AE11, AE12, AE13, AE21, AE22, AE23, AE31, AE32, AE33])
    A = A[:, [0, 1, 3, 4, 5, 10, 7, 11, 9, 13, 6, 14, 16, 8, 15, 17, 2, 12, 18, 19]]
    A_gauss = gauss_pp(A)
    k_row = partial_sub(A_gauss[4, 10:20], A_gauss[5, 10:20])
    l_row = partial_sub(A_gauss[6, 10:20], A_gauss[7, 10:20])
    m_row = partial_sub(A_gauss[8, 10:20], A_gauss[9, 10:20])
    B11 = k_row[:4]
    B12 = k_row[4:8]
    B13 = k_row[8:13]
    B21 = l_row[:4]
    B22 = l_row[4:8]
    B23 = l_row[8:13]
    B31 = m_row[:4]
    B32 = m_row[4:8]
    B33 = m_row[8:13]
    p1 = pz4pz3(B23, B12)-pz4pz3(B13, B22)
    p2 = pz4pz3(B13, B21)-pz4pz3(B23, B11)
    p3 = pz3pz3(B11, B22)-pz3pz3(B12, B21)
    n_row = pz7pz3(p1, B31)+pz7pz3(p2, B32)+pz6pz4(p3, B33)
    n_row_scaled = n_row/n_row[0]
    e_val = solvePoly(n_row_scaled[1:])
    z_list = []
    for i in e_val:
        if np.abs(i.imag) < 1e-5:
            z_list.append(i.real)
    E_list = []
    for z in z_list:
        pz6 = np.array([z**6, z**5, z**4, z**3, z**2, z, 1])
        pz7 = np.concatenate([[z**7], pz6])
        x = np.sum(p1*pz7)/np.sum(p3*pz6)
        y = np.sum(p2*pz7)/np.sum(p3*pz6)
        E = x*X+y*Y+z*Z+W
        U, _, V = np.linalg.svd(E)
        E = U@np.diag([1, 1, 0])@V
        E_list.append(E)
    return E_list