import numpy as np
from estimate_motion import estimate_motion

def generate_pts(K1, K2, baseline, n_point=50, min_point=10, add_noise=False, sigma=1e-3):
    size=0
    while size<min_point:
        theta = np.deg2rad((np.random.uniform()-0.5)*360)
        axis = np.random.uniform(size=(3))
        axis = axis/np.sqrt(np.sum(axis**2))
        v = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        R = np.eye(3)+np.sin(theta)*v+(1-np.cos(theta))*(v@v)
        t = np.random.uniform(size=(3))-0.5
        t_cross = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
        E_gt = t_cross@R
        E_gt = E_gt/E_gt[2][2]
        P1_L = K1@np.hstack([np.eye(3), np.zeros((3, 1))])
        P2_L = K2@np.hstack([R, t.reshape((3, 1))])
        P1_R = K1@np.hstack([np.eye(3), np.zeros((3, 1))])
        P2_R = K2@np.hstack([R, t.reshape((3, 1))])
        X = np.vstack([np.random.uniform(size=(3, n_point))-0.5, np.ones((1, n_point))])
        X1_L = P1_L@X
        X2_L = P2_L@X
        X1_R = P1_R@(X-np.vstack([np.eye(3).T@np.array([[baseline], [0], [0]]), 0]))
        X2_R = P2_R@(X-np.vstack([R.T@np.array([[baseline], [0], [0]]), 0]))
        mask = (X1_L[2, :]>0)&(X2_L[2, :]>0)&(X1_R[2, :]>0)&(X2_R[2, :]>0)
        size = np.sum(mask)
    X1_L = X1_L[:, mask]/X1_L[2, mask]
    X2_L = X2_L[:, mask]/X2_L[2, mask]
    X1_R = X1_R[:, mask]/X1_R[2, mask]
    X2_R = X2_R[:, mask]/X2_R[2, mask]
    if add_noise==True:
        X1L_noise = np.vstack([np.random.normal(size=(X1_L.shape[1])), np.random.normal(size=(X1_L.shape[1])), np.zeros(X1_L.shape[1])])
        X2L_noise = np.vstack([np.random.normal(size=(X1_L.shape[1])), np.random.normal(size=(X1_L.shape[1])), np.zeros(X1_L.shape[1])])
        X1R_noise = np.vstack([np.random.normal(size=(X1_L.shape[1])), np.random.normal(size=(X1_L.shape[1])), np.zeros(X1_L.shape[1])])
        X2R_noise = np.vstack([np.random.normal(size=(X1_L.shape[1])), np.random.normal(size=(X1_L.shape[1])), np.zeros(X1_L.shape[1])])
        X1_L += sigma*X1L_noise
        X2_L += sigma*X2L_noise
        X1_R += sigma*X1R_noise
        X2_R += sigma*X2R_noise
    return X[:, mask], X1_L, X2_L, X1_R, X2_R, R, t

def stereo_triangulate(kpts1, kpts2, fx, fy, baseline):
    xl = kpts1[0]
    yl = kpts1[1]
    xr = kpts2[0]
    Z = fx*baseline/(xl-xr)
    X = xl*Z/fx
    Y = yl*Z/fy
    return np.array([X, Y, Z])

def reprojection(pos, K, R, t, baseline=0):
    P = K@np.hstack([R, t.reshape((3, 1))])
    ret = P@(pos.reshape(4, 1)-np.vstack([R.T@np.array([[baseline], [0], [0]]), 0]))
    return ret/ret[2]

def getT(kpts1, kpts2, kpts3, fx, fy, baseline, R, T):
    pos = stereo_triangulate(kpts1, kpts2, fx, fy, baseline)
    pos = R@pos
    s = (pos[2]*kpts3[0]-pos[0]*fx)/(T[0]*fx-T[2]*kpts3[0])
    return s*T

if __name__=="__main__":
    K = np.eye(3)
    baseline=0.2
    X, X1_L, X2_L, X1_R, X2_R, R, t = generate_pts(K, K, baseline, add_noise=True)
    print(R)
    print(t)
    R_est, T_est = estimate_motion(X1_L, X2_L, K, K)
    T_est = getT(X1_L[:2, 0], X1_R[:2, 0], X2_L[:2, 0], K[0][0], K[1][1], 0.2, R_est, T_est)
    print(R_est)
    print(T_est)