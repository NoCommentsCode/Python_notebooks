import numpy as np

def weno_5(U, h):
    N = U.size
    C = np.array([[1./3., 5./6., -1./6.],
                  [-1./6., 5./6., 1./3.],
                  [1./3., -7./6., 11./6.]])
    C_wave = C[::-1, ::-1]
    Ur = np.zeros((3, N))
    Ul = np.zeros((3, N))
    Beta = np.zeros((3, N))
    Beta[0, :-2] = 13./12. * (U[:-2] - 2. * U[1:-1] + U[2:]) ** 2 + 1./4. * (3. * U[:-2] - 4. * U[1:-1] + U[2:]) ** 2
    Beta[1, 1:-1] = 13./12. * (U[:-2] - 2. * U[1:-1] + U[2:]) ** 2 + 1./4. * (U[:-2] - U[2:]) ** 2
    Beta[2, 2:] = 13./12. * (U[:-2] - 2. * U[1:-1] + U[2:]) ** 2 + 1./4. * (U[:-2] - 4. * U[1:-1] + 3. * U[2:]) ** 2
    Beta_wave = Beta[::-1, ::-1]
    d = np.array([0.3, 0.6, 0.1])
    d_wave = d[::-1]
    eps = 1e-40
    alpha = d / (eps + Beta.T) ** 2
    alpha_wave = d_wave / (eps + Beta_wave.T) ** 2
    W = alpha.T / np.sum(alpha, axis = 1)
    W_wave = alpha_wave.T / np.sum(alpha_wave, axis = 1)
    for r in range(3):
        right = N - 2 + r
        Ur[r, r:right] = C[r, 0] * U[:-2] + C[r, 1] * U[1:-1] + C[r, 2] * U[2:]
        Ul[r, r:right] = C_wave[r, 0] * U[:-2] + C_wave[r, 1] * U[1:-1] + C_wave[r, 2] * U[2:]
    U_R = np.sum(W * Ur, axis = 0)
    U_L = np.sum(W_wave * Ul, axis = 0)
    U_big = np.zeros(len(U) + 1)
    U_big[0] = U_L[0]
    U_big[-1] = U_R[-1]
    U_big[1:-1] = 0.5 * (U_R[:-1] + U_L[1:] + (U_R[:-1] - U_L[1:]) * np.sign(U_R[:-1] + U_L[1:]))
    Flux = (U_big[1:] - U_big[:-1]) / h
    return Flux

def weno_m_5(U, h):
    N = U.size
    C = np.array([[1./3., 5./6., -1./6.],
                  [-1./6., 5./6., 1./3.],
                  [1./3., -7./6., 11./6.]])
    C_wave = C[::-1, ::-1]
    Ur = np.zeros((3, N))
    Ul = np.zeros((3, N))
    Beta = np.zeros((3, N))
    Beta[0, :-2] = 13./12. * (U[:-2] - 2. * U[1:-1] + U[2:]) ** 2 + 1./4. * (3. * U[:-2] - 4. * U[1:-1] + U[2:]) ** 2
    Beta[1, 1:-1] = 13./12. * (U[:-2] - 2. * U[1:-1] + U[2:]) ** 2 + 1./4. * (U[:-2] - U[2:]) ** 2
    Beta[2, 2:] = 13./12. * (U[:-2] - 2. * U[1:-1] + U[2:]) ** 2 + 1./4. * (U[:-2] - 4. * U[1:-1] + 3. * U[2:]) ** 2
    Beta_wave = Beta[::-1, ::-1]
    d = np.array([0.3, 0.6, 0.1])
    d_wave = d[::-1]
    eps = 1e-40
    alpha = d / (eps + Beta.T) ** 2
    alpha_wave = d_wave / (eps + Beta_wave.T) ** 2
    W = alpha.T / np.sum(alpha, axis = 1)
    W_wave = alpha_wave.T / np.sum(alpha_wave, axis = 1) # (3, 3001)
    alpha_m = np.zeros((N, 3))
    alpha_m_wave = np.zeros((N, 3))
    for k in range(3):
        alpha_m[:, k] = W[k, :] * (d[k] + d[k] ** 2 - 3 * d[k] * W[k, :] + W[k, :] ** 2) / (d[k] ** 2 + W[k, :] * (1 - 2 * d[k]))
        alpha_m_wave[:, k] = W_wave[k, :] * (d_wave[k] + d_wave[k] ** 2 - 3 * d_wave[k] * W_wave[k, :] + W_wave[k, :] ** 2) / (d_wave[k] ** 2 + W_wave[k, :] * (1 - 2 * d_wave[k]))
    W_m = alpha_m.T / np.sum(alpha_m, axis = 1) # (3001, 3)
    W_m_wave = alpha_m_wave.T / np.sum(alpha_m_wave, axis = 1)
    for r in range(3):
        right = N - 2 + r
        Ur[r, r:right] = C[r, 0] * U[:-2] + C[r, 1] * U[1:-1] + C[r, 2] * U[2:]
        Ul[r, r:right] = C_wave[r, 0] * U[:-2] + C_wave[r, 1] * U[1:-1] + C_wave[r, 2] * U[2:]
    U_R = np.sum(W_m * Ur, axis = 0)
    U_L = np.sum(W_m_wave * Ul, axis = 0)
    U_big = np.zeros(len(U) + 1)
    U_big[0] = U_L[0]
    U_big[-1] = U_R[-1]
    U_big[1:-1] = 0.5 * (U_R[:-1] + U_L[1:] + (U_R[:-1] - U_L[1:]) * np.sign(U_R[:-1] + U_L[1:]))
    Flux = (U_big[1:] - U_big[:-1]) / h
    return Flux
def weno_z_5(U, h):
    N = U.size
    C = np.array([[1./3., 5./6., -1./6.],
                  [-1./6., 5./6., 1./3.],
                  [1./3., -7./6., 11./6.]])
    C_wave = C[::-1, ::-1]
    Ur = np.zeros((3, N))
    Ul = np.zeros((3, N))
    Beta = np.zeros((3, N))
    Beta[0, :-2] = 13./12. * (U[:-2] - 2. * U[1:-1] + U[2:]) ** 2 + 1./4. * (3. * U[:-2] - 4. * U[1:-1] + U[2:]) ** 2
    Beta[1, 1:-1] = 13./12. * (U[:-2] - 2. * U[1:-1] + U[2:]) ** 2 + 1./4. * (U[:-2] - U[2:]) ** 2
    Beta[2, 2:] = 13./12. * (U[:-2] - 2. * U[1:-1] + U[2:]) ** 2 + 1./4. * (U[:-2] - 4. * U[1:-1] + 3. * U[2:]) ** 2
    Beta_wave = Beta[::-1, ::-1]
    eps = 1e-40
    tau_5 = np.abs(Beta[0, :] - Beta[2, :])
    Beta_z = np.zeros((3, N))
    Beta_z = 1. / (1. + tau_5 / (eps + Beta))
    Beta_z_wave = np.zeros((3, N))
    Beta_z_wave = 1. / (1. + tau_5 / (eps + Beta_wave))
    d = np.array([0.3, 0.6, 0.1])
    d_wave = d[::-1]
    alpha_z = d / (eps + Beta_z.T) ** 2
    alpha_z_wave = d_wave / (eps + Beta_z_wave.T) ** 2
    W_z = alpha_z.T / np.sum(alpha_z, axis = 1)
    W_z_wave = alpha_z_wave.T / np.sum(alpha_z_wave, axis = 1) # (3, 3001)
    for r in range(3):
        right = N - 2 + r
        Ur[r, r:right] = C[r, 0] * U[:-2] + C[r, 1] * U[1:-1] + C[r, 2] * U[2:]
        Ul[r, r:right] = C_wave[r, 0] * U[:-2] + C_wave[r, 1] * U[1:-1] + C_wave[r, 2] * U[2:]
    U_R = np.sum(W_z * Ur, axis = 0)
    U_L = np.sum(W_z_wave * Ul, axis = 0)
    U_big = np.zeros(len(U) + 1)
    U_big[0] = U_L[0]
    U_big[-1] = U_R[-1]
    U_big[1:-1] = 0.5 * (U_R[:-1] + U_L[1:] + (U_R[:-1] - U_L[1:]) * np.sign(U_R[:-1] + U_L[1:]))
    Flux = (U_big[1:] - U_big[:-1]) / h
    return Flux
def weno_zm_5(U, h):
    N = U.size
    C = np.array([[1./3., 5./6., -1./6.],
                  [-1./6., 5./6., 1./3.],
                  [1./3., -7./6., 11./6.]])
    C_wave = C[::-1, ::-1]
    Ur = np.zeros((3, N))
    Ul = np.zeros((3, N))
    Beta = np.zeros((3, N))
    Beta[0, :-2] = 13./12. * (U[:-2] - 2. * U[1:-1] + U[2:]) ** 2 + 1./4. * (3. * U[:-2] - 4. * U[1:-1] + U[2:]) ** 2
    Beta[1, 1:-1] = 13./12. * (U[:-2] - 2. * U[1:-1] + U[2:]) ** 2 + 1./4. * (U[:-2] - U[2:]) ** 2
    Beta[2, 2:] = 13./12. * (U[:-2] - 2. * U[1:-1] + U[2:]) ** 2 + 1./4. * (U[:-2] - 4. * U[1:-1] + 3. * U[2:]) ** 2
    Beta_wave = Beta[::-1, ::-1]
    eps = 1e-40
    tau_5 = np.abs(Beta[0, :] - Beta[2, :])
    Beta_z = np.zeros((3, N))
    Beta_z = 1. / (1. + tau_5 / (eps + Beta))
    Beta_z_wave = np.zeros((3, N))
    Beta_z_wave = 1. / (1. + tau_5 / (eps + Beta_wave))
    d = np.array([0.3, 0.6, 0.1])
    d_wave = d[::-1]
    alpha_z = d / (eps + Beta_z.T) ** 2
    alpha_z_wave = d_wave / (eps + Beta_z_wave.T) ** 2
    W_z = alpha_z.T / np.sum(alpha_z, axis = 1)
    W_z_wave = alpha_z_wave.T / np.sum(alpha_z_wave, axis = 1) # (3, 3001)
    alpha_zm = np.zeros((N, 3))
    alpha_zm_wave = np.zeros((N, 3))
    for k in range(3):
        alpha_zm[:, k] = W_z[k, :] * (d[k] + d[k] ** 2 - 3 * d[k] * W_z[k, :] + W_z[k, :] ** 2) / (d[k] ** 2 + W_z[k, :] * (1 - 2 * d[k]))
        alpha_zm_wave[:, k] = W_z_wave[k, :] * (d_wave[k] + d_wave[k] ** 2 - 3 * d_wave[k] * W_z_wave[k, :] + W_z_wave[k, :] ** 2) / (d_wave[k] ** 2 + W_z_wave[k, :] * (1 - 2 * d_wave[k]))
    W_zm = alpha_zm.T / np.sum(alpha_zm, axis = 1)
    W_zm_wave = alpha_zm_wave.T / np.sum(alpha_zm_wave, axis = 1) # (3, 3001)
    for r in range(3):
        right = N - 2 + r
        Ur[r, r:right] = C[r, 0] * U[:-2] + C[r, 1] * U[1:-1] + C[r, 2] * U[2:]
        Ul[r, r:right] = C_wave[r, 0] * U[:-2] + C_wave[r, 1] * U[1:-1] + C_wave[r, 2] * U[2:]
    U_R = np.sum(W_zm * Ur, axis = 0)
    U_L = np.sum(W_zm_wave * Ul, axis = 0)
    U_big = np.zeros(len(U) + 1)
    U_big[0] = U_L[0]
    U_big[-1] = U_R[-1]
    U_big[1:-1] = 0.5 * (U_R[:-1] + U_L[1:] + (U_R[:-1] - U_L[1:]) * np.sign(U_R[:-1] + U_L[1:]))
    Flux = (U_big[1:] - U_big[:-1]) / h
    return Flux