print('hello')
def Newthon(F, Xo, N):
    n = Xo.size
    X = np.zeros([N, n])
    X[0] = Xo
    def __deriv(F, X0, i, j, eps = 1e-5):
        dx = np.zeros(n)
        dx[j] += eps
        return (F[i](X0 + dx) - F[i](X0 - dx)) / (2. * eps)
    def __jacobian(F, X0):
        J = np.zeros([n, n])
        for i in range(0, n):
            for j in range(0, n):
                dF = __deriv(F, X0, i, j)
                J[i][j] = dF
        return J
    for i in range(1, N):
        J = np.linalg.det(__jacobian(F, X[i - 1]))
        X[i] = X[i - 1] - np.array([F[k](X[i - 1]) for k in range(n)]) / J
    return X