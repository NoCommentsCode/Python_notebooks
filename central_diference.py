import sympy as sp
import numpy as np

class CentralDiference:
    def __init__(self, consts, order : int):
        self.order = order
        self.consts = consts
        """
        consts: [A1, A2, ... An] n = k / 2
        U' = An(U[j + n] - U[j - n]) + An-1(U[j + n - 1] - U[j - n + 1]) + ... + A1(U[j + 1] - U[j - 1]))
        """
    def get_cd_as_function(self):
        k = self.order
        n = k // 2
        def cd_function(U, h : float):
            U_n = np.zeros_like(U)
            for i in range(1, n + 1):
                if i == n:
                    U_n[n:-n] += (U[2 * n:] - U[:-2 * n]) / h * float(self.consts[n - 1])
                else:
                    U_n[n:-n] += (U[n + i:-n + i] - U[n - i:-n - i]) / h * float(self.consts[i - 1])
            return U_n
        return cd_function
    def print_cd(self):
        cd = '{'
        for i in range(1, self.order // 2):
            cd += str(np.abs(self.consts[i - 1])) + '(U[j + ' + str(i) + '] - U[j - ' + str(i) + ']) ' + ('+ ' if self.consts[i] > 0 else '- ')
        k = self.order // 2
        cd += str(np.abs(self.consts[k - 1])) + '(U[j + ' + str(k) + '] - U[j - ' + str(k) + '])} / h'
        print(cd)

def generate_cd(k : int):
    n = k // 2
    A = sp.zeros(n, n)
    for i in range(n):
        for j in range(n):
            A[j, i] = (n - i) ** (2 * j + 1)
    x = sp.zeros(n, 1)
    x[0, 0] = sp.Rational(1, 2)
    consts = A.inv() * x
    return CentralDiference(consts[::-1], k)
"""
cd_2 = generate_cd(2).get_cd_as_function()
cd_4 = generate_cd(4).get_cd_as_function()
cd_6 = generate_cd(6).get_cd_as_function()
cd_8 = generate_cd(8).get_cd_as_function()
cd_10 = generate_cd(10).get_cd_as_function()
cd_12 = generate_cd(12).get_cd_as_function()
cd_14 = generate_cd(14).get_cd_as_function()
cd_16 = generate_cd(16).get_cd_as_function()
"""