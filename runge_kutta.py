import numpy as np

def Butcher_table(order):
    if order == 2:
        c = np.array([0, 1])
        b = np.array([1/2, 1/2])
        A = np.array([[0, 0],
                      [1, 0]])
    elif order == 3:
        c = np.array([0, 1/3, 2/3])
        b = np.array([1/4, 0, 3/4])
        A = np.array([[0,   0,   0],
                      [1/3, 0,   0],
                      [0,   2/3, 0]])
        
    elif order == 4:
        c = np.array([0, 1/2, 1/2, 1])
        b = np.array([1/6, 1/3, 1/3, 1/6])
        A = np.array([[0,   0,   0, 0],
                      [1/2, 0,   0, 0],
                      [0,   1/2, 0, 0],
                      [0,   0,   1, 0]])
    elif order == 5:
        c = np.array([0, 1/4, 1/4, 1/2, 3/4, 1])
        b = np.array([7/90, 0, 16/45, 2/15, 16/45, 7/90])
        A = np.array([[0,     0,   0,    0,    0,   0],
                      [1/4,   0,   0,    0,    0,   0],
                      [1/8,   1/8, 0,    0,    0,   0],
                      [0,     0,   1/2,  0,    0,   0],
                      [3/16, -3/8, 3/8,  9/16, 0,   0],
                      [-3/7,  8/7, 6/7, -12/7, 8/7, 0]])
    elif order == 7:
        c = np.array([0, 1/6, 1/3, 1/2, 2/11, 2/3, 6/7, 0, 1])
        b = np.array([0, 0, 0, 32/105, 1771561/6289920, 243/2560, 16807/74880, 77/1440, 11/270])
        A = np.array([[0,          0,   0,          0,            0,            0,         0,         0,     0],
                      [ 1/6,       0,   0,          0,            0,            0,         0,         0,     0],
                      [ 0,         1/3, 0,          0,            0,            0,         0,         0,     0],
                      [ 1/8,       0,   3/8,        0,            0,            0,         0,         0,     0],
                      [ 148/1331,  0,   150/1331,  -56/1331,      0,            0,         0,         0,     0],
                      [-404/243,   0,  -170/27,     4024/1701,    10648/1701,   0,         0,         0,     0],
                      [ 2466/2401, 0,   1242/343,  -19176/16807, -51909/16807,  1053/2401, 0,         0,     0],
                      [ 5/154,     0,   0,          96/539,      -1815/20384,  -405/2464,  49/1144,   0,     0],
                      [-113/32,    0,  -195/22,     32/7,         29403/3584,  -729/512,   1029/1408, 21/16, 0]])
    elif order == 8:
        s21 = np.sqrt(21)
        c = np.array([1/2, 1/2, (7+s21)/14, (7+s21)/14, 1/2, (7-s21)/14, (7-s21)/14, 1/2, (7+s21)/14, 1])
        b = np.array([1/20, 0, 0, 0, 0, 0, 0, 49/180, 16/45, 49/180, 1/20])
        A = np.array([[0,           0,             0,                 0,                   0,                  0,                   0,                  0,               0,                0,             0],
                      [1/2,         0,             0,                 0,                   0,                  0,                   0,                  0,               0,                0,             0],
                      [1/4,         1/4,           0,                 0,                   0,                  0,                   0,                  0,               0,                0,             0],
                      [1/7,         (-7-3*s21)/98, (21+5*s21)/49,     0,                   0,                  0,                   0,                  0,               0,                0,             0],
                      [(11+s21)/84, 0,             (18+4*s21)/63,     (21-s21)/252,        0,                  0,                   0,                  0,               0,                0,             0],
                      [(5+s21)/48,  0,             (9+s21)/36,        (-231+14*s21)/360,   (63-7*s21)/80,      0,                   0,                  0,               0,                0,             0],
                      [(10-s21)/42, 0,             (-432+92*s21)/315, (633-145*s21)/90,    (-504+115*s21)/70,  (63-13*s21)/35,      0,                  0,               0,                0,             0],
                      [1/14,        0,             0,                 0,                   (14-3*s21)/126,     (13-3*s21)/63,       1/9,                0,               0,                0,             0],
                      [1/32,        0,             0,                 0,                   (91-21*s21)/576,    11/72,               (-385-75*s21)/1152, (63+13*s21)/128, 0,                0,             0],
                      [1/14,        0,             0,                 0,                   1/9,                (-733-147*s21)/2205, (515+111*s21)/504,  (-51-11*s21)/56, (132+28*s21)/245, 0,             0],
                      [0,           0,             0,                 0,                   (-42+7*s21)/18,     (-18+28*s21)/45,     (-273-53*s21)/72,   (301+53*s21)/72, (28-28*s21)/45,   (49-7*s21)/18, 0]])
    else:
        print('Unknown order! Used order 2')
        c = np.array([0, 1])
        b = np.array([1/2, 1/2])
        A = np.array([[0, 0],
                      [1, 0]])
    return {'a' : A, 'b' : b, 'c' : c}

def runge_kutta(k, U, dt, Deriv, args):
    table = Butcher_table(k)
    K = np.zeros((table['a'].shape[0], U.size))
    K[0, :] = Deriv(U, *args)
    for i in range(1, K.shape[0]):
        a = table['a'][i, :].copy()
        K[i, :] = Deriv(U + dt * np.sum(a.reshape((-1, 1)) * K, axis = 0), *args)
    return U + dt * np.sum(table['b'].reshape((-1, 1)) * K, axis = 0)

def runge_kutta_function(k):
    def rk(U, dt, Deriv, args):
        table = Butcher_table(k)
        K = np.zeros((table['a'].shape[0], U.size))
        K[0, :] = Deriv(U, *args)
        for i in range(1, K.shape[0]):
            a = table['a'][i, :].copy()
            K[i, :] = Deriv(U + dt * np.sum(a.reshape((-1, 1)) * K, axis = 0), *args)
        return U + dt * np.sum(table['b'].reshape((-1, 1)) * K, axis = 0)
    return rk

"""
def rk_8(U, h, r, Deriv):
    dt = r * h
    return runge_kutta(8, U, dt, Deriv, (h,)), dt
def rk_7(U, h, r, Deriv):
    dt = r * h
    return runge_kutta(7, U, dt, Deriv, (h,)), dt
def rk_5(U, h, r, Deriv):
    dt = r * h
    return runge_kutta(5, U, dt, Deriv, (h,)), dt
def rk_4(U, h, r, Deriv):
    dt = r * h
    return runge_kutta(4, U, dt, Deriv, (h,)), dt
def rk_3(U, h, r, Deriv):
    dt = r * h
    return runge_kutta(3, U, dt, Deriv, (h,)), dt
def rk_2(U, h, r, Deriv):
    dt = r * h
    return runge_kutta(2, U, dt, Deriv, (h,)), dt
"""