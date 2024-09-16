import numpy as np
import flint


def solve_system_mod(a, mod = 2) :
    n = len(a)
    m = len(a[0])

    A = flint.nmod_mat(n, m, a.flatten().tolist(), mod)
    A_ = A.rref()[0]
    M = np.array([np.array([int(num.str()) for num in a]) for a in A_.table()])
    M_ = M[:,:m-1]
    b_ = M[:,m-1]
    res = back_substitution_mod(M_,b_, mod = mod)

    return res


def back_substitution_mod(M, b, mod = 2) :

    n = len(M)
    m = len(M[0])
    
    pivots = np.full(n, -1)
    for i in range(n) :
        f = np.nonzero(M[i])[0]

        if len(f) != 0 :
          pivots[i] = f[0]

    c = b.copy()

    x = np.zeros(m, dtype=int)

    for row in range(n-1, -1, -1) :
        if pivots[row] == -1 :
            if c[row] != 0 :
                raise ValueError
            else :
                continue
        else :
            x[pivots[row]] = c[row]    
            c = (c - (M[:,pivots[row]] * c[row])) % mod

    return x