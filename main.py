import cmath
import math
import sympy
from sympy import *
from sympy.abc import a, b, x, t
from sympy.physics.quantum.dagger import Dagger
from itertools import combinations_with_replacement, permutations
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time
from numpy.polynomial.polynomial import polygrid2d
from matplotlib.ticker import LinearLocator
import os.path


def print_coeffs(poly): # Two-variate polynomial represented by a 2 dimensional list of coefficients is "flattened"
    coefflist = []
    for i in range(d_A+1):
        for j in range(d_B+1):
            coefflist.append(
                "complex(" + str(np.real(poly[i, j])) + "," + str(np.imag(poly[i, j])) + ")")
    return coefflist


for i in range(10000): # Search for a counterexample of degree d_A + d_B

    # Gradient descent

    step_over = false

    d_A = 2  # Degree in a and in b
    d_B = 2
    N = np.max((2*d_A+1, 2*d_B+1))

    gr = np.zeros((N, N, d_A+1, d_B+1), dtype=np.clongdouble)
    rng = np.random.default_rng()

    gradp = np.zeros((d_A+1, d_B+1), dtype=np.clongdouble)
    gradq = np.zeros((d_A+1, d_B+1), dtype=np.clongdouble)

    # Initialize coefficients with random values
    p = rng.random((d_A+1, d_B+1))+1j*rng.random((d_A+1, d_B+1))
    q = rng.random((d_A+1, d_B+1))-1j*rng.random((d_A+1, d_B+1))

    # Initialize the grid of roots of unity
    for u in range(N):
        for v in range(N):
            aa = np.exp(np.pi*1j*u/N)
            bb = np.exp(np.pi*1j*v/N)

            for k in range(d_A+1):
                for l in range(d_B+1):
                    gr[u, v, k, l] = aa**(-2*k)*bb**(-2*l)

    gr2 = np.reciprocal(gr)

    def F_grad():
        # Evaluate P and Q on the grid of roots of unity
        sp = np.tensordot(gr, p)
        sq = np.tensordot(gr, q)

        ff = 1.0-np.square(np.abs(sp))-np.square(np.abs(sq))  # 1-|P|^2-|Q|^2

        # Calculate gradient
        gp = np.tensordot(-4.0*ff*sp, gr2)/(N*N)
        gq = np.tensordot(-4.0*ff*sq, gr2)/(N*N)

        err = np.average(np.square(ff))  # Integral of (1-|P|^2-|Q|^2)^2

        return gp, gq, err

    err = 100
    rate = .2  # Learn rate
    obj = 1e-28  # Target value for F
    threshold = obj
    cnt = 1

    t1 = time.time()

    while err > obj:
        gradp, gradq, err = F_grad()  # Calculate gradient and integral

        p -= rate*gradp
        q -= rate*gradq

        # Modify leading coefficient and constant term
        if err > threshold:
            q[d_A, d_B] = -p[d_A, d_B] * \
                np.conjugate(p[0, 0])/np.conjugate(q[0, 0])
            q[d_A, 0] = -p[d_A, 0] * \
                np.conjugate(p[0, d_B])/np.conjugate(q[0, d_B])
            r = np.sqrt(np.square(np.linalg.norm(p)) +
                        np.square(np.linalg.norm(q)))
            p = p / r
            q = q / r

        if cnt % 5000 == 0:  # Log
            t2 = time.time()
            print('#', cnt, ': ', err)
            print('Time:', t2-t1, 'seconds')
            t1 = t2
        if cnt % 100000 == 0:
            print(",\n".join(print_coeffs(p)))
            print(",\n".join(print_coeffs(q)))
        if (cnt > 5000):
            step_over = true
            break
        cnt = cnt+1
    if (step_over):
        continue

    print(",\n".join(print_coeffs(p)))
    print(",\n".join(print_coeffs(q)))

    # Decomposition

    # Reduce the problem to the univariate case

    exponent_order = []
    for i in range(d_A, -d_A-1, -2):
        for j in range(d_B, -d_B-1, -2):
            exponent_order.append([i, j])

    precision = math.log10(obj)

    p_coeffs_list = p.reshape(1, (d_A+1)*(d_B+1)).tolist()[0]
    q_coeffs_list = q.reshape(1, (d_A+1)*(d_B+1)).tolist()[0]

    # Since SymPy cannot handle Laurent polynomials, from now on, we multiply each Laurent polynomial by a**d_A*b**d_B or t**(d_A+d_B) as appropriate in order to get a proper polynomial.

    def make_poly(coeffs):
        pol = Poly(0, [a, b])
        for i in range((d_A+1)*(d_B+1)):
            pol += Poly(coeffs[i]*a**exponent_order[i][0]*b **
                        exponent_order[i][1]*a**d_A*b**d_B, [a, b])
        return pol

    p_poly = Poly(make_poly(p_coeffs_list).subs(a, t).subs(b, t), t)
    p_conj_poly = Poly(make_poly([conjugate(x) for x in p_coeffs_list]).subs(
        a, 1/t).subs(b, 1/t)*t**(2*(d_A+d_B)), t)
    q_poly = Poly(make_poly(q_coeffs_list).subs(a, t).subs(b, t), t)
    q_conj_poly = Poly(make_poly([conjugate(x) for x in q_coeffs_list]).subs(
        a, 1/t).subs(b, 1/t)*t**(2*(d_A+d_B)), t)

    F = Matrix([
        [Poly(make_poly(p_coeffs_list)), Poly(make_poly(q_coeffs_list))],
        [-Poly(make_poly([conjugate(x) for x in q_coeffs_list]).subs(a, 1/a).subs(b, 1/b)*a**(2*d_A)*b**(2*d_B), [a, b]),
         Poly(make_poly([conjugate(x) for x in p_coeffs_list]).subs(a, 1/a).subs(b, 1/b)*a**(2*d_A)*b**(2*d_B), [a, b])]
    ])

    p_coeffs = p_poly.all_coeffs()  # Coefficients in descending order
    q_coeffs = q_poly.all_coeffs()
    p_conj_coeffs = p_conj_poly.all_coeffs()
    q_conj_coeffs = q_conj_poly.all_coeffs()

    def pad_with_zeros(coeffs):

        # Keep the length of the coefficient list constant even if there is no term of degree 2*(d_A+d_B) in the polynomial

        length = len(coeffs)
        if (length < 2*(d_A+d_B)+1):
            for _ in range((2*(d_A+d_B)+1 - length)):
                coeffs.insert(0, 0)
        return coeffs

    p_coeffs = pad_with_zeros(p_coeffs)
    q_coeffs = pad_with_zeros(q_coeffs)
    p_conj_coeffs = pad_with_zeros(p_conj_coeffs)
    q_conj_coeffs = pad_with_zeros(q_conj_coeffs)

    # Implement Haah decomposition
    # All variable names as in Haah

    C_matrices = []
    for i in range(d_A+d_B):
        C_matrices.append([])
        for j in range(d_A+d_B+1):
            C_matrices[i].append(Matrix([[0, 0], [0, 0]]))
    m = d_A+d_B
    for j in range(0, d_A+d_B+1):
        C_matrices[m-1][j] = Matrix([
            [p_coeffs[2*(d_A+d_B)-2*j], q_coeffs[2*(d_A+d_B)-2*j]
             ], [-q_conj_coeffs[2*(d_A+d_B)-2*j], p_conj_coeffs[2*(d_A+d_B)-2*j]]
        ])

    P_matrices = [0] * (d_A+d_B)

    def trace(matrix):
        sum = 0
        for i in range(shape(matrix)[0]):
            sum += matrix.row(i)[i]
        return sum

    for m in range(d_A+d_B, 0, -1):
        dagger = simplify(Dagger(Matrix(C_matrices[m-1][m])))
        P_m = simplify(dagger*Matrix(C_matrices[m-1][m]) / trace(dagger*Matrix(C_matrices[m-1][m]))) if trace(
            dagger*Matrix(C_matrices[m-1][m])) != 0 else Matrix([[0, 0], [0, 0]])
        P_matrices[m-1] = P_m
        if (m > 1):
            for j in range(0, m):
                C_matrices[m-2][j] = simplify(C_matrices[m-1]
                                              [j]*(eye(2)-P_m) + C_matrices[m-1][j+1]*P_m)

    E0 = simplify(C_matrices[0][0]*(eye(2)-P_matrices[0]
                                    ) + C_matrices[0][1]*P_matrices[0])

    def E_p(P):
        return t*P+1/t*(eye(2)-P)

    # Test all possible permutations of a's and b's

    perm = list(set([''.join(x)
                for x in permutations(['a'] * d_A + ['b'] * d_B)]))

    good_decomp = false

    for x in perm:
        product = E0
        for i in range(d_A+d_B):
            product = product * Matrix([[(E_p(P_matrices[i]).row(0)[0]*t), (E_p(P_matrices[i]).row(0)[1]*t)],
                                        [(E_p(P_matrices[i]).row(1)[0]*t), (E_p(P_matrices[i]).row(1)[1]*t)]]).subs(t, x[i])
        final_product = Matrix([[Poly(product.row(0)[0], [a, b]),
                                Poly(product.row(0)[1], [a, b])],
                               [Poly(product.row(1)[0], [a, b]),
                                Poly(product.row(1)[1], [a, b])]])
        diff = Matrix(
            [
                [Poly(final_product.row(0)[0]-F.row(0)[0], [a, b]),
                 Poly(final_product.row(0)[1]-F.row(0)[1], [a, b])],
                [Poly(final_product.row(1)[0]-F.row(1)[0], [a, b]),
                 Poly(final_product.row(1)[1]-F.row(1)[1], [a, b])]
            ]
        )

        if(max([abs(x) for x in Poly(diff.row(0)[0], [a, b]).coeffs()+Poly(diff.row(0)[1], [a, b]).coeffs()+Poly(diff.row(1)[0], [a, b]).coeffs()+Poly(diff.row(1)[1], [a, b]).coeffs()]) < 10**(precision/4+1)):
            print("The permutation", x, "yields a decomposition.\n")
            good_decomp = true
            break

    if (not good_decomp):
        print("No decomposition exists for the following p and q:")
        print("p:", p_poly/t**(d_A+d_B))
        print("q:", q_poly/t**(d_A+d_B), "\n")

    # Save result to the corresponding CSV file
    # f = open('a%db%d.csv' % (d_A, d_B), 'a')
    # writer = csv.writer(f)
    # writer.writerow([p, q, x if good_decomp else "none"])
    # f.close()
