import numpy as np
from poly import Poly
from Crypto.Util.number import bytes_to_long, long_to_bytes
import random 
from Zmod import Zmod

def random_ternary_list(d1, d2, degree):
    if d1 + d2 > degree + 1:
        raise ValueError("d1 + d2 cannot exceed degree + 1 (polynomial length)")

    a = [0] * (degree + 1)

    # Choose unique indices for 1s and -1s
    ones_indices = random.sample(range(degree + 1), d1)
    remaining_indices = list(set(range(degree + 1)) - set(ones_indices))
    minus_ones_indices = random.sample(remaining_indices, d2)

    # Assign values to selected indices
    for i in ones_indices:
        a[i] = 1
    for i in minus_ones_indices:
        a[i] = -1

    return a


def random_poly_list(degree, bound):
    return np.random.randint(-bound, bound + 1, size=degree + 1).tolist()


def message_2_poly(message, n):
 
    temp = bytes_to_long(message)
    coeffs = []
    while temp > 0:
        coeffs.append(temp % n)
        temp //= n
    
    return Poly(n, coeffs)

def poly_2_message(poly):
    modulus = int(poly.n)
    temp = poly.coeffs
    message = 0

    for i in range(len(temp) - 1, -1, -1):
        message *= modulus
        message += int(temp[i])
    
    return long_to_bytes(int(message))

# df = 14
# N = 107
# Rp = 3
# Rq = 32

# f = Poly(Rq, [-1, 1, 1, 0, -1, 0, 1, 0, 0, 1, -1])
# f_inv = Poly(Rq, [5, 9, 6, 16, 4, 15, 16, 22, 20, 18, 30])

# #print(f.inv_mod_xN_prime_pow(11))
# # Fq = Poly(Rq, f).inv_mod_xN(N)

# df = 14

# r = random_ternary_list(df + 1, df, N - 1)
# print(r)
# print(Poly(Rq, r).inv_mod_xN_prime_pow(11))