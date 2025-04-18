

# This file was *autogenerated* from the file test.sage
from sage.all_cmdline import *   # import sage library

_sage_const_2 = Integer(2); _sage_const_1 = Integer(1); _sage_const_0 = Integer(0); _sage_const_5 = Integer(5); _sage_const_11 = Integer(11); _sage_const_32 = Integer(32)
def inv(a, p, r, N):
    P = a.parent()
    b = a.change_ring(Zmod(_sage_const_2 )).inverse_mod(x**N - _sage_const_1 )
    q = p
    while q < p ** r:
        q = q*q
        b = b.change_ring(Zmod(q))

        b = (b*(_sage_const_2  - a.change_ring(Zmod(q)) * b)) % (x**N - _sage_const_1 ).change_ring(Zmod(q))

      

    
    return b.change_ring(Zmod(p**r)) 

P = PolynomialRing(ZZ, names=('x',)); (x,) = P._first_ngens(1)
f = P([-_sage_const_1 , _sage_const_1 , _sage_const_1 , _sage_const_0 , _sage_const_1 , _sage_const_0 , -_sage_const_1 , _sage_const_0 , _sage_const_0 , _sage_const_1 , -_sage_const_1 ])


# Compute modular inverse of f in (Z/2^5 Z)[X]/(X^11 - 1)
f_inv = inv(f, _sage_const_2 , _sage_const_5 , _sage_const_11 )

# Define the modulus (X^11 - 1) in Zmod(32)
m = (x**_sage_const_11  - _sage_const_1 ).change_ring(Zmod(_sage_const_32 ))

# Multiply f_inv by f and reduce modulo m
result = (f_inv * f.change_ring(Zmod(_sage_const_32 ))) % m
print(f_inv)
# Print the result
print(result)

