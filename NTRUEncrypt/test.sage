def inv(a, p, r, N):
    P = a.parent()
    b = a.change_ring(Zmod(2)).inverse_mod(x^N - 1)
    q = p
    while q < p ^ r:
        q = q*q
        b = b.change_ring(Zmod(q))

        b = (b*(2 - a.change_ring(Zmod(q)) * b)) % (x^N - 1).change_ring(Zmod(q))

      

    
    return b.change_ring(Zmod(p**r)) 

P.<x> = PolynomialRing(ZZ)
f = P([-1, 1, 1, 0, 1, 0, -1, 0, 0, 1, -1])


# Compute modular inverse of f in (Z/2^5 Z)[X]/(X^11 - 1)
f_inv = inv(f, 2, 5, 11)

# Define the modulus (X^11 - 1) in Zmod(32)
m = (x^11 - 1).change_ring(Zmod(32))

# Multiply f_inv by f and reduce modulo m
result = (f_inv * f.change_ring(Zmod(32))) % m
print(f_inv)
# Print the result
print(result)