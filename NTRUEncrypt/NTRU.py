from poly import Poly
from Zmod import Zmod
from common import *

class NTRU:
    def __init__(self, N, p, q, df, dg, d):
        self.N = N
        self.Rp = p
        self.Rq = q
        self.d = d
        self.df = df
        self.dg = dg
        self.h = None
        self.f = None #private key
        self.g = None #private key
        self.Fp = None
        self.Fq = None

    def key_gen(self):
        while True:
            try:
                self.f = random_ternary_list(self.df + 1, self.df, self.N - 1)
                self.Fp = Poly(self.Rp, self.f).inv_mod_xN_prime_pow(self.N)
                self.Fq = Poly(self.Rq, self.f).inv_mod_xN_prime_pow(self.N)
                
                # If either inversion fails (returns None), generate a new f and retry
                if self.Fp is not None and self.Fq is not None:
                    break

            except Exception as e:
                continue

        self.g = random_ternary_list(self.dg, self.dg, self.N - 1)
        self.h = (self.Fq * Poly(self.Rq, self.g)).divmod_convol_inplace(self.N) #Over Zq

        return self.h
    
    def encrypt(self, m):

        f_m = message_2_poly(m, self.Rp)
        r = random_ternary_list(self.d, self.d, self.N - 1)

        e = ((Poly(self.Rq, r) * self.Rp) * self.h + f_m.change_ring(self.Rq)).divmod_convol_inplace(self.N)

        return e
    
    def decrypt(self, e):
        temp = (Poly(self.Rq, self.f) * e).divmod_convol_inplace(self.N)
        
        center = temp.center_lift()
        f_m = (self.Fp * Poly(self.Rp, center)).divmod_convol_inplace(self.N)

        return poly_2_message(f_m)

def test():
    N = 107
    p = 3
    q = 64
    df = 14
    dg = 12
    d = 5

    m = b'Minh ngu vai loz'
    ntru = NTRU(N, p, q, df, dg, d)
    ntru.key_gen()
    e = ntru.encrypt(m)
    m_ = ntru.decrypt(e)

    assert(m == m_)

for i in range(1000):
    print(i)
    test()

