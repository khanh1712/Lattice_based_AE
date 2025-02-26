from poly import Poly
from Zmod import Zmod
from common import *
import time


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
        self.h = Poly(self.Rq, self.g).mul_mod_convolution_ring(self.Fq, self.N)

        return self.h
    
    def encrypt_poly(self, f_m: Poly) -> Poly:
        r = random_ternary_list(self.d, self.d, self.N - 1)

        e = (Poly(self.Rq, r) * self.Rp).mul_mod_convolution_ring( self.h, self.N) + f_m.change_ring(self.Rq)

        return e
    
    def decrypt_poly(self, e: Poly) -> Poly:
        temp = Poly(self.Rq, self.f).mul_mod_convolution_ring(e, self.N) 
        
        center = temp.center_lift()
        f_m = Poly(self.Rp, center).mul_mod_convolution_ring(self.Fp, self.N)

        return f_m

    def encrypt(self, m_list: np.ndarray[np.complex128]) -> np.ndarray[np.complex128]:
        poly_list = complex_array_2_polys(m_list, self.Rp)
        enc_poly = []
        
        for p in poly_list:
            enc_poly.append(self.encrypt_poly(p))
        
        return polys_2_complex_array(enc_poly)
    
    def decrypt(self, enc_list: np.ndarray[np.complex1280]) -> np.ndarray[np.complex128]:
        enc_poly = complex_array_2_polys(enc_list, self.Rp)
        
        m_list = []
        
        for p in enc_poly:
            m_list.append(self.decrypt_poly(p))
        
        return polys_2_complex_array(m_list)
         
# def test():
#     #Moderate security parameter
#     N = 107
#     p = 3
#     q = 64
#     df = 14
#     dg = 12
#     d = 5

#     m = b'Minh ngu vai loz'
#     t = time.time()
#     ntru = NTRU(N, p, q, df, dg, d)
#     ntru.key_gen()
#     e = ntru.encrypt(m)
#     m_ = ntru.decrypt(e)
#     print(time.time() - t)
#     assert(m == m_)

# for i in range(100):
#     print(i)
#     test()

