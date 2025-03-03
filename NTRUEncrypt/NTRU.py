from poly import Poly
from Zmod import Zmod
from common import *
import time
import random
from tqdm import tqdm

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
        self.test = None

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
    
    def encrypt_block(self, f_m: Poly) -> Poly:
        # f_m = message_2_poly(m, self.Rp)
        r = random_ternary_list(self.d, self.d, self.N - 1)

        e = (Poly(self.Rq, r) * self.Rp).mul_mod_convolution_ring(self.h, self.N).add_inplace(f_m.change_ring(self.Rq))

        return e
    
    def decrypt_block(self, e: Poly) -> Poly:
        temp = Poly(self.Rq, self.f.copy()).mul_mod_convolution_ring(e, self.N) 
        
        center = temp.center_lift()
        f_m = Poly(self.Rp, center).mul_mod_convolution_ring(self.Fp, self.N)

        return f_m

    def encrypt(self, m_list: np.ndarray[np.complex128]) -> np.ndarray[np.complex128]:
        
        length = len(m_list)
        rounding = np.ceil(self.N / 2)
       
        pad = lambda x: np.pad(x, pad_width=(0, max(0,int(rounding -  len(x)))), mode='constant', constant_values=0)

        polys = complex_array_2_polys(m_list, self.Rp, self.N // 2)
        enc_polys = []
        for p in polys:
            tmp = self.encrypt_block(p)
            enc_polys.append(tmp)
        self.test = polys
        
        enc_polys = polys_2_complex_array(enc_polys,pad)
       
        return enc_polys


    def decrypt(self, enc_list: np.ndarray[np.complex128]) -> np.ndarray[np.complex128]:
        
        polys = complex_array_2_polys(enc_list, self.Rq, self.N // 2 + 1)


        #print(all(np.array_equal(p.coeffs, p_.coeffs) for p, p_ in zip(self.test, polys)))
        m_list = [self.decrypt_block(p) for p in polys]
        
        
        return polys_2_complex_array(m_list)

        
    
def test():

    #Those parameters are just only for testing purpose which are used to fit with quam encode -> need to check for security
    N = 107
    p = 7
    q = 256
    df = 14
    dg = 12
    d = 5

    rand = lambda : random.choice([1,2,3,4,5,6])
    m = np.array([rand() + 1j*rand() for i in range(350)])

   
    ntru = NTRU(N, p, q, df, dg, d)
    ntru.key_gen()
    e = ntru.encrypt(m)
    m_ = ntru.decrypt(e)
    # print('[+] Ciphertext:',e)
    # print('[+] Decrypted message:', m_ := ntru.decrypt(e))
  
    
    if all([a == b for a, b in zip(m,m_)]):
        pass
    else:
        
        print(m)
        print(m_)
        print('Wrong!')
        exit(0)

t = time.time()
for i in tqdm(range(1000)):
    test()
print(time.time() - t)

