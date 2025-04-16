import numpy as np
from sage.stats.distributions.discrete_gaussian_integer import DiscreteGaussianDistributionIntegerSampler

def random_matrix(n, m, sigma):
    D = DiscreteGaussianDistributionIntegerSampler(sigma)
    L = np.array([[np.int32(D()) for _ in range(m)] for __ in range(n)])
    return L

class LWE:
    def __init__(self, n1, n2, l, p, q, sigma):

        self.n1 = n1
        self.n2 = n2
        self.l = 2*l
        self.p = p
        self.q = q
        self.A = np.array([[np.random.randint(0, p) for _ in range(self.n2)] for __ in range(self.n1)])
        self.sigma = sigma
        self.R1 = None 
        self.R2 = None  
        self.P = None

    def encode(self, m):
        # Ensure proper type conversion
        return ((self.p // self.q) * ((np.array(m, dtype=np.int64)).astype(np.int64) % self.q))
    
    def decode(self, c):
        m = []
        interval = self.p // (self.q * 2)  
        range_ = self.p // self.q
        for c_ in c:
            is_zero = True
            delta = c_ % self.p  
            for scale in range(1, self.q):
                if scale*range_ - interval <= delta <= scale*range_ + interval:
                    if scale % 2 == 0:
                        m.append(scale - self.q)
                    else: m.append(scale)
                    is_zero = False
                    break
            if is_zero == True: 
                m.append(0)

        return np.array(m, dtype=np.int64)
    
    def gen_key(self):
        self.R1 = random_matrix(self.n1, self.l, self.sigma)
        self.R2 = random_matrix(self.n2, self.l, self.sigma)
        self.P = (self.R1 - np.matmul(self.A, self.R2) % self.p) % self.p
        return self.P
    
    def encrypt(self, m):
        m = np.array(m, dtype=np.int64).flatten()
        e1 = random_matrix(1, self.n1, self.sigma).astype(np.int64)
        e2 = random_matrix(1, self.n2, self.sigma).astype(np.int64)
        e3 = random_matrix(1, self.l, self.sigma).astype(np.int64)
        e3 = (e3 + self.encode(m)) % self.p
    
        c1 = (np.matmul(e1, self.A) % self.p + e2) % self.p
        c2 = (np.matmul(e1, self.P) % self.p + e3) % self.p
        return [c1, c2]
    
    def decrypt(self, c):
        c1, c2 = c
        c1 = np.array(c1, dtype=np.int64)
        c2 = np.array(c2, dtype=np.int64)
        
        m_with_error = (np.matmul(c1, self.R2) % self.p + c2) % self.p
        return self.decode(m_with_error.flatten())
    
    def encrypt_complex(self, complex_array):
     
        real_parts = np.real(complex_array).astype(np.int64)
        imag_parts = np.imag(complex_array).astype(np.int64)
        
        concatenated = np.concatenate([real_parts, imag_parts])
        c1, c2 = self.encrypt(concatenated)
        
        return c1[0] + 1j*c2[0]
    
    def decrypt_complex(self, c):
        real_parts = np.real(c).astype(np.int64)
        imag_parts = np.imag(c).astype(np.int64)
        decrypted = self.decrypt([real_parts, imag_parts])
        half_len = len(decrypted) // 2
        real_parts = decrypted[:half_len]
        imag_parts = decrypted[half_len:]
        return real_parts + 1j * imag_parts
