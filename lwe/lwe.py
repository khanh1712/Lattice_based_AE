import numpy as np
import random
import cmath
import hashlib
from Crypto.Cipher import AES

class ToyLWE:
    def __init__(self, n, q, p):
        self.n = n
        self.q = q
        self.p = p
        self.delta = round(q / p)
        
        self.S = np.array([random.randint(0, q-1) for _ in range(n)])
    
    def generate_matrix_from_seed(self, seed):
        # Use a more cryptographically secure method to generate matrix A from seed
        matrix = []
        hash_input = seed
        for i in range(self.n):
            # Hash the seed concatenated with index to generate each element
            hash_output = hashlib.sha256(hash_input).digest()
            # Convert hash to integer and reduce modulo q
            matrix.append(int.from_bytes(hash_output, 'big') % self.q)
            hash_input = hash_output
        return np.array(matrix)
    
    def encrypt_real(self, m_real):
        if abs(m_real) >= self.p/2:
            raise ValueError(f"Real part must have absolute value less than p/2={self.p/2}")
        
        m_shifted = (m_real + self.p//2) % self.p
        
        seed = random.randbytes(16)
        
        A = self.generate_matrix_from_seed(seed)
        
        e = random.randint(-self.delta//2, self.delta//2)
        
        dot_product = np.dot(A, self.S) % self.q
        
        b = (dot_product + self.delta * m_shifted + e) % self.q
        
        return seed, b
    
    def decrypt_real(self, seed, b):
        A = self.generate_matrix_from_seed(seed)
        
        dot_product = np.dot(A, self.S) % self.q
        x = (b - dot_product) % self.q
        
        if x > self.q // 2:
            x = x - self.q
            
        m_shifted = round(x / self.delta)
        
        m_real = (m_shifted - self.p//2) % self.p
        if m_real > self.p//2:
            m_real -= self.p
            
        return m_real
    
    def encrypt_complex(self, m_complex):
        m_real = int(round(m_complex.real))
        m_imag = int(round(m_complex.imag))
        
        return (self.encrypt_real(m_real), self.encrypt_real(m_imag))
    
    def decrypt_complex(self, real_ciphertext, imag_ciphertext):
        seed_real, b_real = real_ciphertext
        seed_imag, b_imag = imag_ciphertext
        
        real_part = self.decrypt_real(seed_real, b_real)
        imag_part = self.decrypt_real(seed_imag, b_imag)
        
        return complex(real_part, imag_part)

def encrypt_complex_array(lwe, complex_array):
    return [lwe.encrypt_complex(complex_num) for complex_num in complex_array]

def decrypt_complex_array(lwe, ciphertexts):
    return [lwe.decrypt_complex(real_cipher, imag_cipher) for real_cipher, imag_cipher in ciphertexts]

if __name__ == "__main__":
    n = 500
    q = 2**16
    p = 256
    
    lwe = ToyLWE(n, q, p)
    
    complex_array = [
        complex(42, 73),
        complex(-15, 30),
        complex(100, -50),
        complex(-120, -90)
    ]
    print(f"Original complex array: {complex_array}")
    
    encrypted = encrypt_complex_array(lwe, complex_array)
    print(f"Encrypted {len(encrypted)} complex numbers")
    
    seed_real, b_real = encrypted[0][0]
    seed_imag, b_imag = encrypted[0][1]
    print(f"Sample ciphertext for real part seed: {seed_real}")
    print(f"Sample ciphertext for real part b: {b_real}")
    print(f"Sample ciphertext for imaginary part seed: {seed_imag}")
    print(f"Sample ciphertext for imaginary part b: {b_imag}")
    
    decrypted = decrypt_complex_array(lwe, encrypted)
    print(f"Decrypted complex array: {decrypted}")